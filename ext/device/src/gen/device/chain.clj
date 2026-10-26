(ns gen.device.chain
  "Whole MCMC runs as one device program (bean genclj-a1hu, research
   prototype): a model's constrained generate and an S-step chain of a
   gradient kernel, staged (gen.device.stage) from gen.clj's own code, the
   handler's generate and gen.inference.mcmc's move, and looped by a scan.

   (hmc-run model args observations selection opts steps) is the staged
   program of a key, as gen.parity-bench's unit: the key split in two, a
   generate on the first, then kernel/run's steps on the second (its
   step keys split the same way), each step mcmc/hmc's move. Its outputs:
   {:values [steps d] (the selected values after each step, flat as
   cm/to-array), :accepted [steps], :start [d]}. The draws are the
   handler's: the same Threefry bits from the same keys; only rounding
   differs (float32 on the device)."
  {:api :internal}
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.batched]
            [gen.choicemap :as cm]
            [gen.array.protocols :as p]
            [gen.device.stage :as st]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace]))

(def ^:private hmc-move
  "mcmc's move as a function of arrays and keys (private there: the
   prototype reads it; a design would make it public)."
  @#'gen.inference.mcmc/hmc-move)

(def ^:private mala-move @#'gen.inference.mcmc/mala-move)

(defn- step-words
  "kernel/run's step keys, (rng/split key n), as one uint32 [n 2] array."
  [key n]
  (st/threefry-words (p/-key-words key) (array/arange 0 n 1 {:dtype :uint32})))

(defn- run-program
  "The staged program of a key: generate, then steps of (move trace key),
   a function of the trace whose selected values are set to v."
  [model args observations selection steps move]
  (st/stage
   (fn [key]
     (let [[k0 k1] (rng/split key 2)
           tr (:trace (gfi/generate model (st/constants args) (st/constants observations) k0))
           others (sel/get-selected (:choices tr) (sel/complement selection))
           ;; the selected values in mcmc's order (its flat-values')
           proto (grad/continuous-selection tr selection)
           v0 (cm/to-array proto)
           [_ [vs accepted]]
           (st/scan (fn [v w]
                      (let [t (assoc tr :choices (cm/merge others (cm/from-array proto v)))
                            {:keys [values accepted?]} (move t (st/->StagedKey w))
                            v' (cm/to-array values)]
                        [v' [v' accepted?]]))
                    v0 (step-words k1 steps) steps)]
       {:values vs :accepted accepted :start v0}))
   (st/key-spec)))

(defn hmc-run
  "The staged program of a key: generate, then steps of HMC (opts: :L,
   :eps)."
  [model args observations selection {:keys [L eps] :or {L 10 eps 0.1} :as opts} steps]
  (run-program model args observations selection steps
               (fn [t k] (hmc-move t selection (assoc opts :L L :eps eps) k))))

(defn mala-run
  "The staged program of a key: generate, then steps of MALA (tau)."
  [model args observations selection tau steps]
  (run-program model args observations selection steps
               (fn [t k] (mala-move t selection tau {} k))))

;; ---------------------------------------------------------------------------
;; Batched runs: the batch tracer over the staging backend, as gen.batched
;; runs the handler (one key; draws gain the particle axis)

(def ^:private batched-run
  "gen.batched's run of an op over n particles, guarded (private there)."
  @#'gen.batched/run)

(defn importance-run
  "The staged program of a key: n importance samples from the prior, as
   gen.batched/generate's run (importance-sampling with :batched? true):
   {:log-ml-estimate, :log-weights [n]}."
  [model args observations n]
  (st/stage
   (fn [key]
     (let [{w :weight} (batched-run model :generate (st/constants args) n
                                    {:key key :constraints (st/constants observations)})
           w (batch/unwrap w [n])]
       {:log-ml-estimate (array/- (array/logsumexp w) (Math/log n)) :log-weights w}))
   (st/key-spec)))

(defn- flat-trace
  "A trace of model on args whose choices are observations and the selected
   values v, flat in proto's order: what a move reads (its model,
   arguments and choices)."
  [model args observations proto v]
  (gen.trace/->Trace model args (cm/merge observations (cm/from-array proto v)) nil nil))

(defn mala-chains-run
  "The staged program of a key: n chains, a batched constrained generate
   from the prior, then steps of mala-particles (each step's move under
   the batch tracer on that step's key, as gen.batched/map-particles runs
   it): {:values [steps n d], :accepted, :diverged and :start-ok [steps
   n], :start [n d]}. proto is
   the selected values' structure, from any trace (their order mcmc's)."
  [model args observations selection proto tau n steps]
  (st/stage
   (fn [key]
     (let [[k0 k1] (rng/split key 2)
           args (st/constants args)
           observations (st/constants observations)
           {t :trace} (batched-run model :generate args n {:key k0 :constraints observations})
           v0 (batch/unwrap (cm/to-array (grad/continuous-selection t selection)) [n])
           [_ [vs accepted diverged start-ok]]
           (st/scan (fn [v w]
                      (let [k (st/->StagedKey w)
                            [v' acc div ok] ((array/vmap (fn [v] (let [{:keys [values accepted? diverged? start-ok]}
                                                                       (mala-move (flat-trace model args observations proto v)
                                                                                  selection tau {} k)]
                                                                   [(cm/to-array values) accepted? diverged? start-ok])))
                                             v)]
                        [v' [v' acc div ok]]))
                    v0 (step-words k1 steps) steps)]
       {:values vs :accepted accepted :diverged diverged :start-ok start-ok :start v0}))
   (st/key-spec)))

;; ---------------------------------------------------------------------------
;; Counted chains: each step's key derived in the loop from the counter, so
;; a program holds no [steps 2] array of keys (a thread's program, gen.device.cuda)

(defn key-at
  "(rng/split key n)'s key at the staged counter t (an int32 scalar, below
   n): Threefry of the counter [0 t], as fold-in's."
  [key t]
  (st/->StagedKey (array/reshape (st/threefry-words (p/-key-words key) (array/reshape (array/astype t :uint32) [1])) [2])))

(defn- counted-program
  "run-program's program, its steps' keys derived in the loop."
  [model args observations selection steps move]
  (st/stage
   (fn [key]
     (let [[k0 k1] (rng/split key 2)
           tr (:trace (gfi/generate model (st/constants args) (st/constants observations) k0))
           others (sel/get-selected (:choices tr) (sel/complement selection))
           proto (grad/continuous-selection tr selection)
           v0 (cm/to-array proto)
           [_ [vs accepted]]
           (st/scan (fn [v t]
                      (let [tr' (assoc tr :choices (cm/merge others (cm/from-array proto v)))
                            {:keys [values accepted?]} (move tr' (key-at k1 t))
                            v' (cm/to-array values)]
                        [v' [v' accepted?]]))
                    v0 nil steps)]
       {:values vs :accepted accepted :start v0}))
   (st/key-spec)))

(defn hmc-chain
  "hmc-run's program with its steps' keys derived in the loop (the same
   draws): one chain's, for a thread of gen.device.cuda."
  [model args observations selection {:keys [L eps] :or {L 10 eps 0.1} :as opts} steps]
  (counted-program model args observations selection steps
                   (fn [t k] (hmc-move t selection (assoc opts :L L :eps eps) k))))

(defn mala-chain
  "mala-run's program with its steps' keys derived in the loop."
  [model args observations selection tau steps]
  (counted-program model args observations selection steps
                   (fn [t k] (mala-move t selection tau {} k))))

;; ---------------------------------------------------------------------------
;; Data as inputs: a program of a key and the data, which a device holds once
;; (genclj-4zmv: ndreg's X is 10^7 floats, too many for a constant)

(defn- spec-of [x] [(array/shape x) (if (= :float (array/dtype-kind (array/dtype x))) :float32 (array/dtype x))])

(defn importance-program
  "importance-run's program with the model's array arguments and the
   observations' values as inputs, after the key: (program key args-leaves…
   obs-leaves…), args a vector whose leaves are arrays, observations a
   choice map. Returns {:program, :inputs (the host arrays to upload, in
   input order after the key)}."
  [model args observations n]
  (let [leaves (cm/leaves observations)
        paths (mapv first leaves)
        obs-values (mapv second leaves)]
    {:program (st/stage
               (fn [key args obs-vals]
                 (let [obs (reduce (fn [m [p v]] (cm/set-choice m p v)) cm/EMPTY (map vector paths obs-vals))
                       {w :weight} (batched-run model :generate args n {:key key :constraints obs})
                       w (batch/unwrap w [n])]
                   {:log-ml-estimate (array/- (array/logsumexp w) (Math/log n)) :log-weights w}))
               (st/key-spec)
               (array/tree-map spec-of args)
               (mapv spec-of obs-values))
     :inputs (into (vec (array/tree-leaves args)) obs-values)}))
