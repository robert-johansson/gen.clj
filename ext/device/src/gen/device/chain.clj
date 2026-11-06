(ns gen.device.chain
  "Whole MCMC runs as one device program (bean genclj-a1hu, research
   prototype): a model's constrained generate and an S-step chain of a
   gradient kernel, staged (gen.array.stage) from gen.clj's own code, the
   handler's generate and gen.inference.mcmc's move, and looped by a scan.
   The lowerings' tests and benches run these programs; kernel/run's
   :device stages its own (gen.inference.device).

   (mala-run model args observations selection tau steps) is the staged
   program of a key, as the parity bench's unit (gen.parity-models): the
   key split in two, a generate on the first, then kernel/run's steps on
   the second (its step keys split the same way), each step mcmc/mala's
   move. Its outputs: {:values [steps d] (the selected values after each
   step, flat as cm/to-array), :accepted [steps], :start [d]}. The draws
   are the handler's: the same Threefry bits from the same keys; only
   rounding differs (float32 on the device). mala-chain and hmc-chain are
   the same with each step's key derived in the loop; chain-program and
   weight-program take the model's data as inputs; weight-run is one
   particle's constrained generate."
  {:api :internal}
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.array.protocols :as p]
            [gen.array.stage :as st]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]))

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
           (array/scan (fn [v w]
                      (let [t (assoc tr :choices (cm/merge others (cm/from-array proto v)))
                            {:keys [values accepted?]} (move t (st/->StagedKey w))
                            v' (cm/to-array values)]
                        [v' [v' accepted?]]))
                    v0 (step-words k1 steps) steps)]
       {:values vs :accepted accepted :start v0}))
   (st/key-spec)))

(defn mala-run
  "The staged program of a key: generate, then steps of MALA (tau)."
  [model args observations selection tau steps]
  (run-program model args observations selection steps
               (fn [t k] (mcmc/mala-move t selection tau {} k))))

;; ---------------------------------------------------------------------------
;; A particle

(defn weight-run
  "One particle's constrained generate as a program of a key, the model's
   arguments and the observations staged as constants: {:weight}. n runs
   on (rng/split key n)[i] are scalar importance sampling's n particles
   (gen.inference.importance without :batched?)."
  [model args observations]
  (st/stage
   (fn [key] {:weight (:weight (gfi/generate model (st/constants args) (st/constants observations) key))})
   (st/key-spec)))

;; ---------------------------------------------------------------------------
;; Counted chains: each step's key derived in the loop from the counter, so
;; a program holds no [steps 2] array of keys (a thread's program, gen.device.cuda)

(defn key-at
  "(rng/split key n)'s key at the staged counter t (an int32 scalar, below
   n): Threefry of the counter [0 t], as fold-in's."
  [key t]
  (st/->StagedKey (array/reshape (st/threefry-words (p/-key-words key) (array/reshape (array/astype t :uint32) [1])) [2])))

(defn- counted-chain
  "A counted chain's body on a key, the model's arguments and observations
   (staged values or constants): generate, then steps of (move trace key)."
  [model key args observations selection steps move]
  (let [[k0 k1] (rng/split key 2)
        tr (:trace (gfi/generate model args observations k0))
        others (sel/get-selected (:choices tr) (sel/complement selection))
        proto (grad/continuous-selection tr selection)
        v0 (cm/to-array proto)
        [_ [vs accepted]]
        (array/scan (fn [v t]
                   (let [tr' (assoc tr :choices (cm/merge others (cm/from-array proto v)))
                         {:keys [values accepted?]} (move tr' (key-at k1 t))
                         v' (cm/to-array values)]
                     [v' [v' accepted?]]))
                 v0 nil steps)]
    {:values vs :accepted accepted :start v0}))

(defn- counted-program
  "run-program's program, its steps' keys derived in the loop."
  [model args observations selection steps move]
  (st/stage
   (fn [key] (counted-chain model key (st/constants args) (st/constants observations) selection steps move))
   (st/key-spec)))

(defn hmc-chain
  "The staged program of a key: generate, then steps of HMC (opts: :L,
   :eps), its steps' keys derived in the loop (the same draws as
   kernel/run's): one chain's, a thread's program."
  [model args observations selection {:keys [L eps] :or {L 10 eps 0.1} :as opts} steps]
  (counted-program model args observations selection steps
                   (fn [t k] (mcmc/hmc-move t selection (assoc opts :L L :eps eps) k))))

(defn mala-chain
  "mala-run's program with its steps' keys derived in the loop."
  [model args observations selection tau steps]
  (counted-program model args observations selection steps
                   (fn [t k] (mcmc/mala-move t selection tau {} k))))

;; ---------------------------------------------------------------------------
;; Data as inputs: a program of a key and the data, which a device holds once
;; (genclj-4zmv: ndreg's X is 10^7 floats, too many for a constant)

(defn- spec-of [x] [(array/shape x) (if (= :float (array/dtype-kind (array/dtype x))) st/*float* (array/dtype x))])

(defn- with-replay
  "{:program :inputs} whose program can be run on the handler from a key
   (:replay), its data the host arguments it was staged from."
  [{:keys [program] :as m} args obs-values]
  (assoc-in m [:program :replay] (fn [key] ((:f program) key args obs-values))))

(defn chain-program
  "A counted chain (mala-chain's or hmc-chain's) with the model's array
   arguments and the observations' values as inputs after the key, as
   weight-program's: {:program :inputs}. kernel: [:mala tau] or [:hmc
   opts]."
  [model args observations selection kernel steps]
  (let [leaves (cm/leaves observations)
        paths (mapv first leaves)
        obs-values (mapv second leaves)
        [kind k-opts] kernel
        move (case kind
               :mala (fn [t k] (mcmc/mala-move t selection k-opts {} k))
               :hmc (fn [t k] (mcmc/hmc-move t selection (merge {:L 10 :eps 0.1} k-opts) k)))]
    (with-replay
     {:program (st/stage
               (fn [key args obs-vals]
                 (let [obs (reduce (fn [m [p v]] (cm/set-choice m p v)) cm/EMPTY (map vector paths obs-vals))]
                   (counted-chain model key args obs selection steps move)))
               (st/key-spec)
               (array/tree-map spec-of args)
               (mapv spec-of obs-values))
     :inputs (into (vec (array/tree-leaves args)) obs-values)}
     args obs-values)))

(defn weight-program
  "One particle's constrained generate as a program of a key and the data:
   (program key args-leaves… obs-leaves…) → {:weight}, the model's
   arguments and the observations' values inputs: (program key
   args-leaves… obs-leaves…), args a vector whose leaves are arrays,
   observations a choice map. A thread's (or a lane group's) program: n
   particles are n of them, each on (rng/split key n)[i]. Returns
   {:program, :inputs (the host arrays to upload, in input order after the
   key)}."
  [model args observations]
  (let [leaves (cm/leaves observations)
        paths (mapv first leaves)
        obs-values (mapv second leaves)]
    (with-replay
     {:program (st/stage
               (fn [key args obs-vals]
                 (let [obs (reduce (fn [m [p v]] (cm/set-choice m p v)) cm/EMPTY (map vector paths obs-vals))]
                   {:weight (:weight (gfi/generate model args obs key))}))
               (st/key-spec)
               (array/tree-map spec-of args)
               (mapv spec-of obs-values))
     :inputs (into (vec (array/tree-leaves args)) obs-values)}
     args obs-values)))
