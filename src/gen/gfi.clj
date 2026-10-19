(ns gen.gfi
  "The Generative Function Interface, shaped like Gen.jl's.

   Every operation is a pure function of its arguments and an explicit PRNG
   key: the same key gives the same result. The public functions take the key
   as an optional last argument. Omitting it draws a key from the implicit key
   source, the single impure point in gen.clj: a root key folded with a call
   counter. The root seed is random for each JVM (so separate processes are
   independent); implicit-seed reads it back, and seed! fixes it, so any run can
   be reproduced.

     (simulate gf args)                          → trace
     (generate gf args constraints)              → {:trace :weight}
     (update trace constraints)                  → {:trace :weight :retdiff :discard}
     (update trace args argdiffs constraints)
     (regenerate trace selection)                → {:trace :weight :retdiff}
     (regenerate trace args argdiffs selection)
     (assess gf args choices)                    → {:weight :retval}
     (propose gf args)                           → {:choices :weight :retval}
     (project trace selection)                   → weight

   update and regenerate without arguments keep the trace's own arguments,
   with no-change argdiffs; with arguments they take one argdiff per
   argument. Weights follow Gen.jl and the Cusumano-Towner thesis
   (docs/research/genjl-parity.md §3).

   A trace is called on an address or path for its choice there, like
   Gen.jl's trace[addr]: (tr :slope), (tr [:y 0]); :choices, :retval, :score
   and :args are its fields."
  (:refer-clojure :exclude [update])
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defprotocol IGenerativeFunction
  (-simulate [gf args key])
  (-generate [gf args constraints key])
  (-update [gf trace args argdiffs constraints key])
  (-regenerate [gf trace args argdiffs selection key])
  (-assess [gf args choices])
  (-propose [gf args key])
  (-project [gf trace selection]))

(defn generative-function?
  "Does x implement the GFI? Inline implementations answer by an instance
   check; satisfies? (slow when the answer is no) only covers extensions."
  [x]
  (or (instance? gen.gfi.IGenerativeFunction x)
      (and (some? x) (not (keyword? x)) (not (vector? x)) (not (number? x))
           (satisfies? IGenerativeFunction x))))

(defn run-op
  "Run a GFI op on gf from a map of its inputs (:key :constraints :prev
   :argdiffs :selection). With a non-empty :batch-shape the op runs batched:
   its scalar code under the batch tracer (gen.array.batch), which gives
   every generative function a batched form with the same meaning. Returns
   the op's result map ({:trace} for simulate). The one dispatch point for
   generative functions that call others (splices, combinators)."
  [gf op args {:keys [key constraints prev argdiffs selection batch-shape] :as opts}]
  (if (seq batch-shape)
    (batch/batched batch-shape (run-op gf op args (dissoc opts :batch-shape)))
    (case op
      :simulate {:trace (-simulate gf args key)}
      :generate (-generate gf args constraints key)
      :assess (-assess gf args constraints)
      :update (-update gf prev args argdiffs constraints key)
      :regenerate (-regenerate gf prev args argdiffs selection key))))

;; ---------------------------------------------------------------------------
;; The implicit key source
;; ---------------------------------------------------------------------------

(defonce ^:private implicit
  (atom (let [seed (.nextLong (java.security.SecureRandom.))]
          {:seed seed :key (rng/key seed)})))

(defonce ^:private counter (java.util.concurrent.atomic.AtomicLong.))

(defn implicit-seed
  "The seed of the implicit key source: random for each JVM unless seed! set
   it. Record it to reproduce a run."
  []
  (:seed @implicit))

(defn seed!
  "Reset the implicit key source to seed: later calls without a key repeat
   exactly."
  [seed]
  (reset! implicit {:seed (long seed) :key (rng/key seed)})
  (.set ^java.util.concurrent.atomic.AtomicLong counter 0)
  nil)

(defn next-key
  "A fresh key from the implicit key source (deterministic given the seed and
   the order of calls)."
  []
  (rng/fold-in (:key @implicit) (.getAndIncrement ^java.util.concurrent.atomic.AtomicLong counter)))

;; ---------------------------------------------------------------------------
;; Operations
;; ---------------------------------------------------------------------------

(defn check-args
  "The arguments as a vector: they must be a sequential collection (or nil,
   no arguments)."
  [op args]
  (if (or (nil? args) (sequential? args))
    (vec args)
    (throw (ex-info (str (name op) ": the arguments must be a vector of arguments, got " (pr-str args))
                    {:op op :args args}))))

(defn check-choice-map
  "x, a choice map (nil is the empty one). A plain Clojure map is an error:
   it would be read as no choices at all."
  [op what x]
  (cond (nil? x) cm/EMPTY
        (cm/choicemap? x) x
        :else (throw (ex-info (str (name op) ": " what " must be a choice map, not a " (.getName (class x))
                                   (when (map? x) ": use (gen.choicemap/from-map m)"))
                              {:op op :value-type (class x)}))))

(defn check-selection
  "x, a selection."
  [op x]
  (if (sel/selection? x)
    x
    (throw (ex-info (str (name op) ": the selection must be a selection, such as (gen.selection/select :a), not "
                         (pr-str x))
                    {:op op :value x}))))

(defn eval-trace!
  "Force the evaluation of tr's arrays (choices, return value and score) on
   a lazy backend, and return tr. Inference loops call it once per
   iteration (docs/array-api.md §9), so that a lazy backend's graph stays
   one iteration deep. With also, a tree of other arrays (a weight, say) is
   evaluated with it. A no-op on eager backends."
  ([tr] (eval-trace! tr nil))
  ([tr also]
   ;; one evaluation for the trace and also (e.g. a weight)
   (array/eval! [(:choices tr) (:retval tr) (:score tr) also])
   tr))

(defn simulate
  "Sample a trace of gf on args."
  ([gf args] (simulate gf args (next-key)))
  ([gf args key] (-simulate gf (check-args :simulate args) key)))

(defn generate
  "A trace of gf on args consistent with constraints (a choice map), and the
   log weight log p(τ; x) - log q(υ; x, σ), where υ are the choices the
   internal proposal sampled and σ the constraints."
  ([gf args constraints] (generate gf args constraints (next-key)))
  ([gf args constraints key]
   (-generate gf (check-args :generate args) (check-choice-map :generate "the constraints" constraints) key)))

(defn check-argdiffs
  "One argdiff per argument: a missing one would silently mean no-change."
  [op args argdiffs]
  (when-not (= (count args) (count argdiffs))
    (throw (ex-info (str (name op) ": " (count args) " arguments but " (count argdiffs) " argdiffs")
                    {:args (count args) :argdiffs (count argdiffs)}))))

(defn update
  "The trace with new arguments and constrained choices. Returns the new trace,
   the log weight, the retdiff, and the discarded choices (overwritten or no
   longer visited), which make the update reversible."
  ([trace constraints] (update trace constraints (next-key)))
  ([trace constraints key]
   (let [args (:args trace)]
     (update trace args (diff/no-changes (count args)) constraints key)))
  ([trace args argdiffs constraints] (update trace args argdiffs constraints (next-key)))
  ([trace args argdiffs constraints key]
   (check-argdiffs :update (check-args :update args) argdiffs)
   (-update (:gen-fn trace) trace (vec args) (vec argdiffs) (check-choice-map :update "the constraints" constraints)
            key)))

(defn regenerate
  "The trace with new arguments and the selected choices resampled from the
   internal proposal. Returns the new trace, the log weight and the retdiff."
  ([trace selection] (regenerate trace selection (next-key)))
  ([trace selection key]
   (let [args (:args trace)]
     (regenerate trace args (diff/no-changes (count args)) selection key)))
  ([trace args argdiffs selection] (regenerate trace args argdiffs selection (next-key)))
  ([trace args argdiffs selection key]
   (check-argdiffs :regenerate (check-args :regenerate args) argdiffs)
   (-regenerate (:gen-fn trace) trace (vec args) (vec argdiffs) (check-selection :regenerate selection) key)))

(defn assess
  "The log density of fully specified choices, and the return value."
  [gf args choices]
  (-assess gf (check-args :assess args) (check-choice-map :assess "the choices" choices)))

(defn propose
  "Sample choices from gf, with their log density and the return value."
  ([gf args] (propose gf args (next-key)))
  ([gf args key] (-propose gf (check-args :propose args) key)))

(defn project
  "The log density of the selected choices of a trace."
  [trace selection]
  (-project (:gen-fn trace) trace (check-selection :project selection)))
