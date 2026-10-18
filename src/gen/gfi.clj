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
  (:require [gen.array.batch :as batch]
            [gen.diff :as diff]
            [gen.random :as rng]))

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

(defn simulate
  "Sample a trace of gf on args."
  ([gf args] (simulate gf args (next-key)))
  ([gf args key] (-simulate gf (vec args) key)))

(defn generate
  "A trace of gf on args consistent with constraints (a choice map), and the
   log weight log p(τ; x) - log q(υ; x, σ), where υ are the choices the
   internal proposal sampled and σ the constraints."
  ([gf args constraints] (generate gf args constraints (next-key)))
  ([gf args constraints key] (-generate gf (vec args) constraints key)))

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
   (check-argdiffs :update args argdiffs)
   (-update (:gen-fn trace) trace (vec args) (vec argdiffs) constraints key)))

(defn regenerate
  "The trace with new arguments and the selected choices resampled from the
   internal proposal. Returns the new trace, the log weight and the retdiff."
  ([trace selection] (regenerate trace selection (next-key)))
  ([trace selection key]
   (let [args (:args trace)]
     (regenerate trace args (diff/no-changes (count args)) selection key)))
  ([trace args argdiffs selection] (regenerate trace args argdiffs selection (next-key)))
  ([trace args argdiffs selection key]
   (check-argdiffs :regenerate args argdiffs)
   (-regenerate (:gen-fn trace) trace (vec args) (vec argdiffs) selection key)))

(defn assess
  "The log density of fully specified choices, and the return value."
  [gf args choices]
  (-assess gf (vec args) choices))

(defn propose
  "Sample choices from gf, with their log density and the return value."
  ([gf args] (propose gf args (next-key)))
  ([gf args key] (-propose gf (vec args) key)))

(defn project
  "The log density of the selected choices of a trace."
  [trace selection]
  (-project (:gen-fn trace) trace selection))
