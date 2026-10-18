(ns gen.gfi
  "The Generative Function Interface, shaped like Gen.jl's.

   Every operation is a pure function of its arguments and an explicit PRNG
   key: the same key gives the same result. The public functions take the key
   as an optional last argument. Omitting it draws a key from the implicit key
   source, the single impure point in gen.clj: a root key (seed 0 unless
   seed! is called) folded with a call counter.

     (simulate gf args)                          → trace
     (generate gf args constraints)              → {:trace :weight}
     (update trace constraints)                  → {:trace :weight :retdiff :discard}
     (update trace args argdiffs constraints)
     (regenerate trace selection)                → {:trace :weight :retdiff}
     (regenerate trace args argdiffs selection)
     (assess gf args choices)                    → {:weight :retval}
     (propose gf args)                           → {:choices :weight :retval}
     (project trace selection)                   → weight

   update and regenerate with the trace's own arguments assert that no
   argument changed. Weights follow Gen.jl and the Cusumano-Towner thesis
   (docs/research/genjl-parity.md §3)."
  (:refer-clojure :exclude [update])
  (:require [gen.diff :as diff]
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

(defn generative-function? [x] (satisfies? IGenerativeFunction x))

;; ---------------------------------------------------------------------------
;; The implicit key source
;; ---------------------------------------------------------------------------

(defonce ^:private root-key (atom (rng/key 0)))
(defonce ^:private counter (java.util.concurrent.atomic.AtomicLong.))

(defn seed!
  "Reset the implicit key source: later calls without a key draw from seed."
  [seed]
  (reset! root-key (rng/key seed))
  (.set ^java.util.concurrent.atomic.AtomicLong counter 0)
  nil)

(defn next-key
  "A fresh key from the implicit key source (deterministic given the seed and
   the order of calls)."
  []
  (rng/fold-in @root-key (.getAndIncrement ^java.util.concurrent.atomic.AtomicLong counter)))

;; ---------------------------------------------------------------------------
;; Operations
;; ---------------------------------------------------------------------------

(defn simulate
  "Sample a trace of gf on args."
  ([gf args] (simulate gf args (next-key)))
  ([gf args key] (-simulate gf (vec args) key)))

(defn generate
  "A trace of gf on args consistent with constraints (a choice map), and the
   log weight log p(τ; x) - log q(τ; x, constraints) of the internal proposal."
  ([gf args constraints] (generate gf args constraints (next-key)))
  ([gf args constraints key] (-generate gf (vec args) constraints key)))

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
