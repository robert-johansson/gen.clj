(ns gen.trace
  "Traces: immutable records of one execution of a generative function.

   A trace holds the generative function, its arguments, its random choices
   (a choice map), its return value and its score, log p(choices; args).
   Calling a trace on an address or path returns the choice there, like Gen.jl's
   trace[addr]: (tr :slope), (tr [:y 0]).

   A trace also records its maker, the generative function that made it
   (maker). Giving a trace another :gen-fn, (assoc tr :gen-fn g), keeps the
   maker, so update and regenerate see that g did not make it
   (gen.gfi/revise).

   A trace of a generative function with encapsulated randomness (thesis
   §4.5) also holds that randomness, a key, under :gen.trace/omega, and its
   score is an estimate."
  (:require [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(defrecord Trace [gen-fn args choices retval score]
  clojure.lang.IFn
  (invoke [_ addr] (cm/get-choice choices addr))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn make-trace
  "A trace from a map with :gen-fn :args :choices :retval :score, made by
   its :gen-fn."
  [m]
  (map->Trace (assoc m ::maker (:gen-fn m))))

(defn trace? [x] (instance? Trace x))

(defn maker
  "The generative function that made tr."
  [tr]
  (get tr ::maker (:gen-fn tr)))

(defn as-made-by
  "tr as a trace gf made: a wrapper's trace of the model it wraps (an
   internal proposal's, say) is the wrapper's own."
  [tr gf]
  (assoc tr :gen-fn gf ::maker gf))
