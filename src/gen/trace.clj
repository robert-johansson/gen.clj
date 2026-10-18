(ns gen.trace
  "Traces: immutable records of one execution of a generative function.

   A trace holds the generative function, its arguments, its random choices
   (a choice map), its return value and its score, log p(choices; args).
   Calling a trace on an address or path returns the choice there, like Gen.jl's
   trace[addr]: (tr :slope), (tr [:y 0])."
  (:require [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(defrecord Trace [gen-fn args choices retval score]
  clojure.lang.IFn
  (invoke [_ addr] (cm/get-choice choices addr))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn make-trace
  "A trace from a map with :gen-fn :args :choices :retval :score."
  [m]
  (map->Trace m))

(defn trace? [x] (instance? Trace x))
