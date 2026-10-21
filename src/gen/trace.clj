(ns gen.trace
  "Traces: immutable records of one execution of a generative function.

   A trace holds the generative function, its arguments, its random choices
   (a choice map), its return value and its score, log p(choices; args).
   Calling a trace on an address or path returns what is there, like
   Gen.jl's trace[addr]: the choice, (tr :slope) or (tr [:y 0]), or the
   return value of the call there; at the empty path, (tr []), the trace's
   own. (subtrace tr addr) is the call's trace (Gen.jl's
   get_call(tr, addr).subtrace).

   A trace also records its maker, the generative function that made it
   (maker). Giving a trace another :gen-fn, (assoc tr :gen-fn g), keeps the
   maker, so update and regenerate see that g did not make it
   (gen.gfi/revise).

   A trace of a generative function with encapsulated randomness (thesis
   §4.5) also holds that randomness, a key, under :gen.trace/omega, and its
   score is an estimate."
  (:require [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(declare subtrace)

(defrecord Trace [gen-fn args choices retval score]
  clojure.lang.IFn
  (invoke [this addr]
    (let [v (cm/get-choice choices addr)]
      (if (some? v) v (:retval (subtrace this addr)))))
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

;; ---------------------------------------------------------------------------
;; Calls
;; ---------------------------------------------------------------------------

(defprotocol ICalls
  "How a generative function's traces hold the calls it made, each the
   trace of a callee at an address. A generative function that makes none
   (a distribution) has the default, extended to Object."
  (-call [gf tr path]
    "[subtrace more] for the call in tr, a trace gf made, whose address is a
     prefix of path (a non-empty vector), more the rest of the path; nil
     when there is none."))

(extend-protocol ICalls
  Object
  (-call [_ _ _] nil))

(defn subtrace
  "The trace of the call at an address or path in tr (Gen.jl's
   get_call(tr, addr).subtrace), tr itself at the empty path; nil where no
   call is. A path that goes on past a call continues in its trace."
  [tr addr]
  (loop [tr tr
         p (if (vector? addr) addr [addr])]
    (if (empty? p)
      tr
      (when-let [[sub more] (-call (maker tr) tr p)]
        (recur sub more)))))
