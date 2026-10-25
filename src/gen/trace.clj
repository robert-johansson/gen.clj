(ns gen.trace
  "Traces: immutable records of one execution of a generative function.

   A trace holds the generative function, its arguments, its random choices
   (a choice map), its return value and its score, log p(choices; args).
   Calling a trace on an address or path returns what is there, like
   Gen.jl's trace[addr]: the choice, (tr :slope) or (tr [:y 0]), or the
   return value of the call there; at the empty path, (tr []), the trace's
   own. (subtrace tr addr) is the call's trace (Gen.jl's
   get_call(tr, addr).subtrace). Both are loud where there is no such
   choice or call, and take a not-found value to give there instead, as
   gen.choicemap's lookups do: (tr :b nil).

   A keyword reads the record's fields, not the choices: (:choices tr),
   (:score tr) and the others are the fields, and (:slope tr) is nil
   whatever the trace holds at :slope. Read a choice by calling the trace,
   (tr :slope) (DESIGN.md §9: kept, documented, not loud).

   A trace also records its maker, the generative function that made it
   (maker). Giving a trace another :gen-fn, (assoc tr :gen-fn g), keeps the
   maker, so update and regenerate see that g did not make it
   (gen.gfi/revise), and project reads it by its maker.

   A trace of a generative function with encapsulated randomness (thesis
   §4.5) also holds that randomness, a key, under :gen.trace/omega, and its
   score is an estimate."
  {:api :stable}
  (:require [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]))

(set! *warn-on-reflection* true)

(declare subtrace trace?)

(def ^:private none (Object.))

(def ^:dynamic *body-call*
  "While a generative function's body runs, the function (f gf args) that
   makes a plain call (gf & args) in it (gen.dynamic binds it; see
   gen.gfi/untraced-call); nil outside a body."
  nil)

(defn- for-host
  "v as host code reads it. Outside a body, a 0-d boolean array is a host
   boolean, so (if (tr :a) …) branches on its value on every backend, not
   on an array object (always true); on a lazy backend the read evaluates
   it, at what is an inference-loop boundary. Inside a body, values stay
   arrays: gen's conditionals read them (array/truthy?). A batched value,
   one per particle, stays one too."
  [v]
  (if (and (nil? *body-call*) (array/backend-array? v) (not (batch/batched? v))
           (= :bool (array/dtype v)) (= [] (array/shape v)))
    (array/truthy? v)
    v))

(defrecord Trace [gen-fn args choices retval score]
  clojure.lang.IFn
  (invoke [this addr]
    (let [v (this addr none)]
      (if (identical? v none) (throw (cm/lookup-error "trace: no choice or call" choices (cm/path addr))) v)))
  (invoke [this addr not-found]
    (let [v (cm/get-choice choices addr none)]
      (if (identical? v none)
        (let [s (subtrace this addr none)] (if (identical? s none) not-found (for-host (:retval s))))
        (for-host v))))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(def ^:private fields #{:gen-fn :args :choices :retval :score})

(defn omega-at
  "The path, through tr's fields, to a trace in it that keeps encapsulated
   randomness (:gen.trace/omega): [] for tr itself, or a call's. nil when
   there is none. A trace's calls are not listed, so this looks through
   every field that a kind keeps them in, every map, vector and trace
   beside a trace's own data."
  [tr]
  (letfn [(within [x at]
            (cond (trace? x) (if (some? (::omega x))
                               at
                               (some (fn [[k v]] (when-not (fields k) (within v (conj at k)))) x))
                  (and (map? x) (not (record? x))) (some (fn [[k v]] (within v (conj at k))) x)
                  (vector? x) (some #(within (x %) (conj at %)) (range (count x)))
                  :else nil))]
    (within tr [])))

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
   get_call(tr, addr).subtrace), tr itself at the empty path. A path that
   goes on past a call continues in its trace. Loud where no call is, as
   get_call's KeyError; with not-found, it gives not-found there."
  ([tr addr]
   (let [s (subtrace tr addr none)]
     (if (identical? s none) (throw (cm/lookup-error "subtrace: no call" (:choices tr) (cm/path addr))) s)))
  ([tr addr not-found]
   (loop [tr tr
          p (cm/path addr)]
     (if (empty? p)
       tr
       (if-let [[sub more] (-call (maker tr) tr p)]
         (recur sub more)
         not-found)))))
