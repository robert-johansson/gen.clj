(ns gen.diff
  "Change hints for incremental computation, as in Gen.jl.

   An argdiff is the caller's assertion about how an argument changed (Gen.jl's
   trust model). update and regenerate take one per argument and return a
   retdiff describing how the return value changed, so combinators such as Map
   and Unfold can skip the elements that did not change. unknown-change is
   always sound; the others enable work to be skipped.")

(set! *warn-on-reflection* true)

(defrecord NoChange [])
(defrecord UnknownChange [])

(def no-change "The value did not change." (->NoChange))
(def unknown-change "The value may have changed arbitrarily." (->UnknownChange))

(defrecord VectorDiff [new-length prev-length updated])

(defn vector-diff
  "A vector went from prev-length to new-length elements. updated maps the
   index of each retained element that may have changed to its diff; elements
   not in updated, and below both lengths, are unchanged."
  [new-length prev-length updated]
  (->VectorDiff new-length prev-length updated))

(defn no-change? [d] (instance? NoChange d))
(defn unknown-change? [d] (instance? UnknownChange d))
(defn vector-diff? [d] (instance? VectorDiff d))

(defn no-changes
  "Argdiffs asserting that none of n arguments changed."
  [n]
  (vec (repeat n no-change)))

(defn changed?
  "Might a value with diff d have changed?"
  [d]
  (not (no-change? d)))

(defn element-changed?
  "Might element i of a vector with diff d have changed (or be new)?"
  [d i]
  (cond (no-change? d) false
        (vector-diff? d) (or (>= i (:prev-length d)) (contains? (:updated d) i))
        :else true))
