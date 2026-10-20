(ns gen.diff
  "Change hints for incremental computation, as in Gen.jl.

   An argdiff is the caller's assertion about how an argument changed (Gen.jl's
   trust model). update and regenerate take one per argument and return a
   retdiff describing how the return value changed, so combinators such as Map
   and Unfold can skip the elements that did not change. unknown-change is
   always sound; the others enable work to be skipped.

   Diffs: no-change, unknown-change, vector-diff (new and previous lengths,
   the changed retained elements), int-diff (a number changed by an
   amount), set-diff (elements added and deleted) and dict-diff (entries
   added, keys deleted, keys whose values changed), Gen.jl's.

   (propagate f args diffs) is the diff of (apply f args) from its
   arguments' diffs, for f a var (Gen.jl's Diffed propagation): the static
   language (gen ^:static) applies it to a binding that is one call, so
   that a precise diff reaches a callee (a Map then reruns only the
   elements that changed). It has rules for count, nth, get, conj, assoc,
   +, -, inc, dec and vec, and is open: (defmethod propagate #'my-fn [_
   args diffs] …). (consistent? prev new d) says whether d describes the
   change from prev to new."
  (:require [gen.array :as array]))

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

(defrecord IntDiff [amount])

(defn int-diff
  "A number changed by amount (Gen.jl's IntDiff); no-change for 0."
  [amount]
  (if (zero? amount) (->NoChange) (->IntDiff amount)))

(defrecord SetDiff [added deleted])

(defn set-diff
  "A set gained the elements of added and lost those of deleted (Gen.jl's
   SetDiff); no-change when both are empty."
  [added deleted]
  (if (and (empty? added) (empty? deleted)) (->NoChange) (->SetDiff (set added) (set deleted))))

(defrecord DictDiff [added deleted updated])

(defn dict-diff
  "A map gained the entries of added (key to value), lost the keys in
   deleted, and the values of the keys in updated (key to diff) may have
   changed; every other key kept its value (Gen.jl's DictDiff)."
  [added deleted updated]
  (if (and (empty? added) (empty? deleted) (empty? updated))
    (->NoChange)
    (->DictDiff (into {} added) (set deleted) (into {} updated))))

(defn no-change? [d] (instance? NoChange d))
(defn unknown-change? [d] (instance? UnknownChange d))
(defn vector-diff? [d] (instance? VectorDiff d))
(defn int-diff? [d] (instance? IntDiff d))
(defn set-diff? [d] (instance? SetDiff d))
(defn dict-diff? [d] (instance? DictDiff d))

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

(defn consistent?
  "Does diff d describe the change from prev to new: equal values for
   no-change, the lengths and the unchanged elements of a vector-diff, the
   amount of an int-diff, the elements of a set-diff, the entries of a
   dict-diff? (unknown-change describes any change.)"
  [prev new d]
  (cond (no-change? d) (= prev new)
        (unknown-change? d) true
        (vector-diff? d) (and (= (:prev-length d) (count prev)) (= (:new-length d) (count new))
                              (every? #(or (contains? (:updated d) %) (= (nth prev %) (nth new %)))
                                      (range (min (count prev) (count new)))))
        (int-diff? d) (== (- (double new) (double prev)) (double (:amount d)))
        (set-diff? d) (= new (into (apply disj (set prev) (:deleted d)) (:added d)))
        (dict-diff? d) (and (= (set (keys new)) (into (apply disj (set (keys prev)) (:deleted d)) (keys (:added d))))
                            (every? (fn [[k v]] (= v (get new k))) (:added d))
                            (every? #(or (contains? (:updated d) %) (contains? (:added d) %) (= (get prev %) (get new %)))
                                    (filter #(contains? prev %) (keys new))))
        :else false))

;; ---------------------------------------------------------------------------
;; Propagation (Gen.jl's Diffed)
;; ---------------------------------------------------------------------------

(defmulti propagate
  "The diff of (apply f args) given args' diffs, f a var: no-change when no
   argument changed, unknown-change when no rule knows better."
  (fn [f _args _diffs] f))

(defmethod propagate :default [_ _ diffs]
  (if (every? no-change? diffs) no-change unknown-change))

(defn- unchanged-or [diffs f] (if (every? no-change? diffs) no-change (f)))

(defmethod propagate #'clojure.core/count [_ [coll] [d :as diffs]]
  (unchanged-or diffs #(cond (vector-diff? d) (int-diff (- (:new-length d) (:prev-length d)))
                             (set-diff? d) (int-diff (- (count (:added d)) (count (:deleted d))))
                             (dict-diff? d) (int-diff (- (count (:added d)) (count (:deleted d))))
                             :else unknown-change)))

(defn- element-diff
  "The diff of coll's element at key k (unchanged), coll's diff being d."
  [coll d k]
  (cond (vector-diff? d) (if (element-changed? d k) (get (:updated d) k unknown-change) no-change)
        (dict-diff? d) (cond (or (contains? (:added d) k) (contains? (:deleted d) k)) unknown-change
                             :else (get (:updated d) k no-change))
        (set-diff? d) (if (or (contains? (:added d) k) (contains? (:deleted d) k)) unknown-change no-change)
        :else unknown-change))

(defmethod propagate #'clojure.core/nth [_ [coll k] [d dk :as diffs]]
  (unchanged-or diffs #(if (no-change? dk) (element-diff coll d k) unknown-change)))

(defmethod propagate #'clojure.core/get [_ [coll k] [d dk & more :as diffs]]
  (unchanged-or diffs #(if (and (no-change? dk) (every? no-change? more)) (element-diff coll d k) unknown-change)))

(defmethod propagate #'clojure.core/conj [_ [coll x] [d dx :as diffs]]
  ;; a vector grows at its end: precise when its length did not change
  (unchanged-or diffs #(let [n (count coll)]
                         (cond (not (vector? coll)) unknown-change
                               (no-change? d) (vector-diff (inc n) n {})
                               (and (vector-diff? d) (= (:new-length d) (:prev-length d)))
                               (vector-diff (inc n) (inc n) (cond-> (:updated d) (changed? dx) (assoc n dx)))
                               :else unknown-change))))

(defmethod propagate #'clojure.core/assoc [_ [coll k x] [d dk dx :as diffs]]
  (unchanged-or diffs #(cond (not (and (no-change? d) (no-change? dk))) unknown-change
                             (vector? coll) (let [n (count coll)]
                                              (if (< k n) (vector-diff n n {k dx}) (vector-diff (inc n) n {})))
                             (map? coll) (if (contains? coll k) (dict-diff {} #{} {k dx}) (dict-diff {k x} #{} {}))
                             :else unknown-change)))

(defn- sum-of-amounts [diffs sign]
  (if (every? #(or (no-change? %) (int-diff? %)) diffs)
    (int-diff (reduce + (map-indexed (fn [i d] (* (if (and sign (pos? i)) -1 1) (if (int-diff? d) (:amount d) 0))) diffs)))
    unknown-change))

(doseq [v [#'clojure.core/+ #'array/+]]
  (defmethod propagate v [_ _ diffs] (unchanged-or diffs #(sum-of-amounts diffs false))))
(doseq [v [#'clojure.core/- #'array/-]]
  (defmethod propagate v [_ args diffs]
    (unchanged-or diffs #(if (= 1 (count args)) (let [[d] diffs] (if (int-diff? d) (int-diff (- (:amount d))) unknown-change))
                            (sum-of-amounts diffs true)))))
(doseq [v [#'clojure.core/inc #'clojure.core/dec #'clojure.core/vec #'clojure.core/identity]]
  (defmethod propagate v [_ _ [d]] d))
