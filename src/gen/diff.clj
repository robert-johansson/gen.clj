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
  "Might element i of a vector with diff d have changed (or be new, or be
   gone)?"
  [d i]
  (cond (no-change? d) false
        (vector-diff? d) (or (>= i (min (:prev-length d) (:new-length d))) (contains? (:updated d) i))
        :else true))

(defn consistent?
  "Does diff d describe the change from prev to new: equal values for
   no-change, the lengths and the unchanged elements of a vector-diff, the
   amount of an int-diff, the elements of a set-diff, the entries of a
   dict-diff, and each element diff of the elements it names? (unknown-change
   describes any change.)"
  [prev new d]
  (cond (no-change? d) (= prev new)
        (unknown-change? d) true
        (vector-diff? d) (and (= (:prev-length d) (count prev)) (= (:new-length d) (count new))
                              (every? #(if-let [e (get (:updated d) %)]
                                         (consistent? (nth prev %) (nth new %) e)
                                         (= (nth prev %) (nth new %)))
                                      (range (min (count prev) (count new)))))
        (int-diff? d) (== (- (double new) (double prev)) (double (:amount d)))
        ;; added elements (keys) were absent and deleted ones present, so
        ;; that count's rule can trust them
        (set-diff? d) (and (not-any? #(contains? (set prev) %) (:added d)) (every? #(contains? (set prev) %) (:deleted d))
                           (= new (into (apply disj (set prev) (:deleted d)) (:added d))))
        (dict-diff? d) (and (not-any? #(contains? prev %) (keys (:added d))) (every? #(contains? prev %) (:deleted d))
                            (= (set (keys new)) (into (apply disj (set (keys prev)) (:deleted d)) (keys (:added d))))
                            (every? (fn [[k v]] (= v (get new k))) (:added d))
                            (every? #(if-let [e (get (:updated d) %)]
                                       (consistent? (get prev %) (get new %) e)
                                       (= (get prev %) (get new %)))
                                    (filter #(contains? prev %) (keys new))))
        :else false))

(defn compute
  "The diff from prev to now, as precise as these kinds state it: no-change
   for equal values; for two vectors a vector-diff naming the elements that
   differ; for two maps a dict-diff, two sets a set-diff, two integers an
   int-diff; unknown-change otherwise (e.g. to build argdiffs from values)."
  [prev now]
  (cond (= prev now) no-change
        (and (vector? prev) (vector? now))
        (vector-diff (count now) (count prev)
                     (into {} (for [i (range (min (count prev) (count now))) :when (not= (prev i) (now i))]
                                [i (compute (prev i) (now i))])))
        (and (map? prev) (map? now) (not (record? prev)) (not (record? now)))
        (dict-diff (into {} (remove #(contains? prev (key %))) now)
                   (set (remove #(contains? now %) (keys prev)))
                   (into {} (for [[k v] now :when (and (contains? prev k) (not= v (prev k)))] [k (compute (prev k) v)])))
        (and (set? prev) (set? now)) (set-diff (remove prev now) (remove now prev))
        (and (integer? prev) (integer? now)) (int-diff (- now prev))
        :else unknown-change))

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

(defn- backend-key?
  "Is k a backend array? The guarded lookups read such a key on the host,
   which a rule cannot (and a fresh array is another map key each run)."
  [k]
  (and (array/array? k) (not (array/host? k))))

(defn- element-diff
  "The diff of coll's element at key k (unchanged), coll's diff being d: a
   vector-diff on a vector, a dict-diff on a map, a set-diff on a set;
   anything else, and a backend array as the key, unknown-change. On a
   vector a number that is not an integer may find an element too (nth
   truncates it); any other key finds nothing both times."
  [coll d k]
  (cond (backend-key? k) unknown-change
        (and (vector-diff? d) (vector? coll)) (cond (integer? k) (if (element-changed? d k)
                                                                   (get (:updated d) k unknown-change)
                                                                   no-change)
                                                    (number? k) unknown-change
                                                    :else no-change)
        (and (dict-diff? d) (map? coll)) (cond (or (contains? (:added d) k) (contains? (:deleted d) k)) unknown-change
                                               :else (get (:updated d) k no-change))
        (and (set-diff? d) (set? coll)) (if (or (contains? (:added d) k) (contains? (:deleted d) k)) unknown-change no-change)
        :else unknown-change))

;; Each rule gives the diff from the previous call's result to this one's.
;; A call of another arity than the rule's is unknown-change.

(defmethod propagate #'clojure.core/nth [_ [coll k :as args] [d dk :as diffs]]
  (unchanged-or diffs #(if (and (= 2 (count args)) (no-change? dk)) (element-diff coll d k) unknown-change)))

(defmethod propagate #'clojure.core/get [_ [coll k :as args] [d dk & more :as diffs]]
  (unchanged-or diffs #(if (and (<= 2 (count args) 3) (no-change? dk) (every? no-change? more))
                         (element-diff coll d k)
                         unknown-change)))

(defmethod propagate #'clojure.core/conj [_ [coll x :as args] [d dx :as diffs]]
  ;; on a vector whose length did not change, both results hold x last
  (unchanged-or diffs #(let [n (count coll)]
                         (cond (not (and (= 2 (count args)) (vector? coll))) unknown-change
                               (no-change? d) (vector-diff (inc n) (inc n) {n dx})
                               (and (vector-diff? d) (= (:new-length d) (:prev-length d)))
                               (vector-diff (inc n) (inc n) (cond-> (:updated d) (changed? dx) (assoc n dx)))
                               :else unknown-change))))

(defmethod propagate #'clojure.core/assoc [_ [coll k x :as args] [d dk dx :as diffs]]
  ;; with coll and k unchanged, both results hold k: only its value changed
  (unchanged-or diffs #(cond (not (and (= 3 (count args)) (no-change? d) (no-change? dk))) unknown-change
                             (backend-key? k) unknown-change
                             (vector? coll) (let [n (count coll)]
                                              (if (< k n) (vector-diff n n {k dx}) (vector-diff (inc n) (inc n) {n dx})))
                             (map? coll) (dict-diff {} #{} {k dx})
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
;; amounts are host numbers: lengths, or what a rule computed from them
(doseq [v [#'clojure.core/inc #'clojure.core/dec]]
  (defmethod propagate v [_ _ [d]] (if (or (no-change? d) (int-diff? d)) d unknown-change)))
(defmethod propagate #'clojure.core/vec [_ _ [d]] (if (or (no-change? d) (vector-diff? d)) d unknown-change))
(defmethod propagate #'clojure.core/identity [_ _ [d]] d)
