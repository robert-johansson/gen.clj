(ns gen.diff-test
  "Change hints (gen.diff): the diff kinds, consistent? (does a diff describe
   a change), compute, and propagate's rules, each checked to give a diff
   that describes the change it predicts, over random calls with every
   argument edited or not."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as tgen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.diff :as diff]
            [gen.test-support :as ts]))

(deftest the-diff-kinds
  (testing "empty changes are no-change"
    (is (diff/no-change? (diff/int-diff 0)))
    (is (diff/no-change? (diff/set-diff #{} #{})))
    (is (diff/no-change? (diff/dict-diff {} #{} {}))))
  (testing "consistent?"
    (is (diff/consistent? [1 2 3] [1 9 3] (diff/vector-diff 3 3 {1 diff/unknown-change})))
    (is (not (diff/consistent? [1 2 3] [1 9 3] (diff/vector-diff 3 3 {0 diff/unknown-change}))))
    (is (diff/consistent? [1 2] [1 2 5] (diff/vector-diff 3 2 {})))
    (is (diff/consistent? 4 7 (diff/int-diff 3)))
    (is (not (diff/consistent? 4 7 (diff/int-diff 2))))
    (is (diff/consistent? #{1 2} #{2 3} (diff/set-diff #{3} #{1})))
    (is (diff/consistent? {:a 1 :b 2} {:a 1 :b 5 :c 0} (diff/dict-diff {:c 0} #{} {:b diff/unknown-change})))
    (is (not (diff/consistent? {:a 1 :b 2} {:a 7 :b 2} (diff/dict-diff {} #{} {:b diff/unknown-change}))))
    (is (diff/consistent? [1] [2] diff/unknown-change))
    (is (not (diff/consistent? [1] [2] diff/no-change)))))

;; Random calls of every rule's function, each argument edited or not (a
;; changed element, an appended or dropped one, a new or removed key, a
;; fresh value), the diffs from diff/compute: the rule's diff must describe
;; the change of the result. (The first version edited only the collection
;; and never shrank one: the review of 2026-10-20 found five unsound rules
;; it passed.)

(def ^:private small tgen/small-integer)
(def ^:private a-key (tgen/one-of [(tgen/choose 0 4) (tgen/elements [:a :b :c]) (tgen/elements [0.0 1.0 1.5])]))
(def ^:private a-coll
  (tgen/one-of [(tgen/vector small 0 4) (tgen/map (tgen/elements [:a :b :c 0 1]) small) (tgen/set small)]))

(defn- edited
  "A generator of the value after an edit of x (or none)."
  [x fresh]
  (let [structural (cond (and (vector? x) (seq x)) [(tgen/fmap #(assoc x (mod % (count x)) (inc (nth x (mod % (count x))))) small)
                                                    (tgen/return (pop x))
                                                    (tgen/fmap #(conj x %) small)]
                         (vector? x) [(tgen/fmap #(conj x %) small)]
                         (map? x) [(tgen/fmap (fn [[k v]] (assoc x k v)) (tgen/tuple (tgen/elements [:a :b :c 0 1]) small))
                                   (tgen/return (if (seq x) (dissoc x (first (keys x))) x))]
                         (set? x) [(tgen/fmap #(conj x %) small) (tgen/return (if (seq x) (disj x (first x)) x))]
                         :else [])]
    (tgen/one-of (concat [(tgen/return x) fresh] structural))))

(defn- call
  "A generator of [f prev-args new-args] for f with argument kinds."
  [f & kinds]
  (tgen/bind (apply tgen/tuple kinds)
             (fn [prev] (tgen/fmap (fn [new] [f prev new])
                                   (apply tgen/tuple (map edited prev kinds))))))

(def ^:private calls
  (tgen/one-of [(call #'clojure.core/count a-coll)
                (call #'clojure.core/nth (tgen/vector small 0 4) (tgen/one-of [(tgen/choose 0 4) (tgen/elements [0.0 1.0 1.7])]))
                (call #'clojure.core/get a-coll a-key)
                (call #'clojure.core/get a-coll a-key small)
                (call #'clojure.core/conj (tgen/vector small 0 4) small)
                (call #'clojure.core/conj (tgen/vector small 0 4) small small)
                (call #'clojure.core/assoc (tgen/one-of [(tgen/vector small 0 4) (tgen/map a-key small)]) a-key small)
                (call #'clojure.core/assoc (tgen/map a-key small) a-key small a-key small)
                (call #'clojure.core/vec a-coll)
                (call #'clojure.core/inc small)
                (call #'clojure.core/dec small)
                (call #'clojure.core/+ small small)
                (call #'clojure.core/- small small)
                (call #'clojure.core/- small)
                (call #'clojure.core/identity a-coll)]))

(defn- result [f args] (try (apply @f args) (catch Exception _ ::throws)))

(defspec every-rule-describes-the-change 2000
  (prop/for-all [[f prev new] calls]
    (let [r0 (result f prev) r1 (result f new)]
      (or (= ::throws r0) (= ::throws r1)
          (diff/consistent? r0 r1 (diff/propagate f new (mapv diff/compute prev new)))))))

(deftest the-review-s-counterexamples
  ;; each was unsound before the review of 2026-10-20
  (let [d #(diff/propagate %1 %2 %3) nc diff/no-change uc diff/unknown-change]
    (testing "conj with a changed element: the last element changed, the length did not"
      (is (= (diff/vector-diff 3 3 {2 uc}) (d #'clojure.core/conj [[1 2] 3] [nc uc]))))
    (testing "assoc appending, or of a new key: both results hold it"
      (is (= (diff/vector-diff 3 3 {2 uc}) (d #'clojure.core/assoc [[1 2] 2 9] [nc nc uc])))
      (is (= (diff/dict-diff {} #{} {:k uc}) (d #'clojure.core/assoc [{} :k 9] [nc nc uc]))))
    (testing "variadic calls"
      (is (diff/unknown-change? (d #'clojure.core/assoc [{} :p 1 :q 2] [nc nc nc nc uc])))
      (is (diff/unknown-change? (d #'clojure.core/conj [[] 1 2] [nc nc uc]))))
    (testing "get past a shrunk end, and a key that is not an index"
      (is (diff/unknown-change? (d #'clojure.core/get [[1 2] 2 0] [(diff/vector-diff 2 3 {}) nc nc])))
      (is (diff/no-change? (d #'clojure.core/get [[1 2] :k 0] [(diff/vector-diff 2 3 {}) nc nc]))))
    (testing "nth truncates a number, and the guarded lookups read an array key (confirmation review N1)"
      (is (diff/unknown-change? (d #'clojure.core/nth [[1 2] 1.0] [(diff/vector-diff 2 2 {1 uc}) nc])))
      (let [k (array/array [1])]
        (is (diff/unknown-change? (d #'clojure.core/nth [[1 2] k] [(diff/vector-diff 2 2 {1 uc}) nc])))
        (is (diff/unknown-change? (d #'clojure.core/get [{1 2} k] [(diff/dict-diff {} #{} {1 uc}) nc])))
        (is (diff/unknown-change? (d #'clojure.core/assoc [{} k 1] [nc nc uc])))))
    (testing "vec, inc and dec pass on only the diffs of their kind"
      (is (diff/unknown-change? (d #'clojure.core/vec [{:a 1}] [(diff/dict-diff {} #{} {:a uc})])))
      (is (diff/unknown-change? (d #'clojure.core/inc [1] [(diff/vector-diff 1 1 {})]))))
    (testing "consistent? checks element diffs, and set- and dict-diffs' added and deleted"
      (is (not (diff/consistent? [1 2] [1 5] (diff/vector-diff 2 2 {1 (diff/int-diff 1)}))))
      (is (not (diff/consistent? #{1 2} #{1 2} (diff/set-diff #{1} #{})))))))

(deftest unchanged-is-same-value
  ;; the post-M5 review's P1-01 and P1-03 (DESIGN.md §9, D1): unchanged is
  ;; gen.gfi/same-value?, by class and order and bounded, where
  ;; clojure.core/= equates values code tells apart and walks an infinite
  ;; lazy value forever. Planted bug (2026-10-22): = back in consistent? and
  ;; compute fails each pair's three assertions, and the lazy ones time out
  (let [within #(ts/within 5000 %)]
    (doseq [[a b] [[0.0 -0.0] [[1.0 2.0] '(1.0 2.0)]
                   [(sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0)] [1 (int 1)]]]
      (testing (str (pr-str a) " and " (pr-str b))
        (is (not (diff/consistent? a b diff/no-change)))
        (is (diff/changed? (diff/compute a b)))
        (is (diff/element-changed? (diff/compute [:k a] [:k b]) 1) "as an element")))
    (testing "the same values, of one class and order"
      (is (diff/consistent? [1.0 {:a -0.0}] [1.0 {:a -0.0}] diff/no-change))
      (is (diff/no-change? (diff/compute (sorted-map :b 1.0 :a 2.0) (sorted-map :a 2.0 :b 1.0)))))
    (testing "an infinite lazy value counts as changed, at once"
      (is (false? (within #(diff/consistent? (repeat 1.0) (repeat 1.0) diff/no-change))))
      (is (diff/unknown-change? (within #(diff/compute (iterate inc 0) (iterate inc 0))))))))

(defspec int-diffs-add-up 200
  (prop/for-all [a tgen/small-integer b tgen/small-integer da tgen/small-integer db tgen/small-integer]
    (and (diff/consistent? (+ a b) (+ (+ a da) (+ b db)) (diff/propagate #'clojure.core/+ [(+ a da) (+ b db)] [(diff/int-diff da) (diff/int-diff db)]))
         (diff/consistent? (- a b) (- (+ a da) (+ b db)) (diff/propagate #'clojure.core/- [(+ a da) (+ b db)] [(diff/int-diff da) (diff/int-diff db)]))
         (diff/consistent? (inc a) (inc (+ a da)) (diff/propagate #'clojure.core/inc [(+ a da)] [(diff/int-diff da)])))))

(defspec map-diffs-describe-the-change 200
  ;; assoc on a map, and get of a key, with the map's dict-diff
  (prop/for-all [m (tgen/map (tgen/elements [:a :b :c :d]) tgen/small-integer)
                 k (tgen/elements [:a :b :c :d])
                 x tgen/small-integer
                 j (tgen/elements [:a :b :c :d])]
    (let [m' (assoc m k x)
          d (cond (not (contains? m k)) (diff/dict-diff {k x} #{} {})
                  (= x (m k)) diff/no-change
                  :else (diff/dict-diff {} #{} {k diff/unknown-change}))]
      (and (diff/consistent? m m' d)
           (diff/consistent? (get m j) (get m' j) (diff/propagate #'clojure.core/get [m' j] [d diff/no-change]))
           (diff/consistent? (count m) (count m') (diff/propagate #'clojure.core/count [m'] [d]))
           (diff/consistent? (assoc m j 0) (assoc m' j 0) (diff/propagate #'clojure.core/assoc [m' j 0] [d diff/no-change diff/no-change]))))))

(defn- double-all [v] (mapv #(* 2 %) v))

(deftest a-new-rule
  ;; propagate is open: a rule for a function of one's own
  (defmethod diff/propagate #'double-all [_ _ [d]] d)
  (try
    (is (= (diff/vector-diff 3 3 {1 diff/unknown-change})
           (diff/propagate #'double-all [[1 2 3]] [(diff/vector-diff 3 3 {1 diff/unknown-change})])))
    (finally (remove-method diff/propagate #'double-all))))
