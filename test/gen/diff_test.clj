(ns gen.diff-test
  "Change hints (gen.diff): the diff kinds, consistent? (does a diff describe
   a change), and propagate's rules, each checked to give a diff that
   describes the change it predicts, over random vectors, maps and edits."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as tgen]
            [clojure.test.check.properties :as prop]
            [gen.diff :as diff]))

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

;; a random vector and an edit of it: [prev new diff]
(def ^:private edited-vector
  (tgen/let [v (tgen/vector tgen/small-integer 1 6)
             kind (tgen/elements [:set :append :same])
             i (tgen/choose 0 (dec (count v)))
             x tgen/small-integer]
    (case kind
      :set [v (assoc v i x) (diff/vector-diff (count v) (count v) (if (= x (v i)) {} {i diff/unknown-change}))]
      :append [v (conj v x) (diff/vector-diff (inc (count v)) (count v) {})]
      :same [v v diff/no-change])))

(defspec propagated-diffs-describe-the-change 300
  ;; each rule, applied to a vector before and after a random edit, with the
  ;; edit's diff: the diff it gives describes the change of the result
  (prop/for-all [[v v' d] edited-vector
                 j (tgen/choose 0 5)
                 y tgen/small-integer]
    (let [j (mod j (count v))
          check (fn [f args args']
                  (diff/consistent? (apply @f args) (apply @f args')
                                    (diff/propagate f args' (into [d] (repeat (dec (count args)) diff/no-change)))))]
      (and (check #'clojure.core/count [v] [v'])
           (check #'clojure.core/nth [v j] [v' j])
           (check #'clojure.core/get [v j] [v' j])
           (check #'clojure.core/conj [v y] [v' y])
           (or (>= j (count v')) (check #'clojure.core/assoc [v j y] [v' j y]))
           (check #'clojure.core/vec [v] [v'])))))

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
