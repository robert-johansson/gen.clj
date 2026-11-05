(ns gen.llm.potential-test
  "Potentials (gen.llm.potential): the dense next-token weights and
   products under the potential contract's laws (gen.laws.potential,
   whose own tests hold their planted bugs), the end tokens a potential
   declares, misuse."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.branching-contract :as bc]
            [gen.laws.potential :as potential-laws]
            [gen.llm.potential :as potential]))

(def ^:private n 6)

(def ^:private no-repeats
  "Sequences whose tokens never repeat the one before: 0 or -Inf as a
   prefix, and complete only at even lengths."
  (potential/potential n {:prefix (fn [ctx] (if (some (fn [[a b]] (= a b)) (partition 2 1 ctx)) ##-Inf 0.0))
                          :complete (fn [ctx] (if (and (even? (count ctx)) (not-any? (fn [[a b]] (= a b)) (partition 2 1 ctx)))
                                                0.0 ##-Inf))}))

(def ^:private by-length
  "A soft potential: each token costs 0.1 per unit of its value."
  (potential/potential n {:prefix (fn [ctx] (* -0.1 (reduce + 0 ctx)))
                          :complete (fn [ctx] (- (* -0.1 (reduce + 0 ctx)) 1.0))}))

(potential-laws/deflaws dense [[:no-repeats no-repeats n] [:product (potential/product no-repeats by-length) n]] 100)

;; a soft potential has no zero prefixes, so no cases of that law
(potential-laws/deflaws soft [[:by-length by-length n]] 100
  (dissoc potential-laws/all :a-zero-prefix-has-zero-extensions))

(deftest dense-next-weights
  (let [w (potential/logw-next no-repeats [1 2])]
    (is (= ##-Inf (aget w 2)) "repeating 2")
    (is (= 0.0 (double (aget w 3))))
    (is (= 0.0 (double (aget w n))) "complete at length 2")))

(deftest a-product-sums-its-potentials
  (let [p (potential/product no-repeats by-length)]
    (is (= (+ (potential/prefix no-repeats [1 2]) (potential/prefix by-length [1 2])) (potential/prefix p [1 2])))
    (is (= ##-Inf (potential/prefix p [3 3])))
    (is (= -1.3 (potential/complete p [1 2])))))

;; A potential declares the end tokens it reads contexts with, or has no
;; opinion (DESIGN §9, "A potential declares its end tokens"): a product
;; carries its factors' set, and two factors that disagree are an error
;; when it is made; check-end-tokens is the one comparison with a
;; completion's.
(deftest potentials-declare-their-end-tokens
  (let [ended (potential/potential n {:prefix (constantly 0.0) :complete (constantly 0.0) :end-tokens #{5}})
        other (potential/potential n {:prefix (constantly 0.0) :complete (constantly 0.0) :end-tokens #{4 5}})]
    (is (nil? (potential/end-tokens by-length)) "no opinion")
    (is (= #{5} (potential/end-tokens ended)))
    (is (nil? (potential/end-tokens "not a potential")))
    (is (= #{5} (potential/end-tokens (potential/product by-length ended))) "a factor's set")
    (is (= #{5} (potential/end-tokens (potential/product ended (potential/product ended no-repeats)))))
    (is (nil? (potential/end-tokens (potential/product by-length no-repeats))))
    (is (= :mismatched-end-tokens (bc/reason #(potential/product ended by-length other))) "two factors disagree")
    (testing "the check"
      (is (= ended (potential/check-end-tokens "here" ended #{5})))
      (is (= by-length (potential/check-end-tokens "here" by-length #{4})) "no opinion passes")
      (is (= :mismatched-end-tokens (bc/reason #(potential/check-end-tokens "here" ended #{4}))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^here: " (potential/check-end-tokens "here" ended #{}))
          "naming its caller"))
    (is (= :invalid-end-tokens (bc/reason #(potential/potential n {:prefix identity :complete identity :end-tokens #{n}}))))
    (is (= :invalid-end-tokens (bc/reason #(potential/potential n {:prefix identity :complete identity :end-tokens [1]}))))))

(deftest misuse-is-loud
  (is (= :zero-prefix (bc/reason #(potential/logw-next no-repeats [3 3]))))
  (is (= :invalid-token (bc/reason #(potential/prefix no-repeats 3))))
  (is (= :invalid-potential (bc/reason #(potential/potential 0 {:prefix identity :complete identity}))))
  (is (= :invalid-potential (bc/reason #(potential/potential n {:prefix identity}))))
  (is (= :invalid-potential (bc/reason #(potential/product))))
  (is (= :invalid-potential
         (bc/reason #(potential/logw-next (potential/product by-length (potential/potential 3 {:prefix (constantly 0.0)
                                                                                             :complete (constantly 0.0)}))
                                          [1])))))
