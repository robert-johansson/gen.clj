(ns gen.llm.potential-test
  "Potentials (gen.llm.potential) and the contract check
   (gen.llm.potential-contract) other potentials use: the dense next-token
   weights, products, the end tokens a potential declares, misuse; and the
   check's power, against planted inconsistencies."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.branching-contract :as bc]
            [gen.llm.potential :as potential]
            [gen.llm.potential-contract :as contract]))

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

(def ^:private contexts [[] [1] [1 2] [3 3] [0 5 2]])

(deftest dense-next-weights-are-consistent
  (doseq [p [no-repeats by-length (potential/product no-repeats by-length)]]
    (is (= [] (contract/check-consistent p n contexts 1e-6))))
  (let [w (potential/logw-next no-repeats [1 2])]
    (is (= ##-Inf (aget w 2)) "repeating 2")
    (is (= 0.0 (double (aget w 3))))
    (is (= 0.0 (double (aget w n))) "complete at length 2")))

(deftest a-product-sums-its-potentials
  (let [p (potential/product no-repeats by-length)]
    (is (= (+ (potential/prefix no-repeats [1 2]) (potential/prefix by-length [1 2])) (potential/prefix p [1 2])))
    (is (= ##-Inf (potential/prefix p [3 3])))
    (is (= -1.3 (potential/complete p [1 2])))))

(deftest the-check-catches-inconsistent-potentials
  ;; planted: a logw-next off by 0.5 at one token, at the end, or short
  (doseq [[what bad] {"one token" (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w 3 (float (+ 0.5 (aget w 3)))) w))
                      "the end" (fn [ctx] (let [^floats w (potential/logw-next by-length ctx)] (aset w n (float 0.0)) w))
                      "too short" (fn [ctx] (java.util.Arrays/copyOf ^floats (potential/logw-next by-length ctx) (int n)))}]
    (let [p (potential/potential n {:prefix #(potential/prefix by-length %) :complete #(potential/complete by-length %)
                                    :logw-next bad})]
      (is (seq (contract/check-consistent p n contexts 1e-6)) what))))

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
