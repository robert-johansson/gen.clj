(ns gen.inference.kernel-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; Toy kernels on vectors that record which steps ran, with reversals that
;; record a primed step.
(defn- step [label] (fn [t _] {:trace (conj t label)}))
(defn- toy [label] (k/kernel (step label) (step (keyword (str (name label) "'")))))

(def a (toy :a))
(def b (toy :b))
(def c (toy :c))

(defn- run1 [kernel] (:trace (kernel [] (rng/key 0))))

(deftest reversals
  (is (= [:a] (run1 a)))
  (is (= [:a'] (run1 (k/reversal a))))
  (is (= [:a] (run1 (k/reversal (k/reversal a)))) "the reversal of the reversal is the kernel")
  (testing "the reversal of a chain is the reversed chain of reversals"
    (is (= [:a :b :c] (run1 (k/chain a b c))))
    (is (= [:c' :b' :a'] (run1 (k/reversal (k/chain a b c))))))
  (testing "cycles and repeats, including a cycle that stops mid-round"
    (is (= [:a :b :c :a] (run1 (k/cycle 4 [a b c]))))
    (is (= [:a' :c' :b' :a'] (run1 (k/reversal (k/cycle 4 [a b c])))))
    (is (= [:a :b] (run1 (k/cycle [a b]))))
    (is (= [:b' :b' :b'] (run1 (k/reversal (k/repeat 3 b))))))
  (testing "a mixture's reversal mixes the reversals with the same weights"
    (is (= [:b] (run1 (k/mix [a b] [0 1]))))
    (is (= [:b'] (run1 (k/reversal (k/mix [a b] [0 1]))))))
  (testing "a kernel without a declared reversal"
    (is (not (k/reversible? (step :x))))
    (is (not (k/reversible? (k/chain a (step :x)))))
    (is (= [:a :x] (run1 (k/chain a (step :x)))) "still runs")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no declared reversal" (k/reversal (k/chain a (step :x)))))))

(deftest every-step-gets-its-own-key
  (let [record-key (fn [t key] {:trace (conj t key)})
        keys (run1 (k/repeat 5 record-key))]
    (is (= 5 (count (set keys))))))

(deftest running-a-chain
  (let [count-up (fn [n _] {:trace (inc n)})]
    (is (= {:traces [5 8 11 14] :acceptance-rate nil}
           (k/run count-up 0 {:samples 4 :burn 2 :thin 3 :key (rng/key 1)})))
    (is (= {:traces [] :acceptance-rate nil} (k/run count-up 0 {:samples 0})))
    (is (thrown? clojure.lang.ExceptionInfo (k/run count-up 0 {:samples 3 :thin 0}))))
  (testing "combinators report their moves' accept decisions, and run counts every move"
    (let [accept (fn [t _] {:trace t :accepted? true})
          reject (fn [t _] {:trace t :accepted? false})]
      (is (= [true false false] (:accepted ((k/chain accept reject reject) 0 (rng/key 2)))))
      (is (= [true false true false] (:accepted ((k/cycle 4 [accept reject]) 0 (rng/key 2)))))
      (is (= [true false true false] (:accepted ((k/repeat 2 (k/chain accept reject)) 0 (rng/key 2)))) "nested")
      (is (nil? (:accepted ((k/chain accept (fn [t _] {:trace t})) 0 (rng/key 2)))) "unless a move reports none")
      (is (= 0.25 (:acceptance-rate (k/run (k/chain accept reject reject reject) 0 {:samples 5 :key (rng/key 3)})))))))

(def model
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (if a [0.6 0.3 0.1] [0.2 0.3 0.5])))]
      (trace :obs (d/bernoulli (nth [0.1 0.5 0.9] b))))))

(def observations (cm/choicemap :obs true))

(deftest an-mh-cycle-targets-the-posterior
  (let [start (:trace (gfi/generate model [] (cm/choicemap :a true :b 0 :obs true) (rng/key 0)))
        kernel (k/checked (k/cycle [(k/mh (sel/select :a)) (k/mh (sel/select :b))]) observations)
        ends (map #(peek (:traces (k/run kernel start {:samples 1 :burn 30 :key %})))
                  (rng/split (rng/key 2) 2000))
        p-a (/ (count (filter #(% :a) ends)) 2000.0)
        ;; exact: p(a | obs) by enumeration over b
        joint (fn [a] (reduce + (for [b [0 1 2]]
                                  (Math/exp (:weight (gfi/assess model [] (cm/choicemap :a a :b b :obs true)))))))
        exact (/ (joint true) (+ (joint true) (joint false)))]
    (is (< (Math/abs (- p-a exact)) (* 5 (Math/sqrt (/ (* exact (- 1 exact)) 2000)))))
    (is (k/reversible? kernel))
    (let [{:keys [acceptance-rate]} (k/run (k/mh (sel/select :b)) start {:samples 100 :key (rng/key 3)})]
      (is (< 0.0 acceptance-rate 1.0)))))

(deftest checked-kernels-catch-changed-observations
  (let [start (:trace (gfi/generate model [] (cm/choicemap :obs true) (rng/key 0)))
        bad (k/checked (k/mh (sel/select :obs)) observations)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed the observation"
                          (dotimes [i 20] (bad start (rng/key i)))))))

(deftest checked-kernels-catch-a-proposal-that-writes-an-observation
  ;; the custom-proposal requirement of Alg 5 (p. 92): never propose an observed address
  (let [start (:trace (gfi/generate model [] (cm/choicemap :obs true) (rng/key 0)))
        writes-obs (gen [tr] (trace :obs (d/bernoulli 0.5)))
        bad (k/checked (k/mh writes-obs []) observations)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed the observation"
                          (dotimes [i 20] (bad start (rng/key i)))))))

(deftest a-mixture-of-kernels-targets-the-posterior
  (let [start (:trace (gfi/generate model [] (cm/choicemap :a true :b 0 :obs true) (rng/key 0)))
        kernel (k/mix [(k/mh (sel/select :a)) (k/mh (sel/select :b)) (k/mh (sel/select :a :b))] [0.3 0.3 0.4])
        ends (map #(peek (:traces (k/run kernel start {:samples 1 :burn 60 :key %}))) (rng/split (rng/key 5) 2000))
        p-a (/ (count (filter #(% :a) ends)) 2000.0)
        joint (fn [a] (reduce + (for [b [0 1 2]]
                                  (Math/exp (:weight (gfi/assess model [] (cm/choicemap :a a :b b :obs true)))))))
        exact (/ (joint true) (+ (joint true) (joint false)))]
    (is (< (Math/abs (- p-a exact)) (* 5 (Math/sqrt (/ (* exact (- 1 exact)) 2000)))))))

(deftest mixture-weights-are-validated
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-negative" (k/mix [a b] [-1 1]))))
