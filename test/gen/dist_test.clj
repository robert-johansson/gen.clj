(ns gen.dist-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private fixtures (edn/read-string (slurp "test/fixtures/distributions.edn")))

(defn- constructor [k] @(ns-resolve 'gen.dist (symbol (name k))))

(deftest log-probs-match-scipy
  (doseq [{:keys [dist params value log-prob]} fixtures]
    (let [got (dc/log-prob (apply (constructor dist) params) (if (vector? value) (a/array value) value))]
      (is (or (= log-prob (a/item got))
              (< (Math/abs (- (double (a/item got)) log-prob)) (* 1e-12 (max 1.0 (Math/abs log-prob)))))
          (str (name dist) " " params " at " value)))))

(deftest values-and-shapes
  (let [k (rng/key 3)]
    (testing "scalar draws are host values"
      (is (instance? Double (dc/sample (d/normal 0 1) k)))
      (is (instance? Boolean (dc/sample (d/bernoulli 0.5) k)) "bernoulli gives booleans, as Gen.jl")
      (is (instance? Long (dc/sample (d/categorical [0.5 0.5]) k)) "categorical gives 0-based integers")
      (is (instance? Long (dc/sample (d/poisson 3.0) k))))
    (testing "batch shapes broadcast with parameter shapes"
      (is (= [5] (a/shape (dc/sample (d/normal 0 1) k [5]))))
      (is (= [5 3] (a/shape (dc/sample (d/normal (a/zeros [3]) 1) k [5 1]))))
      (is (= [4] (a/shape (dc/sample (d/categorical-logits (a/zeros [4 6])) k))) "one draw per row of logits")
      (is (= [2 3] (a/shape (dc/sample (d/dirichlet [1.0 2.0 3.0]) k [2]))))
      (is (= [7 2] (a/shape (dc/sample (d/mvnormal [0.0 0.0] [[1.0 0.0] [0.0 1.0]]) k [7])))))
    (testing "array parameters given as Clojure vectors"
      (is (a/allclose? (a/array [0.9 0.1]) (:probs (:params (d/categorical [0.9 0.1]))))))))

(deftest validation
  (is (thrown? clojure.lang.ExceptionInfo (d/normal 0 -1)))
  (is (thrown? clojure.lang.ExceptionInfo (d/gamma 0 1)))
  (is (thrown? clojure.lang.ExceptionInfo (d/bernoulli 1.5)))
  (is (thrown? clojure.lang.ExceptionInfo (d/uniform 2 1)))
  (testing "categorical takes probabilities; logits are caught with a hint"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"categorical-logits"
                          (d/categorical [-1.2 -0.4 -2.0])))
    (is (thrown? clojure.lang.ExceptionInfo (d/categorical [0.5 0.6])))
    (is (some? (d/categorical-logits [-1.2 -0.4 -2.0])))))

(deftest supports
  (is (= [false true] (dc/support (d/bernoulli 0.3))))
  (is (= [0 1 2] (dc/support (d/categorical [0.2 0.3 0.5]))))
  (is (= [2 3 4 5] (dc/support (d/uniform-discrete 2 5))))
  (is (= [0 1 2 3] (dc/support (d/binom 3 0.5))))
  (testing "probabilities over the support sum to 1"
    (doseq [dist [(d/bernoulli 0.3) (d/categorical [0.2 0.3 0.5]) (d/binom 6 0.35)]]
      (is (a/allclose? 1.0 (reduce + (map #(Math/exp (a/item (dc/log-prob dist %))) (dc/support dist))))))))

(deftest outside-the-support
  (is (= ##-Inf (dc/log-prob (d/categorical [0.5 0.5]) 2)))
  (is (= ##-Inf (dc/log-prob (d/categorical [0.5 0.5]) 0.5)) "non-integers are outside the support")
  (is (= ##-Inf (dc/log-prob (d/categorical [0.5 0.5]) -1)))
  (is (= ##-Inf (dc/log-prob (d/bernoulli 0.3) 0.5)))
  (is (= ##-Inf (dc/log-prob (d/poisson 2.0) 1.5)))
  (is (= ##-Inf (dc/log-prob (d/dirichlet [1.0 1.0]) (a/array [0.7 0.7])))))

(deftest distributions-are-generative-functions
  (let [dist (d/normal 1.0 2.0)
        k (rng/key 8)
        tr (gfi/simulate dist [] k)]
    (is (= (:retval tr) (cm/get-value (:choices tr))))
    (is (= (:score tr) (:weight (gfi/assess dist [] (:choices tr)))))
    (is (= 0.0 (:weight (gfi/generate dist [] cm/EMPTY k))))
    (let [{g :trace w :weight} (gfi/generate dist [] (cm/->Value 3.0) k)]
      (is (= 3.0 (:retval g)))
      (is (= w (dc/log-prob dist 3.0))))
    (let [{t2 :trace w :weight disc :discard} (gfi/update tr (cm/->Value 0.0))]
      (is (= 0.0 (:retval t2)))
      (is (a/allclose? w (- (dc/log-prob dist 0.0) (:score tr))))
      (is (= (:choices tr) disc)))
    (is (= (:score tr) (gfi/project tr sel/all)))
    (is (= 0.0 (gfi/project tr (sel/select :x))))
    (is (= tr (:trace (gfi/regenerate tr sel/none k))))
    (is (= 0.0 (:weight (gfi/regenerate tr sel/all k))))))

(defn- close? [actual expected band] (< (Math/abs (- (double actual) (double expected))) band))

(deftest sampler-moments
  ;; fixed key, so deterministic; bands about five standard errors
  (let [k (rng/key 99) n 100000
        check (fn [dist mean var]
                (let [x (dc/sample dist k [n])]
                  (is (close? (a/mean x) mean (* 5 (Math/sqrt (/ var n)))) (str dist " mean"))
                  (is (close? (a/variance x) var (* 0.05 var)) (str dist " variance"))))]
    (check (d/normal 0.5 2.0) 0.5 4.0)
    (check (d/uniform -1.0 3.0) 1.0 (/ 16.0 12))
    (check (d/beta 2.5 0.7) (/ 2.5 3.2) (/ (* 2.5 0.7) (* 3.2 3.2 4.2)))
    (check (d/gamma 2.5 1.5) 3.75 5.625)
    (check (d/inv-gamma 4.0 2.0) (/ 2.0 3) (/ 4.0 18))
    (check (d/exponential 1.5) (/ 1 1.5) (/ 1 2.25))
    (check (d/laplace 1.0 0.5) 1.0 0.5)
    (check (d/poisson 3.5) 3.5 3.5)
    (check (d/poisson 42.0) 42.0 42.0)
    (check (d/binom 10 0.3) 3.0 2.1)
    (check (d/binom 200 0.7) 140.0 42.0)
    (check (d/geometric 0.25) 3.0 12.0)
    (check (d/neg-binom 2.5 0.4) 3.75 9.375)
    (check (d/uniform-discrete 2 5) 3.5 1.25)
    (check (d/categorical [0.2 0.5 0.3]) 1.1 0.49)
    (check (d/piecewise-uniform [0.0 1.0 3.0 4.0] [0.5 0.2 0.3]) 1.7 (- (+ (/ 0.5 3) (* 0.2 (/ 13.0 3)) (* 0.3 (/ 37.0 3))) 2.89))))
