(ns gen.linalg-test
  (:refer-clojure :exclude [+ * >=])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ * >=]]
            [gen.array.linalg :as la]
            [gen.random :as rng]))

(defn- spd
  "A random symmetric positive-definite n×n matrix (batched by batch-shape)."
  [key batch-shape n]
  (let [m (rng/normal key (into (vec batch-shape) [n n]))]
    (+ (array/matmul m (array/transpose m (into (vec (range (count batch-shape)))
                                                [(inc (count batch-shape)) (count batch-shape)])))
       (* n (array/eye n)))))

(def gen-case (gen/tuple gen/nat (gen/choose 1 6)))

(defspec cholesky-factors-reconstruct 100
  (prop/for-all [[seed n] gen-case]
    (let [m (spd (rng/key seed) [] n)
          l (la/cholesky m)]
      (and (array/allclose? (array/matmul l (array/transpose l)) m {:rtol (ts/tol 1e-10) :atol (ts/tol 1e-10)})
           (array/allclose? l (* l (>= (array/expand-dims (array/arange n) -1) (array/arange n))))))))

(defspec solves-solve 100
  (prop/for-all [[seed n] gen-case]
    (let [[k1 k2] (rng/split (rng/key seed))
          m (spd k1 [] n)
          b (rng/normal k2 [n])
          l (la/cholesky m)]
      (and (array/allclose? (array/matmul m (la/solve m b)) b {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-9)})
           (array/allclose? (array/matmul l (la/solve-triangular l b)) b {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-9)})
           (array/allclose? (array/matmul (array/transpose l) (la/solve-triangular l b {:transpose? true})) b
                            {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-9)})
           (array/allclose? (array/matmul m (la/inv m)) (array/eye n) {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-9)})))))

(defspec logdets-agree 100
  (prop/for-all [[seed n] gen-case]
    (let [m (spd (rng/key seed) [] n)
          [sign logabs] (la/slogdet m)]
      (and (= 1.0 (ts/host sign))
           ;; with an absolute band too: log-determinants near 0 (det near 1)
           ;; carry the absolute error of a log near 1
           (array/allclose? logabs (la/cholesky-logdet (la/cholesky m)) {:rtol (ts/tol 1e-10) :atol (ts/tol 1e-10)})))))

(deftest batched-matches-one-at-a-time
  (let [ms (spd (rng/key 5) [3] 4)
        bs (rng/normal (rng/key 6) [3 4])]
    (is (array/allclose? (la/cholesky ms) (array/stack (mapv #(la/cholesky (array/take ms %)) (range 3)))))
    (is (array/allclose? (la/solve ms bs) (array/stack (mapv #(la/solve (array/take ms %) (array/take bs %)) (range 3)))))
    (is (array/allclose? (second (la/slogdet ms))
                         (array/stack (mapv #(second (la/slogdet (array/take ms %))) (range 3)))))
    (testing "one right-hand side broadcast against a batch of matrices"
      (is (= [3 4] (array/shape (la/solve ms (array/take bs 0))))))))

(deftest known-values-and-failure-modes
  (is (array/allclose? (array/array [[2.0 0.0] [1.0 1.0]])
                       (la/cholesky (array/array [[4.0 2.0] [2.0 2.0]]))))
  (is (array/allclose? 44.6 (la/det (array/array [[4.0 2.0 0.6] [2.0 5.0 1.0] [0.6 1.0 3.0]]))))
  (is (= -1.0 (ts/host (first (la/slogdet (array/array [[0.0 1.0] [1.0 0.0]]))))) "a row swap flips the sign")
  (is (every? #(Double/isNaN %) (flatten (array/->clj (la/cholesky (array/array [[1.0 2.0] [2.0 1.0]])))))
      "a matrix that is not positive definite gives NaN, not an exception")
  (is (= [0.0 ##-Inf] (ts/host (la/slogdet (array/array [[1.0 2.0] [2.0 4.0]])))) "singular")
  (is (= [1.0 2.0] (array/->clj (la/diagonal (array/array [[1.0 9.0] [9.0 2.0]])))))
  (is (thrown? clojure.lang.ExceptionInfo (la/cholesky (array/ones [2 3])))))

(deftest solve-checks-its-right-hand-side
  ;; third review B12
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"do not match"
                        (la/solve (array/eye 3) (array/ones [4 2])))))
