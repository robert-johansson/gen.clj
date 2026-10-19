(ns gen.linalg-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.array.linalg :as la]
            [gen.random :as rng]))

(defn- spd
  "A random symmetric positive-definite n×n matrix (batched by batch-shape)."
  [key batch-shape n]
  (let [m (rng/normal key (into (vec batch-shape) [n n]))]
    (a/+ (a/matmul m (a/transpose m (into (vec (range (count batch-shape)))
                                          [(inc (count batch-shape)) (count batch-shape)])))
         (a/* n (a/eye n)))))

(def gen-case (gen/tuple gen/nat (gen/choose 1 6)))

(defspec cholesky-factors-reconstruct 100
  (prop/for-all [[seed n] gen-case]
    (let [m (spd (rng/key seed) [] n)
          l (la/cholesky m)]
      (and (a/allclose? (a/matmul l (a/transpose l)) m {:rtol 1e-10 :atol 1e-10})
           (a/allclose? l (a/* l (a/>= (a/expand-dims (a/arange n) -1) (a/arange n))))))))

(defspec solves-solve 100
  (prop/for-all [[seed n] gen-case]
    (let [[k1 k2] (rng/split (rng/key seed))
          m (spd k1 [] n)
          b (rng/normal k2 [n])
          l (la/cholesky m)]
      (and (a/allclose? (a/matmul m (la/solve m b)) b {:rtol 1e-9 :atol 1e-9})
           (a/allclose? (a/matmul l (la/solve-triangular l b)) b {:rtol 1e-9 :atol 1e-9})
           (a/allclose? (a/matmul (a/transpose l) (la/solve-triangular l b {:transpose? true})) b
                        {:rtol 1e-9 :atol 1e-9})
           (a/allclose? (a/matmul m (la/inv m)) (a/eye n) {:rtol 1e-9 :atol 1e-9})))))

(defspec logdets-agree 100
  (prop/for-all [[seed n] gen-case]
    (let [m (spd (rng/key seed) [] n)
          [sign logabs] (la/slogdet m)]
      (and (= 1.0 (ts/host sign))
           (a/allclose? logabs (la/cholesky-logdet (la/cholesky m)) {:rtol 1e-10})))))

(deftest batched-matches-one-at-a-time
  (let [ms (spd (rng/key 5) [3] 4)
        bs (rng/normal (rng/key 6) [3 4])]
    (is (a/allclose? (la/cholesky ms) (a/stack (mapv #(la/cholesky (a/take ms %)) (range 3)))))
    (is (a/allclose? (la/solve ms bs) (a/stack (mapv #(la/solve (a/take ms %) (a/take bs %)) (range 3)))))
    (is (a/allclose? (second (la/slogdet ms))
                     (a/stack (mapv #(second (la/slogdet (a/take ms %))) (range 3)))))
    (testing "one right-hand side broadcast against a batch of matrices"
      (is (= [3 4] (a/shape (la/solve ms (a/take bs 0))))))))

(deftest known-values-and-failure-modes
  (is (a/allclose? (a/array [[2.0 0.0] [1.0 1.0]])
                   (la/cholesky (a/array [[4.0 2.0] [2.0 2.0]]))))
  (is (a/allclose? 44.6 (la/det (a/array [[4.0 2.0 0.6] [2.0 5.0 1.0] [0.6 1.0 3.0]]))))
  (is (= -1.0 (ts/host (first (la/slogdet (a/array [[0.0 1.0] [1.0 0.0]]))))) "a row swap flips the sign")
  (is (every? #(Double/isNaN %) (flatten (a/->clj (la/cholesky (a/array [[1.0 2.0] [2.0 1.0]])))))
      "a matrix that is not positive definite gives NaN, not an exception")
  (is (= [0.0 ##-Inf] (ts/host (la/slogdet (a/array [[1.0 2.0] [2.0 4.0]])))) "singular")
  (is (= [1.0 2.0] (a/->clj (la/diagonal (a/array [[1.0 9.0] [9.0 2.0]])))))
  (is (thrown? clojure.lang.ExceptionInfo (la/cholesky (a/ones [2 3])))))

(deftest solve-checks-its-right-hand-side
  ;; third review B12
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"do not match"
                        (la/solve (a/eye 3) (a/ones [4 2])))))
