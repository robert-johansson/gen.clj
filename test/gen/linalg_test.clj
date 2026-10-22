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
                        (la/solve (array/eye 3) (array/ones [4 2]))))
  (testing "a number is no right-hand side: an ex-info on every backend, where the JVM's threw a
            NullPointerException (genclj-d681; the check removed, it does again)"
    (doseq [[label f] [["solve" la/solve] ["solve-triangular" la/solve-triangular]]]
      (let [e (try (f (array/eye 2) 1.0) nil (catch Exception e e))]
        (is (instance? clojure.lang.ExceptionInfo e) label)
        (is (= [[2 2] []] (:shapes (ex-data e))) label)))))

;; ---------------------------------------------------------------------------
;; One dtype for a backend's linalg (genclj-d681, the review's I1-05)
;; ---------------------------------------------------------------------------

(defn- linalg-dtypes
  "The dtypes of the array operands of every call (f) makes to the JVM
   backend's linalg, one set per call."
  [f]
  (let [calls (atom [])
        linalg @#'gen.backend.jvm/linalg]
    (with-redefs [gen.backend.jvm/linalg
                  (fn [op args]
                    (swap! calls conj [op (set (map array/dtype (filter #(and (array/array? %) (not (array/host? %))) args)))])
                    (linalg op args))]
      (f))
    @calls))

(deftest a-backends-linalg-takes-one-dtype
  ;; gen.array.linalg promotes the operands together, and the tape's rules
  ;; keep the cotangent and their identities in the matrix's dtype. Planted
  ;; (2026-10-22): each operand converted on its own, the tape's identities
  ;; in the default float, and the cotangent not cast, each fail. The JVM
  ;; backend's float32 arrays; its linalg is the one recorded
  (when (ts/jvm?)
    (let [m32 (array/array [[3.0 1.0] [1.0 2.0]] {:dtype :float32})
          l32 (array/array [[2.0 0.0] [1.0 1.5]] {:dtype :float32})
          r64 (array/array [1.0 2.0])
          r32 (array/array [1.0 2.0] {:dtype :float32})
          w64 (array/array [0.5 -1.5])
          one-dtype? (fn [f] (let [calls (linalg-dtypes f)] (and (seq calls) (every? #(= 1 (count (second %))) calls))))]
      (doseq [[label f] [["solve, float32 matrix and float64 rhs" #(la/solve m32 r64)]
                         ["solve, float64 matrix and float32 rhs" #(la/solve (array/astype m32 :float64) r32)]
                         ["solve-triangular, float32 and float64" #(la/solve-triangular l32 r64)]
                         ["inv of float32" #(la/inv m32)]
                         ["the tape: cholesky's rule" #((array/grad (fn [m] (array/sum (la/cholesky m)))) m32)]
                         ["the tape: slogdet's rule" #((array/grad (fn [m] (second (la/slogdet m)))) m32)]
                         ["the tape: solve's rule, a float64 cotangent of a float32 result"
                          #((array/grad (fn [m] (array/sum (* w64 (la/solve m r32))))) m32)]
                         ["the tape: solve-triangular's rule" #((array/grad (fn [l] (array/sum (* w64 (la/solve-triangular l r32))))) l32)]]]
        (is (one-dtype? f) label)))))

(deftest mixed-dtypes-promote
  ;; as gen.array's ops: a float32 operand with a float64 one gives float64,
  ;; whichever is the matrix; the JVM backend gave the matrix's
  (when (ts/float64?)
    (let [m32 (array/array [[3.0 1.0] [1.0 2.0]] {:dtype :float32})
          r64 (array/array [1.0 2.0] {:dtype :float64})]
      (is (= :float64 (array/dtype (la/solve m32 r64))))
      (is (= :float64 (array/dtype (la/solve (array/astype m32 :float64) (array/astype r64 :float32)))))
      (is (= :float64 (array/dtype (la/solve-triangular m32 r64))))
      (is (= :float32 (array/dtype (la/inv m32))) "inv keeps its matrix's dtype")
      (is (ts/close? (array/array [0.0 1.0] {:dtype :float64}) (la/solve m32 r64)) "[[3 1] [1 2]] x = [1 2]"))))

;; ---------------------------------------------------------------------------
;; Diagonals read, not computed (genclj-d681, the review's I1-09)
;; ---------------------------------------------------------------------------

(deftest diagonals-of-non-finite-entries
  ;; multiplied by the identity, an infinite or NaN entry made the zeros
  ;; beside it NaN; planted back (2026-10-22), both fail
  (let [nan? #(Double/isNaN %)]
    (testing "diag of a vector"
      (is (= [[1.0 0.0] [0.0 ##Inf]] (ts/host (array/diag (array/array [1.0 ##Inf])))))
      (let [[[a b] [c d]] (ts/host (array/diag (array/array [##NaN 1.0])))]
        (is (and (nan? a) (= [0.0 0.0 1.0] [b c d])))))
    (testing "la/diagonal, batched"
      (is (= [1.0 2.0] (ts/host (la/diagonal (array/array [[1.0 ##Inf] [0.0 2.0]])))))
      (let [ms (array/array [[[1.0 ##Inf 3.0] [4.0 5.0 6.0] [7.0 8.0 9.0]] [[-1.0 0.0 0.0] [0.0 ##-Inf 0.0] [0.0 0.0 2.5]]])]
        (is (= [[1.0 5.0 9.0] [-1.0 ##-Inf 2.5]] (ts/host (la/diagonal ms))))))
    (testing "its gradient: the identity's pattern"
      (is (= [[1.0 0.0] [0.0 1.0]] (ts/host ((array/grad #(array/sum (la/diagonal %))) (array/array [[1.0 2.0] [3.0 4.0]]))))))))
