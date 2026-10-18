(ns gen.array-test
  (:require [clojure.test :refer [deftest is testing are]]
            [gen.array :as a]))

(defn- thrown-data
  "ex-data of the exception thrown by (f), or nil."
  [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(deftest host-math
  (testing "number-only ops stay on the host and follow NumPy semantics"
    (are [expected actual] (= expected actual)
      3 (a/+ 1 2)
      3.5 (a/+ 1 2.5)
      0.5 (a// 1 2)
      2 (a/+ true true)
      -4 (a/floor-div -7 2)
      1 (a/mod -7 2)
      0.5 (a/mod -1.5 1.0)
      0 (a/mod 5 0)
      8 (a/pow 2 3)
      -1 (a/negate true)
      1.0 (a/where true 1 2.0)
      true (a/< 1 2)
      false (a/== ##NaN ##NaN)
      true (a/!= ##NaN ##NaN)
      2.0 (a/round 2.5)
      4.0 (a/round 3.5)))
  (testing "variadic arithmetic"
    (is (= 10 (a/+ 1 2 3 4)))
    (is (= 0 (a/+)))
    (is (= -5 (a/- 5)))
    (is (= 0.25 (a// 4)))
    (is (= 24 (a/* 1 2 3 4)))))

(deftest creation-and-conversion
  (is (= [[1 2] [3 4]] (a/->clj (a/array [[1 2] [3 4]]))))
  (is (= :int64 (a/dtype (a/array [1 2]))))
  (is (= :float64 (a/dtype (a/array [1.0 2]))))
  (is (= :bool (a/dtype (a/array [true false]))))
  (is (= :float64 (a/dtype (a/array []))))
  (is (= [0] (a/shape (a/array []))))
  (is (= 3.0 (a/array 3.0)) "0-d arrays are host values on the JVM backend")
  (is (= [1.0 2.0] (a/->clj (a/array [1 2] {:dtype :float64}))))
  (is (thrown-data #(a/array [[1 2] [3]])) "ragged data is rejected")
  (is (thrown-data #(a/array [[1 2] [3 [4]]])) "deeply ragged data is rejected")
  (is (thrown-data #(a/+ [1 2] 1)) "Clojure vectors are not arrays")
  (is (= [[0.0 0.0]] (a/->clj (a/zeros [1 2]))))
  (is (= [1 1 1] (a/->clj (a/full [3] 1))))
  (is (= [0 1 2 3] (a/->clj (a/arange 4))))
  (is (= [1.0 1.5 2.0] (a/->clj (a/arange 1 2.5 0.5))))
  (is (= [0.0 0.5 1.0] (a/->clj (a/linspace 0 1 3))))
  (is (= [7 7] (a/->clj (a/full-like (a/array [1 2]) 7))))
  (is (= 5 (a/item (a/array [[5]]))))
  (is (thrown-data #(a/item (a/array [1 2])))))

(deftest broadcasting
  (is (= [[11 12 13] [21 22 23]] (a/->clj (a/+ (a/array [1 2 3]) (a/array [[10] [20]])))))
  (is (= [[2 4 6]] (a/->clj (a/* (a/array [[1 2 3]]) 2))))
  (is (= [3.0 2.0 1.0] (a/->clj (a/- 4 (a/array [1.0 2.0 3.0])))))
  (is (:shapes (thrown-data #(a/+ (a/array [1 2 3]) (a/array [1 2]))))))

(deftest dtype-promotion
  (testing "host numbers are weakly typed"
    (is (= :float32 (a/dtype (a/+ (a/array [1.5] {:dtype :float32}) 2.5))))
    (is (= :float64 (a/dtype (a/+ (a/array [1 2]) 2.5))))
    (is (= :int64 (a/dtype (a/+ (a/array [1 2]) 2))))
    (is (= :int64 (a/dtype (a/+ (a/array [true]) 1)))))
  (testing "strong dtypes promote"
    (is (= :float64 (a/dtype (a/+ (a/array [1] {:dtype :int32}) (a/array [1.0])))))
    (is (= :float32 (a/dtype (a/+ (a/array [1] {:dtype :int64}) (a/array [1.0] {:dtype :float32})))))
    (is (= :int64 (a/dtype (a/+ (a/array [1] {:dtype :int32}) (a/array [1]))))))
  (testing "division and float functions produce floats"
    (is (= [0.5 1.0] (a/->clj (a// (a/array [1 2]) 2))))
    (is (= :float64 (a/dtype (a/exp (a/array [0 1]))))))
  (testing "comparisons and logic produce bools"
    (is (= [true false] (a/->clj (a/< (a/array [1 3]) 2))))
    (is (= [false true] (a/->clj (a/logical-not (a/array [1.0 0.0]))))))
  (testing "float32 results are rounded like MLX's"
    (is (= [(double (float 0.3))] (a/->clj (a/+ (a/array [0.1] {:dtype :float32}) 0.2))))
    (is (= [(double (float (+ (double (float 0.1)) (double (float 0.2)))))]
           (a/->clj (a/+ (a/array [0.1] {:dtype :float32}) (a/array [0.2] {:dtype :float32}))))))
  (testing "int32 wraps"
    (is (= [Integer/MIN_VALUE] (a/->clj (a/+ (a/array [Integer/MAX_VALUE] {:dtype :int32}) 1))))))

(deftest elementwise-math
  (is (a/allclose? (a/array [1.0 (Math/E)]) (a/exp (a/array [0.0 1.0]))))
  (is (a/allclose? (a/array [0.0 1.0]) (a/log (a/array [1.0 (Math/E)]))))
  (is (= [1 0 -1] (a/->clj (a/sign (a/array [5 0 -2])))))
  (is (= [3 1] (a/->clj (a/abs (a/array [-3 1])))))
  (is (= [1 -4] (a/->clj (a/floor-div (a/array [3 -7]) 2))))
  (is (= [1 1] (a/->clj (a/mod (a/array [3 -7]) 2))))
  (is (= [true false] (a/->clj (a/isnan (a/array [##NaN 1.0])))))
  (is (= [false true false] (a/->clj (a/isfinite (a/array [##Inf 1.0 ##NaN])))))
  (is (= [1.0 -5.0] (a/->clj (a/where (a/array [true false]) (a/array [1.0 2.0]) -5))))
  (is (= [1 3] (a/->clj (a/maximum (a/array [1 2]) (a/array [0 3])))))
  (is (= [0.0 2.0 3.0] (a/->clj (a/clip (a/array [-1.0 2.0 9.0]) 0 3)))))

(deftest stable-composites
  (testing "logaddexp, softplus, sigmoid at extremes"
    (is (= ##-Inf (a/logaddexp ##-Inf ##-Inf)))
    (is (= ##Inf (a/logaddexp ##Inf 0.0)))
    (is (a/allclose? 1000.6931471805599 (a/logaddexp 1000.0 1000.0)))
    (is (a/allclose? 800.0 (a/softplus 800.0)))
    (is (a/allclose? 1.0 (a/sigmoid 800.0)))
    (is (a/allclose? 0.0 (a/sigmoid -800.0)))
    (is (= 0.0 (a/xlogy 0.0 0.0))))
  (testing "logsumexp and softmax"
    (is (a/allclose? 1000.6931471805599 (a/logsumexp (a/array [1000.0 1000.0]))))
    (is (= ##-Inf (a/logsumexp (a/array [##-Inf ##-Inf]))))
    (is (a/allclose? (a/array [[0.6931471805599453] [2.0]])
                     (a/logsumexp (a/array [[0.0 0.0] [2.0 ##-Inf]]) {:axis 1 :keep-dims? true})))
    (is (a/allclose? 1.0 (a/sum (a/softmax (a/array [1.0 2.0 3.0])))))
    (is (a/allclose? (a/log (a/softmax (a/array [1.0 2.0 3.0])))
                     (a/log-softmax (a/array [1.0 2.0 3.0]))))))

(deftest reduction-ops
  (let [x (a/array [[1 2 3] [4 5 6]])]
    (is (= 21 (a/sum x)))
    (is (= [5 7 9] (a/->clj (a/sum x {:axis 0}))))
    (is (= [[6] [15]] (a/->clj (a/sum x {:axis 1 :keep-dims? true}))))
    (is (= 720 (a/prod x)))
    (is (= 6 (a/max x)))
    (is (= [1 4] (a/->clj (a/min x {:axis -1}))))
    (is (= 5 (a/argmax x)))
    (is (= [2 2] (a/->clj (a/argmax x {:axis 1}))))
    (is (= 3.5 (a/mean x)))
    (is (= [2.5 3.5 4.5] (a/->clj (a/mean x {:axis 0})))))
  (is (= 2 (a/sum (a/array [true true false]))) "summing bools counts")
  (is (= true (a/any (a/array [false true]))))
  (is (= false (a/all (a/array [1 0]))))
  (is (Double/isNaN (a/max (a/array [1.0 ##NaN 3.0]))) "NaN propagates")
  (is (= 1 (a/argmax (a/array [1.0 ##NaN 3.0]))) "the first NaN wins")
  (is (= 0 (a/argmax (a/array [##-Inf ##-Inf]))))
  (is (= 1.25 (a/variance (a/array [1.0 2.0 3.0 4.0]))))
  (is (a/allclose? (/ 5.0 3.0) (a/variance (a/array [1.0 2.0 3.0 4.0]) {:ddof 1})))
  (is (= 7 (a/sum 7)) "reducing a 0-d array is the identity"))

(deftest scans
  (is (= [1 3 6] (a/->clj (a/cumsum (a/array [1 2 3])))))
  (is (= [6 5 3] (a/->clj (a/cumsum (a/array [1 2 3]) {:reverse? true}))))
  (is (= [[1 2] [4 6]] (a/->clj (a/cumsum (a/array [[1 2] [3 4]]) {:axis 0}))))
  (is (= [1 2 6] (a/->clj (a/cumprod (a/array [1 2 3])))))
  (is (a/allclose? (a/log (a/cumsum (a/exp (a/array [0.0 1.0 2.0]))))
                   (a/logcumsumexp (a/array [0.0 1.0 2.0])))))

(deftest shape-ops
  (let [x (a/array [[1 2 3] [4 5 6]])]
    (is (= [[1 2] [3 4] [5 6]] (a/->clj (a/reshape x [3 -1]))))
    (is (= [1 2 3 4 5 6] (a/->clj (a/flatten x))))
    (is (= [[1 4] [2 5] [3 6]] (a/->clj (a/transpose x))))
    (is (= [1 2 1 3] (a/shape (a/expand-dims x [0 2]))))
    (is (= [3] (a/shape (a/squeeze (a/array [[[1] [2] [3]]])))))
    (is (= [[1 2 3] [1 2 3]] (a/->clj (a/broadcast-to (a/array [1 2 3]) [2 3]))))
    (is (= [[1 2] [3 4] [5 6]] (a/->clj (a/concatenate [(a/array [[1 2]]) (a/array [[3 4] [5 6]])]))))
    (is (= [[1 3] [2 4]] (a/->clj (a/stack [(a/array [1 2]) (a/array [3 4])] 1))))
    (is (thrown-data #(a/reshape x [4 2])))
    (is (thrown-data #(a/squeeze x 0)))))

(deftest slicing
  (let [x (a/arange 10)]
    (are [expected args] (= expected (a/->clj (apply a/slice x 0 args)))
      [2 3 4] [2 5]
      [7 8 9] [-3 nil]
      [0 2 4 6 8] [nil nil 2]
      [9 6 3 0] [nil nil -3]
      [5 4 3] [5 2 -1]
      [] [5 2]
      [0 1 2 3 4 5 6 7 8 9] [-100 100]))
  (is (= [[2 3] [6 7]] (a/->clj (a/slice (a/reshape (a/arange 8) [2 4]) 1 2 nil)))))

(deftest indexing
  (let [x (a/array [[1 2] [3 4] [5 6]])]
    (is (= [3 4] (a/->clj (a/take x 1))))
    (is (= [5 6] (a/->clj (a/take x -1))))
    (is (= [[5 6] [1 2]] (a/->clj (a/take x (a/array [2 0])))))
    (is (= [2 4 6] (a/->clj (a/take x 1 1))))
    (is (= 4 (a/take (a/array [3 4]) 1)) "a 0-d result is a host value")
    (is (thrown-data #(a/take x 3)) "out-of-range indices throw")
    (is (thrown-data #(a/take x 1.0)) "indices must be integers")
    (is (= [[2] [3] [6]] (a/->clj (a/take-along-axis x (a/array [[1] [0] [1]]) 1)))))
  (testing "pick covers every batch-shape case of a categorical log-prob"
    (let [lp (a/array [[0.1 0.9] [0.6 0.4] [0.5 0.5]])]
      (is (= [0.9 0.6 0.5] (a/->clj (a/pick lp (a/array [1 0 1])))) "per-row index")
      (is (= [0.9 0.4 0.5] (a/->clj (a/pick lp 1))) "one shared index")
      (is (= [0.1 0.9 0.9] (a/->clj (a/pick (a/array [0.1 0.9]) (a/array [0 1 1])))) "batched index, shared logits")
      (is (= 0.9 (a/pick (a/array [0.1 0.9]) 1)) "scalar case"))))

(deftest matmul
  (is (= [[19 22] [43 50]] (a/->clj (a/matmul (a/array [[1 2] [3 4]]) (a/array [[5 6] [7 8]])))))
  (is (= [1 2] (a/->clj (a/matmul (a/array [1 2]) (a/array [[1 0] [0 1]])))))
  (is (= 11 (a/matmul (a/array [1 2]) (a/array [3 4]))))
  (is (= [2 2 2] (a/shape (a/matmul (a/ones [2 2 3]) (a/ones [3 2])))) "batched")
  (is (= [[3 4] [6 8]] (a/->clj (a/outer (a/array [1 2]) (a/array [3 4])))))
  (is (= [[1.0 0.0] [0.0 1.0]] (a/->clj (a/eye 2))))
  (is (= [[1 0] [0 2]] (a/->clj (a/diag (a/array [1 2])))))
  (is (= [1 5] (a/->clj (a/diag (a/array [[1 2 3] [4 5 6]])))))
  (is (thrown-data #(a/matmul (a/ones [2 3]) (a/ones [2 3])))))

(deftest branching
  (is (true? (a/truthy? (a/< 1.0 2.0))))
  (is (true? (a/truthy? 0)) "any number is truthy, as in Clojure")
  (is (true? (a/truthy? (a/array [0]))) "also a one-element numeric array")
  (is (false? (a/truthy? (a/array [false]))) "booleans branch by their value")
  (is (false? (a/truthy? false)))
  (is (true? (a/truthy? (a/array [[true]]))))
  (is (false? (a/truthy? nil)))
  (is (= {:gen.array/batched-branch true :shape [2]}
         (thrown-data #(a/truthy? (a/array [true false]))))
      "branching on a batch throws a recognizable error"))

(deftest comparison-helpers
  (is (a/equal? (a/array [1 2]) (a/array [1.0 2.0])))
  (is (not (a/equal? (a/array [1 2]) (a/array [[1 2]]))))
  (is (not (a/equal? (a/array [##NaN]) (a/array [##NaN]))))
  (is (a/allclose? (a/array [1.0 ##Inf]) (a/array [1.000001 ##Inf])))
  (is (a/allclose? (a/array [##NaN]) (a/array [##NaN]) {:equal-nan? true})))

(deftest trees
  (let [t {:a (a/array [1 2]) :b [3.0 {:c (a/array [4])}]}]
    (is (= 3 (count (a/tree-leaves t))))
    (is (= {:a [2 4] :b [6.0 {:c [8]}]}
           (a/tree-map #(a/->clj (a/* 2 %)) t)))
    (is (identical? t (a/eval! t)))))
