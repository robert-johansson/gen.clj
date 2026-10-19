(ns gen.array-test
  (:require [gen.test-support :as ts]
            [clojure.test :refer [deftest is testing are]]
            [gen.array :as array]
            [gen.random]))

(defn- thrown-data
  "ex-data of the exception thrown by (f), or nil."
  [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(deftest host-math
  (testing "number-only ops stay on the host and follow NumPy semantics"
    (are [expected actual] (= expected actual)
      3 (array/+ 1 2)
      3.5 (array/+ 1 2.5)
      0.5 (array// 1 2)
      2 (array/+ true true)
      -4 (array/floor-div -7 2)
      1 (array/mod -7 2)
      0.5 (array/mod -1.5 1.0)
      0 (array/mod 5 0)
      8 (array/pow 2 3)
      -1 (array/negate true)
      1.0 (array/where true 1 2.0)
      true (array/< 1 2)
      false (array/== ##NaN ##NaN)
      true (array/!= ##NaN ##NaN)
      2.0 (array/round 2.5)
      4.0 (array/round 3.5)))
  (testing "variadic arithmetic"
    (is (= 10 (array/+ 1 2 3 4)))
    (is (= 0 (array/+)))
    (is (= -5 (array/- 5)))
    (is (= 0.25 (array// 4)))
    (is (= 24 (array/* 1 2 3 4)))))

(deftest creation-and-conversion
  (is (= [[1 2] [3 4]] (array/->clj (array/array [[1 2] [3 4]]))))
  (is (= (ts/int-dtype) (array/dtype (array/array [1 2]))) "the backend's default dtypes")
  (is (= (ts/float-dtype) (array/dtype (array/array [1.0 2]))))
  (is (= :bool (array/dtype (array/array [true false]))))
  (is (= (ts/float-dtype) (array/dtype (array/array []))))
  (is (= [0] (array/shape (array/array []))))
  (is (= 3.0 (ts/host (array/array 3.0))))
  (if (ts/jvm?)
    (is (= 3.0 (array/array 3.0)) "0-d arrays are host values on the JVM backend")
    (is (= [] (array/shape (array/array 3.0))) "and backend arrays on lazy backends"))
  (is (= [1.0 2.0] (array/->clj (array/array [1 2] {:dtype (ts/float-dtype)}))))
  (is (thrown-data #(array/array [[1 2] [3]])) "ragged data is rejected")
  (is (thrown-data #(array/array [[1 2] [3 [4]]])) "deeply ragged data is rejected")
  (is (thrown-data #(array/+ [1 2] 1)) "Clojure vectors are not arrays")
  (is (= [[0.0 0.0]] (array/->clj (array/zeros [1 2]))))
  (is (= [1 1 1] (array/->clj (array/full [3] 1))))
  (is (= [0 1 2 3] (array/->clj (array/arange 4))))
  (is (= [1.0 1.5 2.0] (array/->clj (array/arange 1 2.5 0.5))))
  (is (= [0.0 0.5 1.0] (array/->clj (array/linspace 0 1 3))))
  (is (= [7 7] (array/->clj (array/full-like (array/array [1 2]) 7))))
  (is (= 5 (array/item (array/array [[5]]))))
  (is (thrown-data #(array/item (array/array [1 2])))))

(deftest broadcasting
  (is (= [[11 12 13] [21 22 23]] (array/->clj (array/+ (array/array [1 2 3]) (array/array [[10] [20]])))))
  (is (= [[2 4 6]] (array/->clj (array/* (array/array [[1 2 3]]) 2))))
  (is (= [3.0 2.0 1.0] (array/->clj (array/- 4 (array/array [1.0 2.0 3.0])))))
  (is (:shapes (thrown-data #(array/+ (array/array [1 2 3]) (array/array [1 2]))))))

(deftest dtype-promotion
  (testing "host numbers are weakly typed"
    (is (= :float32 (array/dtype (array/+ (array/array [1.5] {:dtype :float32}) 2.5))))
    (is (= (ts/float-dtype) (array/dtype (array/+ (array/array [1 2]) 2.5))))
    (is (= (ts/int-dtype) (array/dtype (array/+ (array/array [1 2]) 2))))
    (is (= (ts/int-dtype) (array/dtype (array/+ (array/array [true]) 1)))))
  (testing "strong dtypes promote"
    (is (= (ts/float-dtype) (array/dtype (array/+ (array/array [1] {:dtype :int32}) (array/array [1.0])))))
    (is (= :float32 (array/dtype (array/+ (array/array [1] {:dtype :int64}) (array/array [1.0] {:dtype :float32})))))
    (is (= (ts/int-dtype) (array/dtype (array/+ (array/array [1] {:dtype :int32}) (array/array [1]))))))
  (testing "division and float functions produce floats"
    (is (= [0.5 1.0] (array/->clj (array// (array/array [1 2]) 2))))
    (is (= (ts/float-dtype) (array/dtype (array/exp (array/array [0 1]))))))
  (testing "comparisons and logic produce bools"
    (is (= [true false] (array/->clj (array/< (array/array [1 3]) 2))))
    (is (= [false true] (array/->clj (array/logical-not (array/array [1.0 0.0]))))))
  (testing "float32 results are rounded like MLX's"
    (is (= [(double (float 0.3))] (array/->clj (array/+ (array/array [0.1] {:dtype :float32}) 0.2))))
    (is (= [(double (float (+ (double (float 0.1)) (double (float 0.2)))))]
           (array/->clj (array/+ (array/array [0.1] {:dtype :float32}) (array/array [0.2] {:dtype :float32}))))))
  (testing "host infinities and large doubles meet float32 arrays (Clojure's float cast would throw)"
    (is (= [##Inf ##-Inf] (array/->clj (array/* (array/array [1.0 -1.0] {:dtype :float32}) ##Inf))))
    (is (= [##Inf] (array/->clj (array/+ (array/array [1.0] {:dtype :float32}) 1e300))))
    (let [[pole one] (array/->clj (array/lgamma (array/array [0.0 1.0] {:dtype :float32})))]
      (is (= ##Inf pole) "lgamma's poles, a where on a host ##Inf")
      (is (< (abs one) 1e-6) "log Γ(1) = 0, to float32 rounding")))
  (testing "int32 wraps"
    (is (= [Integer/MIN_VALUE] (array/->clj (array/+ (array/array [Integer/MAX_VALUE] {:dtype :int32}) 1))))))

(deftest elementwise-math
  (is (array/allclose? (array/array [1.0 (Math/E)]) (array/exp (array/array [0.0 1.0]))))
  (is (array/allclose? (array/array [0.0 1.0]) (array/log (array/array [1.0 (Math/E)]))))
  (is (= [1 0 -1] (array/->clj (array/sign (array/array [5 0 -2])))))
  (is (= [3 1] (array/->clj (array/abs (array/array [-3 1])))))
  (is (= [1 -4] (array/->clj (array/floor-div (array/array [3 -7]) 2))))
  (is (= [1 1] (array/->clj (array/mod (array/array [3 -7]) 2))))
  (is (= [true false] (array/->clj (array/isnan (array/array [##NaN 1.0])))))
  (is (= [false true false] (array/->clj (array/isfinite (array/array [##Inf 1.0 ##NaN])))))
  (is (= [1.0 -5.0] (array/->clj (array/where (array/array [true false]) (array/array [1.0 2.0]) -5))))
  (is (= [1 3] (array/->clj (array/maximum (array/array [1 2]) (array/array [0 3])))))
  (is (= [0.0 2.0 3.0] (array/->clj (array/clip (array/array [-1.0 2.0 9.0]) 0 3)))))

(deftest stable-composites
  (testing "logaddexp, softplus, sigmoid at extremes"
    (is (= ##-Inf (array/logaddexp ##-Inf ##-Inf)))
    (is (= ##Inf (array/logaddexp ##Inf 0.0)))
    (is (array/allclose? 1000.6931471805599 (array/logaddexp 1000.0 1000.0)))
    (is (array/allclose? 800.0 (array/softplus 800.0)))
    (is (array/allclose? 1.0 (array/sigmoid 800.0)))
    (is (array/allclose? 0.0 (array/sigmoid -800.0)))
    (is (= 0.0 (array/xlogy 0.0 0.0))))
  (testing "logsumexp and softmax"
    (is (array/allclose? 1000.6931471805599 (array/logsumexp (array/array [1000.0 1000.0]))))
    (is (= ##-Inf (ts/host (array/logsumexp (array/array [##-Inf ##-Inf])))))
    (is (array/allclose? (array/array [[0.6931471805599453] [2.0]])
                         (array/logsumexp (array/array [[0.0 0.0] [2.0 ##-Inf]]) {:axis 1 :keep-dims? true})))
    (is (array/allclose? 1.0 (array/sum (array/softmax (array/array [1.0 2.0 3.0])))))
    (is (array/allclose? (array/log (array/softmax (array/array [1.0 2.0 3.0])))
                         (array/log-softmax (array/array [1.0 2.0 3.0]))))))

(deftest reduction-ops
  (let [x (array/array [[1 2 3] [4 5 6]])]
    (is (= 21 (ts/host (array/sum x))))
    (is (= [5 7 9] (array/->clj (array/sum x {:axis 0}))))
    (is (= [[6] [15]] (array/->clj (array/sum x {:axis 1 :keep-dims? true}))))
    (is (= 720 (ts/host (array/prod x))))
    (is (= 6 (ts/host (array/max x))))
    (is (= [1 4] (array/->clj (array/min x {:axis -1}))))
    (is (= 5 (ts/host (array/argmax x))))
    (is (= [2 2] (array/->clj (array/argmax x {:axis 1}))))
    (is (= 3.5 (ts/host (array/mean x))))
    (is (= [2.5 3.5 4.5] (array/->clj (array/mean x {:axis 0})))))
  (is (= 2 (ts/host (array/sum (array/array [true true false])))) "summing bools counts")
  (is (= true (ts/host (array/any (array/array [false true])))))
  (is (= false (ts/host (array/all (array/array [1 0])))))
  (is (Double/isNaN (ts/number (array/max (array/array [1.0 ##NaN 3.0])))) "NaN propagates")
  (is (= 1 (ts/host (array/argmax (array/array [1.0 ##NaN 3.0])))) "the first NaN wins")
  (is (= 0 (ts/host (array/argmax (array/array [##-Inf ##-Inf])))))
  (is (= 1.25 (ts/host (array/variance (array/array [1.0 2.0 3.0 4.0])))))
  (is (array/allclose? (/ 5.0 3.0) (array/variance (array/array [1.0 2.0 3.0 4.0]) {:ddof 1})))
  (is (= 7 (array/sum 7)) "reducing a 0-d array is the identity"))

(deftest scans
  (is (= [1 3 6] (array/->clj (array/cumsum (array/array [1 2 3])))))
  (is (= [6 5 3] (array/->clj (array/cumsum (array/array [1 2 3]) {:reverse? true}))))
  (is (= [[1 2] [4 6]] (array/->clj (array/cumsum (array/array [[1 2] [3 4]]) {:axis 0}))))
  (is (= [1 2 6] (array/->clj (array/cumprod (array/array [1 2 3])))))
  (is (ts/close? (array/log (array/cumsum (array/exp (array/array [0.0 1.0 2.0]))))
                 (array/logcumsumexp (array/array [0.0 1.0 2.0])))))

(deftest shape-ops
  (let [x (array/array [[1 2 3] [4 5 6]])]
    (is (= [[1 2] [3 4] [5 6]] (array/->clj (array/reshape x [3 -1]))))
    (is (= [1 2 3 4 5 6] (array/->clj (array/flatten x))))
    (is (= [[1 4] [2 5] [3 6]] (array/->clj (array/transpose x))))
    (is (= [1 2 1 3] (array/shape (array/expand-dims x [0 2]))))
    (is (= [3] (array/shape (array/squeeze (array/array [[[1] [2] [3]]])))))
    (is (= [[1 2 3] [1 2 3]] (array/->clj (array/broadcast-to (array/array [1 2 3]) [2 3]))))
    (is (= [[1 2] [3 4] [5 6]] (array/->clj (array/concatenate [(array/array [[1 2]]) (array/array [[3 4] [5 6]])]))))
    (is (= [[1 3] [2 4]] (array/->clj (array/stack [(array/array [1 2]) (array/array [3 4])] 1))))
    (is (thrown-data #(array/reshape x [4 2])))
    (is (thrown-data #(array/squeeze x 0)))))

(deftest slicing
  (let [x (array/arange 10)]
    (are [expected args] (= expected (array/->clj (apply array/slice x 0 args)))
      [2 3 4] [2 5]
      [7 8 9] [-3 nil]
      [0 2 4 6 8] [nil nil 2]
      [9 6 3 0] [nil nil -3]
      [5 4 3] [5 2 -1]
      [] [5 2]
      [0 1 2 3 4 5 6 7 8 9] [-100 100]))
  (is (= [[2 3] [6 7]] (array/->clj (array/slice (array/reshape (array/arange 8) [2 4]) 1 2 nil)))))

(deftest indexing
  (let [x (array/array [[1 2] [3 4] [5 6]])]
    (is (= [3 4] (array/->clj (array/take x 1))))
    (is (= [5 6] (array/->clj (array/take x -1))))
    (is (= [[5 6] [1 2]] (array/->clj (array/take x (array/array [2 0])))))
    (is (= [2 4 6] (array/->clj (array/take x 1 1))))
    (is (= 4 (ts/host (array/take (array/array [3 4]) 1))) "a 0-d result is a host value")
    (is (thrown-data #(array/take x 3)) "out-of-range indices throw")
    (is (thrown-data #(array/take x 1.0)) "indices must be integers")
    (is (= [[2] [3] [6]] (array/->clj (array/take-along-axis x (array/array [[1] [0] [1]]) 1)))))
  (testing "pick covers every batch-shape case of a categorical log-prob"
    (let [lp (array/array [[0.1 0.9] [0.6 0.4] [0.5 0.5]])]
      ;; the picked elements exactly (as stored in the backend's float)
      (is (array/equal? (array/array [0.9 0.6 0.5]) (array/pick lp (array/array [1 0 1]))) "per-row index")
      (is (array/equal? (array/array [0.9 0.4 0.5]) (array/pick lp 1)) "one shared index")
      (is (array/equal? (array/array [0.1 0.9 0.9]) (array/pick (array/array [0.1 0.9]) (array/array [0 1 1]))) "batched index, shared logits")
      (is (array/equal? (array/array 0.9) (array/pick (array/array [0.1 0.9]) 1)) "scalar case"))))

(deftest matmul
  (is (= [[19 22] [43 50]] (array/->clj (array/matmul (array/array [[1 2] [3 4]]) (array/array [[5 6] [7 8]])))))
  (is (= [1 2] (array/->clj (array/matmul (array/array [1 2]) (array/array [[1 0] [0 1]])))))
  (is (= 11 (ts/host (array/matmul (array/array [1 2]) (array/array [3 4])))))
  (is (= [2 2 2] (array/shape (array/matmul (array/ones [2 2 3]) (array/ones [3 2])))) "batched")
  (is (= [[3 4] [6 8]] (array/->clj (array/outer (array/array [1 2]) (array/array [3 4])))))
  (is (= [[1.0 0.0] [0.0 1.0]] (array/->clj (array/eye 2))))
  (is (= [[1 0] [0 2]] (array/->clj (array/diag (array/array [1 2])))))
  (is (= [1 5] (array/->clj (array/diag (array/array [[1 2 3] [4 5 6]])))))
  (is (thrown-data #(array/matmul (array/ones [2 3]) (array/ones [2 3])))))

(deftest branching
  (is (true? (array/truthy? (array/< 1.0 2.0))))
  (is (true? (array/truthy? 0)) "any number is truthy, as in Clojure")
  (is (true? (array/truthy? (array/array [0]))) "also a one-element numeric array")
  (is (false? (array/truthy? (array/array [false]))) "booleans branch by their value")
  (is (false? (array/truthy? false)))
  (is (true? (array/truthy? (array/array [[true]]))))
  (is (false? (array/truthy? nil)))
  (is (true? (array/truthy? (array/array [0.0 0.0]))) "a numeric array is truthy, as any Clojure value")
  (is (= {:shape [2]} (thrown-data #(array/truthy? (array/array [true false]))))
      "a bool array of several elements is ambiguous: an error, not a batch"))

(deftest comparison-helpers
  (is (array/equal? (array/array [1 2]) (array/array [1.0 2.0])))
  (is (not (array/equal? (array/array [1 2]) (array/array [[1 2]]))))
  (is (not (array/equal? (array/array [##NaN]) (array/array [##NaN]))))
  (is (array/allclose? (array/array [1.0 ##Inf]) (array/array [1.000001 ##Inf])))
  (is (array/allclose? (array/array [##NaN]) (array/array [##NaN]) {:equal-nan? true})))

(deftest trees
  (let [t {:a (array/array [1 2]) :b [3.0 {:c (array/array [4])}]}]
    (is (= 3 (count (array/tree-leaves t))))
    (is (= {:a [2 4] :b [6.0 {:c [8]}]}
           (array/tree-map #(array/->clj (array/* 2 %)) t)))
    (is (identical? t (array/eval! t)))))

(deftest compiled-functions
  ;; on the JVM backend compile is the identity; on MLX, mlx_compile
  (let [f (fn [x {:keys [mu sd]}] (array/sum (array/- (array/negate (array/log sd)) (array/* 0.5 (array/square (array// (array/- x mu) sd))))))
        cf (array/compile f)
        x (array/array [0.5 -1.0 2.0 0.25])]
    (testing "the same values as the function, for trees in and out"
      (is (ts/close? (f x {:mu 0.5 :sd 2.0}) (cf x {:mu 0.5 :sd 2.0})))
      (is (ts/close? (f x {:mu 0.5 :sd 2.0}) (cf x {:mu 0.5 :sd 2.0})) "replayed")
      (is (ts/close? (f (array/array [1.0 2.0]) {:mu 0.0 :sd 1.0}) (cf (array/array [1.0 2.0]) {:mu 0.0 :sd 1.0}))
          "another shape")
      (let [g (array/compile (fn [x] {:s (array/sum x) :m (array/max x)}))]
        (is (= #{:s :m} (set (keys (g x)))))
        (is (ts/close? (array/max x) (:m (g x))))))
    (testing "gradients compile too"
      (is (ts/close? ((array/grad #(array/sum (array/square %))) x) ((array/compile (array/grad #(array/sum (array/square %)))) x))))
    (testing "keys are not arguments: the trace would replay their draws"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"key's draws"
                            ((array/compile (fn [k x] x)) (gen.random/key 1) x))))))
