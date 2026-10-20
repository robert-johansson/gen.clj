(ns gen.dist-test
  (:refer-clojure :exclude [-])
  (:require [gen.test-support :as ts]
            [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [-]]
            [gen.array.special]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private fixtures (edn/read-string (slurp "test/fixtures/distributions.edn")))

(defn- constructor [k] @(ns-resolve 'gen.dist (symbol (name k))))

(deftest log-probs-match-scipy
  (doseq [{:keys [dist params value log-prob]} fixtures]
    ;; scipy's bernoulli values are 0/1; gen.clj's are booleans, as Gen.jl's
    (let [value (if (= :bernoulli dist) (== 1 value) value)
          got (dc/log-prob (apply (constructor dist) params) (if (vector? value) (array/array value) value))]
      (is (or (= log-prob (array/item got))
              (< (Math/abs (- (double (array/item got)) log-prob)) (* (ts/tol 1e-12) (max 1.0 (Math/abs log-prob)))))
          (str (name dist) " " params " at " value)))))

(deftest values-and-shapes
  (let [k (rng/key 3)]
    (testing "scalar draws: host values on the JVM backend, 0-d arrays on others"
      (is (instance? Double (ts/host (dc/sample (d/normal 0 1) k))))
      (is (instance? Boolean (ts/host (dc/sample (d/bernoulli 0.5) k))) "bernoulli gives booleans, as Gen.jl")
      (is (instance? Long (ts/host (dc/sample (d/categorical [0.5 0.5]) k))) "categorical gives 0-based integers")
      (is (instance? Long (ts/host (dc/sample (d/poisson 3.0) k))))
      (when (ts/jvm?)
        (is (instance? Double (dc/sample (d/normal 0 1) k)))))
    (testing "batch shapes broadcast with parameter shapes"
      (is (= [5] (array/shape (dc/sample (d/normal 0 1) k [5]))))
      (is (= [5 3] (array/shape (dc/sample (d/normal (array/zeros [3]) 1) k [5 1]))))
      (is (= [4] (array/shape (dc/sample (d/categorical-logits (array/zeros [4 6])) k))) "one draw per row of logits")
      (is (= [2 3] (array/shape (dc/sample (d/dirichlet [1.0 2.0 3.0]) k [2]))))
      (is (= [7 2] (array/shape (dc/sample (d/mvnormal [0.0 0.0] [[1.0 0.0] [0.0 1.0]]) k [7])))))
    (testing "array parameters given as Clojure vectors"
      (is (array/allclose? (array/array [0.9 0.1]) (:probs (:params (d/categorical [0.9 0.1]))))))))

(deftest validation
  (is (thrown? clojure.lang.ExceptionInfo (d/normal 0 -1)))
  (is (thrown? clojure.lang.ExceptionInfo (d/gamma 0 1)))
  (is (thrown? clojure.lang.ExceptionInfo (d/bernoulli 1.5)))
  (is (thrown? clojure.lang.ExceptionInfo (d/uniform 2 1)))
  (testing "categorical takes probabilities; logits are caught with a hint"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"categorical-logits"
                          (d/categorical [-1.2 -0.4 -2.0])))
    (is (thrown? clojure.lang.ExceptionInfo (d/categorical [0.5 0.6])))
    (is (some? (d/categorical-logits [-1.2 -0.4 -2.0]))))
  (testing "array parameters are validated too"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"std must be positive" (d/normal 0 [1.0 -1.0])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"in \[0, 1\]" (d/bernoulli [0.5 1.5])))
    (is (some? (d/normal [0 0] [1.0 2.0]))))
  (testing "counts and integer bounds must be integers"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom 2.5 0.3)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom -1 0.3)))
    (is (some? (d/binom 3.0 0.3)) "an integral float counts")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"low must be an integer" (d/uniform-discrete 0.5 3)))
    (is (some? (d/uniform-discrete -2 3)))))

(deftest supports
  (is (= [false true] (dc/support (d/bernoulli 0.3))))
  (is (= [0 1 2] (dc/support (d/categorical [0.2 0.3 0.5]))))
  (is (= [2 3 4 5] (dc/support (d/uniform-discrete 2 5))))
  (is (= [0 1 2 3] (dc/support (d/binom 3 0.5))))
  (testing "probabilities over the support sum to 1"
    (doseq [dist [(d/bernoulli 0.3) (d/categorical [0.2 0.3 0.5]) (d/binom 6 0.35)]]
      (is (array/allclose? 1.0 (reduce + (map #(Math/exp (array/item (dc/log-prob dist %))) (dc/support dist))))))))

(deftest outside-the-support
  (is (= ##-Inf (ts/host (dc/log-prob (d/categorical [0.5 0.5]) 2))))
  (is (= ##-Inf (ts/host (dc/log-prob (d/categorical [0.5 0.5]) 0.5))) "non-integers are outside the support")
  (is (= ##-Inf (ts/host (dc/log-prob (d/categorical [0.5 0.5]) -1))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"booleans" (dc/log-prob (d/bernoulli 0.3) 0.5))
      "a bernoulli value is a boolean: 0/1 (GenMLX's) would be scored one way and branched on the other")
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"booleans" (dc/log-prob (d/bernoulli 0.3) 0)))
  (is (= ##-Inf (dc/log-prob (d/poisson 2.0) 1.5)))
  (is (= ##-Inf (ts/host (dc/log-prob (d/dirichlet [1.0 1.0]) (array/array [0.7 0.7]))))))

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
      (is (array/allclose? w (- (dc/log-prob dist 0.0) (:score tr))))
      (is (= (:choices tr) disc)))
    (is (= (:score tr) (gfi/project tr sel/all)))
    (is (= 0.0 (gfi/project tr (sel/select :x))))
    (is (= tr (:trace (gfi/regenerate tr sel/none k))))
    (is (= 0.0 (:weight (gfi/regenerate tr sel/all k))))))

(defn- close? [actual expected band] (< (Math/abs (- (ts/number actual) (ts/number expected))) band))

(deftest sampler-moments
  ;; fixed key, so deterministic; bands about five standard errors
  (let [k (rng/key 99) n 100000
        check (fn [dist mean var]
                (let [x (dc/sample dist k [n])]
                  (is (close? (array/mean x) mean (* 5 (Math/sqrt (/ var n)))) (str dist " mean"))
                  (is (close? (array/variance x) var (* 0.05 var)) (str dist " variance"))))]
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
    (check (d/piecewise-uniform [0.0 1.0 3.0 4.0] [0.5 0.2 0.3]) 1.7 (- (+ (/ 0.5 3) (* 0.2 (/ 13.0 3)) (* 0.3 (/ 37.0 3))) 2.89))
    ;; Distributions.jl's families (the variance band needs a finite fourth
    ;; moment: Pareto's shape above 4, Student's nu above 4)
    (let [g #(Math/exp (array/item (array/lgamma (double %))))]
      (check (d/chi-squared 3.0) 3.0 6.0)
      (check (d/f-dist 3.0 12.0) (/ 12.0 10) (/ (* 2 144.0 13) (* 3 100 8)))
      (check (d/gumbel 1.0 2.0) (+ 1.0 (* 2.0 0.5772156649015329)) (/ (* Math/PI Math/PI 4.0) 6))
      (check (d/log-normal 0.2 0.5) (Math/exp 0.325) (* (- (Math/exp 0.25) 1) (Math/exp 0.65)))
      (check (d/logistic 0.5 1.5) 0.5 (/ (* Math/PI Math/PI 2.25) 3))
      (check (d/nakagami 1.5 2.0) (* (/ (g 2.0) (g 1.5)) (Math/sqrt (/ 2.0 1.5)))
             (* 2.0 (- 1 (* (/ 1 1.5) (Math/pow (/ (g 2.0) (g 1.5)) 2)))))
      (check (d/pareto 5.0 1.5) (/ (* 5.0 1.5) 4.0) (/ (* 1.5 1.5 5.0) (* 16.0 3.0)))
      (check (d/student-t 6.0) 0.0 1.5)
      (check (d/triangular 0.0 2.0 0.5) (/ 2.5 3) (/ 3.25 18))
      (check (d/weibull 1.5 2.0) (* 2.0 (g (+ 1 (/ 1 1.5)))) (* 4.0 (- (g (+ 1 (/ 2 1.5))) (Math/pow (g (+ 1 (/ 1 1.5))) 2)))))))

(deftest the-levy-distribution-s-cdf
  ;; no mean: P(X <= mu + 4 sigma) = erfc(sqrt(1/8)), within five standard errors
  (let [n 100000
        x (dc/sample (d/levy 1.0 0.5) (rng/key 98) [n])
        p (gen.array.special/erfc (Math/sqrt (/ 0.5 (* 2 2.0))))
        f (/ (ts/number (array/sum (array/astype (array/<= x 3.0) :float32))) n)]
    (is (< (Math/abs (- f p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))) (str f " against " p))))

;; ---------------------------------------------------------------------------
;; Mixtures and products
;; ---------------------------------------------------------------------------

(def normal-mixture (d/homogeneous-mixture d/normal))
(def normal-uniform (d/heterogeneous-mixture [d/normal d/uniform]))
(def normal-bernoulli (d/product d/normal d/bernoulli))

(defn- density [dist v] (Math/exp (array/item (dc/log-prob dist v))))

(deftest mixture-and-product-densities
  (let [m (normal-mixture [0.3 0.7] [-2.0 1.0] [0.5 1.0])]
    (doseq [v [-2.5 0.0 1.7]]
      (is (close? (array/item (dc/log-prob m v))
                  (Math/log (+ (* 0.3 (density (d/normal -2.0 0.5) v)) (* 0.7 (density (d/normal 1.0 1.0) v))))
                  (ts/tol 1e-12)))))
  (let [h (normal-uniform [0.4 0.6] [0.0 1.0] [-1.0 1.0])]
    (is (close? (array/item (dc/log-prob h 0.5)) (Math/log (+ (* 0.4 (density (d/normal 0 1) 0.5)) (* 0.6 0.5))) (ts/tol 1e-12)))
    (is (close? (array/item (dc/log-prob h 3.0)) (Math/log (* 0.4 (density (d/normal 0 1) 3.0))) (ts/tol 1e-12))
        "outside one component's support"))
  (let [p (normal-bernoulli [0.5 2.0] [0.3])]
    (is (close? (array/item (dc/log-prob p [1.0 true])) (+ (array/item (dc/log-prob (d/normal 0.5 2.0) 1.0)) (Math/log 0.3)) (ts/tol 1e-12)))
    (is (vector? (dc/sample p (rng/key 1))))))

(deftest mixture-moments
  (let [k (rng/key 98) n 100000
        check (fn [dist mean var]
                (let [x (dc/sample dist k [n])]
                  (is (close? (array/mean x) mean (* 5 (Math/sqrt (/ var n)))) "mean")
                  (is (close? (array/variance x) var (* 0.05 var)) "variance")))]
    ;; mean Σ wμ, variance Σ w(σ² + μ²) - mean²
    (check (normal-mixture [0.3 0.7] [-2.0 1.0] [0.5 1.0]) 0.1 (- (+ (* 0.3 4.25) (* 0.7 2.0)) 0.01))
    (check (normal-uniform [0.5 0.5] [3.0 1.0] [-1.0 1.0]) 1.5 (- (+ (* 0.5 10.0) (* 0.5 (/ 1.0 3))) 2.25))))

(deftest mixtures-validate-their-weights
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (normal-mixture [0.5 0.6] [0.0 1.0] [1.0 1.0])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one weight" (normal-uniform [1.0] [0.0 1.0] [0.0 1.0]))))

(deftest review-regressions
  (testing "product sums each factor over its own value (review 4)"
    (let [p ((d/product d/normal d/bernoulli) [(array/zeros [3]) 1.0] [0.3])]
      (is (close? (array/item (dc/log-prob p [(array/zeros [3]) true]))
                  (+ (* 3 (array/item (dc/log-prob (d/normal 0 1) 0.0))) (Math/log 0.3)) (ts/tol 1e-12)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one element per factor" (dc/log-prob p [(array/zeros [3])])))))
  (testing "heterogeneous mixtures keep each component's values (review 14)"
    (let [x (dc/sample ((d/heterogeneous-mixture [d/poisson d/normal]) [0.01 0.99] [3.0] [0.5 1.0]) (rng/key 1) [200])]
      (is (some #(not= % (Math/floor %)) (array/->clj x)) "normal draws are not truncated to integers")))
  (testing "bad parameters are errors, not NaN (review 15)"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive definite" (d/mvnormal [0.0 0.0] [[1.0 2.0] [2.0 1.0]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"symmetric" (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.3 1.0]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive" (d/dirichlet [-1.0 2.0])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"increasing" (d/piecewise-uniform [0 2 1] [0.5 0.5])))))

(dc/defdist softmax-categorical
  "A user-defined distribution without a value-shape clause."
  [logits]
  (sample [key shape] (rng/categorical key logits shape))
  (log-prob [v] (array/pick (array/log-softmax logits) v)))

(deftest user-distributions-without-a-value-shape-are-not-rejected
  ;; third review B13: the default value shape (the parameters' broadcast
  ;; shape, [3] here) rejected the valid constraint 1
  (let [m (gen [] (trace :k (softmax-categorical [0.0 1.0 2.0])))]
    (is (map? (gfi/generate m [] (cm/choicemap :k 1) (rng/key 1))))))

;; ---------------------------------------------------------------------------
;; Distributions.jl's families: native, so they differentiate and batch
;; ---------------------------------------------------------------------------

(def ^:private further
  "[name constructor params value]: each family at an interior value."
  [["chi-squared" d/chi-squared [3.0] 2.0] ["f-dist" d/f-dist [3.0 10.0] 1.2] ["gumbel" d/gumbel [1.0 2.0] 0.5]
   ["levy" d/levy [1.0 0.5] 3.0] ["log-normal" d/log-normal [0.2 0.5] 1.5] ["logistic" d/logistic [0.5 1.5] 0.3]
   ["nakagami" d/nakagami [1.5 2.0] 1.2] ["pareto" d/pareto [3.0 1.5] 2.5] ["student-t" d/student-t [5.0] 1.3]
   ["triangular" d/triangular [0.0 2.0 0.5] 1.4] ["weibull" d/weibull [1.5 2.0] 1.5]])

(deftest gradients-of-the-further-families
  ;; with respect to the value and each parameter, against central finite
  ;; differences of the log density (float64 on the JVM, a band on float32)
  (let [h 1e-6
        fd (fn [f x] (/ (- (f (+ x h)) (f (- x h))) (* 2 h)))]
    (doseq [[label make params v] further
            i (range -1 (count params))]
      (let [lp (fn [x] (ts/number (dc/log-prob (apply make (if (neg? i) params (assoc params i x))) (if (neg? i) x v))))
            at (if (neg? i) v (params i))
            ;; the parameter as an array, so the tape differentiates it
            g (if (neg? i)
                (array/grad #(dc/log-prob (apply make params) %))
                (array/grad #(dc/log-prob (apply make (assoc params i %)) v)))]
        (is (ts/close? (fd lp at) (ts/number (g (array/array at))) {:rtol (ts/tol 1e-5) :atol (ts/tol 1e-6)})
            (str label (if (neg? i) " d/dv" (str " d/dparam" i))))))))

(def further-model
  (gen []
    (let [nu (trace :nu (d/gamma 4 1))
          t (trace :t (d/student-t nu))
          w (trace :w (d/weibull 1.5 2.0))
          l (trace :l (d/log-normal 0.0 0.5))
          tri (trace :tri (d/triangular 0.0 2.0 0.5))]
      (trace :y (d/logistic (array/+ (array/+ t w) (array/+ l tri)) 1.0)))))

(laws/deflaws further [[:model further-model [] []]] 100)

(deftest the-further-families-batch
  (let [ps (batched/simulate further-model [] 8 (rng/key 97))]
    (is (nil? (batched/fallback-reason ps)))))
