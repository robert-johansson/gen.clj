(ns gen.dist-test
  (:refer-clojure :exclude [-])
  (:require [gen.test-support :as ts]
            [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [-]]
            [gen.array.special]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :as dsl]
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
  (testing "a family's parameters are arrays or their data: keywords are an error at construction, naming both"
    (doseq [[label f] [["labels" #(d/normal [:a :b] 1.0)] ["a keyword" #(d/normal :a 1.0)] ["a map" #(d/categorical {0 0.5 1 0.5})]]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^(normal: mu|categorical: probs) must be an array or array data" (f))
          label))
    (is (= {:distribution "normal" :param "mu"} (ex-data (try (d/normal [:a :b] 1.0) (catch clojure.lang.ExceptionInfo e e))))))
  (testing "counts and integer bounds must be integers"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom 2.5 0.3)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom -1 0.3)))
    (is (some? (d/binom 3.0 0.3)) "an integral float counts")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"low must be an integer" (d/uniform-discrete 0.5 3)))
    (is (some? (d/uniform-discrete -2 3))))
  (testing "array parameters are checked as host ones are (genclj-ggdl, P2-05)"
    ;; they drew values of density -Inf, or NaN (poisson [-1]); planted
    ;; back, each check host-only (or neg-binom's p > 0, or piecewise's
    ;; vectors, dropped) lets its case through. Clojure data is checked on
    ;; the host on every backend.
    (doseq [[label f] {"neg-binom p = 0" #(d/neg-binom 2.0 0.0) "geometric p = 0" #(d/geometric 0.0)
                       "neg-binom [0]" #(d/neg-binom 2.0 [0.0 0.5])
                       "poisson [-1]" #(d/poisson [-1.0 2.0]) "uniform [1] [0]" #(d/uniform [1.0] [0.0])
                       "uniform-discrete [3] [1]" #(d/uniform-discrete [3] [1]) "binom [2.5]" #(d/binom [2.5] 0.5)
                       "binom [-1]" #(d/binom [-1 2] 0.5) "uniform-discrete [0.5]" #(d/uniform-discrete [0.5] [3])
                       "piecewise-uniform of 2-d parameters" #(d/piecewise-uniform [[0.0 1.0 2.0] [0.0 2.0 4.0]]
                                                                                    [[0.5 0.5] [0.25 0.75]])}]
      (is (thrown? clojure.lang.ExceptionInfo (f)) label))
    (is (every? some? [(d/poisson [0.0 2.0]) (d/uniform [0.0] [1.0]) (d/binom [2 3] 0.5) (d/uniform-discrete [1] [3])
                       (d/neg-binom 2.0 [1.0 0.5]) (d/geometric [1.0 0.5])])
        "valid arrays pass"))
  (testing "a nil parameter is an error at construction, naming both (genclj-khnp, I3-13)"
    ;; (d/normal 0.0 nil) constructed, and its first sample threw "shape: not
    ;; an array: nil" (planted: no nil check)
    (doseq [[expected f] [[{:distribution "normal" :param "std"} #(d/normal 0.0 nil)]
                          [{:distribution "bernoulli" :param "p"} #(d/bernoulli nil)]
                          [{:distribution "categorical" :param "probs"} #(d/categorical nil)]]]
      (is (= expected (ts/thrown-data (f))))))
  (testing "a boolean for a numeric parameter is an error naming both, and so is an infinite count (cross-review 4)"
    ;; since I3-16 gen.array's comparisons took true as 1.0, so bernoulli,
    ;; binom and poisson constructed with true, and the others threw a raw
    ;; ClassCastException; binom's n of Inf passed check-integer (planted:
    ;; no check-numeric; check-integer without isfinite)
    (doseq [[expected f] [[{:distribution "bernoulli" :param "p"} #(d/bernoulli true)]
                          [{:distribution "binom" :param "p"} #(d/binom 3 true)]
                          [{:distribution "poisson" :param "lam"} #(d/poisson true)]
                          [{:distribution "geometric" :param "p"} #(d/geometric true)]
                          [{:distribution "normal" :param "std"} #(d/normal 0.0 true)]
                          [{:distribution "normal" :param "mu"} #(d/normal [true false] 1.0)]]]
      (is (= expected (ts/thrown-data (f)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom ##Inf 0.5)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"n must be an integer" (d/binom [3 ##Inf] 0.5))))
  (testing "a scalar where a family's last axis holds its values (rule 6)"
    ;; (d/categorical 0.5) threw "Axis -1 is out of range for rank 0",
    ;; naming neither (planted: no check-vector in categorical and in
    ;; categorical-logits)
    (doseq [[expected f] [[{:distribution "categorical" :param "probs" :shape []} #(d/categorical 0.5)]
                          [{:distribution "categorical" :param "probs" :shape []} #(d/categorical (array/array 0.5))]
                          [{:distribution "categorical-logits" :param "logits" :shape []} #(d/categorical-logits 0.3)]
                          [{:distribution "dirichlet" :param "alpha" :shape []} #(d/dirichlet 2.0)]]]
      (is (= expected (ts/thrown-data (f)))))))

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
  ;; fixed key, so deterministic; bands of five standard errors: the mean's
  ;; from the true variance, the sample variance's from the sample's own
  ;; fourth central moment, sd ≈ √((m4 - s⁴)/n) (review 4: a 5% band was
  ;; 1.9 sd for a Pareto; families whose eighth moment is infinite have no
  ;; estimable band, so their parameters keep it finite)
  (let [k (rng/key 99) n 100000
        check (fn [dist mean var]
                (let [x (dc/sample dist k [n])
                      c (array/- x (array/mean x))
                      s2 (ts/number (array/mean (array/square c)))
                      m4 (ts/number (array/mean (array/square (array/square c))))]
                  (is (close? (array/mean x) mean (* 5 (Math/sqrt (/ var n)))) (str dist " mean"))
                  (is (close? (array/variance x) var (* 5 (Math/sqrt (/ (max 0.0 (- m4 (* s2 s2))) n)))) (str dist " variance"))))]
    (check (d/normal 0.5 2.0) 0.5 4.0)
    (check (d/uniform -1.0 3.0) 1.0 (/ 16.0 12))
    (check (d/beta 2.5 0.7) (/ 2.5 3.2) (/ (* 2.5 0.7) (* 3.2 3.2 4.2)))
    (check (d/beta 0.3 0.5) 0.375 (/ 0.15 (* 0.64 1.8)))
    (check (d/gamma 2.5 1.5) 3.75 5.625)
    (check (d/inv-gamma 10.0 2.0) (/ 2.0 9) (/ 4.0 (* 81 8)))
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
      (check (d/f-dist 3.0 20.0) (/ 20.0 18) (/ (* 2 400.0 21) (* 3 324 16)))
      (check (d/gumbel 1.0 2.0) (+ 1.0 (* 2.0 0.5772156649015329)) (/ (* Math/PI Math/PI 4.0) 6))
      (check (d/log-normal 0.2 0.5) (Math/exp 0.325) (* (- (Math/exp 0.25) 1) (Math/exp 0.65)))
      (check (d/logistic 0.5 1.5) 0.5 (/ (* Math/PI Math/PI 2.25) 3))
      (check (d/nakagami 1.5 2.0) (* (/ (g 2.0) (g 1.5)) (Math/sqrt (/ 2.0 1.5)))
             (* 2.0 (- 1 (* (/ 1 1.5) (Math/pow (/ (g 2.0) (g 1.5)) 2)))))
      (check (d/pareto 9.0 1.5) (/ (* 9.0 1.5) 8.0) (/ (* 1.5 1.5 9.0) (* 64.0 7.0)))
      (check (d/student-t 10.0) 0.0 1.25)
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

(def mvnormal-mixture
  "Gen.jl's mixture_of_mvnormals, HomogeneousMixture(mvnormal, [1, 2]): means
   of shape [K d], covariances [K d d]."
  (d/homogeneous-mixture d/mvnormal [1 2]))

(defn- isotropic-log-density
  "log N(x; mu, s2·I)."
  [x mu s2]
  (- (* -0.5 (count x) (Math/log (* 2 Math/PI s2)))
     (/ (reduce + (map #(Math/pow (- %1 %2) 2) x mu)) (* 2 s2))))

(deftest mixtures-of-multivariate-families
  ;; F2 (M5 inventory): the family's log density was taken elementwise, the
  ;; value against a component axis it does not have (-4.135 for -4.754
  ;; here); Gen.jl's mixture.jl:138
  (let [w [0.4 0.6] mus [[0.0 0.0] [1.0 1.0]] s2 [2.3 1.0]
        covs (mapv (fn [s] [[s 0.0] [0.0 s]]) s2)
        m (mvnormal-mixture w mus covs)
        ;; log wₖ N(x; μₖ, sₖ²·I) per component, and their log-sum-exp
        lps (fn [x] (mapv #(+ (Math/log %1) (isotropic-log-density x %2 %3)) w mus s2))
        lp (fn [x] (Math/log (reduce + (map #(Math/exp %) (lps x)))))
        x [1.123 -2.123]]
    (is (close? (dc/log-prob m (array/array x)) (lp x) (ts/tol 1e-12)))
    (is (ts/close? (dc/log-prob m (array/array [x [0.0 3.0]])) (array/array [(lp x) (lp [0.0 3.0])]))
        "elementwise over a batch of values")
    (testing "at a trace site, where a value has one component's shape"
      (is (= [2] (dc/value-shape m)))
      (is (close? (:weight (gfi/generate (gen [] (trace :x m)) [] (cm/choicemap :x (array/array x)) (rng/key 1)))
                  (lp x) (ts/tol 1e-12))))
    (testing "gradients: Σₖ rₖ (μₖ - x)/sₖ² for the value, rₖ (x - μₖ)/sₖ² for mean k"
      (let [r (mapv #(Math/exp (- % (lp x))) (lps x))
            dmus (mapv (fn [rk mu s] (mapv #(* rk (/ (- %1 %2) s)) x mu)) r mus s2)]
        (is (ts/close? ((array/grad #(dc/log-prob m %)) (array/array x)) (array/array (mapv - (apply mapv + dmus)))))
        (is (ts/close? ((array/grad #(dc/log-prob (mvnormal-mixture w % covs) (array/array x))) (array/array mus))
                       (array/array dmus)))))))

(dsl/dist half-count
  "Half a poisson count: discrete, of float values."
  [lam]
  (array// (d/poisson lam) 2.0))

(def ^:private vector-normals
  "Gen.jl's HomogeneousMixture(broadcasted_normal, [1, 1]): components of
   vector values, means of shape [K d]."
  (d/homogeneous-mixture d/normal [1 1]))

(deftest homogeneous-mixtures-mix-whole-values
  ;; genclj-e6b0 (the post-M5 review's P2-06): vector-valued components were
  ;; mixed coordinate by coordinate, and dims made no difference: -3.22 for
  ;; (0, 10) between clusters at (0, 0) and (10, 10), where -51.84 is right.
  ;; Planted back (no whole-values), the values fail; summing the batch too
  ;; (value-axes as the whole value shape, the review's first direction, or
  ;; over v's batch axes) fails the batches.
  (let [lp-normal (fn [x mu] (reduce + (map #(- (* -0.5 (Math/pow (- %1 %2) 2)) 0.9189385332046728) x mu)))
        mix (fn [w lps] (Math/log (reduce + (map #(* %1 (Math/exp %2)) w lps))))
        mus [[0.0 0.0] [10.0 10.0]]
        m (vector-normals [0.5 0.5] mus 1.0)]
    (testing "a whole vector, as mvnormal components score it"
      (is (close? (dc/log-prob m (array/array [0.0 10.0])) (mix [0.5 0.5] (map #(lp-normal [0.0 10.0] %) mus)) (ts/tol 1e-12)))
      (is (ts/close? (dc/log-prob m (array/array [0.0 10.0]))
                     (dc/log-prob (mvnormal-mixture [0.5 0.5] mus [[[1.0 0.0] [0.0 1.0]] [[1.0 0.0] [0.0 1.0]]]) (array/array [0.0 10.0])))))
    (testing "a batch of values"
      (let [xs [[0.0 10.0] [1.0 1.0] [9.0 11.0]]]
        (is (ts/close? (dc/log-prob m (array/array xs)) (array/array (mapv (fn [x] (mix [0.5 0.5] (map #(lp-normal x %) mus))) xs))))))
    (testing "a batch of mixtures: weights [B K] and means [B K d]"
      (let [w [[0.3 0.7] [0.9 0.1]]
            ms [[[0.0 0.0] [1.0 2.0]] [[3.0 3.0] [0.0 -1.0]]]
            x [0.5 1.5]]
        (is (ts/close? (dc/log-prob (vector-normals w ms 1.0) (array/array x))
                       (array/array (mapv (fn [wb mb] (mix wb (map #(lp-normal x %) mb))) w ms))))))
    (testing "a batch of scalar mixtures: weights [B K] and means [B K], dims [0 0] (the verifier's)"
      (let [w [[0.3 0.7] [0.5 0.5]] ms [[0.0 1.0] [2.0 3.0]]]
        (is (ts/close? (dc/log-prob ((d/homogeneous-mixture d/normal [0 0]) w ms 1.0) (array/array [0.5 0.5]))
                       (array/array (mapv (fn [wb mb] (mix wb (map #(lp-normal [0.5] [%]) mb))) w ms))))))))

(deftest mixtures-of-discrete-and-continuous-families-are-errors
  ;; genclj-e6b0 (P2-02): the density added a pmf to a pdf, so given y = 0
  ;; P(w = 0.8 | y = 0) came out 0.267 where 0.80 is exact (planted: no
  ;; check, and both mixtures are made)
  (doseq [[label make] {"poisson and normal" #((d/heterogeneous-mixture [d/poisson d/normal]) [0.5 0.5] [3.0] [0.0 1.0])
                        "spike and slab" #((d/heterogeneous-mixture [d/uniform-discrete d/normal]) [0.3 0.7] [0 0] [0.0 1.0])}]
    (is (= {:distribution "heterogeneous-mixture" :discrete [true false]} (ts/thrown-data (make))) label)))

(deftest mixture-draws-pick-one-component-per-element
  ;; one pick per element of the log density: a vector value is one
  ;; component's whole draw (the pick met the vector's own axis: a shape
  ;; error, or coordinates of different components), and the elements of a
  ;; batched parameter or of batched weights pick and draw independently
  ;; (they shared one pick, or one draw); bands of five standard errors
  (let [n 4000
        band (fn [p] (* 5 (Math/sqrt (/ (* p (- 1 p)) n))))
        frac (fn [pred xs] (/ (count (filter pred xs)) (double n)))]
    ;; [batch shape]: mvnormal's batch axes lead its values; an elementwise
    ;; family's batch shape broadcasts with its values, as normal's own does
    (doseq [[label far batch] [["vector values" (mvnormal-mixture [0.3 0.7] [[0.0 0.0] [100.0 -100.0]]
                                                                  [[[1.0 0.0] [0.0 1.0]] [[1.0 0.0] [0.0 1.0]]])
                                [n]]
                               ;; genclj-e6b0: one pick per coordinate (planted: the old pick)
                               ["vector values of an elementwise family"
                                (vector-normals [0.3 0.7] [[0.0 0.0] [100.0 -100.0]] 1.0) [n 1]]]]
      (testing label
        (let [xs (ts/host (dc/sample far (rng/key 21) batch))]
          (is (every? (fn [[a b]] (= (> a 50) (< b -50))) xs) "both coordinates from one component")
          (is (close? (frac (fn [[a _]] (> a 50)) xs) 0.7 (band 0.7))))))
    (testing "a batched parameter: eight mixtures, which all agree with probability 2⁻⁷"
      (let [m (normal-mixture [0.5 0.5] (array/stack [(array/zeros [8]) (array/full [8] 100.0)] -1) [0.1 0.1])
            xs (ts/host (dc/sample m (rng/key 22) [n 1]))]
        (is (close? (frac #(apply = (map (fn [v] (> v 50)) %)) xs) (/ 1.0 128) (band (/ 1.0 128))))))
    (testing "batched weights: three mixtures, the first two of the first component only"
      (let [m (normal-mixture (array/array [[1.0 0.0] [1.0 0.0] [0.5 0.5]]) [0.0 100.0] [1.0 1.0])
            xs (ts/host (dc/sample m (rng/key 23) [n 1]))]
        ;; independent standard normals: E[x y] = 0, with variance 1
        (is (close? (/ (reduce + (map (fn [[a b]] (* a b)) xs)) n) 0.0 (/ 5 (Math/sqrt n))))
        (is (close? (frac #(> (nth % 2) 50) xs) 0.5 (band 0.5)))))))

(deftest homogeneous-mixtures-check-their-components
  (let [covs [[[1.0 0.0] [0.0 1.0]] [[1.0 0.0] [0.0 1.0]]]
        e (try (mvnormal-mixture [0.5 0.5] [[0.0 0.0] [1.0 1.0] [2.0 2.0]] covs) nil
               (catch clojure.lang.ExceptionInfo e e))]
    (is (re-find #"3 components" (str (ex-message e))))
    (is (= {:distribution "homogeneous-mixture" :shape [3 2] :dim 1 :components 2} (ex-data e)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one dim per parameter"
                          ((d/homogeneous-mixture d/mvnormal [1]) [0.5 0.5] [[0.0 0.0] [1.0 1.0]] covs)))
    (is (ts/close? (dc/log-prob (normal-mixture [0.3 0.7] [-2.0 1.0] [1.0 1.0]) 0.5)
                   (dc/log-prob (normal-mixture [0.3 0.7] [-2.0 1.0] 1.0) 0.5))
        "a parameter without a component axis is every component's")))

(def mvnormal-mixture-model
  "A mixture of mvnormals whose second mean is a choice."
  (gen []
    (let [m (trace :m (d/normal 0.0 1.0))]
      (trace :x (mvnormal-mixture [0.4 0.6] (array/stack [(array/zeros [2]) (array/+ m (array/ones [2]))])
                                  [[[2.3 0.0] [0.0 2.3]] [[1.0 0.0] [0.0 1.0]]])))))

(laws/deflaws mvnormal-mixture [[:model mvnormal-mixture-model [] []]] 100)

(deftest mixtures-of-multivariate-families-batch
  ;; one run for every particle, each particle's score its scalar run's
  (let [ps (binding [batched/*spot-checks* :all] (batched/simulate mvnormal-mixture-model [] 8 (rng/key 31)))]
    (is (nil? (batched/fallback-reason ps)))))

(deftest review-regressions
  (testing "product sums each factor over its own value (review 4)"
    (let [p ((d/product d/normal d/bernoulli) [(array/zeros [3]) 1.0] [0.3])]
      (is (close? (array/item (dc/log-prob p [(array/zeros [3]) true]))
                  (+ (* 3 (array/item (dc/log-prob (d/normal 0 1) 0.0))) (Math/log 0.3)) (ts/tol 1e-12)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one element per factor" (dc/log-prob p [(array/zeros [3])])))))
  (testing "heterogeneous mixtures keep each component's values (review 14)"
    ;; two discrete families, of integer and of float values (a mixture of
    ;; poisson and normal is an error since genclj-e6b0)
    (let [x (dc/sample ((d/heterogeneous-mixture [d/poisson half-count]) [0.5 0.5] [3.0] [3.0]) (rng/key 1) [200])]
      (is (some #(not= % (Math/floor %)) (array/->clj x)) "half counts are not truncated to integers")))
  (testing "bad parameters are errors, not NaN (review 15)"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive definite" (d/mvnormal [0.0 0.0] [[1.0 2.0] [2.0 1.0]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"symmetric" (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.3 1.0]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive" (d/dirichlet [-1.0 2.0])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"increasing" (d/piecewise-uniform [0 2 1] [0.5 0.5])))))

(deftest mvnormal-rejects-singular-covariances
  ;; F1 (M5 inventory): a singular matrix can have a Cholesky factor through
  ;; rounding, which the old check (no NaN in the factor) passed: Gen.jl's
  ;; 2·ones(2, 2) (dist_dsl.jl:29; log-prob 15.24). Rows u, v and u + v
  ;; have a last pivot 2.2 times n·ε·aⱼⱼ, which no bound on the pivots alone
  ;; sees. Both stay rejected at the bound 1/ε: their trace(H⁻¹) is 2.0/ε
  ;; and 2.15/ε, as rounding leaves a singular matrix.
  (doseq [cov [[[2.0 2.0] [2.0 2.0]] [[8.0 -6.0 2.0] [-6.0 5.0 -1.0] [2.0 -1.0 1.0]]]]
    (let [e (try (d/mvnormal (vec (repeat (count cov) 0.0)) cov) nil (catch clojure.lang.ExceptionInfo e e))]
      (is (re-find #"positive definite" (str (ex-message e))) (str cov))
      (is (= {:distribution "mvnormal" :param "cov" :dtype (ts/float-dtype)} (dissoc (ex-data e) :suggested-jitter))
          (str cov))))
  (testing "covariances singular at float32 precision only"
    ;; 1 - 2⁻²⁶ is 1 in float32 (a zero pivot); 1 - 2⁻²⁴ is a float32 whose
    ;; trace(H⁻¹)·ε is 2. In float64 both give the closed form, within κ·ε.
    (doseq [rho [(- 1.0 (Math/pow 2 -26)) (- 1.0 (Math/pow 2 -24))]]
      (let [cov [[1.0 rho] [rho 1.0]]]
        (if (ts/f32?)
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"float32" (d/mvnormal [0.0 0.0] cov)) (str rho))
          (is (close? (dc/log-prob (d/mvnormal [0.0 0.0] cov) (array/array [1.0 1.0]))
                      (- (- (Math/log (* 2 Math/PI))) (* 0.5 (Math/log (* (- 1 rho) (+ 1 rho)))) (/ 1 (+ 1 rho)))
                      (* (/ (+ 1 rho) (- 1 rho)) (Math/ulp 1.0)))
              (str rho)))
        (when (ts/jvm?)
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"float32"
                                (d/mvnormal [0.0 0.0] (array/astype (array/array cov) :float32)))
              (str rho)))))))

(deftest mvnormal-suggests-a-jitter
  ;; the owner's decision: the bound holds at every precision, so in
  ;; float32 a Gaussian-process kernel needs a jitter at float32's scale.
  ;; When the factor exists, the error names the smallest 1, 2 or 5 × 10ᵏ
  ;; that passes once added to the diagonal.
  (let [n 50
        xs (mapv #(/ (double %) (dec n)) (range n))
        ;; squared exponential, length scale 0.1 on [0, 1], 1e-6 on the diagonal
        kernel (vec (for [a xs] (vec (for [b xs] (+ (Math/exp (* -50.0 (Math/pow (- a b) 2))) (if (= a b) 1e-6 0.0))))))
        mu (vec (repeat n 0.0))]
    (when-not (ts/f32?)
      (is (some? (d/mvnormal mu kernel)) "the same kernel passes in float64"))
    ;; float32: host data on a float32 backend, a float32 array on the JVM's
    (when (or (ts/f32?) (ts/jvm?))
      (let [cov (if (ts/f32?) kernel (array/astype (array/array kernel) :float32))
            plus (fn [jitter] (if (ts/f32?)
                                (vec (map-indexed (fn [i row] (update row i + jitter)) kernel))
                                (array/+ cov (array/* jitter (array/eye n {:dtype :float32})))))
            e (try (d/mvnormal mu cov) nil (catch clojure.lang.ExceptionInfo e e))
            jitter (:suggested-jitter (ex-data e))]
        (is (= :float32 (:dtype (ex-data e))))
        (is (number? jitter))
        (is (re-find (re-pattern (str "jitter of " jitter)) (str (ex-message e))))
        (is (some? (d/mvnormal mu (plus jitter))) "adding it passes")
        (is (thrown? clojure.lang.ExceptionInfo (d/mvnormal mu (plus (* 0.4 jitter))))
            "0.4 times it, at most the next smaller suggestion, does not"))))
  (testing "none for a factor that fails, also when a jitter would mend it"
    (let [e (try (d/mvnormal [0.0 0.0] [[1.0 1.0001] [1.0001 1.0]]) nil (catch clojure.lang.ExceptionInfo e e))]
      (is (= {:distribution "mvnormal" :param "cov" :dtype (ts/float-dtype)} (ex-data e))))))

(deftest ^:slow mvnormal-takes-a-1000-point-kernel-at-float32-accuracy
  ;; the bound's calibration (docs/research/covariance/f32_gp.py: a float32
  ;; log density's error is (0.05–0.4)·ε·trace(H⁻¹)). The squared-exponential
  ;; kernel of 1,000 points (variance 1, length scale 0.1 on [0, 1]) passes
  ;; in float32 with a jitter of 1e-3 (ε·trace(H⁻¹) = 0.12) and fails with
  ;; 1e-5 (11.7); in float64, 1e-10 passes (0.002)
  (let [n 1000
        jvm (array/backend-named :jvm)
        kernel (fn [jitter]
                 (array/with-backend jvm
                   (let [x (array/linspace 0.0 1.0 n)
                         d (array/- (array/expand-dims x -1) (array/expand-dims x 0))]
                     (array/+ (array/exp (array/* -0.5 (array/square (array/* 10.0 d)))) (array/* jitter (array/eye n))))))
        mu (vec (repeat n 0.0))
        ;; float32: host data on a float32 backend, a float32 array on the JVM's
        float32 (fn [k] (if (ts/f32?) (array/->clj k) (array/astype k :float32)))]
    (when-not (ts/f32?)
      (is (some? (d/mvnormal mu (array/->clj (kernel 1e-10)))) "1e-10 passes in float64"))
    (when (or (ts/f32?) (ts/jvm?))
      (is (some? (d/mvnormal mu (float32 (kernel 1e-3)))) "1e-3 passes in float32")
      (let [e (try (d/mvnormal mu (float32 (kernel 1e-5))) nil (catch clojure.lang.ExceptionInfo e e))
            jitter (:suggested-jitter (ex-data e))]
        (is (= :float32 (:dtype (ex-data e))) "1e-5 does not")
        (is (<= 1e-4 jitter 1e-3) "the suggested jitter is near 1e-3")
        (is (some? (d/mvnormal mu (float32 (kernel (+ 1e-5 jitter))))) "and adding it passes")))
    (when (ts/f32?)
      ;; what the bound stands for: within 0.1 nats of float64. Here
      ;; ε·trace(H⁻¹) = 0.12, so the script's largest ratio, 0.4, predicts at
      ;; most 0.05 nats (it measured 0.009)
      (let [k (kernel 1e-3)
            exact (fn [y] (array/with-backend jvm (dc/log-prob (d/mvnormal (array/zeros [n]) k) y)))
            y (array/with-backend jvm (dc/sample (d/mvnormal (array/zeros [n]) k) (rng/key 41)))]
        (is (close? (dc/log-prob (d/mvnormal mu (array/->clj k)) (array/array (array/->clj y))) (exact y) 0.1))))))

(dc/defdist softmax-categorical
  "A user-defined distribution without a value-shape clause."
  [logits]
  (sample [key shape] (rng/categorical key logits shape))
  (log-prob [v] (array/pick (array/log-softmax logits) v)))

(deftest defdist-checks-its-clauses
  ;; genclj-ggdl (I3-06): a clause of another name was dropped silently, so
  ;; a misspelled validate skipped the checks and value-shape the shape's;
  ;; a repeated one kept the last (planted: no check, and each defines)
  (doseq [[problem form] {{:unknown '[valdate]}
                          '(gen.dist.core/defdist v-pos [x] (valdate (gen.dist.core/check-positive "v" "x" x))
                             (sample [k s] 1.0) (log-prob [v] 0.0))
                          {:unknown '[logprob] :missing '[log-prob]}
                          '(gen.dist.core/defdist v-exp [rate] (sample [k s] 1.0) (logprob [v] 0.0))
                          {:repeated '[log-prob]}
                          '(gen.dist.core/defdist v-twice [x] (sample [k s] 1.0) (log-prob [v] 0.0) (log-prob [v] 1.0))
                          {:flag-and-clause '[value-shape]}
                          '(gen.dist.core/defdist ^:elementwise v-both [x] (sample [k s] 1.0) (log-prob [v] 0.0)
                             (value-shape [] []))}]
    (is (= problem (dissoc (ts/thrown-data (try (eval form) (catch clojure.lang.Compiler$CompilerException e (throw (ex-cause e)))))
                           :dist))
        (pr-str form))))

(deftest the-distribution-layer-calls-only-gen-array
  ;; rule 3, genclj-z6iy (I3-11): gen.dist and gen.dist.core called
  ;; gen.array.protocols for a backend's default dtypes and a batch's
  ;; particles; now array/defaults and array/all-particles (planted: the
  ;; protocols required again)
  (doseq [n '[gen.dist gen.dist.core gen.dist.dsl gen.dist.host]]
    (require n)
    (is (not-any? #(or (= 'gen.array.protocols (ns-name %)) (.startsWith (name (ns-name %)) "gen.backend"))
                  (vals (ns-aliases n)))
        (str n))
    (is (not-any? #(.startsWith (.getName ^Class %) "gen.array.protocols") (vals (ns-imports n))) (str n)))
  (is (= {:float (ts/float-dtype) :int (ts/int-dtype)} (array/defaults array/*backend*))))

(deftest every-family-declares-its-value-shape
  ;; genclj-wmug (I3-07): value shapes and discreteness were registered in
  ;; two lists far from the definitions, so a family left out had no shape
  ;; check, silently (review 1's bug 6, review 3's B5). Each of gen.dist's
  ;; families now declares both where it is defined (planted: weibull
  ;; without ^:elementwise, poisson without ^:discrete).
  (let [families (for [[sym v] (ns-publics 'gen.dist)
                       :let [t (keyword "gen.dist" (name sym))]
                       :when (and (::dc/constructor (meta v)) (contains? (methods dc/-sample) t))]
                   t)]
    (is (= 37 (count families)))
    (is (every? #(contains? (methods dc/-value-shape) %) families)
        (str (remove #(contains? (methods dc/-value-shape) %) families)))
    (is (= #{::d/bernoulli ::d/categorical ::d/categorical-logits ::d/uniform-discrete ::d/geometric ::d/poisson
             ::d/binom ::d/neg-binom}
           (set (filter #(contains? (methods dc/-discrete?) %) families))))))

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
  ;; differences of the log density, which run on the JVM backend in
  ;; float64 whatever the backend under test (a float32 one's would be
  ;; noise: review 4)
  (let [h 1e-6
        jvm (array/backend-named :jvm)
        fd (fn [f x] (/ (- (f (+ x h)) (f (- x h))) (* 2 h)))]
    (doseq [[label make params v] further
            i (range -1 (count params))]
      (let [lp (fn [x] (array/with-backend jvm
                         (ts/number (dc/log-prob (apply make (if (neg? i) params (assoc params i x))) (if (neg? i) x v)))))
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

(deftest the-further-families-validate-their-parameters
  ;; host values and the elements of concrete arrays (review 4: Nakagami
  ;; and the triangular checked host values only)
  (doseq [[label f] {"chi-squared" #(d/chi-squared -1.0) "chi-squared array" #(d/chi-squared [1.0 -1.0])
                     "f-dist" #(d/f-dist 3.0 0.0) "gumbel" #(d/gumbel 0.0 -1.0) "levy" #(d/levy 0.0 0.0)
                     "log-normal" #(d/log-normal 0.0 -0.5) "logistic" #(d/logistic 0.0 0.0)
                     "nakagami" #(d/nakagami -1.0 1.0) "nakagami array" #(d/nakagami [-1.0 2.0] 1.0)
                     "nakagami omega" #(d/nakagami 1.0 [1.0 0.0]) "pareto" #(d/pareto -1.0 1.0)
                     "student-t" #(d/student-t 0.0) "weibull" #(d/weibull 1.0 -2.0)
                     "triangular" #(d/triangular 1.0 0.0 0.5) "triangular mode" #(d/triangular 0.0 1.0 2.0)
                     "triangular arrays" #(d/triangular [0.0 1.0] [1.0 0.0] [0.5 0.5])
                     "triangular array mode" #(d/triangular [0.0] [1.0] [1.5])}]
    (is (thrown? clojure.lang.ExceptionInfo (f)) label)))

(deftest the-further-families-at-their-edges
  ;; review 4: draws that underflowed to 0 (an infinite density) or made a
  ;; quotient infinite, the F density at infinity, the t's far tail
  (let [n 100000
        host (fn [x] (map ts/number (array/->clj x)))
        finite? (fn [xs] (every? #(Double/isFinite %) xs))]
    (testing "f-dist with a small denominator: finite draws, finite densities, -Inf at Inf"
      (let [f (d/f-dist 4.0 0.02)
            x (dc/sample f (rng/key 90) [n])]
        (is (finite? (host x)))
        (is (finite? (host (dc/log-prob f x))))
        (is (= ##-Inf (ts/number (dc/log-prob f ##Inf))))))
    (testing "small shapes: no zero draws, finite densities"
      ;; except a float32 Weibull's draws beyond the float range (E^100 for
      ;; E > Float/MAX_VALUE^0.01): +Inf, of log density -Inf, as Pareto's,
      ;; at the rate exp(-Float/MAX_VALUE^0.01) ≈ 0.088, within five
      ;; standard errors
      (doseq [[label dist] {"chi-squared 0.02" (d/chi-squared 0.02) "weibull 0.01" (d/weibull 0.01 1.0)
                            "nakagami 0.01" (d/nakagami 0.01 1.0)}]
        (let [x (dc/sample dist (rng/key 91) [n])
              xs (host x)
              lps (host (dc/log-prob dist x))
              overflow? (and (= label "weibull 0.01") (= :float32 (array/dtype x)))]
          (is (every? pos? xs) label)
          (is (every? (fn [[v lp]] (if (Double/isInfinite v) (= ##-Inf lp) (Double/isFinite lp))) (map vector xs lps)) label)
          (if overflow?
            (let [p (Math/exp (- (Math/pow Float/MAX_VALUE 0.01)))
                  k (count (filter #(Double/isInfinite %) xs))]
              (is (< (Math/abs (- k (* n p))) (* 5 (Math/sqrt (* n p (- 1 p))))) (str label ": " k " +Inf draws")))
            (is (finite? xs) label)))))
    (testing "Student's t at nu = 0.1: P(|T| > 1e16), within five standard errors"
      ;; P(|T| > t) ≈ 2 c ν^((ν-1)/2) t^(-ν), c = Γ((ν+1)/2)/(√(νπ) Γ(ν/2)), for large t
      (let [nu 0.1 t 1e16
            g #(Math/exp (ts/number (array/lgamma (double %))))
            p (* 2 (/ (g (/ (+ nu 1) 2)) (* (Math/sqrt (* nu Math/PI)) (g (/ nu 2)))) (Math/pow nu (/ (- nu 1) 2)) (Math/pow t (- nu)))
            f (/ (count (filter #(> (Math/abs (double %)) t) (host (dc/sample (d/student-t nu) (rng/key 92) [n])))) (double n))]
        (is (< (Math/abs (- f p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))) (str f " against " p))))
    (testing "the triangular at its mode, also when the mode is a bound"
      (is (ts/close? (Math/log 2.0) (dc/log-prob (d/triangular 0.0 1.0 1.0) 1.0)))
      (is (ts/close? (Math/log 2.0) (dc/log-prob (d/triangular 0.0 1.0 0.0) 0.0))))
    (testing "a gradient outside the support is finite"
      (is (finite? [(ts/number ((array/grad #(dc/log-prob (d/weibull % 2.0) -1.0)) 1.5))
                    (ts/number ((array/grad #(dc/log-prob (d/weibull 1.5 %) -1.0)) 2.0))])))))

;; ---------------------------------------------------------------------------
;; Gamma, beta and dirichlet at their edges
;; ---------------------------------------------------------------------------

(defn- in-float32
  "The family's parameters as float32 arrays of n elements, so a draw of no
   batch shape is n float32 draws (the JVM backend's 0-d arrays are host
   doubles)."
  [family n & params]
  (apply family (map #(array/full [n] % {:dtype :float32}) params)))

(deftest gamma-beta-and-dirichlet-at-their-edges
  ;; genclj-uli3 (the post-M5 review's P2-01): their draws were not floored,
  ;; as §9 says every family's are. gamma(0.001, 1000) drew 0, of density
  ;; +Inf, about half the time (90 % in float32); beta(0.001, 0.001) drew 0,
  ;; 1 (x/(x + y) rounds to 1 when y < x·2^-53) and NaN (0/0); a sparse
  ;; dirichlet drew exact zeros. Planted back, each fails: the floors, the
  ;; clamp below 1, beta's x/(x + y) of linear draws, the boost's power of U
  ;; (log of 0: NaN where both underflow), the float32 exponential drawn in
  ;; float64 (the draws' dtype) and the boost's E·α for E/α (the law).
  (let [n 2000
        tiny #(if (= :float32 (array/dtype %)) (double Float/MIN_NORMAL) Double/MIN_NORMAL)
        top #(if (= :float32 (array/dtype %)) (double (Math/nextDown (float 1.0))) (Math/nextDown 1.0))
        host (fn [x] (map ts/number (flatten (array/->clj x))))
        check (fn [label dist batch dtype upper]
                (let [x (dc/sample dist (rng/key 93) batch)
                      xs (host x)]
                  (is (= dtype (array/dtype x)) label)
                  (is (every? #(<= (tiny x) % (upper x)) xs) label)
                  (is (every? #(Double/isFinite %) (host (dc/log-prob dist x))) label)
                  xs))
        dtypes (if (ts/f32?) [:float32] [(ts/float-dtype) :float32])]
    (doseq [dtype dtypes
            :let [f32 (and (= :float32 dtype) (not (ts/f32?)))
                  make (fn [family k & params] (if f32 (apply in-float32 family k params) (apply family params)))
                  batch (fn [k] (if f32 [] [k]))]]
      (testing (name dtype)
        (check "gamma(0.001, 1000)" (make d/gamma n 0.001 1000.0) (batch n) dtype (constantly Double/MAX_VALUE))
        (let [xs (check "beta(0.001, 0.001)" (make d/beta n 0.001 0.001) (batch n) dtype top)
              below (count (filter #(< % 0.5) xs))]
          ;; symmetric: half below 1/2, within five standard errors
          (is (< (Math/abs (- below (/ n 2))) (* 5 (Math/sqrt (/ n 4)))) (str below " of " n " below 1/2")))
        (let [k 1000
              d (if f32 (d/dirichlet (array/full [k] 0.01 {:dtype :float32})) (d/dirichlet (array/full [k] 0.01)))
              x (dc/sample d (rng/key 94) [200])]
          (is (= dtype (array/dtype x)))
          (is (every? #(<= (tiny x) %) (host x)) "no component is 0")
          (is (every? #(Double/isFinite %) (host (dc/log-prob d x)))))))
    (testing "a small shape's law: P(X < 1e-30) = (1e-30)^0.01 for beta(0.01, 1), within five standard errors"
      (let [m 20000
            p (Math/pow 1e-30 0.01)
            below (count (filter #(< % 1e-30) (host (dc/sample (d/beta 0.01 1.0) (rng/key 95) [m]))))]
        (is (< (Math/abs (- (/ below (double m)) p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) m)))) (str below " of " m))))
    (testing "an update that keeps a floored draw has the score change as its weight (it was NaN)"
      ;; g ~ gamma(0.005, 1) underflows with probability 2^(-1075/200) = 0.024
      (let [model (gen [s] (trace :g (d/gamma 0.005 1.0)) (trace :x (d/normal s 1.0)))
            trs (map #(gfi/simulate model [0.0] %) (rng/split (rng/key 96) 400))
            floored (filter #(= (tiny (% :g)) (ts/number (% :g))) trs)]
        (is (seq floored) "some traces hold a floored draw")
        (doseq [tr floored]
          (let [x (ts/number (tr :x))]
            (is (ts/close? (:weight (gfi/update tr [1.0] [diff/unknown-change] cm/EMPTY (rng/key 1)))
                           (- (* -0.5 (Math/pow (- x 1.0) 2)) (* -0.5 x x))))))))))

;; ---------------------------------------------------------------------------
;; Float32 samplers at their edges
;; ---------------------------------------------------------------------------

(deftest float32-samplers-stay-in-their-supports
  ;; genclj-3d2x. P2-08: laplace's low bound, nextafter(-1, 0), was
  ;; float64's, which is -1 in float32, where log1p(-|u|) is -Inf: a +Inf
  ;; draw about once in 2^23 on MLX (planted: the float64 bound for both).
  ;; P2-11 and the verifier's poisson: in float32 an inversion's partial
  ;; sums fall short of 1, binom(37, 0.25)'s by 1.1e-6 and poisson(9.1)'s
  ;; by 3e-7, so a uniform above them counted every later step, 40 > n,
  ;; and 40 for the poisson (planted: u not scaled to their total). At the
  ;; largest float32 below 1 the counts stay in the support.
  (is (< -1.0 (double (float (#'d/above-minus-one :float32)))))
  (is (< -1.0 (#'d/above-minus-one :float64)))
  (let [f32 #(array/array [%] {:dtype :float32})
        top (f32 (double (Math/nextDown (float 1.0))))]
    (doseq [[n p] [[37.0 0.25] [7.0 0.41]]]
      (is (<= (ts/number (#'d/binomial-inversion top (f32 n) (f32 p) (f32 (- 1.0 p)))) n) (str "binom(" n ", " p ")")))
    (is (< (ts/number (#'d/poisson-inversion top (f32 9.1))) 40) "poisson(9.1)")))

;; ---------------------------------------------------------------------------
;; Gradients at zero probabilities and at the parameters' bounds
;; ---------------------------------------------------------------------------

(defn- with-a-zero
  "A probability vector with an exact 0 that varies with b: [b, 1 - b, 0·b]."
  [b]
  (array/stack [b (- 1.0 b) (array/* 0.0 b)]))

(def ^:private at-the-edges
  "[label log-density at step]: a log density of one argument, at a point
   where a probability is 0 or a parameter at its bound, and the side of it
   the finite difference takes (+1 or -1, the side inside the domain)."
  (let [mix (d/homogeneous-mixture d/normal)]
    [["categorical, picking b" #(dc/log-prob (d/categorical (with-a-zero %)) 0) 0.25 1]
     ["categorical, picking 1 - b" #(dc/log-prob (d/categorical (with-a-zero %)) 1) 0.25 1]
     ["categorical-logits, a -Inf logit" #(dc/log-prob (d/categorical-logits (array/stack [% 0.0 ##-Inf])) 0) 0.5 1]
     ["bernoulli true at p = 1" #(dc/log-prob (d/bernoulli %) true) 1.0 -1]
     ["bernoulli false at p = 0" #(dc/log-prob (d/bernoulli %) false) 0.0 1]
     ["a mixture with a weight of 0" #(dc/log-prob (mix (with-a-zero %) [0.0 1.0 2.0] 1.0) 0.3) 0.25 1]
     ["a mixture's weight at 0" #(dc/log-prob (mix (array/stack [% (- 1.0 %)]) [0.0 1.0] 1.0) 0.3) 0.0 1]
     ["a mixture's weight at 1" #(dc/log-prob (mix (array/stack [% (- 1.0 %)]) [0.0 1.0] 1.0) 0.3) 1.0 -1]
     ["beta-uniform at theta = 0" #(dc/log-prob (d/beta-uniform % 2.0 3.0) 0.3) 0.0 1]
     ["beta-uniform at theta = 1" #(dc/log-prob (d/beta-uniform % 2.0 3.0) 0.3) 1.0 -1]
     ["binom at p = 0" #(dc/log-prob (d/binom 5 %) 0) 0.0 1]
     ["binom at p = 1" #(dc/log-prob (d/binom 5 %) 5) 1.0 -1]
     ["neg-binom at p = 1" #(dc/log-prob (d/neg-binom 2.0 %) 0) 1.0 -1]
     ["geometric at p = 1" #(dc/log-prob (d/geometric %) 0) 1.0 -1]
     ["poisson at lambda = 0" #(dc/log-prob (d/poisson %) 0) 0.0 1]]))

(defn- matches-its-finite-difference
  "Does the gradient of lp (a log density of one argument) at x match a
   second-order one-sided finite difference from the side step (+1 or -1),
   on the JVM backend in float64 whatever the backend under test?"
  [label lp at step]
  (let [h 1e-5
        f #(array/with-backend (array/backend-named :jvm) (ts/number (lp %)))
        fd (/ (clojure.core/- (* 4 (f (+ at (* step h)))) (* 3 (f at)) (f (+ at (* 2 step h)))) (* 2 step h))
        g (ts/number ((array/grad lp) (array/array at)))]
    (is (ts/close? fd g {:rtol (ts/tol 1e-6) :atol (ts/tol 1e-8)}) (str label ": " g " against " fd))))

(deftest gradients-at-zero-probabilities-and-bounds
  ;; genclj-rnbd (from memo.clj): categorical took the log of every
  ;; probability before its pick, and log's gradient at a 0 it did not pick,
  ;; 0/0, made the gradient NaN (4.0 is right at b = 0.25). Bernoulli's where
  ;; over log p and log1p(-p), a mixture's log of a weight of 0 and
  ;; beta-uniform's branches had the same trap. Planted back, each gives NaN
  ;; here: (pick-guarded identity (log probs) v), (where v (log p) (log1p
  ;; (- p))), log-weight's (log w) for (log (stand-in …)), and xlogy
  ;; without its stand-in (binom, poisson). Dropping zero-weight-term, in
  ;; log-mixture or in beta-uniform, fails the derivatives at a weight's
  ;; bound (-1 for 0.22); pick-guarded without its stand-in, the gradient
  ;; outside the support.
  ;; Each gradient against a second-order one-sided finite difference, on
  ;; the JVM backend in float64 whatever the backend under test.
  (doseq [[label lp at step] at-the-edges]
    (matches-its-finite-difference label lp at step))
  (testing "outside the support, where the pick is clipped to a 0, the gradient is 0"
    (is (= 0.0 (ts/number ((array/grad #(dc/log-prob (d/categorical (with-a-zero %)) 5)) (array/array 0.25))))))
  (testing "a weight of 0 whose component's density is beyond the float range times the mixture's"
    ;; the slope e^(lp₀ - L) = e^500000: the value stays exact, and the
    ;; gradient is the largest float (planted: without the cap, both NaN)
    (let [lp #(dc/log-prob ((d/homogeneous-mixture d/normal) (array/stack [% (- 1.0 %)]) [0.0 1000.0] 1.0) 0.0)
          [v g] ((array/value-and-grad lp) (array/array 0.0))]
      (is (ts/close? v (dc/log-prob (d/normal 1000.0 1.0) 0.0)))
      (is (= (if (ts/f32?) (double Float/MAX_VALUE) Double/MAX_VALUE) (ts/number g))))))

(dsl/dist coin
  "A bernoulli's draw as 1.0 or 0.0."
  [p]
  (array/where (d/bernoulli p) 1.0 0.0))

(dsl/dist labeled-numbers
  "A categorical's draw picking one of three numbers."
  [probs]
  (array/take (array/array [10.0 20.0 30.0]) (d/categorical probs)))

(dsl/dist binom-labels
  "A binomial's count picking one of four numbers: array labels over an
   enumerable draw."
  [p]
  (array/take (array/array [10.0 20.0 30.0 40.0]) (d/binom 3 p)))

(dsl/dist shifted-count
  "A poisson's count plus one: a draw without a support, inverted."
  [lam]
  (array/+ 1.0 (d/poisson lam)))

(deftest a-log-mass-of-0-has-gradient-0
  ;; DESIGN §9 (PROVISIONAL, genclj-rnbd's follow-up): a family that
  ;; computes a log mass and then selects sent a cotangent of 0 into log at
  ;; a mass of 0, so a pushforward's match over an enumerable draw had a
  ;; NaN gradient (the coin at p = 0 observed 0.0, the [b 1-b 0] labels);
  ;; xlogy's and xlog1py's derivatives at their pole are infinite, so
  ;; binom, poisson, geometric and neg-binom at a parameter's bound with v
  ;; outside the mass had ±Inf, and NaN through array labels over binom.
  ;; Planted back, each fails: categorical's
  ;; log, bernoulli's rnbd branches, log1m-mass without its stand-in, each
  ;; family's stand-in parameter, piecewise-uniform's log.
  (doseq [[label lp at step] [["the coin at p = 0, observed 0.0" #(dc/log-prob (coin %) 0.0) 0.0 1]
                              ["the coin at p = 1, observed 1.0" #(dc/log-prob (coin %) 1.0) 1.0 -1]
                              ["[b 1-b 0]'s labels, observed b's" #(dc/log-prob (labeled-numbers (with-a-zero %)) 10.0) 0.25 1]
                              ["binom's labels at p = 0, observed 0's" #(dc/log-prob (binom-labels %) 10.0) 0.0 1]
                              ["a shifted poisson at lambda = 0, observed 0's" #(dc/log-prob (shifted-count %) 1.0) 0.0 1]]]
    (matches-its-finite-difference label lp at step))
  (testing "a value of mass 0: log density -Inf, gradient 0"
    (doseq [[label lp at] [["the coin at p = 0, observed 1.0" #(dc/log-prob (coin %) 1.0) 0.0]
                           ["[b 1-b 0]'s labels, observed the 0's" #(dc/log-prob (labeled-numbers (with-a-zero %)) 30.0) 0.25]
                           ["binom's labels at p = 0, observed 1's" #(dc/log-prob (binom-labels %) 20.0) 0.0]
                           ["a shifted poisson at lambda = 0, observed 2's" #(dc/log-prob (shifted-count %) 3.0) 0.0]
                           ["binom(5, 0) at 2" #(dc/log-prob (d/binom 5 %) 2) 0.0]
                           ["binom(5, 1) at 3" #(dc/log-prob (d/binom 5 %) 3) 1.0]
                           ["poisson(0) at 2" #(dc/log-prob (d/poisson %) 2) 0.0]
                           ["geometric(1) at 2" #(dc/log-prob (d/geometric %) 2) 1.0]
                           ["neg-binom(2, 1) at 2" #(dc/log-prob (d/neg-binom 2.0 %) 2) 1.0]
                           ["piecewise-uniform's bin of mass 0" #(dc/log-prob (d/piecewise-uniform [0.0 1.0 2.0 3.0] (with-a-zero %)) 2.5)
                            0.25]]]
      (let [[v g] ((array/value-and-grad lp) (array/array at))]
        (is (= [##-Inf 0.0] [(ts/number v) (ts/number g)]) label)))))

(deftest float-limits-by-dtype
  ;; m3-grammar's cross-review: half floats took float64's constants, so
  ;; largest-float's cap was Inf in float16 (a mixture's weight of 0 then
  ;; NaN) and floored's floor 0 (planted: float16 as float64). Each
  ;; against IEEE's definition, from the precision p and the exponent's bits.
  (doseq [[dtype p bits] [[:float64 53 11] [:float32 24 8] [:float16 11 5] [:bfloat16 8 8]]]
    (let [emax (dec (bit-shift-left 1 (dec bits)))]
      (is (= {:tiny (Math/pow 2 (- 1 emax)) :max (* (- 2 (Math/pow 2 (- 1 p))) (Math/pow 2 emax)) :eps (Math/pow 2 (- 1 p))}
             (dc/float-limits dtype))
          (name dtype))))
  (when-not (ts/jvm?)
    (testing "a float16 mixture with a weight of 0"
      (let [h #(array/array % {:dtype :float16})
            m ((d/homogeneous-mixture d/normal) (h [0.5 0.5 0.0]) (h [0.0 1.0 2.0]) (h [1.0 1.0 1.0]))]
        (is (Double/isFinite (ts/number (dc/log-prob m (h 0.3)))))))))

(deftest probabilities-sum-to-1-within-their-dtype
  ;; genclj-ggdl (the verifier's P2-07 point): the bound on |Σp - 1| was a
  ;; fixed 1e-6, float64's. A float32 sum of 10⁴ terms lies further off, so
  ;; dirichlet's own draws scored -Inf at K = 10⁴ on MLX's CPU (before
  ;; genclj-y0mp's blocked sum), and float32 probabilities that normalized
  ;; to 1 were refused. The bound is dc/probability-tolerance, 4·log2(K)·ε,
  ;; and at least 1e-6 (planted: the fixed 1e-6 again, in
  ;; check-probabilities and in dirichlet's support).
  (let [k 10000
        off (fn [delta] (into [(+ (/ 1.0 k) delta)] (repeat (dec k) (/ 1.0 k))))
        jvm32 #(array/with-backend (array/backend-named :jvm) (array/array % {:dtype :float32}))]
    (testing "categorical's probabilities, checked on the host"
      (is (some? (d/categorical (jvm32 (off 3e-6)))) "3e-6 off: within float32's 6.3e-6")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be probabilities" (d/categorical (jvm32 (off 1e-4)))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be probabilities" (d/categorical (off 3e-6)))
          "float64 data keeps its 1e-6"))
    (testing "dirichlet's support in float32, on the backend under test"
      (let [dd (d/dirichlet (array/full [k] 1.0 {:dtype :float32}))
            v #(array/array (off %) {:dtype :float32})]
        (is (Double/isFinite (ts/number (dc/log-prob dd (v 3e-6)))))
        (is (= ##-Inf (ts/number (dc/log-prob dd (v 1e-4)))))))
    (is (= [1e-6 1e-6] [(dc/probability-tolerance 3 :float64) (dc/probability-tolerance k :float64)]))
    (is (ts/close? (* 4 (/ (Math/log k) (Math/log 2)) (Math/ulp (float 1.0))) (dc/probability-tolerance k :float32)))))

;; ---------------------------------------------------------------------------
;; GenMLX's further families: the truncated normal, the circular families,
;; Wishart and inverse Wishart (genclj-z2gh). Their log densities are
;; checked against scipy's in log-probs-match-scipy.
;; ---------------------------------------------------------------------------

(def ^:private s2 [[1.0 0.5] [0.5 2.0]])
(def ^:private s3 [[1.0 0.5 0.1] [0.5 2.0 -0.3] [0.1 -0.3 1.5]])

(deftest the-genmlx-families-validate-their-parameters
  ;; planted: the checks of lo < hi, rho < 1, df > p - 1 and of a square
  ;; scale, each dropped, let their cases through
  (doseq [[expected f] [[{:distribution "truncated-normal" :param "sigma" :value 0.0} #(d/truncated-normal 0.0 0.0 -1.0 1.0)]
                        [{:distribution "von-mises" :param "kappa"} #(d/von-mises 0.0 [1.0 -1.0])]
                        [{:distribution "wrapped-cauchy" :param "rho" :value -0.1} #(d/wrapped-cauchy 0.0 -0.1)]
                        [{:distribution "wrapped-normal" :param "sigma"} #(d/wrapped-normal 0.0 [1.0 -1.0])]
                        [{:distribution "wishart" :param "scale" :shape [2 3]} #(d/wishart 3.0 [[1.0 0.0 0.0] [0.0 1.0 0.0]])]
                        [{:distribution "wishart" :param "scale" :shape [2]} #(d/wishart 3.0 [1.0 1.0])]
                        [{:distribution "inv-wishart" :param "scale"} #(d/inv-wishart 3.0 [[1.0 0.5] [0.3 1.0]])]]]
    (is (= expected (ts/thrown-data (f))) (pr-str expected)))
  (testing "PROVISIONAL edges (genclj-z2gh): loud errors"
    (doseq [[label f] {"lo = hi" #(d/truncated-normal 0.0 1.0 1.0 1.0) "lo > hi" #(d/truncated-normal 0.0 1.0 1.0 0.0)
                       "both bounds -Inf" #(d/truncated-normal 0.0 1.0 ##-Inf ##-Inf)
                       "lo > hi in an array" #(d/truncated-normal 0.0 1.0 [0.0 2.0] [1.0 1.0])
                       "kappa = 0" #(d/von-mises 0.0 0.0) "rho = 1" #(d/wrapped-cauchy 0.0 1.0)
                       "rho = 1 in an array" #(d/wrapped-cauchy 0.0 [0.5 1.0])}]
      (is (thrown? clojure.lang.ExceptionInfo (f)) label))
    (doseq [[label f] {"wishart df = p - 1" #(d/wishart 1.0 s2) "inv-wishart df = p - 1" #(d/inv-wishart 2.0 s3)
                       "wishart df below in an array" #(d/wishart [3.0 0.5] s2)}]
      (is (= {:distribution (if (re-find #"^inv" label) "inv-wishart" "wishart") :param "df" :dimension (if (re-find #"inv" label) 3 2)}
             (ts/thrown-data (f)))
          label))
    (testing "a scale that is not positive definite, with mvnormal's jitter suggestion"
      (is (re-find #"wishart: scale must be positive definite" (ex-message (try (d/wishart 3.0 [[1.0 2.0] [2.0 1.0]]) (catch Exception e e)))))
      (is (= {:distribution "inv-wishart" :param "scale" :dtype (ts/float-dtype)}
             (dissoc (ts/thrown-data (d/inv-wishart 3.0 [[2.0 2.0] [2.0 2.0]])) :suggested-jitter)))))
  (testing "and what is allowed"
    (is (every? some? [(d/truncated-normal 0.0 1.0 ##-Inf ##Inf) (d/truncated-normal 0.0 1.0 -1e-300 1e-300)
                       (d/von-mises 0.0 1e-300) (d/von-mises 0.0 1e300) (d/wrapped-cauchy 0.0 0.0)
                       (d/wrapped-normal 0.0 1e-300) (d/wishart 1.0001 s2) (d/inv-wishart [2.5 3.0] s3)]))))

(defn- circle?
  "Are all of x's elements angles in [-π, π]?"
  [x]
  (array/all-true? (array/logical-and (array/>= x (- Math/PI)) (array/<= x Math/PI))))

(defn- host-erfc [x] (gen.array.special/erfc (double x)))

(defn- normal-mass
  "Φ(b) - Φ(a) on the host, from erfc in the form that does not cancel."
  [a b]
  (cond (>= a 0) (* 0.5 (- (host-erfc (/ a (Math/sqrt 2))) (host-erfc (/ b (Math/sqrt 2)))))
        (<= b 0) (* 0.5 (- (host-erfc (/ (- b) (Math/sqrt 2))) (host-erfc (/ (- a) (Math/sqrt 2)))))
        :else (- 1.0 (* 0.5 (host-erfc (/ b (Math/sqrt 2)))) (* 0.5 (host-erfc (/ (- a) (Math/sqrt 2)))))))

(defn- truncated-normal-moments
  "[mean variance] of normal(mu, sigma) on [lo, hi]: with r(x) = φ(x)/Z, Z
   the mass, μ + σ(r(α) - r(β)) and σ²(1 + α·r(α) - β·r(β) - (r(α) - r(β))²)."
  [mu sigma lo hi]
  (let [a (/ (- lo mu) sigma) b (/ (- hi mu) sigma)
        z (normal-mass a b)
        r (fn [x] (if (Double/isInfinite x) 0.0 (/ (Math/exp (- (* -0.5 x x) 0.9189385332046728)) z)))
        xr (fn [x] (if (Double/isInfinite x) 0.0 (* x (r x))))
        m (- (r a) (r b))]
    [(+ mu (* sigma m)) (* sigma sigma (- (+ 1.0 (xr a)) (xr b) (* m m)))]))

(defn- bessel-i
  "I_order(x) e^-x by its power series, on the host (for x up to 50)."
  [order x]
  (reduce + (for [k (range 300)]
              (Math/exp (- (* (+ (* 2 k) order) (Math/log (/ x 2.0)))
                           (gen.array.special/lgamma (double (inc k))) (gen.array.special/lgamma (double (+ k order 1))) x)))))

(defn- mean-within
  "Is the mean of the draws x (a vector of them) within five standard
   errors of m, for a variance v?"
  [x m v]
  (close? (array/mean x) m (* 5 (Math/sqrt (/ v (array/size x))))))

(deftest the-genmlx-families-sample-their-moments
  ;; fixed keys; bands of five standard errors from the true variances (the
  ;; truncated normal's sample variance from its sample's fourth moment, as
  ;; sampler-moments). Planted, each fails: the von Mises acceptance
  ;; without its log c, the envelope's (1 - ρ)/(1 + ρ) without the 2κ of
  ;; its denominator, the wrapped Cauchy's (1 - ρ)/(1 + ρ) as 1 - ρ, the
  ;; truncated normal's interval never mirrored, Bartlett's diagonal of
  ;; χ²(df) for χ²(df - i), or of its means' roots (only the variances
  ;; fail), inv-wishart's C as A⁻ᵀ·Lᵀ.
  (let [n 100000]
    (testing "the truncated normal: mean and variance"
      (doseq [[mu sigma lo hi :as params] [[0.5 2.0 -1.0 3.0] [0.0 1.0 5.0 6.0] [0.0 1.0 0.0 ##Inf] [1.0 0.5 ##-Inf 0.0]
                                          [0.0 1.0 8.0 ##Inf] [2.0 0.5 -12.0 -11.0] [-1.0 3.0 ##-Inf ##Inf]]]
        (let [[m v] (truncated-normal-moments mu sigma lo hi)
              x (dc/sample (apply d/truncated-normal params) (rng/key 61) [n])
              c (array/- x (array/mean x))
              v-hat (ts/number (array/mean (array/square c)))
              m4 (ts/number (array/mean (array/square (array/square c))))]
          (is (close? (array/mean x) m (* 5 (Math/sqrt (/ v n)))) (str params " mean"))
          (is (close? v-hat v (* 5 (Math/sqrt (/ (- m4 (* v-hat v-hat)) n)))) (str params " variance"))
          (is (array/all-true? (array/logical-and (array/>= x lo) (array/<= x hi))) (str params " within [lo, hi]")))))
    (testing "the circular families: E cos(θ - μ), E cos 2(θ - μ) and E sin(θ - μ)"
      ;; [family mu param m1 m2 m4], mk = E cos k(θ - μ): von Mises Iₖ(κ)/I₀(κ),
      ;; wrapped Cauchy ρᵏ, wrapped normal e^(-k²σ²/2)
      (doseq [[make mu p m1 m2 m4] (concat (for [[mu kappa] [[0.7 0.5] [-2.0 4.0] [3.0 50.0]]]
                                             (into [d/von-mises mu kappa] (map #(/ (bessel-i % kappa) (bessel-i 0 kappa)) [1 2 4])))
                                           (for [[mu rho] [[0.5 0.0] [0.5 0.7] [-3.0 0.99]]]
                                             [d/wrapped-cauchy mu rho rho (* rho rho) (Math/pow rho 4)])
                                           ;; uniform draws from sigma = 9 on (planted: σ·z
                                           ;; wrapped at 1e300 too, all draws 0 or ±π)
                                           (for [[mu sigma] [[0.5 0.3] [-3.0 2.0] [1.0 8.9] [1.0 10.0] [0.3 1e300]]]
                                             (into [d/wrapped-normal mu sigma] (map #(Math/exp (* -0.5 % % sigma sigma)) [1 2 4]))))]
        (let [label (str (:type (make mu p)) " " [mu p])
              x (dc/sample (make mu p) (rng/key 62) [n])
              d (array/- x mu)]
          (is (circle? x) label)
          (is (mean-within (array/cos d) m1 (- (/ (+ 1 m2) 2) (* m1 m1))) (str label " cos"))
          (is (mean-within (array/cos (array/* 2.0 d)) m2 (- (/ (+ 1 m4) 2) (* m2 m2))) (str label " cos 2"))
          (is (mean-within (array/sin d) 0.0 (/ (- 1 m2) 2)) (str label " sin")))))
    (testing "von Mises at a large kappa: κ(1 - cos(θ - μ)) is χ²(1)/2 to O(1/κ), mean 1/2, variance 1/2"
      (doseq [kappa [1e6 1e12]]
        (let [x (dc/sample (d/von-mises 0.25 kappa) (rng/key 63) [n])]
          (is (circle? x))
          (is (mean-within (array/* kappa (array/* 2.0 (array/square (array/sin (array/* 0.5 (array/- x 0.25))))))
                           0.5 0.5)
              (str kappa)))))
    (testing "Wishart and inverse Wishart: each entry's mean and variance"
      ;; Wishart: df·S, variance df(Sᵢⱼ² + SᵢᵢSⱼⱼ); inverse Wishart: Ψ/(ν - p - 1),
      ;; variance ((ν - p + 1)Ψᵢⱼ² + (ν - p - 1)ΨᵢᵢΨⱼⱼ)/((ν - p)(ν - p - 1)²(ν - p - 3)).
      ;; A variance's band needs a finite eighth moment: so the inverse
      ;; Wishart's variances at ν = 30 only
      (let [m 20000
            ;; [mean variance], each of Sᵢⱼ, Sᵢᵢ, Sⱼⱼ and df
            wishart [(fn [sij _ _ df] (* df sij)) (fn [sij sii sjj df] (* df (+ (* sij sij) (* sii sjj))))]
            inverse (fn [nu p]
                      [(fn [sij _ _ _] (/ sij (- nu p 1)))
                       (fn [sij sii sjj _] (/ (+ (* (+ (- nu p) 1) sij sij) (* (- nu p 1) sii sjj))
                                              (* (- nu p) (Math/pow (- nu p 1) 2) (- nu p 3))))])]
        (doseq [[label dist [mean variance] variances?]
                [["wishart(4, S2)" (d/wishart 4.0 s2) wishart true]
                 ["wishart(1.5, S2), small chi-squared shapes" (d/wishart 1.5 s2) wishart true]
                 ["wishart(5.5, S3)" (d/wishart 5.5 s3) wishart true]
                 ["inv-wishart(10, S2)" (d/inv-wishart 10.0 s2) (inverse 10.0 2) false]
                 ["inv-wishart(12, S3)" (d/inv-wishart 12.0 s3) (inverse 12.0 3) false]
                 ["inv-wishart(30, S3)" (d/inv-wishart 30.0 s3) (inverse 30.0 3) true]]]
          (let [s (ts/host (:scale (:params dist)))
                df (ts/number (:df (:params dist)))
                x (dc/sample dist (rng/key 64) [m])
                c (array/- x (array/mean x {:axis 0}))
                [means v-hats m4s] (map #(ts/host (array/mean % {:axis 0})) [x (array/square c) (array/square (array/square c))])]
            (doseq [i (range (count s)) j (range (count s))
                    :let [sij (get-in s [i j]) sii (get-in s [i i]) sjj (get-in s [j j])
                          v (variance sij sii sjj df)
                          v-hat (get-in v-hats [i j])]]
              (is (close? (get-in means [i j]) (mean sij sii sjj df) (* 5 (Math/sqrt (/ v m)))) (str label " mean " [i j]))
              (when variances?
                (is (close? v-hat v (* 5 (Math/sqrt (/ (- (get-in m4s [i j]) (* v-hat v-hat)) m)))) (str label " variance " [i j]))))
            (is (array/all-true? (array/== x (array/transpose x [0 2 1]))) (str label ": exactly symmetric"))))))))

(defn- log-normal-tail
  "log Q(x), the standard normal's mass above x, for x >= 50, by its
   asymptotic series (to x^-8: within 1e-15 there)."
  [x]
  (let [y (/ 1.0 (* x x))]
    (+ (* -0.5 x x) (- (Math/log x)) -0.9189385332046728
       (Math/log (- 1.0 (* y (- 1.0 (* 3.0 y (- 1.0 (* 5.0 y (- 1.0 (* 7.0 y))))))))))))

(deftest truncated-normal-draws-far-into-the-tails
  ;; on [50, 51] and its mirror [-51, -50], where Φ and erf round to 1: a
  ;; naive inverse CDF draws every value at a bound or NaN (planted: the
  ;; draws as √2·erfinv(2(Φ(a) + u·(Φ(b) - Φ(a))) - 1), with erf's Φ, all
  ;; fail). The mean, (φ(a) - φ(b))/Z, and P(X <= 50.01), from the tail's
  ;; asymptotic logs, within five standard errors.
  (let [n 100000
        a 50.0 b 51.0 x 50.01
        lqa (log-normal-tail a)
        lz (+ lqa (Math/log (- 1.0 (Math/exp (- (log-normal-tail b) lqa)))))
        log-phi #(- (* -0.5 % %) 0.9189385332046728)
        mean (- (Math/exp (- (log-phi a) lz)) (Math/exp (- (log-phi b) lz)))
        p (/ (- 1.0 (Math/exp (- (log-normal-tail x) lqa))) (- 1.0 (Math/exp (- (log-normal-tail b) lqa))))]
    (doseq [[label dist sign] [["[50, 51]" (d/truncated-normal 0.0 1.0 a b) 1.0]
                               ["[-51, -50]" (d/truncated-normal 0.0 1.0 (- b) (- a)) -1.0]]]
      (let [z (array/* sign (dc/sample dist (rng/key 65) [n]))
            below (ts/number (array/mean (array/astype (array/<= z x) (ts/float-dtype))))]
        (is (array/all-true? (array/logical-and (array/>= z a) (array/<= z b))) label)
        ;; the variance of a draw is below 1/a², the exponential's
        (is (close? (array/mean z) mean (* 5 (/ 1.0 a (Math/sqrt n)))) label)
        (is (close? below p (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))) (str label ": P(|X| <= 50.01) = " below " for " p))
        (is (every? ts/finite? (ts/host (dc/log-prob dist (array/* sign z))))
            (str label ": finite log densities"))))))

(deftest the-genmlx-families-at-their-edges
  (let [lp (fn [dist v] (ts/number (dc/log-prob dist v)))]
    (testing "a truncated normal with infinite bounds is the normal"
      (doseq [v [-3.0 0.2 5.0]]
        (is (ts/close? (lp (d/normal 0.3 1.5) v) (lp (d/truncated-normal 0.3 1.5 ##-Inf ##Inf) v)))))
    (testing "a narrow interval at 0: -log(hi - lo) at its middle, to O((hi - lo)²) (scipy's 1 - Φ(a) - Φ(-b) is 6.6e-7 off)"
      ;; planted: the mass across 0 as log1p(-(Φ(a) + Φ(-b))) fails, and so
      ;; does taking [0, h] for a tail (b' <= 0 for b' < 0)
      (doseq [[lo hi] [[-1e-10 1e-10] [0.0 1e-10] [-1e-10 0.0]]]
        (is (ts/close? (- (Math/log (- hi lo))) (lp (d/truncated-normal 0.0 1.0 lo hi) (/ (+ lo hi) 2))) (str [lo hi]))))
    (testing "rho = 0 is the uniform on the circle, and a tiny kappa nearly so"
      (doseq [v [-3.0 0.0 2.0]]
        (is (ts/close? (- (Math/log (* 2 Math/PI))) (lp (d/wrapped-cauchy 1.0 0.0) v)))
        (is (ts/close? (- (Math/log (* 2 Math/PI))) (lp (d/von-mises 1.0 1e-20) v)))))
    (when-not (ts/f32?)
      (testing "rho near 1: log((1 + ρ)/(1 - ρ)/2π) at the mode, from (1 - ρ)² + 4ρ sin²(d/2)"
        ;; planted: 1 + ρ² - 2ρ cos d, which cancels to 0 or 2e-16 for 1e-24
        (let [rho (- 1.0 1e-12)]
          (is (ts/close? (- (Math/log (/ (+ 1.0 rho) (- 1.0 rho))) (Math/log (* 2 Math/PI)))
                         (lp (d/wrapped-cauchy 0.5 rho) 0.5))))))
    (testing "a tiny sigma's wrapped normal is the normal, a large one's the uniform"
      (is (ts/close? (lp (d/normal 0.0 0.01) 0.02) (lp (d/wrapped-normal 0.0 0.01) 0.02)))
      (doseq [v [0.0 1e-30]]
        (is (ts/close? (lp (d/normal 0.0 1e-30) v) (lp (d/wrapped-normal 0.0 1e-30) v))))
      (is (ts/close? (- (Math/log (* 2 Math/PI))) (lp (d/wrapped-normal 0.0 1e10) 1.0))))
    (testing "log(I₀(κ)e^-κ), of the von Mises' normalizer, within 1e-14 of the power series around its switch at 20"
      ;; planted: either series cut short (20 terms of the power series, 8
      ;; of the asymptotic one) fails
      (doseq [kappa [0.5 5.0 12.0 17.0 19.9 20.0 20.1 23.0 30.0 45.0]]
        (is (ts/close? (Math/log (bessel-i 0 kappa)) (#'d/log-i0e (array/array kappa))
                       {:rtol (ts/tol 1e-14) :atol (ts/tol 1e-14)})
            (str kappa))))
    (testing "the wrapped normal's two series agree at their switch, and each within 1e-14 near it"
      ;; planted: three terms of the direct sum (|k| <= 1) or one of the
      ;; Fourier series fail
      (doseq [sigma [2.0 2.25 2.5] v [-3.0 0.0 1.5 Math/PI]]
        (let [d (- v 0.3)
              direct (#'d/log-wrapped-normal (array/array d) (array/array sigma))
              exact (Math/log (reduce + (for [k (range -40 41)]
                                          (Math/exp (lp (d/normal 0.0 sigma) (+ (#'d/wrap-angle d) (* 2 Math/PI k)))))))]
          (is (ts/close? exact direct {:rtol (ts/tol 1e-14) :atol (ts/tol 1e-14)}) (str sigma " " v)))))
    (testing "draws on the circle also where an angle is lost to rounding"
      ;; mu = -2.5e16 wraps to -4 (planted: the wrap without its clip)
      (doseq [dist [(d/von-mises -2.5e16 2.0) (d/wrapped-cauchy -2.5e16 0.5) (d/wrapped-normal -2.5e16 0.5)]]
        (is (circle? (dc/sample dist (rng/key 68) [100])) (str (:type dist)))))
    (testing "outside the support: -Inf, with a finite gradient"
      (doseq [[label dist v] [["truncated-normal" (d/truncated-normal 0.0 1.0 -1.0 1.0) 1.5]
                              ["von-mises" (d/von-mises 0.0 2.0) 3.5] ["wrapped-cauchy" (d/wrapped-cauchy 0.0 0.5) -3.5]
                              ["wrapped-normal" (d/wrapped-normal 0.0 0.5) 4.0]]]
        (is (= ##-Inf (lp dist v)) label))
      (doseq [[label f at] [["truncated-normal mu" #(dc/log-prob (d/truncated-normal % 1.0 -1.0 1.0) 1.5) 0.2]
                            ["von-mises kappa" #(dc/log-prob (d/von-mises 0.0 %) 3.5) 2.0]
                            ["wrapped-normal sigma" #(dc/log-prob (d/wrapped-normal 0.0 %) 4.0) 0.5]]]
        (is (ts/finite? ((array/grad f) (array/array at))) label)))
    (testing "matrices outside the support: not symmetric, or not positive definite"
      ;; Julia's isposdef: exact symmetry (planted: the symmetry test
      ;; dropped, the asymmetric value scores its lower triangle's matrix)
      (doseq [make [d/wishart d/inv-wishart]
              [label v] [["asymmetric" [[2.0 0.3] [0.31 3.0]]] ["indefinite" [[1.0 2.0] [2.0 1.0]]] ["singular" [[1.0 1.0] [1.0 1.0]]]]]
        (is (= ##-Inf (lp (make 4.0 s2) (array/array v))) label)
        (is (array/all-true? (array/isfinite ((array/grad #(dc/log-prob (make 4.0 s2) %)) (array/array v))))
            (str label ": a finite gradient")))
      (is (= [##-Inf true] (map ts/host [(array/take (dc/log-prob (d/wishart 4.0 s2) (array/array [[[1.0 2.0] [2.0 1.0]] [[2.0 0.3] [0.3 3.0]]])) 0)
                                         (array/isfinite (array/take (dc/log-prob (d/wishart 4.0 s2) (array/array [[[1.0 2.0] [2.0 1.0]] [[2.0 0.3] [0.3 3.0]]])) 1))]))
          "a batch of values, one of them indefinite"))
    (testing "matrix value shapes: df's and scale's batch axes, then [p p]"
      (is (= [2 2] (dc/value-shape (d/wishart 3.0 s2))))
      (is (= [4 2 2] (dc/value-shape (d/wishart [3.0 4.0 5.0 6.0] s2))))
      (is (= [3 3 3] (dc/value-shape (d/inv-wishart 4.0 [s3 s3 s3]))))
      (is (= [5 4 2 2] (array/shape (dc/sample (d/wishart [3.0 4.0 5.0 6.0] s2) (rng/key 1) [5 1]))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape"
                            (gfi/generate (gen [] (trace :w (d/wishart 3.0 s2))) [] (cm/choicemap :w (array/eye 3)) (rng/key 1)))))))

(def ^:private genmlx-families
  "[name constructor params value]: each family at an interior value, the
   truncated normal also in a tail and with an infinite bound."
  [["truncated-normal" d/truncated-normal [0.3 1.5 -1.0 2.0] 0.5]
   ["truncated-normal in a tail" d/truncated-normal [0.0 1.0 5.0 6.0] 5.3]
   ["truncated-normal above an infinite lo" d/truncated-normal [0.3 1.5 ##-Inf 2.0] 0.5]
   ["truncated-normal below an infinite hi" d/truncated-normal [0.3 1.5 1.0 ##Inf] 1.5]
   ["von-mises" d/von-mises [0.3 2.0] 1.0] ["von-mises at a large kappa" d/von-mises [0.3 50.0] 0.4]
   ["wrapped-cauchy" d/wrapped-cauchy [0.3 0.6] -2.0]
   ["wrapped-normal" d/wrapped-normal [0.3 0.7] 2.5] ["wrapped-normal of a wide sigma" d/wrapped-normal [0.3 3.0] 2.5]])

(deftest gradients-of-the-genmlx-families
  ;; with respect to the value and each finite parameter, against central
  ;; finite differences on the JVM backend in float64, as
  ;; gradients-of-the-further-families. Planted: standardized without its
  ;; stand-in (∞·0: NaN for sigma beside an infinite bound), log1mexp's
  ;; near branch without its stand-in (NaN), log-i0e's series without its
  ;; stand-in (NaN at kappa 50) each fail
  (let [h 1e-6
        jvm (array/backend-named :jvm)
        fd (fn [f x] (/ (- (f (+ x h)) (f (- x h))) (* 2 h)))]
    (doseq [[label make params v] genmlx-families
            i (range -1 (count params))
            :let [at (if (neg? i) v (params i))]
            :when (Double/isFinite at)]
      (let [lp (fn [x] (array/with-backend jvm
                         (ts/number (dc/log-prob (apply make (if (neg? i) params (assoc params i x))) (if (neg? i) x v)))))
            g (if (neg? i)
                (array/grad #(dc/log-prob (apply make params) %))
                (array/grad #(dc/log-prob (apply make (assoc params i %)) v)))]
        (is (ts/close? (fd lp at) (ts/number (g (array/array at))) {:rtol (ts/tol 1e-5) :atol (ts/tol 1e-6)})
            (str label (if (neg? i) " d/dv" (str " d/dparam" i)))))))
  (testing "at rho = 0, the wrapped Cauchy's bound: a one-sided difference"
    (matches-its-finite-difference "wrapped-cauchy d/drho" #(dc/log-prob (d/wrapped-cauchy 0.3 %) -2.0) 0.0 1)))

(deftest gradients-of-the-matrix-families
  ;; the tape's Cholesky gives a symmetric matrix's gradient on its lower
  ;; triangle: entry (i, j), i > j, is the derivative along Eᵢⱼ + Eⱼᵢ, the
  ;; symmetric perturbation a finite difference takes here (on the JVM
  ;; backend in float64), and the diagonal along Eᵢᵢ. Also df's. Planted:
  ;; the trace term as tr(S⁻¹X) of S's lower triangle alone fails
  (let [h 1e-6
        jvm (array/backend-named :jvm)
        x3 [[2.0 0.3 -0.2] [0.3 3.0 0.4] [-0.2 0.4 1.0]]
        bump (fn [m i j e] (cond-> (update-in m [i j] + e) (not= i j) (update-in [j i] + e)))]
    (doseq [[label make] [["wishart" d/wishart] ["inv-wishart" d/inv-wishart]]]
      (let [lp (fn [df s x] (array/with-backend jvm (ts/number (dc/log-prob (make df s) (array/array x)))))
            gx (ts/host ((array/grad #(dc/log-prob (make 5.0 s3) %)) (array/array x3)))
            gs (ts/host ((array/grad #(dc/log-prob (make 5.0 %) (array/array x3))) (array/array s3)))
            close (fn [fd g what] (is (ts/close? fd g {:rtol (ts/tol 1e-5) :atol (ts/tol 1e-6)}) (str label " " what)))]
        (close (/ (- (lp (+ 5.0 h) s3 x3) (lp (- 5.0 h) s3 x3)) (* 2 h))
               (ts/number ((array/grad #(dc/log-prob (make % s3) (array/array x3))) (array/array 5.0)))
               "d/ddf")
        (doseq [i (range 3) j (range 3)]
          (if (< i j)
            (is (= 0.0 (get-in gx [i j]) (get-in gs [i j])) (str label " above the diagonal " [i j]))
            (do (close (/ (- (lp 5.0 s3 (bump x3 i j h)) (lp 5.0 s3 (bump x3 i j (- h)))) (* 2 h)) (get-in gx [i j]) (str "d/dx" [i j]))
                (close (/ (- (lp 5.0 (bump s3 i j h) x3) (lp 5.0 (bump s3 i j (- h)) x3)) (* 2 h)) (get-in gs [i j]) (str "d/dscale" [i j])))))))))

(def genmlx-families-model
  "A model of every one of GenMLX's further families, the matrix ones with
   a df that is a choice."
  (gen []
    (let [mu (trace :mu (d/normal 0.0 1.0))
          x (trace :x (d/truncated-normal mu 1.0 0.0 ##Inf))
          t (trace :t (d/von-mises mu 4.0))
          c (trace :c (d/wrapped-cauchy t 0.5))
          w (trace :w (d/wrapped-normal c 0.8))
          df (trace :df (d/uniform 3.0 6.0))
          s (trace :s (d/wishart df [[1.0 0.3] [0.3 1.0]]))]
      (trace :v (d/inv-wishart (array/+ df 1.0) s))
      (array/+ x w))))

(laws/deflaws genmlx-families [[:model genmlx-families-model [] []]] 100)

(deftest the-genmlx-families-batch
  ;; one run for every particle, each particle's score its scalar run's
  (let [ps (binding [batched/*spot-checks* :all] (batched/simulate genmlx-families-model [] 8 (rng/key 66)))]
    (is (nil? (batched/fallback-reason ps)))))

(deftest the-samplers-helpers
  ;; the inversion of log Φ, a truncated normal draw's: in float64 log Φ of
  ;; its x is y within a few ulps of y's magnitude, over y in [-1e6, log 1/2]
  ;; (planted: one Newton step, or two, fail), and in float32, what MLX's
  ;; runs, x is float64's within float32's precision (planted: its first
  ;; guess by erfinv down to y = -20, where 2e^y - 1 rounds to -1 in
  ;; float32, gives -Inf). And von Mises draws in float32 whose scale is
  ;; 1/√κ at any kappa (planted: acos(f) of f = (1 + r·z)/(r + z), Best and
  ;; Fisher's own form, gives the mode at kappa 1e10 and beyond)
  (let [jvm (array/backend-named :jvm)
        ys (into [(- (Math/log 2.0))] (map #(- (Math/pow 10.0 %))) (range -0.15 6.0 0.01))
        x64 (array/with-backend jvm (#'d/ndtri-left (array/array ys)))
        x32 (ts/host (#'d/ndtri-left (array/array ys {:dtype :float32})))]
    (is (every? true? (map #(<= (Math/abs (- %1 %2)) (* 8 (Math/ulp (double %2))))
                           (array/with-backend jvm (array/->clj (#'d/log-ndtr x64))) ys)))
    (is (every? true? (map #(<= (Math/abs (- %1 %2)) (* 1e-6 (max 1.0 (Math/abs %2)))) x32 (array/->clj x64)))))
  (doseq [kappa [1e-3 2.0 1e4 1e10 1e30]]
    (let [n 20000
          x (#'d/von-mises-angle (rng/key 67) (array/full [n] kappa {:dtype :float32}) [n] {:dtype :float32})
          ;; E[2κ(1 - cos θ)], 1 at a large kappa, 2κ(1 - I₁/I₀) below
          m (if (< kappa 10) (* 2 kappa (- 1 (/ (bessel-i 1 kappa) (bessel-i 0 kappa)))) 1.0)]
      (is (= :float32 (array/dtype x)))
      (is (circle? x))
      ;; κ·2sin²(θ/2) is about χ²(1)/2 at a large kappa (variance 1/2); at a
      ;; small one its variance is below 2κ², times 4
      (is (mean-within (array/* (* 2.0 kappa) (array/* 2.0 (array/square (array/sin (array/* 0.5 x)))))
                       m (if (< kappa 10) (* 16 kappa kappa) 2.0))
          (str kappa)))))
