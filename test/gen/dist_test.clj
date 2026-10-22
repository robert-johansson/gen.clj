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
  (let [h 1e-5
        jvm (array/backend-named :jvm)]
    (doseq [[label lp at step] at-the-edges]
      (let [f #(array/with-backend jvm (ts/number (lp %)))
            fd (/ (clojure.core/- (* 4 (f (+ at (* step h)))) (* 3 (f at)) (f (+ at (* 2 step h)))) (* 2 step h))
            g (ts/number ((array/grad lp) (array/array at)))]
        (is (ts/close? fd g {:rtol (ts/tol 1e-6) :atol (ts/tol 1e-8)}) (str label ": " g " against " fd)))))
  (testing "outside the support, where the pick is clipped to a 0, the gradient is 0"
    (is (= 0.0 (ts/number ((array/grad #(dc/log-prob (d/categorical (with-a-zero %)) 5)) (array/array 0.25))))))
  (testing "a weight of 0 whose component's density is beyond the float range times the mixture's"
    ;; the slope e^(lp₀ - L) = e^500000: the value stays exact, and the
    ;; gradient is the largest float (planted: without the cap, both NaN)
    (let [lp #(dc/log-prob ((d/homogeneous-mixture d/normal) (array/stack [% (- 1.0 %)]) [0.0 1000.0] 1.0) 0.0)
          [v g] ((array/value-and-grad lp) (array/array 0.0))]
      (is (ts/close? v (dc/log-prob (d/normal 1000.0 1.0) 0.0)))
      (is (= (if (ts/f32?) (double Float/MAX_VALUE) Double/MAX_VALUE) (ts/number g))))))
