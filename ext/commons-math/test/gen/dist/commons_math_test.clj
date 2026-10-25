(ns gen.dist.commons-math-test
  "Commons Math's distributions as gen.clj's (gen.dist.commons-math): the
   discrete families' masses against closed forms and their means, finite
   supports, the GFI laws on a model over them; and gen.dist's native
   Distributions.jl families against Commons Math's own, wrapped with real,
   two independent implementations (log densities, and the means of draws
   from each). Run with clojure -X:test:commons-math."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.dist.commons-math :as cm]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [org.apache.commons.math3.distribution
            ChiSquaredDistribution FDistribution GumbelDistribution LevyDistribution LogNormalDistribution
            LogisticDistribution NakagamiDistribution ParetoDistribution TDistribution TriangularDistribution
            WeibullDistribution]
           [org.apache.commons.math3.random RandomGenerator]))

(defn- lgamma ^double [x] (ts/number (array/lgamma (double x))))
(defn- log-choose [n k] (- (lgamma (inc n)) (lgamma (inc k)) (lgamma (inc (- n k)))))
(defn- h [n s] (reduce + (map #(Math/pow % (- s)) (range 1 (inc n)))))

(def discrete
  "[name distribution points closed-form-log-mass mean variance]"
  [["hypergeometric" (cm/hypergeometric 5 7 4) [0 2 4]
    (fn [k] (- (+ (log-choose 5 k) (log-choose 7 (- 4 k))) (log-choose 12 4)))
    (/ 20 12.0) (* 4 (/ 5 12.0) (/ 7 12.0) (/ 8 11.0))]
   ["zipf" (cm/zipf 10 1.2) [1 3 10]
    (fn [k] (- (* -1.2 (Math/log k)) (Math/log (h 10 1.2))))
    (/ (h 10 0.2) (h 10 1.2)) (- (/ (h 10 -0.8) (h 10 1.2)) (Math/pow (/ (h 10 0.2) (h 10 1.2)) 2))]])

(deftest discrete-masses-and-means
  (doseq [[[name dist points log-mass mean variance] key] (map vector discrete (rng/split (rng/key 1) 2))]
    (doseq [x points]
      (is (ts/close? (log-mass x) (dc/log-prob dist x) {:rtol 1e-9 :atol 1e-12}) (str name " at " x)))
    (let [n 4000
          xs (map ts/number (array/->clj (dc/sample dist key [n])))
          m (/ (reduce + xs) n)]
      (is (< (Math/abs (- m mean)) (* 5 (Math/sqrt (/ variance n)))) (str name ": " m " against " mean)))))

(deftest supports
  (is (= [0 1 2 3 4] (dc/support (cm/hypergeometric 5 7 4))))
  (is (= (range 1 11) (dc/support (cm/zipf 10 1.2))))
  (is (= ##-Inf (dc/log-prob (cm/zipf 10 1.2) 11)))
  (is (= ##-Inf (dc/log-prob (cm/zipf 10 1.2) 1e12)) "beyond an int")
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer" (cm/zipf -1 1.2)))
  (testing "counts that are not integers (review 4: truncated)"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer" (cm/hypergeometric 2.5 7 4)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be an integer" (cm/zipf 10.7 1.2)))))

(def ^:private ^RandomGenerator no-rng nil)

(def native-and-commons
  "[name native wrapped points]: gen.dist's family and Commons Math's, by real."
  [["chi-squared" (d/chi-squared 3.0) ((cm/real "chi2" #(ChiSquaredDistribution. no-rng (double %))) 3.0) [0.5 2.0 7.0]]
   ["f-dist" (d/f-dist 3.0 10.0) ((cm/real "f" #(FDistribution. no-rng (double %1) (double %2))) 3.0 10.0) [0.3 1.2 4.0]]
   ["gumbel" (d/gumbel 1.0 2.0) ((cm/real "gumbel" #(GumbelDistribution. no-rng (double %1) (double %2))) 1.0 2.0) [-1.0 1.0 5.0]]
   ["levy" (d/levy 1.0 0.5) ((cm/real "levy" #(LevyDistribution. no-rng (double %1) (double %2))) 1.0 0.5) [1.2 3.0 50.0]]
   ["log-normal" (d/log-normal 0.2 0.5) ((cm/real "log-normal" #(LogNormalDistribution. no-rng (double %1) (double %2))) 0.2 0.5) [0.3 1.0 3.0]]
   ["logistic" (d/logistic 0.5 1.5) ((cm/real "logistic" #(LogisticDistribution. no-rng (double %1) (double %2))) 0.5 1.5) [-2.0 0.5 3.0]]
   ["nakagami" (d/nakagami 1.5 2.0)
    ((cm/real "nakagami" #(NakagamiDistribution. no-rng (double %1) (double %2) NakagamiDistribution/DEFAULT_INVERSE_ABSOLUTE_ACCURACY)) 1.5 2.0)
    [0.3 1.2 2.5]]
   ;; Commons Math's Pareto takes (scale, shape), Distributions.jl (shape, scale)
   ["pareto" (d/pareto 3.0 1.5) ((cm/real "pareto" #(ParetoDistribution. no-rng (double %2) (double %1))) 3.0 1.5) [1.6 2.5 6.0]]
   ["student-t" (d/student-t 5.0) ((cm/real "t" #(TDistribution. no-rng (double %))) 5.0) [-2.0 0.0 1.3]]
   ;; and triangular (lower, mode, upper) where Distributions.jl has (lower, upper, mode)
   ["triangular" (d/triangular 0.0 2.0 0.5) ((cm/real "triangular" #(TriangularDistribution. no-rng (double %1) (double %3) (double %2))) 0.0 2.0 0.5)
    [0.2 0.5 1.4]]
   ["weibull" (d/weibull 1.5 2.0) ((cm/real "weibull" #(WeibullDistribution. no-rng (double %1) (double %2))) 1.5 2.0) [0.4 1.5 4.0]]])

(deftest native-families-agree-with-commons-math
  (doseq [[name native wrapped points] native-and-commons
          x points]
    (is (ts/close? (dc/log-prob wrapped x) (dc/log-prob native x) {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-12)}) (str name " at " x)))
  (testing "the means of 4000 draws from each, within five standard errors of each other (not the Lévy's, which has none)"
    (doseq [[[name native wrapped] key] (map vector (remove (comp #{"levy"} first) native-and-commons)
                                            (rng/split (rng/key 2) (count native-and-commons)))]
      (let [n 4000
            draws (fn [dist k] (map ts/number (array/->clj (dc/sample dist k [n]))))
            [k1 k2] (rng/split key)
            a (draws native k1) b (draws wrapped k2)
            mean #(/ (reduce + %) n)
            variance #(let [m (mean %)] (/ (reduce + (map (fn [x] (* (- x m) (- x m))) %)) (dec n)))]
        (is (< (Math/abs (- (mean a) (mean b))) (* 5 (Math/sqrt (/ (+ (variance a) (variance b)) n)))) name)))))

(def model
  (gen []
    (let [k (trace :k (cm/hypergeometric 5 7 4))
          z (trace :z (cm/zipf 6 1.1))]
      (trace :y (d/normal (array/+ k z) 1)))))

(laws/deflaws commons-math [[:model model [] []]] 60 (merge laws/all laws/implementation))
