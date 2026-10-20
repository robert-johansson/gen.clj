(ns gen.dist.commons-math-test
  "Commons Math's distributions as gen.clj's (gen.dist.commons-math): each
   family's log density against its closed form, the mean of 4000 draws
   against the true mean (a band of five standard errors from the true
   variance), finite supports, and the GFI laws on a model over several
   families. Run with clojure -X:test:commons-math."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.special]
            [gen.dist :as d]
            [gen.dist.commons-math :as cm]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- lgamma ^double [x] (ts/number (array/lgamma (double x))))
(defn- log-choose [n k] (- (lgamma (inc n)) (lgamma (inc k)) (lgamma (inc (- n k)))))
(def ^:private euler-gamma 0.5772156649015329)
(defn- gamma-fn [x] (Math/exp (lgamma x)))

(def families
  "[name distribution points closed-form-log-density mean variance]"
  (let [h (fn [n s] (reduce + (map #(Math/pow % (- s)) (range 1 (inc n)))))]
    [["chi-squared" (cm/chi-squared 3.0) [0.5 2.0 7.0]
      (fn [x] (- (* 0.5 (Math/log x)) (* 0.5 x) (* 1.5 (Math/log 2.0)) (lgamma 1.5))) 3.0 6.0]
     ["f-dist" (cm/f-dist 3.0 10.0) [0.3 1.2 4.0]
      (fn [x] (- (* 0.5 (- (+ (* 3 (Math/log (* 3 x))) (* 10 (Math/log 10.0))) (* 13 (Math/log (+ (* 3 x) 10)))))
                 (Math/log x) (- (+ (lgamma 1.5) (lgamma 5.0)) (lgamma 6.5))))
      1.25 (/ (* 2 100 11) (* 3 64 6.0))]
     ["gumbel" (cm/gumbel 1.0 2.0) [-1.0 1.0 5.0]
      (fn [x] (let [z (/ (- x 1.0) 2.0)] (- (- (Math/log 2.0)) z (Math/exp (- z)))))
      (+ 1.0 (* 2.0 euler-gamma)) (/ (* Math/PI Math/PI 4.0) 6)]
     ["log-normal" (cm/log-normal 0.2 0.5) [0.3 1.0 3.0]
      (fn [x] (- (- (Math/log (* x 0.5 (Math/sqrt (* 2 Math/PI))))) (/ (Math/pow (- (Math/log x) 0.2) 2) (* 2 0.25))))
      (Math/exp (+ 0.2 0.125)) (* (- (Math/exp 0.25) 1) (Math/exp (+ 0.4 0.25)))]
     ["logistic" (cm/logistic 0.5 1.5) [-2.0 0.5 3.0]
      (fn [x] (let [z (/ (- x 0.5) 1.5)] (- (- z) (Math/log 1.5) (* 2 (Math/log1p (Math/exp (- z)))))))
      0.5 (/ (* Math/PI Math/PI 2.25) 3)]
     ["nakagami" (cm/nakagami 1.5 2.0) [0.3 1.2 2.5]
      (fn [x] (+ (Math/log 2.0) (* 1.5 (Math/log 1.5)) (- (lgamma 1.5)) (* -1.5 (Math/log 2.0))
                 (* 2.0 (Math/log x)) (- (/ (* 1.5 x x) 2.0))))
      (* (/ (gamma-fn 2.0) (gamma-fn 1.5)) (Math/sqrt (/ 2.0 1.5)))
      (* 2.0 (- 1 (* (/ 1 1.5) (Math/pow (/ (gamma-fn 2.0) (gamma-fn 1.5)) 2))))]
     ["pareto" (cm/pareto 3.0 1.5) [1.6 2.5 6.0]
      (fn [x] (- (+ (Math/log 3.0) (* 3.0 (Math/log 1.5))) (* 4.0 (Math/log x))))
      (/ (* 3.0 1.5) 2.0) (/ (* 1.5 1.5 3.0) (* 4.0 1.0))]
     ["student-t" (cm/student-t 5.0) [-2.0 0.0 1.3]
      (fn [x] (- (lgamma 3.0) (lgamma 2.5) (* 0.5 (Math/log (* 5 Math/PI))) (* 3.0 (Math/log1p (/ (* x x) 5)))))
      0.0 (/ 5.0 3.0)]
     ["triangular" (cm/triangular 0.0 2.0 0.5) [0.2 0.5 1.4]
      (fn [x] (Math/log (if (< x 0.5) (/ (* 2 x) (* 2.0 0.5)) (/ (* 2 (- 2.0 x)) (* 2.0 1.5)))))
      (/ 2.5 3) (/ (- (+ 4.0 0.25) 1.0) 18)]
     ["weibull" (cm/weibull 1.5 2.0) [0.4 1.5 4.0]
      (fn [x] (- (+ (Math/log 0.75) (* 0.5 (Math/log (/ x 2.0)))) (Math/pow (/ x 2.0) 1.5)))
      (* 2.0 (gamma-fn (+ 1 (/ 1 1.5)))) (* 4.0 (- (gamma-fn (+ 1 (/ 2 1.5))) (Math/pow (gamma-fn (+ 1 (/ 1 1.5))) 2)))]
     ["hypergeometric" (cm/hypergeometric 5 7 4) [0 2 4]
      (fn [k] (- (+ (log-choose 5 k) (log-choose 7 (- 4 k))) (log-choose 12 4)))
      (/ 20 12.0) (* 4 (/ 5 12.0) (/ 7 12.0) (/ 8 11.0))]
     ["zipf" (cm/zipf 10 1.2) [1 3 10]
      (fn [k] (- (* -1.2 (Math/log k)) (Math/log (h 10 1.2))))
      (/ (h 10 0.2) (h 10 1.2)) (- (/ (h 10 -0.8) (h 10 1.2)) (Math/pow (/ (h 10 0.2) (h 10 1.2)) 2))]]))

(deftest log-densities-are-the-closed-forms
  (doseq [[name dist points log-density] families
          x points]
    (is (ts/close? (log-density x) (dc/log-prob dist x) {:rtol 1e-9 :atol 1e-12}) (str name " at " x))))

(deftest means-of-draws
  (doseq [[[name dist _ _ mean variance] key] (map vector families (rng/split (rng/key 1) (count families)))]
    (let [n 4000
          xs (map ts/number (array/->clj (dc/sample dist key [n])))
          m (/ (reduce + xs) n)]
      (is (< (Math/abs (- m mean)) (* 5 (Math/sqrt (/ variance n)))) (str name ": " m " against " mean)))))

(deftest the-levy-distribution-s-median
  ;; no finite mean: P(X ≤ μ + 4σ) = erfc(√(1/8)), within five standard errors
  (let [n 4000
        xs (map ts/number (array/->clj (dc/sample (cm/levy 1.0 0.5) (rng/key 2) [n])))
        p (gen.array.special/erfc (Math/sqrt (/ 0.5 (* 2 2.0))))
        f (/ (count (filter #(<= % 3.0) xs)) (double n))]
    (is (ts/close? (- (+ (* 0.5 (Math/log (/ 0.5 (* 2 Math/PI)))) (* -1.5 (Math/log 2.0))) (/ 0.5 (* 2 2.0)))
                   (dc/log-prob (cm/levy 1.0 0.5) 3.0)))
    (is (< (Math/abs (- f p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))) (str f " against " p))))

(deftest supports
  (is (= [0 1 2 3 4] (dc/support (cm/hypergeometric 5 7 4))))
  (is (= (range 1 11) (dc/support (cm/zipf 10 1.2))))
  (is (= ##-Inf (dc/log-prob (cm/zipf 10 1.2) 11)))
  (is (= ##-Inf (dc/log-prob (cm/zipf 10 1.2) 1e12)) "beyond an int")
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid parameters" (cm/weibull -1.0 2.0))))

(def model
  (gen []
    (let [nu (trace :nu (d/gamma 4 1))
          t (trace :t (cm/student-t nu))
          w (trace :w (cm/weibull 1.5 2.0))
          k (trace :k (cm/hypergeometric 5 7 4))]
      (trace :y (d/normal (+ t w k) 1)))))

(laws/deflaws commons-math [[:model model [] []]] 60)
