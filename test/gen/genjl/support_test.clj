(ns gen.genjl.support-test
  "gen.genjl.support against Julia's and Gen.jl's own answers (Julia 1.12,
   Gen 0.4.8)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.genjl.support :as genjl]
            [gen.test-support :as ts]))

(deftest approx-is-julias-isapprox
  (is (genjl/approx? 1.0 (+ 1.0 1e-9)))
  (is (genjl/approx? [1.0 2.0] [1.0 (+ 2.0 1e-9)]))
  (is (genjl/approx? ##Inf ##Inf) "equal, before finiteness")
  (is (not (genjl/approx? ##NaN ##NaN)))
  (is (not (genjl/approx? 0.0 1e-20)) "no atol by default: relative only")
  (is (genjl/approx? 0.0 1e-9 {:atol 1e-8}))
  (is (not (genjl/approx? [1.0 2.0] [1.0 2.0 3.0])) "lengths differ")
  (is (not (genjl/approx? (array/zeros [2 3]) (array/zeros [3 2]))) "shapes differ")
  (is (not (genjl/approx? [0.0] 0.0)) "a vector is not a number")
  (when-not (ts/f32?)
    (is (not (genjl/approx? 1.0 (+ 1.0 1e-7))) "float64's √eps, 1.5e-8")))

(deftest logpdf-is-the-log-density
  ;; normal's at its mean: -log(2π)/2
  (is (genjl/approx? -0.9189385332046727 (genjl/logpdf d/normal 0.0 0.0 1.0)))
  (testing "an array value: the sum of its elements' log densities"
    (is (genjl/approx? (- (* -1 (Math/log (* 2 Math/PI))) 0.5)
                       (genjl/logpdf d/normal (array/array [0.0 1.0]) (array/zeros [2]) 1.0))))
  (testing "a value of another shape than the distribution's: an error, as Julia's DimensionMismatch"
    (is (thrown? clojure.lang.ExceptionInfo (genjl/logpdf d/normal (array/zeros [1 3]) (array/zeros [2 3]) 1.0)))))

(deftest choice-map-queries
  (let [c (cm/choicemap :x 1 [:y :z] 2)]
    (is (genjl/has-value? c :x))
    (is (genjl/has-value? c [:y :z]))
    (is (not (genjl/has-value? c :y)) "a node")
    (is (not (genjl/has-value? c :w)) "nothing")
    (is (= {:values 1 :submaps 1} (genjl/shallow c)))
    (is (= {:values 0 :submaps 0} (genjl/shallow cm/EMPTY)))))
