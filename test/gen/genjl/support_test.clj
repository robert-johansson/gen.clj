(ns gen.genjl.support-test
  "gen.genjl.support against Julia's own answers (isapprox's, Julia 1.12)."
  (:require [clojure.test :refer [deftest is]]
            [gen.genjl.support :as genjl]
            [gen.test-support :as ts]))

(deftest approx-is-julias-isapprox
  (is (genjl/approx? 1.0 (+ 1.0 1e-9)))
  (is (genjl/approx? [1.0 2.0] [1.0 (+ 2.0 1e-9)]))
  (is (genjl/approx? ##Inf ##Inf) "equal, before finiteness")
  (is (not (genjl/approx? ##NaN ##NaN)))
  (is (not (genjl/approx? 0.0 1e-20)) "no atol by default: relative only")
  (is (genjl/approx? 0.0 1e-9 {:atol 1e-8}))
  (is (not (genjl/approx? [1.0 2.0] [1.0 2.0 3.0])) "shapes differ")
  (when-not (ts/f32?)
    (is (not (genjl/approx? 1.0 (+ 1.0 1e-7))) "float64's √eps, 1.5e-8")))
