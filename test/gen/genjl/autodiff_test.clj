(ns gen.genjl.autodiff-test
  "Gen.jl 0.4.8's test/autodiff.jl on gen.array's gradients: a scalar
   filled into an array has the sum of the array's cotangent as its
   gradient. Gen.jl's deriv!(y, grad) on the tape is the gradient of
   sum(grad .* y) here, and its fill is array/broadcast-to; array/full of a
   tracked scalar waits for G14 (at the end)."
  (:refer-clojure :exclude [*])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [*]]
            [gen.genjl.support :as genjl]
            [gen.random :as random]))

;; autodiff.jl:3
(deftest fill
  (doseq [shape [[3] [3 2]]]
    (testing shape
      (let [cotangent (random/uniform (random/key 0) shape)
            deriv ((array/grad #(array/sum (* cotangent (array/broadcast-to % shape)))) 2.3)]
        (is (genjl/approx? deriv (array/sum cotangent)))))))

;; Pending G14, array/full of a tape value (genclj-3mx6):
;;   autodiff.jl:10, 18  the same with (array/full shape x) for the fill
