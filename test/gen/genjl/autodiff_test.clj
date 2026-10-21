(ns gen.genjl.autodiff-test
  "Gen.jl 0.4.8's test/autodiff.jl on gen.array's gradients: a scalar
   filled into an array has the sum of the array's cotangent as its
   gradient. Gen.jl's deriv!(y, grad) on the tape is the gradient of
   sum(grad .* y) here, and its fill is array/full (G14), as
   array/broadcast-to is too."
  (:refer-clojure :exclude [*])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [*]]
            [gen.genjl.support :as genjl]
            [gen.random :as random]))

;; autodiff.jl:3: fill(x, 3) and fill(x, 3, 2), and the same by broadcast-to
(deftest fill
  (doseq [shape [[3] [3 2]]
          [label fill] [["full" #(array/full %2 %1)] ["broadcast-to" array/broadcast-to]]]
    (testing [label shape]
      (let [cotangent (random/uniform (random/key 0) shape)
            deriv ((array/grad #(array/sum (* cotangent (fill % shape)))) 2.3)]
        (is (genjl/approx? deriv (array/sum cotangent)))))))
