(ns gen.genjl.finite-diff-test
  "gen.genjl.finite-diff against derivatives known in closed form."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as genjl]
            [gen.test-support :as ts]))

(deftest finite-differences
  (testing "a number argument: the partial derivative"
    (let [f (fn [x y] (* x x y))]
      (is (< (Math/abs (- 12.0 (fd/finite-diff f [3.0 2.0] 0))) 1e-8))
      (is (< (Math/abs (- 9.0 (fd/finite-diff f [3.0 2.0] 1))) 1e-8))))
  (testing "an array argument, from the backend under test: every partial, in its shape, in float64"
    (let [partials (fd/finite-diff #(array/sum (array/square %)) [(array/array [1.0 2.0 3.0])] 0)]
      (is (= :float64 (array/dtype partials)))
      (is (genjl/approx? [2.0 4.0 6.0] partials {:rtol 1e-8 :atol 1e-8}))))
  (testing "symmetric: (j, k) and (k, j) move together, so one triangle's read counts half on each"
    (is (genjl/approx? [[0.0 0.5] [0.5 0.0]]
                       (fd/finite-diff #(array/item (array/slice (array/slice % 0 0 1) 1 1 2))
                                       [(array/array [[1.0 2.0] [2.0 3.0]])] 0 {:symmetric? true})
                       {:rtol 1e-8 :atol 1e-8})))
  (testing "another step"
    (is (< (Math/abs (- 6.0 (fd/finite-diff (fn [x] (* x x)) [3.0] 0 {:dx 1e-3}))) 1e-9))))

(deftest logpdf-gradients
  (testing "normal(0.4; 0.2, 0.3): -(x - μ)/σ², (x - μ)/σ², -1/σ + (x - μ)²/σ³"
    (let [[gx gmu gs] (fd/logpdf-grad d/normal 0.4 0.2 0.3)]
      (is (fd/grad-close? gx (/ -0.2 0.09) {:rtol 1e-12}))
      (is (fd/grad-close? gmu (/ 0.2 0.09) {:rtol 1e-12}))
      (is (fd/grad-close? gs (+ (/ -1 0.3) (/ 0.04 0.027)) {:rtol 1e-12}))))
  (testing "an integer or boolean value or argument: nil, as Gen.jl's nothing, the others' given"
    (let [[gx gp] (fd/logpdf-grad d/bernoulli true 0.3)]
      (is (nil? gx))
      (is (fd/grad-close? gp (/ 1 0.3) {:rtol 1e-12}) "d log p / dp"))
    (let [[gx gn gp] (fd/logpdf-grad d/binom 2 5 0.3)]
      (is (= [nil nil] [gx gn]))
      (is (some? gp)))))

(deftest isapprox
  (is (fd/grad-close? 1.0 (+ 1.0 1e-9)))
  (is (not (fd/grad-close? 1.0 1.01)) "outside float64's band and float32's")
  (is (fd/grad-close? (array/array [1.0 1e-12]) (array/array [1.0 0.0])) "in the norm, not element by element")
  (is (not (fd/grad-close? (array/array [1.0 2.0]) (array/array [[1.0 2.0]]))) "shapes must agree")
  (if (ts/f32?)
    (is (fd/grad-close? 1.0 1.001) "float32's band, rtol 2e-3")
    (do (is (not (fd/grad-close? 1.0 (+ 1.0 1e-7))) "isapprox's default rtol, √eps")
        (is (fd/grad-close? 1.0 (+ 1.0 1e-8) {:rtol 1e-7}) "or the testset's own"))))
