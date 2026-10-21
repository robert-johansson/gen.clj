(ns gen.genjl.finite-diff-test
  "gen.genjl.finite-diff against derivatives known in closed form."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.genjl.finite-diff :as fd]
            [gen.test-support :as ts]))

(deftest finite-differences
  (testing "a number argument: the partial derivative"
    (let [f (fn [x y] (* x x y))]
      (is (< (Math/abs (- 12.0 (fd/finite-diff f [3.0 2.0] 0))) 1e-8))
      (is (< (Math/abs (- 9.0 (fd/finite-diff f [3.0 2.0] 1))) 1e-8))))
  (testing "an array argument: every partial, in its shape"
    (is (array/allclose? (array/array [2.0 4.0 6.0])
                         (fd/finite-diff #(array/sum (array/square %)) [(array/array [1.0 2.0 3.0])] 0)
                         {:rtol 1e-8 :atol 1e-8})))
  (testing "symmetric: (j, k) and (k, j) move together, so one triangle's read counts half on each"
    (is (array/allclose? (array/array [[0.0 0.5] [0.5 0.0]])
                         (fd/finite-diff #(array/item (array/slice (array/slice % 0 0 1) 1 1 2))
                                         [(array/array [[1.0 2.0] [2.0 3.0]])] 0 {:symmetric? true})
                         {:rtol 1e-8 :atol 1e-8}))))

(deftest log-densities
  (testing "an array value: the sum of its elements' log densities, as Gen.jl's logpdf"
    (is (ts/close? ((fd/logpdf d/normal) (array/array [0.0 1.0]) (array/zeros [2]) 1.0)
                   (- (* -1 (Math/log (* 2 Math/PI))) 0.5))))
  (testing "a value of another shape than the parameters': an error, as Gen.jl's DimensionMismatch"
    (is (thrown? clojure.lang.ExceptionInfo ((fd/logpdf d/normal) (array/zeros [1 3]) (array/zeros [2 3]) 1.0))))
  (testing "another step"
    (is (< (Math/abs (- 6.0 (fd/finite-diff (fn [x] (* x x)) [3.0] 0 {:dx 1e-3}))) 1e-9))))

(deftest logpdf-gradients
  (testing "normal(0.4; 0.2, 0.3): -(x - μ)/σ², (x - μ)/σ², -1/σ + (x - μ)²/σ³"
    (let [[gx gmu gs] (fd/logpdf-grad d/normal 0.4 0.2 0.3)]
      (is (fd/grad-close? gx (/ -0.2 0.09) {:rtol 1e-12}))
      (is (fd/grad-close? gmu (/ 0.2 0.09) {:rtol 1e-12}))
      (is (fd/grad-close? gs (+ (/ -1 0.3) (/ 0.04 0.027)) {:rtol 1e-12}))))
  (testing "an integer or boolean value or argument: nil, as Gen.jl's nothing"
    (is (nil? (first (fd/logpdf-grad d/bernoulli false 0.3))))
    (let [[gx gn gp] (fd/logpdf-grad d/binom 2 5 0.3)]
      (is (= [nil nil] [gx gn]))
      (is (some? gp)))))

(deftest isapprox
  (is (fd/grad-close? 1.0 (+ 1.0 1e-9)))
  (is (not (fd/grad-close? 1.0 1.001)))
  (is (not (fd/grad-close? (array/array [1.0 2.0]) (array/array [[1.0 2.0]]))) "shapes must agree")
  (when-not (ts/f32?)
    (is (not (fd/grad-close? 1.0 (+ 1.0 1e-7))) "isapprox's default rtol, √eps")
    (is (fd/grad-close? 1.0 (+ 1.0 1e-8) {:rtol 1e-7}) "or the testset's own")))
