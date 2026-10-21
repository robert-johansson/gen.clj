(ns gen.genjl.product-test
  "Gen.jl 0.4.8's test/modeling_library/product.jl, ported: products of
   independent distributions, their densities as the factors' sum, and
   their gradients against finite differences (gen.genjl.finite-diff).
   gen.clj's product groups each factor's parameters (one vector per
   factor), so the ports call it from Gen.jl's flat arguments; a value is a
   vector of the factors' values. has_output_grad and has_argument_grads
   are n/a."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.genjl.finite-diff :as fd]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- density
  "Gen.jl's logpdf, on the host."
  [make x & args]
  (ts/number (apply (fd/logpdf make) x args)))

(def ^:private discrete-product
  "ProductDistribution(bernoulli, binom), from its flat arguments."
  (let [product (d/product d/bernoulli d/binom)]
    (fn [p1 n p2] (product [p1] [n p2]))))

(deftest product-of-discrete-distributions
  ;; product.jl:3
  (let [[p1 n p2] [0.5 3 0.9]
        args [[true 2] p1 n p2]]
    (is (dc/discrete? (discrete-product p1 n p2)))
    (is (= [Boolean Long] (mapv class (ts/host (dc/sample (discrete-product p1 n p2) (rng/key 1))))) "a draw")
    (is (ts/close? (+ (density d/bernoulli true p1) (density d/binom 2 n p2)) (apply density discrete-product args)))
    (doseq [i [1 3]]
      (is (fd/logpdf-grad-matches? discrete-product args i) (str "argument " i)))))

(def ^:private continuous-product
  "ProductDistribution(uniform, normal), from its flat arguments."
  (let [product (d/product d/uniform d/normal)]
    (fn [low high mu std] (product [low high] [mu std]))))

(deftest product-of-continuous-distributions
  ;; product.jl:32
  (let [[low high mu std] [-0.5 0.5 0.0 1.0]
        args [[0.1 0.7] low high mu std]]
    (is (not (dc/discrete? (continuous-product low high mu std))))
    (is (every? number? (ts/host (dc/sample (continuous-product low high mu std) (rng/key 1)))) "a draw")
    (is (ts/close? (+ (density d/uniform 0.1 low high) (density d/normal 0.7 mu std)) (apply density continuous-product args)))
    (testing "logpdf_grad against finite differences"
      (is (fd/logpdf-grad-matches? continuous-product args 0) "the value's two elements")
      (doseq [i (range 1 5)]
        (is (fd/logpdf-grad-matches? continuous-product args i) (str "argument " i))))))

(def ^:private dissimilar-product
  "ProductDistribution(bernoulli, normal), from its flat arguments."
  (let [product (d/product d/bernoulli d/normal)]
    (fn [p mu std] (product [p] [mu std]))))

(deftest product-of-dissimilarly-typed-distributions
  ;; product.jl:64
  (let [[p mu std] [0.5 0.0 1.0]
        args [[false 0.3] p mu std]]
    (is (not (dc/discrete? (dissimilar-product p mu std))))
    (is (= [Boolean Double] (mapv class (ts/host (dc/sample (dissimilar-product p mu std) (rng/key 1))))) "a draw")
    (is (ts/close? (+ (density d/bernoulli false p) (density d/normal 0.3 mu std)) (apply density dissimilar-product args)))
    (doseq [i (range 1 4)]
      (is (fd/logpdf-grad-matches? dissimilar-product args i) (str "argument " i)))))
