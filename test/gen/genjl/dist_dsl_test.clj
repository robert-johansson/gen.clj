(ns gen.genjl.dist-dsl-test
  "Gen.jl 0.4.8's test/modeling_library/dist_dsl.jl, ported: pushforward
   distributions (@dist, gen.dist.dsl/dist), their densities and gradients
   against their base distributions', and relabeled categoricals, whose
   labels are keywords and records. gen.clj's dist is untyped, so Julia's
   MethodErrors for parameters of the wrong type are n/a; categorical's
   values are 0-based (D2)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dynamic :as dynamic]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as genjl]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- grads
  "Gen.jl's logpdf_grad, on the host."
  [make & args]
  (ts/host (apply fd/logpdf-grad make args)))

(defn- draw [dist] (ts/host (dc/sample dist (rng/key 1))))

(dist exp-normal [x] (array/exp (d/normal x 0.001)))
(dist shifted-normal [mu sigma] (+ (d/normal mu sigma) 1.0))
(dist shifted-std-normal [] (+ (d/normal 0.0 1.0) 1.0))
(dist vec-normal [mu sigma] (d/broadcasted-normal (+ mu 1.0) (* sigma 2.0)))
(dist transformed-mvnormal [mu sigma] (d/mvnormal (+ mu 1.0) (* sigma 2.0)))
(dist real-minus-uniform [a b] (- 1.0 (d/uniform a b)))
(dist int-bounded-uniform [low high] (d/uniform low high))

(dist labeled-cat [labels probs] (nth labels (d/categorical probs)))

(def ^:private dict "Gen.jl's Dict(1 => :a, 2 => :b), 0-based." {0 :a 1 :b})
(dist dict-cat [probs] (get dict (d/categorical probs)))

;; Julia's @enum Fruit apple orange: Fruit(i) the i-th label. A helper
;; looking up by the draw is guarded, as the body is (on MLX the draw is
;; an array, read where the lookup needs it)
(dynamic/defn-guarded ^:private fruit [i] (nth [:apple :orange] i))
(dist enum-cat [probs] (fruit (d/categorical probs)))

(defrecord ^:private MyLabel [name])

(deftest dist-dsl-untyped
  ;; dist_dsl.jl:1
  (testing "a transformed distribution"
    (is (< (Math/abs (- 1.0 (draw (exp-normal 0.0)))) 5))
    (is (ts/close? (genjl/logpdf d/normal 0.0 0.0 0.001) (genjl/logpdf exp-normal 1.0 0.0))))
  (testing "its gradients"
    (is (= (genjl/logpdf d/normal 0.0 0.0 1.0) (genjl/logpdf shifted-normal 1.0 0.0 1.0)))
    (is (= (grads d/normal -1.0 0.0 1.0) (grads shifted-normal 0.0 0.0 1.0))))
  (testing "with no parameters"
    (is (= (genjl/logpdf d/normal 0.0 0.0 1.0) (genjl/logpdf shifted-std-normal 1.0)))
    (is (= [(first (grads d/normal -1.0 0.0 1.0))] (grads shifted-std-normal 0.0))))
  (testing "a multivariate distribution: transformed parameters"
    (let [[x mu sigma] [(array/zeros [2]) (array/zeros [2]) (array/ones [2])]
          transformed (grads vec-normal x mu sigma)
          original (grads d/broadcasted-normal x (array/ones [2]) (* 2.0 (array/ones [2])))]
      (is (= (genjl/logpdf d/broadcasted-normal x (array/ones [2]) (* 2.0 (array/ones [2]))) (genjl/logpdf vec-normal x mu sigma)))
      (is (= (first original) (first transformed)))
      (is (= (second original) (second transformed)))
      (is (= (mapv #(* 2.0 %) (nth original 2)) (nth transformed 2)))))
  (testing "a covariance that is singular in floating point: an error on both sides (D15)"
    ;; Gen.jl's 2·ones(2, 2), whose Cholesky factor exists only by rounding,
    ;; gives 15.24 there (U3); gen.clj rejects it, whether given or made,
    ;; where the backend can read an array's values (rule 5: an eager one)
    (let [[x mu sigma] [(array/zeros [2]) (array/zeros [2]) (array/ones [2 2])]
          singular? (fn [f] (try (f) false (catch clojure.lang.ExceptionInfo e (boolean (re-find #"positive definite" (ex-message e))))))]
      (when (array/concrete? sigma)
        (is (singular? #(genjl/logpdf transformed-mvnormal x mu sigma)) "logpdf of the transformed distribution")
        (is (singular? #(genjl/logpdf d/mvnormal x (array/ones [2]) (* 2.0 (array/ones [2 2])))) "logpdf of mvnormal itself")
        (is (singular? #(grads transformed-mvnormal x mu sigma)) "logpdf_grad of the transformed distribution")
        (is (singular? #(grads d/mvnormal x (array/ones [2]) (* 2.0 (array/ones [2 2])))) "logpdf_grad of mvnormal itself"))
      (is (singular? #(genjl/logpdf d/mvnormal x (array/ones [2]) [[2.0 2.0] [2.0 2.0]]))
          "given as data, which every backend validates (dc/->checkable)")))
  (testing "labels from a vector"
    (let [probs (array/array [0.5 0.5])]
      (is (= :b (draw (labeled-cat [:a :b] (array/array [0.0 1.0])))))
      (is (genjl/approx? (Math/log 0.5) (genjl/logpdf labeled-cat :b [:a :b] probs)))
      (is (= (into [nil] (grads d/categorical 1 probs)) (grads labeled-cat :b [:a :b] probs)) "nil for the labels too")
      (is (= ##-Inf (genjl/logpdf labeled-cat :c [:a :b] probs)))))
  (testing "labels from a map"
    (let [probs (array/array [0.5 0.5])]
      (is (= :b (draw (dict-cat (array/array [0.0 1.0])))))
      (is (genjl/approx? (Math/log 0.5) (genjl/logpdf dict-cat :b probs)))
      (is (= (grads d/categorical 1 probs) (grads dict-cat :b probs)))
      (is (= ##-Inf (genjl/logpdf dict-cat :c probs)))))
  (testing "an enum's labels"
    (let [probs (array/array [0.5 0.5])]
      (is (= :orange (draw (enum-cat (array/array [0.0 1.0])))))
      (is (genjl/approx? (Math/log 0.5) (genjl/logpdf enum-cat :orange probs)))
      (is (= ##-Inf (genjl/logpdf enum-cat :orange (array/array [1.0]))))
      (is (= (grads d/categorical 1 probs) (grads enum-cat :orange probs)))))
  (testing "1 - uniform (Gen.jl's issue 253)"
    (is (< (draw (real-minus-uniform 1.0 2.0)) 0))
    (is (= 0.0 (genjl/logpdf real-minus-uniform -0.5 1.0 2.0)))
    (is (= (grads d/uniform 1.5 1.0 2.0) (grads real-minus-uniform -0.5 1.0 2.0)))))

(deftest dist-dsl-typed
  ;; dist_dsl.jl:72, untyped: what does not test a MethodError
  (testing "symbol_cat, labeled-cat's keywords"
    (let [probs (array/array [0.5 0.5])]
      (is (= :b (draw (labeled-cat [:a :b] (array/array [0.0 1.0])))))
      (is (= ##-Inf (genjl/logpdf labeled-cat :c [:a :b] probs)))
      (is (= (into [nil] (grads d/categorical 1 probs)) (grads labeled-cat :b [:a :b] probs)))))
  (testing "the typed uniform, with the gradients of float bounds, which Julia's Ints get too"
    (is (<= 0.0 (draw (int-bounded-uniform 0 1)) 1.0))
    (is (= 0.0 (genjl/logpdf int-bounded-uniform 0.5 0 1)))
    (is (= (grads d/uniform 0.5 0.0 1.0) (grads int-bounded-uniform 0.5 0.0 1.0))))
  (testing "mylabel_cat: records for labels"
    (is (= (->MyLabel :b) (draw (labeled-cat [(->MyLabel :a) (->MyLabel :b)] (array/array [0.0 1.0])))))
    (is (= 0.0 (genjl/logpdf labeled-cat (->MyLabel :a) [(->MyLabel :a)] (array/array [1.0]))))))
