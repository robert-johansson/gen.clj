(ns gen.inference.proposal-test
  "The internal-proposal override: the general GFI laws over models whose
   proposals make some, all, or a structure-deciding part of the open
   choices; exactness with the exact posterior as the proposal; loud errors."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- open? [sigma addr] (not (cm/has-value? (cm/get-submap sigma addr))))

(defn- given [sigma addr] (when-not (open? sigma addr) (cm/get-choice sigma addr)))

(def normal-normal
  (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)) mu)))

(def two-latents
  (gen [] (let [a (trace :a (d/normal 0 1))
                b (trace :b (d/normal a 1))]
            (trace :y (d/normal b 0.5))
            b)))

(def branching
  (gen [] (let [z (trace :z (d/bernoulli 0.3))
                x (trace :x (d/normal (if z 2.0 -2.0) 1))]
            (trace :y (d/normal x 1))
            x)))

(def with-args
  (gen [m] (let [x (trace :x (d/normal m 2))] (trace :y (d/normal x 1)) x)))

(def law-models
  [[:proposes-the-latent
    (prop/with-internal-proposal normal-normal
      (gen [sigma] (when (open? sigma :mu) (trace :mu (d/normal 1.0 1.5)))))
    [] []]
   [:proposes-one-of-two-from-the-data
    (prop/with-internal-proposal two-latents
      (gen [sigma] (when (open? sigma :a)
                     (trace :a (d/normal (* 0.5 (or (given sigma :y) 0.0)) 1.2)))))
    [] []]
   [:proposes-the-branch
    (prop/with-internal-proposal branching
      (gen [sigma] (when (open? sigma :z) (trace :z (d/bernoulli 0.7)))))
    [] []]
   [:proposal-reads-the-arguments
    (prop/with-internal-proposal with-args
      (gen [sigma m] (when (open? sigma :x) (trace :x (d/normal (+ m 0.5) 1.5)))))
    [1.0] [[2.0]]]])

(laws/deflaws internal-proposal law-models 60 laws/general)

(deftest the-exact-posterior-as-proposal-gives-the-evidence
  ;; mu ~ N(0, 1), y ~ N(mu, 1): mu | y ~ N(y/2, √½), and p(y) = N(y; 0, √2).
  ;; With that posterior as q, every key's weight is log p(y) exactly
  (let [y 1.3
        exact (prop/with-internal-proposal normal-normal
                (gen [sigma] (when (open? sigma :mu)
                               (trace :mu (d/normal (* 0.5 (given sigma :y)) (Math/sqrt 0.5))))))
        evidence (dc/log-prob (d/normal 0 (Math/sqrt 2)) y)]
    (doseq [k (range 8)]
      (is (ts/close? evidence (:weight (gfi/generate exact [] (cm/choicemap :y y) (rng/key k)))
                     {:rtol 1e-12 :atol 1e-12})
          (str "key " k)))
    (testing "the default proposal does not have this property"
      (let [ws (map #(ts/number (:weight (gfi/generate normal-normal [] (cm/choicemap :y y) (rng/key %)))) (range 8))]
        (is (apply not= ws))))))

(deftest the-distribution-is-the-model's
  (let [pgf (prop/with-internal-proposal normal-normal (gen [sigma] (trace :mu (d/normal 5 0.1))))
        choices (cm/choicemap :mu 0.2 :y 1.0)]
    (is (ts/close? (:weight (gfi/assess normal-normal [] choices)) (:weight (gfi/assess pgf [] choices))))))

(deftest misuse-is-loud
  (testing "a proposal for a constrained choice"
    (let [pgf (prop/with-internal-proposal normal-normal (gen [sigma] (trace :mu (d/normal 0 1))))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"which the constraints give"
                            (gfi/generate pgf [] (cm/choicemap :mu 0.5 :y 1.0) (rng/key 1))))))
  (testing "a proposal calling a generative function at an address cannot be replayed"
    (let [inner (gen [] (trace :mu (d/normal 0 1)))
          model (gen [] (let [mu (splice :sub inner)] (trace :y (d/normal mu 1))))
          pgf (prop/with-internal-proposal model (gen [sigma] (when (open? sigma [:sub :mu]) (splice :sub inner))))
          tr (:trace (gfi/generate pgf [] (cm/choicemap :y 1.0) (rng/key 1)))]
      (is (= #{[:sub :mu] [:y]} (set (cm/addresses (:choices tr)))) "generate works")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot be replayed"
                            (gfi/project tr (gen.selection/select :y))))))
  (testing "both are dynamic generative functions"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dynamic generative function"
                          (prop/with-internal-proposal (d/normal 0 1) normal-normal)))))
