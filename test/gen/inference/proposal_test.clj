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
            [gen.inference.mh :as mh]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
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
;; every address has a fixed support: Def. 4.1.1 has content for a proposal
;; other than the prior (law audit gap 8)
(laws/deflaws internal-proposal law-models 60 laws/fixed-support)

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
                     {:rtol (ts/tol 1e-12) :atol (ts/tol 1e-12)})
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

(def gated
  "b decides whether :x exists."
  (gen [] (let [b (trace :b (d/bernoulli 0.5))]
            (when b (trace :x (d/normal 0 1)))
            (trace :y (d/normal 0.3 1)))))

(def propose-b (gen [sigma] (when (open? sigma :b) (trace :b (d/bernoulli 0.5)))))

(deftest regenerating-a-structure-deciding-choice
  (let [pgf (prop/with-internal-proposal gated propose-b)
        tr (:trace (gfi/generate pgf [] (cm/choicemap :b true :x 0.4 :y 1.0) (rng/key 1)))
        outcomes (map #(try (let [{t' :trace w :weight} (gfi/regenerate tr (gen.selection/select :b) %)]
                              [(t' :b) (t' :x) w])
                            (catch clojure.lang.ExceptionInfo e (ex-message e)))
                      (rng/split (rng/key 2) 12))]
    (testing "a new b that keeps :x keeps its value, with the projection ratio as weight"
      (is (seq (filter vector? outcomes)))
      (doseq [[b x w] (filter vector? outcomes)]
        (is (true? (ts/host b)))
        (is (ts/close? 0.4 x) "the unselected :x is kept")
        (is (ts/close? 0.0 w) "b's proposal is its prior: the weight is 0")))
    (testing "a new b that drops :x is a loud error"
      (is (seq (remove vector? outcomes)))
      (is (every? #(re-find #"leaves an unselected choice unvisited" %) (remove vector? outcomes))))
    (testing "selecting what b decides works"
      (let [{t' :trace} (gfi/regenerate tr (gen.selection/select :b :x) (rng/key 3))]
        (is (some? (t' :b)))))))

(deftest ^:slow generate-under-an-override-is-unbiased
  ;; Alg. 16 (law audit gap D): with q proposing the branch z ~ Bern(0.7)
  ;; and y = 0.5, E[exp(w)] = p(y) = 0.3 N(.5; 2, √2) + 0.7 N(.5; -2, √2)
  (let [pgf (prop/with-internal-proposal branching (gen [sigma] (when (open? sigma :z) (trace :z (d/bernoulli 0.7)))))
        evidence (Math/log (+ (* 0.3 (Math/exp (ts/number (dc/log-prob (d/normal 2.0 (Math/sqrt 2)) 0.5))))
                              (* 0.7 (Math/exp (ts/number (dc/log-prob (d/normal -2.0 (Math/sqrt 2)) 0.5))))))
        n 4000
        ratios (mapv #(Math/exp (- (ts/number (:weight (gfi/generate pgf [] (cm/choicemap :y 0.5) %))) evidence))
                     (rng/split (rng/key 7) n))
        m (/ (reduce clojure.core/+ ratios) n)
        sd (Math/sqrt (/ (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % m) 2) ratios)) (dec n)))]
    (is (< (Math/abs (clojure.core/- m 1.0)) (* 5 (/ sd (Math/sqrt n)))) (str m " ± " (/ sd (Math/sqrt n))))))

(deftest ^:slow ^:heavy mh-under-an-override-keeps-the-posterior
  ;; law audit gap 6: from 4000 exact posterior draws (y = 1.3: mu | y ~
  ;; N(0.65, sqrt(1/2))), one selection-MH move with q = N(1, 1.5) as the
  ;; internal proposal keeps mean and variance within 5 standard errors
  (let [gf (prop/with-internal-proposal normal-normal
             (gen [sigma] (when (open? sigma :mu) (trace :mu (d/normal 1.0 1.5)))))
        n 4000
        v 0.5
        mus (mapv #(clojure.core/+ 0.65 (clojure.core/* (Math/sqrt v) %)) (ts/host (dc/sample (d/normal 0 1) (rng/key 95) [n])))
        moved (mapv (fn [mu k] (let [[k0 k1] (rng/split k)]
                                 (ts/number ((:trace (mh/mh (:trace (gfi/generate gf [] (cm/choicemap :mu mu :y 1.3) k0))
                                                            (sel/select :mu) k1))
                                             :mu))))
                    mus (rng/split (rng/key 96) n))
        m (clojure.core// (reduce clojure.core/+ moved) n)
        var (clojure.core// (reduce clojure.core/+ (map #(let [e (clojure.core/- % m)] (clojure.core/* e e)) moved)) (dec n))]
    (is (< (Math/abs (clojure.core/- m 0.65)) (clojure.core/* 5 (Math/sqrt (clojure.core// v n)))))
    (is (< (Math/abs (clojure.core/- var v)) (clojure.core/* 5 v (Math/sqrt (clojure.core// 2.0 (dec n))))))))
