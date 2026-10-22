(ns gen.inference.proposal-test
  "The internal-proposal override: the general GFI laws over models whose
   proposals make some, all, or a structure-deciding part of the open
   choices, and over models that splice one; exactness with the exact
   posterior as the proposal; the worked examples of pp. 143-144; loud
   errors."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.inference.mh :as mh]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

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

(def proposes-the-latent
  (prop/with-internal-proposal normal-normal
    (gen [sigma] (when (open? sigma :mu) (trace :mu (d/normal 1.0 1.5))))))

(def proposal-reads-the-arguments
  (prop/with-internal-proposal with-args
    (gen [sigma m] (when (open? sigma :x) (trace :x (d/normal (+ m 0.5) 1.5))))))

(def grows-with-n
  "n latents and their sum's noisy observation."
  (gen [n] (let [xs (mapv #(trace (keyword (str "x" %)) (d/normal 0 1)) (range n))]
             (trace :y (d/normal (reduce + 0.0 xs) 1)))))

(def proposes-what-n-adds
  (prop/with-internal-proposal grows-with-n
    (gen [sigma n] (doseq [i (range n) :let [a (keyword (str "x" i))]]
                     (when (open? sigma a) (trace a (d/normal 0.5 1.5)))))))

(def law-models
  [[:proposes-the-latent proposes-the-latent [] []]
   [:proposes-one-of-two-from-the-data
    (prop/with-internal-proposal two-latents
      (gen [sigma] (when (open? sigma :a)
                     (trace :a (d/normal (* 0.5 (or (given sigma :y) 0.0)) 1.2)))))
    [] []]
   [:proposes-the-branch
    (prop/with-internal-proposal branching
      (gen [sigma] (when (open? sigma :z) (trace :z (d/bernoulli 0.7)))))
    [] []]
   [:proposal-reads-the-arguments proposal-reads-the-arguments [1.0] [[2.0]]]
   ;; an update to more arguments makes fresh choices, which the proposal
   ;; draws (genclj-nqsm: update-under, shared with with-exact-proposal);
   ;; fewer arguments would leave an unselected choice to regenerate unvisited
   [:proposes-what-the-arguments-add proposes-what-n-adds [1] [[2] [3]]]])

(laws/deflaws internal-proposal law-models 60 laws/general)
;; every address has a fixed support: Def. 4.1.1 has content for a proposal
;; other than the prior (law audit gap 8)
(laws/deflaws internal-proposal law-models 60 laws/fixed-support)

;; §4.4, p. 154 (law audit 4.c): a model that calls a generative function with
;; an internal proposal, which proposes the callee's choices for it. Planted
;; bugs (2026-10-20): the override's project dropping log q (the audit's C3)
;; fails generate-weight-is-the-projection and the Eq. 4.1 laws over these;
;; its generate leaving q out (forward sampling) fails them too
(def spliced-models
  [[:splices-an-override
    (gen [] (let [mu (splice :p proposes-the-latent)] (trace :z (d/normal mu 1)) mu))
    [] []]
   [:splices-an-override-with-arguments
    (gen [m] (let [x (splice :p proposal-reads-the-arguments m)] (trace :z (d/normal x 1))))
    [1.0] [[2.0]]]])

(laws/deflaws spliced-internal-proposal spliced-models 60 laws/general)
(laws/deflaws spliced-internal-proposal spliced-models 60 laws/fixed-support)

(def exact-normal-normal
  "normal-normal with its exact posterior as q: given y, mu ~ N(y/2, √½)."
  (prop/with-internal-proposal normal-normal
    (gen [sigma] (when (open? sigma :mu)
                   (trace :mu (d/normal (* 0.5 (given sigma :y)) (Math/sqrt 0.5)))))))

(deftest the-exact-posterior-as-proposal-gives-the-evidence
  ;; mu ~ N(0, 1), y ~ N(mu, 1): mu | y ~ N(y/2, √½), and p(y) = N(y; 0, √2).
  ;; With that posterior as q, every key's weight is log p(y) exactly
  (let [y 1.3
        exact exact-normal-normal
        evidence (dc/log-prob (d/normal 0 (Math/sqrt 2)) y)]
    (doseq [k (range 8)]
      (is (ts/close? evidence (:weight (gfi/generate exact [] (cm/choicemap :y y) (rng/key k)))
                     {:rtol (ts/tol 1e-12) :atol (ts/tol 1e-12)})
          (str "key " k)))
    (testing "the default proposal does not have this property"
      (let [ws (map #(ts/number (:weight (gfi/generate normal-normal [] (cm/choicemap :y y) (rng/key %)))) (range 8))]
        (is (apply not= ws))))
    ;; §4.4 (law audit 4.c). Planted bug (2026-10-20): the override's
    ;; generate leaving q out fails every key
    (testing "and a model that splices it uses its proposal: z, unconstrained, adds nothing"
      (let [model (gen [] (let [mu (splice :p exact)] (trace :z (d/normal mu 1))))]
        (doseq [k (range 8)]
          (is (ts/close? evidence (:weight (gfi/generate model [] (cm/choicemap [:p :y] y) (rng/key k)))
                         {:rtol (ts/tol 1e-12) :atol (ts/tol 1e-12)})
              (str "key " k)))))))

;; ---------------------------------------------------------------------------
;; The worked examples of pp. 143-144
;; ---------------------------------------------------------------------------

(def abc
  "The model of p. 143's table: p({a F c F}) = 0.45, p({a F c T}) = 0.05,
   p({a T b F c F}) = 0.05, p({a T b F c T}) = 0.2, p({a T b T c F}) =
   p({a T b T c T}) = 0.125."
  (gen [] (let [a (trace :a (d/bernoulli 0.5))
                b (when a (trace :b (d/bernoulli 0.5)))]
            (trace :c (d/bernoulli (if a (if b 0.5 0.8) 0.1))))))

(def abc-proposal
  "The internal proposal of p. 143: a ~ Bern(0.6), then b ~ Bern(0.5) when a
   is true, and c ~ Bern(0.5), each where σ leaves it open. For σ1 = {c T}
   it gives q({a F}) = 0.4, q({a T b F}) = q({a T b T}) = 0.3. (Its table
   for σ2 = {b F}, whose {a F} branches leave b unvisited, is a loud error
   in gen.clj: generate visits every constraint.)"
  (gen [sigma] (let [a (if (open? sigma :a) (trace :a (d/bernoulli 0.6)) (given sigma :a))]
                 (when (and a (open? sigma :b)) (trace :b (d/bernoulli 0.5)))
                 (when (open? sigma :c) (trace :c (d/bernoulli 0.5))))))

(deftest the-thesis-s-generate-and-regenerate-examples
  ;; law audit, finding 9: every outcome's weight exactly, the thesis's own
  ;; among them. Planted bugs (2026-10-20): the override's generate weight
  ;; without -log q fails the generate example; its project dropping log q
  ;; (the audit's C3) fails the regenerate example
  (let [pgf (prop/with-internal-proposal abc abc-proposal)
        outcome (fn [tr] (if (ts/host (tr :a)) [true (ts/host (tr :b))] [false]))
        log #(Math/log %)]
    (testing "p. 143: generate given σ1 = {c T}; υ = {a F} weighs log 0.05 - log 0.4"
      (let [expected {[false] (- (log 0.05) (log 0.4))
                      [true false] (- (log 0.2) (log 0.3))
                      [true true] (- (log 0.125) (log 0.3))}
            rs (map #(gfi/generate pgf [] (cm/choicemap :c true) %) (rng/split (rng/key 8) 16))]
        (is (= (set (keys expected)) (set (map (comp outcome :trace) rs))) "every outcome drawn")
        (doseq [{tr :trace w :weight} rs]
          (is (ts/close? (expected (outcome tr)) w) (str (outcome tr))))))
    (testing "p. 144: regenerating {a, b} of {a T b F c T}; υ = {a F} weighs log (0.05 · 0.3)/(0.2 · 0.4)"
      (let [tr (:trace (gfi/generate pgf [] (cm/choicemap :a true :b false :c true) (rng/key 9)))
            expected {[false] (log (/ (* 0.05 0.3) (* 0.2 0.4)))
                      [true false] 0.0
                      [true true] (log (/ (* 0.125 0.3) (* 0.2 0.3)))}
            rs (map #(gfi/regenerate tr (sel/select :a :b) %) (rng/split (rng/key 10) 16))]
        (is (= (set (keys expected)) (set (map (comp outcome :trace) rs))) "every outcome drawn")
        (doseq [{t :trace w :weight} rs]
          (is (ts/close? (expected (outcome t)) w) (str (outcome t))))))))

(deftest the-distribution-is-the-model's
  (let [pgf (prop/with-internal-proposal normal-normal (gen [sigma] (trace :mu (d/normal 5 0.1))))
        choices (cm/choicemap :mu 0.2 :y 1.0)]
    (is (ts/close? (:weight (gfi/assess normal-normal [] choices)) (:weight (gfi/assess pgf [] choices))))))

(deftest calls-are-the-model's
  (let [pgf (prop/with-internal-proposal (gen [] (splice :inner normal-normal)) (gen [_sigma] nil))
        tr (gfi/simulate pgf [] (rng/key 4))]
    (is (= (ts/host (tr [:inner :mu])) (ts/host (tr :inner))))
    (is (= normal-normal (:gen-fn (trace/subtrace tr :inner))))))

(def ^:private flips-with-n
  "n coins, rarely heads, each heads with a value below it; its proposal
   flips each coin heads."
  (prop/with-internal-proposal
    (gen [n] (doseq [i (range n)]
               (when (trace (keyword (str "b" i)) (d/bernoulli 0.001))
                 (trace (keyword (str "z" i)) (d/normal 0 1)))))
    (gen [sigma n] (doseq [i (range n) :let [a (keyword (str "b" i))]]
                     (when (open? sigma a) (trace a (d/bernoulli 0.999)))))))

(deftest updates-stay-the-wrapper-s
  ;; genclj-nqsm: update-under, shared by the two wrappers, keeps the trace
  ;; the wrapper's, with fresh choices or without, and a proposal whose fresh
  ;; choices change the model's structure is a loud error naming the
  ;; wrapper (planted: the trace left as gf's; no structure check)
  (let [m proposes-what-n-adds
        tr (gfi/simulate m [1] (rng/key 3))]
    (is (identical? m (:gen-fn (:trace (gfi/update tr [1] [diff/no-change] (cm/choicemap :y 0.3) (rng/key 4))))))
    (is (identical? m (:gen-fn (:trace (gfi/update tr [2] [diff/unknown-change] cm/EMPTY (rng/key 5)))))))
  (let [tr (gfi/simulate flips-with-n [1] (rng/key 6))]
    (is (= {:b0 false} (ts/host (cm/to-map (:choices tr)))) "the first coin, forward sampled: tails")
    (is (= {:wrapper "with-internal-proposal"}
           (ts/thrown-data (gfi/update tr [2] [diff/unknown-change] cm/EMPTY (rng/key 7))))
        "the model's update draws the new coin tails, the proposal heads, with a z"))
  (testing "a change in the arguments' number reaches the model's update (genclj-pwkr), whatever the argdiffs say"
    ;; m keeps its value: the weight is log N(m; 0, 1) - log N(m; 1, 1)
    (let [m (prop/with-internal-proposal
              (gen [x & [y]] (trace :m (d/normal (+ x (or y 0.0)) 1)))
              (gen [sigma & _] (when (open? sigma :m) (trace :m (d/normal 0.0 2.0)))))
          tr (gfi/simulate m [0.0 1.0] (rng/key 8))
          v (ts/number (tr :m))]
      (is (ts/close? (- (* -0.5 v v) (* -0.5 (* (- v 1.0) (- v 1.0))))
                     (:weight (gfi/update tr [0.0] [diff/no-change] cm/EMPTY (rng/key 9))))))))

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

(deftest a-change-in-the-number-of-arguments-is-regenerated
  ;; the post-M5 review's P1-02, found by its verifier: regenerate-under
  ;; kept the trace, old arguments and all, when nothing was selected and
  ;; every argdiff was no-change, though there were fewer arguments.
  ;; Planted bug (2026-10-22): without the count check, both fail
  (let [model (gen [_x & more] (trace :z (d/normal 0.0 1.0)) (trace :y (d/normal (count more) 1.0)))
        pgf (prop/with-internal-proposal model (gen [sigma _x & _more] (when (open? sigma :z) (trace :z (d/normal 0.0 10.0)))))
        tr (:trace (gfi/generate pgf [0.0 1.0 2.0] (cm/choicemap :y 0.0) (rng/key 1)))
        r (gfi/regenerate tr [0.0] [diff/no-change] sel/none (rng/key 2))]
    (is (= [0.0] (:args (:trace r))))
    (is (ts/close? (- (dc/log-prob (d/normal 0.0 1.0) 0.0) (dc/log-prob (d/normal 2.0 1.0) 0.0)) (:weight r)))))

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
