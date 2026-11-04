(ns gen.inference.proposal-test
  "The internal-proposal override: the general GFI laws over models whose
   proposals make some, all, or a structure-deciding part of the open
   choices, and over models that splice one; exactness with the exact
   posterior as the proposal; the worked examples of pp. 143-144;
   regenerate's drops (Eq. 4.1); loud errors."
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
   ;; draws (genclj-nqsm: update-under, shared with with-exact-proposal)
   [:proposes-what-the-arguments-add proposes-what-n-adds [1] [[2] [3]]]
   ;; fewer arguments leave an unselected choice unvisited, which regenerate
   ;; drops (Eq. 4.1, genclj-bnlx); it threw before, so the laws with new
   ;; arguments said nothing about such a move
   [:drops-what-fewer-arguments-leave proposes-what-n-adds [3] [[1] [2]]]])

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

;; genclj-bnlx, found by the book's chapter 031: a selected :a that turns
;; false leaves the unselected :b unvisited, and regenerate threw where
;; Eq. 4.1 drops it (B ⊆ A_σ). The general laws over the thesis's example,
;; and the retained density ratio, whose densities the override's
;; projections are. Planted bugs (2026-11-04): the old loud error fails
;; regenerate-keeps-unselected-choices and the two ratio laws by exception;
;; the reverse move proposing given σ, the forward move's unselected
;; choices, in place of σ' = τ'|_{S^c} fails the ratio laws, with new
;; arguments too
(def structure-models
  [[:decides-the-structure (prop/with-internal-proposal abc abc-proposal) [] []]])

(laws/deflaws structure-deciding-internal-proposal structure-models 100
  (merge laws/general laws/fixed-support
         (select-keys laws/forward-sampling [:regenerate-weight-is-the-retained-density-ratio])))

(def reads-what-it-drops
  "abc-proposal, but a's proposal reads b: Bern(0.9) when σ gives b, else
   Bern(0.6). Regenerating :a alone of {a T b F c T}, q proposes given b,
   which the move to a = F drops; the reverse move proposes given
   σ' = {c T}."
  (gen [sigma] (let [a (if (open? sigma :a) (trace :a (d/bernoulli (if (open? sigma :b) 0.6 0.9))) (given sigma :a))]
                 (when (and a (open? sigma :b)) (trace :b (d/bernoulli 0.5)))
                 (when (open? sigma :c) (trace :c (d/bernoulli 0.5))))))

(defn- regenerate-outcomes
  "{outcome [weight …]} of regenerating :a alone of the trace of pgf at tau
   over 40 keys: an outcome is [a] or [a b]."
  [pgf tau seed]
  (let [tr (:trace (gfi/generate pgf [] (cm/from-map tau) (rng/key 0)))
        outcome (fn [t] (if (ts/host (t :a)) [true (ts/host (t :b))] [false]))]
    (reduce (fn [m {t :trace w :weight}] (update m (outcome t) (fnil conj []) (ts/number w)))
            {} (map #(gfi/regenerate tr (sel/select :a) %) (rng/split (rng/key seed) 40)))))

(deftest regenerate-drops-what-the-new-run-leaves
  ;; Eq. 4.1: w = p(τ'; x') q(υ'; x, σ') / (p(τ; x) q(υ; x', σ)),
  ;; σ = τ|_{S^c} and σ' = τ'|_{S^c}, with S = {a}. Planted bugs
  ;; (2026-11-04): the old loud error fails every case; the reverse move
  ;; given σ fails the drops; given σ' restricted to τ's choices (the
  ;; projections' form), or the forward move given σ less the choices it
  ;; drops (a revise's), fails reads-what-it-drops alone: its q reads b
  (let [log #(Math/log %)
        check (fn [pgf tau seed expected]
                (let [outcomes (regenerate-outcomes pgf tau seed)]
                  (is (= (set (keys expected)) (set (keys outcomes))) (str tau ": every outcome drawn"))
                  (doseq [[o ws] outcomes, w ws]
                    (is (ts/close? (expected o) w) (str tau " to " o)))))]
    (testing "the thesis's q: a move to a = F drops b, with p. 144's weight"
      (let [pgf (prop/with-internal-proposal abc abc-proposal)]
        (check pgf {:a true :b false :c true} 11
               {[false] (log (/ (* 0.05 0.3) (* 0.2 0.4))) [true false] 0.0})
        ;; the reverse moves: q's fresh b is unselected, so σ' holds it
        (check pgf {:a false :c true} 12
               {[false] 0.0
                [true false] (log (/ (* 0.2 0.4) (* 0.05 0.3)))
                [true true] (log (/ (* 0.125 0.4) (* 0.05 0.3)))})))
    (testing "a q that reads the choice the move drops: q(υ; σ) given b, q(υ'; σ') without it"
      (let [pgf (prop/with-internal-proposal abc reads-what-it-drops)]
        (check pgf {:a true :b false :c true} 13
               {[false] (log (/ (* 0.05 0.3) (* 0.2 0.1))) [true false] 0.0})
        (check pgf {:a false :c true} 14
               {[false] 0.0
                [true false] (log (/ (* 0.2 0.1) (* 0.05 0.3)))
                [true true] (log (/ (* 0.125 0.1) (* 0.05 0.3)))})))))

(deftest ^:heavy mh-that-drops-a-choice-keeps-the-posterior
  ;; Eq. 4.1's weight makes selection MH on :a alone stationary for the
  ;; posterior given c = T, {a F}: 0.05, {a T b F}: 0.2, {a T b T}: 0.125
  ;; over 0.375, whatever the proposal reads: from 4000 exact posterior
  ;; draws, one move each under reads-what-it-drops leaves each state's
  ;; frequency within 5 standard errors. Planted bug (2026-11-04): the
  ;; forward move proposing given σ less the choices it drops (a revise's
  ;; rerun) gives {a F} a frequency of 0.082 for 0.133, 9 standard errors
  ;; off. (The reverse move given σ' restricted to τ's choices moves the
  ;; frequencies by about one standard error: that plant is
  ;; regenerate-drops-what-the-new-run-leaves's to catch.)
  (let [pgf (prop/with-internal-proposal abc reads-what-it-drops)
        states [{:a false :c true} {:a true :b false :c true} {:a true :b true :c true}]
        posterior (mapv #(clojure.core// % 0.375) [0.05 0.2 0.125])
        n 4000
        moved (mapv (fn [k] (let [[k0 k1 k2] (rng/split k 3)
                                  start (states (long (ts/number (dc/sample (d/categorical posterior) k0))))
                                  tr (:trace (gfi/generate pgf [] (cm/from-map start) k1))
                                  t (:trace (mh/mh tr (sel/select :a) k2))]
                              (ts/host (cm/to-map (:choices t)))))
                    (rng/split (rng/key 15) n))
        counts (frequencies moved)]
    (doseq [[state p] (map vector states posterior)]
      (is (< (Math/abs (clojure.core/- (clojure.core// (get counts state 0) n) p))
             (clojure.core/* 5 (Math/sqrt (clojure.core// (clojure.core/* p (clojure.core/- 1 p)) n))))
          (str state ": " (get counts state 0) " of " n)))))

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
  ;; b's proposal is its prior, so every move weighs 0 (Eq. 4.1). A new b
  ;; that drops :x was a loud error before genclj-bnlx. Planted bug
  ;; (2026-11-04): the reverse move given σ, :x included, weighs the drop
  ;; -log N(0.4; 0, 1)
  (let [pgf (prop/with-internal-proposal gated propose-b)
        tr (:trace (gfi/generate pgf [] (cm/choicemap :b true :x 0.4 :y 1.0) (rng/key 1)))
        outcomes (map #(let [{t' :trace w :weight} (gfi/regenerate tr (sel/select :b) %)]
                         [(ts/host (t' :b)) (:choices t') w])
                      (rng/split (rng/key 2) 12))]
    (testing "a new b that keeps :x keeps its value"
      (is (seq (filter first outcomes)))
      (doseq [[_ choices w] (filter first outcomes)]
        (is (ts/close? 0.4 (cm/get-choice choices :x)) "the unselected :x is kept")
        (is (ts/close? 0.0 w))))
    (testing "a new b that drops :x drops it, and keeps :y"
      (is (seq (remove first outcomes)))
      (doseq [[_ choices w] (remove first outcomes)]
        (is (= #{[:b] [:y]} (set (cm/addresses choices))))
        (is (ts/close? 1.0 (cm/get-choice choices :y)))
        (is (ts/close? 0.0 w))))
    (testing "selecting what b decides works"
      (let [{t' :trace} (gfi/regenerate tr (sel/select :b :x) (rng/key 3))]
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
