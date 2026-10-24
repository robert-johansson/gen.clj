(ns gen.encapsulated-test
  "Encapsulated randomness (thesis §4.5): untraced draws in gen bodies and
   estimated densities (gen.custom/estimated).
   - The laws that hold for one draw of ω (gen.laws/encapsulated; the
     general set's assess against generate on the same key).
   - ω is kept unless the change hints say something changed, and untraced
     draws never shift a site's key.
   - The statistical laws: Eq. 4.3 (E_q̊ ξ = p) and Eq. 4.4 (E_p̊ 1/ξ =
     1/p, through simulate); a body's untraced draws are independent;
     pseudo-marginal MH reaches the exact posterior (Eqs. 4.15-4.16), with
     untraced draws, with an importance-sampling likelihood and with a
     particle filter's (p. 160); importance sampling with an encapsulated
     proposal is unbiased (§4.5.4).
   - Loud misuse, and operations that need exact densities (gradients,
     elliptical slice, exact and replayed proposals) refuse encapsulated
     randomness rather than fix one draw of it silently."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as is]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.transform :as tf]
            [gen.inference.proposal :as prop]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

;; host statistics (gen.array's operators would make arrays on MLX)

(defn- mean-sd [xs]
  (let [n (count xs)
        m (clojure.core// (reduce clojure.core/+ xs) n)
        v (clojure.core// (reduce clojure.core/+ (map #(let [e (clojure.core/- % m)] (clojure.core/* e e)) xs)) (dec n))]
    [m (Math/sqrt v)]))

(defn- npdf [x m s]
  (let [z (clojure.core// (clojure.core/- x m) s)]
    (clojure.core// (Math/exp (clojure.core/* -0.5 z z)) (clojure.core/* s (Math/sqrt (clojure.core/* 2 Math/PI))))))

(defn- within?
  "Is the sample mean of xs within 5 standard errors of mu?"
  [xs mu]
  (let [[m s] (mean-sd xs)]
    (< (Math/abs (clojure.core/- m mu)) (clojure.core/* 5 (clojure.core// s (Math/sqrt (count xs)))))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; Eqs. 4.8-4.10: a mixture of normals whose component index is encapsulated,
;; so ξ = N(a; μ_i, 1) for the index drawn
(def mix-weights [0.2 0.5 0.3])
(def mix-means [-2.0 0.0 3.0])

(def mixture
  (gen []
    (let [i (untraced (d/categorical mix-weights))]
      (trace :a (d/normal (array/take (array/array mix-means) i) 1.0)))))

(defn- p-mixture [a] (reduce clojure.core/+ (map #(clojure.core/* %1 (npdf a %2 1.0)) mix-weights mix-means)))

;; for Eq. 4.4, components that overlap: g(a)/ξ = N(a; 0, 0.5)/N(a; μ_i, 1)
;; is at most about 3.9, so the sample mean is near normal (with the
;; mixture above, rare ratios in the hundreds carry the mean, and a sample
;; of 4000 missed them on Metal: z = -6.2)
(def overlapping
  (gen []
    (let [i (untraced (d/categorical [0.3 0.4 0.3]))]
      (trace :a (d/normal (array/take (array/array [-1.0 0.0 1.0]) i) 1.0)))))

;; Eqs. 4.11-4.14: black-box stochastic code between two traced choices
(defn- simulator
  "Stochastic code with a key: a value in [0, 1.3) that a raises."
  [key a]
  (+ (rng/uniform key) (array/where a 0.3 0.0)))

(def with-simulator
  (gen [p]
    (let [a (trace :a (d/bernoulli p))
          result (simulator (untraced-key) a)]
      (trace :b (d/bernoulli (array/where (array/> result 0.6) 0.1 0.9))))))

(def noisy-kernel (gen [x] (trace :y (d/normal (+ x (untraced (d/normal 0 0.5))) 1))))

(def noisy-step (gen [_t s] (trace :s (d/normal (+ (* 0.9 s) (untraced (d/normal 0 0.5))) 1))))

(def spliced
  (gen [m]
    (let [mu (trace :mu (d/normal m 1))]
      (splice :inner with-simulator 0.4)
      (trace :y (d/normal (+ mu (untraced (d/normal 0 1))) 1)))))

;; two untraced draws in one body, the second from ω folded with 1, which a
;; body of one draw never reaches (law audit E21). Planted bug (2026-10-20):
;; assess folding every draw with 0 passes every law over the other models,
;; and fails assess-is-generate and the same-key score over this one
(def two-draws
  (gen [m] (let [z1 (untraced (d/normal 0 1))
                 z2 (untraced (d/normal 0 1))
                 x (trace :x (d/normal (+ m z1) 1))]
             (trace :y (d/normal (- x z2) 1)))))

;; ω in a dynamic body whose update makes fresh sites (law audit, finding 6):
;; the encapsulated update law saw fresh choices only in Map and Unfold, so a
;; handler update that adds a fresh site's log density instead of leaving
;; -log q out (C11) passed it. Planted bug (2026-10-20): C11 fails
;; update-weight-is-the-score-change-less-fresh over this model
(def growing-noisy
  (gen [n] (let [e (untraced (d/normal 0 0.5))]
             (doseq [i (range n)] (trace [:y i] (d/normal e 1))))))

(def untraced-models
  [["mixture" mixture [] []]
   ["simulator" with-simulator [0.4] [[0.7]]]
   ;; growing alternatives too, so that updates make fresh choices (law audit gap 5)
   ["map" (comb/map noisy-kernel) [[0.1 0.2 0.3]] [[[0.1 0.5 0.3]] [[0.1 0.2]] [[0.1 0.2 0.3 0.4]]]]
   ["unfold" (comb/unfold noisy-step) [3 0.0] [[2 0.0] [3 0.5] [5 0.0]]]
   ["spliced" spliced [0.0] [[1.0]]]
   ["two-draws" two-draws [0.0] [[1.0]]]
   ["growing-noisy" growing-noisy [2] [[3] [4]]]])

(laws/deflaws untraced untraced-models 100 laws/encapsulated)

;; An estimated density: y given θ, with y ~ N(θ + z, 1) and z ~ N(0, 1), so
;; p(y; θ) = N(y; θ, √2), estimated by importance sampling over z from its
;; prior. simulate draws y from its marginal and the samples given y: one is
;; y's own latent z, at a uniform position, the others from the prior.
(def n-importance 5)

(defn- is-log-estimate [theta y zs]
  (- (array/logsumexp (dc/log-prob (d/normal (+ theta zs) 1.0) y)) (Math/log n-importance)))

(def likelihood
  (custom/estimated
   {:addresses [:y]
    :estimate (fn [[theta] choices key]
                (is-log-estimate theta (cm/get-choice choices :y) (dc/sample (d/normal 0 1) key [n-importance])))
    :simulate (fn [[theta] key]
                (let [[k-z k-y k-j k-zs] (rng/split key 4)
                      z (dc/sample (d/normal 0 1) k-z [])
                      y (dc/sample (d/normal (+ theta z) 1) k-y [])
                      j (rng/randint k-j 0 n-importance)
                      zs (array/where (array/== (array/arange n-importance) j) z (dc/sample (d/normal 0 1) k-zs [n-importance]))]
                  {:choices (cm/choicemap :y y) :log-estimate (is-log-estimate theta y zs)}))}))

(def pm-model
  "θ ~ N(0, 1), and y given θ by the estimated likelihood."
  (gen [] (let [theta (trace :theta (d/normal 0 1))] (splice :y likelihood theta) theta)))

;; an estimator has no proposal density, so regenerate resamples its choices
;; all at once (its joint sampler) or not at all; these have one each. The
;; joint sampler draws ω otherwise than the estimate: no same-key score
(def estimated-laws
  (dissoc laws/encapsulated :score-is-assessed-on-the-same-key))

(laws/deflaws estimated [["likelihood" likelihood [0.5] [[1.5]]] ["pm-model" pm-model [] []]] 60 estimated-laws)

;; ---------------------------------------------------------------------------
;; The keep-or-redraw rule
;; ---------------------------------------------------------------------------

(defn- omega [tr] (:gen.trace/omega tr))

(deftest omega-is-kept-unless-something-changes
  (let [tr (gfi/simulate mixture [] (rng/key 1))]
    (is (some? (omega tr)))
    (testing "an identity update and regenerating nothing keep it, with weight 0"
      (doseq [r [(gfi/update tr cm/EMPTY (rng/key 2)) (gfi/regenerate tr sel/none (rng/key 3))]]
        (is (= (omega tr) (omega (:trace r))))
        (is (== 0.0 (ts/number (:weight r))))))
    (testing "a constraint redraws it, and the weight is the change in the estimate"
      (let [{t :trace w :weight} (gfi/update tr (cm/choicemap :a 0.5) (rng/key 4))]
        (is (not= (omega tr) (omega t)))
        (is (ts/close? w (- (:score t) (:score tr))))))
    (testing "so do changed arguments, with no constraint"
      (let [tr (gfi/simulate with-simulator [0.4] (rng/key 5))
            {t :trace} (gfi/update tr [0.7] [gen.diff/unknown-change] cm/EMPTY (rng/key 6))]
        (is (not= (omega tr) (omega t))))))
  (testing "also under a parameter store (review 3: nothing tested a store's skipping)"
    (let [gf (gen.dynamic/with-params (gen [] (let [e (untraced (d/normal 0 1))] (trace :a (d/normal (+ e (param :w)) 1)))) {:w 1.0})
          tr (gfi/simulate gf [] (rng/key 9))
          r (gfi/update tr cm/EMPTY (rng/key 10))]
      (is (= (omega tr) (omega (:trace r))))
      (is (== 0.0 (ts/number (:weight r))))))
  (testing "a changed store or gen-fn at the call redraws it, in a dynamic callee too (review 5: a revise)"
    (let [noisy (gen [] (let [e (untraced (d/normal 0 1))] (trace :x (d/normal (+ e (param :w)) 1))))
          model (gen [] (let [a (trace :a (d/uniform 0 1))] (splice :c (gen.dynamic/with-params noisy {:w (* 10.0 a)}))))
          callee-omega #(omega (get (:gen.dynamic/subtraces %) [:c]))
          tr (gfi/simulate model [] (rng/key 11))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 12))]
      (is (some? (callee-omega tr)))
      (is (not= (callee-omega tr) (callee-omega t)))
      (is (ts/close? w (- (:score t) (:score tr))))
      (let [tr (gfi/simulate (gen.dynamic/with-params noisy {:w 1.0}) [] (rng/key 13))
            {t :trace w :weight} (gfi/update (assoc tr :gen-fn (gen.dynamic/with-params noisy {:w 5.0})) cm/EMPTY (rng/key 14))]
        (is (not= (omega tr) (omega t)))
        (is (ts/close? w (- (:score t) (:score tr)))))))
  (testing "a Map's unchanged elements keep theirs"
    (let [m (comb/map noisy-kernel)
          tr (gfi/simulate m [[0.1 0.2 0.3]] (rng/key 7))
          {t :trace} (gfi/update tr (cm/choicemap [1 :y] 0.0) (rng/key 8))
          omegas #(mapv omega (:gen.combinators/subtraces %))]
      (is (= ((omegas tr) 0) ((omegas t) 0)))
      (is (not= ((omegas tr) 1) ((omegas t) 1)))
      (is (= ((omegas tr) 2) ((omegas t) 2))))))

(deftest untraced-draws-leave-the-sites-keys-alone
  (let [plain (gen [] (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1)))
        noisy (gen [] (trace :x (d/normal 0 1)) (untraced (d/normal 0 1)) (untraced-key) (trace :y (d/normal 0 1)))
        k (rng/key 9)]
    (is (= (ts/host (:choices (gfi/simulate plain [] k))) (ts/host (:choices (gfi/simulate noisy [] k)))))
    (is (nil? (omega (gfi/simulate plain [] k))))))

(deftest an-estimate-is-run-again-when-the-number-of-arguments-changes
  ;; the post-M5 review's P1-02 (DESIGN.md §9, G3): an estimated density
  ;; kept its trace, old arguments and estimate, on no-change argdiffs for
  ;; fewer arguments. Planted bug (2026-10-22): its unchanged? without the
  ;; count check fails all four
  (let [lp #(dc/log-prob (d/normal (count %1) 1.0) %2)
        by-count (custom/estimated {:addresses [:y] :estimate (fn [args choices _key] (lp args (cm/get-choice choices :y)))})
        tr (:trace (gfi/generate by-count [1.0 2.0] (cm/choicemap :y 0.0) (rng/key 1)))
        expected (- (lp [1.0] 0.0) (lp [1.0 2.0] 0.0))]
    (doseq [[op r] [[:update (gfi/update tr [1.0] [gen.diff/no-change] cm/EMPTY (rng/key 2))]
                    [:regenerate (gfi/regenerate tr [1.0] [gen.diff/no-change] sel/none (rng/key 3))]]]
      (testing (name op)
        (is (= [1.0] (:args (:trace r))))
        (is (ts/close? expected (:weight r)))))))

(deftest an-estimated-density-falls-back-to-particles
  ;; for the reason it has (review, 2026-10-20: the stated one was a spot
  ;; check's), through simulate and through generate, which estimates (the
  ;; confirmation review: it had no test)
  (let [reason #(:message (batched/fallback-reason %))
        constraints (cm/choicemap [:y :y] 1.5)]
    (binding [batched/*spot-checks* 0]
      (is (= "An estimated density in a batched run" (reason (batched/simulate likelihood [0.4] 4 (rng/key 50)))))
      (is (= "An estimated density in a batched run"
             (reason (:particles (batched/generate pm-model [] constraints 4 (rng/key 51)))))))))

(def noisy-normal (gen [] (let [mu (trace :mu (d/normal 0 1)) e (untraced (d/normal 0 0.3))] (trace :y (d/normal (+ mu e) 1)) mu)))

;; an override over a model with ω: assess draws what generate draws on the
;; same key (review, 2026-10-20: it gave the model another key)
(laws/deflaws omega-override
  [[:override (gen.inference.proposal/with-internal-proposal
               noisy-normal (gen [sigma] (when-not (cm/has-value? (cm/get-submap sigma :mu)) (trace :mu (d/normal 0.5 1)))))
    [] []]]
  60 {:assess-is-generate-with-every-choice laws/assess-is-generate-with-every-choice})

(deftest a-batched-run-falls-back-to-particles
  ;; each particle's ω must reproduce its own draws
  (let [k (rng/key 10)
        ps (batched/simulate mixture [] 6 k)]
    (is (re-find #"untraced" (:message (batched/fallback-reason ps))))
    (is (= (mapv #(ts/host (% :a)) (batched/traces ps))
           (mapv #(ts/host ((gfi/simulate mixture [] %) :a)) (rng/split k 6))))))

;; ---------------------------------------------------------------------------
;; Statistical laws
;; ---------------------------------------------------------------------------

(deftest ^:slow ^:heavy eq-4-3-the-estimate-is-unbiased
  (testing "untraced draws: the mixture"
    (let [xis (mapv #(Math/exp (ts/number (:weight (gfi/assess mixture [] (cm/choicemap :a 0.5) %))))
                    (rng/split (rng/key 11) 4000))]
      (is (within? xis (p-mixture 0.5)))))
  (testing "an estimated density"
    (let [xis (mapv #(Math/exp (ts/number (:weight (gfi/assess likelihood [0.4] (cm/choicemap :y 1.0) %))))
                    (rng/split (rng/key 12) 4000))]
      (is (within? xis (npdf 1.0 0.4 (Math/sqrt 2.0)))))))

(deftest ^:slow ^:heavy eq-4-4-simulate-draws-omega-given-the-choices
  ;; E[g(τ)/ξ] = ∫ g = 1 under simulate's (τ, ω) ~ p(τ) p̊(ω; τ). For the
  ;; overlapping mixture ξ is N(a; μ_i, 1) for the drawn i, and g is
  ;; narrower than every component, so g/ξ is bounded (≤ 2·e^(2/3) ≈ 3.9).
  ;; For the estimator ξ is the mean of five likelihood terms, one at the
  ;; latent that produced y and four at prior draws: g/ξ is unbounded, and
  ;; its second moment is finite only through the four prior terms (the
  ;; latent's term alone gives 1/ξ an infinite one). Planted bug (2026-10-20): a joint
  ;; sampler drawing ω apart from τ moves the mixture's mean by about 0.2,
  ;; against a band of about 0.07
  (testing "untraced draws: overlapping components"
    (let [rs (mapv (fn [k] (let [tr (gfi/simulate overlapping [] k)]
                             (clojure.core// (npdf (ts/number (tr :a)) 0.0 0.5) (Math/exp (ts/number (:score tr))))))
                   (rng/split (rng/key 13) 4000))]
      (is (within? rs 1.0))))
  (testing "an estimated density's joint sampler"
    (let [rs (mapv (fn [k] (let [tr (gfi/simulate likelihood [0.4] k)]
                             (clojure.core// (npdf (ts/number (tr :y)) 0.4 0.5) (Math/exp (ts/number (:score tr))))))
                   (rng/split (rng/key 14) 4000))]
      (is (within? rs 1.0)))))

(deftest ^:slow ^:heavy eq-4-3-on-black-box-code-and-combinators
  ;; law audit gap 2: the exact marginals of the thesis's black-box example
  ;; (Eqs. 4.11-4.14) and of untraced Map and Unfold elements, at residuals of
  ;; one sign (with mixed signs a shared-key bug moved the mean by 2.8 se only)
  (let [mean-xi (fn [gf args choices seed] (mapv #(Math/exp (ts/number (:weight (gfi/assess gf args choices %))))
                                                  (rng/split (rng/key seed) 4000)))
        n3 (Math/pow (npdf 1.5 0.0 (Math/sqrt 1.25)) 3)]
    (testing "a simulator between two choices: p(a, b) with b given a through U + 0.3a"
      (is (within? (mean-xi with-simulator [0.4] (cm/choicemap :a true :b true) 60) (* 0.4 0.34)))
      (is (within? (mean-xi with-simulator [0.4] (cm/choicemap :a false :b true) 61) (* 0.6 0.58))))
    (testing "a Map of untraced kernels: N(y; x, √1.25) per element"
      (is (within? (mean-xi (comb/map noisy-kernel) [[0.1 0.2 0.3]]
                            (cm/choicemap [0 :y] 1.6 [1 :y] 1.7 [2 :y] 1.8) 62)
                   n3)))
    (testing "an Unfold of untraced steps: each innovation 1.5"
      (is (within? (mean-xi (comb/unfold noisy-step) [3 0.0]
                            (cm/choicemap [0 :s] 1.5 [1 :s] (+ (* 0.9 1.5) 1.5) [2 :s] (+ (* 0.9 2.85) 1.5)) 63)
                   n3)))
    ;; law audit, finding 9. Planted bug (2026-10-20): a splice that gives its
    ;; callee the caller's key, so both bodies draw one ω (the callee's
    ;; uniform is then the quantile of the caller's normal), gives 0.0152
    ;; against 0.0085, a band of 0.0011
    (testing "a splice: the callee's ω is its own, E[ξ] = N(μ; m, 1) p(a, b) N(y; μ, √2)"
      (is (within? (mean-xi spliced [0.0] (cm/choicemap :mu 0.2 [:inner :a] true [:inner :b] true :y -1.3) 69)
                   (* (npdf 0.2 0.0 1.0) 0.4 0.34 (npdf -1.3 0.2 (Math/sqrt 2.0))))))))

(deftest ^:heavy untraced-draws-in-a-body-are-independent
  ;; §4.5.2 (law audit E21): a body's untraced draws are its black-box code's
  ;; own randomness, the n-th from ω folded with n, so two of them are
  ;; independent. Planted bug (the audit's B2, 2026-10-20): every draw from ω
  ;; itself, another model's randomness consistently, so every pointwise law
  ;; passes; here the correlation is then 1.0, and E[ξ] 0.1295 against
  ;; 0.1583, 12 standard errors off (z1 + z2 in place of the difference moved
  ;; it by 4.7 only)
  (testing "two draws are uncorrelated: the sample correlation within 5/√n of 0"
    (let [two (gen [] [(untraced (d/normal 0 1)) (untraced (d/normal 0 1))])
          n 2000
          pairs (mapv #(mapv ts/number (:retval (gfi/simulate two [] %))) (rng/split (rng/key 4242) n))]
      (is (< (Math/abs (ts/correlation pairs)) (clojure.core// 5.0 (Math/sqrt n))))))
  (testing "Eq. 4.3 with y ~ N(z1 - z2, 1): E[ξ] = N(1.5; 0, √3)"
    (let [m (gen [] (let [z1 (untraced (d/normal 0 1)) z2 (untraced (d/normal 0 1))] (trace :y (d/normal (- z1 z2) 1))))
          xis (mapv #(Math/exp (ts/number (:weight (gfi/assess m [] (cm/choicemap :y 1.5) %)))) (rng/split (rng/key 4243) 4000))]
      (is (within? xis (npdf 1.5 0.0 (Math/sqrt 3.0)))))))

(def theta-z-y
  "θ ~ N(0, 1), z untraced N(0, 1), y ~ N(θ + z, 1): p(y) = N(y; 0, √3), and
   θ given y = 1.5 is N(0.5, √(2/3))."
  (gen [] (let [theta (trace :theta (d/normal 0 1))
                z (untraced (d/normal 0 1))]
            (trace :y (d/normal (+ theta z) 1))
            theta)))

(deftest ^:slow ^:heavy pseudo-marginal-importance-sampling-is-unbiased
  ;; law audit gap 4: generate with some constraints on a model with
  ;; untraced draws, E[exp w] = p(y)
  (let [ws (mapv #(Math/exp (ts/number (:weight (gfi/generate theta-z-y [] (cm/choicemap :y 1.5) %))))
                 (rng/split (rng/key 64) 4000))]
    (is (within? ws (npdf 1.5 0.0 (Math/sqrt 3.0))))))

(defn- pmmh-thetas
  "θ after steps selection-MH moves on θ, in n chains of model from the
   prior given observations."
  [model observations n steps key]
  (mapv (fn [k]
          (let [[k0 & ks] (rng/split k (inc steps))]
            (ts/number ((reduce (fn [tr k] (:trace (mh/mh tr (sel/select :theta) k)))
                                (:trace (gfi/generate model [] observations k0))
                                ks)
                        :theta))))
        (rng/split key n)))

(defn- normal-ok?
  "Are the θs N(mu, √v)? Mean within 5 standard errors, variance within 5
   of its standard errors."
  [thetas mu v]
  (let [n (count thetas)
        [m s] (mean-sd thetas)]
    (and (< (Math/abs (clojure.core/- m mu)) (clojure.core/* 5 (Math/sqrt (clojure.core// v n))))
         (< (Math/abs (clojure.core/- (clojure.core/* s s) v)) (clojure.core/* 5 v (Math/sqrt (clojure.core// 2.0 (dec n))))))))

(defn- posterior-ok?
  "Are the θs N(y/3, √(2/3)) (θ ~ N(0, 1), y ~ N(θ, √2))?"
  [thetas y]
  (normal-ok? thetas (clojure.core// y 3.0) (clojure.core// 2.0 3.0)))

;; Planted bugs (2026-10-20), each caught on the JVM backend:
;; - keeping ω when θ changes (keeps-omega? true for regenerate): each chain
;;   then targets the posterior given its first z, mean y/2 = 0.75 on
;;   average against y/3 = 0.5, and the untraced case fails;
;; - an estimated density keeping its estimate when its argument changes:
;;   the estimated case fails;
;; - never keeping ω: the identity laws and omega-is-kept-… fail;
;; - assess ignoring its key: assess-is-generate-… and Eq. 4.3 fail;
;; - untraced draws taking the sites' keys: untraced-draws-leave-… fails.
(deftest ^:slow ^:heavy pseudo-marginal-mh-targets-the-exact-posterior
  (testing "untraced draws: y ~ N(θ + z, 1) with z untraced, one-sample estimates"
    (let [model (gen [] (let [theta (trace :theta (d/normal 0 1))
                              z (untraced (d/normal 0 1))]
                          (trace :y (d/normal (+ theta z) 1))
                          theta))]
      (is (posterior-ok? (pmmh-thetas model (cm/choicemap :y 1.5) 2000 150 (rng/key 15)) 1.5))))
  (testing "an estimated likelihood"
    (is (posterior-ok? (pmmh-thetas pm-model (cm/choicemap [:y :y] 1.5) 2000 100 (rng/key 16)) 1.5)))
  ;; planted bug (audit, 2026-10-20): an update that keeps ω under a
  ;; constraint gives mean 0.74
  (testing "through update: MH with a random-walk proposal, Eq. 4.16 (law audit gap 3)"
    (let [walk (gen [t] (trace :theta (d/normal (t :theta) 0.8)))
          thetas (mapv (fn [k] (let [[k0 & ks] (rng/split k 151)]
                                 (ts/number ((reduce (fn [tr k] (:trace (mh/mh tr walk [] k)))
                                                     (:trace (gfi/generate theta-z-y [] (cm/choicemap :y 1.5) k0)) ks)
                                             :theta))))
                       (rng/split (rng/key 65) 2000))]
      (is (posterior-ok? thetas 1.5)))))

;; PMMH with an SMC estimator (thesis p. 160, law audit E17): θ ~ N(0, 1), and
;; y_0..y_2 given θ from a linear-Gaussian state-space model, y_t ~ N(x_t + θ,
;; 0.5) with x_t ~ N(0.8 x_{t-1}, 1), x_{-1} = 0. The likelihood is a particle
;; filter's estimate on y - θ (10 particles, resampled at every step), so the
;; estimate carries the log marginal likelihood across resamplings
(def ssm (comb/unfold (gen [_t prev] (let [x (trace :x (d/normal (* 0.8 prev) 1))] (trace :y (d/normal x 0.5)) x))))

(def ssm-ys [0.3 1.1 -0.4])

(defn- pf-log-estimate [theta ys key]
  (let [ks (rng/split key (* 2 (count ys)))
        obs (fn [t] (cm/choicemap [t :y] (- (ys t) theta)))]
    (pf/log-ml-estimate
     (reduce (fn [state t]
               (-> state
                   (pf/resample {:key (ks (dec (* 2 t)))})
                   (pf/step [(inc t) 0.0] [gen.diff/unknown-change gen.diff/no-change] (obs t) {:key (ks (* 2 t))})))
             (pf/initialize ssm [1 0.0] (obs 0) 10 {:key (ks 0)})
             (range 1 (count ys))))))

(def smc-likelihood
  (custom/estimated
   {:addresses (mapv #(vector :y %) (range (count ssm-ys)))
    :estimate (fn [[theta] choices key]
                (pf-log-estimate theta (mapv #(cm/get-choice choices [:y %]) (range (count ssm-ys))) key))}))

(def smc-model
  (gen [] (let [theta (trace :theta (d/normal 0 1))] (splice :ys smc-likelihood theta) theta)))

(defn- ssm-log-likelihood
  "log p(ys | θ) by the Kalman filter, on the host."
  [ys theta]
  (first (reduce (fn [[lml m p] y]
                   (let [m- (clojure.core/* 0.8 m)
                         p- (clojure.core/+ (clojure.core/* 0.64 p) 1.0)
                         s (clojure.core/+ p- 0.25)
                         k (clojure.core// p- s)
                         e (clojure.core/- y theta m-)]
                     [(clojure.core/+ lml (Math/log (npdf e 0.0 (Math/sqrt s))))
                      (clojure.core/+ m- (clojure.core/* k e))
                      (clojure.core/* (clojure.core/- 1.0 k) p-)]))
                 [0.0 0.0 0.0] ys)))

(deftest ^:slow ^:heavy pmmh-with-a-particle-filter-estimator-targets-the-exact-posterior
  ;; log p(y | θ) is quadratic in θ, so its values at -1, 0, 1 give the
  ;; Gaussian posterior: precision 1 + A, mean b/(1 + A). 1000 chains of 40
  ;; selection-MH moves from the prior; mean and variance within 5 of their
  ;; standard errors (measured 0.164 and 0.526 against 0.138 and 0.515, bands
  ;; 0.114 and 0.115; the log estimate's sd is 0.72). Planted bugs
  ;; (2026-10-20): the filter's estimate without the log marginal likelihood
  ;; its resamplings carried (the last step's alone) gives -0.197 and 0.934;
  ;; an estimated density that keeps its estimate when θ changes gives 0.048
  ;; and 0.932, about the prior
  (let [[l- l0 l+] (map #(ssm-log-likelihood ssm-ys %) [-1.0 0.0 1.0])
        precision (clojure.core/- 1.0 (clojure.core/+ l+ l- (clojure.core/* -2.0 l0)))
        mu (clojure.core// (clojure.core// (clojure.core/- l+ l-) 2.0) precision)
        obs (apply cm/choicemap (mapcat (fn [t y] [[:ys :y t] y]) (range) ssm-ys))
        thetas (pmmh-thetas smc-model obs 1000 40 (rng/key 70))]
    (is (normal-ok? thetas mu (clojure.core// 1.0 precision)) (str (mean-sd thetas) " against " mu))))

(deftest ^:slow ^:heavy involutive-mcmc-is-pseudo-marginal
  ;; the involution's update takes the move's key, so it redraws ω, a
  ;; pseudo-marginal move (review, 2026-10-20: with a fixed key the chain
  ;; sampled the posterior given one z, mean 1.07 against 0.5). A
  ;; volume-preserving involution: (θ, u) to (θ + u, -u)
  (let [shift (tf/involution (tf/involutive (tf/transform [mi ai mo ao]
                                              (let [u (tf/read ai :u :continuous)]
                                                (tf/write mo :theta (+ (tf/read mi :theta :continuous) u) :continuous)
                                                (tf/write ao :u (array/negate u) :continuous)))))
        q (gen [_] (trace :u (d/normal 0 0.8)))
        thetas (mapv (fn [k] (let [[k0 & ks] (rng/split k 151)]
                               (ts/number ((reduce (fn [tr k] (:trace (mh/involutive-mh tr q [] shift k)))
                                                   (:trace (gfi/generate theta-z-y [] (cm/choicemap :y 1.5) k0)) ks)
                                           :theta))))
                     (rng/split (rng/key 68) 2000))]
    (is (posterior-ok? thetas 1.5))))

(deftest ^:slow ^:heavy mh-with-an-encapsulated-proposal-keeps-the-posterior
  ;; §4.5.4, law audit gap 4b: θ' ~ N(θ ± 0.6, 0.5), the sign an untraced
  ;; coin; from exact posterior draws (θ ~ N(0, 1), y ~ N(θ, 1), y = 1: N(0.5,
  ;; √0.5)), five moves keep the posterior. Planted bug (audit): the forward
  ;; score from a fresh ω instead of the proposing one gives variance 1.04
  (let [model (gen [] (let [theta (trace :theta (d/normal 0 1))] (trace :y (d/normal theta 1)) theta))
        q (gen [t] (trace :theta (d/normal (+ (t :theta) (array/where (untraced (d/bernoulli 0.5)) 0.6 -0.6)) 0.5)))
        n 4000
        starts (mapv #(clojure.core/+ 0.5 (clojure.core/* (Math/sqrt 0.5) %)) (ts/host (dc/sample (d/normal 0 1) (rng/key 66) [n])))
        thetas (mapv (fn [theta k] (let [[k0 & ks] (rng/split k 6)]
                                     (ts/number ((reduce (fn [tr k] (:trace (mh/mh tr q [] k)))
                                                         (:trace (gfi/generate model [] (cm/choicemap :theta theta :y 1.0) k0)) ks)
                                                 :theta))))
                     starts (rng/split (rng/key 67) n))
        [m s] (mean-sd thetas)]
    (is (< (Math/abs (clojure.core/- m 0.5)) (clojure.core/* 5 (Math/sqrt (clojure.core// 0.5 n)))))
    (is (< (Math/abs (clojure.core/- (clojure.core/* s s) 0.5)) (clojure.core/* 5 0.5 (Math/sqrt (clojure.core// 2.0 (dec n))))))))

(deftest ^:slow ^:heavy importance-sampling-with-an-encapsulated-proposal
  ;; §4.5.4: a proposal whose density is estimated keeps the marginal
  ;; likelihood estimate unbiased: E[p(θ, y)/ξ_q] = p(y) = N(y; 0, √2)
  (let [model (gen [] (let [theta (trace :theta (d/normal 0 1))] (trace :y (d/normal theta 1))))
        q (gen [] (trace :theta (d/normal (array/where (untraced (d/bernoulli 0.5)) 1.0 -1.0) 1)))
        obs (cm/choicemap :y 1.0)
        ws (mapv (fn [k] (let [[k1 k2] (rng/split k)
                               {c :choices wq :weight} (gfi/propose q [] k1)]
                           (Math/exp (ts/number (- (:weight (gfi/generate model [] (cm/merge c obs) k2)) wq)))))
                 (rng/split (rng/key 17) 4000))
        [_ s] (mean-sd ws)
        exact (npdf 1.0 0.0 (Math/sqrt 2.0))]
    (is (within? ws exact))
    (testing "through importance-sampling with the proposal"
      (let [{lml :log-ml-estimate} (is/importance-sampling model [] obs 4000 {:proposal q :proposal-args [] :key (rng/key 18)})]
        (is (< (Math/abs (clojure.core/- (Math/exp (ts/number lml)) exact)) (clojure.core/* 5 (clojure.core// s (Math/sqrt 4000)))))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (testing "untraced draws from a distribution"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"untraced draws from a distribution"
                          (gfi/simulate (gen [] (untraced 3)) [] (rng/key 19)))))
  (testing "an untraced draw after the body returned"
    (let [f (:retval (gfi/simulate (gen [] (fn [] (untraced (d/normal 0 1)))) [] (rng/key 20)))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"after the generative function returned" (f)))))
  (testing "an estimated density"
    (let [no-sim (custom/estimated {:addresses [:y] :estimate (:estimate likelihood)})
          tr (gfi/simulate likelihood [0.4] (rng/key 21))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs :simulate" (gfi/simulate no-sim [0.4] (rng/key 22))))
      (testing "regenerate takes all of its choices (its joint sampler, weight 0) or none (review 5, R5-6)"
        (is (ts/close? 0.0 (:weight (gfi/regenerate tr (sel/select :y) (rng/key 23)))))
        (let [two (custom/estimated {:addresses [:y1 :y2] :estimate (fn [_ _ _] 0.0)})
              t2 (:trace (gfi/generate two [] (cm/choicemap :y1 1.0 :y2 2.0) (rng/key 24)))]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot resample some of the choices only"
                                (gfi/regenerate t2 (sel/select :y1) (rng/key 25))))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a function" (custom/estimated {:addresses [:y] :estimate 3})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #":addresses names its choices" (custom/estimated {:estimate (:estimate likelihood)})))
      (testing "choices other than all of them (law audit, 2026-10-20: a strict subset gave a trace missing a choice)"
        (let [two (custom/estimated {:addresses [:y1 :y2] :estimate (fn [_ _ _] 0.0)})]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs every choice of the estimator"
                                (gfi/generate two [] (cm/choicemap :y1 1.0) (rng/key 32))))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[:y3\] was not visited"
                                (gfi/assess two [] (cm/choicemap :y1 1.0 :y2 2.0 :y3 0.0) (rng/key 33))))))))
  (testing "a body's untraced would hide a var of that name"
    (let [ns (create-ns 'gen.encapsulated-test.shadowed)]
      (intern ns 'untraced (fn [x] x))
      (binding [*ns* ns]
        (refer 'clojure.core)
        (is (re-find #"would hide the var"
                     (try (eval '(gen.dynamic/gen [] (untraced 1))) ""
                          (catch clojure.lang.Compiler$CompilerException e (ex-message (ex-cause e)))))))))
  (testing "a hand-made variadic body fn of the old hidden parameters"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes trace, splice, param and untraced"
                          (gen.dynamic/make-gen-fn (fn [_trace _splice _param & args] (count args)) '([& args]))))
    (testing "and a fixed arity one (review, 2026-10-20)"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes trace, splice, param and untraced"
                            (gen.dynamic/make-gen-fn (fn [_trace _splice _param x y] [x y]) '([x y])))))
    (testing "and the old variadic one for named arguments, which would shift them (review 3)"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes trace, splice, param and untraced"
                            (gen.dynamic/make-gen-fn (fn [_trace _splice _param & args] args) '([x]))))))
  (testing "a variadic body fn of the four hidden parameters fits named arguments (review 3)"
    (let [gf (gen.dynamic/make-gen-fn (fn [_trace _splice _param _untraced & args] (vec args)) '([x y]))]
      (is (= [1 2] (:retval (gfi/simulate gf [1 2] (rng/key 0))))))))

(deftest operations-that-need-exact-densities-are-loud
  ;; each would otherwise fix one draw of ω, silently: a gradient of the
  ;; estimate at that draw, a likelihood slice, a conditional
  (let [model (gen [] (let [theta (trace :theta (d/normal 0 1))
                            z (untraced (d/normal 0 1))]
                        (trace :y (d/normal (+ theta z) 1))))
        tr (:trace (gfi/generate model [] (cm/choicemap :y 1.5) (rng/key 24)))
        pm (:trace (gfi/generate pm-model [] (cm/choicemap [:y :y] 1.5) (rng/key 25)))
        fs (gen [] (let [f (trace :f (d/mvnormal [0.0 0.0] [[1.0 0.0] [0.0 1.0]]))]
                     (trace :y (d/normal (+ (array/sum f) (untraced (d/normal 0 1))) 1))))
        fs-tr (:trace (gfi/generate fs [] (cm/choicemap :y 0.5) (rng/key 26)))
        bern (gen [] (let [b (trace :b (d/bernoulli 0.5))]
                       (trace :y (d/normal (+ (array/where b 1.0 0.0) (untraced (d/normal 0 1))) 1))))]
    (doseq [[label f] [["choice-gradients" #(grad/choice-gradients tr (sel/select :theta))]
                       ["choice-gradients of an estimated density" #(grad/choice-gradients pm (sel/select :theta))]
                       ["hmc" #(mcmc/hmc tr (sel/select :theta) {:key (rng/key 27)})]
                       ["mala" #(mcmc/mala tr (sel/select :theta) 0.1 (rng/key 28))]
                       ["map-optimize" #(mcmc/map-optimize tr (sel/select :theta))]
                       ["elliptical-slice" #(mcmc/elliptical-slice fs-tr :f [0.0 0.0] [[1.0 0.0] [0.0 1.0]] (rng/key 29))]
                       ["the exact proposal" #(gfi/generate (enum/with-exact-proposal bern [:b]) [] (cm/choicemap :y 0.5)
                                                             (rng/key 30))]
                       ["the exact proposal's assess" #(gfi/assess (enum/with-exact-proposal bern [:b]) []
                                                                   (cm/choicemap :b true :y 0.5) (rng/key 35))]

                       ["an internal proposal's density"
                        #(let [gf (prop/with-internal-proposal model (gen [_] (trace :theta (d/normal (untraced (d/normal 0 1)) 1))))]
                           (gfi/project (:trace (gfi/generate gf [] (cm/choicemap :y 1.5) (rng/key 31))) (sel/select :theta)))]]]
      (testing label
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs exact densities" (f)))))))

;; ---------------------------------------------------------------------------
;; Plain calls in a body (genclj-zxis)
;; ---------------------------------------------------------------------------

;; (gf & args) in a body ran gf with a key from the implicit source, so a run
;; with a fixed key did not repeat, silently (found writing the book's
;; chapter 012). The body makes the call now: with the next key from its ω,
;; which it takes only when the callee drew. Planted (2026-10-24): the
;; implicit key back fails "repeats with its key"; taking ω for every call
;; fails "a call that draws nothing"; never taking it fails "an estimate"
;; and "kept by an update".

(def ^:private noise (gen [] (trace :z (d/normal 0 1))))

(def ^:private noisy
  "A plain call of a generative function that draws."
  (gen [] (let [z (noise)] (trace :x (d/normal z 1)) z)))

(def ^:private twice (gen [x] (* 2.0 x)))

(def ^:private quiet
  "A plain call of a generative function that draws nothing."
  (gen [] (trace :x (d/normal (twice 1.5) 1))))

(deftest plain-calls-in-a-body-draw-from-its-omega
  (testing "a run with a plain call repeats with its key, and another key draws another value"
    (let [run #(ts/host (:retval (gfi/simulate noisy [] (rng/key %))))]
      (is (= (run 1) (run 1)))
      (is (not= (run 1) (run 2)))))
  (testing "a call that draws nothing costs nothing: no ω, an exact score, a batched run"
    (let [tr (gfi/simulate quiet [] (rng/key 3))]
      (is (nil? (:gen.trace/omega tr)))
      (is (map? (grad/choice-gradients tr (sel/select :x))) "a gradient, which needs the exact density")
      (is (nil? (batched/fallback-reason (batched/simulate quiet [] 4 (rng/key 4)))))))
  (testing "a call that drew makes the score an estimate: ω kept, gradients loud, a batched run falls back"
    (let [tr (gfi/simulate noisy [] (rng/key 5))]
      (is (some? (:gen.trace/omega tr)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs exact densities, but a plain call that made random choices"
                            (grad/choice-gradients tr (sel/select :x))))
      (is (re-find #"plain call that made random choices in a batched run"
                   (str (:message (batched/fallback-reason (batched/simulate noisy [] 4 (rng/key 6)))))))))
  (testing "an update that changes nothing keeps ω, so the plain call draws the same"
    (let [tr (gfi/simulate noisy [] (rng/key 7))
          t2 (:trace (gfi/update tr [] [] cm/EMPTY (rng/key 8)))]
      (is (= (ts/host (:retval tr)) (ts/host (:retval t2))))))
  (testing "outside a body a plain call still runs, with a key from the implicit source"
    (is (number? (ts/host (noise))))))
