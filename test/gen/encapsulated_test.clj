(ns gen.encapsulated-test
  "Encapsulated randomness (thesis §4.5): untraced draws in gen bodies and
   estimated densities (gen.custom/estimated).
   - The laws that hold for one draw of ω (gen.laws/encapsulated; the
     general set's assess against generate on the same key).
   - ω is kept unless the change hints say something changed, and untraced
     draws never shift a site's key.
   - The statistical laws: Eq. 4.3 (E_q̊ ξ = p) and Eq. 4.4 (E_p̊ 1/ξ =
     1/p, through simulate); pseudo-marginal MH reaches the exact posterior
     (Eqs. 4.15-4.16), with untraced draws and with an importance-sampling
     likelihood; importance sampling with an encapsulated proposal is
     unbiased (§4.5.4).
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

(def untraced-models
  [["mixture" mixture [] []]
   ["simulator" with-simulator [0.4] [[0.7]]]
   ;; growing alternatives too, so that updates make fresh choices (law audit gap 5)
   ["map" (comb/map noisy-kernel) [[0.1 0.2 0.3]] [[[0.1 0.5 0.3]] [[0.1 0.2]] [[0.1 0.2 0.3 0.4]]]]
   ["unfold" (comb/unfold noisy-step) [3 0.0] [[2 0.0] [3 0.5] [5 0.0]]]
   ["spliced" spliced [0.0] [[1.0]]]])

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

;; an estimator has no proposal density, so regenerate cannot resample its
;; choice: the laws without a selection that may hold it
(def estimated-laws
  ;; and its joint sampler draws ω otherwise than its estimate: no same-key score
  (apply dissoc laws/encapsulated [:score-is-assessed-on-the-same-key
                                   :regenerate-keeps-unselected-values :regenerate-weight-is-the-projection-ratio
                                   :regenerate-weight-with-new-arguments-is-the-projection-ratio
                                   :regenerate-weight-is-the-retained-density-ratio :regenerating-everything-has-weight-zero]))

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

(deftest an-estimated-density-falls-back-to-particles
  ;; for the reason it has (review, 2026-10-20: the stated one was a spot check's)
  (is (= "An estimated density in a batched run"
         (:message (batched/fallback-reason (batched/simulate likelihood [0.4] 4 (rng/key 50)))))))

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
  ;; mixture, g is narrower than every component, so g/ξ is bounded (≤ 3.9);
  ;; for the estimator it is not, and its variance is finite through the
  ;; prior samples among the five. Planted bug (2026-10-20): a joint
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
                   n3)))))

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

(defn- posterior-ok?
  "Are the θs N(y/3, √(2/3)) (θ ~ N(0, 1), y ~ N(θ, √2))? Mean within 5
   standard errors, variance within 5 of its standard errors."
  [thetas y]
  (let [n (count thetas)
        [m s] (mean-sd thetas)
        v (/ 2.0 3.0)]
    (and (< (Math/abs (clojure.core/- m (clojure.core// y 3.0))) (clojure.core/* 5 (Math/sqrt (clojure.core// v n))))
         (< (Math/abs (clojure.core/- (clojure.core/* s s) v)) (clojure.core/* 5 v (Math/sqrt (clojure.core// 2.0 (dec n))))))))

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
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot resample a selected choice"
                            (gfi/regenerate tr (sel/select :y) (rng/key 23))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a function" (custom/estimated {:addresses [:y] :estimate 3})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #":addresses names its choices" (custom/estimated {:estimate (:estimate likelihood)})))
      (testing "choices other than all of them (law audit, 2026-10-20: a strict subset gave a trace missing a choice)"
        (let [two (custom/estimated {:addresses [:y1 :y2] :estimate (fn [_ _ _] 0.0)})]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs every choice of the estimator"
                                (gfi/generate two [] (cm/choicemap :y1 1.0) (rng/key 32))))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs every choice of the estimator"
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
                            (gen.dynamic/make-gen-fn (fn [_trace _splice _param x y] [x y]) '([x y])))))))

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
