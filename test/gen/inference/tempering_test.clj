(ns gen.inference.tempering-test
  "Tempered SMC (gen.inference.tempering) against exact log marginal
   likelihoods: a conjugate normal model whose likelihood is far tighter
   than its prior (where importance sampling from the prior fails, checked
   here), a model whose structure varies (a branch on a draw picks the
   mean), a random walk (Kalman filtering) and a two-state HMM
   (enumeration of its 64 paths). Bands: the mean of five runs' estimates
   (fixed keys) within 5 standard errors of the exact value, σ/√5 for the
   estimator's measured spread σ (spread), and the runs' standard
   deviation below 2.5σ (a precision the estimator must have: a band from
   the runs' own spread, a t statistic on 4 degrees of freedom, grew with a
   bug that scatters them, to 2.2 nats; genclj-7z2s). The posterior
   particles' mean of the conjugate model's latent agrees with the exact
   posterior mean. Planted against the moves (2026-11-01): dropping their
   (1 - β) correction (the moves then target the posterior at every step)
   fails the tests, as does leaving out the random-walk moves (the prior
   regenerations alone scatter the conjugate model's estimates from -0.3
   to 11.6 nats against an exact 14.4). The first again on the measured
   bands (2026-11-06): it fails the conjugate model's (by 3.2 nats), the
   branching one's and the HMM's; the random walk's passes it."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.proposal :as prop]
            [gen.inference.tempering :as tempering]
            [gen.random :as rng]
            [gen.state-space :as ss]
            [gen.test-support :as ts]))

(defn- runs
  "The log-ML estimates of five runs (keys 1 to 5)."
  [model args observations opts]
  (mapv #(:log-ml-estimate (tempering/smc model args observations (assoc opts :key (rng/key %)))) (range 1 6)))

(def ^:private spread
  "Nats: each model's estimator's standard deviation at 200 particles,
   measured over keys 101 to 140 on the JVM (2026-11-06; their means were
   within 2 standard errors of the exact values, as the estimator's low
   bias of about σ²/2 allows)."
  {:conjugate 0.29 :branching 0.14 :walk 0.13 :hmm 0.25})

(defn- within-band?
  "Are the estimates' mean within five standard errors of exact, σ/√n for
   the model's measured spread σ, and their standard deviation below 2.5σ
   (for an estimator of spread σ, n = 5 runs exceed it with probability
   P(χ²₄ > 25) = 5e-5, a normal's past 4 standard errors)?"
  [estimates exact model]
  (let [sigma (double (spread model))
        sd (ts/sd estimates)]
    {:ok (and (ts/within-5-se? estimates exact (* sigma sigma)) (< sd (* 2.5 sigma)))
     :mean (ts/mean estimates) :sd sd :exact exact :band (* 5 (/ sigma (Math/sqrt (count estimates)))) :estimates estimates}))

;; ---------------------------------------------------------------------------
;; Conjugate normal: mu ~ N(0, 100), y_i ~ N(mu, 0.1), twenty of them
;; ---------------------------------------------------------------------------

(def ^:private conjugate
  (gen [n]
    (let [mu (trace :mu (d/normal 0 100))]
      (dotimes [i n] (trace [:y i] (d/normal mu 0.1)))
      mu)))

(def ^:private ys (mapv #(+ 3.0 (* 0.1 (Math/sin (* 1.7 %)))) (range 20)))

(def ^:private ys-observed (reduce (fn [m [i y]] (cm/set-choice m [:y i] y)) cm/EMPTY (map-indexed vector ys)))

(def ^:private conjugate-exact
  "log N(ys; 0, s2 I + t2 11ᵀ), s2 = 0.01 and t2 = 10000, in closed form
   on the host in float64: det = s2^(n-1) (s2 + n t2), and the inverse is
   (I - t2/(s2 + n t2) 11ᵀ)/s2. The covariance's condition number is about
   2e7, so mvnormal's log density in a backend's float32 is off by up to
   0.7 nats (CUDA, 2026-11-04: 15.07 against 14.359); the reference must
   not depend on the backend."
  (let [n (count ys) s2 0.01 t2 10000.0
        sy (reduce + ys)
        syy (reduce + (map #(* % %) ys))
        quad (/ (- syy (/ (* t2 sy sy) (+ s2 (* n t2)))) s2)
        logdet (+ (* (dec n) (Math/log s2)) (Math/log (+ s2 (* n t2))))]
    (* -0.5 (+ (* n (Math/log (* 2 Math/PI))) logdet quad))))

(def ^:private conjugate-posterior-mean
  (let [n (count ys) prec (+ (/ 1 10000.0) (/ n 0.01))]
    (/ (/ (reduce + ys) 0.01) prec)))

(deftest ^:slow a-tight-likelihood
  (testing "importance sampling from the prior falls far short at the same 200 particles"
    ;; the median of 9 keys' shortfalls, so no key's luck decides it: a
    ;; prior draw within a few of the posterior's sd 0.022 is likely enough
    ;; at 2000 particles (17 keys of 40 short by > 3 nats on the JVM), and
    ;; the one-key claim there failed on Metal's float32 draws (2026-11-04);
    ;; at 200, 39 of 40, the median 150 nats
    (let [shortfalls (mapv #(- conjugate-exact (ts/host (:log-ml-estimate (is/importance-sampling conjugate [20] ys-observed 200 {:key (rng/key %)}))))
                           (range 1 10))]
      (is (< 3 (nth (sort shortfalls) 4)) (pr-str shortfalls))))
  (testing "tempered SMC agrees with the exact value"
    (let [r (within-band? (runs conjugate [20] ys-observed {:particles 200}) conjugate-exact :conjugate)]
      (is (:ok r) (pr-str r))))
  (testing "its particles are the posterior's"
    (let [{:keys [traces]} (tempering/smc conjugate [20] ys-observed {:particles 400 :key (rng/key 7)})
          mus (mapv #(ts/host (% :mu)) traces)
          post-var (/ 1 (+ (/ 1 10000.0) (/ 20 0.01)))]
      (is (true? (ts/within-5-se? mus conjugate-posterior-mean post-var))
          (pr-str {:mean (ts/mean mus) :exact conjugate-posterior-mean})))))

;; ---------------------------------------------------------------------------
;; A structure that varies: a branch on a draw picks the mean
;; ---------------------------------------------------------------------------

(def ^:private branching
  (gen []
    (let [x (if (trace :b (d/bernoulli 0.3))
              (trace :x (d/normal 0 1))
              (trace :x (d/normal 3 1)))]
      (trace :y (d/normal x 0.1)))))

(deftest ^:slow a-structure-that-varies
  (let [y 2.2
        sd (Math/sqrt 1.01)
        exact (Math/log (+ (* 0.3 (Math/exp (ts/host (dc/log-prob (d/normal 0 sd) y))))
                           (* 0.7 (Math/exp (ts/host (dc/log-prob (d/normal 3 sd) y))))))
        r (within-band? (runs branching [] (cm/choicemap :y y) {:particles 200}) exact :branching)]
    (is (:ok r) (pr-str r))))

;; ---------------------------------------------------------------------------
;; A random walk (Kalman) and a two-state HMM (enumeration)
;; ---------------------------------------------------------------------------

(def ^:private walk
  (comb/unfold (gen [t z] (let [z' (trace :z (d/normal z 0.5))] (trace :y (d/normal z' 1.0)) z'))))

(def ^:private walk-ys [0.5 1.2 1.8 2.5 1.5 0.9])

(deftest ^:slow a-random-walk
  (let [exact (ts/host (:weight (gfi/assess ss/kalman [6 {:mean 0.0 :cov 0.25}
                                                       {:transition 1.0 :transition-cov 0.25 :emission 1.0 :emission-cov 1.0}]
                                            (apply cm/choicemap (interleave (range) walk-ys)))))
        obs (reduce (fn [m [t y]] (cm/set-choice m [t :y] y)) cm/EMPTY (map-indexed vector walk-ys))
        r (within-band? (runs walk [6 0.0] obs {:particles 200}) exact :walk)]
    (is (:ok r) (pr-str r))))

(def ^:private hmm
  (comb/unfold (gen [t state transition means]
                 (let [s (trace :z (d/categorical (array/take transition state)))]
                   (trace :y (d/normal (array/take means s) 1.0))
                   s))))

(deftest ^:slow an-hmm
  (let [ys [-0.6 -1.4 1.7 2.5 0.4 2.2]
        args [6 0 (array/array [[0.9 0.1] [0.2 0.8]]) (array/array [-1.0 2.0])]
        obs (reduce (fn [m [t y]] (cm/set-choice m [t :y] y)) cm/EMPTY (map-indexed vector ys))
        paths (reduce (fn [ps _] (for [p ps s [0 1]] (conj p s))) [[]] ys)
        joint (fn [path] (reduce (fn [m [t s]] (cm/set-choice m [t :z] s)) obs (map-indexed vector path)))
        exact (ts/host (array/logsumexp (array/array (mapv #(ts/host (:weight (gfi/assess hmm args (joint %)))) paths))))
        r (within-band? (runs hmm args obs {:particles 200}) exact :hmm)]
    (is (:ok r) (pr-str r))))

(deftest a-run-fixed-in-advance
  ;; a run given a previous run's schedule and scales adapts nothing (its
  ;; estimate's exponential is then unbiased); on the same key it repeats
  ;; the previous run exactly
  (let [r (tempering/smc branching [] (cm/choicemap :y 2.2) {:particles 100 :key (rng/key 3)})
        fixed (tempering/smc branching [] (cm/choicemap :y 2.2)
                             {:particles 100 :key (rng/key 3) :betas (:betas r) :scales (:scales r)})]
    (is (= (:log-ml-estimate r) (:log-ml-estimate fixed)))
    (is (= (:betas r) (:betas fixed)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"rise from 0 to 1"
                          (tempering/smc branching [] (cm/choicemap :y 2.2) {:key (rng/key 3) :betas [0.0 0.7 0.5 1.0] :scales [{} {} {}]})))))

;; ---------------------------------------------------------------------------
;; Impossible starts: a likelihood whose support depends on a latent
;; (genclj-dm71)
;; ---------------------------------------------------------------------------

(def ^:private truncating
  "z ~ U(0, 2), y ~ U(0, z): y = 1.5 is possible only for z > 1.5, which
   three starts in four miss."
  (gen [] (let [z (trace :z (d/uniform 0 2))] (trace :y (d/uniform 0 z)))))

(def ^:private truncating-exact
  "log p(y = 1.5) = log ∫ ½ · 1/z dz over (1.5, 2) = log(½ log(4/3))."
  (Math/log (* 0.5 (Math/log (/ 4.0 3.0)))))

(deftest impossible-starts
  ;; Before 1.2.1 a start that gives the observations density 0 had the log
  ;; weight -Inf - -Inf = NaN, and smc threw "A log weight is NaN" (the
  ;; health check's probe: 4 of 6 starts impossible). Every step's β is
  ;; positive, so such a start's incremental weight is -Inf whatever its
  ;; weight at β = 0, and the estimate does not read it. The second case
  ;; starts from another internal proposal (half its starts impossible),
  ;; where the path runs from it to the posterior: generate's weight less
  ;; the projection is 0 there too, by the GFI's laws
  (testing "latents from their prior"
    (let [r (within-band? (runs truncating [] (cm/choicemap :y 1.5) {:particles 200}) truncating-exact)]
      (is (:ok r) (pr-str r))))
  (testing "latents from another internal proposal"
    (let [q (gen [_] (trace :z (d/uniform 1 2)))
          r (within-band? (runs (prop/with-internal-proposal truncating q) [] (cm/choicemap :y 1.5) {:particles 200})
                          truncating-exact)]
      (is (:ok r) (pr-str r))))
  (testing "the posterior particles are possible ones"
    (let [{:keys [traces]} (tempering/smc truncating [] (cm/choicemap :y 1.5) {:particles 50 :key (rng/key 9)})]
      (is (every? #(< 1.5 (ts/host (% :z)) 2.0) traces)))))

(deftest a-move-s-acceptance
  ;; accept? (private: the moves' rule) rejects a move to a trace where the
  ;; observations have density 0, whose ratio w - (1 - β)(l' - l) would be
  ;; -Inf + Inf, and is loud for any other NaN, as MH's (gen.inference.mh):
  ;; before 1.2.1 it rejected every NaN silently
  (let [accept? #'tempering/accept?
        key (rng/key 4)]
    (is (false? (accept? ##-Inf -1.2 ##-Inf 0.5 key)) "an impossible trace")
    (is (false? (accept? -3.0 -1.2 ##-Inf 0.5 key)) "an impossible trace, whatever w")
    (is (true? (accept? 0.0 -1.2 -1.2 0.5 key)) "a ratio of 1")
    (doseq [[w l l'] [[##NaN -1.2 -1.0] [0.1 -1.2 ##NaN] [##Inf -1.2 ##Inf]]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"NaN" (accept? w l l' 0.5 key)) (pr-str [w l l'])))))

(deftest the-result-keys
  ;; DESIGN §9's result keys: :ess is one host number where it appears, so
  ;; the steps' values are :ess-by-step; :acceptance-rates holds each
  ;; step's, as device's runs hold each chain's (genclj-vjbo: they were
  ;; :ess and :acceptance before 1.2.1)
  (let [r (tempering/smc branching [] (cm/choicemap :y 2.2) {:particles 50 :key (rng/key 5)})
        steps (:steps r)]
    (is (= #{:log-ml-estimate :traces :betas :steps :ess-by-step :acceptance-rates :scales} (set (keys r))))
    (is (= (inc steps) (count (:betas r))))
    (doseq [k [:ess-by-step :acceptance-rates :scales]]
      (is (= steps (count (k r))) (str k)))
    (is (every? #(and (instance? Double %) (<= 1.0 % 50.0)) (:ess-by-step r)) (pr-str (:ess-by-step r)))
    (is (every? #(<= 0.0 % 1.0) (:acceptance-rates r)) (pr-str (:acceptance-rates r)))))

(deftest loud-errors
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no observations"
                        (tempering/smc conjugate [3] cm/EMPTY {:key (rng/key 1)}))))
