(ns gen.inference.tempering-test
  "Tempered SMC (gen.inference.tempering) against exact log marginal
   likelihoods: a conjugate normal model whose likelihood is far tighter
   than its prior (where importance sampling from the prior fails, checked
   here), a model whose structure varies (a branch on a draw picks the
   mean), a random walk (Kalman filtering) and a two-state HMM
   (enumeration of its 64 paths). Bands: 5 standard errors of the mean of
   five runs' estimates (fixed keys), floored at 0.05 nats, and the runs'
   standard deviation below max-sd (a precision the estimator must have:
   a band from its own spread grows with a bug that scatters it). The posterior
   particles' mean of the conjugate model's latent agrees with the exact
   posterior mean. Planted against the moves (2026-11-01): dropping their
   (1 - β) correction (the moves then target the posterior at every step)
   fails the tests, as does leaving out the random-walk moves (the prior
   regenerations alone scatter the conjugate model's estimates from -0.3
   to 11.6 nats against an exact 14.4)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.tempering :as tempering]
            [gen.random :as rng]
            [gen.state-space :as ss]
            [gen.test-support :as ts]))

(defn- runs
  "The log-ML estimates of five runs (keys 1 to 5)."
  [model args observations opts]
  (mapv #(:log-ml-estimate (tempering/smc model args observations (assoc opts :key (rng/key %)))) (range 1 6)))

(def ^:private max-sd
  "Nats: the five runs' standard deviation must stay below this, five times
   the conjugate model's measured 0.2, so that a high-variance estimator
   cannot pass on the width of its own band."
  1.0)

(defn- within-band? [estimates exact]
  (let [n (count estimates)
        m (/ (reduce + estimates) n)
        sd (Math/sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) estimates)) (dec n)))
        band (max 0.05 (* 5 (/ sd (Math/sqrt n))))]
    {:ok (and (< (abs (- m exact)) band) (< sd max-sd)) :mean m :sd sd :exact exact :band band :estimates estimates}))

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
  "log N(ys; 0, 0.01 I + 10000 11ᵀ)."
  (let [n (count ys)
        cov (array/+ (array/* 0.01 (array/eye n)) (array/full [n n] 10000.0))]
    (ts/host (dc/log-prob (d/mvnormal (array/zeros [n]) cov) (array/array ys)))))

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
    (let [r (within-band? (runs conjugate [20] ys-observed {:particles 200}) conjugate-exact)]
      (is (:ok r) (pr-str r))))
  (testing "its particles are the posterior's"
    (let [{:keys [traces]} (tempering/smc conjugate [20] ys-observed {:particles 400 :key (rng/key 7)})
          mus (mapv #(ts/host (% :mu)) traces)
          m (/ (reduce + mus) (count mus))
          post-sd (Math/sqrt (/ 1 (+ (/ 1 10000.0) (/ 20 0.01))))]
      (is (< (abs (- m conjugate-posterior-mean)) (* 5 (/ post-sd (Math/sqrt 400))))
          (pr-str {:mean m :exact conjugate-posterior-mean})))))

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
        r (within-band? (runs branching [] (cm/choicemap :y y) {:particles 200}) exact)]
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
        r (within-band? (runs walk [6 0.0] obs {:particles 200}) exact)]
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
        r (within-band? (runs hmm args obs {:particles 200}) exact)]
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

(deftest loud-errors
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no observations"
                        (tempering/smc conjugate [3] cm/EMPTY {:key (rng/key 1)}))))
