(ns gen.inference-bench
  "GenMLX's inference benchmarks in gen.clj (bean genclj-ls29): its
   bench/linreg.cljs, hmm.cljs, gmm.cljs and particle_scaling.cljs. Each
   section times its algorithms and measures their accuracy against an
   exact answer:
   - linear regression: the analytic posterior;
   - a hidden Markov model: the forward algorithm's log marginal likelihood;
   - a Gaussian mixture: the exact log marginal likelihood (the means
     integrated out, the assignments enumerated).

   A time is the median of timed runs after a warm-up, forcing the result;
   an accuracy is the mean over three keys. Compared within gen.clj, so run
   it on each backend (spot checks of batched runs are on, as by default;
   -Dgen.bench.spot-checks=0 turns them off):
     clojure -M:bench -m gen.inference-bench
     clojure -J-Dgen.backend=mlx [-J-Dgen.mlx.device=cpu] -M:bench -m gen.inference-bench

   Left out of GenMLX's: NUTS (bean genclj-tvc0) and HMC's step-size
   adaptation. The linear regression's chains:
   - GenMLX's: MH with a step of 0.5 for 200 samples, HMC for 100.
   - The same tuning information for both algorithms, the sd of the
     posterior's narrowest direction: HMC's step is that sd (half the
     leapfrog integrator's stability limit there), the tuned MH step 2.38/√2
     times it (Roberts, Gelman and Gilks' scaling for a Gaussian target).
   - The same time: MH steps cost about a twentieth of HMC's on the JVM
     backend, so the MH chains also run 20 times as many steps (4000
     samples after 2000), about HMC's time there."
  (:require [gen.array :as a]
            [gen.array.protocols :as p]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.kernel :as k]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.particle-filter :as pf]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Timing and reporting
;; ---------------------------------------------------------------------------

(defn- time-ms [f]
  (let [t0 (System/nanoTime)]
    (f)
    (/ (- (System/nanoTime) t0) 1e6)))

(defn- median-ms
  "The median time of (run key): a warm-up, then 3 timed runs, or 7 after 10
   more warm-ups when a run takes under 50 ms (so the JIT has compiled it)."
  [run]
  (let [key (rng/key 0)
        fast? (< (time-ms #(run key)) 50.0)
        reps (if fast? 7 3)]
    (when fast? (dotimes [_ 10] (run key)))
    (nth (sort (repeatedly reps #(time-ms (fn [] (run key))))) (quot reps 2))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

(defn- row
  "Time (run key), then measure its result (a map of numbers) for three
   keys and print the means with describe. Says so when a batched run fell
   back to one particle at a time."
  [label run measure describe]
  (let [ms (median-ms run)
        results (mapv run (rng/split (rng/key 1) 3))
        measures (map measure results)]
    (println (format "  %-34s %9.1f ms  |  %s" label ms
                     (describe (into {} (map (fn [m] [m (mean (map m measures))])) (keys (first measures))))))
    (when-let [reason (some-> (first results) :particles b/fallback-reason)]
      (println "    fell back to one particle at a time:" (:message reason)))
    (flush)))

(defn- logsumexp [xs]
  (let [m (apply max xs)]
    (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))

(defn- normal-lp [x mu] (- (* -0.5 (- x mu) (- x mu)) (* 0.5 (Math/log (* 2 Math/PI)))))

(defn- describe-lml [{:keys [lml-error ess]}]
  (format "log-ML off by %.2f%s" lml-error (if ess (format ", ESS %.1f" ess) "")))

;; ---------------------------------------------------------------------------
;; 1. Linear regression (bench/linreg.cljs): 20 observations
;; ---------------------------------------------------------------------------

(def ^:private linreg-xs (vec (range 20)))

(def ^:private linreg-ys
  "2x + 0.5 + N(0, 1) noise, from a fixed host generator so that every
   backend sees the same data."
  (let [r (java.util.Random. 42)]
    (mapv #(+ (* 2.0 %) 0.5 (.nextGaussian r)) linreg-xs)))

(defn- regression-posterior
  "The posterior of [slope intercept] under N(0, 10²) priors and unit noise:
   {:mean [m0 m1] :cov [[s00 s01] [s01 s11]]}."
  [xs ys]
  (let [n (double (count xs))
        sx (reduce + xs)
        sxx (reduce + (map * xs xs))
        sxy (reduce + (map * xs ys))
        sy (reduce + ys)
        prior 0.01
        p00 (+ sxx prior)
        p11 (+ n prior)
        det (- (* p00 p11) (* sx sx))
        s00 (/ p11 det)
        s01 (/ (- sx) det)
        s11 (/ p00 det)]
    {:mean [(+ (* s00 sxy) (* s01 sy)) (+ (* s01 sxy) (* s11 sy))]
     :cov [[s00 s01] [s01 s11]]}))

(def ^:private posterior (regression-posterior linreg-xs linreg-ys))

(def ^:private slope-mean (get-in posterior [:mean 0]))
(def ^:private slope-sd (Math/sqrt (get-in posterior [:cov 0 0])))

(defn- slope-error
  "How far slope is from the posterior mean, in posterior sds."
  [slope]
  (/ (Math/abs (- slope slope-mean)) slope-sd))

(def ^:private narrowest-sd
  "The sd of the posterior's narrowest direction (its smallest eigenvalue's
   square root): what limits both HMC's and a random walk's step."
  (let [[[s00 s01] [_ s11]] (:cov posterior)
        tr (+ s00 s11)
        det (- (* s00 s11) (* s01 s01))]
    (Math/sqrt (/ (- tr (Math/sqrt (- (* tr tr) (* 4 det)))) 2))))

(def ^:private hmc-eps
  "Half the leapfrog integrator's stability limit (2 sd) in the narrowest
   direction."
  narrowest-sd)

(def ^:private mh-step
  "Roberts, Gelman and Gilks' random-walk scale for a d = 2 Gaussian target,
   2.38/√d, in the narrowest direction."
  (* (/ 2.38 (Math/sqrt 2)) narrowest-sd))

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace [:y j] (d/normal (a/+ (a/* slope x) intercept) 1)))
      slope)))

(def ^:private linreg-obs
  (apply cm/choicemap (mapcat (fn [j y] [[:y j] y]) (range) linreg-ys)))

(def random-walk
  "GenMLX's compiled MH proposal: a Gaussian step on both latents."
  (gen [tr step]
    (trace :slope (d/normal (tr :slope) step))
    (trace :intercept (d/normal (tr :intercept) step))))

(defn- chain
  "A run of kernel from a trace generated with the observations."
  [kernel samples burn key]
  (let [[k-start k-run] (rng/split key)]
    (k/run kernel (:trace (gfi/generate linreg [linreg-xs] linreg-obs k-start))
           {:samples samples :burn burn :key k-run})))

(defn- measure-chain [{:keys [traces acceptance-rate]}]
  {:slope-error (slope-error (mean (map #(a/item (% :slope)) traces)))
   :acceptance (or acceptance-rate Double/NaN)})

(defn- describe-chain [{:keys [slope-error acceptance]}]
  (format "slope off by %.2f sd, acceptance %.2f" slope-error acceptance))

(defn- hmc [opts]
  (k/kernel (fn [tr key] (mcmc/hmc tr (sel/select :slope :intercept) opts key))))

(defn- linreg-section []
  (println (format "\n== Linear regression: 20 observations; posterior slope %.4f ± %.4f; HMC eps %.4f, MH step %.4f =="
                   slope-mean slope-sd hmc-eps mh-step))
  (row "MH, step 0.5 (200, burn 100)" #(chain (k/mh random-walk [0.5]) 200 100 %) measure-chain describe-chain)
  (row "MH, step 0.5 (4000, burn 2000)" #(chain (k/mh random-walk [0.5]) 4000 2000 %) measure-chain describe-chain)
  (row "MH, tuned step (4000, burn 2000)" #(chain (k/mh random-walk [mh-step]) 4000 2000 %)
       measure-chain describe-chain)
  (row "HMC, L = 10 (100, burn 50)" #(chain (hmc {:L 10 :eps hmc-eps}) 100 50 %) measure-chain describe-chain)
  (row "HMC compiled (100, burn 50)" #(chain (hmc {:L 10 :eps hmc-eps :compile? true}) 100 50 %)
       measure-chain describe-chain)
  (doseq [n [1000 10000]]
    (row (str "importance, batched, N = " n)
         #(imp/importance-sampling linreg [linreg-xs] linreg-obs n {:key % :batched? true})
         (fn [{ps :particles lw :log-normalized-weights}]
           {:slope-error (slope-error (reduce + (map #(* (Math/exp %1) %2) lw (a/->clj (b/choice ps :slope)))))
            :ess (w/ess lw)})
         #(format "slope off by %.2f sd, ESS %.1f" (:slope-error %) (:ess %)))))

;; ---------------------------------------------------------------------------
;; 2. Hidden Markov model (bench/hmm.cljs): 2 states, 50 steps
;; ---------------------------------------------------------------------------

(def ^:private hmm-ys
  "GenMLX's data (seed 42, hardcoded there)."
  [3.070 -2.705 -1.695 -3.586 -3.487 -2.084 -1.983 -3.072
   -0.495 -2.301 0.161 -1.769 -1.536 -0.805 0.851 2.223
   1.253 2.740 2.009 -0.089 1.709 2.528 -1.942 -2.348
   -2.661 -2.928 -2.599 -1.194 -3.515 -1.982 -2.571 -2.338
   -2.913 -1.772 0.147 -1.991 -1.350 2.660 1.810 2.121
   2.972 5.104 1.414 3.214 2.140 1.725 2.608 1.461
   2.415 -1.998])

(def ^:private hmm-T (count hmm-ys))
(def ^:private hmm-transition [[0.9 0.1] [0.1 0.9]])
(def ^:private hmm-means [-2.0 2.0])

(def ^:private transition (a/array hmm-transition))
(def ^:private initial (a/array [0.5 0.5]))
(def ^:private means (a/array hmm-means))

(def hmm-step
  "One step: z_t ~ categorical(row z_{t-1} of the transitions), y_t ~ N(mu_z, 1)."
  (gen [t z-prev]
    (let [z (trace :z (d/categorical (if (zero? t) initial (a/take transition z-prev 0))))]
      (trace :y (d/normal (a/take means z) 1))
      z)))

(def hmm (comb/unfold hmm-step))

(defn- hmm-observation [t] (cm/choicemap [t :y] (hmm-ys t)))

(def ^:private hmm-obs (reduce cm/merge (map hmm-observation (range hmm-T))))

(def ^:private hmm-exact-lml
  "The forward algorithm's log p(y_0, ..., y_49)."
  (let [states (range 2)
        emit (fn [t j] (normal-lp (hmm-ys t) (hmm-means j)))
        forward (fn [alpha t]
                  (mapv (fn [j]
                          (+ (logsumexp (map #(+ (alpha %) (Math/log (get-in hmm-transition [% j]))) states))
                             (emit t j)))
                        states))]
    (logsumexp (reduce forward (mapv #(+ (Math/log 0.5) (emit 0 %)) states) (range 1 hmm-T)))))

(defn- run-filter
  "A particle filter over every observation, resampling when the ESS falls
   under n/2: its final state and its log marginal likelihood estimate."
  [n batched? key]
  (let [ks (rng/split key (* 2 hmm-T))
        state (reduce (fn [state t]
                        (-> state
                            (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                            (pf/step [(inc t) 0] [diff/unknown-change diff/no-change] (hmm-observation t)
                                     {:key (ks (* 2 t))})))
                      (pf/initialize hmm [1 0] (hmm-observation 0) n {:key (ks 0) :batched? batched?})
                      (range 1 hmm-T))]
    (assoc state :log-ml-estimate (pf/log-ml-estimate state))))

(defn- lml-measure
  "Measures a result holding :log-ml-estimate (and weights, for an ESS)
   against the exact log marginal likelihood."
  [exact]
  (fn [{lml :log-ml-estimate lw :log-normalized-weights}]
    (cond-> {:lml-error (Math/abs (- lml exact))} lw (assoc :ess (w/ess lw)))))

(defn- hmm-section []
  (println (format "\n== Hidden Markov model: 2 states, %d steps; exact log-ML %.4f ==" hmm-T hmm-exact-lml))
  (doseq [[label n batched?] [["importance, scalar, N = 100" 100 false]
                              ["importance, batched, N = 1000" 1000 true]]]
    (row label #(imp/importance-sampling hmm [hmm-T 0] hmm-obs n {:key % :batched? batched?})
         (lml-measure hmm-exact-lml) describe-lml))
  (doseq [[label n batched?] [["particle filter, scalar, N = 100" 100 false]
                              ["particle filter, batched, N = 100" 100 true]
                              ["particle filter, batched, N = 250" 250 true]]]
    (row label #(run-filter n batched? %) (lml-measure hmm-exact-lml) describe-lml)))

;; ---------------------------------------------------------------------------
;; 3. Gaussian mixture (bench/gmm.cljs): 3 components, 8 observations
;; ---------------------------------------------------------------------------

(def ^:private gmm-ys [-3.5 -4.1 0.3 -0.2 3.8 4.2 0.1 3.6])
(def ^:private gmm-weights [0.3 0.5 0.2])
(def ^:private mixture (a/array gmm-weights))

(def gmm
  (gen [ys]
    (let [mus (a/stack [(trace :mu0 (d/normal 0 10)) (trace :mu1 (d/normal 0 10)) (trace :mu2 (d/normal 0 10))])]
      (doseq [i (range (count ys))]
        (let [z (trace [:z i] (d/categorical mixture))]
          (trace [:y i] (d/normal (a/take mus z) 1))))
      mus)))

(def ^:private gmm-obs
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) gmm-ys)))

(defn- cluster-lml
  "log p(ys) of the observations of one component: mu ~ N(0, 10²) and
   y_i ~ N(mu, 1), so ys ~ N(0, I + 100·11ᵀ)."
  [ys]
  (let [n (count ys)
        s (reduce + ys)
        ss (reduce + (map * ys ys))
        v 100.0]
    (- (* -0.5 n (Math/log (* 2 Math/PI)))
       (* 0.5 (Math/log (+ 1 (* n v))))
       (* 0.5 (- ss (/ (* v s s) (+ 1 (* n v))))))))

(defn- assignment-lml
  "log p(zs, ys) with the means integrated out."
  [zs]
  (+ (reduce + (map #(Math/log (gmm-weights %)) zs))
     (reduce + (for [c (range 3)] (cluster-lml (keep-indexed #(when (= c (zs %1)) %2) gmm-ys))))))

(def ^:private gmm-exact-lml
  "log p(ys): every one of the 3^8 assignments, the means integrated out."
  (let [assignments (reduce (fn [zss _] (for [zs zss z (range 3)] (conj zs z))) [[]] gmm-ys)]
    (logsumexp (map assignment-lml assignments))))

(defn- gmm-section []
  (println (format "\n== Gaussian mixture: 3 components, %d observations; exact log-ML %.4f =="
                   (count gmm-ys) gmm-exact-lml))
  (doseq [[label n batched?] [["importance, scalar, N = 200" 200 false]
                              ["importance, batched, N = 1000" 1000 true]
                              ["importance, batched, N = 10000" 10000 true]]]
    (row label #(imp/importance-sampling gmm [gmm-ys] gmm-obs n {:key % :batched? batched?})
         (lml-measure gmm-exact-lml) describe-lml)))

;; ---------------------------------------------------------------------------
;; 4. Particle scaling (bench/particle_scaling.cljs): batched importance
;; ---------------------------------------------------------------------------

(def ^:private scaling-xs [1.0 2.0 3.0 4.0 5.0])

(def ^:private scaling-obs
  (apply cm/choicemap (mapcat (fn [j y] [[:y j] y]) (range) [2.3 4.7 6.1 8.9 10.2])))

(defn- scaling-section []
  (println "\n== Particle scaling: batched importance sampling, linear regression with 5 observations ==")
  (let [times (into (sorted-map)
                    (for [n [100 250 500 1000 2500 5000 10000]]
                      [n (median-ms #(imp/importance-sampling linreg [scaling-xs] scaling-obs n
                                                              {:key % :batched? true}))]))]
    (doseq [[n ms] times]
      (println (format "  N = %-6d %8.2f ms  |  %.2f µs per particle" n ms (/ (* 1000 ms) n))))
    (println (format "  10,000 against 100 particles: %.1f× the time for 100× the particles"
                     (/ (times 10000) (times 100))))))

;; ---------------------------------------------------------------------------

(def ^:private spot-checks
  (some-> (System/getProperty "gen.bench.spot-checks") Long/parseLong))

(defn -main [& _]
  (binding [b/*spot-checks* (or spot-checks b/*spot-checks*)]
    (let [backend a/*backend*]
      (println "backend:" (p/-name backend)
               (if (= :mlx (p/-name backend))
                 (str ((requiring-resolve 'gen.backend.mlx/device) backend) " " (:float (p/-defaults backend)))
                 (:float (p/-defaults backend)))
               "| spot checks" b/*spot-checks*
               "|" (System/getProperty "java.vm.name") (System/getProperty "java.version")))
    (linreg-section)
    (hmm-section)
    (gmm-section)
    (scaling-section))
  (shutdown-agents))
