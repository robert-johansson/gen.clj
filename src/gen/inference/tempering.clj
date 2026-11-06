(ns gen.inference.tempering
  "Tempered sequential Monte Carlo, adaptive annealing from the prior to the
   posterior: the log marginal likelihood of observations under a
   generative function, and particles of its posterior. It works where
   importance sampling from the prior fails (a posterior far from the
   prior) and for models whose structure varies (regenerate handles fresh
   and discarded choices).

     (smc model args observations)          → {:log-ml-estimate :traces :betas …}
     (smc model args observations opts)

   The targets are π_β(x) ∝ p(x) L(x)^β for β from 0 to 1, L the
   observations' density (gfi/project with their selection): π_0 is the
   latent choices' prior, π_1 their posterior. (With another internal
   proposal q, π_β(x) ∝ q(x) L(x)^β, L then generate's weight, which
   gfi/project gives by the GFI's laws: from q to the same posterior.)
   The particles start as generate's traces with the observations; a
   start that gives them density 0 (a likelihood whose support depends on
   a latent) weighs 0 from the first step on, and a run where every start
   does is a loud error. Each step
   - picks the next β by bisection, so that the incremental weights
     (β' - β) log L keep :target-ess of the particles' effective sample
     size (all the way to 1 when they can);
   - adds their log mean to the estimate, and resamples (systematic);
   - moves every particle by :sweeps sweeps of single-site
     Metropolis-Hastings on π_β', each move's weight less (1 - β') Δ log L,
     which leaves π_β' invariant: each latent address of the particle's
     trace regenerated from its prior (global moves, discrete choices),
     and a continuous one also moved by a normal random walk whose scale
     is that address's spread across the particles (local moves, which a
     posterior far narrower than its prior needs); a walk that would
     change the trace's structure is rejected.
   The estimate is consistent: it converges to the log marginal likelihood
   as :particles grows. Its exponential is an unbiased estimate of the
   marginal likelihood only when the schedule of βs and the moves' scales
   are fixed in advance, and here both adapt to the particles (a small
   bias in practice). For an unbiased run, give a previous run's :betas
   and :scales: the run then adapts nothing.

   opts: :particles (200), :sweeps (2), :target-ess (0.5, a fraction of the
   particles), :max-steps (1000; a loud error past it), :key; :betas (the
   schedule, from 0 to 1) and :scales (each step's random-walk scales, a
   vector of maps from address to scale), a previous run's, to fix them.
   The result: :log-ml-estimate (host), :traces (the posterior particles),
   :betas (the schedule taken), :steps (their number), and one entry a
   step in each of :ess-by-step (the incremental weights' effective
   sample size before the step resamples, host), :acceptance-rates (the
   fraction of the step's moves accepted, host) and :scales (the
   random-walk scales the step used, which :scales takes back).

   A particle runs one at a time (the moves are regenerates of traces);
   the weights are read on the host at each step, the inference loop's
   boundary."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- host ^double [x] (double (array/item x)))

(defn- uniform ^double [key] (host (rng/uniform key)))

(defn- step-weights
  "Each particle's log weight lw plus (b - beta) log L."
  [lw ls ^double beta ^double b]
  (let [d (- b beta)]
    (mapv (fn [^double lwi ^double l] (if (zero? d) lwi (+ lwi (* d l)))) lw ls)))

(defn- next-beta
  "The largest b in (beta, 1] whose incremental weights keep target of the
   n particles' effective sample size; 1 when it does."
  ^double [lw ls ^double beta ^double target]
  (let [ok? (fn [^double b] (>= (w/ess (step-weights lw ls beta b)) target))]
    (if (ok? 1.0)
      1.0
      (loop [lo beta hi 1.0 i 0]
        (if (or (= i 60) (< (- hi lo) 1e-12))
          (max lo (+ beta 1e-12))
          (let [mid (* 0.5 (+ lo hi))]
            (if (ok? mid) (recur mid hi (inc i)) (recur lo mid (inc i)))))))))

(defn- latents
  "The paths of a trace's choices that are not observed."
  [trace observed]
  (into [] (comp (map first) (remove observed)) (cm/leaves (:choices trace))))

(defn- addresses [trace] (into #{} (map first) (cm/leaves (:choices trace))))

(defn- continuous? [v]
  (or (float? v) (and (array/backend-array? v) (= :float (array/dtype-kind (array/dtype v))))))

(defn- scales-of
  "Each continuous latent address's random-walk scale: the spread of its
   values across the particles that have it (elementwise for an array),
   floored at 1e-6."
  [traces observed]
  (let [values (reduce (fn [m t]
                         (reduce (fn [m [p v]] (if (and (not (observed p)) (continuous? v)) (update m p (fnil conj []) v) m))
                                 m (cm/leaves (:choices t))))
                       {} traces)]
    (into {} (keep (fn [[p vs]]
                     (when (and (>= (count vs) 2) (apply = (map array/shape vs)))
                       [p (array/maximum (array/std (array/stack vs) {:axis 0}) 1e-6)])))
          values)))

(defn- accept?
  "Metropolis-Hastings on π_beta (beta > 0): the move's log weight w (log p
   ratio less the proposal's), less (1 - beta) times its change in log L,
   from l to l'. A move to a trace where the observations have density 0
   (l' -Inf, and w with it) is rejected, since π_beta is 0 there: its ratio
   would be -Inf + Inf. Otherwise MH's rule (mh/accept?), loud for a NaN
   ratio."
  [w l l' beta key]
  (let [l' (double l')]
    (and (not (== ##-Inf l'))
         (mh/accept? (- (host w) (* (- 1.0 (double beta)) (- l' (double l)))) (uniform key)
                     {:kernel :tempering :weight (host w) :log-likelihood l :proposed-log-likelihood l'}))))

(defn- regenerate-move
  "The latent at a resampled from the model's internal proposal (its
   prior): [trace l accepted?]."
  [t l beta obs-sel a key]
  (let [[k1 k2] (rng/split key)
        {t' :trace wt :weight} (gfi/regenerate t (sel/from-paths [a]) k1)
        l' (host (gfi/project t' obs-sel))]
    (if (accept? wt l l' beta k2) [t' l' true] [t l false])))

(defn- walk-move
  "The continuous latent at a moved by a normal random walk of scale eps (a
   symmetric proposal, so the update's weight is the log p ratio); a move
   that changes the trace's structure is rejected, which keeps the kernel
   reversible: [trace l accepted?]."
  [t l beta obs-sel a eps key]
  (let [[k1 k2 k3] (rng/split key 3)
        x (cm/get-choice (:choices t) a)
        x' (array/+ x (array/* eps (rng/normal k1 (array/shape x))))
        {t' :trace wt :weight} (gfi/update t (cm/set-choice cm/EMPTY a x') k2)]
    (if (not= (addresses t) (addresses t'))
      [t l false]
      (let [l' (host (gfi/project t' obs-sel))]
        (if (accept? wt l l' beta k3) [t' l' true] [t l false])))))

(defn- move
  "One particle's sweeps on π_beta, from trace with observations' log
   density l: each latent address regenerated from its prior, and a
   continuous one also moved by a random walk of its scale in eps (when it
   has one): [trace l accepted proposed]."
  [trace l beta observed obs-sel eps sweeps key]
  (reduce
   (fn [[t l acc tried] [s j a]]
     (if-not (contains? (addresses t) a)
       [t l acc tried]
       (let [k (rng/fold-in (rng/fold-in key s) j)
             [k1 k2] (rng/split k)
             [t l ok1] (regenerate-move t l beta obs-sel a k1)
             e (when (continuous? (cm/get-choice (:choices t) a)) (eps a))
             [t l ok2] (if e (walk-move t l beta obs-sel a e k2) [t l nil])]
         [t l (+ acc (if ok1 1 0) (if ok2 1 0)) (+ tried 1 (if e 1 0))])))
   [trace (double l) 0 0]
   (for [s (range sweeps) [j a] (map-indexed vector (latents trace observed))] [s j a])))

(defn smc
  "Tempered SMC of model on args, conditioned on observations (a choice
   map): {:log-ml-estimate :traces :betas :steps :ess-by-step
   :acceptance-rates :scales}. See the namespace's docstring."
  ([model args observations] (smc model args observations nil))
  ([model args observations {:keys [particles sweeps target-ess max-steps key betas scales]
                             :or {particles 200 sweeps 2 target-ess 0.5 max-steps 1000}}]
   (let [n (long particles)
         key (or key (gfi/next-key))
         observed (set (map first (cm/leaves observations)))
         obs-sel (sel/from-paths observed)
         _ (when (empty? observed)
             (throw (ex-info "tempering/smc: no observations; the marginal likelihood is 1" {})))
         starts (mapv #(gfi/generate model args observations %) (rng/split (rng/fold-in key 0) n))
         traces (mapv :trace starts)
         ls (mapv #(host (gfi/project % obs-sel)) traces)
         ;; generate's weight less log L: 0 up to rounding, whatever the
         ;; internal proposal, by the GFI's law that generate's weight is
         ;; the projection on the constraints (gen.laws'
         ;; generate-weight-is-the-projection). A start that gives the
         ;; observations density 0 has both -Inf, so 0, the law's value; its
         ;; lw is never read anyway: every step's β is above 0, so the first
         ;; step weighs it lw + β·(-Inf) = -Inf, and the estimate after it,
         ;; log-sum-exp(lw + β·l) - log n, does not read it
         lw (mapv (fn [{:keys [weight]} ^double l] (if (== ##-Inf l) 0.0 (- (host weight) l))) starts ls)
         _ (when (every? #(== ##-Inf %) ls)
             (throw (ex-info "tempering/smc: every particle gives the observations probability 0" {:particles n})))
         target (* (double target-ess) n)
         _ (when (and betas (not (and (= 0.0 (double (first betas))) (= 1.0 (double (peek (vec betas))))
                                      (apply < (map double betas)))))
             (throw (ex-info "tempering/smc: :betas rise from 0 to 1" {:betas betas})))
         _ (when (and scales (not= (count scales) (dec (count betas))))
             (throw (ex-info "tempering/smc: :scales holds one map a step of :betas" {:steps (dec (count betas)) :scales (count scales)})))]
     (loop [step 1 beta 0.0 traces traces ls ls lw lw
            log-z (- (w/log-sum-exp lw) (Math/log n)) taken [0.0] ess [] rates [] used []]
       (when (> step max-steps)
         (throw (ex-info (str "tempering/smc: β reached only " beta " after " max-steps " steps") {:beta beta :steps max-steps})))
       (let [b (if betas (double (nth betas step)) (next-beta lw ls beta target))
             inc-w (step-weights lw ls beta b)
             log-z (+ log-z (- (w/log-sum-exp inc-w) (w/log-sum-exp lw)))
             step-key (rng/fold-in key step)
             [k-resample k-move] (rng/split step-key)
             idx (w/resample :systematic inc-w n k-resample)
             traces (mapv traces idx)
             ls (mapv ls idx)
             eps (if scales (nth scales (dec step)) (scales-of traces observed))
             moved (mapv (fn [i t l] (move t l b observed obs-sel eps sweeps (rng/fold-in k-move i))) (range n) traces ls)
             tried (reduce + (map #(nth % 3) moved))
             accepted (reduce + (map #(nth % 2) moved))
             traces (mapv first moved)
             ls (mapv second moved)
             taken (conj taken b)
             ess (conj ess (w/ess inc-w))
             rates (conj rates (if (pos? tried) (/ (double accepted) tried) 0.0))
             used (conj used eps)]
         (if (>= b 1.0)
           {:log-ml-estimate log-z :traces traces :betas taken :steps step :ess-by-step ess :acceptance-rates rates
            :scales used}
           (recur (inc step) b traces ls (vec (repeat n 0.0)) log-z taken ess rates used)))))))
