(ns gen.inference.nuts
  "Stan's NUTS sampler: a warmup that adapts the step size and a diagonal
   metric to the target, then draws, by gen.inference.mcmc's nuts, or by its
   hmc as adaptive HMC (L fixed, where Stan's static HMC fixes L·eps).

     (adapt trace selection opts key)   → {:trace :eps :inv-metric :stats}
     (sample trace selection opts key)  → {:traces :stats :eps :inv-metric :divergences
                                           :warmup {:stats :divergences}}
                                           (:draws for :traces with :keep)

   The warmup is Stan's (hmc_nuts_diag_e_adapt):
   - The step size starts from Stan's heuristic: double or halve it until
     one leapfrog step's acceptance crosses 0.8. Dual averaging (Hoffman and
     Gelman 2014) then moves the acceptance statistic toward :target-accept,
     with Stan's γ, t0 and κ and μ = log(10·eps), eps the initial step size.
   - The metric adapts in windows: after an initial buffer of iterations
     that adapt the step size alone, the variances of each window's draws,
     regularized as Stan's ((n/(n+5))·var + 1e-3·(5/(n+5))), become the
     inverse metric at the window's end, where the step size is initialized
     again and dual averaging restarts, μ = log(10·eps) at the new step
     size. The windows double, the last stretched to the terminal buffer,
     which adapts the step size alone.
   - The step size drawn with is the dual average's, exp(x̄).

   Options (Stan's defaults):
     :samples        draws after the warmup (sample's, required)
     :warmup         1000 iterations
     :kernel         :nuts, or :hmc (with :L, 10)
     :max-depth      10, NUTS's
     :eps            1.0, the initial step size
     :jitter         0: each transition steps by eps·(1 + jitter·(2u - 1)),
                     u uniform (Stan's stepsize_jitter, in the warmup too),
                     eps the nominal step size dual averaging adapts
     :inv-metric     the initial inverse metric, in any form mcmc's
                     kernels take (the identity)
     :adapt-metric?  true; false adapts the step size alone
     :target-accept  0.8 (Stan's delta); :gamma 0.05, :t0 10, :kappa 0.75
     :init-buffer 75, :term-buffer 50, :window 25: when they do not fit in
                     the warmup, 15%, 10% and the rest, as Stan's; a warmup
                     under 20 iterations adapts the step size alone
     :compile?       mcmc's (each leapfrog step compiled), :key
     :keep           sample's: a function of a trace, or a map of them,
                     kept for each draw in place of its trace, as
                     gen.inference.kernel/run's :keep

   Each draw's statistics are its kernel's: nuts's {:accepted? :accept-stat
   :tree-depth :n-leapfrog :diverged? :energy :eps}, and for :hmc
   {:accept-stat :n-leapfrog :accepted? :diverged? :eps}, the acceptance
   statistic min(1, α), :eps the step taken (jittered). The result counts
   the divergences after the warmup and in it. With :warmup 0, sample
   draws at :eps and :inv-metric as given, such as an adapt's. The
   inverse metric returned, adapted or given, is labeled: a choice map of
   the selected choices' variances by address (mcmc/inv-metric), which
   the kernels take as :inv-metric. The state is
   threaded through each iteration and every iteration takes its own key,
   so the same key gives the same result; omitting one draws from the
   implicit key source.

   Divergences are one warning among several. A chain can miss a region it
   cannot enter without diverging: on Neal's funnel, after the warmup, most
   chains never reach the neck and report none. So run several chains and
   read R-hat and the ESS too (gen.inference.diagnostics: R-hat under 1.01
   and a bulk ESS of 100 a chain, as Vehtari et al. 2021 advise), and make
   a hierarchical scale non-centered, x = σ·z with z ~ N(0, 1) traced, as
   Stan's guide does."
  {:api :stable}
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.kernel :as kernel]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def ^:private defaults
  {:warmup 1000 :kernel :nuts :L 10 :max-depth 10 :eps 1.0 :jitter 0.0 :adapt-metric? true
   :target-accept 0.8 :gamma 0.05 :t0 10.0 :kappa 0.75 :init-buffer 75 :term-buffer 50 :window 25})

(defn- settings
  "opts over the defaults, checked: loud for a value outside its range."
  [opts]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "The sampler takes an options map, not " (pr-str opts)) {:value opts})))
  (let [s (merge defaults opts)
        positive? #(and (number? %) (Double/isFinite (double %)) (pos? (double %)))
        checks {:warmup nat-int? :L pos-int? :max-depth pos-int? :eps positive? :adapt-metric? boolean?
                :kernel #{:nuts :hmc} :target-accept #(and (positive? %) (< (double %) 1.0)) :gamma positive?
                :t0 #(and (number? %) (>= (double %) 0.0)) :kappa #(and (positive? %) (<= (double %) 1.0))
                :jitter #(and (number? %) (<= 0.0 (double %)) (< (double %) 1.0))
                :init-buffer nat-int? :term-buffer nat-int? :window pos-int?}]
    (doseq [[k ok?] checks]
      (when-not (ok? (get s k))
        (throw (ex-info (str "The sampler's " k " cannot be " (pr-str (get s k))) {:reason :invalid-option k (get s k)}))))
    s))

;; ---------------------------------------------------------------------------
;; The step size
;; ---------------------------------------------------------------------------

(def ^:private log-four-fifths (Math/log 0.8))

(defn- one-step
  "The log acceptance of one leapfrog step of size eps by the kernel from
   trace, its momenta drawn on key: min(0, H0 - H), -Inf when the step
   diverged (NUTS's at :max-depth 1, which steps forwards or backwards
   at random, as good as forwards for momenta drawn symmetrically)."
  [trace selection {:keys [kernel]} eps inv key]
  (case kernel
    :nuts (Math/log (:accept-stat (mcmc/nuts trace selection {:eps eps :inv-metric inv :max-depth 1} key)))
    :hmc (Math/min 0.0 (double (array/item (:log-alpha (mcmc/hmc trace selection {:L 1 :eps eps :inv-metric inv} key)))))))

(defn- initial-eps
  "Stan's step-size heuristic (base_hmc's init_stepsize) from eps: with
   fresh momenta each time, double it while one step's acceptance stays
   above 0.8, or halve it while it stays below, and return the first step
   size where it crosses. A step size past 1e7 or reaching 0 is a loud
   error, as Stan's."
  [trace selection opts eps inv key]
  (let [eps (double eps)]
    (if (or (zero? eps) (> eps 1e7) (Double/isNaN eps))
      eps
      (let [[k-direction k-steps] (rng/split key)
            up? (> (double (one-step trace selection opts eps inv k-direction)) log-four-fifths)]
        (loop [i 0 eps eps]
          (let [change (double (one-step trace selection opts eps inv (rng/fold-in k-steps i)))]
            (if (if up? (not (> change log-four-fifths)) (not (< change log-four-fifths)))
              eps
              (let [eps (if up? (* 2.0 eps) (* 0.5 eps))]
                (cond (> eps 1e7)
                      (throw (ex-info "The step size grew past 1e7 with every step accepted: the posterior is improper. Check the model."
                                      {:reason :improper-posterior :eps eps}))
                      (zero? eps)
                      (throw (ex-info "No step size is small enough to accept one leapfrog step: the posterior may not be continuous."
                                      {:reason :no-step-size}))
                      :else (recur (inc i) eps))))))))))

(defn- dual-averaging
  "Stan's stepsize_adaptation, restarted toward mu."
  [mu]
  {:mu mu :counter 0 :s-bar 0.0 :x-bar 0.0})

(defn- learn
  "Stan's learn_stepsize: the dual average after one acceptance statistic
   (capped at 1), with :eps its next step size, exp(x)."
  [{:keys [mu counter s-bar x-bar]} stat {:keys [target-accept gamma t0 kappa]}]
  (let [counter (inc (long counter))
        eta (/ 1.0 (+ counter (double t0)))
        s-bar (+ (* (- 1.0 eta) (double s-bar)) (* eta (- (double target-accept) (Math/min 1.0 (double stat)))))
        x (- (double mu) (/ (* s-bar (Math/sqrt counter)) (double gamma)))
        x-eta (Math/pow counter (- (double kappa)))]
    {:mu mu :counter counter :s-bar s-bar :x-bar (+ (* (- 1.0 x-eta) (double x-bar)) (* x-eta x)) :eps (Math/exp x)}))

(defn- final-eps
  "The step size after the warmup, exp(x̄); eps, the one just initialized,
   when no iteration followed the last restart (where Stan's would be
   exp(0), 1)."
  [da eps]
  (if (pos? (long (:counter da))) (Math/exp (:x-bar da)) eps))

;; ---------------------------------------------------------------------------
;; The metric
;; ---------------------------------------------------------------------------

(defn- windows
  "Stan's windowed adaptation (windowed_adaptation.hpp) over n warmup
   iterations: the [first last] iterations of each window of the metric's
   estimation, or nil under 20 iterations. Buffers that do not fit become
   15% and 10% of n, the window the rest. Each next window doubles, and
   ends at the terminal buffer when the one after it would pass it."
  [n {:keys [init-buffer term-buffer window]}]
  (let [n (long n)]
    (when (>= n 20)
      (let [[init term base] (if (> (+ (long init-buffer) (long window) (long term-buffer)) n)
                               (let [init (long (* 0.15 n)) term (long (* 0.1 n))]
                                 [init term (- n init term)])
                               [init-buffer term-buffer window])
            last-end (- n (long term) 1)]
        (loop [start (long init) size (long base) end (+ (long init) (long base) -1) acc []]
          (let [acc (conj acc [start end])]
            (if (>= end last-end)
              acc
              (let [size (* 2 size)
                    next-end (+ end size)]
                (recur (inc end) size (if (>= (+ next-end (* 2 size)) (inc last-end)) last-end next-end) acc)))))))))

(defn- flat
  "The trace's selected values as one flat array, in the order of the
   labeled inverse metric's cm/to-array (canonical address order)."
  [trace selection]
  (cm/to-array (grad/continuous-selection trace selection)))

(defn- window-variance
  "Stan's var_adaptation at a window's end: the draws' sample variances
   (flat, as the draws and inv, the inverse metric so far, are),
   regularized toward 1e-3 as Stan's, (n/(n+5))·var + 1e-3·(5/(n+5)). With
   fewer than two draws Stan's estimator leaves the variances as they were,
   so inv is regularized. Loud when one is not finite, as Stan's."
  [draws inv]
  (let [n (double (count draws))
        var (if (< n 2.0) inv (array/variance (array/stack draws) {:axis 0 :ddof 1}))
        regularized (array/+ (array/* (/ n (+ n 5.0)) var) (* 1e-3 (/ 5.0 (+ n 5.0))))]
    (when-not (array/truthy? (array/all (array/isfinite regularized)))
      (throw (ex-info "The metric's adaptation overflowed: a variance of the window's draws is not finite (the posterior may be too wide, or improper)."
                      {:reason :metric-overflow :draws (count draws)})))
    regularized))

;; ---------------------------------------------------------------------------
;; Warmup and draws
;; ---------------------------------------------------------------------------

(defn- transition
  "One draw by the kernel at step size eps and inverse metric inv: [trace
   stats]."
  [trace selection {:keys [kernel L max-depth jitter compile?]} eps inv key]
  (case kernel
    :nuts (let [r (mcmc/nuts trace selection {:eps eps :jitter jitter :inv-metric inv :max-depth max-depth
                                              :compile? compile?} key)]
            [(:trace r) (dissoc r :trace)])
    :hmc (let [r (mcmc/hmc trace selection {:L L :eps eps :jitter jitter :inv-metric inv :compile? compile?} key)]
           [(:trace r) {:accept-stat (Math/min 1.0 (Math/exp (double (array/item (:log-alpha r)))))
                        :n-leapfrog L :accepted? (:accepted? r) :diverged? (:diverged? r) :eps (get r :eps eps)}])))

(defn- divergences [stats] (count (filter :diverged? stats)))

(defn adapt
  "Stan's warmup from trace: :warmup iterations of the kernel adapting its
   step size and (unless :adapt-metric? is false) a diagonal inverse
   metric. Returns the adapted state, {:trace :eps :inv-metric :stats}: the
   last trace, the step size and inverse metric to draw with (a choice map
   of variances by address, mcmc/inv-metric's), and each iteration's
   statistics. See the namespace for the options."
  ([trace selection opts] (adapt trace selection opts nil))
  ([trace selection opts key]
   (let [{:keys [warmup eps inv-metric adapt-metric?] :as opts} (settings opts)
         [k-init k-steps] (rng/split (or key (:key opts) (gfi/next-key)))
         ws (when adapt-metric? (windows warmup opts))
         window-end? (set (map second ws))
         in-window? (fn [i] (and (seq ws) (<= (long (ffirst ws)) (long i) (long (second (peek ws))))))
         step-keys (if (pos? (long warmup)) (rng/split k-steps warmup) [])
         ;; the metric labeled by address, as the kernels take it and adapt returns it
         inv (mcmc/inv-metric trace selection inv-metric)]
     (loop [i 0
            trace trace
            eps (initial-eps trace selection opts eps inv k-init)
            inv inv
            da (dual-averaging (Math/log (* 10.0 (double (:eps opts)))))
            draws []
            stats (transient [])]
       (if (= i warmup)
         {:trace trace :eps (final-eps da eps) :inv-metric inv :stats (persistent! stats)}
         (let [[k-step k-init] (rng/split (step-keys i))
               [t s] (transition trace selection opts eps inv k-step)
               ;; one iteration's graph at a time on a lazy backend
               t (gfi/eval-trace! t)
               da (learn da (:accept-stat s) opts)
               draws (if (in-window? i) (conj draws (flat t selection)) draws)
               stats (conj! stats s)]
           (if (window-end? i)
             (let [inv (cm/from-array (grad/continuous-selection t selection) (window-variance draws (cm/to-array inv)))
                   eps (initial-eps t selection opts (:eps da) inv k-init)]
               (recur (inc i) t eps inv (dual-averaging (Math/log (* 10.0 eps))) [] stats))
             (recur (inc i) t (:eps da) inv da draws stats))))))))

(defn sample
  "Stan's sampler from trace: the warmup (adapt), then :samples draws at the
   adapted step size and inverse metric. Returns {:traces :stats :eps
   :inv-metric :divergences :warmup}: the draws' traces and statistics, the
   step size and inverse metric they used, how many diverged, and the
   warmup's statistics and divergences. With :keep, :draws in place of
   :traces: (keep trace) for each draw, its arrays read on the host, so
   that the sampler holds its current trace alone (or a map of vectors,
   for a map of functions; gen.inference.kernel/run's :keep). See the
   namespace for the options."
  ([trace selection opts] (sample trace selection opts nil))
  ([trace selection opts key]
   (let [{:keys [samples warmup keep] :as opts} (settings opts)
         _ (when-not (nat-int? samples)
             (throw (ex-info (str "sample needs :samples ≥ 0, not " (pr-str samples)) {:reason :invalid-option :samples samples})))
         [read finish] (if (some? keep) (kernel/keeping keep) [identity nil])
         [k-warmup k-draws] (rng/split (or key (:key opts) (gfi/next-key)))
         adapted (if (pos? (long warmup))
                   (adapt trace selection opts k-warmup)
                   {:trace trace :eps (:eps opts) :inv-metric (mcmc/inv-metric trace selection (:inv-metric opts))
                    :stats []})
         {:keys [eps inv-metric]} adapted
         draw-keys (if (pos? (long samples)) (rng/split k-draws samples) [])]
     (loop [i 0 t (:trace adapted) kept (transient []) stats (transient [])]
       (if (= i samples)
         (let [stats (persistent! stats)]
           (assoc (if finish {:draws (finish (persistent! kept))} {:traces (persistent! kept)})
                  :stats stats :eps eps :inv-metric inv-metric
                  :divergences (divergences stats)
                  :warmup {:stats (:stats adapted) :divergences (divergences (:stats adapted))}))
         (let [[t s] (transition t selection opts eps inv-metric (draw-keys i))
               t (gfi/eval-trace! t)]
           (recur (inc i) t (conj! kept (read t)) (conj! stats s))))))))
