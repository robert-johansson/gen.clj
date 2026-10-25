(ns gen.inference.diagnostics
  "Convergence diagnostics for MCMC: effective sample sizes, R-hat and
   Monte Carlo standard errors, as Vehtari, Gelman, Simpson, Carpenter and
   Bürkner (2021), \"Rank-normalization, folding, and localization: an
   improved R-hat for assessing convergence of MCMC\", define them, and as
   ArviZ and Stan's posterior package compute them.

     (ess-bulk draws)     ESS of the rank-normalized split chains: the bulk's
     (ess-tail draws)     the smaller ESS of the 5% and 95% quantiles
     (ess-mean draws)     ESS of the split chains: the mean's (Stan's basic ESS)
     (r-hat draws)        rank-normalized split R-hat: the larger of the
                          bulk's and the folded draws'
     (split-r-hat draws)  split R-hat of the draws themselves
     (mcse-mean draws)    Monte Carlo standard error of the mean
     (mcse-sd draws)      Monte Carlo standard error of the standard deviation
     (summary draws)      mean, sd, quantiles, MCSEs, bulk and tail ESS, R-hat
     (draws traces addr)  a choice's draws from chains of traces

   Draws are one quantity's host numbers: a vector of chains, each a vector
   of draws ([chain][draw]), or one chain, a vector of numbers. These run at
   inference-loop boundaries:

     (let [runs (mapv #(kernel/run k (start %) {:samples 1000 :key %}) keys)]
       (summary (draws (map :traces runs) :x)))

   These effective sample sizes are of a chain's autocorrelated draws: not
   the effective sample size of weighted samples (gen.inference.weights/ess
   and gen.inference.particle-filter/ess), which are independent but
   unequally weighted.

   Every ESS splits each chain in two (dropping an odd chain's middle draw),
   estimates the autocorrelations from the split chains' autocovariances and
   the variance between them, and sums them with Geyer's initial monotone
   sequence, as Stan does. For S split draws it is at most S·log10(S) (an
   antithetic chain's exceeds S, a chain of fewer than 10 draws gets the
   cap). The autocovariances are direct sums, computed lag by lag only as
   far as Geyer's truncation reaches: O(S·T) for truncation at lag T, a few
   autocorrelation times for chains that mix, up to half a chain's length
   (O(S·n)) for chains that do not.

   As ArviZ, constant draws (a quantile's indicator included) have the
   nominal ESS S; when every draw is the same, R-hat and mcse-sd are 0/0,
   NaN (and mcse-mean 0), and chains stuck at different values have R-hat
   +Inf. R-hat of one chain is that of its halves, as Stan's posterior
   computes it (ArviZ's is NaN).

   Invalid draws are loud errors, with :reason in the ex-data:

     :invalid-draws   not a non-empty vector of chains or of numbers
     :non-numeric     a draw is not a number (read arrays with draws)
     :non-finite      a draw is NaN or infinite
     :unequal-chains  the chains differ in length
     :too-few-draws   the chains have fewer than 4 draws
     :out-of-range    the draws' second moments overflow a double, or for
                      an ESS underflow to 0 (spreads past about 1e±154;
                      1e±77 for mcse-sd)
     :invalid-prob    (summary) a quantile's probability is not in [0, 1]
     :non-scalar      (draws) a value is not a number, boolean or 0-d array"
  {:api :stable}
  (:require [gen.array :as array]
            [gen.array.special :as special]
            [gen.trace :as trace])
  (:import (java.util Arrays)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Draws
;; ---------------------------------------------------------------------------

(defn- invalid! [reason message data]
  (throw (ex-info message (assoc data :reason reason))))

(def ^:private ^:const min-draws 4)

(defn- chain-array
  "Chain c's draws as a double[], each a finite number."
  ^doubles [^long c chain]
  (let [out (double-array (count chain))]
    (loop [j 0 xs (seq chain)]
      (when xs
        (let [x (first xs)]
          (cond (not (number? x))
                (invalid! :non-numeric (str "Draw " j " of chain " c " is not a number: " (pr-str x)
                                            "; read a choice's values on the host with draws")
                          {:chain c :draw j :value x})
                (not (Double/isFinite (double x)))
                (invalid! :non-finite (str "Draw " j " of chain " c " is " x)
                          {:chain c :draw j :value x}))
          (aset out j (double x))
          (recur (inc j) (next xs)))))
    out))

(defn- chains
  "The draws as a vector of double[], one per chain, checked."
  [draws]
  (when-not (and (sequential? draws) (seq draws))
    (invalid! :invalid-draws "Draws are a non-empty vector of chains, or one chain of numbers"
              {:draws draws}))
  (let [cs (vec (map-indexed (fn [c chain]
                               (if (sequential? chain)
                                 (chain-array c chain)
                                 (invalid! :invalid-draws (str "Chain " c " is not a vector of draws")
                                           {:chain c :value chain})))
                             (if (sequential? (first draws)) draws [draws])))
        lengths (mapv #(alength ^doubles %) cs)]
    (when-not (apply = lengths)
      (invalid! :unequal-chains "The chains differ in length" {:lengths lengths}))
    (when (< (long (first lengths)) min-draws)
      (invalid! :too-few-draws (str "The chains have " (first lengths) " draws, fewer than " min-draws)
                {:draws (first lengths) :minimum min-draws}))
    cs))

(defn- pooled
  "The chains' draws in one double[], chain after chain."
  ^doubles [cs]
  (let [n (alength ^doubles (first cs))
        out (double-array (* (count cs) n))]
    (dotimes [c (count cs)]
      (System/arraycopy (cs c) 0 out (* c n) n))
    out))

(defn- unpool
  "x, the draws of m chains pooled, as the m chains."
  [^doubles x ^long m]
  (let [n (quot (alength x) m)]
    (mapv (fn [^long c] (Arrays/copyOfRange x (* c n) (* (inc c) n))) (range m))))

(defn- split
  "Each chain as two: its first ⌊n/2⌋ draws and its last ⌊n/2⌋ (an odd
   chain's middle draw dropped)."
  [cs]
  (vec (mapcat (fn [^doubles x]
                 (let [n (alength x) h (quot n 2)]
                   [(Arrays/copyOfRange x 0 h) (Arrays/copyOfRange x (- n h) n)]))
               cs)))

(defn- sorted ^doubles [^doubles x] (doto (aclone x) (Arrays/sort)))

;; ---------------------------------------------------------------------------
;; Statistics of host doubles
;; ---------------------------------------------------------------------------

;; a loop used as a value boxes (Clojure makes it a closure), so each
;; reduction is a function of its own

(defn- total ^double [^doubles x] (areduce x i s 0.0 (+ s (aget x i))))

(defn- squared-deviations ^double [^doubles x ^double mu]
  (areduce x i s 0.0 (let [d (- (aget x i) mu)] (+ s (* d d)))))

(defn- mean ^double [^doubles x] (/ (total x) (alength x)))

(defn- variance
  "The sample variance (divided by n - 1)."
  ^double [^doubles x]
  (/ (squared-deviations x (mean x)) (dec (alength x))))

(defn- median
  "The middle value of x, or the mean of the two middle values."
  ^double [^doubles x]
  (let [s (sorted x)
        h (quot (alength s) 2)]
    (if (odd? (alength s))
      (aget s h)
      (/ (+ (aget s (dec h)) (aget s h)) 2.0))))

(defn- quantile
  "The p quantile of sorted values, interpolated linearly between order
   statistics (Hyndman and Fan's type 7, the default of numpy and R), as
   numpy computes it."
  ^double [^doubles sorted ^double p]
  (let [last-index (dec (alength sorted))
        h (* p last-index)
        lo (long (Math/floor h))
        t (- h lo)
        a (aget sorted lo)
        b (aget sorted (min (inc lo) last-index))]
    (if (< t 0.5)
      (+ a (* t (- b a)))
      (- b (* (- 1.0 t) (- b a))))))

(defn- below
  "How many of the sorted values are below v (inclusive?: not above v)."
  ^long [^doubles sorted ^double v inclusive?]
  (loop [lo 0 hi (alength sorted)]
    (if (< lo hi)
      (let [mid (unsigned-bit-shift-right (+ lo hi) 1)
            x (aget sorted mid)]
        (if (if inclusive? (<= x v) (< x v))
          (recur (inc mid) hi)
          (recur lo mid)))
      lo)))

(defn- ranks
  "Each value's rank among xs, from 1; tied values share the mean of their
   ranks."
  ^doubles [^doubles xs]
  (let [s (sorted xs)]
    (amap xs i _ (let [v (aget xs i)]
                   (* 0.5 (+ 1.0 (below s v false) (below s v true)))))))

(defn- normal-quantile
  "Φ⁻¹(p), the standard normal quantile."
  ^double [^double p]
  (* (Math/sqrt 2.0) (special/erfinv (- (* 2.0 p) 1.0))))

;; ---------------------------------------------------------------------------
;; Transformed chains
;; ---------------------------------------------------------------------------

(defn- z-scale
  "The chains rank-normalized together: a draw of average rank r among all
   S draws becomes Φ⁻¹((r - 3/8)/(S + 1/4)) (Blom's)."
  [cs]
  (let [x (pooled cs)
        s (alength x)
        r (ranks x)]
    (unpool (amap r i _ (normal-quantile (/ (- (aget r i) 0.375) (+ s 0.25)))) (count cs))))

(defn- fold
  "The chains folded: each draw's distance from the median of all of them."
  [cs]
  (let [med (median (pooled cs))]
    (mapv (fn [^doubles x] (amap x i _ (Math/abs (- (aget x i) med)))) cs)))

(defn- indicator
  "The chains as 1 where a draw is at most q, else 0."
  [cs ^double q]
  (mapv (fn [^doubles x] (amap x i _ (if (<= (aget x i) q) 1.0 0.0))) cs))

;; ---------------------------------------------------------------------------
;; ESS and R-hat of chains
;; ---------------------------------------------------------------------------

(defn- lag-product
  "Σ_i x_i·x_{i+t}."
  ^double [^doubles x ^long t]
  (let [n (- (alength x) t)]
    (loop [i 0 s 0.0]
      (if (< i n)
        (recur (inc i) (+ s (* (aget x i) (aget x (+ i t)))))
        s))))

(defn- autocorrelation
  "The autocorrelation at lag t of chains centered at their means, w the
   mean of their variances and var+ the estimate of the marginal variance
   (w·(n - 1)/n plus the variance of their means): 1 - (w - γ_t)/var+, for
   γ_t their autocovariances averaged."
  ^double [centered ^double w ^double var+ ^long t]
  (let [n (alength ^doubles (first centered))
        gamma (/ (double (reduce + (map #(lag-product % t) centered)))
                 (* n (count centered)))]
    (- 1.0 (/ (- w gamma) var+))))

(defn- autocorrelation-time
  "τ = -1 + 2·Σ ρ_t by Geyer's initial monotone sequence, as Stan and ArviZ
   compute it: the pair sums ρ_2k + ρ_2k+1 are added while they are
   positive, each capped at the one before it, as far as lag n - 3; the
   first pair left out adds its even term when that is positive, Stan's
   improvement for antithetic chains."
  ^double [centered ^double w ^double var+]
  (let [n (alength ^doubles (first centered))
        p0 (+ 1.0 (autocorrelation centered w var+ 1))]
    ;; sum: the monotone pair sums before pair k - 1, monotone and pair:
    ;; pair k - 1's monotone and raw sums, even its even term
    (loop [k 1 sum 0.0 monotone p0 even 1.0 pair p0]
      (if (and (pos? pair) (< (+ (* 2 k) 2) n))
        (let [e (autocorrelation centered w var+ (* 2 k))
              p (+ e (autocorrelation centered w var+ (inc (* 2 k))))]
          (recur (inc k) (+ sum monotone) (Math/min monotone p) e p))
        (+ -1.0 (* 2.0 sum) (if (or (pos? even) (>= pair 0.0)) even 0.0))))))

(defn- representable!
  "Throws unless ok?: the draws' second moments overflowed a double (or
   underflowed to 0 for draws that differ), so no ESS or R-hat comes of
   them."
  [ok? data]
  (when-not ok?
    (invalid! :out-of-range "The draws' second moments overflow or underflow a double: rescale the draws"
              data)))

(defn- constant? [cs]
  (let [v (aget ^doubles (first cs) 0)]
    (every? (fn [^doubles x] (areduce x i same true (and same (== (aget x i) v)))) cs)))

(defn- ess*
  "The effective sample size of two or more chains of equal length, as Stan
   computes it: S/τ, at most S·log10(S), for S draws in all; S when they
   are constant."
  ^double [cs]
  (let [m (count cs)
        n (alength ^doubles (first cs))
        size (* m n)]
    (if (constant? cs)
      (double size)
      (let [means (double-array (map mean cs))
            centered (mapv (fn [^doubles x ^double mu] (amap x i _ (- (aget x i) mu))) cs means)
            w (/ (double (reduce + (map #(lag-product % 0) centered))) (* m (dec n)))
            var+ (+ (/ (* w (dec n)) n) (variance means))]
        (representable! (and (Double/isFinite var+) (pos? var+)) {:variance var+})
        (/ size (Math/max (autocorrelation-time centered w var+) (/ 1.0 (Math/log10 size))))))))

(defn- r-hat*
  "Gelman and Rubin's potential scale reduction of two or more chains of
   equal length n: √((n - 1)/n + B/(n·W)), W the mean of the chains'
   variances and B/n the variance of their means."
  ^double [cs]
  (let [n (alength ^doubles (first cs))
        w (mean (double-array (map variance cs)))
        b (* n (variance (double-array (map mean cs))))]
    (representable! (Double/isFinite (+ w b)) {:within w :between b})
    (Math/sqrt (/ (+ (/ b w) (dec n)) n))))

(defn- quantile-ess
  "The ESS of the p quantile: of the indicator of draws at most the pooled
   draws' p quantile."
  ^double [cs ^double p]
  (ess* (split (indicator cs (quantile (sorted (pooled cs)) p)))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn ess-bulk
  "The bulk effective sample size: the ESS of the rank-normalized split
   chains, how many independent draws the chains are worth for the bulk
   of the distribution. Rank normalization makes it defined for any
   distribution, one without a mean too."
  ^double [draws]
  (ess* (z-scale (split (chains draws)))))

(defn ess-tail
  "The tail effective sample size: the smaller of the ESSs of the 5% and
   95% quantiles, each the ESS of the indicator of draws at or below the
   quantile of all of them."
  ^double [draws]
  (let [cs (chains draws)]
    (Math/min (quantile-ess cs 0.05) (quantile-ess cs 0.95))))

(defn ess-mean
  "The effective sample size of the mean: the ESS of the split chains
   themselves (Stan's basic ESS, ArviZ's \"mean\" method). It assumes the
   draws have a finite variance; ess-bulk does not."
  ^double [draws]
  (ess* (split (chains draws))))

(defn r-hat
  "The rank-normalized split R-hat: the larger of the split R-hats of the
   rank-normalized draws (the bulk) and of the rank-normalized folded
   draws, their distances from the median (the tails). Near 1 when the
   chains agree; Vehtari et al. recommend using the draws only when it is
   below 1.01."
  ^double [draws]
  (let [cs (split (chains draws))
        bulk (r-hat* (z-scale cs))
        tails (r-hat* (z-scale (fold cs)))]
    ;; draws of two values on either side of the median fold to a constant,
    ;; whose R-hat is NaN; the bulk's stands, as in ArviZ
    (if (Double/isNaN tails) bulk (Math/max bulk tails))))

(defn split-r-hat
  "The split R-hat of the draws themselves, without rank normalization:
   Gelman et al.'s (BDA3), the R-hat of the split chains."
  ^double [draws]
  (r-hat* (split (chains draws))))

(defn mcse-mean
  "The Monte Carlo standard error of the draws' mean: sd/√ess-mean."
  ^double [draws]
  (let [cs (chains draws)]
    (/ (Math/sqrt (variance (pooled cs))) (Math/sqrt (ess* (split cs))))))

(defn mcse-sd
  "The Monte Carlo standard error of the draws' standard deviation, by the
   delta method, as ArviZ: for d the draws less their mean, √(Var(d²) /
   (4·E[d²]·ess)), ess the ESS of d²."
  ^double [draws]
  (let [cs (chains draws)
        mu (mean (pooled cs))
        squares (mapv (fn [^doubles x] (amap x i _ (let [d (- (aget x i) mu)] (* d d)))) cs)
        d2 (pooled squares)
        e2 (mean d2)
        e4 (mean (amap d2 i _ (* (aget d2 i) (aget d2 i))))]
    (Math/sqrt (/ (- e4 (* e2 e2)) (* 4.0 e2 (ess* (split squares)))))))

(defn summary
  "A quantity's summary: {:mean :sd :quantiles :mcse-mean :mcse-sd
   :ess-bulk :ess-tail :r-hat}, over all the draws. :sd is the sample
   standard deviation, and :quantiles a sorted map from probability to
   quantile, at 5%, 50% and 95% or at opts' :probs, interpolated linearly
   between order statistics (the default of numpy and R)."
  ([draws] (summary draws {}))
  ([draws {:keys [probs] :or {probs [0.05 0.5 0.95]}}]
   (doseq [p probs]
     (when-not (and (number? p) (<= 0.0 (double p) 1.0))
       (invalid! :invalid-prob (str "A quantile's probability must be in [0, 1], not " (pr-str p))
                 {:prob p})))
   (let [x (pooled (chains draws))
         s (sorted x)]
     {:mean (mean x)
      :sd (Math/sqrt (variance x))
      :quantiles (into (sorted-map) (map (fn [p] [p (quantile s (double p))])) probs)
      :mcse-mean (mcse-mean draws)
      :mcse-sd (mcse-sd draws)
      :ess-bulk (ess-bulk draws)
      :ess-tail (ess-tail draws)
      :r-hat (r-hat draws)})))

;; ---------------------------------------------------------------------------
;; Draws from traces
;; ---------------------------------------------------------------------------

(defn- host-number
  "A choice's value as a host double: a number, a boolean as 0 or 1, or a
   0-d array read on the host."
  ^double [v addr]
  (cond (number? v) (double v)
        (boolean? v) (if v 1.0 0.0)
        (and (array/backend-array? v) (zero? (long (array/ndim v)))) (host-number (array/item v) addr)
        :else (invalid! :non-scalar (str "The value at " (pr-str addr) " is not a scalar: " (pr-str v))
                        {:address addr :value v})))

(defn draws
  "The values at addr of chains of traces, as draws: of one chain, a seq
   of traces (as gen.inference.kernel/run's :traces), a vector of numbers;
   of a seq of chains, a vector of vectors. Each value must be a number, a
   boolean (as 0 or 1) or a 0-d array, read on the host."
  [traces addr]
  (let [chain (fn [ts]
                (when-not (sequential? ts)
                  (invalid! :invalid-draws (str "Not a seq of traces, or of chains of them, but a " (type ts))
                            {:value ts}))
                (mapv #(if (trace/trace? %)
                         (host-number (% addr) addr)
                         (invalid! :invalid-draws (str "Not a trace, but a " (type %)) {:value %}))
                      ts))]
    (if (and (sequential? traces) (not (trace/trace? (first traces))))
      (mapv chain traces)
      (chain traces))))
