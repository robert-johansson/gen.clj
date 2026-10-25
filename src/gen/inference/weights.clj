(ns gen.inference.weights
  "Log weights of a weighted collection of samples (importance samples,
   particles): normalization, effective sample size, the log marginal
   likelihood estimate, and resampling.

   Log weights are a seq of numbers (or 0-d arrays) or a 1-d array. They are
   read on the host, so these functions belong at inference-loop boundaries,
   never inside a model. Weights that make normalization meaningless fail
   loudly, with ::invalid-weights in the ex-data:

     :empty       there are no weights
     :impossible  every weight is -Inf (no sample is possible)
     :nan         some weight is NaN
     :infinite    some weight is +Inf

   One -Inf weight among finite ones is fine: that sample has probability 0
   and is never resampled.

   Resamplers return a vector of n indices. They take the key as an optional
   last argument; omitting it draws one from the implicit key source."
  {:api :stable}
  (:require [gen.array :as array]
            [gen.gfi :as gfi]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- host-weights
  "The log weights as a double[]."
  ^doubles [log-weights]
  (cond
    (sequential? log-weights)
    (if (every? array/host? log-weights)
      (double-array log-weights)
      (double-array (array/->clj (array/stack (vec log-weights)))))

    (= 1 (array/ndim log-weights)) (double-array (array/->clj log-weights))

    :else (throw (ex-info "Log weights must be a seq of numbers or a 1-d array"
                          {:shape (array/shape log-weights)}))))

(defn- invalid! [reason ^doubles w]
  (throw (ex-info (case reason
                    :empty "There are no log weights to normalize."
                    :impossible (str "All " (alength w) " log weights are -Inf: no sample is possible "
                                     "under the observations. Check the observations against the "
                                     "model's support, or use a better proposal.")
                    :nan "A log weight is NaN, so the weights cannot be normalized."
                    :infinite "A log weight is +Inf, so the weights cannot be normalized.")
                  {::invalid-weights reason :n (alength w)})))

(defn- log-total
  "log Σ exp(w), checked: finite, or a loud error."
  ^double [^doubles w]
  ;; Math/max propagates NaN
  (let [m (areduce w i m Double/NEGATIVE_INFINITY (Math/max m (aget w i)))]
    (cond
      (zero? (alength w)) (invalid! :empty w)
      (Double/isNaN m) (invalid! :nan w)
      (== m Double/POSITIVE_INFINITY) (invalid! :infinite w)
      (== m Double/NEGATIVE_INFINITY) (invalid! :impossible w)
      :else (+ m (Math/log (areduce w i s 0.0 (+ s (Math/exp (- (aget w i) m)))))))))

(defn log-sum-exp
  "log Σ exp(w) of the log weights: -Inf when they are all -Inf (a total
   of 0), and a loud error for NaN or +Inf."
  ^double [log-weights]
  (let [w (host-weights log-weights)]
    (if (and (pos? (alength w)) (every? #(== ##-Inf %) w))
      ##-Inf
      (log-total w))))

(defn normalize
  "Gen.jl's normalize_weights: {:log-total log Σ exp(w),
   :log-weights the normalized log weights, a vector}."
  [log-weights]
  (let [w (host-weights log-weights)
        t (log-total w)]
    {:log-total t
     :log-weights (mapv #(- (double %) t) w)}))

(defn log-mean-exp
  "log((1/n) Σ exp(w)): the log marginal likelihood estimate of n importance
   samples with these log weights."
  ^double [log-weights]
  (let [w (host-weights log-weights)]
    (- (log-total w) (Math/log (alength w)))))

(defn- probabilities
  "The normalized weights exp(w - log-total) as a double[]."
  ^doubles [log-weights]
  (let [w (host-weights log-weights)
        t (log-total w)]
    (amap w i _ (Math/exp (- (aget w i) t)))))

(defn ess
  "The effective sample size 1 / Σ p², where p are the normalized weights:
   n for equal weights, 1 when one sample carries all the weight. Accepts
   normalized or unnormalized log weights."
  ^double [log-weights]
  (let [p (probabilities log-weights)]
    (/ 1.0 (areduce p i s 0.0 (+ s (* (aget p i) (aget p i)))))))

;; ---------------------------------------------------------------------------
;; Resampling
;; ---------------------------------------------------------------------------

(defn- cdf
  "Running sums of the probabilities."
  ^doubles [^doubles p]
  (let [c (double-array (alength p))]
    (loop [i 0 s 0.0]
      (when (< i (alength p))
        (let [s (+ s (aget p i))]
          (aset c i s)
          (recur (inc i) s))))
    c))

(defn- last-positive ^long [^doubles p]
  (loop [i (dec (alength p))]
    (if (pos? (aget p i)) i (recur (dec i)))))

(defn- search
  "The index i whose interval [c(i-1), c(i)) contains u·c(n-1), for u in
   [0, 1): the first i with c(i) > u·c(n-1). A sample with probability 0 has
   an empty interval, so it is never found. If rounding puts the target past
   the end, the last sample with positive probability."
  ^long [^doubles c ^long fallback ^double u]
  (let [t (* u (aget c (dec (alength c))))]
    (loop [lo 0 hi (alength c)]
      (if (< lo hi)
        (let [mid (unsigned-bit-shift-right (+ lo hi) 1)]
          (if (> (aget c mid) t) (recur lo mid) (recur (inc mid) hi)))
        (if (< lo (alength c)) lo fallback)))))

(defn- draw
  "The index for each uniform position in us."
  [^doubles p us]
  (let [c (cdf p)
        fallback (last-positive p)]
    (mapv #(search c fallback (double %)) us)))

(defn- uniforms
  "n uniforms in [0, 1) as a vector."
  [key n]
  (if (zero? n) [] (array/->clj (rng/uniform key [n]))))

(defn multinomial
  "n indices drawn independently in proportion to the weights."
  ([log-weights n] (multinomial log-weights n (gfi/next-key)))
  ([log-weights n key]
   (draw (probabilities log-weights) (uniforms key n))))

(defn systematic
  "n indices at the evenly spaced positions (j + u)/n, for one uniform u.
   Each sample i appears floor(n·p_i) or ceil(n·p_i) times."
  ([log-weights n] (systematic log-weights n (gfi/next-key)))
  ([log-weights n key]
   (let [u (double (array/item (rng/uniform key)))
         n (long n)]
     (draw (probabilities log-weights) (map #(/ (+ (double %) u) n) (range n))))))

(defn stratified
  "n indices at the positions (j + u_j)/n, one independent uniform per
   stratum."
  ([log-weights n] (stratified log-weights n (gfi/next-key)))
  ([log-weights n key]
   (let [n (long n)]
     (draw (probabilities log-weights)
           (map-indexed (fn [j u] (/ (+ (double j) (double u)) n)) (uniforms key n))))))

(defn- whole-copies
  "residual's deterministic copies of a sample, for x = n·p_i: floor(x),
   or ceil(x) when x lies within a relative 1e-9 below it (DESIGN.md §9,
   the ten decisions' eighth). n·p_i is a rounded product, so floor alone
   lost copies: 10 equal weights gave x = 0.9999999999999998 and no copy,
   and residual resampling silently became multinomial. A sample rounded
   up has a residual within that of 1, and the residuals sum to the
   integer n - Σ floor(x), so no more than that many round up: the copies
   stay at most n."
  ^long [^double x]
  (let [c (Math/ceil x)]
    (long (if (<= (- c x) (* 1e-9 x)) c (Math/floor x)))))

(defn residual
  "whole-copies of n·p_i of each sample i (floor(n·p_i), rounded up within
   a relative 1e-9 of the next integer), then the remaining indices drawn
   multinomially in proportion to the residuals n·p_i less those copies."
  ([log-weights n] (residual log-weights n (gfi/next-key)))
  ([log-weights n key]
   (let [p (probabilities log-weights)
         n (long n)
         copies (mapv #(whole-copies (* n (double %))) p)
         fixed (into [] (mapcat (fn [i c] (repeat c i)) (range) copies))
         r (- n (count fixed))]
     (when (neg? r)
       (throw (ex-info (str "residual: " (count fixed) " whole copies of " n " draws") {:copies copies :n n})))
     (if (pos? r)
       (let [residuals (double-array (map (fn [pi c] (max 0.0 (- (* n (double pi)) (double c)))) p copies))]
         (into fixed (draw residuals (uniforms key r))))
       fixed))))

(def ^:private resamplers
  {:multinomial multinomial :systematic systematic :stratified stratified :residual residual})

(defn resample
  "n indices drawn with method: :multinomial, :systematic, :stratified or
   :residual. All are unbiased: each index i is expected n·p_i times."
  ([method log-weights n] (resample method log-weights n (gfi/next-key)))
  ([method log-weights n key]
   (if-let [f (resamplers method)]
     (f log-weights n key)
     (throw (ex-info (str "Unknown resampling method " method)
                     {:method method :methods (set (keys resamplers))})))))
