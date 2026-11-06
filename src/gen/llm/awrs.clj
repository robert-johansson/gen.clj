(ns gen.llm.awrs
  "Adaptive weighted rejection sampling (AWRS; Lipkin et al., COLM 2025):
   a token from an LM's row restricted to the tokens a condition accepts,
   checking only the tokens it draws, with an unbiased estimate of the
   accepted mass Z = Σ_t p(t)·1[t accepted]. Token SMC's :awrs proposal
   (gen.llm.smc) uses it where a potential has no cheap mask: a check per
   drawn token, never a sweep of the vocabulary.

     (sample row weigh key opts)  → {:token t :weight w :log-z log Ẑ :checks k}

   row is the LM's next-token log-probabilities (a float array, normalized,
   as gen.llm.lm/next-logprobs gives); weigh a function of a token, its log
   weight, -Inf for a token the condition rejects (a potential's twist:
   token SMC's prefix difference); key a gen.random key. The result's
   :token is drawn from p restricted to the accepted tokens and :log-z is
   log Ẑ, whose exponential has expectation Z; together they are properly
   weighted: E[Ẑ·f(token)] = Σ_t p(t)·1[accepted]·f(t), for any f. :weight
   is weigh's value at :token (the condition's checks are kept, so the
   accepted token is not weighed twice), :checks the number of weigh calls.
   Where nothing is accepted (or a budget runs out), log Ẑ is -Inf, an
   estimate of Z = 0 (a token is returned still: the last one drawn).

   opts, the estimator (the paper's appendix H.3):

   - :method :recursive (the default; RAWRS, Def. 10): draws without
     replacement until the first accepted token x_n, then one more draw,
     x_{n+1}; Ẑ is the mass not yet drawn before x_n (as a share of the
     row's), m_n, when x_{n+1} is accepted too, and p(x_n) when it is not.
     From Z = p(x)·1[x] + (1 - p(x))·E[1[X] | X ≠ x], applied down the
     draws. It checks every token it draws, one more than those it draws
     before accepting.
   - :method :geometric (GAWRS, Def. 9): draws without replacement of the
     rejected tokens only, and counts the samples a draw with replacement
     would have made: before each draw, a geometric number of repeats of
     rejected tokens (phantoms, drawn, never checked). It stops at
     :max-accepts accepts (2, the least that leaves the estimate unbiased:
     the negative binomial's (s - 1)/(n - 1)) and returns the first.
   - :max-rejects R (default unbounded): the most rejections a run makes
     (phantoms included, for :geometric), so at most R checks (:recursive:
     its R-th draw is the base case, and an accept before it is followed
     by one draw more) or R + :max-accepts - 1 (:geometric: it stops at its
     R-th rejection or its last accept, whichever comes first), R rounded
     up; a run that reaches it stops with the estimator's value there
     (Def. 10's base case, Def. 8's s/(n - 1)), which may be 0. Still
     properly weighted.

   Randomness is purely functional: draw i uses the uniform gen.random
   derives from (fold-in key (* 2 i)), its phantoms (fold-in key (+ (* 2
   i) 1)), on the host, so the same key draws the same tokens from the
   same row. A weigh that returns NaN is loud (:invalid-weight).

   Cost: one exp pass over the row a sample, then a sum and a partial
   walk of the row's masses a draw (about 1 ms at 151,936 tokens, two
   draws, on an M4)."
  {:api :experimental}
  (:require [gen.array :as array]
            [gen.random :as rng])
  (:import (java.util BitSet)))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- uniform
  "The uniform in [0, 1) gen.random derives from key, a host double."
  ^double [key]
  (double (rng/uniform key [] {:backend (array/backend-named :jvm) :dtype :float64})))

(defn- masses
  "exp(row[t]) for every token: the one exp pass a sample makes over its
   row, the masses its draws read."
  ^doubles [^floats row]
  (let [n (alength row)
        out (double-array n)]
    (dotimes [t n] (aset out t (Math/exp (aget row t))))
    out))

(defn- mass-left
  "Σ ps[t] over the tokens not in removed, in the tokens' order."
  ^double [^doubles ps ^BitSet removed]
  (let [n (alength ps)]
    (loop [t 0 total 0.0]
      (if (< t n)
        (recur (inc t) (if (.get removed t) total (+ total (aget ps t))))
        total))))

(defn- draw-left
  "A token not in removed, of probability its share of the mass left
   (total), by the inverse CDF at u over the masses ps: -1 when none is
   left."
  ^long [^doubles ps ^BitSet removed ^double total ^double u]
  (let [n (alength ps)
        target (* u total)]
    (loop [t 0 acc 0.0 last -1]
      (if (< t n)
        (if (.get removed t)
          (recur (inc t) acc last)
          (let [p (aget ps t)
                acc (+ acc p)]
            (if (and (< target acc) (pos? p))
              t
              (recur (inc t) acc (if (pos? p) t last)))))
        last))))

(defn- checked
  "weigh's value at token t, loud for NaN."
  ^double [weigh ^long t]
  (let [w (double (weigh t))]
    (when (Double/isNaN w)
      (throw (ex-info (str "awrs: the weight of token " t " is NaN") {:reason :invalid-weight :token t})))
    w))

(defn- accepted? [^double w] (> w Double/NEGATIVE_INFINITY))

(defn- draw-key [key ^long i] (rng/fold-in key (* 2 i)))

(defn- phantom-key [key ^long i] (rng/fold-in key (inc (* 2 i))))

(defn- result [t w log-z checks]
  {:token t :weight w :log-z log-z :checks checks})

;; ---------------------------------------------------------------------------
;; Recursive AWRS (Def. 10)
;; ---------------------------------------------------------------------------

(defn- recursive
  "RAWRS: draws without replacement, the first accepted token and one more
   draw; m_i, the mass left before draw i as a share of the row's."
  [^floats row weigh key ^double max-rejects]
  (let [ps (masses row)
        removed (BitSet. (alength ps))
        whole (mass-left ps removed)]
    (loop [i 0 checks 0 last-t -1 left whole]
      (let [t (draw-left ps removed left (uniform (draw-key key i)))]
        (if (neg? t)
          ;; every token of positive probability rejected: Z is 0
          (result last-t Double/NEGATIVE_INFINITY Double/NEGATIVE_INFINITY checks)
          (let [w (checked weigh t)
                checks (inc checks)
                m (/ left whole)]
            (cond
              ;; the budget's draw: the base case, 1[x] times m
              (>= (inc i) max-rejects)
              (result t w (if (accepted? w) (Math/log m) Double/NEGATIVE_INFINITY) checks)

              (accepted? w)
              (let [p (/ (aget ps t) whole)]
                (.set removed t)
                (let [left' (mass-left ps removed)
                      t' (draw-left ps removed left' (uniform (draw-key key (inc i))))]
                  (if (neg? t')
                    ;; nothing left to draw: 1 - q_n is 0, so Ẑ is m_n
                    (result t w (Math/log m) checks)
                    (let [w' (checked weigh t')]
                      (result t w (Math/log (if (accepted? w') m p)) (inc checks))))))

              :else (do (.set removed t) (recur (inc i) checks t (mass-left ps removed))))))))))

;; ---------------------------------------------------------------------------
;; Geometric AWRS (Def. 9)
;; ---------------------------------------------------------------------------

(defn- phantoms
  "How many samples with replacement land on the rejected tokens (mass q,
   of the row's) before one lands elsewhere: geometric, P(k) = q^k (1 - q),
   from u."
  ^double [^double q ^double u]
  (cond (<= q 0.0) 0.0
        (>= q 1.0) Double/POSITIVE_INFINITY
        :else (Math/floor (/ (Math/log (- 1.0 u)) (Math/log q)))))

(defn- geometric
  "GAWRS: a run without replacement of the rejected tokens, read as a run
   with replacement by counting phantoms; stops at max-accepts accepts or
   max-rejects rejections, Clipped WRS's estimate on the counts (Def. 8)."
  [^floats row weigh key max-rejects max-accepts]
  (let [max-rejects (double max-rejects)
        max-accepts (long max-accepts)
        ps (masses row)
        removed (BitSet. (alength ps))
        whole (mass-left ps removed)
        weights (java.util.HashMap.)]
    ;; left: the mass not removed, summed again only when a draw removes one
    (loop [i 0 s 0 r 0.0 checks 0 first-t -1 first-w Double/NEGATIVE_INFINITY last-t -1 left whole]
      (let [k (phantoms (- 1.0 (/ left whole)) (uniform (phantom-key key i)))
            stop-rejected (fn [^long checks]
                            ;; stopped on a rejection, at the budget: s/(n - 1)
                            (let [n (+ (double s) max-rejects)]
                              (if (zero? s)
                                (result last-t Double/NEGATIVE_INFINITY Double/NEGATIVE_INFINITY checks)
                                (result first-t first-w (Math/log (/ (double s) (- n 1.0))) checks))))]
        (if (>= (+ r k) max-rejects)
          (stop-rejected checks)
          (let [r (+ r k)
                t (draw-left ps removed left (uniform (draw-key key i)))]
            (if (neg? t)
              (result last-t Double/NEGATIVE_INFINITY Double/NEGATIVE_INFINITY checks)
              (let [seen (.get weights t)
                    w (if seen (double seen) (checked weigh t))
                    checks (if seen checks (inc checks))]
                (when-not seen (.put weights t w))
                (if (accepted? w)
                  (let [s (inc s)
                        [first-t first-w] (if (zero? (dec s)) [t w] [first-t first-w])]
                    (if (== s max-accepts)
                      ;; stopped on an accept: (s - 1)/(n - 1)
                      (result first-t first-w (Math/log (/ (double (dec s)) (- (+ (double s) r) 1.0))) checks)
                      (recur (inc i) s r checks (long first-t) (double first-w) t left)))
                  (let [r (inc r)]
                    (.set removed t)
                    (if (>= r max-rejects)
                      (stop-rejected checks)
                      (recur (inc i) s r checks first-t first-w t (mass-left ps removed)))))))))))))

;; ---------------------------------------------------------------------------
;; The sampler
;; ---------------------------------------------------------------------------

(defn check-options
  "opts, checked: :method (:recursive or :geometric), :max-rejects (at
   least 1), :max-accepts (:geometric's, an integer of at least 2); loud
   otherwise (:invalid-option)."
  [{:keys [method max-rejects max-accepts] :or {method :recursive} :as opts}]
  (let [bad (fn [msg] (throw (ex-info (str "awrs: " msg) {:reason :invalid-option :opts opts})))]
    (when-not (or (nil? opts) (map? opts)) (bad (str "the options are a map, not " (pr-str opts))))
    (when-not (#{:recursive :geometric} method) (bad (str ":method is :recursive or :geometric, not " (pr-str method))))
    (when-not (or (nil? max-rejects) (and (number? max-rejects) (<= 1.0 (double max-rejects))))
      (bad (str ":max-rejects is at least 1, not " (pr-str max-rejects))))
    (when (and max-accepts (= :recursive method)) (bad ":max-accepts is :geometric's"))
    (when-not (or (nil? max-accepts) (and (int? max-accepts) (<= 2 (long max-accepts))))
      (bad (str ":max-accepts is an integer of at least 2, not " (pr-str max-accepts))))
    opts))

(defn sample
  "A token drawn from row restricted to the tokens weigh accepts (a weight
   above -Inf), and an unbiased estimate of their mass: {:token :weight
   :log-z :checks}. opts: :method (:recursive, :geometric), :max-rejects,
   :max-accepts (see the namespace's docstring)."
  ([row weigh key] (sample row weigh key {}))
  ([^floats row weigh key opts]
   (let [{:keys [method max-rejects max-accepts] :or {method :recursive}} (check-options opts)
         max-rejects (double (or max-rejects Double/POSITIVE_INFINITY))]
     (when-not (and (instance? (Class/forName "[F") row) (pos? (alength row)))
       (throw (ex-info "awrs: the row is a non-empty float array of log-probabilities" {:reason :invalid-row})))
     (if (= :recursive method)
       (recursive row weigh key max-rejects)
       (geometric row weigh key max-rejects (or max-accepts 2))))))
