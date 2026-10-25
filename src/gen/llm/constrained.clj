(ns gen.llm.constrained
  "Completions under a potential (gen.llm.token's :potential, gen.llm.potential):
   each token's distribution is the LM's times the potential's next-token
   weights, renormalized at its site. Under a grammar (gen.llm.grammar), the
   LM's row masked to the tokens the grammar allows, renormalized: GenMLX's
   grammar-masked decoder.

     (weighted end-tokens row ws)   → [q log-z]: the site's distribution
     (site-row potential end-tokens row tokens)
                                    → q, the site's after tokens, or nil
     (conditional-weight trace)     → log p(x) + w(x) - log q(x), a completion's
                                      trace's importance weight for the global
                                      conditional

   The distribution is local: q(x) = Π_i p(x_i | x_<i) exp(w_i(x_i)) / Z_i,
   w_i the potential's next-token weights and Z_i the normalizer at site i,
   which depends on the tokens before it. That is not the LM conditioned on
   the potential, π(x) ∝ p(x) exp(w(x)) (w the potential's weight of x as a
   whole sequence), unless Π_i Z_i is the same for every x: renormalizing
   token by token favors sequences whose early tokens leave little of the
   LM's mass disallowed. A constrained completion's GFI weights are q's: its
   score is log q(x), assess and generate weigh by q, and its laws hold for
   q. For π, the completion is a proposal: conditional-weight is log p(x) +
   w(x) - log q(x), with p the unconstrained completion's density (the LM's
   ledger, as its score), so its mean over q is Z = Σ p(x) exp(w(x)), and
   for a constraint the probability that the LM's completion satisfies it;
   self-normalized, these weights give expectations under π (importance
   sampling, or token SMC's :masked proposal, gen.llm.smc, whose steps
   weigh the same).

   A token's end is the potential's end: an end token of the completion takes
   the end's weight (the last of logw-next's n-vocab + 1 entries), the
   potential's weight of the tokens before it as a whole sequence.
   A token the potential gives weight 0 (a constraint, or a token kept by
   update or regenerate) scores -Inf: such a trace has density 0, and a
   move to it is rejected. After it the potential's weights are undefined
   (no sequence it begins has weight), so the tokens follow the LM alone,
   drawn from and scored by its rows: every site keeps a distribution. A
   context of positive weight where the potential allows no token (a
   grammar complete there, with no end token to end the completion) is a
   loud error."
  {:api :experimental}
  (:require [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; The site's distribution
;; ---------------------------------------------------------------------------

(defmacro ^:private fold-weights
  "acc folded over a site's tokens t, 0 to n-vocab - 1 (ws holds n-vocab +
   1 weights), w bound to each one's potential weight, an end token's (of
   ends, a sorted int array) the end's: (fold-weights [t w ws ends] [acc
   init] body), body the next acc. One primitive loop, the next end token
   tracked by its index; a function's body, so that it does not box."
  [[t w ws ends] [acc init] & body]
  `(let [ws# ~ws
         ends# ~ends
         n# (dec (alength ^"[F" ws#))
         k# (alength ^"[I" ends#)
         end-weight# (double (aget ^"[F" ws# n#))]
     (loop [~t 0 j# 0 ~acc ~init]
       (if (< ~t n#)
         (let [end?# (and (< j# k#) (== ~t (aget ^"[I" ends# j#)))
               ~w (if end?# end-weight# (double (aget ^"[F" ws# ~t)))]
           (recur (inc ~t) (if end?# (inc j#) j#) (do ~@body)))
         ~acc))))

(defn- max-weight
  "The largest of the site's log weights: the LM's log-probability plus the
   potential's weight."
  ^double [^floats row ^floats ws ^ints ends]
  (fold-weights [t w ws ends] [m Double/NEGATIVE_INFINITY]
    (Math/max m (+ (double (aget row t)) w))))

(defn- sum-exp
  "Σ exp(x - m) over the site's log weights x, skipping those of weight 0."
  ^double [^floats row ^floats ws ^ints ends ^double m]
  (fold-weights [t w ws ends] [s 0.0]
    (let [x (+ (double (aget row t)) w)]
      (if (Double/isInfinite x) s (+ s (Math/exp (- x m)))))))

(defn- allowed-count
  "For weights of 0 and -Inf only (a constraint's: a grammar's mask), how
   many tokens the site allows; -1 when some weight is neither."
  ^long [^floats ws ^ints ends]
  (fold-weights [_ w ws ends] [c 0]
    (if (neg? c)
      c
      (if (== w 0.0) (inc c) (if (== w Double/NEGATIVE_INFINITY) c -1)))))

(defn- disallowed-mass
  "Σ p(t) over the tokens a constraint disallows."
  ^double [^floats row ^floats ws ^ints ends]
  (fold-weights [t w ws ends] [s 0.0]
    (if (Double/isInfinite w) (+ s (Math/exp (aget row t))) s)))

(defn- log-normalizer
  "log Σ_t p(t) exp(w(t)) at the site, -Inf when no token has weight. Under
   a constraint that disallows few tokens, by the row's total of 1 less
   their mass: an exp for each disallowed token, not for each allowed one
   (when that mass is under a half, so that the difference keeps its
   precision)."
  ^double [^floats row ^floats ws ^ints ends]
  (let [n (alength row)
        allowed (allowed-count ws ends)]
    (if (< (- n allowed) allowed)
      (let [d (disallowed-mass row ws ends)]
        (if (< d 0.5)
          (Math/log1p (- d))
          (let [m (max-weight row ws ends)] (+ m (Math/log (sum-exp row ws ends m))))))
      (let [m (max-weight row ws ends)]
        (if (Double/isInfinite m) m (+ m (Math/log (sum-exp row ws ends m))))))))

(defn weighted
  "The LM's row (log-probabilities by token, a float array, normalized)
   times a potential's next-token weights ws (n-vocab + 1 entries, the end
   last; an end token, of end-tokens, takes the end's), renormalized: [q
   log-z], q a new float array of the site's log-probabilities and log-z
   the log of the normalizer, Σ_t p(t) exp(w(t)). [nil -Inf] when the
   weights allow no token. A few passes over the row, allocating only q."
  [end-tokens ^floats row ^floats ws]
  (let [n (alength row)]
    (when-not (== (alength ws) (inc n))
      (throw (ex-info (str "constrained: the potential's next-token weights have " (alength ws) " entries, not "
                           (inc n) " (the LM's vocabulary and the end)")
                      {:reason :invalid-weights :entries (alength ws) :n-vocab n})))
    (let [ends (int-array (sort end-tokens))
          log-z (log-normalizer row ws ends)]
      (if (Double/isInfinite log-z)
        [nil Double/NEGATIVE_INFINITY]
        (let [q (float-array n)]
          (fold-weights [t w ws ends] [_ 0]
            (do (aset q t (unchecked-float (- (+ (double (aget row t)) w) log-z))) 0))
          [q log-z])))))

;; ---------------------------------------------------------------------------
;; A completion's sites (gen.llm.token draws and scores its tokens by them
;; under a potential; its run's state is its own)
;; ---------------------------------------------------------------------------

(defn site-row
  "A site's distribution under potential: row, the LM's log-probabilities
   of the next token after the prompt and tokens, times the potential's
   next-token weights after tokens, renormalized, an end token of
   end-tokens taking the end's weight (weighted's q); nil where the
   potential allows no token."
  ^floats [potential end-tokens ^floats row tokens]
  (first (weighted end-tokens row (potential/logw-next potential tokens))))

(defn check-potential
  "potential, checked for a completion over n-vocab tokens with end-tokens:
   a potential that gives the empty context weight, whose next-token
   weights are the vocabulary's and the end's, and whose end tokens, if it
   declares any (a grammar, which leaves them out of the text), are the
   completion's (potential/check-end-tokens)."
  [potential ^long n-vocab end-tokens]
  (when-not (satisfies? potential/Potential potential)
    (throw (ex-info (str "token/completion: :potential is a potential (gen.llm.potential), not " (pr-str potential))
                    {:reason :invalid-potential :potential potential})))
  (when (Double/isInfinite (potential/prefix potential []))
    (throw (ex-info "token/completion: the potential gives every sequence weight 0 (its prefix of no tokens is -Inf)"
                    {:reason :zero-prefix})))
  (let [n (alength (potential/logw-next potential []))]
    (when-not (== n (inc n-vocab))
      (throw (ex-info (str "token/completion: the potential's next-token weights have " n " entries, not " (inc n-vocab)
                           " (the LM's vocabulary and the end)")
                      {:reason :invalid-weights :entries n :n-vocab n-vocab}))))
  (potential/check-end-tokens "token/completion" potential end-tokens))

;; ---------------------------------------------------------------------------
;; The global conditional
;; ---------------------------------------------------------------------------

(def ^:private completion?
  "gen.llm.token/completion?, resolved when first called: gen.llm.token
   requires this namespace."
  (delay (requiring-resolve 'gen.llm.token/completion?)))

(defn conditional-weight
  "The importance weight of tr, a completion's trace (gen.llm.token), for
   the LM's completion weighed by its potential, π(x) ∝ p(x) exp(w(x)):
   log p(x) + w(x) - log q(x), with p(x) the unconstrained completion's
   density of tr's tokens (the LM's scores, its end token's included), w(x)
   the potential's weight of the tokens before an end token as a whole
   sequence (all of them when the completion stopped at max-tokens), and
   log q(x) tr's score. Its mean over the completion's traces is Z = Σ p(x)
   exp(w(x)). -Inf for a trace of score -Inf; 0 for a completion with no
   potential (q is p). Any other trace is an error, a model's that calls a
   completion too: its completion's trace is (gen.trace/subtrace tr addr)."
  ^double [tr]
  (let [{:keys [lm potential] :as gf} (:gen-fn tr)
        _ (when-not (@completion? gf)
            (throw (ex-info (str "conditional-weight: a completion's trace (gen.llm.token), not a trace of "
                                 (pr-str gf) " (a completion a model calls: its trace is gen.trace/subtrace's)")
                            {:reason :invalid-trace :gen-fn gf})))
        [prompt] (:args tr)
        tokens (:retval tr)
        score (double (:score tr))]
    (cond (nil? potential) 0.0
          (== Double/NEGATIVE_INFINITY score) Double/NEGATIVE_INFINITY
          :else (let [text (if (and (seq tokens) (contains? (:end-tokens gf) (peek tokens))) (pop tokens) tokens)
                      log-p (double (if (seq tokens) (reduce + 0.0 (lm/score lm prompt tokens)) 0.0))]
                  (- (+ log-p (potential/complete potential text)) score)))))
