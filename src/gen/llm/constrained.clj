(ns gen.llm.constrained
  "Completions under a potential (gen.llm.token's :potential, gen.llm.potential):
   each token's distribution is the LM's times the potential's next-token
   weights, renormalized at its site. Under a grammar (gen.llm.grammar), the
   LM's row masked to the tokens the grammar allows, renormalized: GenMLX's
   grammar-masked decoder.

     (weighted end-tokens row ws)   → [q log-z]: the site's distribution
     (conditional-weight trace)     → log p(x) + w(x) - log q(x), the trace's
                                      importance weight for the global conditional

   The distribution is local: q(x) = Π_i p(x_i | x_<i) exp(w_i(x_i)) / Z_i,
   w_i the potential's next-token weights and Z_i the normalizer at site i,
   which depends on the tokens before it. That is not the LM conditioned on
   the potential, π(x) ∝ p(x) exp(w(x)) (w the potential's weight of x as a
   whole sequence), unless Π_i Z_i is the same for every x: renormalizing
   token by token favors sequences whose early tokens leave little of the
   LM's mass disallowed. A constrained
   completion's GFI weights are q's: its score is log q(x), assess and
   generate weigh by q, and its laws hold for q. For π, the completion is a
   proposal: conditional-weight is log p(x) + w(x) - log q(x), with p the
   unconstrained completion's density (the LM's ledger, as its score), so
   its mean over q is Z = Σ p(x) exp(w(x)), and for a constraint the
   probability that the LM's completion satisfies it; self-normalized,
   these weights give expectations under π (importance sampling, or token
   SMC's :masked proposal, gen.llm.smc, whose steps weigh the same).

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
  (:require [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; The site's distribution
;; ---------------------------------------------------------------------------

(defn- max-weight
  "The largest of the site's log weights: the LM's log-probability plus the
   potential's weight, an end token's (ends, sorted) the end's. One pass,
   the next end token tracked by its index."
  ^double [^floats row ^floats ws ^ints ends]
  (let [n (alength row)
        k (alength ends)
        end-weight (double (aget ws n))]
    (loop [t 0 j 0 m Double/NEGATIVE_INFINITY]
      (if (< t n)
        (if (and (< j k) (== t (aget ends j)))
          (recur (inc t) (inc j) (Math/max m (+ (double (aget row t)) end-weight)))
          (recur (inc t) j (Math/max m (+ (double (aget row t)) (double (aget ws t))))))
        m))))

(defn- sum-exp
  "Σ exp(x - m) over the site's log weights x (as max-weight's), skipping
   those of weight 0."
  ^double [^floats row ^floats ws ^ints ends ^double m]
  (let [n (alength row)
        k (alength ends)
        end-weight (double (aget ws n))]
    (loop [t 0 j 0 s 0.0]
      (if (< t n)
        (let [end? (and (< j k) (== t (aget ends j)))
              x (+ (double (aget row t)) (if end? end-weight (double (aget ws t))))
              s (if (Double/isInfinite x) s (+ s (Math/exp (- x m))))]
          (recur (inc t) (if end? (inc j) j) s))
        s))))

(defn weighted
  "The LM's row (log-probabilities by token, a float array) times a
   potential's next-token weights ws (n-vocab + 1 entries, the end last;
   an end token, of end-tokens, takes the end's), renormalized: [q log-z],
   q a new float array of the site's log-probabilities and log-z the log
   of the normalizer, Σ_t p(t) exp(w(t)). [nil -Inf] when the weights allow
   no token. Three passes over the row, allocating only q."
  [end-tokens ^floats row ^floats ws]
  (let [n (alength row)]
    (when-not (== (alength ws) (inc n))
      (throw (ex-info (str "constrained: the potential's next-token weights have " (alength ws) " entries, not "
                           (inc n) " (the LM's vocabulary and the end)")
                      {:reason :invalid-weights :entries (alength ws) :n-vocab n})))
    (let [ends (int-array (sort end-tokens))
          k (alength ends)
          end-weight (double (aget ws n))
          m (max-weight row ws ends)]
      (if (Double/isInfinite m)
        [nil Double/NEGATIVE_INFINITY]
        (let [log-z (+ m (Math/log (sum-exp row ws ends m)))
              q (float-array n)]
          (loop [t 0 j 0]
            (when (< t n)
              (let [end? (and (< j k) (== t (aget ends j)))]
                (aset q t (unchecked-float (- (+ (double (aget row t)) (if end? end-weight (double (aget ws t)))) log-z)))
                (recur (inc t) (if end? (inc j) j)))))
          [q log-z])))))

;; ---------------------------------------------------------------------------
;; A completion's sites (gen.llm.token calls these for a completion with a
;; potential; gf a completion, state a run's {:ctx :tokens :scores})
;; ---------------------------------------------------------------------------

(defn- end-tokens
  "The completion's end tokens."
  [gf]
  (:end-tokens gf))

(defn- dead?
  "Has a token of the run so far weight 0, so that no sequence it begins
   has weight?"
  [{:keys [scores]}]
  (boolean (some #(Double/isInfinite (double %)) scores)))

(defn draw-next
  "The run's next token, drawn by draw (a function of a row of
   log-probabilities) from its site's distribution, and its score: [t
   [score]]. After a token of weight 0, from the LM's row, scored by it.
   Loud where the context has weight but the potential allows no token."
  [{:keys [lm potential] :as gf} {:keys [ctx tokens] :as state} draw]
  (let [row (lm/next-logprobs lm ctx)]
    (if (dead? state)
      (let [t (long (draw row))] [t [(double (aget row t))]])
      (let [[^floats q] (weighted (end-tokens gf) row (potential/logw-next potential tokens))]
        (when-not q
          (throw (ex-info (str "constrained: the potential allows no token at position " (count tokens)
                               " (a grammar complete there, with no end tokens to end it?)")
                          {:reason :zero-weight :position (count tokens)})))
        (let [t (long (draw q))]
          [t [(double (aget q t))]])))))

(defn run-scores
  "The log-probabilities of run, tokens known to follow the run so far,
   each at its site: the LM's rows at their contexts read in one call, the
   potential's weights at each. -Inf for a token of weight 0, the LM's
   row's entry for each token after one."
  [{:keys [lm potential] :as gf} {:keys [ctx tokens] :as state} run]
  (let [n (count run)
        rows (lm/batch-next-logprobs lm (mapv #(into ctx (subvec run 0 %)) (range n)))]
    (loop [i 0 out [] dead (dead? state)]
      (if (== i n)
        out
        (let [t (int (run i))
              ^floats row (rows i)]
          (if dead
            (recur (inc i) (conj out (double (aget row t))) true)
            (let [[^floats q] (weighted (end-tokens gf) row (potential/logw-next potential (into tokens (subvec run 0 i))))
                  s (if q (double (aget q t)) Double/NEGATIVE_INFINITY)]
              (recur (inc i) (conj out s) (Double/isInfinite s)))))))))

(defn check-potential
  "potential, checked for a completion over n-vocab tokens: a potential that
   gives the empty context weight."
  [potential ^long n-vocab]
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
  potential)

;; ---------------------------------------------------------------------------
;; The global conditional
;; ---------------------------------------------------------------------------

(defn conditional-weight
  "The importance weight of tr, a trace of a completion under a potential,
   for the LM's completion weighed by the potential, π(x) ∝ p(x) exp(w(x)):
   log p(x) + w(x) - log q(x), with p(x) the unconstrained completion's
   density of tr's tokens (the LM's scores, its end token's included), w(x)
   the potential's weight of the tokens before an end token as a whole
   sequence (all of them when the completion stopped at max-tokens), and
   log q(x) tr's score. Its mean over the completion's traces is Z = Σ p(x)
   exp(w(x)). -Inf for a trace of score -Inf; 0 for a completion with no
   potential (q is p)."
  ^double [tr]
  (let [{:keys [lm potential] :as gf} (:gen-fn tr)
        [prompt] (:args tr)
        tokens (:retval tr)
        score (double (:score tr))]
    (cond (nil? potential) 0.0
          (Double/isInfinite score) Double/NEGATIVE_INFINITY
          :else (let [text (if (and (seq tokens) (contains? (end-tokens gf) (peek tokens))) (pop tokens) tokens)
                      log-p (double (if (seq tokens) (reduce + 0.0 (lm/score lm prompt tokens)) 0.0))]
                  (- (+ log-p (potential/complete potential text)) score)))))
