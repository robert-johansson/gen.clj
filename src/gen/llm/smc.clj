(ns gen.llm.smc
  "Token-level sequential Monte Carlo over a completion (gen.llm.token),
   twisted by a potential (gen.llm.potential), as GenMLX's token SMC and
   genlm-control's SMC: particles are token sequences that grow by one
   token a step, reweighted and resampled. The interface is
   gen.inference.particle-filter's:

     (initialize completion [prompt max-tokens] n opts)  → state
     (step state), (step state {:key k})                 → state, each live particle one token further
     (maybe-resample state opts)                         → state
     (run completion [prompt max-tokens] n opts)         → the final state
     (log-ml-estimate state)                             → log Ẑ
     (particles state), (traces state)                   → token vectors, the completion's traces

   The target is the completion's distribution weighed by the potential:
   π(x) ∝ p(x) · exp(complete(x)), p the completion's density (end tokens
   and max-tokens as gen.llm.token has them) and complete the potential's
   weight of x as a whole sequence (the tokens before an end token). The
   potential's prefix weights twist the intermediate targets: a step adds
   a token's next-token weight (an end token's is the potential's end
   entry), and a completion stopped at max-tokens adds complete - prefix,
   so a particle's log weight telescopes to log p(x)/q(x) + complete(x).
   exp(log-ml-estimate) estimates Z = Σ p(x)·exp(complete(x)), unbiasedly;
   with a 0/-Inf potential (a constraint), the probability that a
   completion satisfies it.

   opts at initialize: :potential (none: weight 1), :proposal, :key.
   - :lm (the default), the bootstrap: tokens drawn from the LM's row.
   - :masked, the LM's row times the potential's next-token weights,
     renormalized (for a constraint, the row masked to its allowed tokens):
     the locally optimal proposal of a one-step twist.
   - a function (fn [context row] → q), q a float array of the next
     token's log-probabilities (normalized), given the context (the prompt
     and the tokens so far) and the LM's row.
   A token's weight is log p - log q + its twist, with p the LM's score for
   it (its ledger's, as the completion's, so a particle's log p is its
   trace's score) and q the proposal's.

   One read a step: the live particles' contexts go to
   lm/batch-next-logprobs together, so with :max-branches at least the
   number of particles, a step is one advance. Particles hold tokens; the
   LM's trie owns every branch, and resampling copies token vectors.
   Draws are on the host: particle i's token at step t from (fold-in
   (fold-in key t) i), by token/dist. Resampling (maybe-resample, or run's)
   is when the effective sample size falls below :ess-threshold (n/2 by
   default), by :method (:systematic by default; any of
   gen.inference.weights'), and moves the weights' mean into :log-ml."
  (:require [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.inference.weights :as w]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; A state: {:completion :args [prompt max-tokens] :potential :proposal
;; :key :t (steps so far) :particles [{:tokens :done? :log-p}] :log-weights
;; :log-ml :resampled?}. :log-p is a particle's log p(tokens), its trace's
;; score; a particle whose weight fell to 0 stops, :dead?. Step t draws
;; with (fold-in (fold-in key t) 0) and resamples with (fold-in (fold-in
;; key t) 1).

(defn- step-key [state ^long purpose]
  (rng/fold-in (rng/fold-in (:key state) (:t state)) purpose))

(defn- check-args [completion [prompt max-tokens :as args]]
  (when-not (instance? gen.llm.token.Completion completion)
    (throw (ex-info (str "smc: a completion (gen.llm.token/completion) is the model, not " (pr-str completion))
                    {:reason :invalid-model :model completion})))
  (when-not (and (= 2 (count args)) (sequential? prompt) (seq prompt) (nat-int? max-tokens))
    (throw (ex-info (str "smc: the arguments are [prompt max-tokens], a non-empty token vector and a"
                         " non-negative integer, not " (pr-str args))
                    {:reason (if (and (sequential? prompt) (empty? prompt)) :no-tokens :invalid-args) :args args})))
  [(vec prompt) (long max-tokens)])

(defn- check-proposal [proposal potential]
  (cond
    (= :lm proposal) proposal
    (= :masked proposal) (if potential
                           proposal
                           (throw (ex-info "smc: the :masked proposal needs a :potential"
                                           {:reason :invalid-option :proposal proposal})))
    (fn? proposal) proposal
    :else (throw (ex-info (str "smc: the proposal is :lm, :masked or a function, not " (pr-str proposal))
                          {:reason :invalid-option :proposal proposal}))))

(defn initialize
  "n particles of the completion on [prompt max-tokens], none drawn yet,
   their log weights the potential's prefix weight of no tokens. opts:
   :potential, :proposal (:lm, :masked or a function) and :key (see the
   namespace's docstring)."
  ([completion args n] (initialize completion args n {}))
  ([completion args n {:keys [potential proposal key] :or {proposal :lm}}]
   (let [args (check-args completion args)]
     (when-not (pos-int? n)
       (throw (ex-info (str "smc: the number of particles is a positive integer, not " (pr-str n))
                       {:reason :invalid-args :n n})))
     (let [w0 (if potential (potential/prefix potential []) 0.0)
           done? (zero? (long (second args)))]
       {:completion completion
        :args args
        :potential potential
        :proposal (check-proposal proposal potential)
        :key (or key (gfi/next-key))
        :t 0
        :particles (vec (repeat n {:tokens [] :done? done? :log-p 0.0}))
        :log-weights (vec (repeat n (if done? (if potential (potential/complete potential []) 0.0) w0)))
        :log-ml 0.0
        :resampled? false}))))

;; ---------------------------------------------------------------------------
;; One step
;; ---------------------------------------------------------------------------

(defn- end-entry
  "Where token t's twist is in the potential's next-token weights: the end
   for an end token."
  ^long [completion ^long t]
  (if (contains? (:end-tokens completion) t) (long (:n-vocab completion)) t))

(defn- twist-row
  "The potential's next-token weights after tokens, checked."
  ^floats [state tokens]
  (let [^floats ws (potential/logw-next (:potential state) tokens)
        n (inc (long (:n-vocab (:completion state))))]
    (when-not (== n (alength ws))
      (throw (ex-info (str "smc: the potential's next-token weights have " (alength ws) " entries, not " n
                           " (the vocabulary and the end)")
                      {:reason :invalid-potential :entries (alength ws) :expected n})))
    ws))

(defn- max-of ^double [^doubles x]
  (areduce x t m Double/NEGATIVE_INFINITY (Math/max m (aget x t))))

(defn- sum-exp ^double [^doubles x ^double m]
  (areduce x t s 0.0 (+ s (Math/exp (- (aget x t) m)))))

(defn- log-sum-exp
  "log Σ exp(x), -Inf when every entry is."
  ^double [^doubles x]
  (let [m (max-of x)]
    (if (Double/isInfinite m) m (+ m (Math/log (sum-exp x m))))))

(defn- masked
  "The LM's row times the twist, renormalized: the masked proposal, and its
   log normalizer (-Inf when the twist allows no token)."
  [completion ^floats row ^floats ws]
  (let [n (alength row)
        x (double-array n)]
    (dotimes [t n]
      (aset x t (+ (aget row t) (aget ws t))))
    ;; an end token's twist is the end's
    (doseq [t (:end-tokens completion)]
      (aset x (int t) (+ (aget row (int t)) (aget ws n))))
    (let [z (log-sum-exp x)]
      (if (Double/isInfinite z)
        [nil z]
        (let [q (float-array n)]
          (dotimes [t n] (aset q t (unchecked-float (- (aget x t) z))))
          [q z])))))

(defn- proposal-row
  "The proposal's row for a particle at context with the LM's row (nil if
   no token is possible under a masked proposal)."
  ^floats [state context ^floats row ws]
  (let [proposal (:proposal state)]
    (cond
      (= :lm proposal) row
      (= :masked proposal) (first (masked (:completion state) row ws))
      :else (let [q (proposal context row)]
              (when-not (and (instance? float/1 q) (== (alength ^floats q) (alength row)))
                (throw (ex-info "smc: a proposal function returns a float array of the vocabulary's log-probabilities"
                                {:reason :invalid-row :value q})))
              q))))

(defn- advance-particle
  "particle (live, at context, the LM's row there) one token further: [the
   particle, its log weight's increment]."
  [state particle context ^floats row key]
  (let [{:keys [completion potential]} state
        m (:lm completion)
        ws (when potential (twist-row state (:tokens particle)))
        q (proposal-row state context row ws)]
    (if (nil? q)
      ;; the potential allows no next token: a particle of weight 0
      [(assoc particle :done? true :dead? true) Double/NEGATIVE_INFINITY]
      (let [t (long (dc/sample (token/dist q) key))
            log-p (double (first (lm/score m context [t])))
            tokens (conj (:tokens particle) t)
            end? (contains? (:end-tokens completion) t)
            capped? (and (not end?) (== (count tokens) (long (second (:args state)))))
            twist (if ws (double (aget ^floats ws (int (end-entry completion t)))) 0.0)
            ;; stopped at max-tokens: the whole sequence's weight replaces the
            ;; prefix's (a particle of weight 0 stays 0)
            twist (if (and potential capped? (not (Double/isInfinite twist)))
                    (+ twist (- (potential/complete potential tokens) (potential/prefix potential tokens)))
                    twist)
            dw (+ (- log-p (double (aget q (int t)))) twist)
            dead? (Double/isInfinite dw)]
        [(cond-> {:tokens tokens :done? (or end? capped? dead?) :log-p (+ (double (:log-p particle)) log-p)}
           dead? (assoc :dead? true))
         dw]))))

(defn step
  "Every live particle one token further: their contexts read in one
   lm/batch-next-logprobs, a token drawn from each one's proposal, and each
   one's log weight increased by log p - log q plus the twist. Finished
   particles are kept as they are. opts: :key (by default the state's,
   folded with the step)."
  ([state] (step state {}))
  ([state {:keys [key]}]
   (let [{:keys [completion particles t]} state
         [prompt] (:args state)
         live (filterv #(not (:done? (particles %))) (range (count particles)))
         contexts (mapv #(into prompt (:tokens (particles %))) live)
         rows (when (seq live) (lm/batch-next-logprobs (:lm completion) contexts))
         draws (or key (step-key state 0))
         moved (mapv (fn [i context row] (advance-particle state (particles i) context row (rng/fold-in draws i)))
                     live contexts rows)]
     (assoc (reduce (fn [s [i [p dw]]]
                      (-> s
                          (assoc-in [:particles i] p)
                          (update-in [:log-weights i] #(+ (double %) (double dw)))))
                    state (map vector live moved))
            :t (inc (long t))
            :resampled? false))))

;; ---------------------------------------------------------------------------
;; Resampling, the estimate, the results
;; ---------------------------------------------------------------------------

(defn- dead-all?
  "Has every particle fallen to weight 0?"
  [state]
  (every? #(== Double/NEGATIVE_INFINITY (double %)) (:log-weights state)))

(defn ess
  "The effective sample size of the particles' weights (0 when every one
   has weight 0)."
  ^double [state]
  (if (dead-all? state) 0.0 (w/ess (:log-weights state))))

(defn resample
  "The particles drawn in proportion to their weights (:method, default
   :systematic; :key); the weights become equal and their log mean moves
   into :log-ml. Resampling copies token vectors."
  ([state] (resample state {}))
  ([state {:keys [method key] :or {method :systematic}}]
   (when (dead-all? state)
     (throw (ex-info "smc/resample: every particle has weight 0 (the potential allows none of their sequences)"
                     {:reason :impossible :n (count (:log-weights state))})))
   (let [lw (:log-weights state)
         n (count lw)
         parents (w/resample method lw n (or key (step-key state 1)))]
     (assoc state
            :particles (mapv (:particles state) parents)
            :log-weights (vec (repeat n 0.0))
            :log-ml (+ (double (:log-ml state)) (w/log-mean-exp lw))
            :resampled? true))))

(defn maybe-resample
  "Resample when the effective sample size is below :ess-threshold (n/2 by
   default); :resampled? says whether it did."
  ([state] (maybe-resample state {}))
  ([state {:keys [ess-threshold] :as opts}]
   (let [n (count (:log-weights state))]
     (if (< (ess state) (double (or ess-threshold (/ n 2.0))))
       (resample state opts)
       (assoc state :resampled? false)))))

(defn done?
  "Has every particle finished (an end token, or max-tokens)?"
  [state]
  (every? :done? (:particles state)))

(defn run
  "initialize, then step (and maybe-resample, while particles go on) until
   every particle has finished: the final state. opts: initialize's and
   maybe-resample's (:ess-threshold, :method)."
  ([completion args n] (run completion args n {}))
  ([completion args n opts]
   (loop [state (initialize completion args n opts)]
     (if (done? state)
       state
       (let [state (step state)]
         (recur (if (done? state) state (maybe-resample state (dissoc opts :key)))))))))

(defn log-ml-estimate
  "The estimate of log Z, unbiased in exp space: the log marginal
   likelihood of the potential under the completion (0 without one; -Inf
   when every particle has fallen to weight 0, an estimate of Z = 0)."
  ^double [state]
  (if (dead-all? state)
    Double/NEGATIVE_INFINITY
    (+ (double (:log-ml state)) (w/log-mean-exp (:log-weights state)))))

(defn particles
  "The particles' token vectors, in order."
  [state]
  (mapv :tokens (:particles state)))

(defn traces
  "The particles as traces of the completion, in order (by generate with
   every token forced, which reads nothing the run did not); their scores
   are the particles' log p. A dead particle (weight 0) has no trace: nil.
   Every particle must have finished."
  [state]
  (let [{:keys [completion args]} state]
    (mapv (fn [{:keys [tokens done? dead?]}]
            (when-not done?
              (throw (ex-info "smc/traces: a particle has not finished: step the state until done?"
                              {:reason :not-done :tokens tokens})))
            (when-not dead?
              (:trace (gfi/generate completion args (token/forced tokens) (:key state)))))
          (:particles state))))
