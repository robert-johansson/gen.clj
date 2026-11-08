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
   and max-tokens as gen.llm.token has them; its arguments checked as it
   checks them, token/check-args: tokens of the vocabulary, within the LM's
   :limit) and complete the potential's
   weight of x as a whole sequence (the tokens before an end token). The
   potential's prefix weights twist the intermediate targets: a step adds
   a token's next-token weight (an end token's is the potential's end
   entry), and a completion stopped at max-tokens adds complete - prefix,
   so a particle's log weight telescopes to log p(x)/q(x) + complete(x).
   exp(log-ml-estimate) estimates Z = Σ p(x)·exp(complete(x)), unbiasedly;
   with a 0/-Inf potential (a constraint), the probability that a
   completion satisfies it.

   A :critic is a second potential, applied at the drawn token only (Loula
   et al., ICLR 2025: cheap potentials in the proposal, expensive ones as
   critics): after a particle draws token t, its weight gains the critic's
   prefix(ctx ++ [t]) - prefix(ctx), or complete(ctx) - prefix(ctx) for an
   end token, complete less prefix at max-tokens: one critic call per
   particle and step (its prefix of the tokens so far is kept), where the
   potential's twist reads a whole next-token row. The target gains its
   weight: π(x) ∝ p(x) · exp(complete(x) + critic-complete(x)), and Z
   with it. A critic whose prefix weights are all 0 weighs at the end
   only. The proposal never reads it.

   The completion is one without a potential of its own: SMC draws and
   weighs by the LM's rows, so a constrained completion is refused
   (:constrained-completion). Its target, the LM's completions held to the
   potential, is SMC's with that potential as :potential, and the :masked
   proposal draws as the constrained completion's sites do. A :potential
   that declares end tokens (potential/end-tokens: a grammar's) must
   declare the completion's (:mismatched-end-tokens); one that rules every
   sequence out is no error, an estimate of Z = 0.

   opts at initialize: :potential (none: weight 1), :critic (none), :proposal,
   :awrs, :key, :host-pool.
   - :lm (the default), the bootstrap: tokens drawn from the LM's row.
   - :masked, the LM's row times the potential's next-token weights,
     renormalized (for a constraint, the row masked to its allowed tokens;
     gen.llm.constrained/weighted, as a constrained completion's sites):
     the locally optimal proposal of a one-step twist.
   - a function (fn [context row] → q), q a float array of the next
     token's log-probabilities (normalized), given the context (the prompt
     and the tokens so far) and the LM's row.
   - :awrs, adaptive weighted rejection sampling (gen.llm.awrs; Lipkin et
     al., COLM 2025), for a potential with no cheap row (a parser or a
     reader on partial text): the LM's row restricted to the tokens of
     finite twist, drawn by rejection, checking only the tokens it draws
     (a prefix call each, never a sweep of the vocabulary), with an
     unbiased estimate Ẑ of their mass. The weight is random: log Ẑ plus
     the token's twist, an unbiased estimate of the masked proposal's.
     :awrs, a map of gen.llm.awrs/sample's options (:method :recursive or
     :geometric, :max-rejects, :max-accepts), chooses the estimator. Its
     draws and checks run on the calling thread.
   A token's weight is log p - log q + its twist, with p the LM's score for
   it (its ledger's, as the completion's, so a particle's log p is its
   trace's score) and q the proposal's (:awrs's: the row's, less log Ẑ).

   One read a step: the live particles' contexts go to
   lm/batch-next-logprobs together, so with :max-branches at least the
   number of particles, a step is one advance. Particles hold tokens; the
   LM's trie owns every branch, and resampling copies token vectors.
   Draws are on the host: particle i's token at step t from (fold-in
   (fold-in (fold-in key t) 0) i), by token/dist. The particles' proposal rows (:lm's,
   :masked's) and draws, pure functions of their rows, twists and keys,
   run on a host pool, collected in order, so every result is the calling
   thread's alone, bit for bit, whatever the pool's size; the LM's and the
   potential's calls, and a proposal function's, stay on the calling
   thread. :host-pool is an ExecutorService (host-pool makes one), false
   for the calling thread alone, or by default a shared pool of daemon
   threads, of gen.llm.host's default size (the system property
   gen.llm.host-threads, or half the available processors up to 4; 1
   turns it off). Do not run SMC inside a task of the pool it is given: a
   step waits for its tasks, queued behind the one that waits, so the pool
   can deadlock. Resampling (maybe-resample, or run's) is when the
   effective sample size falls below :ess-threshold (n/2 by default), by
   :method (:systematic by default; any of gen.inference.weights'), and
   moves the weights' mean into :log-ml."
  {:api :experimental}
  (:require [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.llm.awrs :as awrs]
            [gen.llm.constrained :as constrained]
            [gen.llm.host :as host]
            [gen.inference.weights :as w]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.token :as token]
            [gen.random :as rng])
  (:import [java.util.concurrent ExecutorService]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; A state: {:completion :args [prompt max-tokens] :potential :critic
;; :proposal :key :host-pool (an ExecutorService or false) :t (steps so
;; far) :particles [{:tokens :done? :log-p :critic}] :log-weights :log-ml
;; :resampled?}. :log-p is a particle's log p(tokens), its trace's score;
;; :critic, with a critic, the critic's prefix weight of its tokens; a
;; particle whose weight fell to 0 stops, :dead?. Step t draws
;; with (fold-in (fold-in key t) 0) and resamples with (fold-in (fold-in
;; key t) 1).

(defn- step-key [state ^long purpose]
  (rng/fold-in (rng/fold-in (:key state) (:t state)) purpose))

(defn- check-completion
  "completion, checked: a completion (gen.llm.token) without a potential
   of its own, which SMC would neither draw nor weigh by."
  [completion]
  (when-not (token/completion? completion)
    (throw (ex-info (str "smc: a completion (gen.llm.token/completion) is the model, not " (pr-str completion))
                    {:reason :invalid-model :model completion})))
  (when (:potential completion)
    (throw (ex-info (str "smc: the completion has a potential of its own, which SMC neither draws nor weighs by;"
                         " give it as SMC's :potential instead, with the :masked proposal to draw as the"
                         " constrained completion does")
                    {:reason :constrained-completion :potential (:potential completion)})))
  completion)

(defn host-pool
  "A fixed pool of n daemon threads for the particles' host work (their
   proposal rows and draws), to give initialize as :host-pool; close it
   when done, or let the JVM's exit end it. Never run SMC from one of its
   own tasks: a step waits on tasks queued behind that one (a deadlock)."
  ^ExecutorService [n]
  (host/pool "gen.llm.smc host-" n))

(def ^:private shared-pool
  "The default pool, of host/default-threads threads; none (the calling
   thread) when that is 1."
  (delay (when (< 1 (host/default-threads)) (host-pool (host/default-threads)))))

(defn- pool-of
  "initialize's :host-pool, checked: an ExecutorService, false (the
   calling thread), or by default the shared pool (false if it has one
   thread)."
  [opts]
  (if (contains? opts :host-pool)
    (let [pool (:host-pool opts)]
      (when-not (or (false? pool) (instance? ExecutorService pool))
        (throw (ex-info (str "smc: :host-pool is an ExecutorService or false, not " (pr-str pool))
                        {:reason :invalid-option :host-pool pool})))
      pool)
    (or @shared-pool false)))

(defn- check-proposal [proposal potential]
  (cond
    (= :lm proposal) proposal
    (#{:masked :awrs} proposal) (if potential
                                  proposal
                                  (throw (ex-info (str "smc: the " proposal " proposal needs a :potential")
                                                  {:reason :invalid-option :proposal proposal})))
    (fn? proposal) proposal
    :else (throw (ex-info (str "smc: the proposal is :lm, :masked, :awrs or a function, not " (pr-str proposal))
                          {:reason :invalid-option :proposal proposal}))))

(defn initialize
  "n particles of the completion on [prompt max-tokens], none drawn yet,
   their log weights the potential's (and the critic's) prefix weight of
   no tokens. opts: :potential, :critic, :proposal (:lm, :masked or a
   function), :key and :host-pool (see the namespace's docstring; never a
   pool whose own task runs this SMC)."
  ([completion args n] (initialize completion args n {}))
  ([completion args n {:keys [potential critic proposal key awrs] :or {proposal :lm} :as opts}]
   (let [args (token/check-args :smc (check-completion completion) args)]
     ;; only the end tokens: a potential that rules every sequence out is
     ;; an estimate of Z = 0, not an error
     (when potential (potential/check-end-tokens "smc" potential (:end-tokens completion)))
     (when critic
       (when-not (potential/potential? critic)
         (throw (ex-info (str "smc: the :critic is a potential (gen.llm.potential), not " (pr-str critic))
                         {:reason :invalid-option :critic critic})))
       (potential/check-end-tokens "smc's :critic" critic (:end-tokens completion)))
     (when-not (pos-int? n)
       (throw (ex-info (str "smc: the number of particles is a positive integer, not " (pr-str n))
                       {:reason :invalid-args :n n})))
     (let [done? (zero? (long (second args)))
           ;; the empty context's weights, as a prefix (or as the whole
           ;; sequence, when max-tokens is 0)
           weight-of (fn [p] (cond (nil? p) 0.0
                                   done? (potential/complete p [])
                                   :else (potential/prefix p [])))
           c0 (when critic (potential/prefix critic []))]
       {:completion completion
        :args args
        :potential potential
        :critic critic
        :proposal (check-proposal proposal potential)
        :awrs (when (contains? opts :awrs)
                (when-not (= :awrs proposal)
                  (throw (ex-info "smc: the :awrs options are the :awrs proposal's"
                                  {:reason :invalid-option :proposal proposal})))
                (awrs/check-options awrs))
        :key (or key (gfi/next-key))
        :host-pool (pool-of opts)
        :t 0
        :particles (vec (repeat n (cond-> {:tokens [] :done? done? :log-p 0.0} critic (assoc :critic c0))))
        :log-weights (vec (repeat n (+ (double (weight-of potential)) (double (weight-of critic)))))
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

(defn- function-row
  "A proposal function's row for a particle at context, checked."
  ^floats [proposal context ^floats row]
  (let [q (proposal context row)]
    (when-not (and (instance? float/1 q) (== (alength ^floats q) (alength row)))
      (throw (ex-info "smc: a proposal function returns a float array of the vocabulary's log-probabilities"
                      {:reason :invalid-row :value q})))
    q))

(defn- proposed
  "A particle's proposal row and its token drawn with key: [q t], or nil
   if the potential allows no next token (under :masked). q is the
   function's row f-row when the proposal is a function, else :lm's (the
   LM's row) or :masked's, made here: a pure function of the rows and the
   key, so it may run on the host pool."
  [state ^floats row ws f-row key]
  (when-let [q (case (:proposal state)
                 :lm row
                 :masked (first (constrained/weighted (:end-tokens (:completion state)) row ws))
                 f-row)]
    [q (long (dc/sample (token/dist q) key))]))

(defn- row-twist
  "Token t's twist, from the potential's next-token weights ws (0 without
   a potential): its entry (the end's for an end token), and for a
   particle stopped at max-tokens the whole sequence's weight in place of
   the prefix's (a token of weight 0 stays 0)."
  ^double [state particle ws ^long t]
  (let [{:keys [completion potential]} state
        tokens (conj (:tokens particle) t)
        end? (contains? (:end-tokens completion) t)
        capped? (and (not end?) (== (count tokens) (long (second (:args state)))))
        twist (if ws (double (aget ^floats ws (int (end-entry completion t)))) 0.0)]
    (if (and potential capped? (not (Double/isInfinite twist)))
      (+ twist (- (potential/complete potential tokens) (potential/prefix potential tokens)))
      twist)))

(defn- weigher
  "The twist of each next token of particle, from the potential's prefix
   and complete weights (no row): prefix(tokens ++ [t]) less its prefix
   of the tokens so far, complete for an end token (once for all of them)
   and at max-tokens. :awrs's condition: a token of twist -Inf is
   rejected."
  [state particle]
  (let [{:keys [completion potential]} state
        tokens (:tokens particle)
        base (potential/prefix potential tokens)
        ended (delay (- (potential/complete potential tokens) base))
        max-tokens (long (second (:args state)))]
    (fn [t]
      (let [t (long t)]
        (cond (contains? (:end-tokens completion) t) @ended
              (== (inc (count tokens)) max-tokens) (- (potential/complete potential (conj tokens t)) base)
              :else (- (potential/prefix potential (conj tokens t)) base))))))

(defn- moved
  "particle (live, at context) moved by a draw {:t the token, :log-q its
   proposal's log-probability, :twist}, and its log weight's increment:
   [the particle, the increment]; a particle of weight 0 for no draw (nil:
   the proposal had no token to draw)."
  [state particle context draw]
  (if (nil? draw)
    [(assoc particle :done? true :dead? true) Double/NEGATIVE_INFINITY]
    (let [{:keys [completion]} state
          t (long (:t draw))
          log-p (double (first (lm/score (:lm completion) context [t])))
          tokens (conj (:tokens particle) t)
          end? (contains? (:end-tokens completion) t)
          capped? (and (not end?) (== (count tokens) (long (second (:args state)))))
          dw (+ (- log-p (double (:log-q draw))) (double (:twist draw)))
          ;; the critic, where the particle still has weight: its weight of
          ;; the tokens now (as the whole sequence at an end or at
          ;; max-tokens), less its prefix weight of those before
          critic-w (when (and (:critic state) (not (Double/isInfinite dw)))
                     (cond end? (potential/complete (:critic state) (:tokens particle))
                           capped? (potential/complete (:critic state) tokens)
                           :else (potential/prefix (:critic state) tokens)))
          dw (if critic-w (+ dw (- (double critic-w) (double (:critic particle)))) dw)
          dead? (Double/isInfinite dw)]
      [(cond-> {:tokens tokens :done? (or end? capped? dead?) :log-p (+ (double (:log-p particle)) log-p)}
         critic-w (assoc :critic critic-w)
         dead? (assoc :dead? true))
       dw])))

(defn- awrs-draw
  "particle's draw by AWRS from the LM's row with key: the token, its
   log-probability under the proposal (the row's less log Ẑ) and its
   twist; nil where Ẑ is 0 (no token of finite twist, or a budget spent)."
  [state particle ^floats row key]
  (let [{:keys [token weight log-z]} (awrs/sample row (weigher state particle) key (or (:awrs state) {}))]
    (when-not (== Double/NEGATIVE_INFINITY (double log-z))
      {:t token :log-q (- (double (aget row (int token))) (double log-z)) :twist weight})))

(defn step
  "Every live particle one token further: their contexts read in one
   lm/batch-next-logprobs, a token drawn from each one's proposal, and each
   one's log weight increased by log p - log q plus the twist. The
   proposal rows (:lm's, :masked's) and the draws run on the state's host
   pool; the LM's and the potential's calls, a proposal function's and
   :awrs's draws, on the calling thread. Finished particles are kept as
   they are. opts: :key (by default the state's, folded with the step)."
  ([state] (step state {}))
  ([state {:keys [key]}]
   (let [{:keys [completion potential proposal particles t]} state
         [prompt] (:args state)
         live (into [] (remove #(:done? (particles %))) (range (count particles)))
         contexts (mapv #(into prompt (:tokens (particles %))) live)
         ;; the LM's rows, lent (read only, never written: token/dist,
         ;; the :masked and :awrs proposals) unless a proposal function,
         ;; the caller's, receives them
         rows (when (seq live)
                (if (fn? proposal)
                  (lm/batch-next-logprobs (:lm completion) contexts)
                  (lm/batch-next-logprobs-lent (:lm completion) contexts)))
         draws (or key (step-key state 0))
         drawn (if (= :awrs proposal)
                 (mapv (fn [i row] (awrs-draw state (particles i) row (rng/fold-in draws i))) live rows)
                 (let [twists (mapv #(when potential (twist-row state (:tokens (particles %)))) live)
                       f-rows (if (fn? proposal) (mapv #(function-row proposal %1 %2) contexts rows) (repeat nil))
                       proposals (host/in-order (:host-pool state)
                                                (fn [[i row ws f-row]] (proposed state row ws f-row (rng/fold-in draws i)))
                                                (map vector live rows twists f-rows))]
                   (mapv (fn [i ws [q t]]
                           (when q
                             {:t t :log-q (aget ^floats q (int t)) :twist (row-twist state (particles i) ws t)}))
                         live twists proposals)))
         moves (mapv #(moved state (particles %1) %2 %3) live contexts drawn)]
     (assoc (reduce (fn [s [i [p dw]]]
                      (-> s
                          (assoc-in [:particles i] p)
                          (update-in [:log-weights i] #(+ (double %) (double dw)))))
                    state (map vector live moves))
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
