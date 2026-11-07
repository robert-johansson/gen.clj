(ns gen.llm.grpo
  "Group relative policy optimization (GRPO) of a LoRA adapter
   (gen.llm.lora) with verifiable rewards, and its variants DAPO, Dr. GRPO
   and bnpo, each as its paper defines its loss; over any model family of
   gen.llm.models, its rollouts drawn by gen.llm.array-engine.

     (rollouts lm prompts opts)          → K completions of each prompt, each
                                           token's old log-probability recorded
     (exact-match opts), (format-match re) → rewards: (fn [prompt completion])
     (rewards reward rollouts)           → [[r …] …] by group, every one finite
     (advantages rewards opts)           → [[A …] …], (r - mean)/(std + eps)
     (batches rollouts advantages opts)  → the step's padded, masked batches
     (loss m batch opts)                 → [the summed loss, the log-probs]
     (trainer adapter optimizer key)     → a trainer, a value (sft's, and its round)
     (round m tokenizer trainer prompts reward opts) → [trainer' info]
     (train m tokenizer trainer prompts reward opts) → {:trainer :history}
     (evaluate lm prompts reward opts)   → {:mean :per-prompt :rewards}

   Checkpoints are gen.llm.sft's (save-checkpoint, load-checkpoint,
   latest-checkpoint): a GRPO trainer is an SFT trainer with two more
   keys, so a checkpoint holds the round too.

   A round (Shao et al. 2024, \"DeepSeekMath\", §4.1, Algorithm 1): the
   policy π_θold, the base model under the trainer's adapter, draws K
   completions o_1 … o_K of each of P prompts q (rollouts); a reward
   function scores each (rewards), and each group's rewards give the
   advantages (advantages); then :iterations (μ) optimizer steps on those
   rollouts, the old log-probabilities fixed, each on the loss below over
   all P·K completions, accumulated over microbatches of :batch-size
   rows by gen.llm.sft/step (its :loss), whose machinery, checkpoints and
   loud non-finite steps it is. The next round draws new rollouts from
   the policy so trained.

   Rollouts. Each completion is drawn by gen.llm.sample/complete through
   an LM over the array engine with the adapted model (dropout off), every
   token from its own key: completion j of prompt i of round s from
   (rng/split (fold-in k-samples s) (P·K))[i·K + j], its step t from
   fold-in t. A prompt is a map: :prompt, a user message's text or chat
   messages, rendered by Qwen's chat template with a generation prompt
   (gen.llm.chat/render, :chat its options), BOS first where the tokenizer
   adds one; its other keys are the reward's. A completion is its drawn
   tokens and, when it stopped at one, its end token: the policy chose to
   stop, so the end token is trained (as TRL's completion mask keeps the
   first EOS). Each token's old log-probability, log π_θold(o_t | q, o<t),
   is recorded as it is drawn: the engine's row at the sampling
   temperature T, before top-k and top-p (sample/complete's
   :token-logprobs?). The papers' π_θold is the policy, and the ratio's
   numerator is the policy's too, at T (the family's log-probs'
   :temperature), so before the adapter moves every ratio is 1 but for
   float round-off between the engine's decode and the training forward:
   the check that each token's log-probability is read where it was drawn.
   The default sampling, {:temp 1 :top-k nil :top-p 1}, is the policy
   itself (the papers' rollouts are at temperature 1). With a top-k or
   top-p the completions are drawn from a truncated distribution while
   the ratio's denominator stays the policy's, as TRL and verl compute it:
   the step then corrects nothing for the truncation.

   Batches. Row i·K + j is prompt i's tokens, then completion j's, then
   padding (:pad) to the microbatch's longest row: positions start at 0
   in every row, as when it was drawn, and causal attention never lets a
   real token read the padding after it; nothing pads inside a row. The
   loss reads the family's log-probabilities of every row at positions 1 …
   T-1 (log p of token t + 1 given those before it), masked to the
   completion's tokens (:mask, 1 there; the prompt and padding 0): a
   completion of C tokens after a prompt of n has its targets at n - 1 …
   n + C - 2.

   Rewards. A reward is a pure function (fn [prompt completion]) of the
   prompt (its map) and the completion's text (its drawn tokens decoded,
   without the end token), returning a finite number, run on the host one
   completion after another. A reward that is not a finite number is loud
   (:non-finite-reward): never a silent skip. exact-match and
   format-match are the built-in checkers.

   Advantages. A group's rewards r_1 … r_K give each of its completions
   Â_i = (r_i - mean(r)) / (std(r) + eps), std the population's (ddof 0;
   TRL's is Bessel's) and eps 1e-4 (TRL's, mlx-node's), every token of o_i
   the same; with :scale-rewards? false (Dr. GRPO's), r_i - mean(r). A
   group whose rewards are all equal has advantages exactly 0: it carries
   no gradient but the KL's, and round's info counts it (:flat-groups),
   since a round of flat groups learns nothing (\"RLVR sharpens, it does
   not teach\").

   The loss. With the ratio ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) /
   π_θold(o_{i,t} | q, o_{i,<t}), per token

     ℓ_{i,t} = -min(ρ_{i,t}·Â_i, clip(ρ_{i,t}, 1 - ε_low, 1 + ε_high)·Â_i)
               + β·D_{i,t},
     D_{i,t} = π_ref/π_θ - log(π_ref/π_θ) - 1          (DeepSeekMath, Eq. 4: k3)

   of the completion's tokens, the reference π_ref the base model (the
   adapter at scale 0: no copy of the weights), and the step's loss the
   ℓ aggregated by :loss-type, each the negated objective of its paper,
   over the N = P·K completions of a round, |o_i| completion i's tokens:

     :grpo     (1/N) Σ_i (1/|o_i|) Σ_t ℓ_{i,t}: the mean over completions of
               each one's mean (Shao et al. 2024, \"DeepSeekMath\", Eq. 3);
               defaults ε 0.2, β 0.04 (its §4.2)
     :dapo     (1/Σ_i |o_i|) Σ_i Σ_t ℓ_{i,t}: every token of the step alike
               (Yu et al. 2025, \"DAPO\", Eq. 8 and 12, the token-level loss);
               ε_low 0.2 and ε_high 0.28 (Clip-Higher, Eq. 10; its §4.1), β 0
               (DAPO drops the KL, its §2.3)
     :dr-grpo  (1/(N·L)) Σ_i Σ_t ℓ_{i,t}, L the generation budget :max-tokens,
               and Â_i = r_i - mean(r), no std (Liu et al. 2025, \"Understanding
               R1-Zero-Like Training\", Dr. GRPO, its Appendix A: (1/G) Σ_i Σ_t;
               its code divides each completion's sum by the constant
               MAX_TOKENS, \"e.g., generation budget\", which this follows: the
               objective times 1/L, a constant scale of the step); β 0
     :bnpo     each microbatch's (1/Σ_{i∈b} |o_i|) Σ_{i∈b} Σ_t ℓ_{i,t}, the
               step the mean of its microbatches': token-level, normalized
               in the local batch, TRL's loss_type \"bnpo\" (mlx-node's). The
               name has no paper behind this form: BNPO (Xiao et al. 2025,
               arXiv 2506.02864) is another method, an advantage normalized
               by a Beta density (its Eq. 11), with no token aggregation of
               its own. With one microbatch a step it is :dapo. β 0

   The step divides once, by the sum of the batches' :count (gen.llm.sft's
   accumulation): :grpo's count is a microbatch's completions, :dapo's its
   completion tokens, :dr-grpo's its completions times L, :bnpo's 1, each
   token weighted (:weights) by 1/|o_i|, 1, 1 and 1/(the microbatch's
   tokens): so accumulated microbatches give the whole step's loss but
   for rounding, except :bnpo's, which is by its definition a function of
   them. DAPO's dynamic sampling (Eq. 11: resampling groups all right or
   all wrong) and overlong shaping (Eq. 13) are not the loss's: a reward
   function can shape lengths, and round's info counts the flat groups.

   Keys. A trainer's :key is SFT's (each optimizer step's dropout, sft/step);
   its :rollout-key splits into [k-prompts k-samples]: round s takes the
   prompts (sft/order k-prompts n s P) (in epochs, each prompt once an
   epoch) and draws its rollouts from (fold-in k-samples s). A round is
   so a function of the trainer value, and a run resumed from a
   checkpoint (sft/load-checkpoint) is the run that saved it, bit for bit
   on the JVM backend (gen.llm.grpo-test).

   The log. round's info, a round's: {:round :step (the trainer's
   optimizer steps after it) :reward-mean :reward-std (over the round's
   completions) :groups :flat-groups :completions :tokens (completion
   tokens) :length-mean :truncated (completions cut at :max-tokens)
   :entropy (the mean of -log π_θold over the completion tokens: the
   sampled tokens' surprisal, an estimate of the policy's entropy a token
   at T, exact in expectation when sampling is the policy) :iterations,
   one map per optimizer step: {:step :loss :grad-norm :ratio-max (max
   |ρ - 1| over the tokens; at a round's first, float noise) :clip-fraction
   (the tokens whose clip is active: ρ below 1 - ε_low with Â < 0, or
   above 1 + ε_high with Â > 0) :kl (the mean of D over the tokens, when
   the reference is read)} :seconds {:rollouts :train}}.

   MLX's memory, as gen.llm.sft's docstring has it: set MLX's limit to
   what it holds between rounds (the weights, the adapter, the
   optimizer's state) plus about 1.25 times a round's own peak. Measured
   on fenrir's CUDA (2026-11-07; Qwen3-0.6B Q8_0, rank 8 on every
   projection of the last 8 blocks, 8 prompts × 8 completions of at most
   32 tokens, microbatches of 16, gen.llm.grpo-llama-test), at the held
   bytes' limit: 0.76 GB held, a round's peak 6.4-7.6 GB. The rollouts
   make it, not the step: each decode step of the array engine leaves
   its group's cache (64 rows) to the JVM's collector, which frees it
   late. At the limit so set, 9.3 GB, 40 rounds took 8 s each (5.5 s of
   rollouts, 2.6 s of two optimizer steps), peaking at 8.8 GB.

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [clojure.string :as str]
            [gen.array :as array :refer [- * < >]]
            [gen.inference.options :as options]
            [gen.llm.array-engine :as ae]
            [gen.llm.chat :as chat]
            [gen.llm.lm :as lm]
            [gen.llm.lora :as lora]
            [gen.llm.models :as models]
            [gen.llm.sample :as sample]
            [gen.llm.sft :as sft]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- invalid
  ([message] (invalid message {}))
  ([message data] (ex-info (str "grpo: " message) (merge {:reason :invalid-args} data))))

(def policy-sampling
  "Sampling from the policy itself: every token, at temperature 1
   (gen.llm.sample's :sampling), rollouts' default."
  {:temp 1.0 :top-k nil :top-p 1.0})

;; ---------------------------------------------------------------------------
;; Rollouts
;; ---------------------------------------------------------------------------

(defn- messages-of [i {:keys [prompt] :as p}]
  (cond
    (not (map? p)) (throw (invalid (str "prompt " i " is a map with :prompt, not " (pr-str p)) {:prompt i}))
    (string? prompt) [{:role :user :content prompt}]
    (and (sequential? prompt) (seq prompt) (every? map? prompt)) (vec prompt)
    :else (throw (invalid (str "prompt " i "'s :prompt is a user message's text or chat messages, not " (pr-str prompt))
                          {:prompt i}))))

(defn- context
  "The tokens of prompt i (a prompt map) by a tokenizer (gen.llm.lm's
   map: :encode, :add-bos?, :bos): its messages in Qwen's chat template
   with a generation prompt, BOS first where the tokenizer adds one."
  [{:keys [encode add-bos? bos]} i p chat-opts]
  (let [text (chat/render (messages-of i p) (assoc chat-opts :generation-prompt? true))]
    (into (if (and add-bos? bos) [bos] []) (encode text {:add-special? false :parse-special? true}))))

(defn- lm-tokenizer
  "lm's tokenizer, as context reads it."
  [lm]
  (assoc (lm/vocab lm) :encode (fn [text opts] (lm/encode lm text opts))))

(defn- check-sampling [sampling]
  (when-not (and (map? sampling) (number? (:temp sampling)) (pos? (double (:temp sampling))))
    (throw (invalid (str ":sampling is gen.llm.sample's {:temp :top-k :top-p}, :temp positive, not " (pr-str sampling))
                    {:sampling sampling})))
  sampling)

(defn rollouts
  "K (:group-size) completions of each of prompts (maps with :prompt, the
   namespace's docstring) by lm, all in lockstep (gen.llm.sample/complete):
   [{:prompt p :context [token …] :completions [{:tokens :end-token :stop
   :logprobs :text} …]} …], :tokens the drawn tokens (the end token not
   among them), :stop :end or :max, :logprobs each token's old
   log-probability at the sampling temperature, the end token's last
   when it stopped at one, :text the tokens decoded. Options:
     :key         the rollouts' key (required): completion j of prompt i
                  from (rng/split key (P·K))[i·K + j]
     :group-size  K (8)
     :sampling    gen.llm.sample's (policy-sampling: the model at
                  temperature 1)
     :max-tokens  the most tokens a completion draws, its end token
                  among them (256)
     :end-tokens  the tokens that end one (the LM's :eog)
     :chat        gen.llm.chat/render's options (:thinking?, …)"
  [lm prompts {:keys [key group-size sampling max-tokens end-tokens chat]
               :or {group-size 8 sampling policy-sampling max-tokens 256} :as opts}]
  (options/check `rollouts opts #{:key :group-size :sampling :max-tokens :end-tokens :chat})
  (when-not (rng/key? key) (throw (invalid (str "rollouts takes a :key (gen.random/key), not " (pr-str key)))))
  (when-not (pos-int? group-size)
    (throw (invalid (str ":group-size is a positive integer, not " (pr-str group-size)) {:group-size group-size})))
  (when-not (pos-int? max-tokens) (throw (invalid (str ":max-tokens is a positive integer, not " (pr-str max-tokens)))))
  (when-not (and (sequential? prompts) (seq prompts))
    (throw (invalid (str "rollouts takes prompts, a non-empty sequence, not " (pr-str prompts)))))
  (check-sampling sampling)
  (let [prompts (vec prompts)
        k (long group-size)
        contexts (let [tok (lm-tokenizer lm)] (mapv #(context tok %1 %2 chat) (range) prompts))
        drawn (sample/complete lm (vec (for [c contexts _ (range k)] c))
                               (cond-> {:sampling sampling :max-tokens max-tokens :token-logprobs? true :key key}
                                 end-tokens (assoc :end-tokens end-tokens)))]
    (mapv (fn [i p c]
            {:prompt p
             :context c
             :completions (mapv (fn [{:keys [tokens stop end-token token-logprobs]}]
                                  {:tokens tokens :stop stop :end-token end-token :logprobs token-logprobs
                                   :text (lm/decode lm tokens)})
                                (subvec drawn (c/* (long i) k) (c/* (inc (long i)) k)))})
          (range) prompts contexts)))

(defn- trained-tokens
  "A completion's tokens trained: its drawn tokens, and its end token when
   it stopped at one."
  [{:keys [tokens stop end-token]}]
  (if (= :end stop) (conj tokens end-token) tokens))

;; ---------------------------------------------------------------------------
;; Rewards
;; ---------------------------------------------------------------------------

(defn exact-match
  "A reward: 1.0 when a completion's answer is its prompt's, else 0.0. The
   prompt's answer is its :answer (or opts' :key's value), as a string,
   trimmed. The completion's is its text trimmed, or with :extract (a
   regex) its last match's first group (the whole match for a regex
   without a group), none matching 0.0."
  ([] (exact-match nil))
  ([{:keys [key extract] :or {key :answer} :as opts}]
   (options/check `exact-match opts #{:key :extract})
   (when-not (or (nil? extract) (instance? java.util.regex.Pattern extract))
     (throw (invalid (str ":extract is a regex, not " (pr-str extract)))))
   (fn [prompt completion]
     (let [want (get prompt key)
           got (if extract
                 (let [m (last (re-seq extract completion))]
                   (if (vector? m) (second m) m))
                 completion)]
       (when (nil? want)
         (throw (invalid (str "exact-match: the prompt has no " key) {:prompt prompt})))
       (if (and (some? got) (= (str/trim (str want)) (str/trim got))) 1.0 0.0)))))

(defn format-match
  "A reward: 1.0 when a completion's text, trimmed, matches the regex re
   whole (re-matches), else 0.0: a format checker."
  [re]
  (when-not (instance? java.util.regex.Pattern re)
    (throw (invalid (str "format-match takes a regex, not " (pr-str re)))))
  (fn [_ completion] (if (re-matches re (str/trim completion)) 1.0 0.0)))

(defn rewards
  "The rewards of rollouts' completions, by group: [[r …] …], (reward
   prompt text) on the host, one after another. A reward that is not a
   finite number is loud (:non-finite-reward, naming the prompt and the
   completion)."
  [reward rollouts]
  (when-not (ifn? reward) (throw (invalid (str "a reward is a function of a prompt and a completion, not " (pr-str reward)))))
  (mapv (fn [i {:keys [prompt completions]}]
          (mapv (fn [j {:keys [text]}]
                  (let [r (reward prompt text)]
                    (when-not (and (number? r) (Double/isFinite (double r)))
                      (throw (ex-info (str "grpo: the reward of prompt " i "'s completion " j " is " (pr-str r)
                                           ", not a finite number: no step trains on it")
                                      {:reason :non-finite-reward :prompt i :completion j :value r :text text})))
                    (double r)))
                (range) completions))
        (range) rollouts))

;; ---------------------------------------------------------------------------
;; Advantages
;; ---------------------------------------------------------------------------

(defn flat?
  "Are a group's rewards all equal (its advantages 0)?"
  [group]
  (every? #(c/== (double %) (double (first group))) group))

(defn advantages
  "Each completion's advantage, by group: (r - mean(r)) / (std(r) + eps)
   over its group's rewards r (std the population's), or r - mean(r) with
   :scale-rewards? false; exactly 0 for a group whose rewards are all
   equal. Options: :scale-rewards? (true), :eps (1e-4)."
  ([rewards] (advantages rewards nil))
  ([rewards {:keys [scale-rewards? eps] :or {scale-rewards? true eps 1e-4} :as opts}]
   (options/check `advantages opts #{:scale-rewards? :eps})
   (when-not (and (number? eps) (c/>= (double eps) 0.0)) (throw (invalid (str ":eps is a number >= 0, not " (pr-str eps)))))
   (mapv (fn [group]
           (let [k (double (count group))
                 mean (c// (double (reduce c/+ 0.0 group)) k)
                 std (Math/sqrt (c// (double (reduce c/+ 0.0 (map #(let [d (c/- (double %) mean)] (c/* d d)) group))) k))]
             (if (flat? group)
               (vec (repeat (count group) 0.0))
               (mapv #(let [d (c/- (double %) mean)] (if scale-rewards? (c// d (c/+ std (double eps))) d)) group))))
         rewards)))

;; ---------------------------------------------------------------------------
;; Batches
;; ---------------------------------------------------------------------------

(def loss-types
  "The losses, each its paper's (the namespace's docstring), with their
   defaults."
  {:grpo {:eps-low 0.2 :eps-high 0.2 :beta 0.04 :scale-rewards? true}
   :dapo {:eps-low 0.2 :eps-high 0.28 :beta 0.0 :scale-rewards? true}
   :dr-grpo {:eps-low 0.2 :eps-high 0.2 :beta 0.0 :scale-rewards? false}
   :bnpo {:eps-low 0.2 :eps-high 0.2 :beta 0.0 :scale-rewards? true}})

(defn- check-loss-type [t]
  (when-not (contains? loss-types t)
    (throw (invalid (str ":loss-type is one of " (vec (keys loss-types)) ", not " (pr-str t)) {:loss-type t})))
  t)

(defn- row-of
  "A completion as a row: {:tokens (its prompt's, then its own) :start
   (its first target: the prompt's length - 1) :old (its log-probs) :n
   (its tokens) :advantage}."
  [context completion advantage]
  (let [own (trained-tokens completion)
        old (:logprobs completion)]
    (when-not (c/== (count own) (count old))
      (throw (invalid (str "a completion of " (count own) " tokens trained has " (count old) " log-probabilities"
                           " (rollouts' completions, with :logprobs)")
                      {:tokens (count own) :logprobs (count old)})))
    {:tokens (into context own) :start (dec (count context)) :old old :n (count own) :advantage advantage}))

(defn- divisor
  "A microbatch's share of the step's divisor, by loss type: its
   completions (:grpo), its completion tokens (:dapo), its completions
   times the generation budget (:dr-grpo), or 1 (:bnpo)."
  [loss-type completions tokens budget]
  (case loss-type
    :grpo (long completions)
    :dapo (long tokens)
    :dr-grpo (c/* (long completions) (long budget))
    :bnpo 1))

(defn- batch-of
  "Rows as one microbatch of loss-type with generation budget budget."
  [rows loss-type budget pad]
  (let [t (max 2 (reduce max (map (comp count :tokens) rows)))
        tokens-in (reduce c/+ 0 (map :n rows))
        weight (fn [{:keys [n]}]
                 (case loss-type
                   :grpo (c// 1.0 (double n))
                   (:dapo :dr-grpo) 1.0
                   :bnpo (c// 1.0 (double tokens-in))))
        targets (fn [{:keys [start n]} f]
                  (mapv (fn [p] (let [k (c/- (long p) (long start))] (if (c/< -1 k (long n)) (f k) 0.0))) (range (dec t))))]
    {:tokens (array/array (mapv #(into (:tokens %) (repeat (c/- t (count (:tokens %))) pad)) rows) {:dtype :int32})
     :mask (array/array (mapv #(targets % (constantly 1.0)) rows))
     :weights (array/array (mapv (fn [r] (let [w (weight r)] (targets r (constantly w)))) rows))
     :old (array/array (mapv (fn [r] (targets r #(double ((:old r) %)))) rows))
     :advantages (array/array (mapv (fn [r] [(double (:advantage r))]) rows))
     :count (divisor loss-type (count rows) tokens-in budget)
     :completions (count rows)
     :completion-tokens tokens-in}))

(defn batches
  "rollouts with their advantages (by group, advantages') as microbatches
   of :batch-size rows (all of them by default), row i·K + j completion j
   of prompt i: {:tokens int32 [B T] (the prompt's tokens, the
   completion's, then :pad to the longest row) :mask [B T-1] (1 at the
   completion's targets) :weights [B T-1] (each target's weight in the
   summed loss, by :loss-type) :old [B T-1] (the old log-probabilities
   there, 0 elsewhere) :advantages [B 1] :count (the microbatch's share of
   the step's divisor) :completions :completion-tokens} (the namespace's
   docstring). Options: :loss-type (:grpo), :batch-size, :max-tokens (the
   generation budget, :dr-grpo's L; required for it), :pad (0)."
  [rollouts advantages {:keys [loss-type batch-size max-tokens pad] :or {loss-type :grpo pad 0} :as opts}]
  (options/check `batches opts #{:loss-type :batch-size :max-tokens :pad})
  (check-loss-type loss-type)
  (when (and (= :dr-grpo loss-type) (not (pos-int? max-tokens)))
    (throw (invalid (str ":dr-grpo divides by the generation budget: give :max-tokens, a positive integer, not " (pr-str max-tokens)))))
  (when-not (nat-int? pad) (throw (invalid (str ":pad is a token id, not " (pr-str pad)))))
  (when-not (= (count rollouts) (count advantages))
    (throw (invalid (str (count rollouts) " groups of rollouts, but " (count advantages) " of advantages"))))
  (let [rows (vec (mapcat (fn [{:keys [context completions]} as]
                            (when-not (= (count completions) (count as))
                              (throw (invalid (str "a group of " (count completions) " completions, but " (count as) " advantages"))))
                            (map #(row-of context %1 %2) completions as))
                          rollouts advantages))
        size (or batch-size (count rows))]
    (when-not (pos-int? size) (throw (invalid (str ":batch-size is a positive integer, not " (pr-str batch-size)))))
    (mapv #(batch-of (vec %) loss-type max-tokens pad) (partition-all size rows))))

(defn- log-probs-options
  "The family's log-probs options of the loss options."
  [{:keys [checkpoint? chunk temperature]}]
  (cond-> {:checkpoint? (boolean checkpoint?)}
    chunk (assoc :chunk chunk)
    (and temperature (not (c/== 1.0 (double temperature)))) (assoc :temperature temperature)))

(defn with-reference
  "batches with :ref, the reference's log-probabilities [B T-1] at T: the
   base model m's (the adapter at scale 0), read without a gradient."
  [m batches {:keys [temperature chunk]}]
  (mapv (fn [b]
          (let [r (models/log-probs m (:tokens b) (log-probs-options {:checkpoint? false :chunk chunk :temperature temperature}))]
            (array/eval! r)
            (assoc b :ref r)))
        batches))

;; ---------------------------------------------------------------------------
;; The loss
;; ---------------------------------------------------------------------------

(defn- k3
  "π_ref/π_θ - log(π_ref/π_θ) - 1 of d = log π_ref - log π_θ (DeepSeekMath's
   Eq. 4), as expm1(d) - d."
  [d]
  (- (array/expm1 d) d))

(defn- ratio-of
  "ρ = π_θ/π_θold = exp(lp - old) at the completions' tokens (on), 1
   elsewhere."
  [lp old on]
  (array/exp (array/where on (- lp old) 0.0)))

(defn loss
  "[batch b's summed loss under model m (adapted), the log-probabilities
   [B T-1] it read]: Σ :weights·ℓ over the completions' tokens, ℓ the
   clipped surrogate's negation plus β times the k3 KL to b's :ref (the
   namespace's docstring), differentiable with respect to an adapter.
   Options: :eps-low, :eps-high, :beta (β > 0 needs b's :ref,
   with-reference's), :temperature (T, 1), and the family's
   :checkpoint? and :chunk."
  [m {:keys [tokens mask weights old advantages ref]} {:keys [eps-low eps-high beta] :as opts}]
  (let [beta (double beta)
        lp (models/log-probs m tokens (log-probs-options opts))
        on (> mask 0.0)
        ratio (ratio-of lp old on)
        surrogate (array/minimum (* ratio advantages)
                                 (* (array/clip ratio (c/- 1.0 (double eps-low)) (c/+ 1.0 (double eps-high))) advantages))
        per-token (if (pos? beta)
                    (do (when-not ref
                          (throw (invalid "β > 0 needs the reference's log-probabilities: the batch's :ref (with-reference)")))
                        (- (* beta (k3 (array/where on (- ref lp) 0.0))) surrogate))
                    (array/negate surrogate))]
    [(array/sum (* weights per-token)) lp]))

(defn- stats
  "An optimizer step's {:ratio-max :clip-fraction :kl} from the log-probs
   each batch's loss read (sft/step's :reported)."
  [batches reported {:keys [eps-low eps-high]}]
  (let [lo (c/- 1.0 (double eps-low))
        hi (c/+ 1.0 (double eps-high))
        per (mapv (fn [{:keys [mask old advantages ref]} lp]
                    (let [on (> mask 0.0)
                          ratio (ratio-of lp old on)
                          clipped (array/logical-or (array/logical-and (< ratio lo) (< advantages 0.0))
                                                    (array/logical-and (> ratio hi) (> advantages 0.0)))
                          xs (cond-> [(array/max (array/where on (array/abs (- ratio 1.0)) 0.0))
                                      (array/sum (array/where (array/logical-and on clipped) 1.0 0.0))
                                      (array/sum mask)]
                               ref (conj (array/sum (array/where on (k3 (array/where on (- ref lp) 0.0)) 0.0))))]
                      (mapv (comp double array/item) (array/eval! xs))))
                  batches reported)
        tokens (reduce c/+ 0.0 (map #(nth % 2) per))]
    (cond-> {:ratio-max (reduce max 0.0 (map first per))
             :clip-fraction (c// (double (reduce c/+ 0.0 (map second per))) tokens)}
      (every? #(c/== 4 (count %)) per) (assoc :kl (c// (double (reduce c/+ 0.0 (map #(nth % 3) per))) tokens)))))

;; ---------------------------------------------------------------------------
;; The trainer and its rounds
;; ---------------------------------------------------------------------------

(defn trainer
  "A trainer of adapter by optimizer (gen.learning's), its randomness from
   key: gen.llm.sft's trainer (its :key the first of (split key)) with
   :round 0 and :rollout-key, the second."
  [adapter optimizer key]
  (when-not (rng/key? key) (throw (invalid (str "trainer takes a key (gen.random/key), not " (pr-str key)))))
  (let [[k-train k-rollouts] (rng/split key)]
    (assoc (sft/trainer adapter optimizer k-train) :round 0 :rollout-key k-rollouts)))

(defn- check-trainer [t]
  (when-not (and (map? t) (nat-int? (:round t)) (rng/key? (:rollout-key t)) (map? (:adapter t)))
    (throw (invalid "a GRPO trainer is gen.llm.sft's with :round and :rollout-key, as trainer makes it"))))

(def ^:private round-keys
  #{:group-size :iterations :loss-type :eps-low :eps-high :beta :scale-rewards? :eps :sampling :max-tokens :end-tokens
    :chat :batch-size :pad :checkpoint? :chunk :kl? :engine})

(defn- round-options
  "A round's options with the defaults, of the loss type's too, checked."
  [opts]
  (options/check `round opts round-keys)
  (let [loss-type (check-loss-type (get opts :loss-type :grpo))
        o (merge {:group-size 8 :iterations 1 :loss-type loss-type :eps 1e-4 :sampling policy-sampling :max-tokens 256
                  :pad 0 :checkpoint? true :chunk 16384 :kl? true}
                 (loss-types loss-type)
                 opts)
        {:keys [group-size iterations eps-low eps-high beta]} o]
    (when-not (and (int? group-size) (c/>= (long group-size) 2))
      (throw (invalid (str ":group-size is an integer of at least 2 (a group's advantages are relative), not " (pr-str group-size))
                      {:group-size group-size})))
    (when-not (pos-int? iterations) (throw (invalid (str ":iterations (μ) is a positive integer, not " (pr-str iterations)))))
    (when-not (and (number? eps-low) (c/< 0.0 (double eps-low) 1.0))
      (throw (invalid (str ":eps-low is in (0, 1), not " (pr-str eps-low)) {:eps-low eps-low})))
    (when-not (and (number? eps-high) (pos? (double eps-high)) (Double/isFinite (double eps-high)))
      (throw (invalid (str ":eps-high is a positive number, not " (pr-str eps-high)) {:eps-high eps-high})))
    (when-not (and (number? beta) (c/>= (double beta) 0.0) (Double/isFinite (double beta)))
      (throw (invalid (str ":beta is a finite number >= 0, not " (pr-str beta)) {:beta beta})))
    (when-not (or (nil? (:engine o)) (map? (:engine o)))
      (throw (invalid (str ":engine is gen.llm.array-engine/engine's options, not " (pr-str (:engine o))))))
    (check-sampling (:sampling o))
    o))

(defn- seconds [t0] (c// (double (c/- (System/nanoTime) (long t0))) 1e9))

(defn- population-std [xs]
  (let [n (double (count xs))
        mean (c// (double (reduce c/+ 0.0 xs)) n)]
    [mean (Math/sqrt (c// (double (reduce c/+ 0.0 (map #(let [d (c/- (double %) mean)] (c/* d d)) xs))) n))]))

(defn- drawn
  "The rollouts of a round's prompts, drawn from key by an LM over the
   array engine with m adapted by the trainer's adapter."
  [m tokenizer trainer prompts {:keys [group-size max-tokens engine chat] :as o} key]
  (let [n (c/* (count prompts) (long group-size))
        branches (c/+ n (count prompts) 8)
        ;; every context and completion, a shared prefix counted again: a bound
        cells (c/+ (c/* n (long max-tokens)) (reduce c/+ 0 (map #(count (context tokenizer %1 %2 chat)) (range) prompts)))]
    (with-open [e (ae/engine (merge {:max-branches branches} engine {:model (lora/adapt m (:adapter trainer))}))
                l (lm/lm e tokenizer {:max-branches branches :max-cells cells})]
      (rollouts l prompts (assoc (select-keys o [:group-size :sampling :max-tokens :end-tokens :chat]) :key key)))))

(defn round
  "[trainer' info]: one round of trainer on model m (the frozen base its
   adapter adapts, of any family of gen.llm.models), prompts its prompts:
   K completions of each drawn by gen.llm.array-engine over the adapted
   model (an LM of tokenizer, the model's: gen.llm.llama/tokenizer of its
   vocabulary), their rewards by reward, the advantages, then :iterations
   optimizer steps (gen.llm.sft/step) on the loss over them, the old
   log-probabilities fixed; trainer' its :round the next. info is the
   round's log (the namespace's docstring). Options:
     :group-size     K, at least 2 (8)
     :iterations     μ, optimizer steps on a round's rollouts (1)
     :loss-type      :grpo, :dapo, :dr-grpo or :bnpo (:grpo)
     :eps-low, :eps-high, :beta, :scale-rewards?
                     the loss type's (loss-types) unless given
     :eps            the advantages' (1e-4)
     :sampling       gen.llm.sample's (policy-sampling)
     :max-tokens     a completion's budget, its end token among them (256)
     :end-tokens     the tokens that end a completion (the tokenizer's :eog)
     :chat           gen.llm.chat/render's options for the prompts
     :batch-size     rows a microbatch (all of the round's)
     :pad            the padding token (0)
     :checkpoint?, :chunk   the family's log-probs' (true, 16384: sft's)
     :kl?            read the reference's log-probabilities for info's :kl
                     even when :beta is 0 (true): one forward of the base a
                     round
     :engine         gen.llm.array-engine/engine's options but :model
                     (:max-branches defaults to the round's completions,
                     its prompts and 8)
   Loud for a reward that is not finite (:non-finite-reward) and a loss
   or gradient that is not finite (:non-finite, sft/step's): never a
   silent skip."
  [m tokenizer trainer prompts reward opts]
  (check-trainer trainer)
  (let [{:keys [iterations loss-type beta kl? sampling] :as o} (round-options opts)
        s (long (:round trainer))
        [_ k-samples] (rng/split (:rollout-key trainer))
        t0 (System/nanoTime)
        ro (drawn m tokenizer trainer (vec prompts) o (rng/fold-in k-samples s))
        rollout-seconds (seconds t0)
        t1 (System/nanoTime)
        rs (rewards reward ro)
        adv (advantages rs (select-keys o [:scale-rewards? :eps]))
        temperature (double (:temp sampling))
        bs (batches ro adv (select-keys o [:loss-type :batch-size :max-tokens :pad]))
        bs (if (or (pos? (double beta)) kl?) (with-reference m bs {:temperature temperature :chunk (:chunk o)}) bs)
        loss-opts (assoc (select-keys o [:eps-low :eps-high :beta]) :temperature temperature)
        step-opts (cond-> {:loss (fn [m' b opts] (loss m' b (merge loss-opts opts))) :checkpoint? (:checkpoint? o)}
                    (:chunk o) (assoc :chunk (:chunk o)))
        [t its] (reduce (fn [[t its] _]
                          (let [[t info] (sft/step m t bs step-opts)]
                            [t (conj its (merge (select-keys info [:step :loss :grad-norm])
                                                (stats bs (:reported info) o)))]))
                        [trainer []]
                        (range iterations))
        completions (mapcat :completions ro)
        lengths (map #(count (trained-tokens %)) completions)
        logprobs (mapcat :logprobs completions)
        [mean std] (population-std (apply concat rs))]
    [(assoc t :round (inc s))
     {:round s
      :step (:step t)
      :loss-type loss-type
      :reward-mean mean
      :reward-std std
      :groups (count rs)
      :flat-groups (count (filter flat? rs))
      :completions (count completions)
      :tokens (reduce c/+ 0 lengths)
      :length-mean (c// (double (reduce c/+ 0 lengths)) (double (count lengths)))
      :truncated (count (filter #(= :max (:stop %)) completions))
      :entropy (c// (double (reduce c/+ 0.0 (map #(c/- (double %)) logprobs))) (double (count logprobs)))
      :iterations its
      :seconds {:rollouts rollout-seconds :train (seconds t1)}}]))

(def ^:private train-keys #{:rounds :prompts-per-round :callback :save-dir :save-every})

(defn train
  "trainer trained on model m by rounds (round's) until its :round is
   :rounds, round s on :prompts-per-round of prompts (maps with :prompt,
   the namespace's docstring) taken in epochs of an order drawn from the
   trainer's :rollout-key (gen.llm.sft/order), so a trainer loaded from a
   checkpoint goes on as the run that saved it would have. Returns
   {:trainer :history}, :history each round's info with :prompts (the
   indices of its prompts). Options: round's, and
     :rounds             the round to train to (required)
     :prompts-per-round  P (8)
     :callback           called after each round with its info and the
                         :trainer; its value is ignored
     :save-dir           where gen.llm.sft/save-checkpoint writes, every
                         :save-every rounds (default none) and after the
                         last
     :save-every"
  [m tokenizer trainer prompts reward {:keys [rounds prompts-per-round callback save-dir save-every]
                                       :or {prompts-per-round 8} :as opts}]
  (options/check `train opts (into round-keys train-keys))
  (check-trainer trainer)
  (let [prompts (vec prompts)
        n (count prompts)
        round-opts (apply dissoc opts train-keys)
        [k-prompts] (rng/split (:rollout-key trainer))]
    (when (zero? n) (throw (invalid "train takes at least one prompt")))
    (when-not (pos-int? prompts-per-round)
      (throw (invalid (str ":prompts-per-round is a positive integer, not " (pr-str prompts-per-round)))))
    (when-not (and (nat-int? rounds) (c/>= (long rounds) (long (:round trainer))))
      (throw (invalid (str ":rounds is the round to train to, at least the trainer's " (:round trainer) ", not " (pr-str rounds))
                      {:rounds rounds :round (:round trainer)})))
    (when-not (or (nil? save-every) (pos-int? save-every))
      (throw (invalid (str ":save-every is a positive integer, not " (pr-str save-every)))))
    (round-options round-opts)
    (loop [t trainer history []]
      (if (c/>= (long (:round t)) (long rounds))
        {:trainer t :history history}
        (let [idx (sft/order k-prompts n (:round t) prompts-per-round)
              [t info] (round m tokenizer t (mapv prompts idx) reward round-opts)
              info (assoc info :prompts idx)
              done? (c/>= (long (:round t)) (long rounds))]
          (when callback (callback (assoc info :trainer t)))
          (when (and save-dir (or done? (and save-every (zero? (rem (long (:round t)) (long save-every))))))
            (sft/save-checkpoint t save-dir))
          (recur t (conj history info)))))))

(defn evaluate
  "The rewards of lm's completions of prompts (maps with :prompt):
   {:mean (over every completion) :per-prompt [each prompt's mean]
   :rewards [[r …] …] :rollouts}. Options: rollouts', its :key required,
   :group-size the completions a prompt (1); greedy decoding is
   :sampling {:temp 1 :top-k 1 :top-p 1}."
  [lm prompts reward opts]
  (let [ro (rollouts lm prompts (merge {:group-size 1} opts))
        rs (rewards reward ro)
        means (mapv #(c// (double (reduce c/+ 0.0 %)) (double (count %))) rs)]
    {:mean (c// (double (reduce c/+ 0.0 (apply concat rs))) (double (count (apply concat rs))))
     :per-prompt means
     :rewards rs
     :rollouts ro}))

