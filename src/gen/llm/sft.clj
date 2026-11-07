(ns gen.llm.sft
  "Supervised fine-tuning (SFT): a LoRA adapter (gen.llm.lora) trained on
   chat examples. Its loss is the mean negative log-likelihood of the
   assistant's tokens under the adapted model, by the model's family
   (gen.llm.models/nll: Qwen3's, gen.llm.models.qwen3, or any family that
   registers its own).

     (example tokenizer ex)          → {:tokens :mask}: a chat example tokenized,
                                       :mask true at the tokens trained on
     (batch examples)                → {:tokens :mask :count}: examples padded
     (nll m batch)                   → the summed NLL of a batch's trained tokens
     (evaluate m examples)           → {:nll :tokens}: the mean NLL per trained token
     (trainer adapter optimizer key) → a trainer, a value: {:adapter :optimizer
                                       :optimizer-state :step :key}
     (step m trainer batches)        → [trainer' info]: one optimizer step
     (train m trainer examples opts) → {:trainer :history :evaluations}: steps to :steps
     (save-checkpoint trainer dir)   → the trainer written under dir, atomically
     (load-checkpoint path)          → the trainer again, exactly
     (latest-checkpoint dir)         → the path of dir's last complete checkpoint

   Examples. A chat example is {:messages [...]}, gen.llm.chat's messages,
   whose assistant messages are trained on, or {:prompt :completion}: the
   prompt a user message's text (or messages), the completion the
   assistant's reply to it (a string), the only turn trained on. example
   renders it with Qwen's chat template (gen.llm.chat/segments; its options
   as :chat, the turns rendered without a generation prompt) and tokenizes
   the whole text once (the tokenizer a gen.llm.lm tokenizer map:
   gen.llm.llama/tokenizer of the GGUF's vocabulary, BOS first where it
   adds one, special tokens parsed). A token is trained on when its bytes
   lie in an assistant turn's content, or are its end token <|im_end|>
   (with :end-token?, the default): the model learns where its reply
   ends. The template's text, the think block, the newline after the end
   token and every other turn are context. The tokens' bytes are matched
   to the text, so a token that crosses into a trained span from outside
   is a loud error, as is an example with nothing to train on.

   Batches. batch pads examples to the longest (each first truncated to
   :max-length tokens, keeping its start), right-padded with :pad, which
   causal attention never lets a real token read: {:tokens int32 [B T]
   :mask float [B T-1] :count}, :mask 1 at the targets trained on (the
   log-probabilities a family's log-probs gives, positions 1 to T-1), :count
   their number. An example truncated to nothing to train on is loud.

   The loss is the mean over a step's trained tokens: each microbatch's
   summed NLL (the family's nll with the mask, its blocks checkpointed and
   its output projection in vocabulary chunks) and its gradient with respect
   to the adapter's params are added up over the step's batches, then
   divided once by their tokens together, so k microbatches step as one
   batch of them all would (gradient accumulation), but for rounding. The
   optimizer is any of gen.learning's (adamw, under clip-norm, clip-value
   and warmup-cosine), stepped along the negated mean gradient (gen.learning
   ascends). A loss or gradient that is not finite is a loud error
   (:non-finite), the trainer not stepped: never a silent skip. step's
   :loss puts another summed loss in the NLL's place, over batches of
   its own with their :count: gen.llm.grpo's policy-gradient losses run
   on this machinery.

   Keys are values: step n of a trainer draws its adapter's dropout from
   (fold-in k n), [k k'] = (split (:key trainer)), a key per microbatch
   split from it; train draws epoch e's order of the examples from
   (fold-in k' e). So a trainer's next step depends on the trainer value
   alone, and a checkpoint resumes exactly: a run saved at step n, loaded
   and trained on to step m is the run trained to m without a stop, bit
   for bit.

   Checkpoints. save-checkpoint writes dir/step-<n>/ (n the trainer's
   :step, eight digits): adapter.gguf, the adapter as llama.cpp's LoRA GGUF
   (lora/save-gguf, float32: gen.llm.llama/adapter serves it), and
   trainer.edn, the whole trainer as gen.serialize's EDN (its arrays
   exact in every dtype: the params and the optimizer's state, its step
   counts, the key). Both are written in a hidden directory beside it,
   forced to the disk, then moved to step-<n> in one atomic rename: a
   reader never sees a checkpoint half written, and a crash leaves only
   the hidden directory, which latest-checkpoint ignores. A checkpoint
   loads on the backend and dtype it was written from.

   A model whose family registers no nll with gen.llm.models is
   :unsupported.

   MLX's memory (genclj-az5s). A step's peak depends on MLX's memory
   limit, a process-wide setting this namespace leaves alone: under a
   loose one (gen.mlx.memory-bytes, by default half of RAM) MLX keeps the
   buffers of kernels it has queued, and most of the checkpoints' saving
   is lost. Set the limit to what MLX holds before a step (the weights,
   the adapter, the optimizer's state) plus about 1.25 times a step's own
   peak, which bench/gen/sft_bench.clj measures: the saving is then the
   checkpoints', at no cost in time. Measured on fenrir's CUDA
   (2026-11-07; Qwen3-0.6B Q8_0, rank 8 on every projection of the last
   8 blocks, native gradients, checkpointed, 0.76 GB held), the peak past
   that, at the default limit, at the limit so set, and at the held bytes
   alone (MLX then waits for its queue at every kernel, 3-10% slower):
     4 rows of 256 tokens   6.1 GB   1.8 GB   1.5 GB   (274, 273, 305 ms)
     8 rows of 512 tokens  24.6 GB  11.7 GB   9.2 GB   (968, 967, 988 ms)
   Unchecked, the same steps peak at 11.4 and 32.1 GB at the default.

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [clojure.java.io :as io]
            [gen.array :as array :refer [+ * /]]
            [gen.inference.options :as options]
            [gen.learning :as learn]
            [gen.llm.chat :as chat]
            [gen.llm.lora :as lora]
            [gen.llm.models :as models]
            [gen.random :as rng]
            [gen.serialize :as ser])
  (:import [java.io File FileOutputStream]
           [java.nio.charset StandardCharsets]
           [java.nio.file CopyOption Files StandardCopyOption]))

(set! *warn-on-reflection* true)

(defn- invalid
  ([message] (invalid message {}))
  ([message data] (ex-info (str "sft: " message) (merge {:reason :invalid-args} data))))

;; ---------------------------------------------------------------------------
;; Examples: chat turns tokenized, the assistant's tokens marked
;; ---------------------------------------------------------------------------

(defn- utf8 ^bytes [^String s] (.getBytes s StandardCharsets/UTF_8))

(defn- messages-of
  "[messages, the indices of those trained on] of a chat example."
  [ex]
  (cond
    (and (map? ex) (contains? ex :messages) (not (contains? ex :prompt)))
    (let [ms (vec (:messages ex))
          trained (into #{} (keep-indexed (fn [i m] (when (= :assistant (:role m)) i))) ms)]
      (when (empty? trained)
        (throw (invalid "an example's :messages hold no assistant message to train on" {:reason :nothing-trained})))
      [ms trained])

    (and (map? ex) (contains? ex :prompt) (contains? ex :completion) (not (contains? ex :messages)))
    (let [{:keys [prompt completion]} ex
          ms (cond (string? prompt) [{:role :user :content prompt}]
                   (and (sequential? prompt) (every? map? prompt)) (vec prompt)
                   :else (throw (invalid (str "an example's :prompt is a string or messages, not " (pr-str prompt)))))]
      (when-not (string? completion)
        (throw (invalid (str "an example's :completion is the assistant's reply, a string, not " (pr-str completion)))))
      [(conj ms {:role :assistant :content completion}) #{(count ms)}])

    :else (throw (invalid (str "an example is {:messages [...]} or {:prompt :completion}, not " (pr-str ex))))))

(defn- trained-spans
  "[text, the byte spans [start end) trained on] of segments (chat's),
   those of the messages trained whose part is in parts."
  [segments trained parts]
  (loop [[s & more] segments at 0 text (StringBuilder.) spans []]
    (if-not s
      [(str text) spans]
      (let [n (alength (utf8 (:text s)))
            end (c/+ (long at) n)]
        (recur more end (.append text ^String (:text s))
               (if (and (contains? trained (:message s)) (contains? parts (:part s)))
                 (conj spans [at end])
                 spans))))))

(defn- token-spans
  "The byte span [start end) of each of tokens in the text they encode (a
   token added before the text, BOS, spans [0 0)): loud when their bytes
   are not the text's."
  [{:keys [token-bytes]} tokens ^bytes text bos?]
  (let [n (alength text)]
    (loop [i 0 at 0 spans []]
      (if (c/== i (count tokens))
        (do (when-not (c/== at n)
              (throw (ex-info (str "sft: the tokens spell " at " of the text's " n " bytes")
                              {:reason :misaligned-tokens :bytes at :text-bytes n})))
            spans)
        (if (and bos? (zero? i))
          (recur 1 at (conj spans [0 0]))
          (let [^bytes b (token-bytes (tokens i))
                end (c/+ at (alength b))]
            (when-not (and (c/<= end n) (java.util.Arrays/equals b 0 (alength b) text at end))
              (throw (ex-info (str "sft: token " i " (" (tokens i) ") is not the text's bytes at " at)
                              {:reason :misaligned-tokens :token (tokens i) :index i :at at})))
            (recur (inc i) end (conj spans [at end]))))))))

(defn example
  "A chat example tokenized: {:tokens [token …] :mask [bool …]}, :mask
   true at each token trained on (the namespace's docstring). ex is
   {:messages [...]} or {:prompt :completion}; tokenizer a gen.llm.lm
   tokenizer map (:encode, :token-bytes, :add-bos? and :bos). Options:
     :chat        gen.llm.chat/render's options (:template, :thinking?,
                  :tools, …); the turns are rendered without a generation
                  prompt
     :end-token?  train each assistant turn's end token, <|im_end|> (true)
   Loud for an example with nothing to train on (:nothing-trained), for
   tokens whose bytes are not the text's or cross a trained span's edge
   (:misaligned-tokens, :straddling-token), and for unknown options."
  ([tokenizer ex] (example tokenizer ex nil))
  ([tokenizer ex {:keys [chat end-token?] :or {end-token? true} :as opts}]
   (options/check `example opts #{:chat :end-token?})
   (when-not (and (map? tokenizer) (ifn? (:encode tokenizer)) (ifn? (:token-bytes tokenizer)))
     (throw (invalid (str "example takes a tokenizer map (gen.llm.lm's: :encode, :token-bytes), not " (pr-str tokenizer)))))
   (when (contains? chat :generation-prompt?)
     (throw (invalid "example renders its turns without a generation prompt: :chat takes no :generation-prompt?")))
   (let [[messages trained] (messages-of ex)
         segments (chat/segments messages (assoc chat :generation-prompt? false))
         [text spans] (trained-spans segments trained (if end-token? #{:content :end} #{:content}))
         bos? (boolean (and (:add-bos? tokenizer) (:bos tokenizer)))
         tokens (vec (concat (when bos? [(:bos tokenizer)])
                             ((:encode tokenizer) text {:add-special? false :parse-special? true})))
         bytes (utf8 text)
         token-spans (token-spans tokenizer tokens bytes bos?)
         mask (mapv (fn [i [s e]]
                      (let [s (long s) e (long e)
                            inside (some (fn [[a b]] (c/<= (long a) s e (long b))) spans)
                            crossing (some (fn [[a b]] (and (c/< s (long b)) (c/> e (long a)))) spans)]
                        (cond (c/== s e) false
                              inside true
                              crossing (throw (ex-info (str "sft: token " i " (" (tokens i) ") crosses the edge of a trained span")
                                                       {:reason :straddling-token :token (tokens i) :index i :bytes [s e]}))
                              :else false)))
                    (range) token-spans)]
     (when-not (some true? mask)
       (throw (invalid "an example with no token to train on (its assistant turns are empty)" {:reason :nothing-trained})))
     (when (first mask)
       (throw (invalid "an example's first token is trained on, with nothing before it to predict it from"
                       {:reason :nothing-trained})))
     {:tokens tokens :mask mask})))

;; ---------------------------------------------------------------------------
;; Batches
;; ---------------------------------------------------------------------------

(defn- check-example [i {:keys [tokens mask]}]
  (when-not (and (vector? tokens) (vector? mask) (= (count tokens) (count mask)) (every? integer? tokens)
                 (every? boolean? mask))
    (throw (invalid (str "example " i " is not {:tokens :mask} as example gives it") {:example i}))))

(defn- targets
  "The mask of a row's targets, the log-probabilities of its positions 1 to
   T-1 (a family's log-probs'), from its tokens' mask: 1.0 where token
   t + 1 is trained on."
  [mask]
  (mapv #(if % 1.0 0.0) (rest mask)))

(defn batch
  "examples (example's) as one batch: {:tokens int32 [B T] :mask float [B
   T-1] :count}, each row an example's tokens, truncated to :max-length
   (keeping its start) and right-padded with :pad to the longest; :mask
   1 at the targets trained on (row b's position t the prediction of its
   token t + 1), :count their number. Options: :max-length (none), :pad
   (a token id, 0). Loud for an example left with nothing to train on
   (:nothing-trained), naming it."
  ([examples] (batch examples nil))
  ([examples {:keys [max-length pad] :or {pad 0} :as opts}]
   (options/check `batch opts #{:max-length :pad})
   (when-not (and (sequential? examples) (seq examples))
     (throw (invalid (str "batch takes examples, a non-empty sequence, not " (pr-str examples)))))
   (when-not (or (nil? max-length) (and (int? max-length) (c/>= (long max-length) 2)))
     (throw (invalid (str ":max-length is an integer of at least 2, not " (pr-str max-length)) {:max-length max-length})))
   (when-not (nat-int? pad)
     (throw (invalid (str ":pad is a token id, not " (pr-str pad)) {:pad pad})))
   (let [rows (vec (map-indexed (fn [i ex]
                                  (check-example i ex)
                                  (let [{:keys [tokens mask]} (if (and max-length (c/> (count (:tokens ex)) (long max-length)))
                                                                (-> ex (update :tokens subvec 0 max-length) (update :mask subvec 0 max-length))
                                                                ex)]
                                    (when-not (some true? (rest mask))
                                      (throw (invalid (str "example " i " has nothing to train on"
                                                           (when max-length (str " in its first " max-length " tokens")))
                                                      {:reason :nothing-trained :example i :max-length max-length})))
                                    [tokens mask]))
                                examples))
         t (max 2 (reduce max (map (comp count first) rows)))
         padded (fn [xs fill] (into xs (repeat (c/- t (count xs)) fill)))
         masks (mapv (fn [[_ mask]] (targets (padded mask false))) rows)]
     {:tokens (array/array (mapv (fn [[tokens]] (padded tokens pad)) rows) {:dtype :int32})
      :mask (array/array masks)
      :count (long (reduce c/+ 0 (map (fn [m] (count (filter pos? m))) masks)))})))

;; ---------------------------------------------------------------------------
;; The loss
;; ---------------------------------------------------------------------------

(def ^:private loss-defaults {:checkpoint? true :chunk 16384})

(defn nll
  "The negative log-likelihood of batch's trained tokens under model m (an
   adapted one, lora/adapt's, or not), summed: a 0-d array, differentiable
   as its family's nll is (gen.llm.models/nll). Options: :checkpoint?
   (true) and :chunk (16384, vocabulary entries a chunk of the output
   projection), as the families' log-probs take them."
  ([m batch] (nll m batch nil))
  ([m {:keys [tokens mask]} opts]
   (options/check `nll opts #{:checkpoint? :chunk})
   (models/nll m tokens (merge loss-defaults opts {:mask mask}))))

(defn- batches-of
  "examples in batches of size."
  [examples size batch-opts]
  (mapv #(batch % batch-opts) (partition-all size examples)))

(defn evaluate
  "The mean negative log-likelihood per trained token of examples
   (example's) under model m (adapted or not): {:nll the mean, a host
   double, :tokens their number}, read in batches. Options: :batch-size
   (8), batch's :max-length and :pad, nll's :chunk."
  ([m examples] (evaluate m examples nil))
  ([m examples {:keys [batch-size chunk] :or {batch-size 8} :as opts}]
   (options/check `evaluate opts #{:batch-size :max-length :pad :chunk})
   (when-not (pos-int? batch-size)
     (throw (invalid (str ":batch-size is a positive integer, not " (pr-str batch-size)) {:batch-size batch-size})))
   (let [[total n] (reduce (fn [[total n] b]
                             [(c/+ (double total) (double (array/item (nll m b (cond-> {:checkpoint? false} chunk (assoc :chunk chunk))))))
                              (c/+ (long n) (long (:count b)))])
                           [0.0 0]
                           (batches-of examples batch-size (select-keys opts [:max-length :pad])))]
     {:nll (c// (double total) (double n)) :tokens n})))

;; ---------------------------------------------------------------------------
;; The trainer and its step
;; ---------------------------------------------------------------------------

(defn trainer
  "A trainer of adapter (lora/init's, or one loaded) by optimizer (one of
   gen.learning's), its randomness from key: {:adapter :optimizer
   :optimizer-state :step 0 :key}, a value that step and train advance
   and save-checkpoint writes."
  [adapter optimizer key]
  (when-not (and (map? adapter) (map? (:config adapter)) (map? (:params adapter)))
    (throw (invalid (str "trainer takes an adapter {:config :params} (lora/init's), not " (pr-str adapter)))))
  (when-not (and (map? optimizer) (:kind optimizer))
    (throw (invalid (str "trainer takes an optimizer (gen.learning's, a map with a :kind), not " (pr-str optimizer)))))
  (when-not (rng/key? key)
    (throw (invalid (str "trainer takes a key (gen.random/key), not " (pr-str key)))))
  {:adapter adapter
   :optimizer optimizer
   :optimizer-state (learn/init optimizer (:params adapter))
   :step 0
   :key key})

(defn- check-trainer [{:keys [adapter optimizer step key] :as trainer}]
  (when-not (and (map? trainer) (map? adapter) (map? optimizer) (nat-int? step) (rng/key? key)
                 (contains? trainer :optimizer-state))
    (throw (invalid "a trainer is {:adapter :optimizer :optimizer-state :step :key}, as trainer makes it"))))

(defn- step-keys
  "[k-steps k-order]: the keys of a trainer's steps and of its examples'
   order."
  [trainer]
  (rng/split (:key trainer)))

(defn- global-norm
  "The Euclidean norm of every leaf of the tree g together, a 0-d array."
  [g]
  (array/sqrt (reduce + (map #(array/sum (* % %)) (array/tree-leaves g)))))

(defn- summed-loss
  "[the summed loss of batch b under m adapted by adapter (its dropout
   from key k), its gradient with respect to the adapter's params, the
   values the loss reported (nil for none)]: the loss :loss's (step's
   option) of the adapted model, b and the other loss options, else the
   summed NLL of b's trained tokens. A :loss's reported array x of b's
   :mask's shape comes back through a term Σ z·stop-gradient(x) at z = 0,
   differentiated with respect to z (its gradient is x, its value 0), so
   neither the loss nor its gradient changes."
  [m adapter b k {:keys [loss] :as loss-opts}]
  (let [loss-opts (dissoc loss-opts :loss)
        adapted (fn [params] (lora/adapt m (assoc adapter :params params) {:key k}))]
    (if-not loss
      (let [nll (fn [params] (models/nll (adapted params) (:tokens b) (assoc loss-opts :mask (:mask b))))]
        (conj ((array/value-and-grad nll) (:params adapter)) nil))
      (let [reported? (volatile! false)
            f (fn [params z]
                (let [out (loss (adapted params) b loss-opts)
                      [l x] (if (vector? out) out [out nil])]
                  (if (some? x)
                    (do (when-not (= (array/shape x) (array/shape z))
                          (throw (invalid (str "a :loss reports an array of the batch's :mask's shape " (array/shape z)
                                               ", not " (array/shape x))
                                          {:shapes [(array/shape x) (array/shape z)]})))
                        (vreset! reported? true)
                        (+ l (array/sum (* z (array/stop-gradient x)))))
                    l)))
            [l [g x]] ((array/value-and-grad f {:argnums [0 1]}) (:params adapter) (array/zeros-like (:mask b)))]
        [l g (when @reported? x)]))))

(defn- accumulated
  "[the summed loss, its gradient, the batches' counts (their :count
   summed: sft's trained tokens), what the loss reported, by batch] of
   batches together, batch i's dropout from (keys i): each batch's added
   to the sums and evaluated before the next, so a step holds one batch's
   graph at a time."
  [m adapter batches keys loss-opts]
  (reduce (fn [[total grads n-tokens reported] [b k]]
            (let [[l g x] (summed-loss m adapter b k loss-opts)
                  acc [(if total (+ total l) l) (if grads (learn/update-given + grads g) g)]]
              (array/eval! (cond-> acc x (conj x)))
              (conj acc (c/+ (long n-tokens) (long (:count b))) (conj reported x))))
          [nil nil 0 []]
          (map vector batches keys)))

(defn step
  "[trainer' info]: one optimizer step of trainer on model m (the frozen
   base its adapter adapts) along the mean NLL per trained token of
   batches (batch's), one batch or several accumulated: each one's summed
   NLL and gradient are added up, evaluated batch by batch, and divided
   once by their trained tokens together. info is {:step (the step taken,
   from 0) :loss (the mean NLL before the step) :tokens :grad-norm (the
   mean gradient's Euclidean norm, before the optimizer clips it)}, host
   numbers. Loud (:non-finite) when the loss or a gradient is not
   finite, the trainer unstepped. Options: nll's :checkpoint? (true) and
   :chunk, and
     :loss  another loss in place of the NLL (policy gradients:
            gen.llm.grpo): (fn [m' b opts]) of the adapted model m' (its
            dropout drawn), a batch b (any map with :mask, an array of
            its per-target shape, and :count, its share of the divisor)
            and the options (:checkpoint? :chunk), returning b's summed
            loss, a 0-d array differentiable with respect to the
            adapter, or [loss x], x an array of :mask's shape whose
            values (its gradient stopped) come back in info's :reported,
            one per batch (nil where a loss reported none). The step
            divides the batches' summed losses by their :count summed
            (info's :tokens)."
  ([m trainer batches] (step m trainer batches nil))
  ([m trainer batches opts]
   (options/check `step opts #{:checkpoint? :chunk :loss})
   (check-trainer trainer)
   (when-not (or (nil? (:loss opts)) (ifn? (:loss opts)))
     (throw (invalid (str ":loss is a function of the adapted model, a batch and the options, not " (pr-str (:loss opts))))))
   (let [batches (if (map? batches) [batches] (vec batches))
         _ (when (empty? batches) (throw (invalid "step takes at least one batch")))
         {:keys [adapter optimizer optimizer-state] n :step} trainer
         [k-steps] (step-keys trainer)
         keys (rng/split (rng/fold-in k-steps n) (count batches))
         loss-opts (merge loss-defaults opts)
         [total grads n-tokens reported] (accumulated m adapter batches keys loss-opts)
         ;; the mean over every trained token of the step: 1/N once
         loss (/ total (double n-tokens))
         grads (array/tree-map #(/ % (double n-tokens)) grads)
         norm (global-norm grads)
         [loss-v norm-v] (mapv (comp double array/item) (array/eval! [loss norm]))]
     (when-not (and (Double/isFinite loss-v) (Double/isFinite norm-v))
       (throw (ex-info (str "sft: step " n "'s " (if (Double/isFinite loss-v) "gradient" "loss") " is not finite (loss "
                            loss-v ", gradient norm " norm-v "): the trainer is not stepped")
                       {:reason :non-finite :step n :loss loss-v :grad-norm norm-v})))
     (let [[params state] (learn/step optimizer optimizer-state (:params adapter) (array/tree-map array/negate grads))]
       [(assoc trainer :adapter (assoc adapter :params params) :optimizer-state state :step (inc n))
        (cond-> {:step n :loss loss-v :tokens n-tokens :grad-norm norm-v}
          (:loss opts) (assoc :reported (vec reported)))]))))

;; ---------------------------------------------------------------------------
;; Checkpoints
;; ---------------------------------------------------------------------------

(defn- step-name [n] (format "step-%08d" (long n)))

(def ^:private step-pattern #"step-(\d{8})")

(defn- write-synced!
  "text written to file f, forced to the disk."
  [^File f ^String text]
  (with-open [out (FileOutputStream. f)]
    (.write out (utf8 text))
    (.sync (.getFD out))))

(defn save-checkpoint
  "Writes trainer under the directory dir (made if missing) as
   dir/step-<n>, n its :step: adapter.gguf (lora/save-gguf's, float32) and
   trainer.edn (gen.serialize's, exact), written beside it and renamed
   into place at once (the namespace's docstring). Loud (:exists) when
   that checkpoint is there already. Returns its path."
  [trainer dir]
  (check-trainer trainer)
  (let [root (io/file (str dir))
        final (io/file root (step-name (:step trainer)))
        tmp (io/file root (str "." (.getName final) ".tmp-" (System/nanoTime)))]
    (.mkdirs root)
    (when (.exists final)
      (throw (ex-info (str "sft: a checkpoint " final " exists already") {:reason :exists :path (str final)})))
    (when-not (.mkdir tmp)
      (throw (ex-info (str "sft: cannot make the directory " tmp) {:reason :io :path (str tmp)})))
    (lora/save-gguf (:adapter trainer) (str (io/file tmp "adapter.gguf")))
    (write-synced! (io/file tmp "trainer.edn") (ser/write trainer))
    (Files/move (.toPath tmp) (.toPath final) (into-array CopyOption [StandardCopyOption/ATOMIC_MOVE]))
    (str final)))

(defn load-checkpoint
  "The trainer a checkpoint (save-checkpoint's directory) holds, exactly as
   it was saved, its arrays on gen.array's *backend*."
  [path]
  (let [f (io/file (str path) "trainer.edn")]
    (when-not (.isFile f)
      (throw (ex-info (str "sft: no checkpoint at " path " (no trainer.edn)") {:reason :no-checkpoint :path (str path)})))
    (let [t (ser/read (slurp f :encoding "UTF-8"))]
      (check-trainer t)
      t)))

(defn latest-checkpoint
  "The path of the complete checkpoint of the highest step under dir, or
   nil when there is none."
  [dir]
  (let [names (keep #(let [nm (.getName ^File %)]
                       (when (and (.isDirectory ^File %) (re-matches step-pattern nm) (.isFile (io/file % "trainer.edn"))) nm))
                    (or (.listFiles (io/file (str dir))) []))]
    (when (seq names)
      (str (io/file (str dir) (last (sort names)))))))

;; ---------------------------------------------------------------------------
;; Training
;; ---------------------------------------------------------------------------

(defn- permutation
  "A uniform permutation of 0 … n-1 from key."
  [key n]
  (->> (array/->clj (rng/uniform key [n]))
       (map-indexed vector)
       (sort-by peek)
       (mapv first)))

(defn order
  "The indices of the examples (of n) that step s takes, size of them:
   positions s·size on of the epochs' orders laid end to end, epoch e's a
   uniform permutation of 0 … n-1 from (fold-in k-order e). So each
   epoch takes every example once, and a step is a function of the key
   and s alone (train's order; gen.llm.grpo's prompts)."
  [k-order n s size]
  (let [start (c/* (long s) (long size))
        end (c/+ start (long size))
        e0 (quot start (long n))
        e1 (quot (dec end) (long n))
        laid (into [] (mapcat #(permutation (rng/fold-in k-order %) n)) (range e0 (inc e1)))]
    (subvec laid (c/- start (c/* e0 (long n))) (c/- end (c/* e0 (long n))))))

(defn train
  "trainer trained on model m's examples (example's) until its :step is
   :steps: each step on :accumulate batches of :batch-size examples, in
   epochs of the examples in an order drawn from the trainer's key (the
   namespace's docstring), so a trainer loaded from a checkpoint goes on
   as the run that saved it would have. Returns {:trainer :history
   :evaluations}: :history each step's info (step's), :evaluations
   [{:step :nll :tokens}] the held-out examples' evaluate at the steps
   it was run. Options:
     :steps        the step to train to (required)
     :batch-size   examples a batch (8)
     :accumulate   batches a step (1)
     :max-length, :pad          batch's
     :checkpoint?, :chunk       step's
     :callback     called after each step with its info and the
                   :trainer; its value is ignored
     :save-dir     where save-checkpoint writes, every :save-every
                   steps (default none) and after the last
     :save-every
     :held-out     examples evaluated (evaluate) before the first step,
                   every :eval-every steps and after the last
     :eval-every"
  [m trainer examples {:keys [steps batch-size accumulate callback save-dir save-every held-out eval-every]
                       :or {batch-size 8 accumulate 1} :as opts}]
  (options/check `train opts #{:steps :batch-size :accumulate :max-length :pad :checkpoint? :chunk :callback
                               :save-dir :save-every :held-out :eval-every})
  (check-trainer trainer)
  (let [examples (vec examples)
        n (count examples)]
    (when (zero? n) (throw (invalid "train takes at least one example")))
    (doseq [[k v] [[:batch-size batch-size] [:accumulate accumulate]]]
      (when-not (pos-int? v) (throw (invalid (str k " is a positive integer, not " (pr-str v)) {k v}))))
    (when-not (and (nat-int? steps) (c/>= (long steps) (long (:step trainer))))
      (throw (invalid (str ":steps is the step to train to, at least the trainer's " (:step trainer) ", not " (pr-str steps))
                      {:steps steps :step (:step trainer)})))
    (doseq [[k v] [[:save-every save-every] [:eval-every eval-every]]]
      (when-not (or (nil? v) (pos-int? v)) (throw (invalid (str k " is a positive integer, not " (pr-str v)) {k v}))))
    (let [batch-opts (select-keys opts [:max-length :pad])
          step-opts (select-keys opts [:checkpoint? :chunk])
          eval-opts (cond-> (assoc batch-opts :batch-size batch-size) (:chunk opts) (assoc :chunk (:chunk opts)))
          [_ k-order] (step-keys trainer)
          evaluation (fn [t] (when (seq held-out)
                               (assoc (evaluate (lora/adapt m (:adapter t)) held-out eval-opts) :step (:step t))))
          size (c/* (long batch-size) (long accumulate))]
      (loop [t trainer
             history []
             evaluations (if-let [e (evaluation t)] [e] [])]
        (if (c/>= (long (:step t)) (long steps))
          {:trainer t :history history :evaluations evaluations}
          (let [idx (order k-order n (:step t) size)
                bs (mapv #(batch (mapv examples %) batch-opts) (partition-all batch-size idx))
                [t info] (step m t bs step-opts)
                done? (c/>= (long (:step t)) (long steps))]
            (when callback (callback (assoc info :trainer t)))
            (when (and save-dir (or done? (and save-every (zero? (rem (long (:step t)) (long save-every))))))
              (save-checkpoint t save-dir))
            (recur t
                   (conj history info)
                   (if-let [e (when (or done? (and eval-every (zero? (rem (long (:step t)) (long eval-every)))))
                                (evaluation t))]
                     (conj evaluations e)
                     evaluations))))))))
