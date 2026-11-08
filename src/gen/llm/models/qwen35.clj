(ns gen.llm.models.qwen35
  "Qwen3.5's and Qwen3.6's forward passes over gen.array, dense
   (architecture qwen35) and mixture-of-experts (qwen35moe), from a GGUF
   file's weights or a Hugging Face checkpoint's
   (docs/research/llm-mlx-engine.md §2, P4 of §6), beside
   gen.llm.models.qwen3 and with its functions:

     (load path)                       → a model {:config :weights} from a GGUF file,
                                         or a checkpoint's directory (load's docstring)
     (load path {:dtype :float16})     → … computing in another float dtype
     (model gguf)                      → the same from gen.array/load-gguf's result
     (config metadata)                 → the architecture from a GGUF's metadata
     (forward m tokens)                → logits [B T vocab] of every position
     (prefill m tokens)                → [logits cache]: the logits, and the cache after the tokens
     (step m cache tokens)             → [logits cache']: the tokens read after cache
     (offset cache)                    → the positions cache holds
     (offsets cache)                   → … in each row
     (static-cache m rows capacity)    → an empty cache of fixed capacity
     (compile-step m)                  → step over static caches, compiled (array/compile),
                                         consuming each cache with :donate? true
     (log-probs m tokens)              → log p [B T-1] of each next token, for training
     (nll m tokens)                    → -Σ log p, supervised fine-tuning's loss
     (select-rows cache idx)           → a static cache of cache's rows idx
     (concat-rows caches)              → a static cache of the caches' rows
     (rewind cache offsets)            → a static cache with rows cleared (back at 0)
     (grow cache capacity)             → a static cache of a larger capacity
     (paged-pools m n-blocks bs)       → the attention layers' paged KV pools, by layer
     (paged-states m rows)             → the GatedDeltaNet layers' states of rows rows
     (paged-step m layers positions tables slots tokens)
                                       → [logits layers']: a token a row, the pools and
                                         the rows' states merged by layer (qwen3's
                                         paged-step for Qwen3.5)
     (compile-paged-step m)            → paged-step compiled (:donate? true)
     (paged-batch-step m layers batch) → {:logits :scores :layers}: decode rows and a prefill
                                         chunk in one forward (qwen3's paged-batch-step for
                                         Qwen3.5: the chunk one sequence through the
                                         GatedDeltaNet layers)
     (compile-paged-batch-step m)      → paged-batch-step compiled (:donate? true)

   tokens, :logits :last, prefill's :capacity and the steps are as
   qwen3's: a row's tokens are positions offset, offset + 1, … after a
   cache of offset positions, and (step m (second (prefill m a)) b) gives
   the logits (prefill m (a ++ b)) gives b's positions, within float
   round-off (exactly on the JVM backend for reads of fewer than 64 tokens:
   test/gen/llm/models/qwen35_test.clj).

   The layers. Most are GatedDeltaNet's linear attention, every
   full_attention_interval-th (4: layers 3, 7, …) gated attention, or as
   attention.recurrent_layers says. The tensors' meaning is llama.cpp's
   (src/models/qwen35.cpp, qwen35moe.cpp and delta-net-base.cpp, v0.9.0),
   which wrote the GGUF, not Hugging Face's: its converter adds Qwen3-Next's
   +1 to the RMSNorm weights (but ssm_norm's), stores A as ssm_a = -exp(A_log)
   and orders the value heads tiled (value head h reads key head h mod
   key-heads); GenMLX's qwen35_forward.cljs gave the structure.
   - Every block: RMSNorm (attn_norm), the layer, the residual; RMSNorm
     (post_attention_norm), the feed-forward, the residual.
   - GatedDeltaNet: the projection attn_qkv of the normed x, convolved
     causally per channel by ssm_conv1d's 4 taps over the layer's last 3
     inputs and this read's (the conv state), through SiLU, split into
     q, k (key heads) and v (value heads); q and k l2-normed per head (eps
     the RMS eps), repeated (tiled) to the value heads, q scaled by
     1/√key-dim; beta = sigmoid(ssm_beta x), the log decay g =
     ssm_a·softplus(ssm_alpha x + ssm_dt.bias); the gated delta rule
     (gen.array/gated-delta) from the layer's recurrent state; its output
     RMSNormed per head (ssm_norm) times silu(attn_gate x), through ssm_out.
   - Gated attention: attn_q's projection holds each head's query and then
     its output gate; q and k RMSNormed per head (attn_q_norm, attn_k_norm),
     RoPE on the first rope.dimension_count dims of a head (NEOX's pairs,
     base rope.freq_base: interleaved M-RoPE, whose sections a text
     token's equal positions make one rotation), attention with grouped
     KV heads, causal, scaled by 1/√head-dim, times sigmoid(gate), through
     attn_output.
   - The feed-forward: SwiGLU (ffn_gate, ffn_up, ffn_down); or (qwen35moe)
     the router's softmax (ffn_gate_inp) over the experts, the
     expert_used_count most likely, their probabilities renormalized to
     sum 1, each one's SwiGLU (ffn_gate_exps, ffn_up_exps, ffn_down_exps:
     stacks gathered by gather-qmm, or by take for dense ones) weighted
     by them, plus the shared expert's SwiGLU (ffn_*_shexp) times
     sigmoid(ffn_gate_inp_shexp · x).
   - RMSNorm (output_norm) and the output projection (output, or the
     embedding: tied).
   The MTP block (nextn_predict_layers, the blocks past the forward's)
   is llama.cpp's draft head for speculative decoding, which its forward
   does not run either: its tensors are recognized and not read. Every
   other tensor of the file must be one of these, and RoPE scaling,
   expert groups and M-RoPE sections that leave a text token's dims
   unrotated are loud (:unsupported), as tensors llama.cpp does not read.

   A cache is a value, {:offset n :layers […]}: an attention layer's {:k
   :v}, keys and values [B KV-heads n head-dim] grown by concatenation; a
   GatedDeltaNet layer's {:conv :state}, its last 3 projected inputs
   [B 3 channels] and its recurrent state [B value-heads value-dim
   key-dim] (float32 for a half-float model), replaced by each read. A
   static cache (static-cache, or prefill's :capacity) is qwen3's
   {:capacity :offsets :positions :layers}: its attention layers' slots
   written in place (put-along-axis) and masked by each row's position,
   its GatedDeltaNet layers' states replaced, so a step's shapes stay
   fixed and compile-step compiles it as qwen3's (on MLX one trace per
   shape of tokens, :donate? consuming the cache). A recurrent state
   cannot be moved back: rewind takes a row to its own offset or to 0 (a
   slot reused), and anything else is loud (:unsupported); keep the cache
   from before (a value) instead, as speculative decoding's rollback.

   A projection adapted by gen.llm.lora/adapt is {:base w :lora …}, its
   product w's plus a LoRA adapter's term (gen.llm.lora/targets' qwen35
   and qwen35moe: the attention's, GatedDeltaNet's input and output
   projections, the feed-forward's, the shared expert's, or the routed
   experts' stacks, whose terms the tokens routed to each take): every
   function here takes the adapted model, and log-probs and nll
   differentiate it with respect to the adapter.

   The forward is gen.array code: on MLX its rms-norm, rope and sdpa are
   MLX's fused kernels and gated-delta mlx-lm's Metal kernel (reads of
   any length), elsewhere gated-delta's composite (token by token below 64
   tokens, in chunks of 64 from there); the tape (or MLX's native
   gradients) differentiates it, through the composite; on the JVM
   backend tiny models run in float64. Experimental: outside the 1.x
   promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= == load])
  (:require [clojure.core :as c]
            [clojure.string :as str]
            [gen.array :as array :refer [+ * /]]
            [gen.array.paged :as paged]
            [gen.llm.lora :as lora]
            [gen.llm.models :as models]
            [gen.llm.models.checkpoint :as checkpoint]
            [gen.llm.models.common :as common]))

(set! *warn-on-reflection* true)

(def ^:private who "qwen35")

(defn- unsupported
  ([message] (common/unsupported who message))
  ([message data] (common/unsupported who message data)))

(defn- invalid
  ([message] (common/invalid who message))
  ([message data] (common/invalid who message data)))

;; ---------------------------------------------------------------------------
;; The configuration, from a GGUF's metadata
;; ---------------------------------------------------------------------------

(defn- rope-covered?
  "Does interleaved M-RoPE with these sections ([t h w e], as ggml's
   imrope) rotate each of the dims/2 pairs of a text token by its
   position? Pair i's sector is i mod Σsections, and sector s reads the
   t, h or w position (each a text token's own) when s mod 3 is 0, 1 or
   2 and s < 3·(that section), else e's, 0 for text: no rotation."
  [[t h w :as sections] dims]
  (let [n (long (reduce c/+ 0 sections))]
    (and (c/== 4 (count sections)) (pos? n)
         (every? (fn [i]
                   (let [s (mod (long i) n)]
                     (case (int (mod s 3))
                       0 (c/< s (c/* 3 (long t)))
                       1 (c/< s (c/* 3 (long h)))
                       2 (c/< s (c/* 3 (long w))))))
                 (range (quot (long dims) 2))))))

(defn config
  "Qwen3.5's architecture from a GGUF's metadata (gen.array/load-gguf's
   :metadata), llama.cpp's keys: {:arch :hidden :layers :kinds :mtp-layers
   :heads :kv-heads :head-dim :rope-dims :rope-base :eps :context
   :delta-key-heads :delta-value-heads :delta-key-dim :delta-value-dim
   :delta-conv}, and :ffn for qwen35, or :experts :experts-used
   :expert-ffn :shared-ffn for qwen35moe. :layers counts the blocks the
   forward runs (block_count less nextn_predict_layers, the MTP block),
   :kinds is each one's :delta (GatedDeltaNet) or :attention. Loud
   (:unsupported) for another architecture, RoPE scaling, expert groups,
   keys and values of different sizes, and M-RoPE sections that leave a
   text token's dims unrotated, which llama.cpp's graphs do not compute
   either."
  [metadata]
  (let [arch (get metadata "general.architecture")
        k (fn [suffix] (get metadata (str arch "." suffix)))
        need (fn [suffix]
               (let [v (k suffix)]
                 (when-not (and (integer? v) (pos? (long v)))
                   (throw (invalid (str "the metadata's " arch "." suffix " must be a positive integer, not " (pr-str v))
                                   {:key (str arch "." suffix)})))
                 (long v)))]
    (when-not (#{"qwen35" "qwen35moe"} arch)
      (throw (unsupported (str "the model's architecture is " (pr-str arch) ", not qwen35 or qwen35moe") {:architecture arch})))
    (let [moe? (= "qwen35moe" arch)
          blocks (need "block_count")
          mtp (long (or (k "nextn_predict_layers") 0))
          layers (c/- blocks mtp)
          heads (need "attention.head_count")
          head-dim (need "attention.key_length")
          rope-dims (if (k "rope.dimension_count") (need "rope.dimension_count") head-dim)
          sections (k "rope.dimension_sections")
          interval (if (k "full_attention_interval") (need "full_attention_interval") 4)
          recurrent (k "attention.recurrent_layers")
          kinds (if recurrent
                  (mapv #(if % :delta :attention) (take layers recurrent))
                  (mapv #(if (zero? (rem (inc (long %)) interval)) :attention :delta) (range layers)))
          v-heads (need "ssm.time_step_rank")
          k-heads (need "ssm.group_count")
          key-dim (need "ssm.state_size")
          inner (need "ssm.inner_size")
          scaling (k "rope.scaling.type")
          ffn (k "feed_forward_length")
          cfg (merge {:arch arch
                      :hidden (need "embedding_length")
                      :layers layers
                      :kinds kinds
                      :mtp-layers mtp
                      :heads heads
                      :kv-heads (need "attention.head_count_kv")
                      :head-dim head-dim
                      :rope-dims rope-dims
                      :rope-base (double (or (k "rope.freq_base") 10000.0))
                      :eps (double (or (k "attention.layer_norm_rms_epsilon") 1e-6))
                      :context (k "context_length")
                      :delta-key-heads k-heads
                      :delta-value-heads v-heads
                      :delta-key-dim key-dim
                      :delta-value-dim (quot inner v-heads)
                      :delta-conv (need "ssm.conv_kernel")}
                     (if moe?
                       (let [used (need "expert_used_count")]
                         {:experts (need "expert_count")
                          :experts-used used
                          :expert-ffn (if (k "expert_feed_forward_length")
                                        (need "expert_feed_forward_length")
                                        (quot (need "feed_forward_length") used))
                          :shared-ffn (if (k "expert_shared_feed_forward_length")
                                        (need "expert_shared_feed_forward_length")
                                        (need "feed_forward_length"))})
                       {:ffn (need "feed_forward_length")}))]
      (when (c/< layers 1)
        (throw (invalid "no blocks for the forward (block_count less nextn_predict_layers)" {:blocks blocks :mtp mtp})))
      (when (and recurrent (not (and (sequential? recurrent) (c/>= (count recurrent) layers) (every? boolean? recurrent))))
        (throw (invalid (str arch ".attention.recurrent_layers must be a boolean a block, not " (pr-str recurrent)))))
      (when (and (k "attention.value_length") (not= head-dim (need "attention.value_length")))
        (throw (unsupported "keys and values of different sizes" {:key-length head-dim :value-length (k "attention.value_length")})))
      (when-not (and (even? rope-dims) (c/<= rope-dims head-dim))
        (throw (invalid (str "RoPE over " rope-dims " dims of a head of " head-dim) {:rope-dims rope-dims :head-dim head-dim})))
      (when-not (and (sequential? sections) (every? integer? sections) (rope-covered? (vec sections) rope-dims))
        (throw (unsupported (str "M-RoPE sections " (pr-str sections) " that leave some of a text token's "
                                 rope-dims " RoPE dims unrotated (or none given)")
                            {:rope-sections sections :rope-dims rope-dims})))
      (when (and scaling (not= "none" scaling))
        (throw (unsupported (str "RoPE scaling (" scaling ")") {:rope-scaling scaling})))
      (when (and moe? (c/> (long (or (k "expert_group_count") 0)) 1))
        (throw (unsupported "expert groups" {:expert-groups (k "expert_group_count")})))
      (when (and moe? (c/> (long (:experts-used cfg)) (long (:experts cfg))))
        (throw (invalid "more experts used than there are" (select-keys cfg [:experts :experts-used]))))
      (when-not (zero? (rem (long heads) (long (:kv-heads cfg))))
        (throw (invalid "the query heads are not a multiple of the KV heads" (select-keys cfg [:heads :kv-heads]))))
      (when-not (and (zero? (rem v-heads k-heads)) (c/== inner (c/* v-heads key-dim)))
        (throw (invalid (str "GatedDeltaNet's value heads (" v-heads ") must be a multiple of its key heads (" k-heads
                             ") and its inner size (" inner ") value heads of the key size (" key-dim ")")
                        {:value-heads v-heads :key-heads k-heads :inner inner :key-dim key-dim})))
      (when (and ffn (not moe?) (not (integer? ffn)))
        (throw (invalid (str "the metadata's " arch ".feed_forward_length must be an integer"))))
      cfg)))

;; ---------------------------------------------------------------------------
;; The weights, by llama.cpp's names
;; ---------------------------------------------------------------------------

(defn- mtp-tensor?
  "Is name a tensor of the MTP blocks (blk.i for i at or past layers)?"
  [name layers]
  (when-let [[_ i] (re-matches #"blk\.(\d+)\..*" name)]
    (c/>= (Long/parseLong i) (long layers))))

(defn model
  "A model {:config :weights} from gen.array/load-gguf's result: config
   from its metadata (with :vocab, the embedding's rows, and :dtype, the
   dtype it computes in), and the weights by llama.cpp's names, each
   checked against the configuration's shape. Options: :dtype, a float
   dtype (default: the backend's default float), to which the float
   weights are cast, the quantized matrices' scales and biases among them.
   Loud when a tensor the forward reads is missing or of another shape
   (:invalid-args), and when the file holds one it does not read but the
   MTP block's (:unsupported).

   The weights: {:embed :norm :output :layers}, a layer's {:attn-norm
   :post-norm}, and a GatedDeltaNet layer's {:qkv :z :conv :dt-bias :a
   :beta :alpha :ssm-norm :out} or an attention layer's {:q :k :v :q-norm
   :k-norm :o}, and its feed-forward's {:gate :up :down} (dense) or {:moe
   {:router :gate :up :down :shared-gate :shared {:gate :up :down}}},
   the experts' a stack [experts out in]. A matrix is an array [out in],
   or a quantized one as gen.array/load-gguf gives Q8_0, Q4_0, Q4_1 and
   K-quant tensors, {:wq :scales :biases :bits :group-size} (each tensor
   its own bits; Q6_K's load dense); the conv kernel,
   ssm_a, ssm_dt.bias, the norms and the shared expert's gate are dense
   arrays."
  ([gguf] (model gguf nil))
  ([{:keys [metadata tensors]} {:keys [dtype] :as opts}]
   (common/check-options who :model opts #{:dtype})
   (let [cfg (config metadata)
         {:keys [hidden layers kinds heads kv-heads head-dim ffn delta-key-heads delta-value-heads delta-key-dim
                 delta-value-dim delta-conv experts expert-ffn shared-ffn]} cfg
         dtype (common/check-dtype who (or dtype (:float (array/defaults array/*backend*))))
         seen (atom #{})
         tensor (fn tensor
                  ([name shape] (tensor name shape false))
                  ([name shape dense?]
                   (let [w (or (get tensors name)
                               (throw (invalid (str "the file has no tensor " name) {:tensor name})))
                         s (common/weight-shape w)]
                     (when (and shape (not= shape s))
                       (throw (invalid (str "tensor " name " has shape " s ", not " shape) {:tensor name :shape s :expected shape})))
                     (when (and dense? (map? w))
                       (throw (unsupported (str "tensor " name " is quantized; the forward reads it dense") {:tensor name})))
                     (swap! seen conj name)
                     (common/cast-weight who w dtype))))
         embed (tensor "token_embd.weight" nil)
         vocab (first (common/weight-shape embed))
         _ (when-not (= [vocab hidden] (common/weight-shape embed))
             (throw (invalid (str "the embedding has shape " (common/weight-shape embed) ", not [vocab " hidden "]")
                             {:tensor "token_embd.weight"})))
         key-width (c/* (long delta-key-heads) (long delta-key-dim))
         value-width (c/* (long delta-value-heads) (long delta-value-dim))
         channels (c/+ (c/* 2 key-width) value-width)
         layer (fn [i]
                 (let [t (fn ([nm shape] (tensor (str "blk." i "." nm ".weight") shape))
                           ([nm shape dense?] (tensor (str "blk." i "." nm ".weight") shape dense?)))
                       mixer (if (= :attention (kinds i))
                               {:q (t "attn_q" [(c/* 2 heads head-dim) hidden])
                                :k (t "attn_k" [(c/* kv-heads head-dim) hidden])
                                :v (t "attn_v" [(c/* kv-heads head-dim) hidden])
                                :q-norm (t "attn_q_norm" [head-dim] true)
                                :k-norm (t "attn_k_norm" [head-dim] true)
                                :o (t "attn_output" [hidden (c/* heads head-dim)])}
                               {:qkv (t "attn_qkv" [channels hidden])
                                :z (t "attn_gate" [value-width hidden])
                                :conv (t "ssm_conv1d" [channels delta-conv] true)
                                :dt-bias (tensor (str "blk." i ".ssm_dt.bias") [delta-value-heads] true)
                                :a (tensor (str "blk." i ".ssm_a") [delta-value-heads] true)
                                :beta (t "ssm_beta" [delta-value-heads hidden])
                                :alpha (t "ssm_alpha" [delta-value-heads hidden])
                                :ssm-norm (t "ssm_norm" [delta-value-dim] true)
                                :out (t "ssm_out" [hidden value-width])})
                       feed (if experts
                              {:moe (merge {:router (t "ffn_gate_inp" [experts hidden])
                                            :gate (t "ffn_gate_exps" [experts expert-ffn hidden])
                                            :up (t "ffn_up_exps" [experts expert-ffn hidden])
                                            :down (t "ffn_down_exps" [experts hidden expert-ffn])}
                                           (when (contains? tensors (str "blk." i ".ffn_up_shexp.weight"))
                                             {:shared-gate (t "ffn_gate_inp_shexp" [hidden] true)
                                              :shared {:gate (t "ffn_gate_shexp" [shared-ffn hidden])
                                                       :up (t "ffn_up_shexp" [shared-ffn hidden])
                                                       :down (t "ffn_down_shexp" [hidden shared-ffn])}}))}
                              {:gate (t "ffn_gate" [ffn hidden])
                               :up (t "ffn_up" [ffn hidden])
                               :down (t "ffn_down" [hidden ffn])})]
                   (merge {:attn-norm (t "attn_norm" [hidden] true)
                           :post-norm (t "post_attention_norm" [hidden] true)}
                          mixer feed)))
         weights {:embed embed
                  :norm (tensor "output_norm.weight" [hidden] true)
                  :output (when (contains? tensors "output.weight") (tensor "output.weight" [vocab hidden]))
                  :layers (mapv layer (range layers))}
         unread (sort (remove #(or (@seen %) (mtp-tensor? % layers)) (keys tensors)))]
     (when (seq unread)
       (throw (unsupported (str "the file holds tensors Qwen3.5's forward does not read: " (str/join ", " (take 8 unread))
                                (when (c/> (count unread) 8) ", …"))
                           {:tensors (vec unread)})))
     {:config (assoc cfg :vocab vocab :dtype dtype) :weights weights})))

;; ---------------------------------------------------------------------------
;; Hugging Face checkpoints, read as llama.cpp's converter writes their GGUF
;; ---------------------------------------------------------------------------

(def ^:private skipped
  "The prefixes of a checkpoint's tensors the text model never reads: the
   vision tower (Hugging Face's, mlx-vlm's) and the MTP block."
  ["model.visual." "visual." "vision_tower." "model.vision_tower." "mtp." "model.mtp."])

(def ^:private prefixes
  "The text model's prefixes: Qwen3_5ForConditionalGeneration's, mlx-lm's
   saved layout's, a causal LM's."
  ["model.language_model." "language_model.model." "model."])

(defn- skipped? [name] (some #(str/starts-with? name %) skipped))

(defn- strip
  "A tensor's name after its text model's prefix (\"layers.0.…\",
   \"embed_tokens.weight\"), lm_head's \"lm_head.weight\", or the name
   itself."
  [name]
  (if (#{"lm_head.weight" "language_model.lm_head.weight"} name)
    "lm_head.weight"
    (or (some #(when (str/starts-with? name %) (subs name (count %))) prefixes) name)))

(defn- text-config
  "config.json's text model's: its text_config, or itself."
  [c]
  (let [tc (get c "text_config" c)]
    (when-not (map? tc) (throw (invalid "config.json's text_config is not an object")))
    tc))

(defn- hf-metadata
  "The metadata llama.cpp's converter writes for a Qwen3.5 or Qwen3.5-MoE
   checkpoint's config.json (conversion/qwen.py's Qwen3_5TextModel and
   Qwen3NextModel), floats as the GGUF's float32, with its defaults: RoPE
   over a quarter of a head (partial_rotary_factor 0.25), M-RoPE's
   sections [11 11 10 0], a full attention layer every 4th; layer_types
   as the recurrent layers, the MTP block's not. Loud (:unsupported) for
   another model_type, an activation but SiLU, attention without its
   output gate, M-RoPE not interleaved, dense layers in a mixture of
   experts or its top weights not renormalized, and RoPE scaling (which
   config refuses once it is metadata)."
  [c]
  (let [tc (text-config c)
        k #(get tc %)
        types #{"qwen3_5" "qwen3_5_moe" "qwen3_5_text" "qwen3_5_moe_text"}
        model-type (get c "model_type")
        moe? (some? (k "num_experts"))
        arch (if moe? "qwen35moe" "qwen35")
        a #(str arch "." %)
        rope (or (k "rope_parameters") (k "rope_scaling") {})
        rope-type (or (get rope "rope_type") (get rope "type"))
        theta (or (get rope "rope_theta") (k "rope_theta"))
        head-dim (k "head_dim")
        rope-head (or head-dim (when (and (integer? (k "hidden_size")) (integer? (k "num_attention_heads")))
                                 (quot (long (k "hidden_size")) (long (k "num_attention_heads")))))
        factor (or (get rope "partial_rotary_factor") (k "partial_rotary_factor") 0.25)
        sections (get rope "mrope_section" [11 11 10 0])
        mtp (long (or (k "mtp_num_hidden_layers") 0))
        layer-types (k "layer_types")]
    (when-not (and (types model-type) (or (identical? c tc) (types (k "model_type"))))
      (throw (unsupported (str "the checkpoint's model_type is " (pr-str model-type) ", not one of " (vec (sort types)))
                          {:model-type model-type})))
    (when-not (contains? #{nil "silu"} (k "hidden_act"))
      (throw (unsupported (str "the activation " (pr-str (k "hidden_act")) " (Qwen3.5's is SiLU)") {:hidden-act (k "hidden_act")})))
    (when (false? (k "attn_output_gate"))
      (throw (unsupported "attention without its output gate")))
    (when (and sections (false? (get rope "mrope_interleaved")))
      (throw (unsupported "M-RoPE not interleaved (llama.cpp's qwen35 computes the interleaved one)")))
    (when (and moe? (or (seq (k "mlp_only_layers")) (not (contains? #{nil 1} (k "decoder_sparse_step")))
                        (false? (k "norm_topk_prob"))))
      (throw (unsupported (str "a mixture of experts with dense layers among its own (mlp_only_layers, decoder_sparse_step)"
                               " or its top experts' weights not renormalized (norm_topk_prob)"))))
    (when (and layer-types (not (and (sequential? layer-types)
                                      (every? #{"linear_attention" "full_attention"} layer-types))))
      (throw (invalid (str "config.json's layer_types are not linear_attention or full_attention: " (pr-str layer-types)))))
    (cond-> {"general.architecture" arch
             (a "context_length") (k "max_position_embeddings")
             (a "embedding_length") (k "hidden_size")
             (a "block_count") (when (integer? (k "num_hidden_layers")) (c/+ (long (k "num_hidden_layers")) mtp))
             (a "attention.head_count") (k "num_attention_heads")
             (a "attention.head_count_kv") (k "num_key_value_heads")
             (a "attention.key_length") head-dim
             (a "attention.value_length") head-dim
             (a "attention.layer_norm_rms_epsilon") (checkpoint/float32 (k "rms_norm_eps"))
             (a "rope.dimension_count") (when (integer? rope-head) (long (c/* (double rope-head) (double factor))))
             (a "rope.dimension_sections") (when (sequential? sections) (vec (take 4 (concat sections (repeat 0)))))
             (a "full_attention_interval") (or (k "full_attention_interval") 4)
             (a "ssm.conv_kernel") (k "linear_conv_kernel_dim")
             (a "ssm.state_size") (k "linear_key_head_dim")
             (a "ssm.group_count") (k "linear_num_key_heads")
             (a "ssm.time_step_rank") (k "linear_num_value_heads")
             (a "ssm.inner_size") (when (and (integer? (k "linear_value_head_dim")) (integer? (k "linear_num_value_heads")))
                                    (c/* (long (k "linear_value_head_dim")) (long (k "linear_num_value_heads"))))}
      theta (assoc (a "rope.freq_base") (checkpoint/float32 theta))
      (k "intermediate_size") (assoc (a "feed_forward_length") (k "intermediate_size"))
      (pos? mtp) (assoc (a "nextn_predict_layers") mtp)
      layer-types (assoc (a "attention.recurrent_layers") (into (mapv #(= "linear_attention" %) layer-types)
                                                                (repeat mtp false)))
      (not (contains? #{nil "default"} rope-type)) (assoc (a "rope.scaling.type") rope-type)
      moe? (assoc (a "expert_count") (k "num_experts")
                  (a "expert_used_count") (k "num_experts_per_tok")
                  (a "expert_feed_forward_length") (k "moe_intermediate_size")
                  (a "expert_shared_feed_forward_length") (k "shared_expert_intermediate_size")))))

(def ^:private block-names
  "A block's tensors (Hugging Face's names, after layers.i.) by llama.cpp's
   (after blk.i.), and what the converter does to each: :norm (Qwen3-Next's
   +1, in a Hugging Face checkpoint), :qkv (the value rows' heads to tiled
   order), [:rows d] and [:columns d] (the value heads, d values each, to
   tiled order along the rows or the columns), :a-log (-exp, the heads
   tiled), :conv (the kernel squeezed to [C K], its value channels tiled),
   :flat (to a vector)."
  {"input_layernorm.weight" ["attn_norm.weight" :norm]
   "post_attention_layernorm.weight" ["post_attention_norm.weight" :norm]
   "self_attn.q_proj.weight" ["attn_q.weight"]
   "self_attn.k_proj.weight" ["attn_k.weight"]
   "self_attn.v_proj.weight" ["attn_v.weight"]
   "self_attn.o_proj.weight" ["attn_output.weight"]
   "self_attn.q_norm.weight" ["attn_q_norm.weight" :norm]
   "self_attn.k_norm.weight" ["attn_k_norm.weight" :norm]
   "linear_attn.in_proj_qkv.weight" ["attn_qkv.weight" :qkv]
   "linear_attn.in_proj_z.weight" ["attn_gate.weight" [:rows :value-dim]]
   "linear_attn.in_proj_b.weight" ["ssm_beta.weight" [:rows 1]]
   "linear_attn.in_proj_a.weight" ["ssm_alpha.weight" [:rows 1]]
   "linear_attn.A_log" ["ssm_a" :a-log]
   "linear_attn.dt_bias" ["ssm_dt.bias" [:rows 1]]
   "linear_attn.conv1d.weight" ["ssm_conv1d.weight" :conv]
   "linear_attn.norm.weight" ["ssm_norm.weight"]
   "linear_attn.out_proj.weight" ["ssm_out.weight" [:columns :value-dim]]
   "mlp.gate_proj.weight" ["ffn_gate.weight"]
   "mlp.up_proj.weight" ["ffn_up.weight"]
   "mlp.down_proj.weight" ["ffn_down.weight"]
   "mlp.gate.weight" ["ffn_gate_inp.weight"]
   "mlp.switch_mlp.gate_proj.weight" ["ffn_gate_exps.weight"]
   "mlp.switch_mlp.up_proj.weight" ["ffn_up_exps.weight"]
   "mlp.switch_mlp.down_proj.weight" ["ffn_down_exps.weight"]
   "mlp.shared_expert.gate_proj.weight" ["ffn_gate_shexp.weight"]
   "mlp.shared_expert.up_proj.weight" ["ffn_up_shexp.weight"]
   "mlp.shared_expert.down_proj.weight" ["ffn_down_shexp.weight"]
   "mlp.shared_expert_gate.weight" ["ffn_gate_inp_shexp.weight" :flat]})

(defn- convention
  "Whose conventions the checkpoint's tensors (names stripped) follow, by
   its conv kernels' layout, as mlx-lm's sanitize tells them: :hf
   (PyTorch's Conv1d, [C 1 K]: RMSNorm weights w of 1 + w) or :mlx (MLX's
   [C K 1]: mlx-lm's saved layout, the +1 in them already)."
  [tensors]
  (let [layouts (set (for [[name w] tensors
                           :when (re-matches #"layers\.\d+\.linear_attn\.conv1d\.weight" name)
                           :let [s (common/weight-shape w)]]
                       (cond (and (c/== 3 (count s)) (c/== 1 (long (s 1)))) :hf
                             (and (c/== 3 (count s)) (c/== 1 (long (s 2)))) :mlx
                             :else (throw (invalid (str "the conv kernel " name " is " s ", neither [C 1 K] nor [C K 1]")
                                                   {:tensor name :shape s})))))]
    (when-not (c/== 1 (count layouts))
      (throw (invalid (if (seq layouts) "conv kernels of both layouts, [C 1 K] and [C K 1]" "the checkpoint has no conv kernels"))))
    (first layouts)))

(defn- converted
  "The weight w (Hugging Face's, or mlx-lm's) as the GGUF holds it, by the
   converter's transform op (block-names'), the value heads tiled for k
   key heads of r value heads each, d values a head."
  [op w {:keys [k r d key-width hf?]} name]
  (let [dense #(checkpoint/dense who name %)
        tiled-from (fn [w from size]
                     (checkpoint/concat-rows
                      [(checkpoint/rows w 0 from)
                       (checkpoint/tiled who (checkpoint/rows w from (first (common/weight-shape w))) 0 k r size)]))]
    (cond
      (nil? op) w
      (= :norm op) (if hf? (checkpoint/shift-norm (dense w)) (dense w))
      (= :qkv op) (tiled-from w (c/* 2 (long key-width)) d)
      (= :a-log op) (checkpoint/tiled who (checkpoint/neg-exp (dense w)) 0 k r 1)
      (= :conv op) (tiled-from (array/reshape (dense w) [(first (array/shape w)) -1]) (c/* 2 (long key-width)) d)
      (= :flat op) (array/reshape (dense w) [-1])
      :else (let [[axis size] op]
              (checkpoint/tiled who w ({:rows 0 :columns 1} axis) k r (if (= :value-dim size) d size))))))

(defn- experts
  "tensors (names stripped) with each layer's experts' matrices
   (layers.i.mlp.experts.e.gate_proj.weight, …: Hugging Face's, one an
   expert) stacked [experts out in] under mlx-lm's switch_mlp names, as
   llama.cpp's converter stacks them. Loud (:unsupported) for the fused
   gate_up_proj form, a layout no checkpoint at hand shows; an expert
   missing is loud (:invalid-args)."
  [tensors n]
  (let [pattern #"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        groups (dissoc (group-by (fn [[name]] (when-let [[_ i _ proj] (re-matches pattern name)] [i proj])) tensors) nil)]
    (when-let [fused (first (filter #(re-matches #"layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)(\.weight)?" %)
                                    (keys tensors)))]
      (throw (unsupported (str "the experts' fused matrices (" fused "): a layout not read; one matrix an expert, or"
                               " mlx-lm's switch_mlp stacks, are")
                          {:tensor fused})))
    (reduce-kv (fn [ts [i proj] entries]
                 (let [by-expert (into {} (map (fn [[name w]] [(Long/parseLong (nth (re-matches pattern name) 2)) w])) entries)]
                   (when-not (= (set (keys by-expert)) (set (range (long n))))
                     (throw (invalid (str "layer " i "'s " proj " has experts " (vec (sort (keys by-expert))) ", not 0 to " (dec (long n)))
                                     {:layer i})))
                   (-> (apply dissoc ts (map first entries))
                       (assoc (str "layers." i ".mlp.switch_mlp." proj ".weight")
                              (checkpoint/stack who (mapv by-expert (range (long n))))))))
               tensors groups)))

(defn- gguf-name
  "[llama.cpp's name, the converter's transform] of a checkpoint's tensor
   (its name stripped), or [the name] for one the GGUF does not have
   (which model then refuses, naming it)."
  [name]
  (or (when-let [[_ i rest] (re-matches #"layers\.(\d+)\.(.+)" name)]
        (when-let [[g op] (block-names rest)] [(str "blk." i "." g) op]))
      ({"embed_tokens.weight" ["token_embd.weight"] "norm.weight" ["output_norm.weight" :norm]
        "lm_head.weight" ["output.weight"]} name)
      [name]))

(defn- checkpoint-gguf
  "The GGUF llama.cpp's converter writes from the Qwen3.5 or Qwen3.5-MoE
   checkpoint at path (its config.json checked first; the vision tower and
   the MTP block not read), as gen.array/load-gguf gives one,
   {:metadata :tensors :types :files}."
  [path]
  (let [config (checkpoint/config who path)
        metadata (hf-metadata config)
        {:keys [tensors types files]} (checkpoint/read who path config (complement skipped?))
        tc (text-config config)
        stripped (into {} (map (fn [[name w]] [(strip name) w])) tensors)
        stripped (if (get tc "num_experts") (experts stripped (get tc "num_experts")) stripped)
        n-k (long (or (get tc "linear_num_key_heads") 1))
        layout {:k n-k
                :r (quot (long (or (get tc "linear_num_value_heads") n-k)) n-k)
                :d (get tc "linear_value_head_dim")
                :key-width (c/* n-k (long (or (get tc "linear_key_head_dim") 0)))
                :hf? (= :hf (convention stripped))}
        renamed (checkpoint/output who
                                   (into {} (map (fn [[name w]]
                                                   (let [[g op] (gguf-name name)]
                                                     [g (converted op w layout name)])))
                                         stripped)
                                   (get tc "tie_word_embeddings" (get config "tie_word_embeddings"))
                                   (get tc "vocab_size"))
        ;; a tensor's dtype in the file (a stack of experts', its first's)
        type-of (fn [name] (first (gguf-name (strip (str/replace name #"\.experts\.0\." ".switch_mlp.")))))]
    {:metadata metadata
     :tensors renamed
     :types (select-keys (into {} (map (fn [[name t]] [(type-of name) t])) types) (keys renamed))
     :files files}))

(defn load
  "A Qwen3.5 or Qwen3.6 model (dense or MoE) from path, on *backend*:
   (model (gen.array/load-gguf path) opts) for a GGUF file; for a Hugging
   Face checkpoint (its directory, or its .safetensors file or index), its
   tensors as the GGUF llama.cpp's converter writes from it would give
   them, so that a checkpoint and its GGUF give one model:
   - config.json's text model (its text_config) gives the metadata;
   - the tensors are renamed (model.language_model.layers.i.linear_attn.
     in_proj_qkv.weight is blk.i.attn_qkv.weight, …), under the prefixes
     of Qwen3_5ForConditionalGeneration, of a causal LM (model.) and of
     mlx-lm's saved layout (language_model.model.); the vision tower and
     the MTP block are never read;
   - the converter's transforms are done: the RMSNorm weights but the
     gated norm's get Qwen3-Next's +1, in float32; ssm_a is -exp(A_log),
     in float32; the conv kernel is squeezed to [C K]; the value heads go
     from grouped (Hugging Face's: value head h reads key head h / (value
     heads / key heads)) to tiled order in attn_qkv's value rows,
     attn_gate, ssm_beta, ssm_alpha, ssm_a, ssm_dt.bias, the conv kernel's
     value channels and ssm_out's columns (the identity for as many value
     heads as key heads, Qwen3.5-0.8B's); the experts' matrices, one an
     expert, are stacked;
   - mlx-lm's saved checkpoints (mlx-community's), told by their conv
     kernels' layout [C K 1], hold the +1 already; their matrices load
     quantized, by config.json's quantization (gen.llm.models.checkpoint).
   A checkpoint of another model_type is loud (:unsupported), as are the
   other configurations hf-metadata and config refuse, the experts' fused
   gate_up_proj, and a tensor the forward does not read."
  ([path] (load path nil))
  ([path opts] (model (if (checkpoint/checkpoint? path) (checkpoint-gguf path) (array/load-gguf path)) opts)))

;; ---------------------------------------------------------------------------
;; The layers
;; ---------------------------------------------------------------------------

(defn- state-dtype
  "The dtype of a GatedDeltaNet's recurrent state for a model computing
   in dtype: float32 for the half floats, where the rule is computed."
  [dtype]
  (if (#{:float16 :bfloat16} dtype) :float32 dtype))

(defn- causal-conv
  "The causal depthwise convolution of padded [B K-1+t C], the layer's
   last K-1 inputs then t new ones, by kernel [C K]: the output [B t C],
   channel c's at i Σⱼ padded[i + j, c]·kernel[c, j] (ggml_ssm_conv: tap
   j reads the window oldest first)."
  [padded kernel t]
  (let [[ch kk] (array/shape kernel)]
    (reduce + (for [j (range kk)]
                (* (array/slice padded 1 j (c/+ (long j) (long t)))
                   (array/reshape (array/slice kernel 1 j (inc (long j))) [ch]))))))

(defn- conv-state
  "The conv state after a read: the newest K-1 inputs of padded [B K-1+t
   C], the next read's window's start."
  [padded t]
  (let [n (long (second (array/shape padded)))]
    (array/slice padded 1 t n)))

(defn- l2-norm
  "x divided by √(Σ x² + eps) over its last axis of n, as llama.cpp's
   build_gdn_l2_norm: an RMSNorm of eps/n, over √n."
  [x eps]
  (let [n (double (peek (array/shape x)))]
    (* (array/rms-norm x nil (c// (double eps) n)) (c// 1.0 (Math/sqrt n)))))

(defn- key-heads-tiled
  "y [B T key-heads D] repeated to value-heads heads, tiled: value head h
   reads key head h mod key-heads (the order llama.cpp's converter gives
   the value heads, ggml_repeat's)."
  [y value-heads]
  (let [hk (long (nth (array/shape y) 2))]
    (if (c/== hk (long value-heads)) y (array/concatenate (vec (repeat (quot (long value-heads) hk) y)) 2))))

(defn- delta-rule
  "A GatedDeltaNet layer's sequence part: the projected inputs qkv [B T
   channels], convolved after the entry's conv state, and the gated delta
   rule over them from its recurrent state (entry {:conv :state}, nil
   for zeros, the start), with beta and the log decay g [B T value-heads]:
   [y [B T value-heads value-dim], the entry after them]. The conv state
   after is the newest K-1 inputs, or with end (an int array [1], the
   tokens of the T that are a sequence's, the rest padding) those up to
   its end."
  [{:keys [eps dtype delta-conv delta-key-heads delta-value-heads delta-key-dim delta-value-dim]} w qkv beta g entry end]
  (let [[b t channels] (array/shape qkv)
        hk (long delta-key-heads) hv (long delta-value-heads) dk (long delta-key-dim) dv (long delta-value-dim)
        width (c/* hk dk)
        kk (dec (long delta-conv))
        padded (array/concatenate [(or (:conv entry) (array/zeros [b kk channels] {:dtype (array/dtype qkv)})) qkv] 1)
        mixed (common/silu (causal-conv padded (:conv w) t))
        heads (fn [from to n d] (array/reshape (array/slice mixed 2 from to) [b t n d]))
        q (* (key-heads-tiled (l2-norm (heads 0 width hk dk) eps) hv) (c// 1.0 (Math/sqrt dk)))
        k (key-heads-tiled (l2-norm (heads width (c/* 2 width) hk dk) eps) hv)
        v (heads (c/* 2 width) channels hv dv)
        state (or (:state entry) (array/zeros [b hv dv dk] {:dtype (state-dtype dtype)}))
        [y state] (array/gated-delta q k v g beta state)]
    [(if (= dtype (array/dtype y)) y (array/astype y dtype))
     {:conv (if end
              (array/take padded (+ (array/arange 0 kk 1 {:dtype :int32}) end) 1)
              (conv-state padded t))
      :state state}]))

(defn- delta-gates
  "[beta g] of a GatedDeltaNet layer's normed x [B T hidden]: the write
   strengths sigmoid(ssm_beta x) and the log decays
   ssm_a·softplus(ssm_alpha x + ssm_dt.bias), [B T value-heads] each."
  [w x]
  [(array/sigmoid (common/linear x (:beta w)))
   (* (array/softplus (+ (common/linear x (:alpha w)) (:dt-bias w))) (:a w))])

(defn- delta-out
  "A GatedDeltaNet layer's output of the normed x [B T hidden] and its
   rule's y [B T value-heads value-dim]: y RMSNormed per head times
   silu(attn_gate x), through ssm_out."
  [{:keys [eps delta-value-heads delta-value-dim]} w x y]
  (let [[b t] (array/shape x)
        hv (long delta-value-heads) dv (long delta-value-dim)
        z (array/reshape (common/linear x (:z w)) [b t hv dv])
        y (* (array/rms-norm y (:ssm-norm w) eps) (common/silu z))]
    (common/linear (array/reshape y [b t (c/* hv dv)]) (:out w))))

(defn- delta-net
  "One GatedDeltaNet layer of the normed x [B T hidden] after its cache
   entry {:conv :state} (nil: zeros, the start): [its output, the entry
   after x]."
  [cfg w x entry]
  (let [qkv (common/linear x (:qkv w))
        [beta g] (delta-gates w x)
        [y entry] (delta-rule cfg w qkv beta g entry nil)]
    [(delta-out cfg w x y) entry]))

(defn- query-and-gate
  "[query gate] of attn_q's projection [B T heads 2 head-dim]: each head's
   query, then its output gate."
  [qg]
  [(array/slice qg 3 0 1) (array/slice qg 3 1 2)])

(defn- gated-qkv
  "[q k v gate] of the normed x [B T hidden]: q [B heads T head-dim] and
   k, v [B KV-heads T head-dim], q and k normed per head and rotated over
   their first rope-dims at positions offset… (a host integer, or an int
   array of one per row); the gate [B T heads·head-dim]."
  [{:keys [heads kv-heads head-dim rope-dims rope-base eps]} w x offset]
  (let [[b t] (array/shape x)
        [q gate] (query-and-gate (array/reshape (common/linear x (:q w)) [b t heads 2 head-dim]))
        heads-of (fn [y n] (array/transpose (array/reshape y [b t n head-dim]) [0 2 1 3]))
        rotate (fn [y] (array/rope y rope-dims {:base rope-base :offset offset}))]
    [(rotate (array/rms-norm (heads-of q heads) (:q-norm w) eps))
     (rotate (array/rms-norm (heads-of (common/linear x (:k w)) kv-heads) (:k-norm w) eps))
     (heads-of (common/linear x (:v w)) kv-heads)
     (array/reshape gate [b t (c/* (long heads) (long head-dim))])]))

(defn- attend
  "The attention of q over the keys k and values v under mask, gated by
   sigmoid(gate), through the output projection: [B T hidden]."
  [{:keys [heads head-dim]} w q k v gate mask]
  (let [[b _ t] (array/shape q)]
    (-> (array/sdpa q k v {:mask mask})
        (array/transpose [0 2 1 3])
        (array/reshape [b t (c/* (long heads) (long head-dim))])
        (* (array/sigmoid gate))
        (common/linear (:o w)))))

(defn- attention
  "One gated attention layer of the normed x [B T hidden] at positions
   offset… after its cache entry (nil for none): [its output, the entry
   after x]."
  [cfg w x offset entry]
  (let [[q k v gate] (gated-qkv cfg w x offset)
        k (if entry (array/concatenate [(:k entry) k] 2) k)
        v (if entry (array/concatenate [(:v entry) v] 2) v)]
    [(attend cfg w q k v gate :causal) {:k k :v v}]))

(defn- static-attention
  "One gated attention layer of the normed x [B T hidden] after its
   static cache entry, at (static-forward's {:positions :mask :slots}):
   its keys and values written over their slots, [its output, the entry
   after x]."
  [cfg w x {:keys [positions mask slots]} entry]
  (let [[q k v gate] (gated-qkv cfg w x positions)
        k (array/put-along-axis (:k entry) slots k 2)
        v (array/put-along-axis (:v entry) slots v 2)]
    [(attend cfg w q k v gate mask) {:k k :v v}]))

(defn- picked
  "The indices [N k] (int32) of the k largest entries of p [N E] along
   its last axis, largest first, the first of equal entries first (as
   argmax): k argmaxes, each entry picked masked."
  [p k]
  (let [ids (array/arange 0 (peek (array/shape p)) 1 {:dtype :int32})]
    (loop [p p i 0 idx []]
      (if (c/== i (long k))
        (array/concatenate idx -1)
        (let [j (array/expand-dims (array/astype (array/argmax p {:axis -1}) :int32) -1)]
          (recur (array/where (array/== ids j) Double/NEGATIVE_INFINITY p) (inc i) (conj idx j)))))))

(defn- ranked
  "picked's indices by ranks: an entry's rank is the number of entries of
   its row larger, or equal and before it (argmax's order), counted over
   [N E E] comparisons; the entry of rank i is the i-th index (one-hot
   sums). About 10 kernels whatever k, where picked takes about 11 a
   pick (on MLX's CUDA, 2026-11-08). Ranks are a permutation of each row
   unless it holds a NaN, whose indices are clamped into range."
  [p k]
  (let [e (long (peek (array/shape p)))
        ids (array/arange 0 e 1 {:dtype :int32})
        pe (array/expand-dims p -1)
        pj (array/expand-dims p -2)
        before (array/< ids (array/expand-dims ids -1))
        ranks (array/sum (array/astype (array/logical-or (array/> pj pe) (array/logical-and (array/== pj pe) before)) :int32)
                         {:axis -1})
        hit (array/== (array/expand-dims ranks -2) (array/expand-dims (array/arange 0 k 1 {:dtype :int32}) -1))]
    (array/minimum (array/astype (array/sum (array/where hit ids 0) {:axis -1}) :int32) (dec e))))

(def ^:private ranked-limit
  "The most comparisons (rows × experts²) top-k makes by ranks: 2^24
   (64 MB of int32 counts, 256 rows of 256 experts); a longer read picks."
  16777216)

(defn- top-k
  "[indices values] of the k largest entries of p [N E] along its last
   axis, largest first: [N k] each, the indices int32 (the first of equal
   entries first, as argmax). The indices carry no gradient
   (stop-gradient: MLX's native gradients refuse a gather's VJP with
   respect to its indices), the values p's. By ranks (ranked) up to
   ranked-limit comparisons, else by picks (picked): the same indices.
   On MLX's CUDA, whose host encodes a kernel in about 5 µs, the picks
   were 3500 of a Qwen3.6-35B-A3B decode step's 5400 kernels: its
   encoding took 21 ms a step, by ranks 14 (genclj-p0ad)."
  [p k]
  (let [[n e] (array/shape p)
        idx (array/stop-gradient (if (c/<= (c/* (long n) (long e) (long e)) (long ranked-limit)) (ranked p k) (picked p k)))]
    [idx (array/take-along-axis p idx -1)]))

(defn- gathered
  "x's matrices [.. 1 in] times their experts' matrices of the stack w
   [experts out in] (transposed): expert indices [N k], x's matrix for
   each by lhs (its leading axes broadcasting to [N k]): [N k 1 out].
   Quantized stacks by gather-qmm, dense ones gathered by take; an
   adapted stack (gen.llm.lora's) adds its experts' terms."
  [x w indices lhs]
  (cond (lora/adapted? w) (+ (gathered x (:base w) indices lhs) (lora/delta-gathered x w indices))
        (map? w) (array/gather-qmm x (:wq w) (:scales w) (:biases w)
                                   (assoc (select-keys w [:bits :group-size]) :lhs-indices lhs :rhs-indices indices))
        :else (array/matmul x (array/transpose (array/take w indices 0) [0 1 3 2]))))

(defn- routing
  "[indices weights] of the router's probabilities probs [N experts]: the
   k most likely experts' indices [N k] (top-k's), and their
   probabilities renormalized to sum 1 (their sum clamped from below at
   6.1e-5, llama.cpp's)."
  [probs k]
  (let [[idx p] (top-k probs k)]
    [idx (/ p (array/maximum (array/sum p {:axis -1 :keep-dims? true}) 6.103515625e-5))]))

(defn- moe
  "The mixture of experts of the normed x [B T hidden], as llama.cpp's
   build_moe_ffn with qwen35moe's settings (softmax gating, the top k
   weights renormalized, no scale), plus the shared expert."
  [{:keys [experts-used dtype]} {:keys [router gate up down shared shared-gate]} x]
  (let [s (array/shape x)
        hidden (peek s)
        xs (array/reshape x [-1 hidden])
        n (long (first (array/shape xs)))
        k (long experts-used)
        logits (common/linear xs router)
        probs (array/softmax (if (#{:float16 :bfloat16} dtype) (array/astype logits :float32) logits) {:axis -1})
        [idx p] (routing probs k)
        p (if (= dtype (array/dtype p)) p (array/astype p dtype))
        x1 (array/reshape xs [n 1 1 hidden])
        token (array/reshape (array/arange 0 n 1 {:dtype :int32}) [n 1])
        act (* (common/silu (gathered x1 gate idx token)) (gathered x1 up idx token))
        y (array/squeeze (gathered act down idx (array/reshape (array/arange 0 (c/* n k) 1 {:dtype :int32}) [n k])) [2])
        routed (array/sum (* y (array/expand-dims p -1)) {:axis 1})
        out (if shared
              (+ routed (* (common/swiglu shared xs)
                           (array/sigmoid (common/linear xs (array/reshape shared-gate [1 hidden])))))
              routed)]
    (array/reshape out s)))

(defn- feed-forward [cfg w x]
  (if-let [e (:moe w)]
    (moe cfg e x)
    (common/swiglu w x)))

(defn- block
  "One block of x [B T hidden], (mixer x') giving [the output, the cache
   entry after it] of the normed x': [x after the block, the entry]."
  [{:keys [eps] :as cfg} w x mixer]
  (let [[a entry] (mixer (array/rms-norm x (:attn-norm w) eps))
        x (+ x a)]
    [(+ x (feed-forward cfg w (array/rms-norm x (:post-norm w) eps))) entry]))

(defn- logits-of
  "[logits, the cache entries after them] of tokens ids [B T] (an int
   array) through every block, block i's layer (mixer i w x');
   logits :all, :last, or {:rows idx}: the rows idx (int32 [O]) of B,
   [O T vocab]."
  [{:keys [layers eps dtype] :as cfg} weights ids mixer logits]
  (let [t (long (second (array/shape ids)))
        [h entries] (reduce (fn [[h entries] i]
                              (let [w (nth (:layers weights) i)
                                    [h e] (block cfg w h #(mixer i w %))]
                                [h (conj entries e)]))
                            [(common/embed (:embed weights) ids dtype) []]
                            (range layers))
        h (cond (= :last logits) (array/slice h 1 (dec t) t)
                (map? logits) (array/take h (:rows logits) 0)
                :else h)]
    [(common/linear (array/rms-norm h (:norm weights) eps) (or (:output weights) (:embed weights))) entries]))

;; ---------------------------------------------------------------------------
;; Training: log-probabilities and their gradients
;; ---------------------------------------------------------------------------

(defn- train-block
  "x [B T hidden] after one block w of kind (:delta or :attention), read
   from position 0 without a cache: (block …)'s x."
  [cfg kind x w]
  (first (block cfg w x (if (= :attention kind)
                          #(attention cfg w % 0 nil)
                          #(delta-net cfg w % nil)))))

(defn log-probs
  "The log-probabilities [B T-1] of each row of tokens (B rows of T ids,
   T >= 2) at positions 1 … T-1, each given the tokens before it: log p
   of the next token, as a model is trained by; qwen3/log-probs' for
   Qwen3.5 and Qwen3.6. Its logits are never made whole: the output
   projection runs :chunk vocabulary entries at a time into
   gen.array/selective-log-softmax. Differentiable with respect to the
   dense weights and a LoRA adapter's (gen.llm.lora/adapt's); a quantized
   matrix is frozen, its gradient with respect to the activations only.
   The gradient goes through gen.array/gated-delta's composite (the
   chunked form from 64 tokens): MLX's Metal kernel has no VJP and is
   not used under MLX's native gradients, nor on the tape. Options:
     :checkpoint?  true: every block, and every chunk of the output
                   projection, is checkpointed (array/checkpoint): its
                   activations are made again in the backward pass rather
                   than kept from the forward, so a gradient holds one
                   block's at a time, for about one more forward pass of
                   compute. The values and gradients are the same.
     :chunk        vocabulary entries per chunk of the output projection
                   (default: the whole vocabulary, one chunk).
     :temperature  the log-probabilities of the logits divided by it, a
                   positive number (default 1: the model's own): the
                   tempered distribution a sampler draws from
                   (gen.llm.sample's :temp; policy-gradient training,
                   gen.llm.grpo).
   Experimental: outside the 1.x promise (DESIGN §9)."
  ([m tokens] (log-probs m tokens nil))
  ([{:keys [config weights]} tokens {:keys [checkpoint? chunk temperature] :as opts}]
   (common/check-options who :log-probs opts #{:checkpoint? :chunk :temperature})
   (common/check-temperature who temperature)
   (let [{:keys [eps dtype vocab kinds]} config
         ids (common/token-array who tokens vocab)
         [_ t] (array/shape ids)
         _ (when (c/< (long t) 2)
             (throw (invalid (str "log-probs: rows of at least 2 tokens, not " t) {:tokens t})))
         block-of (into {} (for [kind [:delta :attention]
                                 :let [f #(train-block config kind %1 %2)]]
                             [kind (if checkpoint? (array/checkpoint f) f)]))
         h (reduce (fn [h [kind w]] ((block-of kind) h w))
                   (common/embed (:embed weights) ids dtype)
                   (map vector kinds (:layers weights)))
         hn (array/rms-norm (array/slice h 1 0 (dec t)) (:norm weights) eps)
         w (or (:output weights) (:embed weights))
         head (common/tempered-head temperature)]
     (array/selective-log-softmax (if checkpoint? head #(head %1 %2 hn w))
                                  (array/slice ids 1 1 t)
                                  (cond-> {:vocab vocab :chunk chunk} checkpoint? (assoc :args [hn w]))))))

(defn nll
  "The negative log-likelihood of tokens, the loss of supervised
   fine-tuning: -Σ log-probs, each position's times :mask's weight where
   given (a float array [B T-1], 0 at the positions not trained on, a
   prompt's), a 0-d array. Options: :mask, and log-probs' :checkpoint?
   and :chunk. Experimental: outside the 1.x promise (DESIGN §9)."
  ([m tokens] (nll m tokens nil))
  ([m tokens {:keys [mask] :as opts}]
   (common/check-options who :nll opts #{:mask :checkpoint? :chunk})
   (common/nll-of who (log-probs m tokens (dissoc opts :mask)) mask)))

;; ---------------------------------------------------------------------------
;; Static caches
;; ---------------------------------------------------------------------------

(defn- static? [cache] (contains? cache :capacity))

(defn- check-static [f cache]
  (when-not (static? cache)
    (throw (invalid (str (name f) " takes a static cache (static-cache, or prefill's :capacity), not "
                         (if (map? cache) "a concatenating cache" (pr-str cache)))
                    {:fn f}))))

(defn- static-forward
  "[logits, the static cache's layers after ids, its positions after
   them] of tokens ids [B T] (an int array) after the layers at positions
   (int32 [B]): arrays in and arrays out, the step compile-step compiles."
  [{:keys [kinds kv-heads head-dim] :as cfg} weights layers positions ids logits]
  (let [[b t] (array/shape ids)
        at (when-let [i (first (keep-indexed (fn [i kind] (when (= :attention kind) i)) kinds))]
             {:positions positions
              :mask (common/static-mask positions t (nth (array/shape (:k (nth layers i))) 2))
              :slots (common/write-slots positions [b kv-heads t head-dim])})
        [out entries] (logits-of cfg weights ids
                                 (fn [i w x]
                                   (if (= :attention (kinds i))
                                     (static-attention cfg w x at (nth layers i))
                                     (delta-net cfg w x (nth layers i))))
                                 logits)]
    [out entries (+ positions t)]))

(defn static-cache
  "An empty static cache of rows rows and capacity positions each, every
   row at offset 0: its attention layers' slots and its GatedDeltaNet
   layers' conv and recurrent states zero (one zero array of each shape,
   shared until a step writes). Its attention layers hold 2 × rows ×
   KV-heads × capacity × head-dim floats each, its GatedDeltaNet layers
   rows × (3 × channels + value-heads × value-dim × key-dim) whatever the
   capacity: for Qwen3.5-0.8B in float32, 24 KiB a position and 19 MiB a
   row."
  [{:keys [config]} rows capacity]
  (common/check-cache-arg who rows capacity)
  (let [{:keys [kinds kv-heads head-dim dtype delta-conv delta-key-heads delta-value-heads delta-key-dim delta-value-dim]} config
        kv (array/zeros [rows kv-heads capacity head-dim] {:dtype dtype})
        conv (array/zeros [rows (dec (long delta-conv))
                           (c/+ (c/* 2 (long delta-key-heads) (long delta-key-dim)) (c/* (long delta-value-heads) (long delta-value-dim)))]
                          {:dtype dtype})
        state (array/zeros [rows delta-value-heads delta-value-dim delta-key-dim] {:dtype (state-dtype dtype)})]
    {:capacity (long capacity)
     :offsets (vec (repeat rows 0))
     :positions (array/zeros [rows] {:dtype :int32})
     :layers (mapv #(if (= :attention %) {:k kv :v kv} {:conv conv :state state}) kinds)}))

(defn select-rows
  "The static cache of cache's rows idx (host integers, in order, repeats
   allowed), each at its own offset: a resample's ancestors, or a row
   alone."
  [cache idx]
  (check-static :select-rows cache)
  (let [idx (vec idx)]
    (common/check-row-indices who idx (count (:offsets cache)))
    (let [ix (array/array idx {:dtype :int32})
          pick #(array/take % ix 0)]
      {:capacity (:capacity cache)
       :offsets (mapv #((:offsets cache) %) idx)
       :positions (pick (:positions cache))
       :layers (mapv #(update-vals % pick) (:layers cache))})))

(defn rewind
  "The static cache with each row at its offset in offsets (a vector of
   one per row, or a host integer for every row): its own (as it is), or
   0, the row cleared as if it never read (its GatedDeltaNet states
   zeroed, its attention slots left to be written over, as qwen3's
   rewind). A recurrent state cannot be moved back to another position,
   so any other offset is loud (:unsupported): keep the cache from before
   it (a value) instead. It shares cache's arrays but a cleared row's
   states: under compile-step's :donate? step one of the two only."
  [cache offsets]
  (check-static :rewind cache)
  (let [os (:offsets cache)
        new (common/new-offsets who os offsets)]
    (when-let [[o n] (first (remove (fn [[o n]] (or (c/== (long o) (long n)) (zero? (long n)))) (map vector os new)))]
      (throw (unsupported (str "rewind: a row at " o " goes back to its own offset or to 0, not " n
                               " (a recurrent state cannot be moved back: keep the cache from before instead)")
                          {:offset n :row-offset o})))
    (let [cleared (mapv (fn [o n] (and (pos? (long o)) (zero? (long n)))) os new)
          keep (array/array (mapv not cleared) {:dtype :bool})
          clear (fn [x] (array/where (array/reshape keep (into [(count os)] (repeat (dec (array/ndim x)) 1))) x 0.0))]
      (assoc cache
             :offsets new
             :positions (array/array new {:dtype :int32})
             :layers (if (some true? cleared)
                       (mapv #(if (contains? % :state) (update-vals % clear) %) (:layers cache))
                       (:layers cache))))))

(defn grow
  "The static cache with capacity positions a row (at least its own), its
   rows, offsets and states as they were: each attention layer's keys and
   values extended by zero slots. A step's shapes change with the
   capacity, so a compiled step traces again; grow by blocks."
  [cache capacity]
  (check-static :grow cache)
  (let [old (long (:capacity cache))]
    (common/check-grow who old capacity)
    (if (c/== old (long capacity))
      cache
      (let [extend (fn [x]
                     (let [[rows heads _ dim] (array/shape x)]
                       (array/concatenate [x (array/zeros [rows heads (c/- (long capacity) old) dim] {:dtype (array/dtype x)})] 2)))]
        (assoc cache
               :capacity (long capacity)
               :layers (mapv #(if (contains? % :k) (update-vals % extend) %) (:layers cache)))))))

(defn concat-rows
  "The static cache of the static caches' rows, in order, each at its own
   offset: caches of one model and one capacity."
  [caches]
  (let [caches (vec caches)]
    (common/check-concat who caches static?)
    (let [cat #(array/concatenate (vec %) 0)]
      {:capacity (:capacity (first caches))
       :offsets (into [] (mapcat :offsets) caches)
       :positions (cat (map :positions caches))
       :layers (apply mapv (fn [& es] (into {} (map (fn [k] [k (cat (map k es))])) (keys (first es)))) (map :layers caches))})))

;; ---------------------------------------------------------------------------
;; Reading tokens
;; ---------------------------------------------------------------------------

(defn- cache-rows
  "The rows of a cache's arrays."
  [cache]
  (if (static? cache)
    (count (:offsets cache))
    (first (array/shape (first (vals (first (:layers cache))))))))

(defn offsets
  "The positions each of cache's rows holds, a vector (nil for nil, no
   cache)."
  [cache]
  (cond (nil? cache) nil
        (static? cache) (:offsets cache)
        :else (vec (repeat (cache-rows cache) (long (:offset cache))))))

(defn offset
  "The positions cache holds (0 for nil, no cache): every row's. Loud for
   a static cache whose rows hold different numbers (read those with
   offsets)."
  ^long [cache]
  (if (static? cache)
    (let [os (:offsets cache)]
      (when-not (apply = os)
        (throw (invalid "offset: the cache's rows sit at different offsets (offsets reads them)" {:offsets os})))
      (long (first os)))
    (long (or (:offset cache) 0))))

(defn- check-rows
  "tokens' b rows are cache's (when there is one)."
  [cache b]
  (when cache
    (let [kb (cache-rows cache)]
      (when-not (= b kb)
        (throw (invalid (str "tokens have " b " rows where the cache has " kb) {:rows b :cache-rows kb}))))))

(defn- advanced
  "The static cache after a chunk of t tokens, its layers and positions the
   step's."
  [cache layers positions t]
  (assoc cache :layers layers :positions positions :offsets (mapv #(c/+ (long %) (long t)) (:offsets cache))))

(defn- run
  "[logits cache'] of tokens after cache (nil for none, a concatenating
   cache or a static one); logits :all or :last."
  [{:keys [config weights]} cache tokens logits]
  (common/check-logits who logits)
  (let [ids (common/token-array who tokens (:vocab config))
        [b t] (array/shape ids)
        kinds (:kinds config)]
    (check-rows cache b)
    (if (static? cache)
      (do (common/check-capacity who cache t)
          (let [[out layers positions] (static-forward config weights (:layers cache) (:positions cache) ids logits)]
            [out (advanced cache layers positions t)]))
      (let [start (offset cache)
            [out entries] (logits-of config weights ids
                                     (fn [i w x]
                                       (let [entry (get-in cache [:layers i])]
                                         (if (= :attention (kinds i))
                                           (attention config w x start entry)
                                           (delta-net config w x entry))))
                                     logits)]
        [out {:offset (c/+ start (long t)) :layers entries}]))))

(defn forward
  "The logits [B T vocab] of every position of tokens (B rows of T ids),
   read from position 0."
  [m tokens]
  (first (run m nil tokens :all)))

(defn prefill
  "[logits cache]: the logits of tokens (B rows of T ids) read from
   position 0, [B T vocab] (or [B 1 vocab] with :logits :last), and the
   cache after them: a concatenating cache, or with :capacity n a static
   one of n positions a row."
  ([m tokens] (prefill m tokens nil))
  ([m tokens {:keys [logits capacity] :or {logits :all} :as opts}]
   (common/check-options who :prefill opts #{:logits :capacity})
   (if capacity
     (let [ids (common/token-array who tokens (:vocab (:config m)))]
       (run m (static-cache m (first (array/shape ids)) capacity) ids logits))
     (run m nil tokens logits))))

(defn step
  "[logits cache']: the logits of tokens (B rows of T ids, one or a
   chunk) read after cache (row b's from position (offsets cache)[b]; nil
   reads from 0, as prefill), and the cache after them, of cache's kind.
   Options: :logits, as prefill's."
  ([m cache tokens] (step m cache tokens nil))
  ([m cache tokens {:keys [logits] :or {logits :all} :as opts}]
   (common/check-options who :step opts #{:logits})
   (run m cache tokens logits)))

(defn compile-step
  "step over static caches, compiled by array/compile, as qwen3's: a
   function (f cache tokens) → [logits cache'], the values (step m cache
   tokens opts) gives (within float32's round-off), one trace per shape
   of tokens and cache on MLX, step itself elsewhere; a concatenating
   cache is loud (:invalid-args). The model's weights are evaluated here
   (array/eval!). Options:
     :logits   as step's;
     :donate?  true: each call consumes the cache it steps (array/compile's
               :donate, experimental): on MLX its arrays are donated, the
               attention slots written in place, and reading the consumed
               cache again is loud (:donated); where compile is step
               itself (the JVM backend) nothing is caught: never step a
               consumed cache again (fork one first with select-rows)."
  ([m] (compile-step m nil))
  ([{:keys [config weights]} {:keys [logits donate?] :or {logits :all} :as opts}]
   (common/check-options who :compile-step opts #{:logits :donate?})
   (common/check-logits who logits)
   (array/eval! weights)
   (let [f (array/compile (fn [layers positions ids] (static-forward config weights layers positions ids logits))
                          (when donate? {:donate #{0 1}}))]
     (fn [cache tokens]
       (check-static :compile-step cache)
       (let [ids (common/token-array who tokens (:vocab config))
             [b t] (array/shape ids)]
         (check-rows cache b)
         (common/check-capacity who cache t)
         (let [[out layers positions] (f (:layers cache) (:positions cache) ids)]
           [out (advanced cache layers positions t)]))))))

;; ---------------------------------------------------------------------------
;; Paged (genclj-6qb6): the attention layers' keys and values in paged
;; pools, the GatedDeltaNet layers' states a row each
;; ---------------------------------------------------------------------------

(defn paged-pools
  "Empty paged KV pools of n-blocks blocks of block-size positions for
   the attention layers, in qwen3/paged-pools' layouts and the model's
   dtype: a vector by layer, an attention layer's {:k :v} and a
   GatedDeltaNet layer's {} (its states are a row's, paged-states'). A
   step's layers merge them with the rows' states (paged-step)."
  [{:keys [config]} n-blocks block-size]
  (doseq [[what v] [[:n-blocks n-blocks] [:block-size block-size]]]
    (when-not (and (integer? v) (pos? (long v)))
      (throw (invalid (str "paged-pools: " (name what) " must be a positive integer, not " (pr-str v)) {what v}))))
  (let [{:keys [kinds kv-heads head-dim dtype]} config
        x (quot 16 (long ({:float16 2 :bfloat16 2 :float32 4 :float64 8} dtype)))]
    (when-not (zero? (rem (long head-dim) x))
      (throw (unsupported (str "a head of " head-dim " is not whole groups of " x " (the paged key layout)")
                          {:head-dim head-dim :group x})))
    (let [k (array/zeros [n-blocks kv-heads (quot (long head-dim) x) block-size x] {:dtype dtype})
          v (array/zeros [n-blocks kv-heads head-dim block-size] {:dtype dtype})]
      (mapv #(if (= :attention %) {:k k :v v} {}) kinds))))

(defn paged-states
  "The GatedDeltaNet layers' states of rows rows: a vector by layer, a
   GatedDeltaNet layer's {:conv [rows K-1 channels] :state [rows
   value-heads value-dim key-dim]} and an attention layer's {}: zeros
   (the start) with rows, or a concatenating cache's (one after a read,
   paged-cache's)."
  [m rows-or-cache]
  (if (integer? rows-or-cache)
    (mapv #(select-keys % [:conv :state]) (:layers (static-cache m rows-or-cache 1)))
    (mapv #(select-keys % [:conv :state]) (:layers rows-or-cache))))

(defn- pools? [entry] (contains? entry :k))

(defn paged-cache
  "The concatenating cache of one row holding a sequence's first len
   positions: layers by layer, the attention layers' pools (paged-pools')
   and the GatedDeltaNet layers' states of its row (paged-states' of one
   row), its blocks (block ids, a vector, of block-size positions each)
   gathered: for reading several tokens after it (step), whose keys and
   values paged-entries then gives, and its states paged-states."
  [_m layers blocks len block-size]
  (let [len (long len)
        tables (array/array [(vec blocks)] {:dtype :int32})]
    (when-not (c/<= len (c/* (count blocks) (long block-size)))
      (throw (invalid (str "paged-cache: " (count blocks) " blocks of " block-size " cannot hold " len " positions")
                      {:blocks (count blocks) :len len})))
    {:offset len
     :layers (mapv (fn [{:keys [k v] :as e}]
                     (if (pools? e)
                       (let [[gk gv] (paged/gathered k v tables)]
                         {:k (array/slice gk 2 0 len) :v (array/slice gv 2 0 len)})
                       e))
                   layers)}))

(defn paged-entries
  "The keys and values of a concatenating cache of one row at its
   positions from on, by layer: an attention layer's [k v], each [n
   KV-heads head-dim] (what gen.array/paged-write writes), a GatedDeltaNet
   layer's []."
  [cache from]
  (mapv (fn [{:keys [k v] :as e}]
          (if-not (pools? e)
            []
            (let [[b hk t d] (array/shape k)
                  n (c/- (long t) (long from))]
              (when-not (and (= 1 b) (c/<= 0 n))
                (throw (invalid "paged-entries: a cache of one row, from at most its offset" {:shape (array/shape k) :from from})))
              (mapv #(array/transpose (array/reshape (array/slice % 2 from t) [hk n d]) [1 0 2]) [k v]))))
        (:layers cache)))

(defn- paged-attention
  "One gated attention layer of the normed x [R 1 hidden], a token a row,
   over its pools entry: each row's key and value written at its slot (at
   its position, rotated there), then attention over each row's blocks
   (tables) to its position, gated: [its output, the pools after x]."
  [{:keys [heads kv-heads head-dim] :as cfg} w x {:keys [positions tables slots]} entry]
  (let [r (first (array/shape x))
        [q k v gate] (gated-qkv cfg w x positions)
        [kp vp] (array/paged-write (:k entry) (:v entry)
                                   (array/reshape k [r kv-heads head-dim]) (array/reshape v [r kv-heads head-dim]) slots)
        o (array/paged-attention (array/reshape q [r heads head-dim]) kp vp tables (+ positions 1))]
    [(-> (array/reshape o [r 1 (c/* (long heads) (long head-dim))])
         (* (array/sigmoid gate))
         (common/linear (:o w)))
     {:k kp :v vp}]))

(defn- paged-forward
  "[logits [R 1 vocab], the layers after the tokens] of one token a row
   (paged-step's): arrays in and arrays out, the step compile-paged-step
   compiles."
  [{:keys [kinds] :as cfg} weights layers positions tables slots ids]
  (let [at {:positions positions :tables tables :slots slots}]
    (logits-of cfg weights ids
               (fn [i w x]
                 (if (= :attention (kinds i))
                   (paged-attention cfg w x at (nth layers i))
                   (delta-net cfg w x (nth layers i))))
               :last)))

(defn- check-paged
  "The paged step's operands: layers of the model's kinds, R rows."
  [{:keys [kinds]} layers positions tables slots ids]
  (let [r (first (array/shape ids))]
    (when-not (and (vector? layers) (= (count kinds) (count layers))
                   (every? (fn [[kind e]] (if (= :attention kind)
                                            (pools? e)
                                            (and (= #{:conv :state} (set (keys e)))
                                                 (= r (first (array/shape (:state e)))) (= r (first (array/shape (:conv e)))))))
                           (map vector kinds layers)))
      (throw (invalid (str "paged-step: the layers must be a vector by layer, each attention layer's pools (paged-pools')"
                           " and each GatedDeltaNet layer's states of the " r " rows (paged-states')")
                      {:layers (if (vector? layers) (mapv (comp vec keys) layers) layers)})))
    (when-not (= [r 1] (array/shape ids))
      (throw (invalid (str "a paged step reads one token a row, [R 1], not " (array/shape ids)) {:tokens (array/shape ids)})))
    (doseq [[what x shape] [[:positions positions [r]] [:slots slots [r]]]]
      (when-not (= shape (array/shape x))
        (throw (invalid (str (name what) " must be [R] " shape ", not " (array/shape x)) {what (array/shape x)}))))
    (when-not (and (= 2 (array/ndim tables)) (= r (first (array/shape tables))))
      (throw (invalid (str "tables must be [R MB] for " r " rows, not " (array/shape tables)) {:tables (array/shape tables)})))))

(defn paged-step
  "[logits layers']: qwen3/paged-step's, the logits [R 1 vocab] of one
   token a row, for Qwen3.5's layers: layers is a vector by layer, each
   attention layer's paged pools (paged-pools'), each GatedDeltaNet
   layer's states of the step's R rows, in row order (paged-states', or a
   step's), merged; layers' the same after the tokens: the pools written,
   the states the rows' next. positions, tables, slots and tokens are
   qwen3/paged-step's, for the attention layers. Its logits are a static
   cache's within float round-off; the layers are values
   (compile-paged-step's :donate? consumes them instead)."
  [{:keys [config weights]} layers positions tables slots tokens]
  (let [ids (common/token-array who tokens (:vocab config))]
    (check-paged config layers positions tables slots ids)
    (paged-forward config weights layers positions tables slots ids)))

(defn- with-held
  "layers (by layer) with held's states (by layer: what each layer's entry
   lacks) merged in; held empty (or nil) for none."
  [layers held]
  (if (seq held) (mapv merge held layers) layers))

(defn compile-paged-step
  "paged-step compiled by array/compile, as qwen3/compile-paged-step: a
   function (f layers positions tables slots tokens) → [logits layers'],
   or (f layers positions tables slots tokens held), held a vector by
   layer of states the step reads but never consumes (a GatedDeltaNet
   layer's {:conv :state}, an attention layer's {}), each layer's
   states then in layers or in held. Options: :donate? true, each call
   consumes layers, the pools (written in place on MLX) and the states in
   them; give the states in held instead when others still read them (a
   group's rows read in place)."
  ([m] (compile-paged-step m nil))
  ([{:keys [config weights]} {:keys [donate?] :as opts}]
   (common/check-options who :compile-paged-step opts #{:donate?})
   (array/eval! weights)
   (let [f (array/compile (fn [layers held positions tables slots ids]
                            (paged-forward config weights (with-held layers held) positions tables slots ids))
                          (when donate? {:donate #{0}}))
         go (fn [layers positions tables slots tokens held]
              (let [ids (common/token-array who tokens (:vocab config))]
                (check-paged config (with-held layers held) positions tables slots ids)
                (f layers (vec held) positions tables slots ids)))]
     (fn
       ([layers positions tables slots tokens] (go layers positions tables slots tokens nil))
       ([layers positions tables slots tokens held] (go layers positions tables slots tokens held))))))

;; ---------------------------------------------------------------------------
;; Batched paged steps (genclj-6qb6): decode rows and a prefill chunk, as
;; qwen3's (genclj-b3ow), the chunk one sequence through the GatedDeltaNet
;; layers
;; ---------------------------------------------------------------------------

(defn- batch-attention
  "One gated attention layer of the normed x [R 1 hidden] of a batch
   (paged-batch-step's), a token a row, over its pools entry: every row's
   key and value written at its slot (at its position, rotated there),
   then common/batch-attention's attention (decode rows over their
   tables, the chunk's rows over its blocks gathered), gated: [its
   output, the pools after x]."
  [{:keys [heads kv-heads head-dim] :as cfg} w x {:keys [positions slots] :as at} entry]
  (let [r (first (array/shape x))
        [q k v gate] (gated-qkv cfg w x positions)
        [kp vp] (array/paged-write (:k entry) (:v entry)
                                   (array/reshape k [r kv-heads head-dim]) (array/reshape v [r kv-heads head-dim]) slots)
        o (common/batch-attention (array/reshape q [r heads head-dim]) kp vp at)]
    [(-> (array/reshape o [r 1 (c/* (long heads) (long head-dim))])
         (* (array/sigmoid gate))
         (common/linear (:o w)))
     {:k kp :v vp}]))

(defn- batch-delta-net
  "One GatedDeltaNet layer of the normed x [R 1 hidden] of a batch: its
   first d rows (decode rows) each one token of its own sequence, from
   its row of entry's {:conv :state} (d rows); the rest, with a chunk,
   one sequence's tokens in order, from entry's {:chunk-conv
   :chunk-state} (one row), those past :chunk-len padding, which leave
   its states as they are (no decay, no write; the conv state taken at
   its end). [its output, the entry after x: {:conv :state} of the d rows
   and {:chunk-conv :chunk-state} of the chunk]."
  [{:keys [delta-value-heads delta-value-dim] :as cfg} w x {:keys [d chunk chunk-len]} entry]
  (let [r (long (first (array/shape x)))
        d (long d)
        n (c/- r d)
        hv (long delta-value-heads) dv (long delta-value-dim)
        qkv (common/linear x (:qkv w))                                    ; [R 1 C]
        [beta g] (delta-gates w x)                                        ; [R 1 Hv]
        rows (fn [y ^long from ^long to] (if (and (zero? from) (c/== to r)) y (array/slice y 0 from to)))
        [yd ed] (when (pos? d)
                  (delta-rule cfg w (rows qkv 0 d) (rows beta 0 d) (rows g 0 d) (select-keys entry [:conv :state]) nil))
        [yc ec] (when chunk
                  (let [one (fn [y] (array/reshape (rows y d r) [1 n (peek (array/shape y))]))
                        live (when chunk-len (array/reshape (array/< (array/arange 0 n 1 {:dtype :int32}) chunk-len) [1 n 1]))
                        masked (fn [y] (if live (array/where live y 0.0) y))]
                    (delta-rule cfg w (one qkv) (masked (one beta)) (masked (one g))
                                {:conv (:chunk-conv entry) :state (:chunk-state entry)} chunk-len)))
        yc (when yc (array/reshape yc [n 1 hv dv]))
        y (cond (and yd yc) (array/concatenate [yd yc] 0) yd yd :else yc)]
    [(delta-out cfg w x y)
     (cond-> {}
       ed (merge ed)
       ec (assoc :chunk-conv (:conv ec) :chunk-state (:state ec)))]))

(defn- batch-forward
  "paged-batch-step's arrays in and out: {:logits [O 1 vocab] :scores [O]
   (with :targets) :layers}."
  [{:keys [kinds] :as cfg} weights layers {:keys [ids positions slots tables chunk chunk-len out targets]}]
  (let [at {:positions positions :slots slots :tables tables :chunk chunk :chunk-len chunk-len
            :d (if tables (first (array/shape tables)) 0)}
        [logits layers] (logits-of cfg weights ids
                                   (fn [i w x]
                                     (if (= :attention (kinds i))
                                       (batch-attention cfg w x at (nth layers i))
                                       (batch-delta-net cfg w x at (nth layers i))))
                                   {:rows out})]
    (cond-> {:logits logits :layers layers}
      targets (assoc :scores (common/target-scores logits targets)))))

(defn- check-batch
  "paged-batch-step's operands: layers of the model's kinds (the
   attention layers' pools, the GatedDeltaNet layers' states of the D
   decode rows and of the chunk), R rows, decode rows first, a chunk's
   after them."
  [{:keys [kinds]} layers {:keys [ids positions slots tables chunk chunk-len out targets]}]
  (let [r (long (first (array/shape ids)))
        d (long (if tables (first (array/shape tables)) 0))
        bad (fn [msg data] (throw (invalid (str "paged-batch-step: " msg) data)))
        rows (fn [x] (when (array/array? x) (first (array/shape x))))
        want (cond-> #{} (pos? d) (into [:conv :state]) chunk (into [:chunk-conv :chunk-state]))]
    (when-not (and (vector? layers) (= (count kinds) (count layers))
                   (every? (fn [[kind e]]
                             (if (= :attention kind)
                               (pools? e)
                               (and (map? e) (= want (set (keys e)))
                                    (every? #(= d (rows (% e))) (filter #{:conv :state} want))
                                    (every? #(= 1 (rows (% e))) (filter #{:chunk-conv :chunk-state} want)))))
                           (map vector kinds layers)))
      (bad (str "the layers must be a vector by layer, each attention layer's pools (paged-pools') and each"
                " GatedDeltaNet layer's states: {:conv :state} of the " d " decode rows (paged-states'), and"
                (if chunk " {:chunk-conv :chunk-state} of the chunk's one row" " no chunk's"))
           {:layers (if (vector? layers) (mapv #(when (map? %) (vec (keys %))) layers) layers)}))
    (when-not (= [r 1] (array/shape ids))
      (bad (str "a batch reads one token a row, [R 1], not " (array/shape ids)) {:tokens (array/shape ids)}))
    (doseq [[what x] [[:positions positions] [:slots slots]]]
      (when-not (and (array/array? x) (= [r] (array/shape x)))
        (bad (str (name what) " must be an int array [R] " [r]) {what (when (array/array? x) (array/shape x))})))
    (when (and tables (not (and (= 2 (array/ndim tables)) (c/<= 1 d r))))
      (bad (str ":tables must be [D MB] for D decode rows, at most R " r ", not " (array/shape tables))
           {:tables (array/shape tables)}))
    (when-not (= (some? chunk) (c/< d r))
      (bad (if chunk
             (str "a :chunk table, but no rows after the " d " decode rows")
             (str "the rows after the " d " decode rows read a chunk: a :chunk table is missing"))
           {:rows r :decode-rows d}))
    (when (and chunk (not (and (= 2 (array/ndim chunk)) (= 1 (first (array/shape chunk))))))
      (bad (str ":chunk must be [1 MB], not " (array/shape chunk)) {:chunk (array/shape chunk)}))
    (when (and chunk-len (not (and chunk (array/array? chunk-len) (= [1] (array/shape chunk-len)))))
      (bad ":chunk-len must be an int array [1], with a :chunk" {:chunk-len (when (array/array? chunk-len) (array/shape chunk-len))}))
    (when-not (and (array/array? out) (= 1 (array/ndim out)))
      (bad ":out must be the rows whose logits are computed, an int array [O]" {:out (when (array/array? out) (array/shape out))}))
    (when (and targets (not (and (array/array? targets) (= (array/shape out) (array/shape targets)))))
      (bad (str ":targets must be an int array [O] " (array/shape out)) {:targets (when (array/array? targets) (array/shape targets))}))))

(def ^:private batch-keys #{:tokens :positions :slots :tables :chunk :chunk-len :out :targets})

(defn- batch-arrays
  "batch with its :tokens as :ids, an int array [R 1]."
  [batch vocab]
  (-> batch (dissoc :tokens) (assoc :ids (common/token-array who (:tokens batch) vocab))))

(defn paged-batch-step
  "One forward of R rows, a token a row, as qwen3/paged-batch-step's, for
   Qwen3.5's layers: {:logits [O 1 vocab] :scores [O] (with :targets)
   :layers}. layers is a vector by layer: each attention layer's paged
   pools (paged-pools'), each GatedDeltaNet layer's states of the D
   decode rows, in row order ({:conv :state}, paged-states', when D is
   positive), and of the chunk's sequence ({:chunk-conv :chunk-state}, of
   one row, with a :chunk), merged; :layers the same after the rows.
   batch is qwen3/paged-batch-step's (:tokens :positions :slots :tables
   :chunk :out :targets) and
     :chunk-len  (optional, with a :chunk) [1] the chunk's tokens: its
                 rows past them are padding, which leave its states as
                 they were (absent: every chunk row is the sequence's).
   A decode row is one token of its own sequence; the chunk's rows are
   one sequence's tokens in order through the GatedDeltaNet layers
   (gen.array/gated-delta over them), as through the attention layers
   (each attending to the positions up to its own). Its logits are
   paged-step's (decode rows) and a concatenating cache's (a chunk)
   within float round-off; the layers are values, as paged-step's."
  [{:keys [config weights]} layers batch]
  (common/check-options who :paged-batch-step batch batch-keys)
  (let [b (batch-arrays batch (:vocab config))]
    (check-batch config layers b)
    (batch-forward config weights layers b)))

(defn compile-paged-batch-step
  "paged-batch-step compiled by array/compile: a function (f layers
   batch) → {:logits :scores :layers}, or (f layers batch held), held a
   vector by layer of states read but never consumed (a GatedDeltaNet
   layer's {:conv :state} or {:chunk-conv :chunk-state}, each in layers
   or in held), one trace per batch's structure and shapes. The model's
   weights are evaluated here. Options: :donate? true, each call
   consumes layers (the pools written in place on MLX, and the states in
   them), as compile-paged-step's."
  ([m] (compile-paged-batch-step m nil))
  ([{:keys [config weights]} {:keys [donate?] :as opts}]
   (common/check-options who :compile-paged-batch-step opts #{:donate?})
   (array/eval! weights)
   (let [f (array/compile (fn [layers held b] (batch-forward config weights (with-held layers held) b))
                          (when donate? {:donate #{0}}))
         go (fn [layers batch held]
              (common/check-options who :paged-batch-step batch batch-keys)
              (let [b (batch-arrays batch (:vocab config))]
                (check-batch config (with-held layers held) b)
                (f layers (vec held) b)))]
     (fn
       ([layers batch] (go layers batch nil))
       ([layers batch held] (go layers batch held))))))

;; ---------------------------------------------------------------------------
;; The family, by its architectures (gen.llm.models)
;; ---------------------------------------------------------------------------

(defmethod models/log-probs "qwen35" [m tokens & [opts]] (log-probs m tokens opts))

(defmethod models/log-probs "qwen35moe" [m tokens & [opts]] (log-probs m tokens opts))

(defmethod models/nll "qwen35" [m tokens & [opts]] (nll m tokens opts))

(defmethod models/nll "qwen35moe" [m tokens & [opts]] (nll m tokens opts))
