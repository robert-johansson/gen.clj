(ns gen.llm.models.qwen3
  "Qwen3's forward pass over gen.array, from a GGUF file's weights
   (docs/research/llm-mlx-engine.md §2, P2 of §6):

     (load path)                       → a model {:config :weights} from a GGUF file
     (load path {:dtype :float16})     → … computing in another float dtype
     (model gguf)                      → the same from gen.array/load-gguf's result
     (config metadata)                 → the architecture from a GGUF's metadata
     (forward m tokens)                → logits [B T vocab] of every position
     (prefill m tokens)                → [logits cache]: the logits, and the cache after the tokens
     (step m cache tokens)             → [logits cache']: the tokens read after cache
     (offset cache)                    → the positions cache holds

   tokens are B rows of T token ids each: host vectors (checked against the
   vocabulary) or an int array [B T]. A row's tokens are positions offset,
   offset + 1, …, offset + T - 1 after a cache of offset positions (0
   without one), every row at the same offset. A step reads one token a
   row, or a chunk of several: the logits of (step m (second (prefill m
   a)) b) are those (prefill m (a ++ b)) gives b's positions, within float
   round-off (exactly on the JVM backend: test/gen/llm/models/
   qwen3_test.clj). :logits :last (an option of prefill and step) computes
   only the last position's logits, [B 1 vocab], sparing the output
   projection of the others.

   A cache is a value: {:offset n :layers [{:k :v} …]}, each layer's keys
   and values [B KV-heads n head-dim], grown by concatenation, so a cache
   continued twice gives two caches and leaves itself as it was (a fork
   is free). Weights are values too: {:embed :norm :output :layers}, a
   layer's {:attn-norm :q :k :v :q-norm :k-norm :o :ffn-norm :gate :up
   :down}. A matrix is an array [out in], or a quantized one as
   gen.array/load-gguf gives Q8_0, Q4_0 and Q4_1 tensors, {:wq :scales
   :biases :bits :group-size}, multiplied by quantized-matmul (the weights
   frozen: a gradient is with respect to the dense weights and the
   activations). :output is nil for a tied output (Qwen3-0.6B's), which
   multiplies by the embedding.

   The tensors' meaning is llama.cpp's (src/models/qwen3.cpp, v0.9.0),
   which wrote the GGUF; GenMLX's qwen3_forward.cljs gave the structure:
   - embeddings by take (and dequantize);
   - per block: RMSNorm (attn_norm); the q, k and v projections; per-head
     RMSNorm of q and k (attn_q_norm, attn_k_norm); RoPE at the positions,
     NEOX's (the pairs i and i + head-dim/2) over the whole head, base
     rope.freq_base; attention with grouped KV heads (query head h reads
     KV head h / (heads / KV-heads)), causal, scaled by 1/√head-dim; the
     output projection (attn_output) and the residual; RMSNorm (ffn_norm)
     and SwiGLU, ffn_down(silu(ffn_gate x) · ffn_up x), and the residual;
   - RMSNorm (output_norm) and the output projection.
   Every tensor of the file must be one of these: a file with others
   (biases, a fused attn_qkv) is a model this namespace does not compute,
   a loud error, as llama.cpp's loader refuses tensors it does not read;
   so is RoPE scaling in its metadata.

   The forward is gen.array code: on MLX its rms-norm, rope and sdpa are
   MLX's fused kernels; the tape (or MLX's native gradients)
   differentiates it; on the JVM backend tiny models run in float64.
   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= == load])
  (:require [clojure.core :as c]
            [clojure.string :as str]
            [gen.array :as array :refer [+ *]]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; The configuration, from a GGUF's metadata
;; ---------------------------------------------------------------------------

(defn- unsupported
  ([message] (unsupported message {}))
  ([message data] (ex-info (str "qwen3: " message) (merge {:reason :unsupported} data))))

(defn- invalid
  ([message] (invalid message {}))
  ([message data] (ex-info (str "qwen3: " message) (merge {:reason :invalid-args} data))))

(defn config
  "Qwen3's architecture from a GGUF's metadata (gen.array/load-gguf's
   :metadata), llama.cpp's keys: {:hidden :layers :heads :kv-heads
   :head-dim :ffn :eps :rope-base :context}. Loud (:unsupported) for
   another architecture than qwen3, RoPE scaling, RoPE over part of a head,
   or keys and values of different sizes, which llama.cpp's qwen3 graph
   does not compute either."
  [metadata]
  (let [arch (get metadata "general.architecture")
        k (fn [suffix] (get metadata (str "qwen3." suffix)))
        need (fn [suffix]
               (let [v (k suffix)]
                 (when-not (and (integer? v) (pos? (long v)))
                   (throw (invalid (str "the metadata's qwen3." suffix " must be a positive integer, not " (pr-str v))
                                   {:key (str "qwen3." suffix)})))
                 (long v)))]
    (when-not (= "qwen3" arch)
      (throw (unsupported (str "the model's architecture is " (pr-str arch) ", not qwen3") {:architecture arch})))
    (let [hidden (need "embedding_length")
          heads (need "attention.head_count")
          head-dim (if (k "attention.key_length") (need "attention.key_length") (quot hidden heads))
          scaling (k "rope.scaling.type")
          cfg {:hidden hidden
               :layers (need "block_count")
               :heads heads
               :kv-heads (need "attention.head_count_kv")
               :head-dim head-dim
               :ffn (need "feed_forward_length")
               :eps (double (or (k "attention.layer_norm_rms_epsilon") 1e-6))
               :rope-base (double (or (k "rope.freq_base") 10000.0))
               :context (k "context_length")}]
      (when (and (k "attention.value_length") (not= head-dim (need "attention.value_length")))
        (throw (unsupported "keys and values of different sizes" {:key-length head-dim :value-length (k "attention.value_length")})))
      (when (and (k "rope.dimension_count") (not= head-dim (need "rope.dimension_count")))
        (throw (unsupported "RoPE over part of a head" {:rope-dims (k "rope.dimension_count") :head-dim head-dim})))
      (when (and scaling (not= "none" scaling))
        (throw (unsupported (str "RoPE scaling (" scaling ")") {:rope-scaling scaling})))
      (when-not (zero? (rem (long heads) (long (:kv-heads cfg))))
        (throw (invalid "the query heads are not a multiple of the KV heads" (select-keys cfg [:heads :kv-heads]))))
      cfg)))

;; ---------------------------------------------------------------------------
;; The weights, by llama.cpp's names
;; ---------------------------------------------------------------------------

(defn- float-array?
  "Is x a dense float array (not a quantized matrix)?"
  [x]
  (and (array/array? x) (= :float (array/dtype-kind (array/dtype x)))))

(defn- cast-weight
  "w (an array, or a quantized matrix's map) with its floats in dtype."
  [w dtype]
  (let [cast (fn [x] (if (= dtype (array/dtype x)) x (array/astype x dtype)))]
    (cond (map? w) (-> w (update :scales cast) (update :biases cast))
          (float-array? w) (cast w)
          :else (throw (invalid (str "a weight is neither a float array nor a quantized matrix: " (pr-str (type w))))))))

(defn- weight-shape
  "The shape a weight stands for: an array's, or a quantized matrix's
   once dequantized."
  [w]
  (if (map? w)
    (let [s (array/shape (:wq w))]
      (conj (pop s) (quot (c/* (long (peek s)) 32) (long (:bits w)))))
    (array/shape w)))

(defn model
  "A model {:config :weights} from gen.array/load-gguf's result: config
   from its metadata (with :vocab, the embedding's rows, and :dtype, the
   dtype it computes in), and the weights by llama.cpp's names, each
   checked against the configuration's shape. Options: :dtype, a float
   dtype (default: the backend's default float), to which the float
   weights are cast, the quantized matrices' scales and biases among them.
   Loud when a tensor the forward reads is missing or of another shape
   (:invalid-args), and when the file holds one it does not read
   (:unsupported)."
  ([gguf] (model gguf nil))
  ([{:keys [metadata tensors]} {:keys [dtype]}]
   (let [cfg (config metadata)
         {:keys [hidden layers heads kv-heads head-dim ffn]} cfg
         dtype (or dtype (:float (array/defaults array/*backend*)))
         _ (when-not (#{:float16 :bfloat16 :float32 :float64} dtype)
             (throw (invalid (str ":dtype must be a float dtype, not " (pr-str dtype)) {:dtype dtype})))
         read (atom #{})
         tensor (fn [name shape]
                  (let [w (or (get tensors name)
                              (throw (invalid (str "the file has no tensor " name) {:tensor name})))
                        s (weight-shape w)]
                    (when (and shape (not= shape s))
                      (throw (invalid (str "tensor " name " has shape " s ", not " shape) {:tensor name :shape s :expected shape})))
                    (swap! read conj name)
                    (cast-weight w dtype)))
         embed (tensor "token_embd.weight" nil)
         vocab (first (weight-shape embed))
         _ (when-not (= [vocab hidden] (weight-shape embed))
             (throw (invalid (str "the embedding has shape " (weight-shape embed) ", not [vocab " hidden "]") {:tensor "token_embd.weight"})))
         weights {:embed embed
                  :norm (tensor "output_norm.weight" [hidden])
                  :output (when (contains? tensors "output.weight") (tensor "output.weight" [vocab hidden]))
                  :layers (vec (for [i (range layers)
                                     :let [t (fn [nm shape] (tensor (str "blk." i "." nm ".weight") shape))]]
                                 {:attn-norm (t "attn_norm" [hidden])
                                  :q (t "attn_q" [(c/* heads head-dim) hidden])
                                  :k (t "attn_k" [(c/* kv-heads head-dim) hidden])
                                  :v (t "attn_v" [(c/* kv-heads head-dim) hidden])
                                  :q-norm (t "attn_q_norm" [head-dim])
                                  :k-norm (t "attn_k_norm" [head-dim])
                                  :o (t "attn_output" [hidden (c/* heads head-dim)])
                                  :ffn-norm (t "ffn_norm" [hidden])
                                  :gate (t "ffn_gate" [ffn hidden])
                                  :up (t "ffn_up" [ffn hidden])
                                  :down (t "ffn_down" [hidden ffn])}))}
         unread (sort (remove @read (keys tensors)))]
     (when (seq unread)
       (throw (unsupported (str "the file holds tensors Qwen3's forward does not read: " (str/join ", " (take 8 unread))
                                (when (c/> (count unread) 8) ", …"))
                           {:tensors (vec unread)})))
     {:config (assoc cfg :vocab vocab :dtype dtype) :weights weights})))

(defn load
  "(model (gen.array/load-gguf path) opts): a Qwen3 GGUF file's model, on
   *backend*."
  ([path] (load path nil))
  ([path opts] (model (array/load-gguf path) opts)))

;; ---------------------------------------------------------------------------
;; The forward
;; ---------------------------------------------------------------------------

(defn- linear
  "x·wᵀ for a weight w [out in]: dense, or quantized."
  [x w]
  (if (map? w)
    (array/quantized-matmul x (:wq w) (:scales w) (:biases w) (select-keys w [:bits :group-size]))
    (array/matmul x (array/transpose w))))

(defn- silu [x] (* x (array/sigmoid x)))

(defn- embed
  "The embeddings [B T hidden] of tokens [B T] (an int array), in dtype."
  [w tokens dtype]
  (if (map? w)
    (array/dequantize (array/take (:wq w) tokens 0) (array/take (:scales w) tokens 0) (array/take (:biases w) tokens 0)
                      (assoc (select-keys w [:bits :group-size]) :dtype dtype))
    (array/take w tokens 0)))

(defn- attention
  "One block's attention of x [B T hidden] at positions offset… after the
   layer's cache entry (nil for none): [its output, the entry after x]."
  [{:keys [heads kv-heads head-dim eps rope-base]} w x offset entry]
  (let [[b t] (array/shape x)
        heads-of (fn [y n] (array/transpose (array/reshape y [b t n head-dim]) [0 2 1 3]))   ; [B n T D]
        rotate (fn [y] (array/rope y head-dim {:base rope-base :offset offset}))
        q (rotate (array/rms-norm (heads-of (linear x (:q w)) heads) (:q-norm w) eps))
        k (rotate (array/rms-norm (heads-of (linear x (:k w)) kv-heads) (:k-norm w) eps))
        v (heads-of (linear x (:v w)) kv-heads)
        k (if entry (array/concatenate [(:k entry) k] 2) k)
        v (if entry (array/concatenate [(:v entry) v] 2) v)
        o (-> (array/sdpa q k v {:mask :causal})
              (array/transpose [0 2 1 3])
              (array/reshape [b t (c/* heads head-dim)]))]
    [(linear o (:o w)) {:k k :v v}]))

(defn- block
  "One block of x at offset after its cache entry: [x after it, the entry]."
  [{:keys [eps] :as cfg} w x offset entry]
  (let [[a entry] (attention cfg w (array/rms-norm x (:attn-norm w) eps) offset entry)
        x (+ x a)
        m (array/rms-norm x (:ffn-norm w) eps)]
    [(+ x (linear (* (silu (linear m (:gate w))) (linear m (:up w))) (:down w))) entry]))

(defn- token-array
  "tokens as an int array [B T]: host rows checked against the vocabulary,
   or an int array as it is."
  [tokens vocab]
  (cond
    (and (array/array? tokens) (not (array/host? tokens)))
    (do (when-not (and (= 2 (array/ndim tokens)) (= :int (array/dtype-kind (array/dtype tokens))))
          (throw (invalid (str "tokens must be an int array [B T], not " (array/dtype tokens) " " (array/shape tokens)))))
        tokens)

    (and (sequential? tokens) (seq tokens) (every? sequential? tokens))
    (let [n (count (first tokens))]
      (when-not (and (pos? n) (every? #(= n (count %)) tokens))
        (throw (invalid "tokens must be rows of one positive length" {:lengths (mapv count tokens)})))
      (doseq [row tokens t row]
        (when-not (and (integer? t) (c/<= 0 (long t)) (c/< (long t) (long vocab)))
          (throw (ex-info (str "qwen3: token " (pr-str t) " is not in the vocabulary [0, " vocab ")")
                          {:reason :invalid-token :token t :vocab vocab}))))
      (array/array tokens {:dtype :int32}))

    :else (throw (invalid (str "tokens must be rows of token ids (vectors) or an int array [B T], not " (pr-str tokens))))))

(defn offset
  "The positions cache holds (0 for nil, no cache)."
  ^long [cache]
  (long (or (:offset cache) 0)))

(defn- run
  "[logits cache'] of tokens after cache (nil for none); :logits :all or
   :last."
  [{:keys [config weights]} cache tokens {:keys [logits] :or {logits :all}}]
  (let [{:keys [layers vocab eps dtype]} config
        _ (when-not (#{:all :last} logits)
            (throw (invalid (str ":logits must be :all or :last, not " (pr-str logits)) {:logits logits})))
        ids (token-array tokens vocab)
        [b t] (array/shape ids)
        start (offset cache)]
    (when cache
      (let [kb (first (array/shape (:k (first (:layers cache)))))]
        (when-not (= b kb)
          (throw (invalid (str "tokens have " b " rows where the cache has " kb) {:rows b :cache-rows kb})))))
    (let [[h entries] (reduce (fn [[h entries] i]
                                (let [[h e] (block config (nth (:layers weights) i) h start (get-in cache [:layers i]))]
                                  [h (conj entries e)]))
                              [(embed (:embed weights) ids dtype) []]
                              (range layers))
          h (if (= :last logits) (array/slice h 1 (dec (long t)) t) h)
          out (linear (array/rms-norm h (:norm weights) eps) (or (:output weights) (:embed weights)))]
      [out {:offset (c/+ start (long t)) :layers entries}])))

(defn forward
  "The logits [B T vocab] of every position of tokens (B rows of T ids),
   read from position 0."
  [m tokens]
  (first (run m nil tokens nil)))

(defn prefill
  "[logits cache]: the logits of tokens (B rows of T ids) read from
   position 0, [B T vocab] (or [B 1 vocab] with :logits :last), and the
   cache after them."
  ([m tokens] (prefill m tokens nil))
  ([m tokens opts] (run m nil tokens opts)))

(defn step
  "[logits cache']: the logits of tokens (B rows of T ids, one or a
   chunk) read after cache (from position (offset cache); nil reads from
   0, as prefill), and the cache after them. Options as prefill's."
  ([m cache tokens] (step m cache tokens nil))
  ([m cache tokens opts] (run m cache tokens opts)))
