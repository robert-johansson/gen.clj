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
     (offsets cache)                   → … in each row
     (static-cache m rows capacity)    → an empty cache of fixed capacity
     (compile-step m)                  → step over static caches, compiled (array/compile),
                                         consuming each cache with :donate? true
     (log-probs m tokens)              → log p [B T-1] of each next token, for training
     (nll m tokens)                    → -Σ log p, supervised fine-tuning's loss
     (select-rows cache idx)           → a static cache of cache's rows idx
     (concat-rows caches)              → a static cache of the caches' rows
     (rewind cache offsets)            → a static cache with its rows back at offsets
     (grow cache capacity)             → a static cache of a larger capacity
     (paged-pools m n-blocks bs)       → empty paged KV pools (gen.array/paged-write's layouts)
     (paged-step m pools positions tables slots tokens)
                                       → [logits pools']: a token a row over paged pools
     (compile-paged-step m)            → paged-step compiled, consuming the pools with
                                         :donate? true
     (paged-batch-step m pools batch)  → {:logits :scores :pools}: decode rows and a
                                         prefill chunk over paged pools, one forward
     (compile-paged-batch-step m)      → paged-batch-step compiled (:donate? true)

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
   is free).

   A static cache (static-cache, or prefill's :capacity) is a value of
   fixed shapes instead, for compiled steps (genclj-tjvd): {:capacity n
   :offsets [o …] :positions :layers [{:k :v} …]}, each layer's keys and
   values [B KV-heads n head-dim] preallocated, a row's first o slots
   holding its positions. :offsets are the rows' positions on the host
   and :positions the same as an int32 array [B] on the device, so a
   step's arrays keep their shapes as the offsets grow: it writes its
   keys and values over the slots at each row's offset (put-along-axis,
   an overwrite), rotates them at those positions (rope's :offset an
   array) and masks the slots past each query's own position (sdpa's
   bool mask), so rows may sit at different offsets. What the slots past
   a row's offset hold is never read: zeros in a new cache, or what a
   rewound row read before (rewind), which the next steps write over.
   Its logits are the concatenating cache's: exactly on the JVM backend
   (the masked slots' weights are exact zeros, summed after the others),
   within float32's round-off on MLX. A step past the capacity is a loud
   error (:capacity-exceeded), checked on the host.
   compile-step compiles the step over static caches with array/compile:
   on MLX, one trace per shape of tokens (a decode step, a prefill
   chunk), replayed; a cache continued twice is still two caches, each
   step copying the cache it writes. With :donate? a step consumes its
   cache instead, and MLX updates it in place (array/compile's :donate):
   measured on Metal, Qwen3-0.6B decodes 1.3 times as fast that way at
   batch 1 and 1.6 times at batch 64 (bench/gen/qwen3_bench.clj).

   Weights are values too: {:embed :norm :output :layers}, a
   layer's {:attn-norm :q :k :v :q-norm :k-norm :o :ffn-norm :gate :up
   :down}. A matrix is an array [out in], or a quantized one as
   gen.array/load-gguf gives Q8_0, Q4_0, Q4_1 and K-quant tensors, {:wq
   :scales :biases :bits :group-size} (each tensor its own bits; Q6_K's
   load dense), multiplied by quantized-matmul (the weights
   frozen: a gradient is with respect to the dense weights and the
   activations). :output is nil for a tied output (Qwen3-0.6B's), which
   multiplies by the embedding. A projection adapted by gen.llm.lora/adapt
   is {:base w :lora …}, its product w's plus a LoRA adapter's term: every
   function here takes the adapted model, and log-probs and nll
   differentiate it with respect to the adapter.

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
            [gen.array :as array :refer [+ * <=]]
            [gen.array.paged :as paged]
            [gen.llm.lora :as lora]
            [gen.llm.models :as models]))

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

(defn- check-options
  "opts (a map or nil) holding only the keys accepted: else loud, as
   gen.clj's inference refuses options it does not read."
  [f opts accepted]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "qwen3/" (name f) ": options must be a map, not " (pr-str opts))
                    {:reason :invalid-options :fn f})))
  (when-let [unknown (seq (remove accepted (keys opts)))]
    (throw (ex-info (str "qwen3/" (name f) ": unknown options " (vec unknown) "; it takes " (vec (sort accepted)))
                    {:reason :unknown-options :fn f :unknown (vec unknown) :accepted accepted}))))

(defn config
  "Qwen3's architecture from a GGUF's metadata (gen.array/load-gguf's
   :metadata), llama.cpp's keys: {:architecture \"qwen3\" :hidden :layers
   :heads :kv-heads :head-dim :ffn :eps :rope-base :context}. Loud (:unsupported) for
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
          cfg {:architecture "qwen3"
               :hidden hidden
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
  ([{:keys [metadata tensors]} {:keys [dtype] :as opts}]
   (check-options :model opts #{:dtype})
   (let [cfg (config metadata)
         {:keys [hidden layers heads kv-heads head-dim ffn]} cfg
         dtype (or dtype (:float (array/defaults array/*backend*)))
         _ (when-not (#{:float16 :bfloat16 :float32 :float64} dtype)
             (throw (invalid (str ":dtype must be a float dtype, not " (pr-str dtype)) {:dtype dtype})))
         seen (atom #{})
         tensor (fn [name shape]
                  (let [w (or (get tensors name)
                              (throw (invalid (str "the file has no tensor " name) {:tensor name})))
                        s (weight-shape w)]
                    (when (and shape (not= shape s))
                      (throw (invalid (str "tensor " name " has shape " s ", not " shape) {:tensor name :shape s :expected shape})))
                    (swap! seen conj name)
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
         unread (sort (remove @seen (keys tensors)))]
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
  "x·wᵀ for a weight w [out in]: dense, quantized, or adapted (gen.llm.lora's:
   its base's product plus the adapter's term)."
  [x w]
  (cond (lora/adapted? w) (+ (linear x (:base w)) (lora/delta x w))
        (map? w) (array/quantized-matmul x (:wq w) (:scales w) (:biases w) (select-keys w [:bits :group-size]))
        :else (array/matmul x (array/transpose w))))

(defn- silu [x] (* x (array/sigmoid x)))

(defn- embed
  "The embeddings [B T hidden] of tokens [B T] (an int array), in dtype."
  [w tokens dtype]
  (if (map? w)
    (array/dequantize (array/take (:wq w) tokens 0) (array/take (:scales w) tokens 0) (array/take (:biases w) tokens 0)
                      (assoc (select-keys w [:bits :group-size]) :dtype dtype))
    (array/take w tokens 0)))

(defn- qkv
  "[q k v] of x [B T hidden], each [B heads T head-dim] (k and v of the KV
   heads): q and k normed per head and rotated at positions offset… (a
   host integer, or an int array of one per row)."
  [{:keys [heads kv-heads head-dim eps rope-base]} w x offset]
  (let [[b t] (array/shape x)
        heads-of (fn [y n] (array/transpose (array/reshape y [b t n head-dim]) [0 2 1 3]))   ; [B n T D]
        rotate (fn [y] (array/rope y head-dim {:base rope-base :offset offset}))]
    [(rotate (array/rms-norm (heads-of (linear x (:q w)) heads) (:q-norm w) eps))
     (rotate (array/rms-norm (heads-of (linear x (:k w)) kv-heads) (:k-norm w) eps))
     (heads-of (linear x (:v w)) kv-heads)]))

(defn- attend
  "The attention of q over the keys k and values v under mask, through the
   output projection: [B T hidden]."
  [{:keys [heads head-dim]} w q k v mask]
  (let [[b _ t] (array/shape q)]
    (-> (array/sdpa q k v {:mask mask})
        (array/transpose [0 2 1 3])
        (array/reshape [b t (c/* heads head-dim)])
        (linear (:o w)))))

(defn- attention
  "One block's attention of x [B T hidden] at positions offset… after the
   layer's cache entry (nil for none): [its output, the entry after x]."
  [cfg w x offset entry]
  (let [[q k v] (qkv cfg w x offset)
        k (if entry (array/concatenate [(:k entry) k] 2) k)
        v (if entry (array/concatenate [(:v entry) v] 2) v)]
    [(attend cfg w q k v :causal) {:k k :v v}]))

(defn- static-attention
  "One block's attention of x [B T hidden] after the layer's static cache
   entry, at (static-forward's {:positions :mask :slots}): its keys and
   values written over their slots, [its output, the entry after x]."
  [cfg w x {:keys [positions mask slots]} entry]
  (let [[q k v] (qkv cfg w x positions)
        k (array/put-along-axis (:k entry) slots k 2)
        v (array/put-along-axis (:v entry) slots v 2)]
    [(attend cfg w q k v mask) {:k k :v v}]))

(defn- block
  "One block of x [B T hidden], (attention x') giving [the output, the
   cache entry after it] of the normed x': [x after the block, the entry]."
  [{:keys [eps]} w x attention]
  (let [[a entry] (attention (array/rms-norm x (:attn-norm w) eps))
        x (+ x a)
        m (array/rms-norm x (:ffn-norm w) eps)]
    [(+ x (linear (* (silu (linear m (:gate w))) (linear m (:up w))) (:down w))) entry]))

(defn- logits-of
  "[logits, the cache entries after them] of tokens ids [B T] (an int
   array) through every block, block i's attention (attention i w x');
   logits :all, :last, or {:rows idx}: the rows idx (int32 [O]) of B, [O
   T vocab]."
  [{:keys [layers eps dtype] :as cfg} weights ids attention logits]
  (let [t (long (second (array/shape ids)))
        [h entries] (reduce (fn [[h entries] i]
                              (let [w (nth (:layers weights) i)
                                    [h e] (block cfg w h #(attention i w %))]
                                [h (conj entries e)]))
                            [(embed (:embed weights) ids dtype) []]
                            (range layers))
        h (cond (= :last logits) (array/slice h 1 (dec t) t)
                (map? logits) (array/take h (:rows logits) 0)
                :else h)]
    [(linear (array/rms-norm h (:norm weights) eps) (or (:output weights) (:embed weights))) entries]))

;; ---------------------------------------------------------------------------
;; Training: log-probabilities and their gradients
;; ---------------------------------------------------------------------------

(declare token-array)

(defn- train-block
  "x [B T hidden] after one block w, read from position 0 without a
   cache: (block …)'s x."
  [cfg x w]
  (first (block cfg w x #(attention cfg w % 0 nil))))

(defn- rows
  "The rows start to stop - 1 of a weight [out in], dense or quantized."
  [w start stop]
  (if (map? w)
    (reduce #(update %1 %2 array/slice 0 start stop) w [:wq :scales :biases])
    (array/slice w 0 start stop)))

(defn- head-logits
  "The logits of the vocabulary entries start to stop - 1 for the normed
   states hn [B T hidden], by those rows of the output weight w."
  [start stop hn w]
  (linear hn (rows w start stop)))

(defn log-probs
  "The log-probabilities [B T-1] of each row of tokens (B rows of T ids,
   T >= 2) at positions 1 … T-1, each given the tokens before it: log p
   of the next token, as a model is trained by. Its logits are never made
   whole: the output projection runs :chunk vocabulary entries at a time
   into gen.array/selective-log-softmax. Differentiable with respect to
   the dense weights (array/value-and-grad of a function that puts them
   in the model) and a LoRA adapter's (gen.llm.lora/adapt's); a quantized
   matrix is frozen, its gradient with respect to the activations only.
   Options:
     :checkpoint?  true: every block, and every chunk of the output
                   projection, is checkpointed (array/checkpoint): its
                   activations are made again in the backward pass rather
                   than kept from the forward, so a gradient holds one
                   block's at a time, for about one more forward pass of
                   compute. The values and gradients are the same.
     :chunk        vocabulary entries per chunk of the output projection
                   (default: the whole vocabulary, one chunk).
   Experimental: outside the 1.x promise (DESIGN §9)."
  ([m tokens] (log-probs m tokens nil))
  ([{:keys [config weights]} tokens {:keys [checkpoint? chunk] :as opts}]
   (check-options :log-probs opts #{:checkpoint? :chunk})
   (let [{:keys [eps dtype vocab]} config
         ids (token-array tokens vocab)
         [_ t] (array/shape ids)
         _ (when (c/< (long t) 2)
             (throw (invalid (str "log-probs: rows of at least 2 tokens, not " t) {:tokens t})))
         block-of (if checkpoint? (array/checkpoint #(train-block config %1 %2)) #(train-block config %1 %2))
         h (reduce block-of (embed (:embed weights) ids dtype) (:layers weights))
         hn (array/rms-norm (array/slice h 1 0 (dec t)) (:norm weights) eps)
         w (or (:output weights) (:embed weights))]
     (array/selective-log-softmax (if checkpoint? head-logits #(head-logits %1 %2 hn w))
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
   (check-options :nll opts #{:mask :checkpoint? :chunk})
   (let [lp (log-probs m tokens (dissoc opts :mask))]
     (when (and mask (not= (array/shape lp) (array/shape mask)))
       (throw (invalid (str "nll: the mask's shape " (array/shape mask) " is not the log-probabilities' " (array/shape lp))
                       {:shapes [(array/shape mask) (array/shape lp)]})))
     (array/negate (array/sum (if mask (* mask lp) lp))))))

;; ---------------------------------------------------------------------------
;; Static caches: slots written in place of a concatenation
;; ---------------------------------------------------------------------------

(defn- static? [cache] (contains? cache :capacity))

(defn- chunk-positions
  "The positions of a chunk of t tokens, [B 1 t 1]: row b's i-th at
   positions[b] + i."
  [positions t]
  (+ (array/reshape positions [(first (array/shape positions)) 1 1 1])
     (array/reshape (array/arange 0 t 1 {:dtype :int32}) [1 1 t 1])))

(defn- static-mask
  "The slots of a static cache of capacity that a chunk of t tokens' queries
   see, a bool array [B 1 t capacity]: query i of row b, at position
   positions[b] + i, sees the slots up to its own (causal); the slots past
   it hold the chunk's later tokens or padding."
  [positions t capacity]
  (<= (array/arange 0 capacity 1 {:dtype :int32}) (chunk-positions positions t)))

(defn- write-slots
  "Where a chunk's keys or values go, as put-along-axis's indices: an int
   array of their shape [B KV-heads t head-dim], positions[b] + i at the
   chunk's i-th position."
  [positions [_ _ t _ :as shape]]
  (array/broadcast-to (chunk-positions positions t) shape))

(defn- static-forward
  "[logits, the static cache's layers after ids, its positions after
   them] of tokens ids [B T] (an int array) after the layers at positions
   (int32 [B]): arrays in and arrays out, the step compile-step compiles."
  [{:keys [kv-heads head-dim] :as cfg} weights layers positions ids logits]
  (let [[b t] (array/shape ids)
        capacity (nth (array/shape (:k (first layers))) 2)
        at {:positions positions
            :mask (static-mask positions t capacity)
            :slots (write-slots positions [b kv-heads t head-dim])}
        [out entries] (logits-of cfg weights ids (fn [i w x] (static-attention cfg w x at (nth layers i))) logits)]
    [out entries (+ positions t)]))

(defn static-cache
  "An empty static cache of rows rows and capacity positions each, every
   row at offset 0 and every slot zero, its arrays in the model's dtype
   (one zero array, shared until a step writes). It holds 2 × layers ×
   rows × KV-heads × capacity × head-dim floats: for Qwen3-0.6B in
   float32, 224 KiB a position a row."
  [{:keys [config]} rows capacity]
  (doseq [[what v] [[:rows rows] [:capacity capacity]]]
    (when-not (and (integer? v) (pos? (long v)))
      (throw (invalid (str "static-cache: " (name what) " must be a positive integer, not " (pr-str v)) {what v}))))
  (let [{:keys [layers kv-heads head-dim dtype]} config
        zero (array/zeros [rows kv-heads capacity head-dim] {:dtype dtype})]
    {:capacity (long capacity)
     :offsets (vec (repeat rows 0))
     :positions (array/zeros [rows] {:dtype :int32})
     :layers (vec (repeat layers {:k zero :v zero}))}))

(defn- check-static [f cache]
  (when-not (static? cache)
    (throw (invalid (str (name f) " takes a static cache (static-cache, or prefill's :capacity), not "
                         (if (map? cache) "a concatenating cache" (pr-str cache)))
                    {:fn f}))))

(defn select-rows
  "The static cache of cache's rows idx (host integers, in order, repeats
   allowed), each at its own offset: a resample's ancestors, or a row
   alone."
  [cache idx]
  (check-static :select-rows cache)
  (let [n (count (:offsets cache))
        idx (vec idx)]
    (when (empty? idx)
      (throw (invalid "select-rows: no rows" {:rows idx})))
    (doseq [i idx]
      (when-not (and (integer? i) (c/<= 0 (long i)) (c/< (long i) n))
        (throw (invalid (str "select-rows: " (pr-str i) " is not a row of a cache of " n) {:row i :rows n}))))
    (let [ix (array/array idx {:dtype :int32})
          pick #(array/take % ix 0)]
      {:capacity (:capacity cache)
       :offsets (mapv #((:offsets cache) %) idx)
       :positions (pick (:positions cache))
       :layers (mapv (fn [{:keys [k v]}] {:k (pick k) :v (pick v)}) (:layers cache))})))

(defn rewind
  "The static cache with each row back at its offset in offsets (a vector
   of one per row, or a host integer for every row), each at most the
   row's own: the positions past it dropped, as if never read. Only the
   offsets move; the slots keep what they hold until the next steps write
   over them (put-along-axis), so a rewound cache reads as one that never
   read those positions: speculative decoding's rollback, or a slot
   reused. It shares cache's arrays: under compile-step's :donate? step
   one of the two only."
  [cache offsets]
  (check-static :rewind cache)
  (let [os (:offsets cache)
        new (if (integer? offsets) (vec (repeat (count os) offsets)) offsets)]
    (when-not (and (sequential? new) (= (count new) (count os)))
      (throw (invalid (str "rewind: the offsets are one per row (" (count os) ") or one integer, not " (pr-str offsets))
                      {:offsets offsets :rows (count os)})))
    (doseq [[o n] (map vector os new)]
      (when-not (and (integer? n) (c/<= 0 (long n) (long o)))
        (throw (invalid (str "rewind: " (pr-str n) " is not an offset from 0 to the row's " o) {:offset n :row-offset o}))))
    (let [new (mapv long new)]
      (assoc cache :offsets new :positions (array/array new {:dtype :int32})))))

(defn grow
  "The static cache with capacity positions a row (at least its own), its
   rows and offsets as they were: each layer's keys and values extended by
   zero slots. A step's shapes change with the capacity, so a compiled step
   traces again; grow by blocks."
  [cache capacity]
  (check-static :grow cache)
  (let [old (long (:capacity cache))]
    (when-not (and (integer? capacity) (c/<= old (long capacity)))
      (throw (invalid (str "grow: the capacity is an integer of at least the cache's " old ", not " (pr-str capacity))
                      {:capacity capacity :cache-capacity old})))
    (if (c/== old (long capacity))
      cache
      (let [{:keys [k]} (first (:layers cache))
            [rows heads _ dim] (array/shape k)
            pad (array/zeros [rows heads (c/- (long capacity) old) dim] {:dtype (array/dtype k)})
            extend #(array/concatenate [% pad] 2)]
        (assoc cache
               :capacity (long capacity)
               :layers (mapv (fn [{:keys [k v]}] {:k (extend k) :v (extend v)}) (:layers cache)))))))

(defn concat-rows
  "The static cache of the static caches' rows, in order, each at its own
   offset: caches of one model and one capacity."
  [caches]
  (let [caches (vec caches)]
    (when (empty? caches)
      (throw (invalid "concat-rows: no caches" {})))
    (doseq [cache caches] (check-static :concat-rows cache))
    (when-not (apply = (map :capacity caches))
      (throw (invalid (str "concat-rows: caches of different capacities " (mapv :capacity caches))
                      {:capacities (mapv :capacity caches)})))
    (when-not (apply = (map (comp count :layers) caches))
      (throw (invalid "concat-rows: caches of different numbers of layers" {:layers (mapv (comp count :layers) caches)})))
    (let [cat #(array/concatenate (vec %) 0)]
      {:capacity (:capacity (first caches))
       :offsets (into [] (mapcat :offsets) caches)
       :positions (cat (map :positions caches))
       :layers (apply mapv (fn [& es] {:k (cat (map :k es)) :v (cat (map :v es))}) (map :layers caches))})))

;; ---------------------------------------------------------------------------
;; Reading tokens
;; ---------------------------------------------------------------------------

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

(defn offsets
  "The positions each of cache's rows holds, a vector (nil for nil, no
   cache)."
  [cache]
  (cond (nil? cache) nil
        (static? cache) (:offsets cache)
        :else (vec (repeat (first (array/shape (:k (first (:layers cache))))) (long (:offset cache))))))

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

(defn- check-logits [logits]
  (when-not (#{:all :last} logits)
    (throw (invalid (str ":logits must be :all or :last, not " (pr-str logits)) {:logits logits}))))

(defn- check-rows
  "tokens' b rows are cache's (when there is one)."
  [cache b]
  (when cache
    (let [kb (if (static? cache) (count (:offsets cache)) (first (array/shape (:k (first (:layers cache))))))]
      (when-not (= b kb)
        (throw (invalid (str "tokens have " b " rows where the cache has " kb) {:rows b :cache-rows kb}))))))

(defn- check-capacity
  "A chunk of t tokens fits in every row of the static cache."
  [{:keys [capacity offsets]} t]
  (let [last-offset (long (apply max offsets))]
    (when (c/> (c/+ last-offset (long t)) (long capacity))
      (throw (ex-info (str "qwen3: " t " tokens after offset " last-offset " exceed the static cache's capacity of " capacity)
                      {:reason :capacity-exceeded :capacity capacity :offsets offsets :tokens t})))))

(defn- advanced
  "The static cache after a chunk of t tokens, its layers and positions the
   step's."
  [cache layers positions t]
  (assoc cache :layers layers :positions positions :offsets (mapv #(c/+ (long %) (long t)) (:offsets cache))))

(defn- run
  "[logits cache'] of tokens after cache (nil for none, a concatenating
   cache or a static one); logits :all or :last."
  [{:keys [config weights]} cache tokens logits]
  (check-logits logits)
  (let [ids (token-array tokens (:vocab config))
        [b t] (array/shape ids)]
    (check-rows cache b)
    (if (static? cache)
      (do (check-capacity cache t)
          (let [[out layers positions] (static-forward config weights (:layers cache) (:positions cache) ids logits)]
            [out (advanced cache layers positions t)]))
      (let [start (offset cache)
            [out entries] (logits-of config weights ids
                                     (fn [i w x] (attention config w x start (get-in cache [:layers i])))
                                     logits)]
        [out {:offset (c/+ start (long t)) :layers entries}]))))

;; ---------------------------------------------------------------------------
;; Paged KV caches (genclj-1daf): pools of blocks, one query a row
;; ---------------------------------------------------------------------------

(defn paged-pools
  "Empty paged KV pools of n-blocks blocks of block-size positions, a
   layer's {:k :v} in gen.array/paged-write's layouts and the model's
   dtype ([NB KV-heads head-dim/x BS x] and [NB KV-heads head-dim BS]):
   a vector, by layer (one zero array, shared until a step writes). They
   hold 2 × layers × KV-heads × head-dim floats a position, as a static
   cache's row does."
  [{:keys [config]} n-blocks block-size]
  (doseq [[what v] [[:n-blocks n-blocks] [:block-size block-size]]]
    (when-not (and (integer? v) (pos? (long v)))
      (throw (invalid (str "paged-pools: " (name what) " must be a positive integer, not " (pr-str v)) {what v}))))
  (let [{:keys [layers kv-heads head-dim dtype]} config
        x (quot 16 (long ({:float16 2 :bfloat16 2 :float32 4 :float64 8} dtype)))]
    (when-not (zero? (rem (long head-dim) x))
      (throw (ex-info (str "qwen3: a head of " head-dim " is not whole groups of " x " (the paged key layout)")
                      {:reason :unsupported :head-dim head-dim :group x})))
    (let [k (array/zeros [n-blocks kv-heads (quot (long head-dim) x) block-size x] {:dtype dtype})
          v (array/zeros [n-blocks kv-heads head-dim block-size] {:dtype dtype})]
      (vec (repeat layers {:k k :v v})))))

(defn paged-cache
  "The concatenating cache of one row holding a sequence's first len
   positions from paged pools, its blocks (block ids, a vector, of
   block-size positions each) gathered: for reading several tokens after
   it (step), whose keys and values paged-entries then gives."
  [_m pools blocks len block-size]
  (let [len (long len)
        tables (array/array [(vec blocks)] {:dtype :int32})]
    (when-not (c/<= len (c/* (count blocks) (long block-size)))
      (throw (invalid (str "paged-cache: " (count blocks) " blocks of " block-size " cannot hold " len " positions")
                      {:blocks (count blocks) :len len})))
    {:offset len
     :layers (mapv (fn [{:keys [k v]}]
                     (let [[gk gv] (paged/gathered k v tables)]
                       {:k (array/slice gk 2 0 len) :v (array/slice gv 2 0 len)}))
                   pools)}))

(defn paged-entries
  "The keys and values of a concatenating cache of one row at its
   positions from on, by layer [k v], each [n KV-heads head-dim]: what
   gen.array/paged-write writes."
  [cache from]
  (mapv (fn [{:keys [k v]}]
          (let [[b hk t d] (array/shape k)
                n (c/- (long t) (long from))]
            (when-not (and (= 1 b) (c/<= 0 n))
              (throw (invalid "paged-entries: a cache of one row, from at most its offset" {:shape (array/shape k) :from from})))
            (mapv #(array/transpose (array/reshape (array/slice % 2 from t) [hk n d]) [1 0 2]) [k v])))
        (:layers cache)))

(defn- paged-attention
  "One block's attention of x [R 1 hidden], a token a row, over the
   layer's pools entry: its key and value written at slots (at positions,
   rotated there), then attention over each row's blocks (tables) to its
   position: [its output, the pools after x]."
  [{:keys [heads kv-heads head-dim] :as cfg} w x {:keys [positions tables slots]} entry]
  (let [r (first (array/shape x))
        [q k v] (qkv cfg w x positions)
        [kp vp] (array/paged-write (:k entry) (:v entry)
                                   (array/reshape k [r kv-heads head-dim]) (array/reshape v [r kv-heads head-dim]) slots)
        o (array/paged-attention (array/reshape q [r heads head-dim]) kp vp tables (+ positions 1))]
    [(linear (array/reshape o [r 1 (c/* heads head-dim)]) (:o w)) {:k kp :v vp}]))

(defn- paged-forward
  "[logits [R 1 vocab], the pools after the tokens] of one token a row,
   ids [R 1] at positions (int32 [R]), written at slots (int32 [R]) and
   attending over tables (int32 [R MB]): arrays in and arrays out, the
   step compile-paged-step compiles."
  [cfg weights pools positions tables slots ids]
  (let [at {:positions positions :tables tables :slots slots}]
    (logits-of cfg weights ids (fn [i w x] (paged-attention cfg w x at (nth pools i))) :last)))

(defn- check-paged
  "The paged step's operands: pools of the model's layers, R rows."
  [{:keys [layers]} pools positions tables slots ids]
  (let [r (first (array/shape ids))]
    (when-not (and (vector? pools) (= layers (count pools)))
      (throw (invalid (str "the pools must be a vector of the model's " layers " layers (paged-pools)") {:layers (count pools)})))
    (when-not (= [r 1] (array/shape ids))
      (throw (invalid (str "a paged step reads one token a row, [R 1], not " (array/shape ids)) {:tokens (array/shape ids)})))
    (doseq [[what x shape] [[:positions positions [r]] [:slots slots [r]]]]
      (when-not (= shape (array/shape x))
        (throw (invalid (str (name what) " must be [R] " shape ", not " (array/shape x)) {what (array/shape x)}))))
    (when-not (and (= 2 (array/ndim tables)) (= r (first (array/shape tables))))
      (throw (invalid (str "tables must be [R MB] for " r " rows, not " (array/shape tables)) {:tables (array/shape tables)})))))

(defn paged-step
  "[logits pools']: the logits [R 1 vocab] of one token a row, read after
   each row's positions in paged pools (paged-pools'), and the pools after
   them. Row i's token is position positions[i] of its sequence, written
   at slots[i] (gen.llm.paged/prepare's), and attends over its first
   positions[i] + 1 positions, its blocks the row i of tables [R MB].
   positions, tables and slots are int32 arrays; tokens are R rows of one
   id or an int array [R 1]. Its logits are a static cache's within
   float round-off (on the JVM backend, float64, a masked position's
   weight is an exact 0), the pools being values: a step leaves the pools
   it was given as they were (compile-paged-step's :donate? consumes
   them instead)."
  [{:keys [config weights]} pools positions tables slots tokens]
  (let [ids (token-array tokens (:vocab config))]
    (check-paged config pools positions tables slots ids)
    (paged-forward config weights pools positions tables slots ids)))

(defn compile-paged-step
  "paged-step compiled by array/compile: a function (f pools positions
   tables slots tokens) → [logits pools'], one trace per shape (rows,
   table width, pool size). The model's weights are evaluated here, as
   compile-step's. Options: :donate? true, each call consumes the pools
   it steps (array/compile's :donate, experimental): on MLX the step
   writes its keys and values into the pools in place, and reading the
   consumed pools again is loud (:donated)."
  ([m] (compile-paged-step m nil))
  ([{:keys [config weights]} {:keys [donate?] :as opts}]
   (check-options :compile-paged-step opts #{:donate?})
   (array/eval! weights)
   (let [f (array/compile (fn [pools positions tables slots ids] (paged-forward config weights pools positions tables slots ids))
                          (when donate? {:donate #{0}}))]
     (fn [pools positions tables slots tokens]
       (let [ids (token-array tokens (:vocab config))]
         (check-paged config pools positions tables slots ids)
         (f pools positions tables slots ids))))))

;; ---------------------------------------------------------------------------
;; Batched paged steps (genclj-b3ow): decode rows and a prefill chunk
;; ---------------------------------------------------------------------------

(defn- batch-attention
  "One block's attention of x [R 1 hidden], a token a row, over the
   layer's pools entry: every row's key and value written at its slot (at
   its position, rotated there), then the first d rows' attention by
   paged attention over their tables, and the rest's (a prefill chunk of
   one sequence) over the chunk table's blocks gathered, each row to its
   own position (a causal mask): [its output, the pools after x]."
  [{:keys [heads kv-heads head-dim] :as cfg} w x {:keys [positions tables slots chunk d]} entry]
  (let [r (long (first (array/shape x)))
        d (long d)
        [q k v] (qkv cfg w x positions)
        [kp vp] (array/paged-write (:k entry) (:v entry)
                                   (array/reshape k [r kv-heads head-dim]) (array/reshape v [r kv-heads head-dim]) slots)
        q (array/reshape q [r heads head-dim])
        decoded (when (pos? d)
                  (array/paged-attention (if (c/== d r) q (array/slice q 0 0 d)) kp vp tables
                                         (+ (if (c/== d r) positions (array/slice positions 0 0 d)) 1)))
        chunked (when chunk
                  (let [n (c/- r d)
                        [gk gv] (paged/gathered kp vp chunk)                           ; [1 Hk T D]
                        t (nth (array/shape gk) 2)
                        qc (array/transpose (array/reshape (array/slice q 0 d r) [1 n heads head-dim]) [0 2 1 3])
                        mask (<= (array/arange 0 t 1 {:dtype :int32})
                                 (array/reshape (array/slice positions 0 d r) [1 1 n 1]))]
                    (-> (array/sdpa qc gk gv {:mask mask})                             ; [1 heads n D]
                        (array/transpose [0 2 1 3])
                        (array/reshape [n heads head-dim]))))
        o (cond (and decoded chunked) (array/concatenate [decoded chunked] 0)
                decoded decoded
                :else chunked)]
    [(linear (array/reshape o [r 1 (c/* heads head-dim)]) (:o w)) {:k kp :v vp}]))

(defn- batch-forward
  "paged-batch-step's arrays in and out: {:logits [O 1 vocab] :scores [O]
   (with :targets) :pools}."
  [cfg weights pools {:keys [ids positions slots tables chunk out targets]}]
  (let [at {:positions positions :slots slots :tables tables :chunk chunk
            :d (if tables (first (array/shape tables)) 0)}
        [logits pools] (logits-of cfg weights ids (fn [i w x] (batch-attention cfg w x at (nth pools i))) {:rows out})]
    (cond-> {:logits logits :pools pools}
      targets (assoc :scores (array/reshape (array/- (array/take-along-axis logits (array/reshape targets [-1 1 1]) 2)
                                                     (array/logsumexp logits {:axis -1 :keep-dims? true}))
                                            [-1])))))

(defn- check-batch
  "paged-batch-step's operands: pools of the model's layers, R rows,
   decode rows first, a chunk's after them."
  [{:keys [layers]} pools {:keys [ids positions slots tables chunk out targets]}]
  (let [r (long (first (array/shape ids)))
        d (long (if tables (first (array/shape tables)) 0))
        bad (fn [msg data] (throw (invalid (str "paged-batch-step: " msg) data)))]
    (when-not (and (vector? pools) (= layers (count pools)))
      (bad (str "the pools must be a vector of the model's " layers " layers (paged-pools)") {:layers (count pools)}))
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
    (when-not (and (array/array? out) (= 1 (array/ndim out)))
      (bad ":out must be the rows whose logits are computed, an int array [O]" {:out (when (array/array? out) (array/shape out))}))
    (when (and targets (not (and (array/array? targets) (= (array/shape out) (array/shape targets)))))
      (bad (str ":targets must be an int array [O] " (array/shape out)) {:targets (when (array/array? targets) (array/shape targets))}))))

(defn- batch-arrays
  "batch with its :tokens as :ids, an int array [R 1]."
  [batch vocab]
  (-> batch (dissoc :tokens) (assoc :ids (token-array (:tokens batch) vocab))))

(def ^:private batch-keys #{:tokens :positions :slots :tables :chunk :out :targets})

(defn paged-batch-step
  "One forward of R rows over paged pools (paged-pools'), a token a row,
   and the pools after them: {:logits [O 1 vocab] :scores [O] (with
   :targets) :pools}. batch (int32 arrays but :tokens):
     :tokens     R rows of one id, or an int array [R 1]
     :positions  [R] each row's position in its sequence
     :slots      [R] where each row's key and value are written
                 (gen.llm.paged/prepare's)
     :tables     [D MB] the block tables of the first D rows (decode
                 rows), each attending over its first position + 1
                 positions by paged attention (absent when D is 0)
     :chunk      [1 MB] the block table of the sequence the other R - D
                 rows read (a prefill chunk at consecutive positions),
                 which attend over its blocks gathered, each to its own
                 position (absent when D is R)
     :out        [O] the rows whose logits are computed
     :targets    [O] (optional) a token for each: :scores, its log-
                 probability at that row (in the model's dtype).
   Every row's key and value are written before any row attends, so a
   chunk's rows see each other's. Its logits are paged-step's (decode
   rows) and a concatenating cache's (a chunk), within float round-off;
   the pools are values, as paged-step's."
  [{:keys [config weights]} pools batch]
  (check-options :paged-batch-step batch batch-keys)
  (let [b (batch-arrays batch (:vocab config))]
    (check-batch config pools b)
    (batch-forward config weights pools b)))

(defn compile-paged-batch-step
  "paged-batch-step compiled by array/compile: a function (f pools batch)
   → {:logits :scores :pools}, one trace per batch's structure and shapes
   (decode rows, chunk rows, table widths, rows read, pool size). The
   model's weights are evaluated here. Options: :donate? true, each call
   consumes the pools it steps (on MLX, written in place), as
   compile-paged-step's."
  ([m] (compile-paged-batch-step m nil))
  ([{:keys [config weights]} {:keys [donate?] :as opts}]
   (check-options :compile-paged-batch-step opts #{:donate?})
   (array/eval! weights)
   (let [f (array/compile (fn [pools b] (batch-forward config weights pools b))
                          (when donate? {:donate #{0}}))]
     (fn [pools batch]
       (check-options :paged-batch-step batch batch-keys)
       (let [b (batch-arrays batch (:vocab config))]
         (check-batch config pools b)
         (f pools b))))))

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
   (check-options :prefill opts #{:logits :capacity})
   (if capacity
     (let [ids (token-array tokens (:vocab (:config m)))]
       (run m (static-cache m (first (array/shape ids)) capacity) ids logits))
     (run m nil tokens logits))))

(defn step
  "[logits cache']: the logits of tokens (B rows of T ids, one or a
   chunk) read after cache (row b's from position (offsets cache)[b]; nil
   reads from 0, as prefill), and the cache after them, of cache's kind.
   Options: :logits, as prefill's."
  ([m cache tokens] (step m cache tokens nil))
  ([m cache tokens {:keys [logits] :or {logits :all} :as opts}]
   (check-options :step opts #{:logits})
   (run m cache tokens logits)))

(defn compile-step
  "step over static caches, compiled by array/compile: a function
   (f cache tokens) → [logits cache'], the values (step m cache tokens
   opts) gives (within float32's round-off: MLX fuses the step's
   elementwise ops). Its arrays are the cache's and the tokens', so a
   trace serves every offset: on MLX one trace per shape of tokens (a
   decode step, a prefill chunk) and cache, replayed; elsewhere step
   itself. The host checks a call's tokens, rows and capacity first, as
   step's. A concatenating cache is loud (:invalid-args): its shapes grow
   every step. The model's weights are evaluated here (array/eval!), so a
   trace holds them as constants rather than the casts that made them.
   Options:
     :logits   as step's;
     :donate?  true: each call consumes the cache it steps (array/compile's
               :donate, experimental). On MLX its arrays are then donated,
               so the step writes its slots in place instead of copying
               the cache, and reading the consumed cache again is loud
               (:donated); where compile is step itself (the JVM backend)
               nothing is caught: never step a consumed cache again (fork
               one first with select-rows, a copy). The values are the
               same either way."
  ([m] (compile-step m nil))
  ([{:keys [config weights]} {:keys [logits donate?] :or {logits :all} :as opts}]
   (check-options :compile-step opts #{:logits :donate?})
   (check-logits logits)
   (array/eval! weights)
   (let [f (array/compile (fn [layers positions ids] (static-forward config weights layers positions ids logits))
                          (when donate? {:donate #{0 1}}))]
     (fn [cache tokens]
       (check-static :compile-step cache)
       (let [ids (token-array tokens (:vocab config))
             [b t] (array/shape ids)]
         (check-rows cache b)
         (check-capacity cache t)
         (let [[out layers positions] (f (:layers cache) (:positions cache) ids)]
           [out (advanced cache layers positions t)]))))))

;; ---------------------------------------------------------------------------
;; The family, by its architecture (gen.llm.models)
;; ---------------------------------------------------------------------------

(defmethod models/log-probs "qwen3" [m tokens & [opts]] (log-probs m tokens opts))

(defmethod models/nll "qwen3" [m tokens & [opts]] (nll m tokens opts))
