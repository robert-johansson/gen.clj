(ns gen.llm.models.qwen35-test
  "Qwen3.5's forward over gen.array (gen.llm.models.qwen35, experimental),
   on the tiny random Qwen3.5 and Qwen3.5-MoE of test/fixtures/gguf
   (generate.py: vocabulary 128, hidden 32, 3 GatedDeltaNet blocks of 2
   key and 4 value heads of 32, then a gated attention block of 4 query
   and 2 KV heads of 16, RoPE on 8; the MoE's 4 experts, 2 used, and a
   shared one), as F32 and as Q8_0, on the backend under test:
   1. a cached prefill, a chunk and steps give the full forward's logits
      (on the JVM backend exactly), and a read of 64 tokens or more (the
      gated delta rule's chunked form) the token loop's;
   2. a batch of contexts gives each context's logits alone (exactly);
   3. the tape's gradients through the forward match central differences
      (the JVM backend, float64): of GatedDeltaNet's weights (the
      projection, the conv kernel, ssm_a, the dt bias), the attention's,
      the embedding, the router's and the shared expert's (F32), and the
      norms through the quantized products (Q8_0);
   4. a cache is a value: continued twice with one token it gives the same
      logits, with another token others, and stays as it was;
   5. on MLX, the JVM backend's logits within float32's band (below);
   6. training's log-probs are the forward's log-softmax at the next
      tokens, plain, chunked and checkpointed, and at a temperature its
      logits divided by it (temperature 1 the model's own, bit for bit).
   Each of 1-4 fails under a planted bug, run here as a negative control
   (with-redefs-fn, asserted as more than 0.01): a cached read's positions
   shifted by one, its KV cache dropped, its recurrent state dropped, the
   conv state keeping its oldest inputs (not its newest), the mask not
   causal (1); the norm reading the batch's other rows (2); a step's
   token ignored (another token's logits then differ by 0) (4); a
   gradient against another entry's difference (3); the temperature
   ignored (6).
   A wrong convention is self-consistent: the conv kernel applied newest
   input first, the query and the output gate swapped, KV heads tiled
   pass 1-4, so the real model's logits are checked against llama.cpp's
   (gen.llm.models.qwen35-llama-test).
   The bands: on the JVM backend 0 for 1, 2 and 4 (the same ops in the
   same order), and 1e-12 for a read in chunks against the token loop
   (float64's round-off: measured ≤ 1.1e-13). On MLX, 2e-4 in a logit
   (|logit| <= 5 here) for 1, 2, 4 and 5: four times the largest
   difference measured (Mac mini M4, 2026-11-07, Metal, the kernel
   computing the rule), 5.0e-5 (a batch against each row alone, Q8_0;
   MLX against the JVM's float64 4.0e-5 over 4 contexts); float32's
   round-off through 4 blocks. A long read's, 5e-4: 4.5 times the 1.1e-4
   measured between 80 tokens in chunks and token by token, float32's
   round-off over 80 updates of the recurrent states.
   The planted bugs' margins (JVM, F32): positions shifted 0.24, the KV
   cache dropped 1.2, the recurrent state dropped 3.8, the conv state's
   oldest inputs 5.5, not causal 0.32 (1); a long read's triangular
   inverse dropped 2.8 (1); the norm across rows 5.1 (2); gradients'
   relative errors measured ≤ 1.7e-8 (3); the temperature ignored 1.9
   (6)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.nn]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen35-tiny-q8_0.gguf"
   :moe-f32 "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"
   :moe-q8_0 "test/fixtures/gguf/qwen35moe-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen35/load f)])) files)))

(defn- host [x] (array/->clj x))

(def ^:private ctx [5 17 42 3 88 11 60 2 9])
(def ^:private other [5 17 42 3 88 11 60 2 70])
(def ^:private long-ctx (mapv #(mod (+ 7 (* 41 %)) 128) (range 80)))

(defn- row
  "Position i of logits [B T V], as [B 1 V]."
  [x i]
  (array/slice x 1 i (inc i)))

(defn- max-diff ^double [a b]
  (double (array/item (array/max (array/abs (array/- a b))))))

(def ^:private band
  "The largest |Δ logit| two computations of one distribution differ by
   on the backend under test: exactly none on the JVM backend (float64,
   the same ops in the same order), float32's round-off on MLX (the
   namespace's docstring)."
  (if (ts/jvm?) 0.0 2e-4))

(def ^:private chunk-band
  "… between a long read (80 tokens: the gated delta rule in chunks, or
   MLX's kernel) and the token loop: float64's round-off on the JVM
   backend, float32's on MLX."
  (if (ts/jvm?) 1e-12 5e-4))

(defn- cache-gap
  "max |Δ logit| between the full forward of ctx and its cached reading:
   a prefill of 4, a chunk of 3 and two steps."
  [m]
  (let [full (qwen35/forward m [ctx])
        [_ c4] (qwen35/prefill m [(subvec ctx 0 4)])
        [chunk c7] (qwen35/step m c4 [(subvec ctx 4 7)])
        [s7 c8] (qwen35/step m c7 [[(ctx 7)]])
        [s8 _] (qwen35/step m c8 [[(ctx 8)]])]
    (max (max-diff (array/slice full 1 4 7) chunk) (max-diff (row full 7) s7) (max-diff (row full 8) s8))))

(defn- with-method
  "(f) with gen.array/gated-delta computed by method (:loop or :chunked)
   whatever the length: its composite, never a backend's kernel."
  [method f]
  (let [gd array/gated-delta]
    (with-redefs-fn {#'array/gated-delta (fn [q k v g beta s & _] (gd q k v g beta s {:method method}))} f)))

(defn- long-gap
  "max |Δ logit| between readings of the long context (80 tokens) and its
   reading token by token (the rule's :loop): read whole in chunks of 64
   (:chunked), read whole as the backend reads it (:auto: in chunks, or
   by MLX's kernel), and as a prefill of 70 and 10 steps after it."
  [m]
  (let [looped (with-method :loop #(qwen35/forward m [long-ctx]))
        chunked (with-method :chunked #(qwen35/forward m [long-ctx]))
        [_ c] (qwen35/prefill m [(subvec long-ctx 0 70)])
        [steps] (qwen35/step m c [(subvec long-ctx 70 80)])]
    (max (max-diff chunked looped) (max-diff (qwen35/forward m [long-ctx]) looped)
         (max-diff (array/slice looped 1 70 80) steps))))

(defn- batch-gap
  "max |Δ logit| between a batch of ctx and other and each read alone."
  [m]
  (let [both (qwen35/forward m [ctx other])]
    (max (max-diff (array/slice both 0 0 1) (qwen35/forward m [ctx]))
         (max-diff (array/slice both 0 1 2) (qwen35/forward m [other])))))

(defn- fork-gaps
  "[|Δ| of one cache continued twice with the same token, |Δ| of it
   continued with another token, |Δ| of the cache's own logits read again
   after both]."
  [m]
  (let [[_ c8] (qwen35/prefill m [(subvec ctx 0 8)])
        [a _] (qwen35/step m c8 [[(ctx 8)]])
        [b _] (qwen35/step m c8 [[(other 8)]])
        [a' _] (qwen35/step m c8 [[(ctx 8)]])]
    [(max-diff a a') (max-diff a b) (max-diff (row (qwen35/forward m [ctx]) 8) a')]))

;; ---------------------------------------------------------------------------
;; The configuration and the tensors
;; ---------------------------------------------------------------------------

(def ^:private layers-config
  {:hidden 32 :layers 4 :kinds [:delta :delta :delta :attention] :heads 4 :kv-heads 2 :head-dim 16 :rope-dims 8
   :rope-base 10000.0 :eps (double (float 1e-6)) :context 256 :delta-key-heads 2 :delta-value-heads 4
   :delta-key-dim 32 :delta-value-dim 32 :delta-conv 4 :vocab 128})

(deftest the-config-is-the-gguf-s
  (let [{:keys [f32 q8_0 moe-f32 moe-q8_0]} @models]
    (is (= (assoc layers-config :arch "qwen35" :mtp-layers 1 :ffn 64 :dtype (ts/float-dtype)) (:config f32) (:config q8_0)))
    (is (= (assoc layers-config :arch "qwen35moe" :mtp-layers 0 :experts 4 :experts-used 2 :expert-ffn 32 :shared-ffn 32
                  :dtype (ts/float-dtype))
           (:config moe-f32) (:config moe-q8_0)))
    (is (map? (get-in q8_0 [:weights :layers 0 :qkv])) "Q8_0's matrices stay quantized")
    (is (= [4 32 1] (array/shape (get-in moe-q8_0 [:weights :layers 0 :moe :gate :scales])))
        "a stack of quantized experts [experts out in/32]")
    (is (nil? (get-in f32 [:weights :output])) "a tied output")
    (let [md (:metadata (array/load-gguf (:f32 files)))
          moe-md (:metadata (array/load-gguf (:moe-f32 files)))
          reason (fn [m] (:reason (ts/thrown-data (qwen35/config m))))]
      (is (= :unsupported (reason (assoc md "general.architecture" "qwen3"))))
      (is (= :unsupported (reason (assoc md "qwen35.rope.scaling.type" "yarn"))))
      (is (= :unsupported (reason (assoc md "qwen35.rope.dimension_sections" [2 1 0 0]))) "a text token's dims unrotated")
      (is (= :unsupported (reason (assoc md "qwen35.rope.dimension_sections" [1 1 1 1]))))
      (is (= :unsupported (reason (dissoc md "qwen35.rope.dimension_sections"))))
      (is (= :unsupported (reason (assoc md "qwen35.attention.value_length" 8))))
      (is (= :unsupported (reason (assoc moe-md "qwen35moe.expert_group_count" 2))))
      (is (= :invalid-args (reason (dissoc md "qwen35.ssm.state_size"))))
      (is (= :invalid-args (reason (assoc md "qwen35.attention.head_count_kv" 3))))
      (is (= :invalid-args (reason (assoc md "qwen35.ssm.group_count" 3))) "value heads not a multiple of the key heads")
      (is (= :invalid-args (reason (assoc moe-md "qwen35moe.expert_used_count" 5))))
      (testing "the layers' kinds: every 4th attention, or as attention.recurrent_layers says"
        (is (= [:delta :attention :delta :attention] (:kinds (qwen35/config (assoc md "qwen35.full_attention_interval" 2)))))
        (is (= [:attention :delta :delta :delta]
               (:kinds (qwen35/config (assoc md "qwen35.attention.recurrent_layers" [false true true true false])))))))))

(deftest every-tensor-is-read-and-checked
  (let [gguf (array/load-gguf (:f32 files))
        t (:tensors gguf)
        data (fn [g] (ts/thrown-data (qwen35/model g)))]
    (is (some #(.startsWith ^String % "blk.4.nextn") (keys t)) "the fixture holds an MTP block, recognized and not read")
    (is (= {:reason :unsupported :tensors ["blk.0.attn_q.bias"]}
           (select-keys (data (assoc-in gguf [:tensors "blk.0.attn_q.bias"] (array/zeros [64]))) [:reason :tensors]))
        "a tensor the forward does not read")
    (is (= :invalid-args (:reason (data (update gguf :tensors dissoc "blk.1.ssm_alpha.weight")))) "a tensor missing")
    (is (= :invalid-args (:reason (data (assoc-in gguf [:tensors "blk.3.attn_q.weight"] (t "blk.3.attn_k.weight")))))
        "a tensor of another shape")
    (is (= :unsupported (:reason (data (assoc-in gguf [:tensors "blk.0.ssm_conv1d.weight"]
                                                 {:wq (array/zeros [256 1] {:dtype :uint32}) :scales (array/zeros [256 1])
                                                  :biases (array/zeros [256 1]) :bits 8 :group-size 32}))))
        "a quantized tensor the forward reads dense")
    (is (= :invalid-args (:reason (ts/thrown-data (qwen35/model gguf {:dtype :int32})))))
    (testing "an output of its own (untied): the embedding as one gives the tied model's logits"
      (let [tied (:f32 @models)
            untied (qwen35/model (assoc-in gguf [:tensors "output.weight"] (t "token_embd.weight")))]
        (is (some? (get-in untied [:weights :output])))
        (is (= (host (qwen35/forward tied [ctx])) (host (qwen35/forward untied [ctx]))))))))

;; ---------------------------------------------------------------------------
;; 1. A cached reading, and a long one
;; ---------------------------------------------------------------------------

(deftest a-cached-reading-is-the-full-forward
  (doseq [[k m] @models]
    (testing (name k)
      (is (<= (cache-gap m) band))
      (let [[last-row] (qwen35/prefill m [ctx] {:logits :last})
            [steps c] (qwen35/step m (second (qwen35/prefill m [(subvec ctx 0 3)])) [(subvec ctx 3 9)] {:logits :last})]
        (is (= [1 1 128] (array/shape last-row)))
        (is (<= (max-diff last-row (row (qwen35/forward m [ctx]) 8)) band) ":logits :last is the last row")
        (is (<= (max-diff steps last-row) band))
        (is (= 9 (qwen35/offset c)))
        (is (= [1 2 9 16] (array/shape (get-in c [:layers 3 :k]))) "an attention layer's [B KV-heads positions head-dim]")
        (is (= [1 3 256] (array/shape (get-in c [:layers 0 :conv]))) "the conv state [B 3 channels]")
        (is (= [1 4 32 32] (array/shape (get-in c [:layers 1 :state]))) "the recurrent state [B value-heads Dv Dk]"))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          attention @#'qwen35/attention
          delta-net @#'qwen35/delta-net
          sdpa array/sdpa]
      (doseq [[what redefs] {"a cached read's positions shifted by one"
                             {#'qwen35/attention (fn [cfg w x offset entry] (attention cfg w x (if entry (inc offset) offset) entry))}
                             "a cached read's KV cache dropped"
                             {#'qwen35/attention (fn [cfg w x offset _] (attention cfg w x offset nil))}
                             "a cached read's recurrent state dropped"
                             {#'qwen35/delta-net (fn [cfg w x entry] (delta-net cfg w x (dissoc entry :state)))}
                             "the conv state keeping its oldest inputs, not its newest"
                             {#'qwen35/conv-state (fn [padded _] (array/slice padded 1 0 3))}
                             "attention not causal"
                             {#'array/sdpa (fn [q k v opts] (sdpa q k v (dissoc opts :mask)))}}]
        (is (< 0.01 (with-redefs-fn redefs #(cache-gap m))) what)))))

(deftest a-long-read-in-chunks-is-the-token-loop
  (doseq [[k m] @models]
    (is (<= (long-gap m) chunk-band) (name k)))
  (testing "a planted bug fails it: the chunks' triangular inverse dropped"
    (is (< 0.01 (with-redefs-fn {#'gen.array.nn/unit-lower-inverse
                                 (fn [_ a] (array/broadcast-to (array/eye (peek (array/shape a))) (array/shape a)))}
                  #(long-gap (:f32 @models)))))))

;; ---------------------------------------------------------------------------
;; 2. A batch, 4. a value
;; ---------------------------------------------------------------------------

(deftest a-batch-is-each-context-alone
  (doseq [[k m] @models]
    (is (<= (batch-gap m) band) (name k)))
  (testing "a planted bug fails it: the norm reading the batch's other rows"
    (let [rms-norm array/rms-norm]
      (is (< 0.01 (with-redefs-fn {#'array/rms-norm (fn [x w eps] (rms-norm (array/+ x (array/mean x {:axis 0 :keep-dims? true})) w eps))}
                    #(batch-gap (:f32 @models))))))))

(deftest a-cache-is-a-value
  (doseq [[k m] @models]
    (let [[same other-token again] (fork-gaps m)]
      (is (= 0.0 same) (str (name k) ": the same token twice from one cache"))
      (is (< 0.1 other-token) (str (name k) ": another token's logits"))
      (is (<= again band) (str (name k) ": the cache unchanged by its continuations"))))
  (testing "a planted bug fails it: a step's token ignored"
    (let [embed common/embed]
      (is (= 0.0 (second (with-redefs-fn {#'gen.llm.models.common/embed
                                          (fn [w tokens dtype]
                                            (embed w (if (= 1 (second (array/shape tokens))) (array/zeros-like tokens) tokens) dtype))}
                           #(fork-gaps (:f32 @models)))))))))

;; ---------------------------------------------------------------------------
;; 3. Gradients (the JVM backend, float64)
;; ---------------------------------------------------------------------------

(defn- log-p
  "log p(ctx[8] | ctx[0..7]) under the model m."
  [m]
  (let [[lg] (qwen35/prefill m [(subvec ctx 0 8)] {:logits :last})]
    (array/item (array/sum (array/pick (array/log-softmax lg) (array/array [[(ctx 8)]] {:dtype :int32}))))))

(defn- log-p-of
  "log p as a function of the weights at paths (into the model)."
  [m paths]
  (fn [& ws]
    (let [[lg] (qwen35/prefill (reduce (fn [m [p w]] (assoc-in m p w)) m (map vector paths ws)) [(subvec ctx 0 8)] {:logits :last})]
      (array/sum (array/pick (array/log-softmax lg) (array/array [[(ctx 8)]] {:dtype :int32}))))))

(defn- central-difference
  "d log p / d (the entry ij of the weight at path), by central differences."
  [m path ij]
  (let [w (host (get-in m path))
        at (fn [d] (log-p (assoc-in m path (array/array (update-in w ij + d) {:dtype (array/dtype (get-in m path))}))))
        h 1e-5]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- worst-relative-error
  "The largest error of the tape's gradient a against central differences
   b, |a - b| / max(|b|, 1e-3), over entries of the weights at paths."
  [m paths entries]
  (let [grads (apply (array/grad (log-p-of m paths) {:argnums (vec (range (count paths)))})
                     (map #(get-in m %) paths))]
    (reduce max (for [[path g] (map vector paths grads)
                      ij (entries path)]
                  (let [a (get-in (host g) ij)
                        b (central-difference m path ij)]
                    (/ (Math/abs (- a b)) (max 1e-3 (Math/abs b))))))))

(deftest gradients-are-central-differences
  ;; the band 1e-6, above central differences' round-off at h = 1e-5. A
  ;; gradient compared with another entry's difference is off by O(1)
  (if-not (ts/jvm?)
    (println "gen.llm.models.qwen35-test: the gradients are checked on the JVM backend (float64)")
    (let [{:keys [f32 q8_0 moe-f32]} @models
          w (fn [& ks] (into [:weights :layers] ks))
          entries {(w 0 :qkv) [[0 0] [200 31]] (w 1 :conv) [[3 0] [130 3]] (w 0 :a) [[1] [3]] (w 2 :dt-bias) [[0] [2]]
                   (w 1 :alpha) [[2 7]] (w 3 :q) [[17 5] [100 30]] [:weights :embed] [[(ctx 2) 4] [(ctx 8) 9]]
                   (w 0 :moe :router) [[0 3] [3 30]] (w 3 :moe :shared :down) [[5 7]] (w 2 :moe :shared-gate) [[11]]
                   (w 0 :ssm-norm) [[0] [31]] (w 3 :q-norm) [[2]] [:weights :norm] [[7]]}]
      (is (< (worst-relative-error f32 [(w 0 :qkv) (w 1 :conv) (w 0 :a) (w 2 :dt-bias) (w 1 :alpha) (w 3 :q) [:weights :embed]]
                                   entries)
             1e-6)
          "F32: GatedDeltaNet's weights, the attention's, the embedding")
      (is (< (worst-relative-error moe-f32 [(w 0 :moe :router) (w 3 :moe :shared :down) (w 2 :moe :shared-gate)] entries) 1e-6)
          "F32 MoE: the router, the shared expert and its gate")
      (is (< (worst-relative-error q8_0 [(w 0 :ssm-norm) (w 3 :q-norm) [:weights :norm]] entries) 1e-6)
          "Q8_0: norms, through the quantized products")
      (testing "a planted bug fails it: the gradient of one entry against another's difference"
        (let [path (w 1 :conv)
              g (host ((array/grad (log-p-of f32 [path])) (get-in f32 path)))]
          (is (< 1e-3 (/ (Math/abs (- (get-in g [3 0]) (central-difference f32 path [3 1])))
                         (Math/abs (central-difference f32 path [3 1]))))))))))

(deftest top-k-by-ranks-is-top-k-by-picks
  ;; the router's top k by ranks (a decode step's) against k argmaxes,
  ;; index for index: probabilities, and entries of 3 levels (ties in
  ;; every row: the first of equal entries first), up to k = E. Planted
  ;; bugs: ties broken last first fails the levels' cases; a rank counting
  ;; equal entries after it too (>= for >) fails them; values read at the
  ;; next index fail every case
  (let [ranked @#'qwen35/ranked
        picked @#'qwen35/picked
        top-k @#'qwen35/top-k
        cases (for [[i [n e k]] (map-indexed vector [[1 4 2] [6 16 8] [5 256 8] [3 9 9] [2 7 1]])
                    levels? [false true]]
                (let [u (rng/uniform (rng/key i) [n e])]
                  [[n e k levels?] (if levels? (array/floor (array/* 3 u)) (array/softmax (array/log u) {:axis -1})) k]))]
    (doseq [[what p k] cases]
      (testing (pr-str what)
        (let [want (picked p k)
              [idx vals] (top-k p k)]
          (is (= :int32 (array/dtype (ranked p k)) (array/dtype idx)))
          (is (= (ts/host want) (ts/host (ranked p k)) (ts/host idx)))
          (is (= (ts/host (array/take-along-axis p want -1)) (ts/host vals))))))
    (testing "beyond ranked-limit, top-k picks"
      (with-redefs [qwen35/ranked-limit 0]
        (let [[_ p k] (last cases)]
          (is (= (ts/host (picked p k)) (ts/host (first (top-k p k))))))))))

(deftest a-model-loads-in-its-dtype-a-tensor-at-a-time
  ;; qwen35/load makes each tensor in the model's dtype as it loads
  ;; (gen.array/load-gguf's :dtype; the router float32 under a half
  ;; float), never the whole file in its own dtype first: its weights are
  ;; model's of the file loaded as it is, then cast, bit for bit (dtype,
  ;; shape and each value's bits). Planted bug: the routers loaded in the
  ;; half float fails here (2026-11-08)
  (let [bits (fn [x] (if (or (nil? x) (number? x))
                       x
                       [(array/dtype x) (array/shape x)
                        (mapv #(Double/doubleToRawLongBits (double %)) (flatten [(array/->clj x)]))]))
        dtypes (if (ts/jvm?) [:float32 :float64] [:float16 :bfloat16 :float32])]
    (doseq [k [:f32 :q8_0 :moe-f32 :moe-q8_0]
            dtype dtypes]
      (testing (str (name k) " in " (name dtype))
        (let [mine (qwen35/load (files k) {:dtype dtype})
              theirs (qwen35/model (array/load-gguf (files k)) {:dtype dtype})]
          (is (= (:config theirs) (:config mine)))
          (is (= (mapv bits (array/tree-leaves (:weights theirs))) (mapv bits (array/tree-leaves (:weights mine))))))))))

(deftest a-half-float-router-computes-in-float32
  ;; genclj-fxag: under a half float the router's weights load in float32
  ;; and its product is float32's, so that its top k orders experts as
  ;; float32 does, not as logits rounded to a half float's ulp (2^-7 at 8
  ;; in float16). Logits 8 and 8 + 2^-9 are apart in float32 and one
  ;; value in a half float. Planted bugs (MLX's CUDA, 2026-11-08):
  ;; router-dtype the model's dtype fails 3 checks (its own, the loaded
  ;; weights' in both half floats); the product in the model's dtype, the
  ;; logits widened after (the code before genclj-fxag), fails the
  ;; near-tie's 2
  (let [router-dtype @#'qwen35/router-dtype
        router-logits @#'qwen35/router-logits
        top-k @#'qwen35/top-k]
    (is (= :float32 (router-dtype :float16) (router-dtype :bfloat16) (router-dtype :float32)))
    (is (= :float64 (router-dtype :float64)))
    (is (= (ts/float-dtype) (array/dtype (get-in @models [:moe-f32 :weights :layers 0 :moe :router])))
        "a model in the backend's float keeps its router in it")
    (if (ts/jvm?)
      (println "qwen35-test: the router under half floats is checked on MLX (the JVM backend has none)")
      (doseq [half [:float16 :bfloat16]]
        (testing (name half)
          (let [layers (get-in (qwen35/load (:moe-f32 files) {:dtype half}) [:weights :layers])
                xs (array/astype (array/array [[1.0 1.0]]) half)
                router (array/array [[4.0 4.0] [4.0 (+ 4.0 (Math/pow 2 -9))]] {:dtype :float32})
                logits (router-logits half router xs)]
            (is (every? #(= :float32 (array/dtype (get-in % [:moe :router]))) layers) "the router's weights")
            (is (= half (array/dtype (get-in layers [0 :moe :shared-gate]))) "the rest in the model's dtype")
            (is (= :float32 (array/dtype logits)) "the logits")
            (is (= [[1]] (ts/host (first (top-k (array/softmax logits {:axis -1}) 1)))) "the larger logit's expert")))))))

(deftest a-read-with-the-reference-s-experts-is-its-own
  ;; against-llama's forced routing (genclj-fxag), which
  ;; cuda-reference-test's float16 check of a mixture of experts reads:
  ;; the experts a reference read records (each position's, each layer's
  ;; top k), forced on the same model, give its own readings (whole, a
  ;; chunk, steps); on the JVM backend exactly. Other experts move them:
  ;; forced one off (each index + 1 mod E), the worst reading 4.88 nats.
  ;; Planted bug: each forward forced with the next position's experts
  ;; fails the own check (4.63; JVM, 2026-11-08)
  (let [m (:moe-f32 @models)
        {:keys [experts experts-used]} (:config m)
        r (against/reference m (subvec ctx 0 6) 3)
        band (if (ts/jvm?) 0.0 2e-4)
        read-with (fn [forced] (mapv :max-diff (against/compare-readings m (assoc r :forced forced) 0)))
        one-off (mapv (fn [layers] (mapv (fn [idx] (mapv #(mod (inc (long %)) (long experts)) idx)) layers)) (:experts r))]
    (is (= 9 (count (:experts r))) "the prompt's 6 positions and 3 steps")
    (is (every? (fn [layers] (and (seq layers) (every? #(= experts-used (count %)) layers))) (:experts r)))
    (is (every? #(<= % band) (read-with (:experts r))) "the reference's experts")
    (is (< 0.5 (reduce max (read-with one-off))) "experts one off (a planted bug)")))

(deftest fewer-kernels-are-the-old-forms
  ;; genclj-yesw: the decode step's forms that make fewer kernels on MLX
  ;; equal the forms they replaced, value for value (pure data movement
  ;; and the same elementwise ops), on every backend:
  ;; - the key heads tiled by one broadcast's copy, not a concatenation of
  ;;   the repeats; and under a half float, cast before tiling is cast
  ;;   after it;
  ;; - the query's and key's heads l2-normed in one norm, not two;
  ;; - the attention's output times sigmoid of its gate read in place in
  ;;   the projection, not made whole first: for a paged step's [R heads
  ;;   head-dim] output and a chunk's [B T heads head-dim] (sdpa's,
  ;;   transposed).
  ;; Planted bugs in the new forms fail it (on the JVM backend): the heads
  ;; repeated grouped (each key head's repeats together, the GQA order)
  ;; 6 checks; the query and key swapped 2; the gate unsquashed (no
  ;; sigmoid) 2. Each case also tells a planted bug in the old form apart
  ;; from the new.
  (let [tiled @#'qwen35/key-heads-tiled
        gated @#'qwen35/gated
        query-and-gate @#'qwen35/query-and-gate
        concatenated (fn [y hv] (array/concatenate (vec (repeat (quot (long hv) (long (nth (array/shape y) 2))) y)) 2))
        grouped (fn [y hv] (let [[b t hk d] (array/shape y)]
                             (array/reshape (array/broadcast-to (array/expand-dims y 3) [b t hk (quot (long hv) (long hk)) d])
                                            [b t hv d])))]
    (testing "the key heads tiled"
      (doseq [[i hv] [[1 4] [2 6]]]
        (let [y (rng/normal (rng/key i) [2 3 2 8])]
          (is (= (host (concatenated y hv)) (host (tiled y hv))) (str hv " value heads"))
          (is (not= (host (grouped y hv)) (host (tiled y hv))) "the grouped order is told apart")))
      (when-not (ts/jvm?)
        (let [y (array/astype (rng/normal (rng/key 3) [2 1 2 8]) :float16)]
          (is (= (host (array/astype (concatenated y 4) :float32)) (host (tiled (array/astype y :float32) 4)))
              "a half float cast before tiling"))))
    (testing "the query and key normed at once"
      (let [query-and-key @#'qwen35/query-and-key
            l2-norm @#'qwen35/l2-norm
            {:keys [eps delta-key-heads delta-value-heads delta-key-dim] :as cfg} (:config (:moe-f32 @models))
            [hk hv dk] [delta-key-heads delta-value-heads delta-key-dim]
            width (* hk dk)
            mixed (rng/normal (rng/key 8) [2 3 (+ (* 2 width) 64)])
            heads (fn [from to] (array/reshape (array/slice mixed 2 from to) [2 3 hk dk]))
            old-q (array/* (concatenated (l2-norm (heads 0 width) eps) hv) (/ 1.0 (Math/sqrt dk)))
            old-k (concatenated (l2-norm (heads width (* 2 width)) eps) hv)
            [q k] (query-and-key cfg mixed (array/dtype mixed))]
        (is (= (host old-q) (host q)) "q")
        (is (= (host old-k) (host k)) "k")
        (is (not= (host old-k) (host (concatenated (array/reshape (l2-norm (array/reshape (heads width (* 2 width)) [2 3 1 (* hk dk)]) eps)
                                                                   [2 3 hk dk])
                                                   hv)))
            "the key's heads normed as one are told apart")
        (when-not (ts/jvm?)
          ;; a half-float model's: normed and scaled in float16, then cast
          ;; to the rule's float32 (the old form cast after tiling)
          (let [m16 (array/astype mixed :float16)
                heads16 (fn [from to] (array/reshape (array/slice m16 2 from to) [2 3 hk dk]))
                [q16 k16] (query-and-key cfg m16 :float32)]
            (is (= (host (array/astype (array/* (concatenated (l2-norm (heads16 0 width) eps) hv) (/ 1.0 (Math/sqrt dk))) :float32))
                   (host q16))
                "q, float16 cast to float32")
            (is (= (host (array/astype (concatenated (l2-norm (heads16 width (* 2 width)) eps) hv) :float32)) (host k16))
                "k, float16 cast to float32")))))
    (testing "the attention's output gated"
      (let [cfg {:heads 2 :head-dim 4}
            [b t] [2 3]
            qg (rng/normal (rng/key 4) [b t 2 2 4])
            [_ gate] (query-and-gate qg)
            whole (fn [o] (array/* (array/reshape o [b t 8]) (array/sigmoid (array/reshape gate [b t 8]))))
            paged (rng/normal (rng/key 5) [(* b t) 2 4])
            chunk (array/transpose (rng/normal (rng/key 6) [b 2 t 4]) [0 2 1 3])]
        (is (= [b t 2 1 4] (array/shape gate)))
        (doseq [[what o] [[:paged paged] [:chunk chunk]]]
          (is (= (host (whole o)) (host (gated cfg o gate))) (name what))
          (is (not= (host (whole o)) (host (gated cfg o (first (query-and-gate qg))))) "the query's half is told apart"))))))

;; ---------------------------------------------------------------------------
;; 5. MLX against the JVM backend
;; ---------------------------------------------------------------------------

(deftest mlx-is-the-jvm-backend-within-float32
  (if (ts/jvm?)
    (println "gen.llm.models.qwen35-test: MLX is compared with the JVM backend on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [[k file] files]
        (let [mine (get @models k)
              theirs (array/with-backend jvm (qwen35/load file))
              on-jvm (fn [f] (array/with-backend jvm (host (f theirs))))
              gap (fn [f] (reduce max (map #(Math/abs (- %1 %2)) (flatten (host (f mine))) (flatten (on-jvm f)))))]
          (is (<= (gap #(qwen35/forward % [ctx other])) band) (str (name k) ": a batch's forward"))
          (is (<= (gap #(first (qwen35/step % (second (qwen35/prefill % [(subvec ctx 0 5)])) [(subvec ctx 5 9)]))) band)
              (str (name k) ": a cached chunk"))
          (is (<= (gap #(qwen35/forward % [long-ctx])) chunk-band) (str (name k) ": a long read")))))))

;; ---------------------------------------------------------------------------
;; 6. Training's log-probs
;; ---------------------------------------------------------------------------

(defn- log-probs-gap
  "max |Δ| between log-probs of [ctx other] under opts and the forward's
   log-softmax (its logits divided by the temperature t) at the next
   tokens."
  [m opts t]
  (let [ids (array/array [ctx other] {:dtype :int32})
        want (array/pick (array/log-softmax (array// (array/slice (qwen35/forward m [ctx other]) 1 0 8) t))
                         (array/slice ids 1 1 9))
        lp (qwen35/log-probs m [ctx other] opts)]
    (is (= [2 8] (array/shape lp)))
    (max-diff lp want)))

(def ^:private log-probs-band
  "log-probs against the forward's log-softmax: float64's round-off on
   the JVM backend (one softmax by chunks, the other whole), float32's
   band on MLX (the namespace's docstring)."
  (if (ts/jvm?) 1e-12 band))

(deftest log-probs-are-the-forward-s
  (let [training {:checkpoint? true :chunk 48}]
    (doseq [[k m] @models]
      (doseq [opts [nil {:chunk 48} training {:checkpoint? true}]]
        (is (<= (log-probs-gap m opts 1.0) log-probs-band) (str (name k) " " opts)))
      (doseq [opts [{:temperature 0.7} (assoc training :temperature 0.7)]]
        (is (<= (log-probs-gap m opts 0.7) log-probs-band) (str (name k) " " opts ": the logits divided by the temperature")))
      (is (= (host (qwen35/log-probs m [ctx other] training)) (host (qwen35/log-probs m [ctx other] (assoc training :temperature 1.0))))
          (str (name k) ": temperature 1 is the model's own, bit for bit")))
    (testing "a planted bug fails it: the temperature ignored"
      (is (< 0.01 (with-redefs-fn {#'common/tempered-head (fn [_] common/head-logits)}
                    #(log-probs-gap (:f32 @models) {:temperature 0.7} 0.7)))))))

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:q8_0 @models)
        reason (fn [f] (:reason (ts/thrown-data (f))))]
    (is (= :invalid-token (reason #(qwen35/forward m [[1 2 128]]))))
    (is (= :invalid-token (reason #(qwen35/forward m [[1 -1]]))))
    (is (= :invalid-args (reason #(qwen35/forward m [[1 2] [3]]))) "ragged rows")
    (is (= :invalid-args (reason #(qwen35/forward m [[]]))))
    (is (= :invalid-args (reason #(qwen35/forward m [1 2]))) "a row, not rows")
    (is (= :invalid-args (reason #(qwen35/prefill m [[1 2]] {:logits :first}))))
    (is (= :unknown-options (reason #(qwen35/prefill m [[1 2]] {:last? true}))))
    (is (= :unknown-options (reason #(qwen35/model (array/load-gguf (:q8_0 files)) {:dtpye :float32}))))
    (is (= :invalid-args (reason #(qwen35/step m (second (qwen35/prefill m [[1 2]])) [[3] [4]]))) "a cache of other rows")
    (is (= :invalid-args (reason #(qwen35/log-probs m [ctx] {:temperature 0}))))
    (is (= :invalid-args (reason #(qwen35/log-probs m [ctx] {:temperature ##Inf}))))
    (is (= :unknown-options (reason #(qwen35/nll m [ctx] {:temperature 0.7}))) "nll is at the model's own temperature")
    (is (= [2 3 128] (array/shape (qwen35/forward m (array/array [[1 2 3] [4 5 6]] {:dtype :int32})))) "an int array")))
