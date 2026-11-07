(ns gen.llm.models.qwen3-test
  "Qwen3's forward over gen.array (gen.llm.models.qwen3, experimental), on
   the tiny random Qwen3 of test/fixtures/gguf (generate.py: vocabulary
   128, hidden 64, 3 blocks, 4 query and 2 KV heads of 16), as F32 and as
   Q8_0, on the backend under test:
   1. a cached prefill, a chunk and steps give the full forward's logits
      (on the JVM backend exactly);
   2. a batch of contexts gives each context's logits alone (exactly);
   3. the tape's gradients through the forward match central differences
      (the JVM backend, float64): of a dense weight and the embedding (F32),
      and of the norms through the quantized products (Q8_0);
   4. a cache is a value: continued twice with one token it gives the same
      logits, with another token others, and stays as it was;
   5. on MLX, the JVM backend's logits within float32's band (below).
   Each of 1-4 fails under a planted bug, run here as a negative control
   (with-redefs-fn), by (2026-11-07, every backend alike): a cached read's
   positions shifted by one (0.98 in a logit), its cache dropped (7.7),
   the mask not causal (3.5) (1); the norm reading the batch's other rows
   (2.5) (2); a step's token ignored (another token's logits then differ
   by 0) (4); a gradient against another entry's difference (3).
   A wrong convention is self-consistent: KV heads tiled instead of
   grouped, RoPE's traditional pairs for NEOX's or the q/k norms skipped
   pass 1-4, so the real model's logits are checked against llama.cpp's
   (gen.llm.models.qwen3-llama-test).
   The band on MLX, 4e-5 in a logit (|logit| <= 6 here), for 1, 2, 4 and
   5: four times the largest difference measured (Mac mini M4,
   2026-11-07), 9.0e-6 between MLX and the JVM backend's float64 over 12
   contexts on Metal (4.1e-6 on MLX's CPU), and 6.2e-6 between two
   computations on MLX; float32's round-off through 3 blocks. On the JVM
   backend the band is 0: the same ops in the same order."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(defn- host [x] (array/->clj x))

(def ^:private ctx [5 17 42 3 88 11 60 2 9])
(def ^:private other [5 17 42 3 88 11 60 2 70])

(defn- row
  "Position i of logits [B T V], as [B 1 V]."
  [x i]
  (array/slice x 1 i (inc i)))

(defn- max-diff ^double [a b]
  (double (array/item (array/max (array/abs (array/- a b))))))

(def ^:private band
  "The largest |Δ logit| two computations of one distribution differ by
   on the backend under test: exactly none on the JVM backend (float64,
   the same ops in the same order), float32's round-off on MLX."
  (if (ts/jvm?) 0.0 4e-5))

(defn- cache-gap
  "max |Δ logit| between the full forward of ctx and its cached reading:
   a prefill of 4, a chunk of 3 and two steps."
  [m]
  (let [full (qwen3/forward m [ctx])
        [_ c4] (qwen3/prefill m [(subvec ctx 0 4)])
        [chunk c7] (qwen3/step m c4 [(subvec ctx 4 7)])
        [s7 c8] (qwen3/step m c7 [[(ctx 7)]])
        [s8 _] (qwen3/step m c8 [[(ctx 8)]])]
    (max (max-diff (array/slice full 1 4 7) chunk) (max-diff (row full 7) s7) (max-diff (row full 8) s8))))

(defn- batch-gap
  "max |Δ logit| between a batch of ctx and other and each read alone."
  [m]
  (let [both (qwen3/forward m [ctx other])]
    (max (max-diff (array/slice both 0 0 1) (qwen3/forward m [ctx]))
         (max-diff (array/slice both 0 1 2) (qwen3/forward m [other])))))

(defn- fork-gaps
  "[|Δ| of one cache continued twice with the same token, |Δ| of it
   continued with another token, |Δ| of the cache's own logits read again
   after both]."
  [m]
  (let [[_ c8] (qwen3/prefill m [(subvec ctx 0 8)])
        [a _] (qwen3/step m c8 [[(ctx 8)]])
        [b _] (qwen3/step m c8 [[(other 8)]])
        [a' _] (qwen3/step m c8 [[(ctx 8)]])]
    [(max-diff a a') (max-diff a b) (max-diff (row (qwen3/forward m [ctx]) 8) a')]))

(deftest the-config-is-the-gguf-s
  (let [{:keys [f32 q8_0]} @models]
    (is (= {:hidden 64 :layers 3 :heads 4 :kv-heads 2 :head-dim 16 :ffn 128 :eps (double (float 1e-6))
            :rope-base 10000.0 :context 256 :vocab 128 :dtype (ts/float-dtype)}
           (:config f32) (:config q8_0)))
    (is (map? (get-in q8_0 [:weights :layers 0 :q])) "Q8_0's matrices stay quantized")
    (is (= {:bits 8 :group-size 32} (select-keys (get-in q8_0 [:weights :embed]) [:bits :group-size])))
    (is (nil? (get-in f32 [:weights :output])) "a tied output")
    (let [md (:metadata (array/load-gguf (:f32 files)))
          reason (fn [m] (:reason (ts/thrown-data (qwen3/config m))))]
      (is (= :unsupported (reason (assoc md "general.architecture" "llama"))))
      (is (= :unsupported (reason (assoc md "qwen3.rope.scaling.type" "yarn"))))
      (is (= :unsupported (reason (assoc md "qwen3.rope.dimension_count" 8))))
      (is (= :invalid-args (reason (dissoc md "qwen3.block_count"))))
      (is (= :invalid-args (reason (assoc md "qwen3.attention.head_count_kv" 3)))))))

(deftest every-tensor-is-read-and-checked
  (let [gguf (array/load-gguf (:f32 files))
        t (:tensors gguf)
        data (fn [g] (ts/thrown-data (qwen3/model g)))]
    (is (= {:reason :unsupported :tensors ["blk.0.attn_q.bias"]}
           (select-keys (data (assoc-in gguf [:tensors "blk.0.attn_q.bias"] (array/zeros [64]))) [:reason :tensors]))
        "a tensor the forward does not read")
    (is (= :invalid-args (:reason (data (update gguf :tensors dissoc "blk.1.ffn_up.weight")))) "a tensor missing")
    (is (= :invalid-args (:reason (data (assoc-in gguf [:tensors "blk.1.attn_q.weight"] (t "blk.1.attn_k.weight")))))
        "a tensor of another shape")
    (is (= :invalid-args (:reason (ts/thrown-data (qwen3/model gguf {:dtype :int32})))))
    (testing "an output of its own (untied): the embedding as one gives the tied model's logits"
      (let [tied (:f32 @models)
            untied (qwen3/model (assoc-in gguf [:tensors "output.weight"] (t "token_embd.weight")))
            scaled (qwen3/model (assoc-in gguf [:tensors "output.weight"] (array/* 2.0 (t "token_embd.weight"))))]
        (is (some? (get-in untied [:weights :output])))
        (is (= (host (qwen3/forward tied [ctx])) (host (qwen3/forward untied [ctx]))))
        (is (ts/close? (array/* 2.0 (qwen3/forward tied [ctx])) (qwen3/forward scaled [ctx])))))))

(deftest a-cached-reading-is-the-full-forward
  (doseq [[k m] @models]
    (testing (name k)
      (is (<= (cache-gap m) band))
      (let [[last-row] (qwen3/prefill m [ctx] {:logits :last})
            [steps c] (qwen3/step m (second (qwen3/prefill m [(subvec ctx 0 3)])) [(subvec ctx 3 9)] {:logits :last})]
        (is (= [1 1 128] (array/shape last-row)))
        (is (<= (max-diff last-row (row (qwen3/forward m [ctx]) 8)) band) ":logits :last is the last row")
        (is (<= (max-diff steps last-row) band))
        (is (= 9 (qwen3/offset c)))
        (is (= [1 2 9 16] (array/shape (get-in c [:layers 2 :k]))) "[B KV-heads positions head-dim]"))))
  (testing "planted bugs fail it"
    (let [m (:f32 @models)
          attention @#'qwen3/attention
          sdpa array/sdpa]
      (doseq [[what redefs] {"a cached read's positions shifted by one"
                             {#'qwen3/attention (fn [cfg w x offset entry] (attention cfg w x (if entry (inc offset) offset) entry))}
                             "a cached read's cache dropped"
                             {#'qwen3/attention (fn [cfg w x offset _] (attention cfg w x offset nil))}
                             "attention not causal"
                             {#'array/sdpa (fn [q k v opts] (sdpa q k v (dissoc opts :mask)))}}]
        (is (< 0.01 (with-redefs-fn redefs #(cache-gap m))) what)))))

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
    (let [embed @#'qwen3/embed]
      (is (= 0.0 (second (with-redefs-fn {#'qwen3/embed (fn [w tokens dtype]
                                                          (embed w (if (= 1 (second (array/shape tokens)))
                                                                     (array/zeros-like tokens)
                                                                     tokens)
                                                                 dtype))}
                           #(fork-gaps (:f32 @models)))))))))

;; ---------------------------------------------------------------------------
;; Gradients (the JVM backend, float64)
;; ---------------------------------------------------------------------------

(defn- log-p
  "log p(ctx[8] | ctx[0..7]) under the model m."
  [m]
  (let [[lg] (qwen3/prefill m [(subvec ctx 0 8)] {:logits :last})]
    (array/item (array/sum (array/pick (array/log-softmax lg) (array/array [[(ctx 8)]] {:dtype :int32}))))))

(defn- log-p-of
  "log p as a function of the weights at paths (into the model)."
  [m paths]
  (fn [& ws]
    (let [[lg] (qwen3/prefill (reduce (fn [m [p w]] (assoc-in m p w)) m (map vector paths ws)) [(subvec ctx 0 8)] {:logits :last})]
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
  ;; measured 2026-11-07: errors up to 4.5e-8 (F32: an embedding's entry
  ;; whose gradient is 4e-4, off by 4.5e-11) and 2.2e-9 (Q8_0); every
  ;; entry is within 5e-11, central differences' round-off at h = 1e-5.
  ;; The band 1e-6. A gradient compared with another entry's difference
  ;; is off by O(1)
  (if-not (ts/jvm?)
    (println "gen.llm.models.qwen3-test: the gradients are checked on the JVM backend (float64)")
    (let [{:keys [f32 q8_0]} @models
          f32-paths [[:weights :layers 0 :q] [:weights :embed] [:weights :layers 2 :k-norm]]
          q8-paths [[:weights :layers 0 :attn-norm] [:weights :layers 1 :q-norm] [:weights :norm]]
          entries {[:weights :layers 0 :q] [[0 0] [17 5] [63 63]]
                   [:weights :embed] [[(ctx 2) 4] [(ctx 8) 9] [100 63]]
                   [:weights :layers 2 :k-norm] [[3] [15]]
                   [:weights :layers 0 :attn-norm] [[0] [40]]
                   [:weights :layers 1 :q-norm] [[2] [11]]
                   [:weights :norm] [[7] [63]]}]
      (is (< (worst-relative-error f32 f32-paths entries) 1e-6) "F32: a dense weight, the embedding, a norm")
      (is (< (worst-relative-error q8_0 q8-paths entries) 1e-6) "Q8_0: norms, through the quantized products")
      (testing "a planted bug fails it: the gradient of one entry against another's difference"
        (let [path [:weights :layers 0 :q]
              g (host ((array/grad (log-p-of f32 [path])) (get-in f32 path)))
              a (get-in g [17 5])
              b (central-difference f32 path [17 6])]
          (is (< 1e-3 (/ (Math/abs (- a b)) (Math/abs b)))))))))

;; ---------------------------------------------------------------------------
;; MLX against the JVM backend
;; ---------------------------------------------------------------------------

(deftest mlx-is-the-jvm-backend-within-float32
  (if (ts/jvm?)
    (println "gen.llm.models.qwen3-test: MLX is compared with the JVM backend on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [[k file] files]
        (let [mine (get @models k)
              theirs (array/with-backend jvm (qwen3/load file))
              on-jvm (fn [f] (array/with-backend jvm (host (f theirs))))
              gap (fn [f] (reduce max (map #(Math/abs (- %1 %2)) (flatten (host (f mine))) (flatten (on-jvm f)))))]
          (is (<= (gap #(qwen3/forward % [ctx other])) band) (str (name k) ": a batch's forward"))
          (is (<= (gap #(first (qwen3/step % (second (qwen3/prefill % [(subvec ctx 0 5)])) [(subvec ctx 5 9)]))) band)
              (str (name k) ": a cached chunk")))))))

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:q8_0 @models)
        reason (fn [f] (:reason (ts/thrown-data (f))))]
    (is (= :invalid-token (reason #(qwen3/forward m [[1 2 128]]))))
    (is (= :invalid-token (reason #(qwen3/forward m [[1 -1]]))))
    (is (= :invalid-args (reason #(qwen3/forward m [[1 2] [3]]))) "ragged rows")
    (is (= :invalid-args (reason #(qwen3/forward m [[]]))))
    (is (= :invalid-args (reason #(qwen3/forward m [1 2]))) "a row, not rows")
    (is (= :invalid-args (reason #(qwen3/prefill m [[1 2]] {:logits :first}))))
    (is (= :unknown-options (reason #(qwen3/prefill m [[1 2]] {:last? true}))))
    (is (= :unknown-options (reason #(qwen3/model (array/load-gguf "test/fixtures/gguf/qwen3-tiny-q8_0.gguf") {:dtpye :float32}))))
    (is (= :invalid-args (reason #(qwen3/step m (second (qwen3/prefill m [[1 2]])) [[3] [4]]))) "a cache of other rows")
    (is (= [2 3 128] (array/shape (qwen3/forward m (array/array [[1 2 3] [4 5 6]] {:dtype :int32})))) "an int array")))
