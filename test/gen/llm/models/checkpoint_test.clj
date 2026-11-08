(ns gen.llm.models.checkpoint-test
  "Hugging Face checkpoints read by qwen3/load and qwen35/load (through
   gen.llm.models.checkpoint), on the backend under test, against the
   GGUF files they convert to: test/fixtures/safetensors's checkpoints
   (generate.py there) are test/fixtures/gguf's tiny random models undone,
   each the checkpoint llama.cpp's converter would write that GGUF from.
   1. Each checkpoint and its GGUF give one model: the same config, the
      same weights (host values, each), and the same logits, exactly on
      the JVM backend: Qwen3 (F32, with an lm_head copy of its tied
      embedding; and MLX's 8-bit checkpoint of the Q8_0 GGUF, sharded in
      three, matrices across shards), Qwen3.5 (Hugging Face's layout, its
      RMSNorm weights without Qwen3-Next's +1, A_log, the value heads
      grouped, 2 key heads of 2 value heads each, the conv kernel [C 1 4],
      an MTP block and a vision tensor never read; mlx-lm's saved layout,
      8-bit, its norms holding the +1, the kernel [C 4 1]), Qwen3.5-MoE
      (one matrix an expert, the router and the shared expert; mlx-lm's
      switch_mlp stacks, 8-bit).
   2. MLX's 4-bit checkpoint at group 64, one matrix at 8 bits and group
      32 by its own quantization entry, BF16 norms: each matrix of its
      bits and group size, its values MLX's dequantization (the generator's,
      qwen3-tiny-4bit.edn).
   3. The loud errors: a checkpoint without config.json, quantized
      matrices without a quantization, another quantization (GPTQ, MLX's
      mxfp4), another model_type, sliding windows, RoPE scaling, an
      untied output without lm_head, a vocabulary of another size, a
      tensor the forward does not read (an attention bias), the experts'
      fused gate_up_proj, an expert missing.
   Each of 1's conventions fails under a planted bug, run here as a
   negative control (with-redefs-fn, its logits' gap asserted past 0.01):
   the +1 missed, A_log's sign (exp for -exp), the value heads left
   grouped, mlx-lm's layout read as Hugging Face's (the +1 twice), the
   experts stacked in reverse. Their margins (JVM): 4.8, 5.7, 4.7, 9.4
   and 5.2 in a logit. And planted mutations (2026-11-08), each failing
   a test here: every shard's tensors read from the first shard's file
   (gen.array.safetensors), a matrix's own quantization entry ignored
   (gen.llm.models.checkpoint).
   The bands: 0 on the JVM backend (the same values, the same ops); on
   MLX 2e-4 in a logit (|logit| <= 5 here), qwen35-test's float32 band,
   and ssm_a within float32's ulp (MLX's exp is not the JVM's)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.json :as json]
            [gen.llm.models.checkpoint :as checkpoint]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private st "test/fixtures/safetensors/")
(def ^:private gg "test/fixtures/gguf/")

(def ^:private pairs
  "[family checkpoint GGUF] of each checkpoint."
  [[:qwen3 "qwen3-tiny" "qwen3-tiny-f32.gguf"]
   [:qwen3 "qwen3-tiny-8bit" "qwen3-tiny-q8_0.gguf"]
   [:qwen35 "qwen35-tiny" "qwen35-tiny-f32.gguf"]
   [:qwen35 "qwen35-tiny-mlx-8bit" "qwen35-tiny-q8_0.gguf"]
   [:qwen35 "qwen35moe-tiny" "qwen35moe-tiny-f32.gguf"]
   [:qwen35 "qwen35moe-tiny-mlx-8bit" "qwen35moe-tiny-q8_0.gguf"]])

(def ^:private families
  {:qwen3 {:load qwen3/load :forward qwen3/forward}
   :qwen35 {:load qwen35/load :forward qwen35/forward}})

(def ^:private ctx [[5 17 42 3 88 11 60 2 9] [1 2 3 4 5 6 7 8 9]])

(def ^:private band (if (ts/jvm?) 0.0 2e-4))

(defn- host [x] (array/->clj x))

(defn- thrown [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(defn- max-diff ^double [a b]
  (let [d (double (array/item (array/max (array/abs (array/- a b)))))]
    (if (Double/isNaN d) Double/POSITIVE_INFINITY d)))

(defn- hosted
  "A model's weights with every array read on the host (a quantized
   matrix's words, scales and biases each)."
  [w]
  (cond (map? w) (into {} (map (fn [[k v]] [k (hosted v)])) w)
        (vector? w) (mapv hosted w)
        (nil? w) nil
        (number? w) w
        :else (host w)))

(defn- logits [family m] ((:forward (families family)) m ctx))

(defn- gap
  "max |Δ logit| between the checkpoint's model and its GGUF's, the
   checkpoint loaded as (f) runs (the redefinitions of a planted bug)."
  ([family checkpoint gguf] (gap family checkpoint gguf nil))
  ([family checkpoint gguf redefs]
   (let [{ld :load} (families family)
         mc (if redefs (with-redefs-fn redefs #(ld (str st checkpoint))) (ld (str st checkpoint)))]
     (max-diff (logits family mc) (logits family (ld (str gg gguf)))))))

(deftest checkpoints-are-their-gguf
  (doseq [[family checkpoint gguf] pairs]
    (testing checkpoint
      (let [{ld :load} (families family)
            mc (ld (str st checkpoint))
            mg (ld (str gg gguf))]
        (is (= (:config mg) (:config mc)) "the config")
        (if (ts/jvm?)
          (is (= (hosted (:weights mg)) (hosted (:weights mc))) "the weights")
          ;; MLX's float32 exp may round -exp(A_log) an ulp from the JVM's
          ;; (the generator chose A_log for the JVM's exp, the double's
          ;; rounded): ssm_a within float32's round-off, the rest equal
          (let [without-a (fn [w] (update w :layers (partial mapv #(dissoc % :a))))]
            (is (= (hosted (without-a (:weights mg))) (hosted (without-a (:weights mc)))) "the weights but ssm_a")
            (doseq [[a b] (map vector (:layers (:weights mg)) (:layers (:weights mc)))
                    :when (:a a)]
              (is (ts/close? (:a a) (:a b) {:rtol 2.4e-7 :atol 0.0}) "ssm_a, within float32's ulp"))))
        (is (<= (max-diff (logits family mg) (logits family mc)) band) "the logits")))))

(deftest planted-convention-bugs-fail
  (let [stack checkpoint/stack
        margins {:norm-plus-one-missed (gap :qwen35 "qwen35-tiny" "qwen35-tiny-f32.gguf"
                                            {#'checkpoint/shift-norm #(array/astype % :float32)})
                 :a-log-sign (gap :qwen35 "qwen35-tiny" "qwen35-tiny-f32.gguf"
                                  {#'checkpoint/neg-exp #(array/exp (array/astype % :float32))})
                 :value-heads-grouped (gap :qwen35 "qwen35-tiny" "qwen35-tiny-f32.gguf"
                                           {#'checkpoint/tiled (fn [_ w & _] w)})
                 :mlx-read-as-hf (gap :qwen35 "qwen35-tiny-mlx-8bit" "qwen35-tiny-q8_0.gguf"
                                      {#'qwen35/convention (constantly :hf)})
                 :experts-reversed (gap :qwen35 "qwen35moe-tiny" "qwen35moe-tiny-f32.gguf"
                                        {#'checkpoint/stack (fn [who ws] (stack who (vec (reverse ws))))})}]
    (doseq [[k margin] margins]
      (is (> margin 0.01) (str k ": " margin)))))

(deftest mlx-quantized-checkpoints
  (let [m (qwen3/load (str st "qwen3-tiny-4bit"))
        reference (edn/read-string (slurp (str st "qwen3-tiny-4bit.edn")))
        w (:weights m)
        rows (fn [{:keys [wq scales biases] :as q}]
               (host (array/dequantize (array/slice wq 0 0 2) (array/slice scales 0 0 2) (array/slice biases 0 0 2)
                                       (assoc (select-keys q [:bits :group-size]) :dtype :float32))))]
    (is (= {:bits 4 :group-size 64} (select-keys (get-in w [:layers 0 :q]) [:bits :group-size])))
    (is (= {:bits 4 :group-size 64} (select-keys (:embed w) [:bits :group-size])) "the embedding quantized")
    (is (= {:bits 8 :group-size 32} (select-keys (get-in w [:layers 1 :down]) [:bits :group-size])) "its own entry")
    (is (= {:bits 4 :group-size 64} (select-keys (get-in w [:layers 0 :down]) [:bits :group-size])))
    (doseq [[name k] [["model.layers.0.self_attn.q_proj" [:layers 0 :q]] ["model.layers.1.mlp.down_proj" [:layers 1 :down]]]]
      (if (ts/jvm?)
        (is (= (reference name) (rows (get-in w k))) name)
        (is (ts/close? (array/array (reference name)) (array/array (rows (get-in w k))) {:rtol 1e-6 :atol 1e-7}) name)))
    (let [l (qwen3/forward m ctx)]
      (is (= [2 9 128] (array/shape l)))
      (is (Double/isFinite (double (array/item (array/sum (array/abs l)))))))))

;; ---------------------------------------------------------------------------
;; Loud errors, on altered copies of the fixtures
;; ---------------------------------------------------------------------------

(defn- temp-dir []
  (str (Files/createTempDirectory "checkpoint-test" (make-array FileAttribute 0))))

(defn- to-json
  "x as JSON (maps, vectors, strings, numbers, booleans, nil)."
  [x]
  (cond (map? x) (str "{" (str/join ", " (map (fn [[k v]] (str (to-json (str k)) ": " (to-json v))) x)) "}")
        (sequential? x) (str "[" (str/join ", " (map to-json x)) "]")
        (string? x) (str "\"" (str/replace x "\"" "\\\"") "\"")
        (nil? x) "null"
        :else (str x)))

(defn- write-safetensors
  "A safetensors file at path of F32 tensors, {name [shape values]}."
  [path tensors]
  (let [entries (sort-by key tensors)
        sizes (map (fn [[_ [shape]]] (* 4 (reduce * 1 shape))) entries)
        offsets (reductions + 0 sizes)
        header (.getBytes ^String (to-json (into (sorted-map)
                                                 (map (fn [[name [shape]] begin end]
                                                        [name {"dtype" "F32" "shape" shape "data_offsets" [begin end]}])
                                                      entries offsets (rest offsets))))
                          "UTF-8")
        b (doto (ByteBuffer/allocate (+ 8 (alength header) (last offsets))) (.order ByteOrder/LITTLE_ENDIAN))]
    (.putLong b (alength header))
    (.put b header)
    (doseq [[_ [_ values]] entries v values] (.putFloat b (float v)))
    (io/copy (.array b) (io/file path))))

(defn- f32-tensors
  "The F32 tensors of the checkpoint at path, {name [shape values]}."
  [path]
  (into {} (map (fn [[k v]] [k [(array/shape v) (flatten [(host v)])]]))
        (:tensors (array/with-backend (array/backend-named :jvm) (array/load-safetensors path)))))

(defn- altered
  "A copy of the checkpoint name with its config.json changed by
   (f config) and its tensors by (g tensors) (F32 ones; g nil: the file
   copied as it is)."
  ([name f] (altered name f nil))
  ([name f g]
   (let [d (temp-dir)
         src (str st name)]
     (spit (str d "/config.json") (to-json (f (json/read-file (str src "/config.json")))))
     (if g
       (write-safetensors (str d "/model.safetensors") (g (f32-tensors src)))
       (io/copy (io/file src "model.safetensors") (io/file d "model.safetensors")))
     d)))

(defn- reason [f] (:reason (thrown f)))

(deftest loud-checkpoints
  (testing "no config.json"
    (let [d (temp-dir)]
      (io/copy (io/file st "qwen3-tiny/model.safetensors") (io/file d "model.safetensors"))
      (is (= :no-file (reason #(qwen3/load d))))))
  (testing "quantization"
    (is (= :invalid-args (reason #(qwen3/load (altered "qwen3-tiny-4bit" (fn [c] (dissoc c "quantization" "quantization_config")))))))
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny-4bit" (fn [c] (-> (dissoc c "quantization")
                                                                                    (assoc "quantization_config" {"quant_method" "gptq" "bits" 4}))))))))
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny-4bit" (fn [c] (assoc-in c ["quantization" "mode"] "mxfp4")))))))
    (is (= :invalid-args (reason #(qwen3/load (altered "qwen3-tiny-4bit" (fn [c] (assoc-in c ["quantization" "bits"] 3))))))
        "bits that do not match the shapes"))
  (testing "the configuration"
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny" (fn [c] (assoc c "model_type" "llama")))))))
    (is (= :unsupported (reason #(qwen35/load (str st "qwen3-tiny")))) "a Qwen3 checkpoint read as Qwen3.5")
    (is (= :unsupported (reason #(qwen3/load (str st "qwen35-tiny")))) "and the other way")
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny" (fn [c] (assoc c "use_sliding_window" true)))))))
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny" (fn [c] (assoc c "rope_scaling" {"rope_type" "yarn" "factor" 4})))))))
    (is (= :unsupported (reason #(qwen35/load (altered "qwen35-tiny" (fn [c] (assoc-in c ["text_config" "attn_output_gate"] false)))))))
    (is (= :invalid-args (reason #(qwen3/load (altered "qwen3-tiny" (fn [c] (assoc c "vocab_size" 100)))))))
    (is (= :invalid-args (reason #(qwen3/load (altered "qwen3-tiny" (fn [c] (assoc c "tie_word_embeddings" false))
                                                       (fn [ts] (dissoc ts "lm_head.weight"))))))
        "an untied output without lm_head"))
  (testing "tensors"
    (is (= :unsupported (reason #(qwen3/load (altered "qwen3-tiny" identity
                                                      (fn [ts] (assoc ts "model.layers.0.self_attn.q_proj.bias" [[64] (repeat 64 0.0)])))))))
    (is (= :unsupported (reason #(qwen35/load (altered "qwen35moe-tiny" identity
                                                       (fn [ts] (assoc ts "model.language_model.layers.0.mlp.experts.gate_up_proj"
                                                                       [[4 64 32] (repeat (* 4 64 32) 0.0)]))))))
        "the experts' fused matrices")
    (is (= :invalid-args (reason #(qwen35/load (altered "qwen35moe-tiny" identity
                                                        (fn [ts] (dissoc ts "model.language_model.layers.1.mlp.experts.3.up_proj.weight"))))))
        "an expert missing")
    (testing "the unaltered copy loads, so the errors above are their alterations'"
      (is (map? (qwen35/load (altered "qwen35moe-tiny" identity identity))))
      (is (map? (qwen3/load (altered "qwen3-tiny" identity)))))))
