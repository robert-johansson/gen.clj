(ns gen.llm.convert-test
  "gen.llm.convert (experimental) on the fixtures of test/fixtures/gguf
   (generate.py):
   - convert!'s files are llama.cpp's quantizer's (v0.9.0, its
     llama_model_quantize, on macOS arm64) from the same source, every
     tensor's type and bytes and the metadata: qwen3-small-f16.gguf as
     Q4_K_M with llama-quantize's --tensor-type overrides (Q4_K, Q5_K,
     Q6_K, Q8_0) and as Q8_0 with others (Q4_0, Q4_1, Q5_0, Q5_1, BF16,
     F32);
   - the file types' choices: llama.cpp's for Qwen3-0.6B's 28 layers
     (Q6_K for attn_v and ffn_down in layers 0-2, 5, 8, …, 23 and 24-27,
     the tied embedding Q6_K: its Q4_K_M file, 2026-11-08), and its
     fallbacks for rows of 64 values (Q4_K → Q5_0, Q6_K → Q8_0);
   - recipes: rules in order, :keep, MLX's affine forms as the GGUF types
     of their layouts (and loud for the MLX-only ones), the tensors
     llama.cpp never quantizes kept, requantizing refused unless asked;
   - a file of every tensor kept is its source's, tensor for tensor (a
     split one joined);
   - a converted model's forward is its source's within the
     quantization's error: KL(source || converted) of each position's
     next-token distribution at most twice the largest measured (the JVM
     backend, 2026-11-08), on a random model whose log-probabilities span
     tens of nats;
   - quantize (in memory, MLX's affine layout): each matrix its float
     original within a step (its group's scale: MLX makes 0 exact, so an
     extremum may fall a step short) and float32's rounding, at mlx-lm's
     recipes' bits and group sizes, and the forward within its band.
   Planted (2026-11-08), each failing a test here: only the first rule
   tried (a tensor the second matches skipped: attn_k at Q4_K, not
   Q8_0); ffn_gate taken for a tensor llama.cpp never quantizes (kept
   F16); use_more_bits' last eighth from 6n/8; {:bits 4 :group-size 32}
   written as Q4_0; quantize's matrices quantized in groups of 32 where
   the recipe says 64. The encoders' own (a bias's sign among them) are
   gen.array.gguf-quants-test's."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.convert :as convert]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private dir "test/fixtures/gguf/")
(defn- fixture [name] (str dir name))
(def ^:private small (fixture "qwen3-small-f16.gguf"))

(defn- temp [name]
  (str (Files/createTempDirectory "convert-test" (make-array FileAttribute 0)) "/" name))

(defn- thrown [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(defn- file-bytes
  ^bytes [path {:keys [offset bytes]}]
  (let [out (byte-array bytes)]
    (with-open [f (java.io.RandomAccessFile. (str path) "r")]
      (.seek f (long offset))
      (.readFully f out))
    out))

(defn- same-tensors?
  "Are the tensors of the files a and b the same, each its name, type,
   shape and bytes?"
  [a b]
  (let [ta (into {} (map (juxt :name identity)) (:tensors (gguf/header a)))
        tb (into {} (map (juxt :name identity)) (:tensors (gguf/header b)))]
    (and (= (set (keys ta)) (set (keys tb)))
         (every? (fn [[n i]]
                   (let [j (tb n)]
                     (and (= (:type i) (:type j)) (= (:shape i) (:shape j))
                          (java.util.Arrays/equals (file-bytes a i) (file-bytes b j)))))
                 ta))))

;; ---------------------------------------------------------------------------
;; Against llama.cpp's quantizer
;; ---------------------------------------------------------------------------

(def ^:private references
  "llama.cpp's files (generate.py's llama_quantize calls) and the recipes
   that are the same: llama-quantize's --tensor-type overrides as rules,
   its file type at the end."
  [["qwen3-small-q4_k_m.gguf" [["ffn_up" :q5_k] ["attn_k" :q8_0] :q4_k_m]]
   ["qwen3-small-mixed.gguf" [["attn_q" :q4_0] ["attn_k" :q4_1] ["attn_v" :q5_0] ["attn_output" :q5_1]
                              ["ffn_gate" :bf16] ["ffn_up" :f32] :q8_0]]])

(deftest converted-files-are-llama-cpp-s
  (doseq [[reference recipe] references]
    (testing reference
      (let [out (temp reference)
            {:keys [path bytes tensors]} (convert/convert! small out {:recipe recipe})]
        (is (= out path))
        (is (same-tensors? (fixture reference) out) "every tensor's type, shape and bytes")
        (is (= (:metadata (gguf/header (fixture reference))) (:metadata (gguf/header out))) "the metadata")
        (is (= (.length (java.io.File. ^String (fixture reference))) bytes) "the same size")
        (is (= (map :name (:tensors (gguf/header small))) (map :name tensors)) "the source's order")))))

(defn- types-of [plan] (into {} (map (juxt :name :to)) plan))

(defn- layers-file
  "A GGUF of n blocks of a dense model's attn_v and ffn_down, each [1 256]
   (two axes, as llama.cpp quantizes), a tied embedding and a norm."
  [n]
  (let [row (float-array (map #(Math/sin (double %)) (range 256)))
        f (temp "layers.gguf")]
    (gguf/write-gguf f {:metadata [["general.architecture" :string "qwen3"] ["qwen3.block_count" :uint32 n]]
                        :tensors (concat [{:name "token_embd.weight" :type :f32 :shape [1 256] :data row}
                                          {:name "output_norm.weight" :type :f32 :shape [256] :data row}]
                                         (for [i (range n) t ["attn_v" "ffn_down" "attn_q"]]
                                           {:name (str "blk." i "." t ".weight") :type :f32 :shape [1 256] :data row}))})
    f))

(deftest file-types-choose-as-llama-cpp-does
  (testing "Q4_K_M over 28 layers: llama.cpp's choices for Qwen3-0.6B"
    (let [types (types-of (convert/plan (layers-file 28) {:recipe :q4_k_m}))
          more #{0 1 2 5 8 11 14 17 20 23 24 25 26 27}]
      (doseq [i (range 28) t ["attn_v" "ffn_down"]]
        (is (= (if (more i) :q6_k :q4_k) (types (str "blk." i "." t ".weight"))) (str "layer " i " " t)))
      (is (every? #{:q4_k} (map #(types (str "blk." % ".attn_q.weight")) (range 28))))
      (is (= :q6_k (types "token_embd.weight")) "the tied embedding as the output")
      (is (= :f32 (types "output_norm.weight")) "a norm, never quantized")))
  (testing "Q5_K_M and Q4_K_S"
    (let [m (types-of (convert/plan (layers-file 28) {:recipe :q5_k_m}))
          s (types-of (convert/plan (layers-file 28) {:recipe :q4_k_s}))]
      (is (= :q6_k (m "blk.5.attn_v.weight")))
      (is (= :q5_k (m "blk.6.attn_v.weight")))
      (is (= [:q5_k :q5_k :q5_k :q5_k :q4_k] (mapv #(s (str "blk." % ".attn_v.weight")) (range 5))) "the first 4 attn_v")
      (is (= [:q5_k :q5_k :q5_k :q4_k] (mapv #(s (str "blk." % ".ffn_down.weight")) (range 4))) "ffn_down below 28/8")))
  (testing "rows of 64 values: llama.cpp's fallbacks"
    (let [plan (convert/plan (fixture "qwen3-tiny-f32.gguf") {:recipe :q4_k_m})
          types (types-of plan)]
      (is (= :q5_0 (types "blk.0.attn_q.weight")) "Q4_K → Q5_0")
      (is (= [:q5_0 :q5_0 :q8_0] (mapv #(types (str "blk." % ".attn_v.weight")) (range 3))) "Q6_K (layer 2 of 3) → Q8_0")
      (is (= :q8_0 (types "token_embd.weight")) "the output's rule: Q8_0 for rows not of 256")
      (is (every? #(= :f32 (:to %)) (filter #(re-find #"norm" (:name %)) plan))))))

(deftest recipes
  (let [plan (fn [recipe & [opts]] (types-of (convert/plan small (merge {:recipe recipe} opts))))]
    (testing "rules in order, the first that matches deciding"
      (let [t (plan [[#"attn" :q8_0] [#"attn_q" :q4_0] [#"ffn" :keep]])]
        (is (= :q8_0 (t "blk.0.attn_q.weight")))
        (is (= :f16 (t "blk.0.ffn_up.weight")) ":keep")
        (is (= :f16 (t "token_embd.weight")) "no rule, no file type: kept")))
    (testing "a function of the description"
      (let [t (plan [[#(= 0 (:layer %)) :q8_0] :f16])]
        (is (= :q8_0 (t "blk.0.attn_q.weight")))
        (is (= :f16 (t "token_embd.weight")))))
    (testing "what llama.cpp never quantizes is kept, whatever the rule"
      (is (= :f32 ((plan [[#"" :q8_0]]) "blk.0.attn_norm.weight"))))
    (testing "MLX's affine forms: the GGUF type of their layout, or loud"
      (let [t (plan [[#"attn_q" {:bits 4 :group-size 32}] [#"attn_k" {:bits 5 :group-size 32}] [#"attn_v" {:bits 8 :group-size 32}]])]
        (is (= [:q4_1 :q5_1 :q8_0] (mapv t ["blk.0.attn_q.weight" "blk.0.attn_k.weight" "blk.0.attn_v.weight"]))))
      (is (= :unsupported (:reason (thrown #(plan [[#"" {:bits 4 :group-size 64}]])))) "MLX-only")
      (is (= :invalid-args (:reason (thrown #(plan [[#"" {:bits 7 :group-size 32}]])))) "no such layout"))
    (testing "the embedding and the output at 8 bits, the rest Q4_K_M's"
      (let [t (plan [["^(token_embd|output)\\.weight$" :q8_0] :q4_k_m])]
        (is (= :q8_0 (t "token_embd.weight")))
        (is (= :q4_k (t "blk.0.attn_q.weight")))))
    (testing "loud"
      (is (= :invalid-args (:reason (thrown #(plan :q3_k_m)))) "a file type not ported")
      (is (= :invalid-args (:reason (thrown #(plan [[1 :q8_0]])))) "a pattern")
      (is (= :invalid-args (:reason (thrown #(plan [[#"" :q8_1]])))) "a value")
      (is (= :invalid-args (:reason (thrown #(plan [#"" :q8_0])))) "a rule")
      (is (= :unknown-options (:reason (thrown #(plan :q8_0 {:pure? true})))))
      (is (= :invalid-args (:reason (thrown #(convert/plan small {}))))))
    (testing "requantizing: refused, unless asked"
      (let [q8 (fixture "qwen3-tiny-q8_0.gguf")]
        (is (= {:reason :requantize :from :q8_0 :to :q5_0}
               (select-keys (thrown #(convert/plan q8 {:recipe :q4_k_m})) [:reason :from :to])))
        (is (= :q5_0 ((types-of (convert/plan q8 {:recipe :q4_k_m :allow-requantize? true})) "blk.0.attn_q.weight")))
        (is (= :q8_0 ((types-of (convert/plan q8 {:recipe :q8_0})) "blk.0.attn_q.weight")) "at its own type: copied")))))

(deftest a-file-kept-is-its-source
  (doseq [[source same-as] [["qwen3-small-mixed.gguf" "qwen3-small-mixed.gguf"]
                            ["qwen3-tiny-q8_0-00001-of-00003.gguf" "qwen3-tiny-q8_0.gguf"]]]
    (testing source
      (let [out (temp "kept.gguf")
            _ (convert/convert! (fixture source) out {:recipe [[#"" :keep]]})
            h (gguf/header out)]
        (is (same-tensors? (fixture same-as) out))
        (is (= (dissoc (:metadata (gguf/header (fixture same-as))) "general.file_type" "general.quantization_version")
               (dissoc (:metadata h) "general.quantization_version"))
            "the metadata, but general.file_type (a recipe without a file type) and a split's keys")
        (is (= 2 (get-in h [:metadata "general.quantization_version"])))
        (is (not-any? #(.startsWith ^String % "split.") (keys (:metadata h))))))))

;; ---------------------------------------------------------------------------
;; The converted forward
;; ---------------------------------------------------------------------------

(def ^:private prompts
  [[3 17 200 45 99 128 7 255 0 64 31 150]
   [5 5 5 1 2 3 250 251 252 9 8 7]])

(defn- log-probs [m] (array/log-softmax (qwen3/forward m prompts) {:axis -1}))

(defn- worst-kl
  "The largest KL(a || b) over the positions of prompts."
  [a b]
  (let [pa (log-probs a) pb (log-probs b)]
    (ts/number (array/max (array/sum (array/* (array/exp pa) (array/- pa pb)) {:axis -1})))))

(def ^:private original (delay (qwen3/model (array/load-gguf small))))

(deftest converted-forwards-are-the-original-s-within-the-quantization
  ;; measured (the JVM backend, 2026-11-08), the largest KL over the
  ;; prompts' 24 positions: Q8_0 0.0027, BF16 0.00036, Q6_K 0.044, Q5_K_M
  ;; 0.065, Q4_K_M 0.207, Q4_0 0.366 nats; the bands twice these, rounded
  ;; up
  (doseq [[recipe band] [[:q8_0 0.006] [:bf16 0.001] [:q6_k 0.09] [:q5_k_m 0.14] [:q4_k_m 0.42] [:q4_0 0.75]]]
    (testing (name recipe)
      (let [out (temp "m.gguf")
            _ (convert/convert! small out {:recipe recipe})
            kl (worst-kl @original (qwen3/model (array/load-gguf out)))]
        (is (<= kl band) (str "KL " kl " nats"))
        (is (< 0.0 kl) "the converted model is not the original")))))

(defn- step-errors
  "The largest |w - ŵ| - |scale| (beyond a step, and float32's rounding of
   w) over each quantized matrix of q against its original (dense) in
   gguf."
  [gguf q]
  (for [[name t] (:tensors q)
        :when (= :affine ((:types q) name))]
    (let [w (array/astype ((:tensors gguf) name) :float32)
          d (array/dequantize (:wq t) (:scales t) (:biases t) (assoc (select-keys t [:bits :group-size]) :dtype :float32))
          [n k] (array/shape w)
          g (long (:group-size t))
          err (array/reshape (array/abs (array/- w d)) [n (quot k g) g])
          step (array/expand-dims (array/abs (:scales t)) -1)]
      [name (ts/number (array/max (array/- err (array/+ step (array/* 1e-6 (array/abs (array/reshape w [n (quot k g) g])))))))])))

(deftest quantize-holds-mlx-s-affine-layouts
  ;; MLX's affine quantization makes 0 exact: the scale is edge / q0, q0
  ;; the steps from the extremum of the larger magnitude (the edge) to 0,
  ;; rounded, so the other extremum may fall short of the codes' range by
  ;; up to a step, and a value lies within one step of its code (measured
  ;; 2026-11-08: exactly 1.0 at the most, at every recipe). The largest KL
  ;; measured (the JVM backend): 4 bits 0.675, mixed_4_6 0.574, 8 bits
  ;; 0.0015 nats (MLX's 4 bits in groups of 64 three times Q4_K_M's 0.207
  ;; on this model); the bands twice these, rounded up
  (let [gguf (array/load-gguf small)]
    (doseq [[recipe band] [[:mlx-4bit 1.375] [:mlx-mixed-4-6 1.25] [:mlx-8bit 0.0035]]]
      (testing (name recipe)
        (let [q (convert/quantize gguf {:recipe recipe})
              m (qwen3/model q)
              kl (worst-kl @original m)]
          (is (every? (fn [[_ e]] (<= e 0.0)) (step-errors gguf q)) "each value within a step")
          (is (= :affine ((:types q) "blk.0.attn_q.weight")))
          (is (= 64 (get-in q [:tensors "blk.0.attn_q.weight" :group-size])))
          (is (= :f32 ((:types q) "output_norm.weight")) "a norm kept")
          (is (<= kl band) (str "KL " kl " nats")))))
    (testing "mixed_4_6: 6 bits where use_more_bits says, as mlx-lm"
      (let [q (convert/quantize gguf {:recipe :mlx-mixed-4-6})]
        (is (= 6 (get-in q [:tensors "blk.0.attn_v.weight" :bits])) "layer 0 of 1")
        (is (= 4 (get-in q [:tensors "blk.0.attn_q.weight" :bits])))))
    (testing "loud"
      (is (= :invalid-args (:reason (thrown #(convert/quantize gguf {:recipe [[#"" :q4_k]]})))) "a GGUF type: convert!'s")
      (is (= :invalid-args (:reason (thrown #(convert/quantize gguf {:recipe :q4_k_m})))) "a file type"))))
