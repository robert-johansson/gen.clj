(ns gen.llm.models.qwen3-kquants-test
  "A Qwen3 of K-quants (gen.llm.models.qwen3 over gen.array/load-gguf's
   K-quants) against its float original: qwen3-small-f16.gguf, a tiny
   random Qwen3 whose rows are 256 values (test/fixtures/gguf/
   generate.py), and qwen3-small-q4_k_m.gguf, the same weights quantized
   by llama.cpp's quantizer (v0.9.0) as Q4_K_M, with ffn_up as Q5_K and
   attn_k as Q8_0 (its --tensor-type overrides): Q4_K, Q5_K, Q6_K (attn_v,
   ffn_down and the tied embedding), Q8_0 and F32 in one model, each
   tensor at its own bits.
   - Each quantized weight is its original within the quantization's
     error: a code a step or two from its value (the step a group's
     scale; Q6_K's, a 16-group's largest |w| / 31), a quarter of a step
     on average, as rounding to a code gives. Measured (2026-11-08): the
     mean 0.21 to 0.23 steps, the largest 0.55 (Q8_0), 0.94 (Q4_K), 1.15
     (Q5_K) and 0.70 (Q6_K), where the 6-bit scales and mins round the
     sub-blocks' own; the bounds, 0.3 and 2 steps.
   - The forward reads each tensor at its own bits: its log-probabilities
     are those of the same model with every weight dequantized, dense, to
     the model's dtype (round-off: on the JVM backend the same values in
     float64; on MLX quantized-matmul's kernel against matmul's).
   - Its next-token distributions are the original's within the
     quantization's error: KL(original || K-quants) at each position of
     two prompts at most 0.375 nats, twice the largest measured
     (2026-11-08, the JVM backend: 0.187; most positions under 0.05), on a
     random model whose log-probabilities span 26 nats. Planted (the
     K-quants' bias -(dmin·m) with its sign flipped in gen.array.gguf,
     2026-11-08): KL 13.4 nats, and the Q4_K and Q5_K weights 15 to 31
     steps from their originals on average."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts]))

(def ^:private original-file "test/fixtures/gguf/qwen3-small-f16.gguf")
(def ^:private kquants-file "test/fixtures/gguf/qwen3-small-q4_k_m.gguf")

(def ^:private original (delay (array/load-gguf original-file)))
(def ^:private kquants (delay (array/load-gguf kquants-file)))

(def ^:private prompts
  [[3 17 200 45 99 128 7 255 0 64 31 150]
   [5 5 5 1 2 3 250 251 252 9 8 7]])

(defn- dequantized
  "A quantized tensor's matrix (load-gguf's map), dense in dtype."
  [{:keys [wq scales biases] :as q} dtype]
  (array/dequantize wq scales biases (assoc (select-keys q [:bits :group-size]) :dtype dtype)))

(deftest the-file-holds-every-k-quant
  (is (= {:q4_k #{"attn_q" "attn_output" "ffn_gate"} :q5_k #{"ffn_up"} :q6_k #{"token_embd" "attn_v" "ffn_down"}
          :q8_0 #{"attn_k"}}
         (->> (:types @kquants)
              (remove (comp #{:f32} val))
              (group-by val)
              (into {} (map (fn [[t kvs]] [t (set (map #(second (re-find #"(?:blk\.0\.)?([a-z_]+)\.weight" (key %))) kvs))]))))))
  (is (every? #{:f32} (vals (select-keys (:types @kquants) (filter #(.contains ^String % "norm") (keys (:types @kquants))))))))

(deftest each-weight-is-the-original-within-its-steps
  ;; |w - ŵ| in steps: a quantized group's scale (Q4_K, Q5_K: d·sc; Q8_0:
  ;; d); a Q6_K 16-group's largest |w| / 31 (its step d·sc, near it)
  (doseq [[name t] (sort-by key (:tensors @kquants))
          :let [type (get-in @kquants [:types name])
                w (flatten (ts/host (array/astype ((:tensors @original) name) :float32)))]
          :when (not= :f32 type)]
    (testing (str name " (" (clojure.core/name type) ")")
      (let [steps (if (map? t)
                    (let [d (flatten (ts/host (dequantized t :float32)))
                          s (flatten (ts/host (:scales t)))]
                      (map-indexed (fn [i [x y]] (/ (Math/abs (- (double x) (double y))) (Math/abs (double (nth s (quot i 32))))))
                                   (map vector w d)))
                    (let [d (flatten (ts/host t))]
                      (mapcat (fn [gw gd]
                                (let [step (/ (double (reduce max (map #(Math/abs (double %)) gw))) 31.0)]
                                  (map (fn [x y] (/ (Math/abs (- (double x) (double y))) step)) gw gd)))
                              (partition 16 w) (partition 16 d))))
            mean (/ (reduce + steps) (count steps))
            worst (reduce max steps)]
        (is (<= mean 0.3) (str "mean " mean " steps"))
        (is (<= worst 2.0) (str "largest " worst " steps"))))))

(defn- log-probs [m] (array/log-softmax (qwen3/forward m prompts) {:axis -1}))

(deftest the-forward-reads-each-tensor-at-its-own-bits
  (let [m (qwen3/model @kquants)
        dtype (get-in m [:config :dtype])
        dense (qwen3/model (update @kquants :tensors update-vals #(if (map? %) (dequantized % dtype) %)))
        a (log-probs m)
        b (log-probs dense)]
    (is (every? #(and (map? %) (not (contains? % :lora))) (map (get-in m [:weights :layers 0]) [:q :o :gate :up :k]))
        "the projections multiply by quantized-matmul")
    (is (ts/close? a b {:rtol 0.0 :atol (ts/tol 1e-10)}))))

(deftest log-probs-are-the-original-s-within-the-quantization
  (let [a (log-probs (qwen3/model @original))
        b (log-probs (qwen3/model @kquants))
        kl (array/sum (array/* (array/exp a) (array/- a b)) {:axis -1})
        worst (ts/number (array/max kl))]
    (is (<= worst 0.375) (str "KL " worst " nats"))
    (is (< 0.0 worst) "the quantized model is not the original")))
