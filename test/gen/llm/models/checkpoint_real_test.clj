(ns gen.llm.models.checkpoint-real-test
  "A real Hugging Face checkpoint against the same model's GGUF, both in
   our forward (qwen3/load or qwen35/load, by the GGUF's architecture), on
   MLX in float32: a checkpoint given as -J-Dgen.hf.checkpoint=<its
   directory> and its GGUF as -J-Dgen.hf.gguf=<file> (skipped without
   them, and on the JVM backend, whose doubles would hold the model twice
   over):

     clojure -J-Dgen.backend=mlx -J-Dgen.hf.checkpoint=<dir> -J-Dgen.hf.gguf=<gguf> \\
       -X:test :nses '[gen.llm.models.checkpoint-real-test]'

   1. The weights, each in bfloat16 (both models loaded at :dtype
      :bfloat16, to hold two at once): how many differ, and by how much.
   2. The next-token log-probabilities of 4 prompts of 32 tokens (fixed
      token ids), the models loaded one at a time in float32: max |Δ log
      p| over the GGUF's 64 most likely tokens at every position, and the
      argmax at every position.

   The checks, a GGUF of the checkpoint's own values (BF16, or F16 from a
   BF16 checkpoint: exact but for values below float16's normal range,
   rounded to its subnormals, 2^-24 apart): every weight equal but ssm_a,
   whose -exp(A_log) MLX's exp and PyTorch's (the converter's) may round
   apart, and those small values; |Δ log p| at most 1e-3, float32's noise
   through the model; every argmax equal.

   Measured on the M4 (Metal, 2026-11-08):
   - Qwen/Qwen3.5-0.8B (BF16) against ggml-org's Qwen3.5-0.8B-BF16.gguf:
     no weight differs (ssm_a included, in bfloat16); max |Δ log p|
     2.7e-5; argmax equal at 128 of 128 positions.
   - Qwen/Qwen3-0.6B (BF16, an lm_head copy stored) against ggml-org's
     Qwen3-0.6B-f16.gguf: 197 matrices differ, each by at most 3.0e-8
     (float16's subnormals); max |Δ log p| 4.4e-5; argmax 128 of 128."

  (:require [clojure.test :refer [deftest is]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(defn- family [gguf-path]
  (let [arch (get-in (gguf/header gguf-path) [:metadata "general.architecture"])]
    (if (#{"qwen35" "qwen35moe"} arch)
      {:load qwen35/load :forward qwen35/forward}
      {:load qwen3/load :forward qwen3/forward})))

(def ^:private prompts
  "4 prompts of 32 token ids, fixed (a linear congruence over [200,
   30000): common pieces of Qwen's vocabulary)."
  (vec (for [p (range 4)]
         (vec (for [i (range 32)] (+ 200 (mod (+ (* 7919 (+ i (* 37 p))) (* 104729 p) 13) 29800)))))))

(defn- leaves
  "The arrays of a model's weights, by their paths."
  ([w] (leaves [] w))
  ([path w]
   (cond (map? w) (mapcat (fn [[k v]] (leaves (conj path k) v)) w)
         (vector? w) (mapcat (fn [i v] (leaves (conj path i) v)) (range) w)
         (array/array? w) (when-not (array/host? w) [[path w]])
         :else nil)))

(defn- weight-gaps
  "{path max |Δ|} of the weights that differ between the models."
  [a b]
  (let [bs (into {} (leaves (:weights b)))]
    (into (sorted-map)
          (keep (fn [[path x]]
                  (let [y (bs path)
                        d (double (array/item (array/max (array/abs (array/- (array/astype x :float32) (array/astype y :float32))))))]
                    (when-not (zero? d) [path d]))))
          (leaves (:weights a)))))

(defn- log-probs
  "log p of each prompt by m: a float[] [positions·vocab] each, and the
   vocabulary's size."
  [forward m]
  (mapv (fn [p]
          (let [lp (array/log-softmax (array/astype (forward m [p]) :float32) {:axis -1})]
            [(array/->floats lp) (peek (array/shape lp))]))
        prompts))

(defn- compare-rows
  "{:max-diff :argmax-equal :positions} of our log p against the
   reference's, over the reference's 64 most likely tokens a position."
  [ours theirs]
  (let [rows (for [[[^floats a v] [^floats b]] (map vector ours theirs)
                   r (range (quot (alength b) (long v)))]
               [a b (* (long r) (long v)) (long v)])
        top (fn [^floats b ^long at ^long v]
              (let [sorted (java.util.Arrays/copyOfRange b (int at) (int (+ at v)))]
                (java.util.Arrays/sort sorted)
                (let [threshold (aget sorted (- v 64))]
                  (filter #(>= (aget b (int (+ at (long %)))) threshold) (range v)))))
        argmax (fn [^floats x ^long at ^long v]
                 (reduce (fn [best i] (if (> (aget x (int (+ at (long i)))) (aget x (int (+ at (long best))))) i best)) 0 (range v)))]
    (reduce (fn [acc [^floats a ^floats b at v]]
              (-> acc
                  (update :positions inc)
                  (update :argmax-equal + (if (= (argmax a at v) (argmax b at v)) 1 0))
                  (update :max-diff max (reduce max 0.0 (map #(Math/abs (- (double (aget a (int (+ (long at) (long %)))))
                                                                          (double (aget b (int (+ (long at) (long %)))))))
                                                             (top b at v))))))
            {:positions 0 :argmax-equal 0 :max-diff 0.0}
            rows)))

(deftest ^:slow a-checkpoint-is-its-gguf
  (let [dir (System/getProperty "gen.hf.checkpoint")
        gguf-path (System/getProperty "gen.hf.gguf")]
    (cond
      (not (and dir gguf-path))
      (println "gen.llm.models.checkpoint-real-test: skipped (-J-Dgen.hf.checkpoint=<dir> -J-Dgen.hf.gguf=<gguf>)")

      (ts/jvm?)
      (println "gen.llm.models.checkpoint-real-test: skipped on the JVM backend (run it on MLX)")

      :else
      (let [{ld :load forward :forward} (family gguf-path)
            gaps (let [a (ld dir {:dtype :bfloat16}) b (ld gguf-path {:dtype :bfloat16})]
                   (is (= (:config b) (:config a)) "the config")
                   (weight-gaps a b))
            _ (System/gc)
            ours (log-probs forward (ld dir))
            _ (System/gc)
            theirs (log-probs forward (ld gguf-path))
            {:keys [max-diff argmax-equal positions]} (compare-rows ours theirs)]
        (println "checkpoint-real-test:" dir "against" gguf-path)
        (println "  weights differing (bfloat16):" (count gaps) (pr-str (take 8 gaps)))
        (println "  max |Δ log p| over the GGUF's top 64:" max-diff "; argmax equal" argmax-equal "of" positions)
        (is (every? (fn [[path d]] (or (= :a (peek path)) (<= d 5.960464477539063E-8))) gaps)
            "only ssm_a differs, and an F16 GGUF's values below float16's normal range, by its subnormals' spacing")
        (is (<= max-diff 1e-3))
        (is (= positions argmax-equal))))))
