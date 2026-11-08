(ns gen.llm.models.qwen3-llama-test
  "Qwen3's forward over gen.array (gen.llm.models.qwen3) against the
   llama.cpp engine (gen.llm.llama) on the same GGUF: the check of its
   conventions, which a wrong one passes on the tiny models (KV heads
   tiled instead of grouped, RoPE's traditional pairs, the q/k norms
   skipped are all self-consistent).

   Needs the MLX backend (-J-Dgen.backend=mlx: a 0.6B model's weights are
   too large for the JVM backend's doubles), llama.cpp's libraries and a
   Qwen3 GGUF given as -J-Dgen.llama.test-model (or GEN_LLAMA_TEST_MODEL),
   e.g. Qwen3-0.6B-Q8_0. Skipped with a note otherwise, and for a model of
   another architecture. The same token ids (llama.cpp's tokenizer) go to
   both; for each prompt, read whole, as a prefill and a chunk, and
   through 4 greedy steps (gen.llm.models.against-llama), the next-token
   log-probabilities agree within the band over llama.cpp's 64 most likely
   tokens, with the same argmax and top 5 (as sets, but for a swap across
   the boundary between tokens llama.cpp holds within the band of each
   other: a tie inside the noise).

   The band, 0.075 nats: twice the largest |Δ log p| (3.6e-2, rounded up)
   that bench/gen/qwen3_llama_noise.clj measured between the two engines
   on Metal (Mac mini M4, 2026-11-07, Qwen3-0.6B-Q8_0, MLX's quantized
   matmuls with float32 scales) over the 20 prompts of gen.llama-noise,
   each read whole, as a chunk and through 8 steps: 200 comparisons,
   median 5.7e-3 (steps) to 7.4e-3 (whole and chunk); the argmax equal in
   all, the top 5 in 199, the other a swap of llama.cpp's 5th and 6th,
   3.4e-3 apart. That is inside llama.cpp's own spread between two of its
   computations there, 5.2e-2 (its band 0.1, gen.llm.llama.bands). This
   test's prompts, of 2 to 44 tokens, differ by at most 8.4e-3.
   On a model of K-quants the band is 0.47 nats, twice 0.231 measured the
   same way (2026-11-08, Qwen3-0.6B quantized as Q4_K_M by the pinned
   llama.cpp's quantizer from ggml-org's F16 file: Q4_K and Q6_K, each
   weight llama.cpp's own value): median 4.0e-2 (steps) to 6.6e-2, the
   argmax equal in all 200, the top 5 in 189. llama.cpp's own spread on
   that file is as large, 0.226 (gen.llama-noise, a batched read against
   a lone one), against 5.2e-2 on Q8_0: the gap is its float noise on
   K-quants.
   The planted convention errors fail it by nats (gen.llm.models.against-
   llama's mutations: the least and the median over the bench's 20
   prompts; this test asserts ten times the band on one prompt, where
   they measured 24.3, 18.9 and 2.6): KV heads tiled, 17.3 and 24.6; the
   q/k norms skipped, 11.9 and 18.1; RoPE's traditional pairs, 2.6 and
   6.6.
   Off the Mac the band is 2.4 nats (against-llama/band): twice the
   largest |Δ log p| (1.19, rounded up) that the bench measured on fenrir
   (2026-11-08, MLX's forward on CUDA against llama.cpp v0.9.0's CUDA and
   x86 CPU builds, Qwen3-0.6B-Q8_0) over 40 prompts, gen.llama-noise's 20
   and its 20 of digits (twelve of GRPO's chat prompts of two-digit
   products, eight texts of numbers: -Dgen.bench.prompts=all), each read
   whole, as a chunk and through 8 steps: 400 comparisons a build, median
   0.17 to 0.41; CUDA's largest 1.05 (1.03 on the digits), the CPU
   build's 1.19 (a chat template read as text); the argmax equal in 195
   to 200 of each 200. Both builds quantize the activations to 8 bits for
   Q8_0's dot products, which Metal's does not, and their own spread
   differs: on CUDA a batched read against a lone one reaches 0.85
   (gen.llama-noise), on the CPU nothing (its reads are deterministic,
   its gap a bias). There the planted errors measured at the least 17.4,
   11.9 and 2.45 nats (CUDA; 2.53 on the CPU build)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array.gguf :as gguf]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- band
  ;; Per platform and quantization, from the measurements above (off the
  ;; Mac, K-quants unmeasured)
  []
  (against/band model-path))

(def ^:private mutation-nats
  "What a planted convention error must exceed, on every platform: ten
   times Metal's band (and above the K-quant band: on the Q4_K_M model
   the errors measured 15.4, 14.4 and 2.6 nats at the least over the
   bench's 20 prompts). Off the Mac it lies inside llama.cpp's noise, and
   the errors' margins over the band are the ones measured above."
  0.75)

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the Qwen3-against-llama.cpp tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model's forward needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private qwen3?
  (delay (= "qwen3" (get-in (gguf/header model-path) [:metadata "general.architecture"]))))

(def ^:private shared (atom nil))

(defn- setup
  "[engine model]: llama.cpp's engine and the forward's model, of the test
   model, made once."
  []
  (or @shared
      (reset! shared [((llama 'engine) {:model model-path :max-branches 4 :n-ctx 1024}) (qwen3/load model-path)])))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [[^java.io.Closeable e] @shared]
                                      (.close e)
                                      (reset! shared nil))))))

(defmacro ^:private when-qwen3
  "body when llama.cpp, MLX and a Qwen3 test model are here (counted as
   gen.test-support/when-native counts); a note for a model of another
   architecture."
  [& body]
  `(ts/when-native :llama available?
     (if @qwen3?
       (do ~@body)
       (println "gen.llm.models.qwen3-llama-test: the test model is not a Qwen3; nothing to compare"))))

(def ^:private prompts
  "Prompts of 2 to about 60 tokens."
  ["The capital"
   "The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   (str "In probabilistic programming, a generative function is a program whose random choices are named, "
        "so that inference can constrain some of them to observations and propose the others; its density "
        "is the product of each choice's, and")])

(deftest log-probabilities-are-llama-cpp-s
  (when-qwen3
    (let [[e m] (setup)]
      (doseq [text prompts
              :let [tokens ((llama 'tokenize) e text)]]
        (testing (str (count tokens) " tokens: " (pr-str (subs text 0 (min 30 (count text)))))
          (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-prompt e m tokens 4 (band))]
            (is (<= max-diff (band)) (str what ": |Δ log p| " max-diff))
            (is argmax? (str what ": the argmax"))
            (is top5? (str what ": the top 5"))))))))

(deftest planted-convention-errors-fail-by-nats
  (when-qwen3
    (let [[e m] (setup)
          tokens ((llama 'tokenize) e (second prompts))]
      (doseq [mutation (sort (keys against/mutations))]
        (is (< mutation-nats (against/mutated e m tokens mutation)) (name mutation))))))
