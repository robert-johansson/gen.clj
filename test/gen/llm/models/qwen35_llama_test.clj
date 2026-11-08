(ns gen.llm.models.qwen35-llama-test
  "Qwen3.5's forward over gen.array (gen.llm.models.qwen35) against the
   llama.cpp engine (gen.llm.llama) on the same GGUF: the check of its
   conventions, which a wrong one passes on the tiny models (the conv
   kernel applied newest input first, attn_q's query and output gate
   swapped, KV heads tiled are all self-consistent), as
   gen.llm.models.qwen3-llama-test checks Qwen3's.

   Needs the MLX backend (-J-Dgen.backend=mlx), llama.cpp's libraries and
   a Qwen3.5 GGUF given as -J-Dgen.llama.test-model (or
   GEN_LLAMA_TEST_MODEL), e.g. Qwen3.5-0.8B-Q8_0. Skipped with a note
   otherwise, and for a model of another architecture. The same token ids
   (llama.cpp's tokenizer) go to both; for each prompt, read whole, as a
   prefill and a chunk, and through 4 greedy steps (which carry the KV
   cache, the conv state and the recurrent state), the next-token
   log-probabilities agree within the band over llama.cpp's 64 most
   likely tokens, with the same argmax and top 5 (as sets, but for a swap
   across the boundary between tokens llama.cpp holds within the band of
   each other).

   The band, 0.035 nats: twice the largest |Δ log p| (1.6e-2, rounded
   up) that bench/gen/qwen3_llama_noise.clj measured between the two
   engines on Metal (Mac mini M4, 2026-11-07, Qwen3.5-0.8B-Q8_0, MLX's
   quantized matmuls with float32 scales and the gated-delta Metal
   kernel) over the 20 prompts of gen.llama-noise, each read whole, as a
   chunk and through 8 steps: 200 comparisons, median 1.2e-3 (whole and
   chunk) and 1.35e-3 (steps), the largest a chat template's 29 tokens;
   the argmax and the top 5 equal in all. This test's prompts, of 2 to
   157 tokens, differ by at most 4.3e-3.
   The planted convention errors fail it by nats (against-llama's
   qwen35-mutations; the least and the median over the bench's 20
   prompts; this test asserts ten times qwen3-llama-test's band on the
   prompt of 44 tokens, where the least, RoPE's traditional pairs,
   measured 1.25): the conv state's oldest inputs, 1.4 and 4.3 (it shows
   only in a cached read); the conv kernel reversed, 8.5 and 14.0;
   attn_q's query and gate swapped, 14.0 and 22.3; KV heads tiled, 5.6
   and 9.8; the q/k norms skipped, 11.1 and 16.4; RoPE's traditional
   pairs, 0.14 and 0.34 (RoPE covers 64 of a head's 256 dims, in 6 of 24
   layers: a short prompt barely sees it).
   The value-head order the converter tiles is the identity at 0.8B (16
   key and 16 value heads), and the 0.8B has no experts: a trained model
   checks both off the Mac (Qwen3.6-35B-A3B, below), and the tiny random
   models generate.py --with-vocabulary writes (2 key and 4 value heads;
   a Qwen3.5-MoE of 4 experts, 2 used, and a shared one; the 0.8B's
   tokenizer, as llama.cpp needs one) check both, given as
   -J-Dgen.qwen35.vocab-models=<their directory> (an opt-in check, as
   they are generated: without it a note, and no native suite skipped). Their band, 0.05 nats: twice the largest |Δ log p|
   measured on Metal (2026-11-07), 2.3e-2 over the 3 prompts of 8, 44
   and 157 tokens of vocab-prompts, F32 and Q8_0, dense and MoE (random
   weights: their attention and states are less tame than a trained
   model's); the value heads grouped where the converter tiles them fail
   it by 7.1 to 11.1 nats.
   On Metal a model of K-quants has the band 11.7 nats
   (against-llama/band): twice the largest |Δ log p| (5.85, rounded up)
   that the bench measured on the MacBook M2 Max (2026-11-08,
   Qwen3.6-35B-A3B-Q4_K_M against llama.cpp v0.9.0's Metal build, the
   forward in float32; gen.llama-noise's 20 general prompts and its 20 of
   digits, each read whole, as a chunk and through 8 steps; llama.cpp's
   readings taken in a JVM of their own, as the 28 GB model and
   llama.cpp's 21 GB do not fit the MacBook together): 0.78 on the
   general prompts (median 1.8e-3; 12 of the 20 prompts within 3e-3, as
   a dense model's), 5.85 on the digits (median 0.17-0.24); the argmax
   equal in 199 and 200 of 200. The largest gaps are routing near-ties:
   every gap above 0.27 is of a reading whose forwards hold a near-tie of
   the router's top k (its least log p_8/p_9 over the layers and
   positions read 9e-7 to 5.4e-5, but one of 1.5e-4 on the digits); on
   the general prompts the readings of least margin 1e-4 or more (99 of
   200) are within 3.0e-3 (bench/gen/qwen3_llama_noise.clj's margins).
   llama.cpp's own spread on Metal on this file reaches 2.65 (general)
   and 7.11 (digits, a fork against a fresh read). The planted errors
   there, the least and the median over the 20 general prompts: the conv
   state's oldest inputs 2.34 and 5.81; the conv kernel reversed 9.06 and
   13.8; attn_q's query and gate swapped 10.3 and 15.8; KV heads tiled
   2.65 and 4.52; the q/k norms skipped 10.9 and 13.1; RoPE's traditional
   pairs 0.16 and 0.48; the value heads grouped 7.41 and 14.2; the
   experts' gate and up swapped 9.63 and 14.2; the router's top k not
   renormalized 2.89 and 6.13; the shared expert ungated 7.23 and 13.8.
   Off the Mac the band is 7.1 nats, and 12.3 on a model of K-quants
   (against-llama/band): twice the largest |Δ log p| (rounded up) that
   the bench measured on fenrir as qwen3-llama-test's (2026-11-08, MLX's
   forward on CUDA against llama.cpp v0.9.0's CUDA and x86 CPU builds,
   which quantize the activations to 8 bits; gen.llama-noise's 20 prompts
   and its 20 of digits, each read whole, as a chunk and through 8 steps):
   - Qwen3.5-0.8B-Q8_0: 0.88 on the general prompts (CUDA; 0.87 on the
     CPU build), 3.38 on the digits (CPU 3.53), the largest at greedy
     steps of the chat prompts' replies; median 0.16 to 0.30; the argmax
     equal in 195 to 197 of each 200. llama.cpp's own spread on CUDA:
     0.57 and 2.59. MLX's forward on its CPU gives the same gaps to three
     digits (the digit prompts): the gap is llama.cpp's.
   - Qwen3.6-35B-A3B-Q4_K_M (ggml-org's: Q4_K experts, Q8_0 attention and
     shared expert; 256 experts, 8 used; 16 key and 32 value heads): 2.35
     on the general prompts, 6.13 on the digits (CUDA; the CPU build 1.87
     and 5.88), median 0.25 to 0.88; the argmax equal in 196 and 197 of
     200. llama.cpp's own spread on CUDA is as large, 3.88 and 6.09, and
     MLX's forward on its CPU gives the CUDA gaps to three digits (the
     first 3 digit prompts). The largest gaps are a runner-up's behind a
     top token of p ≈ 1 (llama.cpp -10.9 nats, ours -17.0).
   There the planted errors measured, the least and the median over the
   20 general prompts (0.8B on CUDA; the MoE): the conv state's oldest
   inputs 1.37 and 4.80, 2.34 and 5.92; the conv kernel reversed 8.21
   and 14.4, 9.73 and 13.8; attn_q's query and gate swapped 13.9 and
   22.2, 10.3 and 15.9; KV heads tiled 5.62 and 9.72, 2.65 and 4.34; the
   q/k norms skipped 10.9 and 16.4, 11.2 and 13.1; RoPE's traditional
   pairs 0.14 and 0.39, 0.19 and 0.66. And on the MoE (against-llama's
   value-heads-mutations and moe-mutations): the value heads grouped 6.80
   and 14.2; the experts' gate and up swapped 9.57 and 14.2; the router's
   top k not renormalized 2.78 and 6.08; the shared expert ungated 7.18
   and 13.8."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  ;; Per platform and quantization, from the measurements above
  ;; (against-llama/band); 0 without a test model (the tests skip)
  (if (and model-path (.isFile (java.io.File. ^String model-path))) (against/band model-path) 0.0))

(def ^:private mutation-nats
  "What a planted convention error must exceed, on every platform:
   qwen3-llama-test's, ten times its Metal band. Off the Mac most of the
   errors lie inside the band (llama.cpp's noise there), and Metal's band
   is the check of the conventions."
  0.75)

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the Qwen3.5-against-llama.cpp tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model's forward needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private qwen35?
  (delay (contains? #{"qwen35" "qwen35moe"} (against/architecture model-path))))

(def ^:private shared (atom nil))

(use-fixtures :once (fn [f] (try (f) (finally (reset! shared nil)))))

(defmacro ^:private when-qwen35
  "body when llama.cpp, MLX and a Qwen3.5 test model are here (counted as
   gen.test-support/when-native counts); a note for a model of another
   architecture."
  [& body]
  `(ts/when-native :llama available?
     (if @qwen35?
       (do ~@body)
       (println "gen.llm.models.qwen35-llama-test: the test model is not a Qwen3.5; nothing to compare"))))

(def ^:private prompts
  "Prompts of 2 to about 60 tokens, and one of about 150 (a prefill the
   gated delta rule reads in chunks where it has no kernel)."
  ["The capital"
   "The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   (str "In probabilistic programming, a generative function is a program whose random choices are named, "
        "so that inference can constrain some of them to observations and propose the others; its density "
        "is the product of each choice's, and")
   (apply str (repeat 6 (str "A recurrent layer keeps a state of fixed size, which each token decays, "
                             "corrects and reads; attention keeps every token. ")))])

(defn- setup
  "[readings model]: llama.cpp's readings of each of prompts
   (against-llama/reference's, 4 greedy steps) and the forward's model, of
   the test model, made once. llama.cpp's engine is closed before the
   model loads, so that the two are never in memory at once (on
   Qwen3.6-35B-A3B 21 and 28 GB, more than a 64 GB Mac holds beside its
   own)."
  []
  (or @shared
      (let [rs (with-open [^java.io.Closeable e ((llama 'engine) {:model model-path :max-branches 4 :n-ctx 1024})]
                 (mapv #(against/reference e ((llama 'tokenize) e %) 4) prompts))]
        (reset! shared [rs (qwen35/load model-path)]))))

;; ---------------------------------------------------------------------------
;; Tiny random models with the 0.8B's vocabulary: unequal heads, experts
;; ---------------------------------------------------------------------------

(def ^:private vocab-models
  (System/getProperty "gen.qwen35.vocab-models"))

(def ^:private vocab-models?
  (delay
    (let [skip (fn [why] (println "Skipping the tiny Qwen3.5-against-llama.cpp tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (not (.isFile (java.io.File. (str vocab-models "/qwen35-vocab-f32.gguf")))) (skip (str "no models in " vocab-models))
        (ts/jvm?) (skip "the JVM backend: a vocabulary of 248320 needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t (skip (str (or (ex-message t) (.getName (class t)))))))))))

(def ^:private vocab-band 0.05)

(def ^:private vocab-prompts
  [(second prompts) (nth prompts 3) (nth prompts 4)])

(deftest unequal-heads-and-experts-are-llama-cpp-s
  ;; opt-in: the models are generated (with Python), not committed, so a
  ;; run without them skips no native suite
  (if (nil? vocab-models)
    (println "gen.llm.models.qwen35-llama-test: the tiny models with a vocabulary are checked given"
             "-J-Dgen.qwen35.vocab-models=<dir> (test/fixtures/gguf/generate.py --with-vocabulary)")
    (ts/when-native :llama vocab-models?
     (doseq [f ["qwen35-vocab-f32" "qwen35-vocab-q8_0" "qwen35moe-vocab-f32" "qwen35moe-vocab-q8_0"]
            :let [path (str vocab-models "/" f ".gguf")
                  m (qwen35/load path)]]
      (with-open [^java.io.Closeable e ((llama 'engine) {:model path :max-branches 4 :n-ctx 1024})]
        (doseq [text vocab-prompts
                :let [tokens ((llama 'tokenize) e text)]]
          (testing (str f ", " (count tokens) " tokens")
            (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-prompt e m tokens 4 vocab-band)]
              (is (<= max-diff vocab-band) (str what ": |Δ log p| " max-diff))
              (is argmax? (str what ": the argmax"))
              (is top5? (str what ": the top 5")))
            (is (< mutation-nats (against/mutated e m tokens :value-heads-grouped)) "the value heads grouped"))))))))

(deftest log-probabilities-are-llama-cpp-s
  (when-qwen35
    (let [[rs m] (setup)]
      (doseq [[text r] (map vector prompts rs)]
        (testing (str (count (:tokens r)) " tokens: " (pr-str (subs text 0 (min 30 (count text)))))
          (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-readings m r band)]
            (is (<= max-diff band) (str what ": |Δ log p| " max-diff))
            (is argmax? (str what ": the argmax"))
            (is top5? (str what ": the top 5"))))))))

(deftest planted-convention-errors-fail-by-nats
  (when-qwen35
    (let [[rs m] (setup)
          readings (nth rs 3)]
      (doseq [mutation (sort (keys (against/mutations-of m)))]
        (is (< mutation-nats (against/mutated m readings mutation)) (name mutation))))))
