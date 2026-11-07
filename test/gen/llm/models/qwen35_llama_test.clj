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
   key and 16 value heads), and the 0.8B has no experts: the tiny random
   models generate.py --with-vocabulary writes (2 key and 4 value heads;
   a Qwen3.5-MoE of 4 experts, 2 used, and a shared one; the 0.8B's
   tokenizer, as llama.cpp needs one) check both, given as
   -J-Dgen.qwen35.vocab-models=<their directory> (an opt-in check, as
   they are generated: without it a note, and no native suite skipped). Their band, 0.05 nats: twice the largest |Δ log p|
   measured on Metal (2026-11-07), 2.3e-2 over the 3 prompts of 8, 44
   and 157 tokens of vocab-prompts, F32 and Q8_0, dense and MoE (random
   weights: their attention and states are less tame than a trained
   model's); the value heads grouped where the converter tiles them fail
   it by 7.1 to 11.1 nats."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  ;; Per platform, as llama.cpp's own bands are (DESIGN §9): Metal's from
  ;; the measurement above; elsewhere qwen3-llama-test's (llama.cpp's x86
  ;; and CUDA Q8_0 paths differ from MLX's forward by 0.27 to 0.32 nats
  ;; for Qwen3 on fenrir; Qwen3.5's unmeasured there)
  (if (.startsWith (System/getProperty "os.name") "Mac") 0.035 0.65))

(def ^:private mutation-nats
  "What a planted convention error must exceed, on every platform:
   qwen3-llama-test's, ten times its Metal band."
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

(defn- setup
  "[engine model]: llama.cpp's engine and the forward's model, of the test
   model, made once."
  []
  (or @shared
      (reset! shared [((llama 'engine) {:model model-path :max-branches 4 :n-ctx 1024}) (qwen35/load model-path)])))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [[^java.io.Closeable e] @shared]
                                      (.close e)
                                      (reset! shared nil))))))

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

(def ^:private value-heads-grouped
  "The planted error: value head h reading key head h / (value-heads /
   key-heads), as Hugging Face's checkpoints order them, where the
   converter tiles them (h mod key-heads)."
  {#'qwen35/key-heads-tiled (fn [y hv]
                              (let [[b t hk d] (array/shape y)]
                                (array/reshape (array/broadcast-to (array/expand-dims y 3) [b t hk (quot hv hk) d])
                                                   [b t hv d])))})

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
            (is (< mutation-nats (with-redefs-fn value-heads-grouped
                                   #(reduce max (map :max-diff (against/compare-prompt e m tokens 1 0)))))
                "the value heads grouped"))))))))

(deftest log-probabilities-are-llama-cpp-s
  (when-qwen35
    (let [[e m] (setup)]
      (doseq [text prompts
              :let [tokens ((llama 'tokenize) e text)]]
        (testing (str (count tokens) " tokens: " (pr-str (subs text 0 (min 30 (count text)))))
          (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-prompt e m tokens 4 band)]
            (is (<= max-diff band) (str what ": |Δ log p| " max-diff))
            (is argmax? (str what ": the argmax"))
            (is top5? (str what ": the top 5"))))))))

(deftest planted-convention-errors-fail-by-nats
  (when-qwen35
    (let [[e m] (setup)
          tokens ((llama 'tokenize) e (nth prompts 3))]
      (doseq [mutation (sort (keys against/qwen35-mutations))]
        (is (< mutation-nats (against/mutated e m tokens mutation)) (name mutation))))))
