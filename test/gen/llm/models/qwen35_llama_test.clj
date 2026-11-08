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
   the 12 largest of each set are of readings whose forwards hold a
   near-tie of the router's top k (its least log p_8/p_9 over the layers
   and positions read 9e-7 to 5.4e-5, but one of 1.5e-4 on the digits);
   on the general prompts the readings of least margin 1e-4 or more (99
   of 200) are within 3.0e-3 (bench/gen/qwen3_llama_noise.clj's margins,
   its docstring the table). This test's 5 prompts measured within the
   band, and every planted error past mutation-nats on the 44-token one.
   llama.cpp's own spread on Metal on this file reaches 2.65 (general)
   and 7.11 (digits, a fork against a fresh read). The planted errors
   there, the least and the median over the 20 general prompts: the conv
   state's oldest inputs 2.34 and 5.81; the conv kernel reversed 9.06 and
   13.8; attn_q's query and gate swapped 10.3 and 15.8; KV heads tiled
   2.65 and 4.52; the q/k norms skipped 10.9 and 13.1; RoPE's traditional
   pairs 0.16 and 0.48; the value heads grouped 7.41 and 14.2; the
   experts' gate and up swapped 9.63 and 14.2; the router's top k not
   renormalized 2.89 and 6.13; the shared expert ungated 7.23 and 13.8.
   With llama.cpp's experts forced (the owner's call, 2026-11-08): on a
   mixture of experts llama.cpp's engine records the experts each layer
   picks at each position (gen.llm.llama's :record #{\"ffn_moe_topk\"},
   its eval callback; against-llama/reference's :experts), and each
   reading is compared again with them forced on the forward
   (compare-readings' :forced), within against-llama/forced-band; the
   planted errors, forced, must pass it too where it is tighter than
   mutation-nats (on Metal), else mutation-nats as unforced. The
   readings with the experts recorded (the eval callback computes the
   graph in pieces) are llama.cpp's too: unforced they keep the bands
   above. On Metal it is 0.16 nats: twice the
   largest |Δ log p| (8.0e-2, rounded up) the bench measured forced on
   the MacBook (2026-11-08, the readings recording the experts, two
   JVMs): 6.4e-3 on the general prompts (median 1.1e-3), 8.0e-2 on the
   digits (median 5.5e-3 to 7.3e-3; the chat templates' within 1.6e-2,
   the bare digit runs 2.6e-2 to 8.0e-2); the argmax equal in all 400,
   the top 5 in 399. The same readings unforced: 0.97 and 5.97. This
   test's prompts forced: 1.1e-3 to 2.0e-3 up to 15 tokens, 6.8e-2 (44)
   and 9.2e-2 (157); unforced 0.19 and 1.45. The planted errors forced,
   the least and the median over the general prompts: the conv state's
   oldest inputs 2.23 and 4.47; the conv kernel reversed 8.97 and 12.1;
   attn_q's query and gate swapped 10.6 and 16.5; KV heads tiled 2.43
   and 4.21; the q/k norms skipped 8.98 and 12.7; RoPE's traditional
   pairs 0.056 and 0.20 (0.95 on this test's 44 tokens); the value heads
   grouped 6.62 and 13.3; the experts' gate and up swapped 6.21 and 12.2;
   the router's top k not renormalized 2.97 and 6.41; the shared expert
   ungated 7.59 and 13.6. So every one but RoPE's pairs is past the
   forced band on every prompt, where the unforced 11.7 held them all.
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
   and 13.8.
   Forced off the Mac the band is 3.9 nats (against-llama/forced-band):
   twice the largest |Δ log p| (1.91, rounded up) the bench measured
   forced on fenrir (2026-11-08, llama.cpp's CUDA build recording its
   experts, MLX's forward on CUDA): 0.45 on the general prompts (median
   0.10), 1.91 on the digits (median 0.16 to 0.21; one step's, the next
   0.78), against 2.19 and 6.24 unforced on the same readings; the argmax
   equal in 399 of 400 (393 unforced). The planted errors forced, the
   least and the median over the general prompts: the conv state's
   oldest inputs 2.04 and 4.38; the conv kernel reversed 9.59 and 12.4;
   attn_q's query and gate swapped 11.2 and 16.7; KV heads tiled 2.46 and
   4.35; the q/k norms skipped 8.82 and 12.5; RoPE's traditional pairs
   0.13 and 0.26; the value heads grouped 8.22 and 14.2; the experts' gate
   and up swapped 5.81 and 12.7; the router's top k not renormalized 3.52
   and 6.18; the shared expert ungated 7.74 and 13.6. Six of the ten are
   past the forced band on every general prompt (none past the unforced
   12.3 on every one); llama.cpp's 8-bit activations keep the rest inside
   it, as they keep Qwen3.5's dense errors inside 7.1. So off the Mac the
   forced planted errors are asserted past mutation-nats, as unforced,
   and Metal's forced band is the check of the conventions."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.llm.branching :as br]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  ;; Per platform and quantization, from the measurements above
  ;; (against-llama/band); 0 without a test model (the tests skip)
  (if (and model-path (.isFile (java.io.File. ^String model-path))) (against/band model-path) 0.0))

(def ^:private forced-band
  ;; The band with llama.cpp's experts forced on a mixture of experts
  ;; (against-llama/forced-band); 0 without a test model
  (if (and model-path (.isFile (java.io.File. ^String model-path))) (against/forced-band model-path) 0.0))

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

(defn- experts?
  "Is the GGUF file at path a mixture of experts (its architecture's
   expert_count)?"
  [path]
  (let [md (:metadata ((requiring-resolve 'gen.array.gguf/header) path))]
    (pos? (long (or (get md (str (get md "general.architecture") ".expert_count")) 0)))))

(defn- setup
  "[readings model]: llama.cpp's readings of each of prompts
   (against-llama/reference's, 4 greedy steps) and the forward's model, of
   the test model, made once; on a mixture of experts with llama.cpp's
   experts (:experts, its engine recording them). llama.cpp's engine is
   closed before the model loads, so that the two are never in memory at
   once (on Qwen3.6-35B-A3B 21 and 28 GB, more than a 64 GB Mac holds
   beside its own)."
  []
  (or @shared
      (let [rs (with-open [^java.io.Closeable e ((llama 'engine) (cond-> {:model model-path :max-branches 4 :n-ctx 1024}
                                                                   (experts? model-path) (assoc :record #{"ffn_moe_topk"})))]
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

(defn- forced
  "Readings r with llama.cpp's experts forced on the forward
   (compare-readings' :forced)."
  [r]
  (assoc r :forced (:experts r)))

(deftest log-probabilities-are-llama-cpp-s
  (when-qwen35
    (let [[rs m] (setup)]
      (doseq [[text r] (map vector prompts rs)]
        (testing (str (count (:tokens r)) " tokens: " (pr-str (subs text 0 (min 30 (count text)))))
          (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-readings m r band)]
            (is (<= max-diff band) (str what ": |Δ log p| " max-diff))
            (is argmax? (str what ": the argmax"))
            (is top5? (str what ": the top 5")))
          (when (:experts r)
            (testing ", llama.cpp's experts forced"
              (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-readings m (forced r) forced-band)]
                (is (<= max-diff forced-band) (str what ": |Δ log p| " max-diff))
                (is argmax? (str what ": the argmax"))
                (is top5? (str what ": the top 5"))))))))))

(deftest planted-convention-errors-fail-by-nats
  (when-qwen35
    (let [[rs m] (setup)
          readings (nth rs 3)]
      (doseq [mutation (sort (keys (against/mutations-of m)))]
        (is (< mutation-nats (against/mutated m readings mutation)) (name mutation))
        ;; forced, past the forced band where it is the tighter (Metal's),
        ;; else past mutation-nats as unforced
        (when (:experts readings)
          (is (< (min forced-band mutation-nats) (against/mutated m (forced readings) mutation))
              (str (name mutation) ", llama.cpp's experts forced")))))))

;; ---------------------------------------------------------------------------
;; llama.cpp's experts (gen.llm.llama's :record, its eval callback) forced
;; on the tiny MoE (opt-in, as above)
;; ---------------------------------------------------------------------------

(def ^:private forced-vocab-band
  "The tiny MoE's band with llama.cpp's experts forced: 0.04, twice the
   largest |Δ log p| (1.9e-2, rounded up) measured on Metal (Mac mini M4,
   2026-11-08) over vocab-prompts, F32 and Q8_0, each read whole, as a
   chunk and through 8 steps; unforced the largest is 2.3e-2 (a routing
   flip on the Q8_0's 44 tokens, forced 1.2e-2). Random weights: the
   rest of the gap is not the routing's. llama.cpp's readings recording
   its experts are within 2.2e-4 of its readings without (the callback
   computes the graph in pieces)."
  0.04)

(defn- layer-off
  "A planted bug: each layer the experts of the layer before (the first
   the last's)."
  [experts]
  (mapv (fn [ls] (into [(peek ls)] (pop ls))) experts))

(defn- position-off
  "A planted bug: each position the experts of the next (the last the
   first's)."
  [experts]
  (conj (vec (rest experts)) (first experts)))

(def ^:private moe-vocab-models ["qwen35moe-vocab-f32" "qwen35moe-vocab-q8_0"])

(deftest llama-cpp-s-experts-forced-agree-tightly
  ;; the planted bugs measured 7.1 to 8.8 nats (one layer off) and 8.3 to
  ;; 10.0 (one position off) on Metal, 2026-11-08
  (if (nil? vocab-models)
    (println "gen.llm.models.qwen35-llama-test: llama.cpp's experts are forced on the tiny MoE given"
             "-J-Dgen.qwen35.vocab-models=<dir> (test/fixtures/gguf/generate.py --with-vocabulary)")
    (ts/when-native :llama vocab-models?
      (doseq [f moe-vocab-models
              :let [path (str vocab-models "/" f ".gguf")
                    m (qwen35/load path)]]
        (with-open [^java.io.Closeable e ((llama 'engine) {:model path :max-branches 4 :n-ctx 1024
                                                           :record #{"ffn_moe_topk"}})]
          (doseq [text vocab-prompts
                  :let [tokens ((llama 'tokenize) e text)
                        n (count tokens)
                        r (against/reference e tokens 4)
                        worst (fn [ex] (reduce max (map :max-diff (against/compare-readings m (assoc r :forced ex) 0))))]]
            (testing (str f ", " n " tokens")
              (is (= (+ n 4) (count (:experts r))) "the experts of the prompt's positions and the 4 steps'")
              (is (every? #(= 4 (count %)) (:experts r)) "of each of the 4 layers")
              (is (= (+ 4 1) (count (keep peek (:experts r))))
                  "the last layer's only where logits are read: the prompt's last position, each step's")
              (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-readings m (forced r) forced-vocab-band)]
                (is (<= max-diff forced-vocab-band) (str what ": |Δ log p| " max-diff))
                (is argmax? (str what ": the argmax"))
                (is top5? (str what ": the top 5")))
              (is (< mutation-nats (worst (layer-off (:experts r)))) "the experts read one layer off")
              (is (< mutation-nats (worst (position-off (:experts r)))) "the experts read one position off"))))))))

(defn- router-top-k?
  "Are indices (llama.cpp's ffn_moe_topk row) the k most probable of
   probs (its ffn_moe_probs row): k distinct indices whose probabilities
   do not increase, none below another expert's?"
  [indices probs]
  (let [ps (mapv #(nth probs %) indices)
        others (keep-indexed (fn [i p] (when-not (some #{i} indices) p)) probs)]
    (and (apply distinct? indices)
         (apply >= ps)
         (every? #(<= % (peek ps)) others))))

(deftest recorded-experts-are-the-router-s-top-k
  ;; the callback's own check: each layer's recorded top k, at each
  ;; position, is the top k of the router's probabilities recorded beside
  ;; it (ffn_moe_probs, F32, every expert's); with the next layer's
  ;; probabilities (a planted bug: the tensors' layers mixed up) it fails
  (if (nil? vocab-models)
    (println "gen.llm.models.qwen35-llama-test: llama.cpp's recorded experts are checked given"
             "-J-Dgen.qwen35.vocab-models=<dir>")
    (ts/when-native :llama vocab-models?
      (doseq [f moe-vocab-models
              :let [path (str vocab-models "/" f ".gguf")]]
        (with-open [^java.io.Closeable e ((llama 'engine) {:model path :max-branches 4 :n-ctx 1024
                                                           :record #{"ffn_moe_topk" "ffn_moe_probs"}})]
          (let [tokens ((llama 'tokenize) e (nth prompts 3))
                root (br/open e tokens)
                [b] (br/advance [(br/fork root)] [11])
                ds ((llama 'recorded) e)
                pairs (for [{:keys [tensors]} ds
                            [l rows] (get tensors "ffn_moe_topk")
                            [p idx] rows]
                        [l p idx])
                probs (fn [d l p] (get-in d [:tensors "ffn_moe_probs" l p]))
                check (fn [shift] (for [d ds
                                        [l rows] (get-in d [:tensors "ffn_moe_topk"])
                                        :let [l' (mod (+ l shift) 4)]
                                        [p idx] rows
                                        :when (probs d l' p)]
                                    (router-top-k? idx (probs d l' p))))]
            (testing f
              (is (= 2 (count ds)) "the prompt and the step")
              (is (< 100 (count pairs)))
              (is (every? (fn [[_ _ idx]] (= 2 (count idx))) pairs) "2 experts a position")
              (is (every? true? (check 0)) "each recorded top k is the router's")
              (is (some false? (check 1)) "the next layer's probabilities are not"))
            (br/dispose! b)
            (br/dispose! root)))))))
