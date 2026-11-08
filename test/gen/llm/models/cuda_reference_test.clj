(ns gen.llm.models.cuda-reference-test
  "MLX's forward on NVIDIA against a tight reference (genclj-gddh; the
   owner's call, 2026-11-08): a real model's next-token log-probabilities
   on MLX's CUDA backend, in float32 and in float16 (the engine's default
   there for a quantized file), against the same forward on MLX's CPU
   backend in float32, on the same GGUF. (MLX's CPU computes float16
   over 40 times slower than float32 on fenrir: 6 prompts of Qwen3-0.6B
   took over 25 minutes against 34 s, so the MoE's would take hours.) Off
   the Mac the comparisons against llama.cpp take bands of nats
   (against-llama/band: llama.cpp's CUDA and x86 builds round activations
   to 8 bits), inside which most planted convention errors fall; in
   float32 this reference's bands are tight (5.1e-4 to 0.80 nats), and
   every planted error (against-llama/mutations-of) fails them. A bug that
   MLX's CPU and CUDA code share is left to the Mac's tight bands against
   llama.cpp and to the conformance suite against the JVM backend.

   Needs MLX's CUDA as the backend (-J-Dgen.backend=mlx off macOS, its
   GPU), a Qwen3, Qwen3.5 or Qwen3.6 GGUF given as
   -J-Dgen.llama.test-model (or GEN_LLAMA_TEST_MODEL), and llama.cpp's
   library for the tokenizer (its vocabulary loaded alone). Skipped with a
   note otherwise, counted as the native suite cuda (gen.test-support's
   when-native: -Dgen.require-native=cuda fails a run that skips it). The
   same token ids go to both; for each prompt, read whole, as a prefill
   and a chunk, and through 4 greedy steps (the CPU's argmax), the
   next-token log-probabilities agree within the band over the CPU's 64
   most likely tokens, with the same argmax and top 5 (as sets, but for a
   swap across the boundary between tokens the CPU holds within the band
   of each other). The CPU's readings are taken first, once for both
   dtypes, its model then dropped, so that one model is in memory at a
   time. ^:slow: the namespace takes under a minute on the 0.6B and 0.8B
   and about 5 on the MoE, most of it the CPU's readings.

   The bands, by model and dtype: twice the largest |Δ log p| that
   bench/gen/cuda_reference_noise.clj measured, rounded up (fenrir's RTX
   PRO 6000, sm_120, MLX 0.32.4 built from source, 2026-11-08), over
   gen.llama-noise's 20 general and 20 digit prompts, each read whole, as
   a chunk and through 8 steps (400 comparisons a dtype), against the
   CPU's float32 (the largest on the general prompts, on the digits;
   the medians):
   - Qwen3-0.6B Q8_0: float32 2.51e-4, 2.23e-4 (median 3.4e-5 to
     7.8e-5), the band 5.1e-4; float16 9.24e-2, 8.77e-2 (median 1.7e-2
     to 2.5e-2), the band 0.19. The argmax equal in 399 and the top 5 in
     396 of 400 (strictly: no tie), every one in float32.
   - Qwen3.5-0.8B Q8_0: float32 7.06e-5, 3.75e-4 (median 2.8e-5 to
     4.4e-5), the band 7.5e-4; float16 4.73e-2, 0.240 (median 1.5e-2 to
     2.5e-2), the band 0.48, the largest at greedy steps of the chat
     prompts' replies, as against llama.cpp. The argmax equal in 399 and
     the top 5 in 391 of 400, every one in float32.
   - Qwen3.6-35B-A3B Q4_K_M (the MoE: 256 experts, 8 used): float32
     6.20e-5, 0.397 (median 1.1e-5 to 3.1e-5), the band 0.80: 38 of the
     40 prompts within 1.2e-4, two texts of numbers at 0.23 and 0.40, a
     jump rather than a drift: an expert picked otherwise at a near-tie
     of the router's top 8 (the CPU's log p_8/p_9 4.8e-7 and 3.1e-4;
     genclj-fxag, bench/gen/moe_routing.clj). float16 2.71, 6.07 (median
     0.17 to 0.58) with the router in float16; in float32 since
     genclj-fxag 2.74, 6.01 (median 0.12 to 0.36), the band 12.2: as far
     from the float32 forward as llama.cpp's CUDA build is (its band
     12.3), where the dense models' float16 is 0.02 nats away (median).
     The argmax equal in 400 and 394 of 400, the top 5 in 398 and 347.
     Float16 flips the router's top k at 12% of a prompt's layers and
     positions, each flip moving the hidden state and flipping more.
     With the CPU's experts forced (against-llama's :forced) float16 is
     3.97e-2, 0.172 away (median 1.3e-2 to 2.2e-2), the band 0.35: as
     close as the dense models'. The argmax equal in 399 of 400, the top
     5 in 398.

   The planted convention errors are asserted in float32, where the band
   is tight: they are the same code in both dtypes, and the MoE's float16
   band holds most of them. A planted error fails this test when its worst
   gap over the prompts (whole, chunk and one step: against-llama/mutated)
   is outside the band. The bench's, the least and the median over the 20
   general prompts, in float32 (float16's within 0.1 of them on the dense
   models, 1.0 on the MoE):
   - Qwen3-0.6B: KV heads tiled 17.3 and 24.5; the q/k norms skipped
     11.8 and 18.1; RoPE's traditional pairs 2.62 and 6.61.
   - Qwen3.5-0.8B: the conv state's oldest inputs 1.42 and 4.25; the conv
     kernel reversed 8.44 and 14.0; KV heads tiled 5.60 and 9.76; the q/k
     norms skipped 11.1 and 16.4; attn_q's query and gate swapped 14.0
     and 22.2; RoPE's traditional pairs 0.138 and 0.336 (RoPE covers 64 of
     a head's 256 dims, in 6 of 24 layers).
   - Qwen3.6-35B-A3B: the conv state's oldest inputs 2.34 and 5.88; the
     conv kernel reversed 9.05 and 13.8; KV heads tiled 2.69 and 4.52; the
     q/k norms skipped 11.0 and 13.1; attn_q's query and gate swapped 10.3
     and 15.8; RoPE's traditional pairs 0.163 and 0.511 (inside the band
     on most prompts, outside it at worst over this test's); the value
     heads grouped 7.42 and 14.2; the experts' gate and up swapped 9.63
     and 14.2; the router's top k not renormalized 2.89 and 6.13; the
     shared expert ungated 7.24 and 13.7.
   This test's worst gaps (fenrir, 2026-11-08), the least of them:
   Qwen3-0.6B's RoPE pairs 11.6 (the others 23.7 and 27.7); Qwen3.5-0.8B's
   RoPE pairs 1.68 (the others 5.51 to 25.2); Qwen3.6-35B-A3B's RoPE
   pairs 1.19, outside the band 0.80 on 2 of the 5 prompts (the others
   7.25 to 23.0).

   A mixture of experts' float16 is checked with the CPU's experts forced
   too (float16-forward-with-the-cpu-s-experts-is-the-cpu-s), where its
   band is tight and the planted errors are asserted outside it: the
   router's flips at near-ties are float16's own, and forcing the CPU's
   experts leaves the rest of the forward to check. The bench's planted
   errors so (the least and the median over the 20 general prompts):
   RoPE's traditional pairs 0.057 and 0.204 (at most 0.857); the others
   2.25 (the conv state's oldest inputs) to 10.8 at least.

   The Thor (sm_110, aarch64): unmeasured. The test runs there as it is,
   given MLX's CUDA build for sm_110, llama.cpp's aarch64 library (for
   the tokenizer) and the models (the MoE's float32 model, about 31 GB,
   fits its unified memory, one model at a time). Its bands are to be
   measured there by the bench first, and keyed by platform where they
   differ (as against-llama/band's are): the reference is an aarch64 CPU's
   (MLX's NEON kernels, not x86's), and the GPU another architecture."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.llm.models.against-llama :as against]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(defn- mlx
  "A function of gen.backend.mlx, loaded only when the tests run."
  [sym]
  @(requiring-resolve (symbol "gen.backend.mlx" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping MLX's CUDA against its CPU:" why) (ts/native-unavailable! :cuda why) false)]
      (cond
        (ts/jvm?) (skip "the JVM backend (MLX's CUDA is the backend under test: -J-Dgen.backend=mlx off macOS)")
        (not (ts/cuda?)) (skip (str "the backend computes on " (name (array/device array/*backend*)) ", not MLX's CUDA"))
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (not (#{"qwen3" "qwen35" "qwen35moe"} (against/architecture model-path)))
        (skip (str "not a Qwen3, Qwen3.5 or Qwen3.6 model: " (against/architecture model-path)))
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (llama.cpp's library tokenizes: native/build-llama.sh and clojure -T:build java)"))))))))

(defn- band
  "The band of MLX's CUDA forward against its CPU's on the GGUF file at
   path, in dtype (the measurements above): by family (Qwen3; Qwen3.5 and
   Qwen3.6) and quantization (Q8_0; K-quants, the MoE's), and dtype; with
   forced?, a mixture of experts' with the CPU's experts."
  [path dtype forced?]
  (let [qwen35? (#{"qwen35" "qwen35moe"} (against/architecture path))
        k? (against/k-quants? path)]
    (get-in {:qwen3 {:float32 5.1e-4 :float16 0.19}
             :qwen35 {:float32 7.5e-4 :float16 0.48}
             :qwen35-k {:float32 0.80 :float16 12.2}
             :qwen35-k-forced {:float16 0.35}}
            [(cond (and qwen35? k? forced?) :qwen35-k-forced (and qwen35? k?) :qwen35-k qwen35? :qwen35 :else :qwen3) dtype])))

(def ^:private prompts
  "gen.llm.models.qwen35-llama-test's: of 2 to about 60 tokens, and one
   of about 150."
  ["The capital"
   "The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   (str "In probabilistic programming, a generative function is a program whose random choices are named, "
        "so that inference can constrain some of them to observations and propose the others; its density "
        "is the product of each choice's, and")
   (apply str (repeat 6 (str "A recurrent layer keeps a state of fixed size, which each token decays, "
                             "corrects and reads; attention keeps every token. ")))])

(defn- cpu-readings
  "The readings (against-llama/reference's, 4 steps) of each prompt by the
   model loaded on MLX's CPU in float32, which is dropped on return; the
   prompts' token ids by llama.cpp's tokenizer of the model."
  []
  (let [tokenss (with-open [^java.io.Closeable v ((llama 'vocabulary) model-path)]
                  (mapv #((llama 'tokenize) v %) prompts))
        cpu ((mlx 'make-backend) {:device :cpu})
        ref (array/with-backend cpu (against/load-model model-path {:dtype :float32}))]
    (mapv #(against/reference ref % 4) tokenss)))

(def ^:private reference
  "The CPU's readings, taken once for both dtypes."
  (delay (let [r (cpu-readings)] ((mlx 'collect!)) r)))

(defn- check
  "The forward on CUDA in dtype against the CPU's in float32: every reading
   of every prompt within the band; with planted? every planted error
   outside it, its worst gap over the prompts (a reading it would fail).
   With forced?, a mixture of experts' forward takes the CPU's experts
   (against-llama/compare-readings' :forced)."
  [dtype planted? forced?]
  ((mlx 'collect!))
  (let [refs (cond->> @reference forced? (mapv #(assoc % :forced (:experts %))))
        m (against/load-model model-path {:dtype dtype})
        b (band model-path dtype forced?)]
    (doseq [[text {toks :tokens :as r}] (map vector prompts refs)]
      (testing (str (name dtype) (when forced? ", the CPU's experts") ", " (count toks) " tokens: "
                    (pr-str (subs text 0 (min 30 (count text)))))
        (doseq [{:keys [what max-diff argmax? top5?]} (against/compare-readings m r b)]
          (is (<= max-diff b) (str what ": |Δ log p| " max-diff " (band " b ")"))
          (is argmax? (str what ": the argmax"))
          (is top5? (str what ": the top 5")))))
    (when planted?
      (testing (str (name dtype) (when forced? ", the CPU's experts") ", the planted convention errors")
        (doseq [mutation (sort (keys (against/mutations-of m)))]
          (let [ds (mapv #(against/mutated m % mutation) refs)
                d (reduce max ds)]
            (println (format "cuda-reference-test %s%s %s: worst %.3f nats, by prompt %s (band %s)"
                             (name dtype) (if forced? " forced" "") (name mutation) d
                             (mapv #(Double/parseDouble (format "%.3g" %)) ds) b))
            (is (< b d) (str (name mutation) ": |Δ log p| " d ", inside the band " b))))))))

(deftest ^:slow float32-forward-is-the-cpu-s
  (ts/when-native :cuda available?
    (check :float32 true false)))

(deftest ^:slow float16-forward-is-the-cpu-s
  (ts/when-native :cuda available?
    (check :float16 false false)))

(deftest ^:slow float16-forward-with-the-cpu-s-experts-is-the-cpu-s
  (ts/when-native :cuda available?
    (if (:experts (first @reference))
      (check :float16 true true)
      (println "cuda-reference-test: the CPU's experts forced on a mixture of experts only"))))
