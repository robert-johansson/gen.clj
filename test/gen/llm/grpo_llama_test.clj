(ns gen.llm.grpo-llama-test
  "GRPO (gen.llm.grpo) on a real model: Qwen3-0.6B, a LoRA adapter (rank
   8, alpha 16, on q, k, v, o, gate, up and down of the last 8 blocks)
   trained on a verifiable task of medium difficulty, two-digit products
   (\"What is 64 * 43? Reply with the number only.\"), the reward 1 when
   the reply's last integer is the product (exact-match's :extract):
   - the base model gets 59% of them at temperature 1, its groups of 8
     mostly mixed (11 of 16 in a probe), so most groups carry a gradient;
     it never replies with the number alone, which a format reward could
     not teach (\"RLVR sharpens, it does not teach\");
   - 512 training prompts, 128 held out (no pair of factors shared);
   - rounds of 8 prompts × 8 completions at temperature 1 (the policy),
     at most 32 tokens, :grpo (β 0.04 to the base), μ = 2 optimizer steps
     a round, AdamW (no weight decay) clipped to norm 1, checkpointed;
   - at the first round's first step every ratio π_θ/π_θold is 1 but for
     float32's round-off between the array engine's decode and the
     training forward: each completion's tokens are read where they were
     drawn;
   - the training reward by rounds, printed (a round's 8 prompts differ
     in difficulty: too noisy a measure to assert on);
   - the held-out pass rate (8 completions a prompt at temperature 1,
     the same keys before and after) before and after, and its paired
     difference by prompt with its standard error, asserted at 3; the
     greedy pass rate; the pass rate at each checkpoint (4 a prompt);
   - the adapter, saved in the last checkpoint, loads in llama.cpp
     (gen.llm.llama's :adapters) and reads as the forward within the
     platform's band (qwen3-llama-test's), beside the base model's own
     gap on the same prompts; llama.cpp's greedy replies are scored as
     the array engine's are.
   Prints the numbers. Needs the MLX backend, llama.cpp's libraries and
   Qwen3-0.6B's GGUF as -J-Dgen.llama.test-model (as gen.llm.sft-llama-
   test); skipped with a note otherwise. Rounds, the learning rate and
   the held-out samples a prompt may be changed with -Dgen.grpo.rounds,
   -Dgen.grpo.lr and -Dgen.grpo.samples, for exploration;
   -Dgen.grpo.keep=true keeps the checkpoints. On MLX each round prints
   MLX's active and peak memory, to set its limit (gen.llm.sft's
   docstring)."
  (:require [clojure.test :refer [deftest is]]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.array-engine :as ae]
            [gen.llm.chat :as chat]
            [gen.llm.grpo :as grpo]
            [gen.llm.lm :as lm]
            [gen.llm.lora :as lora]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.sft :as sft]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  "qwen3-llama-test's band, per platform."
  (if (.startsWith (System/getProperty "os.name") "Mac") 0.075 0.65))

(defn- llama
  "A function of gen.llm.llama, loaded only when the test runs."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the GRPO run on a real model:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private qwen3-0-6b?
  (delay (let [{:keys [metadata]} (gguf/header model-path)]
           (and (= "qwen3" (metadata "general.architecture")) (= 28 (metadata "qwen3.block_count"))))))

(defn- property [k default] (if-let [v (System/getProperty (str "gen.grpo." k))] (parse-double v) default))

;; ---------------------------------------------------------------------------
;; The task
;; ---------------------------------------------------------------------------

(defn- product
  "Prompt i: two factors from 12 to 98, drawn from i."
  [i]
  (let [r (java.util.Random. (+ 1000 (long i)))
        a (+ 12 (.nextInt r 87))
        b (+ 12 (.nextInt r 87))]
    {:prompt (str "What is " a " * " b "? Reply with the number only.") :answer (str (* a b)) :factors (hash-set a b)}))

(def ^:private split
  "[training prompts, held-out prompts]: 512 and 128, no pair of factors
   in both."
  (delay (let [train (mapv product (range 512))
               seen (set (map :factors train))
               held (into [] (comp (map product) (remove #(seen (:factors %))) (take 128)) (range 100000 101000))]
           [train held])))

(def ^:private reward (grpo/exact-match {:extract #"-?\d+"}))

(defn- pass-rates
  "Each held-out prompt's pass rate by an LM over engine e: k completions
   a prompt at temperature 1 (greedy for k 1), 16 prompts at a time."
  [e tok prompts k key]
  (vec (mapcat (fn [i ps]
                 (with-open [m (lm/lm e tok {:max-branches (+ (* 16 k) 24) :max-cells (* 16 k 128)})]
                   (:per-prompt (grpo/evaluate m ps reward
                                               {:group-size k :max-tokens 32 :key (rng/fold-in key i)
                                                :sampling (if (= 1 k) {:temp 1.0 :top-k 1 :top-p 1.0} grpo/policy-sampling)}))))
               (range) (partition-all 16 prompts))))

(defn- mean [xs] (/ (reduce + 0.0 xs) (count xs)))

(defn- paired
  "{:before :after :diff :se :z} of two pass rates by prompt."
  [before after]
  (let [ds (map - after before)
        n (count ds)
        d (mean ds)
        se (Math/sqrt (/ (reduce + (map #(let [x (- % d)] (* x x)) ds)) (* (dec n) n)))]
    {:before (mean before) :after (mean after) :diff d :se se :z (/ d se)}))

;; ---------------------------------------------------------------------------
;; The run
;; ---------------------------------------------------------------------------

(deftest ^:slow grpo-raises-a-verifiable-reward-and-llama-cpp-serves-it
  (ts/when-native :llama available?
    (if-not @qwen3-0-6b?
      (println "gen.llm.grpo-llama-test: the test model is not Qwen3-0.6B; nothing run")
      (with-open [^java.io.Closeable vocab ((llama 'vocabulary) model-path)]
        (let [tok ((llama 'tokenizer) vocab)
              m (qwen3/load model-path)
              [train held] @split
              rounds (long (property "rounds" 40))
              samples (long (property "samples" 8))
              n (long (:layers (:config m)))
              ad (lora/init m {:rank 8 :alpha 16 :targets (vec (keys lora/targets)) :layers (range (- n 8) n)} (rng/key 20261107))
              opt (learn/clip-norm (learn/adamw {:lr (property "lr" 5e-5) :weight-decay 0.0}) {:max-norm 1.0})
              dir (str (Files/createTempDirectory "grpo-llama-test" (make-array FileAttribute 0)))
              eval-key (rng/key 99)
              before (with-open [e (ae/engine {:model m :max-branches (+ (* 16 samples) 24) :n-ctx 512})]
                       {:sampled (pass-rates e tok held samples eval-key) :greedy (pass-rates e tok held 1 eval-key)})
              t0 (System/nanoTime)
              {:keys [trainer history]}
              (grpo/train m tok (grpo/trainer ad opt (rng/key 7)) train reward
                          {:rounds rounds :prompts-per-round 8 :group-size 8 :iterations 2 :max-tokens 32 :loss-type :grpo
                           :engine {:n-ctx 512} :save-dir dir :save-every 10
                           :callback (fn [{:keys [round reward-mean reward-std flat-groups entropy length-mean truncated iterations seconds]}]
                                       (println (format "grpo-llama: round %2d reward %.3f ± %.3f flat %d/8 entropy %.3f length %.1f truncated %2d %s rollouts %.1f s train %.1f s"
                                                        round reward-mean reward-std flat-groups entropy length-mean truncated
                                                        (pr-str (mapv (fn [{:keys [loss grad-norm ratio-max clip-fraction kl]}]
                                                                        [(format "loss %.4f" loss) (format "|g| %.3f" grad-norm)
                                                                         (format "max|ρ-1| %.2e" ratio-max) (format "clip %.3f" clip-fraction)
                                                                         (format "kl %.2e" kl)])
                                                                      iterations))
                                                        (:rollouts seconds) (:train seconds)))
                                       (let [{:keys [active peak limit]} ((requiring-resolve 'gen.backend.mlx/memory))]
                                         (println (format "grpo-llama: MLX memory active %.2f GB, peak %.2f GB, limit %.2f GB"
                                                          (/ active 1e9) (/ peak 1e9) (/ limit 1e9))))
                                       (flush))})
              minutes (/ (- (System/nanoTime) t0) 6e10)
              adapted (lora/adapt m (:adapter trainer))
              after (with-open [e (ae/engine {:model adapted :max-branches (+ (* 16 samples) 24) :n-ctx 512})]
                      {:sampled (pass-rates e tok held samples eval-key) :greedy (pass-rates e tok held 1 eval-key)})
              ;; the held-out pass rate at each checkpoint, 4 samples a prompt
              curve (vec (for [r (range 10 (inc rounds) 10)]
                           (let [t (sft/load-checkpoint (str dir (format "/step-%08d" (* 2 r))))]
                             [r (mean (with-open [e (ae/engine {:model (lora/adapt m (:adapter t)) :max-branches 88 :n-ctx 512})]
                                        (pass-rates e tok held 4 eval-key)))])))
              checkpoint (sft/latest-checkpoint dir)
              first-ratio (:ratio-max (first (:iterations (first history))))
              rewards (mapv :reward-mean history)
              sampled (paired (:sampled before) (:sampled after))
              greedy (paired (:greedy before) (:greedy after))]
          (println (format "grpo-llama: %d rounds in %.1f min; the first step's max |ρ - 1| %.3g" (count history) minutes first-ratio))
          (println (format "grpo-llama: the training reward by 10 rounds %s"
                           (pr-str (mapv #(Double/parseDouble (format "%.3f" (mean %))) (partition-all 10 rewards)))))
          (println (format "grpo-llama: held out (%d prompts, %d samples each at temperature 1): before %.3f after %.3f, paired difference %.3f ± %.3f (z %.1f)"
                           (count held) samples (:before sampled) (:after sampled) (:diff sampled) (:se sampled) (:z sampled)))
          (println (format "grpo-llama: held out, greedy: before %.3f after %.3f, paired difference %.3f ± %.3f (z %.1f)"
                           (:before greedy) (:after greedy) (:diff greedy) (:se greedy) (:z greedy)))
          (println (format "grpo-llama: held out by round (4 samples a prompt): %s"
                           (pr-str (mapv (fn [[r p]] [r (Double/parseDouble (format "%.3f" p))]) curve))))
          (is (= (str dir (format "/step-%08d" (* 2 rounds))) checkpoint))
          (is (< first-ratio 1e-3) "the first step's ratios are 1 but for float round-off")
          (is (> (:diff sampled) (* 3 (:se sampled))) "the held-out pass rate rises, at 3 standard errors")
          (with-open [^java.io.Closeable lm-model ((llama 'model) model-path)
                      ^java.io.Closeable a ((llama 'adapter) lm-model (str checkpoint "/adapter.gguf"))
                      ^java.io.Closeable base-e ((llama 'engine) {:model lm-model :max-branches 48 :n-ctx 4096})
                      ^java.io.Closeable e ((llama 'engine) {:model lm-model :adapters {a 1.0} :max-branches 48 :n-ctx 4096})]
            (let [prompt-tokens (fn [p] ((:encode tok) (chat/render [{:role :user :content (:prompt p)}])
                                                       {:add-special? false :parse-special? true}))
                  against-forward (fn [e m] (vec (for [p (take 8 held) v (against/compare-prompt e m (prompt-tokens p) 2 band)] v)))
                  base-cs (against-forward base-e m)
                  cs (against-forward e adapted)
                  worst (fn [cs] (reduce max (map :max-diff cs)))
                  served (pass-rates e tok held 1 eval-key)]
              (println (format "grpo-llama: llama.cpp against the forward, max |Δ log p| over 8 held-out prompts: the base %.3g, the adapter served %.3g (band %s)"
                               (worst base-cs) (worst cs) band))
              (println "grpo-llama: by comparison, the adapter's" (pr-str (mapv #(Double/parseDouble (format "%.3f" (:max-diff %))) cs)))
              (println "grpo-llama: by comparison, the base's   " (pr-str (mapv #(Double/parseDouble (format "%.3f" (:max-diff %))) base-cs)))
              (println (format "grpo-llama: llama.cpp's greedy replies held out %.3f (the array engine's %.3f, the base's %.3f)"
                               (mean served) (mean (:greedy after)) (mean (:greedy before))))
              (is (every? :argmax? cs) (pr-str cs))
              (is (<= (worst cs) band) (pr-str cs))))
          (if-let [keep (System/getProperty "gen.grpo.keep")]
            (println "grpo-llama: the checkpoints kept in" dir "(gen.grpo.keep" keep ")")
            (doseq [f (reverse (file-seq (java.io.File. dir)))] (.delete ^java.io.File f))))))))
