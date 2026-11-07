(ns gen.llm.sft-llama-test
  "Supervised fine-tuning (gen.llm.sft) of a real model: Qwen3-0.6B
   taught to answer in a JSON format it does not use, {\"answer\": …},
   with no instruction to, by a LoRA adapter (rank 8, alpha 16, on q, k,
   v, o, gate, up and down of the last 8 blocks, mlx-lm's custom):
   - 64 chat examples (sums, products, the larger of two numbers,
     capitals), 48 trained on and 16 held out;
   - AdamW at 2e-4 (no weight decay) under a warmup of 2 steps and
     cosine decay over 24, its gradients clipped to norm 1; 24 steps of
     2 accumulated batches of 8 (8 epochs), checkpointed;
   - the held-out NLL per trained token (the reply and its end token)
     falls, as does the training loss;
   - greedy replies to the held-out questions, through gen.llm.array-
     engine with the adapted model, follow the format (the reply is the
     JSON object alone and ends with <|im_end|>) more often after than
     before, counted;
   - the adapter, saved in the last checkpoint, loads in llama.cpp
     (gen.llm.llama's :adapters) and reads as in the forward, within the
     platform's band (qwen3-llama-test's); llama.cpp's greedy replies
     follow the format as the array engine's do.
   Prints the numbers. Needs the MLX backend, llama.cpp's libraries and
   Qwen3-0.6B's GGUF as -J-Dgen.llama.test-model (as gen.llm.lora-llama-
   test); skipped with a note otherwise.

   Measured on fenrir (2026-11-07, MLX CUDA, native gradients, Q8_0; 24
   steps in 10 s, the whole test 23 s):
   - the training loss 3.63 → 0.045 nats a trained token;
   - the held-out NLL at steps 0, 4, …, 24: 3.612, 2.689, 0.719, 0.182,
     0.113, 0.096, 0.093;
   - in the format, before 0/16, after 15/16 (12 correct: the format is
     learnt, some facts lost, Portugal's capital \"Porto\");
   - llama.cpp: 0.533 nats at most against the forward, every argmax
     the same; its replies 14/16 in the format (11 correct).
   Trained without the end token (example's :end-token? false), the same
   run's replies stop at <|im_end|> in only 12 of 16; the other 4 run on
   to the 48 tokens allowed."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.array-engine :as ae]
            [gen.llm.chat :as chat]
            [gen.llm.lm :as lm]
            [gen.llm.lora :as lora]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.sample :as sample]
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
    (let [skip (fn [why] (println "Skipping the SFT run on a real model:" why) (ts/native-unavailable! :llama why) false)]
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

;; ---------------------------------------------------------------------------
;; The examples
;; ---------------------------------------------------------------------------

(def ^:private capitals
  [["France" "Paris"] ["Germany" "Berlin"] ["Italy" "Rome"] ["Spain" "Madrid"] ["Japan" "Tokyo"] ["Egypt" "Cairo"]
   ["Canada" "Ottawa"] ["Sweden" "Stockholm"] ["Norway" "Oslo"] ["Russia" "Moscow"] ["China" "Beijing"]
   ["Greece" "Athens"] ["Portugal" "Lisbon"] ["Poland" "Warsaw"] ["Kenya" "Nairobi"] ["Peru" "Lima"]])

(defn- qa
  "Question i and its answer in the format: [question, answer]."
  [i]
  (let [a (+ 10 (mod (+ 5 (* 17 i)) 80))
        b (+ 3 (mod (+ 11 (* 29 i)) 60))]
    (case (int (mod i 4))
      0 [(str "What is " a " plus " b "?") (+ a b)]
      1 [(str "What is " (mod a 13) " times " (mod b 11) "?") (* (mod a 13) (mod b 11))]
      2 (let [[country capital] (capitals (quot i 4))] [(str "What is the capital of " country "?") (str "\"" capital "\"")])
      3 [(str "Which is larger, " a " or " b "?") (max a b)])))

(defn- reply [answer] (str "{\"answer\": " answer "}"))

(def ^:private format-re #"\{\"answer\": (\"[^\"]*\"|-?\d+)\}")

(defn- follows?
  "Does a completion follow the format: the JSON object alone, ended by
   an end token?"
  [{:keys [stop text]}]
  (and (= :end stop) (boolean (re-matches format-re (str/trim text)))))

(defn- correct? [{:keys [text]} answer] (= (str/trim text) (reply answer)))

(defn- prompt-tokens [tok question]
  ((:encode tok) (chat/render [{:role :user :content question}]) {:add-special? false :parse-special? true}))

(defn- replies
  "Greedy replies to the questions by an LM over engine e: [{:stop :text}]."
  [e tok questions]
  (with-open [m (lm/lm e tok {:max-branches 32 :max-cells 4096})]
    (mapv (fn [{:keys [tokens stop]}] {:stop stop :text ((:decode tok) tokens)})
          (sample/complete m (mapv #(prompt-tokens tok %) questions)
                           {:sampling {:temp 1.0 :top-k 1 :top-p 1.0} :max-tokens 48 :key (rng/key 1)}))))

(defn- count-of [pred rs answers] (count (filter true? (map pred rs answers))))

;; ---------------------------------------------------------------------------
;; The run
;; ---------------------------------------------------------------------------

(deftest ^:slow sft-teaches-a-format-and-llama-cpp-serves-it
  (ts/when-native :llama available?
    (if-not @qwen3-0-6b?
      (println "gen.llm.sft-llama-test: the test model is not Qwen3-0.6B; nothing run")
      (with-open [^java.io.Closeable vocab ((llama 'vocabulary) model-path)]
        (let [tok ((llama 'tokenizer) vocab)
              m (qwen3/load model-path)
              qas (mapv qa (range 64))
              examples (mapv (fn [[q a]] (sft/example tok {:prompt q :completion (reply a)})) qas)
              [train held-out] [(subvec examples 0 48) (subvec examples 48)]
              questions (mapv first (subvec qas 48))
              answers (mapv second (subvec qas 48))
              n (long (:layers (:config m)))
              ad (lora/init m {:rank 8 :alpha 16 :targets (vec (keys (lora/targets "qwen3"))) :layers (range (- n 8) n)} (rng/key 20261107))
              opt (learn/warmup-cosine (learn/clip-norm (learn/adamw {:lr 2e-4 :weight-decay 0.0}) {:max-norm 1.0})
                                       {:warmup 2 :steps 24})
              dir (str (Files/createTempDirectory "sft-llama-test" (make-array FileAttribute 0)))
              t0 (System/nanoTime)
              {:keys [trainer history evaluations]}
              (sft/train m (sft/trainer ad opt (rng/key 7)) train
                         {:steps 24 :batch-size 8 :accumulate 2 :held-out held-out :eval-every 4 :save-dir dir :save-every 8
                          :callback (fn [{:keys [step loss grad-norm tokens]}]
                                      (println (format "sft-llama: step %2d loss %.4f grad-norm %.3f tokens %d" step loss grad-norm tokens)))})
              seconds (/ (- (System/nanoTime) t0) 1e9)
              adapted (lora/adapt m (:adapter trainer))
              before (with-open [e (ae/engine {:model m :max-branches 32 :n-ctx 512})] (replies e tok questions))
              after (with-open [e (ae/engine {:model adapted :max-branches 32 :n-ctx 512})] (replies e tok questions))
              checkpoint (sft/latest-checkpoint dir)]
          (println (format "sft-llama: %d steps in %.0f s; held-out NLL by step %s" (count history) seconds
                           (pr-str (mapv (fn [{:keys [step nll]}] [step (Double/parseDouble (format "%.4f" nll))]) evaluations))))
          (println (format "sft-llama: the format, before %d/16 (correct %d), after %d/16 (correct %d)"
                           (count-of (fn [r _] (follows? r)) before answers) (count-of correct? before answers)
                           (count-of (fn [r _] (follows? r)) after answers) (count-of correct? after answers)))
          (doseq [[q b a] (map vector (take 4 questions) before after)]
            (println "sft-llama:" (pr-str q) "→ before" (pr-str (:text b)) "after" (pr-str (:text a))))
          (doseq [[q a r] (map vector questions answers after)
                  :when (not (and (follows? r) (correct? r a)))]
            (println "sft-llama: after, missed:" (pr-str q) "→" (pr-str r) "want" (pr-str (reply a))))
          (is (= (str dir "/step-00000024") checkpoint))
          (is (< (:nll (last evaluations)) (* 0.5 (:nll (first evaluations)))) (pr-str evaluations))
          (is (< (:loss (last history)) (* 0.5 (:loss (first history)))) (pr-str (map :loss history)))
          (is (<= (count-of (fn [r _] (follows? r)) before answers) 2) "the base model does not use the format")
          (is (>= (count-of (fn [r _] (follows? r)) after answers) 14) "the adapted model does")
          (with-open [^java.io.Closeable lm-model ((llama 'model) model-path)
                      ^java.io.Closeable a ((llama 'adapter) lm-model (str checkpoint "/adapter.gguf"))
                      ^java.io.Closeable e ((llama 'engine) {:model lm-model :adapters {a 1.0} :max-branches 32 :n-ctx 4096})]
            (let [cs (vec (for [q (take 4 questions)
                                v (against/compare-prompt e adapted (prompt-tokens tok q) 2 band)]
                            v))
                  served (replies e tok questions)
                  worst (reduce max (map :max-diff cs))]
              (println (format "sft-llama: llama.cpp serving the adapter: max |Δ log p| %.3g against the forward (band %s); the format %d/16 (correct %d)"
                               worst band (count-of (fn [r _] (follows? r)) served answers) (count-of correct? served answers)))
              (doseq [[q r] (map vector questions served) :when (not (follows? r))]
                (println "sft-llama: llama.cpp, off the format:" (pr-str q) "→" (pr-str r)))
              (is (every? #(and (<= (:max-diff %) band) (:argmax? %)) cs) (pr-str cs))
              ;; greedy replies through llama.cpp's own noise (0.53 nats
              ;; measured against the forward on fenrir's CUDA) may part
              ;; from the forward's: a looser count
              (is (>= (count-of (fn [r _] (follows? r)) served answers) 12) "llama.cpp's replies follow the format")))
          (doseq [f (reverse (file-seq (java.io.File. dir)))] (.delete ^java.io.File f)))))))
