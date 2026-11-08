(ns gen.llm.array-engine-llama-test
  "gen.array's engine (gen.llm.array-engine) over a real Qwen3 GGUF, its
   tokenizer llama.cpp's vocabulary loaded alone (gen.llm.llama/
   vocabulary), against the llama.cpp engine on the same file:
   - the vocabulary alone is the llama.cpp engine's tokenizer;
   - the Branching contract (gen.llm.branching-contract) within this
     engine's band, and the LM contract (gen.llm.lm-contract);
   - its rows and scores against llama.cpp's within the band of
     gen.llm.models.qwen3-llama-test (0.075 nats on Metal), the argmax
     the same;
   - the token generative function's and token SMC's checks
     (gen.llm.token-contract), as gen.llm.llama-token-test and
     gen.llm.llama-smc-test run them on llama.cpp;
   - the shared memory budget with the engine open (gen.array.memory's
     reservations and MLX's budget): its arrays are MLX's active memory,
     a read past the budget (less what a native engine reserved) is
     loud and changes nothing, a lane group stepped past it is consumed
     and its branch freed, loudly, and the engine goes on.

   Needs the MLX backend (-J-Dgen.backend=mlx: a 0.6B model's weights are
   too large for the JVM backend's doubles), llama.cpp's libraries and a
   Qwen3 GGUF as -J-Dgen.llama.test-model (or GEN_LLAMA_TEST_MODEL), e.g.
   Qwen3-0.6B-Q8_0. Skipped with a note otherwise, and for a model of
   another architecture.

   The band, 1e-4 nats: twice the largest |Δ log p| (over the reference's
   64 most likely tokens) between two computations of one distribution on
   this engine, measured by the contract's checks over five prompts and
   continuations (the band's docstring); float32's round-off with the
   reads' shapes. llama.cpp's own spread on Metal is 5.2e-2 (its band
   0.1), since its kernels round activations to half precision. Against
   llama.cpp the same five gave at most 1.4e-2 (reads 4.0e-3 to 1.1e-2,
   advanced together up to 1.4e-2, scores up to 8.9e-3), every argmax
   the same. The engines compute in float32, the dtype the bands were
   measured in, given as :dtype: on MLX's CUDA a quantized file's default
   is float16."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array.gguf :as gguf]
            [gen.array.memory :as reservations]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.against-llama :as against]
            [gen.llm.slots :as slots]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  "This engine's band on Metal: twice the largest |Δ log p| the contract's
   checks measured (Mac mini M4, 2026-11-07, Qwen3-0.6B-Q8_0, float32),
   5.1e-5 over five prompts (extend's branch against a fresh read; the
   others 7.6e-6 to 2.3e-5), rounded up."
  1e-4)

(def ^:private llama-band
  "Against llama.cpp: gen.llm.models.qwen3-llama-test's, per platform
   (against-llama/band: 0.075 on Metal, 2.4 against llama.cpp's CUDA and
   x86 builds, which quantize the activations to 8 bits; on fenrir's CUDA
   this test measured 0.19 to 0.32, 2026-11-08)."
  (delay (against/band model-path)))

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping gen.array's engine on a real model:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model's forward needs MLX (-J-Dgen.backend=mlx)")
        (not= "qwen3" (get-in (gguf/header model-path) [:metadata "general.architecture"]))
        (skip "not a Qwen3 model")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private shared (atom nil))

(defn- setup
  "{:vocab :engine :llama}, made once: the vocabulary alone, gen.array's
   engine (255 branches, for SMC's 32 particles; blocks of 32 positions,
   to keep its groups small) and llama.cpp's."
  []
  (or @shared
      (reset! shared {:vocab ((llama 'vocabulary) model-path)
                      :engine (ae/engine {:model model-path :dtype :float32 :max-branches 255 :block 32})
                      :llama ((llama 'engine) {:model model-path :max-branches 12 :n-ctx 2048})})))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [s @shared]
                                      (doseq [k [:engine :llama :vocab]] (.close ^java.io.Closeable (s k)))
                                      (reset! shared nil))))))

(def ^:private pairs
  [["The capital of Sweden is a city called" " Stockholm, which is known for its many islands and"]
   ["def fibonacci(n):" "\n    if n < 2:\n        return n\n    return fibonacci(n - 1)"]
   ["1, 2, 3, 5, 8, 13," " 21, 34, 55, 89, 144, 233"]])

(defn- tokens [text opts] ((llama 'tokenize) (:vocab (setup)) text opts))

(defn- opts [[p c]]
  {:prompt (tokens p {}) :continuation (tokens c {:add-special? false}) :k 4 :band band})

(deftest the-vocabulary-alone-is-the-engine-s
  (ts/when-native :llama available?
    (let [{v :vocab le :llama} (setup)
          tv ((llama 'tokenizer) v)
          tl ((llama 'tokenizer) le)
          text "Stockholm är Sveriges huvudstad: 1 000 000 människor, ℝ³ → ∞ 🙂"]
      (is (= (dissoc tv :token-bytes :encode :decode) (dissoc tl :token-bytes :encode :decode)))
      (is (every? #(java.util.Arrays/equals ^bytes ((:token-bytes tv) %) ^bytes ((:token-bytes tl) %)) (range (:n-vocab tv))))
      (is (= ((llama 'tokenize) le text) ((llama 'tokenize) v text)))
      (is (= text ((llama 'detokenize) v ((llama 'tokenize) v text {:add-special? false}))))
      (is (= :invalid-token (bc/reason #((llama 'token-bytes) v (:n-vocab tv)))))
      (let [v ((llama 'vocabulary) model-path)]
        (.close ^java.io.Closeable v)
        (is (= :closed (bc/reason #((llama 'tokenize) v "a")))))
      (is (= :no-model (bc/reason #((llama 'vocabulary) "no/such.gguf")))))))

(deftest the-branching-contract-holds-within-the-band
  (ts/when-native :llama available?
    (let [e (:engine (setup))]
      (doseq [pair pairs
              [k check] bc/checks]
        (testing [(first pair) k]
          (is (= [] (bc/run-check check e (opts pair))))))
      (is (zero? (:live (br/info e))) "every check disposes what it makes"))))

(deftest rows-and-scores-are-llama-cpp-s
  ;; a prompt read, its four likeliest continuations advanced together,
  ;; and the continuation scored, by both engines: within llama-band over
  ;; llama.cpp's 64 likeliest tokens, the argmax the same
  (ts/when-native :llama available?
    (let [{engine :engine le :llama} (setup)]
      (doseq [pair pairs
              :let [{:keys [prompt continuation]} (opts pair)
                    a (br/open engine prompt)
                    b (br/open le prompt)
                    top (bc/top-tokens (br/logprobs b) 4)
                    as (br/advance (br/fork a 4) top)
                    bs (br/advance (br/fork b 4) top)]]
        (testing (first pair)
          (is (<= (bc/max-diff (br/logprobs b) (br/logprobs a) 64) @llama-band) "read")
          (is (= (first (bc/top-tokens (br/logprobs b) 1)) (first (bc/top-tokens (br/logprobs a) 1))) "the argmax")
          (doseq [[x y] (map vector as bs)]
            (is (<= (bc/max-diff (br/logprobs y) (br/logprobs x) 64) @llama-band) "advanced together"))
          (is (<= (reduce max (map #(Math/abs (- (double %1) (double %2))) (br/score a continuation) (br/score b continuation)))
                  @llama-band)
              "scored"))
        (br/dispose! (cons a as))
        (br/dispose! (cons b bs))))))

(deftest the-continuous-mode-holds-the-contract-alone-and-at-once
  ;; the continuous mode (:continuous?, genclj-b3ow), its forwards of 8
  ;; tokens so that prompts and continuations take chunks: the Branching
  ;; contract within the band, by one caller and by four at once (whose
  ;; reads mix decode rows and chunks of other callers'); the largest
  ;; |Δ log p| printed
  (ts/when-native :llama available?
    (with-open [e (ae/engine {:model model-path :dtype :float32 :max-branches 64 :paged? true :continuous? true :n-batch 8})]
      (let [worst (atom 0.0)
            seen (fn [_ d] (swap! worst max (double d)))
            at-once [:normalized :fork-matches-fresh :batch-matches-single :score-matches-steps :threads]]
        (binding [bc/*observe* seen]
          (doseq [pair pairs
                  [k check] bc/checks]
            (testing [(first pair) k]
              (is (= [] (bc/run-check check e (opts pair))))))
          (let [results (mapv deref (mapv (fn [pair]
                                            (future (binding [bc/*observe* seen]
                                                      (into {} (map (fn [c] [c (bc/run-check (bc/checks c) e (opts pair))])) at-once))))
                                          (take 4 (cycle pairs))))]
            (doseq [[i r] (map-indexed vector results), c at-once]
              (testing ["at once" i c]
                (is (= [] (r c)))))))
        (println "the continuous mode's largest |Δ log p|:" @worst "(band" band ")" (ae/stats e) (slots/schedule e))
        (is (zero? (:live (br/info e))))))))

(deftest the-lm-contract-holds-within-the-band
  (ts/when-native :llama available?
    (let [{:keys [engine vocab]} (setup)
          tok ((llama 'tokenizer) vocab)
          s {:engine engine
             :lm #(lm/lm engine tok {:max-branches 12})
             :fresh (fn [] (let [e (ae/engine {:model model-path :dtype :float32 :max-branches 16 :block 32})]
                             [e (lm/lm e tok {:max-branches 12})]))}]
      (doseq [[k check] lc/checks]
        (testing k
          (is (= [] (lc/run-check check s (opts (first pairs)))))))
      (is (zero? (:live (br/info engine))) "every check closes what it makes"))))

(deftest the-token-and-smc-checks-hold
  ;; each check on an LM of its own, as llama.cpp's suites have them
  (ts/when-native :llama available?
    (let [{:keys [engine vocab]} (setup)
          tok ((llama 'tokenizer) vocab)
          o {:prompt (tokens "The capital of Sweden is" {})
             :forced (tokens " Stockholm" {:add-special? false})
             :sea-prompt (tokens "Write one short sentence about the sea:" {})
             :date-prompt (tokens "Today's date, written as YYYY-MM-DD, is" {})
             :particles 32}]
      (doseq [[k check] tc/checks]
        (with-open [m (lm/lm engine tok)]
          (testing k
            (is (= [] (tc/run-check check {:engine engine :lm m :completion (token/completion m)} o))))))
      (is (zero? (:live (br/info engine)))))))

;; ---------------------------------------------------------------------------
;; The shared memory budget
;; ---------------------------------------------------------------------------

(def ^:private mb (* 1024 1024))

(defn- with-room
  "(f) with MLX's budget what MLX and the reservations hold now plus room
   bytes (no headroom; room may be negative), restored after."
  [room f]
  (let [budget (requiring-resolve 'gen.backend.mlx/budget)
        set-budget! (requiring-resolve 'gen.backend.mlx/set-budget!)
        collect! (requiring-resolve 'gen.backend.mlx/collect!)
        _ (collect!)
        old (set-budget! {:bytes (+ (long (:active (budget))) (long (reservations/reserved)) (long room)) :headroom 0})]
    (try (f)
         (finally (set-budget! old)
                  (collect!)))))

(deftest the-engine-s-arrays-are-inside-the-shared-budget
  ;; Qwen3-0.6B's KV cache holds 224 KiB a position in float32: a read of
  ;; 200 tokens (capacity 224) makes 49 MB of it
  (ts/when-native :llama available?
    (with-open [e (ae/engine {:model model-path :dtype :float32 :max-branches 8 :block 32})]
      (let [ctx (vec (take 200 (cycle (tokens " the tide is full, the moon lies fair" {:add-special? false}))))
            b (br/open e (tokens "The sea is calm tonight." {}))
            [s] (br/advance [(br/fork b)] [13])
            live (:live (br/info e))]
        (testing "a read past the budget is loud and makes no branch"
          (with-room (* 16 mb)
            (fn []
              (let [x (ts/thrown-data (br/open e ctx))]
                (is (= :mlx-budget-exceeded (:reason x)) (pr-str x))
                (is (= live (:live (br/info e))))))))
        (testing "a native engine's reservation shrinks the room"
          (with-room (* 128 mb)
            (fn []
              (reservations/reserve! ::native (* 112 mb) "a test's native engine")
              (try
                (is (= :mlx-budget-exceeded (:reason (ts/thrown-data (br/open e ctx)))))
                (finally (reservations/release! ::native)))
              (let [c (br/open e ctx)]
                (is (= 200 (br/position c)) "released, the read fits")
                (br/dispose! c)))))
        (testing "a lane group stepped past the budget is consumed: its branch freed, loudly"
          ;; a budget below what MLX holds refuses every evaluation
          (with-room (- (* 64 mb))
            (fn []
              (let [x (ts/thrown-data (br/advance [s] [13]))]
                (is (= :mlx-budget-exceeded (:reason x)))
                (is (= [(slots/slot-of e s)] [nil]) "its slot is freed")
                (is (not (br/live? s)))))))
        (testing "the engine goes on"
          (let [[s2] (br/advance [(br/fork b)] [13])
                [s3] (br/advance [s2] [279])]
            (is (= (+ 2 (br/position b)) (br/position s3)))
            (br/dispose! [b s3])))))))
