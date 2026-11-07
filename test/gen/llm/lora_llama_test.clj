(ns gen.llm.lora-llama-test
  "LoRA adapters (gen.llm.lora) on Qwen3's forward over gen.array against
   the same adapters loaded by llama.cpp (gen.llm.llama's :adapters): the
   check of the adapter's conventions, A and B, their shapes and
   transposes, the scale alpha / rank, the tensor names, which no test on
   one engine sees (an adapter written and read back by the same wrong
   convention is self-consistent).
   1. gguf-py's adapter (test/fixtures/lora/qwen3-0.6b-lora.gguf, rank 2
      on attn_q and attn_v of four layers, alpha 4: s = 2), read by
      lora/load-gguf and applied by lora/adapt, against llama.cpp's
      reading of the same file: the forward's conventions;
   2. an adapter of lora/init, B drawn so that it changes the model (rank
      4 on every projection of four layers, alpha 12: s = 3), written by
      lora/save-gguf and loaded by llama.cpp, at scales 1 and 0.5: the
      writer's conventions;
   3. planted convention errors, each failing 1 or 2: llama.cpp refuses
      the file (loud), or the log-probabilities differ by more than
      mutation-nats;
   4. LoRA steps on the real model lower its loss, and the trained
      adapter, saved, reads in llama.cpp as in the forward: train in MLX,
      serve in llama.cpp.
   Each comparison is gen.llm.models.qwen3-llama-test's
   (gen.llm.models.against-llama): for each prompt, read whole, as a
   chunk and through 4 greedy steps, the next-token log-probabilities
   within the band over llama.cpp's 64 most likely tokens, with the same
   argmax and top 5.

   Needs the MLX backend (-J-Dgen.backend=mlx), llama.cpp's libraries and
   Qwen3-0.6B's GGUF given as -J-Dgen.llama.test-model (the fixtures'
   shapes are its); skipped with a note otherwise, and for another model.

   The band: qwen3-llama-test's, per platform (0.075 on Metal, 0.65
   elsewhere). Measured on fenrir (2026-11-07, MLX CUDA against
   llama.cpp v0.9.0's CUDA build, Qwen3-0.6B-Q8_0), the adapter's size
   (its largest |Δ log p| from the base, our forward) and the largest
   |Δ log p| against llama.cpp, within its CUDA band:
   - gguf-py's adapter: 18.7 nats; 0.445;
   - ours, at scale 1 and 0.5: 2.39 and 1.08 nats; 0.308 and 0.350;
   - ours on k and v (the planted errors'): 12.4 nats; 0.214;
   - trained (4): 13.1 nats; 0.278 (tape), 0.295 (native gradients).
   The planted errors, against mutation-nats (1.3):
   - refused by llama.cpp: A and B swapped; the names without .weight;
   - silent in llama.cpp's checks: A and B transposed and swapped (B·A
     transposed, on the square k and v) 3.53; k and v's names swapped
     3.00; alpha written as alpha / rank 3.93; the layers off by one
     4.67;
   - in the forward: s = alpha (not alpha / rank) 10.0 (gguf-py's
     adapter) and 16.9 (ours); the product transposed on the square
     projections 10.7 and 3.57.
   Not yet run on Metal, whose band (0.075) is the base model's there."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is use-fixtures]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.lora :as lora]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private fixture "test/fixtures/lora/qwen3-0.6b-lora.gguf")

(def ^:private band
  "qwen3-llama-test's band, per platform."
  (if (.startsWith (System/getProperty "os.name") "Mac") 0.075 0.65))

(def ^:private mutation-nats
  "What a planted convention error must exceed: twice the largest
   platform's band."
  1.3)

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the LoRA-against-llama.cpp tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model's forward needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private fits?
  "Is the test model Qwen3-0.6B's shapes, the fixture's?"
  (delay (let [{:keys [metadata]} (gguf/header model-path)]
           (and (= "qwen3" (metadata "general.architecture")) (= 28 (metadata "qwen3.block_count"))
                (= 1024 (metadata "qwen3.embedding_length"))))))

(def ^:private shared (atom nil))

(defn- setup
  "[llama.cpp's model, the forward's model] of the test model, loaded
   once."
  []
  (or @shared (reset! shared [((llama 'model) model-path) (qwen3/load model-path)])))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [[^java.io.Closeable lm] @shared]
                                      (.close lm)
                                      (reset! shared nil))))))

(defmacro ^:private when-qwen3-0-6b
  [& body]
  `(ts/when-native :llama available?
     (if @fits?
       (do ~@body)
       (println "gen.llm.lora-llama-test: the test model is not Qwen3-0.6B, the fixture's; nothing to compare"))))

(def ^:private prompts
  ["The capital"
   "The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   (str "In probabilistic programming, a generative function is a program whose random choices are named, "
        "so that inference can constrain some of them to observations and propose the others; its density "
        "is the product of each choice's, and")])

(defn- temp-path [name]
  (str (Files/createTempDirectory "lora-llama-test" (make-array FileAttribute 0)) "/" name))

(defn- comparisons
  "The forward of the adapted model m' against llama.cpp's engine over its
   model lm with the adapter file at path at scale: every prompt's
   verdicts (against/compare-prompt's), steps greedy steps each; or
   {:refused message} when llama.cpp refuses the file."
  [lm m' path scale prompts steps]
  (let [a (try ((llama 'adapter) lm path)
               (catch clojure.lang.ExceptionInfo x
                 (if (= :adapter-load-failed (:reason (ex-data x))) x (throw x))))]
    (if (instance? Throwable a)
      {:refused (or (:llama-error (ex-data a)) (ex-message a))}
      (with-open [^java.io.Closeable a a
                  ^java.io.Closeable e ((llama 'engine) {:model lm :adapters {a scale} :max-branches 4 :n-ctx 1024})]
        (vec (for [text prompts
                   :let [tokens ((llama 'tokenize) e text)]
                   v (against/compare-prompt e m' tokens steps band)]
               (assoc v :prompt (subs text 0 (min 20 (count text))))))))))

(defn- worst
  "The largest |Δ log p| of comparisons, or :refused."
  [cs]
  (if (map? cs) :refused (reduce max (map :max-diff cs))))

(defn- agree?
  "Do the comparisons hold within the band, argmax and top 5 too?"
  [cs]
  (and (vector? cs) (every? #(and (<= (:max-diff %) band) (:argmax? %) (:top5? %)) cs)))

(defn- effect
  "max |Δ log p| between the adapted model m' and the base m, over every
   prompt's last position (the adapter's size, in nats)."
  [lm m m' prompts]
  (with-open [^java.io.Closeable e ((llama 'engine) {:model lm :max-branches 2 :n-ctx 512})]
    (reduce max (for [text prompts
                      :let [tokens ((llama 'tokenize) e text)
                            lp #(array/log-softmax (first (qwen3/prefill % [tokens] {:logits :last})))]]
                  (array/item (array/max (array/abs (array/- (lp m) (lp m')))))))))

(defn- drawn-b
  "adapter with each B drawn N(0, sd²) from key: an adapter that changes
   the model."
  [adapter sd key]
  (update adapter :params
          (fn [params]
            (into {} (for [[i ts] params]
                       [i (into {} (for [[t {:keys [a b]}] ts]
                                     [t {:a a :b (array/* sd (rng/normal (rng/fold-in (rng/fold-in key i) (count (name t))) (array/shape b)))}]))])))))

(defn- ours
  "An adapter of lora/init on model m, B drawn N(0, sd²): rank 4, alpha 12
   (s = 3), the targets on layers 0, 9, 18 and 26."
  [m targets sd]
  (drawn-b (lora/init m {:rank 4 :alpha 12 :targets targets :layers [0 9 18 26]} (rng/key 20261107)) sd (rng/key 1)))

;; ---------------------------------------------------------------------------
;; 1 and 2. The adapters, by both engines
;; ---------------------------------------------------------------------------

(deftest gguf-py-s-adapter-reads-as-llama-cpp-reads-it
  (when-qwen3-0-6b
    (let [[lm m] (setup)
          m' (lora/adapt m (lora/load-gguf fixture))
          cs (comparisons lm m' fixture 1.0 prompts 4)]
      (println (format "lora-llama: gguf-py's adapter moves log p by %.3g nats; against llama.cpp |Δ log p| at most %.3g"
                       (effect lm m m' prompts) (worst cs)))
      (is (agree? cs) (pr-str (remove #(and (<= (:max-diff %) band) (:argmax? %) (:top5? %)) cs))))))

(deftest our-adapter-reads-in-llama-cpp-as-in-the-forward
  (when-qwen3-0-6b
    (let [[lm m] (setup)
          ad (ours m (vec (keys lora/targets)) 0.02)
          path (lora/save-gguf ad (temp-path "ours.gguf"))]
      (doseq [scale [1.0 0.5]
              :let [m' (lora/adapt m ad {:scale scale})
                    cs (comparisons lm m' path scale prompts 4)]]
        (println (format "lora-llama: our adapter at scale %s moves log p by %.3g nats; against llama.cpp |Δ log p| at most %.3g"
                         scale (effect lm m m' prompts) (worst cs)))
        (is (agree? cs) (str "scale " scale ": " (pr-str (remove #(and (<= (:max-diff %) band) (:argmax? %) (:top5? %)) cs))))))))

;; ---------------------------------------------------------------------------
;; 3. Planted convention errors
;; ---------------------------------------------------------------------------

(def ^:private write-gguf gguf/write-gguf)

(defn- rewritten
  "with-redefs-fn's map: save-gguf's file with its tensors (a vector of
   {:name :shape :data}, in pairs lora_a, lora_b) and metadata entries
   transformed by f and g."
  ([f] (rewritten f identity))
  ([f g]
   {#'gguf/write-gguf (fn [path spec] (write-gguf path (-> spec (update :tensors (comp f vec)) (update :metadata (comp g vec)))))}))

(defn- transposed
  "The host data of a matrix [r c] (a float[]) transposed, [c r]."
  ^floats [^floats data [r c]]
  (let [r (long r) c (long c) out (float-array (* r c))]
    (dotimes [i r] (dotimes [j c] (aset out (+ (* j r) i) (aget data (+ (* i c) j)))))
    out))

(def ^:private delta lora/delta)

(def ^:private writer-mutations
  "Planted errors in what save-gguf writes."
  {:a-b-swapped (rewritten #(vec (mapcat (fn [[a b]] [(assoc a :data (:data b) :shape (:shape b))
                                                     (assoc b :data (:data a) :shape (:shape a))])
                                         (partition 2 %))))
   :transposed (rewritten #(vec (mapcat (fn [[a b]] [(assoc a :data (transposed (:data b) (:shape b)) :shape (vec (reverse (:shape b))))
                                                    (assoc b :data (transposed (:data a) (:shape a)) :shape (vec (reverse (:shape a))))])
                                        (partition 2 %))))
   :alpha-over-rank (rewritten identity (fn [md] (mapv (fn [[k t v]] [k t (if (= k "adapter.lora.alpha") (/ v 4.0) v)]) md)))
   :k-v-names-swapped (rewritten #(mapv (fn [t] (update t :name (fn [n] (cond (re-find #"attn_k" n) (str/replace n "attn_k" "attn_v")
                                                                               (re-find #"attn_v" n) (str/replace n "attn_v" "attn_k")
                                                                               :else n))))
                                        %))
   :layer-off-by-one (rewritten #(mapv (fn [t] (update t :name (fn [n] (str/replace n #"^blk\.(\d+)\." (fn [[_ i]] (str "blk." (inc (parse-long i)) ".")))))) %))
   :weight-suffix-dropped (rewritten #(mapv (fn [t] (update t :name str/replace ".weight." ".")) %))})

(def ^:private forward-mutations
  "Planted errors in the forward's adapter term."
  {:scale-alpha {#'lora/delta (fn [x w] (array/* (first (array/shape (get-in w [:lora :a]))) (delta x w)))}
   :product-transposed {#'lora/delta (fn [x w]
                                       (let [{:keys [a b s]} (:lora w)]
                                         (if (= (first (array/shape b)) (second (array/shape a)))
                                           (array/* s (array/matmul (array/matmul x b) a))   ; x·(B·A), not x·(B·A)ᵀ
                                           (delta x w))))}})

(deftest planted-convention-errors-fail
  ;; on the square k and v projections (1024 × 1024 in Qwen3-0.6B), where
  ;; llama.cpp's shape checks pass a transposed or misnamed adapter; one
  ;; prompt, read whole, as a chunk and one step
  (when-qwen3-0-6b
    (let [[lm m] (setup)
          ad (ours m [:k :v] 0.1)
          prompt [(second prompts)]
          clean (comparisons lm (lora/adapt m ad) (lora/save-gguf ad (temp-path "clean.gguf")) 1.0 prompt 1)
          _ (println (format "lora-llama: the k and v adapter moves log p by %.3g nats; against llama.cpp at most %.3g"
                             (effect lm m (lora/adapt m ad) prompt) (worst clean)))
          _ (is (agree? clean) "the adapter itself agrees")
          margins (into {}
                        (concat
                         (for [[k redefs] (sort writer-mutations)
                               :let [path (with-redefs-fn redefs #(lora/save-gguf ad (temp-path (str (name k) ".gguf"))))]]
                           [k (worst (comparisons lm (lora/adapt m ad) path 1.0 prompt 1))])
                         (for [[k redefs] (sort forward-mutations)
                               [what a path] [[:fixture (lora/load-gguf fixture) fixture]
                                              [:ours ad (lora/save-gguf ad (temp-path "ours-kv.gguf"))]]]
                           [[k what] (with-redefs-fn redefs #(worst (comparisons lm (lora/adapt m a) path 1.0 prompt 1)))])))]
      (println "lora-llama: planted errors, |Δ log p| or :refused:" (pr-str margins))
      (doseq [[k v] margins]
        (is (or (= :refused v) (< mutation-nats v)) (str k ": " v))))))

;; ---------------------------------------------------------------------------
;; 4. Training, then serving
;; ---------------------------------------------------------------------------

(def ^:private text
  (str "Gen.clj is a probabilistic programming system on the JVM. A generative function names its random "
       "choices, so that inference can constrain some of them to observations and propose the others."))

(deftest lora-steps-lower-the-loss-and-llama-cpp-serves-the-result
  ;; rank 8 on every projection of the last 8 layers (mlx-lm's custom), AdamW
  ;; at 1e-3 clipped to norm 1, 5 steps on one text, checkpointed: each step
  ;; lowers the loss; the trained adapter, saved, reads in llama.cpp as in
  ;; the forward. Measured on fenrir (2026-11-07, 37 tokens, tape and
  ;; native gradients alike): the summed nll 188.2 → 156.6 → 113.6 → 69.4
  ;; → 39.3 → 21.4
  (when-qwen3-0-6b
    (let [[lm m] (setup)
          tokens (with-open [^java.io.Closeable e ((llama 'engine) {:model lm :max-branches 2 :n-ctx 512})]
                   ((llama 'tokenize) e text))
          ad (lora/init m {:rank 8 :alpha 16 :targets (vec (keys lora/targets)) :layers (range 20 28)} (rng/key 3))
          opt (learn/clip-norm (learn/adamw {:lr 1e-3}) {:max-norm 1.0})
          loss (fn [params] (qwen3/nll (lora/adapt m (assoc ad :params params)) [tokens] {:checkpoint? true :chunk 32768}))
          [losses params] (loop [i 0 params (:params ad) state (learn/init opt (:params ad)) out []]
                            (let [[l g] ((array/value-and-grad loss) params)
                                  out (conj out (array/item l))]
                              (if (= i 5)
                                [out params]
                                (let [[params state] (learn/step opt state params (array/tree-map array/negate g))]
                                  (recur (inc i) params state out)))))
          trained (assoc ad :params params)
          path (lora/save-gguf trained (temp-path "trained.gguf"))
          m' (lora/adapt m trained)
          cs (comparisons lm m' path 1.0 prompts 2)]
      (println (format "lora-llama: %d tokens, the loss %s; the trained adapter moves log p by %.3g nats; against llama.cpp at most %.3g"
                       (count tokens) (pr-str (mapv #(Math/round (* 1000.0 (double %))) losses)) (effect lm m m' prompts) (worst cs)))
      (is (apply > losses) (pr-str losses))
      (is (agree? cs) (pr-str (remove #(and (<= (:max-diff %) band) (:argmax? %) (:top5? %)) cs))))))
