(ns gen.llm.lora-qwen35-llama-test
  "LoRA adapters (gen.llm.lora) on Qwen3.5's forward over gen.array
   (gen.llm.models.qwen35) against the same adapters loaded by llama.cpp
   (gen.llm.llama's :adapters), as gen.llm.lora-llama-test checks
   Qwen3's: the check of the adapter's conventions on qwen35's tensors
   (attn_qkv, attn_gate, ssm_beta, ssm_alpha and ssm_out of the
   GatedDeltaNet layers; attn_q, which holds each head's query and output
   gate, attn_k, attn_v and attn_output of the gated attention layers;
   the feed-forward's), which no test on one engine sees.
   1. an adapter of lora/init on every target of six layers (three of
      each kind), B drawn so that it changes the model, written by
      lora/save-gguf and loaded by llama.cpp, at scales 1 and 0.5;
   2. planted convention errors, each failing 1: llama.cpp refuses the
      file (loud), or the log-probabilities differ by more than
      mutation-nats;
   3. LoRA steps on the real model lower its loss, and the trained
      adapter, saved, reads in llama.cpp as in the forward;
   4. (opt-in, -J-Dgen.qwen35.vocab-models=<dir>, as
      gen.llm.models.qwen35-llama-test's) the tiny random Qwen3.6 MoE with
      the 0.8B's vocabulary: an adapter on every qwen35moe target, the
      shared expert's (ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp) and
      the routed experts' stacks (ffn_gate_exps, ffn_up_exps,
      ffn_down_exps, llama.cpp's build_lora_mm_id) among them, reads in
      llama.cpp as in the forward; their gate and up names swapped fail.
   Each comparison is gen.llm.models.against-llama's: for each prompt,
   read whole, as a chunk and through greedy steps, the next-token
   log-probabilities within the band over llama.cpp's 64 most likely
   tokens, with the same argmax and top 5.

   Needs the MLX backend (-J-Dgen.backend=mlx), llama.cpp's libraries and
   a Qwen3.5 GGUF given as -J-Dgen.llama.test-model (Qwen3.5-0.8B-Q8_0);
   skipped with a note otherwise, and for another architecture.

   The band: qwen35-llama-test's, per platform (0.035 on Metal, 0.65
   elsewhere); the opt-in models' 0.05. Measured: the namespace's comment
   blocks at each deftest."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is use-fixtures]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.lora :as lora]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(def ^:private band
  "qwen35-llama-test's band, per platform."
  (if (.startsWith (System/getProperty "os.name") "Mac") 0.035 0.65))

(def ^:private mutation-nats
  "What a planted convention error must exceed: lora-llama-test's, twice
   the largest platform's band."
  1.3)

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the Qwen3.5 LoRA-against-llama.cpp tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        (ts/jvm?) (skip "the JVM backend: a real model's forward needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private qwen35?
  (delay (= "qwen35" (against/architecture model-path))))

(def ^:private shared (atom nil))

(defn- setup
  "[llama.cpp's model, the forward's model] of the test model, loaded
   once."
  []
  (or @shared (reset! shared [((llama 'model) model-path) (qwen35/load model-path)])))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [[^java.io.Closeable lm] @shared]
                                      (.close lm)
                                      (reset! shared nil))))))

(defmacro ^:private when-qwen35
  [& body]
  `(ts/when-native :llama available?
     (if @qwen35?
       (do ~@body)
       (println "gen.llm.lora-qwen35-llama-test: the test model is not a dense Qwen3.5; nothing to compare"))))

(def ^:private prompts
  ["The capital"
   "The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   (str "In probabilistic programming, a generative function is a program whose random choices are named, "
        "so that inference can constrain some of them to observations and propose the others; its density "
        "is the product of each choice's, and")])

(defn- temp-path [name]
  (str (Files/createTempDirectory "lora-qwen35-llama-test" (make-array FileAttribute 0)) "/" name))

(defn- comparisons
  "The forward of the adapted model m' against llama.cpp's engine over its
   model lm with the adapter file at path at scale, the comparisons within
   bnd: every prompt's verdicts (against/compare-prompt's), steps greedy
   steps each; or {:refused message} when llama.cpp refuses the file."
  ([lm m' path scale prompts steps] (comparisons lm m' path scale prompts steps band))
  ([lm m' path scale prompts steps bnd]
   (let [a (try ((llama 'adapter) lm path)
                (catch clojure.lang.ExceptionInfo x
                  (if (= :adapter-load-failed (:reason (ex-data x))) x (throw x))))]
     (if (instance? Throwable a)
       {:refused (or (:llama-error (ex-data a)) (ex-message a))}
       (with-open [^java.io.Closeable a a
                   ^java.io.Closeable e ((llama 'engine) {:model lm :adapters {a scale} :max-branches 4 :n-ctx 1024})]
         (vec (for [text prompts
                    :let [tokens ((llama 'tokenize) e text)]
                    v (against/compare-prompt e m' tokens steps bnd)]
                (assoc v :prompt (subs text 0 (min 20 (count text)))))))))))

(defn- worst
  "The largest |Δ log p| of comparisons, or :refused."
  [cs]
  (if (map? cs) :refused (reduce max (map :max-diff cs))))

(defn- agree?
  "Do the comparisons hold within bnd, argmax and top 5 too?"
  ([cs] (agree? cs band))
  ([cs bnd] (and (vector? cs) (every? #(and (<= (:max-diff %) bnd) (:argmax? %) (:top5? %)) cs))))

(defn- disagreements [cs bnd]
  (if (map? cs) cs (remove #(and (<= (:max-diff %) bnd) (:argmax? %) (:top5? %)) cs)))

(defn- effect
  "max |Δ log p| between the adapted model m' and the base m, over every
   prompt's last position (the adapter's size, in nats)."
  [lm m m' prompts]
  (with-open [^java.io.Closeable e ((llama 'engine) {:model lm :max-branches 2 :n-ctx 512})]
    (reduce max (for [text prompts
                      :let [tokens ((llama 'tokenize) e text)
                            lp #(array/log-softmax (first (qwen35/prefill % [tokens] {:logits :last})))]]
                  (array/item (array/max (array/abs (array/- (lp m) (lp m')))))))))

(defn- drawn-b
  "adapter with each B drawn N(0, sd²) from key: an adapter that changes
   the model."
  [adapter sd key]
  (update adapter :params
          (fn [params]
            (into {} (for [[i ts] params]
                       [i (into {} (for [[t {:keys [a b]}] ts]
                                     [t {:a a :b (array/* sd (rng/normal (rng/fold-in (rng/fold-in key i) (Math/floorMod (long (hash t)) 65536))
                                                                        (array/shape b)))}]))])))))

(defn- layers-of
  "Six layers of model m: the first two of each kind and the last of
   each."
  [m]
  (let [kinds (get-in m [:config :kinds])
        of (fn [kind] (keep-indexed (fn [i k] (when (= kind k) i)) kinds))]
    (vec (sort (concat (take 2 (of :delta)) [(last (of :delta))] (take 2 (of :attention)) [(last (of :attention))])))))

(defn- ours
  "An adapter of lora/init on model m, B drawn N(0, sd²): rank 4, alpha 12
   (s = 3), the targets on (layers-of m)."
  [m targets sd]
  (drawn-b (lora/init m {:rank 4 :alpha 12 :targets targets :layers (layers-of m)} (rng/key 20261107)) sd (rng/key 1)))

;; ---------------------------------------------------------------------------
;; 1. Our adapter, by both engines
;; ---------------------------------------------------------------------------

(deftest our-adapter-reads-in-llama-cpp-as-in-the-forward
  ;; every qwen35 target: measured on Metal (Mac mini M4, 2026-11-07,
  ;; Qwen3.5-0.8B-Q8_0, llama.cpp v0.9.0): the printed line
  (when-qwen35
    (let [[lm m] (setup)
          ad (ours m (vec (keys (lora/targets "qwen35"))) 0.01)
          path (lora/save-gguf ad (temp-path "ours.gguf"))]
      (doseq [scale [1.0 0.5]
              :let [m' (lora/adapt m ad {:scale scale})
                    cs (comparisons lm m' path scale prompts 4)]]
        (println (format "lora-qwen35-llama: our adapter at scale %s moves log p by %.3g nats; against llama.cpp |Δ log p| at most %.3g"
                         scale (effect lm m m' prompts) (worst cs)))
        (is (agree? cs) (str "scale " scale ": " (pr-str (disagreements cs band))))))))

;; ---------------------------------------------------------------------------
;; 2. Planted convention errors
;; ---------------------------------------------------------------------------

(def ^:private write-gguf gguf/write-gguf)

(defn- rewritten
  "with-redefs-fn's map: save-gguf's file with its tensors (a vector of
   {:name :shape :data}, in pairs lora_a, lora_b) and metadata entries
   transformed by f and g."
  ([f] (rewritten f identity))
  ([f g]
   {#'gguf/write-gguf (fn [path spec] (write-gguf path (-> spec (update :tensors (comp f vec)) (update :metadata (comp g vec)))))}))

(defn- names-swapped
  "rewritten's map: the tensor names x and y (llama.cpp's) exchanged."
  [x y]
  (rewritten #(mapv (fn [t] (update t :name (fn [n] (cond (str/includes? n (str "." x ".")) (str/replace n (str "." x ".") (str "." y "."))
                                                          (str/includes? n (str "." y ".")) (str/replace n (str "." y ".") (str "." x "."))
                                                          :else n))))
                    %)))

(def ^:private writer-mutations
  "Planted errors in what save-gguf writes."
  {:a-b-swapped (rewritten #(vec (mapcat (fn [[a b]] [(assoc a :data (:data b) :shape (:shape b))
                                                     (assoc b :data (:data a) :shape (:shape a))])
                                         (partition 2 %))))
   :alpha-over-rank (rewritten identity (fn [md] (mapv (fn [[k t v]] [k t (if (= k "adapter.lora.alpha") (/ v 4.0) v)]) md)))
   :beta-alpha-names-swapped (names-swapped "ssm_beta" "ssm_alpha")
   :k-v-names-swapped (names-swapped "attn_k" "attn_v")
   :gate-up-names-swapped (names-swapped "ffn_gate" "ffn_up")
   :layer-off-by-one (rewritten #(mapv (fn [t] (update t :name (fn [n] (str/replace n #"^blk\.(\d+)\." (fn [[_ i]] (str "blk." (inc (parse-long i)) ".")))))) %))})

(def ^:private delta lora/delta)

(def ^:private forward-mutations
  "Planted errors in the forward's adapter term and placement."
  {:scale-alpha {#'lora/delta (fn [x w] (array/* (first (array/shape (get-in w [:lora :a]))) (delta x w)))}
   :beta-alpha-weights-swapped {#'lora/paths {:beta [:alpha] :alpha [:beta]
                                              :gate-shexp [:moe :shared :gate] :up-shexp [:moe :shared :up] :down-shexp [:moe :shared :down]}}})

(deftest planted-convention-errors-fail
  ;; on projections of equal shapes (k and v, beta and alpha, gate and
  ;; up), where llama.cpp's shape checks pass a misnamed adapter; one
  ;; prompt, read whole, as a chunk and one step. Measured on Metal: the
  ;; printed lines
  (when-qwen35
    (let [[lm m] (setup)
          ad (ours m [:k :v :beta :alpha :gate :up] 0.05)
          prompt [(second prompts)]
          clean (comparisons lm (lora/adapt m ad) (lora/save-gguf ad (temp-path "clean.gguf")) 1.0 prompt 1)
          _ (println (format "lora-qwen35-llama: the planted errors' adapter moves log p by %.3g nats; against llama.cpp at most %.3g"
                             (effect lm m (lora/adapt m ad) prompt) (worst clean)))
          _ (is (agree? clean) "the adapter itself agrees")
          margins (into {}
                        (concat
                         (for [[k redefs] (sort writer-mutations)
                               :let [path (with-redefs-fn redefs #(lora/save-gguf ad (temp-path (str (name k) ".gguf"))))]]
                           [k (worst (comparisons lm (lora/adapt m ad) path 1.0 prompt 1))])
                         (for [[k redefs] (sort forward-mutations)
                               :let [path (lora/save-gguf ad (temp-path "forward.gguf"))]]
                           [k (with-redefs-fn redefs #(worst (comparisons lm (lora/adapt m ad) path 1.0 prompt 1)))])))]
      (println "lora-qwen35-llama: planted errors, |Δ log p| or :refused:" (pr-str margins))
      (doseq [[k v] margins]
        (is (or (= :refused v) (< mutation-nats v)) (str k ": " v))))))

;; ---------------------------------------------------------------------------
;; 3. Training, then serving
;; ---------------------------------------------------------------------------

(def ^:private text
  (str "Gen.clj is a probabilistic programming system on the JVM. A generative function names its random "
       "choices, so that inference can constrain some of them to observations and propose the others."))

(deftest lora-steps-lower-the-loss-and-llama-cpp-serves-the-result
  ;; rank 8 on every target of the last 8 layers (6 GatedDeltaNet, 2
  ;; attention), AdamW at 1e-3 clipped to norm 1, 5 steps on one text,
  ;; checkpointed: each step lowers the loss; the trained adapter, saved,
  ;; reads in llama.cpp as in the forward. Measured on Metal: the printed
  ;; line
  (when-qwen35
    (let [[lm m] (setup)
          tokens (with-open [^java.io.Closeable e ((llama 'engine) {:model lm :max-branches 2 :n-ctx 512})]
                   ((llama 'tokenize) e text))
          n (get-in m [:config :layers])
          ad (lora/init m {:rank 8 :alpha 16 :targets (vec (keys (lora/targets "qwen35"))) :layers (range (- n 8) n)} (rng/key 3))
          opt (learn/clip-norm (learn/adamw {:lr 1e-3}) {:max-norm 1.0})
          loss (fn [params] (qwen35/nll (lora/adapt m (assoc ad :params params)) [tokens] {:checkpoint? true :chunk 32768}))
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
      (println (format "lora-qwen35-llama: %d tokens, the loss %s; the trained adapter moves log p by %.3g nats; against llama.cpp at most %.3g"
                       (count tokens) (pr-str (mapv #(Math/round (* 1000.0 (double %))) losses)) (effect lm m m' prompts) (worst cs)))
      (is (apply > losses) (pr-str losses))
      (is (< (last losses) (* 0.5 (first losses))) (pr-str losses))
      (is (agree? cs) (pr-str (disagreements cs band))))))

;; ---------------------------------------------------------------------------
;; 4. The shared expert: the tiny Qwen3.6 MoE with a vocabulary (opt-in)
;; ---------------------------------------------------------------------------

(def ^:private vocab-models (System/getProperty "gen.qwen35.vocab-models"))

(def ^:private vocab-band "qwen35-llama-test's band for these models." 0.05)

(def ^:private vocab-models?
  (delay
    (let [skip (fn [why] (println "Skipping the tiny Qwen3.6 MoE LoRA-against-llama.cpp test:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (not (.isFile (java.io.File. (str vocab-models "/qwen35moe-vocab-f32.gguf")))) (skip (str "no models in " vocab-models))
        (ts/jvm?) (skip "the JVM backend: a vocabulary of 248320 needs MLX (-J-Dgen.backend=mlx)")
        :else (try ((llama 'system-info)) true
                   (catch Throwable t (skip (str (or (ex-message t) (.getName (class t)))))))))))

(deftest the-moe-s-adapter-reads-in-llama-cpp-as-in-the-forward
  ;; opt-in, as qwen35-llama-test's tiny models with a vocabulary (they
  ;; are generated, not committed). Measured on Metal: the printed lines
  (if (nil? vocab-models)
    (println "gen.llm.lora-qwen35-llama-test: the MoE's adapter is checked given -J-Dgen.qwen35.vocab-models=<dir>")
    (ts/when-native :llama vocab-models?
      (doseq [f ["qwen35moe-vocab-f32" "qwen35moe-vocab-q8_0"]
              :let [file (str vocab-models "/" f ".gguf")
                    m (qwen35/load file)
                    ad (drawn-b (lora/init m {:rank 4 :alpha 8 :targets (vec (keys (lora/targets "qwen35moe")))} (rng/key 5)) 0.1 (rng/key 6))
                    path (lora/save-gguf ad (temp-path (str f ".gguf")))]]
        (with-open [^java.io.Closeable lm ((llama 'model) file)]
          (let [cs (comparisons lm (lora/adapt m ad) path 1.0 (rest prompts) 2 vocab-band)
                margins (into (sorted-map)
                              (for [[what x y] [[:shared "ffn_gate_shexp" "ffn_up_shexp"] [:routed "ffn_gate_exps" "ffn_up_exps"]]
                                    :let [swapped (with-redefs-fn (names-swapped x y)
                                                    #(lora/save-gguf ad (temp-path (str f "-" (name what) ".gguf"))))]]
                                [what (worst (comparisons lm (lora/adapt m ad) swapped 1.0 [(second prompts)] 1 vocab-band))]))]
            (println (format "lora-qwen35-llama: %s, the adapter moves log p by %.3g nats; against llama.cpp at most %.3g; gate and up swapped %s"
                             f (effect lm m (lora/adapt m ad) (rest prompts)) (worst cs) (pr-str margins)))
            (is (agree? cs vocab-band) (str f ": " (pr-str (disagreements cs vocab-band))))
            (doseq [[what margin] margins]
              (is (or (= :refused margin) (< mutation-nats margin)) (str f ": the " (name what) " experts' gate and up swapped, " margin)))))))))
