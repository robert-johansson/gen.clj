(ns gen.llm.lora-test
  "LoRA adapters (gen.llm.lora, experimental) on the tiny random Qwen3 of
   test/fixtures/gguf (vocabulary 128, hidden 64, 3 blocks; its q and o
   projections square, k and v [32 64], gate and up [128 64], down [64
   128]), F32 and Q8_0, on the backend under test and its gradients (the
   tape, or MLX's native ones with -Dgen.mlx.grad=native):
   1. a new adapter (B = 0) changes nothing: the base's log-probabilities
      exactly, with dropout, at any scale; :scale 0 too;
   2. init draws A uniform on [-1/√in, 1/√in] from a key per site, B zero;
   3. the adapted loss's gradients with respect to A and B match central
      differences (the JVM backend, float64), through dense and Q8_0
      bases, checkpointed, chunked, masked and under dropout; and the
      checkpointed gradients are the plain ones, exactly on the JVM
      backend (a block's recompute draws the same dropout masks);
   4. on MLX, the gradients are the JVM backend's float64 tape's within
      float32's band;
   5. AdamW steps on the adapter lower the loss;
   6. merging an adapter gives the adapted forward's values;
   7. dropout is unbiased, and keyed;
   8. an adapter saved as llama.cpp's GGUF loads back the same, and an
      adapter gguf-py wrote is written again byte for byte;
   9. gen.llm.array-engine serves an adapted model: its rows are the
      adapted forward's;
   10. misuse is loud.
   Against llama.cpp (the conventions: A and B, their transposes, the
   scale, the tensor names) is gen.llm.lora-llama-test's, on a real
   model.

   Planted bugs (2026-11-07, the JVM backend, with-redefs-fn), each
   failing the tests named: lora/delta dropped (the base alone): 1, 5
   (planted-bugs-fail-the-training-check), 6, 7 and 9; the loss ascended:
   5; a dropout mask drawn from a fresh key at each call: 3 (central
   differences off by 100%, checkpointed gradients off the plain ones by
   0.34-0.40 of the largest) and 7; merge's B·A transposed (the square q
   and o): 6, by 4.8 nats; dropout's 1/(1 - p) dropped: 7; init's B
   nonzero: 1 and 2; save-gguf's A and B swapped: 8.

   The bands on MLX: float32 round-off, four times the largest gap
   measured on fenrir (2026-11-07: CUDA and the MLX CPU, tape and native
   gradients; each band's docstring). It passes on Metal too (Mac mini
   M4, 2026-11-07, tape and native gradients)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lora :as lora]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(def ^:private tokens [[5 17 42 3 88 11 60 2 9] [1 100 3 127 5 64 7 8 0]])

(def ^:private mask
  "The positions trained on: the first row's prompt (its first 3 targets)
   masked."
  (delay (array/array [[0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0]])))

(def ^:private every-target (vec (keys (lora/targets "qwen3"))))

(defn- adapter
  "A new adapter of model k: rank 4, alpha 8 (s = 2), on every projection
   of every layer, but for opts."
  ([k] (adapter k nil))
  ([k opts] (lora/init (@models k) (merge {:rank 4 :alpha 8 :targets every-target} opts) (rng/key 7))))

(defn- with-b
  "adapter with each B drawn N(0, sd²), from key: an adapter that changes
   the model."
  [adapter sd key]
  (update adapter :params
          (fn [params]
            (into {} (for [[i ts] params]
                       [i (into {} (for [[t {:keys [a b]}] ts]
                                     [t {:a a :b (array/* sd (rng/normal (rng/fold-in (rng/fold-in key i) (count (name t)))
                                                                        (array/shape b) {:dtype (array/dtype b)}))}]))])))))

(defn- host [x] (ts/host x))

(defn- max-abs
  "max |x - y| over two arrays (or trees of them), a host double."
  ^double [x y]
  (reduce max 0.0 (map (fn [a b] (Math/abs (- (double a) (double b))))
                       (flatten (host (array/tree-leaves x))) (flatten (host (array/tree-leaves y))))))

(defn- temp-file
  "A path in a new temporary directory, no file there yet."
  [name]
  (str (Files/createTempDirectory "lora-test" (make-array FileAttribute 0)) "/" name))

;; ---------------------------------------------------------------------------
;; 1. A new adapter changes nothing
;; ---------------------------------------------------------------------------

(deftest a-new-adapter-changes-nothing
  ;; B = 0: x·Aᵀ·0 is exactly 0, and so is its scale's product, on every
  ;; backend: the base's values bit for bit
  (doseq [[k m] @models
          :let [base (qwen3/log-probs m tokens)
                ad (adapter k {:dropout 0.25})]]
    (testing (name k)
      (is (= 0.0 (max-abs base (qwen3/log-probs (lora/adapt m ad) tokens))))
      (is (= 0.0 (max-abs base (qwen3/log-probs (lora/adapt m ad {:scale 3.0 :key (rng/key 1)}) tokens))) "dropout, scale 3")
      (is (= 0.0 (max-abs (qwen3/forward m tokens) (qwen3/forward (lora/adapt m ad) tokens))) "the logits")
      (is (= 0.0 (max-abs base (qwen3/log-probs (lora/adapt m (with-b ad 0.5 (rng/key 2)) {:scale 0}) tokens)))
          "scale 0, B ≠ 0")
      (is (< 0.1 (max-abs base (qwen3/log-probs (lora/adapt m (with-b ad 0.5 (rng/key 2))) tokens)))
          "B ≠ 0 changes the model"))))

;; ---------------------------------------------------------------------------
;; 2. init
;; ---------------------------------------------------------------------------

(deftest init-draws-a-uniform-and-b-zero
  (let [m (:f32 @models)
        ad (adapter :f32)
        as (for [[_ ts] (:params ad) [t {:keys [a]}] ts] [t a])]
    (is (= {:architecture "qwen3" :rank 4 :alpha 8.0 :dropout 0.0} (:config ad)))
    (is (= #{0 1 2} (set (keys (:params ad)))))
    (is (every? #(= (set every-target) (set (keys %))) (vals (:params ad))))
    (doseq [[i ts] (:params ad) [t {:keys [a b]}] ts
            :let [[out in] (array/shape (get-in m [:weights :layers i t]))]]
      (is (= [4 in] (array/shape a)) (pr-str [i t]))
      (is (= [out 4] (array/shape b)) (pr-str [i t]))
      (is (= (ts/float-dtype) (array/dtype a) (array/dtype b)))
      (is (every? zero? (flatten (host b))) "B starts at zero"))
    ;; A's entries, scaled by √in to U(-1, 1): mean 0 and variance 1/3
    (let [us (mapcat (fn [[_ a]] (map #(* % (Math/sqrt (second (array/shape a)))) (flatten (host a)))) as)]
      (is (every? #(<= -1.0 % 1.0) us))
      (is (< 4000 (count us)))
      (is (ts/moments-match? us 0.0 (/ 1.0 3.0))))
    (testing "a key per site"
      (is (= (host (get-in ad [:params 1 :v :a]))
             (host (get-in (adapter :f32 {:layers [1] :targets [:v]}) [:params 1 :v :a])))
          "a layer's and target's A whatever else is adapted")
      (is (not= (host (get-in ad [:params 1 :v :a])) (host (get-in ad [:params 2 :v :a]))))
      (is (not= (host (get-in ad [:params 1 :v :a]))
                (host (get-in (lora/init m {:rank 4 :alpha 8 :targets [:v]} (rng/key 8)) [:params 1 :v :a])))))
    (is (= [:q :v] (vec (sort (keys (get-in (lora/init m nil (rng/key 1)) [:params 0])))))
        "the defaults: q and v, rank 8")
    (is (= {:architecture "qwen3" :rank 8 :alpha 16.0 :dropout 0.0} (:config (lora/init m nil (rng/key 1)))))))

;; ---------------------------------------------------------------------------
;; 3. Gradients
;; ---------------------------------------------------------------------------

(def ^:private training {:checkpoint? true :chunk 48})

(defn- loss-of
  "The adapted nll of model k as a function of an adapter's params (and
   the key of its dropout), options opts."
  [k ad opts]
  (fn [params key]
    (qwen3/nll (lora/adapt (@models k) (assoc ad :params params) (when key {:key key})) tokens opts)))

(defn- changed
  "Model k's adapter that changes it: B ≠ 0, so A's gradients are not 0."
  [k opts]
  (with-b (adapter k opts) 0.3 (rng/key 11)))

(defn- value-and-grads
  "[nll, its gradient with respect to the params] of model k's adapter ad,
   options opts, dropout drawn from key (nil: none)."
  [k ad opts key]
  ((array/value-and-grad (loss-of k ad opts)) (:params ad) key))

(def ^:private entries
  "The entries differentiated, by path in the params: A's and B's, in
   the first and last layers, attention and MLP."
  [[0 :q :a 0 0] [0 :q :b 63 3] [0 :k :a 2 40] [1 :v :b 31 1] [1 :o :a 3 63] [2 :gate :a 1 5]
   [2 :up :b 100 2] [2 :down :a 0 127] [2 :down :b 7 0]])

(defn- central-difference
  "d loss / d (the params' entry at path), by central differences."
  [loss params path]
  (let [[i t ab & ij] path
        x (get-in params [i t ab])
        w (host x)
        at (fn [d] (array/item (loss (assoc-in params [i t ab] (array/array (update-in w (vec ij) + d) {:dtype (array/dtype x)})))))
        h 1e-5]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- worst-relative-error
  "The largest error of the gradients of model k's adapted nll (opts, a
   dropout of p from key) against central differences, |a - b| / max(|b|,
   1e-3), over the entries."
  [k opts p key]
  (let [ad (changed k {:dropout p})
        [_ g] (value-and-grads k ad opts key)
        loss #((loss-of k ad opts) % key)]
    (reduce max (for [path entries]
                  (let [a (get-in (host g) path)
                        b (central-difference loss (:params ad) path)]
                    (/ (Math/abs (- (double a) b)) (max 1e-3 (Math/abs b))))))))

(deftest gradients-are-central-differences
  ;; measured 2026-11-07: errors up to 2.4e-8 (F32) and 7.9e-8 (Q8_0),
  ;; central differences' round-off at h = 1e-5. The band 1e-6, as
  ;; qwen3-train-test's
  (if-not (ts/jvm?)
    (println "gen.llm.lora-test: central differences on the JVM backend (float64)")
    (do (is (< (worst-relative-error :f32 (assoc training :mask @mask) 0.0 nil) 1e-6)
            "F32: checkpointed, chunked, masked")
        (is (< (worst-relative-error :f32 training 0.25 (rng/key 5)) 1e-6) "F32: under dropout")
        (is (< (worst-relative-error :q8_0 (assoc training :mask @mask) 0.25 (rng/key 5)) 1e-6)
            "Q8_0: through the quantized products, under dropout"))))

(def ^:private plain-band
  "Checkpointed against plain: on the JVM backend the same ops in the
   same order, exactly (chunks apart: their logsumexps sum in another
   order, round-off). On MLX, float32's round-off relative to the
   largest gradient: four times the largest gap measured on fenrir
   (2026-11-07, CUDA and the CPU, tape and native gradients), 4.1e-7
   (Q8_0 checkpointed and chunked, the MLX CPU's tape; checkpoints alone
   0 under native gradients)."
  (if (ts/jvm?) {:checkpoint 0.0 :chunk 1e-14} {:checkpoint 1.7e-6 :chunk 1.7e-6}))

(defn- gap
  "The largest |Δ| between two results' values and gradients, relative to
   the largest |gradient| (or 1)."
  [a b]
  (let [fa (flatten (host (array/tree-leaves a)))
        fb (flatten (host (array/tree-leaves b)))
        scale (max 1.0 (reduce max (map #(Math/abs (double %)) fb)))]
    (/ (reduce max (map (fn [x y] (Math/abs (- (double x) (double y)))) fa fb)) scale)))

(deftest checkpointed-gradients-are-the-plain-ones
  (doseq [k (keys files)
          [p key] [[0.0 nil] [0.25 (rng/key 5)]]
          :let [ad (changed k {:dropout p})
                plain (value-and-grads k ad nil key)]]
    (testing [k p]
      (is (<= (gap (value-and-grads k ad {:checkpoint? true} key) plain) (:checkpoint plain-band)) "checkpointed")
      (is (<= (gap (value-and-grads k ad training key) plain) (:chunk plain-band)) "checkpointed and chunked"))))

;; ---------------------------------------------------------------------------
;; 4. MLX against the JVM backend's tape
;; ---------------------------------------------------------------------------

(def ^:private mlx-band
  "MLX's gradients against the JVM backend's float64 tape, relative to
   the largest gradient: four times the largest gap measured on fenrir
   (2026-11-07, CUDA and the CPU, tape and native gradients), 3.6e-7
   (Q8_0 on the MLX CPU's tape)."
  1.5e-6)

(deftest mlx-gradients-are-the-tape-s
  ;; without dropout: its masks are drawn from uniforms of the backend's
  ;; float, which round differently in float32 and float64
  (if (ts/jvm?)
    (println "gen.llm.lora-test: MLX's gradients are compared with the JVM backend's on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [[k file] files
              opts [nil training]]
        (let [ad (changed k nil)
              mine (value-and-grads k ad opts nil)
              theirs (array/with-backend jvm
                       (let [m (qwen3/load file)
                             ad (assoc ad :params (array/tree-map #(array/astype (array/array (host %)) :float64) (:params ad)))]
                         (host ((array/value-and-grad (fn [params] (qwen3/nll (lora/adapt m (assoc ad :params params)) tokens opts)))
                                (:params ad)))))]
          (is (<= (gap mine theirs) mlx-band) (str (name k) " " opts (when (ts/native-grad?) ", native"))))))))

;; ---------------------------------------------------------------------------
;; 5. Training steps lower the loss
;; ---------------------------------------------------------------------------

(defn- losses
  "The adapted nll of model k before each of n AdamW steps from a new
   adapter (and after the last), each step along the gradient times sign
   (-1: gen.learning ascends, so the loss's negated gradient), clipped to
   a norm of 10: n + 1 host doubles."
  [k n sign]
  (let [ad (adapter k {:dropout 0.1})
        opt (learn/clip-norm (learn/adamw {:lr 1e-2}) {:max-norm 10.0})
        loss (loss-of k ad (assoc training :mask @mask))]
    (loop [i 0 params (:params ad) state (learn/init opt (:params ad)) out [(array/item (loss (:params ad) nil))]]
      (if (= i n)
        out
        (let [[_ g] ((array/value-and-grad loss) params (rng/fold-in (rng/key 21) i))
              [params state] (learn/step opt state params (array/tree-map #(array/* sign %) g))]
          (recur (inc i) params state (conj out (array/item (loss params nil)))))))))

(defn- lowered?
  "Do the losses fall at every step, the last below 0.6 of the first?"
  [ls]
  (and (apply > ls) (< (last ls) (* 0.6 (first ls)))))

(deftest training-steps-lower-the-loss
  ;; measured 2026-11-07 (the JVM backend): F32 72.3 → 55.8 → 40.2 →
  ;; 27.3 → 16.7 → 8.5; Q8_0 the same within 0.15 (the tiny model
  ;; memorizes 13 targets). On MLX (fenrir) 72.3 → 54.4 → 37.8 → 24.2 →
  ;; 13.3 → 6.4: its float32 uniforms draw other dropout masks
  (doseq [k (keys files)]
    (let [ls (losses k 5 -1.0)]
      (is (lowered? ls) (str (name k) ": " (pr-str ls))))))

(deftest planted-bugs-fail-the-training-check
  (let [delta lora/delta]
    (is (not (lowered? (with-redefs-fn {#'lora/delta (fn [x w] (array/* 0.0 (delta x w)))} #(losses :f32 3 -1.0))))
        "the adapter's term dropped")
    (is (not (lowered? (losses :f32 3 1.0))) "the loss ascended")))

;; ---------------------------------------------------------------------------
;; 6. Merging
;; ---------------------------------------------------------------------------

(def ^:private merge-band
  "|Δ log p| between the merged and the adapted forwards: float64's
   round-off on the JVM backend (6.2e-15 measured); float32's on MLX,
   four times the largest measured on fenrir (2026-11-07), 3.8e-6 (Q8_0
   on CUDA)."
  (if (ts/jvm?) 1e-12 1.6e-5))

(deftest merging-gives-the-adapted-values
  (doseq [[k m] @models
          scale [1.0 -0.5]
          :let [ad (with-b (adapter k) 0.3 (rng/key 3))
                merged (lora/merge m ad {:scale scale})]]
    (testing [k scale]
      (is (<= (max-abs (qwen3/log-probs merged tokens) (qwen3/log-probs (lora/adapt m ad {:scale scale}) tokens)) merge-band))
      (is (< 0.1 (max-abs (qwen3/log-probs merged tokens) (qwen3/log-probs m tokens))) "the merged model is not the base")
      (doseq [[i layer] (map-indexed vector (get-in merged [:weights :layers]))
              t every-target
              :let [{:keys [a b]} (get-in ad [:params i t])]]
        (is (array/array? (layer t)) (pr-str [i t "a dense matrix"]))
        (is (= [(first (array/shape b)) (second (array/shape a))] (array/shape (layer t))) (pr-str [i t]))))))

;; ---------------------------------------------------------------------------
;; 7. Dropout
;; ---------------------------------------------------------------------------

(deftest dropout-is-unbiased-and-keyed
  (let [x (rng/normal (rng/key 1) [2 5 16])
        a (rng/normal (rng/key 2) [3 16])
        b (rng/normal (rng/key 3) [8 3])
        w (fn [key] {:base nil :lora (cond-> {:a a :b b :s 0.5} key (assoc :p 0.3 :key key))})
        exact (lora/delta x (w nil))
        draw (fn [i] (lora/delta x (w (rng/fold-in (rng/key 4) i))))
        ;; a statistic of the term: its entries' sum, and one entry
        sums (mapv #(array/item (array/sum (draw %))) (range 400))
        entry (mapv #(array/item (array/slice (array/reshape (draw %) [-1]) 0 17 18)) (range 400))]
    (is (ts/within-5-se? sums (array/item (array/sum exact))) "unbiased: its sum")
    (is (ts/within-5-se? entry (array/item (array/slice (array/reshape exact [-1]) 0 17 18))) "unbiased: an entry")
    (is (< 0.0 (ts/variance sums)) "random")
    (is (= (host (draw 0)) (host (draw 0))) "the same key, the same masks")
    (is (not= (host (draw 0)) (host (draw 1))))))

;; ---------------------------------------------------------------------------
;; 8. llama.cpp's GGUF
;; ---------------------------------------------------------------------------

(deftest an-adapter-saved-loads-back
  (doseq [[k m] @models
          :let [ad (with-b (adapter k {:targets [:q :v :down] :layers [0 2] :alpha 3.5}) 0.3 (rng/key 3))
                f (temp-file "a.gguf")]]
    (try
      (is (= f (lora/save-gguf ad f)))
      (let [h (gguf/header f)
            back (lora/load-gguf f)
            f32 (fn [x] (host (array/astype x :float32)))]
        (is (= {"general.architecture" "qwen3" "general.type" "adapter" "adapter.type" "lora" "adapter.lora.alpha" 3.5}
               (:metadata h)))
        (is (= :float32 ((:value-types h) "adapter.lora.alpha")))
        (is (= ["blk.0.attn_q.weight.lora_a" "blk.0.attn_q.weight.lora_b" "blk.0.attn_v.weight.lora_a" "blk.0.attn_v.weight.lora_b"
                "blk.0.ffn_down.weight.lora_a" "blk.0.ffn_down.weight.lora_b"
                "blk.2.attn_q.weight.lora_a" "blk.2.attn_q.weight.lora_b" "blk.2.attn_v.weight.lora_a" "blk.2.attn_v.weight.lora_b"
                "blk.2.ffn_down.weight.lora_a" "blk.2.ffn_down.weight.lora_b"]
               (mapv :name (:tensors h))))
        (is (every? #(= :f32 (:type %)) (:tensors h)))
        (is (= [[4 64] [32 4] [4 128] [64 4]] (mapv :shape (take 4 (drop 2 (:tensors h)))))
            "gen.array's shapes: A [r in], B [out r]")
        (is (= (:config ad) (:config back)))
        (is (= (array/tree-map f32 (:params ad)) (array/tree-map f32 (:params back))) "the params, rounded to float32")
        (is (= (ts/float-dtype) (array/dtype (get-in back [:params 0 :q :a]))) "in the backend's float")
        (is (<= (max-abs (qwen3/log-probs (lora/adapt m back) tokens) (qwen3/log-probs (lora/adapt m ad) tokens))
                (if (ts/jvm?) 1e-5 0.0))
            (str (name k) ": the same model, but for float32's rounding of the params")))
      (finally (Files/deleteIfExists (Path/of f (make-array String 0)))))))

(deftest gguf-py-s-adapter-is-written-again-byte-for-byte
  ;; test/fixtures/lora/qwen3-0.6b-lora.gguf, written by llama.cpp's
  ;; gguf-py (generate.py): read, written again under its name
  (let [fixture "test/fixtures/lora/qwen3-0.6b-lora.gguf"
        ad (lora/load-gguf fixture)
        f (temp-file "a.gguf")
        bytes #(Files/readAllBytes (Path/of ^String % (make-array String 0)))]
    (try
      (is (= {:architecture "qwen3" :rank 2 :alpha 4.0 :dropout 0.0} (:config ad)))
      (is (= #{0 7 14 21} (set (keys (:params ad)))))
      (lora/save-gguf ad f {:name "gen.clj test adapter: qwen3-0.6b-lora.gguf"})
      (is (java.util.Arrays/equals ^bytes (bytes fixture) ^bytes (bytes f)))
      (finally (Files/deleteIfExists (Path/of f (make-array String 0)))))))

;; ---------------------------------------------------------------------------
;; 9. The array engine serves an adapted model
;; ---------------------------------------------------------------------------

(def ^:private engine-band
  "gen.llm.array-engine-test's band: 2e-6 on the JVM backend, 2e-5 on MLX
   (3.3e-6 measured here on fenrir, 2026-11-07)."
  (if (ts/jvm?) 2e-6 2e-5))

(defn- forward-row
  "The log-probabilities after ctx by qwen3's forward of model m,
   normalized on the host in double, as floats."
  ^floats [m ctx]
  (let [[logits] (qwen3/prefill m [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        mx (reduce max xs)
        z (+ mx (Math/log (reduce + (map #(Math/exp (- % mx)) xs))))]
    (float-array (map #(- % z) xs))))

(deftest the-array-engine-serves-an-adapted-model
  (doseq [[k m] @models
          :let [adapted (lora/adapt m (with-b (adapter k) 0.3 (rng/key 3)))
                prompt [95 51 72 69 0 67 65 80 73 84 65 76 0 73 83]]]
    (with-open [e (ae/engine {:model adapted :max-branches 8 :block 16 :n-batch 8})]
      (let [b (br/open e prompt)
            [s] (br/advance [(br/fork b)] [42])]
        (is (<= (bc/max-diff (forward-row adapted prompt) (br/logprobs b) 128) engine-band) (str (name k) ": a read"))
        (is (<= (bc/max-diff (forward-row adapted (conj prompt 42)) (br/logprobs s) 128) engine-band) (str (name k) ": a step"))
        (is (< 0.1 (bc/max-diff (forward-row m prompt) (br/logprobs b) 128)) (str (name k) ": not the base's"))
        (br/dispose! [b s])))))

;; ---------------------------------------------------------------------------
;; 10. Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:f32 @models)
        ad (adapter :f32)
        reason (fn [f] (:reason (ts/thrown-data (f))))
        key (rng/key 1)]
    (testing "init"
      (is (= :unknown-options (reason #(lora/init m {:r 4} key))))
      (is (= :invalid-args (reason #(lora/init m {:rank 0} key))))
      (is (= :invalid-args (reason #(lora/init m {:alpha 0} key))))
      (is (= :invalid-args (reason #(lora/init m {:dropout 1.0} key))))
      (is (= :invalid-args (reason #(lora/init m {:targets [:qkv]} key))))
      (is (= :invalid-args (reason #(lora/init m {:targets [:q :q]} key))))
      (is (= :invalid-args (reason #(lora/init m {:layers [3]} key))) "the tiny model has 3 layers")
      (is (= :invalid-args (reason #(lora/init m nil 7))) "a key, not a seed"))
    (testing "adapt"
      (is (= :unknown-options (reason #(lora/adapt m ad {:dropout 0.1}))))
      (is (= :invalid-args (reason #(lora/adapt m ad {:key 3}))))
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:params 0 :q :a] (array/zeros [4 32]))))) "A of another shape")
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:params 0 :q :b] (array/zeros [64 3]))))) "B of another rank")
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:params 5] (get-in ad [:params 0]))))) "a layer m lacks")
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:config :alpha] -1.0))))))
    (testing "GGUF"
      (is (= :invalid-args (reason #(lora/save-gguf (assoc-in ad [:config :architecture] nil) (temp-file "a.gguf")))))
      (is (= :unknown-options (reason #(lora/save-gguf ad (temp-file "a.gguf") {:alpha 2}))))
      (is (= :invalid-args (reason #(lora/load-gguf (:f32 files)))) "a model is no adapter")
      (is (= :unsupported (reason #(lora/load-gguf "test/fixtures/lora/qwen3-0.6b-alora.gguf"))) "an aLoRA"))))
