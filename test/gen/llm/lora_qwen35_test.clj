(ns gen.llm.lora-qwen35-test
  "LoRA adapters (gen.llm.lora, experimental) on Qwen3.5's and Qwen3.6's
   forward (gen.llm.models.qwen35), on the tiny random qwen35 and
   qwen35moe of test/fixtures/gguf (vocabulary 128, hidden 32, 3
   GatedDeltaNet blocks then a gated attention block; the MoE's shared
   expert), F32 and Q8_0, on the backend under test and its gradients
   (the tape, or MLX's native ones with -Dgen.mlx.grad=native):
   1. a new adapter (B = 0) changes nothing: the base's log-probabilities
      exactly, with dropout, at any scale; :scale 0 too;
   2. init adapts each target in the layers that have it (GatedDeltaNet's
      :qkv :z :beta :alpha :out, attention's :q :k :v :o, the
      feed-forward's or the shared expert's), with the family's defaults;
      and each target's weight is the GGUF tensor of llama.cpp's name for
      it (targets);
   3. the adapted loss's gradients with respect to A and B match central
      differences (the JVM backend, float64), on every target, through
      dense and Q8_0 bases, checkpointed, chunked, masked and under
      dropout, and over a row of 70 tokens (gated-delta's chunked form);
      the checkpointed gradients are the plain ones (exactly on the JVM
      backend);
   4. on MLX, the gradients are the JVM backend's float64 tape's within
      float32's band;
   5. AdamW steps on the adapter lower the loss;
   6. merging an adapter gives the adapted forward's values;
   7. an adapter saved as llama.cpp's GGUF, under qwen35's tensor names,
      loads back the same;
   8. gen.llm.array-engine serves an adapted model: its rows are the
      adapted forward's;
   9. misuse is loud.
   Against llama.cpp (the conventions: the names, A and B, the scale) is
   gen.llm.lora-qwen35-llama-test's, on Qwen3.5-0.8B.

   Planted bugs (2026-11-07, the JVM backend, with-redefs-fn), each
   failing the tests named: lora/delta's term zeroed (common/linear's
   base alone): 1, 5 (planted-bugs-fail), 6 and 8 (44 assertions); A's
   gradient stopped in lora/delta: 3 (planted-bugs-fail: relative error
   1.0); the loss ascended: 5; the shared expert's gate and up paths
   swapped: 2 (8 assertions: the weight adapted is not the tensor named
   ffn_gate_shexp); the names of :z and :qkv swapped, or of :beta and
   :alpha (equal shapes, which llama.cpp's checks pass): 2 (8 and 12);
   check-params reading a layer's own targets in place of the family's
   (an adapter on the conv kernel accepted): 9."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.learning :as learn]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lora :as lora]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen35-tiny-q8_0.gguf"
   :moe-f32 "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"
   :moe-q8_0 "test/fixtures/gguf/qwen35moe-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen35/load f)])) files)))

(def ^:private tokens [[5 17 42 3 88 11 60 2 9] [1 100 3 127 5 64 7 8 0]])

(def ^:private long-tokens
  "A row of 70 tokens: gated-delta's chunked form (64 tokens or more)."
  [(mapv #(mod (+ 7 (* 41 %)) 128) (range 70))])

(def ^:private mask
  "The positions trained on: the first row's prompt (its first 3 targets)
   masked."
  (delay (array/array [[0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0]])))

(defn- arch [k] (get-in (@models k) [:config :arch]))

(defn- every-target [k] (vec (keys (lora/targets (arch k)))))

(defn- adapter
  "A new adapter of model k: rank 4, alpha 8 (s = 2), on every projection
   of every layer, but for opts."
  ([k] (adapter k nil))
  ([k opts] (lora/init (@models k) (merge {:rank 4 :alpha 8 :targets (every-target k)} opts) (rng/key 7))))

(defn- with-b
  "adapter with each B drawn N(0, sd²), from key: an adapter that changes
   the model."
  [adapter sd key]
  (update adapter :params
          (fn [params]
            (into {} (for [[i ts] params]
                       [i (into {} (for [[t {:keys [a b]}] ts]
                                     [t {:a a :b (array/* sd (rng/normal (rng/fold-in (rng/fold-in key i) (Math/floorMod (long (hash t)) 65536))
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
  (str (Files/createTempDirectory "lora-qwen35-test" (make-array FileAttribute 0)) "/" name))

;; ---------------------------------------------------------------------------
;; 1. A new adapter changes nothing
;; ---------------------------------------------------------------------------

(deftest a-new-adapter-changes-nothing
  ;; B = 0: x·Aᵀ·0 is exactly 0, and so is its scale's product, on every
  ;; backend: the base's values bit for bit
  (doseq [[k m] @models
          :let [base (qwen35/log-probs m tokens)
                ad (adapter k {:dropout 0.25})]]
    (testing (name k)
      (is (= 0.0 (max-abs base (qwen35/log-probs (lora/adapt m ad) tokens))))
      (is (= 0.0 (max-abs base (qwen35/log-probs (lora/adapt m ad {:scale 3.0 :key (rng/key 1)}) tokens))) "dropout, scale 3")
      (is (= 0.0 (max-abs (qwen35/forward m tokens) (qwen35/forward (lora/adapt m ad) tokens))) "the logits")
      (is (= 0.0 (max-abs base (qwen35/log-probs (lora/adapt m (with-b ad 0.5 (rng/key 2)) {:scale 0}) tokens)))
          "scale 0, B ≠ 0")
      (is (< 0.1 (max-abs base (qwen35/log-probs (lora/adapt m (with-b ad 0.5 (rng/key 2))) tokens)))
          "B ≠ 0 changes the model"))))

(deftest each-target-changes-the-model
  ;; one target at a time, B ≠ 0: each adapter term reaches the output (a
  ;; target wired to nothing, or computed and dropped, would not)
  (doseq [k [:f32 :moe-f32]
          t (every-target k)
          :let [m (@models k)
                ad (with-b (adapter k {:targets [t]}) 0.5 (rng/key 2))]]
    (is (< 1e-3 (max-abs (qwen35/log-probs m tokens) (qwen35/log-probs (lora/adapt m ad) tokens))) (str (name k) " " t))))

;; ---------------------------------------------------------------------------
;; 2. init, and the targets' names
;; ---------------------------------------------------------------------------

(deftest init-adapts-the-layers-that-have-each-target
  (let [delta-dense [:qkv :z :beta :alpha :out :gate :up :down]
        attention-dense [:q :k :v :o :gate :up :down]]
    (is (= {0 delta-dense 1 delta-dense 2 delta-dense 3 attention-dense}
           (update-vals (:params (adapter :f32)) #(vec (sort-by (zipmap (every-target :f32) (range)) (keys %))))))
    (is (= #{:qkv :z :beta :alpha :out :gate-shexp :up-shexp :down-shexp} (set (keys (get-in (adapter :moe-f32) [:params 0])))))
    (is (= #{:q :k :v :o :gate-shexp :up-shexp :down-shexp} (set (keys (get-in (adapter :moe-q8_0) [:params 3]))))))
  (doseq [[k m] @models
          :let [ad (lora/init m nil (rng/key 1))]]
    (testing (name k)
      (is (= {:architecture (arch k) :rank 8 :alpha 16.0 :dropout 0.0} (:config ad)))
      (is (= {0 #{:qkv} 1 #{:qkv} 2 #{:qkv} 3 #{:q :v}} (update-vals (:params ad) (comp set keys)))
          "the defaults: attention's q and v, GatedDeltaNet's qkv")
      (doseq [[i ts] (:params ad) [t {:keys [a b]}] ts
              :let [w (get-in m (into [:weights :layers i] (if (#{:gate-shexp :up-shexp :down-shexp} t)
                                                             [:moe :shared (keyword (subs (name t) 0 (- (count (name t)) 6)))]
                                                             [t])))
                    [out in] (common/weight-shape w)]]
        (is (= [8 in] (array/shape a)) (pr-str [i t]))
        (is (= [out 8] (array/shape b)) (pr-str [i t]))
        (is (every? zero? (flatten (host b))) "B starts at zero"))))
  (is (= (host (get-in (adapter :f32) [:params 1 :beta :a]))
         (host (get-in (adapter :f32 {:layers [1] :targets [:beta]}) [:params 1 :beta :a])))
      "a key per site"))

(def ^:private shared-paths
  {:gate-shexp [:moe :shared :gate] :up-shexp [:moe :shared :up] :down-shexp [:moe :shared :down]})

(defn- adapted-weights
  "{[layer target] the base weight adapt puts the adapter beside} of model
   k with an adapter on every target."
  [k]
  (let [ad (adapter k)
        adapted (lora/adapt (@models k) ad)
        found (atom {})]
    (doseq [[i ts] (:params ad) t (keys ts)]
      ;; where adapt put it: the one adapted weight under layer i
      (letfn [(walk [path x]
                (cond (lora/adapted? x) (when (= (:a (:lora x)) (get-in ad [:params i t :a]))
                                          (swap! found assoc [i t] [path (:base x)]))
                      (and (map? x) (not (contains? x :wq))) (doseq [[k v] x] (walk (conj path k) v))))]
        (walk [] (get-in adapted [:weights :layers i]))))
    @found))

(deftest each-target-is-llama-cpp-s-tensor
  ;; the weight adapt adapts for target t of layer i is the one the GGUF
  ;; names blk.i.<(targets arch) t>.weight, as llama.cpp's adapter names
  ;; it: the names and the weights' paths agree
  (doseq [k [:f32 :moe-q8_0]
          :let [{:keys [tensors]} (array/load-gguf (files k))
                names (lora/targets (arch k))
                found (adapted-weights k)]]
    (is (= (count found) (reduce + (map count (vals (:params (adapter k)))))) "every site found once")
    (doseq [[[i t] [_ w]] found
            :let [tensor (tensors (str "blk." i "." (names t) ".weight"))
                  values (fn [x] (if (map? x) (host (:wq x)) (host (array/astype x :float32))))]]
      (is (some? tensor) (pr-str [k i t]))
      (is (= (values tensor) (values w)) (str (name k) ": layer " i "'s " t " is " (names t))))))

;; ---------------------------------------------------------------------------
;; 3. Gradients
;; ---------------------------------------------------------------------------

(def ^:private training {:checkpoint? true :chunk 48})

(defn- loss-of
  "The adapted nll of model k on rows as a function of an adapter's params
   (and the key of its dropout), options opts."
  [k ad rows opts]
  (fn [params key]
    (qwen35/nll (lora/adapt (@models k) (assoc ad :params params) (when key {:key key})) rows opts)))

(defn- changed
  "Model k's adapter that changes it: B ≠ 0, so A's gradients are not 0."
  [k opts]
  (with-b (adapter k opts) 0.3 (rng/key 11)))

(defn- value-and-grads
  "[nll, its gradient with respect to the params] of model k's adapter ad
   on rows, options opts, dropout drawn from key (nil: none)."
  ([k ad opts key] (value-and-grads k ad tokens opts key))
  ([k ad rows opts key] ((array/value-and-grad (loss-of k ad rows opts)) (:params ad) key)))

(def ^:private entries
  "The entries differentiated, by path in the params: A's and B's of
   every target, GatedDeltaNet's (layers 0-2), the attention's (3), the
   feed-forward's or the shared expert's."
  {:dense [[0 :qkv :a 0 0] [0 :qkv :b 200 3] [1 :z :a 2 10] [1 :beta :b 3 1] [2 :alpha :a 1 31] [2 :out :b 31 2]
           [3 :q :b 127 0] [3 :k :a 3 31] [3 :v :b 5 2] [3 :o :a 0 63] [0 :gate :a 1 5] [2 :up :b 63 2] [3 :down :a 0 63]]
   :moe [[0 :qkv :b 17 1] [1 :z :b 100 0] [2 :beta :a 0 7] [1 :alpha :b 2 3] [0 :out :a 1 120]
         [3 :q :a 2 4] [3 :o :b 30 1] [0 :gate-shexp :a 1 5] [2 :up-shexp :b 31 2] [3 :down-shexp :a 0 31]]})

(defn- entries-of [k] (entries (if (#{:moe-f32 :moe-q8_0} k) :moe :dense)))

(defn- central-difference
  "d loss / d (the params' entry at path), by central differences."
  [loss params path]
  (let [[i t ab & ij] path
        x (get-in params [i t ab])
        w (host x)
        at (fn [d] (array/item (loss (assoc-in params [i t ab] (array/array (update-in w (vec ij) + d) {:dtype (array/dtype x)})))))
        h 1e-5]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- relative-errors
  "{entry error} of the gradients of model k's adapted nll on rows (opts,
   a dropout of p from key) against central differences, |a - b| /
   max(|b|, 1e-3)."
  [k rows opts p key]
  (let [ad (changed k {:dropout p})
        [_ g] (value-and-grads k ad rows opts key)
        loss #((loss-of k ad rows opts) % key)]
    (into {} (for [path (entries-of k)]
               (let [a (get-in (host g) path)
                     b (central-difference loss (:params ad) path)]
                 [path (/ (Math/abs (- (double a) b)) (max 1e-3 (Math/abs b)))])))))

(defn- worst [errors] (reduce max (vals errors)))

(deftest gradients-are-central-differences
  ;; measured 2026-11-07: relative errors up to 2.5e-7 (Q8_0), 8.1e-8
  ;; (F32), central differences' round-off at h = 1e-5. The band 1e-6,
  ;; as lora-test's
  (if-not (ts/jvm?)
    (println "gen.llm.lora-qwen35-test: central differences on the JVM backend (float64)")
    (doseq [k (keys files)]
      (testing (name k)
        (is (< (worst (relative-errors k tokens (assoc training :mask @mask) 0.0 nil)) 1e-6) "checkpointed, chunked, masked")
        (is (< (worst (relative-errors k tokens training 0.25 (rng/key 5))) 1e-6) "under dropout")))))

(deftest gradients-through-the-chunked-rule-are-central-differences
  ;; 70 tokens: gated-delta's chunked form, as a training step's
  ;; sequences take it. Measured 2026-11-07: relative errors up to 2.9e-7
  ;; (F32) and 4.3e-7 (MoE Q8_0), a loss of 69 targets' round-off in the
  ;; differences; the band 2e-6
  (if-not (ts/jvm?)
    (println "gen.llm.lora-qwen35-test: central differences on the JVM backend (float64)")
    (doseq [k [:f32 :moe-q8_0]]
      (is (< (worst (relative-errors k long-tokens training 0.0 nil)) 2e-6) (name k)))))

(def ^:private plain-band
  "Checkpointed against plain: on the JVM backend the same ops in the
   same order, exactly (chunks apart: their logsumexps sum in another
   order, round-off). On MLX, lora-test's band (float32's round-off
   relative to the largest gradient)."
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
   the largest gradient: lora-test's band (float32's round-off), checked
   against the gaps measured here (the deftest's comment)."
  1.5e-6)

(deftest mlx-gradients-are-the-tape-s
  ;; without dropout: its masks are drawn from uniforms of the backend's
  ;; float, which round differently in float32 and float64; tokens and
  ;; long-tokens (the composite's two forms)
  (if (ts/jvm?)
    (println "gen.llm.lora-qwen35-test: MLX's gradients are compared with the JVM backend's on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [[k file] files
              [rows opts] [[tokens nil] [tokens training] [long-tokens training]]]
        (let [ad (changed k nil)
              mine (value-and-grads k ad rows opts nil)
              theirs (array/with-backend jvm
                       (let [m (qwen35/load file)
                             ad (assoc ad :params (array/tree-map #(array/astype (array/array (host %)) :float64) (:params ad)))]
                         (host ((array/value-and-grad (fn [params] (qwen35/nll (lora/adapt m (assoc ad :params params)) rows opts)))
                                (:params ad)))))]
          (println (format "lora-qwen35: MLX against the JVM's tape, %s %d tokens %s%s: %.3g"
                           (name k) (count (first rows)) (pr-str opts) (if (ts/native-grad?) " native" "") (gap mine theirs)))
          (is (<= (gap mine theirs) mlx-band) (str (name k) " " (count (first rows)) " tokens " opts (when (ts/native-grad?) ", native"))))))))

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
        loss (loss-of k ad tokens (assoc training :mask @mask))]
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
  ;; measured 2026-11-07 (the JVM backend), the summed nll over 13
  ;; targets: F32 68.6 → 54.7 → 48.1 → 40.5 → 34.1 → 29.3; Q8_0 68.2 →
  ;; 32.3; the MoE's F32 78.4 → 34.8, Q8_0 78.0 → 35.8
  (doseq [k (keys files)]
    (let [ls (losses k 5 -1.0)]
      (println (format "lora-qwen35: %s nll %s" (name k) (pr-str (mapv #(Math/round (* 10.0 (double %))) ls))))
      (is (lowered? ls) (str (name k) ": " (pr-str ls))))))

(deftest planted-bugs-fail
  (if-not (ts/jvm?)
    (println "gen.llm.lora-qwen35-test: the planted bugs run on the JVM backend")
    (let [delta lora/delta]
      (is (not (lowered? (with-redefs-fn {#'lora/delta (fn [x w] (array/* 0.0 (delta x w)))} #(losses :f32 3 -1.0))))
          "the adapter's term dropped")
      (is (not (lowered? (losses :moe-f32 3 1.0))) "the loss ascended")
      (is (< 0.5 (worst (with-redefs-fn {#'lora/delta (fn [x w] (delta x (update-in w [:lora :a] array/stop-gradient)))}
                          #(relative-errors :f32 tokens training 0.0 nil))))
          "A's gradient stopped: the central differences disagree"))))

;; ---------------------------------------------------------------------------
;; 6. Merging
;; ---------------------------------------------------------------------------

(def ^:private merge-band
  "|Δ log p| between the merged and the adapted forwards: float64's
   round-off on the JVM backend; float32's on MLX, lora-test's band."
  (if (ts/jvm?) 1e-12 1.6e-5))

(deftest merging-gives-the-adapted-values
  (doseq [[k m] @models
          scale [1.0 -0.5]
          :let [ad (with-b (adapter k) 0.3 (rng/key 3))
                merged (lora/merge m ad {:scale scale})]]
    (testing [k scale]
      (is (<= (max-abs (qwen35/log-probs merged tokens) (qwen35/log-probs (lora/adapt m ad {:scale scale}) tokens)) merge-band))
      (is (< 0.1 (max-abs (qwen35/log-probs merged tokens) (qwen35/log-probs m tokens))) "the merged model is not the base")
      (doseq [[i ts] (:params ad)
              [t {:keys [a b]}] ts
              :let [w (get-in merged (into [:weights :layers i] (shared-paths t [t])))]]
        (is (array/array? w) (pr-str [i t "a dense matrix"]))
        (is (= [(first (array/shape b)) (second (array/shape a))] (array/shape w)) (pr-str [i t]))))))

;; ---------------------------------------------------------------------------
;; 7. llama.cpp's GGUF
;; ---------------------------------------------------------------------------

(deftest an-adapter-saved-loads-back
  (doseq [[k m] @models
          :let [ts (if (#{:moe-f32 :moe-q8_0} k) [:qkv :beta :q :up-shexp] [:qkv :beta :q :down])
                ad (with-b (adapter k {:targets ts :layers [0 3] :alpha 3.5}) 0.3 (rng/key 3))
                f (temp-file "a.gguf")]]
    (try
      (is (= f (lora/save-gguf ad f)))
      (let [h (gguf/header f)
            back (lora/load-gguf f)
            f32 (fn [x] (host (array/astype x :float32)))
            ffn (if (#{:moe-f32 :moe-q8_0} k) "ffn_up_shexp" "ffn_down")]
        (is (= {"general.architecture" (arch k) "general.type" "adapter" "adapter.type" "lora" "adapter.lora.alpha" 3.5}
               (:metadata h)))
        (is (= (vec (for [n ["blk.0.attn_qkv" "blk.0.ssm_beta" (str "blk.0." ffn) "blk.3.attn_q" (str "blk.3." ffn)]
                          ab ["a" "b"]]
                      (str n ".weight.lora_" ab)))
               (mapv :name (:tensors h)))
            "llama.cpp's names, in the family's order")
        (is (= [[4 32] [256 4] [4 32] [4 4]] (mapv :shape (take 4 (:tensors h)))) "gen.array's shapes: A [r in], B [out r]")
        (is (= (:config ad) (:config back)))
        (is (= (array/tree-map f32 (:params ad)) (array/tree-map f32 (:params back))) "the params, rounded to float32")
        (is (<= (max-abs (qwen35/log-probs (lora/adapt m back) tokens) (qwen35/log-probs (lora/adapt m ad) tokens))
                (if (ts/jvm?) 1e-5 0.0))
            (str (name k) ": the same model, but for float32's rounding of the params")))
      (finally (Files/deleteIfExists (Path/of f (make-array String 0)))))))

;; ---------------------------------------------------------------------------
;; 8. The array engine serves an adapted model
;; ---------------------------------------------------------------------------

(def ^:private engine-band
  "gen.llm.array-engine-qwen35-test's band: 2e-6 on the JVM backend
   (float rounding), 2e-5 on MLX."
  (if (ts/jvm?) 2e-6 2e-5))

(defn- forward-row
  "The log-probabilities after ctx by qwen35's forward of model m,
   normalized on the host in double, as floats."
  ^floats [m ctx]
  (let [[logits] (qwen35/prefill m [ctx] {:logits :last})
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
;; 9. Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:f32 @models)
        moe (:moe-f32 @models)
        ad (adapter :f32)
        reason (fn [f] (:reason (ts/thrown-data (f))))
        key (rng/key 1)]
    (testing "init"
      (is (= :invalid-args (reason #(lora/init m {:targets [:gate-shexp]} key))) "the MoE's target on the dense model")
      (is (= :invalid-args (reason #(lora/init moe {:targets [:gate]} key))) "the routed experts are no target")
      (is (= :invalid-args (reason #(lora/init m {:targets [:conv]} key))))
      (is (= :invalid-args (reason #(lora/init m {:targets [:q] :layers [0 1 2]} key))) "no layer of these has q")
      (is (= :invalid-args (reason #(lora/init m {:targets [:qkv] :layers [3]} key))) "nor qkv")
      (is (= :unsupported (reason #(lora/init (assoc-in m [:config :arch] "llama") nil key))) "a family LoRA does not know"))
    (testing "adapt"
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:params 0 :q] (get-in ad [:params 3 :q]))))) "q on a GatedDeltaNet layer")
      (is (= :invalid-args (reason #(lora/adapt m (assoc-in ad [:params 0 :conv] {:a (array/zeros [4 4]) :b (array/zeros [256 4])}))))
          "the conv kernel, a weight of matching shapes but no target")
      (is (= :invalid-args (reason #(lora/adapt moe (assoc-in (adapter :moe-f32 {:targets [:qkv]}) [:config :architecture] "qwen35"))))
          "an adapter of another architecture")
      (is (= :invalid-args (reason #(lora/adapt moe ad))) "qwen35's adapter on qwen35moe")
      (let [q3 (qwen3/load "test/fixtures/gguf/qwen3-tiny-f32.gguf")]
        (is (= :invalid-args (reason #(lora/adapt m (lora/init q3 nil key)))) "qwen3's adapter on qwen35")))
    (testing "GGUF"
      (is (= :unsupported (reason #(lora/save-gguf (assoc-in ad [:config :architecture] "llama") (temp-file "a.gguf")))))
      (let [f (temp-file "llama.gguf")]
        (gguf/write-gguf f {:metadata [["general.architecture" :string "llama"] ["general.type" :string "adapter"]
                                        ["adapter.type" :string "lora"] ["adapter.lora.alpha" :float32 2.0]]
                            :tensors [{:name "blk.0.attn_q.weight.lora_a" :type :f32 :shape [2 4] :data (float-array 8)}
                                      {:name "blk.0.attn_q.weight.lora_b" :type :f32 :shape [4 2] :data (float-array 8)}]})
        (is (= :unsupported (reason #(lora/load-gguf f))) "an architecture LoRA does not know"))
      (let [f (temp-file "experts.gguf")]
        (gguf/write-gguf f {:metadata [["general.architecture" :string "qwen35moe"] ["general.type" :string "adapter"]
                                        ["adapter.type" :string "lora"] ["adapter.lora.alpha" :float32 2.0]]
                            :tensors [{:name "blk.0.ffn_up_exps.weight.lora_a" :type :f32 :shape [2 4] :data (float-array 8)}
                                      {:name "blk.0.ffn_up_exps.weight.lora_b" :type :f32 :shape [4 2] :data (float-array 8)}]})
        (is (= :unsupported (reason #(lora/load-gguf f))) "the routed experts")))))
