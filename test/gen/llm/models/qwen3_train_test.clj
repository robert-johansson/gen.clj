(ns gen.llm.models.qwen3-train-test
  "Gradients through Qwen3's forward (gen.llm.models.qwen3's log-probs
   and nll, experimental), on the tiny random Qwen3 of test/fixtures/gguf
   (vocabulary 128, hidden 64, 3 blocks), F32 and Q8_0, on the backend
   under test and its gradients (the tape, or MLX's native ones with
   -Dgen.mlx.grad=native):
   1. log-probs are the forward's log-softmax at the next tokens, plain,
      chunked and checkpointed, and at a temperature its logits divided by
      it;
   2. the training loss's gradients (nll, checkpointed per block, its
      output projection by vocabulary chunks of 48: 48, 48 and 32) match
      central differences (the JVM backend, float64): a dense matrix, the
      tied embedding, norms, and the norms through Q8_0's quantized
      products, a mask included;
   3. checkpointed and chunked gradients are the plain ones: exactly with
      checkpoints alone on the JVM backend, within round-off with chunks
      (the chunks' logsumexps sum in another order), within float32's
      band on MLX;
   4. on MLX, its gradients (the tape's, or MLX's native ones) are the
      JVM backend's float64 tape's, within float32's band (below);
   5. a quantized weight is frozen on the tape, and misuse is loud.
   Planted bugs, each failing 3 (2026-11-07, the JVM backend; they run on
   the tape), off relative to the largest gradient by: the recompute
   reading a block's output for its input, 0.16; the recompute's
   cotangent dropped (seeded with ones), 0.56; a chunk boundary off by
   one (each chunk's last entry dropped), 4.5e-4. Each is more than 400
   times MLX's band.
   The bands on MLX are four times the largest gap measured on fenrir
   (2026-11-07: MLX CUDA and the MLX CPU, native and tape gradients;
   not yet measured on Metal), relative to the largest gradient (15 for
   F32, 4.5 for Q8_0): 2.6e-7 between checkpointed or chunked gradients
   and the plain ones (0 under native gradients for checkpoints alone),
   2.6e-7 between MLX and the JVM backend's float64 tape."
  (:require [clojure.test :refer [deftest is]]
            [gen.array :as array]
            [gen.array.nn :as nn]
            [gen.array.tape :as tape]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(defn- host [x] (ts/host x))

(def ^:private tokens [[5 17 42 3 88 11 60 2 9] [1 100 3 127 5 64 7 8 0]])

(def ^:private mask
  "The positions trained on: the first row's prompt (its first 3 targets)
   masked."
  (delay (array/array [[0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0] [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0]])))

(def ^:private trained
  "The dense weights differentiated, by their paths in :weights."
  {:f32 [[:layers 0 :q] [:embed] [:layers 2 :k-norm] [:layers 1 :down] [:norm]]
   :q8_0 [[:layers 0 :attn-norm] [:layers 1 :q-norm] [:layers 2 :ffn-norm] [:norm]]})

(defn- loss-of
  "nll of tokens as a function of the weights at paths (in the model m's
   :weights), options opts."
  [m paths opts]
  (fn [& ws]
    (qwen3/nll (update m :weights #(reduce (fn [w [p x]] (assoc-in w p x)) % (map vector paths ws))) tokens opts)))

(defn- value-and-grads
  "[nll, its gradients] with respect to the trained weights of model k."
  [k opts]
  (let [m (@models k)
        paths (trained k)]
    (apply (array/value-and-grad (loss-of m paths opts) {:argnums (vec (range (count paths)))})
           (map #(get-in (:weights m) %) paths))))

(defn- gap
  "The largest |Δ| between two results' values and gradients, relative to
   the largest |gradient| (or 1)."
  [a b]
  (let [fa (flatten (host (array/tree-leaves a)))
        fb (flatten (host (array/tree-leaves b)))
        scale (max 1.0 (reduce max (map #(Math/abs (double %)) fb)))]
    (/ (reduce max (map (fn [x y] (Math/abs (- (double x) (double y)))) fa fb)) scale)))

(def ^:private training {:checkpoint? true :chunk 48})

;; ---------------------------------------------------------------------------
;; 1. log-probs
;; ---------------------------------------------------------------------------

(deftest log-probs-are-the-forward-s
  (doseq [[k m] @models]
    (let [ids (array/array tokens {:dtype :int32})
          want (array/pick (array/log-softmax (array/slice (qwen3/forward m tokens) 1 0 8)) (array/slice ids 1 1 9))]
      (doseq [opts [nil {:chunk 48} training {:checkpoint? true}]]
        (let [lp (qwen3/log-probs m tokens opts)]
          (is (= [2 8] (array/shape lp)))
          (is (ts/close? lp want {:rtol 1e-4 :atol (if (ts/jvm?) 1e-12 1e-4)}) (str (name k) " " opts))))
      (is (ts/close? (qwen3/nll m tokens {:mask @mask})
                     (array/negate (array/sum (array/* @mask want)))
                     {:rtol 1e-4 :atol (if (ts/jvm?) 1e-12 1e-3)})
          (str (name k) ": nll, masked"))
      (let [tempered (array/pick (array/log-softmax (array// (array/slice (qwen3/forward m tokens) 1 0 8) 0.7))
                                 (array/slice ids 1 1 9))]
        (doseq [opts [{:temperature 0.7} (assoc training :temperature 0.7)]]
          (is (ts/close? (qwen3/log-probs m tokens opts) tempered {:rtol 1e-4 :atol (if (ts/jvm?) 1e-12 1e-4)})
              (str (name k) " " opts ": the logits divided by the temperature")))
        (is (= (host (qwen3/log-probs m tokens training)) (host (qwen3/log-probs m tokens (assoc training :temperature 1.0))))
            "temperature 1 is the model's own, bit for bit")))))

;; ---------------------------------------------------------------------------
;; 2. Central differences (the JVM backend, float64)
;; ---------------------------------------------------------------------------

(defn- central-difference
  "d nll / d (the entry ij of the weight at path), by central differences."
  [m path ij opts]
  (let [w (host (get-in (:weights m) path))
        at (fn [d] (array/item (qwen3/nll (assoc-in m (into [:weights] path)
                                                    (array/array (update-in w ij + d) {:dtype (array/dtype (get-in (:weights m) path))}))
                                          tokens opts)))
        h 1e-5]
    (/ (- (at h) (at (- h))) (* 2 h))))

(def ^:private entries
  {[:layers 0 :q] [[0 0] [17 5] [63 63]]
   [:embed] [[5 4] [9 9] [100 63] [127 0]]
   [:layers 2 :k-norm] [[3] [15]]
   [:layers 1 :down] [[0 0] [40 100]]
   [:norm] [[7] [63]]
   [:layers 0 :attn-norm] [[0] [40]]
   [:layers 1 :q-norm] [[2] [11]]
   [:layers 2 :ffn-norm] [[1] [60]]})

(defn- worst-relative-error
  "The largest error of the gradients of model k's nll (opts) against
   central differences, |a - b| / max(|b|, 1e-3), over the entries."
  [k opts]
  (let [m (@models k)
        [_ grads] (value-and-grads k opts)]
    (reduce max (for [[path g] (map vector (trained k) grads)
                      ij (entries path)]
                  (let [a (get-in (host g) ij)
                        b (central-difference m path ij opts)]
                    (/ (Math/abs (- a b)) (max 1e-3 (Math/abs b))))))))

(deftest gradients-are-central-differences
  ;; measured 2026-11-07: errors up to 4.4e-8 (F32) and 1.3e-8 (Q8_0),
  ;; central differences' round-off at h = 1e-5. The band 1e-6, as
  ;; qwen3-test's for the forward
  (if-not (ts/jvm?)
    (println "gen.llm.models.qwen3-train-test: central differences on the JVM backend (float64)")
    (do (is (< (worst-relative-error :f32 (assoc training :mask @mask)) 1e-6)
            "F32: a dense matrix, the tied embedding and output, norms; checkpointed, chunked, masked")
        (is (< (worst-relative-error :q8_0 training) 1e-6) "Q8_0: norms, through the quantized products"))))

;; ---------------------------------------------------------------------------
;; 3. Checkpointed and chunked gradients are the plain ones
;; ---------------------------------------------------------------------------

(def ^:private plain-band
  "Checkpoints alone: on the JVM backend the same ops in the same order,
   exactly. Chunks: the logsumexp summed in another order, round-off
   (3.2e-16 measured). On MLX, float32's round-off of the gradients,
   relative to the largest: four times the largest measured (2.6e-7)."
  {:checkpoint (if (ts/jvm?) 0.0 1.1e-6)
   :chunk (if (ts/jvm?) 1e-14 1.1e-6)})

(defn- gaps-from-plain
  "{opts gap} of model k's checkpointed and chunked gradients from the
   plain ones."
  [k]
  (let [plain (value-and-grads k nil)]
    (into {} (for [opts [{:checkpoint? true} {:chunk 48} training]]
               [opts (gap (value-and-grads k opts) plain)]))))

(deftest checkpointed-gradients-are-the-plain-ones
  (doseq [k [:f32 :q8_0]]
    (let [gaps (gaps-from-plain k)]
      (is (<= (gaps {:checkpoint? true}) (:checkpoint plain-band)) (str (name k) ": checkpointed"))
      (is (<= (gaps {:chunk 48}) (:chunk plain-band)) (str (name k) ": chunked"))
      (is (<= (gaps training) (:chunk plain-band)) (str (name k) ": both")))))

(deftest planted-bugs-fail-it
  (if (ts/native-grad?)
    (println "gen.llm.models.qwen3-train-test: the tape's planted bugs run on the tape")
    (let [recompute @#'tape/recompute-vjp
          bounds @#'nn/chunk-bounds
          off (fn [redefs] (with-redefs-fn redefs #(gaps-from-plain :f32)))]
      ;; each by more than ninety times MLX's band
      (is (< 1e-4 (get (off {#'tape/recompute-vjp (fn [inner f ps g need]
                                                    (let [out (f ps)]
                                                      ;; a block's (an output projection's chunk is another shape)
                                                      (recompute inner f (if (= (array/shape out) (array/shape (first ps))) (assoc ps 0 out) ps)
                                                                 g need)))})
                       {:checkpoint? true}))
          "a wrong recompute input: a block's output for its input")
      (is (< 1e-4 (get (off {#'tape/recompute-vjp (fn [inner f ps g need] (recompute inner f ps (array/ones-like g) need))})
                       {:checkpoint? true}))
          "a cotangent dropped: the recompute seeded with ones")
      (is (< 1e-4 (get (off {#'nn/chunk-bounds (fn [v step] (mapv (fn [[s e]] [s (dec e)]) (bounds v step)))})
                       {:chunk 48}))
          "a chunk boundary off by one"))))

;; ---------------------------------------------------------------------------
;; 4. MLX against the JVM backend's tape
;; ---------------------------------------------------------------------------

(def ^:private mlx-band
  "MLX's gradients against the JVM backend's float64 tape, relative to
   the largest gradient: four times the largest measured (2.6e-7, Q8_0
   checkpointed and chunked on the MLX CPU's tape)."
  1.1e-6)

(deftest mlx-gradients-are-the-tape-s
  (if (ts/jvm?)
    (println "gen.llm.models.qwen3-train-test: MLX's gradients are compared with the JVM backend's on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [[k file] files
              opts [nil training]]
        (let [mine (value-and-grads k opts)
              theirs (array/with-backend jvm
                       (let [m (qwen3/load file)
                             paths (trained k)]
                         (host (apply (array/value-and-grad (loss-of m paths opts) {:argnums (vec (range (count paths)))})
                                      (map #(get-in (:weights m) %) paths)))))]
          (is (<= (gap mine theirs) mlx-band) (str (name k) " " opts (when (ts/native-grad?) ", native"))))))))

;; ---------------------------------------------------------------------------
;; 5. Frozen weights, and misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [m (:q8_0 @models)
        reason (fn [f] (:reason (ts/thrown-data (f))))]
    (when-not (ts/native-grad?)
      (is (= :no-gradient (reason #((array/grad (fn [s] (qwen3/nll (assoc-in m [:weights :layers 0 :q :scales] s) tokens)))
                                    (get-in m [:weights :layers 0 :q :scales]))))
          "a quantized matrix is frozen on the tape"))
    (is (= :invalid-args (reason #(qwen3/log-probs m [[5]]))) "one token: nothing to predict")
    (is (= :unknown-options (reason #(qwen3/log-probs m tokens {:checkpoint true}))))
    (is (= :invalid-args (reason #(qwen3/log-probs m tokens {:temperature 0}))))
    (is (= :unknown-options (reason #(qwen3/nll m tokens {:weights @mask}))))
    (is (= :invalid-args (reason #(qwen3/nll m tokens {:mask (array/ones [2 9])}))) "a mask of the tokens' shape")
    (is (= :invalid-token (reason #(qwen3/log-probs m [[1 128]]))))))
