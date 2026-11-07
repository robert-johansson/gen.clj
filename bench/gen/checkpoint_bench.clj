(ns gen.checkpoint-bench
  "Peak memory of a gradient on MLX, its layers checkpointed or not
   (gen.array/checkpoint, genclj-pnu6; Qwen3's training loss,
   genclj-jksq):

     clojure -J-Dgen.backend=mlx -J-Dgen.mlx.grad=native -M:bench -m gen.checkpoint-bench [<Qwen3 GGUF>]

   One GPU JVM at a time. Each row builds a gradient's graph (value-and-
   grad, lazy), collects the JVM's dead arrays (their handles would pin
   MLX's buffers through the evaluation: the graph alone is measured),
   resets MLX's peak memory, evaluates the value and the gradients, and
   reads the peak less the memory held before (the weights): bytes MLX
   held at once. The peak is measured with MLX's memory limit at what it
   held before, so that its eval loop waits for queued kernels whenever
   it holds more and their buffers are freed as the graph allows (the
   graph's own peak); the time, and the queued peak, at the configured
   limit (gen.mlx.memory-bytes), where kernels MLX has queued but not
   yet run keep their buffers. Three ways for each:
   - plain: no checkpoint, every activation kept from the forward pass;
   - ordered: array/checkpoint per layer, its recompute ordered after the
     layer above's gradient (depends on the cotangents, mlx-node's);
   - mlx: array/checkpoint per layer by MLX's own mlx_checkpoint (its
     recompute depends on the outputs only).
   MLX 0.32 builds its evaluation tape breadth first, MLX_BFS_MAX_WIDTH
   (20) wide; 1 makes it depth first, as older MLX's (mlx-node's
   comment assumes that order).
   Sections (gen.bench.sections, default stack,qwen3):
   - stack: gen.bench.layers (4,16) pre-norm residual MLP layers of width
     gen.bench.width (1024, hidden 4x) over gen.bench.tokens rows (2048),
     float32, the gradient with respect to every weight;
   - qwen3 (with a GGUF): the nll of gen.bench.rows (4) rows of
     gen.bench.length (256) tokens, its gradient with respect to every
     dense weight (a Q8_0 model's norms), its output projection in
     vocabulary chunks of gen.bench.chunk (16384), checkpointed with the
     blocks.
   Times are one evaluation's, after one untimed run."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.backend.mlx.memory :as memory]
            [gen.backend.mlx.transforms :as transforms]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.random :as rng])
  (:import [gen.mlx Mlx]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private sections (set (map keyword (str/split (prop "sections" "stack,qwen3") #","))))

(defn- mb [bytes] (format "%,.0f MB" (/ (double bytes) 1e6)))

(defn- measure
  "{:peak :ms :out}: the peak bytes MLX held, past those held before,
   while evaluating (build), a value-and-grad's lazy result, and the time
   it took. With throttle? MLX's memory limit is what it held before, so
   that it waits for its queued kernels whenever it holds more (its
   eval loop's own rule): its buffers are then freed as the graph allows,
   and the peak is the graph's, not the queue's. Unthrottled, the kernels
   MLX has queued but not run keep their buffers, up to the limit."
  [build throttle?]
  (memory/collect!)
  (Mlx/mlx_clear_cache)
  (let [before (Mlx/mlx_get_active_memory)
        out (build)
        limit (Mlx/mlx_get_memory_limit)]
    (memory/collect!)
    (Mlx/mlx_reset_peak_memory)
    (when throttle? (memory/set-memory-limit! before))
    (try
      (let [t0 (System/nanoTime)]
        (array/eval! out)
        (let [ms (/ (- (System/nanoTime) t0) 1e6)
              peak (- (Mlx/mlx_get_peak_memory) before)]
          {:peak peak :ms ms :out out}))
      (finally (memory/set-memory-limit! limit)))))

(defn- rows
  "Each way's measurement of (build checkpoint-of), checkpoint-of turning
   a layer function into the way's: an untimed run, a timed one at MLX's
   memory limit, and one throttled for the graph's peak."
  [label build]
  (let [ways [["plain" identity nil]
              ["ordered" array/checkpoint :ordered]
              ["mlx" array/checkpoint :plain]]
        results (vec (for [[way ck order] ways]
                       (binding [transforms/*checkpoint-order* (or order :ordered)]
                         (measure #(build ck) false)
                         (let [free (measure #(build ck) false)
                               throttled (measure #(build ck) true)]
                           [way (assoc throttled :ms (:ms free) :queued (:peak free) :throttled-ms (:ms throttled))]))))
        [_ plain] (first results)]
    (println (str "\n" label))
    (println (format "  %-8s %12s %10s %14s %12s %14s" "way" "peak" "ms" "max |Δ grad|" "queued peak" "throttled ms"))
    (doseq [[way {:keys [peak ms out queued throttled-ms]}] results]
      (let [d (reduce max 0.0 (map (fn [a b] (double (array/item (array/max (array/abs (array/- a b))))))
                                   (array/tree-leaves out) (array/tree-leaves (:out plain))))]
        (println (format "  %-8s %12s %10.1f %14.3g %12s %14.1f" way (mb peak) ms d (mb queued) throttled-ms))))))

;; ---------------------------------------------------------------------------
;; A stack of residual MLP layers
;; ---------------------------------------------------------------------------

(defn- mlp-layer
  "x [n d] + (silu(x'·w1) ⊙ (x'·w2))·w3, x' = rms-norm(x): a pre-norm
   transformer's SwiGLU block."
  [x {:keys [w1 w2 w3]}]
  (let [xn (array/rms-norm x nil 1e-5)
        a (array/matmul xn w1)]
    (array/+ x (array/matmul (array/* (array/* a (array/sigmoid a)) (array/matmul xn w2)) w3))))

(defn- stack-of [^long layers]
  (let [n (long-prop "tokens" 2048)
        d (long-prop "width" 1024)
        h (* 4 d)
        k (rng/key 7)
        scale (fn [x s] (array/* x (/ 1.0 (Math/sqrt (double s)))))
        ws (vec (for [i (range layers)]
                  (let [[k1 k2 k3] (rng/split (rng/fold-in k i) 3)]
                    {:w1 (scale (rng/normal k1 [d h]) d)
                     :w2 (scale (rng/normal k2 [d h]) d)
                     :w3 (scale (rng/normal k3 [h d]) (* 4 h))})))
        x (rng/normal (rng/key 8) [n d])
        _ (array/eval! [ws x])
        build (fn [ck]
                ((array/value-and-grad (fn [ws] (array/mean (array/square (reduce (ck mlp-layer) x ws))))) ws))]
    (rows (format "stack: %d layers, %d tokens, width %d (hidden %d), float32; weights %s"
                  layers n d h (mb (* 4 layers 3 d h)))
          build)))

(defn- stack-section []
  (doseq [layers (map parse-long (str/split (prop "layers" "4,16") #","))]
    (stack-of layers)))

;; ---------------------------------------------------------------------------
;; Qwen3's training loss
;; ---------------------------------------------------------------------------

(defn- dense-paths
  "The paths of the model's dense float weights in :weights (a Q8_0
   model's norms; every weight of an F32 one)."
  [weights]
  (letfn [(walk [path x]
            (cond (and (map? x) (contains? x :wq)) []
                  (map? x) (mapcat (fn [[k v]] (walk (conj path k) v)) x)
                  (vector? x) (mapcat (fn [i v] (walk (conj path i) v)) (range) x)
                  (and (array/array? x) (= :float (array/dtype-kind (array/dtype x)))) [path]
                  :else []))]
    (vec (walk [] weights))))

(defn- qwen3-section [path]
  (let [m (qwen3/load path)
        b (long-prop "rows" 4)
        t (long-prop "length" 256)
        chunk (long-prop "chunk" 16384)
        vocab (:vocab (:config m))
        tokens (vec (for [r (range b)] (vec (for [i (range t)] (mod (+ 7 (* 131 r) (* 977 i)) vocab)))))
        paths (dense-paths (:weights m))
        ws (mapv #(get-in (:weights m) %) paths)
        _ (array/eval! (:weights m))
        build (fn [ck]
                ((array/value-and-grad (fn [ws]
                                         (qwen3/nll (update m :weights #(reduce (fn [w [p x]] (assoc-in w p x)) % (map vector paths ws)))
                                                    tokens {:checkpoint? (not= ck identity) :chunk chunk})))
                 ws))]
    (rows (format "qwen3 %s: %d rows of %d tokens, vocabulary chunks of %d, %d dense weights differentiated"
                  (last (str/split path #"/")) b t chunk (count paths))
          build)))

(defn -main [& [gguf]]
  (println "gen.checkpoint-bench:" (System/getProperty "gen.mlx.grad" "tape") "gradients,"
           ((requiring-resolve 'gen.backend.mlx/device) array/*backend*))
  (when (sections :stack) (stack-section))
  (when (and gguf (sections :qwen3)) (qwen3-section gguf))
  (shutdown-agents))
