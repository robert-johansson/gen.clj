(ns gen.read-profile
  "Where a decode step's read of its logits (gen.array/->floats) spends its
   time on MLX, the read alone of gen.step-profile's (genclj-9y6y):

     clojure -J-Dgen.backend=mlx -M:bench -m gen.read-profile <Qwen3 or Qwen3.5 GGUF>

   One GPU JVM at a time. After gen.bench.prompt tokens (64) read into a
   static cache of gen.bench.batch rows (16) and gen.bench.warm steps (16),
   gen.bench.steps steps (32), each step's outputs evaluated first, then
   its logits read as the engine does and, apart, in parts:
   - the logits' dtype, shape, row-contiguity and strides;
   - astype: the logits cast to float32 on the device and evaluated (a
     float32 model's are themselves);
   - pointer: mlx_array_data_float32, which on CUDA moves a device buffer
     to unified memory (MLX's Buffer::raw_ptr: a cudaMallocManaged, a
     cudaMemcpy, the device buffer freed);
   - copy: the bytes copied into a float array (MemorySegment.toArray);
   - read: array/->floats of the logits, as the engine reads them (on a
     step of its own: an array already moved would read faster).
   gen.bench.dtype (float16) as gen.qwen3-bench's."
  (:require [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35])
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [java.util LinkedHashMap]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:prompt (long-prop "prompt" 64)
   :batch (long-prop "batch" 16)
   :steps (long-prop "steps" 32)
   :warm (long-prop "warm" 16)
   :dtype (keyword (prop "dtype" "float16"))})

(def ^:private ^LinkedHashMap totals (LinkedHashMap.))

(defmacro ^:private span [k & body]
  `(let [t0# (System/nanoTime)]
     (try ~@body (finally (let [^"[J" a# (or (.get totals ~k) (let [a# (long-array 1)] (.put totals ~k a#) a#))]
                            (aset a# 0 (+ (aget a# 0) (- (System/nanoTime) t0#))))))))

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- load-model [path opts]
  (if (#{"qwen35" "qwen35moe"} (get-in (gguf/header path) [:metadata "general.architecture"]))
    (qwen35/load path opts)
    (qwen3/load path opts)))

(defn- prefill [m & args] (apply (if (qwen35? m) qwen35/prefill qwen3/prefill) m args))
(defn- compile-step [m opts] ((if (qwen35? m) qwen35/compile-step qwen3/compile-step) m opts))
(defn- select-rows [m cache idx] ((if (qwen35? m) qwen35/select-rows qwen3/select-rows) cache idx))

(defn- argmax ^long [^floats xs ^long off ^long n]
  (loop [j 1 best 0 v (aget xs off)]
    (if (< j n)
      (let [x (aget xs (+ off j))] (if (> x v) (recur (inc j) j x) (recur (inc j) best v)))
      best)))

(defn- layout
  "dtype, shape, row-contiguous?, strides of x's MLX array."
  [x]
  (let [h (long ((requiring-resolve 'gen.backend.mlx.array/handle) x))
        nd (gen.mlx.Mlx/mlx_array_ndim h)
        ^MemorySegment st (.reinterpret ^MemorySegment (gen.mlx.Mlx/mlx_array_strides h) (* 8 nd))]
    {:dtype (array/dtype x) :shape (array/shape x)
     :row-contiguous? (gen.mlx.Mlx/_mlx_array_is_row_contiguous h)
     :strides (vec (.toArray st ValueLayout/JAVA_LONG))}))

(def ^:private modes
  "How a mode reads the logits l: a float array, its parts timed under
   [mode part]."
  {:engine (fn [l] (span [:engine :read] (array/->floats l)))
   ;; the engine's read in parts: the cast on the device, its pointer
   ;; (which moves it to unified memory on CUDA), the bytes copied, the
   ;; cast freed at once (as ->floats frees it)
   :freed (fn [l]
            (let [h (long ((requiring-resolve 'gen.backend.mlx.array/handle) l))
                  s (long ((requiring-resolve 'gen.backend.mlx.array/stream) array/*backend*))
                  n (long (reduce * (array/shape l)))
                  f (span [:freed :astype] (let [f (gen.mlx.Mlx/mlx_astype h (int 10) s)] (gen.mlx.Mlx/mlx_array_eval f) f))]
              (try
                (let [^MemorySegment p (span [:freed :pointer] (gen.mlx.Mlx/mlx_array_data_float32 f))]
                  (span [:freed :copy] (.toArray (.reinterpret p (* 4 n)) ValueLayout/JAVA_FLOAT)))
                (finally (span [:freed :free] (gen.mlx.MlxRaw/mlx_array_free f))))))
   ;; the same, the cast a gen.array array, freed when the GC finds it
   :kept (fn [l]
           (let [f (span [:kept :astype] (let [f (array/astype l :float32)] (array/eval! f) f))
                 h (long ((requiring-resolve 'gen.backend.mlx.array/handle) f))
                 n (long (reduce * (array/shape f)))
                 ^MemorySegment p (span [:kept :pointer] (gen.mlx.Mlx/mlx_array_data_float32 h))]
             (span [:kept :copy] (.toArray (.reinterpret p (* 4 n)) ValueLayout/JAVA_FLOAT))))
   ;; the float16 logits themselves read (their bits, by mlx-c's uint16
   ;; accessor: its float16 one is absent where the compiler lacks
   ;; _Float16), widened on the host
   :f16 (fn [l]
          (let [h (long ((requiring-resolve 'gen.backend.mlx.array/handle) l))
                n (long (reduce * (array/shape l)))
                ^MemorySegment p (span [:f16 :pointer] (gen.mlx.Mlx/mlx_array_data_uint16 h))
                ^shorts hs (span [:f16 :copy] (.toArray (.reinterpret p (* 2 n)) ValueLayout/JAVA_SHORT))]
            (span [:f16 :widen] (let [out (float-array n)] (dotimes [j n] (aset out j (Float/float16ToFloat (aget hs j)))) out))))})

(defn- go
  "n steps of step from cache c, each read by mode."
  [step c toks n batch mode]
  (let [read (modes mode)]
    (loop [i 0 c c toks toks]
      (if (< i n)
        (let [[l c'] (step c toks)
              _ (span [mode :eval] (array/eval! [l c']))
              _ (when (and (zero? i) (= mode :engine)) (println "- logits:" (pr-str (layout l))))
              ^floats xs (read l)
              v (long (peek (array/shape l)))]
          (recur (inc i) c' (mapv #(vector (argmax xs (* (long %) v) v)) (range batch))))
        c))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (let [{:keys [prompt batch steps warm dtype]} settings
        m (load-model path {:dtype dtype})
        order [:engine :freed :kept :f16 :engine]
        capacity (* 128 (quot (+ prompt warm (* (count order) steps) 127) 128))
        _ (array/eval! (:weights m))
        [_ c0] (prefill m [(mapv #(mod (+ 1000 (* 7919 (long %))) (long (get-in m [:config :vocab]))) (range prompt))] {:logits :last :capacity capacity})
        c0 (array/eval! (select-rows m c0 (repeat batch 0)))
        step (compile-step m {:logits :last :donate? true})
        toks (mapv #(vector (mod (+ 13 (long %)) (long (get-in m [:config :vocab])))) (range batch))
        c (go step c0 toks warm batch :engine)
        _ (.clear totals)
        c (reduce (fn [c mode] (go step c toks steps batch mode)) c order)]
    (array/eval! c)
    (println "## The logits' read on MLX\n")
    (println "- model:" path)
    (println "- settings:" (pr-str settings) "modes in order:" order)
    (println "- MLX device:" ((requiring-resolve 'gen.backend.mlx.array/device) array/*backend*) "\n")
    (println (format "| %-44s | %8s |" "per step (ms)" "ms"))
    (println "|---|---:|")
    (doseq [[k ^"[J" a] totals
            :let [runs (if (= :engine (first k)) 2 1)]]
      (println (format "| %-44s | %8.3f |" (pr-str k) (/ (aget a 0) 1e6 steps runs))))
    (println "- MLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory)))))
  (shutdown-agents))
