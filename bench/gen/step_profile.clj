(ns gen.step-profile
  "Where the host's time goes in one compiled decode step of a Qwen3 or
   Qwen3.5 on MLX, in generation (each step's rows read on the host and the
   next tokens chosen there), as gen.llm.array-engine runs it (genclj-c7uf):

     clojure -J-Dgen.backend=mlx -M:bench -m gen.step-profile <Qwen3 or Qwen3.5 GGUF>

   One GPU JVM at a time. After gen.bench.prompt tokens (64) read into a
   static cache of gen.bench.batch rows (1), gen.bench.steps steps (256,
   after gen.bench.warm untimed ones, 32) each:
   - call: the compiled step's call (compile-step's fn), its graph built,
     not evaluated; broken down into the host's own work (wrapper: the
     token array, compile's argument trees, the step's checks) and
     mlx_compile's call (gen.backend.mlx.transforms/compile-fn, timed by
     wrapping it). With gen.bench.parts=true the profile's own copy of
     compile-fn (as of genclj-c7uf) splits that call: in (the input
     vector), apply (MLX's closure call: its trace replayed into a new
     graph), donate (the consumed inputs' handles released), out (the
     outputs wrapped);
   - encode: mlx_async_eval of the step's outputs (MLX's graph walked and
     its kernels encoded and committed; it grows with the batch, 2.8 ms at
     1 row and 10.5 at 16 for Qwen3-0.6B on the M4, so it likely also
     waits for the device, MLX bounding the work in flight);
     gen.bench.split=false leaves it out (array/eval! alone);
   - wait: array/eval! after it, the device's remaining time;
   - read: array/->floats of the logits;
   - normalize: each row normalized on the host as the engine does
     (gen.llm.host/log-softmax-rows, on a pool of gen.llm.host's default
     size and the calling thread; before genclj-9y6y the engine's own);
   - draw: each row's argmax on the host, the next tokens.
   Also: the JVM's GC (collections and time) and the MLX backend's own
   collections (System.gc under array or memory pressure,
   MlxRuntime/collect) during the timed steps, and the device's floor: the
   same steps with fixed tokens, evaluated every 16 (no host round trip).
   gen.bench.dtype (float32) as gen.qwen3-bench's."
  (:require [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.array-engine]
            [gen.llm.host :as host]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35])
  (:import [java.lang.management GarbageCollectorMXBean ManagementFactory]
           [java.util LinkedHashMap]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:prompt (long-prop "prompt" 64)
   :batch (long-prop "batch" 1)
   :steps (long-prop "steps" 256)
   :warm (long-prop "warm" 32)
   :dtype (keyword (prop "dtype" "float32"))})

;; ---------------------------------------------------------------------------
;; Spans
;; ---------------------------------------------------------------------------

(def ^:private ^LinkedHashMap totals (LinkedHashMap.))

(defn- add! [k ^long ns]
  (let [^longs a (or (.get totals k) (let [a (long-array 2)] (.put totals k a) a))]
    (aset a 0 (+ (aget a 0) ns))
    (aset a 1 (inc (aget a 1)))))

(defmacro ^:private span [k & body]
  `(let [t0# (System/nanoTime)]
     (try ~@body (finally (add! ~k (- (System/nanoTime) t0#))))))

(defn- gc-totals
  "[collections, ms] of the JVM's collectors so far."
  []
  (reduce (fn [[n t] ^GarbageCollectorMXBean g] [(+ (long n) (.getCollectionCount g)) (+ (long t) (.getCollectionTime g))])
          [0 0] (ManagementFactory/getGarbageCollectorMXBeans)))

(defn- mlx-collections ^long [] (aget (gen.mlx.MlxRuntime/lifetimeStats) 1))

;; ---------------------------------------------------------------------------
;; The compiled call, timed: the backend's p/-compile wrapped
;; ---------------------------------------------------------------------------

(def ^:private timing? (volatile! false))

(defmacro ^:private timed [k & body]
  `(if @timing? (span ~k ~@body) (do ~@body)))

(def ^:private settle-fresh
  ;; MlxRuntime/settleFresh, absent before genclj-c7uf: found reflectively,
  ;; so that the profile also runs (whole calls timed) on a binding without it
  (try (let [m (.getMethod gen.mlx.MlxRuntime "settleFresh" (make-array Class 0))]
         #(.invoke m nil (object-array 0)))
       (catch NoSuchMethodException _ (constantly nil))))

(defn- instrumented-compile-fn
  "gen.backend.mlx.transforms/compile-fn as of genclj-c7uf, its call's
   parts timed: :mlx-in (the input vector), :mlx-apply (MLX's call, the
   output handles read), :mlx-donate (the consumed inputs released, their
   bytes credited), :mlx-out (the outputs wrapped)."
  [b f {:keys [shapeless? donate]}]
  (let [wrap (requiring-resolve 'gen.backend.mlx.array/wrap)
        wrap-all (requiring-resolve 'gen.backend.mlx.array/wrap-all)
        handle (requiring-resolve 'gen.backend.mlx.array/handle)
        fence (requiring-resolve 'gen.backend.mlx.array/fence)
        retained (requiring-resolve 'gen.backend.mlx.operands/retained)
        donate! (requiring-resolve 'gen.backend.mlx.transforms/donate!)
        tracing (requiring-resolve 'gen.backend.mlx.transforms/*tracing-compile*)
        run (fn [^longs ins]
              (let [outs (with-bindings {tracing true} (f (mapv #(wrap b %) ins)))]
                (long-array (map #(retained b :compile %) outs))))
        cl (gen.mlx.MlxRuntime/closure (reify gen.mlx.MlxRuntime$Closure (apply [_ ins] (run ins))))
        ^gen.mlx.MlxRuntime$Compiled c (try (gen.mlx.MlxRuntime$Compiled. cl (boolean shapeless?))
                                            (finally (gen.mlx.MlxRaw/mlx_closure_free cl)))
        compiled (.-handle c)
        g (fn [xs]
            (timed :mlx-call
              (let [in (timed :mlx-in (gen.mlx.MlxRuntime/vectorOf (long-array (map #(handle %) xs))))
                    hs (try
                         (.enter c)
                         (timed :mlx-apply
                           (let [out (try (gen.mlx.Mlx/mlx_closure_apply compiled in) (finally (.exit c)))]
                             (try (gen.mlx.MlxRuntime/vectorArrays out)
                                  (finally (gen.mlx.MlxRaw/mlx_vector_array_free out)))))
                         (finally
                           (gen.mlx.MlxRaw/mlx_vector_array_free in)
                           (fence xs)))]
                (timed :mlx-donate (doseq [i donate] (donate! (nth xs i))))
                (timed :mlx-out (try (wrap-all b hs) (finally (settle-fresh)))))))]
    (gen.mlx.MlxRuntime/adoptCompiled g c)
    g))

(defn- timed-compile
  "compile-fn replaced: the profile's own copy with its parts timed
   (gen.bench.parts=true), or the backend's with its call timed whole."
  [compile]
  (if (= "true" (prop "parts" "false"))
    instrumented-compile-fn
    (fn [b f opts]
      (let [g (compile b f opts)]
        (fn [xs] (timed :mlx-call (g xs)))))))

;; ---------------------------------------------------------------------------
;; The model's family
;; ---------------------------------------------------------------------------

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- load-model [path opts]
  (if (#{"qwen35" "qwen35moe"} (get-in (gguf/header path) [:metadata "general.architecture"]))
    (qwen35/load path opts)
    (qwen3/load path opts)))

(defn- prefill [m & args] (apply (if (qwen35? m) qwen35/prefill qwen3/prefill) m args))
(defn- compile-step [m opts] ((if (qwen35? m) qwen35/compile-step qwen3/compile-step) m opts))
(defn- select-rows [m cache idx] ((if (qwen35? m) qwen35/select-rows qwen3/select-rows) cache idx))

(defn- prompt-tokens [^long n] (mapv #(+ 1000 (mod (* 7919 (long %)) 20000)) (range n)))

;; ---------------------------------------------------------------------------
;; Generation, step by step
;; ---------------------------------------------------------------------------

;; The engine's normalization as of the commit measured: gen.llm.host's
;; log-softmax-rows (genclj-9y6y), else the engine's own (normalized, and
;; normalized-in-parts from genclj-c7uf), found by name so that the
;; profile also runs on the commits before.
(def ^:private log-softmax-rows (some-> (resolve 'gen.llm.host/log-softmax-rows) deref))
(def ^:private normalized (some-> (resolve 'gen.llm.array-engine/normalized) deref))
(def ^:private normalized-in-parts (some-> (resolve 'gen.llm.array-engine/normalized-in-parts) deref))

(defn- normalize
  "The rows of xs (n floats each) normalized as gen.llm.array-engine's
   host-rows does."
  [pool xs ^long rows ^long n]
  (let [threads (host/default-threads)]
    (cond log-softmax-rows (log-softmax-rows pool xs n "profile")
          (and normalized-in-parts (< rows threads)) (mapv #(normalized-in-parts pool (* 4 threads) xs % n "profile") (range rows))
          :else (host/in-order pool #(normalized xs % n "profile") (vec (range rows))))))

(defn- argmax ^long [^floats xs]
  (let [n (alength xs)]
    (loop [j 1 best 0 v (aget xs 0)]
      (if (< j n)
        (let [x (aget xs j)] (if (> x v) (recur (inc j) j x) (recur (inc j) best v)))
        best))))

(defn- leaves [x] (filterv #(and (array/array? %) (not (array/host? %))) (array/tree-leaves x)))

(defn- generate
  "steps steps from cache by step, each step's rows read, normalized and
   their argmaxes the next tokens; the spans recorded when timed?."
  [step cache pool rows steps timed?]
  (vreset! timing? timed?)
  (let [split? (= "true" (prop "split" "true"))
        handle (requiring-resolve 'gen.backend.mlx.array/handle)
        async (fn [xs] (let [v (gen.mlx.MlxRuntime/vectorOf (long-array (map #(handle %) xs)))]
                         (try (gen.mlx.Mlx/mlx_async_eval v) (finally (gen.mlx.MlxRaw/mlx_vector_array_free v)))))]
    (loop [i 0 c cache toks (mapv #(vector (+ 13 (long %))) (range rows))]
      (if (< i steps)
        (let [t0 (System/nanoTime)
              [l c'] (timed :call (step c toks))
              _ (when split? (timed :encode (async (leaves [l c']))))
              _ (timed :wait (array/eval! [l c']))
              xs (timed :read (array/->floats l))
              n (long (peek (array/shape l)))
              rs (timed :normalize (normalize pool xs rows n))
              next (timed :draw (mapv #(vector (argmax %)) rs))]
          (when timed? (add! :step (- (System/nanoTime) t0)))
          (recur (inc i) c' next))
        (do (vreset! timing? false) c)))))

(defn- floor-ms
  "ms a step of fixed tokens evaluated every 16 steps: the device's time."
  [step cache ^long rows ^long steps]
  (let [t0 (System/nanoTime)]
    (loop [i 0 c cache]
      (if (< i steps)
        (let [[l c'] (step c (vec (repeat rows [(+ 1000 i)])))]
          (when (zero? (mod (inc i) 16)) (array/eval! [l c']))
          (recur (inc i) c'))
        (array/eval! c)))
    (/ (- (System/nanoTime) t0) 1e6 steps)))

(defn- report [^long steps]
  (let [ms (fn [k] (if-let [^longs a (.get totals k)] (/ (aget a 0) 1e6 steps) 0.0))
        call (ms :call)
        mlx (ms :mlx-call)
        parts? (.containsKey totals :mlx-apply)
        rows (concat
              [["call (the step's graph built)" call]
               ["  host wrapper (tokens, compile's trees, checks)" (- call mlx)]
               ["  mlx_compile's call (inputs, replay, outputs, donation)" mlx]]
              (when parts?
                [["    in (the input vector)" (ms :mlx-in)]
                 ["    apply (MLX's call: the trace replayed)" (ms :mlx-apply)]
                 ["    donate (consumed inputs released)" (ms :mlx-donate)]
                 ["    out (outputs wrapped)" (ms :mlx-out)]])
              [["encode (mlx_async_eval: graph walked, kernels encoded)" (ms :encode)]
              ["wait (the device's remaining time)" (ms :wait)]
              ["read (->floats)" (ms :read)]
              ["normalize (host, in double)" (ms :normalize)]
              ["draw (argmax on the host)" (ms :draw)]
              ["step" (ms :step)]])]
    (println (format "| %-58s | %8s |" "per step" "ms"))
    (println "|---|---:|")
    (doseq [[label v] rows] (println (format "| %-58s | %8.3f |" label (double v))))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (let [{:keys [prompt batch steps warm dtype]} settings
        m (load-model path {:dtype dtype})
        capacity (* 128 (quot (+ prompt warm steps 127) 128))
        _ (array/eval! (:weights m))
        [_ c0] (prefill m [(prompt-tokens prompt)] {:logits :last :capacity capacity})
        c0 (array/eval! (select-rows m c0 (repeat batch 0)))
        pool (host/pool "profile rows-" (host/default-threads))
        b array/*backend*]
    (println "## One compiled decode step's host time\n")
    (println "- model:" path)
    (println "- settings:" (pr-str (assoc settings :capacity capacity)))
    (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
    (println "- MLX device:" ((requiring-resolve 'gen.backend.mlx.array/device) b) "\n")
    (alter-var-root (requiring-resolve 'gen.backend.mlx.transforms/compile-fn) timed-compile)
    (let [step (compile-step m {:logits :last :donate? true})
          warm-c (generate step (array/eval! (select-rows m c0 (range batch))) pool batch warm false)
          [gc0 gt0] (gc-totals)
          mc0 (mlx-collections)
          t0 (System/nanoTime)
          _ (generate step warm-c pool batch steps true)
          wall (/ (- (System/nanoTime) t0) 1e6)
          [gc1 gt1] (gc-totals)
          mc1 (mlx-collections)]
      (report steps)
      (println (format "\n- %d steps at batch %d: %.1f ms, %.1f tokens/s (rows read and drawn on the host)"
                       steps batch wall (/ (* 1e3 batch steps) wall)))
      (println (format "- JVM GC: %d collections, %d ms; the MLX backend's collections (System.gc): %d"
                       (- (long gc1) (long gc0)) (- (long gt1) (long gt0)) (- mc1 mc0)))
      (let [f (floor-ms step (array/eval! (select-rows m c0 (range batch))) batch steps)]
        (println (format "- the device's floor (fixed tokens, evaluated every 16 steps): %.3f ms a step, %.1f tokens/s"
                         f (/ (* 1e3 batch) f))))
      (println "- MLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory)))))
    (.shutdown ^java.util.concurrent.ExecutorService pool)
    (shutdown-agents)))
