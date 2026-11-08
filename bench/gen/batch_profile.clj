(ns gen.batch-profile
  "Where the time of one compiled batched decode step goes (genclj-qu77):
   the family's compile-paged-batch-step (the array engine's continuous
   mode) of Qwen3 or Qwen3.5 on MLX, at several row counts:

     clojure -J-Dgen.backend=mlx -M:bench -m gen.batch-profile <Qwen3 or Qwen3.5 GGUF>

   One GPU JVM at a time. Each of gen.bench.rows (1,8,16) decode rows sits
   gen.bench.prompt (64) positions into its own sequence of paged blocks
   (block size 16, a table of 8 blocks a row), and gen.bench.steps (48)
   steps (after gen.bench.warm, 8) read a token a row, each step's rows
   read on the host, normalized (gen.llm.host/log-softmax-rows) and their
   argmaxes the next tokens, as the engine does. A Qwen3.5's GatedDeltaNet
   states are given as the continuous mode gives them
   (gen.bench.states, lane,held):
   - lane: in layers, consumed (donated) by the step, which writes them
     in place;
   - held: in held, read but not consumed (a group whose rows forks still
     read: most of the continuous mode's forwards), the step's states new
     arrays, the last step's dropped (freed by the GC);
   - held-freed: the same, the last step's states released at the next
     step (array/release!), as the engine releases a group no branch
     holds (genclj-qu77).
   Per step, ms: call (the compiled call: the host's wrapper, then
   mlx_compile's call), encode (mlx_async_eval: the graph walked and its
   kernels encoded), wait (the device's remaining time), read, normalize,
   draw; the traces (calls of the family's batch-forward: a trace per new
   structure or shape), the JVM's GC, and the MLX backend's collections
   (System.gc). Then the device's floor (fixed tokens, evaluated every 8
   steps: no host round trip), and with gen.bench.parts=true each part
   of the forward compiled alone at each row count, in the same floor's
   way: the embedding, every GatedDeltaNet layer's mixer (its projections,
   convolution, gated-delta and output) and gated-delta alone, every
   attention layer's mixer, every feed-forward, the head (norm and output
   projection). gen.bench.dtype (float32).
   Measured on the Mac mini M4 (2026-11-08, Qwen3.5-0.8B Q8_0, ms a
   step at 1, 8 and 16 rows), before genclj-qu77: float32 lane 12.8,
   38.0, 34.4 (8 rows slower than 16: MLX's qmv_wide below its qmv batch
   limit, gen.backend.mlx.nn/qmm-rows), held 13.0, 39.6, 40.8 (the GC);
   float16 lane 12.4, 20.1, 32.7, held 12.6, 23.1, 38.7. After: float32
   lane 8 rows 28.7; float16 held-freed 20.2 at 8 rows, 32.8 at 16.
   gated-delta runs at the memory's speed: 18 layers consuming their
   states take 0.44 ms a row (each state read and written once, 86 GB/s);
   not consuming them (gen.bench.parts' row), 0.71, in new buffers."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.host :as host]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng])
  (:import [java.lang.management GarbageCollectorMXBean ManagementFactory]
           [java.util LinkedHashMap]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:rows (mapv parse-long (str/split (prop "rows" "1,8,16") #","))
   :prompt (long-prop "prompt" 64)
   :steps (long-prop "steps" 48)
   :warm (long-prop "warm" 8)
   :states (mapv keyword (str/split (prop "states" "lane,held") #","))
   :dtype (keyword (prop "dtype" "float32"))
   :parts? (= "true" (prop "parts" "false"))})

(def ^:private block-size 16)
(def ^:private width 8)

;; ---------------------------------------------------------------------------
;; Spans
;; ---------------------------------------------------------------------------

(def ^:private ^LinkedHashMap totals (LinkedHashMap.))

(defn- add! [k ^long ns]
  (let [^longs a (or (.get totals k) (let [a (long-array 2)] (.put totals k a) a))]
    (aset a 0 (+ (aget a 0) ns))
    (aset a 1 (inc (aget a 1)))))

(def ^:private timing? (volatile! false))

(defmacro ^:private timed [k & body]
  `(if @timing?
     (let [t0# (System/nanoTime)]
       (try ~@body (finally (add! ~k (- (System/nanoTime) t0#)))))
     (do ~@body)))

(defn- gc-totals []
  (reduce (fn [[n t] ^GarbageCollectorMXBean g] [(+ (long n) (.getCollectionCount g)) (+ (long t) (.getCollectionTime g))])
          [0 0] (ManagementFactory/getGarbageCollectorMXBeans)))

(defn- mlx-collections ^long [] (aget (gen.mlx.MlxRuntime/lifetimeStats) 1))

(def ^:private traces (atom 0))

(defn- timed-compile
  "gen.backend.mlx.transforms/compile-fn with its call timed (:mlx-call)."
  [compile]
  (fn [b f opts]
    (let [g (compile b f opts)]
      (fn [xs] (timed :mlx-call (g xs))))))

;; ---------------------------------------------------------------------------
;; The model's family
;; ---------------------------------------------------------------------------

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- load-model [path opts]
  (if (#{"qwen35" "qwen35moe"} (get-in (gguf/header path) [:metadata "general.architecture"]))
    (qwen35/load path opts)
    (qwen3/load path opts)))

(defn- i32 [xs] (array/array (vec xs) {:dtype :int32}))

(defn- batch
  "The batch of r decode rows at position pos, tokens toks (r of them):
   row i's blocks i·width …"
  [^long r ^long pos toks]
  (let [blk (fn [^long i] (+ (* i width) (quot pos block-size)))]
    {:tokens (mapv vector toks)
     :positions (i32 (repeat r pos))
     :slots (i32 (map #(+ (* block-size (long (blk %))) (rem pos block-size)) (range r)))
     :tables (i32 (map (fn [i] (range (* (long i) width) (* (inc (long i)) width))) (range r)))
     :out (i32 (range r))}))

(defn- leaves [x] (filterv #(and (array/array? %) (not (array/host? %))) (array/tree-leaves x)))

(defn- async! [x]
  (let [handle (requiring-resolve 'gen.backend.mlx.array/handle)
        v (gen.mlx.MlxRuntime/vectorOf (long-array (map #(handle %) (leaves x))))]
    (try (gen.mlx.Mlx/mlx_async_eval v) (finally (gen.mlx.MlxRaw/mlx_vector_array_free v)))))

(defn- argmax ^long [^floats xs]
  (let [n (alength xs)]
    (loop [j 1 best 0 v (aget xs 0)]
      (if (< j n)
        (let [x (aget xs j)] (if (> x v) (recur (inc j) j x) (recur (inc j) best v)))
        best))))

(defn- stepper
  "(f carry toks pos) → [logits carry'] for the model's compiled batch step
   in states mode (Qwen3.5: carry {:pools :states}; Qwen3: the pools)."
  [m mode]
  (if (qwen35? m)
    (let [f (qwen35/compile-paged-batch-step m {:donate? true})
          pools-of (fn [layers] (mapv #(select-keys % [:k :v]) layers))
          states-of (fn [layers] (mapv #(select-keys % [:conv :state]) layers))
          ;; :held-freed: the last step's held states released at the next
          ;; step, once the last step was evaluated
          dead (volatile! nil)]
      (fn [{:keys [pools states]} toks pos]
        (when-let [d @dead] (array/release! d) (vreset! dead nil))
        (let [r (count toks)
              b (batch r pos toks)
              {:keys [logits layers]} (case mode
                                        :lane (f (mapv merge pools states) b)
                                        :held (f pools b states)
                                        :held-freed (do (vreset! dead states) (f pools b states)))]
          [logits {:pools (pools-of layers) :states (states-of layers)}])))
    (let [f (qwen3/compile-paged-batch-step m {:donate? true})]
      (fn [pools toks pos]
        (let [{:keys [logits pools]} (f pools (batch (count toks) pos toks))]
          [logits pools])))))

(defn- start
  "The carry of r rows: empty pools, zero states."
  [m ^long r]
  (let [n (* (+ r 1) width)]
    (doto (if (qwen35? m)
            {:pools (mapv #(select-keys % [:k :v]) (qwen35/paged-pools m n block-size))
             :states (qwen35/paged-states m r)}
            (qwen3/paged-pools m n block-size))
      array/eval!)))

(defn- generate
  "steps steps of r rows from carry, each step's rows read, normalized
   and drawn; the spans recorded when timed?."
  [step carry pool r pos0 steps timed?]
  (vreset! timing? timed?)
  (loop [i 0 c carry toks (vec (repeat r 13))]
    (if (< i steps)
      (let [t0 (System/nanoTime)
            [l c'] (timed :call (step c toks (+ pos0 i)))
            _ (timed :encode (async! [l c']))
            _ (timed :wait (array/eval! [l c']))
            xs (timed :read (array/->floats l))
            n (long (peek (array/shape l)))
            rs (timed :normalize (host/log-softmax-rows pool xs n "profile"))
            next (timed :draw (mapv argmax rs))]
        (when timed? (add! :step (- (System/nanoTime) t0)))
        (recur (inc i) c' next))
      (do (vreset! timing? false) c))))

(defn- floor-ms
  "ms a step of fixed tokens, evaluated every 8 steps."
  [step carry r pos0 steps]
  (let [t0 (System/nanoTime)]
    (loop [i 0 c carry]
      (if (< i steps)
        (let [[l c'] (step c (vec (repeat r (+ 1000 i))) (+ pos0 i))]
          (when (zero? (mod (inc i) 8)) (array/eval! [l c']))
          (recur (inc i) c'))
        (array/eval! c)))
    (/ (- (System/nanoTime) t0) 1e6 steps)))

(defn- profile-row
  [m pool r mode]
  (let [{:keys [prompt steps warm]} settings
        step (stepper m mode)
        c (generate step (start m r) pool r prompt warm false)
        _ (.clear totals)
        tr0 @traces
        [gc0 gt0] (gc-totals)
        mc0 (mlx-collections)
        c (generate step c pool r (+ prompt warm) steps true)
        [gc1 gt1] (gc-totals)
        mc1 (mlx-collections)
        ms (fn [k] (if-let [^longs a (.get totals k)] (/ (aget a 0) 1e6 steps) 0.0))
        floor (floor-ms step c r (+ prompt warm steps) 32)]
    {:rows r :mode mode
     :host (- (ms :call) (ms :mlx-call)) :mlx-call (ms :mlx-call)
     :encode (ms :encode) :wait (ms :wait) :read (ms :read) :normalize (ms :normalize) :draw (ms :draw)
     :step (ms :step) :traces (- (long @traces) (long tr0))
     :gc (format "%d/%d ms" (- (long gc1) (long gc0)) (- (long gt1) (long gt0))) :mlx-gc (- mc1 mc0)
     :floor floor}))

;; ---------------------------------------------------------------------------
;; The forward's parts, each compiled alone
;; ---------------------------------------------------------------------------

(defn- part-ms
  "ms a call of the compiled f of args, evaluated every 8 calls, after 4."
  [f args]
  (dotimes [_ 4] (array/eval! (apply f args)))
  (let [n 32
        t0 (System/nanoTime)]
    (loop [i 0 outs []]
      (if (< i n)
        (let [o (apply f args)
              outs (conj outs o)]
          (if (zero? (mod (inc i) 8))
            (do (array/eval! outs) (recur (inc i) []))
            (recur (inc i) outs)))
        (/ (- (System/nanoTime) t0) 1e6 n)))))

(defn- parts-35
  "Qwen3.5's parts at r rows: {part ms}."
  [{:keys [config weights] :as m} ^long r]
  (let [{:keys [kinds hidden dtype eps]} config
        layers (:layers weights)
        delta-net @#'qwen35/batch-delta-net
        attention @#'qwen35/batch-attention
        feed-forward @#'qwen35/feed-forward
        x (array/astype (rng/normal (rng/key 1) [r 1 hidden]) dtype)
        ids (i32 (map vector (range 100 (+ 100 r))))
        b (batch r 80 (vec (repeat r 1)))
        at (assoc (select-keys b [:positions :slots :tables]) :d r)
        states (qwen35/paged-states m r)
        pools (mapv #(select-keys % [:k :v]) (qwen35/paged-pools m (* (inc r) width) block-size))
        di (filterv #(= :delta (kinds %)) (range (count kinds)))
        ai (filterv #(= :attention (kinds %)) (range (count kinds)))
        {:keys [delta-value-heads delta-key-dim delta-value-dim]} config
        [k1 k2 k3 k4 k5] (rng/split (rng/key 2) 5)
        sd (if (#{:float16 :bfloat16} dtype) :float32 dtype)
        qkvgb (doto [(array/astype (rng/normal k1 [r 1 delta-value-heads delta-key-dim]) sd)
                     (array/astype (rng/normal k2 [r 1 delta-value-heads delta-key-dim]) sd)
                     (array/astype (rng/normal k3 [r 1 delta-value-heads delta-value-dim]) sd)
                     (array/astype (array/* -0.1 (rng/uniform k4 [r 1 delta-value-heads])) sd)
                     (array/astype (rng/uniform k5 [r 1 delta-value-heads]) sd)]
                array/eval!)
        _ (array/eval! [x ids states pools])
        fs {:embed [(array/compile (fn [ids] (common/embed (:embed weights) ids dtype))) [ids]]
            :delta-mixers [(array/compile (fn [x states] (mapv #(delta-net config (layers %) x at (states %)) di))) [x states]]
            :gated-delta [(array/compile (fn [states [q k v g beta]]
                                           (mapv #(array/gated-delta q k v g beta (:state (states %))) di)))
                          [states qkvgb]]
            :attention-mixers [(array/compile (fn [x pools] (mapv #(attention config (layers %) x at (pools %)) ai))) [x pools]]
            :feed-forwards [(array/compile (fn [x] (mapv #(feed-forward config (layers %) x) (range (count kinds))))) [x]]
            :norms [(array/compile (fn [x] (mapv #(array/rms-norm x (:attn-norm (layers %)) eps) (range (count kinds))))) [x]]
            :head [(array/compile (fn [x] (common/linear (array/rms-norm x (:norm weights) eps)
                                                         (or (:output weights) (:embed weights))))) [x]]}]
    (into {} (map (fn [[k [f args]]] [k (part-ms f args)])) fs)))

(defn- parts-3
  "Qwen3's parts at r rows: {part ms}."
  [{:keys [config weights] :as m} ^long r]
  (let [{:keys [layers hidden dtype eps]} config
        ws (:layers weights)
        attention @#'qwen3/batch-attention
        x (array/astype (rng/normal (rng/key 1) [r 1 hidden]) dtype)
        ids (i32 (map vector (range 100 (+ 100 r))))
        b (batch r 80 (vec (repeat r 1)))
        at (assoc (select-keys b [:positions :slots :tables]) :d r)
        pools (qwen3/paged-pools m (* (inc r) width) block-size)
        _ (array/eval! [x ids pools])
        fs {:embed [(array/compile (fn [ids] (common/embed (:embed weights) ids dtype))) [ids]]
            :attention-mixers [(array/compile (fn [x pools] (mapv #(attention config (ws %) x at (pools %)) (range layers)))) [x pools]]
            :feed-forwards [(array/compile (fn [x] (mapv #(common/swiglu (ws %) x) (range layers)))) [x]]
            :head [(array/compile (fn [x] (common/linear (array/rms-norm x (:norm weights) eps)
                                                         (or (:output weights) (:embed weights))))) [x]]}]
    (into {} (map (fn [[k [f args]]] [k (part-ms f args)])) fs)))

;; ---------------------------------------------------------------------------

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (let [{:keys [rows dtype states parts?]} settings
        m (load-model path {:dtype dtype})
        _ (array/eval! (:weights m))
        pool (host/pool "profile rows-" (host/default-threads))
        modes (if (qwen35? m) states [:pools])]
    (println "## One compiled batch step (compile-paged-batch-step)\n")
    (println "- model:" path)
    (println "- settings:" (pr-str settings))
    (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
    (alter-var-root (requiring-resolve 'gen.backend.mlx.transforms/compile-fn) timed-compile)
    (let [fwd (if (qwen35? m) #'qwen35/batch-forward #'qwen3/batch-forward)
          f @fwd
          cols [:rows :mode :step :host :mlx-call :encode :wait :read :normalize :draw :floor :traces :gc :mlx-gc]
          results (with-redefs-fn {fwd (fn [& args] (swap! traces inc) (apply f args))}
                    (fn [] (vec (for [r rows mode modes] (profile-row m pool r mode)))))]
      (println)
      (println (str "| " (str/join " | " (map name cols)) " |"))
      (println (str "|" (str/join "|" (repeat (count cols) "---:")) "|"))
      (doseq [res results]
        (println (str "| " (str/join " | " (map #(let [v (% res)] (if (double? v) (format "%.2f" v) (str v))) cols)) " |"))))
    (when parts?
      (println "\n### The forward's parts, each compiled alone (ms a call, evaluated every 8)\n")
      (let [ps (mapv #(if (qwen35? m) (parts-35 m %) (parts-3 m %)) rows)
            ks (keys (first ps))]
        (println (str "| part | " (str/join " | " (map #(str % " rows") rows)) " |"))
        (println (str "|---|" (str/join "|" (repeat (count rows) "---:")) "|"))
        (doseq [k ks]
          (println (str "| " (name k) " | " (str/join " | " (map #(format "%.2f" (double (get % k))) ps)) " |")))))
    (println "\n- MLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
    (.shutdown ^java.util.concurrent.ExecutorService pool)
    (shutdown-agents)))
