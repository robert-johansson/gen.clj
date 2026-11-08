(ns gen.step-census
  "The nodes and kernels of one compiled batch-1 decode step of a Qwen3.5
   or Qwen3.6 GGUF on MLX (genclj-yesw): the continuous mode's
   compiled paged batch step (gen.batch-profile's, states in place), and
   one layer of each part compiled alone:

     clojure -J-Dgen.backend=mlx -M:bench -m gen.step-census <qwen35 or qwen35moe GGUF>

   One GPU JVM at a time. For the step and each part it prints the
   compiled graph's nodes by primitive (MLX's print_graph of a compiled
   call's outputs before they are evaluated: the tape the host walks and
   encodes, fused elementwise chains as one Compiled node), and writes
   the whole graph to gen.bench.graphs (a directory; default target/census).
   With KPROF_LIB set (fenrir's kprof2/kprof3, injected by
   CUDA_INJECTION64_PATH), each part's calls are marked, so its kernels
   are counted and named. Parts, at gen.bench.rows (1) rows:
   - step: the whole step, gen.bench.steps (48) of them, each evaluated;
   - delta: one GatedDeltaNet layer's mixer (batch-delta-net);
   - gated-delta: gen.array/gated-delta alone, on that layer's shapes;
   - attention: one attention layer's mixer (batch-attention);
   - ffn: one feed-forward (the mixture of experts or the SwiGLU);
   - block: one GatedDeltaNet block (norms, mixer, feed-forward).
   gen.bench.dtype (float16). gen.bench.variant measures a what-if, not
   gen.clj's forward: short-sigmoid, gen.array's sigmoid as 1/(1 + e^-x)
   and softplus as logaddexp(x, 0) (MLX's own op), shorter elementwise
   chains than the stable composites, which MLX's compile fuses within
   its depth of 11; mlx-softmax, softmax as MLX's one primitive; both."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))

(def ^:private settings
  {:rows (parse-long (prop "rows" "1"))
   :dtype (keyword (prop "dtype" "float16"))
   :steps (parse-long (prop "steps" "48"))
   :graphs (prop "graphs" "target/census")
   :variant (keyword (prop "variant" "none"))
   :parts (set (map keyword (str/split (prop "parts" "step,delta,gated-delta,attention,ffn,block") #",")))})

;; ---------------------------------------------------------------------------
;; kprof's marks, MLX's print_graph
;; ---------------------------------------------------------------------------

(defn- downcall ^MethodHandle [^SymbolLookup lookup ^String name ^FunctionDescriptor fd]
  (let [f (.find lookup name)]
    (when (.isPresent f)
      (.downcallHandle (Linker/nativeLinker) ^MemorySegment (.get f) fd (make-array java.lang.foreign.Linker$Option 0)))))

(def ^:private ^MethodHandle mark-handle
  (delay (when-let [lib (System/getenv "KPROF_LIB")]
           (downcall (SymbolLookup/libraryLookup ^String lib (Arena/global)) "kprof_mark"
                     (FunctionDescriptor/ofVoid (into-array java.lang.foreign.MemoryLayout [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))))))

(defn- mark!
  "kprof's kernels since the last mark, under label, for calls calls."
  [label ^long calls]
  (when-let [^MethodHandle h @mark-handle]
    (with-open [arena (Arena/ofConfined)]
      (let [^java.util.List args [(.allocateFrom arena ^String label) (int calls)]] (.invokeWithArguments h args)))))

(def ^:private libc
  (delay (let [l (.defaultLookup (Linker/nativeLinker))]
           {:fopen (downcall l "fopen" (FunctionDescriptor/of ValueLayout/ADDRESS (into-array java.lang.foreign.MemoryLayout [ValueLayout/ADDRESS ValueLayout/ADDRESS])))
            :fclose (downcall l "fclose" (FunctionDescriptor/of ValueLayout/JAVA_INT (into-array java.lang.foreign.MemoryLayout [ValueLayout/ADDRESS])))})))

(defn- leaves [x] (filterv #(and (array/array? %) (not (array/host? %))) (array/tree-leaves x)))

(defn- print-graph!
  "MLX's print_graph of x's arrays (unevaluated) into the file path."
  [x path]
  (io/make-parents (io/file path))
  (let [handle (requiring-resolve 'gen.backend.mlx.array/handle)
        v (gen.mlx.MlxRuntime/vectorOf (long-array (map #(handle %) (leaves x))))
        namer (gen.mlx.Mlx/mlx_node_namer_new)
        {:keys [^MethodHandle fopen ^MethodHandle fclose]} @libc]
    (try
      (with-open [arena (Arena/ofConfined)]
        (let [^java.util.List open-args [(.allocateFrom arena ^String (str path)) (.allocateFrom arena "w")]
              ^MemorySegment f (.invokeWithArguments fopen open-args)
              ^java.util.List close-args [f]]
          (try (gen.mlx.Mlx/mlx_print_graph f namer v)
               (finally (.invokeWithArguments fclose close-args)))))
      (finally (gen.mlx.Mlx/mlx_node_namer_free namer)
               (gen.mlx.Mlx/mlx_vector_array_free v)))))

(defn- census
  "The nodes of the graph file path by primitive: a sorted map name → count."
  [path]
  (let [lines (drop 2 (str/split-lines (slurp path)))]
    (into (sorted-map)
          (frequencies (keep #(let [p (first (str/split % #" " 2))] (when (seq p) (if (str/starts-with? p "Compiled") "Compiled" p)))
                             lines)))))

(defn- report!
  "The graph of out under label: written, and its census printed."
  [label out calls]
  (let [path (str (:graphs settings) "/" (str/replace label #"[^A-Za-z0-9._-]+" "-") ".txt")]
    (print-graph! out path)
    (let [c (census path)]
      (println (format "\n#### %s: %d nodes (%s)\n" label (reduce + (vals c)) path))
      (println (str/join ", " (map (fn [[k v]] (str k " " v)) (sort-by (comp - val) c))))
      (flush)
      calls)))

;; ---------------------------------------------------------------------------

(def ^:private bp (fn [s] @(requiring-resolve (symbol "gen.batch-profile" s))))

(defn- timed-calls
  "n calls of (f) each evaluated, after 4, marked under label: ms a call."
  [label f ^long n]
  (dotimes [_ 4] (array/eval! (f)))
  (mark! "warm" 4)
  (let [t0 (System/nanoTime)]
    (dotimes [_ n] (array/eval! (f)))
    (let [ms (/ (- (System/nanoTime) t0) 1e6 n)]
      (mark! label n)
      ms)))

(defn- step-part
  "The whole compiled step at r rows: its graph once warm, then steps
   steps, each evaluated."
  [m ^long r]
  (let [stepper ((bp "stepper") m :lane)
        carry (volatile! ((bp "start") m r))
        pos (volatile! 64)
        one (fn []
              (let [[l c] (stepper @carry (vec (repeat r (+ 1000 (long @pos)))) @pos)]
                (vreset! carry c) (vswap! pos inc)
                [l c]))]
    (dotimes [_ 8] (array/eval! (one)))
    (let [out (one)]
      (report! (str "step " r) out 1)
      (array/eval! out))
    (mark! "warm" 9)
    (let [n (long (:steps settings))
          t0 (System/nanoTime)]
      (dotimes [_ n] (array/eval! (one)))
      (let [ms (/ (- (System/nanoTime) t0) 1e6 n)]
        (mark! (str "step " r) n)
        ms))))

(defn- layer-parts
  "One layer of each part, compiled alone at r rows: {part ms}."
  [{:keys [config weights] :as m} ^long r]
  (let [{:keys [kinds hidden dtype delta-value-heads delta-key-dim delta-value-dim]} config
        layers (:layers weights)
        delta-net @#'qwen35/batch-delta-net
        attention @#'qwen35/batch-attention
        feed-forward @#'qwen35/feed-forward
        block @#'qwen35/block
        width (bp "width") block-size (bp "block-size")
        x (array/astype (rng/normal (rng/key 1) [r 1 hidden]) dtype)
        b ((bp "batch") r 80 (vec (repeat r 1)))
        at (assoc (select-keys b [:positions :slots :tables]) :d r)
        states (qwen35/paged-states m r)
        pools (mapv #(select-keys % [:k :v]) (qwen35/paged-pools m (* (inc r) (long width)) block-size))
        di (first (filter #(= :delta (kinds %)) (range (count kinds))))
        ai (first (filter #(= :attention (kinds %)) (range (count kinds))))
        [k1 k2 k3 k4 k5] (rng/split (rng/key 2) 5)
        sd (if (#{:float16 :bfloat16} dtype) :float32 dtype)
        qkvgb (doto [(array/astype (rng/normal k1 [r 1 delta-value-heads delta-key-dim]) dtype)
                     (array/astype (rng/normal k2 [r 1 delta-value-heads delta-key-dim]) dtype)
                     (array/astype (rng/normal k3 [r 1 delta-value-heads delta-value-dim]) dtype)
                     (array/astype (array/* -0.1 (rng/uniform k4 [r 1 delta-value-heads])) dtype)
                     (array/astype (rng/uniform k5 [r 1 delta-value-heads]) dtype)]
                array/eval!)
        state (array/astype (:state (states di)) sd)
        _ (array/eval! [x states pools state])
        fs {:delta [(fn [x s] (delta-net config (layers di) x at s)) [x (states di)]]
            :gated-delta [(fn [[q k v g beta] s] (array/gated-delta q k v g beta s)) [qkvgb state]]
            :attention [(fn [x p] (attention config (layers ai) x at p)) [x (pools ai)]]
            :ffn [(fn [x] (feed-forward config (layers di) x)) [x]]
            :block [(fn [x s] (block config (layers di) x #(delta-net config (layers di) % at s))) [x (states di)]]}]
    (into (sorted-map)
          (keep (fn [[part [f args]]]
                  (when ((:parts settings) part)
                    (let [g (array/compile f)
                          label (str (name part) " " r)]
                      (array/eval! (apply g args))
                      (report! label (apply g args) 1)
                      [part (timed-calls label #(apply g args) 32)]))))
          fs)))

(defn- mlx-softmax
  "MLX's softmax of x along its last axis (precise), one primitive: a
   what-if (gen.array's softmax is a composite)."
  ([x] (mlx-softmax x nil))
  ([x _]
   (let [b (array/backend-of :softmax [x])
         handle (requiring-resolve 'gen.backend.mlx.array/handle)
         wrap (requiring-resolve 'gen.backend.mlx.array/wrap)
         stream (requiring-resolve 'gen.backend.mlx.array/stream)]
     (wrap b (gen.mlx.Mlx/mlx_softmax_axis (long (handle x)) (int -1) true (long (stream b)))))))

(def ^:private short-sigmoid
  {#'array/sigmoid (fn [x] (array// 1.0 (array/+ 1.0 (array/exp (array/negate x)))))
   #'array/softplus (fn [x] (array/logaddexp x 0.0))})

(def ^:private variants
  {:none {}
   :short-sigmoid short-sigmoid
   :mlx-softmax {#'array/softmax mlx-softmax}
   :both (assoc short-sigmoid #'array/softmax mlx-softmax)})

(defn- census-main [path]
  (let [{:keys [rows dtype parts]} settings
        m (qwen35/load path {:dtype dtype})
        _ (array/eval! (:weights m))]
    (println "## The nodes and kernels of a compiled decode step (gen.step-census)\n")
    (println "- model:" path)
    (println "- settings:" (pr-str settings) "kprof:" (some? @mark-handle))
    (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
    (mark! "load" 0)
    (let [step (when (parts :step) (step-part m rows))
          ps (layer-parts m rows)]
      (println "\n### ms a call (each evaluated)\n")
      (when step (println "- step:" (format "%.3f" (double step))))
      (doseq [[k v] ps] (println (str "- " (name k) ": " (format "%.3f" (double v))))))
    (shutdown-agents)))

;; ---------------------------------------------------------------------------
;; Costs: a node's and a kernel's host cost, from chains of n of each kind
;; ---------------------------------------------------------------------------

(defn- chain-ms
  "ms a call of the compiled f of x, each evaluated, over 200 calls after 20."
  [f x]
  (let [g (array/compile f)]
    (dotimes [_ 20] (array/eval! (g x)))
    (let [t0 (System/nanoTime)]
      (dotimes [_ 200] (array/eval! (g x)))
      (/ (- (System/nanoTime) t0) 1e6 200))))

(defn- costs
  "µs a node of each kind, from chains of n (gen.bench.n, 400) of it:
   reshapes (views), exps between reshapes (a kernel each), aranges, and
   copies (a transpose's reshape)."
  []
  (let [n (long (parse-long (prop "n" "400")))
        x (array/astype (rng/normal (rng/key 1) [1 2048]) :float16)
        _ (array/eval! x)
        base (chain-ms (fn [x] (array/+ x 1.0)) x)
        fs {:reshape (fn [x] (array/+ (reduce (fn [y i] (array/reshape y (if (even? i) [2048 1] [1 2048]))) x (range n)) 1.0))
            :exp-kernel (fn [x] (reduce (fn [y i] (array/reshape (array/exp y) (if (even? i) [2048 1] [1 2048]))) x (range n)))
            :arange (fn [x] (reduce (fn [y i] (array/+ y (array/astype (array/arange i (+ i 2048) 1 {:dtype :int32}) :float16))) x (range n)))
            :copy (fn [x] (reduce (fn [y _] (array/reshape (array/transpose (array/reshape y [32 64])) [1 2048])) x (range n)))}]
    (println "\n### Host costs, chains of" n "(µs a node above a one-add call of" (format "%.3f ms)" base) "\n")
    (doseq [[k f] fs]
      (let [ms (chain-ms f x)]
        (mark! (str "cost " (name k)) 200)
        (println (format "- %s: %.3f ms a call, %.2f µs each" (name k) ms (/ (* 1000.0 (- ms base)) n)))
        (flush)))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (when (= "costs" path) (costs) (shutdown-agents) (System/exit 0))
  (with-redefs-fn (variants (:variant settings)) #(census-main path)))
