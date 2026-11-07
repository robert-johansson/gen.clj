(ns gen.qwen3-bench
  "Qwen3's forward over gen.array (gen.llm.models.qwen3) on MLX, eager and
   compiled, against llama.cpp on the same GGUF (genclj-tjvd; the CUDA
   checkpoint's run 2, genclj-ssiv); and Qwen3.5's or Qwen3.6's
   (gen.llm.models.qwen35) for a GGUF of that architecture (genclj-xnx3).
   The same command runs on Metal and on MLX's CUDA build:

     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-bench <Qwen3 or Qwen3.5 GGUF>

   One GPU JVM at a time. Without llama.cpp's libraries the llama.cpp rows
   are skipped with a note. Every row is the median of gen.bench.runs runs
   (3) after one untimed warm run (which traces a compiled step).

   Sections (gen.bench.sections, default all, comma-separated):
   - prefill: a prompt of gen.bench.prompt tokens (512) at batch 1, read
     eager (prefill, the concatenating cache), compiled over a static
     cache in one chunk and in chunks of gen.bench.chunk (128), and by
     llama.cpp (gen.llm.branching/open);
   - decode: gen.bench.steps tokens (64) at batch 1 after that prompt,
     greedy: each step's argmax read on the host, then the next step. Eager
     over the concatenating cache, eager and compiled over a static cache
     of the prompt and the steps' capacity, compiled consuming the cache
     (compile-step's :donate?), and llama.cpp (gen.llm.branching/advance,
     its next-token log-probabilities normalized on the host as the
     Branching protocol gives them; fixed tokens, not their argmax). One
     more row, compiled and consuming, reads fixed tokens and evaluates
     once every 16 steps: no host round trip a step, so mostly the
     device's time;
   - batch: decode at gen.bench.batches rows (4,16,64) after a prompt of
     gen.bench.batch-prompt tokens (128), gen.bench.batch-steps steps (32):
     every row the same prompt, its own first token; eager concatenating,
     compiled copying, compiled consuming, and llama.cpp's sequences (a
     branch forked);
   - kernels: MLX's primitives in one decode step's graph (batch 1),
     eager and compiled, from mlx_export_to_dot: MLX counts no kernels.
     Views (reshape, transpose, squeeze, broadcast, slice) launch none;
     the rest are about one launch each (a fast attention two);
   - donation: whether a compiled step writes the cache's buffer in place,
     the cache held and consumed (the buffer's address before and after).
   gen.bench.llama-bench=<llama-bench binary> also runs llama-bench on the
   GGUF (-p prompt -n steps, every layer on the GPU), llama.cpp's ceiling.

   The model computes in gen.bench.dtype (float32, the backend's default
   float; the Q8_0 scales and biases are cast to it at load), and its
   caches hold keys and values in it: llama.cpp's hold float16 (its
   default KV type), and its Q8_0 blocks keep float16 scales.
   gen.bench.dtype=float16 computes in float16 instead."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment ValueLayout]
           [java.lang.invoke MethodHandle]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))
(defn- longs-prop [k default] (mapv parse-long (str/split (prop k default) #",")))

(def ^:private settings
  (let [prompt (long-prop "prompt" 512)
        steps (long-prop "steps" 64)]
    {:prompt prompt
     :steps steps
     :capacity (long-prop "capacity" (+ prompt steps))
     :chunk (long-prop "chunk" 128)
     :batches (longs-prop "batches" "4,16,64")
     :batch-prompt (long-prop "batch-prompt" 128)
     :batch-steps (long-prop "batch-steps" 32)
     :runs (long-prop "runs" 3)
     :dtype (keyword (prop "dtype" "float32"))
     :sections (set (map keyword (str/split (prop "sections" "prefill,decode,batch,kernels,donation") #",")))
     :llama-bench (System/getProperty "gen.bench.llama-bench")}))

;; ---------------------------------------------------------------------------
;; Timing
;; ---------------------------------------------------------------------------

(defn- median-ms
  "The median of (:runs settings) timings of (f (setup)) in ms, after one
   untimed run; setup's work is not timed."
  [setup f]
  (f (setup))
  (let [ts (sort (for [_ (range (:runs settings))]
                   (let [s (setup)
                         t0 (System/nanoTime)]
                     (f s)
                     (/ (- (System/nanoTime) t0) 1e6))))]
    (nth ts (quot (count ts) 2))))

(defn- row [label ms tokens]
  (println (format "| %-52s | %9.1f | %9.1f |" label (double ms) (/ (double tokens) (/ (double ms) 1e3))))
  (flush))

(defn- table [title]
  (println (str "\n" title "\n"))
  (println (format "| %-52s | %9s | %9s |" "" "ms" "tokens/s"))
  (println "|---|---:|---:|"))

;; ---------------------------------------------------------------------------
;; The model's family
;; ---------------------------------------------------------------------------

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- prefill [m & args] (apply (if (qwen35? m) qwen35/prefill qwen3/prefill) m args))
(defn- step [m & args] (apply (if (qwen35? m) qwen35/step qwen3/step) m args))
(defn- compile-step [m opts] ((if (qwen35? m) qwen35/compile-step qwen3/compile-step) m opts))
(defn- static-cache [m rows capacity] ((if (qwen35? m) qwen35/static-cache qwen3/static-cache) m rows capacity))
(defn- select-rows [m cache idx] ((if (qwen35? m) qwen35/select-rows qwen3/select-rows) cache idx))

(defn- load-model [path opts]
  (if (#{"qwen35" "qwen35moe"} (get-in (gguf/header path) [:metadata "general.architecture"]))
    (qwen35/load path opts)
    (qwen3/load path opts)))

;; ---------------------------------------------------------------------------
;; The prompt and the caches
;; ---------------------------------------------------------------------------

(defn- prompt-tokens
  "n token ids of ordinary text's range (the speeds do not depend on them)."
  [^long n]
  (mapv #(+ 1000 (mod (* 7919 (long %)) 20000)) (range n)))

(defn- tile-concatenating
  "A concatenating cache's row 0 as rows rows."
  [{:keys [offset layers]} rows]
  (let [idx (array/zeros [rows] {:dtype :int32})]
    {:offset offset :layers (mapv (fn [entry] (update-vals entry #(array/take % idx 0))) layers)}))

(defn- greedy
  "steps tokens decoded from cache by (step cache tokens) → [logits cache'],
   rows rows, row b's first token 13 + b; each step's argmax read on the
   host."
  [step cache ^long rows ^long steps]
  (loop [i 0 c cache toks (mapv #(vector (+ 13 (long %))) (range rows))]
    (when (< i steps)
      (let [[l c'] (step c toks)
            next (array/argmax (array/reshape l [rows -1]) {:axis -1})]
        (array/eval! [next c'])
        (recur (inc i) c' (mapv vector (array/->clj next)))))))

(defn- pipelined
  "steps fixed tokens read from cache by step (one row), evaluated once
   every 16 steps: no host round trip a step."
  [step cache ^long steps]
  (loop [i 0 c cache]
    (if (< i steps)
      (let [[l c'] (step c [[(+ 1000 i)]])]
        (when (zero? (mod (inc i) 16)) (array/eval! [l c']))
        (recur (inc i) c'))
      (array/eval! c))))

;; ---------------------------------------------------------------------------
;; MLX's graph: its primitives, from mlx_export_to_dot
;; ---------------------------------------------------------------------------

(def ^:private libc
  (delay (let [linker (Linker/nativeLinker)
               lookup (.defaultLookup linker)
               handle (fn [name ^FunctionDescriptor d] (.downcallHandle linker (.get (.find lookup name)) d (make-array Linker$Option 0)))]
           {:fopen (handle "fopen" (FunctionDescriptor/of ValueLayout/ADDRESS (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/ADDRESS])))
            :fclose (handle "fclose" (FunctionDescriptor/of ValueLayout/JAVA_INT (into-array MemoryLayout [ValueLayout/ADDRESS])))})))

(def ^:private views #{"Reshape" "Transpose" "Squeeze" "Broadcast" "ExpandDims" "Slice" "AsStrided"})

(defn- primitives
  "MLX's primitives in the graph of the tree x's unevaluated arrays:
   {:total :kernels (the non-views) :by {name count}}."
  [x]
  (let [handle (requiring-resolve 'gen.backend.mlx.array/handle)
        mlx? (requiring-resolve 'gen.backend.mlx.array/mlx-array?)
        file (str (java.io.File/createTempFile "qwen3-step" ".dot"))]
    (with-open [arena (Arena/ofConfined)]
      (let [{:keys [^MethodHandle fopen ^MethodHandle fclose]} @libc
            f (.invokeWithArguments fopen (object-array [(.allocateFrom arena ^String file) (.allocateFrom arena "w")]))
            v (gen.mlx.MlxRuntime/vectorOf (long-array (map #(handle %) (filter mlx? (array/tree-leaves x)))))
            namer (gen.mlx.Mlx/mlx_node_namer_new)]
        (try (gen.mlx.Mlx/mlx_export_to_dot ^MemorySegment f namer v)
             (finally (.invokeWithArguments fclose (object-array [f]))
                      (gen.mlx.Mlx/mlx_node_namer_free namer)
                      (gen.mlx.MlxRaw/mlx_vector_array_free v)))))
    (let [names (keep #(second (re-find #"label =\"([^\"]*)\", shape=rectangle" %)) (str/split-lines (slurp file)))
          by (frequencies (map #(first (str/split % #"[ (]")) names))]
      (.delete (java.io.File. file))
      {:total (count names)
       :kernels (reduce + 0 (for [[k n] by :when (not (views k))] n))
       :by (into (sorted-map) by)})))

(defn- buffer-address ^long [x]
  (array/eval! x)
  (.address ^MemorySegment (gen.mlx.Mlx/mlx_array_data_float32 ((requiring-resolve 'gen.backend.mlx.array/handle) x))))

;; ---------------------------------------------------------------------------
;; llama.cpp through gen.llm.llama
;; ---------------------------------------------------------------------------

(defn- llama-fn [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))
(defn- br [sym] @(requiring-resolve (symbol "gen.llm.branching" (name sym))))

(defn- llama-available? []
  (try ((llama-fn 'system-info)) true
       (catch Throwable t
         (println "\nNo llama.cpp:" (or (ex-message t) (.getName (class t))) "(-J-Dgen.llama.lib=<lib>)")
         false)))

(defn- llama-decode
  "steps tokens decoded from branches (advance consumes them), then the
   last disposed. The tokens are fixed, not argmaxes: an argmax over each
   row's host log-probabilities (151,936 floats) would add the bench's own
   host work to llama.cpp's (the speeds do not depend on the tokens)."
  [branches ^long steps]
  (loop [i 0 bs branches]
    (if (< i steps)
      (recur (inc i) ((br 'advance) bs (mapv #(+ 1000 (mod (* 31 (+ i (long %))) 20000)) (range (count bs)))))
      ((br 'dispose!) bs))))

;; ---------------------------------------------------------------------------
;; The sections
;; ---------------------------------------------------------------------------

(defn- prefill-section [m e]
  (let [{:keys [prompt chunk]} settings
        tokens (prompt-tokens prompt)
        f (compile-step m {:logits :last})
        static #(static-cache m 1 prompt)]
    (table (format "### Prefill: %d tokens, batch 1" prompt))
    (row "MLX eager (concatenating cache)" (median-ms (constantly nil) (fn [_] (array/eval! (prefill m [tokens] {:logits :last})))) prompt)
    (row "MLX compiled, static cache, one chunk" (median-ms static (fn [c] (array/eval! (f c [tokens])))) prompt)
    (row (format "MLX compiled, static cache, chunks of %d" chunk)
         (median-ms static (fn [c] (reduce (fn [c part] (let [[l c] (f c [(vec part)])] (array/eval! [l c]) c)) c (partition-all chunk tokens))))
         prompt)
    (when e
      (row "llama.cpp (gen.llm.llama: open)" (median-ms (constantly nil) (fn [_] ((br 'dispose!) [((br 'open) e tokens)]))) prompt))))

(defn- decode-section [m e]
  (let [{:keys [prompt steps capacity]} settings
        tokens (prompt-tokens prompt)
        [_ concatenating] (prefill m [tokens] {:logits :last})
        [_ static] (prefill m [tokens] {:logits :last :capacity capacity})
        _ (array/eval! [concatenating static])
        fresh #(array/eval! (select-rows m static [0]))
        copying (compile-step m {:logits :last})
        consuming (compile-step m {:logits :last :donate? true})
        eager #(step m %1 %2 {:logits :last})]
    (table (format "### Decode: %d tokens after %d, batch 1, greedy (static capacity %d)" steps prompt capacity))
    (row "MLX eager (concatenating cache)" (median-ms (constantly concatenating) #(greedy eager % 1 steps)) steps)
    (row "MLX eager, static cache" (median-ms fresh #(greedy eager % 1 steps)) steps)
    (row "MLX compiled, static cache (each step copies it)" (median-ms fresh #(greedy copying % 1 steps)) steps)
    (row "MLX compiled, static cache consumed (:donate?)" (median-ms fresh #(greedy consuming % 1 steps)) steps)
    (row "… the same, fixed tokens, evaluated every 16 steps" (median-ms fresh #(pipelined consuming % steps)) steps)
    (when e
      (row "llama.cpp (gen.llm.llama: advance, host log-probs)"
           (median-ms #((br 'open) e tokens) #(llama-decode [%] steps))
           steps))))

(defn- batch-section [m e]
  (let [{:keys [batch-prompt batch-steps batches]} settings
        tokens (prompt-tokens batch-prompt)
        capacity (+ batch-prompt batch-steps)
        [_ concatenating] (prefill m [tokens] {:logits :last})
        [_ static] (prefill m [tokens] {:logits :last :capacity capacity})
        copying (compile-step m {:logits :last})
        consuming (compile-step m {:logits :last :donate? true})
        eager #(step m %1 %2 {:logits :last})]
    (array/eval! [concatenating static])
    (doseq [n batches]
      (let [made (* (long n) batch-steps)
            rows #(array/eval! (select-rows m static (repeat n 0)))]
        (table (format "### Decode at batch %d: %d tokens a row after %d (static capacity %d)" n batch-steps batch-prompt capacity))
        (row "MLX eager (concatenating cache)"
             (median-ms #(array/eval! (tile-concatenating concatenating n)) #(greedy eager % n batch-steps)) made)
        (row "MLX compiled, static cache (each step copies it)" (median-ms rows #(greedy copying % n batch-steps)) made)
        (row "MLX compiled, static cache consumed (:donate?)" (median-ms rows #(greedy consuming % n batch-steps)) made)
        (when e
          (row "llama.cpp (gen.llm.llama: n sequences, advance)"
               (median-ms #(let [b ((br 'open) e tokens)] (into [b] ((br 'fork) b (dec (long n)))))
                          #(llama-decode % batch-steps))
               made))))))

(defn- kernels-section [m]
  (let [{:keys [prompt capacity]} settings
        tokens (prompt-tokens prompt)
        [_ concatenating] (prefill m [tokens] {:logits :last})
        [_ static] (prefill m [tokens] {:logits :last :capacity capacity})
        compiled (compile-step m {:logits :last})
        _ (array/eval! [concatenating static (compiled static [[13]])])
        show (fn [label x]
               (let [{:keys [total kernels by]} (primitives x)]
                 (println (format "| %-34s | %6d | %6d |" label total kernels))
                 by))]
    (println "\n### MLX's primitives in one decode step (batch 1)\n")
    (println (format "| %-34s | %6s | %6s |" "" "all" "kernels"))
    (println "|---|---:|---:|")
    (show "eager (concatenating cache)" (step m concatenating [[13]] {:logits :last}))
    (show "eager, static cache" (step m static [[13]] {:logits :last}))
    (let [by (show "compiled, static cache" (compiled static [[13]]))]
      (println "\nThe compiled step's primitives:" (str/join ", " (for [[k n] by] (str k " " n)))))))

(defn- donation-section [m]
  (let [{:keys [prompt capacity]} settings
        [_ static] (prefill m [(prompt-tokens prompt)] {:logits :last :capacity capacity})
        check (fn [f]
                (let [c (array/eval! (select-rows m static [0]))
                      keys-of #(:k (first (filter :k (:layers %))))
                      before (buffer-address (keys-of c))
                      [_ c'] (f c [[13]])]
                  (if (= before (buffer-address (keys-of c'))) "in place" "copied")))]
    (println (format "\n### Donation: a compiled step's cache buffer (the first attention layer's keys)\n\n- the cache held: %s\n- the cache consumed (:donate?): %s"
                     (check (compile-step m {:logits :last}))
                     (check (compile-step m {:logits :last :donate? true}))))))

(defn- llama-bench [path]
  (when-let [bin (:llama-bench settings)]
    (let [{:keys [prompt steps]} settings
          lib (System/getProperty "gen.llama.lib")
          pb (doto (ProcessBuilder. ^"[Ljava.lang.String;" (into-array String [bin "-m" path "-p" (str prompt) "-n" (str steps) "-ngl" "99" "-r" (str (:runs settings))]))
               (.redirectErrorStream true))]
      (when lib (.put (.environment pb) (if (str/includes? (System/getProperty "os.name") "Mac") "DYLD_LIBRARY_PATH" "LD_LIBRARY_PATH") lib))
      (println (str "\n### llama-bench (" bin ")\n"))
      (println (slurp (.getInputStream (.start pb)))))))

(defn- mlx-description []
  (let [b array/*backend*
        v (gen.mlx.Mlx/mlx_version)]
    (try (format "MLX %s, device %s, defaults %s"
                 (.getString ^MemorySegment (.reinterpret ^MemorySegment (gen.mlx.Mlx/mlx_string_data v) 64) 0)
                 ((requiring-resolve 'gen.backend.mlx.array/device) b)
                 (array/defaults b))
         (finally (gen.mlx.Mlx/mlx_string_free v)))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (println "## Qwen3 or Qwen3.5 on MLX, eager and compiled, against llama.cpp\n")
  (println "- model:" path)
  (println "-" (mlx-description))
  (println "- settings:" (pr-str (dissoc settings :sections)))
  (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
  (let [m (load-model path {:dtype (:dtype settings)})
        sections (:sections settings)
        e (when (and (some sections [:prefill :decode :batch]) (llama-available?))
            ((llama-fn 'engine) {:model path :max-branches (inc (long (reduce max 1 (:batches settings))))
                                 :n-ctx (max 4096 (+ (:prompt settings) (:steps settings) 64
                                                     (* (long (reduce max 1 (:batches settings))) (:batch-steps settings))))}))]
    (array/eval! (:weights m))
    (try
      (when e (println "- llama.cpp:" (pr-str (select-keys ((br 'info) e) [:n-ctx :max-branches :n-vocab])) (System/getProperty "gen.llama.lib")))
      (when (sections :prefill) (prefill-section m e))
      (when (sections :decode) (decode-section m e))
      (when (sections :batch) (batch-section m e))
      (when (sections :kernels) (kernels-section m))
      (when (sections :donation) (donation-section m))
      (finally (when e (.close ^java.io.Closeable e)))))
  (llama-bench path)
  (println "\nMLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
  (shutdown-agents))
