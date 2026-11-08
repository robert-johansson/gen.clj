(ns gen.smc-profile
  "Where one step of token SMC spends its time, on gen.array's engine
   (gen.llm.array-engine: static, paged or continuous) or llama.cpp's
   (gen.llm.llama), through the LM protocol, as gen.array-engine-bench's
   token SMC row runs it:

     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.smc-profile <Qwen3 GGUF>

   One GPU JVM at a time; gen.bench.engine picks the engine (paged;
   static, paged, continuous or llama), gen.bench.dtype the array
   engine's float (float16), gen.bench.particles (16), gen.bench.smc-
   tokens (32), gen.bench.runs timed runs (5) after one untimed (which
   traces the compiled steps). Every row is a span's time, summed over
   the timed runs and divided by their SMC steps (ms a step), with the
   calls a step; spans nest, so the rows are not a partition:
   - smc step: gen.llm.smc/step, the whole step on the calling thread;
     lm read: its lm/batch-next-logprobs; advance: the engine's advance
     (the hand-off to the engine's thread and back, its books, the
     step); row copies: br/logprobs (the LM's copy of each row; none
     where SMC reads them lent, lm/batch-next-logprobs-lent);
     draws: the proposals and draws on the SMC host pool (host/in-order,
     wall time); moved: each particle's score lookup and weight;
     resample: maybe-resample;
   - on the engine's thread: engine step (the mode's step: decode-step,
     paged-decode-step, or batch-forward!), its prepare and copy-blocks
     (the paged books and copy-on-write), step call (the compiled
     step's call: its graph), eval! (array/eval!), rows (host-rows: the
     read and the normalization), ->floats (the read), widen
     (halves->floats), normalize (log-softmax-rows); llama.cpp's
     decode (llama_decode) and rows (its rows normalized on its pool);
   - MLX's evaluations (gen.backend.mlx.memory/evaluating*): total, the
     time inside (on the evaluation thread, under its lock: MLX's
     encoding, the device's wait and a read's copy), and the hand-off
     (total less inside: CUDA's evaluation thread).
   With gen.bench.floor=true the paged step's device floor too: the last
   step's compiled call replayed gen.bench.floor-steps (32) times with
   its own inputs (the same tokens, written where they were), each
   evaluated (no read, no host work between): MLX's encoding and the
   device's time for that step's shape, and once more with the logits
   read (->floats) and normalized."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.host :as ahost]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.cache]
            [gen.llm.host :as host]
            [gen.llm.lm :as lm]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.slots]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.trie]
            [gen.random :as rng])
  (:import [java.lang.management GarbageCollectorMXBean ManagementFactory]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic LongAdder]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:engine (keyword (prop "engine" "paged"))
   :dtype (keyword (prop "dtype" "float16"))
   :particles (long-prop "particles" 16)
   :smc-tokens (long-prop "smc-tokens" 32)
   :prompt (long-prop "prompt" 64)
   :runs (long-prop "runs" 5)
   :block-size (long-prop "block-size" 16)
   :pool-blocks (long-prop "pool-blocks" 128)
   :floor (parse-boolean (prop "floor" false))
   :floor-steps (long-prop "floor-steps" 32)})

;; ---------------------------------------------------------------------------
;; Spans
;; ---------------------------------------------------------------------------

(def ^:private ^ConcurrentHashMap totals (ConcurrentHashMap.))
(def ^:private order (atom []))

(defn- add! [label ^long ns]
  (let [^objects a (.computeIfAbsent totals label
                                     (reify java.util.function.Function
                                       (apply [_ _] (swap! order conj label) (object-array [(LongAdder.) (LongAdder.)]))))]
    (.add ^LongAdder (aget a 0) ns)
    (.increment ^LongAdder (aget a 1))))

(defn- timed [label f]
  (fn [& args]
    (let [t0 (System/nanoTime)]
      (try (apply f args) (finally (add! label (- (System/nanoTime) t0)))))))

(def ^:private caller (atom nil))

(defn- wrap! [v label]
  (alter-var-root v #(timed label %)))

(defn- instrument!
  "The spans' vars wrapped (once per JVM)."
  []
  (wrap! #'smc/step "smc step")
  (wrap! #'lm/batch-next-logprobs "  lm read")
  (wrap! #'lm/batch-next-logprobs-lent "  lm read (rows lent)")
  (wrap! #'br/advance "    advance (calling thread)")
  (wrap! #'br/logprobs "    row copies (br/logprobs)")
  (when (= "true" (System/getProperty "gen.bench.books"))
    (wrap! #'br/fork "    fork (calling thread)")
    (wrap! #'br/fork-each "    fork-each (calling thread)")
    (wrap! #'br/dispose! "    dispose! (calling thread)")
    (wrap! #'gen.llm.cache/next-logprobs! "   cache next-logprobs!")
    (wrap! #'gen.llm.cache/answer! "    cache answer!")
    (wrap! #'gen.llm.cache/answer-round! "     answer-round!")
    (wrap! #'gen.llm.cache/step! "     step!")
    (wrap! #'gen.llm.cache/inputs! "      inputs!")
    (wrap! #'gen.llm.cache/fits? "     fits?")
    (wrap! #'gen.llm.cache/still-steps "     still-steps")
    (wrap! #'gen.llm.cache/add-entry "      add-entry")
    (wrap! #'gen.llm.cache/drop-entry "      drop-entry")
    (wrap! #'gen.llm.cache/check-context "    check-context")
    (wrap! #'gen.llm.trie/match "      trie match")
    (wrap! #'gen.llm.trie/insert "       trie insert")
    (wrap! #'gen.llm.trie/remove-entry "       trie remove")
    (wrap! #'gen.llm.slots/make-branch! "        make-branch!")
    (wrap! #'gen.llm.slots/advance* "      slots advance*")
    (wrap! #'gen.llm.slots/fork* "      slots fork*")
    (wrap! #'gen.llm.slots/dispose* "      slots dispose*"))
  (alter-var-root #'host/in-order
                  (fn [f] (fn [pool g xs]
                            (let [t0 (System/nanoTime)]
                              (try (f pool g xs)
                                   (finally (add! (if (identical? (Thread/currentThread) @caller) "  draws (host/in-order)" "      llama rows (in-order)")
                                                  (- (System/nanoTime) t0))))))))
  (wrap! #'gen.llm.smc/moved "  moved (score, weight)")
  (wrap! #'smc/maybe-resample "resample")
  (wrap! #'gen.llm.array-engine/decode-step "      engine step (static)")
  (wrap! #'gen.llm.array-engine/paged-decode-step "      engine step (paged)")
  (wrap! #'gen.llm.array-engine/batch-forward! "      engine step (forward)")
  (wrap! #'gen.llm.array-engine/prepared "        prepare (books)")
  (wrap! #'gen.llm.array-engine/copy-blocks! "        copy-blocks")
  (wrap! #'gen.llm.array-engine/i32 "        int32 arguments")
  (wrap! #'gen.llm.array-engine/host-rows "        rows (read, normalize)")
  (wrap! #'array/->floats "          ->floats")
  (wrap! #'ahost/halves->floats "            widen")
  (wrap! #'host/log-softmax-rows "          normalize")
  (alter-var-root #'array/eval!
                  (fn [f] (fn [t]
                            (let [t0 (System/nanoTime)]
                              (try (f t) (finally (add! (if (identical? (Thread/currentThread) @caller) "eval! (calling thread)" "        eval! (engine thread)")
                                                        (- (System/nanoTime) t0))))))))
  (doseq [v [#'qwen3/compile-paged-step #'qwen3/compile-paged-batch-step #'qwen3/compile-step]]
    (alter-var-root v (fn [compile] (fn [& args] (timed "        step call (graph)" (apply compile args))))))
  (when-let [ev (try (requiring-resolve 'gen.backend.mlx.memory/evaluating*) (catch Exception _ nil))]
    (alter-var-root ev (fn [f] (fn [g]
                                 (let [t0 (System/nanoTime)
                                       inside (long-array 1)]
                                   (try (f (fn [] (let [t1 (System/nanoTime)]
                                                    (try (g) (finally (aset inside 0 (+ (aget inside 0) (- (System/nanoTime) t1))))))))
                                        (finally (let [total (- (System/nanoTime) t0)]
                                                   (add! "MLX evaluations: total" total)
                                                   (add! "  inside (lock, encode, wait, read)" (aget inside 0))
                                                   (add! "  hand-off" (- total (aget inside 0)))))))))))
  (when-let [d (try (requiring-resolve 'gen.llm.llama/decode!) (catch Exception _ nil))]
    (alter-var-root d #(timed "      llama decode" %))))

(defn- gc-totals []
  (reduce (fn [[n t] ^GarbageCollectorMXBean g] [(+ (long n) (.getCollectionCount g)) (+ (long t) (.getCollectionTime g))])
          [0 0] (ManagementFactory/getGarbageCollectorMXBeans)))

(defn- report [steps wall-ms]
  (let [steps (long steps)]
    (println (format "| %-44s | %8s | %8s |" "span" "ms/step" "calls/step"))
    (println "|---|---:|---:|")
    (doseq [label @order]
      (let [^objects a (.get totals label)
            ns (.sum ^LongAdder (aget a 0))
            n (.sum ^LongAdder (aget a 1))]
        (println (format "| %-44s | %8.3f | %8.2f |" (str/replace label " " " ") (/ ns 1e6 steps) (/ (double n) steps)))))
    (println (format "\nsteps %d, wall %.1f ms a run, %.3f ms a step" steps wall-ms (/ (* wall-ms (:runs settings)) steps)))))

;; ---------------------------------------------------------------------------
;; The device floor: the last paged step replayed
;; ---------------------------------------------------------------------------

(def ^:private last-step (atom nil))

(defn- capture-steps!
  "The paged step's compiled function wrapped to keep its last call's
   function, inputs and outputs (gen.bench.floor)."
  []
  (alter-var-root #'qwen3/compile-paged-step
                  (fn [compile] (fn [& args]
                                  (let [f (apply compile args)]
                                    (fn [pools pos tables sl toks]
                                      (let [[_ layers :as out] (f pools pos tables sl toks)]
                                        (reset! last-step {:f f :args [pos tables sl toks] :pools layers})
                                        out)))))))

(defn- floor!
  "The last step's call replayed n times on the engine's thread, writing
   the same keys and values where it wrote them; its pools stay the
   engine's (the last replay's): ms a step evaluated alone, and with its
   rows read and normalized."
  [e n]
  ((requiring-resolve 'gen.llm.slots/call)
   e (fn [s]
       (let [{:keys [f args]} @last-step
             state (.-state ^gen.llm.array_engine.PagedSlots s)
             pool (host/pool "floor-" (host/default-threads))
             go (fn [read?]
                  (let [t0 (System/nanoTime)]
                    (dotimes [_ n]
                      (let [[logits pools] (apply f (:pools @state) args)]
                        (array/eval! [logits pools])
                        (vswap! state assoc :pools pools)
                        (when read? (host/log-softmax-rows pool (array/->floats logits) (peek (array/shape logits)) "floor"))))
                    (/ (- (System/nanoTime) t0) 1e6 n)))]
         (go false)
         (println (format "\ndevice floor (rows %s): %.3f ms a step evaluated alone, %.3f with its rows read and normalized"
                          (first (array/shape (first args))) (go false) (go true)))
         (.shutdown pool)))))

;; ---------------------------------------------------------------------------
;; The run
;; ---------------------------------------------------------------------------

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(defn- make-engine [path]
  (let [{:keys [engine dtype block-size pool-blocks]} settings]
    (case engine
      :static (ae/engine {:model path :max-branches 64 :block 64 :dtype dtype})
      :paged (ae/engine {:model path :max-branches 64 :dtype dtype :paged? true :block-size block-size :pool-blocks pool-blocks})
      :continuous (ae/engine {:model path :max-branches 64 :dtype dtype :paged? true :continuous? true
                              :block-size block-size :pool-blocks pool-blocks})
      :llama ((llama 'engine) {:model path :max-branches 64 :n-ctx 8192}))))

(defn- smc-run [e tok prompt]
  (let [{:keys [particles smc-tokens]} settings]
    (with-open [m (lm/lm e tok)]
      (let [state (smc/run (token/completion m {:end-tokens #{}}) [prompt smc-tokens] particles {:key (rng/key 7)})]
        (:t state)))))

(defn -main [path & _]
  (println "## One token-SMC step, by span\n")
  (println "- model:" path)
  (println "- settings:" (pr-str settings))
  (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version")
           (System/getProperty "java.vm.name") "; JVMCI compiler" (System/getProperty "jdk.graal.CompilerConfiguration" "-")
           (pr-str (filter #(str/includes? % "JVMCI") (.getInputArguments (ManagementFactory/getRuntimeMXBean)))))
  (reset! caller (Thread/currentThread))
  (instrument!)
  (when (:floor settings) (capture-steps!))
  (with-open [^java.io.Closeable v ((llama 'vocabulary) path)]
    (let [tok ((llama 'tokenizer) v)
          text (str/join " " (repeat 40 "The sea is calm tonight, and the tide is full."))
          prompt (vec (take (:prompt settings) ((llama 'tokenize) v text)))]
      (with-open [^java.io.Closeable e (make-engine path)]
        (smc-run e tok prompt)
        (smc-run e tok prompt)
        (.clear totals)
        (reset! order [])
        (let [[gc0 gct0] (gc-totals)
              t0 (System/nanoTime)
              steps (reduce + (repeatedly (:runs settings) #(smc-run e tok prompt)))
              wall (/ (- (System/nanoTime) t0) 1e6 (:runs settings))
              [gc1 gct1] (gc-totals)]
          (println (format "\n### %s, %s: %.1f tokens/s (%.1f ms a run)\n" (name (:engine settings)) (name (:dtype settings))
                           (/ (* 1000.0 (:particles settings) (:smc-tokens settings)) wall) wall))
          (report steps wall)
          (println (format "JVM GC: %d collections, %d ms over the timed runs" (- (long gc1) (long gc0)) (- (long gct1) (long gct0))))
          (when-not (= :llama (:engine settings)) (println "engine stats:" (pr-str (ae/stats e))))
          (when (and (:floor settings) (= :paged (:engine settings)) @last-step)
            (floor! e (:floor-steps settings)))))))
  (shutdown-agents)
  (System/exit 0))
