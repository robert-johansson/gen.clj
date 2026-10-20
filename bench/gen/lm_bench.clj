(ns gen.lm-bench
  "The LM protocol's costs (gen.llm.lm over gen.llm.cache) on the pure-JVM
   toy engines: what the cache adds to a call over the Branching operations
   it makes, against context length; hit rates, reads and evictions in
   inference-shaped workloads; the caller's allocation per call; and, at
   Qwen3's vocabulary (151,936 tokens), the costs behind next-logprobs'
   value shape (a float array copy, a gen.array conversion, a draw).

   Run: clojure -M:bench -m gen.lm-bench          (the toy engines)
        clojure -M:bench -m gen.lm-bench memory   (the memory held and churned;
                                                   try -J-Xmx2g too)
   With llama.cpp's libraries (native/build-llama.sh) and a model:
        clojure -M:bench -m gen.lm-bench model.gguf"
  (:require [criterium.core :as crit]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.llm.branching :as br]
            [gen.llm.cell-engine :as ce]
            [gen.llm.lm :as lm]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng])
  (:import [java.lang.management ManagementFactory]))

(set! *warn-on-reflection* true)

(defn- us [f] (* 1e6 (first (:mean (crit/quick-benchmark* f {})))))

(defn- allocated
  "Bytes this thread allocates per call of f, over n calls."
  ^double [f n]
  (let [bean ^com.sun.management.ThreadMXBean (ManagementFactory/getThreadMXBean)
        id (.threadId (Thread/currentThread))]
    (dotimes [_ n] (f))
    (let [before (.getThreadAllocatedBytes bean id)]
      (dotimes [_ n] (f))
      (/ (double (- (.getThreadAllocatedBytes bean id) before)) n))))

(def ^:private n-vocab 97)

(defn- context [n] (into [95] (map #(mod % 4)) (range (dec n))))

;; ---------------------------------------------------------------------------
;; What the cache adds to a call
;; ---------------------------------------------------------------------------

(defn- overhead []
  (println "\n## Per call, toy engine (97 tokens), µs: the LM against the Branching operations it makes")
  (println "| context | hit | one-token read | fork+advance+dispose | 32 one-token reads | fork 32+advance+dispose | 8 tokens scored from the trie |")
  (println "|---|---|---|---|---|---|---|")
  (doseq [n [10 100 1000]]
    ;; a: one-token reads of 90 children in turn, so each evicts at steady
    ;; state; b: 32 children a call, each call's disjoint from the last, so
    ;; that its 33 branches hold none of them. The engine keeps 32 more
    ;; branches for the Branching operations.
    (with-open [^java.io.Closeable e (toy/engine {:max-branches 130 :n-ctx 2048})
                a (lm/lm e (tt/tokenizer n-vocab) {:max-branches 64 :max-cells 8192})
                b (lm/lm e (tt/tokenizer n-vocab) {:max-branches 33 :max-cells 8192})]
      (let [ctx (context n)
            _ (lm/next-logprobs a ctx)
            _ (lm/next-logprobs b ctx)
            hit (us #(lm/next-logprobs a ctx))
            t (volatile! 0)
            one (us #(lm/next-logprobs a (conj ctx (vswap! t (fn [x] (mod (inc (long x)) 90))))))
            k (volatile! 0)
            batch (us #(let [k (vswap! k (fn [x] (mod (+ (long x) 32) 96)))]
                         (lm/batch-next-logprobs b (mapv (fn [i] (conj ctx (mod (+ (long k) (long i)) 96))) (range 32)))))
            branch (br/open e ctx)
            raw-one (us #(br/dispose! (br/advance [(br/fork branch)] [1])))
            raw-batch (us #(br/dispose! (br/advance (br/fork branch 32) (vec (range 32)))))
            path (into ctx [1 2 3 0 1 2 3 0])
            _ (doseq [i (range 9)] (lm/next-logprobs a (subvec path 0 (+ n i))))
            scored (us #(lm/score a ctx (subvec path n)))]
        (br/dispose! branch)
        (println (format "| %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" n hit one raw-one batch raw-batch scored))))))

(defn- allocation []
  (println "\n## Allocation on the caller's thread per call, bytes (toy engine, 97 tokens: a row is 404 bytes)")
  (with-open [^java.io.Closeable e (toy/engine {:max-branches 64 :n-ctx 2048})
              m (lm/lm e (tt/tokenizer n-vocab) {:max-cells 8192})]
    (let [ctx (context 100)
          t (volatile! 0)]
      (lm/next-logprobs m ctx)
      (println "| call | bytes |")
      (println "|---|---|")
      (println (format "| hit, 100-token context | %.0f |" (allocated #(lm/next-logprobs m ctx) 20000)))
      (println (format "| one-token read, an eviction, 64 branches | %.0f |"
                       (allocated #(lm/next-logprobs m (conj ctx (vswap! t (fn [x] (mod (inc (long x)) 90))))) 20000))))))

;; ---------------------------------------------------------------------------
;; Workloads
;; ---------------------------------------------------------------------------

(defn- draw [^java.util.Random rng ^floats row]
  (let [u (.nextDouble rng)]
    (loop [t 0 acc 0.0]
      (let [acc (+ acc (Math/exp (aget row t)))]
        (if (or (< u acc) (= t (dec (alength row)))) t (recur (inc t) acc))))))

(defn- smc [m prompt n steps]
  (let [rng (java.util.Random. 7)]
    (loop [s 0 ps (vec (repeat n prompt))]
      (when (< s steps)
        (let [rows (lm/batch-next-logprobs m ps)
              grown (mapv #(conj %1 (draw rng %2)) ps rows)]
          (recur (inc s) (mapv (fn [_] (grown (.nextInt rng n))) grown)))))))

(defn- mh [m len iterations]
  (let [rng (java.util.Random. 11)
        grow (fn [ctx k] (reduce (fn [ctx _] (conj ctx (draw rng (lm/next-logprobs m ctx)))) ctx (range k)))
        prompt [95 40 73]]
    (loop [i 0 trace (grow prompt len)]
      (when (< i iterations)
        (let [site (+ 3 (.nextInt rng len))
              proposal (grow (subvec trace 0 site) (- (count trace) site))]
          (lm/score m prompt (subvec proposal 3))
          (recur (inc i) (if (.nextBoolean rng) proposal trace)))))))

(defn- workloads []
  (println "\n## Workloads, cell engine (llama.cpp's cell accounting, 97 tokens)")
  (println "| workload | bounds | requests | hit rate | reads | tokens read / request | forks | evictions | µs / request | backstop |")
  (println "|---|---|---|---|---|---|---|---|---|---|")
  (doseq [[what bounds f] [["SMC, 32 particles × 100 steps" {:max-branches 64 :max-cells 4096} #(smc % [95 40 73] 32 100)]
                           ["SMC, 32 × 100, few cells" {:max-branches 64 :max-cells 1024} #(smc % [95 40 73] 32 100)]
                           ["MH, a 50-token trace, 200 iterations" {:max-branches 16 :max-cells 1024} #(mh % 50 200)]
                           ["MH, the same, 4 branches" {:max-branches 4 :max-cells 1024} #(mh % 50 200)]]]
    (with-open [^java.io.Closeable e (ce/engine (assoc bounds :n-vocab n-vocab :n-ctx 4096))
                m (lm/lm e (tt/tokenizer n-vocab) bounds)]
      (let [t0 (System/nanoTime)
            _ (f m)
            secs (/ (- (System/nanoTime) t0) 1e9)
            {:keys [requests hits reads read-tokens forks evictions]} (lm/info m)]
        (println (format "| %s | %d branches, %d cells | %d | %.2f | %d | %.2f | %d | %d | %.1f | %d |"
                         what (:max-branches bounds) (:max-cells bounds) requests (/ (double hits) requests)
                         reads (/ (double read-tokens) requests) forks evictions (/ (* 1e6 secs) requests)
                         (:collections (br/info e))))))))

;; ---------------------------------------------------------------------------
;; next-logprobs' value shape, at a real vocabulary's size
;; ---------------------------------------------------------------------------

(defn- value-shape []
  (let [v 151936
        ^floats lp (toy/toy-logprobs v [95 40 73])
        to-array (fn [^floats lp] (array/array (let [d (double-array (alength lp))]
                                                 (dotimes [j (alength lp)] (aset d j (double (aget lp j))))
                                                 d)))
        x (to-array lp)
        dist (d/categorical-logits x)
        key (rng/key 1)
        host-draw (fn [^floats lp key]
                    (let [u (double (rng/uniform key))]
                      (loop [t 0 acc 0.0]
                        (let [acc (+ acc (Math/exp (aget lp t)))]
                          (if (or (< u acc) (= t (dec (alength lp)))) t (recur (inc t) acc))))))]
    (println "\n## A row of 151,936 tokens (Qwen3's vocabulary), JVM backend, µs")
    (println "| operation | µs |")
    (println "|---|---|")
    (doseq [[what f] [["the toy engine computing the row (its 'decode')" #(toy/toy-logprobs v [95 40 73])]
                      ["next-logprobs' copy (aclone of the float row)" #(aclone lp)]
                      ["float row → double[] → gen.array (array/array clones it again)" #(to-array lp)]
                      ["categorical-logits sample (Gumbel-max over the row)" #(dc/sample dist key)]
                      ["categorical-logits log-prob of a token (log-softmax, pick)" #(dc/log-prob dist 17)]
                      ["a host draw: one gen.random uniform, inverse CDF over the float row" #(host-draw lp key)]]]
      (println (format "| %s | %.0f |" what (us f))))))

;; ---------------------------------------------------------------------------
;; Memory
;; ---------------------------------------------------------------------------

(defn- retained
  "The heap in use after full collections, in bytes (a measurement, never
   on the LM's path)."
  ^long []
  (dotimes [_ 4] (System/gc) (Thread/sleep 100))
  (let [r (Runtime/getRuntime)] (- (.totalMemory r) (.freeMemory r))))

(defn- gc-counts
  "[collections milliseconds] of every collector so far."
  []
  (reduce (fn [[n ms] ^java.lang.management.GarbageCollectorMXBean b]
            [(+ (long n) (.getCollectionCount b)) (+ (long ms) (.getCollectionTime b))])
          [0 0] (ManagementFactory/getGarbageCollectorMXBeans)))

(defn- memory []
  (let [v 151936
        heap (.maxMemory (Runtime/getRuntime))]
    (println (format "\n## Memory, Qwen3's vocabulary (151,936 tokens; a row is 593 KiB), max heap %d MiB"
                     (quot heap 1048576)))
    (with-open [^java.io.Closeable e (toy/engine {:n-vocab v :max-branches 64 :n-ctx 4096})
                m (lm/lm e (tt/tokenizer v) {:max-cells 65536})]
      (let [prompt [95 40 73]
            _ (lm/next-logprobs m prompt)
            before (retained)
            _ (lm/batch-next-logprobs m (mapv #(conj prompt %) (range 63)))
            after (retained)
            [n0 ms0] (gc-counts)
            t0 (System/nanoTime)
            _ (dotimes [_ 5000] (lm/next-logprobs m prompt))
            us (/ (- (System/nanoTime) t0) 1e3 5000)
            [n1 ms1] (gc-counts)]
        (println (format "Held per entry (63 more entries): %.0f KiB" (/ (- after before) 63.0 1024)))
        (println (format "Hits, each copying its row: %.1f µs each; %d collections, %d ms of GC over 5,000"
                         us (- (long n1) (long n0)) (- (long ms1) (long ms0))))))
    (with-open [^java.io.Closeable e (toy/engine {:n-vocab n-vocab :max-branches 32 :n-ctx 4096})
                m (lm/lm e (tt/tokenizer n-vocab) {:max-cells 65536})]
      (let [before (retained)
            ;; 32 entries of 2,000 tokens, no two sharing a token after the first
            _ (doseq [i (range 32)] (lm/next-logprobs m (into [i] (map #(mod (+ i %) 90)) (range 1999))))
            after (retained)
            tokens (:tokens (lm/info m))]
        (println (format "Held per token on a path (%d tokens, rows of 97): %.0f bytes" tokens
                         (/ (- after before 0.0) tokens)))))))

;; ---------------------------------------------------------------------------
;; llama.cpp
;; ---------------------------------------------------------------------------

(defn- mean-ms
  "The mean wall-clock milliseconds of (f i) over i below n, after two
   warm-up calls."
  ^double [n f]
  (f -1) (f -2)
  (let [t0 (System/nanoTime)]
    (dotimes [i n] (f i))
    (/ (- (System/nanoTime) t0) 1e6 n)))

(defn- llama-costs
  "The LM's costs over llama.cpp with model: per call, against the
   Branching operations it makes, and a token-SMC-shaped workload."
  [model]
  (let [engine @(requiring-resolve 'gen.llm.llama/engine)
        tokenizer @(requiring-resolve 'gen.llm.llama/tokenizer)
        tokenize @(requiring-resolve 'gen.llm.llama/tokenize)]
    (with-open [^java.io.Closeable e (engine {:model model :max-branches 96 :n-ctx 8192})]
      (let [t0 (System/nanoTime)
            tok (tokenizer e)]
        (println (format "\n## llama.cpp: %s\n\nThe tokenizer (every token's bytes, %d tokens): %.0f ms"
                         (:desc (:model (br/info e))) (:n-vocab tok) (/ (- (System/nanoTime) t0) 1e6)))
        (let [prompt (tokenize e "The capital of Sweden is a city called")
              branch (br/open e prompt)
              ;; the 2,000 likeliest next tokens: children never asked before
              ts (vec (take 2000 (sort-by #(- (aget ^floats (br/logprobs branch) (int %))) (range (:n-vocab tok)))))
              rows (with-open [m (lm/lm e tok {:max-branches 64 :max-cells 6144})]
                     (lm/next-logprobs m prompt)
                     [["hit (the prompt)" (mean-ms 200 (fn [_] (lm/next-logprobs m prompt))) nil]
                      ["one-token read (a new child of the prompt)"
                       (mean-ms 64 (fn [i] (lm/next-logprobs m (conj prompt (ts (+ 2 i))))))
                       (mean-ms 64 (fn [_] (br/dispose! (br/advance [(br/fork branch)] [(ts 0)]))))]
                      ["8 one-token reads, one advance"
                       (mean-ms 16 (fn [i] (lm/batch-next-logprobs m (mapv #(conj prompt (ts (+ 100 (* 8 (+ 2 i)) %))) (range 8)))))
                       (mean-ms 16 (fn [_] (br/dispose! (br/advance (br/fork branch 8) (subvec ts 0 8)))))]
                      ["32 one-token reads, one advance"
                       (mean-ms 8 (fn [i] (lm/batch-next-logprobs m (mapv #(conj prompt (ts (+ 400 (* 32 (+ 2 i)) %))) (range 32)))))
                       (mean-ms 8 (fn [_] (br/dispose! (br/advance (br/fork branch 32) (subvec ts 0 32)))))]
                      ["16 tokens scored, read"
                       (mean-ms 16 (fn [i] (lm/score m prompt (vec (repeat 16 (ts (+ 900 i)))))))
                       (mean-ms 16 (fn [i] (br/score branch (vec (repeat 16 (ts (+ 950 i)))))))]
                      ["16 tokens scored from the trie"
                       (mean-ms 200 (fn [_] (lm/score m prompt (vec (repeat 16 (ts 900)))))) nil]])]
          (br/dispose! branch)
          (println "\n| call | LM, ms | the Branching operations alone, ms |")
          (println "|---|---|---|")
          (doseq [[what lm-ms raw-ms] rows]
            (println (format "| %s | %.2f | %s |" what lm-ms (if raw-ms (format "%.2f" raw-ms) "–"))))
          (with-open [m (lm/lm e tok {:max-branches 64 :max-cells 6144})]
            (let [t0 (System/nanoTime)
                  _ (smc m prompt 16 64)
                  secs (/ (- (System/nanoTime) t0) 1e9)
                  {:keys [requests hits reads evictions]} (lm/info m)]
              (println (format "\nToken SMC, 16 particles × 64 steps: %.1f ms a step, %d reads, %d evictions, hit rate %.2f, backstop %d"
                               (/ (* 1e3 secs) 64) reads evictions (/ (double hits) requests)
                               (:collections (br/info e)))))))))))

(defn -main
  "The toy engines' costs; with memory, only the memory held and churned;
   with a GGUF file, llama.cpp's."
  [& [arg]]
  (case arg
    nil (do (overhead) (allocation) (workloads) (value-shape) (memory))
    "memory" (memory)
    (llama-costs arg))
  (shutdown-agents))
