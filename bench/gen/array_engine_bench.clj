(ns gen.array-engine-bench
  "gen.array's engine (gen.llm.array-engine) against llama.cpp's
   (gen.llm.llama) on the same GGUF, through the Branching and LM
   protocols, as their users see them (genclj-8rnc):

     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.array-engine-bench <Qwen3 GGUF>

   One GPU JVM at a time. Rows (each the median of gen.bench.runs runs, 3,
   after one untimed warm run, which traces the compiled steps):
   - decode: a prompt of gen.bench.prompt tokens (64) read, forked
     gen.bench.batch times (16) and the forks advanced together
     gen.bench.steps times (32), by fixed tokens: tokens/s is batch ×
     steps over the time, every row's next-token log-probabilities
     normalized on the host (the Branching protocol's form). On the array
     engine the first step stacks the forks' rows, the rest step a lane
     group in place;
   - token SMC: gen.llm.smc/run of a completion of gen.bench.smc-tokens
     tokens (32, no end tokens, so every particle reads them all) with
     gen.bench.particles particles (16), by the bootstrap proposal, over a
     new LM of each engine: tokens/s is particles × tokens over the time.
     Each step is one LM read of the particles' distinct contexts (forks
     of the LM's entries, so the array engine stacks their rows, a copy a
     step, at a batch size that varies with the resampling).
   Measured on the Mac mini M4 (2026-11-07, Qwen3-0.6B-Q8_0, the defaults):
   decode 470 tokens/s against llama.cpp's 816 (the compiled step itself,
   bench/gen/qwen3_bench.clj, 467: the protocols cost nothing visible);
   token SMC 325 against 940 (124 stacked steps of 8.5 rows on average).
   The array engine computes in gen.bench.dtype (float32, the backend's
   default float), with blocks of gen.bench.block positions (64).
   gen.bench.sections (decode,smc) picks the rows; gen.bench.paged-kernel
   false makes the paged modes attend by gen.array's composite, the
   backend's kernel (Metal's or CUDA's) declined. On fenrir's CUDA (RTX
   PRO 6000, 2026-11-08, genclj-6qb6: each engine in a JVM of its own,
   C2, gen.bench.runs 5), Qwen3-0.6B Q8_0 in float16, token SMC: static
   797, paged 1158 (999 on the composite), continuous 1202 (989),
   llama.cpp's CUDA build 2228; 8 clients: 147, 219 (181), 555 (431),
   503. In float32 SMC 740, 1022 (721), 857 (858), 2214. Qwen3-8B Q8_0
   in float16: SMC 620, 842 (766), 841 (791), 1484; 8 clients 73, 81
   (76), 231 (220), 144. A JVM's median varies by up to 30% from one
   JVM to the next on fenrir: genclj-4acj (2026-11-08) alternated main
   and its branch, the medians of 5 or 6 JVMs. Token SMC on the 0.6B in
   float16: paged 1008 → 1272, continuous 1056 → 1445, llama.cpp 2177 →
   2579 (its LM gains the lent rows and forks too); one JVM each: static
   940 (797 in genclj-6qb6's session), float32 paged 834 → 983,
   continuous 957 → 972, llama.cpp
   2124 → 2576; the 8B paged 833 → 840, continuous 848 → 889, llama.cpp
   1455 → 1600; 8 clients unchanged (550 continuous, llama.cpp 501).
   Its paged mode (gen.bench.engines, of static, paged and llama;
   gen.bench.block-size 16, gen.bench.pool-blocks, gen.bench.min-rows),
   measured on the same Mac (2026-11-07, pool-blocks 128): float32 decode
   485-488 tokens/s against the static 466 and llama.cpp's 810, token SMC
   390-409 against 318 and 771; float16 decode 601 against 568, SMC 540
   against 619 (two more synchronizations a step, the copy-on-write's
   read and write, cost more there than the static engine's halved
   copy). (2026-11-08, genclj-6qb6: a block is now copied only for a
   second writer, 44 copies a run where 1012 were; token SMC 680-703 in
   float32 and 742-758 in float16, against 604 and 713 before, static
   345 and 649, llama.cpp 924. The pool's size no longer matters: the
   8 ms a GB was a GC genclj-c7uf removed. Qwen3.5-0.8B: paged 595 and
   698 against static 645 and 728, llama.cpp 276.)
   Its continuous mode (genclj-b3ow; gen.bench.engines continuous, :wait
   gen.bench.wait) batches concurrent callers' reads, and two more
   sections measure it (gen.bench.sections clients,lone):
   - clients: gen.bench.clients (8) threads at once, each with an LM of
     its own over the shared engine, each simulating a completion of
     gen.bench.client-tokens tokens (64, no end tokens) after a prompt of
     its own: tokens/s is clients × tokens over the wall time, and each
     LM read's latency (a token's) is reported at its median, 90th
     percentile and largest; then each client's time to its first token
     (from the start to its first read's end: its prompt admitted, at the
     median and largest), prompt admission's share of the wall time (the
     continuous mode's forwards that read a chunk of a longer read, and
     the paged mode's runs), and each timed run's traces of the batched
     step (a trace per new shape; genclj-6qb6). The clients run
     gen.bench.warm times (1) untimed first; gen.bench.max-chunks sets
     the continuous engine's :max-chunks;
   - lone: one such client alone, the same latencies.
   token SMC runs gen.bench.smc-clients runs at once (1), each over an
   LM of its own.
   Measured on the same Mac (2026-11-07, Qwen3-0.6B-Q8_0): 8 clients in
   float16 302 tokens/s continuous (17.6 ms a read at the median) against
   the paged mode's 90 (81.5 ms), the static's 86 and llama.cpp's 103
   (71.9 ms); in float32 175 against 82, 78 and 92. A lone client is as
   fast as in the paged mode (81.4 against 76.9 tokens/s, float32, 9
   runs). Token SMC at 16 particles 605 (float32) and 670 (float16)
   against the paged 586 and 641; two runs at once 799 against 596
   (llama.cpp 889). docs/research/llm-mlx-engine.md §8 has the rest.
   Qwen3.5-0.8B-Q8_0 (2026-11-08, genclj-qu77: float32 products of 5-15
   rows as 16, dropped states released; main's behaviour and the change
   alternated in one session, two rounds in float16): 8 clients in
   float16 216-232 → 255-268 tokens/s continuous (p90 read 53-60 → 29
   ms), in float32 154 → 205 (p50 40 → 29 ms); llama.cpp 69-70 (66-69).
   Token SMC continuous 872-907 → 921-937 (float16), 724 → 756 (float32);
   llama.cpp 280. Qwen3-0.6B-Q8_0 the same way: float32 token SMC static
   352 → 529 (its dropped caches released), 8 continuous clients 214 →
   264; float16 8 clients 368 → 382 (p90 27.5 → 18.3 ms), llama.cpp 98,
   token SMC llama.cpp 944. bench/gen/batch_profile.clj breaks a step
   down. Several prompts' chunks a forward (genclj-6qb6, 2026-11-08;
   gen.bench.warm 5, runs 6; main, then the branch), 8 clients: Qwen3-0.6B
   float16 402 → 422 tokens/s, read p90 16.8 → 14.9 ms, time to first
   token p50 215 → 210 ms (largest 429 → 361), admission 27.6 → 23.9% of
   the wall time; float32 273 → 287, 27.9 → 23.7, 270 → 231 (479 → 428),
   22.0 → 18.5%; llama.cpp 99 tokens/s, p90 77-80 ms, first token 312-319
   ms. Qwen3.5-0.8B float16 278 → 285, 27.8 → 22.0, 296 → 277 (612 →
   596), 27.0 → 24.0%; float32 208 → 217, 38.8 → 30.6, 324 → 346 (701 →
   676), 23.5 → 21.1%; llama.cpp 70 tokens/s, p90 112-114, first token
   413-431. The same branch with :max-chunks 1 against 16 in one session:
   Qwen3 float16 398 → 441, first token 252 → 197; Qwen3.5 float16 275 →
   298, 328 → 274. With one warm run, 0 to 7 of each timed run's forwards
   trace a new shape (50-125 ms each), moving these by up to 20%. The
   paged mode's runs by the compiled batch step (main, then the branch;
   gen.bench.warm 2, runs 5): token SMC Qwen3-0.6B float16 811 → 864,
   float32 691 → 757, Qwen3.5-0.8B float16 887 → 970, float32 706 → 783;
   8 clients' first token at the median 319 → 249, 373 → 268, 447 → 317,
   556 → 340 ms, tokens/s within ±5% (102 → 97, 91 → 93, 78 → 77, 70 →
   73)."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.gfi :as gfi]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.lm :as lm]
            [gen.llm.slots :as slots]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:prompt (long-prop "prompt" 64)
   :batch (long-prop "batch" 16)
   :steps (long-prop "steps" 32)
   :particles (long-prop "particles" 16)
   :smc-tokens (long-prop "smc-tokens" 32)
   :runs (long-prop "runs" 3)
   ;; the clients' untimed runs first (each traces the forwards' new shapes)
   :warm (long-prop "warm" 1)
   :block (long-prop "block" 64)
   :dtype (keyword (prop "dtype" "float32"))
   :sections (set (map keyword (str/split (prop "sections" "decode,smc") #",")))
   ;; the engines: gen.bench.engines, of static, paged and llama
   :engines (set (map keyword (str/split (prop "engines" "static,paged,continuous,llama") #",")))
   ;; the paged mode's (genclj-1daf)
   :block-size (long-prop "block-size" 16)
   :pool-blocks (long-prop "pool-blocks" 128)
   :min-rows (long-prop "min-rows" 1)
   ;; the continuous mode's (genclj-b3ow)
   :wait (parse-double (prop "wait" 0.25))
   ;; the engine's :max-chunks, when given (genclj-6qb6)
   :max-chunks (some-> (System/getProperty "gen.bench.max-chunks") parse-long)
   :clients (long-prop "clients" 8)
   :client-tokens (long-prop "client-tokens" 64)
   :smc-clients (long-prop "smc-clients" 1)
   ;; false: the paged modes attend by gen.array's composite, the
   ;; backend's kernel declined (Metal's or CUDA's; genclj-6qb6)
   :paged-kernel (parse-boolean (prop "paged-kernel" true))})

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(defn- median-ms
  "The median of (:runs settings) timings of (f) in ms, after one untimed
   run."
  [f]
  (f)
  (let [ts (sort (repeatedly (:runs settings) #(let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6))))]
    (nth ts (quot (count ts) 2))))

(defn- decode
  "The decode row's work on engine e: prompt read, batch forks advanced
   steps times, everything disposed."
  [e prompt]
  (let [{:keys [batch steps]} settings
        root (br/open e prompt)
        tokens (vec (take batch (cycle [13 220 279 323 315 264 304 374])))]
    (loop [bs (br/fork root batch) i 0]
      (if (= i steps)
        (br/dispose! (conj bs root))
        (recur (br/advance bs tokens) (inc i))))))

(defn- smc-run
  "The token SMC row's work on engine e and tokenizer tok: gen.bench.smc-
   clients runs at once, each over a new LM (closed after), each with a
   prompt of its own."
  [e tok prompt]
  (let [{:keys [particles smc-tokens smc-clients]} settings
        k (long smc-clients)
        one (fn [i p]
              (with-open [m (if (== 1 k) (lm/lm e tok) (lm/lm e tok {:max-branches (quot (long (:max-branches (br/info e))) k)
                                                                    :max-cells 2048}))]
                (smc/run (token/completion m {:end-tokens #{}}) [p smc-tokens] particles {:key (rng/key (+ 7 (long i)))})))]
    (if (== 1 k)
      (one 0 prompt)
      (mapv deref (mapv (fn [i] (future (one i (into [(first prompt)] (drop (* 3 (long i)) (rest prompt)))))) (range k))))))

(defn- timed-lm
  "m with each read's time (ms) conj'd onto the atom times, and with
   first-read (an atom of nil) and start (a promise of the clock's start, in
   ns), the time from start to its first read's end: the client's time
   to its first token."
  ([m times] (timed-lm m times nil nil))
  ([m times first-read start]
   (reify lm/LM
    (-tokenizer [_] (lm/-tokenizer m))
    (-next-logprobs [_ contexts]
      (let [t0 (System/nanoTime)
            v (lm/-next-logprobs m contexts)
            t1 (System/nanoTime)]
        (swap! times conj (/ (- t1 t0) 1e6))
        (when (and first-read (nil? @first-read))
          (reset! first-read (/ (- t1 (long @start)) 1e6)))
        v))
    (-score [_ context continuation] (lm/-score m context continuation))
    (-info [_] (lm/-info m)))))

;; Prompt admission's time (genclj-6qb6): the forwards that read a chunk
;; of a longer read (the continuous mode's, and since genclj-6qb6 the
;; paged mode's runs; before it, its paged-read-run), timed by wrapping
;; the engine's (private) functions while the clients run;
;; and the batched steps' traces (calls of the families' batch-forward,
;; which a compiled step makes once per new shape).
(def ^:private admission-ns (atom 0))
(def ^:private traces (atom 0))

(defn- admitting
  "(f) with admission-ns counting the ns of the engine's forwards that
   read a chunk (more than one token of a read) and of its paged runs."
  [f]
  (let [forward @#'ae/batch-forward!
        ;; the paged mode's eager run, on a commit before genclj-6qb6
        run-var (resolve 'gen.llm.array-engine/paged-read-run)
        timed (fn [g] (fn [& args]
                        (let [t0 (System/nanoTime)]
                          (try (apply g args)
                               (finally (swap! admission-ns + (- (System/nanoTime) t0)))))))]
    (with-redefs-fn (cond-> {(requiring-resolve 'gen.llm.models.qwen3/batch-forward)
                             (let [g @(requiring-resolve 'gen.llm.models.qwen3/batch-forward)] (fn [& args] (swap! traces inc) (apply g args)))
                             (requiring-resolve 'gen.llm.models.qwen35/batch-forward)
                             (let [g @(requiring-resolve 'gen.llm.models.qwen35/batch-forward)] (fn [& args] (swap! traces inc) (apply g args)))
                             #'ae/batch-forward! (fn [ctx reads]
                                                   (if (some #(< 1 (count (:tokens %))) reads)
                                                     ((timed forward) ctx reads)
                                                     (forward ctx reads)))}
                      run-var (assoc run-var (timed @run-var)))
      f)))

(defn- client-run
  "n completion clients at once on engine e (tokenizer tok), client i
   after a prompt of its own (prompt shifted by 5 i), each over an LM of
   its own: [wall-ms read-latencies]."
  [e tok prompt n]
  (let [{:keys [client-tokens]} settings
        n (long n)
        times (atom [])
        firsts (vec (repeatedly n #(atom nil)))
        go (promise)
        branches (quot (long (:max-branches (br/info e))) n)
        one (fn [i]
              (with-open [m (lm/lm e tok {:max-branches (min 4 branches) :max-cells 1024})]
                @go
                (gfi/simulate (token/completion (timed-lm m times (firsts i) go) {:end-tokens #{}})
                              [(into [(first prompt)] (drop (* 5 (long i)) (rest prompt))) client-tokens]
                              (rng/key (+ 11 (long i))))))
        fs (mapv (fn [i] (future (one i))) (range n))
        ;; every LM made before the clock starts
        _ (Thread/sleep 200)
        _ (reset! admission-ns 0)
        _ (reset! traces 0)
        t0 (System/nanoTime)]
    (deliver go t0)
    (run! deref fs)
    [(/ (- (System/nanoTime) t0) 1e6) @times (mapv deref firsts) (/ @admission-ns 1e6) @traces]))

(defn- percentile [xs q]
  (let [v (vec (sort xs))]
    (if (empty? v) Double/NaN (double (v (min (dec (count v)) (long (* (double q) (count v)))))))))

(defn- clients-rows
  "The clients (or lone) rows of an engine: the median wall time of
   (:runs settings) runs after an untimed one, and the latencies of every
   timed run's reads."
  [label e tok prompt n]
  (admitting
   (fn []
     (dotimes [_ (:warm settings)] (client-run e tok prompt n))
     (let [runs (vec (repeatedly (:runs settings) #(client-run e tok prompt n)))
           ms (nth (sort (map first runs)) (quot (count runs) 2))
           lat (mapcat second runs)
           ttft (mapcat #(nth % 2) runs)
           share (/ (reduce + (map #(nth % 3) runs)) (reduce + (map first runs)))]
       (println (format "| %-44s | %9.1f | %9.1f | %7.1f | %7.1f | %7.1f | %7.1f | %7.1f | %5.1f%% | %s |" label ms
                        (/ (* 1000.0 n (:client-tokens settings)) ms)
                        (percentile lat 0.5) (percentile lat 0.9) (reduce max 0.0 lat)
                        (percentile ttft 0.5) (reduce max 0.0 ttft) (* 100.0 share)
                        (mapv #(nth % 4) runs)))))))

(defn- row [label ms tokens]
  (println (format "| %-44s | %9.1f | %9.1f |" label ms (/ (* 1000.0 tokens) ms))))

(defn- run
  "-main's rows, for the model at path."
  [path]
  (let [{:keys [batch steps particles smc-tokens prompt]} settings
        engines (filterv (fn [[k]] ((:engines settings) k))
                         [[:static "gen.array (MLX)" #(ae/engine {:model path :max-branches 64 :block (:block settings) :dtype (:dtype settings)})]
                          [:paged (str "gen.array paged (MLX), min-rows " (:min-rows settings) (when-not (:paged-kernel settings) ", composite"))
                           #(ae/engine {:model path :max-branches 64 :dtype (:dtype settings) :paged? true
                                        :block-size (:block-size settings) :pool-blocks (:pool-blocks settings)
                                        :min-rows (:min-rows settings)})]
                          [:continuous (str "gen.array continuous (MLX), wait " (:wait settings) (when-not (:paged-kernel settings) ", composite"))
                           #(ae/engine (cond-> {:model path :max-branches 64 :dtype (:dtype settings) :paged? true :continuous? true
                                                :block-size (:block-size settings) :pool-blocks (:pool-blocks settings)
                                                :min-rows (:min-rows settings) :wait (:wait settings)}
                                         (:max-chunks settings) (assoc :max-chunks (:max-chunks settings))))]
                          [:llama "llama.cpp" #((llama 'engine) {:model path :max-branches 64 :n-ctx 8192})]])
        stats (fn [k e] (when-not (= :llama k)
                          (println "|" (pr-str (ae/stats e)) (if (= :continuous k) (pr-str (slots/schedule e)) "") "| | |")))]
    (with-open [^java.io.Closeable v ((llama 'vocabulary) path)]
      (let [tok ((llama 'tokenizer) v)
            text (str/join " " (repeat 40 "The sea is calm tonight, and the tide is full."))
            prompt-tokens (vec (take prompt ((llama 'tokenize) v text)))]
        (when ((:sections settings) :decode)
          (println "\n### Decode at batch" batch "," steps "steps after" prompt "tokens\n")
          (println "|                                              |        ms |  tokens/s |")
          (println "|---|---:|---:|")
          (doseq [[k label make] engines]
            (with-open [^java.io.Closeable e (make)]
              (row label (median-ms #(decode e prompt-tokens)) (* batch steps))
              (stats k e))))
        (when ((:sections settings) :smc)
          (println "\n### Token SMC," (:smc-clients settings) "run(s) at once," particles "particles," smc-tokens "tokens each (bootstrap)\n")
          (println "|                                              |        ms |  tokens/s |")
          (println "|---|---:|---:|")
          (doseq [[k label make] engines]
            (with-open [^java.io.Closeable e (make)]
              (row label (median-ms #(smc-run e tok prompt-tokens)) (* (:smc-clients settings) particles smc-tokens))
              (stats k e))))
        (doseq [[section n] [[:clients (:clients settings)] [:lone 1]]
                :when ((:sections settings) section)]
          (println "\n###" n (if (= 1 n) "client" "clients at once") "," (:client-tokens settings) "tokens each, an LM each\n")
          (println "|                                              |   wall ms |  tokens/s | p50 ms  | p90 ms  | max ms  | TTFT p50 | TTFT max | admission | traces |")
          (println "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
          (doseq [[k label make] engines]
            (with-open [^java.io.Closeable e (make)]
              (clients-rows label e tok prompt-tokens n)
              (stats k e))))))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (println "## gen.array's engine against llama.cpp's, through the protocols\n")
  (println "- model:" path)
  (println "- settings:" (pr-str settings))
  (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
  (with-redefs-fn (if (:paged-kernel settings) {} {(requiring-resolve 'gen.array.paged/fused-attention) (fn [& _] nil)})
    #(run path))
  (println "\nMLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
  (shutdown-agents))
