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
   (76), 231 (220), 144.
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
     percentile and largest;
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
   llama.cpp 280. bench/gen/batch_profile.clj breaks a step down."
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
  "m with each read's time (ms) conj'd onto the atom times."
  [m times]
  (reify lm/LM
    (-tokenizer [_] (lm/-tokenizer m))
    (-next-logprobs [_ contexts]
      (let [t0 (System/nanoTime)
            v (lm/-next-logprobs m contexts)]
        (swap! times conj (/ (- (System/nanoTime) t0) 1e6))
        v))
    (-score [_ context continuation] (lm/-score m context continuation))
    (-info [_] (lm/-info m))))

(defn- client-run
  "n completion clients at once on engine e (tokenizer tok), client i
   after a prompt of its own (prompt shifted by 5 i), each over an LM of
   its own: [wall-ms read-latencies]."
  [e tok prompt n]
  (let [{:keys [client-tokens]} settings
        n (long n)
        times (atom [])
        go (promise)
        branches (quot (long (:max-branches (br/info e))) n)
        one (fn [i]
              (with-open [m (lm/lm e tok {:max-branches (min 4 branches) :max-cells 1024})]
                @go
                (gfi/simulate (token/completion (timed-lm m times) {:end-tokens #{}})
                              [(into [(first prompt)] (drop (* 5 (long i)) (rest prompt))) client-tokens]
                              (rng/key (+ 11 (long i))))))
        fs (mapv (fn [i] (future (one i))) (range n))
        ;; every LM made before the clock starts
        _ (Thread/sleep 200)
        t0 (System/nanoTime)]
    (deliver go true)
    (run! deref fs)
    [(/ (- (System/nanoTime) t0) 1e6) @times]))

(defn- percentile [xs q]
  (let [v (vec (sort xs))]
    (if (empty? v) Double/NaN (double (v (min (dec (count v)) (long (* (double q) (count v)))))))))

(defn- clients-rows
  "The clients (or lone) rows of an engine: the median wall time of
   (:runs settings) runs after an untimed one, and the latencies of every
   timed run's reads."
  [label e tok prompt n]
  (client-run e tok prompt n)
  (let [runs (repeatedly (:runs settings) #(client-run e tok prompt n))
        ms (nth (sort (map first runs)) (quot (count runs) 2))
        lat (mapcat second runs)]
    (println (format "| %-44s | %9.1f | %9.1f | %7.1f | %7.1f | %7.1f |" label ms
                     (/ (* 1000.0 n (:client-tokens settings)) ms)
                     (percentile lat 0.5) (percentile lat 0.9) (reduce max 0.0 lat)))))

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
                           #(ae/engine {:model path :max-branches 64 :dtype (:dtype settings) :paged? true :continuous? true
                                        :block-size (:block-size settings) :pool-blocks (:pool-blocks settings)
                                        :min-rows (:min-rows settings) :wait (:wait settings)})]
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
          (println "|                                              |   wall ms |  tokens/s | p50 ms  | p90 ms  | max ms  |")
          (println "|---|---:|---:|---:|---:|---:|")
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
