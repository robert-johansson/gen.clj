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
   gen.bench.sections (decode,smc) picks the rows."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.lm :as lm]
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
   :sections (set (map keyword (str/split (prop "sections" "decode,smc") #",")))})

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
  "The token SMC row's work on engine e and tokenizer tok: a new LM, a run,
   the LM closed."
  [e tok prompt]
  (let [{:keys [particles smc-tokens]} settings]
    (with-open [m (lm/lm e tok)]
      (smc/run (token/completion m {:end-tokens #{}}) [prompt smc-tokens] particles {:key (rng/key 7)}))))

(defn- row [label ms tokens]
  (println (format "| %-44s | %9.1f | %9.1f |" label ms (/ (* 1000.0 tokens) ms))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (println "## gen.array's engine against llama.cpp's, through the protocols\n")
  (println "- model:" path)
  (println "- settings:" (pr-str settings))
  (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
  (let [{:keys [batch steps particles smc-tokens prompt]} settings
        engines [["gen.array (MLX)" #(ae/engine {:model path :max-branches 64 :block (:block settings) :dtype (:dtype settings)})]
                 ["llama.cpp" #((llama 'engine) {:model path :max-branches 64 :n-ctx 8192})]]]
    (with-open [^java.io.Closeable v ((llama 'vocabulary) path)]
      (let [tok ((llama 'tokenizer) v)
            text (str/join " " (repeat 40 "The sea is calm tonight, and the tide is full."))
            prompt-tokens (vec (take prompt ((llama 'tokenize) v text)))]
        (when ((:sections settings) :decode)
          (println "\n### Decode at batch" batch "," steps "steps after" prompt "tokens\n")
          (println "|                                              |        ms |  tokens/s |")
          (println "|---|---:|---:|")
          (doseq [[label make] engines]
            (with-open [^java.io.Closeable e (make)]
              (row label (median-ms #(decode e prompt-tokens)) (* batch steps))
              (when (= "gen.array (MLX)" label) (println "|" (pr-str (ae/stats e)) "| | |")))))
        (when ((:sections settings) :smc)
          (println "\n### Token SMC," particles "particles," smc-tokens "tokens each (bootstrap)\n")
          (println "|                                              |        ms |  tokens/s |")
          (println "|---|---:|---:|")
          (doseq [[label make] engines]
            (with-open [^java.io.Closeable e (make)]
              (row label (median-ms #(smc-run e tok prompt-tokens)) (* particles smc-tokens))
              (when (= "gen.array (MLX)" label) (println "|" (pr-str (ae/stats e)) "| | |"))))))))
  (println "\nMLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
  (shutdown-agents))
