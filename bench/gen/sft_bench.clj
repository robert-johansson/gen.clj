(ns gen.sft-bench
  "A LoRA SFT step's peak memory and time on MLX under MLX's memory
   limit (gen.llm.sft, genclj-az5s):

     clojure -J-Dgen.backend=mlx -J-Dgen.mlx.grad=native -M:bench -m gen.sft-bench <Qwen3 GGUF>

   One GPU JVM at a time. The step: sft/step of a rank-8 adapter on q, k,
   v, o, gate, up and down of the last 8 blocks, AdamW, on a batch of
   gen.bench.rows (4) rows of gen.bench.length (256) tokens, half of each
   row's targets trained on, in gen.bench.accumulate (1) batches; its
   blocks checkpointed or not (:checkpoint?), its output projection in
   vocabulary chunks of gen.bench.chunk (16384). Each row: one untimed
   step, then for each limit a step from the same trainer (its arrays
   evaluated, the JVM's dead arrays collected and MLX's cache cleared
   first), MLX's peak memory past what it held before (the weights, the
   adapter and the optimizer's state), and its time. The limits:
   - default: gen.mlx.memory-bytes as configured (half of RAM);
   - tight: what MLX held before the step, so its eval loop waits for
     queued kernels whenever it holds more (the graph's own peak);
   - measured: the held memory plus 1.25 times the tight peak, a limit
     set from a measured first step."
  (:require [gen.array :as array]
            [gen.backend.mlx.memory :as memory]
            [gen.learning :as learn]
            [gen.llm.lora :as lora]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.sft :as sft]
            [gen.random :as rng])
  (:import [gen.mlx Mlx]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(defn- mb [bytes] (format "%,.0f MB" (/ (double bytes) 1e6)))

(defn- batch-of
  "A batch of b rows of t tokens (deterministic ids), every other target
   trained on."
  [vocab b t]
  (let [tokens (vec (for [r (range b)] (vec (for [i (range t)] (mod (+ 7 (* 131 r) (* 977 i)) vocab)))))]
    {:tokens (array/array tokens {:dtype :int32})
     :mask (array/array (vec (for [_ (range b)] (vec (for [i (range (dec t))] (if (odd? i) 1.0 0.0))))))
     :count (* b (quot (dec t) 2))}))

(defn- measure
  "{:peak :ms}: one step of trainer on batches at MLX's memory limit
   (limit-of before), before the bytes MLX holds at the step's start."
  [m trainer batches opts limit-of]
  (array/eval! [trainer batches])
  (memory/collect!)
  (Mlx/mlx_clear_cache)
  (let [before (Mlx/mlx_get_active_memory)
        limit (Mlx/mlx_get_memory_limit)]
    (Mlx/mlx_reset_peak_memory)
    (when-let [l (limit-of before)] (memory/set-memory-limit! (long l)))
    (try
      (let [t0 (System/nanoTime)
            [t] (sft/step m trainer batches opts)]
        (array/eval! t)
        {:ms (/ (- (System/nanoTime) t0) 1e6) :peak (- (Mlx/mlx_get_peak_memory) before) :before before})
      (finally (memory/set-memory-limit! limit)))))

(defn -main [& [gguf]]
  (when-not gguf (throw (ex-info "gen.sft-bench takes a Qwen3 GGUF" {})))
  (println "gen.sft-bench:" (System/getProperty "gen.mlx.grad" "tape") "gradients,"
           ((requiring-resolve 'gen.backend.mlx/device) array/*backend*)
           "; MLX's memory limit" (mb (Mlx/mlx_get_memory_limit)))
  (let [m (qwen3/load gguf)
        n (long (:layers (:config m)))
        b (long-prop "rows" 4)
        t (long-prop "length" 256)
        k (long-prop "accumulate" 1)
        chunk (long-prop "chunk" 16384)
        ad (lora/init m {:rank 8 :targets (vec (keys lora/targets)) :layers (range (- n 8) n)} (rng/key 1))
        tr (sft/trainer ad (learn/clip-norm (learn/adamw {:lr 1e-4}) {:max-norm 1.0}) (rng/key 2))
        batches (vec (repeat k (batch-of (:vocab (:config m)) b t)))]
    (array/eval! (:weights m))
    (println (format "%d batch(es) of %d rows of %d tokens, vocabulary chunks of %d; adapter %s"
                     k b t chunk (mb (* 4 (reduce + (map array/size (array/tree-leaves (:params ad))))))))
    (println (format "  %-12s %-9s %12s %10s" "checkpoint?" "limit" "peak" "ms"))
    (doseq [ck [false true]
            :let [opts {:checkpoint? ck :chunk chunk}]]
      (sft/step m tr batches opts)
      (let [tight (measure m tr batches opts identity)
            measured-limit (fn [before] (+ before (* 1.25 (:peak tight))))
            rows [["default" (measure m tr batches opts (constantly nil))]
                  ["tight" (measure m tr batches opts identity)]
                  ["measured" (measure m tr batches opts measured-limit)]]]
        (doseq [[what {:keys [peak ms]}] rows]
          (println (format "  %-12s %-9s %12s %10.0f" ck what (mb peak) ms)))
        (println (format "  (held before the step: %s)" (mb (:before tight)))))))
  (shutdown-agents))
