(ns gen.prefill-bench
  "Prompt admission's forwards on MLX (genclj-6qb6): the family's
   compiled paged batch step (gen.llm.array-engine's continuous mode)
   reading K prefill chunks of t rows each, fresh prompts from position 0,
   and the attention part alone (every attention layer's paged-write and
   gen.array/paged-attention of a run, the composite: each sequence's
   blocks gathered, then sdpa):

     clojure -J-Dgen.backend=mlx -M:bench -m gen.prefill-bench <Qwen3 or Qwen3.5 GGUF>

   One GPU JVM at a time. For each [K t] of gen.bench.shapes
   (1x32,1x64,2x32,4x32,8x32,4x64), ms a forward (the median of
   gen.bench.runs, 10, after 3 untimed, each evaluated) and a chunk, and
   the attention alone; then K chunks of one forward against K forwards
   of one chunk. gen.bench.dtype (float32).
   Measured on the Mac mini M4 (2026-11-08, genclj-6qb6), ms: Qwen3-0.6B
   Q8_0 in float16, one chunk of 32 rows 18.8, of 64 27.8; 8 chunks of
   32 in one forward 84.6 against 150.7 in eight (float32 98.4 against
   166.0); the attention alone (28 layers' writes, gathers and sdpa) 1.8
   to 7.1, 8 to 10% of the forward. Qwen3.5-0.8B Q8_0 in float16: 22.9,
   33.4; 8 x 32 111.1 against 183.4 (float32 138.5 against 200.0); its 6
   attention layers 0.6 to 1.6, 1 to 3%. So a prefill kernel over paged
   keys would gain at most those few ms: the chunks' matrix products are
   the forward."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))

(def ^:private settings
  {:shapes (mapv (fn [s] (mapv parse-long (str/split s #"x"))) (str/split (prop "shapes" "1x32,1x64,2x32,4x32,8x32,4x64") #","))
   :runs (parse-long (prop "runs" 10))
   :dtype (keyword (prop "dtype" "float32"))})

(def ^:private bs 16)

(defn- qwen35? [m] (contains? #{"qwen35" "qwen35moe"} (get-in m [:config :arch])))

(defn- load-model [path opts]
  (if (#{"qwen35" "qwen35moe"} (get-in (gguf/header path) [:metadata "general.architecture"]))
    (qwen35/load path opts)
    (qwen3/load path opts)))

(defn- i32 [xs] (array/array (vec xs) {:dtype :int32}))

(defn- batch
  "K fresh prompts of t tokens, chunk j's blocks 1 + j·t/bs …"
  [^long k ^long t qwen35?]
  (let [mb (quot t bs)
        blk (fn [^long j ^long p] (+ 1 (* j mb) (quot p bs)))]
    (cond-> {:tokens (vec (for [j (range k) p (range t)] [(+ 100 (mod (* 7 (+ j p)) 900))]))
             :positions (i32 (for [_ (range k) p (range t)] p))
             :slots (i32 (for [j (range k) p (range t)] (+ (* bs (long (blk j p))) (rem p bs))))
             :chunk (i32 (for [j (range k)] (range (+ 1 (* j mb)) (+ 1 (* (inc j) mb)))))
             :out (i32 (for [j (range k)] (+ (* j t) (dec t))))}
      qwen35? (assoc :chunk-len (i32 (repeat k t))))))

(defn- median-ms
  "The median ms of (f), evaluated, over the runs, after 3."
  [f]
  (dotimes [_ 3] (f))
  (let [ts (sort (repeatedly (:runs settings) #(let [t0 (System/nanoTime)] (f) (/ (- (System/nanoTime) t0) 1e6))))]
    (nth ts (quot (count ts) 2))))

(defn- forward-fn
  "(f) one forward of k chunks of t rows, its pools threaded."
  [m step ^long k ^long t]
  (let [q35? (qwen35? m)
        nb (+ 2 (* k (quot t bs)))
        pools (volatile! (if q35? (qwen35/paged-pools m nb bs) (qwen3/paged-pools m nb bs)))
        b (batch k t q35?)]
    (array/eval! @pools)
    (fn []
      (if q35?
        (let [states (mapv #(if (contains? % :state) {:chunk-conv (:conv %) :chunk-state (:state %)} {})
                           (qwen35/paged-states m k))
              {:keys [logits layers]} (step (mapv merge @pools states) b)]
          (array/eval! [logits layers])
          (vreset! pools (mapv #(select-keys % [:k :v]) layers)))
        (let [{:keys [logits] :as r} (step @pools b)]
          (array/eval! [logits (:pools r)])
          (vreset! pools (:pools r)))))))

(defn- attention-fn
  "(f) the attention layers' writes and runs alone for k chunks of t rows."
  [{:keys [config]} ^long k ^long t]
  (let [{:keys [kinds layers heads kv-heads head-dim dtype]} config
        n (if kinds (count (filter #{:attention} kinds)) layers)
        nb (+ 2 (* k (quot t bs)))
        pools (vec (repeat n (first (filter :k (qwen3/paged-pools {:config (assoc config :layers 1)} nb bs)))))
        b (batch k t false)
        [k1 k2 k3] (rng/split (rng/key 3) 3)
        q (array/astype (rng/normal k1 [(* k t) heads head-dim]) dtype)
        kk (array/astype (rng/normal k2 [(* k t) kv-heads head-dim]) dtype)
        v (array/astype (rng/normal k3 [(* k t) kv-heads head-dim]) dtype)
        f (array/compile (fn [pools q kk v slots chunk positions]
                           (mapv (fn [{kp :k vp :v}]
                                   (let [[kp vp] (array/paged-write kp vp kk v slots)]
                                     (array/paged-attention (array/reshape q [k t heads head-dim]) kp vp chunk
                                                            (array/reshape (array/+ positions 1) [k t]))))
                                 pools)))]
    (array/eval! [pools q kk v])
    (fn [] (array/eval! (f pools q kk v (:slots b) (:chunk b) (:positions b))))))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (let [m (load-model path {:dtype (:dtype settings)})
        step ((if (qwen35? m) qwen35/compile-paged-batch-step qwen3/compile-paged-batch-step) m {:donate? true})]
    (println "## Prompt admission's forwards (genclj-6qb6)\n")
    (println "- model:" path)
    (println "- settings:" (pr-str settings) "\n")
    (println "| K chunks x t rows | forward ms | ms a chunk | attention alone ms |")
    (println "|---|---:|---:|---:|")
    (let [rows (into {} (for [[k t] (:shapes settings)]
                          (let [fwd (median-ms (forward-fn m step k t))
                                att (median-ms (attention-fn m k t))]
                            (println (format "| %d x %d | %.1f | %.1f | %.1f |" k t fwd (/ fwd k) att))
                            [[k t] fwd])))]
      (println "\n| K chunks of t | one forward ms | K forwards of one chunk ms |")
      (println "|---|---:|---:|")
      (doseq [[k t] (:shapes settings)
              :when (and (< 1 (long k)) (rows [1 t]))]
        (println (format "| %d x %d | %.1f | %.1f |" k t (rows [k t]) (* (long k) (double (rows [1 t]))))))))
  (shutdown-agents))
