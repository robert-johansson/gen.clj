(ns gen.cuda-reference-noise
  "MLX's forward on CUDA against the same forward on MLX's CPU, on the
   same GGUF (gen.llm.models.qwen3, or qwen35 for a Qwen3.5 or Qwen3.6, by
   its architecture), for the bands of gen.llm.models.cuda-reference-test
   (genclj-gddh): the Branching contract's statistic, max |Δ log p| over
   the CPU's 64 most likely tokens, on gen.llama-noise's prompts (its set,
   -Dgen.bench.prompts: general, the default; digits; or all, each set
   apart), each read whole, as a prefill and a chunk, and through 8 greedy
   steps, the CPU's argmax (gen.llm.models.against-llama's reference and
   compare-readings). CUDA's forward in each dtype of -Dgen.bench.dtypes
   (default float32,float16) against the CPU's in each of
   -Dgen.bench.ref-dtypes (default float32: MLX's CPU computes float16
   about 15 times slower than float32 on fenrir, one thread). Then the
   planted convention errors' differences in each dtype (against-llama's
   mutations-of the model; the least of them is their margin) on the
   general prompts, against the CPU in the same dtype (else the first of
   ref-dtypes), unless -Dgen.bench.mutations=false. One model at a time in
   memory: the CPU's readings are taken, its model dropped, then CUDA's
   loaded.

   Measured on fenrir (2026-11-08, -Dgen.bench.prompts=all): the bands and
   planted errors in gen.llm.models.cuda-reference-test's docstring. The
   CPU's readings of the 40 prompts took 258 s (Qwen3-0.6B), 322 s
   (Qwen3.5-0.8B) and 1373 s (Qwen3.6-35B-A3B, about 31 GB resident).

   Run on MLX's CUDA (llama.cpp's library for the tokenizer only, its
   vocabulary loaded alone), one GPU JVM at a time:
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.cuda-reference-noise Qwen3-0.6B-Q8_0.gguf"
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.backend.mlx :as mlx]
            [gen.llama-noise :as noise]
            [gen.llm.llama :as llama]
            [gen.llm.models.against-llama :as against]))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)]
    (format "n %4d  median %.2e  p99 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.99 n)))) (peek v))))

(defn- seconds [t0] (/ (- (System/nanoTime) (long t0)) 1e9))

(defn- report
  "The comparisons rows (a vector of compare-readings' results a prompt,
   items' order) of the prompts items ({:text :tokens}), printed under
   title."
  [title items rows]
  (let [all (apply concat rows)
        kind (fn [r] (let [w (:what r)] (if (vector? w) :step w)))]
    (println (str "\n" title ": |Δ log p| over the reference's top 64 tokens, by reading:"))
    (doseq [[k rs] (sort-by key (group-by kind all))]
      (println (format "  %-6s %s" (name k) (summary (map :max-diff rs)))))
    (println (format "overall max: %.3e; argmax equal in %d of %d, top 5 in %d of %d"
                     (reduce max (map :max-diff all))
                     (count (filter :argmax? all)) (count all) (count (filter :top5? all)) (count all)))
    (println "by prompt (tokens: whole, chunk, worst step; argmax and top 5 all equal?):")
    (doseq [[{:keys [text tokens]} results] (map vector items rows)]
      (println (format "  %3d  %.2e %.2e %.2e  %s %s  %s" (count tokens)
                       (:max-diff (first results)) (:max-diff (second results))
                       (reduce max 0.0 (map :max-diff (drop 2 results)))
                       (every? :argmax? results) (every? :top5? results)
                       (pr-str (subs text 0 (min 40 (count text)))))))))

(defn- cpu-readings
  "The readings (against-llama/reference's, 8 steps) of each set's prompts
   by the model at path in dtype on the CPU backend cpu: {set-name
   [readings]}. The model is dropped on return."
  [path dtype cpu sets]
  (let [t0 (System/nanoTime)
        ref (array/with-backend cpu (against/load-model path {:dtype dtype}))
        _ (println (format "the CPU's model in %s loaded in %.1f s" (name dtype) (seconds t0)))
        t1 (System/nanoTime)
        out (into {} (for [[set-name items] sets]
                       (let [t (System/nanoTime)
                             rs (mapv #(against/reference ref (:tokens %) 8) items)]
                         (println (format "  the %s prompts: %.1f s" set-name (seconds t)))
                         (flush)
                         [set-name rs])))]
    (println (format "the CPU's readings in %s: %.1f s" (name dtype) (seconds t1)))
    out))

(defn- dtypes-prop [k default]
  (mapv keyword (str/split (System/getProperty k default) #",")))

(defn -main [model & _]
  (let [dtypes (dtypes-prop "gen.bench.dtypes" "float32,float16")
        ref-dtypes (dtypes-prop "gen.bench.ref-dtypes" "float32")
        cpu (mlx/make-backend {:device :cpu})]
    (println "backend" (array/backend-name array/*backend*) (array/device array/*backend*)
             "model" model "architecture" (against/architecture model) "CUDA's dtypes" dtypes "the CPU's" ref-dtypes)
    (with-open [v (llama/vocabulary model)]
      (let [sets (noise/prompt-sets v)
            refs (into {} (for [rd ref-dtypes]
                            (let [r (cpu-readings model rd cpu sets)]
                              (mlx/collect!)
                              [rd r])))]
        (doseq [dtype dtypes]
          (println (str "\n=== CUDA in " (name dtype)))
          (let [t0 (System/nanoTime)
                m (against/load-model model {:dtype dtype})
                _ (println (format "CUDA's model in %s loaded in %.1f s" (name dtype) (seconds t0)))
                t1 (System/nanoTime)]
            (doseq [rd ref-dtypes
                    [set-name items] sets]
              (report (str "CUDA " (name dtype) " against the CPU's " (name rd) ", the " set-name " prompts (" (count items) ")")
                      items (mapv #(against/compare-readings m % 0) (get-in refs [rd set-name]))))
            (println (format "CUDA's comparisons: %.1f s" (seconds t1)))
            (when-not (= "false" (System/getProperty "gen.bench.mutations"))
              (let [rd (if (contains? refs dtype) dtype (first ref-dtypes))]
                (if-let [general (get-in refs [rd "general"])]
                  (do (println (str "\nplanted convention errors in " (name dtype) " against the CPU's " (name rd)
                                    ", the worst |Δ log p| of each general prompt (whole, chunk, one step):"))
                      (doseq [mutation (sort (keys (against/mutations-of m)))]
                        (let [ds (mapv #(against/mutated m % mutation) general)]
                          (println (format "  %-24s min %.3f  median %.3f  max %.3f nats" (name mutation)
                                           (reduce min ds) (nth (sort ds) (quot (count ds) 2)) (reduce max ds))))))
                  (println "\n(the planted errors are measured on the general prompts: -Dgen.bench.prompts=general or all)")))))
          (mlx/collect!))))
    (shutdown-agents)))
