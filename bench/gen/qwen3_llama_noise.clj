(ns gen.qwen3-llama-noise
  "A forward over gen.array (gen.llm.models.qwen3, or qwen35 for a Qwen3.5
   or Qwen3.6 GGUF, by its architecture) against the llama.cpp engine on
   the same GGUF, for the band of gen.llm.models.qwen3-llama-test (and
   qwen35-llama-test), measured as gen.llama-noise measures llama.cpp's
   own: the Branching contract's statistic, max |Δ log p| over llama.cpp's
   64 most likely tokens, on gen.llama-noise's prompts (its set,
   -Dgen.bench.prompts: general, the default, its 20 prompts; digits; or
   all, each set apart), each read whole, as a prefill and a chunk, and
   through 8 greedy steps (gen.llm.models.against-llama). Then the planted
   convention errors' differences (against-llama's mutations-of the model)
   on the general prompts (the least of them is their margin), unless
   -Dgen.bench.mutations=false.

   Run on MLX (with native/build-mlx.sh's and native/build-llama.sh's
   libraries, one GPU JVM at a time):
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-llama-noise Qwen3-0.6B-Q8_0.gguf
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-llama-noise Qwen3.5-0.8B-Q8_0.gguf"
  (:require [gen.array :as array]
            [gen.llama-noise :as noise]
            [gen.llm.llama :as llama]
            [gen.llm.models.against-llama :as against]))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)]
    (format "n %4d  median %.2e  p99 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.99 n)))) (peek v))))

(defn- against-llama
  "The forward of m against e on the prompts items ({:text :tokens}),
   printed."
  [e m items]
  (let [rows (vec (for [{:keys [text tokens]} items]
                    {:text text :n (count tokens) :results (against/compare-prompt e m tokens 8 0)}))
        all (mapcat :results rows)
        kind (fn [r] (let [w (:what r)] (if (vector? w) :step w)))]
    (println "\n|Δ log p| over llama.cpp's top 64 tokens, by reading:")
    (doseq [[k rs] (sort-by key (group-by kind all))]
      (println (format "  %-6s %s" (name k) (summary (map :max-diff rs)))))
    (println (format "overall max: %.3e; argmax equal in %d of %d, top 5 in %d of %d"
                     (reduce max (map :max-diff all))
                     (count (filter :argmax? all)) (count all) (count (filter :top5? all)) (count all)))
    (println "by prompt (tokens: whole, chunk, worst step; argmax and top 5 all equal?):")
    (doseq [{:keys [text n results]} rows]
      (println (format "  %3d  %.2e %.2e %.2e  %s %s  %s" n
                       (:max-diff (first results)) (:max-diff (second results))
                       (reduce max (map :max-diff (drop 2 results)))
                       (every? :argmax? results) (every? :top5? results)
                       (pr-str (subs text 0 (min 40 (count text)))))))))

(defn -main [model & _]
  (println "backend" (array/backend-name array/*backend*) "model" model "architecture" (against/architecture model))
  (let [m (against/load-model model)]
    (with-open [e (llama/engine {:model model :max-branches 4 :n-ctx 1024})]
      (println (llama/system-info))
      (doseq [[set-name items] (noise/prompt-sets e)]
        (println (str "\n== the " set-name " prompts (" (count items) ")"))
        (against-llama e m items))
      (when-not (= "false" (System/getProperty "gen.bench.mutations"))
        (println "\nplanted convention errors, the worst |Δ log p| of each general prompt (whole, chunk, one step):")
        (doseq [mutation (sort (keys (against/mutations-of m)))]
          (let [ds (for [text noise/prompts] (against/mutated e m (llama/tokenize e text) mutation))]
            (println (format "  %-24s min %.2f  median %.2f  max %.2f nats" (name mutation)
                             (reduce min ds) (nth (sort ds) (quot (count ds) 2)) (reduce max ds)))))))
    (shutdown-agents)))
