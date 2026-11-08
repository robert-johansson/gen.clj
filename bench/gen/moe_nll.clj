(ns gen.moe-nll
  "Does float16 make a mixture of experts predict worse, or only
   differently? (genclj-fxag) The mean negative log-likelihood of a text's
   tokens (each predicted from the ones before it, in windows of
   gen.bench.window tokens, 256, read fresh) by gen.llm.models.qwen35 on
   MLX's CUDA in float32, in float16, and in float16 with the router's
   product in float16 (f16-old, the code before genclj-fxag: its weights
   and logits rounded to float16), paired by token against float32: the
   mean difference and its standard error over the windows (their means'
   spread). The text: the files of gen.bench.texts (default README.md and
   DESIGN.md, read in order and cut to gen.bench.windows windows, 16).

   Measured on fenrir (2026-11-08, Qwen3.6-35B-A3B Q4_K_M, 16 windows of
   256, 4,080 tokens predicted): float32 2.9683 nats a token; float16
   -0.0039 ± 0.0020 from it, f16-old -0.0025 ± 0.0014. A token's |Δ| at
   the median 0.016 (0.019), p90 0.10 (0.11), at most 0.87 (0.86).
   Float16 predicts as well as float32: its routing flips (bench/gen/
   moe_routing.clj) move single predictions, not their quality. On Metal
   (the MacBook M2 Max, 2026-11-08, the same text and windows): float32
   2.9684, float16 -0.0025 ± 0.0015 from it, f16-old -0.0046 ± 0.0018; a
   token's |Δ| at the median 0.016 (0.019), p90 0.10 (0.12), at most 0.79
   (0.84).

     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.moe-nll <qwen35moe GGUF>

   One GPU JVM at a time (llama.cpp's library for the tokenizer only)."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen35 :as qwen35]))

(defn- native [ns sym] @(requiring-resolve (symbol ns (name sym))))

(defn- old-router
  "The router before genclj-fxag: its product in the model's dtype."
  [dtype router xs]
  (array/astype (common/linear xs (common/cast-weight 'bench router dtype)) :float32))

(defn- nll
  "-log p of each of window's tokens after the first, by model m: a
   double[] of (count window) - 1."
  [m window]
  (let [[lg] (qwen35/prefill m [window])
        [_ t v] (array/shape lg)
        lp (array/log-softmax (array/astype (array/reshape lg [t v]) :float32))
        next-tokens (array/array (mapv vector (rest window)) {:dtype :int32})]
    (mapv #(- (double %)) (array/->floats (array/take-along-axis (array/slice lp 0 0 (dec (long t))) next-tokens -1)))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

(defn- mean-se
  "The mean of the windows' mean values and its standard error."
  [per-window]
  (let [ms (mapv mean per-window)
        mu (mean ms)
        n (count ms)
        var (/ (reduce + (map #(let [d (- % mu)] (* d d)) ms)) (dec n))]
    [mu (Math/sqrt (/ var n))]))

(defn -main [model & _]
  (let [window (parse-long (System/getProperty "gen.bench.window" "256"))
        windows (parse-long (System/getProperty "gen.bench.windows" "16"))
        files (str/split (System/getProperty "gen.bench.texts" "README.md,DESIGN.md") #",")
        text (str/join "\n\n" (map slurp files))
        tokens (with-open [^java.io.Closeable v ((native "gen.llm.llama" 'vocabulary) model)]
                 ((native "gen.llm.llama" 'tokenize) v text))
        ws (vec (take windows (partition window tokens)))
        _ (println "backend" (array/backend-name array/*backend*) (array/device array/*backend*) "model" model
                   (format "text %d tokens, %d windows of %d" (count tokens) (count ws) window))
        run (fn [m redefs] (with-redefs-fn redefs #(mapv (fn [w] (nll m (vec w))) ws)))
        f32 (let [m (qwen35/load model {:dtype :float32})] (run m {}))
        _ ((native "gen.backend.mlx" 'collect!))
        m16 (qwen35/load model {:dtype :float16})
        variants {"f16" (run m16 {})
                  "f16-old" (run m16 {#'qwen35/router-logits old-router})}
        report (fn [label per-window]
                 (let [[mu se] (mean-se per-window)
                       diffs (mapv (fn [a b] (mapv - a b)) per-window f32)
                       [dmu dse] (mean-se diffs)
                       abs (sort (map #(Math/abs (double %)) (apply concat diffs)))]
                   (println (format "%-8s NLL %.4f ± %.4f nats a token; against float32 %+.4f ± %.4f; |Δ| a token median %.3f, p90 %.3f, max %.3f"
                                    label mu se dmu dse (nth abs (quot (count abs) 2)) (nth abs (long (* 0.9 (count abs)))) (last abs)))))]
    (report "f32" f32)
    (doseq [[k v] variants] (report k v))
    (shutdown-agents)))
