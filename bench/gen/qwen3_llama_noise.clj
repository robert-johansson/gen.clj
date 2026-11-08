(ns gen.qwen3-llama-noise
  "A forward over gen.array (gen.llm.models.qwen3, or qwen35 for a Qwen3.5
   or Qwen3.6 GGUF, by its architecture) against the llama.cpp engine on
   the same GGUF, for the band of gen.llm.models.qwen3-llama-test (and
   qwen35-llama-test), measured as gen.llama-noise measures llama.cpp's
   own: the Branching contract's statistic, max |Δ log p| over llama.cpp's
   64 most likely tokens, on gen.llama-noise's prompts (its set,
   -Dgen.bench.prompts: general, the default, its 20 prompts; digits; or
   all, each set apart; the first -Dgen.bench.prompt-limit of each), each
   read whole, as a prefill and a chunk, and through 8 greedy steps
   (gen.llm.models.against-llama). Then the planted convention errors'
   differences (against-llama's mutations-of the model) on the general
   prompts (the least of them is their margin), unless
   -Dgen.bench.mutations=false.

   On a mixture of experts (unless -Dgen.bench.margins=false), the same
   again by the readings' routing: a reading's margin is the least log
   p_k/p_k+1 of the forward's router (its k-th and (k+1)-th experts'
   probabilities) over every layer and position its forwards read (a
   step's, the prompt's too), and the gaps are given over the readings of
   margin at least θ, for each θ of margin-thresholds, the planted errors'
   too (each measured by its own forward's margins). Where llama.cpp's
   router and ours tie, a float's difference picks other experts (as
   float16 does against float32, bench/gen/moe_routing.clj), and
   llama.cpp's routing cannot be read to force it on the forward (as
   against-llama's :forced does against a model).

   One model in memory at a time: llama.cpp's readings of every prompt
   are taken first (against-llama/reference) and its engine closed, then
   the forward's model loaded and compared with them (compare-readings),
   so that a model as large as Qwen3.6-35B-A3B (llama.cpp's 21 GB and the
   forward's 28 GB) runs on a 64 GB Mac. Or in two JVMs:
   -Dgen.bench.readings-out=<file> writes llama.cpp's readings (each
   distribution's 256 most likely tokens, as EDN) and exits, and
   -Dgen.bench.readings-in=<file> compares the forward with them, without
   llama.cpp's engine (a reading's other tokens at -1e30: the statistic
   reads llama.cpp's top 64, and a token of the forward's top 5 outside
   its 256 differs from them by more than any tie).

   Run on MLX (with native/build-mlx.sh's and native/build-llama.sh's
   libraries, one GPU JVM at a time):
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-llama-noise Qwen3-0.6B-Q8_0.gguf
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-llama-noise Qwen3.5-0.8B-Q8_0.gguf"
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [gen.array :as array]
            [gen.llama-noise :as noise]
            [gen.llm.branching-contract :refer [top-tokens]]
            [gen.llm.llama :as llama]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.qwen35 :as qwen35]))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)]
    (format "n %4d  median %.2e  p99 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.99 n)))) (peek v))))

(defn- seconds [t0] (/ (- (System/nanoTime) (long t0)) 1e9))

(defn- llama-readings
  "llama.cpp's readings (against-llama/reference's, 8 steps) of each set's
   prompts, its engine of the model at path closed on return: {set-name
   [readings]}."
  [path sets]
  (with-open [e (llama/engine {:model path :max-branches 4 :n-ctx 1024})]
    (println (llama/system-info))
    (let [t0 (System/nanoTime)
          out (into {} (for [[set-name items] sets]
                         [set-name (mapv #(against/reference e (:tokens %) 8) items)]))]
      (println (format "llama.cpp's readings: %.1f s" (seconds t0)))
      out)))

;; llama.cpp's readings in a file (gen.bench.readings-out, -in): each
;; distribution its 256 most likely tokens, the rest -1e30 when read back
(def ^:private kept 256)

(defn- sparse [^floats lp]
  (let [ts (top-tokens lp kept)]
    [ts (mapv #(double (aget lp (int %))) ts)]))

(defn- dense ^floats [vocab [ts vs]]
  (let [a (float-array vocab (float -1e30))]
    (doseq [[t v] (map vector ts vs)] (aset a (int t) (float v)))
    a))

(defn- write-readings!
  "Writes llama.cpp's readings refs ({set-name [readings]}) to file."
  [file model refs]
  (spit file (pr-str {:model model
                      :vocab (alength ^floats (:root (first (val (first refs)))))
                      :sets (update-vals refs (fn [rs] (mapv (fn [{:keys [tokens root steps]}]
                                                               {:tokens tokens :root (sparse root)
                                                                :steps (mapv (fn [[t lp]] [t (sparse lp)]) steps)})
                                                             rs)))})))

(defn- read-readings
  "The readings write-readings! wrote to file, of the prompts sets (their
   tokens checked): {set-name [readings]}."
  [file sets]
  (let [{:keys [vocab] :as r} (edn/read-string (slurp file))
        refs (update-vals (:sets r) (fn [rs] (mapv (fn [{:keys [tokens root steps]}]
                                                     {:tokens tokens :root (dense vocab root)
                                                      :steps (mapv (fn [[t s]] [t (dense vocab s)]) steps)})
                                                   rs)))]
    (doseq [[set-name items] sets]
      (when-not (= (mapv :tokens items) (mapv :tokens (get refs set-name)))
        (throw (ex-info (str "the readings in " file " are not of the " set-name " prompts' tokens") {:file file :set set-name}))))
    refs))

(defn- against-llama
  "The forward of m against llama.cpp's readings rs of the prompts items
   ({:text :tokens}), printed."
  [m items rs]
  (let [rows (mapv (fn [{:keys [text tokens]} r] {:text text :n (count tokens) :results (against/compare-readings m r 0)})
                   items rs)
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
                       (pr-str (subs text 0 (min 40 (count text)))))))
    (flush)))

;; The router's margins on a mixture of experts (gen.bench.margins)

(def ^:private router-top-k @#'qwen35/top-k)

(def ^:private margin-thresholds [0.0 1e-4 1e-3 1e-2 3e-2 1e-1])

(defn- least-margin
  "The least log p_k/p_k+1 over the rows (of e) of probs, a float[] of
   the router's probabilities, k the experts used: how near its top k is
   to a tie."
  ^double [^floats probs e k]
  (let [e (long e) k (long k)]
    (reduce min Double/POSITIVE_INFINITY
            (for [r (range (quot (alength probs) e))]
              (let [^floats row (java.util.Arrays/copyOfRange probs (int (* r e)) (int (* (inc r) e)))]
                (java.util.Arrays/sort row)
                (Math/log (/ (aget row (int (- e k))) (aget row (int (- e k 1))))))))))

(defn- with-margins
  "[(f)'s result, the least margin (least-margin's) of each forward f
   makes, a prefill or a step of qwen35 model m, in order]."
  [m f]
  (let [{:keys [experts experts-used]} (:config m)
        groups (atom [])
        mark (fn [g] (fn [& args] (swap! groups conj []) (apply g args)))
        out (with-redefs-fn {#'qwen35/prefill (mark qwen35/prefill)
                             #'qwen35/step (mark qwen35/step)
                             #'qwen35/top-k (fn [p k]
                                              (swap! groups #(conj (pop %) (conj (peek %) p)))
                                              (router-top-k p k))}
              f)
        ;; a lower arity calls the full one through its var: an outer
        ;; call's group stays empty
        gs (filterv seq @groups)]
    (array/eval! (vec (apply concat gs)))
    [out (mapv (fn [ps] (reduce min Double/POSITIVE_INFINITY (map #(least-margin (array/->floats %) experts experts-used) ps))) gs)]))

(defn- tie-free
  "compare-readings of m against readings, each result with its :margin:
   the least of the forwards it reads (compare-readings': the prompt
   whole; its first half and the chunk after; the steps after the whole
   prompt)."
  [m readings]
  (let [[rs g] (with-margins m #(against/compare-readings m readings 0))
        n (count (:steps readings))]
    (when-not (= (count g) (+ 3 n))
      (throw (ex-info "tie-free: compare-readings' forwards are not the prompt's, its halves' and the steps'" {:forwards (count g)})))
    (mapv #(assoc %1 :margin %2) rs
          (into [(g 0) (min (g 1) (g 2))] (map (fn [i] (reduce min (g 0) (subvec g 3 (+ 3 i)))) (range 1 (inc n)))))))

(defn- median [xs] (let [v (vec (sort xs))] (v (quot (count v) 2))))

(defn- margin-report
  "The gaps of readings by their router's margin, then the planted
   errors' (on the general prompts, whole, chunk and one step), printed."
  [m sets refs]
  (println "
by the router's margin: the readings whose forwards' least log p_k/p_k+1 is at least θ")
  (doseq [[set-name items] sets]
    (let [rows (mapv #(tie-free m %) (get refs set-name))
          all (vec (apply concat rows))]
      (println (str "  the " set-name " prompts (" (count items) "):"))
      (doseq [θ margin-thresholds
              :let [kept (filter #(>= (double (:margin %)) (double θ)) all)
                    prompts (count (filter (fn [r] (some #(>= (double (:margin %)) (double θ)) r)) rows))]]
        (println (format "    θ %.0e: %3d of %d readings (%2d prompts): max %.3e, median %.3e" θ (count kept) (count all) prompts
                         (if (seq kept) (reduce max (map :max-diff kept)) Double/NaN)
                         (if (seq kept) (median (map :max-diff kept)) Double/NaN))))
      (println "    the 12 largest gaps (gap, margin, reading):")
      (doseq [r (take 12 (sort-by (comp - :max-diff) all))]
        (println (format "      %.3e  %.2e  %s" (:max-diff r) (:margin r) (pr-str (:what r)))))
      (flush)))
  (when (and (not= "false" (System/getProperty "gen.bench.mutations")) (get refs "general"))
    (println "  the planted errors on the general prompts, by θ: the least and the median over the prompts of each one's"
             "worst gap on its kept readings (whole, chunk, one step; the mutated forward's margins), and the prompts kept:")
    (doseq [mutation (sort (keys (against/mutations-of m)))]
      (let [rows (with-redefs-fn ((against/mutations-of m) mutation)
                   #(mapv (fn [r] (tie-free m (update r :steps (fn [s] (vec (take 1 s)))))) (get refs "general")))]
        (println (format "    %-24s %s" (name mutation)
                         (str/join "  " (for [θ margin-thresholds
                                              :let [ws (keep (fn [r] (when-let [k (seq (filter #(>= (double (:margin %)) (double θ)) r))]
                                                                      (reduce max (map :max-diff k))))
                                                             rows)]]
                                          (if (seq ws)
                                            (format "%.0e: %.3f %.3f (%d)" θ (reduce min ws) (median ws) (count ws))
                                            (format "%.0e: - (0)" θ))))))
        (flush)))))

(defn- compare-all
  "The forward of the model at path against llama.cpp's readings refs
   ({set-name [readings]}) of the prompts sets, then the planted errors,
   printed."
  [path sets refs]
  (let [t0 (System/nanoTime)
        m (against/load-model path)]
    (println (format "the forward's model loaded in %.1f s" (seconds t0)))
    (doseq [[set-name items] sets]
      (println (str "\n== the " set-name " prompts (" (count items) ")"))
      (against-llama m items (get refs set-name)))
    (when-not (= "false" (System/getProperty "gen.bench.mutations"))
      (if-let [general (get refs "general")]
        (do (println "\nplanted convention errors, the worst |Δ log p| of each general prompt (whole, chunk, one step):")
            (doseq [mutation (sort (keys (against/mutations-of m)))]
              (let [ds (mapv #(against/mutated m % mutation) general)]
                (println (format "  %-24s min %.2f  median %.2f  max %.2f nats" (name mutation)
                                 (reduce min ds) (nth (sort ds) (quot (count ds) 2)) (reduce max ds)))
                (flush))))
        (println "\n(the planted errors are measured on the general prompts: -Dgen.bench.prompts=general or all)")))
    (when (and (get-in m [:config :experts]) (not= "false" (System/getProperty "gen.bench.margins")))
      (margin-report m sets refs))))

(defn -main [model & _]
  (println "backend" (array/backend-name array/*backend*) "model" model "architecture" (against/architecture model))
  (let [sets (with-open [v (llama/vocabulary model)] (noise/prompt-sets v))
        in (System/getProperty "gen.bench.readings-in")
        out (System/getProperty "gen.bench.readings-out")
        refs (if in (read-readings in sets) (llama-readings model sets))]
    (if out
      (do (write-readings! out model refs)
          (println "llama.cpp's readings written to" out))
      (compare-all model sets refs))
    (shutdown-agents)))
