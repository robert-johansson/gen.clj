(ns gen.qwen35-continuous-noise
  "The band of gen.array's engine on Qwen3.5 (the tiny random Qwen3.5 and
   Qwen3.5-MoE: gen.llm.array-engine-qwen35-continuous-test's and
   array-engine-qwen35-paged-test's), measured on the backend under test:
   the |Δ log p| between two computations of one distribution (the
   Branching contract's statistic, over the reference's 64 most likely
   tokens; a score's |Δ| at its token):
   - the continuous test's concurrent clients (6, on a fresh engine each
     of -Dgen.bench.runs runs, default 10): alone and concurrently against
     qwen35's forward, and concurrently against alone;
   - more clients alone (-Dgen.bench.extra, default 20: contexts to 144
     tokens, the test's to 44), with their largest rows' lengths;
   - every comparison of the two test namespaces, each run
     -Dgen.bench.test-runs times (default 3) with its band 0, so that each
     reports its |Δ| as a failure;
   and then each of the continuous test's planted bugs' largest |Δ|,
   which the band must stay under.

   Run on MLX (one GPU JVM at a time):
     clojure -J-Dgen.backend=mlx -M:test:bench -m gen.qwen35-continuous-noise"
  (:require [clojure.test :as test]
            [gen.array :as array]
            [gen.llm.array-engine-qwen35-continuous-test :as t]
            [gen.llm.array-engine-qwen35-paged-test]
            [gen.llm.branching-contract :as bc]))

(def ^:private engine #'t/engine)
(def ^:private client @#'t/client)
(def ^:private clients @#'t/clients)
(def ^:private forward (memoize (fn [k ctx] (@#'t/forward-row k ctx))))

(defn- row-gaps
  "|Δ| of each row and score of a client's result r against the forward."
  [k r]
  (concat (map (fn [[ctx lp]] (bc/max-diff (forward k ctx) lp 64)) (:seen r))
          (map (fn [[ctx tok s]] (Math/abs (- (double (aget ^floats (forward k ctx) (int tok))) (double s)))) (:scores r))))

(defn- alone-gaps
  "|Δ| of each row and score of a concurrent result r against its run alone a."
  [a r]
  (concat (map (fn [[[_ l1] [_ l2]]] (bc/max-diff l1 l2 64)) (map vector (:seen a) (:seen r)))
          (map (fn [[[_ _ s1] [_ _ s2]]] (Math/abs (- (double s1) (double s2)))) (map vector (:scores a) (:scores r)))))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)
        q (fn [p] (if (zero? n) 0.0 (v (min (dec n) (long (Math/floor (* (double p) n)))))))]
    (format "n %4d  p50 %.3g  p90 %.3g  p99 %.3g  max %.3g" n (q 0.5) (q 0.9) (q 0.99) (q 1.0))))

(defn- clients-gaps
  "The continuous test's clients' |Δ| for model k: {what [|Δ| …]}, printed
   as measured."
  [k runs extra]
  (let [all (atom {})
        record! (fn [what xs]
                  (swap! all update what (fnil into []) xs)
                  (println (format "  %-26s %s" (name what) (summary xs)))
                  (flush))]
    (dotimes [_ runs]
      (with-open [e (engine k)]
        (let [[alone concurrent] (clients e 6)]
          (record! :alone-vs-forward (mapcat #(row-gaps k %) alone))
          (record! :concurrent-vs-forward (mapcat #(row-gaps k %) concurrent))
          (record! :concurrent-vs-alone (mapcat alone-gaps alone concurrent)))))
    (with-open [e (engine k {:max-branches 80})]
      (let [rs (mapv #(client e %) (range 6 (+ 6 (long extra))))
            by-len (for [r rs [ctx lp] (:seen r)] [(count ctx) (bc/max-diff (forward k ctx) lp 64)])]
        (record! :more-clients-vs-forward (mapcat #(row-gaps k %) rs))
        (println "    largest rows [length |Δ|]:" (pr-str (take 8 (sort-by (comp - second) by-len))))
        (println "    max |Δ| by length (from):"
                 (pr-str (into (sorted-map) (map (fn [[b xs]] [(* 16 (long b)) (format "%.2g" (reduce max 0.0 (map second xs)))]))
                               (group-by #(quot (long (first %)) 16) by-len))))))
    @all))

(def ^:private number "([0-9.]+(?:[eE][-+]?[0-9]+)?)")

(defn- reported
  "The |Δ| values a failure report m names: a contract's or a client's
   message, or an (is (<= |Δ| band)) with the band 0."
  [m]
  (let [s (pr-str (:actual m))]
    (map #(Double/parseDouble (second %))
         (concat (re-seq (re-pattern (str "\\|Δ(?: log p)?\\| =? ?" number)) s)
                 (re-seq (re-pattern (str "\\(<= " number " 0\\.0\\)")) s)))))

(defn- comparisons
  "Every |Δ| the test namespace ns-sym's comparisons report with its band
   (and its contract options' band) 0: {deftest [|Δ| …]}."
  [ns-sym]
  (let [seen (atom {})
        band (ns-resolve ns-sym 'band)
        opts (ns-resolve ns-sym 'contract-opts)]
    (binding [test/report (fn [m]
                            (case (:type m)
                              :fail (swap! seen update (:name (meta (first test/*testing-vars*))) (fnil into []) (reported m))
                              :error (println "  error in" (first test/*testing-vars*) (pr-str (:actual m)))
                              nil))]
      (with-redefs-fn {band 0.0 opts (assoc @opts :band 0.0)}
        #(test/test-ns ns-sym)))
    @seen))

(defn -main [& _]
  (let [runs (Long/getLong "gen.bench.runs" 10)
        extra (Long/getLong "gen.bench.extra" 20)
        test-runs (Long/getLong "gen.bench.test-runs" 3)
        device (when-not (= :jvm (array/backend-name array/*backend*))
                 ((requiring-resolve 'gen.backend.mlx/device) array/*backend*))
        _ (println "backend" (array/backend-name array/*backend*) "device" device "os" (System/getProperty "os.name")
                   "bands now" @#'t/band @(ns-resolve 'gen.llm.array-engine-qwen35-paged-test 'band))
        results (into {} (for [k [:dense :moe]]
                           (do (println "== the continuous test's clients," k) [k (clients-gaps k runs extra)])))
        tests (into {} (for [n ['gen.llm.array-engine-qwen35-continuous-test 'gen.llm.array-engine-qwen35-paged-test]]
                         (do (println "==" n "with its band 0," test-runs "runs")
                             [n (apply merge-with into (repeatedly test-runs #(comparisons n)))])))]
    (println "\n== the continuous test's planted bugs (dense, 4 clients): the largest |Δ| against the forward and alone")
    (doseq [[what redefs] (merge @#'t/planted (when device @#'t/planted-on-mlx))]
      (let [r (try (with-redefs-fn redefs
                     #(with-open [e (engine :dense)]
                        (let [[alone concurrent] (clients e 4)]
                          [(reduce max 0.0 (mapcat (fn [r] (row-gaps :dense r)) (concat alone concurrent)))
                           (reduce max 0.0 (mapcat alone-gaps alone concurrent))])))
                   (catch Throwable x (str "threw " (ex-message x))))]
        (println (format "  %-72s %s" what (if (string? r) r (apply format "forward %.3g  alone %.3g" r))))
        (flush)))
    (println "\n== overall")
    (doseq [[k m] results [what xs] (sort-by key m)]
      (println (format "  %-6s %-40s %s" (name k) (name what) (summary xs))))
    (doseq [[n m] tests [what xs] (sort-by key m)]
      (println (format "  %-47s %s" (str (last (re-find #"qwen35-(\w+)-test" (str n))) " " what) (summary xs))))
    (println (format "the tests' comparisons' largest: %.4g; with the more clients': %.4g"
                     (reduce max 0.0 (concat (for [[_ m] results [what xs] m :when (not= what :more-clients-vs-forward) x xs] x)
                                             (for [[_ m] tests [_ xs] m x xs] x)))
                     (reduce max 0.0 (concat (for [[_ m] results [_ xs] m x xs] x)
                                             (for [[_ m] tests [_ xs] m x xs] x)))))
    (shutdown-agents)))
