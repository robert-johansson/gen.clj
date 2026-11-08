(ns gen.moe-routing
  "The router's choices of a mixture of experts (gen.llm.models.qwen35's
   qwen35moe) on MLX's CUDA against the same forward on MLX's CPU in
   float32 (genclj-fxag): which experts each layer's top k picks at each
   position, and how the next-token log-probabilities' gap tracks the
   flips (a position and layer whose top-k set differs from the CPU's).
   Instrumented by with-redefs of qwen35's private top-k and
   router-logits, not through the model's API.

   On gen.llama-noise's prompts (-Dgen.bench.prompts, all by default),
   each prompt is prefilled (the log-probabilities at every position) and
   stepped 8 times by the CPU's greedy tokens. The CPU in float32 first
   (its model then dropped), recording its router's probabilities [E] by
   layer and position, and the top 64 log-probabilities by position.
   Then CUDA, each variant of -Dgen.bench.variants (default all of them):
   - f32: CUDA in float32;
   - f32-forced: CUDA in float32, each layer's experts the CPU's (its own
     router's probabilities weighting them);
   - f16-old: float16 with the router's product in float16 (its weights
     and logits rounded to float16, the softmax float32): the code before
     genclj-fxag;
   - f16-old-forced: f16-old with the CPU's experts;
   - f16: float16 as qwen35 computes it (the router in float32);
   - f16-forced: f16 with the CPU's experts.
   For each: the flips' rate by layer; the gap (max |Δ log p| over the
   CPU's 64 most likely tokens) at positions with no flip up to them, a
   flip before only, and a flip at them; the readings' gap (the prompt's
   last position and the 8 steps: the reference test's :whole and steps);
   each prompt's. Unless -Dgen.bench.readings=false, every variant also
   through against-llama/compare-readings (whole, chunk, steps:
   cuda-reference-noise's statistic, against the CPU's readings rebuilt
   from their top 64; a forced variant with the CPU's experts, its
   :forced), and with -Dgen.bench.mutations=true the float16 and forced
   variants' planted convention errors (against-llama/mutated) on the
   general prompts.

   Measured on fenrir (RTX PRO 6000, MLX 0.32.4, 2026-11-08,
   Qwen3.6-35B-A3B Q4_K_M, 40 prompts: 1,246 positions, 40 layers; the
   CPU's readings 1389 s, a variant about 40 s):
   - f32: 10 top-k sets of 49,840 differ (0.02%), at 4 positions of the
     two prompts whose gap jumps (0.23 and 0.40 nats), each first at a
     near-tie (the CPU's log p_8/p_9 4.8e-7, 3.1e-4), the rest after it.
     Every position with no flip at or before it is within 1.2e-4;
     f32-forced within 5.4e-4.
   - f16-old: 14.5% differ (1.2% in layer 0, about 20% from layer 20), at
     78% of the positions. The gap where no flip came before: median
     0.011, at most 0.09; at a flip median 0.27, at most 6.07. The
     readings' (cuda-reference-noise's) medians 0.17-0.21 (general) and
     0.49-0.58 (digits), at most 2.71 and 6.07.
   - f16-old-forced: every position within 0.24 (median 0.015), as the
     dense models' float16; its own router would differ at 1.9%, mostly
     ties of float16 logits.
   - f16: 11.6% differ (65% of positions); the readings' medians
     0.12-0.15 and 0.32-0.36, at most 2.74 and 6.01; f16-forced within
     0.17 (median 0.015), its own router differing at 1.0%, its readings
     at most 3.97e-2 and 0.172 (medians 0.013-0.022), the planted errors
     at least 0.057 (RoPE's traditional pairs; median 0.204), the
     others 2.25.
   The flips, not float16's rounding elsewhere, make the gap: forced to
   the CPU's experts, float16 is as close as a dense model's. A flip at
   a near-tie moves the hidden state, which flips more experts in later
   layers and positions.

   With -Dgen.bench.reference=gpu the reference is the same forward in
   float32 on the backend the variants run on (its GPU: Metal's float32
   against its float16, where MLX's CPU would take the longest), each
   prompt's readings still taken first and its model dropped. Measured so
   on the MacBook M2 Max (Metal, 2026-11-08, the same model and 40
   prompts, -Xmx8g; the reference's readings 50 s, a variant 25-30 s):
   - f16: 12.9% of the top-k sets differ (69% of positions); the gap
     where no flip came before at most 0.10 (median 0.012); the readings'
     medians 0.13-0.16 (general) and 0.30-0.41 (digits), at most 2.73
     and 5.91, the argmax equal in 198 and 197 of 200.
   - f16-forced: its own router differing at 1.1%; the readings' medians
     0.013 and 0.019-0.023, at most 0.040 and 0.278, the argmax always
     equal; the planted errors at least 0.054 (RoPE's traditional pairs;
     median 0.209), the others 2.23.
   As on CUDA: the flips make float16's gap.

   Run on MLX's CUDA, one GPU JVM at a time (the CPU's model about 31 GB
   resident; then CUDA's):
     clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.moe-routing Qwen3.6-35B-A3B-Q4_K_M.gguf"
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.branching-contract :refer [top-tokens]]
            [gen.llm.models.against-llama :as against]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen35 :as qwen35]))

(defn- native
  "A function of a namespace that loads native code (MLX's backend,
   llama.cpp's), resolved when the bench runs."
  [ns sym]
  @(requiring-resolve (symbol ns (name sym))))

(def ^:private top-k @#'qwen35/top-k)

(def ^:private calls
  "The router's probabilities [N E] of the forward running, by layer."
  (atom []))

(defn- recording [probs k] (swap! calls conj probs) (top-k probs k))

(defn- old-router
  "The router before genclj-fxag: its product in the model's dtype (the
   weights cast to it), the logits cast to float32 for the softmax."
  [dtype router xs]
  (array/astype (common/linear xs (common/cast-weight 'bench router dtype)) :float32))

(defn- seconds [t0] (/ (- (System/nanoTime) (long t0)) 1e9))

(defn- rows
  "The log-probabilities [T vocab] of logits [1 T vocab] in float32, a
   float[] of T rows."
  ^floats [logits]
  (let [[_ t v] (array/shape logits)]
    (array/->floats (array/log-softmax (array/astype (array/reshape logits [t v]) :float32)))))

(defn- row ^floats [^floats a v i]
  (java.util.Arrays/copyOfRange a (int (* (long v) (long i))) (int (* (long v) (inc (long i))))))

(defn- forward
  "Runs f (a forward of the model) recording the router: [f's result,
   the router's probabilities by layer, each a float[] of its N rows]."
  [f]
  (reset! calls [])
  (let [out (f)
        ps @calls]
    (array/eval! ps)
    [out (mapv array/->floats ps)]))

(defn- by-position
  "The router's float[]s by layer (rows rows each) as [position [layer
   float[E]]], appended to acc."
  [acc layers e]
  (let [n (quot (alength ^floats (first layers)) (long e))]
    (into acc (for [i (range n)] (mapv #(row % e i) layers)))))

(defn- read-prompt
  "The model m's readings of tokens, then steps (the step tokens; nil:
   8 greedy ones): {:lp [position float[vocab]] :router [position [layer
   float[E]]] :steps [tokens]}. With forced (position → layer → [k]
   indices), every layer's experts are those."
  [m tokens steps forced e]
  (let [n (count tokens)
        plan (fn [positions]
               (when forced
                 (fn [probs _k]
                   (let [layer (count @calls)
                         idx (array/array (mapv #(get-in forced [% layer]) positions) {:dtype :int32})]
                     (swap! calls conj probs)
                     [idx (array/take-along-axis probs idx -1)]))))
        run (fn [positions f]
              (if-let [p (plan positions)]
                (with-redefs-fn {#'qwen35/top-k p} #(forward f))
                (forward f)))
        [[lg cache] rs] (run (range n) #(qwen35/prefill m [tokens]))
        v (peek (array/shape lg))
        lps (rows lg)
        acc {:lp (mapv #(row lps v %) (range n)) :router (by-position [] rs e) :steps []}]
    (loop [i 0 cache cache acc acc]
      (if (= i 8)
        acc
        (let [t (if steps (steps i) (first (top-tokens (peek (:lp acc)) 1)))
              [[lg c] rs] (run [(+ n i)] #(qwen35/step m cache [[t]]))]
          (recur (inc i) c (-> acc
                               (update :lp conj (rows lg))
                               (update :router by-position rs e)
                               (update :steps conj t))))))))

(defn- top-set
  "The k largest entries' indices of p (a float[] of probabilities), the
   first of equal ones first (argmax's order, top-k's): [the indices in
   order, the log of the k-th over the (k+1)-th]. One sort of (-p, index)
   packed in longs (a non-negative float's bits order it)."
  [^floats p k]
  (let [n (alength p)
        packed (long-array n)]
    (dotimes [j n]
      (aset packed j (bit-or (bit-shift-left (- 0x7fffffff (long (Float/floatToIntBits (aget p j)))) 32) j)))
    (java.util.Arrays/sort packed)
    (let [idx (mapv #(bit-and (aget packed (int %)) 0xffffffff) (range (inc (long k))))]
      [(pop idx) (Math/log (/ (aget p (int (idx (dec (long k))))) (aget p (int (peek idx)))))])))

(defn- summary [xs]
  (if (empty? xs)
    "n    0"
    (let [v (vec (sort xs)) n (count v)]
      (format "n %5d  median %.2e  p90 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.9 n)))) (peek v)))))

(defn- reference
  "The CPU's data of a prompt: per position the top 64 tokens and their
   log-probabilities, the router's top-k sets and margins by layer, and
   the greedy steps."
  [{:keys [lp router steps]} k]
  {:top (mapv (fn [^floats l] (let [ts (top-tokens l 64)] [ts (float-array (map #(aget l (int %)) ts))])) lp)
   :sets (mapv (fn [layers] (mapv #(top-set % k) layers)) router)
   :steps steps
   :vocab (alength ^floats (first lp))})

(defn- gap ^double [[ts ^floats vs] ^floats lp]
  (reduce max 0.0 (map-indexed (fn [j t] (Math/abs (- (aget vs (int j)) (aget lp (int t))))) ts)))

(defn- compare-prompt
  "Per position of a prompt: {:gap :flips [layers flipped] :margins
   [the CPU's margin at each flip] :own [the variant's margin there]}."
  [ref got k]
  (mapv (fn [p]
          (let [g (gap ((:top ref) p) ((:lp got) p))
                cpu ((:sets ref) p)
                own (mapv #(top-set % k) ((:router got) p))
                flipped (filterv #(not= (set (first (cpu %))) (set (first (own %)))) (range (count cpu)))]
            {:gap g :flips flipped
             :margins (mapv #(second (cpu %)) flipped)
             :own (mapv #(second (own %)) flipped)
             :changed (mapv #(count (remove (set (first (cpu %))) (first (own %)))) flipped)}))
        (range (count (:top ref)))))

(defn- report
  "Prints a variant's comparison: rows [[item ref positions]]."
  [title rows layers]
  (let [ps (mapcat #(nth % 2) rows)
        n (count ps)
        flips (mapcat :flips ps)
        by-layer (frequencies flips)
        state (fn [[item _ positions]]
                (let [n-tok (count (:tokens item))]
                  (loop [i 0 seen? false out []]
                    (if (= i (count positions))
                      out
                      (let [p (positions i) here? (seq (:flips p))]
                        (recur (inc i) (or seen? (boolean here?))
                               (conj out (assoc p :bucket (cond here? :at seen? :before :else :none)
                                                :reading? (>= i (dec n-tok))))))))))
        tagged (mapcat state rows)]
    (println (str "\n=== " title))
    (println (format "positions %d, layer-positions %d: flips %d (%.3f%%), positions with a flip %d (%.1f%%); experts changed per flip %s"
                     n (* n layers) (count flips) (/ (* 100.0 (count flips)) (* n layers))
                     (count (filter (comp seq :flips) ps)) (/ (* 100.0 (count (filter (comp seq :flips) ps))) (max 1 n))
                     (frequencies (mapcat :changed ps))))
    (println "flips by layer (count; rate in % of positions):")
    (doseq [row (partition-all 10 (range layers))]
      (println "  " (str/join "  " (map #(format "%2d:%3d %5.2f" % (get by-layer % 0) (/ (* 100.0 (get by-layer % 0)) (max 1 n))) row))))
    (println "the CPU's margin (log p_k / p_k+1) at the flips:" (summary (mapcat :margins ps)))
    (println "the variant's own margin there:                  " (summary (mapcat :own ps)))
    (println "the gap |Δ log p| by position:")
    (doseq [b [:none :before :at]]
      (println (format "  %-7s %s" (name b) (summary (map :gap (filter #(= b (:bucket %)) tagged))))))
    (println (format "  %-7s %s" "all" (summary (map :gap tagged))))
    (println (format "  %-7s %s" "readings" (summary (map :gap (filter :reading? tagged)))))
    (println "  by the flips at the position (0, 1, 2, 3+):")
    (doseq [[label pred] [["0" #(empty? (:flips %))] ["1" #(= 1 (count (:flips %)))] ["2" #(= 2 (count (:flips %)))]
                          ["3+" #(<= 3 (count (:flips %)))]]]
      (println (format "    %-3s %s" label (summary (map :gap (filter pred tagged))))))
    (println "by prompt (positions; flips, positions flipped; readings' worst gap, median gap of all positions):")
    (doseq [[item _ positions] rows]
      (let [n-tok (count (:tokens item))
            reading (subvec positions (dec n-tok))
            worst (reduce max (map :gap reading))]
        (println (format "  %3d  %4d %3d  %.2e %.2e  %s" (count positions) (count (mapcat :flips positions))
                         (count (filter (comp seq :flips) positions)) worst
                         (nth (sort (map :gap positions)) (quot (count positions) 2))
                         (pr-str (subs (:text item) 0 (min 32 (count (:text item)))))))
        (when (and (< 0.05 worst) (< (count (mapcat :flips positions)) 12))
          (doseq [[i p] (map-indexed vector positions) :when (seq (:flips p))]
            (println (format "        position %d: layers %s, the CPU's margins %s, its own %s, gap there %.2e"
                             i (:flips p) (mapv #(format "%.2e" %) (:margins p)) (mapv #(format "%.2e" %) (:own p)) (:gap p)))))))
    (flush)))

(defn- readings
  "The CPU's data as against-llama/reference's readings: each reading's
   log-probabilities a float[vocab] of its top 64, the rest -1e30; with
   forced?, :forced the CPU's experts (compare-readings forces them)."
  [{:keys [top steps vocab sets]} n-tok forced?]
  (let [dense (fn [[ts ^floats vs]]
                (let [a (float-array vocab (float -1e30))]
                  (dorun (map-indexed (fn [j t] (aset a (int t) (aget vs (int j)))) ts))
                  a))]
    (cond-> {:root (dense (top (dec n-tok)))
             :steps (mapv (fn [i] [(steps i) (dense (top (+ n-tok i)))]) (range (count steps)))}
      forced? (assoc :forced (mapv (fn [layers] (mapv first layers)) sets)))))

(defn- set-readings [title sets refs m forced?]
  (doseq [[set-name items] sets]
    (let [rs (mapv (fn [item ref] (against/compare-readings m (assoc (readings ref (count (:tokens item)) forced?) :tokens (:tokens item)) 0))
                   items (get refs set-name))
          all (apply concat rs)
          kind (fn [r] (let [w (:what r)] (if (vector? w) :step w)))]
      (println (str "\n" title ", the " set-name " prompts, against-llama/compare-readings (cuda-reference-noise's statistic):"))
      (doseq [[k xs] (sort-by key (group-by kind all))]
        (println (format "  %-6s %s" (name k) (summary (map :max-diff xs)))))
      (println (format "  overall max %.3e; argmax equal in %d of %d, top 5 in %d"
                       (reduce max (map :max-diff all)) (count (filter :argmax? all)) (count all) (count (filter :top5? all))))
      (flush))))

(defn- mutations [title sets refs m forced?]
  (when-let [items (some (fn [[s xs]] (when (= "general" s) xs)) sets)]
    (println (str "\n" title ": the planted convention errors, worst |Δ log p| of each general prompt (whole, chunk, one step):"))
    (doseq [mutation (sort (keys (against/mutations-of m)))]
      (let [ds (mapv (fn [item ref] (against/mutated m (assoc (readings ref (count (:tokens item)) forced?) :tokens (:tokens item)) mutation))
                     items (get refs "general"))]
        (println (format "  %-24s min %.3f  median %.3f  max %.3f nats" (name mutation)
                         (reduce min ds) (nth (sort ds) (quot (count ds) 2)) (reduce max ds)))
        (flush)))))

(defn- cpu-references
  "[the model's config, the CPU's references by set] of the model at path
   on the CPU backend cpu, in float32; the model is dropped on return."
  [path cpu sets]
  (let [t0 (System/nanoTime)
        m (array/with-backend cpu (against/load-model path {:dtype :float32}))
        {:keys [experts experts-used layers] :as cfg} (:config m)
        _ (println (format "the CPU's model loaded in %.1f s: %d layers, %d experts, %d used" (seconds t0) layers experts experts-used))
        t1 (System/nanoTime)
        refs (array/with-backend cpu
               (into {} (for [[set-name items] sets]
                          (let [t (System/nanoTime)
                                rs (mapv (fn [{:keys [tokens]}]
                                           (reference (with-redefs-fn {#'qwen35/top-k recording}
                                                        #(read-prompt m tokens nil nil experts))
                                                      experts-used))
                                         items)]
                            (println (format "  the %s prompts: %.1f s" set-name (seconds t)))
                            (flush)
                            [set-name rs]))))]
    (println (format "the CPU's readings: %.1f s" (seconds t1)))
    [cfg refs]))

(def ^:private variants
  {"f32" {:dtype :float32}
   "f32-forced" {:dtype :float32 :forced? true}
   "f16-old" {:dtype :float16 :old? true}
   "f16-old-forced" {:dtype :float16 :old? true :forced? true}
   "f16" {:dtype :float16}
   "f16-forced" {:dtype :float16 :forced? true}})

(defn -main [model & _]
  (let [names (str/split (System/getProperty "gen.bench.variants" "f32,f32-forced,f16-old,f16-old-forced,f16,f16-forced") #",")
        _ (System/setProperty "gen.bench.prompts" (System/getProperty "gen.bench.prompts" "all"))
        ;; gen.bench.reference=gpu: the reference on the backend the
        ;; variants run on (Metal's float32 against its float16)
        cpu (if (= "gpu" (System/getProperty "gen.bench.reference" "cpu"))
              array/*backend*
              ((native "gen.backend.mlx" (quote make-backend)) {:device :cpu}))]
    (println "backend" (array/backend-name array/*backend*) (array/device array/*backend*) "model" model "variants" names)
    (with-open [vocab ^java.io.Closeable ((native "gen.llm.llama" (quote vocabulary)) model)]
      (let [sets ((native "gen.llama-noise" (quote prompt-sets)) vocab)
            [{:keys [experts experts-used layers]} refs] (cpu-references model cpu sets)
            _ ((native "gen.backend.mlx" (quote collect!)))]
        (doseq [[dtype vs] (group-by (comp :dtype variants) names)]
          (let [t2 (System/nanoTime)
                m (against/load-model model {:dtype dtype})]
            (println (format "\nCUDA's model in %s loaded in %.1f s" (name dtype) (seconds t2)))
            (doseq [v vs]
              (let [{:keys [old? forced?]} (variants v)
                    t3 (System/nanoTime)
                    redefs (cond-> {#'qwen35/top-k recording} old? (assoc #'qwen35/router-logits old-router))
                    rows (with-redefs-fn redefs
                           #(vec (for [[set-name items] sets
                                       [item ref] (map vector items (get refs set-name))]
                                   (let [forced (when forced? (mapv (fn [layers] (mapv first layers)) (:sets ref)))
                                         got (read-prompt m (:tokens item) (:steps ref) forced experts)]
                                     [item ref (compare-prompt ref got experts-used)]))))]
                (report (str v " against the CPU's float32 (" (format "%.1f s" (seconds t3)) ")") rows layers)
                (when-not (= "false" (System/getProperty "gen.bench.readings"))
                  (with-redefs-fn (if old? {#'qwen35/router-logits old-router} {})
                    #(do (set-readings v sets refs m forced?)
                         (when (and (= "true" (System/getProperty "gen.bench.mutations")) (or forced? (= :float16 dtype)))
                           (mutations v sets refs m forced?)))))))
            ((native "gen.backend.mlx" (quote collect!)))))))
    (shutdown-agents)))
