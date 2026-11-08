(ns gen.gdn-states-bench
  "Qwen3.5's GatedDeltaNet states between decode steps (genclj-6qb6): a
   step's rows taken from groups (the array engine's paged and
   continuous modes) against a pool of every slot's states, on MLX:

     clojure -J-Dgen.backend=mlx -M:bench -m gen.gdn-states-bench

   A step is gen.array/gated-delta of one token a row over every
   GatedDeltaNet layer's recurrent state (gen.bench.layers 18 layers of
   [16 128 128] float32, Qwen3.5-0.8B's: 18 MiB a row), compiled, its
   inputs fixed. Its rows (gen.bench.rows, 16) are a resample's: the
   rows of the last step's group in another order, some twice. Rows, ms
   a step (of gen.bench.steps 60 steps after 5 untimed):
   - lane: the group itself stepped, consumed (donated): no copy;
   - take: the rows taken from the group (one gather) and stepped;
   - two groups, padded: taken from two groups, concatenated, then zero
     rows concatenated for the padding (the paged mode's stacking before
     genclj-6qb6's continuous mode: three copies);
   - slices, one concat: the same rows as runs of slices (views) and the
     zero rows, one concatenate (one copy);
   - pool, two calls: the rows gathered from a pool of gen.bench.pool
     rows (66) and evaluated, then stepped and written back into the
     pool (put-blocks), consuming it;
   - pool, one call: gathered, stepped and written back in one call
     consuming the pool: MLX copies the pool, its read holding it;
   - pool, deferred: one call that first writes the last step's states
     into the pool (consuming it), then gathers the rows from the pool
     written and steps them, its states written by the next step.
   - group read, not consumed: the group itself read by a step that does
     not consume it (forks still read it), its states freed by the GC;
   - take, not consumed: the same after a take.
   Each row's median, and its mean, which counts the GC's collections
   of what is not consumed (MlxRuntime collects after
   gen.mlx.gc-new-bytes of new arrays). The kernel itself reads and
   writes each row once (as a copy does), so a lane is the floor.
   Measured on the Mac mini M4 (2026-11-08, 16 rows, mean ms a step):
   lane 6.7, group read 12.5, take 17.7 (not consumed 21.3), two groups
   padded 42.1 (takes and one concatenate 30.1, slices and one
   concatenate 23.9), pool in two calls 26.0, in one call 18.2 (at 8
   rows 32.3: MLX copied the pool), deferred 20.0."
  (:require [gen.array :as array]
            [gen.array.paged :as gpaged]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))
(defn- long-prop ^long [k default] (parse-long (prop k default)))

(def ^:private settings
  {:layers (long-prop "layers" 18) :heads (long-prop "heads" 16) :dim (long-prop "dim" 128)
   :rows (long-prop "rows" 16) :pool (long-prop "pool" 66) :steps (long-prop "steps" 60)
   :dtype (keyword (prop "dtype" "float32"))})

(defn- inputs
  "Fixed q, k, v, g and beta of r rows, one token each."
  [r]
  (let [{:keys [heads dim]} settings
        [k1 k2 k3 k4 k5] (rng/split (rng/key 3) 5)
        unit (fn [x] (array// x (array/sqrt (array/sum (array/* x x) {:axis -1 :keep-dims? true}))))]
    (doto {:q (unit (rng/normal k1 [r 1 heads dim])) :k (unit (rng/normal k2 [r 1 heads dim]))
           :v (rng/normal k3 [r 1 heads dim])
           :g (array/* -0.1 (rng/uniform k4 [r 1 heads]))
           :beta (rng/uniform k5 [r 1 heads])}
      array/eval!)))

(defn- step
  "Every layer's states [r H D D] after one token: gated-delta's."
  [states {:keys [q k v g beta]}]
  (mapv #(second (array/gated-delta q k v g beta %)) states))

(defn- zero-states [n]
  (let [{:keys [layers heads dim]} settings
        z (array/zeros [n heads dim dim])]
    (doto (vec (repeat layers z)) array/eval!)))

(defn- warm-states
  "States of n rows, stepped a few times from zeros (not zeros: a copy
   of zeros could be cheaper)."
  [n]
  (let [f (array/compile step)
        in (inputs n)]
    (doto (nth (iterate #(f % in) (zero-states n)) 3) array/eval!)))

(defn- median-ms
  "[median mean] ms of (f x) over the steps, x threaded through (f x) → x',
   each evaluated, after 5 untimed."
  [f x]
  (let [x (nth (iterate #(doto (f %) array/eval!) x) 5)
        [ts _] (reduce (fn [[ts x] _]
                         (let [t0 (System/nanoTime)
                               x' (doto (f x) array/eval!)]
                           [(conj ts (/ (- (System/nanoTime) t0) 1e6)) x']))
                       [[] x]
                       (range (:steps settings)))]
    [(nth (sort ts) (quot (count ts) 2)) (/ (reduce + ts) (count ts))]))

(defn- resample
  "A resample's rows of n: some twice, some not at all, out of order."
  [n]
  (vec (take n (cycle (concat (range 1 n 3) (range 0 n 2) (range 2 n 5))))))

(defn -main [& _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (println "## GatedDeltaNet states between steps\n")
  (println "- settings:" (pr-str settings))
  (let [{:keys [rows pool]} settings
        r (long rows)
        in (inputs r)
        idx (resample r)
        ix (array/array idx {:dtype :int32})
        stepped (array/compile step {:donate #{0}})
        rows-of (fn [states ix] (mapv #(array/take % ix 0) states))
        half (quot r 2)
        out (fn [label [ms mean]] (println (format "| %-42s | %8.2f | %8.2f |" label ms mean)))]
    (println "\n| states | ms a step (median) | (mean, with the GC) |")
    (println "|---|---:|---:|")
    ;; lane: the group itself
    (out "lane (in place)" (median-ms #(stepped % in) (warm-states r)))
    ;; the group itself, read but not consumed (its old states freed by
    ;; the GC)
    (let [read (array/compile step)]
      (out "group read, not consumed" (median-ms #(read % in) (warm-states r)))
      (out "take, not consumed" (median-ms #(read (rows-of % ix) in) (warm-states r))))
    ;; take: one gather of the group's rows
    (out "take" (median-ms #(stepped (rows-of % ix) in) (warm-states r)))
    ;; two groups, padded: the rows of two groups of r rows, half each,
    ;; taken, concatenated, zero padding concatenated
    (let [other (warm-states r)
          ia (array/array (subvec idx 0 (dec half)) {:dtype :int32})
          ib (array/array (subvec idx 0 half) {:dtype :int32})
          pad 1]
      (out "two groups, padded (3 copies)"
           (median-ms (fn [g]
                        (let [a (rows-of g ia) b (rows-of other ib)
                              stacked (mapv #(array/concatenate [%1 %2] 0) a b)]
                          (stepped (mapv #(array/concatenate [% (array/zeros (assoc (array/shape %) 0 pad))] 0) stacked) in)))
                      (warm-states r)))
      (out "two groups, takes + one concat (2 copies)"
           (median-ms (fn [g]
                        (stepped (mapv (fn [a b] (array/concatenate [a b (array/zeros (assoc (array/shape a) 0 pad))] 0))
                                       (rows-of g ia) (rows-of other ib))
                                 in))
                      (warm-states r)))
      (out "slices, one concat (1 copy)"
           (median-ms (fn [g]
                        (stepped (mapv (fn [x y]
                                         (array/concatenate
                                          (-> []
                                              (into (map #(array/slice x 0 % (inc (long %)))) (subvec idx 0 (dec half)))
                                              (into (map #(array/slice y 0 % (inc (long %)))) (subvec idx 0 half))
                                              (conj (array/zeros (assoc (array/shape x) 0 pad))))
                                          0))
                                       g other)
                                 in))
                      (warm-states r))))
    ;; the pool: S rows, the step's rows at slots
    (let [slots (array/array (vec (range 3 (+ 3 r))) {:dtype :int32})
          gather (array/compile rows-of)
          put (array/compile (fn [p states slots in] (mapv #(gpaged/put-blocks %1 %2 slots) p (step states in))) {:donate #{0}})
          one (array/compile (fn [p ix slots in] (mapv #(gpaged/put-blocks %1 %2 slots) p (step (rows-of p ix) in))) {:donate #{0}})
          deferred (array/compile (fn [p prev slots ix in]
                                    (let [p (mapv #(gpaged/put-blocks %1 %2 slots) p prev)]
                                      [p (step (rows-of p ix) in)]))
                                  {:donate #{0 1}})
          ;; the step reads the pool's rows idx + 3, writes rows 3 …
          pix (array/array (mapv #(+ 3 (long %)) idx) {:dtype :int32})]
      (out "pool, two calls" (median-ms (fn [p] (let [g (doto (gather p pix) array/eval!)] (put p g slots in))) (warm-states pool)))
      (out "pool, one call" (median-ms #(one % pix slots in) (warm-states pool)))
      (out "pool, deferred write" (median-ms (fn [[p prev]] (deferred p prev slots pix in)) [(warm-states pool) (warm-states r)]))))
  (println "\nMLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
  (shutdown-agents))
