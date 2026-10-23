(ns gen.switch-bench
  "A Switch whose index differs across the particles, batched by partition
   (bean genclj-7tqw), against one particle at a time, which is what such a
   model ran as before: importance sampling on a mixture of K components of
   different structures (component j makes j + 1 choices), from 100 to
   10,000 particles, and an MH sweep of 1,000 chains that moves particles
   between the components. Each time is the median of timed runs after a
   warm-up, with spot checks on (as by default):
     clojure -M:bench -m gen.switch-bench
     clojure -J-Dgen.backend=mlx [-J-Dgen.mlx.device=cpu] -M:bench -m gen.switch-bench"
  (:refer-clojure :exclude [+ - *])
  (:require [gen.array :as array :refer [+ - *]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

(defn- time-ms [f]
  (let [t0 (System/nanoTime)]
    (f)
    (/ (- (System/nanoTime) t0) 1e6)))

(defn- median-ms
  "The median time of (f): 5 timed runs after 3 warm-ups, or 11 after 20
   when a run takes under 50 ms."
  [f]
  (let [fast? (< (time-ms f) 50.0)
        reps (if fast? 11 5)]
    (dotimes [_ (if fast? 20 3)] (f))
    (nth (sort (repeatedly reps #(time-ms f))) (quot reps 2))))

(defn- row [label one batched]
  (let [a (median-ms one) b (median-ms batched)]
    (println (format "| %s | %.2f ms | %.2f ms | %.1f× |" label a b (/ a b)))
    (flush)))

(defn- component
  "Component j of a mixture: x around m and j choices of its own."
  [j]
  (gen [m] (let [us (mapv #(trace [:u %] (d/normal 0 1)) (range j))]
             (trace :x (d/normal (+ m (* 2.0 j) (reduce + 0.0 us)) 1.0)))))

(defn- mixture
  "A mixture of k components of different structures, y observed around
   the chosen one's x."
  [k]
  (let [components (apply comb/switch (map component (range k)))
        ps (vec (repeat k (/ 1.0 k)))]
    (gen [] (let [i (trace :k (d/categorical ps))]
              (trace :y (d/normal (splice :c components i 0.0) 1.0))))))

(def observation (cm/choicemap :y 2.5))

(defn- importance [model n batched?]
  #(imp/importance-sampling model [] observation n {:batched? batched? :key (rng/key 1)}))

(defn- sweep
  "An MH move of each of n chains on the mixture: the index and the
   component redrawn together, and accepted per chain."
  [ps]
  #(mh/mh-particles ps (sel/select :k :c) (rng/key 2)))

(defn -main [& _]
  (let [m2 (mixture 2) m5 (mixture 5)]
    (dotimes [_ 3]
      (doseq [m [m2 m5] n [100 1000] batched? [false true]] ((importance m n batched?))))
    (println "## Importance sampling on a mixture of K components of different structures\n")
    (println "| K, particles | one at a time | batched | speedup |")
    (println "|---|---|---|---|")
    (doseq [[k m] [[2 m2] [5 m5]]
            n [100 1000 10000]]
      (row (str k ", " n) (importance m n false) (importance m n true)))
    (println "\n## An MH sweep of 1,000 chains on the mixture of 2, moving particles between components\n")
    (println "| chains | one at a time | batched | speedup |")
    (println "|---|---|---|---|")
    (let [ps (:particles (b/generate m2 [] observation 1000 (rng/key 3)))
          one (assoc ps :trace nil :traces (b/traces ps))]
      (row "1,000" (sweep one) (sweep ps))))
  (shutdown-agents))
