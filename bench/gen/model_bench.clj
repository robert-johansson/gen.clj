(ns gen.model-bench
  "Model-level costs on the JVM backend: GFI operations on small scalar
   models, batched against per-particle execution, and gradients. The
   baseline the MLX backend must beat for small models (its eval floor is
   tens of µs); and the cost per traced choice, linreg with K
   observations (against Gen.jl's dynamic DSL, docs/benchmarks.md).
   Run: clojure -M:bench -m gen.model-bench              (add :c2 to compare with C2)
        clojure -M:bench -m gen.model-bench per-choice   (the cost per choice alone)
        clojure -M:bench -m gen.model-bench long-unfold  (a 500-step Unfold's running total)"
  (:refer-clojure :exclude [+ *])
  (:require [criterium.core :as crit]
            [gen.array :as array :refer [+ *]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      slope)))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def step (gen [t s] (let [x (trace :x (d/normal (* 0.9 s) 1))] (trace :y (d/normal x 0.5)) x)))
(def unfolded (comb/unfold step))

(defn- us-per-call [f] (* 1e6 (first (:mean (crit/quick-benchmark* f {})))))

(defn- per-choice
  "simulate and generate of linreg with K observations (K + 2 choices), K =
   10, 100, 1000: µs a call and a choice."
  []
  (println "\n== Per traced choice: linreg with K observations (K + 2 choices) ==")
  (doseq [k [10 100 1000]]
    (let [xs (vec (range (double k)))
          obs (apply cm/choicemap (mapcat (fn [i] [[:y i] (* 2.0 i)]) (range k)))
          key (rng/key 1)
          s (us-per-call #(gfi/simulate linreg [xs] key))
          g (us-per-call #(gfi/generate linreg [xs] obs key))]
      (println (format "  K=%-5d simulate %10.2f µs (%.3f µs a choice)   generate %10.2f µs (%.3f µs a choice)"
                       k s (/ s (+ k 2)) g (/ g (+ k 2)))))))

(defn- model-costs []
  (let [xs (vec (range 10.0))
        obs (apply cm/choicemap (mapcat (fn [i] [[:y i] (* 2.0 i)]) (range 10)))
        k (rng/key 1)
        tr (:trace (gfi/generate linreg [xs] obs k))
        htr (gfi/simulate hierarchical [10] k)
        utr (:trace (gfi/generate unfolded [50 0.0] (apply cm/choicemap (mapcat (fn [t] [[t :y] 0.1]) (range 50))) k))
        ps (:particles (b/generate linreg [xs] obs 1000 k))
        cases [["linreg (12 sites): simulate" #(gfi/simulate linreg [xs] k)]
               ["linreg: generate with 10 observations" #(gfi/generate linreg [xs] obs k)]
               ["linreg: assess" #(gfi/assess linreg [xs] (:choices tr))]
               ["linreg: update one choice" #(gfi/update tr (cm/choicemap :slope 1.5) k)]
               ["linreg: regenerate :slope" #(gfi/regenerate tr (sel/select :slope) k)]
               ["linreg: choice gradients (slope, intercept)" #(grad/choice-gradients tr (sel/select :slope :intercept))]
               ["hierarchical (10 splices): simulate" #(gfi/simulate hierarchical [10] k)]
               ["hierarchical: update :mu" #(gfi/update htr (cm/choicemap :mu 0.3) k)]
               ["unfold (50 steps): extend by one step" #(gfi/update utr [51 0.0] [diff/unknown-change diff/no-change] (cm/choicemap [50 :y] 0.2) k)]
               ["linreg ×1000 particles, per particle: generate" #(dotimes [i 1000] (gfi/generate linreg [xs] obs (rng/fold-in k i)))]
               ["linreg ×1000 particles, batched: generate" #(b/generate linreg [xs] obs 1000 k)]
               ["linreg ×1000 particles, batched: regenerate :slope" #(b/regenerate ps [xs] [diff/no-change] (sel/select :slope) k)]]]
    (println "JVM:" (System/getProperty "java.vm.name") (System/getProperty "java.version")
             (if (some #(= "-XX:-UseJVMCICompiler" %) (.getInputArguments (java.lang.management.ManagementFactory/getRuntimeMXBean)))
               "(C2)" "(Graal JIT)"))
    (doseq [[label f] cases]
      (println (format "%-52s %10.2f µs" label (us-per-call f))))))

(defn- settled
  "f with its result's scores and weights evaluated: on a lazy backend
   (MLX) a call computes only what is read."
  [f]
  (fn [] (let [r (f)]
           (array/eval! [(-> r :trace :score) (:weight r) (-> r :particles :trace :score) (:weights r)])
           r)))

(defn- long-unfold
  "A 500-step Unfold, scalar and as 1000 batched particles: an update that
   revisits one step (its :y, so no later step), whose score changes by one
   step's, and an extension by one step (a particle filter's), against
   generating every step. Batched runs without spot checks: the run alone.
   On any backend (-J-Dgen.backend=mlx), the scores evaluated."
  []
  (println "\n== A 500-step Unfold: revisiting one step, extending by one ==")
  (binding [b/*spot-checks* 0]
    (let [n 500
          k (rng/key 1)
          obs (apply cm/choicemap (mapcat (fn [t] [[t :y] 0.1]) (range n)))
          tr (:trace (gfi/generate unfolded [n 0.0] obs k))
          ps (:particles (b/generate unfolded [n 0.0] obs 1000 k))
          same [diff/no-change diff/no-change]
          longer [diff/unknown-change diff/no-change]
          cases [["scalar: generate" #(gfi/generate unfolded [n 0.0] obs k)]
                 ["scalar: update one step's :y" #(gfi/update tr [n 0.0] same (cm/choicemap [250 :y] 0.3) k)]
                 ["scalar: extend by one step" #(gfi/update tr [(inc n) 0.0] longer (cm/choicemap [n :y] 0.2) k)]
                 ["×1000 batched: generate" #(b/generate unfolded [n 0.0] obs 1000 k)]
                 ["×1000 batched: update one step's :y" #(b/update ps [n 0.0] same (cm/choicemap [250 :y] 0.3) k)]
                 ["×1000 batched: extend by one step" #(b/update ps [(inc n) 0.0] longer (cm/choicemap [n :y] 0.2) k)]]]
      (doseq [[label f] cases]
        (println (format "  %-50s %10.2f µs" label (us-per-call (settled f))))))))

(defn -main [& [arg]]
  (case arg
    "per-choice" (per-choice)
    "long-unfold" (long-unfold)
    (do (model-costs) (per-choice))))
