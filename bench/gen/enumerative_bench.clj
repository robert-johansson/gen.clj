(ns gen.enumerative-bench
  "Batched enumeration against one point at a time (bean genclj-7oc1):
   enumerative-inference over a linear regression's grid of slopes and
   intercepts, from 100 to 100,000 points, and with-exact-proposal's
   enumeration of 10 to 1000 assignments. Each time is the median of timed
   runs after a warm-up, with spot checks on (as by default):
     clojure -M:bench -m gen.enumerative-bench
     clojure -J-Dgen.backend=mlx [-J-Dgen.mlx.device=cpu] -M:bench -m gen.enumerative-bench"
  (:refer-clojure :exclude [+ - *])
  (:require [gen.array :as array :refer [+ - *]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.random :as rng]))

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

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.5))))))

(def xs (mapv #(- (* 0.25 %) 2.0) (range 20)))

(def observations
  (apply cm/choicemap (mapcat (fn [i x] [[:y i] (+ (* 1.5 x) 0.3 (* 0.2 (Math/sin (* 7.0 x))))]) (range) xs)))

(defn- grid
  "A k × k grid of slopes and intercepts in [-4, 4]."
  [k]
  (let [ends (mapv #(- (* (/ 8.0 k) %) 4.0) (range (inc k)))]
    (enum/choice-vol-grid [:slope ends :continuous] [:intercept ends :continuous])))

(def digits
  "n digits observed through their number: 10^n assignments to enumerate."
  (gen [n]
    (let [ds (mapv #(trace [:d %] (d/categorical (array/array (vec (repeat 10 0.1))))) (range n))]
      (trace :y (d/normal (reduce + (map * ds (map #(Math/pow 10.0 %) (range (dec n) -1 -1)))) 5.0)))))

(defn- warm-up!
  "Both paths, on every size, until the JIT has compiled them."
  []
  (dotimes [_ 3]
    (doseq [k [10 32 100]
            opts [{} {:batched? true}]]
      (enum/enumerative-inference linreg [xs] observations (grid k) (assoc opts :key (rng/key 1)))))
  (doseq [n [1 2 3] opts [{} {:batched? true}]]
    (dotimes [_ 20]
      (gfi/generate (enum/with-exact-proposal digits (mapv #(vector :d %) (range n)) opts) [n]
                    (cm/choicemap :y 7.0) (rng/key 2)))))

(defn -main [& _]
  (warm-up!)
  (println "## enumerative-inference: linreg (20 observations), a k × k grid\n")
  (println "| points | one at a time | batched | speedup |")
  (println "|---|---|---|---|")
  (doseq [k [10 32 100 316]]
    (let [g (grid k)
          run (fn [opts] #(enum/enumerative-inference linreg [xs] observations g (assoc opts :key (rng/key 1))))]
      (row (str (count g)) (run {}) (run {:batched? true}))))
  (let [g (grid 316)]
    (row (str (count g) ", in chunks of 10,000")
         #(enum/enumerative-inference linreg [xs] observations g {:key (rng/key 1)})
         #(enum/enumerative-inference linreg [xs] observations g {:key (rng/key 1) :batched? true :chunk-size 10000})))
  (println "\n## with-exact-proposal: generate, n digits enumerated\n")
  (println "| assignments | one at a time | batched | speedup |")
  (println "|---|---|---|---|")
  (doseq [n [1 2 3]]
    (let [obs (cm/choicemap :y (- (Math/pow 10.0 n) 3.0))
          run (fn [opts] (let [g (enum/with-exact-proposal digits (mapv #(vector :d %) (range n)) opts)]
                           #(gfi/generate g [n] obs (rng/key 2))))]
      (row (str (long (Math/pow 10.0 n))) (run {}) (run {:batched? true}))))
  (shutdown-agents))
