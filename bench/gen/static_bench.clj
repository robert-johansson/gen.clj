(ns gen.static-bench
  "Compiled scores (bean genclj-rj78): a static generative function against
   the same bodies without the flag, on GenMLX's linear regression with a
   static kernel in its Map. Times assess, choice-gradients, and HMC and
   MALA steps plain and compiled, forcing each result (MLX is lazy), over
   five rounds after warm-up; prints each op's median and range, in µs.
   Run: clojure [-J-Dgen.backend=mlx [-J-Dgen.mlx.device=cpu]] -M:bench -m gen.static-bench"
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as array :refer [+ *]]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def point (gen ^:static [x slope intercept] (let [y (trace :y (d/normal (+ (* slope x) intercept) 1))] y)))

(def linreg
  (gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          n (count xs)
          ys (splice :ys (comb/map point) xs (vec (repeat n slope)) (vec (repeat n intercept)))]
      ys)))

(def plain-point (assoc point :static nil))

(def plain-linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          n (count xs)
          ys (splice :ys (comb/map plain-point) xs (vec (repeat n slope)) (vec (repeat n intercept)))]
      ys)))

(defn- round
  "µs per call over n calls of f."
  [f n]
  (let [t (System/nanoTime)]
    (dotimes [_ n] (f))
    (/ (- (System/nanoTime) t) 1e3 n)))

(defn- timed
  "[median low high] µs per call of f over five rounds of n calls, after
   max(20, n) warm-up calls."
  [f n]
  (dotimes [_ (max 20 n)] (f))
  (let [rs (sort (repeatedly 5 #(round f n)))]
    [(nth rs 2) (first rs) (last rs)]))

(defn- ops [tr s]
  (let [xs (first (:args tr))]
    [["assess" 1 #(array/eval! (:weight (gfi/assess (:gen-fn tr) [xs] (:choices tr) (rng/key 2))))]
     ["choice-gradients" 1 #(array/eval! (cm/to-map (:gradients (grad/choice-gradients tr s))))]
     ["HMC, L = 10, plain" 10 #(array/eval! (:score (:trace (mcmc/hmc tr s {:L 10 :eps 0.001} (rng/key 3)))))]
     ["HMC, L = 10, compiled" 10 #(array/eval! (:score (:trace (mcmc/hmc tr s {:L 10 :eps 0.001 :compile? true} (rng/key 3)))))]
     ["MALA, plain" 1 #(array/eval! (:score (:trace (mcmc/mala tr s 1e-5 (rng/key 3)))))]
     ["MALA, compiled" 1 #(array/eval! (:score (:trace (mcmc/mala tr s 1e-5 {:compile? true :key (rng/key 3)}))))]]))

(defn- fmt [[m lo hi]] (format "%8.1f (%.1f–%.1f)" m lo hi))

(defn -main [& _]
  (println "backend" (p/-name array/*backend*) "device" (System/getProperty "gen.mlx.device" "default"))
  (doseq [n-obs [10 100]]
    (let [xs (vec (map double (range n-obs)))
          obs (reduce (fn [m i] (cm/set-choice m [:ys i :y] (clojure.core/* 0.5 i))) cm/EMPTY (range n-obs))
          s (sel/select :slope :intercept)
          ts (:trace (gfi/generate linreg [xs] obs (rng/key 1)))
          td (:trace (gfi/generate plain-linreg [xs] obs (rng/key 1)))
          calls (if (= n-obs 10) 2000 300)]
      (println (format "N = %d: median (range) µs per call over 5 rounds, static / dynamic" n-obs))
      (doseq [[[label cost f] [_ _ g]] (map vector (ops ts s) (ops td s))]
        (let [n (max 10 (quot calls cost))]
          (println (format "  %-22s %s   %s" label (fmt (timed f n)) (fmt (timed g n))))
          (flush)))))
  (shutdown-agents))
