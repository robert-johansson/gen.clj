(ns gen.mcmc-bench
  "Plain against compiled HMC and MALA steps (docs/array-api.md §8), on
   Bayesian linear regression with N observations held as one array and two
   latents, w and b. Each line checks that the two kernels make the same move,
   then times 20 steps after warm-up, forcing each step's score, and counts
   the JVM collections in them (MLX collects after every
   gen.mlx.gc-new-bytes of new arrays).
   Run: clojure [-J-Dgen.backend=mlx] [-J-Dgen.mlx.gc-new-bytes=…] -M:bench -m gen.mcmc-bench"
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as a :refer [+ - * /]]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel])
  (:import [java.lang.management GarbageCollectorMXBean ManagementFactory]))

(def regression
  (gen [xs]
    (let [w (trace :w (d/normal 0 10))
          b (trace :b (d/normal 0 10))]
      (trace :y (d/normal (+ (* w xs) b) 1.0))
      w)))

(defn- collections []
  (reduce + (map #(.getCollectionCount ^GarbageCollectorMXBean %)
                 (ManagementFactory/getGarbageCollectorMXBeans))))

(defn- per-step
  "[mean ms per call, collections] over n calls of f, after 3 warm-ups."
  [f n]
  (dotimes [_ 3] (f))
  (let [g0 (collections) t0 (System/nanoTime)]
    (dotimes [_ n] (f))
    [(/ (- (System/nanoTime) t0) 1e6 n) (- (collections) g0)]))

(defn -main [& _]
  (println "backend" (p/-name a/*backend*)
           "gc-new-bytes" (System/getProperty "gen.mlx.gc-new-bytes" "default"))
  (doseq [n [100 10000 1000000]]
    (let [xs (rng/normal (rng/key 1) [n])
          ys (+ (* 2.0 xs) 0.5 (rng/normal (rng/key 2) [n]))
          tr (:trace (gfi/generate regression [xs] (cm/choicemap :y ys) (rng/key 3)))
          s (sel/select :w :b)
          kernels [["HMC " #(mcmc/hmc tr s {:L 10 :eps 0.001} (rng/key 4))
                    #(mcmc/hmc tr s {:L 10 :eps 0.001 :compile? true} (rng/key 4))]
                   ["MALA" #(mcmc/mala tr s 1e-5 (rng/key 4))
                    #(mcmc/mala tr s 1e-5 {:compile? true :key (rng/key 4)})]]]
      (doseq [[label plain compiled] kernels]
        (let [r1 (plain) r2 (compiled)
              same? (and (= (:accepted? r1) (:accepted? r2))
                         (= (a/item ((:trace r1) :w)) (a/item ((:trace r2) :w))))
              [tp gp] (per-step #(a/eval! (:score (:trace (plain)))) 20)
              [tc gc] (per-step #(a/eval! (:score (:trace (compiled)))) 20)]
          (printf "N=%-8d %s same move: %-5s | per step: plain %7.2f ms (%2d gcs)  compiled %7.2f ms (%2d gcs)%n"
                  n label same? tp gp tc gc)
          (flush)))))
  (shutdown-agents))
