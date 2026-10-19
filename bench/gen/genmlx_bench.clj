(ns gen.genmlx-bench
  "GenMLX's bench/cost_per_particle.cljs in gen.clj, for the same-hardware
   comparison (bean genclj-ls29): a normal-normal model (one latent, five
   observations), scalar against batched, for generate, importance sampling,
   MH and a 5-step particle filter. Each point is the median of timed runs
   after a warm-up, as GenMLX's harness, and forces evaluation of its result.

   Runs on the default backend:
     clojure -M:bench -m gen.genmlx-bench                                  JVM
     clojure -J-Dgen.backend=mlx -M:bench -m gen.genmlx-bench              MLX, GPU
     clojure -J-Dgen.backend=mlx -J-Dgen.mlx.device=cpu -M:bench -m gen.genmlx-bench"
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.protocols :as p]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.kernel :as k]
            [gen.inference.particle-filter :as pf]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ys [2.8 3.1 2.9 3.2 3.0])

(def model
  "GenMLX's: mu ~ N(0, 10), y_i ~ N(mu, 1)."
  (gen [ys]
    (let [mu (trace :mu (d/normal 0 10))]
      (doseq [[i _] (map-indexed vector ys)]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def obs (apply cm/choicemap (mapcat (fn [i] [[:y i] (ys i)]) (range 5))))

(def growing
  "The same model with t + 1 observations, for the particle filter: a step
   adds y_t (gen.clj's filters extend traces; GenMLX's smc constrains a
   choice the model already has)."
  (gen [t]
    (let [mu (trace :mu (d/normal 0 10))]
      (dotimes [i (inc t)]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(defn- time-ms [f]
  (let [t0 (System/nanoTime)]
    (f)
    (/ (clojure.core/- (System/nanoTime) t0) 1e6)))

(defn- bench
  "Median of reps timed runs after warm-ups (more for cheap points, so the
   JIT has compiled them)."
  [reps f]
  (dotimes [_ (if (clojure.core/< (time-ms f) 50.0) 20 2)] (f))
  (nth (sort (mapv (fn [_] (time-ms f)) (range reps))) (quot reps 2)))

(defn- row [label n units ms]
  (println (format "  %-22s N=%-5d total %9.1f ms  |  %.4f ms/%s"
                   label n ms (clojure.core// ms units) (if (= units n) "particle" "unit")))
  (flush))

(declare run)

(defn- reps-for [scalar? n] (if (and scalar? (clojure.core/>= n 1000)) 3 7))

(def ^:private spot-checks
  "-Dgen.bench.spot-checks=0 turns gen.batched's spot checks off (GenMLX
   has none; each is a scalar run, about 0.5 ms on MLX)."
  (some-> (System/getProperty "gen.bench.spot-checks") Long/parseLong))

(defn -main [& _]
  (binding [b/*spot-checks* (or spot-checks b/*spot-checks*)]
    (run))
  (shutdown-agents))

(defn run []
  (let [backend array/*backend*]
    (println "backend:" (p/-name backend)
             (if (= :mlx (p/-name backend))
               (str ((requiring-resolve 'gen.backend.mlx/device) backend) " " (:float (p/-defaults backend)))
               (:float (p/-defaults backend)))
             "| spot checks" b/*spot-checks*
             "|" (System/getProperty "java.vm.name") (System/getProperty "java.version"))

    (println "\n== Anchor: tiny graph + host read ==")
    (let [x (array/array 1.5)]
      (row "eval-loop" 200 200 (bench 7 #(dotimes [_ 200] (array/item (+ x 0.5))))))

    (println "\n== Pair 1: generate — scalar loop vs batched ==")
    (doseq [n [1 10 100 1000 3000]]
      (row "scalar  generate" n n
           (bench (reps-for true n)
                  #(doseq [key (rng/split (rng/key 7) n)]
                     (array/item (:weight (gfi/generate model [ys] obs key)))))))
    (doseq [n [1 10 100 1000 3000 10000]]
      (row "batched generate" n n
           (bench 7 #(array/->clj (:weights (b/generate model [ys] obs n (rng/key 7)))))))

    (println "\n== Pair 2: importance sampling — scalar vs batched ==")
    (doseq [n [1 10 100 1000]]
      (row "scalar  importance" n n
           (bench (reps-for true n) #(:log-ml-estimate (imp/importance-sampling model [ys] obs n {:key (rng/key 11)})))))
    (doseq [n [1 10 100 1000 3000 10000]]
      (row "batched importance" n n
           (bench 7 #(:log-ml-estimate (imp/importance-sampling model [ys] obs n {:key (rng/key 11) :batched? true})))))

    (println "\n== Pair 3: MH (10 steps on :mu) — scalar chains ==")
    (let [kernel (k/mh (sel/select :mu))
          start (fn [i] (:trace (gfi/generate model [ys] obs (rng/key (clojure.core/+ 20 i)))))]
      (doseq [c [1 5 20]]
        (row "scalar  mh chains" c (clojure.core/* c 10)
             (bench (reps-for true (clojure.core/* c 100))
                    #(dotimes [i c]
                       (array/item (:score (peek (:traces (k/run kernel (start i) {:samples 10 :key (rng/key i)}))))))))))

    (println "\n== Pair 4: particle filter (5 steps) — scalar vs batched ==")
    (let [run (fn [n batched?]
                (let [ks (rng/split (rng/key 31) 5)
                      init (pf/initialize growing [0] (cm/choicemap [:y 0] (ys 0)) n {:key (ks 0) :batched? batched?})]
                  (pf/log-ml-estimate
                   (reduce (fn [state t]
                             (pf/step state [t] [diff/unknown-change] (cm/choicemap [:y t] (ys t)) {:key (ks t)}))
                           init (range 1 5)))))]
      (doseq [n [1 10 100 500]]
        (row "scalar  filter" n (clojure.core/* n 5) (bench (reps-for true n) #(run n false))))
      (doseq [n [1 10 100 1000 3000 10000]]
        (row "batched filter" n (clojure.core/* n 5) (bench 7 #(run n true)))))
    (println "\n== done ==")))
