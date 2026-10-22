;; per-site costs of a distribution on host scalars: construction (with its
;; validation), the value-shape check, log-prob, and the sample's helpers
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng]
         '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.choicemap :as cm])
(defn time-ns [n f]
  (dotimes [_ n] (f))
  (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (/ (- (System/nanoTime) t0) (double n))))
(let [nd (d/normal 0.0 1.0) k (rng/key 0)]
  (println "construct (normal 0 1) ns  " (time-ns 500000 #(d/normal 0.0 1.0)))
  (println "check-value-shape ns       " (time-ns 500000 #(dc/check-value-shape nd 0.5)))
  (println "log-prob ns                " (time-ns 500000 #(dc/log-prob nd 0.5)))
  (println "sample ns                  " (time-ns 500000 #(dc/sample nd k))))
(let [m (gen [] (trace :x (d/normal 0.0 1.0)))
      c (cm/from-map {:x 0.5}) k (rng/key 1)]
  (println "generate, one constrained site ns" (time-ns 200000 #(gfi/generate m [] c k))))
