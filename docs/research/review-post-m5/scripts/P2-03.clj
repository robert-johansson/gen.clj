;; P2-03: a pushforward whose body broadcasts its draw to a larger shape is scored as several draws
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.dist.dsl :refer [dist]] '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

;; one coin, its value [b, 2b]: P([1 2]) = p, P([0 0]) = 1 - p
(dist coin-pair [p] (array/* (array/astype (d/bernoulli p) :float64) (array/array [1.0 2.0])))
(let [dd (coin-pair 0.3)]
  (println "sample:" (dc/sample dd (rng/key 1)) "value-shape:" (dc/value-shape dd))
  (println "log-prob [1 2]: per element" (dc/log-prob dd (array/array [1.0 2.0])) "; as a trace's score"
           (:weight (gfi/assess dd [] (cm/->Value (array/array [1.0 2.0])) (rng/key 1)))
           "; exact" (Math/log 0.3))
  (println "log-prob [1 0] (impossible):" (dc/log-prob dd (array/array [1.0 0.0])) "; as a score"
           (:weight (gfi/assess dd [] (cm/->Value (array/array [1.0 0.0])) (rng/key 1))) "; exact -Inf"))

;; one normal draw shared by two outputs: x + [0, 1] lies on a line, no density in R^2
(dist shifted-pair [mu] (array/+ (d/normal mu 1.0) (array/array [0.0 1.0])))
(let [dd (shifted-pair 0.0)]
  (println "shifted-pair log-prob [0.3 1.3] (on the line):" (dc/log-prob dd (array/array [0.3 1.3]))
           " [0.3 5.0] (off it, probability 0):" (dc/log-prob dd (array/array [0.3 5.0]))))

;; in a model: a trace's score counts the coin twice
(def m (gen [] (trace :v (coin-pair 0.3))))
(let [tr (:trace (gfi/generate m [] (cm/from-map {:v (array/array [1.0 2.0])}) (rng/key 2)))]
  (println "model score with v = [1 2]:" (:score tr) "exact" (Math/log 0.3)))
