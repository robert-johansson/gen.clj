;; Verifier, P2-03: a dist picking a vector label (array/take of a [K 2] array along axis 0).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.dist.dsl :refer [dist]] '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(def points (array/array [[0.0 0.0] [1.0 1.0] [2.0 5.0]]))
(dist pick-point [probs] (array/take points (d/categorical probs) 0))
(let [dd (pick-point [0.2 0.3 0.5])
      score (fn [v] (:weight (gfi/assess dd [] (cm/->Value (array/array v)) (rng/key 1))))]
  (println "sample:" (dc/sample dd (rng/key 2)))
  (println "score of [1 1]:" (score [1.0 1.0]) "; exact log 0.3 =" (Math/log 0.3))
  (println "score of [1 5] (no point):" (score [1.0 5.0]) "; exact -Inf")
  (println "score of [0 5] (no point):" (score [0.0 5.0]) "; exact -Inf"))
;; the same with host labels (nth on a vector of vectors) compares whole values with =
(dist pick-host [probs] (nth [[0.0 0.0] [1.0 1.0] [2.0 5.0]] (d/categorical probs)))
(let [dd (pick-host [0.2 0.3 0.5])]
  (println "host labels: log-prob of [1 1]" (dc/log-prob dd [1.0 1.0]) "; of [1 5]" (dc/log-prob dd [1.0 5.0])))
