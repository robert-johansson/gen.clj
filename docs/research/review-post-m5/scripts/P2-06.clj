;; P2-06: a homogeneous mixture of vector-valued normal components mixes each coordinate separately
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

;; Gen.jl's HomogeneousMixture(broadcasted_normal, [1, 1]): two clusters in
;; the plane, at (0, 0) and (10, 10), unit diagonal covariance
(def gmm (d/homogeneous-mixture d/normal [1 1]))
(def mix (gmm [0.5 0.5] [[0.0 0.0] [10.0 10.0]] [[1.0 1.0] [1.0 1.0]]))
(def v (array/array [0.0 10.0]))   ; one coordinate from each cluster
(let [lp1 (array/item (array/sum (dc/log-prob (d/normal [0.0 0.0] [1.0 1.0]) v)))
      lp2 (array/item (array/sum (dc/log-prob (d/normal [10.0 10.0] [1.0 1.0]) v)))
      exact (+ (Math/log 0.5) (max lp1 lp2) (Math/log1p (Math/exp (- (min lp1 lp2) (max lp1 lp2)))))]
  (println "log density of (0, 10): mixture of the two components (Gen.jl):" exact)
  (println "  gen.clj, as a trace site scores it:" (:weight (gfi/assess mix [] (cm/->Value v) (rng/key 1))))
  (println "  per coordinate:" (dc/log-prob mix v)))
;; the draws: how often the two coordinates come from different clusters
(let [xs (map #(array/->clj (dc/sample mix %)) (rng/split (rng/key 2) 20000))
      split (count (filter (fn [[a b]] (> (Math/abs (- a b)) 5.0)) xs))]
  (println "draws with one coordinate near each cluster:" split "of 20000 (a mixture of the components: none)"))
;; the same mixture with mvnormal components mixes whole vectors
(let [mv ((d/homogeneous-mixture d/mvnormal [1 2]) [0.5 0.5] [[0.0 0.0] [10.0 10.0]]
          [[[1.0 0.0] [0.0 1.0]] [[1.0 0.0] [0.0 1.0]]])]
  (println "with mvnormal components (identity covariance):" (dc/log-prob mv v)))
