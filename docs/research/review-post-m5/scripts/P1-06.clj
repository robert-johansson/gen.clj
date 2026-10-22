;; P1-06: gfi/project projects by the trace's :gen-fn, update and regenerate by
;; its maker (revising a trace given another gen-fn): a trace given another
;; gen-fn is projected with the new gen-fn's -project over the maker's records.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.combinators :as comb])

(def g1 (gen [] (trace :x (d/normal 0.0 1.0))))
(def g2 (gen [] (trace :x (d/normal 0.0 3.0))))
(let [tr (:trace (gfi/generate g1 [] (cm/from-map {:x 1.0}) (rng/key 1)))
      moved (assoc tr :gen-fn g2)]
  (println "dynamic trace given g2 | project all" (gfi/project moved sel/all)
           "| g1's score" (:score tr)
           "| g2's density of the same choices" (:weight (gfi/assess g2 [] (:choices tr) (rng/key 2)))
           "| update's weight (revise: g2 minus g1)" (:weight (gfi/update moved cm/EMPTY (rng/key 3)))))
(let [k (gen [x] (trace :y (d/normal x 1.0)))
      tr (:trace (gfi/generate (comb/map k) [[0.0 1.0]] (cm/from-map {0 {:y 0.5} 1 {:y 0.5}}) (rng/key 1)))
      moved (assoc tr :gen-fn (gen [xs] (trace [0 :y] (d/normal 0.0 1.0)) (trace [1 :y] (d/normal 1.0 1.0))))]
  (println "Map trace given a dynamic gen-fn | project all" (gfi/project moved sel/all) "| score" (:score tr)))
