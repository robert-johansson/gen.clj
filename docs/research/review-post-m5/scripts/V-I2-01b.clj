;; V-I2-01b (verifier): the third review's B2 scenario (combinators_test.clj
;; switches-of-compatible-branches-keep-shared-addresses), with one branch a
;; Map and the other a dynamic function making the same addresses: an
;; update of :i alone loses the observations.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.combinators :as comb])
(def ys (gen [mu] (trace :y (d/normal mu 1))))
(def as-map (comb/map ys))
(def as-dyn (gen [mus] (mapv (fn [i] (splice i (gen [mu] (trace :y (d/normal (+ mu 1.0) 1))) (nth mus i))) (range (count mus)))))
(doseq [[label sw] [["two Maps (the test's case)" (comb/switch as-map (comb/map (gen [mu] (trace :y (d/normal (+ mu 1.0) 1)))))]
                    ["a Map, then a dynamic gf " (comb/switch as-map as-dyn)]]]
  (let [m (gen [] (splice :s sw (trace :i (d/categorical [0.5 0.5])) [0.0 0.0 0.0]))
        obs (cm/choicemap :i 0 [:s 0 :y] 1.5 [:s 1 :y] 1.7 [:s 2 :y] 2.2)
        tr (:trace (gfi/generate m [] obs (rng/key 1)))
        {t :trace w :weight} (gfi/update tr (cm/choicemap :i 1) (rng/key 2))]
    (println label "| observations after the update:" (mapv #(t [:s % :y]) (range 3)) "| weight" w)))
