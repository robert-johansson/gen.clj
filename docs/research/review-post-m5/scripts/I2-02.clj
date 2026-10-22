;; I2-02: a gen nested in a static body is recognized by its symbol's name only
;; (static.clj outside-gens), so an aliased (dyn/gen …) has its inner trace
;; hoisted into the outer static body.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.dynamic.static :as static])

(def plain (gen ^:static [] (let [g (gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g)] y)))
(def aliased (dyn/gen ^:static [] (let [g (dyn/gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g)] y)))
(def dynamic (dyn/gen [] (let [g (dyn/gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g)] y)))

(doseq [[label gf] [["static, gen" plain] ["static, dyn/gen" aliased] ["dynamic body" dynamic]]]
  (println label "choices:" (cm/to-map (:choices (gfi/simulate gf [] (rng/key 1))))
           " nodes:" (mapv (juxt :kind :addr) (:nodes (static/ir gf)))))
