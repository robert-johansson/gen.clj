;; V-I2-02 (verifier): an aliased nested gen in a static body, on another
;; model than the reviewer's: one inner gen spliced at two addresses, which
;; the dynamic body draws twice and the aliased static body once (a
;; different model, not only a moved address). And the fully qualified
;; spelling, which outside-gens names, as a control.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.dynamic.static :as static])

(def two-plain (gen ^:static [] (let [g (gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g) z (splice :z g)] [y z])))
(def two-qualified (gen.dynamic/gen ^:static [] (let [g (gen.dynamic/gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g) z (splice :z g)] [y z])))
(def two-aliased (dyn/gen ^:static [] (let [g (dyn/gen [] (trace :a (d/normal 0.0 1.0))) y (splice :y g) z (splice :z g)] [y z])))

(doseq [[label gf] [["plain    " two-plain] ["qualified" two-qualified] ["aliased  " two-aliased]]]
  (let [t (gfi/simulate gf [] (rng/key 5))
        tp (gfi/simulate (assoc gf :static nil) [] (rng/key 5))]
    (println label "static retval" (:retval t) "choices" (cm/to-map (:choices t)) "score" (:score t))
    (println label "dynamic retval" (:retval tp) "choices" (cm/to-map (:choices tp)) "score" (:score tp))
    (println label "nodes" (mapv (juxt :kind :addr) (:nodes (static/ir gf))) "\n")))
