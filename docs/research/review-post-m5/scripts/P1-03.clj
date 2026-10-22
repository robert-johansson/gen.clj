;; P1-03: update's retdiff compares the return values with clojure.core/=, which on
;; an infinite lazy return value (DESIGN: "Realization stops after 10,000
;; elements of a return value") never ends when the values are equal.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm])

(def m (gen []
         (let [x0 (trace :x0 (d/normal 0.0 1.0))]
           (trace :y (d/normal x0 1.0))
           (iterate #(* 0.9 %) x0))))

(let [tr (gfi/simulate m [] (rng/key 1))
      _ (println "simulate ok; first values" (take 3 (:retval tr)))
      ;; constrain only the observation: x0, and so the return value, are unchanged
      f (future (gfi/update tr (cm/from-map {:y 0.5}) (rng/key 2)))
      r (deref f 20000 ::timeout)]
  (println "update with a new :y ->" (if (= r ::timeout) "still running after 20 s" (select-keys r [:weight :retdiff])))
  (when (= r ::timeout)
    (let [rt (Runtime/getRuntime)]
      (println "heap used (MB):" (quot (- (.totalMemory rt) (.freeMemory rt)) 1048576))))
  (shutdown-agents)
  (System/exit 0))
