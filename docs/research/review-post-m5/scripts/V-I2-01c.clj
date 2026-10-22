;; V-I2-01c (verifier): more blockers of the direction (revise decides every
;; pair), by with-redefs: a distribution's single value given to a Map, an
;; Unfold or a Recurse as old choices.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.combinators :as comb] '[gen.diff :as diff] '[gen.selection :as sel])
(defmacro probe [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body))
                             (catch Throwable e# (str "THROWS " (.getName (class e#)) ": " (ex-message e#))))))
(def kern (gen [m] (trace :x (d/normal m 1.0))))
(defn switch-update [a b args0 args1]
  (let [sw (comb/switch a b)
        t1 (gfi/simulate sw (into [0] args0) (rng/key 7))]
    (cm/to-map (:choices (:trace (gfi/update t1 (into [1] args1) (into [diff/unknown-change] (repeat (count args1) diff/no-change)) cm/EMPTY (rng/key 8)))))))
(with-redefs [dyn/compatible-callee? (constantly true)]
  (probe "normal -> map   " (switch-update (d/normal 0.0 1.0) (comb/map kern) [] [[0.0]]))
  (probe "normal -> unfold" (switch-update (d/normal 0.0 1.0) (comb/unfold (gen [t s] (trace :x (d/normal s 1.0)))) [] [1 0.0]))
  (probe "map -> normal   " (switch-update (comb/map kern) (d/normal 0.0 1.0) [[0.0]] [])))
(probe "as built: normal -> map" (switch-update (d/normal 0.0 1.0) (comb/map kern) [] [[0.0]]))
