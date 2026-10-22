;; V-P1-05 (verifier): a choice above a call that made no choices passes the
;; address check; what then happens when the call later makes choices.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm] '[gen.diff :as diff]
         '[gen.combinators :as comb])

(def k (gen [x] (trace :z (d/normal x 1.0))))
(def m (gen [xs] (splice [:a :b] (comb/map k) xs) (trace :a (d/normal 0.0 1.0))))

(let [tr (gfi/simulate m [[]] (rng/key 1))]
  (println "1. empty Map at [:a :b], then a choice at :a | choices" (cm/addresses (:choices tr)) "| score" (:score tr))
  (println "2. update to a non-empty Map:"
           (try (let [r (gfi/update tr [[0.0]] [diff/unknown-change] cm/EMPTY (rng/key 2))]
                  [(cm/addresses (:choices (:trace r))) (:weight r)])
                (catch Exception e (ex-message e))))
  (println "3. simulate with a non-empty Map:"
           (try (cm/addresses (:choices (gfi/simulate m [[0.0]] (rng/key 3))))
                (catch Exception e (ex-message e)))))
