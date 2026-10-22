;; V-P1-04 (verifier): return values that are lazy but not LazySeq instances,
;; by other forms than the reviewer's: conj onto a lazy map (a Cons), a vector
;; holding a Cons, and a lazy-seq whose body is a cons (realized: control).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm])

(def xs [0 1 2])

(def conj-body (gen [] (conj (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) xs) 0.0)))
(def vec-of-cons (gen [] [(cons 0.0 (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) xs))]))
(def lazy-cons (gen [] (lazy-seq (cons 0.0 (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) xs)))))

(doseq [[label gf] [["(conj (map …) 0.0)      " conj-body] ["[(cons 0.0 (map …))]    " vec-of-cons]
                    ["(lazy-seq (cons …)), control" lazy-cons]]]
  (let [tr (gfi/simulate gf [] (rng/key 1))]
    (println label "| class" (.getSimpleName (class (:retval tr)))
             "| choices" (cm/addresses (:choices tr)) "| score" (:score tr))
    (println "   generate with [:x 1] constrained:"
             (try (:weight (gfi/generate gf [] (cm/from-map {[:x 1] 0.5}) (rng/key 2)))
                  (catch Exception e (ex-message e))))))

;; 4. conj onto a longer lazy map: conj realizes the first chunk only (32
;;    elements), so the trace holds part of the choices
(def long-conj (gen [] (conj (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) (range 100)) 0.0)))
(let [tr (gfi/simulate long-conj [] (rng/key 1))]
  (println "4. (conj (map … (range 100)) 0.0) | class" (.getSimpleName (class (:retval tr)))
           "| choices after the run" (count (cm/addresses (:choices tr))) "of 100"
           "| reading them all:" (try (count (doall (:retval tr))) (catch Exception e (ex-message e)))))
