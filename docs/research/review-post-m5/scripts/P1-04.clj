;; P1-04: realize! forces only clojure.lang.LazySeq, so a return value that is a
;; lazy sequence of another class (a Cons onto a lazy tail, iterate's Iterate)
;; keeps its trace calls for after the run.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm])

(def xs [0 1 2])

(def lazy-seq-body (gen [] (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) xs)))
(def cons-body (gen [] (cons 0.0 (map (fn [i] (trace [:x i] (d/normal 0.0 1.0))) xs))))
(def iterate-body (gen [] (take 4 (iterate (fn [[i x]] [(inc i) (trace [:x i] (d/normal x 1.0))]) [0 0.0]))))
(def bare-iterate (gen [] (iterate (fn [[i x]] [(inc i) (trace [:x i] (d/normal x 1.0))]) [0 0.0])))

(doseq [[label gf] [["(map …)        " lazy-seq-body] ["(cons x (map …))" cons-body]
                    ["(take 4 (iterate …))" iterate-body] ["(iterate …)     " bare-iterate]]]
  (let [tr (gfi/simulate gf [] (rng/key 1))]
    (println label "| class" (.getSimpleName (class (:retval tr))) "| choices after the run" (cm/addresses (:choices tr))
             "| score" (:score tr))
    (println "   reading 3 elements:" (try (doall (take 3 (:retval tr))) :ok (catch Exception e (ex-message e))))
    (println "   generate with [:x 0] constrained:"
             (try (:weight (gfi/generate gf [] (cm/from-map {[:x 0] 0.5}) (rng/key 2)))
                  (catch Exception e (ex-message e))))))
