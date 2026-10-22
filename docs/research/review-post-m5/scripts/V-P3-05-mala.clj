;; Verifier: does MALA reach the same NaN error? Funnel as in V-P3-05, MALA
;; on both choices; and a single MALA step at a far point with a large tau.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.inference.mcmc :as mcmc])
(def funnel (gen [] (let [v (trace :v (d/normal 0 3))] (trace :x (d/normal 0 (array/exp (array/* 0.5 v)))))))
(def start (:trace (gfi/generate funnel [] (cm/choicemap :v 0.0 :x 1.0) (rng/key 1))))
(doseq [tau [0.05 0.2 0.5]]
  (println "MALA tau" tau ":"
           (loop [i 0 t start acc 0 keys (rng/split (rng/key 2) 3000)]
             (if (empty? keys)
               {:steps i :accepted acc}
               (let [r (try (mcmc/mala t (sel/select :v :x) tau (first keys)) (catch Exception e {:threw (ex-message e)}))]
                 (if (:threw r)
                   {:threw-at-step i :message (:threw r) :v (t :v) :x (t :x)}
                   (recur (inc i) (:trace r) (+ acc (if (:accepted? r) 1 0)) (rest keys))))))))
