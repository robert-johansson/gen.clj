;; Verifier's independent check of P3-05 on Neal's funnel, the textbook
;; target where HMC diverges now and then (Stan reports such transitions as
;; divergent and rejects them): v ~ N(0, 3), x ~ N(0, e^{v/2}). HMC on both
;; at settings fit for the funnel's wide part, from v = 0, x = 1. A
;; divergence has acceptance probability 0, so the chain should go on.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.inference.mcmc :as mcmc])

(def funnel (gen [] (let [v (trace :v (d/normal 0 3))] (trace :x (d/normal 0 (array/exp (array/* 0.5 v)))))))
(def start (:trace (gfi/generate funnel [] (cm/choicemap :v 0.0 :x 1.0) (rng/key 1))))

(doseq [[L eps] [[10 0.2] [20 0.3] [10 0.5]]]
  (let [result (loop [i 0 t start acc 0 lowest 0.0 keys (rng/split (rng/key 2) 2000)]
                 (if (empty? keys)
                   {:steps i :accepted acc :lowest-v lowest}
                   (let [r (try (mcmc/hmc t (sel/select :v :x) {:L L :eps eps} (first keys))
                                (catch Exception e {:threw (ex-message e)}))]
                     (if (:threw r)
                       {:threw-at-step i :message (:threw r) :accepted acc :lowest-v lowest :v (t :v) :x (t :x)}
                       (recur (inc i) (:trace r) (+ acc (if (:accepted? r) 1 0))
                              (min lowest (double ((:trace r) :v))) (rest keys))))))]
    (println "L" L "eps" eps ":" result)))
