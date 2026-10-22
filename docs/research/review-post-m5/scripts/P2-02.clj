;; P2-02: a heterogeneous mixture of a discrete and a continuous family adds a pmf to a pdf
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(def mix (d/heterogeneous-mixture [d/poisson d/normal]))
;; w picks the weights: [0.2 0.8] or [0.8 0.2]; y = 0 observed
(def m (gen [] (let [w (trace :w (d/categorical [0.5 0.5]))]
                 (trace :y (mix (if (= w 0) [0.2 0.8] [0.8 0.2]) [3.0] [0.0 1.0])))))
(let [lw (fn [w] (double (array/item (:weight (gfi/generate m [] (cm/from-map {:w w :y 0}) (rng/key 1))))))
      l0 (Math/exp (lw 0)) l1 (Math/exp (lw 1))]
  (println "gen.clj: P(w = 1 | y = 0) =" (/ l1 (+ l0 l1)))
  ;; exact: y = 0 has probability 0 under the normal component, so the
  ;; likelihood is the Poisson part's weight times e^-3: P(w = 1 | y = 0) = 0.8
  (println "exact:   P(w = 1 | y = 0) =" (/ (* 0.8 (Math/exp -3)) (+ (* 0.8 (Math/exp -3)) (* 0.2 (Math/exp -3))))))
;; the sampler agrees with the exact answer: simulate, keep the runs with y = 0
(let [trs (map #(gfi/simulate m [] %) (rng/split (rng/key 2) 200000))
      hits (filter #(== 0.0 (double (cm/get-value (cm/get-submap (:choices %) :y)))) trs)
      w1 (count (filter #(= 1 (cm/get-value (cm/get-submap (:choices %) :w))) hits))]
  (println "simulated: runs with y = 0:" (count hits) ", of them w = 1:" w1 "=" (double (/ w1 (count hits)))))
(println "discrete? of the mixture:" (dc/discrete? (mix [0.5 0.5] [3.0] [0.0 1.0])))
