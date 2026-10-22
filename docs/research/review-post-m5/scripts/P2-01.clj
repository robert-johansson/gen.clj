;; P2-01: gamma, beta and dirichlet draws that underflow to 0 give +Inf or NaN scores
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(defn tally [label dist n]
  (let [ks (rng/split (rng/key 7) n)
        trs (mapv #(gfi/simulate dist [] %) ks)
        scores (mapv #(double (array/item (:score %))) trs)
        zero (count (filter #(let [v (:retval %)] (and (number? v) (zero? (double v)))) trs))]
    (println label ": n" n
             "| +Inf scores" (count (filter #(= ##Inf %) scores))
             "| NaN scores" (count (filter #(Double/isNaN %) scores))
             "| -Inf scores" (count (filter #(= ##-Inf %) scores))
             "| zero draws" zero)))

(tally "gamma(0.001, 1000) (BUGS' vague dgamma(0.001, 0.001))" (d/gamma 0.001 1000.0) 2000)
(tally "gamma(0.01, 1)" (d/gamma 0.01 1.0) 20000)
(tally "beta(0.001, 0.001)" (d/beta 0.001 0.001) 2000)
(tally "chi-squared(0.002) (floored, for contrast)" (d/chi-squared 0.002) 2000)

;; dirichlet: a sparse prior, alpha 0.01 over 1000 categories
(let [alpha (vec (repeat 1000 0.01))
      dist (d/dirichlet alpha)
      ks (rng/split (rng/key 8) 200)
      trs (mapv #(gfi/simulate dist [] %) ks)
      scores (mapv #(double (array/item (:score %))) trs)]
  (println "dirichlet(0.01 x 1000): n 200 | +Inf scores" (count (filter #(= ##Inf %) scores))
           "| NaN" (count (filter #(Double/isNaN %) scores))
           "| finite" (count (filter #(Double/isFinite %) scores))))

;; in a model: one simulate of a vague precision prior, and what IS sees
(def m (gen [] (let [tau (trace :tau (d/gamma 0.001 1000.0))]
                 (trace :y (d/normal 0.0 (array/sqrt (array// 1.0 (array/maximum tau 1e-300))))))))
(let [trs (mapv #(gfi/simulate m [] %) (rng/split (rng/key 9) 1000))]
  (println "model simulate: +Inf scores" (count (filter #(= ##Inf (double (array/item (:score %)))) trs)) "of 1000"))
(let [ws (mapv #(:weight (gfi/generate m [] (cm/from-map {:y 0.3}) %)) (rng/split (rng/key 10) 1000))]
  (println "generate with y observed (weights finite, the prior's score is not):" (count (filter #(Double/isFinite (double %)) ws)) "finite weights of 1000"))
(let [r (gfi/generate m [] (cm/from-map {:y 0.3}) (rng/key 11))
      tr (:trace r)]
  (println "a trace: tau" (:retval (:choices tr)) "tau's value" (cm/get-value (cm/get-submap (:choices tr) :tau)) "score" (:score tr) "weight" (:weight r)))

;; --- what inference does with such a trace ---
(require '[gen.inference.mh :as mh] '[gen.selection :as sel])
;; a vague gamma prior on a Poisson rate, a zero count observed
(def pm (gen [] (let [lam (trace :lam (d/gamma 0.001 1000.0))
                      x (trace :x (d/normal 0.0 1.0))]
                  (trace :y (d/poisson lam))
                  (trace :z (d/normal x 1.0)))))
(def obs (cm/from-map {:y 0 :z 0.5}))
;; a generate whose prior draw of lam underflowed
(def start (first (filter #(zero? (double (cm/get-value (cm/get-submap (:choices %) :lam))))
                          (map #(:trace (gfi/generate pm [] obs %)) (rng/split (rng/key 12) 50)))))
(println "start: lam" (cm/get-value (cm/get-submap (:choices start) :lam)) "score" (:score start))
;; a random-walk proposal on lam (custom-proposal MH: update-based)
(def rw (gen [tr] (trace :lam (d/gamma 2.0 (array/+ 1e-3 (tr :lam))))))
(let [chain (reductions (fn [tr k] (:trace (mh/mh tr rw [] k))) start (rng/split (rng/key 13) 2000))]
  (println "custom-proposal MH, 2000 steps from lam = 0: distinct lam values"
           (count (distinct (map #(cm/get-value (cm/get-submap (:choices %) :lam)) chain)))))
(let [r (mh/mh start rw [] (rng/key 14))] (println "one step's log-alpha:" (:log-alpha r)))
;; selection MH on x, with lam = 0 retained
(println "selection MH on :x:" (try (:log-alpha (mh/mh start (sel/select :x) (rng/key 15)))
                                    (catch Exception e (ex-message e))))
;; the same start with lam floored at the smallest normal double instead
(let [floored (:trace (gfi/update start (cm/from-map {:lam Double/MIN_NORMAL}) (rng/key 16)))
      chain (reductions (fn [tr k] (:trace (mh/mh tr rw [] k))) floored (rng/split (rng/key 13) 2000))]
  (println "from lam = MIN_NORMAL: score" (:score floored) "distinct lam values" (count (distinct (map #(cm/get-value (cm/get-submap (:choices %) :lam)) chain)))
           "| selection MH on :x log-alpha" (:log-alpha (mh/mh floored (sel/select :x) (rng/key 15)))))
;; beta's -Inf scores: NaN draws
(let [vs (mapv #(dc/sample (d/beta 0.001 0.001) %) (rng/split (rng/key 7) 2000))]
  (println "beta(0.001, 0.001) draws: NaN" (count (filter #(Double/isNaN (double %)) vs))
           "| exactly 0" (count (filter #(zero? (double %)) vs)) "| exactly 1" (count (filter #(== 1.0 (double %)) vs))))
