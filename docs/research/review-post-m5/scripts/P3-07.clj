;; trace-kernel/log-abs-det leaves out a trace choice that the update
;; rewrites when no output depends on its old value, as if it were left
;; alone. A kernel that overwrites mu with fresh noise and keeps nothing of
;; the old mu is then weighed as a bijection, silently.
;; Model: mu ~ N(0, 1), y ~ N(mu, 1), y = 0.8; posterior N(0.4, √0.5).
;; SMCP3 step to the same model (a move-reweight). A valid move's incremental
;; weight has E[exp w] = 1 under the posterior (the target is unchanged).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.trace-kernel :as tk])

(def model (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)))))
(def obs (cm/choicemap :y 0.8))

;; forgets the old mu: (mu, u) → mu' = u, no backward choices
(def k-forget (gen [tr] (let [u (trace :u (d/normal 0 1))] [(cm/choicemap :mu u) (cm/choicemap)])))
(def l-forget (gen [tr] [(cm/choicemap) (cm/choicemap)]))
;; the bijection: (mu, u) → (mu' = u, u' = mu)
(def k-swap (gen [tr] (let [u (trace :u (d/normal 0 1))] [(cm/choicemap :mu u) (cm/choicemap :u (tr :mu))])))

(defn posterior-trace [key]
  (let [[k1 k2] (rng/split key)
        mu (+ 0.4 (* (Math/sqrt 0.5) (array/item (rng/normal k1 []))))]
    (:trace (gfi/generate model [] (cm/merge obs (cm/choicemap :mu mu)) k2))))

(defn mean-exp-w [k l n]
  (let [step (tk/smcp3 {:k k :l l})
        ws (mapv (fn [key] (let [[k1 k2] (rng/split key)]
                             (Math/exp (double (array/item (:weight (step (posterior-trace k1) k2)))))))
                 (rng/split (rng/key 3) n))]
    (/ (reduce + ws) n)))

(let [tr (posterior-trace (rng/key 1))
      {kc :choices ret :retval} (gfi/propose k-forget [tr] (rng/key 2))]
  (println "log |det J| of the forgetting kernel (no error):"
           (tk/log-abs-det k-forget tr [] kc ret (constantly true))))
(println "E[exp w], the swap (a bijection), 20000 draws:" (mean-exp-w k-swap k-swap 20000))
(println "E[exp w], the forgetting kernel, 2000 / 20000 / 200000 draws:"
         (mean-exp-w k-forget l-forget 2000) (mean-exp-w k-forget l-forget 20000) (mean-exp-w k-forget l-forget 200000))
(println "the forgetting kernel with :check? true:"
         (try ((tk/smcp3 {:k k-forget :l l-forget :check? true}) (posterior-trace (rng/key 4)) (rng/key 5)) :no-error
              (catch Exception e (str "loud: " (ex-message e)))))

;; The same move as a transform under updating-translator (SMCP3 with a
;; transform): mu is overwritten from aux u and never read, nothing goes to
;; aux-out. Only the dimension check guards transforms, and it passes (1 = 1).
(require '[gen.inference.transform :as tf])
(def q-fwd (gen [tr] (trace :u (d/normal 0 1))))
(def q-bwd (gen [tr] nil))
(def overwrite (tf/transform [model-in aux-in model-out aux-out]
                 (tf/write model-out :mu (tf/read aux-in :u :continuous) :continuous)))
(let [step (tf/updating-translator {:q-forward q-fwd :q-forward-args [] :q-backward q-bwd :q-backward-args []
                                    :transform overwrite})
      n 20000
      ws (mapv (fn [key] (let [[k1 k2] (rng/split key)]
                           (Math/exp (double (array/item (:weight (step (posterior-trace k1) k2)))))))
               (rng/split (rng/key 6) n))]
  (println "transform version, E[exp w] over" n "draws:" (/ (reduce + ws) n)))

;; Randomness thrown away with no continuous output at all: the early return
;; (empty outs → 0.0) skips the check. Kernel: u ~ N(0, 1), flag' = u > 0.
(def flagged (gen [] (trace :flag (d/bernoulli 0.5)) (trace :mu (d/normal 0 1))))
(def k-discrete (gen [tr] (let [u (trace :u (d/normal 0 1))] [(cm/choicemap :flag (array/> u 0.0)) (cm/choicemap)])))
(let [tr (:trace (gfi/generate flagged [] (cm/choicemap :flag true) (rng/key 8)))
      {kc :choices ret :retval} (gfi/propose k-discrete [tr] (rng/key 9))]
  (println "u thrown away, only a discrete output: log |det J| ="
           (try (tk/log-abs-det k-discrete tr [] kc ret (constantly true)) (catch Exception e (str "loud: " (ex-message e))))))
