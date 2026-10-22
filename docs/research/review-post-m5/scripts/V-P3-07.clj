;; Verifier's independent check of P3-07: a move that discards a continuous
;; choice (rather than rewriting it), and so has no continuous output at
;; all, through the early return. Checked by quadrature, no sampling.
;; Model: flag ~ bernoulli(0.5); z ~ uniform(0, 1) when flag; y ~ N(flag ? z : 0, 1),
;; y = 1. SMCP3 step to the same target: k at flag = true sets flag false
;; (the update discards z) with no backward choices, and does nothing at
;; flag = false; l makes no choices. (true, z) ↦ false for every z: not
;; injective. Its weight is log p(false, y) - log p(true, z, y) on true and 0
;; on false, so under the posterior
;;   E[exp w] = P(false | y) + ∫₀¹ p(false, y) / p(y) dz = 2 P(false | y),
;; and P(false | y) = φ(1) / (φ(1) + Φ(1) - Φ(0)) = 0.41484, so 0.8297, where
;; a valid move gives 1. tk/log-abs-det should be loud; it returns 0.0.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.trace-kernel :as tk])

(def model (gen [] (let [flag (trace :flag (d/bernoulli 0.5))
                         m (if flag (trace :z (d/uniform 0 1)) 0.0)]
                     (trace :y (d/normal m 1)))))
(def k (gen [tr] (if (tr :flag) [(cm/choicemap :flag false) (cm/choicemap)] [(cm/choicemap) (cm/choicemap)])))
(def l (gen [tr] [(cm/choicemap) (cm/choicemap)]))
(def step (tk/smcp3 {:k k :l l}))

(defn trace-of [cs] (:trace (gfi/generate model [] (cm/from-map (assoc cs :y 1.0)) (rng/key 1))))
(defn exp-w [t] (Math/exp (double (array/item (:weight (step t (rng/key 2)))))))
(defn joint [t] (Math/exp (double (array/item (:score t)))))

(let [t (trace-of {:flag true :z 0.3})
      {kc :choices ret :retval} (gfi/propose k [t] (rng/key 3))]
  (println "log |det J| of the discarding move:" (tk/log-abs-det k t [] kc ret (constantly true))))

;; E[exp w] = Σ_flag ∫ p(t | y) exp w(t), by the midpoint rule over z
(let [n 2000
      false-t (trace-of {:flag false})
      zs (map #(/ (+ % 0.5) n) (range n))
      true-ts (map #(trace-of {:flag true :z %}) zs)
      p-false (joint false-t)
      p-true (/ (reduce + (map joint true-ts)) n)
      evidence (+ p-false p-true)
      e (/ (+ (* p-false (exp-w false-t)) (/ (reduce + (map #(* (joint %) (exp-w %)) true-ts)) n)) evidence)]
  (println "P(false | y) =" (/ p-false evidence) " E[exp w] by quadrature =" e
           " derived 2 P(false | y) =" (* 2 (/ p-false evidence)) " a valid move: 1"))
