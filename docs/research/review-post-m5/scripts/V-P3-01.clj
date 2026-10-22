;; Verifier's independent check of P3-01, with a discrete latent q does not
;; propose (so no pathwise term exists, only the score-function one).
;; Model: z1 ~ N(0, 1) (q proposes it, q = N(m, 1) at m = 0, the prior, so
;; z1 cancels), b ~ bernoulli(θ) (the model's internal proposal draws it),
;; y ~ N(b ? 1 : -1, 1), y = 1, θ = 0.5.
;;   log w = log N(1; ±1, 1) = c (b true) or c - 2 (b false)
;;   ELBO(θ) = c - 2(1 - θ),  ∂ELBO/∂θ = 2
;;   ∇θ log p(trace) = ∂/∂θ log Bern(b; θ) = b/θ - (1-b)/(1-θ), mean 0
;; K-sample bound: with n ~ Bin(K, θ) of the b's true,
;;   L(θ) = c + Σ_n Bin(n; K, θ) log((n + (K - n) e^-2) / K),
;; differentiated exactly below.
(require '[gen.dynamic :refer [gen]] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.vi :as vi])

(def model (gen [] (let [z1 (trace :z1 (d/normal 0 1))
                         b (trace :b (d/bernoulli (param :theta)))]
                     (trace :y (d/normal (array/where b 1.0 -1.0) 1)))))
(def q (gen [] (trace :z1 (d/normal (param :m) 1))))
(def obs (cm/choicemap :y 1.0))

(defn mean-se [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (Math/sqrt (/ (reduce + (map #(Math/pow (- % m) 2) xs)) (* n (dec n))))]))

(defn choose [n k] (reduce (fn [acc i] (/ (* acc (- n i)) (inc i))) 1.0 (range k)))

(defn bound-derivative [K theta]
  (reduce + (for [n (range (inc K))]
              (* (choose K n)
                 (Math/log (/ (+ n (* (- K n) (Math/exp -2))) K))
                 (- (* n (Math/pow theta (dec n)) (Math/pow (- 1 theta) (- K n)))
                    (* (- K n) (Math/pow theta n) (Math/pow (- 1 theta) (- K n 1))))))))

(let [K 10]
  (doseq [[name estimator expected] [["elbo-gradient" vi/elbo-gradient 2.0]
                                     ["vimco-gradient" vi/vimco-gradient (bound-derivative K 0.5)]]]
    (let [runs (mapv #(estimator model [] obs q [] {:m 0.0} K {:key % :model-store {:theta 0.5}})
                     (rng/split (rng/key 21) 4000))
          gs (mapv #(double (array/item (:theta (:model-grads %)))) runs)]
      (println name ":model-grads mean, se over 4000 iterations:" (mean-se gs)
               " the objective's exact ∂/∂θ:" expected))))

;; The direction's VIMCO estimator, on the host for this model: the
;; score-function term for the internally proposed b_j takes VIMCO's own
;; learning signal, Σ_j [(L - L₋ⱼ) - w̃_j] ∇θ log f(b_j) + Σ_j w̃_j ∇θ log p(b_j);
;; here both gradients are s_j = b_j/θ - (1-b_j)/(1-θ), so it is
;; Σ_j (L - L₋ⱼ) s_j, against Σ_j w̃_j s_j (what :model-grads computes).
(let [K 10 theta 0.5 n 200000
      us (array/->clj (rng/uniform (rng/key 22) [n K]))
      lme (fn [xs] (let [m (apply max xs)] (+ m (Math/log (/ (reduce + (map #(Math/exp (- % m)) xs)) (count xs))))))
      est (fn [row]
            (let [bs (mapv #(< % theta) row)
                  lw (mapv #(if % 0.0 -2.0) bs)
                  s (mapv #(if % (/ 1 theta) (/ -1 (- 1 theta))) bs)
                  L (lme lw)
                  wn (let [m (apply max lw) e (mapv #(Math/exp (- % m)) lw) t (reduce + e)] (mapv #(/ % t) e))
                  loo (fn [j] (lme (assoc lw j (/ (- (reduce + lw) (lw j)) (dec K)))))]
              [(reduce + (map-indexed (fn [j sj] (* (- L (loo j)) sj)) s))
               (reduce + (map * wn s))]))
      rs (mapv est us)]
  (println "host VIMCO: the direction's estimator" (mean-se (map first rs))
           " Σ w̃ s" (mean-se (map second rs)) " exact" (bound-derivative K theta)))
