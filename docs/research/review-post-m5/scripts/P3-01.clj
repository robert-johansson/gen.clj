;; VI's :model-grads when the model has a latent q does not propose, whose
;; distribution depends on θ. Model: z1 ~ N(0, 1) (q proposes it),
;; z2 ~ N(θ, 1) (the model's internal proposal draws it), y ~ N(z1 + z2, 1),
;; y = 3. q: z1 ~ N(m, 1), m = 0. The log weight is
;; log N(y; z1 + z2, 1) + log N(z1; 0, 1) - log N(z1; m, 1) (z2's prior
;; cancels against its proposal), so
;;   ELBO(θ) = E[-½(y - z1 - z2)²] + const,  ∂ELBO/∂θ = y - m - θ = 3 at θ = 0.
;; vi's model gradient is E[∇θ log p(z1, z2, y; θ)] = E[z2 - θ] = 0.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.vi :as vi])

(def model (gen [] (let [z1 (trace :z1 (d/normal 0 1))
                         z2 (trace :z2 (d/normal (param :theta) 1))]
                     (trace :y (d/normal (array/+ z1 z2) 1)))))
(def q (gen [] (trace :z1 (d/normal (param :m) 1))))
(def obs (cm/choicemap :y 3.0))

(defn mean-se [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (Math/sqrt (/ (reduce + (map #(Math/pow (- % m) 2) xs)) (* n (dec n))))]))

(doseq [[name estimator] [["elbo-gradient" vi/elbo-gradient] ["vimco-gradient" vi/vimco-gradient]]]
  (let [gs (mapv #(double (array/item (:theta (:model-grads (estimator model [] obs q [] {:m 0.0} 10
                                                                        {:key % :model-store {:theta 0.0}})))))
                 (rng/split (rng/key 1) 2000))]
    (println name ":model-grads mean, se over 2000 iterations:" (mean-se gs))))

;; The objective with one key is a function of θ (z2 = θ + ε under it): its
;; derivative by central difference, against :model-grads with that key
(let [k (rng/key 7) h 1e-4
      obj (fn [t] (:objective (vi/elbo-gradient model [] obs q [] {:m 0.0} 10 {:key k :model-store {:theta t}})))]
  (println "one key: central difference of the objective" (/ (- (obj h) (obj (- h))) (* 2 h))
           " :model-grads" (array/item (:theta (:model-grads (vi/elbo-gradient model [] obs q [] {:m 0.0} 10
                                                                                 {:key k :model-store {:theta 0.0}}))))))
(println "expected: the ELBO's ∂/∂θ = 3.0")
(let [k (rng/key 8) h 1e-4
      obj (fn [t] (:objective (vi/vimco-gradient model [] obs q [] {:m 0.0} 10 {:key k :model-store {:theta t}})))]
  (println "VIMCO, one key: central difference of the bound" (/ (- (obj h) (obj (- h))) (* 2 h))
           " :model-grads" (array/item (:theta (:model-grads (vi/vimco-gradient model [] obs q [] {:m 0.0} 10
                                                                                  {:key k :model-store {:theta 0.0}}))))))
