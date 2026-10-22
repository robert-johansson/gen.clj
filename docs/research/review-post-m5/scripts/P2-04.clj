;; P2-04: a pushforward of a non-enumerable discrete draw gives mass to values off its lattice
(require '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.dist.dsl :refer [dist]] '[gen.random :as rng] '[gen.array :as array])

(dist micro-count [lam] (array/* 1e-6 (d/poisson lam)))      ; values k·1e-6
(dist reciprocal [lam] (array// 1.0 (array/+ 1 (d/poisson lam))))  ; values 1/(k+1)
(let [dd (micro-count 3.0)]
  (println "micro-count(3): log-prob at 2e-6 (k = 2)" (dc/log-prob dd 2e-6) "exact" (dc/log-prob (d/poisson 3.0) 2))
  (println "  at 1.5e-6 (no k)" (dc/log-prob dd 1.5e-6) "exact -Inf")
  (println "  at 2.4e-6 (no k)" (dc/log-prob dd 2.4e-6) "exact -Inf")
  ;; its total mass over a fine grid of [0, 1e-5]: the true mass of [0, 1e-5] is P(K <= 10)
  (let [grid (map #(* % 1e-8) (range 0 1001))
        mass (reduce + (map #(Math/exp (double (dc/log-prob dd %))) grid))]
    (println "  summed mass over the grid 0, 1e-8, ..., 1e-5 (a pmf sums to at most 1):" mass)))
(let [dd (reciprocal 500.0)]
  (println "reciprocal(500): at 1/501 (k = 500)" (dc/log-prob dd (/ 1.0 501)) "; at 1/501.5 (no k)" (dc/log-prob dd (/ 1.0 501.5))))
