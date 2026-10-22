;; Verifier, P2-02: a spike-and-slab mixture (a point mass at 0 and a normal), the density at the atom.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(def spike-slab (d/heterogeneous-mixture [d/uniform-discrete d/normal]))
;; P(Y = 0) = w_spike exactly; at 0 the density w.r.t. counting + Lebesgue is w_spike, and at y ≠ 0 it is w_slab·φ(y)
(let [dd (spike-slab [0.3 0.7] [0 0] [0.0 1.0])]
  (println "log-prob at 0:" (dc/log-prob dd 0) "; w.r.t. counting + Lebesgue" (Math/log 0.3)
           "; what gen.clj adds:" (Math/log (+ 0.3 (* 0.7 (/ 1 (Math/sqrt (* 2 Math/PI)))))))
  (println "log-prob at 0.5:" (dc/log-prob dd 0.5) "; exact" (Math/log (* 0.7 (Math/exp (- (* 0.5 0.25))) (/ 1 (Math/sqrt (* 2 Math/PI))))))
  (println "P(Y = 0) by simulation:" (/ (count (filter #(== 0.0 (double %)) (array/->clj (dc/sample dd (rng/key 3) [100000])))) 100000.0) "; exact 0.3")
  (println "discrete?" (dc/discrete? dd)))
;; the posterior of the spike weight from one exact zero: w ∈ {0.1, 0.9}, prior ½ each
;; exact: P(w = 0.9 | y = 0) = 0.9/(0.9 + 0.1) = 0.9
(def m (gen [] (let [hi (trace :hi (d/bernoulli 0.5))
                     w (if hi 0.9 0.1)]
                 (trace :y (spike-slab [w (- 1.0 w)] [0 0] [0.0 1.0])))))
(let [lw (fn [hi] (double (array/item (:weight (gfi/generate m [] (cm/from-map {:hi hi :y 0}) (rng/key 1))))))
      a (Math/exp (lw true)) b (Math/exp (lw false))]
  (println "P(w = 0.9 | y = 0): gen.clj" (/ a (+ a b)) "; exact 0.9"))
