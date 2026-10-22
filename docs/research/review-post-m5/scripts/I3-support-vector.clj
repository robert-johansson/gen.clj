;; Can a draw's log-probs over its whole support come from one call?
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc])
(let [c (d/categorical [0.2 0.3 0.5])
      b (d/bernoulli 0.3)
      u (d/uniform-discrete 2 5)]
  (println (array/->clj (dc/log-prob c (array/array (dc/support c)))))
  (println (array/->clj (dc/log-prob b (array/array (dc/support b)))))
  (println (array/->clj (dc/log-prob u (array/array (dc/support u))))))
