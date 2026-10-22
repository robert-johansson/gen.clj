(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng])
(defmacro show [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body)) (catch Throwable e# (str (.getSimpleName (class e#)) ": " (ex-message e#))))))

;; piecewise-uniform with a leading batch axis on its parameters: value-shape
;; claims [2]; does sampling/log-prob treat row b with bounds[b], probs[b]?
(let [bounds (array/array [[0.0 1.0 2.0] [10.0 11.0 12.0]])
      probs (array/array [[0.5 0.5] [0.5 0.5]])
      pu (d/piecewise-uniform bounds probs)]
  (show "value-shape" (dc/value-shape pu))
  (show "sample" (array/->clj (dc/sample pu (rng/key 3))))
  (show "log-prob [0.5 10.5] (expect [log 0.5, log 0.5] = -0.693 each)" (array/->clj (dc/log-prob pu (array/array [0.5 10.5])))))

;; float32 on the JVM backend
(show "float32 array" (array/dtype (array/array [1.0 2.0] {:dtype :float32})))
;; dirichlet of 1000 components in float32: are its own draws in its support?
(let [alpha (array/full [1000] 1.0 {:dtype :float32})
      dd (d/dirichlet alpha)
      lps (for [i (range 20)] (array/item (dc/log-prob dd (dc/sample dd (rng/key i)))))]
  (show "float32 dirichlet(1000) own draws scored -Inf" (count (filter #(= % Double/NEGATIVE_INFINITY) lps)))
  (show "  sum error of one draw" (array/item (array/- (array/sum (dc/sample dd (rng/key 0))) 1.0))))
