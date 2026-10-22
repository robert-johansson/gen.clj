;; array/diag of a vector and la/diagonal of a matrix multiply by the
;; identity, so a non-finite entry spreads NaN where the result should hold
;; the entry (la/diagonal) or 0 (array/diag): 0 · Inf and 0 · NaN are NaN.
(require '[gen.array :as array] '[gen.array.linalg :as la])
(println "array/diag [1 Inf]   ->" (array/->clj (array/diag (array/array [1.0 ##Inf]))) " expected [[1 0] [0 Inf]]")
(println "array/diag [NaN 1]   ->" (array/->clj (array/diag (array/array [##NaN 1.0]))) " expected [[NaN 0] [0 1]]")
(println "la/diagonal [[1 Inf] [0 2]] ->" (array/->clj (la/diagonal (array/array [[1.0 ##Inf] [0.0 2.0]]))) " expected [1 2]")
(println "array/diag of the matrix     ->" (array/->clj (array/diag (array/array [[1.0 ##Inf] [0.0 2.0]]))))
