;; gen.array.linalg passes operands of different dtypes to the backend, which
;; the contract (gen.array.protocols: "operands share one dtype") rules out;
;; the JVM backend then takes the matrix's dtype, so the result depends on
;; which operand is float32. And la/diagonal reads a diagonal through a
;; product with the identity, so an infinite entry off the diagonal is NaN on it.
(require '[gen.array :as array] '[gen.array.linalg :as la])
(let [m32 (array/array [[2.0 0.0] [0.0 4.0]] {:dtype :float32})
      m64 (array/array [[2.0 0.0] [0.0 4.0]])
      r32 (array/array [1.0 1.0] {:dtype :float32})
      r64 (array/array [1.0 1.0])]
  (println "matmul f32·f64          ->" (array/dtype (array/matmul m32 r64)) "(gen.array promotes)")
  (println "matmul f64·f32          ->" (array/dtype (array/matmul m64 r32)))
  (println "solve f32 matrix, f64 rhs ->" (array/dtype (la/solve m32 r64)))
  (println "solve f64 matrix, f32 rhs ->" (array/dtype (la/solve m64 r32)))
  (println "solve-triangular f32, f64 ->" (array/dtype (la/solve-triangular m32 r64)))
  (println "solve-triangular f64, f32 ->" (array/dtype (la/solve-triangular m64 r32))))
(let [a (array/array [[1.0 ##Inf] [0.0 2.0]])]
  (println "la/diagonal of [[1 Inf] [0 2]] ->" (array/->clj (la/diagonal a))
           " array/diag ->" (array/->clj (array/diag a))))
