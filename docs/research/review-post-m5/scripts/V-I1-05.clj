;; Verifier's checks, one JVM:
;; - I1-05: mixed-dtype operands reach the backend from gen.array.linalg;
;;   the JVM's result dtype follows the matrix, so precision depends on which
;;   operand is float32; and what inv does today (the direction's caveat:
;;   promoting would make inv of a float32 matrix float64 on the JVM backend,
;;   since inv's identity takes the default float dtype).
;; - I1-09: the diagonals at non-finite entries, against NumPy's
;;   np.diag / np.diagonal (exact reads: np.diag([1, inf]) = [[1, 0], [0, inf]]),
;;   and cholesky-logdet through la/diagonal.
;; - I1-02: under the batch tracer, a per-particle broadcast error names the
;;   lifted (batch) shapes, since the check runs inside the inner backend.
;; - I1-10 / I1-12: the cost of a requiring-resolve per call, and of
;;   satisfies? on a miss (compile on the JVM backend), against instance?.
(require '[gen.array :as array] '[gen.array.linalg :as la] '[gen.array.batch :as batch]
         '[gen.array.protocols :as p] '[gen.backend.jvm :as jvm])

(println "I1-05")
(let [m32 (array/array [[3.0 1.0] [1.0 2.0]] {:dtype :float32})
      m64 (array/array [[3.0 1.0] [1.0 2.0]])
      r32 (array/array [0.1 0.7] {:dtype :float32})
      r64 (array/array [0.1 0.7])
      show (fn [label x] (println (format "  %-34s %-8s %s" label (array/dtype x) (pr-str (array/->clj x)))))]
  (show "solve f64 m, f64 rhs" (la/solve m64 r64))
  (show "solve f32 m, f64 rhs" (la/solve m32 r64))
  (show "solve f64 m, f32 rhs" (la/solve m64 r32))
  (show "solve-triangular f32 m, f64 rhs" (la/solve-triangular m32 r64))
  (show "solve-triangular f64 m, f32 rhs" (la/solve-triangular m64 r32))
  (show "matmul f32 m, f64 v" (array/matmul m32 r64))
  (show "+ f32, f64" (array/+ r32 r64))
  (show "inv f32 m (eye is the default float)" (la/inv m32)))

(println "I1-09")
(let [show (fn [label x expected] (println (format "  %-34s %-26s expected %s" label (pr-str (array/->clj x)) expected)))]
  (show "diag [1 Inf]" (array/diag (array/array [1.0 ##Inf])) "[[1 0] [0 Inf]]")
  (show "diag [NaN 1]" (array/diag (array/array [##NaN 1.0])) "[[NaN 0] [0 1]]")
  (show "diag [-Inf 2]" (array/diag (array/array [##-Inf 2.0])) "[[-Inf 0] [0 2]]")
  (show "la/diagonal [[1 Inf] [0 2]]" (la/diagonal (array/array [[1.0 ##Inf] [0.0 2.0]])) "[1 2]")
  (show "la/diagonal [[1 0] [NaN 2]]" (la/diagonal (array/array [[1.0 0.0] [##NaN 2.0]])) "[1 2]")
  (show "array/diag [[1 Inf] [0 2]]" (array/diag (array/array [[1.0 ##Inf] [0.0 2.0]])) "[1 2]")
  (show "cholesky-logdet [[1 0] [Inf 2]]" (la/cholesky-logdet (array/array [[1.0 0.0] [##Inf 2.0]]))
        (str (* 2 (+ (Math/log 1.0) (Math/log 2.0))))))

(println "I1-02")
(let [report (fn [label f] (try (f) (println "  " label "no error")
                                (catch Exception e (println (format "  %-26s %s %s" label (ex-message e) (pr-str (ex-data e)))))))]
  (report "unbatched [3] -> [2 2]" #(array/broadcast-to (array/zeros [3]) [2 2]))
  (report "batched [5]x[3] -> [2 2]" #(array/broadcast-to (batch/wrap (array/zeros [5 3]) [5]) [2 2]))
  (report "batched concatenate" #(array/concatenate [(batch/wrap (array/zeros [5 3 2]) [5]) (array/zeros [3 4])] 0)))

(println "I1-10 / I1-12 costs")
(defn per-call [label f]
  (dotimes [_ 200000] (f))
  (let [n 2000000 t0 (System/nanoTime)]
    (dotimes [_ n] (f))
    (println (format "  %-44s %8.1f ns" label (/ (- (System/nanoTime) t0) (double n))))))
(per-call "requiring-resolve 'gen.array/tree-leaves" #(requiring-resolve 'gen.array/tree-leaves))
(per-call "satisfies? p/ICompiling jvm/backend (a miss)" #(satisfies? p/ICompiling jvm/backend))
(per-call "instance? ICompiling jvm/backend" #(instance? gen.array.protocols.ICompiling jvm/backend))
(let [f (array/compile (fn [x] (array/+ x 1.0))) x (array/array [1.0 2.0])]
  (per-call "compiled fn on the JVM backend" #(f x))
  (per-call "the fn itself" #(array/+ x 1.0)))
(flush)
