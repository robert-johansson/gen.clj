(ns gen.backend.mlx.linalg
  "The MLX backend's linear algebra (gen.backend.mlx): matrix products, and
   decompositions and solves on MLX's CPU stream (MLX has them only there),
   NaN where the contract has it rather than MLX's failure. MLX has no VJP
   for its Cholesky, LU or inverse (GenMLX's fork patched some), so in
   native mode each one is one mlx_custom_vjp whose VJP is the tape's rule
   (gen.array.tape/linalg-vjp): no MLX patch."
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.backend.mlx.array :refer [code cpu-stream defaults stream wrap]]
            [gen.backend.mlx.operands :refer [as-array dtype-on scalar-handle shape-of with-operands with-temps]]
            [gen.backend.mlx.transforms :refer [custom-vjp custom-vjps?]])
  (:import [gen.mlx Mlx]))

(set! *warn-on-reflection* true)

(def ^:private linalg-vjp
  ;; gen.array.tape's rules, resolved on first use, once: the tape is
  ;; written in gen.array, which loads the backend
  (delay (requiring-resolve 'gen.array.tape/linalg-vjp)))

(defn matmul [b x y]
  (let [d (dtype-on b x)
        s (stream b)]
    (with-operands b :matmul [hx x d hy y d]
      (if (= :float (dt/kind d))
        (wrap b (Mlx/mlx_matmul hx hy s))
        ;; MLX multiplies floats only: ints go through the widest float the
        ;; device has (exact while products stay below 2^24 or 2^53)
        (with-temps [t]
          (let [f (code (:float (defaults b)))]
            (wrap b (Mlx/mlx_astype (t (Mlx/mlx_matmul (t (Mlx/mlx_astype hx f s)) (t (Mlx/mlx_astype hy f s)) s))
                                    (code d) s))))))))

(defn- batch-broadcast
  "Handles of a [.. n n] and rhs [.. n k] broadcast to one batch shape (MLX's
   solves want equal batch axes; gen.array's broadcast them), new ones
   registered with t (with-temps)."
  [b ha sa hr sr t]
  (let [ha (long ha) hr (long hr)
        ba (subvec sa 0 (- (count sa) 2))
        br (subvec sr 0 (- (count sr) 2))
        batch (sh/broadcast ba br)
        s (cpu-stream b)
        to (fn [^long h shape tail]
             (if (= shape (into batch tail))
               h
               (long (t (Mlx/mlx_broadcast_to h (int-array (map int (into batch tail))) s)))))]
    [(to ha sa (subvec sa (- (count sa) 2))) (to hr sr (subvec sr (- (count sr) 2)))]))

(defn- diag-of-temp
  "The diagonals of the matrices h, [.. n], a handle registered with t."
  ^long [t ^long h s]
  (long (t (Mlx/mlx_diagonal h (int 0) (int -2) (int -1) (long s)))))

(defn- linalg* [b op args]
  (let [cpu (cpu-stream b)]
    (with-temps [t]
      (let [nan (fn ^long [dtype] (t (scalar-handle ##NaN dtype)))]
        (case op
          ;; MLX returns a partial factor for a matrix that is not positive
          ;; definite; LAPACK leaves the failing pivot non-positive there, so the
          ;; contract's NaN is where a diagonal entry is not positive
          :cholesky
          (let [[a] args
                d (dtype-on b a)]
            (with-operands b :cholesky [h a nil]
              (let [l (t (Mlx/mlx_linalg_cholesky h false cpu))
                    pd (t (Mlx/mlx_all_axis (t (Mlx/mlx_greater (diag-of-temp t l cpu) (t (scalar-handle 0 d)) cpu))
                                            (int -1) true cpu))]
                (wrap b (Mlx/mlx_where (t (Mlx/mlx_expand_dims pd (int -1) cpu)) l (nan d) cpu)))))

          ;; MLX's triangular solve inverts the matrix, which fails on a zero
          ;; diagonal entry (reported at a later evaluation, as for :solve): a NaN
          ;; there instead gives NaN solutions, as a degenerate solve gives
          ;; non-finite ones on the JVM backend (not an error)
          :solve-triangular
          (let [[a rhs {:keys [lower? transpose?]}] args
                d (dtype-on b a)
                sa (shape-of a)]
            (with-operands b :solve-triangular [ha a nil hr rhs nil]
              (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) t)
                    n (int (peek sa))
                    eye (t (Mlx/mlx_eye n n (int 0) (int (code :bool)) cpu))
                    zero-diag (t (Mlx/mlx_expand_dims (t (Mlx/mlx_equal (diag-of-temp t ha cpu) (t (scalar-handle 0 d)) cpu))
                                                      (int -2) cpu))
                    safe (t (Mlx/mlx_where (t (Mlx/mlx_logical_and eye zero-diag cpu)) (nan d) ha cpu))]
                ;; Lᵀx = b is solving with the upper triangle of Lᵀ
                (wrap b (if transpose?
                          (Mlx/mlx_linalg_solve_triangular (t (Mlx/mlx_swapaxes safe (int -1) (int -2) cpu)) hr
                                                           (boolean lower?) cpu)
                          (Mlx/mlx_linalg_solve_triangular safe hr (not lower?) cpu))))))

          ;; A singular system fails inside MLX's solve, and MLX reports a CPU
          ;; task's failure at a later evaluation, of other arrays. So singular
          ;; matrices (a zero pivot of the LU factor) are never given to it: the
          ;; identity stands in, and their solutions are NaN
          :solve
          (let [[a rhs] args
                d (dtype-on b a)
                sa (shape-of a)]
            (with-operands b :solve [ha a nil hr rhs nil]
              (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) t)
                    n (int (peek sa))
                    [lu _] (mapv t (Mlx/mlx_linalg_lu_factor ha cpu))
                    singular (t (Mlx/mlx_expand_dims
                                 (t (Mlx/mlx_any_axis (t (Mlx/mlx_equal (diag-of-temp t lu cpu) (t (scalar-handle 0 d)) cpu))
                                                      (int -1) true cpu))
                                 (int -1) cpu))
                    eye (t (Mlx/mlx_eye n n (int 0) (int (code d)) cpu))
                    safe (t (Mlx/mlx_where singular eye ha cpu))
                    x (t (Mlx/mlx_linalg_solve safe hr cpu))]
                (wrap b (Mlx/mlx_where singular (nan d) x cpu)))))

          ;; from the LU factor (MLX's slogdet throws on a singular matrix):
          ;; sign · Π sign(u_ii) · (-1)^swaps and Σ log |u_ii|, so a singular
          ;; matrix gives [0 -Inf], as on the JVM backend
          :slogdet
          (let [[a] args
                d (dtype-on b a)
                sa (shape-of a)]
            (with-operands b :slogdet [h a nil]
              (let [[lu piv] (mapv t (Mlx/mlx_linalg_lu_factor h cpu))
                    u (diag-of-temp t lu cpu)
                    n (peek sa)
                    ;; the results wrapped as they are made: logabs's MlxArray
                    ;; owns its handle (its phantom reference frees it) while
                    ;; the handles after it are made
                    logabs (wrap b (Mlx/mlx_sum_axis (t (Mlx/mlx_log (t (Mlx/mlx_abs u cpu)) cpu)) (int -1) false cpu))
                    idx (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) cpu))
                    swaps (t (Mlx/mlx_sum_axis (t (Mlx/mlx_astype (t (Mlx/mlx_not_equal piv idx cpu)) (int (code :int32)) cpu))
                                               (int -1) false cpu))
                    odd (t (Mlx/mlx_remainder swaps (t (scalar-handle 2 :int32)) cpu))
                    perm (t (Mlx/mlx_subtract (t (scalar-handle 1 d))
                                              (t (Mlx/mlx_multiply (t (scalar-handle 2 d)) (t (Mlx/mlx_astype odd (int (code d)) cpu)) cpu))
                                              cpu))
                    sign (wrap b (Mlx/mlx_multiply (t (Mlx/mlx_prod_axis (t (Mlx/mlx_sign u cpu)) (int -1) false cpu)) perm cpu))]
                [sign logabs]))))))))

(defn linalg
  "-linalg's op on args: in native mode one custom VJP of the op, with the
   tape's rule."
  [b op args]
  (if-not (custom-vjps?)
    (linalg* b op args)
    (let [[arrays opts] (p/linalg-operands op args)
          rule @linalg-vjp
          outs (custom-vjp b
                           (fn [xs] (let [r (linalg* b op (into xs opts))] (if (vector? r) r [r])))
                           (fn [ps cs os]
                             (let [[out g] (if (= op :slogdet) [(second os) (second cs)] [(first os) (first cs)])]
                               (rule b op (into ps opts) out g (mapv (constantly true) arrays))))
                           (mapv #(as-array b % nil) arrays))]
      (if (= op :slogdet) outs (first outs)))))
