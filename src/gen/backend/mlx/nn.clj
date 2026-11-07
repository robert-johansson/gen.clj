(ns gen.backend.mlx.nn
  "The MLX backend's neural-network ops (gen.backend.mlx): MLX's fast
   kernels for gen.array's composites rms-norm, rope and sdpa, and
   mlx-lm's Metal kernel for gated-delta (gen.backend.mlx.gated-delta)
   (p/IFused), and its quantized weights (p/IQuantized: mlx_quantize,
   mlx_dequantize, mlx_quantized_matmul, mlx_gather_qmm).

   The fast kernels take float32 and the half floats. float64 (the CPU
   device's 64-bit dtypes) is declined: MLX's fallbacks compute rms_norm
   and rope's angles in float32, where the composite is exact. So is rope
   of 2 axes (MLX's wants 3 or more), and attention under a causal mask
   with more queries than keys (MLX's fallback averages the values for
   the queries that see none, where the composite gives zeros). Under
   MLX's native gradients a kernel is used where MLX's VJP agrees with the
   composite's gradient (gen.array.nn-test checks them): rope with :freqs
   is declined there, as MLX has no VJP with respect to them, and sdpa,
   whose VJP on Metal fails for 8 queries or fewer (MLX 0.32.4).

   MLX's quantized kernels take float32 and the half floats: float64 is
   quantized in float32 (MLX's algorithm computes in float32 anyway), and
   its products are dequantized and multiplied, since MLX's CPU kernels
   fail on float64 inside a CPU task (docs/array-api.md §13's hazard).
   gather_qmm reads its indices unchecked, so they are normalized and
   clamped first, as take's."
  {:api :internal}
  (:require [gen.array.shape :as sh]
            [gen.backend.mlx.array :refer [code device fence stream wrap wrap-all]]
            [gen.backend.mlx.gated-delta :as gated-delta]
            [gen.backend.mlx.paged :as paged]
            [gen.backend.mlx.paged-attention :as paged-attention]
            [gen.backend.mlx.operands :refer [dtype-on operand scalar-handle shape-of with-operands with-temps]]
            [gen.backend.mlx.transforms :refer [*differentiating*]])
  (:import [gen.mlx Mlx MlxRuntime]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Fast kernels for gen.array's composites. An optional operand's handle is
;; operand's (the array's own), 0 (MLX's null) for none, and the array is
;; fenced after the downcall
;; ---------------------------------------------------------------------------

(def ^:private kernel-dtypes #{:float32 :float16 :bfloat16})

(defn- rms-norm [b x weight eps]
  (with-operands b :rms-norm [hx x nil]
    (try
      (wrap b (Mlx/mlx_fast_rms_norm hx (if weight (operand b weight nil) 0) (float eps) (stream b)))
      (finally (fence weight)))))

(defn- rope
  "MLX's rope of x (3 axes or more), at an offset or one per row."
  [b x dims {:keys [traditional? base scale offset freqs]}]
  (let [s (stream b)
        base (when base (Float/valueOf (float base)))]
    (with-operands b :rope [hx x nil]
      (try
        (let [hf (if freqs (operand b freqs nil) 0)]
          (wrap b (if (integer? offset)
                    (Mlx/mlx_fast_rope hx (int dims) (boolean traditional?) base (float scale) (int offset) hf s)
                    (Mlx/mlx_fast_rope_dynamic hx (int dims) (boolean traditional?) base (float scale)
                                               (operand b offset nil) hf s))))
        (finally (fence freqs) (fence offset))))))

(defn- sdpa-handle
  "A new handle: MLX's attention of the handles hq, hk and hv under mask
   (nil, :causal, or an array whose handle is hm)."
  [hq hk hv scale mask hm s]
  (Mlx/mlx_fast_scaled_dot_product_attention
   (long hq) (long hk) (long hv) (float scale)
   (cond (identical? :causal mask) "causal" mask "array" :else "")
   (long hm) 0 false (long s)))

(defn- sdpa
  "MLX's attention. A query that sees no key under a bool mask gets
   zeros, as gen.array's composite gives: MLX's fallback (the CPU) averages
   every value there, and its Metal kernels give zeros or another row
   (measured 2026-11-07), so its row is zeroed here."
  [b q k v {:keys [scale mask]}]
  (let [arr (when-not (keyword? mask) mask)
        s (stream b)]
    (with-operands b :sdpa [hq q nil hk k nil hv v nil]
      (try
        (let [hm (if arr (operand b arr nil) 0)]
          (with-temps [t]
            (wrap b (if (and arr (= :bool (dtype-on b arr)))
                      (Mlx/mlx_where (t (Mlx/mlx_any_axis hm (int -1) true s))
                                     (t (sdpa-handle hq hk hv scale mask hm s))
                                     (t (scalar-handle 0.0 (dtype-on b q)))
                                     s)
                      (sdpa-handle hq hk hv scale mask hm s)))))
        (finally (fence arr))))))

(def ^:private mac?
  "Does this JVM run on macOS (MLX's GPU Metal there, CUDA elsewhere)?"
  (delay (.contains (.toLowerCase (System/getProperty "os.name")) "mac")))

(defn fused
  "p/-fused: op on args by MLX's fast kernel, or nil to decline."
  [b op args]
  (when (kernel-dtypes (dtype-on b (first args)))
    (case op
      :rms-norm (let [[x w eps] args] (rms-norm b x w eps))
      :rope (let [[x dims {:keys [offset freqs] :as opts}] args]
              (when-not (or (and *differentiating* freqs)
                            ;; MLX's rope wants 3 axes or more
                            (< (count (shape-of x)) 3)
                            ;; mlx_fast_rope takes an int offset
                            (and (integer? offset) (not (<= Integer/MIN_VALUE (long offset) Integer/MAX_VALUE))))
                (rope b x dims opts)))
      :gated-delta (let [[q k v g beta state] args]
                     ;; mlx-lm's Metal kernel: no VJP, so not under MLX's
                     ;; gradients; Metal only (the GPU on macOS)
                     (when (and (not *differentiating*) (= :gpu (device b)) @mac? (gated-delta/applies? b q))
                       (gated-delta/gated-delta b q k v g beta state)))
      :paged-write (let [[k-pool v-pool k v slots] args] (paged/paged-write b k-pool v-pool k v slots))
      :paged-attention (let [[q k-pool v-pool tables lens opts] args]
                         ;; kernels-community's paged_attention kernel:
                         ;; Metal only, no VJP (applies? declines both)
                         (when (paged-attention/applies? b q k-pool tables)
                           (paged-attention/paged-attention b q k-pool v-pool tables lens opts)))
      :sdpa (let [[q k v {:keys [mask] :as opts}] args]
              ;; a causal mask hides every key from the first Tq - Tk
              ;; queries, which MLX's fallback averages over; and under
              ;; MLX's gradients, its Metal VJP fails for 8 queries or fewer
              ;; ("the vector kernels do not produce a logsumexp", MLX
              ;; 0.32.4, at evaluation): the composite, both
              (when-not (or *differentiating*
                            (and (identical? :causal mask) (> (long (nth (shape-of q) 2)) (long (nth (shape-of k) 2)))))
                (sdpa b q k v opts)))
      nil)))

;; ---------------------------------------------------------------------------
;; Quantized weights
;; ---------------------------------------------------------------------------

(defn- mode-name [mode] (name mode))

(defn- quantize [b w {:keys [mode group-size bits]}]
  (let [d (dtype-on b w)
        s (stream b)]
    (with-temps [t]
      (with-operands b :quantize [hw w nil]
        ;; MLX's CPU kernel has no float64: its algorithm computes in
        ;; float32, so the float32 matrix gives the same values
        (let [hw (if (= d :float64) (t (Mlx/mlx_astype hw (int (code :float32)) s)) hw)
              v (t :vector (Mlx/mlx_quantize hw (int group-size) (int bits) (mode-name mode) 0 s))
              [wq scales biases] (wrap-all b (MlxRuntime/vectorArrays v))]
          (if (= d :float64)
            [wq
             (with-operands b :quantize [h scales nil] (wrap b (Mlx/mlx_astype h (int (code :float64)) s)))
             (with-operands b :quantize [h biases nil] (wrap b (Mlx/mlx_astype h (int (code :float64)) s)))]
            [wq scales biases]))))))

(defn- dequantize-handle
  "A new handle: the matrices wq, scales and biases (handles) stand for, in dtype."
  [hw hs hb {:keys [mode group-size bits]} dtype s]
  (Mlx/mlx_dequantize (long hw) (long hs) (long hb) (int group-size) (int bits) (mode-name mode) 0 (int (code dtype)) (long s)))

(defn- dequantize [b wq scales biases {:keys [dtype] :as opts}]
  (with-operands b :dequantize [hw wq nil hs scales nil hb biases nil]
    (wrap b (dequantize-handle hw hs hb opts dtype (stream b)))))

(defn- swapped-handle
  "A new handle: h with its last two axes swapped."
  ^long [^long h s]
  (Mlx/mlx_swapaxes h (int -1) (int -2) (long s)))

(defn- quantized-matmul [b x wq scales biases {:keys [mode group-size bits transpose?] :as opts}]
  (let [d (dtype-on b x)
        s (stream b)
        sx (shape-of x)
        qmm (fn ^long [^long hx ^long hw ^long hs ^long hb]
              (Mlx/mlx_quantized_matmul hx hw hs hb (boolean transpose?) (int group-size) (int bits) (mode-name mode) s))]
    (with-operands b :quantized-matmul [hx x nil hw wq nil hs scales nil hb biases nil]
      (cond
        (= d :float64)
        (with-temps [t]
          (let [w (t (dequantize-handle hw hs hb opts d s))]
            (wrap b (Mlx/mlx_matmul hx (if transpose? (t (swapped-handle w s)) w) s))))

        ;; A vector x goes through as a [1 K] matrix: MLX 0.32.4's CUDA qmv
        ;; reads out.shape(-2) with no rank check (qmv.cu:433), so a 1-d x
        ;; fails there ("SmallVector out of range"; found on fenrir,
        ;; 2026-11-07). The same values on every device.
        (= 1 (count sx))
        (with-temps [t]
          (let [out (t (qmm (t (Mlx/mlx_reshape hx (int-array [1 (first sx)]) s)) hw hs hb))]
            (wrap b (Mlx/mlx_reshape out (int-array [-1]) s))))

        ;; Leading axes go through as one matrix of rows. MLX 0.32.4's CUDA
        ;; kernels take M from x's second-last axis and loop over the rest,
        ;; so a decode step's [B 1 K] becomes B products of one row, each
        ;; reading the weights again: 12-13x slower at 64 rows than the
        ;; same product as [B K], with equal values (measured on fenrir's
        ;; RTX PRO 6000, 2026-11-07). The weights are shared, so the rows
        ;; are independent and the reshape exact.
        (> (count sx) 2)
        (with-temps [t]
          (let [lead (pop sx)
                rows (long (reduce clojure.core/* 1 lead))
                out (t (qmm (t (Mlx/mlx_reshape hx (int-array [rows (peek sx)]) s)) hw hs hb))]
            (wrap b (Mlx/mlx_reshape out (int-array (conj lead -1)) s))))

        :else
        (wrap b (qmm hx hw hs hb))))))

(defn- clamped-indices
  "A new handle: the int indices h into n matrices, negative ones from the
   end and the rest clamped (MLX's gathers read them unchecked)."
  ^long [^long h ^long n s]
  (let [s (long s)]
    (with-temps [t]
      (let [i32 (t (Mlx/mlx_astype h (int (code :int32)) s))
            zero (t (scalar-handle 0 :int32))
            neg (t (Mlx/mlx_less i32 zero s))
            wrapped (t (Mlx/mlx_add i32 (t (scalar-handle n :int32)) s))]
        (Mlx/mlx_clip (t (Mlx/mlx_where neg wrapped i32 s)) zero (t (scalar-handle (dec n) :int32)) s)))))

(defn- matrices
  "The number of matrices of an array of shape sh: its leading axes' size."
  ^long [shape]
  (sh/size (subvec shape 0 (- (count shape) 2))))

(defn- gather-qmm [b x wq scales biases lhs rhs {:keys [mode group-size bits transpose? sorted?] :as opts}]
  (let [d (dtype-on b x)
        s (stream b)
        sx (shape-of x)]
    (with-operands b :gather-qmm [hx x nil hw wq nil hs scales nil hb biases nil hl lhs nil hr rhs nil]
      (with-temps [t]
        (let [hl (t (clamped-indices hl (matrices sx) s))
              hr (t (clamped-indices hr (matrices (shape-of wq)) s))]
          (if (= d :float64)
            ;; dequantized, gathered and multiplied, as the JVM backend does
            (let [w (t (dequantize-handle hw hs hb opts d s))
                  sq (shape-of wq)
                  [r c] [(nth sq (- (count sq) 2)) (quot (* (long (peek sq)) 32) (long bits))]
                  is (shape-of lhs)
                  n (sh/size is)
                  flat (fn [h shape] (t (Mlx/mlx_reshape (long h) (int-array shape) s)))
                  xs (flat hx [(matrices sx) (nth sx (- (count sx) 2)) (peek sx)])
                  ws (flat w [(matrices sq) r c])
                  pick (fn [h i] (t (Mlx/mlx_take_axis (long h) (long (flat i [n])) (int 0) s)))
                  wsel (pick ws hr)
                  out (t (Mlx/mlx_matmul (pick xs hl) (if transpose? (t (swapped-handle wsel s)) wsel) s))]
              (wrap b (Mlx/mlx_reshape out (int-array (into is [(nth sx (- (count sx) 2)) (if transpose? r c)])) s)))
            (wrap b (Mlx/mlx_gather_qmm hx hw hs hb hl hr (boolean transpose?) (int group-size) (int bits)
                                        (mode-name mode) 0 (boolean sorted?) s))))))))

(defn quantized
  "p/-quantized on the MLX backend b."
  [b op args]
  (case op
    :quantize (let [[w opts] args] (quantize b w opts))
    :dequantize (let [[wq scales biases opts] args] (dequantize b wq scales biases opts))
    :quantized-matmul (let [[x wq scales biases opts] args] (quantized-matmul b x wq scales biases opts))
    :gather-qmm (let [[x wq scales biases lhs rhs opts] args] (gather-qmm b x wq scales biases lhs rhs opts))))
