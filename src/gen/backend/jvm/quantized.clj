(ns gen.backend.jvm.quantized
  "The JVM backend's kernels for quantized weights (gen.backend.jvm), in
   MLX's affine layout: a row of K values in groups of g, each value
   scale·q + bias for its group's scale and bias, q an unsigned integer
   of b bits; a row's q packed as one little-endian stream of bits,
   value j at bits [j·b, (j+1)·b), in K·b/32 uint32 words (MLX packs 2, 4
   and 8 bits per word, and 3, 5 and 6 bits through bytes: the same
   stream). The kernels work on the backend's storage, double[]s,
   uint32 words exactly.

   quantize is MLX's algorithm (its CPU kernel, mlx/backend/cpu/
   quantized.cpp), computed in float32 as MLX computes it, so it gives
   MLX's words, scales and biases for a float32 matrix."
  {:api :internal})

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Dequantizing
;; ---------------------------------------------------------------------------

(defn- code-at
  "Value j of bits bits of the row whose words start at base in w."
  ^long [^doubles w ^long base ^long j ^long bits]
  (let [bit (* j bits)
        word (+ base (bit-shift-right bit 5))
        shift (bit-and bit 31)
        lo (unsigned-bit-shift-right (long (aget w word)) shift)
        v (if (> (+ shift bits) 32)
            (bit-or lo (bit-shift-left (long (aget w (inc word))) (- 32 shift)))
            lo)]
    (bit-and v (dec (bit-shift-left 1 bits)))))

(defn- dequantize-row
  "Row r of k values into out, from its words (kw a row), scales and
   biases (the row's groups at r·k/g)."
  [^doubles out ^doubles w ^doubles scales ^doubles biases layout]
  (let [r (long (:r layout)) k (long (:k layout)) kw (long (:kw layout))
        g (long (:g layout)) bits (long (:bits layout)) round32? (boolean (:round32? layout))
        base (* r kw)
        o (* r k)
        s0 (quot o g)]
    (dotimes [j k]
      (let [gi (+ s0 (quot j g))
            q (double (code-at w base j bits))
            s (aget scales gi)
            b (aget biases gi)]
        ;; q·s is exact in a double (q has at most 8 bits): rounded once
        ;; here, or the product and the sum each to float32 as MLX's CPU
        ;; computes them, two ops
        (aset out (+ o j) (if round32?
                            (double (unchecked-float (+ (double (unchecked-float (* q s))) b)))
                            (+ (* q s) b)))))))

(defn dequantize
  "The rows·k values of the packed words w (rows of kw words), scales and
   biases (rows of k/g groups), as doubles: scale·q + bias. round32?
   rounds the product and the sum to float32 apart (MLX's two ops, for a
   float32 result); otherwise the double is the result, rounded once by
   the caller to its dtype."
  ^doubles [^doubles w ^doubles scales ^doubles biases layout]
  (let [rows (long (:rows layout))
        out (double-array (* rows (long (:k layout))))]
    (dotimes [r rows]
      (dequantize-row out w scales biases (assoc layout :r r)))
    out))

;; ---------------------------------------------------------------------------
;; Quantizing: MLX's CPU kernel, in float32 (every op rounded to float, as
;; C's float arithmetic: a double op on two floats, rounded, is the float
;; op exactly)
;; ---------------------------------------------------------------------------

(defn- f32 ^double [^double x] (double (unchecked-float x)))

(defn- group-scale!
  "Group i's scale and bias into scales and biases, from its g values at
   i·g of w, as MLX computes them: its extrema in float32; the scale
   spanning them over n-bins steps (at least 1e-7), signed so that the
   extremum of the larger magnitude, the edge, is a whole number q0 of
   steps; then, unless q0 is 0, the scale edge/q0 and the bias the edge."
  [^doubles w ^doubles scales ^doubles biases layout]
  (let [i (long (:i layout)) g (long (:g layout)) n-bins (double (:n-bins layout))
        base (* i g)
        eps (f32 1e-7)]
    (loop [j 0 lo Double/POSITIVE_INFINITY hi Double/NEGATIVE_INFINITY]
      (if (< j g)
        (let [v (f32 (aget w (+ base j)))]
          (recur (inc j) (Math/min lo v) (Math/max hi v)))
        (let [mask (> (Math/abs lo) (Math/abs hi))
              scale (Math/max (f32 (/ (f32 (- hi lo)) n-bins)) eps)
              scale (if mask scale (- scale))
              edge (if mask lo hi)
              q0 (Math/rint (f32 (/ edge scale)))]
          (if (== q0 0.0)
            (do (aset scales i scale) (aset biases i 0.0))
            (do (aset scales i (f32 (/ edge q0))) (aset biases i edge))))))))

(defn- pack-row!
  "Row r's k values of w quantized by their groups' scales and biases,
   packed into out's words at r·kw."
  [^longs out ^doubles w ^doubles scales ^doubles biases layout]
  (let [r (long (:r layout)) k (long (:k layout)) kw (long (:kw layout))
        g (long (:g layout)) bits (long (:bits layout)) n-bins (double (:n-bins layout))
        o (* r k)
        base (* r kw)]
    (dotimes [j k]
      (let [gi (quot (+ o j) g)
            v (Math/rint (f32 (/ (f32 (- (f32 (aget w (+ o j))) (aget biases gi))) (aget scales gi))))
            q (long (Math/min (Math/max v 0.0) n-bins))
            bit (* j bits)
            word (+ base (bit-shift-right bit 5))
            shift (bit-and bit 31)]
        (aset out word (bit-or (aget out word) (bit-and (bit-shift-left q shift) 0xFFFFFFFF)))
        (when (> (+ shift bits) 32)
          (aset out (inc word) (bit-or (aget out (inc word)) (unsigned-bit-shift-right q (- 32 shift)))))))))

(defn quantize
  "[words scales biases] (double[]s) of w's rows of k values (groups of g,
   bits bits): a row's words kw = k·bits/32 of them, the scales and
   biases one per group, float32 values."
  [^doubles w layout]
  (let [k (long (:k layout)) g (long (:g layout)) bits (long (:bits layout))
        n (alength w)
        rows (quot n k)
        groups (quot n g)
        kw (quot (* k bits) 32)
        scales (double-array groups)
        biases (double-array groups)
        words (long-array (* rows kw))
        layout (assoc layout :kw kw :n-bins (double (dec (bit-shift-left 1 bits))))]
    (dotimes [i groups]
      (group-scale! w scales biases (assoc layout :i i)))
    (dotimes [r rows]
      (pack-row! words w scales biases (assoc layout :r r)))
    [(let [d (double-array (alength words))]
       (dotimes [i (alength words)] (aset d i (double (aget words i))))
       d)
     scales
     biases]))
