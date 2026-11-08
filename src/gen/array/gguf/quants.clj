;; Ported from llama.cpp: https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/ggml/src/ggml-quants.c
;; Upstream commit: cd959bae99ff4bed22bef2604f45241c09bbb8c8
;; License: MIT (Copyright (c) 2023-2026 The ggml authors); see
;;          gen.clj's THIRD-PARTY-NOTICES
;; Changes: quantize_row_{q8_0,q4_0,q4_1,q5_0,q5_1,q4_K,q5_K,q6_K}_ref,
;;          make_qkx2_quants and make_qx_quants (rmse_type 1, no weights)
;;          ported to Clojure, writing a block's bytes into a ByteBuffer;
;;          ggml-impl.h's ggml_compute_fp32_to_bf16 for BF16.
(ns gen.array.gguf.quants
  "GGML's tensor types encoded from float32 values on the host, by
   llama.cpp's reference quantizers (ggml-quants.c's quantize_row_*_ref,
   what llama-quantize writes without an importance matrix), ported: the
   body of gen.array.gguf/write-gguf's quantized tensors.

   - F32, F16 (round to nearest even) and BF16 (ggml's rounding, NaN kept
     quiet);
   - Q8_0, Q4_0, Q4_1, Q5_0 and Q5_1: blocks of 32 values, a float16 scale
     d (and Q4_1's and Q5_1's float16 minimum m);
   - Q4_K and Q5_K: super-blocks of 256, 8 sub-blocks of 32 whose scale
     and minimum are searched (make_qkx2_quants) and then quantized to 6
     bits over the super-block's float16 d and dmin;
   - Q6_K: super-blocks of 256, 16 sub-blocks of 16 whose int8 scales over
     the super-block's float16 d are searched (make_qx_quants).

   Every operation is C's float arithmetic, one rounding each (a float
   operation computed in double and rounded once to float is the float
   operation's result). The K-quants' searches are fused where the
   compiled C fuses them: Apple clang (-ffp-contract=on, -O3) contracts a
   product and a sum in one expression (`s += a*b`, `a*b + c`, `a*b - c*d`
   as fma(a, b, -(c·d))) into one rounding (Math/fma over floats here),
   except in the loops it vectorizes, where it rounds the products and
   then sums them in order: make_qkx2_quants' first error sum and its
   search's three sums, and make_qx_quants' first sums (read from the
   disassembly of v0.9.0's libggml-base, native/build-llama.sh's build
   for macOS arm64). So the bytes are those llama.cpp's quantizer writes
   there (test/gen/array/gguf_quants_test.clj, against it). A build that
   fuses otherwise (GCC's -ffp-contract=fast, another vectorizer) may
   choose other codes in a few of the K-quants' sub-blocks, each code a
   valid one.
   Written in plain Clojure: it requires nothing of gen.array."
  {:api :internal}
  (:import [java.nio ByteBuffer ByteOrder]
           [java.util Collection]
           [java.util.concurrent ExecutionException ForkJoinPool Future]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; The types
;; ---------------------------------------------------------------------------

(def types
  "The types written: each its GGML id, the values of a block and its
   bytes (a plain type: 1 value). I32 is copied, never encoded."
  {:f32 {:id 0 :values 1 :block 4}
   :i32 {:id 26 :values 1 :block 4}
   :f16 {:id 1 :values 1 :block 2}
   :bf16 {:id 30 :values 1 :block 2}
   :q8_0 {:id 8 :values 32 :block 34}
   :q4_0 {:id 2 :values 32 :block 18}
   :q4_1 {:id 3 :values 32 :block 20}
   :q5_0 {:id 6 :values 32 :block 22}
   :q5_1 {:id 7 :values 32 :block 24}
   :q4_k {:id 12 :values 256 :block 144}
   :q5_k {:id 13 :values 256 :block 176}
   :q6_k {:id 14 :values 256 :block 210}})

(defn tensor-bytes
  "The bytes n values of type t take (n a multiple of its block)."
  ^long [t ^long n]
  (let [{:keys [values block]} (types t)]
    (* (long block) (quot n (long values)))))

;; ---------------------------------------------------------------------------
;; C's float arithmetic
;; ---------------------------------------------------------------------------

(defmacro ^:private f32
  "x (a double) rounded to float32, as a double."
  [x]
  `(double (unchecked-float ~x)))

(defmacro ^:private fma32
  "a·b + c rounded once to float32 (clang's contracted a*b + c), as a
   double; a, b and c float32 values."
  [a b c]
  `(double (Math/fma (unchecked-float ~a) (unchecked-float ~b) (unchecked-float ~c))))

(defn- nearest
  "ggml's nearest_int of x (a float32 value), its trick: the bits of the
   float x + 1.5·2^23. For |x| < 2^22 the nearest integer, ties to even;
   for others, an infinity or a NaN (a scale's reciprocal past float's
   range), what the trick gives (an infinity -2^22, a NaN 0)."
  ^long [^double x]
  (let [i (Float/floatToRawIntBits (unchecked-float (+ x 12582912.0)))]
    (- (bit-and i 0x007fffff) 0x00400000)))

(defn- to-int32
  "C's float to int on arm64 (fcvtzs): toward zero, saturating at int32's
   range, a NaN 0, as clang compiles a float stored to an int8_t (its low
   byte then taken)."
  ^long [^double x]
  (cond (Double/isNaN x) 0
        (>= x 2147483647.0) 2147483647
        (<= x -2147483648.0) -2147483648
        :else (long x)))

(defn- to-uint32
  "C's float to unsigned int on arm64 (fcvtzu): toward zero, saturating at
   0 and uint32's largest, a NaN 0 (a float stored to a uint8_t)."
  ^long [^double x]
  (cond (Double/isNaN x) 0
        (<= x 0.0) 0
        (>= x 4294967295.0) 4294967295
        :else (long x)))

(defn- clamp ^long [^long lo ^long hi ^long x] (Math/max lo (Math/min hi x)))

(defmacro ^:private half
  "The float16 bits of x (a float32 value), rounded to nearest even, as
   a short."
  [x]
  `(Float/floatToFloat16 (unchecked-float ~x)))

(defn- bf16
  "The bfloat16 bits of x (a float32 value), as ggml_compute_fp32_to_bf16
   rounds it: to nearest even, a NaN kept quiet."
  ^long [^double x]
  (let [u (Integer/toUnsignedLong (Float/floatToRawIntBits (unchecked-float x)))]
    (if (> (bit-and u 0x7fffffff) 0x7f800000)
      (bit-or (unsigned-bit-shift-right u 16) 64)
      (bit-and (unsigned-bit-shift-right (+ u 0x7fff (bit-and (unsigned-bit-shift-right u 16) 1)) 16) 0xffff))))

(defn- round-away
  "C's roundf of x (a float32 value), halves away from zero (exact in
   double), then to an int (to-int32): an infinity or a NaN as arm64's
   frinta and fcvtzs give them."
  ^long [^double x]
  (if (or (Double/isNaN x) (Double/isInfinite x))
    (to-int32 x)
    (let [a (Math/floor (+ (Math/abs x) 0.5))]
      (to-int32 (if (neg? x) (- a) a)))))

(defmacro ^:private each
  "body for each i of [from, to)."
  [[i from to] & body]
  `(let [to# (long ~to)]
     (loop [~i (long ~from)]
       (when (< ~i to#)
         ~@body
         (recur (inc ~i))))))

(defn- put-byte! [^ByteBuffer b ^long at ^long v] (.put b (int at) (unchecked-byte v)))

;; ---------------------------------------------------------------------------
;; Blocks of 32: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
;; ---------------------------------------------------------------------------

(defn- abs-max!
  "The largest |x| of the n values of x at base (amax), and the first
   value that has it (max): into r [amax max]."
  [^floats x ^long base ^long n ^doubles r]
  (aset r 0 0.0)
  (aset r 1 0.0)
  (each [j 0 n]
    (let [v (double (aget x (+ base j)))
          a (Math/abs v)]
      (when (< (aget r 0) a)
        (aset r 0 a)
        (aset r 1 v)))))

(defn- min-max!
  "The least and largest of the 32 values of x at base: into r [min max]."
  [^floats x ^long base ^doubles r]
  (aset r 0 (double Float/MAX_VALUE))
  (aset r 1 (- (double Float/MAX_VALUE)))
  (each [j 0 32]
    (let [v (double (aget x (+ base j)))]
      (when (< v (aget r 0)) (aset r 0 v))
      (when (> v (aget r 1)) (aset r 1 v)))))

(defn- q8-0!
  "quantize_row_q8_0_ref: blocks [from, to) of x into b. d = max |x| / 127,
   q = roundf(x / d)."
  [^floats x ^ByteBuffer b ^long from ^long to]
  (let [r (double-array 2)]
    (each [i from to]
      (let [base (* i 32)
            at (* i 34)
            _ (abs-max! x base 32 r)
            d (f32 (/ (aget r 0) 127.0))
            id (if (zero? d) 0.0 (f32 (/ 1.0 d)))]
        (.putShort b (int at) (half d))
        (each [j 0 32]
          (put-byte! b (+ at 2 j) (round-away (f32 (* (double (aget x (+ base j))) id)))))))))

(defn- pack-32!
  "A block's 32 codes (bytes; 4 or 5 bits but in a degenerate block) as
   Q4_0's, Q4_1's, Q5_0's and Q5_1's qs at at of b (value j's low 4 bits
   at byte j, value j + 16's at its high nibble; Q4's as C ors them, the
   bytes unmasked), and returned their fifth bits as Q5's qh (value j's at
   bit j)."
  ^long [^ByteBuffer b ^long at ^longs codes masked?]
  (let [qh (long-array 1)]
    (each [j 0 16]
      (let [x0 (aget codes j) x1 (aget codes (+ j 16))]
        (put-byte! b (+ at j) (if masked?
                                (bit-or (bit-and x0 0xF) (bit-shift-left (bit-and x1 0xF) 4))
                                (bit-or x0 (bit-shift-left x1 4))))
        (aset qh 0 (bit-or (aget qh 0)
                           (bit-shift-left (bit-and (bit-shift-right x0 4) 1) j)
                           (bit-shift-left (bit-and (bit-shift-right x1 4) 1) (+ j 16))))))
    (aget qh 0)))

(defn- symmetric!
  "quantize_row_q4_0_ref and quantize_row_q5_0_ref (bits 4 or 5): blocks
   [from, to) of x into b. d = max / -2^(bits-1), max the value of the
   largest magnitude; q = min(2^bits - 1, trunc(x / d + 2^(bits-1) + 1/2)),
   the min an int8_t's, as C's (to-int32's low byte)."
  [^floats x ^ByteBuffer b bits from to]
  (let [bits (long bits) from (long from) to (long to)
        r (double-array 2)
        codes (long-array 32)
        half-range (double (bit-shift-left 1 (dec bits)))
        top (dec (bit-shift-left 1 bits))
        q4? (= 4 bits)
        size (if q4? 18 22)
        offset (+ half-range 0.5)]
    (each [i from to]
      (let [base (* i 32)
            at (* i size)
            _ (abs-max! x base 32 r)
            d (f32 (/ (aget r 1) (- half-range)))
            id (if (zero? d) 0.0 (f32 (/ 1.0 d)))]
        (each [j 0 32]
          (aset codes j (bit-and (Math/min top (long (unchecked-byte (to-int32 (f32 (+ (f32 (* (double (aget x (+ base j))) id)) offset))))))
                                 0xff)))
        (.putShort b (int at) (half d))
        (let [qh (pack-32! b (+ at (if q4? 2 6)) codes (not q4?))]
          (when-not q4?
            (.putInt b (int (+ at 2)) (unchecked-int qh))))))))

(defn- asymmetric!
  "quantize_row_q4_1_ref and quantize_row_q5_1_ref (bits 4 or 5): blocks
   [from, to) of x into b. d = (max - min) / (2^bits - 1), m = min,
   q = trunc((x - min) / d + 1/2) (at most 15, Q4_1's, an int8_t's min;
   Q5_1's a uint8_t, unclamped), as C's."
  [^floats x ^ByteBuffer b bits from to]
  (let [bits (long bits) from (long from) to (long to)
        r (double-array 2)
        codes (long-array 32)
        q4? (= 4 bits)
        size (if q4? 20 24)]
    (each [i from to]
      (let [base (* i 32)
            at (* i size)
            _ (min-max! x base r)
            mn (aget r 0)
            d (f32 (/ (f32 (- (aget r 1) mn)) (double (dec (bit-shift-left 1 bits)))))
            id (if (zero? d) 0.0 (f32 (/ 1.0 d)))]
        (each [j 0 32]
          (let [v (f32 (+ (f32 (* (f32 (- (double (aget x (+ base j))) mn)) id)) 0.5))]
            (aset codes j (if q4?
                            (bit-and (Math/min 15 (long (unchecked-byte (to-int32 v)))) 0xff)
                            (bit-and (to-uint32 v) 0xff)))))
        (.putShort b (int at) (half d))
        (.putShort b (int (+ at 2)) (half mn))
        (let [qh (pack-32! b (+ at (if q4? 4 8)) codes (not q4?))]
          (when-not q4?
            (.putInt b (int (+ at 4)) (unchecked-int qh))))))))

;; ---------------------------------------------------------------------------
;; Q4_K and Q5_K
;; ---------------------------------------------------------------------------

(defn- qkx2!
  "make_qkx2_quants (n 32, use_mad false) over the 32 values of x at base
   with weights w: the codes into ls at lbase (laux its scratch), and
   [scale, the_min] into res. A search over nstep + 1 scales,
   (rmin + rdelta·is + nmax) / (max - min), each fitting a scale and a
   minimum to its codes by weighted least squares, the least weighted
   squared error kept (and its minimum the next scale's)."
  [^floats x base ^doubles w ^longs ls lbase ^longs laux nmax rmin rdelta nstep ^doubles res]
  (let [base (long base) lbase (long lbase) nmax (long nmax) nstep (long nstep)
        rmin (double rmin) rdelta (double rdelta)
        ;; s: min max sum_w sum_x scale best_error sum_l sum_l2 sum_xl
        ;; cur_error this_scale this_min
        s (double-array 12)
        x0 (double (aget x base))]
    (aset s 0 x0)
    (aset s 1 x0)
    (aset s 2 (aget w 0))
    (aset s 3 (f32 (* (aget w 0) x0)))
    (each [i 1 32]
      (let [xi (double (aget x (+ base i))) wi (aget w i)]
        (when (< xi (aget s 0)) (aset s 0 xi))
        (when (> xi (aget s 1)) (aset s 1 xi))
        (aset s 2 (f32 (+ (aget s 2) wi)))
        (aset s 3 (fma32 wi xi (aget s 3)))))
    (when (> (aget s 0) 0.0) (aset s 0 0.0))
    (if (== (aget s 1) (aget s 0))
      (do (each [i 0 32] (aset ls (+ lbase i) 0))
          (aset res 0 0.0)
          (aset res 1 (- (aget s 0))))
      (let [mx (aget s 1)
            sum-w (aget s 2)
            sum-x (aget s 3)
            mn (aget s 0)
            iscale (f32 (/ (double nmax) (f32 (- mx mn))))
            scale (f32 (/ 1.0 iscale))]
        (aset s 4 scale)
        (aset s 5 0.0)
        ;; vectorized by clang: the products rounded, then summed in order
        (each [i 0 32]
          (let [xi (double (aget x (+ base i)))
                l (clamp 0 nmax (nearest (f32 (* iscale (f32 (- xi mn))))))
                diff (f32 (- (fma32 scale (double l) mn) xi))]
            (aset ls (+ lbase i) l)
            (aset s 5 (f32 (+ (aget s 5) (f32 (* (aget w i) (f32 (* diff diff)))))))))
        (when (>= nstep 1)
          (each [is 0 (inc nstep)]
            (let [mn (aget s 0)
                  iscale (f32 (/ (f32 (+ (fma32 rdelta (double is) rmin) (double nmax))) (f32 (- mx mn))))]
              (aset s 6 0.0) (aset s 7 0.0) (aset s 8 0.0)
              ;; vectorized by clang: the products rounded, then summed in order
              (each [i 0 32]
                (let [xi (double (aget x (+ base i)))
                      l (clamp 0 nmax (nearest (f32 (* iscale (f32 (- xi mn))))))
                      wl (f32 (* (aget w i) (double l)))]
                  (aset laux i l)
                  (aset s 6 (f32 (+ (aget s 6) wl)))
                  (aset s 7 (f32 (+ (aget s 7) (f32 (* wl (double l))))))
                  (aset s 8 (f32 (+ (aget s 8) (f32 (* wl xi)))))))
              (let [sum-l (aget s 6) sum-l2 (aget s 7) sum-xl (aget s 8)
                    dd (fma32 sum-w sum-l2 (- (f32 (* sum-l sum-l))))]
                (when (> dd 0.0)
                  (aset s 10 (f32 (/ (fma32 sum-w sum-xl (- (f32 (* sum-x sum-l)))) dd)))
                  (aset s 11 (f32 (/ (fma32 sum-l2 sum-x (- (f32 (* sum-l sum-xl)))) dd)))
                  (when (> (aget s 11) 0.0)
                    (aset s 11 0.0)
                    (aset s 10 (f32 (/ sum-xl sum-l2))))
                  (aset s 9 0.0)
                  (let [this-scale (aget s 10) this-min (aget s 11)]
                    (each [i 0 32]
                      (let [diff (f32 (- (fma32 this-scale (double (aget laux i)) this-min) (double (aget x (+ base i)))))]
                        (aset s 9 (fma32 (aget w i) (f32 (* diff diff)) (aget s 9)))))
                    (when (< (aget s 9) (aget s 5))
                      (each [i 0 32] (aset ls (+ lbase i) (aget laux i)))
                      (aset s 5 (aget s 9))
                      (aset s 4 this-scale)
                      (aset s 0 this-min))))))))
        (aset res 0 (aget s 4))
        (aset res 1 (- (aget s 0)))))))

(defn- scale-min
  "Sub-block j's 6-bit scale and min from the 12 scale bytes at at
   (get_scale_min_k4): the scale in bits 0-7, the min in 8-15."
  ^long [^ByteBuffer b ^long at ^long j]
  (let [u0 (bit-and (long (.get b (int (+ at j)))) 0xff)
        u4 (bit-and (long (.get b (int (+ at j 4)))) 0xff)]
    (if (< j 4)
      (bit-or (bit-and u0 63) (bit-shift-left (bit-and u4 63) 8))
      (let [um4 (bit-and (long (.get b (int (+ at (- j 4))))) 0xff)]
        (bit-or (bit-or (bit-and u4 0xF) (bit-shift-left (bit-shift-right um4 6) 4))
                (bit-shift-left (bit-or (bit-shift-right u4 4) (bit-shift-left (bit-shift-right u0 6) 4)) 8))))))

(defn- k-quant!
  "quantize_row_q4_K_ref and quantize_row_q5_K_ref (bits 4 or 5):
   super-blocks [from, to) of x into b."
  [^floats x ^ByteBuffer b bits from to]
  (let [bits (long bits) from (long from) to (long to)
        q4? (= 4 bits)
        size (if q4? 144 176)
        nmax (if q4? 15 31)
        rmin (if q4? -1.0 -0.5)
        rdelta (f32 0.1)
        nstep (if q4? 20 15)
        ls (long-array 256)
        laux (long-array 32)
        w (double-array 32)
        scales (double-array 8)
        mins (double-array 8)
        res (double-array 2)
        sc (byte-array 12)]
    (each [i from to]
      (let [base (* i 256)
            at (* i size)
            mx (double-array 2)]          ; max_scale max_min
        (each [j 0 8]
          (let [sb (+ base (* 32 j))
                sum-x2 (double-array 1)]
            (each [l 0 32]
              (let [v (double (aget x (+ sb l)))]
                (aset sum-x2 0 (fma32 v v (aget sum-x2 0)))))
            (let [av-x (f32 (Math/sqrt (f32 (/ (aget sum-x2 0) 32.0))))]
              (each [l 0 32] (aset w l (f32 (+ av-x (Math/abs (double (aget x (+ sb l)))))))))
            (qkx2! x sb w ls (* 32 j) laux nmax rmin rdelta nstep res)
            (aset scales j (aget res 0))
            (aset mins j (aget res 1))
            (when (> (aget res 0) (aget mx 0)) (aset mx 0 (aget res 0)))
            (when (> (aget res 1) (aget mx 1)) (aset mx 1 (aget res 1)))))
        (let [max-scale (aget mx 0)
              max-min (aget mx 1)
              inv-scale (if (> max-scale 0.0) (f32 (/ 63.0 max-scale)) 0.0)
              inv-min (if (> max-min 0.0) (f32 (/ 63.0 max-min)) 0.0)]
          (java.util.Arrays/fill sc (byte 0))
          (each [j 0 8]
            (let [l-s (Math/min 63 (bit-and (nearest (f32 (* inv-scale (aget scales j)))) 0xff))
                  l-m (Math/min 63 (bit-and (nearest (f32 (* inv-min (aget mins j)))) 0xff))]
              (if (< j 4)
                (do (aset sc j (unchecked-byte l-s))
                    (aset sc (+ j 4) (unchecked-byte l-m)))
                (do (aset sc (+ j 4) (unchecked-byte (bit-or (bit-and l-s 0xF) (bit-shift-left (bit-and l-m 0xF) 4))))
                    (aset sc (- j 4) (unchecked-byte (bit-or (aget sc (- j 4)) (bit-shift-left (bit-shift-right l-s 4) 6))))
                    (aset sc j (unchecked-byte (bit-or (aget sc j) (bit-shift-left (bit-shift-right l-m 4) 6))))))))
          (let [d (long (half (f32 (/ max-scale 63.0))))
                dmin (long (half (f32 (/ max-min 63.0))))
                dv (double (Float/float16ToFloat (unchecked-short d)))
                dminv (double (Float/float16ToFloat (unchecked-short dmin)))]
            (.putShort b (int at) (unchecked-short d))
            (.putShort b (int (+ at 2)) (unchecked-short dmin))
            (each [k 0 12] (.put b (int (+ at 4 k)) (aget sc k)))
            (each [j 0 8]
              (let [sm (scale-min b (+ at 4) j)
                    dd (f32 (* dv (double (bit-and sm 0xff))))]
                (when-not (zero? dd)
                  (let [dm (f32 (* dminv (double (bit-shift-right sm 8))))]
                    (each [ii 0 32]
                      (aset ls (+ (* 32 j) ii)
                            (clamp 0 nmax (nearest (f32 (/ (f32 (+ (double (aget x (+ base (* 32 j) ii))) dm)) dd))))))))))
            (if q4?
              (each [c 0 4]
                (each [l 0 32]
                  (put-byte! b (+ at 16 (* 32 c) l)
                             (bit-or (aget ls (+ (* 64 c) l)) (bit-shift-left (aget ls (+ (* 64 c) l 32)) 4)))))
              (do
                (each [j 0 32] (.put b (int (+ at 16 j)) (byte 0)))
                (each [c 0 4]
                  (let [m1 (bit-shift-left 1 (* 2 c))
                        m2 (bit-shift-left 2 (* 2 c))]
                    (each [j 0 32]
                      (let [l1 (aget ls (+ (* 64 c) j))
                            l2 (aget ls (+ (* 64 c) j 32))
                            qh-at (+ at 16 j)
                            qh (bit-and (long (.get b (int qh-at))) 0xff)
                            qh (cond-> qh (> l1 15) (bit-or m1) (> l2 15) (bit-or m2))]
                        (put-byte! b qh-at qh)
                        (put-byte! b (+ at 48 (* 32 c) j)
                                   (bit-or (if (> l1 15) (- l1 16) l1)
                                           (bit-shift-left (if (> l2 15) (- l2 16) l2) 4)))))))))))))))

;; ---------------------------------------------------------------------------
;; Q6_K
;; ---------------------------------------------------------------------------

(def ^:private ^:const group-max-eps
  "GROUP_MAX_EPS, 1e-15f."
  1.0000000036274937E-15)

(defn- qx-sums!
  "make_qx_quants' sums over the 16 values of x at base for iscale: into
   s [sumlx suml2], with w = x² and l the clamped code. fused?: each
   product fused into its sum, as clang compiles the search's loop; or
   rounded and then summed, as it vectorizes the first."
  [^floats x base iscale ^doubles s fused?]
  (let [base (long base) iscale (double iscale)]
  (aset s 0 0.0)
  (aset s 1 0.0)
  (each [i 0 16]
    (let [xi (double (aget x (+ base i)))
          l (double (clamp -32 31 (nearest (f32 (* iscale xi)))))
          w (f32 (* xi xi))
          wx (f32 (* w xi))
          wl (f32 (* w l))]
      (if fused?
        (do (aset s 0 (fma32 wx l (aget s 0)))
            (aset s 1 (fma32 wl l (aget s 1))))
        (do (aset s 0 (f32 (+ (aget s 0) (f32 (* wx l)))))
            (aset s 1 (f32 (+ (aget s 1) (f32 (* wl l)))))))))))

(defn- qx-codes!
  "The codes 32 + clamp(nearest(iscale·x)) of the 16 values of x at base,
   into ls at lbase."
  [^floats x base iscale ^longs ls lbase]
  (let [base (long base) iscale (double iscale) lbase (long lbase)]
    (each [i 0 16]
      (aset ls (+ lbase i) (+ 32 (clamp -32 31 (nearest (f32 (* iscale (double (aget x (+ base i))))))))))))

(defn- qx!
  "make_qx_quants (n 16, nmax 32, rmse_type 1, no weights) over the 16
   values of x at base: the codes (0 to 63) into ls at lbase, and the
   scale returned. A search over 19 scales, -(32 + is/10) / max, each
   fitting a scale to its codes by least squares weighted by x²."
  ^double [^floats x ^long base ^longs ls ^long lbase]
  (let [r (double-array 2)
        _ (abs-max! x base 16 r)
        amax (aget r 0)
        mx (aget r 1)]
    (if (< amax group-max-eps)
      (do (each [i 0 16] (aset ls (+ lbase i) 0)) 0.0)
      (let [;; s: sumlx suml2 scale best
            s (double-array 4)
            iscale (f32 (/ -32.0 mx))]
        (qx-sums! x base iscale s false)
        (qx-codes! x base iscale ls lbase)
        (aset s 2 (if (zero? (aget s 1)) 0.0 (f32 (/ (aget s 0) (aget s 1)))))
        (aset s 3 (f32 (* (aget s 2) (aget s 0))))
        (each [is -9 10]
          (when-not (zero? is)
            (let [iscale (f32 (/ (- (fma32 (f32 0.1) (double is) 32.0)) mx))
                  best (aget s 3)]
              (qx-sums! x base iscale s true)
              (when (and (> (aget s 1) 0.0) (> (f32 (* (aget s 0) (aget s 0))) (f32 (* best (aget s 1)))))
                (qx-codes! x base iscale ls lbase)
                (aset s 2 (f32 (/ (aget s 0) (aget s 1))))
                (aset s 3 (f32 (* (aget s 2) (aget s 0))))))))
        (aget s 2)))))

(defn- q6-k!
  "quantize_row_q6_K_ref: super-blocks [from, to) of x into b."
  [^floats x ^ByteBuffer b ^long from ^long to]
  (let [ls (long-array 256)
        scales (double-array 16)]
    (each [i from to]
      (let [base (* i 256)
            at (* i 210)
            mx (double-array 2)]          ; max_abs_scale max_scale
        (each [ib 0 16]
          (let [scale (qx! x (+ base (* 16 ib)) ls (* 16 ib))
                a (Math/abs scale)]
            (aset scales ib scale)
            (when (> a (aget mx 0))
              (aset mx 0 a)
              (aset mx 1 scale))))
        (if (< (aget mx 0) group-max-eps)
          (each [k 0 210] (.put b (int (+ at k)) (byte 0)))
          (let [iscale (f32 (/ -128.0 (aget mx 1)))
                d (long (half (f32 (/ 1.0 iscale))))
                dv (double (Float/float16ToFloat (unchecked-short d)))]
            (.putShort b (int (+ at 208)) (unchecked-short d))
            (each [ib 0 16]
              (put-byte! b (+ at 192 ib) (Math/min 127 (nearest (f32 (* iscale (aget scales ib)))))))
            (each [j 0 16]
              (let [dd (f32 (* dv (double (.get b (int (+ at 192 j))))))]
                (when-not (zero? dd)
                  (each [ii 0 16]
                    (aset ls (+ (* 16 j) ii)
                          (+ 32 (clamp -32 31 (nearest (f32 (/ (double (aget x (+ base (* 16 j) ii))) dd))))))))))
            (each [h 0 2]
              (let [j (* 128 h)
                    ql (+ at (* 64 h))
                    qh (+ at 128 (* 32 h))]
                (each [l 0 32]
                  (let [l0 (aget ls (+ j l)) l1 (aget ls (+ j l 32)) l2 (aget ls (+ j l 64)) l3 (aget ls (+ j l 96))]
                    (put-byte! b (+ ql l) (bit-or (bit-and l0 0xF) (bit-shift-left (bit-and l2 0xF) 4)))
                    (put-byte! b (+ ql l 32) (bit-or (bit-and l1 0xF) (bit-shift-left (bit-and l3 0xF) 4)))
                    (put-byte! b (+ qh l) (bit-or (bit-shift-right l0 4) (bit-shift-left (bit-shift-right l1 4) 2)
                                                  (bit-shift-left (bit-shift-right l2 4) 4) (bit-shift-left (bit-shift-right l3 4) 6)))))))))))))

;; ---------------------------------------------------------------------------
;; Encoding
;; ---------------------------------------------------------------------------

(defn- plain!
  "Values [from, to) of x as F32, F16 or BF16 into b."
  [t ^floats x ^ByteBuffer b from to]
  (case t
    :f32 (each [i from to] (.putFloat b (int (* 4 i)) (aget x i)))
    :f16 (each [i from to] (.putShort b (int (* 2 i)) (half (aget x i))))
    :bf16 (each [i from to] (.putShort b (int (* 2 i)) (unchecked-short (bf16 (aget x i)))))))

(defn- encode-range!
  "Blocks [from, to) of the float32 values x, as type t, into b (block i
   at i's offset)."
  [t ^floats x ^ByteBuffer b from to]
  (case t
    (:f32 :f16 :bf16) (plain! t x b from to)
    :q8_0 (q8-0! x b from to)
    :q4_0 (symmetric! x b 4 from to)
    :q5_0 (symmetric! x b 5 from to)
    :q4_1 (asymmetric! x b 4 from to)
    :q5_1 (asymmetric! x b 5 from to)
    :q4_k (k-quant! x b 4 from to)
    :q5_k (k-quant! x b 5 from to)
    :q6_k (q6-k! x b from to)))

(def ^:private chunk-blocks
  "The blocks a task of encode encodes, at least (a var the tests
   redefine, to encode small tensors in ranges)."
  4096)

(defn encode
  "The bytes of the float32 values x (a float[]) as GGML type t (a key of
   types), in a little-endian ByteBuffer of (tensor-bytes t n), by
   llama.cpp's reference quantizer: in ranges of blocks on the common
   fork-join pool, a range of at least chunk-blocks a task, each block
   written by one task. Loud (:invalid-args) for a type not encoded, or a
   count of values not a multiple of the type's block."
  ^ByteBuffer [t ^floats x]
  (let [{:keys [values]} (or (when-not (= :i32 t) (types t))
                             (throw (ex-info (str "encode: " (pr-str t) " is not a type encoded ("
                                                  (vec (sort (keys (dissoc types :i32)))) " are)")
                                             {:reason :unsupported :type t})))
        n (alength x)
        values (long values)]
    (when-not (zero? (rem n values))
      (throw (ex-info (str "encode: " n " values are not a whole number of " (name t) "'s blocks of " values)
                      {:reason :invalid-args :type t :values n})))
    (let [blocks (quot n values)
          size (tensor-bytes t n)
          _ (when (> size Integer/MAX_VALUE)
              (throw (ex-info (str "encode: a tensor of " size " bytes is too large to encode on the host")
                              {:reason :unsupported :type t :bytes size})))
          b (doto (ByteBuffer/allocate (int size)) (.order ByteOrder/LITTLE_ENDIAN))
          pool (ForkJoinPool/commonPool)
          parts (min (long (.getParallelism pool)) (quot blocks (long chunk-blocks)))]
      (if (<= parts 1)
        (encode-range! t x b 0 blocks)
        (let [step (quot (+ blocks (dec parts)) parts)
              tasks (vec (for [p (range parts)
                               :let [from (* (long p) step)]
                               :when (< from blocks)]
                           ^Callable (fn [] (encode-range! t x b from (min blocks (+ from step))))))]
          (doseq [^Future fut (.invokeAll pool ^Collection tasks)]
            (try (.get fut)
                 (catch ExecutionException e (throw (.getCause e)))))))
      b)))
