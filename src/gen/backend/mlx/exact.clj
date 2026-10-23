(ns gen.backend.mlx.exact
  "Exact routes around MLX kernels that are not, written in gen.array's ops
   (on MLX arrays). Each is within a few ulps of the JVM backend's, as the
   backend conformance suite measures (docs/array-api.md §13).

   - float64: MLX's CPU kernels for exp, sin, cos, erf and erfinv are float32
     approximations whatever the dtype (about 1e-6 relative error for exp,
     1e-4 for sin and cos). These go through its exact ones (expm1, tan, log,
     division).
   - expm1 on the GPU: Metal's is off by up to ~660 float32 ulps (4e-5
     relative); Kahan's formula from exp and log is within a few.
   - float32 sums on the CPU: MLX's accumulate sequentially, so their error
     grows with the count (10^6 terms of 10^-6 sum to 1 - 1.2e-3); summed
     in blocks, and the block sums in blocks, it grows with its log
     (genclj-y0mp). Half-float sums there accumulate in their own
     precision (10^5 terms of 10^-5: 1 - 0.75 in float16, 1 - 0.998 in
     bfloat16), so they are summed in float32 and cast back, as JAX's."
  (:refer-clojure :exclude [+ - * / < > >= ==])
  (:require [gen.array :as array :refer [+ - * / < > >= ==]]
            [gen.array.special :as special]))

(set! *warn-on-reflection* true)

(defn- exp
  "e^x from expm1, exact to an ulp: 1/(expm1(-x) + 1) below 0, where
   expm1(x) + 1 would cancel."
  [x]
  (array/where (< x 0.0)
               (/ 1.0 (+ (array/expm1 (array/negate x)) 1.0))
               (+ (array/expm1 x) 1.0)))

(defn- sin
  "sin x = 2t/(1 + t²), t = tan(x/2)."
  [x]
  (let [t (array/tan (* x 0.5))]
    (/ (* 2.0 t) (+ 1.0 (* t t)))))

(defn- cos
  "cos x = (1 - t)(1 + t)/(1 + t²), t = tan(x/2), exact near the zeros of
   cos (where 1 - t² would cancel)."
  [x]
  (let [t (array/tan (* x 0.5))]
    (/ (* (- 1.0 t) (+ 1.0 t)) (+ 1.0 (* t t)))))

(defn- erfc-nonneg
  "erfc(z) for z >= 0, as gen.array.special (Clenshaw's recurrence)."
  [z]
  (let [t (/ 2.0 (+ 2.0 z))
        ty (- (* 4.0 t) 2.0)
        [d dd] (reduce (fn [[d dd] c] [(+ (- (* ty d) dd) c) d])
                       [0.0 0.0]
                       (rseq (subvec special/erfc-coefficients 1)))]
    (* t (exp (- (+ (array/negate (* z z)) (* 0.5 (+ (first special/erfc-coefficients) (* ty d)))) dd)))))

(defn- erf-series
  "erf(x) by its Maclaurin series (for |x| < 0.5)."
  [x]
  (let [x2 (array/negate (* x x))]
    (loop [n 1 term x sum x]
      (if (< n 16)
        (let [term (/ (* term x2) n)]
          (recur (inc n) term (+ sum (/ term (inc (* 2 n))))))
        (* special/two-over-sqrt-pi sum)))))

(defn- erf [x]
  (let [ax (array/abs x)
        tail (- 1.0 (erfc-nonneg ax))]
    (array/where (< ax 0.5) (erf-series x) (array/where (< x 0.0) (array/negate tail) tail))))

(defn- erfinv
  "As gen.array.special/erfinv: Giles' approximation refined by Halley's
   method, with the residual from erfc in the tails. Branch-free, so a
   fixed seven steps (the JVM backend's most)."
  [y]
  (let [ay (array/abs y)
        ;; a stand-in away from ±1 and 0 keeps the discarded lanes finite
        special? (array/logical-or (array/logical-or (array/isnan y) (>= ay 1.0)) (== ay 0.0))
        q (array/where special? 0.5 ay)
        w (array/negate (array/log (* (- 1.0 q) (+ 1.0 q))))
        horner (fn [v cs] (reduce (fn [acc c] (+ (* acc v) c)) (first cs) (rest cs)))
        central (horner (- w 2.5) [2.81022636e-08 3.43273939e-07 -3.5233877e-06 -4.39150654e-06
                                   0.00021858087 -0.00125372503 -0.00417768164 0.246640727 1.50140941])
        tail (horner (- (array/sqrt w) 3.0) [-0.000200214257 0.000100950558 0.00134934322 -0.00367342844
                                             0.00573950773 -0.0076224613 0.00943887047 1.00167406 2.83297682])
        x0 (* q (array/where (< w 5.0) central tail))
        halley (fn [x]
                 (let [scale (* special/sqrt-pi-over-two (exp (* x x)))
                       d (array/where (> q 0.5)
                                      (* (- (- 1.0 q) (erfc-nonneg x)) scale)
                                      (* (- (erf x) q) scale))]
                   (- x (/ d (+ 1.0 (* x d))))))
        ;; as many steps as the JVM backend's adaptive loop at most takes,
        ;; which the far tails (|y| near 1 - 1e-16) need
        x (nth (iterate halley x0) 7)
        signed (array/where (< y 0.0) (array/negate x) x)]
    (array/where (array/logical-or (array/isnan y) (> ay 1.0))
                 ##NaN
                 (array/where (== ay 1.0)
                              (array/where (< y 0.0) ##-Inf ##Inf)
                              (array/where (== ay 0.0) y signed)))))

(defn expm1
  "e^x - 1 by Kahan's formula: (u - 1)·x/log u for u = e^x, exact to a few
   ulps where exp and log are, also near 0 (where u - 1 alone would cancel)."
  [x]
  (let [u (array/exp x)
        um1 (- u 1.0)
        ;; a stand-in keeps the discarded lanes finite
        safe (array/where (array/logical-or (== um1 0.0) (array/isinf u)) 2.0 u)]
    (array/where (== um1 0.0)
                 x
                 (array/where (array/logical-or (array/isinf u) (== um1 -1.0))
                              um1
                              (/ (* (- safe 1.0) x) (array/log safe))))))

(defn logaddexp
  "gen.array's composite logaddexp of float64 MLX arrays, with the exact exp."
  [x y]
  (let [m (array/maximum x y)]
    (array/where (array/isinf m) m (+ m (array/log1p (exp (array/negate (array/abs (- x y)))))))))

(def blocked-above
  "A float32 sum on MLX's CPU of more terms than this is summed in blocks
   (sum); a shorter one is MLX's own (genclj-y0mp)."
  256)

(def block-size
  "The most terms one block of a blocked sum accumulates (sum): 32 (owner,
   genclj-v3uw). Each sum is within a few ulps with blocks of 256 too, but
   an iterated computation compounds them: memo.clj's empowerment demo
   missed float64 by 1.89e-5 with 256 and 5.1e-6 with 32 (JAX's CPU
   3.6e-6), for 1.3-1.5x a long sum's time."
  32)

(defn- block-of
  "The block size for m terms: a divisor of m between half of block-size
   and block-size when there is one, so the blocks are a reshape alone
   (slicing off a remainder makes MLX copy the rest, which doubled a long
   sum's time), else block-size."
  ^long [^long m]
  (or (first (filter #(zero? (rem m (long %))) (range block-size (dec (quot block-size 2)) -1)))
      block-size))

(defn- axis-sum
  "x summed over axis a, which it keeps (of size 1), in blocks (block-of):
   each block's sum, then the block sums' the same way, and any remainder's
   sum added last. Nothing is padded (with +0.0, a padding would turn a
   sum of -0.0s positive)."
  [x a]
  (let [m (nth (array/shape x) a)]
    (if (<= m block-size)
      (array/sum x {:axis a :keep-dims? true})
      (let [d (block-of m)
            q (quot m d)
            n (* q d)
            s (array/shape x)
            blocks (array/reshape (if (== n m) x (array/slice x a 0 n)) (into (conj (subvec s 0 a) q d) (subvec s (inc a))))
            sums (axis-sum (array/sum blocks {:axis (inc a)}) a)]
        (if (== n m) sums (+ sums (array/sum (array/slice x a n m) {:axis a :keep-dims? true})))))))

(defn sum
  "The float32 MLX array x summed over axes (sorted and distinct) on MLX's
   CPU, in blocks (axis-sum): each run of adjacent axes as one axis, the
   runs one at a time. The shape and dtype are MLX's own sum's."
  [x axes keep-dims?]
  (let [s (array/shape x)
        reduced (set axes)
        ;; runs of adjacent reduced axes, each merged into one dimension
        dims (reduce (fn [dims i]
                       (if (and (reduced i) (reduced (dec i)))
                         (conj (pop dims) (update (peek dims) :size clojure.core/* (nth s i)))
                         (conj dims {:size (nth s i) :reduced? (boolean (reduced i))})))
                     [] (range (count s)))
        merged (array/reshape x (mapv :size dims))
        summed (reduce axis-sum merged (rseq (vec (keep-indexed #(when (:reduced? %2) %1) dims))))]
    (array/reshape summed (if keep-dims?
                            (reduce #(assoc %1 %2 1) s axes)
                            (vec (keep-indexed #(when-not (reduced %1) %2) s))))))

(defn half-sum
  "The float16 or bfloat16 MLX array x summed over axes on MLX's CPU: in
   float32 (in blocks, past one: sum), then cast back to its dtype."
  [x axes keep-dims?]
  (array/astype (array/sum (array/astype x :float32) {:axis axes :keep-dims? keep-dims?}) (array/dtype x)))

(declare float64-unary)

(defn route
  "op of the MLX arrays xs by its exact composite."
  [op xs]
  (case op
    (:exp :sin :cos :erf :erfinv) (float64-unary op (first xs))
    :expm1 (expm1 (first xs))
    :logaddexp (logaddexp (first xs) (second xs))))

(defn vjp
  "The cotangents of a route's arguments xs from the cotangent g of its
   result out: the functions' derivatives, which MLX's autodiff takes for
   these composites in native mode."
  [op xs g out]
  (let [[x y] xs]
    (case op
      :exp [(* g out)]
      :sin [(* g (array/cos x))]
      :cos [(array/negate (* g (array/sin x)))]
      :erf [(* g (* (/ 2.0 (Math/sqrt Math/PI)) (array/exp (array/negate (* x x)))))]
      :erfinv [(* g (* (/ (Math/sqrt Math/PI) 2.0) (array/exp (* out out))))]
      :expm1 [(* g (+ out 1.0))]
      ;; ∂/∂x log(eˣ + eʸ) = eˣ⁻ᵒᵘᵗ, exact at ties
      :logaddexp [(* g (array/exp (- x out))) (* g (array/exp (- y out)))])))

(defn float64-unary
  "op (:exp :sin :cos :erf or :erfinv) of the float64 MLX array x."
  [op x]
  (case op
    :exp (exp x)
    :sin (sin x)
    :cos (cos x)
    :erf (erf x)
    :erfinv (erfinv x)))
