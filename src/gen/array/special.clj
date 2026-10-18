(ns gen.array.special
  "Scalar special functions in double precision, shared by host math and the
   JVM backend's kernels. (lgamma and digamma are composites in gen.array, so
   they are differentiable and identical on every backend.)")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def ^:private ^:const two-over-sqrt-pi 1.1283791670955126)
(def ^:private ^:const sqrt-pi-over-two 0.8862269254527579)

(def ^:private ^"[D" erfc-cof
  "Chebyshev coefficients for erfc on [0, ∞) (Numerical Recipes, 3rd ed., §6.2)."
  (double-array
   [-1.3026537197817094 6.4196979235649026e-1 1.9476473204185836e-2
    -9.561514786808631e-3 -9.46595344482036e-4 3.66839497852761e-4
    4.2523324806907e-5 -2.0278578112534e-5 -1.624290004647e-6
    1.303655835580e-6 1.5626441722e-8 -8.5238095915e-8 6.529054439e-9
    5.059343495e-9 -9.91364156e-10 -2.27365122e-10 9.6467911e-11
    2.394038e-12 -6.886027e-12 8.94487e-13 3.13092e-13 -1.12708e-13
    3.81e-16 7.106e-15 -1.523e-15 -9.4e-17 1.21e-16 -2.8e-17]))

(defn- erfc-nonneg
  "erfc(z) for z >= 0, accurate to about 1e-16 relative."
  ^double [^double z]
  (let [t (/ 2.0 (+ 2.0 z))
        ty (- (* 4.0 t) 2.0)
        n (alength erfc-cof)]
    (loop [j (dec n) d 0.0 dd 0.0]
      (if (pos? j)
        (recur (dec j) (+ (- (* ty d) dd) (aget erfc-cof j)) d)
        (* t (Math/exp (+ (- (* z z)) (* 0.5 (+ (aget erfc-cof 0) (* ty d))) (- dd))))))))

(defn- erf-series
  "erf(x) by its Maclaurin series, for |x| < 0.5 (accurate where 1 - erfc is not)."
  ^double [^double x]
  (let [x2 (- (* x x))]
    (loop [n 1 term x sum x]
      (if (< n 16)
        (let [term (/ (* term x2) n)]
          (recur (inc n) term (+ sum (/ term (inc (* 2 n))))))
        (* two-over-sqrt-pi sum)))))

(defn erf
  "The error function."
  ^double [^double x]
  (cond (Double/isNaN x) x
        (< (Math/abs x) 0.5) (erf-series x)
        (pos? x) (- 1.0 (erfc-nonneg x))
        :else (- (erfc-nonneg (- x)) 1.0)))

(defn erfc
  "The complementary error function 1 - erf(x), accurate in the tails."
  ^double [^double x]
  (cond (Double/isNaN x) x
        (< (Math/abs x) 0.5) (- 1.0 (erf-series x))
        (pos? x) (erfc-nonneg x)
        :else (- 2.0 (erfc-nonneg (- x)))))

(defn- erfinv-guess
  "Giles' single-precision approximation of erfinv (|x| < 1), the starting
   point for Halley's method."
  ^double [^double x]
  (let [w (- (Math/log (* (- 1.0 x) (+ 1.0 x))))]
    (* x
       (if (< w 5.0)
         (let [w (- w 2.5)]
           (-> 2.81022636e-08
               (* w) (+ 3.43273939e-07) (* w) (+ -3.5233877e-06)
               (* w) (+ -4.39150654e-06) (* w) (+ 0.00021858087)
               (* w) (+ -0.00125372503) (* w) (+ -0.00417768164)
               (* w) (+ 0.246640727) (* w) (+ 1.50140941)))
         (let [w (- (Math/sqrt w) 3.0)]
           (-> -0.000200214257
               (* w) (+ 0.000100950558) (* w) (+ 0.00134934322)
               (* w) (+ -0.00367342844) (* w) (+ 0.00573950773)
               (* w) (+ -0.0076224613) (* w) (+ 0.00943887047)
               (* w) (+ 1.00167406) (* w) (+ 2.83297682)))))))

(defn- halley-step
  "x - d/(1 + x·d), with d the Newton correction (erf(x) - ay)/erf'(x), since
   erf''/erf' = -2x. In the tail the residual uses erfc(x) - q, so that
   1 - |y| is never lost to cancellation."
  ^double [^double x ^double ay]
  (let [scale (* sqrt-pi-over-two (Math/exp (* x x)))
        d (if (> ay 0.5)
            (* (- (- 1.0 ay) (erfc-nonneg x)) scale)
            (* (- (erf x) ay) scale))]
    (- x (/ d (+ 1.0 (* x d))))))

(defn- refine-erfinv
  "Halley's method from Giles' guess, for 0 < ay < 1. Convergence is cubic:
   once a step's correction is below 1e-6 relative, the result's error is
   below ~1e-18, so the usual cost is a single erf evaluation; poor guesses
   (far tails) take more steps."
  ^double [^double ay]
  (loop [x (double (erfinv-guess ay)) i 0]
    (let [x' (halley-step x ay)]
      (if (or (== i 6) (<= (Math/abs (- x' x)) (* 1e-6 (Math/abs x'))))
        x'
        (recur x' (inc i))))))

(defn erfinv
  "The inverse error function on [-1, 1] (±Inf at ±1, NaN outside): Giles'
   approximation refined by one Halley step, using erfc in the tails so that
   1 - |y| is never lost to cancellation."
  ^double [^double y]
  (let [ay (Math/abs y)]
    (cond (or (Double/isNaN y) (> ay 1.0)) Double/NaN
          (== ay 1.0) (Math/copySign Double/POSITIVE_INFINITY y)
          (== ay 0.0) y
          :else
          (Math/copySign (refine-erfinv ay) y))))
