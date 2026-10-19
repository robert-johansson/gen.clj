(ns gen.backend.mlx.exact
  "Exact routes around MLX kernels that are not, written in gen.array's ops
   (on MLX arrays). Each is within a few ulps of the JVM backend's, as the
   backend conformance suite measures (docs/array-api.md §13).

   - float64: MLX's CPU kernels for exp, sin, cos, erf and erfinv are float32
     approximations whatever the dtype (about 1e-6 relative error for exp,
     1e-4 for sin and cos). These go through its exact ones (expm1, tan, log,
     division).
   - expm1 on the GPU: Metal's is off by up to ~660 float32 ulps (4e-5
     relative); Kahan's formula from exp and log is within a few."
  (:require [gen.array :as array]))

(set! *warn-on-reflection* true)

(defn- exp
  "e^x from expm1, exact to an ulp: 1/(expm1(-x) + 1) below 0, where
   expm1(x) + 1 would cancel."
  [x]
  (array/where (array/< x 0.0)
               (array// 1.0 (array/+ (array/expm1 (array/negate x)) 1.0))
               (array/+ (array/expm1 x) 1.0)))

(defn- sin
  "sin x = 2t/(1 + t²), t = tan(x/2)."
  [x]
  (let [t (array/tan (array/* x 0.5))]
    (array// (array/* 2.0 t) (array/+ 1.0 (array/* t t)))))

(defn- cos
  "cos x = (1 - t)(1 + t)/(1 + t²), t = tan(x/2), exact near the zeros of
   cos (where 1 - t² would cancel)."
  [x]
  (let [t (array/tan (array/* x 0.5))]
    (array// (array/* (array/- 1.0 t) (array/+ 1.0 t)) (array/+ 1.0 (array/* t t)))))

(def ^:private two-over-sqrt-pi 1.1283791670955126)
(def ^:private sqrt-pi-over-two 0.8862269254527579)

(def ^:private erfc-cof
  "As gen.array.special: Chebyshev coefficients for erfc on [0, ∞)."
  [-1.3026537197817094 6.4196979235649026e-1 1.9476473204185836e-2
   -9.561514786808631e-3 -9.46595344482036e-4 3.66839497852761e-4
   4.2523324806907e-5 -2.0278578112534e-5 -1.624290004647e-6
   1.303655835580e-6 1.5626441722e-8 -8.5238095915e-8 6.529054439e-9
   5.059343495e-9 -9.91364156e-10 -2.27365122e-10 9.6467911e-11
   2.394038e-12 -6.886027e-12 8.94487e-13 3.13092e-13 -1.12708e-13
   3.81e-16 7.106e-15 -1.523e-15 -9.4e-17 1.21e-16 -2.8e-17])

(defn- erfc-nonneg
  "erfc(z) for z >= 0, as gen.array.special (Clenshaw's recurrence)."
  [z]
  (let [t (array// 2.0 (array/+ 2.0 z))
        ty (array/- (array/* 4.0 t) 2.0)
        [d dd] (reduce (fn [[d dd] c] [(array/+ (array/- (array/* ty d) dd) c) d])
                       [0.0 0.0]
                       (rseq (subvec erfc-cof 1)))]
    (array/* t (exp (array/- (array/+ (array/negate (array/* z z)) (array/* 0.5 (array/+ (first erfc-cof) (array/* ty d)))) dd)))))

(defn- erf-series
  "erf(x) by its Maclaurin series (for |x| < 0.5)."
  [x]
  (let [x2 (array/negate (array/* x x))]
    (loop [n 1 term x sum x]
      (if (< n 16)
        (let [term (array// (array/* term x2) n)]
          (recur (inc n) term (array/+ sum (array// term (inc (* 2 n))))))
        (array/* two-over-sqrt-pi sum)))))

(defn- erf [x]
  (let [ax (array/abs x)
        tail (array/- 1.0 (erfc-nonneg ax))]
    (array/where (array/< ax 0.5) (erf-series x) (array/where (array/< x 0.0) (array/negate tail) tail))))

(defn- erfinv
  "As gen.array.special/erfinv: Giles' approximation refined by Halley's
   method, with the residual from erfc in the tails. Branch-free, so a
   fixed seven steps (the JVM backend's most)."
  [y]
  (let [ay (array/abs y)
        ;; a stand-in away from ±1 and 0 keeps the discarded lanes finite
        special? (array/logical-or (array/logical-or (array/isnan y) (array/>= ay 1.0)) (array/== ay 0.0))
        q (array/where special? 0.5 ay)
        w (array/negate (array/log (array/* (array/- 1.0 q) (array/+ 1.0 q))))
        horner (fn [v cs] (reduce (fn [acc c] (array/+ (array/* acc v) c)) (first cs) (rest cs)))
        central (horner (array/- w 2.5) [2.81022636e-08 3.43273939e-07 -3.5233877e-06 -4.39150654e-06
                                         0.00021858087 -0.00125372503 -0.00417768164 0.246640727 1.50140941])
        tail (horner (array/- (array/sqrt w) 3.0) [-0.000200214257 0.000100950558 0.00134934322 -0.00367342844
                                                   0.00573950773 -0.0076224613 0.00943887047 1.00167406 2.83297682])
        x0 (array/* q (array/where (array/< w 5.0) central tail))
        halley (fn [x]
                 (let [scale (array/* sqrt-pi-over-two (exp (array/* x x)))
                       d (array/where (array/> q 0.5)
                                      (array/* (array/- (array/- 1.0 q) (erfc-nonneg x)) scale)
                                      (array/* (array/- (erf x) q) scale))]
                   (array/- x (array// d (array/+ 1.0 (array/* x d))))))
        ;; as many steps as the JVM backend's adaptive loop at most takes,
        ;; which the far tails (|y| near 1 - 1e-16) need
        x (nth (iterate halley x0) 7)
        signed (array/where (array/< y 0.0) (array/negate x) x)]
    (array/where (array/logical-or (array/isnan y) (array/> ay 1.0))
                 ##NaN
                 (array/where (array/== ay 1.0)
                              (array/where (array/< y 0.0) ##-Inf ##Inf)
                              (array/where (array/== ay 0.0) y signed)))))

(defn expm1
  "e^x - 1 by Kahan's formula: (u - 1)·x/log u for u = e^x, exact to a few
   ulps where exp and log are, also near 0 (where u - 1 alone would cancel)."
  [x]
  (let [u (array/exp x)
        um1 (array/- u 1.0)
        ;; a stand-in keeps the discarded lanes finite
        safe (array/where (array/logical-or (array/== um1 0.0) (array/isinf u)) 2.0 u)]
    (array/where (array/== um1 0.0)
                 x
                 (array/where (array/logical-or (array/isinf u) (array/== um1 -1.0))
                              um1
                              (array// (array/* (array/- safe 1.0) x) (array/log safe))))))

(defn logaddexp
  "gen.array's composite logaddexp of float64 MLX arrays, with the exact exp."
  [x y]
  (let [m (array/maximum x y)]
    (array/where (array/isinf m) m (array/+ m (array/log1p (exp (array/negate (array/abs (array/- x y)))))))))

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
      :exp [(array/* g out)]
      :sin [(array/* g (array/cos x))]
      :cos [(array/negate (array/* g (array/sin x)))]
      :erf [(array/* g (array/* (/ 2.0 (Math/sqrt Math/PI)) (array/exp (array/negate (array/* x x)))))]
      :erfinv [(array/* g (array/* (/ (Math/sqrt Math/PI) 2.0) (array/exp (array/* out out))))]
      :expm1 [(array/* g (array/+ out 1.0))]
      ;; ∂/∂x log(eˣ + eʸ) = eˣ⁻ᵒᵘᵗ, exact at ties
      :logaddexp [(array/* g (array/exp (array/- x out))) (array/* g (array/exp (array/- y out)))])))

(defn float64-unary
  "op (:exp :sin :cos :erf or :erfinv) of the float64 MLX array x."
  [op x]
  (case op
    :exp (exp x)
    :sin (sin x)
    :cos (cos x)
    :erf (erf x)
    :erfinv (erfinv x)))
