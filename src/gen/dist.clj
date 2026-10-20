(ns gen.dist
  "Gen.jl's built-in distributions, with Gen.jl's names and parameterizations
   (normal by standard deviation, gamma by scale, categorical by
   probabilities), plus categorical-logits, and further continuous
   families with Distributions.jl's names and parameters (chi-squared,
   f-dist, gumbel, levy, log-normal, logistic, nakagami, pareto, student-t,
   triangular, weibull). Discrete values are 0-based
   integers; bernoulli values are booleans.

   Every log-prob is -Inf outside the support. Every sampler is a pure,
   branch-free function of its key: rejection samplers run a fixed number of
   vectorized rounds rather than a data-dependent loop, so they batch and never
   force evaluation."
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= == log log1p exp sqrt square where
                                     lgamma log-beta xlogy xlog1py logical-and logical-or]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.array.shape :as sh]
            [gen.dist.core :as dc :refer [defdist]]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def ^:private half-log-2pi 0.9189385332046728)

(defn- guard
  "lp inside the support, -Inf outside."
  [in-support lp]
  (where in-support lp ##-Inf))

(defn- integral? [v] (== v (array/floor v)))

(defn- opts [& params] {:backend (apply dc/backend-of params)})

(defn- to-int [x & params]
  (array/astype x (dc/int-dtype (apply dc/backend-of x params))))

;; ---------------------------------------------------------------------------
;; Continuous distributions
;; ---------------------------------------------------------------------------

(defdist normal
  "normal(mu, std): the Gaussian with mean mu and standard deviation std.
   Array parameters give an array value whose log density is the sum over its
   elements (Gen.jl's broadcasted_normal)."
  [mu std]
  (validate (dc/check-positive "normal" "std" std))
  (sample [key shape]
    (+ mu (* std (rng/normal key (dc/elementwise-shape shape mu std) (opts mu std)))))
  (log-prob [v]
    (- (+ half-log-2pi (log std) (* 0.5 (square (/ (- v mu) std)))))))

(def broadcasted-normal
  "broadcasted_normal(mu, std): normal with broadcast array parameters (in
   gen.clj, normal itself already broadcasts)."
  normal)

(defdist uniform
  "uniform(low, high): uniform on [low, high]."
  [low high]
  (validate (when (and (array/host? low) (array/host? high))
              (dc/check (< low high) "uniform" "low must be below high" {:low low :high high})))
  (sample [key shape]
    (rng/uniform key (dc/elementwise-shape shape low high) (assoc (opts low high) :low low :high high)))
  (log-prob [v]
    (guard (logical-and (>= v low) (<= v high)) (- (log (- high low))))))

(defdist beta
  "beta(alpha, beta): the beta distribution on [0, 1]."
  [a b]
  (validate (dc/check-positive "beta" "alpha" a) (dc/check-positive "beta" "beta" b))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape a b)
          [k1 k2] (rng/split key)
          x (rng/gamma k1 a s (opts a b))
          y (rng/gamma k2 b s (opts a b))]
      (/ x (+ x y))))
  (log-prob [v]
    (guard (logical-and (>= v 0) (<= v 1))
           (- (+ (xlogy (- a 1) v) (xlog1py (- b 1) (- v))) (log-beta a b)))))

(defdist gamma
  "gamma(shape, scale): the gamma distribution with mean shape·scale."
  [k theta]
  (validate (dc/check-positive "gamma" "shape" k) (dc/check-positive "gamma" "scale" theta))
  (sample [key shape]
    (* theta (rng/gamma key k (dc/elementwise-shape shape k theta) (opts k theta))))
  (log-prob [v]
    (guard (>= v 0) (- (xlogy (- k 1) v) (/ v theta) (lgamma k) (* k (log theta))))))

(defdist inv-gamma
  "inv_gamma(shape, scale): the inverse gamma distribution (1/X for X with
   gamma(shape, 1/scale))."
  [k theta]
  (validate (dc/check-positive "inv-gamma" "shape" k) (dc/check-positive "inv-gamma" "scale" theta))
  (sample [key shape]
    (/ theta (rng/gamma key k (dc/elementwise-shape shape k theta) (opts k theta))))
  (log-prob [v]
    (guard (> v 0) (- (* k (log theta)) (lgamma k) (* (+ k 1) (log v)) (/ theta v)))))

(defdist exponential
  "exponential(rate): the exponential distribution with mean 1/rate."
  [rate]
  (validate (dc/check-positive "exponential" "rate" rate))
  (sample [key shape]
    (/ (rng/exponential key (dc/elementwise-shape shape rate) (opts rate)) rate))
  (log-prob [v]
    (guard (>= v 0) (- (log rate) (* rate v)))))

(defdist laplace
  "laplace(loc, scale): the Laplace (double exponential) distribution."
  [loc scale]
  (validate (dc/check-positive "laplace" "scale" scale))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape loc scale)
                         (assoc (opts loc scale) :low (Math/nextAfter -1.0 0.0) :high 1.0))]
      (+ loc (* scale (array/sign u) (log1p (- (array/abs u)))))))
  (log-prob [v]
    (- (+ (log (* 2 scale)) (/ (array/abs (- v loc)) scale)))))

(defdist cauchy
  "cauchy(x0, gamma): the Cauchy distribution with location x0 and scale gamma."
  [x0 scale]
  (validate (dc/check-positive "cauchy" "scale" scale))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape x0 scale)
                         (assoc (opts x0 scale) :low (Math/ulp 1.0) :high 1.0))]
      (+ x0 (* scale (array/tan (* Math/PI (- u 0.5)))))))
  (log-prob [v]
    (- (+ (log (* Math/PI scale)) (log1p (square (/ (- v x0) scale)))))))

(defdist beta-uniform
  "beta_uniform(theta, alpha, beta): with probability theta a beta(alpha, beta)
   draw, otherwise a uniform draw on [0, 1]."
  [theta a b]
  (validate (dc/check-probability "beta-uniform" "theta" theta)
            (dc/check-positive "beta-uniform" "alpha" a)
            (dc/check-positive "beta-uniform" "beta" b))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape theta a b)
          [k1 k2 k3] (rng/split key 3)]
      (where (rng/bernoulli k1 theta s (opts theta a b))
             (dc/sample (beta a b) k2 s)
             (rng/uniform k3 s (opts theta a b)))))
  (log-prob [v]
    (guard (logical-and (>= v 0) (<= v 1))
           (array/logaddexp (+ (log theta) (dc/log-prob (beta a b) v)) (log1p (- theta))))))

(defdist piecewise-uniform
  "piecewise_uniform(bounds, probs): bin i, chosen with probability probs[i],
   is [bounds[i], bounds[i+1]], uniform within the bin."
  [bounds probs]
  (validate (dc/check-probabilities "piecewise-uniform" "probs" probs)
            (dc/check-increasing "piecewise-uniform" "bounds" bounds)
            (dc/check (= (peek (array/shape bounds)) (inc (peek (array/shape probs)))) "piecewise-uniform"
                      "needs one more bound than probabilities" {:bounds (array/shape bounds) :probs (array/shape probs)}))
  (sample [key shape]
    (let [[k1 k2] (rng/split key)
          i (rng/categorical k1 (log probs) shape (opts bounds probs))
          lo (array/take bounds i)
          hi (array/take bounds (+ i 1))]
      (+ lo (* (- hi lo) (rng/uniform k2 (array/shape i) (opts bounds probs))))))
  (log-prob [v]
    (let [n (array/size probs)
          i (- (array/sum (>= (array/expand-dims v -1) (array/slice bounds 0 0 n)) {:axis -1}) 1)
          i (to-int (array/clip i 0 (dec n)) bounds)]
      (guard (logical-and (>= v (array/take bounds 0)) (<= v (array/take bounds n)))
             (- (log (array/take probs i)) (log (- (array/take bounds (+ i 1)) (array/take bounds i))))))))

(def ^:private machine-epsilon {:float32 (double (Math/ulp (float 1.0))) :float64 (Math/ulp 1.0)})

(defn- factored-dtype
  "The precision cov is factored in: float32 when it or the default
   backend's floats are (host data become the default backend's arrays),
   else float64."
  [cov]
  (if (some #{:float32} [(array/dtype cov) (dc/float-dtype (dc/backend-of))]) :float32 :float64))

(defn- positive-definite?
  "Is the symmetric a (each matrix of a batch) positive definite, as far as
   dtype's rounding can tell? A Cholesky factor L does not show it alone: a
   singular matrix can have one through rounding (2·ones(2, 2)'s last pivot
   is 4e-16, not 0). So the correlation matrix H of L·Lᵀ must also have no
   eigenvalue within the rounding error of an n-term sum, n·ε: trace(H⁻¹),
   between 1/λmin and n/λmin, is below 1/(n·ε)."
  [a dtype]
  (let [n (peek (array/shape a))
        eye (array/with-backend (array/backend a) (array/eye n))
        ;; H⁻¹ = MᵀM for M = L⁻¹·diag(√aⱼⱼ); a NaN factor fails the comparison
        m (la/solve-triangular (la/cholesky a) (* eye (array/expand-dims (sqrt (la/diagonal a)) -2)))]
    (array/all-true? (< (* n (machine-epsilon dtype) (array/sum (square m) {:axis [-2 -1]})) 1.0))))

(defdist mvnormal
  "mvnormal(mu, cov): the multivariate normal with mean vector mu and
   covariance matrix cov (batched over leading axes). A concrete cov must be
   symmetric and positive definite, also to the precision it is factored in."
  [mu cov]
  (validate (dc/check (and (<= 1 (array/ndim mu)) (<= 2 (array/ndim cov))
                           (= (peek (array/shape mu)) (peek (array/shape cov)) (peek (pop (array/shape cov)))))
                      "mvnormal" (str "mu must be a vector and cov a square matrix of its size, got shapes "
                                      (array/shape mu) " and " (array/shape cov))
                      {:mu-shape (array/shape mu) :cov-shape (array/shape cov)})
            (when (array/concrete? cov)
              (let [r (array/ndim cov)
                    cov-t (array/transpose cov (-> (vec (range r)) (assoc (- r 2) (dec r)) (assoc (dec r) (- r 2))))]
                (dc/check (array/all-true? (<= (array/abs (- cov cov-t)) (+ 1e-8 (* 1e-5 (array/abs cov-t)))))
                          "mvnormal" "cov must be symmetric" {})
                (let [dtype (factored-dtype cov)]
                  (dc/check (positive-definite? cov dtype) "mvnormal"
                            (str "cov must be positive definite, and not singular at " (name dtype) " precision")
                            {:param "cov" :dtype dtype})))))
  (sample [key shape]
    (let [d (peek (array/shape mu))
          batch (sh/broadcast (sh/broadcast shape (pop (array/shape mu))) (pop (pop (array/shape cov))))
          eps (rng/normal key (conj batch d) (opts mu cov))]
      (+ mu (array/squeeze (array/matmul (la/cholesky cov) (array/expand-dims eps -1)) -1))))
  (log-prob [v]
    (let [d (peek (array/shape mu))
          l (la/cholesky cov)
          diff (- v mu)
          ;; one column right-hand side per batch element: the batch of the
          ;; value and of the factor broadcast together
          batch (sh/broadcast (pop (array/shape diff)) (pop (pop (array/shape l))))
          z (array/squeeze (la/solve-triangular (array/broadcast-to l (into batch [d d]))
                                                (array/broadcast-to (array/expand-dims diff -1) (into batch [d 1])))
                           -1)]
      (- (* -0.5 (array/sum (square z) {:axis -1}))
         (* 0.5 (la/cholesky-logdet l))
         (* d half-log-2pi)))))

(defdist dirichlet
  "dirichlet(alpha): the Dirichlet distribution over the simplex of the last
   axis of alpha."
  [alpha]
  (validate (dc/check (<= 1 (array/ndim alpha)) "dirichlet" "alpha must be a vector" {:alpha-shape (array/shape alpha)})
            (dc/check-all-positive "dirichlet" "alpha" alpha))
  (sample [key shape]
    (let [k (peek (array/shape alpha))
          g (rng/gamma key alpha (sh/broadcast (conj shape k) (array/shape alpha)) (opts alpha))]
      (/ g (array/sum g {:axis -1 :keep-dims? true}))))
  (log-prob [v]
    (guard (logical-and (array/all (>= v 0) {:axis -1})
                        (<= (array/abs (- (array/sum v {:axis -1}) 1)) 1e-6))
           (- (array/sum (xlogy (- alpha 1) v) {:axis -1})
              (- (array/sum (lgamma alpha) {:axis -1}) (lgamma (array/sum alpha {:axis -1})))))))

;; ---------------------------------------------------------------------------
;; Beyond Gen.jl's built-ins: Distributions.jl's families with its names and
;; parameters (what GenDistributions' users write), native, so they batch,
;; run on every backend and differentiate (gen.dist.commons-math wraps a
;; host library's for the rest)
;; ---------------------------------------------------------------------------

(def ^:private log-2 (Math/log 2.0))

(defn- stand-in
  "v where in-support holds, else s: the other lanes stay finite, so a
   gradient through the guard is not NaN."
  [in-support v s]
  (where in-support v s))

(defn- floored
  "x at least the smallest normal number of its dtype. A gamma draw of a
   small shape (its boost U^(1/α)) or a power of one underflows to 0, where
   a density may be infinite or a quotient infinite; the floor changes only
   values below it."
  [x]
  (array/maximum x (if (= :float32 (array/dtype x)) (double Float/MIN_NORMAL) Double/MIN_NORMAL)))

(defdist chi-squared
  "chi-squared(nu): the χ² distribution with nu degrees of freedom (Chisq)."
  [nu]
  (validate (dc/check-positive "chi-squared" "nu" nu))
  (sample [key shape] (* 2.0 (floored (rng/gamma key (/ nu 2.0) (dc/elementwise-shape shape nu) (opts nu)))))
  (log-prob [v]
    (let [k (/ nu 2.0)]
      (guard (>= v 0) (- (xlogy (- k 1) v) (/ v 2.0) (lgamma k) (* k log-2))))))

(defdist f-dist
  "f-dist(nu1, nu2): Fisher's F distribution, the ratio of χ² draws over
   their degrees of freedom (FDist)."
  [nu1 nu2]
  (validate (dc/check-positive "f-dist" "nu1" nu1) (dc/check-positive "f-dist" "nu2" nu2))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape nu1 nu2)
          [k1 k2] (rng/split key)]
      (/ (* nu2 (floored (rng/gamma k1 (/ nu1 2.0) s (opts nu1 nu2))))
         (* nu1 (floored (rng/gamma k2 (/ nu2 2.0) s (opts nu1 nu2)))))))
  (log-prob [v]
    ;; above 1, (nu1/2 - 1) log x - (nu1+nu2)/2 log(nu2 + nu1 x) as
    ;; -(nu2/2 + 1) log x - (nu1+nu2)/2 log(nu1 + nu2/x): -Inf at Inf, not Inf - Inf
    (let [in (>= v 0)
          x (stand-in in v 1.0)
          big (> x 1.0)
          xb (stand-in big x 2.0)
          kernel (where big
                        (- (- (* (+ (/ nu2 2.0) 1) (log xb))) (* (/ (+ nu1 nu2) 2.0) (log (+ nu1 (/ nu2 xb)))))
                        (- (xlogy (- (/ nu1 2.0) 1) x) (* (/ (+ nu1 nu2) 2.0) (log (+ nu2 (* nu1 x))))))]
      (guard in (- (+ (* 0.5 (+ (* nu1 (log nu1)) (* nu2 (log nu2)))) kernel)
                   (log-beta (/ nu1 2.0) (/ nu2 2.0)))))))

(defdist gumbel
  "gumbel(mu, theta): the Gumbel (maximum) distribution, location mu and
   scale theta (Gumbel)."
  [mu theta]
  (validate (dc/check-positive "gumbel" "theta" theta))
  (sample [key shape] (+ mu (* theta (rng/gumbel key (dc/elementwise-shape shape mu theta) (opts mu theta)))))
  (log-prob [v]
    (let [z (/ (- v mu) theta)]
      (- (+ (log theta) z (exp (- z)))))))

(defdist levy
  "levy(mu, sigma): the Lévy distribution, location mu and scale sigma,
   mu + sigma/Z² for a standard normal Z (Levy)."
  [mu sigma]
  (validate (dc/check-positive "levy" "sigma" sigma))
  (sample [key shape]
    (let [z (rng/normal key (dc/elementwise-shape shape mu sigma) (opts mu sigma))]
      (+ mu (/ sigma (floored (square z))))))
  (log-prob [v]
    (let [in (> v mu)
          d (stand-in in (- v mu) 1.0)]
      (guard in (- (* 0.5 (log (/ sigma (* 2.0 Math/PI)))) (* 1.5 (log d)) (/ sigma (* 2.0 d)))))))

(defdist log-normal
  "log-normal(mu, sigma): e^X for X normal(mu, sigma) (LogNormal)."
  [mu sigma]
  (validate (dc/check-positive "log-normal" "sigma" sigma))
  (sample [key shape]
    (exp (+ mu (* sigma (rng/normal key (dc/elementwise-shape shape mu sigma) (opts mu sigma))))))
  (log-prob [v]
    (let [in (> v 0)
          lx (log (stand-in in v 1.0))]
      (guard in (- (+ lx (log sigma) half-log-2pi (* 0.5 (square (/ (- lx mu) sigma)))))))))

(defdist logistic
  "logistic(mu, theta): the logistic distribution, location mu and scale
   theta (Logistic)."
  [mu theta]
  (validate (dc/check-positive "logistic" "theta" theta))
  ;; the difference of two standard Gumbel draws is standard logistic
  (sample [key shape]
    (let [s (dc/elementwise-shape shape mu theta)
          [k1 k2] (rng/split key)]
      (+ mu (* theta (- (rng/gumbel k1 s (opts mu theta)) (rng/gumbel k2 s (opts mu theta)))))))
  (log-prob [v]
    (let [z (/ (- v mu) theta)]
      (- (+ z (log theta) (* 2.0 (array/logaddexp 0.0 (- z))))))))

(defdist nakagami
  "nakagami(m, omega): the Nakagami distribution, shape m and spread omega,
   the root of a gamma(m, omega/m) draw (Nakagami; m > 0, as
   Distributions.jl's, where Commons Math asks m >= 1/2)."
  [m omega]
  (validate (dc/check-positive "nakagami" "m" m) (dc/check-positive "nakagami" "omega" omega))
  (sample [key shape]
    (sqrt (* (/ omega m) (floored (rng/gamma key m (dc/elementwise-shape shape m omega) (opts m omega))))))
  (log-prob [v]
    (guard (>= v 0) (- (+ log-2 (* m (log m)) (xlogy (- (* 2.0 m) 1) v))
                       (lgamma m) (* m (log omega)) (/ (* m (square v)) omega)))))

(defdist pareto
  "pareto(alpha, theta): the Pareto distribution, shape alpha and scale
   (minimum) theta (Pareto). For a small alpha some draws lie beyond the
   float range: +Inf, of log density -Inf."
  [alpha theta]
  (validate (dc/check-positive "pareto" "alpha" alpha) (dc/check-positive "pareto" "theta" theta))
  (sample [key shape]
    (* theta (exp (/ (rng/exponential key (dc/elementwise-shape shape alpha theta) (opts alpha theta)) alpha))))
  (log-prob [v]
    (let [in (>= v theta)]
      (guard in (- (+ (log alpha) (* alpha (log theta))) (* (+ alpha 1) (log (stand-in in v theta))))))))

(defdist student-t
  "student-t(nu): Student's t distribution with nu degrees of freedom, a
   standard normal over the root of a χ²/nu draw (TDist)."
  [nu]
  (validate (dc/check-positive "student-t" "nu" nu))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape nu)
          [k1 k2] (rng/split key)
          g (rng/gamma k2 (/ nu 2.0) s (opts nu))]
      (* (rng/normal k1 s (opts nu)) (sqrt (/ nu (* 2.0 (floored g)))))))
  (log-prob [v]
    (- (lgamma (/ (+ nu 1) 2.0)) (lgamma (/ nu 2.0)) (* 0.5 (log (* nu Math/PI)))
       (* (/ (+ nu 1) 2.0) (log1p (/ (square v) nu))))))

(defdist triangular
  "triangular(a, b, c): the triangular distribution on [a, b] with mode c
   (TriangularDist)."
  [a b c]
  (validate (when (every? #(and (some? %) (array/concrete? %)) [a b c])
              (dc/check (array/all-true? (logical-and (< a b) (logical-and (<= a c) (<= c b))))
                        "triangular" "needs a < b and a <= c <= b" {:a a :b b :c c})))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape a b c) (opts a b c))
          w (- b a)]
      (where (< u (/ (- c a) w))
             (+ a (sqrt (* u w (- c a))))
             (- b (sqrt (* (- 1.0 u) w (- b c)))))))
  (log-prob [v]
    (let [w (- b a)
          left (> (- c a) 0)
          right (> (- b c) 0)]
      ;; the mode separately: at c = b (or c = a) one side has no width
      (where (== v c)
             (log (/ 2.0 w))
             (where (logical-and (>= v a) (< v c))
                    (log (/ (* 2.0 (- v a)) (* w (stand-in left (- c a) 1.0))))
                    (where (logical-and (> v c) (<= v b))
                           (log (/ (* 2.0 (- b v)) (* w (stand-in right (- b c) 1.0))))
                           ##-Inf))))))

(defdist weibull
  "weibull(alpha, theta): the Weibull distribution, shape alpha and scale
   theta (Weibull). For a small alpha some draws lie beyond the float range
   (in float32, for alpha below about 0.03): +Inf, of log density -Inf."
  [alpha theta]
  (validate (dc/check-positive "weibull" "alpha" alpha) (dc/check-positive "weibull" "theta" theta))
  (sample [key shape]
    (* theta (floored (array/pow (rng/exponential key (dc/elementwise-shape shape alpha theta) (opts alpha theta)) (/ 1.0 alpha)))))
  (log-prob [v]
    (let [in (>= v 0)
          x (/ (stand-in in v theta) theta)]
      (guard in (- (+ (log (/ alpha theta)) (xlogy (- alpha 1) x)) (array/pow x alpha))))))

;; ---------------------------------------------------------------------------
;; Discrete distributions
;; ---------------------------------------------------------------------------

(defdist bernoulli
  "bernoulli(prob): true with probability prob."
  [p]
  (validate (dc/check-probability "bernoulli" "prob" p))
  (sample [key shape] (rng/bernoulli key p (dc/elementwise-shape shape p) (opts p)))
  (log-prob [v]
    ;; 0/1 would be scored as false/true while a body's if takes 0 as true
    (dc/check (= :bool (array/dtype v)) "bernoulli"
              (str "values are booleans (true or false), not " (pr-str (if (array/host? v) v (array/dtype v)))
                   " (GenMLX's bernoulli gave 0/1)")
              {:value-dtype (array/dtype v)})
    (where v (log p) (log1p (- p))))
  (support [] [false true]))

(defn- pick-guarded
  "log-probs[..., v] for integer v in [0, K), -Inf for any other v."
  [log-probs v]
  (let [k (peek (array/shape log-probs))
        ok (logical-and (logical-and (>= v 0) (< v k)) (integral? v))]
    (guard ok (array/pick log-probs (to-int (array/clip v 0 (dec k)) log-probs)))))

(defdist categorical
  "categorical(probs): index i (0-based) with probability probs[i]; probs must
   be probabilities (see categorical-logits for logits)."
  [probs]
  (validate (dc/check-probabilities "categorical" "probs" probs))
  (sample [key shape] (rng/categorical key (log probs) (sh/broadcast shape (pop (array/shape probs))) (opts probs)))
  (log-prob [v] (pick-guarded (log probs) v))
  (support [] (vec (range (peek (array/shape probs))))))

(defdist categorical-logits
  "categorical-logits(logits): index i (0-based) with probability
   softmax(logits)[i]. Any finite logits; the form LLM token sites use."
  [logits]
  (sample [key shape] (rng/categorical key logits (sh/broadcast shape (pop (array/shape logits))) (opts logits)))
  (log-prob [v] (pick-guarded (array/log-softmax logits) v))
  (support [] (vec (range (peek (array/shape logits))))))

(defdist uniform-discrete
  "uniform_discrete(low, high): an integer uniform on low..high inclusive."
  [low high]
  (validate (dc/check-integer "uniform-discrete" "low" low Long/MIN_VALUE)
            (dc/check-integer "uniform-discrete" "high" high Long/MIN_VALUE)
            (when (and (array/host? low) (array/host? high))
              (dc/check (<= low high) "uniform-discrete" "low must not exceed high" {:low low :high high})))
  (sample [key shape] (rng/randint key low (+ high 1) (dc/elementwise-shape shape low high) (opts low high)))
  (log-prob [v]
    (guard (logical-and (logical-and (>= v low) (<= v high)) (integral? v))
           (- (log (+ (- high low) 1)))))
  (support [] (vec (range (dc/support-bound "uniform-discrete" "low" low)
                          (inc (dc/support-bound "uniform-discrete" "high" high))))))

(defdist geometric
  "geometric(p): the number of failures before the first success."
  [p]
  (validate (dc/check-probability "geometric" "p" p)
            (when (array/host? p) (dc/check (> p 0) "geometric" "p must be in (0, 1]" {:p p})))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape p) (opts p))]
      (to-int (array/floor (/ (log1p (- u)) (log1p (- p)))) p)))
  (log-prob [v]
    (guard (logical-and (>= v 0) (integral? v)) (+ (log p) (xlog1py v (- p))))))

;; --- Poisson and binomial samplers: inversion for small means, Hörmann's
;; transformed rejection (PTRS, BTRS) otherwise, each with fixed rounds ------

(def ^:private rejection-rounds 12)
(def ^:private inversion-steps 40)

(defn- first-accepted
  "The value of the first accepted proposal of [accept? value] pairs, else
   fallback."
  [fallback proposals]
  (reduce (fn [acc [ok v]] (where ok v acc)) fallback (reverse proposals)))

(defn- poisson-sample [key lam shape]
  (let [o (opts lam)
        [k-inv k-rej] (rng/split key)
        ;; inversion (lam < 10): count the CDF values below u
        u (rng/uniform k-inv shape o)
        small (first (reduce (fn [[k p cdf] j]
                               (let [p (* p (/ lam (+ j 1)))]
                                 [(+ k (where (> u cdf) 1 0)) p (+ cdf p)]))
                             [0 (exp (- lam)) (exp (- lam))]
                             (range inversion-steps)))
        ;; PTRS (lam >= 10)
        log-lam (log lam)
        b (+ 0.931 (* 2.53 (sqrt lam)))
        aa (+ -0.059 (* 0.02483 b))
        inv-alpha (+ 1.1239 (/ 1.1328 (- b 3.4)))
        vr (- 0.9277 (/ 3.6224 (- b 2)))
        ks (rng/split k-rej (* 2 rejection-rounds))
        proposal (fn [r]
                   (let [uu (- (rng/uniform (ks (* 2 r)) shape o) 0.5)
                         v (rng/uniform (ks (+ 1 (* 2 r))) shape o)
                         us (- 0.5 (array/abs uu))
                         k (array/floor (+ (* (+ (/ (* 2 aa) us) b) uu) lam 0.43))
                         quick (logical-and (>= us 0.07) (<= v vr))
                         reject (logical-or (< k 0) (logical-and (< us 0.013) (> v us)))
                         s (log (/ (* v inv-alpha) (+ (/ aa (* us us)) b)))
                         t (- (* k log-lam) lam (lgamma (+ k 1)))]
                     [(logical-or quick (logical-and (array/logical-not reject) (<= s t))) k]))
        large (first-accepted (array/round lam) (map proposal (range rejection-rounds)))]
    (to-int (where (< lam 10) small large) lam)))

(defn- binomial-sample [key n p shape]
  (let [o (opts n p)
        flip? (> p 0.5)
        p (where flip? (- 1 p) p)
        q (- 1 p)
        [k-inv k-rej] (rng/split key)
        u (rng/uniform k-inv shape o)
        ;; inversion (n·p < 10)
        p0 (exp (* n (log1p (- p))))
        small (first (reduce (fn [[k pk cdf] j]
                               (let [pk (* pk (/ (* (- n j) p) (* (+ j 1) q)))]
                                 [(+ k (where (> u cdf) 1 0)) pk (+ cdf pk)]))
                             [0 p0 p0]
                             (range inversion-steps)))
        ;; BTRS (n·p >= 10)
        spq (sqrt (* n p q))
        b (+ 1.15 (* 2.53 spq))
        aa (+ -0.0873 (* 0.0248 b) (* 0.01 p))
        c (+ (* n p) 0.5)
        vr (- 0.92 (/ 4.2 b))
        alpha (* (+ 2.83 (/ 5.1 b)) spq)
        lpq (log (/ p q))
        m (array/floor (* (+ n 1) p))
        h (+ (lgamma (+ m 1)) (lgamma (+ (- n m) 1)))
        ks (rng/split k-rej (* 2 rejection-rounds))
        proposal (fn [r]
                   (let [uu (- (rng/uniform (ks (* 2 r)) shape o) 0.5)
                         v (rng/uniform (ks (+ 1 (* 2 r))) shape o)
                         us (- 0.5 (array/abs uu))
                         k (array/floor (+ (* (+ (/ (* 2 aa) us) b) uu) c))
                         in-range (logical-and (>= k 0) (<= k n))
                         quick (logical-and (>= us 0.07) (<= v vr))
                         v' (log (/ (* v alpha) (+ (/ aa (* us us)) b)))
                         slow (<= v' (+ (- h (lgamma (+ k 1)) (lgamma (+ (- n k) 1))) (* (- k m) lpq)))]
                     [(logical-and in-range (logical-or quick slow)) k]))
        large (first-accepted (array/round (* n p)) (map proposal (range rejection-rounds)))
        x (where (< (* n p) 10) small large)]
    (to-int (where flip? (- n x) x) n p)))

(defdist poisson
  "poisson(lambda): the Poisson distribution with mean lambda."
  [lam]
  (validate (when (array/host? lam) (dc/check (>= lam 0) "poisson" "lambda must be >= 0" {:lambda lam})))
  (sample [key shape] (poisson-sample key lam (dc/elementwise-shape shape lam)))
  (log-prob [v]
    (guard (logical-and (>= v 0) (integral? v)) (- (xlogy v lam) lam (lgamma (+ v 1))))))

(defdist binom
  "binom(n, p): the number of successes in n trials with success probability p."
  [n p]
  (validate (dc/check-probability "binom" "p" p)
            (dc/check-integer "binom" "n" n 0))
  (sample [key shape] (binomial-sample key n p (dc/elementwise-shape shape n p)))
  (log-prob [v]
    (guard (logical-and (logical-and (>= v 0) (<= v n)) (integral? v))
           (+ (- (lgamma (+ n 1)) (lgamma (+ v 1)) (lgamma (+ (- n v) 1)))
              (xlogy v p) (xlog1py (- n v) (- p)))))
  (support [] (vec (range (inc (dc/support-bound "binom" "n" n))))))

(defdist neg-binom
  "neg_binom(r, p): the number of failures before the r-th success (r may be
   real)."
  [r p]
  (validate (dc/check-positive "neg-binom" "r" r) (dc/check-probability "neg-binom" "p" p))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape r p)
          [k1 k2] (rng/split key)
          lam (* (rng/gamma k1 r s (opts r p)) (/ (- 1 p) p))]
      (poisson-sample k2 lam s)))
  (log-prob [v]
    (guard (logical-and (>= v 0) (integral? v))
           (+ (- (lgamma (+ v r)) (lgamma r) (lgamma (+ v 1)))
              (* r (log p)) (xlog1py v (- p))))))

;; ---------------------------------------------------------------------------
;; Mixtures and products: Gen.jl's HomogeneousMixture, HeterogeneousMixture
;; and ProductDistribution. Each is a family: a function of the component
;; parameters returning a distribution, called at the trace site like any
;; distribution constructor.
;; ---------------------------------------------------------------------------

(defn homogeneous-mixture
  "A family of mixtures of one distribution family (Gen.jl's
   HomogeneousMixture, for families with scalar values and scalar
   parameters per component):
   ((homogeneous-mixture normal) weights mus stds) has components
   (normal (mus k) (stds k)). The last axis of each parameter indexes the
   components; weights are the components' probabilities."
  [family]
  (fn [weights & params]
    (let [_ (dc/check-probabilities "homogeneous-mixture" "weights" (dc/->checkable weights))
          w (dc/->array weights)
          params (mapv dc/->array params)]
      (dc/->Distribution ::homogeneous-mixture {:family family :weights w :params params}))))

(defmethod dc/-sample ::homogeneous-mixture [{{:keys [family weights params]} :params} key shape]
  (let [[k-pick k-draw] (rng/split key)
        k (rng/categorical k-pick (log weights) (sh/broadcast shape (pop (array/shape weights))) (opts weights))]
    (dc/sample (apply family (mapv #(array/pick % k) params)) k-draw shape)))

(defmethod dc/-log-prob ::homogeneous-mixture [{{:keys [family weights params]} :params} v]
  ;; every component at once: the value against the component axis
  (array/logsumexp (+ (log weights) (dc/log-prob (apply family params) (array/expand-dims v -1))) {:axis -1}))

(defn heterogeneous-mixture
  "A family of mixtures of different distribution families (Gen.jl's
   HeterogeneousMixture): ((heterogeneous-mixture [normal uniform]) weights
   [mu std] [low high]) mixes (normal mu std) and (uniform low high).
   Components must have values of one shape."
  [families]
  (let [families (vec families)]
    (fn [weights & component-params]
      (let [w (dc/->array weights)]
        (dc/check (= (count families) (count component-params) (peek (array/shape w)))
                  "heterogeneous-mixture" "needs one weight and one parameter vector per component"
                  {:families (count families) :params (count component-params)})
        (dc/check-probabilities "heterogeneous-mixture" "weights" (dc/->checkable weights))
        (dc/->Distribution ::heterogeneous-mixture
                           {:weights w :components (mapv #(apply %1 %2) families component-params)})))))

(defmethod dc/-sample ::heterogeneous-mixture [{{:keys [weights components]} :params} key shape]
  (let [[k-pick k-draw] (rng/split key)
        k (rng/categorical k-pick (log weights) (sh/broadcast shape (pop (array/shape weights))) (opts weights))
        draws (mapv #(dc/sample %1 %2 shape) components (rng/split k-draw (count components)))
        dtype (reduce dt/promote (map array/dtype draws))]
    ;; draw every component, keep the picked one (branch-free, so it batches);
    ;; the draws share their promoted dtype, so no component is truncated
    (array/pick (array/stack (mapv #(array/astype % dtype) draws) -1) (array/broadcast-to k (array/shape (first draws))))))

(defmethod dc/-log-prob ::heterogeneous-mixture [{{:keys [weights components]} :params} v]
  (array/logsumexp (+ (log weights) (array/stack (mapv #(dc/log-prob % v) components) -1)) {:axis -1}))

(defn product
  "A family of products of independent distributions (Gen.jl's
   ProductDistribution): ((product normal bernoulli) [mu std] [p]) has values
   [x b], a vector with one value per factor (each of its factor's shape). The
   log density sums each factor's, over that factor's value."
  [& families]
  (let [families (vec families)]
    (fn [& factor-params]
      (dc/check (= (count families) (count factor-params)) "product" "needs one parameter vector per factor"
                {:factors (count families) :params (count factor-params)})
      (dc/->Distribution ::product {:factors (mapv #(apply %1 %2) families factor-params)}))))

(defmethod dc/-sample ::product [{{:keys [factors]} :params} key shape]
  (mapv #(dc/sample %1 %2 shape) factors (rng/split key (count factors))))

(defmethod dc/-log-prob ::product [{{:keys [factors]} :params} v]
  (dc/check (and (sequential? v) (= (count v) (count factors))) "product"
            (str "a value needs one element per factor (" (count factors) ")") {:value v})
  ;; each factor's log density summed over its own value's axes; a value of
  ;; another shape than its factor's is an error, not several draws
  (reduce + (map (fn [f x] (dc/check-value-shape f x) (array/sum (dc/log-prob f x))) factors v)))

;; ---------------------------------------------------------------------------
;; Value shapes: the elementwise families take their parameters' broadcast
;; shape; the others declare theirs
;; ---------------------------------------------------------------------------

(doseq [t [::normal ::uniform ::beta ::gamma ::inv-gamma ::exponential ::laplace ::cauchy ::beta-uniform
           ::chi-squared ::f-dist ::gumbel ::levy ::log-normal ::logistic ::nakagami ::pareto ::student-t
           ::triangular ::weibull
           ::bernoulli ::uniform-discrete ::geometric ::poisson ::binom ::neg-binom]]
  (defmethod dc/-value-shape t [d] (dc/elementwise-value-shape d)))

(doseq [t [::bernoulli ::uniform-discrete ::geometric ::poisson ::binom ::neg-binom ::categorical ::categorical-logits]]
  (defmethod dc/-discrete? t [_] true))

(defmethod dc/-discrete? ::homogeneous-mixture [{{:keys [family params]} :params}] (dc/discrete? (apply family params)))
(defmethod dc/-discrete? ::product [{{:keys [factors]} :params}] (every? dc/discrete? factors))
(defmethod dc/-discrete? ::heterogeneous-mixture [{{:keys [components]} :params}] (every? dc/discrete? components))

(defmethod dc/-value-shape ::dirichlet [{{:keys [alpha]} :params}] (array/shape alpha))

(defmethod dc/-value-shape ::mvnormal [{{:keys [mu cov]} :params}]
  (conj (sh/broadcast (pop (array/shape mu)) (pop (pop (array/shape cov)))) (peek (array/shape mu))))

(defmethod dc/-value-shape ::categorical [{{:keys [probs]} :params}] (pop (array/shape probs)))
(defmethod dc/-value-shape ::categorical-logits [{{:keys [logits]} :params}] (pop (array/shape logits)))

(defmethod dc/-value-shape ::piecewise-uniform [{{:keys [bounds probs]} :params}]
  (sh/broadcast (pop (array/shape bounds)) (pop (array/shape probs))))

(defmethod dc/-value-shape ::homogeneous-mixture [{{:keys [weights params]} :params}]
  (pop (reduce sh/broadcast (array/shape weights) (map array/shape params))))

(defmethod dc/-value-shape ::heterogeneous-mixture [{{:keys [components]} :params}]
  (dc/value-shape (first components)))

(defmethod dc/-value-shape ::product [_] nil)
