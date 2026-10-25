(ns gen.dist
  "Gen.jl's built-in distributions, with Gen.jl's names and parameterizations
   (normal by standard deviation, gamma by scale, categorical by
   probabilities), plus categorical-logits, and further continuous
   families with Distributions.jl's names and parameters (chi-squared,
   f-dist, gumbel, levy, log-normal, logistic, nakagami, pareto, student-t,
   triangular, weibull; truncated-normal, von-mises, wishart, inv-wishart),
   and the wrapped circular families with scipy's (wrapped-cauchy,
   wrapped-normal). Discrete values are 0-based integers; bernoulli values
   are booleans; circular values are angles in [-π, π).

   Every log-prob is -Inf outside the support. Every sampler is a pure,
   branch-free function of its key: rejection samplers run a fixed number of
   vectorized rounds rather than a data-dependent loop, so they batch and never
   force evaluation."
  {:api :stable}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= == log log1p exp sqrt square where
                                     lgamma log-beta xlogy xlog1py logical-and logical-or]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.array.shape :as sh]
            [gen.array.special :as special]
            [gen.dist.core :as dc :refer [defdist]]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def ^:private half-log-2pi 0.9189385332046728)

(defn- guard
  "lp inside the support, -Inf outside."
  [in-support lp]
  (where in-support lp ##-Inf))

(defn- stand-in
  "v where in-support holds, else s: the other lanes stay finite, so a
   gradient through the guard is not NaN."
  [in-support v s]
  (where in-support v s))

(defn- check-vector
  "A parameter whose last axis holds a family's values (probabilities,
   logits, concentrations) must have one: a scalar would fail later at an
   axis, naming neither the family nor the parameter."
  [dist-name param-name v]
  (dc/check (<= 1 (array/ndim v)) dist-name (str param-name " must be a vector (its last axis), got a scalar")
            {:param param-name :shape (array/shape v)}))

(defn- largest-float
  "The largest finite float of x's dtype."
  ^double [x]
  (:max (dc/float-limits (array/dtype x))))

(defn- floored
  "x at least the smallest normal number of its dtype. A gamma draw of a
   small shape (its boost U^(1/α)) or a power of one underflows to 0, where
   a density may be infinite or a quotient infinite; the floor changes only
   values below it."
  [x]
  (array/maximum x (:tiny (dc/float-limits (array/dtype x)))))

(defn- below-one
  "x at most the largest float below 1 of its dtype, floored's mirror: a
   beta draw within half an ulp of 1 rounds to 1, where its density may be
   infinite."
  [x]
  (array/minimum x (- 1.0 (* 0.5 (:eps (dc/float-limits (array/dtype x)))))))

(defn- log-gamma-draw
  "The log of a gamma(alpha, 1) draw, which does not underflow as the draw
   does: rng/gamma boosts a shape below 1 as Γ(α+1)·U^(1/α), and the power
   is 0 for most U at a small α, so here it is log Γ(α+1)'s draw - E/α, for
   E = -log U an exponential draw."
  [key alpha shape o]
  (let [boost? (< alpha 1)
        [kg ke] (rng/split key)
        g (rng/gamma kg (where boost? (+ alpha 1) alpha) shape o)]
    (- (log g) (where boost? (/ (rng/exponential ke shape (assoc o :dtype (array/dtype g))) alpha) 0.0))))

(defn- log-weight
  "A mixture weight's log, -Inf at 0 through a stand-in: log's gradient
   there would be 0/0 = NaN, though the mixture sends it nothing."
  [weighted w]
  (guard weighted (log (stand-in weighted w 1.0))))

(defn- log-mass
  "log p for a mass p: -Inf with gradient 0 at 0, through a stand-in, as
   outside a support. log's gradient there would be infinite, and NaN where
   a pick, a where or a mixture discards the -Inf (DESIGN §9)."
  [p]
  (log-weight (> p 0) p))

(defn- log1m-mass
  "log(1 - p), as log1p(-p), for the mass 1 - p: -Inf with gradient 0 at
   p = 1, as log-mass."
  [p]
  (let [below-one (< p 1)]
    (guard below-one (log1p (- (stand-in below-one p 0.0))))))

(defn- zero-weight-term
  "A term of value 0 for a mixture of log density l, carrying the gradient
   of a weight w of 0 (weighted false), which log-weight does not: w times
   ∂l/∂w = e^(lp - l) for its component's log density lp, held constant. So
   a weight that is a function at its bound (beta-uniform's theta at 1) has
   its one-sided derivative. The slope is at most the largest float: it is
   infinite where the component's density is beyond the float range times
   the mixture's."
  [weighted w lp l]
  (let [slope (array/minimum (exp (- lp l)) (largest-float l))]
    (where weighted 0.0 (* w (array/stop-gradient (where (array/isnan slope) 0.0 slope))))))

(defn- log-mixture
  "log Σₖ wₖ e^lpₖ over the last axis, exact in its gradients also at weights
   of 0 (zero-weight-term)."
  [w lps]
  (let [weighted (> w 0)
        kd {:axis -1 :keep-dims? true}
        l (array/logsumexp (+ (log-weight weighted w) lps) kd)]
    (array/squeeze (+ l (array/sum (zero-weight-term weighted w lps l) kd)) -1)))

(defn- integral? [v] (== v (array/floor v)))

(defn- opts [& params] {:backend (apply dc/backend-of params)})

(defn- to-int [x & params]
  (array/astype x (dc/int-dtype (apply dc/backend-of x params))))

;; ---------------------------------------------------------------------------
;; Continuous distributions
;; ---------------------------------------------------------------------------

(defdist ^:elementwise ^:pathwise normal
  "normal(mu, std): the Gaussian with mean mu and standard deviation std.
   Array parameters give an array value whose log density is the sum over its
   elements (Gen.jl's broadcasted_normal)."
  [mu std]
  (validate (dc/check-positive "normal" "std" std))
  (sample [key shape]
    (+ mu (* std (rng/normal key (dc/elementwise-shape shape mu std) (opts mu std)))))
  (log-prob [v]
    (- (+ half-log-2pi (log std) (* 0.5 (square (/ (- v mu) std)))))))

(def ^{::dc/constructor true} broadcasted-normal
  "broadcasted_normal(mu, std): normal with broadcast array parameters (in
   gen.clj, normal itself already broadcasts). A constructor to the dist DSL
   too, as defdist marks its own."
  normal)

(defdist ^:elementwise ^:pathwise uniform
  "uniform(low, high): uniform on [low, high]."
  [low high]
  (validate (when (every? array/concrete? [low high])
              (dc/check (array/all-true? (< low high)) "uniform" "low must be below high" {:low low :high high})))
  (sample [key shape]
    (rng/uniform key (dc/elementwise-shape shape low high) (assoc (opts low high) :low low :high high)))
  (log-prob [v]
    (guard (logical-and (>= v low) (<= v high)) (- (log (- high low))))))

(defdist ^:elementwise beta
  "beta(alpha, beta): the beta distribution on [0, 1]."
  [a b]
  (validate (dc/check-positive "beta" "alpha" a) (dc/check-positive "beta" "beta" b))
  (sample [key shape]
    ;; x/(x + y) for gamma draws x and y, from their logs: either may
    ;; underflow at a small shape (0/0 when both do), and the quotient rounds
    ;; to 1 when y is below x·2^-53
    (let [s (dc/elementwise-shape shape a b)
          [k1 k2] (rng/split key)]
      (below-one (floored (array/sigmoid (- (log-gamma-draw k1 a s (opts a b)) (log-gamma-draw k2 b s (opts a b))))))))
  (log-prob [v]
    (guard (logical-and (>= v 0) (<= v 1))
           (- (+ (xlogy (- a 1) v) (xlog1py (- b 1) (- v))) (log-beta a b)))))

(defdist ^:elementwise gamma
  "gamma(shape, scale): the gamma distribution with mean shape·scale."
  [k theta]
  (validate (dc/check-positive "gamma" "shape" k) (dc/check-positive "gamma" "scale" theta))
  (sample [key shape]
    (floored (* theta (rng/gamma key k (dc/elementwise-shape shape k theta) (opts k theta)))))
  (log-prob [v]
    (guard (>= v 0) (- (xlogy (- k 1) v) (/ v theta) (lgamma k) (* k (log theta))))))

(defdist ^:elementwise inv-gamma
  "inv_gamma(shape, scale): the inverse gamma distribution (1/X for X with
   gamma(shape, 1/scale))."
  [k theta]
  (validate (dc/check-positive "inv-gamma" "shape" k) (dc/check-positive "inv-gamma" "scale" theta))
  (sample [key shape]
    (/ theta (rng/gamma key k (dc/elementwise-shape shape k theta) (opts k theta))))
  (log-prob [v]
    (guard (> v 0) (- (* k (log theta)) (lgamma k) (* (+ k 1) (log v)) (/ theta v)))))

(defdist ^:elementwise ^:pathwise exponential
  "exponential(rate): the exponential distribution with mean 1/rate."
  [rate]
  (validate (dc/check-positive "exponential" "rate" rate))
  (sample [key shape]
    (/ (rng/exponential key (dc/elementwise-shape shape rate) (opts rate)) rate))
  (log-prob [v]
    (guard (>= v 0) (- (log rate) (* rate v)))))

(defn- above-minus-one
  "The float of dtype next above -1: a uniform's low bound for laplace's
   log1p(-|u|), which is -Inf at -1 (as rng/normal's bound for erfinv's
   pole)."
  ^double [dtype]
  (+ -1.0 (* 0.5 (:eps (dc/float-limits dtype)))))

(defdist ^:elementwise ^:pathwise laplace
  "laplace(loc, scale): the Laplace (double exponential) distribution."
  [loc scale]
  (validate (dc/check-positive "laplace" "scale" scale))
  (sample [key shape]
    (let [o (opts loc scale)
          dtype (dc/float-dtype (:backend o))
          u (rng/uniform key (dc/elementwise-shape shape loc scale)
                         (assoc o :dtype dtype :low (above-minus-one dtype) :high 1.0))]
      (+ loc (* scale (array/sign u) (log1p (- (array/abs u)))))))
  (log-prob [v]
    (- (+ (log (* 2 scale)) (/ (array/abs (- v loc)) scale)))))

(defdist ^:elementwise ^:pathwise cauchy
  "cauchy(x0, gamma): the Cauchy distribution with location x0 and scale gamma."
  [x0 scale]
  (validate (dc/check-positive "cauchy" "scale" scale))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape x0 scale)
                         (assoc (opts x0 scale) :low (Math/ulp 1.0) :high 1.0))]
      (+ x0 (* scale (array/tan (* Math/PI (- u 0.5)))))))
  (log-prob [v]
    (- (+ (log (* Math/PI scale)) (log1p (square (/ (- v x0) scale)))))))

(defdist ^:elementwise beta-uniform
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
    ;; beta's density and the uniform's, 1, mixed as log-mixture does
    (let [lp (dc/log-prob (beta a b) v)
          w-uniform (- 1 theta)
          some-beta (> theta 0)
          some-uniform (> w-uniform 0)
          l (array/logaddexp (+ (log-weight some-beta theta) lp) (log-weight some-uniform w-uniform))]
      (guard (logical-and (>= v 0) (<= v 1))
             (+ l (zero-weight-term some-beta theta lp l) (zero-weight-term some-uniform w-uniform 0.0 l))))))

(defdist piecewise-uniform
  "piecewise_uniform(bounds, probs): bin i, chosen with probability probs[i],
   is [bounds[i], bounds[i+1]], uniform within the bin."
  [bounds probs]
  (validate (dc/check (= 1 (array/ndim bounds) (array/ndim probs)) "piecewise-uniform"
                      (str "bounds and probs must be vectors, got shapes " (array/shape bounds) " and " (array/shape probs)
                           " (no batch of bin layouts yet)")
                      {:bounds (array/shape bounds) :probs (array/shape probs)})
            (dc/check-probabilities "piecewise-uniform" "probs" probs)
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
             (- (log-mass (array/take probs i)) (log (- (array/take bounds (+ i 1)) (array/take bounds i)))))))
  ;; one value: bounds and probs are vectors (validate)
  (value-shape [] []))

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
   is 4e-16, not 0). So the correlation matrix H of L·Lᵀ must also be
   conditioned for the dtype: trace(H⁻¹), between 1/λmin and n/λmin, below
   1/ε. The bound is the measured one: a float32 log density's error tracks
   (0.05–0.4)·ε·trace(H⁻¹) on squared-exponential kernels
   (docs/research/covariance/f32_gp.py), so it rejects where the error nears
   0.1 nats; the worst case's 1/(n·ε) was n times too strict."
  [a dtype]
  (let [n (peek (array/shape a))
        eye (array/with-backend (array/backend a) (array/eye n))
        ;; H⁻¹ = MᵀM for M = L⁻¹·diag(√aⱼⱼ); a NaN factor fails the comparison
        m (la/solve-triangular (la/cholesky a) (* eye (array/expand-dims (sqrt (la/diagonal a)) -2)))]
    (array/all-true? (< (* (machine-epsilon dtype) (array/sum (square m) {:axis [-2 -1]})) 1.0))))

(defn- host-max
  "The largest element of x on the host, across every particle of a batched
   value."
  ^double [x]
  (double (array/item (array/max (array/all-particles x)))))

(defn- rung
  "The i-th readable jitter: 1, 2 or 5 times 10^⌊i/3⌋."
  ^double [^long i]
  (Double/parseDouble (str ([1 2 5] (mod i 3)) "e" (Math/floorDiv i 3))))

(defn- suggested-jitter
  "The smallest readable jitter (1, 2 or 5 times a power of ten) that lets
   positive-definite? pass once added to a's diagonal. The search starts at
   the rung that passes for every positive semidefinite a: a jitter δ above
   n·ε·max aⱼⱼ/(1 - n·ε) makes λmin(H) at least δ/(max aⱼⱼ + δ), so
   trace(H⁻¹) ≤ n/λmin(H) is below 1/ε. It goes up from there (to max aⱼⱼ)
   while rounding leaves a short of passing, then down while smaller rungs
   pass (to ε·max aⱼⱼ's decade, below which a jitter is lost to rounding).
   nil when a's factor fails (no jitter at the dtype's scale mends an
   indefinite matrix) or none passes."
  [a dtype]
  (when-not (array/any-true? (array/isnan (la/cholesky a)))
    (let [n (peek (array/shape a))
          eps (machine-epsilon dtype)
          top (host-max (la/diagonal a))
          eye (array/with-backend (array/backend a) (array/eye n {:dtype (array/dtype a)}))
          passes? #(positive-definite? (+ a (* (rung %) eye)) dtype)
          lowest (* 3 (long (Math/floor (Math/log10 (* eps top)))))
          guarantee (/ (* n eps top) (- 1 (* n eps)))
          start (first (drop-while #(< (rung %) (min guarantee top)) (iterate inc lowest)))]
      (when-let [i (first (filter passes? (take-while #(<= (rung %) top) (iterate inc start))))]
        (rung (last (take-while passes? (range i (dec lowest) -1))))))))

(defn- transposed
  "m with its last two axes swapped."
  [m]
  (let [r (array/ndim m)]
    (array/transpose m (-> (vec (range r)) (assoc (- r 2) (dec r)) (assoc (dec r) (- r 2))))))

(defn- check-covariance
  "A concrete m (a covariance or a scale matrix, batched over leading axes)
   must be symmetric and positive definite, also to the precision it is
   factored in; when its factor exists but is too ill-conditioned, the error
   suggests a jitter for its diagonal."
  [dist-name param-name m]
  (when (array/concrete? m)
    (dc/check (array/all-true? (<= (array/abs (- m (transposed m))) (+ 1e-8 (* 1e-5 (array/abs (transposed m))))))
              dist-name (str param-name " must be symmetric") {:param param-name})
    (let [dtype (factored-dtype m)]
      (when-not (positive-definite? m dtype)
        ;; the suggestion reads m's diagonal on the host, which has no
        ;; gradient: an m being differentiated is read on purpose
        (let [jitter (suggested-jitter (array/stop-gradient m) dtype)]
          (throw (ex-info (cond-> (str dist-name ": " param-name " must be positive definite, and not singular at "
                                       (name dtype) " precision")
                            jitter (str "; adding a jitter of " jitter " to its diagonal passes"))
                          (cond-> {:distribution dist-name :param param-name :dtype dtype}
                            jitter (assoc :suggested-jitter jitter)))))))))

(defdist mvnormal
  "mvnormal(mu, cov): the multivariate normal with mean vector mu and
   covariance matrix cov (batched over leading axes). A concrete cov must be
   symmetric and positive definite, also to the precision it is factored in;
   when its factor exists but is too ill-conditioned, the error suggests a
   jitter for its diagonal."
  [mu cov]
  (validate (dc/check (and (<= 1 (array/ndim mu)) (<= 2 (array/ndim cov))
                           (= (peek (array/shape mu)) (peek (array/shape cov)) (peek (pop (array/shape cov)))))
                      "mvnormal" (str "mu must be a vector and cov a square matrix of its size, got shapes "
                                      (array/shape mu) " and " (array/shape cov))
                      {:mu-shape (array/shape mu) :cov-shape (array/shape cov)})
            (check-covariance "mvnormal" "cov" cov))
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
         (* d half-log-2pi))))
  ;; a vector, batched over mu's and cov's leading axes
  (value-shape [] (conj (sh/broadcast (pop (array/shape mu)) (pop (pop (array/shape cov)))) (peek (array/shape mu)))))

(defdist dirichlet
  "dirichlet(alpha): the Dirichlet distribution over the simplex of the last
   axis of alpha."
  [alpha]
  (validate (check-vector "dirichlet" "alpha" alpha)
            (dc/check-positive "dirichlet" "alpha" alpha))
  (sample [key shape]
    ;; g/Σg for gamma draws g, from their logs (a draw underflows to 0 at a
    ;; small alpha), floored where the normalized value still does
    (let [k (peek (array/shape alpha))
          lg (log-gamma-draw key alpha (sh/broadcast (conj shape k) (array/shape alpha)) (opts alpha))]
      (floored (array/softmax lg {:axis -1}))))
  (log-prob [v]
    (guard (logical-and (array/all (>= v 0) {:axis -1})
                        (<= (array/abs (- (array/sum v {:axis -1}) 1))
                            (dc/probability-tolerance (peek (array/shape v)) (array/dtype v))))
           (- (array/sum (xlogy (- alpha 1) v) {:axis -1})
              (- (array/sum (lgamma alpha) {:axis -1}) (lgamma (array/sum alpha {:axis -1}))))))
  (value-shape [] (array/shape alpha)))

;; ---------------------------------------------------------------------------
;; Beyond Gen.jl's built-ins: Distributions.jl's families with its names and
;; parameters (what GenDistributions' users write), native, so they batch,
;; run on every backend and differentiate (gen.dist.commons-math wraps a
;; host library's for the rest)
;; ---------------------------------------------------------------------------

(def ^:private log-2 (Math/log 2.0))

(defdist ^:elementwise chi-squared
  "chi-squared(nu): the χ² distribution with nu degrees of freedom (Chisq)."
  [nu]
  (validate (dc/check-positive "chi-squared" "nu" nu))
  (sample [key shape] (* 2.0 (floored (rng/gamma key (/ nu 2.0) (dc/elementwise-shape shape nu) (opts nu)))))
  (log-prob [v]
    (let [k (/ nu 2.0)]
      (guard (>= v 0) (- (xlogy (- k 1) v) (/ v 2.0) (lgamma k) (* k log-2))))))

(defdist ^:elementwise f-dist
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

(defdist ^:elementwise ^:pathwise gumbel
  "gumbel(mu, theta): the Gumbel (maximum) distribution, location mu and
   scale theta (Gumbel)."
  [mu theta]
  (validate (dc/check-positive "gumbel" "theta" theta))
  (sample [key shape] (+ mu (* theta (rng/gumbel key (dc/elementwise-shape shape mu theta) (opts mu theta)))))
  (log-prob [v]
    (let [z (/ (- v mu) theta)]
      (- (+ (log theta) z (exp (- z)))))))

(defdist ^:elementwise levy
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

(defdist ^:elementwise ^:pathwise log-normal
  "log-normal(mu, sigma): e^X for X normal(mu, sigma) (LogNormal)."
  [mu sigma]
  (validate (dc/check-positive "log-normal" "sigma" sigma))
  (sample [key shape]
    (exp (+ mu (* sigma (rng/normal key (dc/elementwise-shape shape mu sigma) (opts mu sigma))))))
  (log-prob [v]
    (let [in (> v 0)
          lx (log (stand-in in v 1.0))]
      (guard in (- (+ lx (log sigma) half-log-2pi (* 0.5 (square (/ (- lx mu) sigma)))))))))

(defdist ^:elementwise ^:pathwise logistic
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

(defdist ^:elementwise nakagami
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

(defdist ^:elementwise pareto
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

(defdist ^:elementwise student-t
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

(defdist ^:elementwise triangular
  "triangular(a, b, c): the triangular distribution on [a, b] with mode c
   (TriangularDist)."
  [a b c]
  (validate (when (every? array/concrete? [a b c])
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

(defdist ^:elementwise weibull
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
;; GenMLX's further families: the truncated normal, three on the circle and
;; two over matrices, with Distributions.jl's names and parameters where it
;; has them (truncated(Normal), VonMises, Wishart, InverseWishart), and
;; scipy's, centered on mu, for the wrapped families
;; ---------------------------------------------------------------------------

(def ^:private sqrt-2 (Math/sqrt 2.0))
(def ^:private two-pi (* 2.0 Math/PI))
(def ^:private log-2pi (Math/log two-pi))

;; --- The truncated normal -------------------------------------------------

(defn- log-erfc
  "log erfc(z) for finite z >= 0: Numerical Recipes' Chebyshev fit
   t·e^(-z² + P(t)) (gen.array.special's erfc on the host), its log taken
   term by term, so it never underflows (log erfc(40) is -1604.3)."
  [z]
  (let [t (/ 2.0 (+ 2.0 z))
        ty (- (* 4.0 t) 2.0)
        [c0 & cs] special/erfc-coefficients
        ;; Clenshaw's recurrence, from the last coefficient down
        [d dd] (reduce (fn [[d dd] c] [(+ (- (* ty d) dd) c) d]) [0.0 0.0] (reverse cs))]
    (- (+ (log t) (* 0.5 (+ c0 (* ty d)))) (square z) dd)))

(defn- log-ndtr
  "log Φ(x) for x <= 0, Φ the standard normal's CDF, -Inf at -Inf:
   log erfc(-x/√2) - log 2, exact far into the tail."
  [x]
  (let [finite (> x ##-Inf)]
    (guard finite (- (log-erfc (/ (- (stand-in finite x 0.0)) sqrt-2)) log-2))))

(defn- log1mexp
  "log(1 - e^d) for d <= 0, exact at both ends: log(-expm1(d)) above
   -log 2, log1p(-e^d) below (Mächler), each on a stand-in where the other
   holds. -Inf at 0."
  [d]
  (let [near (> d (- log-2))]
    (where near
           (log (- (array/expm1 (stand-in near d (- log-2)))))
           (log1p (- (exp (where near (- log-2) d)))))))

(defn- standardized
  "(x - mu)/sigma for a bound x that may be infinite: an infinite x is its
   own (sigma > 0), and the quotient is taken of a stand-in, so a gradient of
   mu or sigma does not meet ∞·0."
  [x mu sigma]
  (let [finite (array/isfinite x)]
    (where finite (/ (- (stand-in finite x 0.0) mu) sigma) x)))

(defn- normal-interval
  "The log of the standard normal's mass on [a, b] (a < b), and the logs its
   inverse CDF reads, of the interval mirrored when a >= 0, so that its left
   end is below 0. In a tail (b < 0 too) the mass is Φ(b)·(1 - Φ(a)/Φ(b)),
   from log Φ(a) and log Φ(b), which stay exact far into the tail (scipy's
   truncnorm's); across 0 it is (erf(b/√2) - erf(a/√2))/2, a sum of two
   magnitudes, which never cancels (scipy's 1 - Φ(a) - Φ(-b) does for a
   narrow interval). A narrow interval in a tail is as exact as the
   difference of its logs, a relative error of about ε·|a|/(b - a), as
   scipy's. {:flip? :tail? :la :lb :log-mass}, of the mirrored interval: :la
   is log Φ(a), :lb log Φ(b) in a tail and log Φ(-b) across 0."
  [a b]
  (let [flip? (>= a 0)
        a' (where flip? (- b) a)
        b' (where flip? (- a) b)
        tail? (< b' 0)
        la (log-ndtr a')
        lb (log-ndtr (where tail? b' (- b')))]
    {:flip? flip? :tail? tail? :la la :lb lb
     ;; each form on a stand-in interval of its kind where the other holds
     :log-mass (where tail?
                      (+ lb (log1mexp (where tail? (- la lb) -1.0)))
                      (log (* 0.5 (- (array/erf (/ (where tail? 1.0 b') sqrt-2))
                                     (array/erf (/ (where tail? -1.0 a') sqrt-2))))))}))

(def ^:private newton-steps
  "Newton steps of ndtri-left: from its first guesses, two reach 1e-12 and
   three float64's precision, measured on y in [-1e6, log 1/2] against
   scipy's ndtri_exp."
  3)

(defn- ndtri-left
  "The x <= 0 with log Φ(x) = y, for y <= log 1/2 (scipy's ndtri_exp on that
   half): Newton steps on log Φ, whose slope φ/Φ is at least 0.79 there,
   from a first guess by erfinv above y = -10, and by the tail's asymptote
   x² + log(2πx²) = -2y below (where 2e^y - 1 rounds to -1 in float32)."
  [y]
  (let [s (array/maximum (- (* -2.0 y) log-2pi) 2.0)
        guess (where (< y -10.0)
                     (- (sqrt (- s (log s))))
                     (* sqrt-2 (array/erfinv (- (* 2.0 (exp (array/maximum y -10.0))) 1.0))))
        step (fn [x]
               (let [lp (log-ndtr (array/minimum x 0.0))]
                 (- x (* (- lp y) (exp (+ lp half-log-2pi (* 0.5 (square x))))))))]
    (nth (iterate step guess) newton-steps)))

(defn- standard-truncated-normal
  "Draws of the standard normal on [a, b], of shape: its inverse CDF in log
   space (scipy's truncnorm's), Φ(x) = Φ(a) + u·Z solved where it is below
   1/2, and Φ(-x) = Φ(-b) + (1 - u)·Z where it is above, so a draw near
   either end of a wide interval keeps its precision."
  [key a b shape o]
  (let [{:keys [flip? tail? la lb log-mass]} (normal-interval a b)
        dtype (dc/float-dtype (:backend o))
        u (rng/uniform key shape (assoc o :dtype dtype :low (:tiny (dc/float-limits dtype)) :high 1.0))
        below (array/logaddexp la (+ (log u) log-mass))
        above (array/logaddexp lb (+ (log1p (- u)) log-mass))
        from-above (logical-and (array/logical-not tail?) (> below (- log-2)))
        x (ndtri-left (where from-above above below))
        x (where from-above (- x) x)]
    (where flip? (- x) x)))

(defdist ^:elementwise truncated-normal
  "truncated-normal(mu, sigma, lo, hi): normal(mu, sigma) conditioned on
   [lo, hi] (Distributions.jl's truncated(Normal(mu, sigma), lo, hi)); lo
   must be below hi, and either may be infinite. Its density and its draws
   work in log Φ, so they stay exact far into either tail (on
   [μ + 50σ, μ + 51σ], say): a draw inverts the CDF by Newton steps."
  [mu sigma lo hi]
  (validate (dc/check-positive "truncated-normal" "sigma" sigma)
            (when (every? array/concrete? [lo hi])
              (dc/check (array/all-true? (< lo hi)) "truncated-normal" "lo must be below hi" {:lo lo :hi hi})))
  (sample [key shape]
    (let [z (standard-truncated-normal key (standardized lo mu sigma) (standardized hi mu sigma)
                                       (dc/elementwise-shape shape mu sigma lo hi) (opts mu sigma lo hi))]
      ;; within [lo, hi] also where mu + sigma·z rounds past a bound
      (array/clip (+ mu (* sigma z)) lo hi)))
  (log-prob [v]
    (let [{:keys [log-mass]} (normal-interval (standardized lo mu sigma) (standardized hi mu sigma))]
      (guard (logical-and (>= v lo) (<= v hi))
             (- (+ half-log-2pi (log sigma) (* 0.5 (square (/ (- v mu) sigma))) log-mass))))))

;; --- Circular families ----------------------------------------------------

(defn- wrap-angle
  "x as the angle in [-π, π) of the same point on the circle. Where
   rounding leaves it outside (π, which is -π's point, or anywhere for an x
   beyond 2^53·π in float64, whose point is lost), it is -π: a where, not a
   clamp, whose tie at -π would halve a gradient there."
  [x]
  (let [w (- x (* two-pi (array/floor (/ (+ x Math/PI) two-pi))))]
    (where (logical-and (>= w (- Math/PI)) (< w Math/PI)) w (- Math/PI))))

(defn- on-circle?
  "Is v an angle in [-π, π], the circular families' support? (π is -π's
   point, and a wrapped draw can round to it.)"
  [v]
  (logical-and (>= v (- Math/PI)) (<= v Math/PI)))

(def ^:private i0-series
  "The ratios of successive terms of I₀(x)'s power series Σ(x²/4)^k/k!²
   over x²/4, 1/k², last term first: 32 terms, enough to x = 20."
  (mapv #(/ 1.0 (* % %)) (range 32 0 -1)))

(def ^:private i0-asymptotic
  "The ratios of successive terms of I₀(x)'s asymptotic series
   e^x/√(2πx)·Σ((2k-1)!!)²/(k!·(8x)^k) over 1/x, (2k-1)²/8k, last term
   first: 19 terms, enough from x = 20."
  (mapv #(/ (square (- (* 2.0 %) 1.0)) (* 8.0 %)) (range 19 0 -1)))

(defn- nested
  "1 + r₁x(1 + r₂x(1 + …)) for ratios [… r₂ r₁]: a series of positive terms
   from the ratios of its successive terms, so no term over- or underflows."
  [ratios x]
  (reduce (fn [acc r] (+ 1.0 (* (* r x) acc))) 1.0 ratios))

(defn- log-i0e
  "log(I₀(x)·e^-x) for x >= 0, I₀ the modified Bessel function of order 0:
   its power series up to x = 20, its asymptotic series beyond, each on a
   stand-in where the other holds. Within 3e-15 of the log of scipy's i0e on
   [1e-8, 1e4] (the series' log, near x, keeps x's rounding)."
  [x]
  (let [series? (<= x 20.0)
        xs (array/minimum x 20.0)
        xa (array/maximum x 20.0)]
    (where series?
           (- (log (nested i0-series (* 0.25 (square xs)))) xs)
           (- (log (nested i0-asymptotic (/ 1.0 xa))) (* 0.5 (log (* two-pi xa)))))))

(def ^:private von-mises-rounds
  "Proposal rounds per von Mises draw. Best and Fisher's envelope accepts
   with probability above 0.656 for every kappa (its limit as kappa grows),
   so 33 rounds all fail with probability below 1e-15 (and the draw then
   falls back to mu)."
  33)

(defn- first-accepted-round
  "The value of the first accepting round, else fallback, of ok and values
   with the rounds on axis 0: a rejection sampler's fixed rounds drawn as
   one vectorized proposal, so a draw costs a few array ops whatever their
   number, and batches (no value is read)."
  [fallback ok values]
  (let [rounds (first (array/shape ok))
        ;; the first accepting round scores highest, rounds - r
        score (where ok (array/reshape (array/arange rounds 0 -1) (into [rounds] (repeat (dec (array/ndim ok)) 1))) 0)
        first-ok (array/argmax score {:axis 0 :keep-dims? true})]
    (where (array/any ok {:axis 0}) (array/squeeze (array/take-along-axis values first-ok 0) 0) fallback)))

(defn- von-mises-angle
  "Draws of the von Mises distribution centered on 0, of shape: Best and
   Fisher's rejection from a wrapped Cauchy envelope, written in quantities
   that stay exact as kappa grows (1 - ρ and κ(r - 1), never r, which
   rounds to 1, nor acos near 1), so the angle keeps its scale 1/√κ at any
   kappa; at kappa near 0 it is the uniform."
  [key kappa shape o]
  (let [;; at most the largest float over 4, so 2κ stays finite
        k (array/minimum kappa (/ (largest-float kappa) 4.0))
        two-k (* 2.0 k)
        ;; w = √(1 + 4κ²) without overflow, τ = 1 + w, s = √(2τ)
        m (array/maximum two-k 1.0)
        w (* m (sqrt (+ (square (/ 1.0 m)) (square (/ two-k m)))))
        tau (+ 1.0 w)
        s (sqrt (* 2.0 tau))
        ;; τ - 2κ + s, with τ - 2κ = 1 + 1/(w + 2κ): no cancellation
        num (+ 1.0 (/ 1.0 (+ w two-k)) s)
        ;; the envelope's (1 - ρ)/(1 + ρ), and κ(r - 1)
        g (/ num (+ tau two-k s))
        kr (square (/ num (* 2.0 (sqrt (+ tau s)))))
        rs (into [von-mises-rounds] shape)
        [k-angle k-accept] (rng/split key)
        ;; the envelope's draw, by its inverse CDF
        theta (* 2.0 (array/atan (* g (array/tan (* Math/PI (- (rng/uniform k-angle rs o) 0.5))))))
        ;; c = κ(r - cos θ) = κ(r - 1) + 2κ·sin²(θ/2), accepted when c·e^(1-c) >= v
        c (+ kr (* k (* 2.0 (square (array/sin (* 0.5 theta))))))
        ok (<= (log (rng/uniform k-accept rs o)) (+ (log c) (- 1.0 c)))]
    (first-accepted-round 0.0 ok theta)))

(defdist ^:elementwise von-mises
  "von-mises(mu, kappa): the von Mises distribution on the circle, mean
   direction mu and concentration kappa > 0 (VonMises), density
   e^(κ·cos(x - μ))/(2π·I₀(κ)), values in [-π, π) (Distributions.jl's are
   in [μ - π, μ + π]: the same density, on another period). Exact at any
   kappa, its draws by Best and Fisher's rejection."
  [mu kappa]
  (validate (dc/check-positive "von-mises" "kappa" kappa))
  (sample [key shape]
    (wrap-angle (+ mu (von-mises-angle key kappa (dc/elementwise-shape shape mu kappa) (opts mu kappa)))))
  (log-prob [v]
    ;; κ(cos d - 1) as -2κ·sin²(d/2), exact at a large kappa
    (guard (on-circle? v)
           (- (* -2.0 kappa (square (array/sin (* 0.5 (- v mu))))) log-2pi (log-i0e kappa)))))

(defdist ^:elementwise wrapped-cauchy
  "wrapped-cauchy(mu, rho): cauchy(mu, -log rho) wrapped onto the circle
   (scipy's wrapcauchy, centered on mu), density
   (1 - ρ²)/(2π·(1 + ρ² - 2ρ·cos(x - μ))), values in [-π, π); 0 <= rho < 1,
   and rho = 0 is the uniform on the circle."
  [mu rho]
  (validate (dc/check-probability "wrapped-cauchy" "rho" rho)
            (when (array/concrete? rho)
              (dc/check (array/all-true? (< rho 1)) "wrapped-cauchy" "rho must be below 1 (1 is a point mass)"
                        {:param "rho"})))
  (sample [key shape]
    ;; the wrapped draw by its own inverse CDF, 2·atan(tanh(γ/2)·tan(π(u - 1/2)))
    ;; for the Cauchy's scale γ, where (1 - ρ)/(1 + ρ) = tanh(γ/2): no draw
    ;; far out on the line, whose wrap would lose its precision
    (let [u (rng/uniform key (dc/elementwise-shape shape mu rho) (opts mu rho))]
      (wrap-angle (+ mu (* 2.0 (array/atan (* (/ (- 1.0 rho) (+ 1.0 rho)) (array/tan (* Math/PI (- u 0.5))))))))))
  (log-prob [v]
    ;; 1 + ρ² - 2ρ·cos d as (1 - ρ)² + 4ρ·sin²(d/2), exact as rho nears 1
    (guard (on-circle? v)
           (- (+ (log1p (- rho)) (log1p rho))
              log-2pi
              (log (+ (square (- 1.0 rho)) (* 4.0 rho (square (array/sin (* 0.5 (- v mu)))))))))))

(def ^:private wrapped-normal-switch
  "The sigma at which wrapped-normal's log density changes series. Below
   it, the sum of N(d + 2πk; 0, σ) over |k| <= 3 omits less than
   2e^(-24π²/σ²) of the total; above, the Fourier series
   (1 + 2Σ e^(-n²σ²/2)·cos(nd))/2π to n = 3 omits about 2e^(-8σ²). At 2.25
   both are below 1e-17, and smaller away from it."
  2.25)

(defn- log-wrapped-normal
  "log Σₖ N(d + 2πk; 0, sigma) for d in [-π, π]: the direct sum for a small
   sigma, the Fourier series for a large one (wrapped-normal-switch), each
   on a stand-in where the other holds; sigma floored at the smallest
   normal float, so 1/σ stays finite."
  [d sigma]
  (let [direct? (<= sigma wrapped-normal-switch)
        sd (floored (array/minimum sigma wrapped-normal-switch))
        sf (array/maximum sigma wrapped-normal-switch)
        ;; the terms over the nearest (k = 0): e^(-2πk(d + πk)/σ²) <= 1
        others (reduce + (for [k [-3 -2 -1 1 2 3]]
                           (exp (* (/ (* -2.0 Math/PI k) sd) (/ (+ d (* Math/PI k)) sd)))))
        fourier (reduce + (for [n [1 2 3]] (* (exp (* -0.5 n n (square sf))) (array/cos (* n d)))))]
    (where direct?
           (- (log1p others) (+ half-log-2pi (log sd) (* 0.5 (square (/ d sd)))))
           (- (log1p (* 2.0 fourier)) log-2pi))))

(def ^:private wrapped-normal-uniform
  "The sigma from which wrapped-normal draws the uniform on the circle: its
   density differs from the uniform's by 2e^(-σ²/2) at most, below 6e-18
   from 9 on, and a wrap of σ·z would lose more to rounding (all of the
   angle beyond σ = 1e16)."
  9.0)

(defdist ^:elementwise wrapped-normal
  "wrapped-normal(mu, sigma): normal(mu, sigma) wrapped onto the circle,
   density Σₖ N(x + 2πk; μ, σ), values in [-π, π). The sum is two series,
   each truncated where its omitted terms are below 1e-17 of the total.
   From sigma = 9 on, its draws are the uniform's, which it then is to
   float64's precision."
  [mu sigma]
  (validate (dc/check-positive "wrapped-normal" "sigma" sigma))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape mu sigma)
          o (opts mu sigma)
          [k-normal k-uniform] (rng/split key)]
      (where (< sigma wrapped-normal-uniform)
             (wrap-angle (+ mu (* sigma (rng/normal k-normal s o))))
             (rng/uniform k-uniform s (assoc o :low (- Math/PI) :high Math/PI)))))
  (log-prob [v]
    (guard (on-circle? v) (log-wrapped-normal (wrap-angle (- v mu)) sigma))))

;; --- Wishart and inverse Wishart ------------------------------------------

(def ^:private log-pi (Math/log Math/PI))

(defn- check-matrix-family
  "A p×p matrix family's scale must be square matrices, symmetric and
   positive definite (as mvnormal's covariance), and df above p - 1."
  [dist-name df scale]
  (let [s (array/shape scale)
        p (peek s)]
    (dc/check (and (<= 2 (count s)) (= p (peek (pop s)))) dist-name
              (str "scale must be a square matrix (its last two axes), got shape " s) {:param "scale" :shape s})
    (check-covariance dist-name "scale" scale)
    (when (array/concrete? df)
      (dc/check (array/all-true? (> df (dec p))) dist-name (str "df must be above the dimension minus one, " (dec p))
                {:param "df" :dimension p}))))

(defn- matrix-batch
  "The batch axes of a matrix family's draws: shape's, df's and scale's
   leading ones, broadcast."
  [shape df scale]
  (reduce sh/broadcast (vec shape) [(array/shape df) (pop (pop (array/shape scale)))]))

(defn- log-multivariate-gamma
  "log Γ_p(a) = p(p-1)/4·log π + Σⱼ log Γ(a - j/2), j = 0 … p - 1."
  [a p]
  ;; term by term: a host df's are host lgammas, about 20 ns each
  (reduce + (* 0.25 p (dec p) log-pi) (map #(lgamma (- a (* 0.5 %))) (range p))))

(defn- bartlett
  "Bartlett's factor of wishart(df, I) of dimension p, of batch shape: the
   lower triangular A whose A·Aᵀ is so distributed, standard normals below
   its diagonal and the roots of χ²(df - i) draws on it, i = 0 … p - 1. A
   χ² draw is 2·gamma((df - i)/2), from the gamma's log, so a small shape
   does not underflow to 0."
  [key df p batch o]
  (let [[k-chi k-normal] (rng/split key)
        alpha (/ (- (array/expand-dims df -1) (array/arange p)) 2.0)
        diagonal (exp (* 0.5 (+ log-2 (log-gamma-draw k-chi alpha (conj batch p) o))))
        i (array/reshape (array/arange p) [p 1])
        j (array/arange p)]
    (where (> i j)
           (rng/normal k-normal (into batch [p p]) o)
           (where (== i j) (array/expand-dims diagonal -1) 0.0))))

(defn- symmetrized
  "(m + mᵀ)/2, exactly symmetric: a product M·Mᵀ need not be, on every
   backend."
  [m]
  (* 0.5 (+ m (transposed m))))

(defn- spd-factor
  "{:in? :l}: whether v (a batch of matrices) is in a matrix family's
   support, exactly symmetric and positive definite (as Julia's isposdef),
   and its Cholesky factor, the identity's where it is not, so the log
   density there is finite before its guard, and so is its gradient."
  [v]
  (let [p (peek (array/shape v))
        in? (logical-and (array/all (== v (transposed v)) {:axis [-2 -1]})
                         (array/all (array/isfinite (la/cholesky (array/stop-gradient v))) {:axis [-2 -1]}))]
    {:in? in?
     :l (la/cholesky (where (array/expand-dims (array/expand-dims in? -1) -1) v (array/eye p {:dtype (array/dtype v)})))}))

(defn- trace-quotient
  "tr((l·lᵀ)⁻¹·r·rᵀ) = ‖l⁻¹·r‖², for lower triangular l and square r, their
   batches broadcast."
  [l r]
  (let [s (array/shape l)
        full (into (sh/broadcast (pop (pop s)) (pop (pop (array/shape r)))) (subvec s (- (count s) 2)))]
    (array/sum (square (la/solve-triangular (array/broadcast-to l full) (array/broadcast-to r full))) {:axis [-2 -1]})))

(defdist wishart
  "wishart(df, scale): the Wishart distribution over p×p symmetric positive
   definite matrices with df degrees of freedom and scale matrix S
   (Wishart(ν, S)), the law of Σᵢ xᵢxᵢᵀ for df draws xᵢ of mvnormal(0, S)
   when df is an integer; its mean is df·S. df must be above p - 1, and S
   symmetric positive definite (as mvnormal's covariance); both batch over
   their leading axes. A value that is not exactly symmetric or not positive
   definite has log density -Inf. Draws by Bartlett's decomposition."
  [df scale]
  (validate (check-matrix-family "wishart" df scale))
  (sample [key shape]
    (let [p (peek (array/shape scale))
          m (array/matmul (la/cholesky scale) (bartlett key df p (matrix-batch shape df scale) (opts df scale)))]
      (symmetrized (array/matmul m (transposed m)))))
  (log-prob [v]
    (let [p (peek (array/shape scale))
          {:keys [in? l]} (spd-factor v)
          ls (la/cholesky scale)]
      (guard in? (- (* 0.5 (- df p 1) (la/cholesky-logdet l))
                    (* 0.5 (trace-quotient ls l))
                    (* 0.5 df (+ (* p log-2) (la/cholesky-logdet ls)))
                    (log-multivariate-gamma (* 0.5 df) p)))))
  (value-shape [] (let [p (peek (array/shape scale))] (conj (matrix-batch [] df scale) p p))))

(defdist inv-wishart
  "inv-wishart(df, scale): the inverse Wishart distribution over p×p
   symmetric positive definite matrices (InverseWishart(ν, Ψ)), the law of
   W⁻¹ for W with wishart(df, scale⁻¹); its mean is scale/(df - p - 1) for
   df > p + 1. Its parameters and values are checked as wishart's."
  [df scale]
  (validate (check-matrix-family "inv-wishart" df scale))
  (sample [key shape]
    ;; W = L⁻ᵀ·A·Aᵀ·L⁻¹ for scale = L·Lᵀ and Bartlett's A, so W⁻¹ = Cᵀ·C
    ;; for C = A⁻¹·Lᵀ, by a triangular solve (no inverse)
    (let [p (peek (array/shape scale))
          batch (matrix-batch shape df scale)
          a (bartlett key df p batch (opts df scale))
          c (la/solve-triangular a (array/broadcast-to (transposed (la/cholesky scale)) (conj batch p p)))]
      (symmetrized (array/matmul (transposed c) c))))
  (log-prob [v]
    (let [p (peek (array/shape scale))
          {:keys [in? l]} (spd-factor v)
          ls (la/cholesky scale)]
      (guard in? (- (* 0.5 df (- (la/cholesky-logdet ls) (* p log-2)))
                    (log-multivariate-gamma (* 0.5 df) p)
                    (* 0.5 (+ df p 1) (la/cholesky-logdet l))
                    (* 0.5 (trace-quotient l ls))))))
  (value-shape [] (let [p (peek (array/shape scale))] (conj (matrix-batch [] df scale) p p))))

;; ---------------------------------------------------------------------------
;; Discrete distributions
;; ---------------------------------------------------------------------------

(defdist ^:elementwise ^:discrete bernoulli
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
    ;; each branch is -Inf with gradient 0 at its mass of 0: at p = 1 (or 0)
    ;; the one not taken would send 0/0 = NaN, and so would a taken one that
    ;; a pushforward's match discards
    (where v (log-mass p) (log1m-mass p)))
  (support [] [false true]))

(defn- pick-guarded
  "(f xs[..., v]) for integer v in [0, K), -Inf for any other v. The pick
   comes before f, so a gradient reaches only the entry picked: (log probs)
   would send 0/0 = NaN from every 0 entry not picked (and log-mass does
   from the one picked, where a pushforward discards it)."
  [f xs v]
  (let [k (peek (array/shape xs))
        ok (logical-and (logical-and (>= v 0) (< v k)) (integral? v))]
    (guard ok (f (stand-in ok (array/pick xs (to-int (array/clip v 0 (dec k)) xs)) 1.0)))))

(defdist ^:discrete categorical
  "categorical(probs): index i (0-based) with probability probs[i]; probs must
   be probabilities (see categorical-logits for logits)."
  [probs]
  (validate (check-vector "categorical" "probs" probs) (dc/check-probabilities "categorical" "probs" probs))
  (sample [key shape] (rng/categorical key (log probs) (sh/broadcast shape (pop (array/shape probs))) (opts probs)))
  (log-prob [v] (pick-guarded log-mass probs v))
  (support [] (vec (range (peek (array/shape probs)))))
  (value-shape [] (pop (array/shape probs))))

(defdist ^:discrete categorical-logits
  "categorical-logits(logits): index i (0-based) with probability
   softmax(logits)[i]. Any finite logits; the form LLM token sites use."
  [logits]
  (validate (check-vector "categorical-logits" "logits" logits))
  (sample [key shape] (rng/categorical key logits (sh/broadcast shape (pop (array/shape logits))) (opts logits)))
  (log-prob [v] (pick-guarded identity (array/log-softmax logits) v))
  (support [] (vec (range (peek (array/shape logits)))))
  (value-shape [] (pop (array/shape logits))))

(defdist ^:elementwise ^:discrete uniform-discrete
  "uniform_discrete(low, high): an integer uniform on low..high inclusive."
  [low high]
  (validate (dc/check-integer "uniform-discrete" "low" low Long/MIN_VALUE)
            (dc/check-integer "uniform-discrete" "high" high Long/MIN_VALUE)
            (when (every? array/concrete? [low high])
              (dc/check (array/all-true? (<= low high)) "uniform-discrete" "low must not exceed high" {:low low :high high})))
  (sample [key shape] (rng/randint key low (+ high 1) (dc/elementwise-shape shape low high) (opts low high)))
  (log-prob [v]
    (guard (logical-and (logical-and (>= v low) (<= v high)) (integral? v))
           (- (log (+ (- high low) 1)))))
  (support [] (let [lo (dc/support-bound low)
                    hi (dc/support-bound high)]
                (when (and lo hi) (vec (range lo (inc hi)))))))

(defdist ^:elementwise ^:discrete geometric
  "geometric(p): the number of failures before the first success."
  [p]
  (validate (dc/check-probability "geometric" "p" p) (dc/check-positive "geometric" "p" p))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape p) (opts p))]
      (to-int (array/floor (/ (log1p (- u)) (log1p (- p)))) p)))
  (log-prob [v]
    ;; p = 1 puts all the mass at 0: elsewhere -Inf through a stand-in p, as
    ;; log-mass, since xlog1py's derivative at -1 is infinite
    (let [some-mass (array/logical-not (logical-and (== p 1) (> v 0)))
          q (stand-in some-mass p 0.5)]
      (guard (logical-and (logical-and (>= v 0) (integral? v)) some-mass) (+ (log q) (xlog1py v (- q)))))))

;; --- Poisson and binomial samplers: inversion for small means, Hörmann's
;; transformed rejection (PTRS, BTRS) otherwise, each with fixed rounds ------

(def ^:private rejection-rounds 12)
(def ^:private inversion-steps 40)

(defn- inverted
  "An inversion over a pmf's first inversion-steps terms, from its first
   term p0 and (ratio j) = p(j+1)/p(j): the number of its partial sums below
   u times their total. So the count never passes the last term: where
   rounding leaves the total short of 1 (float32's: 1 - 1.1e-6 for binom(37,
   0.25), 1 - 3e-7 for poisson(9.1)), u above it counted every later step
   (40 > n), and the shortfall is spread over the terms instead."
  [u p0 ratio]
  (let [cdfs (reductions + (take inversion-steps (map first (iterate (fn [[p j]] [(* p (ratio j)) (inc j)]) [p0 0]))))
        v (* u (last cdfs))]
    (reduce (fn [k cdf] (+ k (where (> v cdf) 1 0))) 0 cdfs)))

(defn- poisson-inversion [u lam] (inverted u (exp (- lam)) #(/ lam (+ % 1))))

(defn- binomial-inversion [u n p q] (inverted u (exp (* n (log1p (- p)))) #(/ (* (- n %) p) (* (+ % 1) q))))

(defn- poisson-sample [key lam shape]
  (let [o (opts lam)
        [k-inv k-rej] (rng/split key)
        ;; inversion (lam < 10)
        small (poisson-inversion (rng/uniform k-inv shape o) lam)
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
        large (rng/first-accepted (array/round lam) (map proposal (range rejection-rounds)))]
    (to-int (where (< lam 10) small large) lam)))

(defn- binomial-sample [key n p shape]
  (let [o (opts n p)
        flip? (> p 0.5)
        p (where flip? (- 1 p) p)
        q (- 1 p)
        [k-inv k-rej] (rng/split key)
        ;; inversion (n·p < 10)
        small (binomial-inversion (rng/uniform k-inv shape o) n p q)
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
        large (rng/first-accepted (array/round (* n p)) (map proposal (range rejection-rounds)))
        x (where (< (* n p) 10) small large)]
    (to-int (where flip? (- n x) x) n p)))

(defdist ^:elementwise ^:discrete poisson
  "poisson(lambda): the Poisson distribution with mean lambda."
  [lam]
  (validate (dc/check-non-negative "poisson" "lambda" lam))
  (sample [key shape] (poisson-sample key lam (dc/elementwise-shape shape lam)))
  (log-prob [v]
    ;; lambda = 0 puts all the mass at 0: elsewhere -Inf through a stand-in,
    ;; as log-mass, since xlogy's derivative at 0 is infinite
    (let [some-mass (array/logical-not (logical-and (== lam 0) (> v 0)))
          l (stand-in some-mass lam 1.0)]
      (guard (logical-and (logical-and (>= v 0) (integral? v)) some-mass) (- (xlogy v l) l (lgamma (+ v 1)))))))

(defdist ^:elementwise ^:discrete binom
  "binom(n, p): the number of successes in n trials with success probability p."
  [n p]
  (validate (dc/check-probability "binom" "p" p)
            (dc/check-integer "binom" "n" n 0))
  (sample [key shape] (binomial-sample key n p (dc/elementwise-shape shape n p)))
  (log-prob [v]
    ;; p at 0 or 1 puts all the mass at 0 or n: elsewhere -Inf through a
    ;; stand-in p, as log-mass, since xlogy's derivative at 0 is infinite
    (let [some-mass (array/logical-not (logical-or (logical-and (== p 0) (> v 0)) (logical-and (== p 1) (< v n))))
          q (stand-in some-mass p 0.5)]
      (guard (logical-and (logical-and (logical-and (>= v 0) (<= v n)) (integral? v)) some-mass)
             (+ (- (lgamma (+ n 1)) (lgamma (+ v 1)) (lgamma (+ (- n v) 1)))
                (xlogy v q) (xlog1py (- n v) (- q))))))
  (support [] (some->> (dc/support-bound n) inc range vec)))

(defdist ^:elementwise ^:discrete neg-binom
  "neg_binom(r, p): the number of failures before the r-th success (r may be
   real)."
  [r p]
  (validate (dc/check-positive "neg-binom" "r" r) (dc/check-probability "neg-binom" "p" p) (dc/check-positive "neg-binom" "p" p))
  (sample [key shape]
    (let [s (dc/elementwise-shape shape r p)
          [k1 k2] (rng/split key)
          lam (* (rng/gamma k1 r s (opts r p)) (/ (- 1 p) p))]
      (poisson-sample k2 lam s)))
  (log-prob [v]
    ;; p = 1 puts all the mass at 0, as geometric's
    (let [some-mass (array/logical-not (logical-and (== p 1) (> v 0)))
          q (stand-in some-mass p 0.5)]
      (guard (logical-and (logical-and (>= v 0) (integral? v)) some-mass)
             (+ (- (lgamma (+ v r)) (lgamma r) (lgamma (+ v 1)))
                (* r (log q)) (xlog1py v (- q)))))))

;; ---------------------------------------------------------------------------
;; Mixtures and products: Gen.jl's HomogeneousMixture, HeterogeneousMixture
;; and ProductDistribution. Each is a family: a function of the component
;; parameters returning a distribution, called at the trace site like any
;; distribution constructor.
;; ---------------------------------------------------------------------------

(defn- mixture-weights
  "A mixture's weights as an array, checked as the components'
   probabilities (on the last axis)."
  [family-name weights]
  (dc/check-probabilities family-name "weights" (dc/->checkable weights))
  (dc/->array weights))

(defn- component
  "Component k's part of a homogeneous mixture's parameter p, of rank dim in
   one component: its slice on the axis before those, or all of p when it
   has no such axis or one of size 1 (one value for every component)."
  [p dim k n]
  (let [s (array/shape p)
        axis (- (count s) dim 1)]
    (if (neg? axis)
      p
      (do (dc/check (#{1 n} (nth s axis)) "homogeneous-mixture"
                    (str "a parameter of shape " s " has " (nth s axis) " components (axis " axis "), not " n)
                    {:shape s :dim dim :components n})
          (array/take p (if (= 1 (nth s axis)) 0 k) axis)))))

(defn- leading-axes
  "The batch axes of a homogeneous mixture's parameter p, of rank dim in one
   component: those before its component axis (none without one)."
  [p dim]
  (let [s (array/shape p)
        axis (- (count s) dim 1)]
    (if (pos? axis) (subvec s 0 axis) [])))

(defn- mixture
  "The ::mixture of components by weights w, more keys in its params. Its
   components are all discrete or all continuous: a pmf and a pdf would add,
   though they are densities with respect to different measures (Gen.jl's
   types forbid the mixture too)."
  [family-name w components more]
  (let [discrete (mapv dc/discrete? components)]
    (dc/check (apply = discrete) family-name
              "cannot mix a discrete and a continuous family: the pmf and the pdf would add"
              {:discrete discrete})
    (dc/->Distribution ::mixture (merge {:weights w :components components} more))))

(defn homogeneous-mixture
  "A family of mixtures of one distribution family (Gen.jl's
   HomogeneousMixture): ((homogeneous-mixture normal) weights mus stds)
   mixes (normal (mus k) (stds k)). dims, as Gen.jl's, gives each
   parameter's rank in one component (0 by default); the axis before those
   indexes the components, so ((homogeneous-mixture mvnormal [1 2]) weights
   mus covs) takes mus of shape [K d] and covs of shape [K d d]. A parameter
   without that axis is every component's. Weights are the components'
   probabilities.

   A component's value is whole: ((homogeneous-mixture normal [1 1]) w mus
   stds) with mus of shape [K d] mixes vectors, its log density summed over
   their d axis, as mvnormal's is. The axes before a parameter's component
   axis, and the weights' before theirs, are a batch of mixtures."
  ([family] (homogeneous-mixture family nil))
  ([family dims]
   (fn [weights & params]
     (let [w (mixture-weights "homogeneous-mixture" weights)
           n (peek (array/shape w))
           params (mapv dc/->array params)
           dims (or dims (repeat (count params) 0))]
       (dc/check (= (count dims) (count params)) "homogeneous-mixture" "needs one dim per parameter"
                 {:dims (vec dims) :params (count params)})
       ;; the mixture of its K components, as Gen.jl's logpdf loops over
       ;; them (all at once, the value would need the component axis before
       ;; its own axes, whose number a family does not declare)
       (let [components (mapv (fn [k] (apply family (map #(component %1 %2 k n) params dims))) (range n))
             batch (reduce sh/broadcast [] (map leading-axes params dims))]
         ;; a value's axes: a component's value shape beyond the batch (dims
         ;; for a family that declares no value shape)
         (mixture "homogeneous-mixture" w components
                  {:batch batch
                   :value-axes (if-let [s (dc/value-shape (first components))]
                                 (- (count s) (count batch))
                                 (reduce max 0 dims))}))))))

(defn heterogeneous-mixture
  "A family of mixtures of different distribution families (Gen.jl's
   HeterogeneousMixture): ((heterogeneous-mixture [normal uniform]) weights
   [mu std] [low high]) mixes (normal mu std) and (uniform low high).
   Components must have values of one shape, and be all discrete or all
   continuous. Their log densities mix per element: array parameters of an
   elementwise family are a batch of mixtures (a vector-valued component is
   a family of vector values, mvnormal)."
  [families]
  (let [families (vec families)]
    (fn [weights & component-params]
      (let [w (mixture-weights "heterogeneous-mixture" weights)]
        (dc/check (= (count families) (count component-params) (peek (array/shape w)))
                  "heterogeneous-mixture" "needs one weight and one parameter vector per component"
                  {:families (count families) :params (count component-params)})
        (mixture "heterogeneous-mixture" w (mapv #(apply %1 %2) families component-params) nil)))))

(defmethod dc/-sample ::mixture [{{:keys [weights components value-axes]} :params} key shape]
  (let [[k-pick k-draw] (rng/split key)
        ;; the weights' batch axes are independent draws too
        shape (sh/broadcast shape (pop (array/shape weights)))
        draws (mapv #(dc/sample %1 %2 shape) components (rng/split k-draw (count components)))
        dtype (reduce dt/promote (map array/dtype draws))
        ;; one pick per value: a homogeneous mixture's values are a draw's
        ;; last value-axes, others' one per element of the log density, whose
        ;; axes lead a draw's (the rest are the draw's own: mvnormal's vector)
        elements (if value-axes
                   (vec (drop-last value-axes (array/shape (first draws))))
                   (array/shape (dc/log-prob (first components) (first draws))))
        k (rng/categorical k-pick (log weights) elements (opts weights))
        own (- (array/ndim (first draws)) (count elements))]
    ;; draw every component, keep the picked one (branch-free, so it batches);
    ;; the draws share their promoted dtype, so no component is truncated
    (array/pick (array/stack (mapv #(array/astype % dtype) draws) -1)
                (array/reshape k (into (array/shape k) (repeat own 1))))))

(defn- whole-values
  "A homogeneous mixture's component log density lp of v, summed over the
   axes of one value that it keeps: an elementwise family's (normal's, of a
   vector component) beyond the value's batch, the broadcast of v's axes
   before its value-axes and the parameters' batch; none of a family that
   scores whole values (mvnormal)."
  [lp v value-axes batch]
  (let [own (- (array/ndim lp) (count (sh/broadcast (vec (drop-last value-axes (array/shape v))) batch)))]
    (if (pos? own) (array/sum lp {:axis (vec (range (- own) 0))}) lp)))

(defmethod dc/-discrete? ::mixture [{{:keys [components]} :params}] (every? dc/discrete? components))

(defmethod dc/-value-shape ::mixture [{{:keys [weights components]} :params}]
  (some->> (dc/value-shape (first components)) (sh/broadcast (pop (array/shape weights)))))

(defmethod dc/-log-prob ::mixture [{{:keys [weights components value-axes batch]} :params} v]
  ;; each component's log density of the whole value, mixed per batch element
  (log-mixture weights (array/stack (mapv #(cond-> (dc/log-prob % v) value-axes (whole-values v value-axes batch)) components)
                                    -1)))

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

(defmethod dc/-discrete? ::product [{{:keys [factors]} :params}] (every? dc/discrete? factors))

;; a vector of one value per factor: no array shape
(defmethod dc/-value-shape ::product [_] nil)

(defmethod dc/-sample ::product [{{:keys [factors]} :params} key shape]
  (mapv #(dc/sample %1 %2 shape) factors (rng/split key (count factors))))

(defmethod dc/-log-prob ::product [{{:keys [factors]} :params} v]
  (dc/check (and (sequential? v) (= (count v) (count factors))) "product"
            (str "a value needs one element per factor (" (count factors) ")") {:value v})
  ;; each factor's log density summed over its own value's axes; a value of
  ;; another shape than its factor's is an error, not several draws
  (reduce + (map (fn [f x] (dc/check-value-shape f x) (array/sum (dc/log-prob f x))) factors v)))
