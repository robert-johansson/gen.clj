(ns gen.dist
  "Gen.jl's built-in distributions, with Gen.jl's names and parameterizations
   (normal by standard deviation, gamma by scale, categorical by
   probabilities), plus categorical-logits. Discrete values are 0-based
   integers; bernoulli values are booleans.

   Every log-prob is -Inf outside the support. Every sampler is a pure,
   branch-free function of its key: rejection samplers run a fixed number of
   vectorized rounds rather than a data-dependent loop, so they batch and never
   force evaluation."
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as a :refer [+ - * / < > <= >= == log log1p exp sqrt square where
                                     lgamma log-beta xlogy xlog1py logical-and logical-or]]
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

(defn- integral? [v] (== v (a/floor v)))

(defn- opts [& params] {:backend (apply dc/backend-of params)})

(defn- to-int [x & params]
  (a/astype x (dc/int-dtype (apply dc/backend-of x params))))

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
  (validate (when (and (a/host? low) (a/host? high))
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
      (+ loc (* scale (a/sign u) (log1p (- (a/abs u)))))))
  (log-prob [v]
    (- (+ (log (* 2 scale)) (/ (a/abs (- v loc)) scale)))))

(defdist cauchy
  "cauchy(x0, gamma): the Cauchy distribution with location x0 and scale gamma."
  [x0 scale]
  (validate (dc/check-positive "cauchy" "scale" scale))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape x0 scale)
                         (assoc (opts x0 scale) :low (Math/ulp 1.0) :high 1.0))]
      (+ x0 (* scale (a/tan (* Math/PI (- u 0.5)))))))
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
           (a/logaddexp (+ (log theta) (dc/log-prob (beta a b) v)) (log1p (- theta))))))

(defdist piecewise-uniform
  "piecewise_uniform(bounds, probs): bin i, chosen with probability probs[i],
   is [bounds[i], bounds[i+1]], uniform within the bin."
  [bounds probs]
  (validate (dc/check-probabilities "piecewise-uniform" "probs" probs))
  (sample [key shape]
    (let [[k1 k2] (rng/split key)
          i (rng/categorical k1 (log probs) shape (opts bounds probs))
          lo (a/take bounds i)
          hi (a/take bounds (+ i 1))]
      (+ lo (* (- hi lo) (rng/uniform k2 (a/shape i) (opts bounds probs))))))
  (log-prob [v]
    (let [n (a/size probs)
          i (- (a/sum (>= (a/expand-dims v -1) (a/slice bounds 0 0 n)) {:axis -1}) 1)
          i (to-int (a/clip i 0 (dec n)) bounds)]
      (guard (logical-and (>= v (a/take bounds 0)) (<= v (a/take bounds n)))
             (- (log (a/take probs i)) (log (- (a/take bounds (+ i 1)) (a/take bounds i))))))))

(defdist mvnormal
  "mvnormal(mu, cov): the multivariate normal with mean vector mu and
   covariance matrix cov (batched over leading axes)."
  [mu cov]
  (sample [key shape]
    (let [d (peek (a/shape mu))
          batch (sh/broadcast (sh/broadcast shape (pop (a/shape mu))) (pop (pop (a/shape cov))))
          eps (rng/normal key (conj batch d) (opts mu cov))]
      (+ mu (a/squeeze (a/matmul (la/cholesky cov) (a/expand-dims eps -1)) -1))))
  (log-prob [v]
    (let [d (peek (a/shape mu))
          l (la/cholesky cov)
          z (la/solve-triangular l (- v mu))]
      (- (* -0.5 (a/sum (square z) {:axis -1}))
         (* 0.5 (la/cholesky-logdet l))
         (* d half-log-2pi)))))

(defdist dirichlet
  "dirichlet(alpha): the Dirichlet distribution over the simplex of the last
   axis of alpha."
  [alpha]
  (sample [key shape]
    (let [k (peek (a/shape alpha))
          g (rng/gamma key alpha (sh/broadcast (conj shape k) (a/shape alpha)) (opts alpha))]
      (/ g (a/sum g {:axis -1 :keep-dims? true}))))
  (log-prob [v]
    (guard (logical-and (a/all (>= v 0) {:axis -1})
                        (<= (a/abs (- (a/sum v {:axis -1}) 1)) 1e-6))
           (- (a/sum (xlogy (- alpha 1) v) {:axis -1})
              (- (a/sum (lgamma alpha) {:axis -1}) (lgamma (a/sum alpha {:axis -1})))))))

;; ---------------------------------------------------------------------------
;; Discrete distributions
;; ---------------------------------------------------------------------------

(defdist bernoulli
  "bernoulli(prob): true with probability prob."
  [p]
  (validate (dc/check-probability "bernoulli" "prob" p))
  (sample [key shape] (rng/bernoulli key p (dc/elementwise-shape shape p) (opts p)))
  (log-prob [v]
    (where (== v 1) (log p) (where (== v 0) (log1p (- p)) ##-Inf)))
  (support [] [false true]))

(defn- pick-guarded
  "log-probs[..., v] for integer v in [0, K), -Inf for any other v."
  [log-probs v]
  (let [k (peek (a/shape log-probs))
        ok (logical-and (logical-and (>= v 0) (< v k)) (integral? v))]
    (guard ok (a/pick log-probs (to-int (a/clip v 0 (dec k)) log-probs)))))

(defdist categorical
  "categorical(probs): index i (0-based) with probability probs[i]; probs must
   be probabilities (see categorical-logits for logits)."
  [probs]
  (validate (dc/check-probabilities "categorical" "probs" probs))
  (sample [key shape] (rng/categorical key (log probs) (sh/broadcast shape (pop (a/shape probs))) (opts probs)))
  (log-prob [v] (pick-guarded (log probs) v))
  (support [] (vec (range (peek (a/shape probs))))))

(defdist categorical-logits
  "categorical-logits(logits): index i (0-based) with probability
   softmax(logits)[i]. Any finite logits; the form LLM token sites use."
  [logits]
  (sample [key shape] (rng/categorical key logits (sh/broadcast shape (pop (a/shape logits))) (opts logits)))
  (log-prob [v] (pick-guarded (a/log-softmax logits) v))
  (support [] (vec (range (peek (a/shape logits))))))

(defdist uniform-discrete
  "uniform_discrete(low, high): an integer uniform on low..high inclusive."
  [low high]
  (validate (when (and (a/host? low) (a/host? high))
              (dc/check (<= low high) "uniform-discrete" "low must not exceed high" {:low low :high high})))
  (sample [key shape] (rng/randint key low (+ high 1) (dc/elementwise-shape shape low high) (opts low high)))
  (log-prob [v]
    (guard (logical-and (logical-and (>= v low) (<= v high)) (integral? v))
           (- (log (+ (- high low) 1)))))
  (support [] (vec (range low (inc high)))))

(defdist geometric
  "geometric(p): the number of failures before the first success."
  [p]
  (validate (dc/check-probability "geometric" "p" p)
            (when (a/host? p) (dc/check (> p 0) "geometric" "p must be in (0, 1]" {:p p})))
  (sample [key shape]
    (let [u (rng/uniform key (dc/elementwise-shape shape p) (opts p))]
      (to-int (a/floor (/ (log1p (- u)) (log1p (- p)))) p)))
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
                         us (- 0.5 (a/abs uu))
                         k (a/floor (+ (* (+ (/ (* 2 aa) us) b) uu) lam 0.43))
                         quick (logical-and (>= us 0.07) (<= v vr))
                         reject (logical-or (< k 0) (logical-and (< us 0.013) (> v us)))
                         s (log (/ (* v inv-alpha) (+ (/ aa (* us us)) b)))
                         t (- (* k log-lam) lam (lgamma (+ k 1)))]
                     [(logical-or quick (logical-and (a/logical-not reject) (<= s t))) k]))
        large (first-accepted (a/round lam) (map proposal (range rejection-rounds)))]
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
        m (a/floor (* (+ n 1) p))
        h (+ (lgamma (+ m 1)) (lgamma (+ (- n m) 1)))
        ks (rng/split k-rej (* 2 rejection-rounds))
        proposal (fn [r]
                   (let [uu (- (rng/uniform (ks (* 2 r)) shape o) 0.5)
                         v (rng/uniform (ks (+ 1 (* 2 r))) shape o)
                         us (- 0.5 (a/abs uu))
                         k (a/floor (+ (* (+ (/ (* 2 aa) us) b) uu) c))
                         in-range (logical-and (>= k 0) (<= k n))
                         quick (logical-and (>= us 0.07) (<= v vr))
                         v' (log (/ (* v alpha) (+ (/ aa (* us us)) b)))
                         slow (<= v' (+ (- h (lgamma (+ k 1)) (lgamma (+ (- n k) 1))) (* (- k m) lpq)))]
                     [(logical-and in-range (logical-or quick slow)) k]))
        large (first-accepted (a/round (* n p)) (map proposal (range rejection-rounds)))
        x (where (< (* n p) 10) small large)]
    (to-int (where flip? (- n x) x) n p)))

(defdist poisson
  "poisson(lambda): the Poisson distribution with mean lambda."
  [lam]
  (validate (when (a/host? lam) (dc/check (>= lam 0) "poisson" "lambda must be >= 0" {:lambda lam})))
  (sample [key shape] (poisson-sample key lam (dc/elementwise-shape shape lam)))
  (log-prob [v]
    (guard (logical-and (>= v 0) (integral? v)) (- (xlogy v lam) lam (lgamma (+ v 1))))))

(defdist binom
  "binom(n, p): the number of successes in n trials with success probability p."
  [n p]
  (validate (dc/check-probability "binom" "p" p))
  (sample [key shape] (binomial-sample key n p (dc/elementwise-shape shape n p)))
  (log-prob [v]
    (guard (logical-and (logical-and (>= v 0) (<= v n)) (integral? v))
           (+ (- (lgamma (+ n 1)) (lgamma (+ v 1)) (lgamma (+ (- n v) 1)))
              (xlogy v p) (xlog1py (- n v) (- p)))))
  (support [] (vec (range (inc n)))))

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
