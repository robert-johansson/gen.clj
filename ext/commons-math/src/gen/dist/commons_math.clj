(ns gen.dist.commons-math
  "Apache Commons Math 3's univariate distributions as gen.clj
   distributions (gen.dist.host), as GenDistributions makes Distributions.jl's
   Gen distributions. Needs the :commons-math alias.

   (real name make) and (integer name make) make a family of any
   RealDistribution or IntegerDistribution, make building it from the
   parameters (with a nil RandomGenerator: draws come from the key, by the
   library's inverse CDF). The families below are those gen.dist lacks, with
   Distributions.jl's parameters (GenDistributions' users write those).

   The library computes some quantiles by a solver, to an absolute accuracy
   of about 1e-9 (its default), so such draws are that close to exact."
  (:require [gen.dist.host :as host])
  (:import [org.apache.commons.math3.distribution
            AbstractIntegerDistribution AbstractRealDistribution ChiSquaredDistribution FDistribution GumbelDistribution HypergeometricDistribution
            IntegerDistribution LevyDistribution LogNormalDistribution LogisticDistribution
            NakagamiDistribution ParetoDistribution RealDistribution TDistribution
            TriangularDistribution WeibullDistribution ZipfDistribution]
           [org.apache.commons.math3.random RandomGenerator]))

(set! *warn-on-reflection* true)

(defn real
  "A family of the RealDistribution (make params…)."
  [name make]
  (host/family {:name name
                :make make
                :quantile (fn [^RealDistribution d u] (.inverseCumulativeProbability d (double u)))
                :log-density (fn [^RealDistribution d x]
                               (if (instance? AbstractRealDistribution d)
                                 (.logDensity ^AbstractRealDistribution d (double x))
                                 (Math/log (.density d (double x)))))}))

(defn integer
  "A family of the IntegerDistribution (make params…); a finite support
   (bounds below 2^31 - 1) is enumerable. Values beyond an int have mass 0."
  [name make]
  (host/family {:name name
                :discrete? true
                :make make
                :quantile (fn [^IntegerDistribution d u] (.inverseCumulativeProbability d (double u)))
                :log-density (fn [^IntegerDistribution d x]
                               (cond (not (<= Integer/MIN_VALUE x Integer/MAX_VALUE)) ##-Inf
                                     (instance? AbstractIntegerDistribution d)
                                     (.logProbability ^AbstractIntegerDistribution d (int x))
                                     :else (Math/log (.probability d (int x)))))
                :support (fn [^IntegerDistribution d]
                           (let [lo (.getSupportLowerBound d) hi (.getSupportUpperBound d)]
                             (when (< hi Integer/MAX_VALUE) (range lo (inc hi)))))}))

(def ^:private ^RandomGenerator no-rng nil)

(def chi-squared
  "chi-squared(nu): the χ² distribution with nu degrees of freedom (Chisq)."
  (real "chi-squared" (fn [nu] (ChiSquaredDistribution. no-rng (double nu)))))

(def f-dist
  "f-dist(nu1, nu2): Fisher's F distribution (FDist)."
  (real "f-dist" (fn [nu1 nu2] (FDistribution. no-rng (double nu1) (double nu2)))))

(def gumbel
  "gumbel(mu, theta): the Gumbel (maximum) distribution, location mu and
   scale theta (Gumbel)."
  (real "gumbel" (fn [mu theta] (GumbelDistribution. no-rng (double mu) (double theta)))))

(def levy
  "levy(mu, sigma): the Lévy distribution, location mu and scale sigma (Levy)."
  (real "levy" (fn [mu sigma] (LevyDistribution. no-rng (double mu) (double sigma)))))

(def log-normal
  "log-normal(mu, sigma): exp of normal(mu, sigma) (LogNormal)."
  (real "log-normal" (fn [mu sigma] (LogNormalDistribution. no-rng (double mu) (double sigma)))))

(def logistic
  "logistic(mu, theta): the logistic distribution, location mu and scale
   theta (Logistic)."
  (real "logistic" (fn [mu theta] (LogisticDistribution. no-rng (double mu) (double theta)))))

(def nakagami
  "nakagami(m, omega): the Nakagami distribution, shape m and spread omega
   (Nakagami)."
  (real "nakagami" (fn [m omega] (NakagamiDistribution. no-rng (double m) (double omega)
                                                         NakagamiDistribution/DEFAULT_INVERSE_ABSOLUTE_ACCURACY))))

(def pareto
  "pareto(alpha, theta): the Pareto distribution, shape alpha and scale
   theta (Pareto)."
  (real "pareto" (fn [alpha theta] (ParetoDistribution. no-rng (double theta) (double alpha)))))

(def student-t
  "student-t(nu): Student's t distribution with nu degrees of freedom (TDist)."
  (real "student-t" (fn [nu] (TDistribution. no-rng (double nu)))))

(def triangular
  "triangular(a, b, c): the triangular distribution on [a, b] with mode c
   (TriangularDist)."
  (real "triangular" (fn [a b c] (TriangularDistribution. no-rng (double a) (double c) (double b)))))

(def weibull
  "weibull(alpha, theta): the Weibull distribution, shape alpha and scale
   theta (Weibull)."
  (real "weibull" (fn [alpha theta] (WeibullDistribution. no-rng (double alpha) (double theta)))))

(def hypergeometric
  "hypergeometric(s, f, n): the successes among n draws without replacement
   from s successes and f failures (Hypergeometric)."
  (integer "hypergeometric" (fn [s f n] (HypergeometricDistribution. no-rng (int (+ s f)) (int s) (int n)))))

(def zipf
  "zipf(n, s): Zipf's law on 1…n with exponent s."
  (integer "zipf" (fn [n s] (ZipfDistribution. no-rng (int n) (double s)))))
