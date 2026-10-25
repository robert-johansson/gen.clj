(ns gen.dist.commons-math
  "Apache Commons Math 3's univariate distributions as gen.clj
   distributions (gen.dist.host), as GenDistributions makes Distributions.jl's
   Gen distributions. Needs the :commons-math alias.

   (real name make) and (integer name make) make a family of any
   RealDistribution or IntegerDistribution, make building it from the
   parameters (with a nil RandomGenerator: draws come from the key, by the
   library's inverse CDF). hypergeometric and zipf are the discrete families
   gen.dist lacks, with Distributions.jl's parameters. (Its continuous
   families beyond Gen.jl's are native in gen.dist, where they batch, run on
   every backend and differentiate.)

   The library computes some quantiles by a solver, to an absolute accuracy
   of about 1e-9 (its default), so such draws are that close to exact."
  {:api :experimental}
  (:require [gen.dist.core :as dc]
            [gen.dist.host :as host])
  (:import [org.apache.commons.math3.distribution
            AbstractIntegerDistribution AbstractRealDistribution HypergeometricDistribution
            IntegerDistribution RealDistribution ZipfDistribution]
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

(def hypergeometric
  "hypergeometric(s, f, n): the successes among n draws without replacement
   from s successes and f failures (Hypergeometric)."
  (integer "hypergeometric" (fn [s f n]
                              (doseq [[p v] {"s" s "f" f "n" n}] (dc/check-integer "hypergeometric" p v 0))
                              (HypergeometricDistribution. no-rng (int (+ s f)) (int s) (int n)))))

(def zipf
  "zipf(n, s): Zipf's law on 1…n with exponent s."
  (integer "zipf" (fn [n s]
                    (dc/check-integer "zipf" "n" n 1)
                    (ZipfDistribution. no-rng (int n) (double s)))))
