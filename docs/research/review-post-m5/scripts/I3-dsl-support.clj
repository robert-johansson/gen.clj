;; enumerable-support tells "no support" from other errors by the ex-data's
;; keys (:type or :distribution); a host family's "no enumerable support"
;; carries :family, so a dist over one is rethrown instead of inverted.
;; Run with -M:test:commons-math.
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.dist.dsl :refer [dist]]
         '[gen.dist.commons-math :as cmath])
(import '[org.apache.commons.math3.distribution PoissonDistribution])
(defmacro show [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body)) (catch Throwable e# (str (.getSimpleName (class e#)) ": " (ex-message e#) " " (ex-data e#))))))

(def ^{::dc/constructor true} host-poisson
  (cmath/integer "host-poisson" (fn [lam] (PoissonDistribution. nil (double lam) 1e-12 10000000))))
(dist shifted-native [lam] (array/+ (d/poisson lam) 1))
(dist shifted-host [lam] (array/+ (host-poisson lam) 1))

(show "native poisson pushforward, log-prob at 4 (expect log P(X=3), lam 3 = -1.495)" (dc/log-prob (shifted-native 3.0) 4))
(show "host poisson log-prob at 3" (dc/log-prob (host-poisson 3.0) 3))
(show "host poisson pushforward, log-prob at 4" (dc/log-prob (shifted-host 3.0) 4))
