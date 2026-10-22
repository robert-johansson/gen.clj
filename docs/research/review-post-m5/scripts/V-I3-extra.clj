;; Verifier's supplementary checks (I3-02 macroexpansion message; I3-04's outer walk).
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.dsl] '[gen.dist.commons-math :as cmath]
         '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.random :as rng] '[gen.choicemap :as cm]
         '[gen.combinators :as comb] '[gen.diff :as diff] '[gen.inference.particle-filter :as pf])
(import '[org.apache.commons.math3.distribution PoissonDistribution])
(def host-poisson (cmath/integer "host-poisson" (fn [lam] (PoissonDistribution. nil (double lam) 1e-12 10000000))))
(try (eval '(gen.dist.dsl/dist shifted-host [lam] (gen.array/+ (user/host-poisson lam) 1)))
     (catch Throwable e (println "I3-02 dist over an unmarked host family, root cause:"
                                 (ex-message (loop [e e] (if (.getCause e) (recur (.getCause e)) e))))))
;; I3-04: what the outer array/eval! of a pf step walks, per particle, on a lazy backend
(let [kernel (gen [t s] (let [x (trace :x (d/normal s 1.0))] x))
      model (gen [n] (splice :steps (comb/unfold kernel) n 0.0))
      tr (:trace (gfi/generate model [3] cm/EMPTY (rng/key 1)))]
  (println "I3-04 leaves of a whole trace:" (count (array/tree-leaves tr))
           " of [choices retval score]:" (count (array/tree-leaves [(:choices tr) (:retval tr) (:score tr)]))))
(shutdown-agents)
