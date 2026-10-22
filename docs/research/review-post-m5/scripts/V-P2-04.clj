;; Verifier, P2-04: the relative tolerance 1e-5·(1 + |v|) also fails for a large shift.
(require '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.dist.dsl :refer [dist]] '[gen.array :as array])
(dist offset-count [lam] (array/+ 1.0e6 (d/poisson lam)))     ; values 1e6 + k
(let [dd (offset-count 3.0)]
  (println "offset-count: at 1e6 + 2" (dc/log-prob dd 1000002.0) "; at 1e6 + 2.5 (no k)" (dc/log-prob dd 1000002.5)
           "; at 1e6 + 2.3 (no k)" (dc/log-prob dd 1000002.3) "; exact -Inf off the integers"))
(dist small-shift [lam] (array/+ 3.0 (d/poisson lam)))
(let [dd (small-shift 3.0)]
  (println "small-shift: at 5" (dc/log-prob dd 5.0) "; at 5.00001 (within the tolerance 6e-5)" (dc/log-prob dd 5.00001)))
