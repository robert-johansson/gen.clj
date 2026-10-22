;; V-I4-01 (verifier's independent check): smc/initialize's argument check is weaker than the
;; completion's, and smc ignores a completion's own :potential.
;; 1. prompt + max-tokens past the LM's :limit: whether smc/run succeeds depends on the key (the draws).
;; 2. an out-of-vocabulary prompt token: initialize accepts, the first step fails (with which reason?).
;; 3. a constrained completion whose particles DO satisfy its grammar (smc twisted by the same grammar,
;;    :masked): the particles' :log-p is the LM's p, the traces' score the completion's q, so they differ
;;    even with no grammar violation.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-I4-01.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.smc :as smc] '[gen.llm.grammar :as grammar]
         '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt] '[gen.gfi :as gfi] '[gen.random :as rng])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(println "== 1. past the :limit, by key")
(def m (lm/lm (toy/engine {:max-branches 16 :n-ctx 64}) (tt/tokenizer 97)))
(println ":limit" (:limit (lm/info m)))
;; three end tokens of 97: a particle runs ~32 tokens on average; the limit leaves room for 63
(def c3 (token/completion m {:end-tokens #{0 1 2}}))
(println "gfi/simulate [[95] 100]:" (reason #(gfi/simulate c3 [[95] 100] (rng/key 0))))
(println "smc/run [[95] 100], 4 particles, keys 0..11:"
         (mapv (fn [k] (reason #(smc/run c3 [[95] 100] 4 {:key (rng/key k) :host-pool false}))) (range 12)))
(println "== 2. a prompt token outside the vocabulary")
(def c (token/completion m))
(println "smc/initialize [[95 -3] 3]:" (reason #(smc/initialize c [[95 -3] 3] 4 {:key (rng/key 1)})))
(println "smc/step of it:           " (reason #(smc/step (smc/initialize c [[95 -3] 3] 4 {:key (rng/key 1) :host-pool false}))))
(println "smc/initialize [[95 :a] 3]:" (reason #(smc/initialize c [[95 :a] 3] 4 {:key (rng/key 1)})))
(println "smc/step of it:           " (reason #(smc/step (smc/initialize c [[95 :a] 3] 4 {:key (rng/key 1) :host-pool false}))))
(.close ^java.io.Closeable m)

(println "== 3. a constrained completion whose particles satisfy its grammar")
(def m2 (lm/lm (toy/engine {:max-branches 64 :n-ctx 512}) (tt/tokenizer 97) {:max-cells 8192}))
(def g (grammar/grammar m2 "[a-c]{2,3}"))
(def cc (token/completion m2 {:potential g}))
(def st (smc/run cc [[95] 4] 6 {:potential g :proposal :masked :key (rng/key 3) :host-pool false}))
(println "particles      " (smc/particles st))
(println "particles' :log-p (LM's p)" (mapv #(format "%.4f" (double (:log-p %))) (:particles st)))
(println "traces' :score (q)        " (mapv #(some->> (:score %) double (format "%.4f")) (smc/traces st)))
(println "the unconstrained completion's scores of the same tokens:"
         (mapv #(format "%.4f" (double (:score (:trace (gfi/generate (token/completion m2) [[95] 4] (token/forced %) (rng/key 0))))))
               (smc/particles st)))
(.close ^java.io.Closeable m2)
(shutdown-agents)
(System/exit 0)
