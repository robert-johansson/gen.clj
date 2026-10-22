;; I4-01: gen.llm.smc re-implements the completion's argument checks, weaker. It accepts arguments
;; token/completion refuses (a prompt token outside the vocabulary; prompt + max-tokens past the LM's
;; :limit), so a run's validity depends on its draws and smc/traces of a run that succeeded throws;
;; and it accepts a completion with its own :potential, which it ignores, so smc/traces' scores are
;; not the particles' log p.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-01.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.smc :as smc] '[gen.llm.grammar :as grammar]
         '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt] '[gen.gfi :as gfi] '[gen.random :as rng])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(println "== Arguments (an LM whose :limit is 64)")
(def m (lm/lm (toy/engine {:max-branches 16 :n-ctx 64}) (tt/tokenizer 97)))
(def c (token/completion m))
(println ":limit" (:limit (lm/info m)))
(println "gfi/simulate   [[95 500] 3]  " (reason #(gfi/simulate c [[95 500] 3] (rng/key 1))))
(println "smc/initialize [[95 500] 3]  " (reason #(smc/initialize c [[95 500] 3] 4 {:key (rng/key 1)})))
(println "gfi/simulate   [[95] 100]    " (reason #(gfi/simulate c [[95] 100] (rng/key 1))))
(println "smc/initialize [[95] 100]    " (reason #(smc/initialize c [[95] 100] 4 {:key (rng/key 1)})))
;; most tokens end a completion, so the particles end long before the :limit
(def short-c (token/completion m {:end-tokens (set (range 70))}))
(def state (smc/run short-c [[95] 100] 4 {:key (rng/key 1) :host-pool false}))
(println "smc/run [[95] 100], ending early:" (smc/particles state) "log-ml" (smc/log-ml-estimate state))
(println "smc/traces of that run       " (reason #(smc/traces state)))
(.close ^java.io.Closeable m)

(println "== A completion with its own :potential")
(def m2 (lm/lm (toy/engine {:max-branches 64 :n-ctx 512}) (tt/tokenizer 97) {:max-cells 8192}))
(def constrained (token/completion m2 {:potential (grammar/grammar m2 "[a-c]+")}))
(def st (smc/run constrained [[95] 3] 4 {:key (rng/key 1) :host-pool false}))
(println "particles          " (smc/particles st) "(the grammar allows only tokens 65 to 67)")
(println "particles' log p   " (mapv :log-p (:particles st)))
(println "smc/traces' scores " (mapv :score (smc/traces st)))
(.close ^java.io.Closeable m2)
(shutdown-agents)
(System/exit 0)
