;; I4-06: gen.llm.smc's namespace docstring says particle i's token at step t is drawn from
;; (fold-in (fold-in key t) i); the code draws it from (fold-in (fold-in (fold-in key t) 0) i).
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-06.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.smc :as smc] '[gen.dist.core :as dc]
         '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt] '[gen.random :as rng])

(def m (lm/lm (toy/engine {:max-branches 16 :n-ctx 64}) (tt/tokenizer 97)))
(def c (token/completion m {:end-tokens #{}}))
(def k (rng/key 7))
(def prompt [95])
(def s1 (smc/step (smc/initialize c [prompt 3] 8 {:key k :host-pool false})))
(def row (lm/next-logprobs m prompt))
(def drawn (mapv first (smc/particles s1)))
(def documented (mapv #(dc/sample (token/dist row) (rng/fold-in (rng/fold-in k 0) %)) (range 8)))
(def actual (mapv #(dc/sample (token/dist row) (rng/fold-in (rng/fold-in (rng/fold-in k 0) 0) %)) (range 8)))
(println "drawn by smc/step              " drawn)
(println "the docstring's keys           " documented (if (= drawn documented) "same" "DIFFERENT"))
(println "(fold-in (fold-in (fold-in k t) 0) i)" actual (if (= drawn actual) "same" "DIFFERENT"))
(.close ^java.io.Closeable m)
(shutdown-agents)
(System/exit 0)
