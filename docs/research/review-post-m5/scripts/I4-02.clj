;; I4-02: a completion under a grammar must have the grammar's end tokens (DESIGN §9, "A grammar knows its
;; end tokens"): token/completion refuses a mismatch with :mismatched-end-tokens, but only for a bare
;; grammar. The same grammar inside potential/product is accepted, and the mismatch surfaces mid-run.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-02.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.grammar :as grammar]
         '[gen.llm.potential :as potential] '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt]
         '[gen.gfi :as gfi] '[gen.random :as rng])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(def m (lm/lm (toy/engine {:max-branches 64 :n-ctx 512}) (tt/tokenizer 97) {:max-cells 8192}))
(def g (grammar/grammar m "[a-c]+(-[0-9]{1,2})?"))   ; its end tokens: the LM's :eog, #{96}
(println "grammar's end tokens" (grammar/end-tokens g))
(println "bare grammar, completion :end-tokens #{}:   " (reason #(token/completion m {:end-tokens #{} :potential g})))
(println "(product g), completion :end-tokens #{}:     " (reason #(token/completion m {:end-tokens #{} :potential (potential/product g)})))
(def wrapped (token/completion m {:end-tokens #{} :potential (potential/product g)}))
(def matched (token/completion m {:potential (potential/product g)}))
(println "simulate [prompt 8] over keys 0..19, wrapped (mismatched):"
         (frequencies (map (fn [i] (reason #(gfi/simulate wrapped [[95] 8] (rng/key i)))) (range 20))))
(println "simulate [prompt 8] over keys 0..19, matched end tokens:"
         (frequencies (map (fn [i] (reason #(gfi/simulate matched [[95] 8] (rng/key i)))) (range 20))))
;; token SMC takes a grammar as its twist and checks no end tokens at all, bare or wrapped
(require '[gen.llm.smc :as smc])
(def nl (grammar/grammar m "[ab]{3}\\." {:end-tokens #{}}))   ; the text needs the '.' token (14)
(println "grammar [ab]{3}\\. with :end-tokens #{}; under completion end tokens #{14}:"
         (reason #(token/completion m {:end-tokens #{14} :potential nl})))
(println "  as smc's :potential, completion end tokens #{14}:"
         (reason #(smc/initialize (token/completion m {:end-tokens #{14}}) [[95] 6] 16 {:potential nl :key (rng/key 1)})))
(let [st (smc/run (token/completion m {:end-tokens #{14}}) [[95] 6] 16 {:potential nl :proposal :masked :key (rng/key 1) :host-pool false})]
  (println "  smc/run :masked, log-ml-estimate" (smc/log-ml-estimate st) "particles" (frequencies (smc/particles st))))
(println "the same language with the end token's bytes, matched end tokens #{14}:"
         (reason #(grammar/grammar m "[ab]{3}\\." {:end-tokens #{14}})))
(.close ^java.io.Closeable m)
(shutdown-agents)
(System/exit 0)
