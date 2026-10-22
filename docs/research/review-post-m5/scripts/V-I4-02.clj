;; V-I4-02 (verifier's independent check): the end-token agreement of DESIGN §9 ("A grammar knows its
;; end tokens") is enforced only for a bare Grammar given to token/completion.
;; 1. DESIGN's own failing example ([ab]{3} then a token that is an end token), the grammar in a
;;    two-member product with a soft potential: accepted, then every run fails mid-run.
;; 2. a potential/potential wrapping the grammar's functions: also accepted.
;; 3. token SMC with the bare mismatched grammar as its twist, :lm proposal: accepted, log-ml -Inf.
;;    By the documented target, Z = Σ p(x) exp(complete(x)) is exactly 0 here (no x has weight), so
;;    the -Inf is the right number for a target the user did not mean, silently.
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-I4-02.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.grammar :as grammar] '[gen.llm.smc :as smc]
         '[gen.llm.potential :as potential] '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt]
         '[gen.gfi :as gfi] '[gen.random :as rng])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(def m (lm/lm (toy/engine {:max-branches 64 :n-ctx 512}) (tt/tokenizer 97) {:max-cells 8192}))
(def dot 14) ; "." in the toy tokenizer (ASCII 46 - 32)
(def g (grammar/grammar m "[ab]{3}\\." {:end-tokens #{}}))   ; "." is text for this grammar
(def soft (potential/potential 97 {:prefix (fn [ctx] (* -0.1 (count ctx))) :complete (fn [ctx] (* -0.1 (count ctx)))}))
(println "1. bare g, completion :end-tokens #{14}:       " (reason #(token/completion m {:end-tokens #{dot} :potential g})))
(println "   (product g soft), completion #{14}:         " (reason #(token/completion m {:end-tokens #{dot} :potential (potential/product g soft)})))
(def wrapped (token/completion m {:end-tokens #{dot} :potential (potential/product g soft)}))
(println "   simulate [[95] 6] keys 0..9:" (frequencies (map (fn [k] (reason #(gfi/simulate wrapped [[95] 6] (rng/key k)))) (range 10))))
(def by-fns (potential/potential 97 {:prefix #(potential/prefix g %) :complete #(potential/complete g %)
                                     :logw-next #(potential/logw-next g %)}))
(println "2. potential/potential over g's functions, #{14}:" (reason #(token/completion m {:end-tokens #{dot} :potential by-fns})))
(let [st (smc/run (token/completion m {:end-tokens #{dot}}) [[95] 6] 32 {:potential g :key (rng/key 5) :host-pool false})]
  (println "3. smc/run, bare g as twist, :lm proposal: log-ml" (smc/log-ml-estimate st)
           "dead" (count (filter :dead? (:particles st))) "of 32"))
(println "   the matched grammar of the language without the end's byte, [ab]{3}, end tokens #{14}:"
         (let [g2 (grammar/grammar m "[ab]{3}" {:end-tokens #{dot}})
               st (smc/run (token/completion m {:end-tokens #{dot}}) [[95] 6] 32 {:potential g2 :proposal :masked :key (rng/key 5) :host-pool false})]
           [(smc/log-ml-estimate st) (frequencies (smc/particles st))]))
(.close ^java.io.Closeable m)
(shutdown-agents)
(System/exit 0)
