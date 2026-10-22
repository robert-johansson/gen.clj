;; I4-03: the end-token checks of byte-lm and grammar coerce with long, so a token beyond long's range
;; is an IllegalArgumentException where token/completion's (br/index?) gives ex-info; the Branching
;; quality review's M6 fixed this class for token checks (docs/research/llm-branching.md).
;; Run from the snapshot worktree: nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/I4-03.clj
(require '[gen.llm.lm :as lm] '[gen.llm.token :as token] '[gen.llm.bytes :as bytes] '[gen.llm.grammar :as grammar]
         '[gen.llm.toy-engine :as toy] '[gen.llm.toy-tokenizer :as tt])

(defn reason [f]
  (try (f) :no-error
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e) :no-reason))
       (catch Throwable t (.getName (class t)))))

(def m (lm/lm (toy/engine {:max-branches 16 :n-ctx 64}) (tt/tokenizer 97)))
(def big 100000000000000000000N)
(println "token/completion :end-tokens #{big}" (reason #(token/completion m {:end-tokens #{big}})))
(println "bytes/byte-lm    :end-tokens #{big}" (reason #(bytes/byte-lm m {:end-tokens #{big}})))
(println "grammar/grammar  :end-tokens #{big}" (reason #(grammar/grammar m "a+" {:end-tokens #{big}})))
(.close ^java.io.Closeable m)
(shutdown-agents)
(System/exit 0)
