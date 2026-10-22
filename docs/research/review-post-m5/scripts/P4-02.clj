;; P4-02: the grammar's end tokens must be the completion's (DESIGN §9), but only
;; a completion's own :potential is checked: token SMC and a product accept a
;; mismatch, and SMC then estimates another Z, silently.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-02.clj
(require '[gen.gfi :as gfi]
         '[gen.llm.byte-tokenizer :as bt]
         '[gen.llm.grammar :as grammar]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.smc :as smc]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

;; tokens: a=0 b=1 c=2, BOS=3 (<s>), EOS=4 (</s>); the LM's :eog is #{4}
(def tok (bt/tokenizer ["a" "b" "c"]))
(def m (lm/lm (toy/engine {:n-vocab 5 :max-branches 64 :n-ctx 64}) tok {:max-cells 4096}))
(def prompt [3])
(def args [prompt 3])

;; completions of exactly 3 tokens (no end token): EOS is a token, its text "</s>"
(def exact3 (token/completion m {:end-tokens #{}}))
;; any text but newlines: every token's bytes qualify, EOS's "</s>" too
(def matched (grammar/grammar m ".*" {:end-tokens #{}}))
(def mismatched (grammar/grammar m ".*"))          ; the default: the LM's :eog, #{4}, left out of the text

(defn reason [f] (try (f) :no-error (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(println "a completion under the mismatched grammar:"
         (reason #(token/completion m {:end-tokens #{} :potential mismatched})))
(println "the same grammar inside a product:"
         (reason #(token/completion m {:end-tokens #{} :potential (potential/product mismatched)})))

(defn z-hat [g proposal]
  (let [zs (mapv #(Math/exp (smc/log-ml-estimate (smc/run exact3 args 50 {:potential g :proposal proposal :key (rng/key %)})))
                 (range 40))]
    (/ (reduce + zs) (count zs))))

;; exact: P(no EOS among the 3 tokens), by enumeration of the toy model's rows
(def p-no-eos
  (reduce + (for [a (range 4) b (range 4) c (range 4)
                  :let [lp (+ (aget (toy/toy-logprobs 5 prompt) a)
                              (aget (toy/toy-logprobs 5 (conj prompt a)) b)
                              (aget (toy/toy-logprobs 5 (conj prompt a b)) c))]]
              (Math/exp lp))))

(println "token SMC, Z = P(the completion's text is in .*) = 1 for every completion of 3 tokens:")
(println "  matched grammar,    :masked  mean Z-hat =" (z-hat matched :masked))
(println "  mismatched grammar, :masked  mean Z-hat =" (z-hat mismatched :masked) " (no error)")
(println "  mismatched grammar, :lm      mean Z-hat =" (z-hat mismatched :lm))
(println "  exact P(no EOS token among 3) =" p-no-eos)
(shutdown-agents)
(System/exit 0)
