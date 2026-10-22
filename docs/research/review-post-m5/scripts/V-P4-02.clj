;; V-P4-02 (verifier): R2's own mismatch (DESIGN §9, "A grammar knows its end
;; tokens"), a newline that ends the completion but is text to the grammar,
;; on the two paths the constructor check does not cover: token SMC and a
;; product.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-P4-02.clj
(require '[gen.gfi :as gfi]
         '[gen.llm.byte-tokenizer :as bt]
         '[gen.llm.grammar :as grammar]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.smc :as smc]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

;; a=0 b=1 newline=2, <s>=3, </s>=4
(def nl (bt/tokenizer ["a" "b" "\n"]))
(def n-vocab (:n-vocab nl))
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 128}) nl {:max-cells 8192}))
(def prompt [(:bos nl)])
(def args [prompt 6])
(def c (token/completion m {:end-tokens #{2}}))            ; the newline ends a completion

(def matched (grammar/grammar m "[ab]{3}" {:end-tokens #{2}}))     ; the design's pair
(def r2 (grammar/grammar m "[ab]{3}\n" {:end-tokens #{}}))         ; R2's: the newline is text to the grammar

(defn reason [f] (try (f) :no-error (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(println "completion under R2's grammar:" (reason #(token/completion m {:end-tokens #{2} :potential r2})))
(let [gp (reason #(token/completion m {:end-tokens #{2} :potential (potential/product r2)}))]
  (println "completion under (product R2's grammar):" (if (keyword? gp) gp :made))
  (println "  its simulate:" (reason #(gfi/simulate (token/completion m {:end-tokens #{2} :potential (potential/product r2)})
                                                    args (rng/key 1)))))

;; Z the pair means: P(three of a/b, then the newline), by the toy rows
(def z (reduce + (for [x [0 1] y [0 1] w [0 1]
                       :let [ctx prompt
                             lp (+ (aget (toy/toy-logprobs n-vocab ctx) x)
                                   (aget (toy/toy-logprobs n-vocab (conj ctx x)) y)
                                   (aget (toy/toy-logprobs n-vocab (conj ctx x y)) w)
                                   (aget (toy/toy-logprobs n-vocab (conj ctx x y w)) 2))]]
                   (Math/exp lp))))
(println "exact Z = P(three of a/b, then the newline) =" z)

(defn z-hats [g proposal]
  (mapv #(Math/exp (smc/log-ml-estimate (smc/run c args 50 {:potential g :proposal proposal :key (rng/key %)}))) (range 40)))
(defn mean [v] (/ (reduce + 0.0 v) (count v)))
(doseq [[label g] [[:matched matched] [:r2 r2]] proposal [:masked :lm]]
  (println "SMC," label proposal ": mean Z-hat" (mean (z-hats g proposal)) "(no error)"))
(shutdown-agents)
(System/exit 0)
