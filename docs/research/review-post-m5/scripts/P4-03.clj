;; P4-03: conditional-weight of a trace that is not a completion's: 0.0, silently.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-03.clj
(require '[gen.combinators :as comb]
         '[gen.dist :as d]
         '[gen.dynamic :refer [gen]]
         '[gen.gfi :as gfi]
         '[gen.llm.constrained :as constrained]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng]
         '[gen.trace :as trace])

(def n-vocab 5)
(def tokenizer {:n-vocab n-vocab :bos nil :eos 4 :eog #{4} :add-bos? false
                :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 64}) tokenizer))
(def no3 (potential/potential n-vocab {:prefix #(if (some #{3} %) ##-Inf 0.0) :complete #(if (some #{3} %) ##-Inf 0.0)}))
(def answer (token/completion m {:potential no3}))

(def model
  (gen []
    (let [tokens (splice :answer answer [0 1] 4)]
      (trace :liked (d/bernoulli (if (some #{2} tokens) 0.9 0.3)))
      tokens)))

(doseq [k (range 4)]
  (let [tr (gfi/simulate model [] (rng/key k))
        sub (trace/subtrace tr [:answer])
        mtr (gfi/simulate (comb/map answer) [[[0 1] [0 2]] [4 4]] (rng/key k))]
    (println "key" k "| the completion's trace:" (constrained/conditional-weight sub)
             "| the model's trace:" (constrained/conditional-weight tr)
             "| a Map's trace:" (constrained/conditional-weight mtr))))
(shutdown-agents)
(System/exit 0)
