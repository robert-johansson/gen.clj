;; P4-05: token SMC does not check the completion's arguments against the LM's
;; :limit, so whether a run fails depends on its draws, and a run that
;; succeeds has particles smc/traces cannot convert.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-05.clj
(require '[gen.gfi :as gfi]
         '[gen.llm.lm :as lm]
         '[gen.llm.smc :as smc]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(def n-vocab 5)
(def tokenizer
  {:n-vocab n-vocab :bos nil :eos 4 :eog #{4} :add-bos? false
   :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
;; :max-cells 8 makes the LM's :limit 8 tokens
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 64}) tokenizer {:max-cells 8}))
(def c (token/completion m))
(def args [[0 1] 10])                    ; 2 + 10 tokens > the limit of 8

(defn reason [f] (try (f) :no-error (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(println "the LM's limit:" (:limit (lm/info m)))
(println "simulate:" (reason #(gfi/simulate c args (rng/key 0))) "(before any read, for every key)")
(println "smc/run by key:" (frequencies (map #(reason (fn [] (smc/run c args 4 {:key (rng/key %)}))) (range 30))))
(let [ok (first (filter #(= :no-error (reason (fn [] (smc/run c args 4 {:key (rng/key %)})))) (range 30)))
      s (smc/run c args 4 {:key (rng/key ok)})]
  (println "a run that succeeded (key" ok "), its particles:" (smc/particles s))
  (println "smc/traces of it:" (reason #(smc/traces s))))
(shutdown-agents)
(System/exit 0)
