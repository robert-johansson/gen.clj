;; V-P4-03 (verifier): conditional-weight of traces that hold no completion at
;; all, and of a call-at around a constrained completion: 0.0, no error.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-P4-03.clj
(require '[gen.combinators :as comb]
         '[gen.dist :as d]
         '[gen.dynamic :refer [gen]]
         '[gen.gfi :as gfi]
         '[gen.llm.constrained :as constrained]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(defn reason [f] (try (f) (catch Exception e [:error (class e) (:reason (ex-data e))])))
(def normal-model (gen [] (trace :x (d/normal 0.0 1.0))))
(println "a normal model's trace:" (reason #(constrained/conditional-weight (gfi/simulate normal-model [] (rng/key 1)))))
(println "a distribution's trace:" (reason #(constrained/conditional-weight (gfi/simulate (d/normal 0.0 1.0) [] (rng/key 1)))))

(def tokenizer {:n-vocab 5 :bos nil :eos 4 :eog #{4} :add-bos? false
                :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab 5 :max-branches 64 :n-ctx 64}) tokenizer))
(def no0 (potential/potential 5 {:prefix #(if (some #{0} %) ##-Inf 0.0) :complete #(if (some #{0} %) ##-Inf 0.0)}))
(def c (token/completion m {:potential no0}))
(def wrapped (comb/call-at c))
(doseq [k (range 3)]
  (let [tr (gfi/simulate c [[1 2] 3] (rng/key k))
        wtr (gfi/simulate wrapped [[1 2] 3 :answer] (rng/key k))]
    (println "key" k "| the completion's trace:" (constrained/conditional-weight tr)
             "| call-at's trace (same tokens:" (= (:retval tr) (:retval wtr)) "):" (reason #(constrained/conditional-weight wtr)))))
(shutdown-agents)
(System/exit 0)
