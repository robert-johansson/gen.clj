;; P4-06: token/dist weighs an integral non-integer number (3.0) as outside its
;; support, silently, where gen.dist's categorical takes it as index 3, and a
;; completion's constraints refuse it loudly.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-06.clj
(require '[gen.choicemap :as cm]
         '[gen.dist :as d]
         '[gen.dist.core :as dc]
         '[gen.dynamic :refer [gen]]
         '[gen.gfi :as gfi]
         '[gen.llm.lm :as lm]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(def row (toy/toy-logprobs 5 [0 1]))
(def probs (mapv #(Math/exp %) row))

(def with-token (gen [] (trace :t (token/dist row))))
(def with-categorical (gen [] (trace :t (d/categorical probs))))

(defn reason [f] (try (f) (catch clojure.lang.ExceptionInfo e [:error (:reason (ex-data e))])))

(doseq [v [3 3.0]]
  (println "constraint" (pr-str v)
           "| token/dist:" (:weight (gfi/generate with-token [] (cm/from-map {:t v}) (rng/key 0)))
           "| categorical:" (:weight (gfi/generate with-categorical [] (cm/from-map {:t v}) (rng/key 0)))))
(println "row[3] =" (aget row 3))

(def tokenizer {:n-vocab 5 :bos nil :eos 4 :eog #{4} :add-bos? false
                :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab 5 :max-branches 16 :n-ctx 64}) tokenizer))
(println "a completion constrained to 3.0:"
         (reason #(gfi/generate (token/completion m) [[0 1] 2] (cm/from-map {0 3.0}) (rng/key 0))))
(shutdown-agents)
(System/exit 0)
