;; V-P4-06 (verifier): token/dist's log density at integral and non-integral
;; floats, against categorical's, and through a completion's assess.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-P4-06.clj
(require '[gen.dist :as d]
         '[gen.dist.core :as dc]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy])
(def row (toy/toy-logprobs 5 [2 2]))
(def probs (mapv #(Math/exp %) row))
(doseq [v [1 1.0 1.5 (int 1) (float 1.0) 7 -1]]
  (println (pr-str v) (class v) "| token/dist:" (dc/log-prob (token/dist row) v)
           "| categorical:" (try (dc/log-prob (d/categorical probs) v) (catch Exception e [:error (.getMessage e)]))))
(shutdown-agents)
(System/exit 0)
