;; P2-05: parameters the families accept but cannot sample or score
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng] '[gen.array :as array])

(println "neg-binom(2, 0):" (try (let [dd (d/neg-binom 2.0 0.0) v (dc/sample dd (rng/key 1))]
                                   [:accepted :draw v :log-prob (dc/log-prob dd v)])
                                 (catch Exception e [:error (ex-message e)])))
(println "geometric(0) for contrast:" (try (d/geometric 0.0) (catch Exception e [:error (ex-message e)])))
(println "neg-binom(2, 1e-300):" (let [dd (d/neg-binom 2.0 1e-300) v (dc/sample dd (rng/key 1))] [:draw v :log-prob (dc/log-prob dd v)]))
;; piecewise-uniform with a batch of bin layouts (the value-shape method allows it)
(println "piecewise-uniform with 2-d parameters:"
         (try (let [dd (d/piecewise-uniform [[0.0 1.0 2.0] [0.0 2.0 4.0]] [[0.5 0.5] [0.25 0.75]])]
                [:value-shape (dc/value-shape dd)
                 :sample (dc/sample dd (rng/key 2))
                 :log-prob-of [0.5 3.0] (dc/log-prob dd (array/array [0.5 3.0]))
                 :exact [(Math/log 0.5) (Math/log (/ 0.75 2.0))]])
              (catch Exception e [:error (ex-message e)])))
;; array parameters that host checks would reject (rule 5 reads concrete arrays)
(doseq [[label f] {"uniform [1] [0] (low above high)" #(d/uniform (array/array [1.0]) (array/array [0.0]))
                   "poisson [-1]" #(d/poisson (array/array [-1.0]))
                   "uniform-discrete [3] [1]" #(d/uniform-discrete (array/array [3]) (array/array [1]))
                   "binom [2.5] 0.5" #(d/binom (array/array [2.5]) 0.5)
                   "poisson -1 (host, for contrast)" #(d/poisson -1.0)}]
  (println label "->" (try (let [dd (f) v (dc/sample dd (rng/key 3))] [:accepted :draw v :log-prob (dc/log-prob dd v)])
                           (catch Exception e [:error (ex-message e)]))))
