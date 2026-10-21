(ns gen.genjl.importance-sampling-test
  "Gen.jl 0.4.8's test/inference/importance_sampling.jl, ported:
   importance sampling and importance resampling, with the internal
   proposal and a custom one (gen.inference.importance)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.genjl.support :refer [approx?]]
            [gen.inference.importance :as imp]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private model
  (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))))

(def ^:private proposal (gen [] (trace :x (d/normal 0 2))))

(def ^:private y 2.0)

(def ^:private observations (cm/choicemap :y y))

(def ^:private n 4)

(defn- log-sum-exp [log-weights] (:log-total (w/normalize log-weights)))

(defn- observed? [tr] (= y (ts/host (tr :y))))

(deftest importance-sampling
  ;; importance_sampling.jl:1
  (doseq [[label opts] [["the internal proposal" {}] ["a custom proposal" {:proposal proposal :proposal-args []}]]]
    (testing label
      (let [{:keys [traces log-normalized-weights log-ml-estimate]}
            (imp/importance-sampling model [] observations n (assoc opts :key (rng/key 1)))]
        (is (= n (count traces)))
        (is (= n (count log-normalized-weights)))
        (is (approx? (log-sum-exp log-normalized-weights) 0.0 {:atol 1e-14}))
        (is (not (Double/isNaN log-ml-estimate)))
        (doseq [tr traces]
          (is (observed? tr))))))
  (doseq [[label opts] [["resampling, the internal proposal" {}]
                        ["resampling, a custom proposal" {:proposal proposal :proposal-args []}]]]
    (testing label
      (let [log-weights (:log-normalized-weights (imp/importance-sampling model [] observations n {:key (rng/key 2)}))
            {tr :trace lml :log-ml-estimate} (imp/importance-resampling model [] observations n (assoc opts :key (rng/key 3)))]
        ;; importance_sampling.jl:37 and :42 test the log weights of the
        ;; last importance_sampling call again (importance_resampling
        ;; returns none): ported as written
        (is (approx? (log-sum-exp log-weights) 0.0 {:atol 1e-14}))
        (is (not (Double/isNaN lml)))
        (is (observed? tr))))))
