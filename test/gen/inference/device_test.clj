(ns gen.inference.device-test
  "The :device option without the lowerings (ext/device is not on the core
   tests' classpath): a run falls back to the handler, naming why, and its
   result is the handler's."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private model
  (gen [xs]
    (let [s (trace :s (d/normal 0 10))]
      (trace :y (d/normal (array/* s xs) 1.0)))))

(def ^:private lowerings? (some? (try (requiring-resolve 'gen.device/compile-program) (catch Exception _ nil))))

(deftest without-the-lowerings-a-run-is-the-handlers
  (if lowerings?
    (println "gen.inference.device-test: ext/device is on the classpath, the fallback test skips")
    (let [args [(array/array [1.0 2.0 3.0])]
          obs (cm/choicemap :y (array/array [2.1 3.9 6.2]))
          tr (:trace (gfi/generate model args obs (rng/key 0)))
          k (kernel/mala (sel/select :s) 0.01)
          opts {:samples 5 :key (rng/key 1)}
          r (kernel/run k tr (assoc opts :device :jvm))
          i (is/importance-sampling model args obs 20 {:key (rng/key 2) :device :jvm})]
      (testing "kernel/run"
        (is (re-find #"not on the classpath" (:reason (:device-fallback r))))
        (is (= (map :score (:traces (kernel/run k tr opts))) (map :score (:traces r)))))
      (testing "importance sampling"
        (is (re-find #"not on the classpath" (:reason (:device-fallback i))))
        (is (= (:log-weights (is/importance-sampling model args obs 20 {:key (rng/key 2)})) (:log-weights i)))))))
