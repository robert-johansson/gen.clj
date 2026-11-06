(ns gen.inference.device-test
  "The :device option without the lowerings (ext/device is not on the core
   tests' classpath): a run falls back to the handler, naming why, and its
   result is the handler's; run-chains' batched fallback, which a
   combinator's kernel takes with the lowerings too."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.device :as device]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.parity-models :as pm]
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
        (is (= (:log-weights (is/importance-sampling model args obs 20 {:key (rng/key 2)})) (:log-weights i))))
      (testing "importance sampling, batched: the fallback is the batched run (genclj-cxrx)"
        (let [r (is/importance-sampling model args obs 20 {:key (rng/key 2) :device :jvm :batched? true})]
          (is (re-find #"not on the classpath" (:reason (:device-fallback r))))
          (is (b/particles? (:particles r)))
          (is (not (contains? r :traces)))
          (is (= (:log-weights (is/importance-sampling model args obs 20 {:key (rng/key 2) :batched? true}))
                 (:log-weights r))))))))

(defn- mean [xs] (/ (reduce + 0.0 xs) (count xs)))

(defn- standard-error
  "The standard error of the mean of xs (independent draws)."
  [xs]
  (let [m (mean xs)]
    (Math/sqrt (/ (/ (reduce + 0.0 (map #(let [d (- (double %) m)] (* d d)) xs)) (dec (count xs))) (count xs)))))

(defn- same-mean?
  "Do two samples of independent chains' statistics have the same mean, to
   5 standard errors of the difference?"
  [xs ys]
  (< (Math/abs (- (mean xs) (mean ys)))
     (* 5.0 (Math/sqrt (+ (Math/pow (standard-error xs) 2) (Math/pow (standard-error ys) 2))))))

(deftest run-chains-batched-fallback-has-kernel-run-s-statistics
  ;; genclj-qucg: the device refuses a combinator's kernel (MH reads its
  ;; decision on the host), so run-chains runs it as one batched run of its
  ;; batched form, on other draws than kernel/run per chain's: the chains'
  ;; acceptance rates (accepted moves over every move) and divergences
  ;; have kernel/run's distribution, so their means over 24 chains agree
  ;; within 5 standard errors. HMC's eps at the leapfrog's limit (0.117,
  ;; twice the posterior's narrowest sd) makes some proposals diverge;
  ;; at 0.1 none do, and HMC accepts often, so a rate per step and one per
  ;; move differ. Planted bugs (2026-11-06, 1.2.0's): the rate of steps in
  ;; which any move was accepted fails the rates (0.82 against 0.46, 29
  ;; standard errors); the batched moves' :diverged dropped fails the
  ;; divergences (none at all)
  (let [[model args obs] (pm/models :vector)
        n 24
        starts (mapv #(:trace (gfi/generate model args obs (rng/key (+ 100 %)))) (range n))
        opts {:samples 30 :key (rng/key 9)}
        both (fn [eps]
               (let [k (kernel/chain (kernel/mh (sel/select :intercept)) (kernel/hmc pm/selection {:L 8 :eps eps}))]
                 [(mapv #(kernel/run k %1 (assoc opts :key %2)) starts (rng/split (rng/key 9) n))
                  (device/run-chains k starts (assoc opts :device :jvm))]))]
    (testing "acceptance rates, per move"
      (let [[per-chain r] (both 0.1)]
        (is (= :batched (:ran (:device-fallback r))) (:device-fallback r))
        (is (= n (count (:acceptance-rates r))))
        (is (same-mean? (map :acceptance-rate per-chain) (:acceptance-rates r))
            [(mean (map :acceptance-rate per-chain)) (mean (:acceptance-rates r))])))
    (testing "divergences"
      (let [[per-chain r] (both 0.117)]
        (is (= :batched (:ran (:device-fallback r))) (:device-fallback r))
        (is (= n (count (:divergences r))))
        (is (pos? (reduce + (map :divergences per-chain))))
        (is (pos? (reduce + (:divergences r))))
        (is (same-mean? (map :divergences per-chain) (:divergences r))
            [(mean (map :divergences per-chain)) (mean (:divergences r))])))))
