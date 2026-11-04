(ns gen.genjl.hmc-test
  "Gen.jl's test/inference/hmc.jl, ported: 0.4.8's smoke test of HMC
   (gen.inference.mcmc/hmc), and 0.4.9's metric tests (#559). Gen.jl runs
   each again on the same function annotated (grad), for a return-value
   gradient; gen.clj differentiates without an annotation, so there is one
   function. Gen.jl's metric is M, the momenta's covariance, a vector, a
   Diagonal or a dense matrix in the selection's to_array order; gen.clj's
   :inv-metric is M⁻¹ (Stan's), labeled by address: variances by address
   for a diagonal one, mcmc/dense-inv-metric for a dense one. Each port
   below gives gen.clj the inverse of Gen.jl's metric, and keys where
   Gen.jl reseeds its global RNG."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

(def ^:private foo (gen [] (trace :x (d/normal 0 1))))

(def ^:private bar
  (gen [] (let [x (trace :x (d/normal 0 1)) y (trace :y (d/normal 0 1))] (array/+ x y))))

(def ^:private xy (sel/select :x :y))

(defn- dense
  "Gen.jl's dense metric M (rows in x, y order) as gen.clj's: M⁻¹ labeled
   by address."
  [m]
  (mcmc/dense-inv-metric [:x :y] (la/inv (array/array m))))

(deftest hmc
  ;; hmc.jl:1: runs without error
  (let [tr (:trace (gfi/generate foo [] cm/EMPTY (rng/key 1)))]
    (is (trace/trace? (:trace (mcmc/hmc tr (sel/select :x) {:key (rng/key 2)})))))
  (testing "hmc.jl (0.4.9): for normal(0, 1), the gradient is -x"
    (let [tr (:trace (gfi/generate foo [] cm/EMPTY (rng/key 3)))
          {:keys [values gradients]} (grad/choice-gradients tr (sel/select :x))]
      (is (ts/close? (cm/to-array values) (array/negate (cm/to-array gradients))))))
  (testing "hmc.jl (0.4.9): smoke tests with a vector, a Diagonal and a dense metric"
    ;; metric [1.0, 2.0] and Diagonal([1.0, 2.0]) are one diagonal metric
    ;; in gen.clj; [0.5, 1.5] the second's
    (doseq [[label m] [["vector [1.0, 2.0]" {:x 1.0 :y 0.5}]
                       ["dense [1.0 0.1; 0.1 2.0]" (dense [[1.0 0.1] [0.1 2.0]])]
                       ["vector [0.5, 1.5]" {:x 2.0 :y (/ 1.0 1.5)}]
                       ["dense [0.5 0.2; 0.2 1.5]" (dense [[0.5 0.2] [0.2 1.5]])]]]
      (let [tr (:trace (gfi/generate bar [] cm/EMPTY (rng/key 4)))]
        (is (trace/trace? (:trace (mcmc/hmc tr xy {:inv-metric m :key (rng/key 5)}))) label))))
  (testing "hmc.jl (0.4.9): for two normal(0, 1), the gradients are -x and -y"
    (let [tr (:trace (gfi/generate bar [] cm/EMPTY (rng/key 6)))
          {:keys [values gradients]} (grad/choice-gradients tr xy)]
      (is (ts/close? (cm/to-array values) (array/negate (cm/to-array gradients)))))))

(deftest hmc-metric-behavior
  (let [tr (:trace (gfi/generate bar [] cm/EMPTY (rng/key 7)))]
    (testing "different metrics, the same draws: different moves"
      ;; Gen.jl reseeds its RNG before each; metric [10.0, 0.1]
      (let [move (fn [opts] (cm/to-map (:choices (:trace (mcmc/hmc tr xy (merge {:L 5 :key (rng/key 1)} opts))))))]
        (is (not= (ts/host (move nil)) (ts/host (move {:inv-metric {:x 0.1 :y 10.0}}))))))
    (testing "a Diagonal and the equal dense matrix: acceptance rates within 0.2 over 50 seeds"
      ;; gen.clj's dense form of a diagonal metric draws the same momenta and
      ;; moves the same way, so the rates agree closer than Gen.jl asks
      (let [rate (fn [m] (/ (count (filter :accepted? (map #(mcmc/hmc tr xy {:inv-metric m :key (rng/key %)}) (range 1 51))))
                            50.0))]
        (is (< (Math/abs (- (rate {:x 0.5 :y (/ 1.0 3.0)}) (rate (dense [[2.0 0.0] [0.0 3.0]])))) 0.2))))))

(deftest bad-metric-catches
  ;; Gen.jl's bad metrics, given to gen.clj as they are (they have no
  ;; inverse to port): each is refused loudly
  (let [tr (:trace (gfi/generate bar [] cm/EMPTY (rng/key 8)))]
    (doseq [[label m] [["bad dense [-1.0 -20.0; 0.0 1.0]" (mcmc/dense-inv-metric [:x :y] [[-1.0 -20.0] [0.0 1.0]])]
                       ["bad Diagonal([-1.0, -20.0])" {:x -1.0 :y -20.0}]
                       ["bad vector [-5.0, 20.0]" {:x -5.0 :y 20.0}]]]
      (is (= :invalid-inv-metric (:reason (ts/thrown-data (mcmc/hmc tr xy {:inv-metric m :key (rng/key 9)})))) label))))
