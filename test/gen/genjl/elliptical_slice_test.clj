(ns gen.genjl.elliptical-slice-test
  "Gen.jl 0.4.9's test/inference/elliptical_slice.jl, ported: a smoke test
   of elliptical slice sampling (gen.inference.mcmc/elliptical-slice), and
   0.4.9's two checks of its threshold (PR #571, the likelihood ratio in
   place of the update's whole weight, a fix gen.clj had from its port):
   with a constant likelihood the first ellipse point is accepted, its
   draws replayed from the same keys; on a conjugate model the chain's
   mean and variance are the posterior's. The stationary check's band is
   5 standard errors from the chain's effective sample size (this repo's
   convention), not Gen.jl's 0.05. Planted (2026-11-01): the threshold
   compared with the update's whole weight, as 0.4.8 did, fails the
   stationary check."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.trace :as trace]))

(def ^:private mu [0.0 0.0])

(def ^:private cov [[1.0 0.0] [0.0 1.0]])

(def ^:private foo
  (gen [] (let [x (trace :x (d/mvnormal mu cov))]
            (trace :y (d/mvnormal x [[0.01 0.0] [0.0 0.01]])))))

(deftest elliptical-slice
  ;; elliptical_slice.jl:1: runs without error
  (let [tr (:trace (gfi/generate foo [] (cm/choicemap :y (array/array [1.0 1.0])) (rng/key 1)))]
    (is (trace/trace? (:trace (mcmc/elliptical-slice tr :x mu cov (rng/key 2)))))))

(def ^:private prior-only
  (gen [] (trace :x (d/mvnormal mu cov))))

(deftest a-constant-likelihood-accepts-the-first-ellipse-point
  ;; 0.4.9: with no likelihood, every ellipse point is on the slice, so the
  ;; first, at theta, is accepted. Its draws replayed as elliptical-slice
  ;; makes them: nu, u, theta from (rng/split key 3)
  (doseq [seed (range 1 17)]
    (let [key (rng/key seed)
          tr (:trace (gfi/generate prior-only [] cm/EMPTY (rng/key (+ 100 seed))))
          [k-nu _ k-theta] (rng/split key 3)
          m (array/array mu)
          nu (array/- (dc/sample (d/mvnormal mu cov) k-nu) m)
          theta (* 2.0 Math/PI (double (array/item (rng/uniform (rng/fold-in k-theta 0)))))
          expected (array/+ (array/+ (array/* (array/- (tr :x) m) (Math/cos theta)) (array/* nu (Math/sin theta))) m)
          moved (:trace (mcmc/elliptical-slice tr :x mu cov key))]
      (is (array/allclose? expected (moved :x)) seed))))

(def ^:private conjugate
  (gen [] (let [x (trace :x (d/mvnormal [0.0] [[1.0]]))]
            (trace :y (d/normal (array/item x) 1.0)))))

(deftest ^:slow the-chain-is-the-posteriors
  ;; 0.4.9: x ~ N(0, 1), y ~ N(x, 1), y = 1: the posterior is N(1/2, 1/2)
  (let [start (:trace (gfi/generate conjugate [] (cm/choicemap :y 1.0) (rng/key 1)))
        draws (loop [i 0 tr start acc (transient [])]
                (if (= i 5000)
                  (persistent! acc)
                  (let [tr (:trace (mcmc/elliptical-slice tr :x [0.0] [[1.0]] (rng/fold-in (rng/key 2) i)))]
                    (recur (inc i) tr (conj! acc (double (array/item (tr :x))))))))
        n (count draws)
        m (/ (reduce + draws) n)
        v (/ (reduce + (map #(let [d (- % m)] (* d d)) draws)) (dec n))
        ess (diag/ess-mean [draws])
        se-mean (Math/sqrt (/ 0.5 ess))
        ;; the sample variance's standard error, Var(x)·sqrt(2/(ess - 1)) for a normal
        se-var (* 0.5 (Math/sqrt (/ 2.0 (dec ess))))]
    (testing "mean"
      (is (< (abs (- m 0.5)) (* 5 se-mean)) (pr-str {:mean m :ess ess})))
    (testing "variance"
      (is (< (abs (- v 0.5)) (* 5 se-var)) (pr-str {:variance v :ess ess})))))

