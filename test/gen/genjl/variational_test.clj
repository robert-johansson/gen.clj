(ns gen.genjl.variational-test
  "Gen.jl 0.4.8's test/inference/variational.jl, ported: black-box
   variational inference (gen.inference.vi) fitting a Gaussian family to a
   model with no observations, whose posterior is its prior, and VIMCO's
   smoke runs. Gen.jl's GradientDescent and gen.clj's gradient-descent take
   the same decaying step, and both average the score-function gradient over
   the samples, so the fit is Gen.jl's.

   The band is Gen.jl's atol, 0.001. At q = p the log weight is 0 for every
   sample, so the gradient's variance vanishes with the distance to the
   optimum and the fit converges to it: over 8 keys, no parameter ended
   farther than 2e-9 from its target (1.6e-11 to 1.9e-9 by parameter)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.genjl.support :as genjl]
            [gen.inference.vi :as vi]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private model
  ;; variational.jl:5
  (gen []
    (trace :slope (d/normal -1 (Math/exp 0.5)))
    (trace :intercept (d/normal 1 (Math/exp 2.0)))))

(def ^:private approx
  ;; variational.jl:10
  (gen []
    (trace :slope (d/normal (param :slope-mu) (array/exp (param :slope-log-std))))
    (trace :intercept (d/normal (param :intercept-mu) (array/exp (param :intercept-log-std))))))

(def ^:private start {:slope-mu 0.0 :slope-log-std 0.0 :intercept-mu 0.0 :intercept-log-std 0.0})

(def ^:private update
  ;; variational.jl:26: GradientDescent(1., 1000)
  (learn/gradient-descent {:step-size-init 1.0 :step-size-beta 1000}))

(deftest ^:slow ^:heavy black-box-variational-inference
  ;; variational.jl:1
  (testing "BBVI recovers the prior's parameters"
    (let [{:keys [store]} (vi/black-box-vi model [] cm/EMPTY approx [] start update
                                           {:iters 2000 :samples 100 :key (rng/key 1)})]
      (is (genjl/approx? (:slope-mu store) -1.0 {:atol 0.001}))
      (is (genjl/approx? (:slope-log-std store) 0.5 {:atol 0.001}))
      (is (genjl/approx? (:intercept-mu store) 1.0 {:atol 0.001}))
      (is (genjl/approx? (:intercept-log-std store) 2.0 {:atol 0.001}))))
  ;; variational.jl:40: smoke runs, from the start again. Gen.jl's update
  ;; goes on from BBVI's 2,000 steps; gen.clj's optimizer starts afresh.
  (doseq [baseline [:arithmetic :geometric]]
    (testing (str "VIMCO, " (name baseline) " baselines")
      (let [{:keys [history]} (vi/black-box-vimco model [] cm/EMPTY approx [] start update
                                                  {:iters 50 :samples 20 :estimates 100 :baseline baseline
                                                   :key (rng/key 2)})]
        (is (and (= 50 (count history)) (every? #(Double/isFinite (ts/number %)) history)))))))

;; Pending: variational.jl:56, the minimal VAE (3): BBVI and VIMCO training
;; the model's parameters with q's (model_update, G4, genclj-xjte).
