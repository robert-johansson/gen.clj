(ns gen.genjl.variational-test
  "Gen.jl 0.4.8's test/inference/variational.jl, ported: black-box
   variational inference (gen.inference.vi) fitting a Gaussian family to a
   model with no observations, whose posterior is its prior, and VIMCO's
   smoke runs. Gen.jl's GradientDescent and gen.clj's gradient-descent take
   the same decaying step, and both average the score-function gradient over
   the samples, as Gen.jl's black_box_vi! does.

   Gen.jl's step, GradientDescent(1., 1000) (variational.jl:26), diverges
   for about 1 key in 16 (measured: over 16 keys of its 2,000 iterations of
   100 samples, one run's σ collapsed to 0 in its first steps, a loud
   error), so a test that passed would pass by its key. The port takes half
   the step, under which no run of 24 keys (2,000 iterations) or 32 (200)
   diverged. At q = p the log weight is 0 for every sample, so the
   gradient's variance vanishes with the distance to the optimum: the slope
   converges, and the intercept's mean, whose ELBO is flatter, follows its
   schedule's remainder. The bands are 5 sds over those keys.

   On MLX, where every scalar is an array, the JVM's run is 25 million array
   operations (counted on the JVM): bbvi-run gives it one of 1.3 million."
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

(def ^:private targets
  "The prior's parameters, the posterior's (there are no observations)."
  {:slope-mu -1.0 :slope-log-std 0.5 :intercept-mu 1.0 :intercept-log-std 2.0})

(defn- remainder
  "What is left of the intercept mean's initial error after iters steps of
   gradient-descent from step: its ELBO gradient is (1 - μ)/e⁴, so each step
   keeps 1 - η_t/e⁴ of it, η_t = step·1001/(1000 + t)."
  [iters step]
  (reduce (fn [r t] (* r (- 1 (/ (* step (/ 1001.0 (+ 1000 t))) (Math/exp 4))))) 1.0 (range 1 (inc iters))))

(def ^:private bbvi-run
  "The fit's size and its bands, each 5 sds over keys at that size.
   - On the JVM, Gen.jl's 2,000 iterations of 100 samples. The intercept's
     mean ends at 1 - (remainder 2000 0.5), 4.1e-5 short (4.13e-5 over 24
     keys, sd 4.3e-6); the sds are 1.4e-7, 1.9e-7, 4.3e-6 and 4.2e-7.
   - On MLX, 200 iterations of 50. The intercept's mean is still on its way,
     at 1 - (remainder 200 0.5) = 0.813 (-0.188 over 32 keys); the sds are
     0.0016, 0.0021, 0.024 and 0.0065."
  (if (ts/jvm?)
    {:iters 2000 :samples 100 :expected (assoc targets :intercept-mu (- 1.0 (remainder 2000 0.5)))
     :atol {:slope-mu 7.5e-7 :slope-log-std 1e-6 :intercept-mu 2.2e-5 :intercept-log-std 2.1e-6}}
    {:iters 200 :samples 50 :expected (assoc targets :intercept-mu (- 1.0 (remainder 200 0.5)))
     :atol {:slope-mu 0.008 :slope-log-std 0.011 :intercept-mu 0.12 :intercept-log-std 0.033}}))

(def ^:private descent
  "variational.jl:26's GradientDescent(1., 1000) at half its step (see the
   namespace doc)."
  (learn/gradient-descent {:step-size-init 0.5 :step-size-beta 1000}))

(def ^:private vimco-run
  "VIMCO's smoke runs: Gen.jl's 50 iterations of 100 estimates on the JVM,
   5 of 5 on MLX."
  (if (ts/jvm?) {:iters 50 :estimates 100} {:iters 5 :estimates 5}))

(deftest ^:slow ^:heavy black-box-variational-inference
  ;; variational.jl:1
  (testing "BBVI recovers the prior's parameters"
    (let [{:keys [iters samples expected atol]} bbvi-run
          {:keys [store]} (vi/black-box-vi model [] cm/EMPTY approx [] start descent
                                           {:iters iters :samples samples :key (rng/key 1)})]
      (doseq [k [:slope-mu :slope-log-std :intercept-mu :intercept-log-std]]
        (is (genjl/approx? (k store) (k expected) {:atol (k atol)}) (name k)))))
  ;; variational.jl:40: smoke runs, from the start again. Gen.jl's update
  ;; goes on from BBVI's 2,000 steps; gen.clj's optimizer starts afresh.
  (doseq [baseline [:arithmetic :geometric]]
    (testing (str "VIMCO, " (name baseline) " baselines")
      (let [{:keys [iters estimates]} vimco-run
            {:keys [history]} (vi/black-box-vimco model [] cm/EMPTY approx [] start descent
                                                  {:iters iters :samples 20 :estimates estimates :baseline baseline
                                                   :key (rng/key 2)})]
        (is (and (= iters (count history)) (every? #(Double/isFinite (ts/number %)) history)))))))

;; Pending: variational.jl:56, the minimal VAE (3): BBVI and VIMCO training
;; the model's parameters with q's (model_update, G4, genclj-xjte).
