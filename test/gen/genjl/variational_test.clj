(ns gen.genjl.variational-test
  "Gen.jl 0.4.8's test/inference/variational.jl, ported: black-box
   variational inference (gen.inference.vi) fitting a Gaussian family to a
   model with no observations, whose posterior is its prior, VIMCO's smoke
   runs, and the minimal VAE, training a model's parameter with q's
   (:model-store, Gen.jl's model_update). Gen.jl's GradientDescent and
   gen.clj's gradient-descent take the same decaying step, and both average
   the score-function gradient over the samples, as Gen.jl's black_box_vi!
   does.

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
   operations (counted on the JVM): bbvi-run gives it one of 1.3 million,
   and vae-run the VAE's 5.8 million in place of 172 million."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
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
  (reduce (fn [r t] (clojure.core/* r (clojure.core/- 1 (clojure.core// (clojure.core/* step (clojure.core// 1001.0 (clojure.core/+ 1000 t))) (Math/exp 4)))))
          1.0 (range 1 (inc iters))))

(def ^:private bbvi-run
  "The fit's size and its bands, each 5 sds over keys at that size.
   - On the JVM, Gen.jl's 2,000 iterations of 100 samples. The intercept's
     mean ends at 1 - (remainder 2000 0.5), 4.1e-5 short (4.13e-5 over 24
     keys, sd 4.3e-6); the sds are 1.4e-7, 1.9e-7, 4.3e-6 and 4.2e-7.
   - On MLX, 200 iterations of 50. The intercept's mean is still on its way,
     at 1 - (remainder 200 0.5) = 0.813 (-0.188 over 32 keys); the sds are
     0.0016, 0.0021, 0.024 and 0.0065."
  (if (ts/jvm?)
    {:iters 2000 :samples 100 :expected (assoc targets :intercept-mu (clojure.core/- 1.0 (remainder 2000 0.5)))
     :atol {:slope-mu 7.5e-7 :slope-log-std 1e-6 :intercept-mu 2.2e-5 :intercept-log-std 2.1e-6}}
    {:iters 200 :samples 50 :expected (assoc targets :intercept-mu (clojure.core/- 1.0 (remainder 200 0.5)))
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

;; variational.jl:56, the minimal VAE: z_i ~ normal(θ, 1), x_i ~ normal(z_i, 1),
;; and q, z_i ~ normal(c_1 + c_2 x_i, e^log-std), which holds the posterior,
;; normal((x_i + θ) / 2, √½), so the joint optimum is θ = mean(xs), the
;; marginal likelihood's maximum, with the ELBO at log p(xs).

(def ^:private n 10)

(def ^:private vae-model
  (gen []
    (doseq [i (range n)]
      (trace [:x i] (d/normal (trace [:z i] (d/normal (param :theta) 1)) 1)))))

(def ^:private vae-approx
  (gen [xs]
    (let [c (param :mu-coeffs)]
      (doseq [i (range n)]
        (trace [:z i] (d/normal (+ (array/take c 0) (* (array/take c 1) (nth xs i))) (array/exp (param :log-std))))))))

(def ^:private xs
  "variational.jl:81: z_i ~ normal(-1, 1), x_i ~ normal(z_i, 1)."
  (mapv ts/number (:retval (gfi/simulate (gen [] (mapv #(trace [:x %] (d/normal (trace [:z %] (d/normal -1 1)) 1)) (range n)))
                                         [] (rng/key 0)))))

(def ^:private vae-observations (apply cm/choicemap (mapcat (fn [i] [[:x i] (xs i)]) (range n))))

(def ^:private opt-theta (clojure.core// (reduce clojure.core/+ xs) n))

(def ^:private log-marginal-likelihood
  "variational.jl:92: p(z, xs) / p(z | xs) = p(xs) for every z, with z
   from the posterior at opt-theta."
  (let [optimum-approx (gen [] (doseq [i (range n)]
                                 (trace [:z i] (d/normal (clojure.core// (clojure.core/+ (xs i) opt-theta) 2.0) (Math/sqrt 0.5)))))
        approx-trace (gfi/simulate optimum-approx [] (rng/key 1))
        {model-trace :trace} (gfi/generate (dyn/with-params vae-model {:theta opt-theta}) []
                                           (cm/merge (:choices approx-trace) vae-observations) (rng/key 2))]
    (ts/number (- (:score model-trace) (:score approx-trace)))))

(def ^:private vae-start {:mu-coeffs (array/zeros [2]) :log-std 0.0})

(def ^:private vae-run
  "The VAE's sizes and θ's band, its bias plus 5 sds over training keys,
   measured. Gen.jl's rtol of 0.1 for the last ELBO estimate and VIMCO's
   last bound holds at both sizes with room.
   - On the JVM, Gen.jl's. Over keys 1 to 8, BBVI ends with θ 0.049 above
     opt-theta, sd 0.034 (3,000 steps of q's 1e-4 do not quite converge),
     so the band is 0.22: Gen.jl's atol of 0.1 is only 1.5 sd above the
     bias (key 3 ends 0.114 off). The last ELBO estimate is within
     -0.002 ± 0.004 of log p(xs) relatively, VIMCO's bound -0.0003 ± 0.0022
     (bias plus 5 sds: 0.023, 0.012).
   - On MLX, a tenth of the iterations at ten times the learning rates, the
     same descent in fewer steps, and fewer samples (BBVI's 8, VIMCO's 3
     estimates of K = 10). BBVI takes the leave-one-out baseline, which
     Gen.jl's test does not: without it θ spreads 0.22 sd at this size, a
     band of 1.1, nearly all of θ's way from its start (1.17). Over keys 1
     to 32, θ ends 0.040 above opt-theta, sd 0.024, so the band is 0.16;
     the ELBO estimate is within -0.002 ± 0.004, VIMCO's bound
     -0.001 ± 0.006 (bias plus 5 sds: 0.024, 0.031)."
  (if (ts/jvm?)
    {:bbvi {:iters 3000 :samples 20 :q-lr 0.0001 :model-lr 0.002}
     :vimco {:iters 1000 :samples 10 :estimates 10 :q-lr 0.001 :model-lr 0.01}
     :theta-atol 0.22}
    {:bbvi {:iters 300 :samples 8 :baseline :leave-one-out :q-lr 0.001 :model-lr 0.02}
     :vimco {:iters 100 :samples 10 :estimates 3 :q-lr 0.01 :model-lr 0.1}
     :theta-atol 0.16}))

(defn- train-vae
  "q and the model's θ trained from Gen.jl's start by fit (vi/black-box-vi
   or vi/black-box-vimco) at one of vae-run's sizes."
  [fit {:keys [q-lr model-lr] :as size}]
  (fit vae-model [] vae-observations vae-approx [xs] vae-start (learn/sgd {:lr q-lr})
       (assoc (dissoc size :q-lr :model-lr)
              :key (rng/key 1) :model-store {:theta 0.0} :model-optimizer (learn/sgd {:lr model-lr}))))

(deftest ^:slow ^:heavy minimal-vae
  (let [{:keys [bbvi vimco theta-atol]} vae-run]
    ;; variational.jl:107, BBVI with the score-function estimator
    (let [{:keys [model-store history]} (train-vae vi/black-box-vi bbvi)]
      (is (genjl/approx? (:theta model-store) opt-theta {:atol theta-atol}) (str model-store " against " opt-theta))
      (is (genjl/approx? (peek history) log-marginal-likelihood {:rtol 0.1})))
    ;; variational.jl:122, VIMCO
    (let [{:keys [history]} (train-vae vi/black-box-vimco vimco)]
      (is (genjl/approx? (peek history) log-marginal-likelihood {:rtol 0.1})))))
