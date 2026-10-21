(ns gen.genjl.train-test
  "Gen.jl 0.4.8's test/inference/train.jl, ported: a student model's
   parameter gradients, summed over a minibatch of a teacher's data, against
   central finite differences of assess (train.jl:82), which
   gen.genjl.finite-diff takes in float64 on the JVM backend; SGD on the
   teacher's data (train.jl:109); and lecture! and lecture_batched!
   training a proposal toward the posterior (train.jl:126).

   Gen.jl accumulates gradients into each parameter
   (accumulate_param_gradients!, get_param_grad) and an update steps along
   their sum; gen.clj's gradients are values (D7), and its train and
   lecture-batched average them, so the ports scale Gen.jl's steps by the
   batch size. The statistical bands come from each estimate's variance,
   derived beside it. The statistical runs are Gen.jl's on the JVM and
   smaller on MLX, where every scalar is an array (sgd-run, lecture-run),
   each with bands for its size."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as genjl]
            [gen.gradients :as grad]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private teacher
  ;; train.jl:24: z = xor(x, y)
  (gen [] (let [x (trace :x (d/bernoulli 0.5))
                y (trace :y (d/bernoulli 0.5))]
            (trace :z (d/bernoulli (array/where (array/!= x y) 1.0 0.0))))))

(def ^:private student
  ;; train.jl:33: y's probability by the case of z and x
  (gen [z]
    (let [x (trace :x (d/bernoulli (array/sigmoid (param :theta1))))
          prob-y (array/where z
                              (array/where x (array/sigmoid (param :theta2)) (array/sigmoid (param :theta3)))
                              (array/where x (array/sigmoid (param :theta4)) (array/sigmoid (param :theta5))))]
      (trace :y (d/bernoulli prob-y)))))

(def ^:private names [:theta1 :theta2 :theta3 :theta4 :theta5])

(def ^:private start (zipmap names (repeat 0.0)))

(defn- teacher-data
  "train.jl:52, data_generator: [inputs constraints] from one teacher run."
  [key]
  (let [{:keys [choices]} (gfi/propose teacher [] key)]
    [[(cm/get-choice choices :z)] (cm/choicemap :x (cm/get-choice choices :x) :y (cm/get-choice choices :y))]))

(defn- total-log-density
  "train.jl:86: Σ over the minibatch of assess of the student, as a
   function of its five parameters."
  [minibatch]
  (fn [& thetas]
    (let [student (dyn/with-params student (zipmap names thetas))]
      (reduce + (map (fn [[inputs constraints]]
                       (ts/number (:weight (gfi/assess student (ts/host inputs) (cm/from-map (ts/host constraints))))))
                     minibatch)))))

(deftest parameter-gradients-match-finite-differences
  ;; train.jl:73, a minibatch of 100
  (let [minibatch (mapv teacher-data (rng/split (rng/key 1) 100))
        gradients (grad/weighted-sum (repeat 1.0)
                                     (map (fn [[inputs constraints]]
                                            (grad/param-gradients
                                             (:trace (gfi/generate (dyn/with-params student start) inputs constraints
                                                                   (rng/key 2)))))
                                          minibatch))
        density (total-log-density minibatch)]
    (doseq [[i theta] (map-indexed vector names)]
      (testing (str theta)
        (is (genjl/approx? (theta gradients) (fd/finite-diff density (mapv start names) i) {:atol 1e-4}))))))

;; ---------------------------------------------------------------------------
;; train.jl:109, SGD on the teacher's data (statistical)
;; ---------------------------------------------------------------------------

(def ^:private sgd
  "train.jl:109: GradientDescent(0.01, 1000000) along the sum of a
   minibatch's 50 gradients, so 0.5 along their mean."
  (learn/gradient-descent {:step-size-init (* 0.01 50) :step-size-beta 1000000}))

(def ^:private theta1-sd
  "θ1's spread about 0 (p(x | z) = ½) once trained: each step adds
   0.01 Σ (x - σ(θ1)) over the 50 examples, an AR(1) with coefficient
   1 - 0.01·50·σ'(0) = 0.875 and noise sd 0.01·√(50/4), whose stationary sd
   is 0.073 (0.081 over 12 keys). Gen.jl's atol 0.2 is 2.7 of them."
  (let [rho (- 1 (* 0.01 50 0.25))]
    (/ (* 0.01 (Math/sqrt (/ 50 4.0))) (Math/sqrt (- 1 (* rho rho))))))

(def ^:private sgd-run
  "The epochs and θ2-θ5's threshold. y = xor(x, z) given both, so θ2 to θ5
   grow without bound: u = e^|θ| follows du = 0.01·count·u/(1 + u), count
   about 12.5 a step, so u + log u = 1 + 0.125·epochs.
   - On the JVM, Gen.jl's 2,000 epochs and ±5: |θ| = 5.50 (sd 0.008 over
     12 keys), 60 sd past them.
   - On MLX, where every scalar is an array, 30 epochs of the same steps
     (340,000 array operations, counted on the JVM, where Gen.jl's run makes
     22 million): |θ| = 1.25 (1.25 to 1.27, sd at most 0.035 over 20 keys),
     so 1.07 is 5 sd inside it. θ1 is stationary after 30 steps as after
     2,000 (0.875³⁰ = 0.02)."
  (if (ts/jvm?) {:epochs 2000 :threshold 5.0} {:epochs 30 :threshold 1.07}))

(deftest ^:slow ^:heavy sgd-learns-the-teacher
  ;; train.jl:109
  (let [{:keys [epochs threshold]} sgd-run
        {:keys [store]} (learn/train student start sgd teacher-data grad/param-gradients
                                     {:epochs epochs :batches-per-epoch 1 :batch-size 50 :key (rng/key 3)})]
    ;; p(x | z = 0) = p(x | z = 1) = 0.5
    (is (genjl/approx? (:theta1 store) 0.0 {:atol (* 5 theta1-sd)}))
    ;; y | z, x = xor(x, z)
    (is (< (ts/number (:theta2 store)) (- threshold)))
    (is (> (ts/number (:theta3 store)) threshold))
    (is (> (ts/number (:theta4 store)) threshold))
    (is (< (ts/number (:theta5 store)) (- threshold)))))

;; ---------------------------------------------------------------------------
;; train.jl:126, lecture! (statistical)
;; ---------------------------------------------------------------------------

(def ^:private p
  ;; train.jl:131
  (gen [] (let [z (trace :z (d/normal 0.0 1.0))] (trace :x (d/normal (+ z 2.0) 0.3)))))

(def ^:private q
  ;; train.jl:138
  (gen [x] (trace :z (d/normal (+ x (param :theta)) (array/exp (param :log-std))))))

(defn- x-of [tr] [(tr :x)])

(defn- expected-score
  "The expected score of q under store, E log q(z; x + θ, e^s) over p: z - x
   + 2 is p's noise, N(0, 0.3), so -½ log 2π - s - (0.09 + (θ + 2)²)/(2e^{2s}).
   Its maximum, train.jl:155's -0.21, is at θ = -2 and s = log 0.3:
   -½ log 2π - log 0.3 - ½ = -0.2149."
  [{:keys [theta log-std]}]
  (let [theta (ts/number theta) s (ts/number log-std)]
    (- (* -0.5 (Math/log (* 2 Math/PI))) s (/ (+ 0.09 (Math/pow (+ theta 2) 2)) (* 2 (Math/exp (* 2 s)))))))

(def ^:private lecture-run
  "Lectures a step, scores to estimate with, and the trained store's
   spread: θ's and log σ's sd about -2 and log 0.3 after the 100 steps of
   0.1 along a mean of b lectures' gradients. θ's gradient, (z - x - θ)/σ²,
   has sd 1/0.3 at the optimum and pulls with 0.1/σ² = 1.11 (an AR(1) of
   coefficient -0.11); log σ's, (z - x - θ)²/σ² - 1, has sd √2 and pulls with
   0.2 (coefficient 0.8).
   - On the JVM, Gen.jl's 1,000 lectures a step and 10,000 scores: sds 0.0105
     and 0.0075 (0.0096 and 0.0057 over 6 keys).
   - On MLX, 10 and 500 (139,000 array operations a part, counted on the
     JVM): sds 0.106 and 0.089, the larger of derived and measured (0.082 and
     0.089 over 20 keys; derived 0.106 and 0.075)."
  (if (ts/jvm?)
    {:batch 1000 :scores 10000 :theta-sd 0.0105 :log-std-sd 0.0075}
    {:batch 10 :scores 500 :theta-sd 0.106 :log-std-sd 0.089}))

(def ^:private lecture-step
  "train.jl:147: FixedStepGradientDescent(1e-4) along the sum of 1,000
   lectures' gradients, so 0.1 along their mean."
  (learn/sgd {:lr (* 1e-4 1000)}))

(defn- trained
  "q's store after train.jl:148's 100 steps from θ = 0 and log σ = 0, each
   along (gradient store key)."
  [gradient key]
  (reduce (fn [store k] (first (learn/step lecture-step nil store (gradient store k))))
          {:theta 0.0 :log-std 0.0}
          (rng/split key 100)))

(defn- check-trained
  "train.jl:155, isapprox(score, -0.21, atol=5e-2), as its two claims: the
   trained store is near the optimum, within 5 of its sds (lecture-run), and
   lecture's score estimates the store's expected score, the mean of
   lecture-run's scores within 5 of its standard errors (on the JVM the
   scores' sd is 0.69 and the error 0.007; Gen.jl's atol is 7 of them)."
  [store key]
  (let [{:keys [scores theta-sd log-std-sd]} lecture-run
        xs (mapv #(ts/number (:score (learn/lecture p [] q store x-of %))) (rng/split key scores))
        n (count xs)
        mean (/ (reduce + xs) n)
        se (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) xs)) (* n (dec n))))]
    (is (genjl/approx? (:theta store) -2.0 {:atol (* 5 theta-sd)}) "θ")
    (is (genjl/approx? (:log-std store) (Math/log 0.3) {:atol (* 5 log-std-sd)}) "log σ")
    (is (genjl/approx? mean (expected-score store) {:atol (* 5 se)}) "the score")))

(deftest ^:slow ^:heavy lecture-trains-q-toward-the-posterior
  ;; train.jl:126
  (testing "lecture"
    (check-trained (trained (fn [store k] (grad/mean (map #(:grads (learn/lecture p [] q store x-of %))
                                                          (rng/split k (:batch lecture-run)))))
                            (rng/key 4))
                   (rng/key 5)))
  (testing "lecture-batched"
    ;; train.jl:176 scores q, the first part's proposal, not q_batched (U4,
    ;; docs/research/genjl-test-suite.md §4.3), so Gen.jl's assertion holds
    ;; whatever lecture_batched! taught q_batched (which, stepping 0.001
    ;; along the sum of 1,000 gradients, ten times the first part's step,
    ;; stops far from the optimum). The port checks what the test means:
    ;; lecture-batched, at the first part's step, trains q to the optimum.
    (check-trained (trained #(:grads (learn/lecture-batched p [] q %1 x-of (:batch lecture-run) %2)) (rng/key 6))
                   (rng/key 7))))
