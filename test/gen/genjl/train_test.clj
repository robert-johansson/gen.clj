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
   derived beside it."
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

(deftest ^:slow ^:heavy sgd-learns-the-teacher
  ;; train.jl:109
  (let [{:keys [store]} (learn/train student start sgd teacher-data grad/param-gradients
                                     {:epochs 2000 :batches-per-epoch 1 :batch-size 50 :key (rng/key 3)})]
    ;; p(x | z = 0) = p(x | z = 1) = 0.5
    (is (genjl/approx? (:theta1 store) 0.0 {:atol (* 5 theta1-sd)}))
    ;; y = xor(x, z) given both, so θ2 to θ5 grow without bound: e^|θ| gains
    ;; about 0.01 per example of its case, 25,000 in 2,000 steps, and |θ|
    ;; ends at 5.50 (sd 0.008 over 12 keys), 60 sd past the thresholds
    (is (< (ts/number (:theta2 store)) -5))
    (is (> (ts/number (:theta3 store)) 5))
    (is (> (ts/number (:theta4 store)) 5))
    (is (< (ts/number (:theta5 store)) -5))))

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

(def ^:private optimum
  "train.jl:155's -0.21: the best expected score in q's family. z - x + 2
   is p's noise, so θ = -2 and σ = 0.3: -½ log 2π - log 0.3 - ½ = -0.2149."
  (- (* -0.5 (Math/log (* 2 Math/PI))) (Math/log 0.3) 0.5))

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

(defn- scores-near-the-optimum?
  "train.jl:155, isapprox(score, -0.21, atol=5e-2): the mean of 10,000
   lectures' scores under store, within 5 of its standard errors of the
   optimum. The scores' sd is 0.69, so the error is 0.007 (Gen.jl's atol
   is 7). Training leaves the expected score below the optimum by about
   5e-4: θ's spread after training is 0.01 (over 6 keys), and
   (Δθ)²/(2σ²) = 5e-4."
  [store key]
  (let [scores (mapv #(ts/number (:score (learn/lecture p [] q store x-of %))) (rng/split key 10000))
        n (count scores)
        mean (/ (reduce + scores) n)
        se (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) scores)) (* n (dec n))))]
    (genjl/approx? mean optimum {:atol (* 5 se)})))

(deftest ^:slow ^:heavy lecture-trains-q-toward-the-posterior
  ;; train.jl:126
  (testing "lecture"
    (is (scores-near-the-optimum?
         (trained (fn [store k] (grad/mean (map #(:grads (learn/lecture p [] q store x-of %)) (rng/split k 1000))))
                  (rng/key 4))
         (rng/key 5))))
  (testing "lecture-batched"
    ;; train.jl:176 scores q, the first part's proposal, not q_batched (U4,
    ;; docs/research/genjl-test-suite.md §4.3), so Gen.jl's assertion holds
    ;; whatever lecture_batched! taught q_batched (which, stepping 0.001
    ;; along the sum of 1,000 gradients, ten times the first part's step,
    ;; stops far from the optimum). The port checks what the test means:
    ;; lecture-batched, at the first part's step, trains q to the optimum.
    (is (scores-near-the-optimum? (trained #(:grads (learn/lecture-batched p [] q %1 x-of 1000 %2)) (rng/key 6))
                                  (rng/key 7)))))
