(ns gen.inference.mcmc-test
  "HMC, MALA, MAP optimization and elliptical slice sampling on Gaussian
   targets with exact posteriors, including an array-valued latent behind a
   splice; and involutive MH with an AD Jacobian (thesis Eq 3.15)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as a]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Targets
;; ---------------------------------------------------------------------------

(def scalar
  "x ~ N(0, 1), y ~ N(x, 1); y = 1.3 gives x | y ~ N(0.65, √½)."
  (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))))

(def prior-cov (a/array [[1.0 0.5] [0.5 1.0]]))
(def noise-cov (a/array [[0.5 0.0] [0.0 0.5]]))
(def y2 (a/array [1.0 -0.5]))

(def latent (gen [] (trace :z (d/mvnormal [0.0 0.0] prior-cov))))

(def vector-model
  "z ~ N(0, Σ0) inside a splice, y ~ N(z, Σ1)."
  (gen [] (let [z (splice :latent latent)] (trace :y (d/mvnormal z noise-cov)))))

(def flat-vector-model
  (gen [] (let [z (trace :z (d/mvnormal [0.0 0.0] prior-cov))] (trace :y (d/mvnormal z noise-cov)))))

(def posterior-cov (la/inv (a/+ (la/inv prior-cov) (la/inv noise-cov))))
(def posterior-mean (a/matmul posterior-cov (a/matmul (la/inv noise-cov) y2)))

(defn- start-at [model choices] (:trace (gfi/generate model [] choices (rng/key 0))))

(defn- moments [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]))

(defn- matches? [xs mean var]
  (let [[m v] (moments xs)]
    (and (< (Math/abs (- m mean)) (* 5 (Math/sqrt (/ var (count xs)))))
         (< (Math/abs (- v var)) (* 0.12 var)))))

(defn- scalar-stationary? [step]
  (let [n 4000
        [k1 k2] (rng/split (rng/key 21))
        starts (map #(start-at scalar (cm/choicemap :x % :y 1.3))
                    (a/->clj (a/+ 0.65 (a/* (Math/sqrt 0.5) (rng/normal k1 [n])))))]
    (matches? (map #((step %1 %2) :x) starts (rng/split k2 n)) 0.65 0.5)))

(defn- scalar-converges? [step]
  (let [start (start-at scalar (cm/choicemap :x -2.0 :y 1.3))]
    (matches? (map (fn [key] ((reduce step start (rng/split key 60)) :x)) (rng/split (rng/key 22) 1000))
              0.65 0.5)))

(defn- vector-converges?
  "Do chains of step from a fixed start reach the posterior of z (read by
   z-of) in both coordinates?"
  [model choices z-of step]
  (let [start (start-at model choices)
        ends (map (fn [key] (a/->clj (z-of (reduce step start (rng/split key 60))))) (rng/split (rng/key 23) 1000))]
    (every? true? (for [i [0 1]]
                    (matches? (map #(nth % i) ends)
                              (a/item (a/take posterior-mean i))
                              (a/item (a/take (a/take posterior-cov i) i)))))))

;; ---------------------------------------------------------------------------
;; HMC and MALA
;; ---------------------------------------------------------------------------

(def x-sel (sel/select :x))

(deftest hmc-targets-the-posterior
  (let [step (fn [tr key] (:trace (mcmc/hmc tr x-sel {:L 5 :eps 0.3} key)))]
    (is (scalar-stationary? step))
    (is (scalar-converges? step)))
  (testing "a vector latent behind a splice (a nested address)"
    (is (vector-converges? vector-model (cm/choicemap [:latent :z] (a/array [2.0 2.0]) :y y2)
                           #(% [:latent :z])
                           (fn [tr key] (:trace (mcmc/hmc tr (sel/select [:latent :z]) {:L 5 :eps 0.25} key)))))))

(deftest mala-targets-the-posterior
  (let [step (fn [tr key] (:trace (mcmc/mala tr x-sel 0.3 key)))]
    (is (scalar-stationary? step))
    (is (scalar-converges? step)))
  (is (vector-converges? vector-model (cm/choicemap [:latent :z] (a/array [2.0 2.0]) :y y2)
                         #(% [:latent :z])
                         (fn [tr key] (:trace (mcmc/mala tr (sel/select [:latent :z]) 0.15 key))))))

(deftest acceptance-ratios
  (let [tr (start-at scalar (cm/choicemap :x 0.2 :y 1.3))
        rs (map #(mcmc/hmc tr x-sel {:L 3 :eps 0.2} %) (rng/split (rng/key 1) 100))]
    (is (every? #(or (:accepted? %) (identical? tr (:trace %))) rs))
    (is (> (count (filter :accepted? rs)) 80) "small steps on a smooth target are mostly accepted")))

;; ---------------------------------------------------------------------------
;; MAP and elliptical slice
;; ---------------------------------------------------------------------------

(deftest map-optimization-finds-the-mode
  (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr x-sel)) (start-at scalar (cm/choicemap :x -2.0 :y 1.3)) (range 200))]
    (is (< (Math/abs (- (tr :x) 0.65)) 1e-6)))
  (testing "a vector mode behind a splice"
    (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr (sel/select [:latent :z]) {:max-step-size 0.3}))
                     (start-at vector-model (cm/choicemap [:latent :z] (a/array [2.0 2.0]) :y y2))
                     (range 300))]
      (is (a/allclose? (tr [:latent :z]) posterior-mean {:atol 1e-6})))))

(deftest elliptical-slice-targets-the-posterior
  (is (vector-converges? flat-vector-model (cm/choicemap :z (a/array [2.0 2.0]) :y y2)
                         #(% :z)
                         (fn [tr key] (mcmc/elliptical-slice tr :z (a/array [0.0 0.0]) prior-cov key)))))

;; ---------------------------------------------------------------------------
;; Involutive MH with a Jacobian
;; ---------------------------------------------------------------------------

(def positive (gen [] (trace :x (d/gamma 3 1))))

(def log-scale (gen [tr] (trace :u (d/normal 0 0.5))))

(defn- scale-move
  "(x, u) ↦ (x·e^u, -u), an involution with |det J| = e^u, its log Jacobian
   computed by autodiff."
  [tr fwd _ _]
  (let [x (tr :x)
        u (cm/get-choice fwd :u)
        h (fn [v] (let [x (a/take v 0) u (a/take v 1)] (a/stack [(a/* x (a/exp u)) (a/negate u)])))
        {t :trace w :weight} (gfi/update tr (cm/choicemap :x (* x (Math/exp u))))]
    {:trace t :bwd-choices (cm/choicemap :u (- u))
     :weight (a/+ w (grad/log-abs-det-jacobian h (a/array [x u])))}))

(deftest a-volume-changing-involution-targets-the-posterior
  (let [step (fn [tr key] (:trace (mh/involutive-mh tr log-scale [] scale-move key)))
        start (start-at positive (cm/choicemap :x 1.0))
        ends (map (fn [key] ((reduce step start (rng/split key 80)) :x)) (rng/split (rng/key 31) 1000))]
    (is (matches? ends 3.0 3.0) "Gamma(3, 1): mean 3, variance 3"))
  (testing "without the Jacobian the chain is biased, so the test can tell"
    (let [wrong (fn [tr fwd r args] (assoc (scale-move tr fwd r args) :weight
                                           (:weight (gfi/update tr (cm/choicemap :x (* (tr :x) (Math/exp (cm/get-choice fwd :u))))))))
          step (fn [tr key] (:trace (mh/involutive-mh tr log-scale [] wrong key)))
          start (start-at positive (cm/choicemap :x 1.0))
          ends (map (fn [key] ((reduce step start (rng/split key 80)) :x)) (rng/split (rng/key 31) 1000))]
      (is (not (matches? ends 3.0 3.0))))))
