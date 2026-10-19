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

(deftest ^:slow hmc-targets-the-posterior
  (let [step (fn [tr key] (:trace (mcmc/hmc tr x-sel {:L 5 :eps 0.3} key)))]
    (is (scalar-stationary? step))
    (is (scalar-converges? step)))
  (testing "a vector latent behind a splice (a nested address)"
    (is (vector-converges? vector-model (cm/choicemap [:latent :z] (a/array [2.0 2.0]) :y y2)
                           #(% [:latent :z])
                           (fn [tr key] (:trace (mcmc/hmc tr (sel/select [:latent :z]) {:L 5 :eps 0.25} key)))))))

(deftest ^:slow mala-targets-the-posterior
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

(deftest keys-and-errors
  (let [tr (start-at scalar (cm/choicemap :x 0.2 :y 1.3))]
    (testing "hmc takes its key in the options map too"
      (is (= ((:trace (mcmc/hmc tr x-sel {:key (rng/key 5)})) :x) ((:trace (mcmc/hmc tr x-sel {:key (rng/key 5)})) :x))))
    (testing "a selection of no choices is an error, not a no-op accepted"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/hmc tr (sel/select :nope) {})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/mala tr (sel/select :nope) 0.1)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/map-optimize tr (sel/select :nope)))))
    (testing "mvnormal's shapes are checked"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mu must be a vector" (d/mvnormal 0.0 1.0)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"square matrix of its size" (d/mvnormal [0.0 0.0] (a/eye 3)))))
    (testing "elliptical slice of a missing address"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no choice at :nope"
                            (mcmc/elliptical-slice tr :nope (a/array [0.0]) (a/eye 1) (rng/key 1)))))))

(deftest map-optimization-finds-the-mode
  (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr x-sel)) (start-at scalar (cm/choicemap :x -2.0 :y 1.3)) (range 200))]
    (is (< (Math/abs (- (tr :x) 0.65)) 1e-6)))
  (testing "a vector mode behind a splice"
    (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr (sel/select [:latent :z]) {:max-step-size 0.3}))
                     (start-at vector-model (cm/choicemap [:latent :z] (a/array [2.0 2.0]) :y y2))
                     (range 300))]
      (is (a/allclose? (tr [:latent :z]) posterior-mean {:atol 1e-6})))))

(deftest ^:slow elliptical-slice-targets-the-posterior
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

(deftest ^:slow a-volume-changing-involution-targets-the-posterior
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

;; ---------------------------------------------------------------------------
;; Reversible jump: a split/merge move between dimensions (law 17)
;; ---------------------------------------------------------------------------

(def one-or-two
  "k = 1: one mean m ~ N(0,1); k = 2: two means m1, m2 ~ N(0,1). y ~ N(the
   average mean, 1), so P(k = 1 | y) ∝ N(y; 0, √2) and P(k = 2 | y) ∝
   N(y; 0, √1.5)."
  (gen []
    (let [mean (if (trace :two? (d/bernoulli 0.5))
                 (a/* 0.5 (a/+ (trace :m1 (d/normal 0 1)) (trace :m2 (d/normal 0 1))))
                 (trace :m (d/normal 0 1)))]
      (trace :y (d/normal mean 1)))))

(def split-merge-proposal
  "Split with an auxiliary u ~ N(0, 1); merging needs no auxiliary choice."
  (gen [tr] (when-not (tr :two?) (trace :u (d/normal 0 1)))))

(defn- split-merge
  "The involution (m, u) ↔ (m1, m2) = (m - u, m + u), with the dimension
   change and its Jacobian (computed by autodiff: |det| = 2 for a split)."
  [tr fwd _ _]
  (if (tr :two?)
    (let [m1 (tr :m1) m2 (tr :m2)
          h (fn [v] (let [a (a/take v 0) b (a/take v 1)] (a/stack [(a/* 0.5 (a/+ a b)) (a/* 0.5 (a/- b a))])))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :two? false :m (* 0.5 (+ m1 m2))))]
      {:trace t :bwd-choices (cm/choicemap :u (* 0.5 (- m2 m1)))
       :weight (a/+ w (grad/log-abs-det-jacobian h (a/array [m1 m2])))})
    (let [m (tr :m) u (cm/get-choice fwd :u)
          h (fn [v] (let [a (a/take v 0) b (a/take v 1)] (a/stack [(a/- a b) (a/+ a b)])))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :two? true :m1 (- m u) :m2 (+ m u)))]
      {:trace t :bwd-choices cm/EMPTY
       :weight (a/+ w (grad/log-abs-det-jacobian h (a/array [m u])))})))

(deftest ^:slow a-split-merge-move-targets-the-posterior
  (let [y 1.7
        p1 (Math/exp (dc/log-prob (d/normal 0 (Math/sqrt 2.0)) y))
        p2 (Math/exp (dc/log-prob (d/normal 0 (Math/sqrt 1.5)) y))
        exact-two (/ p2 (+ p1 p2))
        step (fn [tr key]
               (let [[k1 k2 k3] (rng/split key 3)
                     tr (:trace (mh/involutive-mh tr split-merge-proposal [] split-merge k1))
                     ;; within-model moves on the means
                     s (if (tr :two?) (sel/select :m1 :m2) (sel/select :m))]
                 (:trace (mh/mh (:trace (mh/mh tr s k2)) s k3))))
        start (start-at one-or-two (cm/choicemap :two? false :m 0.0 :y y))
        ends (map (fn [key] (reduce step start (rng/split key 60))) (rng/split (rng/key 41) 2000))
        frac (/ (count (filter #(% :two?) ends)) 2000.0)]
    (is (< (Math/abs (- frac exact-two)) (* 5 (Math/sqrt (/ (* exact-two (- 1 exact-two)) 2000)))))
    (testing "the involution is an involution, with opposite weights"
      (let [tr start fwd (cm/choicemap :u 0.4)
            {t1 :trace u1 :bwd-choices w1 :weight} (split-merge tr fwd nil nil)
            {t2 :trace u2 :bwd-choices w2 :weight} (split-merge t1 u1 nil nil)]
        (is (= (:choices tr) (:choices t2)))
        (is (= fwd u2))
        (is (< (Math/abs (+ w1 w2)) 1e-12))))))
