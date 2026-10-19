(ns gen.inference.vi-test
  "Black-box VI and VIMCO on a conjugate model, whose posterior is known:
   mu ~ N(0, 1), y ~ N(mu, 1), y = 1.3 gives mu | y ~ N(0.65, √½) and
   log p(y) = log N(1.3; 0, √2). The variational family q = N(m, e^log-s)
   holds the posterior, where the ELBO and the K-sample bound both equal
   log p(y). The estimators' means are checked against exact gradients (the
   ELBO's in closed form, the K-sample bound's by reparameterization), which
   carries the power; the fits check that the loops ascend."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.inference.vi :as vi]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def model (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)) mu)))
(def observations (cm/choicemap :y 1.3))
(def family (gen [] (trace :mu (d/normal (param :m) (array/exp (param :log-s))))))

(def posterior {:m 0.65 :log-s (Math/log (Math/sqrt 0.5))})
(def start {:m -1.0 :log-s 0.5})
(def evidence (ts/number (dc/log-prob (d/normal 0 (Math/sqrt 2)) 1.3)))

(defn- mean-and-se [xs]
  (let [xs (mapv double xs)
        n (count xs)
        mu (/ (reduce clojure.core/+ xs) n)]
    [mu (Math/sqrt (/ (reduce clojure.core/+ (map #(Math/pow (- % mu) 2) xs)) (* n (dec n))))]))

(defn- agrees?
  "Do two estimates [mean se] agree within five standard errors?"
  [[a sa] [b sb]]
  (< (Math/abs (- a b)) (* 5 (Math/sqrt (+ (* sa sa) (* sb sb))))))

(defn- average-of-last [n xs] (/ (reduce clojure.core/+ (take-last n xs)) n))

(deftest at-the-posterior-the-bounds-are-the-evidence
  (doseq [baseline [nil :leave-one-out]]
    (is (ts/close? evidence (:objective (vi/elbo-gradient model [] observations family [] posterior 8 {:baseline baseline :key (rng/key 1)}))
                   {:rtol 1e-6 :atol 1e-6})
        "every log weight is log p(y)"))
  (doseq [baseline [:geometric :arithmetic]]
    (is (ts/close? evidence (:objective (vi/vimco-gradient model [] observations family [] posterior 8 {:baseline baseline :key (rng/key 2)}))
                   {:rtol 1e-6 :atol 1e-6})))
  (testing "elsewhere the ELBO is below the evidence"
    (let [elbos (map #(:objective (vi/elbo-gradient model [] observations family [] start 50 {:key %}))
                     (rng/split (rng/key 3) 20))]
      (is (< (/ (reduce clojure.core/+ elbos) 20) (- evidence 1.0))))))

(deftest ^:slow ^:heavy the-elbo-gradient-is-unbiased
  ;; ELBO(m, s) = E_q[log p(mu, y)] + H(q), so ∂/∂m = 1.3 - 2m and
  ;; ∂/∂log-s = 1 - 2s²: (3.3, -4.44) at the start
  (let [exact {:m (- 1.3 (* 2 (:m start))) :log-s (- 1.0 (* 2 (Math/exp (* 2 (:log-s start)))))}]
    (doseq [baseline [nil :leave-one-out]]
      (let [grads (map #(:grads (vi/elbo-gradient model [] observations family [] start 20 {:baseline baseline :key %}))
                       (rng/split (rng/key 4) 2000))]
        (doseq [k [:m :log-s]]
          (is (agrees? (mean-and-se (map (comp ts/number k) grads)) [(exact k) 0.0])
              (str baseline " " k ": " (mean-and-se (map (comp ts/number k) grads)) " against " (exact k))))))))

(defn- reparameterized-gradient
  "Draws of the K-sample bound's gradient at {:m :log-s} by
   reparameterization, z = m + s·ε: log w = log p(z, y) - log q(z) has
   ∂/∂z = 1.3 - 2z through the joint and ∂/∂log-s = 1 through q's
   normalizer, so ∂ log (1/K) Σ w_k = Σ w̃_k ∂ log w_k. {:m draws :log-s draws}
   from n sets of k draws."
  [{:keys [m log-s]} k n key]
  (let [s (Math/exp log-s)
        e (dc/sample (d/normal 0 1) key [n k])
        z (+ m (* s e))
        dz (- 1.3 (* 2.0 z))
        lw (+ (* -0.5 (array/square z)) (* -0.5 (array/square (- 1.3 z))) (* 0.5 (array/square e)))
        w (array/softmax lw {:axis 1})]
    {:m (ts/host (array/sum (* w dz) {:axis 1}))
     :log-s (ts/host (array/sum (* w (+ 1.0 (* dz (* s e)))) {:axis 1}))}))

(deftest ^:slow ^:heavy the-vimco-gradient-is-unbiased
  (let [reference (reparameterized-gradient start 10 100000 (rng/key 5))]
    (doseq [baseline [:geometric :arithmetic]]
      (let [grads (map #(:grads (vi/vimco-gradient model [] observations family [] start 10 {:baseline baseline :key %}))
                       (rng/split (rng/key 6) 2000))]
        (doseq [k [:m :log-s]]
          (let [estimate (mean-and-se (map (comp ts/number k) grads))]
            (is (agrees? estimate (mean-and-se (reference k)))
                (str baseline " " k ": " estimate " against " (mean-and-se (reference k))))))))))

;; The fits' bands are five standard deviations of the final values over 16
;; keys (Adam at 0.02, 800 iterations), measured: Gen.jl's estimator ends
;; with sd 0.041 in m and 0.031 in log-s; the leave-one-out baseline is
;; exactly zero at the posterior, where every log weight is log p(y), so it
;; stops there.
(deftest black-box-vi-finds-the-posterior
  (let [{:keys [store history]} (vi/black-box-vi model [] observations family [] start (learn/adam {:lr 0.02})
                                                  {:iters 800 :samples 20 :key (rng/key 7)})]
    (is (< (ts/abs (- (:m store) (:m posterior))) 0.21) (str store))
    (is (< (ts/abs (- (:log-s store) (:log-s posterior))) 0.16) (str store))
    (is (< (ts/abs (- (average-of-last 100 history) evidence)) 0.01) "the ELBO reaches the evidence"))
  (testing "with a leave-one-out baseline"
    (let [{:keys [store history]} (vi/black-box-vi model [] observations family [] start (learn/adam {:lr 0.02})
                                                    {:iters 800 :samples 20 :baseline :leave-one-out :key (rng/key 7)})]
      (is (< (ts/abs (- (:m store) (:m posterior))) 1e-3) (str store))
      (is (< (ts/abs (- (:log-s store) (:log-s posterior))) 1e-3) (str store))
      (is (< (ts/abs (- (last history) evidence)) 1e-3)))))

;; The K-sample bound is flat near the posterior (at the start its gradient
;; in m is 0.16 for K = 10, the ELBO's 3.3), so VIMCO's iterates wander
;; there. With K = 4 (Adam at 0.01, 1500 iterations), over 16 keys the
;; final m has sd 0.079, log-s sd 0.064, and the last 100 bounds average
;; 0.005 below the evidence with sd 0.006; the bound at the start is 0.53
;; below. The fit checks that the bound climbs; the gradient test checks
;; the estimator.
(deftest vimco-ascends-the-bound
  (let [at-start (map #(:objective (vi/vimco-gradient model [] observations family [] start 4 {:key %}))
                      (rng/split (rng/key 8) 400))
        {:keys [store history]} (vi/black-box-vimco model [] observations family [] start (learn/adam {:lr 0.01})
                                                    {:iters 1500 :samples 4 :key (rng/key 8)})]
    (is (< (average-of-last 400 at-start) (- evidence 0.25)) "the start is far below")
    (is (< (ts/abs (- (average-of-last 100 history) evidence)) 0.035) (str store))
    (is (< (ts/abs (- (:m store) (:m posterior))) 0.4) (str store))
    (is (< (ts/abs (- (:log-s store) (:log-s posterior))) 0.34) (str store))))

(deftest models-latents-outside-q-come-from-the-model
  ;; q proposes only :mu; :nuisance comes from the model, and the weight
  ;; stays an importance weight of the extended q, so the ELBO is unchanged
  (let [model' (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :nuisance (d/normal 0 1)) (trace :y (d/normal mu 1)) mu))]
    (is (ts/close? evidence (:objective (vi/elbo-gradient model' [] observations family [] posterior 8 {:key (rng/key 9)}))
                   {:rtol 1e-6 :atol 1e-6}))))
