(ns gen.inference.particle-filter-test
  "Particle filtering on a linear-Gaussian state-space model, against the
   exact Kalman filter: x_t ~ N(a·x_{t-1}, q), y_t ~ N(x_t, r), x_{-1} = 0."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.particle-filter :as pf]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def a-coef 0.8)
(def q 1.0)
(def r 0.5)
(def ys [0.3 1.1 0.4 -0.6 0.2 1.4])

(def kernel
  (gen [t prev]
    (let [x (trace :x (d/normal (a/* a-coef prev) q))]
      (trace :y (d/normal x r))
      x)))

(def model (comb/unfold kernel))

(defn- observation [t] (cm/choicemap [t :y] (ys t)))

(def kalman
  "[log p(y_0..y_t) and the filtering mean of x_t] for each t."
  (rest (reductions (fn [[lml m p] y]
                      (let [m- (* a-coef m)
                            p- (+ (* a-coef a-coef p) (* q q))
                            s (+ p- (* r r))
                            k (/ p- s)]
                        [(+ lml (dc/log-prob (d/normal m- (Math/sqrt s)) y))
                         (+ m- (* k (- y m-)))
                         (* (- 1 k) p-)]))
                    [0.0 0.0 0.0]
                    ys)))

(def optimal-proposal
  "The locally optimal proposal p(x_t | x_{t-1}, y_t)."
  (gen [tr t y]
    (let [prev (if (zero? t) 0.0 (tr [(dec t) :x]))
          m- (* a-coef prev)
          v (/ 1.0 (+ (/ 1.0 (* q q)) (/ 1.0 (* r r))))
          m (* v (+ (/ m- (* q q)) (/ y (* r r))))]
      (trace [t :x] (d/normal m (Math/sqrt v))))))

(def initial-proposal
  "The optimal proposal at t = 0, where x_{-1} = 0."
  (gen [y]
    (let [v (/ 1.0 (+ (/ 1.0 (* q q)) (/ 1.0 (* r r))))]
      (trace [0 :x] (d/normal (* v (/ y (* r r))) (Math/sqrt v))))))

(defn- run-filter
  "A filter over every observation, resampling adaptively; the final state."
  [n key & [{:keys [proposal?]}]]
  (let [ks (rng/split key (* 2 (count ys)))
        init (pf/initialize model [1 0.0] (observation 0) n
                            (cond-> {:key (ks 0)}
                              proposal? (assoc :proposal initial-proposal :proposal-args [(ys 0)])))]
    (reduce (fn [state t]
              (-> state
                  (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                  (pf/step [(inc t) 0.0] [diff/unknown-change diff/no-change] (observation t)
                           (cond-> {:key (ks (* 2 t))}
                             proposal? (assoc :proposal optimal-proposal :proposal-args [t (ys t)])))))
            init
            (range 1 (count ys)))))

(defn- exact-lml [] (first (last kalman)))

(deftest the-estimate-approaches-the-exact-marginal-likelihood
  (let [state (run-filter 2000 (rng/key 1))]
    ;; the estimate's sd is about sqrt(12.6/n) for this model and filter (from
    ;; the variance of runs of 20 particles): four of them at n = 2000 is 0.32
    (is (< (Math/abs (- (pf/log-ml-estimate state) (exact-lml))) 0.32))
    (testing "the filtering mean"
      (let [xs (map #(% [(dec (count ys)) :x]) (pf/sample-unweighted-traces state 2000 (rng/key 2)))]
        (is (< (Math/abs (- (/ (reduce + xs) 2000) (second (last kalman)))) 0.1))))))

(defn- unbiased? [proposal?]
  ;; law 20: E[exp(estimate)] = p(y), checked against the runs' own spread
  (let [runs 400
        ratios (map #(Math/exp (- (pf/log-ml-estimate (run-filter 20 % {:proposal? proposal?})) (exact-lml)))
                    (rng/split (rng/key (if proposal? 4 3)) runs))
        mean (/ (reduce + ratios) runs)
        sd (Math/sqrt (/ (reduce + (map #(* (- % mean) (- % mean)) ratios)) (dec runs)))]
    (< (Math/abs (- mean 1.0)) (* 5 (/ sd (Math/sqrt runs))))))

(deftest the-estimate-is-unbiased-in-exp-space
  (is (unbiased? false) "with the internal proposal")
  (is (unbiased? true) "with the locally optimal custom proposal"))

(deftest the-optimal-proposal-reduces-variance
  (let [spread (fn [proposal?]
                 (let [ests (map #(pf/log-ml-estimate (run-filter 20 % {:proposal? proposal?}))
                                 (rng/split (rng/key 5) 100))
                       m (/ (reduce + ests) 100)]
                   (/ (reduce + (map #(* (- % m) (- % m)) ests)) 99)))]
    (is (< (spread true) (spread false)))))

(deftest resampling
  (let [state (pf/initialize model [1 0.0] (observation 0) 100 {:key (rng/key 6)})
        resampled (pf/resample state {:key (rng/key 7)})]
    (is (= (vec (repeat 100 0.0)) (:log-weights resampled)))
    (is (a/allclose? (pf/log-ml-estimate state) (pf/log-ml-estimate resampled))
        "resampling moves the estimate into :log-ml and leaves it unchanged")
    (is (every? (set (:traces state)) (:traces resampled)))
    (is (false? (:resampled? (pf/maybe-resample resampled {:key (rng/key 8)}))) "equal weights: no resampling")
    (is (true? (:resampled? (pf/maybe-resample state {:ess-threshold 101 :key (rng/key 8)}))))))

(deftest a-step-may-only-add-choices
  (let [state (pf/initialize model [1 0.0] (observation 0) 10 {:key (rng/key 9)})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed or removed"
                          (pf/step state [2 0.0] [diff/unknown-change diff/no-change]
                                   (cm/choicemap [0 :y] 5.0 [1 :y] 1.0) {:key (rng/key 10)})))))

(deftest the-incremental-weight-is-the-update-weight
  ;; law 20: with the internal proposal, a step adds log p(y_t | x_t)
  (let [state (pf/initialize model [1 0.0] (observation 0) 20 {:key (rng/key 20)})
        state' (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1) {:key (rng/key 21)})]
    (is (every? true? (map (fn [w w' tr] (< (Math/abs (- (- w' w) (gfi/project tr (sel/select [1 :y])))) 1e-12))
                           (:log-weights state) (:log-weights state') (:traces state'))))))

(deftest a-custom-proposal-in-a-batched-filter-runs-per-particle
  (let [state (pf/initialize model [1 0.0] (observation 0) 30 {:batched? true :key (rng/key 30)})
        state' (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1)
                        {:proposal optimal-proposal :proposal-args [1 (ys 1)] :key (rng/key 31)})]
    (is (contains? state :particles))
    (is (not (contains? state' :particles)) "the batched state became per-particle traces")
    (is (= 30 (count (pf/traces state'))))
    (is (every? #(= (ys 1) (% [1 :y])) (pf/traces state')))
    (is (every? #(Double/isFinite %) (pf/log-weights state')))
    (testing "and initialize with a proposal ignores :batched?"
      (is (contains? (pf/initialize model [1 0.0] (observation 0) 5
                                    {:batched? true :proposal initial-proposal :proposal-args [(ys 0)] :key (rng/key 32)})
                     :traces)))))
