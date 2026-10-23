(ns gen.inference.particle-mcmc-test
  "Conditional SMC, particle Gibbs and PMMH on linear-Gaussian state-space
   models, x_t ~ N(a·x_{t-1}, q), y_t ~ N(x_t, r) (q and r standard
   deviations), x_{-1} = 0: the states against their exact smoothing
   distribution (forward filtering, backward sampling, and the
   Rauch-Tung-Striebel smoother), a on a grid against its exact posterior
   (the Kalman likelihood at each point)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.kernel :as k]
            [gen.inference.particle-filter :as pf]
            [gen.inference.particle-mcmc :as pmcmc]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

;; The states move slowly against noisy observations, so the smoothing
;; distribution is far from the filter's, and a filter of a few particles
;; draws far from it: what lets the tests below see planted bugs
(def q 0.5)
(def r 1.0)

(def ssm
  "The state-space model, args (n init a)."
  (comb/unfold (gen [_t prev a]
                 (let [x (trace :x (d/normal (* a prev) q))]
                   (trace :y (d/normal x r))
                   x))))

;; ---------------------------------------------------------------------------
;; The exact answers
;; ---------------------------------------------------------------------------

(defn- kalman
  "The Kalman filter: for each t, the mean and variance of x_t given
   y_0 … y_t, and log p(y_0 … y_t), host numbers."
  [a ys]
  (vec (rest (reductions (fn [[m p l] y]
                           (let [m- (* a m)
                                 p- (+ (* a a p) (* q q))
                                 s (+ p- (* r r))
                                 k (/ p- s)]
                             [(+ m- (* k (- y m-))) (* (- 1 k) p-)
                              (+ l (ts/number (dc/log-prob (d/normal m- (Math/sqrt s)) y)))]))
                         [0.0 0.0 0.0]
                         ys))))

(defn- gain
  "The smoother's gain at a filtered variance p: the regression of x_t on
   x_{t+1} given y_0 … y_t."
  [a p]
  (/ (* a p) (+ (* a a p) (* q q))))

(defn- smoothed
  "[means variances covariances] of p(x | y): each x_t's mean and variance,
   and the covariance of x_t with x_{t+1}."
  [a ys]
  (let [f (kalman a ys)
        [m-last p-last] (peek f)]
    (loop [t (- (count f) 2) ms (list m-last) ps (list p-last) cs ()]
      (if (neg? t)
        [(vec ms) (vec ps) (vec cs)]
        (let [[m p] (f t)
              j (gain a p)]
          (recur (dec t)
                 (conj ms (+ m (* j (- (first ms) (* a m)))))
                 (conj ps (+ p (* j j (- (first ps) (+ (* a a p) (* q q))))))
                 (conj cs (* j (first ps)))))))))

(defn- posterior-draw
  "x_0 … x_{T-1} drawn from p(x | y) exactly: the filter forward, then each
   x_t given x_{t+1} backward."
  [a ys key]
  (let [f (kalman a ys)
        zs (ts/host (rng/normal key [(count f)]))
        [m-last p-last] (peek f)]
    (loop [t (- (count f) 2) xs (list (+ m-last (* (Math/sqrt p-last) (peek zs))))]
      (if (neg? t)
        (vec xs)
        (let [[m p] (f t)
              j (gain a p)]
          (recur (dec t) (conj xs (+ m (* j (- (first xs) (* a m))) (* (Math/sqrt (- p (* j a p))) (zs t))))))))))

(defn- mean [xs] (/ (reduce + xs) (count xs)))

;; ---------------------------------------------------------------------------
;; Conditional SMC on the states
;; ---------------------------------------------------------------------------

(def a 0.9)
(def ys [0.3 1.1 0.4 -0.6 0.2 1.4])
(def n-steps (count ys))

(def observations (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) ys)))

(def latents (sel/from-paths (for [t (range n-steps)] [t :x])))

(def steps (mapv (fn [t] {:args [(inc t) 0.0 a]}) (range n-steps)))

(def initial-proposal
  "The locally optimal proposal at t = 0, where x_{-1} = 0: p(x_0 | y_0)."
  (gen [y]
    (let [v (/ 1.0 (+ (/ 1.0 (* q q)) (/ 1.0 (* r r))))]
      (trace [0 :x] (d/normal (* v (/ y (* r r))) (Math/sqrt v))))))

(def optimal-proposal
  "The locally optimal proposal p(x_t | x_{t-1}, y_t)."
  (gen [tr t y]
    (let [v (/ 1.0 (+ (/ 1.0 (* q q)) (/ 1.0 (* r r))))]
      (trace [t :x] (d/normal (* v (+ (/ (* a (tr [(dec t) :x])) (* q q)) (/ y (* r r)))) (Math/sqrt v))))))

(def optimal-steps
  (into [(assoc (first steps) :proposal initial-proposal :proposal-args [(ys 0)])]
        (map (fn [t] (assoc (steps t) :proposal optimal-proposal :proposal-args [t (ys t)])))
        (range 1 n-steps)))

(defn- trajectory
  "A trace of the model at the states xs, with the observations."
  [xs]
  (:trace (gfi/generate ssm [(count xs) 0.0 a]
                        (reduce (fn [c [t x]] (cm/set-choice c [t :x] x)) observations (map-indexed vector xs))
                        (rng/key 0))))

(def some-states [0.4 0.9 0.1 -0.8 0.3 1.0])

(defn- states [tr] (mapv #(ts/number (tr [% :x])) (range n-steps)))

(defn- check-smoothing
  "Checks that the draws' means of x_t, x_t² and x_t·x_{t+1} are within
   five standard errors of p(x | y)'s, the errors of independent draws: for
   jointly normal x and y with covariance c, Var(x²) = 2v² + 4μ²v and
   Var(xy) = v_x·v_y + c² + μ_x²·v_y + μ_y²·v_x + 2μ_x·μ_y·c."
  [draws]
  (let [[ms vs cs] (smoothed a ys)
        n (count draws)
        near (fn [what f mu v]
               (let [m (mean (map f draws))]
                 (is (< (ts/abs (- m mu)) (* 5 (Math/sqrt (/ v n)))) (str what ": " m " against " mu))))]
    (doseq [t (range n-steps)
            :let [mu (ms t) v (vs t)]]
      (near (str "E[x_" t "]") #(% t) mu v)
      (near (str "E[x_" t "²]") #(* (% t) (% t)) (+ v (* mu mu)) (+ (* 2 v v) (* 4 mu mu v))))
    (doseq [t (range (dec n-steps))
            :let [[m0 m1 v0 v1 c] [(ms t) (ms (inc t)) (vs t) (vs (inc t)) (cs t)]]]
      (near (str "E[x_" t "·x_" (inc t) "]") #(* (% t) (% (inc t))) (+ c (* m0 m1))
            (+ (* v0 v1) (* c c) (* m0 m0 v1) (* m1 m1 v0) (* 2 m0 m1 c))))))

(defn- after-a-move
  "The states after one particle Gibbs move of n particles over
   filter-steps, from an exact draw of p(x | y), for each of m keys."
  [filter-steps n opts m key]
  (let [kernel (pmcmc/particle-gibbs latents filter-steps n opts)]
    (mapv (fn [key]
            (let [[k-draw k-move] (rng/split key)]
              (states (:trace (kernel (trajectory (posterior-draw a ys k-draw)) k-move)))))
          (rng/split key m))))

;; From exact draws of p(x | y), one move of three particles must leave them
;; exact: 3000 moves for each kind of filter, 51 statistics in all.
;; Planted bugs (2026-10-23), each caught on the JVM backend:
;; - the reference dropped from resampling (particle 0 drawn as the others
;;   are, then given the reference's later states): 6 statistics fail, all
;;   when resampling at every step (E[x_3] 0.116 against 0.200), and
;;   the-reference-survives-resampling fails;
;; - the new reference drawn uniformly, not from the final weights: 27 fail,
;;   in every block;
;; - the reference weighed with its states' density (no project): 9 fail,
;;   and the-reference-is-weighed-… fails;
;; - with the proposal, the reference's weight without its density (no
;;   assess): 3 fail (E[x_4] 0.244 against 0.342), and
;;   the-reference-is-weighed-… fails;
;; - the other particles not observing (constrained by nothing): 44 fail.
;; A move that changes nothing would pass: the next section's chains catch
;; it.
(deftest ^:slow ^:heavy conditional-smc-leaves-the-smoothing-distribution-invariant
  (testing "resampling when the ESS falls below n/2"
    (check-smoothing (after-a-move steps 3 {} 3000 (rng/key 1))))
  (testing "resampling at every step, as Andrieu et al."
    (check-smoothing (after-a-move steps 3 {:ess-threshold ##Inf} 3000 (rng/key 2))))
  (testing "with the locally optimal proposal"
    (check-smoothing (after-a-move optimal-steps 3 {} 3000 (rng/key 3)))))

(deftest the-reference-is-weighed-as-a-particle-drawn-at-its-states
  ;; without resampling (an ESS threshold of 0) the reference's weight sums
  ;; its incremental weights: log p(y_t | x_t) with the internal proposal,
  ;; and log p(y_t | x_{t-1}) with the locally optimal one
  (let [xs some-states
        reference (trajectory xs)
        weight (fn [filter-steps] (first (:log-weights (pmcmc/conditional-smc reference latents filter-steps 4
                                                                              {:ess-threshold 0.0 :key (rng/key 10)}))))
        prev (fn [t] (if (zero? t) 0.0 (xs (dec t))))]
    (is (ts/close? (reduce + (map #(dc/log-prob (d/normal %1 r) %2) xs ys)) (weight steps)))
    (is (ts/close? (reduce + (map (fn [t] (dc/log-prob (d/normal (* a (prev t)) (Math/sqrt (+ (* q q) (* r r)))) (ys t)))
                                  (range n-steps)))
                   (weight optimal-steps)))))

(deftest the-reference-survives-resampling
  ;; a reference far from the data weighs almost nothing at every step;
  ;; resampled at every step, it stays particle 0, with its own lineage
  (let [reference (trajectory [4.0 -4.0 4.0 -4.0 4.0 -4.0])
        state (pmcmc/conditional-smc reference latents steps 10 {:ess-threshold ##Inf :key (rng/key 11)})]
    (is (= (ts/host (:choices reference)) (ts/host (:choices (first (:traces state))))))
    (is (= 0 (first (:parents state))))
    (is (every? pos? (rest (:parents state))) "the others descend from particles of more weight")))

(deftest a-move-is-determined-by-its-key
  (let [reference (trajectory some-states)
        sweep (fn [key] (let [s (pmcmc/conditional-smc reference latents steps 5 {:key key})]
                          [(mapv (comp ts/host :choices) (:traces s)) (ts/host (:log-weights s))]))
        kernel (pmcmc/particle-gibbs latents steps 5)
        move (fn [key] (ts/host (:choices (:trace (kernel reference key)))))]
    (is (= (sweep (rng/key 12)) (sweep (rng/key 12))))
    (is (not= (sweep (rng/key 12)) (sweep (rng/key 13))))
    (is (= (move (rng/key 14)) (move (rng/key 14))))
    (is (k/reversible? kernel) "a particle Gibbs move is its own reversal")))

(deftest invalid-references-are-loud
  (let [reference (trajectory some-states)
        message (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-message e))))]
    (is (re-find #"reference is a trace" (message #(pmcmc/conditional-smc observations latents steps 5))))
    (is (re-find #"two particles" (message #(pmcmc/conditional-smc reference latents steps 1))))
    (is (re-find #"non-empty vector of maps" (message #(pmcmc/conditional-smc reference latents (map :args steps) 5))))
    (testing "a reference the steps do not end at: a shorter one, or one of another model"
      (is (re-find #"last step's :args" (message #(pmcmc/conditional-smc reference latents (pop steps) 5))))
      (is (re-find #"last step's :args"
                   (message #(pmcmc/conditional-smc (gfi/simulate (gen [n] (trace :x (d/normal n 1))) [3] (rng/key 15))
                                                    latents steps 5)))))
    (testing "a model whose step removes a choice"
      (let [model (gen [n] (trace (if (< n 2) :early :late) (d/normal 0 1)) (trace :y (d/normal 0 1)))]
        (is (re-find #"changed or removed"
                     (message #(pmcmc/conditional-smc (gfi/simulate model [2] (rng/key 16)) (sel/select :early :late)
                                                      [{:args [1]} {:args [2]}] 5))))))
    (testing "encapsulated randomness, which the reference would draw afresh"
      (let [model (comb/unfold (gen [_t prev]
                                 (let [x (trace :x (d/normal (+ prev (untraced (d/normal 0 1))) 1))]
                                   (trace :y (d/normal x 1))
                                   x)))]
        (is (re-find #"needs exact densities"
                     (message #(pmcmc/conditional-smc (gfi/simulate model [2 0.0] (rng/key 17)) (sel/from-paths [[0 :x] [1 :x]])
                                                      [{:args [1 0.0]} {:args [2 0.0]}] 5))))))))

;; ---------------------------------------------------------------------------
;; Particle Gibbs and PMMH on a parameter
;; ---------------------------------------------------------------------------

;; a on the grid 0.2, 0.5, 0.8, uniformly a priori; with these observations
;; p(a | y) = 0.199, 0.298, 0.502
(def grid-ys [0.3 1.1 1.4 0.6 1.2 1.4])

(defn- grid-a [i] (+ 0.2 (* 0.3 i)))

(def grid-posterior
  "p(i | y), i the index of a on the grid."
  (let [l (mapv #(peek (peek (kalman (grid-a %) grid-ys))) (range 3))
        m (reduce max l)
        e (mapv #(Math/exp (- % m)) l)]
    (mapv #(/ % (reduce + e)) e)))

(def grid-observations (apply cm/choicemap (mapcat (fn [t y] [[:xs t :y] y]) (range) grid-ys)))

(defn- chains
  "c chains of kernel on model, from draws given the observations, each of
   samples after burn steps."
  [model args observations kernel c samples burn key]
  (mapv (fn [key]
          (let [[k-start k-run] (rng/split key)]
            (:traces (k/run kernel (:trace (gfi/generate model args observations k-start))
                            {:samples samples :burn burn :key k-run}))))
        (rng/split key c)))

(defn- check-posterior
  "Checks that the chains have mixed (rank-normalized R-hat below 1.01 and
   a bulk ESS above 400, Vehtari et al.'s thresholds for trusting their
   standard errors) and that their frequency of each i is within five Monte
   Carlo standard errors of p(i | y), by gen.inference.diagnostics."
  [chains]
  (let [draws (diag/draws chains :i)]
    (is (< (diag/r-hat draws) 1.01) "the chains agree")
    (is (> (diag/ess-bulk draws) 400) "the chains mix")
    (doseq [i (range 3)
            :let [indicator (mapv (fn [chain] (mapv #(if (== i %) 1.0 0.0) chain)) draws)
                  m (mean (flatten indicator))
                  mcse (diag/mcse-mean indicator)]]
      (is (< (ts/abs (- m (grid-posterior i))) (* 5 mcse))
          (str "p(i = " i " | y): " m " against " (grid-posterior i) ", mcse " mcse)))))

(def grid-model
  (gen [n]
    (let [i (trace :i (d/uniform-discrete 0 2))]
      (splice :xs ssm n 0.0 (grid-a i))
      i)))

;; Planted bugs (2026-10-23), on the JVM backend: the new reference drawn
;; uniformly gives p(i = 0) = 0.308 and p(i = 2) = 0.363 (mcse 0.011); the
;; reference weighed with its states' density 0.303 and 0.378; the other
;; particles not observing 0.324 and 0.341; a move that changes nothing an
;; R-hat of 1.32 and a bulk ESS of 10.
(deftest ^:slow ^:heavy particle-gibbs-converges-to-the-posterior-of-a-parameter
  ;; conditional SMC of four particles on the states, then MH on a from its
  ;; prior: R-hat 1.005, bulk ESS 946 of 4000 draws
  (let [n (count grid-ys)
        pg (pmcmc/particle-gibbs (sel/from-paths (for [t (range n)] [:xs t :x]))
                                 (mapv (fn [t] {:args [(inc t)]}) (range n))
                                 4)]
    (check-posterior (chains grid-model [n] grid-observations (k/chain pg (k/mh (sel/select :i))) 4 1000 50 (rng/key 20)))))

(defn- log-likelihood-estimate
  "log p(ys | a) estimated by a bootstrap particle filter of n particles,
   resampling at every step."
  [a ys n key]
  (let [ks (rng/split key (* 2 (count ys)))
        obs (fn [t] (cm/choicemap [t :y] (ys t)))]
    (pf/log-ml-estimate
     (reduce (fn [state t]
               (-> state
                   (pf/resample {:key (ks (dec (* 2 t)))})
                   (pf/step [(inc t) 0.0 a] [(diff/int-diff 1) diff/no-change diff/no-change] (obs t)
                            {:key (ks (* 2 t))})))
             (pf/initialize ssm [1 0.0 a] (obs 0) n {:key (ks 0)})
             (range 1 (count ys))))))

(def filtered-likelihood
  "The observations' density given a, estimated by a filter of two
   particles (the log estimate's sd is about 1.7): PMMH's likelihood."
  (custom/estimated
   {:addresses (mapv #(vector % :y) (range (count grid-ys)))
    :estimate (fn [[a] choices key]
                (log-likelihood-estimate a (mapv #(cm/get-choice choices [% :y]) (range (count grid-ys))) 2 key))}))

(def pmmh-model
  (gen []
    (let [i (trace :i (d/uniform-discrete 0 2))]
      (splice :xs filtered-likelihood (grid-a i))
      i)))

;; Planted bug (2026-10-23), on the JVM backend: the filter's estimate
;; without the log marginal likelihood its resamplings carried (the log mean
;; of the last weights alone) gives p(i = 0) = 0.296 and p(i = 2) = 0.386
;; (mcse 0.007).
(deftest ^:slow ^:heavy pmmh-targets-the-posterior-of-a-parameter-with-two-particles
  ;; Andrieu et al.'s particle marginal Metropolis-Hastings: MH on a from its
  ;; prior, the likelihood a filter's estimate. R-hat 1.001, bulk ESS 1665
  ;; of 8000 draws
  (check-posterior (chains pmmh-model [] grid-observations (k/mh (sel/select :i)) 4 2000 50 (rng/key 21))))
