(ns gen.inference.mcmc-test
  "HMC, MALA, MAP optimization and elliptical slice sampling on Gaussian
   targets with exact posteriors, including an array-valued latent behind a
   splice; and involutive MH with an AD Jacobian (thesis Eq 3.15)."
  (:refer-clojure :exclude [+ - * / >])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * / >]]
            [gen.array.batch :as batch]
            [gen.array.linalg :as la]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.kernel :as k]
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

(def prior-cov (array/array [[1.0 0.5] [0.5 1.0]]))
(def noise-cov (array/array [[0.5 0.0] [0.0 0.5]]))
(def y2 (array/array [1.0 -0.5]))

(def latent (gen [] (trace :z (d/mvnormal [0.0 0.0] prior-cov))))

(def vector-model
  "z ~ N(0, Σ0) inside a splice, y ~ N(z, Σ1)."
  (gen [] (let [z (splice :latent latent)] (trace :y (d/mvnormal z noise-cov)))))

(def flat-vector-model
  (gen [] (let [z (trace :z (d/mvnormal [0.0 0.0] prior-cov))] (trace :y (d/mvnormal z noise-cov)))))

(def posterior-cov (la/inv (+ (la/inv prior-cov) (la/inv noise-cov))))
(def posterior-mean (array/matmul posterior-cov (array/matmul (la/inv noise-cov) y2)))

(defn- start-at [model choices] (:trace (gfi/generate model [] choices (rng/key 0))))

(defn- moments [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]))

(defn- matches? [xs mean var]
  (let [[m v] (moments xs)]
    (and (< (ts/abs (- m mean)) (* 5 (Math/sqrt (/ var (count xs)))))
         (< (ts/abs (- v var)) (* 0.12 var)))))

(defn- scalar-stationary? [step]
  (let [n 4000
        [k1 k2] (rng/split (rng/key 21))
        starts (map #(start-at scalar (cm/choicemap :x % :y 1.3))
                    (array/->clj (+ 0.65 (* (Math/sqrt 0.5) (rng/normal k1 [n])))))]
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
        ends (map (fn [key] (array/->clj (z-of (reduce step start (rng/split key 60))))) (rng/split (rng/key 23) 1000))]
    (every? true? (for [i [0 1]]
                    (matches? (map #(nth % i) ends)
                              (array/item (array/take posterior-mean i))
                              (array/item (array/take (array/take posterior-cov i) i)))))))

;; ---------------------------------------------------------------------------
;; HMC and MALA
;; ---------------------------------------------------------------------------

(def x-sel (sel/select :x))

(deftest ^:slow hmc-targets-the-posterior
  (let [step (fn [tr key] (:trace (mcmc/hmc tr x-sel {:L 5 :eps 0.3} key)))]
    (is (scalar-stationary? step))
    (is (scalar-converges? step)))
  (testing "a vector latent behind a splice (a nested address)"
    (is (vector-converges? vector-model (cm/choicemap [:latent :z] (array/array [2.0 2.0]) :y y2)
                           #(% [:latent :z])
                           (fn [tr key] (:trace (mcmc/hmc tr (sel/select [:latent :z]) {:L 5 :eps 0.25} key)))))))

(deftest ^:slow mala-targets-the-posterior
  (let [step (fn [tr key] (:trace (mcmc/mala tr x-sel 0.3 key)))]
    (is (scalar-stationary? step))
    (is (scalar-converges? step)))
  (is (vector-converges? vector-model (cm/choicemap [:latent :z] (array/array [2.0 2.0]) :y y2)
                         #(% [:latent :z])
                         (fn [tr key] (:trace (mcmc/mala tr (sel/select [:latent :z]) 0.15 key))))))

(deftest acceptance-ratios
  (let [tr (start-at scalar (cm/choicemap :x 0.2 :y 1.3))
        rs (map #(mcmc/hmc tr x-sel {:L 3 :eps 0.2} %) (rng/split (rng/key 1) 100))]
    (is (every? #(or (:accepted? %) (identical? tr (:trace %))) rs))
    (is (> (count (filter :accepted? rs)) 80) "small steps on a smooth target are mostly accepted")))

(def ^:private funnel
  "Neal's funnel: v ~ N(0, 3), x ~ N(0, e^{v/2})."
  (gen [] (let [v (trace :v (d/normal 0 3))] (trace :x (d/normal 0 (array/exp (* 0.5 v)))))))

(deftest a-diverging-proposal-is-rejected-and-counted
  ;; the post-M5 review's P3-05 (DESIGN §9): a trajectory whose position,
  ;; momentum, gradient or energy turns non-finite is rejected, with
  ;; acceptance 0, and reported. Planted bug (2026-10-22): the kernels as
  ;; before throw "a constrained value is NaN" at N(0, 1), eps 3, L 400
  ;; and at sd 0.003, eps 0.1, L 150, and the funnel's chain dies (at L
  ;; 10, eps 0.5 the verifier's died at step 1,309)
  (let [unit (start-at (gen [] (trace :x (d/normal 0 1))) (cm/choicemap :x 0.3))
        tight (start-at (gen [] (trace :x (d/normal 0 0.003))) (cm/choicemap :x 0.001))]
    (doseq [[label tr opts] [["N(0, 1), eps 3, L 400" unit {:L 400 :eps 3.0}]
                             ["sd 0.003, eps 0.1, L 150" tight {:L 150 :eps 0.1}]]
            compile? [false true]]
      (testing (str label (when compile? ", compiled"))
        (is (= {:trace tr :accepted? false :log-alpha ##-Inf :diverged? true}
               (mcmc/hmc tr x-sel (assoc opts :compile? compile?) (rng/key 2))))))
    (testing "an energy that overflows: rejected and reported too"
      (is (:diverged? (mcmc/hmc tight x-sel {:L 100 :eps 0.1} (rng/key 2)))))
    (testing "MALA, its proposal past the float range"
      (is (= {:trace unit :accepted? false :log-alpha ##-Inf :diverged? true} (mcmc/mala unit x-sel 1e300 (rng/key 3))))))
  (testing "the funnel: a chain goes on, and k/run counts the divergences, through a chain of kernels too"
    (let [start (start-at funnel (cm/choicemap :v 0.0 :x 1.0))
          step (fn [tr key] (mcmc/hmc tr (sel/select :v :x) {:L 20 :eps 1.0} key))
          ;; the divergences of steps on keys, run's per step split in m
          by-hand (fn [n m] (reduce (fn [[tr d] key]
                                      (reduce (fn [[tr d] key] (let [r (step tr key)] [(:trace r) (if (:diverged? r) (inc d) d)]))
                                              [tr d] (if (= 1 m) [key] (rng/split key m))))
                                    [start 0] (rng/split (rng/key 4) n)))
          {:keys [traces divergences]} (try (k/run (k/kernel step) start {:samples 200 :key (rng/key 4)})
                                            (catch Exception _ nil))]
      (is (= 200 (count traces)))
      (is (and divergences (< 10 divergences 190)) (str divergences))
      (is (= (second (by-hand 200 1)) divergences))
      (is (= (second (by-hand 100 2))
             (:divergences (k/run (k/chain (k/kernel step) (k/kernel step)) start {:samples 100 :key (rng/key 4)})))))))

(deftest a-parameter-check-failing-mid-trajectory-is-a-divergence
  ;; DESIGN §9 (PROVISIONAL, 2026-10-22): at the funnel's L 3, eps 20, a
  ;; finite v near -1500 underflows exp(v/2) to a std of 0, which the JVM
  ;; backend's concrete check refuses; the density is undefined there, so
  ;; the proposal diverged, and the run's acceptance stays a rate over
  ;; every proposal. Planted bug (2026-10-22): the step's catch rethrowing
  ;; throws "normal: std must be positive" in each case (33 of 300 HMC
  ;; proposals from (0, 1), 166 from (-6, 0.01), as independent proposals);
  ;; MALA's proposal densities by gen.dist's normal throw "normal: mu is
  ;; NaN" on the diverged gradient
  (doseq [[v x] [[0.0 1.0] [-6.0 0.01]]]
    (testing (str "from v = " v ", x = " x ": steps of eps 0.5, then of eps 20")
      (let [start (start-at funnel (cm/choicemap :v v :x x))
            n 150
            hmc (fn [opts] (k/kernel (fn [tr key] (mcmc/hmc tr (sel/select :v :x) opts key))))
            run (try (k/run (k/chain (hmc {:L 10 :eps 0.5}) (hmc {:L 3 :eps 20.0})) start {:samples n :key (rng/key 8)})
                     (catch clojure.lang.ExceptionInfo e (ex-message e)))
            {:keys [acceptance-rate divergences]} (when (map? run) run)]
        (is (map? run) (str run))
        (is (and divergences (<= n divergences (* 2 n))) (str divergences))
        (is (and acceptance-rate (<= 0.0 acceptance-rate (- 1.0 (/ divergences (* 2.0 n))))) (str acceptance-rate)))))
  (testing "MALA too: tau 3000 from v = 0 proposes v near -1500"
    (let [start (start-at funnel (cm/choicemap :v 0.0 :x 0.01))]
      (is (every? :diverged? (map #(mcmc/mala start (sel/select :v :x) 3000.0 %) (rng/split (rng/key 9) 50))))))
  (testing "other errors at a tried position stay loud: a structure change"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"within the trace's structure"
                          (let [model (gen [] (let [x (trace :x (d/normal 0 1))]
                                                (when (array/truthy? (> x 0)) (trace :extra (d/normal 0 1)))
                                                (trace :y (d/normal x 0.3))))
                                tr (:trace (gfi/generate model [] (cm/choicemap :x -0.05 :y 3.0) (rng/key 0)))]
                            (mcmc/hmc tr (sel/select :x) {:L 3 :eps 0.5} (rng/key 72)))))))

(defn- refused-start
  "How f refused its starting trace: :nan-value, by assess's check of a NaN
   value (read where values are concrete, as on the JVM), :non-finite-start,
   by the kernel's check of the start's log density (where values are lazy,
   as on MLX, the NaN reaches it first), or nil when f returned."
  [f]
  (try (f) nil
       (catch clojure.lang.ExceptionInfo e
         (cond (= :non-finite-start (:reason (ex-data e))) :non-finite-start
               (re-find #"constrained value is NaN" (ex-message e)) :nan-value
               :else (throw e)))))

(deftest a-nan-in-the-starting-trace-stays-an-error
  ;; DESIGN §9: only a proposal diverges, and a NaN start is loud on every
  ;; backend, by one check or the other (on Metal the start's check, found
  ;; by m3-llama's light run). Planted bug (2026-10-22): every error of the
  ;; steps taken as a divergence, the start's gradient's too, with the
  ;; start's check gone, passes the trace with a NaN (rejected and reported
  ;; as diverged), all six assertions failing
  (let [tr (start-at scalar (cm/choicemap :x 0.2 :y 1.3))
        nan (assoc tr :choices (cm/set-choice (:choices tr) :x ##NaN))]
    (is (#{:nan-value :non-finite-start} (refused-start #(mcmc/hmc nan x-sel {:L 3 :eps 0.1} (rng/key 5)))))
    (is (#{:nan-value :non-finite-start} (refused-start #(mcmc/mala nan x-sel 0.1 (rng/key 6))))))
  (testing "a start of probability 0 too (DESIGN §9, PROVISIONAL; m3-grammar's cross-review)"
    ;; y ~ U(mu - 1, mu + 1) holds y = 3 at mu = 0 with density 0. Planted
    ;; bug (2026-10-22): no check of the start makes every proposal a
    ;; divergence, 40 of 40, and the chain never moves
    (let [boxed (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/uniform (- mu 1.0) (+ mu 1.0)))))
          start (start-at boxed (cm/choicemap :mu 0.0 :y 3.0))]
      (doseq [compile? [false true]]
        (is (= {:reason :non-finite-start :kernel :hmc}
               (select-keys (ts/thrown-data (mcmc/hmc start (sel/select :mu) {:L 5 :eps 1.0 :compile? compile?} (rng/key 7)))
                            [:reason :kernel])))
        (is (= {:reason :non-finite-start :kernel :mala}
               (select-keys (ts/thrown-data (mcmc/mala start (sel/select :mu) 2.0 {:compile? compile? :key (rng/key 8)}))
                            [:reason :kernel])))))))

;; ---------------------------------------------------------------------------
;; MAP and elliptical slice
;; ---------------------------------------------------------------------------

(deftest keys-and-errors
  (let [tr (start-at scalar (cm/choicemap :x 0.2 :y 1.3))]
    (testing "hmc takes its key in the options map too"
      (is (= ((:trace (mcmc/hmc tr x-sel {:key (rng/key 5)})) :x) ((:trace (mcmc/hmc tr x-sel {:key (rng/key 5)})) :x))))
    (testing "a key as hmc's third argument (rule 2; the post-M5 review's verifier)"
      ;; planted bug (2026-10-22): the 3-arity taking the key for options
      ;; draws from the implicit source, and the moves differ
      (let [moves (repeatedly 4 #((:trace (mcmc/hmc tr x-sel (rng/key 7))) :x))]
        (is (apply = moves))
        (is (not= (first moves) 0.2) "it moved"))
      (is (= ((:trace (mcmc/hmc tr x-sel (rng/key 7))) :x) ((:trace (mcmc/hmc tr x-sel {} (rng/key 7))) :x)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"options map" (mcmc/hmc tr x-sel (rng/key 7) (rng/key 8))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"key or an options map" (mcmc/hmc tr x-sel 7)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"key or an options map" (mcmc/mala tr x-sel 0.1 7))))
    (testing "a selection of no choices is an error, not a no-op accepted"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/hmc tr (sel/select :nope) {})))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/mala tr (sel/select :nope) 0.1)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/map-optimize tr (sel/select :nope)))))
    (testing "mvnormal's shapes are checked"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mu must be a vector" (d/mvnormal 0.0 1.0)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"square matrix of its size" (d/mvnormal [0.0 0.0] (array/eye 3)))))
    (testing "elliptical slice takes Clojure vectors, as distributions do"
      (let [tr (start-at flat-vector-model (cm/choicemap :z (array/array [0.5 0.5]) :y y2))]
        (is (= [2] (array/shape ((mcmc/elliptical-slice tr :z [0.0 0.0] [[1.0 0.5] [0.5 1.0]] (rng/key 2)) :z))))))
    (testing "elliptical slice of a missing address"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no choice at :nope"
                            (mcmc/elliptical-slice tr :nope (array/array [0.0]) (array/eye 1) (rng/key 1)))))))

(deftest map-optimization-finds-the-mode
  (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr x-sel)) (start-at scalar (cm/choicemap :x -2.0 :y 1.3)) (range 200))]
    (is (< (ts/abs (- (tr :x) 0.65)) 1e-6)))
  (testing "a vector mode behind a splice"
    (let [tr (reduce (fn [tr _] (mcmc/map-optimize tr (sel/select [:latent :z]) {:max-step-size 0.3}))
                     (start-at vector-model (cm/choicemap [:latent :z] (array/array [2.0 2.0]) :y y2))
                     (range 300))]
      (is (array/allclose? (tr [:latent :z]) posterior-mean {:atol 1e-6})))))

(deftest map-optimization-shrinks-a-step-outside-the-support
  ;; genclj-2870: a trial step that takes s below 0 leaves gamma's support,
  ;; which stops the trial's run (weight -Inf, DESIGN.md §9) before normal's
  ;; check of its std: a failed step, which the line search halves. Planted
  ;; bug (2026-10-25): the handler not stopping throws "normal: std must be
  ;; positive" here, as it did.
  (let [m (gen [] (let [s (trace :s (d/gamma 1 1))] (trace :y (d/normal 0 s))))
        tr (:trace (gfi/generate m [] (cm/choicemap :s 0.5 :y 0.1) (rng/key 0)))
        s-sel (sel/select :s)
        g (ts/number (cm/get-choice (:gradients (grad/choice-gradients tr s-sel)) :s))
        t2 (mcmc/map-optimize tr s-sel {:max-step-size 1.0})
        step (/ (- (ts/number (t2 :s)) 0.5) g)]
    (is (neg? (+ 0.5 g)) "a full step leaves the support")
    (is (< 0.0 (ts/number (t2 :s))))
    (is (some #(ts/close? step (Math/pow 0.5 %)) (range 1 20)) "the step taken is a halving of the full one")
    (is (> (ts/number (:score t2)) (ts/number (:score tr))))))

(deftest ^:slow elliptical-slice-targets-the-posterior
  (is (vector-converges? flat-vector-model (cm/choicemap :z (array/array [2.0 2.0]) :y y2)
                         #(% :z)
                         (fn [tr key] (mcmc/elliptical-slice tr :z (array/array [0.0 0.0]) prior-cov key)))))

;; ---------------------------------------------------------------------------
;; Involutive MH with a Jacobian
;; ---------------------------------------------------------------------------

(def positive (gen [] (trace :x (d/gamma 3 1))))

(def log-scale (gen [tr] (trace :u (d/normal 0 0.5))))

(defn- scale-move
  "(x, u) ↦ (x·e^u, -u), an involution with |det J| = e^u, its log Jacobian
   computed by autodiff."
  [tr fwd _ _ key]
  (let [x (tr :x)
        u (cm/get-choice fwd :u)
        h (fn [v] (let [x (array/take v 0) u (array/take v 1)] (array/stack [(* x (array/exp u)) (array/negate u)])))
        {t :trace w :weight} (gfi/update tr (cm/choicemap :x (* x (array/exp u))) key)]
    {:trace t :bwd-choices (cm/choicemap :u (- u))
     :weight (+ w (grad/log-abs-det-jacobian h (array/stack [x u])))}))

(deftest ^:slow a-volume-changing-involution-targets-the-posterior
  (let [step (fn [tr key] (:trace (mh/involutive-mh tr log-scale [] scale-move key)))
        start (start-at positive (cm/choicemap :x 1.0))
        ends (map (fn [key] ((reduce step start (rng/split key 80)) :x)) (rng/split (rng/key 31) 1000))]
    (is (matches? ends 3.0 3.0) "Gamma(3, 1): mean 3, variance 3"))
  (testing "without the Jacobian the chain is biased, so the test can tell"
    (let [wrong (fn [tr fwd r args key] (assoc (scale-move tr fwd r args key) :weight
                                               (:weight (gfi/update tr (cm/choicemap :x (* (tr :x) (array/exp (cm/get-choice fwd :u)))) key))))
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
                 (* 0.5 (+ (trace :m1 (d/normal 0 1)) (trace :m2 (d/normal 0 1))))
                 (trace :m (d/normal 0 1)))]
      (trace :y (d/normal mean 1)))))

(def split-merge-proposal
  "Split with an auxiliary u ~ N(0, 1); merging needs no auxiliary choice."
  (gen [tr] (when-not (tr :two?) (trace :u (d/normal 0 1)))))

(defn- split-merge
  "The involution (m, u) ↔ (m1, m2) = (m - u, m + u), with the dimension
   change and its Jacobian (computed by autodiff: |det| = 2 for a split)."
  [tr fwd _ _ key]
  (if (tr :two?)
    (let [m1 (tr :m1) m2 (tr :m2)
          h (fn [v] (let [a (array/take v 0) b (array/take v 1)] (array/stack [(* 0.5 (+ a b)) (* 0.5 (- b a))])))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :two? false :m (* 0.5 (+ m1 m2))) key)]
      {:trace t :bwd-choices (cm/choicemap :u (* 0.5 (- m2 m1)))
       :weight (+ w (grad/log-abs-det-jacobian h (array/stack [m1 m2])))})
    (let [m (tr :m) u (cm/get-choice fwd :u)
          h (fn [v] (let [a (array/take v 0) b (array/take v 1)] (array/stack [(- a b) (+ a b)])))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :two? true :m1 (- m u) :m2 (+ m u)) key)]
      {:trace t :bwd-choices cm/EMPTY
       :weight (+ w (grad/log-abs-det-jacobian h (array/stack [m u])))})))

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
    (is (< (ts/abs (- frac exact-two)) (* 5 (Math/sqrt (/ (* exact-two (- 1 exact-two)) 2000)))))
    (testing "the involution is an involution, with opposite weights"
      (let [tr start fwd (cm/choicemap :u 0.4)
            {t1 :trace u1 :bwd-choices w1 :weight} (split-merge tr fwd nil nil (rng/key 0))
            {t2 :trace u2 :bwd-choices w2 :weight} (split-merge t1 u1 nil nil (rng/key 0))]
        (is (= (:choices tr) (:choices t2)))
        (is (= fwd u2))
        (is (< (ts/abs (+ w1 w2)) 1e-12))))))

(defn- host-normal-lp [x mu sd]
  (clojure.core/- (clojure.core/- (Math/log (clojure.core/* sd (Math/sqrt (clojure.core/* 2 Math/PI)))))
                  (clojure.core// (Math/pow (clojure.core/- x mu) 2) (clojure.core/* 2 sd sd))))

(deftest one-step-by-hand
  ;; plain and compiled kernels run one function (review 3: the comparison
  ;; below cannot see a kernel bug), so a fast test checks a step against
  ;; arithmetic: on scalar with y = 1.3, log p = log N(x; 0, 1) + log N(y; x, 1)
  ;; and ∇x log p = y - 2x, with the kernels' own momentum and noise draws
  (let [y 1.3 x0 0.4
        tr (:trace (gfi/generate scalar [] (cm/choicemap :x x0 :y y) (rng/key 0)))
        grad-x (fn [x] (clojure.core/- y (clojure.core/* 2.0 x)))
        lp (fn [x] (clojure.core/+ (host-normal-lp x 0.0 1.0) (host-normal-lp y x 1.0)))
        draw (fn [key] (ts/number (array/take (rng/normal key [1]) 0)))
        moved (fn [r] (ts/number ((:trace r) :x)))]
    (testing "HMC: two leapfrog steps and the Hamiltonian's change"
      (let [key (rng/key 70) eps 0.3 half (clojure.core// eps 2)
            p0 (draw (first (rng/split key 3)))
            [x p] (loop [i 0 x x0 p p0]
                    (if (clojure.core/< i 2)
                      (let [p (clojure.core/+ p (clojure.core/* half (grad-x x)))
                            x (clojure.core/+ x (clojure.core/* eps p))]
                        (recur (inc i) x (clojure.core/+ p (clojure.core/* half (grad-x x)))))
                      [x p]))
            log-alpha (clojure.core/+ (clojure.core/- (lp x) (lp x0)) (clojure.core/- (host-normal-lp p 0.0 1.0) (host-normal-lp p0 0.0 1.0)))
            r (mcmc/hmc tr (sel/select :x) {:L 2 :eps eps} key)]
        (is (ts/close? log-alpha (:log-alpha r) {:atol (ts/tol 1e-9)}))
        (is (ts/close? (if (:accepted? r) x x0) (moved r)))))
    (testing "MALA: the Langevin proposal and both directions' densities"
      (let [key (rng/key 71) tau 0.2 sd (Math/sqrt (clojure.core/* 2 tau))
            z (draw (first (rng/split key 3)))
            fwd (clojure.core/+ x0 (clojure.core/* tau (grad-x x0)))
            x (clojure.core/+ fwd (clojure.core/* sd z))
            bwd (clojure.core/+ x (clojure.core/* tau (grad-x x)))
            log-alpha (clojure.core/+ (clojure.core/- (lp x) (lp x0)) (clojure.core/- (host-normal-lp x0 bwd sd) (host-normal-lp x fwd sd)))
            r (mcmc/mala tr (sel/select :x) tau key)]
        (is (ts/close? log-alpha (:log-alpha r) {:atol (ts/tol 1e-9)}))
        (is (ts/close? (if (:accepted? r) x x0) (moved r)))))))

(deftest one-step-by-hand-with-a-metric
  ;; the same steps under an inverse metric m (a variance): momenta
  ;; z/√m, positions moving by m·p, kinetic energy ½ m p²; MALA's proposal
  ;; N(x + tau·m·∇, 2 tau·m). Planted bug (2026-10-23): HMC's kinetic
  ;; energy without the metric fails its log-alpha; MALA's drift without
  ;; it fails both
  (let [y 1.3 x0 0.4 m 2.5
        tr (:trace (gfi/generate scalar [] (cm/choicemap :x x0 :y y) (rng/key 0)))
        grad-x (fn [x] (clojure.core/- y (clojure.core/* 2.0 x)))
        lp (fn [x] (clojure.core/+ (host-normal-lp x 0.0 1.0) (host-normal-lp y x 1.0)))
        draw (fn [key] (ts/number (array/take (rng/normal key [1]) 0)))
        moved (fn [r] (ts/number ((:trace r) :x)))]
    (testing "HMC"
      (let [key (rng/key 75) eps 0.3 half (clojure.core// eps 2)
            p0 (clojure.core// (draw (first (rng/split key 3))) (Math/sqrt m))
            [x p] (loop [i 0 x x0 p p0]
                    (if (clojure.core/< i 2)
                      (let [p (clojure.core/+ p (clojure.core/* half (grad-x x)))
                            x (clojure.core/+ x (clojure.core/* eps m p))]
                        (recur (inc i) x (clojure.core/+ p (clojure.core/* half (grad-x x)))))
                      [x p]))
            log-alpha (clojure.core/- (clojure.core/- (lp x) (lp x0)) (clojure.core/* 0.5 m (clojure.core/- (clojure.core/* p p) (clojure.core/* p0 p0))))
            r (mcmc/hmc tr (sel/select :x) {:L 2 :eps eps :inv-metric [m]} key)]
        (is (ts/close? log-alpha (:log-alpha r) {:atol (ts/tol 1e-9)}))
        (is (ts/close? (if (:accepted? r) x x0) (moved r)))))
    (testing "MALA"
      (let [key (rng/key 76) tau 0.2 sd (Math/sqrt (clojure.core/* 2 tau m))
            z (draw (first (rng/split key 3)))
            fwd (clojure.core/+ x0 (clojure.core/* tau m (grad-x x0)))
            x (clojure.core/+ fwd (clojure.core/* sd z))
            bwd (clojure.core/+ x (clojure.core/* tau m (grad-x x)))
            log-alpha (clojure.core/+ (clojure.core/- (lp x) (lp x0)) (clojure.core/- (host-normal-lp x0 bwd sd) (host-normal-lp x fwd sd)))
            r (mcmc/mala tr (sel/select :x) tau {:inv-metric m :key key})]
        (is (ts/close? log-alpha (:log-alpha r) {:atol (ts/tol 1e-9)}))
        (is (ts/close? (if (:accepted? r) x x0) (moved r)))))))

(deftest a-move-that-changes-the-structure-is-named
  ;; :extra exists only for x > 0, and the moves carry x across 0 (review 3:
  ;; the error named assess, not the kernel's requirement)
  (let [model (gen [] (let [x (trace :x (d/normal 0 1))]
                        (when (array/truthy? (> x 0)) (trace :extra (d/normal 0 1)))
                        (trace :y (d/normal x 0.3))))
        tr (:trace (gfi/generate model [] (cm/choicemap :x -0.05 :y 3.0) (rng/key 0)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"within the trace's structure.*\[:extra\]"
                          (mcmc/hmc tr (sel/select :x) {:L 3 :eps 0.5} (rng/key 72))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"within the trace's structure"
                          (mcmc/mala tr (sel/select :x) 0.5 (rng/key 73))))
    (testing "and the other way: leaving :extra behind (review 4)"
      (let [tr (:trace (gfi/generate model [] (cm/choicemap :x 0.05 :extra 0.0 :y -3.0) (rng/key 0)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no longer visits \[:extra\]"
                              (mcmc/hmc tr (sel/select :x) {:L 3 :eps 0.5} (rng/key 74))))))))

(deftest compiled-kernels-make-the-same-moves
  ;; :compile? runs HMC's L steps, or MALA's proposal and its densities, as
  ;; one compiled function (one MLX graph on MLX); each must move as the
  ;; plain kernel does
  (let [obs (cm/choicemap :y 1.3)
        branching (gen [] (let [x (trace :x (d/normal 0 1))]
                            (trace :y (d/normal (if (array/truthy? (> x 0)) x (array/negate x)) 1))))
        ;; in a compiled trace the observed count is an MLX array, which
        ;; dotimes cannot take: the step once threw instead of falling back
        ;; (an independent review)
        counted (gen [] (let [n (trace :n (d/poisson 3))
                              x (trace :x (d/normal 0 1))]
                          (dotimes [i n] (trace [:y i] (d/normal x 1)))))]
    (doseq [[label gf observations selection] [["scalar" scalar obs (sel/select :x)]
                                               ["an array-valued latent in a splice" vector-model (cm/choicemap :y y2)
                                                (sel/select [:latent :z])]
                                               ["a model that branches on a value (falls back)" branching obs
                                                (sel/select :x)]
                                               ["a model that loops over an observed count (falls back)" counted
                                                (cm/choicemap :n 2 [:y 0] 0.5 [:y 1] 1.0) (sel/select :x)]]]
      (testing label
        (let [tr (:trace (gfi/generate gf [] observations (rng/key 1)))]
          (doseq [[kernel move] [[:hmc (fn [opts key] (mcmc/hmc tr selection (merge {:L 5 :eps 0.2} opts) key))]
                                 [:mala (fn [opts key] (mcmc/mala tr selection 0.3 (assoc opts :key key)))]]
                  k (range 5)]
            (let [plain (move nil (rng/key k))
                  compiled (move {:compile? true} (rng/key k))]
              (is (= (:accepted? plain) (:accepted? compiled)) (str kernel))
              (is (ts/close? (:log-alpha plain) (:log-alpha compiled)) (str kernel))
              (is (ts/close? (:score (:trace plain)) (:score (:trace compiled))) (str kernel)))))))))

;; ---------------------------------------------------------------------------
;; Every particle at once: hmc-particles, mala-particles,
;; map-optimize-particles
;; ---------------------------------------------------------------------------

(defn- chains-from
  "n particles of model, every one at the choices."
  [model choices n]
  (b/take (:particles (b/generate model [] choices 1 (rng/key 0))) (repeat n 0)))

(defn- particles-match?
  "Do the n particles' values xs (host numbers) have the posterior's mean
   and variance? Bands of 5 standard errors: the mean's √(var/n), and the
   sample variance's var·√(2/(n - 1)) for a normal posterior."
  [xs mean var]
  (let [n (count xs) [m v] (moments xs)]
    (and (< (Math/abs (clojure.core/- m mean)) (clojure.core/* 5 (Math/sqrt (clojure.core// var n))))
         (< (Math/abs (clojure.core/- v var)) (clojure.core/* 5 var (Math/sqrt (clojure.core// 2.0 (dec n))))))))

(deftest ^:slow parallel-hmc-and-mala-chains-target-the-posterior
  ;; 4000 chains from x = -2, each step one batched gradient per leapfrog
  ;; step for all of them; the chains stay batched. Planted (2026-10-23):
  ;; every chain's momenta particle 0's synchronizes the chains, failing
  ;; HMC's scalar and vector cases
  (let [n 4000
        start (chains-from scalar (cm/choicemap :x -2.0 :y 1.3) n)
        run (fn [step steps key] (reduce (fn [ps k] (:particles (step ps k))) start (rng/split key steps)))]
    (doseq [[label chains] [["hmc" (run #(mcmc/hmc-particles %1 x-sel {:L 5 :eps 0.3} %2) 30 (rng/key 90))]
                            ["mala" (run #(mcmc/mala-particles %1 x-sel 0.3 %2) 60 (rng/key 91))]]]
      (testing label
        (is (b/batched? chains))
        (is (particles-match? (ts/host (b/choice chains :x)) 0.65 0.5)))))
  (testing "a vector latent behind a splice"
    (let [n 2000
          chains (reduce (fn [ps k] (:particles (mcmc/hmc-particles ps (sel/select [:latent :z]) {:L 5 :eps 0.25} k)))
                         (chains-from vector-model (cm/choicemap [:latent :z] (array/array [2.0 2.0]) :y y2) n)
                         (rng/split (rng/key 92) 30))
          zs (ts/host (b/choice chains [:latent :z]))]
      (is (b/batched? chains))
      (doseq [i [0 1]]
        (is (particles-match? (map #(nth % i) zs)
                              (array/item (array/take posterior-mean i))
                              (array/item (array/take (array/take posterior-cov i) i)))
            (str "coordinate " i))))))

(deftest particle-moves-by-hand
  ;; the batched path against arithmetic, particle by particle, with the
  ;; draws the tracer makes from the kernel's keys (one-step-by-hand's
  ;; target: log p = log N(x; 0, 1) + log N(y; x, 1), ∇ = y - 2x). Steps
  ;; long enough that some particles are rejected. Planted (2026-10-23):
  ;; every particle's momenta particle 0's fails HMC's ratios; a fixed
  ;; uniform of 1/2 for the acceptance flips particle 1's verdicts
  (let [y 1.3 n 4
        x0s [-1.5 0.2 0.9 2.4]
        ps (b/set-choices (chains-from scalar (cm/choicemap :x 0.0 :y y) n) (cm/choicemap :x (array/array x0s)))
        grad-x (fn [x] (clojure.core/- y (clojure.core/* 2.0 x)))
        lp (fn [x] (clojure.core/+ (host-normal-lp x 0.0 1.0) (host-normal-lp y x 1.0)))
        drawn (fn [f] (ts/host (batch/unwrap (batch/batched [n] (f)) [n])))
        check (fn [r xs log-alphas key]
                (let [us (drawn #(rng/uniform (nth (rng/split key 3) 2)))]
                  (is (b/batched? (:particles r)))
                  (is (= #{true false} (set (:accepted? r))) "some accepted, some not")
                  (is (ts/close? (array/array log-alphas) (:log-alpha r) {:atol (ts/tol 1e-9)}))
                  (is (= (mapv #(or (>= %1 0.0) (< (Math/log %2) %1)) log-alphas us) (:accepted? r)))
                  (is (ts/close? (array/array (mapv #(if %1 %2 %3) (:accepted? r) xs x0s)) (b/choice (:particles r) :x)))))]
    (testing "HMC"
      (let [key (rng/key 95) eps 1.2 half (clojure.core// eps 2)
            p0s (map first (drawn #(rng/normal (first (rng/split key 3)) [1])))
            ends (map (fn [x0 p0]
                        (loop [i 0 x x0 p p0]
                          (if (clojure.core/< i 2)
                            (let [p (clojure.core/+ p (clojure.core/* half (grad-x x)))
                                  x (clojure.core/+ x (clojure.core/* eps p))]
                              (recur (inc i) x (clojure.core/+ p (clojure.core/* half (grad-x x)))))
                            [x (clojure.core/+ (clojure.core/- (lp x) (lp x0)) (clojure.core/- (host-normal-lp p 0.0 1.0) (host-normal-lp p0 0.0 1.0)))])))
                      x0s p0s)]
        (check (mcmc/hmc-particles ps x-sel {:L 2 :eps eps} key) (mapv first ends) (mapv second ends) key)))
    (testing "MALA"
      (let [key (rng/key 96) tau 1.0 sd (Math/sqrt (clojure.core/* 2 tau))
            zs (map first (drawn #(rng/normal (first (rng/split key 3)) [1])))
            moves (map (fn [x0 z]
                         (let [fwd (clojure.core/+ x0 (clojure.core/* tau (grad-x x0)))
                               x (clojure.core/+ fwd (clojure.core/* sd z))
                               bwd (clojure.core/+ x (clojure.core/* tau (grad-x x)))]
                           [x (clojure.core/+ (clojure.core/- (lp x) (lp x0)) (clojure.core/- (host-normal-lp x0 bwd sd) (host-normal-lp x fwd sd)))]))
                       x0s zs)]
        (check (mcmc/mala-particles ps x-sel tau key) (mapv first moves) (mapv second moves) key)))))

(defn- agree?
  "Are the host values xs and ys (vectors of numbers, or of vectors) equal:
   exactly in float64, within a few ulps in float32? A jittered step is
   computed in float64 on the host by hmc and in the backend's float by
   hmc-particles (its u an array), which in float32 can differ in the last
   place."
  [xs ys]
  (if (ts/f32?) (ts/close? (array/array xs) (array/array ys)) (= xs ys)))

(defn- each-chain-s?
  "Is the -particles kernel's step (step ps key) each chain's scalar step
   (scalar trace key), from its own key (key split n ways)? Exactly, but
   for agree?'s float32 last place."
  [ps step scalar key]
  (let [r (step ps key)
        each (map scalar (b/traces ps) (rng/split key (b/size ps)))]
    (and (= (mapv :accepted? each) (:accepted? r))
         (= (mapv :diverged? each) (:diverged? r))
         (agree? (mapv :log-alpha each) (ts/host (:log-alpha r)))
         (agree? (mapv #(ts/host (cm/to-array (:choices (:trace %)))) each)
                 (mapv #(ts/host (cm/to-array (:choices %))) (b/traces (:particles r)))))))

(deftest particle-moves-fall-back-to-each-chain-s-own
  ;; where the model cannot run batched, or the batched move fails, each
  ;; chain moves as the scalar kernel does from its key (gen.batched's
  ;; fallback). Planted (2026-10-23): tried's NaN in a batched run (no
  ;; fallback) diverges the funnel's batch whole, and fails the second
  (let [chosen (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))
                             x (trace :x (d/normal (if (= k 0) 0.0 3.0) 1))]
                         (trace :y (d/normal x 1))))
        ps (:particles (b/generate chosen [] (cm/choicemap :y 1.3) 6 (rng/key 95)))]
    (is (not (b/batched? ps)))
    (is (each-chain-s? ps #(mcmc/hmc-particles %1 x-sel {:L 3 :eps 0.4} %2) #(mcmc/hmc %1 x-sel {:L 3 :eps 0.4} %2) (rng/key 96)))
    (is (each-chain-s? ps #(mcmc/mala-particles %1 x-sel 0.3 %2) #(mcmc/mala %1 x-sel 0.3 %2) (rng/key 97))))
  ;; where values are concrete: on a lazy backend a parameter's check
  ;; reads no value (rule 5), so the batched move does not fail, and runs
  ;; batched, with draws other than the chains' own keys'
  (when (ts/jvm?)
    (testing "a parameter failing its check at some particles' positions: each chain's divergence its own"
      ;; v ~ N(0, 1000), y ~ N(0, e^v): one step of eps 1000 from v = 0 moves
      ;; v to 1000·p, past exp's range for some momenta (a scale of 0, which
      ;; fails its check)
      (let [scaled (gen [] (let [v (trace :v (d/normal 0 1000))] (trace :y (d/normal 0 (array/exp v)))))
            ps (chains-from scaled (cm/choicemap :v 0.0 :y 1.0) 40)
            s (sel/select :v)
            r (mcmc/hmc-particles ps s {:L 1 :eps 1000.0} (rng/key 98))]
        (is (< 0 (count (filter true? (:diverged? r))) 40) "some diverge, not all")
        (is (each-chain-s? ps #(mcmc/hmc-particles %1 s {:L 1 :eps 1000.0} %2) #(mcmc/hmc %1 s {:L 1 :eps 1000.0} %2) (rng/key 98)))))))

(deftest jittered-particle-steps
  ;; :jitter jitters each chain's step by its own u, as its own hmc's
  ;; would (DESIGN §9, PROVISIONAL): drawn per particle when batched, from
  ;; each chain's key when not. Planted (2026-10-23): one u for every
  ;; particle fails the first; the steps unreported fail the second
  (let [opts {:L 3 :eps 0.3 :jitter 0.5}
        ps (:particles (b/generate scalar [] (cm/choicemap :y 1.3) 6 (rng/key 103)))
        r (mcmc/hmc-particles ps x-sel opts (rng/key 104))
        steps (ts/host (:eps r))]
    (is (b/batched? (:particles r)))
    (is (and (= 6 (count (distinct steps))) (every? #(<= 0.15 % 0.45) steps)) (str steps)))
  (let [chosen (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))
                             x (trace :x (d/normal (if (= k 0) 0.0 3.0) 1))]
                         (trace :y (d/normal x 1))))
        ps (:particles (b/generate chosen [] (cm/choicemap :y 1.3) 4 (rng/key 105)))
        opts {:L 3 :eps 0.3 :jitter 0.5}
        key (rng/key 106)]
    (is (not (b/batched? ps)))
    (is (each-chain-s? ps #(mcmc/hmc-particles %1 x-sel opts %2) #(mcmc/hmc %1 x-sel opts %2) key))
    (is (agree? (mapv #(:eps (mcmc/hmc %1 x-sel opts %2)) (b/traces ps) (rng/split key 4))
                (ts/host (:eps (mcmc/hmc-particles ps x-sel opts key))))))
  (is (= :invalid-jitter (:reason (ts/thrown-data (mcmc/hmc-particles (chains-from scalar (cm/choicemap :x 0.0 :y 1.3) 2)
                                                                      x-sel {:jitter 1.0} (rng/key 107)))))))

(deftest map-optimization-of-every-particle
  ;; each particle's line search is map-optimize's on its trace (no draws:
  ;; exactly, but for round-off). On a Cauchy density from x = 0.5 the
  ;; steps 4 and 2 overshoot and 1 is taken, from x = 3 the first: the
  ;; particles search apart. Planted (2026-10-23): a search without each
  ;; particle's own state (every particle shrinking its step, and taking
  ;; every better one) fails
  (let [cauchy (gen [] (trace :x (d/cauchy 0 1)))
        ps (b/set-choices (chains-from cauchy (cm/choicemap :x 0.0) 6) (cm/choicemap :x (array/array [-3.0 -1.0 -0.5 0.5 1.0 3.0])))
        moved (mcmc/map-optimize-particles ps x-sel {:max-step-size 4.0})]
    (is (b/batched? moved))
    (is (ts/close? (array/stack (mapv #((mcmc/map-optimize % x-sel {:max-step-size 4.0}) :x) (b/traces ps))) (b/choice moved :x))))
  (let [ps (:particles (b/generate scalar [] (cm/choicemap :y 1.3) 6 (rng/key 99)))]
    (testing "to the mode, from every start"
      (let [ps (reduce (fn [ps _] (mcmc/map-optimize-particles ps x-sel)) ps (range 200))]
        (is (b/batched? ps))
        (is (ts/close? (array/full [6] 0.65) (b/choice ps :x) {:atol 1e-6}))))
    (testing "a vector mode behind a splice"
      (let [ps (reduce (fn [ps _] (mcmc/map-optimize-particles ps (sel/select [:latent :z]) {:max-step-size 0.3}))
                       (:particles (b/generate vector-model [] (cm/choicemap :y y2) 3 (rng/key 100)))
                       (range 300))]
        (is (ts/close? (array/broadcast-to posterior-mean [3 2]) (b/choice ps [:latent :z]) {:atol 1e-6}))))))

(deftest particle-kernel-errors
  (let [ps (chains-from scalar (cm/choicemap :x 0.2 :y 1.3) 3)]
    (testing "a start that is not finite, named by particle"
      (let [boxed (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/uniform (- mu 1.0) (+ mu 1.0)))))
            qs (b/set-choices (chains-from boxed (cm/choicemap :mu 0.0 :y 0.5) 3) (cm/choicemap :mu (array/array [0.0 -2.0 0.1])))]
        (is (= {:reason :non-finite-start :kernel :hmc :particles [1]}
               (select-keys (ts/thrown-data (mcmc/hmc-particles qs (sel/select :mu) {:L 2 :eps 0.1} (rng/key 101)))
                            [:reason :kernel :particles])))))
    (testing "no compiled steps"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not compile"
                            (mcmc/hmc-particles ps x-sel {:compile? true} (rng/key 102))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not compile"
                            (mcmc/mala-particles ps x-sel 0.1 {:compile? true}))))))
