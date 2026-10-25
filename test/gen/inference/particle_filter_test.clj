(ns gen.inference.particle-filter-test
  "Particle filtering on a linear-Gaussian state-space model, against the
   exact Kalman filter: x_t ~ N(a·x_{t-1}, q), y_t ~ N(x_t, r), x_{-1} = 0."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.transform :as tf]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def a-coef 0.8)
(def q 1.0)
(def r 0.5)
(def ys [0.3 1.1 0.4 -0.6 0.2 1.4])

(def kernel
  (gen [t prev]
    (let [x (trace :x (d/normal (* a-coef prev) q))]
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

(def optimal-proposals {:initial initial-proposal :proposal optimal-proposal})

(defn- run-filter
  "A filter over every observation, resampling adaptively; the final state.
   With :proposal, each step after the first proposes from it, on (trace, t,
   y_t), and with :initial the first step, on (y_0); the internal proposal
   otherwise."
  [n key & [{:keys [initial proposal]}]]
  (let [ks (rng/split key (* 2 (count ys)))
        init (pf/initialize model [1 0.0] (observation 0) n
                            (cond-> {:key (ks 0)} initial (assoc :proposal initial :proposal-args [(ys 0)])))]
    (reduce (fn [state t]
              (-> state
                  (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                  (pf/step [(inc t) 0.0] [diff/unknown-change diff/no-change] (observation t)
                           (cond-> {:key (ks (* 2 t))} proposal (assoc :proposal proposal :proposal-args [t (ys t)])))))
            init
            (range 1 (count ys)))))

(defn- exact-lml [] (first (last kalman)))

(defn- mean-and-error
  "The mean of xs and its standard error, from their own spread."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)]
    [m (/ (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))) (Math/sqrt n))]))

(defn- mean-near?
  "Is the mean of xs within five standard errors of mu (and 1e-9, for xs
   without spread)?"
  [xs mu]
  (let [[m se] (mean-and-error xs)]
    (< (ts/abs (- m mu)) (+ 1e-9 (* 5 se)))))

(deftest the-estimate-approaches-the-exact-marginal-likelihood
  (let [state (run-filter 2000 (rng/key 1))]
    ;; the estimate's sd is about sqrt(12.6/n) for this model and filter (from
    ;; the variance of runs of 20 particles): five of them at n = 2000 is 0.40
    (is (< (ts/abs (- (pf/log-ml-estimate state) (exact-lml))) 0.40))
    (testing "the filtering mean of 2000 resampled draws, against the Kalman posterior's"
      ;; the draws come from the filter's approximation, so its own error
      ;; adds to theirs: the statistic's sd is 0.020 over 300 filters and
      ;; 0.0225 over 200 others, twice the √(v/2000) = 0.010 of independent
      ;; draws (the confirmation review: a band of 5·0.010 failed 4 of 40
      ;; keys); five of the larger
      (let [xs (map #(% [(dec (count ys)) :x]) (pf/sample-unweighted-traces state 2000 (rng/key 2)))
            [_ m _] (last kalman)]
        (is (< (ts/abs (- (/ (reduce + xs) 2000) m)) (* 5 0.0225)))))))

(defn- unbiased? [proposal?]
  ;; law 20: E[exp(estimate)] = p(y), checked against the runs' own spread
  (mean-near? (map #(Math/exp (- (pf/log-ml-estimate (run-filter 20 % (when proposal? optimal-proposals))) (exact-lml)))
                   (rng/split (rng/key (if proposal? 4 3)) 400))
              1.0))

(deftest ^:heavy the-estimate-is-unbiased-in-exp-space
  (is (unbiased? false) "with the internal proposal")
  (is (unbiased? true) "with the locally optimal custom proposal"))

(deftest the-optimal-proposal-reduces-variance
  (let [spread (fn [proposal?]
                 (let [ests (map #(pf/log-ml-estimate (run-filter 20 % (when proposal? optimal-proposals)))
                                 (rng/split (rng/key 5) 100))
                       m (/ (reduce + ests) 100)]
                   (/ (reduce + (map #(* (- % m) (- % m)) ests)) 99)))]
    (is (< (spread true) (spread false)))))

(deftest resampling
  (let [state (pf/initialize model [1 0.0] (observation 0) 100 {:key (rng/key 6)})
        resampled (pf/resample state {:key (rng/key 7)})]
    (is (= (vec (repeat 100 0.0)) (:log-weights resampled)))
    (is (array/allclose? (pf/log-ml-estimate state) (pf/log-ml-estimate resampled))
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
    (is (every? true? (map (fn [w w' tr] (< (ts/abs (- (- w' w) (gfi/project tr (sel/select [1 :y])))) (ts/tol 1e-12)))
                           (:log-weights state) (:log-weights state') (:traces state'))))))

(deftest a-custom-proposal-in-a-batched-filter-runs-per-particle
  (let [state (pf/initialize model [1 0.0] (observation 0) 30 {:batched? true :key (rng/key 30)})
        state' (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1)
                        {:proposal optimal-proposal :proposal-args [1 (ys 1)] :key (rng/key 31)})]
    (is (contains? state :particles))
    (is (not (contains? state' :particles)) "the batched state became per-particle traces")
    (is (= 30 (count (pf/traces state'))))
    (is (every? #(= (ys 1) (% [1 :y])) (pf/traces state')))
    (is (every? #(ts/finite? %) (pf/log-weights state')))
    (testing "and initialize with a proposal ignores :batched?"
      (is (contains? (pf/initialize model [1 0.0] (observation 0) 5
                                    {:batched? true :proposal initial-proposal :proposal-args [(ys 0)] :key (rng/key 32)})
                     :traces)))))

;; ---------------------------------------------------------------------------
;; GenParticleFilters: priority resampling, resize, rejuvenation, statistics
;; ---------------------------------------------------------------------------

(defn- unbiased-moves?
  "Over many runs of (move state key), the mean of exp(log-ML estimate)
   equals its value before, within five standard errors."
  [state move]
  (let [before (pf/log-ml-estimate state)]
    (mean-near? (map #(Math/exp (- (pf/log-ml-estimate (move state %)) before)) (rng/split (rng/key 11) 4000)) 1.0)))

(defn- unbiased-resampling? [state opts]
  (unbiased-moves? state #(pf/resample %1 (assoc opts :key %2))))

(deftest priority-resampling-and-resizing-stay-unbiased
  (let [state (run-filter 30 (rng/key 10))]
    (is (unbiased-resampling? state {:priority (fn [lw] (map #(* 0.5 %) lw))}) "tempered priorities")
    (is (unbiased-resampling? state {:priority (fn [lw] (map (constantly 0.0) lw))}) "uniform priorities")
    (is (unbiased-resampling? state {:n 50}) "more particles")
    (is (unbiased-resampling? state {:n 12 :method :stratified}) "fewer, stratified")
    (is (unbiased-resampling? state {:n 12 :method :optimal}) "fewer, optimal"))
  (testing "optimal resampling keeps each particle at most once"
    (let [state (run-filter 30 (rng/key 10))]
      (doseq [n [1 5 12 29]]
        (let [parents (:parents (pf/resample state {:n n :method :optimal :key (rng/key 16)}))]
          (is (= n (count parents) (count (distinct parents))) (str n))))
      (is (= (range 30) (sort (:parents (pf/resample state {:method :optimal :key (rng/key 16)})))) "all of them at the same size")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot add" (pf/resample state {:n 31 :method :optimal}))))
    (testing "the threshold: W = [0.1 0.2 0.7] to two keeps 0.7 and draws one of the others with weight 1/c = 0.3"
      (let [state {:traces [:a :b :c] :log-weights (mapv #(Math/log %) [0.1 0.2 0.7]) :log-ml 0.0}
            r (pf/resample state {:n 2 :method :optimal :key (rng/key 17)})]
        (is (= :c (first (:traces r))))
        (is (every? true? (map ts/close? [(Math/log 1.4) (Math/log 0.6)] (:log-weights r))) (str (:log-weights r)))
        (is (ts/close? (pf/log-ml-estimate state) (pf/log-ml-estimate r))))))
  (testing "plain resampling leaves equal weights, as before"
    (is (every? zero? (:log-weights (pf/resample (run-filter 30 (rng/key 10)) {:key (rng/key 12)})))))
  (testing "resize"
    (is (= 50 (count (pf/traces (pf/resize (run-filter 30 (rng/key 10)) 50 {:key (rng/key 13)})))))))

(deftest rejuvenation-moves-particles-and-keeps-weights
  (let [state (run-filter 40 (rng/key 14))
        t (dec (count ys))
        kernel (k/mh (sel/select [t :x]))
        moved (pf/rejuvenate state kernel (rng/key 15))]
    (is (= (:log-weights state) (:log-weights moved)))
    (is (== (pf/log-ml-estimate state) (pf/log-ml-estimate moved)))
    (is (some true? (map #(not= (% [t :x]) (%2 [t :x])) (pf/traces state) (pf/traces moved))) "some particles moved")))

;; A batched filter stays batched through rejuvenation by a kernel with a
;; batched form (genclj-xgz4), and its statistics read the collection
;; without rebuilding its traces.

(defn- batched-filter
  "A batched filter of n particles after the first two observations."
  [n seed]
  (-> (pf/initialize model [1 0.0] (observation 0) n {:batched? true :key (rng/key seed)})
      (pf/step [2 0.0] [diff/unknown-change diff/no-change] (observation 1) {:key (rng/key (inc seed))})))

(defn- as-traces
  "The same particles as scalar traces."
  [state]
  (-> state (dissoc :particles) (assoc :traces (pf/traces state))))

(def drift
  "An asymmetric proposal for x_0: x' ~ N(x/2 + 0.2, 0.7), its density back
   from x' other than there."
  (gen [tr] (trace [0 :x] (d/normal (+ (* 0.5 (tr [0 :x])) 0.2) 0.7))))

(def batched-kernels
  "Kernels with a batched form, and what they move."
  {"MH of a selection" [(k/mh (sel/select [0 :x])) [0 :x]]
   "MH with a proposal" [(k/mh drift []) [0 :x]]
   "a chain" [(k/chain (k/mh (sel/select [1 :x])) (k/mh drift [])) [0 :x]]
   "a kern" [(k/kern [] (doseq [t (range 2)] (k/mh (sel/select [t :x])))) [1 :x]]
   "HMC" [(k/hmc (sel/select [0 :x] [1 :x]) {:L 3 :eps 0.3}) [1 :x]]
   "MALA" [(k/mala (sel/select [0 :x] [1 :x]) 0.2) [1 :x]]})

(deftest a-batched-filter-stays-batched-through-rejuvenation
  ;; Planted bugs (2026-10-25): rejuvenate unbatching always fails every
  ;; case; the batched form of kernel/mh of a selection dropped fails it, "a
  ;; chain" and "a kern"; a chain's batched form making its first move only
  ;; fails "a chain" and "a kern".
  (let [state (batched-filter 40 60)]
    (doseq [[label [kernel addr]] batched-kernels]
      (testing label
        (let [moved (pf/rejuvenate state kernel (rng/key 61))]
          (is (b/batched? (:particles moved)))
          (is (nil? (:fallback moved)))
          (is (= (:log-weights state) (:log-weights moved)))
          (is (== (pf/log-ml-estimate state) (pf/log-ml-estimate moved)))
          (is (not= (ts/host (pf/choice state addr)) (ts/host (pf/choice moved addr))) "some particles moved"))))))

(deftest a-kernel-without-a-batched-form-moves-one-particle-at-a-time
  ;; Planted bug (2026-10-25): rejuvenate not recording the fallback fails
  ;; the first; not refusing it under :required, the second.
  (let [state (batched-filter 12 62)]
    (doseq [[label kernel] {"NUTS" (k/nuts (sel/select [0 :x]) {:eps 0.3 :max-depth 3})
                            "a function" (fn [tr key] (mh/mh tr (sel/select [0 :x]) key))
                            "a mixture" (k/mix [(k/mh (sel/select [0 :x]))] [1.0])
                            "a chain with one" (k/chain (k/mh (sel/select [1 :x])) (k/kern ^:trace [_tr] (k/mh (sel/select [0 :x]))))}]
      (testing label
        (let [moved (pf/rejuvenate state kernel (rng/key 63))]
          (is (re-find #"no batched form" (str (:fallback moved))))
          (is (= 12 (count (:traces moved))))
          (is (not (contains? moved :particles)))
          (is (= (:log-weights state) (:log-weights moved))))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #":batched\? :required.*no batched form"
                              (pf/rejuvenate (assoc state :batched? :required) kernel (rng/key 63))))))
    (testing "a batched form that cannot run batched records why"
      (let [branching (gen [tr] (if (array/> (tr [0 :x]) 0.0)
                                  (trace [0 :x] (d/normal (tr [0 :x]) 0.5))
                                  (trace [0 :x] (d/normal (tr [0 :x]) 0.6))))
            moved (pf/rejuvenate state (k/mh branching []) (rng/key 65))]
        (is (nil? (:fallback state)))
        (is (re-find #"Cannot branch" (:fallback moved)))
        (is (not (b/batched? (:particles moved))))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #":batched\? :required"
                              (pf/rejuvenate (assoc state :batched? :required) (k/mh branching []) (rng/key 65))))))))

(defn- agree?
  "Do the means of xs and ys, independent samples, agree within five
   standard errors of their difference?"
  [xs ys]
  (let [[mx sx] (mean-and-error xs)
        [my sy] (mean-and-error ys)]
    (< (ts/abs (- mx my)) (+ 1e-12 (* 5 (Math/sqrt (+ (* sx sx) (* sy sy))))))))

(deftest batched-and-scalar-rejuvenation-agree
  ;; Each particle's batched move is distributed as its scalar move: from
  ;; 3000 copies of one particle, the batched and the scalar rejuvenation's
  ;; values agree in mean and in square, and in how many moved, at 5 SE.
  ;; Planted bugs (2026-10-25): the batched MH accepting only moves with a
  ;; ratio of at least 1 fails the MH cases; HMC's batched form given twice
  ;; the step size fails "HMC", MALA's twice its tau "MALA".
  (let [start (pf/replicate (pf/view (batched-filter 40 66) [0]) 3000)
        scalar (as-traces start)]
    (is (b/batched? (:particles start)))
    (doseq [[label [kernel addr]] batched-kernels]
      (testing label
        (let [x0 (first (ts/host (pf/choice start addr)))
              b (ts/host (pf/choice (pf/rejuvenate start kernel (rng/key 67)) addr))
              s (ts/host (pf/choice (pf/rejuvenate scalar kernel (rng/key 68)) addr))
              moved (fn [xs] (map #(if (== % x0) 0.0 1.0) xs))]
          (is (agree? b s) "the means")
          (is (agree? (map #(* % %) b) (map #(* % %) s)) "the second moments")
          (is (agree? (moved b) (moved s)) "how many moved"))))))

(deftest the-statistics-read-a-batched-state-without-rebuilding-it
  ;; pf/choice, mean, variance and proportions of a batched state are its
  ;; traces' own, read from the collection: with gen.batched/traces made to
  ;; throw, they still run. Planted bug (2026-10-25): mean reading
  ;; (traces state) fails the last two testing blocks.
  (let [state (batched-filter 30 70)
        scalar (as-traces state)
        x1 #(% [1 :x])
        positive #(array/> (% [1 :x]) 0.0)]
    (testing "the values are the traces'"
      (is (ts/close? (pf/choice state [1 :x]) (pf/choice scalar [1 :x])))
      (is (= [30] (array/shape (pf/choice state [1 :x]))))
      (is (ts/close? (pf/mean state x1) (pf/mean scalar x1)))
      (is (ts/close? (pf/variance state x1) (pf/variance scalar x1)))
      (is (= (set (keys (pf/proportions state positive))) (set (keys (pf/proportions scalar positive)))))
      (is (every? true? (map ts/close? (vals (into (sorted-map) (pf/proportions state positive)))
                             (vals (into (sorted-map) (pf/proportions scalar positive))))))
      (is (ts/close? (pf/mean state #(array/stack [(% [0 :x]) (% [1 :x])]))
                     (array/stack [(pf/mean scalar #(% [0 :x])) (pf/mean scalar x1)]))
          "a vector-valued statistic"))
    (testing "without rebuilding the traces"
      (with-redefs [b/traces (fn [_] (throw (ex-info "rebuilt" {})))]
        (is (ts/finite? (ts/number (pf/mean state x1))))
        (is (ts/finite? (ts/number (pf/variance state x1))))
        (is (map? (pf/proportions state positive)))
        (is (= [30] (array/shape (pf/choice state [1 :x]))))))
    (testing "a function that reads the host runs on each trace"
      (is (ts/close? (pf/mean state #(array/item (% [1 :x]))) (pf/mean scalar x1)))
      (let [up? #(pos? (array/item (% [1 :x])))]
        (is (= (keys (pf/proportions scalar up?)) (keys (pf/proportions state up?))))
        (is (every? true? (map ts/close? (vals (pf/proportions scalar up?)) (vals (pf/proportions state up?))))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #":batched\? :required"
                            (pf/mean (assoc state :batched? :required) #(array/item (% [1 :x]))))))))

(deftest each-particle-is-evaluated-as-it-is-moved
  ;; One evaluation per particle, as it is moved, so that a lazy backend holds
  ;; one particle's graph at a time (docs/array-api.md §13, as importance
  ;; sampling does); eval! is a no-op on the JVM, so the order is recorded.
  ;; Plants: step, rejuvenate and move-reweight building every particle and
  ;; then evaluating each and all once more (the post-M5 review's I3-04,
  ;; [:move :move :move :eval :eval :eval :eval]) fail their three; initialize
  ;; evaluating nothing fails its own.
  (let [events (atom [])
        log! #(swap! events conj %)
        order (fn [f] (reset! events []) (f) @events)
        expected (into [] cat (repeat 3 [:move :eval]))
        logged (fn [kernel] (fn [tr key] (log! :move) (kernel tr key)))
        state (pf/initialize model [1 0.0] (observation 0) 3 {:key (rng/key 16)})
        [eval! generate update'] [array/eval! gfi/generate gfi/update]]
    (with-redefs [array/eval! (fn [x] (log! :eval) (eval! x))]
      (with-redefs [gfi/generate (fn [& args] (log! :move) (apply generate args))
                    gfi/update (fn [& args] (log! :move) (apply update' args))]
        (is (= expected (order #(pf/initialize model [1 0.0] (observation 0) 3 {:key (rng/key 16)}))) "initialize")
        (is (= expected (order #(pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1) {:key (rng/key 17)})))
            "step"))
      (is (= expected (order #(pf/rejuvenate state (logged (k/mh (sel/select [0 :x]))) (rng/key 18)))) "rejuvenate")
      (is (= expected (order #(pf/move-reweight state (logged (pf/reweighting (sel/select [0 :x]))) (rng/key 19))))
          "move-reweight"))))

(deftest weighted-statistics
  (let [state {:traces [{:x 1.0} {:x 3.0}] :log-weights [(Math/log 0.25) (Math/log 0.75)] :log-ml 0.0}
        f :x]
    (is (ts/close? 2.5 (pf/mean state f)))
    (is (ts/close? 0.75 (pf/variance state f)) "0.25·1.5² + 0.75·0.5²")
    (is (= {1.0 0.25 3.0 0.75} (update-vals (pf/proportions state f) #(/ (Math/round (* % 100.0)) 100.0))))))

;; ---------------------------------------------------------------------------
;; GenParticleFilters: stratification, move-reweight, replicate, coalesce,
;; introduce
;; ---------------------------------------------------------------------------

(defn- lml-ratio [state] (Math/exp (- (pf/log-ml-estimate state) (exact-lml))))

(defn- unbiased-runs?
  "Law 20 after f: over runs filters of 20 particles, the mean of
   exp(estimate - exact) is 1 within five standard errors."
  [f seed]
  (mean-near? (map #(let [[k1 k2] (rng/split %)] (lml-ratio (f (run-filter 20 k1) k2))) (rng/split (rng/key seed) 400))
              1.0))

(def last-x [(dec (count ys)) :x])

(def random-walk
  "A move of the last state by a symmetric random walk."
  (gen [tr] (trace last-x (d/normal (tr last-x) 0.5))))

(def wide-walk (gen [tr] (trace last-x (d/normal (tr last-x) 0.8))))

(deftest move-reweight-adds-the-moves-weights
  (let [state (run-filter 30 (rng/key 40))
        moved (pf/move-reweight state (pf/reweighting (sel/select last-x)) (rng/key 41))]
    (testing "regenerating: the weight is regenerate's"
      (is (every? true? (map (fn [tr w w' key]
                               (ts/close? (- w' w) (:weight (gfi/regenerate tr (sel/select last-x) key)) {:atol (ts/tol 1e-9)}))
                             (pf/traces state) (:log-weights state) (:log-weights moved) (rng/split (rng/key 41) 30))))
      (is (some true? (map #(not= (% last-x) (%2 last-x)) (pf/traces state) (pf/traces moved)))))
    (testing "a symmetric proposal: the weight is the model's ratio"
      (let [moved (pf/move-reweight state (pf/reweighting random-walk []) (rng/key 42))]
        (is (every? true? (map (fn [tr tr' w w'] (ts/close? (- w' w) (- (:score tr') (:score tr)) {:atol (ts/tol 1e-9)}))
                               (pf/traces state) (pf/traces moved) (:log-weights state) (:log-weights moved))))))
    (testing "a separate backward proposal scores the way back"
      (let [moved (pf/move-reweight state (pf/reweighting random-walk [] {:backward wide-walk :backward-args []}) (rng/key 42))
            lp (fn [x mu sd] (ts/number (dc/log-prob (d/normal mu sd) x)))]
        (is (every? true? (map (fn [tr tr' w w']
                                 (ts/close? (- w' w) (+ (- (:score tr') (:score tr))
                                                    (- (lp (tr' last-x) (tr last-x) 0.5))
                                                    (lp (tr last-x) (tr' last-x) 0.8))
                                            {:atol (ts/tol 1e-9)}))
                               (pf/traces state) (pf/traces moved) (:log-weights state) (:log-weights moved))))))))

(deftest ^:slow ^:heavy move-reweight-keeps-the-estimate-unbiased
  (is (unbiased-runs? #(pf/move-reweight %1 (pf/reweighting (sel/select last-x)) %2) 43) "regenerating")
  (is (unbiased-runs? #(pf/move-reweight %1 (pf/reweighting random-walk []) %2) 44) "a random walk")
  (is (unbiased-runs? #(pf/move-reweight %1 (pf/reweighting random-walk [] {:backward wide-walk :backward-args []}) %2) 45)
      "a separate backward proposal"))

(deftest replicate-and-dereplicate
  (let [state (pf/initialize model [1 0.0] (observation 0) 4 {:key (rng/key 50)})]
    (doseq [[layout parents] [[:contiguous [0 0 0 1 1 1 2 2 2 3 3 3]] [:interleaved [0 1 2 3 0 1 2 3 0 1 2 3]]]]
      (let [replicated (pf/replicate state 3 {:layout layout})]
        (is (= parents (:parents replicated)) (str layout))
        (is (ts/close? (pf/log-ml-estimate state) (pf/log-ml-estimate replicated)))
        (let [back (pf/dereplicate replicated 3 {:layout layout})]
          (is (= (:traces state) (:traces back)) "keepfirst undoes replicate")
          (is (= (:log-weights state) (:log-weights back))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not groups of 3" (pf/dereplicate state 3)))
    ;; the post-M5 review's I3-14. Planted bug (2026-10-22): the case
    ;; without a default throws a bare IllegalArgumentException
    (is (= :first (:method (ts/thrown-data (pf/dereplicate (pf/replicate state 2) 2 {:method :first}))))))
  (testing "a proposal that writes an observation is named, as in importance sampling (the review's I3-08)"
    ;; planted bug (2026-10-22): the filter's own propose-and-generate, as
    ;; before, throws cm/merge's "merge: the choice maps overlap"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"overlap the observations"
                          (pf/initialize model [1 0.0] (observation 0) 4
                                         {:proposal (gen [] (trace [0 :y] (d/normal 0 1))) :key (rng/key 52)}))))
  (testing "a batched state stays batched"
    (let [state (pf/initialize model [1 0.0] (observation 0) 4 {:batched? true :key (rng/key 50)})
          replicated (pf/replicate state 2 {:layout :interleaved})]
      (is (contains? replicated :particles))
      (is (= (concat (pf/traces state) (pf/traces state)) (pf/traces replicated)))
      (is (= (pf/traces state) (pf/traces (pf/dereplicate replicated 2 {:layout :interleaved}))))
      (is (= 8 (count (pf/traces (pf/move-reweight replicated (pf/reweighting (sel/select [0 :x])) (rng/key 51)))))))))

(deftest dereplicating-by-sampling
  ;; replicas moved apart, then one of each group drawn: the group's mean
  ;; weight keeps the estimate exactly; draws follow the weights in a group
  (let [state (-> (pf/initialize model [1 0.0] (observation 0) 4 {:key (rng/key 51)})
                  (pf/replicate 3)
                  (pf/move-reweight (pf/reweighting (sel/select [0 :x])) (rng/key 52)))
        lw (:log-weights state)]
    (is (ts/close? (pf/log-ml-estimate state)
                   (pf/log-ml-estimate (pf/dereplicate state 3 {:method :sample :key (rng/key 53)}))
                   {:atol (ts/tol 1e-12)}))
    (let [n 4000
          picks (map #(first (:parents (pf/dereplicate state 3 {:method :sample :key %}))) (rng/split (rng/key 54) n))
          p (mapv #(Math/exp %) (:log-normalized-weights (w/normalize (subvec lw 0 3))))]
      (doseq [i (range 3)]
        (let [freq (/ (count (filter #{i} picks)) (double n))]
          (is (< (ts/abs (- freq (p i))) (* 5 (Math/sqrt (/ (* (p i) (- 1 (p i))) n)))) (str i ": " freq " against " (p i))))))))

(deftest coalescing
  (let [state (pf/initialize model [1 0.0] (observation 0) 12 {:key (rng/key 55)})
        resampled (pf/resample state {:priority (fn [lw] (map #(* 0.5 %) lw)) :key (rng/key 56)})
        coalesced (pf/coalesce resampled)
        distinct-parents (distinct (:parents resampled))]
    (is (= (count distinct-parents) (count (pf/traces coalesced))) "one particle per distinct trace")
    (is (= (set (map #(% [0 :x]) (pf/traces resampled))) (set (map #(% [0 :x]) (pf/traces coalesced)))))
    (is (ts/close? (pf/log-ml-estimate resampled) (pf/log-ml-estimate coalesced) {:atol (ts/tol 1e-12)}))
    (testing "the weights add up within a group"
      (let [x (fn [tr] (tr [0 :x]))
            total (fn [st v] (reduce + (keep (fn [[tr w]] (when (= v (x tr)) (Math/exp w)))
                                              (map vector (pf/traces st) (:log-weights st)))))
            ratio (/ (count (pf/traces coalesced)) 12.0)]
        (doseq [v (map x (pf/traces coalesced))]
          (is (ts/close? (* ratio (total resampled v)) (total coalesced v))))))
    (testing ":by"
      (is (= 1 (count (pf/traces (pf/coalesce state {:by (constantly :same)}))))))))

(deftest introducing-particles
  (let [state (pf/resample (pf/initialize model [1 0.0] (observation 0) 10 {:key (rng/key 57)}) {:key (rng/key 58)})
        grown (pf/introduce state (observation 0) 5 {:key (rng/key 59)})]
    (is (= 15 (count (pf/traces grown))))
    (is (zero? (:log-ml grown)))
    (is (every? #(ts/close? (:log-ml state) %) (take 10 (:log-weights grown))) "the old particles carry the estimate")
    (is (= (:log-weights (pf/initialize model [1 0.0] (observation 0) 5 {:key (rng/key 59)})) (drop 10 (:log-weights grown))))
    (testing "with a proposal"
      (let [grown (pf/introduce state (observation 0) 5 {:proposal initial-proposal :proposal-args [(ys 0)] :key (rng/key 60)})]
        (is (= 15 (count (pf/log-weights grown))))))))

(def discrete-model
  (gen [] (let [k (trace :k (d/categorical [0.2 0.5 0.3]))] (trace :y (d/normal k 1)) k)))

(def discrete-evidence
  (Math/log (reduce + (map-indexed #(* %2 (Math/exp (ts/number (dc/log-prob (d/normal %1 1) 1.7)))) [0.2 0.5 0.3]))))

(deftest stratified-initialization
  (let [strata (pf/choice-product [[:k [0 1 2]]])
        obs (cm/choicemap :y 1.7)]
    (is (= [{:k 0} {:k 1} {:k 2}] (map cm/to-map strata)))
    (testing "a stratum for every particle: the estimate is exact"
      (doseq [layout [:contiguous :interleaved]]
        (let [state (pf/initialize discrete-model [] obs 6 {:strata strata :layout layout :key (rng/key 61)})]
          (is (ts/close? discrete-evidence (pf/log-ml-estimate state) {:atol (ts/tol 1e-12)}))
          (is (= (if (= layout :contiguous) [0 0 1 1 2 2] [0 1 2 0 1 2])
                 (mapv #(long (% :k)) (pf/traces state)))))))
    (testing "the remainder goes to random strata, unbiased"
      (let [ratios (mapv #(Math/exp (- (pf/log-ml-estimate (pf/initialize discrete-model [] obs 4 {:strata strata :key %}))
                                       discrete-evidence))
                         (rng/split (rng/key 62) 4000))]
        (is (pos? (second (mean-and-error ratios))))
        (is (mean-near? ratios 1.0))))
    (is (= 4 (count (pf/choice-product {:a [1 2] :b [true false]}))))))

(def re-propose
  "A step's proposal that also moves the previous state."
  (gen [tr t] (trace [(dec t) :x] (d/normal (tr [(dec t) :x]) 0.3)) (trace [t :x] (d/normal 0 1))))

(def move-back (gen [tr t] (trace [(dec t) :x] (d/normal (tr [(dec t) :x]) 0.3))))

(deftest a-step-with-a-backward-proposal
  (let [state (pf/initialize model [1 0.0] (observation 0) 5 {:key (rng/key 70)})
        state' (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1)
                        {:proposal re-propose :proposal-args [1] :backward move-back :backward-args [1] :key (rng/key 71)})]
    ;; the weight: update weight - log q(forward choices) + log q_bwd(discard)
    (doseq [[tr tr' w w' key] (map vector (:traces state) (:traces state') (:log-weights state) (:log-weights state')
                                   (rng/split (rng/key 71) 5))]
      (let [[k-propose k-update] (rng/split key)
            {:keys [choices weight]} (gfi/propose re-propose [tr 1] k-propose)
            {up :weight discard :discard t2 :trace} (gfi/update tr [2 0.0] [diff/unknown-change diff/no-change]
                                                                (cm/merge (observation 1) choices) k-update)]
        (is (= (tr' [0 :x]) (t2 [0 :x])))
        (is (not= (tr [0 :x]) (tr' [0 :x])) "the old state moved")
        (is (ts/close? (- w' w) (+ (- up weight) (:weight (gfi/assess move-back [t2 1] discard)))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed or removed"
                          (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1)
                                   {:proposal re-propose :proposal-args [1] :key (rng/key 72)}))
        "without a backward proposal, still loud")))

;; ---------------------------------------------------------------------------
;; SMC samplers and SMCP3: updating translators as move-reweight kernels
;; ---------------------------------------------------------------------------

(defn- step-args [t] {:p-new-args [(inc t) 0.0] :p-argdiffs [diff/unknown-change diff/no-change]
                      :new-observations (observation t)})

(deftest an-updating-translator-without-a-transform-is-a-proposal-step
  (let [state (pf/initialize model [1 0.0] (observation 0) 8 {:key (rng/key 80)})
        stepped (pf/step state [2 0.0] [diff/unknown-change diff/no-change] (observation 1)
                         {:proposal optimal-proposal :proposal-args [1 (ys 1)] :key (rng/key 81)})
        translated (pf/move-reweight state (tf/updating-translator (assoc (step-args 1) :q-forward optimal-proposal
                                                                          :q-forward-args [1 (ys 1)]))
                                     (rng/key 81))]
    (is (= (map :choices (:traces stepped)) (map :choices (:traces translated))))
    (is (every? true? (map #(ts/close? %1 %2) (:log-weights stepped) (:log-weights translated))))))

(def standard-normal (gen [_] (trace :z (d/normal 0 1))))

(defn- optimal-moments
  "The mean and sd of p(x_t | x_{t-1}, y_t)."
  [prev y]
  (let [v (/ 1.0 (+ (/ 1.0 (* q q)) (/ 1.0 (* r r))))]
    [(* v (+ (/ (* a-coef prev) (* q q)) (/ y (* r r)))) (Math/sqrt v)]))

(defn- reparameterized
  "x_t = m + s·z from z ~ N(0, 1), m and s the optimal proposal's; x_{t-1}
   is read and kept, so it is an unchanged output."
  [t]
  (tf/pair (tf/transform [model-in aux-in model-out aux-out]
             (let [[m s] (optimal-moments (tf/read model-in [(dec t) :x] :continuous) (ys t))]
               (tf/write model-out [t :x] (+ m (* s (tf/read aux-in :z :continuous))) :continuous)))
           (tf/transform [model-in aux-in model-out aux-out]
             (let [[m s] (optimal-moments (tf/read model-in [(dec t) :x] :continuous) (ys t))]
               (tf/write aux-out :z (/ (- (tf/read model-in [t :x] :continuous) m) s) :continuous)))))

(deftest smcp3-a-reparameterized-proposal-weighs-as-the-optimal-one
  (let [state (pf/initialize model [1 0.0] (observation 0) 8 {:key (rng/key 82)})
        [forward _] (reparameterized 1)
        moved (pf/move-reweight state (tf/updating-translator (assoc (step-args 1) :q-forward standard-normal
                                                                     :transform forward :check? true))
                                (rng/key 83))]
    ;; z ~ N(0, 1) mapped by x = m + s·z has density N(x; m, s): the weight
    ;; is the score change less log N(x_1; m, s)
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (let [[m s] (optimal-moments (tr [0 :x]) (ys 1))]
        (is (= (tr [0 :x]) (tr' [0 :x])))
        (is (ts/close? (- w' w) (- (- (:score tr') (:score tr)) (ts/number (dc/log-prob (d/normal m s) (tr' [1 :x]))))))))
    (testing "the round trip catches a wrong inverse"
      (let [[f _] (tf/pair (first (reparameterized 1))
                           (tf/transform [model-in aux-in model-out aux-out]
                             (tf/read model-in [0 :x] :continuous)
                             (tf/write aux-out :z (tf/read model-in [1 :x] :continuous) :continuous)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not give back the forward"
                              (pf/move-reweight state (tf/updating-translator (assoc (step-args 1) :q-forward standard-normal
                                                                                     :transform f :check? true))
                                                (rng/key 84))))))))

(defn- run-smc-samplers
  "The filter with each step an SMC-samplers move: x_{t-1} moved by a random
   walk and x_t proposed, the old x_{t-1} weighed back by the walk."
  [n key]
  (let [ks (rng/split key (* 2 (count ys)))
        init (pf/initialize model [1 0.0] (observation 0) n {:key (ks 0)})]
    (reduce (fn [state t]
              (-> state
                  (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                  (pf/move-reweight (tf/updating-translator (assoc (step-args t) :q-forward re-propose :q-forward-args [t]
                                                                   :q-backward move-back :q-backward-args [t]))
                                    (ks (* 2 t)))))
            init
            (range 1 (count ys)))))

(deftest ^:slow ^:heavy smc-samplers-with-a-backward-kernel-stay-unbiased
  (let [ratios (mapv #(lml-ratio (run-smc-samplers 20 %)) (rng/split (rng/key 85) 400))]
    (is (mean-near? ratios 1.0) (pr-str (mean-and-error ratios)))))

(deftest an-extending-translator-with-a-transform
  ;; x_1 = 2z, z ~ N(0, 1): the weight is the score change less log N(x_1; 0, 2)
  (let [state (pf/initialize model [1 0.0] (observation 0) 6 {:key (rng/key 86)})
        doubled (tf/transform [model-in aux-in model-out aux-out]
                  (tf/write model-out [1 :x] (* 2.0 (tf/read model-in :z :continuous)) :continuous))
        moved (pf/move-reweight state (tf/extending-translator (assoc (step-args 1) :q-forward standard-normal :transform doubled))
                                (rng/key 87))]
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (is (ts/close? (- w' w) (- (- (:score tr') (:score tr)) (ts/number (dc/log-prob (d/normal 0 2) (tr' [1 :x])))))))))

(deftest an-smc-samplers-step-weighs-the-way-back
  ;; the increment: score change - log q(x_0', x_1; x_0) + log L(x_0; x_0')
  (let [state (pf/initialize model [1 0.0] (observation 0) 6 {:key (rng/key 88)})
        moved (pf/move-reweight state (tf/updating-translator (assoc (step-args 1) :q-forward re-propose :q-forward-args [1]
                                                                     :q-backward move-back :q-backward-args [1]))
                                (rng/key 89))]
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (let [fwd (:weight (gfi/assess re-propose [tr 1] (cm/choicemap [0 :x] (tr' [0 :x]) [1 :x] (tr' [1 :x]))))
            bwd (:weight (gfi/assess move-back [tr' 1] (cm/choicemap [0 :x] (tr [0 :x]))))]
        (is (not= (tr [0 :x]) (tr' [0 :x])))
        (is (ts/close? (- w' w) (+ (- (:score tr') (:score tr)) (- bwd fwd))))))))

(def walk-and-standard (gen [tr] (trace :w (d/normal (tr [0 :x]) 0.3)) (trace :z (d/normal 0 1))))
(def walk-back (gen [tr] (trace :back (d/normal (tr [0 :x]) 0.3))))

(def move-and-propose
  "x_0 moved to w, x_1 = m(w) + s·z from the optimal moments given w, the old
   x_0 handed back: |det J| = s."
  (tf/transform [model-in aux-in model-out aux-out]
    (let [x0 (tf/read model-in [0 :x] :continuous)
          w (tf/read aux-in :w :continuous)
          [m s] (optimal-moments w (ys 1))]
      (tf/write model-out [0 :x] w :continuous)
      (tf/write model-out [1 :x] (+ m (* s (tf/read aux-in :z :continuous))) :continuous)
      (tf/write aux-out :back x0 :continuous))))

(deftest smcp3-with-backward-choices
  ;; the walk is symmetric, so the weight is the score change less log N(x_1; m(x_0'), s)
  (let [state (pf/initialize model [1 0.0] (observation 0) 6 {:key (rng/key 90)})
        moved (pf/move-reweight state (tf/updating-translator (assoc (step-args 1) :q-forward walk-and-standard
                                                                     :q-backward walk-back :transform move-and-propose))
                                (rng/key 91))]
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (let [[m s] (optimal-moments (tr' [0 :x]) (ys 1))]
        (is (not= (tr [0 :x]) (tr' [0 :x])))
        (is (ts/close? (- w' w) (- (- (:score tr') (:score tr)) (ts/number (dc/log-prob (d/normal m s) (tr' [1 :x]))))))))))

;; ---------------------------------------------------------------------------
;; A proposal with encapsulated randomness (thesis §4.5.4)
;; ---------------------------------------------------------------------------

(def coin-proposal
  "The locally optimal mean shifted by ±0.6, the sign an untraced coin: the
   proposal's density is estimated, ξ = N(x_t; m ± 0.6, 0.5)."
  (gen [tr t y]
    (let [[m _] (optimal-moments (tr [(dec t) :x]) y)]
      (trace [t :x] (d/normal (+ m (array/where (untraced (d/bernoulli 0.5)) 0.6 -0.6)) 0.5)))))

(deftest ^:slow ^:heavy smc-with-an-encapsulated-proposal-is-unbiased
  ;; §4.5.4 (law audit E19): a step weighs log p - log ξ, ξ the proposal's
  ;; estimate on the ω it proposed with, and E[1/ξ] = 1/q given the choice
  ;; (Eq. 4.4), so E[exp(estimate)] = p(y) still: over 2000 filters of 20
  ;; particles, within five standard errors of the runs' own spread (0.98 ±
  ;; 0.024; with 5 particles the standard error was 0.079, a ratio of 72
  ;; among them). Planted bug (2026-10-20): the proposal's score assessed on
  ;; a fresh ω (a coin apart from the one that moved x_t), which inflates
  ;; 1/ξ, gives 3500 ± 74
  (let [ratios (mapv #(lml-ratio (run-filter 20 % {:proposal coin-proposal})) (rng/split (rng/key 92) 2000))]
    (is (mean-near? ratios 1.0) (pr-str (mean-and-error ratios)))))

(deftest edge-cases-of-the-genparticlefilters-operations
  (testing "optimal resampling skips particles of weight 0"
    (let [state {:traces [:a :b :c :d] :log-weights [0.0 ##-Inf 1.0 2.0] :log-ml 0.0}
          r (pf/resample state {:n 3 :method :optimal :key (rng/key 100)})]
      (is (= #{:a :c :d} (set (:traces r))))
      (is (ts/close? (pf/log-ml-estimate state) (pf/log-ml-estimate r)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive weight"
                            (pf/resample (assoc state :log-weights [0.0 ##-Inf ##-Inf 2.0]) {:n 3 :method :optimal})))))
  (testing "a group of weight 0 dereplicates to its first particle"
    (let [r (pf/dereplicate (pf/replicate {:traces [:a :b] :log-weights [0.0 ##-Inf] :log-ml 0.0} 2) 2
                            {:method :sample :key (rng/key 101)})]
      (is (= [:a :b] (:traces r)))
      (is (= ##-Inf (second (:log-weights r))))))
  (testing "a priority of 0 where the weight is not is loud"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"bias"
                          (pf/resample {:traces [:a :b] :log-weights [0.0 0.0] :log-ml 0.0}
                                       {:priority (fn [_] [0.0 ##-Inf]) :key (rng/key 102)}))))
  (testing "misuse"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"layout" (pf/replicate {:traces [:a] :log-weights [0.0]} 2 {:layout :diagonal})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at least one stratum"
                          (pf/initialize model [1 0.0] (observation 0) 4 {:strata [] :key (rng/key 103)})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no :proposal"
                          (pf/step (pf/initialize model [1 0.0] (observation 0) 2 {:key (rng/key 104)})
                                   [2 0.0] [diff/unknown-change diff/no-change] (observation 1) {:backward move-back})))))

;; ---------------------------------------------------------------------------
;; GenParticleFilters: substate views
;; ---------------------------------------------------------------------------

(def view-idx [2 7 8 19 25])

(deftest a-view-is-a-state-of-its-particles
  (let [state (run-filter 30 (rng/key 30))
        v (pf/view state view-idx)
        lw (mapv (:log-weights state) view-idx)
        xs (mapv #(ts/number (% last-x)) (pf/traces v))]
    (is (not (zero? (:log-ml state))) "the filter resampled, so the source's estimate so far counts")
    (is (= (mapv (pf/traces state) view-idx) (pf/traces v)))
    (is (ts/close? (w/ess lw) (pf/ess v)))
    (testing "the estimate: the source's so far plus the log mean of the view's weights (GenParticleFilters)"
      (is (ts/close? (+ (:log-ml state) (- (w/log-sum-exp lw) (Math/log (count view-idx)))) (pf/log-ml-estimate v)))
      (is (ts/close? (pf/log-ml-estimate state) (pf/log-ml-estimate (pf/view state (range 30)))) "every particle"))
    (is (ts/close? (reduce + (map #(* (Math/exp (- (ts/number %1) (w/log-sum-exp lw))) %2) lw xs))
                   (pf/mean v #(% last-x))))
    (testing "of a batched filter"
      (let [batched (pf/initialize model [1 0.0] (observation 0) 12 {:batched? true :key (rng/key 31)})]
        (is (ts/close? (array/array (mapv #(ts/number (% [0 :x])) (mapv (pf/traces batched) [1 4 5])))
                       (array/array (mapv #(ts/number (% [0 :x])) (pf/traces (pf/view batched [1 4 5]))))))))))

(defn- outside-unchanged? [state state' idx]
  (let [in (set idx)]
    (every? (fn [i] (or (in i) (and (identical? ((:traces state) i) ((:traces state') i))
                                    (identical? ((:log-weights state) i) ((:log-weights state') i)))))
            (range (count (:traces state))))))

(deftest an-operation-on-a-view-moves-its-particles-alone
  (let [state (run-filter 30 (rng/key 32))]
    (testing "resampling keeps the view's total weight, each particle at the mean (GenParticleFilters)"
      (let [r (pf/update-view state view-idx pf/resample {:key (rng/key 33)})
            lw (mapv (:log-weights state) view-idx)
            mean-weight (- (w/log-sum-exp lw) (Math/log (count view-idx)))]
        (is (outside-unchanged? state r view-idx))
        (is (every? #(ts/close? mean-weight ((:log-weights r) %)) view-idx))
        (is (ts/close? (pf/log-ml-estimate state) (pf/log-ml-estimate r)))
        (is (= (:log-ml state) (:log-ml r)))
        (testing "and :parents gives each particle's source"
          (is (every? (fn [i] (if (some #{i} view-idx)
                                (and (some #{((:parents r) i)} view-idx)
                                     (identical? ((:traces r) i) ((:traces state) ((:parents r) i))))
                                (= i ((:parents r) i))))
                      (range 30))))))
    (testing "a step on a view is the view's step"
      (let [args [7 0.0] argdiffs [diff/unknown-change diff/no-change] obs (cm/choicemap [6 :y] 0.5)
            stepped (pf/update-view state view-idx pf/step args argdiffs obs {:key (rng/key 34)})
            alone (pf/step (pf/view state view-idx) args argdiffs obs {:key (rng/key 34)})]
        (is (outside-unchanged? state stepped view-idx))
        (is (= (ts/host (mapv :choices (:traces alone))) (ts/host (mapv #(:choices ((:traces stepped) %)) view-idx))))
        (is (ts/close? (array/array (mapv ts/number (:log-weights alone)))
                       (array/array (mapv #(ts/number ((:log-weights stepped) %)) view-idx))))
        (is (= 7 (count (:retval ((:traces stepped) 2)))))
        (is (= 6 (count (:retval ((:traces stepped) 3)))))))
    (testing "rejuvenation"
      (let [moved (pf/update-view state view-idx pf/rejuvenate (k/mh (sel/select last-x)) (rng/key 35))]
        (is (outside-unchanged? state moved view-idx))
        (is (= (:log-weights state) (:log-weights moved)))))))

(defn- one-step-filter
  "30 particles resampled once (an estimate so far), stepped, then given
   weights of a set moderate spread, host values, so that a view's
   resampling draws several on every backend. (A filter's own weights
   differ by float32 rounding, enough to change the resampled parents: on
   MLX that view was one particle's, ESS 1.12.)"
  [key]
  (let [[k1 k2 k3] (rng/split key 3)]
    (-> (pf/initialize model [1 0.0] (observation 0) 30 {:key k1})
        (pf/resample {:key k2})
        (pf/step [2 0.0] [diff/unknown-change diff/no-change] (observation 1) {:key k3})
        (assoc :log-weights (mapv #(* 0.8 (Math/sin (* 1.7 %))) (range 30))))))

(deftest resampling-a-view-keeps-the-estimate-unbiased
  (let [state (one-step-filter (rng/key 36))
        weighted-sum (fn [st] (reduce + (map #(* (Math/exp (ts/number %1)) (ts/number (%2 [1 :x]))) (:log-weights st) (:traces st))))
        sum-unbiased? (fn [opts]
                        ;; E[Σ w'·x'] = Σ w·x over many resamplings of the view, within five standard errors
                        (mean-near? (map #(weighted-sum (pf/update-view state view-idx pf/resample (assoc opts :key %)))
                                         (rng/split (rng/key 37) 4000))
                                    (weighted-sum state)))]
    (is (< 2.0 (w/ess (mapv (:log-weights state) view-idx))) "the view's weights are spread")
    (is (unbiased-moves? state #(pf/update-view %1 view-idx pf/resample {:key %2 :priority (fn [lw] (map (partial * 0.5) lw))}))
        "tempered priorities")
    (is (unbiased-moves? state #(pf/update-view %1 view-idx pf/resample {:key %2 :method :optimal :n 5})) "optimal")
    (is (sum-unbiased? {}) "a weighted sum")
    (is (sum-unbiased? {:priority (fn [lw] (map (constantly 0.0) lw))}) "a weighted sum, uniform priorities")))

(deftest a-view-keeps-f-s-resampled-flag
  ;; maybe-resample on a view of equal weights does not resample (review 3:
  ;; the state kept its earlier :resampled? true)
  (let [state (pf/maybe-resample (run-filter 30 (rng/key 40)) {:ess-threshold 31 :key (rng/key 41)})
        r (pf/update-view state [0 1 2] pf/maybe-resample {:key (rng/key 42)})]
    (is (true? (:resampled? state)))
    (is (false? (:resampled? r)))))

(deftest a-view-s-misuse-is-loud
  (let [state (run-filter 6 (rng/key 38))]
    (doseq [idx [[] [1 1] [6] [-1] [0.5]]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"distinct particle indices" (pf/view state idx)) (pr-str idx)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"keeps their number"
                          (pf/update-view state [0 1 2] pf/resize 4 {:key (rng/key 39)})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"keeps their number"
                          (pf/update-view state [0 1 2] pf/dereplicate 3)))))
