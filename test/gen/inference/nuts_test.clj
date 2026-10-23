(ns gen.inference.nuts-test
  "NUTS (gen.inference.mcmc/nuts) and Stan's warmup and sampler
   (gen.inference.nuts): the tree's pieces against Stan's rules, the
   transition's invariance from exact draws, chains against exact
   posteriors (a conjugate normal, a correlated Gaussian of unequal scales,
   a hierarchical model in a Map), the adaptation's targets, divergences,
   keys and errors. Bands are 5 standard errors: of independent draws, or
   of a chain's mean (gen.inference.diagnostics's Monte Carlo standard
   error)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as dg]
            [gen.inference.kernel :as k]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.nuts :as nuts]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Statistics
;; ---------------------------------------------------------------------------

(defn- mean [xs] (/ (reduce + xs) (double (count xs))))

(defn- sample-var [xs]
  (let [m (mean xs)] (/ (reduce + (map #(* (- % m) (- % m)) xs)) (double (dec (count xs))))))

(defn- deviation
  "How far the chain's mean is from truth, in Monte Carlo standard errors."
  [xs truth]
  (/ (Math/abs (double (- (mean xs) truth))) (dg/mcse-mean (vec xs))))

(defn- bulk-ess
  "The chain's bulk ESS. The tests ask for 200 at least: a chain that stays
   put has a wide band, which any mean fits, and at 200 a mean's band is
   under 0.35 sd."
  [xs]
  (dg/ess-bulk (vec xs)))

(defn- z
  "The two-sample z statistic of the means of xs and ys, independent draws."
  [xs ys]
  (/ (- (mean xs) (mean ys)) (Math/sqrt (+ (/ (sample-var xs) (count xs)) (/ (sample-var ys) (count ys))))))

(defn- start-at [model args choices] (:trace (gfi/generate model args choices (rng/key 0))))

(defn- value [tr addr] (ts/number (tr addr)))

;; ---------------------------------------------------------------------------
;; Targets
;; ---------------------------------------------------------------------------

(def unit (gen [] (trace :x (d/normal 0 1))))

(def ys [1.1 0.4 2.3 1.7 0.9])

(def conjugate
  "mu ~ N(0, 2), y_i ~ N(mu, 1): mu | y ~ N(Σy / 5.25, 1 / 5.25)."
  (gen [] (let [mu (trace :mu (d/normal 0 2))]
            (dotimes [i (count ys)] (trace [:y i] (d/normal mu 1))))))

(def conjugate-observations (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(def correlated
  "x ~ N(1, 2), y ~ N(0.02 x, 0.05): a Gaussian of scales 2 and 0.064,
   correlation 0.62. E = (1, 0.02), Var = (4, 0.0041), Cov = 0.08."
  (gen [] (let [x (trace :x (d/normal 1 2))] (trace :y (d/normal (* 0.02 x) 0.05)))))

(def xy (sel/select :x :y))

(defn- correlated-draws
  "n exact draws of (x, y) from correlated."
  [n key]
  (let [[k1 k2] (rng/split key)
        xs (array/->clj (+ 1.0 (* 2.0 (rng/normal k1 [n]))))
        es (array/->clj (rng/normal k2 [n]))]
    (map (fn [x e] [x (+ (* 0.02 x) (* 0.05 e))]) xs es)))

(def group (gen [mu] (let [theta (trace :theta (d/normal mu 1))] (trace :y (d/normal theta 0.5)))))

(def group-ys [1.2 -0.3 2.1 0.7])

(def hierarchical
  "mu ~ N(0, 2); per group j, theta_j ~ N(mu, 1) and y_j ~ N(theta_j, 0.5),
   the groups a Map spliced at :groups."
  (gen [n] (let [mu (trace :mu (d/normal 0 2))]
             (splice :groups (comb/map group) (vec (repeat n mu))))))

(def funnel
  "Neal's funnel: v ~ N(0, 3), x_i ~ N(0, e^{v/2}) for nine x."
  (gen [] (let [v (trace :v (d/normal 0 3))]
            (dotimes [i 9] (trace [:x i] (d/normal 0 (array/exp (* 0.5 v))))))))

(def funnel-selection (apply sel/select :v (map #(vector :x %) (range 9))))

(defn- funnel-start [] (start-at funnel [] (apply cm/choicemap :v 0.0 (mapcat #(vector [:x %] 0.5) (range 9)))))

;; ---------------------------------------------------------------------------
;; The tree
;; ---------------------------------------------------------------------------

(deftest the-generalized-criterion-checks-across-the-join
  ;; Betancourt's criterion with Stan's extra checks, on subtrees by hand (1-d,
  ;; identity metric, so a point's velocity is its momentum). Each case
  ;; fails one check alone; the plain criterion (the full span's only) would
  ;; let the first two continue. Planted bugs (2026-10-23): without the extra
  ;; checks the first two cases continue; with the full span's check
  ;; dropped the last does
  (let [point (fn [p] {:p (array/array [p]) :sharp (array/array [p])})
        tree (fn [beg end rho] {:beg (point beg) :end (point end) :rho (array/array [rho])})
        no-u-turn? (fn [a b] (#'mcmc/no-u-turn? a b (+ (:rho a) (:rho b))))]
    (testing "all three checks pass: the trajectory continues"
      (is (true? (no-u-turn? (tree 1.0 1.0 2.0) (tree 1.0 2.0 3.0)))))
    (testing "a's first point against b's first, its momenta and b's first momentum: 1 + 1 - 1 = 1 against -1"
      (is (false? (no-u-turn? (tree 1.0 1.0 2.0) (tree -1.0 2.0 1.0)))))
    (testing "a's last point against b's last, b's momenta and a's last momentum: 2 - 1 = 1 against -1"
      (is (false? (no-u-turn? (tree 2.0 -1.0 1.0) (tree 1.0 1.0 2.0)))))
    (testing "the span: rho -2 - 2 = -4 against a's first velocity 1 (and b's last)"
      (is (false? (no-u-turn? (tree 1.0 3.0 -2.0) (tree 3.0 1.0 -2.0)))))))

(deftest ^:slow the-trajectory-stops-at-its-first-u-turn
  ;; On N(0, 1) the dynamics are a rotation with turning points (momentum
  ;; 0) half a period, π, apart. The criterion holds until a trajectory
  ;; holds one: then an end's momentum opposes the summed momenta. So a
  ;; trajectory runs at least to the turning point nearer one way, π/4 on
  ;; average, and doubles no further once it spans π: at eps 0.15 a span of
  ;; 15 steps (2.25) is shorter than π and one of 31 (4.65) longer, so the
  ;; last doubling starts from 16 points at most, and a trajectory takes at
  ;; most 31 steps. Planted bugs (2026-10-23): a criterion that always holds
  ;; takes 1023 steps; one of flipped sign stops after one step, a mean of
  ;; 1; backward doublings stepping forwards take up to 55
  (let [eps 0.15
        start (start-at unit [] (cm/choicemap :x 0.3))
        steps (map :n-leapfrog (rest (reductions (fn [r key] (mcmc/nuts (:trace r) (sel/select :x) {:eps eps} key))
                                                 {:trace start} (rng/split (rng/key 11) 300))))]
    (is (<= (apply max steps) 31) (str (frequencies steps)))
    (is (>= (mean steps) (/ Math/PI (* 4 eps))) (str (mean steps)))))

(deftest ^:slow at-depth-one-nuts-accepts-as-metropolis
  ;; With max-depth 1 the trajectory is the start and one step: biased
  ;; progressive sampling moves to the step with probability min(1, w), w =
  ;; e^(H0 - H), exactly the transition's acceptance statistic. Over draws
  ;; with acceptances a_i, Σ(moved - a_i) has variance Σ a_i (1 - a_i).
  ;; Planted bug (2026-10-23): plain multinomial sampling between the
  ;; trajectory and the new subtree (w / (1 + w), Barker's) gives |z| = 33
  (let [start (start-at unit [] (cm/choicemap :x 0.5))
        rs (map #(mcmc/nuts start (sel/select :x) {:eps 1.8 :max-depth 1} %) (rng/split (rng/key 12) 2000))
        moved (map #(if (identical? start (:trace %)) 0.0 1.0) rs)
        as (map :accept-stat rs)]
    (is (every? #(= 1 (:n-leapfrog %)) rs))
    (is (< 0.3 (mean as) 0.9) "a test with power: acceptances away from 0 and 1")
    (is (< (Math/abs (double (/ (reduce + (map - moved as)) (Math/sqrt (reduce + (map #(* % (- 1.0 %)) as)))))) 5.0))))

(deftest a-divergence-ends-the-trajectory
  ;; Stan's divergence: an energy error over 1000, finite or not. From x =
  ;; 0.0005 on N(0, 0.001) a step of 1 lands near -250, an energy error near
  ;; 1e16, still finite. Planted bugs (2026-10-23): without the threshold the
  ;; trajectory goes on, 3 steps, and reports no divergence; a diverged
  ;; subtree not reported as one fails the first assertion too
  (let [tight (gen [] (trace :x (d/normal 0 0.001)))
        start (start-at tight [] (cm/choicemap :x 0.0005))
        r (mcmc/nuts start (sel/select :x) {:eps 1.0} (rng/key 13))]
    (is (= {:diverged? true :n-leapfrog 1 :tree-depth 0} (select-keys r [:diverged? :n-leapfrog :tree-depth])))
    (is (identical? start (:trace r)) "the trajectory is the start alone")
    (is (< (:accept-stat r) 1e-100))))

;; ---------------------------------------------------------------------------
;; Exact posteriors
;; ---------------------------------------------------------------------------

(deftest ^:slow ^:heavy nuts-leaves-the-target-invariant
  ;; Exact draws of the correlated Gaussian, each moved by two transitions
  ;; at a fixed step size and inverse metric, against fresh exact draws:
  ;; the means of x, y, x², y² and xy, each a two-sample z within 5. A
  ;; chain that stays put would pass, so the transitions must move: a step
  ;; size this large (their acceptance statistic is 0.67) weighs the
  ;; trajectory's points unequally, where a bug in their weights shows.
  ;; Planted bugs (2026-10-23): the energy without the metric (½ p·p), |z|
  ;; up to 17; weights e^(h - h0) in place of e^(h0 - h), |z| up to 17 (at a
  ;; step size of 0.6, an acceptance of 0.92, they were within 3)
  (let [n 1500
        starts (map (fn [[x y]] (start-at correlated [] (cm/choicemap :x x :y y))) (correlated-draws n (rng/key 14)))
        step (fn [[tr stats] key] (let [r (mcmc/nuts tr xy {:eps 1.0 :inv-metric [3.0 0.006]} key)]
                                    [(:trace r) (conj stats (:accept-stat r))]))
        runs (map (fn [tr key] (reduce step [tr []] (rng/split key 2))) starts (rng/split (rng/key 15) n))
        moved (map (fn [[tr]] [(value tr :x) (value tr :y)]) runs)
        fresh (correlated-draws n (rng/key 16))]
    (is (< 0.5 (mean (mapcat second runs))) "the transitions move")
    (doseq [[label f] [["x" first] ["y" second] ["x²" #(* (first %) (first %))] ["y²" #(* (second %) (second %))]
                       ["xy" #(* (first %) (second %))]]]
      (is (< (Math/abs (double (z (map f moved) (map f fresh)))) 5.0) label))))

(deftest ^:slow ^:heavy nuts-samples-a-conjugate-posterior
  ;; Planted bug (2026-10-23): the energy without the metric freezes the
  ;; chain (a bulk ESS of 1.3, the mean 7.4 against 1.22: 3.9 standard
  ;; errors)
  (let [{:keys [traces divergences]} (nuts/sample (start-at conjugate [] conjugate-observations) (sel/select :mu)
                                                  {:samples 2000 :warmup 500} (rng/key 17))
        mus (dg/draws traces :mu)
        m (/ (reduce + ys) 5.25)]
    (is (zero? divergences))
    (is (<= 200.0 (bulk-ess mus)))
    (is (> 5.0 (deviation mus m)) (str (mean mus)))
    (is (> 5.0 (deviation (map #(* (- % m) (- % m)) mus) (/ 1.0 5.25))))))

(deftest ^:slow ^:heavy nuts-samples-a-correlated-gaussian-of-unequal-scales
  (let [{:keys [traces stats inv-metric]} (nuts/sample (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0)) xy
                                                       {:samples 2000} (rng/key 18))
        xs (dg/draws traces :x)
        ys (dg/draws traces :y)]
    (testing "the moments"
      (is (<= 200.0 (bulk-ess xs)))
      (is (<= 200.0 (bulk-ess ys)))
      (is (> 5.0 (deviation xs 1.0)))
      (is (> 5.0 (deviation ys 0.02)))
      (is (> 5.0 (deviation (map #(* (- % 1.0) (- % 1.0)) xs) 4.0)))
      (is (> 5.0 (deviation (map #(* (- % 0.02) (- % 0.02)) ys) 0.0041)))
      (is (> 5.0 (deviation (map #(* (- %1 1.0) (- %2 0.02)) xs ys) 0.08))))
    (testing "the adapted metric is the last window's variances: 500 draws, so within 5 standard errors of those"
      ;; the standard error of a variance from the sampling chain's
      ;; squared deviations, scaled to the window's 500 draws
      (is (and (array/backend-array? inv-metric) (= [2] (array/shape inv-metric))))
      (doseq [[i vs truth] [[0 xs 4.0] [1 ys 0.0041]]
              :when (array/backend-array? inv-metric)]
        (let [sq (map #(let [c (- % (mean vs))] (* c c)) vs)
              se (* (dg/mcse-mean (vec sq)) (Math/sqrt (/ (count sq) 500.0)))
              n 500.0
              variance (/ (- (ts/number (array/take inv-metric i)) (* 1e-3 (/ 5.0 (+ n 5.0)))) (/ n (+ n 5.0)))]
          (is (< (Math/abs (double (- variance truth))) (* 5.0 se)) (str i ": " variance)))))
    (testing "the metric shortens the trajectories, against adapting the step size alone"
      ;; about 5 steps against 30. Planted bug (2026-10-23): no window's
      ;; variances taken (the identity kept) runs as long as the step size
      ;; alone, 30.2 steps against 30.4
      (let [unit-metric (:stats (nuts/sample (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0)) xy
                                             {:samples 500 :adapt-metric? false} (rng/key 18)))
            steps (mapv :n-leapfrog stats)
            unit-steps (mapv :n-leapfrog unit-metric)]
        (is (< (+ (mean steps) (* 5.0 (dg/mcse-mean steps))) (- (mean unit-steps) (* 5.0 (dg/mcse-mean unit-steps))))
            (str (mean steps) " against " (mean unit-steps)))))))

(def ^:private hierarchical-posterior
  "The exact posterior of (mu, theta_0..3) given the group ys: [mean cov]."
  (let [n (count group-ys)
        ;; the prior: Var mu = 4, Cov(mu, theta) = 4, Cov(theta_i, theta_j) = 4 + δij
        prior (+ (array/full [(inc n) (inc n)] 4.0) (array/diag (array/array (into [0.0] (repeat n 1.0)))))
        ;; y observes the thetas with variance 0.25
        h (array/array (mapv (fn [j] (mapv #(if (= % (inc j)) 1.0 0.0) (range (inc n)))) (range n)))
        precision (+ (la/inv prior) (* 4.0 (array/matmul (array/transpose h) h)))
        cov (la/inv precision)]
    [(array/matmul cov (* 4.0 (array/matmul (array/transpose h) (array/array group-ys)))) cov]))

(deftest ^:slow ^:heavy nuts-moves-choices-inside-a-map
  ;; nested addresses: mu and each group's theta inside the Map spliced at
  ;; :groups, the observations beside them left alone. Planted bug
  ;; (2026-10-23): the energy without the metric puts every mean and
  ;; variance 450 to 2850 standard errors off
  (let [n (count group-ys)
        observations (apply cm/choicemap (mapcat (fn [j y] [[:groups j :y] y]) (range) group-ys))
        selection (apply sel/select :mu (map #(vector :groups % :theta) (range n)))
        {:keys [traces]} (nuts/sample (start-at hierarchical [n] observations) selection {:samples 1500 :warmup 500}
                                      (rng/key 19))
        [m cov] (map ts/host hierarchical-posterior)
        draws (map (fn [tr] (into [(value tr :mu)] (map #(value tr [:groups % :theta]) (range n)))) traces)]
    (is (every? (fn [tr] (every? #(= (nth group-ys %) (value tr [:groups % :y])) (range n))) traces))
    (doseq [i (range (inc n))]
      (is (<= 200.0 (bulk-ess (map #(nth % i) draws))) (str i))
      (is (> 5.0 (deviation (map #(nth % i) draws) (nth m i))) (str "the mean of " i))
      (is (> 5.0 (deviation (map #(let [c (- (nth % i) (nth m i))] (* c c)) draws) (get-in cov [i i])))
          (str "the variance of " i)))))

;; ---------------------------------------------------------------------------
;; Adaptation
;; ---------------------------------------------------------------------------

(deftest dual-averaging-is-stan-s
  ;; learn_stepsize by hand for two statistics, then its identity: s̄_t is
  ;; the mean of (δ - a_i) over t + t0 and x_t = μ - s̄_t √t / γ, so the
  ;; statistics' mean is δ - (μ - log eps_t) γ (t + t0) / t^1.5. Planted
  ;; bugs (2026-10-23): the sign of s̄'s update flipped fails both; κ's
  ;; weight taken as t^κ fails the hand's x̄
  (let [opts {:target-accept 0.8 :gamma 0.05 :t0 10.0 :kappa 0.75}
        mu (Math/log 10.0)
        da1 (#'nuts/learn (#'nuts/dual-averaging mu) 0.5 opts)
        da2 (#'nuts/learn da1 0.9 opts)
        x1 (- mu (/ (/ 0.3 11.0) 0.05))
        s2 (/ 0.2 12.0)
        x2 (- mu (/ (* s2 (Math/sqrt 2.0)) 0.05))
        w (Math/pow 2.0 -0.75)]
    (is (ts/close? (Math/exp x1) (:eps da1)))
    (is (ts/close? s2 (:s-bar da2)))
    (is (ts/close? (Math/exp x2) (:eps da2)))
    (is (ts/close? (+ (* (- 1.0 w) x1) (* w x2)) (:x-bar da2)))
    (testing "the identity over 200 statistics"
      (let [stats (array/->clj (rng/uniform (rng/key 20) [200]))
            da (reduce #(#'nuts/learn %1 %2 opts) (#'nuts/dual-averaging mu) stats)]
        (is (ts/close? (mean stats) (- 0.8 (/ (* (- mu (Math/log (:eps da))) 0.05 210.0) (Math/pow 200.0 1.5))) {:atol 1e-9}))))))

(deftest the-windows-are-stan-s
  ;; Stan's documented schedule for 1000 iterations: buffers of 75 and 50,
  ;; windows of 25, 50, 100, 200 and 500 (the last stretched). Planted bugs
  ;; (2026-10-23): a window stretched only past the terminal buffer, not
  ;; to it, fails 299's; the variances unregularized fail the last
  (let [defaults @#'nuts/defaults]
    (is (= [[75 99] [100 149] [150 249] [250 449] [450 949]] (#'nuts/windows 1000 defaults)))
    (is (= [[75 99] [100 149] [150 249] [250 449] [450 849] [850 1949]] (#'nuts/windows 2000 defaults)))
    (testing "buffers that do not fit: 15%, 75% and 10%"
      (is (= [[15 89]] (#'nuts/windows 100 defaults))))
    (testing "a window cut short by the terminal buffer"
      (is (= [[75 99] [100 109]] (#'nuts/windows 160 defaults))))
    (testing "a window stretched when the next would end exactly at the terminal buffer (149 + 2·50 = 299 - 50)"
      (is (= [[75 99] [100 248]] (#'nuts/windows 299 defaults))))
    (testing "under 20 iterations, no metric"
      (is (nil? (#'nuts/windows 19 defaults)))))
  (testing "a window's variances, regularized: (3/8)·4 + 1e-3·(5/8)"
    (is (ts/close? (array/array [1.500625 1.500625])
                   (#'nuts/window-variance [(array/array [1.0 2.0]) (array/array [3.0 6.0]) (array/array [5.0 4.0])] nil)))))

(deftest ^:slow ^:heavy dual-averaging-brings-the-acceptance-to-its-target
  ;; With the metric fixed nothing restarts, and over the warmup's second
  ;; half the acceptance statistic averages 0.8 (within 5 Monte Carlo
  ;; standard errors). Planted bug (2026-10-23): s̄'s update with the sign
  ;; flipped drives the acceptance to 0
  (let [{:keys [stats]} (nuts/adapt (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0)) xy
                                    {:warmup 1500 :adapt-metric? false :inv-metric [4.0 0.004]} (rng/key 21))
        as (map :accept-stat (drop 750 stats))]
    (is (> 5.0 (deviation as 0.8)) (str (mean as)))))

;; ---------------------------------------------------------------------------
;; Divergences, kernels, keys and errors
;; ---------------------------------------------------------------------------

(deftest ^:slow ^:heavy the-funnel-reports-divergences
  ;; Neal's funnel: the step size adapted to its body diverges in its neck.
  ;; The sampler's chain seldom reaches the neck after the warmup (0 of 300
  ;; draws here, as of 1000 from three keys), but its warmup does, and a
  ;; chain at a fixed step size does. Planted bug (2026-10-23): a diverged
  ;; subtree not reported as a divergence leaves 0 in both
  (let [{:keys [divergences stats warmup]} (nuts/sample (funnel-start) funnel-selection {:samples 300 :warmup 300}
                                                        (rng/key 22))]
    (is (pos? (:divergences warmup)) (str (:divergences warmup)))
    (is (= (:divergences warmup) (count (filter :diverged? (:stats warmup)))))
    (is (= divergences (count (filter :diverged? stats)))))
  (testing "k/run counts NUTS's divergences"
    (let [step (fn [tr key] (mcmc/nuts tr funnel-selection {:eps 0.8} key))
          {:keys [divergences]} (k/run (k/kernel step) (funnel-start) {:samples 100 :key (rng/key 23)})
          by-hand (second (reduce (fn [[tr d] key] (let [r (step tr key)] [(:trace r) (if (:diverged? r) (inc d) d)]))
                                  [(funnel-start) 0] (rng/split (rng/key 23) 100)))]
      (is (pos? divergences))
      (is (= by-hand divergences)))))

(deftest ^:slow ^:heavy adaptive-hmc-samples-the-correlated-gaussian
  ;; planted bug (2026-10-23): hmc's kinetic energy without the metric puts
  ;; the four moments 2350 to 510,000 standard errors off
  (let [{:keys [traces stats]} (nuts/sample (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0)) xy
                                            {:kernel :hmc :L 8 :samples 2000} (rng/key 24))
        xs (dg/draws traces :x)
        ys (dg/draws traces :y)]
    (is (every? #(= 8 (:n-leapfrog %)) stats))
    (is (<= 200.0 (bulk-ess xs)))
    (is (<= 200.0 (bulk-ess ys)))
    (is (> 5.0 (deviation xs 1.0)))
    (is (> 5.0 (deviation ys 0.02)))
    (is (> 5.0 (deviation (map #(* (- % 1.0) (- % 1.0)) xs) 4.0)))
    (is (> 5.0 (deviation (map #(* (- % 0.02) (- % 0.02)) ys) 0.0041)))))

(deftest the-same-key-gives-the-same-run
  ;; planted bug (2026-10-23): the sampler's transitions drawing their keys
  ;; from the implicit source fail the runs' first three checks and the
  ;; key given in the options
  (let [start (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0))
        run (fn [kernel key] (nuts/sample start xy {:kernel kernel :samples 15 :warmup 30} key))
        choices (fn [r] (map #(ts/host (cm/to-map (:choices %))) (:traces r)))]
    (doseq [kernel [:nuts :hmc]]
      (testing (name kernel)
        (let [a (run kernel (rng/key 25)) b (run kernel (rng/key 25)) c (run kernel (rng/key 26))]
          (is (= (choices a) (choices b)))
          (is (= (:stats a) (:stats b)))
          (is (= (:eps a) (:eps b)))
          (is (not= (choices a) (choices c))))))
    (testing "a key in the options, and the transition's"
      (is (= (:stats (nuts/sample start xy {:samples 5 :warmup 0 :eps 0.3 :key (rng/key 27)}))
             (:stats (nuts/sample start xy {:samples 5 :warmup 0 :eps 0.3} (rng/key 27)))))
      (is (= (dissoc (mcmc/nuts start xy (rng/key 28)) :trace) (dissoc (mcmc/nuts start xy {} (rng/key 28)) :trace)))
      (is (= (dissoc (mcmc/nuts start xy {:key (rng/key 28)}) :trace) (dissoc (mcmc/nuts start xy {} (rng/key 28)) :trace))))))

(deftest a-sample-without-warmup-draws-as-given
  (let [start (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0))
        r (nuts/sample start xy {:samples 4 :warmup 0 :eps 0.25 :inv-metric [4.0 0.004] :max-depth 3} (rng/key 29))]
    (is (= 0.25 (:eps r)))
    (is (every? #(= 0.25 (:eps %)) (:stats r)))
    (is (every? #(<= (:tree-depth %) 3) (:stats r)))
    (is (= {:stats [] :divergences 0} (:warmup r)))))

(deftest compiled-nuts-makes-the-same-moves
  ;; :compile? runs each leapfrog step as one compiled function (one MLX
  ;; graph on MLX), the tree on the host; a model that cannot compile falls
  ;; back to the plain step
  (let [branching (gen [] (let [x (trace :x (d/normal 0 1))]
                            (trace :y (d/normal (if (array/truthy? (array/> x 0)) x (array/negate x)) 1))))
        latent (gen [] (trace :z (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.5 1.0]])))
        vector-model (gen [] (let [z (splice :latent latent)] (trace :y (d/mvnormal z [[0.5 0.0] [0.0 0.5]]))))]
    (doseq [[label gf observations selection inv]
            [["the correlated Gaussian, with a metric" correlated (cm/choicemap :x 0.3 :y 0.1) xy [4.0 0.004]]
             ["an array-valued latent in a splice" vector-model (cm/choicemap :y (array/array [1.0 -0.5]))
              (sel/select [:latent :z]) nil]
             ["a model that branches on a value (falls back)" branching (cm/choicemap :y 1.3) (sel/select :x) nil]]
            k (range 3)]
      (testing label
        (let [tr (:trace (gfi/generate gf [] observations (rng/key 1)))
              move (fn [compile?] (mcmc/nuts tr selection {:eps 0.3 :inv-metric inv :compile? compile?} (rng/key k)))
              plain (move false)
              compiled (move true)]
          (is (= (select-keys plain [:n-leapfrog :tree-depth :diverged?]) (select-keys compiled [:n-leapfrog :tree-depth :diverged?])))
          (is (ts/close? (:accept-stat plain) (:accept-stat compiled)))
          (is (ts/close? (:score (:trace plain)) (:score (:trace compiled)))))))))

(deftest a-start-of-probability-0-is-refused
  ;; as hmc's and mala's (DESIGN §9): y ~ U(mu - 1, mu + 1) holds y = 3 at
  ;; mu = 0 with density 0
  (let [boxed (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/uniform (- mu 1.0) (+ mu 1.0)))))
        start (start-at boxed [] (cm/choicemap :mu 0.0 :y 3.0))]
    (doseq [compile? [false true]]
      (is (= {:reason :non-finite-start :kernel :nuts}
             (select-keys (ts/thrown-data (mcmc/nuts start (sel/select :mu) {:compile? compile?} (rng/key 30)))
                          [:reason :kernel]))))
    (testing "the sampler's, naming its kernel"
      (doseq [kernel [:nuts :hmc]]
        (is (= {:reason :non-finite-start :kernel kernel}
               (select-keys (ts/thrown-data (nuts/sample start (sel/select :mu) {:kernel kernel :samples 10} (rng/key 31)))
                            [:reason :kernel])))))))

(deftest options-are-checked
  (let [start (start-at correlated [] (cm/choicemap :x 0.0 :y 0.0))]
    (testing "the inverse metric: one positive variance, or one per selected value"
      (doseq [[label inv] [["a length" [1.0]] ["a zero" [1.0 0.0]] ["a NaN" [1.0 ##NaN]] ["a negative number" -1.0]
                           ["an array of another shape" (array/ones [3])]]
              [kernel f] [[:nuts #(mcmc/nuts start xy {:inv-metric %} (rng/key 32))]
                          [:hmc #(mcmc/hmc start xy {:inv-metric %} (rng/key 32))]
                          [:mala #(mcmc/mala start xy 0.1 {:inv-metric % :key (rng/key 32)})]]]
        (is (= :invalid-inv-metric (:reason (ts/thrown-data (f inv)))) (str kernel ", " label)))
      (is (some? (mcmc/nuts start xy {:inv-metric 2.0} (rng/key 32))) "one variance for all"))
    (testing "the transition's"
      (is (= :invalid-eps (:reason (ts/thrown-data (mcmc/nuts start xy {:eps 0.0} (rng/key 33))))))
      (is (= :invalid-max-depth (:reason (ts/thrown-data (mcmc/nuts start xy {:max-depth 0} (rng/key 33))))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"selects no choices" (mcmc/nuts start (sel/select :nope) (rng/key 33)))))
    (testing "the sampler's"
      (doseq [opts [{:samples -1} {:samples 10 :kernel :mala} {:samples 10 :target-accept 1.0} {:samples 10 :warmup 1.5}
                    {:samples 10 :window 0} {:samples 10 :kappa 0.0} {:samples 10 :eps ##Inf}]]
        (is (= :invalid-option (:reason (ts/thrown-data (nuts/sample start xy opts (rng/key 34))))) (pr-str opts)))
      (is (= :invalid-option (:reason (ts/thrown-data (nuts/sample start xy {} (rng/key 34))))) "samples are required"))))
