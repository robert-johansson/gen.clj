;; The code of book/023-gradient-mcmc.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.gradient-mcmc
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as is]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.inference.nuts :as nuts]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])

(def ys [8.23 5.87 3.99 2.59 0.23 -0.66 -3.53 -6.91 -7.24 -9.90])

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(def line (sel/select :slope :intercept))

;; book: exact
(defn posterior
  "The exact posterior of the line given points (xs, ys): its mean and
   covariance, slope first."
  [xs ys]
  (let [X (array/array (mapv (fn [x] [x 1.0]) xs))
        Xt (array/transpose X)
        prior-precision (array/diag (array/array [(/ 1.0 4.0) (/ 1.0 100.0)]))
        covariance (linalg/inv (+ prior-precision (array/matmul Xt X)))]
    {:mean (array/matmul covariance (array/matmul Xt (array/array ys)))
     :cov covariance}))

(def exact (posterior xs ys))

(array/->clj (:mean exact))
;=> [-1.979 10.143]

(array/->clj (array/sqrt (array/diag (:cov exact))))
;=> [0.1097 0.6807]

;; book: axes
(defn axes
  "The principal axes of a 2×2 covariance, the longer first: [sd
   direction] of each, the direction a unit vector [slope intercept]."
  [cov]
  (let [[[a b] [_ c]] (array/->clj cov)
        middle (/ (+ a c) 2)
        half-gap (Math/sqrt (+ (* (/ (- a c) 2) (/ (- a c) 2)) (* b b)))
        axis (fn [lambda]
               (let [v [b (- lambda a)]
                     norm (Math/sqrt (reduce + (map * v v)))]
                 [(Math/sqrt lambda) (mapv #(/ % norm) v)]))]
    [(axis (+ middle half-gap)) (axis (- middle half-gap))]))

(def ridge (axes (:cov exact)))

(def length (first (first ridge)))

(def width (first (second ridge)))

[length width (/ length width)]
;=> [0.6877 0.05043 13.6]

;; book: exact-draws
(def exact-draws
  (let [p (d/mvnormal (:mean exact) (:cov exact))]
    (mapv #(array/->clj (dc/sample p %)) (rng/split (rng/key 1) 300))))

;; book: ridge-figure
;; figure: figures/023-ridge.svg
(svg/plot [{:points exact-draws :color "gray" :r 2}]
          {:x-label "slope" :y-label "intercept"})

;; book: run-chain
(defn run-chain
  "A chain of the kernel k (a function of a trace and a key that returns
   {:trace …}) from the trace t: burn steps, then samples steps kept.
   Returns {:draws {addr [x …]} :steps [r …]}, the values at addrs as host
   numbers and each kept step's result without its trace. Only the
   current trace is held."
  [k t addrs {:keys [samples burn key] :or {burn 0}}]
  (let [step-keys (rng/split key (+ burn samples))]
    (loop [i 0, t t, draws (zipmap addrs (repeat [])), steps []]
      (if (= i (count step-keys))
        {:draws draws :steps steps}
        (let [r (k t (step-keys i))
              ;; one step's graph at a time on a lazy backend
              t' (gfi/eval-trace! (:trace r))
              keep? (>= i burn)
              read (fn [m a] (update m a conj (array/->clj (t' a))))]
          (recur (inc i) t'
                 (if keep? (reduce read draws addrs) draws)
                 (if keep? (conj steps (dissoc r :trace)) steps)))))))

(defn acceptance
  "The fraction of a chain's kept steps that accepted their move."
  [{:keys [steps]}]
  (/ (count (filter :accepted? steps)) (double (count steps))))

;; book: starts
(def start-particles
  (:particles (batched/generate line-model [xs] (observations ys) 4
                                (rng/key 2))))

(def starts (batched/traces start-particles))

(mapv #(% :slope) starts)
;=> [4.626 0.515 1.459 3.788] ; float64

;; book: walk
(def random-walk
  (gen [tr]
    (trace :slope (d/normal (tr :slope) 0.1))
    (trace :intercept (d/normal (tr :intercept) 0.5))))

(defn walk
  "One MH move by the random walk."
  [t key]
  (mh/mh t random-walk [] key))

(def walk-chains
  (mapv (fn [t k]
          (run-chain walk t [:slope :intercept]
                     {:samples 500 :burn 200 :key k}))
        starts (rng/split (rng/key 3) 4)))

(defn chains
  "The draws at addr of each of runs, a vector of chains."
  [runs addr]
  (mapv #(get-in % [:draws addr]) runs))

(mapv acceptance walk-chains)
;=> [0.412 0.388 0.348 0.428] ; float64

;; book: walk-ess
(mapv diag/ess-bulk (chains walk-chains :slope))
;=> [25.0 24.1 3.1 21.9] ; float64

(every? #(< (diag/ess-bulk %) 100) (chains walk-chains :slope))
;=> true

;; book: walk-figure
;; figure: figures/023-walk.svg
(defn trace-plot
  "Each chain's draws against their index, one color a chain."
  [chains]
  (svg/plot (map (fn [chain color]
                   {:points (map-indexed vector chain) :color color :r 1.2})
                 chains ["darkorange" "seagreen" "firebrick" "purple"])
            {:x-label "draw" :y-label "slope" :y-range [-2.45 -1.35]}))

(trace-plot (chains walk-chains :slope))

;; book: r-hat
(def walk-r-hat
  [(diag/r-hat (chains walk-chains :slope))
   (diag/r-hat (chains walk-chains :intercept))])

walk-r-hat
;=> [1.111 1.129] ; float64

(every? #(> % 1.01) walk-r-hat)
;=> true

;; book: center
(def center
  "The trace at the exact posterior mean, rounded."
  (:trace (gfi/generate line-model [xs]
                        (cm/merge (cm/choicemap :slope -1.979
                                                :intercept 10.143)
                                  (observations ys))
                        (rng/key 0))))

(defn at
  "The trace t moved to the line [slope intercept]."
  [t [slope intercept]]
  (:trace (gfi/update t (cm/choicemap :slope slope :intercept intercept)
                      (rng/key 0))))

;; book: gradient
(defn gradient
  "The gradient of the trace's log density with respect to the line:
   [d/dslope d/dintercept], by choice-gradients."
  [t]
  (let [g (:gradients (grad/choice-gradients t line))]
    [(cm/get-choice g :slope) (cm/get-choice g :intercept)]))

(gradient (at center [-1.5 9.0]))
;=> [-121.785 -14.92]

;; book: gradient-exact
(def precision (linalg/inv (:cov exact)))

(defn exact-gradient
  "The gradient of the exact posterior's log density at the line theta:
   precision · (mean - theta)."
  [theta]
  (array/matmul precision (- (:mean exact) (array/array theta))))

(array/allclose? (array/array (gradient (at center [-1.5 9.0])))
                 (exact-gradient [-1.5 9.0]))
;=> true

;; book: pull
(defn norm
  "The Euclidean length of the vector v."
  [v]
  (Math/sqrt (array/item (reduce + (map * v v)))))

(def mean (array/->clj (:mean exact)))

(defn off-mean
  "The line k standard deviations from the mean along the axis [sd
   direction]."
  [k [sd direction]]
  (mapv + mean (mapv #(* k sd %) direction)))

(defn pull
  "The size of the gradient one standard deviation from the mean along
   the axis."
  [axis]
  (norm (gradient (at center (off-mean 1 axis)))))

[(pull (first ridge)) (pull (second ridge))]
;=> [1.454 19.83]

(array/allclose? (/ (pull (second ridge)) (pull (first ridge)))
                 (/ length width))
;=> true

;; book: run-particles
(defn run-particles
  "Chains of the kernel k (a function of particles and a key, as
   mcmc/hmc-particles) from the particles ps, one a particle: burn steps,
   then samples steps kept. Returns {:draws {addr chains} :acceptance r},
   a chain of host numbers per particle at each of addrs, and the fraction
   of the kept steps' moves accepted."
  [k ps addrs {:keys [samples burn key] :or {burn 0}}]
  (let [step-keys (rng/split key (+ burn samples))]
    (loop [i 0, ps ps, rows (zipmap addrs (repeat [])), accepted []]
      (if (= i (count step-keys))
        {:draws (update-vals rows #(apply mapv vector %))
         :acceptance (/ (count (filter true? accepted))
                        (double (count accepted)))}
        (let [{ps' :particles moved :accepted?} (k ps (step-keys i))
              keep? (>= i burn)
              read (fn [m addr]
                     (update m addr conj
                             (array/->clj (batched/choice ps' addr))))]
          (recur (inc i) ps'
                 (if keep? (reduce read rows addrs) rows)
                 (if keep? (into accepted moved) accepted)))))))

;; book: mala
(def at-center
  "Four particles, each at the trace center."
  (batched/take (:particles (batched/generate line-model [xs]
                                              (:choices center) 1
                                              (rng/key 0)))
                [0 0 0 0]))

(def mala-identity
  (run-particles #(mcmc/mala-particles %1 line 0.004 %2) at-center
                 [:slope :intercept] {:samples 1000 :burn 100
                                      :key (rng/key 4)}))

[(:acceptance mala-identity)
 (diag/ess-bulk (get-in mala-identity [:draws :slope]))]
;=> [0.615 20.0] ; float64

;; book: metric-order
(cm/addresses (grad/continuous-selection center line))
;=> [[:intercept] [:slope]]

(def inv-metric
  (let [[v-slope v-intercept] (array/->clj (array/diag (:cov exact)))]
    (cm/to-array (cm/choicemap :slope v-slope :intercept v-intercept))))

(array/->clj inv-metric)
;=> [0.4634 0.01204]

;; book: mala-metric
(def mala-scaled
  (run-particles #(mcmc/mala-particles %1 line 0.2
                                       {:inv-metric inv-metric :key %2})
                 at-center [:slope :intercept]
                 {:samples 1000 :burn 100 :key (rng/key 4)}))

[(:acceptance mala-scaled)
 (diag/ess-bulk (get-in mala-scaled [:draws :slope]))]
;=> [0.573 168.4] ; float64

(let [ess #(diag/ess-bulk (get-in % [:draws :slope]))]
  [(> (ess mala-scaled) (* 1.5 (ess mala-identity)))
   (< (ess mala-scaled) (* 0.1 4000))])
;=> [true true]

;; book: leapfrog
(defn line-of
  "The trace's line, [slope intercept]."
  [t]
  [(t :slope) (t :intercept)])

(defn leapfrog
  "One leapfrog step of size eps from [t p], a trace and its momentum
   [p-slope p-intercept]: half a step of the momentum along the gradient,
   a step of the position along the momentum, and the other half step of
   the momentum, at the new position."
  [eps [t p]]
  (let [kick (fn [p t] (mapv + p (mapv #(* (/ eps 2) %) (gradient t))))
        p-half (kick p t)
        t' (at t (mapv + (line-of t) (mapv #(* eps %) p-half)))]
    [t' (kick p-half t')]))

(defn energy
  "The Hamiltonian of [t p]: the potential energy -log p(t) plus the
   kinetic energy |p|²/2."
  [[t p]]
  (+ (- (:score t)) (* 0.5 (reduce + (map * p p)))))

;; book: trajectory
(def released
  "At rest, one and a half lengths down the ridge and one width off it."
  [(at center (mapv + (off-mean -1.5 (first ridge))
                    (mapv - (off-mean 1 (second ridge)) mean)))
   [0.0 0.0]])

(def trajectory (vec (take 41 (iterate (partial leapfrog 0.05) released))))

(defn along
  "How far along the ridge the trace's line is, in lengths from the
   mean."
  [t]
  (let [[_ direction] (first ridge)]
    (/ (reduce + (map * (mapv - (line-of t) mean) direction)) length)))

[(along (first (first trajectory))) (along (first (peek trajectory)))]
;=> [-1.5 1.46]

(let [h0 (energy (first trajectory))]
  (apply max (map #(abs (- (energy %) h0)) trajectory)))
;=> 0.12

;; book: trajectory-figure
;; figure: figures/023-trajectory.svg
(defn along-path
  "Points every 1/k of the way along each segment of the path ps, which
   drawn small make a line."
  [ps k]
  (mapcat (fn [[x0 y0] [x1 y1]]
            (for [i (range k) :let [f (/ i (double k))]]
              [(+ x0 (* f (- x1 x0))) (+ y0 (* f (- y1 y0)))]))
          ps (rest ps)))

(let [path (map #(line-of (first %)) trajectory)]
  (svg/plot [{:points exact-draws :color "gray" :r 2}
             {:points (along-path path 8) :color "steelblue" :r 0.7}
             {:points path :color "steelblue" :r 3}]
            {:x-label "slope" :y-label "intercept"}))

;; book: hmc
(def hmc-chains
  (run-particles #(mcmc/hmc-particles %1 line {:L 20 :eps 0.05} %2)
                 start-particles [:slope :intercept]
                 {:samples 500 :burn 100 :key (rng/key 5)}))

(:acceptance hmc-chains)
;=> 0.92 ; float64

(def hmc-r-hat
  [(diag/r-hat (get-in hmc-chains [:draws :slope]))
   (diag/r-hat (get-in hmc-chains [:draws :intercept]))])

(every? #(< % 1.01) hmc-r-hat)
;=> true

(mapv diag/ess-bulk (get-in hmc-chains [:draws :slope]))
;=> [326.1 365.4 359.0 435.5] ; float64

;; book: hmc-check
(defn agrees?
  "Do the chains' mean and standard deviation agree with the exact ones
   within five Monte Carlo standard errors?"
  [chains exact-mean exact-sd]
  (let [{:keys [mean sd mcse-mean mcse-sd]} (diag/summary chains)]
    [(< (abs (- mean exact-mean)) (* 5 mcse-mean))
     (< (abs (- sd exact-sd)) (* 5 mcse-sd))]))

(def exact-sd (array/->clj (array/sqrt (array/diag (:cov exact)))))

[(agrees? (get-in hmc-chains [:draws :slope]) (mean 0) (exact-sd 0))
 (agrees? (get-in hmc-chains [:draws :intercept]) (mean 1) (exact-sd 1))]
;=> [[true true] [true true]]

;; book: hmc-figure
;; figure: figures/023-hmc.svg
(trace-plot (get-in hmc-chains [:draws :slope]))

;; book: cost
(defn ess-per
  "The chains' bulk ESS at :slope per unit of work, n units in all."
  [runs n]
  (/ (diag/ess-bulk runs) n))

[(ess-per (chains walk-chains :slope) (* 4 500))
 (ess-per (get-in hmc-chains [:draws :slope]) (* 4 500 21))]
;=> [0.0155 0.0353] ; float64

;; book: step-size
(defn acceptance-at
  "HMC's acceptance from the trace center with L = 20 steps of size eps,
   over 100 moves."
  [eps]
  (acceptance (run-chain #(mcmc/hmc %1 line {:L 20 :eps eps} %2) center
                         [:slope] {:samples 100 :key (rng/key 6)})))

[(acceptance-at (* 1.8 width)) (acceptance-at (* 2.2 width))]
;=> [0.66 0.0] ; float64

[(> (acceptance-at (* 1.8 width)) 0.2) (acceptance-at (* 2.2 width))]
;=> [true 0.0]

;; book: nuts
(def nuts-run (nuts/sample (first starts) line {:samples 1000} (rng/key 7)))

[(:eps nuts-run) (array/->clj (:inv-metric nuts-run))]
;=> [0.309 [0.472 0.0124]] ; float64

[(:divergences nuts-run) (:divergences (:warmup nuts-run))]
;=> [0 13] ; float64

;; book: nuts-stats
(defn mean-of
  "The mean of the numbers xs."
  [xs]
  (/ (reduce + xs) (double (count xs))))

(def nuts-steps (mean-of (map :n-leapfrog (:stats nuts-run))))

[nuts-steps (frequencies (map :tree-depth (:stats nuts-run)))]
;=> [7.17 {1 115, 2 471, 3 387, 4 27}] ; float64

;; book: nuts-check
(def nuts-draws
  {:slope (diag/draws (:traces nuts-run) :slope)
   :intercept (diag/draws (:traces nuts-run) :intercept)})

[(agrees? (:slope nuts-draws) (mean 0) (exact-sd 0))
 (agrees? (:intercept nuts-draws) (mean 1) (exact-sd 1))]
;=> [[true true] [true true]]

;; book: nuts-metric
(def ess-rate
  "The fraction of the draws' number that their effective number is."
  (/ (diag/ess-bulk (:slope nuts-draws)) 1000.0))

(let [ratios (map / (array/->clj (:inv-metric nuts-run))
                  (array/->clj inv-metric))
      relative-se (Math/sqrt (/ 2.0 (* 500 ess-rate)))]
  (every? #(< (abs (- % 1.0)) (* 5 relative-se)) ratios))
;=> true

;; book: tilt
(def correlation
  (let [[[a b] [_ c]] (array/->clj (:cov exact))]
    (/ b (Math/sqrt (* a c)))))

correlation
;=> -0.886

(def scaled-sds
  "The posterior's standard deviations along its axes, in the units of a
   metric of its exact variances: √(1 ± |r|), r the correlation."
  [(Math/sqrt (+ 1 (abs correlation))) (Math/sqrt (- 1 (abs correlation)))])

(apply / scaled-sds)
;=> 4.06

;; book: centered
(def x-bar (mean-of xs))

(def centered-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          level (trace :level (d/normal (* slope x-bar) 10))
          intercept (- level (* slope x-bar))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def centered-run
  (nuts/sample (:trace (gfi/generate centered-model [xs] (observations ys)
                                     (rng/key 2)))
               (sel/select :slope :level) {:samples 1000} (rng/key 7)))

(def centered-steps (mean-of (map :n-leapfrog (:stats centered-run))))

centered-steps
;=> 4.02 ; float64

;; book: centered-check
(def centered-slopes (diag/draws (:traces centered-run) :slope))

(agrees? centered-slopes (mean 0) (exact-sd 0))
;=> [true true]

(defn per-gradient
  "A run's bulk ESS at :slope per leapfrog step."
  [draws steps]
  (/ (diag/ess-bulk draws) (* 1000 steps)))

[(per-gradient (:slope nuts-draws) nuts-steps)
 (per-gradient centered-slopes centered-steps)]
;=> [0.0356 0.159] ; float64

(> (per-gradient centered-slopes centered-steps)
   (* 2.5 (per-gradient (:slope nuts-draws) nuts-steps)))
;=> true

;; book: hmm-model
(def hmm-step
  (gen [t z]
    (let [z' (trace :z (d/normal z 0.5))]
      (trace :y (d/normal z' 1.0))
      z')))

(def hmm (comb/unfold hmm-step))

(def hmm-ys
  [0.77 -1.59 2.11 -0.15 0.07 0.69 0.73 1.32 2.77 2.17
   3.08 3.85 4.09 1.73 3.27 2.81 0.67 2.54 3.32 1.85])

(def n (count hmm-ys))

(def hmm-observations
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) hmm-ys)))

(def states (sel/from-paths (for [t (range n)] [t :z])))

;; book: hmm-exact
(defn smoothed
  "The exact posterior of the states given ys: their means and standard
   deviations. Its precision is the random walk's, D'D/0.5², plus the
   observations', I/1², D taking differences z_t - z_{t-1} (z_{-1} = 0);
   its mean is the precision's inverse times ys."
  [ys]
  (let [n (count ys)
        D (array/array (vec (for [i (range n)]
                              (vec (for [j (range n)]
                                     (cond (= i j) 1.0
                                           (= j (dec i)) -1.0
                                           :else 0.0))))))
        precision (+ (/ (array/matmul (array/transpose D) D) 0.25)
                     (array/eye n))
        cov (linalg/inv precision)]
    {:mean (array/->clj (array/matmul cov (array/array ys)))
     :sd (array/->clj (array/sqrt (array/diag cov)))}))

(def hmm-exact (smoothed hmm-ys))

(mapv #(Math/round (* 100 %)) (take 5 (:mean hmm-exact)))
;=> [13 9 48 46 59]

;; book: hmm-nuts
(def hmm-start
  (:trace (gfi/generate hmm [n 0.0] hmm-observations (rng/key 8))))

(def hmm-adapted (nuts/adapt hmm-start states {:warmup 300} (rng/key 9)))

(def hmm-run
  (run-chain #(mcmc/nuts %1 states (select-keys hmm-adapted
                                                [:eps :inv-metric]) %2)
             (:trace hmm-adapted) (for [t (range n)] [t :z])
             {:samples 500 :key (rng/key 10)}))

(mean-of (map :n-leapfrog (:steps hmm-run)))
;=> 13.2 ; float64

;; book: hmm-check
(def hmm-summaries
  (mapv #(diag/summary (get-in hmm-run [:draws [% :z]])) (range n)))

(every? true? (map (fn [{:keys [mean mcse-mean]} exact-mean]
                     (< (abs (- mean exact-mean)) (* 5 mcse-mean)))
                   hmm-summaries (:mean hmm-exact)))
;=> true

(apply min (map :ess-bulk hmm-summaries))
;=> 275.5 ; float64

;; book: hmm-is
(:ess (is/importance-sampling hmm [n 0.0] hmm-observations 2000
                              {:key (rng/key 11)}))
;=> 12.0 ; float64

;; book: hmm-walk
(def block-walk
  (gen [tr]
    (dotimes [t n]
      (trace [t :z] (d/normal (tr [t :z]) 0.15)))))

(def hmm-walk
  (run-chain #(mh/mh %1 block-walk [] %2) hmm-start
             (for [t (range n)] [t :z])
             {:samples 5000 :burn 1000 :key (rng/key 12)}))

(def walk-min-ess
  (apply min (map #(diag/ess-bulk (get-in hmm-walk [:draws [% :z]]))
                  (range n))))

[(acceptance hmm-walk) walk-min-ess]
;=> [0.34 8.59] ; float64

(def per-evaluation
  "The worst state's ESS per proposal of the walk, and per gradient of
   NUTS."
  [(/ walk-min-ess 5000)
   (/ (apply min (map :ess-bulk hmm-summaries))
      (* 500 (mean-of (map :n-leapfrog (:steps hmm-run)))))])

per-evaluation
;=> [0.0017 0.0417] ; float64

(< (* 4 (first per-evaluation)) (second per-evaluation))
;=> true

;; book: hmm-figure
;; figure: figures/023-hmm.svg
(defn polyline
  "The function of x through the points (0, ys 0), (1, ys 1), …, linear
   between them."
  [ys]
  (fn [x]
    (let [i (min (long x) (- (count ys) 2))
          f (- x i)]
      (+ (* (- 1 f) (ys i)) (* f (ys (inc i)))))))

(let [{m :mean sd :sd} hmm-exact]
  (svg/plot [{:points (map-indexed vector hmm-ys) :color "gray"}
             {:curves [(polyline m)] :color "black"}
             {:curves [(polyline (mapv + m (mapv #(* 2 %) sd)))
                       (polyline (mapv - m (mapv #(* 2 %) sd)))]
              :color "black" :opacity 0.4 :width 1}
             {:points (map-indexed #(vector %1 (:mean %2)) hmm-summaries)
              :color "steelblue" :r 3}]
            {:x-label "step t" :y-label "state" :x-range [0 (dec n)]}))

;; book: mixture
(def mixture
  (gen []
    (let [z (trace :z (d/categorical [0.4 0.6]))
          x (trace :x (d/normal (+ 1.0 (* 3.0 z)) 1.0))]
      (trace :y (d/normal x 0.5))
      z)))

(def mixture-start
  (:trace (gfi/generate mixture [] (cm/choicemap :y 2.0) (rng/key 13))))

;; book: discrete-error
;; error: choice-gradients: the choice at [:z] is not continuous
#_(mcmc/hmc mixture-start (sel/select :z :x) {:L 5 :eps 0.3} (rng/key 14))

;; book: gibbs-model
(def gibbs-mixture (enum/with-exact-proposal mixture [:z]))

(def gibbs-start
  (:trace (gfi/generate gibbs-mixture [] (cm/choicemap :y 2.0)
                        (rng/key 13))))

(mapv #(:weight (gfi/regenerate gibbs-start (sel/select :z) %))
      (rng/split (rng/key 15) 3))
;=> [0.0 0.0 0.0]

;; book: gibbs
(defn gibbs-hmc
  "A Gibbs move on z, then an HMC move on x."
  [t key]
  (let [[k-z k-x] (rng/split key)
        t (:trace (mh/mh t (sel/select :z) k-z))]
    (mcmc/hmc t (sel/select :x) {:L 5 :eps 0.3} k-x)))

(def mixture-run
  (run-chain gibbs-hmc gibbs-start [:z :x]
             {:samples 3000 :burn 100 :key (rng/key 16)}))

;; book: mixture-exact
(defn x-mean
  "E[x | z, y = 2]: x | z, y is N((1 + 3z + 4·2)/5, √(1/5))."
  [z]
  (/ (+ 1 (* 3 z) 8.0) 5))

(def mixture-exact
  "P(z = 1 | y = 2) and E[x | y = 2], exactly: given z, y is N(1 + 3z,
   √1.25)."
  (let [joint (fn [z prior]
                (let [y-given-z (d/normal (+ 1 (* 3 z)) (Math/sqrt 1.25))]
                  (* prior (array/exp (dc/log-prob y-given-z 2.0)))))
        p1 (/ (joint 1 0.6) (+ (joint 0 0.4) (joint 1 0.6)))]
    {:p1 p1 :x (+ (* (- 1 p1) (x-mean 0)) (* p1 (x-mean 1)))}))

mixture-exact
;=> {:p1 0.3112 :x 1.9867}

;; book: mixture-check
(let [{z-mean :mean z-se :mcse-mean} (diag/summary
                                      (get-in mixture-run [:draws :z]))
      {x-mean :mean x-se :mcse-mean} (diag/summary
                                      (get-in mixture-run [:draws :x]))]
  [(< (abs (- z-mean (:p1 mixture-exact))) (* 5 z-se))
   (< (abs (- x-mean (:x mixture-exact))) (* 5 x-se))])
;=> [true true]

;; book: mixture-figure
;; figure: figures/023-mixture.svg
(defn mixture-density
  "The exact posterior density of x given y = 2."
  [x]
  (let [{p1 :p1} mixture-exact
        component (fn [z] (array/exp (dc/log-prob
                                      (d/normal (x-mean z) (Math/sqrt 0.2))
                                      x)))]
    (+ (* (- 1 p1) (component 0)) (* p1 (component 1)))))

(svg/histogram [{:values (get-in mixture-run [:draws :x])
                 :density mixture-density}]
               {:x-label "x"})

;; book: gp-model
(def gp-n 6)

(def gp-cov
  "A squared-exponential covariance of length scale 1.5 over the points
   0, 1, …, 5, with a little jitter on its diagonal."
  (array/array (vec (for [i (range gp-n)]
                      (vec (for [j (range gp-n)]
                             (+ (Math/exp (/ (* (- i j) (- i j)) -4.5))
                                (if (= i j) 1e-6 0.0))))))))

(def gp
  (gen []
    (let [f (trace :f (d/mvnormal (array/zeros [gp-n]) gp-cov))]
      (dotimes [i gp-n]
        (trace [:y i] (d/normal (array/take f i) 0.5)))
      f)))

(def gp-ys [0.8 1.4 0.9 -0.3 -1.2 -0.6])

(def gp-start
  (:trace (gfi/generate gp [] (observations gp-ys) (rng/key 17))))

;; book: elliptical
(defn slice
  "The kernel of elliptical slice moves of f, under the prior N(0, cov)."
  [cov]
  (fn [t key]
    {:trace (mcmc/elliptical-slice t :f (array/zeros [gp-n]) cov key)}))

(def gp-run
  (run-chain (slice gp-cov) gp-start [:f]
             {:samples 2000 :burn 100 :key (rng/key 18)}))

(def fs (get-in gp-run [:draws :f]))

(every? true? (map not= fs (rest fs)))
;=> true

;; book: elliptical-check
(def gp-exact-mean
  "Σ (Σ + 0.5² I)^-1 y, the exact posterior mean of f."
  (let [noisy (+ gp-cov (* 0.25 (array/eye gp-n)))
        gain (array/matmul gp-cov (linalg/inv noisy))]
    (array/->clj (array/matmul gain (array/array gp-ys)))))

(defn within-five?
  "Do the draws' means of each element agree with exact-mean within five
   Monte Carlo standard errors?"
  [fs exact-mean]
  (every? true? (map (fn [i m]
                       (let [{:keys [mean mcse-mean]}
                             (diag/summary (mapv #(nth % i) fs))]
                         (< (abs (- mean m)) (* 5 mcse-mean))))
                     (range) exact-mean)))

(within-five? fs gp-exact-mean)
;=> true

;; book: solution-1
(def selection-order
  "The exact variances in the selection's order, slope first: the wrong
   order for the metric."
  (array/diag (:cov exact)))

(def misordered
  (run-particles #(mcmc/mala-particles %1 line 0.01
                                       {:inv-metric selection-order :key %2})
                 at-center [:slope :intercept]
                 {:samples 500 :burn 100 :key (rng/key 19)}))

(def conditional-sd
  "The slope's standard deviation given the intercept."
  (let [[[a b] [_ c]] (array/->clj (:cov exact))]
    (Math/sqrt (- a (/ (* b b) c)))))

(defn sds
  "The standard deviation of each chain's draws at addr."
  [run addr]
  (mapv #(:sd (diag/summary %)) (get-in run [:draws addr])))

[conditional-sd (sds misordered :slope) (sds misordered :intercept)]
;=> [0.0509 [0.051 0.054 0.056 0.057] [0.087 0.128 0.075 0.087]] ; float64

[(every? #(< (abs (- % conditional-sd)) (* 0.4 conditional-sd))
         (sds misordered :slope))
 (every? #(< % (/ (exact-sd 1) 3)) (sds misordered :intercept))
 (> (diag/r-hat (get-in misordered [:draws :intercept])) 1.01)]
;=> [true true true]

;; book: solution-1-ordered
(def ordered
  (run-particles #(mcmc/mala-particles %1 line 0.01
                                       {:inv-metric inv-metric :key %2})
                 at-center [:slope :intercept]
                 {:samples 500 :burn 100 :key (rng/key 19)}))

(not-any? #(< % (/ (exact-sd 1) 3)) (sds ordered :intercept))
;=> true

;; book: solution-2
(defn hmc-ess
  "The pooled bulk ESS at :slope of four HMC chains of 500 draws from the
   trace center, with the exact variances as the metric, step size 0.3
   and L steps."
  [L]
  (let [run (run-particles
             #(mcmc/hmc-particles %1 line {:L L :eps 0.3
                                           :inv-metric inv-metric} %2)
             at-center [:slope] {:samples 500 :burn 100 :key (rng/key 20)})]
    (diag/ess-bulk (get-in run [:draws :slope]))))

(def by-length (into (sorted-map) (map (juxt identity hmc-ess)) [1 3 10]))

by-length
;=> {1 47.8 3 274.0 10 6602.1} ; float64

(let [{e1 1 e3 3 e10 10} by-length]
  [(< (* 2 e1) e3) (< e3 2000 e10)])
;=> [true true]

;; book: solution-2-turns
(mapv #(/ (* 10 0.3) % 2 Math/PI) scaled-sds)
;=> [0.348 1.412]

;; book: solution-3
(def long-walks
  (mapv (fn [t k]
          (run-chain walk t [:slope] {:samples 10500 :burn 200 :key k}))
        starts (rng/split (rng/key 21) 4)))

(def long-walk-slopes (chains long-walks :slope))

[(diag/r-hat long-walk-slopes) (ess-per long-walk-slopes (* 4 10500))
 (ess-per (get-in hmc-chains [:draws :slope]) (* 4 500 21))]
;=> [1.0047 0.0289 0.0353] ; float64

(let [ratio (/ (ess-per long-walk-slopes (* 4 10500))
               (ess-per (get-in hmc-chains [:draws :slope]) (* 4 500 21)))]
  [(< (diag/r-hat long-walk-slopes) 1.01) (< 0.5 ratio 2)])
;=> [true true]

;; book: solution-4
(def wide-run
  (run-chain (slice (* 4 gp-cov)) gp-start [:f]
             {:samples 2000 :burn 100 :key (rng/key 22)}))

(within-five? (get-in wide-run [:draws :f]) gp-exact-mean)
;=> true
