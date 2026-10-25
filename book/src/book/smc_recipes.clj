;; book: ns
(ns book.smc-recipes
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as kernel]
            [gen.inference.particle-filter :as pf]
            [gen.inference.trace-kernel :as tk]
            [gen.inference.transform :as tf]
            [gen.inference.weights :as weights]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]))

;; book: model
(def motion
  (gen [t {:keys [y moving]}]
    (let [moving (trace :moving (d/bernoulli (array/where moving 0.75 0.25)))
          velocity (array/where moving (Math/sin (inc t)) 0.0)
          y (trace :y (d/normal (+ y velocity) 0.01))]
      (trace :y-obs (d/normal y 0.25))
      {:y y :moving moving})))

(def object-motion (comb/unfold motion))

(def still
  "The state before the first step: at 0, and not moving."
  {:y 0.0 :moving false})

;; book: addresses
(cm/addresses (:choices (gfi/simulate object-motion [2 still] (rng/key 1))))
;=> [[0 :moving] [0 :y] [0 :y-obs] [1 :moving] [1 :y] [1 :y-obs]]

;; book: data
(def T 10)

(def truth
  "A run of ten steps, still for five and moving for five."
  (:trace (gfi/generate object-motion [T still]
                        (apply cm/choicemap
                               (mapcat (fn [t] [[t :moving] (>= t 5)])
                                       (range T)))
                        (rng/key 22))))

(def ys
  "The run's observations, host numbers."
  (mapv #(array/item (truth [% :y-obs])) (range T)))

(defn rounded
  "x rounded to two decimals."
  [x]
  (/ (Math/round (* 100.0 x)) 100.0))

(mapv rounded ys)
;=> [0.11 -0.03 -0.07 -0.02 -0.16 -0.31 0.43 1.21 2.15 1.02] ; float64

;; book: data-figure
;; figure: figures/053-motion.svg
(let [positions (mapv #(array/item (truth [% :y])) (range T))]
  (svg/paths [{:paths [(map-indexed vector positions)]}
              {:paths [(map-indexed vector ys)] :r 3 :width 0 :color "black"}]
             {:x-label "step t" :y-label "y"}))

;; book: exact
(defn velocity
  "The velocity at step t, moving or not."
  [moving t]
  (if moving (Math/sin (inc t)) 0.0))

(defn kalman-args
  "gen.state-space/kalman's arguments for the positions given the moving
   choices ms: a random walk of spread 0.01 whose step t is offset by its
   velocity, seen with noise 0.25."
  [ms]
  [(count ms) {:mean (velocity (ms 0) 0) :cov 1e-4}
   (fn [t] {:transition 1.0 :transition-cov 1e-4
            :emission 1.0 :emission-cov 0.0625
            :transition-offset (velocity (ms t) t)})])

(defn log-prior
  "log p(ms): each moving choice keeps the one before with probability
   0.75 (the first, not moving)."
  [ms]
  (reduce + (map (fn [prev m] (Math/log (if (= prev m) 0.75 0.25)))
                 (cons false ms) ms)))

(def observed
  "The observations as kalman's choices, at 0 … 9."
  (apply cm/choicemap (interleave (range) ys)))

(defn log-likelihood
  "log p(ys | ms), by the Kalman filter."
  [ms]
  (array/item (:weight (gfi/assess ss/kalman (kalman-args ms) observed))))

(def sequences
  "Every sequence of ten moving choices."
  (vec (for [i (range 1024)] (mapv #(bit-test i %) (range T)))))

(def log-joints
  "log p(ms, ys) of each sequence."
  (mapv #(+ (log-prior %) (log-likelihood %)) sequences))

(def exact-log-ml (weights/log-sum-exp log-joints))

(defn probabilities
  "The normalized weights of the log weights lw."
  [lw]
  (mapv #(Math/exp %) (:log-normalized-weights (weights/normalize lw))))

(def exact-moving
  "P(moving at t | ys) for each t: the posterior weight of the sequences
   that move at t."
  (let [post (probabilities log-joints)]
    (mapv (fn [t] (reduce + (map #(if (%1 t) %2 0.0) sequences post)))
          (range T))))

exact-log-ml
;=> -1.2556 ; float64

(mapv rounded exact-moving)
;=> [0.0 0.0 0.02 0.0 0.0 0.94 0.99 1.0 0.99 0.99] ; float64

;; book: exact-check
(defn joint-log-likelihood
  "log p(ys | ms) from the joint normal of the observations: the position
   at t is the sum of the velocities and of t + 1 steps of spread 0.01, so
   Cov(y_s, y_t) = 0.01^2 (min(s, t) + 1), plus 0.25^2 on the diagonal."
  [ms]
  (let [n (count ms)
        mean (vec (reductions + (map-indexed #(velocity %2 %1) ms)))
        cov (vec (for [s (range n)]
                   (vec (for [t (range n)]
                          (+ (* 1e-4 (inc (min s t)))
                             (if (= s t) 0.0625 0.0))))))]
    (dc/log-prob (d/mvnormal (array/array mean) (array/array cov))
                 (array/array ys))))

(let [ms (mapv #(>= % 5) (range T))]
  (array/allclose? (joint-log-likelihood ms) (log-likelihood ms)))
;=> true

;; book: filter
(def argdiffs
  "A step's argdiffs: the number of steps changes, the start does not."
  [diff/unknown-change diff/no-change])

(defn observation
  "The choice map of observation t of ys."
  [ys t]
  (cm/choicemap [t :y-obs] (ys t)))

(defn latest
  "The selection of step t's choices."
  [t]
  (sel/select [t :moving] [t :y]))

(defn mh-move
  "Before step t, an MH move on every particle's latest choices, those of
   step t - 1, proposed from their prior."
  [state t key]
  (pf/rejuvenate state (kernel/mh (latest (dec t))) key))

(defn particle-filter
  "GenParticleFilters' example filter: n particles of object-motion over
   the observations ys. Before each step, when the effective sample size
   is below half of n, residual resampling, then (move state t key),
   mh-move by default. The final state."
  ([ys n key] (particle-filter ys n mh-move key))
  ([ys n move key]
   (let [[k-init & ks] (rng/split key (count ys))]
     (reduce (fn [state [t k]]
               (let [[k-resample k-move k-step] (rng/split k 3)]
                 (-> (if (< (pf/ess state) (* 0.5 n))
                       (-> state
                           (pf/resample {:method :residual :key k-resample})
                           (move t k-move))
                       state)
                     (pf/step [(inc t) still] argdiffs (observation ys t)
                              {:key k-step}))))
             (pf/initialize object-motion [1 still] (observation ys 0) n
                            {:key k-init})
             (map vector (range 1 (count ys)) ks)))))

(def filtered (particle-filter ys 100 (rng/key 30)))

;; book: statistics
[(pf/mean filtered #(% [4 :moving])) (pf/mean filtered #(% [5 :moving]))]
;=> [0.0 0.9703] ; float64

(pf/proportions filtered #(% [5 :moving]))
;=> {false 0.0297 true 0.9703} ; float64

(let [p (pf/mean filtered #(% [5 :moving]))]
  (array/allclose? (pf/variance filtered #(% [5 :moving])) (* p (- 1 p))))
;=> true

;; book: runs
(defn spread
  "The mean of the numbers xs, their standard deviation, and the standard
   error of the mean: {:mean :sd :se}."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) xs))
                         (dec n)))]
    {:mean mean :sd sd :se (/ sd (Math/sqrt n))}))

(defn summary
  "What a final state says, as host numbers: its estimate of P(moving at t
   | ys) for each t, and of log p(ys)."
  [state]
  {:moving (mapv (fn [t] (array/item (pf/mean state #(% [t :moving]))))
                 (range T))
   :log-ml (pf/log-ml-estimate state)})

(def runs
  "The summaries of 40 filters of 100 particles."
  (mapv #(summary (particle-filter ys 100 %)) (rng/split (rng/key 31) 40)))

;; book: runs-check
(defn agrees?
  "Is the mean of xs within five standard errors of x?"
  [xs x]
  (let [{:keys [mean se]} (spread xs)]
    (< (abs (- mean x)) (* 5 se))))

(defn evidence-ratios
  "exp(estimate - log p(ys)) of each run: their mean is 1 for an unbiased
   estimate of p(ys)."
  [runs log-ml]
  (mapv #(Math/exp (- (:log-ml %) log-ml)) runs))

[(agrees? (map #(get-in % [:moving 5]) runs) (exact-moving 5))
 (every? #(< (get-in % [:moving 4]) 0.01) runs)
 (agrees? (evidence-ratios runs exact-log-ml) 1.0)]
;=> [true true true]

;; book: moving-figure
;; figure: figures/053-moving.svg
(let [filter-means (apply mapv (fn [& ps] (:mean (spread ps)))
                          (map :moving runs))]
  (svg/plot [{:points (map-indexed vector exact-moving) :r 6
              :color "darkorange" :opacity 0.5}
             {:points (map-indexed vector filter-means)}]
            {:x-range [-0.5 9.5] :y-range [-0.05 1.05]
             :x-label "step t" :y-label "P(moving at t | ys)"}))

;; book: residual
(def first-population
  (pf/initialize object-motion [1 still] (observation ys 0) 100
                 {:key (rng/key 32)}))

(let [lw (pf/log-weights first-population)
      expected (mapv #(* 100 %) (probabilities lw))
      copies (frequencies (weights/residual lw 100 (rng/key 33)))]
  [(reduce + (map #(Math/floor %) expected))
   (every? #(>= (get copies % 0) (Math/floor (expected %))) (range 100))])
;=> [82.0 true] ; float64

;; book: strata
(def first-moves (pf/choice-product [[[0 :moving] [false true]]]))

(mapv cm/to-map first-moves)
;=> [{0 {:moving false}} {0 {:moving true}}]

(def stratified
  (pf/initialize object-motion [1 still] (observation ys 0) 20
                 {:key (rng/key 40) :strata first-moves}))

(let [moving (mapv #(% [0 :moving]) (pf/traces stratified))]
  [(frequencies (subvec moving 0 10)) (frequencies (subvec moving 10))])
;=> [{false 10} {true 10}]

;; book: strata-exact
(defn first-density
  "p(moving at 0, y-obs at 0 = y): the moving choice's prior times the
   observation's density, the position summed out."
  [moving y]
  (* (if moving 0.25 0.75)
     (Math/exp (array/item (dc/log-prob (d/normal (velocity moving 0)
                                                  (Math/sqrt (+ 1e-4 0.0625)))
                                        y)))))

(def first-evidence
  "p(y-obs at 0 = ys[0])."
  (+ (first-density false (ys 0)) (first-density true (ys 0))))

(defn first-ratios
  "exp(estimate - log p(ys[0])) of 200 initializations of n particles,
   with opts."
  [n opts key]
  (mapv (fn [k]
          (let [s (pf/initialize object-motion [1 still] (observation ys 0) n
                                 (assoc opts :key k))]
            (Math/exp (- (pf/log-ml-estimate s) (Math/log first-evidence)))))
        (rng/split key 200)))

(def plain-ratios (first-ratios 20 {} (rng/key 41)))

(def stratified-ratios (first-ratios 20 {:strata first-moves} (rng/key 41)))

[(:sd (spread plain-ratios)) (:sd (spread stratified-ratios))]
;=> [0.134 0.0056] ; float64

;; book: strata-check
(defn log-variance
  "The log of the sample variance of xs, and its standard error from their
   fourth moment, sqrt((m4/v^2 - 1)/n): {:log-variance :se}."
  [xs]
  (let [n (count xs)
        {:keys [mean sd]} (spread xs)
        v (* sd sd)
        m4 (/ (reduce + (map #(Math/pow (- % mean) 4) xs)) n)]
    {:log-variance (Math/log v) :se (Math/sqrt (/ (- (/ m4 (* v v)) 1) n))}))

(defn smaller-variance?
  "Is the variance of xs below that of ys, by more than five standard
   errors of the difference of their logs?"
  [xs ys]
  (let [a (log-variance xs)
        b (log-variance ys)]
    (> (- (:log-variance b) (:log-variance a))
       (* 5 (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b))))))))

[(agrees? plain-ratios 1.0) (agrees? stratified-ratios 1.0)
 (smaller-variance? stratified-ratios plain-ratios)]
;=> [true true true]

;; book: enumerated-step
(defn enumerated-step
  "state stepped to args with the observations, each particle extended once
   with each of the values at addr: k copies of every particle (k the number
   of values), block j of the copies stepped with addr constrained to
   (values j), and every weight multiplied by k. A stratified step whose
   strata cover the values."
  [state args observations addr values key]
  (let [k (count values)
        n (count (pf/log-weights state))
        block (fn [j] (range (* j n) (* (inc j) n)))
        extend (fn [s [j kj]]
                 (pf/update-view s (block j) pf/step args argdiffs
                                 (cm/merge observations
                                           (cm/choicemap addr (values j)))
                                 {:key kj}))]
    (-> (reduce extend (pf/replicate state k {:layout :interleaved})
                (map vector (range k) (rng/split key k)))
        (update :log-weights (fn [lw] (mapv #(+ % (Math/log k)) lw))))))

;; book: enumerated-example
(def three
  (pf/initialize object-motion [1 still] (observation ys 0) 3
                 {:key (rng/key 42)}))

(def six
  (enumerated-step three [2 still] (observation ys 1) [1 :moving]
                   [false true] (rng/key 43)))

(mapv #(% [1 :moving]) (pf/traces six))
;=> [false false false true true true]

(= (mapv #(array/item (% [0 :y])) (pf/traces six))
   (into [] (comp cat (map #(array/item (% [0 :y]))))
         [(pf/traces three) (pf/traces three)]))
;=> true

;; book: fearnhead-clifford
(defn fearnhead-clifford
  "Fearnhead and Clifford's filter: n particles of object-motion over ys,
   each step enumerating the moving choice of every particle, then resized
   optimally back to n, each particle kept at most once."
  [ys n key]
  (let [[k-init & ks] (rng/split key (count ys))
        [k-start k-resize] (rng/split k-init)
        start (pf/initialize object-motion [1 still] (observation ys 0) (* 2 n)
                             {:key k-start :strata first-moves})]
    (reduce (fn [state [t k]]
              (let [[k-step k-resize] (rng/split k)]
                (-> state
                    (enumerated-step [(inc t) still] (observation ys t)
                                     [t :moving] [false true] k-step)
                    (pf/resize n {:method :optimal :key k-resize}))))
            (pf/resize start n {:method :optimal :key k-resize})
            (map vector (range 1 (count ys)) ks))))

;; book: optimal
(let [doubled (enumerated-step filtered [(inc T) still]
                               (cm/choicemap [T :y-obs] 1.0)
                               [T :moving] [false true] (rng/key 44))
      halved (pf/resize doubled 100 {:method :optimal :key (rng/key 45)})]
  [(count (pf/log-weights doubled)) (count (distinct (:parents halved)))
   (array/allclose? (pf/log-ml-estimate halved) (pf/log-ml-estimate doubled))])
;=> [200 100 true]

;; book: fc-runs
(def fc-runs
  "The summaries of 40 Fearnhead-Clifford filters of 50 particles."
  (mapv #(summary (fearnhead-clifford ys 50 %)) (rng/split (rng/key 46) 40)))

[(agrees? (map #(get-in % [:moving 5]) fc-runs) (exact-moving 5))
 (agrees? (evidence-ratios fc-runs exact-log-ml) 1.0)]
;=> [true true]

[(:sd (spread (map :log-ml runs))) (:sd (spread (map :log-ml fc-runs)))]
;=> [0.278 0.0504] ; float64

(smaller-variance? (map :log-ml fc-runs) (map :log-ml runs))
;=> true

;; book: resizing
(def grown (pf/resize filtered 400 {:method :residual :key (rng/key 50)}))

[(count (pf/log-weights grown))
 (array/allclose? (pf/log-ml-estimate grown) (pf/log-ml-estimate filtered))]
;=> [400 true]

(def merged (pf/coalesce grown))

[(count (pf/log-weights merged)) (count (distinct (:parents grown)))]
;=> [78 78] ; float64

(defn last-position
  "The position at the last step of a trace of object-motion."
  [tr]
  (tr [(dec (first (:args tr))) :y]))

[(array/allclose? (pf/log-ml-estimate merged) (pf/log-ml-estimate grown))
 (array/allclose? (pf/mean merged last-position) (pf/mean grown last-position))]
;=> [true true]

;; book: replicate
(def copies (pf/replicate filtered 4))

[(array/allclose? (pf/ess copies) (* 4 (pf/ess filtered)))
 (array/allclose? (pf/log-ml-estimate copies) (pf/log-ml-estimate filtered))
 (= (pf/log-weights (pf/dereplicate copies 4)) (pf/log-weights filtered))]
;=> [true true true]

;; book: walk-model
(def walk-step
  (gen [t prev]
    (let [z (trace :z (d/normal prev 1.0))]
      (trace :y (d/normal z 1.0))
      z)))

(def walk (comb/unfold walk-step))

(def origin (array/zeros [2]))

;; book: challenging
(def challenging
  "GenSMCP3's challenging observations, rounded to three decimals."
  [[0.580 0.472] [2.358 1.878] [3.824 3.049] [4.926 5.769] [6.538 7.238]
   [8.615 6.919] [9.684 6.682] [12.973 8.540] [16.238 11.297]
   [18.829 12.899] [22.580 13.978] [21.435 11.692] [21.222 9.403]
   [21.554 6.456] [21.690 5.108] [22.092 4.999] [20.374 3.773]
   [21.841 3.955] [23.279 1.084] [24.585 -1.753] [24.760 -4.794]
   [22.998 -7.250] [21.292 -7.965] [22.108 -9.674] [22.373 -10.381]
   [23.854 -12.382] [21.571 -10.674] [18.111 -11.132] [15.693 -10.511]
   [13.974 -10.895] [14.630 -8.723] [11.489 -9.149] [9.681 -8.231]
   [8.241 -6.778] [9.342 -7.972] [8.469 -10.693] [7.246 -11.552]
   [4.665 -11.173] [4.364 -11.210] [2.547 -10.210] [1.214 -10.875]
   [1.952 -10.996] [3.049 -9.439] [5.894 -10.486] [4.560 -8.913]
   [4.077 -6.389] [4.189 -5.227] [5.224 -3.451] [4.321 0.607]
   [2.950 5.206] [2.453 5.883]])

(defn walk-observation
  "The choice map of observation t of the challenging ones."
  [t]
  (cm/choicemap [t :y] (array/array (challenging t))))

(defn walk-observations
  "The choice map of the first n challenging observations."
  [n]
  (reduce cm/merge (cm/choicemap) (map walk-observation (range n))))

;; book: walk-exact
(def walk-kalman
  "The Kalman filter on the challenging observations: its weight is
   log p(y), its return value the filtered beliefs."
  (let [I (array/eye 2)]
    (gfi/assess ss/kalman
                [(count challenging) {:mean origin :cov I}
                 {:transition I :transition-cov I :emission I :emission-cov I}]
                (apply cm/choicemap
                       (interleave (range) (map array/array challenging))))))

(def walk-log-ml (array/item (:weight walk-kalman)))

(def walk-means
  "The filtered means, E[z_t | y_0 … y_t], host vectors."
  (mapv #(array/->clj (:mean %)) (:retval walk-kalman)))

walk-log-ml
;=> -251.7

;; book: walk-filter
(def walk-argdiffs [diff/unknown-change diff/no-change])

(defn regroup
  "GenSMCP3's resampling: the particles in 5 groups, each resampled by the
   residual method when its effective sample size is below a fifth of its
   size."
  [state key]
  (let [size (quot (count (pf/log-weights state)) 5)]
    (reduce (fn [s [g k]]
              (pf/update-view s (range (* g size) (* (inc g) size))
                              pf/maybe-resample
                              {:ess-threshold (* 0.2 size) :method :residual
                               :key k}))
            state
            (map vector (range 5) (rng/split key 5)))))

(defn walk-filter
  "n particles of walk through the challenging observations, each step
   (advance state t key), then (resample state key): the final state, with
   :means, the filtered mean of z_t after each step t."
  [advance resample n key]
  (let [[k-init & ks] (rng/split key (inc (count challenging)))]
    (reduce (fn [state [t k]]
              (let [[k-advance k-resample] (rng/split k)
                    moved (advance state t k-advance)
                    mean (array/->clj (pf/mean moved #(% [t :z])))]
                (assoc (resample moved k-resample)
                       :means (conj (:means state) mean))))
            (assoc (pf/initialize walk [0 origin] cm/EMPTY n {:key k-init})
                   :means [])
            (map vector (range (count challenging)) ks))))

(defn bootstrap
  "The bootstrap step to t: z_t from its prior."
  [state t key]
  (pf/step state [(inc t) origin] walk-argdiffs (walk-observation t)
           {:key key}))

;; book: ula
(defn previous
  "The state before step t of trace tr: z_t-1, the origin at t = 0."
  [tr t]
  (if (zero? t) origin (tr [(dec t) :z])))

(def step-size
  "The Langevin step's size (GenSMCP3's ULA_STEP_SIZE)."
  0.8)

(defn ula-mean
  "A Langevin step from v up the gradient of log p(z_t = v | prev) +
   log p(y | z_t = v), which is (prev - v) + (y - v)."
  [prev v y]
  (+ v (* step-size (+ (- prev v) (- y v)))))

(def forward
  (gen [tr t y]
    (let [prev (previous tr t)
          v (trace :v (d/normal prev 1.0))]
      (trace :z (d/normal (ula-mean prev v y) (Math/sqrt (* 2 step-size)))))))

(def backward
  (gen [tr t y]
    (trace :v (d/normal (previous tr t) 1.0))))

(defn handover
  "The transform of step t: the forward kernel's z becomes the trace's z_t,
   and its v the backward kernel's; paired with its inverse, which gives
   them back."
  [t]
  (first (tf/pair (tf/transform [model-in aux-in model-out aux-out]
                    (tf/copy aux-in :z model-out [t :z])
                    (tf/copy aux-in :v aux-out :v))
                  (tf/transform [model-in aux-in model-out aux-out]
                    (tf/copy model-in [t :z] aux-out :z)
                    (tf/copy aux-in :v aux-out :v)))))

(defn smcp3-step
  "The SMCP3 step to t, an updating translator (opts: :check?)."
  ([t] (smcp3-step t {}))
  ([t opts]
   (let [y (array/array (challenging t))]
     (tf/updating-translator
      (merge {:p-new-args [(inc t) origin] :p-argdiffs walk-argdiffs
              :new-observations (walk-observation t)
              :q-forward forward :q-forward-args [t y]
              :q-backward backward :q-backward-args [t y]
              :transform (handover t)}
             opts)))))

(defn smcp3
  "Every particle moved by the SMCP3 step to t."
  [state t key]
  (pf/move-reweight state (smcp3-step t) key))

;; book: smcp3-checked
(def five
  "A trace of walk through the first five observations."
  (:trace (gfi/generate walk [5 origin] (walk-observations 5) (rng/key 60))))

(:weight ((smcp3-step 5 {:check? true}) five (rng/key 61)))
;=> -33.0 ; float64

;; book: walk-runs
(defn rms-distance
  "The root mean square distance between the points as and bs, in pairs."
  [as bs]
  (let [squared (fn [a b] (reduce + (map #(let [x (- %1 %2)] (* x x)) a b)))]
    (Math/sqrt (/ (reduce + (map squared as bs)) (count as)))))

(defn walk-summary
  "A final state's estimate of log p(y), the root mean square distance of
   its filtered means from the Kalman filter's, and the means."
  [state]
  {:log-ml (pf/log-ml-estimate state)
   :error (rms-distance (:means state) walk-means)
   :means (:means state)})

(defn walk-runs
  "The summaries of 10 filters of 30 particles, by advance and resample."
  [advance resample key]
  (mapv #(walk-summary (walk-filter advance resample 30 %)) (rng/split key 10)))

(def bootstrap-runs (walk-runs bootstrap regroup (rng/key 62)))

(def smcp3-runs (walk-runs smcp3 regroup (rng/key 62)))

(defn mean-sd
  "The mean and the standard deviation of the numbers xs."
  [xs]
  ((juxt :mean :sd) (spread xs)))

[(mean-sd (map :log-ml bootstrap-runs)) (mean-sd (map :log-ml smcp3-runs))]
;=> [[-1414.5 267.7] [-292.3 5.12]] ; float64

[(mean-sd (map :error bootstrap-runs)) (mean-sd (map :error smcp3-runs))]
;=> [[6.07 0.765] [0.963 0.0675]] ; float64

;; book: walk-check
(defn exceeds?
  "Does the mean of xs exceed that of ys by more than five standard errors
   of their difference?"
  [xs ys]
  (let [a (spread xs)
        b (spread ys)]
    (> (- (:mean a) (:mean b))
       (* 5 (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b))))))))

[(exceeds? (map :log-ml smcp3-runs) (map :log-ml bootstrap-runs))
 (exceeds? (map :error bootstrap-runs) (map :error smcp3-runs))
 (let [{:keys [mean se]} (spread (map :log-ml smcp3-runs))]
   (< (+ mean (* 5 se)) walk-log-ml))]
;=> [true true true]

;; book: walk-figure
;; figure: figures/053-tracks.svg
(svg/paths [{:paths [challenging] :r 2.5 :width 0 :color "gray"}
            {:paths [walk-means] :color "black" :width 2}
            {:paths [(:means (first bootstrap-runs))] :color "darkorange"}
            {:paths [(:means (first smcp3-runs))] :color "steelblue"}]
           {:x-label "x" :y-label "y"})

;; book: kernels
(def ula-forward
  (gen [tr t y]
    (let [prev (previous tr t)
          v (trace :v (d/normal prev 1.0))
          z (trace :z (d/normal (ula-mean prev v y)
                                (Math/sqrt (* 2 step-size))))]
      [(cm/choicemap [t :z] z) (cm/choicemap :v v)])))

(def ula-backward
  (gen [tr t y]
    (let [v (trace :v (d/normal (previous tr t) 1.0))]
      [cm/EMPTY (cm/choicemap :v v :z (tr [t :z]))])))

(defn kernel-step
  "The SMCP3 step to t from the trace kernels (opts: :check?)."
  ([t] (kernel-step t {}))
  ([t opts]
   (let [y (array/array (challenging t))]
     (tk/smcp3 (merge {:p-new-args [(inc t) origin] :p-argdiffs walk-argdiffs
                       :new-observations (walk-observation t)
                       :k ula-forward :k-args [t y]
                       :l ula-backward :l-args [t y]}
                      opts)))))

;; book: kernels-agree
(def population
  "Thirty particles through the first ten observations."
  (pf/initialize walk [10 origin] (walk-observations 10) 30
                 {:key (rng/key 63)}))

(let [by-translator (pf/move-reweight population (smcp3-step 10) (rng/key 64))
      by-kernels (pf/move-reweight population (kernel-step 10) (rng/key 64))]
  [(array/allclose? (array/array (pf/log-weights by-translator))
                    (array/array (pf/log-weights by-kernels)))
   (every? true? (map #(array/allclose? (%1 [10 :z]) (%2 [10 :z]))
                      (pf/traces by-translator) (pf/traces by-kernels)))])
;=> [true true]

;; book: kernels-error
;; error: smcp3: the backward kernel does not give back the forward kernel's
#_(def forgetful
  (gen [tr t y]
    (let [v (trace :v (d/normal (previous tr t) 1.0))]
      [cm/EMPTY (cm/choicemap :v v)])))

#_((kernel-step 5 {:l forgetful :check? true}) five (rng/key 65))

;; book: point-model
(def point-model
  (gen [n]
    (let [x (trace :x (d/normal 0 100))]
      (doseq [i (range n)]
        (trace [:obs i] (d/normal x 1)))
      x)))

(defn obs-sum
  "The sum of the first n observations of trace tr."
  [tr n]
  (reduce + 0.0 (map #(tr [:obs %]) (range n))))

;; book: point-kernels
(def point-forward
  (gen [tr y]
    (let [n (inc (first (:args tr)))
          mean (/ (+ (obs-sum tr (dec n)) y) n)
          new-x (trace :new-x (d/normal mean (Math/sqrt (/ 1.0 (inc n)))))]
      [(cm/choicemap :x new-x) (cm/choicemap :previous-x (tr :x))])))

(defn backward-kernel
  "The backward kernel: x before the step, from the mean of the
   observations before it with variance 1/n (n the number after it); at
   the first step, from N(0, first-sd)."
  [first-sd]
  (gen [tr y]
    (let [n (first (:args tr))
          [mean sd] (if (> n 1)
                      [(/ (obs-sum tr (dec n)) (dec n)) (Math/sqrt (/ 1.0 n))]
                      [0.0 first-sd])
          previous-x (trace :previous-x (d/normal mean sd))]
      [(cm/choicemap :x previous-x) (cm/choicemap :new-x (tr :x))])))

(def point-backward
  "The backward kernel, from the prior at the first step."
  (backward-kernel 100.0))

;; book: point-host-error
;; error: Cannot read a value being differentiated on the host
#_(def host-forward
  (gen [tr y]
    (let [n (inc (first (:args tr)))
          total (+ (array/item (obs-sum tr (dec n))) y)
          new-x (trace :new-x (d/normal (/ total n) 1.0))]
      [(cm/choicemap :x new-x) (cm/choicemap :previous-x (tr :x))])))

#_((tk/smcp3 {:p-new-args [2] :new-observations (cm/choicemap [:obs 1] 2.0)
            :k host-forward :k-args [2.0] :l point-backward :l-args [2.0]})
 (:trace (gfi/generate point-model [1] (cm/choicemap [:obs 0] 1.0)
                       (rng/key 73)))
 (rng/key 74))

;; book: point-filter
(defn point-step
  "The SMCP3 step that adds observation t of ys (with backward, a
   backward kernel other than point-backward)."
  ([ys t] (point-step ys t point-backward))
  ([ys t backward]
   (tk/smcp3 {:p-new-args [(inc t)]
              :new-observations (cm/choicemap [:obs t] (ys t))
              :k point-forward :k-args [(ys t)]
              :l backward :l-args [(ys t)]})))

(defn point-filter
  "GenSMCP3's run_smcp3: n particles of point-model from its prior, moved
   through the observations ys by SMCP3 steps, with residual resampling
   when the effective sample size falls below a fifth of n."
  [ys n key]
  (let [[k-init & ks] (rng/split key (inc (count ys)))]
    (reduce (fn [state [t k]]
              (let [[k-step k-resample] (rng/split k)]
                (-> state
                    (pf/move-reweight (point-step ys t) k-step)
                    (pf/maybe-resample {:ess-threshold (/ n 5.0)
                                        :method :residual :key k-resample}))))
            (pf/initialize point-model [0] cm/EMPTY n {:key k-init})
            (map vector (range (count ys)) ks))))

;; book: point-exact
(defn point-exact
  "The exact answers of point-model on ys: the observations are jointly
   normal, of covariance 100^2 plus 1 on the diagonal, and x's posterior
   mean is Σy/(n + 1/100^2)."
  [ys]
  (let [n (count ys)
        cov (+ (array/eye n) (* 1e4 (array/ones [n n])))]
    {:log-ml (array/item (dc/log-prob (d/mvnormal (array/zeros [n]) cov)
                                      (array/array ys)))
     :mean (/ (reduce + ys) (+ n 1e-4))}))

[(:mean (point-exact [1.0 2.0 3.0])) (:mean (point-exact [1.0 2.0 3.0 10.0]))]
;=> [1.9999 3.9999]

;; book: point-runs
(defn point-summaries
  "The posterior mean of x and the log-ml estimate of 10 filters of 200
   particles on ys."
  [ys key]
  (mapv (fn [k]
          (let [s (point-filter ys 200 k)]
            {:mean (array/item (pf/mean s #(% :x)))
             :log-ml (pf/log-ml-estimate s)}))
        (rng/split key 10)))

(def three-points (point-summaries [1.0 2.0 3.0] (rng/key 70)))

(def four-points (point-summaries [1.0 2.0 3.0 10.0] (rng/key 71)))

[(:mean (spread (map :mean three-points)))
 (:mean (spread (map :mean four-points)))]
;=> [1.9885 3.9893] ; float64

(let [check (fn [runs ys]
              (let [{:keys [mean log-ml]} (point-exact ys)]
                [(agrees? (map :mean runs) mean)
                 (agrees? (evidence-ratios runs log-ml) 1.0)]))]
  [(check three-points [1.0 2.0 3.0]) (check four-points [1.0 2.0 3.0 10.0])])
;=> [[true true] [true true]]

;; book: point-figure
;; figure: figures/053-point.svg
(let [s (point-filter [1.0 2.0 3.0 10.0] 200 (rng/key 72))
      posterior (d/normal 3.9999 (Math/sqrt (/ 1 4.0001)))]
  (svg/histogram [{:values (mapv #(% :x) (pf/traces s))
                   :weights (probabilities (pf/log-weights s))
                   :density #(Math/exp (array/item (dc/log-prob posterior %)))}]
                 {:range [2.5 5.5] :bins 24 :x-label "x"}))

;; book: solution-1
(defn reweight-move
  "Before step t, the latest choices of every particle regenerated from
   their prior, each weight multiplied by the move's (move-reweight)."
  [state t key]
  (pf/move-reweight state (pf/reweighting (latest (dec t))) key))

(let [resampled (pf/resample filtered {:method :residual :key (rng/key 80)})]
  [(pf/ess resampled) (pf/ess (reweight-move resampled T (rng/key 81)))])
;=> [100.0 4.39] ; float64

(def reweight-runs
  (mapv #(summary (particle-filter ys 100 reweight-move %))
        (rng/split (rng/key 82) 40)))

[(agrees? (map #(get-in % [:moving 5]) reweight-runs) (exact-moving 5))
 (agrees? (evidence-ratios reweight-runs exact-log-ml) 1.0)]
;=> [true true]

;; book: solution-2
(defn together
  "Residual resampling of all the particles at once, when the effective
   sample size is below half of their number."
  [state key]
  (pf/maybe-resample state {:method :residual :key key}))

(def together-runs (walk-runs bootstrap together (rng/key 83)))

(:mean (spread (map :log-ml together-runs)))
;=> -421.0 ; float64

[(exceeds? (map :log-ml together-runs) (map :log-ml bootstrap-runs))
 (exceeds? (map :log-ml smcp3-runs) (map :log-ml together-runs))]
;=> [true true]

;; book: solution-3
(let [start (pf/initialize point-model [0] cm/EMPTY 200 {:key (rng/key 84)})
      first-ess (fn [backward]
                  (pf/ess (pf/move-reweight start (point-step [1.0] 0 backward)
                                            (rng/key 85))))]
  [(first-ess (backward-kernel 1.0)) (first-ess point-backward)])
;=> [3.47 156.9] ; float64

;; book: solution-4
(def prior-step
  (gen [tr t] (trace :z (d/normal (previous tr t) 1.0))))

(defn bootstrap-translator
  "The bootstrap step to t as an updating translator: z_t from its prior,
   no backward choices."
  [t]
  (tf/updating-translator
   {:p-new-args [(inc t) origin] :p-argdiffs walk-argdiffs
    :new-observations (walk-observation t)
    :q-forward prior-step :q-forward-args [t]
    :transform (tf/transform [model-in aux-in model-out aux-out]
                 (tf/copy aux-in :z model-out [t :z]))}))

(defn observation-density
  "log p(y_t | z_t) of trace tr's step t: the sum of each coordinate's."
  [tr t]
  (array/sum (dc/log-prob (d/normal (tr [t :z]) 1.0)
                          (array/array (challenging t)))))

(let [moved (pf/move-reweight population (bootstrap-translator 10)
                              (rng/key 86))]
  (every? true? (map (fn [tr before after]
                       (array/allclose? (- after before)
                                        (observation-density tr 10)))
                     (pf/traces moved) (pf/log-weights population)
                     (pf/log-weights moved))))
;=> true

;; book: solution-5
(def seven
  "Importance sampling of the first seven observations: 100 whole paths
   from the prior, weighted by the observations."
  (pf/initialize object-motion [7 still]
                 (apply cm/choicemap (mapcat (fn [t] [[t :y-obs] (ys t)])
                                             (range 7)))
                 100 {:key (rng/key 88)}))

(pf/ess seven)
;=> 14.46 ; float64

(defn resampled
  "The number of distinct particles kept, and exp(estimate after - estimate
   before), of each of 400 resamplings of seven (opts as pf/resample's)."
  [opts key]
  (let [before (pf/log-ml-estimate seven)]
    (mapv (fn [k]
            (let [s (pf/resample seven (assoc opts :key k))]
              {:distinct (count (distinct (:parents s)))
               :ratio (Math/exp (- (pf/log-ml-estimate s) before))}))
          (rng/split key 400))))

(def plain-resamplings (resampled {} (rng/key 87)))

(def tempered-resamplings
  (resampled {:priority (fn [lw] (map #(* 0.5 %) lw))} (rng/key 87)))

[(:mean (spread (map :distinct plain-resamplings)))
 (:mean (spread (map :distinct tempered-resamplings)))]
;=> [24.85 29.51] ; float64

[(exceeds? (map :distinct tempered-resamplings)
           (map :distinct plain-resamplings))
 (agrees? (map :ratio tempered-resamplings) 1.0)]
;=> [true true]
