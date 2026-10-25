;; The code of book/054-gallery.md. Each region, from a `;; book:` line to
;; the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).
;;
;; Bands. Every statistical check has a band of five standard errors from
;; the draws themselves (MCSE for chains, sqrt(1/ESS - 1/n) for importance
;; sampling's log evidence), or an exact value, except the two learned
;; results: the guide's fit (spreads over keys 101-110, beside vi-check)
;; and the trained network's error (over keys 101-110, beside net-check).
;;
;; Planted bugs (2026-10-25), each failing its check: an intercept prior
;; of twice the variance (the line's evidence); the likelihood centered at
;; 4.0 (the height at ten); a walk step of twice the variance, or the last
;; state left out of HMC's selection (the smoothed means); the parabola's
;; priors N(0, 3) (the curves' evidences); the control halved in the Scan,
;; the filtered mean without its weights, or the Kalman offsets a step late
;; (the filter's evidence and means, the closed form); a noise of 0.4 or
;; twice the variance, or HMC on the slope alone (the tight line's moments);
;; the fit without the outlier, or the grid's sd without its square root
;; (the guide's checks); 100 training steps (the network's error); noise
;; 0.3 in the data, or the grid cut at -1.55 (sigma's posterior); the
;; plain and eliminated runs swapped (their variance, their exactness);
;; three heads (the coin); a uniform prior cut at 3.2 (NUTS's moments); Z
;; without the end token, or completions of 5 tokens (token SMC's Z); and
;; one per solution. Smaller errors pass: an intercept prior N(0, 8), a walk
;; step of sd 0.6.

;; book: ns
(ns book.gallery
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.conjugate :as conj]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as k]
            [gen.inference.nuts :as nuts]
            [gen.inference.particle-filter :as pf]
            [gen.inference.vi :as vi]
            [gen.inference.weights :as weights]
            [gen.learning :as learn]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]))

;; book: helpers
(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(defn mean-and-se
  "The mean of the host numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]
    [m (Math/sqrt (/ v n))]))

(defn within?
  "Is x within k standard errors se of v?"
  [x v se k]
  (< (abs (- x v)) (* k se)))

(defn log-ml-se
  "The standard error of a log marginal likelihood estimate from n
   weighted samples with effective sample size ess: sqrt(1/ess - 1/n)."
  [ess n]
  (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n))))

(defn run-chain
  "The host values at addrs along a chain of the kernel from the trace t,
   run by k/run with opts, as {addr [x …]}: k/run keeps them (:keep), and
   holds the current trace alone."
  [kernel t addrs opts]
  (:draws (k/run kernel t
                 (assoc opts :keep (into {} (map (fn [a] [a #(% a)])) addrs)))))

;; book: gaussian-posterior
(defn gaussian-posterior
  "The exact posterior of weights w ~ N(0, prior-var I) given ys ~ N(X w,
   noise-var I), X the rows (features x) for each of xs: its mean and
   covariance, and the evidence log p(ys)."
  [features xs ys prior-var noise-var]
  (let [X (array/array (mapv features xs))
        Xt (array/transpose X)
        n (count xs)
        cov (linalg/inv (+ (/ (array/eye (second (array/shape X))) prior-var)
                           (/ (array/matmul Xt X) noise-var)))]
    {:mean (array/matmul cov (/ (array/matmul Xt (array/array ys)) noise-var))
     :cov cov
     :log-evidence (dc/log-prob
                    (d/mvnormal (array/zeros [n])
                                (+ (* prior-var (array/matmul X Xt))
                                   (* noise-var (array/eye n))))
                    (array/array ys))}))

;; book: line-model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])

(def ys [2.6 4.3 7.1 8.4 10.8 12.3 14.0 16.9])

(def line-exact (gaussian-posterior (fn [x] [x 1.0]) xs ys 100.0 1.0))

(array/->clj (:mean line-exact))
;=> [1.9835 0.6233]

;; book: line-is
(defn importance
  "Importance sampling of line-model's posterior from n prior draws,
   batched: its ESS and log marginal likelihood estimate."
  [n key]
  (select-keys (is/importance-sampling line-model [xs] (observations ys) n
                                       {:batched? true :key key})
               [:ess :log-ml-estimate]))

(def small (importance 1000 (rng/key 1)))

(def large (importance 300000 (rng/key 2)))

[(:ess small) (:ess large)]
;=> [3.28 310.3] ; float64

;; book: line-is-check
(def line-evidence (array/item (:log-evidence line-exact)))

line-evidence
;=> -15.354

[(< (:ess small) 20)
 (within? (:log-ml-estimate large) line-evidence
          (log-ml-se (:ess large) 300000) 5)]
;=> [true true]

;; book: centered
(def x-bar (/ (reduce + xs) (count xs)))

(def centered-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          level (trace :level (d/normal (* slope x-bar) 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope (- x x-bar)) level) 1)))
      [slope (- level (* slope x-bar))])))

;; book: line-mala
(def mala-draws
  (run-chain (k/mala (sel/select :slope :level) 0.01)
             (:trace (gfi/generate centered-model [xs] (observations ys)
                                   (rng/key 3)))
             [:slope :level] {:samples 1000 :burn 100 :key (rng/key 4)}))

(def at-ten
  "The line's height at x = 10 in each draw."
  (mapv (fn [slope level] (+ (* slope (- 10.0 x-bar)) level))
        (:slope mala-draws) (:level mala-draws)))

(select-keys (diag/summary at-ten) [:mean :sd :ess-bulk])
;=> {:mean 20.444 :sd 0.882 :ess-bulk 239.5} ; float64

;; book: line-check
(def exact-at-ten
  "The exact posterior mean and standard deviation of the line's height
   at x = 10: v·mean and √(v·Σv), v = [10 1]."
  (let [v (array/array [10.0 1.0])
        {:keys [mean cov]} line-exact]
    [(array/item (array/sum (* v mean)))
     (Math/sqrt (array/item (array/sum (* v (array/matmul cov v)))))]))

exact-at-ten
;=> [20.4586 0.9181]

(let [{:keys [mean sd mcse-mean mcse-sd]} (diag/summary at-ten)]
  [(within? mean (first exact-at-ten) mcse-mean 5)
   (within? sd (second exact-at-ten) mcse-sd 5)])
;=> [true true]

;; book: line-figure
;; figure: figures/054-line.svg
(let [kept (take-nth 50 (map vector (:slope mala-draws) (:level mala-draws)))
      height (fn [[slope level] x] (+ (* slope (- x x-bar)) level))
      noise (array/->clj (rng/normal (rng/key 5) [200]))]
  (svg/paths [{:paths (for [l kept]
                        [[0.0 (height l 0.0)] [10.5 (height l 10.5)]])
               :opacity 0.3 :width 1}
              {:paths [(map vector xs ys)] :width 0 :r 3 :color "black"
               :opacity 1}
              {:paths [(map #(vector 10.0 (+ %1 %2)) at-ten noise)]
               :width 0 :r 2 :color "darkorange" :opacity 0.25}]
             {:x-range [0.0 10.5]}))

;; book: walk-model
(def walk-step
  (gen [t z]
    (let [z' (trace :z (d/normal z 0.5))]
      (trace :y (d/normal z' 1.0))
      z')))

(def walk (comb/unfold walk-step))

(def walk-ys [0.5 1.2 1.8 2.5 1.5 0.9])

(def walk-observations
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) walk-ys)))

;; book: walk-kalman
(defn series
  "The choice map of a state-space generative function's observations ys:
   (ys t) at address t."
  [ys]
  (apply cm/choicemap (interleave (range) ys)))

(def walk-params
  {:transition 1.0 :transition-cov 0.25 :emission 1.0 :emission-cov 1.0})

(def walk-evidence
  (array/item (:weight (gfi/assess ss/kalman
                                   [6 {:mean 0.0 :cov 0.25} walk-params]
                                   (series walk-ys)))))

walk-evidence
;=> -8.7522

;; book: walk-is
(def walk-is
  (is/importance-sampling walk [6 0.0] walk-observations 20000
                          {:batched? true :key (rng/key 6)}))

[(:ess walk-is) (:log-ml-estimate walk-is)]
;=> [2828.5 -8.7537] ; float64

(within? (:log-ml-estimate walk-is) walk-evidence
         (log-ml-se (:ess walk-is) 20000) 5)
;=> true

;; book: walk-hmc
(def state-addrs (mapv #(vector % :z) (range 6)))

(def walk-draws
  (run-chain (k/hmc (sel/from-paths state-addrs) {:L 10 :eps 0.1})
             (:trace (gfi/generate walk [6 0.0] walk-observations
                                   (rng/key 7)))
             state-addrs {:samples 400 :burn 100 :key (rng/key 41)}))

(def walk-summaries (mapv #(diag/summary (walk-draws %)) state-addrs))

(mapv :mean walk-summaries)
;=> [0.4214 0.8380 1.1849 1.4003 1.3326 1.2434] ; float64

;; book: walk-exact
(defn smoothed
  "The exact posterior of the walk's states given ys: their means and
   standard deviations. Its precision is the walk's, D'D/0.5², plus the
   observations', I/1², D taking the differences z_t - z_t-1 (z_-1 = 0);
   its mean is the precision's inverse times ys."
  [ys]
  (let [n (count ys)
        D (array/array (vec (for [i (range n)]
                              (vec (for [j (range n)]
                                     (cond (= i j) 1.0
                                           (= j (dec i)) -1.0
                                           :else 0.0))))))
        cov (linalg/inv (+ (/ (array/matmul (array/transpose D) D) 0.25)
                           (array/eye n)))]
    {:mean (array/->clj (array/matmul cov (array/array ys)))
     :sd (array/->clj (array/sqrt (linalg/diagonal cov)))}))

(def walk-exact (smoothed walk-ys))

(:mean walk-exact)
;=> [0.4446 0.8754 1.2251 1.4310 1.3696 1.2757]

(every? true? (map (fn [{:keys [mean mcse-mean]} m]
                     (within? mean m mcse-mean 5))
                   walk-summaries (:mean walk-exact)))
;=> true

;; book: walk-figure
;; figure: figures/054-walk.svg
(let [{m :mean sd :sd} walk-exact
      line (fn [vs] (map-indexed vector vs))]
  (svg/paths [{:paths [(line walk-ys)] :width 0 :r 3 :color "gray"}
              {:paths [(line m)] :color "black"}
              {:paths [(line (map #(+ %1 (* 2 %2)) m sd))
                       (line (map #(- %1 (* 2 %2)) m sd))]
               :color "black" :opacity 0.4 :width 1}
              {:paths [(line (map :mean walk-summaries))]
               :width 0 :r 3 :color "steelblue" :opacity 1}]
             {:x-label "step t" :y-label "state"}))

;; book: fusion
(def sensor
  (gen [position]
    (trace :value (d/normal position 0.5))))

(def fusion
  (gen [position]
    [(splice :a sensor position) (splice :b sensor position)]))

(def fused (gfi/simulate fusion [5.0] (rng/key 9)))

(cm/addresses (:choices fused))
;=> [[:a :value] [:b :value]]

;; book: fusion-laws
(let [part #(gfi/project fused (sel/select %))]
  [(array/allclose? (+ (part :a) (part :b)) (:score fused))
   (:weight (gfi/update fused (:choices fused) (rng/key 10)))])
;=> [true 0.0]

;; book: map
(def positions [1.0 2.0 3.0])

(def mapped (gfi/simulate (comb/map sensor) [positions] (rng/key 11)))

(cm/addresses (:choices mapped))
;=> [[0 :value] [1 :value] [2 :value]]

(def element-scores
  (map-indexed (fn [i x] (dc/log-prob (d/normal x 0.5) (mapped [i :value])))
               positions))

(array/allclose? (:score mapped) (reduce + element-scores))
;=> true

;; book: curves
(def noisy-curve
  (gen [f xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (f x) 0.5)))))

(def linear
  (gen [xs]
    (let [b (trace :b (d/normal 0 5))
          m (trace :m (d/normal 0 5))]
      (splice noisy-curve (fn [x] (+ b (* m x))) xs))))

(def quadratic
  (gen [xs]
    (let [c (trace :c (d/normal 0 2))
          b (trace :b (d/normal 0 2))
          a (trace :a (d/normal 0 2))]
      (splice noisy-curve (fn [x] (+ c (* b x) (* a x x))) xs))))

(def curve (comb/switch linear quadratic))

(def curve-xs [1.0 2.0 3.0 4.0 5.0])

(def curve-ys [1.1 3.9 9.2 15.8 25.3])

;; book: curves-exact
(defn curve-log-evidence
  "log p(curve-ys) for a curve of the features of x, its weights N(0, s),
   with noise of standard deviation 0.5."
  [features s]
  (array/item (:log-evidence (gaussian-posterior features curve-xs curve-ys
                                                 (* s s) 0.25))))

(def curve-evidence
  [(curve-log-evidence (fn [x] [1.0 x]) 5.0)
   (curve-log-evidence (fn [x] [1.0 x (* x x)]) 2.0)])

curve-evidence
;=> [-40.521 -9.160]

(- (second curve-evidence) (first curve-evidence))
;=> 31.36

;; book: curves-is
(def curve-is
  (mapv (fn [k key]
          (is/importance-sampling curve [k curve-xs] (observations curve-ys)
                                  200000 {:batched? true :key key}))
        [0 1] (rng/split (rng/key 12))))

(mapv :ess curve-is)
;=> [91.6 219.0] ; float64

(mapv (fn [{:keys [ess log-ml-estimate]} exact]
        (within? log-ml-estimate exact (log-ml-se ess 200000) 5))
      curve-is curve-evidence)
;=> [true true]

;; book: scan-model
(def drift-step
  (gen [z u]
    (let [z' (trace :z (d/normal (+ z u) 0.5))]
      (trace :y (d/normal z' 1.0))
      [z' z'])))

(def drift (comb/scan drift-step))

(def controls (into (vec (repeat 8 0.3)) (repeat 8 -0.3)))

(def readings
  [2.94 2.53 2.18 3.9 3.69 3.48 3.76 5.75 4.95 5.68 1.43 1.35 0.98 3.81 1.4
   -0.75])

;; book: scan-kalman
(def drift-trace
  (:trace (gfi/generate ss/kalman
                        [16 {:mean (first controls) :cov 0.25}
                         (mapv #(assoc walk-params :transition-offset %)
                               controls)]
                        (series readings) (rng/key 13))))

(def drift-evidence (array/item (:score drift-trace)))

drift-evidence
;=> -31.9447

;; book: scan-closed-form
(def walk-covariance
  "The covariance of the 16 states of a walk with steps of variance 0.25
   from z_-1 = 0: Cov(z_s, z_t) = 0.25·(min(s, t) + 1)."
  (array/array (vec (for [s (range 16)]
                      (vec (for [t (range 16)]
                             (* 0.25 (inc (min s t)))))))))

(def drift-mean
  "The mean of the 16 states: the running sum of the controls."
  (array/array (vec (reductions + controls))))

(array/allclose? drift-evidence
                 (dc/log-prob (d/mvnormal drift-mean
                                          (+ walk-covariance (array/eye 16)))
                              (array/array readings)))
;=> true

;; book: scan-filter
(defn filtered-mean
  "The weighted mean of the particle filter state's z at step t."
  [{ps :particles lws :log-weights} t]
  (let [p (array/exp (array/array (:log-normalized-weights
                                    (weights/normalize lws))))]
    (array/item (array/sum (* p (batched/choice ps [t :z]))))))

(defn run-filter
  "A bootstrap particle filter of n particles through the readings,
   batched, resampling when the ESS falls below n/2: its log marginal
   likelihood estimate and each step's filtered mean of z."
  [n key]
  (let [ks (rng/split key 32)
        inputs #(subvec controls 0 (inc %))]
    (loop [t 0
           state (pf/initialize drift [0.0 (inputs 0)]
                                (cm/choicemap [0 :y] (readings 0)) n
                                {:batched? true :key (ks 0)})
           means []]
      (let [means (conj means (filtered-mean state t))]
        (if (= t 15)
          {:log-ml (pf/log-ml-estimate state) :means means}
          (recur (inc t)
                 (-> state
                     (pf/maybe-resample {:key (ks (+ 16 t))})
                     (pf/step [0.0 (inputs (inc t))]
                              [diff/no-change
                               (diff/vector-diff (+ t 2) (inc t) {})]
                              (cm/choicemap [(inc t) :y] (readings (inc t)))
                              {:key (ks (inc t))}))
                 means))))))

;; book: scan-runs
(def filter-runs (mapv #(run-filter 200 %) (rng/split (rng/key 14) 40)))

(def kalman-means (mapv #(array/item (:mean %)) (:retval drift-trace)))

[(let [[m se] (mean-and-se (mapv #(Math/exp (- (:log-ml %) drift-evidence))
                                 filter-runs))]
   (within? m 1.0 se 5))
 (every? (fn [t]
           (let [[m se] (mean-and-se (mapv #((:means %) t) filter-runs))]
             (within? m (kalman-means t) se 5)))
         (range 16))]
;=> [true true]

;; book: scan-figure
;; figure: figures/054-drift.svg
(let [sds (mapv #(Math/sqrt (array/item (:cov %))) (:retval drift-trace))
      line (fn [vs] (map-indexed vector vs))]
  (svg/paths [{:paths [(line readings)] :width 0 :r 3 :color "gray"}
              {:paths [(line kalman-means)] :color "black"}
              {:paths [(line (map #(+ %1 (* 2 %2)) kalman-means sds))
                       (line (map #(- %1 (* 2 %2)) kalman-means sds))]
               :color "black" :opacity 0.4 :width 1}
              {:paths [(line (:means (first filter-runs)))]
               :width 0 :r 3 :color "steelblue" :opacity 1}]
             {:x-label "step t" :y-label "state"}))

;; book: tight-model
(def tight-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 5))
          intercept (trace :intercept (d/normal 0 5))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.3)))
      [slope intercept])))

(def tight-xs [0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5])

(def tight-ys [0.62 1.48 2.41 3.55 4.37 5.61 6.49 7.53 8.42 9.57])

(def tight-start
  (:trace (gfi/generate tight-model [tight-xs] (observations tight-ys)
                        (rng/key 15))))

(def tight-line (sel/select :slope :intercept))

;; book: tight-mh
(def mh-draws
  (run-chain (k/mh tight-line) tight-start [:slope :intercept]
             {:samples 1000 :key (rng/key 16)}))

(def accepted
  "How many moves the chain made: the changes of slope along it."
  (dec (count (dedupe (cons (array/->clj (tight-start :slope))
                            (:slope mh-draws))))))

accepted
;=> 3 ; float64

(< accepted 20)
;=> true

;; book: tight-hmc
(def adapted
  (nuts/adapt tight-start tight-line {:kernel :hmc :L 10 :warmup 300}
              (rng/key 17)))

[(:eps adapted) (cm/to-map (:inv-metric adapted))]
;=> [0.340 {:slope 0.00457 :intercept 0.0284}] ; float64

(def hmc-draws
  (run-chain (k/hmc tight-line
                    (assoc (select-keys adapted [:eps :inv-metric]) :L 10))
             (:trace adapted) [:slope :intercept]
             {:samples 2000 :key (rng/key 18)}))

;; book: tight-check
(def tight-exact
  (gaussian-posterior (fn [x] [x 1.0]) tight-xs tight-ys 25.0 0.09))

(def tight-means (array/->clj (:mean tight-exact)))

(def tight-sds (array/->clj (array/sqrt (linalg/diagonal (:cov tight-exact)))))

[tight-means tight-sds]
;=> [[1.9973 0.5109] [0.06602 0.1762]]

(for [[i addr] [[0 :slope] [1 :intercept]]
      :let [{:keys [mean sd mcse-mean mcse-sd]}
            (diag/summary (hmc-draws addr))]]
  [(within? mean (tight-means i) mcse-mean 5)
   (within? sd (tight-sds i) mcse-sd 5)])
;=> [[true true] [true true]]

;; book: vi-model
(def spread-model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 5))
          log-sigma (trace :log-sigma (d/normal 0 1))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu (array/exp log-sigma))))
      mu)))

(def clustered [2.8 3.1 2.9 3.3 3.0 2.7 3.2 8.5])

(/ (reduce + (pop clustered)) 7)
;=> 3.0

;; book: vi-grid
(def grid
  (enum/choice-vol-grid [:mu (vec (range -2.0 9.0001 0.05)) :continuous]
                        [:log-sigma (vec (range -1.5 2.5001 0.025))
                         :continuous]))

(def enumerated
  (enum/enumerative-inference spread-model [8] (observations clustered) grid
                              {:batched? true :key (rng/key 18)}))

(defn grid-moments
  "The posterior mean and standard deviation of the choice at addr, by
   the enumeration over the grid."
  [addr]
  (let [p (mapv #(Math/exp %) (:log-normalized-weights enumerated))
        x (mapv #(array/item (cm/get-choice (first %) addr)) grid)
        m (reduce + (map * p x))]
    [m (Math/sqrt (reduce + (map #(* %1 (- %2 m) (- %2 m)) p x)))]))

[(grid-moments :mu) (grid-moments :log-sigma)]
;=> [[3.606 0.745] [0.686 0.260]]

;; book: vi-guide
(def guide
  (gen []
    (trace :mu (d/normal (param :mu-mean) (array/exp (param :mu-log-sd))))
    (trace :log-sigma (d/normal (param :ls-mean)
                                (array/exp (param :ls-log-sd))))))

(defn advi
  "The guide's store fit by ADVI: Adam at the rate 0.05 for 500
   iterations, then at 0.002 for 1000 more to settle, each with 10
   pathwise draws."
  [key]
  (let [run (fn [store lr iters k]
              (:store (vi/black-box-vi spread-model [8]
                                       (observations clustered) guide []
                                       store (learn/adam {:lr lr})
                                       {:iters iters :samples 10
                                        :estimator :reparam :key k})))
        [k1 k2] (rng/split key)]
    (-> {:mu-mean 0.0 :mu-log-sd 0.0 :ls-mean 0.0 :ls-log-sd 0.0}
        (run 0.05 500 k1)
        (run 0.002 1000 k2))))

(def fitted (update-vals (advi (rng/key 19)) array/item))

(def guide-moments
  {:mu [(:mu-mean fitted) (Math/exp (:mu-log-sd fitted))]
   :log-sigma [(:ls-mean fitted) (Math/exp (:ls-log-sd fitted))]})

guide-moments
;=> {:mu [3.6059 0.6536] :log-sigma [0.6713 0.2400]} ; float64

;; book: vi-check
;; Over ten fits (keys 101-110), the spreads (standard deviations) were
;; 0.032 of mu's mean, 0.0098 of its sd and 0.0053 of log-sigma's mean.
(let [[mu-mean mu-sd] (:mu guide-moments)
      [ls-mean] (:log-sigma guide-moments)
      [exact-mean exact-sd] (grid-moments :mu)
      [exact-ls-mean] (grid-moments :log-sigma)]
  [(within? mu-mean exact-mean 0.032 5)
   (within? ls-mean exact-ls-mean 0.0053 5)
   (> (- exact-sd mu-sd) (* 5 0.0098))])
;=> [true true true]

;; book: vi-conditional
(def conditional-sds
  "The exact posterior standard deviation of mu given each log-sigma of
   the grid whose row holds over 0.001 of the posterior: [log-sigma sd]
   pairs, in order."
  (->> (map (fn [[c _] lw]
              [(array/item (cm/get-choice c :log-sigma))
               (array/item (cm/get-choice c :mu))
               (Math/exp lw)])
            grid (:log-normalized-weights enumerated))
       (group-by first)
       (keep (fn [[log-sigma cells]]
               (let [total (reduce + (map peek cells))
                     mean #(/ (reduce + (map (fn [[_ x p]] (* p (% x))) cells))
                              total)
                     m (mean identity)]
                 (when (> total 0.001)
                   [log-sigma (Math/sqrt (mean #(* (- % m) (- % m))))]))))
       (sort-by first)))

(apply < (map second conditional-sds))
;=> true

;; book: vi-figure
;; figure: figures/054-spread.svg
(let [at (fn [f] (map (fn [[ls _]] [ls (f ls)]) conditional-sds))]
  (svg/paths [{:paths [conditional-sds] :color "black"}
              {:paths [(at #(/ (Math/exp %) (Math/sqrt 8)))] :color "gray"
               :width 1}
              {:paths [(at (constantly (second (:mu guide-moments))))]
               :color "darkorange"}]
             {:x-label "log-sigma" :y-label "sd of mu given log-sigma"}))

;; book: net
(defn mlp
  "A network of one hidden layer of tanh units, applied to each element of
   the 1-d array xs: w1 and b1 the hidden layer's weights and biases, [h]
   each, and w2 ([h]) and b2 the output's."
  [{:keys [w1 b1 w2 b2]} xs]
  (+ (array/matmul (array/tanh (+ (array/outer xs w1) b1)) w2) b2))

(def net
  (custom/deterministic
   {:params [:w1 :b1 :w2 :b2]
    :apply (fn [[xs] params] (mlp params xs))}))

(def sine-fit
  (gen [xs]
    (trace :ys (d/normal (splice :net net xs) 0.1))))

;; book: net-train
(def train-xs (array/linspace (- Math/PI) Math/PI 50))

(defn initial-weights
  "Random weights for h hidden units (the output's scaled by 1/√h), and
   zero biases."
  [h key]
  (let [[k1 k2] (rng/split key)]
    {:w1 (rng/normal k1 [h]) :b1 (array/zeros [h])
     :w2 (/ (rng/normal k2 [h]) (Math/sqrt h)) :b2 0.0}))

(defn train
  "The weights after steps of Adam on sine-fit's log likelihood of
   sin(train-xs), from initial weights drawn with key."
  [steps key]
  (let [[k-init k-steps] (rng/split key)
        targets (cm/choicemap :ys (array/sin train-xs))
        optimizer (learn/adam {:lr 0.01})
        gradient (fn [store k]
                   (grad/param-gradients
                    (:trace (gfi/generate (dyn/with-params sine-fit store)
                                          [train-xs] targets k))))]
    (loop [store (initial-weights 16 k-init)
           state (learn/init optimizer store)
           ks (rng/split k-steps steps)]
      (if-let [[k & more] (seq ks)]
        (let [[store state] (learn/step optimizer state store
                                        (gradient store k))]
          (recur store state more))
        store))))

(def weights (train 1000 (rng/key 21)))

;; book: net-check
(def test-xs (array/linspace -3.0 3.0 61))

(def net-error
  "The network's root-mean-square distance from sin on test-xs."
  (let [gap (- (mlp weights test-xs) (array/sin test-xs))]
    (array/item (array/sqrt (array/mean (array/square gap))))))

net-error
;=> 0.0026 ; float64

;; Over ten trainings (keys 101-110) the error was 0.0043 on average, with
;; a standard deviation of 0.0017: 0.0043 + 5·0.0017 < 0.013.
(< net-error 0.013)
;=> true

;; book: noisy-sine
(def trained (custom/with-params net weights))

(def noisy-sine
  (gen [xs]
    (let [log-sigma (trace :log-sigma (d/normal -1 1))]
      (trace :ys (d/normal (splice :net trained xs) (array/exp log-sigma)))
      log-sigma)))

(def sine-xs (array/linspace 0.0 2.5 100))

(def sine-ys (+ (array/sin sine-xs) (* 0.2 (rng/normal (rng/key 22) [100]))))

(def sine-observed (cm/choicemap :ys sine-ys))

;; book: noisy-sine-trace
(let [t (gfi/simulate trained [sine-xs] (rng/key 23))]
  [(cm/empty? (:choices t)) (:score t)])
;=> [true 0.0]

;; book: noisy-sine-mh
(def sine-start
  (:trace (gfi/generate noisy-sine [sine-xs] sine-observed (rng/key 24))))

(def sigmas
  (let [draws (run-chain (k/mh (sel/select :log-sigma)) sine-start
                         [:log-sigma]
                         {:samples 2000 :burn 100 :key (rng/key 25)})]
    (mapv #(Math/exp %) (:log-sigma draws))))

(select-keys (diag/summary sigmas) [:mean :mcse-mean])
;=> {:mean 0.2221 :mcse-mean 0.0019} ; float64

;; book: noisy-sine-exact
(def sigma-grid
  (enum/choice-vol-grid [:log-sigma (vec (range -3.0 1.0001 0.005))
                         :continuous]))

(def sigma-posterior
  "The exact posterior mean and standard deviation of sigma, by the
   enumeration over sigma-grid."
  (let [enumerated (enum/enumerative-inference noisy-sine [sine-xs]
                                               sine-observed sigma-grid
                                               {:batched? true
                                                :key (rng/key 26)})
        p (mapv #(Math/exp %) (:log-normalized-weights enumerated))
        s (mapv #(Math/exp (array/item (cm/get-choice (first %) :log-sigma)))
                sigma-grid)
        m (reduce + (map * p s))]
    [m (Math/sqrt (reduce + (map #(* %1 (- %2 m) (- %2 m)) p s)))]))

sigma-posterior
;=> [0.2218 0.0159] ; float64

(let [{:keys [mean mcse-mean]} (diag/summary sigmas)
      [m sd] sigma-posterior]
  [(within? mean m mcse-mean 5) (within? m 0.2 sd 5)])
;=> [true true]

;; book: net-figure
;; figure: figures/054-sine.svg
(let [band (* 2 (first sigma-posterior))
      f #(array/item (mlp weights (array/array [%])))]
  (svg/plot [{:points (map vector (array/->clj train-xs)
                           (array/->clj (array/sin train-xs)))
              :color "gray" :r 1.5}
             {:points (map vector (array/->clj sine-xs) (array/->clj sine-ys))
              :r 2}
             {:curves [f] :width 2}
             {:curves [#(+ (f %) band) #(- (f %) band)] :width 1
              :opacity 0.4}]
            {:x-range [-3.2 3.2] :y-range [-1.6 1.6]}))

;; book: conjugate-model
(def gauss-mean
  (gen [n]
    (let [mu (trace :mu (d/normal 0 10))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def eight [2.8 3.2 2.9 3.5 3.1 2.7 3.3 3.0])

(def eight-evidence
  "log p(ys): the ys are jointly normal, with covariance 100 J + I."
  (array/item (dc/log-prob (d/mvnormal (array/zeros [8])
                                       (+ (* 100.0 (array/ones [8 8]))
                                          (array/eye 8)))
                           (array/array eight))))

eight-evidence
;=> -10.9906

(def eliminated (conj/with-conjugate-proposal gauss-mean :mu (sel/select :y)))

;; book: conjugate-runs
(defn log-ml-runs
  "r independent importance sampling estimates of log p(observations),
   each from n samples: one batched generate of r·n samples, its weights
   split into r runs. Host numbers."
  [model args observations r n key]
  (let [{ws :weights} (batched/generate model args observations (* r n) key)]
    (mapv weights/log-mean-exp (partition n (array/->clj ws)))))

(def plain-runs
  (log-ml-runs gauss-mean [8] (observations eight) 100 200 (rng/key 27)))

(def eliminated-runs
  (log-ml-runs eliminated [8] (observations eight) 100 200 (rng/key 27)))

(defn variance-and-se
  "The sample variance of the host numbers xs and its standard error,
   sqrt((m4 - v²)/n), m4 their fourth central moment."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        sq (mapv #(* (- % m) (- % m)) xs)
        v (/ (reduce + sq) (dec n))]
    [v (Math/sqrt (/ (- (/ (reduce + (map #(* % %) sq)) n) (* v v)) n))]))

(variance-and-se plain-runs)
;=> [0.1033 0.0180] ; float64

[(let [[v se] (variance-and-se plain-runs)] (> v (* 5 se)))
 (every? #(array/allclose? % eight-evidence) eliminated-runs)]
;=> [true true]

;; book: coin
(def coin
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (dotimes [i n]
        (trace [:flip i] (d/bernoulli p)))
      p)))

(def flips
  (cm/choicemap [:flip 0] true [:flip 1] true [:flip 2] false [:flip 3] true
                [:flip 4] true))

(def exact-coin (conj/with-conjugate-proposal coin :p (sel/select :flip)))

;; book: coin-check
(defn log-beta
  "log B(a, b)."
  [a b]
  (array/item (- (+ (array/lgamma (double a)) (array/lgamma (double b)))
                 (array/lgamma (double (+ a b))))))

(array/allclose? (:weight (gfi/generate exact-coin [5] flips (rng/key 28)))
                 (- (log-beta 6 3) (log-beta 2 2)))
;=> true

(let [{ps :particles} (batched/generate exact-coin [5] flips 4000
                                        (rng/key 29))
      [m se] (mean-and-se (array/->clj (batched/choice ps :p)))]
  (within? m (/ 6.0 9.0) se 5))
;=> true

;; book: uniform-mean
(def uniform-mean
  (gen [n]
    (let [mu (trace :mu (d/uniform -10 10))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def five (subvec eight 0 5))

;; book: uniform-refused
;; error: with-conjugate-proposal: the latent at [:mu] is a uniform
#_(gfi/generate (conj/with-conjugate-proposal uniform-mean :mu (sel/select :y))
              [5] (observations five) (rng/key 30))

;; book: uniform-nuts
(def uniform-adapted
  (nuts/adapt (:trace (gfi/generate uniform-mean [5] (observations five)
                                    (rng/key 30)))
              (sel/select :mu) {:warmup 300} (rng/key 31)))

(def uniform-draws
  (:mu (run-chain (k/nuts (sel/select :mu)
                          (select-keys uniform-adapted [:eps :inv-metric]))
                  (:trace uniform-adapted) [:mu]
                  {:samples 1000 :key (rng/key 32)})))

(let [{:keys [mean sd mcse-mean mcse-sd]} (diag/summary uniform-draws)]
  [(within? mean 3.1 mcse-mean 5) (within? sd (Math/sqrt 0.2) mcse-sd 5)])
;=> [true true]

;; book: phone-lm
(def tokenizer
  (bt/tokenizer ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "-" " "
                 "Call me at "]))

(def m
  (lm/lm (toy/engine {:n-vocab (:n-vocab tokenizer)
                      :max-branches 256 :n-ctx 64})
         tokenizer {:max-cells 16384}))

(def end (:eos (lm/vocab m)))

(def prompt (lm/encode m "Call me at "))

prompt
;=> [13 12]

(def free (token/completion m))

;; book: phone-grammar
(def phone (grammar/grammar m "[0-9]{2}-[0-9]{2}"))

(defn text
  "The text of completion x, without its end token."
  [x]
  (lm/decode m (if (= end (peek x)) (pop x) x)))

(defn phone-smc
  "A token SMC run of n particles over completions of the prompt that
   match the grammar, of at most 6 tokens: five characters and the end."
  [n key]
  (smc/run free [prompt 6] n {:potential phone :proposal :masked :key key}))

(def phone-run (phone-smc 20 (rng/key 32)))

(take 4 (map text (smc/particles phone-run)))
;=> ["78-59" "78-77" "35-77" "35-77"] ; float64

(every? #(re-matches #"[0-9]{2}-[0-9]{2}" (text %)) (smc/particles phone-run))
;=> true

;; book: phone-exact
(def z
  "Σ p(x) over the completions the grammar accepts: every path of allowed
   tokens from the prompt, each path's probability that of its tokens and
   the end token."
  (letfn [(walk [tokens log-p]
            (let [^floats row (lm/next-logprobs m (into prompt tokens))
                  s (grammar/state phone tokens)
                  ended (if (grammar/accepting? phone s)
                          (Math/exp (+ log-p (aget row (int end))))
                          0.0)]
              (reduce + ended
                      (for [t (grammar/allowed phone s) :when (not= t end)]
                        (walk (conj tokens t)
                              (+ log-p (aget row (int t))))))))]
    (walk [] 0.0)))

z
;=> 3.7876e-4

;; book: phone-unbiased
(defn z-estimates
  "The estimates of Z by runs of token SMC, n particles each."
  [n runs key]
  (mapv #(Math/exp (smc/log-ml-estimate (phone-smc n %)))
        (rng/split key runs)))

(def twenty-particles (z-estimates 20 200 (rng/key 33)))

(let [[mean se] (mean-and-se twenty-particles)]
  (within? mean z se 5))
;=> true

;; book: solution-1
(def walk-prior {:mean 0.0 :cov 0.25})

(def walk-trace
  (:trace (gfi/generate ss/kalman [6 walk-prior walk-params] (series walk-ys)
                        (rng/key 34))))

(def one-more
  (gfi/update walk-trace [7 walk-prior walk-params]
              [diff/unknown-change diff/no-change diff/no-change]
              (cm/choicemap 6 1.4) (rng/key 35)))

(let [{m :mean P :cov} (peek (:retval walk-trace))
      predictive (d/normal m (array/sqrt (+ P 0.25 1.0)))]
  (array/allclose? (:weight one-more) (dc/log-prob predictive 1.4)))
;=> true

;; book: solution-2
(def twenty-flips
  (apply cm/choicemap (mapcat (fn [i] [[:flip i] (< i 15)]) (range 20))))

(array/allclose? (:weight (gfi/generate exact-coin [20] twenty-flips
                                        (rng/key 36)))
                 (- (log-beta 17 7) (log-beta 2 2)))
;=> true

(let [{ps :particles} (batched/generate exact-coin [20] twenty-flips 4000
                                        (rng/key 37))
      [m se] (mean-and-se (array/->clj (batched/choice ps :p)))]
  (within? m (/ 17.0 24.0) se 5))
;=> true

;; book: solution-3
(def five-particles (z-estimates 5 800 (rng/key 38)))

(let [[mean se] (mean-and-se five-particles)
      [v5 se5] (variance-and-se five-particles)
      [v20 se20] (variance-and-se twenty-particles)]
  [(within? mean z se 5)
   (> (- v5 v20) (* 5 (Math/sqrt (+ (* se5 se5) (* se20 se20)))))])
;=> [true true]
