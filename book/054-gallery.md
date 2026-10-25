# A Gallery of Models

This chapter is a gallery: short worked models, each about a page. Each
page gives the model, its inference in a few lines, a figure where one
helps, and the checks that say the answer is right. The checks are exact
against a closed form or an enumeration where the model has one, and
otherwise hold within five standard errors. Each page ends by naming the
chapter that teaches its tools, so the gallery doubles as an index of
the book by example.

The models port the ten example programs of GenMLX (its
`examples/genmlx`). The ideas carry over; the code is new, and its names
and semantics are gen.clj's, which are Gen.jl's. Two of GenMLX's examples
leaned on features gen.clj does without: an analyzer that found conjugate
structure by reading the model's source, and a one-call `fit` that chose
an algorithm. Their page shows the gen.clj way, which declares the
structure. The token SMC example runs on the book's toy language model, so
it needs no model file.

The chapter's code lives in one namespace:

<!-- code: ns -->
```clojure
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
```

A few helpers recur. `run-chain` runs a Markov chain with `k/run`, which
keeps the values at some addresses as host numbers (`:keep`), not the
traces, so a long chain holds one trace at a time on any backend; its
kernels are `gen.inference.kernel`'s (`k/mh`, `k/hmc`, `k/mala`,
`k/nuts`):

<!-- code: helpers -->
```clojure
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
```

Three pages fit Gaussian linear models, whose posterior and evidence have
closed forms. One function gives both, for features of $x$ with weights
$w \sim N(0, s^2 I)$ and noise of variance $\sigma^2$: the posterior's
precision is $I/s^2 + X^\top X/\sigma^2$, and the data are jointly normal,
$y \sim N(0, s^2 X X^\top + \sigma^2 I)$.

<!-- code: gaussian-posterior -->
```clojure
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
```

## Linear regression

A line through eight points, with broad priors on its slope and
intercept:

$$
\text{slope},\ \text{intercept} \sim N(0, 10), \qquad
y_i \sim N(\text{slope} \cdot x_i + \text{intercept},\ 1).
$$

<!-- code: line-model -->
```clojure
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
```

**Importance sampling.** Drawing lines from the prior and weighting each
by the likelihood is the simplest inference there is, and here a poor
one. The prior spreads over tens of units, while the data pin the slope
to within a few tenths:

<!-- code: line-is -->
```clojure
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
```

A thousand prior draws are worth about three, and three hundred thousand
about three hundred; batched, they run in one pass. With that many, the estimate of the evidence $\log p(y)$ agrees with the exact one
within five of its standard errors, $\sqrt{1/\text{ESS} - 1/n}$ for $n$
draws:

<!-- code: line-is-check -->
```clojure
(def line-evidence (array/item (:log-evidence line-exact)))

line-evidence
;=> -15.354

[(< (:ess small) 20)
 (within? (:log-ml-estimate large) line-evidence
          (log-ml-se (:ess large) 300000) 5)]
;=> [true true]
```

**MALA.** The posterior is a narrow ridge: a steeper line fits the points
only with a lower intercept. MALA moves every coordinate with one step
size, so a ridge forces steps sized for its narrow direction. Measuring
$x$ from its mean $\bar x$ removes most of the ridge. The level
$\text{intercept} + \text{slope} \cdot \bar x$ gets the intercept's prior
moved by $\text{slope} \cdot \bar x$, so the model below is the same
distribution over lines in other coordinates:

<!-- code: centered -->
```clojure
(def x-bar (/ (reduce + xs) (count xs)))

(def centered-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          level (trace :level (d/normal (* slope x-bar) 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope (- x x-bar)) level) 1)))
      [slope (- level (* slope x-bar))])))
```

A chain of a thousand MALA steps then mixes well. Its draws give the
posterior of the line's height at $x = 10$, a prediction:

<!-- code: line-mala -->
```clojure
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
```

The exact posterior of that height is normal, with mean $v \cdot \mu$ and
variance $v^\top \Sigma v$ for $v = (10, 1)$. The chain's mean and
standard deviation agree with it within five Monte Carlo standard errors:

<!-- code: line-check -->
```clojure
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
```

<!-- code: line-figure -->
```clojure
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
```

![The data, lines from the MALA chain, and readings predicted at x = 10
(orange): the line's height plus the noise](figures/054-line.svg)

*Getting Started* fits a line with Metropolis–Hastings, *The Inference
Toolkit* teaches importance sampling, and *Gradient-Based MCMC* the ridge
and MALA.

## A hidden Markov model

A hidden position walks, $z_t \sim N(z_{t-1}, 0.5)$ from $z_{-1} = 0$,
and each step is measured with noise, $y_t \sim N(z_t, 1)$. One step is a
generative function, and Unfold repeats it:

<!-- code: walk-model -->
```clojure
(def walk-step
  (gen [t z]
    (let [z' (trace :z (d/normal z 0.5))]
      (trace :y (d/normal z' 1.0))
      z')))

(def walk (comb/unfold walk-step))

(def walk-ys [0.5 1.2 1.8 2.5 1.5 0.9])

(def walk-observations
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) walk-ys)))
```

The model is linear and Gaussian, so the states can be summed out
exactly. `gen.state-space`'s Kalman generative function does it: its
choices are the observations alone, and its score is $\log p(y)$.

<!-- code: walk-kalman -->
```clojure
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
```

Importance sampling from the prior works on six steps. Its estimate of
the evidence agrees with the Kalman filter's:

<!-- code: walk-is -->
```clojure
(def walk-is
  (is/importance-sampling walk [6 0.0] walk-observations 20000
                          {:batched? true :key (rng/key 6)}))

[(:ess walk-is) (:log-ml-estimate walk-is)]
;=> [2828.5 -8.7537] ; float64

(within? (:log-ml-estimate walk-is) walk-evidence
         (log-ml-se (:ess walk-is) 20000) 5)
;=> true
```

HMC moves all six states at once, along the gradient of the log density,
which suits states that move together:

<!-- code: walk-hmc -->
```clojure
(def state-addrs (mapv #(vector % :z) (range 6)))

(def walk-draws
  (run-chain (k/hmc (sel/from-paths state-addrs) {:L 10 :eps 0.1})
             (:trace (gfi/generate walk [6 0.0] walk-observations
                                   (rng/key 7)))
             state-addrs {:samples 400 :burn 100 :key (rng/key 41)}))

(def walk-summaries (mapv #(diag/summary (walk-draws %)) state-addrs))

(mapv :mean walk-summaries)
;=> [0.4214 0.8380 1.1849 1.4003 1.3326 1.2434] ; float64
```

The exact posterior of the states is normal, with the walk's precision
plus the observations':

<!-- code: walk-exact -->
```clojure
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
```

<!-- code: walk-figure -->
```clojure
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
```

![The readings (gray), the exact posterior mean of each state with two
standard deviations either side, and HMC's means
(blue)](figures/054-walk.svg)

*Composition* teaches Unfold, *Gradient-Based MCMC* runs HMC and NUTS on
a longer walk, and *Exact Structure, Made Explicit* teaches the Kalman
generative function.

## Compositional models

Models compose. A sensor reads a position with noise, and `splice` calls
it twice under two addresses, so the two readings' choices nest without
colliding:

<!-- code: fusion -->
```clojure
(def sensor
  (gen [position]
    (trace :value (d/normal position 0.5))))

(def fusion
  (gen [position]
    [(splice :a sensor position) (splice :b sensor position)]))

(def fused (gfi/simulate fusion [5.0] (rng/key 9)))

(cm/addresses (:choices fused))
;=> [[:a :value] [:b :value]]
```

The score is the sum of the parts. `project` reads the part of the score
a selection's choices make, and the two parts add up to the whole. An
update that sets every choice to the value it has changes nothing, so its
weight is 0:

<!-- code: fusion-laws -->
```clojure
(let [part #(gfi/project fused (sel/select %))]
  [(array/allclose? (+ (part :a) (part :b)) (:score fused))
   (:weight (gfi/update fused (:choices fused) (rng/key 10)))])
;=> [true 0.0]
```

Map replicates a generative function over its arguments, each element
under its index, and its score is the sum of the elements':

<!-- code: map -->
```clojure
(def positions [1.0 2.0 3.0])

(def mapped (gfi/simulate (comb/map sensor) [positions] (rng/key 11)))

(cm/addresses (:choices mapped))
;=> [[0 :value] [1 :value] [2 :value]]

(def element-scores
  (map-indexed (fn [i x] (dc/log-prob (d/normal x 0.5) (mapped [i :value])))
               positions))

(array/allclose? (:score mapped) (reduce + element-scores))
;=> true
```

Composition also builds rival models from shared parts. A line and a
parabola differ in their parameters and share the noisy observation of a
curve, which each splices in. A Switch makes them one generative function
of a branch index:

<!-- code: curves -->
```clojure
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
```

The data come from $y \approx x^2$. Both models are linear in their
weights, so their evidences are exact, and the log Bayes factor, the
difference of the log evidences, is decisive for the parabola:

<!-- code: curves-exact -->
```clojure
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
```

Importance sampling on each branch of the Switch agrees with both:

<!-- code: curves-is -->
```clojure
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
```

*Composition* teaches splice, Map and Switch; *Update and Regenerate*
teaches project and update; *Conditioning* teaches the evidence.

## A scan time series

Scan is Unfold with inputs: at each step its kernel takes the carried
state and that step's input. Here the input is a control that pushes the
walk up for eight steps and down for eight:

<!-- code: scan-model -->
```clojure
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
```

The Kalman generative function takes the controls as offsets, its
parameters a vector of one map per step (step 0's offset is in the
prior's mean). Its score is the evidence:

<!-- code: scan-kalman -->
```clojure
(def drift-trace
  (:trace (gfi/generate ss/kalman
                        [16 {:mean (first controls) :cov 0.25}
                         (mapv #(assoc walk-params :transition-offset %)
                               controls)]
                        (series readings) (rng/key 13))))

(def drift-evidence (array/item (:score drift-trace)))

drift-evidence
;=> -31.9447
```

As a check of the parameters, the readings are jointly normal: their mean
is the running sum of the controls, and their covariance the walk's plus
the noise's.

<!-- code: scan-closed-form -->
```clojure
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
```

A particle filter grows the Scan one step at a time. Each step extends
the inputs by one, and its argdiff, a vector-diff with no element
changed, says that only the new step is new, which lets the Scan's
update leave the others alone (*Composition* shows it):

<!-- code: scan-filter -->
```clojure
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
```

The filter's evidence estimate is unbiased in $\exp$ space, and its
filtered means estimate the Kalman filter's. Forty runs of 200 particles
agree with both at every step, within five standard errors:

<!-- code: scan-runs -->
```clojure
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
```

<!-- code: scan-figure -->
```clojure
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
```

![The readings (gray), the Kalman filter's mean of each state given the
readings so far, with two standard deviations either side, and one
particle filter's means (blue)](figures/054-drift.svg)

*Composition* teaches Scan and its incremental updates, *SMC in Depth*
the particle filter, and *Exact Structure, Made Explicit* the Kalman
generative function inside one.

## Adaptive HMC

A line through ten points with little noise, $\sigma = 0.3$, has a narrow,
correlated posterior:

<!-- code: tight-model -->
```clojure
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
```

Metropolis–Hastings with proposals from the prior, what `k/mh` does
with a selection, almost never lands on the ridge. In a thousand steps it
moves three times:

<!-- code: tight-mh -->
```clojure
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
```

HMC needs a step size and a metric that fit the posterior's scales.
`nuts/adapt` tunes both during a warmup, as Stan does: dual averaging
moves the step size toward an acceptance rate of 0.8, and windows of
draws set the inverse metric to the variances they show. With `:kernel
:hmc` the warmup runs HMC with a fixed number of leapfrog steps; the
default kernel is NUTS (`nuts/sample` adapts and then draws, keeping the
traces). The inverse metric is in the order of the selection's values,
the intercept first. The chain then draws with what the warmup found:

<!-- code: tight-hmc -->
```clojure
(def adapted
  (nuts/adapt tight-start tight-line {:kernel :hmc :L 10 :warmup 300}
              (rng/key 17)))

[(:eps adapted) (array/->clj (:inv-metric adapted))]
;=> [0.340 [0.0284 0.00457]] ; float64

(def hmc-draws
  (run-chain (k/hmc tight-line
                    (assoc (select-keys adapted [:eps :inv-metric]) :L 10))
             (:trace adapted) [:slope :intercept]
             {:samples 2000 :key (rng/key 18)}))
```

The draws agree with the exact posterior's means and standard deviations
within five Monte Carlo standard errors:

<!-- code: tight-check -->
```clojure
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
```

*Gradient-Based MCMC* teaches HMC, NUTS and Stan's warmup.

## Variational inference

Eight readings, one of them an outlier. The model has an unknown mean and
an unknown scale, with a normal prior on the log of the scale:

<!-- code: vi-model -->
```clojure
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
```

No closed form fits this posterior, but it has two dimensions, so a grid
gives it to the digits shown. `enumerative-inference` weighs each cell
of the grid by its posterior probability:

<!-- code: vi-grid -->
```clojure
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
```

The outlier pulls the mean to 3.6, from about 3.0 for the other seven,
and widens the scale to $e^{0.69} \approx 2$.

ADVI fits a guide with independent normals on the two choices, its
parameters trainable, by ascending the ELBO along pathwise gradients:

<!-- code: vi-guide -->
```clojure
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
```

A fitted guide is a learned result, so its band comes from the spread of
ten fits with other keys. The guide's means agree with the exact ones.
Its standard deviation of the mean is too small, by far more than five
spreads:

<!-- code: vi-check -->
```clojure
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
```

The guide cannot express that the mean's spread grows with the scale.
Given $\sigma$, the mean's posterior has a standard deviation near
$\sigma/\sqrt 8$, and on the grid it grows with $\sigma$:

<!-- code: vi-conditional -->
```clojure
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
```

The exact posterior's variance of the mean averages $\sigma^2/8$ over
$\sigma$. The guide's, with $\sigma$ drawn independently, is about
$1/(8\,\mathbb{E}[\sigma^{-2}])$, which is smaller.

<!-- code: vi-figure -->
```clojure
;; figure: figures/054-spread.svg
(let [at (fn [f] (map (fn [[ls _]] [ls (f ls)]) conditional-sds))]
  (svg/paths [{:paths [conditional-sds] :color "black"}
              {:paths [(at #(/ (Math/exp %) (Math/sqrt 8)))] :color "gray"
               :width 1}
              {:paths [(at (constantly (second (:mu guide-moments))))]
               :color "darkorange"}]
             {:x-label "log-sigma" :y-label "sd of mu given log-sigma"}))
```

![The exact posterior standard deviation of mu given log-sigma (black),
near σ/√8 (gray), and the guide's, the same at every log-sigma
(orange)](figures/054-spread.svg)

*Variational Inference* teaches the ELBO, its estimators and the bias of
a guide family; *Debugging with Enumeration* teaches grids.

## A neural network inside a model

A neural network is a deterministic generative function with trainable
parameters: it makes no choices, and its weights live in a store. This
one has a hidden layer of 16 tanh units. A model calls it to predict
$\sin x$, with fixed noise:

<!-- code: net -->
```clojure
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
```

Training ascends the model's log likelihood of $\sin x$ at 50 points,
which is minimizing the squared error, with Adam:

<!-- code: net-train -->
```clojure
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
```

A trained network is a learned result, so the band of its error comes
from ten trainings with other keys:

<!-- code: net-check -->
```clojure
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
```

The trained network goes into a second model, which reads a hundred new
points with noise of unknown size. The points are $\sin x$ plus noise of
standard deviation 0.2:

<!-- code: noisy-sine -->
```clojure
(def trained (custom/with-params net weights))

(def noisy-sine
  (gen [xs]
    (let [log-sigma (trace :log-sigma (d/normal -1 1))]
      (trace :ys (d/normal (splice :net trained xs) (array/exp log-sigma)))
      log-sigma)))

(def sine-xs (array/linspace 0.0 2.5 100))

(def sine-ys (+ (array/sin sine-xs) (* 0.2 (rng/normal (rng/key 22) [100]))))

(def sine-observed (cm/choicemap :ys sine-ys))
```

The network is still a generative function without choices, of score 0:

<!-- code: noisy-sine-trace -->
```clojure
(let [t (gfi/simulate trained [sine-xs] (rng/key 23))]
  [(cm/empty? (:choices t)) (:score t)])
;=> [true 0.0]
```

Metropolis–Hastings on the log of the noise, from the prior:

<!-- code: noisy-sine-mh -->
```clojure
(def sine-start
  (:trace (gfi/generate noisy-sine [sine-xs] sine-observed (rng/key 24))))

(def sigmas
  (let [draws (run-chain (k/mh (sel/select :log-sigma)) sine-start
                         [:log-sigma]
                         {:samples 2000 :burn 100 :key (rng/key 25)})]
    (mapv #(Math/exp %) (:log-sigma draws))))

(select-keys (diag/summary sigmas) [:mean :mcse-mean])
;=> {:mean 0.2221 :mcse-mean 0.0019} ; float64
```

The posterior has one dimension, so a fine grid gives it to the digits
shown. The chain agrees with it, and the posterior puts the noise near
the 0.2 we added, well within five of its standard deviations:

<!-- code: noisy-sine-exact -->
```clojure
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
```

<!-- code: net-figure -->
```clojure
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
```

![The training targets on sin x (gray), the new readings (black), the
network, and two inferred standard deviations of noise either
side](figures/054-sine.svg)

*Learning* teaches parameters, optimizers and the network as a
generative function.

## Exact structure, declared

GenMLX found conjugate pairs by reading a model's source, and its `fit`
chose an algorithm from what it found. gen.clj finds nothing by itself:
the model says where its structure is, with `with-conjugate-proposal`
for a conjugate latent and with `gen.state-space` for chains (the two
time series above). A mean with a normal prior, seen through eight normal
readings, is conjugate:

<!-- code: conjugate-model -->
```clojure
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
```

Plain importance sampling estimates the evidence with an error. With the
mean drawn from its exact posterior, every weight is the evidence
itself. A hundred runs of 200 samples each show the difference:

<!-- code: conjugate-runs -->
```clojure
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
```

GenMLX's analyzer needed static addresses. The declaration does not
care: `gauss-mean` computes its addresses in a loop.

A coin with a beta prior, flipped five times, is conjugate too. With four
heads, the posterior is $\text{beta}(2 + 4, 2 + 1)$, of mean $2/3$, and
the evidence is $B(6, 3)/B(2, 2)$:

<!-- code: coin -->
```clojure
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
```

<!-- code: coin-check -->
```clojure
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
```

A uniform prior on the mean has no conjugate partner here, and the
wrapper says so:

<!-- code: uniform-mean -->
```clojure
(def uniform-mean
  (gen [n]
    (let [mu (trace :mu (d/uniform -10 10))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def five (subvec eight 0 5))
```

<!-- code: uniform-refused -->
```clojure
;; error: with-conjugate-proposal: the latent at [:mu] is a uniform
(gfi/generate (conj/with-conjugate-proposal uniform-mean :mu (sel/select :y))
              [5] (observations five) (rng/key 30))
```

There, NUTS is the tool, after Stan's warmup. The posterior is the
normal of mean $\bar y = 3.1$ and standard deviation $1/\sqrt{5}$, cut at
$\pm 10$, fifteen standard deviations away:

<!-- code: uniform-nuts -->
```clojure
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
```

*Exact Structure, Made Explicit* teaches the five conjugate families and
the state-space generative functions.

## Token SMC

GenMLX's last example asks a language model to complete a prompt with
text that matches a regular expression, by sequential Monte Carlo over
tokens. Here the model is the book's toy engine, whose next-token
distributions are pseudo-random but fixed, with a vocabulary of digits, a
dash, a space and the prompt as one token:

<!-- code: phone-lm -->
```clojure
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
```

The grammar of the completion is two digits, a dash and two digits. It
is a potential on token sequences, and SMC with the masked proposal
draws each token from the model's distribution restricted to the tokens
the grammar allows. A completion here has at most six tokens: the five
characters and the end token. Every particle matches:

<!-- code: phone-grammar -->
```clojure
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
```

The target is the language model's distribution conditioned on the
grammar, and its normalizer $Z$ is the probability that the model's
completion matches. The vocabulary is small enough to sum it exactly,
over every path the grammar allows:

<!-- code: phone-exact -->
```clojure
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
```

SMC's estimate of $Z$ is unbiased. Over 200 runs, its mean agrees with
the exact $Z$ within five standard errors:

<!-- code: phone-unbiased -->
```clojure
(defn z-estimates
  "The estimates of Z by runs of token SMC, n particles each."
  [n runs key]
  (mapv #(Math/exp (smc/log-ml-estimate (phone-smc n %)))
        (rng/split key runs)))

(def twenty-particles (z-estimates 20 200 (rng/key 33)))

(let [[mean se] (mean-and-se twenty-particles)]
  (within? mean z se 5))
;=> true
```

*Language Models as Generative Functions* teaches the language model
protocol and completions, and *Constrained Generation and Token SMC*
potentials, grammars and token SMC.

The gallery ends the main text. The appendices follow, beginning with
*Coming from Gen.jl*.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **One more reading.** Extend the hidden Markov model's Kalman trace
   with a seventh reading, 1.4, by `gfi/update`. Predict the update's
   weight from the last filtered belief, then check it.
2. **More flips.** Flip the coin twenty times, fifteen heads. What is the
   evidence, and the posterior mean? Check both.
3. **Fewer particles.** Run token SMC with 5 particles in place of 20.
   Is the estimate of $Z$ still unbiased? Does its variance grow?

## Solutions

**1.** The weight of an update that adds a step is the log predictive
density of the new reading. The last belief is the state's normal given
the six readings; one more step of the walk adds 0.25 to its variance,
and the noise adds 1:

<!-- code: solution-1 -->
```clojure
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
```

**2.** The posterior is $\text{beta}(2 + 15, 2 + 5)$, of mean $17/24$,
and the evidence is $B(17, 7)/B(2, 2)$:

<!-- code: solution-2 -->
```clojure
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
```

**3.** SMC's estimate is unbiased for any number of particles, so the
mean still agrees with $Z$. Its variance grows, by more than five
standard errors of the difference; 800 runs measure it well enough:

<!-- code: solution-3 -->
```clojure
(def five-particles (z-estimates 5 800 (rng/key 38)))

(let [[mean se] (mean-and-se five-particles)
      [v5 se5] (variance-and-se five-particles)
      [v20 se20] (variance-and-se twenty-particles)]
  [(within? mean z se 5)
   (> (- v5 v20) (* 5 (Math/sqrt (+ (* se5 se5) (* se20 se20)))))])
;=> [true true]
```
