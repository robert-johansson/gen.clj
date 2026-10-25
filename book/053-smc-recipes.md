# SMC Recipes

Part II built particle filters one operation at a time and measured what
each part buys. This chapter collects recipes: particle filters assembled
for problems that come up again and again. Each recipe is a section in
four parts: the problem, the gen.clj code, a check of the answer, and when
to use it. The checks are the book's: exact where the model allows it, by
enumeration or the Kalman filter, and otherwise statistical, at five
standard errors.

The recipes port two libraries built on Gen.jl. GenParticleFilters
supplies a particle filter's building blocks. Its documentation's example,
an object that is either still or moving, opens the chapter, and its
operations follow: initialization with strata, residual resampling,
rejuvenation, resizing and translators. GenSMCP3 automates SMCP3, whose
proposals are probabilistic programs. Its README's toy and its tutorial's
"challenging observations" close the chapter. *Sequential Monte Carlo:
Object Tracking* introduced the filter's operations, *SMC in Depth*
measured their accuracy, and *Trace Translators and Involutive MCMC*
built the translators that SMCP3's steps are; this chapter uses them
without deriving them again.

<!-- code: ns -->
```clojure
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
```

GenParticleFilters' operations have these counterparts. Its state is
mutable and its functions end in `!`; gen.clj's state is a value, and
each operation returns a new one.

| GenParticleFilters | gen.clj |
|--------------------|---------|
| `pf_initialize(model, args, obs, n)` | `pf/initialize` |
| … with `strata` | `pf/initialize` with `:strata` (`pf/choice-product`) |
| `pf_update!(state, args, argdiffs, obs)` | `pf/step` |
| … with a proposal (and a transform) | `pf/step` with `:proposal`; `tf/extending-translator` |
| … with forward and backward proposals | `pf/step` with `:backward`; `tf/updating-translator` |
| … with a trace translator | `pf/move-reweight` |
| … with `strata` | none yet (bean genclj-gvju); by hand below |
| … with `check = false` | none: a step that changes a choice is a loud error |
| `pf_resample!` (methods, `priority_fn`) | `pf/resample`, `pf/maybe-resample` (`:method`, `:priority`) |
| `pf_rejuvenate!`, `pf_move_accept!` | `pf/rejuvenate` with a kernel |
| `pf_move_reweight!`, `move_reweight` | `pf/move-reweight` with `pf/reweighting` |
| `pf_resize!` (multinomial, residual, optimal) | `pf/resize` with `:method` |
| `pf_replicate!`, `pf_dereplicate!` | `pf/replicate`, `pf/dereplicate` |
| `pf_coalesce!`, `pf_introduce!` | `pf/coalesce`, `pf/introduce` |
| `state[idxs]`, a substate | `pf/view`, `pf/update-view` |
| `mean`, `var`, `proportionmap` | `pf/mean`, `pf/variance`, `pf/proportions` |
| `effective_sample_size`, `log_ml_estimate` | `pf/ess`, `pf/log-ml-estimate` |
| `ExtendingTraceTranslator`, `UpdatingTraceTranslator` | `tf/extending-translator`, `tf/updating-translator` |
| GenSMCP3's `@kernel` and `SMCP3Update` | trace kernels and `tk/smcp3` |

## A filter with resampling and rejuvenation

**The problem.** An object at height $y$ is either still or moving. At each
step it keeps its mode with probability 0.75, moves by $\sin t$ when it
moves, drifts by a little noise, and is observed with noise 0.25. From the
observations we want to know when it started moving. This is
GenParticleFilters' example, and its filter is the plain recipe:
resample when the weights grow uneven, rejuvenate after resampling, and
step.

GenParticleFilters writes the model as a loop over $t = 1, \dots, T$. Here
it is an Unfold (*Composition*), whose kernel takes the step $t$, from 0,
and the previous state, and returns the next. The source's velocity at its
time $t$ is $\sin t$, so step $t$ here moves by $\sin(t + 1)$, and the
source's time 6 is step 5. The kernel picks between values with
`array/where`, not `if`, so that it runs on every backend and batched:

<!-- code: model -->
```clojure
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
```

Step $t$'s choices nest under $t$:

<!-- code: addresses -->
```clojure
(cm/addresses (:choices (gfi/simulate object-motion [2 still] (rng/key 1))))
;=> [[0 :moving] [0 :y] [0 :y-obs] [1 :moving] [1 :y] [1 :y-obs]]
```

As in GenParticleFilters' example, the data come from the model: ten
steps, still for the first five and moving for the last five.

<!-- code: data -->
```clojure
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
```

<!-- code: data-figure -->
```clojure
;; figure: figures/053-motion.svg
(let [positions (mapv #(array/item (truth [% :y])) (range T))]
  (svg/paths [{:paths [(map-indexed vector positions)]}
              {:paths [(map-indexed vector ys)] :r 3 :width 0 :color "black"}]
             {:x-label "step t" :y-label "y"}))
```

![The object's heights (blue), still for five steps and then moving, and
the observations (black)](figures/053-motion.svg)

The exact answer comes from enumeration. Given the ten moving choices the
model is linear and Gaussian: the height is a random walk whose step $t$ is
offset by its velocity, so `ss/kalman` (*SMC in Depth*) gives the
likelihood $p(y \mid m)$ of the observations, with the offsets as its
`:transition-offset`, a function of the step. There are $2^{10} = 1024$
sequences $m$. Their priors times their likelihoods are the joint
$p(m, y)$, whose sum is the evidence and whose normalized values are the
posterior:

<!-- code: exact -->
```clojure
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
  (mapv #(Math/exp %) (:log-weights (weights/normalize lw))))

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
```

The object most probably started moving at step 5, the source's time 6:
the posterior puts 0.94 on step 5 and nothing on step 4. A second route to
the likelihood checks the Kalman filter's arguments. The observations are
jointly normal given $m$: the height at $t$ is the sum of $t + 1$ steps,
each its velocity plus noise of spread 0.01, so the covariance of the
heights at $s$ and $t$ is $0.01^2 (\min(s, t) + 1)$, and each observation
adds $0.25^2$:

<!-- code: exact-check -->
```clojure
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
```

The filter is GenParticleFilters' `particle_filter`. Before each step, if
the effective sample size has fallen below half the particles, it
resamples by the residual method and moves each particle's latest choices
with MH. The move proposes them from their prior and leaves the posterior
given the observations so far invariant, so the weights stay as they are
(*Sequential Monte Carlo: Object Tracking*). The source's selection names
step $t$ as well, which the traces do not have yet before the step; it
selects nothing there, so the selection here names step $t - 1$ alone.
`particle-filter` takes the move as an argument, for the workshop:

<!-- code: filter -->
```clojure
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
```

GenParticleFilters reads a filter with `mean(state, addr)`. gen.clj's
`pf/mean` takes a function of the trace in place of an address, so the
statistic can be any function of the choices; a boolean counts as 1 or 0.
`pf/proportions` is `proportionmap`, the weight of each value, and
`pf/variance` is `var`, which for a boolean is $p(1 - p)$:

<!-- code: statistics -->
```clojure
[(pf/mean filtered #(% [4 :moving])) (pf/mean filtered #(% [5 :moving]))]
;=> [0.0 0.9703] ; float64

(pf/proportions filtered #(% [5 :moving]))
;=> {false 0.0297 true 0.9703} ; float64

(let [p (pf/mean filtered #(% [5 :moving]))]
  (array/allclose? (pf/variance filtered #(% [5 :moving])) (* p (- 1 p))))
;=> true
```

This filter puts 0.97 on step 5, near the exact 0.94. One run is one draw
of the filter's randomness; forty runs of it, each with its own key, give
a mean and a standard error. `summary` keeps each run's answers as host
numbers and drops its traces:

<!-- code: runs -->
```clojure
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
```

The mean of the forty estimates of $P(\text{moving at } 5 \mid y)$ agrees
with the exact value within five standard errors, every run puts less
than 1% on step 4, and the estimates of the evidence are unbiased: the
mean of $\exp(\text{estimate} - \log p(y))$ is 1 within five standard
errors.

<!-- code: runs-check -->
```clojure
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
```

<!-- code: moving-figure -->
```clojure
;; figure: figures/053-moving.svg
(let [filter-means (apply mapv (fn [& ps] (:mean (spread ps)))
                          (map :moving runs))]
  (svg/plot [{:points (map-indexed vector exact-moving) :r 6
              :color "darkorange" :opacity 0.5}
             {:points (map-indexed vector filter-means)}]
            {:x-range [-0.5 9.5] :y-range [-0.05 1.05]
             :x-label "step t" :y-label "P(moving at t | ys)"}))
```

![The exact P(moving at t | y) (orange) and the mean of forty filters'
estimates (black)](figures/053-moving.svg)

*Residual resampling*, the filter's method, keeps $\lfloor n w_i \rfloor$
copies of each particle $i$ of normalized weight $w_i$, and draws the rest
in proportion to the remainders $n w_i - \lfloor n w_i \rfloor$. The copies
it keeps cost no randomness. On the first step's 100 particles, 82 copies
are kept so, and every particle gets at least its share:

<!-- code: residual -->
```clojure
(def first-population
  (pf/initialize object-motion [1 still] (observation ys 0) 100
                 {:key (rng/key 32)}))

(let [lw (pf/log-weights first-population)
      expected (mapv #(* 100 %) (probabilities lw))
      copies (frequencies (weights/residual lw 100 (rng/key 33)))]
  [(reduce + (map #(Math/floor %) expected))
   (every? #(>= (get copies % 0) (Math/floor (expected %))) (range 100))])
;=> [82.0 true] ; float64
```

gen.clj's residual resampling has a known flaw: when $n w_i$ is an integer,
as with equal weights, rounding can put it just below, and the floor then
loses the copy (bean genclj-tx1h). The estimate stays unbiased, but the
variance residual resampling was chosen to cut comes back. A filter that
resamples only when the effective sample size has fallen, as this one,
never resamples equal weights.

**When to use it.** This is the default filter for a model that unfolds
over time: resample when the effective sample size falls below $n/2$,
with the residual or systematic method (*SMC in Depth* compares them), and
rejuvenate the choices that the latest observations inform. The MH move
costs one update per particle and needs no weights of its own.

## Covering a discrete choice: strata

**The problem.** The first step has a discrete choice, moving or not, and
one of its values is rare a priori: a plain initialization draws "moving"
for about a quarter of the particles, and the estimate's noise comes
mostly from how many it draws. *Stratified initialization* fixes those
counts. GenParticleFilters' `choiceproduct` lists the *strata*, a choice
map for each combination of values; `pf/choice-product` is its
counterpart. With `:strata`, `pf/initialize` constrains $\lfloor n/k
\rfloor$ particles to each of the $k$ strata (in contiguous blocks, or
interleaved with `:layout :interleaved`), draws the strata of any left
over at random, and adds $\log k$ to every weight:

<!-- code: strata -->
```clojure
(def first-moves (pf/choice-product [[[0 :moving] [false true]]]))

(mapv cm/to-map first-moves)
;=> [{0 {:moving false}} {0 {:moving true}}]

(def stratified
  (pf/initialize object-motion [1 still] (observation ys 0) 20
                 {:key (rng/key 40) :strata first-moves}))

(let [moving (mapv #(% [0 :moving]) (pf/traces stratified))]
  [(frequencies (subvec moving 0 10)) (frequencies (subvec moving 10))])
;=> [{false 10} {true 10}]
```

The evidence of the first observation is a sum over the two values, the
height summed out of each, so it is known exactly. Two hundred
initializations of 20 particles each way estimate it:

<!-- code: strata-exact -->
```clojure
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
```

Both estimates are unbiased, and the stratified one spreads 24 times
less. A difference of variances is measured on their logs: the log of a
sample variance has a standard error of about $\sqrt{(\kappa - 1)/n}$,
where $\kappa$ is the sample's fourth moment over its squared variance (3
for a normal sample). The stratified variance is smaller by far more than
five standard errors of the difference:

<!-- code: strata-check -->
```clojure
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
```

GenParticleFilters' `pf_update!` takes strata too, a *stratified step*:
each particle's new choices constrained to its stratum, every weight
multiplied by $k$. gen.clj's `pf/step` has no strata yet (bean
genclj-gvju). The next recipe builds by hand the case that needs it most.

**When to use it.** Stratify a discrete choice of few values, chiefly a
rare one, at the start of a filter. The continuous choices beside it are
still drawn, so strata remove only the discrete part of the noise.

## Enumerating a discrete choice: Fearnhead and Clifford

**The problem.** The moving choice comes back at every step, and the filter
draws it each time. A particle that draws the improbable value wastes an
update, and one that misses the probable value loses it until a later
move finds it. With two values, every particle can be extended with both:
two copies, one moving and one still, each weighed by the probability of
its value. Then $2n$ particles cover every continuation of the $n$, and
*optimal resampling* (Fearnhead and Clifford 2003) keeps $n$ of them, each
at most once. GenParticleFilters supplies the pieces: `pf_replicate!`, the
stratified `pf_update!` and `pf_optimal_resize!`.

gen.clj has the first and the last, `pf/replicate` and `pf/resize` with
`:method :optimal`. Until `pf/step` takes strata, `pf/update-view` does
the middle by hand: it runs an operation on some of the particles, here a
step of each block of copies with the moving choice constrained to one
value. When the strata cover every value, the copies of a particle are an
enumeration, and the stratified weight $k\,p(\text{value}, y_t \mid
\dots)$ averaged over its $k$ copies is the particle's weight with the
choice summed out:

<!-- code: enumerated-step -->
```clojure
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
```

From three particles, one enumerated step makes six: block 0, the first
three, still at step 1, block 1 moving, and each block a copy of the three
particles' past:

<!-- code: enumerated-example -->
```clojure
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
```

The filter enumerates at every step and resizes back to $n$. It starts
from a stratified initialization of $2n$ particles, which covers the first
step's choice the same way:

<!-- code: fearnhead-clifford -->
```clojure
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
```

Optimal resampling keeps each particle of a large weight once, with its
weight, and draws the others systematically, each at most once, with a
common weight. The kept weights add up to what they were, so the estimate
of the evidence does not change:

<!-- code: optimal -->
```clojure
(let [doubled (enumerated-step filtered [(inc T) still]
                               (cm/choicemap [T :y-obs] 1.0)
                               [T :moving] [false true] (rng/key 44))
      halved (pf/resize doubled 100 {:method :optimal :key (rng/key 45)})]
  [(count (pf/log-weights doubled)) (count (distinct (:parents halved)))
   (array/allclose? (pf/log-ml-estimate halved) (pf/log-ml-estimate doubled))])
;=> [200 100 true]
```

Forty runs of 50 particles agree with the exact answer, and their
estimates of the evidence are unbiased. Each step makes 100 updates, as
the first recipe's filter of 100 particles does before its rejuvenation
moves, but the estimates spread 5.5 times less, a variance 30 times
smaller, far beyond five standard errors:

<!-- code: fc-runs -->
```clojure
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
```

**When to use it.** For a model whose latent state at each step is
discrete with few values, or has a discrete part with few values, such as
a switching model: enumerate the discrete part and resize optimally. The
cost is $k$ updates per particle per step, which pays when the plain
filter would need more than $k$ times the particles. For a choice of many
values, stratify a few of them or draw it.

## Resizing and substates

**The problem.** A filter's number of particles is a budget, and it need
not be fixed. GenParticleFilters' resizing page has four more
operations. `pf/resize` grows or shrinks the population by resampling, by
any method; `pf/coalesce` merges particles that hold the same choices,
which resampling makes; `pf/replicate` makes $k$ copies of every particle,
and `pf/dereplicate` keeps one of each group of $k$, the first or one drawn
by weight; `pf/introduce` adds fresh particles, generated with the
observations so far. The first three keep what the particles say, and the
checks are exact.

Growing the final population of the first recipe to 400 by residual
resampling, and merging its copies again, changes neither the estimate of
the evidence nor any weighted mean:

<!-- code: resizing -->
```clojure
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
```

Replicating multiplies the effective sample size by $k$ and keeps the
estimate, and dereplicating with `:keepfirst` undoes it exactly:

<!-- code: replicate -->
```clojure
(def copies (pf/replicate filtered 4))

[(array/allclose? (pf/ess copies) (* 4 (pf/ess filtered)))
 (array/allclose? (pf/log-ml-estimate copies) (pf/log-ml-estimate filtered))
 (= (pf/log-weights (pf/dereplicate copies 4)) (pf/log-weights filtered))]
;=> [true true true]
```

A *substate*, GenParticleFilters' `state[idxs]`, is some of the
particles: `pf/view` returns one as a state of its own, and
`pf/update-view` runs any operation on it and puts the result back. The
enumerated step above extended blocks of copies with it, and the next
recipe resamples groups of particles independently with it.

**When to use it.** Grow a filter before a stretch of surprising data and
shrink it after; replicate before an exploratory step and dereplicate by
weight after it; coalesce the copies of a discrete model's particles to
save work. Substates run any operation on part of a filter, such as
resampling groups independently.

## SMCP3 proposals: the challenging observations

**The problem.** An object performs a random walk in the plane, $z_t \sim
\mathcal{N}(z_{t-1}, I)$ from $z_{-1} = 0$, and is observed with noise,
$y_t \sim \mathcal{N}(z_t, I)$. GenSMCP3's tutorial gives a sequence of 51
observations that jump: from one step to the next they move by up to four
units, where the walk moves by about one. A bootstrap filter proposes each
$z_t$ from its prior, blind to $y_t$, and loses the object. SMCP3
(Lew et al. 2023) makes the proposal a probabilistic program that looks
at $y_t$, and weighs it correctly with a backward program.

The model is the tutorial's, with `normal` on a vector mean drawing a
vector (Gen.jl's `broadcasted_normal`):

<!-- code: walk-model -->
```clojure
(def walk-step
  (gen [t prev]
    (let [z (trace :z (d/normal prev 1.0))]
      (trace :y (d/normal z 1.0))
      z)))

(def walk (comb/unfold walk-step))

(def origin (array/zeros [2]))
```

The observations, rounded to three decimals:

<!-- code: challenging -->
```clojure
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
```

The model is linear and Gaussian, so the Kalman filter gives the evidence
and the filtered means $E[z_t \mid y_0, \dots, y_t]$ exactly:

<!-- code: walk-exact -->
```clojure
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
```

The filter is the tutorial's. It starts from the model with no steps, and
at each step extends every particle by one step with that step's
observation. Its resampling is GenSMCP3's `maybe_perform_resampling!`: the
30 particles in five groups of six, each group resampled by the residual
method when its effective sample size falls below a fifth of its size,
which `pf/update-view` does group by group. `walk-filter` takes the step
and the resampling as arguments, and records the filtered mean after each
step; the bootstrap step is `pf/step`:

<!-- code: walk-filter -->
```clojure
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
```

The tutorial's SMCP3 step has three parts. The *forward kernel* draws $v$
from the prior, $\mathcal{N}(z_{t-1}, I)$, then takes one step of the
unadjusted Langevin algorithm from $v$ toward the observation: up the
gradient of $\log p(z_t = v \mid z_{t-1}) + \log p(y_t \mid z_t = v)$, of
size $\epsilon = 0.8$, with noise of spread $\sqrt{2\epsilon}$. The
gradient is $(z_{t-1} - v) + (y_t - v)$, since both variances are 1. The
new $z_t$ is that draw. The *backward kernel* scores the $v$ that the step
forgot, here from the prior again. The tutorial notes that this is not
the best backward kernel, which would draw $v$ given $z_{t-1}$ and $z_t$,
but a valid one. The *transform* hands the forward kernel's $z$ to the
trace and its $v$ to the backward kernel; its inverse gives them back. An
*updating translator* (`tf/updating-translator`, GenParticleFilters'
`UpdatingTraceTranslator`) combines them with the model's update, and
`pf/move-reweight` applies it to every particle, as GenParticleFilters'
`pf_update!` with a translator does:

<!-- code: ula -->
```clojure
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
```

The step's weight is the one of *Trace Translators and Involutive MCMC*:

$$
w = \log p(z_t \mid z_{t-1}) + \log p(y_t \mid z_t)
  + \log q_{\text{bwd}}(v) - \log q_{\text{fwd}}(v, z_t)
  + \log \lvert \det J \rvert .
$$

The transform only copies, so $J$ is the identity, and the forward and
backward densities of $v$ are the same normal, so they cancel: the weight
is the model's density of the new step over the density of the Langevin
draw of $z_t$. With `:check? true` the translator runs its inverse, the
backward kernel's way back, and throws unless it gives back the trace and
the forward choices:

<!-- code: smcp3-checked -->
```clojure
(def five
  "A trace of walk through the first five observations."
  (:trace (gfi/generate walk [5 origin] (walk-observations 5) (rng/key 60))))

(:weight ((smcp3-step 5 {:check? true}) five (rng/key 61)))
;=> -33.0 ; float64
```

Ten runs of each filter, with the same keys:

<!-- code: walk-runs -->
```clojure
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
```

The bootstrap filter's estimates of the evidence fall more than a
thousand nats short of the exact $-251.7$, and its filtered means are six
units from the Kalman filter's on average. SMCP3's are within 41 nats and
one unit. Both gaps exceed five standard errors of the difference. A log
estimate falls short on average, since its exponential is the unbiased
one; SMCP3's 30 particles still leave a gap of more than five standard
errors, which more particles would close:

<!-- code: walk-check -->
```clojure
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
```

<!-- code: walk-figure -->
```clojure
;; figure: figures/053-tracks.svg
(svg/paths [{:paths [challenging] :r 2.5 :width 0 :color "gray"}
            {:paths [walk-means] :color "black" :width 2}
            {:paths [(:means (first bootstrap-runs))] :color "darkorange"}
            {:paths [(:means (first smcp3-runs))] :color "steelblue"}]
           {:x-label "x" :y-label "y"})
```

![The observations (gray), the Kalman filter's means (black), and one
run's filtered means of the bootstrap filter (orange) and of SMCP3
(blue)](figures/053-tracks.svg)

**When to use it.** When the observations are informative enough that
proposals from the prior miss them, and the locally optimal proposal of
*SMC in Depth* has no closed form: a proposal program can take a gradient
step, run a few iterations of an optimizer, or draw auxiliary choices,
and the backward kernel and the Jacobian keep the weights right. The
closer the backward kernel is to the conditional of the forgotten choices,
the less the weights vary.

## The kernel DSL: GenSMCP3's form

**The problem.** GenSMCP3 writes the forward and backward programs in
GenTraceKernelDSL's `@kernel` language: a kernel takes the trace, makes
random choices, and returns two choice maps, the update for the model's
trace and the choices the other kernel would make to undo it. There is no
separate transform. gen.clj's `gen.inference.trace-kernel` has the same
form: a *trace kernel* is a generative function whose first argument is
the trace and whose return value is `[update backward]`, and `tk/smcp3` is
GenTraceKernelDSL's `run_smcp3_step`, a translator for `pf/move-reweight`.
The tutorial's step as kernels:

<!-- code: kernels -->
```clojure
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
```

It is the same step. Both split a step's key the same way, and the
forward kernel makes the same choices as the translator's forward
proposal, at the same addresses, so with the same key the two draw the
same values. On thirty particles of ten steps, they give the same weights
and the same new states:

<!-- code: kernels-agree -->
```clojure
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
```

They differ in how they find the Jacobian. A transform's covers its own
reads and writes. A trace kernel's covers the kernel's choices and the
continuous choices of the trace, since the kernel may read any of them:
it runs the kernel again on the trace, regenerated from the choices, and
differentiates the outputs, so its work grows with the trace. With
`:check? true`, `tk/smcp3` also checks that the backward kernel gives back
the forward kernel's choices and that its update restores the trace. A
backward kernel that forgets the forward kernel's $z$ fails it:

<!-- code: kernels-error -->
```clojure
;; error: smcp3: the backward kernel does not give back the forward kernel's
(def forgetful
  (gen [tr t y]
    (let [v (trace :v (d/normal (previous tr t) 1.0))]
      [cm/EMPTY (cm/choicemap :v v)])))

((kernel-step 5 {:l forgetful :check? true}) five (rng/key 65))
```

The README's toy shows a step that revises a choice. A value $x \sim
\mathcal{N}(0, 100)$ is observed with unit noise, one observation per
step:

<!-- code: point-model -->
```clojure
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
```

At each step the forward kernel proposes a new $x$ near the mean of the
observations so far, and hands the old $x$ to the backward kernel, which
must say how the old value could have come about: from the mean of the
earlier observations, or from the prior at the first step. These are the
README's kernels with one change: the README's backward kernel draws the
first old value from $\mathcal{N}(0, 1)$, which is not the prior;
exercise 3 shows what it costs.

<!-- code: point-kernels -->
```clojure
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
```

`obs-sum` adds the observations with gen.array's `+`, not on the host.
The Jacobian follows the kernel's arithmetic, and a value read on the host
would lose its gradient, so reading one is a loud error:

<!-- code: point-host-error -->
```clojure
;; error: Cannot read a value being differentiated on the host
(def host-forward
  (gen [tr y]
    (let [n (inc (first (:args tr)))
          total (+ (array/item (obs-sum tr (dec n))) y)
          new-x (trace :new-x (d/normal (/ total n) 1.0))]
      [(cm/choicemap :x new-x) (cm/choicemap :previous-x (tr :x))])))

((tk/smcp3 {:p-new-args [2] :new-observations (cm/choicemap [:obs 1] 2.0)
            :k host-forward :k-args [2.0] :l point-backward :l-args [2.0]})
 (:trace (gfi/generate point-model [1] (cm/choicemap [:obs 0] 1.0)
                       (rng/key 73)))
 (rng/key 74))
```

The filter is the README's `run_smcp3`: SMCP3 steps from the prior, with
residual resampling when the effective sample size falls below a fifth of
the particles.

<!-- code: point-filter -->
```clojure
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
```

The posterior is conjugate. The observations are jointly normal, each of
variance $1 + 100^2$ and each pair of covariance $100^2$, and the
posterior mean of $x$ is $\sum y / (n + 100^{-2})$: for the README's
observations $(1, 2, 3)$ about 2, and for $(1, 2, 3, 10)$ about 4, as the
README says.

<!-- code: point-exact -->
```clojure
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
```

Ten runs of 200 particles agree with both, the mean and the evidence:

<!-- code: point-runs -->
```clojure
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
```

<!-- code: point-figure -->
```clojure
;; figure: figures/053-point.svg
(let [s (point-filter [1.0 2.0 3.0 10.0] 200 (rng/key 72))
      posterior (d/normal 3.9999 (Math/sqrt (/ 1 4.0001)))]
  (svg/histogram [{:values (mapv #(% :x) (pf/traces s))
                   :weights (probabilities (pf/log-weights s))
                   :density #(Math/exp (array/item (dc/log-prob posterior %)))}]
                 {:range [2.5 5.5] :bins 24 :x-label "x"}))
```

![One run's particles after the observations (1, 2, 3, 10), weighted, and
the exact posterior density of x](figures/053-point.svg)

**When to use it.** Write a step as trace kernels when its proposal is
easiest to say as a program that computes the new values and the backward
choices together, as GenSMCP3's examples do; write it as a transform with
a translator when the trace is long, since the transform's Jacobian does
not grow with it. Both give the same weights.

## Choosing a recipe

| Situation | Recipe |
|-----------|--------|
| A model unfolding over time | resampling below $n/2$ (`pf/maybe-resample`, `:residual`), MH rejuvenation (`pf/rejuvenate`), `pf/step` |
| A rare value of a discrete choice at the start | stratified initialization (`:strata`, `pf/choice-product`) |
| A discrete choice of few values at every step | enumeration (`pf/replicate`, `pf/update-view`) and `pf/resize` with `:method :optimal` |
| A budget of particles that changes | `pf/resize`, `pf/replicate`, `pf/dereplicate`, `pf/coalesce`, `pf/introduce` |
| Informative observations, no closed-form proposal | SMCP3: `tf/updating-translator` with `pf/move-reweight` |
| A proposal written as one program | trace kernels and `tk/smcp3` |
| A step that revises earlier choices | a backward kernel for what the step forgets |

Every recipe ran its model through the generative function interface and
nothing else: the particle filter's operations, the translators and the
kernels are built from `generate`, `update`, `propose` and `assess`. The
next chapter, *A Gallery of Models*, turns from inference to the models
themselves.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Move-reweight.** Replace the first recipe's MH move with a
   move-reweight move (GenParticleFilters' `pf_move_reweight!`): the same
   choices regenerated from their prior, each weight multiplied by the
   move's, `pf/move-reweight` with `pf/reweighting`. Predict the effective
   sample size after the move. Does the filter stay correct?
2. **Resampling together.** Resample the challenging observations'
   bootstrap filter as a whole, when its effective sample size falls below
   15, in place of five groups. Does it catch up with SMCP3?
3. **The README's backward kernel.** Draw the first old value from
   $\mathcal{N}(0, 1)$, as the README does, in place of the prior. Predict
   the effective sample size after the first step of 200 particles.
4. **The bootstrap as SMCP3.** Write the bootstrap step of the
   challenging observations as an updating translator whose forward
   proposal draws $z_t$ from its prior, as GenSMCP3's tutorial does before
   its Langevin step. Predict its weights.
5. **Priorities.** GenParticleFilters' `priority_fn` resamples in
   proportion to priorities other than the weights, such as tempered
   weights, $w^{1/2}$. Predict whether more particles survive a
   resampling, and whether the estimate of the evidence stays unbiased.

## Solutions

**1.** A resampled population has equal weights, and a move-reweight move
makes them unequal again: regenerating the latest choices from their
prior changes how well each particle explains the latest observation, and
the weight records it. Of 100 particles, 4.4 effective remain. The filter
stays correct, as move-reweight keeps the estimate unbiased, but its
answers are noisier. An MH move rejects the bad proposals instead of
paying for them.

<!-- code: solution-1 -->
```clojure
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
```

**2.** Resampling the whole population whenever its effective sample size
falls below half lets good particles replace bad ones across the groups,
and the bootstrap filter's estimate rises from about −1414 to −421. It is
still far below SMCP3's, by more than five standard errors of the
difference: no resampling makes up for proposals that ignore the data.

<!-- code: solution-2 -->
```clojure
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
```

**3.** At the first step the old $x$ came from the prior, whose spread is
100, so it is typically tens of units from 0. A backward kernel of spread
1 gives such a value a density near zero, and the weights then favor the
few particles whose old $x$ happened to be near 0: 3.5 effective particles
of 200, where the prior as the backward kernel keeps 157. The weights are
still valid; they are only far more uneven.

<!-- code: solution-3 -->
```clojure
(let [start (pf/initialize point-model [0] cm/EMPTY 200 {:key (rng/key 84)})
      first-ess (fn [backward]
                  (pf/ess (pf/move-reweight start (point-step [1.0] 0 backward)
                                            (rng/key 85))))]
  [(first-ess (backward-kernel 1.0)) (first-ess point-backward)])
;=> [3.47 156.9] ; float64
```

**4.** The forward proposal is the prior, $q(z_t) = p(z_t \mid z_{t-1})$,
and there are no backward choices, so the weight is the update's,
$\log p(z_t \mid z_{t-1}) + \log p(y_t \mid z_t)$, less $\log q(z_t)$: the
observation's density, $\log p(y_t \mid z_t)$, as for `pf/step`'s
bootstrap step. `dc/log-prob` of a vector normal gives each element's
density, and the weight is their sum:

<!-- code: solution-4 -->
```clojure
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
```

**5.** Tempered priorities are less uneven than the weights, so a
resampling keeps more distinct particles: of 100 importance samples with
an effective sample size of 14, about 30 survive against 25. Each copy of
particle $a$ then carries the weight $w_a / \pi_a$, its weight over its
priority, so the estimate of the evidence stays unbiased:

<!-- code: solution-5 -->
```clojure
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
```
