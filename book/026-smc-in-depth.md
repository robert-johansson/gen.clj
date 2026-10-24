# SMC in Depth

*Sequential Monte Carlo: Object Tracking* built a particle filter and
followed an object with it. This chapter asks what makes such a filter
accurate, and measures each answer against an exact one. A particle filter
is importance sampling that moves through the data: at each step it extends
every sample by one observation, reweights it, and, when the weights grow
uneven, resamples. Each of these has a way to fail and a remedy, and on a
model with a known answer both can be measured.

The chapter ports the GenMLX book's chapter on sequential Monte Carlo: a
state-space model with a Kalman answer, the price of resampling, the
effective sample size, low-variance resampling, path degeneracy and
rejuvenation, SMCP3, and particle MCMC (PMMH, with conditional SMC and
particle Gibbs beside it). Its model is linear and Gaussian, so the
Kalman filter gives every answer exactly: the evidence, the filtering and
smoothing distributions, and a parameter's posterior. Where gen.clj
behaves differently from GenMLX, the text says so. *Conditioning* and *The
Inference Toolkit* introduced the pieces used here: importance sampling,
the effective sample size, and kernels.

<!-- code: ns -->
```clojure
(ns book.smc-in-depth
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.importance :as imp]
            [gen.inference.kernel :as kernel]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.particle-mcmc :as pmcmc]
            [gen.inference.transform :as tf]
            [gen.inference.weights :as weights]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]))
```

## A state-space model with a known answer

The model is the smallest sequential one: a hidden state that drifts, seen
through noise.

$$
z_0 \sim \mathcal{N}(0, 1), \qquad
z_t \sim \mathcal{N}(a z_{t-1}, q), \qquad
y_t \sim \mathcal{N}(z_t, r).
$$

As everywhere in gen.clj, `normal` takes a standard deviation, so $q$ and
$r$ are standard deviations. With $a = 1$ the state is a random walk; the
GenMLX book's chapter used $q = 0.5$ and $r = 1$. An Unfold
(*Composition*) is the model. Its kernel draws step $t$'s state from the
previous one, which the Unfold threads from step to step, and then the
step's observation. The initial state is $z_{-1} = 0$, and the first step
has spread 1 in place of $q$. The parameters are one map, the kernel's
last argument, which every step receives:

<!-- code: model -->
```clojure
(def step
  (gen [t prev {:keys [a q r]}]
    (let [z (trace :z (d/normal (* a prev) (if (zero? t) 1.0 q)))]
      (trace :y (d/normal z r))
      z)))

(def ssm (comb/unfold step))

(def walk {:a 1.0 :q 0.5 :r 1.0})
```

Step $t$'s choices nest under $t$:

<!-- code: addresses -->
```clojure
(cm/addresses (:choices (gfi/simulate ssm [2 0.0 walk] (rng/key 1))))
;=> [[0 :y] [0 :z] [1 :y] [1 :z]]
```

The data are the GenMLX chapter's eight observations:

<!-- code: data -->
```clojure
(def ys [0.7 1.3 1.9 2.4 1.6 0.9 0.2 -0.4])

(defn observation
  "The choice map of observation t of ys."
  [ys t]
  (cm/choicemap [t :y] (ys t)))

(defn observations
  "The choice map of every observation of ys."
  [ys]
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) ys)))
```

Every site is linear and Gaussian, so the states can be summed out
exactly. `gen.state-space/kalman`, the sibling of the `hmm` that
*Composition* used, is a generative function whose only choices are the
observations, each with its density given the ones before, $p(y_t \mid
y_0, \dots, y_{t-1})$, from the Kalman filter. So its score is the
*evidence* $\log p(y)$, and its return value the *filtered* beliefs $p(z_t
\mid y_0, \dots, y_t)$, a mean and a variance at each step:

<!-- code: exact -->
```clojure
(defn kalman-args
  "gen.state-space/kalman's arguments for the model with params, on n
   steps."
  [n {:keys [a q r]}]
  [n {:mean 0.0 :cov 1.0}
   {:transition a :transition-cov (* q q) :emission 1.0 :emission-cov (* r r)}])

(defn exact
  "The Kalman filter of the model with params on the observations ys:
   {:log-ml log p(ys), :beliefs, the mean and standard deviation of each
   z_t given ys up to t}, host numbers."
  [params ys]
  (let [{:keys [weight retval]}
        (gfi/assess ss/kalman (kalman-args (count ys) params)
                    (apply cm/choicemap (interleave (range) ys)))
        belief (fn [{:keys [mean cov]}]
                 {:mean (array/item mean) :sd (Math/sqrt (array/item cov))})]
    {:log-ml (array/item weight) :beliefs (mapv belief retval)}))

(def kalman (exact walk ys))

(:log-ml kalman)
;=> -11.775

(peek (:beliefs kalman))
;=> {:mean 0.3958 :sd 0.6249}
```

The evidence is −11.775, the GenMLX chapter's value, and the last state's
filtered distribution has mean 0.396 and standard deviation 0.625. The
Kalman filter is a recursion, and a second route to the evidence checks
it. The observations are jointly normal: $z_t$ is $z_0$ plus $t$
independent steps, so $\mathrm{Cov}(z_s, z_t) = 1 + q^2 \min(s, t)$, and
each $y_t$ adds $r^2$ of its own. Their joint density at the data is the
evidence:

<!-- code: joint -->
```clojure
(defn walk-covariance
  "The covariance of the random walk's states z_0 … z_n-1: z_t is z_0 plus
   t independent steps, so Cov(z_s, z_t) = 1 + q^2 min(s, t)."
  [n q]
  (array/array (vec (for [s (range n)]
                      (vec (for [t (range n)] (+ 1.0 (* q q (min s t)))))))))

(defn joint-log-ml
  "log p(ys) of the random walk with params: the observations are jointly
   normal, with the states' covariance plus r^2 on the diagonal."
  [{:keys [q r]} ys]
  (let [n (count ys)]
    (dc/log-prob (d/mvnormal (array/zeros [n])
                             (+ (walk-covariance n q) (* r r (array/eye n))))
                 (array/array ys))))

(array/allclose? (joint-log-ml walk ys) (:log-ml kalman))
;=> true
```

These answers are the yardstick for the rest of the chapter. A particle
filter whose estimates do not approach them on this model is wrong.

## Importance sampling: the evidence, not the state

Importance sampling with the model's internal proposal draws each sample's
whole path $z_0, \dots, z_7$ from the prior, and weighs it by how well it
explains the eight observations, $p(y \mid z)$. Three thousand samples,
batched:

<!-- code: importance -->
```clojure
(def is-result
  (imp/importance-sampling ssm [(count ys) 0.0 walk] (observations ys) 3000
                           {:key (rng/key 2) :batched? true}))

[(:log-ml-estimate is-result) (:ess is-result)]
;=> [-11.804 212.9] ; float64
```

As *Conditioning* derived, the standard error of a log evidence estimate
is about $\sqrt{1/\mathrm{ESS} - 1/n}$. The estimate is within five of
them of the exact value:

<!-- code: importance-check -->
```clojure
(defn log-ml-se
  "The standard error of the log marginal likelihood estimate of n
   weighted samples with effective sample size ess: about
   sqrt(1/ess - 1/n)."
  [ess n]
  (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n))))

(< (abs (- (:log-ml-estimate is-result) (:log-ml kalman)))
   (* 5 (log-ml-se (:ess is-result) 3000)))
;=> true
```

But the 3000 samples are worth 213: most paths drawn from the prior stray
from some observation. The same weights estimate the last state's filtered
mean, with a standard error from the weights:

<!-- code: importance-state -->
```clojure
(defn weighted
  "The weighted mean of (f trace) over traces with log weights lws, and
   its standard error, about sqrt(Σ w^2 (f - mean)^2), w the normalized
   weights: {:mean :se}."
  [f traces lws]
  (let [ws (mapv #(Math/exp %) (:log-weights (weights/normalize lws)))
        xs (mapv #(array/item (f %)) traces)
        mean (reduce + (map * ws xs))
        sq (fn [w x] (let [e (* w (- x mean))] (* e e)))]
    {:mean mean :se (Math/sqrt (reduce + (map sq ws xs)))}))

(def is-last
  (weighted #(% [7 :z]) (imp/traces is-result) (:log-weights is-result)))

is-last
;=> {:mean 0.394 :se 0.041} ; float64

(< (abs (- (:mean is-last) (:mean (peek (:beliefs kalman)))))
   (* 5 (:se is-last)))
;=> true
```

It agrees with the exact 0.396 within its standard error, 0.041, which is
about what 213 independent draws from the posterior would give, $0.625 /
\sqrt{213} = 0.043$. Each further observation multiplies another factor
into every weight and makes them more uneven, and over a hundred
observations importance sampling fails outright, as the section on the
effective sample size shows.

## The bootstrap particle filter

A particle filter keeps importance sampling's weighted samples, its
*particles*, but moves them through the data one observation at a time.
It starts as importance sampling on the first observation. At each later
step it extends every particle by one step of the model and multiplies
the particle's weight by the new observation's density. Between steps,
when the weights have grown uneven, it *resamples*: it draws $n$ particles
from the current ones in proportion to their weights and gives them equal
weights, so that particles that explain the data are copied and the others
dropped. When each step's new state is drawn from its prior, as here, this
is the *bootstrap* filter.

gen.clj's particle filter has Gen.jl's operations, the ones *Sequential
Monte Carlo: Object Tracking* uses: `pf/initialize`, `pf/step` (every
particle's trace updated to one more step, with that step's observation),
and `pf/maybe-resample`. Unlike Gen.jl's, its state is a value: each operation
returns a new state. `run-filter` is the loop, written so that later
sections can replace its first step and its steps:

<!-- code: filter -->
```clojure
(def argdiffs
  "A step's argdiffs: the number of steps grows, the rest stays."
  [diff/unknown-change diff/no-change diff/no-change])

(defn bootstrap-init
  "n particles generated with the first observation, batched."
  [params ys n key]
  (pf/initialize ssm [1 0.0 params] (observation ys 0) n
                 {:key key :batched? true}))

(defn bootstrap-step
  "The particles extended by step t, with observation t: each new state
   drawn from its prior, each weight multiplied by its observation's
   density."
  [state params ys t key]
  (pf/step state [(inc t) 0.0 params] argdiffs (observation ys t) {:key key}))

(defn run-filter
  "A particle filter of n particles on the model with params, over the
   observations ys: (init params ys n key), then for each later t, a
   resampling when the effective sample size is below :ess-threshold (n/2
   by default; :method, :multinomial by default), and (advance state
   params ys t key). By default the bootstrap filter. The final state,
   with :ess, the effective sample size after each observation, and
   :resamples, how many times it resampled; with :record, a function of
   the state and t, also :records, its value after each observation."
  [params ys n {:keys [ess-threshold method init advance record key]
                :or {ess-threshold (/ n 2.0) method :multinomial
                     init bootstrap-init advance bootstrap-step}}]
  (let [ks (rng/split key (count ys))
        noted (fn [state t]
                (cond-> (update state :ess conj (pf/ess state))
                  record (update :records conj (record state t))))]
    (reduce (fn [state t]
              (let [[k-resample k-advance] (rng/split (ks t))
                    resampled (pf/maybe-resample
                               state {:ess-threshold ess-threshold
                                      :method method :key k-resample})]
                (-> (advance resampled params ys t k-advance)
                    (update :resamples + (if (:resampled? resampled) 1 0))
                    (noted t))))
            (noted (assoc (init params ys n (first ks))
                          :ess [] :records [] :resamples 0)
                   0)
            (range 1 (count ys)))))
```

The threshold is a number of particles, as in Gen.jl (GenMLX took a
fraction of $n$). With `{:batched? true}` the particles run as one batched
execution, and every later step stays batched. A thousand particles:

<!-- code: bootstrap -->
```clojure
(defn newest-mean
  "The weighted mean of the newest state, z_t, over the particles."
  [state t]
  (array/item (pf/mean state #(% [t :z]))))

(def bootstrap
  (run-filter walk ys 1000 {:key (rng/key 3) :record newest-mean}))

[(pf/log-ml-estimate bootstrap) (:resamples bootstrap)]
;=> [-11.8 2] ; float64
```

The filter resampled twice in seven steps. `newest-mean`, its `:record`,
noted its estimate of each state's filtered mean, $E[z_t \mid y_0, \dots,
y_t]$, after each observation. The figure draws curves through values with
`through`:

<!-- code: through -->
```clojure
(defn through
  "The function of x that is linear between the values vs at x = 0, 1, 2,
   …: a curve through them, for svg/plot."
  [vs]
  (fn [x]
    (let [i (max 0 (min (long (Math/floor x)) (- (count vs) 2)))
          f (- x i)]
      (+ (* (- 1.0 f) (vs i)) (* f (vs (inc i)))))))
```

<!-- code: bootstrap-figure -->
```clojure
;; figure: figures/026-filtered-means.svg
(let [{:keys [beliefs]} kalman
      band (fn [k] (mapv #(+ (:mean %) (* k (:sd %))) beliefs))]
  (svg/plot [{:points (map-indexed vector ys) :color "gray" :r 2}
             {:curves [(through (mapv :mean beliefs))]}
             {:curves [(through (band -2)) (through (band 2))] :opacity 0.3}
             {:points (map-indexed vector (:records bootstrap))}]
            {:x-range [0 7] :y-range [-1.5 3.5]
             :x-label "step t" :y-label "z_t"}))
```

![The filter's estimates of the filtered means (black), the Kalman filter's
means with two standard deviations either side (blue), and the observations
(gray)](figures/026-filtered-means.svg)

One run is one draw of the filter's randomness. Its accuracy shows over
many runs: here 200 filters of 100 particles, and 200 of the same filter
with resampling turned off. A filter that never resamples is importance
sampling done one observation at a time, *sequential importance sampling*:
the same proposal and the same weights.

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

(defn summaries
  "For filters of n particles from each of runs keys (opts as run-filter's),
   the log marginal likelihood estimate and the estimate of the last
   state's mean: host numbers, one map per run."
  [params ys n runs key opts]
  (mapv (fn [k]
          (let [state (run-filter params ys n (assoc opts :key k))]
            {:log-ml (pf/log-ml-estimate state)
             :last (newest-mean state (dec (count ys)))}))
        (rng/split key runs)))

(def filtered (summaries walk ys 100 200 (rng/key 4) {}))

(def sequential-is (summaries walk ys 100 200 (rng/key 4) {:ess-threshold 0}))
```

A particle filter's estimate of the evidence is unbiased for $p(y)$ (not
for its log), whether or not it resamples, a law that gen.clj's tests of
the particle filter check. Over the runs, the mean of
$\exp(\text{estimate} - \log p(y))$ is 1 within five standard errors, for
both:

<!-- code: runs-evidence -->
```clojure
(defn evidence-ratio
  "The spread of exp(estimate - log p(ys)) over runs: its mean is 1 for an
   unbiased estimate of p(ys)."
  [runs log-ml]
  (spread (mapv #(Math/exp (- (:log-ml %) log-ml)) runs)))

(mapv (fn [runs]
        (let [{:keys [mean se]} (evidence-ratio runs (:log-ml kalman))]
          (< (abs (- mean 1.0)) (* 5 se))))
      [filtered sequential-is])
;=> [true true]
```

The difference is in the state. The filter's estimates of the last state's
mean spread half as much as those of sequential importance sampling, with
a quarter of the variance. The log of the ratio of two sample variances has
a standard error of about $\sqrt{4/(n - 1)}$, and the ratio exceeds 1 by
far more than five of them:

<!-- code: runs-spread -->
```clojure
[(:sd (spread (map :last filtered))) (:sd (spread (map :last sequential-is)))]
;=> [0.094 0.18] ; float64

(defn log-variance-ratio
  "log(var a / var b) of two samples of n numbers each, and its standard
   error, about sqrt(4/(n - 1)) for normal samples: {:log-ratio :se}."
  [a b]
  (let [variance #(let [s (:sd (spread %))] (* s s))]
    {:log-ratio (Math/log (/ (variance a) (variance b)))
     :se (Math/sqrt (/ 4.0 (dec (count a))))}))

(let [{:keys [log-ratio se]} (log-variance-ratio (map :last sequential-is)
                                                 (map :last filtered))]
  (> log-ratio (* 5 se)))
;=> true
```

## Effective sample size

The *effective sample size* of the weights, $1 / \sum_i w_i^2$ for
normalized weights, is the filter's diagnostic, as it was for importance
sampling: $n$ when the weights are equal, 1 when one particle has them
all. `pf/ess` computes it, and `run-filter` notes it after each
observation. Eight observations are too few to watch a filter fail, so
take a hundred, simulated from the model:

<!-- code: long-data -->
```clojure
(def long-ys
  (let [t (gfi/simulate ssm [100 0.0 walk] (rng/key 40))]
    (mapv #(array/item (t [% :y])) (range 100))))

(def long-exact (exact walk long-ys))

(:log-ml long-exact)
;=> -179.72 ; float64
```

Two filters of 1000 particles run through them, one never resampling, the
other resampling whenever the effective sample size falls below $n/2$:

<!-- code: long-filters -->
```clojure
(def never (run-filter walk long-ys 1000 {:ess-threshold 0 :key (rng/key 6)}))

(def adaptive (run-filter walk long-ys 1000 {:key (rng/key 6)}))

[(:resamples never) (:resamples adaptive)]
;=> [0 35] ; float64

(mapv #(Math/round (double %)) (take-nth 10 (:ess never)))
;=> [614 35 8 3 2 1 2 2 1 1] ; float64
```

<!-- code: long-ess-figure -->
```clojure
;; figure: figures/026-ess.svg
(svg/plot [{:curves [(through (mapv #(/ % 1000.0) (:ess adaptive)))]
            :color "darkorange"}
           {:curves [(through (mapv #(/ % 1000.0) (:ess never)))]}]
          {:x-range [0 99] :y-range [0 1]
           :x-label "step t" :y-label "ESS / N"})
```

![The effective sample size after each of 100 observations, as a fraction
of 1000 particles: never resampling (blue), and resampling below $n/2$
(orange)](figures/026-ess.svg)

Without resampling, every observation multiplies in a factor that favors
some particles over others, and the effective sample size slides from 614
to a handful within twenty steps, and to one particle by the end. The
estimate of the evidence then rests on that one particle: it is 46 nats
below the exact value. Worse, the error bar computed from the weights,
$\sqrt{1/\mathrm{ESS} - 1/n}$, claims ±1:

<!-- code: long-never -->
```clojure
(peek (:ess never))
;=> 1.0 ; float64

[(pf/log-ml-estimate never) (log-ml-se (peek (:ess never)) 1000)]
;=> [-225.8 0.9995] ; float64

(< (pf/log-ml-estimate never) (- (:log-ml long-exact) 20))
;=> true
```

The filter that resamples did so 35 times: each time the effective sample
size fell below $n/2$, the next step started again from equal weights. Ten
runs of it estimate the evidence within five standard errors of the exact
value:

<!-- code: long-adaptive -->
```clojure
(def adaptive-runs
  (mapv #(pf/log-ml-estimate (run-filter walk long-ys 1000 {:key %}))
        (rng/split (rng/key 7) 10)))

(spread adaptive-runs)
;=> {:mean -179.66 :sd 0.235 :se 0.074} ; float64

(let [{:keys [mean se]} (spread adaptive-runs)]
  (< (abs (- mean (:log-ml long-exact))) (* 5 se)))
;=> true
```

This is SMC's bargain. Resampling turns uneven weights into equal ones, so
the filter can run through any number of observations without its weights
degenerating. The bargain has a price, which the next three sections
measure.

## The price of resampling

Resampling is itself random: it replaces a weighted population with an
unweighted one drawn from it, and the draw adds noise. Take the filter's
first step, 100 particles weighted by the first observation, and its
estimate of $z_0$'s mean, the weighted mean of the particles:

<!-- code: price -->
```clojure
(def population (bootstrap-init walk ys 100 (rng/key 8)))

(def z0s (mapv #(array/item (% [0 :z])) (pf/traces population)))

(def probabilities
  (mapv #(Math/exp %)
        (:log-weights (weights/normalize (pf/log-weights population)))))

(def weighted-mean (reduce + (map * probabilities z0s)))

(def weighted-variance
  (reduce + (map (fn [p z] (* p (- z weighted-mean) (- z weighted-mean)))
                 probabilities z0s)))

[weighted-mean weighted-variance (pf/ess population)]
;=> [0.3556 0.4669 79.7] ; float64
```

Resampling and averaging the copies gives a second estimate.
*Multinomial* resampling, gen.clj's default as Gen.jl's, draws the $n$
copies independently, each particle $i$ with probability $w_i$. The mean of
the copies is then the mean of $n$ independent draws from the weighted
particles: unbiased, with variance $\hat\sigma^2 / n$, where $\hat\sigma^2$
is the particles' weighted variance. Four thousand resamplings of the same
population show it, beside *systematic* resampling, which the next section
describes:

<!-- code: price-resampled -->
```clojure
(defn resampled-means
  "The plain mean of z0s over each of k resamplings of the population, by
   method."
  [method k key]
  (let [lws (pf/log-weights population)]
    (mapv (fn [k] (/ (reduce + (map z0s (weights/resample method lws 100 k)))
                     100.0))
          (rng/split key k))))

(defn variance-of
  "The sample variance of xs and its standard error, from their fourth
   moment: {:variance :se}."
  [xs]
  (let [n (count xs)
        {:keys [mean sd]} (spread xs)
        v (* sd sd)
        m4 (/ (reduce + (map #(Math/pow (- % mean) 4) xs)) n)]
    {:variance v :se (Math/sqrt (/ (- m4 (* v v)) n))}))

(def multinomial-means (resampled-means :multinomial 4000 (rng/key 9)))

(def systematic-means (resampled-means :systematic 4000 (rng/key 9)))
```

Both are unbiased, multinomial's variance is $\hat\sigma^2/n$ within five
standard errors, and systematic adds less, here about half as much:

<!-- code: price-check -->
```clojure
(mapv (fn [means]
        (let [{:keys [mean se]} (spread means)]
          (< (abs (- mean weighted-mean)) (* 5 se))))
      [multinomial-means systematic-means])
;=> [true true]

(let [{:keys [variance se]} (variance-of multinomial-means)]
  (< (abs (- variance (/ weighted-variance 100))) (* 5 se)))
;=> true

(let [{:keys [variance se]} (variance-of systematic-means)]
  (< (+ variance (* 5 se)) (/ weighted-variance 100)))
;=> true
```

The noise multinomial resampling adds, 0.0047, is of the size of the
weighted mean's own variance, which is about $\hat\sigma^2 / \mathrm{ESS}
= 0.0059$ with an effective sample size of 80. So resampling when the weights are
nearly even buys nothing and costs this. A filter should resample only when
the weights need it, which is what the threshold on the effective sample
size does, and in a way that adds little noise.

## Low-variance resampling

The noise comes from how many copies each particle gets. Particle $i$
deserves $e_i = n w_i$ copies. Multinomial resampling draws each copy
independently, so a particle that deserves three may get none, or six.
The *low-variance* schemes in `gen.inference.weights` draw the copies
jointly. On the scale of the running sums of the $e_i$, from 0 to $n$,
particle $i$ owns an interval of length $e_i$, and a copy goes to the
owner of each of $n$ positions:

- *systematic*: the positions $j + u$, $j = 0, \dots, n - 1$, for one
  uniform $u$;
- *stratified*: the positions $j + u_j$, one independent uniform in each
  stratum $[j, j + 1)$;
- *residual*: $\lfloor e_i \rfloor$ copies of each particle first, and the
  rest drawn multinomially, in proportion to the fractional parts.

To compare them, take a weight vector where a few particles dominate,
$w_i \propto e^{-0.3 i}$ for 20 particles, and measure how far each
particle's copies fall from what it deserves:

<!-- code: offspring -->
```clojure
(def geometric
  "The log weights of 20 particles, w_i proportional to exp(-0.3 i): a few
   dominate."
  (mapv #(* -0.3 %) (range 20)))

(def expected
  "Each particle's expected number of copies, N p_i."
  (mapv #(* 20 (Math/exp %)) (:log-weights (weights/normalize geometric))))

(defn count-error
  "The mean squared difference between the copies each particle gets from
   the resampling indices idx and its expected copies."
  [idx]
  (let [copies (frequencies idx)
        sq (fn [i e] (let [d (- (get copies i 0) e)] (* d d)))]
    (/ (reduce + (map-indexed sq expected)) 20)))
```

Each scheme's expected squared error has a closed form, the variance of
each particle's copies averaged over the particles:

<!-- code: offspring-theory -->
```clojure
(defn frac
  "The fractional part of x."
  [x]
  (- x (Math/floor x)))

(def exact-count-error
  "Each scheme's expected count-error, the mean over the particles of the
   variance of their copies. With e_i = N p_i and f_i its fractional part:
   - multinomial, Binomial(N, p_i): e_i (1 - p_i);
   - systematic, floor or ceil of e_i: f_i (1 - f_i);
   - stratified, one draw in each stratum [j, j+1) of the scale of the
     running sums of e, which falls in particle i's interval with the
     length a_ij of their overlap: Σ_j a_ij (1 - a_ij);
   - residual, floor(e_i) copies and R = Σ f draws in proportion to the
     f_i: f_i (1 - f_i / R)."
  (let [n 20
        f (mapv frac expected)
        total-f (reduce + f)
        edges (vec (reductions + 0.0 expected))
        overlap (fn [i j] (max 0.0 (- (min (edges (inc i)) (inc j))
                                      (max (edges i) j))))
        average (fn [g] (/ (reduce + (map g (range n))) n))]
    {:multinomial (average #(* (expected %) (- 1 (/ (expected %) n))))
     :systematic (average #(* (f %) (- 1 (f %))))
     :stratified (average (fn [i] (reduce + (map #(let [a (overlap i %)]
                                                     (* a (- 1 a)))
                                                  (range n)))))
     :residual (average #(* (f %) (- 1 (/ (f %) total-f))))}))

exact-count-error
;=> {:multinomial 0.85 :systematic 0.126 :stratified 0.173 :residual 0.271}
```

Two thousand resamplings by each scheme agree with all four, within five
standard errors:

<!-- code: offspring-check -->
```clojure
(def schemes [:multinomial :stratified :residual :systematic])

(def measured-count-error
  (into {} (for [s schemes]
             [s (spread (mapv #(count-error (weights/resample s geometric 20 %))
                              (rng/split (rng/key 10) 2000)))])))

(mapv (fn [s]
        (let [{:keys [mean se]} (measured-count-error s)]
          (< (abs (- mean (exact-count-error s))) (* 5 se))))
      schemes)
;=> [true true true true]
```

On these weights systematic resampling's error is about a seventh of
multinomial's, and the other two fall between. Systematic resampling beats
stratified with less randomness, one uniform against $n$, and it does so
for every particle. Under the stratified scheme a particle's interval
covers some strata whole, each a certain copy, and overlaps at most two
others in part, by lengths $a$ and $b$, each a coin flip with variance
$a(1 - a)$ and $b(1 - b)$. Under the systematic scheme the positions move
together, so the particle gets $\lfloor e_i \rfloor$ or $\lceil e_i
\rceil$ copies, with variance $f(1 - f)$ for $f$ the fractional part of
$e_i$; since $a + b$ is $f$ or $1 + f$, that is never more than the
stratified variance.

gen.clj's default is multinomial, as Gen.jl's is; GenMLX defaulted to
systematic. `pf/maybe-resample` and `pf/resample` take `:method` (and
`run-filter` passes it on): `:multinomial`, `:systematic`, `:stratified`,
`:residual`, or `:optimal`, Fearnhead and Clifford's, which keeps each
particle at most once and leaves the weights unequal. Exercise 1 shows
that systematic resampling's noise also depends on the particles' order.

## Path degeneracy and rejuvenation

Resampling copies a particle's whole trace, its past states with its
present one. After many resamplings every particle descends from a few
ancestors, and the early states have few distinct values among the
particles: *path degeneracy*. Two hundred particles through the hundred
observations:

<!-- code: degeneracy -->
```clojure
(defn distinct-values
  "How many distinct values of z_t the particles hold, for each of the
   first n-steps t."
  [state n-steps]
  (let [trs (pf/traces state)]
    (mapv (fn [t] (count (distinct (map #(array/item (% [t :z])) trs))))
          (range n-steps))))

(def long-200 (run-filter walk long-ys 200 {:key (rng/key 11)}))

(def long-distinct (distinct-values long-200 100))

[(first long-distinct) (peek long-distinct)]
;=> [2 200] ; float64

(< (first long-distinct) 10)
;=> true
```

The last state has 200 distinct values, since each particle drew it
afresh, but $z_0$ has two. The figure below shows the collapse at every
step. (In gen.clj each particle draws its next state from its own key, so
the copies that resampling makes go separate ways at once. In GenMLX the
copies shared their random state and drew the same next state, so every
state, early or late, had the same number of distinct values.)

The remedy is *rejuvenation*, the "move" of resample-move SMC: after
resampling, move each particle with an MCMC kernel that leaves the
posterior given the observations so far invariant. Since the kernel leaves
the target alone, the weights stay as they are, and the moves pull the
copies apart. `pf/rejuvenate` takes a kernel from `gen.inference.kernel`.
Here it is an MH move on each past state in turn, proposed from its prior:

<!-- code: rejuvenation -->
```clojure
(defn sweep
  "An MH move on each state z_0 … z_t-1 in turn, from its prior: a kernel
   that leaves the posterior given the observations so far invariant."
  [t]
  (apply kernel/chain (for [s (range t)] (kernel/mh (sel/select [s :z])))))

(defn resample-move
  "A bootstrap step, after a sweep that rejuvenates the particles if they
   were just resampled: run-filter's :advance."
  [state params ys t key]
  (let [[k-move k-step] (rng/split key)
        moved (if (:resampled? state)
                (pf/rejuvenate state (sweep t) k-move)
                state)]
    (bootstrap-step moved params ys t k-step)))
```

On the eight observations, resampling at every step to make the collapse
worse, one sweep after each resampling multiplies the distinct values of
the early states several times over:

<!-- code: rejuvenation-run -->
```clojure
(def always (run-filter walk ys 200 {:ess-threshold ##Inf :key (rng/key 12)}))

(def moved
  (run-filter walk ys 200 {:ess-threshold ##Inf :advance resample-move
                           :key (rng/key 12)}))

[(distinct-values always 8) (distinct-values moved 8)]
;=> [[25 28 34 48 57 77 114 200] [148 167 177 183 176 182 187 200]] ; float64

(> (first (distinct-values moved 1)) (* 3 (first (distinct-values always 1))))
;=> true
```

A sweep at step $t$ makes $t$ moves for each particle, so a run of $T$
steps makes up to about $T^2/2$ of them, and `pf/rejuvenate` moves one
particle at a time, leaving a batched filter unbatched. On the hundred
observations the same sweep is affordable batched: `mh/mh-particles` makes
one MH move for every particle at once.

<!-- code: batched-rejuvenation -->
```clojure
(defn batched-sweep
  "The particles of a batched state after an MH move on each z_s, s < t,
   from its prior: mh/mh-particles moves every particle at once."
  [state t key]
  (reduce (fn [state [s k]]
            (update state :particles
                    #(:particles (mh/mh-particles % (sel/select [s :z]) k))))
          state
          (map vector (range t) (rng/split key (max 1 t)))))

(defn batched-resample-move
  "resample-move, batched: run-filter's :advance."
  [state params ys t key]
  (let [[k-move k-step] (rng/split key)
        moved (if (:resampled? state) (batched-sweep state t k-move) state)]
    (bootstrap-step moved params ys t k-step)))

(def long-moved
  (run-filter walk long-ys 200 {:advance batched-resample-move
                                :key (rng/key 11)}))

(first (distinct-values long-moved 1))
;=> 117 ; float64

(> (first (distinct-values long-moved 1)) 50)
;=> true
```

<!-- code: ancestry-figure -->
```clojure
;; figure: figures/026-ancestry.svg
(svg/plot [{:curves [(through (distinct-values long-moved 100))]
            :color "darkorange"}
           {:curves [(through long-distinct)]}]
          {:x-range [0 99] :y-range [0 200]
           :x-label "step t" :y-label "distinct values of z_t"})
```

![The distinct values of each state $z_t$ among 200 particles after a
hundred observations, without rejuvenation (blue) and with a batched MH
sweep after each resampling (orange)](figures/026-ancestry.svg)

What the diversity buys shows against the exact *smoothing* distribution of
$z_0$, its distribution given all the observations. The joint normal of the
states and the observations gives it:

<!-- code: smoothed -->
```clojure
(defn smoothed
  "The exact mean and standard deviation of each z_t given all of ys, for
   the random walk: with K the states' covariance and S = K + r^2 I the
   observations', the mean is K S^-1 ys and the covariance K - K S^-1 K."
  [{:keys [q r]} ys]
  (let [n (count ys)
        K (walk-covariance n q)
        gain (array/matmul K (linalg/inv (+ K (* r r (array/eye n)))))]
    {:mean (array/->clj (array/matmul gain (array/array ys)))
     :sd (mapv #(Math/sqrt %)
               (array/->clj (array/diag (- K (array/matmul gain K)))))}))

(defn first-state
  "The particles' weighted mean and standard deviation of z_0."
  [state]
  {:mean (array/item (pf/mean state #(% [0 :z])))
   :sd (Math/sqrt (array/item (pf/variance state #(% [0 :z]))))})

(let [{:keys [mean sd]} (smoothed walk long-ys)]
  {:mean (first mean) :sd (first sd)})
;=> {:mean 0.7013 :sd 0.5299}

[(first-state long-200) (first-state long-moved)]
;=> [{:mean 0.945 :sd 0.109} {:mean 0.737 :sd 0.491}] ; float64
```

Without rejuvenation the filter's answer about $z_0$ rests on two values,
and its spread is a fifth of the exact one. With rejuvenation its mean and
spread are near the exact ones. More particles delay path degeneracy but
do not stop it: exercise 3 tries ten times as many.

## Better proposals: SMCP3

The bootstrap filter proposes each new state from its prior, without
looking at the observation it is about to be weighed by. When the
observations are precise, most proposals miss them. Take a sharper sensor,
$r = 0.05$, and twenty observations from it:

<!-- code: sharp -->
```clojure
(def sharp {:a 1.0 :q 0.5 :r 0.05})

(def sharp-ys
  (let [t (gfi/simulate ssm [20 0.0 sharp] (rng/key 20))]
    (mapv #(array/item (t [% :y])) (range 20))))

(def sharp-exact (:log-ml (exact sharp sharp-ys)))

(def sharp-bootstrap (run-filter sharp sharp-ys 100 {:key (rng/key 21)}))

[(:resamples sharp-bootstrap)
 (mapv #(Math/round (double %)) (:ess sharp-bootstrap))]
;=> [19 [6 2 19 21 15 15 9 5 18 10 7 5 15 13 3 14 9 12 11 12]] ; float64
```

At every step the 100 particles are worth 21 at most, often a handful,
and the filter resamples at every step. The proposal that looks at the
observation is the *locally optimal* one, $p(z_t \mid z_{t-1}, y_t)$,
which for this model is normal, with the prior's precision plus the
observation's:

<!-- code: optimal -->
```clojure
(defn optimal
  "The mean and standard deviation of the locally optimal proposal,
   p(z_t | z_t-1, y_t), a normal: its precision is the transition's plus
   the observation's, 1/q^2 + 1/r^2 (1 + 1/r^2 at t = 0, from the prior)."
  [{:keys [a q r]} t prev y]
  (let [prior-precision (if (zero? t) 1.0 (/ 1.0 (* q q)))
        precision (+ prior-precision (/ 1.0 (* r r)))]
    [(/ (+ (* prior-precision a prev) (/ y (* r r))) precision)
     (Math/sqrt (/ 1.0 precision))]))

(def first-proposal
  (gen [params y]
    (let [[m s] (optimal params 0 0.0 y)]
      (trace [0 :z] (d/normal m s)))))

(defn optimal-init
  "n particles with z_0 from the optimal proposal: run-filter's :init."
  [params ys n key]
  (pf/initialize ssm [1 0.0 params] (observation ys 0) n
                 {:key key :proposal first-proposal
                  :proposal-args [params (ys 0)]}))
```

*SMCP3* (sequential Monte Carlo with probabilistic program proposals, Lew
et al. 2023) makes a step out of three parts: a *forward kernel* that
proposes auxiliary choices $u$, a *transform* that maps the trace and $u$
to the new trace's choices and to backward auxiliary choices $u'$, and a
*backward kernel* that scores $u'$. The step's weight is

$$
w = w_{\text{update}} + \log q_{\text{bwd}}(u') - \log q_{\text{fwd}}(u)
  + \log \lvert \det J \rvert ,
$$

the importance weight that keeps the filter on its target however the
proposal works. In gen.clj an SMCP3 step is an *updating translator*
(`gen.inference.transform`), and `pf/move-reweight` applies it to every
particle. Here the forward kernel draws $u \sim \mathcal{N}(0, 1)$, the
transform sets $z_t = m + s u$ from the optimal proposal's mean $m$ and
standard deviation $s$, and there are no backward choices, since the step
only adds one:

<!-- code: smcp3 -->
```clojure
(def standard-normal (gen [_] (trace :u (d/normal 0 1))))

(defn reparameterized
  "The transform of SMCP3's step t: z_t = m + s·u from the auxiliary
   choice u, m and s the optimal proposal's given z_t-1."
  [params t y]
  (tf/transform [model-in aux-in model-out aux-out]
    (let [[m s] (optimal params t (tf/read model-in [(dec t) :z] :continuous)
                         y)]
      (tf/write model-out [t :z] (+ m (* s (tf/read aux-in :u :continuous)))
                :continuous))))

(defn smcp3-step
  "An SMCP3 step to t: the trace updated to t + 1 steps with observation
   t, its new choice z_t from the transform of u ~ N(0, 1)."
  [params ys t]
  (tf/updating-translator
   {:p-new-args [(inc t) 0.0 params] :p-argdiffs argdiffs
    :new-observations (observation ys t)
    :q-forward standard-normal
    :transform (reparameterized params t (ys t))}))

(defn smcp3-advance
  "Every particle moved by SMCP3's step t: run-filter's :advance."
  [state params ys t key]
  (pf/move-reweight state (smcp3-step params ys t) key))
```

The pieces combine as the formula says. The update weight is the change in
the trace's score, $\log p(z_t \mid z_{t-1}) + \log p(y_t \mid z_t)$; the
forward kernel's density of $u$ and the Jacobian $s$ make $\log
\mathcal{N}(z_t; m, s)$, the optimal proposal's density; and what remains
is the observation's density with $z_t$ summed out, $p(y_t \mid z_{t-1})$.
The new state drops out of the weight. So does $z_0$ at the first step,
where every particle gets the same weight, $p(y_0)$:

<!-- code: smcp3-weights -->
```clojure
(defn predictive
  "log p(y_t | z_t-1): the observation's density with z_t summed out,
   normal with variance q^2 + r^2 around a·z_t-1."
  [{:keys [a q r]} prev y]
  (dc/log-prob (d/normal (* a prev) (Math/sqrt (+ (* q q) (* r r)))) y))

(let [{:keys [r]} sharp
      start (optimal-init sharp sharp-ys 5 (rng/key 22))
      stepped (smcp3-advance start sharp sharp-ys 1 (rng/key 23))
      first-density (dc/log-prob (d/normal 0.0 (Math/sqrt (+ 1.0 (* r r))))
                                 (sharp-ys 0))
      increments (map - (pf/log-weights stepped) (pf/log-weights start))]
  [(every? #(array/allclose? % first-density) (pf/log-weights start))
   (every? true? (map #(array/allclose? %1 (predictive sharp (%2 [0 :z])
                                                       (sharp-ys 1)))
                      increments (pf/traces start)))])
;=> [true true]
```

The weights then vary only with where the particles were, and the filter
keeps nearly all of them:

<!-- code: smcp3-run -->
```clojure
(def sharp-smcp3
  (run-filter sharp sharp-ys 100 {:init optimal-init :advance smcp3-advance
                                  :key (rng/key 21)}))

[(:resamples sharp-smcp3)
 (mapv #(Math/round (double %)) (:ess sharp-smcp3))]
;=> [0 [100 97 97 97 97 97 96 95 95 95 93 90 90 90 90 90 89 89 88 88]] ; float64
```

It never resampled, and its effective sample size stayed at 88 of 100 or
more. Over forty runs of each filter, SMCP3's estimate of the evidence is
unbiased, and it spreads seventy times less than the bootstrap filter's:

<!-- code: smcp3-runs -->
```clojure
(defn evidence-runs
  "The log marginal likelihood estimates of filters from runs keys."
  [params ys n runs key opts]
  (mapv #(pf/log-ml-estimate (run-filter params ys n (assoc opts :key %)))
        (rng/split key runs)))

(def bootstrap-evidence (evidence-runs sharp sharp-ys 100 40 (rng/key 24) {}))

(def smcp3-evidence
  (evidence-runs sharp sharp-ys 100 40 (rng/key 24)
                 {:init optimal-init :advance smcp3-advance}))

[(:sd (spread bootstrap-evidence)) (:sd (spread smcp3-evidence))]
;=> [2.4 0.0345] ; float64

(- (:mean (spread bootstrap-evidence)) sharp-exact)
;=> -2.25 ; float64

(let [ratio (fn [ls] (spread (mapv #(Math/exp (- % sharp-exact)) ls)))
      {:keys [mean se]} (ratio smcp3-evidence)]
  [(< (abs (- mean 1.0)) (* 5 se))
   (< (* 10 (:sd (spread smcp3-evidence))) (:sd (spread bootstrap-evidence)))])
;=> [true true]
```

The bootstrap filter's estimates average 2.3 nats below the exact value in
log space: an unbiased estimate of $p(y)$ with a large variance is far
below it most of the time. For a step that only adds a choice, `pf/step`
takes a custom proposal (`:proposal`) and weighs it the same way. SMCP3's
steps can also revise earlier choices, with a backward kernel to weigh the
way back; Part III covers trace translators in depth.

## Particle MCMC: PMMH and particle Gibbs

Every filter so far knew the model's parameters. Suppose instead that the
coefficient $a$, here called $\phi$, is unknown, with prior
$\mathcal{N}(0.8, 0.3)$, and must be inferred with the states. The GenMLX
chapter's data and noise levels:

<!-- code: ar -->
```clojure
(def ar-ys [0.2 0.5 0.9 0.7 0.4 0.1 -0.2 -0.5 -0.3 0.0])

(defn ar-params
  "The model's parameters for the coefficient phi."
  [phi]
  {:a phi :q 0.3 :r 0.5})

(def phi-prior (d/normal 0.8 0.3))
```

For each fixed $\phi$ the model is linear and Gaussian, so the Kalman
filter gives the likelihood $p(y \mid \phi)$ exactly, and on a fine grid
the prior times it, normalized, is the exact posterior:

<!-- code: ar-exact -->
```clojure
(def grid
  "Values of phi from -1 to 2.6, six prior standard deviations each side."
  (mapv #(+ -1.0 (* 0.01 %)) (range 361)))

(def grid-posterior
  "p(phi | ys) at each point of the grid, normalized over it: the prior
   times the Kalman filter's likelihood."
  (let [log-joint #(+ (dc/log-prob phi-prior %)
                      (:log-ml (exact (ar-params %) ar-ys)))]
    (mapv #(Math/exp %)
          (:log-weights (weights/normalize (mapv log-joint grid))))))

(def phi-mean (reduce + (map * grid grid-posterior)))

(def phi-sd
  (Math/sqrt (reduce + (map #(* %2 (- %1 phi-mean) (- %1 phi-mean))
                            grid grid-posterior))))

[phi-mean phi-sd]
;=> [0.7092 0.2216]
```

*Particle-marginal Metropolis–Hastings* (PMMH; Andrieu, Doucet and
Holenstein 2010) runs MH on $\phi$, with the likelihood estimated by a
particle filter. Because the filter's estimate of $p(y \mid \phi)$ is
unbiased, the chain targets the exact posterior: the noise in the estimate
slows the chain but does not bias it. gen.clj needs nothing of its own for
PMMH. `gen.custom/estimated` makes a generative function whose density is
an estimate, here the filter's, and a model calls it like any other:

<!-- code: pmmh-model -->
```clojure
(defn filter-log-ml
  "A bootstrap filter's estimate of log p(ys | phi), from n particles."
  [phi ys n key]
  (pf/log-ml-estimate (run-filter (ar-params phi) ys n {:key key})))

(def filtered-likelihood
  "The density of the observations given phi, at addresses 0 … 9,
   estimated by a filter of 20 particles."
  (custom/estimated
   {:addresses (vec (range (count ar-ys)))
    :estimate (fn [[phi] choices key]
                (let [ys (mapv #(cm/get-choice choices [%])
                               (range (count ar-ys)))]
                  (filter-log-ml phi ys 20 key)))}))

(def pmmh-model
  (gen []
    (let [phi (trace :phi phi-prior)]
      (splice :ys filtered-likelihood phi)
      phi)))

(def pmmh-observations
  (apply cm/choicemap (mapcat (fn [t y] [[:ys t] y]) (range) ar-ys)))
```

When MH proposes a new $\phi$, the estimated function's `update` draws a
fresh estimate for it, and its weight is the ratio of the new estimate to
the one the trace holds, the pseudo-marginal ratio. The current trace keeps
its estimate until a proposal replaces it, which is what makes the chain
exact. A chain helper keeps only the values of $\phi$, host numbers:

<!-- code: chains -->
```clojure
(def phi-walk (gen [tr] (trace :phi (d/normal (tr :phi) 0.18))))

(defn chain-values
  "The values of phi along a chain of n steps of the kernel, from a trace
   of model on args generated with the observations: host numbers, each
   trace dropped as the chain moves on."
  [kernel model args observations n key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate model args observations k-start))]
    (second
     (reduce (fn [[tr values] k]
               (let [tr' (:trace (kernel tr k))]
                 [tr' (conj values (array/item (tr' :phi)))]))
             [start []]
             (rng/split k-chain n)))))

(defn moved-fraction
  "The fraction of a chain's steps that changed its value."
  [values]
  (/ (count (filter true? (map not= values (rest values))))
     (double (dec (count values)))))
```

Four chains of a thousand steps, with a random-walk proposal:

<!-- code: pmmh -->
```clojure
(def pmmh-chains
  (mapv #(chain-values (kernel/mh phi-walk []) pmmh-model [] pmmh-observations
                       1000 %)
        (rng/split (rng/key 30) 4)))

(def pmmh-summary (diag/summary (mapv #(subvec % 100) pmmh-chains)))

(select-keys pmmh-summary [:mean :mcse-mean :ess-bulk])
;=> {:mean 0.725 :mcse-mean 0.0129 :ess-bulk 272.6} ; float64

[(< (abs (- (:mean pmmh-summary) phi-mean)) (* 5 (:mcse-mean pmmh-summary)))
 (< (:r-hat pmmh-summary) 1.1)]
;=> [true true]

(mapv moved-fraction pmmh-chains)
;=> [0.623 0.58 0.619 0.608] ; float64
```

The four chains agree with one another: $\hat R$, which compares the
spread within chains to the spread between them, is below 1.1, the usual
threshold. Their mean is within five Monte Carlo standard errors of the
exact 0.709, and their histogram follows the exact posterior:

<!-- code: pmmh-figure -->
```clojure
;; figure: figures/026-pmmh.svg
(let [density (through (mapv #(/ % 0.01) grid-posterior))]
  (svg/histogram [{:values (mapcat #(subvec % 100) pmmh-chains)
                   :density #(density (/ (+ % 1.0) 0.01))}]
                 {:range [-0.2 1.6] :bins 36 :x-label "phi"}))
```

![PMMH's draws of $\phi$, with the exact posterior density from the Kalman
likelihood on a grid](figures/026-pmmh.svg)

*Particle Gibbs* attacks the same problem from the other side. It
alternates a move of the states given $\phi$ with a move of $\phi$ given
the states. The state move is *conditional SMC*: a particle filter in
which one particle, the *reference*, is the current trace's path, kept at
index 0 through every resampling. Drawing a particle from the final weights
is then a valid Gibbs move on the whole path (Andrieu et al. 2010). The
model now has the states as choices:

<!-- code: csmc -->
```clojure
(def latent-model
  (gen [n]
    (let [phi (trace :phi phi-prior)]
      (splice :zs ssm n 0.0 (ar-params phi))
      phi)))

(def latent-observations
  (apply cm/choicemap (mapcat (fn [t y] [[:zs t :y] y]) (range) ar-ys)))

(def states (sel/from-paths (for [t (range (count ar-ys))] [:zs t :z])))

(def pg-steps (mapv (fn [t] {:args [(inc t)]}) (range (count ar-ys))))

(let [reference (:trace (gfi/generate latent-model [(count ar-ys)]
                                      latent-observations (rng/key 31)))
      state (pmcmc/conditional-smc reference states pg-steps 10
                                   {:key (rng/key 32)})
      kept (first (pf/traces state))
      path (fn [tr] (mapv #(array/item (tr [:zs % :z])) (range 10)))]
  (= (path reference) (path kept)))
;=> true
```

`pmcmc/particle-gibbs` is conditional SMC as a kernel, which composes with
an MH move on $\phi$ in `kernel/chain`. Its steps are the model's
arguments at each step of the filter, $n = 1, \dots, 10$; the choices the
trace holds beyond the states, $\phi$ and the observations, are constrained
in every particle:

<!-- code: particle-gibbs -->
```clojure
(def particle-gibbs
  (kernel/chain (pmcmc/particle-gibbs states pg-steps 10)
                (kernel/mh phi-walk [])))

(def pg-chains
  (mapv #(chain-values particle-gibbs latent-model [(count ar-ys)]
                       latent-observations 500 %)
        (rng/split (rng/key 33) 4)))

(def pg-summary (diag/summary (mapv #(subvec % 50) pg-chains)))

(select-keys pg-summary [:mean :mcse-mean :ess-bulk])
;=> {:mean 0.6997 :mcse-mean 0.0195 :ess-bulk 127.5} ; float64

[(< (abs (- (:mean pg-summary) phi-mean)) (* 5 (:mcse-mean pg-summary)))
 (< (:r-hat pg-summary) 1.1)]
;=> [true true]
```

Its chains agree, and their mean is within five Monte Carlo standard
errors of the exact value, too. A check of the mean alone would not be
enough, since chains that do not mix have a large Monte Carlo standard
error, which can swallow a large error. Without the conditional SMC move,
MH on $\phi$ with the states held at their first values cannot leave the
neighborhood of those states:

<!-- code: stuck -->
```clojure
(def stuck-chains
  (mapv #(chain-values (kernel/mh phi-walk []) latent-model [(count ar-ys)]
                       latent-observations 500 %)
        (rng/split (rng/key 33) 4)))

(def stuck-summary (diag/summary (mapv #(subvec % 50) stuck-chains)))

(select-keys stuck-summary [:mean :mcse-mean :r-hat])
;=> {:mean 1.12 :mcse-mean 0.085 :r-hat 1.97} ; float64

(> (:r-hat stuck-summary) 1.1)
;=> true
```

Its mean, 1.12, is 0.41 from the exact 0.709, but its Monte Carlo standard
error is four times the working chains', and the mean lies within five of
them. $\hat R$ near 2 gives it away.

## Choosing an SMC configuration

The remedies combine; a filter usually uses several at once.

| Situation | Reach for |
|-----------|-----------|
| Sequential data, any model | the bootstrap filter: `pf/initialize`, `pf/step`, `pf/maybe-resample` |
| Uneven weights | resampling below an ESS threshold (`:ess-threshold`, $n/2$ by default) |
| Least noise from resampling | `:method :systematic` |
| Early states collapsed to a few ancestors | rejuvenation: `pf/rejuvenate`, or a batched sweep of `mh/mh-particles` |
| Precise observations | a proposal that looks at the data: `pf/step`'s `:proposal`, or an SMCP3 step (`tf/updating-translator` with `pf/move-reweight`) |
| Unknown static parameters | PMMH (MH on `custom/estimated`) or particle Gibbs (`pmcmc/particle-gibbs`) |
| A linear-Gaussian model or a hidden Markov model | the exact answer, `gen.state-space/kalman` or `hmm`, to use and to check the rest |

Through all of it the model stayed the same Unfold: importance sampling,
the bootstrap filter, rejuvenation, SMCP3, PMMH and particle Gibbs ran it
unchanged, through the GFI. The next chapter, *Variational Inference*,
leaves sampling for optimization: it fits a tractable approximation to the
posterior by gradient ascent.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Order matters.** Systematic resampling's positions sweep the
   particles in their order. Sort the population of *The price of
   resampling* by $z_0$, resample it systematically, and predict the
   variance of the resampled mean against the unsorted one's.
2. **How often to resample.** Run the filter of 1000 particles on the
   hundred observations with thresholds 0, 100, 500 and `##Inf`. Predict
   how many times each resamples, and how far each estimate of the
   evidence lands from the exact one.
3. **More particles, or rejuvenation?** Run the hundred observations with
   2000 particles and no rejuvenation. Predict how many distinct values of
   $z_0$ survive.
4. **Brute force against a better proposal.** On the sharp sensor, give
   the bootstrap filter ten times the particles, 1000. Does its estimate
   of the evidence spread as little as SMCP3's with 100?
5. **What the estimate costs PMMH.** Replace the filtered likelihood with
   the exact one, `gen.state-space/kalman`, and run the same chains.
   Predict how often they move, and how many independent draws they are
   worth.

## Solutions

**1.** In order of $z_0$, neighboring positions pick neighboring values,
so the copies' mean differs from the weighted mean only by the rounding at
the ends of a few intervals. Its variance falls more than tenfold:

<!-- code: solution-1 -->
```clojure
(def by-value
  "The population's particles in order of their z_0."
  (vec (sort-by z0s (range 100))))

(defn sorted-systematic-means
  "resampled-means for systematic resampling of the particles in order of
   their z_0."
  [k key]
  (let [lws (mapv (pf/log-weights population) by-value)]
    (mapv (fn [k]
            (let [idx (weights/systematic lws 100 k)]
              (/ (reduce + (map #(z0s (by-value %)) idx)) 100.0)))
          (rng/split key k))))

(def sorted-means (sorted-systematic-means 4000 (rng/key 9)))

(mapv (comp :variance variance-of) [systematic-means sorted-means])
;=> [0.00218 0.000144] ; float64

(let [{:keys [variance se]} (variance-of sorted-means)]
  (< (+ variance (* 5 se)) (/ (:variance (variance-of systematic-means)) 5)))
;=> true
```

**2.** Threshold 0 never resamples and `##Inf` resamples at every one of
the 99 steps; the others resample when they must. Every threshold that
resamples at all lands within a nat of the exact evidence here, and never
resampling lands 54 nats below it:

<!-- code: solution-2 -->
```clojure
(def by-threshold
  (into {} (for [threshold [0 100 500 ##Inf]]
             (let [state (run-filter walk long-ys 1000
                                     {:ess-threshold threshold
                                      :key (rng/key 40)})]
               [threshold
                {:resamples (:resamples state)
                 :error (- (pf/log-ml-estimate state)
                           (:log-ml long-exact))}]))))

(mapv #(:resamples (by-threshold %)) [0 100 500 ##Inf])
;=> [0 13 35 99] ; float64

(mapv #(:error (by-threshold %)) [0 100 500 ##Inf])
;=> [-53.6 -0.15 0.49 0.58] ; float64

[(< (:error (by-threshold 0)) -20)
 (every? #(< (abs (:error (by-threshold %))) 2) [100 500 ##Inf])]
;=> [true true]
```

**3.** Ten times the particles leave 15 distinct values of $z_0$ where 200
particles left 2: still fewer than one particle in a hundred. The
ancestors of $n$ particles coalesce over a number of steps that grows with
$n$, so more particles push the collapse further back, at $n$ times the
cost, but over a long enough series it comes. Rejuvenation fixed it with
200.

<!-- code: solution-3 -->
```clojure
(def long-2000 (run-filter walk long-ys 2000 {:key (rng/key 42)}))

(first (distinct-values long-2000 1))
;=> 15 ; float64

(< (first (distinct-values long-2000 1)) 100)
;=> true
```

**4.** Ten times the particles narrow the bootstrap filter's spread from
2.4 to 0.44, which is still more than ten times SMCP3's with a tenth of
the particles. A proposal that looks at the data does what no number of
blind ones can do cheaply.

<!-- code: solution-4 -->
```clojure
(def brute-evidence
  (evidence-runs sharp sharp-ys 1000 20 (rng/key 41) {}))

[(:sd (spread brute-evidence)) (:sd (spread smcp3-evidence))]
;=> [0.437 0.0345] ; float64

(> (:sd (spread brute-evidence)) (* 5 (:sd (spread smcp3-evidence))))
;=> true
```

**5.** With the exact likelihood the target is the same, so the chains
agree with the exact posterior too. They move more often, since a noisy
estimate sometimes overrates the current $\phi$ and holds the chain there,
and their draws are worth more. PMMH pays for the estimate's noise in
mixing, not in bias.

<!-- code: solution-5 -->
```clojure
(def exact-model
  (gen []
    (let [phi (trace :phi phi-prior)
          [n prior params] (kalman-args (count ar-ys) (ar-params phi))]
      (splice :ys ss/kalman n prior params)
      phi)))

(def exact-chains
  (mapv #(chain-values (kernel/mh phi-walk []) exact-model [] pmmh-observations
                       1000 %)
        (rng/split (rng/key 30) 4)))

(def exact-summary (diag/summary (mapv #(subvec % 100) exact-chains)))

[(< (abs (- (:mean exact-summary) phi-mean)) (* 5 (:mcse-mean exact-summary)))
 (< (:r-hat exact-summary) 1.1)
 (> (apply min (map moved-fraction exact-chains))
    (apply max (map moved-fraction pmmh-chains)))]
;=> [true true true]

[(:ess-bulk exact-summary) (:ess-bulk pmmh-summary)]
;=> [324.5 272.6] ; float64
```
