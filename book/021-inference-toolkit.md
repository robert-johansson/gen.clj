# The Inference Toolkit

Part I built inference by hand. *Conditioning* turned `generate`'s weight
into importance sampling, and *Update and Regenerate* turned
`regenerate`'s weight into Metropolis–Hastings. This chapter puts those
loops away and takes up the library that replaces them: weighted samples
and their resampling (`gen.inference.importance`, `gen.inference.weights`),
MCMC moves and their composition (`gen.inference.kernel`), and the
diagnostics that say whether a chain can be trusted
(`gen.inference.diagnostics`). The library is small because its pieces
share types. An importance sampler returns log weights, and every function
of weights takes them. An MCMC move is a *kernel*, one function type, and
kernels compose into kernels.

This chapter ports the GenMLX book's *The Inference Toolkit*. Its two
models, a line through eight points and a mixture of two groups, are small
enough to solve exactly, so every estimate is checked against the exact
answer, within five of its standard errors. The GenMLX book's section on
its edit interface is left out, since gen.clj has none; its nearest
relative here is `gen.inference.particle-filter/move-reweight`, whose moves
return a weight in place of an accept decision.

<!-- code: ns -->
```clojure
(ns book.inference-toolkit
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.kernel :as k]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## A model with an exact answer

The GenMLX book's running example is a line through eight points, with
the same broad prior on the slope $\mu$ and the intercept $b$:

$$
\mu \sim \mathcal{N}(0, 10), \qquad b \sim \mathcal{N}(0, 10), \qquad
y_i \mid x_i \sim \mathcal{N}(\mu x_i + b,\ 1).
$$

It is *Getting Started*'s model with a broader prior on the slope, and
points near the line $y = 2x + 0.5$:

<!-- code: model -->
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

(defn observations
  "The choice map fixing each [addr i] to (vs i)."
  [addr vs]
  (apply cm/choicemap (mapcat (fn [i v] [[addr i] v]) (range) vs)))
```

The model is linear and Gaussian, so the posterior of $(\mu, b)$ is
normal. For the matrix $X$ of rows $(x_i, 1)$, its precision is
$\Lambda = \Lambda_0 + X^\top X$, where $\Lambda_0 = I/100$ is the
prior's, and its mean is $\Lambda^{-1} X^\top y$:

<!-- code: exact -->
```clojure
(def prior-precision
  "The prior's precision matrix: the inverse of its covariance, 100 I."
  (array/diag (array/array [0.01 0.01])))

(defn design
  "The matrix of rows (x, 1)."
  [xs]
  (array/array (mapv (fn [x] [x 1.0]) xs)))

(defn posterior
  "The exact posterior of the line given the points (xs, ys), a normal of
   [slope intercept]: its precision matrix, mean and covariance."
  [xs ys]
  (let [X (design xs)
        precision (+ prior-precision (array/matmul (array/transpose X) X))]
    {:precision precision
     :mean (linalg/solve precision (array/matmul (array/transpose X)
                                                 (array/array ys)))
     :cov (linalg/inv precision)}))

(def exact (posterior xs ys))

(defn sds
  "The standard deviations of a posterior's slope and intercept."
  [{:keys [cov]}]
  (array/->clj (array/sqrt (array/diag cov))))

[(array/->clj (:mean exact)) (sds exact)]
;=> [[1.9835 0.6233] [0.1539 0.7768]]

(def slope-mean (first (array/->clj (:mean exact))))

(def slope-sd (first (sds exact)))
```

The points narrow the slope from the prior's standard deviation of 10 to
0.154. The marginal likelihood follows from Bayes' rule, as in
*Conditioning*: $\log p(y) = \log p(\mu, b, y) - \log p(\mu, b \mid y)$ at
any line. `assess` gives the first term and the exact posterior the
second, and the same value at two lines checks the posterior as well:

<!-- code: exact-log-ml -->
```clojure
(defn log-ml
  "log p(ys; xs), by Bayes' rule at the line [s b]: the log joint density
   less the exact posterior's log density there."
  [xs ys [s b]]
  (let [{:keys [mean cov]} (posterior xs ys)
        line (cm/choicemap :slope s :intercept b)]
    (- (:weight (gfi/assess line-model [xs]
                            (cm/merge line (observations :y ys))))
       (dc/log-prob (d/mvnormal mean cov) (array/array [s b])))))

(def exact-log-ml (log-ml xs ys [2.0 0.5]))

[exact-log-ml (log-ml xs ys [0.0 0.0])]
;=> [-15.354 -15.354]
```

## Importance sampling

`imp/importance-sampling` runs `generate` with the observations $n$
times, as *Conditioning*'s loop did. Each sample draws the line from the
model's internal proposal, here the prior, and is weighted by how well it
explains the points. With `:batched? true` the samples are one batched run
of the model, its code running once on arrays of every sample's values.
The result holds the log weights, raw and normalized, the log marginal
likelihood estimate, the effective sample size, and the samples as a
particle collection (`:particles`; `imp/traces` makes scalar traces of
them):

<!-- code: importance -->
```clojure
(def n-samples 4000)

(def is-result
  (imp/importance-sampling line-model [xs] (observations :y ys) n-samples
                           {:key (rng/key 1) :batched? true}))

(sort (keys is-result))
;=> (:ess :log-ml-estimate :log-normalized-weights :log-weights :particles)

[(:ess is-result) (:log-ml-estimate is-result)]
;=> [5.80 -16.56] ; float64
```

The 4000 samples are worth fewer than six independent draws from the
posterior. The *effective sample size*, $1/\sum_i w_i^2$ for normalized
weights $w_i$, is the health check of importance sampling, and here it
raises an alarm.

How many should they be worth? As $n$ grows, $\mathrm{ESS}/n$ tends to
$1/\int p^2/q$, for $q$ the proposal and $p$ the posterior: the
*efficiency* of the proposal (Kong 1992). For a normal prior with
precision $\Lambda_0$ and a normal posterior with precision $\Lambda$ and
mean $m$, the integral is Gaussian:

$$
\int \frac{p^2}{q} = \frac{|\Lambda|}{|\Lambda_0|^{1/2}\,
|2\Lambda - \Lambda_0|^{1/2}}
\exp\!\left(m^\top \Lambda\, (2\Lambda - \Lambda_0)^{-1} \Lambda_0\, m\right).
$$

Written this way it needs no difference of two large numbers, which
float32 would round away:

<!-- code: efficiency -->
```clojure
(defn efficiency
  "The fraction of importance samples from the prior that count, as their
   number grows: 1 / (the integral of p^2/q), for q the prior and p the
   exact posterior of the line given the points (xs, ys)."
  [xs ys]
  (let [{P :precision m :mean} (posterior xs ys)
        P0 prior-precision
        A (- (* 2.0 P) P0)
        log-integral (+ (array/log (linalg/det P))
                        (* -0.5 (array/log (linalg/det P0)))
                        (* -0.5 (array/log (linalg/det A)))
                        (array/sum (* (array/matmul P m)
                                      (linalg/solve A (array/matmul P0 m)))))]
    (array/item (array/exp (- log-integral)))))

(def eff (efficiency xs ys))

[eff (* eff n-samples)]
;=> [0.001063 4.25]
```

In the limit, one sample in a thousand counts: 4000 are worth about four.

### More data, fewer samples that count

The prior stays where it is while each observation narrows the
posterior, so fewer of the prior's draws land where the posterior has its
mass. With the first one, two, four and eight points, the efficiency
falls by a factor of four or five with each doubling:

<!-- code: collapse -->
```clojure
(defn first-points
  "The first m points, as [xs ys]."
  [m]
  [(subvec xs 0 m) (subvec ys 0 m)])

(mapv #(apply efficiency (first-points %)) [1 2 4 8])
;=> [0.098 0.0187 0.00432 0.00106]
```

Ten runs of 4000 samples on each of the first three agree with these
limits, within five standard errors of their mean:

<!-- code: collapse-check -->
```clojure
(defn ess-fractions
  "ESS/n of runs of importance sampling with n samples on the first m
   points, one run per key."
  [m n ks]
  (let [[xs' ys'] (first-points m)]
    (mapv #(/ (:ess (imp/importance-sampling line-model [xs']
                                             (observations :y ys') n
                                             {:key % :batched? true}))
              n)
          ks)))

(defn sd-of
  "The sample standard deviation of the numbers vs."
  [vs]
  (let [mean (/ (reduce + vs) (count vs))]
    (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) vs))
                  (dec (count vs))))))

(defn mean-se
  "The mean of the numbers vs, and its standard error."
  [vs]
  {:mean (/ (reduce + vs) (count vs))
   :se (/ (sd-of vs) (Math/sqrt (count vs)))})

(every? (fn [m]
          (let [{:keys [mean se]} (mean-se (ess-fractions m n-samples
                                                          (rng/split
                                                           (rng/key m) 10)))]
            (< (abs (- mean (apply efficiency (first-points m)))) (* 5 se))))
        [1 2 4])
;=> true
```

At eight points the expected ESS is about four, and the estimate is as
uncertain as the few weights it rests on. Ten runs range from about two to
about six:

<!-- code: collapse-noise -->
```clojure
(let [fs (ess-fractions 8 n-samples (rng/split (rng/key 8) 10))]
  [(* n-samples (apply min fs)) (* n-samples (apply max fs))])
;=> [2.15 6.36] ; float64
```

### What the efficiency costs

The efficiency also sets the standard errors of what importance sampling
estimates. The marginal likelihood estimate is the mean of the $n$
weights, each with mean $p(y)$ and relative variance $\int p^2/q - 1$, so
its relative standard error is exactly $((1/\mathrm{eff} - 1)/n)^{1/2}$:
about a half for these 4000 samples. This run's estimate is within five of
those of $p(y)$:

<!-- code: log-ml-se -->
```clojure
(def relative-se (Math/sqrt (/ (- (/ 1.0 eff) 1.0) n-samples)))

relative-se
;=> 0.485

(< (abs (- (Math/exp (- (:log-ml-estimate is-result) exact-log-ml)) 1.0))
   (* 5 relative-se))
;=> true
```

Its log, −16.56, is below the exact −15.35: the estimate of $p(y)$ is
unbiased, but its log is biased low (*Conditioning*'s first exercise).

A posterior mean is estimated by the weighted average of a function $f$
over the samples. `imp/estimate` computes it without keeping a trace: it
applies $f$, which returns a host number, to each sample as it is drawn.
A rule of thumb puts its standard error at $\mathrm{sd}_p(f) /
(n \cdot \mathrm{eff})^{1/2}$, as if the samples were $n \cdot \mathrm{eff}$
independent draws from the posterior:

<!-- code: estimate -->
```clojure
(def slope-estimate
  (imp/estimate #(array/item (% :slope)) line-model [xs] (observations :y ys)
                n-samples {:key (rng/key 2)}))

slope-estimate
;=> {:estimate 2.048 :log-ml-estimate -15.566 :ess 5.65} ; float64

(def rule-se (/ slope-sd (Math/sqrt (* eff n-samples))))

[rule-se (< (abs (- (:estimate slope-estimate) slope-mean)) (* 5 rule-se))]
;=> [0.0747 true]
```

Forty batched runs test both standard errors. A batched run's samples
are one particle collection, and `batched/choice` reads a choice across
all of them as one array, so the weighted mean of the slope needs no
trace at all. The forty estimates of the slope center on the exact mean,
and their spread agrees with the rule, within five standard errors of
each. Their estimates of $p(y)$, averaged, are within five standard
errors of the exact value, where one run's estimate could be off by half:

<!-- code: estimate-spread -->
```clojure
(defn weighted-mean
  "The weighted mean of the choice at addr over the samples of a batched
   importance-sampling result: batched/choice reads it across them."
  [result addr]
  (array/item (array/sum (* (array/exp (array/array
                                        (:log-normalized-weights result)))
                            (batched/choice (:particles result) addr)))))

(def runs
  "Forty batched runs: each one's weighted mean of the slope and its
   estimate of p(y) over the exact p(y)."
  (mapv #(let [r (imp/importance-sampling line-model [xs] (observations :y ys)
                                          n-samples {:key % :batched? true})]
           {:slope (weighted-mean r :slope)
            :ratio (Math/exp (- (:log-ml-estimate r) exact-log-ml))})
        (rng/split (rng/key 13) 40)))

(def slope-means (mapv :slope runs))

(sd-of slope-means)
;=> 0.0569 ; float64

(let [{:keys [mean se]} (mean-se slope-means)
      sd (sd-of slope-means)]
  [(< (abs (- mean slope-mean)) (* 5 se))
   (< (abs (- sd rule-se)) (* 5 (/ sd (Math/sqrt (* 2 39)))))])
;=> [true true]

(< (abs (- (:mean (mean-se (mapv :ratio runs))) 1.0))
   (* 5 (/ relative-se (Math/sqrt 40))))
;=> true

(Math/round (/ (Math/pow (/ slope-sd 0.01) 2) eff))
;=> 222914
```

The last line is the cost of precision: to bring the slope's standard
error down to 0.01, this proposal needs $(0.154/0.01)^2/\mathrm{eff}$,
about 223 000 samples. Importance sampling is only as good as its
proposal, and the prior is a poor one for a posterior whose slope is
sixty times narrower.

## Weights and resampling

`gen.inference.weights` holds the functions of log weights that every
sampler shares: normalization, the effective sample size, the log of the
mean weight, and resampling. Four samples with weights proportional to
10, 6, 3 and 1 show them at a scale we can check by hand:

<!-- code: weights -->
```clojure
(def toy-log-weights (mapv #(Math/log %) [10.0 6.0 3.0 1.0]))

(mapv #(Math/exp %) (:log-normalized-weights (w/normalize toy-log-weights)))
;=> [0.5 0.3 0.15 0.05]

[(w/ess toy-log-weights) (w/log-mean-exp toy-log-weights) (Math/log 5.0)]
;=> [2.7397 1.6094 1.6094]
```

The ESS is $1/(0.5^2 + 0.3^2 + 0.15^2 + 0.05^2) = 2.74$, and the log of
the mean weight is $\log(20/4)$.

*Resampling* turns weighted samples into unweighted ones: it draws $n$
indices, each sample's in proportion to its weight. `w/resample` has four
schemes. *Multinomial* resampling draws the $n$ indices independently.
*Systematic* resampling places $n$ evenly spaced positions $(j + u)/n$
with one uniform $u$; *stratified* resampling draws one uniform in each of
the $n$ strata; *residual* resampling keeps $\lfloor n p_i \rfloor$ copies
of each sample and draws the rest multinomially. One draw of ten indices
by each, as counts of the four samples:

<!-- code: resample -->
```clojure
(def schemes [:multinomial :systematic :stratified :residual])

(defn counts
  "How many times each of the four samples is among the indices."
  [indices]
  (mapv (fn [i] (count (filter #{i} indices))) (range 4)))

(mapv #(counts (w/resample % toy-log-weights 10 (rng/key 9))) schemes)
;=> [[5 5 0 0] [5 3 1 1] [5 3 1 1] [5 3 1 1]] ; float64
```

The expected counts are $n p_i$ = 5, 3, 1.5 and 0.5. Systematic resampling
always gives one of the two whole numbers nearest, over 2000 keys:

<!-- code: systematic -->
```clojure
(def expected-counts [5.0 3.0 1.5 0.5])

(def repeats
  "Each method's counts for 2000 keys."
  (into {} (map (fn [m]
                  [m (mapv #(counts (w/resample m toy-log-weights 10 %))
                           (rng/split (rng/key 10) 2000))]))
        schemes))

(every? (fn [cs]
          (every? true? (map (fn [c e] (<= (Math/floor e) c (Math/ceil e)))
                             cs expected-counts)))
        (repeats :systematic))
;=> true
```

All four schemes are unbiased: over the 2000 keys, each sample's mean
count is within five standard errors of $n p_i$:

<!-- code: unbiased -->
```clojure
(defn unbiased?
  "Is each sample's mean count, over the repeats, within five standard
   errors of n times its probability?"
  [cs]
  (every? (fn [i]
            (let [{:keys [mean se]} (mean-se (mapv #(% i) cs))]
              (<= (abs (- mean (expected-counts i))) (* 5 se))))
          (range 4)))

(mapv #(unbiased? (repeats %)) schemes)
;=> [true true true true]
```

They differ in their variance. The third sample's count varies with a
standard deviation of $\sqrt{10 \cdot 0.15 \cdot 0.85} = 1.13$ under
multinomial resampling, and of 0.5 under the other three, which only
ever choose between 1 and 2:

<!-- code: spread -->
```clojure
(def spreads
  "Each method's standard deviation of the third sample's count."
  (mapv (fn [m] (sd-of (mapv #(% 2) (repeats m)))) schemes))

spreads
;=> [1.11 0.5 0.5 0.5] ; float64

(let [[multinomial & others] spreads]
  (every? #(< (* 2 %) multinomial) others))
;=> true

(every? (fn [m] (every? #(#{1 2} (% 2)) (repeats m))) (rest schemes))
;=> true
```

Resampling adds no diversity that the weights lack. Drawing 500 indices
from the 4000 weighted samples of the line, sample $i$ appears at least
once with probability $1 - (1 - p_i)^{500}$, so the expected number of
distinct samples is the sum of these. The indicators that each sample
appears are negatively associated, so the sum of their variances bounds
the variance of the count:

<!-- code: degeneracy -->
```clojure
(def resampled
  (w/resample :multinomial (:log-normalized-weights is-result) 500
              (rng/key 11)))

(defn expected-distinct
  "The expected number of distinct indices among m multinomial draws, and
   a bound on its standard error: index i is drawn at least once with
   probability 1 - (1 - p_i)^m."
  [log-weights m]
  (let [hit (mapv #(- 1.0 (Math/pow (- 1.0 (Math/exp %)) m)) log-weights)]
    {:mean (reduce + hit)
     :se (Math/sqrt (reduce + (map #(* % (- 1.0 %)) hit)))}))

(def distinct-500
  (expected-distinct (:log-normalized-weights is-result) 500))

[(count (distinct resampled)) (:mean distinct-500)]
;=> [17 15.08] ; float64

(< (abs (- (count (distinct resampled)) (:mean distinct-500)))
   (* 5 (:se distinct-500)))
;=> true
```

Of the 500 draws, all but 17 are copies. The weights put almost all their
mass on a few samples, as the ESS of 5.8 said, and resampling copies those
few.

`imp/importance-resampling` is Gen.jl's `importance_resampling`: it draws
$n$ samples and returns one trace, chosen in proportion to the weights,
with the log marginal likelihood estimate. One sample at a time, it keeps
only the trace it will return; batched, it draws the index from one
batched run:

<!-- code: sir -->
```clojure
(def one-draw
  (imp/importance-resampling line-model [xs] (observations :y ys) 1000
                             {:key (rng/key 12) :batched? true}))

(sort (keys one-draw))
;=> (:log-ml-estimate :trace)
```

Its trace is a draw from the posterior only as $n$ grows. Two hundred
draws, each from its own $n$ samples, show how the draws narrow toward the
posterior, whose standard deviation is 0.154. Their mean agrees with the
exact mean already at 1000 samples:

<!-- code: sir-draws -->
```clojure
(defn sir-slopes
  "The slopes of m importance-resampling draws, each from n particles."
  [n m key]
  (mapv #(array/item ((:trace (imp/importance-resampling
                               line-model [xs] (observations :y ys) n
                               {:key % :batched? true}))
                      :slope))
        (rng/split key m)))

(def sir (into {} (map (fn [n] [n (sir-slopes n 200 (rng/key n))]))
               [10 100 1000]))

(mapv #(sd-of (sir %)) [10 100 1000])
;=> [2.28 0.652 0.241] ; float64

(let [[a b c] (mapv #(sd-of (sir %)) [10 100 1000])]
  (> a b c slope-sd))
;=> true

(let [{:keys [mean se]} (mean-se (sir 1000))]
  (< (abs (- mean slope-mean)) (* 5 se)))
;=> true
```

<!-- code: sir-figure -->
```clojure
;; figure: figures/021-sir.svg
(let [exact-slope (d/normal slope-mean slope-sd)]
  (svg/histogram [{:values (sir 100) :label "100 samples"}
                  {:values (sir 1000) :label "1000 samples"
                   :density #(array/exp (dc/log-prob exact-slope %))}]
                 {:range [0 4] :bins 40 :x-label "slope"}))
```

![The slopes of 200 importance-resampling draws, each from 100 or 1000
samples, and the exact posterior density of the
slope](figures/021-sir.svg)

At 1000 samples each run's ESS is about one, and its draw is still wider
than the posterior. The proposal, not the resampler, is what limits
importance sampling here. Markov chains do better: they do not guess the
line from the prior, but move it a little at a time.

## Kernels: one type

A *kernel* is a function of a trace and a key that returns a map with a
new trace, `(kernel trace key)` → `{:trace …}`: one step of a Markov chain
that leaves the posterior unchanged, so that a draw from the posterior
goes in and a draw from the posterior comes out. It is Gen.jl's kernel,
which returns a trace and metadata, as a map. A trace from `generate`
starts a chain. This one starts far from the posterior:

<!-- code: start -->
```clojure
(def start
  (:trace (gfi/generate line-model [xs] (observations :y ys) (rng/key 3))))

[(start :slope) (start :intercept)]
;=> [-2.474 -11.267] ; float64
```

`k/mh` makes a Metropolis–Hastings kernel. With a selection it proposes
the selected choices from the internal proposal (`regenerate`), as
*Update and Regenerate*'s `mh-step` did. Its result adds the accept
decision and the log acceptance ratio:

<!-- code: kernel -->
```clojure
(def slope-move (k/mh (sel/select :slope)))

(def moved (slope-move start (rng/key 4)))

(sort (keys moved))
;=> (:accepted? :log-alpha :trace)
```

With a proposal generative function and its arguments, `k/mh` proposes
from it. The proposal is called on the trace and the arguments, and the
move is weighed both ways (*Getting Started*'s third exercise). A random
walk proposes a choice near its current value:

<!-- code: random-walk -->
```clojure
(def walk
  (gen [tr addr step]
    (trace addr (d/normal (tr addr) step))))

(defn random-walk
  "The MH kernel that proposes the choice at addr from a normal around its
   current value, of standard deviation step."
  [addr step]
  (k/mh walk [addr step]))

(sort (keys ((random-walk :slope 0.3) start (rng/key 4))))
;=> (:accepted? :log-alpha :trace)
```

A kernel that moves only the slope keeps the intercept where it is, so
its chain samples the slope's *conditional* posterior given the start's
intercept, not the posterior. For a fixed intercept $b$, the slope's
conditional is normal with precision $1/100 + \sum_i x_i^2$ and mean
$\sum_i x_i (y_i - b)$ over the precision:

<!-- code: conditional -->
```clojure
(defn slope-given
  "The exact conditional of the slope given the intercept b and the points:
   a normal whose precision is the prior's 1/100 plus the sum of x^2."
  [b]
  (let [precision (+ 0.01 (reduce + (map * xs xs)))]
    (d/normal (/ (reduce + (map (fn [x y] (* x (- y b))) xs ys)) precision)
              (Math/sqrt (/ 1.0 precision)))))

(def conditional (slope-given (array/item (start :intercept))))

(:params conditional)
;=> {:mu 4.0817 :std 0.07001} ; float64
```

Two kernels on the slope, one from the prior and one a random walk, run
20 100 steps each from the start. `k/run` collects a chain, here the
slope's draws as host numbers (`:keep`); its options come below. The
prior proposal is accepted once in a hundred steps, the random walk about
once in four:

<!-- code: conditional-chains -->
```clojure
(defn slope-draws
  "The slope's draws from a chain of the kernel from start (20100 steps,
   the first 100 dropped, then every tenth kept), with the acceptance rate."
  [kernel key]
  (k/run kernel start
         {:samples 2000 :burn 100 :thin 10 :key key :keep #(% :slope)}))

(def by-prior (slope-draws slope-move (rng/key 5)))

(def by-walk (slope-draws (random-walk :slope 0.3) (rng/key 5)))

(mapv :acceptance-rate [by-prior by-walk])
;=> [0.0100 0.273] ; float64
```

Both chains agree with the exact conditional mean, 4.08, and neither with
the posterior mean of 1.98. They differ in what their draws are worth: the
random walk's 2000 kept draws count as about 1800 independent ones, the
prior's as about 100.

<!-- code: conditional-check -->
```clojure
(defn near?
  "Is the mean of the draws within five Monte Carlo standard errors of mu?"
  [draws mu]
  (let [{:keys [mean mcse-mean]} (diag/summary draws)]
    (< (abs (- mean mu)) (* 5 mcse-mean))))

(mapv (fn [{:keys [draws]}]
        [(near? draws (:mu (:params conditional)))
         (near? draws slope-mean)])
      [by-prior by-walk])
;=> [[true false] [true false]]

(mapv #(diag/ess-bulk (:draws %)) [by-prior by-walk])
;=> [106.6 1838.6] ; float64
```

The prior proposal fails for the reason importance sampling did. It
proposes a slope from $\mathcal{N}(0, 10)$, where the conditional has a
standard deviation of 0.07, so nearly every proposal lands where the
points are improbable and is rejected. A random walk proposes near a slope
that already fits, and is accepted often.

## Composing kernels

The combinators of `gen.inference.kernel` take kernels and return a
kernel. `k/chain` applies its kernels in turn, `k/repeat` one kernel $n$
times, and `k/cycle` its kernels in a round, $n$ steps in all. Each step
gets its own key, split from the one given. The combinators never look
inside a trace, so a kernel on vectors that appends a label shows what
they do:

<!-- code: toy -->
```clojure
(defn recorder
  "A kernel on vectors that appends label, its reversal the label primed:
   a stand-in for a move, to show what the combinators do."
  [label]
  (let [primed (keyword (str (name label) "'"))]
    (k/kernel (fn [v _] {:trace (conj v label)})
              (fn [v _] {:trace (conj v primed)}))))

(def a (recorder :a))

(def b (recorder :b))

(def c (recorder :c))

(defn applied
  "What the kernel makes from the empty vector."
  [kernel]
  (:trace (kernel [] (rng/key 0))))

(mapv applied [(k/chain a b c) (k/repeat 3 a) (k/cycle 5 [a b c])])
;=> [[:a :b :c] [:a :a :a] [:a :b :c :a :b]]
```

A kernel may declare its *reversal*, the kernel that undoes it in
distribution (Gen.jl's `reversal`). An MH kernel is its own reversal. The
reversal of a chain is the chain of the reversals, in the reverse order,
and a combinator has one only when each of its kernels does:

<!-- code: reversal -->
```clojure
(mapv applied [(k/reversal (k/chain a b c)) (k/reversal (k/cycle 5 [a b c]))])
;=> [[:c' :b' :a'] [:b' :a' :c' :b' :a']]

(k/reversible? (k/chain a (fn [v _] {:trace v})))
;=> false
```

`k/mix` picks one of its kernels at random for each step, with
probabilities proportional to its weights:

<!-- code: mix -->
```clojure
(def mixed (k/mix [a b] [0.7 0.3]))

(def picks
  (frequencies (map #(first (:trace (mixed [] %)))
                    (rng/split (rng/key 6) 2000))))

picks
;=> {:a 1435 :b 565} ; float64

(< (abs (- (/ (picks :a) 2000.0) 0.7)) (* 5 (Math/sqrt (/ (* 0.7 0.3) 2000))))
;=> true
```

On the line, a chain of two random walks moves the slope and then the
intercept, and so samples the posterior itself. A combinator's result
reports the accept decision of each MH move it made, in order, as
`:accepted`:

<!-- code: block -->
```clojure
(def block (k/chain (random-walk :slope 0.1) (random-walk :intercept 0.5)))

(:accepted (block start (rng/key 14)))
;=> [true false] ; float64
```

Any function of a trace and a key that returns `{:trace …}` is a kernel
(one that returns the trace alone is not, and a combinator says so
loudly), so the library's other moves compose with these.
`gen.inference.mcmc`'s Hamiltonian Monte Carlo, the subject of
*Gradient-Based MCMC*, returns the same map, with whether its trajectory
diverged, and `k/hmc` makes it a kernel, its own reversal, as `k/mh` does
for MH (`k/mala`, `k/nuts`, `k/elliptical-slice` and `k/map-optimize`
do for the rest). Chained after the random walks, it still targets the
posterior:

<!-- code: hmc -->
```clojure
(def hmc-move (k/hmc (sel/select :slope :intercept) {:L 10 :eps 0.1}))

(sort (keys (hmc-move start (rng/key 15))))
;=> (:accepted? :diverged? :log-alpha :trace)

(def mixed-run
  (k/run (k/chain block hmc-move) start
         {:samples 500 :burn 100 :key (rng/key 16)}))

(near? (diag/draws (:traces mixed-run) :slope) slope-mean)
;=> true
```

A kernel must leave the observations alone, since a move that changed one
would condition on other data. `k/checked` wraps a kernel and checks, after
each step, that the trace still holds the observations, as Gen.jl's
`check=true` does. Selecting an observed address is the classic mistake:

<!-- code: checked -->
```clojure
;; error: The kernel changed the observation at [:y 0]
(def careless (k/checked (k/mh (sel/select [:y 0])) (observations :y ys)))
(careless start (rng/key 8))
```

## Running a chain

`k/run` applies a kernel again and again from a trace. Its options are
`:samples`, the number of traces to keep; `:burn`, the steps dropped first
(0 by default); `:thin`, the steps per kept trace (1); `:key`; and
`:keep`, below. It returns the kept traces, the acceptance rate and the
number of diverged proposals. The same key makes the same chain, draw for
draw:

<!-- code: run -->
```clojure
(def run-opts {:samples 1000 :burn 1000 :thin 5 :key (rng/key 9)})

(def line-run (k/run block start run-opts))

(sort (keys line-run))
;=> (:acceptance-rate :divergences :traces)

[(count (:traces line-run)) (:acceptance-rate line-run)]
;=> [1000 0.601] ; float64

(= (diag/draws (:traces line-run) :slope)
   (diag/draws (:traces (k/run block start run-opts)) :slope))
;=> true
```

A kernel on numbers shows the bookkeeping. `k/run` takes `:burn` +
`:samples` · `:thin` steps and keeps every `:thin`-th after the burn-in.
The rate is over every MH move of every step, here three moves a step,
one of them accepted. (The GenMLX book's rate counted the steps that
changed the trace, which climbs with every kernel a combinator adds.)

<!-- code: run-steps -->
```clojure
(defn count-up
  "A kernel on numbers: one MH-like move that adds one, always accepted."
  [n _]
  {:trace (inc n) :accepted? true})

(defn stay
  "A kernel on numbers: one MH-like move that is always rejected."
  [n _]
  {:trace n :accepted? false})

(k/run (k/chain count-up stay stay) 0
       {:samples 4 :burn 2 :thin 3 :key (rng/key 0)})
;=> {:traces [5 8 11 14] :acceptance-rate 0.3333 :divergences 0}
```

After 1000 steps of burn-in, the chain's draws agree with the exact
posterior, its mean and standard deviation within five Monte Carlo
standard errors for the slope and for the intercept:

<!-- code: run-check -->
```clojure
(defn agrees?
  "Do the draws' mean and standard deviation agree with mu and sd, each
   within five of its Monte Carlo standard errors?"
  [draws mu sd]
  (let [s (diag/summary draws)]
    [(< (abs (- (:mean s) mu)) (* 5 (:mcse-mean s)))
     (< (abs (- (:sd s) sd)) (* 5 (:mcse-sd s)))]))

(let [[_ intercept-mean] (array/->clj (:mean exact))
      [_ intercept-sd] (sds exact)]
  [(agrees? (diag/draws (:traces line-run) :slope) slope-mean slope-sd)
   (agrees? (diag/draws (:traces line-run) :intercept)
            intercept-mean intercept-sd)])
;=> [[true true] [true true]]
```

The burn-in matters. From the start's slope of −2.47, the chain takes
some two hundred steps to reach the posterior. The mean of its first
hundred slopes is more than ten posterior standard deviations below the
posterior's mean, and draws from those steps would pull an average toward
the start:

<!-- code: burn-in -->
```clojure
(def early
  (diag/draws (:traces (k/run block start {:samples 300 :key (rng/key 9)}))
              :slope))

(def hundreds (mapv #(:mean (mean-se (subvec early % (+ % 100)))) [0 100 200]))

hundreds
;=> [-0.92 1.63 1.96] ; float64

(< (first hundreds) (- slope-mean (* 10 slope-sd)))
;=> true
```

<!-- code: burn-in-figure -->
```clojure
;; figure: figures/021-burn-in.svg
(svg/series early {:y-label "slope"})
```

![The slope along the first 300 steps of the chain, from a poor
start](figures/021-burn-in.svg)

`k/run` holds every kept trace, and on the MLX backend thousands of
traces held at once exhaust its buffers. With `:keep`, a function of a
trace or a map of such functions, it keeps their values instead, read on
the host, under `:draws` in place of `:traces`, and holds only the current
trace. The code below keeps host numbers: `run-draws` keeps the values at
some addresses, a map from address to draws, each read as `diag/draws`
reads a trace's (a boolean as 0 or 1).

## Diagnostics

`gen.inference.diagnostics` says what a chain's correlated draws are
worth, and whether the chain has reached the posterior. It computes the
measures that Stan and ArviZ report, as Vehtari et al. (2021) define them: the *bulk ESS*, how
many independent draws the chains are worth for the bulk of the
distribution; the *tail ESS*, the same for its 5% and 95% quantiles;
*R-hat*, which compares chains with each other and each chain's halves;
and the Monte Carlo standard errors of the mean and the standard
deviation. Their draws are host numbers, a vector of chains or one chain,
and `diag/draws` reads them from traces. These effective sample sizes are
of correlated draws, not of weighted samples: `w/ess` is the other kind.

The standard check runs several chains from scattered starts. Four starts
from the prior are far apart:

<!-- code: chains -->
```clojure
(defn run-draws
  "The draws at addrs of a chain of the kernel from start, run with opts:
   a map from address to draws, kept as the chain runs, each read as
   diag/draws reads it (a boolean as 0 or 1)."
  [kernel start opts addrs]
  (let [draw (fn [a] #(first (diag/draws [%] a)))
        readers (into {} (map (juxt identity draw)) addrs)]
    (:draws (k/run kernel start (assoc opts :keep readers)))))

(def starts
  (mapv #(:trace (gfi/generate line-model [xs] (observations :y ys) %))
        (rng/split (rng/key 20) 4)))

(mapv #(array/item (% :slope)) starts)
;=> [6.542 19.059 -0.824 -1.294] ; float64

(def four
  (mapv #(run-draws block %1 {:samples 1000 :burn 1000 :thin 20 :key %2}
                    [:slope :intercept])
        starts (rng/split (rng/key 21) 4)))
```

R-hat is near 1 when the chains agree, and Vehtari et al. recommend
using the draws only when it is below 1.01. Here it is, for both choices,
and the four chains' 4000 draws are worth about 1500 independent ones:

<!-- code: r-hat -->
```clojure
(def slope-chains (mapv :slope four))

(def intercept-chains (mapv :intercept four))

(mapv (juxt diag/r-hat diag/ess-bulk) [slope-chains intercept-chains])
;=> [[1.0022 1538.3] [1.0014 1563.7]] ; float64

(every? #(< (diag/r-hat %) 1.01) [slope-chains intercept-chains])
;=> true
```

`diag/summary` gives every measure at once, with quantiles. The pooled
draws agree with the exact posterior's mean and standard deviation:

<!-- code: summary -->
```clojure
(diag/summary slope-chains)

(agrees? slope-chains slope-mean slope-sd)
;=> [true true]
```

The tail ESS is the ESS of the indicator of a draw below a quantile, so it
gives the standard error of a tail probability. The fraction of draws
below the exact 5% quantile, $m - 1.6449\,\sigma$, is within five such
standard errors of 0.05:

<!-- code: tail -->
```clojure
(let [q05 (- slope-mean (* 1.6449 slope-sd))
      pooled (apply concat slope-chains)
      below (/ (count (filter #(< % q05) pooled)) (double (count pooled)))
      se (Math/sqrt (/ (* 0.05 0.95) (diag/ess-tail slope-chains)))]
  (< (abs (- below 0.05)) (* 5 se)))
;=> true
```

A chain that has not mixed fails these checks. Random walks with steps of
0.01 and 0.05 accept most proposals and move almost nowhere: in 1000
steps, the four chains have not met. R-hat is far above 1.01, and the
4000 draws are worth about four:

<!-- code: stuck -->
```clojure
(def small-steps
  (k/chain (random-walk :slope 0.01) (random-walk :intercept 0.05)))

(def stuck
  (mapv #(:slope (run-draws small-steps %1 {:samples 1000 :key %2} [:slope]))
        starts (rng/split (rng/key 22) 4)))

[(diag/r-hat stuck) (diag/ess-bulk stuck)]
;=> [3.67 4.4] ; float64

(def stuck-rate
  (:acceptance-rate
   (k/run small-steps (first starts) {:samples 1000 :key (rng/key 23)})))

stuck-rate
;=> 0.796 ; float64

[(> (diag/r-hat stuck) 1.1)
 (< (diag/ess-bulk stuck) 20)
 (> stuck-rate (:acceptance-rate line-run))]
;=> [true true true]
```

<!-- code: stuck-figure -->
```clojure
;; figure: figures/021-stuck.svg
(defn path
  "A chain's draws as a function of the iteration, for svg/plot's curves."
  [draws]
  (fn [x] (nth draws (min (dec (count draws)) (long x)))))

(svg/plot (mapv (fn [draws color] {:curves [(path draws)] :color color})
                stuck ["steelblue" "darkorange" "seagreen" "firebrick"])
          {:x-range [0 999] :x-label "iteration" :y-label "slope"})
```

![The slope along four chains with small steps, from the four
starts](figures/021-stuck.svg)

A high acceptance rate is no sign of a good chain: these chains accept
four moves in five, more than the chains that found the posterior, which
accepted three in five.

## Gibbs moves on discrete choices

A *Gibbs move* draws a choice from its exact conditional given every other
choice. For a choice with a finite support, gen.clj computes the
conditional by enumeration. The second model has one: eight readings, each
from one of two groups, around $-2$ or around 2, with an unknown share
$p$ of readings in the group around 2:

$$
p \sim \mathrm{Beta}(2, 2), \qquad z_i \sim \mathrm{Bernoulli}(p), \qquad
y_i \mid z_i \sim \mathcal{N}(4 z_i - 2,\ 1),
$$

with $z_i$ read as 1 for true and 0 for false.

`array/where` picks the group's center, so the model's arithmetic stays
gen.array's and the model runs batched, every sample's code at once:

<!-- code: groups -->
```clojure
(def groups
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (dotimes [i n]
        (let [z (trace [:z i] (d/bernoulli p))]
          (trace [:y i] (d/normal (array/where z 2.0 -2.0) 1))))
      p)))

(def readings [-2.3 1.8 2.6 -0.4 0.3 2.2 -2.9 1.5])

(def n-readings (count readings))

(batched/batched? (:particles (imp/importance-sampling
                               groups [n-readings] (observations :y readings)
                               100 {:key (rng/key 36) :batched? true})))
;=> true
```

The exact posterior sums over the $2^8 = 256$ assignments $z$ of the
readings to the groups. Given $z$ with $h$ readings in the group around 2,
$p$'s posterior is $\mathrm{Beta}(2 + h, 2 + 8 - h)$, so Bayes' rule at
$p = 1/2$ gives $\log p(z, y)$ with $p$ summed out, as it gave the line's
marginal likelihood. `enum/choice-vol-grid`, Gen.jl's `choice_vol_grid`,
makes the assignments:

<!-- code: groups-exact -->
```clojure
(def assignments
  "Every assignment of the readings to the groups, as choice maps of the
   [:z i]: Gen.jl's choice_vol_grid, 2^8 points."
  (mapv first (apply enum/choice-vol-grid
                     (for [i (range n-readings)] [[:z i] [false true]]))))

(defn in-group-one
  "How many readings the assignment z puts in the group around 2."
  [z]
  (count (filter true? (cm/values z))))

(defn log-joint
  "log p(z, readings) with the bias p summed out, by Bayes' rule at p = 1/2:
   given h readings in the group around 2, p's posterior is
   beta(2 + h, 2 + n - h)."
  [z]
  (let [h (in-group-one z)
        choices (cm/merge (cm/choicemap :p 0.5) z
                          (observations :y readings))]
    (- (:weight (gfi/assess groups [n-readings] choices))
       (dc/log-prob (d/beta (+ 2 h) (+ 2 (- n-readings h))) 0.5))))

(def posterior-z (w/normalize (mapv log-joint assignments)))

(defn exact-mean
  "The exact posterior mean of (f z) over the assignments z."
  [f]
  (reduce + (map (fn [z lw] (* (Math/exp lw) (f z)))
                 assignments (:log-normalized-weights posterior-z))))

(defn group-one?
  "1 when the assignment z puts reading i in the group around 2, else 0."
  [z i]
  (if (array/truthy? (cm/get-choice z [:z i])) 1.0 0.0))

(def exact-z (mapv (fn [i] (exact-mean #(group-one? % i))) (range n-readings)))

(def exact-p (exact-mean #(/ (+ 2.0 (in-group-one %)) (+ 4.0 n-readings))))

exact-z
;=> [0.0002 0.9994 1.0 0.2485 0.8126 0.9999 0.0 0.9979]

exact-p
;=> 0.5882
```

Reading 3 ($-0.4$) is probably in the group around $-2$, and reading 4
(0.3) probably in the group around 2, though neither is certain; the
others are all but certain. The posterior mean of $p$ is 0.588.

`enum/with-exact-proposal` keeps a generative function's distribution and
traces, and changes its internal proposal: the choices at the addresses
given are drawn from their exact conditional, by enumerating their
supports. Regenerating such a choice is then a Gibbs move, and `k/mh` with
its selection is a Gibbs kernel. Its acceptance ratio is 1, the log 0, so
it is always accepted:

<!-- code: gibbs -->
```clojure
(def gibbs-groups
  (enum/with-exact-proposal groups (mapv #(vector :z %) (range n-readings))))

(def group-start
  (:trace (gfi/generate gibbs-groups [n-readings] (observations :y readings)
                        (rng/key 30))))

(def gibbs-4 (k/mh (sel/select [:z 4])))

(select-keys (gibbs-4 group-start (rng/key 31)) [:accepted? :log-alpha])
;=> {:accepted? true :log-alpha 0.0}
```

The move draws reading 4's group from its conditional given the rest of
the trace: $p\,\mathcal{N}(0.3; 2, 1)$ against $(1 - p)\,\mathcal{N}(0.3;
-2, 1)$, with the start's $p$. A thousand moves agree with it within five
standard errors:

<!-- code: gibbs-conditional -->
```clojure
(defn z-given
  "The exact conditional probability that reading i is in the group around
   2, given the bias p: p N(y; 2, 1) against (1 - p) N(y; -2, 1)."
  [i p]
  (let [y (readings i)
        one (* p (array/exp (dc/log-prob (d/normal 2.0 1) y)))
        other (* (- 1.0 p) (array/exp (dc/log-prob (d/normal -2.0 1) y)))]
    (array/item (/ one (+ one other)))))

(def start-p (array/item (group-start :p)))

[start-p (z-given 4 start-p)]
;=> [0.4117 0.699] ; float64

(let [n 1000
      ones (count (filter #((:trace (gibbs-4 group-start %)) [:z 4])
                          (rng/split (rng/key 32) n)))
      q (z-given 4 start-p)]
  (< (abs (- (/ ones (double n)) q)) (* 5 (Math/sqrt (/ (* q (- 1 q)) n)))))
;=> true
```

`k/kern` is Gen.jl's `@kern`: it writes a kernel as statements, loops and
branches, here a Gibbs move for each reading and then an MH move for $p$.
Its loops are over the arguments, never the trace, so the result is a
chain, and its reversal is automatic:

<!-- code: sweep -->
```clojure
(def gibbs-sweep
  (k/kern [n]
    (doseq [i (range n)]
      (k/mh (sel/select [:z i])))
    (k/mh (sel/select :p))))

(def sweep (gibbs-sweep n-readings))

(count (:accepted (sweep group-start (rng/key 33))))
;=> 9

(k/reversible? sweep)
;=> true
```

The sweep samples the exact posterior: the share $p$ and the groups of the
two uncertain readings agree with their exact means, within five Monte
Carlo standard errors:

<!-- code: sweep-check -->
```clojure
(def group-run
  (run-draws sweep group-start {:samples 1000 :burn 100 :key (rng/key 34)}
             [:p [:z 3] [:z 4]]))

[(near? (:p group-run) exact-p)
 (near? (group-run [:z 3]) (exact-z 3))
 (near? (group-run [:z 4]) (exact-z 4))]
;=> [true true true]
```

Of the sweep's nine moves, the eight Gibbs moves are always accepted: a
chain of them alone has an acceptance rate of exactly 1.

<!-- code: gibbs-only -->
```clojure
(def z-moves
  (apply k/chain (for [i (range n-readings)] (k/mh (sel/select [:z i])))))

(:acceptance-rate (k/run z-moves group-start {:samples 50 :key (rng/key 35)}))
;=> 1.0
```

## Choosing an algorithm

The toolkit's pieces suggest a first choice for each kind of problem.
Later chapters take up the rest.

| Situation | In gen.clj | Where |
|-----------|------------|-------|
| A prior that is a fair proposal | `imp/importance-sampling`, `imp/estimate` | this chapter, *Conditioning* |
| A peaked posterior over a few continuous choices | `k/mh` with random walks, chained | this chapter, *MCMC and MAP* |
| Many correlated continuous choices | HMC, MALA and NUTS (`gen.inference.mcmc`; as kernels, `k/hmc`, `k/mala`, `k/nuts`) | *Gradient-Based MCMC* |
| Discrete choices with small supports | `enum/with-exact-proposal`: Gibbs moves | this chapter, *Debugging with Enumeration* |
| Data that arrive in sequence | `gen.inference.particle-filter` | *Sequential Monte Carlo* |
| The marginal likelihood, to compare models | `:log-ml-estimate` of importance sampling or a particle filter | this chapter, *SMC in Depth* |
| A posterior to draw from fast, many times | `gen.inference.vi` | *Variational Inference* |

What ties the table together is what this chapter began with. Every
method in it is built from the same parts: weights from `generate`, kernels
of one type, and the generative function interface of Part I beneath both.
Importance sampling weighs samples from a proposal; MH proposes a move and
accepts it on its weight; a Gibbs move proposes from the exact conditional,
whose weight is always 1; a particle filter moves a population of traces
through a sequence of observations. The model is the same under all of
them. The next chapter, *MCMC and MAP*, takes MH further, on Gen.jl's
regression with outliers.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Resampling cannot add diversity.** Resample 4000 indices, in place of
   500, from the line's 4000 weighted samples. Predict the number of
   distinct samples among them, multinomially and systematically. (For
   systematic resampling, sample $i$ appears at least once with
   probability $\min(1, n p_i)$.)
2. **The step size.** Run the slope's random walk from the start with
   steps of 0.02 and 2.0, beside the 0.3 above. Predict the acceptance
   rates and which step's draws are worth the most.
3. **A sweep without $p$.** Leave the MH move on $p$ out of the Gibbs
   sweep. Which distribution does the chain sample now, and does the check
   against the exact posterior catch it?
4. **Fooling R-hat.** Run four chains of the slope's random walk alone,
   each with its own key, from the one poor start. Predict R-hat, and
   whether the chains have found the posterior.

## Solutions

**1.** Each draw is a sample with probability its weight, so more draws
find more of the samples with small weights, but the weights of all but
about twenty are negligible. Multinomially, the expected count is
$\sum_i 1 - (1 - p_i)^{4000}$, about 21:

<!-- code: solution-1 -->
```clojure
(def log-p (:log-normalized-weights is-result))

(def distinct-4000 (expected-distinct log-p 4000))

(def multinomial-4000
  (count (distinct (w/resample :multinomial log-p 4000 (rng/key 40)))))

[multinomial-4000 (:mean distinct-4000)]
;=> [20 21.31] ; float64

(< (abs (- multinomial-4000 (:mean distinct-4000))) (* 5 (:se distinct-4000)))
;=> true
```

Systematically, the expected count is $\sum_i \min(1, n p_i)$, about 23,
and at least the 21 samples with $n p_i \ge 1$ appear, each
$\lfloor n p_i \rfloor$ times or more:

<!-- code: solution-1-systematic -->
```clojure
(def systematic-4000
  (count (distinct (w/resample :systematic log-p 4000 (rng/key 41)))))

[systematic-4000
 (reduce + (map #(min 1.0 (* 4000 (Math/exp %))) log-p))
 (count (filter #(>= (* 4000 (Math/exp %)) 1.0) log-p))]
;=> [23 22.81 21] ; float64
```

Either way, 4000 draws hold about twenty distinct samples, against 17
among 500: resampling chooses among the samples it has, and more draws
only copy the same few more often.

**2.** A small step is accepted almost always, but each move is short, so
the chain crawls; a large step is rarely accepted, so the chain sits
still. Both are correct and both are slow. The step of 0.3, near four
times the conditional's standard deviation of 0.07, is worth several times
either:

<!-- code: solution-2 -->
```clojure
(def steps [0.02 0.3 2.0])

(def by-step (mapv #(slope-draws (random-walk :slope %) (rng/key 42)) steps))

(mapv :acceptance-rate by-step)
;=> [0.897 0.277 0.0426] ; float64

(mapv #(diag/ess-bulk (:draws %)) by-step)
;=> [107.3 1824.9 581.5] ; float64

(let [[small middle large] (mapv #(diag/ess-bulk (:draws %)) by-step)]
  [(every? #(near? (:draws %) (:mu (:params conditional))) by-step)
   (> middle (* 2 small))
   (> middle (* 2 large))])
;=> [true true true]
```

**3.** Without the move on $p$, $p$ stays at the start's 0.41. Given $p$,
the readings' groups are independent, and the chain samples each from its
conditional given that $p$: reading 4 is in the group around 2 with
probability 0.699, not the posterior's 0.813. The check catches it:

<!-- code: solution-3 -->
```clojure
(def z-run
  (run-draws z-moves group-start {:samples 1000 :key (rng/key 43)} [[:z 4]]))

[(near? (z-run [:z 4]) (z-given 4 start-p))
 (near? (z-run [:z 4]) (exact-z 4))]
;=> [true false]
```

**4.** R-hat compares the chains with each other and each chain's halves,
and these agree: they all sample the slope's conditional given the
start's intercept. R-hat is below 1.01 and the ESS is high, but the mean
is the conditional's 4.08, not the posterior's 1.98:

<!-- code: solution-4 -->
```clojure
(def slope-only
  (mapv #(:slope (run-draws (random-walk :slope 0.3) start
                            {:samples 1000 :burn 100 :thin 5 :key %}
                            [:slope]))
        (rng/split (rng/key 44) 4)))

[(diag/r-hat slope-only) (diag/ess-bulk slope-only)]
;=> [1.0006 2969.8] ; float64

[(< (diag/r-hat slope-only) 1.01)
 (near? slope-only slope-mean)
 (near? slope-only (:mu (:params conditional)))]
;=> [true false true]
```

R-hat can say that chains disagree, never that they found the posterior.
Starts scattered over the prior, as above, and a kernel that moves every
choice are what give it the chance to see a problem.
