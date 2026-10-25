# Exact Structure, Made Explicit

Some parts of a model can be integrated by hand. A normal prior on a mean
observed through normal noise has a normal posterior and a closed-form
evidence; a chain of linear-Gaussian states sums out with the Kalman
filter; a chain of discrete states sums out with the forward algorithm.
Where such structure is used, an estimate loses the Monte Carlo error of
the dimensions it integrates: the evidence comes out exact, and the
posterior is drawn exactly. This chapter shows how gen.clj uses it, and
measures what it buys against inference that ignores it.

The chapter ports the GenMLX book's *Analytical Elimination*, reframed.
GenMLX found conjugate structure by itself: an analyzer read the model's
source and replaced sampling with closed forms inside `generate`. gen.clj
finds nothing by itself. The user says where the structure is, with
`gen.inference.conjugate/with-conjugate-proposal` for a conjugate latent
and with `gen.state-space`'s Kalman and hidden Markov generative functions
for chains, and each answer is checked here against one computed without
them: a joint Gaussian, an integral, or an enumeration of every path.

<!-- code: ns -->
```clojure
(ns book.exact-structure
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.conjugate :as conj]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.weights :as weights]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]
            [gen.trace :as trace]))
```

## Found, or declared

GenMLX's analyzer worked on the schema its compiler extracted from a
`gen` body's source. For each pair of trace sites where one's value was the
other's parameter, it looked the pair of distributions up in a table of
conjugate families and, on a match, wired in a handler that integrated the
prior out: the weight of `generate` became $\log p(\text{data})$ in closed
form, and the latent's value in the trace became its posterior mean. Its
chapter presented this as a warning turned into a feature, and promised
that "nothing you ever write can be made *wrong* by L3 -- at worst it is
not made faster."

gen.clj's design gives that up on purpose, for three reasons in DESIGN.

- **It was where the bugs were.** The survey of GenMLX found that "~25k LOC
  are optimization paths that re-implement the same semantics (L1/L2
  compiled paths, L3 analytical elimination, …). That is where most of
  GenMLX's ~150 core bug fixes came from" (§2). Six of those were silent
  wrong marginals of the analyzer's (§9 names them: 94qc, yy8u, rmy7,
  5ytq, jmk4, su6q), models for which the promise above did not hold.
- **A trace must stay a sample.** The first principle: "Every
  optimization is a plug-in that must be observationally equivalent to the
  handler" (§3.1). Its corollary: "no 'generate returns the posterior
  mean with a marginal weight'. Conjugacy is either the thesis's
  *optimal internal proposal* (sample the latent from its exact
  conditional, with weight log p̄(σ); thesis §4.1, pp. 145–146) or an
  explicitly separate marginalized generative function." A posterior mean is not a draw from the
  posterior, and an MH move or a particle filter that reads the trace
  afterwards would treat it as one.
- **The user knows.** §9's entry on conjugacy begins: "Chosen by the user,
  never detected (§3.1's corollary)."

The running example is the GenMLX chapter's: a mean $\mu$ with a normal
prior, and six observations of it through unit noise,

$$
\mu \sim \mathcal{N}(0, 3), \qquad y_i \mid \mu \sim \mathcal{N}(\mu, 1).
$$

GenMLX's analyzer needed static addresses, so its model wrote out `:y0`
to `:y5`; a loop of addresses switched detection off. Here the loop is
fine, since nothing reads the source:

<!-- code: model -->
```clojure
(def ys [1.8 2.2 1.5 2.5 2.0 1.9])

(def noisy-mean
  (gen [n]
    (let [mu (trace :mu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(def obs (observations ys))
```

The observations are jointly normal: each has variance $9 + 1$, and any
two share the prior's 9. So the evidence has a closed form that owes
nothing to the posterior, the density of $\mathcal{N}(0, 9J + I)$ at the
data, $J$ the matrix of ones. The function takes the noise's standard
deviation too, for a later section:

<!-- code: evidence -->
```clojure
(defn log-evidence
  "log p(ys) for noisy-mean with the noise's standard deviation s (1 by
   default): the ys are jointly normal, with mean 0 and covariance
   9J + s^2 I (J all ones), the prior's variance shared by every pair."
  ([ys] (log-evidence ys 1.0))
  ([ys s]
   (let [n (count ys)]
     (dc/log-prob (d/mvnormal (array/zeros [n])
                              (+ (* 9.0 (array/ones [n n]))
                                 (* s s (array/eye n))))
                  (array/array ys)))))

(log-evidence ys)
;=> -8.02603
```

It is the GenMLX chapter's value. Plain `generate` knows nothing of the
structure. It draws $\mu$ from its prior and weights the trace by the
likelihood of the data at that $\mu$, so the weight changes with the key,
by tens of nats:

<!-- code: plain-weights -->
```clojure
(mapv #(:weight (gfi/generate noisy-mean [6] obs (rng/key %))) [1 2 3])
;=> [-5.889 -79.476 -28.095] ; float64
```

## A conjugate proposal

`with-conjugate-proposal` takes a generative function, the address of a
latent and its *children*, the choices whose distributions take the latent
as a parameter (here a selection, every choice under `:y`). It returns a
generative function with the same distribution and the same traces. Only
its *internal proposal* changes, the distribution `generate` draws the
unconstrained choices from: it draws the latent from its exact posterior
given the observed children. Its weight, $\log p(\tau) - \log q(\mu)$, is
then $\log p(y)$ whatever $\mu$ is drawn, the same for every key, and the
trace holds a draw from the posterior:

<!-- code: conjugate -->
```clojure
(def conjugate-mean
  (conj/with-conjugate-proposal noisy-mean :mu (sel/select :y)))

(def runs (mapv #(gfi/generate conjugate-mean [6] obs (rng/key %)) [1 2 3]))

(mapv #(array/allclose? (:weight %) (log-evidence ys)) runs)
;=> [true true true]

(mapv #((:trace %) :mu) runs)
;=> [2.237 2.883 1.847] ; float64
```

The posterior is normal, with precision $1/9 + n$ and mean $\sum_i y_i$
over that precision:

<!-- code: posterior -->
```clojure
(defn posterior
  "mu's exact posterior in noisy-mean given ys: precision 1/9 + n, mean
   Σy over the precision. {:mean :sd}."
  [ys]
  (let [precision (+ (/ 1.0 9.0) (count ys))]
    {:mean (/ (reduce + ys) precision) :sd (Math/sqrt (/ 1.0 precision))}))

(posterior ys)
;=> {:mean 1.9473 :sd 0.4045}
```

Four thousand traces, one batched `generate`, check that the wrapper
draws from it. A mean of $n$ draws has standard error $\sigma/\sqrt n$; a
sample variance has one of $\sqrt{(m_4 - v^2)/n}$, $m_4$ the fourth central
moment:

<!-- code: stats -->
```clojure
(defn mean-and-se
  "The mean of the host numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]
    [m (Math/sqrt (/ v n))]))

(defn variance-and-se
  "The sample variance of the host numbers xs and its standard error,
   sqrt((m4 - v^2) / n), m4 their fourth central moment."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        sq (mapv #(* (- % m) (- % m)) xs)
        v (/ (reduce + sq) (dec n))
        m4 (/ (reduce + (map #(* % %) sq)) n)]
    [v (Math/sqrt (/ (- m4 (* v v)) n))]))
```

<!-- code: posterior-draws -->
```clojure
(def mus
  (let [{ps :particles} (batched/generate conjugate-mean [6] obs 4000
                                          (rng/key 4))]
    (array/->clj (batched/choice ps :mu))))

(let [{:keys [mean sd]} (posterior ys)
      [m se] (mean-and-se mus)
      [v v-se] (variance-and-se mus)]
  [(< (abs (- m mean)) (* 5 se))
   (< (abs (- v (* sd sd))) (* 5 v-se))])
;=> [true true]
```

Both agree with the closed form within five standard errors. Since the
weight has no variance, importance sampling with one sample estimates the
evidence exactly:

<!-- code: one-sample -->
```clojure
(def one-sample
  (imp/importance-sampling conjugate-mean [6] obs 1 {:key (rng/key 5)}))

(array/allclose? (:log-ml-estimate one-sample) (log-evidence ys))
;=> true
```

`regenerate` on the latent is then a Gibbs move: it draws $\mu$ afresh from
the same posterior, keeps every observation, and has weight 0, so MH
always accepts it. The new $\mu$ differs from the old:

<!-- code: gibbs-move -->
```clojure
(let [tr (:trace (first runs))
      {moved :trace w :weight} (gfi/regenerate tr (sel/select :mu) (rng/key 6))]
  [(array/allclose? w 0.0 {:atol 1e-4})
   (array/allclose? (moved [:y 0]) (tr [:y 0]))
   (array/allclose? (moved :mu) (tr :mu))])
;=> [true true false]
```

How the wrapper does this explains what it checks. `generate` runs the
model twice. A *reference run* puts the latent at its prior mean and each
unobserved child at a value its distribution fixes, and records the
latent's prior and its children's distributions; from those it computes
the posterior. A *final run* puts the latent at a draw from that
posterior, keeps every other choice of the reference run, and draws the
unobserved children given the latent. Comparing the two runs, which used
two values of the latent, shows whether anything but the declared
children depends on it (below, *What the wrapper refuses*). *Debugging
with Enumeration* met the same idea for finite choices,
`enum/with-exact-proposal`, which enumerates where this wrapper
integrates.

## The variance it removes

An importance sampling estimate of the evidence is a random variable: run
it with another key and it changes. Its variance across keys is its Monte
Carlo error, and the conjugate proposal removes it. To measure it, a
thousand independent runs of 100 samples each come from one batched
`generate` of 100,000 samples, split a hundred at a time:

<!-- code: log-ml-runs -->
```clojure
(defn log-ml-runs
  "r independent importance sampling estimates of log p(observations),
   each from n samples: one batched generate of r·n samples, its weights
   split into r runs. Host numbers."
  [model args observations r n key]
  (let [{ws :weights} (batched/generate model args observations (* r n) key)]
    (mapv weights/log-mean-exp (partition n (array/->clj ws)))))
```

<!-- code: variance-full -->
```clojure
(def plain-runs (log-ml-runs noisy-mean [6] obs 1000 100 (rng/key 7)))

(def conjugate-runs (log-ml-runs conjugate-mean [6] obs 1000 100 (rng/key 7)))

(variance-and-se plain-runs)
;=> [0.0562 0.0026] ; float64

[(let [[v se] (variance-and-se plain-runs)] (> v (* 5 se)))
 (every? #(array/allclose? % (log-evidence ys)) conjugate-runs)]
;=> [true true]
```

Plain importance sampling's estimates have variance 0.056, more than
five of its standard errors from zero, and the conjugate proposal's are
all the exact evidence. The GenMLX chapter reported the same contrast
from four trials per particle count; with a thousand, the variance's
standard error is under 5% of it.

A model rarely integrates out completely. Give the noise an unknown scale
$s$, with a gamma prior of mean 1, and $\mu$ is still conjugate to the
observations given $s$, but $s$ is not:

<!-- code: noisy-scale -->
```clojure
(def noisy-scale
  (gen [n]
    (let [s (trace :s (d/gamma 4 0.25))
          mu (trace :mu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu s)))
      mu)))

(def conjugate-scale
  (conj/with-conjugate-proposal noisy-scale :mu (sel/select :y)))
```

The wrapper now draws $s$ from its prior, as plain `generate` does, and
$\mu$ from its posterior given $s$, so its weight is $\log p(y \mid s)$:
exact given $s$, and varying with it. The evidence itself is an integral
over $s$ of $p(s)\,p(y \mid s)$, and `log-evidence` gives $p(y \mid s)$ in
closed form, so a sum over a fine grid of $s$ computes it:

<!-- code: scale-evidence -->
```clojure
(def s-grid
  "Midpoints of 2000 intervals of width 0.0025 covering s in (0, 5]."
  (mapv #(* 0.0025 (+ % 0.5)) (range 2000)))

(def s-log-posterior
  "log p(s) + log p(ys | s) at each point of s-grid: mu summed out by
   log-evidence."
  (mapv #(array/item (+ (dc/log-prob (d/gamma 4 0.25) %) (log-evidence ys %)))
        s-grid))

(def scale-evidence
  (+ (Math/log 0.0025) (weights/log-sum-exp s-log-posterior)))

scale-evidence
;=> -6.5276
```

Both estimators of it vary now. The conjugate one varies much less, about
28 times less here, and the gap is more than five standard errors of the
difference:

<!-- code: variance-partial -->
```clojure
(def plain-scale-runs
  (log-ml-runs noisy-scale [6] obs 200 100 (rng/key 8)))

(def conjugate-scale-runs
  (log-ml-runs conjugate-scale [6] obs 200 100 (rng/key 8)))

(mapv (comp first variance-and-se) [plain-scale-runs conjugate-scale-runs])
;=> [0.558 0.0198] ; float64

(let [[v1 se1] (variance-and-se plain-scale-runs)
      [v2 se2] (variance-and-se conjugate-scale-runs)]
  (> (- v1 v2) (* 5 (Math/sqrt (+ (* se1 se1) (* se2 se2))))))
;=> true
```

Both are unbiased for $p(y)$: an importance sampling estimate is unbiased
in exp space, so its ratio to the exact evidence averages 1. Its log is
biased low (Jensen's inequality), more so the more it varies, as the
plain estimates' long left tail in the figure shows.

<!-- code: unbiased -->
```clojure
(defn ratio-check
  "Is the mean of exp(estimate - exact) over the runs within 5 standard
   errors of 1, as it is for unbiased estimates of exp(exact)?"
  [runs exact]
  (let [[m se] (mean-and-se (mapv #(Math/exp (- % exact)) runs))]
    (< (abs (- m 1.0)) (* 5 se))))

[(ratio-check plain-scale-runs scale-evidence)
 (ratio-check conjugate-scale-runs scale-evidence)]
;=> [true true]
```

<!-- code: variance-figure -->
```clojure
;; figure: figures/042-variance.svg
(svg/histogram [{:values plain-scale-runs :label "plain"}
                {:values conjugate-scale-runs :label "conjugate"}]
               {:bins 50 :x-label "estimate of log p(y), 100 samples each"})
```

![Estimates of the evidence from 100 samples each: plain importance
sampling, and the same with mu drawn from its conditional
posterior](figures/042-variance.svg)

This is the Rao–Blackwell theorem at work: integrating a dimension out
exactly can only lower an estimator's variance. What remains is the
variance of the dimensions sampled, here $s$'s.

## Five families

The wrapper knows the pairs a latent's family has, in Gen.jl's
parameterizations:

| The latent | Its children |
|------------|--------------|
| `normal(mu, std)` | `normal(latent, std)` |
| `mvnormal(mu, cov)` | `mvnormal(latent, cov)` |
| `beta(a, b)` | `bernoulli(latent)`, `binom(n, latent)` |
| `gamma(shape, scale)` | `poisson(latent)` |
| `dirichlet(alpha)` | `categorical(latent)` |

An array latent is elementwise, and a child may observe each element
several times: `(d/normal mu [1.0 2.0])` is two observations of a scalar
`mu`. The GenMLX chapter checked three families against closed forms;
here are its first two, and the Dirichlet in place of its third, which
gen.clj does not pair (the workshop takes it up). A probability with a
beta prior, seen through coin flips, has an evidence that is a product of
predictive probabilities: each flip's is its outcome's pseudo-count in the
prior plus its count so far, over the total of both. That is the Pólya
urn, and it holds for the Dirichlet and the categorical as well:

<!-- code: coin -->
```clojure
(def coin
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (doseq [i (range n)]
        (trace [:x i] (d/bernoulli p)))
      p)))

(def flips (cm/choicemap [:x 0] true [:x 1] true [:x 2] false))

(defn urn-log-evidence
  "log p(cs) of categories cs drawn from a dirichlet(alpha) probability
   vector summed out: each draw has probability (its category's alpha +
   its count so far) / (Σ alpha + draws so far), the Pólya urn."
  [alpha cs]
  (loop [counts (vec alpha) cs cs total 0.0]
    (if-let [[c & more] (seq cs)]
      (recur (update counts c + 1)
             more
             (+ total (Math/log (/ (counts c) (reduce + counts)))))
      total)))

(urn-log-evidence [2 2] [0 0 1])
;=> -2.302585

(def conjugate-coin (conj/with-conjugate-proposal coin :p (sel/select :x)))

(array/allclose? (:weight (gfi/generate conjugate-coin [3] flips (rng/key 9)))
                 (urn-log-evidence [2 2] [0 0 1]))
;=> true
```

The probability of the flips is $1/10$, GenMLX's number. A rate with a
gamma prior, seen through Poisson counts, makes each count negative
binomial given the ones before (GenMLX's gamma took a rate; Gen.jl's, and
gen.clj's, a scale, which is the same at 1):

<!-- code: counts -->
```clojure
(def counts
  (gen [n]
    (let [rate (trace :rate (d/gamma 3 1))]
      (doseq [i (range n)]
        (trace [:k i] (d/poisson rate)))
      rate)))

(defn lgamma
  "log Γ(x) of a host number."
  [x]
  (array/item (array/lgamma (double x))))

(defn negative-binomial-log-evidence
  "log p(ks) of poisson counts ks with a gamma(shape, scale) rate summed
   out: each count is negative binomial given the ones before, and adds
   itself to the shape and 1 to the rate 1/scale."
  [shape scale ks]
  (loop [a (double shape) b (/ 1.0 scale) ks ks total 0.0]
    (if-let [[k & more] (seq ks)]
      (recur (+ a k) (+ b 1.0) more
             (+ total (lgamma (+ a k)) (- (lgamma a)) (- (lgamma (inc k)))
                (* a (Math/log (/ b (+ b 1.0))))
                (- (* k (Math/log (+ b 1.0))))))
      total)))

(negative-binomial-log-evidence 3 1 [2 4])
;=> -3.847256

(def conjugate-counts
  (conj/with-conjugate-proposal counts :rate (sel/select :k)))

(array/allclose? (:weight (gfi/generate conjugate-counts [2]
                                        (cm/choicemap [:k 0] 2 [:k 1] 4)
                                        (rng/key 10)))
                 (negative-binomial-log-evidence 3 1 [2 4]))
;=> true
```

And a probability vector with a Dirichlet prior, seen through categories,
is the urn with three colors:

<!-- code: categories -->
```clojure
(def categories
  (gen [n]
    (let [probs (trace :probs (d/dirichlet [1.0 2.0 0.5]))]
      (doseq [i (range n)]
        (trace [:c i] (d/categorical probs)))
      probs)))

(def drawn [0 2 1 1 0 1])

(def conjugate-categories
  (conj/with-conjugate-proposal categories :probs (sel/select :c)))

(def drawn-observations
  (apply cm/choicemap (mapcat (fn [i c] [[:c i] c]) (range) drawn)))

(array/allclose? (:weight (gfi/generate conjugate-categories [6]
                                        drawn-observations (rng/key 11)))
                 (urn-log-evidence [1.0 2.0 0.5] drawn))
;=> true
```

## What the wrapper refuses

GenMLX's analyzer classified how a child's parameter depended on the
latent: *direct*, *affine* or *nonlinear*. gen.clj accepts only direct
ones: a child's parameter must be the latent's value itself, checked by
identity. A mean of $2\mu$ is a function of the latent, and a loud error:

<!-- code: affine-error -->
```clojure
;; error: with-conjugate-proposal: the mu of the child at [:y] is not the latent
(def doubled
  (gen []
    (let [mu (trace :mu (d/normal 0 3))]
      (trace :y (d/normal (* 2 mu) 1)))))

(gfi/generate (conj/with-conjugate-proposal doubled :mu [:y])
              [] (cm/choicemap :y 1.0) (rng/key 12))
```

A child of a family not conjugate to the latent's is one too:

<!-- code: family-error -->
```clojure
;; error: with-conjugate-proposal: the child at [:z] is a laplace
(def heavy-tailed
  (gen []
    (let [mu (trace :mu (d/normal 0 3))]
      (trace :y (d/normal mu 1))
      (trace :z (d/laplace mu 1)))))

(gfi/generate (conj/with-conjugate-proposal heavy-tailed :mu [:y :z])
              [] (cm/choicemap :y 1.0 :z 0.5) (rng/key 13))
```

These checks look at the structure of a run, so they hold on every
backend. A second kind reads values: the children's other parameters and
every other choice's density must be the same at both runs' values of the
latent, or the weight would not be exact. Like the distributions' checks
of their parameters, these run on an eager backend, the JVM's; a lazy
one, MLX's, reads no values there (the namespace's docstring has the
details). The six GenMLX bugs named above are each a test in
`test/gen/inference/conjugate_test.clj`: a latent's name rebound to
$\mu + 1$, a quadratic mean an affine probe accepted, an offset a Kalman
elimination dropped, a noise that depended on the latent, a non-conjugate
child beside a conjugate one, and a vector latent probed along its
diagonal only. Each was a wrong number in GenMLX, and each is a loud error
or an exact answer here.

## Gibbs moves in a chain

The wrapper's `regenerate` on the latent is a Gibbs move, so the wrapper
also improves MCMC. On `noisy-scale`, alternate an MH move on $s$ from its
prior with one on $\mu$. On the plain model the move on $\mu$ proposes from
$\mu$'s prior, most of whose draws land far from the data; on the wrapped
one it draws $\mu$ from its posterior given $s$:

<!-- code: gibbs-chain -->
```clojure
(defn sweeps
  "num-iters sweeps of MH on :s from its prior, then on :mu, from a trace
   of model given obs: each sweep's s, mu and whether the move on mu was
   accepted, as host values."
  [model num-iters key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate model [6] obs k-start))]
    (loop [tr start ks (rng/split k-chain num-iters) out []]
      (if-let [[k & more] (seq ks)]
        (let [[k-s k-mu] (rng/split k)
              tr (:trace (mh/mh tr (sel/select :s) k-s))
              {tr :trace accepted? :accepted?} (mh/mh tr (sel/select :mu) k-mu)]
          (recur tr more (conj out {:s (array/item (tr :s))
                                    :mu (array/item (tr :mu))
                                    :accepted? accepted?})))
        out))))

(def gibbs-sweeps (sweeps conjugate-scale 2000 (rng/key 14)))

(def plain-sweeps (sweeps noisy-scale 2000 (rng/key 14)))

(defn accepted
  "The fraction of sweeps whose move on mu was accepted."
  [sweeps]
  (/ (count (filter :accepted? sweeps)) (double (count sweeps))))

[(accepted gibbs-sweeps) (accepted plain-sweeps)]
;=> [1.0 0.076] ; float64

[(every? :accepted? gibbs-sweeps) (< (accepted plain-sweeps) 0.2)]
;=> [true true]
```

Every Gibbs move is accepted, where fewer than one in ten of the plain
ones is. The grid above holds $s$'s posterior, which gives the exact
posterior means: $s$'s directly, and $\mu$'s as the average over $s$ of
its conditional mean given $s$. The six observations lie close together,
and pull $s$ down to about half its prior mean of 1:

<!-- code: gibbs-exact -->
```clojure
(def s-posterior
  "The posterior probabilities of s-grid's points."
  (mapv #(Math/exp %) (:log-weights (weights/normalize s-log-posterior))))

(def exact-means
  "E[s | ys], and E[mu | ys]: E[mu | ys, s], Σy/s^2 over the precision
   1/9 + n/s^2, averaged over s's posterior."
  {:s (reduce + (map * s-posterior s-grid))
   :mu (reduce + (map (fn [p s]
                        (let [precision (+ (/ 1.0 9.0) (/ (count ys) (* s s)))]
                          (* p (/ (reduce + ys) (* s s) precision))))
                      s-posterior s-grid))})

exact-means
;=> {:s 0.5249 :mu 1.9718}

(defn chain-check
  "The summaries of the sweeps' s and mu after the first 200, and whether
   each mean is within 5 MCSE of the exact one."
  [sweeps]
  (let [summary (fn [k] (diag/summary [(mapv k (subvec sweeps 200))]))
        summaries {:s (summary :s) :mu (summary :mu)}]
    [(update-vals summaries #(select-keys % [:mean :mcse-mean :ess-bulk]))
     (every? (fn [[k {:keys [mean mcse-mean]}]]
               (< (abs (- mean (exact-means k))) (* 5 mcse-mean)))
             summaries)]))

(:mu (first (chain-check gibbs-sweeps)))
;=> {:mean 1.9748 :mcse-mean 0.0062 :ess-bulk 1577.7} ; float64

(:mu (first (chain-check plain-sweeps)))
;=> {:mean 1.9446 :mcse-mean 0.0231 :ess-bulk 111.4} ; float64

(mapv (comp second chain-check) [gibbs-sweeps plain-sweeps])
;=> [true true]
```

Both chains agree with both means within five Monte Carlo standard
errors, but for $\mu$ the Gibbs chain's 1800 draws are worth about 1600
independent ones, and the plain chain's about 110.

## Linear-Gaussian chains: the Kalman generative function

A conjugate pair is the simplest case. A linear-Gaussian state-space model
is normal all the way down: a state evolves by a linear map plus Gaussian
noise and is seen through Gaussian noise, so the whole path of states can
be integrated out at once, by the Kalman filter. The GenMLX chapter's
model has three steps:

$$
z_0 \sim \mathcal{N}(0, 1), \qquad
z_t \sim \mathcal{N}(0.9\, z_{t-1}, 0.5), \qquad
y_t \sim \mathcal{N}(z_t, 0.3).
$$

As an Unfold (*Composition*) whose steps trace the state:

<!-- code: ar-model -->
```clojure
(def ar-step
  (gen [t prev]
    (let [z (trace :z (if (zero? t)
                        (d/normal 0 1)
                        (d/normal (* 0.9 prev) 0.5)))]
      (trace :y (d/normal z 0.3))
      z)))

(def ar-chain (comb/unfold ar-step))

(def kys [0.4 0.7 1.1])
```

GenMLX's analyzer read the mean `0.9 · z` as an affine link and ran a
Kalman recursion inside the handler. In gen.clj that mean is what the
wrapper refuses above. Linear-Gaussian chains are instead
`gen.state-space/kalman`, DESIGN's "explicitly separate marginalized
generative function": an Unfold over a filter step whose state is the
*belief*, the distribution of the current state given the observations so
far. Its only choices are the observations, at addresses 0 to $n-1$, each
with density $p(y_t \mid y_0, \dots, y_{t-1})$, so its score is the exact
evidence. It takes $(n, \text{prior}, \text{params})$, with covariances
where `normal` takes standard deviations: $0.25 = 0.5^2$ and $0.09 =
0.3^2$.

<!-- code: ar-kalman -->
```clojure
(def ar-prior {:mean 0.0 :cov 1.0})

(def ar-params
  {:transition 0.9 :transition-cov 0.25 :emission 1.0 :emission-cov 0.09})

(defn series
  "The choice map of a state-space generative function's observations ys:
   (ys t) at address t."
  [ys]
  (apply cm/choicemap (interleave (range) ys)))

(def ar-exact
  (:weight (gfi/assess ss/kalman [3 ar-prior ar-params] (series kys))))

ar-exact
;=> -2.501852
```

It is GenMLX's number. The observations are jointly normal, and their
covariance is the states' plus $r^2$ on the diagonal: the brute-force
answer, with no recursion in it.

<!-- code: ar-joint -->
```clojure
(defn ar-covariance
  "The covariance of z_0 … z_n-1 when Var z_0 = v0 and z_t = a z_t-1 plus
   noise of standard deviation q: Var z_t = a^2 Var z_t-1 + q^2, and
   Cov(z_s, z_t) = a^|t-s| Var z_min(s,t)."
  [n a q v0]
  (let [variances (vec (reductions (fn [v _] (+ (* a a v) (* q q)))
                                   v0 (range (dec n))))]
    (vec (for [s (range n)]
           (vec (for [t (range n)]
                  (* (Math/pow a (abs (- t s))) (variances (min s t)))))))))

(defn ar-log-evidence
  "log p(ys) of the chain: the ys are jointly normal, with the states'
   covariance plus r^2 on the diagonal."
  [ys a q r]
  (let [n (count ys)]
    (dc/log-prob (d/mvnormal (array/zeros [n])
                             (+ (array/array (ar-covariance n a q 1.0))
                                (* r r (array/eye n))))
                 (array/array ys))))

(array/allclose? (ar-log-evidence kys 0.9 0.5 0.3) ar-exact)
;=> true
```

Importance sampling on the explicit chain, which draws each path of
states from the prior, estimates the same number. GenMLX's chapter saw
its estimates scatter, off by 0.11 at 3000 samples; here the estimate is
held to five of its standard errors, from its effective sample size as
*SMC in Depth* derived it:

<!-- code: ar-importance -->
```clojure
(def ar-observations
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) kys)))

(def ar-is
  (imp/importance-sampling ar-chain [3 0.0] ar-observations 3000
                           {:key (rng/key 15) :batched? true}))

(defn log-ml-se
  "The standard error of the log marginal likelihood estimate of n
   weighted samples with effective sample size ess: about
   sqrt(1/ess - 1/n)."
  [ess n]
  (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n))))

[(:log-ml-estimate ar-is) (:ess ar-is)]
;=> [-2.403 278.3] ; float64

(< (abs (- (:log-ml-estimate ar-is) ar-exact))
   (* 5 (log-ml-se (:ess ar-is) 3000)))
;=> true
```

The 3000 paths are worth 278, after only three observations.

Being an Unfold, the Kalman generative function grows incrementally. An
update to one more step adds the new observation's predictive density and
leaves the earlier steps alone, the same objects as before:

<!-- code: ar-update -->
```clojure
(def ar-trace (:trace (gfi/generate ss/kalman [3 ar-prior ar-params]
                                    (series kys) (rng/key 16))))

(def extended
  (gfi/update ar-trace [4 ar-prior ar-params]
              [diff/unknown-change diff/no-change diff/no-change]
              (cm/choicemap 3 1.6) (rng/key 17)))

(array/allclose? (:weight extended)
                 (- (ar-log-evidence (conj kys 1.6) 0.9 0.5 0.3) ar-exact))
;=> true

(every? #(identical? (trace/subtrace ar-trace %)
                     (trace/subtrace (:trace extended) %))
        (range 3))
;=> true
```

Its return value is the vector of filtered beliefs, each step's mean and
covariance; here the mean and standard deviation of each $z_t$ given the
observations up to $t$:

<!-- code: ar-beliefs -->
```clojure
(mapv (fn [{:keys [mean cov]}] [mean (Math/sqrt (array/item cov))])
      (:retval ar-trace))
;=> [[0.367 0.2873] [0.6182 0.2647] [0.9767 0.2638]]
```

## Inside a particle filter

The Kalman generative function can be one part of a larger model. Suppose
the chain's coefficient is unknown, one of 0.5, 0.9 and 1, equally likely
a priori. With the states traced, a particle filter must sample both the
coefficient and every state; with the states summed out by the Kalman
generative function, it samples only the coefficient. That is a
*Rao–Blackwellized* particle filter, and it is the same particle filter
code on another model:

<!-- code: regimes -->
```clojure
(def coefficients [0.5 0.9 1.0])

(def regime-step
  (gen [t prev a]
    (let [z (trace :z (if (zero? t)
                        (d/normal 0 1)
                        (d/normal (* a prev) 0.5)))]
      (trace :y (d/normal z 0.3))
      z)))

(def sampled-states
  (gen [n]
    (let [k (trace :a (d/uniform-discrete 0 2))
          a (array/take (array/array coefficients) k)]
      (splice :ys (comb/unfold regime-step) n 0.0 a))))

(def summed-states
  (gen [n]
    (let [k (trace :a (d/uniform-discrete 0 2))
          a (array/take (array/array coefficients) k)]
      (splice :ys ss/kalman n ar-prior (assoc ar-params :transition a)))))
```

The data drift upward over twelve steps. Given each coefficient the Kalman
filter gives the likelihood exactly, so the evidence and the coefficient's
posterior follow by summing over the three:

<!-- code: drift -->
```clojure
(def drift [0.3 0.9 1.4 1.2 1.9 2.5 2.2 2.8 3.1 2.6 3.3 3.9])

(def drift-log-likelihoods
  "log p(drift | a) for each coefficient a, by the Kalman filter."
  (mapv (fn [a]
          (array/item (:weight (gfi/assess ss/kalman
                                           [(count drift) ar-prior
                                            (assoc ar-params :transition a)]
                                           (series drift)))))
        coefficients))

(def drift-evidence
  (- (weights/log-sum-exp drift-log-likelihoods) (Math/log 3)))

drift-evidence
;=> -11.4555

(mapv #(Math/exp %) (:log-weights (weights/normalize drift-log-likelihoods)))
;=> [0.0 0.0364 0.9636]
```

A coefficient of 0.5 pulls the state back toward 0 too fast for this
drift, and 1 is 26 times as probable as 0.9. A bootstrap filter, as in
*SMC in Depth*, runs each model through the data one step at a time. For
`summed-states` each step is an update of the Kalman generative function
to one more observation, whose weight is that observation's predictive
density given the coefficient:

<!-- code: filters -->
```clojure
(defn run-filter
  "A bootstrap particle filter of n particles on model through drift,
   batched, resampling when the ESS falls below n/2; observation is the
   choice map of step t's observation. Its state."
  [model observation n key]
  (let [steps (count drift)
        ks (rng/split key (* 2 steps))]
    (reduce (fn [state t]
              (-> state
                  (pf/maybe-resample {:key (ks (+ steps t))})
                  (pf/step [(inc t)] [diff/unknown-change] (observation t)
                           {:key (ks t)})))
            (pf/initialize model [1] (observation 0) n
                           {:batched? true :key (ks 0)})
            (range 1 steps))))

(defn sampled-observation [t] (cm/choicemap [:ys t :y] (drift t)))

(defn summed-observation [t] (cm/choicemap [:ys t] (drift t)))

(defn filter-runs
  "The log evidence estimates of r runs of a filter of n particles."
  [model observation n r key]
  (mapv #(pf/log-ml-estimate (run-filter model observation n %))
        (rng/split key r)))
```

Two hundred and fifty runs of each, with 400 particles:

<!-- code: filter-variance -->
```clojure
(def bootstrap-runs
  (filter-runs sampled-states sampled-observation 400 250 (rng/key 18)))

(def rao-blackwell-runs
  (filter-runs summed-states summed-observation 400 250 (rng/key 18)))

(mapv (comp first variance-and-se) [bootstrap-runs rao-blackwell-runs])
;=> [0.157 0.0046] ; float64

(let [[v1 se1] (variance-and-se bootstrap-runs)
      [v2 se2] (variance-and-se rao-blackwell-runs)]
  (> (- v1 v2) (* 5 (Math/sqrt (+ (* se1 se1) (* se2 se2))))))
;=> true

[(ratio-check bootstrap-runs drift-evidence)
 (ratio-check rao-blackwell-runs drift-evidence)]
;=> [true true]
```

The filter that samples the states varies about 34 times more, by more
than five standard errors of the difference, and both are unbiased. What
the Rao–Blackwellized filter has left to sample is the coefficient: its
noise is in how the particles share out three values.

## Hidden Markov chains: the forward algorithm

The discrete analog is the hidden Markov model, whose states are
categories and whose chain sums out by the forward algorithm. The
classic example is a casino that switches between a fair die and a loaded
one, which shows a six half the time. It keeps the fair die with
probability 0.95 and the loaded one with 0.9:

<!-- code: casino -->
```clojure
(def dice
  "The two dice's probabilities of faces 0 … 5, 5 standing for a six:
   the fair die's, then the loaded one's."
  (array/array [(vec (repeat 6 (/ 1.0 6))) [0.1 0.1 0.1 0.1 0.1 0.5]]))

(def switches
  "The probabilities of the next die given this one: row 0 the fair's."
  (array/array [[0.95 0.05] [0.1 0.9]]))

(def casino {:transition switches :emission (d/categorical dice)})

(def start (array/array [0.5 0.5]))

(def rolls [5 5 2 0 1 3 2 5 0 1 5 2 0 5 3 5 4 5 0 3 0 1 3 4 5 2 4 4 3 2
            3 2 3 5 5 0 1 3 3 4 4 4 5 5 5 5 5 2 3 5 0 5 5 5 0 1 5 5 3 5])

(def first-rolls (subvec rolls 0 10))

(def casino-exact
  (:weight (gfi/assess ss/hmm [10 start casino] (series first-rolls))))

casino-exact
;=> -17.488016
```

`gen.state-space/hmm` takes the first state's probabilities, a transition
matrix and an emission: one distribution batched over the states on its
leading axis, here `(d/categorical dice)`, whose row $k$ is die $k$'s
faces. Its choices are the rolls and its score the evidence, here of the
first ten rolls. The brute-force answer traces the dice as choices, in an
Unfold, and enumerates every path of ten dice, $2^{10}$ of them, as
*Debugging with Enumeration* did:

<!-- code: casino-enumeration -->
```clojure
(def casino-step
  (gen [t prev]
    (let [die (trace :die (d/categorical (if (zero? t)
                                           start
                                           (array/take switches prev 0))))]
      (trace :roll (d/categorical (array/take dice die 0)))
      die)))

(def every-path
  (apply enum/choice-vol-grid (map (fn [t] [[t :die] [0 1]]) (range 10))))

(def enumerated
  (enum/enumerative-inference
   (comb/unfold casino-step) [10 0]
   (apply cm/choicemap (mapcat (fn [t r] [[t :roll] r]) (range) first-rolls))
   every-path {:batched? true :key (rng/key 20)}))

[(count every-path)
 (array/allclose? (:log-ml-estimate enumerated) casino-exact)]
;=> [1024 true]
```

The enumeration also gives each die's posterior, the *smoothing*
marginals $P(\text{loaded at } t \mid \text{all ten rolls})$. The hidden
Markov generative function does not hold those: its beliefs are filtered,
given the rolls up to $t$. `ss/sample-states` draws paths of states from
their posterior given a trace's observations, by forward filtering and
backward sampling, and the share of paths loaded at $t$ estimates the
smoothing marginal:

<!-- code: casino-smoothing -->
```clojure
(def smoothed
  "P(die t is loaded | the ten rolls), by enumeration."
  (mapv #(get (enum/marginal enumerated [% :die]) 1 0.0) (range 10)))

smoothed
;=> [0.730 0.660 0.430 0.305 0.242 0.217 0.222 0.259 0.217 0.207]

(def casino-trace
  (:trace (gfi/generate ss/hmm [10 start casino] (series first-rolls)
                        (rng/key 21))))

(def die-paths
  (mapv #(mapv array/item (ss/sample-states casino-trace %))
        (rng/split (rng/key 22) 2000)))

(defn loaded-share
  "The share of paths whose die t is loaded."
  [paths t]
  (/ (count (filter #(= 1 (% t)) paths)) (double (count paths))))

(every? (fn [t]
          (let [p (smoothed t)]
            (< (abs (- (loaded-share die-paths t) p))
               (* 5 (Math/sqrt (/ (* p (- 1 p)) (count die-paths)))))))
        (range 10))
;=> true
```

All ten agree with the enumeration within five standard errors. The two
sixes that open the rolls make the loaded die likely at first; the next
rolls, few of them sixes, pull it down. Over all sixty rolls, from 500
paths, the loaded die shows where the sixes cluster:

<!-- code: casino-figure -->
```clojure
;; figure: figures/042-casino.svg
(let [tr (:trace (gfi/generate ss/hmm [60 start casino] (series rolls)
                               (rng/key 23)))
      paths (mapv #(mapv array/item (ss/sample-states tr %))
                  (rng/split (rng/key 24) 500))]
  (svg/paths [{:paths [(mapv (fn [t] [t (loaded-share paths t)]) (range 60))]}
              {:paths [(keep (fn [t] (when (= 5 (rolls t)) [t 1.05]))
                             (range 60))]
               :width 0 :r 2.5 :color "firebrick"}]
             {:x-label "roll t" :y-label "P(loaded at t | rolls)"
              :y-range [0.0 1.1]}))
```

![The probability that the loaded die is in play at each roll, given all
sixty rolls; the red dots mark the sixes](figures/042-casino.svg)

## Inside MCMC

With the dice summed out, MCMC on the casino's parameters runs in their
few dimensions, not in sixty more of states. Suppose the loaded die's
probability of a six, $p_6$, is unknown, uniform a priori, and its other
faces share the rest:

<!-- code: six-model -->
```clojure
(defn casino-with
  "The casino's parameters with the loaded die's probability of a six p6,
   its other faces sharing the rest."
  [p6]
  (let [others (array/full [5] (/ (- 1.0 p6) 5.0))]
    {:transition switches
     :emission (d/categorical
                (array/stack [(array/full [6] (/ 1.0 6))
                              (array/concatenate [others
                                                  (array/reshape p6 [1])])]))}))

(def unknown-six
  (gen [n]
    (let [p6 (trace :p6 (d/uniform 0 1))]
      (splice :rolls ss/hmm n start (casino-with p6))
      p6)))

(def roll-observations
  (apply cm/choicemap (mapcat (fn [t r] [[:rolls t] r]) (range) rolls)))
```

The posterior of one parameter can be computed on a grid: the prior is
flat, and the hidden Markov generative function gives each point's
likelihood exactly.

<!-- code: six-grid -->
```clojure
(def p6-grid
  "Midpoints of 200 intervals covering [0, 1]."
  (mapv #(* 0.005 (+ % 0.5)) (range 200)))

(def p6-posterior
  "The posterior probabilities of p6-grid's points: the prior is uniform,
   and the HMM gives each point's likelihood exactly."
  (let [lls (mapv #(array/item
                    (:weight (gfi/assess ss/hmm [60 start (casino-with %)]
                                         (series rolls))))
                  p6-grid)]
    (mapv #(Math/exp %) (:log-weights (weights/normalize lls)))))

(def p6-mean (reduce + (map * p6-posterior p6-grid)))

p6-mean
;=> 0.6135
```

MH that proposes $p_6$ from its prior, 1500 steps, agrees with the grid
within five Monte Carlo standard errors:

<!-- code: six-chain -->
```clojure
(def six-chain
  (let [[k-start k-chain] (rng/split (rng/key 25))
        start-tr (:trace (gfi/generate unknown-six [60] roll-observations
                                       k-start))]
    (loop [tr start-tr ks (rng/split k-chain 1500) out []]
      (if-let [[k & more] (seq ks)]
        (let [tr (:trace (mh/mh tr (sel/select :p6) k))]
          (recur tr more (conj out (array/item (tr :p6)))))
        out))))

(def six-summary (diag/summary [(subvec six-chain 100)]))

(select-keys six-summary [:mean :mcse-mean :ess-bulk])
;=> {:mean 0.6178 :mcse-mean 0.0058 :ess-bulk 538.3} ; float64

(< (abs (- (:mean six-summary) p6-mean)) (* 5 (:mcse-mean six-summary)))
;=> true
```

<!-- code: six-figure -->
```clojure
;; figure: figures/042-six.svg
(svg/histogram [{:values (subvec six-chain 100)
                 :density (fn [x]
                            (let [i (min 199 (long (/ x 0.005)))]
                              (/ (p6-posterior i) 0.005)))}]
               {:bins 30 :range [0.0 1.0]
                :x-label "p6, the loaded die's probability of a six"})
```

![MH draws of p6 given the sixty rolls, and the posterior density from
the grid](figures/042-six.svg)

The prior's proposals stay in $[0, 1]$, where the loaded die's
probabilities are valid. The dice themselves, which the chain never
sampled, come back given any of its draws. Take a trace with $p_6$ at the
chain's last value; `ss/sample-states` on its call at `:rolls`, the
hidden Markov generative function's part, draws the sixty dice given
$p_6$ and the rolls. A draw of $p_6$ and then of the dice is a draw from
their joint posterior.

<!-- code: six-states -->
```clojure
(def six-trace
  (:trace (gfi/generate unknown-six [60]
                        (cm/set-choice roll-observations :p6 (peek six-chain))
                        (rng/key 31))))

(def dice-path (ss/sample-states (trace/subtrace six-trace :rolls)
                                 (rng/key 32)))

(count dice-path)
;=> 60
```

## Limits

Two things are not supported yet, each a bean.

**One latent per wrapper, and no affine children** (genclj-mwha). A model
with two conjugate latents, each with its own observations:

<!-- code: two-means -->
```clojure
(def two-means
  (gen [n m]
    (let [mu (trace :mu (d/normal 0 3))
          nu (trace :nu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu 1)))
      (doseq [i (range m)]
        (trace [:z i] (d/normal nu 1))))))

(def zs [-0.4 0.3 -1.1])

(def both-observed
  (cm/merge obs (apply cm/choicemap (mapcat (fn [i z] [[:z i] z]) (range) zs))))

(def mu-only (conj/with-conjugate-proposal two-means :mu (sel/select :y)))

(mapv #(:weight (gfi/generate mu-only [6 3] both-observed (rng/key %))) [1 2])
;=> [-31.313 -46.765] ; float64
```

Wrapping $\mu$ integrates it out, but $\nu$ is still drawn from its prior,
and the weight varies. A second wrapper cannot go around the first, which
is not a dynamic generative function:

<!-- code: nested-error -->
```clojure
;; error: with-conjugate-proposal needs a dynamic generative function
(conj/with-conjugate-proposal mu-only :nu (sel/select :z))
```

When the latents are independent, the model can be written as calls of
wrapped generative functions, one call for each latent. Then each call's
weight is exact, and so is their sum:

<!-- code: one-each -->
```clojure
(def grouped
  (gen [n m]
    (splice :a conjugate-mean n)
    (splice :b conjugate-mean m)))

(def grouped-observations
  (apply cm/choicemap (concat (mapcat (fn [i y] [[:a :y i] y]) (range) ys)
                              (mapcat (fn [i z] [[:b :y i] z]) (range) zs))))

(mapv #(array/allclose? (:weight (gfi/generate grouped [6 3]
                                               grouped-observations
                                               (rng/key %)))
                        (+ (log-evidence ys) (log-evidence zs)))
      [1 2])
;=> [true true]
```

Latents that depend on each other, $\theta \sim \mathcal{N}(\mu, 1)$ under
$\mu \sim \mathcal{N}(0, 3)$, say, have no such workaround: one of them
can be wrapped, which gives Gibbs moves on it, but a joint proposal for
both would need the dependence between them. Nor are affine children
supported, as *What the wrapper refuses* showed; a sound linear tracer
would be needed, and linear-Gaussian chains are the Kalman generative
function's meanwhile. The extended Kalman filter, GenMLX's linearized
sibling for nonlinear chains, is not ported (genclj-mgvu); a nonlinear
chain is a particle filter's.

**Missing observations** (genclj-llbf). A real series has gaps. When an
observation of the Kalman or hidden Markov generative function is left
out of the constraints, it is forward-sampled from its predictive
distribution, not summed out. Take the drifting series without step 5:

<!-- code: gap -->
```clojure
(def walk-args [12 ar-prior (assoc ar-params :transition 1.0)])

(def gapped
  "drift's observations without step 5's."
  (apply cm/choicemap (mapcat (fn [t y] (when (not= t 5) [t y]))
                              (range) drift)))

(def gap-weights
  (array/->clj (:weights (batched/generate ss/kalman walk-args gapped 4000
                                           (rng/key 26)))))

(- (apply max gap-weights) (apply min gap-weights))
;=> 8.08 ; float64
```

The weights now vary, by 8 nats here, since the steps after the gap
condition on a sampled observation. They are still correct importance
weights. The exact answer leaves step 5's row and column out of the joint
Gaussian, and the weights' exponentials average to it:

<!-- code: gap-exact -->
```clojure
(def gap-evidence
  "log p of drift but step 5, from the joint Gaussian of the walk's
   observations with step 5's row and column left out."
  (let [kept (vec (remove #{5} (range 12)))
        full (ar-covariance 12 1.0 0.5 1.0)
        cov (mapv (fn [s] (mapv (fn [t] (+ (get-in full [s t])
                                           (if (= s t) 0.09 0.0)))
                                kept))
                  kept)]
    (array/item (dc/log-prob (d/mvnormal (array/zeros [11]) (array/array cov))
                             (array/array (mapv drift kept))))))

gap-evidence
;=> -9.75597

(ratio-check gap-weights gap-evidence)
;=> true
```

The filter could instead skip the update at the gap, predicting only, and
stay exact. Until the bean decides how a gap is said, a step can be made
to observe nothing of the state: with an emission of 0 the Kalman gain is
0, the belief passes through unchanged, and the dummy observation adds
its own density, $\log \mathcal{N}(0; 0, 1)$, which is subtracted:

<!-- code: gap-workaround -->
```clojure
(defn blind-at
  "Kalman parameters that, at step gap, observe nothing of the state: an
   emission of 0, its noise 1."
  [params gap]
  (fn [t] (if (= t gap) (assoc params :emission 0.0 :emission-cov 1.0) params)))

(let [[n prior params] walk-args
      w (:weight (gfi/assess ss/kalman [n prior (blind-at params 5)]
                             (cm/set-choice gapped 5 0.0)))]
  (array/allclose? (- w (dc/log-prob (d/normal 0 1) 0.0)) gap-evidence))
;=> true
```

## Choosing

| Structure in the model | Use | Its weight |
|-----------------------------|------------------------------------|----------------------------------|
| A latent with conjugate children, nothing else depending on it | `with-conjugate-proposal` | the evidence, exact |
| The same, beside choices that are not conjugate | `with-conjugate-proposal` | exact given those choices, which are sampled |
| Finite choices | `enum/with-exact-proposal` | exact, the choices enumerated |
| A linear-Gaussian chain of states | `gen.state-space/kalman`; states by `ss/sample-states` | the evidence, exact |
| A chain of discrete states | `gen.state-space/hmm`; states by `ss/sample-states` | the evidence, exact |
| An affine child, or two dependent latents | not yet (genclj-mwha) | the model's own proposal's |
| A gap in a series | not yet (genclj-llbf) | forward-sampled; or a blind step, by hand |

In every case the structure is declared, and the generative function's
meaning stays the model's: the wrapper keeps its distribution and traces,
and the state-space generative functions are models in their own right,
whose laws `test/gen/state_space_test.clj` checks like any other's. The
next chapter, *Language Models as Generative Functions*, takes the same
interface to a model of text.

## Workshop

Each exercise has a tested solution at the end of the chapter. The first
uses this model, a rate with a gamma prior seen through exponential waits:

<!-- code: solution-1-model -->
```clojure
(def waits
  (gen [n]
    (let [rate (trace :rate (d/gamma 2 1))]
      (doseq [i (range n)]
        (trace [:t i] (d/exponential rate)))
      rate)))

(def waiting (cm/choicemap [:t 0] 0.5 [:t 1] 1.5))
```

1. **A sixth pair.** The gamma and the exponential are conjugate, and
   GenMLX's table had them. Wrap `waits` with `with-conjugate-proposal`,
   its children the waits. Predict what happens. Then compute the
   evidence of the two waits by hand, and check plain importance sampling
   against it.
2. **A Kalman filter by hand.** Write the scalar Kalman recursion on host
   numbers and check it against `ar-exact`. Then break it as the GenMLX
   chapter's last exercise did: drop the $a^2 P$ term from the predicted
   variance. Predict whether the evidence goes up or down.
3. **The $1/N$ you can see.** The variance of plain importance sampling's
   log evidence falls like $1/N$ in the number of samples. Check it: is
   $N$ times the variance the same at $N = 100$ and $N = 400$, within five
   standard errors?
4. **Smoothing a chain.** Draw 2000 paths of the AR chain's states with
   `ss/sample-states` from `ar-trace`, and check their means against the
   smoothing means from the joint Gaussian of the states and the
   observations.

## Solutions

**1.** The wrapper refuses it: an exponential child is not in the gamma's
row of the table, whose only child is the Poisson.

<!-- code: solution-1-error -->
```clojure
;; error: with-conjugate-proposal: the child at [:t 0] is a
(gfi/generate (conj/with-conjugate-proposal waits :rate (sel/select :t))
              [2] waiting (rng/key 27))
```

By hand, each wait is Lomax given the ones before, and the evidence is
GenMLX's number. Plain importance sampling from the rate's prior agrees
with it within five standard errors; with two observations, its 10,000
samples are worth 7000:

<!-- code: solution-1 -->
```clojure
(defn lomax-log-evidence
  "log p(ts) of exponential waits ts with a gamma(shape, scale) rate
   summed out: each wait is Lomax given the ones before, a b^a / (b +
   t)^(a+1) for shape a and rate b = 1/scale, and adds 1 to the shape and
   itself to the rate."
  [shape scale ts]
  (loop [a (double shape) b (/ 1.0 scale) ts ts total 0.0]
    (if-let [[t & more] (seq ts)]
      (recur (+ a 1.0) (+ b t) more
             (+ total (Math/log a) (* a (Math/log b))
                (- (* (+ a 1.0) (Math/log (+ b t))))))
      total)))

(lomax-log-evidence 2 1 [0.5 1.5])
;=> -2.60269

(def waits-is
  (imp/importance-sampling waits [2] waiting 10000
                           {:key (rng/key 28) :batched? true}))

(:ess waits-is)
;=> 7143.3 ; float64

(< (abs (- (:log-ml-estimate waits-is) (lomax-log-evidence 2 1 [0.5 1.5])))
   (* 5 (log-ml-se (:ess waits-is) 10000)))
;=> true
```

**2.** Dropping $a^2 P$ forgets the uncertainty carried from the previous
step, so every predictive variance after the first shrinks. Here the
evidence goes up, to −2.454 from −2.502: the narrower predictions fit
these three observations better. The agreement of three computations,
the recursion, `ss/kalman` and the joint Gaussian, catches the error
whichever way it goes.

<!-- code: solution-2 -->
```clojure
(defn hand-kalman
  "log p(ys) of the chain z_0 ~ N(0, 1), z_t ~ N(a z_t-1, q), y_t ~ N(z_t,
   r), by the Kalman recursion on host numbers: predict the state, score
   y_t by its predictive N(m, P + r^2), then correct. ahead is the
   predicted variance as a function of the last one, a^2 P + q^2."
  ([ys a q r] (hand-kalman ys a q r (fn [p] (+ (* a a p) (* q q)))))
  ([ys a q r ahead]
   (loop [t 0 m 0.0 p 1.0 total 0.0]
     (if (= t (count ys))
       total
       (let [[m p] (if (zero? t) [m p] [(* a m) (ahead p)])
             s (+ p (* r r))
             e (- (ys t) m)
             gain (/ p s)]
         (recur (inc t) (+ m (* gain e)) (* (- 1.0 gain) p)
                (- total (* 0.5 (+ (Math/log (* 2 Math/PI s))
                                   (/ (* e e) s))))))))))

(array/allclose? (hand-kalman kys 0.9 0.5 0.3) ar-exact)
;=> true

(hand-kalman kys 0.9 0.5 0.3 (fn [_] (* 0.5 0.5)))
;=> -2.4535
```

**3.** Within five standard errors, yes: 5.6 and 4.9, each known to about
0.3. A variance falling like $1/\sqrt N$ would have doubled the second.
The match is the delta method's, $\mathrm{Var}(\log \hat Z) \approx
\mathrm{Var}(w/Z)/N$, which holds as $N$ grows.

<!-- code: solution-3 -->
```clojure
(def runs-400 (log-ml-runs noisy-mean [6] obs 500 400 (rng/key 29)))

(let [[v1 se1] (variance-and-se plain-runs)
      [v2 se2] (variance-and-se runs-400)]
  [(* 100 v1) (* 400 v2)])
;=> [5.62 4.87] ; float64

(let [[v1 se1] (variance-and-se plain-runs)
      [v2 se2] (variance-and-se runs-400)]
  (< (abs (- (* 100 v1) (* 400 v2)))
     (* 5 (Math/sqrt (+ (* 100 se1 100 se1) (* 400 se2 400 se2))))))
;=> true
```

**4.** The states and the observations are jointly normal, so the states'
posterior is the Gaussian conditional: with $\Sigma$ the states'
covariance and $C = \Sigma + r^2 I$ the observations', its mean is $\Sigma
C^{-1} y$ and its covariance $\Sigma - \Sigma C^{-1} \Sigma$. The last
state's smoothing distribution is its filtering one, the Kalman generative
function's last belief (0.9767 and 0.2638). The sampled paths agree with
every mean within five standard errors:

<!-- code: solution-4 -->
```clojure
(defn ar-smoothing
  "The mean and standard deviation of each z_t given ys, by conditioning
   the joint Gaussian of the states and the observations: with Σ the
   states' covariance and C = Σ + r^2 I the observations', E[z | ys] = Σ
   C^-1 ys and Cov[z | ys] = Σ - Σ C^-1 Σ."
  [ys a q r]
  (let [n (count ys)
        sigma (array/array (ar-covariance n a q 1.0))
        c-inv-sigma (linalg/solve (+ sigma (* r r (array/eye n))) sigma)
        cov (- sigma (array/matmul sigma c-inv-sigma))]
    {:mean (array/->clj (array/matmul (array/transpose c-inv-sigma)
                                      (array/array ys)))
     :sd (mapv #(Math/sqrt %) (array/->clj (array/diag cov)))}))

(def ar-smoothed (ar-smoothing kys 0.9 0.5 0.3))

ar-smoothed
;=> {:mean [0.4548 0.7046 0.9767] :sd [0.2616 0.2451 0.2638]}

(def state-draws
  (mapv #(mapv array/item (ss/sample-states ar-trace %))
        (rng/split (rng/key 30) 2000)))

(every? (fn [t]
          (let [[m se] (mean-and-se (mapv #(% t) state-draws))]
            (< (abs (- m (get-in ar-smoothed [:mean t])))
               (* 5 (/ (get-in ar-smoothed [:sd t]) (Math/sqrt 2000))))))
        (range 3))
;=> true
```
