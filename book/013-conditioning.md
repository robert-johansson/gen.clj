# Conditioning

*Getting Started* ran a model forward with `simulate`, then conditioned it
on data with one call to `generate`. This chapter is about that call.
Conditioning asks a model a question in reverse: given that some of its
choices took the values we observed, what were the others likely to be?
The model's code does not change. `generate` runs it with a choice map of
the observed values, keeps those choices at their values, and returns one
number besides the trace, the *weight*. Every inference algorithm in this
book is built on that number.

This chapter ports Gen.jl's *A Bottom-Up Introduction to Gen* (choice
maps and their probabilities, `generate`'s internal proposal, importance
sampling by hand) and the GenMLX book's chapter on conditioning (the
weight, `project`, the marginal likelihood, prior against posterior,
rejection sampling). Its models are small enough to solve exactly, so
every estimate is checked against the exact answer, within five of its
standard errors.

<!-- code: ns -->
```clojure
(ns book.conditioning
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## A model we can solve by hand

Gen.jl's bottom-up introduction works with a model of three choices, each
true or false:

<!-- code: foo -->
```clojure
(def foo
  (gen [prob-a]
    (let [val (if (trace :a (d/bernoulli prob-a))
                (trace :b (d/bernoulli 0.6))
                true)]
      (and (trace :c (d/bernoulli (if val 0.9 0.2))) val))))
```

`:a` is true with probability `prob-a`, and only then does the model make
the choice `:b`. `:c` is true with probability 0.9 when `val` is true and
0.2 when it is false. The model returns `c && val`, as Gen.jl's `foo`
does. A `bernoulli` choice is a boolean, as in Gen.jl (GenMLX's was 0 or
1), and the body branches on it with `if`.

A run of `foo` makes one of six choice maps. For arguments $x$, the
probability $p(t; x)$ of a choice map $t$ is the product of the
probabilities of its choices. At `prob-a` 0.3, with ¬ for false:

| choice map $t$ | $p(t; 0.3)$ | return value |
|----------------|-------------|--------------|
| a, b, c | 0.3 · 0.6 · 0.9 = 0.162 | true |
| a, b, ¬c | 0.3 · 0.6 · 0.1 = 0.018 | false |
| a, ¬b, c | 0.3 · 0.4 · 0.2 = 0.024 | false |
| a, ¬b, ¬c | 0.3 · 0.4 · 0.8 = 0.096 | false |
| ¬a, c | 0.7 · 0.9 = 0.63 | true |
| ¬a, ¬c | 0.7 · 0.1 = 0.07 | false |

`gfi/assess` computes the table. It takes a complete choice map, runs the
model on it without drawing anything, and returns $\log p(t; x)$ as its
`:weight`, with the return value. Since it draws nothing for `foo`, it
needs no key:

<!-- code: table -->
```clojure
(def every-choice-map
  [(cm/choicemap :a true :b true :c true)
   (cm/choicemap :a true :b true :c false)
   (cm/choicemap :a true :b false :c true)
   (cm/choicemap :a true :b false :c false)
   (cm/choicemap :a false :c true)
   (cm/choicemap :a false :c false)])

(defn probability
  "p(t; prob-a): the probability that foo makes the choice map t."
  [t prob-a]
  (array/exp (:weight (gfi/assess foo [prob-a] t))))

(def table (mapv #(probability % 0.3) every-choice-map))

table
;=> [0.162 0.018 0.024 0.096 0.63 0.07]

(array/allclose? 1.0 (reduce + table))
;=> true

(mapv #(:retval (gfi/assess foo [0.3] %)) every-choice-map)
;=> [true false false false true false]
```

The six probabilities sum to 1, as the probabilities of all the ways a
program can run must. A trace's score is the log of its row. Key 1 makes
{¬a, c}, as the run in Gen.jl's tutorial did, with score $\log 0.63$.
(Values marked `; float64` are the JVM backend's: a float32 backend
draws others from the same keys.)

<!-- code: score -->
```clojure
(def tr (gfi/simulate foo [0.3] (rng/key 1)))

(cm/to-map (:choices tr))
;=> {:a false, :c true} ; float64

(array/allclose? (:score tr) (array/log (probability (:choices tr) 0.3)))
;=> true
```

## Constrained execution

Now the question in reverse. Suppose we observed that `:a` was true and
`:c` false. What was `:b`? `gfi/generate` takes the model, its arguments,
a choice map of *constraints* and a key. It returns a map of a trace and a
weight, where Gen.jl's returns a tuple:

<!-- code: generate -->
```clojure
(def constraints (cm/choicemap :a true :c false))

(def generated (gfi/generate foo [0.3] constraints (rng/key 2)))

(cm/to-map (:choices (:trace generated)))
;=> {:a true, :b false, :c false} ; float64

(:weight generated)
;=> -1.427 ; float64
```

The trace holds the constrained choices at their given values, and a
value of `:b`, which no constraint fixed, drawn as `simulate` would draw
it.

Look at the address `:c`. Under `simulate`, the handler draws `:c` from
its distribution and adds its log probability to the score. Under
`generate` with `:c` constrained, the handler takes the given value, adds
its log probability to the score, and adds it to the weight as well. The
model's code is the same in both: `(trace :c …)` names a choice and its
distribution, and says nothing about where the value comes from.
Conditioning is not an edit to the model but another reading of it; *The
Handler Loop* shows the code that does the reading.

Two extremes bear this out. With no constraints, `generate` is
`simulate`: the same key makes the same choices, and the weight is 0.
With every choice constrained, nothing is drawn, and the weight is the
whole score:

<!-- code: extremes -->
```clojure
(let [t (gfi/simulate foo [0.3] (rng/key 4))
      free (gfi/generate foo [0.3] cm/EMPTY (rng/key 4))
      fixed (gfi/generate foo [0.3] (:choices t) (rng/key 5))]
  [(= (mapv array/->clj (cm/values (:choices t)))
      (mapv array/->clj (cm/values (:choices (:trace free)))))
   (:weight free)
   (array/allclose? (:weight fixed) (:score t))])
;=> [true 0.0 true]
```

A constraint must name a choice the run makes. With `:a` false, `foo`
never makes `:b`, so a constraint on `:b` is an error, not a value
silently ignored:

<!-- code: unvisited -->
```clojure
;; error: generate: the constraint at [:b] was not visited
(gfi/generate foo [0.3] (cm/choicemap :a false :b true) (rng/key 3))
```

## The weight, precisely

`generate` draws the unconstrained choices from the model's *internal
proposal* $q$: it runs the model forward, keeping each constrained choice
at its value and drawing each other choice from its distribution. Here
that means drawing `:b` from `bernoulli(0.6)`, so `generate` makes either
{a, b, ¬c}, with $q = 0.6$, or {a, ¬b, ¬c}, with $q = 0.4$. The weight is

$$
\log \frac{p(t; x)}{q(t; x, \sigma)},
$$

the probability of the trace under the model over the probability that
the internal proposal made it, given the constraints $\sigma$. Keys 2 and
3 make the two traces:

<!-- code: two-runs -->
```clojure
(def two-runs (mapv #(gfi/generate foo [0.3] constraints (rng/key %)) [2 3]))

(mapv #((:trace %) :b) two-runs)
;=> [false true] ; float64

(mapv :weight two-runs)
;=> [-1.427 -3.507] ; float64
```

These are $\log(0.096 / 0.4) = \log 0.24$ and $\log(0.018 / 0.6) = \log
0.03$, and Gen.jl's tutorial prints the second as −3.506557897319982. From
the table, by hand:

<!-- code: weight-by-hand -->
```clojure
(defn weight-by-hand
  "log p(t) - log q(t) for a trace t of foo under the constraints: the
   internal proposal q drew only :b, from bernoulli(0.6)."
  [t]
  (let [q (if (array/truthy? (t :b)) 0.6 0.4)]
    (- (array/log (probability (:choices t) 0.3)) (Math/log q))))

(every? #(array/allclose? (:weight %) (weight-by-hand (:trace %))) two-runs)
;=> true
```

The probability of `:b` cancels, being a factor of both $p$ and $q$. What
remains is the probability of the constrained choices given the values
the run drew: 0.3 · 0.8 when `:b` is false, 0.3 · 0.1 when it is true.
`gfi/project` computes that. Given a trace and a *selection* of
addresses, it returns the part of the score that the selected choices
contribute:

<!-- code: weight-as-project -->
```clojure
(def constrained (sel/from-choicemap constraints))

(every? #(array/allclose? (:weight %) (gfi/project (:trace %) constrained))
        two-runs)
;=> true
```

So the weight measures how well a run explains the observations. A run
with `:b` true makes the observed ¬c unlikely, and its weight is eight
times smaller.

Continuous choices work the same way, with densities for probabilities.
A sensor reads an unknown quantity $\mu$ four times, with noise of
standard deviation 1, and before the readings we believe $\mu \sim
\mathcal{N}(0, 2)$:

<!-- code: sensor -->
```clojure
(def sensor
  (gen [n]
    (let [mu (trace :mu (d/normal 0 2))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(defn observations
  "The choice map fixing each [addr i] to (xs i)."
  [addr xs]
  (apply cm/choicemap (mapcat (fn [i x] [[addr i] x]) (range) xs)))

(def readings [2.3 1.6 2.9 2.2])
```

Constrained by the readings, `generate` draws $\mu$ from its prior, and
its weight is the sum of the readings' log densities at that $\mu$:

<!-- code: sensor-weight -->
```clojure
(let [{:keys [trace weight]}
      (gfi/generate sensor [4] (observations :y readings) (rng/key 6))
      mu (trace :mu)]
  (array/allclose? weight
                   (reduce + (map #(dc/log-prob (d/normal mu 1) %) readings))))
;=> true
```

Since the score is a sum over choices, the projections onto a selection
and onto its complement add up to it (*Choice Maps, Traces and
Selections* covers selections):

<!-- code: partition -->
```clojure
(let [t (gfi/simulate sensor [4] (rng/key 7))
      s (sel/select :mu)]
  (array/allclose? (:score t)
                   (+ (gfi/project t s) (gfi/project t (sel/complement s)))))
;=> true
```

## Importance sampling by hand

A weight scores one run; many runs, with their weights, estimate the
posterior. Run `generate` $n$ times. Each run draws its unconstrained
choices from the internal proposal, and its weight says how well they
explain the observations, so a weighted average over the runs estimates
an average under the posterior. This is *importance sampling*, with the
internal proposal as its proposal.

The weights are logs, and the exponential of a very negative one
underflows to 0. Subtracting the largest log weight first, the
*max-shift*, scales every weight by the same factor, which cancels when
they are normalized:

<!-- code: underflow -->
```clojure
(zero? (Math/exp -1000.0))
;=> true

(Math/exp (- -1001.0 -1000.0))
;=> 0.3679

(defn importance-by-hand
  "n traces of model on args under the observations, their normalized
   weights, and the log of their average weight: {:traces :weights :log-ml}."
  [model args observations n key]
  (let [samples (mapv #(gfi/generate model args observations %)
                      (rng/split key n))
        log-ws (mapv #(array/item (:weight %)) samples)
        top (apply max log-ws)
        ws (mapv #(Math/exp (- % top)) log-ws)
        total (reduce + ws)]
    {:traces (mapv :trace samples)
     :weights (mapv #(/ % total) ws)
     :log-ml (+ top (Math/log (/ total n)))}))
```

The log of the average weight, `:log-ml`, adds the shift back; what it
estimates comes below. With normalized weights $w_i$ and values $f_i$,
the estimate of $f$'s posterior mean is $\hat f = \sum_i w_i f_i$, with
standard error about $\sqrt{\sum_i w_i^2 (f_i - \hat f)^2}$:

<!-- code: estimate -->
```clojure
(defn estimate
  "The weighted average of f over traces with normalized weights (equal
   weights when none are given), and its standard error: {:mean :se}."
  ([f traces]
   (estimate f traces (vec (repeat (count traces) (/ 1.0 (count traces))))))
  ([f traces weights]
   (let [xs (mapv #(array/item (f %)) traces)
         mean (reduce + (map * weights xs))
         sq (fn [p x] (let [e (* p (- x mean))] (* e e)))]
     {:mean mean :se (Math/sqrt (reduce + (map sq weights xs)))})))
```

Gen.jl's tutorial asks: given that `:c` is false, how probable is it that
`:a` was true? The table answers exactly. The rows with ¬c sum to
$p(\neg c) = 0.018 + 0.096 + 0.07$, and those with a among them to
0.114. Two thousand weighted runs agree, within five standard errors:

<!-- code: posterior-a -->
```clojure
(def given-not-c (cm/choicemap :c false))

(def p-not-c (let [[_ tt _ tf _ f] table] (+ tt tf f)))

(def exact-a (let [[_ tt _ tf] table] (/ (+ tt tf) p-not-c)))

[p-not-c exact-a]
;=> [0.184 0.6196]

(def by-hand (importance-by-hand foo [0.3] given-not-c 2000 (rng/key 8)))

(defn a-true
  "1 when the trace's :a is true, 0 when it is false."
  [t]
  (if (array/truthy? (t :a)) 1.0 0.0))

(def a-estimate (estimate a-true (:traces by-hand) (:weights by-hand)))

a-estimate
;=> {:mean 0.612 :se 0.0146} ; float64

(< (abs (- (:mean a-estimate) exact-a)) (* 5 (:se a-estimate)))
;=> true
```

Observing ¬c raised the probability of a from its prior 0.3 to 0.62: of
the three kinds of run, {a, ¬b} explains ¬c best, with weight 0.8
against 0.1 for the other two.

The average weight estimates something too. Under the internal proposal,
the expected weight (in exp space) is $\sum_t q(t) \cdot p(t)/q(t) =
\sum_t p(t)$, over the traces the constraints allow: the probability of
the observations themselves, here $p(\neg c) = 0.184$. This is the
*marginal likelihood*, the probability the model gives the observations
with every other choice averaged out. The standard error of its log
estimate is about $\sqrt{1/\mathrm{ESS} - 1/n}$, where $\mathrm{ESS} =
1/\sum_i w_i^2$ is the *effective sample size*: $n$ for equal weights, 1
when one sample carries all the weight. (The weights' variance over their
squared mean is $n/\mathrm{ESS} - 1$, and the average of $n$ weights has
an $n$th of it.)

<!-- code: log-ml -->
```clojure
(defn log-ml-se
  "The standard error of the log marginal likelihood estimate of samples
   with these normalized weights: about sqrt(1/ESS - 1/n)."
  [weights]
  (let [ess (/ 1.0 (reduce + (map * weights weights)))]
    (Math/sqrt (- (/ 1.0 ess) (/ 1.0 (count weights))))))

(< (abs (- (:log-ml by-hand) (Math/log p-not-c)))
   (* 5 (log-ml-se (:weights by-hand))))
;=> true
```

Gen.jl's tutorial goes on to draw one of the $n$ traces in proportion to
their weights, which turns importance sampling into a sampler. That is
*importance resampling*, which the library provides as
`imp/importance-resampling`, beside the importance sampler below.

## The marginal likelihood

The sensor's posterior is known exactly. The prior and the noise are
normal, so the posterior is normal, with precision $1/2^2 + 4$ (one for
each reading) and mean $\sum_i y_i$ over the precision. The marginal
likelihood follows from Bayes' rule, $p(\mu \mid y) = p(\mu, y) / p(y)$,
turned around: $\log p(y) = \log p(\mu, y) - \log p(\mu \mid y)$, at any
$\mu$. `assess` gives the joint density, and the exact posterior the
other term. Getting the same value at every $\mu$ checks the posterior
too, since a wrong one gives a different value at each:

<!-- code: sensor-exact -->
```clojure
(defn sensor-posterior
  "The exact posterior of mu given readings ys: a normal whose precision is
   the prior's, 1/2^2, plus one for each reading."
  [ys]
  (let [precision (+ (/ 1.0 4.0) (count ys))]
    (d/normal (/ (reduce + ys) precision) (Math/sqrt (/ 1.0 precision)))))

(:params (sensor-posterior readings))
;=> {:mu 2.1176 :std 0.4851}

(defn log-ml-by-bayes
  "log p(observations) by Bayes' rule at a value v of the choice at addr,
   given that choice's exact posterior:
   log p(v, observations) - log p(v | observations)."
  [model args observations addr posterior v]
  (- (:weight (gfi/assess model args (cm/set-choice observations addr v)))
     (dc/log-prob posterior v)))

(defn sensor-log-ml
  "log p(readings), by Bayes' rule at mu."
  [mu]
  (log-ml-by-bayes sensor [4] (observations :y readings)
                   :mu (sensor-posterior readings) mu))

[(sensor-log-ml 0.0) (sensor-log-ml 3.0)]
;=> [-6.113 -6.113]
```

The library's sampler, `imp/importance-sampling`, takes the model, its
arguments, the observations, the number of samples and options. It
returns the traces, their normalized log weights and the log marginal
likelihood estimate, as Gen.jl's `importance_sampling` does. With
`:batched? true`, the samples run as one batched execution, the model's
code running once on arrays of every sample's values; `imp/traces` gives
the traces either way. Its estimate is within five standard errors of the
exact value, and its ten thousand samples are worth far fewer, about a
fifth as many:

<!-- code: built-in -->
```clojure
(def result
  (imp/importance-sampling sensor [4] (observations :y readings) 10000
                           {:key (rng/key 12) :batched? true}))

(defn probabilities
  "The normalized weights of an importance-sampling result."
  [result]
  (mapv #(Math/exp %) (:log-normalized-weights result)))

(def sensor-traces (imp/traces result))

(def sensor-weights (probabilities result))

(:log-ml-estimate result)
;=> -6.067 ; float64

(< (abs (- (:log-ml-estimate result) (sensor-log-ml 0.0)))
   (* 5 (log-ml-se sensor-weights)))
;=> true

(w/ess (:log-normalized-weights result))
;=> 1968.4 ; float64

(< (w/ess (:log-normalized-weights result)) 5000)
;=> true
```

## Prior against posterior

The internal proposal draws $\mu$ from its prior, so the samples' values
of $\mu$, unweighted, are draws from the prior. Weighted, the same values
estimate the posterior. Their averages show conditioning at work: the
prior's mean is 0, and the readings pull the posterior's to 2.12.

<!-- code: prior-vs-posterior -->
```clojure
(def prior-mu (estimate #(% :mu) sensor-traces))

(def posterior-mu (estimate #(% :mu) sensor-traces sensor-weights))

[(< (abs (:mean prior-mu)) (* 5 (:se prior-mu)))
 (< (abs (- (:mean posterior-mu) (:mu (:params (sensor-posterior readings)))))
    (* 5 (:se posterior-mu)))]
;=> [true true]
```

<!-- code: prior-posterior-figure -->
```clojure
;; figure: figures/013-prior-posterior.svg
(let [mus (mapv #(% :mu) sensor-traces)
      density (fn [dist] #(array/exp (dc/log-prob dist %)))]
  (svg/histogram
   [{:values mus :label "prior" :density (density (d/normal 0 2))}
    {:values mus :weights sensor-weights :label "posterior"
     :density (density (sensor-posterior readings))}]
   {:range [-6 6] :bins 60 :x-label "mu"}))
```

![The same draws of $\mu$, unweighted (the prior) and weighted (the
posterior), with the exact densities](figures/013-prior-posterior.svg)

The posterior's standard deviation is about a quarter of the prior's,
0.485 against 2, so most draws from the prior land where the readings are
improbable and carry almost no weight. That is why the effective sample
size is a fifth of $n$. The more the data narrow the posterior, the more
draws from the prior are wasted; exercise 3 shows the extreme, and Part
II's algorithms are the remedy.

## Rejection sampling

Importance sampling keeps every run and weights it. For discrete
observations there is a more literal way to condition: run the model
forward and keep only the runs whose choices match the observations. The
runs kept are exact draws from the posterior, and the fraction kept
estimates the marginal likelihood. Gen.jl's bottom-up introduction begins
this way, answering questions about a program by filtering its runs.

The GenMLX book's coin has a bias $p \sim \mathrm{Beta}(2, 2)$ and three
flips, observed as heads, heads, tails. A $\mathrm{Beta}(a, b)$ prior and
$h$ heads in $n$ flips make a $\mathrm{Beta}(a + h, b + n - h)$
posterior, here $\mathrm{Beta}(4, 3)$, with mean 4/7. Bayes' rule, as for
the sensor, gives the marginal likelihood, exactly 0.1:

<!-- code: coin -->
```clojure
(def coin
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (dotimes [i n]
        (trace [:flip i] (d/bernoulli p)))
      p)))

(def flips [true true false])

(defn coin-posterior
  "The exact posterior of the coin's p: beta(2 + heads, 2 + tails)."
  [flips]
  (let [heads (count (filter true? flips))]
    (d/beta (+ 2 heads) (+ 2 (- (count flips) heads)))))

(defn coin-ml
  "p(flips), by Bayes' rule at p."
  [flips p]
  (array/exp (log-ml-by-bayes coin [(count flips)] (observations :flip flips)
                              :p (coin-posterior flips) p)))

(every? #(array/allclose? 0.1 (coin-ml flips %)) [0.2 0.7])
;=> true
```

Rejection sampling, by hand, keeps a fraction within five standard
errors of 0.1, and the kept runs' mean bias is within five of 4/7:

<!-- code: rejection -->
```clojure
(defn agrees?
  "Does trace t hold the observations' values?"
  [t observations]
  (every? #(= (array/->clj (cm/get-choice (:choices t) %))
              (array/->clj (cm/get-choice observations %)))
          (cm/addresses observations)))

(defn rejection
  "The traces, among n simulations of model on args, that agree with the
   observations."
  [model args observations n key]
  (into [] (comp (map #(gfi/simulate model args %))
                 (filter #(agrees? % observations)))
        (rng/split key n)))

(def kept (rejection coin [3] (observations :flip flips) 5000 (rng/key 13)))

(count kept)
;=> 513 ; float64

(let [rate (/ (count kept) 5000.0)
      {:keys [mean se]} (estimate #(% :p) kept)]
  [(< (abs (- rate 0.1)) (* 5 (Math/sqrt (/ (* 0.1 0.9) 5000))))
   (< (abs (- mean (/ 4.0 7.0))) (* 5 se))])
;=> [true true]
```

Rejection threw away nine runs in ten, and each further observed flip
throws away more (exercise 2). Importance sampling on the same coin
agrees with both exact answers (exercise 1): rejection, importance
sampling and the exact formulas compute one posterior, the one
`generate`'s weight defines. The next chapter, *The Handler Loop*,
opens up `simulate` and `generate` to show the code that reads one model
in these two ways.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **One sample's marginal likelihood.** With one sample, the log marginal
   likelihood estimate is the weight of one `generate`. For the coin,
   predict the average of many such estimates. Is it $\log 0.1$? (Under
   $\mathrm{Beta}(a, b)$, the mean of $\log p$ is $\psi(a) - \psi(a +
   b)$, and that of $\log(1 - p)$ is $\psi(b) - \psi(a + b)$, where $\psi$
   is the digamma function.)
2. **Rejection gets hopeless.** Observe six flips: heads, heads, tails,
   heads, heads, heads. Predict the fraction of runs rejection keeps, then
   run it.
3. **A prior far from the data.** Add 6 to every reading. Predict the
   effective sample size of 10000 importance samples, and whether the
   estimate of the posterior mean, with its standard error, can be
   trusted.

## Solutions

**1.** The weight is $2 \log p + \log(1 - p)$, for two heads and a tail.
Under the prior $\mathrm{Beta}(2, 2)$ its mean is $3(\psi(2) - \psi(4)) =
-2.5$, below $\log 0.1 \approx -2.303$. The average weight is unbiased
for $p(\text{flips})$, but the log of the average is biased low, since
the log of an average exceeds the average of the logs (Jensen's
inequality). The bias shrinks as the number of samples grows: with 5000,
it is lost within the standard error, and the estimate agrees with
$\log 0.1$, as the weighted mean of $p$ does with 4/7.

<!-- code: solution-1 -->
```clojure
(* 3 (- (array/digamma 2.0) (array/digamma 4.0)))
;=> -2.5

(let [one-sample #(:weight (gfi/generate coin [3] (observations :flip flips) %))
      estimates (mapv one-sample (rng/split (rng/key 18) 3000))
      {:keys [mean se]} (estimate identity estimates)]
  [(< (abs (- mean -2.5)) (* 5 se))
   (< mean (- (Math/log 0.1) (* 5 se)))])
;=> [true true]

(let [r (imp/importance-sampling coin [3] (observations :flip flips) 5000
                                 {:key (rng/key 14) :batched? true})
      ws (probabilities r)
      {:keys [mean se]} (estimate #(% :p) (imp/traces r) ws)]
  [(< (abs (- (:log-ml-estimate r) (Math/log 0.1))) (* 5 (log-ml-se ws)))
   (< (abs (- mean (/ 4.0 7.0))) (* 5 se))])
;=> [true true]
```

**2.** $p(\text{flips}) = B(7, 3) / B(2, 2) = 1/42$: one run in 42 is
kept, four times fewer than with three flips. Each further flip divides
the fraction again, so rejection sampling is hopeless for more than a
handful of discrete observations.

<!-- code: solution-2 -->
```clojure
(def six [true true false true true true])

(array/allclose? (/ 1.0 42.0) (coin-ml six 0.5))
;=> true

(let [n 5000
      kept (rejection coin [6] (observations :flip six) n (rng/key 19))
      p (/ 1.0 42.0)]
  (< (abs (- (/ (count kept) (double n)) p))
     (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))))
;=> true
```

**3.** Readings near 8.25 lie four prior standard deviations from 0, so
almost no draw from the prior comes near the posterior, and the few
nearest carry all the weight. The effective sample size collapses, and
the estimate cannot be trusted. Neither can its standard error, computed
from the same few weights: in this run the estimate is 7.21, twelve of
its standard errors from the exact 7.765.

<!-- code: solution-3 -->
```clojure
(def far-readings (mapv #(+ % 6.0) readings))

(def far
  (imp/importance-sampling sensor [4] (observations :y far-readings) 10000
                           {:key (rng/key 20) :batched? true}))

(w/ess (:log-normalized-weights far))
;=> 4.13 ; float64

(< (w/ess (:log-normalized-weights far)) 50)
;=> true

[(:mu (:params (sensor-posterior far-readings)))
 (estimate #(% :mu) (imp/traces far) (probabilities far))]
;=> [7.765 {:mean 7.205 :se 0.045}] ; float64
```

A proposal that looks at the readings does better: that is where Part II
begins.
