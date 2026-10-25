# Inference as Operations on the GFI

*The Generative Function Interface* defined the operations of generative
functions and traces: what `simulate`, `generate`, `update`, `regenerate`,
`assess`, `project` and `propose` compute, and what each weight means.
This chapter puts them to work. It takes the inference algorithms of Part
II (Monte Carlo, importance sampling, a trained proposal,
Metropolis–Hastings, Hamiltonian Monte Carlo, particle filtering and
annealed importance sampling) and writes each in a few lines over those
operations. None of them looks inside a model. Each asks the model for
traces and weights, and does arithmetic on the answers.

The idea is Cusumano-Towner's, from the thesis that defined the interface:
the operations play the role for inference that automatic differentiation
plays for learning (thesis §1.1). A model is written once, its operations
come from it, and an algorithm written once against the operations runs on
every model.

This chapter ports the thesis's chapter 3, *Implementing Inference Using
Generative Functions and Traces*, and the GenMLX book's *Inference as ADT
Operations*. The algorithms carry the thesis's numbers, and its examples
appear where they have exact answers. Every algorithm is checked: against
an exact answer, by enumeration or a conjugate closed form, or against the
gen.clj library function that implements it, draw for draw with the same
keys. In two places the thesis's pseudocode, as printed, would give wrong
answers; the text points them out and checks both versions.

The thesis writes operations as methods of generative functions and
traces. In gen.clj they are functions of `gen.gfi`, each with an explicit
key:

| Thesis | gen.clj |
|--------|---------|
| $\mathcal{P}$.SIMULATE($x$) | `(gfi/simulate P x key)`: a trace |
| $\mathcal{P}$.GENERATE($x, \sigma$) | `(gfi/generate P x σ key)`: `{:trace :weight}` |
| $\mathbf{t}$.UPDATE($x', \delta, \sigma$) | `(gfi/update t x' δ σ key)`: `{:trace :weight :retdiff :discard}` |
| $\mathbf{t}$.CHOICES(), $\mathbf{t}$.LOGPDF() | `(:choices t)`, `(:score t)` |
| $\mathbf{t}[a]$ | `(t a)` |
| $\mathbf{t}$.GRADIENT($A$, 0) | `(grad/choice-gradients t A)` |
| $\mathcal{Q}$.SIMULATE, then $\mathbf{s}$.LOGPDF() | `(gfi/propose Q x key)`: `{:choices :weight :retval}` |
| $\mathcal{Q}$.GENERATE($x, \sigma$), every choice in $\sigma$ | `(gfi/assess Q x σ)`: `{:weight :retval}` |

<!-- code: ns -->
```clojure
(ns book.inference-as-operations
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.diagnostics :as diag]
            [gen.inference.importance :as imp]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## Monte Carlo is simulate

*Simple Monte Carlo* estimates the expectation of a *test function* $g$
under a distribution $p$ by averaging $g$ over samples from $p$ (the
thesis's Eq. 3.1):

$$
\mathbb{E}_{\tau \sim p}[g(\tau)] \approx \frac{1}{n} \sum_{i=1}^{n}
g(\tau^{(i)}), \qquad \tau^{(i)} \overset{\text{iid}}{\sim} p.
$$

A generative function is a distribution on choice maps, and `simulate`
samples it, so Algorithm 1 is a loop over `simulate`. We use the program
of the thesis's importance-sampling example (§3.2.1), whose expectation is
known. It flips a fair coin at each step and, while the coin comes up
true, draws an $x$ near the step's number. The test function sums the
squares of the $x$'s:

<!-- code: squares -->
```clojure
(def squares
  (gen []
    (loop [i 0]
      (when (trace [:go i] (d/bernoulli 0.5))
        (trace [:x i] (d/normal (inc i) 1))
        (recur (inc i))))))

(defn g
  "The test function: the sum of the squares of the trace's x's."
  [t]
  (loop [i 0 total 0.0]
    (if (t [:go i])
      (recur (inc i) (+ total (Math/pow (array/item (t [:x i])) 2)))
      total)))
```

The number of $x$'s is random, so traces differ in their addresses: this
model has stochastic structure. Addresses count from 0, so `[:x 0]` is
the first $x$, with mean 1. As the thesis recommends, the test function
lives outside the model, so the same traces serve any test function.

<!-- code: monte-carlo -->
```clojure
(defn simple-monte-carlo
  "Algorithm 1's terms: (f t) for n traces t simulated from model on args.
   Their mean is the estimate."
  [model args f n key]
  (mapv #(f (gfi/simulate model args %)) (rng/split key n)))

(defn mean-se
  "The mean of the numbers xs, and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sq (reduce + (map #(Math/pow (- % mean) 2) xs))]
    {:mean mean :se (Math/sqrt (/ sq (* n (dec n))))}))

(def mc-terms (simple-monte-carlo squares [] g 4000 (rng/key 1)))

(mean-se mc-terms)
;=> {:mean 7.37 :se 0.476} ; float64
```

The exact answer is a sum over the number $k$ of $x$'s. A run with $k$
of them has probability $0.5^{k+1}$, and the $i$-th $x$ (from 1) has
$\mathbb{E}[x^2] = i^2 + 1$, so

$$
\mathbb{E}[g] = \sum_{i \ge 1} 0.5^i\,(1 + i^2) = 1 + 6 = 7.
$$

The same sum gives the standard deviation of one term, 27.8, so $n$ terms
have a standard error of $27.8/\sqrt{n}$. `exact-moments` sums both over
$k$, for a program that goes on with probability $c$; the estimate is
within five standard errors of 7:

<!-- code: exact-squares -->
```clojure
(defn exact-moments
  "The mean and standard deviation of one term g(t) p(t)/q(t), t drawn
   from the program that goes on with probability c (squares, for
   c = 0.5), summed over the number k of x's: k has probability
   c^k (1 - c), the weight is (0.5/c)^k (0.5/(1 - c)), and the x_i are
   independent N(i, 1), with E[x^2] = i^2 + 1 and Var[x^2] = 4i^2 + 2."
  [c]
  (let [term (fn [k]
               (let [indices (range 1 (inc k))
                     mean-g (reduce + 0.0 (map #(+ (* % %) 1.0) indices))
                     var-g (reduce + 0.0 (map #(+ (* 4.0 % %) 2.0) indices))
                     prob (* (Math/pow c k) (- 1.0 c))
                     weight (* (Math/pow (/ 0.5 c) k) (/ 0.5 (- 1.0 c)))]
                 [(* prob weight mean-g)
                  (* prob weight weight (+ var-g (* mean-g mean-g)))]))
        [m1 m2] (reduce (fn [[a b] [x y]] [(+ a x) (+ b y)])
                        (map term (range 300)))]
    {:mean m1 :sd (Math/sqrt (- m2 (* m1 m1)))}))

(exact-moments 0.5)
;=> {:mean 7.0 :sd 27.82}

(defn within?
  "Is the mean of the terms within five standard errors, sd/sqrt(n), of the
   exact mean?"
  [terms {:keys [mean sd]}]
  (< (abs (- (:mean (mean-se terms)) mean))
     (* 5 (/ sd (Math/sqrt (count terms))))))

(within? mc-terms (exact-moments 0.5))
;=> true
```

## Importance sampling

### Regular importance sampling

*Regular importance sampling* draws from a *proposal* $q$ in place of
$p$, and corrects each sample by its *importance weight* $p/q$ (Eq. 3.2):

$$
\mathbb{E}_{\tau \sim p}[g(\tau)] \approx \frac{1}{n} \sum_{i=1}^{n}
g(\tau^{(i)})\, w^{(i)}, \qquad w^{(i)} =
\frac{p(\tau^{(i)})}{q(\tau^{(i)})}, \qquad \tau^{(i)}
\overset{\text{iid}}{\sim} q.
$$

The thesis's key idea is that a proposal is a generative function too.
Algorithm 2 simulates the proposal $\mathcal{Q}$, generates the model
$\mathcal{P}$ constrained by the proposal's choices, and weighs the
sample by $\exp(\mathbf{t}.\text{LOGPDF}() - \mathbf{s}.\text{LOGPDF}())$.
Every choice of the model is constrained, so `generate` draws nothing,
and the new trace's score is $\log p(\tau)$. The thesis's proposal goes
on with probability 0.7, so its traces are longer and reach the large
values of $g$ more often:

<!-- code: importance -->
```clojure
(def proposal
  (gen []
    (loop [i 0]
      (when (trace [:go i] (d/bernoulli 0.7))
        (trace [:x i] (d/normal (inc i) 1))
        (recur (inc i))))))

(defn regular-importance
  "Algorithm 2's terms: for each of n samples, the proposal simulated, the
   model generated with its choices, and (f t) weighted by p(t)/q(s).
   Their mean is the estimate."
  [model proposal f n key]
  (mapv (fn [k]
          (let [[k-q k-p] (rng/split k)
                s (gfi/simulate proposal [] k-q)
                t (:trace (gfi/generate model [] (:choices s) k-p))]
            (* (Math/exp (array/item (- (:score t) (:score s)))) (f t))))
        (rng/split key n)))

(def is-terms (regular-importance squares proposal g 4000 (rng/key 2)))

(mean-se is-terms)
;=> {:mean 7.08 :se 0.130} ; float64

(within? is-terms (exact-moments 0.7))
;=> true
```

Both estimators are unbiased. Importance sampling's terms have a standard
deviation of 8.32 against simple Monte Carlo's 27.8, so for the same
standard error it needs eleven times fewer samples:

<!-- code: variance-ratio -->
```clojure
(exact-moments 0.7)
;=> {:mean 7.0 :sd 8.322}

(/ (Math/pow (:sd (exact-moments 0.5)) 2)
   (Math/pow (:sd (exact-moments 0.7)) 2))
;=> 11.18
```

The running means of the two estimators show it. Simple Monte Carlo's
(orange) is still lurching after thousands of samples, each long run of
the coin a jump; importance sampling's (blue) settles near 7 early:

<!-- code: running-figure -->
```clojure
;; figure: figures/032-running-means.svg
(defn running-means
  "The points [i, the mean of the first i terms], every tenth i."
  [terms]
  (->> (reductions + terms)
       (map-indexed (fn [i s] [(inc i) (/ s (inc i))]))
       (take-nth 10)))

(svg/paths [{:paths [[[0 7.0] [4000 7.0]]] :color "gray" :width 1}
            {:paths [(running-means mc-terms)] :color "darkorange"}
            {:paths [(running-means is-terms)] :color "steelblue"}]
           {:y-range [3 11] :x-label "samples n" :y-label "estimate"})
```

![The running means of simple Monte Carlo (orange) and importance
sampling (blue) over 4000 samples, and the exact value
7](figures/032-running-means.svg)

The proposal must sample at the same addresses as the model, with values
of the same kind, and it must be able to make every trace the model can
(Eq. 3.3):

$$
p(\tau) > 0 \implies q(\tau) > 0.
$$

Neither Gen nor gen.clj checks this condition. The first exercise breaks
it, and the estimate converges to another number, without an error.

### Self-normalized importance sampling

Inference wants a conditional distribution: the latent choices $\sigma$
given the observations $\rho$, $p(\sigma \mid \rho) = p(\sigma \oplus
\rho)/\bar p(\rho)$. Its density cannot be evaluated, since the
*marginal likelihood* $\bar p(\rho)$ is unknown, but the joint density
$p(\sigma \oplus \rho)$ can. *Self-normalized importance sampling*
estimates $\bar p(\rho)$ from the same samples and divides by the estimate
(Eqs. 3.4 and 3.5):

$$
\mathbb{E}_{\sigma \sim p(\cdot \mid \rho)}[g(\sigma)] \approx \frac{1}{n}
\sum_{i=1}^{n} g(\sigma^{(i)})\, w^{(i)}, \qquad w^{(i)} =
\frac{p(\sigma^{(i)} \oplus \rho)/\hat z}{q(\sigma^{(i)})}, \qquad \hat z =
\frac{1}{n} \sum_{j=1}^{n} \frac{p(\sigma^{(j)} \oplus
\rho)}{q(\sigma^{(j)})}.
$$

Algorithm 3 keeps the weights as logs and normalizes them with NORMALIZE,
which subtracts the largest before exponentiating, so that nothing
overflows. Its normalized weights are Eq. 3.4's $w^{(i)}/n$, and they sum
to 1. As printed, Algorithm 3 takes NORMALIZE's log total,
$\log \sum_i \tilde w^{(i)}$, as $\log \hat z$; Eq. 3.5 makes $\hat z$ a
mean, so its log is the total less $\log n$. `log-ml` subtracts it, and
agrees with the library's `w/log-mean-exp`. Four weights of 10, 6, 3 and
1 check both by hand: their mean, 5, is $\hat z$, and the printed
algorithm's total is four times as large:

<!-- code: normalize -->
```clojure
(defn normalize
  "Algorithm 3's NORMALIZE of the log weights ls: the normalized weights,
   and log Σ exp(l), computed from the largest l so that no exp
   overflows. Loud when every weight is 0."
  [ls]
  (let [top (apply max ls)]
    (when (== top ##-Inf)
      (throw (ex-info "Every log weight is -Inf: no sample is possible"
                      {:log-weights ls})))
    (let [total (+ top (Math/log (reduce + (map #(Math/exp (- % top)) ls))))]
      {:weights (mapv #(Math/exp (- % total)) ls) :log-total total})))

(defn log-ml
  "Eq. 3.5's log z-hat of the log weights ls: the log of their mean, which is
   NORMALIZE's log total less log n."
  [ls]
  (- (:log-total (normalize ls)) (Math/log (count ls))))

(def toy-log-weights (mapv #(Math/log %) [10.0 6.0 3.0 1.0]))

[(:weights (normalize toy-log-weights)) (log-ml toy-log-weights)]
;=> [[0.5 0.3 0.15 0.05] 1.6094]

[(:log-total (normalize toy-log-weights)) (Math/log 20.0)]
;=> [2.9957 2.9957]

[(array/allclose? (log-ml toy-log-weights) (w/log-mean-exp toy-log-weights))
 (array/allclose? (array/array (:weights (normalize toy-log-weights)))
                  (array/exp (array/array (:log-normalized-weights
                                           (w/normalize toy-log-weights)))))]
;=> [true true]
```

The thesis also asks the converse of Eq. 3.3, $q(\sigma) > 0 \implies
p(\sigma \oplus \rho) > 0$ (Eq. 3.7), so that no weight is 0. When every
weight is 0 there is nothing to normalize, and `normalize` says so:

<!-- code: normalize-error -->
```clojure
;; error: Every log weight is -Inf
(normalize [##-Inf ##-Inf ##-Inf])
```

The thesis's example (p. 83) has an exact answer by enumeration. Two
coins $a$ and $b$ decide how likely a third, $c$, is to come up true, and
we observe $c$ true. The proposal `both` flips fair coins for $a$ and $b$;
`only-a` flips one for $a$ alone:

<!-- code: noisy-and -->
```clojure
(def noisy-and
  (gen []
    (let [a (trace :a (d/bernoulli 0.1))
          b (trace :b (d/bernoulli 0.2))]
      (trace :c (d/bernoulli (if (and a b) 0.9 0.1))))))

(def both
  (gen []
    (trace :a (d/bernoulli 0.5))
    (trace :b (d/bernoulli 0.5))))

(def only-a
  (gen []
    (trace :a (d/bernoulli 0.5))))

(def c-true (cm/choicemap :c true))
```

`assess` scores each of the four assignments of $a$ and $b$ with $c$
true. Their sum is $\bar p(\rho)$, and the posterior probability that $a$
is true is 0.224:

<!-- code: noisy-and-exact -->
```clojure
(defn joint
  "p(a, b, c = true) of the latent choices m, by assess."
  [m]
  (let [choices (cm/from-map (assoc m :c true))]
    (array/item (array/exp (:weight (gfi/assess noisy-and [] choices))))))

(def latent-maps
  [{:a true :b true} {:a true :b false} {:a false :b true} {:a false :b false}])

(mapv joint latent-maps)
;=> [0.018 0.008 0.018 0.072]

(def p-c (reduce + (map joint latent-maps)))

(def p-a (/ (+ (joint {:a true :b true}) (joint {:a true :b false})) p-c))

[p-c p-a]
;=> [0.116 0.2241]
```

`self-normalized` is Algorithm 3. The weight is a function, `by-score` by
default, for a reason that comes below. Each trace is replaced by the
value of the test function on it, so a long run keeps host numbers, not
traces:

<!-- code: self-normalized -->
```clojure
(defn by-score
  "Algorithm 3's log weight: the model trace's score less the proposal
   trace's."
  [t _ s]
  (- (:score t) (:score s)))

(defn self-normalized
  "Algorithm 3: n samples, each the proposal simulated on proposal-args and
   the model generated on args with its choices merged with the
   observations, weighed by (weigh t generate's-weight s), by-score by
   default. Each trace is replaced by (f t), so none is kept: {:values
   the (f t), :weights normalized, :log-ml log z-hat}."
  [model args observations {:keys [proposal proposal-args f weigh]
                            :or {proposal-args [] weigh by-score}}
   n key]
  (let [sample (fn [k]
                 (let [[k-q k-p] (rng/split k)
                       s (gfi/simulate proposal proposal-args k-q)
                       {t :trace wt :weight}
                       (gfi/generate model args
                                     (cm/merge (:choices s) observations) k-p)]
                   [(f t) (array/item (weigh t wt s))]))
        samples (mapv sample (rng/split key n))
        ls (mapv second samples)]
    {:values (mapv first samples)
     :weights (:weights (normalize ls))
     :log-ml (log-ml ls)}))

(defn indicator
  "1.0 when the trace's choice at addr is true, else 0.0."
  [addr]
  (fn [t] (if (t addr) 1.0 0.0)))
```

The self-normalized estimate has a standard error from the weights, by
the delta method. With `both`, the estimate of $P(a \mid c)$ is within
five of them of 0.224:

<!-- code: snis-estimate -->
```clojure
(defn estimate
  "The self-normalized estimate Σ w f of a result of self-normalized, and
   its standard error by the delta method, sqrt(Σ w^2 (f - estimate)^2)."
  [{:keys [values weights]}]
  (let [mean (reduce + (map * weights values))]
    {:mean mean
     :se (Math/sqrt (reduce + (map (fn [wi v] (Math/pow (* wi (- v mean)) 2))
                                   weights values)))}))

(defn near?
  "Is the estimate {:mean :se} within five standard errors of x?"
  [{:keys [mean se]} x]
  (< (abs (- mean x)) (* 5 se)))

(def by-both
  (self-normalized noisy-and [] c-true {:proposal both :f (indicator :a)}
                   2000 (rng/key 3)))

(estimate by-both)
;=> {:mean 0.2213 :se 0.0086} ; float64

(near? (estimate by-both) p-a)
;=> true
```

The estimate $\hat z$ of $\bar p(\rho)$ is unbiased, and its relative
standard error is about $\sqrt{1/\text{ESS} - 1/n}$, as *Conditioning*
derived:

<!-- code: snis-log-ml -->
```clojure
(defn relative-se
  "The relative standard error of z-hat, from the normalized weights:
   sqrt(Σ w^2 - 1/n), about sqrt(1/ESS - 1/n)."
  [{:keys [weights]}]
  (Math/sqrt (- (reduce + (map #(* % %) weights)) (/ 1.0 (count weights)))))

(defn unbiased?
  "Is the estimate z-hat of a result within five relative standard errors of
   the exact z?"
  [result z]
  (< (abs (- (Math/exp (- (:log-ml result) (Math/log z))) 1.0))
     (* 5 (relative-se result))))

[(:log-ml by-both) (Math/log p-c) (unbiased? by-both p-c)]
;=> [-2.142 -2.1542 true] ; float64
```

The proposal must also cover the posterior (Eq. 3.6):

$$
p(\sigma \oplus \rho) > 0 \implies q(\sigma) > 0.
$$

`only-a` breaks it: it never proposes $b$, so it gives no density to any
choice map that holds $b$, and the thesis calls it invalid. In gen.clj,
`generate` does not fail on its choices. It draws $b$ from the model's
*internal proposal*, here $b$'s prior, as it draws any unconstrained
choice. Algorithm 3's weight then counts $b$'s prior in the model's score
but not in the proposal's: with $q(a)$ the density of `only-a`, the
samples come from $q(a)\,p(b)$ and are weighed by $p(a)\,p(b)\,p(c \mid a,
b)/q(a)$, one factor $p(b)$ too many.
The estimate converges to the posterior of a model that weighs $b$'s prior
twice, 0.140, and is far from 0.224:

<!-- code: only-a -->
```clojure
(def by-a
  (self-normalized noisy-and [] c-true {:proposal only-a :f (indicator :a)}
                   2000 (rng/key 4)))

(def p-a-squared
  "The posterior probability of a that weighing b's prior twice targets:
   in proportion to p(a) Σ_b p(b)^2 p(c | a, b)."
  (let [r (fn [a] (* (if a 0.1 0.9)
                     (reduce + (map (fn [b] (* (Math/pow (if b 0.2 0.8) 2)
                                               (if (and a b) 0.9 0.1)))
                                    [true false]))))]
    (/ (r true) (+ (r true) (r false)))))

p-a-squared
;=> 0.1404

[(near? (estimate by-a) p-a) (near? (estimate by-a) p-a-squared)]
;=> [false true]
```

`generate`'s weight is the remedy, from the thesis's chapter 4 (§§4.1,
4.2). It is $\log p(\tau) - \log q(\upsilon; \sigma)$, where $\upsilon$
are the choices the internal proposal drew, so subtracting the proposal's
score from it gives the weight of the combined proposal: `only-a`'s
choices and the internal proposal's. With that weight `only-a` is valid:

<!-- code: by-weight -->
```clojure
(defn by-weight
  "The log weight from generate's weight, which subtracts the log density
   of the choices generate's internal proposal drew: generate's weight
   less the proposal trace's score."
  [_ wt s]
  (- wt (:score s)))

(def by-a-weight
  (self-normalized noisy-and [] c-true
                   {:proposal only-a :f (indicator :a) :weigh by-weight}
                   2000 (rng/key 4)))

[(near? (estimate by-a-weight) p-a) (unbiased? by-a-weight p-c)]
;=> [true true]
```

When the proposal makes every latent choice, `generate` draws nothing,
its weight is the trace's score, and the two weighings are the same
number:

<!-- code: by-weight-same -->
```clojure
(= (:weights by-both)
   (:weights (self-normalized noisy-and [] c-true
                              {:proposal both :f (indicator :a)
                               :weigh by-weight}
                              2000 (rng/key 3))))
;=> true
```

The library's `imp/importance-sampling` with a `:proposal` weighs this
way. With the same keys it makes the same estimate:

<!-- code: by-weight-library -->
```clojure
(array/allclose? (:log-ml by-a-weight)
                 (:log-ml-estimate (imp/importance-sampling
                                    noisy-and [] c-true 2000
                                    {:proposal only-a :key (rng/key 4)})))
;=> true
```

The rest of the chapter returns to the model of *Update and Regenerate*'s
last section: the mean $\mu$ of noisy measurements,

$$
\mu \sim \mathcal{N}(0, 1), \qquad y_i \mid \mu \sim \mathcal{N}(\mu,
\text{noise}),
$$

now with eight measurements, and the number of measurements and their
noise as arguments. Later sections change both arguments:

<!-- code: mean-model -->
```clojure
(def mean-model
  (gen [n noise]
    (let [mu (trace :mu (d/normal 0 1))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu noise)))
      mu)))

(def ys [1.1 2.3 0.7 1.9 1.4 2.8 1.2 1.6])

(def n-ys (count ys))

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))
```

The prior is conjugate. The posterior is normal with precision $1 +
n/\text{noise}^2$, and the measurements are jointly normal, each with
variance $1 + \text{noise}^2$ and covariance 1 between any two, so
$\bar p(\rho)$ is a multivariate normal density:

<!-- code: mean-exact -->
```clojure
(defn exact-posterior
  "The exact posterior of mu given ys with noise: a normal whose precision
   is 1 + n/noise^2 and whose mean is the data's sum over noise^2, over
   that precision."
  [ys noise]
  (let [precision (+ 1.0 (/ (count ys) (* noise noise)))]
    {:mean (/ (/ (reduce + ys) (* noise noise)) precision)
     :sd (Math/sqrt (/ 1.0 precision))}))

(defn exact-log-ml
  "log p(ys) with noise: the ys are jointly normal, each with variance
   1 + noise^2, and covariance 1."
  [ys noise]
  (let [n (count ys)]
    (array/item (dc/log-prob (d/mvnormal (array/zeros [n])
                                         (+ (* noise noise (array/eye n))
                                            (array/ones [n n])))
                             (array/array ys)))))

(def posterior (exact-posterior ys 1.0))

(def exact-z (Math/exp (exact-log-ml ys 1.0)))

[posterior (exact-log-ml ys 1.0)]
;=> [{:mean 1.4444 :sd 0.3333} -11.2612]
```

Self-normalized importance sampling with the prior of $\mu$ as the
proposal agrees with both, within five standard errors:

<!-- code: mean-snis -->
```clojure
(def mu-prior
  (gen []
    (trace :mu (d/normal 0 1))))

(def by-prior
  (self-normalized mean-model [n-ys 1.0] (observations ys)
                   {:proposal mu-prior :f #(array/item (% :mu))}
                   1000 (rng/key 5)))

[(near? (estimate by-prior) (:mean posterior)) (unbiased? by-prior exact-z)]
;=> [true true]
```

## Training a proposal

The prior is a poor proposal when the data move the posterior far from
it. A better one looks at the data. Section 3.3 of the thesis trains a
family of proposals $q(\sigma; \rho, \theta)$ on data simulated from the
model itself, by maximizing the expected log density of the latent
choices given the observations (Eq. 3.8):

$$
\max_\theta\ \mathbb{E}_{(\sigma \oplus \rho) \sim p}\bigl[\log q(\sigma;
\rho, \theta)\bigr],
$$

which is equivalent to minimizing the expected divergence from the
posterior to the proposal (Eq. 3.9):

$$
\min_\theta\ \mathbb{E}_{\rho \sim p}\bigl[D_{\text{KL}}\bigl(p(\cdot \mid
\rho)\ \|\ q(\cdot; \rho, \theta)\bigr)\bigr].
$$

`guess` is a data-driven proposal for $\mu$: a normal whose mean is
linear in the data's mean and whose standard deviation is $e^{\theta_3}$.
Its parameters $\theta$ are an argument, as Algorithm 4 has them; the
thesis marks such an argument `(grad)`, and gen.clj's `choice-gradients`
takes `:argnums`. The posterior has this form, with mean $n/(n + 1)$ times
the data's mean and variance $1/(n + 1)$, so the family contains it:

<!-- code: guess -->
```clojure
(def guess
  (gen [ys theta]
    (let [[a b log-sd] theta
          mean-y (/ (reduce + ys) (count ys))]
      (trace :mu (d/normal (+ (* a mean-y) b) (array/exp log-sd))))))

(def best-theta
  "The posterior's own parameters: mean (n/(n + 1)) times the data's mean,
   variance 1/(n + 1)."
  [(/ n-ys (+ n-ys 1.0)) 0.0 (* -0.5 (Math/log (+ n-ys 1.0)))])

best-theta
;=> [0.8889 0.0 -1.0986]
```

The two objectives differ by a constant. For each $\rho$, the expected
$\log q$ under the posterior is $-D_{\text{KL}}$ less the posterior's
entropy $H$, which here is the same for all data. Eq. 3.8's objective,
estimated by `simulate` and `assess` at an arbitrary $\theta$, agrees with
Eq. 3.9's divergence, computed in closed form for two normals, within five
standard errors; at the posterior's own parameters the divergence is 0:

<!-- code: objective -->
```clojure
(defn objective-terms
  "Eq. 3.8's terms: log q(mu; ys, theta) for m traces of the model, by
   assess."
  [theta m key]
  (mapv (fn [k]
          (let [[latents ys] (split-choices
                              (gfi/simulate mean-model [n-ys 1.0] k))]
            (array/item (:weight (gfi/assess guess [ys theta] latents)))))
        (rng/split key m)))

(defn expected-kl
  "Eq. 3.9's E over the model's data of KL(p(mu | ys) || q(mu; ys, theta)),
   both normal: the posterior has mean (n/(n + 1)) times the data's mean
   and variance s^2 = 1/(n + 1), and the data's mean has variance
   1 + 1/n."
  [[a b log-sd]]
  (let [s2 (/ 1.0 (+ n-ys 1.0))
        q2 (Math/exp (* 2 log-sd))
        miss (+ (* (Math/pow (- (/ n-ys (+ n-ys 1.0)) a) 2)
                   (+ 1.0 (/ 1.0 n-ys)))
                (* b b))]
    (- (+ (- log-sd (* 0.5 (Math/log s2))) (/ (+ s2 miss) (* 2 q2))) 0.5)))

(def posterior-entropy
  "The entropy of the posterior of mu, the same for all data."
  (* 0.5 (Math/log (/ (* 2 Math/PI Math/E) (+ n-ys 1.0)))))

(let [theta [0.5 0.2 -0.5]
      {:keys [mean se]} (mean-se (objective-terms theta 4000 (rng/key 7)))]
  (< (abs (- mean (- (- (expected-kl theta)) posterior-entropy))) (* 5 se)))
;=> true

(array/allclose? (expected-kl best-theta) 0.0 {:atol 1e-9})
;=> true
```

Algorithm 4 simulates the model, splits the trace's choices into latents
and observations, generates the proposal constrained by the latents, and
takes the gradient of its log density with respect to $\theta$. It
averages $m$ such gradients and steps uphill:

<!-- code: train -->
```clojure
(defn split-choices
  "Algorithm 4's split of a trace of mean-model into its latent choice mu,
   as a choice map, and its observations, as values."
  [t]
  [(cm/choicemap :mu (t :mu)) (mapv #(t [:y %]) (range n-ys))])

(defn theta-gradient
  "The gradient of log q(mu; ys, theta) at theta for one trace of the
   model: guess generated with the trace's mu, differentiated in its
   second argument."
  [theta key]
  (let [[k-sim k-gen] (rng/split key)
        [latents ys] (split-choices (gfi/simulate mean-model [n-ys 1.0] k-sim))
        s (:trace (gfi/generate guess [ys theta] latents k-gen))]
    (second (:arg-grads (grad/choice-gradients s sel/none nil
                                               {:argnums [1]})))))

(defn train
  "Algorithm 4, with its iterates: theta, then k steps of stochastic
   gradient ascent of step size alpha, each along the mean gradient of m
   traces of the model."
  [theta k m alpha key]
  (reductions (fn [theta step-key]
                (let [gs (mapv #(theta-gradient theta %) (rng/split step-key m))
                      u (apply mapv (fn [& xs] (/ (reduce + xs) m)) gs)]
                  (mapv (fn [x ux] (+ x (* alpha (array/item ux)))) theta u)))
              theta
              (rng/split key k)))
```

Three hundred steps of 20 traces each, from $\theta = 0$, land near the
posterior's parameters. A trained parameter is a learned function, so the
band comes from the spread of ten trainings with other keys:

<!-- code: trained -->
```clojure
(def path (vec (train [0.0 0.0 0.0] 300 20 0.02 (rng/key 8))))

(def trained-theta (peek path))

trained-theta
;=> [0.920 0.046 -1.081] ; float64

[(expected-kl (first path)) (expected-kl trained-theta)]
;=> [1.0986 0.0140] ; float64

;; each parameter's spread over ten trainings (keys 101 to 110):
;; [0.023 0.014 0.024]
(every? true? (map (fn [x best spread] (< (abs (- x best)) (* 5 spread)))
                   trained-theta best-theta [0.023 0.014 0.024]))
;=> true
```

The expected divergence, which the closed form gives at every step,
falls from 1.1 at the start to 0.014 at the end, most of the way within
the first hundred steps:

<!-- code: training-figure -->
```clojure
;; figure: figures/032-training.svg
(svg/series (mapv expected-kl path)
            {:x-label "step" :y-label "expected KL divergence"})
```

![The expected KL divergence from the posterior to the proposal along
Algorithm 4's 300 steps](figures/032-training.svg)

In self-normalized importance sampling, the trained proposal's 1000
samples are worth 925, against the prior's 160. Its estimates agree with
the exact answers too:

<!-- code: trained-snis -->
```clojure
(defn ess
  "The effective sample size of a result's normalized weights, 1/Σ w^2."
  [{:keys [weights]}]
  (/ 1.0 (reduce + (map #(* % %) weights))))

(def by-guess
  (self-normalized mean-model [n-ys 1.0] (observations ys)
                   {:proposal guess :proposal-args [ys trained-theta]
                    :f #(array/item (% :mu))}
                   1000 (rng/key 5)))

[(ess by-prior) (ess by-guess)]
;=> [159.5 925.2] ; float64

[(> (ess by-guess) 500)
 (near? (estimate by-guess) (:mean posterior))
 (unbiased? by-guess exact-z)]
;=> [true true true]
```

*Learning* trains proposals with the library's `lecture-batched`, which
keeps $\theta$ in a parameter store in place of an argument.

## Markov chains

A Markov chain needs no proposal for the whole posterior. It starts from
a trace consistent with the observations, $\mathbf{t}_0 =
\mathcal{P}.\text{GENERATE}(\upsilon_0 \oplus \rho)$, and applies a
*kernel* again and again, $\mathbf{t}_i = \text{KERN}(\mathbf{t}_{i-1})$
(thesis §3.4.1). A kernel that leaves $p(\cdot \mid \rho)$ unchanged, and
can reach all of it, makes a chain whose traces, after a burn-in, are
draws from the posterior.

### Metropolis–Hastings with a generative function as the proposal

Algorithm 5 makes a kernel from a proposal generative function
$\mathcal{Q}$ that takes the current trace as its argument. It proposes
choices $\sigma$, updates the trace with them, and gets back the new
trace $\mathbf{t}'$, the weight $\log w = \log p(\mathbf{t}') - \log
p(\mathbf{t})$ and the discard $\sigma'$, the choices the update
overwrote or dropped. The discard is the move back: proposing $\sigma'$
from $\mathbf{t}'$ restores $\mathbf{t}$. So the acceptance ratio is

$$
\log \alpha = \log w + \log q(\sigma'; \mathbf{t}') - \log q(\sigma;
\mathbf{t}).
$$

The thesis's example (p. 92) changes the trace's structure. The model
flips $a$, flips $b$ only when $a$ is true, and makes $c$ likely when
both are; we observe $c$ true. The proposal switches $a$ with probability
0.9, and draws a fresh $b$ when it switches $a$ to true:

<!-- code: switch-model -->
```clojure
(def switch-model
  (gen []
    (let [a (trace :a (d/bernoulli 0.5))
          z (and a (trace :b (d/bernoulli 0.5)))]
      (trace :c (d/bernoulli (if z 0.9 0.1))))))

(def switch-proposal
  (gen [t]
    (if (t :a)
      (trace :a (d/bernoulli 0.1))
      (when (trace :a (d/bernoulli 0.9))
        (trace :b (d/bernoulli 0.5))))))
```

The posterior over the three latent states is the thesis's table:

<!-- code: switch-exact -->
```clojure
(def states [{:a false} {:a true :b false} {:a true :b true}])

(defn state-trace
  "The trace of switch-model with the latent choices m and c true."
  [m]
  (:trace (gfi/generate switch-model [] (cm/from-map (assoc m :c true))
                        (rng/key 0))))

(def target
  "The posterior probability of each state, from the traces' scores."
  (let [ps (mapv #(array/item (array/exp (:score (state-trace %)))) states)]
    (mapv #(/ % (reduce + ps)) ps)))

target
;=> [0.1667 0.0833 0.75]
```

`mh-move` is Algorithm 5's move, and `mh-kernel` its kernel. `propose`
gives the forward choices with their log density, the thesis's
$\mathcal{Q}$.SIMULATE and $\mathbf{s}$.LOGPDF. `assess` scores the
backward choices; it is the thesis's $\mathcal{Q}$.GENERATE with every
choice given, and stricter: when $\sigma'$ is not a complete choice map of
$\mathcal{Q}$ on $\mathbf{t}'$, it fails loudly. Algorithm 5 asserts that
the discard holds no observed address, which would mean the proposal
changed the data; `mh-move` raises an error then:

<!-- code: mh-kernel -->
```clojure
(defn mh-move
  "Algorithm 5's move of the trace t by the proposal's choices sigma: the
   updated trace, the update's log weight, and log q(σ'; t'), the
   backward proposal's log density of the discard σ', by assess. Loud
   when σ' holds an observed address: the proposal changed the data.
   keys: the update's and assess's."
  [t proposal args sigma observed [k-update k-assess]]
  (let [{t' :trace log-w :weight sigma' :discard} (gfi/update t sigma k-update)
        changed (filter (set (cm/addresses observed)) (cm/addresses sigma'))]
    (when (seq changed)
      (throw (ex-info (str "The proposal changed the observations at "
                           (vec changed))
                      {:addresses (vec changed)})))
    {:trace t'
     :log-w log-w
     :log-q' (:weight (gfi/assess proposal (into [t'] args) sigma' k-assess))}))

(defn log-alpha
  "The log acceptance ratio of a move whose forward choices have log
   density log-q: log w + log q(σ'; t') - log q(σ; t)."
  [{:keys [log-w log-q']} log-q]
  (+ log-w (- log-q' log-q)))

(defn mh-kernel
  "Algorithm 5: the MH kernel of proposal, a generative function called on
   (t, args…), for a model conditioned on observed."
  [proposal args observed]
  (fn [t key]
    (let [[k-propose k-update k-accept k-assess] (rng/split key 4)
          {sigma :choices log-q :weight}
          (gfi/propose proposal (into [t] args) k-propose)
          move (mh-move t proposal args sigma observed [k-update k-assess])
          u (array/item (rng/uniform k-accept))]
      (if (< (Math/log u) (array/item (log-alpha move log-q)))
        (:trace move)
        t))))
```

The thesis lists every move of this proposal: from each latent state
$\upsilon$, each choice map $\sigma$ the proposal can make, the new
latent state $\upsilon'$ and the reverse choices $\sigma'$. `update`
makes the same table, $\sigma'$ as its discard:

<!-- code: switch-table -->
```clojure
(def moves
  "Every choice map switch-proposal can make from each state."
  {{:a false} [{:a false} {:a true :b false} {:a true :b true}]
   {:a true :b false} [{:a true} {:a false}]
   {:a true :b true} [{:a true} {:a false}]})

(defn latents
  "The latent choices of a trace of switch-model, as host values."
  [t]
  (dissoc (update-vals (cm/to-map (:choices t)) array/->clj) :c))

(defn proposal-prob
  "q(sigma; t) of the proposal, by assess."
  [proposal t sigma]
  (array/item (array/exp (:weight (gfi/assess proposal [t]
                                              (cm/from-map sigma))))))
```

<!-- code: table-check -->
```clojure
(defn table-row
  "The thesis's row for the move from state u by the choices sigma:
   [u σ u' σ'], u' the new trace's latent choices and σ' the discard."
  [u sigma]
  (let [{t' :trace sigma' :discard}
        (gfi/update (state-trace u) (cm/from-map sigma) (rng/key 0))]
    [u sigma (latents t') (update-vals (cm/to-map sigma') array/->clj)]))

(def thesis-table
  "The thesis's table of the moves of q4_mh (p. 93): [υ σ υ' σ']."
  [[{:a false} {:a false} {:a false} {:a false}]
   [{:a false} {:a true :b false} {:a true :b false} {:a false}]
   [{:a false} {:a true :b true} {:a true :b true} {:a false}]
   [{:a true :b false} {:a true} {:a true :b false} {:a true}]
   [{:a true :b false} {:a false} {:a false} {:a true :b false}]
   [{:a true :b true} {:a true} {:a true :b true} {:a true}]
   [{:a true :b true} {:a false} {:a false} {:a true :b true}]])

(= thesis-table
   (vec (for [[u sigmas] moves, sigma sigmas] (table-row u sigma))))
;=> true

(mapv (fn [[u sigmas]]
        (reduce + (map #(proposal-prob switch-proposal (state-trace u) %)
                       sigmas)))
      moves)
;=> [1.0 1.0 1.0]
```

The proposal's probabilities from each state sum to 1, so the table holds
every move, and the kernel's transition probabilities can be computed
exactly. A kernel leaves the posterior $\pi$ unchanged when one step from
$\pi$ gives $\pi$ again, $\pi K = \pi$. For Algorithm 5's kernel it does,
up to rounding:

<!-- code: transitions -->
```clojure
(defn transitions
  "The exact transition probabilities of the MH kernel of proposal from
   state u, {u' p}: for each choice map in (moves u), its probability
   times its acceptance probability min(1, exp(ratio)), the rest of the
   mass staying at u. ratio is a function of the move and log q, such as
   log-alpha."
  [ratio proposal moves u]
  (let [t (state-trace u)
        accepted (fn [sigma]
                   (let [q (proposal-prob proposal t sigma)
                         move (mh-move t proposal [] (cm/from-map sigma) c-true
                                       [(rng/key 0) (rng/key 0)])
                         la (array/item (ratio move (Math/log q)))]
                     {(latents (:trace move)) (* q (min 1.0 (Math/exp la)))}))
        moved (apply merge-with + (map accepted (moves u)))]
    (merge-with + moved {u (- 1.0 (reduce + (vals moved)))})))

(defn switch-row
  "The transitions of switch-proposal's MH kernel under ratio, a function
   of the state."
  [ratio]
  #(transitions ratio switch-proposal moves %))

(defn after-step
  "π K: the distribution over the states after one step from the target,
   (row u) the step's transitions from state u."
  [row]
  (let [rows (mapv row states)]
    (mapv (fn [v] (reduce + (map (fn [p r] (* p (get r v 0.0))) target rows)))
          states)))

(after-step (switch-row log-alpha))
;=> [0.1667 0.0833 0.75]

(array/allclose? (array/array (after-step (switch-row log-alpha)))
                 (array/array target))
;=> true
```

As printed, Algorithm 5 computes the acceptance probability as $\exp(\log
w - \mathbf{s}'.\text{LOGPDF}() + \mathbf{s}.\text{LOGPDF}())$, with the
two proposal terms' signs reversed. Algorithm 6's acceptance line has
them the right way round. The proposal here is asymmetric (it draws $b$
going one way, not the other), so the reversed signs move the chain away
from the posterior in one step:

<!-- code: printed -->
```clojure
(defn printed-log-alpha
  "Algorithm 5's ratio as the thesis prints it: log w - log q(σ'; t') +
   log q(σ; t)."
  [{:keys [log-w log-q']} log-q]
  (+ log-w (- log-q log-q')))

(after-step (switch-row printed-log-alpha))
;=> [0.4479 0.0271 0.525]
```

The template of §3.4.1 is a loop. `run-chain` applies a kernel $n$ times
and keeps a value of each trace, not the trace. The library's `mh/mh`
with a proposal is this kernel: with the same keys, it makes the same
chain, state for state:

<!-- code: run-chain -->
```clojure
(defn run-chain
  "§3.4.1's template: n steps of kernel from the trace t, t_i = (kernel
   t_i-1 key_i), keeping (f t_i), not the trace."
  [kernel t n f key]
  (first (reduce (fn [[out t] k]
                   (let [t' (kernel t k)] [(conj out (f t')) t']))
                 [[] t]
                 (rng/split key n))))

(def switch-chain
  (run-chain (mh-kernel switch-proposal [] c-true) (state-trace {:a false})
             3000 latents (rng/key 9)))

(= switch-chain
   (run-chain (fn [t k] (:trace (mh/mh t switch-proposal [] k)))
              (state-trace {:a false}) 3000 latents (rng/key 9)))
;=> true
```

The chain's frequencies of the three states agree with the posterior,
within five Monte Carlo standard errors:

<!-- code: switch-chain-check -->
```clojure
(defn agrees?
  "Is the mean of the draws within five Monte Carlo standard errors of x?"
  [draws x]
  (let [{:keys [mean mcse-mean]} (diag/summary draws)]
    (< (abs (- mean x)) (* 5 mcse-mean))))

(mapv (fn [state p]
        (agrees? (mapv #(if (= % state) 1.0 0.0) switch-chain) p))
      states target)
;=> [true true true]
```

The thesis's first invalid proposal (p. 93) samples $c$, an observed
address. Its update would change the data, and the move's discard shows
it:

<!-- code: observed-error -->
```clojure
;; error: The proposal changed the observations at [[:c]]
(def flip-c
    (gen [t]
      (trace :c (d/bernoulli 0.5))))
((mh-kernel flip-c [] c-true) (state-trace {:a false}) (rng/key 13))
```

The same kernel takes any proposal. A random walk on $\mu$ is symmetric,
and its chain agrees with the running model's posterior in mean and
standard deviation:

<!-- code: walk -->
```clojure
(def walk
  (gen [t step]
    (trace :mu (d/normal (t :mu) step))))

(def mean-start
  (:trace (gfi/generate mean-model [n-ys 1.0] (observations ys)
                        (rng/key 10))))

(def walk-draws
  (run-chain (mh-kernel walk [0.5] (observations ys)) mean-start 3000
             #(array/item (% :mu)) (rng/key 11)))

(let [{:keys [mean sd mcse-mean mcse-sd]}
      (diag/summary (subvec walk-draws 100))]
  [(< (abs (- mean (:mean posterior))) (* 5 mcse-mean))
   (< (abs (- sd (:sd posterior))) (* 5 mcse-sd))])
;=> [true true]
```

A *Gibbs move* proposes from the exact conditional of the selected choice
given the rest. As an Algorithm 5 proposal it is always accepted (the
thesis's `q_var_gibbs`): its log acceptance ratio is 0. The running
model's conditional of $\mu$ is its posterior given the trace's
arguments, which the proposal reads from the trace:

<!-- code: gibbs -->
```clojure
(def gibbs
  (gen [t]
    (let [[n noise] (:args t)
          total (reduce + (map #(t [:y %]) (range n)))
          precision (+ 1.0 (/ n (* noise noise)))]
      (trace :mu (d/normal (/ (/ total (* noise noise)) precision)
                           (array/sqrt (/ 1.0 precision)))))))

(let [{sigma :choices log-q :weight}
      (gfi/propose gibbs [mean-start] (rng/key 12))
      move (mh-move mean-start gibbs [] sigma (observations ys)
                    [(rng/key 0) (rng/key 0)])]
  (array/allclose? (log-alpha move log-q) 0.0 {:atol 1e-4}))
;=> true
```

### MH over regenerate

*Update and Regenerate* built a kernel from `regenerate`, which resamples
selected choices from the internal proposal and returns the acceptance
ratio as its weight. It is Algorithm 5 with the internal proposal as
$\mathcal{Q}$: for the value of $\mu$ that `regenerate` drew,
Algorithm 5's ratio with $\mu$'s prior as the proposal is `regenerate`'s
weight:

<!-- code: regenerate-is-mh -->
```clojure
(def prior-proposal
  (gen [t]
    (trace :mu (d/normal 0 1))))

(let [{t' :trace w :weight} (gfi/regenerate mean-start (sel/select :mu)
                                            (rng/key 12))
      sigma (cm/choicemap :mu (t' :mu))
      log-q (:weight (gfi/assess prior-proposal [mean-start] sigma))
      move (mh-move mean-start prior-proposal [] sigma (observations ys)
                    [(rng/key 0) (rng/key 0)])]
  (array/allclose? w (log-alpha move log-q) {:atol 1e-4}))
;=> true
```

The thesis's chapter 4 calls the kernel *selection MH* (§4.3). It is the
library's `mh/mh` with a selection, draw for draw:

<!-- code: regenerate-kernel -->
```clojure
(defn regenerate-kernel
  "MH over regenerate: resample the selection from the internal proposal,
   and accept with probability min(1, exp(weight))."
  [selection]
  (fn [t key]
    (let [[k-move k-accept] (rng/split key)
          {t' :trace w :weight} (gfi/regenerate t selection k-move)]
      (if (< (Math/log (array/item (rng/uniform k-accept))) (array/item w))
        t'
        t))))

(= (run-chain (regenerate-kernel (sel/select :mu)) mean-start 500
              #(array/item (% :mu)) (rng/key 13))
   (run-chain (fn [t k] (:trace (mh/mh t (sel/select :mu) k))) mean-start 500
              #(array/item (% :mu)) (rng/key 13)))
;=> true
```

### Hamiltonian Monte Carlo

Hamiltonian Monte Carlo is Algorithm 5 with a cleverer move (Algorithm
6). It draws a *momentum* for each selected address from a generative
function $\mathcal{Q}_{\text{HMC}}$, simulates a ball rolling on the
surface $-\log p$ with $L$ leapfrog steps (each a half step of the
momenta along the gradient, a step of the positions along the momenta,
and another half step), negates the momenta, and accepts by

$$
\log \alpha = \mathbf{t}_L.\text{LOGPDF}() - \mathbf{t}_0.\text{LOGPDF}() -
\mathbf{s}.\text{LOGPDF}() + \mathbf{s}'.\text{LOGPDF}(),
$$

$\mathbf{s}$ and $\mathbf{s}'$ the momenta before and after. It needs one
more operation, the gradient, which the thesis lists among the trace's
operations; `grad/choice-gradients` gives it. The model is *Getting
Started*'s line, whose posterior is a narrow, tilted ridge:

<!-- code: line-model -->
```clojure
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])

(def line-ys [8.23 5.87 3.99 2.59 0.23 -0.66 -3.53 -6.91 -7.24 -9.90])

(def line-exact
  "The exact posterior of the line: precision diag(1/4, 1/100) + X^T X,
   for X the rows (x, 1); its mean and standard deviations."
  (let [X (array/array (mapv (fn [x] [x 1.0]) xs))
        Xt (array/transpose X)
        cov (linalg/inv (+ (array/diag (array/array [0.25 0.01]))
                           (array/matmul Xt X)))]
    {:mean (array/->clj (array/matmul cov (array/matmul
                                           Xt (array/array line-ys))))
     :sd (array/->clj (array/sqrt (array/diag cov)))}))

line-exact
;=> {:mean [-1.979 10.143] :sd [0.1097 0.6807]}
```

$\mathcal{Q}_{\text{HMC}}$ is a generative function with a standard
normal at each address, as the thesis writes it:

<!-- code: momenta -->
```clojure
(def momenta
  (gen [addrs]
    (doseq [a addrs]
      (trace a (d/normal 0 1)))))

(defn gradient
  "The gradient of t's log density at the addresses addrs, {addr value}:
   the thesis's t.GRADIENT."
  [t addrs]
  (let [g (:gradients (grad/choice-gradients t (apply sel/select addrs)))]
    (into {} (map (fn [a] [a (cm/get-choice g a)])) addrs)))

(defn step-along
  "The map x moved by eps times the map v, address by address."
  [x eps v]
  (into {} (map (fn [[a xa]] [a (+ xa (* eps (v a)))])) x))
```

Each leapfrog step moves the positions with one `update`:

<!-- code: leapfrogs -->
```clojure
(defn leapfrogs
  "Algorithm 6's L leapfrog steps of size eps from the trace t with the
   momenta p, {addr value}: a half step of the momenta along the gradient,
   a full step of the positions along the momenta (one update), and a
   half step at the new positions. [t' p']."
  [t p addrs L eps key]
  (loop [t t, p p, g (gradient t addrs), ks (rng/split key L)]
    (if (empty? ks)
      [t p]
      (let [p-half (step-along p (/ eps 2) g)
            positions (step-along (into {} (map (fn [a] [a (t a)])) addrs)
                                  eps p-half)
            t' (:trace (gfi/update t (cm/from-map positions) (first ks)))
            g' (gradient t' addrs)]
        (recur t' (step-along p-half (/ eps 2) g') g' (rest ks))))))
```

The acceptance has Algorithm 5's form. The leapfrogs' change of score is
the update's weight, and the momenta are the proposal's choices, forward
and backward, so `hmc-kernel` accepts with `log-alpha`:

<!-- code: hmc-kernel -->
```clojure
(defn hmc-kernel
  "Algorithm 6: HMC on the addresses addrs, L leapfrog steps of size eps,
   the momenta a trace of the generative function momenta. Accepted on
   Algorithm 5's ratio, the leapfrogs' change of score as the update's
   weight and the momenta as the proposal's choices."
  [addrs L eps]
  (fn [t key]
    (let [[k-momenta k-steps k-accept] (rng/split key 3)
          s (gfi/simulate momenta [addrs] k-momenta)
          [t' p'] (leapfrogs t (into {} (map (fn [a] [a (s a)])) addrs)
                             addrs L eps k-steps)
          s' (gfi/assess momenta [addrs]
                         (cm/from-map (update-vals p' array/negate)))
          move {:trace t' :log-w (- (:score t') (:score t))
                :log-q' (:weight s')}]
      (if (< (Math/log (array/item (rng/uniform k-accept)))
             (array/item (log-alpha move (:score s))))
        t'
        t))))
```

Three hundred moves of 20 steps of 0.05, from a start drawn by
`generate`, agree with the exact posterior after the first 50:

<!-- code: hmc-chain -->
```clojure
(def line-start
  (:trace (gfi/generate line-model [xs] (observations line-ys) (rng/key 14))))

(defn line-of
  "The trace's line as host numbers, [slope intercept]."
  [t]
  [(array/item (t :slope)) (array/item (t :intercept))])

(def hmc-draws
  (subvec (run-chain (hmc-kernel [:slope :intercept] 20 0.05) line-start 300
                     line-of (rng/key 15))
          50))

(defn line-agrees?
  "Do the draws' mean and standard deviation of the slope and of the
   intercept agree with the exact ones, within five MCSE?"
  [draws]
  (vec (for [i [0 1]
             :let [{:keys [mean sd mcse-mean mcse-sd]}
                   (diag/summary (mapv #(% i) draws))]]
         [(< (abs (- mean ((:mean line-exact) i))) (* 5 mcse-mean))
          (< (abs (- sd ((:sd line-exact) i))) (* 5 mcse-sd))])))

(line-agrees? hmc-draws)
;=> [[true true] [true true]]
```

The library's `mcmc/hmc` makes the same move, from flat arrays of the
selected values; its chain passes the same checks:

<!-- code: hmc-library -->
```clojure
(def library-draws
  (subvec (run-chain (fn [t k]
                       (:trace (mcmc/hmc t (sel/select :slope :intercept)
                                         {:L 20 :eps 0.05} k)))
                     line-start 300 line-of (rng/key 16))
          50))

(line-agrees? library-draws)
;=> [[true true] [true true]]
```

Why is the energy test the right acceptance? The leapfrog steps followed
by the negation are an *involution*: applied twice, they give back the
start. Twenty steps from $(\mathbf{t}, p)$, then twenty from the end with
the momenta negated, return to $\mathbf{t}$ with momenta $-p$:

<!-- code: involution -->
```clojure
(let [p {:slope 0.3 :intercept -1.2}
      addrs [:slope :intercept]
      [t1 p1] (leapfrogs line-start p addrs 20 0.05 (rng/key 17))
      [t2 p2] (leapfrogs t1 (update-vals p1 array/negate) addrs 20 0.05
                         (rng/key 18))]
  [(array/allclose? (array/array (line-of t2))
                    (array/array (line-of line-start)) {:atol 1e-3})
   (array/allclose? (array/array (mapv (update-vals p2 array/negate) addrs))
                    (array/array (mapv p addrs)) {:atol 1e-3})])
;=> [true true]
```

A move that is its own inverse and keeps volume, as the leapfrog does,
needs only the density ratio of the two ends. *Trace Translators and
Involutive MCMC* makes this the general construction, with a Jacobian
where the volume changes.

## Populations of the same moves

The last two algorithms carry traces through a sequence of distributions,
with the moves above between the steps.

### Resample-move particle filtering

A *particle filter* keeps $n$ weighted traces, its *particles*, as data
arrive (Algorithm 7). It starts as importance sampling on the first
observation. At each later step it resamples the particles in proportion
to their weights, moves each with an MCMC kernel that leaves the current
posterior unchanged (*rejuvenation*), and extends each with `update` to
the next observation, whose weight is the particle's new weight. The
estimate $\log \hat z$ adds up each step's log mean weight.

`advance` is one particle's step, a kernel and then an update. A step may
only add choices, so a non-empty discard is an error, Algorithm 7's
assertion:

<!-- code: advance -->
```clojure
(defn advance
  "One particle's step in Algorithms 7 and 8: kernel moves the trace t at
   its own arguments, and update takes it to the step's arguments and
   observations. Loud when update discards a choice: a step only adds.
   {:trace :weight}, the weight a host number."
  [t kernel {:keys [args argdiffs observations]} k-kernel k-update]
  (let [{t' :trace w :weight discard :discard}
        (gfi/update (kernel t k-kernel) args argdiffs observations k-update)]
    (when-not (cm/empty? discard)
      (throw (ex-info "A step discarded choices" {:discard discard})))
    {:trace t' :weight (array/item w)}))
```

Resampling draws indices from host numbers; `particle-filter` uses the
library's multinomial resampler of `gen.inference.weights` on the
normalized log weights, as the library's filter does:

<!-- code: particle-filter -->
```clojure
(defn particle-filter
  "Algorithm 7: n traces of model generated on the first step's arguments
   with its observations; then at each later step, resampled in proportion
   to their weights, and each advanced with kernel. The final traces, their
   log weights, and log z-hat, the sum over the steps of the log mean
   weight."
  [model first-step steps kernel n key]
  (let [[k-init k-steps] (rng/split key)
        init (mapv #(gfi/generate model (:args first-step)
                                  (:observations first-step) %)
                   (rng/split k-init n))
        ls (mapv #(array/item (:weight %)) init)]
    (reduce
     (fn [{:keys [traces log-weights log-z]} [step k]]
       (let [[k-resample k-kernel k-update] (rng/split k 3)
             total (:log-total (normalize log-weights))
             parents (w/multinomial (mapv #(- % total) log-weights) n
                                    k-resample)
             moved (mapv (fn [i kk ku] (advance (traces i) kernel step kk ku))
                         parents (rng/split k-kernel n) (rng/split k-update n))
             ls (mapv :weight moved)]
         {:traces (mapv :trace moved) :log-weights ls
          :log-z (+ log-z (log-ml ls))}))
     {:traces (mapv :trace init) :log-weights ls :log-z (log-ml ls)}
     (map vector steps (rng/split k-steps (count steps))))))
```

For the running model, a step adds a measurement: the model's first
argument grows by one, and the step's observation is the new $y$. No
latent choice is added, so the update's weight is the new measurement's
log density, which `project` reads from the new trace:

<!-- code: data-steps -->
```clojure
(def first-observation
  {:args [1 1.0] :observations (cm/choicemap [:y 0] (ys 0))})

(def data-steps
  "Step j adds the observation y_j: the model's first argument grows."
  (mapv (fn [j] {:args [(inc j) 1.0]
                 :argdiffs [diff/unknown-change diff/no-change]
                 :observations (cm/choicemap [:y j] (ys j))})
        (range 1 n-ys)))

(let [t (:trace (gfi/generate mean-model (:args first-observation)
                              (:observations first-observation) (rng/key 25)))
      {:keys [args argdiffs observations]} (first data-steps)
      {t' :trace w :weight} (gfi/update t args argdiffs observations
                                        (rng/key 26))]
  (array/allclose? w (gfi/project t' (sel/select [:y 1]))))
;=> true
```

Rejuvenation is the random walk of the MH section. A hundred particles
estimate $\log \bar p(\rho)$:

<!-- code: filtered -->
```clojure
(def rejuvenate (mh-kernel walk [0.5] (observations ys)))

(def filtered
  (particle-filter mean-model first-observation data-steps rejuvenate 100
                   (rng/key 19)))

[(:log-z filtered) (exact-log-ml ys 1.0)]
;=> [-11.488 -11.261] ; float64
```

The library's particle filter is this loop. Its operations, with the
same keys, make the same estimate:

<!-- code: pf-library -->
```clojure
(defn library-filter
  "The particle filter of the library's operations, with the same keys as
   particle-filter: pf/resample, pf/rejuvenate and pf/step at each step."
  [model first-step steps kernel n key]
  (let [[k-init k-steps] (rng/split key)]
    (reduce (fn [state [step k]]
              (let [[k-resample k-kernel k-update] (rng/split k 3)]
                (-> state
                    (pf/resample {:key k-resample})
                    (pf/rejuvenate (fn [t k] {:trace (kernel t k)}) k-kernel)
                    (pf/step (:args step) (:argdiffs step) (:observations step)
                             {:key k-update}))))
            (pf/initialize model (:args first-step) (:observations first-step)
                           n {:key k-init})
            (map vector steps (rng/split k-steps (count steps))))))

(array/allclose? (:log-z filtered)
                 (pf/log-ml-estimate
                  (library-filter mean-model first-observation data-steps
                                  rejuvenate 100 (rng/key 19))))
;=> true
```

The estimate of $\bar p(\rho)$ is unbiased. Over 100 filters of 20
particles, $\hat z/\bar p(\rho)$ averages 1 within five standard errors,
with the moves and without them:

<!-- code: pf-unbiased -->
```clojure
(defn filter-runs
  "log z-hat of runs of the particle filter of n particles with kernel,
   one per key."
  [kernel n runs key]
  (mapv #(:log-z (particle-filter mean-model first-observation data-steps
                                  kernel n %))
        (rng/split key runs)))

(defn ratio-near-one?
  "Is the mean of z-hat/z, over the runs' log z-hat, within five standard
   errors of 1?"
  [log-zs]
  (let [{:keys [mean se]} (mean-se (mapv #(/ (Math/exp %) exact-z) log-zs))]
    (< (abs (- mean 1.0)) (* 5 se))))

(def with-moves (filter-runs rejuvenate 20 100 (rng/key 20)))

(def without-moves (filter-runs (fn [t _] t) 20 100 (rng/key 20)))

[(ratio-near-one? with-moves) (ratio-near-one? without-moves)]
;=> [true true]
```

Rejuvenation does not buy unbiasedness; it buys precision. Without moves,
the spread of $\log \hat z$ is more than twice as large:

<!-- code: pf-spread -->
```clojure
(defn sd-se
  "The standard deviation of the numbers xs, and its standard error, about
   sd/sqrt(2(n - 1)) for a normal spread."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(Math/pow (- % mean) 2) xs)) (dec n)))]
    {:sd sd :se (/ sd (Math/sqrt (* 2 (dec n))))}))

(defn smaller?
  "Is the first spread smaller than the second by more than five standard
   errors of their difference?"
  [a b]
  (> (- (:sd b) (:sd a))
     (* 5 (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b)))))))

(mapv (comp :sd sd-se) [with-moves without-moves])
;=> [0.430 1.096] ; float64

(smaller? (sd-se with-moves) (sd-se without-moves))
;=> true
```

The reason is degeneracy. Without moves, $\mu$ keeps the values it was
drawn with at the first step, and each resampling copies some of them and
drops others. After seven resamplings, 9 of the 100 values are left. The
moves give the copies new values:

<!-- code: pf-distinct -->
```clojure
(def unmoved
  (particle-filter mean-model first-observation data-steps (fn [t _] t) 100
                   (rng/key 19)))

(defn distinct-mus
  "How many distinct values of mu a filter's particles hold."
  [state]
  (count (distinct (mapv #(array/item (% :mu)) (:traces state)))))

[(distinct-mus filtered) (distinct-mus unmoved)]
;=> [91 9] ; float64
```

The particles without moves sit on a few values; with moves they spread
over the posterior:

<!-- code: pf-figure -->
```clojure
;; figure: figures/032-rejuvenation.svg
(defn weighted-mus
  "A filter's particles' mu, with their normalized weights, as a histogram
   layer."
  [state label]
  {:values (mapv #(array/item (% :mu)) (:traces state))
   :weights (:weights (normalize (:log-weights state)))
   :label label})

(let [exact (d/normal (:mean posterior) (:sd posterior))]
  (svg/histogram [(assoc (weighted-mus filtered "with moves")
                         :density #(array/exp (dc/log-prob exact %)))
                  (weighted-mus unmoved "without")]
                 {:range [0 3] :bins 30 :x-label "mu"}))
```

![The final particles' values of $\mu$, weighted, with rejuvenation
(blue) and without (orange), and the exact posterior
density](figures/032-rejuvenation.svg)

### Annealed importance sampling

*Annealed importance sampling* (AIS) carries one trace through a sequence
of distributions that changes an argument of the model from an easy value
to the one of interest (Algorithm 8). It starts as importance sampling
at the easy value. At each step a kernel moves the trace at the current
argument, and `update` changes the argument; the update's weight,
$\log p(\tau; x_j) - \log p(\tau; x_{j-1})$, adds to $\log \hat z$. The
argument must not change which choices the model makes. In code, it is
the particle filter's `advance` with one particle and no resampling:

<!-- code: ais -->
```clojure
(defn ais
  "Algorithm 8: one trace of model generated on the first step's
   arguments with the observations, then advanced with kernel through the
   steps, each update's weight added to log z-hat. [t log z-hat]."
  [model first-step steps kernel key]
  (let [[k-init k-steps] (rng/split key)
        {t :trace w :weight} (gfi/generate model (:args first-step)
                                           (:observations first-step) k-init)]
    (reduce (fn [[t log-z] [step k]]
              (let [[k-kernel k-update] (rng/split k)
                    {t' :trace w :weight}
                    (advance t kernel step k-kernel k-update)]
                [t' (+ log-z w)]))
            [t (array/item w)]
            (map vector steps (rng/split k-steps (count steps))))))
```

For the running model the argument is the noise. At a noise of 10 the
measurements say almost nothing, the posterior is nearly the prior, and
importance sampling from the prior is easy. `noise-steps` lowers the
noise to 1 in equal ratios of $1/\text{noise}^2$:

<!-- code: noise-steps -->
```clojure
(defn noise-steps
  "m steps that lower the noise of the model of ys from 10 to 1, as
   1/sqrt(beta) for beta from 0.01 to 1 in equal ratios: [the first step,
   the rest]."
  [ys m]
  (let [noise (fn [j] (Math/pow 10.0 (- 1.0 (/ j (double m)))))]
    [{:args [(count ys) (noise 0)] :observations (observations ys)}
     (mapv (fn [j] {:args [(count ys) (noise j)]
                    :argdiffs [diff/no-change diff/unknown-change]
                    :observations cm/EMPTY})
           (range 1 (inc m)))]))

(mapv #(second (:args %)) (second (noise-steps ys 4)))
;=> [5.6234 3.1623 1.7783 1.0]
```

The kernel is the Gibbs move, which reads the noise from the trace's
arguments, so it targets each step's distribution. A hundred runs of 20
steps estimate $\bar p(\rho)$ without bias:

<!-- code: ais-run -->
```clojure
(def gibbs-kernel (mh-kernel gibbs [] (observations ys)))

(defn ais-runs
  "log z-hat of runs of AIS on ys with m steps and the Gibbs kernel, one per
   key."
  [ys m runs key]
  (let [[first-step steps] (noise-steps ys m)]
    (mapv #(second (ais mean-model first-step steps gibbs-kernel %))
          (rng/split key runs))))

(def annealed (ais-runs ys 20 100 (rng/key 21)))

(ratio-near-one? annealed)
;=> true
```

More steps make the sequence of distributions finer, and the estimate more
precise. From 5 steps to 80, the spread of $\log \hat z$ shrinks to a
fifth:

<!-- code: ais-steps -->
```clojure
(def few (ais-runs ys 5 100 (rng/key 22)))

(def many (ais-runs ys 80 100 (rng/key 23)))

(mapv (comp :sd sd-se) [few annealed many])
;=> [1.98 0.614 0.367] ; float64

(smaller? (sd-se many) (sd-se few))
;=> true
```

On this model the prior is a fair proposal, and importance sampling with
as many samples as AIS takes steps does about as well. The third exercise
moves the data where the prior is poor.

## The catalog

Every algorithm of this chapter is a short program over the same
operations:

| Algorithm (thesis) | Operations | Here | Library |
|---------------------------|--------------------|----------------------|-------------------------------|
| Simple Monte Carlo (Alg. 1) | `simulate` | `simple-monte-carlo` | |
| Regular importance sampling (Alg. 2) | `simulate`, `generate` | `regular-importance` | |
| Self-normalized importance sampling (Alg. 3) | `simulate`, `generate` | `self-normalized` | `imp/importance-sampling` |
| Training a proposal (Alg. 4) | `simulate`, `generate`, gradient | `train` | `learn/lecture-batched` |
| MH with a proposal (Alg. 5) | `propose`, `update`, `assess` | `mh-kernel` | `mh/mh` |
| MH over regenerate (§4.3) | `regenerate` | `regenerate-kernel` | `mh/mh` |
| Hamiltonian Monte Carlo (Alg. 6) | `simulate`, `update`, gradient, `assess` | `hmc-kernel` | `mcmc/hmc` |
| Resample-move particle filter (Alg. 7) | `generate`, `update`, `project`, a kernel | `particle-filter` | `pf/initialize`, `pf/resample`, `pf/rejuvenate`, `pf/step` |
| Annealed importance sampling (Alg. 8) | `generate`, `update`, a kernel | `ais` | |

The model never changed to suit an algorithm. A different algorithm is a
different proposal, a different schedule of arguments, or a different way
of keeping score. That is why the library is small, and why a model
written in Part I runs under every row.

## Where the operations are not enough

The catalog has limits, and some of them the operations cannot see.

**The addresses must match.** Algorithms 2 and 3 pair each choice of the
proposal with the model's choice at the same address. A proposal in
another representation has nothing to pair with. Here the model draws a
point in Cartesian coordinates and the proposal in polar ones:

<!-- code: polar -->
```clojure
(def point
  (gen []
    (trace :x (d/normal 0 1))
    (trace :y (d/normal 0 1))))

(def polar
  (gen []
    (trace :r (d/gamma 2 1))
    (trace :theta (d/uniform 0 (* 2 Math/PI)))))
```

`generate` refuses choices at addresses the model never visits:

<!-- code: polar-error -->
```clojure
;; error: generate: the constraint at [:r] was not visited
(self-normalized point [] cm/EMPTY {:proposal polar :f (constantly 1.0)}
                   10 (rng/key 24))
```

A *trace translator* (thesis §3.6) maps one representation to the other
with a differentiable bijection, and adds its Jacobian to the weight. A
move that changes the number of choices, as reversible-jump MCMC's do,
needs the same tools. Both are the subject of *Trace Translators and
Involutive MCMC*.

**The conditions are the user's.** The operations compute weights; they
do not check that a proposal covers the target (Eqs. 3.3 and 3.6), that
a kernel leaves the posterior unchanged, or that it can reach all of it.
`only-a` gave a wrong answer without an error, and so did Algorithm 5
with its signs reversed. gen.clj checks what it can: `assess` refuses an
incomplete choice map, `mh-move` a proposal that changes the data,
`normalize` weights that are all 0. The fourth exercise shows a
composition of correct kernels that is wrong, which no operation can
detect; the thesis's Proposition 3.4.1 states the condition it breaks.

**Some algorithms need more than the seven.** Hamiltonian Monte Carlo and
training need gradients, which the thesis counts among the trace's
operations. Gibbs moves on discrete choices need the support of a
distribution to enumerate (*Debugging with Enumeration*). Pseudo-marginal
methods need generative functions whose densities are estimates
(encapsulated randomness, the thesis's §4.5).

The next chapter, *Trace Translators and Involutive MCMC*, takes up the
first limit.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Break Eq. 3.3.** Give `squares` a proposal that goes on with
   probability 0.7 for its first three steps and stops at the fourth.
   Predict what regular importance sampling converges to.
2. **Forget the reverse move.** Accept Algorithm 5's moves on `log w`
   alone, without the two proposal densities. Predict whether the switch
   kernel still leaves the posterior unchanged, and whether the random
   walk's would.
3. **Data far from the prior.** Shift every measurement up by 3, where
   the prior puts little mass. Compare AIS with 20 steps against
   importance sampling with 20 samples from the prior, over 100 runs
   each. Predict which estimate of $\log \bar p(\rho)$ spreads more, and
   which falls below the exact value.
4. **A mixture that depends on the trace.** A kernel applies the switch
   kernel with some probability and otherwise leaves the trace alone.
   With probability 0.5 in every state, does it leave the posterior
   unchanged? With probability 0.9 when $a$ is false and 0.1 when it is
   true?

## Solutions

**1.** A trace with more than three $x$'s has proposal probability 0,
so the estimate converges to the expectation of $g$ over the traces with
at most three: 2.4375, not 7. Nothing fails:

<!-- code: solution-1 -->
```clojure
(def three
  (gen []
    (loop [i 0]
      (when (trace [:go i] (d/bernoulli (if (< i 3) 0.7 0.0)))
        (trace [:x i] (d/normal (inc i) 1))
        (recur (inc i))))))

(def truncated
  "E[g] over the traces of squares with at most three x's: k x's have
   probability 0.5^(k+1), and then E[g] = Σ_i (i^2 + 1)."
  (reduce + (map (fn [k] (* (Math/pow 0.5 (inc k))
                            (reduce + 0.0 (map #(+ (* % %) 1.0)
                                               (range 1 (inc k))))))
                 (range 4))))

truncated
;=> 2.4375

(let [est (mean-se (regular-importance squares three g 4000 (rng/key 30)))]
  [(near? est truncated) (near? est 7.0)])
;=> [true false]
```

**2.** The switch proposal is asymmetric: from $\{a = \text{false}\}$ it
proposes $\{a = \text{true}, b\}$ with probability $0.9 \cdot 0.5$, and
the reverse move has probability 0.9. Without the densities the kernel
drifts from the posterior. A random walk proposes $\mu'$ from
$\mathcal{N}(\mu, 0.5)$ and the reverse from $\mathcal{N}(\mu', 0.5)$,
the same density, so the two terms cancel and dropping them changes
nothing:

<!-- code: solution-2 -->
```clojure
(defn forgetful
  "The acceptance ratio without the proposal's densities: log w alone."
  [{:keys [log-w]} _]
  log-w)

(after-step (switch-row forgetful))
;=> [0.2792 0.0458 0.675]

(let [{sigma :choices log-q :weight}
      (gfi/propose walk [mean-start 0.5] (rng/key 31))
      move (mh-move mean-start walk [0.5] sigma (observations ys)
                    [(rng/key 0) (rng/key 0)])]
  (array/allclose? (:log-q' move) log-q))
;=> true
```

**3.** Data near 4.6 lie in the prior's tail, so importance sampling
from the prior finds samples with large weights only rarely: its
estimates spread widely and, as logs of unbiased estimates, fall far
below the exact value. AIS starts where the prior is a good proposal and
lets the Gibbs move follow the posterior as the noise falls. Its spread
grows from 0.6 to 1.5; importance sampling's is 11:

<!-- code: solution-3 -->
```clojure
(def far (mapv #(+ % 3.0) ys))

(defn sampled-runs
  "log z-hat of runs of importance sampling on ys with k samples from the
   prior, one per key."
  [ys k runs key]
  (mapv (fn [run-key]
          (log-ml (mapv #(array/item (:weight (gfi/generate
                                               mean-model [(count ys) 1.0]
                                               (observations ys) %)))
                        (rng/split run-key k))))
        (rng/split key runs)))

(def far-annealed (ais-runs far 20 100 (rng/key 32)))

(def far-sampled (sampled-runs far 20 100 (rng/key 33)))

[(exact-log-ml far 1.0) (:mean (mean-se far-annealed))
 (:mean (mean-se far-sampled))]
;=> [-19.595 -20.8 -43.4] ; float64

(mapv (comp :sd sd-se) [annealed far-annealed far-sampled])
;=> [0.614 1.50 11.16] ; float64

(smaller? (sd-se far-annealed) (sd-se far-sampled))
;=> true
```

**4.** With a constant probability the mixture leaves the posterior
unchanged: each component does, so their mixture does. With a
probability that depends on $a$, it does not. Proposition 3.4.1 asks that
a component's probability be the same at both ends of any move that
component can make, and the switch kernel changes $a$:

<!-- code: solution-4 -->
```clojure
(defn mixture-row
  "The transitions of a kernel that applies switch-proposal's MH kernel
   with probability (theta u) in state u, and otherwise stays."
  [theta]
  (fn [u]
    (merge-with + (update-vals ((switch-row log-alpha) u) #(* (theta u) %))
                {u (- 1.0 (theta u))})))

(array/allclose? (array/array (after-step (mixture-row (constantly 0.5))))
                 (array/array target))
;=> true

(after-step (mixture-row #(if (:a %) 0.1 0.9)))
;=> [0.0467 0.1433 0.81]
```
