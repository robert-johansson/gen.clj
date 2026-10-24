# Debugging with Enumeration

Importance sampling (*Conditioning*) and MCMC (*MCMC and MAP*,
*Gradient-Based MCMC*) approximate a posterior with samples. When their
answer looks wrong, one of two things is at fault. The algorithm may have
failed to approximate the posterior: its proposal was poor, its chain never
mixed, or it drew too few samples. Or the model may be at fault: the
algorithm computed its posterior faithfully, and that posterior is not what
we meant. The samples alone cannot tell these apart.

*Enumerative inference* can. It visits every point of a grid over the
latent choices and weighs each by its posterior probability. For finite
choices the result is the exact posterior; for continuous ones it is an
approximation as good as the grid is fine and wide. Enumeration is slow: it
spends as long where the posterior is negligible as where it is large, and
its cost grows exponentially with the number of choices. But for a model
of a few dimensions it gives a yardstick, an answer close enough to exact
to judge an approximate algorithm by, and to judge the model by.

This chapter ports Gen.jl's *Debugging Models with Enumerative Inference*:
a robust regression enumerated on discrete and continuous grids, a
misspecified model diagnosed, and a hierarchical model that repairs it. It
adds gen.clj's exact proposals, which enumerate finite choices inside other
algorithms. Every enumerated answer is checked against one computed another
way, so the yardstick is itself checked.

<!-- code: ns -->
```clojure
(ns book.enumeration
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## Enumeration for discrete models

We start with a model whose latent choices are all finite, so that
enumeration is exact. It is Gen.jl's robust linear regression: a line, and
points that lie near it unless they are outliers. Each point has a flag,
true with a small probability, that says which it is. An inlier's $y$ is
normal about the line with standard deviation 1; an outlier's is normal
about 0 with standard deviation 5. The chapter's first models differ only
in their priors, so one function makes them from a map of priors:

<!-- code: model -->
```clojure
(defn robust-regression
  "A line through points, some of them outliers: a slope and an intercept
   from their priors, then for each x a flag, true with probability
   prob-outlier, and a y, normal about 0 with sd 5 for an outlier and
   about the line with sd 1 for an inlier."
  [{:keys [slope-prior intercept-prior prob-outlier]}]
  (gen [xs]
    (let [slope (trace :slope slope-prior)
          intercept (trace :intercept intercept-prior)]
      (doseq [[i x] (map-indexed vector xs)]
        (let [outlier? (trace [:data i :outlier?] (d/bernoulli prob-outlier))]
          (trace [:data i :y]
                 (d/normal (array/where outlier? 0.0 (+ (* slope x) intercept))
                           (array/where outlier? 5.0 1.0))))))))

(def discrete-priors
  {:slope-prior (d/uniform-discrete -2 2)
   :intercept-prior (d/uniform-discrete -2 2)
   :prob-outlier 0.1})

(def discrete-regression (robust-regression discrete-priors))
```

The model picks the mean and the standard deviation of each $y$ with
`array/where`, where Gen.jl's branches with `if`. Both define the same
distribution, but `where` treats every run alike, so a batched execution
can run many points of a grid at once. A test of a choice whose value
differs across the points makes a batched run fall back to one point at a
time, as *The Handler Loop* showed. The discrete model's slope and
intercept are integers from −2 to 2, each equally likely, and one point in
ten is an outlier.

The data are five points near the line $y = x$, with the middle one moved
up to 4:

<!-- code: discrete-data -->
```clojure
(def xs [-2.0 -1.0 0.0 1.0 2.0])

(def discrete-ys [-1.59 -0.92 4.0 0.14 2.18])

(defn observations
  "The choice map fixing each [:data i :y] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:data i :y] y]) (range) ys)))
```

<!-- code: discrete-data-figure -->
```clojure
;; figure: figures/024-discrete-data.svg
(let [points (map vector xs discrete-ys)]
  (svg/plot [{:curves [(fn [x] x)] :color "gray"}
             {:points (remove #{[0.0 4.0]} points)}
             {:points [[0.0 4.0]] :color "firebrick"}]))
```

![The data, the line $y = x$ (gray), and the point moved to 4
(red)](figures/024-discrete-data.svg)

### The grid

`enum/choice-vol-grid` takes a spec for each latent choice: its address
and the values to try. Its points are every combination of them: 5 slopes,
5 intercepts and 2 values for each of the five flags, 800 points. A point
is a pair: a choice map of the latent choices, and the log of the volume
of latent space the point stands for, which is 0 for discrete choices.
Points come in row-major order, the last spec varying fastest (Gen.jl's
grid varies the first fastest):

<!-- code: discrete-grid -->
```clojure
(def flag-specs
  "A grid spec for each point's outlier flag: both of its values."
  (for [i (range (count xs))] [[:data i :outlier?] [false true]]))

(def discrete-grid
  (apply enum/choice-vol-grid
         [:slope (range -2 3)]
         [:intercept (range -2 3)]
         flag-specs))

(count discrete-grid)
;=> 800

(defn describe
  "A grid point as [slope intercept flags log-volume]."
  [[choices log-volume]]
  [(cm/get-choice choices :slope)
   (cm/get-choice choices :intercept)
   (mapv #(cm/get-choice choices [:data % :outlier?]) (range (count xs)))
   log-volume])

(describe (first discrete-grid))
;=> [-2 -2 [false false false false false] 0.0]

(describe (second discrete-grid))
;=> [-2 -2 [false false false false true] 0.0]

(describe (peek discrete-grid))
;=> [2 2 [true true true true true] 0.0]
```

`enum/enumerative-inference` takes the model, its arguments, the
observations and the grid. At each point it runs `generate`, constrained
by the observations and the point's choices, and weights the trace by
`generate`'s weight plus the point's log-volume. Normalized, the weights
are the posterior probabilities of the points. With `:batched? true`, the
800 points run as the particles of one batched execution. The result
holds the traces, here as collections of particles (`enum/traces` gives
them one by one), their normalized log weights in grid order, and the log
of the weights' sum, which estimates the log marginal likelihood
$\log p(ys)$:

<!-- code: discrete-run -->
```clojure
(def discrete-result
  (enum/enumerative-inference discrete-regression [xs]
                              (observations discrete-ys) discrete-grid
                              {:key (rng/key 1) :batched? true}))

(sort (keys discrete-result))
;=> (:batches :log-ml-estimate :log-normalized-weights)

(count (enum/traces discrete-result))
;=> 800

(:log-ml-estimate discrete-result)
;=> -12.444
```

Here the estimate is exact. Every latent choice is on the grid, so the
sum runs over every way the model could have made the data. A sum in the
other order checks it by hand. Given the line, each flag concerns only its
own point, so it sums out point by point:

$$
p(y_i \mid \text{line}) = 0.1\, \mathcal{N}(y_i; 0, 5) +
0.9\, \mathcal{N}(y_i; \text{slope} \cdot x_i + \text{intercept}, 1),
$$

and the 25 lines are equally probable a priori. The same sums give each
point's probability of being an outlier, which we use below:

<!-- code: discrete-exact -->
```clojure
(defn point-likelihood
  "p(y | slope, intercept) for the point (x, y) with its flag summed out:
   an outlier with probability prob-outlier, an inlier otherwise."
  [prob-outlier slope intercept x y]
  (let [density (fn [mean sd] (array/exp (dc/log-prob (d/normal mean sd) y)))]
    (+ (* prob-outlier (density 0.0 5.0))
       (* (- 1.0 prob-outlier) (density (+ (* slope x) intercept) 1.0)))))

(defn discrete-by-hand
  "The discrete model's answers by hand: the flags summed out point by
   point, then the 25 equally probable lines. {:log-ml, and :outliers, the
   probability that each point is an outlier}."
  [ys]
  (let [lines (for [slope (range -2 3) intercept (range -2 3)]
                [slope intercept])
        likelihood (fn [[s b]]
                     (reduce * (map #(point-likelihood 0.1 s b %1 %2) xs ys)))
        joint (mapv #(/ (likelihood %) 25.0) lines)
        evidence (reduce + joint)
        outlier (fn [i [s b] p]
                  (let [y (ys i)
                        out (* 0.1 (array/exp (dc/log-prob (d/normal 0 5) y)))]
                    (* (/ p evidence)
                       (/ out (point-likelihood 0.1 s b (xs i) y)))))]
    {:log-ml (array/log evidence)
     :outliers (mapv (fn [i] (reduce + (map #(outlier i %1 %2) lines joint)))
                     (range (count xs)))}))

(def discrete-exact (discrete-by-hand discrete-ys))

(array/allclose? (:log-ml-estimate discrete-result) (:log-ml discrete-exact))
;=> true
```

### Marginals

The probabilities of the 800 points are the *joint* posterior of every
latent choice. Summing out some choices leaves the posterior of the
others, their *marginal*. The grid's row-major order makes this an array
operation, as in Gen.jl: in the grid's shape, with the 32 assignments of
the flags as one axis, the probabilities form a [5 5 32] array, and
summing its last axis leaves the posterior of the line, a table of slope
by intercept:

<!-- code: discrete-table -->
```clojure
(defn probabilities
  "The posterior probabilities of an enumeration's points, in grid order."
  [result]
  (mapv #(Math/exp %) (:log-normalized-weights result)))

(def discrete-table
  "The posterior of the line: the points' probabilities in the grid's
   shape, summed over the 32 assignments of the flags."
  (-> (array/array (probabilities discrete-result))
      (array/reshape [5 5 32])
      (array/sum {:axis 2})
      array/->clj))

(get-in discrete-table [3 2])
;=> 0.684
```

<!-- code: discrete-figure -->
```clojure
;; figure: figures/024-discrete-posterior.svg
(svg/heatmap (range -2 3) (range -2 3) discrete-table
             {:labels? true :mark [0 1] :marginals? true
              :x-label "intercept" :y-label "slope"})
```

![The posterior of the line, with each choice's marginal in the bars; the
true line's cell is outlined](figures/024-discrete-posterior.svg)

The true line, slope 1 and intercept 0, is the most probable, with 0.684;
the lines beside it, with intercepts −1 and 1, share most of the rest.
`enum/marginal` gives one choice's marginal directly, as a map from each
value to its probability. The slope's agrees with the table's row sums:

<!-- code: discrete-marginal -->
```clojure
(enum/marginal discrete-result :slope)
;=> {-2 1.1E-4 -1 7.0E-4 0 0.0449 1 0.9464 2 0.008}

(array/allclose?
 (array/array (mapv (enum/marginal discrete-result :slope) (range -2 3)))
 (array/array (mapv #(reduce + %) discrete-table)))
;=> true
```

The flags' marginals say which points are outliers. The middle point is
one with probability 0.917. The others, which the line explains well, are
less likely outliers than the prior's one in ten. Both agree with the sums
by hand:

<!-- code: discrete-outliers -->
```clojure
(defn outlier-probabilities
  "The posterior probability that each point is an outlier."
  [result]
  (mapv #(get (enum/marginal result [:data % :outlier?]) true 0.0)
        (range (count xs))))

(outlier-probabilities discrete-result)
;=> [0.035 0.028 0.917 0.047 0.042]

(array/allclose? (array/array (outlier-probabilities discrete-result))
                 (array/array (:outliers discrete-exact)))
;=> true
```

### Queries

gen.clj has no functions yet for posterior queries such as expectations
and entropies, but each is a short sum over an enumeration. A choice's
posterior mean weights its values by their probabilities. The *entropy*
of a distribution, $-\sum p \log p$, measures its uncertainty in nats:
$\log n$ for $n$ equally probable values. The prior over the 25 lines has
entropy $\log 25 = 3.22$. The posterior's is 1.06, as uncertain as a
choice among $e^{1.06} \approx 3$ equally probable lines:

<!-- code: queries -->
```clojure
(defn expectation
  "The posterior mean of the choice at addr, from its marginal: each value
   weighted by its probability."
  [result addr]
  (reduce + (map (fn [[v p]] (* v p)) (enum/marginal result addr))))

(defn entropy
  "The entropy of the probabilities ps, in nats: the sum of -p log p."
  [ps]
  (- (reduce + (map #(if (pos? %) (* % (Math/log %)) 0.0) ps))))

[(expectation discrete-result :slope) (expectation discrete-result :intercept)]
;=> [0.961 0.03]

[(Math/log 25.0) (entropy (apply concat discrete-table))]
;=> [3.219 1.063]
```

## Enumeration for continuous models

A continuous choice has no list of values to try. A grid over it divides
a range into intervals, and each point stands for a cell of the latent
space. Gen.jl's continuous model gives the line normal priors:

<!-- code: continuous-model -->
```clojure
(def continuous-priors
  {:slope-prior (d/normal 0 1)
   :intercept-prior (d/normal 0 2)
   :prob-outlier 0.1})

(def continuous-regression (robust-regression continuous-priors))
```

A continuous spec gives the intervals' endpoints and `:continuous`. The
slope's endpoints run from −3 to 3 in steps of 0.25, making 24 intervals,
and the intercept's from −4 to 4 in steps of 0.5, making 16. With the flags
that makes 12288 points. A point sits at the middle of its cell (the
default `:anchor`, `:midpoint`; `:left` and `:right` are the others), and
its log-volume is the log of the cell's area, $\log(0.25 \cdot 0.5)$. Its
weight, the posterior density at the midpoint times the area, approximates
the posterior probability of the whole cell. This is the midpoint rule of
numerical integration.

<!-- code: continuous-grid -->
```clojure
(defn endpoints
  "The endpoints of the intervals of width step from lo to hi."
  [lo hi step]
  (mapv #(+ lo (* step %)) (range (inc (Math/round (/ (- hi lo) step))))))

(def line-specs
  [[:slope (endpoints -3 3 0.25) :continuous]
   [:intercept (endpoints -4 4 0.5) :continuous]])

(def continuous-grid
  (apply enum/choice-vol-grid (concat line-specs flag-specs)))

(count continuous-grid)
;=> 12288

(describe (first continuous-grid))
;=> [-2.875 -3.75 [false false false false false] -2.0794]

(array/allclose? (peek (first continuous-grid)) (Math/log (* 0.25 0.5)))
;=> true
```

The data come from the line with slope −1.21 and intercept 2.56, which
lie between the grid's midpoints, plus noise of standard deviation 1. The
second point is moved to 0:

<!-- code: continuous-data -->
```clojure
(def continuous-ys [5.74 0.0 1.58 1.9 -0.64])
```

<!-- code: continuous-data-figure -->
```clojure
;; figure: figures/024-continuous-data.svg
(let [points (map vector xs continuous-ys)]
  (svg/plot [{:curves [(fn [x] (+ (* -1.21 x) 2.56))] :color "gray"}
             {:points (remove #{[-1.0 0.0]} points)}
             {:points [[-1.0 0.0]] :color "firebrick"}]))
```

![The data, the true line (gray), and the point moved to 0
(red)](figures/024-continuous-data.svg)

<!-- code: continuous-run -->
```clojure
(def continuous-result
  (enum/enumerative-inference continuous-regression [xs]
                              (observations continuous-ys) continuous-grid
                              {:key (rng/key 2) :batched? true}))

(:log-ml-estimate continuous-result)
;=> -13.0441
```

### Checking the yardstick

Now the enumeration approximates, and an approximation used as a
yardstick needs checking. This model happens to have an exact answer.
Given which points are outliers, it is linear and Gaussian, as *Getting
Started*'s model was. The outliers' $y$s are independent normals about 0.
The inliers' are jointly normal: with $X$ the matrix of their rows
$(x_i, 1)$ and $S$ the prior covariance of the line, their covariance is
$X S X^\top + I$. That gives $p(ys \mid \text{flags})$, and the line's
posterior given the flags is normal, with the mean of *Getting Started*'s
formula. Summing $p(\text{flags})\, p(ys \mid \text{flags})$ over the 32
assignments gives $p(ys)$, and weighting each assignment's answers by its
posterior probability gives the exact means and outlier probabilities:

<!-- code: exact-posterior -->
```clojure
(def configurations
  "Every assignment of the five flags, in the grid's order."
  (vec (for [bits (range 32)] (mapv #(bit-test bits %) (range 4 -1 -1)))))

(defn given-flags
  "A robust regression's answers given which points are outliers (flags),
   for normal priors about 0: {:log-p, log p(flags) + log p(ys | flags),
   and :mean, the posterior mean of [slope intercept]}. The outliers' ys
   are normal about 0 with sd 5. The inliers' are jointly normal, with
   covariance X S X' + I, X their rows [x 1] and S the line's prior
   covariance, and the line's posterior is as in Getting Started."
  [{:keys [slope-prior intercept-prior prob-outlier]} ys flags]
  (let [ins (keep-indexed (fn [i out?] (when-not out? i)) flags)
        outs (keep-indexed (fn [i out?] (when out? i)) flags)
        log-flags (+ (* (count outs) (Math/log prob-outlier))
                     (* (count ins) (Math/log (- 1.0 prob-outlier))))
        log-outs (reduce + 0.0 (map #(dc/log-prob (d/normal 0 5) (ys %)) outs))
        variance #(array/square (:std (:params %)))
        S (array/diag (array/array [(variance slope-prior)
                                    (variance intercept-prior)]))]
    (if (empty? ins)
      {:log-p (+ log-flags log-outs) :mean [0.0 0.0]}
      (let [X (array/array (mapv (fn [i] [(xs i) 1.0]) ins))
            Xt (array/transpose X)
            y (array/array (mapv ys ins))
            cov (+ (array/matmul X (array/matmul S Xt)) (array/eye (count ins)))
            post (linalg/inv (+ (linalg/inv S) (array/matmul Xt X)))]
        {:log-p (+ log-flags log-outs
                   (dc/log-prob (d/mvnormal (array/zeros [(count ins)]) cov) y))
         :mean (array/->clj (array/matmul post (array/matmul Xt y)))}))))

(defn exact-posterior
  "A robust regression's exact answers, for normal priors about 0: the
   answers given each assignment of the flags, weighted by its posterior
   probability. {:log-ml :outliers :mean}"
  [priors ys]
  (let [parts (mapv #(given-flags priors ys %) configurations)
        {:keys [log-total log-weights]} (w/normalize (mapv :log-p parts))
        ps (mapv #(Math/exp %) log-weights)
        weighted (fn [f] (reduce + (map * ps (map f configurations parts))))]
    {:log-ml log-total
     :outliers (mapv (fn [i] (weighted (fn [c _] (if (c i) 1.0 0.0))))
                     (range (count xs)))
     :mean (mapv (fn [j] (weighted (fn [_ part] ((:mean part) j)))) [0 1])}))

(def continuous-exact (exact-posterior continuous-priors continuous-ys))

(:log-ml continuous-exact)
;=> -13.0439

(- (:log-ml-estimate continuous-result) (:log-ml continuous-exact))
;=> -2.37E-4 ; float64

(< 1e-4
   (- (:log-ml continuous-exact) (:log-ml-estimate continuous-result))
   1e-3)
;=> true

(outlier-probabilities continuous-result)
;=> [0.464 0.437 0.035 0.057 0.089]

(:outliers continuous-exact)
;=> [0.464 0.437 0.035 0.057 0.089]

[(expectation continuous-result :slope)
 (expectation continuous-result :intercept)]
;=> [-0.687 1.521]

(:mean continuous-exact)
;=> [-0.687 1.521]
```

The grid's log marginal likelihood is within $2.4 \times 10^{-4}$ of the
exact value, and its outlier probabilities and means agree to three
decimals. The small error can come from two places: cells too coarse for
the midpoint rule, or probability that lies beyond the grid's edges.
Halving the cells, with four times the points, leaves the error about
where it was. Widening the grid to slopes from −5 to 5 and intercepts from
−8 to 8 removes it, to within rounding:

<!-- code: grid-error -->
```clojure
(defn log-ml-error
  "The error of the continuous model's log marginal likelihood from a
   grid over the line given by specs, and over the flags."
  [specs]
  (let [grid (apply enum/choice-vol-grid (concat specs flag-specs))
        result (enum/enumerative-inference continuous-regression [xs]
                                           (observations continuous-ys) grid
                                           {:key (rng/key 3) :batched? true})]
    (- (:log-ml-estimate result) (:log-ml continuous-exact))))

(def finer
  [[:slope (endpoints -3 3 0.125) :continuous]
   [:intercept (endpoints -4 4 0.25) :continuous]])

(def wider
  [[:slope (endpoints -5 5 0.25) :continuous]
   [:intercept (endpoints -8 8 0.5) :continuous]])

(log-ml-error finer)
;=> -2.97E-4 ; float64

(< (abs (log-ml-error wider)) 5e-5)
;=> true
```

So the error was the probability outside the grid. The cells were fine
enough; the grid was not quite wide enough. When there is no exact answer
to compare with, the same question can be asked of the enumeration
itself: exercise 1 shows how a grid too narrow gives itself away.

### The posterior

The probabilities in the grid's shape, [24 16 32], summed over the flags,
give the posterior of the line:

<!-- code: continuous-table -->
```clojure
(def continuous-table
  "The posterior of the line on the continuous grid: [slope][intercept]."
  (-> (array/array (probabilities continuous-result))
      (array/reshape [24 16 32])
      (array/sum {:axis 2})
      array/->clj))

(defn midpoints
  "The midpoints of the intervals between endpoints."
  [ends]
  (mapv #(/ (+ %1 %2) 2.0) ends (rest ends)))
```

<!-- code: continuous-figure -->
```clojure
;; figure: figures/024-continuous-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.5)) (midpoints (endpoints -3 3 0.25))
             continuous-table
             {:mark [2.56 -1.21] :marginals? true
              :x-label "intercept" :y-label "slope"})
```

![The posterior of the line under the continuous model, with the true
line (cross)](figures/024-continuous-posterior.svg)

The posterior has two modes. One holds the true line and makes the second
point an outlier. The other is a flatter line, which passes near the
second point and makes the first an outlier instead. The outlier
probabilities above say the same: 0.464 for the first point, 0.437 for the
second.

### Conditional posteriors

Fixing a latent choice and enumerating the others gives a *conditional*
posterior. For the posterior given that the second point is an outlier,
we add its flag to the observations and leave it off the grid. The
weights' sum is then $\log p(ys, \text{flag}_1)$, so its difference from
the full enumeration's is $\log p(\text{flag}_1 \mid ys)$, the log of the
flag's marginal:

<!-- code: conditional -->
```clojure
(def given-outlier
  (cm/merge (observations continuous-ys)
            (cm/choicemap [:data 1 :outlier?] true)))

(def conditional-grid
  (apply enum/choice-vol-grid
         (concat line-specs
                 (remove #(= [:data 1 :outlier?] (first %)) flag-specs))))

(count conditional-grid)
;=> 6144

(def conditional-result
  (enum/enumerative-inference continuous-regression [xs] given-outlier
                              conditional-grid
                              {:key (rng/key 4) :batched? true}))

(def flag-1-probability
  (Math/exp (- (:log-ml-estimate conditional-result)
               (:log-ml-estimate continuous-result))))

flag-1-probability
;=> 0.437

(array/allclose? flag-1-probability
                 ((outlier-probabilities continuous-result) 1))
;=> true
```

The same conditional posterior is a slice of the full enumeration: the
points where the second flag is true, which are index 1 of axis 3 in the
grid's shape [24 16 2 2 2 2 2], renormalized:

<!-- code: conditional-slice -->
```clojure
(def conditional-table
  (-> (array/array (probabilities conditional-result))
      (array/reshape [24 16 16])
      (array/sum {:axis 2})))

(def sliced-table
  "The full enumeration's points where point 1 is an outlier: axis 3 of
   the grid's shape at index 1 (true), renormalized."
  (let [slice (-> (array/array (probabilities continuous-result))
                  (array/reshape [24 16 2 2 2 2 2])
                  (array/take 1 3)
                  (array/sum {:axis [2 3 4 5]}))]
    (/ slice (array/sum slice))))

(array/allclose? conditional-table sliced-table)
;=> true
```

<!-- code: conditional-figure -->
```clojure
;; figure: figures/024-conditional-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.5)) (midpoints (endpoints -3 3 0.25))
             (array/->clj conditional-table)
             {:mark [2.56 -1.21] :marginals? true
              :x-label "intercept" :y-label "slope"})
```

![The posterior of the line given that the second point is an
outlier](figures/024-conditional-posterior.svg)

Given that the second point is an outlier, the posterior has one mode,
around the true line. Conditioning helps where the joint posterior is too
large to enumerate: fix some choices at values of interest, enumerate the
rest, and ask whether the answer makes sense.

## Diagnosing model misspecification

Inference can go wrong in two ways:

- **Inference failure.** The algorithm does not approximate the posterior
  well: a poor importance proposal, an MCMC kernel that does not mix, or
  too little computation.
- **Model misspecification.** The posterior, however well approximated,
  does not match what we believe the data say.

Both can happen at once. A grid can be made as fine and wide as we can
afford, so enumeration approximates the posterior as closely as we need.
That tells us whether another algorithm reached the posterior, and whether
the posterior agrees with our intuitions.

Gen.jl's misspecified model has narrow priors on the line, with standard
deviation $\sqrt{0.5}$, and an outlier probability of one in four. Its
data come from a steep line, with slope 2.8 and intercept −2.4, and have
no outliers:

<!-- code: misspecified-model -->
```clojure
(def misspecified-priors
  {:slope-prior (d/normal 0 (Math/sqrt 0.5))
   :intercept-prior (d/normal 0 (Math/sqrt 0.5))
   :prob-outlier 0.25})

(def misspecified (robust-regression misspecified-priors))

(def steep-ys [-8.02 -5.34 -2.13 -1.19 2.57])
```

### Importance resampling

Gen.jl's tutorial draws 100 traces, each by importance resampling from
2000 samples, batched here. Each trace is reduced to host values, its line
and flags, so that the loop keeps no traces:

<!-- code: resampling -->
```clojure
(defn line-of
  "A trace's line and flags as host values: {:slope :intercept :outliers}."
  [tr]
  {:slope (array/item (tr :slope))
   :intercept (array/item (tr :intercept))
   :outliers (mapv #(tr [:data % :outlier?]) (range (count xs)))})

(def resampled
  (mapv (fn [k]
          (line-of (:trace (imp/importance-resampling
                            misspecified [xs] (observations steep-ys) 2000
                            {:key k :batched? true}))))
        (rng/split (rng/key 5) 100)))

(defn mean-and-se
  "The mean of independent draws vs and its standard error."
  [vs]
  (let [n (count vs)
        mean (/ (reduce + vs) n)
        squares (reduce + (map #(array/square (- % mean)) vs))]
    {:mean mean :se (Math/sqrt (/ squares (* n (dec n))))}))

[(mean-and-se (map :slope resampled)) (mean-and-se (map :intercept resampled))]
;=> [{:mean 0.638 :se 0.086} {:mean -0.763 :se 0.074}] ; float64
```

The mean slope is 0.64, far from the true 2.8, and the mean intercept
−0.76, far from −2.4. The flags say why. Importance sampling explains the
data as a flat line with outliers at both ends, where the steep line
needed none:

<!-- code: resampled-outliers -->
```clojure
(defn outlier-frequencies
  "The fraction of draws (line-of maps) in which each point is an outlier."
  [draws]
  (mapv (fn [i] (/ (count (filter #(get-in % [:outliers i]) draws))
                   (double (count draws))))
        (range (count xs))))

(outlier-frequencies resampled)
;=> [0.91 0.83 0.21 0.2 0.46] ; float64
```

Is this an inference failure? It may well be. The proposal is the prior,
which rarely draws a line as steep as the true one, so 2000 samples may
never come near it.

### MCMC from the true line

MCMC moves through the latent space rather than guessing from the prior.
Gen.jl's kernel is like those of *MCMC and MAP*: a random walk on the line,
then a move on each flag that resamples it from its prior, each an MH
move. `mcmc-draws` keeps
host values of each sweep, not its trace, so that a long chain holds no
traces on any backend:

<!-- code: mcmc -->
```clojure
(def line-proposal
  (gen [tr]
    (trace :slope (d/normal (tr :slope) 0.5))
    (trace :intercept (d/normal (tr :intercept) 0.5))))

(defn mcmc-sweep
  "A random-walk MH move on the line, then an MH move on each flag that
   resamples it from its prior."
  [tr key]
  (let [[k-line & k-flags] (rng/split key (inc (count xs)))]
    (reduce (fn [tr [i k]]
              (:trace (mh/mh tr (sel/select [:data i :outlier?]) k)))
            (:trace (mh/mh tr line-proposal [] k-line))
            (map vector (range) k-flags))))

(defn mcmc-draws
  "The lines and flags (line-of) of n sweeps of a chain from the trace
   start, after burn-in sweeps."
  [sweep start n burn-in key]
  (->> (rng/split key (+ burn-in n))
       (reductions sweep start)
       (drop (inc burn-in))
       (mapv line-of)))
```

The chain starts at the true line, so it cannot fail by never visiting it.
It moves away and settles where importance sampling did:

<!-- code: mcmc-run -->
```clojure
(def at-the-truth
  (:trace (gfi/generate misspecified [xs]
                        (cm/merge (observations steep-ys)
                                  (cm/choicemap :slope 2.8 :intercept -2.4))
                        (rng/key 6))))

(def chain (mcmc-draws mcmc-sweep at-the-truth 10000 500 (rng/key 7)))

(defn chain-mean
  "The mean of a chain's draws vs, with its Monte Carlo standard error."
  [vs]
  (let [{:keys [mean mcse-mean]} (diag/summary vs)]
    {:mean mean :se mcse-mean}))

[(chain-mean (map :slope chain)) (chain-mean (map :intercept chain))]
;=> [{:mean 0.63 :se 0.083} {:mean -0.854 :se 0.058}] ; float64

(outlier-frequencies chain)
;=> [0.845 0.784 0.201 0.163 0.467] ; float64
```

Two algorithms agree, and one of them started at the truth. That is
strong evidence that the posterior itself is the problem, but it is not
proof: a chain may still have failed to mix.

### Enumeration decides

A grid from −4 to 4 in steps of 0.25 on both the slope and the intercept,
with the flags, has 32768 points:

<!-- code: misspecified-enumeration -->
```clojure
(def wide-specs
  [[:slope (endpoints -4 4 0.25) :continuous]
   [:intercept (endpoints -4 4 0.25) :continuous]])

(def misspecified-grid
  (apply enum/choice-vol-grid (concat wide-specs flag-specs)))

(count misspecified-grid)
;=> 32768

(def misspecified-result
  (enum/enumerative-inference misspecified [xs] (observations steep-ys)
                              misspecified-grid
                              {:key (rng/key 8) :batched? true}))

[(expectation misspecified-result :slope)
 (expectation misspecified-result :intercept)]
;=> [0.689 -0.867]

(outlier-probabilities misspecified-result)
;=> [0.84 0.771 0.194 0.163 0.434]

(def misspecified-table
  (-> (array/array (probabilities misspecified-result))
      (array/reshape [32 32 32])
      (array/sum {:axis 2})
      array/->clj))
```

<!-- code: misspecified-figure -->
```clojure
;; figure: figures/024-misspecified-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.25)) (midpoints (endpoints -4 4 0.25))
             misspecified-table
             {:mark [-2.4 2.8] :marginals? true
              :x-label "intercept" :y-label "slope"})
```

![The posterior of the line under the misspecified model, with the true
line (cross)](figures/024-misspecified-posterior.svg)

The grid reaches far into the prior's tails on every side, and its log
marginal likelihood agrees with the exact one. Both algorithms agree with
the exact posterior means, within five of their standard errors:

<!-- code: misspecified-checks -->
```clojure
(def misspecified-exact (exact-posterior misspecified-priors steep-ys))

(array/allclose? (:log-ml-estimate misspecified-result)
                 (:log-ml misspecified-exact))
;=> true

(defn agrees?
  "Is the estimate {:mean :se} within five standard errors of x?"
  [{:keys [mean se]} x]
  (< (abs (- mean x)) (* 5 se)))

(let [[slope intercept] (:mean misspecified-exact)]
  [(agrees? (mean-and-se (map :slope resampled)) slope)
   (agrees? (mean-and-se (map :intercept resampled)) intercept)
   (agrees? (chain-mean (map :slope chain)) slope)
   (agrees? (chain-mean (map :intercept chain)) intercept)])
;=> [true true true true]
```

So neither algorithm failed. The posterior itself prefers a flat line with
outliers, and puts only 0.114 on slopes above 2:

<!-- code: steep-mass -->
```clojure
(defn probability-above
  "The posterior probability that the choice at addr exceeds x."
  [result addr x]
  (reduce + (keep (fn [[v p]] (when (> v x) p)) (enum/marginal result addr))))

(probability-above misspecified-result :slope 2.0)
;=> 0.114
```

Lines drawn from the enumerated posterior show it. Most are flat and pass
among the middle points; a few are steep:

<!-- code: misspecified-lines -->
```clojure
;; figure: figures/024-misspecified-lines.svg
(let [picks (w/multinomial (:log-normalized-weights misspecified-result) 40
                           (rng/key 9))
      line (fn [i]
             (let [[choices] (misspecified-grid i)
                   slope (cm/get-choice choices :slope)
                   intercept (cm/get-choice choices :intercept)]
               (fn [x] (+ (* slope x) intercept))))]
  (svg/plot [{:curves (map line picks) :opacity 0.25}
             {:curves [(fn [x] (+ (* 2.8 x) -2.4))] :color "black"}
             {:points (map vector xs steep-ys)}]))
```

![Forty lines drawn from the enumerated posterior (blue), the true line
(black) and the data](figures/024-misspecified-lines.svg)

The three methods agree on the outliers too:

<!-- code: outliers-figure -->
```clojure
;; figure: figures/024-outliers.svg
(let [shifted (fn [dx ps] (map vector (map #(+ % dx) xs) ps))]
  (svg/plot [{:points (shifted -0.15 (outlier-frequencies resampled))
              :color "steelblue" :r 4}
             {:points (shifted 0.0 (outlier-frequencies chain))
              :color "darkorange" :r 4}
             {:points (shifted 0.15 (outlier-probabilities misspecified-result))
              :r 4}]
            {:x-range [-2.5 2.5] :y-range [0 1] :y-label "P(outlier)"}))
```

![Each point's probability of being an outlier, by importance resampling
(blue), MCMC (orange) and enumeration (black)](figures/024-outliers.svg)

The model is misspecified: it does not capture what we believe these data
should teach it.

## Addressing model misspecification

The priors on the line are too narrow and the outlier probability too
high, so widening the one and lowering the other would repair the model
for these data (exercise 3 tries each). But other data might really be a
flat line with many outliers, and the model should be able to say so.
Gen.jl's remedy is *hyperpriors*: priors on the parameters of the priors.
The variances of the slope and the intercept get inverse-gamma priors, and
the outlier probability a uniform one, $\mathrm{Beta}(1, 1)$:

<!-- code: hierarchical-model -->
```clojure
(def hierarchical
  (gen [xs]
    (let [slope-var (trace :slope-var (d/inv-gamma 1 1))
          intercept-var (trace :intercept-var (d/inv-gamma 1 1))
          slope (trace :slope (d/normal 0 (array/sqrt slope-var)))
          intercept (trace :intercept (d/normal 0 (array/sqrt intercept-var)))
          prob-outlier (trace :prob-outlier (d/beta 1 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (let [outlier? (trace [:data i :outlier?] (d/bernoulli prob-outlier))]
          (trace [:data i :y]
                 (d/normal (array/where outlier? 0.0 (+ (* slope x) intercept))
                           (array/where outlier? 5.0 1.0))))))))
```

The latent space has grown by three dimensions, so the grid coarsens: 8
intervals on the slope and the intercept, 5 on each variance and on the
outlier probability, with the flags 256000 points. `:chunk-size` runs them
as batches of 32000 particles, to bound the memory of each batched run:

<!-- code: hierarchical-enumeration -->
```clojure
(def coarse-specs
  [[:slope (endpoints -4 4 1) :continuous]
   [:intercept (endpoints -4 4 1) :continuous]
   [:slope-var (endpoints 0 5 1) :continuous]
   [:intercept-var (endpoints 0 5 1) :continuous]
   [:prob-outlier (endpoints 0 1 0.2) :continuous]])

(def coarse-grid (apply enum/choice-vol-grid (concat coarse-specs flag-specs)))

(count coarse-grid)
;=> 256000

(def hierarchical-result
  (enum/enumerative-inference hierarchical [xs] (observations steep-ys)
                              coarse-grid
                              {:key (rng/key 10) :batched? true
                               :chunk-size 32000}))

(def hierarchical-table
  (-> (array/array (probabilities hierarchical-result))
      (array/reshape [8 8 (* 5 5 5 32)])
      (array/sum {:axis 2})
      array/->clj))
```

<!-- code: hierarchical-figure -->
```clojure
;; figure: figures/024-hierarchical-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 1)) (midpoints (endpoints -4 4 1))
             hierarchical-table
             {:labels? true :mark [-2.4 2.8] :marginals? true
              :x-label "intercept" :y-label "slope"})
```

![The posterior of the line under the hierarchical model, on the coarse
grid](figures/024-hierarchical-posterior.svg)

The most probable cell now holds the true line, with probability 0.622. A
second, smaller mode remains near the flat line with outliers:

<!-- code: hierarchical-mode -->
```clojure
(defn most-probable-cell
  "The [row column] of a table's largest value."
  [table]
  (apply max-key #(get-in table %)
         (for [i (range (count table)) j (range (count (first table)))] [i j])))

(let [[i j] (most-probable-cell hierarchical-table)]
  [(nth (midpoints (endpoints -4 4 1)) i) (nth (midpoints (endpoints -4 4 1)) j)
   (get-in hierarchical-table [i j])])
;=> [2.5 -2.5 0.622]
```

Why does the new model work? Gen.jl's tutorial looks at the conditional
posterior of the hyperparameters, given the true line. Fixing the line
leaves room for a finer grid, 10 intervals on each hyperparameter:

<!-- code: hyperparameters -->
```clojure
(def at-truth
  (cm/merge (observations steep-ys) (cm/choicemap :slope 2.8 :intercept -2.4)))

(def hyper-specs
  [[:slope-var (endpoints 0 5 0.5) :continuous]
   [:intercept-var (endpoints 0 5 0.5) :continuous]
   [:prob-outlier (endpoints 0 1 0.1) :continuous]])

(def hyper-result
  (enum/enumerative-inference hierarchical [xs] at-truth
                              (apply enum/choice-vol-grid
                                     (concat hyper-specs flag-specs))
                              {:key (rng/key 11) :batched? true}))

(defn cell-probabilities
  "The marginal of the choice at addr on a continuous grid, in the order of
   its cells."
  [result addr]
  (mapv val (sort (enum/marginal result addr))))
```

Two of these conditionals can be checked exactly. Given the slope $s$, the
slope's variance depends on nothing else, and its inverse-gamma prior is
conjugate to the normal: the conditional is $\mathrm{InvGamma}(1 + 1/2,\
1 + s^2/2)$, which is $\mathrm{InvGamma}(1.5, 4.92)$ at $s = 2.8$. On the
grid's cells:

<!-- code: conjugate -->
```clojure
(defn normalized
  "The numbers xs scaled to sum to 1."
  [xs]
  (let [total (reduce + xs)] (mapv #(/ % total) xs)))

(defn on-cells
  "The probabilities a density f gives cells of equal width: its values at
   their midpoints, normalized, as the grid weights them."
  [f mids]
  (normalized (mapv #(array/item (f %)) mids)))

(def variance-cells (midpoints (endpoints 0 5 0.5)))

(array/allclose?
 (array/array (cell-probabilities hyper-result :slope-var))
 (array/array (on-cells #(array/exp (dc/log-prob (d/inv-gamma 1.5 4.92) %))
                        variance-cells)))
;=> true
```

Given the line, the outlier probability's conditional is its flat prior
times the probability of the data with the flags summed out, $\prod_i
p(y_i \mid \text{line})$, which `point-likelihood` from the discrete
model's check computes:

<!-- code: prob-outlier -->
```clojure
(def probability-cells (midpoints (endpoints 0 1 0.1)))

(def prob-outlier-cells
  (on-cells (fn [p] (reduce * (map #(point-likelihood p 2.8 -2.4 %1 %2)
                                   xs steep-ys)))
            probability-cells))

(array/allclose? (array/array (cell-probabilities hyper-result :prob-outlier))
                 (array/array prob-outlier-cells))
;=> true

(defn cell-mean
  "The mean of cells with these midpoints and probabilities."
  [mids ps]
  (reduce + (map * mids ps)))

[(cell-mean variance-cells (cell-probabilities hyper-result :slope-var))
 (cell-mean variance-cells
            (on-cells #(array/exp (dc/log-prob (d/inv-gamma 1 1) %))
                      variance-cells))
 (cell-mean probability-cells prob-outlier-cells)]
;=> [2.728 1.468 0.188]
```

The slope's variance has a conditional mean of 2.73, against 1.47 for its
prior on the same cells, and the outlier probability's is 0.19, against
the prior's 0.5. Given a steep line, the model raises the variance to
allow it and lowers the outlier probability, instead of forcing the line
flat:

<!-- code: prob-outlier-figure -->
```clojure
;; figure: figures/024-prob-outlier.svg
(svg/histogram [{:values probability-cells :label "prior"}
                {:values probability-cells :label "posterior"
                 :weights (cell-probabilities hyper-result :prob-outlier)}]
               {:range [0 1] :bins 10 :x-label "prob-outlier"})
```

![The outlier probability's prior and its conditional posterior given the
true line](figures/024-prob-outlier.svg)

Having checked the model on a coarse grid, we can use MCMC for a finer
picture. The kernel adds a move on each hyperparameter, resampling it
from its prior:

<!-- code: hierarchical-mcmc -->
```clojure
(defn hierarchical-sweep
  "mcmc-sweep, then an MH move on each hyperparameter that resamples it
   from its prior."
  [tr key]
  (let [[k-sweep & ks] (rng/split key 4)]
    (reduce (fn [tr [addr k]] (:trace (mh/mh tr (sel/select addr) k)))
            (mcmc-sweep tr k-sweep)
            (map vector [:slope-var :intercept-var :prob-outlier] ks))))

(def hierarchical-chain
  (mcmc-draws hierarchical-sweep
              (:trace (gfi/generate hierarchical [xs] (observations steep-ys)
                                    (rng/key 12)))
              10000 500 (rng/key 13)))

(def steep-fraction
  (chain-mean (map #(if (> (:slope %) 1.5) 1.0 0.0) hierarchical-chain)))

steep-fraction
;=> {:mean 0.905 :se 0.031} ; float64

(> (- (:mean steep-fraction) (* 5 (:se steep-fraction))) 0.5)
;=> true
```

<!-- code: hierarchical-mcmc-figure -->
```clojure
;; figure: figures/024-hierarchical-mcmc.svg
(let [cells (midpoints (endpoints -4 4 0.25))
      bin (fn [v] (long (Math/floor (* 4 (+ v 4)))))
      counts (frequencies (for [{:keys [slope intercept]} hierarchical-chain
                                :when (and (< -4 slope 4) (< -4 intercept 4))]
                            [(bin slope) (bin intercept)]))]
  (svg/heatmap cells cells
               (mapv (fn [i] (mapv (fn [j] (get counts [i j] 0)) (range 32)))
                     (range 32))
               {:mark [-2.4 2.8] :marginals? true
                :x-label "intercept" :y-label "slope"}))
```

![MCMC draws of the line under the hierarchical model, binned on a fine
grid](figures/024-hierarchical-mcmc.svg)

The chain spends nine tenths of its time on slopes above 1.5, near the
true line, where the misspecified model put hardly any probability.

## Enumerating inside inference

Each grid so far enumerated the five flags along with the line, so it had
32 times the points it would have without them. gen.clj can instead sum
finite choices out inside `generate`. `enum/with-exact-proposal` wraps a
model, keeping its distribution and its traces, and replaces its internal
proposal for the choices at the given addresses: `generate` draws them
from their exact conditional given every other choice, which it finds by
enumerating their values. When every other choice is constrained, its
weight is the log probability of the constrained choices with the
enumerated ones summed out. With every latent choice of the discrete model
enumerated, one `generate` gives $\log p(ys)$ exactly, whatever the key.
With `{:batched? true}`, the wrapper runs the 800 assignments it enumerates
as one batched execution:

<!-- code: exact-proposal -->
```clojure
(def flag-addresses (mapv #(vector :data % :outlier?) (range (count xs))))

(def every-choice
  (enum/with-exact-proposal discrete-regression
                            (into [:slope :intercept] flag-addresses)
                            {:batched? true}))

(mapv #(:weight (gfi/generate every-choice [xs] (observations discrete-ys) %))
      [(rng/key 14) (rng/key 15)])
;=> [-12.444 -12.444]

(array/allclose? (:weight (gfi/generate every-choice [xs]
                                        (observations discrete-ys)
                                        (rng/key 14)))
                 (:log-ml discrete-exact))
;=> true
```

With only the flags enumerated, a grid over the line suffices: 384 points
in place of 12288, with the same log marginal likelihood, and each point's
probability the sum of its 32 points in the full grid:

<!-- code: summed-grid -->
```clojure
(def summed-regression
  (enum/with-exact-proposal continuous-regression flag-addresses
                            {:batched? true}))

(def line-grid (apply enum/choice-vol-grid line-specs))

(count line-grid)
;=> 384

(def summed-result
  (enum/enumerative-inference summed-regression [xs]
                              (observations continuous-ys) line-grid
                              {:key (rng/key 16) :batched? true}))

(array/allclose? (:log-ml-estimate summed-result)
                 (:log-ml-estimate continuous-result))
;=> true

(def summed-flags
  "The full grid's probabilities summed over the flags, point by point of
   the line's grid."
  (map #(reduce + %) (partition 32 (probabilities continuous-result))))

(array/allclose? (array/array (probabilities summed-result))
                 (array/array summed-flags))
;=> true
```

The hierarchical model's grid shrinks the same way, from 256000 points to
8000:

<!-- code: summed-hierarchical -->
```clojure
(def summed-hierarchical
  (enum/enumerative-inference (enum/with-exact-proposal hierarchical
                                                         flag-addresses)
                              [xs] (observations steep-ys)
                              (apply enum/choice-vol-grid coarse-specs)
                              {:key (rng/key 17) :batched? true}))

(array/allclose? (:log-ml-estimate summed-hierarchical)
                 (:log-ml-estimate hierarchical-result))
;=> true
```

An importance sampler gains too. With the exact proposal, a sample's
weight is the likelihood of its line with the flags summed out, so the
flags no longer add noise to the weights. On the continuous model, the
effective sample size of 2000 samples rises from about 30 to about 250,
and both estimates of the mean slope agree with the exact one:

<!-- code: summed-importance -->
```clojure
(defn importance-estimate
  "The importance estimate of the posterior mean of the choice at addr, n
   samples of model given ys, with its standard error, about the root of
   the posterior variance over the ESS: {:mean :se :ess}."
  [model ys addr n key]
  (let [run (fn [f] (imp/estimate f model [xs] (observations ys) n
                                  {:key key :batched? true}))
        {mean :estimate ess :ess} (run #(array/item (% addr)))
        {square :estimate} (run #(array/square (array/item (% addr))))]
    {:mean mean :se (Math/sqrt (/ (- square (* mean mean)) ess)) :ess ess}))

(def plain-estimate
  (importance-estimate continuous-regression continuous-ys :slope 2000
                       (rng/key 18)))

(def summed-estimate
  (importance-estimate summed-regression continuous-ys :slope 2000
                       (rng/key 18)))

[(:ess plain-estimate) (:ess summed-estimate)]
;=> [28.8 254.2] ; float64

[(< (:ess plain-estimate) 100) (> (:ess summed-estimate) 150)]
;=> [true true]

(let [slope (first (:mean continuous-exact))]
  [(agrees? plain-estimate slope) (agrees? summed-estimate slope)])
;=> [true true]
```

In MCMC, `mh` on the wrapped model's flags is a Gibbs move: it draws them
from their exact conditional, and is always accepted:

<!-- code: gibbs -->
```clojure
(def summed-trace
  (:trace (gfi/generate summed-regression [xs] (observations continuous-ys)
                        (rng/key 19))))

(select-keys (mh/mh summed-trace (apply sel/select flag-addresses) (rng/key 20))
             [:accepted? :log-alpha])
;=> {:accepted? true :log-alpha 0.0}
```

Enumeration needs a latent space small enough to cover, even when part of
it is summed out this way. A moving object has a latent position at every
time step, and no grid could cover them all. The next chapter, *Sequential
Monte Carlo: Object Tracking*, conditions such a model one step at a time.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **A grid too narrow.** Enumerate the continuous model on slopes from
   −1 to 1 only. Predict the posterior mean of the slope. Without the exact
   answer, how could the enumeration show that its grid is too narrow?
2. **A kernel that forgets.** Drop the moves on the flags from the MCMC
   kernel, and start a chain on the misspecified model at the true line
   with no outliers. Predict the chain's mean slope. Which posterior does
   the chain sample?
3. **Prior or outlier rate?** In the misspecified model, either widen the
   priors on the line to standard deviation 2, or lower the outlier
   probability to 0.02. Predict which change recovers the steep line, and
   enumerate both.
4. **A large ESS, a wrong answer.** Give the misspecified model the exact
   proposal for its flags, and estimate its mean slope from 2000 importance
   samples. Predict the effective sample size, and whether the estimate
   agrees with the exact 0.689.

## Solutions

**1.** The grid cuts off the mode around the true line, whose slope is
−1.21, so it finds a posterior mean of −0.23 where the exact one is
−0.69. Its own probabilities give it away. A grid wide enough leaves
almost nothing in its first and last cells, $2.8 \times 10^{-5}$ for the
slope on the full grid. On the narrow one the edge cells hold 0.22: the
posterior runs off the grid's edge.

<!-- code: solution-1 -->
```clojure
(def narrow-result
  (enum/enumerative-inference continuous-regression [xs]
                              (observations continuous-ys)
                              (apply enum/choice-vol-grid
                                     [:slope (endpoints -1 1 0.25) :continuous]
                                     (second line-specs)
                                     flag-specs)
                              {:key (rng/key 21) :batched? true}))

[(expectation narrow-result :slope) (first (:mean continuous-exact))]
;=> [-0.233 -0.687]

(defn edge-mass
  "The posterior probability of the first and last cells of the choice at
   addr on a continuous grid."
  [result addr]
  (let [ps (cell-probabilities result addr)] (+ (first ps) (peek ps))))

[(edge-mass narrow-result :slope) (edge-mass continuous-result :slope)]
;=> [0.223 2.8E-5]
```

**2.** The flags never change, so the chain samples the posterior of the
line given its starting flags: no outliers. That posterior is normal,
with the mean of *Getting Started*'s formula, slope 2.11. The chain agrees
with it, and not with the full posterior's 0.69. Its standard error, 0.01,
is small, and nothing in the chain itself looks wrong; only an exact
answer shows which posterior it sampled.

<!-- code: solution-2 -->
```clojure
(defn line-sweep
  "mcmc-sweep without its moves on the flags."
  [tr key]
  (:trace (mh/mh tr line-proposal [] key)))

(def no-outliers
  (apply cm/choicemap (mapcat (fn [addr] [addr false]) flag-addresses)))

(def stuck
  (mcmc-draws line-sweep
              (:trace (gfi/generate misspecified [xs]
                                    (cm/merge at-truth no-outliers)
                                    (rng/key 22)))
              5000 500 (rng/key 23)))

(def all-inliers
  (given-flags misspecified-priors steep-ys [false false false false false]))

(:mean all-inliers)
;=> [2.111 -2.016]

(chain-mean (map :slope stuck))
;=> {:mean 2.111 :se 0.01} ; float64

(let [slope (chain-mean (map :slope stuck))]
  [(agrees? slope (first (:mean all-inliers)))
   (agrees? slope (first (:mean misspecified-exact)))])
;=> [true false]
```

**3.** Both help, the wider priors more. With standard deviation 2, a
steep line is no longer improbable a priori, and the posterior puts 0.92
on slopes above 2. With rare outliers, the flat line's explanation costs
more, and the probability rises to 0.63, but the narrow priors still
resist the steep line. The hierarchical model lets the data choose both.

<!-- code: solution-3 -->
```clojure
(defn enumerate-steep
  "The misspecified model's enumeration on the wide grid, with other priors."
  [priors]
  (enum/enumerative-inference (robust-regression priors) [xs]
                              (observations steep-ys) misspecified-grid
                              {:key (rng/key 25) :batched? true}))

(def wide-priors
  (assoc misspecified-priors
         :slope-prior (d/normal 0 2) :intercept-prior (d/normal 0 2)))

(def rare-outliers (assoc misspecified-priors :prob-outlier 0.02))

(for [priors [wide-priors rare-outliers]
      :let [result (enumerate-steep priors)]]
  [(expectation result :slope) (probability-above result :slope 2.0)])
;=> ([2.467 0.919] [2.083 0.632])
```

**4.** The flags no longer add noise, so the effective sample size is
large: 564 of 2000 in this run. Yet the estimate, 0.35, is eleven of its
standard errors below the exact 0.689. The steep mode is the reason. The
posterior gives slopes above 1.5 with intercepts below −1.5 probability
0.16, but the prior proposes such lines so rarely that 2000 samples hold
only 0.54 of them on average, and 58% of runs hold none. A sampler that never
reaches a region cannot weigh it, and the effective sample size, computed
from the samples' weights, cannot see what no sample reached. The exact
answer can.

<!-- code: solution-4 -->
```clojure
(def misleading
  (importance-estimate (enum/with-exact-proposal misspecified flag-addresses)
                       steep-ys :slope 2000 (rng/key 26)))

[(:ess misleading) (:mean misleading) (:se misleading)]
;=> [564.4 0.35 0.031] ; float64

(/ (- (:mean misleading) (first (:mean misspecified-exact))) (:se misleading))
;=> -10.9 ; float64

(defn steep-mass
  "The probability a [slope][intercept] table on the wide grid gives the
   steep lines: slopes above 1.5 with intercepts below -1.5."
  [table]
  (let [cells (midpoints (endpoints -4 4 0.25))]
    (reduce + (for [[i slope] (map-indexed vector cells)
                    [j intercept] (map-indexed vector cells)
                    :when (and (> slope 1.5) (< intercept -1.5))]
                (get-in table [i j])))))

(def prior-table
  "The misspecified prior of the line on the wide grid's cells."
  (let [cells (midpoints (endpoints -4 4 0.25))
        ps (on-cells #(array/exp (dc/log-prob (d/normal 0 (Math/sqrt 0.5)) %))
                     cells)]
    (mapv (fn [p] (mapv #(* p %) ps)) ps)))

[(steep-mass misspecified-table) (* 2000 (steep-mass prior-table))]
;=> [0.164 0.544]

(Math/pow (- 1.0 (steep-mass prior-table)) 2000)
;=> 0.58
```
