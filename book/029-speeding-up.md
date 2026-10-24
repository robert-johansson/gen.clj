# Speeding Up

Every inference program in this Part is a loop of GFI operations: a
generate, then updates and regenerates by the thousand. This chapter is
about making those loops cheaper without changing what they compute.
gen.clj has three ways to do it. The *static language* gives a model a
graph, so that an update reruns only the parts of the model a change
reaches. *Batched execution* runs $N$ particles in one execution of the
model, so that the handler's work at each site is done once instead of $N$
times. On MLX, `array/compile` records a step's array operations as one
graph, which MLX fuses and replays.

None of the three changes what an operation means. *The Handler Loop* put
the rule this way: the handler is the semantics, and a faster path must be
observationally equivalent to it. So each section below checks two things:
that the faster path does less work, and that it gives what the handler
gives. That is the same choices under the same keys where gen.clj promises
it, and the same distribution where it promises only that.

This chapter ports Gen.jl's *Speeding Up Inference with the Static Modeling
Language* and the GenMLX book's *Going Fast: Vectorization and the GPU*.
Gen.jl's tutorial times its programs. Times depend on the machine, the JIT
and whatever else is running, so the checks here count the work instead,
and the text cites the speed-ups measured in `docs/benchmarks.md`. The
chapter's code lives in one namespace:

<!-- code: ns -->
```clojure
(ns book.speeding-up
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))
```

## A slow inference program

Gen.jl's tutorial speeds up inference in a *robust regression*: points
near a line, some of which are outliers. Each point $i$ has a flag $o_i$.
An inlier lies near the line, and an outlier is drawn from a broad normal
around 0:

$$
\text{slope}, \text{intercept} \sim \mathcal{N}(0, 2), \qquad
\text{noise} \sim \Gamma(1, 1), \qquad
p \sim \mathcal{U}(0, 1),
$$

$$
o_i \sim \text{Bernoulli}(p), \qquad
y_i \sim \begin{cases}
\mathcal{N}(0, 10) & \text{if } o_i, \\
\mathcal{N}(\text{slope} \cdot x_i + \text{intercept},\ \text{noise})
& \text{otherwise.}
\end{cases}
$$

Written as Gen.jl writes it, the points are a loop in the body:

<!-- code: loop-model -->
```clojure
(def loop-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (vec (map-indexed
            (fn [i x]
              (let [[mu sd] (if (trace [:data i :is-outlier]
                                       (d/bernoulli prob-outlier))
                              [0.0 10.0]
                              [(+ (* x slope) intercept) noise])]
                (trace [:data i :y] (d/normal mu sd))))
            xs)))))
```

`d/gamma` takes a shape and a scale, as Gen.jl's `gamma` does. The
inference program is MH by *block resimulation*. Each move selects a block
of choices and resimulates it from its prior, which is regenerate's
internal proposal (*Update and Regenerate*). One sweep moves the line's
three parameters together, then each point's flag in turn, then the
outlier probability:

<!-- code: block-update -->
```clojure
(defn sweep
  "MH over tr, one move for each selection of blocks in turn, each
   resimulating the selected choices from their prior."
  [tr blocks key]
  (reduce (fn [tr [block k]] (:trace (mh/mh tr block k)))
          tr
          (map vector blocks (rng/split key (count blocks)))))

(defn block-resimulation-update
  "One sweep of block resimulation over tr: the line's parameters, then
   each point's outlier flag in turn, then the outlier probability."
  [tr key]
  (let [[xs] (:args tr)]
    (sweep tr
           (concat [(sel/select :noise :slope :intercept)]
                   (for [i (range (count xs))]
                     (sel/select [:data i :is-outlier]))
                   [(sel/select :prob-outlier)])
           key)))
```

The program generates a trace from the observations and runs the sweeps.
It calls `gfi/eval-trace!` after each, as an inference loop should: on MLX
that evaluates the trace's arrays, so the lazy graph stays one sweep deep,
and on the JVM backend it does nothing.

<!-- code: inference -->
```clojure
(defn observations
  "The choice map of the y values ys."
  [ys]
  (reduce (fn [c [i y]] (cm/set-choice c [:data i :y] y))
          cm/EMPTY
          (map-indexed vector ys)))

(defn block-resimulation-inference
  "A trace of model on xs given ys, after iters sweeps of block
   resimulation."
  [model xs ys iters key]
  (let [[k-init k-sweeps] (rng/split key)
        tr (:trace (gfi/generate model [xs] (observations ys) k-init))]
    (reduce (fn [tr k] (gfi/eval-trace! (block-resimulation-update tr k)))
            tr
            (rng/split k-sweeps iters))))
```

## Counting the work

Gen.jl times this program as the number of points grows, and finds the
time quadratic in it. Here we count instead. *The Handler Loop* wrapped a
transition in middleware to watch the states it receives. `counted` wraps
the regenerate transition to count the points it visits, the sites at a
path ending in `:y`. The transition it calls is the handler's own, so the
model means what it meant. `sweep-visits` runs one sweep on random data
and returns the count:

<!-- code: counted -->
```clojure
(defn counted
  "gf with its regenerate transition wrapped to count, in the atom visits,
   the points it visits: the sites at a path ending in :y. The transition
   it wraps is the handler's own, so gf means what it meant."
  [gf visits]
  (let [regenerate (dyn/site-transitions :regenerate)]
    (dyn/with-transitions gf
      {:regenerate (fn [state p dist k]
                     (when (= :y (peek p)) (swap! visits inc))
                     (regenerate state p dist k))})))

(defn sweep-visits
  "The points one sweep visits, on n random points, of the model that
   (make visits) builds. The sweep is block-resimulation-update, or the
   function of a trace and a key given."
  [make n key & [sweep-fn]]
  (let [visits (atom 0)
        [k-x k-y k-init k-sweep] (rng/split key 4)
        xs (array/->clj (rng/uniform k-x [n]))
        ys (array/->clj (rng/uniform k-y [n]))
        tr (:trace (gfi/generate (make visits) [xs] (observations ys) k-init))]
    ((or sweep-fn block-resimulation-update) tr k-sweep)
    @visits))
```

One sweep on $n$ points visits $n(n + 2)$ of them:

<!-- code: loop-visits -->
```clojure
(def sizes [1 3 7 10 30 70 100])

(def loop-visits
  (mapv #(sweep-visits (fn [v] (counted loop-model v)) % (rng/key 1)) sizes))

loop-visits
;=> [3 15 63 120 960 5040 10200]

(= loop-visits (mapv #(* % (+ % 2)) sizes))
;=> true
```

The sweep makes $n + 2$ moves, and each move's regenerate runs the whole
body, visiting all $n$ points. A dynamic body is opaque to the handler,
which cannot know what a changed choice reaches without running the body.
Yet a flag's move needs only its own point. Given the line, the noise and
the outlier probability, point $i$'s flag and $y_i$ are independent of
every other point. So a sweep could visit $3n$ points: $n$ for the line's
move, one for each flag's, and $n$ for the outlier probability's.

## Map is not enough

Gen.jl's first rewrite makes that independence explicit with the Map
combinator (*Composition*). A kernel models one point:

<!-- code: point -->
```clojure
(def point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          [mu sd] (if outlier? [0.0 10.0] [(+ (* x slope) intercept) noise])]
      (trace :y (d/normal mu sd)))))
```

The model draws the four global choices and maps the kernel over the
points. It takes the kernel as an argument, so that we can count the
kernel's visits below:

<!-- code: with-map -->
```clojure
(defn with-map
  "The robust regression with its points made by a Map of kernel."
  [kernel]
  (let [points (comb/map kernel)]
    (gen [xs]
      (let [slope (trace :slope (d/normal 0 2))
            intercept (trace :intercept (d/normal 0 2))
            noise (trace :noise (d/gamma 1 1))
            prob-outlier (trace :prob-outlier (d/uniform 0 1))
            n (count xs)]
        (splice :data points xs (repeat n prob-outlier) (repeat n noise)
                (repeat n slope) (repeat n intercept))))))

(def model-with-map (with-map point))
```

Map puts element $i$'s choices under address $i$, and the call is at
`:data`, so the model has the loop model's addresses: `[:data 0
:is-outlier]`, `[:data 0 :y]` and so on. It also has the same density.
Assessed on the same choices, the two models weigh the same, so the
inference program needs no change:

<!-- code: same-model -->
```clojure
(def xs5 [0.0 1.0 2.0 3.0 4.0])

(def five (gfi/simulate loop-model [xs5] (rng/key 2)))

(take 4 (cm/addresses (:choices five)))
;=> [[:data 0 :is-outlier] [:data 0 :y] [:data 1 :is-outlier] [:data 1 :y]]

(= (cm/addresses (:choices five))
   (cm/addresses (:choices (gfi/simulate model-with-map [xs5] (rng/key 2)))))
;=> true

(array/allclose?
 (:weight (gfi/assess loop-model [xs5] (:choices five) (rng/key 3)))
 (:weight (gfi/assess model-with-map [xs5] (:choices five) (rng/key 3))))
;=> true
```

It does no better:

<!-- code: map-visits -->
```clojure
(def map-visits
  (mapv #(sweep-visits (fn [v] (with-map (counted point v))) % (rng/key 1))
        sizes))

(= map-visits loop-visits)
;=> true
```

A Map revisits only the elements whose arguments may have changed, or that
a constraint or a selection reaches (*Composition*). It learns which
arguments changed from the argdiffs it is given. In a dynamic body, a
splice compares each argument with the previous call's. One that is the
same object, here `xs`, has not changed; any other may have, since the
handler does not know how it was computed. The four `(repeat n …)`
sequences are made anew each time the body runs. So every move tells the
Map that every element's arguments may have changed, and the Map revisits
them all. Gen.jl's dynamic language passes no change information to a
callee at all. gen.clj's knows an unchanged argument by its identity,
which is not enough here.

## The static language

`gen ^:static` declares a body whose structure is fixed. It is one `let`,
whose bindings are calls at addresses that start with a constant (`trace`,
`splice`) or pure expressions of the arguments and earlier bindings,
followed by the return expression, which may itself be a call. Branching
happens only inside called generative functions. `with-map`'s body is such
a body, so the flag is the only change:

<!-- code: static-model -->
```clojure
(defn static-with-map
  "with-map's model in the static language: the same body, flagged."
  [kernel]
  (let [points (comb/map kernel)]
    (gen ^:static [xs]
      (let [slope (trace :slope (d/normal 0 2))
            intercept (trace :intercept (d/normal 0 2))
            noise (trace :noise (d/gamma 1 1))
            prob-outlier (trace :prob-outlier (d/uniform 0 1))
            n (count xs)]
        (splice :data points xs (repeat n prob-outlier) (repeat n noise)
                (repeat n slope) (repeat n intercept))))))

(def static-model-with-map (static-with-map point))
```

The flag is not a second semantics. The body runs under the handler, with
the handler's own `trace` and `splice`, and the same body without the flag
is the dynamic function it was. What the flag adds is the body's *graph*,
which the macro builds as it expands the body: a node for each binding,
and for each node the names it reads (`static/ir`).

<!-- code: graph -->
```clojure
(def graph (static/ir static-model-with-map))

(mapv :kind (:nodes graph))
;=> [:trace :trace :trace :trace :compute :splice]

(mapv :inputs (:nodes graph))
;=> [[] [] [] [] [xs] [xs n prob-outlier noise slope intercept]]
```

There are four choices, a computation (`n`, which reads `xs`) and a call,
whose arguments read `xs`, `n` and the four choices. `drawing` turns the
graph into nodes and edges for a figure. `static/dot` renders it in
Graphviz's DOT, as Gen.jl's `draw_graph` does:

<!-- code: drawing -->
```clojure
(defn drawing
  "The nodes and edges of a static generative function's graph, for
   svg/dag: an argument as text, a choice as an ellipse, a call as a box, a
   computation as a diamond and the return value as a circle, each node
   named by its address or by the name it binds."
  [gf]
  (let [{:keys [args nodes return]} (static/ir gf)
        shapes {:trace :ellipse :splice :box :compute :diamond}
        id (fn [n] (or (:addr n) (first (:names n))))
        binder (into {} (for [n nodes s (:names n)] [s (id n)]))
        source #(get binder % %)]
    {:nodes (concat (for [a (apply concat (:fixed args))] {:id a :shape :text})
                    (for [n nodes] {:id (id n) :shape (shapes (:kind n))})
                    [{:id 'return :shape :circle}])
     :edges (concat (for [n nodes s (:inputs n)] [(source s) (id n)])
                    (for [s (:inputs return)] [(source s) 'return]))}))

(str/includes? (static/dot static-model-with-map) "\":noise\" -> \":data\"")
;=> true
```

<!-- code: graph-figure -->
```clojure
;; figure: figures/029-graph.svg
(let [{:keys [nodes edges]} (drawing static-model-with-map)]
  (svg/dag nodes edges))
```

![The static model's graph: the argument `xs`, the four choices
(ellipses), the computation `n` (a diamond), the call at `:data` (a box)
and the return value (a circle). An arrow goes from each node to each node
that reads it.](figures/029-graph.svg)

A body that is not static is a compile error when the macro expands it.
The loop model's points are a loop, which runs `trace` any number of
times:

<!-- code: not-static -->
```clojure
;; error: Syntax error macroexpanding gen
(gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 2))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (* slope x) 1)))
      slope))
```

`static/could-be-static` runs the flag's analysis on a dynamic function's
source and says why it fails, and `static/lint` runs it on every
generative function of a namespace. The flag is never switched on for
you. GenMLX analyzes every body automatically, and DESIGN.md records that
its triage found that analysis the least reliable part of GenMLX.

<!-- code: lint -->
```clojure
(static/could-be-static model-with-map)
;=> {:static? true}

(let [{:keys [static? reason]} (static/could-be-static loop-model)]
  [static? (str/includes? reason "calls trace, which may run any number")])
;=> [false true]
```

Gen.jl's static language is stricter: no loops, a unique name on the left
of every assignment, and a return of one name. In gen.clj a binding may be
any pure Clojure, destructuring included, and the return expression may be
a call.

## What an update skips

In update and regenerate, the static walk reruns a node only when one of
its inputs changed or a constraint or a selection reaches it. A node it
keeps keeps its value and its record, with weight 0, as an unchanged rerun
would leave them. It also consumes its key, so that the walk draws what
the dynamic body draws. A sweep of the static model visits $3n$ points:

<!-- code: static-visits -->
```clojure
(def static-visits
  (mapv #(sweep-visits (fn [v] (static-with-map (counted point v))) %
                       (rng/key 1))
        sizes))

static-visits
;=> [3 9 21 30 90 210 300]

(= static-visits (mapv #(* 3 %) sizes))
;=> true
```

<!-- code: visits-figure -->
```clojure
;; figure: figures/029-visits.svg
(svg/plot [{:curves [#(* % (+ % 2)) #(* 3 %)] :color "gray" :width 1}
           {:points (map vector sizes loop-visits) :color "darkorange"}
           {:points (map vector sizes static-visits) :color "steelblue"}]
          {:x-range [0 102] :x-label "number of data points"
           :y-label "points visited in one sweep"})
```

![Points visited by one sweep of block resimulation. The dynamic models
(orange; the loop and the Map visit the same) visit $n(n + 2)$, the static
model (blue) $3n$. The gray curves are the two
formulas.](figures/029-visits.svg)

Gen.jl's measured times follow the same curves, quadratic for the loop and
the Map and linear for the static model. On gen.clj's JVM backend,
`docs/benchmarks.md` measures an update that skips a Map of 1000
observations about 130 times faster with the flag than without it, and an
update that must rerun the Map about as fast.

A flag's move shows where the difference comes from. Regenerating point
2's flag, the dynamic model's Map revisits all five points and the static
model's revisits point 2 alone. `revisited` lists the points whose
subtraces changed, as in *Composition*:

<!-- code: revisited -->
```clojure
(defn revisited
  "The indices of the points whose subtraces differ between traces a and b
   of a model whose points are a Map at :data."
  [a b]
  (let [pa (trace/subtrace a :data)
        pb (trace/subtrace b :data)]
    (vec (remove #(identical? (trace/subtrace pa %) (trace/subtrace pb %))
                 (range (count (:retval pa)))))))

(defn five-points
  "A trace of model on xs5, given five y values."
  [model]
  (:trace (gfi/generate model [xs5] (observations [1 2 3 4 5]) (rng/key 4))))

(defn flag-moved
  "The points that regenerating point 2's outlier flag revisits, in a
   trace of model."
  [model]
  (let [tr (five-points model)
        moved (gfi/regenerate tr (sel/select [:data 2 :is-outlier])
                              (rng/key 5))]
    (revisited tr (:trace moved))))

[(flag-moved model-with-map) (flag-moved static-model-with-map)]
;=> [[0 1 2 3 4] [2]]
```

The selection reaches the call at `:data`, so the static walk reruns it.
But it hands the Map argdiffs from the graph. `(repeat n noise)` reads
`n` and `noise`, neither of which changed, so its argdiff is no-change,
though the sequence is a new one. Told that no argument changed, the Map
revisits only the point the selection reaches. The line's move and the
outlier probability's change an input of every point, so they revisit all
$n$: those are the two $n$s of $3n$. Exercise 2 traces each move through
the graph.

## The same chain

Skipping must not change the chain. The *static law* in
`test/gen/laws.clj`, which the tests run on every static model they
build, requires the static and the dynamic function of one body to give
the same choices under the same keys, with weights and scores equal up to
rounding. Here it is for whole chains. Twenty sweeps of each model, from
the same key, end at the same choices:

<!-- code: same-chain -->
```clojure
(defn same-choices?
  "Do traces a and b hold the same choices, exactly, and scores that agree
   up to rounding?"
  [a b]
  (let [ca (:choices a)
        cb (:choices b)]
    (and (= (cm/addresses ca) (cm/addresses cb))
         (every? true? (map array/equal? (cm/values ca) (cm/values cb)))
         (array/allclose? (:score a) (:score b)))))

(def xs10 (array/->clj (rng/uniform (rng/key 6) [10])))

(def ys10 (mapv #(+ (* 2.0 %) 0.5) xs10))

(defn chain
  "model's trace after 20 sweeps on (xs10, ys10), from one key."
  [model]
  (block-resimulation-inference model xs10 ys10 20 (rng/key 7)))

(same-choices? (chain model-with-map) (chain static-model-with-map))
;=> true
```

The scores agree up to rounding, not bit for bit: a kept node keeps its
old log density, and the Map's total is then summed in another order.

Gen.jl's last rewrite makes the kernel static too. `point`'s body is
static as it stands, since its bindings are a call and a destructured
`if`, a pure expression. (Gen.jl's static language takes only the ternary
form of `if`.) The kernel with the flag makes a sweep visit the same
points, and gives the same chain:

<!-- code: static-point -->
```clojure
(static/could-be-static point)
;=> {:static? true}

(def static-point
  (gen ^:static [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          [mu sd] (if outlier? [0.0 10.0] [(+ (* x slope) intercept) noise])]
      (trace :y (d/normal mu sd)))))

(def fully-static-model (static-with-map static-point))

[(sweep-visits (fn [v] (static-with-map (counted static-point v))) 30
               (rng/key 1))
 (same-choices? (chain static-model-with-map) (chain fully-static-model))]
;=> [90 true]
```

A static kernel cannot change how many points a sweep visits, since the
Map decides that. It changes what each visit costs. A static function's
assess runs a *score function* that the macro generates from the graph:
straight-line code that reads each choice and adds its log density,
without the handler. It gives the handler's weight:

<!-- code: static-assess -->
```clojure
(let [args [1.0 0.1 0.5 -1.0 2.0]
      choices (cm/choicemap :is-outlier false :y 1.2)]
  (array/allclose? (:weight (gfi/assess static-point args choices (rng/key 0)))
                   (:weight (gfi/assess point args choices (rng/key 0)))))
;=> true
```

Everything built on assess uses it: HMC, MALA and MAP, choice gradients,
MH's backward densities. `docs/benchmarks.md` measures the static assess
on the JVM backend about twice as fast as the dynamic one (1.9 times at 10
observations, 2.6 at 100), and HMC and MALA steps 15 to 30% faster.
Gen.jl's tutorial found a static kernel made little difference in recent
versions of Julia; in gen.clj the gain is in assess and the gradients
built on it.

## Checking the inference program

Gen.jl ends by checking that inference still works. Its data follow the
model's own story: 50 points from a line of slope $-1$ and intercept 2,
with inlier noise 0.5 and outlier probability 0.1, and then two points
moved far above the line. Here the model itself generates them,
constrained to those parameters:

<!-- code: synthetic -->
```clojure
(def xs (mapv #(- (* 10.0 (/ % 49.0)) 5.0) (range 50)))

(def true-line
  (cm/choicemap :slope -1.0 :intercept 2.0 :noise 0.5 :prob-outlier 0.1))

(def ys
  (-> (:retval (:trace (gfi/generate loop-model [xs] true-line (rng/key 8))))
      (assoc 44 13.0 46 14.0)))
```

Then 500 sweeps of the static model:

<!-- code: fit -->
```clojure
(def sweeps
  (let [[k-init k-sweeps] (rng/split (rng/key 9))
        start (:trace (gfi/generate static-model-with-map [xs]
                                    (observations ys) k-init))]
    (reductions (fn [tr k] (gfi/eval-trace! (block-resimulation-update tr k)))
                start
                (rng/split k-sweeps 500))))

(def fitted (last sweeps))

[(fitted :slope) (fitted :intercept)]
;=> [-0.944 1.998] ; float64

[(fitted [:data 44 :is-outlier]) (fitted [:data 46 :is-outlier])]
;=> [true true] ; float64

(count (filter true? (map #(not= (%1 :slope) (%2 :slope))
                          sweeps (rest sweeps))))
;=> 7 ; float64
```

<!-- code: fit-figure -->
```clojure
;; figure: figures/029-fit.svg
(let [outlier? #(fitted [:data % :is-outlier])
      at (fn [i] [(xs i) (ys i)])
      line #(+ (* (fitted :slope) %) (fitted :intercept))]
  (svg/plot [{:curves [line] :color "black" :width 1}
             {:points (map at (remove outlier? (range 50)))}
             {:points (map at (filter outlier? (range 50)))
              :color "darkorange"}]
            {:x-range [-5.5 5.5] :y-range [-5 16]}))
```

![The data, with the line of the static model's trace after 500 sweeps.
The points the trace calls outliers are orange.](figures/029-fit.svg)

The chain's line ends near the line the data came from, and it calls both
planted points outliers. It got there slowly: the line moved in 7 of the
500 sweeps. A move proposes a whole line from the prior, and the data
reject nearly every one. Block resimulation is a slow algorithm, and
better moves for the line, a random walk or a gradient step, are the
business of *MCMC and MAP* and *Gradient-Based MCMC*. This chapter's
business is the cost of a sweep, which the static language made linear in
the number of points.

## Batched execution

The static language makes an update do less. Batched execution makes $N$
executions into one. `gen.batched` runs a model's code once for $N$
particles, under the *batch tracer* (`gen.array.batch`): every value has a
particle axis, every gen.array operation acts per particle, and every draw
is independent per particle. The handler's work at each site, its
bookkeeping, checks and choice-map writes, is done once for all $N$. The
transitions are the handler's own: *The Handler Loop* noted that GenMLX
keeps batched copies of its transitions, and that gen.clj has one set,
which the tracer runs.

GenMLX's chapter starts from a small model with an exact answer:
$\mu \sim \mathcal{N}(0, 2)$, and observations $y_i \mid \mu \sim
\mathcal{N}(\mu, 1)$. Twenty thousand particles with one observation are
one run:

<!-- code: batched-prior -->
```clojure
(def normal-mean
  (gen [n]
    (let [mu (trace :mu (d/normal 0 2))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def prior-particles (batched/simulate normal-mean [1] 20000 (rng/key 10)))

[(batched/batched? prior-particles)
 (array/shape (batched/choice prior-particles :mu))
 (array/shape (batched/scores prior-particles))]
;=> [true [20000] [20000]]
```

Each choice is an array with one value per particle, and so is the score:
20000 log densities. The particles are a sample from the prior. The mean
of $\mu$ is within five standard errors of 0, its variance within five
standard errors of 4, and the variance of $y$ within five of $4 + 1 = 5$:

<!-- code: prior-moments -->
```clojure
(defn moments
  "The mean and the variance of the host numbers vs."
  [vs]
  (let [n (count vs)
        mean (/ (reduce + vs) n)]
    [mean (/ (reduce + (map #(* (- % mean) (- % mean)) vs)) (dec n))]))

(let [n 20000
      [m-mu v-mu] (moments (array/->clj (batched/choice prior-particles :mu)))
      [_ v-y] (moments (array/->clj (batched/choice prior-particles [:y 0])))
      ;; five standard errors of a mean, and of a normal sample's variance
      band-mean (fn [v] (* 5 (Math/sqrt (/ v n))))
      band-var (fn [v] (* 5 v (Math/sqrt (/ 2.0 (dec n)))))]
  [(< (abs m-mu) (band-mean 4.0))
   (< (abs (- v-mu 4.0)) (band-var 4.0))
   (< (abs (- v-y 5.0)) (band-var 5.0))])
;=> [true true true]
```

Each particle is a trace. `batched/traces` rebuilds them by the scalar
handler from each particle's choices (here one particle's, taken with
`batched/take`), and a rebuilt trace holds its particle's values and
score:

<!-- code: particle-is-trace -->
```clojure
(let [t (first (batched/traces (batched/take prior-particles [7])))
      mus (batched/choice prior-particles :mu)
      scores (batched/scores prior-particles)]
  [(array/allclose? (t :mu) (array/take mus 7))
   (array/allclose? (:score t) (array/take scores 7))])
;=> [true true]
```

`{:batched? true}` in Part I ran importance sampling this way.
`docs/benchmarks.md` measures batched importance sampling on a linear
regression 5 to 90 times faster than one particle at a time on the JVM
backend, from 100 particles to 10000. GenMLX's chapter, where each scalar
run is its own trip to the GPU, measures a far larger factor.

## The same distribution, not the same draws

A batched run draws its values from random arrays with a particle axis, so
for one key its draws differ from those of $N$ scalar runs. What it
promises is the distribution. With five observations, the posterior of
$\mu$ and the log marginal likelihood have closed forms:

<!-- code: exact -->
```clojure
(def ys5 [1.2 0.4 2.1 1.7 0.9])

(def observed
  (reduce (fn [c [i y]] (cm/set-choice c [:y i] y))
          cm/EMPTY
          (map-indexed vector ys5)))

(defn exact
  "normal-mean's posterior of mu given the observations ys, normal with
   this mean and sd, and its log marginal likelihood: the ys are normal
   with mean 0 and covariance I + 4J, J a matrix of ones."
  [ys]
  (let [n (count ys)
        s (reduce + ys)
        ss (reduce + (map #(* % %) ys))
        precision (+ 0.25 n)]
    {:mean (/ s precision)
     :sd (Math/sqrt (/ 1.0 precision))
     :log-ml (- (* -0.5 n (Math/log (* 2.0 Math/PI)))
                (* 0.5 (Math/log (+ 1.0 (* 4.0 n))))
                (* 0.5 (- ss (/ (* 4.0 s s) (+ 1.0 (* 4.0 n))))))}))

(def posterior (exact ys5))

posterior
;=> {:mean 1.2 :sd 0.4364 :log-ml -7.192}
```

Importance sampling estimates both, one sample at a time and batched, with
the same key. Both estimates are within five standard errors of the exact
values, and the two differ:

<!-- code: batched-is -->
```clojure
(defn agrees?
  "Are importance sampling's estimates of mu's posterior mean and of the
   log marginal likelihood, from n samples, within five standard errors of
   the exact ones?"
  [{:keys [estimate log-ml-estimate ess]} n]
  (let [{:keys [mean sd log-ml]} posterior]
    [(< (abs (- estimate mean)) (* 5 (/ sd (Math/sqrt ess))))
     (< (abs (- log-ml-estimate log-ml))
        (* 5 (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n)))))]))

(defn sampled
  "Importance sampling's estimate of mu's posterior mean from 10000
   samples, with opts beside the key."
  [opts]
  (imp/estimate #(% :mu) normal-mean [5] observed 10000
                (merge {:key (rng/key 11)} opts)))

(def one-at-a-time (sampled {}))

(def all-at-once (sampled {:batched? true}))

[(agrees? one-at-a-time 10000) (agrees? all-at-once 10000)]
;=> [[true true] [true true]]

(== (:estimate one-at-a-time) (:estimate all-at-once))
;=> false
```

The standard error of the posterior mean is the posterior sd over
$\sqrt{\text{ESS}}$, and that of the log marginal likelihood is
$\sqrt{1/\text{ESS} - 1/n}$, as in *Composition*.

Where nothing is drawn, a batched run gives the scalar values, up to
rounding. `batched/generate` takes per-particle constraints, `{:per-particle
cs}`: one choice map for each particle, beside the shared constraints.
With every choice given, each particle's weight is its scalar weight, the
log joint density. Here five values of $\mu$ are scored in one run:

<!-- code: per-particle -->
```clojure
(def grid [0.0 0.5 1.0 1.5 2.0])

(def scored
  (batched/generate normal-mean [5] observed 5 (rng/key 12)
                    {:per-particle (mapv #(cm/choicemap :mu %) grid)}))

(defn joint
  "The log density of normal-mean's observations with mu, by the scalar
   handler."
  [mu]
  (:weight (gfi/generate normal-mean [5]
                         (cm/merge observed (cm/choicemap :mu mu))
                         (rng/key 0))))

(array/allclose? (:weights scored) (array/array (mapv joint grid)))
;=> true
```

Batched enumeration scores a grid of choices this way, one grid point per
particle (*Debugging with Enumeration*).

## When a model cannot batch

The robust regression's kernel branches on its flag, and the flag differs
across particles. Clojure's `if` cannot take both branches, and the `gen`
macro guards every `if` in a body (*The Handler Loop*). A branch on a
batched value raises an error that the batched run catches, and the
collection falls back to one scalar trace per particle. The result is
correct, only not batched, and `batched/fallback-reason` says why:

<!-- code: fallback -->
```clojure
(def flag-particles (batched/simulate model-with-map [xs5] 100 (rng/key 13)))

[(batched/batched? flag-particles)
 (:message (batched/fallback-reason flag-particles))]
;=> [false "Cannot branch on a batched value: it differs across particles"]
```

A choice between two values can be one array operation instead:
`array/where` takes, for each particle, one value or the other. With it
the kernel runs batched, and it is the same model, with the same density:

<!-- code: where-point -->
```clojure
(def where-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))
          sd (array/where outlier? 10.0 noise)]
      (trace :y (d/normal mu sd)))))

(def where-model (static-with-map where-point))

(def where-particles (batched/simulate where-model [xs5] 100 (rng/key 13)))

(batched/batched? where-particles)
;=> true

(array/allclose? (:score five)
                 (:weight (gfi/assess where-model [xs5] (:choices five)
                                      (rng/key 3))))
;=> true
```

Every one of its particles agrees with the scalar handler:

<!-- code: wrong -->
```clojure
(defn wrong
  "The number of particles of ps whose batched score is not the scalar
   handler's score of their own choices."
  [ps]
  (count (remove true? (map array/allclose?
                            (map :score (batched/traces ps))
                            (array/->clj (batched/scores ps))))))

(wrong where-particles)
;=> 0
```

The guards cover a `gen` body, not the helper functions it calls. Clojure
has no hook for its own `if`, and a batched value is true to it, so a raw
`if` in a helper takes one branch for every particle, silently. Two
further layers catch this. A batched comparison or logical operation that
nothing reads by the end of the run, which is what a raw `if` on it
leaves, fails the run:

<!-- code: unread -->
```clojure
(defn inlier-spread
  "The standard deviation of a point, given whether it is an inlier: a
   branch outside gen, unguarded."
  [inlier? noise]
  (if inlier? noise 10.0))

(def unread-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))
          sd (inlier-spread (array/logical-not outlier?) noise)]
      (trace :y (d/normal mu sd)))))

(let [ps (batched/simulate (static-with-map unread-point) [xs5] 100
                           (rng/key 13))]
  [(batched/batched? ps)
   (str/starts-with? (:message (batched/fallback-reason ps))
                     "A batched comparison was never read")])
;=> [false true]
```

A raw `if` on a choice itself leaves nothing unread. For that, the run is
checked against the ground truth. After a batched simulate or generate,
and on a schedule of updates, `batched/*spot-checks*` evenly spaced
particles are run again through the scalar handler from their choices, and
a different score means the fallback:

<!-- code: helper -->
```clojure
(defn spread
  "The standard deviation of a point: a branch outside gen, unguarded."
  [outlier? noise]
  (if outlier? 10.0 noise))

(def helper-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))]
      (trace :y (d/normal mu (spread outlier? noise))))))

(def helper-model (static-with-map helper-point))

(def helper-particles (batched/simulate helper-model [xs5] 100 (rng/key 13)))

batched/*spot-checks*
;=> 8

[(batched/batched? helper-particles)
 (str/starts-with? (:message (batched/fallback-reason helper-particles))
                   "The batched run disagrees with the scalar handler")]
;=> [false true]
```

Without the spot checks the run would have stayed batched, with most of
its particles' scores wrong:

<!-- code: unchecked -->
```clojure
(def unchecked
  (binding [batched/*spot-checks* 0]
    (batched/simulate helper-model [xs5] 100 (rng/key 13))))

[(batched/batched? unchecked) (wrong unchecked)]
;=> [true 81] ; float64
```

Spot checks are a sample, so they catch a bug in proportion to the
particles it touches (exercise 4). Two remedies are deterministic.
`array/where` keeps a helper batched, as above. `dyn/defn-guarded`
defines a helper with a `gen` body's guards, so that its branch on a
batched value raises the error and the run falls back every time, here
with the spot checks turned off:

<!-- code: guarded -->
```clojure
(dyn/defn-guarded guarded-spread
  "spread, with a gen body's guards."
  [outlier? noise]
  (if outlier? 10.0 noise))

(def guarded-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))]
      (trace :y (d/normal mu (guarded-spread outlier? noise))))))

(let [ps (binding [batched/*spot-checks* 0]
           (batched/simulate (static-with-map guarded-point) [xs5] 100
                             (rng/key 13)))]
  [(batched/batched? ps) (:message (batched/fallback-reason ps))])
;=> [false "Cannot branch on a batched value: it differs across particles"]
```

## Many chains at once

MCMC batches across chains. `mh/mh-particles` makes one selection-MH move
on each particle of a collection, so the particles are independent chains,
each accepted or rejected on its own, and the moves of all of them are one
batched regenerate. The accepted proposals are merged into the old
particles with `batched/select`, which is `array/where` over the
particles: the GenMLX book's merge by mask.

One test of a move is that it leaves the posterior unchanged. Start 2000
chains at exact posterior draws, given as per-particle constraints, and
move each chain 20 times:

<!-- code: many-chains -->
```clojure
(def starts
  (let [{:keys [mean sd]} posterior]
    (mapv #(+ mean (* sd %)) (array/->clj (rng/normal (rng/key 14) [2000])))))

(def chains
  (:particles (batched/generate normal-mean [5] observed 2000 (rng/key 15)
                                {:per-particle (mapv #(cm/choicemap :mu %)
                                                     starts)})))

(def moved
  (reduce (fn [ps k] (:particles (mh/mh-particles ps (sel/select :mu) k)))
          chains
          (rng/split (rng/key 16) 20)))

(batched/batched? moved)
;=> true
```

The chains are still distributed as the posterior. The mean of $\mu$ is
within five standard errors of the exact mean, and its sd within five
standard errors of the exact sd:

<!-- code: stationary -->
```clojure
(defn posterior-spread?
  "Are the particles' values of mu distributed as the posterior: their
   mean and sd within five standard errors of the exact ones?"
  [ps]
  (let [k (batched/size ps)
        {:keys [mean sd]} posterior
        [m v] (moments (array/->clj (batched/choice ps :mu)))]
    [(< (abs (- m mean)) (* 5 (/ sd (Math/sqrt k))))
     (< (abs (- (Math/sqrt v) sd)) (* 5 (/ sd (Math/sqrt (* 2 (dec k))))))]))

(posterior-spread? moved)
;=> [true true]
```

<!-- code: chains-figure -->
```clojure
;; figure: figures/029-chains.svg
(let [{:keys [mean sd]} posterior]
  (svg/histogram [{:values (array/->clj (batched/choice moved :mu))
                   :density #(array/exp (dc/log-prob (d/normal mean sd) %))}]
                 {:x-label "mu"}))
```

![$\mu$ across 2000 chains after 20 batched MH moves each, against the
exact posterior density.](figures/029-chains.svg)

The test has power. Moves that resimulate $\mu$ from the prior and accept
every proposal are not MH, and they leave the chains distributed as the
prior, which the test rejects:

<!-- code: blind -->
```clojure
(def blind
  (reduce (fn [ps k]
            (:particles (batched/regenerate ps [5] [diff/no-change]
                                            (sel/select :mu) k)))
          chains
          (rng/split (rng/key 16) 20)))

(posterior-spread? blind)
;=> [false false]
```

Other algorithms have batched forms too: `pf/initialize` and
`vi/black-box-vi` take `{:batched? true}`, and HMC, MALA and MAP have
`mcmc/hmc-particles`, `mala-particles` and `map-optimize-particles`.

## Compiling on MLX

The third path is the backend's. On MLX an array is a node of a lazy
graph, and each operation is a call into MLX through the native binding,
so a step of a few hundred operations makes a few hundred calls.
`array/compile` traces a function once for each structure, shape and dtype
of its arguments, into one MLX graph, which MLX fuses into fewer kernels
and replays on later calls. The function must be pure in its arguments'
values. It may not read a value on the host (`truthy?`, `item`), since the
trace is not run again, and a key may not be an argument, since the
replayed trace would replay its draws. On the JVM backend, compile calls
the function, so the same code runs on both:

<!-- code: compile -->
```clojure
(defn log-joint
  "The log density of normal-mean's observations with mu, as a function of
   mu alone, which compile takes."
  [mu]
  (:weight (gfi/assess normal-mean [5] (cm/merge observed (cm/choicemap :mu mu))
                       (rng/key 0))))

(def compiled-log-joint (array/compile log-joint))

(array/allclose? (compiled-log-joint 1.0) (log-joint 1.0))
;=> true
```

<!-- code: compile-key -->
```clojure
;; error: compile: the arguments must be arrays
((array/compile (fn [k] (rng/normal k []))) (rng/key 17))
```

HMC and MALA take `{:compile? true}`. HMC's L leapfrog steps then run as
one compiled function of the selected values and the other choices, and
MALA's proposal and densities likewise. On the JVM backend the compiled
kernel makes the plain kernel's move:

<!-- code: compiled-hmc -->
```clojure
(def start (:trace (gfi/generate normal-mean [5] observed (rng/key 18))))

(defn hmc-move
  "One HMC move of start's mu, L = 10 steps of 0.1, with opts."
  [opts]
  (mcmc/hmc start (sel/select :mu) (merge {:L 10 :eps 0.1} opts) (rng/key 19)))

(def plain (hmc-move {}))

(def fused (hmc-move {:compile? true}))

(:accepted? plain)
;=> true ; float64

[(= (:accepted? plain) (:accepted? fused))
 (array/allclose? ((:trace plain) :mu) ((:trace fused) :mu))]
;=> [true true]
```

`docs/benchmarks.md` measures a compiled HMC step 4 to 8 times faster than
a plain one on Metal, and MALA 1.2 to 2.8 times. On the JVM backend the
compiled kernel gains less, and only because it does less work: it
differentiates the model's assess as a function of the values, where the
plain kernel updates the trace at each leapfrog step. A static model's
score function makes that assess straight-line code, which compile fuses
whole. Steps that draw inside the model, such as MH's regenerate and a
particle filter's extension, are not compiled: their keys are host values,
and a compiled trace would replay their draws. GenMLX also compiles whole
static models into fused kernels; gen.clj does not.

GenMLX's chapter ends with memory: a long loop on Metal must free its
arrays by hand (`mx/tidy`) or meet Metal's limit of about 500,000 buffers.
gen.clj's array API has no memory management. On MLX an array is freed
once it is unreachable, and the backend collects when the live arrays or
the bytes made since the last collection pass a threshold
(`docs/array-api.md`, §13). An inference loop's part is to evaluate at its
boundary, as `block-resimulation-inference` does with `gfi/eval-trace!`,
and to keep host numbers rather than traces where it keeps many results.

## Choosing a path

The three paths save different work, and each is held to the handler in
its own way:

| Path | What it saves | What it gives | What holds it to the handler |
|------------------|----------------------|----------------------|----------------------|
| The static language | reruns of the nodes a change does not reach | the same choices under the same keys | the static law |
| Batched execution | the handler's work at each site, $N - 1$ times | the same distribution, and the scalar values where nothing is drawn | the guards, unread comparisons and spot checks |
| Compiling on MLX | MLX's calls, one per operation | the same values | the same code on the JVM backend |

The three combine. A static model runs batched, and its score function
compiles. A model whose structure varies, whose branches differ across
particles, still runs, one particle at a time, and gives the same answers.
Each path plugs in below the GFI, and the inference code above it never
learns which one ran. Part III, *Theory*, turns from how fast the
operations run to why their weights are right: the interface as the
thesis defines it, and the laws that hold each of these paths to it.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Split the line block.** Move the line's three parameters one at a
   time, in three blocks, instead of together. Predict how many points a
   sweep on 10 points visits for the loop model and for the static model,
   then count.
2. **Which moves rerun the Map?** In the static model, predict which
   points each kind of move revisits: the line's, a flag's and the
   outlier probability's. Then update point 2's $y$ with `gfi/update`.
   Explain each from the graph.
3. **Batch the loop model.** Does `loop-model` run batched? Predict, then
   write its branch with `array/where` and check that the new model runs
   batched and has the same density.
4. **A rare branch.** Give `helper-model` an outlier probability of
   0.998, so that a particle with an inlier is rare, and run 100 of its
   particles batched. Does it fall back? How many particles are wrong?
   Then bind `batched/*spot-checks*` to `:all`.

## Solutions

**1.** The loop model makes $n + 4$ moves, each visiting every point:
$10 \cdot 14 = 140$. The static model's three line moves visit every
point, its flags' moves one each, and the outlier probability's every
point: $3n + n + n = 50$.

<!-- code: solution-1 -->
```clojure
(defn split-update
  "block-resimulation-update with the line's three parameters moved one at
   a time."
  [tr key]
  (let [[xs] (:args tr)]
    (sweep tr
           (concat (map sel/select [:noise :slope :intercept])
                   (for [i (range (count xs))]
                     (sel/select [:data i :is-outlier]))
                   [(sel/select :prob-outlier)])
           key)))

[(sweep-visits (fn [v] (counted loop-model v)) 10 (rng/key 1) split-update)
 (sweep-visits (fn [v] (static-with-map (counted point v))) 10 (rng/key 1)
               split-update)]
;=> [140 50]
```

**2.** The line's parameters and the outlier probability are inputs of
every point, through the Map's arguments, so their moves revisit them
all. A flag is an input of no other point. A constraint on point 2's $y$
reaches only element 2, and no argument changes, so the update revisits
point 2 alone:

<!-- code: solution-2 -->
```clojure
(let [tr (five-points static-model-with-map)
      after #(:trace (gfi/regenerate tr % (rng/key 5)))]
  (mapv #(revisited tr (after %))
        [(sel/select :noise :slope :intercept)
         (sel/select [:data 2 :is-outlier])
         (sel/select :prob-outlier)]))
;=> [[0 1 2 3 4] [2] [0 1 2 3 4]]

(let [tr (five-points static-model-with-map)]
  (revisited tr (:trace (gfi/update tr (cm/choicemap [:data 2 :y] 3.5)
                                    (rng/key 5)))))
;=> [2]
```

**3.** No: its `if` on a flag differs across particles, so it falls back.
With `array/where` it runs batched, and it weighs the loop model's
choices as the loop model does:

<!-- code: solution-3 -->
```clojure
(def where-loop-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (vec (map-indexed
            (fn [i x]
              (let [outlier? (trace [:data i :is-outlier]
                                    (d/bernoulli prob-outlier))
                    mu (array/where outlier? 0.0 (+ (* x slope) intercept))
                    sd (array/where outlier? 10.0 noise)]
                (trace [:data i :y] (d/normal mu sd))))
            xs)))))

(let [batched? #(batched/batched? (batched/simulate % [xs5] 100 (rng/key 20)))]
  [(batched? loop-model)
   (batched? where-loop-model)
   (array/allclose? (:score five)
                    (:weight (gfi/assess where-loop-model [xs5] (:choices five)
                                         (rng/key 3))))])
;=> [false true true]
```

**4.** With 5 points and an outlier probability of 0.998, a particle has
an inlier with probability $1 - 0.998^5 \approx 1\%$. Here one particle
of the 100 has an inlier, and the 8 evenly spaced checks miss it, so the
run stays batched with that particle's score wrong. Checking every
particle finds it. A raw `if` on a choice is safe only in a `gen` body or
a guarded helper; otherwise write it with `array/where`.

<!-- code: solution-4 -->
```clojure
(defn rare
  "helper-model's 100 particles with the outlier probability given as
   0.998, under checks spot checks."
  [checks]
  (binding [batched/*spot-checks* checks]
    (:particles (batched/generate helper-model [xs5]
                                  (cm/choicemap :prob-outlier 0.998)
                                  100 (rng/key 20)))))

(let [sampled (rare 8)
      every (rare :all)]
  [(batched/batched? sampled) (wrong sampled) (batched/batched? every)])
;=> [true 1 false] ; float64
```
