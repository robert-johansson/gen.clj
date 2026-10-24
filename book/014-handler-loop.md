# The Handler Loop

*Getting Started* said that `(trace :slope (d/normal 0 2))` does not say
whether the value is drawn fresh or fixed to one we observed: whatever
runs the model decides. This chapter opens that thing, the *handler*. It
is small enough to read in one sitting: one pure function per operation,
threaded through a run by a loop that keeps the run's state in a mutable
cell. Its smallness is why the rest of the book composes as it does:
nested models, batched execution and faster paths all plug into the same
slot.

This chapter ports *How It Works: The Handler Loop* from *The GenMLX
Book*. gen.clj's handler follows GenMLX's design and differs from it in a
few places, noted where they come up. The chapter's code lives in one
namespace:

<!-- code: ns -->
```clojure
(ns book.handler-loop
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.trace :as trace]))
```

## A random choice is an effect

In programming-language terms, `trace` is an *effect*: an operation
whose meaning is not fixed where it is called. The body raises it ("a
value from this distribution, at this address"), and a handler, chosen
from outside the body, decides what raising it does. Simulate's handler
draws the value. Generate's takes it from the constraints when they hold
one. Assess's requires it there. The body cannot tell which one runs it.

| Functional programming | Probabilistic programming |
|------------------------|---------------------------|
| pure function | generative function (`gen`) |
| effect | random choice (`trace`), call (`splice`) |
| handler | GFI operation (`simulate`, `generate`, …) |
| immutable value | trace |

One body, three operations:

<!-- code: model -->
```clojure
(def two-normals
  (gen []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal x 1))]
      (+ x y))))

(def tr (gfi/simulate two-normals [] (rng/key 1)))

(tr :x)
;=> 0.716 ; float64

(let [obs (cm/choicemap :x 3.0)]
  ((:trace (gfi/generate two-normals [] obs (rng/key 1))) :x))
;=> 3.0

(:weight (gfi/assess two-normals [] (cm/choicemap :x 3.0 :y 2.0) (rng/key 1)))
;=> -6.8379
```

Simulate drew `:x`, generate took it from the constraints, and assess
scored the values given: $\log \mathcal{N}(3; 0, 1) + \log \mathcal{N}(2;
3, 1) = -5.4189 - 1.4189$.

## The handler is a pure function

Every operation that runs a body is built from one shape of function, a
*transition*:

```
(state, path, dist, key) -> [value state']
```

It takes the handler's *state*, an immutable map, with the site's path,
distribution and key, and returns the value the body gets, with a new
state. The standard transitions, one per operation, are in the public map
`dyn/site-transitions`, so that middleware can wrap them (below). We can
call one outside any model, on a state we build by hand:

<!-- code: transition -->
```clojure
(def simulate-site (dyn/site-transitions :simulate))

(def empty-state {:choices cm/EMPTY :scores {} :score 0.0})

(def step (simulate-site empty-state [:x] (d/normal 0 1) (rng/key 2)))

(let [[x state'] step]
  [x (:scores state') (:score state')])
;=> [-0.192 {[:x] -0.937} -0.937] ; float64

(let [[x state'] step
      [x-again _] (simulate-site empty-state [:x] (d/normal 0 1) (rng/key 2))]
  [(array/equal? x (cm/get-choice (:choices state') :x))
   (array/allclose? (:score state') (dc/log-prob (d/normal 0 1) x))
   (cm/empty? (:choices empty-state))
   (array/equal? x x-again)])
;=> [true true true true]
```

Simulate's transition draws a value with the key it is given, scores it
with the distribution's log density, and records both in a new state:
the choice at its path, its log density under `:scores`, and the sum so
far under `:score`. The state we passed in still holds no choices, since
`assoc` and `update` return new maps over shared structure, and the same
arguments give the same value. That is the whole meaning of *pure* here,
and as the checks show, it can be tested with no setup and no mocks.

Two things differ from GenMLX's transition. gen.clj's takes its key as
an argument, where GenMLX's splits one it finds in the state: here the
loop around the transitions hands each site its key. And gen.clj's
records each choice's log density, as Gen.jl's traces do, so update and
regenerate weigh against the recorded values and `project` reads a trace
without running its body again.

## Generate is simulate plus one `if`

Generate's transition differs from simulate's by one branch. If the
constraints hold a value at the path, it takes that value, scores it,
and adds the log density to both the score and the weight. Otherwise
it calls simulate's transition:

<!-- code: generate-transition -->
```clojure
(def generate-site (dyn/site-transitions :generate))

(def observing-x
  (assoc empty-state :mode :generate :weight 0.0
         :constraints (cm/choicemap :x 3.0)))

(let [[x state'] (generate-site observing-x [:x] (d/normal 0 1) (rng/key 2))]
  [x (:weight state')])
;=> [3.0 -5.4189]

(let [dist (d/normal 0 1)
      [v state'] (generate-site observing-x [:y] dist (rng/key 2))
      [v' state''] (simulate-site observing-x [:y] dist (rng/key 2))]
  [(array/equal? v v') (array/equal? (:score state') (:score state''))
   (:weight state')])
;=> [true true 0.0]
```

The constrained site returns the 3.0 we gave, with the weight $\log
\mathcal{N}(3; 0, 1)$, the density of the one observation. At `:y`,
where the constraints hold nothing, the result is simulate's, and the
weight stays 0. This is the weight of *Conditioning*, accumulating one
site at a time. The other transitions are a few branches each over the
same state, differing in which values they take and what they add to the
weight:

| Transition | At a site, it … |
|-------------|-------------------------------------------------------|
| `:simulate` | draws the value |
| `:generate` | takes a constrained value, adding its log density to the weight; else draws |
| `:assess` | takes the given value, adding its log density to the weight |
| `:update` | takes a constrained value, else keeps the old one, adding the change in log density; a new site draws |
| `:regenerate` | draws a selected or a new site; keeps an unselected one, adding the change in log density |

Propose is a simulate, and project sums the log densities a trace
recorded. So five small transitions hold every per-site rule of the GFI;
*Update and Regenerate* takes up the last two. Where a rule has nothing
to work with, it says so. Assess given too few choices is an error, not
a silent draw:

<!-- code: assess-error -->
```clojure
;; error: assess: no value for the choice at [:y]
(gfi/assess two-normals [] (cm/choicemap :x 3.0) (rng/key 3))
```

## The state is a map

To look at the states the real handler threads, we wrap a transition in
*middleware*: a map from operation to transition, which a model given it
by `dyn/with-transitions` uses in place of the standard one. This one
keeps each state it receives in an atom of ours:

<!-- code: watched -->
```clojure
(defn watched
  "model with the transition of op wrapped to keep, in the atom seen, each
   state it receives. The atom is ours, for looking; the transition it
   wraps is the handler's own."
  [model op seen]
  (let [transition (dyn/site-transitions op)]
    (dyn/with-transitions model
      {op (fn [state p dist k]
            (swap! seen conj state)
            (transition state p dist k))})))

(def seen (atom []))

(def watched-run
  (gfi/generate (watched two-normals :generate seen) []
                (cm/choicemap :x 3.0) (rng/key 1)))

(mapv (juxt :mode :score :weight) @seen)
;=> [[:generate 0.0 0.0] [:generate -5.4189 -5.4189]]
```

The second site received the score and the weight of the constrained
`:x`: each state carries on where the last left off. Every operation
starts from a map with the same keys, where GenMLX gives each operation
its own. `:mode` names the operation, `:key` is the key the next site
splits, `:choices`, `:scores` and `:score` hold the choices so far, their
log densities and their sum, and `:weight` the weight so far. The rest
are for some operations only: the constraints (generate, update,
assess), the previous trace (update, regenerate), the selection
(regenerate), the replaced choices (update), the traces of the calls,
and the paths used so far, for the address checks:

<!-- code: state-keys -->
```clojure
(every? (set (keys (first @seen)))
        [:mode :key :choices :scores :score :weight :constraints
         :prev :selection :discard :subtraces :visited])
;=> true
```

## The score is a fold

Nothing in the transitions arranges this threading. It is function
composition, and it is why the score of a model with several sites is
the sum of their log densities. Threading a state through a list of
sites is a `reduce`, one key per site. With `reductions` in its place we
see every state on the way, and the last one's score is the sum,
computed again from its choices:

<!-- code: fold -->
```clojure
(defn site-states
  "Every state of threading state through transition at each [path dist]
   of sites, with one key each from key's split: state itself first."
  [transition state sites key]
  (reductions (fn [s [[p dist] k]] (second (transition s p dist k)))
              state
              (map vector sites (rng/split key (count sites)))))

(def sites [[[:slope] (d/normal 0 10)] [[:noise] (d/normal 0 1)]])

(def states (site-states simulate-site empty-state sites (rng/key 4)))

(mapv :score states)
;=> [0.0 -3.626 -4.728] ; float64

(defn log-density
  "The sum of the log densities of state's choices at sites."
  [state sites]
  (reduce + (map (fn [[p dist]]
                   (dc/log-prob dist (cm/get-choice (:choices state) p)))
                 sites)))

(array/allclose? (:score (last states)) (log-density (last states) sites))
;=> true
```

This is *Getting Started*'s score, derived rather than asserted. The
joint log density is a left fold of `+` over the sites' log densities,
and the handler is the fold.

## One run's mutable cell

A body does not thread a state. It calls `(trace :x dist)` and expects a
value back, like any Clojure code. Something has to turn the pure
transitions into a `trace` that seems to change the world. In gen.clj
that is the loop that runs a body. It keeps the state in a `volatile!`,
Clojure's light mutable cell for one thread, and gives the body a
`trace` that reads the state, calls the transition, writes the new state
back and returns the value. Here is that loop, without its checks:

<!-- code: loop -->
```clojure
(defn run-body
  "Run body, a function of trace, under transition from state, as gen.clj's
   handler loop does without its checks: at each site, split the state's
   key, give the site one key and carry the other. The final state, with
   the body's value at :retval."
  [transition state body]
  (let [vol (volatile! state)
        trace (fn [addr dist]
                (let [[k-next k-site] (rng/split (:key @vol))
                      [v state'] (transition (assoc @vol :key k-next)
                                             (cm/path addr) dist k-site)]
                  (vreset! vol state')
                  v))
        retval (body trace)]
    (assoc @vol :retval retval)))
```

It is the real mechanism, less the checks. Driving `two-normals`' body,
written as a function of `trace`, through it gives the trace that
simulate gives, exactly, and the same for generate:

<!-- code: by-hand -->
```clojure
(defn two-normals-body
  "two-normals' body, as a function of trace."
  [trace]
  (let [x (trace :x (d/normal 0 1))
        y (trace :y (d/normal x 1))]
    (+ x y)))

(defn same-run?
  "Do a and b (traces or handler states) hold the same choices, score and
   return value, exactly?"
  [a b]
  (let [ca (:choices a) cb (:choices b)]
    (and (= (cm/addresses ca) (cm/addresses cb))
         (every? true? (map array/equal? (cm/values ca) (cm/values cb)))
         (array/equal? (:score a) (:score b))
         (array/equal? (:retval a) (:retval b)))))

(let [k (rng/key 1)
      start (assoc empty-state :key k)
      obs (cm/choicemap :x 3.0)
      observing (assoc start :mode :generate :weight 0.0 :constraints obs)]
  [(same-run? (run-body simulate-site start two-normals-body)
              (gfi/simulate two-normals [] k))
   (same-run? (run-body generate-site observing two-normals-body)
              (:trace (gfi/generate two-normals [] obs k)))])
;=> [true true]
```

The real loop also checks each site's address and distribution, and the
constraints at the end of the run, with loud errors rather than silent
defaults. A path may be used once in a run, and generate, update and
assess must visit every constraint, since a misspelled address would
otherwise be ignored:

<!-- code: twice-error -->
```clojure
;; error: Address [:x] is used twice
(gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace :x (d/normal 0 1)))
              [] (rng/key 1))
```

<!-- code: unvisited-error -->
```clojure
;; error: generate: the constraint at [:z] was not visited
(gfi/generate two-normals [] (cm/choicemap :z 1.0) (rng/key 1))
```

Two things differ from GenMLX. First, the loop splits the key, not the
transition, so every site takes one key from the run's sequence whether
it draws or not. Constraining a choice then never shifts another site's
draws: generate with a simulated trace's `:x`, and the same key,
reproduces its `:y`. GenMLX's generate transition takes no key at a
constrained site, so there a constraint changes every later draw;
exercise 2 builds that loop and watches it happen.

<!-- code: one-key-per-site -->
```clojure
(let [k (rng/key 5)
      t (gfi/simulate two-normals [] k)
      again (:trace (gfi/generate two-normals [] (cm/choicemap :x (t :x)) k))]
  (array/equal? (t :y) (again :y)))
;=> true
```

Second, gen.clj's loop keeps two cells: the state, and the transition
in effect. A splice without an address, Gen.jl's `{*} ~`, runs its
callee's body under the caller's loop, and the second cell swaps in the
callee's middleware for that body. Both cells are made when a run starts
and dropped when it returns. They never escape into a trace and never
cross a thread, since a body runs synchronously, start to finish.

## The `gen` macro

The body gets its `trace` from the `gen` macro, which rewrites the body
into a function of four hidden parameters (`trace`, `splice`, `param`
and `untraced`) before the arguments. The loop passes them in. The
expansion is short enough to check whole:

<!-- code: expansion -->
```clojure
(= (macroexpand-1 '(gen [b] (if b 1 2)))
   '(gen.dynamic/dynamic-gf
     (fn* ([trace splice param untraced b]
           (let* [untraced-key (fn* ([] (untraced)))]
             (if (gen.array/truthy? b) 1 2))))
     (quote ([b] (if b 1 2)))
     nil))
;=> true
```

It holds the body function, the quoted source (for printing and for the
check of the number of arguments), and the graph of a static body (none
here; see below). Because `trace` is a local, it works wherever a
function works: in `doseq`, in `mapv`, in a closure, in a recursive
helper. The handler does not care how control reached a `trace` call, so
models are ordinary Clojure, not a restricted sublanguage. GenMLX's
macro reads `trace` from a runtime object instead, to the same effect.

The expansion also shows the one change the macro makes to what a body
means: every `if` test is wrapped in `array/truthy?`. A comparison of
arrays returns an array, and Clojure's `if` counts every array as true.
On the JVM backend a single number is a host value, which `if` reads
correctly, but a one-element array is an array, as is every MLX value:

<!-- code: guard -->
```clojure
(def one-false (array/array [false]))

((fn [b] (if b :then :else)) one-false)
;=> :then

(:retval (gfi/simulate (gen [b] (if b :then :else)) [one-false] (rng/key 1)))
;=> :else
```

The plain function took the wrong branch. The rewrite reaches `when`,
`cond`, `and` and `or`, which expand to `if`, and puts guarded versions
in place of clojure.core's truthiness tests and lookups (`not`, `filter`,
`get`, …). Under the batch tracer, a test of a value that differs across
particles is a loud error, so a batched run falls back to one particle
at a time rather than take one branch for all. Helper functions the body
calls are not rewritten; exercise 4 is about them.

## The handler recurses

`splice` calls another generative function at an address. The handler
takes one key from its sequence and runs the callee's own GFI operation
with it, so a dynamic callee runs its own loop, with its own cells. Then
it folds the result into the caller's state: the callee's trace at the
address, its score added, its choices nested under the address. The
score is the joint, $\log p(\theta) + \log p(\text{obs} \mid \theta)$:

<!-- code: splice -->
```clojure
(def measurement
  (gen [mu]
    (trace :obs (d/normal mu 1))))

(def two-level
  (gen []
    (let [theta (trace :theta (d/normal 0 2))
          obs (splice :m measurement theta)]
      {:theta theta :obs obs})))

(def nested (gfi/simulate two-level [] (rng/key 6)))

(cm/addresses (:choices nested))
;=> [[:m :obs] [:theta]]

(array/allclose? (:score nested)
                 (+ (dc/log-prob (d/normal 0 2) (nested :theta))
                    (dc/log-prob (d/normal (nested :theta) 1)
                                 (nested [:m :obs]))))
;=> true
```

The caller keeps the callee's trace. It is the trace that simulating
`measurement` on its own gives, with the key the call took, the second of
the caller's sequence:

<!-- code: subtrace -->
```clojure
(let [[k-next _] (rng/split (rng/key 6))
      [_ k-call] (rng/split k-next)]
  (same-run? (trace/subtrace nested :m)
             (gfi/simulate measurement [(nested :theta)] k-call)))
;=> true
```

A generative function built from generative functions is again one,
with the same operations and the same accounting of weights: the GFI is
closed under composition. Nothing in the transitions had to be told
this, because a call is more state threading. The callee can be any
generative function, since the handler only calls its GFI; *Composition*
builds the combinators on the same footing.

## What purity buys

Composition is the first of three things one transition signature buys.

**Batched execution.** The transitions touch values only through
gen.array's operations. Under the batch tracer (`gen.array.batch`) each
operation acts per particle and each draw is per particle, so the same
transitions and the same body run $N$ particles at once:

<!-- code: batched -->
```clojure
(def particles (batched/simulate two-normals [] 1000 (rng/key 7)))

[(batched/batched? particles) (array/shape (batched/scores particles))]
;=> [true [1000]]

(let [t (nth (batched/traces particles) 17)]
  (array/allclose? (array/take (batched/scores particles) 17)
                   (:weight (gfi/assess two-normals [] (:choices t)
                                        (rng/key 0)))))
;=> true
```

No transition was written for the batch. GenMLX keeps batched copies of
its transitions; gen.clj has one set, and a batched run is the scalar
code under the tracer. The scalar handler stays the ground truth: after a
batched run, a few particles are run again through it, and a mismatch
makes the run fall back to one particle at a time.

**Faster paths.** Middleware replaces a transition, and it may change
what a model means. An optimization may not, and it plugs in elsewhere.
A *dispatcher*
(`dyn/with-dispatch`) runs before the handler and may return an
operation's result itself, from a closed form, say; the GFI laws check
that it agrees with the handler. The static language is such a path.
`(gen ^:static [] (let …))` declares a body whose graph is known, so
update skips the nodes that did not change and assess runs a score
function generated from the graph, without the handler. Its results are
the handler's:

<!-- code: static -->
```clojure
(def two-normals-static
  (gen ^:static []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal x 1))]
      (+ x y))))

(let [k (rng/key 9)
      choices (cm/choicemap :x 0.5 :y 1.0)]
  [(same-run? (gfi/simulate two-normals-static [] k)
              (gfi/simulate two-normals [] k))
   (array/allclose? (:weight (gfi/assess two-normals-static [] choices k))
                    (:weight (gfi/assess two-normals [] choices k)))])
;=> [true true]
```

On MLX a third path compiles: a gradient step of HMC or MALA runs the
model's assess, handler and all, under `array/compile`, which records its
array operations once as an MLX graph that MLX fuses and replays. GenMLX
also compiles whole static models into fused kernels; gen.clj does not.
The rule is the same everywhere: the handler is the semantics, and a
faster path must be observationally equivalent to it. The inference code
above never learns which path ran.

## Value semantics

Purity holds below the handler too. The values a transition draws and
adds are gen.array's arrays, and no gen.array operation changes an
array: each returns a new one. On the JVM backend a 0-d array is a host
value, a Clojure number or boolean, so a scalar model makes no arrays.
On MLX an array is a node of a lazy computation graph: an operation adds
a node and computes nothing until something evaluates the graph or reads
a value on the host. So gen.clj keeps evaluation out of the semantics.
No handler, transition or distribution calls `array/eval!`; inference
loops call it once per iteration (`gfi/eval-trace!`), so a lazy graph
stays one iteration deep. On an eager backend it does nothing, and on
every backend it returns what it was given:

<!-- code: values -->
```clojure
(array/shape (:score tr))
;=> []

(identical? tr (gfi/eval-trace! tr))
;=> true
```

A branch on a value is the exception: `truthy?` reads the value, and on
MLX that evaluates it. And arrays have no `=`, which is why this
chapter's checks compare with `array/equal?` and `array/allclose?`.

## Where the impurity lives

gen.clj is not pure in the type-theoretic sense, and it is worth being
exact about where. Three places hold mutable state that the semantics
reads:

- **One run's cells**: the two volatiles of the loop, and those of a few
  other runners (kernels, trace translators, proposal replays,
  enumeration), each made and dropped inside one synchronous call.
- **The implicit key source**, for an operation given no key.
- **Native handles**, the MLX backend's and llama.cpp's, inside their
  membranes.

The implicit key source is the only impure point in randomness. A call
without a key takes one from a root key folded with a call counter, so
two identical calls give different results. The root seed is random for
each JVM; `gfi/seed!` fixes it and `gfi/implicit-seed` reads it, to
reproduce a run. Here we seed it twice with 14, then seed it again with
the seed it had:

<!-- code: implicit-key -->
```clojure
(defn two-draws
  "Two simulated values of :x, after seeding the implicit key source."
  [seed]
  (gfi/seed! seed)
  [((gfi/simulate two-normals []) :x) ((gfi/simulate two-normals []) :x)])

(def before (gfi/implicit-seed))

(let [draws (two-draws 14)
      again (two-draws 14)]
  (gfi/seed! before)
  [(array/equal? (first draws) (second draws))
   (every? true? (map array/equal? draws again))])
;=> [false true]
```

Dynamic vars (the backend in effect, the store of parameters) are
configuration, which a run reads and never changes. The few caches
(compiled steps, the batch backends' interned values) change how fast an
operation runs, never what it returns.

This is a deliberate trade. Pure transitions can be tested by passing a
map and checking the map that comes back. They compose, so middleware
can wrap them and faster paths can stand in for them. Bodies can be read
in several ways (sampled, conditioned, scored, batched) because they
commit to none. The cells are the price, paid once, at the seam between
the body and the transitions. *Choice Maps, Traces and Selections* turns
to the data the handler builds: the choice maps it fills, the traces it
returns and the selections that point into them.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Half the evidence.** Generate `two-normals` with only `:y`
   constrained, to 2.0. Generate's transition adds nothing to the weight
   at `:x`. What is the weight, as a log density?
2. **Sparing keys.** Write a loop like `run-body` in which a constrained
   site takes no key, as in GenMLX. Constrain `:x` to a simulated
   trace's value and run generate with the trace's key through both
   loops. Which one reproduces the trace's `:y`?
3. **A third level.** Splice `two-level` under `:outer` in a new model.
   Predict the addresses, then check that the score is still the sum of
   the leaves' log densities. Why does the handler need no new code?
4. **An unguarded helper.** Move the branch of the guard example into a
   helper function `pick` that a `gen` body calls. Which branch does it
   take on `one-false`? Then define it with `dyn/defn-guarded`.

## Solutions

**1.** The weight is the log density of the constrained value only,
$\log \mathcal{N}(2; x, 1)$ at the drawn $x$. The draw of `:x` is the
proposal, and it cancels against its prior:

<!-- code: solution-1 -->
```clojure
(let [obs (cm/choicemap :y 2.0)
      {t :trace w :weight} (gfi/generate two-normals [] obs (rng/key 10))]
  (array/allclose? w (dc/log-prob (d/normal (t :x) 1) 2.0)))
;=> true
```

**2.** gen.clj's loop reproduces it, and the sparing loop does not:
there `:y` takes the key that `:x` took in the simulate, and draws
another value.

<!-- code: solution-2 -->
```clojure
(defn run-body-sparing
  "run-body, except that a site the state's constraints hold takes no key,
   as in GenMLX."
  [transition state body]
  (let [vol (volatile! state)
        trace (fn [addr dist]
                (let [p (cm/path addr)
                      [k-next k-site] (if (cm/has-value? (:constraints @vol) p)
                                        [(:key @vol) nil]
                                        (rng/split (:key @vol)))
                      [v state'] (transition (assoc @vol :key k-next)
                                             p dist k-site)]
                  (vreset! vol state')
                  v))
        retval (body trace)]
    (assoc @vol :retval retval)))

(let [k (rng/key 5)
      t (gfi/simulate two-normals [] k)
      start (assoc empty-state :key k :mode :generate :weight 0.0
                   :constraints (cm/choicemap :x (t :x)))]
  (mapv (fn [run]
          (let [state (run generate-site start two-normals-body)]
            (array/equal? (t :y) (cm/get-choice (:choices state) :y))))
        [run-body run-body-sparing]))
;=> [true false]
```

**3.** The addresses nest one level deeper, and the score is the same
sum. A call is more state threading, so a level needs no new code:

<!-- code: solution-3 -->
```clojure
(def three-level
  (gen []
    (splice :outer two-level)))

(def deeper (gfi/simulate three-level [] (rng/key 11)))

(cm/addresses (:choices deeper))
;=> [[:outer :m :obs] [:outer :theta]]

(let [theta (deeper [:outer :theta])]
  (array/allclose? (:score deeper)
                   (+ (dc/log-prob (d/normal 0 2) theta)
                      (dc/log-prob (d/normal theta 1)
                                   (deeper [:outer :m :obs])))))
;=> true
```

**4.** The helper is ordinary Clojure, outside the macro's rewrite, so
its `if` counts the array as true. `defn-guarded` applies a body's
guards to a helper:

<!-- code: solution-4 -->
```clojure
(defn pick
  "A branch on b, unguarded."
  [b]
  (if b :then :else))

(dyn/defn-guarded pick-guarded
  "A branch on b's value."
  [b]
  (if b :then :else))

[(:retval (gfi/simulate (gen [b] (pick b)) [one-false] (rng/key 1)))
 (:retval (gfi/simulate (gen [b] (pick-guarded b)) [one-false] (rng/key 1)))]
;=> [:then :else]
```
