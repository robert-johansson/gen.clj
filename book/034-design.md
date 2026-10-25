# Design Rationale

The chapters of this Part took the generative function interface as
given. They defined its operations, built inference from them and
derived acceptance ratios for moves between traces. This chapter asks
why gen.clj is built the way it is: why one handler, why explicit keys,
why loud errors, why layers, and what the design gives up.

Every decision in this chapter is recorded in `DESIGN.md`, gen.clj's
design document, and the chapter cites the section that makes it: §3 for
the principles, and §9's entries, the conventions and decisions, by their
titles. Where DESIGN records a decision without a reason, the chapter
states the decision and gives none. Each principle comes with a small
example, run and checked like the rest of the book. The chapter follows
the plan of the GenMLX book's *Design Rationale*. gen.clj is a reboot
of GenMLX, not a port line by line, and where the two decided
differently, the chapter says so.

The chapter's code lives in one namespace:

<!-- code: ns -->
```clojure
(ns book.design
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.gradients :as grads]
            [gen.inference.conjugate :as conj]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## A reboot, and why

gen.clj ports GenMLX, which runs ClojureScript on Apple's MLX, to Clojure
on the JVM, with Gen.jl's capabilities as the long-term bar (DESIGN §1).
§2 records what a reading of GenMLX's code found. GenMLX is about 62,000
lines of source. Its mathematical core (choice maps, selections,
traces, diffs, the scalar handler transitions, the distribution core and
`defdist`, the kernel algebra, importance sampling, the MH family, trace
translators and the catalog of GFI laws) is 3,000 to 4,000 lines, and
clean: GenMLX's own audit found no formula errors in about 40
derivations. About 25,000 lines are optimization paths that implement
the same semantics again: compiled paths, analytical elimination, fused
and vectorized MCMC variants, six batched copies of the scalar
transitions, combinators with three paths each. Most of GenMLX's roughly
150 fixes of core bugs were there. In the words of GenMLX's audit, which
§2 quotes: "The mathematics is not the risk surface; the duplication
is."

So gen.clj ported the core first and left the rest as deferred work,
each piece with a note on how to port it cleanly (§1). GenMLX's dynamic
generative function was rewritten rather than ported: there "every `gen`
call runs schema extraction, compilation and conjugacy analysis", so
the namespace's closure held 27 namespaces and 16,600 lines (§2). Two of
§3's principles read as the answer to the duplication. The first: the
handler is the semantics, and every optimization a plug-in
observationally equivalent to it. The third: one implementation per
semantic, so batched execution is the same code under a batch tracer,
"never as copies (GenMLX has six)".

## One body, every reading

Here is a model to carry through the chapter: a mean $\mu$ with a normal
prior of standard deviation 2, and three observations around it.

<!-- code: model -->
```clojure
(def noisy-mean
  (gen []
    (let [mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(def tr (gfi/simulate noisy-mean [] (rng/key 1)))

[(tr :mu) (:score tr)]
;=> [1.4318 -5.8904] ; float64
```

It is one body. The operations read it in different ways, and none of
them needed code written for this model. Generate with every choice of
the trace reproduces the trace, with its score as the weight, and assess
gives the same number:

<!-- code: readings -->
```clojure
(let [{g :trace w :weight} (gfi/generate noisy-mean [] (:choices tr)
                                         (rng/key 2))
      {a :weight} (gfi/assess noisy-mean [] (:choices tr) (rng/key 3))]
  [(= (:choices g) (:choices tr))
   (array/allclose? w (:score tr))
   (array/allclose? a (:score tr))])
;=> [true true true]
```

The score has a closed form, and so has its derivative in $\mu$:

$$
\log p(\mu, y) = \log \mathcal{N}(\mu; 0, 2) + \sum_{i=0}^{2} \log
\mathcal{N}(y_i; \mu, 1), \qquad \frac{\partial}{\partial \mu} \log p =
-\frac{\mu}{4} + \sum_{i=0}^{2} (y_i - \mu).
$$

<!-- code: closed-form -->
```clojure
(defn log-joint
  "log p(mu, ys) when mu is normal around 0 with standard deviation
   prior-sd (noisy-mean's is 2) and each y normal around mu with standard
   deviation 1. mu may be an array of one value per particle."
  [prior-sd mu ys]
  (reduce + (dc/log-prob (d/normal 0 prior-sd) mu)
          (map #(dc/log-prob (d/normal mu 1) %) ys)))

(defn d-log-joint
  "The derivative of (log-joint 2 mu ys) in mu: -mu/4 + the sum of
   (y - mu)."
  [mu ys]
  (reduce + (- (/ mu 4.0)) (map #(- % mu) ys)))

(defn ys-of
  "The three observations of a trace of noisy-mean."
  [t]
  (mapv #(t [:y %]) (range 3)))

(array/allclose? (:score tr) (log-joint 2 (tr :mu) (ys-of tr)))
;=> true
```

The gradient comes from the same body. `choice-gradients` runs the
model's assess, the handler's own operation, under gen.array's tape, a
reverse-mode autodiff that records one rule per array operation (DESIGN
§5.2):

<!-- code: gradient -->
```clojure
(let [{:keys [gradients]} (grads/choice-gradients tr (sel/select :mu))]
  (array/allclose? (cm/get-choice gradients :mu)
                   (d-log-joint (tr :mu) (ys-of tr))))
;=> true
```

Batched execution runs the same body once for many *particles*, under
the batch tracer, where each value holds one entry per particle (DESIGN
§9, *Batched execution*). A thousand particles conditioned on three
observations are one run. Their scores, their weights (the
observations' log likelihood, since each $\mu$ was drawn from its
prior) and their gradients all have the closed forms, particle by
particle:

<!-- code: batched -->
```clojure
(def ys [1.2 0.4 2.1])

(def observations
  (cm/choicemap [:y 0] (ys 0) [:y 1] (ys 1) [:y 2] (ys 2)))

(def particles
  (batched/generate noisy-mean [] observations 1000 (rng/key 4)))

(let [ps (:particles particles)
      mus (batched/choice ps :mu)
      {:keys [gradients]} (grads/choice-gradients ps (sel/select :mu))]
  [(batched/batched? ps)
   (array/shape mus)
   (array/allclose? (batched/scores ps) (log-joint 2 mus ys))
   (array/allclose? (:weights particles)
                    (- (log-joint 2 mus ys) (dc/log-prob (d/normal 0 2) mus)))
   (array/allclose? (cm/get-choice gradients :mu) (d-log-joint mus ys))])
;=> [true [1000] true true true]
```

The last check is gradients under the batch tracer: each particle gets
the gradient of its own scalar run, as JAX's `vmap` would give it (§9,
*Gradients under the batch tracer*). No gradient code was written for
batches, and no batched code for the model.

Finally the static language. The same body under `gen ^:static` is
analyzed into a graph when it is defined. Simulate, generate, update and
regenerate walk that graph with the handler's own `trace` and `splice`,
and assess runs a score function generated from it (§9, *The static
modeling language*: "a flag on `gen` … not a second macro or a second
semantics"). With the same key it makes the same trace:

<!-- code: static -->
```clojure
(def noisy-mean-static
  (gen ^:static []
    (let [mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(let [t (gfi/simulate noisy-mean-static [] (rng/key 1))]
  [(= (:choices t) (:choices tr)) (array/allclose? (:score t) (:score tr))])
;=> [true true]
```

That is six readings: simulate, generate, assess, gradients, batched
execution and the static walk, of one body written once.

## The handler is the semantics

DESIGN's first principle reads: "The GFI is the contract; the handler is
the semantics. Every optimization is a plug-in that must be
observationally equivalent to the handler. None of them may change what
a GFI operation *means*" (§3.1). The third says how: one implementation
per semantic (§3.3). The handler's transitions, one small pure function
per operation, say what a random choice means under each operation;
*The Handler Loop* took them apart.

An optimization therefore enters at a seam, and does not copy the
semantics. The batch tracer is one: it runs the transitions unchanged,
on values with a particle axis. The static language is another: it
decides which nodes an update reruns and reruns them with the handler's
code, and its generated assess is held to the handler by a law. A third
is the *dispatcher*, `dyn/with-dispatch`, the "dispatch seam" of §4's
L2. It gets the first chance at every operation and may answer it, from
a closed form say, or return nil to leave it to the handler. Here is one
that answers assess for `noisy-mean` from `log-joint`, with the prior's
standard deviation as an argument:

<!-- code: dispatcher -->
```clojure
(defn closed-form-assess
  "A dispatcher for noisy-mean: assess's weight from log-joint, with the
   prior's standard deviation given, in place of the handler's run; every
   other operation is left to the handler (nil)."
  [prior-sd]
  (fn [_gf op _args {:keys [constraints]}]
    (when (= op :assess)
      (let [mu (cm/get-choice constraints :mu)
            ys (mapv #(cm/get-choice constraints [:y %]) (range 3))]
        {:weight (log-joint prior-sd mu ys) :retval mu}))))

(def fast-mean (dyn/with-dispatch noisy-mean (closed-form-assess 2)))

(array/allclose? (:weight (gfi/assess fast-mean [] (:choices tr) (rng/key 5)))
                 (:score tr))
;=> true
```

It agrees with the handler on this trace. That is one input; *Laws are
tests* below holds it to the handler on many, and shows what happens
when the closed form is wrong.

The principle has a corollary that DESIGN states outright: no "generate
returns the posterior mean with a marginal weight" (§3.1). Conjugacy is
one of two things. It is either the thesis's *optimal internal
proposal*, which samples the latent from its exact conditional with the
weight $\log \bar p(\sigma)$ (thesis §4.1, pp. 145–146), or an
explicitly separate generative function with the latent summed out. The
first is `with-conjugate-proposal` (§9, *Conjugacy, and Kalman and HMM
generative functions*, provisional until the owner confirms it).
`noisy-mean` is normal–normal, and its marginal likelihood has a closed
form: the $y_i$ are jointly normal with mean 0 and covariance $I + 4J$,
$J$ a matrix of ones.

<!-- code: exact-marginal -->
```clojure
(defn log-marginal
  "log p(ys) for noisy-mean: the ys are normal with mean 0 and covariance
   I + 4J (J all ones), whose inverse is I - 4J/(1 + 4n) and determinant
   1 + 4n."
  [ys]
  (let [n (count ys)
        s (reduce + ys)
        ss (reduce + (map #(* % %) ys))
        c (+ 1.0 (* 4.0 n))]
    (* -0.5 (+ (* n (Math/log (* 2.0 Math/PI))) (Math/log c)
               (- ss (* (/ 4.0 c) s s))))))

(log-marginal ys)
;=> -4.93814
```

With the proposal, generate's weight is exactly that number, for any
key, and the trace still holds a $\mu$, drawn from the posterior, a
different one for each key:

<!-- code: conjugate -->
```clojure
(def conjugate-mean
  (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1] [:y 2]]))

(let [[r1 r2] (map #(gfi/generate conjugate-mean [] observations (rng/key %))
                   [5 6])]
  [(array/allclose? (:weight r1) (log-marginal ys))
   (array/allclose? (:weight r2) (log-marginal ys))
   (array/equal? ((:trace r1) :mu) ((:trace r2) :mu))])
;=> [true true false]
```

The second kind are `gen.state-space`'s Kalman and hidden Markov
generative functions, whose states are summed out and whose scores are
exact marginal likelihoods (§9, the same entry). Neither is chosen for
the user: "Chosen by the user, never detected (§3.1's corollary)." The
same entry records that six silently wrong marginals of GenMLX's are
loud errors or exact here, each a test.

## Random choices and their handler

*The Handler Loop* described `(trace addr dist)` as an *effect*: the
body names a need, a value from this distribution at this address, and a
handler chosen from outside the body decides what it means. The readings
above are that idea at work, since the body never learns which operation
runs it. What DESIGN fixes is that the handler is written once (§3.3),
and that the body is ordinary Clojure, whose `if` tests the `gen` macro
wraps in `array/truthy?`, "so a one-element array branches on its value
and a batched array throws instead of silently taking one branch" (§9,
*Branching on array values*).

Here gen.clj departs from GenMLX, where every `gen` call ran schema
extraction, compilation and conjugacy analysis (§2). gen.clj builds a
body's graph only when the body is flagged `^:static`, and its check
that the body is static is sound, "since `trace` and `splice` are the
macro's own locals, unlike GenMLX's schema extraction over quoted source
by symbol name" (§9, *The static modeling language*). Whether a dynamic
body could be static is a question the user asks, with a lint:

<!-- code: lint -->
```clojure
(static/could-be-static noisy-mean)
;=> {:static? true}
```

`noisy-mean` would take the flag, but it runs as the dynamic body it is
until the flag is written. DESIGN gives the reason for asking rather
than switching: the lint lets "a model get the static language's speed
where it fits without an implicit switch: GenMLX's automatic analysis
of every body is what the triage found least reliable" (§9, *Compiled
scores and a lint, in M4*).

## Layers

gen.clj is built in layers (DESIGN §4; the namespaces are those of the
project's guide, `CLAUDE.md`):

| Layer | What it holds (§4) | Namespaces |
|---|------------------------|---------------------|
| L0 | the array API: backend-neutral ops, gradients, keys | `gen.array`, `gen.backend.*`, `gen.random` |
| L1 | the data algebra: choice maps, selections, traces, diffs | `gen.choicemap`, `gen.trace`, … |
| L2 | the GFI: protocols, the handler, `gen`, the dispatch seam | `gen.gfi`, `gen.dynamic`, `gen.batched`, … |
| L3 | distributions | `gen.dist`, `gen.dist.core`, … |
| L4 | combinators | `gen.combinators`, `gen.custom`, `gen.state-space` |
| L5 | inference | `gen.inference.*`, `gen.gradients`, `gen.learning` |

Above them sit the LLM modules: a runtime over engines such as
llama.cpp, and the generative functions built on it.

![gen.clj's layers. An arrow points to what a layer builds
on.](figures/034-layers.svg)

The rule at the bottom reads: "Everything above L0 calls only the array
API. No backend type escapes L0 except the opaque array value. The same
model and inference code runs on either backend" (§4). The rule can be
checked against the code itself, by reading the `ns` form of every
source file:

<!-- code: sources -->
```clojure
(defn source-files
  "The Clojure source files under dir."
  [dir]
  (filter #(str/ends-with? (.getName ^java.io.File %) ".clj")
          (file-seq (io/file dir))))

(defn ns-requires
  "[name requires] of the source file f's ns form: requires, the gen.clj
   namespaces its :require clauses name."
  [f]
  (let [form (with-open [r (java.io.PushbackReader. (io/reader f))]
               (read r))
        [_ name & clauses] form]
    [name (set (for [clause clauses
                     :when (and (seq? clause) (= :require (first clause)))
                     spec (rest clause)
                     :let [lib (if (sequential? spec) (first spec) spec)]
                     :when (str/starts-with? (str lib) "gen.")]
                 lib))]))

(def requires
  "Every namespace of gen.clj's src/, with the gen.clj namespaces it
   requires."
  (into (sorted-map) (map ns-requires) (source-files "src/gen")))

(defn closure
  "The namespaces root loads: itself, what it requires, and so on."
  [root]
  (loop [seen #{} todo [root]]
    (if-let [n (peek todo)]
      (recur (conj seen n) (into (pop todo) (remove seen (requires n))))
      seen)))

(defn under?
  "Is the namespace named s the namespace root, or one inside it?"
  [s root]
  (or (= s root) (str/starts-with? s (str root "."))))

(defn layer
  "The layer of a gen.clj namespace, as CLAUDE.md's table gives it."
  [ns-name]
  (let [s (str ns-name)
        in? (fn [& roots] (some #(under? s %) roots))]
    (cond (in? "gen.array" "gen.backend" "gen.random") :L0
          (in? "gen.choicemap" "gen.selection" "gen.trace" "gen.diff") :L1
          (in? "gen.gfi" "gen.serialize" "gen.dynamic" "gen.batched"
               "gen.partition") :L2
          (in? "gen.dist") :L3
          (in? "gen.combinators" "gen.custom" "gen.state-space") :L4
          (in? "gen.inference" "gen.gradients" "gen.learning") :L5
          (in? "gen.llm") :llm)))
```

The only namespace that requires a backend is `gen.array`:

<!-- code: backend-rule -->
```clojure
(set (for [[n libs] requires
           :when (some #(str/starts-with? (str %) "gen.backend") libs)]
       n))
;=> #{gen.array}
```

So models, distributions and inference algorithms reach a backend only
through gen.array's operations, and the same code runs on the pure-JVM
backend, on MLX's CPU and on its GPU. The three are "all first-class":
each passes the conformance suite and the test battery, and "the core
assumes none of them" (§9, *Three ways to run, all first-class*). The
one exception DESIGN records to calling only gen.array is the LLM layer:
its engines compute on host floats with clojure.core, below the GFI like
a backend, and a token is drawn on the host from one uniform derived
from its site's key (§9, *The Branching protocol and the llama.cpp
engine*; *The token generative function's foundations*).

Two more principles concern dependencies. "No aggregator namespaces;
keep dependency closures small" (§3.8), and "The core depends on Clojure
plus the native membrane, nothing else" (§3.6, with the table of §8).
Here is everything `gen.dynamic`, the `gen` macro's namespace, loads, by
layer, and the core's one dependency:

<!-- code: closure -->
```clojure
(sort (distinct (map layer (closure 'gen.dynamic))))
;=> (:L0 :L1 :L2 :L3)

(keys (:deps (edn/read-string (slurp "deps.edn"))))
;=> (org.clojure/clojure)
```

A `gen` body loads no combinator and no inference, where GenMLX's
`dynamic.cljs` required "the whole compile + L3 stack", L3 being its
analytical elimination (§2). The closure does reach gen.clj's L3: the
handler draws and scores through the distributions' record and
multimethods, which live in `gen.dist.core`. The layers order what each
part builds on, not a strict order of requires, as §4 says.

## Explicit models, immutable traces

A trace is a value, a record no operation changes (§9, *Traces are
values*). DESIGN does not list it among the places where mutable state
may live: the semantics reads such state only in one run's local cells,
the implicit key source and native handles (§3.4). Arrays
are immutable values too (§5.1), and on the JVM backend they compare by
value, so traces holding them compare with `=` (§9, *Arrays are
values*). An operation that revises a trace returns a new one, in the
shape Gen.jl gives it: update returns the new trace, a weight, a diff of
the return value and the choices it discarded (§3.2, "Gen.jl-shaped GFI
from day one"; GenMLX's three update protocols collapse into one).

<!-- code: update -->
```clojure
(def moved (gfi/update tr (cm/choicemap :mu 0.5) (rng/key 8)))

(sort (keys moved))
;=> (:discard :retdiff :trace :weight)

[(tr :mu) ((:trace moved) :mu) (cm/get-choice (:discard moved) :mu)]
;=> [1.4318 0.5 1.4318] ; float64

(array/allclose? (:weight moved) (- (:score (:trace moved)) (:score tr)))
;=> true
```

The original trace still holds its $\mu$, which the discard also
records, and the weight is the change in score: every choice was either
constrained or kept.

A trace also stores each choice's log density, as Gen.jl's traces do,
"so update and regenerate weights use the true old values and `project`
never re-runs the model" (§9, *Traces store each choice's
log-probability*). Middleware that counts the handler's transitions
shows it: simulate runs four of them, one per choice, and project none:

<!-- code: project -->
```clojure
(defn counted
  "gf with every site transition counting its calls in the atom runs."
  [gf runs]
  (dyn/with-transitions gf
    (into {} (for [[op transition] dyn/site-transitions]
               [op (fn [state path dist key]
                     (swap! runs inc)
                     (transition state path dist key))]))))

(let [runs (atom 0)
      t (gfi/simulate (counted noisy-mean runs) [] (rng/key 1))
      after-simulate @runs
      prior (gfi/project t (sel/select :mu))]
  [after-simulate
   (- @runs after-simulate)
   (array/allclose? prior (dc/log-prob (d/normal 0 2) (t :mu)))])
;=> [4 0 true]
```

The model's other inputs are explicit too. Constraints are choice maps,
and a plain Clojure map in their place is an error (below). Trainable
parameters are an explicit store, a map attached with `dyn/with-params`,
and gradients are values: `param-gradients` returns a map, and an
optimizer returns a new store (§9, *Trainable parameters are an explicit
store*; *Learning* uses them). A static model's graph is data as well,
"for inspection and a DOT rendering" (§9, *The static modeling
language*); *Speeding Up* draws one.

## Randomness is a function of the key

"Every GFI op is a pure function of its inputs plus an explicit PRNG
key: the same key gives the same result" (§3.5). Keys travel in the
handler's state, never in metadata, and are never silently replaced by
fresh ones. Every site consumes one key whether it draws or is
constrained, so constraining a choice never shifts another site's draws;
*The Handler Loop* showed it, and showed that GenMLX's generate, which
gives a constrained site no key, does not have this property.

The keys are JAX's: the partitionable Threefry-2x32-20, in pure Clojure,
bit-identical to JAX, except that `fold-in` data past $2^{32}$ gives keys
of its own where JAX would mask it (§9, *Randomness*). Splitting the key
of seed 0 gives the two keys `jax.random.split` gives in JAX's default
mode:

<!-- code: jax-keys -->
```clojure
(mapv rng/key-data (rng/split (rng/key 0)))
;=> [[1797259609 2579123966] [928981903 3453687069]]
```

There is one impure point. An operation given no key draws one from the
*implicit key source*, whose root seed is random for each JVM;
`gfi/seed!` fixes it and `gfi/implicit-seed` reads it (§3.5). It serves
only the calls that omit their key. Seeding it differently changes the
keyless runs and leaves the keyed ones alone:

<!-- code: seed-free -->
```clojure
(def implicit-seed-before (gfi/implicit-seed))

(let [run (fn [seed]
            (gfi/seed! seed)
            [(gfi/simulate noisy-mean [] (rng/key 9))
             (gfi/simulate noisy-mean [])])
      [keyed-1 keyless-1] (run 1)
      [keyed-2 keyless-2] (run 2)]
  (gfi/seed! implicit-seed-before)
  [(= (:choices keyed-1) (:choices keyed-2))
   (= (:choices keyless-1) (:choices keyless-2))])
;=> [true false]
```

DESIGN keeps this rule where it is easy to break. A body may call
another generative function as a plain function, `(noise)`, as Gen.jl
allows. That call once took its key from the implicit source, so a run
with a fixed key did not repeat, silently; writing *Introduction to
Modeling* found it. Now the call takes the next key from the body's own
encapsulated randomness, and a run repeats with its key (§9,
*Encapsulated randomness*, "Plain calls in a body"):

<!-- code: plain-call -->
```clojure
(def noise
  (gen []
    (trace :e (d/normal 0 1))))

(def with-noise
  (gen []
    (+ (trace :mu (d/normal 0 1)) (noise))))

(array/equal? (:retval (gfi/simulate with-noise [] (rng/key 10)))
              (:retval (gfi/simulate with-noise [] (rng/key 10))))
;=> true
```

The other mutable pieces of gen.clj are caches and registries, the
compiled-step caches and the backends' bookkeeping among them, which
"never change what an operation means" (§3.4). DESIGN restated the
principle in this form on 2026-10-22, when the older wording, that
mutation is confined to three places, had stopped holding.

## Errors are loud

The project's rule for errors (rule 6 of its guide, which DESIGN's
entries invoke) is that they are loud. A shape mismatch, an unvisited
constraint, a repeated address and the like throw an `ex-info` with
data, and never degrade into NaN or a silent default. Each case below is
an entry of §9, with the reason it gives.

A plain map given as constraints would read as no constraints at all, so
it is an error that names the function to use (*Constraints are choice
maps*):

<!-- code: error-map -->
```clojure
;; error: generate: the constraints must be a choice map
(gfi/generate noisy-mean [] {:mu 1.0} (rng/key 11))
```

A Bernoulli value is a boolean, as in Gen.jl. GenMLX's were 0 and 1,
and a body's `if` takes 0 as true while the density scores it as false
(*Bernoulli values are booleans*):

<!-- code: error-bernoulli -->
```clojure
;; error: bernoulli: values are booleans
(def coin
  (gen []
    (trace :heads (d/bernoulli 0.3))))

(gfi/generate coin [] (cm/choicemap :heads 1) (rng/key 11))
```

A lookup of a choice that is not there throws, as Gen.jl throws
KeyError, where it once returned nil (*Lookups are loud*):

<!-- code: error-lookup -->
```clojure
;; error: trace: no choice or call at [:nu]
(tr :nu)
```

Each such lookup has a lenient form, with a not-found value as `nth`
has, and the questions never throw:

<!-- code: lenient -->
```clojure
[(tr :nu nil) (cm/has-value? (:choices tr) :nu)]
;=> [nil false]
```

Loudness has a limit where a check would read values on the host. On MLX
an array is a node of a lazy graph, and reading it forces an evaluation.
So a float cast to an integer type truncates toward zero, saturates at
the type's range and takes NaN to 0, as Java's casts do, since "a loud
error would need a host read of every cast on a lazy backend" (*Casts to
integers are Java's*):

<!-- code: casts -->
```clojure
(array/->clj (array/astype (array/array [1e20 -1e20 ##NaN 2.7 -2.7]) :int32))
;=> [2147483647 -2147483648 0 2 -2]
```

MLX clamps array indices rather than checking them, an exception to
NumPy's semantics that §5.3 documents. A draw beyond the float range is
+Inf with log density −Inf, for another reason: the value is exact for
what is stored, as IEEE floats and Julia's Distributions in Float32 give
(§9, *A draw beyond the float range is +Inf*). One rule is an obligation
on the user, as in the thesis and Gen.jl: values drawn with `untraced`
must not decide which addresses a run visits, "since no sound check
short of rerunning the body exists" (§9, *Encapsulated randomness*).

## Laws are tests

"The thesis's GFI invariants become test.check properties, one suite per
feature" (§3.7). For milestone M4, Gen.jl parity, "complete GFI law
coverage" was part of the exit criterion: every law the thesis states for
the features gen.clj has is a test (§1). Every new kind of generative
function runs them, and the static language and the dispatcher are each
held to the handler by one. *Verification: the Laws* is about them; here
is one at work. The law `dispatch-matches-handler` runs each operation
with and without the dispatcher, under the same keys, and compares the
results. It holds for the closed form above over 30 random cases:

<!-- code: law -->
```clojure
(defn dispatch-law
  "The law dispatch-matches-handler over 30 cases of gf, a dispatched
   noisy-mean, with a fixed seed: test.check's result."
  [gf]
  (laws/quick-check 30
              (laws/dispatch-matches-handler (laws/cases [["mean" gf [] []]]))
              {:seed 1}))

(select-keys (dispatch-law fast-mean) [:pass? :num-tests :vacuous])
;=> {:pass? true :num-tests 30 :vacuous 0}
```

A law's result names what failed (§9, *Law results*). Planting a likely
mistake, the prior's variance 4 read as its standard deviation, fails
it, and the failure names the check that compared the two weights:

<!-- code: law-bug -->
```clojure
(def wrong-prior (dispatch-law (dyn/with-dispatch noisy-mean
                                 (closed-form-assess 4))))

[(:pass? wrong-prior) (first (:check (:result-data wrong-prior)))]
;=> [false close?]
```

## The honest limitations

The design gives some things up, and DESIGN says which.

**No automatic elimination.** gen.clj never finds a conjugate pair by
itself (§3.1's corollary; §9, *Conjugacy*). Without
`with-conjugate-proposal`, generate draws $\mu$ from its prior, and its
weight is an importance weight that changes with the key, where the
wrapped model's was exact:

<!-- code: plain-weights -->
```clojure
(mapv #(:weight (gfi/generate noisy-mean [] observations (rng/key %))) [5 6])
;=> [-16.461 -11.222] ; float64
```

**No compilation ladder.** GenMLX's compiled paths are among the
optimization paths of §2's finding, and gen.clj has none of them. In
their place are three faster paths, each a plug-in held to the handler,
which *Speeding Up* takes in turn. The static language, by an explicit
flag, reruns only the nodes a change reaches, and generates a score
function for assess: 1.9 times faster at 10 observations and 2.6 times
at 100, on the JVM backend (§9, *Compiled scores and a lint, in M4*).
Batched execution runs many particles as one: importance sampling 5 to
90 times faster on the JVM backend, from 100 to 10,000 particles (§9,
*Batched execution*). And `array/compile` fuses a function's array
operations into one MLX graph. The graph of a static model is where
further compilation would start: "the way to compiled scores, batching
without the tracer, and CUDA graphs" (§9, *The static modeling
language*).

**What batching cannot promise.** A batched run is the scalar code under
the tracer, so a model whose code treats particles differently cannot
run batched. It falls back to one scalar trace per particle, with the
same answers:

<!-- code: fallback -->
```clojure
(def two-paths
  (gen []
    (if (trace :b (d/bernoulli 0.5))
      (trace :x (d/normal 0 1))
      (trace :z (d/normal 0 10)))))

(def mixed (batched/simulate two-paths [] 100 (rng/key 12)))

[(batched/batched? mixed) (:message (batched/fallback-reason mixed))]
;=> [false "Cannot branch on a batched value: it differs across particles"]

(count (batched/traces mixed))
;=> 100
```

Inside a `gen` body the fallback is guaranteed: the macro guards `if`,
`case` and `catch`, and puts guarded versions in place of clojure.core's
truthiness, lookup and type tests. Only Java interop on a choice value,
and dispatch on its class, see the batch. A plain helper function
outside `gen` is covered by two later mechanisms: a batched comparison
that nothing read raises the fallback, and spot checks rerun a few
particles through the scalar handler, the ground truth. What is left is
probabilistic: in a plain helper, "a `case` on a choice, a vector lookup
keyed by one, or a raw `if` on a comparison also used elsewhere is found
by the spot checks with high probability", and `dyn/defn-guarded` gives
a helper the guards deterministically (§9, *Batched execution*).

**MLX's costs per operation.** On MLX every array operation is a call
into MLX through the native binding. DESIGN §5.3 records the costs as
built: a downcall of 2.4 to 3.8 ns, an operation about 170 ns, an
evaluation about 11 µs on the CPU and 115 µs on the GPU; "the binding is
not the cost". On MLX even a scalar model's scores are arrays (§9,
*Running totals never NaN*). For scalar-heavy models, which are most
models, the pure-JVM backend "is likely *faster* than MLX, with no FFI
and no ~tens-of-µs eval floor. Large batched workloads are what the GPU
is for" (§5.2).

None of these touches what an operation means. They concern the cost of
an answer: its variance, which models run fast, and on what hardware.
*Verification: the Laws* turns to how the meaning is checked.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **An untraced draw.** Add `(untraced (d/normal 0 1))` as the first
   binding of `noisy-mean`'s body. Simulate it with key 1. Predict:
   are its choices `tr`'s? Its score?
2. **Two children of three.** Wrap `noisy-mean` with
   `with-conjugate-proposal`, declaring only `[:y 0]` and `[:y 1]` as
   the children of `:mu`. What does generate do? Then pass
   `{:check-exact? false}`. Is the weight still the exact log marginal
   likelihood? Is it still a correct importance weight?
3. **Where a backend is known.** Which layers does `gen.inference.mh`
   load? Does anything it loads, other than `gen.array`, require a
   backend?
4. **Undoing an update.** Update `(:trace moved)` with `(:discard
   moved)` as its constraints. Predict the trace, the weight and the
   discard.

## Solutions

**1.** Both are unchanged. An untraced draw takes its key from the
run's encapsulated randomness, which is folded from the operation's key
apart from the sites' sequence, so it never shifts a site's key (§9,
*Encapsulated randomness*). It is not a choice and adds nothing to the
score:

<!-- code: solution-1 -->
```clojure
(def noisy-mean-untraced
  (gen []
    (let [z (untraced (d/normal 0 1))
          mu (trace :mu (d/normal 0 2))
          y0 (trace [:y 0] (d/normal mu 1))
          y1 (trace [:y 1] (d/normal mu 1))
          y2 (trace [:y 2] (d/normal mu 1))]
      mu)))

(let [t (gfi/simulate noisy-mean-untraced [] (rng/key 1))]
  [(= (:choices t) (:choices tr)) (array/allclose? (:score t) (:score tr))])
;=> [true true]
```

**2.** `[:y 2]` depends on `:mu` but is not declared a child, so the
weight could not be exact, and generate says so:

<!-- code: solution-2-error -->
```clojure
;; error: with-conjugate-proposal: the choice at [:y 2] depends on the latent
(gfi/generate (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1]])
              [] observations (rng/key 13))
```

With `{:check-exact? false}` the weight is an importance weight. It
changes with the key, but it is still correct: the mean of its
exponential is the marginal likelihood $\bar p(\sigma)$ of the
observations, which 10,000 keys confirm within five standard errors:

<!-- code: solution-2 -->
```clojure
(def two-children
  (conj/with-conjugate-proposal noisy-mean :mu [[:y 0] [:y 1]]
                                {:check-exact? false}))

(def two-children-weights
  (mapv #(array/item (:weight (gfi/generate two-children [] observations %)))
        (rng/split (rng/key 14) 10000)))

(defn mean-and-se
  "The mean of the host numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) xs)) (dec n))]
    [mean (Math/sqrt (/ variance n))]))

(let [[mean se] (mean-and-se (mapv #(Math/exp %) two-children-weights))]
  ;; the mean of exp(weight) is p(ys), within five standard errors
  [(apply = two-children-weights)
   (< (abs (- mean (Math/exp (log-marginal ys)))) (* 5 se))])
;=> [false true]
```

**3.** `gen.inference.mh` loads layers L0, L1, L2 and L5: no
distribution and no combinator. Only `gen.array` requires a backend:

<!-- code: solution-3 -->
```clojure
(sort (distinct (map layer (closure 'gen.inference.mh))))
;=> (:L0 :L1 :L2 :L5)

(set (for [n (closure 'gen.inference.mh)
           :when (some #(str/starts-with? (str %) "gen.backend") (requires n))]
       n))
;=> #{gen.array}
```

**4.** The discard holds the choices the update replaced, so giving it
back restores `tr`'s choices. The weight is the first update's, negated,
and the new discard holds the value the first update wrote:

<!-- code: solution-4 -->
```clojure
(def back (gfi/update (:trace moved) (:discard moved) (rng/key 15)))

[(= (:choices (:trace back)) (:choices tr))
 (array/allclose? (:weight back) (- (:weight moved)))
 (cm/get-choice (:discard back) :mu)]
;=> [true true 0.5]
```
