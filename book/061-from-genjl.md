# Coming from Gen.jl

gen.clj follows Gen.jl's semantics, and its names where Clojure allows. A
model written in Gen.jl ports line by line, and the generative function
interface means the same thing in both. This appendix collects what
differs, for a reader who knows Gen.jl. Like the chapters, every example
in it runs in the book's test.

The code lives in one namespace:

<!-- code: ns -->
```clojure
(ns book.from-genjl
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))
```

## Models

| Gen.jl | gen.clj |
|--------|---------|
| `@gen function f(x) … end` | `(def f (gen [x] …))` |
| `x ~ normal(0, 1)` | `(let [x (trace :x (d/normal 0 1))] …)` |
| `{:a => 1} ~ normal(0, 1)` | `(trace [:a 1] (d/normal 0 1))` |
| `{:z} ~ g(args…)` | `(splice :z g args…)` |
| `{*} ~ g(args…)` | `(splice g args…)` |
| `@param theta` | `(param :theta)`, the store given by `with-params` |
| `@gen (static) function …` | `(gen ^:static [args] …)` |
| `@dist` | `gen.dist.dsl/dist` |

gen.clj has no `~` sugar: every choice names its address, so `trace`
always takes one. Gen.jl's nested address `:a => 1 => :b` is the path
`[:a 1 :b]`. A call at an address puts the callee's choices under it:

<!-- code: model -->
```clojure
(def sensor
  (gen [mu]
    (trace :reading (d/normal mu 0.5))))

(def model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 10))]
      (doseq [i (range n)]
        (splice [:sensor i] sensor mu))
      mu)))

(def tr (gfi/simulate model [3] (rng/key 1)))

(cm/addresses (:choices tr))
;=> [[:mu] [:sensor 0 :reading] [:sensor 1 :reading] [:sensor 2 :reading]]
```

An unaddressed splice, Gen.jl's `{*}`, puts them among the caller's own:

<!-- code: splice-namespace -->
```clojure
(def shared
  (gen []
    (let [mu (trace :mu (d/normal 0 10))]
      (splice sensor mu))))

(cm/addresses (:choices (gfi/simulate shared [] (rng/key 2))))
;=> [[:mu] [:reading]]
```

## Indices start at 0

Clojure counts from 0, and so does gen.clj, everywhere Gen.jl counts from
1: the addresses of Map's elements and Unfold's steps, Recurse's nodes
(the root is 0), the values of `categorical` and `categorical-logits`
(0 to n − 1), and Switch's branch index (branch 0 is the first).

<!-- code: zero-based -->
```clojure
(def flips (comb/map (gen [p] (trace :b (d/bernoulli p)))))

(cm/addresses (:choices (gfi/simulate flips [[0.1 0.9]] (rng/key 3))))
;=> [[0 :b] [1 :b]]

(def die (gen [] (trace :face (d/categorical [0.25 0.25 0.25 0.25]))))

(set (for [k (rng/split (rng/key 4) 200)]
       (array/item ((gfi/simulate die [] k) :face))))
;=> #{0 1 2 3}
```

## Randomness takes a key

Gen.jl draws from Julia's global random number generator. gen.clj's
randomness is a pure function of a *key*: every operation that draws
takes one, the same key gives the same draws, and `rng/split` makes
independent keys from one. At the REPL a key may be left out, and gen.clj
draws one from its implicit source, which `gfi/seed!` fixes. The keys are
JAX's Threefry; within one precision, a key draws the same values on every
backend.

<!-- code: keys -->
```clojure
(= (array/->clj (:score (gfi/simulate model [3] (rng/key 1))))
   (array/->clj (:score tr)))
;=> true
```

## Traces and choice maps

| Gen.jl | gen.clj |
|--------|---------|
| `get_choices(tr)` | `(:choices tr)` |
| `tr[addr]` | `(tr addr)` |
| `get_retval(tr)`, `tr[]` | `(:retval tr)`, `(tr [])` |
| `get_score(tr)` | `(:score tr)` |
| `get_args(tr)` | `(:args tr)` |
| `get_gen_fn(tr)` | `(:gen-fn tr)` |
| `choicemap((:a, 1.0))` | `(cm/choicemap :a 1.0)` |
| `cm[:a]` | `(cm :a)`, `(cm/get-choice cm :a)` |
| `get_submap(cm, :a)` | `(cm/get-submap cm :a)` |
| `has_value(cm, :a)` | `(cm/has-value? cm :a)` |
| `merge(a, b)` | `(cm/merge a b)` |
| `to_array`, `from_array` | `cm/to-array`, `cm/from-array` |
| `select(:a, :b)` | `(sel/select :a :b)` |
| `selectall()` | `sel/all` |
| `complement(s)` | `(sel/complement s)` |

A trace is a record, and called with an address it returns the choice
there, as `tr[addr]` does; at the empty path, its return value:

<!-- code: trace-fields -->
```clojure
[(= model (:gen-fn tr)) (:args tr) (= (:retval tr) (tr []) (tr :mu))]
;=> [true [3] true]
```

A choice map called with an address returns the value there, as
`cm[addr]` does, the trace's choices included; a not-found value goes
second:

<!-- code: choicemap-call -->
```clojure
(def readings (:choices tr))

[(= (readings :mu) (tr :mu)) (some? (readings [:sensor 0 :reading]))
 (readings [:sensor 7 :reading] nil)]
;=> [true true nil]
```

Like `cm[addr]`, it is an error where the address holds nested choices,
not the submap, which `cm/get-submap` returns:

<!-- code: choicemap-call-node -->
```clojure
;; error: choice map: no value at [:sensor]: nested choices are there
(readings :sensor)
```

One lookup differs from Gen.jl. A keyword reads the record's fields, so
`(:mu tr)` is `nil` whatever the trace holds at `:mu`. Call the trace:

<!-- code: record-field-trap -->
```clojure
[(:mu tr) (some? (tr :mu))]
;=> [nil true]
```

Lookups of a missing address are loud, as Gen.jl's `KeyError` is; each
also takes a not-found value, as `nth` does:

<!-- code: lookup-error -->
```clojure
;; error: trace: no choice or call at [:nu]
(tr :nu)
```

`cm/merge` is Gen.jl's: overlapping choice maps are an error.
`cm/override` lets the second win.

<!-- code: merge-error -->
```clojure
;; error: merge: the choice maps overlap
(cm/merge (cm/choicemap :a 1.0) (cm/choicemap :a 2.0))
```

Selections add a union and an intersection to Gen.jl's, which has
neither:

<!-- code: selections -->
```clojure
(def latent (sel/complement (sel/select :sensor)))

(def with-first (sel/union latent (sel/select [:sensor 0])))

(def only-mu (sel/intersection latent (sel/select :mu :sensor)))

[(sel/selected? with-first [:sensor 0 :reading])
 (sel/selected? with-first [:sensor 1 :reading])
 (sel/selected? only-mu :mu) (sel/selected? only-mu [:sensor 1])]
;=> [true false true false]
```

## The generative function interface

The operations are Gen.jl's, with a key last and maps for results:

| Gen.jl | gen.clj |
|--------|---------|
| `simulate(gf, args)` | `(gfi/simulate gf args key)` |
| `(tr, w) = generate(gf, args, cm)` | `{:keys [trace weight]} (gfi/generate gf args cm key)` |
| `update(tr, args, argdiffs, cm)` | `(gfi/update tr args argdiffs cm key)`, giving `:trace :weight :retdiff :discard` |
| `regenerate(tr, args, argdiffs, sel)` | `(gfi/regenerate tr args argdiffs sel key)` |
| `assess(gf, args, cm)` | `(gfi/assess gf args cm key)`, giving `:weight :retval` |
| `project(tr, sel)` | `(gfi/project tr sel)` |
| `propose(gf, args)` | `(gfi/propose gf args key)` |
| `NoChange()`, `UnknownChange()` | `diff/no-change`, `diff/unknown-change` |
| `VectorDiff` | `diff/vector-diff` |

`update` and `regenerate` also take just the constraints (or selection)
and a key, the arguments unchanged.

<!-- code: update -->
```clojure
(def observed
  (:trace (gfi/generate model [3] (cm/choicemap [:sensor 0 :reading] 1.2)
                        (rng/key 5))))

(let [{t2 :trace w :weight discard :discard}
      (gfi/update observed [3] [diff/no-change] (cm/choicemap :mu 1.0)
                  (rng/key 6))]
  [(t2 :mu)
   (cm/addresses discard)
   (array/allclose? w (- (:score t2) (:score observed)))])
;=> [1.0 [[:mu]] true]
```

## Distributions

gen.clj's distributions have Gen.jl's names and parameterizations:
`normal(mu, std)`, `gamma(shape, scale)`, `categorical(probs)` beside
`categorical-logits`, and so on. A bernoulli's value is a boolean. The
families Gen.jl does not have take Distributions.jl's parameters, which
GenDistributions' users write. *The Distributions* lists them all.

## The inference library

Functions that mutate in Gen.jl (their names end in `!`) return new
values in gen.clj: a particle filter's state, a parameter store and an
optimizer's state are values, passed on.

| Gen.jl | gen.clj |
|--------|---------|
| `metropolis_hastings`, `mh` | `mh/mh` |
| `hmc`, `mala` | `mcmc/hmc`, `mcmc/mala` (and `mcmc/nuts`) |
| `elliptical_slice` | `mcmc/elliptical-slice` |
| `map_optimize` | `mcmc/map-optimize` |
| `importance_sampling` | `is/importance-sampling` (with `:log-weights` and `:ess`) |
| `importance_resampling` | `is/importance-resampling` |
| `pf_initialize`, `pf_update!` | `pf/initialize`, `pf/step` |
| `pf_resample!`, `pf_rejuvenate!` | `pf/resample`, `pf/rejuvenate` |
| `black_box_vi!`, `black_box_vimco!` | `vi/black-box-vi`, `vi/black-box-vimco` |
| `train!` | `learning/train` (it draws fresh data per batch) |
| `choice_gradients` | `gradients/choice-gradients` |
| `accumulate_param_gradients!`, `apply!` | `gradients/param-gradients`, `learning/step` |

## Booleans in host code

Inside a `gen` body, every `if` reads its test's value on any backend. In
host code (an inference loop, an analysis), a trace or a choice map
called with an address returns a boolean choice as a host boolean. A
value read another way (`cm/get-choice`, a return value) is a 0-d array
on MLX, which Clojure's `if` takes as true: branch on it through
`array/truthy?`.

<!-- code: booleans -->
```clojure
(def coin (gen [] (trace :heads (d/bernoulli 0.0))))

(def ctr (gfi/simulate coin [] (rng/key 7)))

[(ctr :heads)
 (if (ctr :heads) :yes :no)
 (if ((:choices ctr) :heads) :yes :no)
 (if (array/truthy? (cm/get-choice (:choices ctr) :heads)) :yes :no)]
;=> [false :no :no :no]
```

## Plain calls

A plain call of a generative function inside a body, `(f args…)`, is
Gen.jl's untraced call: its choices are not the caller's. Its key comes
from the caller's encapsulated randomness, so the caller's run still
repeats with its own key. When the callee makes random choices, the
caller's score no longer accounts for them: it is an estimate, which
operations that need exact densities refuse. `splice` is usually what is
meant.

<!-- code: plain-call -->
```clojure
(def noise (gen [] (trace :z (d/normal 0 1))))

(def with-plain-call (gen [] (+ (noise) 1.0)))

(let [run #(gfi/simulate with-plain-call [] (rng/key 8))]
  [(= (array/->clj (:retval (run))) (array/->clj (:retval (run))))
   (cm/empty? (:choices (run)))])
;=> [true true]
```

## What Gen.jl does not have

- **Batched execution:** a model runs N particles at once, unchanged
  (*Speeding Up*).
- **Backends:** one array API on the JVM and on MLX's CPU and GPUs
  (*Arrays, Backends and Gradients*).
- **The laws:** the interface's laws run as tests, on any generative
  function (*Verification: the Laws*).
- **Language models as generative functions** (Part IV).
- **Loud errors** where Gen.jl may continue: a repeated address, a
  constraint no site visits, a NaN parameter.
