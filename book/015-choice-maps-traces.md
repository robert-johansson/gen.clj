# Choice Maps, Traces and Selections

The chapters so far have used three data structures without stopping to
define them. This chapter looks at the data itself. A *trace* records one
run of a generative function, with what it needs to run again. A *choice
map* holds the run's random choices, by address. A *selection* names a
set of addresses, so that an operation can act on part of a trace. They
are small enough to read in full, and they are the vocabulary of
inference: update, regenerate, MCMC and SMC all take them and return
them.

This chapter ports chapter 14 of *The GenMLX Book*, *Choice Maps, Traces,
and Selections*, and adds what gen.clj does its own way: loud lookups,
Gen.jl's `merge`, traces called with an address, and traces written as
EDN. The code lives in one namespace:

<!-- code: ns -->
```clojure
(ns book.choice-maps-traces
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.serialize :as serialize]
            [gen.trace :as trace]))
```

## Choice maps have two node types

A choice map is built from two record types in `gen.choicemap`. A *value*
is a leaf, holding one random choice. A *node* is a branch, a persistent
Clojure map from addresses to choice maps. That is the whole data type:

<!-- code: two-types -->
```clojure
(def leaf (cm/->Value 3.0))

(def node (cm/->Node {:x (cm/->Value 1.0) :y (cm/->Value 2.0)}))

[(cm/has-value? leaf) (cm/has-value? node)
 (cm/has-value? (cm/get-submap node :x))]
;=> [true false true]

(= cm/EMPTY (cm/get-submap node :z))
;=> true

(pr-str node)
;=> "#gen/choicemap {:x 1.0, :y 2.0}"
```

`has-value?` tells the two apart. `get-submap` goes down one level, and
where nothing is it returns `cm/EMPTY`, the node with no entries, never
`nil`. Being records over Clojure maps, choice maps get value equality and
structural sharing from Clojure, and they print as tagged maps.

## Building and reading

`cm/choicemap` takes pairs of an address and a value; a value that is a
choice map or a plain map nests. `cm/from-map` lifts a nested Clojure
map. All three forms below make one choice map, in which `:params`
branches to `:slope` and `:intercept`:

<!-- code: build -->
```clojure
(def c
  (cm/choicemap :params (cm/choicemap :slope 2.0 :intercept 1.0)
                :y0 5.0))

[(= c (cm/from-map {:params {:slope 2.0 :intercept 1.0} :y0 5.0}))
 (= c (cm/choicemap [:params :slope] 2.0 [:params :intercept] 1.0
                    :y0 5.0))]
;=> [true true]
```

The third form shows what an address is. An *address* is any value except
a vector: a keyword, an integer, a string. A vector is a *path*, one
address per level of the tree. `[:params :slope]` is `:params`, then
`:slope`; the observations of *Getting Started*, `[:y 0]` and on, are
`:y`, then an integer, which Gen.jl writes `:y => 0`.

`get-choice` follows an address or a path to a leaf and returns its value,
`get-value` returns a leaf's value, and `cm/to-map` turns a choice map back
into plain data. `cm/addresses` lists every leaf's path, and `cm/values`
their values in the same order:

<!-- code: read -->
```clojure
[(cm/get-choice c [:params :slope]) (cm/get-value (cm/get-submap c :y0))]
;=> [2.0 5.0]

(cm/to-map (cm/get-submap c :params))
;=> {:slope 2.0 :intercept 1.0}

[(cm/addresses c) (cm/values c)]
;=> [[[:params :intercept] [:params :slope] [:y0]] [1.0 2.0 5.0]]
```

The order is canonical, addresses sorted, whatever order the map was built
in (`c` was built slope first), so choice maps with the same addresses
list their values in the same order.

## Lookups are loud

GenMLX's `get-choice` returns `nil` where there is no value, and the `nil`
travels on, to fail later and elsewhere. gen.clj throws at the lookup, as
Gen.jl throws its `KeyError`:

<!-- code: lookup-error -->
```clojure
;; error: get-choice: no value at [:z]: nothing is there
(cm/get-choice c :z)
```

The error's data says why: nothing is there, a node is there, or the path
runs on through a leaf. Every lookup also takes a not-found value, as
`nth` does, and the questions `has-value?` and `has-submap?` never throw.
Use those where a missing choice is expected rather than a mistake:

<!-- code: lookups -->
```clojure
(defn reason
  "What the lookup (f) finds: :found, or its error's :reason."
  [f]
  (try (f) :found
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(mapv #(reason (fn [] (cm/get-choice c %)))
      [[:params :slope] :z :params [:y0 :a]])
;=> [:found :absent :node :through-value]

[(cm/get-choice c :z nil) (cm/has-value? c :z) (cm/has-submap? c :params)]
;=> [nil false true]
```

## Choice maps are immutable

Functions that seem to modify a choice map return a new one and leave
their input as it was. `set-choice` sets the value at a path. `cm/merge`
follows Gen.jl rather than GenMLX: it is the union of choice maps whose
addresses are disjoint.

<!-- code: immutable -->
```clojure
(def c2 (cm/set-choice c [:params :slope] 3.0))

[(cm/get-choice c2 [:params :slope]) (cm/get-choice c [:params :slope])]
;=> [3.0 2.0]

(def a (cm/choicemap :x 1.0 :y 2.0))

(cm/to-map (cm/merge a (cm/choicemap :z 3.0)))
;=> {:x 1.0 :y 2.0 :z 3.0}
```

Two values at one address are a loud error, where GenMLX's `merge-cm` lets
the second map win. Two sets of observations that both fix an address are
usually a mistake, and a silent winner would hide it:

<!-- code: merge-error -->
```clojure
;; error: merge: the choice maps overlap
(cm/merge a (cm/choicemap :y 99.0 :z 3.0))
```

When one map should win, say so with `cm/override`. The first map is
unchanged:

<!-- code: override -->
```clojure
[(cm/to-map (cm/override a (cm/choicemap :y 99.0 :z 3.0))) (cm/to-map a)]
;=> [{:x 1.0 :y 99.0 :z 3.0} {:x 1.0 :y 2.0}]
```

The whole system relies on this. `update` returns a new trace and leaves
the old one, so an algorithm can keep both and choose. The cost is small,
because a new choice map shares structure with the old. Set one choice in
a map of a thousand, and the other 999 leaves are the same objects in both:

<!-- code: sharing -->
```clojure
(def big
  (apply cm/choicemap (mapcat (fn [i] [[:y i] (double i)]) (range 1000))))

(def revised (cm/set-choice big [:y 500] -1.0))

(count (filter #(identical? (cm/get-submap big %) (cm/get-submap revised %))
               (cm/addresses big)))
;=> 999
```

Only the path from the root to the new leaf is new (Exercise 1). Inference
holds thousands of traces and revises them constantly; sharing makes that
affordable.

## Traces: five fields, self-contained

A trace is the record that `simulate`, `generate` and `update` make. Here
is one, from the model of *Getting Started* on three points:

<!-- code: model -->
```clojure
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def tr (gfi/simulate line-model [[1.0 2.0 3.0]] (rng/key 1)))
```

`gen.trace` defines a trace as a record of five fields: `:gen-fn`, the
generative function that made it; `:args`, the arguments it ran on;
`:choices`, the choice map; `:retval`, what the body returned; and
`:score`, $\log p(\text{choices} \mid \text{args})$. A kind of generative
function may keep more under namespaced keys: a dynamic function's trace
keeps each choice's log density and each call's trace, which `regenerate`
and `project` read instead of running the model again.

<!-- code: fields -->
```clojure
(take 5 (keys tr))
;=> [:gen-fn :args :choices :retval :score]

[(identical? line-model (:gen-fn tr)) (:args tr)]
;=> [true [[1.0 2.0 3.0]]]

[(:retval tr) (:score tr)]
;=> [[1.432 -5.475] -9.497] ; float64

(every? qualified-keyword? (drop 5 (keys tr)))
;=> true
```

Traces have no print method yet, and one printed whole fills a screen, so
this book shows their fields.

A trace answers for its choices. Called with an address or a path, it
returns the choice there, as Gen.jl's `trace[addr]` does, and at the empty
path its return value. It takes a not-found value too. One lookup stays
quiet: a trace is a record, so a keyword called on it, or `get`, reads the
record's fields. `(:score tr)` is the score, but `(:slope tr)` is `nil`,
whatever the trace holds at `:slope`. Call the trace instead:

<!-- code: call -->
```clojure
[(tr :slope) (tr [:y 0])]
;=> [1.432 -5.395] ; float64

[(= (tr []) (:retval tr)) (tr :noise nil)]
;=> [true nil]

[(:slope tr) (get tr :slope)]
;=> [nil nil]
```

Without a not-found value, an address with nothing there is a loud error:

<!-- code: call-error -->
```clojure
;; error: trace: no choice or call at [:noise]: nothing is there
(tr :noise)
```

The first field makes a trace *self-contained*: an operation given a trace
needs nothing else to run its model again. `gfi/update` takes the trace
and new constraints, but neither the model nor its arguments. Given the
trace's own choices, it changes nothing, and its weight, a difference of
log densities, is zero. Given a new slope, it returns a new trace, and the
old one keeps its slope:

<!-- code: self-contained -->
```clojure
(def same (gfi/update tr (:choices tr) (rng/key 2)))

[(array/allclose? (:weight same) 0.0)
 (= (:choices (:trace same)) (:choices tr))]
;=> [true true]

(def moved (:trace (gfi/update tr (cm/choicemap :slope 0.5) (rng/key 3))))

[(moved :slope) (tr :slope)]
;=> [0.5 1.432] ; float64
```

Everything the update needed traveled inside the trace, so inference can
pass traces around as its currency without carrying the model alongside.
What a weight means when it is not zero is the subject of *Update and
Regenerate*.

## Composed traces nest

`(splice addr gf & args)` calls another generative function inside a
model, with its choices under `addr`. Here two noisy sensors of one
position are two calls of one sub-model:

<!-- code: sensors -->
```clojure
(def sensor
  (gen [mu]
    (trace :value (d/normal mu 0.5))))

(def fusion
  (gen [position]
    (let [a (splice :sensor-a sensor position)
          b (splice :sensor-b sensor position)]
      {:a a :b b})))

(def ft (gfi/simulate fusion [5.0] (rng/key 4)))

(cm/addresses (:choices ft))
;=> [[:sensor-a :value] [:sensor-b :value]]
```

The two calls make a choice at the same address, `:value`, yet they never
collide, because each lives under its own call's address. The tree of the
choice map is the namespacing: a sub-model uses whatever addresses it
likes, and the caller keeps them apart by where it calls it.

Each call keeps its own trace, with five fields of its own, which
`trace/subtrace` returns (Gen.jl's `get_call(tr, addr).subtrace`). Its
choices are the caller's under the call's address, and the caller's trace,
called with that address, returns the call's return value. The caller's
score is the sum of its calls' scores:

<!-- code: subtrace -->
```clojure
(def a-trace (trace/subtrace ft :sensor-a))

[(identical? sensor (:gen-fn a-trace)) (:args a-trace)
 (= (:choices a-trace) (cm/get-submap (:choices ft) :sensor-a))
 (= (ft :sensor-a) (:retval a-trace) (a-trace :value))]
;=> [true [5.0] true true]

(array/allclose? (:score ft)
                 (+ (:score a-trace) (:score (trace/subtrace ft :sensor-b))))
;=> true
```

The trace of a composed model is a tree whose leaves are choices and
whose score is their sum. Nesting adds depth, never a special case.
*Composition* builds larger models from such calls.

## Traces as data

`gen.serialize` writes choice maps as EDN text with gen.clj's tags, and
reads them back with `clojure.edn`, which evaluates no code. A trace is
written as its arguments, choices and score, never its generative
function, which is code. To read a trace back you give the generative
function, and `read-trace` rebuilds the trace with `generate`, every
choice constrained. The rebuilt trace is the written one:

<!-- code: write -->
```clojure
(serialize/write c)
;=> "#gen/choicemap {:params {:intercept 1.0, :slope 2.0}, :y0 5.0}"

(= c (serialize/read (serialize/write c)))
;=> true

(def tr-again (serialize/read-trace line-model (serialize/write-trace tr)))

(let [parts (juxt :args :choices :retval :score)]
  (= (parts tr) (parts tr-again)))
;=> true
```

The written score makes the read exact. `read-trace` checks that the
rebuilt score is the written one, bit for bit, so a different generative
function is a loud error, even one that makes choices at the same
addresses. Here a sensor with a noise of 1.0 reads the trace of one with
0.5:

<!-- code: read-trace-error -->
```clojure
;; error: read-trace: the rebuilt score
(def wide-sensor
  (gen [mu]
    (trace :value (d/normal mu 1.0))))

(serialize/read-trace wide-sensor (serialize/write-trace a-trace))
```

## Selections name subsets

Some operations act on part of a trace: `regenerate` draws a chosen subset
of the choices again, and `project` scores one. A *selection* names that
subset, a set of addresses with everything under them. It answers two
questions: is the leaf at a path selected (`sel/selected?`), and what is
selected under an address (`sel/get-subselection`, the counterpart of
`get-submap`). `sel/select` picks addresses or paths, and `sel/all` and
`sel/none` select everything and nothing:

<!-- code: select -->
```clojure
(def s (sel/select :x :y))

[(sel/selected? s :x) (sel/selected? s :z)
 (sel/selected? sel/all :anything) (sel/selected? sel/none :x)]
;=> [true false true false]
```

`sel/hierarchical` reaches under an address, a call's, say. Selections
have the tree shape of the choice maps they cut, so a path selects the
same thing. As in Gen.jl, selecting an address selects everything under
it. The converse does not hold: the slope's selection does not select a
leaf at `:params`, since it selects only part of what is under `:params`.

<!-- code: hierarchical -->
```clojure
(def slope-only (sel/hierarchical :params (sel/select :slope)))

[(sel/selected? slope-only [:params :slope])
 (sel/selected? slope-only [:params :intercept])
 (sel/selected? (sel/get-subselection slope-only :params) :slope)]
;=> [true false true]

(= slope-only (sel/select [:params :slope]))
;=> true

[(sel/selected? (sel/select :params) [:params :slope])
 (sel/selected? slope-only :params)]
;=> [true false]
```

## Selections are a Boolean algebra

`sel/complement` is every address a selection does not select. `sel/all`
is the top, `sel/none` the bottom, and the complement is negation: the
complement of the top is the bottom, of the bottom the top, and two
complements give back the selection itself, the same object. Negation
holds path by path, nested paths and selections included:

<!-- code: algebra -->
```clojure
[(= sel/none (sel/complement sel/all))
 (= sel/all (sel/complement sel/none))
 (identical? s (sel/complement (sel/complement s)))]
;=> [true true true]

(def some-paths
  [[:x] [:y] [:z] [:y 0] [:params :slope] [:params :intercept]])

(defn negated?
  "Does the complement of selection t select exactly the paths of
   some-paths that t does not?"
  [t]
  (every? #(not= (sel/selected? t %) (sel/selected? (sel/complement t) %))
          some-paths))

(every? negated? [sel/all sel/none s slope-only (sel/select :params)])
;=> true
```

Because selections combine by laws rather than by looking at a model, an
algorithm can build one from pieces and hand it to any GFI operation.
"Everything but the slope" is `(sel/complement (sel/select :slope))`,
whatever else the model holds. gen.selection has no union or intersection
of two selections; `select` makes the union of the addresses it names,
which covers this book's uses. Exercise 3 adds the rest of the algebra.

## The three types, working together

`gfi/project` puts all three types to work. It takes a trace and a
selection and returns the part of the trace's score that the selected
choices contribute, the sum of their log densities. `sel/none` contributes
nothing, `sel/all` the whole score, and the slope its prior log density:

<!-- code: project -->
```clojure
[(array/allclose? (gfi/project tr sel/none) 0.0)
 (array/allclose? (gfi/project tr sel/all) (:score tr))
 (array/allclose? (gfi/project tr (sel/select :slope))
                  (dc/log-prob (d/normal 0 2) (tr :slope)))]
;=> [true true true]
```

A selection and its complement are disjoint and cover every choice, so
their projections split the score, whichever selection we start from, in
a composed trace too:

<!-- code: partition -->
```clojure
(defn splits-score?
  "Do trace t's projections on selection s and on its complement add up to
   its score?"
  [t s]
  (array/allclose? (+ (gfi/project t s) (gfi/project t (sel/complement s)))
                   (:score t)))

(every? #(splits-score? tr %)
        [sel/none (sel/select :slope) (sel/select :y) (sel/select [:y 1])])
;=> true

(every? #(splits-score? ft %)
        [(sel/select :sensor-a) (sel/select [:sensor-b :value])])
;=> true
```

The Boolean algebra and the score being a sum are one fact seen from two
sides: a trace's score is a sum over its leaves, and a selection chooses
which leaves to add.

The types convert into one another, too. `sel/from-choicemap` selects the
addresses of a choice map, so the complement of the observations'
selection is every latent choice, whatever the model names them: the
choices an MCMC algorithm moves. `sel/get-selected` cuts out what a
selection selects:

<!-- code: latents -->
```clojure
(def observations (cm/choicemap [:y 0] -3.9 [:y 1] -2.6 [:y 2] -1.1))

(def latents (sel/complement (sel/from-choicemap observations)))

(cm/addresses (sel/get-selected (:choices tr) latents))
;=> [[:intercept] [:slope]]
```

That is the whole data algebra: a record of one run, a tree of its
choices, and an algebra that names their subsets. *Update and Regenerate*
puts two operations on them to work, revising a trace rather than reading
it.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **What a revision shares.** `c2` is `c` with a new slope. Predict which
   of `c`'s choice maps (the root, `:params`, the slope, the intercept,
   `:y0`) are the same objects in `c2`, then check with `identical?`.
2. **Everything but the observations.** Select the latent choices of
   `line-model` two ways: `(sel/select :slope :intercept)` and
   `(sel/complement (sel/select :y))`. Do they project `tr` the same? A
   later version of the model adds a noise level at `:noise`; which of
   the two selects it?
3. **A union.** A selection is anything that implements the two methods
   of `sel/ISelection`, `-selected?` and `-subselection`. Write the union
   of two selections as a record, and their intersection by De Morgan's
   law. Is the projection on the union of two disjoint selections the sum
   of their projections?

## Solutions

**1.** `set-choice` makes new choice maps along the path from the root to
the new leaf: the root, the `:params` node and the leaf. Everything off
that path is shared:

<!-- code: solution-1 -->
```clojure
(mapv #(identical? (cm/get-submap c %) (cm/get-submap c2 %))
      [[] [:params] [:params :slope] [:params :intercept] [:y0]])
;=> [false false false true true]
```

**2.** On `line-model` the two are the same set. Only the complement
selects `:noise`: the selection by name lists what to take, the
complement what to leave.

<!-- code: solution-2 -->
```clojure
(def by-name (sel/select :slope :intercept))

(def but-ys (sel/complement (sel/select :y)))

(array/allclose? (gfi/project tr by-name) (gfi/project tr but-ys))
;=> true

[(sel/selected? by-name :noise) (sel/selected? but-ys :noise)]
;=> [false true]
```

**3.** The union selects a leaf when either selection does, and its
subselection under an address is the union of theirs. The intersection is
the complement of the union of the complements. `project` reads a
selection only through the two methods, so it takes the new kind, and the
projections add:

<!-- code: solution-3 -->
```clojure
(defrecord Union [p q]
  sel/ISelection
  (-selected? [_ addr] (or (sel/selected? p addr) (sel/selected? q addr)))
  (-subselection [_ addr]
    (->Union (sel/get-subselection p addr) (sel/get-subselection q addr))))

(defn intersection
  "What both selections p and q select, by De Morgan's law."
  [p q]
  (sel/complement (->Union (sel/complement p) (sel/complement q))))

(let [yz (sel/select :y :z)]
  [(mapv #(sel/selected? (->Union s yz) %) [:w :x :y :z])
   (mapv #(sel/selected? (intersection s yz) %) [:w :x :y :z])])
;=> [[false true true true] [false false true false]]

(array/allclose? (gfi/project tr (->Union (sel/select :slope) (sel/select :y)))
                 (+ (gfi/project tr (sel/select :slope))
                    (gfi/project tr (sel/select :y))))
;=> true
```
