# Arrays, Backends and Gradients

Every chapter so far has computed with gen.array: the `+` and `*` in a
model's body, the log densities its distributions add to a score, the
gradients HMC follows. This chapter is about that layer, the bottom one
of the layers *Design Rationale* drew. It has four parts: one array API,
two backends under it, the keys every draw takes, and the gradients. It
matters because every other layer is written against the API alone. So a
model runs unchanged on the pure-JVM backend and on MLX, on a CPU or a
GPU, and, less obviously, under differentiation and batched execution,
which are backends of their own.

The sources are gen.clj's specification of the API and its backend
contract, `docs/array-api.md`, and DESIGN.md §5. The chapter is new to
this book: the GenMLX book assumed MLX, and gen.clj has two backends.
The book runs on the JVM backend, and a reader may run it on MLX. Where
the chapter shows what one backend does, its code names that backend, so
that each check holds wherever it runs. What only MLX does is described,
with its source, and shown where it can be by code that is a no-op on
the JVM backend.

The chapter's code lives in one namespace. Besides the aliases of the
earlier chapters, it uses four of the namespaces underneath:
`gen.array.protocols`, the backend contract; `gen.array.dtype`, the
dtypes and their promotion; `gen.array.batch`, the batch tracer; and
`gen.random.threefry`, the generator behind the keys.

<!-- code: ns -->
```clojure
(ns book.arrays-backends
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dtype]
            [gen.array.protocols :as p]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.random.threefry :as threefry]
            [gen.selection :as sel]))
```

## Numbers are arrays

An *array* in gen.clj is one of three things: a Clojure number, a
Clojure boolean, or a *backend array*, an immutable n-dimensional array
that a backend made. Numbers and booleans are 0-d arrays, arrays of
shape `[]`, that belong to no backend. `array/array` makes a backend
array from host data:

<!-- code: values -->
```clojure
(def m (array/array [[1.0 2.0] [3.0 4.0]]))

[(array/shape m) (array/ndim m) (array/shape 2.5) (array/ndim 2.5)]
;=> [[2 2] 2 [] 0]

[(array/host? 2.5) (array/host? true) (array/host? m) (array/array? m)]
;=> [true true false true]
```

An operation whose operands are all numbers runs on the host and
returns a number, with no backend involved: *host math*. gen.array's
operators extend Clojure's, and on numbers they give Clojure's results
with two exceptions. `/` divides as floats, never giving a `Ratio`, and
integer overflow wraps, where clojure.core's `+` throws:

<!-- code: host-math -->
```clojure
[(+ 1 2) (* 2 3.5) (/ 1 2) (array/exp 0.0)]
;=> [3 7.0 0.5 1.0]

(clojure.core// 1 2)
;=> 1/2

(+ Long/MAX_VALUE 1)
;=> -9223372036854775808
```

That is why every chapter's namespace refers gen.array's `+ - * /` in
place of clojure.core's: arithmetic written once works on numbers and on
arrays alike. The project's own namespaces refer the comparisons too;
the book calls those `array/<` and so on. Four names are never referred:
in gen.array `max` and `min` are reductions, `take` is a gather and
`flatten` a reshape, which a reader would take for clojure.core's.

A host number has a dtype, the same on every backend: a `Double` is
float64, a `Long` int64 and a `Boolean` bool.

<!-- code: host-dtypes -->
```clojure
(mapv array/dtype [2.5 3 true])
;=> [:float64 :int64 :bool]
```

The JVM backend goes one step further: every 0-d array it makes is a
host value. A sum over a whole array is a `Double`. So a scalar model,
whose values are all 0-d, never allocates an array, and it pays
host-math prices: about 4 ns for a `+` of two numbers, against 55 to
65 ns for two one-element arrays (docs/array-api.md §11). MLX keeps 0-d
results as MLX arrays, which stay lazy, as *Two backends* below explains.
The code names the JVM backend where it shows what that backend does:

<!-- code: jvm -->
```clojure
(def jvm
  "The JVM backend, named: code that shows what it does runs on it
   wherever the book runs."
  (array/backend-named :jvm))
```

<!-- code: zero-d -->
```clojure
(array/with-backend jvm
  (let [s (array/sum (array/array [1.0 2.0 3.0]))]
    [s (instance? Double s)]))
;=> [6.0 true]

(array/item (array/sum (array/array [1.0 2.0 3.0])))
;=> 6.0
```

Code that needs a host number from a 0-d result, on any backend, asks
for it with `array/item`, as the second form does.

Arrays have no `=`: an MLX array is not a Clojure value that `=`
understands. Compare them with `array/equal?`, which is exact, or with
`array/allclose?`, which allows for rounding:

<!-- code: equality -->
```clojure
[(array/equal? (array/array [1 2]) (array/array [1 2]))
 (array/equal? (+ 0.1 0.2) 0.3)
 (array/allclose? (+ 0.1 0.2) 0.3)]
;=> [true false true]
```

## Broadcasting

Elementwise operations broadcast by NumPy's rules. Shapes are aligned
from their last axis, and an axis of length 1, or a missing one,
stretches to match the other. A column of three plus a row of three is
a table:

<!-- code: broadcast -->
```clojure
(def column (array/array [[0.0] [10.0] [20.0]]))

(def row (array/array [1.0 2.0 3.0]))

(def table (+ column row))

[(array/shape column) (array/shape row) (array/shape table)]
;=> [[3 1] [3] [3 3]]

table
;=> [[1.0 2.0 3.0] [11.0 12.0 13.0] [21.0 22.0 23.0]]
```

Shapes that do not broadcast are a loud error, before any value is
computed:

<!-- code: broadcast-error -->
```clojure
;; error: Shapes [3] and [2] do not broadcast
(+ row (array/array [1.0 2.0]))
```

Reductions take an options map: `:axis`, one axis or several (all by
default), and `:keep-dims?`. Following NumPy and MLX, `array/max` is a
reduction and `array/maximum` is elementwise:

<!-- code: reductions -->
```clojure
[(array/sum table {:axis 0}) (array/shape (array/sum table {:axis -1}))]
;=> [[33.0 36.0 39.0] [3]]

(array/shape (array/sum table {:axis -1 :keep-dims? true}))
;=> [3 1]

[(array/max row) (array/maximum row 2.0)]
;=> [3.0 [2.0 2.0 3.0]]
```

`array/pick` is the one operation NumPy does not have. It indexes the
last axis by an int array, broadcasting over the others. Picked from a
table of log probabilities, one row per choice, it is a categorical log
density for every row at once. `d/categorical-logits` computes its log
density so, from a log-softmax:

<!-- code: pick -->
```clojure
(array/pick (array/log (array/array [[0.1 0.9] [0.5 0.5]]))
            (array/array [1 0]))
;=> [-0.1054 -0.6931]
```

## Dtypes and promotion

A *dtype* is a keyword: `:bool`, `:int32`, `:int64`, `:uint32`,
`:float16`, `:bfloat16`, `:float32` or `:float64`. The JVM backend
implements the five in the table below. Each backend has default dtypes,
float64 and int64 on the JVM backend, float32 and int32 on MLX. Core
code names a dtype only where it must, as for indices, which are ints.
An array made from host data takes the defaults of the backend that
makes it:

<!-- code: defaults -->
```clojure
(array/defaults jvm)
;=> {:float :float64 :int :int64}

(let [{:keys [float int]} (array/defaults array/*backend*)]
  [(= float (array/dtype (array/array [1.5 2.0])))
   (= int (array/dtype (array/array [1 2])))])
;=> [true true]
```

When arrays of two dtypes meet, the result has their *promoted* dtype.
Promotion is computed once, in gen.array, before a backend is called, so
every backend promotes alike: a backend only ever sees operands that
share a dtype. Here are the sums of each pair, on the JVM backend:

<!-- code: promotion -->
```clojure
(def dtypes [:bool :int32 :int64 :float32 :float64])

(defn promoted
  "The dtype of the sum of two arrays of the dtypes a and b."
  [a b]
  (array/dtype (+ (array/zeros [1] {:dtype a}) (array/zeros [1] {:dtype b}))))

(def promotion
  "The dtype of a sum, by the dtypes of its operands, on the JVM backend."
  (array/with-backend jvm
    (into {} (for [a dtypes] [a (mapv #(promoted a %) dtypes)]))))

(promotion :bool)
;=> [:int64 :int32 :int64 :float32 :float64]

(promotion :int32)
;=> [:int32 :int32 :int64 :float32 :float64]

(promotion :int64)
;=> [:int64 :int64 :int64 :float32 :float64]

(promotion :float32)
;=> [:float32 :float32 :float32 :float32 :float64]

(promotion :float64)
;=> [:float64 :float64 :float64 :float64 :float64]
```

| `+` | bool | int32 | int64 | float32 | float64 |
|------|------|------|------|------|------|
| **bool** | int64 | int32 | int64 | float32 | float64 |
| **int32** | int32 | int32 | int64 | float32 | float64 |
| **int64** | int64 | int64 | int64 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 |

Integers meet at the wider one, integers and floats meet at the float,
and floats at the wider one. Two cells are not NumPy's. An integer array
meeting a float32 one gives float32, as in JAX, where NumPy gives
float64 for int64 and float32. And two bool arrays add to the backend's
default int, as in Julia, where NumPy's sum of two booleans is their
logical or.

Numbers are *weakly typed*, as in JAX. A number meeting an array takes
the array's dtype and never widens it, unless it raises the kind: a
float meeting an int array gives the backend's default float. That is
what keeps a model's literal constants, `(* 0.5 x)`, from turning its
float32 arrays into float64 ones on MLX:

<!-- code: weak -->
```clojure
(let [f32 (array/array [1.0 2.0] {:dtype :float32})
      i32 (array/array [1 2] {:dtype :int32})]
  [(array/dtype (+ f32 2.5)) (array/dtype (+ i32 2))
   (= (array/dtype (+ i32 2.5)) (:float (array/defaults array/*backend*)))])
;=> [:float32 :int32 true]
```

Booleans in arithmetic count as integers, as in Julia: `(+ true true)`
is 2 and `(array/negate true)` is -1. Comparisons return bool arrays,
and so do Bernoulli draws, as Gen.jl's are `Bool`. (GenMLX's were 0 and
1 as floats, which its code then compared with 1.)

<!-- code: booleans -->
```clojure
[(+ true true) (array/negate true) (* true 2.5)]
;=> [2 -1 2.5]

(let [b (array/array [true false true])]
  [(array/sum b) (array/dtype (array/< row 2.5))
   (array/dtype (rng/bernoulli (rng/key 1) 0.5 [3]))])
;=> [2 :bool :bool]
```

A float32 array holds float32 values, also on the JVM backend. That
backend stores every dtype in a `double[]`, lets the dtype carry the
meaning, and rounds each result to its dtype, which is what lets it
mirror MLX's float32 numerics:

<!-- code: float32 -->
```clojure
(= (array/->clj (array/array [0.1] {:dtype :float32})) [(double (float 0.1))])
;=> true

(array/item (array/array 0.1 {:dtype :float32}))
;=> 0.10000000149011612
```

The same storage holds an int64 value exactly only up to $2^{53}$,
which is ample for counts and indices (docs/array-api.md §5). Past it,
the JVM backend rounds as a `double` does:

<!-- code: storage -->
```clojure
(array/with-backend jvm
  (array/->clj (+ (array/array [(long (Math/pow 2 53))]) 1)))
;=> [9007199254740992]
```

A float cast to an integer dtype follows Java's casts, as *Design
Rationale* showed: it truncates toward zero and saturates at the
dtype's range.

## Two backends

A *backend* is a value that implements one protocol, `IBackend` in
`gen.array.protocols`. Its methods are what gen.array calls once it has
found an operation's backend and dtype. gen.clj has two backends.

**The JVM backend** is pure Clojure, always available, with no native
dependency. It is eager: an operation computes its result before it
returns, with primitive loops over `double[]`, one small function per
operation so that the JIT compiles each. It computes in float64 by
default, which makes it the reference the MLX backend is tested against:
a conformance suite runs every operation of the backend contract on
both backends with the same data, and compares (docs/array-api.md §13).
It is the book's gate.

**The MLX backend** binds Apple's MLX through its C API, mlx-c, with
Java's Foreign Function and Memory API. It runs on MLX's CPU device and
on its GPU: Metal on macOS, CUDA on Linux. It differs from the JVM
backend in three ways that code must respect, and in none that changes
what an operation means (DESIGN.md §5.3, docs/array-api.md §13):

- **It is lazy.** An MLX operation adds a node to a graph, at about
  170 ns, and computes nothing. Reading a value on the host evaluates
  the graph: `array/item`, `array/->clj`, `truthy?`, printing, `=`. An
  evaluation costs about 11 µs on the CPU and 115 µs on the GPU.
- **It computes in float32 by default,** and in int32 for integers.
  Float64 is available on the CPU device, with `-Dgen.mlx.dtypes=64`.
- **Its 0-d results are MLX arrays,** so a scalar model stays lazy too.

`-Dgen.backend=mlx` makes MLX the default, and `-Dgen.mlx.device=cpu` or
`gpu` chooses its device. All three ways to run, the pure-JVM backend,
MLX's CPU and MLX's GPU, are first-class: each passes the conformance
suite and the test battery, and the core assumes none of them (DESIGN.md
§9, *Three ways to run, all first-class*).

### Evaluating at the boundaries

Laziness puts one obligation on inference code: evaluate at the
boundary of each iteration. `array/eval!` forces every array in a tree
and returns the tree, and `gfi/eval-trace!` does it for a trace's
choices, return value and score. gen.clj's inference loops call it once
per iteration, for MCMC moves, particle-filter steps, importance samples
and optimizer steps, so a lazy graph stays one iteration deep. On the
JVM backend both return their argument and do nothing else. A loop of
our own does the same as the library's:

<!-- code: eval -->
```clojure
(def noisy
  (gen []
    (let [mu (trace :mu (d/normal 0 1))]
      (trace :y (d/normal mu 0.5)))))

(defn mh-chain
  "n MH moves of tr's :mu, the trace evaluated after each move."
  [tr n key]
  (reduce (fn [tr k] (gfi/eval-trace! (:trace (mh/mh tr (sel/select :mu) k))))
          tr
          (rng/split key n)))

(def start (:trace (gfi/generate noisy [] (cm/choicemap :y 1.2) (rng/key 2))))

(identical? start (gfi/eval-trace! start))
;=> true

((mh-chain start 100 (rng/key 3)) :mu)
;=> 1.151 ; float64
```

Core code never evaluates inside a model, a handler transition or a
distribution, where GenMLX did (docs/array-api.md §9): what an
operation means never depends on when its values are computed.

### Memory

The array API has no memory management. An MLX array is freed once the
JVM finds it unreachable, but the JVM's garbage collector does not see
MLX's buffers, and Metal allows only about 500,000 of them, one for
every array with data, also on the CPU device. So the backend bounds
memory itself (docs/array-api.md §13, *Threads and memory*):

- It collects, by running the JVM's collector and freeing what it found,
  when 100,000 arrays are live, when MLX's active memory passes an
  eighth of RAM, or when the arrays made since the last collection hold
  a sixteenth of it.
- It bounds MLX's cache of freed buffers at a sixteenth of RAM and MLX's
  memory limit at half. MLX's defaults are all of RAM for both, and with
  them one test JVM grew to 40 GB on a 32 GB machine.
- At Metal's buffer limit it collects and tries once more. At the limit
  again, the error is loud: an ex-info with `:reason :mlx-buffer-limit`
  and a hint.

One thing no collection can bound. An unevaluated graph holds every
array it reads, inside MLX, after the JVM has dropped their handles, and
MLX exposes no graph size. Only evaluation bounds a graph, which is the
reason for evaluating at the boundary. *Speeding Up* gave the other half
of a long loop's discipline: keep host numbers, not traces, where it
keeps many results.

### What laziness changes

A lazy value cannot be checked without being evaluated. `array/concrete?`
says whether a value can be read without forcing a computation: yes for
host numbers and the JVM backend's arrays, no for MLX's. Distributions
validate a parameter given as host data on every backend, and an array
parameter only when it is concrete:

<!-- code: concrete -->
```clojure
(array/with-backend jvm (array/concrete? (array/array [0.3 0.7])))
;=> true
```

<!-- code: validation -->
```clojure
;; error: bernoulli: prob must be in [0, 1], got 1.5
(array/with-backend jvm (d/bernoulli (array/array 1.5)))
```

On MLX the same array parameter is not read, so an invalid probability
there is not an error. For the same reason, MLX clamps an out-of-range
array index, where the JVM backend raises:

<!-- code: index-check -->
```clojure
;; error: Index 5 is out of bounds for a dimension of size 2
(array/with-backend jvm
  (array/take (array/array [1.0 2.0]) (array/array [5])))
```

Elsewhere, where MLX's semantics differ from NumPy's, the backend
follows NumPy's. MLX's `sign` of NaN is 0, its `argmax` skips NaNs, its
integer floor division truncates toward zero, and its float64 `exp`,
`sin` and `erf` on the CPU are float32 approximations. The backend
computes those functions exactly, and gives NaN, the first NaN's index
and the floor, as the JVM backend does:

<!-- code: numpy -->
```clojure
(let [x (array/array [-7.0 ##NaN 7.0])]
  [(array/isnan (array/sign x))
   (array/floor-div (array/array [-7 7]) 2)
   (array/argmax x)])
;=> [[false true false] [-4 3] 1]
```

The remaining differences are documented rather than hidden: Metal
flushes float32 subnormals to zero, array indices are clamped, and
degenerate linear algebra may give NaN where the JVM backend gives an
infinity (docs/array-api.md §13).

## Keys and Threefry

Every draw takes a *key*, and the same key gives the same draw, as
*Getting Started* showed. A key is a pair of 32-bit words, a value of
pure Clojure that never touches a backend. It is not an array:

<!-- code: keys -->
```clojure
(def k (rng/key 42))

[(pr-str k) (rng/key-data k) (array/array? k)]
;=> ["#gen/key[0 42]" [0 42] false]

(= (rng/split k 5) (rng/split k 5))
;=> true

(= (rng/fold-in k 3) ((rng/split k 5) 3))
;=> true
```

`rng/split` makes keys from a key, and `rng/fold-in` derives the key at
an index, which is the key at that index of any split long enough. The
scheme is JAX's default one, "partitionable" Threefry: Threefry-2x32
with 20 rounds, the counter-based generator of Random123 that JAX and
MLX use, with JAX's layout of keys and draws (docs/array-api.md §7). The
key at index $i$ of a split is the Threefry hash of the counter $i$
under the key. So gen.clj's keys are JAX's, bit for bit, and so are its
uniform draws: the scheme was verified against JAX 0.9.1, and the
project's test fixtures hold JAX's own keys and draws:

<!-- code: jax -->
```clojure
(def jax
  "JAX's keys and draws for the key 42 (test/fixtures/random.edn)."
  (edn/read-string (slurp "test/fixtures/random.edn")))

(= (mapv rng/key-data (rng/split k 3))
   (:out (first (filter #(= 3 (:n %)) (:split jax)))))
;=> true

(= (array/->clj (rng/uniform k [6])) (:out (:uniform-f64 jax)))
;=> true ; float64

(= (array/->clj (rng/uniform k [6] {:dtype :float32}))
   (:out (:uniform-f32 jax)))
;=> true
```

The float64 check is marked `; float64`: on MLX's GPU, whose floats are
float32, the default draw is a float32 one.

Element $i$ of a draw depends only on the key and $i$. A shorter draw is
a prefix of a longer one, and a shape only arranges the elements:

<!-- code: index-local -->
```clojure
(let [u (array/->clj (rng/uniform k [6]))]
  [(= (array/->clj (rng/uniform k [3])) (subvec u 0 3))
   (= (array/->clj (rng/uniform k [2 3])) [(subvec u 0 3) (subvec u 3 6)])])
;=> [true true]
```

So a batch of particles can draw one value each from a site's key, and
particle $i$'s draw does not depend on how many particles there are.

### Why float32 draws differ

Every sampler is built on one backend primitive, `-random-uniform`: a
uniform array of a shape and a dtype for a key. Everything else (normal,
Gumbel, gamma, Bernoulli, categorical) is a composite on top of it,
written once in `gen.random`, so the two backends sample from identical
laws. The uniform depends on the dtype. For element $i$, Threefry hashes
the counter $i$ to two 32-bit words, $w_0$ and $w_1$. A float64 uniform
takes the top 52 bits of the 64-bit word $w_0 w_1$ as its mantissa. A
float32 uniform takes the top 23 bits of the 32-bit word $w_0 \oplus
w_1$, their exclusive or:

<!-- code: bits -->
```clojure
(defn uniform-64
  "The float64 uniform of a 64-bit Threefry word w: its top 52 bits."
  [w]
  (* (unsigned-bit-shift-right w 12) (Math/pow 2 -52)))

(defn uniform-32
  "The float32 uniform of a 32-bit Threefry word w: its top 23 bits."
  [w]
  (* (unsigned-bit-shift-right w 9) (Math/pow 2 -23)))

[(= (mapv uniform-64 (threefry/bits64 k 6)) (:out (:uniform-f64 jax)))
 (= (mapv uniform-32 (threefry/bits32 k 6)) (:out (:uniform-f32 jax)))]
;=> [true true]

(first (:out (:uniform-f64 jax)))
;=> 0.4267

(first (:out (:uniform-f32 jax)))
;=> 0.4887
```

The two dtypes use different bits, so a key's float32 uniform is not its
float64 uniform rounded: the first element is 0.4267 in float64 and
0.4887 in float32. Across 10,000 elements of one key, the two are
unrelated:

<!-- code: pairs -->
```clojure
(def pairs
  "The float64 and float32 uniforms of the key 7, element by element."
  (array/with-backend jvm
    (let [k (rng/key 7)]
      (mapv vector
            (array/->clj (rng/uniform k [10000]))
            (array/->clj (rng/uniform k [10000] {:dtype :float32}))))))
```

<!-- code: uniform-figure -->
```clojure
;; figure: figures/041-uniforms.svg
(svg/plot [{:points (take 600 pairs) :r 1.5 :opacity 0.6}]
          {:x-range [0 1] :y-range [0 1]
           :x-label "float64 uniform" :y-label "float32 uniform"})
```

![The float64 uniform (across) and the float32 uniform (up) of each of
the first 600 elements of one key's draws.](figures/041-uniforms.svg)

Their sample correlation is within five of its standard errors,
$1/\sqrt{n}$, of zero:

<!-- code: correlation -->
```clojure
(defn correlation
  "The sample correlation of the pairs [x y]."
  [pairs]
  (let [n (count pairs)
        mean (fn [vs] (/ (reduce + vs) n))
        xs (map first pairs)
        ys (map second pairs)
        mx (mean xs)
        my (mean ys)
        dx (map #(- % mx) xs)
        dy (map #(- % my) ys)
        dot (fn [a b] (reduce + (map * a b)))]
    (/ (dot dx dy) (Math/sqrt (* (dot dx dx) (dot dy dy))))))

(< (abs (correlation pairs)) (/ 5 (Math/sqrt (count pairs))))
;=> true
```

A normal draw is $\sqrt 2\,\operatorname{erfinv}(u)$ of a uniform $u$ on
$[\operatorname{nextafter}(-1, 0), 1)$, as in JAX. It agrees with JAX's
to rounding, not bit for bit, since the two compute erfinv by different
routines:

<!-- code: normals -->
```clojure
(def sqrt-2 (Math/sqrt 2.0))

(let [lo (Math/nextAfter -1.0 0.0)
      u (rng/uniform k [6] {:low lo :high 1.0})]
  (array/allclose? (rng/normal k [6]) (* sqrt-2 (array/erfinv u))))
;=> true ; float64

(array/allclose? (rng/normal k [6]) (array/array (:out (:normal-f64 jax))))
;=> true ; float64
```

So in float32 every sampled value differs, and with it everything
downstream: a trace, an acceptance, the path of a chain. That is why the
book marks sampled values `; float64`, and states statistical results as
checks with bands, which hold for both. The same key's float64 and
float32 normals are both standard normals, their means and variances
within five standard errors of 0 and 1, and unrelated to each other:

<!-- code: normal-dtypes -->
```clojure
(defn moments
  "The mean and the variance of the values xs."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)]
    [mean (/ (reduce + (map #(array/square (- % mean)) xs)) (dec n))]))

(defn standard?
  "Are the mean and the variance of n standard normal draws within five
   standard errors of 0 and 1? (Their errors are 1/√n and √(2/n).)"
  [xs]
  (let [n (count xs)
        [mean variance] (moments xs)]
    (and (< (abs mean) (* 5 (Math/sqrt (/ 1.0 n))))
         (< (abs (- variance 1.0)) (* 5 (Math/sqrt (/ 2.0 n)))))))

(def normals
  "The key 7's float64 and float32 standard normals, 10,000 of each."
  (array/with-backend jvm
    (let [k (rng/key 7)]
      [(array/->clj (rng/normal k [10000]))
       (array/->clj (rng/normal k [10000] {:dtype :float32}))])))

(mapv standard? normals)
;=> [true true]

(let [[f64 f32] normals]
  (< (abs (correlation (map vector f64 f32))) (/ 5 (Math/sqrt 10000))))
;=> true
```

MLX computes the same Threefry with its own uint32 operations. The same
key draws bit-identical uniforms on MLX and on the JVM backend, in
float32 and float64 alike, and since every sampler is a composite over
the uniform, the two backends' draws then agree up to the rounding of
each operation (docs/array-api.md §13, *Randomness*). Because keys are
values, nothing in gen.clj keeps random state. The one implicit source
is the GFI's optional key at the REPL, which `gfi/seed!` fixes.

## Gradients

`array/grad` turns a function into its gradient function: the gradient
with respect to argument 0 by default, or to the arguments at
`:argnums`. `array/value-and-grad` gives the value too. An argument may
be a tree of maps and vectors, and its gradient has the same structure.
A normal's log density, written in gen.dist with gen.array's operators,
is differentiable in both parameters:

<!-- code: grad -->
```clojure
(defn log-density
  "The log density of 1.5 under a normal of mean mu and standard
   deviation sigma."
  [mu sigma]
  (dc/log-prob (d/normal mu sigma) 1.5))

((array/grad log-density) 0.5 2.0)
;=> 0.25

((array/grad log-density {:argnums [0 1]}) 0.5 2.0)
;=> [0.25 -0.375]

((array/value-and-grad log-density {:argnums [0 1]}) 0.5 2.0)
;=> [-1.7371 [0.25 -0.375]]

((array/grad (fn [{:keys [mu sigma]}] (log-density mu sigma)))
 {:mu 0.5 :sigma 2.0})
;=> {:mu 0.25 :sigma -0.375}
```

At $x = 1.5$, $\mu = 0.5$ and $\sigma = 2$ the derivatives are $(x -
\mu)/\sigma^2 = 0.25$ and $-1/\sigma + (x - \mu)^2/\sigma^3 = -0.375$.

### The tape

The gradients come from `gen.array.tape`, a reverse-mode automatic
differentiation written against gen.array's public API, by default on
every backend. `grad` wraps the differentiated arguments as *tracked*
values. A tracked value is immutable: it holds its primal value, the
tracked values it was computed from, and a closure for its VJP, which
maps a cotangent of the value to cotangents of its inputs. There is no
global tape: the graph is the values' links, and the backward pass sorts
it and sweeps it once. Tracked values belong to a *tape backend*, made
afresh for each call, which wraps the real backend. It computes each
primitive operation's value on the real backend and records the
operation's VJP rule. Inside a differentiated function, a value's
backend is the tape:

<!-- code: tape-backend -->
```clojure
(def seen (atom nil))

((array/grad (fn [x]
               (reset! seen (array/backend-name (array/backend x)))
               (* x x)))
 3.0)
;=> 6.0

@seen
;=> :tape
```

The rules form a table keyed by the primitive operations' keywords.
Composite operations, which gen.array writes once from primitives, need
no rules, since the tape differentiates their primitives. The gradient
of `logsumexp` comes out as `softmax`. `lgamma` is a Lanczos series with
a reflection formula below 1/2, and its gradient comes out as `digamma`,
a separate composite, to rounding (an absolute $10^{-6}$ allows for
float32's, where digamma crosses zero):

<!-- code: composites -->
```clojure
(let [v (array/array [1.0 2.0 3.0])]
  (array/allclose? ((array/grad array/logsumexp) v) (array/softmax v)))
;=> true

(def xs-lgamma (mapv #(* 0.1 %) (range 1 51)))

(def lgamma-slopes (mapv (array/grad array/lgamma) xs-lgamma))

(array/allclose? (array/array lgamma-slopes)
                 (array/digamma (array/array xs-lgamma))
                 {:atol 1e-6})
;=> true
```

<!-- code: lgamma-figure -->
```clojure
;; figure: figures/041-lgamma.svg
(svg/plot [{:curves [array/lgamma] :color "gray"}
           {:curves [array/digamma]}
           {:points (map vector xs-lgamma lgamma-slopes) :r 2}]
          {:x-range [0.05 5] :y-range [-6 4]
           :y-label "log-gamma and digamma"})
```

![$\log\Gamma(x)$ (gray), its derivative $\psi(x)$ (blue), and the tape's
gradient of `array/lgamma` at 50 points (dots).](figures/041-lgamma.svg)

Control flow that depends on values works, as in Gen.jl. Comparisons,
`floor` and the other operations whose derivative is zero almost
everywhere give untracked values, which read and branch freely:

<!-- code: branching -->
```clojure
(defn kinked
  "|x|, by a branch on x's sign."
  [x]
  (if (array/truthy? (array/< x 0)) (array/negate x) x))

(mapv (array/grad kinked) [-2.0 3.0])
;=> [-1.0 1.0]
```

### Reading a value being differentiated

A tracked float read on the host would lose its gradient. The number it
gives has no link back to the tape, so the path through it would be
missing from the gradient, without a word. So reading one is a loud
error (DESIGN.md §9, genclj-de4g), and the message names the ways out:

<!-- code: host-read -->
```clojure
;; error: Cannot read a value being differentiated on the host
((array/grad (fn [x] (* x (array/item x)))) 3.0)
```

Host arithmetic on a tracked value, with clojure.core's operators or
`Math/…`, is the same error, since a tracked value is not a `Number`:

<!-- code: host-arithmetic -->
```clojure
;; error: Host arithmetic (clojure.core's operators, Math/…, double)
((array/grad (fn [x] (Math/exp x))) 1.0)
```

`array/stop-gradient` blocks differentiation, so it is the way to read
on purpose. The gradient of $x \cdot x$ at 3 is 6. With one factor
stopped it is 3, the other factor's path alone, whether the stopped
factor stays an array or is read with `array/item`:

<!-- code: stop-gradient -->
```clojure
[((array/grad (fn [x] (* x x))) 3.0)
 ((array/grad (fn [x] (* x (array/stop-gradient x)))) 3.0)
 ((array/grad (fn [x] (* x (array/item (array/stop-gradient x))))) 3.0)]
;=> [6.0 3.0 3.0]
```

Until genclj-de4g, `array/item` returned the primal. *Learning* found a
gradient that came out as the prior's term alone, through such a read
in a model. PyTorch's `.item()` detaches silently; gen.clj's does not.

### Differentiating a model

The handler calls only gen.array, so a model's log density is
differentiable through the GFI: `gfi/assess` under `grad` runs the
handler on tracked values. The model `noisy` above draws $\mu \sim
\mathcal{N}(0, 1)$ and $y \sim \mathcal{N}(\mu, 0.5)$, so the
derivative in $\mu$ of $\log p(\mu, y)$ is $-\mu + (y - \mu)/0.25$:

<!-- code: model-gradient -->
```clojure
(defn joint
  "The log density of noisy's choices, mu and y = 1.2, as a function of
   mu."
  [mu]
  (:weight (gfi/assess noisy [] (cm/choicemap :mu mu :y 1.2) (rng/key 0))))

(array/allclose? ((array/grad joint) 0.5)
                 (+ (- 0.5) (/ (- 1.2 0.5) (* 0.5 0.5))))
;=> true
```

That is what `gen.gradients/choice-gradients` computes, and with it
HMC, MALA, MAP and variational inference (*Gradient-Based MCMC*).

### Custom VJPs

Some code differentiation cannot see through: host arithmetic, a
solver's iterations. Some it should not, where a rule is cheaper or
steadier. `array/custom-vjp` gives a function its gradient:
`(vjp args out g)` returns one cotangent per argument, from the
arguments, the function's value and the cotangent $g$ of the result. The
cube root by `Math/cbrt` is host arithmetic. Given its derivative,
$1/(3x^{2/3})$, it differentiates, also inside a larger function:

<!-- code: custom-vjp -->
```clojure
(def cbrt
  (array/custom-vjp (fn [x] (Math/cbrt (array/item x)))
                    (fn [_ out g] [(/ g (* 3.0 out out))])))

[(cbrt 8.0) ((array/grad cbrt) 8.0) ((array/grad #(* % (cbrt %))) 8.0)]
;=> [2.0 0.08333 2.6667]
```

At 8 the derivative of $x^{4/3}$, $x$ times its cube root, is
$x^{1/3} + x/(3x^{2/3}) = 2 + 8/12$. A rule with the wrong number of cotangents
is a loud error:

<!-- code: custom-vjp-error -->
```clojure
;; error: custom-vjp: the VJP returned
((array/grad (array/custom-vjp (fn [x] (Math/cbrt (array/item x)))
                               (fn [_ _ g] [g g])))
 8.0)
```

gen.custom's `:gradient`, which *Learning* used for Kepler's equation,
is built on `array/custom-vjp`.

### MLX's native gradients

On MLX the tape is the default as well. With `-Dgen.mlx.grad=native`,
`grad` and `value-and-grad` use MLX's own transformation,
`mlx_value_and_grad`, instead (docs/array-api.md §8). MLX then
differentiates the graph the function builds, with its own VJPs, except
where gen.clj gives it the tape's: MLX has no VJP for its Cholesky, LU
or inverse, so each linear-algebra operation is an MLX custom VJP whose
rule is the tape's, one definition for both. Native mode is the way to
nested gradients, such as Hessians, which the tape does not compute yet:

<!-- code: nested -->
```clojure
;; error: Differentiating inside a differentiated function is not supported
((array/grad (fn [x] ((array/grad (fn [y] (* x y y))) x))) 1.0)
```

Native mode is opt-in, after measurement: faster for plain HMC, by
about 25% at a million observations, and for compiled MALA (1.7×),
slower for compiled HMC (25 to 30%), and level elsewhere. The tape stays
the default, as the owner confirmed, because its gradients are the same
rules on every backend, where native ones depend on MLX's VJPs. MLX's
`maximum` gives a tie's gradient to its second argument, where the tape
splits it, and in native mode a host read still detaches silently: mlx-c
does not say which arrays are being differentiated (DESIGN.md §9).

## Compiling on MLX

`array/compile` is the backend's own faster path, which *Speeding Up*
used in HMC and MALA. On MLX it traces a function, once for each
structure, shapes and dtypes of its arguments, into one MLX graph, which
MLX fuses into fewer kernels and replays on later calls. Arguments and
results are arrays or trees of them. On the JVM backend the compiled
function is the function itself. Compile composes with `value-and-grad`,
the case measured below: the value and gradient of a log density, as one
step of a gradient method takes them.

<!-- code: compile -->
```clojure
(def log-density-step
  (array/value-and-grad (fn [{:keys [mu sigma]}] (log-density mu sigma))))

(def compiled-step (array/compile log-density-step))

(let [params {:mu 0.5 :sigma 2.0}
      [v g] (log-density-step params)
      [cv cg] (compiled-step params)]
  [(array/allclose? v cv)
   (array/allclose? (:mu g) (:mu cg))
   (array/allclose? (:sigma g) (:sigma cg))])
;=> [true true true]
```

Measured on Metal (docs/array-api.md §8), the value and gradient of a
log density over 10,000 elements take 0.63 ms plain and 0.26 ms
compiled, and ten leapfrog steps 3.9 and 0.63 ms. Over a million
elements they take 9.8 and 0.97 ms, and 229 and 21 ms, where the JVM
backend takes 256 ms for the leapfrog steps.

A compiled function is replayed, not rerun, and three rules follow. It
may not read a value on the host (`truthy?`, `item`), a loud error on
MLX. A key may not be one of its arguments, since the replay would
replay its draws (*Speeding Up* showed that error). And steps that draw
inside the model, such as MH's regenerate and a particle filter's
extension, are not compiled at all, since their keys are host values.
The JVM backend cannot enforce the first rule, so a function meant for
compile is written as for MLX.

At this point the GenMLX book had *The Compilation Ladder*: five levels
of compiled execution, from shape-based batching through generative
functions compiled from an analysis of the model's source, compiled
inference sweeps and automatic analytical elimination to one fused graph
for a whole fit, chosen site by site by an analyzer. gen.clj has no
ladder, by design. DESIGN.md's reading of GenMLX found about 25,000 of
its lines in optimization paths that implemented the same semantics
again, and most of its core bug fixes there (§2). In their place are
three plug-ins, each held to the handler: the static language and
batched execution, which *Speeding Up* took in turn, and `array/compile`,
which fuses array operations below the GFI and changes no value.

## Why core code never names a backend

Rule 3 of the project's guide, `CLAUDE.md`, reads "Core code calls only
`gen.array`, never a backend", and *Design Rationale* checked it against
the source: the only namespace that requires a backend is `gen.array`.
Here is what the rule buys.

gen.array finds an operation's backend from its array operands, or from
`array/*backend*` when it has none, as for creation functions. It
promotes the dtypes, and it calls one method of the backend's protocol,
with the operation named by a keyword. The protocol has 27 methods, and
one of them, `-elementwise`, covers 46 operations: unary math,
arithmetic, comparisons, logic, `where` and `stop-gradient`. `array/exp`
is that call, after promotion:

<!-- code: surface -->
```clojure
[(count (:sigs p/IBackend)) (count dtype/elementwise-ops)]
;=> [27 46]

(let [x (array/with-backend jvm (array/array [0.0 1.0]))]
  (array/equal? (array/exp x)
                (p/-elementwise (array/backend x) :exp [x] :float64)))
;=> true
```

Operations are data, so a backend is small, the tape's VJP rules are a
table keyed by the same keywords, and the conformance suite iterates
over the operations.

Two of gen.clj's backends are not hardware. The tape is one, a backend
that wraps the real one. The batch tracer, `gen.array.batch`, is the
other: the backend under batched execution. Its values hold one value
per particle, with leading batch axes, and present to the code they flow
through the shape of one particle. `batch/wrap` makes three particles of
a vector of three, and a function written for one value runs on all
three at once:

<!-- code: batched -->
```clojure
(def points (array/with-backend jvm (array/array [0.0 1.0 2.0])))

(def particles (batch/wrap points [3]))

[(array/shape points) (array/shape particles)
 (array/backend-name (array/backend particles))]
;=> [[3] [] :batch]

(defn standard-log-density
  "The log density of x under the standard normal."
  [x]
  (dc/log-prob (d/normal 0 1) x))

(array/allclose? (batch/unwrap (standard-log-density particles) [3])
                 (standard-log-density points))
;=> true
```

The tracer aligns batched and shared values by their per-particle axes,
as JAX's vmap does. A per-particle scalar meeting a shared vector of
three gives each particle that vector plus its own value, although the
batch has three particles too:

<!-- code: per-particle -->
```clojure
(def shared (array/with-backend jvm (array/array [10.0 20.0 30.0])))

(batch/unwrap (+ particles shared) [3])
;=> [[10.0 20.0 30.0] [11.0 21.0 31.0] [12.0 22.0 32.0]]
```

Batching by shape alone, a leading axis of particles and broadcasting,
was GenMLX's way and gen.clj's first. It cannot tell a per-particle
scalar of shape $[N]$ from a shared vector of length $N$, and it added
the two element by element (docs/array-api.md §8b).

Values of two backends never meet in one operation, and no operation
moves a value between backends: read it with `array/->clj` and make the
array again on the other (docs/array-api.md §3). Two batched runs of
different sizes are two batch backends, and their particles cannot meet:

<!-- code: two-batches -->
```clojure
;; error: add: arrays from different backends (:batch and :batch)
(+ particles (batch/wrap (array/with-backend jvm (array/array [1.0 2.0])) [2]))
```

Reading a batched value on the host cannot give one answer for every
particle. That is the batched-branch error, which makes a batched run
fall back to one particle at a time, as *Speeding Up* showed:

<!-- code: batched-read -->
```clojure
;; error: Cannot read on the host a batched value
(array/item particles)
```

The wrapping backends compose. A gradient under the tracer is vmap's:
each particle gets the gradient of its own scalar run, by one backward
pass (docs/array-api.md §8b). Inside it, a value belongs to the tape, which wraps
the tracer, which wraps the JVM backend:

<!-- code: wrapped -->
```clojure
(defn wrapped
  "The names of backend b and of the backends it wraps, outermost first."
  [b]
  (cons (array/backend-name b)
        (when (instance? gen.array.protocols.IWrappingBackend b)
          (wrapped (p/-inner b)))))

(def chain-seen (atom nil))

(def slopes
  ((array/grad (fn [x]
                 (reset! chain-seen (wrapped (array/backend x)))
                 (standard-log-density x)))
   particles))

[(batch/unwrap slopes [3]) @chain-seen]
;=> [[-0.0 -1.0 -2.0] (:tape :batch :jvm)]
```

So a model or a distribution that named a backend, by computing with
the JVM backend's arrays or with clojure.core's arithmetic on its
values, would lose MLX, differentiation and batching at once: each is a
backend the same code runs on. Written against gen.array, the one
`dc/log-prob` above gives a number on the JVM backend, a lazy array on
MLX, a tracked value under `grad`, a batch under the tracer, and the
batch's gradients under both.

The next chapter, *Exact Structure, Made Explicit*, turns from the layer
under the GFI to structure over it: models whose exact answers gen.clj
computes, when the model says so.

## Workshop

Each exercise has a tested solution at the end of the chapter.

1. **Predict the dtype.** On the JVM backend, predict the dtypes of an
   int64 array plus a float32 array, of an int32 array times 2.5, of the
   sum of a comparison's bools, and of an int64 array divided by itself.
   Which differs from NumPy?
2. **A Gumbel by hand.** `rng/gumbel` draws $-\log(-\log u)$ for $u$
   uniform on $[t, 1)$, $t$ the least normal float, as JAX's does. Write
   it from `rng/uniform`. Check that it draws `rng/gumbel`'s values bit
   for bit, and that the mean of 10,000 draws is Euler's constant
   $\gamma \approx 0.5772$ within five standard errors (the standard
   Gumbel's standard deviation is $\pi/\sqrt 6$). Does it agree with
   JAX's?
3. **Which gradients survive?** Predict the tape's gradient at 2.5 of
   $x\lfloor x\rfloor$, of `(array/where (array/> x 0) (* x x) 0.0)`,
   of $x$ times the host number of $\lfloor x\rfloor$, and of
   `(Math/sqrt x)`. Then run them.
4. **Straight-through rounding.** `array/round`'s derivative is zero
   almost everywhere, so the tape gives it no gradient. Give it the
   identity's gradient with `array/custom-vjp`, the straight-through
   estimator of quantized networks, and compare the gradients of
   $\operatorname{round}(x)^2$ at 1.3 with and without it.
5. **A shared parameter, batched.** Under the tracer, take the gradient
   in $\mu$ of a reading's log density, $\log\mathcal{N}(x; \mu, 2)$,
   for four readings at once, with $\mu = 1$ shared by all four. Predict
   what it holds, check it against four scalar gradients, and check that
   its sum is the gradient of the four readings' total log density.

## Solutions

**1.** An integer array and a float32 one meet at float32, JAX's rule,
where NumPy gives float64. A weak float raises an int32 array to the
default float, the bools of a comparison sum to the default int, and
`/` always divides as floats:

<!-- code: solution-1 -->
```clojure
(array/with-backend jvm
  (let [i64 (array/array [1 2])
        i32 (array/array [1 2] {:dtype :int32})
        f32 (array/array [1.0 2.0] {:dtype :float32})]
    (mapv array/dtype [(+ i64 f32) (* i32 2.5) (array/sum (array/< i64 2))
                       (/ i64 i64)])))
;=> [:float32 :float64 :int64 :float64]
```

**2.** The same operations on the same uniform give the same bits. The
mean lands within five standard errors of $\gamma$, and the draws agree
with JAX's to rounding, as the normals did:

<!-- code: solution-2 -->
```clojure
(defn gumbel
  "Standard Gumbel values of shape from key: -log(-log u), u uniform on
   [tiny, 1), tiny the least normal float64."
  [key shape]
  (let [u (rng/uniform key shape {:low Double/MIN_NORMAL :high 1.0})]
    (array/negate (array/log (array/negate (array/log u))))))

(def euler-gamma 0.5772156649015329)

(array/with-backend jvm
  (let [key (rng/key 9)
        g (gumbel key [10000])
        sd (/ Math/PI (Math/sqrt 6.0))]
    [(array/equal? g (rng/gumbel key [10000]))
     (< (abs (- (array/item (array/mean g)) euler-gamma))
        (* 5 (/ sd (Math/sqrt 10000))))]))
;=> [true true]

(array/allclose? (gumbel k [6]) (array/array (:out (:gumbel-f64 jax))))
;=> true ; float64
```

**3.** `floor` gives an untracked value, a constant to the tape, so the
first gradient is $\lfloor 2.5\rfloor = 2$. `where` passes the cotangent
to the branch it takes, $2x = 5$. Reading $\lfloor x\rfloor$ on the host
is allowed, since it has no gradient to lose, and gives 2 again:

<!-- code: solution-3 -->
```clojure
[((array/grad (fn [x] (* x (array/floor x)))) 2.5)
 ((array/grad (fn [x] (array/where (array/> x 0) (* x x) 0.0))) 2.5)
 ((array/grad (fn [x] (* x (array/item (array/floor x))))) 2.5)]
;=> [2.0 5.0 2.0]
```

`Math/sqrt` is host arithmetic on a tracked value, the loud error:

<!-- code: solution-3-error -->
```clojure
;; error: Host arithmetic
((array/grad (fn [x] (Math/sqrt x))) 2.5)
```

**4.** Without the rule the result does not depend on $x$ as far as the
tape can see, so the gradient is 0. With it, the gradient of
$\operatorname{round}(x)^2$ is $2\operatorname{round}(x) = 2$:

<!-- code: solution-4 -->
```clojure
(def straight-through-round
  (array/custom-vjp array/round (fn [_ _ g] [g])))

[((array/grad #(array/square (array/round %))) 1.3)
 ((array/grad #(array/square (straight-through-round %))) 1.3)]
;=> [0.0 2.0]
```

**5.** Under the tracer a shared argument's gradient is per particle, as
vmap's: four values, $(x_i - \mu)/4$. They are the four scalar
gradients, and their sum is the gradient of the total, which is what a
shared parameter's learning needs:

<!-- code: solution-5 -->
```clojure
(def readings (array/with-backend jvm (array/array [-1.0 0.5 2.0 3.5])))

(defn reading-density
  "The log density of a reading x under a normal of mean mu and sd 2."
  [mu x]
  (dc/log-prob (d/normal mu 2.0) x))

(def per-reading
  (array/with-backend jvm
    (let [xs (batch/wrap readings [4])
          slopes (batch/batched [4]
                   ((array/grad reading-density) 1.0 xs))]
      (batch/unwrap slopes [4]))))

per-reading
;=> [-0.5 -0.125 0.25 0.625]

(array/with-backend jvm
  (let [one-at-a-time (mapv #((array/grad reading-density) 1.0 %)
                            (array/->clj readings))
        total (array/grad #(array/sum (reading-density % readings)))]
    [(array/allclose? per-reading (array/array one-at-a-time))
     (array/allclose? (array/sum per-reading) (total 1.0))]))
;=> [true true]
```
