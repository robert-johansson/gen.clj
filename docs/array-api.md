# The gen.clj array API

*Status: approved by the owner 2026-10-18 (bean genclj-hcbs). This specifies the layer every
other part of gen.clj is written against (DESIGN.md §4, L0). The backend
contract is real code: `src/gen/array/protocols.clj`. Everything else here is
specification, implemented by the JVM backend in M1 (bean genclj-a0qg).*

## 1. Goals

1. **One API for all core code.** GFI, distributions, combinators and inference
   call only `gen.array`, never a backend. The same model runs on the JVM
   backend (M1) and on MLX (M2).
2. **Values, not objects.** Arrays are immutable. No operation's meaning depends
   on laziness, devices or memory management.
3. **Scalar models at near-primitive speed.** Most probabilistic programs are
   scalar-heavy. On the JVM backend they should cost nanoseconds, not the
   microseconds GenMLX paid per MLX op.
4. **Shape-based batching for free.** NumPy broadcasting everywhere, so a model
   written for scalars runs on `[N]`-shaped particles unchanged.
5. **Differentiable.** `grad` over arbitrary trees of arrays.
6. **Purely functional randomness.** Keys are values; same key, same result.
7. **A small backend surface.** Ops are data (keywords), so a backend is one
   protocol implementation, and a new backend costs days, not months.

## 2. Namespaces

| Namespace | Alias | Who uses it | Contents |
|---|---|---|---|
| `gen.array` | referred (see below), or `a` | everyone | values, creation, math, reductions, shape, indexing, eval, `grad` |
| `gen.array.linalg` | `la` | distributions, inference | cholesky, triangular solve, solve, inv, slogdet |
| `gen.random` | `rng` | handler, distributions | keys (pure Clojure) and samplers built on one backend primitive |
| `gen.array.protocols` | `p` | backend implementers | `IArray`, `IBackend` |
| `gen.array.tape` | — | the JVM backend | generic reverse-mode AD over the public API |
| `gen.backend.jvm` | — | — | M1 backend |
| `gen.backend.mlx` | — | — | M2 backend |

**House style: refer the math.** gen.array's operators extend Clojure's to
arrays, and on numbers they give Clojure's results: `/` never returns a
`Ratio` (it divides as floats), and integer overflow wraps instead of throwing.
So a namespace refers them in place of `clojure.core`'s, the convention
`clojure.core.matrix.operators` established, and math reads as plain Clojure:

```clojure
(ns my.models
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :refer [+ - * / < > <= >= == exp log sqrt square sum]]
            [gen.dist :refer [normal]]))

(def linreg
  (gen [xs]
    (let [slope     (trace :slope (normal 0 10))
          intercept (trace :intercept (normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (normal (+ (* slope x) intercept) 1)))
      slope)))
```

On the JVM backend `slope` is a plain `Double`; batched, it is an `[N]` array;
on MLX, an MLX array. The code is the same in all three cases.

A distribution body, compared with GenMLX:

```clojure
;; GenMLX
(mx/negative (mx/add LOG-2PI-HALF (mx/log sigma)
                     (mx/multiply HALF (mx/square (mx/divide (mx/subtract v mu) sigma)))))
;; gen.clj
(- (+ HALF-LOG-2PI (log std) (* 0.5 (square (/ (- v mu) std)))))
```

- **Safe to refer in place of `clojure.core`:** `+ - * / < > <= >= == abs mod`.
  On numbers they mean what Clojure means.
- **Never refer:**
  - `max` and `min`: in gen.array they are reductions (`(max x {:axis 0})`); use
    `maximum`/`minimum` elementwise.
  - `take` and `flatten`: they gather and reshape.

  Call these qualified (`a/max`, `a/take`), so `(max a b)` keeps meaning what it
  means in Clojure.
- Names with no `clojure.core` counterpart (`exp`, `log`, `sum`, `where`,
  `reshape`, …) can be referred freely.
- Code that needs ordinary integer arithmetic in a hot loop (an index counter)
  uses `inc`, or `clojure.core/+` explicitly.

## 3. Values

- **An array is one of:**
  - a Clojure number: `Long` is int64 and `Double` is float64;
  - a `Boolean` (bool);
  - a backend array (anything implementing `IArray`).

  Numbers and booleans are **0-d arrays that belong to no backend.**
- **Host math.** An op whose operands are all numbers or booleans runs on the
  host and returns a number or boolean. There is no backend involvement and no
  allocation (3.8 ns per `add`, measured, §11).
  - Host math follows the API's semantics, not `clojure.core`'s:
    `(a// 1 2)` is `0.5`, never a `Ratio`.
- **Weak typing for numbers (the JAX rule).** When a number meets a backend
  array, it is coerced to that array's dtype and never promotes it:
  - float32 array + `2.5` → float32;
  - int32 array + `2.5` → the backend's default float dtype.
- **One backend per op.** Mixing arrays from two backends throws. (Explicit
  transfer, `(a/to-backend x b)`, comes with M2.)
- **No `=` on arrays.** Compare with `a/equal?` (exact) or `a/allclose?`.
  Numbers still compare with `=`, but core code doesn't rely on that.
- **Trees.** Nested maps and vectors whose leaves are arrays are first-class
  inputs to `eval!`, `grad` and `tree-map`. Traces and choicemaps are trees in
  this sense.

## 4. Backends

- **Ops on arrays** find their backend from their operands (§3). A tracked value
  (§8) takes precedence, so differentiation composes with any backend.
- **Ops without array operands** use `a/*backend*`. These are the creation
  functions (`array`, `zeros`, `ones`, `full`, `arange`, `eye`, `linspace`) and
  samplers given only a shape.
  - Its root value is the JVM backend. Use `(a/with-backend b ...)` locally, or
    set it once at application start.
  - It's a dynamic var, consulted only by creation functions. It is
    configuration, not state that computation depends on.
- **`*-like` variants** (`zeros-like`, `full-like`, …) create on the backend
  and dtype of an existing array.

## 5. Dtypes

- **Keywords:** `:bool :int32 :int64 :uint32 :float16 :bfloat16 :float32 :float64`.
  Each backend declares its defaults:
  - JVM: `{:float :float64 :int :int64}`;
  - MLX: `{:float :float32 :int :int32}`.

  Core code never names a dtype unless it must (e.g. indices are ints).
- **Promotion:** NumPy's table, plus weak numbers. It is implemented once, in
  `gen.array`, so every backend promotes identically. Backends only see operands
  that already share a dtype.
- **Comparisons return `:bool`,** and so do `bernoulli` samples. That matches
  Gen.jl's `Bool`. GenMLX returned 0/1 floats, which `where`-based code then had
  to compare against `ONE`. Bool arrays work in `a/where` and promote to int or
  float in arithmetic, exactly as in NumPy.
- **JVM storage:** every dtype is stored in a `double[]`. int64 values are exact
  up to 2^53, which is ample for counts and indices; bool is 0/1. One storage
  type keeps the backend small. The dtype tag carries the semantics.

## 6. Op surface v1

**Primitive** ops are what a backend implements, via the `IBackend` method
named. **Composite** ops are written once in `gen.array` in terms of
primitives, and are therefore automatically differentiable and identical
across backends. A backend may list composites in `-native-ops` to supply a
faster or more accurate version. The table is the v1 scope: what the M1 core
needs, taken from GenMLX's measured call counts
(docs/research/genmlx-porting-surface.md §(d)).

| Group | Public functions (`a/…`) | Primitive op keywords → `IBackend` method | Composite |
|---|---|---|---|
| creation | `array` `scalar` `zeros` `ones` `full` `eye` `arange` `linspace` `*-like` | `-from-host` `-full` `-arange` | `eye`, `linspace` |
| conversion | `astype` `item` `->clj` `eval!` `array?` `shape` `ndim` `size` `dtype` | `-astype` `-to-host` `-eval!` `-shape` `-dtype` | `ndim`, `size` |
| arithmetic | `+ - * /` (variadic), `negate`, `pow`, `maximum` `minimum`, `mod` `floor-div`, `abs` `sign` | `:add :subtract :multiply :divide :power :maximum :minimum :remainder :floor-divide :negative :abs :sign` | — |
| exp/log | `exp` `log` `expm1` `log1p` `sqrt` `square` `logaddexp` `xlogy` | `:exp :log :expm1 :log1p :sqrt` | `square`, `logaddexp`, `xlogy` |
| trig | `sin` `cos` `tan` `tanh` `sigmoid` `softplus` | `:sin :cos :tan :tanh` | `sigmoid`, `softplus` |
| rounding | `floor` `ceil` `round` `clip` | `:floor :ceil :round` | `clip` |
| special | `erf` `erfinv` `lgamma` `digamma` `log-beta` | `:erf :erfinv` (native in MLX; closed-form derivatives) | `lgamma`, `digamma`, `log-beta` |
| predicates | `isnan` `isinf` `isfinite` | `:isnan :isinf` | `isfinite` |
| comparison | `< <= > >= == !=` | `:less :less-equal :greater :greater-equal :equal :not-equal` | — |
| logic | `logical-and` `logical-or` `logical-not` `where` | `:logical-and :logical-or :logical-not :where` | — |
| reductions | `sum` `prod` `max` `min` `argmax` `argmin` `all` `any` `mean` `var` `std` `logsumexp` | `-reduce` with `:sum :prod :max :min :argmax :argmin :all :any` | `mean`, `var`, `std`, `logsumexp` |
| normalization | `softmax` `log-softmax` | — | both |
| scans | `cumsum` `cumprod` `logcumsumexp` | `-scan` with `:cumsum :cumprod :cummax :cummin` | `logcumsumexp` |
| shape | `reshape` `transpose` `squeeze` `expand-dims` `broadcast-to` `stack` `concatenate` `flatten` `tile` | `-reshape` `-transpose` `-broadcast-to` `-concatenate` | `squeeze`, `expand-dims`, `stack`, `flatten`, `tile` |
| indexing | `slice` `take` `take-along-axis` `pick` `scatter-add` | `-slice` `-take` `-take-along-axis` `-scatter-add` | `pick` |
| linear algebra | `matmul` `outer` `diag` · `la/cholesky` `la/solve-triangular` `la/solve` `la/inv` `la/slogdet` | `-matmul` `-linalg` | `outer`, `diag` |
| gradients | `grad` `value-and-grad` `stop-gradient` | `-value-and-grad`, `:stop-gradient` | — |
| trees | `tree-map` `tree-leaves` | — | both |

**Naming conventions:**
- Arithmetic uses Clojure's symbols (`+`, `<`, `==`, `!=`), referred per §2.
- Following NumPy and MLX, `a/maximum` is elementwise and `a/max` is a
  reduction.
- Reductions take an options map: `(a/sum x {:axis -1 :keep-dims? true})`.
- **`a/pick`** is the one new op. It indexes the last axis of `x` by an int
  array, broadcasting batch dimensions:
  - `(a/pick log-probs v)` *is* a categorical log-prob;
  - it replaces the three shape branches GenMLX's `categorical` needed
    (dist.cljs:590-660; bug genmlx-ql6a).

**Deliberately absent from v1:**
- the LLM ops (sdpa, rope, rms-norm, quantized matmul): M3;
- sort, top-k and general scatter: added when a ported module needs them
  (`scatter-add` along an axis, the transpose of `take-along-axis`, came with
  autodiff: §12.8);
- `vmap` and `compile`: §8;
- devices and streams: backend options in M2.

## 7. Randomness (`gen.random`)

- **The scheme is JAX's default ("partitionable") Threefry** (decided
  2026-10-18), verified bit-for-bit against JAX 0.9.1 and Random123's
  known-answer vectors:
  - `split(k, n)[i] = threefry(k, [hi32(i) lo32(i)])` and `fold-in(k, d) = threefry(k, [0 d])`.
  - Element i of any draw depends only on the key and i.
  - MLX uses the older, non-partitionable layout, so the MLX backend will
    implement this scheme itself; it is only uint32 arithmetic.
- **Keys are backend-neutral values.**
  - A key is an immutable pair of uint32 words (a small deftype), not an array.
  - `rng/key`, `rng/split` and `(rng/split k n)` are pure Clojure Threefry2x32,
    as used by MLX and JAX. They cost nanoseconds, never touch a backend, and
    make "same key, same trace" hold across backends.
- **One backend primitive: `-random-uniform`** (key → uniform [0,1) array of a
  shape and dtype). Every sampler is a composite on top of it, written once:
  - normal: `√2·erfinv(2u−1)`, as MLX does;
  - gumbel: `−log(−log u)`;
  - categorical: argmax(logits + gumbel);
  - bernoulli: `u < p`, giving bool;
  - exponential: `−log1p(−u)`;
  - gamma: Marsaglia–Tsang with a fixed number of vectorized proposals. That
    keeps it branch-free, so there's no eval inside a sampler (a GenMLX smell,
    dist.cljs:410-445);
  - beta and dirichlet: via gamma.
  - Because the algorithms are shared, the backends sample from identical laws.
- **Stretch goal (bean genclj-ldpy):** make the JVM backend's float32 mode
  bit-identical to MLX's `random.uniform`. The two backends would then produce
  identical traces for the same key, which makes an extremely strong
  cross-backend test.
- **Shapes broadcast with parameters.** Batched sampling is just a larger shape.
- **Keys never live in metadata, and nothing silently invents one.** The GFI's
  optional key (DESIGN.md §3.5) is the only implicit source. It is a documented,
  seedable default at the outermost API.

## 8. Differentiation

- **API.**
  - `(a/grad f)` differentiates with respect to argument 0;
    `(a/grad f {:argnums [0 2]})` with respect to several arguments.
  - `(a/value-and-grad f opts)` returns `[value grads]`.
  - Arguments may be trees; each gradient has the same structure as its
    argument.
  - `a/stop-gradient` blocks differentiation.
- **JVM backend: `gen.array.tape`, a generic reverse-mode AD.**
  - `grad` wraps the inputs as *tracked* values. A tracked value is immutable:
    it carries its primal value, its parents and a VJP closure. There is no
    global tape, so it is purely functional.
  - Ops on tracked values compute the primal through the normal API and record
    the VJP. The backward pass is a topological sweep with a local transient
    accumulator.
  - VJP rules form a table keyed by the primitive op keywords of §6, and are
    written against the public API. Composites therefore need no rules, and the
    same tape works on any backend; it could even test MLX's native AD.
  - `a/item` on a tracked value returns its primal. Data-dependent control flow
    is fine (non-differentiable branches, as in Gen.jl).
  - Scope for M1 is first order. Nested AD (Hessians) is bean genclj-hors.
- **MLX backend:** `-value-and-grad` maps to `mlx_value_and_grad` (M2).
- **Reserved names:** `vjp`, `jvp`, `custom-vjp`, `vmap`, `compile`.
  - `compile` will be the identity on the JVM backend, and `mlx_compile` on MLX
    (M2).
  - `vmap` is not needed by the core: batching is shape-based, as in GenMLX.

## 8a. Branching on values

Clojure's `if` treats every non-nil, non-false value as true. A comparison of
arrays returns an array: a bool array when batched, an MLX 0-d array on MLX. So
`(if (< x 0) …)` would silently take the first branch there for every particle.
Two layers make this safe:

1. **`truthy?`** gives a value's Clojure truthiness, the same on every
   backend:
   - A boolean, host or a one-element bool array, counts by its value.
   - Any number, host or a one-element numeric array, is true, as in Clojure.
   - A value that differs across the particles of a batch throws `ex-info`
     with `{:gen.array/batched-branch true}`: one branch cannot serve every
     particle. Batched code uses `where` (values) or a switch combinator
     (structure) instead. *Revised 2026-10-18:* numbers were falsy at 0,
     which broke `(or 0 5)` and `(when-let [i 0] …)` inside `gen`.
2. **Inside `gen` bodies, conditionals branch on values** (bean genclj-pa7x).
   - The `gen` macro macroexpands the body and wraps every `if` test in
     `truthy?`. That also covers `when`, `cond`, `and`, `or` and `if-let`, which
     expand to `if`.
   - It also substitutes guarded versions (`gen.dynamic.core`) for the
     clojure.core functions that test truthiness or look up by key with an
     `if` of their own: `not` (so `if-not`), `boolean`, `true?`, `false?`,
     the type predicates, `every?`, `some`, `filter`, `remove`, `take-while`,
     `complement`, `get`, `contains?`, `find`, `get-in` and the like. They
     behave as clojure.core's on ordinary values and raise the batched-branch
     error on a batched one. *Added 2026-10-19 after the second acceptance
     test:* `(if-not b …)` and `(filter pred xs)` in a body took one branch
     for every particle.
   - Non-array tests behave exactly as in Clojure.
   - A model written with plain Clojure conditionals is correct on every
     backend in scalar mode.
   - In batched mode, branching on a batched value raises the batched-branch
     error. So does any host predicate on one (`=`, `case`, map keys, which
     compare or hash it; see §8b). The error tells the batched executor to fall
     back to running particles one at a time. That is correct, just slower.
     Batching is an optimization, and the handler is ground truth.
   - Outside `gen` bodies (helper functions a model calls), a raw `if` or
     `case` on an array is not guarded: branch with `a/truthy?` or `where`
     there. What still catches it (§8b):
     - `=`, `not=` and `condp =` reach a batched value anywhere, through `=`.
     - A batched comparison that nothing reads, the trace a raw `if` on it
       leaves, raises the fallback when the run ends.
     - Spot checks against the scalar handler catch the rest (a `case` on a
       choice, or a vector lookup keyed by one) with high probability, not
       certainty.
   - A bool array of several elements, unbatched, is ambiguous (as in NumPy):
     `truthy?` throws, without the batched-branch flag, and names `all`,
     `any` and `where`. A numeric array is true, as any Clojure value.
     *Revised 2026-10-19 after the acceptance test:* it raised the
     batched-branch error, a leftover of shape-based batching.

## 8b. Batching by tracing (vmap semantics)

*Decided 2026-10-18 by the agent, for the owner's review; it replaces
shape-based batching after the M1 review (docs/research/review-2026-10-18.md).*

Shape-based batching (prefix a batch axis, let broadcasting do the rest) is
ambiguous: it cannot tell a per-particle scalar of shape [N] from a shared
vector of length N. An [N] parameter therefore met a [3] observation element
by element. `gen.array.batch` is a batch tracer instead, a wrapping backend
like the AD tape:
- **Batched values.** A batched value wraps an array whose leading axes are
  the batch. To the code it flows through, it has its per-particle shape
  (`a/shape`), and every op acts per particle: batched and unbatched operands
  align by their per-particle axes, as with JAX's `vmap`.
- **Draws.** Under the batch backend, draws gain the batch axes, one
  independent draw per particle, even when every parameter is unbatched
  (`a/effective-backend`).
- **Shared values.** Creation functions and constants stay unbatched, shared
  by every particle.
- **Reading on the host.** Reading a batched value on the host (`a/item`,
  `a/->clj`, `truthy?` of a boolean, `=`, hashing) raises the batched-branch
  error. So does host math on one, a `ClassCastException` naming `Batched`.

Running a generative function's scalar code under the batch backend runs it
batched (`gfi/run-op` with `:batch-shape`). Every generative function
therefore has a batched form. It has the same meaning for code inside `gen`
bodies, which the macro guards (§8a). Any batched run that fails falls back
to one particle at a time, and `gen.batched/fallback-reason` says why. For
one key, a batched run draws other samples than a scalar run (the draws are
made per site for all particles at once), with the same distribution.

Code outside `gen` bodies is not guarded. Two mechanisms cover it
(*added 2026-10-19 after the acceptance tests*):
- **Unread comparisons.** Every batched bool an op makes is registered, and
  every op marks the values it reads. A raw `if` reads nothing (Clojure's
  `if` tests the object, which is true). So a batched bool that is unread
  when the run ends, and that the run does not hand back as a choice or
  return value, is the trace of such a branch. It raises the fallback. This
  catches the common case deterministically: a helper's
  `(if (< x c) …)`, however rarely the other branch is taken.
- **Spot checks.** After a batched simulate or generate, 8 evenly spaced
  particles (`*spot-checks*`: a number, `:all` or 0) are re-run through the
  scalar handler from their choices. So are the 1st, 2nd, 4th, 8th, 16th, …
  update or regenerate of a collection (`*verified-updates*`). A score or
  array return value that differs from the batched one raises the
  fallback. One found after unchecked updates is a loud error instead, since
  their results may be wrong too.
  - Each check is a scalar run, so the doubling schedule keeps a long
    particle filter linear, at about two checks of the final trace.
  - Spot checks are probabilistic. A branch that few particles take can
    escape 8 checks: a `case` on a choice, or a vector lookup keyed by one,
    in a helper.
  - Binding `*spot-checks*` to `:all` checks every particle, at the cost of a
    scalar run each, to validate a model's batched form.
*Hardened 2026-10-19 after a third review* (docs/research/review-2026-10-19.md):
- `=` reaches the batch in either order;
- validation reads every particle;
- linear algebra aligns per-particle ranks;
- resampling gathers lazily rather than re-running the model.

Measured: 5–90× faster importance sampling on the JVM backend, spot checks
included (`docs/benchmarks.md`).

## 9. Evaluation and memory

- `(a/eval! tree)` returns its argument and is a no-op on eager backends.
- `(a/item x)` returns a 0-d array's host number or boolean.
- `(a/->clj x)` returns nested vectors.
- Core code calls `eval!` only at inference-loop boundaries (e.g. once per MCMC
  iteration). Never inside handlers, transitions or distributions, which is the
  GenMLX smell this rule avoids.
- The API exposes **no memory management.** For MLX (M2), arrays are freed by a
  `Cleaner` whose frees are queued to the owning thread and drained at `eval!`
  boundaries. Native-memory pressure triggers collection. If that proves
  insufficient, a scoping construct is added to the MLX backend, not to the
  core API. The open design work is in bean genclj-gip5.

## 10. Errors

- Shape mismatches, mixed backends, unsupported dtypes and out-of-range axes
  throw `ex-info` with `{:op … :shapes … :dtypes …}`.
- Nothing produces `NaN` from a shape error. NaNs arise only from the
  mathematics (e.g. `log` of a negative), as in NumPy.

## 11. Performance

**Measured on the implemented JVM backend** (M4, Oracle GraalVM 25, Graal JIT,
criterium `quick-benchmark`; reproduce with `clojure -M:bench -m
gen.array-bench`):

| Operation | Time |
|---|---|
| raw `clojure.core/+` on boxed doubles (baseline) | 2.5 ns |
| `+` on two numbers (host path) | 3.8 ns |
| `exp` on a number | 12.5 ns |
| `+` on two `[1]` arrays | ~55–65 ns |
| `+` `[100]` + number | 109 ns |
| `+` `[100]` + `[100]` | 104 ns |
| `+` `[10000]` + `[10000]` | 4.7 µs (≈0.5 ns per element, memory bound) |
| `exp` `[10000]` | 15.5 µs |
| `sum` `[10000]` | 5.2 µs |
| `logsumexp` `[10000]` | 31 µs |
| `+` `[100 100]` + `[100]` (broadcast) | 7.7 µs |
| `matmul` `[100 100]` · `[100 100]` | 0.40 ms (naive loop) |

How to read these numbers:
- **Scalar models pay host-math prices**, a few nanoseconds per op. GenMLX
  measured about 5 µs per MLX op plus a ~67 µs eval floor. The MLX backend earns
  its place on large batched and LLM workloads.
- **Small arrays cost about 50–100 ns of fixed dispatch per op.** That is
  amortized across particles in batched execution.
- **The earlier dispatch prototype measured Graal's JIT vectorizing elementwise
  loops about 2× better than C2,** supporting Graal as the default.

## 12. Decisions (approved by the owner 2026-10-18)

1. **Operator names:** Clojure's symbols (`+ - * / < <= > >= == !=`), with
   `maximum` (elementwise) vs `max` (reduction). *Revised 2026-10-18:* the
   house style refers the operators instead of writing `a/+` (§2).
2. **Numbers and booleans are 0-d arrays,** with number-only ops computed on the
   host and returning numbers (§3). On the JVM backend this makes trace values
   plain `Double`s, so ordinary Clojure control flow works on them.
3. **Bool dtype** for comparisons and `bernoulli` (Gen.jl's `Bool`), instead of
   GenMLX's 0/1 floats.
4. **Keys are backend-neutral values** (pure Clojure Threefry), and
   `-random-uniform` is the single backend random primitive (§7).
5. **Composite ops are defined once over primitives,** with optional native
   overrides (§6). This includes the special functions.
6. **Generic tape AD written against the public API** (§8), instead of a
   JVM-specific AD.
7. **JVM storage is `double[]` for all dtypes** (int64 exact to 2^53) (§5).
8. **`scatter-add` along an axis is a primitive** (added 2026-10-18 with the
   tape, for the owner's review). The VJPs of `slice`, `take` and
   `take-along-axis` are scatter-adds, and no composite of the v1 primitives
   computes them in linear time. It is the exact transpose of
   `take-along-axis` and maps onto MLX's `scatter_add` in M2. Tape backends
   are found by a marker protocol, `IWrappingBackend`: an op mixing tracked
   values with the wrapped backend's arrays runs on the tape.
9. **Batching by tracing (vmap semantics)**, §8b (agent, for the owner's
   review), replacing shape-based batching.
10. **`truthy?` follows Clojure for numbers** (any number is true), §8a. A
    bool array of several elements is ambiguous, an error (2026-10-19).
11. **Batched runs are spot-checked against the scalar handler** (§8b;
    2026-10-19, agent, for the owner's review): a few scalar runs per batched
    run, on a doubling schedule for updates. A mismatch after unchecked
    updates is a loud error.
12. **`gen` bodies substitute guarded clojure.core functions** (§8a;
    2026-10-19, agent, for the owner's review): `gen.dynamic.core`'s `not`,
    `filter`, `get`, … in place of clojure.core's, inside bodies only.
13. **An unread batched comparison raises the fallback** (§8b; 2026-10-19,
    agent, for the owner's review): the tracer marks the values ops read,
    an implementation detail of the tracer that never changes a value.
