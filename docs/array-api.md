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
| `gen.array` | referred (see below), or `array` | everyone | values, creation, math, reductions, shape, indexing, eval, `grad` |
| `gen.array.linalg` | `la` | distributions, inference | cholesky, triangular solve, solve, inv, slogdet |
| `gen.random` | `rng` | handler, distributions | keys (pure Clojure) and samplers built on one backend primitive |
| `gen.array.protocols` | `p` | backend implementers | `IArray`, `IBackend` |
| `gen.array.tape` | — | the JVM backend | generic reverse-mode AD over the public API |
| `gen.backend.jvm` | — | — | M1 backend |
| `gen.backend.mlx` | — | — | M2 backend |

**House style: refer the math, alias the rest `array`.** gen.array's operators extend Clojure's to
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

  Call these qualified (`array/max`, `array/take`), so `(max a b)` keeps
  meaning what it means in Clojure.
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
    `(array// 1 2)` is `0.5`, never a `Ratio`.
- **Weak typing for numbers (the JAX rule).** When a number meets a backend
  array, it is coerced to that array's dtype and never promotes it:
  - float32 array + `2.5` → float32;
  - int32 array + `2.5` → the backend's default float dtype.
- **One backend per op.** Mixing arrays from two backends throws. (Explicit
  transfer, `(array/to-backend x b)`, comes with M2.)
- **No `=` on arrays.** Compare with `array/equal?` (exact) or `array/allclose?`.
  Numbers still compare with `=`, but core code doesn't rely on that.
- **Trees.** Nested maps and vectors whose leaves are arrays are first-class
  inputs to `eval!`, `grad` and `tree-map`. Traces and choicemaps are trees in
  this sense.

## 4. Backends

- **Ops on arrays** find their backend from their operands (§3). A tracked value
  (§8) takes precedence, so differentiation composes with any backend.
- **Ops without array operands** use `array/*backend*`. These are the creation
  functions (`array`, `zeros`, `ones`, `full`, `arange`, `eye`, `linspace`) and
  samplers given only a shape.
  - Its root value is the JVM backend. Use `(array/with-backend b ...)` locally, or
    set it once at application start.
  - It's a dynamic var, consulted only by creation functions. It is
    configuration, not state that computation depends on.
- **`*-like` variants** (`zeros-like`, `full-like`, …) create on the backend
  and dtype of an existing array.
- **A fill value** (`full`, `full-like`) is a scalar: a host one, or a 0-d
  array. An array fill value is an operand: a tracked one makes the fill on
  the tape, so it has a gradient (the cotangent's sum). An array of values
  is an error; `broadcast-to` fills with one.

## 5. Dtypes

- **Keywords:** `:bool :int32 :int64 :uint32 :float16 :bfloat16 :float32 :float64`.
  Each backend declares its defaults, which `(array/defaults b)` gives:
  - JVM: `{:float :float64 :int :int64}`;
  - MLX: `{:float :float32 :int :int32}`.

  Core code never names a dtype unless it must (e.g. indices are ints).
- **Promotion:** NumPy's table, plus weak numbers, except for booleans in
  arithmetic (below). It is implemented once, in `gen.array`, so every backend
  promotes identically. Backends only see operands that already share a dtype.
- **Comparisons return `:bool`,** and so do `bernoulli` samples. That matches
  Gen.jl's `Bool`. GenMLX returned 0/1 floats, which `where`-based code then had
  to compare against `ONE`. Bool arrays work in `array/where` and promote to int or
  float in arithmetic as in Julia (DESIGN.md §9): `(+ true true)` is 2 and
  `(negate true)` is -1, where NumPy's `+` and `*` on two booleans are logical
  or and and, and its `-` an error. Mixed with numbers, and in reductions,
  the two agree.
- **Casts to integers are Java's** (DESIGN.md §9): a float truncates toward
  zero, saturates at the dtype's range (the infinities too) and takes NaN to
  0; an integer narrowed wraps. Every backend gives the same, MLX by emulation
  (its own cast is C++'s, undefined out of range).
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

| Group | Public functions (`array/…`) | Primitive op keywords → `IBackend` method | Composite |
|---|---|---|---|
| creation | `array` `scalar` `zeros` `ones` `full` `eye` `arange` `linspace` `*-like` | `-from-host` `-full` `-arange` | `eye`, `linspace` |
| conversion | `astype` `item` `->clj` `eval!` `array?` `host?` `backend-array?` `shape` `ndim` `size` `dtype` | `-astype` `-to-host` `-eval!` `-shape` `-dtype` | `ndim`, `size` |
| arithmetic | `+ - * /` (variadic), `negate`, `pow`, `maximum` `minimum`, `mod` `floor-div`, `abs` `sign` | `:add :subtract :multiply :divide :power :maximum :minimum :remainder :floor-divide :negative :abs :sign` | — |
| exp/log | `exp` `log` `expm1` `log1p` `sqrt` `square` `logaddexp` `xlogy` | `:exp :log :expm1 :log1p :sqrt` | `square`, `logaddexp` (native in MLX: one op, and its gradient is exact at ties under MLX's autodiff), `xlogy` |
| trig | `sin` `cos` `tan` `tanh` `sinh` `cosh` `asin` `acos` `atan` `atan2` `sigmoid` `softplus` | `:sin :cos :tan :tanh :sinh :cosh :arcsin :arccos :arctan :arctan2` | `sigmoid`, `softplus` |
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
- Following NumPy and MLX, `array/maximum` is elementwise and `array/max` is a
  reduction.
- Reductions take an options map: `(array/sum x {:axis -1 :keep-dims? true})`.
- **`array/pick`** is the one new op. It indexes the last axis of `x` by an int
  array, broadcasting batch dimensions:
  - `(array/pick log-probs v)` *is* a categorical log-prob;
  - it replaces the three shape branches GenMLX's `categorical` needed
    (dist.cljs:590-660; bug genmlx-ql6a).

**Deliberately absent from v1:**
- the LLM ops (sdpa, rope, rms-norm, quantized matmul): M3;
- sort, top-k and general scatter: added when a ported module needs them
  (`scatter-add` along an axis, the transpose of `take-along-axis`, came with
  autodiff: §12.8);
- `vmap` and `compile`: §8;
- devices and streams: backend options (§13).

## 7. Randomness (`gen.random`)

- **The scheme is JAX's default ("partitionable") Threefry** (decided
  2026-10-18), verified bit-for-bit against JAX 0.9.1 and Random123's
  known-answer vectors:
  - `split(k, n)[i] = threefry(k, [hi32(i) lo32(i)])` and `fold-in(k, d) =
    threefry(k, [hi32(d) lo32(d)])`, so `fold-in(k, d) = split(k, ·)[d]`.
    fold-in takes non-negative 64-bit data (owner, 2026-10-22, the post-M5
    review's P2-09): below 2^32 its key is JAX's, bit for bit; beyond, it
    is distinct, where JAX masks typed data to 32 bits and Recurse's deep
    node numbers then shared keys. Its data is a non-negative integer:
    negative or fractional data is a loud error.
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
  - beta and dirichlet: via gamma, from the gammas' logs (a sigmoid, a
    softmax), which do not underflow.
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
  - `(array/grad f)` differentiates with respect to argument 0;
    `(array/grad f {:argnums [0 2]})` with respect to several arguments.
  - `(array/value-and-grad f opts)` returns `[value grads]`.
  - Arguments may be trees; each gradient has the same structure as its
    argument.
  - `array/stop-gradient` blocks differentiation.
- **JVM backend: `gen.array.tape`, a generic reverse-mode AD.**
  - `grad` wraps the inputs as *tracked* values. A tracked value is immutable:
    it carries its primal value, its parents and a VJP closure. There is no
    global tape, so it is purely functional.
  - Ops on tracked values compute the primal through the normal API and record
    the VJP. The backward pass is a topological sweep with a local transient
    accumulator.
  - An op whose operands are all host scalars computes on the host, as it
    does untaped (`array/host-elementwise`, `array/host-reduce`), so a
    function's taped values are its values on every backend. On MLX the tape
    once made them float32 arrays: two scalars 4.5e-8 apart were equal to
    some ops and not to others, and `logaddexp`'s gradient went entirely to
    one argument (found by the tape's property test, 2026-10-19).
  - VJP rules form a table keyed by the primitive op keywords of §6, and are
    written against the public API. Composites therefore need no rules, and the
    same tape works on any backend; it could even test MLX's native AD.
  - `array/item` on a tracked value returns its primal. Data-dependent control flow
    is fine (non-differentiable branches, as in Gen.jl).
  - Scope for M1 is first order. Nested AD (Hessians) is bean genclj-hors.
- **MLX backend:** the same tape by default (§13). With
  `-Dgen.mlx.grad=native`, `array/grad` and `array/value-and-grad` use
  `mlx_value_and_grad` instead (bean genclj-fuvr), which also nests:
  - MLX differentiates the graph the function builds, so its VJPs apply.
    MLX has none for its Cholesky, LU or inverse, so each linear-algebra op
    is one `mlx_custom_vjp` whose VJP is the tape's rule
    (`gen.array.tape/linalg-vjp`, one definition for both); MLX is not
    patched.
  - The exact float64 composites (exp, sin, cos, erf, erfinv, expm1,
    logaddexp) are custom VJPs too, with the functions' derivatives, so
    MLX does not differentiate their Newton steps and branches.
  - MLX's `maximum` gives a tie's gradient to its second argument (the
    tape splits it). So composites avoid it where they are smooth:
    `softplus` uses a double `where`, and `logaddexp` is MLX's own op.
  - MLX's `Full` multiplies a fill's cotangent by the fill value (its JVP
    is right). So the backend fills with an array value by a cast and
    `broadcast_to`, whose VJP MLX has right, and keeps `mlx_full` for host
    scalars (genclj-hf5z; a draft upstream issue is
    `docs/research/mlx-upstream/issue-1-full-vjp.md`).
  - Inside `array/compile` the custom VJPs are left out, and MLX
    differentiates the composites' own graphs (it has no VJP for linear
    algebra there). MLX keeps compiled graphs in a thread-local cache, and
    destroying it at the main thread's exit released the custom functions'
    Java closures by an upcall into a JVM shutting down: a crash, and then a
    JVM error reporter that spun instead of exiting (test
    `compiled-native-gradients-survive-the-exit`).
  - **Opt-in, measured (2026-10-19).** The whole battery passes in native
    mode on Metal (294 tests). Against the tape (`docs/benchmarks.md`):
    faster for plain HMC (about 25% at a million observations) and compiled
    MALA (1.7×), slower for compiled HMC (25–30%), level or close elsewhere.
    The tape stays the default (confirmed by the owner, 2026-10-20): its
    gradients are the same rules on every
    backend, where native ones depend on MLX's VJPs (ties, the linear algebra
    it cannot differentiate, the compile cache at exit). Native mode is the
    choice for nested gradients, which the tape does not do.
- **Custom VJPs** (2026-10-20): `(array/custom-vjp f vjp)` returns f with
  its gradient given: under `value-and-grad`, `(vjp args out g)` gives one
  cotangent per argument, each of its argument's shape (nil for none). The
  arguments are arrays or host numbers, the result one float array or
  number. A backend implements the optional protocol `ICustomVjp`: the tape
  records one node whose VJP is the rule; MLX in native mode makes one
  `mlx_custom_vjp` (where f runs inside the transformation, so it may not
  read values on the host); every other backend calls f. A rule with the
  wrong number of cotangents is a loud error.
- **Reserved names:** `vjp`, `jvp`, `vmap`.
- **`array/compile`** (2026-10-19, bean genclj-8b88):
  - **On MLX.** `(array/compile f)` traces f once per input structure, shapes
    and dtypes into one MLX graph, which `mlx_compile` fuses and replays.
    Maps in the arguments match whatever order they list their keys in (an
    independent review found arguments swapped when they did not, fixed
    2026-10-19). MLX keeps its compiled graphs per thread, so a compiled
    function called from another thread traces there again.
  - **Threads** (bean genclj-hfwo). A compiled function may be called from
    any thread. Each MLX closure is compiled with an id of its own, never
    reused, and freeing it erases its traces from the cache of every
    thread that called it (`MlxRuntime.Compiled`).
    - Before, mlx_compile's closure erased only the compiling thread's
      trace. Another thread's trace stayed, with every constant it
      captured, until that thread exited.
    - That trace was keyed by the address of a deleted function, which a
      later compiled function could be given.
    - From a virtual thread, whose MLX cache is its carrier thread's, the
      trace is erased after each call, so a compiled function traces every
      call there.
  - **On the JVM backend**, it is f itself.
  - **Arguments** are arrays or trees of them, and so are results.
  - **f must not read a value on the host** (`truthy?`, `item`): that is a
    loud error, since the trace is not run again.
  - **Key arguments are errors,** since the trace would replay their draws.
  - **Measured** for value and gradient, and 10 leapfrog steps, of a
    log-density:
    - Metal, 10,000 elements: 0.63 → 0.26 ms and 3.9 → 0.63 ms.
    - Metal, a million: 9.8 → 0.97 ms and 229 → 21 ms (the JVM backend
      takes 256 ms).
    - The MLX CPU: 2–8×.
  - **In inference:** `(mcmc/hmc tr sel {:compile? true})` runs its L
    leapfrog steps as one compiled function of the selected values and the
    other choices, cached per model, arguments, structure, L and eps;
    `(mcmc/mala tr sel tau {:compile? true})` does the same for MALA's
    proposal and its densities. A model that reads a value on the host falls
    back to the plain kernel, which makes the same moves. It is opt-in
    because the first call per structure pays for the trace and compile.
    Measured on Metal, per step, on a regression over N observations with
    two latents (`gen.mcmc-bench`, `docs/benchmarks.md`):
    - HMC, N = 1M: 34.9 → 5.9 ms (the JVM backend: 168 → 126 ms); N = 100:
      12.2 → 1.5 ms.
    - MALA, N = 1M: 6.1 → 4.6 ms; N = 100: 1.6 → 1.3 ms.
  - **Not compiled: steps that draw inside the model** (MH's regenerate,
    particle-filter extensions). Keys are host values, split on the host, so
    a compiled trace would replay its draws. GenMLX's fused chains and
    compiled SMC draw their noise outside instead; those are beans
    genclj-l9ts and genclj-cdlp.
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
     particle. Code that should stay batched computes both values and picks
     with `where`. A branch in structure (different choices, or a switch
     combinator on a per-particle index) runs one particle at a time.
     *Revised 2026-10-18:* numbers were falsy at 0, which broke `(or 0 5)`
     and `(when-let [i 0] …)` inside `gen`.
2. **Inside `gen` bodies, conditionals branch on values** (bean genclj-pa7x).
   - The `gen` macro macroexpands the body and wraps every `if` test in
     `truthy?`. That also covers `when`, `cond`, `and`, `or` and `if-let`, which
     expand to `if`.
   - It also substitutes guarded versions (`gen.dynamic.core`) for the
     clojure.core functions that test truthiness or look up by key with an
     `if` of their own: `not` (so `if-not`), `boolean`, `true?`, `false?`,
     `every?`, `some`, `filter`, `remove`, `take-while`, `complement`, `get`,
     `contains?`, `find`, `get-in` and the like. They behave as
     clojure.core's on ordinary values and raise the batched-branch error
     on a batched one. *Added 2026-10-19 after the second acceptance test:*
     `(if-not b …)` and `(filter pred xs)` in a body took one branch for
     every particle.
   - Representation tests (`class`, `instance?`, `number?`, `coll?`, …, and
     gen.array's `host?`, `array?` and `backend-array?`) answer for a
     particle's value, whose type the batch's dtype and per-particle shape
     determine.
   - A 0-d backend array counts as its host value wherever a body branches,
     compares or looks up on it (MLX's 0-d arrays are MLX arrays; the JVM
     backend's are host scalars already): `case`, `=`, `not=`, `true?`,
     `false?`, `identical?`, the lookups, `nth`, and the representation
     tests, which see a 0-d value as a host scalar of its kind. As with
     `truthy?`, reading the value evaluates it. So a body written in plain
     Clojure behaves alike on every backend. *Added 2026-10-19 with the MLX
     backend:* on MLX, a body's `(case k 0 …)` took the default clause and
     `(= k 0)` was false.
   - **Plain helper functions** outside `gen` get none of this. On MLX they
     see arrays, not numbers: clojure.core's `=` with a number is false and
     its arithmetic throws. A helper should use gen.array's operators or
     `array/item`, or be `defn-guarded`.
   - A `catch` clause in a body raises the fallback first in a batched run:
     a caught exception may be thrown for some particles only. And a
     batched-branch error is logged when raised, so a run in which code
     caught one falls back anyway, also in a plain helper. *Added after the
     fourth acceptance test.*
   - Non-array tests behave exactly as in Clojure.
   - A model written with plain Clojure conditionals is correct on every
     backend in scalar mode.
   - In batched mode, branching on a batched value raises the batched-branch
     error. So does any host predicate on one (`=`, `case`, map keys, which
     compare or hash it; see §8b). The error tells the batched executor to fall
     back to running particles one at a time. That is correct, just slower.
     Batching is an optimization, and the handler is ground truth.
   - Outside `gen` bodies (helper functions a model calls), a raw `if` or
     `case` on an array is not guarded. Define a helper that branches on
     values with `gen.dynamic/defn-guarded` (or `guarded-fn`), which applies
     a body's guards to it, or branch with `array/truthy?` or `where` there. For
     a plain helper, what still catches a branch (§8b):
     - `=`, `not=` and `condp =` reach a batched value anywhere, through `=`.
     - A batched comparison that nothing else reads, the trace a raw `if` on
       it leaves, raises the fallback when the run ends. One that is also
       read (by `where`, `or`'s value, or as a return value) is not caught
       this way.
     - Spot checks against the scalar handler catch the rest (a `case` on a
       choice, a raw `if` on a bernoulli's value, a vector lookup keyed by a
       choice) with high probability, not certainty. A branch that few
       particles take can escape them.
   - Known differences from plain Clojure inside bodies, all deliberate: a
     non-canonical `(Boolean. false)` is false to a body's `if` (as
     `truthy?` reads it); a guarded function is not `=` to clojure.core's
     (`(= not clojure.core/not)` is false in a body); and macros used in a
     body are expanded before the body compiles, so a macro that reads
     `&env` does not see the body's own locals.
   - What a body cannot guard is code it calls that looks at a batched
     value's object: Java interop (`(.equals Boolean/TRUE b)` is false), a
     multimethod dispatching on `class` or `type`, or a protocol extended to
     `Object`, which dispatch on the batch rather than a particle; and
     `pr-str`, which prints the whole batch.
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
  (`array/shape`), and every op acts per particle: batched and unbatched operands
  align by their per-particle axes, as with JAX's `vmap`.
- **Draws.** Under the batch backend, draws gain the batch axes, one
  independent draw per particle, even when every parameter is unbatched
  (`array/effective-backend`).
- **Shared values.** Creation functions and constants stay unbatched, shared
  by every particle.
- **Reading on the host.** Reading a batched value on the host (`array/item`,
  `array/->clj`, `truthy?` of a boolean, `=`, hashing) raises the batched-branch
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
  when the run ends, and that the run does not hand back anywhere in its
  trace (choices, return values, sub-traces), is the trace of such a
  branch. It raises the fallback. This catches the common case
  deterministically: a helper's `(if (< x c) …)` on a comparison used for
  nothing else, however rarely the other branch is taken. A comparison
  that is never used at all raises it too (needlessly); one that something
  else also reads is not caught this way.
- **Spot checks.** After a batched simulate or generate, 8 evenly spaced
  particles (`*spot-checks*`: a number, `:all` or 0) are re-run through the
  scalar handler from their choices. So are the 1st, 2nd, 4th, 8th, 16th, …
  update or regenerate of a collection (`*verified-updates*`). A score or
  array return value that differs from the batched one raises the
  fallback. One found after unchecked updates is a loud error instead, since
  their results may be wrong too.
  - Each check is a scalar run, so the doubling schedule keeps a long
    particle filter linear, at about two checks of the final trace.
  - On a lazy backend (MLX) the batched trace is read on the host once,
    and the checked particles are re-run on the JVM backend, the
    reference: microseconds each, where a scalar MLX run costs
    evaluations. The comparison is at the precision of the batched float.
    A model that closes over the lazy backend's arrays cannot run on the
    JVM, so it is checked on its own backend. *Added 2026-10-19, bean
    genclj-w5y0.*
  - Results are checked when read. The accessors (`traces`, `scores`,
    `choice`), the particle filter's estimate and a fallback all spot-check
    a collection whose last operations went unchecked
    (`gen.batched/check`). A mismatch there is the loud error too.
  - A scalar re-run of a particle's choices that fails (it visits other
    addresses: a structural divergence) counts as a mismatch.
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

- `(array/eval! tree)` returns its argument and is a no-op on eager backends.
- `(array/item x)` returns a 0-d array's host number or boolean.
- `(array/->clj x)` returns nested vectors.
- Core code calls `eval!` only at inference-loop boundaries (e.g. once per MCMC
  iteration). Never inside handlers, transitions or distributions, which is the
  GenMLX smell this rule avoids.
- The API exposes **no memory management.** On MLX, each array is freed
  once it is unreachable, through phantom references that the threads
  making arrays drain (§13; the plan of queuing frees to an owning thread
  proved unnecessary). If that proves
  insufficient under native-memory pressure, a scoping construct is added to
  the MLX backend, not to the core API (bean genclj-gip5).

## 10. Errors

- Shape mismatches, mixed backends, unsupported dtypes and out-of-range axes
  throw `ex-info` with `{:op … :shapes … :dtypes …}`.
- gen.array and gen.array.linalg check shapes, with gen.array.shape's rules,
  before they call a backend, and gen.array parses host data
  (gen.array.host) for every backend (genclj-q45d). So the backends do not
  repeat the checks, and under the batch tracer an error names the shapes
  the model wrote, not the batch's.
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
   house style refers the operators instead of writing `array/+` (§2).
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
14. **`gen.dynamic/defn-guarded` and `guarded-fn`** (§8a; 2026-10-19,
    agent, for the owner's review): a helper function with a body's guards,
    the deterministic way to branch on values outside `gen`.
15. **MLX threads: shared streams and an evaluation lock** (§13;
    2026-10-19, agent, for the owner's review). Streams any thread can use
    (`mlx_stream_new_thread_unsafe`), with evaluation serialized by one
    lock. This replaces "one platform thread per MLX context", which would
    cost a thread handoff per op or per read.
16. **MLX arrays are freed through phantom references,** drained by the
    threads that make arrays, with pressure-driven collections (§13, §9;
    agent, for the owner's review). This revises the first version, which
    freed on a `Cleaner`'s thread and did not keep up under load.
17. **MLX 0-d arrays stay MLX arrays,** and out-of-range array indices are
    clamped (§13; agent, for the owner's review). Both keep the backend
    lazy. Host indices are still checked.
18. **The MLX backend follows NumPy's semantics where MLX's differ** (§13's
    table; agent, for the owner's review): NaN in `sign` and `argmax`,
    integer floor division, and float64 accuracy on the CPU. Metal's
    flushing of subnormals is documented instead.
19. **In `gen` bodies, a 0-d backend array counts as its host value** for
    `case`, `=`, lookups and type tests (§8a; 2026-10-19, agent, for the
    owner's review). It extends what `truthy?` already did for `if`, so
    that a body in plain Clojure means the same on MLX as on the JVM.
20. **Distribution parameters given as host data are validated on the
    host** (§13; agent, for the owner's review). This holds on every
    backend. Lazy arrays are still not read.

## 13. The MLX backend (M2)

*Added 2026-10-19 with the backend (`gen.backend.mlx`, bean genclj-kgvx).
The decisions are §12 items 15–18, for the owner's review.*

**The binding.** mlx-c over FFM (DESIGN.md §5.3):
- `native/build-mlx.sh` builds libmlxc v0.7.0 against the official MLX
  0.32.2 wheel, which mlx-c pins (checked by sha256). No Metal compiler is
  needed.
- `native/bindgen.clj` generates the Java layer from the mlx-c headers:
  - `MlxRaw` has one `invokeExact` static per C function (665 of 674);
  - `Mlx` wraps 558 of them with checked statuses and out-parameters as
    return values.
- `MlxRuntime` is hand-written. It loads the library, installs the error
  handler and holds the per-thread scratch and the closures.
- `clojure -T:build java` compiles it into `target/classes`.
- Measured costs, M4, Graal and C2 alike:
  - an FFM downcall, 2.4–3.8 ns;
  - an op as a graph node, about 170 ns;
  - evaluation, about 11 µs on the CPU and about 115 µs on the GPU.

**Values.**
- MLX arrays are lazy. Reading one on the host evaluates it: `item`,
  `->clj`, `truthy?`, printing, `=`.
- 0-d results are MLX arrays, not host scalars as on the JVM backend, so
  scalar code stays lazy.
- Host scalar operands are converted to the operand dtype per op.
- `concrete?` is false, so distributions validate host parameters only.

**Configuration.** System properties, read when the namespace loads; or
`(mlx/make-backend {:device :cpu :dtypes 64})`.
- `gen.mlx.device`: `cpu` or `gpu` (default `gpu` when there is one).
- `gen.mlx.dtypes`: `32` (float32/int32) or `64` (float64/int64, CPU only).
- `-Dgen.backend=mlx` makes it the default backend.

**Threads and memory.**
- Ops run on streams made with `mlx_stream_new_thread_unsafe`, which any
  thread can use. One lock serializes evaluation, so arrays move freely
  between threads, futures included.
- **Freeing.** Each handle belongs to its `MlxArray`. `MlxRuntime/adopt`
  gives it a phantom reference, and the threads that make arrays drain
  the reference queue, so frees happen on MLX-using threads. A free only
  drops a reference: MLX's reference counts are atomic and its allocators
  lock. (A `Cleaner` came first. Its thread's timing was not enough under
  load.)
- **Collecting under pressure.** The GC does not see MLX's buffers, and
  Metal allows about 500,000 of them, every array with data holding one,
  also on the CPU device. So the backend collects (`System.gc`, then
  drains) when 100,000 arrays are live (`gen.mlx.gc-arrays`) or MLX's
  active memory passes an eighth of RAM (`gen.mlx.gc-bytes`). The
  threshold rises to twice what survives a collection, capped at 60% of
  Metal's limit.
- **Collecting by new bytes.** It also collects once the arrays made since
  the last collection hold a sixteenth of RAM (`gen.mlx.gc-new-bytes`),
  counting their sizes before they are evaluated. A dead intermediate holds
  its buffer until the GC finds its `MlxArray`, and while that lives MLX
  cannot free the buffer during evaluation either. Without this trigger a
  chain of large ops kept every intermediate: 100,000 float64 beta draws
  (exact float64 composites inside rejection rounds) took over 11 GB, and
  with it about 1.7 GB. A collection costs 10–25 ms. Uncompiled steps on
  large arrays notice: plain HMC at a million observations collects every
  second step, 29.7 → 34.9 ms per step, while its compiled steps make few
  new arrays and do not (`docs/benchmarks.md`). A larger
  `gen.mlx.gc-new-bytes` trades memory for fewer pauses.
- **At Metal's buffer limit** (genclj-9tyh, DESIGN §9). Graphs can still
  reach the limit, since the cap counts only the arrays the JVM owns, and
  MLX recovers fully once buffers are freed.
  - So where a buffer is made (an array from host data, or an
    evaluation), "[metal::malloc] Resource limit exceeded" runs a full
    collection, whatever the thresholds, and clears MLX's buffer cache.
    Then the call runs once more.
  - At the limit again the buffers are held, and the error is loud: an
    ex-info with `:reason :mlx-buffer-limit`, `:live-arrays`, `:cap` and
    the hint (eval! as you go, and drop graphs and arrays no longer
    needed), with MLX's error as its cause.
- **CUDA (Linux)** (2026-10-19, the Jetson Thor). Every evaluation runs on
  one thread, with the caller's bindings: MLX's CUDA backend fails when a
  stream is evaluated from another thread than before, and the stream then
  stays broken. Metal and the CPU devices let any thread evaluate, under the
  evaluation lock. MLX's prebuilt wheel has no code for every GPU (none for
  the Thor's sm_110); `native/build-mlx.sh cuda13-source` builds MLX for the
  machine's own. A host read inside a compiled function reads differently
  under CUDA ("eval an array without a primitive"), and is the same loud
  ex-info. MLX's CUDA graphs are off unless `MLX_USE_CUDA_GRAPHS` is set:
  MLX 0.32 counts graph-cache misses from the start and throws ("Cache
  thrashing is happening") once they pass twice the cache's capacity, so a
  long run of varied shapes always failed (the battery, after a minute).
  Kernels then launch one by one: the same values, slower launches.
- **Compile on MLX's CPU device needs a compatible g++** (2026-10-19). MLX
  builds each fused CPU kernel at run time with the system's g++, from a
  header the wheel ships, and GCC 13 on Linux rejects that header (it
  redeclares the built-in `_Float128`). The first compile on the CPU device
  probes this with a two-op function; where MLX cannot build it,
  `array/compile` is the identity on the CPU device (the same values,
  unfused) and a warning says so once. `gen.backend.mlx/compiles?` tells;
  `-Dgen.mlx.cpu-compile=true|false` skips the probe. Metal and CUDA are
  unaffected (bean genclj-hjzh).
- **MLX's own limits are bounded** when the namespace loads (2026-10-19).
  MLX defaults both to all of RAM: its buffer cache (freed buffers kept for
  reuse) and its memory limit (past which it frees the cache and waits for
  running work before allocating). With those defaults one test JVM grew
  to 40 GB on a 32 GB machine, and two together exhausted memory and
  caused a kernel panic (watchdog timeout, compressor and swap full). The
  cache is now a sixteenth of RAM (`gen.mlx.cache-bytes`) and the memory
  limit half of RAM (`gen.mlx.memory-bytes`); `set-cache-limit!` and
  `set-memory-limit!` change them at run time.
- **Lazy graphs pin their leaves.** An unevaluated graph keeps its input
  buffers alive, every host number used as an operand among them. Host
  scalars are therefore shared constants (4,096 per dtype,
  `gen.mlx.constants`), and Threefry takes its key as one buffer.
- **Evaluation at loop boundaries (§9).** Inference loops evaluate once
  per iteration (`gfi/eval-trace!`): MCMC runs, particle-filter steps,
  scalar importance sampling, and optimizer steps (`learn/step` itself,
  so any loop over it, genclj-9e9e). So a graph stays one iteration deep.
  Before this, scalar importance sampling with 10,000 particles kept
  10,000 pending traces, about 250,000 buffers.
- **The backend cannot see an unevaluated graph's size.** A graph holds
  every array it reads in MLX's own nodes, after the JVM has freed their
  handles: the live count, MLX's active and cache memory and the JVM heap
  all stay small while the process grows. MLX exposes no graph size, so
  neither the collections nor the budget checks can bound it; only
  evaluation does. Measured on Metal: about 270 KB of resident memory per
  lecture of a training loop that stepped without evaluating (genclj-9e9e).
  `gen.backend.mlx/evaluated?` tells a value from a graph.
- Decompositions and solves run on the CPU stream, the only one where MLX
  implements them.

**Randomness.** `-random-uniform` computes JAX's partitionable Threefry with
MLX's uint32 ops. The same key draws bit-identical uniforms on MLX and on
the JVM backend, in float32 and float64 alike. Since every sampler is a
composite over this primitive, the two backends sample identically up to
the rounding of each op (bean genclj-ldpy's stretch goal).

**Gradients.** The backend-generic tape (`gen.array.tape`) by default;
MLX's own with `-Dgen.mlx.grad=native` (§8).

**Where MLX's semantics differ, and what the backend does.** Found by the
conformance suite.

| MLX | The backend |
|---|---|
| `sign(NaN)` is 0 | NaN, as NumPy |
| `argmax`/`argmin` skip NaN; indices are uint32 | the first NaN wins; the default int dtype |
| integer `floor_divide` truncates toward 0 | floors: `(x - x mod y) / y` |
| integer division by 0 is undefined (a trap on x86) | 0, as the JVM backend |
| negative-step slices read a stop of -1 from the end | NumPy's bounds |
| float64 `exp`, `sin`, `cos`, `erf`, `erfinv` on the CPU are float32 approximations (up to 1e-4 relative) | exact routes via `expm1`, `tan`, `log` (`gen.backend.mlx.exact`), within a few ulps |
| Metal's float32 `expm1` is off by up to 660 ulps | Kahan's formula from `exp` and `log`, 3 ulps |
| Metal's float32 `atan(-0.0)` is +0.0 | x itself at zero, as NumPy |
| empty arrays have no data pointer | read as empty |
| Cholesky of a matrix that is not positive definite returns a partial factor | NaN where a diagonal entry is not positive (LAPACK's failing pivot) |
| a singular `solve` fails inside a CPU task, reported at a later evaluation of other arrays | the identity stands in for matrices with a zero LU pivot; their solutions are NaN |
| a zero diagonal fails the triangular solve (it inverts), reported later | NaN on the diagonal: NaN solutions |
| `slogdet` throws on a singular matrix | computed from the LU factor: `[0 -Inf]` |
| solves want equal batch axes | broadcast first |
| integer `power` gives 0 for a negative exponent, and for every element of an array holding one | the JVM backend's semantics: exact for exponents ≥ 0 (on a stand-in), through floats with Java's cast for negative ones |
| gathers do not check bounds | array indices: negative from the end, clamped (host indices are checked) |
| `matmul` multiplies floats only | integers through the widest float the device has: exact while products stay below 2^24 (float32) or 2^53 (float64) |
| `remainder` of an exact negative multiple, `(mod -4.0 2.0)`, is -0.0 | kept (NumPy and the JVM backend give 0.0; the two are `==`) |

**A hazard to know.** MLX runs CPU work as asynchronous tasks. An
exception inside one is raised at a later synchronization, against whatever
arrays are evaluated then. The backend therefore never gives MLX inputs its
kernels fail on. New native ops must keep that rule.

**Remaining differences, documented rather than hidden:**
- **Subnormals.** Metal flushes float32 subnormals to zero: `sqrt(5.9e-39)`
  is 0, and the sqrt of a negative subnormal is -0, not NaN.
- **Validation.** Array parameters and constrained array values are not
  read on a lazy backend (rule 5), so a NaN or an invalid probability
  vector in an MLX array is not an error there. Parameters given as host
  data (numbers, Clojure vectors) are validated on the host before they
  become arrays, on every backend.
- **Integer overflow in `power`.** The JVM backend computes through
  doubles and saturates; MLX multiplies exactly and wraps.
- **Indices.** An out-of-range array index is clamped, not an error. A lazy
  backend cannot check it without evaluating. The JVM backend is the one
  that checks.
- **Degenerate linear algebra** gives non-finite values on both backends,
  but not always the same ones: NaN on MLX where the JVM's substitutions
  give ±Inf.
- **Accuracy.** Float32 results measured against the JVM backend's
  correctly rounded ones, in ulps:
  - Metal: at most 3, except `erf` (12).
  - The CPU: at most 2, except `exp` (14) and `erf` (about 24).
  - float64 on the CPU: within a few ulps everywhere, `cos` within 1e-13
    relative near its zeros.

**Conformance** (`test/gen/backend_conformance_test.clj`). Every op of the
contract, and the composites built on them, is a test.check property. Each
runs on the tested backend and on the JVM backend, the reference, with the
same data and dtypes:
- **Dtypes.** Where the JVM backend's 64-bit defaults gave a result's dtype,
  its value is rounded to the tested backend's 32-bit defaults, as if
  computed in float64 and stored as float32.
- **Tolerances**, by op class:
  - exact: ints, bools, comparisons, selections, shapes, indexing and the
    IEEE-exact `+ - * / sqrt`;
  - transcendental: 1e-5 relative in float32, 1e-13 in float64;
  - accumulating: relative to the terms.
- **Errors.** They must agree: `ex-info` on both, with the same `:shapes`.
- **Results.** It passes on Metal, on the CPU in float32 and on the CPU in
  float64, at 400 trials per property. Two planted bugs were caught: a
  swapped `<`/`<=`, and a reversed `cumsum` direction.
