# gen.clj's decision log

The decisions behind gen.clj's design and conventions, with who made each,
when, why, and the alternatives not taken. [DESIGN.md §9](../DESIGN.md#9-conventions-current)
states the conventions in force and links each to its entry here; until
1.2.1 this log was DESIGN.md §9 itself, and a citation of "DESIGN §9,
<title>" means the entry of that title.

- **Entries are never rewritten.** A later decision is a new entry that
  names the entry it supersedes; the earlier one stays as it was decided.
  A PROVISIONAL entry gains one dated line when the owner confirms it.
- **By release, then by date.** An entry sits under the release whose work
  it decided, in the order of its first date; one amended later (approved,
  confirmed, as built) stays where it was first decided. Entries moved here
  keep their words, so their "above" and "below" refer to DESIGN.md §9's
  old order; the titles they name are the headings here.
- **New decisions** are appended under the current release, as `### <title>`
  with the attribution (who, the date, the bean) as the first line, and
  DESIGN.md §9's summary is updated in the same commit.

## Before 1.0 (M1-M5 and the book, 2026-10-18 to 2026-10-25)

### Namespace root: `gen`

(`gen.choicemap`, `gen.dynamic`, …). The collision
with probcomp/Gen.clj's `gen.*` is accepted, since the two are never on one
classpath. Maven coordinates are decided at first publish.

### Indices are 0-based

(Clojure idiom, as in GenMLX; D2): Map's and
Unfold's addresses, Recurse's nodes (root 0), categorical's values
(categorical-logits' too: 0 to n - 1) and Switch's branch index (branch 0
is the first). Gen.jl is 1-based; document this in a migration note.

### Distributions use Gen.jl names and parameterizations

(`normal(mu, std)`,
`gamma(shape, scale)`, `categorical(probs)`), plus explicitly named variants
where they matter numerically (e.g. `categorical-logits`). Each GenMLX name
whose meaning changed has a safeguard that makes a naive port fail loudly
(`docs/porting-genmlx.md`). Families Gen.jl does not have take
Distributions.jl's parameters, which GenDistributions' users write
(`pareto(alpha, theta)`, `triangular(a, b, c)`; M4, 2026-10-20).

### House style: gen.array's operators are referred

in place of
`clojure.core`'s (`(+ a b)`, not `(array/add a b)`); `max`, `min`, `take` and
`flatten` are never referred (`docs/array-api.md` §2, §12).

### Branching on array values

the `gen` macro wraps every `if` test in
`array/truthy?`, so a one-element array branches on its value and a batched
array throws instead of silently taking one branch (`docs/array-api.md` §8a).

### Randomness

JAX's partitionable Threefry-2x32-20 in pure Clojure,
bit-identical to JAX. Every trace site consumes one key. The implicit key
source has a random root seed for each JVM (`gfi/seed!` fixes it). See §3.5.
`fold-in` takes a long's bits (owner, 2026-10-22, the post-M5 review's
P2-09): it hashes `[hi32 d, lo32 d]`. Its data is a non-negative integer
below 2^63: negative or fractional data is a loud error, which no caller
passes. The old 32-bit mask wrapped negatives as JAX does, and a long
cast truncated 2.5 to 2.
- Below 2^32 a key is JAX's, bit for bit. Beyond, it is distinct, where
  JAX masks typed data to 32 bits.
- With the mask, Recurse's node numbers past 2^32 (depth 32 for
  max-branch 2, 16 for 4, 11 for 8) gave deep nodes one key, and their
  draws repeated, silently. The implicit key source's counter repeated
  after 2^32 keys.
- Now `fold-in(k, d) = split(k)[d]` for every d.
- Alternatives: a loud error for data ≥ 2^32; or keys folded child by
  child in Recurse, which would change every Recurse program's draws.

### Traces store each choice's log-probability

(as Gen.jl does), so update
and regenerate weights use the true old values and `project` never re-runs
the model.

### Trainable parameters are an explicit store

(decided 2026-10-18 by the
agent, for the owner's review): a map of name → value, attached with
`(dyn/with-params gf store)` as a record field. `(param :w)` in a body reads
the store in effect, which reaches every nested splice and combinator kernel
(one namespace per model, like Pyro's store). A missing parameter is a loud
error. **What an update or regenerate may keep of a trace is decided in
two places, for every kind of generative function** (revised 2026-10-20
after reviews 5 and 6; reviews 3 and 4 had found one kind after another that
kept a stale call while each kind checked for itself: Map, Unfold,
Recurse, estimated densities, the proposal overrides, a deterministic
callee picked by a choice). No argument shows a store or a function, so
no kind's own skipping could see either.
- **The maker.** A trace records the generative function that made it
  (`gen.trace/maker`; `(assoc tr :gen-fn g)` keeps it, and a wrapper's
  trace of its model is the wrapper's, `trace/as-made-by`). `gfi/run-op`,
  the dispatch of every nested call and of `gfi/update`,
  `gfi/regenerate` and gen.batched, revises (`gfi/revise`) a trace whose
  maker the generative function does not compute as. `gfi/project`
  reads a trace by its maker too, as Gen.jl's project dispatches on the
  trace's type (the post-M5 review's P1-06: by `:gen-fn`, a Map's trace
  given a dynamic function projected to 0). Particles given
  another gen-fn are revised per particle as when batched (their
  per-particle traces are rebuilt by their maker and given the new
  gen-fn). That is
  `gfi/same-gen-fn?`: identical, or of one class and the same by its
  kind's rule (`gfi/IRevisable`). A record's fields must be the same
  values (`gfi/same-value?`), as code holding them could tell (review
  6). Values must be of one class. Scalars compare by `.equals`, which
  tells −0.0 from 0.0. Maps, sets and sequential collections compare by
  metadata and element by element in iteration order, so an array-map's
  order or a vector against a list counts, and nothing is hashed. Fns
  compare by their closed-over values, and generative functions
  recursively. Arrays compare by identity, never read, and a comparison
  looks at most 256 values. A DynamicGF compares its body fn, store,
  middleware and dispatcher. So a Switch, a closure or a combinator a body builds again
  is the same one, and its calls stay incremental. (On MLX a computed
  scalar is an array, so a closure over one is another gen-fn on every
  run there: still correct, but its calls run again.)
- **The store.** Each trace a call makes is marked with the store in
  effect at the call (`dyn/run-call`). A call under another store than
  the mark (compared by `gfi/same-value?`) revises the old trace, which
  is projected under its own store.
- **A revise** keeps nothing of the old trace but its choices. A
  DynamicGF revising a dynamic trace runs the handler's own update or
  regenerate against it, as an `if` would. Every site is rescored with
  the recorded log probabilities, and the static walk skips nothing for
  a trace another gen-fn made. Each nested call checks its own context,
  and middleware sees the operation asked for. Every other kind
  generates from the old choices (`gfi/generate-again`): update's
  weight is generate's less the old score, and its discard every old
  choice that did not survive unchanged. Regenerate's weight is
  generate's less the old trace's projection on the old choices it kept
  (Eq. 4.1). An old choice the new run leaves unvisited is dropped where
  it is left. Each call in the revise's generate runs its callee once
  more without it: `gfi/*revising*` carries the revise's constraints and
  the call's path, so only old choices are dropped, never a constraint.
  A structure change thus costs each call one more run: 2n kernel runs
  for an n-element Map, not n²/2. A proposal override reruns once more,
  without the drops, so that its proposal sees the final constraints
  (3n). A dropped choice cancels against the reverse move in regenerate.
  A constraint the new run leaves is the usual unvisited error. Public
  operations run outside the revise (`gfi/outside-revise`), so a
  constraint user code gives stays checked. The retdiff is
  unknown-change, and ω is drawn afresh in every kind: a changed store
  or gen-fn is a change of what the function depends on, though no hint
  shows it.
- **Unvisited constraints** are one error kind (`gfi/unvisited`:
  `:unvisited`, `:paths`), raised by every kind and named under each
  call's address on the way up (`gfi/under`), in assess too. A user
  function's own unvisited error (any of gen.custom's functions) is
  neither one of the function's choices nor named under its call, so a
  revise does not drop it.
- **Local checks** remain where no call shows the change: the static
  walk's skip, for a node reading `(param :w)`, which has no inputs.
  `dyn/with-params` and `with-transitions` take dynamic generative
  functions, whose bodies read them, and `custom/with-params` a
  deterministic function that names the `:params` its code reads (G5).

Gradients are values: `gen.gradients/param-gradients` returns a map,
and optimizers return a new store. This is Gen.jl's `@param`,
`init_param!` and `accumulate_param_gradients!`/`apply!`, but functional.

### Batched execution

(decided 2026-10-18 by the agent, for the owner's
review; revised the same night after the M1 review): `gen.batched` holds N
particles as one batched trace. A batched run is the model's scalar code
under the batch tracer `gen.array.batch`, with vmap semantics:
- values have per-particle shapes, and each op acts per particle;
- draws gain the batch axis;
- reading a batched value on the host raises the fallback.

Every generative function thus runs batched. A model whose code treats
particles differently (branches, host predicates, host math) fails batched
and falls back to one scalar trace per particle. Inside `gen` bodies this
is guaranteed: the macro guards `if`, `case` and `catch`, and substitutes
guarded versions of clojure.core's truthiness, lookup and type tests
(`not`, `filter`, `get`, `instance?`, …; `gen.dynamic.core`). Only Java
interop on a choice value, and dispatch on its class (multimethods,
protocols extended to `Object`), see the batch. Clojure has no hook for its own `if`, so
helper functions outside `gen` are covered by two later mechanisms (added
2026-10-19 after the acceptance tests; docs/array-api.md §8a–§8b):
- a batched comparison that nothing read raises the fallback;
- spot checks re-run a few particles of each batched simulate and generate,
  and of a doubling schedule of updates, through the scalar handler, the
  ground truth. A mismatch falls back, or is a loud error after unchecked
  updates. Results read from a collection (its accessors, a particle
  filter's estimate) are checked first. On a lazy backend the re-runs
  are first on the JVM backend, and only a mismatch there is re-run on
  the batch's own, which decides (2026-10-24, genclj-i8ui, by the
  agent). The JVM run is float64, and a float32 batch of an
  ill-conditioned model differs from it by its rounding: the M2
  review's regression (means near 4000, sd 0.01, 1000 observations)
  fell back on MLX, 0.07 nats apart against a tolerance of 0.034. Not
  taken: a tolerance from the sites' log-density terms. Their sum is
  the score, but the error grows with |y|/sd (the cancellation in
  y − μ), so no tolerance at the score's scale bounds it.

What remains is probabilistic: in a plain helper, a `case` on a choice, a
vector lookup keyed by one, or a raw `if` on a comparison also used
elsewhere is found by the spot checks with high probability.
`gen.dynamic/defn-guarded` gives helpers a body's guards, deterministically.
Measured: 5–90× faster importance sampling on the JVM backend (100–10,000
particles; docs/benchmarks.md).

### Bernoulli values are booleans

as in Gen.jl: 0/1 (GenMLX's) is an
error, since a body's `if` takes 0 as true while the density scores it as
false. NaN parameters and NaN constrained values are errors too.

### Testing

reference values come from scipy and JAX, generated once into
committed EDN fixtures (`test/fixtures/`); the tests never need Python.
Against GenMLX, deterministic quantities (log-densities, weights given fixed
choices) must match exactly, and sampled quantities are compared
statistically: the same seed cannot reproduce GenMLX's samples, because
draws also depend on how keys are threaded through a run.

### GFI signatures are Gen.jl-shaped

(§3.2). `generate` throws on
constraints at unvisited addresses.

### Dtypes

the JVM backend is float64 (the reference); MLX is float32. Core
code never assumes a dtype. Bool vs 0/1 is settled in the array-API design.

### Constraints are choice maps

(2026-10-19, after the second acceptance
test): a plain Clojure map given as constraints, observations or choices
is an error naming `cm/from-map`, since it would read as no constraints at
all. `nil` is the empty choice map. Selections and argument vectors are
checked the same way.

### Bodies run to completion

lazy sequences in a body's return value are
realized during the run, and a `trace` or `splice` after the run is a loud
error. Before this, `(map … (trace …))` at the end of a body gave a trace
without its choices, silently. Lazy splice arguments are realized before
the call. A lazy call with `trace` in a discarded position (e.g. `(map …)`
followed by more forms, also through `->>`) is a compile error in `gen`.
Realization stops after 10,000 elements of a return value, and later trace
calls are loud. So is a `let` binding of such a lazy call that is never
used (2026-10-19). Elsewhere a lazy sequence's trace calls run when its
elements are realized, Clojure's semantics, chunking included: `mapv`,
`doseq` and `run!` are eager.

### MLX is upstream, patched as little as possible

(owner, 2026-10-19).
CPU and Metal use the official MLX wheel, unpatched. If a fix is ever
needed, for CUDA first, it comes as a clean patch series on a pinned
upstream tag (`native/patches/`): one concern per patch, rebased at each
MLX bump, dropped once upstream has it. Never a long-lived fork.
GenMLX's fork was audited (`docs/research/mlx-fork-audit.md`): none of its
41 patches is needed for CPU or Metal.

### The MLX backend

(2026-10-19, agent, for the owner's review;
`docs/array-api.md` §12 items 15–18 and §13):
- Streams any thread can use, with evaluation serialized by one lock, in
  place of one platform thread per MLX context (which would cost a thread
  handoff per op or read).
- Arrays are freed through phantom references, drained by the threads
  that make arrays, so frees run on MLX-using threads, close to the
  planned design. The backend collects under array or memory pressure,
  since the GC does not see MLX's buffers (Metal allows about 500,000).
  Host-scalar constants are shared, so lazy graphs pin few buffers.
- 0-d results stay lazy MLX arrays.
- Array indices are clamped, not checked: a lazy backend cannot check
  them without evaluating.
- Memory is bounded by default (2026-10-19, after a kernel panic from
  memory exhaustion by two test JVMs): MLX's buffer cache to a sixteenth
  of RAM, its memory limit to half, and the backend collects past an
  eighth of RAM of active memory. MLX's own defaults are all of RAM.
- Where MLX's semantics differ from NumPy's, the backend follows NumPy's:
  NaN in `sign` and `argmax`, integer floor division, float64 accuracy.
  So too the sign of a zero (owner, 2026-10-21, genclj-3mx6): Metal's
  float32 `atan(-0.0)` is +0.0, and the backend returns x itself at
  zero, as NumPy and the JVM backend do.
  So too a float32 sum's accuracy (owner, 2026-10-22, genclj-y0mp, found
  by memo.clj). MLX's CPU accumulates a float32 sum sequentially, so its
  error grows with the number of terms: 10⁶ terms of 10⁻⁶ sum to
  1 − 1.2e-3, where Metal and JAX give 1 to within an ulp or two. On the
  CPU device the backend sums float32 in blocks, then sums the block sums,
  so the error grows with log n, as with NumPy's pairwise sum. `mean`,
  `variance`, `logsumexp` and `softmax` are sums too. Metal and CUDA are
  unchanged.
  Blocks of 32 (owner, 2026-10-22, genclj-v3uw, from memo.clj). A sum of
  more than 256 terms is blocked; a shorter one is still MLX's own sum.
  The blocks are 32 terms, not 256. At either size each sum is within a
  few ulps, but an iterated computation compounds those errors:
  memo.clj's empowerment demo (five Blahut–Arimoto levels over 3125 ×
  169 × 169 cells) missed float64 by 1.89e-5 with blocks of 256 and by
  5.1e-6 with blocks of 32 (mean 1.2e-6). JAX's CPU misses by 3.6e-6
  and Metal by 2.2e-6. Blocking the sums of 33 to 256 terms too gave
  4.4e-6, at a cost to short sums that was not measured. A long sum
  takes 1.3–1.5× as long as with blocks of 256.
  Half floats (owner, 2026-10-22, genclj-y0mp). MLX's CPU accumulates
  float16 and bfloat16 sums in their own precision: 10⁵ terms of 10⁻⁵
  came to 1 − 0.75 in float16 and 1 − 0.998 in bfloat16. Blocks alone
  left −3.8e-4 and −4e-2. So on the CPU a half-float sum is computed in
  float32, in blocks, and cast back to its dtype, as JAX and XLA do.
  Metal, measured, sums half floats well (float16 within half an ulp,
  bfloat16 within 1.2 ulps), so its sums are unchanged. Float64 sums stay
  sequential, on the JVM backend and on MLX's CPU: about 1e-11 relative
  at 10⁶ terms.
- Configured by `gen.mlx.device` (`cpu`, `gpu`) and `gen.mlx.dtypes`
  (`32`, `64`).
- In `gen` bodies, a 0-d backend array counts as its host value for
  `case`, `=`, lookups and type tests, as it did for `if`, so plain
  Clojure in a body means the same on every backend. Helper functions
  outside `gen` must read values themselves.
- Distribution parameters given as host data are validated on the host,
  so also on MLX.
- Its Java binding is compiled by `clojure -T:build java` into
  `target/classes`, which is on the classpath. As a git dependency, its
  `:deps/prep-lib` builds it.

### Exact enumeration

(2026-10-19, moved early at the owner's request, for
GenGDT; bean genclj-3jp2): `gen.inference.enumerative`.
- Gen.jl's `enumerative_inference` and `choice_vol_grid`, as in 0.4.8,
  with the grid in row-major order: the last spec varies fastest, where
  Julia's iterator varies the first.
- Exact latents inside a model are an internal proposal, not a new
  density. `with-exact-proposal` keeps a model's distribution and traces
  and draws the named finite choices from their exact conditional (thesis
  pp. 145–146, the route §2 intends for L3). Generate's weight is then
  log p(observations) exactly when the other choices are observed, and
  regenerate on those choices is Gibbs. In an Unfold each step's weight
  is exact given its state, so the total is exact only when the next
  state does not depend on the enumerated choices; otherwise it is an
  unbiased estimate that changes with the key (caught by a side agent's
  review, 2026-10-19). `{:check-retval? true}` makes that case an error,
  for uses that need a known answer.
- Not GenMLX's `exact.cljs`: a second handler transition, "collapsed"
  traces (no choices, a probability table as return value) and the memo
  API.
- It satisfies the general GFI laws, not the forward-sampling ones.
  Supports come from the distributions. A structure or support that
  depends on the enumerated choices is a loud error, as is a choice made
  by a callee at an address (wrap the callee instead). Batched, it is the
  same code under the batch tracer.

### CUDA on the Jetson AGX Thor (sm_110): MLX built from its source

(2026-10-19). The `mlx-cuda-13` wheel has code for sm_75 to sm_121 and
PTX only for compute_120, so nothing in it runs on compute capability 11.0
("invalid resource handle" at the first kernel). `native/build-mlx.sh
cuda13-source` builds upstream v0.32.2, unpatched, for the machine's own
architecture (six minutes on the Thor); the owner's GenMLX fork carries
further Thor patches (a full-pool allocator, graph limits, an NVRTC cache),
for capacity and speed, not needed for correctness. MLX compiles some
kernels at run time with NVRTC from headers beside its library, which the
script copies. On CUDA every evaluation runs on one thread, since MLX's
CUDA encoders are per thread (a cross-thread evaluation failed, and then
left the stream broken).

### MLX's CPU compile, where the system's g++ cannot build MLX's kernels, falls back to the identity

(2026-10-19, found on the Jetson Thor: GCC 13
rejects the header MLX's wheel ships). Probed once, with a warning; values
are unchanged, only unfused. Not patched in MLX, and no g++ wrapper: a
watch bean (genclj-hjzh) revisits it with newer wheels.

### Pushforward distributions, Gen.jl's `@dist`

(M4, 2026-10-19):
`(gen.dist.dsl/dist name [params] body)` expands into a `defdist`. The body
makes exactly one draw (a call of a `defdist` constructor, which the macro
finds by var metadata). A discrete draw's density sums the mass of the
support values mapped to v, so any function works; a continuous draw's
inverts the path to it through +, −, ×, ÷, negate, exp and log, with the
log-Jacobians. The name `dist` follows Gen.jl's macro (owner, 2026-10-20).
Labels need not be numbers (genclj-414l, G7). A `defdist` parameter
marked `^:host` may be host data (`dc/host-data?`: keywords, strings,
records, maps, functions, collections of them), kept as given, while
array data still becomes an array; every other parameter converts as
before, so a family given keywords fails at construction with an ex-info
naming the distribution and the parameter. Every parameter of a `dist` is
`^:host`, since its body decides what it is, as in Gen.jl's untyped
`@dist`: one `labeled-cat` takes numeric labels (arrays, batched) and
keywords. The body is guarded as a `gen` body is: a lookup by the draw
reads its host value on every backend, and raises the batched fallback
on a batched draw. Over an enumerable support, a label's density
compares by `=` (`array/==` where both values are numbers), so a given
label is scored on the host, without the fallback, and the gradient
reaches the probabilities; a label itself has none (Gen.jl's
`nothing`). An array label is compared whole, so a vector label is one
value (the post-M5 review's P2-03: elementwise, a coin broadcast to
[1 2] counted twice). A draw without an enumerable support keeps its own
shape: a body that broadcasts it is an error, as such values have no
density. Its inverse is kept within a few ulps and a quarter of the
lattice's spacing (P2-04: with a fixed relative 1e-5, `(* 1e-6 (poisson
3))`'s masses over a grid of [0, 1e-5] summed to 97.4). A host parameter
is the same for every particle; a draw of
labels in a batched run falls back to one particle at a time, and
`(sample d key [n])` of labels is an ex-info, as no array holds them.
The alternatives: conversion by shape for every parameter, which would
move a numeric family's error for keyword data from construction to a
bare "not an array" at first use (rule 6); and a mark on each host
parameter of a `dist`, an annotation Gen.jl's untyped `@dist` does not
need, since a `dist` parameter is read only by the body and by the
draw's constructor, which validates its own (approved under the owner's
delegation, 2026-10-21).

### The internal-proposal override

(thesis Alg. 16; M4, 2026-10-19):
`(gen.inference.proposal/with-internal-proposal gf q)`. q is a dynamic
generative function called as `(q constraints & args)`; it proposes some
of the open choices at gf's addresses, and gf's forward sampling makes the
rest, so generate's weight is log p(τ) − log q(τ_q | σ) − log f(τ_f | σ, τ_q).
project replays q against a trace (its choices taken from the trace, as
constraints are) for its density and its addresses; a q that calls another
generative function at an address cannot be replayed, a loud error. Two
general laws came with it, holding for every internal proposal:
generate's weight is the projection on the constraints, and regenerate's
is the ratio of projections on the retained choices (Eq. 4.1). Before
them the general set checked project only on everything, which missed a
planted bug. Regenerate kept every unselected choice of the trace, a
move that would drop one a loud error (the review, 2026-10-20; kept by
the owner the same day), until the book's chapter 031 showed the laws
failing by exception on the thesis's own example (genclj-bnlx,
2026-11-04, in 1.1's scope). Regenerate now follows Eq. 4.1: σ = τ|_{S^c},
q proposes given all of σ, the new run drops the choices of σ it leaves
unvisited (B ⊆ A_σ), and the reverse move proposes given σ' = τ'|_{S^c}.
**Confirmed by the owner (2026-11-04 evening):** q sees all of σ, the dropped choices
included, and the reverse move all of σ', as the equation reads, so
selection MH stays stationary whatever q reads. The ratio of projections
(the general law) is that weight when q reads only the constraints its
run keeps (forward sampling, the thesis's example); for a q that reads a
choice the move drops it is not, and the law does not hold
(proposal-test's reads-what-it-drops). A revise (`gfi/generate-again`)
still reruns q without the drops, as before. `with-exact-proposal` and
`with-conjugate-proposal` keep the loud error; their proposals read only
the constraints their runs visit, so their reverse weight leaves out the
choices of σ' the old trace lacks.

### gen.array's alias is `array`

(owner, 2026-10-19): `(array/shape x)`,
`(array/item x)`, the namespace's last segment, as the Clojure style guide
has it. It replaced `a`, which the owner found ugly (2026-10-18). The
operators and math stay referred in place of `clojure.core`'s, the
convention of Clojure's numeric libraries (`clojure.core.matrix.operators`,
fastmath's `use-primitive-operators`), so a model reads
`(+ (* slope x) intercept)`, never `(array/+ (array/* slope x) intercept)`.

### Three ways to run, all first-class

(owner, 2026-10-19). The pure-JVM
backend (always available, no native dependency), MLX on the CPU, and MLX
on the GPU (Metal, CUDA). Each passes the conformance suite and the test
battery, a user picks per model (`-Dgen.backend`, `-Dgen.mlx.device`, or
`array/with-backend` for a region), and the core assumes none of them.
- The pure-JVM backend gets CPU speed work on its own terms (bean
  genclj-exha). MLX's CPU device is an alternative and a yardstick for it,
  never a requirement.
- Native performance libraries (BLAS/LAPACK through Accelerate or
  OpenBLAS, and others) are welcome later as optional plug-ins where they
  give real speedups (bean genclj-w4kw), never needed to run.

### Casts to integers are Java's

(owner, 2026-10-20; genclj-kw88). On
every backend, a float cast to an integer dtype truncates toward zero,
saturates at the dtype's range (±Inf too) and takes NaN to 0. An integer
cast to a narrower one wraps, as Java's `(int)` of a long does. Before
this, the backends disagreed on NaN, ±Inf and out-of-range values. The
JVM backend threw on ±Inf and did not clamp int32. MLX's GPU saturated
but took NaN to MIN for int64. MLX's x86 CPU gave MIN for all of them
(the M2 review, finding 11, confirmed on the RTX-6000 machine). A loud error would need
a host read of every cast on a lazy backend.

### Booleans in arithmetic count, as in Julia

(owner, 2026-10-20;
genclj-6pnx): `+`, `-`, `*` and `negate` on booleans give integers,
`(+ true true)` is 2, `(negate true)` is -1. With numbers and in
reductions, NumPy, JAX and Julia agree. Unlike NumPy and JAX, whose `+`
and `*` on two booleans are logical or and and, and `-` an error. The
reasons:
- Gen.jl models are Julia code.
- Code ported from GenMLX counts and gates with arithmetic on bernoulli
  draws, which were 0/1 there. Under NumPy's rule a count would become a
  logical or, silently.
- Julia's rule agrees with `sum`, where NumPy's `+` does not.

`(* true true)` is the integer 1 (Julia keeps a Bool), the same value
under one rule.

### A draw beyond the float range is +Inf

(owner, 2026-10-20), with log
density -Inf, exact for the value stored, as IEEE floats and Julia's
Distributions in Float32 give. In float32 this is common at small shapes:
about 9 % of weibull(0.01, 1)'s draws pass Float/MAX_VALUE, and Pareto's
docstring already said so. A cap at the largest float would give finite
densities at the price of a silent atom. A loud error would need a host
read of each draw on a lazy backend, or a threshold on the parameters.
The other edge stays as it is: a draw that underflows to 0, where a
density can be +Inf, is floored at the dtype's smallest normal number.
That holds in every family, gamma, beta and dirichlet included, which
review 4's floor missed: gamma(0.001, 1000) scored +Inf on 963 of 2,000
draws. Its mirror (owner, 2026-10-22, the post-M5 review's P2-01): a
beta draw that rounds to 1 is clamped to the largest float below 1. At
beta(0.001, 0.001), 8,802 of 20,000 draws rounded to 1 and scored +Inf.
Beta and dirichlet are drawn from their gammas' logs (the sigmoid of a
difference, a softmax), so the floor and the clamp see where a draw
lies: x/(x + y) of two underflowed gammas was 0/0 = NaN (448 of 2,000
draws), where flooring the gammas first would put the draw at 1/2. A
shape below 1 boosts in log space, log Γ(α+1)'s draw - E/α for an
exponential E (genclj-uli3). Every beta and dirichlet draw changes for a
given key (confirmed by the owner, 2026-10-22; genclj-uli3). The
old draws could be kept for α ≥ 1, at the cost of a second sampler.

### Differences from Gen.jl's test suite

(owner, 2026-10-20, after the
M5 inventory, `docs/research/genjl-test-suite.md` §4):
- *Retdiffs are more precise (D1).* A dynamic function's retdiff is
  no-change when its return value is the previous one, where Gen.jl's
  dynamic language always says UnknownChange. So an unchanged Map or
  Unfold element is absent from a VectorDiff, and a splice turns an
  argdiff into no-change for an argument identical to the previous call's.
  This is sound, since a diff claims only what holds. The ports assert
  gen.clj's diffs, and that they are consistent (`diff/consistent?`).
  *Unchanged* means `gfi/same-value?` (owner, 2026-10-22, the post-M5
  review's P1-01 and P1-03): by class and order, so −0.0 differs from
  0.0 and a vector from a list, and bounded. Past its budget, or for a
  lazy value not yet realized, the answer is unknown-change: sound, at
  the price of this precision there. An array return value is the same
  only as itself (same-value? reads no array), so one rebuilt equal is
  unknown-change. With `clojure.core/=` the retdiff, `diff/compute` and
  `diff/consistent?` gave wrong weights (−6.283 where −4π is right) and
  hung on an infinite lazy return value.
- *Lookups are loud (D4).* `cm/get-choice`, `cm/get-value` and
  `(tr addr)` at a missing or node address throw ex-info, as Gen.jl
  throws KeyError, where they returned nil (rule 6). Built in M5, with a
  not-found arity for the lenient lookup (the bullet "Lookups are loud"
  below).
- *Beta is exact at 0 and 1 (D6):* +Inf when α < 1 (or β < 1), where
  Gen.jl substitutes a finite value. As with draws beyond the float
  range, gen.clj keeps the exact value.
- *Regenerate's weight is Eq. 4.1's, under a Switch too (U1;
  genclj-c5rl).* A choice resampled from its prior adds nothing, and an
  old choice the new run drops cancels against the reverse move, so the
  weight is the ratio of the densities of the choices kept. Gen.jl
  0.4.8's Switch, with an index argdiff other than NoChange (the path of
  every Switch called from `@gen` code), weighs the change of score
  instead: it adds log p(z′) − log p(z) for the resampled choices and
  subtracts a dropped choice's log density, so selection MH on a choice
  under it targets p(z)² p(y | z). switch.jl:85, :89, :94, :288, :292 and
  :294 assert Gen.jl's formula; the port checks them on Gen.jl's own
  moves, which the oracle recorded, and Eq. 4.1 on gen.clj's. Reporting
  upstream is the owner's call (the draft is
  `docs/research/genjl-upstream/issue-1-switch-regenerate.md`).
- *No default positional arguments (G3).* Clojure has none. A rest
  parameter (`[x & [y z]]`) covers calls, update and regenerate with any
  number of arguments, and a trace's args are the ones given. A change
  in the number of arguments is a change, whatever the argdiffs say: a
  rest parameter's diff, an estimated density and the internal
  proposal's regenerate compare the counts (the post-M5 review's P1-02:
  they kept stale work). A multi-arity `gen` stays possible if parity is
  wanted later.
- *The static IR's internal shape is not ported (G15).* Gen.jl's node
  kinds (argument, parameter and per-expression nodes) are its Julia
  representation. `static/ir` has one node per binding by design, and the
  ported tests check behavior, not the IR's form.
- *A covariance singular in floating point is an error (F1).* mvnormal
  rejects a covariance whose Cholesky factor fails, or whose condition
  leaves the density no correct digits: trace(H⁻¹) ≥ 1/ε, where H is the
  correlation matrix of L·Lᵀ (between 1/λmin and n/λmin) and ε is that of
  the dtype the density is computed in (float32 on MLX, host data
  included). The bound is the measured one, not the worst case's: a
  float32 log density's error tracks (0.05–0.4)·ε·trace(H⁻¹) on squared-
  exponential kernels of 50 to 1,000 points, so the bound rejects where
  the error nears 0.1 nats. The worst case's 1/(n·ε), the first version,
  rejected kernels accurate to 1e-4 nats (n = 1,000, jitter 0.1), the
  owner's correction the same night. Gen.jl, through LAPACK's potrf, rejects only a failed
  factorization, so it accepts `2·ones(2, 2)` (a pivot positive by
  rounding) and gives 15.24. dist_dsl.jl:29's case is ported as the
  error. The check runs on host values and concrete arrays only (rule 5).
  The same rule holds at every precision (the owner, the same night): in
  float32 it also rejects matrices that are positive definite but whose
  float32 density is off by tenths of a nat or more, such as a
  1,000-point GP kernel with a jitter of 1e-5 (0.75 nats). The error
  names the dtype and suggests a jitter at its scale (2e-4 there).

### Distributions from a host library, GenDistributions' `DistributionsBacked`

(M4, 2026-10-20; decided by the agent, for the
owner's review): `(gen.dist.host/family {:name :make :quantile
:log-density})` makes a family of univariate distributions from a host
object built from the parameters, its quantile function and its log
density (and `:discrete?`, `:support`). A draw is the quantile of a
Threefry uniform in (0, 1), so it is a pure function of the key (rule 2)
and the same on every backend; a library's own random generator is never
used. Parameters and values are read on the host, as the guarded
lookups read them: a batched one raises the batched fallback, as does a
draw in a batched run (it would be every particle's), one being
differentiated is a loud error (the library has no gradient), and a lazy
backend evaluates an array to read it, the cost of a host library.
Apache Commons Math 3 is the JVM's Distributions.jl: `gen.dist.commons-math`
(`real` and `integer` for any of its distributions, and its discrete
hypergeometric and zipf) lives in `ext/commons-math` behind the
`:commons-math` alias, outside the core's dependencies (§8). Some of its
quantiles come from a solver, to about 1e-9. A host library's draws read
their parameters on the host, fall back from batched runs and have no
gradients, so the continuous families Distributions.jl has beyond Gen.jl's
built-ins are native in gen.dist instead, with its names and parameters:
chi-squared, f-dist, gumbel, levy, log-normal, logistic, nakagami, pareto,
student-t, triangular and weibull, which batch, run on every backend and
differentiate (checked against scipy's densities and, in the extension's
tests, against Commons Math's).

### Custom gradients, `array/custom-vjp`

(M4, 2026-10-20; one of §8's
reserved names): `(array/custom-vjp f vjp)` is f, differentiated through
by `(vjp args out g)`, one cotangent per argument. It is an optional
backend protocol (`ICustomVjp`): the tape records it as one node, MLX's
native mode makes it one `mlx_custom_vjp`, and elsewhere it is f. On it
rests Gen.jl's CustomGradientGF: `gen.custom/deterministic` with
`:gradient` (or CustomDetermGF's `:gradient-with-state`) runs its
function on the arguments' values, so the function may be host code,
and gives the tape the rule, a tree argument's gradient spread over its
leaves. Single outputs; cotangents must have their arguments' shapes;
an f that closes over a value being differentiated is an error (its
gradient would be lost); in native mode f runs inside MLX's
transformation and may not read values on the host, and inside
`array/compile` a custom VJP under native gradients is an error (MLX's
compiled graphs leave custom VJPs out).

### Trace transforms

(M4, 2026-10-20): `gen.inference.transform`, Gen.jl's
@transform as `(deftransform name [model-in aux-in model-out aux-out]
body)`, the body calling `read`, `write` and `copy` with Gen.jl's tags
(the four names are bound to the spaces, as in Gen.jl's header). The
body is run once for its outputs and again under `array/grad` for the
Jacobian of its continuous part, the continuous reads substituted;
outputs are the continuous writes, copies of continuous reads, and the
continuous reads an update keeps unwritten (so an involution may read a
choice it leaves alone). Translators are `(fn [trace key] → {:trace
:weight})`, so a particle filter takes one as a move-reweight kernel;
the symmetric translator is `involution`, the function
`mh/involutive-mh` already took. An involution must write every new
choice: its update samples nothing, a loud error otherwise. The general
translator subtracts the forward proposal's score (Eq. 3.12); Gen.jl's
GeneralTraceTranslator adds it (src/inference/trace_translators.jl on
master, while its SymmetricTraceTranslator subtracts), a sign bug the
change-of-variables test, whose weight is exactly 0, would catch. SMCP3 is
GenParticleFilters' UpdatingTraceTranslator with a transform
(`updating-translator`), applied to a filter by `pf/move-reweight`;
without a transform it is Del Moral's SMC samplers.

### Kernels that read the trace, `(kern ^:trace [tr & args] …)`

(owner,
2026-10-20, after the final law audit): Gen.jl's `@kern` reads the trace
in its loops, branches and lets, as the thesis's p. 102 kernel loops over
`trace[:degree]`. `kern` builds its chain from its arguments alone, so it
can't. With `^:trace` on its parameters, the first parameter is bound to
the trace as it is when each statement is evaluated, a kernel statement
applies its kernel and continues with the result, and the reversal runs
the statements reversed (loops backwards, each kernel's reversal). The
thesis's invariant (pp. 100–101) is checked per construct, as Gen.jl
checks it: after the body of each loop, branch and let, what the
construct read (its collections, its condition, its values) is read again
and must be unchanged, or it is a loud error (the reversal would not undo
the kernel), and the reversal checks its constructs the same way. A later
statement may change what an earlier construct read: the p. 102 kernel's
structure move changes the `:degree` its first loop read. Prop. 3.4.1
asks only that each construct's own body keep it (the reversal's argument
is at `trace-kernel`). Gen.jl checks only with `check=true`; gen.clj
always checks. Values compare on the host (kernels run at the inference
loop's boundary): host data by `=`, with no budget, and a fn, record or
generative function by what it holds (its closed-over values, its
fields, its kind's rule), so a kernel built again from the same values
is the same kernel (the post-M5 review's P3-06: with `=`, a kernel bound
in a let was a violation). A kernel statement whose arguments read the
trace is a dependent mixture too, so it is checked as a let of its
kernel (P3-02: unchecked, as in the thesis and Gen.jl,
`(mh (select (if (tr :flag) :x :flag)))` made flag absorbing).
(Corrected after the confirmation read of the M4 tests, C1: the first
check compared every construct with the final trace, which rejected the
p. 102 kernel in its own order.)

### Trace kernels, GenTraceKernelDSL

(M4, 2026-10-20): no new macro. A
kernel is a `gen` function whose first argument is the model trace and
whose return value is `[update backward]`; its choices are the auxiliary
randomness, so propose and assess give the forward and backward scores.
log |det J| reruns the kernel's assess with the trace's and the kernel's
continuous choices substituted by tracked values, under `array/grad`,
on the trace generate makes from the substituted choices, so what the
kernel reads from a call's value, the return value or the score is
differentiated too (the post-M5 review's P3-04: with the old trace's
call values, log |det J| was log 1.3 off, silently). Only the kernel's
assess needs exact densities. The model's generate may draw
encapsulated randomness, at key 0, so pseudo-marginal moves work, as
involutions do; a kernel whose outputs depend on that randomness (the
score of an estimated density, say) fails a check of the outputs at the
trace's own choices, loudly (confirmed by the owner, 2026-10-22;
m3-grammar's cross-review, genclj-1ufo).
It leaves out only the trace's choices no output depends on and the
move leaves alone: a choice the update rewrites or discards is an input,
less an observation SMCP3's `:new-observations` replace, which is data
(P3-07: a kernel that overwrote mu with fresh noise weighed as a
bijection, E[exp w] ≈ 7 where 1 is right). A transform's translators
check the same of its unread inputs.
GenTraceKernelDSL also drops constant outputs and kernel choices no
output uses, which hides moves that are not bijections (found by the law
audit, 2026-10-20), so here they make the matrix non-square or singular,
a loud error. A continuous choice the update keeps but an output depends
on counts as an unchanged output. Calls of other generative functions
return their values (GenTraceKernelDSL returns their traces).

### Transform bodies are eager

(2026-10-20): a lazy call (for, map, …)
holding a write, copy or tcall is an error when the transform is
defined (use doseq, run! or mapv), as `gen` bodies run to completion.
A discrete read returns a host value, so the body branches with a plain
`if`: on MLX a sampled boolean is an array, always truthy, which sent a
random walk down the merge branch (the full battery on Metal, the same
night).

### Recurse

(M4, 2026-10-20): `(comb/recurse production aggregation
max-branch)` as Gen.jl's, with 0-based nodes (root 0, child j of node i
at i·max-branch + j + 1), choices at `[i :production]` and
`[i :aggregation]`, the production kernel returning `{:value v :children
[inputs]}`, and one argument (no `root_idx`). An update's new nodes add
their generate weights, where Gen.jl's code adds their scores, which is
the same only when every new choice is constrained; regenerate is
implemented too (Gen.jl has none).

### Encapsulated randomness

(thesis §4.5; owner, 2026-10-20): a
generative function may own randomness ω outside its choice map, its
score an unbiased estimate ξ of its density (Eq. 4.3), and the trace
keeps ω. An update or regenerate keeps ω when nothing it depends on
changed (no-change argdiffs, no constraint or selection under it, and
the same store and gen-fn at the call: a revise draws it afresh), so an
identity update weighs exactly 0 and a call
that is skipped keeps its ω; otherwise it draws ω′ with the operation's
key and weighs log ξ′ − log ξ (regenerate less its proposal ratio). The
thesis redraws ω at every update (§4.5.1), so its identity updates need
not weigh 0. Both are exact: keeping ω is the null move on (τ, ω),
redrawing it the pseudo-marginal refresh. This rule, GenMLX's
(`encapsulated.cljs`), keeps the identity laws (an identity update
weighs 0, regenerating nothing changes nothing) for every generative
function, lets Map, Unfold and the static language skip unchanged calls,
and reruns an estimator only when its inputs change. Laws that compare
separate runs (assess against a trace's score, generate with every
choice against it, propose against assess, an update and its reverse)
hold only in expectation for a function with ω, under either rule, since
each run draws its own ω, except that simulate and assess on one key draw
the same ω for untraced draws (a law). Such functions get their own law
set: the pointwise laws without those, and the thesis's Eqs. 4.3–4.16 as
statistical laws.
- **In gen bodies (§4.5.2),** `(untraced dist)` draws without an address
  and `(untraced-key)` hands other stochastic code a key; untraced values
  add nothing to the score, which is then the choices' density given ω.
  ω is one key, folded from the operation's key apart from the sites'
  sequence (an untraced draw never shifts a site's key; the n-th draw
  takes ω folded with n), and the trace keeps it as `:gen.trace/omega`.
  The handler's weights are then exact with no new weight code: Eq. 4.7
  on the extended space of choices and untraced draws, the internal
  proposal of the choices conditioned on ω (the handler draws them
  interleaved), a different kernel from the ω-free Eq. 4.7 (regenerating
  the mixture's :a weighs exactly 0) with the same target. Untraced values must not decide which addresses are
  visited (p. 159): an obligation, as in the thesis and Gen.jl, since no
  sound check short of rerunning the body exists. A batched run falls
  back to particles, since a particle's ω must reproduce its own draws.
  `untraced` and `untraced-key` are reserved in bodies as `trace` is:
  the macro refuses a body that would hide a var of either name.
- **Plain calls in a body (genclj-zxis, 2026-10-24; the owner: fix).**
  `(gf & args)` in a body, Gen.jl's untraced call, ran gf with a key from
  the implicit source, so a run with a fixed key did not repeat,
  silently (found writing the book's chapter 012). The body makes the
  call now (`gen.trace/*body-call*`, bound while it runs): gf runs with
  the next key from ω, which the body takes only when gf drew (made
  choices, or keeps encapsulated randomness of its own). So a call that
  drew is an untraced draw in every respect: the run repeats with its
  key, ω is kept by an update that changes nothing, the score is an
  estimate (an error where exact densities are needed), and a batched
  run falls back. A call that drew nothing costs nothing: the score stays
  exact, the trace keeps no ω (gen.serialize writes it), and a batched run
  stays batched. Outside a body a plain call still takes the implicit
  key. The first recommendation, a loud error for plain calls, was
  dropped: Gen.jl allows them, and ω already gives them a meaning.
- **Host code branching on a boolean choice (genclj-x6ox, 2026-10-24;
  the owner: fix).** On MLX a 0-d boolean is an array object, which
  Clojure's `if` takes as true, so `(if (tr :b) …)` in host code (an
  inference loop, an analysis) took the true branch, silently. Outside a
  body, a trace called with an address returns a 0-d boolean as a host
  boolean (on a lazy backend the read evaluates it, at what is an
  inference-loop boundary). Inside a body, and for batched values,
  values stay arrays: gen's conditionals read them. Other host code that
  branches on a boolean array uses `array/truthy?`. Making trace lookups
  host values for every dtype was rejected: numbers branch alike on
  every backend, and inside bodies a read would force evaluation.
- **Estimators (§4.5.3),** `gen.custom/estimated`: `:addresses` (its
  choices), `:estimate` (fn [args choices key] → log ξ, ω drawn from the
  key, which the trace keeps) and an optional joint `:simulate`. generate
  with constraints needs all of the choices (any other set is a loud
  error: the law audit found a strict subset gave a trace missing one);
  without, it is simulate with weight 0, valid on the extended space. A
  batched run falls back to particles, as for untraced draws. regenerate
  with every choice selected draws them and ω afresh from the joint
  sampler, weight 0, as generate does (review 5, R5-6: a revise did so
  already, and the kind itself refused). regenerate of some of the
  choices only, and simulate without a joint sampler, are loud errors:
  an estimator has no proposal density to weigh them by. Pseudo-marginal
  MH and PMMH are then MH on a model with either.
- **assess takes a key,** as every other operation (rule 2), since it
  draws ω; with the same key it weighs what generate weighs on the same
  choices, a general law. `(assess gf args choices)` draws one from the
  implicit source.
- **Operations that need exact densities refuse ω** (`gfi/with-exact-density`):
  gradients (choice and parameter gradients, HMC, MALA, MAP, VI's and
  learning's gradients), elliptical slice, the exact proposal, an
  internal proposal's replayed density and a trace kernel's Jacobian.
  Each computed its density with a fixed key, so with ω it would fix one
  draw of it and target the wrong distribution, silently (found the same
  day, after an owner's question: all eight ran without error).
- **An involution takes a key** (owner, 2026-10-20): `(fn [trace
  fwd-choices fwd-retval proposal-args key])`, one argument past Gen.jl's
  shape, as every gen.clj operation takes a key. Without one, the review
  found, its update redrew ω from a fixed key and the chain sampled the
  posterior given one z; with the move's key the update is a
  pseudo-marginal refresh, so split-merge and reversible-jump moves work
  on models with encapsulated randomness (Gen.jl's global RNG gives it
  this too). `involutive-mh` splits its key in three (the proposal's,
  the acceptance's, the involution's), so the first two are unchanged.

### The static modeling language

(Gen.jl's `@gen (static)`; owner,
2026-10-20): a flag on `gen`, `(gen ^:static [args] (let [...] ret))`,
not a second macro or a second semantics. The body is an ordinary `gen`
body: one `let` whose bindings are calls at constant addresses (`trace`,
`splice`) or pure expressions of the arguments and earlier bindings (any
pure Clojure, as Gen.jl allows any Julia), then the return expression;
branching happens only inside called generative functions. The flag
makes the macro build the model's graph (as data, for inspection and a
DOT rendering) and throw at macroexpansion when the body is not static (a
branch at the top level, a computed address, `trace` or `splice` passed
as a value). The check is sound, since `trace` and `splice` are the
macro's own locals, unlike GenMLX's schema extraction over quoted source
by symbol name. simulate, generate, update and regenerate walk the graph
with the handler's own trace and splice (simulate and generate run every
node; update and regenerate rerun a node only when a parent changed or
it is constrained or selected), and assess runs the generated score fn. A law requires the static and the dynamic function of the
same body to give identical traces, weights and discards under the same
keys, and a sound retdiff (the static one may be more precise). The
graph is also the way
to compiled scores, batching without the tracer, and CUDA graphs: a fixed
structure runs the same kernels every time.
- **How it runs:** the macro builds one guarded fn per node, of `param`
  and the node's inputs (over-approximated from the symbols it uses, so
  a dependency is never missed), and the runner walks the nodes in order
  with the handler's own trace and splice. A skipped node keeps its value
  and its site's or call's record (weight 0, as an unchanged rerun gives)
  and consumes its key, so the walk draws what the dynamic body draws;
  the trace keeps every node's value for the next update, with the
  parameter store in effect. A node is skipped only when the previous
  trace is this graph's (not another static function's at the address, a
  Switch's other branch) and the store is the same (a callee reads it, an
  argument may), conditions the review of the same day found missing.
  Every value comes from the node's guarded fn, as the body computes it.
  A gf with middleware (`with-transitions`) reruns every node, so its
  transitions see every site; a kept callee's own middleware does not
  run. A trace node rerun because an input changed keeps its value, so
  what depends only on that value still skips. The analysis sees the
  enclosing locals, a gen body's too, so a local that shadows a var is
  the local.
- **Not static:** a body that is not one `let`, an address whose first
  key is computed, a `trace` or `splice` where it is not evaluated
  exactly once whenever its expression is (a branch, a function, a `let`
  inside an expression, an `and` or `or`: see the rewrites below), a
  splice without an address, an untraced draw, an address used twice or
  one under another: each a compile error that names the form.
- **Macros and quoted data (M5, G11 and G12; owner, 2026-10-21):** a
  binding's head, and the return expression's, is macroexpanded before
  it is classified, as `gen` expands a dynamic body, so a user macro that
  expands to a `trace` or a `splice` is that call; a computation is its
  expansion, whose names are its inputs (its form shown as written; the
  review of the evening found a head macro reading a local unbound).
  Only the head is expanded, so a call hidden in a macro inside an
  expression is the compiler's error (trace is unbound there), as before:
  catching it would take a walk that knows the expression's locals; and
  a macro nested so must be hygienic, reading only the locals it is
  given. Quoted data is data: it may mention `trace`, and its symbols are
  no inputs.
- **Static rewrites (M5, G2; owner, 2026-10-21; built in genclj-lgij).**
  The rewrites live in the static analysis only; the body fn, the dynamic
  semantics, is unchanged, and the static law runs over every rewritten
  body.
  - *A computed address is a node of its own kind (B1),* not Gen.jl's
    `call_at`/`choice_at` nodes: `(trace [:x i] d)` stays a trace node
    and `(splice [:x i] gf …)` a call node. The address is the longest
    leading run of constant keys (the first key must be one, as Gen.jl
    requires a Symbol) followed by keys computed from the node's inputs;
    it runs as the dynamic body does (one site, its key, its address);
    it is skipped only when its inputs, the keys' among them, are
    unchanged and no constraint or selection reaches its previous path;
    assess and the overlap check see its constant prefix, as for a
    splice. Gen.jl's rewrite into a splice would split the key, so
    `(gen [i] …)` and `(gen ^:static [i] …)` of one source would draw
    different values, and gen.dynamic would load gen.combinators.
  - *A call nested in an expression is hoisted* into a node bound to a
    fresh name, placed before the binding in Clojure's evaluation order
    (arguments left to right, the innermost call first), where it is
    evaluated exactly once: a function call's arguments, a vector, map
    or set literal's elements, and an `if`'s test (so `when`'s, and
    `cond`'s first). Not a nested `let`'s init. The hoisted call may not
    read a local bound inside the expression.
  - *A call in a branch* (an `if`'s, a later `cond` test, `and`, `or`)
    stays a compile error naming the call and why. Gen.jl hoists it and
    runs it whether or not the branch is taken, so its static function is
    a different model from the same dynamic body: a recorded difference
    (D16, `docs/research/genjl-test-suite.md`).
- **Compiled scores and a lint, in M4 (owner, 2026-10-20).** A static
  function's graph gives its log density as a pure function of its
  choices, which HMC, MALA, MAP and gradients use for static models, so
  `array/compile` can fuse a whole score (genclj-rj78). A lint says which
  dynamic functions the flag would take, so a model gets the static
  language's speed where it fits without an implicit switch
  (genclj-ef7r): GenMLX's automatic analysis of every body is what the
  triage found least reliable.
  - *Scores:* the macro generates one straight-line fn per static body,
    `(fn [param args choices key] → [weight retval])`, the bindings as
    locals: a trace node's value read from the choices with its log
    density, a splice's callee assessed (a static callee by its own score
    fn, under its own store, with the handler's checks of its choices and
    arguments). A static gf's `assess` is that fn, so everything built on assess
    (HMC, MALA, MAP, choice gradients, MH's backward densities) uses it.
    Keys are split as the handler's assess splits them, so a callee's
    encapsulated randomness draws the same; under `with-exact-density`,
    where a draw is an error anyway, none is split. On the JVM backend,
    idle, assess is 1.9× faster at 10 observations and 2.6× at 100;
    gradients, HMC and MALA gain 1.15–1.3×, their cost being autodiff's
    (`docs/benchmarks.md`), and on MLX a
    compiled step was one fused graph already (`array/compile` traced the
    handler), so there the gain is the trace, not the step.
  - *The lint:* `(static/could-be-static gf)` runs the flag's analysis
    over a dynamic gf's source in its namespace; `(static/lint ns …)`
    reports every generative function var. A call in return position,
    `(let [...] (trace :y …))`, is static too: a last node, its value
    returned.

### Change hints and Diffed propagation

(Gen.jl's IntDiff, SetDiff,
DictDiff and Diffed; M4, 2026-10-20): `gen.diff` adds `int-diff`,
`set-diff` and `dict-diff` beside `vector-diff` (each collapsing to
no-change when empty), and `consistent?`, whether a diff describes the
change between two values, which the laws' retdiff soundness now uses
for every kind. `(diff/propagate f args diffs)` is an open multimethod on
the function's var giving the diff of a call, with rules for `count`,
`nth`, `get`, `conj`, `assoc`, `+`, `-` (clojure.core's and gen.array's),
`inc`, `dec`, `vec` and `identity`; Gen.jl dispatches on a Diffed value
type, which Clojure's core functions do not take, so here the rule
belongs to the call. In a static body every name carries a diff: an
argument's from its argdiff, a binding that is one call of a var on
symbols and constants from its rule, a bare symbol's from its input, a
splice's from its callee's retdiff, anything else no-change or
unknown-change. A splice
hands its callee its arguments' diffs (an identical argument stays
no-change), so a Map reruns only the elements a vector-diff names, and
the function's retdiff is its return expression's. A rule gives the diff
from the previous call's result to this one's, for its own arity only,
and an element's diff only for a collection of its diff's kind, with a
key of a kind the rule can compare (a backend array, which the guarded
lookups read on the host, and a number that is not an integer, which
`nth` truncates, give unknown-change); the
first rules gave the diff from the argument to the result, so a changed
conj or assoc value went unseen (the review, the same day).
`(diff/compute prev now)` gives the precise diff of two host values.
Every rule is checked at every argument position against random edits
(a changed element, key or default, an append or a shrink, a new or
removed key, a fresh value). The static law runs with computed argdiffs
on a model where each rule's result is read by a choice or a Map, so a
rule that understates a change alters a weight; one that overstates it
costs only a rerun, which the diff tests catch.

### The Branching protocol and the llama.cpp engine

(2026-10-20, agent;
approved by the owner the same day; `docs/research/llm-branching.md`):
- A branch is a value (its tokens and next-token log-probabilities,
  float arrays normalized in double) holding a state, a KV-cache
  sequence. advance, extend and resample consume their branches and
  return successors, so a stale branch is a loud error, never another
  state; fork and score leave theirs alone.
- dispose! frees at once and is loud on a branch that is not live; a
  live branch dropped unreachable is freed by a phantom reference at the
  engine's next operation, and a GC runs when sequences or KV cells run
  short (traces will hold branches). An unreachable engine is closed.
  The GC is a last resort: resample frees the dead itself, and with
  `-XX:+DisableExplicitGC` an exhausted engine is a loud error. Memory
  management stays in the native membrane, out of semantics code (rule
  3), as with MLX's arrays.
- One platform thread per engine; the calls are synchronous.
- score reads into a scratch copy, so no sequence ever loses its end
  (which hybrid models cannot undo); the cache is always unified (forks
  copy metadata), sliding-window layers full-size; at most 255 branches.
- `extend`, a teacher-forced advance with its scores, is added to the
  list above; `advance` is the batch operation, `dispose!` has a bang.
- The engine layer computes on host floats with clojure.core, not
  gen.array (it sits below the GFI, like a backend).
- **Engines never sample.** They give log-probabilities only, so a token
  is drawn by gen.random with its site's key (rule 2); no engine adds
  llama.cpp's sampler.
- *Open, for genclj-0cvv:* scores depend on batch shape (up to 5e-2
  nats a token), so law 2 (score = assess's weight) holds exactly only
  if simulate and assess compute a sequence in the same shapes;
  otherwise it needs a band. The agent's lean is exact; the owner
  decides when that bean starts.

### The LM protocol

(2026-10-20, agent; approved by the owner the same
night, every recommendation below; `docs/research/llm-lm-protocol.md`):
- Callers hold tokens, never branches. The LM's cache owns every live
  branch, each the entry of its context in a trie of what the engine has
  read; a context is read from the deepest held prefix (forked, so it
  stays) or from nothing. Eviction is `dispose!` under bounds
  (`:max-branches`, by default every branch the engine has free;
  `:max-cells`, by default `:n-ctx`), least recently used, and for cells
  among the branches whose disposal frees cells; a read consumes its
  source only when branches are short. No branch is dropped, so the
  engine's GC backstop never runs: `(info engine)` counts its collections
  (`:collections`, added to `gen.llm.slots`), and the tests require none.
- Batching is explicit: `batch-next-logprobs` reads a call's one-token
  reads in one advance, so a token SMC step is one decode. Recommended:
  no autobatcher, since coalescing callers by timing would let thread
  timing choose batch composition and so llama.cpp's results; if one is
  ever needed, a lockstep batcher in worker order, which stays
  deterministic.
- The cache's choices are a function of its calls, so the same calls on a
  new engine give the same results, bit for bit (tested on a toy engine
  whose results depend on batch shape and cell placement, as
  llama.cpp's).
- Recommended value shape: `next-logprobs` returns a new host float array
  (the engine's form, normalized on the host already, read there by byte
  marginals, masks and weights; the copy is 17 µs at Qwen3's 151,936
  tokens, against a 5.8 ms decode, and humongous below a 4 GB heap but
  reclaimed at no measurable cost), and the token GF converts at its
  site. A held entry costs its row and about 7 KiB, so the cache holds
  at most `:max-branches` rows; recommended: LLM work on a heap of 8 GB
  or more, or 4 MB G1 regions, where no row is humongous. Measured for
  genclj-0cvv: a `categorical-logits` draw at that size costs 4.6 ms on
  the JVM backend (Gumbel-max), a host inverse-CDF draw with one
  gen.random uniform about 335 µs on average over keys (corrected
  2026-10-21: the 78 µs first given was one key's; the draw scans the
  row in token order, half of it on average).
- Every token read keeps the score it was read with (the first read
  wins), so `score` along what was read reads nothing (assess after
  simulate, token SMC's prefix potentials). A fresh context is read as its
  first token and then teacher-forced, so that each of its tokens is
  scored.
- A tokenizer is a map: the vocabulary's size and special tokens, and
  `:token-bytes`, `:encode`, `:decode`; llama.cpp's is `llama/tokenizer`.

### The token generative function's foundations

(owner, 2026-10-20,
before genclj-0cvv starts):
- *Score once.* Each token's score is computed once and kept by the LM
  protocol's trie (the first read wins), as GenMLX's score-once ledger.
  simulate, assess, update and regenerate read it, so the GFI laws hold
  exactly within one LM. Across fresh LMs, or after an evicted row is read
  again, values agree within the measured band (llama_noise: 0.1 nats a
  token on Metal; long contexts and CPU builds need their own). Every token
  is computed in whatever batch shape its call gives, so this is the one
  exactness llama.cpp allows without giving up batching.
- *The trie owns the branches.* Traces hold tokens, never a branch; the LM
  protocol's cache owns every live branch and frees by explicit `dispose!`,
  so the engine's GC backstop is never on an inference path.
- *Host draws.* A token is drawn by a distribution in `gen.llm`, by inverse
  CDF over the row from one uniform that gen.random derives from the
  site's key (rule 2 holds; draws are deterministic), and its log
  probability is read from the row. It is the one exception to rule 3
  ("core code calls only gen.array"), confined to the LLM layer, which
  computes on host floats as the engine does: a `categorical-logits` draw
  over 151,936 tokens costs 4.6 ms, about a decode, a host draw 78 µs
  [corrected 2026-10-21, agent: about 335 µs on average over keys, the
  78 µs being one key's; still a fourteenth of a `categorical-logits`
  draw].

### The token generative function

(2026-10-21, agent; confirmed by the
owner, 2026-10-21; `docs/research/llm-token-gf.md`):
- `gen.llm.token/completion` over an LM is a generative function of
  `[prompt max-tokens]` (a token vector, a non-negative integer, both
  arguments so update can change them; together within the LM's
  `:limit`, checked before any read). Token i is at address i, as
  Map's and Unfold's elements are; a spliced completion's tokens are
  `[:story 0]`, `[:story 1]`, …
- An end token is a choice: drawn, scored, kept, and it ends the
  completion. The end tokens default to the LM's `:eog`; `:end-tokens`
  replaces them, `#{}` gives exactly max-tokens. The return value is the
  token vector, the end token included.
- **Confirmed by the owner, 2026-10-21:** the option is
  `:end-tokens`, not `:eos` as first built. It is a set, while
  `(lm/vocab m)`'s `:eos` is a single token, so
  `{:eos #{(:eos (lm/vocab m))}}` read as a pun (review-lm's
  recommendation). Alternatives: `:eog`, the vocabulary's name for its
  set; `:eos` as before.
- Scores are the LM's ledger's (`lm/score`), never the drawn row's; a
  trace keeps them, and update and regenerate keep them for every token
  whose context is unchanged (a constraint equal to the token there, or a
  redraw equal to it, changes nothing) and rescore the tokens after the
  first change, a run of them in one LM call. So the laws hold exactly
  within one LM while it holds its rows, and a revision that changes no
  token weighs exactly 0 after any eviction (corrected 2026-10-21: the
  first build kept scores only before the first constraint or selection,
  so a constraint equal to its token rescored the rest, review-lm M1).
- `token/dist` is the host distribution (usable at any site); token i is
  drawn from `(fold-in key i)`. It takes an integral double as its
  token (3.0 is token 3), as gen.dist's discrete families do (owner,
  2026-10-22, the post-M5 review's P4-06). A draw is loud when it finds
  its row's probabilities not summing to 1 (within 1e-4), at no cost:
  the walk holds the total. Under a batch tracer the operations raise
  the batched-branch error: gen.batched falls back to one particle at a
  time. Native batching belongs to token SMC (one
  `lm/batch-next-logprobs` per step), since batched traces cannot hold
  completions of different lengths.
- `token/forced` builds constraints (a prefix, text), `token/from` the
  suffix selection; no temperature or top-k (another distribution, a
  later option).
- On M4's GFI (m3-main, 2026-10-21): unvisited constraints are
  `gfi/unvisited` errors, every address at once, so a revise drops old
  tokens past a new end; a completion keeps `IRevisable`'s defaults (no
  store; another completion's trace is revised from its choices), and
  satisfies the general, forward-sampling, fixed-support and across law
  sets.

### Token SMC and potentials

(2026-10-21, agent; confirmed by the owner,
2026-10-21; `docs/research/llm-token-smc.md`):
- A potential (`gen.llm.potential`) has `prefix` and `complete` log
  weights of a token context and `logw-next`, a float array of n-vocab
  + 1 entries (the end an index past the vocabulary), consistent as
  genlm-control's. Contexts are the generated tokens; an LM's end token
  stands for the end.
- `gen.llm.smc` takes a completion and its arguments, so its target is
  the token generative function's distribution times exp(complete):
  a completion stopped at max-tokens is complete. The prefix weights
  twist the steps; exp(log-ml-estimate) is unbiased for Z = Σ
  p(x)·exp(complete(x)).
- A token weighs log p − log q + its twist, p the LM's ledger's score,
  so particles agree with the completion's traces exactly. Proposals:
  the LM's row, the row masked by the twist and renormalized, or a
  function.
- **A constrained completion is refused** (owner, 2026-10-22, the
  post-M5 review's P4-01): `smc/run` and `smc/initialize` throw
  `:constrained-completion` for a completion with its own `:potential`,
  naming SMC's `:potential` option instead. For such a completion the two
  sentences above disagree: the target says the completion's distribution,
  but p is the LM's score. SMC drew by the LM's rows and ignored the
  completion's potential, so particles broke a grammar and `smc/traces`
  scored them -Inf. The same target stays reachable through SMC's
  `:potential` with the `:masked` proposal. The alternative was p as the
  completion's site distribution, a second way to say the same target.
- The interface and state are gen.inference.particle-filter's;
  resampling ESS-gated at n/2, systematic by default. One
  `lm/batch-next-logprobs` a step; particles are token vectors and the
  LM's trie owns the branches.
- **The host pool (2026-10-21, m3-par; confirmed by the owner,
  2026-10-21):** each particle's proposal row and draw, pure functions
  of its row, twist and key, run on a fixed pool of daemon threads,
  collected in particle order under the caller's bindings, so a run is
  the calling thread's alone bit for bit at every pool size; the LM's,
  the potential's and a proposal function's calls stay on the calling
  thread. `gen.llm.smc` owns one shared pool, made on first use, never
  the common pool; `:host-pool` takes the caller's own
  (`smc/host-pool n`) or `false`. Its size: half the processors up to 4,
  as the llama.cpp engine's row pool, or `-Dgen.llm.host-threads` (1
  turns it off). Both pools come from `gen.llm.host` (owner, 2026-10-22,
  the post-M5 review's I4-04): one pool function, one default size and
  one ordered map, with a pool for each user (one shared pool could
  deadlock). So the llama.cpp engine's pool follows
  `gen.llm.host-threads` too. Alternatives: the LM owning it, no default
  pool, a pool for each run. Measured (quiet, Qwen3's vocabulary, 64
  particles): 4 threads take 17% (bootstrap) to 20% (masked) off a step
  on the toy engine; on llama.cpp with Qwen3-0.6B, with R6's
  `constrained/weighted`, the masked proposal's step went from 106 to
  51 ms. Recommended, not built: rows that carry their cumulative
  probabilities from the engine (a draw by binary search;
  `docs/research/llm-token-smc.md`).

### A hybrid model's steps split over its recurrent cells

(2026-10-21,
agent; confirmed by the owner, 2026-10-21, with its default by device below;
`docs/research/llm-branching.md`, Recurrent states and split steps):
- A recurrent or hybrid model's decode copies every state from the
  lowest to the highest cell its sequences land on (llama.cpp's
  `find_slot` and `build_rs`): on Qwen3.5-0.8B (Metal) 3.5 ms for each
  state in the span not read. The LM's cached entries sit between a
  step's cells, so its batched reads cost 3.6 times raw operations.
- `gen.llm.llama.recurrent` (pure, internal, pinned to llama.cpp
  620d222) replays where llama.cpp puts the states, and the llama.cpp
  engine splits a step into decodes over runs of adjacent cells when the
  replay finds it cheaper: only for recurrent and hybrid models, and as a
  function of the engine's own operations, never of timing, so the same
  calls give the same rows bit for bit. Measured: the LM's 8 batched
  reads 106 → 32 ms (raw 26), 32 reads 186 → 98 ms (raw 78).
- The replay stops (one decode a step from then on) after a decode that
  fails other than for room, and when a slot's state position
  (`llama_memory_seq_pos_min`) disagrees with it, loudly on System.err.
  Where a state sits changes no result, only the cost.
- **Default by device (owner, 2026-10-21; genclj-kw88):** the split is on
  only where the model's weights are on Metal. On CUDA (the RTX-6000 machine's RTX PRO
  6000, shared) a split step of Qwen3.5-0.8B cost about twice one decode
  (52 against 24 ms), an extra decode costing more there than the copies
  it saves; genclj-mxi9 measures CUDA and the CPU properly. The property
  still says otherwise either way.
- Names for the owner: the property `gen.llama.split-recurrent` (default
  by device, above; false keeps one decode a step, exactly as before), the info key
  `:split-recurrent?`, the diagnostics `llama/step-splits`, the namespace
  `gen.llm.llama.recurrent`. An upstream llama.cpp issue (copy only the
  states that moved) is drafted for the owner, not filed (genclj-g5gx).

### The shared memory budget

(2026-10-21, agent; confirmed by the owner,
2026-10-21; genclj-zztc; `docs/research/llm-branching.md`, The shared
memory budget):
- MLX's memory limit is no cap, and the machine runs short well below
  it with a model's weights mapped beside MLX's arrays. gen.clj keeps a
  budget of its own, shared by MLX and native engines.
- `gen.array.memory` (L0, beside the backends; plain numbers, no native
  library) holds reservations: `reserve!`, `release!`, `reserved`,
  `reservations-held`. A llama.cpp engine reserves its footprint
  (`:footprint` in its info: its weights once, as `llama_model_size`,
  and the KV-cache, recurrent-state, output and compute buffers
  llama.cpp reports allocating) when made and releases it when closed.
- The MLX backend checks after every evaluation, in the membrane, that
  its active memory fits the budget less the reservations and a
  headroom; past it, it collects and checks again, then throws
  `:mlx-budget-exceeded` with the numbers. `budget` and
  `set-budget! {:bytes :headroom}` in `gen.backend.mlx`.
- Defaults for the owner: the budget half the physical memory
  (`gen.mlx.budget-bytes`, as MLX's own limit), the headroom a sixteenth
  (`gen.mlx.budget-headroom-bytes`). The names above are recommendations
  too.

### Grammar-constrained decoding

(2026-10-21, agent; confirmed by the
owner, 2026-10-21; `docs/research/llm-grammar.md`):
- *Our own engine, no instaparse* (DESIGN §8). Masks ask whether a text
  with a token appended is still a prefix of the language; a DFA answers
  by one transition a byte, shared by every token through the trie of
  their bytes, while instaparse has no parser state to carry from token
  to token, so each token would be a reparse. GenMLX used it only to read
  the regex syntax, a 150-line descent here. Measured at Qwen3's 151,936
  tokens (M4, quiet): every mask of a 117-state tool-call grammar in 5 ms,
  of a 132-state JSON object in 41 ms; a constrained token costs 270 µs
  (Qwen3) to 490 µs (250,000 tokens) beyond its draw, 5 to 8 % of a
  5.8 ms llama.cpp decode, the dense `logw-next` row a part of it.
- *Languages over bytes, characters as UTF-8*, because tokens are byte
  strings that cross character boundaries; `[^"]` admits every character,
  not printable ASCII (GenMLX's genmlx-6dax). The regex syntax is
  java.util.regex's where it accepts, and loud elsewhere.
- *Exact masks*: a token is allowed only into a state some sequence of
  the vocabulary's tokens completes, every state the tokens reach
  explored when the grammar is made; so a grammar's `prefix` is 0 or -Inf
  by reachability and a constrained completion never strands. An empty
  language and one the vocabulary cannot spell are loud.
- *What a grammar holds* is the text: the tokens' bytes concatenated, as
  the LM's decode (special tokens by their rendered bytes, as GenMLX's
  tool calls relied on), read byte by byte, so a token may end inside a
  character. A completion's end tokens are not text: the end is allowed
  where the text is complete. A token with no bytes is allowed wherever
  the text may go on.
- *A grammar knows its end tokens* (2026-10-21, after the round-3
  review's R2; confirmed by the owner, 2026-10-21):
  `(grammar/grammar m language {:end-tokens ts})`, by default the LM's
  `:eog` (none for a byte vocabulary). It leaves them out of the text,
  its trie and its liveness, so a language that needs an end token's
  bytes is `:inexpressible` when the grammar is made; and a completion
  under a grammar must have the same end tokens (`:mismatched-end-tokens`
  otherwise). The first build read every token as text, end tokens
  included, while the completion gives an end token the end's weight: a
  grammar `[ab]{3}\n` with a newline end token passed `prefix` and
  stranded every completion. Alternatives: the completion passes its end
  tokens to the potential on every call (a change to gen.llm.potential's
  protocol, for every potential); or the completion makes its grammar
  itself, `(token/completion m {:grammar language})` (one way less to
  disagree, but grammars no longer potentials shared with token SMC).
- *A potential declares its end tokens* (owner, 2026-10-22, the post-M5
  review's P4-02 and I4-02). The check above was a type test for
  `Grammar` in the completion's constructor. A grammar inside
  `potential/product` passed it, and token SMC never checked, so a
  mismatched pair silently estimated another Z (0.37 where the matched
  pair gives 1).
  - Now the declaration is an optional protocol in gen.llm.potential:
    `(potential/end-tokens p)` is a set, or nil for no opinion. A grammar
    declares its own set. A token-level potential has no opinion, since
    its contexts never hold an end token; `potential` takes an optional
    `:end-tokens`.
  - A product carries its factors' set, and is
    `:mismatched-end-tokens` when it is made if two factors disagree.
  - One check in gen.llm.potential compares a potential's set with the
    end tokens it meets. `token/completion` and `smc/initialize`
    (for SMC's `:potential`) call it, and only the end-token check:
    SMC keeps a potential that rules everything out as a -Inf
    estimate, not an error.
  - Being a separate protocol, it leaves existing implementations of
    `Potential` working.
  - Passing the end tokens on every call was rejected. It changes every
    potential's protocol and costs every call, but it removes no check:
    a grammar's trie and liveness are built for one set, so a grammar
    would still have to compare them.
- *The middleware is the completion's `:potential` option*: the token
  GF's transitions are written once and the potential reweighs each
  site's distribution (row times weights, renormalized; an end token
  takes the end's weight), the semantics in `gen.llm.constrained`
  (`weighted`, also for token SMC's masked proposal).
- *Its GFI weights are the local distribution's*, q(x) = Π_i p(x_i | x_<i)
  exp(w_i(x_i)) / Z_i, which is not the LM conditioned on the potential,
  π(x) ∝ p(x) exp(w(x)): renormalizing token by token favors sequences
  whose early tokens leave little mass disallowed (tested on the toy
  engine). `constrained/conditional-weight` is log p(x) + w(x) - log q(x),
  the importance weight for π, whose mean is the probability that the
  LM's completion satisfies the grammar.
  It takes a completion's trace only (owner, 2026-10-22, the post-M5
  review's P4-03), and throws for any other: a model's, a distribution's,
  a `call-at`'s. It used to read the trace's generative function's
  `:potential` and give 0.0, the weight of a plain completion, for every
  trace without one, so importance weights over a model that splices a
  constrained completion silently became the proposal's own. The
  alternative was a compositional definition, the sum over the trace's
  completion subtraces: a new feature, which would want its own design.
- *Weight 0*: a constrained or kept token of weight 0 scores -Inf and the
  sites after it are the LM's rows, so moves to it are rejected, never
  NaN; a context of weight with no token allowed is loud.
- *Byte-level marginals* (`bytes/byte-lm`): a text's probability over
  every tokenization, exact by default, a beam for a lower bound.

### Compiled functions across threads

(owner, 2026-10-21; genclj-hfwo).
A compiled function may be called from any thread.
- Each MLX closure compiles under an id of its own, never reused, and
  freeing it erases its traces from every thread's compile cache. Before,
  a second thread's trace leaked, and a later compiled function at a
  reused address replayed it (wrong results, reproduced on Metal).
- Accepted cost: on a virtual thread, whose MLX cache is its carrier's, a
  compiled function retraces every call. Measured on a 20-op chain of
  10,000 floats: 513 µs a call, against 289 µs on a platform thread and
  566 µs uncompiled. gen.clj itself compiles on the calling platform
  thread and uses no virtual threads.
- docs/array-api.md's compile entry says so (its "Threads" item), an
  addition to the approved API doc that the owner accepted with the fix.

### Metal's buffer limit: collect and retry once

(owner, 2026-10-21; genclj-9tyh). MLX on macOS allocates every buffer
through Metal, which
allows about 499,000. The backend's array cap (60% of that, genclj-hche)
counts only the arrays the JVM owns, so graphs can still reach the limit.
- Measured: MLX recovers fully after "[metal::malloc] Resource limit
  exceeded", both when an array is made and mid-evaluation. Dropped and
  collected, the arrays free their buffers, the same graph evaluates to
  its value, and a fill reaches the limit again: no leak.
- So at the two places a buffer is made (an array from host data, and
  an evaluation), the limit error runs a full collection and the call is
  retried once. A second failure is a loud ex-info, `:mlx-buffer-limit`,
  with the live count and the hint (eval! as you go, and drop graphs).
- Alternatives were only a clearer error (no retry), or MLX's own
  message as before.

### An optimizer step evaluates what it returns

(genclj-9e9e).
`learn/step` evaluates its new store and optimizer state: a step is an
inference loop's boundary (rule 3). A training loop written by hand that
did not evaluate (train-test's lecture loop) made each store a graph over
every step before it. MLX's nodes held it after the JVM freed the
handles, invisible to the backend's live count, MLX's active and cache
memory and the heap: about 270 KB of resident memory per lecture on
Metal, 1 GB to 10 GB in 8 s at Gen.jl's size, where `learn/train`, which
evaluated its steps, stayed flat. `train` and black-box VI no longer
evaluate the step themselves (VI's history holds host numbers: each
draw's log weight is read). A step inside a gradient (tape values) or under the batch
tracer is unchanged: the wrappers evaluate their primals. The other loops
a user drives step by step were checked: MCMC runs, particle-filter
steps, rejuvenation and move-reweight already evaluate per iteration;
MH, MALA, HMC, MAP and elliptical slice read their decision, which
evaluates the new choices, and their lazy score is one step deep (20,000
hand-driven MH and MALA steps that read nothing stayed bounded). The
alternative was to fix only the test's loop and document the rule, which
would leave the trap for the next loop written by hand (approved under
the owner's delegation, 2026-10-21).

### A vector is a path, everywhere

(owner, 2026-10-21; M5's G9 and F3).
gen.choicemap's rule, an address is any value except a vector and a
vector is a path, now holds for selections too; before, `sel/selected?`
and `sel/get-subselection` read a vector as one (absent) address and
silently answered false and `none`, and `cm/set-submap` stored a vector
key.
- The protocol methods become `-selected?` and `-subselection`, one
  address at a time, implemented and never called by users (as
  IChoiceMap's `-get-submap`). The public `selected?` and
  `get-subselection` take an address or a path, walked an address at a
  time; the empty path is the root (`root-selected?`, and `s` itself).
  As in Gen.jl, selecting an address selects every path under it.
- `sel/set-subselection` is Gen.jl's `set_subselection!`, persistent: on
  `none` or a hierarchical selection it nests, on `all`, a complement or
  a custom selection it is a loud ex-info. `hierarchical` builds through
  it, so a vector key nests.
- `cm/set-submap` takes an address or a path, as `get-submap` does; the
  handlers' single-address fast paths (`set-value`, `remove-submap`) are
  loud on a vector. No public function reads a vector as one address.
- Alternatives were separate path functions (two names for one idea, and
  a rule for selections unlike choice maps') or a path record in place
  of vectors (churn: vectors are paths in `select`, `from-paths`,
  `cm/addresses` and every trace). Built in M5 (bean 10).

### Parameter gradients through a combinator's trace

(owner, 2026-10-21; from M5's map.jl port). `param-gradients` on a
combinator's trace (Map,
Unfold, Switch, Recurse, call-at) differentiates its callees' parameter
stores, accumulated over the elements, as Gen.jl's
`accumulate_param_gradients!` does on a Map trace. Before, it threw unless
the trace's own generative function was a dynamic one holding a store, so
a combinator at the top had to be spliced into one. D7's rule stands, one
store per generative function and gradients as values; this extends where
the stores are found.
- *The design (genclj-umww; approved under the owner's delegation,
  2026-10-21).* `gfi/stores` gives the stores a generative function holds, in
  its callees' shape, and `gfi/with-stores` gives a tree of that shape back
  (protocol `gfi/IStores`; Object holds none). A dynamic or deterministic
  function holds its own store (`with-params`); a Map and an Unfold their
  kernel's; a Switch `{branch stores}` over the branches that hold some; a
  Recurse `{:production … :aggregation …}`; combinators nest.
  `param-gradients` differentiates that tree, so its result has the same
  shape: a Map's is its kernel's, summed over the elements, and a Switch's
  has a gradient for each holding branch, zero for one the trace did not
  take. Each store is its own namespace: two branches' `:std` are two
  parameters, as Gen.jl's per-function parameters are.
- gen.learning steps the tree as any other (`learn/step` on `(gfi/stores
  gf)` and the gradients), and `gfi/with-stores` gives the stepped stores
  back to the combinator.
- A dynamic function's callees are in its body, not its fields, so their
  own stores are not its: their gradients are the call's trace's,
  `(param-gradients (trace/subtrace tr addr) retgrad)`, with the caller's
  cotangent of the call's value as retgrad when the caller uses it
  (switch.jl:199's port). call-at and choice-at are dynamic functions
  here, so their own store is what a trace of theirs differentiates, and
  the callee reads it.
- Alternatives: one map merging the callees' stores by name, which makes
  two branches' `:std` one parameter, unlike Gen.jl; and reaching every
  store in the trace's tree, which a dynamic body's callees cannot be given
  back.

### A custom function's trainable parameters

(M5's G5, genclj-2294; approved under the owner's delegation, 2026-10-21).
`(custom/deterministic
{:params [:w] …})` names the parameters a deterministic function reads,
and `:apply-with-state` (or `:apply`) takes their values as a second
argument, `{name value}`. They come from the store in effect: the calling
model's, or the function's own (`custom/with-params`, Gen.jl's fields of a
CustomDetermGF) over it, as a dynamic function's `(param name)` reads. So
a model's store trains them with gen.learning, and a store change runs
the function again (the call's store mark), never update-with-state with
stale values.
- `param-gradients` differentiates them through the function's code, or,
  with `:gradient-with-state`, by `:param-gradient-with-state`, `(fn [state
  args retgrad] {name gradient})`, Gen.jl's
  `accumulate_param_gradients_determ!`. The function then sees the
  parameters' values, not the tape's, as it sees the arguments'. Its
  scale factor is the caller's weight on the returned gradient (D7).
- Loud: a missing parameter; `:params` with the stateless `:gradient`;
  `:gradient-with-state` without `:param-gradient-with-state` (or the
  reverse); `with-params` on a function without `:params`.
- Alternatives: the parameters as a dynamic var the user's code reads
  (implicit, and the custom gradient could not see them as inputs), or as
  extra arguments (they would show in the trace's args and argdiffs).

### A call's trace and return value by address; merge and Recurse's numbering

(owner, 2026-10-21; M5's G1, F3, G10, G13; m3-grammar's design).
- `(trace/subtrace tr addr)` is Gen.jl's `get_call(tr, addr).subtrace`:
  the trace of the call at an address or a path, loud where no call is
  (D4, "Lookups are loud" below); `[]` is `tr` itself, and a path into a
  call descends into its subtrace. `(tr addr)` is Gen.jl's `trace[addr]`:
  the choice there, otherwise the call's return value, so `(tr [])` is the
  return value (`trace[]`) and Map's `(tr i)` is element i's.
- The mechanism is a protocol in gen.trace dispatched on the trace's
  generative function (a call's subtrace for a path prefix; Object's
  default nil), so each combinator keeps its representation private:
  dynamic functions (call-at and choice-at with them), Map and Unfold by
  index, Switch through its branch, Recurse by [node :production] and
  [node :aggregation].
- The path rule's edges: `[]` is the root for `cm/set-choice` and
  `cm/set-submap`; `cm/from-map` nests a vector key as a path (an overlap
  is a loud merge error); a vector inside a path is a loud error wherever a
  path is walked.
- `cm/merge` takes 0 to n choice maps (none gives EMPTY, merge's
  identity). Recurse's numbering is Gen.jl's, 0-based: `get-child`,
  `get-child-num`, `get-parent`, their arguments checked loudly.

### Lookups are loud

(M5's D4, genclj-ocwu; approved under the owner's delegation, 2026-10-21).
One rule: a lookup throws where Gen.jl throws
KeyError, and, given a not-found value, gives that value there instead,
as `nth` does. The questions never throw.
- Loud:
  - `(cm/get-choice c a)` where no value is: nothing there, a node, or a path through a value;
  - `(cm/get-value c)` of a node or of nothing;
  - `(cm/get-submap c a)` on a path through a value;
  - `(tr a)` where neither a choice nor a call is;
  - `(trace/subtrace tr a)` where no call is.

  The error is an ex-info with data `{:reason :path :at}`. `:reason` says what the choices hold at the path: `:absent`, `:value`, `:node` or `:through-value`. `:at` is the prefix holding a value when the path goes through one.
- Lenient:
  - each lookup with a not-found argument, such as `(cm/get-choice c a nil)` or `(tr a nil)`, which never builds an exception;
  - the questions `(cm/has-value? c a)`, a new address arity, and `(cm/has-submap? c a)`, now false at a value as Gen.jl's has_submap is. Both are false on a path through a value, where Gen.jl's has_value throws (untested upstream).
- `get-submap` at a value is the Value, as before. A value is the choice
  map at its address: a distribution's trace has one for its choices, and
  the handler and the combinators hand it to the callee there as its
  constraints. Gen.jl 0.4.8 throws KeyError there, so its four such
  assertions (`assignment.jl:158, 228, 285`, `static_dsl.jl:613`) port as
  this difference.
- The handler's and the combinators' walks use the not-found arity, so
  their speed is unchanged. A value where a run calls (a constraint, or an
  old choice in a revise) stays the unvisited error that a revise drops.
  A loud walk there made the kind-changing model's laws fail (the spike,
  `genclj-ocwu`).
- Clojure's `get` on a choice map or a trace reads the record's fields,
  not addresses. It never was a lookup, and the earlier sentence calling
  it the nil-returning one is struck.
- **A breaking change** for code built on gen.clj that reads an absent
  choice as nil (memo.clj, GenGDT and others): `(tr :b)`, `(cm/get-choice
  c a)`, `(cm/get-value c)` and `(trace/subtrace tr a)` now throw there.
  The migration is to pass a not-found value, `(tr :b nil)`, or to ask
  `(cm/has-value? c a)` first.
- Alternatives:
  - `get-submap` loud at a value too. Gen.jl's four assertions would then port as written, but internal code would need a second accessor for "whatever is at p".
  - Nil-returning functions under new names, which makes two vocabularies for one lookup.

### Filling with an array is loud

(owner, 2026-10-21; M5's G14).
`array/full` and `full-like` take a scalar fill value (a host number or a
0-d array, tracked on the tape too); an array of values is an ex-info on
every backend, pointing to `broadcast-to`. Before, the JVM threw a
ClassCastException and MLX broadcast it silently.

### llama.cpp's band by device

(owner, 2026-10-21; genclj-kw88). The
Branching contract's llama.cpp tests take their band from the device's
measured noise: twice its measured maximum, 0.1 nats on Metal (as before)
and about 0.26 on CUDA with float weights (f16: max 0.13). With quantized
weights on CUDA the contract's noise checks are skipped with a note naming
DESIGN §6's measurement, never given a band wide enough to hide it.
- **With the float models measured (owner, 2026-10-21):** CUDA's band is
  0.53, twice Gemma-3-270M F16's 0.26, and 0.23 for long branches without
  a sliding window. Past a sliding window on CUDA the noise checks are
  skipped with a note, as for quantized weights: the measured 1.3 nats
  would need a band of 2.6. A watch bean (genclj-gjiz) keeps the cause,
  and an upstream report if it is llama.cpp's. The alternatives were the
  rule's 2.6 and investigating before M3's close.

### Variational inference trains the model's parameters too

(M5's G4; m3-grammar's design). `vi/black-box-vi` and `black-box-vimco` take
`:model-store` and `:model-optimizer`, Gen.jl's `model_update`: the model
runs under that store, and each iteration also steps it along the
objective's gradient in the model's parameters, ∇θ log p(x, z) averaged
over the draws (the ELBO's) or weighted by the normalized weights w̃_k
(the K-sample bound's). q's draws do not depend on θ, but a latent the
model samples itself, which q does not propose, may depend on θ
(owner, 2026-10-22, the post-M5 review's P3-01). Its draw's
score-function term, ∇θ log p(z; θ) times the learning signal, is added:
the ELBO's log weight, or VIMCO's own leave-one-out signal. So each
gradient is unbiased for the iteration's estimate. (The ELBO's signal is
the log weight less the mean log weight of the iteration's other draws,
with ∇θ log w kept whole. Against dropping its zero-mean
−∇θ log p(z; θ), that lowers a coin's per-iteration sd from 0.68 to 0.28
and leaves a shift's about the same (1.92, against 1.89);
`vi/model-gradients` has the derivation. p(z; θ) is the density the
model's internal proposal drew z with, score − project of the
constrained addresses, its prior for a gen function.) Without that term
it was biased: 0.005 where 2 is exact, and for VIMCO 1.452 where 1.622
is. The alternative was a documented restriction with a loud error, but
such latents are a supported use. One option without the other is a
loud error, not q's optimizer taken silently; the result holds
`:model-store`, and `elbo-gradient` and `vimco-gradient` return
`:model-grads`. The alternative, one store tree `{:q … :model …}` under
one optimizer (as gen.learning allows), was dropped: it gives both stores
one step size, where Gen.jl's minimal VAE trains q and the model at two
(1e-4 and 0.002). (Approved under the owner's delegation, 2026-10-21.)

### Git

the main repo is public-ready, with `.beans/`, `.beans.yml`,
`.claude/` and `*.pdf` gitignored. Its remote is
`robert-johansson/gen.clj`, private from M4's close (the owner,
2026-10-21: until gen.clj has a license), public since 2026-10-26 (the
owner); main is pushed at a milestone's close or with the owner's
go-ahead. The license (owner, 2026-10-26; genclj-fhdz): MIT, `LICENSE`.
The ports of Gen.jl's test suite (0.4.8's, and 0.4.9's new tests;
`test/gen/genjl/`) derive from
Gen.jl's and keep its Apache License 2.0, a copy beside them, as its
section 4 asks. `.beans/` is its own private repo
(`robert-johansson/gen.clj-beans`), synced by SessionStart/PreCompact/Stop
hooks (see `.beans/hooks/README.txt`).

### lgamma and digamma at their edges are SciPy's

(confirmed by the owner,
2026-10-22; m3-grammar's cross-review, genclj-9gyi): lgamma(+Inf) and
digamma(+Inf) are +Inf, as in SciPy and Julia, where gen.array gave NaN (the
composite and its bit-identical host path). lgamma(−Inf) is −Inf (|Γ| goes
to 0 between the poles), where the pole branch gave +Inf; digamma(−0.0) is
+Inf, the limit from below, keeping the sign of a zero, where `(+ z 0.0)`
made it −Inf.

### A log mass of 0 has gradient 0, not NaN

(confirmed by the owner,
2026-10-22; m3-lm, genclj-rnbd).
- The problem: a family that computes every support value's log mass and
  then selects (categorical's pick, bernoulli's two branches, a `dist`
  pushforward's matches over an enumerable draw) sends a cotangent of 0
  into log at a mass of 0. That gives 0 · ∞ = NaN, which flows into
  whatever computed the parameters. `(dist coin [p] (where (bernoulli p)
  1.0 0.0))` at p = 0, observed 0.0, had a NaN gradient.
- The fix is in the families: they take the log through a stand-in
  where the mass is 0, the convention gen.dist already uses outside a
  support. So a value of mass 0 has log density −Inf and gradient 0.
- gen.array's xlogy and xlog1py keep their derivatives, which are
  JAX's, because changing L0 would alter them for every caller. The
  families guard their arguments instead.
- The alternative was to leave it, with a NaN gradient whenever a
  parameter vector holds an exact 0.

### Probabilities sum to 1 within their dtype

(confirmed by the owner,
2026-10-22; m3-lm, genclj-ggdl). check-probabilities (also the mixtures'
weights) and dirichlet's support bound |Σp − 1| by 4·log2(K)·ε of the
values' dtype, never below 1e-6. That's room for two blocked or pairwise
sums: one that normalized the values, one that checks them (genclj-y0mp
measured each within 0.8·log2(K)·ε). The fixed 1e-6 refused float32
probabilities at K = 10⁴; float32 there now gets 6.3e-6, and float64 keeps
1e-6. The alternative was K·ε, which would accept data normalized by a naive
sequential float32 sum but is 0.12 at K = 10⁶ in float32, so it would barely
check. Under log2(K) such data is a loud error.

### A mixture of a discrete and a continuous family is an error

when it
is made (owner, 2026-10-22, the post-M5 review's P2-02, genclj-e6b0). Its
density would add a pmf to a pdf: given y = 0, the posterior came out
0.267 where 0.80 is exact. Gen.jl's types forbid it.

### A diverging HMC trajectory is rejected

(owner, 2026-10-22, the
post-M5 review's P3-05). A leapfrog trajectory whose position, momentum
or energy turns non-finite is a rejected proposal, with acceptance 0, as
Stan treats a divergence. The run counts its divergences, so they are
not silent (the move's `:diverged?`, `:divergences` in `k/run`'s result
and in a combinator's). MALA's proposal is treated the same way. The
trajectory freezes at its last finite state, so no gradient is taken at
a NaN and a compiled step needs no branch. Before, a divergence threw
"constrained value is NaN", a rule meant for user data: HMC on Neal's
funnel at L 10, eps 0.5 died at step 1,309 from a valid state.
- *A parameter check failing at a tried position is a divergence*
  (confirmed by the owner, 2026-10-22; genclj-gomm).
  - Example: at the funnel's L 3, eps 20, a finite v near −1500
    underflows exp(v/2) to a std of 0. The JVM backend's concrete check
    threw "normal: std must be positive" (33 of 300 proposals), where
    MLX's compiled path sees only NaN and already counted a divergence.
    The density is undefined there, and Stan likewise rejects a proposal
    on a domain error inside the leapfrog.
  - Only the ex-infos that name a parameter are caught (`:distribution`
    and `:param`: dc/check-positive and its kin, check-param-not-nan,
    mvnormal's positive definiteness), inside the leapfrog and MALA
    steps. A check that names none stays loud (uniform's low < high,
    mvnormal's shape and symmetry), as do a structure change and a
    value's shape.
  - The alternative was to keep these loud, as "non-finite" reads
    literally, so the outcome would depend on the backend.
- *The start must be finite* (confirmed by the owner, 2026-10-22;
  m3-grammar's cross-review): a starting trace whose score or gradient
  is not finite, NaN or −Inf (a log of 0), is an error for HMC and MALA,
  as Stan refuses a log(0) start. Otherwise every proposal from it
  counted as a divergence, and the chain never moved: 40 of 40 on both.
  MH and the other kernels may still start there.
- *One divergence rule for HMC, MALA and NUTS* (genclj-gqd3,
  2026-11-04, MALA's threshold included; confirmed by the owner that evening): a transition whose energy error H' − H
  passes 1000 diverged, Stan's max_deltaH, which NUTS already used. For
  HMC the energy error is −log α; MALA is one leapfrog step of size
  √(2τ), so its −log α is that step's. Before, HMC and MALA counted only
  non-finite energies: the book's chapter 023 saw an energy error of
  1e28 at eps 0.11 rejected but not counted. The -particles kernels and
  the device chains read the same verdict. Accept decisions change only
  where a uniform of exactly 0 accepted such a proposal (`:log-alpha`
  reads −Inf there now). The alternative for MALA, its rule kept
  (non-finite only), would have counted a step HMC counts as a
  divergence as none.

### GenMLX extras, round 1

(owner, 2026-10-23; epic genclj-0qds). Twelve
deferred-port beans in three waves, chosen from the comparison of
2026-10-23 (genclj-dglq): diagnostics, NUTS, distributions and
serialization; gradients, enumeration and Switch under the batch tracer,
and Mix, Scan and dimap; conjugate proposals with Kalman and HMM
generative functions, the fuzzer's remainder, particle Gibbs, and
Laplace. Main is pushed after each wave once its checks pass. The owner
took these recommendations:
- **No `delta`.** Gen.jl has none, and GenMLX's scores by exact float
  equality; a fixed value is better left untraced, or a `gen.custom`
  deterministic function.
- **Serialization is EDN in the core,** arrays as tagged literals. Transit
  would add a dependency (rule 8).
- **The diagnostics' reference values come from ArviZ,** installed into
  GenMLX's venv for `test/fixtures/generate.py` only; tests never need it.
- **NUTS is Stan's:** multinomial NUTS with the generalized no-U-turn
  criterion, dual averaging toward an acceptance of 0.8, and Stan's
  windowed warmup with a diagonal metric, so results compare with Stan's.
- **A Switch whose index differs across particles batches by
  partition:** the particles are grouped by index, each branch runs
  batched over its group, and the results are scattered back. Masked
  choices (GenJAX's Mask) were the alternative; they would make every
  reader of a batched trace honor a mask, and run the branches not taken.
- Numeric edge semantics in the new code are the owner's (as always);
  while the round runs autonomously, each is recorded here PROVISIONAL
  with its alternatives and listed in the wave's report.

#### MCMC diagnostics

(genclj-hfca; confirmed by the owner, 2026-10-25).
`gen.inference.diagnostics` computes bulk and tail ESS, the ESS of the
mean, rank-normalized and split R-hat, the MCSE of the mean and sd, and
summaries, as Vehtari et al. (2021) define them and ArviZ and Stan's
posterior compute them, on host draws ([chain][draw], or one chain).
Checked against ArviZ 1.3 (to 1e-12, quantiles bit for bit) and,
without Python, against AR(1) chains' known ESS and shifted chains'
split R-hat. Autocovariances are direct sums up to Geyer's truncation,
O(S·T), not an FFT. The edges:
- constant draws as ArviZ: ESS S; R-hat NaN when every draw is the
  same, +Inf for chains stuck apart (alternatives: posterior's NA, a
  loud error, GenMLX's 1.0; a constant quantile indicator is normal
  for a discrete choice, so it must not throw);
- one chain's R-hat is its halves' (posterior's; ArviZ gives NaN);
- loud errors (`:non-finite`, `:unequal-chains`, `:too-few-draws`
  below 4, `:out-of-range` for second moments past a double's range)
  where ArviZ returns NaN; chains of 4 to 9 draws get ArviZ's
  S·log10(S) cap;
- constant means exactly equal (ArviZ's threshold, 1e-15, is absolute,
  so scale-dependent);
- summary's sd divides by n − 1, its default quantiles are 5%, 50% and
  95% (posterior's); no MCSE of quantiles, which needs a beta quantile
  function.

#### Serialization

(genclj-unjc; confirmed by the owner, 2026-10-25).
`gen.serialize` writes EDN read by clojure.edn with gen.clj's tags only
(`#gen/array {:dtype :shape :data}`, `#gen/choicemap`, `#gen/value`,
`#gen/key`, `#gen/trace`), never eval; an unknown or malformed tag, or
text after the value, is a loud error. A float is the shortest decimal
of its double, exact for every float dtype (NaN loses its payload).
Host numbers stay host numbers; backend arrays, MLX's 0-d ones too, are
tagged and read back onto `*backend*`. Values are EDN data only;
anything else (functions, records, lazy seqs, sorted collections) is
refused, naming its path. As in GenSerialization, a trace is its
arguments and choices, never its generative function, rebuilt by
generate with every choice constrained:
- The written score is generate's, computed at write time with the
  trace's own generative function: Map and Unfold keep scores
  incrementally (as Gen.jl's), so an updated trace's score equals
  generate's only within rounding (129 of 200 Map regenerates moved
  the last bits; at most 8e-14 after 2000 steps). The alternatives,
  a tolerance or the original score on the rebuilt trace, were not
  taken.
- A trace records the backend and float dtype it was written under, and
  `read-trace` refuses any other (`:other-numerics`): an exact rebuild
  is per backend and dtype, since log densities round differently;
  `read-trace-data` with `gfi/generate` loads it elsewhere. A
  rounding-error bound for cross-backend reads was the alternative.
  The MLX device is not recorded, so CPU and GPU float32 may differ as
  `:score-differs`.
- read-trace checks its rebuild's score against the written one bit for
  bit, and write-trace and read-trace check generate's weight against
  its score up to rounding (relative 1e-9 in float64, 1e-4 in float32):
  no choice missing (genclj-s5fj, 2026-10-25).
- Traces with encapsulated randomness are not serializable yet (the
  lead's call): ω is kept per call, generate draws it afresh and
  fold-in is one-way, so write-trace refuses an ω anywhere. The hook
  (`:omegas` for generate, an ICalls `-calls`) is genclj-t8uj.
- `gen.array/backend-name` is public, for the recorded numerics.
- One `#gen/array` form for printing and writing (genclj-5k7q,
  2026-11-04; confirmed by the owner that evening): a backend array prints as `write` writes
  it, and a choice map's leaf whose value is a map prints tagged
  `#gen/value`, so printed text reads back. The map form was kept over
  the printed `#gen/array[:float64 [[…]]]`, whose nested vectors lose
  the shape of an empty array ([0 3] prints as []), and so files
  written before read unchanged.

#### NUTS and Stan's warmup

(genclj-tvc0). `mcmc/nuts` is one
transition of Stan's base_nuts: multinomial sampling within subtrees,
biased progressive sampling between each new subtree and the
trajectory, the generalized criterion with Stan's extra checks across
merged subtrees, `:max-depth` 10, and a divergence at an energy error
over 1000 besides hmc's (a non-finite step, a parameter check failing
at a tried position). The trace is updated once per draw; `:compile?`
compiles each leapfrog step and runs the tree on the host. hmc, mala
and nuts take `:inv-metric`, Stan's inv_metric (the identity by
default, under which hmc's and mala's moves are bit-identical to
before). `gen.inference.nuts` (`adapt`, `sample`) is Stan's
hmc_nuts_diag_e_adapt, for nuts or (`:kernel :hmc`) hmc with L fixed.
On a 10-d Gaussian NUTS costs about 2× the best-tuned HMC's gradients
per effective draw, with no L to tune (`docs/benchmarks.md`).
Confirmed by the owner (2026-10-25):
- the names `:inv-metric` (Stan's, NumPyro's and BlackJAX's; not
  `:metric`, which leaves M against M⁻¹ unclear) and `:eps` everywhere
  (hmc's name), so an adapted state merges into the kernels' options;
- the first μ is log(10·the given `:eps`), default 1, as Stan's
  services set it (Hoffman and Gelman use the heuristic's ε); with no
  dual-averaging iteration after the last restart, the re-initialized
  ε is kept (Stan's would be exp(0) = 1); a window under 2 draws keeps
  the regularized current metric, as Stan;
- `:warmup 0` samples at the given `:eps` and `:inv-metric` (CmdStan
  refuses adaptation without warmup);
- adaptive HMC fixes L (Gen.jl's parameter; Stan's static HMC fixes
  L·eps), its acceptance statistic min(1, α), 0 when it diverged;
- the step-size heuristic steps with the chosen kernel itself (NUTS at
  depth 1, or hmc at L 1);
- loud errors for an invalid metric, step size, depth or option, and
  Stan's runtime errors (`:improper-posterior` past a step of 1e7,
  `:no-step-size`, `:metric-overflow`).
- **Stan's step-size jitter** (owner-approved, 2026-10-23): `:jitter`
  on hmc and nuts (and the sampler) steps each transition by
  eps·(1 + jitter·(2u − 1)), u from the transition's own key; the
  default 0 draws nothing, so every result is bit-identical to before.
  It is Neal's remedy for a fixed L that resonates: on Metal, adaptive
  HMC at L 8 on the correlated Gaussian adapted eps 0.4913, L·eps 3.93
  on the fast mode's period 3.88, bulk ESS 23 (the JVM landed at
  0.642). The sampler jitters every transition, warmup included, and
  dual averaging adapts the nominal step (Stan's choice; confirmed 2026-10-25;
  the alternative, no jitter in the warmup, leaves the adapted step
  blind to the jitter's effect on acceptance).

#### GenMLX's further distributions

(genclj-z2gh). `truncated-normal`,
`von-mises`, `wishart` and `inv-wishart` take Distributions.jl's
parameters; `wrapped-cauchy` (scipy's wrapcauchy, centered on mu) and
`wrapped-normal` have none there. Each is exact at its extremes: the
truncated normal in log Φ (draws by a log-space inverse CDF, exact on
[50σ, 51σ]); von Mises draws by Best and Fisher's sampler rewritten in
1 − ρ and κ(r − 1), exact to κ = 1e300; the wrapped normal's density
by two series omitting less than 1e-17; Wishart draws by Bartlett's
decomposition, the inverse Wishart's by a triangular solve.
mvnormal's covariance check is shared as `check-covariance`.
Confirmed by the owner (2026-10-25):
- invalid parameters are loud: lo ≥ hi (or both bounds infinite on
  one side), κ ≤ 0, ρ outside [0, 1), df ≤ p − 1, a scale that is not
  symmetric positive definite (alternatives: lo = hi as a point mass;
  κ = 0 or a negative ρ as the uniform or μ + π; the singular Wishart);
- large κ stays exact, with no normal approximation (numpy switches at
  1e6, Distributions.jl at 700); κ is clamped at the largest float/4
  only against overflow;
- circular values lie in [-π, π) (a draw that rounds to π, or a huge
  mu's lost angle, becomes -π), and a value outside [-π, π] scores
  -Inf (GenMLX's density was periodic, Distributions.jl's von Mises
  support is [μ − π, μ + π]); a porting note says to wrap angles first;
- the wrapped normal draws the uniform from σ = 9 (equal to its law
  within 6e-18; wrapping σz broke there), and σ is floored at the
  smallest normal float;
- a matrix value must be exactly symmetric and positive definite
  (Julia's isposdef), else -Inf; its gradient is on the lower triangle
  (the tape's Cholesky convention);
- Wishart draws at df near p − 1 are often singular at the float's
  precision (in float64, p = 2: 15667 of 20000 at df = 1.01, 2 at 1.5,
  none from 2), and such a draw scores -Inf, never NaN, as a draw past
  the float range does (alternatives: floor Bartlett's diagonal, which
  would change most draws near p − 1; a larger minimum df; an error);
- von Mises draws fall back to mu if all 33 rejection rounds fail
  (probability below 1e-15).

#### Per-particle constraints and batched enumeration

(genclj-7oc1; confirmed by the owner, 2026-10-25). `gen.batched/generate`
and `update`
take `{:per-particle cs}`, n choice maps, each particle's own
constraints beside the shared ones (an overlap or a wrong count is
loud). Which constraints are per particle is said, never read from a
shape. In a batched run they are stacked into batched values, which
the handler reads as one value per particle; maps that do not stack
(different addresses) run one particle at a time, with a fallback
reason, rather than failing. `gen.batched/project` gives the
particles' projections. `enumerative-inference` with `:batched?`
(opt-in) makes the grid points the particles of one run: `:batches`
replaces `:traces` (`enum/traces` and `marginal` read either),
`:chunk-size` bounds a run, and the weights equal the scalar loop's
point by point when every choice is given; choices left out of the
grid are drawn per chunk, so a chunked run's draws differ from one
pass. `with-exact-proposal` takes `:batched?` too (opt-in: it pays from
some tens of assignments, 0.6× at 10, 20× at 1000); inside a batched
run, or with `:check-retval?`, it enumerates one assignment at a time.
Measured 2.4× (100 grid points) to 44× (10^5).

#### Gradients under the batch tracer

(genclj-7dti; confirmed by the owner, 2026-10-25). `array/value-and-grad` of
batched arguments, or anywhere in a
batched run, is vmap's: each particle gets the value and gradient of its
own scalar run (docs/array-api.md §8b, *Gradients*).
- **The tape wraps the tracer.** The tracer's `-value-and-grad` runs
  `gen.array.tape` over itself, so every primal and every VJP rule
  (written against gen.array; linear algebra against the tape's inner
  backend) runs per particle. gen.array's backend join became
  transitive (a wrapping backend wraps what its inner backend wraps),
  the tracer passes an op with no batched operand to its inner backend,
  and `value-and-grad` dispatches on the effective backend, so a
  gradient inside a batched run is on the tracer even when every leaf is
  shared. The alternative, the tracer over the tape (whole-batch arrays
  on the tape, the [n] output summed), needs a batch backend per tape
  and a join of the enclosing run's batched values with it; rejected.
  MLX's native gradients cannot wrap the tracer, so it runs inside them
  (`p/INativeGrad`): MLX differentiates the sum over the particles at
  whole-batch leaves, and custom VJPs reach it lifted (the tracer's
  `ICustomVjp`).
- **A shared argument's gradient is per particle** (vmap's), not the sum
  over particles the bean proposed: a particle's result must be its
  scalar run's (§3.3). The sum, what a shared parameter's learning
  needs, is one reduction.
- **One kind in both modes** (after the MLX run): a value is batched when
  a batched value reached it, and a gradient when its argument or the
  value is. The tape's gradients pass through `by-kind`; MLX's native
  path takes MLX's ordinary gradient when the value is shared, and runs
  again with shared leaves broadcast when the value is batched.
- **Not supported:** a batched run inside a differentiated function (the
  tape outside the tracer) is a loud error (unrelated backends);
  differentiate inside the run and sum. memo.clj will need it for fits
  whose gradient HMC takes from outside (genclj-hwk1, the backlog).
- **In inference:** `gen.batched/map-particles` runs a function of a
  trace once on a collection's batched trace, with a batched run's
  guards and the fallback to one trace at a time (keys split n ways);
  `set-choices` gives each particle its own values (remade as
  `select`'s, checked when read). On them: `choice-gradients` of a
  particle collection; `mcmc/hmc-particles`, `mala-particles` and
  `map-optimize-particles`, n chains each accepted on its own, whose
  moves are the scalar kernels' own functions of arrays (`hmc-move`,
  `mala-move`, `map-move`) under the tracer; black-box VI's opt-in
  `:batched?` draws. HMC's and MALA's acceptance ratios now use the
  trajectory's log densities (assess weights), and a trace is updated
  only when its move is accepted. Measured (JVM): 8–221× for choice
  gradients, 4–246× for an HMC step, 10 to 1000 particles.
- The edges: `:compile?` with a -particles kernel is a loud error (a
  batched step runs under the tracer, which compile does not see
  through); a parameter failing its check at a tried position in a
  batched move falls back, each chain then diverging on its own (as the
  scalar kernels'); a particle whose start is not finite is a loud error
  naming the particles (`:particles`); a MAP line search runs until
  every particle has its step (one host read per step size tried);
  map-particles' non-array leaves must agree across particles (loud
  otherwise). Batched NUTS is out of scope (tree depths differ per
  chain).
- **`hmc-particles` takes `:jitter`,** each chain's step jittered by its
  own u, as its own `hmc`'s would be: drawn per particle under the
  tracer, from each chain's key in the fallback (where the step is
  exactly its hmc's), reported as `:eps`, an [n] array. Alternatives: one
  shared u per transition (chains correlated through the step), or a
  loud error. `mala-particles` takes none, as `mala`.

#### Switches by partition

(genclj-7tqw; the marked choices confirmed by the owner, 2026-10-25). As
the owner decided, a Switch whose index differs across a batch's
particles runs each branch batched on its own particles
(`gen.partition`: gather, join, scatter). Each part's backend draws the
whole batch's draws at its rows (confirmed; a site in G groups draws
G·n values), so a particle draws the same in any partition, bit for
bit; the alternative, draws of the part's size from a key folded with
the branch, would make a particle's draws depend on the others'
branches. The trace holds each group's subtrace and its choices by
part (`gen.array.batch/Parts`), which gen.batched reads per particle;
arguments and per-particle constraints go into a part by rows, results
come back by scatter. Updates run each old group's particles that go
to one branch together and join a branch's results, so there is one
group per branch; `select` joins traces rather than rebuilding them.
Gradients with respect to choices outside a partitioned Switch stay
batched; inside it they go per particle. Loud (confirmed): return
values that cannot be one batched value (shape, dtype or structure
differing across branches, int against float); a partitioned trace
revised by another generative function; a selection under choices
held by part; constraints by part whose particles span groups. A
partition needs a one-axis batch. Measured: importance sampling on a
mixture of structurally different components 3.8× (100 particles) to
21.5× (10,000) faster than one particle at a time, an MH sweep of 1000
chains between components 6.4×.

#### Scan, Mix and dimap

(genclj-2gma; confirmed by the owner, 2026-10-25).
In gen.combinators, on the handler path, with the general,
forward-sampling, fixed-support and across laws and the fuzzer.
- `(comb/scan kernel)` is JAX's scan, built like Unfold: arguments
  `(init xs & params)`, `(kernel carry x & params)` returns
  `[carry' y]`, and Scan returns `[carry ys]` (GenMLX's took
  `[init inputs]` and no parameters). A step is revisited when its
  carry, its input (by a vector-diff of xs), the parameters, its
  constraints or its selection may have changed; a part of `[carry y]`
  that is the same value is no-change, so a changed output alone leaves
  the next step alone. xs is a sequential collection (an array is not
  split along its first axis, as JAX's is).
- `(comb/mix gf…)` and `mix-logits` are a categorical at `:index` and a
  Switch at `:component`: the traces of
  `(gen [w & args] (apply splice :component (switch gf…) (trace :index
  (categorical w)) args))`, so the GFI is the handler's and a Mix
  batches as its Switch does (per particle, by partition). Weights are
  the first argument, as probabilities (Gen.jl's mixtures); GenMLX's
  log weights are a loud error. The alternatives were weights fixed at
  construction or computed from the arguments (GenMLX), GenMLX's
  `:component-idx` beside the component's choices, and per-component
  arguments (GenJAX). A changed Mix argument reaches the component as
  unknown-change.
- `(comb/dimap gf f g)`, `contramap` and `map-retval` call gf on
  `(apply f args)` and return `(g v)`; the traces are gf's. f is opaque
  to change hints: no argument changed means the previous inner
  arguments, no-change; otherwise an identical inner argument is
  no-change and the rest unknown-change. g runs again only on a changed
  inner return value.
- No Mask: it is `(when active? (splice …))` in a body, or `(switch
  {true gf false (gen [& _] v)})`, which batches per particle when v is
  a value of gf's return kind (with v nil it falls back, correctly).

#### Running totals never NaN

(genclj-awd0, a bug the static fuzzer found; the remedy confirmed by the
owner, 2026-10-25). Map's, Unfold's and Scan's scores are
running totals (score + new − old per changed element), as Gen.jl's.
After a choice left its support and came back, the total was
−Inf + Inf = NaN where the density is finite, silently. A general law
holds every kind to it: an updated or regenerated trace's score is the
assessed density of its choices, through chains that leave a support
and come back (models with bounded kernels exercise it).
- **The remedy: counts beside a finite sum (genclj-ul72; the owner
  kept it, 2026-10-24, after the MLX measurements below).** A trace
  keeps its total as three
  numbers: the sum of the finite element scores, the count of scores
  not above −Inf, and the count not below +Inf. A NaN is in both
  counts, as −Inf + Inf is NaN. A change updates them in O(1), and the
  score is their IEEE sum: what summing the element scores gives
  (assess), so +Inf scores (beta(a, 1) at 0 for a < 1) and NaN ones
  come and go exactly too. The finite part rounds as the running total
  it was before awd0. Scalar runs branch on the host value (no array
  ops); a batched or MLX run pays 9 ops per element added or taken out
  and 6 per trace for the score. Measured (`gen.model-bench
  long-unfold`, a 500-step Unfold, JVM backend, paired runs): a batched
  update revisiting one step, 650–714 µs before, 274–278 µs now (284
  before awd0); a batched extension by one step (a particle filter's)
  118 µs before, 132 µs now; batched generate and every scalar case
  unchanged within noise. On MLX, where even a scalar model's scores
  are arrays, each run once (CPU / Metal): scalar generate 130 → 152 /
  175 → 197 ms (+17% / +12%), scalar revisit 864 → 102 / 1367 → 286
  µs, scalar extension +13% / +9%; batched generate +8% / +10%, batched
  revisit 1785 → 924 / 2655 → 1331 µs, batched extension +13% / +4%.
  array/compile would fuse the 9 ops on MLX's scalar path only (the
  batch tracer runs uncompiled); a vectorized tally of a run's new
  elements would round unlike the scalar handler, which spot checks
  compare.
- **The remedy it replaced (awd0's):** where an update or regenerate
  subtracted an element's score and the total was NaN, the score was
  the sum of the elements' scores: on the host only then, and per
  particle under the batch tracer by a `where`, which summed every
  element whenever one was taken out. Free on the extension, but O(n)
  per revisit under the batch tracer.
- Alternatives: always recomputing (3× on the batched extension); a
  pairwise sum tree (O(log n) per change, its rounding unlike assess's
  sequential sum).

#### Conjugacy, and Kalman and HMM generative functions

(genclj-702q; Kalman and HMM confirmed by the owner, 2026-10-26, with
gen.state-space in the stable tier; conjugacy, experimental, PROVISIONAL,
owner to confirm). Chosen by the user, never detected
(§3.1's corollary).
- `gen.inference.conjugate/with-conjugate-proposal gf latent children`
  makes one latent's exact posterior given its observed children gf's
  internal proposal for it (thesis pp. 145–146), as
  `with-exact-proposal` does for finite choices: normal–normal and
  mvnormal–mvnormal with known noise, beta–bernoulli and beta–binomial,
  gamma(shape, scale)–poisson, dirichlet–categorical.
- Correct by construction: a reference run puts the latent at its
  prior mean and each open child at a value its distribution fixes, so
  the proposal of the other choices has an exact density whatever
  depends on what; a final run draws the latent from the posterior,
  keeps the other choices and draws the open children given the latent.
  The weight, log p(τ) − log q(τ∖σ | σ) for the proposal used, is then
  a correct importance weight for any model, and exact (log
  p(observations), no variance; regenerate on the latent a Gibbs move
  of weight 0) when only the children depend on the latent. A mismatch
  costs exactness, never correctness: GenMLX's silent wrong marginals
  (94qc, yy8u, rmy7, 5ytq, jmk4, su6q) are loud errors or exact here,
  each a test.
- A child's parameter must be the latent's value itself, checked by
  identity; any function of it is a loud error, so affine Gaussian
  children are not conjugate pairs here (linear-Gaussian chains are
  `gen.state-space/kalman`'s; a sound linear tracer would be needed,
  never numeric probes). Structural checks run on every backend; the
  value checks (the children's other parameters and every other
  choice's density unchanged by the latent) read values only on an
  eager backend, as parameter validation does. `{:check-exact? false}`
  allows other choices to depend on the latent (the weight unbiased,
  not exact). Unobserved declared children are drawn after the latent;
  a choice depending on one is loud by default. One latent per
  wrapper; zero visited children is loud.
- `gen.state-space/kalman` and `hmm` (L4: generative functions in their
  own right, DESIGN §3's explicitly separate marginalized ones) are
  Unfolds over a filter step whose state is the belief: the
  observations are the choices at addresses 0..n−1 and the states are
  summed out, so scores are exact marginal likelihoods, updates to more
  steps incremental, and every law (forward-sampling ones too) holds.
  Arguments are `(n prior params)`, params a map (unknown keys loud), a
  function of t or a vector of maps; the prior is the first state's
  (Dynamax's convention); Kalman updates in Joseph's form; an HMM's
  emission is one distribution batched over the states. Unobserved
  steps are forward-sampled (no missing-data mask yet).
  `sample-states` is forward filtering, backward sampling, a function
  of a trace. Their fixtures come from no filter: the joint Gaussian of
  the series, and every path of states.

#### Laplace and Fisher information

(genclj-xg1j; PROVISIONAL, owner to confirm). `gen.inference.laplace`:
Hessians of a trace's log
density, in selected continuous choices or in trainable parameters, by
central differences of the reverse-mode gradient, symmetrized (the tape
is first order; core code calls only gen.array), with a step of
ε^⅓·max(1, |x|) from the dtype's epsilon (Nocedal and Wright §8.1;
GenMLX used 0.005 in every dtype). Modes by Newton's method with
Levenberg damping where −H is not positive definite, Armijo
backtracking, and a Newton decrement of ε·max(1, |log p|) as the stop
(repeated map-optimize converges linearly, and the Hessian is needed
anyway). The Laplace estimate is `:log-ml-estimate`, as importance
sampling's; its proposal is a gen function of ignored arguments that
traces each selected leaf from its Gaussian conditional, for
importance sampling and `with-internal-proposal` alike (a leaf of two
or more axes is an error). A stationary point that is no maximum, and
an observed Fisher information that is not positive definite, are loud
errors (GenMLX damped and warned); `natural-gradient` damps, as a step
direction. The observed information is of the trace's whole log
density: complete-data information when it holds latents, not of an
estimated marginal as GenMLX's was. `bic` takes n explicitly.

#### Particle MCMC

(genclj-gcus; PROVISIONAL, owner to confirm).
`gen.inference.particle-mcmc` (Andrieu, Doucet and Holenstein 2010):
`conditional-smc` is a particle filter conditioned on a trace, built on
gen.inference.particle-filter's functions, its selected choices the
reference trajectory held at particle 0, its other choices constrained
in every particle; `particle-gibbs` is its move as a symmetric kernel
(the CSMC kernel is reversible), composing with the MH kernels. PMMH
has no API of its own: it is MH on a `gen.custom/estimated` density
whose estimate runs a particle filter. The choices: inputs are a trace
and a selection, as mh's; one sweep over a vector of steps, not a
`:reference` on the filter's step functions (every other filter
operation would then have to keep or refuse a reference); the
reference's internal-proposal weight by project on its new latents
(GenMLX projects onto the observations; they differ only when the
arguments change an older choice's density); multinomial resampling
only; the ESS threshold n/2 by default, as `maybe-resample`;
encapsulated randomness a loud error (the reference's ω would be drawn
afresh, genclj-t8uj); the last step's arguments must equal the trace's;
n ≥ 2. The CSMC state's log-ml-estimate is conditional, so biased, and
says so. Ancestor sampling is deferred: generically it costs O(n·T²).

#### The path-equivalence fuzzer's remainder

(genclj-y0ij). GenMLX's
PEF, for gen.clj's second execution path, the static language:
gen.random-static-models (test code) generates static bodies as data
and writes each as a `gen ^:static` form, evaluated at test time;
gen.static-fuzz-test holds them to `laws/static` (and the general
laws, `^:slow`), and HMC, MALA and NUTS steps, compiled and plain, on
the static model and its dynamic body, along random chains whose
kernels, selections, arguments and metrics vary. A failing case prints
its spec and the form it became. genclj-f5fy's generated code is
checked by the same generator. It found two bugs, both fixed:
- **gen.diff's +, −, inc and dec rules gave an int-diff through a
  float operand,** exact only up to rounding (an unsound retdiff under
  the laws). They now give one only when every operand is an integer
  (confirmed by the owner, 2026-10-25: a host integer, or an array of an :int dtype kind; not
  an integral-valued float, which would be exact only below 2^53 and
  need a host read of an array); otherwise unknown-change.
- **Map's and Unfold's scores turned NaN** (−Inf + Inf) when an update
  returned an element to its support (genclj-awd0, fixed in wave 2).

### The book is Markdown whose code runs

(owner, 2026-10-24; genclj-ceds).
No Clerk and no notebooks. The chapters are Markdown files (`book/`), and
their code snippets are verified for real. A chapter's code lives in its
own namespace (`book/src/book/…`), written and run at the REPL; the
Markdown includes its named regions (`;; book: name` in the source,
`<!-- code: name -->` in the Markdown), written in by `gen.book sync`. A
test runs each chapter's namespace form by form on every backend and
fails when an included block differs from its region. A `;=>` comment
after a form checks its result, a number to the digits it shows. A
statistical claim is a check written in the code. Figures are SVG files
made by the book's own code, and a test fails when one is stale. (The
first spike tested code blocks inline in the Markdown, as memo.clj's
HANDBOOK.md does; the owner preferred code in .clj files, for the REPL.)
The JVM backend is the book's gate (owner, 2026-10-24): gen.book-test
runs the chapters there on every change (about 12 s for Part I), and on
another backend only with -Dgen.book.mlx=true. Part I took 19 minutes on
MLX's CPU and more on Metal, its examples running one model at a time.
gen.clj's correctness on MLX is the battery's business (the conformance
suite, the laws, the library's tests); the book's MLX run checks only
that its own code is backend-neutral, worth doing at milestones (1.0),
as information. Chapters stay written for every backend (; float64,
bands that hold in float32, booleans through array/truthy?).
One book in five Parts (owner, 2026-10-24): I Foundations, II Inference,
III Theory, IV The System, V Applications, and appendices, with Gen.jl's
tutorials as worked chapters inside them, not a separate tutorial site.
`gen.book pdf` builds it with Pandoc.
The book's checks follow the test rule, with no exceptions (owner,
2026-10-24, genclj-fblt): a claim that one method beats another passes
at 5 standard errors of the difference, like a two-sided band, and a
chapter buys that with runs, not a lower threshold (Part II's 022 had
used 3). A comparison of Monte Carlo estimates takes 5 standard errors
(MCSE for chains); a learned function (a trained network, a proposal)
takes 5× a spread measured over 8–10 training keys, the spread written
beside the check; a check whose band is too wide to catch a plausible
error (023's metric) gets more draws, or claims only what it can show.

### Law results

(2026-10-24, genclj-4tjd, the owner's go). A law in
`gen.laws` returns a test.check result, not a boolean: its `checks` fail
naming the check and the values it compared, and a case the law says
nothing about (constraints another run made that this one cannot visit)
is `vacuous`. `deflaws` counts the vacuous cases and fails a law with
more than 60% of them (`laws/max-vacuous`: the highest share measured
was the fuzz's 32%). Seeds span 2^31 (`laws/seeds`; `gen/nat` drew 0–99
only), and the fuzz's random models take an argument with alternatives,
their label the spec.

### The test loop

(2026-10-24, genclj-elyc, the owner's go: a 7-minute
battery is not run after each change). A REPL helper, `(t)` in
`dev/user.clj` (`-M:test:dev`), reloads what changed with tools.namespace
(a `:dev` dependency only) and runs the tests the reload reached, without
`^:slow`. `gen.parallel` (`-X:test:par`, with any tier) runs the test
namespaces in 5 JVMs of 2 cores each, sharded by recorded times: the
fast tier 87 → ~50 s, the battery 408 → ~150 s on the M4. It refuses
MLX and llama.cpp runs (one GPU JVM at a time). Not taken: fewer cases
per property (less power), tests in parallel threads of one JVM
(`with-redefs` in nine files), C1-only JIT (the fast tier took 200 s).

### Reading a value being differentiated on the host is an error

(2026-10-24, genclj-de4g; confirmed by the owner 2026-10-25: gen.clj's
decisions come first, and memo.clj, then GenGDT, adapt to them). Under
gen.array.tape, `array/item` or `array/->clj` of a tracked float throws
ex-info (`:shape`, `:dtype`) naming the ways out: compute with gen.array's
operators, give the function a custom gradient (gen.custom's `:gradient`,
`array/custom-vjp`), or read `(array/stop-gradient x)` to drop the
gradient on purpose; to branch or index, compare with gen.array's
operators or read `(array/floor x)`. A host read used only to branch or
index (`(int (array/item x))`) lost nothing before and is refused now:
the price of a rule that cannot tell it from a read that loses a
gradient. Until now the read returned the primal and the path
through it was missing from the gradient without a word: chapter 028
found d/dm of log N(m; 0, 1) + log N(3; 2m, 1) at m = 0.8 coming out
-0.8, the prior's term alone, instead of 2.0, through `item` in a body
and in a `custom/deterministic` without `:gradient`. Rule 6: PyTorch's
`.item()` detaches silently; gen.clj does not.
- Bools and ints have no gradient, and comparisons, floor and the other
  zero-gradient ops give untracked values, so their reads stay allowed:
  `if` on a comparison, `truthy?`, a Switch's index. Parameter checks
  read comparisons (`all-true?` of `(> v 0)`), so they needed no change.
- gen.clj's own differentiation-neutral reads take the primal through
  stop-gradient: a gen body's `=`, `case` and lookups on a 0-d value
  (`gen.dynamic.core/host-value`), a distribution's printed parameters,
  and mvnormal's jitter suggestion (found by the battery: gen.genjl's
  dist DSL test of a singular covariance under a gradient).
- Host arithmetic on a tracked value (clojure.core's operators, `Math/…`,
  `double`) was a raw ClassCastException. `value-and-grad` rethrows it as
  the same kind of ex-info, the cast as its cause: caught at the
  boundary, by the cast's message naming the tracked class, so host
  arithmetic elsewhere costs nothing. (Guarding clojure.core's operators
  in `gen` bodies would cover bodies only and tax their host arithmetic.)
  A JIT that drops the message after many throws at one site
  (OmitStackTraceInFastThrow) leaves the raw exception.
- A batched tracked value's read is the batched-branch error first, so a
  batched run falls back, and the particle's run raises this error.
- Not covered: MLX's native gradients (`-Dgen.mlx.grad=native`, opt-in).
  There the read still detaches silently (d/dx of x·item(x) at 1.5 comes
  out 1.5, not 3.0; MLX's CPU, 2026-10-24): the differentiated values are
  MLX's own arrays, and mlx-c does not expose MLX's `is_tracer`, which
  would tell them from the others.

### The pathwise estimator, ADVI's

(genclj-uiqm; the owner: before the book's VI chapter, 2026-10-24).
`black-box-vi` and `elbo-gradient` take
`:estimator :reparam` beside the score-function estimator (the default,
Gen.jl's): z is q's sample for a fixed key, a differentiable function of
q's store through gen.array's tape, so the gradient of the ELBO estimate
itself is unbiased, its variance under a quarter of the score-function
estimator's on the conjugate test model. A family declares a pathwise
sampler with `^:pathwise` on `defdist` (`gen.dist.core/pathwise?`, false
by default): normal, uniform, exponential, laplace, cauchy, gumbel,
log-normal and logistic, whose samples are differentiable transforms of
noise drawn apart from their parameters. mvnormal waits for the tape's
Cholesky gradient; gamma and beta sample by rejection. Loud where it does
not apply: middleware on q's sites refuses a choice from another family
(whose pathwise gradient would silently be zero; a generative function q
calls keeps its own transitions, unchecked), the model is assessed on
the observations and q's choices so a latent q does not propose is
assess's loud error, and `:batched?` is not yet supported with it. An
automatic mean-field guide is left for later.

### Ten decisions from the book's friction

(owner, 2026-10-25, on the
lead's recommendations; each bean names its decision):
1. 022's "MAP beats drift" ordering gets the runs to clear 5 SE with a
   margin, at about 135 s of chapter time (genclj-fblt).
2. The book test splits into one namespace per Part, so `gen.parallel`
   spreads them; Part I and the appendices stay in the fast tier, Parts
   II–V are `^:slow`, as statistical tests are.
3. 028's ESS thresholds are stated at 5 standard errors of the ESS
   estimate (genclj-fblt).
4. The layers of §4 order what builds on what, not a strict order of
   requires (gen.dynamic, L2, requires gen.dist.core, L3, which requires
   gen.gfi, L2): documented, not untangled (genclj-aa3s).
5. A choice outside its own support stops the run: when a choice's
   log-density is −Inf, the operation returns weight −Inf without
   running the rest of the body, so MH rejects and map-optimize's line
   search shrinks its step. Every other parameter check stays loud
   (genclj-q7cq, genclj-2870).
6. Promotion follows JAX's rule when an integer meets a float (int64 +
   float32 is float32), so float32 models stay float32 (float64 is
   CPU-only on MLX); docs/array-api.md §5 states it (genclj-7ndy).
7. Dynamic bodies keep passing unknown-change to splices; the static
   language is the way to incremental splices, and the docs say so
   (genclj-zszp).
8. Residual resampling rounds n·p_i up when it is within a relative 1e-9
   of the next integer, so equal weights keep their copies; the copies
   stay at most n (genclj-tx1h).
9. A batched run that falls back to one particle at a time records why
   in its result (`:fallback`, a written message, never a raw exception);
   `:batched? :required` throws with that reason instead. Nothing prints
   (genclj-ynpc).
10. `splice` treats an argument as unchanged when it is identical, or a
   host scalar, keyword or string equal to the old one; collections and
   arrays compare by identity (comparing arrays would read them to the
   host on MLX), and the docs give the idiom: reuse the same value
   (genclj-nvo0).

Item 5 as implemented (2026-10-25, the agent; genclj-q7cq, genclj-2870):
- One rule at the record of every site and call (gen.dynamic's weigh,
  and the static score fn's static-site): a log density or call score of
  -Inf on the host (`gfi/impossible?`) stops the run, in every operation,
  simulate included. A dynamic caller stops at a callee's -Inf score.
- The result: weight -Inf, or NaN for an update or regenerate of a trace
  that is impossible too (neither trace is possible, so MH's NaN check
  stays loud for a chain started at an impossible trace). The trace holds
  the choices and calls up to and including the impossible one, score
  -Inf, return value nil. An update's discard holds the old choices it
  replaced or did not reach (Def 2.3.1 with the partial trace's
  addresses), so updating back with it restores the old trace. The
  constraints after the stop are not checked: the run never reached
  them. assess of the partial choices stops at the same choice, so the
  trace is consistent with assess, and the laws hold with σ restricted to
  what the run reached (gen.laws' reached).
- Combinators: Unfold, Scan and Recurse pass a part's return value on, so
  they stop at a part whose score is -Inf (drop the parts after it, an
  update discarding them); dimap maps no such return value. Map and Switch
  pass nothing on and are unchanged; their scores keep genclj-ul72's IEEE
  sums (a Map of a +Inf element and an impossible one is NaN).
- Batched, lazy (MLX) and differentiated runs, whose log densities are
  arrays never read here, cannot stop: they go on, and the handler sets
  score and weight to -Inf where a site was -Inf (per particle), whatever
  ran after it gave (NaN from a parameter made invalid, +Inf). Their
  traces hold the whole run's choices, and its sites' densities as
  computed (zeroing them would corrupt the static language's reuse of
  recorded densities); gen.batched/traces rebuilds a particle's scalar
  trace, which stops. So the weight laws check only w = -Inf for a move
  outside the support (gen.laws' outside?). A downstream parameter check
  that reads a concrete batch (the JVM) and fails for such a particle
  raises the fallback, which stops it.
- Callers that read the new trace before accepting (MH's backward
  proposal, the translators and the trace-kernel DSL, a particle filter's
  discard check and backward proposal, is/estimate's f) skip that work for
  a move of weight -Inf. with-exact-proposal's enumeration needs the
  forward proposal's density of the other choices where an observation is
  impossible, so its runs give the observations at the model's own
  addresses density 0 (an observation inside a call that stops is a loud
  error there); its projection of an impossible trace is -Inf.

Item 5, the owner's two rulings on the implementation (2026-10-25):
- A move from an impossible trace to another impossible one weighs NaN,
  not -Inf (−Inf − −Inf is undefined), and MH's NaN check stays loud: a
  chain at an impossible trace is the caller's mistake, which -Inf would
  hide by rejecting every move forever.
- An invalid proposal stays loud (the law audit's gap J). When a move's
  new trace is possible but the backward proposal gives the move's
  reverse choices density 0, the proposal cannot reverse its own move
  (the thesis's third requirement for MH proposals, §3.4.2), so MH
  throws ex-info naming the proposal; this is a broken proposal, not the
  model leaving its support (genclj-ocxs).

The second ruling as implemented (2026-10-25, the agent; genclj-ocxs):
`gfi/assess-backward` weighs every move's backward proposal (MH with a
proposal, involutive MH, the translators, trace kernels and SMCP3, a
particle filter's step with a backward proposal; rejuvenation and
move-reweight through MH's move) and throws ex-info with `:proposal` and
`:reverse-choices` when the backward weight is -Inf while both traces'
scores are above -Inf, read on the host (a move is an inference-loop
boundary, so MLX checks too). A move from an impossible trace keeps the
first ruling's NaN error, which takes precedence. The rule also catches a
reverse choice of density 0 with nothing left unvisited, which was quiet
before decision 5 too: book 033's exercise 3 (a merge writing a share
above 1) and 052's exercise 2 (split shares restricted to the middle
half, which 052 had shown still keeping the posterior by rejection) now
stop with the error.

The invalid-proposal rule as built is wider than before decision 5: it
also throws where a proposal cannot reverse some moves although nothing
is left unvisited (such a chain rejects those moves and still targets the
posterior); confirmed by the owner (2026-10-25): loud, since the thesis
requires reversibility and such a proposal is almost always a mistake.

Item 9 as implemented (2026-10-25, the agent; genclj-ynpc): the fallback
is `gen.batched/fallback` of the particles (`fallback-reason` keeps its
ex-data), and `:fallback` in the results of b/generate, update and
regenerate, is/importance-sampling, importance-resampling and estimate,
pf's state (initialize and step), mh-particles, the -particles kernels,
choice-gradients of particles, VI's batched estimators and fits, and
batched enumerative-inference. map-particles and project return arrays,
so map-particles* and project* return {:result :fallback}. A raw
exception's reason is written (host math's ClassCastException is said
so). `:batched? :required` binds `gen.batched/*required*`: a run that
would fall back throws ex-info with the reason (`:fallback` in its
data), so does an operation on particles already held one at a time,
and a custom proposal or `:strata`, which run one at a time, are errors;
a particle filter keeps `:batched? :required` in its state for its
steps.

### Traces are values

(owner, 2026-10-25, from the book's chapter 034): a
trace is an immutable record (gen.trace's `Trace`); every operation that
revises one returns a new trace and leaves the old one as it was, so a
rejected move needs no undo and old traces can be kept, compared and
shared.

### Arrays are values

JVM arrays compare by dtype, shape and (bitwise)
elements, so traces and choice maps holding them compare with `=`. MLX
arrays compare by dtype, shape and elements too, reading both on the host
(found by the book's chapter 034, 2026-10-25).

## 1.0.0 (2026-10-26)

### A 1.0 for dependents

(owner, 2026-10-24): memo.clj's 1.0 should
probably wait for gen.clj's 1.0 rather than pin a commit. Until then
dependents pin a commit. There is no rush for memo.clj: gen.clj's 1.0
comes when its remaining work is done (owner, 2026-10-24). The 1.0's
scope (which surface it freezes, modules first or not, the post-M5
fixes, the book's Parts I–III first) is open: genclj-04h2.
**The 1.0's shape** (owner, 2026-10-25, on the lead's proposal):

#### The promise

names, arities, result-map keys and documented
semantics (the GFI and its laws) stay put through 1.x; minor releases
add; removals wait for 2.0, after a deprecation cycle (`:deprecated`
metadata and a docstring note, nothing printed). The exact draws for a
key are not promised across releases, only their distributions.

#### Three tiers, marked in code

(`{:api :stable | :experimental |
:internal}` on each namespace). Stable: gen.array, gen.random, the data
layer (choicemap, selection, trace, diff), gen.gfi, gen.dynamic,
gen.dist with defdist and gen.dist.dsl, gen.combinators, gen.custom,
gen.state-space, gen.serialize, inference's core (importance, mh,
kernel, mcmc, particle-filter, enumerative, transform, vi, diagnostics,
weights), gen.gradients, gen.learning, gen.batched (and the public
accessors of a batched run's values, in gen.array: genclj-9efy); stable for implementers, the GFI protocol and
gen.array.protocols. Experimental (shipped, outside the promise):
gen.llm.*, trace-kernel, conjugate, laplace, particle-mcmc, partition,
dist.host, and gen.laws until it moves to ext/laws (it has, and is
stable: "The laws for users", below). Internal: tape,
batch, dtype, shape, the dynamic language's guard, static and core, the
backends. Dependents that reach internals get public replacements
(memo.clj: a per-particle accessor, and gen.array's own tracked?,
broadcast and dtype-kind).

#### One artifact

gen.llm ships inside it, marked experimental;
optional dependencies stay in ext/ (ext/commons-math, ext/laws).
Namespace names are what the promise freezes, so a later split only
adds a coordinate.

#### Enforced

a committed snapshot of the stable vars and arglists
fails a test when the stable surface changes; the API appendix (063)
is generated from the tier marks; a CHANGELOG from now on. Maven or
Clojars coordinates are decided at first publish.
As built (2026-10-25, confirmed by the owner; genclj-dv8x): the mark is
the ns form's attribute map. "Stable for implementers" is `:stable`
with an `:implementers` note (gen.gfi, gen.array.protocols), not a
fourth tier: the promise and its check are the same, and gen.gfi holds
the user API beside its protocols; through 1.x such a protocol keeps
its methods and gains none. Namespaces the list does not name: stable
gen.array.linalg (array-api.md §2's `la`), gen.dist.core (defdist,
and the multimethods a distribution extends), gen.inference.proposal
(internal proposals, under enumerative's with-exact-proposal and
gen.learning) and gen.inference.nuts (Stan's warmup for mcmc's nuts);
experimental every gen.llm namespace and gen.dist.commons-math (over
dist.host); internal gen.array.special, .host, .tree and .memory,
gen.random.threefry and gen.backend.mlx.exact. The snapshot is
test/fixtures/api-snapshot.edn, written by `clojure -M:test -m gen.api
write` (gen.api documents it and deprecation).

#### Confirmed by the owner

(2026-10-25, on the lead's recommendations):
the protocols that implementers extend (gen.gfi's, gen.array.protocols)
keep their methods and gain none through 1.x; a new capability comes
as a new protocol with a fallback for implementations without it (as
gen.array's optional native ops do; Gen's edit would be an IEdit whose
fallback goes through update). The tier choices for namespaces this list
did not name stand as built (linalg, dist.core, proposal and nuts
stable; dist.commons-math experimental; array.special, host, tree,
memory, random.threefry and backend.mlx.exact internal); every gen.llm
namespace stays experimental. `array/vmap`, a public way to run a
function batched over a leading axis, joins the stable surface before
the tag (genclj-t3ur), so dependents and the book leave gen.array.batch.
As built (2026-10-25, confirmed by the owner; docs/array-api.md §8b):
`(array/vmap f {:in-axes [0 nil …]})`, in-axes 0 or nil only (another
axis is a transpose away, and 1.x can add them); trees of arrays in and
out, a shared result broadcast; a batched run's guards, with no
fallback (vmap asks for a batched run); draws per particle even from a
shared key, as every batched run's; nesting and a gradient through a
vmap refused loudly, a vmap of a gradient per particle.

#### The laws for users, in ext/laws

(2026-10-25; genclj-u6je).
Confirmed by the owner (2026-10-25): gen.laws lives in the module
ext/laws and is `{:api :stable}`, with its user API: the check over
models, the law sets, deflaws, checks and vacuous. As built (the agent):
ext/laws/src/gen/laws.clj, its namespace unchanged, with its own
deps.edn (test.check, and gen.clj as `:local/root "../.."`, so a git
dependency with `:deps/root "ext/laws"` brings the gen.clj of the same
commit unless the dependent names gen.clj/gen.clj itself); in this
repository the `:laws` alias (ext/laws/src and test.check) and `:test`,
which includes ext/laws/src and its tests (ext/laws/test). It needs
nothing of test/: the few host helpers it took from gen.test-support
are its own. `(laws/verify models opts laws)`, deflaws' arguments but
the prefix, runs a law set (all by default) and returns a sorted map
of the laws that failed to what each says (the check and its values,
or the error, the smallest case by its model's label, the run's seed);
`{}` when all hold. Every run, verify's and a deflaws spec's, draws its
seed from the clock unless given `:seed`, as test.check's defspec
does, so the battery keeps meeting new cases, and a failure's `:seed`
reproduces it; nothing is printed. `identity-update-reports-no-change`
left forward-sampling for a set of its own, `laws/implementation`
(all is 22 laws), which gen.clj's suites run beside all on its own
kinds; `laws/estimated` is encapsulated without
`:score-is-assessed-on-the-same-key`. checks, vacuous?,
vacuous-on-unvisitable and unvisitable? are public, and vacuity stays
decided by the error's message, documented: a hand-written generative
function throws gfi/unvisited's error.

#### The laws' names, confirmed by the owner

(2026-10-25 night, "as
recommended", on the lead's review of the names the agent chose after
the tier was approved). Kept: `verify` (models first, as deflaws;
`(is (= {} (laws/verify models)))`), `failure-of`, `estimated`, and
`identity-update-reports-no-change` out of forward-sampling (the thesis
does not ask for precise retdiffs, and unknown-change is always
correct). Renamed: `check` to `quick-check` (test.check's quick-check
that also counts vacuous cases; `check` was one letter from `checks`,
which does something else), and `implementation` to `conventions`
(gen.clj's habits, not laws, which a user's generative function need
not keep). What stability means for a law set: a minor release may add
a law to one, with a CHANGELOG line, since a dependent's suite may then
fail; renaming or removing a law is breaking. Stable with the rest, as
the snapshot has them: each law's var (35), `cases`, `close?`,
`across`, `max-vacuous` and `seeds`. `close?`'s tolerances (float64: a
relative and an absolute 1e-12; float32: a relative 1e-4 and an
absolute 1e-3) were not in that answer: the lead had made no
recommendation and recorded them as confirmed with the rest, wrongly.
The owner then chose (2026-10-25 night, on the lead's recommendation)
that they may change in a minor release, with a CHANGELOG line and its
reason, as a law set's contents may: they encode float precision,
which a new backend or dtype may make us revisit. Its name and arity
are stable. The v1.0.0 tag waits for the night's checks, as the gate
has it (owner, the same answer).

#### Eight more, confirmed by the owner

(2026-10-25, on the lead's
recommendations): per-particle results use plural keys (`:log-alphas`,
`:accepted`, `:diverged`), not singular keys holding vectors; chains
take `:keep` and return `:draws`; `k/nuts` is its own reversal (NUTS's
transition satisfies detailed balance); NUTS's `:accepted?` means it
moved, so `k/run`'s acceptance rate is a move rate there (Stan's
accept_stat stays in nuts/sample's stats); `k/elliptical-slice` and
`k/map-optimize` stay; a trace's keyword lookup reads its record fields,
as documented, not loud (genclj-nhtx: a custom type would change every
assoc on traces); choice maps are callable as traces are, loud with a
not-found arity (genclj-a2t8); training's history is the mean generate
weight on held-out examples, as Gen.jl's train! (genclj-59cf).
As built (2026-10-25, confirmed by the owner, 2026-10-25; genclj-a2t8): `(c
addr)`, `(c path)` and `(c addr not-found)` are get-choice's lookups,
loud in the same places. At nested choices the call is an error that
points to get-submap, not the submap, as Gen.jl's `cm[addr]` (a
KeyError) and a trace's call are: a call reads values, a submap is a
choice map, and a call that answered with either would leave every
caller to ask which. Like a trace's call, it reads a 0-d boolean as a
host boolean outside a body (genclj-x6ox; the lead's choice), so
`((:choices tr) a)` is `(tr a)`; get-choice returns the value as
stored. The body flag is gen.choicemap's `*in-body*` (no-doc), which
gen.dynamic binds beside `*body-call*`: gen.choicemap cannot see
gen.trace.
As built (2026-10-25, confirmed by the owner, 2026-10-25; genclj-ufco):
`(sel/union & ss)` and `(sel/intersection & ss)`, none and all of no
selections. all and none absorb or drop out, hierarchical selections
join or meet address by address (a union of selects is their select,
=), others are Union and Intersection records of their parts, which
root-selected? and top-addresses read. A complement's subselection is
simplified (none where it was the complement of all). A selection
prints as its tree, `#gen/selection {:x all, :y {0 all}}`, with all,
none and (complement …), (union …), (intersection …) as names: for
reading, not readable back. The laws: union, intersection and De
Morgan's path by path and through get-selected (gen.selection-test);
project-splits-the-score also over disjoint unions and De Morgan's
intersection, on every kind (gen.laws).
Traces print compactly (2026-10-25, confirmed by the owner, 2026-10-25;
genclj-qvnu, whose bean leaves readable-or-display to the owner):
display only, as the bean recommends, `#gen/trace[maker {:args …
:score … :choices …}]` with the choices as a nested map, and a batched
trace's `{:args … :batch-shape [n]}`; gen.serialize refuses the form
loudly (its #gen/trace is write-trace's map) and stays the round trip.
clojure.pprint prints records as maps whatever their print method, so
pprint still shows a trace's fields (choice maps' too).

#### Approved in advance by the owner

(2026-10-25): `ext/laws`' user
API (the check over models, the law sets, deflaws, checks and vacuous)
joins the stable tier once gen.laws lives there (genclj-u6je); and the
public handler-state design as its agent records it (genclj-q6sc),
with any surprising point brought back to the owner.

#### Before the tag

the friction that changes shape lands (one
result-key vocabulary, kernels returning {:trace :accepted?}, labeled
metrics, matching batched return shapes, selections' union and
intersection, callable choice maps, trace keyword lookups, public
handler state, an optimizer interface and training callbacks, laws for
users, particle-filter accessors, the transform API); bug fixes that
keep the shape may follow in 1.x. The gate: the book's Parts I-V
merged and reviewed; the battery green on the JVM backend, MLX CPU and
Metal, and once on CUDA; the book once on MLX; memo.clj and GenGDT
passing against the release candidate. On MLX's CPU (owner, 2026-10-25
night, on the lead's recommendation): the full battery in float32, the
light tier in float64, whose slow statistical tests run in float64 on
the JVM backend and whose kernels the conformance suite checks against
it (a full float64 run did not fit in 4 h on the Thor). TF32 on CUDA
(owner, 2026-10-25 night, on the lead's recommendation; the night's
rerun confirmed it as the cause: with MLX_ENABLE_TF32=0 the matrix
families' gradients, the batched matrix VJPs, the vmapped matmul and
the Cholesky and solve checks pass on the RTX-6000 machine): off for
1.0.
MLX runs float32 matrix products on CUDA in TF32 by default (a 10-bit
mantissa, about 1e-3 relative: a log density of 100 off by 0.1 nats),
so the gate's CUDA run sets MLX_ENABLE_TF32=0, the install page and the
CHANGELOG say CUDA needs it for float32 accuracy, and after 1.0 the
backend turns it off itself, with an opt-in (genclj-p3pr).
Confirmed by the owner (2026-10-26 morning; the lead's decision of the
night, under the owner's "tag v1.0 without asking me questions"):
half-float sums on CUDA, which fail backend conformance's
long-half-float-sums (MLX accumulates them in their own precision
there, as on its CPU before genclj-y0mp), are a known issue of 1.0.0
(CHANGELOG), not fixed before the tag: routing them through half-sum
on CUDA is a backend change only another CUDA run could verify
(genclj-tblx). The book's MLX run, information by the rule above, was
stopped after Part II's failures (genclj-ac70) to free the MLX slot
for the gate's checks.

#### The result keys

(the agent, 2026-10-25, confirmed by the owner; genclj-4jpw):
one meaning per key across gen.inference, gen.gradients, gen.learning
and gen.batched. A key that holds a log says so in its name, but for
Gen.jl's `:weight` and `:score`. "Host" is a host number or a vector of
them, read at the inference loop; "array" a gen.array value (a 0-d
one is a host number on the JVM backend). A result about n particles
holds the scalar keys in the plural, one value per particle
(`:weights` for `:weight`), with `:particles` for `:trace`.
- `:trace`, `:traces`: a trace (a kernel's next one, the one drawn, a
  chain's last); a vector of scalar traces. `:particles`: a
  gen.batched collection; `:batches`, a vector of them.
- `:weight`: a log weight, array: Gen.jl's of generate, update,
  regenerate, assess and propose, and a translator's, an involution's
  or a move-reweight kernel's (its incremental importance weight).
  `:weights`, `:discards`: gen.batched's per particle, an [n] array and
  a vector of n choice maps.
- `:log-weights`: a weighted collection's raw log importance weights
  (importance sampling, a particle filter's state), host.
  `:log-normalized-weights`: the same less their log-sum-exp, so that
  their exponentials sum to 1 (w/normalize, importance sampling,
  enumeration), host. `:log-total`: their log-sum-exp (w/normalize),
  host.
- `:log-ml-estimate`: the log of an estimate of the marginal
  likelihood (importance sampling's log-mean-exp, enumeration's
  log-sum-exp over a grid, Laplace's), host. `:log-ml`: what a particle
  filter's resamplings have moved out of its weights, host. `:ess`: the
  weights' effective sample size, 1/Σp², host. `:estimate`: is/estimate's
  self-normalized expectation, host.
- `:accepted?`: whether a kernel took its move, a boolean, in every
  kernel's result: MH's, HMC's and MALA's accept decision; NUTS's,
  whether it drew a state other than the start; map-optimize's,
  whether its line search found a step; elliptical slice's and
  particle Gibbs's, always; a combinator's, whether any of its moves
  did. `:accepted`: a vector of them, a combinator's moves in order or
  one per particle. `:log-alpha`: the log MH acceptance ratio (mh,
  mh/move, hmc, mala), host; `:log-alphas` per particle, an [n] array.
  `:diverged?`: an HMC, MALA or NUTS proposal diverged; `:diverged`
  per particle; `:divergences`: how many did (a count).
  `:acceptance-rate`: kernel/run's fraction of moves accepted, host or
  nil. `:draws`: what kernel/run's and nuts/sample's `:keep` kept in
  place of `:traces`, host: (keep t) for each kept trace, or for a map
  of functions a map of such vectors.
- NUTS: `:eps` the step size taken or adapted (host; per particle an
  [n] array), `:inv-metric` (a choice map of variances by address),
  `:accept-stat`, `:tree-depth`, `:n-leapfrog`, `:energy` (host),
  `:stats` (a map of them per iteration), `:warmup` (the warmup's
  `:stats` and `:divergences`).
- `:fallback`: why a batched run went one particle at a time, a
  written message. `:result`: map-particles*'s or project*'s value
  beside it. A particle filter's state: `:parents` (each particle's
  source index), `:resampled?`.
- Gradients and learning: `:values` (the selected choices, a choice
  map), `:gradients` (theirs, the same shape), `:arg-grads`, `:store`,
  `:grads` and `:model-grads` (store-shaped arrays), `:objective`
  (host), `:history`. Transforms: `:log-abs-det` (log, array:
  tf/run's, and beside `:weight`, which includes it, an involution's
  and a translator's), `:model` and `:aux` (choice maps),
  `:bwd-choices`.
  `:grads` and `:model-grads` (store-shaped arrays), `:state` and
  `:model-state` (an optimizer's state, which the option of the same
  name takes back), `:objective` (host), `:history` (host: train's
  held-out weight by epoch, VI's estimate by iteration). Transforms: `:log-abs-det` (log, array), `:model`
  and `:aux` (choice maps), `:bwd-choices`.
Renamed for it: w/normalize's `:log-weights` (it meant the normalized
logs there, the raw ones elsewhere) to `:log-normalized-weights`;
mh/move's `:weight` to `:log-alpha` (mh/mh's name for the same ratio),
a host double now; the -particles kernels' `:accepted?`, `:log-alpha`
and `:diverged?` (per-particle values under scalar names, and a
vector under a `?` key is always truthy) to `:accepted`, `:log-alphas`
and `:diverged`. gen.batched's `{:particles :weights}` stays: it
mirrors gfi/generate's `{:trace :weight}`, as b/simulate's particles
mirror gfi/simulate's trace.

#### Kernels' shape

(the agent, 2026-10-25, confirmed by the owner; genclj-4hu0):
a kernel is (fn [trace key]) → `{:trace :accepted? …}`, and every
kernel of the library returns that map: mcmc's map-optimize and
elliptical-slice too (map-optimize-particles `{:particles
:accepted}`). A result without `:trace`, such as a bare trace, is a
loud error in kernel/run, the combinators and pf/rejuvenate, where it
carried nil on. gen.inference.kernel's hmc, mala, nuts and
elliptical-slice make mcmc's moves kernels, each its own reversal (as
the MH kernels are; NUTS's transition keeps detailed balance), and
map-optimize a kernel without one, since it leaves no distribution
unchanged; their options take no `:key`, which the kernel takes at
each application. kernel/run and nuts/sample take `:keep`. `(kern []
…)` and `(kern ^:trace [tr] …)` are the kernel itself, not a function
of no arguments.

#### Kernels' batched form

(the agent, 2026-10-25, confirmed by the owner, 2026-10-25, the spot check's
delay included;
genclj-xgz4): a kernel says it can run batched by carrying a batched
form, `(fn [ps key])` → `{:particles …}` with the -particles moves'
keys (`:accepted` per particle, `:fallback`): the Kernel record's
fields `particles` and `reverse-particles` (`kernel/batched-form`;
`kernel/kernel`'s options, `symmetric`'s second argument). mh (of a
selection or a proposal), hmc and mala (without `:compile?`: a batched
step runs under the batch tracer) and map-optimize have one; chain,
repeat, cycle and kern without ^:trace combine their kernels' when
each has one (a reversal's from the reversals'), reporting whether
any move took each particle. nuts, elliptical-slice, involutive-mh
(its involution is user code that often reads the host), mix (it
chooses per particle), checked, kern ^:trace and plain functions have
none: pf/rejuvenate moves a batched filter one particle at a time with
them, the state becoming one of traces with `:fallback` saying so, and
`:batched? :required` throws (decision 9). MH with a proposal runs
batched by gen.batched/move, the scalar move's own code (propose,
update, assess-backward) under the batch tracer; assess-backward's
invalid-proposal check reads every particle as a validation
(`array/any-true?`), so the move stays batched, and a particle that
fails it fails the batched run, whose scalar run raises the error.
pf/mean, variance and proportions of a batched state run f once on
the batched trace (map-particles); pf/choice reads one address of
every particle (batched/choice). An unchecked collection's spot check
is made once for its trace (a delay kept on the collection, a value),
however often it is read. Alternatives: a protocol on kernels (a
field keeps a kernel one value, and a function stays a kernel); a
table in the particle filter from kernels to -particles moves (closed
to users' kernels).

#### Labeled metrics

(the agent, 2026-10-25, confirmed by the owner, 2026-10-25; genclj-qs62):
`:inv-metric` gives a variance to each selected choice by its
address, a choice map or a map (`cm/from-map`'s: `{:slope 0.012
:intercept 0.46}`), each a number or an array of the choice's shape
(or flat with its size); one number is still a variance for every
value. A vector or array is taken only when the selected values are
one choice, whose elements it follows in order; for more it is a loud
error (`:unlabeled-inv-metric`), since the flat order is the choice
map's (addresses sorted, intercept before slope), not the
selection's, and a wrong order mis-scales every move without an
error. A choice map lacking a selected address, or holding one not
selected, is loud. `nuts/adapt` and `nuts/sample` return
`:inv-metric` labeled so, adapted or given (the identity as ones),
and `mcmc/inv-metric` labels any form the kernels take. The dense
metric (genclj-q9ge, not built) is left room: it will be a value of
its own, a matrix with its addresses' order, so it takes neither the
number, the map nor the vector form. Alternative: a plain map only
(no choice map), or vectors in the selection's order, which a
selection of a subtree (several choices, one address) leaves
undefined.

#### The transform API

(the agent, 2026-10-25, confirmed by the owner, 2026-10-25; tf/involution
keeps its proposal separate, as Gen.jl's mh(trace, proposal, args,
involution);
genclj-xsml), additions only:
- `tf/run` takes `{:retained selection}`, the model choices the move
  keeps: their continuous reads, neither written nor copied, are
  unchanged outputs of the Jacobian, as an involution's update keeps
  them. Without it every read is consumed, as before: tf/run updates
  nothing, so it cannot tell a kept choice from a discarded one.
- An involution's result and every translator's hold `:log-abs-det`
  beside `:weight`, which still includes it (the glossary's
  meaning). A move outside the support is not weighed: weight -Inf
  and no `:log-abs-det`, where 0 would claim a volume it never
  measured.
- `mh/move` takes `:fwd-choices u`: u given, assessed under the
  proposal with k-propose, so `:log-alpha` is the move's ratio for
  that u (what enumerating auxiliary choices needs); a u of density 0
  under the proposal is loud. Alternative: a separate function, with
  the same result.
- tf/involution's function is also `(h trace u key)` and `(h trace
  u)` (the implicit key), beside mh/move's `(h trace u retval args
  key)`, Gen.jl's four arguments and a key; the short forms have no
  forward return value.
- `(tf/retval space)` in a body: the return value of what made an
  input space's choices, model-in's the trace's (none for a choice
  map), aux-in's the proposal's (an involution's fwd-retval, a
  translator's q-forward's, in a check the backward proposal's,
  tf/run's `:aux-retval`). Read on the host and constant to the
  Jacobian, for structure such as a path (chapter 052's subtree
  move, as the notebook's involution takes its path). An involution's
  check applies the transform again to the backward choices, where
  that value is the backward proposal's, which the involution does
  not hold (mh/involutive-mh does): `{:check? true :proposal q}`
  gives it, and a check that needs it without one is loud.
  Alternative, the owner's call: tf/involution holding its proposal
  always, as Gen.jl's SymmetricTraceTranslator(q, args, h) does, so
  that q is given once.
- Messages: an involution's check says "The involution applied twice
  does not give back the trace" (or "the forward choices"), a
  translator's "The inverse transform …" as before; a transform made
  by tf/transform without an inverse is "The anonymous transform
  (tf/transform)", where its name printed empty.

#### The optimizer interface

(the agent, 2026-10-25, confirmed by the owner, 2026-10-25;
genclj-y2ab): an optimizer stays a map with a `:kind`, and the
multimethods `learning/-init` and `learning/-step` dispatch on it, as
gen.dist.core's do on a distribution's `:type` (a protocol on records
would make optimizers values that print and compare less plainly).
`-step` is pure, `[store' state']`; `learning/step` checks that shape
and evaluates it. `learning/update-given` (the gradient-tree walk) is
public, for a user's `-step`. The built-ins follow PyTorch's update
rules and defaults, ascending: momentum (`:nesterov?`), rmsprop
(`:decay`, PyTorch's alpha), adagrad, adamw; `clip-norm` wraps another
optimizer and clips by the global norm (optax's clip_by_global_norm),
its state the wrapped one's. Constructors check their options loudly:
rates at least 0 (a negative rate, a descent, is an error), momenta,
decays and betas in [0, 1), eps and max-norm above 0. Learning-rate
schedules are not built in.

#### Training's and VI's shapes

(the agent, 2026-10-25, confirmed by the owner, 2026-10-25, descent as an
error included; on
the owner's decision of the history; genclj-59cf): train has Gen.jl's
options and defaults (`:epoch-size` 1, `:batch-size` 1,
`:batches-per-epoch` 1, `:evaluation-size` the epoch's size); a
minibatch larger than its epoch is loud, where Gen.jl's randperm
fails. Its `:callback` takes a map, `{:epoch :minibatch :objective
:store}`, counted from 0, after each step, where Gen.jl passes
(epoch, minibatch, objective): a store is a value here, so the
callback needs it to watch training, and a map lets 1.x add keys. VI's
fits return `:traces`, q's draws of the last iteration (with
`:batched?` `:particles`; VIMCO's one per estimate, resampled, as
Gen.jl's), and call `:callback` after each iteration's step with
`{:iter :objective :traces :store}`; Gen.jl calls it before the step.
train and VI return `:state` and take it back to resume. The book's
ADVI (027, 054) still restarts Adam at its lower rate: resuming the
state left the structured guide's KL at 0.06 to 0.11 after the same
2000 iterations, against 0.0015 to 0.022 restarted (keys 11 to 18).

### The handler's state and middleware

(owner, 2026-10-25, approved in
advance as recorded here; genclj-q6sc, found writing the book's chapter
014, whose middleware needed an outside atom to report anything):
- **The state** is public. `dyn/initial-state` (op, and opts: `:key`
  and the operation's `:constraints`, `:prev`, `:argdiffs` and
  `:selection`, nothing else) makes the state an operation starts
  from, the same sixteen keys for every operation, each documented in
  its docstring: the operation's inputs (`:mode :constraints :prev
  :selection :argdiffs`), what the run has made so far (`:choices
  :scores :subtraces :score :weight :discard`, written by the standard
  transitions at sites and by the loop at calls), and the loop's own
  (`:key :visited :omega :omega-key :omega-count`). Keys in
  gen.dynamic's namespace are the handler's internals.
- **What middleware writes.** It reads every key. It writes keys of
  its own namespace (qualified keywords outside gen.dynamic's and
  gen.trace's) and `:constraints`, to condition the rest of the run on
  values of its own, as gen.inference.conjugate's reference run and
  proposal's replay already did; it changes the model through what it
  hands the transition it wraps (a distribution, a key). After each of
  middleware's transitions the loop checks, by identity and reading no
  value, that the other inputs and the loop's keys are as they were,
  that a choice was recorded at the path, and that the result is
  [value state']; at the run's end an unqualified key the state does
  not have, or one of gen.trace's, is an error. The totals (`:choices
  :scores :score :weight :discard`) are the standard transitions' to
  write: a middleware that writes one itself breaks the trace, and is
  unsupported rather than checked, since checking a sum would read
  values. A run without middleware pays one identity test per site and
  one lookup per run.
- **Data out: middleware's own keys.** They ride the state from site
  to site and the run returns them: on the trace it makes (simulate,
  generate, update, regenerate), in the result of assess and propose,
  which make none. They are data about the run: no operation reads
  them, so they change no operation's meaning (§3.1). Chosen over one
  channel key (`:middleware/out`): namespaced keys are how a Clojure
  map is extended, middleware compose without sharing a channel, and
  writing and reading them needs no new function. Not metadata: the
  partition walks (resampling, a Switch by partition) go through a
  trace's fields, not its metadata, so batched values there would go
  stale.
- **In update and regenerate** each run starts without them: the trace
  an update makes holds its own run's, and middleware that wants the
  previous run's reads them on `(:prev state)`. So a chain of updates
  does not accumulate data unless its middleware asks to.
- **In a batched run** the middleware runs once for all the particles,
  under the batch tracer: it sees batched values, and what it writes
  from them is batched. Its keys are one value on the collection's
  batched trace (`array/particle` and `array/particles` read it),
  gathered with the trace by `batched/take` and `select`. A
  collection that fell back holds each particle's own scalar run's.
  `batched/traces` rebuilds a particle's trace by a scalar generate, so
  that trace holds the generate's keys, not the batched run's. Spot
  checks compare scores and return values, not these keys. A
  dispatcher's result holds none, since no transition ran.

## 1.1.0 (2026-11-05)

### Device programs: the prototype before the note's review

(owner, 2026-10-26; genclj-a1hu). The research into the fastest gen.clj on
CUDA builds its XLA prototype alongside the design note rather than
after the owner's review of it, and the prototype may change the core
on its research branch (`device-programs`) where it needs to. Its two
core changes: keys whose words are a staging backend's array
(`p/IStagedKey`, which `gen.random`'s split and fold-in dispatch on;
still values, rule 2), and a loop (`scan`, in `ext/device` for now; a
core `array/scan` is the design question). Neither reaches main until
the research's recommendation is decided; that decision is recorded
here when it is made.

### Device programs: per-particle keys

(owner, 2026-10-31; the device note's decision 3, genclj-a1hu, genclj-8x5p).
A device run of n
particles or chains from a root key runs particle (or chain) i on
`(rng/split key n)[i]`, so it equals the handler's unbatched runs:
importance sampling without `:batched?`, and kernel/run's chain per
key. It does not equal a `gen.batched` run from the same key (one key
per site with a particle axis); the two agree in distribution. Why:
every lowering runs one particle's scalar program per thread, lane or
loop step (the JVM lowering, CUDA C, SIMD lanes), and the per-particle
form is checkable bit for bit against the handler (float64 on the JVM
backend) or to float32 rounding. The XLA prototype's many-chain
program (`chain/mala-chains-run`, gen.batched's draws) moves to
per-chain keys if that target is taken up.

### Device programs work for every model, sustainably

(owner, 2026-10-31; genclj-8x5p). "We want something that really works in a
sustainable way for all models": the device path is not tuned to the
parity benchmarks. Arrays stay arrays (small ones in registers, large
ones in memory under loops, data as inputs, so code does not grow with
data); each missing op is added once, for every model; coverage is
measured over the battery's models, for simulate, a constrained
generate and MALA's gradients and accepts (a census: on 2026-10-31,
of 286 models, simulate 158, generate 174 and MALA 94 of the 194 with
continuous latents stage and equal the handler bit for bit in float64,
149 all three; no mismatch; the rest need gathers or branch on a draw,
and a few need argmax or cholesky), never over hand-picked ones; and a
model that does
not stage runs on the handler with its reason recorded, never with
another result.

### = on a 0-d MLX array and a host value is loud

(owner, 2026-10-31; genclj-8rdm). clojure.core/= on a 0-d MLX array and a
host number or
boolean throws. It answered false, where the JVM backend (whose 0-d
arrays are host values) compares values, so plain Clojure outside a
gen body (whose guarded = reads the value) took the other branch on
every run: the device census found a random test model doing it, and
the GFI laws, which check self-consistency, could not. Every other
comparison is unchanged: two MLX arrays by value; nil, keywords,
collections and arrays of other shapes unequal. The reversed form
(= 0 k) cannot be caught (clojure.core/= dispatches on its first
argument), so the advice stands: compare draws with gen.array
(array/==), or inside a gen body. The device programs' staged values
follow the same rule (genclj-ryg0).

### Device programs: an out-of-range gather index is the handler's error

(owner, 2026-10-31; genclj-8x5p). Compiled code cannot throw in the
middle of a run, so each run records a flag when a gather's index falls
outside its axis (reading a clamped index meanwhile, to stay in bounds),
and the entry point checks the flags after the run: it runs the first
flagged run's staged function on the handler from that run's key, so
the error is the handler's own; when the handler does not throw, a
generic error names the run. A run that hit a bad index returns no
results.

### Device programs: the note's decisions 1, 2, 4 and 5

(owner, 2026-10-31; genclj-a1hu, genclj-8x5p).
1. The staging backend, array/scan and staged keys move into the core
   at L0, marked {:api :experimental}, outside the 1.x promise while
   they settle; rule 2 reads "a staged key is a value too". The
   lowerings (JVM, Metal, CUDA C) are extensions.
2. Users ask with a :device option on the existing entry points
   (kernel/run, importance sampling, then the particle filter):
   {:device :jvm | :metal | :cuda}, the result shaped as today's. A
   model that does not stage runs on the handler with the reason
   recorded (this section, "every model, sustainably").
4. GPU targets: generated CUDA C (NVRTC, no new dependency) and Metal
   through MLX first; the all-particles form for large shared data
   next (a library GEMM or MLX's matmul); XLA through PJRT only if
   that falls short.
5. Equivalence (rule 1): a float64 program on the JVM equals the
   handler bit for bit; a float32 program (and the GPUs) equals it to
   a float32 band, its accept decisions equal except where log-alpha
   lies within rounding of log u, the tests counting such flips
   against their expected number; statistics checked against the
   handler. Whether the JVM path runs float64 by default is the
   epic's decision 7, still open.

### Tempered SMC, a log marginal likelihood for any model

(proposed 2026-11-01, at the owner's request; shipped as built,
experimental, in 1.1.0's scope, owner, 2026-11-04; genclj-87x1). A new
experimental
namespace, gen.inference.tempering, built on the GFI alone (no new
combinator): targets p(x) L(x)^β, L the observations' density read by
project; β chosen adaptively (half the effective sample size kept);
systematic resampling every step; moves by single-site
Metropolis-Hastings on the tempered target, each move's weight less
(1 - β) Δ log L: regenerations from the prior (global, discrete) and,
for continuous latents, normal random walks scaled by the address's
spread across the particles (a walk that changes the structure is
rejected). Prior regenerations alone were measured to fail a
posterior 45000 times narrower than its prior. One particle at a time
for now. The schedule and the walks' scales adapt to the particles, so
the estimate is consistent, not unbiased; a run given a previous run's
:betas and :scales adapts nothing, and its exponential is unbiased. Why: importance sampling from the prior cannot judge a model
whose posterior is far from its prior (the codegen bake-off's
changepoint), and comparing synthesized models by evidence needs it
(genclj-zjz7).

### gen.laws' tolerance scales with the scores

(owner, 2026-11-01; genclj-6txl). close? compared weights within a fixed
absolute 1e-12
(float64), which assumed scores up to ~100: correct models whose
traces score in the thousands failed laws by ~1e-11 (the codegen
grader found it on four book models). The absolute tolerance becomes
1e-12 × max(1, |score|) over the traces a check's weights come from,
and float32's 1e-3 alike; the relative tolerances stay. A planted
error of 1e-6 still fails for scores below ~1e5. A CHANGELOG line, as
gen.laws' policy requires for a tolerance change.

### An address-less splice runs under its caller's middleware

(2026-11-04, genclj-u23m; the lead's on an agent's fix while the owner was
away, confirmed by the owner that evening). (splice gf & args) is
Gen.jl's {*} ~ gf(args): the callee's choices are the caller's, so the
caller's with-transitions middleware runs at the callee's sites too,
unless the callee has middleware of its own for the operation, which
runs there instead (a callee at an address runs its own transitions,
as before). This was M1's rule (63483e4); the third review (a5b0219)
switched the callee's body to the standard transitions, listing
inheritance among its "Likely" bugs with no reason given and no test.
That made every middleware in gen.inference miss those choices:
with-exact-proposal and with-conjugate-proposal threw "wrap that one"
for a choice they could not wrap, with-internal-proposal's project and
regenerate threw, and the reparameterization estimator silently used a
choice without a pathwise sampler. The alternative, a private flag only
with-exact-proposal honours, would have left the other three. Marked
Breaking in the CHANGELOG: user middleware that changes the model now
changes it at those sites too.

### MLX's gathers keep clamping in 1.1.0

(2026-11-04, genclj-d05w; the lead's while the owner was away, confirmed by
the owner that evening; what follows is the owner's call after 1.1.0,
genclj-08un). An out-of-range array index throws on the JVM backend
and is clamped on MLX (a negative one counted from the end first),
because a lazy backend cannot check without evaluating. That is
gen.array/take's documented contract (docs/array-api.md §13, decision
17), so making MLX loud changes a stable function's documented
behaviour: Breaking, not a consolidation release's fix. 1.1.0 keeps it,
and the conformance suite now names it (out-of-range-array-indices:
eager throws, lazy clamps; planted, a clamping JVM backend fails it).
The owner's options after 1.1.0: keep it; an opt-in check on MLX that
reads each gather's bounds (loud, at a read per gather, for debugging);
or loud by default through an error flag computed in the graph and
raised at the next evaluation, as the device programs do (genclj-i2c4),
which rule 6 favours but which is Breaking.

### A weight that is 0 compares as weights do

(2026-11-04, genclj-s5fj; the lead's while the owner was away, confirmed by
the owner that evening).
Five checks asked for exactly 0 (generate without constraints, update
without constraints, regenerate of nothing and of everything, project
on nothing); they now use close? at the scale of the trace's scores.
Why: the GFI's weights are mathematical quantities, and a hand-written
generate whose weight is the score less the log density of what it drew
is 0 only up to rounding; chapter 045's urn had to special-case "no
constraints" for the law alone, and every other weight check already
allowed rounding. The cost: a bug that leaves a weight of order the
rounding is no longer caught there (a planted 1e-6 still fails in
float64, 0.1 in float32). Under gen.laws' policy, a tolerance change
in a minor release with a CHANGELOG line. The alternative, exact zero
with a documented reason, keeps the urn's special case and asks every
hand-written generative function for one.

### Codegen as a library: the core gets no code evaluation

(owner, 2026-11-04; genclj-uw7w). gen.clj's core, gen.llm, experimental, no
new
dependencies: gen.llm.sample (sampling with temperature, top-k and top-p;
n completions in lockstep over the LM protocol), gen.llm.chat (Qwen's
chat template, thinking on or off, the tool-call format and parser),
gen.llm.agent (the episode loop: tools are functions the caller supplies,
with a session, submit and budgets). Everything that evaluates
LLM-written code (a namespace session, an eval tool with repair and
vetting, a subprocess runner) goes in a module, ext/codegen, with the
gen.clj codegen harness (tasks, grader, clj-kondo, parinferish), which
leaves test/. The core never executes generated code. The owner first
chose all three in gen.llm, then, the eval helpers made explicit, this
split. memo.clj's synthesis module builds on both (its REPL agent,
memoclj-pcw0); memo.clj's core depends on neither. Built as
gen.codegen.session (a namespace kept across turns; :read, :repair,
:vet, :explain given the throwable, namespace and forms, :show) and
gen.codegen.process (tools in a JVM of their own, restarted when a
response is tainted, late or lost), both experimental; the harness's
REPL tools, grader and driver run on them and on gen.llm's three, its
prompts rendered as before, character for character.

### The kmet integration's plan: plan now, couple thinly

(owner, 2026-11-04, "write the plan with your suggestions"; genclj-ytew,
docs/research/kmet-integration.md §10).
- **Why now.** kmet (upstream a4f7102d) has the seam M6 lacked: the
  `:llm` composition row, replaceable by a classpath plugin. What it
  still lacks is optional for gen.clj: log-probabilities (gen.clj keeps
  them), persistent eval (ext/codegen's sessions, or `clojure_eval` into
  an nREPL), and subagents (agent handles and server mode).
- **Coupling.** The integration couples to kmet only through the `:llm`
  contract (`kmet.ai.llm/send-message`'s options and callbacks) and
  composition rows, never through kmet's internal namespaces or its
  extension API. kmet stays at a pinned commit, bumped deliberately with
  its JVM tests.
- **Roles.** Research runs in gen.llm.agent (exact keys,
  log-probabilities, K candidates); kmet is the shell a person uses.
  gen.clj owns the machinery. memo.clj supplies a language, its hooks
  and its models through its synthesis module; its core stays free of
  LLM code.
- **The question.** Does a small local model become a useful modelling
  assistant when it can evaluate code? It is measured on the codegen
  bake-off's tasks and memo.clj's, against fixed baselines at matched
  compute: one shot, best-of-K, and the REPL agent.
- **Models.** Qwen3.6-35B-A3B on the MacBook or the RTX-6000 machine; Qwen3.5-0.8B for
  development on this Mac.
- **Where.** An `ext/kmet` module in gen.clj for the spike, with kmet as a
  git dependency at the pinned commit (EPL-2.0, a dependency, never
  copied); the final home is decided at the spike's end (genclj-wqka).
- **The probe's model endpoint is local:** llama.cpp's server, rebuilt
  from the pinned v0.7.0 with its tools on, serving the same GGUF. No
  hosted model unless the owner chooses one.
- **Stages,** each gated on beating the one before at matched compute:
  0. the question, tasks and baselines;
  1. a zero-code probe (kmet's agent, `clojure_eval` into an nREPL
     holding gen.clj and memo.clj);
  2. the `:llm` plugin spike;
  3. a gen.clj eval tool row;
  4. best-of-K with verifiers;
  5. repair and SMC.
- Nothing is built until the owner's go on the stages.

### gen.clj 1.1.0: a consolidation release

(owner, 2026-11-04, "yes to all three", on the lead's recommendation;
genclj-qsa2).

#### 1.1.0, not 1.0.1

The stable surface grew since v1.0.0
(the snapshot: kernel/run's `:device`, `laws/*score-scale*`), and
`laws/close?`'s tolerance change is a minor release's ("The laws'
names", above). A maintenance branch off v1.0.0 would carry nothing a
dependent waits for. memo.clj's ext/synth pins main for gen.llm.agent
and ext/codegen while memo.clj's core and GenGDT pin v1.0.0: the tag
puts them on one coordinate again.

#### Its scope, in order

(1) The work in progress closed: tempered
SMC (its changepoint question moved under genclj-zjz7), the REPL
codegen's report, the Metal device programs' parity cells, the book's
appendices 061, 062 and 065. (2) Gen.jl 0.4.9 parity: the dense metric
for HMC, MALA and NUTS is the release's addition, and the local oracle
moves to Gen 0.4.9 (the download approved). (3) The book's bugs in
stable namespaces, fixes that keep the shape: serialize's round trips,
exact proposals through an address-less splice,
with-internal-proposal's regenerate, MLX's gathers loud on an
out-of-range index, HMC's missed divergence. (4) 1.0.0's CUDA known
issues: TF32 off by default, the abort at exit, half-float sums.
(5) Docs: the README's times, the API appendix 063, the owner's review
of Part V and the appendices.

#### Off the critical path

the device programs (experimental, landing
on main for a later 1.x) and kmet's stages (each on the owner's go).

#### The gate, lighter than 1.0's

(genclj-w2pg): every test on the JVM
backend, on MLX's CPU and on Metal; CUDA once, on the Thor (the RTX-6000 machine
runs on the CPU only for now, owner, 2026-11-04) or on the RTX-6000 machine when its
GPU is back; the book on the JVM backend; memo.clj (its ext/synth
too) and GenGDT green against the release candidate. The release
notes lead with the Breaking `=` on 0-d MLX arrays: gen.backend.mlx
is internal, but the change reaches user code that compares draws.

#### As the release went

(owner, 2026-11-04 evening). The owner's
review of Part V and the appendices (genclj-7yjc) comes after 1.1.0
("maybe this can come after"): the tag waits for the gate's checks
only. MLX's gathers stay clamped in 1.1.0 rather than becoming loud
(above; genclj-08un decides what follows). CUDA ran twice: the full
battery on the Thor, and on the owner's go a short check on the
RTX-6000 machine (its GPU back), because the Thor's matrix suites
pass even with TF32 forced on. The MLX CPU float64 light tier ran on
the MacBook (owner, "can we run some of the tests on the macbook"),
beside the Mac's Metal and float32 runs.

#### 1.1.0, then 1.2.0 with memo.clj

(owner, 2026-11-04 night). "We
push v1.1.0 of gen.clj first, and then add more features to v1.2.0
of gen.clj - then let memo.clj v1.2.0 depend on that gen.clj": this
release stays 1.1.0 (a relayed request to renumber it 1.2.0, to match
memo.clj, was withdrawn); gen.clj 1.2.0 is a later release with more
features, which memo.clj 1.2.0 (memoclj-s3wr) pins.

#### One chain fast on MLX, and GPU speed

(owner, 2026-11-04, "go with option 1", on the lead's recommendation;
genclj-fpnf, genclj-h8rn).
One HMC chain (L=32, 1000 steps) ran in 21.9 ms as a Metal program (one
GPU thread, 32,000 dependent leapfrog steps), 5.2 s on the handler on
MLX, against GenJAX's 5.82 on the M4's CPU; the JVM lowering's program
runs it in under 1 ms, but kernel/run's `{:device :jvm}` took 4.4 ms on
the JVM backend and 7.2 (139 with a :keep per step) on MLX.
- **In 1.1.0 (option 1):** a single chain runs on a CPU core (the JVM
  lowering) whatever the backend, and kernel/run's device path costs
  little beyond its program: its entry and its result (traces and
  kept values) built on the host, MLX arrays made only where the
  result needs them. Whether a run picks its device itself (the
  epic's decision 5) stays open: the owner's call, with the lead's
  recommendation once the fast path is measured.
- **Explored, in 1.1.0 if it is ready, else after (option 2):** the
  Metal program faster for few chains: a simdgroup of lanes per chain
  sharing a step's data-parallel work and its sums, and fewer
  statements a leapfrog step. The owner would like a very good GPU
  speed in 1.1.0, and is prepared to skip it.
- **As measured** (2026-11-04): option 1 is built (genclj-rtcd: one
  HMC chain, L=32, through kernel/run 2.4 ms on the JVM backend and
  4.7 on MLX). A Metal kernel runs 1 to 1024 chains in the same 22
  ms, so one chain's time is one instruction stream's latency, and
  GenJAX's GPU is slower still there (101 ms on the RTX-6000 machine).

#### 1.1.0's GPU work, the dense metric and the order

(owner, 2026-11-04, "go with your recommendations"; genclj-qsa2).
- **GPU speed is many chains and many particles on Metal,** where the
  GPU is already far ahead but reachable only through ext/device's
  internal API. First importance sampling's `:metal` result built on
  the host (as kernel/run's, genclj-rtcd). Then
  **`kernel/run-particles`**, a new stable function beside the
  `*-particles` moves (not kernel/run taking a collection, whose
  result would change shape with its argument): S steps of the
  kernel's batched form over every particle of a gen.batched
  collection, returning `:particles` (the last), with `:keep` `:draws`
  per chain (what gen.inference.diagnostics reads), and
  `:acceptance-rates` and `:divergences` per chain; `{:device
  :metal}` runs every chain in one kernel. Chain i runs on `(rng/split
  key n)[i]`, as device runs do (decision 3), so Metal equals the
  handler's chains to float32 rounding; if the batched form draws
  otherwise, the lead brings it back. If it grows, 1.1.0 ships
  without it. One chain's GPU latency (lanes, genclj-98s2) waits
  until after 1.1.0.
- **The dense metric** (genclj-q9ge, Gen.jl 0.4.9's #559): the stable
  `:inv-metric` (Stan's M⁻¹) takes a dense matrix too; no `:metric`
  option. Gen.jl's `metric` is M: appendix 061 and the docstrings say
  so. The kernels only (HMC, MALA, NUTS); Stan's dense_e warmup
  adaptation and MALA's step-size adaptation are beans of their own,
  after 1.1.0.
  As built (2026-11-04, the lead; the constructor's name and
  arguments confirmed by the owner the same day, a new stable var):
  `(mcmc/dense-inv-metric addresses matrix)`, a record whose factories
  are ^:no-doc, its addresses (or paths) the rows' order; a bare `[n
  n]` matrix only for one selected choice of n elements (unless the
  choice has that shape: its variances), so a Gen.jl model with a
  matrix over several choices meets a loud error naming the dense
  form. Refused unless finite, symmetric and positive definite.
  Symmetric (owner, 2026-11-04, on the lead's recommendation after a
  side agent's note): |m - mᵀ| ≤ ε·max(1, max|m|), ε 1e-8 in float64
  (Stan's check_symmetric, in practice) and 1e-5 in float32 (about a
  hundred rounding steps: a metric computed in float32, by la/inv
  say, is off-symmetric by its rounding), and the kernels then use
  (m + mᵀ)/2, exactly symmetric. Stan's absolute 1e-8 as first built
  refused such float32 metrics. The
  factors (L with M⁻¹ = L·Lᵀ, and L⁻¹) are computed once a move on
  the matrix's own backend, then made arrays of the current one, so a
  staged, batched or compiled step takes them as constants (a device
  run with a dense metric is the handler's bit for bit). nuts' warmup
  refuses a dense metric unless `:adapt-metric?` is false.
- **Gen.jl's position step** (its hmc leaves M⁻¹ out, genclj-u420):
  the lead drafts a note with the line and a small example; the owner
  posts it.
- **The order:** the dense metric; importance sampling's host result,
  then kernel/run-particles on Metal; the book's bugs in stable
  namespaces; the CUDA known issues on the Thor; the docs; the gate.
- **Revised: many chains experimental first, and a batched device
  result (owner, 2026-11-04, on the lead's recommendations).** The
  keys question came back as the lead said it might: a gen.batched
  run draws one key per site with the particle axis inside the draw
  (decision 3), so the kernels' batched forms cannot draw a device
  run's chains, `(rng/split key n)[i]` each. So 1.1.0 ships an
  experimental `gen.inference.device/run-chains`: n chains from n
  traces, chain i on `(rng/split key n)[i]`, `:metal` in one kernel
  and `:jvm` on every core, falling back to kernel/run per chain, so
  it equals per-chain kernel/run (bit for bit on the JVM backend in
  float64, to float32 rounding on the GPU). The stable
  kernel/run-particles waits until after 1.1.0, for per-particle keys
  in the batch tracer, so that its batched handler draws the device's
  chains. And importance sampling with `:device` follows `:batched?`:
  asked for batched, its result holds `:particles` (a gen.batched
  collection over the program's outputs, as the batched handler's
  result does) in place of n scalar traces; draws stay per particle.
  As built (2026-11-04, the lead; experimental, for the owner's look):
  `(device/run-chains kernel traces opts)` returns `{:traces` (or
  `:draws`) `:acceptance-rates :divergences}`, one entry per chain,
  plural keys as the 1.0 review's. The lowerings gained per-run
  inputs (a program's `:per-run` data inputs, run r's at r times the
  input's size: each chain's start), in gen.device.scalar for both
  dialects. Because a function `:keep` makes a trace per kept step
  (4096 chains of 1000 steps are four million traces: seconds on the
  JVM backend, past Metal's buffer limit on MLX), run-chains also
  takes `:keep-choices [address …]`: those choices read from the
  program's outputs as host values, no trace made, equal to a `:keep`
  of `{address #(% address)}` (its handler fallback runs exactly
  that). Not `:keep` with an address: a keyword is a function, and on
  a trace it reads the record's fields. A carried leaf that is one
  object in every chain's start (the observations, typically) is one
  input the chains share, so large data costs no copy per chain and
  no fallback (a side agent's note, 2026-11-04; the owner: "not
  interested in solutions that give us much longer time / large
  difference compared to GenJAX"); a large per-run input is copied
  into its run's scratch buffer. `:metal`'s outputs are copied into
  Java arrays (one copy, nothing boxed) when first read, and a 0-d
  input goes to Metal as [1] (MLX hands a kernel a 0-d array as a
  scalar, which the generated code cannot index: the static parity
  model's Metal failure, genclj-649c). **The fallback is batched**
  (owner, 2026-11-04, on the lead's recommendation after a side
  agent's note): a model the device cannot run (about 4 in 10 of the
  census's continuous models for MALA) runs as one batched run of the
  kernel's batched form, about a hundred times faster than chain by
  chain (4096 HMC chains of 1000 steps: ~1.6 h chain by chain on the
  JVM backend). Its draws are the batched run's, not the device's
  chains (the same distribution), and `:device-fallback` says so,
  until per-particle keys (genclj-asog) make them the same; a kernel
  with no batched form runs chain by chain.
- **TF32 off by default, as built (2026-11-04; genclj-p3pr):**
  MlxRuntime, which already set MLX_USE_CUDA_GRAPHS=0 through libc's
  setenv before MLX loads, sets MLX_ENABLE_TF32=0 too, without
  overwriting the environment's value; `-Dgen.mlx.tf32=true|false`
  decides over both. MLX reads the variable once, at its first matrix
  product. It gates TF32 on CUDA and, on Macs with neural accelerators
  (MLX's NAX), Metal's reduced-precision float32 products as well:
  **full float32 on every device** (the lead's while the owner was
  away, confirmed by the owner, 2026-11-04 evening), the M4
  unaffected, a newer Mac slower but agreeing with CUDA and the CPU.
  The owner left conjugacy, Laplace and particle MCMC (experimental)
  provisional until they move into the stable tier. The test starts fresh JVMs (the JVM's environment
  snapshot may come after the setenv).

## 1.2.0 (2026-11-06)

### GenLM's gaps

(owner, 2026-11-04): the survey's top five become
beans for after 1.1.0 (an epic): a critic at the drawn token in
token SMC, AWRS for constraints without a mask, the potential
contract as laws, potential combinators, and context-free grammar
potentials (docs/research/genlm-survey-2026-11-04.md). The owner then
made that epic **GenLM parity** (genclj-ygbm): every bean whose main
purpose is a GenLM counterpart moves under it (token SMC's
rejuvenation and batched potentials, the byte-level generative
function, structured output, sampling filters, constrained codegen,
instaparse grammars as constraints); related beans whose purpose is
elsewhere (LLM training, engine speed, M7) stay and are listed there.

### 1.2.0's theme: language models

(owner, 2026-11-05, "good idea for the 1.2.0 milestone", on the lead's
recommendation; genclj-ucfg).
gen.clj 1.2.0, which memo.clj 1.2.0 pins, is a language-model
release: llama.cpp v0.9.0 (its unified KV cache for token SMC and
lockstep sampling), GenLM parity's core (the potential contract as
laws, potential combinators, a critic at the drawn token, AWRS,
instaparse's regular subset to DFAs, then context-free grammars by
Earley), Qwen3.8's thinking template, and the small decisions and
fixes after 1.1.0 (get on arrays, MLX's gathers, the MLX pin, the
owner's book review). The device programs keep landing as
experimental and do not gate it: the owner confirmed the theme and
that speed is not part of 1.2.0's milestone ("ok with speed not part
of the 1.2.0 milestone"). memo.clj 1.2.0's needs from gen.clj, as
the scope's floor: its tracker names the tag and llama.cpp v0.9.0
(memoclj-s3wr, memoclj-670y); asking its session waits (owner,
2026-11-05).

### 1.2.0 worked autonomously, CUDA on fenrir

(owner, 2026-11-05).
"keep working on the entire milestone v1.2.0 autonomously", after the
lead's plan (tpvn, then the potential laws, the GenLM core in order;
"decisions sound good"): decisions that come up are PROVISIONAL here,
listed in the report. The milestone's GPU runs, its gate's CUDA
battery included: "I grant the gpu run on the fenrir and other
machines if needed - but I prefer the fenrir only". So CUDA goes to
fenrir's RTX PRO 6000, the Thor only if fenrir cannot serve. Pushes
still wait for a go.
Approved (owner, 2026-11-05 afternoon, "I approve all the provisional
decisions"): the seven the milestone made, each marked below
(genclj-fzez's loud lookups, genclj-08un's opt-in gather check,
temper's kept zeros and after's unclosed sequences of genclj-f3dv,
AWRS's recursive default, ext/instaparse on :test, the CFG's
byte-level liveness).

### Potential combinators

(2026-11-05; genclj-f3dv; the marked choices approved by the owner,
2026-11-05: "I approve all the provisional decisions"). gen.llm.potential
gains
genlm-control's combinators, each checked by gen.laws.potential:
`temper` (weights times a finite beta of either sign; a zero weight
stays zero, so the support is kept and beta 0 is the indicator:
approved, genlm's power is p^beta with no rule for negative beta),
`normalize` (next weights less log Z, Z over the tokens and the end
but the potential's end tokens, for which a completion's end stands;
prefix the sum along the context), `prompted` (an LM's
log-probabilities after a prompt; the end is log Σ over its end tokens,
default the LM's :eog, which it declares), `coerce` (a potential over
the 256 bytes read through a vocabulary's byte trie, one next-byte row
per byte prefix of weight; end tokens hold no text and a context
holding one has no weight, as a grammar's), and `after` (a potential
over the tokens after the first close token, those before free; a
sequence that never closes has no weight as a whole: approved, the
alternative weighs an unclosed sequence as an empty answer). Temperature
and the tempered sampler are compositions (`temper` of `prompted`,
`normalize` of that), not functions of their own; genclj-paid keeps
the samplers.

### AWRS in token SMC

(2026-11-05; genclj-oigp; the default approved by the owner, 2026-11-05: "I
approve all the provisional decisions"). `gen.llm.awrs` ports the paper's
bounded-cost
estimators (Lipkin et al., COLM 2025, App. H.3): recursive (Def. 10,
the default: never more checks than the geometric's on our measure,
App. H.3.4) and geometric (Def. 9, read through its with-replacement
equivalent, Clipped WRS's (s - 1)/(n - 1) and s/(n - 1)), each with an
optional :max-rejects. Token SMC's `:proposal :awrs` takes the
potential's twist as the condition (a token of twist -Inf is
rejected), so a soft potential works too: the weight is log Ẑ plus
the accepted token's twist, an unbiased estimate of the masked
proposal's. Draws are on the host from the step's key (one uniform per
draw, a second for the phantoms), and the checks, potential calls, run
on the calling thread. Def. 2's two-loop AWRS is left out: the
geometric estimator unbounded is its alternative form (App. H.3.2).

### instaparse grammars as constraints

(2026-11-05; genclj-kb88; where it lives approved by the owner, 2026-11-05:
"I approve all the provisional decisions"). The bean left
the place to the owner: the converter reads only instaparse's grammar
data, so it could sit in gen.llm with no dependency, but reading
EBNF text, and testing against instaparse's parser, need instaparse.
Chosen: a module, `ext/instaparse` (`gen.llm.instaparse`: `term`,
`grammar`), with instaparse 1.5.0 (EPL 1.0) its one dependency, as a
git dependency with `:deps/root "ext/instaparse"`, and on the `:test`
alias (its tests run in the battery, and the book's chapter 044 uses
it). The core's dependencies are unchanged. Recursion unrolls only
to an explicit `:max-depth` (refused without one); lookahead is
refused; ordered choice is a union, as instaparse's is. Two
differences from instaparse's parser are documented, not hidden: a
regex terminal is every string its regex matches (instaparse takes
the first, greedy match), and instaparse 1.5.0's counted repeat of a
parser that matches the empty text rejects the empty text, which the
language has.

### MLX's gathers: the clamp kept, an opt-in check added

(2026-11-05; genclj-08un; the owner's call, approved by the owner,
2026-11-05: "I approve all the provisional decisions").
The owner's options: keep the documented clamp; an opt-in check (a
system property, a read per gather); or loud by default through an
error flag raised at the next evaluation (rule 6's choice, Breaking).
Chosen: the second, `-Dgen.mlx.check-gathers=true`, beside the clamp,
since it is small, reversible and breaks no dependent, and makes a
suspected out-of-range gather findable now; loud by default is left
for 2.0 (or a later 1.x with a Breaking line), when the device
programs' error flags (genclj-i2c4) can carry it without a read per
gather. Not checked inside mlx_compile's trace, where nothing can be
read.

### Arrays are no Clojure collections: get, nth and count are loud

(2026-11-05; genclj-fzez; the owner's call, approved by the owner,
2026-11-05: "I approve all the provisional decisions"). The owner's options
were loud
everywhere, or reads that work where an array is concrete. Chosen:
loud everywhere, on every array kind (JvmArray, MlxArray, the tape's
Tracked, Staged; Batched by its batched-branch error), as ILookup and
Indexed whose methods throw `:not-a-collection` naming `->clj`,
`item` and `take`. Why: rule 6 (get answered nil), one behaviour on
every backend and inside gen bodies (a working read on concrete
arrays would make code pass on the JVM backend and fail staged or
batched), and no host reads that look like collection access. Breaking
in 1.x's CHANGELOG (get's nil was a behaviour); the battery had no
use of it. memo.clj's side: memoclj-fksr (its Qwen3.8 agent read
arrays with get).

### Context-free grammars by Earley

(2026-11-05; genclj-h5rb; byte-level liveness approved by the owner,
2026-11-05: "I approve all the provisional decisions").
`gen.llm.cfg`, in the core with no dependency: rules over bytes
(keywords, strings, `(cfg/regex r)` inlined as right-linear rules from
its minimal DFA), unproductive rules dropped so a non-empty Earley set
is an exact viable prefix, Aycock and Horspool's nullable handling;
`cfg/potential` weighs contexts 0/-Inf and its masks sweep the token
trie with an Earley chart per depth (`gen.llm.bytes/sweep-with`).
Decided: liveness is the bytes', not the tokens' (a vocabulary
lacking a byte the language needs can strand a completion), since
token-level exactness over a context-free language has no finite
exploration as the DFA's has; a byte-level BPE vocabulary never
strands. No state between calls: prefix and complete recognize from
the start (pure, rule 3), at a cost that grows with the text
(stateful potentials are genclj-9dvz's). Measured at Qwen3's
vocabulary: a JSON grammar's mask in 0.1-0.7 ms at a value's start,
a prefix of 661 bytes in 0.6 ms (`gen.instaparse-bench`); but inside a
string, which the bench did not then reach and where most of a JSON
text's bytes are, 39-88 ms with the bench's `[a-z ]*` strings and
238-363 ms with JSON's `[^"]*` (the 2026-11-06 health check, and
genclj-jq2d on a loaded M4), since nearly every token is allowed there
and each trie node visited built an Earley set. `gen.llm.instaparse/cfg`
reads any instaparse grammar without lookahead into one. **PROVISIONAL
(2026-11-06, genclj-jq2d; the owner to confirm):** the three options
together: a regex element a lexeme its DFA scans (an item carries the
DFA's state), the Earley sets a lazy DFA (successors kept by byte, a
canon per chart, the chart only the sets that predict), and the
lexer/parser split as masks in parts (per lexeme state, the tokens
that stay inside it, made with the potential; only the tokens that
end a lexeme or begin with a byte an item waits for are swept). Masks
stay exact. Measured the same way: inside a string 0.2 ms (from
358-363), at a value's start 0.06-0.2 ms, 0.6-0.8 ms past 661 bytes
(the context's recognition from the start); the potential made in the
trie's time (docs/research/llm-grammar.md, the 2026-11-06 addendum).

## 1.2.1

### 1.2.1: the health check's fixes

(owner, 2026-11-06). "file these
as beans under the 1.2.1 milestone", after the lead's health check of
v1.2.0 (c0fc80c): the milestone genclj-9mq3, its findings in five epics
(silent wrong results and loud-error gaps; the tests; 1.2.0's LLM work
followed through to a consumer; device programs and packaging; docs and
housekeeping). Not yet decided: the release's scope (which fixes that
make code throw wait for 1.3.0) and the numerics edge semantics, each
the owner's call in genclj-uzxp with the lead's recommendation.

### 1.2.1's owner calls

(PROVISIONAL, owner to confirm; genclj-uzxp).
*Item 9, the scope, confirmed by the owner (2026-11-06, before the tag,
asked after a side agent's note that a patch release breaks nothing):
"Keep as built": every bean under the milestone in 1.2.1, the changes
that make code throw labelled Breaking (only code that was already
silently wrong), unknown option keys in 1.3.0.*
*Items 1-8, the numeric edges, approved by the owner (2026-11-06, before
the tag, after a side agent's note listing them: "I am fine with these
if they are based on good decisions"), on the grounds the lead gave:
floor-div as Python, NumPy and JAX (the only rule that keeps x = y*q + r
with mod); int64 overflow loud where the JVM backend already threw raw;
empty extrema refused as NumPy, JAX and MLX refuse them; an infinite
value -Inf in every family (every proper density's limit); randint exact
(the float formula left its range). The arithmetic calls under them,
listed below (randint's bits, NumPy's int arange, the infinite value
outside every support, the parameters kept at their limits), with them.
The rest below stay PROVISIONAL.*
The owner, 2026-11-06: "work on the 44 beans under the milestone,
genclj-9mq3 - work fully autonomous and cross them all off and then
release v1.2.1". Taken as the answer to the scope (item 9): every bean
under the milestone ships in 1.2.1, the inference parameter checks
(genclj-mhgn) included. The rest are the lead's recommendations, taken
under the overnight policy:
1. *Non-finite parameters* (in 1.2.1): a distribution refuses a
   non-finite host parameter (and an array's, when `concrete?`) where
   its code gives NaN there; a family whose code computes the limit
   keeps it, and a -Inf logit stays a zero probability. Breaking by
   the CHANGELOG's definition, but only for code that scored NaN.
2. *int64 overflow* (in 1.2.1): the JVM backend throws `ex-info`
   (`:reason :overflow`) where an int64 result leaves the range, in
   place of a raw IllegalArgumentException on read; MLX's wrap is a
   documented divergence in docs/array-api.md §13.
3. *Float floor-div* (in 1.2.1): NumPy's rule, floor((x - mod(x, y)) /
   y) with its rounding, so x = y*q + r holds: `(floor-div 1.0 0.1)`
   is 9.0. A Changed line.
4. *max and min over an empty axis* (in 1.2.1): loud on every backend
   (`:reason :empty-reduction`), as NumPy and JAX raise.
5. *`=` on arrays:* the JVM's bitwise, MLX's numeric equality stay; the
   divergence is documented in §13.
6. *Unknown option keys* wait for 1.3.0 (genclj-vux3): refusing
   them breaks code that was right (a shared options map).
7. *Loud lookups through Indexed:* genclj-fzez's design stays;
   `counted?` and `indexed?` answering true is noted in §13.
8. *solve's vector rule* stays as documented (a right-hand side one
   axis short is a batch of vectors); inv is fixed by broadcasting the
   identity (genclj-l0n4); NumPy 2's rule revisited for 2.0.
The beans' own owner calls, the same way: CUDA's lowering parked
(genclj-z1gj: no consumer, its tests skipped by name), DESIGN's §9 log
in a file of its own (genclj-vp8u), Batched's read flag volatile
(genclj-vajv), the git deletions done (genclj-wg76: merged branches
only, the stale stash saved as a patch first), memo.clj told and
asked through its session (genclj-fshl, genclj-e3q9).
The calls made while building them, PROVISIONAL the same way (each in
its bean):
- *randint* (genclj-psgi): exact integers, low + floor(k*r / 2^b) in
  int64, k from the default float's uniforms (one float64's 52 bits;
  three float32s' 23 bits each). Integers agree across backends that
  share a default float, as every sampler's draws do; the JVM's
  float64 draws are unchanged (the book's checked values with them),
  float32's change. An empty range, a fractional host bound and a host
  bound past int32 on an int32 backend are loud.
- *Integer arange* follows NumPy (and MLX), a bool arange is refused on
  every backend (genclj-06ba); an int64 result past +-2^63 throws
  `:overflow` on the JVM backend, 2^63 itself (the double nearest
  Long/MAX_VALUE) accepted; `rng/key` refuses any float, 2.0 too, as
  JAX's PRNGKey does; `rng/key-data` of a staged key is loud
  (genclj-kdau). Empty extrema are refused in gen.array before any
  backend, argmax and argmin with them; logsumexp keeps -Inf and
  softmax an empty array.
- *Infinite values* are outside every distribution's support: log
  density -Inf, gradient never NaN (normal's stays its +-Inf limit)
  (genclj-zkbw); the DSL's exp step stands in 0.5 outside its domain.
  The non-finite parameters kept: truncated-normal's infinite bounds,
  wrapped-normal's +Inf sigma (the uniform on the circle), a -Inf
  logit (item 1). defdist also refuses a repeated parameter
  (genclj-8326). genclj-tggk: no change, the gains measured at 1-12%
  against a new defdist mechanism; the real cost (mvnormal's
  positive-definite? check) is a bean of its own.
- *agent/run* takes the template's options in its opts, beside
  `:thinking?` (genclj-fshl); `coerce` and `after` read the
  potential's next weights at the empty context when made, so a
  prompted LM's prompt is read at construction (genclj-kdau); the GC's
  two branching tests retry System/gc for up to 60 s under a new tag
  `^:gc` (genclj-n2ia).
- *The GFI's* (genclj-k3xn, -7y8r, -kdau, -8326, -vjbo): gen refuses a
  reserved name as a parameter (destructured too) and as an enclosing
  local the body reads (`:reserved-name`); gen's own bindings are
  marked for gen.dynamic.guard, so nested gens still read theirs. The
  address order: numbers by value whatever their class (NaN last),
  Comparable values of one class by value, the rest by class name, a
  tie by class, then equality, printed form and hash. A negative
  selection index throws in every kind (Recurse's project too); one
  past the elements still selects nothing. Every Switch branch is
  checked. gen.serialize keeps the 1e-4 float32 tolerance of its score
  check (`:numerics` names a backend, not a device, and MLX's CPU and
  GPU round differently), its docstring corrected; `sample-states` of
  an impossible trace throws (`:impossible`) where it returned fewer
  states.
- *Inference's* (genclj-mhgn, -dm71, -vjbo, -i1jl): `:invalid-eps`,
  `:invalid-L` and `:invalid-tau`; an impossible start's tempered weight
  is 0 by generate's projection law, whatever the internal proposal;
  tempering's accept rejects a -Inf target and throws on any other NaN
  (`:nan-log-alpha`); tempering's per-step keys renamed `:ess-by-step`
  and `:acceptance-rates` (experimental); `kernel/checked-kernel`
  `^:no-doc` (the var stays); the metric parsing in
  gen.inference.mcmc.metric (internal), its record still
  gen.inference.mcmc's.
- *The device path's* (genclj-qucg, -cxrx, -ri2d, -z1gj, -2j7v):
  kernel/run `{:device :metal}` for one chain runs the JVM lowering
  and reports `:device :jvm`; CUDA parked as its tests skipping by name
  (`-Dgen.cuda.parked=false` runs them), its manual frees documented;
  the parity models in test/ (gen.parity-models), shared by the bench
  and ext/device's tests; one bounded LRU cache of 64 compiled
  programs (gen.inference.cache, internal) for mcmc and the device,
  where each emptied its own whole; a run error the device finds is
  replayed on the handler for that chain, the handler's error thrown,
  else a fallback (the difference is rounding); importance sampling's
  `:batched?` with `:device` returns gen.batched particles.
- *The tests'* (genclj-7z2s, -k3a8, -ruh8): one set of 5-SE helpers
  in gen.test-support; HMC's and NUTS's negative controls planted
  with with-redefs-fn on gen.inference.mcmc's private normal-lp and
  energy (no public input plants them: a wrong gradient leaves HMC
  exact); `-Dgen.require-native=mlx,llama` counts skips per test and
  fails a run where a named suite skipped one or ran none
  (gen.test-runner, :test's exec-fn).
- *CFG masks* (genclj-jq2d): all three options built together (regex
  elements as lexemes scanned by their DFA, Earley sets as a lazy DFA,
  masks made in parts per lexeme state), exact; see the Earley entry
  above. An Earley item is an origin and a numbered position, so an
  alternative has no length limit (genclj-mkos). llama.cpp v0.9.0
  measured against v0.7.0 on the M4: no gain for lockstep and SMC
  branches, 5-16% slower at 4-16 branches after a long prompt
  (genclj-ajse; 1.2.0's claim corrected in the CHANGELOG).
- *The refactors'* (genclj-i1jl, behaviour unchanged, checked by
  differential runs against the old code): gen.combinators over six
  internal parts (calls, elements, chain, switch, recurse, wrap), its
  records, constructors and class names kept in the facade; Unfold and
  Scan one chain walk; gen.dynamic's execute as functions over a run
  context, the static graph walk in gen.dynamic.static/run; selections
  held by a partition through a ^:no-doc protocol gen.partition extends.
  gen.backend.mlx over six internal parts (array, memory, operands,
  linalg, random, transforms; the MLX array class is now
  gen.backend.mlx.array.MlxArray); gen.array's scan, compile, vmap and
  particle functions in gen.array.transform behind the same vars;
  Threefry's schedule and erfinv's coefficients written once as data.
  `-linalg` keeps its arity: gen.array.protocols' methods stay through
  1.x (the 1.0 promise), so a ^:no-doc `p/linalg-operands` splits a
  call's arguments by op in place of `(remove map? args)`; moving the
  options to their own argument waits for 2.0. gen.array.stage no
  longer requires gen.choicemap: its constants follow the map's own
  order (the program's values unchanged).
- *memo.clj's needs* (genclj-e3q9; memo-clj-96's answer, 2026-11-06):
  none of 1.2.0's GenLM tools is used by memo.clj, a floor of zero; its
  one candidate with evidence is a bracket-balance or CFG constraint over
  an agent tool call's code (a draft bean under genclj-p55l, the owner's
  call). The lead's recommendation stands: no new GenLM features until a
  consumer uses one.

## 1.3.0

### mlx-node parity, on mlx-c and our own C++

(owner, 2026-11-07, genclj-lv7t)

The owner's words: "make a milestone bean of that we have complete 'on-par'
functionality in gen.clj as mlx-node offers, but that we have implemented it
using mlx-c and/or our own C++ (for paged attention and continuous batching)".

**The aim.** gen.clj gets every feature mlx-node has:
- its model families and modalities;
- its engine, with paged attention, continuous batching and speculative
  decoding;
- its training: SFT, and GRPO with its loss variants;
- its server, CLI and apps.

**How it is built.**
- Over mlx-c, through gen.clj's generated FFM binding.
- Where mlx-c cannot reach, over a C++ library of our own on MLX's C++ API:
  new MLX primitives with their own GPU dispatch, such as paged attention and
  the continuous batching over it.
- mlx-node is never a dependency.

**Why this split.** mlx-node's C++ was read from its source (genclj-q4xv,
2026-11-07). It writes its own C ABI of 377 functions over MLX's C++ API.
mlx-c reaches all of it except paged attention's custom primitives:
- its op wrappers;
- its gradients, compile, vmap and gradient checkpointing;
- its fused blocks;
- its custom Metal kernels (through mlx_fast_metal_kernel).

**What this supersedes.**
- The lead's recommendation of 2026-11-06 (in genclj-q4xv): aim at what the
  research needs, not mlx-node's breadth, and leave out vision, audio, OCR,
  reranking and the server.
- DESIGN §6's reason for llama.cpp first, "it keeps gen.clj a PPL rather than
  an LLM inference engine". It still explains M3's order, but no longer limits
  scope.
- The 1.2.0 entry's "no new GenLM features until a consumer uses one" is about
  GenLM's features, so this does not touch it.

**Open, in the milestone:**
- where each part lives (the lead recommends gen.clj's core for the engine,
  families and training; ext/ modules for modalities, server, CLI and apps;
  native/ for the C++ library);
- whether on par takes in the desktop app, dashboard, agent and privacy filter;
- the order against 1.3.0, 1.4.0, M6 and M7;
- porting mlx-node's MIT kernels with attribution, or writing our own.

### 1.3.0, 1.4.0 and 1.5.0

(owner, 2026-11-07; genclj-lv7t, genclj-ytew)

The owner, in response to the milestone above: "I think this should be come
gen.clj v1.3.0 (moving the kmet agent to v1.4.0 instead, and the MSA to
v1.5.0) / on par would not mean desktop app, dashboard - and the 'agent' will
later use the kmet agent instead".

**The releases:**
- 1.3.0 is mlx-node parity (genclj-lv7t).
- 1.4.0 is the kmet integration (genclj-ytew, until now M6).
- 1.5.0 is the MSA in gen.clj, built on kmet.

This settles the question of 2026-11-06, which was never recorded: whether
1.3.0 should be kmet hosting gen.clj and 1.4.0 the MSA (order B), or the
reverse (order A). The new order is neither.

**Out of 1.3.0's parity:**
- mlx-node's desktop app and its dashboard.
- Its coding agent: kmet is the agent, in 1.4.0.

**Lead's calls, PROVISIONAL:**
- Unknown option keys (genclj-vux3) stay in 1.3.0. 1.3.0 is the next minor
  release, which the 1.2.1 entry waited for.
- genclj-gvfa (the MLX-native engine) and genclj-q4xv (training) move under
  1.3.0's milestone.

### 1.3.0: where the parts live, its hosts, the M5

(owner, 2026-11-07; genclj-lv7t)

**Where the parts live.** The owner: "go with your recommendation".
- **gen.clj's core:** the engine, the model families and training, as
  gen.llm.mlx under gen.llm's protocols.
- **ext/ modules:** the modalities, the server and the CLI, so rule 8's
  minimal dependencies hold for the core.
- **native/:** the C++ library for paged attention and continuous batching,
  as a native artifact of its own.

**Hosts.** The owner: "can you use the fenrir for this? (and Thor for
backup...) you may download as much as you need to the fenrir and the Thor -
but check so space is available".
- fenrir does 1.3.0's CUDA and big-model work, with the Thor as backup.
- Downloads go to either, after a space check.
- Space checked 2026-11-07 15:46:
  - fenrir: 169 GB free on its shared 1.9 TB root (91% used); GPU idle.
  - Thor: 146 GB free; its Hugging Face cache already holds 325 GB, including
    MLX checkpoints of Qwen3.5-0.8B, Qwen3.6-35B-A3B, Qwen3.8-27B and a
    Qwen3.8-Flash-Next.
- Lead's call, PROVISIONAL: leave at least 50 GB free on fenrir's shared
  disk. A model that does not fit waits for the owner (deleting old caches is
  the owner's call).

**M5.** The owner expects access to an M5 Mac in 2-3 weeks. The int8
Neural Accelerator kernels stay in 1.3.0, tested then.

### 1.3.0's kernels: ported from their origins, everything around them ours

(owner, 2026-11-07; genclj-lv7t)

The owner approved the lead's recommendation ("yes to the kernels").

**Ported, with attribution.** The GPU kernels' source is ported from where
each kernel originated:
- the gated-delta kernels, from mlx-node (MIT), or from mlx-lm (Apple, MIT)
  where mlx-node's follow it;
- the int8 and W8A8 matmuls, from mlx-node;
- paged attention, from Hugging Face's kernels-community/paged-attention
  (Apache-2.0, vLLM's design), not from mlx-node's copy, which dropped the
  Apache notice.

Each ported file carries a header with its source URL, commit and license,
and has a line in a third-party notices file. The Apache-2.0 files keep their
license, so only the C++ library carries both MIT and Apache-2.0.

**Ours:**
- every launch, as Clojure over mlx_fast_metal_kernel;
- the engine, under gen.llm's protocols;
- the fused blocks, as Clojure plus array/compile;
- the C++ primitives for paged attention, written against our MLX pin with
  no process-wide state;
- any CUDA kernels.

**Tested as every optimization is.** Each kernel is a native op behind a
composite of gen.array ops, which is its reference, checked with planted
mutations.

**Why.** The kernels hold mlx-node's measured tuning, and the same math
written again buys nothing. mlx-node's host code does not fit gen.clj:
- it keeps caches in static globals, against rule 3;
- it builds against mlx-node's own fork of MLX.

### Inference over agents: best-of-K and program proposals in 1.5.0, the rest in 1.6.0

(owner, 2026-11-07; genclj-ojok, genclj-tw4p)

The owner approved the lead's recommendation ("yes to ... M7 split").

**1.5.0, the MSA, takes:**
- the model of a trajectory (genclj-mtwo);
- best-of-K with verifiers (the kmet plan's stage 4);
- the LLM proposing programs scored by marginal likelihood (genclj-0izr).

The MSA's search over synthesized programs needs all three.

**M7 becomes 1.6.0 and keeps:**
- SMC over trajectories;
- repair by MH or regeneration;
- the probability of success;
- inverse planning;
- learning from session scores.

These stay gated on best-of-K's results at matched compute, as the kmet plan
already gates them.

### 1.3.0: an early CUDA checkpoint

(owner, 2026-11-07; genclj-ssiv)

**What.** As soon as Qwen3 runs on MLX, its decode on fenrir through MLX's
CUDA backend is measured against llama.cpp on the same GGUF. Four
measurements:
- prefill and decode speed at batch 1;
- decode at batches up to 64;
- the cost of a training step;
- the kernel count per decode step.

**Why.** MLX on CUDA is young. Used op by op, it ran MCMC 200-500x slower than
GenJAX (docs/research/device-programs.md §1), and an LLM's decode step is
hundreds of small ops as well.

**The decision it feeds**, recorded here before the engine's speed work
continues on CUDA:
- keep going, if MLX on CUDA is close enough;
- otherwise serve NVIDIA from llama.cpp only, train on Metal, or look at XLA
  through PJRT for NVIDIA.

### 1.3.0's engine design: the seven recommendations approved

(owner, 2026-11-07; genclj-j2nb)

The owner answered "yes to all four" to the lead's questions before an
autonomous run. The first question was whether llm-mlx-engine.md's seven
recommendations stand.

1. **Weights** come from GGUF files through MLX's loader. The Q8_0 scales and
   biases are cast to float32 at load: exact, for about 11% more weight memory.
2. **The tokenizer.** Families with a GGUF use llama.cpp's vocabulary, loaded
   vocab_only. Checkpoints without a GGUF get a tokenizer.json reader
   (genclj-44hz).
3. **gen.array gains these operations** (an array-API change; docs/array-api.md
   §6):
   - rms-norm, rope and sdpa, as composites, with MLX's fast ops as native
     ops;
   - quantized-matmul, dequantize and gather-qmm, as primitives.
4. **The engine** is a Slots implementation under gen.llm's protocols.
5. **Caches** concatenate first. Static capacity with an explicit release
   follows after measurement, and paged blocks come in 1.3.0's own C++.
6. **Lockstep particles** use lane groups. Logits stay on the device; the
   protocols are unchanged.
7. **Gradients:** MLX's native mode for training, with the tape as the
   reference. Quantized weights stay frozen, with LoRA beside them.

**Also granted.** The owner's other three answers:
- mlx-lm and mlx-node may be installed on this Mac outside the repo, as
  references;
- memo.clj's session may be asked to repoint its fenrir scripts;
- main may be pushed once the Qwen3 check against llama.cpp passes.

### Inference refuses option keys it does not read

(PROVISIONAL, owner to confirm; genclj-vux3, 2026-11-07)

1.3.0 is the minor release the 1.2.1 entry waited for. An unknown key is now
an ex-info error, `{:reason :unknown-options :fn :unknown :accepted}`. The
check is one internal helper, gen.inference.options/check. The lead's calls,
taken on the implementing agent's recommendations:

1. **A shared options map is refused, with no exception;** a qualified key is
   refused like any other. To migrate: `(select-keys opts (:accepted (ex-data e)))`.
   - Not taken: letting qualified keys through, so callers could carry their
     own data in the map; an explicit opt-out.
2. **Scope:** the bean's six areas, plus mh/move (pf/reweighting passes its
   options on). kernel/kernel and tf/run share the helper; tf/run's error
   data changed form.
3. **Keys read only in some cases stay accepted:**
   - :float is accepted where :device reads it;
   - the -particles kernels accept :compile? and still refuse it when true.
4. **Keys refused although they meant something before:**
   - nuts/adapt refuses :samples and :keep;
   - pf/resize refuses :n, which its argument used to overwrite silently;
   - importance-resampling refuses the undocumented :device;
   - pf/reweighting refuses mh/move's undocumented :fwd-choices.
   Any of them returns by adding the key to that function's set.
5. **Remaining:** these stable functions still ignore unknown keys:
   enumerative's options, learning/train, tf's translators, the diagnostics'
   quantiles, and the gradient functions. A follow-up bean in 1.3.0 covers
   them, so the release carries a single break.

### llama.cpp LoRA adapters: models that engines share

(PROVISIONAL, owner to confirm; genclj-vdu4, 2026-11-07)

gen.llm.llama gains `model` (a value engines share) and `adapter` (a LoRA
GGUF loaded for a model), and engines take `:adapters {adapter scale}`. The
binding's eight adapter functions are written in jextract 25's exact form;
jextract was not installed, and native/jextract-llama.sh lists them for the
next regeneration. The lead's calls, on the implementing agent's
recommendations:

1. **Several adapters over one base:** one model value shared by engines,
   each engine with its own context.
   - Not taken: a model per engine, which loads the weights twice; adapters
     per branch, where llama.cpp rebuilds its graph each time the set
     changes.
2. **An engine's adapter set is fixed for its life.**
   - Not taken: set-adapters! while no branch is live.
3. **Concurrency:** engines over one model decode concurrently under the
   model's read lock; loading or freeing an adapter, and freeing the model,
   take its write lock. llama.cpp's test-thread-safety covers several
   contexts per model.
4. **Closing a model or adapter in use is loud (:in-use),** and closing a
   model frees its adapters.
   - Not taken: reference counting.
5. **aLoRA adapters are refused by engines (:unsupported-adapter);** their
   invocation tokens stay readable.
6. **Adapters are loaded for a model first;** an engine takes no adapter
   paths.
7. **A JVM exiting with a llama.cpp model or engine still open aborts on
   Metal** (ggml-metal's exit assertion). This was true before. It is
   documented, and a bean weighs a shutdown hook.

### gen.array's LLM operations: two optional protocols, uint32 on the JVM

(PROVISIONAL, owner to confirm; genclj-ezdd, genclj-vjwr, genclj-ty8m,
2026-11-07)

**What was built.**
- **Seven new gen.array functions, experimental:** rms-norm, rope and sdpa
  (composites in gen.array.nn), and quantize, dequantize, quantized-matmul
  and gather-qmm, in MLX's affine layout (groups of 32, 64 or 128; 2, 3, 4,
  5, 6 or 8 bits).
- **Two new optional protocols,** so IBackend gains nothing (the 1.x rule):
  - IFused (-fused b op args): a fused kernel, or nil to decline. MLX's fast
    ops implement it; wrapping backends don't, so the composite runs under
    the tape, the batch tracer and staging.
  - IQuantized (-quantized b op args): op-keyed like -linalg. A backend
    without it gives :unsupported, named.

**The lead's calls, on the implementing agent's recommendations:**
1. **quantize is added**, beyond the seven names the owner approved: tests and
   conversion (genclj-t007) need it.
2. **The JVM backend gains uint32 storage**, exact in doubles and wrapping like
   int32. Not taken: packed words as int64, which would make the dtypes
   differ between backends.
3. **mxfp4, nvfp4 and mxfp8 are :unsupported on every backend for now.** Their
   scales are uint8, which gen.array lacks. The owner's call is in
   genclj-9ti1. The agent recommends adding :uint8; the alternative is uint32
   scales cast at MLX's boundary.
4. **A numerics edge: a query that sees no key gets zeros**, on every
   backend.
   - MLX disagrees with itself here: its CPU fallback averages the values;
     its Metal kernels give zeros or another row. The MLX backend zeroes
     that row.
   - A bool mask hides keys with the dtype's lowest finite score, not -Inf,
     so gradients stay finite.
   - Not taken: the uniform average (MLX's fallback, JAX); NaN.
5. **In float32, the JVM's dequantize rounds the product and the sum
   separately**, as MLX's CPU does. Metal fuses them, within one ulp.
6. **Gradients.** Under MLX's native gradients, scales and biases get MLX's
   own gradients. Only the tape refuses them (:no-gradient), with respect to
   x only.
7. **Float64 on MLX is quantized in float32 and multiplied after
   dequantizing** (MLX has no float64 CPU kernel). Out-of-range gather
   indices clamp, as take's do.

**MLX's fused ops are declined where they differ from the composite:**
- float64 (MLX computes in float32);
- rope over two axes, and with :freqs under native gradients;
- causal masks with more queries than keys;
- sdpa under native gradients: MLX 0.32.4's Metal backward fails for 8
  queries or fewer ("the vector kernels do not produce a logsumexp").

**Two findings about MLX 0.32.4.** Metal rounds an exact half away from zero
in quantize; the CPU and the JVM round to even. The JVM's quantize is
otherwise bit-identical to MLX's.

THIRD-PARTY-NOTICES (with CLAUDE.md's rule 10) records ported files' origins.
No kernel is ported yet.

### The CUDA checkpoint decides on compiled steps

(PROVISIONAL, the lead, 2026-11-07, after a side agent's note; refines
"1.3.0: an early CUDA checkpoint"; genclj-ssiv, genclj-tjvd)

**What changes.** The checkpoint takes two runs:
- **Run 1, diagnostic.** Qwen3's forward eager, op by op. It gives the kernel
  count and where the time goes. It does not decide.
- **Run 2, deciding.** Decode with compiled steps over static-capacity
  caches. The owner's decision about MLX on NVIDIA is taken on this run.

**Why.** Run op by op, MLX on CUDA was launch-bound in October, hundreds of
times slower than GenJAX on MCMC. Uncompiled numbers would understate MLX.

**The new order.** genclj-tjvd now starts at the forward's level, and no
longer waits for the Slots engine. The engine adopts its caches after.

### GGUF weights and Qwen3's forward on gen.array

(PROVISIONAL, owner to confirm; genclj-cr5a, genclj-gvfa, 2026-11-07)

**What was built.**
- gen.array/load-gguf (experimental), through a new optional protocol,
  ILoading.
  - A host reader, gen.array.gguf, checks every file's header and types first.
  - MLX loads with mlx_load_gguf. The JVM backend reads the file itself, bit
    for bit equal to MLX's triples.
- gen.llm.models.qwen3 (experimental): load, forward, prefill and step, over
  caches that are values.

**Checked against llama.cpp on Metal.** Qwen3-0.6B Q8_0, 20 prompts, each
read whole, as a chunk and through 8 steps (200 comparisons):
- max |Δ log p| over llama.cpp's top 64: 0.036 (median 0.006-0.007);
- argmax equal 200 of 200; top 5 equal 199 of 200.
- Planted convention errors fail by:
  - KV heads tiled: 17-25 nats;
  - q/k norms skipped: 12-18;
  - RoPE traditional: 2.6-6.6.

**The lead's calls, on the implementing agent's recommendations:**
1. **The namespace is gen.llm.models.qwen3** (mlx-lm's layout), since the code
   is backend-neutral and the JVM backend runs it. The engine can still be
   gen.llm.mlx. Not taken: gen.llm.mlx.qwen3, as llm-mlx-engine.md named it.
2. **A numerics edge: Q8_0's and Q4_0's biases are computed in float32**
   (-128·d, -8·d), not cast from MLX's float16 biases. They are identical
   unless d > 511.75, where MLX's overflow to -inf.
3. **GGUF files with float64 metadata are refused on MLX** (:unsupported).
   MLX 0.32.4's loader misreads them and the process dies with SIGBUS (found
   here; see mlx/io/gguf.cpp). Not taken: falling back to the host reader.
   An upstream issue is the owner's call.
4. **F16 tensors load as float32 on the JVM backend**, which has no float16.
5. **K-quants and BF16 are loud**, where MLX would silently turn them into
   float16. K-quants come with genclj-kqxr.
6. **Strict reading:** every tensor must be read, RoPE scaling is refused, and
   unknown option keys are loud.
7. **The band against llama.cpp on Metal is 0.075, twice the measured max.**
   The top-5 check allows a swap across the boundary within the band. Not
   taken: llama.cpp's own band of 0.1.
8. **The real-model test runs on the MLX backend only**; on the JVM backend
   it counts as a llama skip.

### Unknown option keys: the rest of L5

(PROVISIONAL, owner to confirm; genclj-yrb6, 2026-11-07)

**Covered.** The same check and error form for:
- enumerative's grids;
- transform's translators;
- diagnostics/summary;
- gradients/choice-gradients;
- learning/train and the eight optimizer constructors. These had a silent
  bug: (adam {:learning-rate 0.01}) ran at 0.001.

The CHANGELOG keeps one Breaking line for both beans.

**Calls:**
- The helper stays internal in gen.inference.options, required by
  gen.learning and gen.gradients (all L5).
- The deterministic translator keeps its specific errors for :q-forward and
  :q-backward, checked first.
- Translators refuse each other's keys.
- Functions below L5 are the owner's call (genclj-c6jo).

### Compile donates arguments: gen.array/compile's :donate

(PROVISIONAL, the lead, 2026-11-07; genclj-tjvd)

**The finding.** Qwen3-0.6B on Metal, static cache, batch 1.
- MLX donates a buffer (writes in place) only once the consumed cache's
  handles are released.
- While the JVM still holds them, a compiled step copies the cache: 21.4
  ms/token, no faster than eager concatenation (20.5).
- With the handles released it updates in place: 13.7 ms/token compiled
  against 17.4 eager.
- A compiled step is 805 MLX primitives, against 1,268 eager.

**The call.** gen.array/compile gains an experimental :donate option, after
JAX's donate_argnums: (array/compile f {:donate #{0}}).
- The compiled function consumes those arguments. On MLX their handles are
  released once the call returns its lazy outputs.
- Using a donated array afterwards is a loud ex-info (:reason :donated).
- Where compile is the identity (the JVM backend), donation does nothing,
  and reuse is not caught there.
- qwen3's compiled step takes {:donate? true}. The Slots engine would donate
  a slot's own cache and copy a forked one.

**Not taken.** gen.backend.mlx/release!, called by the engine: core code
would call a backend, against rule 3.

### Qwen3's static caches and compiled steps

(PROVISIONAL, owner to confirm; genclj-tjvd, 2026-11-07)

Measured on Metal (M4), Qwen3-0.6B Q8_0, float32, tokens/s:

| Decode | Eager | Compiled, copying | Compiled, donating | llama.cpp via gen.clj | llama-bench |
|---|---:|---:|---:|---:|---:|
| batch 1 | 54 | 57 | 75 | 99 | 135 |
| batch 16 | 277 | 312 | 467 | 861 | |
| batch 64 | 361 | 393 | 636 | 999 | |

In float16 the donating step decodes 89 tokens/s at batch 1 (127 pipelined)
and matches llama.cpp at batches 4 and 64. A compiled decode step is 805 MLX
primitives, against 1,268 eager.

**The lead's calls, on the implementing agent's recommendations:**
1. **KV writes use scatter-add** into preallocated, zeroed slots, keeping the
   invariant that slots past a row's offset hold zeros. No new slice-update
   op; the array API is unchanged.
2. **New names in gen.llm.models.qwen3:** static-cache, compile-step (with
   :donate?), select-rows, concat-rows, offsets, and prefill's :capacity.
   - Rows may sit at different offsets.
   - offset is now loud for such rows.
   - Overflow is loud (:capacity-exceeded), checked on the host.
3. **compile-step evaluates the weights when it builds the step**, a boundary
   of the inference loop, so a trace captures them as constants.
4. **array/compile validates only :donate.** Its other keys wait for
   genclj-c6jo.
5. **Precision is the owner's call.**
   - float32 scales and KV cost about 25% against float16.
   - llama.cpp's layout is a float16 KV cache with float32 activations; it
     needs a cache-dtype option (genclj-5gfm).
6. **Two follow-ups:**
   - each compiled call costs 1.4-2.5 ms of host time (the engine's
     follow-up, genclj-8rnc);
   - llama-bench belongs in native/build-llama.sh as a tools target.

### KV writes overwrite: gen.array/put-along-axis

(PROVISIONAL, the lead, 2026-11-07, after a side agent's note; supersedes
item 1 of "Qwen3's static caches and compiled steps"; genclj-8rnc)

**The hazard.** The static cache wrote K/V with scatter-add into slots
assumed to hold zeros. That holds only while a cache moves forward. A
cleared slot whose buffers are reused, an offset moved back, or a
speculative rollback would leave old values, and the next write would add
onto them silently (rule 6).

**The call.** gen.array gains an experimental put-along-axis: an overwrite,
the counterpart of take-along-axis, as scatter-add is its transpose. It goes
through a new optional protocol (the 1.x rule), with MLX's
mlx_put_along_axis. The static cache writes with it and drops the zero-slot
invariant.
- A test reuses a cache after a rewind, and fails with the old write.
- Decode speed is re-measured against the scatter-add numbers.

**Not taken.**
- Zeroing slots on every rewind or clear: a second pass, and every rewind
  path must remember to do it.
- Keeping scatter-add and refusing rewinds: speculative decoding and slot
  reuse need them.
*As built (2026-11-07, eacb15b on array-engine), four calls by the
implementing agent, PROVISIONAL with the rest:*
1. **The shape contract is scatter-add's:** indices and values share one
   shape, equal to x's except on the axis. No broadcasting yet.
2. **Staging implements it**, since the compiled-step tests emulate compile on
   the JVM by staging. The device lowerings refuse it ("no lowering").
3. **Repeated indices** are loud on the JVM backend (:repeated-indices) and
   unspecified on MLX, as documented.
4. **qwen3/rewind** (experimental) moves rows' offsets back.

**Checked.**
- A test rewinds caches to 5, to 0, and one row of two, then compares them
  with the concatenating cache, plain, compiled and donating. With the old
  scatter-add write planted, it fails by more than 0.01.
- Decode on Metal is unchanged:

| Write | Batch 1 (tokens/s) | Batch 16 (tokens/s) |
|---|---:|---:|
| put-along-axis | 79-80 | 461-469 |
| scatter-add | 78-80 | 460-465 |

### The CUDA checkpoint's result: keep going (the owner's call)

(the lead's recommendation, 2026-11-07, for the owner to decide;
genclj-ssiv; "1.3.0: an early CUDA checkpoint" and "The CUDA checkpoint
decides on compiled steps")

**The measurement.** On fenrir's RTX PRO 6000: Qwen3 on MLX's CUDA backend,
compiled with donation, against llama.cpp v0.9.0 on the same Q8_0 GGUFs,
float16, tokens/s.

| | MLX | llama.cpp via gen.clj | llama-bench |
|---|---:|---:|---:|
| 8B decode, batch 1, evaluated every 16 steps (graphs on) | 183 | 139 | 168 |
| 0.6B decode, batch 1, evaluated every 16 steps | 408 | 405 | 888 |
| 0.6B decode, 64 rows (graphs on) | 4122 | 2697 | |
| 8B decode, 64 rows (graphs on) | 1741 | 2178 | |
| 8B decode, 16 rows (graphs on) | 696 | 1147 | |
| 0.6B prefill 512 (one chunk, graphs on) | 51.9k | 52.4k | 62.8k |
| 8B prefill 512 (one chunk, graphs on) | 8.9k | 15.5k | 15.5k |

**What the checkpoint found and fixed.**
- **Batched decode:** MLX 0.32.4's CUDA quantized matmul ran a decode step's
  [B 1 K] as B one-row products. Leading axes now go through as one matrix
  (qmm-rows), which made batched decode 3-7x faster.
- **Host-bound at batch 1:** batch-1 decode is host-bound (the GPU is busy
  1.6-2.0 of 4.9 ms per step), so the engine's evaluation boundaries matter
  (genclj-8rnc).
- **The band:** llama.cpp's x86/CUDA Q8_0 path differs from MLX's forward by
  up to 0.32 nats; MLX's own forward is device-independent. The band against
  llama.cpp is per platform.

**The lead's recommendation: keep MLX on NVIDIA.**
- It matches or beats llama.cpp at batch 1, and at 64 rows on the small
  model.
- The 8B's remaining 1.3-1.8x at 16-64 rows and its prefill are speed work
  (fused kernels, prefill chunking), not a reason to switch.
- llama.cpp stays the reference engine and serves where it is faster.

Not taken, still open: serve NVIDIA from llama.cpp only; train on Metal
only; XLA through PJRT.
*Correction (the lead, 2026-11-07, after a side agent's note).* The batch-1
rows above that match or beat llama.cpp (408 against 405 on the 0.6B, 183
against 168 on the 8B) used fixed tokens evaluated every 16 steps: scoring
and teacher forcing.

**In generation**, each token is read on the host before the next step,
since tokens are drawn on the host from the site's key (DESIGN §9). There
MLX reaches about 70% of llama.cpp at batch 1:

| Batch 1, compiled and donating | MLX | llama.cpp |
|---|---:|---:|
| 0.6B | 283-292 | 405 |
| 8B | 94-97 | 139 |

The batched rows (4, 16 and 64) already read every token, so they are
generation numbers: with the qmm-rows fix, MLX beats llama.cpp at 64 rows on
the 0.6B and trails the 8B by 1.3-1.8x.

**What is left is host time per step.** At batch 1 the GPU is busy 1.6-2.0 of
4.9 ms per step, and a compiled call costs 1.4-2.5 ms to build.

**The recommendation stands with this caveat.** The batch-1 generation gap is
the engine's host-cost work (genclj-8rnc), re-measured in generation mode
before the owner decides.

### gen.llm.array-engine: a Branching engine over gen.array

(PROVISIONAL, owner to confirm; genclj-8rnc, 2026-11-07)

**What was built.**
- **Slots.** A slot holds a row of a group: a static cache of one or more
  rows, shared by copy.
- **Runs.** Runs are eager Qwen3 steps in forwards of :n-batch tokens, scored
  on the device.
- **Steps.** A step is one compiled decode with donation.
  - A lane group, held only by its stepping slots, is stepped in place.
  - Any other step stacks its rows first (one copy).
  - Capacities grow by :block (qwen3/grow).
- **Reads.** Logits are read once per step and normalized on the host in
  double, as the llama engine does.
- **Tokenizer.** It is llama.cpp's vocabulary, loaded vocab-only
  (gen.llm.llama/vocabulary).

**Checked.**
- The Branching contract holds within 2e-6 (tiny model, JVM), 2e-5 (tiny,
  MLX) and 1e-4 (Qwen3-0.6B, Metal; measured 5.1e-5).
- Against llama.cpp the worst gap is 1.4e-2 with every argmax equal.
- Throughput on Metal (Qwen3-0.6B Q8_0, float32):

| Metal, Qwen3-0.6B Q8_0, float32 | array engine | llama.cpp |
|---|---:|---:|
| decode, 16 rows (tokens/s) | 470 | 816 |
| token SMC, 16 particles (tokens/s) | 325 | 940 |

Decode equals the raw compiled step: the protocols cost nothing visible.
Token SMC stacks every step, because the LM forks its entries and
resampling varies the batch; paged attention would remove that copy.

**The lead's calls, on the implementing agent's recommendations:**
1. **The engine is gen.llm.array-engine**, backend-neutral like
   gen.llm.models.qwen3. This supersedes the gen.llm.mlx named in "1.3.0:
   where the parts live".
2. **gen.array/->floats** (experimental, through an optional protocol
   IHostFloats) reads a float array to the host in one call. Not taken:
   ->clj, which costs about a decode per step at 16 rows.
3. **Rows are normalized on the host in double, scores on the device.**
4. **A lane group whose in-place step fails loses its slots,** loudly, and
   frees them. Not taken: never consuming a lane group, which copies every
   step.
5. **The LM's :max-cells default (:n-ctx) bounds nothing for this engine**:
   documented, and the tests pass :max-cells. Not taken: the engine reporting
   a default.
6. **After the LM evicts a parent, SMC's rescored values differ by float
   rounding.** The tiny-model test uses 512 branches.
7. **New names:** qwen3/rewind, qwen3/grow, llama/vocabulary,
   array-engine/stats and groups. Defaults: :max-branches 32, :block 128,
   :n-batch 256, :n-ctx 4096.

**Left for later:**
- sessions that pin a branch (D2);
- speculative decoding, which needs a truncate on the Branching protocol
  (D4);
- paged attention, which removes SMC's stacking copy and the memory per
  generation held (D6).

### Training's memory tools: array/checkpoint and selective-log-softmax

(PROVISIONAL, owner to confirm; genclj-pnu6, genclj-jksq, 2026-11-07)

**What was built.** Three experimental pieces:
- **array/checkpoint** goes through a new optional protocol, ICheckpoint.
  - The tape records one node, which reruns f on its own tape.
  - MLX's native gradients use mlx_custom_vjp, recomputing on
    depends(depends(primals, outputs), cotangents), as mlx-node does.
- **array/selective-log-softmax** computes logit[idx] - logsumexp by
  vocabulary chunks, and never makes [B T V].
- **qwen3/log-probs and qwen3/nll** give the SFT loss.

**Measured on fenrir's CUDA**, float32, native gradients:
- Qwen3-0.6B's NLL of 4x256 tokens: 5.6 GB beyond the weights unchecked,
  1.55 GB checkpointed.
- 16 SwiGLU layers: 5.2 GB against 2.8 GB.

**Gradients agree:**
- the tape against central differences: 4.4e-8;
- checkpointed against plain: exact on the JVM;
- MLX against the JVM's float64 tape: 2.6e-7 of the largest gradient.

**The lead's calls, on the implementing agent's recommendations:**
1. **The checkpoint orders the recompute by depends, by default.**
   - At MLX 0.32's breadth-first evaluation it measured the same as MLX's
     own mlx_checkpoint.
   - It was 19% better on a depth-first tape (MLX_BFS_MAX_WIDTH=1).
   - It is never worse. Not taken: plain mlx_checkpoint, which is less code.
2. **A known silent case.** Under MLX's native gradients, a checkpointed
   function that closes over a differentiated value silently loses that
   gradient. The tape is loud (:closes-over). This is as custom-vjp is
   today: mlx-c can't detect MLX's tracers.
3. **selective-log-softmax refuses unknown option keys.** It is below L5,
   ahead of genclj-c6jo's decision for the rest.
4. **The saving shows only under a tight MLX memory limit.** Under a loose
   limit, MLX keeps queued kernels' buffers. A bean decides whether a
   training step sets the limit, or the docs say so.
5. **A tie forcing only the differentiated inputs' gradients is a follow-up
   bean**: 14% better on a layer stack, but 24% slower on Qwen3.

Not yet run on Metal. It was checked on fenrir's CUDA and CPU, native and
tape.

### 1.3.0's scope narrowed

(owner, 2026-11-07 evening; genclj-lv7t, genclj-k3b1; supersedes the scope in
"mlx-node parity, on mlx-c and our own C++")

The owner, on the list of what 1.3.0 would still lack after the autonomous
run: "only paged attention and continuous batching (our C++) is important
for us at this stage - the rest could be kept in beans for later todos".

**In 1.3.0:**
- the foundation (gen.array's LLM operations);
- weights and quantization;
- the Qwen3 and Qwen3.5/3.6 families;
- the engine, with paged attention and continuous batching through our C++
  library;
- training: LoRA, SFT, GRPO.

**Moved out of 1.3.0**, kept as beans under genclj-k3b1 ("mlx-node parity:
the rest", no version yet):
- the families Gemma4, LFM2, Nemotron, Qwen3.8-Flash-Next and DeepSeek
  V4.1;
- the modalities: vision, OCR, speech, embeddings and reranking, the privacy
  filter;
- the server and the CLI.

**The autonomous run's order changes.** Paged attention and continuous
batching (genclj-9a1l, genclj-1daf, genclj-b3ow) move ahead of the
remaining weight formats.

### LoRA: gen.llm.lora, and llama.cpp's adapter GGUF

(PROVISIONAL, owner to confirm; genclj-yc3b, genclj-p4mk, 2026-11-07)

**What was built.** gen.llm.lora (experimental, backend-neutral) has init,
adapt, merge, save-gguf and load-gguf. Adapted weights are {:base :lora},
and a family's forward dispatches on them. A pure-Clojure GGUF writer in
gen.array.gguf writes atomically, byte-identical to gguf-py's.

**Checked.**
- **The round trip.** An adapter applied in MLX, and the same adapter loaded
  by llama.cpp (fenrir, CUDA), agree within the platform band: largest Δ log
  p 0.21-0.45 against shifts of 1-19 nats, argmax and top 5 equal. A trained
  adapter does too.
- **Planted convention errors** are refused by llama.cpp or exceed 1.3 nats:
  A and B transposed and swapped, names swapped, alpha written as
  alpha/rank, layers off by one, scale alpha instead of alpha/rank.
- **Training.** Qwen3-0.6B, rank 8 on the last 8 layers, AdamW: the NLL fell
  from 188 to 21 in five steps.

**The lead's calls, on the implementing agent's recommendations:**
1. **LoRA is its own namespace,** with {:base :lora} weights the families
   dispatch on. Not taken: LoRA per family; an :adapters key on the model.
2. **Params are keyed by layer, then target.**
3. **The scale is alpha/rank**, as llama.cpp and PEFT have it. mlx-lm's
   "scale 20" is alpha = 20·rank.
4. **Defaults:** rank 8, alpha 16, targets q and v, all layers.
5. **A is uniform, from fold-in keys by site; B starts at zero.**
6. **Dropout is on only when adapt gets a :key.**
7. **The GGUF writer is pure Clojure.** Not taken: mlx_save_gguf (MLX only);
   a public array/save-gguf.
8. **load-gguf is strict:** it refuses norm tensors, aLoRA, embedding or
   output tensors, quantized tensors and mixed ranks. A missing alpha means
   alpha = rank.
9. **merge dequantizes and doesn't re-quantize.**
10. **Adapters take the model's dtype.**
11. **lora-llama's mutation threshold is 1.3 nats** (twice fenrir's band).
12. **A known cost:** under native gradients the checkpoint also computes
    the frozen weights' cotangents (genclj-8xua).

The lora-llama test has not yet run on Metal (0.075 band).

### Qwen3.5/3.6 and gen.array/gated-delta

(PROVISIONAL, owner to confirm; genclj-xnx3, 2026-11-07)

**What was built.**
- **gen.llm.models.qwen35** (experimental): dense and MoE, following
  llama.cpp's qwen35, qwen35moe and delta-net-base graphs.
  - GatedDeltaNet layers hold {:conv :state}, as values.
  - MoE uses a softmax router, renormalized top-k, gather-qmm, and a
    sigmoid-gated shared expert.
  - Static caches and compiled steps work as qwen3's do.
- **gen.array/gated-delta** (experimental) has :method :loop, :chunked or
  :auto. The chunked form is llama.cpp's, with the triangular inverse done
  by doubling blocks.
- **mlx-lm's gated-delta Metal kernel**, behind IFused's :gated-delta. Its
  origin was checked: mlx-lm PR #454, which mlx-node's step kernels copy. It
  was ported from mlx-lm at 760c5ab (v0.31.1, MIT, Apple), with the header
  and a THIRD-PARTY-NOTICES line.
- **The array engine** picks the family from the GGUF's architecture.

**Checked on Metal, Qwen3.5-0.8B Q8_0, against llama.cpp.**
- 200 comparisons: max |Δ log p| 1.6e-2 (median 1.3e-3), argmax and top 5
  equal in all.
- Planted errors fail by these margins:
  - conv state stale: 1.4-8.9 nats;
  - conv kernel reversed: 8.5-18.5;
  - gate swapped: 14-29;
  - KV heads tiled: 5.6-14.4;
  - norms skipped: 11-21.5;
  - RoPE traditional: 0.14-4.1, at least 1.25 on the 44-token prompt the
    test uses.
- Speed against llama.cpp, tokens/s:
  - batch 1: 74.5 against 79.3;
  - batch 16: 437 against 418;
  - prefill: 1968 against 2293.

**The lead's calls, on the implementing agent's recommendations:**
1. **gated-delta is an experimental gen.array op.** g is the log decay; every
   head has its own keys (the model tiles the key heads); the options are
   :method and :chunk-size; :auto runs a backend kernel, otherwise chunked
   from 64 tokens.
2. **gen.llm.models.common** (internal) holds what the families share. qwen3
   moves onto it later (genclj-dnt5).
3. **rewind on a recurrent cache** allows only the row's own offset or 0.
   Anything else is loud (:unsupported): a recurrent state can't move back.
4. **The recurrent state stays in float32** for half-float models.
5. **The unequal-heads check is a stand-in**, opt-in on tiny random models
   with the 0.8B's tokenizer. Grouping the V heads wrongly fails it by
   7-11 nats. A trained model with unequal heads and a real Qwen3.6 MoE wait
   for k-quants (genclj-utti, blocked by genclj-kqxr).
6. **The band off the Mac is 0.65, borrowed from Qwen3** and not measured
   for Qwen3.5 (genclj-2krw: a CUDA kernel and the band).
   Superseded: measured as 7.1 (Q8_0) and 12.3 (the MoE's K-quants), in
   "The real Qwen3.6 MoE, the off-Mac bands from one place".

**Deferred:**
- mlx-node's chunked kernel, only if measured (genclj-afy0);
- the CUDA kernel (genclj-2krw);
- tape replay for speculative rollback (genclj-ond4).

### Training covers Qwen3.5/3.6 too

(owner, 2026-11-07 evening; genclj-8h0l)

The owner, on a side agent's note that LoRA plugged only into Qwen3: "I want
qwen3.5 and qwen3.6". 1.3.0's training (LoRA, SFT, GRPO) covers the Qwen3.5/3.6
family as well: the night loop's 0.8B and the bake-off's 35B-A3B.
- LoRA plugs into qwen35's projections.
- Adapters export under llama.cpp's qwen35 tensor names.
- SFT and GRPO are family-agnostic, taking the family through the model as
  the array engine does.

### Paged attention through metal_kernel: no C++ of ours (route A)

(PROVISIONAL, the lead, 2026-11-07 night, owner to confirm; genclj-9a1l,
genclj-1daf; refines "1.3.0's scope narrowed")

The owner's words were "paged attention and continuous batching (our C++)".
That assumed mlx-c could not reach paged attention. The spike (branch paged,
53829d8) found it can.

**The reads.** kernels-community's paged_attention v1 runs as an MLX
metal_kernel, the kernel code unchanged.
- It matches sdpa over the gathered caches: 3.1e-6 in float32.
- It is as fast as sdpa over dense caches: 1.10 against 1.07 ms in float32
  and 0.58 against 0.87 ms in float16, for one layer, 16 rows, contexts to
  1023.

**The writes.** metal_kernel's outputs are always fresh buffers, with no
input/output aliasing, but the write needs no kernel.
- MLX's own scatter (put-along-axis) under compile's :donate writes the
  pool in place: 0.3-0.7 ms whatever the pool's size, the same pointer.
- It copies when the pool is held, so value semantics hold.

**Route B was proven too.** A C++ library (libgenclj) built against the
wheel's headers, a Primitive with its own eval_gpu, bound by FFM. It costs:
- a second native artifact;
- an ABI pinned exactly to MLX's internals;
- MLX's internal CUDA encoder, for CUDA.

**The call: route A.**
- gen.array gains experimental ops (paged write, copy-blocks, paged
  attention), each a composite of take, put-along-axis and sdpa as its
  reference. MLX implements them by donated scatter and metal_kernel; CUDA
  later by mlx_fast_cuda_kernel with vLLM's kernels.
- The C++ spike is kept on a branch as the fallback, if continuous batching
  needs what metal_kernel cannot do. genclj-9a1l is parked.

**The kernels' provenance.** They are ported from kernels-community at
28a49f7 under Apache-2.0. The header and THIRD-PARTY-NOTICES record the
chain vLLM (Apache-2.0) → mistral.rs's Metal port → kernels-community.

### SFT: gen.llm.sft, and the family dispatch gen.llm.models

(PROVISIONAL, owner to confirm; genclj-stpo, genclj-az5s, 2026-11-07 night)

**What was built.**
- **gen.llm.sft** (experimental):
  - examples with assistant-only masks, found from chat/segments' byte spans;
  - padded batches;
  - mean NLL per trained token;
  - a trainer value, and a step that accumulates batches and divides once.
    A non-finite loss or gradient is loud (:non-finite), never a skipped
    step;
  - train, with keys, held-out evaluation and checkpoints. Resume is
    bit-exact.
- **gen.llm.models:** architecture, plus log-probs and nll as multimethods
  each family registers.
- **gen.learning:** warmup-cosine (a schedule over any optimizer) and
  clip-value, experimental in a stable namespace.

**The real run** (fenrir CUDA). Qwen3-0.6B Q8_0, LoRA rank 8 on the last 8
layers, 48 JSON-format chat examples, 16 held out, 24 steps, AdamW 2e-4.
- Held-out NLL fell from 3.61 to 0.093 per trained token.
- Greedy replies in the format: 0 of 16 before, 15 of 16 after.
- llama.cpp serving the checkpoint's adapter: within 0.53 nats (band 0.65),
  argmax equal, 14 of 16 in the format.

**The lead's calls, on the implementing agent's recommendations:**
1. **The end token <|im_end|> is trained by default** (:end-token?). Without
   it, 4 of 16 replies never stopped. Not taken: mlx-node's choice of never
   training it.
2. **Checkpoints are trainer.edn plus adapter.gguf.** EDN is exact in every
   dtype and holds any optimizer's state: 95 MB for 1.44M parameters with
   moments. Not taken: safetensors, compact, worth it when adapters grow.
3. **The loss is the mean over every trained token in a step**, so
   accumulated batches equal one big batch.
4. **The schedule is Hugging Face's:** a warmup's first step has factor 0.
5. **Truncation keeps the start**, as Hugging Face and mlx-lm do.
6. **The think block is context**, not trained.
7. **MLX's memory limit is documented, not set by sft.** On CUDA, a limit of
   the held memory plus 1.25 times a step's peak keeps the checkpoint's
   saving (1.8 GB against 6.1 GB beyond the held 0.76 GB) at no time cost.
   Setting it automatically would change a global from library code (an
   array-API addition).

Not yet run on Metal (genclj-h7ru).

### The engine's host time: donated arrays leave MLX's fresh-bytes count

(PROVISIONAL, owner to confirm; genclj-c7uf, 2026-11-07 night)

**The finding.** Most of a decode step's host time was System.gc, not the
compiled call.
- MlxRuntime collects once the arrays made since the last collection hold
  gen.mlx.gc-new-bytes.
- A donating step's outputs counted as new, though they take over the
  donated cache's buffers.
- So a full GC ran about every 25 steps at batch 1, and every 2 at 16 rows.

**Changes:**
- A donated array's bytes come off the count.
- The engine normalizes rows in parts on the host thread pool, summing in
  order, so the result is bit-identical.
- compile computes the donated leaves once per argument structure.

**Measured on Metal, Qwen3-0.6B.**
- Through the engine at batch 1: 74.7 → 92.5 tokens/s. llama.cpp through
  gen.clj gives 93.5-96.4.
- The step profile at 16 rows: 377 → 555-559 tokens/s.
- qwen3_bench at 16 rows: 468 → 662 (llama.cpp 806).
- Qwen3.5-0.8B at 16 rows: 418 → 541 (llama.cpp 405).
- Outputs are bit-identical to main at batches 1, 3 and 16 on both
  families.

**What is left at batch 1** is MLX's own encoding of about 776 kernels, about
3 ms of host time per step. The lever there is fewer kernels (fusion), not
Java.

**The lead's calls, on the implementing agent's recommendations:**
1. The memory-accounting change in MlxRuntime/donate. It is
   backend-internal and leaves values unchanged.
2. gen.llm.host-threads stays at its default until fenrir's 8-thread
   measurement.
3. On CUDA a step's two hand-offs to the evaluation thread stay for now.
   Reading an evaluated array without the hand-off is untested there.

### A batched particle outside its support moves as its scalar trace does

(PROVISIONAL, owner to confirm; genclj-n1s3, 2026-11-07 night)

**The bug.** It was found by a random-seed fuzz failure, and has existed since
9277423 (2026-10-25, "A choice outside its support stops the run").
- When a batched particle's choice leaves its support, the batched run can't
  stop for that particle, so its batched trace holds the choices made after.
- Its scalar trace stops there.
- b/update, regenerate, move and project then kept, weighed, discarded or
  counted those later choices. The scalar operation draws them afresh.

It broke rule 1 silently.

**The fix.** It lives in gen.batched, running the scalar handler and copying
none of its logic.
- A particle whose score is -Inf or NaN, read only where scores are concrete,
  runs its scalar operation alone, with the keys the fallback gives it.
- The trace is joined back as a batch of one, so the collection stays
  batched.
- On MLX the scalar runs don't stop either, so nothing changes there.

The CHANGELOG notes the draws change.

**Not fixed** (genclj-bdzy):
- b/choice and map-particles (gradients) still read the choices after the
  stop.
- An update that takes a particle out of its support still discards
  differently from the scalar one (gen.fuzz-test exempts it).
*The generation-mode re-measure (2026-11-07 22:46, main 7401be1;
genclj-ssiv).*

**Greedy decode** (the argmax read on the host). With compiled steps,
donation, the qmm-rows fix and the host-cost fix, MLX on fenrir's CUDA now
beats llama.cpp from 16 rows up on the small models:

| Greedy decode, tokens/s | MLX | llama.cpp |
|---|---:|---:|
| 0.6B, 64 rows | 7267 | 2839 |
| Qwen3.5-0.8B, 64 rows | 6418 | 1828 |
| 8B, 64 rows | 2369 | 2170 |
| 8B, 16 rows | 814 | 1118 |

**At batch 1 MLX is 0.5-0.75x of llama.cpp.**

**Through the Branching protocol** MLX is still 1.4-5.8x behind llama.cpp
through gen.clj. Every row is read and normalized on the host, and that path
(12-37 ms per step at 16 rows) now exceeds the device's step. It is gen.clj's
host path, not MLX: genclj-9y6y, which also covers two CUDA anomalies in the
0.6B.

**The lead's recommendation stands:** keep MLX on NVIDIA. Its batched decode
is now faster than llama.cpp's on the models gen.clj's research uses most.

### LoRA for Qwen3.5/3.6

(PROVISIONAL, owner to confirm; genclj-8h0l, 2026-11-07 night)

**What was built.** gen.llm.lora covers qwen35 and qwen35moe on every
projection:
- attention q (with its output gate), k, v, o;
- GatedDeltaNet's qkv, z, beta, alpha and out;
- the feed-forward;
- the MoE's shared expert;
- the routed experts, as opt-in targets in llama.cpp's build_lora_mm_id
  layout.

qwen35 registers its log-probs and nll in gen.llm.models, so SFT trains it
unchanged.

**Checked on Metal against llama.cpp.**
- Qwen3.5-0.8B with an adapter on every target: at most 0.0039 nats (band
  0.035), with planted convention errors failing by 1.7-17.6 nats or refused.
- Five AdamW steps took the NLL from 162 to 1.0, and llama.cpp serves the
  result within 0.0053.

**The lead's calls, on the implementing agent's recommendations:**
1. **One lora/targets map by architecture**, with the model's weight keys as
   target names.
2. **qwen35's default targets are [:q :v :qkv].** [:q :v] would reach only
   the attention layers, 6 of 24.
3. **A target adapts the chosen layers that have it,** and is loud only when
   none does.
4. **Routed experts are opt-in.** A real MoE and the gathered term's memory
   are genclj-wfv1.
5. **adapt refuses an adapter of another architecture,** as llama.cpp does.
6. **gen.llm.models.common requires gen.llm.lora.**
7. **Site keys keep Qwen3's indices,** so its draws are unchanged.
8. **The MoE's routing indices carry no gradient** (stop-gradient). MLX
   refuses a gather's VJP with respect to its indices.
9. **A custom VJP for gated-delta's chunked form is a draft** (genclj-oze3).
   The rule takes 17-26% of a 0.8B LoRA step, so it is worth doing later,
   not now.

Not yet run on CUDA. Qwen3.5's off-Mac band is unmeasured.

### Paged attention, as built

(PROVISIONAL, owner to confirm; genclj-1daf, 2026-11-07 night; builds on
"Paged attention through metal_kernel")

**What was built.**
- **Three gen.array ops** (experimental, in vLLM's layouts): paged-write,
  copy-blocks and paged-attention. Each is a composite of put-along-axis,
  take and sdpa, which is its reference.
- **On MLX:**
  - IFused's :paged-attention runs kernels-community's v1 kernel through
    metal_kernel;
  - :paged-write is two donated scatters on the pools.
- **gen.llm.paged:** a block allocator and block tables, as host values.
  Forks share blocks, and a shared, partly filled block is copied before a
  write.
- **The model and engine:** qwen3's paged step, and the engine's opt-in
  :paged? mode.
- **Licensing:** LICENSE-APACHE-2.0 at the root, for the ported kernel.

**Checked.**
- The kernel against the composite: 2.9e-6 in float32, within one ulp in
  half floats.
- Qwen3's paged decode against its concatenating cache: 0 on the JVM, 3.8e-6
  on Metal.
- The Branching, LM, token and SMC contracts pass in the paged mode.

**Measured on Metal** (Qwen3-0.6B Q8_0, 16 particles):
- token SMC in float32: 390-409 tokens/s, against the static engine's 318
  and llama.cpp's 771;
- in float16: 540 against the static 619. Copy-on-write's two extra syncs
  per step cost more there than the static engine's copy.

**Two MLX findings.**
1. **A donated write costs about 8 ms per GB of buffer written**, however
   few slots it touches. Reads stay flat. So pools are sized to need: 128
   blocks, doubling. Long contexts are bound by this, and the static cache
   probably pays the same. genclj-1ovd follows up: a Metal capture, and 64 MB
   segments.
2. **A take, then a put, on one pool within one compiled call copies the
   pool.** So copy-on-write is a gathered, evaluated read, then a donated
   put.

**The lead's calls, on the implementing agent's recommendations:**
1. **The port** is from kernels-community, Apache-2.0: v1's body and
   helpers only, with the chain vLLM → mistral.rs → kernels-community
   recorded.
2. **Writes and copies are MLX scatters.** The cache kernels aren't ported:
   metal_kernel can't write in place.
3. **The three ops are public, experimental gen.array ops.** Not taken:
   internal to gen.llm.
4. **v1 only.** Past about 8k positions the composite runs; v2 is
   genclj-6qb6.
5. **Engine defaults:** :paged? off, :block-size 16, :pool-blocks 128
   (doubling), :min-rows 1. Qwen3 only for now; Qwen3.5 is refused loudly
   (genclj-6qb6). Not taken: paged by default; that waits until float16 SMC
   beats static.
6. **A failed paged step loses every branch, loudly (:lost-slots)**, since
   the pools were donated. Not taken: non-donating steps, which copy the
   pools.
7. **Copy-on-write is two calls.** Folding the put into the step is
   genclj-6qb6.
8. **Multi-token reads stay eager;** only steps are paged. Paged prefill
   belongs to continuous batching (genclj-b3ow).
9. **On MLX the kernel clamps block ids and lengths;** eager backends check
   them loudly.

The C++ spike is parked unmerged on the branch paged-cpp-spike. Its build
left libgenclj in the gitignored native/build; nothing loads it.
*Correction (the lead, 2026-11-07 night, after a side agent's note).* The
line above, "faster than llama.cpp's on the models gen.clj's research uses
most", holds only for decoding where just the chosen token is read back.
- gen.clj draws its tokens from the full distribution, read back and
  normalized on the host: token SMC, grammar constraints, the token
  generative function.
- On that path, through the array engine, MLX is still 1.4-5.8x behind
  llama.cpp on fenrir.
- The gap is gen.clj's host path, not the GPU (genclj-9y6y, the hostpath
  agent, running).

The owner's decision on MLX for NVIDIA should wait for genclj-9y6y's numbers
on that path.

### Continuous batching: its design (opt-in, no protocol change)

(PROVISIONAL, owner to confirm; genclj-b3ow, 2026-11-07 night; the design
before its build)

**No protocol changes.** Branching, LM and Slots stay as they are. Batching
is a new optional protocol, implemented by the engine's paged mode under
:continuous? true.

**How a forward is built.**
- Between forwards, the engine's loop admits every queued request and runs
  one forward over all of them: every one-token read, and a prefill chunk for
  the oldest multi-token read, within :n-batch.
- It waits for other callers at most a quarter of the last forward's time.
  A lone caller is never delayed.
- Engines without Batching run exactly as today.

**The owner's call (item 7 of the design).** Under :continuous?, concurrent
callers' results depend on how their calls were gathered, within the
engine's band. Draws at near-ties may then differ between runs. This
revisits, at the engine layer, the reasoning behind the LM's "no
autobatcher" decision; that is why it is opt-in.
- A single-threaded program never batches across calls, so it stays
  bit-reproducible.
- Concurrent clients each hold their own LM over the shared engine. An LM
  that served concurrent callers would change its contract (the owner's call;
  not done).

### Continuous batching, as built

(PROVISIONAL, owner to confirm; genclj-b3ow, 2026-11-08; builds on
"Continuous batching: its design")

**What was built.**
- **A task loop in gen.llm.slots** replaces the engine thread's executor.
  The optional Batching protocol (slots-forward) is used only when an engine
  declares :batching.
  - advance, extend and open become requests, held in flight.
  - Between forwards, the loop runs one forward over every one-token read,
    plus a prefill chunk for the oldest longer read.
- **qwen3/paged-batch-step:**
  - decode rows go through the paged kernel;
  - the chunk goes through its gathered blocks and sdpa;
  - logits are computed only for the rows read.
- Engines without Batching behave as before, shown by a recording test. The
  llama.cpp, static and toy suites pass unchanged.

**Measured on Metal** (Qwen3-0.6B Q8_0), tokens/s with the median read
latency:

| | static | paged | continuous | llama.cpp |
|---|---:|---:|---:|---:|
| 8 clients, float16 | 86 (84 ms) | 90 (82 ms) | **302 (18 ms)** | 103 (72 ms) |
| 8 clients, float32 | 78 | 82 | **175** | 92 |
| token SMC, 16 particles, float16 | 638 | 641 | 670 | 900 |
| 2 SMC runs at once, float32 | | 596 | 799 | 889 |

- A lone client is unchanged: 81 against 77 tokens/s.
- llama.cpp batches only within one call, so its concurrent clients take
  turns.

**The lead's calls, on the implementing agent's recommendations:**
1. **Opt-in (:continuous?),** since concurrent callers' results depend on how
   calls were gathered, within the band. Not taken: a deterministic lockstep
   batcher with registered workers, which needs a registration API.
2. **The scheduler lives in gen.llm.slots** as an optional protocol. Not
   taken: two thread code paths.
3. **Prefill chunks use the gathered blocks and sdpa.** A dedicated paged
   prefill kernel is a later option.
4. **The wait for returning callers is a quarter of the last forward's
   time.**
5. **Each concurrent client holds its own LM** over the shared engine.

**Left:**
- more than one prefill chunk per forward;
- copy-on-write inside the step (genclj-6qb6);
- batching for llama.cpp's Slots (its llama_decode takes mixed batches);
- the CUDA kernel;
- two edge cases in the paged mode: score dropping :lost-slots, and a failed
  normalization leaving consumed pools in the engine's state.

### GRPO: gen.llm.grpo over LoRA, family-agnostic

(PROVISIONAL, owner to confirm; genclj-waym, 2026-11-08)

**What was built.** gen.llm.grpo (experimental):
- rollouts through the array engine, with each token's log-probability
  recorded where it is drawn;
- rewards, and group advantages;
- the clipped surrogate, with a k3 KL term against the base model (the
  adapter at scale 0, no copy);
- four aggregations;
- a round and a train loop on SFT's trainer, through a new :loss hook in
  sft/step.

**The losses.**
- **grpo:** DeepSeekMath Eq. 3; defaults ε 0.2, β 0.04.
- **dapo:** a token sum over all tokens, ε_high 0.28 (DAPO Eqs. 8 and 12).
- **dr-grpo:** the sum over N·L with Â = r − mean (Dr. GRPO's code).
- **bnpo:** TRL's per-microbatch token mean.

**The real run on fenrir's CUDA.** Qwen3-0.6B Q8_0, LoRA rank 8 on the last 8
blocks, two-digit multiplication with an exact reward, 40 rounds of 8 × 8
in 5.5 minutes.
- Step-0 ratios: max |ρ − 1| is 1e-4.
- Held-out accuracy (128 prompts × 8, same keys): 0.615 → 0.771. The paired
  difference is 0.155 ± 0.022 (z 7.0).
- llama.cpp serving the adapter: argmax equal, greedy 0.781 against the
  engine's 0.789.
- The max |Δ log p| was 0.90 with the adapter, and 0.73 for the base model
  alone, both over CUDA's 0.65 band on these digit-heavy prompts. The test
  holds the adapter to 1.3 nats; the band is re-measured in genclj-fc6u.
  Superseded: the band off the Mac is 2.4 and the adapter is held to it
  (1.07 nats), in "The real Qwen3.6 MoE, the off-Mac bands from one place".

**The lead's calls, on the implementing agent's recommendations:**
1. **Old log-probabilities** are recorded where each token is drawn (the
   policy at the sampling temperature, before top-k/top-p), so step-0 ratios
   are 1. Not taken: recomputing them with a training forward, which makes
   the step-0 check vacuous.
2. **:bnpo is TRL's aggregation.** The BNPO paper (arXiv 2506.02864, Eq. 11)
   instead normalizes advantages by a Beta density. Not taken: the paper's
   advantage as an option, or dropping :bnpo. The owner's call.
3. **The advantages' std** is the population std, with eps 1e-4. TRL applies
   Bessel's correction.
4. **Dr. GRPO divides by N·L**, which differs from its paper's 1/G by a
   constant factor in the learning rate.
5. **Defaults:**
   - β is 0.04 for grpo only;
   - truncated completions are kept;
   - the end token is trained;
   - flat groups stay in the loss (no dynamic sampling) and are counted.
6. **MLX's memory limit is documented** (a round peaks at 6.4-7.6 GB), not
   set.
7. **Each round makes a new array engine** (rollouts take 5.5 of each round's
   8 s). Reusing engines is a follow-up.
8. **qwen35's log-probs need :temperature** before GRPO runs on Qwen3.5 at
   T ≠ 1. Until then it is loud (genclj-qfu2).

### k-quants: exact, in MLX's affine layout or float32

(PROVISIONAL, owner to confirm; genclj-kqxr, 2026-11-08)

**How each type loads.** Every value is exactly llama.cpp's
dequantize_row_*, bit for bit, on random blocks and on the real Qwen3-0.6B
Q4_K_M's tensors.
- **Q4_K and Q5_K:** MLX's affine 4 and 5 bits at group 32, one group per
  sub-block. scale = d·sc and bias = -(dmin·m), both in float32.
- **Q5_0 and Q5_1:** the same; Q4_K_M files fall back to them.
- **Q6_K:** float32, exact. MLX accepts groups of 32, 64 or 128 only, and
  Q6_K's sub-blocks of 16 have their own scales.

**Checked against llama.cpp on Metal.** Qwen3-0.6B quantized to Q4_K_M by
the pinned llama.cpp:
- max |Δ log p| 0.231 (median 0.04-0.066), argmax 200/200;
- llama.cpp's own spread on that file is 0.226 (against 0.052 on Q8_0), so
  the gap is its K-quant float noise;
- qwen3-llama-test's band on K-quant files is 0.47;
- Qwen3.5-0.8B Q4_K_M: 0.414, with llama.cpp's own spread at 0.460.

**The repack** runs at 5.4 GB/s (Q4_K) on the M4. A whole model loads at
1.2-1.4 GB/s of file.

**The lead's calls, on the implementing agent's recommendations:**
1. **Q6_K is stored as dense float32:** exact, 4.9 times the file's bytes. Not
   taken, either worth later:
   - two 6-bit affine parts at group 32, summed: exact, 2 matmuls;
   - a Metal kernel for group 16.

   Not taken: requantizing to 8-bit or float16, which isn't exact.
2. **Scales and biases are float32**, so Q4_K takes 6 bits a weight against 4.5
   in the file. float16 isn't exact (d·sc has up to 17 significant bits).
   - The 20 GB Qwen3.6-35B-A3B Q4_K_M needs about 28 GB: the 64 GB MacBook,
     not this 32 GB Mac.
   - Memory against exactness is the owner's call.
3. **Q5_0 and Q5_1 are added**, beyond the bean's three types.
4. **On MLX, a file with any of these types loads through the host reader.**
   MLX's loader would silently turn them into float16. Other files load as
   before.
5. **The repack uses the fork-join common pool** inside gen.array.gguf; L0 has
   no pool of its own.
6. **qwen35-llama-test has no K-quant band yet.** It would be about 0.83, which
   is above its RoPE mutation's 0.19.

*Also found (paged2, 2026-11-08):* the "8 ms per GB" cost of donated writes,
noted under "Paged attention, as built", was MlxRuntime's fresh-bytes GC
from before genclj-c7uf's fix.
- On main, a donated paged write takes 0.3-1.9 ms, flat in the pool's size,
  with no GCs.
- Long contexts are not bound by it, and the pools need no segments
  (genclj-1ovd).

### Small fixes; the engine's host path, phase 2

(PROVISIONAL, owner to confirm; genclj-qfu2, genclj-h7ru, genclj-5t5l,
genclj-9y6y, 2026-11-08)

**Small fixes.**
- qwen35/log-probs takes :temperature, through gen.llm.models.common's
  check-temperature and tempered-head.
- A GRPO round on Qwen3.5 starts with |ρ - 1| at 1.3e-5 at most, on Metal.
- SFT passes on Metal: tape and native suites; sft-llama-test within
  0.017 nats of llama.cpp; held-out NLL equal to fenrir's to four digits.

**The paged mode's edge cases.**
- score keeps :lost-slots and frees the lost branches.
- A failed row normalization leaves the slots over the pools the step wrote.
- Reads normalize before writing their books, in the static mode too.

**The lead's calls, on the implementing agents' recommendations:**
1. **After a failed normalization a paged step keeps its branches.** The
   continuous mode still loses those it read; aligning it is genclj-8ni2.
2. **The read-ordering fix went beyond its bean's wording:** the same kind
   of bug, in the static mode too.

**The engine's host path** (genclj-9y6y; numbers in its bean):
- **Anomaly 1, the 0.6B's slow read at 16 rows** (26.8 ms on CUDA). MLX
  moves a buffer the host reads to managed memory, then recycles it into
  device allocations by size alone, so the next device write and host read
  page-fault. 16-bit logits are now read as their bits and widened on the
  host, bit-identical: 4.9 ms. The root, in MLX's allocator, is genclj-p9sb (the
  owner's call).
- **Anomaly 2, the batch-1 floor:** bisect results on fenrir, recorded when
  read.
- **Graal against C2 on x86:** measured, decided after the logs. C2 runs
  llama.cpp's path 20% faster at the 0.6B's batch 1.
- **Defaults kept:** a pool of 4 host threads (8 gives about 20% more at 16
  rows), and llama.cpp's own normalization.
- **A scores-only advance** for consumers needing less than full rows is a
  protocol change: genclj-t46p (the owner's call).

### The sampling path on fenrir after genclj-9y6y (for the owner's NVIDIA decision)

(the lead, 2026-11-08 ~01:30; genclj-9y6y, genclj-ssiv)

**The re-measure.** array-engine-bench, float16, rows read and normalized on
the host as gen.clj samples, tokens/s, MLX against llama.cpp through
gen.clj, Graal / C2:

| Model, batch | MLX before | MLX now | llama.cpp |
|---|---:|---:|---:|
| 0.6B, b1 | 132 / 148 | 216 / 213 | 445 / 505 |
| 0.6B, b16 | 337 / 425 | 841 / 864 | 1879 / 2484 |
| 0.8B, b1 | 143 / 169 | 174 / 182 | 353 / 414 |
| 0.8B, b16 | 484 / 744 | 602 / 705 | 1570 / 1900 |
| 8B, b1 | 84 / 76 | 86 / 88 | 140 / 145 |
| 8B, b16 | 251 / 281 | 475 / 470 | 1135 / 1282 |

- **The gap** on gen.clj's own path fell from 1.4-5.8x to 1.6-2.9x.
- **The step is still host-bound.** At the 0.6B's 16 rows, Graal: the read
  takes 4.9 ms, normalization 3.9, the whole step 19, against a device floor
  of about 5.
- **Anomaly 2 was not a code change:** at 834313a the floor measures 288-298
  today. The host-cost merge raised it to 345-357.
- **The other measured paths:**
  - On Metal with continuous batching (8 clients), MLX is 3x llama.cpp.
  - Token SMC on Qwen3.5-0.8B is 2.5x llama.cpp's (698 against 276).
  - Greedy decode with the chosen token read beats llama.cpp from 16 rows up.

**C2 on x86** (PROVISIONAL, the lead, on the implementing agent's
recommendation). fenrir's LLM benches and serving run under C2 (the :c2
alias). Graal, our x86 hosts' default JIT, handicaps host loops: C2 speeds
llama.cpp's path through gen.clj by 10-35%, and is never worse for MLX's.

**The lead's recommendation to the owner.** Keep MLX on NVIDIA.
- Its compiled decode and continuous batching are competitive or ahead.
- gen.clj's per-token sampling path still trails llama.cpp 1.6-2.9x on
  fenrir.
- The levers left are host-side:
  - widening f16 four at a time;
  - MLX's allocator (genclj-p9sb);
  - a scores-only advance (genclj-t46p, a protocol change);
  - device-side normalization, the owner's call under the engine's
    decision 2.

**Addendum (the lead, 2026-11-08 morning, after a side agent's note): the
NVIDIA leads hold in float16 only.** Every lead over llama.cpp on fenrir
measured overnight is float16. The engine computes in float32 by default
there.
- Qwen3-0.6B Q8_0, continuous, 8 clients: 555 against 503 in float16; 409
  against 502 in float32 ("Paged attention on CUDA").
- The MoE loses more in float32: qmm_sm80 takes half floats only, so
  float32 falls back to qmm_naive, 21.7 against 17.3 ms a step at batch 1
  ("The MoE on CUDA").
- On the Mac the lead holds in either: 8 clients, Qwen3 287 against 99 and
  Qwen3.5 217 against 70 in float32 ("Paged multi-token reads").
- So this decision goes with the default precision for quantized models on
  CUDA. The lead recommends float16 there: the weights are already 8 or 4
  bits, and llama.cpp's CUDA path rounds activations to 8 bits. Defaults are
  a numerics call, so it is the owner's. With float32 kept, MLX on NVIDIA is
  about 20% behind llama.cpp with 8 clients.

### Paged follow-ups: copy only for a second writer; Qwen3.5 paged

(PROVISIONAL, owner to confirm; genclj-6qb6, genclj-1ovd, 2026-11-08)

**Copy-on-write.** gen.llm.paged records how far each block has been written.
A holder appends in place unless another holder wrote past its end, so only a
second writer copies. Copies per SMC run fell from 1012 to 44.
- A test writes to a parent after its fork, and both histories hold.
- Folding the copy into the compiled step measured within noise of two
  calls, so it is dropped.
- MLX's block write is one scatter of slices: IFused :put-blocks.

**Token SMC**, Qwen3-0.6B, 16 particles, tokens/s, before → after:

| | float32 | float16 |
|---|---:|---:|
| paged | 604 → 701 | 713 → 787 |
| continuous | 637 → 751 | 722 → 836 |
| static | 345 | 649 |
| llama.cpp | 924 | |

**Qwen3.5 in the paged mode.**
- Its attention layers are paged.
- Its GatedDeltaNet states stay one row each: a group stepped by its own
  slots goes in place; any other step stacks the states (:stacked-steps).
- The contracts pass on the tiny Qwen3.5 and its MoE variant.
- On Qwen3.5-0.8B it doesn't pay yet: paged 595-698 against static 645-728
  tokens/s, and llama.cpp 276. The states' stacking dominates at short
  contexts.
- The continuous mode stays Qwen3-only, and refuses Qwen3.5 loudly.

**The write cost (genclj-1ovd)** was gen.clj's own GC, gone on main (see the
k-quants entry's note). Closed.

**The lead's calls, on the implementing agent's recommendations:**
1. **Copy only for a second writer.** Not taken: vLLM's rule, which always
   copies a shared partial block.
2. **The in-step copy is dropped;** two calls are kept.
3. **IFused :put-blocks** is internal.
4. **Qwen3.5 stacks its states,** as in the static mode. Not taken: a state
   pool, which meets the same MLX limit on gathering and writing in one
   call.

*Merged with genclj-5t5l's failure-safety.* A paged read normalizes its row
before writing the pools, and a step stores its pools at once.
- Under Qwen3.5 the step also consumes the recurrent states. After a failed
  normalization those rows' groups were donated, so they are lost: the
  continuous mode's behaviour, genclj-8ni2's to align.

### Safetensors checkpoints: gen.array/load-safetensors

(PROVISIONAL, owner to confirm; genclj-9pn4, 2026-11-08)

**What was built.**
- **array/load-safetensors** (experimental) takes a file, an index or a
  checkpoint directory. MLX reads with mlx_load_safetensors on the CPU
  stream; the JVM backend uses a host reader.
- **gen.array.json** is a strict internal JSON reader, with no dependency.
- **qwen3/load and qwen35/load take a checkpoint** wherever they took a GGUF.
  The checkpoint becomes the GGUF llama.cpp's converter would write from it,
  checked against the pinned llama.cpp's conversion/qwen.py: Qwen3-Next's +1
  on the norms, ssm_a = -exp(A_log), value heads tiled, the conv kernel
  squeezed, per-expert matrices stacked.
- **MLX-community quantized checkpoints** load as load-gguf's quantized maps.

**Checked.**
- Six tiny checkpoints (Qwen3, Qwen3.5 with unequal heads, MoE, sharded,
  mlx-lm's 8-bit layout) give the same config, weights and logits as their
  GGUFs on the JVM: a gap of 0.0.
- Qwen3.5-0.8B's HF bf16 against ggml-org's BF16 GGUF on Metal: every weight
  equal, max |Δ log p| 2.7e-5, argmax 128/128.
- Qwen3-0.6B against its F16 GGUF: 4.4e-5.
- Planted at real scale: the +1 missed moves log p by 25.6 nats.

**The lead's calls, on the implementing agent's recommendations:**
1. **ILoading gains -load-safetensors.** It is an experimental, unreleased
   protocol, so the 1.x rule doesn't bind.
2. **load-gguf reads BF16,** through the host reader on MLX.
3. **F64 tensors load on the JVM only.** MLX 0.32.4's loader rejects files
   holding them.
4. **The +1 is applied in float32,** as transformers does. mlx-lm does it in
   bf16.
5. **Eps and theta are rounded to float32,** as the GGUF stores them.
6. **A tied output drops its stored lm_head copy.**
7. **The fused gate_up_proj [E, 2I, H] stays loud.** The converter's source
   confirms its layout, so support is a few lines when wanted.
8. **mlx-lm's layout is detected from the conv kernel's shape,** as mlx-lm's
   sanitize does.
9. **The JSON reader is strict:** duplicate keys are loud, and NaN and
   Infinity are accepted.
10. **Fixtures add 2.1 MB** under test/fixtures/safetensors, from random
    weights, regenerated deterministically.

### Cleanup: continuous failures, the llama.cpp exit hook, f16 widening

(PROVISIONAL, owner to confirm; genclj-8ni2, genclj-8dtg, genclj-ntre,
2026-11-08)

**The continuous mode's failures.** batch-forward! stores its pools at once,
normalizes, then writes the books, as the paged step does. A score that loses
every slot fails the requests in flight with :lost-slots.
- Not taken: scoring through the scheduler. That would batch scores (no
  longer bit-for-bit alone) and need a scratch slot per concurrent score.

**llama.cpp at exit.** A shutdown hook closes each engine on its own thread
after its current operation, then each unused model under its write lock. It
waits at most gen.llama.close-at-exit seconds (default 10; 0 turns it off).
- gen.llm.slots/closer-of (experimental) is new.
- A child JVM exiting with engines open, one decoding: exit status 134 (the
  ggml-metal assertion) without the hook, 0 with it.
- The MLX engine exits cleanly in every mode.
- Not taken: GGML_METAL_NO_RESIDENCY=1, its cost unmeasured.

**llama-bench.** native/build-llama.sh --bench builds it statically into
native/build/llama/<platform>/bin/llama-bench.
- The line for CLAUDE.md's Running section is left for the owner: the lead
  doesn't edit CLAUDE.md on an agent's request.
- genclj-ntre stays open until then.

**f16 widening.** gen.array.host/halves->floats is bit-identical on all 65,536
patterns.
- On the M4, at 16 rows: float16 0.56 → 0.38 ns per entry, bfloat16
  0.43 → 0.10 (Graal).
- Most of the gain comes from bounding loops by the array's length, not by a
  long argument. gen.llm.host's kernels loop to long arguments too, which may
  be why Graal on x86 doesn't vectorize them: worth a widen-bench on fenrir.

### The real Qwen3.6 MoE, the off-Mac bands from one place, the book on x86

(PROVISIONAL, owner to confirm; genclj-utti, genclj-fc6u, 2026-11-08)

**The real MoE checks out.** Qwen3.6-35B-A3B (Q4_K_M) on fenrir, MLX CUDA
under C2, against llama.cpp's CUDA build:
- Its largest gap over llama.cpp's top 64 is 2.35 nats on the 20 general
  prompts and 6.13 on 20 digit prompts. The medians are 0.25 to 0.75, and the
  argmax matches in 197 and 196 of 200. Against the CPU build the largest
  gaps are 1.87 and 5.88.
- The gap is llama.cpp's. Its own batched and lone reads on CUDA differ by
  3.88 and 6.09. MLX's forward on its CPU gives our CUDA gaps to three digits.
- The largest gaps sit at the second-ranked token behind a top token of
  p ≈ 1 (llama.cpp −10.9 nats, ours −17.0).
- The band is 12.3 nats (twice 6.13, rounded up).
- The planted errors, least and median over the general prompts:
  - value heads grouped 6.80 / 14.2;
  - experts' gate and up swapped 9.57 / 14.2;
  - router not renormalized 2.78 / 6.08;
  - shared expert ungated 7.18 / 13.8.
  qwen35's renormalization became a private function (routing) so the router
  error could be planted. against-llama/mutations-of adds the value-head and
  MoE errors where a model has them.
- **Unequal heads are checked on a trained model:** the MoE has 16 key and
  32 value heads. No 4B or 9B download was needed.
- **Decode speed, tokens/s:** 47.9 against llama.cpp's 208 at batch 1, and
  488 against 964 at batch 16. MoE decode on MLX CUDA is 2 to 4.3× behind,
  which bears on the owner's NVIDIA decision (see "The sampling path on
  fenrir").

**Off-Mac bands come from one place: against-llama/band.**
- The bands are per platform and family. Each is twice the largest gap
  measured against llama.cpp's CUDA and CPU builds over 40 prompts (20
  general, 12 of GRPO's chat prompts, 8 texts of numbers), rounded up:

  | Model | vs CUDA | vs CPU | llama.cpp's own (CUDA) | Band |
  |---|---|---|---|---|
  | Qwen3-0.6B Q8_0 | 1.05 | 1.19 | 0.85 | 2.4 (was 0.65) |
  | Qwen3.5-0.8B Q8_0 | 3.38 | 3.53 | 2.59 | 7.1 (was 0.65) |
  | Qwen3.6-35B-A3B Q4_K_M | 6.13 | 5.88 | 6.09 | 12.3 |

- The cause is llama.cpp's activation rounding. Its CPU build is
  deterministic (its own spread is 1e-7) yet 1.2 to 3.5 nats from ours: both
  its CUDA and x86 CPU builds round activations to 8 bits for their dot
  products, and its Metal build does not. MLX on its CPU matches its CUDA to
  three digits.
- The Mac's bands are unchanged: Qwen3 0.075 (Q8_0) and 0.47 (K-quants),
  Qwen3.5 0.035.
- The qwen3, qwen35, LoRA, SFT and GRPO tests all take their band from it.
  GRPO's adapter is now held to the band, where it used to be held to
  twice it.
- **The catch:** off the Mac most planted errors fall inside these bands.
  Metal's bands and the fixed 0.75-nat planted-error threshold are what
  check the conventions.
- Not taken:
  - a band per prompt set (general prompts only: Qwen3.5 1.8, the MoE 4.7);
  - another measure off the Mac (the max over tokens above −10 nats, or a
    prompt's median);
  - a llama.cpp build that keeps activations in float, as the reference;
  - keeping 0.65 in the LoRA and SFT tests, whose own prompts measured at
    most 0.53.
- Still unmeasured: K-quant bands for Qwen3.5 on Metal (still 0.035; about
  0.83 by the earlier k-quants measurement) and for Qwen3 off the Mac.
- This supersedes "Qwen3.5/3.6 and gen.array/gated-delta" point 6 (0.65
  borrowed from Qwen3) and GRPO's band note (1.3 nats, re-measured here).

**grpo-llama-test had crashed on main** since the Qwen3.5 LoRA merge, where
lora/targets became keyed by family. It is fixed. On fenrir's rerun the
adapter is 1.07 nats from llama.cpp (base 0.731), and the held-out pass rate
went 0.615 → 0.771 (z 7.1).

**The book on x86: the JVM's math, not a bug.**
- Every failure on fenrir (14 values and 3 figures in chapters 023, 028 and
  054) is a `; float64` display of a chaotic run: NUTS's and HMC's
  adaptation, and a tanh network after 3000 Adam steps. Their banded checks
  pass.
- The JVM on x86 rounds exp, log, log10, pow, tan, tanh and cbrt differently
  from fdlibm: 1 to 2 ulp on 2 to 12% of inputs, identically under C2 and
  Graal. On the Mac only sin and cos differ (7%).
- gen.book takes a fingerprint of the JVM's math results (math-fingerprint)
  and compares it with the Mac's (reference-math?). Where they differ,
  `; float64` values and figures are not checked, a note says so, and sync
  writes no figures. gen.book-test covers both branches.
- The whole book passes on fenrir under C2 and Graal. On the Mac every value
  is still checked.
- Not taken:
  - StrictMath (fdlibm) for transcendentals everywhere. Results would be
    identical on every platform, but x86 transcendentals get 1.3 to 2.5×
    slower (pow 5× under C2), about 100 call sites change, and the Mac's
    sin/cos displays would need recording again. That is a numerics call
    for the owner.
  - Fewer shown digits: a chaotic run's tree-depth counts differ entirely.

### Converting models: gen.llm.convert, by llama.cpp's quantizers

(PROVISIONAL, owner to confirm; genclj-t007, genclj-whi8, 2026-11-08)

**What it is.** gen.llm.convert (experimental):
- `convert!` writes a GGUF file's model as another GGUF file on the host, a
  tensor at a time, with no backend. `plan` is its dry run.
- `quantize` holds MLX's affine layouts in memory, at any of MLX's bit widths
  and group sizes, through gen.array/quantize. It takes mlx-lm's recipes
  (`:mlx-4bit`, `:mlx-8bit`, `:mlx-mixed-4-6`, `:mlx-mixed-3-6`).
- A recipe is one of llama.cpp's file types (`:q4_k_m`, `:q8_0`, `:bf16`, …,
  with its per-tensor choices and fallbacks), or `[pattern value]` rules that
  may end in one, as llama-quantize's `--tensor-type`.
- gen.array.gguf/write-gguf writes every type: F32, F16, BF16, Q8_0, Q4_0,
  Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K. It used to write F32 only.

**The decisions:**
1. **The output format is GGUF only, for now.** gen.clj and llama.cpp both
   load it, and it carries the tokenizer and architecture. MLX-only layouts
   stay in memory; a file for them waits for genclj-k8gx (an MLX safetensors
   writer). Not taken: MLX safetensors now, which needs the safetensors
   loader's names.
2. **GGUF's types are encoded by llama.cpp's own quantizers,** ported into
   gen.array.gguf.quants (MIT, in THIRD-PARTY-NOTICES), not by
   gen.array/quantize. So our files equal llama-quantize's. Not taken: MLX's
   algorithm for Q4_1 and Q5_1, whose codes differ from llama-quantize's.
3. **`{:bits 8 :group-size 32}` maps to Q8_0,** whose bias is fixed, not
   free; 4/32 maps to Q4_1 and 5/32 to Q5_1. Other widths are refused as
   MLX-only. Not taken: refusing 8/32.
4. **Byte-exactness is pinned to Apple clang's build of llama.cpp.** The port
   follows libggml's disassembly: clang's fused multiply-adds, which it uses
   only in loops it does not vectorize, and arm64's saturating float-to-int
   conversion (Qwen3.5's subnormal blocks). A GCC build may differ in a few
   K-quant sub-blocks.
5. **BF16 loads,** reversing "K-quants and BF16 are loud" for BF16 (genclj-9pn4
   on main did the same). One host reader widens F16 and BF16 in gguf and
   safetensors: gen.array.host/halves->floats. Not taken: quantfmt's scalar
   bfloat16-value inside gguf, leaving safetensors its own copy.
6. **Requantizing is refused** unless `:allow-requantize?` is set, as in
   llama.cpp. A recipe made only of rules drops `general.file_type`.
7. **MLX's in-memory scales are float32,** where mlx-lm keeps float16.
8. **ggml-quants.c's origin is not traced:** native/build-llama.sh's checkout
   is shallow. THIRD-PARTY-NOTICES cites llama.cpp at v0.9.0 (cd959ba). The
   owner checks the file's history before a release.

**The evidence:**
- Byte for byte, every tensor, the metadata and the file size equal llama.cpp
  v0.9.0's quantizer output. This holds on the fixtures (all 8 quantized
  types, BF16, F16 and F32) and on whole models: ggml-org's Qwen3-0.6B as Q8_0
  (2.2 s on the M4) and Q4_K_M (27 s), Qwen3.5-0.8B as Q4_K_M and Q8_0, and
  Qwen3.5-0.8B BF16 → Q8_0 and Qwen3-0.6B F16 → BF16 after the merge.
- llama.cpp reads our files as its own: its log-probs on them equal those on
  its own files, and qwen3-llama-test and qwen35-llama-test pass on them.
- Our reader reads what the writer wrote, for every type, on both backends.
- On Metal, Qwen3-0.6B's KL against F16, excluding position 0
  (median / max): Q8_0 0.0005 / 0.005; Q4_K_M 0.058 / 0.36; MLX 4-bit
  group 64 0.15 / 1.17.
- Mutations: 15 planted, 13 caught. One FMA mutation changes no fixture byte
  and only the real-model test catches it. One is equivalent: no input tells
  it apart.

**docs/benchmarks.md** gains "Language models (1.3.0)": decode at batch
1/16/64, prefill, token SMC, continuous batching, training, llama.cpp's
ceilings, loading and converting weights, and what is not yet measured
(genclj-pnw2). Every number has its setting and source, and disagreeing
sources are both shown.

**The merge with main.** quantfmt and main had each made BF16 load.
gen.llm.convert's writing and host-values are quantfmt's. The one reader is
main's widening. main's load-safetensors stays, and so does main's word for
MLX's own BF16 reading, "unverified": MLX 0.32.4's gguf.cpp has no BF16 case.
The two BF16 load tests are one, bf16-loads-its-bits, against gguf-py's
reference and llama.cpp's.

### Continuous batching for Qwen3.5/3.6; paged attention v2

(PROVISIONAL, owner to confirm; genclj-6qb6, 2026-11-08)

**The continuous mode runs Qwen3.5/3.6,** dense and MoE, where it used to
refuse them.
- qwen35/paged-batch-step (and compile-paged-batch-step) is qwen3's batched
  step for the hybrid family. Each decode row is one token of its own
  sequence. A request's prefill chunk runs as one sequence through the
  GatedDeltaNet layers, in the same forward as the other requests' decode
  rows.
- The chunk's padding rows get beta = 0 and g = 0, so they leave its state
  unchanged. Its conv state is taken at `:chunk-len`, so chunks still pad to
  a power of 2 and don't trace once per length.
- Decode and chunk attention is common/batch-attention, shared with qwen3.

**States stay groups and are read in place, not pooled.** A step reads its
GatedDeltaNet states (paged and continuous modes) in one of three ways:
- a lane, a group only the step's slots hold: stepped in place and consumed;
- a shared group's consecutive rows, the common case with LMs: read in place
  through a new optional `held` argument of the compiled steps, not consumed;
- anything else: one copy (a gather, or slices and one concatenate), where
  there used to be up to three.

Metal, 16 rows, ms per step with GC:

| Option | ms |
|---|---:|
| Lane | 6.7 |
| Shared group in place | 12.5 |
| One gather | 17.7 |
| Two groups, padded | 23.9 (42 before) |
| A state pool | 18–26 |

- A pool can't win: the Metal gated-delta kernel's output can't be written
  into a pool in place, so a pool needs a write-back copy. In one call MLX
  sometimes copies the whole pool (32 ms at 8 rows).
- Not taken: a pool; a C++ primitive that writes in place (route B).
- **A consumed lane's slots are reported lost** (`:lost-slots`) if its rows
  then fail to normalize. The paged mode used to leave them holding consumed
  arrays. Not taken: never consuming lanes, which keeps the slots but costs
  more GC (6.7 → 12.5 ms).
- Not taken: unpadded chunks, which trace once per length; separate pools
  and states arguments in place of `held`.

**Measured, Qwen3.5-0.8B Q8_0 on Metal, tokens/s:**

| | continuous | paged | static | llama.cpp |
|---|---:|---:|---:|---:|
| 8 clients, f16 | **233** (p50 21.6 ms) | 74 | 72 | 69 |
| 8 clients, f32 | **152** (p50 38.7 ms) | 66 | 65 | 69 |
| SMC, 16 particles, f16 | **885** | 826 | 805 | 282 |
| SMC, 16 particles, f32 | **726** | 677 | 684 | 281 |

- The paged mode also gained: 744 → 826 (f16) and 627 → 677 (f32).
- At decode batch 16, f32: continuous 461, static 441, llama.cpp 411.
- On the real 0.8B with 4 concurrent callers, rows are at most 2.2e-5 nats
  from the model's own forward.
- **An open finding:** a forward costs about 35 ms at 8 rows, about the same
  as at 16 (12 ms at 1). That is a large fixed cost per step, worth profiling.

**Tests.** JVM: 13 namespaces, 119 tests. MLX: 9 namespaces, 79 tests.
array_engine_qwen35_continuous_test covers:
- the batched step against the concatenating cache;
- the Branching, LM and token/SMC contracts on the tiny Qwen3.5 and its MoE;
- concurrent clients against each running alone;
- the three ways of reading states.

Five planted bugs fail the clients by 1.5 to 6.3 nats on both backends:
- a decode row given another row's states;
- a chunk started from zeros;
- padding rows writing the state;
- a chunk run as separate sequences;
- a freed slot's states kept.

**Paged attention v2 on Metal.**
- kernels-community's paged_attention_v2_reduce is ported from the same file
  and commit as v1 (28a49f7). The partition kernel is v1's body with
  PARTITION_SIZE 512, its maxima and sums as outputs. Rule 10's headers and
  THIRD-PARTY-NOTICES are extended.
- The source was the copy the v1 agent downloaded. The lead re-fetched the
  file from Hugging Face at 28a49f7: byte-identical (51,650 bytes).
- v2 runs behind the same op past v1's 32 KiB of threadgroup scores. Table
  widths are padded to powers of 2, so it starts at about 4k-token contexts,
  where the composite used to run.
- Against the composite at 1 to 32,768 positions: 3.1e-6 in f32, 1.8e-4 in
  f16. Three planted source bugs fail it by 0.16 to 0.28.
- Qwen3-0.6B decode at 16k positions: 78 ms for 1 row and 279 ms for 8 rows,
  against the composite's 351 and 3,338. At 4k: 23 against 102.
- Partials stay in the operands' dtype, as upstream keeps them. Not taken:
  float32 partials, which means editing the ported kernel.
- docs/array-api.md's paged-attention paragraph now names v2. It describes
  the backend; the API is unchanged.

### Paged attention on CUDA: vLLM's v1 through mlx_fast_cuda_kernel

(PROVISIONAL, owner to confirm; genclj-1daf, genclj-6qb6, 2026-11-08)

**What's built.** gen.backend.mlx.paged-attention-cuda (new) runs vLLM's
paged attention v1 through mlx_fast_cuda_kernel, with the Metal kernel's
operands, dtypes and block layouts.
- nn.clj's `:paged-attention` tries Metal's kernel first (v1, or v2 past its
  limit), then CUDA's, then the composite. The composite stays the
  reference.
- The integers vLLM's launcher passes (KV heads, table width, strides) are
  read from the inputs' shapes, so one compiled kernel covers every table
  width.
- It covers float32, float16, and bfloat16 from compute capability 8.0 (read
  from MLX's device info); blocks of 8, 16 or 32 positions; heads up to 256.
- Still route A: no C++ of ours. Nothing blocked the port.

**The decisions:**
1. **Ported from vLLM itself,** at 258f8de, the last commit with the kernel
   (vLLM deleted it in d715b3a, 2026-07-02): csrc/libtorch_stable/attention's
   attention_kernels.cuh, paged_attention_v1.cu and attention_utils.cuh;
   csrc/attention's attention_generic.cuh and dtype files; csrc/cuda_compat.h.
   - The files are Apache-2.0, adapted by vLLM from NVIDIA's
     FasterTransformer, with no NOTICE file in either.
   - They are joined, each keeping its license header, with rule 10's header
     and Changes line and a THIRD-PARTY-NOTICES entry. The kernel function
     is verbatim. Includes NVRTC lacks are replaced by a short prologue, and
     the fp8 conversions are declared, not ported.
   - Not taken: kernels-community's CUDA copy, or an older vLLM tag.
2. **Shared memory in five tiers** (1,024, 2,048, 4,096, 8,192 and 16,000
   positions), each a kernel of its own name, with the composite past 16,000
   (64 KiB, which every GPU MLX's CUDA backend supports). MLX sets the
   extra-shared-memory attribute only once per kernel name. Not taken:
   - one kernel at the maximum (lower occupancy at short contexts);
   - a kernel per table width (more compiles);
   - a per-device cap (about 25,000 positions on sm_120), which MLX's
     device info can't report.
3. **A namespace of its own,** for its own provenance entry. Not taken: one
   namespace for both kernels.
4. **128 threads a block,** vLLM's CUDA default. Metal uses 256.
5. **array-engine-llama-test takes the per-platform band** (against-llama's:
   2.4 off the Mac). It had used Metal's 0.075, and failed on CUDA where the
   gap is 0.19 to 0.32. Not taken: keeping it Metal-only.

**mlx_fast_cuda_kernel's limits that shaped it:**
- shared memory is fixed when a kernel is made (hence the tiers);
- there are no scalar arguments, so the scale goes in as a 0-d array and the
  integers come from shapes;
- there is no disk cache, so each process compiles each variant once (about
  0.8 s for the first launch, then 0.9 ms);
- MLX hashes the 63 KB source on every launch, a small host cost (Metal does
  the same).
Had the port stalled, a hand-written kernel, not our own C++, was the
fallback: one native artifact and no ABI tied to MLX's internals.

**Tests on fenrir (RTX PRO 6000, C2):**
- The engine set passes: 102 tests, 1,094 assertions, no native suite
  skipped. With CUDA graphs on: 19 tests, 0 failures.
- Against the composite: 2.9e-6 in float32, and one output rounding step in
  the half floats (2.0e-3 float16, 1.6e-2 bfloat16). The bands are four times
  those.
- 12,000 positions run the kernel at its largest shared memory, and 16,500
  fall back to the composite.
- Four planted bugs fail by at least 0.47 against bands of 6e-2: the next
  sequence's block table, keys read one table entry on, the scale dropped,
  and the lengths one short.
- After merging with v2 on main: the Mac's Metal run passes (63 tests), and
  so does the JVM battery (2,409).

**Measured on fenrir, tokens/s** (each engine in its own JVM, median of 5;
paged and continuous cells show kernel / composite; 8 clients at once):

| Model, dtype | Run | Static | Paged | Continuous | llama.cpp CUDA |
|---|---|---:|---:|---:|---:|
| 0.6B Q8_0, f16 | SMC 16 | 797 | 1158 / 999 | 1202 / 989 | 2228 |
| 0.6B Q8_0, f16 | 8 clients | 147 | 219 / 181 | **555** / 431 | 503 |
| 0.6B Q8_0, f32 | SMC 16 | 740 | 1022 / 721 | 857 / 858 | 2214 |
| 0.6B Q8_0, f32 | 8 clients | 104 | 195 / 151 | 409 / 375 | 502 |
| 0.6B F16, f16 | SMC 16 | 778 | 920 / 841 | 986 / 971 | 2195 |
| 0.6B F16, f16 | 8 clients | 142 | 187 / 149 | **527** / 451 | 454 |
| 8B Q8_0, f16 | SMC 16 | 620 | 842 / 766 | 841 / 791 | 1484 |
| 8B Q8_0, f16 | 8 clients | 73 | 81 / 76 | **231** / 220 | 144 |

- The kernel speeds the paged mode by 9 to 42%.
- Continuous batching now beats llama.cpp with 8 clients on CUDA, as it does
  on Metal (302 against 103 there).
- Token SMC on CUDA is still 1.8 to 2.6× behind llama.cpp. On Metal the gap
  is 836 against 924.

**For the owner:** in float16 SMC the paged mode now beats the static one on
both Metal and CUDA. That was the stated condition for making paged the
default. Flipping it is the owner's call; the default is still static.

### Token SMC on CUDA: reads from MLX's buffer, rows lent, forks in one call

(PROVISIONAL, owner to confirm; genclj-4acj, 2026-11-08)

**Where the gap was.** On fenrir (C2), with Qwen3-0.6B Q8_0 in f16 and 16
particles, one SMC step took 17.1 ms paged against llama.cpp's 7.3. The
profile is bench/gen/smc_profile.clj.
- Reading the logits to the host took 3.1 ms (1.2 of it widening), copying
  each row 1.8, fork hand-offs to the engine thread 0.6, and normalization
  2.2.
- MLX does 2.34 evaluations a step; llama.cpp syncs once.
- **Most of what remains is MLX's own encoding on CUDA:** 7.0 ms encoding
  against 0.3 waiting on the GPU, at 16 rows. Turning CUDA graphs off,
  raising MLX_MAX_OPS_PER_BUFFER from 50 to 1000, or pinning to P-cores
  changes nothing.
- The 8B is bound by the GPU instead: 13.3 ms an evaluation against about
  6.5 for llama.cpp, whose 8-bit matmuls are faster.

**The fixes.** Every one gives the same draws for the same key.
1. **MLX reads 16-bit floats straight from its buffer** in Java, without
   copying the bit patterns, bit-identical to before. Reads of 2^18 entries or
   more run in parallel chunks on Java's common pool, float32 included.
2. **On CUDA, a read of at least 1 MiB is first moved to the host in one
   cudaMemPrefetchAsync.** MlxRuntime calls libcudart.so.13 directly; where it
   is absent (macOS, a CPU build) nothing happens. The property
   gen.mlx.prefetch-bytes sets the threshold, and 0 turns it off.
   - It replaces page faults on MLX's managed memory.
   - Memory that isn't managed is left alone, its error cleared so MLX never
     sees it.
3. **Rows lent, not copied:** token SMC and completions borrow the LM's own
   row arrays (lm/batch-next-logprobs-lent, the optional protocol LentRows).
   SMC still copies under a user-function proposal.
4. **A step's forks in one call** to the engine thread, all or none
   (br/fork-each, the optional protocol ForkEach). The LM cache uses it.
   - Both protocols are new and optional, so the 1.x promise holds.

**Measured on fenrir, tokens/s** (median across JVMs, main and the branch
alternated because a JVM's result varies by up to 30%):

| Run | main | branch | llama.cpp, main | llama.cpp, branch |
|---|---:|---:|---:|---:|
| SMC 0.6B f16, paged | 1008 | 1272 | 2177 | 2579 |
| SMC 0.6B f16, continuous | 1056 | 1445 | | |
| SMC 0.6B f32, paged / continuous | 834 / 957 | 983 / 972 | 2124 | 2576 |
| SMC 8B f16, paged | 835 | 882 | 1455 | 1600 |
| 8 clients 0.6B, continuous | 522 | 549 | 505 | 501 |
| 8 clients 8B, continuous | – | 236 | – | 144 |

- llama.cpp gains too: its LM path also takes the lent rows and the
  one-call forks.
- The 1158 recorded for paged-cuda was a high sample; today it gives 970.
- Under Graal: paged 1200, llama.cpp 2132, against C2's 1272 and 2579.
  Keep C2 on x86.
- 4 host threads stay the default (8 moved results by at most ±4%).

**Not taken:**
- Borrowed rows: keeping the copies; copying on a thread pool; relaxing
  next-logprobs's promise of new arrays.
- ForkEach: one engine call a fork; batching through resample, which
  consumes the parents.
- The prefetch: leaving the page faults; patching MLX's C++; a direct
  device-to-host copy, which needs a device pointer MLX doesn't expose.
- Parallel reads: single-threaded reads; a pool the backend owns; a new
  array function that reads raw 16-bit patterns so normalization skips
  widening, which would change the backend contract.

**Levers left, the owner's call:**
- fused q/k/v and gate/up projections (about 84 of about 600 kernels a step;
  it affects LoRA);
- a newer MLX;
- a compiled paged prefill;
- the scores-only advance (genclj-t46p).

**Tests.**
- smc-test checks lent rows against a copying LM. Rows returned reversed fail
  7 checks; a draw writing into its row fails 8.
- branching-test covers fork-each. Reversed counts fail it, and so does a
  refusal that leaves forks behind.
- mlx-test checks reads over every float16 and bfloat16 bit pattern, across
  chunks and past the parallel threshold. Planted bugs in the Java fail 18,
  12 and 1 checks.
- array-engine-qwen35-continuous-test's band (2e-5, measured on Metal) fails
  on CUDA by 2.2–2.4e-5, on main as well. Its per-platform band is being
  measured (cuda-check).

### Qwen3.5's step on Metal: 5-15 float32 rows as 16, dead states released

(PROVISIONAL, owner to confirm; genclj-qu77, 2026-11-08)

**What the profile found.** bench/gen/batch_profile.clj times one compiled
paged batch step on the M4, Qwen3.5-0.8B Q8_0, in ms per step:

| | 1 row | 8 rows | 16 rows |
|---|---:|---:|---:|
| f32, states in place | 12.8 | **38.0** | 34.4 |
| f32, states read in place by forks | 13.0 | 39.6 | 40.8 |
| f16, states read in place by forks | 12.6 | 23.1 | 38.7 |
| Qwen3-0.6B f32 | 10.4 | **30.4** | 26.9 |

1. **8 float32 rows were slower than 16.** Below its limit of 13 to 15 rows
   on the M4, MLX 0.32.4 uses qmv_wide, which re-reads the weights for every
   5 rows; 16 rows use qmm. The output projection took 9.6 ms at 8 rows and
   5.5 at 16. In float16 qmv_wide is the faster one.
2. **Steps that read their states in place paid for the GC.** Each step made
   18.9 MB of new states per row, so MLX ran System.gc every 7 to 13 steps
   and allocated fresh buffers in between: about 3 ms a step at 8 rows (f16)
   and 6 ms at 16.
3. **The gated-delta kernel is at memory speed:** 0.44 ms a row (86 GB/s).
   Variants gained at most 7%, so the ported kernel stays.
4. **No recompiles in steady decode.** In the 8-client bench, 2 or 3 mixed
   forwards a run still retrace (50 to 125 ms each).
5. **The rest is the host round trip.** At 8 rows in f16 a step takes
   20.2 ms against a pipelined floor of 13.0: read 1.2, normalize 1.4, draw
   0.5, and about 3.3 ms of GPU idle while encoding.

**The decisions:**
1. **gen.backend.mlx.nn's qmm-rows pads float32 inputs of 5 to 15 rows with
   zero rows to 16,** so MLX uses qmm. It applies only on Metal generations
   15 and 16 (M3, M4, not Ultra), where it was measured.
   - f32 at 8 rows: 38.0 → 28.7 ms (Qwen3-0.6B 30.4 → 22.5).
   - **Draws can change** at near-ties: values move within round-off. The
     CHANGELOG has a Draws line.
   - Not measured: M5, M2, Ultra (the M5's limits differ).
   - Not taken: raising :min-rows, which costs gated-delta work for every
     padded row; asking MLX upstream to tune qmv_wide; leaving it.
2. **gen.array/release! (experimental), with the optional protocol
   IReleasing.** On MLX it frees an array's buffer now, as a donated
   array's; a later read is loud (:donated). On the JVM backend it does
   nothing.
   - The engine releases any group of states or caches no slot holds any
     more, in all three modes. It is a runtime; rule 3 keeps release! out of
     models, handlers and distributions, as its docstring says.
   - Held steps are now as fast as in-place ones: f16 at 8 rows
     23.1 → 20.2 ms, at 16 rows 38.7 → 32.8.
   - Not taken: an internal-only function; consuming dead groups through a
     donating no-op compile (no new API, but a hack); shortening
     MlxRuntime's 10 ms GC quiet wait (no buffer reuse).

**Measured, Qwen3.5-0.8B Q8_0 on Metal, tokens/s** (main and the change run
alternately in one session, three rounds for f16):

| | before | after | llama.cpp |
|---|---:|---:|---:|
| 8 clients, f16 continuous | 216–243 | **255–268** | 69–70 |
| read latency p90 (ms) | 53–60 | **29** | 111–123 |
| 8 clients, f32 continuous | 154 | **205** | 66–69 |
| SMC 16, f16 continuous | 871–907 | 917–937 | 280 |
| SMC 16, f32 continuous | 724 | 756 | 281 |

- Qwen3-0.6B gains too: f32 static SMC 352 → 529, and f32 continuous with 8
  clients 214 → 264.

**Tests.**
- nn-test's quantized-matmul-of-any-rows checks 1 to 17 rows against
  dequantize-then-matmul, and on Metal that each row equals the same row of
  a 16-row product. Taking the product's last rows fails 22 checks; skipping
  the padding fails 8.
- dropped-groups-are-released runs all three modes. Releasing a group still
  held fails all three on the JVM and throws :donated on Metal. Never
  releasing also fails all three.

**Left, the owner's call:**
- mixed-forward recompiles: bucket the shapes, or accept them as warm-up;
- more than one prefill chunk a forward (prompt admission is about 23% of an
  8-client run, at 44 to 61 ms a chunk);
- half-float recurrent states, which halve their traffic but change numerics;
- host normalization and host threads (genclj-t46p).

### Main on fenrir's CUDA; Qwen3.5's engine bands per platform

(PROVISIONAL, owner to confirm; genclj-cbvi, 2026-11-08)

**Main (55edf11, after the night's merges) passes on MLX's CUDA** (fenrir,
C2, cuda-121.sh's options) once the Qwen3.5 engine tests' bands are
widened.

| Suites | Tests | Assertions | Failures |
|---|---:|---:|---:|
| mlx, conformance, array.paged, llm.paged, gguf, convert | 104 | 1,155 | 0 |
| the array-engine suites, Qwen3 and Qwen3.5 | 57 | 508 | 0 |
| smc, branching, lm, models.qwen3, models.qwen35 | 92 | 832 | 0 |
| llama.cpp comparisons, Qwen3-0.6B and Qwen3.5-0.8B | 12 | 296 | 0 |
| the light tier of gen.array.\*, gen.backend.\*, gen.llm.\* | 885 | 9,226 | 0 |

- The JVM battery under C2 there passes: 2,412 tests.
- On fenrir, -X:test:fast:par is not GPU-free. env.sh exports GEN_MLX_LIB, so
  the battery runs MLX's native tests (about 1 GB of GPU memory) and counts
  as a GPU JVM.

**The band.** bench/gen/qwen35_continuous_noise.clj measures it: the test's
clients over 10 runs, 20 more clients with contexts up to 144 tokens (the
test reaches 44), and every comparison of both Qwen3.5 engine test
namespaces.
- **CUDA:** the largest gap is 6.8e-5, on a row of 121 tokens; median 3.8e-6,
  99th percentile 4.6e-5. The gaps are heavy-tailed over contexts, not a
  drift. A wrong state gives gaps of nats.
- **MLX's CPU on x86:** the old 2e-5 band failed there too, at main, by up to
  2.53e-5. The largest gap is 3.7e-5.
- **New bands, twice the largest gap rounded up:** CUDA 1.4e-4; MLX's CPU off
  the Mac 8e-5. The JVM keeps 2e-6 and the Mac 2e-5. gen.test-support gains
  ts/cuda?.
- **A planted bug:** the decode rows' states rounded to float16 each step
  (gaps 7e-4 to 5e-3) fails the deftests 35 times on CUDA and 40 on the CPU
  under the new bands. A decode row given another row's states fails 27
  times. The structural planted bugs give gaps of at least 4 nats.
- Not taken: a band from the tests' own rows alone (about 5e-5 on CUDA).
  A 51-token row already reaches 5.3e-5 there, so it would only happen to
  pass.

**Still open:**
- **The Mac's band for Qwen3.5 was never measured;** 2e-5 is Qwen3's.
  Measured after the merge (the lead, the M4's Metal, 2026-11-08 ~06:15):
  largest 1.67e-5 (a client of the 20 more), 1.38e-5 over the tests'
  comparisons, so 2e-5 had been only 1.2× it. The Mac's band is now 4e-5,
  by the same rule. The float16 planted bug gives 3.5e-3 on Metal. MLX's
  CPU on the Mac is not measured and takes Metal's band.
- The Thor's CUDA (sm_110) and aarch64 Linux CPUs fall under these bands
  unmeasured.
- array-engine-qwen35-test keeps 2e-5. It passed on CUDA and x86, but the
  bench doesn't measure it; it could share the per-platform band.

### Paged multi-token reads: several prompts' chunks in one forward

(PROVISIONAL, owner to confirm; genclj-6qb6, 2026-11-08)

**The op.** gen.array/paged-attention (experimental) now also takes a run of
T queries per sequence: q [B T Hq D] and lens [B T]. Query t reads the first
lens[b t] positions of its sequence's blocks; for a causal prefill that is
its own position plus one.
- The run's keys are written first (paged-write). Each sequence's blocks are
  then gathered once and read with one sdpa.
- The Metal and CUDA kernels stay single-query (rank 3), so a run always
  takes this composite.

**The engine.**
- **Continuous mode:** a forward reads K prompts' chunks, with [K MB] tables.
  - Chunks are taken first come, first served. Each has t rows, the first
    chunk's power of 2.
  - K is the largest power of 2 the budget allows, capped by the new
    `:max-chunks` option (default 16). Chunks are never padded; the rest
    wait a forward. A chunk's padding rows sit at its last position.
  - Qwen3.5's GatedDeltaNet runs the K chunks as a batch of K sequences
    (`:chunk-len [K]`). Several chunks' states are stacked, one copy.
- **Paged mode:** runs go through the same compiled batch step, in place of
  the eager read over a concatenating cache. paged-read-run and its separate
  write are gone.
- **Per forward:** 8 chunks of 32 tokens take 84.6 ms in one forward against
  150.7 ms as 8 forwards (Qwen3 f16). Qwen3.5: 111 against 183 ms.

**The decisions:**
1. **paged-attention extended to rank 4,** lens [B T]. Not taken: a new op;
   keeping it internal to gen.llm.
2. **Composite only, by measurement.** bench/gen/prefill_bench.clj times the
   attention part (writes, gathers, sdpa) at 8 to 10% of a prefill forward
   for Qwen3-0.6B and 1 to 3% for Qwen3.5-0.8B: the matrix products are the
   forward. Not taken: porting vLLM's prefill path; T single-query kernel
   calls, which re-read the context once a query.
3. **The chunk policy:** K a power of 2 rounded down, t from the oldest read,
   `:max-chunks` 16. Not taken: padding K up to a power of 2 (wasted rows);
   any K (more shapes to compile); t from the longest read (fewer split
   prompts, fewer chunks a forward).
4. **A behaviour change in the paged mode:** a run spanning several forwards
   that fails in a later one now loses its slot (`:lost-slots`). Before, the
   slot was left as it was. Not taken: keeping runs eager; compiling only
   runs that fit one forward.
5. **Recompiles on new forward shapes** (50 to 125 ms each, once a shape) are
   the owner's call: accept them as warm-up, or compile common shapes when
   the engine opens.

**Tests.**
- paged_test: a run equals each sequence's dense causal sdpa and the
  single-query reads (the kernel on Metal); 1e-12 on the JVM, 1e-5 on MLX.
- The continuous tests: several prompts in one forward equal the model's
  forward, for Qwen3, Qwen3.5 and the MoE.
- Planted bugs, all failing:
  - in the op, the causal mask a position long, and reading the next
    sequence's blocks (both over 0.1);
  - in the engine, a chunk's tokens at reversed offsets (0.6 to 4 nats), a
    chunk reading the next chunk's table (0.6 to 4.4), and chunks run as
    sequences of their own (4.6);
  - in the paged mode, a run blind to its last position.
- On the real models in f32, rows are within 7.2e-5 (Qwen3) and 3.5e-5
  (Qwen3.5) of their own forward. array-engine-llama-test's continuous gap
  is 4.1e-5 against a band of 1e-4.

**Measured on Metal, 8 clients** (5 warm runs, 6 timed; main → branch;
llama.cpp's tokens/s, p90 and time to first token beside them):

| | tok/s | p90 ms | TTFT p50 (max) ms | admission share | llama.cpp |
|---|---|---|---|---|---|
| Qwen3 f16 | 402 → 422 | 16.8 → 14.9 | 215 → 210 (429 → 361) | 27.6 → 23.9% | 99, 80, 312 |
| Qwen3 f32 | 273 → 287 | 27.9 → 23.7 | 270 → 231 (479 → 428) | 22.0 → 18.5% | 99, 77, 319 |
| Qwen3.5 f16 | 278 → 285 | 27.8 → 22.0 | 296 → 277 (612 → 596) | 27.0 → 24.0% | 70, 114, 413 |
| Qwen3.5 f32 | 208 → 217 | 38.8 → 30.6 | 324 → 346 (701 → 676) | 23.5 → 21.1% | 70, 112, 431 |

- Paged-mode token SMC (16 particles), tokens/s: Qwen3 811 → 864 (f16) and
  691 → 757 (f32); Qwen3.5 887 → 970 (f16) and 706 → 783 (f32).
- In the paged mode with 8 clients, time to first token at p50 is 22 to 39%
  lower; throughput moves by ±5%.
- Each new forward shape compiles once, so with one warm run a single run
  can move by up to 20%.

**Left:** speculative decoding itself (the op reads a draft's tokens, but no
engine uses it yet); recompiles on new forward shapes.

**On CUDA (fenrir, 2026-11-08, after the merge with moe-cuda):** the paged
and engine suites pass, on the GPU and on MLX's CPU there. Run with the band
at 0, every comparison sits at least 3.8× inside its band:

| Comparison | CUDA | x86 CPU | Band |
|---|---:|---:|---:|
| a run against dense sdpa | 8.3e-7 | 6.6e-7 | 1e-5 |
| a run against single-query reads (CUDA's kernel) | 2.6e-6 | 6.6e-7 | 1e-5 |
| Qwen3, several prompts | 4.8e-6 | 3.8e-6 | 2e-5 |
| Qwen3.5 dense, several prompts | 1.19e-5 | 9.1e-6 | 1.4e-4 / 8e-5 |
| Qwen3.5 MoE, several prompts | 1.19e-5 | 8.1e-6 | 1.4e-4 / 8e-5 |

### The MoE on CUDA: top-k by ranks

(PROVISIONAL, owner to confirm; genclj-p0ad, 2026-11-08)

**What the profile found.** Qwen3.6-35B-A3B Q4_K_M in float16, the compiled
paged step on fenrir, ms at 1/8/16 rows. The profilers are
bench/gen/moe_profile.clj and a CUPTI library built on fenrir
(~/mlx/gen-clj-tools/kprof2) that names and counts kernels.

| | before | after |
|---|---|---|
| step | 25.2 / 39.4 / 52.7 | 17.3 / 31.4 / 44.2 |
| encode (mlx_async_eval) | 21.0 / 26.5 / 30.5 | 14.2 / 20.6 / 25.1 |
| kernels a step | 5365 / 5464 / 5464 | 2545 / 2645 / 2645 |
| GPU busy | 14.1 / 18.8 / 22.9 | 9.4 / 15.2 / 19.8 |

- **The router's top-k was the main cost.** Eight argmaxes, each masking the
  last pick, cost about 11 kernels a pick: 3,500 of the step's 5,400. MLX's
  CUDA host spends about 5 µs encoding a kernel, so the batch-1 step was bound
  by the host. Larger CUDA graphs, or none, change nothing.
- **Ruled out:** host round trips in routing; copies of the expert stacks
  (row-contiguous); sorted indices (MLX's sorted kernel on CUDA takes only the
  non-affine modes); Q6_K in the hot path (only output.weight).
- **gather_qmm switches kernels at 8 pairs.** With one row a pair, it uses qmv
  below 8 pairs and qmm_sm80 from 8. Batch 1 has exactly 8, so a gate (N=512)
  runs on only 32 CTAs: 41 µs a call, 4.9 ms a step, the largest GPU item.
  At 16 rows it is near memory speed.
- **float32 is slower:** qmm_sm80 takes half floats only, so float32 falls
  back to qmm_naive (21.7 / 42.1 / 60.2 ms a step).

**The decision: qwen35's top-k ranks the experts.**
- Each expert's rank is counted over [N E E] comparisons, ties going to the
  first index as argmax does, in about 10 kernels a layer.
- Above 2^24 comparisons (a prefill chunk of more than 256 tokens) it falls
  back to the picks.
- The indices and values are identical: no draws change.
- Not taken:
  - a gen.array top-k op, which needs an API and conformance work (and MLX's
    topk does not break ties as argmax does);
  - top-k on logits, skipping the 256-way softmax, which moves draws;
  - ranking always, which costs N·E²·4 bytes.
- Tests: qwen35-test's top-k-by-ranks-is-top-k-by-picks covers probabilities
  and 3-level ties, k up to E, and the fallback. Planted bugs: ties broken
  last-first fails 5 checks, `>=` for `>` fails 20, values read at the next
  index fails 10. On fenrir's CUDA the Qwen3.5 suites and qwen35-llama-test
  on the real MoE pass (55 tests, 1,594 assertions).

**Measured, tokens/s, float16, the same GGUF:**

| | before | after | llama.cpp |
|---|---:|---:|---:|
| batch 1 (compiled static) | 47.9 | 79.6 | 210 (llama-bench tg64: 278) |
| 16 rows (compiled static) | 488 | 612 | 982 |
| 16 rows, continuous engine | 314.5 | 377.6 | 1015 |
| 8 clients, continuous (6 warm runs) | 124.5 | 149.6 | 160.6 |
| 8-client median read (ms) | 45–47 | 36–38 | 45.6 |

- The batch-1 gap is now 2.6×, the 16-row gap 1.6×.
- With 8 clients the median read is now faster than llama.cpp's; throughput
  still trails because of stalls. Mixed forwards retrace per new input shape
  (200 to 430 ms each on the MoE), still after 6 warm runs. The varying part
  is probably the block tables' widths, not confirmed.

**Left, the owner's call:**
- **Batch-1 gathers on qmm_sm80.** Splitting into chunks of 4 or fewer pairs
  cuts gate/up from 2.29 to 1.40 ms, but is slower at 8 and 16 rows and adds
  kernels to a host-bound step; not taken. Alternatives: an MLX patch so one
  row a pair always takes gather_qmv; a CUDA kernel of ours; gate and up
  stacked at load, which touches LoRA, the checkpoint loader and the planted
  mutations.
- **The batch-1 floor is MLX's encoding,** about 2,500 kernels at 5 µs each.
  Remaining: fewer primitives in the generic Qwen3.5 layers (about 180 copies
  and 156 aranges a step; collapsing the aranges measured neutral), or
  upstream MLX work.
- **Continuous-mode retraces:** pre-trace the mixed signatures at engine
  start, bucket table widths and row counts, or accept the stalls. prefill
  (genclj-6qb6) raised the same question.
- **float16 by default for quantized models on CUDA.** The engine computes
  in float32 by default there.
- Not proposed: smaller products in chunks of 7 rows or fewer, which helps
  only the shared expert at 8 rows (2.70 → 1.75 ms) and hurts the dense Q8_0
  products.

**Checked on fenrir after the merge** (b4bf4b0, with prefill's runs; the
lead's verification run, 2026-11-08 ~06:20): every suite passes on MLX's
CUDA:
- the paged tests;
- the six array-engine namespaces;
- smc, lm, qwen3, qwen35 and qwen35-static;
- the llama.cpp comparisons on Qwen3-0.6B, Qwen3.5-0.8B and the Qwen3.6 MoE,
  with none skipped.
The JVM battery under C2 there passes too (2,418 tests).
