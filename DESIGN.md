# gen.clj — Design

*2026-10-18. The decisions below were confirmed by the owner on 2026-10-18;
each has a completed decision bean (`beans list --tag decision`). Evidence and
sources live in `docs/research/`.*

## 1. Goal

Port GenMLX (`~/code/genmlx`, ClojureScript on nbb + MLX via mlx-node) to
Clojure on the JVM. In order:

1. **M1 — the clean core runs on the JVM, CPU-only, on a pure-JVM backend** (no
   native dependencies; nothing but a JDK).
2. **M2 — an MLX backend for the GPU**: Metal on macOS and CUDA on Linux, from
   the *same* Clojure code.
3. **M3 — local LLMs as generative functions** (what mlx-node gives GenMLX today).
4. **M4 — Gen.jl parity.** The long-term bar: gen.clj is exactly as capable as Gen.jl.
   Its exit criterion includes **complete GFI law coverage**: every law the
   Cusumano-Towner thesis states for the features gen.clj has is a test
   (decided with the owner 2026-10-18). Each milestone tests the laws of the
   features it builds; a coverage matrix in `docs/research/genjl-parity.md` §3
   tracks every law.
5. **M5 — Gen.jl's own test suite passes** (the owner, 2026-10-20: after M4
   closes). Gen.jl 0.4.8's tests (38 files, about 1,900 assertions) are
   ported file by file to gen.clj tests, each citing its source. A table
   marks each as ported or not applicable, with a reason (Julia-only
   machinery). Gen.jl itself, installed on this Mac (Gen 0.4.8 on Julia
   1.12.6; its suite passes here, `docs/research/genjl-test-suite.md`), is
   the oracle for expected values, written as EDN fixtures as scipy's and
   JAX's are, so the tests never need Julia. Its seeded statistical tests
   are compared by their statistics, as for GenMLX. A port that fails is a
   gap to fix, or a documented difference.
6. **M6 — gen.clj inside kmet** and **7. M7 — inference over agents**
   (drafts, the owner, 2026-10-20; `docs/research/kmet-integration.md`).
   kmet is a Clojure port of the pi coding agent, which the owner's fork
   brings to the JVM. M6 serves kmet from a local model through gen.clj,
   with sessions as traces (scored and edited) and tool-call grammars. M7
   treats the agent as a generative function: SMC over trajectories with
   verifiers, inverse planning, and the LLM proposing programs through
   `run_code`. Both come after M3's token generative function. Neither core
   depends on the other; the integration is a third artifact.

**Status (2026-10-19): M1 is complete.** The pure-JVM backend runs every
layer: arrays with reverse-mode AD, the data algebra, the GFI and dynamic
language, the distributions, Map/Unfold/Switch, batched execution and the
inference library. It is checked by the law suite (audited against the
thesis, and fuzzed over random programs), by exact-answer statistical tests
and by reference values. Three code reviews and four black-box acceptance
passes found the remaining bugs, nearly all in batched execution, and they
are fixed (`docs/research/acceptance-2026-10-19.md`). Benchmarks are in
`docs/benchmarks.md`.

**M2 (2026-10-20): the MLX backend runs on Metal, on MLX's CPU backend
and on CUDA (the Jetson AGX Thor, sm_110).** It passes the backend
conformance suite against the JVM reference on every device, and the full
battery (298 tests, slow and heavy included) on Metal, with the tape and
with native gradients, on CUDA, and on MLX's CPU backend: float32 on the
Mac and the Thor, float64 split between them (the Thor's first 15
namespaces, the Mac's other 9). M2 was first closed on MLX's CPU light
tier and reopened the same morning to finish these. Every property passed a soak (ten times its
cases, fresh seeds) on each device after the soaks found one wrong test
assumption (only Metal flushes float32 subnormals). The record is
`docs/research/m2-verification-2026-10-20.md`. CUDA needs MLX v0.32.2
built from source for sm_110, unpatched. Benchmarks, within gen.clj by the
owner's choice, are in `docs/benchmarks.md`. The RTX PRO 6000 (x86_64,
sm_120) comes later (genclj-kw88); native autodiff stays opt-in
(genclj-fuvr).

**M4 (2026-10-20): Gen.jl parity, but for the static modeling language and
encapsulated randomness,** decided by the owner on 2026-10-20 (§9) and
under way (genclj-1ogo, then genclj-0q3a and the change hints of
genclj-vghe).
Added: call_at and choice_at, the internal-proposal override, custom
deterministic generative functions with custom gradients
(`array/custom-vjp`), `@dist`, `@kern`, wake-sleep's lecture, pair and
unpair, GenParticleFilters' operations, black-box VI and VIMCO, the trace
transform DSL with its translators, SMCP3, GenTraceKernelDSL's trace
kernels, and Recurse. A code review and an interim audit of the laws
against the thesis, the same night, found bugs in gen.clj (fixed, each
with a test that fails without the fix) and four upstream: Gen.jl's
general translator adds the forward proposal's score, its Recurse update
weighs new nodes by their scores, GenParticleFilters' optimal resize
biases the estimate, and GenTraceKernelDSL's Jacobian trimming hides
non-bijections (§9).

Only the beautiful core is ported first. Everything else in GenMLX that *could*
be ported is tracked as a deferred bean with a note on how to port it cleanly
(`beans list --tag deferred-port`).

## 2. What we are porting — findings

(Details: `docs/research/genmlx-quality-triage.md`, `docs/research/genmlx-porting-surface.md`.)

- GenMLX is ~62k LOC of source and ~118k LOC of tests. Its **mathematical core is
  ~3–4k LOC and clean**: choicemap, selection, trace, diff, protocols, the scalar
  handler transitions, runtime, dist/core + `defdist`, dispatch, kernel algebra,
  importance sampling, the MH family, trace translators, and the GFI law catalog.
  GenMLX's own audit (genmlx-pif1) found zero formula errors in ~40 derivations:
  *"The mathematics is not the risk surface; the duplication is."*
- **~25k LOC are optimization paths that re-implement the same semantics** (L1/L2
  compiled paths, L3 analytical elimination, fused/compiled/vectorized MCMC
  variants, six batched copies of the scalar transitions, triple-path
  combinators). That is where most of GenMLX's ~150 core bug fixes came from.
- **~18k LOC are research verticals** (llm, agents, control, world, program synthesis).
- `dynamic.cljs` requires the whole compile + L3 stack, a transitive closure of
  27 namespaces and 16.6k LOC, because every `gen` call runs schema
  extraction, compilation and conjugacy analysis. **DynamicGF is rewritten
  (~400 LOC), not ported.**
- The core is synchronous and pure Clojure. Outside the MLX membrane, JS interop
  is trivial (`js/Math` → `clojure.math`, `catch :default` → `catch Exception`,
  `#js` runtime object → `deftype`).

## 3. Principles

1. **The GFI is the contract; the handler is the semantics.** Every optimization
   is a plug-in that must be observationally equivalent to the handler. None of
   them may change what a GFI operation *means*.
   - Corollary: no "generate returns the posterior mean with a marginal weight".
     Conjugacy is either the thesis's *optimal internal proposal* (sample the
     latent from its exact conditional, with weight log p̄(σ); thesis §4.1,
     pp. 145–146) or an explicitly separate marginalized generative function.
2. **Gen.jl-shaped GFI from day one** (decided):
   `update(trace, args, argdiffs, constraints) → {:trace :weight :retdiff :discard}` and
   `regenerate(trace, args, argdiffs, selection) → {:trace :weight :retdiff}`.
   Retdiffs drive incremental computation in Map and Unfold. GenMLX's three
   update protocols collapse into one.
3. **One implementation per semantic.** Batched execution is part of M1 (decided):
   the *same* code, run under the batch tracer (`gen.array.batch`, vmap
   semantics: docs/array-api.md §8b), never as copies (GenMLX has six).
4. **Mutation is confined to three places**: the single `volatile!` that
   `gen.dynamic/execute` threads through one model run, the implicit key
   source in `gen.gfi`, and native handles inside the membrane. Hot paths have
   no atoms, production dispatch has no test hooks, and the semantics layer
   makes no memory-management calls (eval and cleanup happen only at
   inference-loop boundaries).
5. **Randomness is purely functional** (decided). Every GFI op is a pure
   function of its inputs plus an explicit PRNG key: the same key gives the same
   result. Keys are threaded through handler state, never stored in metadata,
   and never silently replaced by fresh keys. The only impure point is the
   outermost API: omitting the key draws one from the implicit key source,
   whose root seed is random for each JVM (`gfi/implicit-seed` reads it,
   `gfi/seed!` fixes it). Every trace site consumes one key whether it samples
   or is constrained, so constraining a choice never shifts the randomness of
   another site.
6. **Minimal dependencies.** The core depends on Clojure plus the native
   membrane, nothing else (see §8).
7. **Laws are tests.** The thesis's GFI invariants become test.check properties,
   one suite per feature (`docs/research/genjl-parity.md` §3 lists 20). Gen.jl
   and Gen.clj reference values become example tests.
8. **No aggregator namespaces; keep dependency closures small.**

## 4. Architecture

```
modules (optional deps)   gen.llm.*  …later extras (deferred beans)
L5  inference             importance, mh, kernels, hmc/mala/map, particle filter
L4  combinators           map, unfold, switch (recurse in M4)
L3  distributions         Distribution record, open multimethods, defdist
L2  GFI                   protocols · handler transitions · runtime · dynamic (gen macro) · dispatch seam
L1  data algebra          choicemap · selection · trace · diff (argdiffs + retdiffs)
L0  array API             backend-neutral ops, grad, PRNG keys
      ├─ JVM backend      pure Clojure: double[] arrays, tape AD, Threefry          (M1)
      └─ MLX backend      generated Java statics (long handles) → libmlxc + libmlx  (M2)
```

Everything above L0 calls only the array API. No backend type escapes L0
except the opaque array value. The same model and inference code runs on either
backend.

## 5. Numerics: one array API, two backends

(Details: `docs/research/numerics-backends.md`.)

### 5.1 The array API (L0)

The core is written against one backend-neutral API. Its semantics follow
GenMLX's `mx/` contract, so GenMLX code ports with renames:
- Immutable values.
- Broadcasting ops.
- `grad` / `value-and-grad`.
- Splittable PRNG keys.
- `eval!` / `materialize!` are timing-only: a no-op on eager backends, and the
  core never relies on laziness.

The surface starts from the ops the core actually uses, not GenMLX's full
245-function membrane. The full specification is in **`docs/array-api.md`**:
- Clojure numbers are backend-neutral 0-d arrays (host math).
- Ops are keywords.
- Composite ops are defined once over primitives.
- Keys are values.
- Gradients come from a generic tape.

The backend contract is `src/gen/array/protocols.clj`.

### 5.2 JVM backend (M1): pure Clojure, CPU-only, no native dependencies

- **Arrays:** a deftype over a contiguous `double[]`, with primitive-hinted
  loops and broadcasting by index arithmetic. Eager. **float64 by default**,
  which makes it an exact reference for testing the MLX backend.
- **Autodiff:** a small tape-based reverse mode. A tracked array type
  implements the same protocol and records one VJP rule per op. Most of the
  core (simulate/generate/update/regenerate/assess, IS, MH, particle filtering)
  needs **no AD at all**. AD is needed only for HMC, MALA, MAP, VI, choice
  gradients and translator Jacobians.
- **Linear algebra:** hand-written Cholesky and triangular solves (EJML only if
  LU, QR or eig become necessary).
- **Special functions:** lgamma, digamma, erf, erfinv, Bessel.
- **PRNG:** Threefry2x32 (as MLX and JAX). Stretch goal: bit-compatible with
  MLX, so both backends produce identical traces for the same key.
- **Performance:** for scalar-heavy models (most PPL models) it is likely
  *faster* than MLX, with no FFI and no ~tens-of-µs eval floor. Large batched
  workloads are what the GPU is for.

### 5.3 MLX backend (M2): mlx-c via Panama FFM

**Decision.** Bind MLX's official C API (mlx-c v0.7.0, which pins MLX v0.32.2)
with Java's Foreign Function & Memory API (final since JDK 22). This is the only
option that gives, from **one** API: CPU (macOS arm64 and Linux x86_64/arm64),
Metal, CUDA, grad/vjp/jvp/custom_function, vmap, compile, splittable PRNG keys,
linalg and quantized matmul.

- **Binding design** (lessons from the Feb-2026 Coffi experiment in
  `~/code/old.Gen.clj/src/gen/mlx/`, which used `invokeWithArguments`, a
  per-call `Arena`, and Clojure maps as handles):
  - Generate Java static wrappers with `invokeExact`.
  - Lower every `struct { void* ctx; }` handle to `long` (one-pointer structs
    use the pointer ABI on AArch64 and SysV x86-64; covered by ABI tests).
  - Reuse a zeroed per-thread scratch for out-parameters.
  - Install an error-handler upcall → `MlxException` (mlx-c's default handler
    calls `exit(-1)`).
  - Use one global closure upcall stub plus a payload-id registry of IFns,
    released through the dtor callback.
  - Free arrays through scopes (tidy-like) with escape marking, plus a
    `Cleaner` backstop that queues frees for the owning thread.
  - Run each MLX context on one long-lived platform thread, because streams are
    thread-affine.
  - Expected binding cost is ~10–30 ns per call. That is well below MLX's own
    ~1 µs per graph node and ~67 µs eval floor, so no C shim is needed.
- **Linalg decompositions run on the CPU stream** (Metal reports them NYI; CUDA
  has only Cholesky). GenMLX already routes them that way.
- **GenMLX-specific C++ composites** (lgamma, digamma, Bessel i0e/i1e,
  searchsorted) get re-expressed as composite mx ops.
- **Rejected as the backend:**
  - TornadoVM: no autodiff, vmap or array-graph model; kernels must be Java
    static methods. Watch-listed.
  - DJL / libtorch: no vmap or compile from Java; JNI on every op.
  - Neanderthal / Deep Diamond: no general autodiff; no Metal.
  - ND4J: stalled.
  - Babylon HAT: not shipped.
- **Both backends pass one shared conformance suite.** In it, the float64 JVM
  backend is the reference for the float32 MLX backend.
- **As built (2026-10-19; `docs/array-api.md` §13):**
  - The binding is generated from the mlx-c headers (`native/bindgen.clj`)
    over a hand-written runtime, and libmlxc is built against the official
    MLX wheel (`native/build-mlx.sh`).
  - Measured costs, on Graal and C2 alike: a downcall 2.4–3.8 ns, an op
    about 170 ns, evaluation about 11 µs (CPU) and 115 µs (GPU). The binding
    is not the cost.
  - Streams are shareable, with one evaluation lock, and arrays are freed
    through phantom references that the threads making arrays drain
    (`MlxRuntime/adopt`; a `Cleaner` came first and did not keep up under
    load). These replace the planned thread per context and queued frees
    (§9).
  - Uniform draws are bit-identical to the JVM backend's for a key.
  - Where MLX's semantics differ from NumPy's, the backend follows NumPy's.
    The exceptions are documented: Metal flushes subnormals to zero, and
    array indices are clamped.

## 6. Local LLMs: a protocol stack, llama.cpp first

(Details: `docs/research/llm-backends.md`.)

```
LLM generative functions   token GF · grammar middleware · token SMC · structured output
LM protocol                vocab/encode/decode · next-logprobs · batch-next-logprobs · score   (genlm-shaped,
                           content-addressed, with a token-prefix trie + autobatcher)
Branching protocol         open · fork · advance-batch · dispose · teacher-forced score · resample
engines                    llama.cpp (llama.h via our own filtered jextract FFM)  |  MLX-native forwards (later)
```

- **Engine 1: llama.cpp.**
  - Platforms: Metal, CUDA and CPU.
  - Primitives: full logits via `llama_get_logits_ith`; cheap forks via
    `llama_memory_seq_cp` (metadata-only with `kv_unified`); multi-sequence
    batched decode.
  - Models: ~150 architectures, including Qwen3, Qwen3.5/3.6 (hybrid
    GatedDeltaNet, MoE), Qwen3-VL, Gemma 3 and 4.
  - Releases: semver since mid-2026 (v0.7.0 on 2026-10-15). We pin to those and
    regenerate bindings with a compile check.
  - Why it comes first: it keeps gen.clj a PPL rather than an LLM inference
    engine, and it does not depend on M2.
- **Engine 2: MLX-native.** Port GenMLX's own Qwen3/Qwen3.5 forward passes over
  the M1 membrane. That gives numeric parity with GenMLX, device-resident
  logits, immutable caches (forks are free, no sequence cap) and gradients for
  LoRA/SFT/GRPO.
- **Tokenizers.** llama.cpp uses the vocabulary embedded in the GGUF
  (`llama_token_to_piece` gives raw bytes). The MLX path uses
  `ai.djl.huggingface:tokenizers`. Chat templates come from the GGUF or Jinjava.
- **Not chosen:**
  - llamaj.cpp as a dependency: 16★, off Maven Central. Use it as a reference
    for the jextract recipe.
  - kherud java-llama.cpp: stale, no raw logits.
  - Jlama: no Qwen3.
  - jitllm (TornadoVM): no public logits or KV-fork API. Watch-listed.
  - HTTP servers: only top-k logprobs.
- **Open risk.** Loading llama.cpp's ggml-metal and MLX's Metal backend in one
  process. Spike bean.

## 7. Platform

- **JVM:** Oracle GraalVM 25 (JDK 25 LTS, Graal JIT); Clojure CLI 1.12.5;
  Apple M4 with 32 GB for development. Linux CUDA hosts as in GenMLX (Thor
  sm_110, RTX PRO 6000 sm_120).
- **FFM:** final in 25. Run with `--enable-native-access=ALL-UNNAMED`.
- **Benchmarks:** compare downcall cost under Graal JIT and C2
  (`-XX:-UseJVMCICompiler`) before optimizing anything.
- **Vector API:** still incubating in 25, and Graal's support is "initial". It
  could later speed up the JVM backend; M1 uses plain primitive loops.

## 8. Dependencies

| Library | Verdict | Scope |
|---|---|---|
| Clojure 1.12 | yes | core |
| test.check (official) | yes | `:test`. GFI laws as properties, run with clojure.test `defspec` |
| malli (official) | edges only | `gen.dev` instrumentation; LLM structured output (schema → grammar) |
| clojure.spec | no | Overlaps malli; specs aren't data; spec2 has been alpha for years |
| instaparse (official) | decide in M3 | LLM grammar module only |
| promesa | no | The core is synchronous; JVM tokenizers are synchronous; virtual threads cover concurrency |
| SCI | later | Sandboxed eval of LLM-generated code (program synthesis) |
| clojure.math (built in) | yes | Replaces GenMLX's ~370 `js/Math` calls |
| criterium | yes | `:bench` |
| Apache Commons Math 3 | extension | `:commons-math` (`ext/commons-math`): its distributions as gen.clj's, GenDistributions' role (M4, 2026-10-20) |

The nbb-compatibility forks GenMLX uses (malli, instaparse, test.check) are
**not needed** on the JVM.

## 9. Conventions (decided 2026-10-18)

- **Namespace root: `gen`** (`gen.choicemap`, `gen.dynamic`, …). The collision
  with probcomp/Gen.clj's `gen.*` is accepted, since the two are never on one
  classpath. Maven coordinates are decided at first publish.
- **Map/Unfold addresses are 0-based** (Clojure idiom, as in GenMLX). Gen.jl is
  1-based; document this in a migration note.
- **Distributions use Gen.jl names and parameterizations** (`normal(mu, std)`,
  `gamma(shape, scale)`, `categorical(probs)`), plus explicitly named variants
  where they matter numerically (e.g. `categorical-logits`). Each GenMLX name
  whose meaning changed has a safeguard that makes a naive port fail loudly
  (`docs/porting-genmlx.md`). Families Gen.jl does not have take
  Distributions.jl's parameters, which GenDistributions' users write
  (`pareto(alpha, theta)`, `triangular(a, b, c)`; M4, 2026-10-20).
- **House style: gen.array's operators are referred** in place of
  `clojure.core`'s (`(+ a b)`, not `(array/add a b)`); `max`, `min`, `take` and
  `flatten` are never referred (`docs/array-api.md` §2, §12).
- **Branching on array values:** the `gen` macro wraps every `if` test in
  `array/truthy?`, so a one-element array branches on its value and a batched
  array throws instead of silently taking one branch (`docs/array-api.md` §8a).
- **Randomness:** JAX's partitionable Threefry-2x32-20 in pure Clojure,
  bit-identical to JAX. Every trace site consumes one key. The implicit key
  source has a random root seed for each JVM (`gfi/seed!` fixes it). See §3.5.
- **Traces store each choice's log-probability** (as Gen.jl does), so update
  and regenerate weights use the true old values and `project` never re-runs
  the model.
- **Trainable parameters are an explicit store** (decided 2026-10-18 by the
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
    maker the generative function does not compute as. Particles given
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
    `with-params` and `with-transitions` take only dynamic generative
    functions, whose bodies read them.

  Gradients are values: `gen.gradients/param-gradients` returns a map,
  and optimizers return a new store. This is Gen.jl's `@param`,
  `init_param!` and `accumulate_param_gradients!`/`apply!`, but functional.
- **Batched execution** (decided 2026-10-18 by the agent, for the owner's
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
    filter's estimate) are checked first.

  What remains is probabilistic: in a plain helper, a `case` on a choice, a
  vector lookup keyed by one, or a raw `if` on a comparison also used
  elsewhere is found by the spot checks with high probability.
  `gen.dynamic/defn-guarded` gives helpers a body's guards, deterministically.
  Measured: 5–90× faster importance sampling on the JVM backend (100–10,000
  particles; docs/benchmarks.md).
- **Constraints are choice maps** (2026-10-19, after the second acceptance
  test): a plain Clojure map given as constraints, observations or choices
  is an error naming `cm/from-map`, since it would read as no constraints at
  all. `nil` is the empty choice map. Selections and argument vectors are
  checked the same way.
- **Bernoulli values are booleans,** as in Gen.jl: 0/1 (GenMLX's) is an
  error, since a body's `if` takes 0 as true while the density scores it as
  false. NaN parameters and NaN constrained values are errors too.
- **Casts to integers are Java's** (owner, 2026-10-20; genclj-kw88). On
  every backend, a float cast to an integer dtype truncates toward zero,
  saturates at the dtype's range (±Inf too) and takes NaN to 0. An integer
  cast to a narrower one wraps, as Java's `(int)` of a long does. Before
  this, the backends disagreed on NaN, ±Inf and out-of-range values. The
  JVM backend threw on ±Inf and did not clamp int32. MLX's GPU saturated
  but took NaN to MIN for int64. MLX's x86 CPU gave MIN for all of them
  (the M2 review, finding 11, confirmed on fenrir). A loud error would need
  a host read of every cast on a lazy backend.
- **Booleans in arithmetic count, as in Julia** (owner, 2026-10-20;
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
- **A draw beyond the float range is +Inf** (owner, 2026-10-20), with log
  density -Inf, exact for the value stored, as IEEE floats and Julia's
  Distributions in Float32 give. In float32 this is common at small shapes:
  about 9 % of weibull(0.01, 1)'s draws pass Float/MAX_VALUE, and Pareto's
  docstring already said so. A cap at the largest float would give finite
  densities at the price of a silent atom. A loud error would need a host
  read of each draw on a lazy backend, or a threshold on the parameters.
  The other edge stays as it is: a draw that underflows to 0, where a
  density can be +Inf, is floored at the dtype's smallest normal number.
- **Differences from Gen.jl's test suite** (owner, 2026-10-20, after the
  M5 inventory, `docs/research/genjl-test-suite.md` §4):
  - *Retdiffs are more precise (D1).* A dynamic function's retdiff is
    no-change when its return value is the previous one, where Gen.jl's
    dynamic language always says UnknownChange. So an unchanged Map or
    Unfold element is absent from a VectorDiff, and a splice turns an
    argdiff into no-change for an argument identical to the previous call's.
    This is sound, since a diff claims only what holds. The ports assert
    gen.clj's diffs, and that they are consistent (`diff/consistent?`).
  - *Lookups are loud (D4).* `cm/get-choice`, `cm/get-value` and
    `(tr addr)` at a missing or node address throw ex-info, as Gen.jl
    throws KeyError, where they returned nil (rule 6). Clojure's `get` on a
    choice map stays the nil-returning lookup. To be done in M5.
  - *Beta is exact at 0 and 1 (D6):* +Inf when α < 1 (or β < 1), where
    Gen.jl substitutes a finite value. As with draws beyond the float
    range, gen.clj keeps the exact value.
  - *No default positional arguments (G3).* Clojure has none. A rest
    parameter (`[x & [y z]]`) covers calls, update and regenerate with any
    number of arguments, and a trace's args are the ones given. A
    multi-arity `gen` stays possible if parity is wanted later.
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
- **Bodies run to completion:** lazy sequences in a body's return value are
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
- **Arrays are values:** JVM arrays compare by dtype, shape and (bitwise)
  elements, so traces and choice maps holding them compare with `=`.
- **Testing:** reference values come from scipy and JAX, generated once into
  committed EDN fixtures (`test/fixtures/`); the tests never need Python.
  Against GenMLX, deterministic quantities (log-densities, weights given fixed
  choices) must match exactly, and sampled quantities are compared
  statistically: the same seed cannot reproduce GenMLX's samples, because
  draws also depend on how keys are threaded through a run.
- **GFI signatures are Gen.jl-shaped** (§3.2). `generate` throws on
  constraints at unvisited addresses.
- **Dtypes:** the JVM backend is float64 (the reference); MLX is float32. Core
  code never assumes a dtype. Bool vs 0/1 is settled in the array-API design.
- **MLX is upstream, patched as little as possible** (owner, 2026-10-19).
  CPU and Metal use the official MLX wheel, unpatched. If a fix is ever
  needed, for CUDA first, it comes as a clean patch series on a pinned
  upstream tag (`native/patches/`): one concern per patch, rebased at each
  MLX bump, dropped once upstream has it. Never a long-lived fork.
  GenMLX's fork was audited (`docs/research/mlx-fork-audit.md`): none of its
  41 patches is needed for CPU or Metal.
- **The MLX backend** (2026-10-19, agent, for the owner's review;
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
- **Exact enumeration** (2026-10-19, moved early at the owner's request, for
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
- **CUDA on the Jetson AGX Thor (sm_110): MLX built from its source**
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
- **MLX's CPU compile, where the system's g++ cannot build MLX's kernels,
  falls back to the identity** (2026-10-19, found on the Jetson Thor: GCC 13
  rejects the header MLX's wheel ships). Probed once, with a warning; values
  are unchanged, only unfused. Not patched in MLX, and no g++ wrapper: a
  watch bean (genclj-hjzh) revisits it with newer wheels.
- **Pushforward distributions, Gen.jl's `@dist`** (M4, 2026-10-19):
  `(gen.dist.dsl/dist name [params] body)` expands into a `defdist`. The body
  makes exactly one draw (a call of a `defdist` constructor, which the macro
  finds by var metadata). A discrete draw's density sums the mass of the
  support values mapped to v, so any function works; a continuous draw's
  inverts the path to it through +, −, ×, ÷, negate, exp and log, with the
  log-Jacobians. Non-numeric labels wait for parameters that are not arrays.
  The name `dist` follows Gen.jl's macro (owner, 2026-10-20).
- **Distributions from a host library, GenDistributions'
  `DistributionsBacked`** (M4, 2026-10-20; decided by the agent, for the
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
- **The internal-proposal override** (thesis Alg. 16; M4, 2026-10-19):
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
  planted bug. Regenerate keeps every unselected choice of the trace, so a
  selected choice that decides structure cannot drop one: such a move is a
  loud error, as with `with-exact-proposal` (the review, 2026-10-20; kept
  by the owner the same day).
- **Custom gradients, `array/custom-vjp`** (M4, 2026-10-20; one of §8's
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
- **Trace transforms** (M4, 2026-10-20): `gen.inference.transform`, Gen.jl's
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
- **Kernels that read the trace, `(kern ^:trace [tr & args] …)`** (owner,
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
  loop's boundary). (Corrected after the confirmation read of the M4 tests,
  C1: the first check compared every construct with the final trace, which
  rejected the p. 102 kernel in its own order.)
- **Trace kernels, GenTraceKernelDSL** (M4, 2026-10-20): no new macro. A
  kernel is a `gen` function whose first argument is the model trace and
  whose return value is `[update backward]`; its choices are the auxiliary
  randomness, so propose and assess give the forward and backward scores.
  log |det J| reruns the kernel's assess with the trace's and the kernel's
  continuous choices substituted by tracked values, under `array/grad`.
  It leaves out only the trace's choices no output depends on:
  GenTraceKernelDSL also drops constant outputs and kernel choices no
  output uses, which hides moves that are not bijections (found by the law
  audit, 2026-10-20), so here they make the matrix non-square or singular,
  a loud error. A continuous choice the update keeps but an output depends
  on counts as an unchanged output. Calls of other generative functions
  return their values (GenTraceKernelDSL returns their traces).
- **Transform bodies are eager** (2026-10-20): a lazy call (for, map, …)
  holding a write, copy or tcall is an error when the transform is
  defined (use doseq, run! or mapv), as `gen` bodies run to completion.
  A discrete read returns a host value, so the body branches with a plain
  `if`: on MLX a sampled boolean is an array, always truthy, which sent a
  random walk down the merge branch (the full battery on Metal, the same
  night).
- **Recurse** (M4, 2026-10-20): `(comb/recurse production aggregation
  max-branch)` as Gen.jl's, with 0-based nodes (root 0, child j of node i
  at i·max-branch + j + 1), choices at `[i :production]` and
  `[i :aggregation]`, the production kernel returning `{:value v :children
  [inputs]}`, and one argument (no `root_idx`). An update's new nodes add
  their generate weights, where Gen.jl's code adds their scores, which is
  the same only when every new choice is constrained; regenerate is
  implemented too (Gen.jl has none).
- **Encapsulated randomness** (thesis §4.5; owner, 2026-10-20): a
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
- **The static modeling language** (Gen.jl's `@gen (static)`; owner,
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
  - **Not static:** a body that is not one `let`, an address computed
    from a value, a `trace` or `splice` inside an expression (a branch on
    a choice included), a splice without an address, an untraced draw, an
    address used twice or one under another: each a compile error that
    names the form.
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
      where a draw is an error anyway, none is split. On the JVM backend
      assess is 1.8–2.8× faster (10 and 100 observations); gradients and
      HMC gain about 1.25×, their cost being autodiff's, and on MLX a
      compiled step was one fused graph already (`array/compile` traced the
      handler), so there the gain is the trace, not the step.
    - *The lint:* `(static/could-be-static gf)` runs the flag's analysis
      over a dynamic gf's source in its namespace; `(static/lint ns …)`
      reports every generative function var. A call in return position,
      `(let [...] (trace :y …))`, is static too: a last node, its value
      returned.
- **Change hints and Diffed propagation** (Gen.jl's IntDiff, SetDiff,
  DictDiff and Diffed; M4, 2026-10-20): `gen.diff` adds `int-diff`,
  `set-diff` and `dict-diff` beside `vector-diff` (each collapsing to
  no-change when empty), and `consistent?`, whether a diff describes the
  change between two values, which the laws' retdiff soundness now uses
  for every kind. `(diff/propagate f args diffs)` is an open multimethod on
  the function's var giving the diff of a call, with rules for `count`,
  `nth`, `get`, `conj`, `assoc`, `+`, `-` (clojure.core's and gen.array's),
  `inc`, `dec` and `vec`; Gen.jl dispatches on a Diffed value type, which
  Clojure's core functions do not take, so here the rule belongs to the
  call. In a static body every name carries a diff: an argument's from its
  argdiff, a binding that is one call of a var on symbols and constants
  from its rule, a bare symbol's from its input, a splice's from its
  callee's retdiff, anything else no-change or unknown-change. A splice
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
- **gen.array's alias is `array` (owner, 2026-10-19):** `(array/shape x)`,
  `(array/item x)`, the namespace's last segment, as the Clojure style guide
  has it. It replaced `a`, which the owner found ugly (2026-10-18). The
  operators and math stay referred in place of `clojure.core`'s, the
  convention of Clojure's numeric libraries (`clojure.core.matrix.operators`,
  fastmath's `use-primitive-operators`), so a model reads
  `(+ (* slope x) intercept)`, never `(array/+ (array/* slope x) intercept)`.
- **Three ways to run, all first-class (owner, 2026-10-19).** The pure-JVM
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
- **Git:** the main repo is public-ready, with `.beans/`, `.beans.yml`,
  `.claude/` and `*.pdf` gitignored. `.beans/` is its own private repo
  (`robert-johansson/gen.clj-beans`), synced by SessionStart/PreCompact/Stop
  hooks (see `.beans/hooks/README.txt`).

## 10. Deferred (tracked, not forgotten)

`beans list --tag deferred-port` lists every GenMLX area left out of M1–M4, each
with its location in GenMLX, a quality assessment, and how to port it cleanly.
`beans list --tag watch` lists external technologies to re-evaluate (TornadoVM,
jitllm, Raster).
