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
   the *same* handler transitions, parameterized by `:batch-shape` (`[]` for one
   trace, `[N]` for N particles), never as copies (GenMLX has six).
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
  (`docs/porting-genmlx.md`).
- **House style: gen.array's operators are referred** in place of
  `clojure.core`'s (`(+ a b)`, not `(a/add a b)`); `max`, `min`, `take` and
  `flatten` are never referred (`docs/array-api.md` §2, §12).
- **Branching on array values:** the `gen` macro wraps every `if` test in
  `a/truthy?`, so a one-element array branches on its value and a batched
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
  error. Gradients are values: `gen.gradients/param-gradients` returns a map,
  and optimizers return a new store. This is Gen.jl's `@param`,
  `init_param!` and `accumulate_param_gradients!`/`apply!`, but functional.
- **Batched execution** (decided 2026-10-18 by the agent, for the owner's
  review): `gen.batched` holds N particles as one batched trace, run through
  the same transitions with `:batch-shape [N]`; generative functions that call
  others go through one dispatch, `gfi/run-op`. Constraints given to
  `gen.batched` are shared by every particle (`:shared-constraints?`): shape
  alone cannot tell a shared value from a per-particle one. A model that
  branches on per-particle values falls back to one scalar trace per particle,
  so batching never changes meaning. Measured: 14–26× faster importance
  sampling on the JVM backend (100–10,000 particles).
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
- **Git:** the main repo is public-ready, with `.beans/`, `.beans.yml`,
  `.claude/` and `*.pdf` gitignored. `.beans/` is its own private repo
  (`robert-johansson/gen.clj-beans`), synced by SessionStart/PreCompact/Stop
  hooks (see `.beans/hooks/README.txt`).

## 10. Deferred (tracked, not forgotten)

`beans list --tag deferred-port` lists every GenMLX area left out of M1–M4, each
with its location in GenMLX, a quality assessment, and how to port it cleanly.
`beans list --tag watch` lists external technologies to re-evaluate (TornadoVM,
jitllm, Raster).
