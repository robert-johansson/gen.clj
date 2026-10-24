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

**M4 is complete (2026-10-21).** Every feature of Gen.jl's checklist that
gen.clj targets is in (`docs/research/genjl-parity.md` §1), and every law the
thesis states for them is a test (§3, after a final law audit and a
confirmation read of the new tests). It was verified on the JVM (main: 818
tests, 4,016 assertions), on Metal (the full battery, 790 tests, and every
namespace changed after it), and on MLX's CPU in float32 and float64 (751
tests each). Reviews 5 to 7 and the confirmation read found the last bugs
(fixed, each with a test). Benchmarks of the static language's compiled
scores are in `docs/benchmarks.md`. The night's owner decisions are in §9.
Next: M5 (Gen.jl's own suite; its coverage table is
`docs/research/genjl-test-suite.md`) and M3's remainder.

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
4. **Semantics never reads mutable state that changes meaning.** Such
   state is confined to three places:
   - the run-local `volatile!`s that thread one run's state: the two that
     `gen.dynamic/execute` threads through a model run (the transition and
     the state), and those of the kernel, transform, internal-proposal and
     enumeration runners;
   - the implicit key source in `gen.gfi`;
   - native handles inside the membrane.

   Hot paths have no atoms, production dispatch has no test hooks, and the
   semantics layer makes no memory-management calls (eval and cleanup
   happen only at inference-loop boundaries). The other mutable pieces are
   caches and registries that never change what an operation means:
   - L0: gen.array.protocols' lazy-backends flag, the batch backends'
     interning, gen.array.memory's registry, compile's trace cache, and
     MLX's constants, budget and CPU-compile probe;
   - L5: the compiled-step caches;
   - the LLM runtime's engines and caches, which are membrane state: the
     LM's cache, the slot engine's books, the llama engine's replay and
     smc's pool.

   (Restated by the owner, 2026-10-22, after the post-M5 review's I1 and
   I4: "mutation is confined to three places" had stopped holding.)
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
L5  inference             importance, mh, kernels, hmc/mala/nuts/map, particle filter, diagnostics
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
                           content-addressed, with a token-prefix trie; batching explicit)
Branching protocol         open · fork · advance · extend · score · resample · dispose!
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
- **As built (2026-10-20, genclj-kwxd; approved by the owner the same day,
  `docs/research/llm-branching.md`).** The binding is jextract's over
  llama.h v0.7.0 (`native/jextract-llama.sh`, sources committed, the
  library's version checked at load), and llama.cpp builds with
  `native/build-llama.sh`. `gen.llm.branching` is the Branching protocol
  (open, fork, advance, extend, score, resample, dispose!), `gen.llm.slots`
  runs any engine over sequence slots on a thread of its own, and
  `gen.llm.llama` is llama.cpp's. On Metal the same read from the same
  state is bit-identical, while different batch shapes give a token's
  log-probability differing by up to 5e-2 nats (median 2e-3 to 4e-3).
- **Quantized weights off Metal (measured 2026-10-21, fenrir; genclj-kw88).**
  With Q8_0 weights, llama.cpp's x86 CPU and CUDA builds give log-probabilities
  about 0.22 nats a token (median; up to about 2) from Metal's, which agrees
  with an f32 forward pass within 0.008. The CPU build is batch-invariant;
  CUDA's also varies with batch composition by about 0.2 (max about 1). Flash
  attention, CUDA graphs and cuBLAS-only matmuls change nothing; f16 weights
  bring both within about 0.01 of Metal, and CUDA's batch variation to 0.13 at
  most (likely cause, unproven: int8 dot products with quantized activations
  on those backends). So on CUDA and x86 CPU, exact inference (scores, SMC
  weights, comparisons across devices) wants float (F16/BF16) weights;
  quantized ones are for speed.
- **Float weights on CUDA (measured 2026-10-21, fenrir; genclj-kw88).** Two
  computations of one distribution differ by at most 0.13 (Qwen3-0.6B f16),
  0.15 (Qwen3.5-0.8B BF16) and 0.26 (Gemma-3-270M F16; 0.078 as F32,
  converted from the F16 file without loss); branches without a sliding
  window, past 512 tokens, by at most 0.11. Past Gemma's sliding window they
  differ by up to 1.3 nats as F16 and 0.68 as F32 (median 0.02 to 0.05;
  the worst, a fork at 1,000 tokens of the code text advanced, is the same
  case in both), where llama.cpp's x86 CPU build gives 0.25 and Metal 0.28.
  F16's half-precision matrix products account for half of it; the rest is
  a CUDA path past the window, not found yet (flash attention's kernel and
  the f16 KV cache are the suspects; watch bean genclj-gjiz).
- **As built (2026-10-20, genclj-mse0; approved under the owner's
  delegation, 2026-10-21; `docs/research/llm-lm-protocol.md`).**
  `gen.llm.lm` is the LM protocol (vocab, encode, decode, token-bytes,
  byte-vocab, next-logprobs, batch-next-logprobs, score) over any
  Branching engine. It is
  content-addressed: a trie of token prefixes (`gen.llm.trie`) records what
  the engine has read, each token with its score, and the cache
  (`gen.llm.cache`) owns every live branch, bounded in branches and in KV
  cells counted as llama.cpp's unified cache counts them (`gen.llm.cells`),
  evicting least-recently-used ones with `dispose!`. Batching is explicit
  (a call's one-token reads are one advance), with no autobatcher
  (recommended), and the same calls on a new engine give the same results,
  bit for bit.
- **As built (2026-10-21, genclj-0cvv; confirmed by the owner, 2026-10-21,
  `docs/research/llm-token-gf.md`).** `gen.llm.token/completion` is the
  token generative function over an LM: arguments `[prompt max-tokens]`,
  token i at address i, until an end token (kept) or max-tokens; every GFI
  operation, with each token's score from the LM's ledger (score once) and
  host draws by `token/dist` from `(fold-in key i)`. The GFI laws hold
  exactly on the toy engines, also on one whose reads differ with their
  shapes as llama.cpp's do.
- **As built (2026-10-21, genclj-loyp; confirmed by the owner, 2026-10-21,
  `docs/research/llm-token-smc.md`).** `gen.llm.potential` weighs token
  sequences after genlm-control's `Potential` (prefix, complete, the next
  token's weights over the vocabulary and the end). `gen.llm.smc` is token
  SMC over a completion, twisted by a potential, with
  gen.inference.particle-filter's interface: one batched LM read a step,
  bootstrap, masked or custom proposals, ESS-gated resampling, an
  estimate unbiased in exp space, particles convertible to the
  completion's traces.
- **As built (2026-10-21, genclj-q38h; confirmed by the owner, 2026-10-21,
  `docs/research/llm-grammar.md`).** Grammar-constrained decoding, on our
  own regex engine (no instaparse). `gen.llm.regex` builds regular
  languages over bytes as terms (characters as UTF-8; intersection and
  complement) and their minimal DFAs by Brzozowski derivatives
  (`gen.llm.dfa`); its regex syntax agrees with java.util.regex where it
  accepts. `gen.llm.grammar` reads a language through an LM's vocabulary
  with exact masks (every state its tokens reach explored; a token allowed
  only into a state tokens can complete), swept over a trie of the tokens'
  bytes (`gen.llm.bytes`), and is a potential (`gen.llm.potential`).
  `(token/completion m {:potential p})` is the completion under a potential:
  each token's distribution the LM's row times the potential's weights,
  renormalized at its site (`gen.llm.constrained`), whose GFI laws hold;
  `constrained/conditional-weight` is the importance weight of a
  completion's trace for the LM conditioned on the potential.
  `gen.llm.bytes` also gives a text's probability over its tokenizations.

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
| instaparse (official) | no (the owner, 2026-10-21: §9, grammar-constrained decoding) | our own regex → DFA engine (`gen.llm.regex`) |
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
- **The book is Markdown whose code runs** (owner, 2026-10-24; genclj-ceds).
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
  One book in five Parts (owner, 2026-10-24): I Foundations, II Inference,
  III Theory, IV The System, V Applications, and appendices, with Gen.jl's
  tutorials as worked chapters inside them, not a separate tutorial site.
  `gen.book pdf` builds it with Pandoc.
- **A 1.0 for dependents** (owner, 2026-10-24): memo.clj's 1.0 should
  probably wait for gen.clj's 1.0 rather than pin a commit. Until then
  dependents pin a commit. There is no rush for memo.clj: gen.clj's 1.0
  comes when its remaining work is done (owner, 2026-10-24). The 1.0's
  scope (which surface it freezes, modules first or not, the post-M5
  fixes, the book's Parts I–III first) is open: genclj-04h2.
- **Indices are 0-based** (Clojure idiom, as in GenMLX; D2): Map's and
  Unfold's addresses, Recurse's nodes (root 0), categorical's values
  (categorical-logits' too: 0 to n - 1) and Switch's branch index (branch 0
  is the first). Gen.jl is 1-based; document this in a migration note.
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
- **lgamma and digamma at their edges are SciPy's** (confirmed by the owner,
  2026-10-22; m3-grammar's cross-review, genclj-9gyi): lgamma(+Inf) and
  digamma(+Inf) are +Inf, as in SciPy and Julia, where gen.array gave NaN (the
  composite and its bit-identical host path). lgamma(−Inf) is −Inf (|Γ| goes
  to 0 between the poles), where the pole branch gave +Inf; digamma(−0.0) is
  +Inf, the limit from below, keeping the sign of a zero, where `(+ z 0.0)`
  made it −Inf.
- **A draw beyond the float range is +Inf** (owner, 2026-10-20), with log
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
- **A log mass of 0 has gradient 0, not NaN** (confirmed by the owner,
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
- **Probabilities sum to 1 within their dtype** (confirmed by the owner,
  2026-10-22; m3-lm, genclj-ggdl). check-probabilities (also the mixtures'
  weights) and dirichlet's support bound |Σp − 1| by 4·log2(K)·ε of the
  values' dtype, never below 1e-6. That's room for two blocked or pairwise
  sums: one that normalized the values, one that checks them (genclj-y0mp
  measured each within 0.8·log2(K)·ε). The fixed 1e-6 refused float32
  probabilities at K = 10⁴; float32 there now gets 6.3e-6, and float64 keeps
  1e-6. The alternative was K·ε, which would accept data normalized by a naive
  sequential float32 sum but is 0.12 at K = 10⁶ in float32, so it would barely
  check. Under log2(K) such data is a loud error.
- **A mixture of a discrete and a continuous family is an error** when it
  is made (owner, 2026-10-22, the post-M5 review's P2-02, genclj-e6b0). Its
  density would add a pmf to a pdf: given y = 0, the posterior came out
  0.267 where 0.80 is exact. Gen.jl's types forbid it.
- **Differences from Gen.jl's test suite** (owner, 2026-10-20, after the
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
- **Law results** (2026-10-24, genclj-4tjd, the owner's go). A law in
  `gen.laws` returns a test.check result, not a boolean: its `checks` fail
  naming the check and the values it compared, and a case the law says
  nothing about (constraints another run made that this one cannot visit)
  is `vacuous`. `deflaws` counts the vacuous cases and fails a law with
  more than 60% of them (`laws/max-vacuous`: the highest share measured
  was the fuzz's 32%). Seeds span 2^31 (`laws/seeds`; `gen/nat` drew 0–99
  only), and the fuzz's random models take an argument with alternatives,
  their label the spec.
- **The test loop** (2026-10-24, genclj-elyc, the owner's go: a 7-minute
  battery is not run after each change). A REPL helper, `(t)` in
  `dev/user.clj` (`-M:test:dev`), reloads what changed with tools.namespace
  (a `:dev` dependency only) and runs the tests the reload reached, without
  `^:slow`. `gen.parallel` (`-X:test:par`, with any tier) runs the test
  namespaces in 5 JVMs of 2 cores each, sharded by recorded times: the
  fast tier 87 → ~50 s, the battery 408 → ~150 s on the M4. It refuses
  MLX and llama.cpp runs (one GPU JVM at a time). Not taken: fewer cases
  per property (less power), tests in parallel threads of one JVM
  (`with-redefs` in nine files), C1-only JIT (the fast tier took 200 s).
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
- **A diverging HMC trajectory is rejected** (owner, 2026-10-22, the
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
- **Trace kernels, GenTraceKernelDSL** (M4, 2026-10-20): no new macro. A
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
- **Change hints and Diffed propagation** (Gen.jl's IntDiff, SetDiff,
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
- **The Branching protocol and the llama.cpp engine** (2026-10-20, agent;
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
- **The LM protocol** (2026-10-20, agent; approved by the owner the same
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
- **The token generative function's foundations** (owner, 2026-10-20,
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
- **The token generative function** (2026-10-21, agent; confirmed by the
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
- **Token SMC and potentials** (2026-10-21, agent; confirmed by the owner,
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
- **A hybrid model's steps split over its recurrent cells** (2026-10-21,
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
    only where the model's weights are on Metal. On CUDA (fenrir's RTX PRO
    6000, shared) a split step of Qwen3.5-0.8B cost about twice one decode
    (52 against 24 ms), an extra decode costing more there than the copies
    it saves; genclj-mxi9 measures CUDA and the CPU properly. The property
    still says otherwise either way.
  - Names for the owner: the property `gen.llama.split-recurrent` (default
    by device, above; false keeps one decode a step, exactly as before), the info key
    `:split-recurrent?`, the diagnostics `llama/step-splits`, the namespace
    `gen.llm.llama.recurrent`. An upstream llama.cpp issue (copy only the
    states that moved) is drafted for the owner, not filed (genclj-g5gx).
- **The shared memory budget** (2026-10-21, agent; confirmed by the owner,
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
- **Grammar-constrained decoding** (2026-10-21, agent; confirmed by the
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
- **Compiled functions across threads (owner, 2026-10-21; genclj-hfwo).**
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
- **Metal's buffer limit: collect and retry once (owner, 2026-10-21;
  genclj-9tyh).** MLX on macOS allocates every buffer through Metal, which
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
- **An optimizer step evaluates what it returns (genclj-9e9e).**
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
- **A vector is a path, everywhere (owner, 2026-10-21; M5's G9 and F3).**
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
- **Parameter gradients through a combinator's trace (owner, 2026-10-21;
  from M5's map.jl port).** `param-gradients` on a combinator's trace (Map,
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
- **A custom function's trainable parameters (M5's G5, genclj-2294;
  approved under the owner's delegation, 2026-10-21).** `(custom/deterministic
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
- **A call's trace and return value by address; merge and Recurse's
  numbering (owner, 2026-10-21; M5's G1, F3, G10, G13; m3-grammar's
  design).**
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
- **Lookups are loud (M5's D4, genclj-ocwu; approved under the owner's
  delegation, 2026-10-21).** One rule: a lookup throws where Gen.jl throws
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
- **Filling with an array is loud (owner, 2026-10-21; M5's G14).**
  `array/full` and `full-like` take a scalar fill value (a host number or a
  0-d array, tracked on the tape too); an array of values is an ex-info on
  every backend, pointing to `broadcast-to`. Before, the JVM threw a
  ClassCastException and MLX broadcast it silently.
- **llama.cpp's band by device (owner, 2026-10-21; genclj-kw88).** The
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
- **Variational inference trains the model's parameters too (M5's G4;
  m3-grammar's design).** `vi/black-box-vi` and `black-box-vimco` take
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
- **Git:** the main repo is public-ready, with `.beans/`, `.beans.yml`,
  `.claude/` and `*.pdf` gitignored. Its remote is the private
  `robert-johansson/gen.clj` (the owner, 2026-10-21, at M4's close: private
  until gen.clj has a license); main is pushed at a milestone's close or with
  the owner's go-ahead. `.beans/` is its own private repo
  (`robert-johansson/gen.clj-beans`), synced by SessionStart/PreCompact/Stop
  hooks (see `.beans/hooks/README.txt`).
- **GenMLX extras, round 1 (owner, 2026-10-23; epic genclj-0qds).** Twelve
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
  - **MCMC diagnostics (genclj-hfca; PROVISIONAL, owner to confirm).**
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
  - **Serialization (genclj-unjc; PROVISIONAL, owner to confirm).**
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
      bit, and its weight against its score (no choice missing).
    - Traces with encapsulated randomness are not serializable yet (the
      lead's call): ω is kept per call, generate draws it afresh and
      fold-in is one-way, so write-trace refuses an ω anywhere. The hook
      (`:omegas` for generate, an ICalls `-calls`) is genclj-t8uj.
    - `gen.array/backend-name` is public, for the recorded numerics.
  - **NUTS and Stan's warmup (genclj-tvc0).** `mcmc/nuts` is one
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
    PROVISIONAL (owner to confirm):
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
      dual averaging adapts the nominal step (Stan's choice; PROVISIONAL;
      the alternative, no jitter in the warmup, leaves the adapted step
      blind to the jitter's effect on acceptance).
  - **GenMLX's further distributions (genclj-z2gh).** `truncated-normal`,
    `von-mises`, `wishart` and `inv-wishart` take Distributions.jl's
    parameters; `wrapped-cauchy` (scipy's wrapcauchy, centered on mu) and
    `wrapped-normal` have none there. Each is exact at its extremes: the
    truncated normal in log Φ (draws by a log-space inverse CDF, exact on
    [50σ, 51σ]); von Mises draws by Best and Fisher's sampler rewritten in
    1 − ρ and κ(r − 1), exact to κ = 1e300; the wrapped normal's density
    by two series omitting less than 1e-17; Wishart draws by Bartlett's
    decomposition, the inverse Wishart's by a triangular solve.
    mvnormal's covariance check is shared as `check-covariance`.
    PROVISIONAL (owner to confirm):
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

  - **Per-particle constraints and batched enumeration (genclj-7oc1;
    PROVISIONAL, owner to confirm).** `gen.batched/generate` and `update`
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
- **Gradients under the batch tracer (genclj-7dti; PROVISIONAL, owner to
  confirm).** `array/value-and-grad` of batched arguments, or anywhere in a
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
  - **Switches by partition (genclj-7tqw; PROVISIONAL where marked).** As
    the owner decided, a Switch whose index differs across a batch's
    particles runs each branch batched on its own particles
    (`gen.partition`: gather, join, scatter). Each part's backend draws the
    whole batch's draws at its rows (PROVISIONAL; a site in G groups draws
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
    batched; inside it they go per particle. Loud (PROVISIONAL): return
    values that cannot be one batched value (shape, dtype or structure
    differing across branches, int against float); a partitioned trace
    revised by another generative function; a selection under choices
    held by part; constraints by part whose particles span groups. A
    partition needs a one-axis batch. Measured: importance sampling on a
    mixture of structurally different components 3.8× (100 particles) to
    21.5× (10,000) faster than one particle at a time, an MH sweep of 1000
    chains between components 6.4×.
  - **Scan, Mix and dimap (genclj-2gma; PROVISIONAL, owner to confirm).**
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
  - **Running totals never NaN (genclj-awd0, a bug the static fuzzer
    found; the remedy PROVISIONAL).** Map's, Unfold's and Scan's scores are
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
  - **Conjugacy, and Kalman and HMM generative functions (genclj-702q;
    PROVISIONAL, owner to confirm).** Chosen by the user, never detected
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
  - **Laplace and Fisher information (genclj-xg1j; PROVISIONAL, owner to
    confirm).** `gen.inference.laplace`: Hessians of a trace's log
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
  - **Particle MCMC (genclj-gcus; PROVISIONAL, owner to confirm).**
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
  - **The path-equivalence fuzzer's remainder (genclj-y0ij).** GenMLX's
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
      (PROVISIONAL: a host integer, or an array of an :int dtype kind; not
      an integral-valued float, which would be exact only below 2^53 and
      need a host read of an array); otherwise unknown-change.
    - **Map's and Unfold's scores turned NaN** (−Inf + Inf) when an update
      returned an element to its support (genclj-awd0, fixed in wave 2).

## 10. Deferred (tracked, not forgotten)

`beans list --tag deferred-port` lists every GenMLX area left out of M1–M4, each
with its location in GenMLX, a quality assessment, and how to port it cleanly.
`beans list --tag watch` lists external technologies to re-evaluate (TornadoVM,
jitllm, Raster).
