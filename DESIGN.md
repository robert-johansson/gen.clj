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
   1.12.6, 0.4.9 since 2026-11-04 in the oracle's project; its suite passes
   here, `docs/research/genjl-test-suite.md`), is
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
modules                   gen.llm.* (the LM runtime; token, potential, token-SMC generative functions)
L5  inference             importance, mh, kernels, hmc/mala/nuts/map, particle filter, diagnostics,
                          enumerative, proposal, transforms (involutive MCMC, SMCP3), trace kernels,
                          vi, conjugate, laplace, particle MCMC; gradients, learning
L4  combinators           map, unfold, switch, recurse, call-at, choice-at, scan, mix, dimap;
                          custom (deterministic, custom gradients, estimated); state-space
L3  distributions         Distribution record, open multimethods, defdist, dist (pushforwards), host families
L2  GFI                   protocols · handler transitions · dynamic (gen macro, static language) ·
                          batched · partition · serialize · dispatch seam
L1  data algebra          choicemap · selection · trace · diff (argdiffs + retdiffs)
L0  array API             backend-neutral ops, grad, PRNG keys
      ├─ JVM backend      pure Clojure: double[] arrays, tape AD, Threefry          (M1)
      └─ MLX backend      generated Java statics (long handles) → libmlxc + libmlx  (M2)
```

The layers order what each part builds on, not a strict order of requires:
gen.dynamic (L2) requires gen.dist.core (L3), and gen.dist.core requires
gen.gfi (L2) (owner, 2026-10-25: documented, not untangled; found by the
book's chapter 034).

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
- **PRNG:** Threefry2x32 (as MLX and JAX), bit-compatible with MLX: a key
  draws the same uniforms on both backends (§5.3; genclj-ldpy).
- **Performance:** for scalar-heavy models (most PPL models) it is likely
  *faster* than MLX, with no FFI and no ~tens-of-µs eval floor. Large batched
  workloads are what the GPU is for.

### 5.3 MLX backend (M2): mlx-c via Panama FFM

**Decision.** Bind MLX's official C API (mlx-c v0.7.0, which pins MLX v0.32.2;
since 1.2.0 mlx-c v0.8.0 and MLX v0.32.4)
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
- **Quantized weights off Metal (measured 2026-10-21, the RTX-6000 machine; genclj-kw88).**
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
- **Float weights on CUDA (measured 2026-10-21, the RTX-6000 machine; genclj-kw88).** Two
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
| test.check (official) | yes | `:test`, and `ext/laws` (`:laws`): the GFI laws as properties, run with clojure.test specs (`deflaws`) or `laws/verify` |
| malli (official) | edges only | `gen.dev` instrumentation; LLM structured output (schema → grammar) |
| clojure.spec | no | Overlaps malli; specs aren't data; spec2 has been alpha for years |
| instaparse (official) | no as the engine (the owner, 2026-10-21: §9, grammar-constrained decoding); yes as a grammar front end in `ext/instaparse`, on `:test` (2026-11-05, approved by the owner the same day: §9, instaparse grammars as constraints) | our own regex → DFA engine (`gen.llm.regex`) |
| promesa | no | The core is synchronous; JVM tokenizers are synchronous; virtual threads cover concurrency |
| SCI | later | Sandboxed eval of LLM-generated code (program synthesis) |
| clojure.math (built in) | yes | Replaces GenMLX's ~370 `js/Math` calls |
| criterium | yes | `:bench` |
| Apache Commons Math 3 | extension | `:commons-math` (`ext/commons-math`): its distributions as gen.clj's, GenDistributions' role (M4, 2026-10-20) |

The nbb-compatibility forks GenMLX uses (malli, instaparse, test.check) are
**not needed** on the JVM.

## 9. Conventions (current)

The conventions in force, by topic, each linked to the entry that decided it
in [docs/decisions.md](docs/decisions.md), the dated log (who, when, why, the
alternatives); "DESIGN §9, <title>" cites the entry of that title. A new
decision is appended there, and this summary changes with it.

### Releases and the 1.x promise

- Through 1.x, names, arities, result-map keys and documented semantics stay
  put; minor releases add; a removal waits for 2.0 after a deprecation
  cycle. Draws for a key are not promised, only their distributions.
  [decided](docs/decisions.md#the-promise)
- Each namespace is `{:api :stable | :experimental | :internal}`; the
  protocols implementers extend keep their methods and gain none (a new
  capability is a new protocol). [decided](docs/decisions.md#three-tiers-marked-in-code), [as built](docs/decisions.md#enforced),
  [confirmed](docs/decisions.md#confirmed-by-the-owner)
- Experimental: gen.llm.\*, dist.host, trace-kernel, partition, tempering and
  the device path; conjugacy, Laplace and particle MCMC also PROVISIONAL.
  [decided](docs/decisions.md#three-tiers-marked-in-code), [tempering](docs/decisions.md#tempered-smc-a-log-marginal-likelihood-for-any-model), [staging](docs/decisions.md#device-programs-the-notes-decisions-1-2-4-and-5), [provisional](docs/decisions.md#110s-gpu-work-the-dense-metric-and-the-order)
- The stable surface is a committed snapshot (`clojure -M:test -m gen.api
  write`), appendix 063 its listing; a change dependents see gets a CHANGELOG
  line. [decided](docs/decisions.md#enforced)
- One artifact, gen.llm inside; optional dependencies in ext/ modules, so the
  core depends on Clojure alone. [decided](docs/decisions.md#one-artifact), [instaparse](docs/decisions.md#instaparse-grammars-as-constraints),
  [codegen](docs/decisions.md#codegen-as-a-library-the-core-gets-no-code-evaluation)
- Dependents (memo.clj, GenGDT) pin a release (memo.clj 1.2.0 pins 1.2.0)
  and adapt to gen.clj's decisions. [decided](docs/decisions.md#a-10-for-dependents), [1.2.0](docs/decisions.md#110-then-120-with-memoclj),
  [adapt](docs/decisions.md#reading-a-value-being-differentiated-on-the-host-is-an-error)
- Growing the stable surface takes a minor release; a patch may make silently
  wrong code loud (a Breaking line), but refusing what right code passes
  waits for 1.3.0. [decided](docs/decisions.md#110-not-101), [1.2.1](docs/decisions.md#121s-owner-calls)
- The gate: every test on the JVM backend, MLX's CPU and Metal, CUDA once,
  the book on the JVM backend, memo.clj and GenGDT on the release candidate.
  [1.0's](docs/decisions.md#before-the-tag), [1.1.0's](docs/decisions.md#the-gate-lighter-than-10s)
- 1.1.0 consolidated 1.0 ([decided](docs/decisions.md#genclj-110-a-consolidation-release), [scope](docs/decisions.md#its-scope-in-order),
  [off the path](docs/decisions.md#off-the-critical-path), [as it went](docs/decisions.md#as-the-release-went)); 1.2.0 is the language-model
  release ([decided](docs/decisions.md#120s-theme-language-models)); 1.2.1 the health check's fixes ([decided](docs/decisions.md#121-the-health-checks-fixes)).

### Process

- The owner decides, numeric edge semantics always; autonomous work records
  its calls PROVISIONAL in the log, with alternatives, and lists them in its
  report. [decided](docs/decisions.md#genmlx-extras-round-1), [1.2.0](docs/decisions.md#120-worked-autonomously-cuda-on-fenrir), [1.2.1](docs/decisions.md#121s-owner-calls)
- The decision log is docs/decisions.md; this section summarizes it.
  [decided](docs/decisions.md#121s-owner-calls)
- The repository is public since 2026-10-26, MIT; Gen.jl's test-suite ports
  keep Apache 2.0; `.beans/` is a private repo; main is pushed at a
  milestone's close or on the owner's go. [decided](docs/decisions.md#git)
- GPU runs go to fenrir's RTX PRO 6000, the Thor only if fenrir cannot serve.
  [decided](docs/decisions.md#120-worked-autonomously-cuda-on-fenrir)
- kmet is coupled only through its `:llm` contract at a pinned commit; each
  stage waits for the owner's go. [decided](docs/decisions.md#the-kmet-integrations-plan-plan-now-couple-thinly)

### Namespaces and house style

- The namespace root is `gen`. [decided](docs/decisions.md#namespace-root-gen)
- gen.array's operators and math are referred in place of clojure.core's
  (never `max`, `min`, `take`, `flatten`); the rest through `array`.
  [decided](docs/decisions.md#house-style-genarrays-operators-are-referred), [alias](docs/decisions.md#genarrays-alias-is-array)
- The layers of §4 order what builds on what, not requires. [decided](docs/decisions.md#ten-decisions-from-the-books-friction)
- Indices are 0-based: Map's and Unfold's addresses, Recurse's nodes,
  categorical's values, Switch's branches. [decided](docs/decisions.md#indices-are-0-based)

### Arrays, dtypes and numerics' edges

- Three ways to run, all first-class: the pure-JVM backend, MLX's CPU, MLX's
  GPU (Metal, CUDA); native libraries only as plug-ins. [decided](docs/decisions.md#three-ways-to-run-all-first-class)
- The JVM backend is float64, MLX float32, and core code assumes no dtype;
  int64 + float32 is float32, as in JAX. [decided](docs/decisions.md#dtypes), [promotion](docs/decisions.md#ten-decisions-from-the-books-friction)
- Arrays are values: the JVM compares by dtype, shape and bits, MLX
  numerically, a documented divergence. [decided](docs/decisions.md#arrays-are-values), [kept](docs/decisions.md#121s-owner-calls)
- `=` on a 0-d MLX array and a host value throws: compare with `array/==` or
  in a gen body. [decided](docs/decisions.md#-on-a-0-d-mlx-array-and-a-host-value-is-loud)
- Arrays are no collections: `get`, `nth` and `count` throw
  `:not-a-collection` on every kind. [decided](docs/decisions.md#arrays-are-no-clojure-collections-get-nth-and-count-are-loud), [kept](docs/decisions.md#121s-owner-calls)
- Casts to integers are Java's: truncate, saturate, NaN to 0, narrowing
  wraps. [decided](docs/decisions.md#casts-to-integers-are-javas)
- Booleans in arithmetic count, as in Julia (`(+ true true)` is 2);
  bernoulli's values are booleans. [decided](docs/decisions.md#booleans-in-arithmetic-count-as-in-julia), [bernoulli](docs/decisions.md#bernoulli-values-are-booleans)
- int64 overflow throws `:overflow` on the JVM backend (MLX wraps); integer
  `arange` is NumPy's, a bool one refused. [decided](docs/decisions.md#121s-owner-calls)
- Float `floor-div` is NumPy's; extrema of an empty axis throw
  `:empty-reduction`; `solve` takes a right-hand side one axis short as a
  batch of vectors. [decided](docs/decisions.md#121s-owner-calls)
- lgamma and digamma at their edges are SciPy's; a log mass of 0 has
  gradient 0. [edges](docs/decisions.md#lgamma-and-digamma-at-their-edges-are-scipys), [log mass](docs/decisions.md#a-log-mass-of-0-has-gradient-0-not-nan)
- An out-of-range gather throws on the JVM backend and clamps on MLX, checked
  there with `-Dgen.mlx.check-gathers=true`; loud by default waits for 2.0.
  [decided](docs/decisions.md#the-mlx-backend), [kept](docs/decisions.md#mlxs-gathers-keep-clamping-in-110), [opt-in](docs/decisions.md#mlxs-gathers-the-clamp-kept-an-opt-in-check-added)
- `array/full` takes a scalar fill, never an array. [decided](docs/decisions.md#filling-with-an-array-is-loud)
- Where MLX differs from NumPy the backend follows NumPy; MLX's CPU sums
  float32 in blocks of 32, half floats in float32. [decided](docs/decisions.md#the-mlx-backend)
- Float32 matrix products are full float32 on every device: TF32 is off
  unless `-Dgen.mlx.tf32=true`. [decided](docs/decisions.md#110s-gpu-work-the-dense-metric-and-the-order), [1.0's workaround](docs/decisions.md#before-the-tag)
- MLX is upstream, patched as little as possible: the official wheel, a
  source build for the Thor's CUDA; compile is the identity where g++ fails.
  [decided](docs/decisions.md#mlx-is-upstream-patched-as-little-as-possible), [the Thor](docs/decisions.md#cuda-on-the-jetson-agx-thor-sm_110-mlx-built-from-its-source), [g++](docs/decisions.md#mlxs-cpu-compile-where-the-systems-g-cannot-build-mlxs-kernels-falls-back-to-the-identity)
- MLX's memory is bounded; one budget covers MLX and native engines; at
  Metal's buffer limit the backend collects and retries once.
  [bounds](docs/decisions.md#the-mlx-backend), [budget](docs/decisions.md#the-shared-memory-budget), [buffers](docs/decisions.md#metals-buffer-limit-collect-and-retry-once)
- A compiled function may be called from any thread. [decided](docs/decisions.md#compiled-functions-across-threads)

### Differentiation

- Reading a value being differentiated on the host is an error: compute with
  gen.array, give a custom gradient (`array/custom-vjp`), or read
  `stop-gradient`. [decided](docs/decisions.md#reading-a-value-being-differentiated-on-the-host-is-an-error), [custom-vjp](docs/decisions.md#custom-gradients-arraycustom-vjp)
- Gradients under the batch tracer are vmap's, per particle;
  `array/vmap` takes in-axes 0 or nil. [decided](docs/decisions.md#gradients-under-the-batch-tracer), [vmap](docs/decisions.md#confirmed-by-the-owner)

### Randomness and keys

- Keys are JAX's Threefry-2x32-20; every site consumes one; the implicit
  source's root is random per JVM (`gfi/seed!`); `fold-in` takes non-negative
  integers below 2^63. [decided](docs/decisions.md#randomness)
- `rng/key` refuses floats; a staged key is a value too. [decided](docs/decisions.md#121s-owner-calls),
  [staged](docs/decisions.md#device-programs-the-notes-decisions-1-2-4-and-5)
- `randint` draws exact int64 integers from the default float's uniforms.
  [decided](docs/decisions.md#121s-owner-calls)
- A device run of n particles or chains runs i on `(rng/split key n)[i]`,
  as unbatched handler runs do (gen.batched draws otherwise). [decided](docs/decisions.md#device-programs-per-particle-keys)
- Untraced draws, and plain calls of a generative function in a body, take
  keys from ω. [decided](docs/decisions.md#encapsulated-randomness)
- A particle draws the same under a Switch by partition, whatever the
  partition. [decided](docs/decisions.md#switches-by-partition)

### The GFI, traces and choice maps

- GFI signatures are Gen.jl's, every operation, assess too, taking a key;
  unvisited constraints are one error, `gfi/unvisited`. [decided](docs/decisions.md#gfi-signatures-are-genjl-shaped),
  [unvisited](docs/decisions.md#trainable-parameters-are-an-explicit-store), [assess](docs/decisions.md#encapsulated-randomness)
- Traces are immutable values that store each choice's log probability.
  [values](docs/decisions.md#traces-are-values), [log probabilities](docs/decisions.md#traces-store-each-choices-log-probability)
- Constraints are choice maps: a plain map is an error, nil the empty one.
  [decided](docs/decisions.md#constraints-are-choice-maps)
- A vector is a path everywhere, `[]` the root; `cm/merge` takes 0 to n maps.
  [decided](docs/decisions.md#a-vector-is-a-path-everywhere), [edges](docs/decisions.md#a-calls-trace-and-return-value-by-address-merge-and-recurses-numbering)
- Lookups are loud where Gen.jl throws KeyError (ex-info
  `{:reason :path :at}`); a not-found argument is lenient; `has-value?` and
  `has-submap?` never throw. [decided](docs/decisions.md#lookups-are-loud)
- `(tr addr)` is the choice or the call's return value; outside a body a 0-d
  boolean comes back a host boolean; a keyword reads the record's fields.
  [decided](docs/decisions.md#a-calls-trace-and-return-value-by-address-merge-and-recurses-numbering), [booleans](docs/decisions.md#encapsulated-randomness), [keywords](docs/decisions.md#eight-more-confirmed-by-the-owner)
- Choice maps are callable; selections have union and intersection; traces
  print compactly, for display only. [decided](docs/decisions.md#eight-more-confirmed-by-the-owner)
- The address order is total: numbers by value (NaN last), one class by
  value, the rest by class name. [decided](docs/decisions.md#121s-owner-calls)
- What an update or regenerate keeps depends on the trace's maker and the
  store at the call: either changed, the trace is revised. [decided](docs/decisions.md#trainable-parameters-are-an-explicit-store)
- Retdiffs are more precise than Gen.jl's (`gfi/same-value?`); regenerate's
  weight is Eq. 4.1's, under a Switch too. [decided](docs/decisions.md#differences-from-genjls-test-suite)
- Change hints are int-, set-, dict- and vector-diffs and `diff/propagate`'s
  rules; arithmetic gives an int-diff on integers only. [decided](docs/decisions.md#change-hints-and-diffed-propagation),
  [integers](docs/decisions.md#the-path-equivalence-fuzzers-remainder)
- `splice` takes an identical argument (or an equal host scalar, keyword or
  string) as unchanged; dynamic bodies pass unknown-change to splices.
  [decided](docs/decisions.md#ten-decisions-from-the-books-friction)
- A choice outside its support stops the run (weight -Inf); a move from an
  impossible trace weighs NaN; an irreversible proposal is loud.
  [decided](docs/decisions.md#ten-decisions-from-the-books-friction)
- A weight that is 0 compares as weights do, by close?. [decided](docs/decisions.md#a-weight-that-is-0-compares-as-weights-do)
- Bodies run to completion; transform bodies are eager. [decided](docs/decisions.md#bodies-run-to-completion),
  [transforms](docs/decisions.md#transform-bodies-are-eager)
- `gen` guards `if`, `case`, `catch` and clojure.core's lookups; in a body a
  0-d array counts as its host value. [decided](docs/decisions.md#branching-on-array-values), [guards](docs/decisions.md#batched-execution),
  [0-d](docs/decisions.md#the-mlx-backend)
- `gen` refuses a reserved name (`trace`, `splice`, `untraced`, …) as a
  parameter or a local it reads. [decided](docs/decisions.md#121s-owner-calls), [reserved](docs/decisions.md#encapsulated-randomness)
- The handler's state is public (`dyn/initial-state`); middleware writes its
  own keys and `:constraints`, returned on the trace. [decided](docs/decisions.md#the-handlers-state-and-middleware),
  [in advance](docs/decisions.md#approved-in-advance-by-the-owner)
- An address-less splice runs under its caller's middleware, unless the
  callee has its own. [decided](docs/decisions.md#an-address-less-splice-runs-under-its-callers-middleware)
- Encapsulated randomness: ω kept when nothing it depends on changed;
  `untraced`, `gen.custom/estimated`; exact-density operations refuse it.
  [decided](docs/decisions.md#encapsulated-randomness)
- Trainable parameters are an explicit store (`dyn/with-params`,
  `(param :w)`), gradients values, reached in combinators by `gfi/stores`.
  [decided](docs/decisions.md#trainable-parameters-are-an-explicit-store), [combinators](docs/decisions.md#parameter-gradients-through-a-combinators-trace), [custom](docs/decisions.md#a-custom-functions-trainable-parameters)
- Serialization is EDN with gen.clj's tags; a trace is rebuilt by generate,
  refused under other numerics; traces with ω are not serializable yet.
  [decided](docs/decisions.md#serialization), [EDN](docs/decisions.md#genmlx-extras-round-1)
- Batched execution is the scalar code under the batch tracer, falling back
  to one particle at a time with its reason in `:fallback`.
  [decided](docs/decisions.md#batched-execution), [fallback](docs/decisions.md#ten-decisions-from-the-books-friction)
- `gen.batched` takes per-particle constraints; enumeration takes
  `:batched?`. [decided](docs/decisions.md#per-particle-constraints-and-batched-enumeration)
- `(gen ^:static …)` is the same semantics with a graph: a body that is not
  static is a compile error; assess is a compiled score. [decided](docs/decisions.md#the-static-modeling-language)

### Combinators

- Recurse is Gen.jl's, 0-based; Scan is JAX's; Mix a categorical and a
  Switch, its weights probabilities; dimap; no Mask. [Recurse](docs/decisions.md#recurse),
  [Scan, Mix, dimap](docs/decisions.md#scan-mix-and-dimap)
- A Switch whose index differs across particles batches by partition.
  [decided](docs/decisions.md#genmlx-extras-round-1), [built](docs/decisions.md#switches-by-partition)
- Map's, Unfold's and Scan's scores are counts beside a finite sum, never
  NaN. [decided](docs/decisions.md#running-totals-never-nan)
- Combinators refuse what is not a generative function; a negative selection
  index throws in every kind. [decided](docs/decisions.md#121s-owner-calls)
- Kalman and HMM generative functions (stable) score exact marginal
  likelihoods; `sample-states` of an impossible trace throws.
  [decided](docs/decisions.md#conjugacy-and-kalman-and-hmm-generative-functions), [impossible](docs/decisions.md#121s-owner-calls)
- `with-exact-proposal` draws finite choices from their exact conditional;
  `with-internal-proposal` regenerates by Eq. 4.1. [enumeration](docs/decisions.md#exact-enumeration),
  [internal proposals](docs/decisions.md#the-internal-proposal-override)

### Distributions

- Gen.jl's names and parameterizations (`categorical` of probabilities,
  `categorical-logits`), else Distributions.jl's; no `delta`.
  [decided](docs/decisions.md#distributions-use-genjl-names-and-parameterizations), [no delta](docs/decisions.md#genmlx-extras-round-1)
- Distributions.jl's continuous families are native; host-library families
  draw the quantile of a Threefry uniform. [decided](docs/decisions.md#distributions-from-a-host-library-gendistributions-distributionsbacked)
- Pushforwards are `(dist name [params] body)`, Gen.jl's `@dist`.
  [decided](docs/decisions.md#pushforward-distributions-genjls-dist)
- GenMLX's further families are exact at their extremes; circular values lie
  in [-π, π). [decided](docs/decisions.md#genmlxs-further-distributions)
- A draw past the float range is +Inf; an underflow is floored at the
  smallest normal; beta is exact at 0 and 1. [decided](docs/decisions.md#a-draw-beyond-the-float-range-is-inf), [beta](docs/decisions.md#differences-from-genjls-test-suite)
- Probabilities sum to 1 within 4·log2(K)·ε (at least 1e-6); a discrete and
  continuous mixture is an error. [sums](docs/decisions.md#probabilities-sum-to-1-within-their-dtype), [mixtures](docs/decisions.md#a-mixture-of-a-discrete-and-a-continuous-family-is-an-error)
- A non-finite host parameter is refused where a family's code gives NaN;
  infinite values lie outside every support. [decided](docs/decisions.md#121s-owner-calls)
- A covariance singular in floating point is an error. [decided](docs/decisions.md#differences-from-genjls-test-suite)
- A family with a pathwise sampler is marked `^:pathwise`. [decided](docs/decisions.md#the-pathwise-estimator-advis)

### Inference

- A kernel is `(fn [trace key])` → `{:trace :accepted? …}`, with an optional
  batched form; `(kern ^:trace [tr] …)` reads the trace, always checked.
  [shape](docs/decisions.md#kernels-shape), [batched](docs/decisions.md#kernels-batched-form), [kern](docs/decisions.md#kernels-that-read-the-trace-kern-trace-tr--args-)
- One vocabulary of result keys, plural per particle; chains take `:keep`
  and return `:draws`. [decided](docs/decisions.md#the-result-keys), [plural](docs/decisions.md#eight-more-confirmed-by-the-owner)
- HMC, MALA and NUTS reject and count a divergence (energy error over 1000,
  a non-finite step, a failing parameter check); starts must be finite.
  [decided](docs/decisions.md#a-diverging-hmc-trajectory-is-rejected)
- NUTS and its warmup are Stan's, with `:eps`, `:inv-metric` and `:jitter`.
  [decided](docs/decisions.md#nuts-and-stans-warmup), [Stan's](docs/decisions.md#genmlx-extras-round-1)
- `:inv-metric` is labeled by address, or `mcmc/dense-inv-metric`; there is
  no `:metric`; invalid `:eps`, `:L`, `:tau` throw. [labeled](docs/decisions.md#labeled-metrics),
  [dense](docs/decisions.md#110s-gpu-work-the-dense-metric-and-the-order), [checks](docs/decisions.md#121s-owner-calls)
- Trace transforms, translators and SMCP3; the general translator subtracts
  the forward proposal's score. [decided](docs/decisions.md#trace-transforms), [the API](docs/decisions.md#the-transform-api)
- A trace kernel is a gen function returning `[update backward]`.
  [decided](docs/decisions.md#trace-kernels-gentracekerneldsl)
- MCMC diagnostics follow Vehtari et al. (2021), checked against ArviZ.
  [decided](docs/decisions.md#mcmc-diagnostics), [ArviZ](docs/decisions.md#genmlx-extras-round-1)
- Residual resampling rounds n·p_i up within a relative 1e-9 of an integer.
  [decided](docs/decisions.md#ten-decisions-from-the-books-friction)
- VI trains the model's parameters too; `:estimator :reparam` is the
  pathwise estimator. [model](docs/decisions.md#variational-inference-trains-the-models-parameters-too), [pathwise](docs/decisions.md#the-pathwise-estimator-advis)
- Optimizers are maps with a `:kind`, PyTorch's rules, ascending; train and
  VI resume from `:state`; a step evaluates what it returns.
  [optimizers](docs/decisions.md#the-optimizer-interface), [training](docs/decisions.md#trainings-and-vis-shapes), [evaluation](docs/decisions.md#an-optimizer-step-evaluates-what-it-returns)
- Batched inference: `map-particles` and the `-particles` kernels.
  [decided](docs/decisions.md#gradients-under-the-batch-tracer)
- Tempered SMC estimates a log marginal likelihood for any model; its accept
  throws on NaN. [decided](docs/decisions.md#tempered-smc-a-log-marginal-likelihood-for-any-model), [1.2.1](docs/decisions.md#121s-owner-calls)
- Conjugate proposals, Laplace and particle MCMC are experimental and
  PROVISIONAL. [conjugacy](docs/decisions.md#conjugacy-and-kalman-and-hmm-generative-functions), [Laplace](docs/decisions.md#laplace-and-fisher-information), [particle MCMC](docs/decisions.md#particle-mcmc)
- Compiled programs share one LRU cache of 64. [decided](docs/decisions.md#121s-owner-calls)

### Device programs (experimental)

- The staging backend, `array/scan` and staged keys are in the core at L0;
  the lowerings (JVM, Metal, CUDA C) are ext/device's. [decided](docs/decisions.md#device-programs-the-notes-decisions-1-2-4-and-5),
  [the prototype](docs/decisions.md#device-programs-the-prototype-before-the-notes-review)
- `{:device :jvm | :metal | :cuda}` on kernel/run and importance sampling;
  a model that does not stage runs on the handler, its reason recorded.
  [decided](docs/decisions.md#device-programs-the-notes-decisions-1-2-4-and-5), [every model](docs/decisions.md#device-programs-work-for-every-model-sustainably)
- A float64 program on the JVM equals the handler bit for bit; float32 and
  GPU programs agree within a band. [decided](docs/decisions.md#device-programs-the-notes-decisions-1-2-4-and-5)
- A run's error (an out-of-range gather too) is replayed on the handler, and
  the handler's error thrown. [decided](docs/decisions.md#device-programs-an-out-of-range-gather-index-is-the-handlers-error), [run errors](docs/decisions.md#121s-owner-calls)
- One chain runs on a CPU core (the JVM lowering), whatever the backend.
  [decided](docs/decisions.md#one-chain-fast-on-mlx-and-gpu-speed), [1.2.1](docs/decisions.md#121s-owner-calls)
- `device/run-chains` runs chain i on `(rng/split key n)[i]`, its fallback
  batched; `kernel/run-particles` waits for per-particle keys.
  [decided](docs/decisions.md#110s-gpu-work-the-dense-metric-and-the-order)
- CUDA's lowering is parked, its tests skipped by name. [decided](docs/decisions.md#121s-owner-calls)

### Language models (experimental)

- An engine's branch is a value that advance, extend and resample consume;
  engines never sample. [decided](docs/decisions.md#the-branching-protocol-and-the-llamacpp-engine)
- The LM protocol's callers hold tokens; its trie owns the branches;
  batching is explicit, with no autobatcher. [decided](docs/decisions.md#the-lm-protocol)
- Each token is scored once and kept; tokens are drawn on the host from the
  site's key, rule 3's one exception. [decided](docs/decisions.md#the-token-generative-functions-foundations)
- `token/completion` is a generative function of `[prompt max-tokens]`;
  end tokens are choices (`:end-tokens`). [decided](docs/decisions.md#the-token-generative-function)
- Token SMC targets the completion times exp(complete); a completion with its
  own `:potential` is refused. [decided](docs/decisions.md#token-smc-and-potentials)
- Grammars: our own regex-to-DFA engine over bytes, exact masks; a potential
  declares its end tokens. [decided](docs/decisions.md#grammar-constrained-decoding)
- Context-free grammars by Earley, pure, byte-level liveness, masks in parts
  (PROVISIONAL). [decided](docs/decisions.md#context-free-grammars-by-earley), [masks](docs/decisions.md#121s-owner-calls)
- instaparse grammars are constraints through ext/instaparse.
  [decided](docs/decisions.md#instaparse-grammars-as-constraints)
- Potential combinators are genlm-control's, checked as laws; AWRS is token
  SMC's `:awrs` proposal. [combinators](docs/decisions.md#potential-combinators), [AWRS](docs/decisions.md#awrs-in-token-smc)
- A hybrid model splits its steps over recurrent cells only on Metal.
  [decided](docs/decisions.md#a-hybrid-models-steps-split-over-its-recurrent-cells)
- llama.cpp's noise bands are per device. [decided](docs/decisions.md#llamacpps-band-by-device)
- gen.llm.sample, chat and agent are the core's; evaluating LLM-written code
  is ext/codegen's alone. [decided](docs/decisions.md#codegen-as-a-library-the-core-gets-no-code-evaluation)
- GenLM parity is an epic, 1.2.0's theme, with llama.cpp v0.9.0.
  [epic](docs/decisions.md#genlms-gaps), [theme](docs/decisions.md#120s-theme-language-models)
- mlx-node parity is 1.3.0 (kmet 1.4.0, the MSA 1.5.0, inference over
  agents 1.6.0): its features over mlx-c, with C++ of our own only for paged
  attention and continuous batching; mlx-node never a dependency.
  [decided](docs/decisions.md#mlx-node-parity-on-mlx-c-and-our-own-c), [releases](docs/decisions.md#130-140-and-150), [split](docs/decisions.md#inference-over-agents-best-of-k-and-program-proposals-in-150-the-rest-in-160)
- GPU kernels are ported from where they originated, with attribution and
  their license kept; launches, engines and primitives are ours; each kernel
  is checked against its composite. [decided](docs/decisions.md#130s-kernels-ported-from-their-origins-everything-around-them-ours)
- gen.array's LLM operations are experimental: rms-norm, rope and sdpa are
  composites (gen.array.nn) a backend may fuse (IFused), and quantize,
  dequantize, quantized-matmul and gather-qmm are IQuantized's, in MLX's
  affine layout; the JVM backend stores uint32. [decided](docs/decisions.md#genarrays-llm-operations-two-optional-protocols-uint32-on-the-jvm)
- GGUF weights load through gen.array/load-gguf (ILoading; a host reader on
  the JVM backend, mlx_load_gguf on MLX); model families live in
  gen.llm.models.*, backend-neutral; Qwen3 is checked against llama.cpp
  within 0.075 nats on Metal. [decided](docs/decisions.md#gguf-weights-and-qwen3s-forward-on-genarray)
- The MLX engine: GGUF weights through MLX's loader (Q8_0 scales float32),
  llama.cpp's vocabulary, gen.array's LLM operations (rms-norm, rope, sdpa;
  quantized-matmul, dequantize, gather-qmm), a Slots engine, MLX's native
  gradients with the tape as reference. [decided](docs/decisions.md#130s-engine-design-the-seven-recommendations-approved)

### Tests, laws and the book

- Reference values come from scipy and JAX as committed fixtures; against
  GenMLX, statistics, never exact samples. [decided](docs/decisions.md#testing)
- A law returns `checks` and `vacuous`; a law mostly vacuous (over 60%)
  fails; seeds come from `laws/seeds`. [decided](docs/decisions.md#law-results)
- gen.laws is ext/laws, stable; a minor release may add a law or change
  close?'s tolerances, with a CHANGELOG line. [module](docs/decisions.md#the-laws-for-users-in-extlaws), [names](docs/decisions.md#the-laws-names-confirmed-by-the-owner)
- close?'s absolute tolerance scales with the scores. [decided](docs/decisions.md#genlaws-tolerance-scales-with-the-scores)
- The test loop is `(t)` at a REPL; `gen.parallel` shards over 5 JVMs, JVM
  backend only. [decided](docs/decisions.md#the-test-loop)
- One set of 5-SE helpers; negative controls by `with-redefs-fn`;
  `-Dgen.require-native` fails runs that skipped. [decided](docs/decisions.md#121s-owner-calls)
- Differences from Gen.jl's test suite are recorded with their reasons.
  [decided](docs/decisions.md#differences-from-genjls-test-suite)
- The book is Markdown whose code runs, its claims at 5 standard errors, the
  JVM backend its gate. [decided](docs/decisions.md#the-book-is-markdown-whose-code-runs)
- The book's test is one namespace per Part, Parts II–V `^:slow`.
  [decided](docs/decisions.md#ten-decisions-from-the-books-friction)

## 10. Deferred (tracked, not forgotten)

`beans list --tag deferred-port` lists every GenMLX area left out of M1–M4, each
with its location in GenMLX, a quality assessment, and how to port it cleanly.
`beans list --tag watch` lists external technologies to re-evaluate (TornadoVM,
jitllm, Raster).
