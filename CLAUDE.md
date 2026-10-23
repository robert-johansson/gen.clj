# gen.clj — guide for agents

gen.clj ports GenMLX (`~/code/genmlx`, ClojureScript on MLX) to Clojure on the
JVM, aiming at Gen.jl's capabilities. The design and its decisions are in
`DESIGN.md`. Read it, and the docs it points to, before changing architecture.

- `docs/array-api.md`: the array API and its backend contract (approved).
- `docs/porting-genmlx.md`: how to translate GenMLX code, with the distribution
  safeguards.
- `docs/research/`: the evidence behind the decisions (backends, LLMs, GenMLX
  triage, Gen.jl parity checklist and the GFI laws).

## Work tracking: beans

All work is tracked in beans (`.beans/`, a private git repo synced by hooks; see
`.beans/hooks/README.txt`).

- **Finding work:** `beans list --ready`, `beans show <id>`, `beans roadmap`.
- **Working:** put a bean in-progress before starting, tick its checklist as
  you go, and close it with a `## Summary of Changes`.
- **Flags:** the tag flag is `--tag` (repeatable). In beans 0.4.2, `--ready`
  does not inherit blocking from a blocked parent, so put `--blocked-by` on the
  leaf tasks.
- **Deferred work:** everything in GenMLX left for later is a `deferred-port`
  bean. External technologies to revisit are `watch` beans.

## Running

```
clojure -X:test                       # every test (clojure.test + test.check), ~90 s
clojure -X:test:fast                  # without the ^:slow statistical tests, ~25 s
clojure -X:test:conformance           # the backend conformance suite (array level)
clojure -X:test:commons-math          # also the Commons Math extension (ext/commons-math)
clojure -J-Dgen.backend=mlx -X:test   # any suite on the MLX backend (add -J-Dgen.mlx.device=cpu|gpu, -J-Dgen.mlx.dtypes=64)
clojure -M:bench -m gen.array-bench   # per-op costs; add :c2 to compare C2 with Graal's JIT
clojure -M:bench -m gen.model-bench   # model-level costs (GFI ops, batching, gradients)
clojure -J-Dgen.llama.lib=<lib> -J-Dgen.llama.test-model=<gguf> -X:test :nses '[gen.llm.llama-test gen.llm.llama-lm-test gen.llm.llama-token-test gen.llm.llama-smc-test]'
clojure -J-Dgen.llama.lib=<lib> -M:bench -m gen.llama-noise <gguf>              # llama.cpp's float noise and costs
clojure -J-Dgen.llama.lib=<lib> -M:test:bench -m gen.llama-long-noise <gguf>    # the same past a sliding window
```

**The MLX backend** needs its native libraries and Java binding, built once:
`native/build-mlx.sh` (libmlxc against the pinned MLX wheel), then
`clojure -T:build java` (compiles `src-java/` into `target/classes`). After
changing mlx-c's version, regenerate the binding with `clojure -M
native/bindgen.clj`. The backend conformance suite
(`gen.backend-conformance-test`) compares a backend with the JVM backend;
without the native libraries the MLX tests skip.

**llama.cpp** (the local-LLM engine, `gen.llm.llama`) needs its libraries,
built once: `native/build-llama.sh` builds the pinned release (v0.7.0; Metal
on macOS, `cpu` or `cuda` on Linux) into `native/build/llama/<platform>`,
then `clojure -T:build java`. The JVM finds them with
`-J-Dgen.llama.lib=<dir>/lib` (default: that build directory). On the M4 Mac
they are installed at `~/gen-clj-models/llama-v0.7.0`, beside the test
models (Qwen3-0.6B, Qwen3.5-0.8B and Gemma-3-270M, Q8_0 GGUFs). The llama.cpp
tests take a model with `-J-Dgen.llama.test-model=<gguf>` and skip without
one; run one model JVM at a time. Everything else in `gen.llm` runs on the
pure-JVM toy engines (`test/gen/llm/toy_engine.clj`, `cell_engine.clj`).

**Test conventions.**
- **Determinism.** Statistical tests use fixed keys, so they are deterministic. Their
  bands come from the estimator's variance (5 standard errors, or a
  measured spread), never from a tolerance that happens to pass.
- **Power.** A new test must be able to fail. Check a statistical or law test
  against a planted bug (a mutation) before trusting it, as the suites
  document in place.
- **Tiers.** Tag tests that run more than a few seconds `^:slow`. Tag
  scalar statistical tests that take minutes on MLX (one evaluation per read
  value) `^:heavy`; `-X:test:light` skips both, for quick MLX runs.
- **Backends.** Tests run on every backend: read results with
  `gen.test-support/host`, compare with `close?` or `tol` (float32 aware),
  write models with gen.array's operators, and keep host-number assumptions
  behind `ts/jvm?`.

**Fixtures.** Reference values in `test/fixtures/*.edn` come from scipy and JAX
via `test/fixtures/generate.py`. Python is needed only to regenerate them:
`~/code/genmlx/.venv/bin/python -I test/fixtures/generate.py`. Tests never need
Python.

## Layers (only L0 knows a backend)

| Layer | Namespaces |
|---|---|
| L0 arrays | `gen.array` (public API), `.dtype`, `.shape`, `.special`, `.linalg`, `.tape` (autodiff), `.batch` (batching tracer), `.protocols`; `gen.backend.jvm`; `gen.backend.mlx` (+ `.exact`; Java binding in `src-java/gen/mlx`); `gen.random`, `gen.random.threefry` |
| L1 data | `gen.choicemap`, `gen.selection`, `gen.trace`, `gen.diff` |
| L2 GFI | `gen.gfi` (protocol + API), `gen.serialize` (choice maps, arrays and traces as EDN), `gen.dynamic` (handler, DynamicGF, `gen`), `gen.dynamic.guard` (the `gen` body rewriter), `gen.dynamic.static` (the static language's graph and its code), `gen.dynamic.core` (guarded clojure.core fns for `gen` bodies) |
| L3 distributions | `gen.dist.core` (record, `defdist`), `gen.dist`, `gen.dist.dsl` (`dist`: pushforwards, Gen.jl's `@dist`), `gen.dist.host` (host-library families, GenDistributions; `gen.dist.commons-math` in `ext/`) |
| L4 combinators | `gen.combinators` (map, unfold, switch, recurse, call-at, choice-at), `gen.custom` (deterministic generative functions, custom gradients, estimated densities) |
| L5 inference | `gen.inference.weights`, `.importance`, `.mh`, `.kernel`, `.particle-filter`, `.mcmc` (HMC, MALA, NUTS, MAP, elliptical slice), `.nuts` (Stan's warmup and sampler), `.enumerative` (grids, exact proposals), `.proposal` (internal-proposal override), `.transform` (Gen.jl's @transform, trace translators, SMCP3), `.trace-kernel` (GenTraceKernelDSL), `.vi` (black-box VI, VIMCO), `.diagnostics` (ESS, R-hat, MCSE); `gen.gradients`, `gen.learning` |
| LLM runtime | `gen.llm.branching` (the Branching protocol), `.slots` (slot engines on one thread), `.llama` (llama.cpp; Java binding in `src-java/gen/llama`), `gen.llm.lm` (the LM protocol), `.cache`, `.trie`, `.cells` (the LM's cache, prefix trie, KV-cell ledger) |
| LLM generative functions | `gen.llm.token` (completions, the token distribution), `gen.llm.potential` (potentials), `gen.llm.smc` (token SMC) |

## Rules

1. **The handler is the semantics.** Optimizations are later plug-ins that must
   be observationally equivalent and must never change what a GFI operation
   means. There is one set of handler transitions; batched execution is the
   same code under the batch tracer (`gen.array.batch`), so never write
   batched copies.
2. **Randomness is purely functional.** Every operation takes a key, and every
   site consumes one key. The implicit key source (`gfi/seed!`,
   `gfi/implicit-seed`) is the only impure point. Keys never go in metadata.
3. **Core code calls only `gen.array`, never a backend.**
   - No `eval!` inside models, handlers or distributions: only at
     inference-loop boundaries.
   - No hidden mutable state, and no memory management in semantics code.
4. **House style.** Refer gen.array's operators in place of `clojure.core`'s:
   `(:refer-clojure :exclude [+ - * / < > <= >= ==])`. Never refer `max`, `min`,
   `take` or `flatten`, which are reductions or gathers there. Call the rest
   through the alias `array`, the namespace's last segment: `(array/shape x)`.
   Tests of the array API itself may qualify the operators, to name the
   function under test.
5. **Distributions follow Gen.jl's names and parameterizations.**
   `categorical` takes probabilities; `categorical-logits` takes logits.
   Validate host parameters, and read array values only when
   `array/concrete?` says so.
6. **Errors are loud.** Shape mismatches, unvisited constraints, repeated
   addresses and the like throw `ex-info` with data. Never let one degrade into
   NaN or a silent default.
7. **Laws are tests.** The GFI laws live in `test/gen/laws.clj` (general and
   forward-sampling sets); every new generative function kind instantiates them
   with `(laws/deflaws prefix models n)`. Coverage is tracked in
   `docs/research/genjl-parity.md` §3.
   - Numerics are checked against fixtures.
   - Sampled quantities use fixed keys and variance-derived bands.
   - Against GenMLX, compare statistics, never exact samples.
8. **Minimal dependencies.** The core is Clojure only. Test and benchmark tools
   live in aliases.

## JVM performance notes (the JVM backend)

- **Warning flags.** Kernels use `*warn-on-reflection*` and
  `*unchecked-math* :warn-on-boxed`. Keep them warning-free.
- **Kernel size.** One small function per op loop (the `defunary`/`defbinary`
  macros); the JIT does not compile huge methods.
- **Loops used as values box.** Clojure compiles a `loop` in a binding or
  argument position into a closure, which boxes its result. Use the
  `fold-range` macro (statement-position loop) instead.
- **Hints inside syntax-quoted macros** must be class strings: `^"[D"`, `^"[J"`.
- **Primitive arguments.** Functions with primitive-hinted parameters take at
  most 4 arguments.
- **0-d arrays are host values** on the JVM backend (Double/Long/Boolean), so
  scalar models never allocate arrays.
