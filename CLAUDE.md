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
clojure -X:test                       # all tests (clojure.test + test.check)
clojure -M:bench -m gen.array-bench   # per-op costs; add :c2 to compare C2 with Graal's JIT
```

**Fixtures.** Reference values in `test/fixtures/*.edn` come from scipy and JAX
via `test/fixtures/generate.py`. Python is needed only to regenerate them:
`~/code/genmlx/.venv/bin/python -I test/fixtures/generate.py`. Tests never need
Python.

## Layers (only L0 knows a backend)

| Layer | Namespaces |
|---|---|
| L0 arrays | `gen.array` (public API), `.dtype`, `.shape`, `.special`, `.linalg`, `.tape` (autodiff), `.protocols`; `gen.backend.jvm`; `gen.random`, `gen.random.threefry` |
| L1 data | `gen.choicemap`, `gen.selection`, `gen.trace`, `gen.diff` |
| L2 GFI | `gen.gfi` (protocol + API), `gen.dynamic` (handler, DynamicGF, `gen`) |
| L3 distributions | `gen.dist.core` (record, `defdist`), `gen.dist` |
| L4 combinators | `gen.combinators` (map, unfold, switch) |
| L5 inference | `gen.inference.weights`, `.importance`, `.mh`, `.kernel`, `.particle-filter`, `.mcmc` (HMC, MALA, MAP, elliptical slice); `gen.gradients`, `gen.learning` |

## Rules

1. **The handler is the semantics.** Optimizations are later plug-ins that must
   be observationally equivalent and must never change what a GFI operation
   means. There is one set of handler transitions, parameterized by
   `:batch-shape`; never write batched copies.
2. **Randomness is purely functional.** Every operation takes a key, and every
   site consumes one key. The implicit key source (`gfi/seed!`,
   `gfi/implicit-seed`) is the only impure point. Keys never go in metadata.
3. **Core code calls only `gen.array`, never a backend.**
   - No `eval!` inside models, handlers or distributions: only at
     inference-loop boundaries.
   - No hidden mutable state, and no memory management in semantics code.
4. **House style.** Refer gen.array's operators in place of `clojure.core`'s:
   `(:refer-clojure :exclude [+ - * / < > <= >= ==])`. Never refer `max`, `min`,
   `take` or `flatten`, which are reductions or gathers there.
5. **Distributions follow Gen.jl's names and parameterizations.**
   `categorical` takes probabilities; `categorical-logits` takes logits.
   Validate host parameters, and read array values only when
   `a/concrete?` says so.
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
