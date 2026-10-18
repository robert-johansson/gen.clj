# Benchmarks (JVM backend)

Measured 2026-10-18 on an Apple M4 (32 GB), Oracle GraalVM 25.0.4, with
criterium's quick-benchmark (mean time per call). Reproduce with
`clojure -M:bench -m gen.model-bench` (add `:c2` for C2) and
`clojure -M:bench -m gen.array-bench` for per-op costs (`docs/array-api.md` §11).

## Model-level costs

| Operation | Graal JIT | C2 |
|---|---|---|
| linreg (12 sites): simulate | 20.8 µs | — |
| linreg: generate with 10 observations | 16.4 µs | — |
| linreg: assess | 14.4 µs | — |
| linreg: update one choice | 13.9 µs | — |
| linreg: regenerate `:slope` | 13.7 µs | — |
| linreg: choice gradients (slope, intercept) | 92.0 µs | — |
| hierarchical (10 splices): simulate | 32.8 µs | — |
| hierarchical: update `:mu` | 32.3 µs | — |
| Unfold (50 steps): extend by one step | 5.3 µs | — |
| linreg × 1000 particles, one at a time: generate | 17.2 ms | — |
| linreg × 1000 particles, batched: generate | 1.12 ms | — |
| linreg × 1000 particles, batched: regenerate `:slope` | 0.59 ms | — |

An earlier run, before the type-predicate fix below, compared the JITs.
Graal's JIT was about 30% faster on the scalar, allocation-heavy handler
paths: 35 µs against 47 µs for linreg simulate. C2 was about 30% faster on
the batched array kernels: 0.83 ms against 1.16 ms for 1000-particle
generate. Graal is the default.

## What batching buys

Batched importance sampling on linreg (10 observations), against one particle
at a time: 13.8× faster at 100 particles, 23.3× at 1000, and 25.6× at 10,000.
At 10,000 particles a particle costs about 1.3 µs.

## Fixed

- `choicemap?` and `generative-function?` used `satisfies?`. It is slow when
  the answer is no, and that is asked of every leaf value. Instance checks
  made scalar models about 2× faster, and spliced models 3.7×
  (hierarchical simulate: 120 µs → 33 µs).
- `erfinv` bound a `loop` as a value, which compiles to a boxing closure.

## Known hotspots (beans)

- **Scalar normal draws cost about 1.6 µs,** mostly erfinv's Newton
  refinement to full float64 accuracy. The refinement keeps draws exact
  against JAX. A double-precision initial approximation would need one step.
- **Choice gradients cost about 6× an assess** (tape allocation per op).
