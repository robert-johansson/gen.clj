# Benchmarks

Measured 2026-10-18 on an Apple M4 (32 GB), Oracle GraalVM 25.0.4, with
criterium's quick-benchmark (mean time per call). Reproduce with
`clojure -M:bench -m gen.model-bench` (add `:c2` for C2) and
`clojure -M:bench -m gen.array-bench` for per-op costs (`docs/array-api.md` §11).

## Model-level costs

| Operation | Graal JIT | C2 |
|---|---|---|
| linreg (12 sites): simulate | 17.0 µs | — |
| linreg: generate with 10 observations | 16.4 µs | — |
| linreg: assess | 14.4 µs | — |
| linreg: update one choice | 13.9 µs | — |
| linreg: regenerate `:slope` | 13.7 µs | — |
| linreg: choice gradients (slope, intercept) | 92.0 µs | — |
| hierarchical (10 splices): simulate | 31.0 µs | — |
| hierarchical: update `:mu` | 33.7 µs | — |
| Unfold (50 steps): extend by one step | 5.2 µs | — |
| linreg × 1000 particles, one at a time: generate | 16.6 ms | — |
| linreg × 1000 particles, batched: generate | 0.29 ms | — |
| linreg × 1000 particles, batched: regenerate `:slope` | 0.18 ms | — |

An earlier run, before the type-predicate fix below, compared the JITs.
Graal's JIT was about 30% faster on the scalar, allocation-heavy handler
paths: 35 µs against 47 µs for linreg simulate. C2 was about 30% faster on
the batched array kernels: 0.83 ms against 1.16 ms for 1000-particle
generate. Graal is the default.

## What batching buys

Batched importance sampling on linreg (10 observations), against one particle
at a time (both paths after the fixes below):

| particles | one at a time | batched | speedup |
|---|---|---|---|
| 100 | 9.96 ms | 1.03 ms | 9.7× |
| 1,000 | 39.0 ms | 1.53 ms | 25.5× |
| 10,000 | 176 ms | 4.27 ms | 41.1× |

At 10,000 particles a particle costs about 0.43 µs.

**Spot checks and the comparison registry** (2026-10-19) make batched runs
safe for helper functions. Spot checks re-run 8 particles of each batched run
through the scalar handler (`gen.batched/*spot-checks*`). The registry tracks
the batched comparisons nothing read. Linreg with 20 observations, warmed
up, averaged over 300 runs:

| particles | one at a time | batched, unguarded | batched, guarded | speedup (guarded) |
|---|---|---|---|---|
| 100 | 3.46 ms | 0.17 ms | 0.64 ms | 5.4× |
| 1,000 | 34.7 ms | 0.44 ms | 0.75 ms | 46× |
| 10,000 | ≈340 ms | 3.34 ms | 3.65 ms | ≈93× |

The spot checks cost a fixed 0.3–0.4 ms, which matters only for small
collections. The registry's cost is within noise, and so are the NaN checks
on distribution parameters (about 2% of a scalar generate or a choice
gradient).

## Fixed

- `choicemap?` and `generative-function?` used `satisfies?`. It is slow when
  the answer is no, and that is asked of every leaf value. Instance checks
  made scalar models about 2× faster, and spliced models 3.7×
  (hierarchical simulate: 120 µs → 33 µs).
- `erfinv` bound a `loop` as a value, which compiles to a boxing closure.
- `erfinv` now refines Giles' guess with Halley's method (cubic convergence),
  usually in one erf evaluation instead of three Newton steps. Normal draws
  went from 1.6 µs to 0.49 µs, and batched generate over 1000 particles from
  1.12 ms to 0.29 ms, 56× faster than one particle at a time. The scipy and
  JAX fixtures still hold to 1e-15.

## Known hotspots (beans)

- **Choice gradients cost about 6× an assess** (tape allocation per op).

## Against GenMLX, same hardware (2026-10-19, bean genclj-ls29)

GenMLX's `bench/cost_per_particle.cljs` and gen.clj's port of it,
`bench/gen/genmlx_bench.clj`, both on the M4:
- GenMLX: nbb on Bun, MLX through mlx-node, Metal.
- gen.clj: Oracle GraalVM 25.

The model is a normal-normal: `mu ~ N(0, 10)` and five observations
`y_i ~ N(mu, 1)`. Each point is a median after warm-up and forces its
result. Reproduce with:
- `bun run --bun nbb bench/cost_per_particle.cljs` in `~/code/genmlx`;
- `clojure [-J-Dgen.backend=mlx] [-J-Dgen.bench.spot-checks=0] -M:bench -m gen.genmlx-bench`.

**Per particle (or per step), the scalar paths:**

| | GenMLX (Metal) | gen.clj, MLX on Metal | gen.clj, JVM |
|---|---|---|---|
| generate, one particle at a time (N = 1000) | 0.93 ms | 0.46 ms | 0.020 ms |
| importance sampling (N = 1000) | 1.85 ms | 0.49 ms | 0.010 ms |
| MH, one step | 4.2 ms | 1.1 ms | 0.014 ms |
| particle filter, one particle-step (N = 500) | 1.76 ms | 0.28 ms | 0.007 ms |

**Total, the batched paths (N = 3000):**

| | GenMLX (Metal) | gen.clj, Metal, spot checks off | gen.clj, Metal, spot checks on | gen.clj, JVM (spot checks on) |
|---|---|---|---|---|
| generate | 1.4 ms | 0.8 ms | 8.3 ms | 1.2 ms |
| importance sampling | 1.4 ms | 0.7 ms | 5.7 ms | 0.9 ms |
| particle filter, 5 steps | 9.2 ms | 3.8 ms | 27 ms | 2.4 ms |
| generate, N = 10,000 | — | 1.1 ms | 6.2 ms | 1.8 ms |

How to read these:
- **Scalar paths.** gen.clj on MLX is 2–6× faster than GenMLX through the
  same library. The JVM backend is 50–300× faster again, at nanoseconds per
  op where MLX evaluates for microseconds. Scalar models belong on the JVM
  backend.
- **Batched paths.** gen.clj on Metal is about twice as fast as GenMLX
  without spot checks. With them, each batched operation also re-runs 8
  particles through the scalar handler. That costs about 0.5 ms each on
  MLX, so 4–5 ms per operation, which dominates at these sizes (on the JVM
  it is 0.3 ms). Running the spot checks on the JVM backend is bean
  genclj-w5y0.
- **Small models.** At these sizes the JVM backend's batched path matches
  Metal's. The GPU earns its place on larger arrays and models.
- **The first MLX measurement** found batched runs falling back to one
  particle at a time on Metal: the spot check compared values in float64,
  which Metal lacks. It now compares on the host, at the precision of the
  narrower float.
