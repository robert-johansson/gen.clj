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

GenMLX's figures are from one run on the morning of 2026-10-19 and were not
re-measured. gen.clj's were re-measured in the evening, after the memory
bounds (§13 of `docs/array-api.md`) and the spot-check fix below.

**Per particle (or per step), the scalar paths:**

| | GenMLX (Metal) | gen.clj, MLX on Metal | gen.clj, JVM |
|---|---|---|---|
| generate, one particle at a time (N = 1000) | 0.93 ms | 0.48 ms | 0.018 ms |
| importance sampling (N = 1000) | 1.85 ms | 0.49 ms | 0.010 ms |
| MH, one step | 4.2 ms | 1.15 ms | 0.014 ms |
| particle filter, one particle-step (N = 500) | 1.76 ms | 0.29 ms | 0.007 ms |

**Total, the batched paths (N = 3000):**

| | GenMLX (Metal) | gen.clj, Metal, spot checks off | gen.clj, Metal, spot checks on | gen.clj, JVM (spot checks on) |
|---|---|---|---|---|
| generate | 1.4 ms | 0.8 ms | 1.5 ms | 1.0 ms |
| importance sampling | 1.4 ms | 0.7 ms | 1.3 ms | 1.0 ms |
| particle filter, 5 steps | 9.2 ms | 4.0 ms | 5.4 ms | 2.3 ms |
| generate, N = 10,000 | — | 0.9 ms | 2.1 ms | 1.9 ms |

How to read these:
- **Scalar paths.** gen.clj on MLX is 2–6× faster than GenMLX through the
  same library. The JVM backend is 25–80× faster again, at nanoseconds per
  op where MLX evaluates for microseconds. Scalar models belong on the JVM
  backend.
- **Batched paths.** gen.clj on Metal is about twice as fast as GenMLX
  without spot checks. With them (8 particles re-run through the scalar
  handler), it matches GenMLX, which has no such checks, and the filter
  stays ahead. On MLX the re-runs are on the JVM backend: the batched trace
  is read once, and each re-run takes microseconds (bean genclj-w5y0). On
  MLX itself, they cost 0.5 ms each, 4–5 ms per operation: the first
  measurement (8.3 ms for generate at 3000).
- **Small models.** At these sizes the JVM backend's batched path matches
  Metal's. The GPU earns its place on larger arrays and models.
- **The first MLX measurement** found batched runs falling back to one
  particle at a time on Metal: the spot check compared values in float64,
  which Metal lacks. It now compares on the host, at the precision of the
  narrower float.
- **The evening re-measurement** found the runs without spot checks as slow
  as those with them (1.4, 1.3 and 5.3 ms). Bisected to 968580a: the checks
  read every particle on the host even with no particle to check. They now
  read nothing then. The memory bounds were ruled out first: turning off the
  collection trigger, or the cache and memory limits, changed nothing.

## MCMC steps, plain and compiled (2026-10-19)

`bench/gen/mcmc_bench.clj`: Bayesian linear regression with N observations
held as one array and two latents, `w` and `b`. HMC takes 10 leapfrog steps.
Mean per step over 20 steps after warm-up, each forcing its score; every
line also checks that the plain and compiled kernels make the same move.
Reproduce with `clojure [-J-Dgen.backend=mlx] -M:bench -m gen.mcmc-bench`.

| Per step | Metal, plain | Metal, compiled | JVM, plain | JVM, compiled |
|---|---|---|---|---|
| HMC, N = 100 | 12.2 ms | 1.5 ms | 5.0 ms | 2.7 ms |
| HMC, N = 10,000 | 9.0 ms | 1.6 ms | 3.7 ms | 3.2 ms |
| HMC, N = 1,000,000 | 34.9 ms | 5.9 ms | 168 ms | 126 ms |
| MALA, N = 100 | 1.6 ms | 1.3 ms | 0.42 ms | 0.45 ms |
| MALA, N = 10,000 | 2.2 ms | 1.3 ms | 0.61 ms | 0.57 ms |
| MALA, N = 1,000,000 | 6.1 ms | 4.6 ms | 27 ms | 26 ms |

- **Compile on the GPU.** It pays off for HMC at every size (6–8×), and
  MALA gains 1.2–1.7×.
- **The compiled kernel also does less work.** At each leapfrog step the
  plain kernel updates the trace and takes its choice gradients; the
  compiled one differentiates the model's `assess` as a function of the
  values. So it is faster on the JVM backend too, where compile is the
  identity.
- **The memory trigger's cost.** MLX collects after every sixteenth of RAM
  of new arrays (`gen.mlx.gc-new-bytes`, 2 GB here). Plain HMC at a million
  observations makes enough for a collection every second step, about 13 ms
  each: 29.7 ms per step with the trigger off, 34.9 ms with it (18%).
  Compiled steps make few new arrays and do not notice (5.8 against
  5.9 ms). Raising `gen.mlx.gc-new-bytes` trades memory for fewer pauses.
- **Before the memory bounds** (5 steps each, same model): HMC at a million,
  31.6 → 7.1 ms; MALA, 8.9 → 4.2 ms. Other changes since have made most
  paths faster, so the trigger's cost shows only in the comparison above.
