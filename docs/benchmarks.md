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

## GenMLX's inference benchmarks, within gen.clj (2026-10-19)

`bench/gen/inference_bench.clj` ports GenMLX's `bench/linreg.cljs`,
`hmm.cljs`, `gmm.cljs` and `particle_scaling.cljs`. Each algorithm is timed
(median after warm-up, forcing its result), and its accuracy is measured
against an exact answer:
- linear regression: the analytic posterior;
- the HMM: the forward algorithm;
- the mixture: the means integrated out, all 3⁸ assignments enumerated.

GenMLX's own runs only measure ESS. Compared within gen.clj (GenMLX was not
re-run). Left out: NUTS (bean genclj-tvc0) and HMC's step-size adaptation;
HMC takes a fixed step at half the stability limit of the posterior's
narrowest direction. Reproduce with
`clojure [-J-Dgen.backend=mlx] [-J-Dgen.mlx.device=cpu] [-J-Dgen.bench.spot-checks=0] -M:bench -m gen.inference-bench`.

| Median time | JVM | Metal, spot checks on | Metal, checks off | MLX CPU, float32 |
|---|---|---|---|---|
| linreg (22 sites): MH, random walk, 300 steps | 47 ms | 546 ms | 536 ms | 349 ms |
| linreg: HMC, L = 10, 150 steps | 431 ms | 2226 ms | 2201 ms | 1690 ms |
| linreg: HMC compiled, 150 steps | 345 ms | 340 ms | 332 ms | 231 ms |
| linreg: batched IS, N = 10,000 | 4.4 ms | 5.7 ms | 2.4 ms | 4.9 ms |
| HMM (50 steps): scalar IS, N = 100 | 77 ms | 2241 ms | 2208 ms | 1697 ms |
| HMM: batched IS, N = 1000 | 13.2 ms | 72 ms | 21 ms | 51 ms |
| HMM: particle filter, scalar, N = 100 | 66 ms | 5024 ms | 4866 ms | 4073 ms |
| HMM: particle filter, batched, N = 100 | 12.2 ms | 162 ms | 98 ms | 127 ms |
| HMM: particle filter, batched, N = 250 | 14.1 ms | 165 ms | 97 ms | 117 ms |
| GMM (8 points): scalar IS, N = 200 | 13.7 ms | 851 ms | 819 ms | 639 ms |
| GMM: batched IS, N = 10,000 | 13.3 ms | 15.7 ms | 6.3 ms | 13.7 ms |
| scaling: batched IS, N = 100 | 0.18 ms | 1.49 ms | 0.80 ms | 0.82 ms |
| scaling: batched IS, N = 10,000 | 3.0 ms | 3.2 ms | 1.5 ms | 2.5 ms |

Accuracy, the mean over three keys (JVM; MLX draws other random numbers,
with errors of the same order):
- linreg, at GenMLX's sample counts: MH with its step of 0.5 (acceptance
  0.14) ends 3.2 posterior sds from the slope, HMC 0.27 sd (acceptance
  0.92). That gap is tuning and budget, not the algorithms (a side agent's
  review, 2026-10-19): HMC's step came from the exact posterior, and an MH
  step costs about a twentieth of an HMC step. Given HMC's time (4000
  samples after 2000), MH comes within 0.07 sd with GenMLX's step and
  0.18 sd with a step tuned like HMC's, 2.38/√2 times the posterior's
  narrowest sd (acceptance 0.04 and 0.54). A tuned MH chain at GenMLX's
  200 samples never reaches the posterior from a prior draw (155 sd off):
  its small steps need the longer burn-in.
- linreg, batched IS: 2.5 sd at N = 1000 (ESS 1.0), 0.21 at 10,000 (ESS 3.6).
- HMM: IS collapses (ESS 1–1.6; log-ML off by 53 at N = 100 and 32 at
  1000). The particle filter is within 0.25–0.4.
- GMM: IS collapses too (ESS 1.2–1.4; log-ML off by 28, 12 and 4.0 at
  N = 200, 1000 and 10,000).

How to read these:
- **At GenMLX's model sizes the JVM backend is fastest or level almost
  everywhere.** MLX wins only with compile (HMC on MLX's CPU: 231 ms
  against 345) and on the largest batched runs with spot checks off
  (about 2× at N = 10,000; with them, the default, the JVM is level).
- **Scalar paths on MLX cost 12–76× the JVM's** (MH 12×, the particle
  filter 76×), from MLX's per-op and per-evaluation costs.
- **Particle scaling.** From 100 to 10,000 particles the time grows 2×
  on Metal, 3× on MLX's CPU and 17× on the JVM backend.
- **The batched particle filter on MLX takes 7–13× the JVM's,** 2–3 ms
  per step on a two-state model (bean genclj-j1le).
- **Spot checks cost by the choice.** With 100 choices (the HMM) they add
  51 ms to a batched IS on Metal; with 6 (the cost-per-particle model),
  0.7 ms (bean genclj-n9ow).

## CUDA on the Jetson AGX Thor (2026-10-20, bean genclj-ls29)

The same two benchmarks on the Thor's GPU (sm_110, MLX v0.32.2 built from
source, float32), at 7751b76. Two CPU JVMs of the night's verification
(MLX's CPU backend in float64) ran beside them on the Thor's 14 cores, so
host-side costs may be somewhat inflated. Metal's columns are from the
tables above and below.

| Median time | CUDA, checks on | CUDA, checks off | Metal, checks on | Metal, checks off |
|---|---|---|---|---|
| linreg: MH, random walk, 300 steps | 2164 ms | 2819 ms | 546 ms | 536 ms |
| linreg: HMC, L = 10, 150 steps | 12648 ms | 13114 ms | 2226 ms | 2201 ms |
| linreg: HMC compiled, 150 steps | 1615 ms | 1583 ms | 340 ms | 332 ms |
| linreg: batched IS, N = 10,000 | 16.4 ms | 8.7 ms | 5.7 ms | 2.4 ms |
| HMM: scalar IS, N = 100 | 10710 ms | 9615 ms | 2241 ms | 2208 ms |
| HMM: batched IS, N = 1000 | 221 ms | 125 ms | 72 ms | 21 ms |
| HMM: particle filter, scalar, N = 100 | 18505 ms | 18079 ms | 5024 ms | 4866 ms |
| HMM: particle filter, batched, N = 100 | 834 ms | 399 ms | 162 ms | 98 ms |
| GMM: scalar IS, N = 200 | 4048 ms | 3224 ms | 851 ms | 819 ms |
| GMM: batched IS, N = 10,000 | 40 ms | 20 ms | 15.7 ms | 6.3 ms |
| scaling: batched IS, N = 10,000 | 11.2 ms | 5.9 ms | 3.2 ms | 1.5 ms |

MCMC steps (per step, plain → compiled):

| Per step | CUDA | Metal |
|---|---|---|
| HMC, N = 100 | 42.8 → 6.6 ms | 12.2 → 1.5 ms |
| HMC, N = 10,000 | 28.1 → 6.1 ms | 9.0 → 1.6 ms |
| HMC, N = 1,000,000 | 59.5 → 7.0 ms | 34.9 → 5.9 ms |
| MALA, N = 100 | 8.6 → 5.5 ms | 1.6 → 1.3 ms |
| MALA, N = 10,000 | 6.3 → 3.6 ms | 2.2 → 1.3 ms |
| MALA, N = 1,000,000 | 9.6 → 6.0 ms | 6.1 → 4.6 ms |

- **At these sizes the Thor's GPU is 2.5–6× slower than Metal** on the
  inference benchmark, scalar and batched paths alike: the costs are per
  operation and per evaluation (kernel launches, MLX's CUDA backend on one
  evaluation thread, graphs off), not arithmetic. The gap narrows with
  size: at a million observations plain HMC is 1.7× Metal's time and
  compiled HMC 1.2× (7.0 ms against 5.9).
- **Compile helps CUDA most:** HMC 4.6–8.5×, MALA 1.6–1.8×, as it fuses
  away launches.
- **Plain and compiled steps disagree** in two of the six MCMC lines
  (MALA at N = 10,000, HMC at a million: "same move: false"), where Metal
  agreed in all: compiled CUDA kernels round their reductions differently,
  so a near-threshold acceptance can flip. Not a correctness failure of
  either kernel, but worth a look (with the owner's Thor patches, which
  include per-thread encoders and graph limits).

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

**On MLX's CPU device** (same benchmark, `-J-Dgen.mlx.device=cpu`, with
`-J-Dgen.mlx.dtypes=64` for float64), plain → compiled:

| Per step | MLX CPU, float32 | MLX CPU, float64 |
|---|---|---|
| HMC, N = 100 | 9.3 → 1.1 ms | 19.6 → 17.4 ms |
| HMC, N = 10,000 | 6.2 → 1.0 ms | 8.6 → 5.5 ms |
| HMC, N = 1,000,000 | 34.9 → 7.5 ms | 69.6 → 17.2 ms |
| MALA, N = 100 | 1.7 → 0.7 ms | 6.2 → 4.2 ms |
| MALA, N = 10,000 | 1.5 → 0.8 ms | 4.8 → 12.6 ms |
| MALA, N = 1,000,000 | 6.6 → 5.3 ms | 17.2 → 11.1 ms |

- **MLX's CPU device in float32 is about as fast as Metal** at these sizes,
  and faster at N = 100, where there is no GPU dispatch.
- **Against the JVM backend, in float64:** MLX's CPU device is 2.4–7×
  faster at a million observations, the yardstick for the JVM backend's
  large-array kernels (bean genclj-exha). For small models the JVM backend
  wins: plain MALA at N = 100 takes 0.42 ms against 6.2 ms.
- **Unexplained:** some compiled float64 steps are slower than expected
  (MALA at N = 10,000: 12.6 ms compiled against 4.8 ms plain; HMC at
  N = 100 barely gains). Bean genclj-h6uu.
**Native gradients against the tape** (`-J-Dgen.mlx.grad=native`, the same
benchmark, one run each), per step at a million observations, plain →
compiled:

| | Metal, tape | Metal, native | MLX CPU, tape | MLX CPU, native |
|---|---|---|---|---|
| HMC | 35.3 → 5.9 ms | 25.5 → 7.6 ms | 34.2 → 7.2 ms | 26.5 → 9.0 ms |
| MALA | 5.8 → 4.7 ms | 6.0 → 2.8 ms | 5.9 → 5.6 ms | 7.4 → 3.1 ms |

At N = 10,000 native is a little faster nearly everywhere; on the scalar
linear regression of the inference benchmark (22 sites), plain HMC takes
1918 ms native against 2201 ms with the tape, compiled 317 against 335 ms.
Mixed, so the tape stays the default (docs/array-api.md §8).

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
