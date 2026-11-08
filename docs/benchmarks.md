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

## Against Gen.jl, same machine (2026-10-21, bean genclj-h6uu)

gen.clj's dynamic language against Gen.jl's dynamic DSL, on the same
models and operations as `gen.model-bench` and the scalar paths of
`gen.genmlx-bench`, on this Mac with nothing else running.

**Settings.**
- **Machine:** Apple M4 (10 cores: 4 performance, 6 efficiency), 32 GB,
  macOS 27.0.1. Idle: load 1.2, no other JVM or Julia process.
- **Run:** 2026-10-21, 08:38 to 08:42. One process at a time, each under
  `nice -n 10`, CPU only.
- **gen.clj:** at c392062 with this branch's per-choice section, on Oracle
  GraalVM 25.0.4 (Graal JIT), the JVM backend, float64.
  - `clojure -M:bench -m gen.model-bench`: criterium's quick-benchmark, a
    5 s warm-up, a GC, 6 samples of about 0.1 s; the mean time per call.
  - `clojure -M:bench -m gen.genmlx-bench`: 20 warm-up runs (2 if a run
    takes 50 ms or more), then the median of 7 timed runs, or 3 for scalar
    runs of 1000 particles or more.
- **Gen.jl:** 0.4.8 on Julia 1.12.6 (Homebrew), the default v1.12
  environment, whose Gen is identical to the tag 0.4.8. One thread,
  `--startup-file=no`. BenchmarkTools is not installed, so the script times
  `time_ns` loops built as the two harnesses above: a 5 s warm-up, a GC and
  6 samples of about 0.1 s, the mean per call; or 20 warm-ups and the
  median of 7 (3). The global RNG is seeded with 1.
- **The script** is `bench/genjl/genjl_bench.jl`. Rerun it with
  `nice -n 10 julia --startup-file=no bench/genjl/genjl_bench.jl` and Gen
  0.4.8 in the active environment; add `quick` for a short check that is
  not a timing run. Its models are gen.clj's line for line, 1-based where
  gen.clj's addresses are 0-based. linreg, for instance:

  ```julia
  @gen function linreg(xs::Vector{Float64})
      slope = @trace(normal(0, 10), :slope)
      intercept = @trace(normal(0, 10), :intercept)
      for (i, x) in enumerate(xs)
          @trace(normal(slope * x + intercept, 1), (:y, i))
      end
      return slope
  end
  ```

  The other models: `hierarchical`, 10 calls of a one-choice
  `observation` at `(:obs, i)`; `Unfold` of a step that traces `:x` and
  `:y`; the normal-normal model of `gen.genmlx-bench`, and its growing
  version for the particle filter. Gen.jl's operations are the GFI's own:
  `simulate`, `generate`, `assess`, `update` and `regenerate` with
  `NoChange` argdiffs, `choice_gradients`, `importance_sampling`, `mh`,
  `initialize_particle_filter` and `particle_filter_step!`.

**`gen.model-bench`'s operations, µs a call:**

| Operation | gen.clj | Gen.jl | gen.clj / Gen.jl |
|---|---|---|---|
| linreg (12 sites): simulate | 19.15 | 2.75 | 7.0 |
| linreg: generate with 10 observations | 19.68 | 3.06 | 6.4 |
| linreg: assess | 18.38 | 2.48 | 7.4 |
| linreg: update one choice | 17.30 | 7.43 | 2.3 |
| linreg: regenerate `:slope` | 16.66 | 5.85 | 2.8 |
| linreg: choice gradients (slope, intercept) | 117.33 | 20.62 | 5.7 |
| hierarchical (10 calls): simulate | 38.55 | 6.01 | 6.4 |
| hierarchical: update `:mu` | 44.37 | 18.06 | 2.5 |
| Unfold (50 steps): extend by one step | 6.64 | 1.10 | 6.0 |
| linreg × 1000, one at a time: generate | 19,820 | 3,187 | 6.2 |
| linreg × 1000, batched: generate | 488 | — | — |
| linreg × 1000, batched: regenerate `:slope` | 405 | — | — |

Gen.jl has no batched execution. gen.clj's batched generate of 1000
particles takes 488 µs, 6.5 times faster than Gen.jl's 1000 one at a time.

**The cost per traced choice:** linreg with K observations, so K + 2
choices, µs a call (µs a choice):

| K | gen.clj simulate | Gen.jl simulate | gen.clj generate | Gen.jl generate |
|---|---|---|---|---|
| 10 | 19.63 (1.64) | 2.66 (0.22) | 19.31 (1.61) | 3.25 (0.27) |
| 100 | 162.8 (1.60) | 18.00 (0.18) | 159.2 (1.56) | 21.72 (0.21) |
| 1000 | 1708 (1.71) | 190.7 (0.19) | 1854 (1.85) | 236.8 (0.24) |

Where the time goes, as far as these numbers tell:
- **Per choice.** Each added choice costs about the same from 10 to 1000
  choices, a little more past 100 on both sides:
  - simulate, from K = 10 to 100: 1.59 µs a choice in gen.clj against
    0.17 µs in Gen.jl (9.3 times); from 100 to 1000, 1.72 against 0.19
    (8.9 times);
  - generate: 1.55 against 0.21 (7.6 times), then 1.88 against 0.24 (7.9
    times).
- **Fixed cost.** The line through K = 10 and 100 leaves 0.5 to 0.8 µs a
  call on either side (simulate and generate). So neither has a fixed cost
  that matters here: the gap is in the work done for each traced choice.
- **By operation.** The gap is largest for the operations that build a
  trace from its choices (simulate, generate, assess: 6.4 to 7.4 times) and
  for choice gradients (5.7). It is smallest for those that revise a trace
  (update and regenerate: 2.3 to 2.8 times), where Gen.jl costs more than
  its own simulate (7.4 and 5.9 µs against 2.8) and gen.clj less (17.3 and
  16.7 against 19.2). The data do not say which steps of a traced choice
  cost what: that needs a profile of gen.clj's handler.

**`gen.genmlx-bench`'s scalar paths, ms a particle (a unit for MH and the
filter), at the largest N:**

| | gen.clj | Gen.jl | gen.clj / Gen.jl | gen.clj batched |
|---|---|---|---|---|
| generate (N = 3000) | 0.0096 | 0.0012 | 8.0 | 0.0005 |
| importance sampling (N = 1000) | 0.0100 | 0.0012 | 8.3 | 0.0004 |
| MH, one step (20 chains of 10) | 0.0141 | 0.0031 | 4.5 | — |
| particle filter, a particle-step (N = 500) | 0.0069 | 0.0020 | 3.5 | 0.0002 (N = 1000) |

At small N gen.clj's scalar rows also carry its harness's JIT warm-up
(0.26 ms a particle at N = 1 for generate), where Julia compiled at the
first call. gen.clj's batched paths are 2.4 to 10 times faster than
Gen.jl's scalar ones from about 1000 particles.

**What it means.** On scalar models, Gen.jl's dynamic DSL builds a trace
6.4 to 7.4 times faster than gen.clj's dynamic language (7.6 to 9.3 times
for each added choice). It is 5.7 times faster at choice gradients, and
2.3 to 2.8 times at updates. gen.clj's batched execution, which Gen.jl
lacks, more than closes the gap for many particles. The static language
(`gen ^:static`, above) speeds up assess 1.9 to 2.6 times. The rest of the
gap is the dynamic handler's per-choice cost, the target for the CPU-path
work (genclj-4n51). The gen.clj figures here are about 12% slower than the
table at the top (19.2 against 17.0 µs for linreg simulate, 2026-10-18):
other code has landed since, and the conditions differ.

### Where gen.clj's 1.7 µs a choice goes (JDK Flight Recorder)

**Setup.**
- linreg simulate at K = 1000 (1002 choices) on the JVM backend, same key
  each call.
- A 10 s warm-up, then a JFR recording with the `profile` settings over
  40 s of back-to-back simulates on one thread. That gave 22,492 simulates
  at 1,779 µs each (1.78 µs a choice), 3,110 execution samples and 11,348
  allocation samples.
- One JVM, nice'd, beside another agent's MLX run (2026-10-21 08:45). The
  proportions hold; absolute times may move.
- Separately, the thread's allocated bytes (ThreadMXBean) were counted over
  a run: 5,520,272 bytes a simulate at K = 1000, **5.5 KB a choice** (4.9 KB
  at K = 10, 5.1 KB at K = 100).
- Each sample is attributed to the innermost gen.clj function on its stack
  that names a phase. gen.array's operators are see-through, so their work
  counts for whatever called them.

| phase | time (samples) | allocation |
|---|---|---|
| **the handler's bookkeeping** | **50.6%** | **66.2%** |
| choicemap: `set-choice` into the run's choices | 13.7% | 21.6% |
| `record`: the state's `:scores` (a map by path) and `:score` | 13.5% | 21.3% |
| the trace function: the `:visited` set, the state | 10.8% | 15.5% |
| `check-address!`: used paths, their prefixes, the choices under them | 7.2% | 2.7% |
| `site-key`: the state's `:key` | 3.3% | 5.1% |
| the rest of the handler | 2.1% | 0.3% |
| **the distribution** (normal: its draw's shape and options, the affine transform, erfinv, log-prob, construction checks) | **28.8%** | **16.5%** |
| **random bits** (the site's key split, threefry, the JVM backend's 0-d uniform) | **17.3%** | **13.8%** |
| **the model body** (its loop and arithmetic) | 2.8% | 2.8% |
| other | 0.5% | 0.2% |

**What the profile shows:**
- **Over half the time is the handler's bookkeeping, and two thirds of the
  allocation.** For every choice the run:
  - updates its state, a hash map of 16 keys, five times: `:key`,
    `:choices`, `:scores`, `:score` and `:visited`;
  - adds the path (a vector `[:y i]`) to three persistent collections keyed
    by it: the choices, the scores and the visited set;
  - looks the path and its prefixes up again in `check-address!`.

  The hottest frames are Clojure's `PersistentHashMap` assoc, hash and
  node-copying (`BitmapIndexedNode.assoc` alone is 11% of samples), and by
  class the allocation is mostly their node arrays (`Object[]` 39%,
  `INode[]` 10%).
- **The distribution's 29% is mostly general-purpose code on two doubles.**
  - Including the random bits under it, normal's draw takes 34% of the time
    and its log-prob 6%.
  - The draw computes its shape (`elementwise-shape`, varargs and a reduce
    over the parameters' shapes: 6.0%) and an options map.
  - It then goes through gen.array's generic operators. On host numbers
    each operator classifies its operands' types first: `dtype/kind` and
    `op-class` are 4% of the samples.
  - erfinv's Halley step is only 3%.
- **Random bits, 17%:**
  - The site's key split is the smaller part: threefry's `hash2x32` is 1.5%.
  - Most is the JVM backend's uniform kernel for a 0-d draw
    (`random-uniform`: 9%, largely its own frame), and `gen.random`'s
    uniform and normal around it (4%).
- **Boxing and the GC:**
  - `java.lang.Double` is 7.3% of the bytes, about 25 boxed doubles a
    choice. `ArraySeq` (varargs) is 5%.
  - The young collections paused for 0.5 ms each, about 10 a second: under
    1% of the time. Allocation costs at the allocation sites and in memory
    traffic, not in pauses.
- **The rest is small:** the model body's own loop and arithmetic is 2.8%,
  and nothing else is large.

**Recommendations** (for the owner, ranked by expected gain; none built):
1. **The handler's state as run-local accumulators** (most of the 51% of
   the time and 66% of the allocation).
   - Keep one value per used path, for example a map from the path to
     `[value log-prob]`. It answers "visited", gives the scores, and builds
     the choices, so each choice costs one insertion where it now costs
     three, plus the lookups.
   - Hold the run's mutable parts in a record, or in transients that never
     leave the run, rather than a 16-key hash map assoc'ed five times a
     choice.
   - The transitions and their results stay as they are. Rule 1, the
     handler as the semantics, is about what they compute, and the laws
     check it. But whether run-local mutation fits rule 3 ("no hidden
     mutable state") is the owner's call.
2. **A host-scalar path for scalar distributions** (most of the 29%).
   - When every parameter is a host number and the shape is `[]`, normal's
     sample and log-prob need neither `elementwise-shape`, the options map
     nor gen.array's type dispatch: a uniform, one erfinv and a few
     primitive flops.
   - It must give the same bits as today's path: the same uniform, then
     erfinv, then the affine transform. That keeps every fixed-key test and
     fixture as it is, and they check it.
3. **A scalar uniform in gen.random** (about half of the 17%). A 0-d uniform
   could come straight from threefry's bits as a double, without the
   backend's array kernel (a `double[]` of one, `make`, the generic
   affine). The site's key split stays, as rule 2 requires.

Together these three cover about 97% of the samples, but how much each
recovers has to be measured after it is built. Gen.jl's 0.19 µs a choice
comes from a different design (a mutable trace, and one global stateful
generator rather than a key per site). Rule 2's keys are a cost gen.clj
chooses to pay: here at most the 1.5% of `hash2x32` and the split.

The profile is reproducible:
- `clojure -M:bench -m gen.choice-profile jfr out.jfr` records it;
- read it with `jfr view hot-methods` and `jfr view allocation-by-site`
  (which counts a 26 GB sample from JFR's own start-up that the phase table
  leaves out);
- `clojure -M:bench -m gen.choice-profile alloc` counts the bytes a choice.

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

## NUTS against HMC with a tuned L (2026-10-23, bean genclj-tvc0)

`bench/gen/nuts_bench.clj`: a 10-d Gaussian of scales 0.1 to 10
(log-spaced) and correlations 0.5^|i − j|, one `mvnormal` choice. Each
kernel runs Stan's warmup (`gen.inference.nuts`: 1000 iterations adapting
the step size and a diagonal metric), then draws 2000 times. A draw costs
a gradient per leapfrog step and one at its start. The effective sample
size is the smallest over the coordinates of their bulk and tail ESSs
(`gen.inference.diagnostics`). JVM backend, one run (key 1). Reproduce with
`clojure [-J-Dgen.backend=mlx] -M:bench -m gen.nuts-bench`.

| Kernel | Gradients per draw | Per effective draw | Smallest ESS | Warmup gradients | Step size |
|---|---|---|---|---|---|
| NUTS | 9.9 | 16.9 | 1164 | 25,840 | 0.48 |
| HMC, L = 1 | 2 | 160 | 25 | 2000 | 0.52 |
| HMC, L = 2 | 3 | 31.5 | 191 | 3000 | 0.45 |
| HMC, L = 4 (the best L) | 5 | 9.4 | 1065 | 5000 | 0.48 |
| HMC, L = 8 | 9 | 64 | 283 | 9000 | 0.45 |
| HMC, L = 16 | 17 | 1421 | 24 | 17,000 | 0.50 |
| HMC, L = 32 | 33 | 217 | 304 | 33,000 | 0.44 |
| HMC, L = 64 | 65 | 1851 | 70 | 65,000 | 0.43 |

- **NUTS costs about 1.8 times the best L's gradients per effective
  draw** (16.9 against 9.4), with no L to choose. Off the best L, HMC is 3
  to 197 times worse: once the metric makes the target nearly round, a
  trajectory of L steps returns near its start at some L (16 here).
- **Per gradient, a NUTS step costs about 1.4× an HMC step** on the JVM
  (1.1 ms for 6.2 steps against 0.78 ms for 7, at depth 3). JDK Flight
  Recorder puts two thirds of a transition in the gradients, under 6% in
  the tree's own logic (the criterion, the draws, the keys), and 8% in
  the trace's update.
- No divergences after the warmup.

## Gradients of every particle at once (2026-10-23, bean genclj-7dti)

`bench/gen/batched_gradients_bench.clj`: one batched gradient (the tape
over the batch tracer, docs/array-api.md §8b) against n scalar gradients,
each line after checking that the two agree. JVM backend, one run, after
warm-up. Reproduce with
`clojure [-J-Dgen.backend=mlx] -M:bench -m gen.batched-gradients-bench`.

| Gradient | n | Batched | One at a time | Speedup |
|---|---|---|---|---|
| a regression's log density (2 parameters, 50 observations) | 10 | 0.99 ms | 2.7 ms | 2.7× |
| | 100 | 0.78 ms | 7.6 ms | 9.8× |
| | 1000 | 1.9 ms | 39 ms | 20× |
| `choice-gradients` of linreg (K = 10 trace sites) | 10 | 0.58 ms | 4.5 ms | 7.8× |
| | 100 | 0.41 ms | 23 ms | 56× |
| | 1000 | 0.58 ms | 129 ms | 221× |
| an HMC step, L = 10, on linreg (`hmc-particles` against `hmc`) | 10 | 3.9 ms | 16 ms | 4.0× |
| | 100 | 3.3 ms | 124 ms | 38× |
| | 1000 | 5.0 ms | 1232 ms | 246× |

- **Through the GFI the gain grows with the trace sites,** as for
  batched importance sampling: the handler's per-site cost is paid once
  for all the particles, and the tape's per-op cost once for all their
  gradients.
- **An HMC step of the particles** adds to its L + 1 batched gradients the
  trace's remaking (`gen.batched/set-choices`) and its spot check, read at
  the next step: 8 scalar runs, a fixed cost.

## The static language's skipping (2026-10-20, bean genclj-0q3a)

An update on the JVM backend (this Mac, a Metal test run beside it) of
`(gen ^:static [xs] (let [mu (trace :mu …) noise (trace :noise …) _ (splice
:ys (comb/map point) xs (repeat mu)) _ (trace :extra (d/normal 0 noise))]
mu))` against the same body without the flag, per update:

| Observations | update :noise (the Map skipped), static / dynamic | update :mu (the Map reruns), static / dynamic |
|---|---|---|
| 10 | 62 / 153 µs | 105 / 84 µs |
| 100 | 28 / 450 µs | 347 / 377 µs |
| 1000 | 23 / 2974 µs (130×) | 2934 / 2932 µs |

What is skipped costs nothing; when everything reruns, the walk of the
graph costs about 20 µs over the plain body at this size (its env maps),
lost in the work beyond about 100 observations.

**Compiled scores** (genclj-rj78; this Mac, idle, 2026-10-20 at 566b6f2,
`clojure [-J-Dgen.backend=mlx [-J-Dgen.mlx.device=cpu]] -M:bench -m
gen.static-bench`): a static gf's assess is a generated straight-line fn of
its choices. On GenMLX's linear regression with a static kernel in its Map,
against the same bodies without the flag; each figure is the median of five
rounds after warm-up, in µs per call, each result forced:

| | JVM, N = 10 | JVM, N = 100 | Metal, N = 10 | Metal, N = 100 | MLX CPU, N = 10 | MLX CPU, N = 100 |
|---|---|---|---|---|---|---|
| assess, static / dynamic | 32 / 60 | 105 / 275 | 369 / 408 | 2195 / 2569 | 185 / 222 | 1455 / 1762 |
| choice-gradients | 108 / 132 | 932 / 1192 | 581 / 599 | 4186 / 4536 | 368 / 391 | 3087 / 3495 |
| HMC, L = 10, plain | 1355 / 1587 | 10664 / 13453 | 7390 / 7720 | 52042 / 55633 | 5548 / 5802 | 39536 / 43526 |
| HMC, L = 10, compiled | 1295 / 1595 | 10653 / 13493 | 1780 / 1778 | 7443 / 7285 | 1067 / 1043 | 4697 / 4690 |
| MALA, plain | 289 / 331 | 2317 / 2738 | 2574 / 2624 | 12699 / 13201 | 1541 / 1606 | 9323 / 9878 |
| MALA, compiled | 293 / 338 | 2358 / 2778 | 1550 / 1528 | 4655 / 4708 | 784 / 769 | 3286 / 3208 |

- The static function is never slower than the dynamic one. On the JVM its
  assess is 1.9× faster at 10 observations and 2.6× at 100, and its HMC and
  MALA steps 1.15–1.3×. An earlier measurement, beside a Metal test run,
  had plain HMC slower static than dynamic at N = 10 (3038 against 2574 µs);
  idle it is faster (1355 against 1587).
- Compiling a step pays on MLX, where it fuses the step into one graph: HMC
  4.2× (Metal) and 5.2× (MLX CPU) at N = 10, 7.0× and 8.4× at N = 100, and
  MALA 1.7–2.8×. On the JVM a step is host code already, so compiling gains
  nothing. Compiled steps trace one graph whether the function is static or
  dynamic, so the two then cost the same.
- At these sizes the JVM backend is competitive: a compiled HMC step at
  N = 100 costs 10.7 ms on the JVM, 7.4 ms on Metal and 4.7 ms on MLX's CPU.

## MLX's CPU on the Thor against the Mac (2026-10-20, bean genclj-7fzy)

The full battery on MLX's CPU took 9186 s on the Thor and 1419 s on the
Mac in float32 at 7751b76 (6.5×), and the float64 soak ran 8.9× slower per
property. Profiled on one combinator law (`combinators-identity-update-changes-nothing`,
100 cases, float32) and a per-operation microbenchmark, each machine
running one other test JVM:

| | Mac M4 | Thor | Thor, pinned to 2 cores |
|---|---|---|---|
| The law, 100 cases | 0.95 s | 6.8–7.3 s | 3.2–4.0 s |
| JVM: 1000 assocs on a small map | 37 µs | 78 µs | 81 µs |
| JVM: thread ping-pong round trip | 4.0 µs | 17.1 µs | 4.0 µs |
| MLX: one op, built (no evaluation) | 0.20 µs | 0.87 µs | 0.75 µs |
| MLX: one op inside an evaluated graph | 1.3 µs | 8.1 µs | 4.8 µs |
| MLX: one value read (evaluation round trip) | 12 µs | 26 µs | 18 µs |
| MLX: sum(exp(v)), 10⁶ elements | 0.59 ms | 8.7 ms | 9.1 ms |
| MLX: 256×256 matmul, read | 0.076 ms | 7.6 ms | 7.8 ms |

- **Where the time goes:** 85–90% of the law's time is in MLX's
  evaluation (JFR: `mlx_array_eval` under `to-host`), of graphs of tiny
  ops; a case makes about 4000 of them, most from Threefry's random bits
  (uint32 ops). The JVM's own work is under 3%.
- **Hand-offs:** MLX 0.32's CPU encoder queues two tasks per op on the
  stream's worker thread (a mutex and a condition variable): mid-run the
  worker wakes 37,000 times a second. Linux puts each wake-up on another
  idle core: unpinned, a JVM thread ping-pong takes 17 µs, pinned to two
  cores 4 µs, as on the Mac. Pinning (`taskset -c 4,5`) halves the law's
  time; background activity on other cores does the same. Not the
  clocks (busy cores at 2601 MHz, MAXN) nor deep idle (cc7 rarely
  entered).
- **What remains, pinned (about 3.5×):** a Thor core is about half an M4
  performance core (OpenSSL RSA-2048 and ECDSA P-256 signing: 2.1× and
  1.9× on one core; the JVM's map churn 2.1×), and MLX's per-op cost on
  Linux is higher still (3.7–3.8× per op, built or evaluated). All 14
  cores beat the M4's 10 by about 15% on the same OpenSSL tests.
- **Array work:** MLX's CPU SIMD is Accelerate's on macOS only (Linux
  builds `base_simd` with one lane), and the mlx-cpu wheel bundles netlib's
  reference BLAS and LAPACK: elementwise math 15× and matmul 100× slower
  than the Mac's here. These tests do little of either.
- **float64** is about 8× float32 on both machines, from gen.clj's exact
  float64 erfinv (about 1900 ops per normal draw; genclj-tftl).
- **Remedies, by bean:** fewer ops per scalar draw (Threefry on the host
  for scalar keys, genclj-eof0), on every device; the erfinv route
  (genclj-tftl). Pinning the JVM to two cores helps runs on the Thor.

## Batched enumeration (2026-10-23, bean genclj-7oc1)

`clojure -M:bench -m gen.enumerative-bench`, on the JVM backend (M4,
GraalVM 25), with other JVMs running (load average 6–8): medians after
warm-up, spot checks on. `enumerative-inference` on a linear regression
with 20 observations over a k × k grid of slopes and intercepts (every
choice given), one point at a time against `{:batched? true}` (the points
as the particles of one batched run, per-particle constraints); and
`with-exact-proposal`'s generate enumerating n digits observed through
their number (10ⁿ assignments), against `{:batched? true}`.

| grid points | one at a time | batched | speedup |
|---|---|---|---|
| 100 | 4.30 ms | 1.78 ms | 2.4× |
| 1,024 | 35.9 ms | 3.80 ms | 9.5× |
| 10,000 | 409 ms | 25.4 ms | 16× |
| 99,856 | 6.17 s | 139 ms | 44× |
| 99,856, in chunks of 10,000 | 4.81 s | 150 ms | 32× |

| assignments | one at a time | batched | speedup |
|---|---|---|---|
| 10 | 0.49 ms | 0.77 ms | 0.6× |
| 100 | 4.83 ms | 1.32 ms | 3.7× |
| 1,000 | 40.9 ms | 2.01 ms | 20× |

- **Where the batched path's time goes** (10,000 points, 17 ms under the
  same load): the grid's choice maps, about two thirds of it: checking that
  no point sets an observed choice (4.9 ms) and stacking the points into
  batched constraints (6.2 ms, after `choicemap/stack`'s host scalars were
  made one array, five times faster than stacking them one by one). The
  batched run itself, spot checks included, is the rest. A grid kept in
  stacked form (one choice map of [n] arrays) would skip both.
- **Few assignments lose:** the spot checks re-run up to 8 particles
  through the scalar handler, so `with-exact-proposal`'s `:batched?` pays
  only from some tens of assignments; it is opt-in for that reason.

## Switches by partition (2026-10-23, bean genclj-7tqw)

`clojure -M:bench -m gen.switch-bench`, on the JVM backend (M4, GraalVM
25, load average about 1.2): medians after warm-up, spot checks on. A
mixture of K components of different structures (component j makes j + 1
choices) whose index differs across the particles, batched by partition,
against one particle at a time, which is how such a model ran before (its
batched run fell back).

| K, particles | one at a time | batched | speedup |
|---|---|---|---|
| 2, 100 | 3.84 ms | 1.02 ms | 3.8× |
| 2, 1,000 | 24.8 ms | 1.71 ms | 14.5× |
| 2, 10,000 | 146 ms | 6.80 ms | 21.5× |
| 5, 100 | 1.96 ms | 0.75 ms | 2.6× |
| 5, 1,000 | 17.3 ms | 1.49 ms | 11.7× |
| 5, 10,000 | 182 ms | 9.11 ms | 20.0× |

An MH sweep of 1,000 chains on the mixture of 2 (the index and the
component redrawn together, `mh-particles`, so particles move between
components and `select` joins the two traces): 16.6 ms one at a time, 2.58
ms batched, 6.4×.

- A site run in G groups draws G·n values (each group takes its rows of
  the whole batch's draws), and each group pays the handler's per-site
  cost once: at 10,000 particles, 5 groups cost 1.3× what 2 do.

## Language models (1.3.0)

1.3.0's MLX engine: the forwards gen.llm.models.qwen3 and .qwen35, the
engine gen.llm.array-engine, and training in gen.llm.sft, .lora and .grpo.
Measured on 2026-11-07 and 2026-11-08, against llama.cpp v0.9.0 through
gen.llm.llama on the same GGUF files.

**Machines.**
- **The Mac:** a Mac mini M4, 32 GB, Metal; MLX 0.32.4 and llama.cpp's
  Metal build.
- **fenrir:** a shared x86_64 host with an RTX PRO 6000, CUDA; MLX 0.32.4
  built from source and llama.cpp's CUDA build (sm_120a). gen.clj sets
  `MLX_USE_CUDA_GRAPHS=0` and `MLX_ENABLE_TF32=0` unless the environment
  sets them. Where a source gives both, fenrir's figures are CUDA graphs
  off / on.
- **The MacBook:** a MacBook Pro M2 Max, 64 GB, Metal; MLX 0.32.4 and
  llama.cpp's Metal build. The Qwen3.6 MoE's Metal runs are its, under a
  memory guard.
- **The Jetson AGX Thor** took no 1.3.0 measurement.

**Settings stated once.**
- **Models.** Qwen3-0.6B and Qwen3-8B are Q8_0 GGUFs wherever a source
  names the quantization. Qwen3.5-0.8B is Q8_0 on the Mac; on fenrir its
  quantization is Q8_0 where a source states it. The MoE is
  Qwen3.6-35B-A3B Q4_K_M (256 experts, 8 picked a token), the same file on
  every machine.
- **dtype** is the dtype MLX computes in. The Q8_0 scales and biases are
  cast to it at load, and the KV cache holds it. llama.cpp keeps its own: a
  float16 KV cache and float16 Q8_0 scales.
- **JIT.** Stated only where a table names Graal or C2; elsewhere not
  stated.
- **Modes.** Eager runs the forward op by op over a concatenating cache.
  Compiled runs one decode step through `array/compile` over a
  static-capacity cache, either copying the cache or donating it
  (compile's `:donate`: the cache written in place).
- **Paths.**
  - Greedy decode (`gen.qwen3-bench`) reads each step's argmax on the host.
  - Through the protocol (`gen.array-engine-bench`) reads every row's
    next-token distribution and normalizes it on the host in double, as the
    Branching protocol gives it. Token SMC, grammars and the token
    generative function take this path.
  - The device's floor reads fixed tokens and evaluates once every 16
    steps, with no host round trip.
- **Caches.** Static is the dense static-capacity cache. Paged is the paged
  KV cache (`:paged?`, the engine's default from 2026-11-08). Continuous is
  the paged mode with continuous batching (`:continuous?`, opt-in).
- **Several clients need the continuous mode.** Every many-client win below
  is continuous. The paged mode alone serves concurrent clients in turn,
  near llama.cpp's speed on the Mac and under half of it on CUDA. Open a
  serving engine with `:continuous? true`.

**Reproduce.** One GPU JVM at a time. `-J-Dgen.bench.dtype=float16`
computes in float16, `-J-Dgen.bench.sections=…` picks a bench's parts,
`-M:bench:c2` runs C2, and `-J-Dgen.bench.llama-bench=<binary>` makes
`gen.qwen3-bench` run llama-bench too (`native/build-llama.sh --bench`
builds it into `native/build/llama/<platform>/bin/llama-bench`).
`-J-Dgen.bench.warm-runs=<n>` sets `gen.array-engine-bench`'s untimed
runs.

```
clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.qwen3-bench <GGUF>         # prefill, greedy decode, batches, kernels, donation
clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.array-engine-bench <GGUF>  # through the protocol: decode, token SMC, clients
clojure -J-Dgen.backend=mlx -M:bench -m gen.step-profile <GGUF>                                # one step's host time
clojure -J-Dgen.backend=mlx -M:bench -m gen.read-profile <GGUF>                                # the logits' read
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.memory-bytes=8000000000 -M:bench -m gen.paged-bench <GGUF>
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.grad=native -M:bench -m gen.sft-bench <GGUF>
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.grad=native -M:bench -m gen.checkpoint-bench [<GGUF>]
clojure -J-Dgen.backend=mlx [-J-Dgen.mlx.grad=native] -M:bench -m gen.gated-delta-bench [<Qwen3.5 GGUF>]
clojure -M:bench -m gen.gguf-bench <GGUF> [MB of each type]
clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.smc-profile <Qwen3 GGUF>   # an SMC step's parts
clojure -J-Dgen.backend=mlx -M:bench -m gen.batch-profile <Qwen3 or Qwen3.5 GGUF>              # a paged batch step's parts
clojure -J-Dgen.backend=mlx -M:bench -m gen.moe-profile <qwen35moe GGUF>                        # the MoE's step
clojure -J-Dgen.backend=mlx -M:bench -m gen.step-census <qwen35 or qwen35moe GGUF>              # a step's primitives and kernels
clojure -J-Dgen.backend=mlx -M:bench -m gen.prefill-bench <Qwen3 or Qwen3.5 GGUF>               # several prompts' chunks a forward
clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.moe-routing <qwen35moe GGUF> # top-k flips against MLX's CPU
clojure -J-Dgen.backend=mlx -J-Dgen.llama.lib=<lib> -M:bench -m gen.moe-nll <qwen35moe GGUF>     # float16's NLL against float32's
```

### Decode at batch 1, 16 and 64

#### Greedy decode on Metal

**Qwen3-0.6B Q8_0, float32.** Static capacity = prompt + steps; median of
3 runs. Batch 1 decodes after a 512-token prompt, batches 4 to 64 after 128
tokens.

| Decode | MLX eager | MLX compiled, copying | MLX compiled, donating | llama.cpp via gen.clj | llama-bench |
|---|---:|---:|---:|---:|---:|
| batch 1 | 54.3 tokens/s | 57.3 tokens/s | 75.2 tokens/s | 99.4 tokens/s | 135.1 tokens/s |
| batch 4 | 176 tokens/s | 179 tokens/s | 224 tokens/s | 370 tokens/s | — |
| batch 16 | 277 tokens/s | 312 tokens/s | 467 tokens/s | 861 tokens/s | — |
| batch 64 | 361 tokens/s | 393 tokens/s | 636 tokens/s | 999 tokens/s | — |

Source: genclj-tjvd (2026-11-07); bench/gen/qwen3_bench.clj.

- **Pipelined,** the donating step decodes 91.0 tokens/s at batch 1 (127.3
  in float16): genclj-tjvd's parenthesized figures, the next step built
  while the last runs.
- **In float16** the donating step decodes 89.2 tokens/s at batch 1, 376
  at 4 (llama.cpp 368), 628 at 16 (858) and 999 at 64 (985).
- **The llama.cpp column** reads fixed tokens and normalizes each row on
  the host, as the Branching protocol gives them.
- **put-along-axis** in place of scatter-add for the cache's writes left
  decode unchanged: 79-80 against 78-80 tokens/s at batch 1, 461-469
  against 460-465 at 16 (genclj-8rnc, 2026-11-07; dtype not stated).

**After the host-cost fix** (genclj-c7uf). A donated array's bytes now
leave MLX's fresh-bytes count, so donating steps no longer trigger
System.gc, and rows are normalized across the host pool. Batch 1 after 512
tokens, and batch 16. The dtype and the MLX step's mode are not stated (the
bench's default dtype is float32):

| Model | batch 1: MLX | batch 1: llama.cpp via gen.clj | batch 16: MLX | batch 16: llama.cpp via gen.clj |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 80.4 tokens/s | 102.7 tokens/s | 661.9 tokens/s | 806.0 tokens/s |
| Qwen3.5-0.8B Q8_0 | 78.2 tokens/s | 70.0 tokens/s | 540.7 tokens/s | 404.8 tokens/s |

Source: genclj-c7uf (2026-11-07, branch host-cost at 79dec65); bench/gen/qwen3_bench.clj.

Before the fix (main 468a65b), Qwen3-0.6B gave 75.3 against 100.2 at
batch 1 and 467.7 against 824.8 at 16; Qwen3.5-0.8B 72.3 against 82.6 and
417.9 against 411.6. Qwen3.5-0.8B's first measurement (genclj-xnx3,
2026-11-07, compiled) was 74.5 against 79.3 at batch 1 and 437 against 418
at 16.

#### Greedy decode on fenrir (CUDA)

**MLX compiled and donating, float16**, with the quantized-matmul fix for
a decode step's [B 1 K] (qmm-rows) and the host-cost fix. llama.cpp
through gen.clj on the same GGUFs. CUDA graphs off / on:

| Model | batch 1: MLX | batch 1: llama.cpp | batch 16: MLX | batch 16: llama.cpp | batch 64: MLX | batch 64: llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 221 / 225 tokens/s | 416 / 422 tokens/s | 1949 / 2181 tokens/s | 1952 / 2056 tokens/s | 6277 / 7267 tokens/s | 2884 / 2839 tokens/s |
| Qwen3.5-0.8B | 205 / 222 tokens/s | 352 / 353 tokens/s | 1970 / 2161 tokens/s | 1436 / 1402 tokens/s | 5945 / 6418 tokens/s | 1931 / 1828 tokens/s |
| Qwen3-8B | 105 / 106 tokens/s | 139 / 137 tokens/s | 720 / 814 tokens/s | 1134 / 1118 tokens/s | 2003 / 2369 tokens/s | 2058 / 2170 tokens/s |

Source: genclj-ssiv (2026-11-07 22:46, run ssiv2-130 at main 7401be1); docs/decisions.md, the generation-mode re-measure; bench/gen/qwen3_bench.clj.

The same day's earlier runs, superseded:
- **ssiv-130**, before qmm-rows and the host-cost fix. Batch 1: 283 / 292
  (0.6B) and 94 / 97 (8B), against llama.cpp's 405 and 139. The 8B at 16
  and 64 rows: 219-246, against 1135 and 2008-2069.
- **qmm-rows-130**, the matmul fix alone. 16 rows: 1647 / 1917 (0.6B) and
  623 / 696 (8B), against 2004 / 1911 and 1152 / 1147. 64 rows: 3930 /
  4122 and 1529 / 1741, against 2845 / 2697 and 2170 / 2178. The source
  says the fix made batched decode 3-7× faster.
- **The 0.6B's batch-1 figure fell** from 283 / 292 to 221 / 225 between
  ssiv-130 and ssiv2-130. genclj-9y6y traced its floor's drop to the
  machine's state, not a commit (next table).

#### The device's floor

**Batch 1, compiled and donating, fixed tokens evaluated once every 16
steps:**

| Machine, run | Model | dtype | Floor (graphs off / on where two) |
|---|---|---|---:|
| Mac | Qwen3-0.6B | float32 | 91.0 tokens/s |
| Mac | Qwen3-0.6B | float16 | 127.3 tokens/s |
| fenrir, ssiv-130 | Qwen3-0.6B | float16 | 408 / 409 tokens/s |
| fenrir, ssiv-130 | Qwen3-8B | float16 | 139 / 183 tokens/s |
| fenrir, ssiv2-130 | Qwen3-0.6B | float16 | 340 tokens/s (graphs not stated) |
| fenrir, ssiv2-130 | Qwen3-8B | float16 | 163 / 198 tokens/s |
| fenrir, 2026-11-08, at 834313a | Qwen3-0.6B | not stated | 288-298 tokens/s |
| fenrir, 2026-11-08, after the host-cost merge | Qwen3-0.6B | not stated | 345-357 tokens/s |

Source: genclj-tjvd (2026-11-07); genclj-ssiv (2026-11-07); genclj-9y6y and docs/decisions.md, "The sampling path on fenrir after genclj-9y6y" (2026-11-08); bench/gen/qwen3_bench.clj.

With graphs, the 8B's floor (183) beats llama-bench's tg64 (168). In
generation, with each token read on the host, MLX reached about 70% of
llama.cpp at batch 1 in the same run; the lead's later summary puts batch
1 at 0.5-0.75× of llama.cpp.

#### Through the protocol

**Metal, after the host-cost fix.** Settings as c7uf's baseline gives
them: a 64-token prompt forked into the batch, advanced by fixed tokens,
128 steps at batch 1 and 32 at 16, every row normalized on the host.
llama.cpp ran in the same JVM. The dtype is not stated (the bench's
default is float32):

| Model, batch | MLX static | llama.cpp via gen.clj |
|---|---:|---:|
| Qwen3-0.6B Q8_0, 1 | 92.5, 92.6 tokens/s | 93.5, 96.4 tokens/s |
| Qwen3.5-0.8B Q8_0, 1 | 73.7 tokens/s | 77.5 tokens/s |
| Qwen3-0.6B Q8_0, 16 | 430-530 tokens/s | 627-773 tokens/s |
| Qwen3.5-0.8B Q8_0, 16 | 374-404 tokens/s | 346-387 tokens/s |

Source: genclj-c7uf (2026-11-07, branch host-cost at 79dec65); bench/gen/array_engine_bench.clj.

- **The batch-16 runs were noisy:** llama.cpp itself ranged 627-829
  tokens/s.
- **Before the fix,** in the same session on main: Qwen3-0.6B 74.7
  (llama.cpp 102.5) at batch 1 and 468 (829) at 16; Qwen3.5-0.8B 68.1
  (74.5) and 362-366 (353-395). c7uf's earlier baseline run on main gave
  85.6 (93.1) and 467 (797) for Qwen3-0.6B, 66.9 (74.4) and 365 (400) for
  Qwen3.5-0.8B.
- **The first engine run** (genclj-8rnc, 2026-11-07, float32, batch 16):
  470 against llama.cpp's 816. That equals the compiled step's own 467, so
  the protocols cost nothing visible at 16 rows.
- **The paged mode** (2026-11-07, a pool of 128 blocks, the bench's
  defaults otherwise; bench/gen/array_engine_bench.clj's docstring):
  485-488 tokens/s in float32 against the static 466 and llama.cpp's 810;
  601 in float16 against the static 568.

**fenrir, MLX in float16, Graal / C2.** Rows read and normalized on the
host as gen.clj samples; llama.cpp through gen.clj. CUDA graphs not
stated:

| Model, batch | MLX before genclj-9y6y | MLX after | llama.cpp via gen.clj |
|---|---:|---:|---:|
| Qwen3-0.6B, 1 | 132 / 148 tokens/s | 216 / 213 tokens/s | 445 / 505 tokens/s |
| Qwen3-0.6B, 16 | 337 / 425 tokens/s | 841 / 864 tokens/s | 1879 / 2484 tokens/s |
| Qwen3.5-0.8B, 1 | 143 / 169 tokens/s | 174 / 182 tokens/s | 353 / 414 tokens/s |
| Qwen3.5-0.8B, 16 | 484 / 744 tokens/s | 602 / 705 tokens/s | 1570 / 1900 tokens/s |
| Qwen3-8B, 1 | 84 / 76 tokens/s | 86 / 88 tokens/s | 140 / 145 tokens/s |
| Qwen3-8B, 16 | 251 / 281 tokens/s | 475 / 470 tokens/s | 1135 / 1282 tokens/s |

Source: genclj-9y6y (2026-11-08); docs/decisions.md, "The sampling path on fenrir after genclj-9y6y" (2026-11-08); bench/gen/array_engine_bench.clj.

- **The gap.** The source states that MLX's gap behind llama.cpp on this
  path fell from 1.4-5.8× to 1.6-2.9×.
- **ssiv2-130** (2026-11-07, JIT not stated) gave 132-144 against 454-459
  (0.6B, batch 1) and 481-495 against 1409-1457 (0.8B, batch 16).
- **Host threads.** A host pool of 8 threads in place of the default 4:
  MLX 8-27% faster, and llama.cpp as much (ssiv2-130); about 20% more at
  16 rows (genclj-9y6y).

#### Where a step's host time goes

**Metal, one compiled donating step, before the host-cost fix** (main
468a65b). Qwen3-0.6B Q8_0 unless stated, float32, a static cache of
capacity 384. 256 steps after a 64-token prompt and 32 warm steps, each
step's rows read, normalized and drawn (argmax) on the host:

| Part | batch 1 | batch 16 | Qwen3.5-0.8B, batch 1 |
|---|---:|---:|---:|
| call: host wrapper | 0.164 ms | 0.313 ms | 0.219 ms |
| call: input vector | 0.007 ms | 0.015 ms | 0.012 ms |
| call: MLX's replay (mlx_closure_apply) | 0.228 ms | 0.263 ms | 0.329 ms |
| call: outputs wrapped (System.gc inside) | 1.248 ms | 12.498 ms | 0.488 ms |
| call: donation | 0.012 ms | 0.024 ms | 0.013 ms |
| encode (mlx_async_eval) | 2.800 ms | 10.631 ms | 4.025 ms |
| wait (the device's remaining time) | 6.820 ms | 12.606 ms | 7.979 ms |
| read (->floats) | 0.080 ms | 0.740 ms | 0.122 ms |
| normalize (host, double) | 0.873 ms | 3.507 ms | 1.395 ms |
| draw (argmax) | 0.042 ms | 0.561 ms | 0.064 ms |
| step | 12.29 ms | 41.19 ms | 14.67 ms |
| speed | 81.4 tokens/s | 388 tokens/s | 68.2 tokens/s |
| MLX's collections (System.gc) | 10 in 256 steps | 42 in 96 steps | 4 in 256 steps |
| the device's floor, a step | 9.32 ms | 51.3 ms | 7.64 ms |

Source: genclj-c7uf (2026-11-07); bench/gen/step_profile.clj.

**After the host-cost fix.** main and the fix alternated in one session
(Qwen3-0.6B Q8_0, float32, capacity 384):

| | main | host-cost (79dec65) |
|---|---:|---:|
| batch 1, a step | 12.4-15.0 ms | 10.4-10.8 ms |
| batch 1, speed | 66-81 tokens/s | 93-96 tokens/s |
| batch 1, call / normalize | 1.65-2.13 / 0.84-1.29 ms | 0.41-0.49 / 0.47-0.49 ms |
| batch 16, a step | 42.5 ms | 28.6-28.9 ms |
| batch 16, speed | 377 tokens/s | 555-559 tokens/s |

Source: genclj-c7uf (2026-11-07); bench/gen/step_profile.clj.

- **At batch 1 what remains is MLX's.** Encode and wait take 9.6-10.8 ms
  against a pipelined floor of 7.9-8.5 ms. The host's own work is about 1
  ms a step: call 0.4, read 0.1, normalize 0.5.
- **MLX_MAX_OPS_PER_BUFFER** at 10 or 200 did not help: 11.8 and 11.9 ms a
  step, against 10.95 ms.
- **genclj-9y6y's normalization** (2026-11-08) took the Metal profile's
  batch-16 step from 31-35 to 27-28 ms, normalize from 4.8-7.9 to 1.8-2.2
  ms. dtype not stated.
- **A compiled call** costs about 1.4-2.5 ms of host time to build
  (genclj-tjvd, Metal).

**fenrir.**
- **At 16 rows** (ssiv2-130, 2026-11-07), the host's read, normalization
  and draw took 12-37 ms a step, against a device floor of 3.4-18 ms.
- **After genclj-9y6y** (2026-11-08), Qwen3-0.6B at 16 rows under Graal:
  the read 4.9 ms, normalization 3.9 ms, the whole step 19 ms, against a
  device floor of about 5 ms.
- **At batch 1** (ssiv-130), the GPU was busy 1.6-2.0 ms of a 4.9 ms step
  (0.6B).
- **The logits' read on CUDA** (genclj-9y6y). ->floats cast float16 to
  float32 on the device. MLX's CUDA allocator then recycled the host-read
  buffer into device allocations, which page-faulted. Reading the 16-bit
  values' bits and widening them on the host (bit-identical) took the
  0.6B's read at 16 rows to 4.9 ms. It was 26.8 ms in ssiv2-130 and 28.5
  ms in genclj-9y6y's summary; the 0.8B's read was 5.2 ms. At batch 1 the
  read swung 0.57-2.18 ms.

**Row normalization** (gen.llm.host/log-softmax-rows), per entry, before →
after:

| Machine, JIT | one thread | 16 rows, a pool of 4 |
|---|---:|---:|
| Mac, JIT not stated | 4.8 → 2.0 ns | 1.30 → 0.65 ns |
| fenrir, Graal | 13 → 5.0 ns | 3.5 → 1.6 ns |
| fenrir, C2 | 2.3 ns (after) | 0.93 ns (after) |
| fenrir, Graal: the llama.cpp engine's normalization (Java) | 3.2 ns | 1.04 ns |

Source: genclj-9y6y (2026-11-08).

**Kernels per decode step** (batch 1):

| Machine, model | Step | Count |
|---|---|---:|
| Mac, Qwen3-0.6B | eager, concatenating cache | 1,268 MLX primitives (945 not views) |
| Mac, Qwen3-0.6B | eager, static cache | 1,927 MLX primitives (1,288 not views) |
| Mac, Qwen3-0.6B | compiled | 805 MLX primitives (574 not views) |
| fenrir, Qwen3-0.6B | compiled | 776 kernels (kprof) |
| fenrir, Qwen3-8B | compiled | 993 kernels (kprof) |
| fenrir, the MoE, float16 | compiled paged, batch 1, before top-k by ranks | 5,365 kernels (16 rows: 5,464) |
| fenrir, the MoE, float16 | compiled paged, batch 1, after top-k by ranks | 2,545 kernels (16 rows: 2,645) |
| fenrir, the MoE, float16 | compiled, batch 1, before / after fewer kernels | 2,574 / 2,184 kernels |
| fenrir, Qwen3.5-0.8B, float16 | compiled, batch 1, before / after fewer kernels | 746 / 632 kernels |

Source: genclj-tjvd (2026-11-07; MLX's primitives from mlx_export_to_dot, views launch no kernel); genclj-ssiv (2026-11-07); bench/gen/qwen3_bench.clj. The MoE and Qwen3.5 rows: genclj-p0ad and genclj-yesw (2026-11-08; fenrir's CUPTI kernel counters); docs/decisions.md, "The MoE on CUDA: top-k by ranks" and "Qwen3.5's decode step in fewer kernels"; bench/gen/moe_profile.clj, bench/gen/step_census.clj.

#### Qwen3.5 and the MoE at batch 1 and 16 rows

**fenrir, the MoE in float16, the router's top-k by ranks** (genclj-p0ad).
The router picked its 8 experts by 8 argmaxes, each masking the last pick:
about 3,500 of the step's 5,400 kernels, with MLX's CUDA host encoding a
kernel in about 5 µs. Ranking the experts gives the same indices and values
in about 10 kernels a layer. A compiled paged step, ms at 1 / 8 / 16 rows:

| | before | after |
|---|---|---|
| step | 25.2 / 39.4 / 52.7 ms | 17.3 / 31.4 / 44.2 ms |
| encode (mlx_async_eval) | 21.0 / 26.5 / 30.5 ms | 14.2 / 20.6 / 25.1 ms |
| GPU busy | 14.1 / 18.8 / 22.9 ms | 9.4 / 15.2 / 19.8 ms |

Tokens/s, the same GGUF:

| | before | after | llama.cpp via gen.clj |
|---|---:|---:|---:|
| batch 1 (`gen.qwen3-bench`, compiled static) | 47.9 | 79.6 | 210 (llama-bench tg64: 278) |
| 16 rows (compiled static) | 488 | 612 | 982 |
| 16 rows, the continuous engine | 314.5 | 377.6 | 1015 |

Source: genclj-p0ad (2026-11-08); docs/decisions.md, "The MoE on CUDA: top-k by ranks"; bench/gen/moe_profile.clj, bench/gen/qwen3_bench.clj, bench/gen/array_engine_bench.clj.

- **float32 is slower on CUDA:** MLX's qmm_sm80 takes half floats only, so
  float32 falls back to qmm_naive, 21.7 / 42.1 / 60.2 ms a step.
- **gather_qmm switches kernels at 8 pairs:** at batch 1 a gate's product
  runs on 32 CTAs, 41 µs a call, 4.9 ms a step, the largest GPU item.

**fenrir, float16, the engine at batch 1, before and after Qwen3.5's step
in fewer kernels** (genclj-yesw: gated delta's CUDA kernel for reads
under 64 tokens, and q and k normed together, the key heads tiled by one
copy, the attention gate read in place, the MoE's aranges shared).
Tokens/s, main and the branch alternated, runs spread about ±6%:

| Model, mode | main | branch | llama.cpp via gen.clj |
|---|---:|---:|---:|
| the MoE, continuous | 49–53 | 56–59 | 139 |
| the MoE, paged | 52–59 | 54–62 | — |
| the MoE, static | 40 | 48–51 | — |
| Qwen3.5-0.8B, paged | 136–156 | 155–161 | 255 |
| Qwen3.5-0.8B, continuous | 117–124 | 134–151 | — |
| Qwen3.5-0.8B, static | 82–84 | 133–136 | — |

Source: genclj-yesw (2026-11-08); docs/decisions.md, "Qwen3.5's decode step in fewer kernels; gated-delta on CUDA"; bench/gen/array_engine_bench.clj.

- **The MoE's static time to first token** fell from 478 to 243 ms: short
  prefill chunks take the kernel too.
- **On Metal** the change measured within noise of main.
- **llama.cpp stays about 2.4× faster at batch 1.** A simple kernel costs
  about 2.4 µs to encode on MLX's CUDA.
- **float16 against float32**, the MoE on fenrir before the change
  (genclj-fxag, 2026-11-08): batch 1 continuous 52 against 38, paged 59
  against 49 (llama.cpp 139).

**The MacBook's Metal, the MoE, float32 → float16** (2–3 JVMs):

| | gen.clj | llama.cpp's Metal build |
|---|---|---|
| decode, batch 1 | 35.9–36.5 → 39.4–39.5 tokens/s | 48.3–49.1 tokens/s |
| decode, 16 rows | 172–215 → 176–225 tokens/s | 174 tokens/s |

Source: genclj-h80d's MacBook run (2026-11-08); docs/decisions.md, "The Qwen3.6 MoE on the MacBook's Metal"; bench/gen/array_engine_bench.clj. After the loader's changes (genclj-0nxn) the engine decodes at batch 1 at 36.5 tokens/s in float32 and 40.1 in float16 there, under the strict memory guard.

**The Mac (M4), Qwen3.5-0.8B Q8_0, one compiled paged batch step**
(genclj-qu77), ms a step:

| | 1 row | 8 rows | 16 rows |
|---|---:|---:|---:|
| float32, states in place | 12.8 | 38.0 | 34.4 |
| float32, states read in place by forks | 13.0 | 39.6 | 40.8 |
| float16, states read in place by forks | 12.6 | 23.1 | 38.7 |
| Qwen3-0.6B, float32 | 10.4 | 30.4 | 26.9 |

Source: genclj-qu77 (2026-11-08); docs/decisions.md, "Qwen3.5's step on Metal: 5-15 float32 rows as 16, dead states released"; bench/gen/batch_profile.clj.

- **8 float32 rows were slower than 16:** below 13 to 15 rows on the M4,
  MLX 0.32.4 multiplies by qmv_wide, which re-reads the weights every 5
  rows. Padding 5 to 15 float32 rows to 16 (M3 and M4 only) took 8 rows
  from 38.0 to 28.7 ms, Qwen3-0.6B's from 30.4 to 22.5.
- **Held states paid for the GC:** each step made 18.9 MB of new states a
  row. Releasing the groups no slot holds (`gen.array/release!`) took
  float16 from 23.1 to 20.2 ms at 8 rows and from 38.7 to 32.8 at 16.
- **The gated-delta kernel runs at memory speed:** 0.44 ms a row (86 GB/s).
- **The rest is the host round trip:** at 8 rows in float16 a step takes
  20.2 ms against a pipelined floor of 13.0 (read 1.2, normalize 1.4, draw
  0.5, and about 3.3 ms of GPU idle while encoding).

### Prefill

**A 512-token prompt at batch 1, the Mac:**

| Model, dtype | MLX eager | MLX compiled, one chunk | MLX compiled, 128-token chunks | llama.cpp via gen.clj | llama-bench |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0, float32 | 1,560 tokens/s | 2,399 tokens/s | 1,940 tokens/s | 2,981 tokens/s | 3,020 tokens/s |
| Qwen3-0.6B Q8_0, float16 | — | 2,971 tokens/s | — | — | — |

Source: genclj-tjvd (2026-11-07); bench/gen/qwen3_bench.clj.

Qwen3.5-0.8B Q8_0, compiled (chunking and dtype not stated): 1,968 tokens/s,
against llama.cpp's 2,293 (genclj-xnx3, genclj-afy0; 2026-11-07).

**fenrir, MLX compiled in one chunk, float16, CUDA graphs on:**

| Model | MLX | llama.cpp via gen.clj | llama-bench |
|---|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 51.9k tokens/s | 52.4k tokens/s | 62.8k tokens/s |
| Qwen3-8B Q8_0 | 8.9k tokens/s | 15.5k tokens/s | 15.5k tokens/s |

Source: genclj-ssiv (2026-11-07, run ssiv-130); docs/decisions.md, "The CUDA checkpoint's result" (2026-11-07); bench/gen/qwen3_bench.clj.

- **The 8B** is 1.7× behind llama.cpp, as the source states. No later
  prefill figure is recorded.
- **Two llama-bench runs disagree.** Another run the same day gave pp512 of
  about 68.3k (0.6B) and 16.65k (8B) tokens/s (see llama.cpp's ceilings).
  The sources do not say which run is later.

### Token SMC

**The Mac, Qwen3-0.6B Q8_0, 16 particles.** `gen.llm.smc/run` of a
32-token completion (no end tokens) by the bootstrap proposal, over a new
LM of each engine; tokens/s is particles × tokens over the time. The final
code (a8cc3e5), whose copy-on-write copies a block only for a second
writer:

| dtype | MLX static | MLX paged | MLX continuous | llama.cpp via gen.clj |
|---|---:|---:|---:|---:|
| float32 | 345 tokens/s | 701 tokens/s | 751 tokens/s | 924 tokens/s |
| float16 | 649 tokens/s | 787 tokens/s | 836 tokens/s | — |

Source: genclj-6qb6 (2026-11-08); docs/decisions.md, "Paged follow-ups" (2026-11-08); bench/gen/array_engine_bench.clj.

- **The static and llama.cpp figures** are from genclj-6qb6's runs before
  the final code. llama.cpp, which keeps its own float16 KV cache, has one
  figure.
- **Copies per run** fell from 1012 to 44 with the change.
- **Qwen3.5-0.8B** (genclj-6qb6, 2026-11-08): paged 595 against static 645
  tokens/s in float32, 698 against 728 in float16; llama.cpp 276. The
  continuous mode refused Qwen3.5 then; it runs it from the same day's
  "Continuous batching for Qwen3.5/3.6" (the tables below).
- **Two SMC runs at once** (genclj-b3ow, 2026-11-07, float32, before the
  copy change): paged 596.1, continuous 799.3, llama.cpp 888.7 tokens/s.
- **Earlier single runs**, superseded (float32 unless stated):
  - genclj-8rnc (2026-11-07): static 325 against llama.cpp's 940, with 124
    stacked steps of 8.5 rows on average;
  - genclj-1daf (2026-11-07): paged 390-409, static 318, llama.cpp 771; in
    float16, paged 540 against static 619;
  - genclj-b3ow (2026-11-07): static 342.6, paged 586.3, continuous 605.1,
    llama.cpp 958.6; in float16, 637.7, 640.5, 670.5 and 900.1;
  - genclj-6qb6 before its change (d9f99bd): paged 604 and continuous 637;
    in float16, 713 and 722.
- **Run to run,** the static engine's float32 figure was 318, 325, 342.6
  and 345, and llama.cpp's ranged 771-958.6.

**The Mac and the MacBook, 2026-11-08's changes, 16 particles.** Each
change measured against main in one session, alternated; tokens/s, before
→ after:

| Change | Model, mode | float16 | float32 | llama.cpp via gen.clj |
|---|---|---:|---:|---:|
| 5-15 float32 rows as 16, dead states released (genclj-qu77) | Qwen3.5-0.8B, continuous | 871–907 → 917–937 | 724 → 756 | 280–281 |
| the same | Qwen3-0.6B, static | — | 352 → 529 | — |
| several prompts' chunks a forward (genclj-6qb6) | Qwen3-0.6B, paged | 811 → 864 | 691 → 757 | — |
| the same | Qwen3.5-0.8B, paged | 887 → 970 | 706 → 783 | — |
| shapes compiled at open (genclj-i3fp) | Qwen3-0.6B, paged | 859 → 845 | 769 → 753 | 965 |
| the same | Qwen3.5-0.8B, paged | 967 → 925 | 786 → 757 | 287 |
| MLX's buffer cache given back after open (genclj-v1kg) | Qwen3.5-0.8B, paged | 933–941 → 954–971 | 705–758 → 772–778 | — |

Source: docs/decisions.md, "Qwen3.5's step on Metal", "Paged multi-token reads", "The paged default and shapes compiled at open, as built", "The open-time compile gives back MLX's buffer cache" and "The Qwen3.6 MoE on the MacBook's Metal" (2026-11-08); bench/gen/array_engine_bench.clj.

- **The cost of compiling at open,** measured apart (genclj-v1kg, 4–5 JVMs
  a setting, medians), Qwen3.5-0.8B paged: float16 964 without the
  precompile, 939 with it, 964 with MLX's buffer cache cleared after it;
  float32 770, 753 and 775. The warm-up left 1.9 GB in MLX's buffer cache
  against its 2.1 GB limit (ram/16), so the steps' freed buffers missed it.
  Qwen3-0.6B's steps hold only 0.5–0.6 GB of cache and did not move.
- **The MoE on the MacBook** (genclj-h80d, 2–3 JVMs): paged SMC 544–548
  tokens/s in float32, 621–622 in float16, llama.cpp's Metal build 653–656.

**fenrir (CUDA), Qwen3-0.6B and Qwen3-8B, 16 particles.** Paged
attention's CUDA kernel (vLLM's v1) against the composite it replaced,
each engine in its own JVM, median of 5, C2; tokens/s, paged and
continuous cells kernel / composite:

| Model, dtype | static | paged | continuous | llama.cpp via gen.clj |
|---|---:|---:|---:|---:|
| 0.6B Q8_0, float16 | 797 | 1158 / 999 | 1202 / 989 | 2228 |
| 0.6B Q8_0, float32 | 740 | 1022 / 721 | 857 / 858 | 2214 |
| 0.6B F16, float16 | 778 | 920 / 841 | 986 / 971 | 2195 |
| 8B Q8_0, float16 | 620 | 842 / 766 | 841 / 791 | 1484 |

Source: genclj-1daf, genclj-6qb6 (2026-11-08); docs/decisions.md, "Paged attention on CUDA: vLLM's v1 through mlx_fast_cuda_kernel"; bench/gen/array_engine_bench.clj.

- **The kernel's first launch** in a process compiles it (NVRTC), about 0.8
  s a variant, then 0.9 ms.
- **The 1158** was a high sample: a later run of main gave 970 (genclj-4acj).

**fenrir, the SMC step's host path** (genclj-4acj: 16-bit reads from MLX's
buffer, the read's pages prefetched on CUDA, rows lent, a step's forks in
one call). One SMC step of Qwen3-0.6B Q8_0 in float16 took 17.1 ms paged
against llama.cpp's 7.3: the logits' read 3.1 ms (1.2 of it widening),
copying each row 1.8, the forks' hand-offs 0.6, normalization 2.2, and MLX
encoding 7.0 ms against 0.3 waiting on the GPU. Tokens/s, the median across
JVMs, main and the branch alternated (a JVM's result varies by up to 30%),
C2:

| Run | main | branch | llama.cpp, main | llama.cpp, branch |
|---|---:|---:|---:|---:|
| 0.6B float16, paged | 1008 | 1272 | 2177 | 2579 |
| 0.6B float16, continuous | 1056 | 1445 | — | — |
| 0.6B float32, paged / continuous | 834 / 957 | 983 / 972 | 2124 | 2576 |
| 8B float16, paged | 835 | 882 | 1455 | 1600 |

Source: genclj-4acj (2026-11-08); docs/decisions.md, "Token SMC on CUDA: reads from MLX's buffer, rows lent, forks in one call"; bench/gen/smc_profile.clj, bench/gen/array_engine_bench.clj.

- **llama.cpp gains too:** its LM path takes the lent rows and the
  one-call forks.
- **Under Graal:** paged 1200, llama.cpp 2132, against C2's 1272 and 2579.
- **The 8B is bound by the GPU,** 13.3 ms an evaluation against about 6.5
  for llama.cpp, whose 8-bit products are faster.

**fenrir, float16 the default for quantized files** (genclj-1vk7, after the
above): Qwen3-0.6B Q8_0 paged 1462, continuous 1517, static 1136 tokens/s,
llama.cpp 2552; the MoE paged 847, continuous 818, llama.cpp 1886. Source:
docs/decisions.md, "float16 on CUDA as built" (2026-11-08);
bench/gen/array_engine_bench.clj. Computed from these figures, token SMC on
CUDA stays 1.7 to 2.3× behind llama.cpp. On the Mac, Qwen3-0.6B's paged
SMC in float16 (845–864) is 10–12% behind llama.cpp's 965, and
Qwen3.5-0.8B's (925–970) over 3× ahead of its 287.

### Continuous batching with many clients

**The Mac, Qwen3-0.6B Q8_0, the paged KV cache.** 8 client threads at
once, each with its own LM over the shared engine, each a 64-token
completion (no end tokens) after a prompt of its own. tokens/s is clients ×
tokens over the wall time; the median latency of a read is in parentheses.

| Clients, dtype | MLX static | MLX paged | MLX continuous | llama.cpp via gen.clj |
|---|---:|---:|---:|---:|
| 8, float16 | 85.7 tokens/s (84.0 ms) | 90.2 tokens/s (81.5 ms) | 302.0 tokens/s (17.6 ms) | 102.7 tokens/s (71.9 ms) |
| 8, float32 | 77.9 tokens/s (91.5 ms) | 82.0 tokens/s (88.8 ms) | 174.8 tokens/s (35.5 ms) | 92.1 tokens/s (80.1 ms) |
| 1, float32 (9 runs) | — | 76.9 tokens/s (11.5 ms) | 81.4 tokens/s (11.3 ms) | — |
| 1, float16 | 83.5 tokens/s (10.7 ms) | 86.4 tokens/s (10.4 ms) | 90.4 tokens/s (10.2 ms) | 97.3 tokens/s (9.1 ms) |

Source: genclj-b3ow (2026-11-07); docs/research/llm-mlx-engine.md §8; docs/decisions.md, "Continuous batching, as built" (2026-11-08); bench/gen/array_engine_bench.clj.

- **llama.cpp** batches only within one call, so its concurrent clients
  take turns.
- **A continuous batch step** costs what the paged step costs at the same
  rows: 11.5 against 11.1 ms at 1 row, 30.3 against 29.9 ms at 8
  (float32).
- **A forward of 8 clients** in float32 took 38 ms: the step 27-30 ms, the
  host's rows 3.2 ms, copy-on-write's two calls 2.7 ms.
- **8 rows against 16.** In float32 an 8-row step (29.9 ms) is slower than
  a 16-row one (25.5 ms); float16's 8-row step takes 14.9 ms. The source
  attributes it to MLX's quantized matmul at 8 rows.
- **Not re-measured** after genclj-6qb6's copy change.

**After 1.3.0's overnight work (2026-11-08).** 8 clients, 64-token
completions; tokens/s, with the median or p90 read latency where the source
gives it.

| Machine, model, dtype | MLX static | MLX paged | MLX continuous | llama.cpp via gen.clj |
|---|---:|---:|---:|---:|
| Mac, Qwen3-0.6B Q8_0, f16 | — | — | 422 (p90 14.9 ms) | 99 (p90 80 ms) |
| Mac, Qwen3-0.6B Q8_0, f32 | — | — | 287 (p90 23.7 ms) | 99 (p90 77 ms) |
| Mac, Qwen3.5-0.8B Q8_0, f16 | 72 | 74 | 255–285 (p90 22–29 ms) | 69–70 (p90 114 ms) |
| Mac, Qwen3.5-0.8B Q8_0, f32 | 65 | 66 | 205–217 (p90 30.6 ms) | 66–70 (p90 112 ms) |
| fenrir, Qwen3-0.6B Q8_0, f16 | 147 | 219 | 549–555 | 501–505 |
| fenrir, Qwen3-0.6B Q8_0, f32 | 104 | 195 | 409 | 502 |
| fenrir, Qwen3-8B Q8_0, f16 | 73 | 81 | 231–236 | 144 |
| fenrir, Qwen3.6-35B-A3B Q4_K_M, f16 | — | — | 150 (median 36–38 ms) | 161 (median 45.6 ms) |

Source: docs/decisions.md's entries "Continuous batching for Qwen3.5/3.6",
"Paged attention on CUDA", "Token SMC on CUDA", "Qwen3.5's step on Metal",
"Paged multi-token reads" and "The MoE on CUDA" (2026-11-08);
bench/gen/array_engine_bench.clj. The paged and static cells are the
earliest of those runs. Later changes moved the continuous mode most.

- **On CUDA the leads hold in float16 only.** The engine's default there
  is float16 for quantized models (the owner's calls, 2026-11-08).
- **The MoE's throughput trailed** because mixed forwards retraced per new
  shape (200–430 ms each). With shapes bucketed and compiled at open, and
  the changes below, it leads (the MoE rows of the next tables).

**The parts of that day, 8 clients, 64-token completions.** Tokens/s, each
change against main in one session (alternated); the read latency's p90 or
median where the source gives it:

| Change | Machine, model, dtype | before | after | llama.cpp via gen.clj |
|---|---|---:|---:|---:|
| paged attention's CUDA kernel (genclj-1daf, genclj-6qb6); composite → kernel, continuous | fenrir, Qwen3-0.6B Q8_0, f16 | 431 | 555 | 503 |
| the same | fenrir, Qwen3-0.6B Q8_0, f32 | 375 | 409 | 502 |
| the same | fenrir, Qwen3-0.6B F16, f16 | 451 | 527 | 454 |
| the same | fenrir, Qwen3-8B Q8_0, f16 | 220 | 231 | 144 |
| the same, paged (one caller's read at a time) | fenrir, Qwen3-0.6B Q8_0, f16 | 181 | 219 | 503 |
| 5-15 float32 rows as 16, dead states released (genclj-qu77) | Mac, Qwen3.5-0.8B, f16 | 216–243 (p90 53–60 ms) | 255–268 (p90 29 ms) | 69–70 (p90 111–123 ms) |
| the same | Mac, Qwen3.5-0.8B, f32 | 154 | 205 | 66–69 |
| the same | Mac, Qwen3-0.6B, f32 | 214 | 264 | — |
| top-k by ranks (genclj-p0ad), 6 warm runs | fenrir, the MoE, f16 | 124.5 (median 45–47 ms) | 149.6 (median 36–38 ms) | 160.6 (median 45.6 ms) |
| Qwen3.5's step in fewer kernels (genclj-yesw) | fenrir, the MoE, f16 | 165–166 | 183 | 162 |

Source: docs/decisions.md, "Paged attention on CUDA", "Qwen3.5's step on Metal", "The MoE on CUDA: top-k by ranks" and "Qwen3.5's decode step in fewer kernels" (2026-11-08); bench/gen/array_engine_bench.clj.

- **The MoE in float16 against float32** on fenrir (genclj-fxag, the router
  in float32): 171 against 118 tokens/s, llama.cpp 161.
- **The MoE on the MacBook** (genclj-h80d, float32 → float16): 99–111 →
  93–118 tokens/s, the read's p90 55–61 → 50–64 ms, against llama.cpp's
  47.8 tokens/s and 161 ms.
- **The float16 default on CUDA** (genclj-1vk7), after 6 warm runs:
  Qwen3-0.6B 718 tokens/s (llama.cpp 504); 251 after one warm run, before
  the shapes were compiled at open.

**Several prompts' chunks in one forward** (genclj-6qb6), the Mac, 8
clients, 5 warm runs and 6 timed, main → branch (TTFT: the time to the
first token; admission: the share of the run spent reading prompts):

| Model, dtype | tokens/s | p90 read | TTFT p50 (max) | admission | llama.cpp: tokens/s, p90, TTFT p50 |
|---|---|---|---|---|---|
| Qwen3-0.6B, f16 | 402 → 422 | 16.8 → 14.9 ms | 215 → 210 ms (429 → 361) | 27.6 → 23.9% | 99, 80 ms, 312 ms |
| Qwen3-0.6B, f32 | 273 → 287 | 27.9 → 23.7 ms | 270 → 231 ms (479 → 428) | 22.0 → 18.5% | 99, 77 ms, 319 ms |
| Qwen3.5-0.8B, f16 | 278 → 285 | 27.8 → 22.0 ms | 296 → 277 ms (612 → 596) | 27.0 → 24.0% | 70, 114 ms, 413 ms |
| Qwen3.5-0.8B, f32 | 208 → 217 | 38.8 → 30.6 ms | 324 → 346 ms (701 → 676) | 23.5 → 21.1% | 70, 112 ms, 431 ms |

Source: genclj-6qb6 (2026-11-08); docs/decisions.md, "Paged multi-token reads: several prompts' chunks in one forward"; bench/gen/array_engine_bench.clj, bench/gen/prefill_bench.clj.

- **One forward of 8 chunks of 32 tokens** takes 84.6 ms against 150.7 ms
  as 8 forwards (Qwen3, float16); Qwen3.5 111 against 183 ms.
- **Attention is a small part of a prefill forward:** writes, gathers and
  sdpa take 8–10% of it for Qwen3-0.6B and 1–3% for Qwen3.5-0.8B, so the
  runs read through the composite, not a kernel.
- **In the paged mode** with 8 clients the median time to first token is
  22–39% lower, throughput within ±5%.

**The paged mode's parts, the Mac (MLX 0.32.4):**

| Operation (one layer, 16 rows, contexts to 1023) | float32 | float16 |
|---|---:|---:|
| paged attention, kernels-community's v1 kernel | 1.10 ms | 0.58 ms |
| sdpa over dense caches | 1.07 ms | 0.87 ms |
| the blocks gathered, then sdpa | 6.2 ms | 6.5 ms |

Source: genclj-1daf (2026-11-07); bench/gen/paged_spike.clj (e2).

**Donated writes** (Qwen3-0.6B Q8_0's geometry and weights, float32;
median of 20 calls after 3, no collections):

| Pools' total size | paged-write of 16 slots, one pool pair | the same, 28 pairs (one a layer) |
|---|---:|---:|
| 0.14 GB | 0.57 ms | 1.88 ms |
| 0.5 GB | 0.65 ms | 1.79 ms |
| 1 GB | 0.49 ms | 1.62 ms |
| 2 GB | 0.45 ms | 1.43 ms |
| 3 GB | 0.33 ms | 1.23 ms |

Source: genclj-1ovd (2026-11-08); bench/gen/paged_bench.clj.

Qwen3's compiled donating static step at 8 rows took 27.6, 27.9 and 29.1 ms
at capacities 256, 1024 and 2048 (0.47, 1.9 and 3.8 GB). genclj-1daf's
earlier "8 ms per GB of pool written" was MlxRuntime's GC trigger, before
genclj-c7uf's fix.

**A paged decode step of 16 rows with copy-on-write** (Qwen3-0.6B Q8_0,
pools of 256 blocks a layer), by the rows that copy their last block. Two
calls is what the engine keeps: the blocks gathered and evaluated, then put
by gen.backend.mlx.paged's put-blocks scatter in a donating call.

| Rows copying | float32, two calls | float32, put-along-axis's put | float32, the copy in the step | float16, two calls | float16, put-along-axis's put | float16, the copy in the step |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 22.5 ms | — | — | 18.5 ms | — | — |
| 2 | 22.8 ms | 23.6 ms | 23.4 ms | 19.2 ms | 19.5 ms | 19.1 ms |
| 16 | 25.2 ms | 25.9 ms | 24.9 ms | 21.8 ms | 21.3 ms | 20.6 ms |

Source: genclj-6qb6 (2026-11-08); bench/gen/paged_bench.clj.

The copy in the step was dropped, since token SMC measured the same either
way. An earlier run (genclj-6qb6, before put-blocks; float32, all 16 rows
copying) gave 21.9 ms with no copy, 25.7 ms in two calls and 27.0 ms in the
step.

### Opening an engine: shapes compiled at open

The paged and continuous modes bucket their shapes (rows, chunk lengths and
copy counts to powers of 2, block tables at least 256 positions wide) and
trace the common ones when the engine opens (`:precompile?`, default true:
21 to 34 shapes). **The Mac**, main → branch, alternated (main moved up to
6% between passes):

| | main | branch |
|---|---|---|
| traces after open, 8 clients / SMC | 16–27 / 5 | 0 / 0 |
| the largest read, Qwen3-0.6B f16 | 475 ms | 326 ms |
| the largest read, Qwen3.5-0.8B f16 / f32 | 603 / 734 ms | 467 / 496 ms |
| 8 clients, Qwen3-0.6B f16 / f32 (tokens/s) | 420 / 283 | 431 / 279 |
| 8 clients, Qwen3.5-0.8B f16 / f32 (tokens/s) | 303 / 227 | 295 / 224 |
| open, paged / continuous | 0.3–0.5 s | 1.8–2.6 s / 2.4–4.4 s |

Source: genclj-i3fp, genclj-17it (2026-11-08); docs/decisions.md, "The paged default and shapes compiled at open, as built"; bench/gen/array_engine_bench.clj. llama.cpp gave 107 and 73 tokens/s with 8 clients in the same runs; token SMC is in the Token SMC tables.

- **Splitting a continuous forward** into its decode rows and its chunks
  (two compiled calls, so the shapes stay few) costs 3–7 ms a mixed
  forward: about −2% tokens/s and +4–6% time to first token on Qwen3.5
  with 8 clients.
- **The 256-position floor:** four 64-token chunks on Qwen3 take 112 → 120
  ms in float16 and 132 → 151 ms in float32.
- **The MoE** (fenrir, the ~20 s model load included): paged 28 s to open,
  its 27 shapes compiled in 7.3 s (float16) or 8.4 s (float32); continuous
  34–40 s, its 34 shapes in 10–18.5 s; no traces after (genclj-fxag;
  docs/decisions.md, "The MoE in float16"). On the MacBook it opens in
  14–30 s, 0 traces after (genclj-h80d).
- **MLX's buffer cache** after the open-time compile, and what giving it
  back buys: the Token SMC section (genclj-v1kg). Its open times and time
  to first token did not move; on CUDA it is unmeasured.

### Training steps

**An SFT step's peak memory and time** (`gen.sft-bench`). Qwen3-0.6B Q8_0;
a rank-8 LoRA adapter on q, k, v, o, gate, up and down of the last 8
blocks; AdamW; vocabulary chunks of 16384; half of each row's targets
trained. The peak is past what MLX held before the step (0.76 GB: the
weights, the adapter and the optimizer's state), at three MLX memory
limits. The default limit is 67 GB on fenrir and 17 GB on the Mac. dtype
not stated.

| Machine, gradients, checkpointed | Batch | Default limit | Held + 1.25 × the tight peak | Held (tight) |
|---|---|---:|---:|---:|
| fenrir, native, yes | 4 × 256 tokens | 6.1 GB, 274 ms | 1.8 GB, 273 ms | 1.5 GB, 305 ms |
| fenrir, native, yes | 8 × 512 tokens | 24.6 GB, 968 ms | 11.7 GB, 967 ms | 9.2 GB, 988 ms |
| fenrir, native, yes | 4 batches of 4 × 256 | 5.9 GB | 2.1 GB | 1.4 GB |
| fenrir, native, no | 4 × 256 tokens | 11.4 GB | — | — |
| fenrir, native, no | 8 × 512 tokens | 32.1 GB | — | — |
| fenrir, tape, yes | 4 × 256 tokens | 4.9 GB | 2.1 GB | 1.6 GB |
| Mac, native, yes | 4 × 256 tokens | 2.2 GB, 1819 ms | 2.09 GB, 1813 ms | 1.67 GB, 2144 ms |
| Mac, native, no | 4 × 256 tokens | 3.6 GB, 1452 ms | 3.6 GB, 1420 ms | 3.3 GB, 1653 ms |
| Mac, native, yes | 4 batches of 4 × 256 | 2.2 GB | 2.09 GB | 1.67 GB |
| Mac, native, no | 4 batches of 4 × 256 | 3.8 GB | — | — |
| Mac, tape, yes (default column at 8 GB) | 4 × 256 tokens | 1.55 GB | 1.55 GB | 1.37 GB |
| Mac, tape, no (limit 8 GB) | 4 × 256 tokens | 12.3 GB | — | — |

Source: genclj-az5s (2026-11-07, fenrir); genclj-h7ru (2026-11-08, the Mac); bench/gen/sft_bench.clj.

On CUDA the measured limit (held + 1.25 × peak) keeps the checkpoints'
saving at no time cost. On Metal the default limit keeps most of it (2.2
against 2.1 GB): Metal's default limit does not keep queued kernels'
buffers, as CUDA's did. The checkpoints themselves save 3.6 → 2.2 GB
(native) and 12.3 → 1.55 GB (tape) on the Mac, at about 25% more time.

**A gradient's peak memory** (`gen.checkpoint-bench`, fenrir's CUDA,
float32, native gradients, MLX's limit at what it held before):

| Gradient | Unchecked | Checkpointed |
|---|---:|---:|
| Qwen3-0.6B Q8_0's NLL, 4 × 256 tokens | 5.6 GB, 87 ms | 1.55 GB, 120 ms |
| a stack of 16 SwiGLU layers | 5.2 GB | 2.8 GB |

Source: genclj-pnu6 and genclj-jksq (2026-11-07); docs/decisions.md, "Training's memory tools" (2026-11-07); bench/gen/checkpoint_bench.clj.

- **1.55 GB** is the decisions entry's figure; genclj-pnu6 and genclj-jksq
  round it to 1.5 GB. genclj-pnu6 states +22-38% time for the checkpoints.
- **Depth first** (MLX_BFS_MAX_WIDTH=1), the ordered checkpoint held 0.96
  GB against MLX's own mlx_checkpoint's 1.18 GB on Qwen3.
- **At a loose memory limit** both forms peak at 10-15 GB.
- **Not run on Metal.**

**Gated delta's gradient** (`gen.gated-delta-bench`, the Mac, Qwen3.5-0.8B
Q8_0). The rule alone at the 0.8B's GatedDeltaNet sizes (1 row, 16 heads,
Dk = Dv = 128, float32). The training step: the NLL of one row and its
gradient with respect to a rank-8 LoRA adapter (default targets),
checkpointed. Median of 5. Tape / native gradients:

| Tokens | Rule: kernel forward | Rule: composite forward | Rule: composite's gradient | Training step | Step, the rule replaced by a no-cost stand-in | The rule's share |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 2.4 ms | 13.2 ms | 37.2 / 13.1 ms | 1849 / 1535 ms | 1448 / 1279 ms | 22% / 17% |
| 512 | 2.2 ms | 10.2 ms | 39.8 / 22.0 ms | 3572 / 3243 ms | 2749 / 2489 ms | 23% / 23% |
| 1024 | 3.7 ms | 11.4 ms | 140.9 / 50.1 ms | 7762 / 6446 ms | 5773 / 5094 ms | 26% / 21% |

Source: genclj-8h0l (2026-11-07); bench/gen/gated_delta_bench.clj.

**Real runs** (dtype not stated):

| Run | Machine | Setting | Time | Result |
|---|---|---|---:|---|
| SFT (sft-llama-test) | fenrir | Qwen3-0.6B Q8_0; LoRA rank 8 on all 7 projections of the last 8 layers; 48 JSON-format chat examples, 16 held out; AdamW 2e-4; 24 steps | 10 s | held-out NLL 3.61 → 0.093 per trained token |
| SFT (sft-llama-test) | the Mac | the same | 28 s native, 30 s tape | held-out NLL 3.6116 → 0.0931 |
| GRPO (grpo-llama-test) | fenrir | Qwen3-0.6B Q8_0; LoRA rank 8 on the last 8 blocks; two-digit products, exact reward; 40 rounds of 8 × 8 | 5.5 min; rollouts 5.5 s of each round's 8 s | held-out accuracy 0.615 → 0.771; a round peaks at 6.4-7.6 GB |

Source: genclj-stpo (2026-11-07); genclj-h7ru (2026-11-08); genclj-waym and docs/decisions.md, "GRPO" (2026-11-08).

- **SFT checkpoints.** trainer.edn holds 94.6 MB for 1.44M parameters with
  AdamW's moments: saved in 0.7 s, loaded in 3.1 s (genclj-stpo; machine
  not stated).
- **LoRA steps.** Five AdamW steps took Qwen3-0.6B's NLL from 188.2 to 21.4
  (genclj-yc3b, fenrir) and Qwen3.5-0.8B's from 162.4 to 1.0 (genclj-8h0l,
  the Mac). Their times are not stated.

### llama.cpp's ceilings

**fenrir.** llama-bench v0.9.0 over gen.clj-src's libllama, every layer on
the GPU, flash attention auto, 5 repetitions:

| Model | pp512 | tg128 |
|---|---:|---:|
| Qwen3-0.6B Q8_0, GPU | about 68.3k tokens/s steady (65.2k ± 6.1k with the clock ramp) | 886 tokens/s |
| Qwen3-8B Q8_0, GPU | about 16.65k tokens/s steady | 169.0 tokens/s |
| Qwen3-0.6B Q8_0, CPU, 4 threads | 246 tokens/s | 42.5 tokens/s |
| Qwen3-0.6B Q8_0, CPU, 8 threads | 459 tokens/s | 58.2 tokens/s |

Source: genclj-whi8 (2026-11-07).

- **Bandwidth.** The source puts the 8B's decode at about 1.47 TB/s of
  weights read, about 80% of the card's 1.8 TB/s.
- **The CUDA checkpoint's own llama-bench runs** (genclj-ssiv, ssiv-130,
  2026-11-07) gave tg64 888 (0.6B) and 168 (8B), and a 512-token prefill
  of 62.8k (0.6B) and 15.5k (8B). They disagree with the table above, and neither source
  says which run is later.

**The Mac.** llama-bench built from the pinned llama.cpp source with tools
on, outside the repository, run by `gen.qwen3-bench` (`-p` the prompt,
`-n` the steps). Qwen3-0.6B Q8_0: prefill of 512 tokens 3,020 tokens/s,
decode 135.1 tokens/s (genclj-tjvd, 2026-11-07).

**gen.clj against the ceiling,** the same machine, model and run:

| Machine, model | Measure | llama-bench | llama.cpp via gen.clj | via gen.clj ÷ llama-bench (computed) | MLX, compiled |
|---|---|---:|---:|---:|---:|
| Mac, Qwen3-0.6B Q8_0 | prefill 512 | 3,020 tokens/s | 2,981 tokens/s | 0.99× | 2,399 tokens/s (float32, one chunk) |
| Mac, Qwen3-0.6B Q8_0 | decode, batch 1 | 135.1 tokens/s | 99.4 tokens/s | 0.74× | 75.2 tokens/s (float32, donating) |
| fenrir, Qwen3-0.6B Q8_0 | prefill 512 | 62.8k tokens/s | 52.4k tokens/s | 0.83× | 51.9k tokens/s (float16, graphs on) |
| fenrir, Qwen3-0.6B Q8_0 | decode, batch 1 | 888 tokens/s | 405 tokens/s | 0.46× | 283 / 292 tokens/s (float16, graphs off / on) |
| fenrir, Qwen3-8B Q8_0 | prefill 512 | 15.5k tokens/s | 15.5k tokens/s | 1.00× | 8.9k tokens/s (float16, graphs on) |
| fenrir, Qwen3-8B Q8_0 | decode, batch 1 | 168 tokens/s | 139 tokens/s | 0.83× | 94 / 97 tokens/s (float16, graphs off / on) |

Source: genclj-tjvd (2026-11-07); genclj-ssiv (2026-11-07, run ssiv-130); docs/decisions.md, "The CUDA checkpoint's result" (2026-11-07); bench/gen/qwen3_bench.clj.

gen.clj's llama.cpp decode normalizes every row on the host (the Branching
protocol's form). On 2026-11-08, through `gen.array-engine-bench`,
llama.cpp through gen.clj on fenrir gave 445 / 505 tokens/s (Graal / C2)
for the 0.6B at batch 1, and 140 / 145 for the 8B.

### Loading k-quant weights

**The host reader's repack** (`gen.gguf-bench`, the Mac, on
Qwen3.6-35B-A3B Q4_K_M's tensors), by the file's bytes:

| Tensor type | Repack speed |
|---|---:|
| Q4_K | 5.4 GB/s |
| Q8_0 | 7.2 GB/s |
| Q6_K | 2.6 GB/s |

Source: genclj-kqxr (2026-11-08); docs/decisions.md, "k-quants" (2026-11-08); bench/gen/gguf_bench.clj.

- **A whole Qwen3-0.6B Q4_K_M** loads on MLX at 1.2-1.4 GB/s of its file.
- **Memory.** Q6_K loads as float32, 4.9 times the file's bytes. Q4_K takes
  6 bits a weight against the file's 4.5. The 20 GB Qwen3.6-35B-A3B Q4_K_M
  needs about 28 GB in float32, 23.1 GB in float16 cast at load (next).

### Loading a model: time and memory

**The MoE on the MacBook, from a cold file cache** (genclj-0nxn: load-gguf
casts each tensor to the model's dtype as it loads, reads past the file
cache, and decodes a chunk at a time into reused buffers). Before → after:

| | float16 before | float16 after | float32 before | float32 after |
|---|---|---|---|---|
| MLX's peak | 33.7 GB | 23.1 GB | 28.4 GB | 28.4 GB |
| the process's peak RSS | 33.1 GB | 25.3 GB | 35.2 GB | 31.7 GB |
| heap used | 2.5 GB | 0.34 GB | 3.0 GB | 0.34 GB |
| file cache, before → after the load | 2.8 → 21.2 GB | 4.0 → 4.0 GB | 4.0 → 23.9 GB | 4.0 → 4.0 GB |
| worst memory pressure | warn | normal | normal | normal |
| load time | 20.7 s | 12.3 s | 16.6 s | 11.1 s |

Source: genclj-0nxn (2026-11-08); docs/decisions.md, "load-gguf's memory: past the file cache, cast at load, bounded buffers". The file cache was emptied by mmap and msync(MS_INVALIDATE) before each load.

- **The engine** then opens with the default heap under the strict guard
  (killed at the first "warn"), with 11 GB available at its peak in float32
  and 19 GB in float16; its MLX peak is 29.7 and 24.3 GB.
- **On fenrir** the float16 load's MLX peak fell from 33.7 to 23.1 GB, its
  heap from 2.2 to 0.7 GB, and the page cache it left from +20.5 to +0.1 GB.

**Load time, cold, by reader** (genclj-vn28; load-gguf with no `:dtype`,
one JVM a load, the file evicted first, median of 5). MLX's own loader
(mlx_load_gguf) reads files of F32, F16, I32, Q8_0, Q4_0 and Q4_1; the host
reader reads every type (the K-quants, BF16, Q5_0/Q5_1 only through it):

| Model, machine | mlx_load_gguf | host reader | host ÷ mlx | file cache after (mlx / host) |
|---|---:|---:|---:|---|
| Qwen3-0.6B Q8_0, the Mac | 1.18 s | 0.81 s | 0.69 | 0.64 / 0.006 GB |
| Qwen3.5-0.8B Q8_0, the Mac | 1.40 s | 0.92 s | 0.65 | 0.83 / 0.011 GB |
| Qwen3-0.6B Q8_0, fenrir | 0.40 s | 0.73 s | 1.82 | 0.64 / 0.006 GB |
| Qwen3.5-0.8B Q8_0, fenrir | 0.53 s | 0.87 s | 1.66 | 0.83 / 0.011 GB |
| Qwen3-8B Q8_0, fenrir | 3.36 s | 7.49 s | 2.23 | 8.71 / 0.006 GB |

Source: genclj-vn28 (2026-11-08); docs/decisions.md, "mlx_load_gguf's file cache evicted; the host reader slower on CUDA".

- **The host reader on fenrir's 8B:** reads 1.48 s (5.9 GB/s), decoding
  0.83 s, mlx_array_new_data 2.0 s, and about 3.2 s untimed (most likely
  the zeroed arena and its copies). Cached reads (`-Dgen.gguf.cache=true`)
  took 8.19 s.
- **mlx_load_gguf stays for its types** (the owner's 20% rule, missed on
  CUDA), and the pages its load brings into the file cache are evicted
  after it: 2–5% of its time (the Mac: 0.6B 1.20 s, 0.8B 1.47 s; fenrir:
  0.6B 0.41, 0.8B 0.51, 8B 3.45 s), the header's 6–12 MB left cached.

### Converting and quantizing models

**Conversion's time, the Mac (M4, 10 cores), on the host** (gen.llm.convert/
convert!: llama.cpp's reference quantizers, ported, a tensor at a time on
the fork-join pool; Graal JIT, a fresh JVM). Each file is byte for byte
what llama.cpp's quantizer writes from the same source.

| Source | File type | Time | File |
|---|---|---:|---:|
| Qwen3-0.6B F16 (ggml-org) | Q8_0 | 1.9-4.9 s | 805 MB |
| Qwen3-0.6B F16 | Q4_K_M | 26.7-27.4 s | 484 MB |
| Qwen3-0.6B F16 | Q4_K_M, embedding Q8_0 | 18.3 s | 600 MB |
| Qwen3.5-0.8B BF16 (ggml-org) | Q4_K_M | 18.6-26 s | 542 MB |
| Qwen3.5-0.8B BF16 | Q8_0 | 1.9 s | 834 MB |

Source: genclj-t007 (2026-11-08); test/gen/llm/convert_real_test.clj.

The K-quants' searches dominate. Computed from the table: about 56 MB/s
of the 1.5 GB F16 file for Q4_K_M, against 300-800 MB/s for Q8_0.

**What each quantization costs, Qwen3-0.6B on Metal:** KL(F16 ||
quantized) of the next-token distribution at each position of four
18-token prompts, the forward gen.llm.models.qwen3 in float32 on MLX. The
first position (one token of context) is an outlier for every
quantization, so it is shown apart.

| Weights | median | 90th percentile | largest after the first | first positions |
|---|---:|---:|---:|---:|
| Q8_0 (convert!) | 0.00054 | 0.0026 | 0.0050 | 0.058-0.093 |
| MLX 8 bits, groups of 64 (quantize) | 0.00057 | 0.0034 | 0.0059 | 0.13-0.22 |
| Q4_K_M (convert!) | 0.058 | 0.30 | 0.36 | 8.3-11.3 |
| Q4_K_M, embedding Q8_0 (convert!) | 0.051 | 0.26 | 0.34 | 8.3-11.3 |
| MLX 4 bits, groups of 64 (quantize) | 0.15 | 0.58 | 1.17 | 2.2-4.4 |
| MLX mixed_4_6 (quantize) | 0.088 | 0.38 | 1.86 | 1.5-3.2 |
| MLX mixed_3_6 (quantize) | 0.63 | 2.07 | 3.87 | 1.6-3.8 |

Source: genclj-t007 (2026-11-08, Mac mini M4, the GPU slot); the KL in
nats. MLX's layouts hold float32 scales and biases in memory (gen.array's
choice); mlx-lm's files hold float16.

### Half precision and the MoE's routing

**How often the MoE's top-k sets differ from MLX's CPU forward in
float32** (`gen.moe-routing`: 40 prompts, 1,246 positions, 40 layers, so
49,840 sets):

| Forward | Sets that differ | Gap with the CPU's experts forced |
|---|---|---|
| fenrir's CUDA, float32 | 10 (0.02%), at 4 positions of 2 prompts, each from a near-tie | within 5.4e-4 nats |
| fenrir's CUDA, float16, router in float16 | 14.5% (layer 0 1.2%, about 20% from layer 20); 78% of positions have one | within 0.24, median 0.015 |
| fenrir's CUDA, float16, router in float32 (kept) | 11.6% (the router's own share 1.9% → 1.0%) | — |
| the MacBook's Metal, float16 | 12.9% | within 0.040 (general prompts), 0.278 (digits) |

Source: genclj-fxag and genclj-h80d (2026-11-08); docs/decisions.md, "The MoE in float16: its routing flips; the carve-out lifted" and "The Qwen3.6 MoE on the MacBook's Metal"; bench/gen/moe_routing.clj.

- **The gap follows the flips:** by the number of flips at a position (0,
  1, 2, 3 or more), the median gap is 0.03, 0.11, 0.21 and 0.33 nats.
  The float32 router takes the reference statistic's median from 0.17–0.58
  to 0.12–0.36 nats; the largest stays about 6.
- **Quality is float32's** (`gen.moe-nll`, 4,080 tokens): float16's NLL
  minus float32's is −0.0039 ± 0.0020 nats a token on CUDA (−0.0025 ±
  0.0014 with the router in float16), −0.0025 ± 0.0015 on the MacBook.
- **Dense models in float16** move about 0.02 nats (median) from MLX's CPU
  in float32 on CUDA (genclj-1vk7; bench/gen/cuda_reference_noise.clj).
- **Against llama.cpp on its own experts** (genclj-ansx; llama.cpp's
  choices recorded through its eval callback, ours forced to them), max
  |Δ log p| general / digit prompts: on Metal 0.97 / 5.97 unforced, 6.4e-3 /
  8.0e-2 forced (band 0.16); on CUDA 2.19 / 6.24 and 0.45 / 1.91 (band
  3.9), where llama.cpp's CUDA build rounds activations to 8 bits.

### Not yet measured

The open questions are genclj-pnw2's, under the 1.3.x follow-ups
(genclj-sgue).

- **The Thor:** nothing in 1.3.0, its power mode included; its bands
  (sm_110) fall under fenrir's unmeasured.
- **Larger and other models:** a dense model larger than Qwen3-8B, MLX
  4-bit weights at speed against GGUF Q4_K_M, and decode on k-quant files
  other than the Qwen3.6 MoE.
- **MLX's own ceiling** outside the JVM (mlx-lm), and mlx-node on the same
  weights.
- **Batch 4 on fenrir.** genclj-ssiv ticks it, but no figure is recorded.
  Batch 64 through the protocol, on either machine. llama.cpp with CUDA
  graphs off on fenrir.
- **Batch-1 generation on fenrir after 1.3.0's speed work,** greedy
  (`gen.qwen3-bench`), and prefill there: the figures above predate
  2026-11-08's changes (genclj-c7uf's leftovers).
- **Token SMC with 64 particles**, anywhere (genclj-c7uf).
- **The KV fork cost and the full-logits overhead per token;** generate,
  assess and update over token sites against the raw engine call, on a real
  model.
- **Long contexts:** prefill of 2k-8k tokens, and paged attention's v2
  kernel past v1's contexts, at speed (genclj-afy0 for Qwen3.5's long
  prefill).
- **On CUDA:** giving back MLX's buffer cache after the open-time compile
  (genclj-v1kg), and the float16 default's band against llama.cpp at the
  engine's level (array-engine-llama-test runs float32 there).
- **The M3/M4 row padding elsewhere:** the M2, the M5 and an Ultra, whose
  quantized products' limits differ (genclj-qu77).
- **checkpoint_bench on Metal** (genclj-pnu6, genclj-jksq).
- **The shared memory budget at real scale,** llama.cpp and MLX together
  (genclj-h80d).
- **GenLM's evaluation domains** (ext/llm-eval).
