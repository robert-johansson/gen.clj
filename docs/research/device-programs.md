# Device programs: the fastest gen.clj on CUDA

*Research note, 2026-10-26, bean genclj-a1hu, for the owner's review.
Measured on fenrir (RTX PRO 6000 Blackwell Workstation, sm_120, driver
595.71.05, CUDA 13) and the M4 Mac. The prototype is on branch
`device-programs`, in `ext/device`, outside the core. GenJAX's numbers
are GenMLX's parity suite's, measured on the same card and driver
(`~/code/genmlx/docs/genjax-parity.md`, 2026-07; specs in fenrir's
`~/mlx/genjax-parity/specs`). The owner approved building the prototype
alongside this note (DESIGN.md §9, 2026-10-26).*

## Summary

- **gen.clj on the RTX PRO 6000 is 300–700× slower than GenJAX today**
  (60–140× with `:compile?`), and the card does almost nothing of use.
  On the parity workloads (Bayesian linear regression: IS, MALA, HMC,
  many-chain MALA) every step costs MLX's op count times 8–17 µs, plus
  host reads. CUPTI shows 550–4,600 kernels per step of 1.5–6 µs each,
  every one nearly empty, and the host busy launching and waiting (§1).
- **The same workloads as one device program each run at GenJAX's speed.**
  The prototype stages gen.clj's own code (the handler's generate and
  assess, the tape's gradients, the batch tracer, `mcmc`'s moves,
  gen.random's Threefry) into a graph, loops it on the device and
  compiles it with XLA through PJRT. That is 190–440× faster than MLX
  today on the chains (36× on IS), at parity with GenJAX for MALA and
  8-step HMC, within 1.6–3× for 32-step HMC and many chains, with the
  handler's draws to float32 rounding (§3).
- **The graph is small and general:** a staging backend under the tape
  and the batch tracer, keys that can be device values, and a loop (`scan`).
  Nothing about a model or a kernel is written twice (rule 1). XLA is one
  lowering of it; generated CUDA C (one thread per chain, no per-step
  launch) is another (§4).
- **Generated CUDA C is faster still,** 14–18× GenJAX on single chains
  and 5.5× on 4096 (15× counting the kernel alone): one thread per chain,
  the whole run one kernel, from the same staged program, with the
  handler's draws (bit-identical for the static model) (§4).
- **Recommendation** (§5): take the staging backend, staged keys and
  `scan` into the core as L0; make kernels' moves public as functions of
  arrays and keys, with one entry point that runs a whole inference as a
  program; generated CUDA C first (no new dependency), XLA via PJRT
  second (for large arrays, at the cost of its plugin); MLX stays for
  Metal and interactive work.

## 1. Where the time goes today

### The parity workloads

`gen.parity-bench` runs the specs' cells on gen.clj's public API: a
constrained generate from the prior, then `kernel/run` of S steps (or
`mala-particles` over N chains, or batched `importance-sampling`), the
slope read each step, from a key to host numbers; the first of three
warm-ups reported apart, then up to 20 timed runs. MALA's tau is
GenJAX's ε²/2 (its "step size" is the proposal's sd). Acceptance and
slope tails agree with GenJAX's and the exact posterior's at S=1000. The model is the specs'
ten observations in three forms: one site per observation (dynamic,
`:sites`, the specs' form), the same in the static language (`:static`),
and one [10] site (`:vector`, gen.clj's idiom).

Medians, the vector model, ms:

| cell | GenJAX (XLA) | MLX CUDA | MLX `:compile?` | JVM backend (fenrir's CPU) | gen.clj device program (XLA) |
|---|---|---|---|---|---|
| IS N=1 | 0.018 | 4.25 | — | 0.105 | 0.55 |
| IS N=30000 | 0.019 | 11.2 | — | 11.6 | 0.31 |
| MALA S=10 | 0.21 | 56 | 26 | 1.5 | 1.5 |
| MALA S=100 | 1.62 | 454 | 217 | 12 | 2.0 |
| MALA S=1000 | 15.6 | 4847 | 2239 | 106 | 13.1 |
| HMC L=8 S=10 | 0.36 | 144 | 34 | 4.3 | 0.58 |
| HMC L=8 S=100 | 3.67 | 1498 | 338 | 42 | 5.4 |
| HMC L=8 S=1000 | 32.6 | 15 570 | 3245 | 430 | 35.2 |
| HMC L=32 S=10 | 1.06 | 478 | 59 | 15 | 2.7 |
| HMC L=32 S=100 | 10.5 | 4936 | 602 | 153 | 18.2 |
| HMC L=32 S=1000 | 100.8 | 48 936 | 5984 | 1541 | 164.5 |
| MALA, 8 chains, S=100 | 1.54 | 963 | — | 33 | 3.2 |
| MALA, 4096 chains, S=100 | 1.57 | 829 | — | 460 | 6.2 |
| MALA, 8 chains, S=1000 | 11.2 | 7576 | — | 327 | 15.0 |
| MALA, 4096 chains, S=1000 | 13.2 | 7666 | — | 4568 | 40.7 |

(Raw EDN: fenrir `~/mlx/gen-clj-logs/a1hu/{mlx-0701,jvm-0735,device-0748,device-0750}.edn`;
the static and sites forms are there too. On MLX the static form costs
1.5–2.4× the vector form, on the JVM backend 1–2.3×. The device
program's S=10 cells are its per-call floor and noisy, 0.4–1.5 ms
between runs; IS is from the first device run, the chains from the
second, after ae99b50.)

### What MLX's time is

- **Op count.** gen.parity-bench counts MLX arrays made per step: a MALA
  step makes 288 (68 compiled), an HMC step 1200 at L=8 and 4270 at L=32
  (84–97 compiled), the static form about 2.5× as many. Time per step is
  that count times 8–17 µs: MALA 4.85 ms / 288 = 16.8 µs per array, HMC
  15.6 ms / 1200 = 13 µs, static HMC 22.3 ms / 2914 = 7.7 µs.
- **Host round trips.** Every scalar step reads 5 values on the host (the
  start's check, divergence, the accept decision, log-alpha, and run's
  :keep); a many-chain step reads 9. Each is an evaluation, a GPU sync.
- **N does not matter.** The many-chain cells cost the same per step at 8
  and 4096 chains (7–9 ms), and IS the same at 1 and 30,000 particles:
  the GPU's arithmetic is not what the time is spent on.

### The profile (CUPTI)

There is no Nsight on fenrir, so a small CUPTI injection library
(`kprof`, built against the CUPTI in the owner's genjax-parity venv and
loaded into the JVM by `CUDA_INJECTION64_PATH`) totals every kernel, copy,
synchronization and CUDA API call each second; each cell's window of the
vector model's run (`kprof-0752`, about 8 % slower than without the
library) gives:

| cell | kernels/step | GPU busy | host in launch calls | host in syncs | GPU µs per kernel |
|---|---|---|---|---|---|
| IS N=30000 | 1134 | 5 % | 15 % | 1 % | 1.9 |
| MALA S=1000 | 552 | 41 % | 49 % | 21 % | 4.0 |
| MALA S=1000 compiled | 397 | 38 % | 47 % | 14 % | 2.8 |
| HMC L=8 S=1000 | 1471 | 52 % | 51 % | 29 % | 5.8 |
| HMC L=32 S=1000 | 4632 | 58 % | 53 % | 13 % | 6.3 |
| HMC L=8 S=1000 compiled | 690 | 35 % | 37 % | 9 % | 2.0 |
| HMC L=32 S=1000 compiled | 1754 | 34 % | 27 % | 4 % | 1.4 |
| MALA, 8 chains, S=1000 | 577 | 30 % | 40 % | 16 % | 4.8 |
| MALA, 4096 chains, S=1000 | 582 | 34 % | 37 % | 17 % | 5.1 |

(Each window holds the cell's warm-ups, timed runs and its counting run.
"Host in launch calls" is the time inside `cudaLaunchKernel*`, which
overlaps the GPU's; "in syncs" is `cudaEventSynchronize`.)

- **MLX launches 550–1,650 kernels per step (4,600 at L=32),** one or two
  per MLX array, each running 1.5–6 µs on a nearly empty grid: the
  arithmetic of a step is a few hundred flops. Over the run: 61.7 million
  `cudaLaunchKernelExC` calls costing 198 s of host time (3.2 µs each),
  and 165,000 `cudaEventSynchronize` calls waiting 111 s.
- **`:compile?` halves the kernel count, no more** (HMC L=8: 1,471 → 690
  per step): MLX fuses elementwise chains inside a step, but the step is
  still hundreds of kernels and the trace is still updated on the host.
- **CUDA graphs (genclj-h35j) would remove the host's launch cost, not
  the kernel count:** GenMLX measured about 0.97 µs per launch replayed
  from captured graphs on this card, so 550 kernels keep a MALA step near
  0.5 ms, against XLA's 13 µs for the whole step.

### The verdict on the hypothesis

Confirmed: launch and host overhead, not computing. The fix is not a
faster op (MLX's per-op cost is already its own business, genclj-h35j and
genclj-eof0 shave it) but fewer, larger launches and no host in the loop:
a whole run as one program.

## 2. The graph it takes (as built in ext/device)

### A staging backend: gen.clj's own code, recorded

The device program is not a second implementation of a kernel or a model.
It is gen.clj's own code run once against a backend that records
instead of computing: `gen.device.stage`, an L0 backend like the JVM's.
Every `gen.array` op on a staged value adds a node `{:op :args :shape
:dtype :attrs}`; host numbers in an op are constants. Because the backend
contract is the one the JVM and MLX backends implement, the layers above
compose with it unchanged:

- **Gradients** come from the tape (`gen.array.tape`), which wraps the
  staging backend as it wraps the JVM's: `value-and-grad` records the
  backward pass's nodes in the same graph, so the gradient is generated
  into the program, not taken at run time.
- **Particles** come from the batch tracer (`gen.array.batch`), which
  wraps it as it wraps MLX: `gen.batched`'s run of the handler and
  `array/vmap` give the [N …] program, the particle axis a shape like any
  other.
- **The model** is the handler: `gfi/generate`, `gfi/assess` (the static
  language's generated score fn), the dynamic DSL's sites, the
  distributions' log densities. A static model and its dynamic body give
  the same program up to node count (2379 and 1032 nodes for the sites
  and vector forms of the parity model).
- **The kernel** is `gen.inference.mcmc`'s move, `hmc-move` and
  `mala-move`, which the module already writes "as a function of arrays
  and keys only" so that it runs batched. The prototype reads them (they
  are private); the design would make the move a kernel's public form.

### Keys as values of the device

Rule 2 keeps keys values, and a staged key is one: a uint32 [2] array of
the staging backend (`p/IStagedKey`). `gen.random`'s split and fold-in
dispatch on it and emit Threefry-2x32-20 as uint32 ops (adds, xors,
rotations), the scheme of `gen.random.threefry`, so a staged program draws
the host's bits for the same key: bit-identical uniforms, normals to
erfinv's float32 rounding. A draw on the batch tracer from one key gains
the particle axis, as `gen.batched`'s does, so batched programs draw what
`gen.batched` draws.

### Loops: `scan`

The iteration loop is `(scan f carry xs n)`, JAX's `lax.scan`: `f` is
staged once on parameters for one step and the loop is a node holding its
body's graph. Values of enclosing graphs that the body reads are captured
as extra loop operands. A chain is a scan over kernel/run's step keys,
`(rng/split key S)` as one [S 2] array, carrying the selected values;
each step's outputs (values, accept flags) are stacked. HMC's L leapfrog
steps stay a Clojure loop inside the body (unrolled: compile time grows
with L, not run time).

### Lowering: StableHLO

`gen.device.stablehlo` writes the graph as MLIR in generic op syntax:
elementwise ops with NumPy broadcasting made explicit
(`broadcast_in_dim`), reductions and scatters with their regions, a scan
as a `stablehlo.while` (counter, carries, output buffers updated by
`dynamic_update_slice`, inputs read by `dynamic_slice`). erf, erfinv,
sinh, cosh and the inverse trigonometric functions are CHLO ops, which
XLA decomposes. Two lowerings serve fusion: a slice's gradient is a
`pad`, and a whole reduction of up to 16 elements an elementwise chain.

### Running: PJRT by FFM

`gen.device.pjrt` binds XLA's PJRT C API (API 0.76) with Java's FFM, from
Clojure: every PJRT function takes one args struct, so one downcall
descriptor serves all, and the structs' offsets are generated by the C
compiler (`native/pjrt_offsets.c`). The CUDA plugin is
jax-cuda13-plugin's `xla_cuda_plugin.so` (here from the owner's
genjax-parity venv on fenrir), with the client's BFC allocator growing on
demand (XLA preallocates 75 % of the card by default, wrong on a shared
host): the process held 578–586 MiB. A minimal CompileOptionsProto (one
replica, one partition) is enough, and XLA needs no ptxas or libdevice
setup (float64 exp, log, tanh and log1p compile and agree with the JVM
to 1 ulp). The binding's own cost is about 6 µs of a 37 µs execute.

### What stages, and what does not

Stages (shown): fixed-structure models, static or dynamic bodies whose
structure does not depend on drawn values, their constrained generate,
assess and gradients; HMC and MALA moves, which `mcmc` already writes as
functions of arrays and keys; batched runs (IS, many chains).

Should stage, not tried: MH (the handler's regenerate takes staged keys,
but `mh` accepts on the host; it needs a `where` over the trace's choices,
as `accept-particles` does per particle), SMC's extensions under the
tracer, MAP and VI steps. Resampling needs a gather (`-take`), which the
staging backend does not have yet.

Does not stage (in the prototype, each a loud error at staging):

- a host read of a staged value (`truthy?`, `item`, `->clj`): a branch
  on a drawn value outside `where`, or a loop until a condition in a
  model's own code (a `while` node beside `scan` would take it; gen.clj's
  samplers are branch-free already: gamma runs twelve vectorized
  Marsaglia–Tsang rounds, so it stages);
- structure that depends on draws: Map or Unfold of a drawn length,
  Recurse, a Switch on a drawn index (the batch tracer's partition has no
  staged form yet);
- LLM generative functions, host-library distributions (Commons Math),
  custom generative functions that call host code;
- float64 draws (staged Threefry makes float32 uniforms; float64 would
  take the 52-bit path, and the card runs float64 at 1/64 speed anyway).

### How it keeps the rules

- **Rule 1.** The program is the handler's code staged, not a copy. The
  prototype's checks run the program and the handler from one key and
  compare draws: HMC (static and vector models), batched IS (1000
  particles: log-ML −17.54296 both) and many-chain MALA agree with the
  handler on MLX to float32 rounding. A design would make that a law
  (`laws/device`: for random models and kernels, the staged run equals
  kernel/run to a band set by the float32 condition of the chain).
- **Rule 2.** Keys stay values; a staged key is a value of the program.
- **Rule 3.** Core code calls only gen.array. The staging backend is L0;
  PJRT, StableHLO and the plugin are an extension.
- **Rule 6.** A host read during staging is a loud error naming why.
- **Rule 8.** The staging backend and `scan` are pure Clojure. The CUDA C
  target adds no dependency (MLX's CUDA build already links NVRTC and the
  driver); XLA's plugin would be an optional native dependency, as MLX
  is.

## 3. The prototype's results

### Equivalence (rule 1)

Checked on the Mac by running each program on XLA's CPU (JAX's CPU
client, a dev check only) against the handler on MLX's CPU (float32)
from the same key:

- the parity HMC unit (generate + 10 steps, L=8): the chain equals the
  handler's to ~1e-6, static and vector forms alike;
- batched IS, 1000 particles: log-ML −17.54296 on both, weights equal to
  float32 rounding;
- many-chain MALA, 8 chains × 10 steps: every chain's slope equals the
  handler's at every step;
- the density's value and gradient, draws from staged keys (uniforms
  bit-identical, normals to erfinv's float32 rounding), and a scan.

On fenrir's GPU the static and vector programs give identical chains
(acceptances and tails match to every digit printed), and at S=1000
every cell's acceptance and slope tail agree with the handler's and
GenJAX's. The CUDA C target's checks against MLX's handler on the GPU
are in §4.

### Speed (fenrir, XLA through PJRT)

The last column of the table in §1. Against MLX today: 370× on MALA,
440× on 8-step HMC (92× against `:compile?`), 300× on 32-step HMC, 190×
on 4096 chains, 36× on IS. Against GenJAX: MALA level (13.1 against 15.6
ms at S=1000), 8-step HMC level (35 against 33 ms), 32-step HMC 1.6×,
many chains 1.3× at 8 chains and 3.1× at 4096 (41 against 13 ms), IS 16×
(the binding's per-call floor, below).

Two lowering details mattered (ae99b50): a slice's gradient (the tape's
`unslice`, a scatter-add) recorded as a `pad`, and whole reductions of up
to 16 elements written as elementwise chains, so that XLA fuses them:
8-step HMC 64 → 35 ms, 32-step 245 → 165 ms, 4096 chains 46 → 41 ms.

### Warm-up

Staging and StableHLO take 10–90 ms per program, independent of S and N
(the loop is staged once). XLA's compile is most of the warm-up: 0.25 s
(IS), 0.7 s (MALA), 1.4 s (HMC L=8), 5 s (HMC L=32, whose leapfrog loop
is a Clojure loop inside the step, unrolled), against JAX's jit 0.35–1.9
s. A compiled program is reused for every key; a cache by model,
arguments' shapes and settings would make it once per shape.

### What is left on the table

- **The per-call floor.** A call costs 0.25–0.3 ms however small the
  program (IS at N=1). The binding calls PJRT through
  `invokeWithArguments`, allocates and frees per call, uploads the key and
  reads every output back. The PJRT agent measured 37 µs for an execute
  with its arguments on the device, 97 µs with one small copy each way.
  invokeExact handles and reading only what the caller wants would bring
  this to the GenJAX floor's order (19 µs).
- **HMC at L=32 and many chains** (1.6× and 3×): the step body is larger
  than GenJAX's (Threefry for each draw's key in the graph, the move's
  finiteness checks), about 26 kernels per step on average (§4); fewer
  would need XLA-side tuning (its fusion, its command buffers), which the
  CUDA C target makes moot for these cells.

## 4. Two targets

XLA (measured above) compiles a step into a few dozen fused kernels and
runs each iteration of the loop as one CUDA graph launched from the
host. CUPTI on the XLA run (`xla-kprof`): 175,178 `cuGraphLaunch` calls
for 174,790 steps at about 28 µs of host time each, about 26 kernels per
step of about 1.3 µs each, and no copy to the host per step (the loop's
trip count is known). That is a floor of about 13–45 µs per step whatever
the work, which is why 4096 chains cost 41 ms and GenJAX's 13 ms. For
gen.clj's usual workloads, small models with tens to thousands of
particles or chains, the floor is the cost.

Generated CUDA C (step 4) takes the same per-chain program and makes one
kernel of the whole run: one thread per chain, the scan a for loop in
the thread, state in registers, Threefry per thread, nothing launched per
step and nothing back to the host until the end.

### CUDA C, measured (step 4)

`gen.device.cuda` (built by a forked agent, checked here) generates the
kernel from the same staged per-chain program (`chain/hmc-chain`,
`mala-chain`: the run's step keys derived in the loop, fold-in of the
counter, the same draws): every node scalarized at generation time (an
array is a vector of C expressions, so reshapes, broadcasts, slices,
pads and concatenations cost nothing), a scan a `for` loop, float32 with
no fused multiply-adds (as MLX's op-by-op kernels round). NVRTC compiles
it for sm_120 in 20–310 ms (830 ms for the static model at L=32); the
kernels use 40–48 registers, no stack, no spills. Thread i runs the
chain from `(rng/split key n)[i]`: n chains are n runs of the scalar
kernel, as kernel/run's (gen.batched's many-chain draws, one key with a
particle axis, would be a different but equally valid assignment).

Medians, ms, vector model (static within a few %), a run being a launch,
the wait, and every output copied to pinned host memory; the kernel's
own time in brackets where the copy matters:

| cell | CUDA C | XLA | GenJAX |
|---|---|---|---|
| MALA S=100, 1 chain | 0.16 | 2.0 | 1.62 |
| MALA S=1000, 1 chain | 0.86 | 13.1 | 15.6 |
| MALA S=1000, 4096 chains | 2.4 (0.85) | 40.7 | 13.2 |
| MALA S=1000, 65,536 chains | 27 (2.1) | — | — |
| HMC L=8 S=100, 1 chain | 0.23 | 5.4 | 3.67 |
| HMC L=8 S=1000, 1 chain | 2.05 | 35.2 | 32.6 |
| HMC L=8 S=1000, 4096 chains | 3.6 (2.05) | — | — |
| HMC L=8 S=1000, 65,536 chains | 28 (3.5) | — | — |
| HMC L=32 S=1000, 1 chain | 7.2 | 164.5 | 100.8 |

- A thread costs about 0.85 µs per MALA step and 2 µs per 8-step HMC
  step, and the kernel's time does not change from 1 to 4096 chains (the
  card holds them all at once); at 65,536 it grows 2–3×, and copying
  every sample back (0.5 GB) is most of the run.
- 14–18× GenJAX on single chains, 5.5× on 4096 chains (15× on the
  kernel alone), 23× XLA's on 32-step HMC.
- Equivalence: against MLX CUDA's handler (4 chains × 100 steps, HMC L=8
  and MALA, from `(rng/split root 4)[i]`) the static model is
  bit-identical and the vector model within 3.5e-5 relative (its 10-term
  sum's order), no accept decision different; against the XLA program
  on thread 0's key within 3e-5, the same accept flags. `cuda_test` (129
  assertions, on MLX) checks determinism, that thread i does not depend
  on the launch size, and that it equals kernel/run.
- Refused, loudly: scatter-add (the pads replace it), gathers, a scan
  over inputs (counted scans only), nested scans, a scan's outputs read
  in the thread, float64, batched matmul, and nodes past 4096 elements.
  So: no resampling yet (it needs a gather), no SMC, and models with
  large arrays per particle belong to XLA.

The two targets are complementary. CUDA C wins where gen.clj usually is:
small per-particle state, many chains or particles, long loops; it has
no floor per step and no host in the loop. XLA wins where the work per
step is large arrays (big data, matrix products, GPs: cuBLAS and
fusion), and takes any program the staging backend makes, including
gathers and scatters.

## 5. Recommendation

1. **Take the graph into the core, at L0.** The staging backend
   (`gen.array.stage`, pure Clojure, about 400 lines), `array/scan` (a
   host loop on eager backends, a loop node on the staging backend) and
   staged keys (`p/IStagedKey`; rule 2's wording gains "or a staging
   backend's array"). These are the shared asset: every lowering reads
   them, and nothing above L0 changes. Tests: the device programs against
   the handler from the same keys, as `laws/device` (draws to a band set
   by float32, accept decisions equal or flipped only within rounding).
2. **Make kernels' moves public as functions of arrays and keys,** as
   `mcmc`'s hmc-move and mala-move already are internally, and give MH a
   `where` accept over the trace's choices. Add one entry point that runs
   a whole kernel/run, importance sampling or particle filter as a
   program: `(device/run kernel trace {:samples S :keep …})`, the same
   result as kernel/run, compiled once per model, argument shapes,
   settings and S. A model that does not stage (§2) is a loud error with
   the reason, or with `:fallback true` the handler's path, recorded as
   gen.batched records its fallbacks.
3. **Generated CUDA C as the NVIDIA target for small particles** (an
   extension, `ext/cuda`): libnvrtc and the driver, both of which MLX's
   CUDA build already links (`libnvrtc.so.13`, `libcuda.so.1`), so no
   new dependency. Extend it with gathers
   (resampling, so SMC), a reduction across threads (IS's log-ML,
   resampling's weights) and the batched semantics, then SMC and
   many-chain kernels.
4. **XLA via PJRT as the target for large arrays** (an extension,
   `ext/xla`): the PJRT plugin (jax-cuda13-plugin's .so, 321 MB) is the
   cost, beside the CUDA pip libraries it loads (cuBLAS, cuDNN, NCCL:
   those MLX's CUDA build loads already; NVSHMEM, 193 MB, besides); it also brings
   TPUs and AMD GPUs if they ever matter (genclj-7npm). Lower priority
   than CUDA C for gen.clj's workloads. Its per-call floor (0.25 ms in the
   prototype) needs invokeExact handles before it is fair to it.
5. **MLX stays** for Metal, eager and interactive work and the LLM side;
   on NVIDIA it is not the fast path. genclj-h35j (CUDA graphs) would cut
   the host's launch cost but not the kernel count (§1).
6. **The JVM's CPU is a third lowering of the same graph,** worth its own
   research: today's JVM backend runs a MALA step in 0.1 ms on fenrir's
   CPU (7× GenJAX on the GPU, 45× MLX CUDA); a staged step compiled to a
   JVM method (Raster, genclj-q4wn, or our own bytecode) would remove the
   handler's per-op cost there too.

## 6. For the owner to decide

1. Adopt the staging backend, `array/scan` and staged keys into the
   core (L0), with rule 2's wording amended? (Recommended.)
2. The device-run API: an option of kernel/run, importance-sampling and
   the particle filter (`:device :cuda`), or a namespace of its own
   (`gen.device/run`)? And its fallback: loud by default, or the
   handler's path with the reason recorded?
3. Many-chain draws on a device: per-chain keys (`(rng/split key n)[i]`,
   kernel/run's, what CUDA C does naturally) or gen.batched's one key
   with a particle axis (what the XLA programs do)? Both are valid; the
   choice decides which handler run a device run equals.
4. CUDA C (ext/cuda) first, XLA (ext/xla) second? (Recommended; XLA's
   plugin size is the main cost.)
5. Rounding: device programs are float32 and fuse differently from the
   handler; accept decisions can flip where log-alpha is within rounding
   of log u, after which chains part. The tests compare draws to a band
   and flips to a count; is that the equivalence rule 1 asks for?
