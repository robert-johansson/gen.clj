# JVM numerics backends for gen.clj

*Research notes, 2026-10-18. Produced by a research agent with web access and
spot-checked: mlx-c tags v0.7.0 → MLX v0.32.2 and `mlx_detail_vmap_*` were
verified directly. The "Could not verify" items at the end are unverified, as
flagged. Summarized in DESIGN.md §5.*

> **Owner decision after this research (2026-10-18):** M1 is CPU-only on a
> **pure-JVM backend** with no MLX dependency. MLX via FFM becomes a second
> backend in M2 for the GPU. Both sit behind one array API (DESIGN.md §5). The
> recommendation below ("MLX-CPU first") is kept for the record.

## TL;DR
- **MLX via `mlx-c` + Panama FFM is the only option that meets every requirement
  from one API.** It covers CPU (macOS arm64, Linux x86_64/arm64), Metal and CUDA,
  plus grad/vjp/jvp/custom_vjp, vmap, compile, splittable PRNG keys, linalg and
  quantized matmul. The C API is mature (mlx-c v0.7.0 tracks MLX 0.32.2; Ollama
  and mlx-swift ship on it).
- **The Feb-2026 Coffi experiment was slow because of how it called FFM, not
  because of FFM.** It used `invokeWithArguments` with an Object array, a
  per-call `Arena/ofAuto`, and Clojure maps as handles. Generated Java static
  wrappers with `invokeExact`, struct handles lowered to `long`, and reused
  scratch memory should bring binding cost to about 10–30 ns per call. That is
  dwarfed by MLX's own costs: GenMLX measured a ~67 µs eval floor and ~5 µs per
  op in the scheduler walk. No C shim is needed for performance.
- **TornadoVM is current and fast-moving (7.3.1; Metal and CUDA backends), but it
  is the wrong layer.** It has no autodiff, no vmap and no array-graph model, and
  kernels must be Java static methods.
- **Recommendation:** MLX-via-FFM for both phases, behind a backend protocol,
  plus a small pure-JVM reference backend later for tests and tiny scalar models.

## 1. Comparison

| Option | CPU | Metal | CUDA | Autodiff | vmap | compile | Quantized | Latest release | Clojure fit | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **MLX via mlx-c + FFM (build it)** | ✓ mac (Accelerate), Linux x86_64/arm64 | ✓ | ✓ | ✓ grad / vag / vjp / jvp / custom_vjp / custom_function | ✓ via `mlx_detail_vmap_*` (as mlx-swift) | ✓ `mlx_compile` | ✓ qmm / gather_qmm / qqmm | MLX 0.32.4 (2026-10-14); mlx-c 0.7.0 (2026-09-29) | Good, with a Java glue layer | **Primary** |
| jmlx (Alipsa), Java 25 FFM over mlx-c | — | ✓ only | ✗ | ✓ value_and_grad | ? | ? | ✓ | created Aug 2026, 0★, MIT | Java | Prior art |
| TornadoVM 7.3.1 | OpenCL device / JVM fallback | ✓ native MSL backend (since 4.0) | ✓ CUDA C via NVRTC | ✗ | ✗ | kernel JIT only, no graph fusion | library tasks or hand kernels | 2026-10-15 | Poor: kernels must be Java | No as backend; niche kernels at most |
| Uncomplicate: Neanderthal 0.66.1 / Deep Diamond 0.45.0 / ClojureCUDA 0.32 | ✓ MKL / OpenBLAS / Accelerate | ✗ | ✓ | ✗ (DD: layer backprop only) | ✗ | ✗ | ✗ | Aug 2026, EPL | Excellent | BLAS/LAPACK for a JVM backend |
| DJL 0.38.0 / PyTorch 2.7.1 | ✓ | ✓ ("mps") | ✓ | ✓ GradientCollector; no custom VJP from Java | ✗ | ✗ | limited | 2026-09-09, maintenance mode | Java objects, JNI per op | No |
| libtorch via JavaCPP 1.5.14 (snapshot at 2.14) | ✓ | ✓ MPS | ✓ | ✓ autograd; custom Function awkward | ✗ (Python only) | ✗ (Dynamo is Python) | partial | Aug 2026 | Heavy | No |
| JAX via libpython-clj | ✓ | ✗ in practice (jax-metal reportedly abandoned) | ✓ | ✓ | ✓ | ✓ jit | ✓ | — | Embedded CPython, GIL | Parity oracle only |
| TensorFlow Java 1.2.0 (TF 2.21) | ✓ | ✗ | ✓ | partial graph gradients | ✗ | graph | ✗ | 2026-07-24 | Poor | No |
| ONNX Runtime Java 1.31 | ✓ | CoreML EP | ✓ | inference only | ✗ | — | ✓ | 2026-10-09 | OK | No |
| Apache MXNet | — | — | — | — | — | — | — | archived (1.9.1, 2022) | — | Dead |
| ND4J / DL4J / SameDiff | ✓ | ✗ | ✓ | ✓ SameDiff | ✗ | ✗ | ✗ | last 1.0.0-M2.1 (Aug 2022) | Meh | Stalled |
| Babylon HAT | Java MT | ✗ (OpenCL only on mac) | ✓ (+HIP, SPIR-V) | ✗ | ✗ | kernel codegen | ✗ | Code Reflection JEP draft "Submitted", no target | ✗ needs javac code models | Watch |
| Java Vector API | SIMD | — | — | — | — | — | — | incubator: JEP 508 (25), 529 (26), 537 (27) | Fine | Building block |
| Raster (replikativ) | ✓ SIMD | ✗ | OpenCL only | ✓ fwd + reverse in its compiler | ✗ | ✓ AOT typed kernels | ✗ | created Mar 2026, active, MIT | Clojure-native | Watch / ideas for a JVM backend |
| Emmy | ✓ scalar/symbolic | ✗ | ✗ | fwd + tape reverse, scalar | ✗ | ✗ | ✗ | v0.32.0 (2024), GPL-3.0 | Good | No (license, no tensors) |
| dtype-next / tech.ml.dataset | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | active (Oct 2026) | Excellent | Buffers / IO only |
| Kotlin: KotlinDL / Kotlin∇ / SKaiNET | ✓ | ✗ | ✗ | ✓ (SKaiNET, Kotlin∇) | ✗ | ✗ | ✗ | KotlinDL 2023, Kotlin∇ 2021 | Poor | No |
| Own Clojure reverse-mode AD over arrays | ✓ | ✗ | ✗ | ✓ (we build it) | we build it | we build it | ✗ | — | Best | Reference / small-model backend |

## 2. Deep dive: MLX via mlx-c via FFM

### Versions and platform status
- **mlx-c**: tags only, no GitHub Releases. v0.7.0 was tagged 2026-09-29.
  `CMakeLists.txt` FetchContent pins MLX **v0.32.2**, and the
  `MLX_C_USE_SYSTEM_MLX` option uses an installed MLX instead. MIT licensed.
  Recent commits added the missing stream functions and
  `new_thread_unsafe_stream`. https://github.com/ml-explore/mlx-c
- **MLX**: latest 0.32.4 (2026-10-14). CUDA work: Cholesky via cuSOLVER (0.32.3),
  CUDA kernel JIT/caching (0.32.0), faster quantized gather-matmul (0.32.4).
  https://github.com/ml-explore/mlx/releases
- **Official 0.32.4 wheels on PyPI** each contain `libmlx.so`/`.dylib`, the
  headers and `share/cmake/MLX` (verified by unpacking `mlx-cpu`):
  - `mlx-cpu`: manylinux_2_35 x86_64 + aarch64
  - `mlx-cuda-12` / `mlx-cuda-13`: x86_64 + aarch64
  - `mlx-metal`: macOS arm64

  These are libraries we can link mlx-c against. jmlx already builds mlx-c
  against the mlx-metal wheel.
  https://pypi.org/project/mlx-cuda-13/ · https://pypi.org/project/mlx-cpu/
- **CUDA requirements:** SM ≥ 7.5, driver ≥ 550.54.14 (≥ 580 for CUDA 13),
  glibc ≥ 2.35. Source builds use `-DMLX_BUILD_CUDA=ON`.
  https://ml-explore.github.io/mlx/build/html/install.html
- **The Linux x86_64 CPU backend works, with two caveats:**
  - The CPU wheel bundles a 683 KB `libblas.so.3` with no OpenBLAS symbols
    (probably reference netlib). Build against OpenBLAS for real GEMM speed.
  - CPU `compile` shells out to `g++` at runtime; without it there is no CPU
    fusion. https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/jit_compiler.cpp
- **Linalg decompositions run CPU-only.** QR, SVD, inv, Cholesky, eig/eigh and LU
  report "NYI" on Metal. On CUDA everything except Cholesky has "no CUDA
  implementation". GenMLX's CPU-stream routing carries over unchanged.
  (`mlx/backend/metal/primitives.cpp`, `mlx/backend/cuda/primitives.cpp`)

### API coverage (v0.7.0 headers)
- **`transforms.h`**: `mlx_vjp`, `mlx_jvp`, `mlx_value_and_grad`,
  `mlx_custom_vjp`, `mlx_custom_function` (optional custom vjp/jvp/vmap
  closures), `mlx_checkpoint`, `mlx_eval`, `mlx_async_eval`.
- **vmap**: there is no public `mlx_vmap`. `transforms_impl.h` exposes
  `mlx_detail_vmap_trace` and `mlx_detail_vmap_replace`; mlx-swift and the old
  Gen.clj build vmap on these.
- **`compile.h`**: `mlx_compile(res, fun, shapeless)`, `mlx_detail_compile`
  (stable fun id plus constants), and enable/disable/set_compile_mode.
- **`random.h`**: key, split, split_num, bits, uniform, normal, truncated_normal,
  categorical, gumbel, laplace, multivariate_normal, permutation.
- **`linalg.h`**: cholesky, cholesky_inv, solve, solve_triangular, inv, tri_inv,
  qr, svd, eig, eigh, lu, lu_factor, det, slogdet, pinv, norm.
- **`fast.h`**: sdpa, rope, rope_dynamic, rms_norm, layer_norm, cross_entropy,
  and **user Metal and CUDA kernels** (`mlx_fast_metal_kernel_*`,
  `mlx_fast_cuda_kernel_*`).
- **`ops.h`**: quantize, dequantize, quantized_matmul, gather_qmm, qqmm,
  searchsorted.
- **`memory.h`**: clear_cache, set/get memory limit, cache limit, wired limit,
  active/peak/cache memory. `metal.h` has capture and the metallib path;
  `cuda.h` has only `mlx_cuda_is_available`.
- Device selection uses the same `mlx_device` gpu type on Metal and CUDA, so the
  **Clojure code is identical across the two**.
- **Gap**: GenMLX's custom C++ ops in mlx-node's `mlx-sys`
  (`mlx_genmlx_gamma.cpp`, `_bessel`, `_searchsorted`, `_ext`, fused ops).
  mlx-c has no lgamma or digamma. Re-express them as composite ops, as
  `fast_*_kernel` source kernels wrapped in `custom_function` for gradients, or
  as a tiny `extern "C"` C++ library linked against the same libmlx (for
  semantics, not speed).

### What we'd build
- **Native**: `libmlxc` built against a pinned libmlx (the matching wheel, or
  GenMLX's patched MLX fork for CUDA). Keep mlx-c and MLX at exactly matching
  commits, because mlx-c depends on the C++ ABI. Note the skew: mlx-c 0.7.0
  pins 0.32.2 while the newest wheels are 0.32.4.
- **Java glue: a small generated Java module, not Coffi.**
  - Write a tiny generator over the regular, auto-generated mlx-c headers
    (mlx-c has its own Python generator). It emits `static final MethodHandle`
    fields and static methods. Jextract can stay as a reference.
  - **Lower every handle struct to `JAVA_LONG`.** All mlx-c handles are
    `struct { void* ctx; }` passed by value. On AArch64 (macOS and Linux) and
    SysV x86-64, a one-pointer struct is classified exactly like a pointer
    (passed in a general register, returned in x0/rax). So
    `int mlx_add(mlx_array*, mlx_array, mlx_array, mlx_stream)` becomes
    `(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG) → JAVA_INT`. This avoids
    jextract's per-handle MemorySegments and SegmentAllocator struct returns.
    It relies on the ABI, not the FFM contract, so cover it with tests.
  - **Per-thread scratch for out-params.** Each op writes its result through an
    8-byte out-pointer. Zero it before every call: mlx-c dereferences a
    non-null `ctx`, which the old shim comments note crashes.
  - **Error handler.** mlx-c's default handler is `printf` followed by
    `exit(-1)` (`error.cpp`). Install an upcall that stores the message in a
    ThreadLocal, then throw `MlxException` on a non-zero status.
  - **No `Linker.Option.critical` for ops.** Critical functions must not upcall,
    and the error handler is an upcall. Use `critical(true)` only for heap-array
    import/export around `mlx_array_new_data` / `mlx_array_data_*`, after
    validating arguments in Java.
  - **Closures.** Use one global upcall stub for
    `mlx_closure_new_func_payload` (plus value_and_grad, custom and vmap
    variants) and one dtor stub. The payload is a `long` id into a registry of
    IFns, and MLX calls the dtor when it releases the closure. This is the same
    pattern as Ollama's `cgo.Handle`. It avoids per-closure stub creation
    (expensive) and handles lifetime correctly, since compile caches outlive
    Java objects. Upcalls run synchronously on the caller's thread (confirmed
    by jmlx). They must catch every Throwable, because an exception escaping an
    upcall kills the JVM.
- **Clojure layer.** Make direct, type-hinted static calls such as
  `(Mx/add a b s)` with `^long` handles. Clojure cannot emit
  signature-polymorphic `invokeExact` call sites, which is why the Java layer
  exists. Use protocols only for coarse-grained backend dispatch.

### Why the Coffi experiment was slow, and expected overhead
- **Causes.** `src/gen/mlx/*.clj` call `.invokeWithArguments` on
  `object-array`s (boxing, spreading, no inlining), allocate an `Arena/ofAuto`
  per call, and represent handles as `{:ctx seg}` maps that Coffi serializes.
  Each of these is a known slow path, which is why a C shim looked necessary.
- **FFM call cost.** A call-only FFM benchmark measured ~50 ns on a 2019 laptop
  with JDK 19, slightly faster than JNI
  (https://github.com/zakgof/java-native-benchmark). On an M4 with C2, expect
  around 10 ns (not verified).
- **Node-API comparison.** Typically 100–300 ns per call plus the JS wrapper and
  finalizer (rule of thumb, not verified). FFM should match or beat GenMLX's
  NAPI path.
- **Where the time actually goes.** MLX graph-node construction is about 1 µs
  (estimate), and eval has a floor. GenMLX's `docs/genjax-parity.md` measured
  ~67 µs per `eval!` and ~5 µs per op; `docs/gpu-cost-per-particle.md` gives
  5–10 µs per op for kernel launch.
- **The old HMC number.** 1.2 s for 500 steps is ~2.4 ms per step. Profile
  before blaming the binding: where eval and sync happen, and whether the whole
  leapfrog + MH step is compiled with `mlx_compile`, likely matter more.

### Lifetime, threads, streams
- **Arrays** are heap objects owned by mlx-c, held as a plain `long` with no
  Arena.
  - Use a `with-mlx-scope` (tidy-like) block that frees non-escaping handles on
    exit.
  - Add a `Cleaner` backstop that **pushes freed handles onto a queue drained
    by the owning thread**, so frees never run off-thread. Whether off-thread
    frees are safe is still an open question for jmlx.
  - Don't register a Cleaner for every temporary in hot loops; each
    registration costs ~100 ns or more.
- **Streams are thread-affine.**
  - A GPU command encoder is registered per thread, and evaluating on another
    thread fails (`stream.h` docs). `mlx_stream_new_thread_unsafe` and
    thread-local streams exist, and 0.7.0 added `mlx_clear_streams`.
  - Use one long-lived **platform** thread per MLX context. Virtual threads
    cannot be MLX threads (documented by jmlx).
  - Parallel chains go through vmap, as in GenMLX.
- Run the JVM with `--enable-native-access=ALL-UNNAMED`.

### Is CUDA via mlx-c realistic today? Yes.
- Ollama's MLX engine is Go/cgo over mlx-c, built on Linux and Windows with
  `-DOLLAMA_MLX_BACKENDS=cuda_v13` (CUDA 13+, cuDNN 9+), with a source override
  for local mlx-c checkouts.
  https://docs.ollama.com/development · https://github.com/ollama/ollama/tree/main/mlx
- The cleanest route for us is to build mlx-c against **GenMLX's own patched MLX
  fork**, which is validated on Thor sm_110 and RTX PRO 6000 sm_120 and carries
  CUDA fixes that upstream lacks.
- mlx-node upstream claims only "experimental, inference only, aarch64 GB10"
  for CUDA.

### Risks
- The vmap entry points are `detail` APIs.
- mlx-c lags MLX by a few patch versions, and the ABI must match exactly.
- The struct→long lowering relies on platform ABIs (needs tests).
- The GenMLX-specific ops must be ported.
- Cleaner thread-safety and stream lifecycle remain open.
- The CUDA backend is still churning (open sm_120 graph bugs as of 2026-10-14).
- The only existing Java binding, jmlx, is Mac-only, has 0★ and is two months old.

## 3. TornadoVM
- **Status.**
  - v7.3.1 (2026-10-15); roughly weekly releases since September.
  - 4.0 (Apr 2026) added the **native Apple Metal backend**.
  - 5.0 (Jul 2026) added the CUDA C/NVRTC backend.
  - 6.0 (Sep 2026) **replaced JNI with FFM**, **removed the PTX and SPIR-V
    backends**, and dropped the JVMCI dependency.
  - Current backends: **OpenCL, CUDA and Metal**.
  - Licensing: the API is Apache-2.0; the runtime is GPLv2 with Classpath
    Exception. https://github.com/beehive-lab/TornadoVM/releases
- **JDKs.**
  - SDK lines are `jdk21` and `jdk22plus`; the FAQ says the latter "runs on any
    JDK from 22 upwards". GraalVM is listed only for JDK 21.
  - Oracle GraalVM 25 is plausible, since TornadoVM uses its own relocated
    `tornado.graal` copy, but this is **undocumented, so test it**. The
    `tornado-mlx` README says JDK 21.
  - Apps launch with `@$TORNADOVM_HOME/tornado-argfile`; from the Clojure CLI
    that means `:jvm-opts`.
- **Programming model.**
  - `TaskGraph` (data transfers plus tasks) → `snapshot()` → `TornadoExecutionPlan`.
  - Kernels: `@Parallel`/`@Reduce` loops, the `KernelContext` API (thread ids,
    local memory, barriers), or `TileContext` (CUDA cuTile only).
  - Off-heap types: `FloatArray`, `HalfFloatArray`, `BFloat16Array`, and others.
  - Kernel restrictions: no objects, recursion, exceptions or dynamic allocation.
    Kernels are specialized to runtime sizes, so a new shape means a recompile.
  - Library tasks wrap cuBLAS, cuDNN, cuFFT and cuDF.
  - Since 7.3.0, `tornado-mlx` runs **MLX's Metal kernels**. It requires an
    M4-class GPU and has no linalg, gather/scatter or autodiff.
- **Performance.**
  - jitLLM (formerly GPULlama3.java) reports Llama-3.2-1B Q8_0 prefill going
    from 4.9k to 29k tok/s on an A100 ("past llama.cpp"), and Metal prefill
    from 40 to 2372 tok/s on an M4 Pro.
  - Release notes claim "6x LLM decode" on Metal from shared command buffers.
    https://github.com/beehive-lab/GPULlama3.java
- **As a PPL backend.**
  - It has no tensors, broadcasting, autodiff, vmap, lazy graph or PRNG. We would
    have to build an MLX/JAX equivalent on top: shape/stride tensors, ~150
    kernels, VJP and batching rules, a fusion compiler that emits kernels, and
    Threefry. That is person-years.
  - Its per-dispatch overhead suits coarse, fixed graphs, not many small eager ops.
- **From Clojure.**
  - Clojure `fn` bytecode (instance `invoke`, Var derefs, boxed `Numbers`, `RT`)
    cannot be compiled by TornadoVM.
  - Since 7.0, PR #1085 adds `TaskGraph.task(id, java.lang.reflect.Method, args…)`
    for **runtime-generated static kernels** and finds `@Parallel` even when the
    system class loader can't see the class. Clojure macros could therefore emit
    kernels via ASM, insn or the ClassFile API, but that means writing a compiler.
  - Realistically, kernels are written in Java and orchestrated from Clojure.
  - MLX's `fast_metal_kernel` / `fast_cuda_kernel` already cover custom kernels
    *inside* the autodiff graph.

## 4. JDK 25 / GraalVM 25
- **FFM** has been final since JDK 22 (JEP 454). jextract build 25-jextract+2-4
  (2025-11-25) is available for macOS aarch64 and Linux x64/aarch64.
  https://jdk.java.net/jextract/
- **Graal JIT and FFM.** Graal as JIT supports FFM on HotSpot. **No Graal-vs-C2
  downcall benchmark was found**; run JMH under both (`-XX:-UseJVMCICompiler`
  selects C2).
- **Vector API.** Tenth incubator in JDK 25 (JEP 508), eleventh in 26 (JEP 529,
  "no substantial change"), twelfth in 27 (JEP 537). It stays incubating until
  Valhalla previews; JEP 401 is now aimed at JDK 28. GraalVM 25's JIT has only
  "**initial**, experimental" Vector API optimization (arithmetic, reduce,
  compare, blend; disable with `OptimizeVectorAPI`), so C2 may beat Graal on
  Vector API kernels.
  https://www.graalvm.org/release-notes/JDK_25 · https://openjdk.org/jeps/529
- **Raster** needs JDK 24+ (ClassFile API). Its benchmarks used a Valhalla JDK 27
  EA build.

## 5. Recommendation
- **(a) CPU-first phase.**
  - Build the MLX-via-FFM binding now and run it on the CPU device (Accelerate on
    macOS, OpenBLAS-linked MLX on Linux). The complete GenMLX transform set is
    available from day one, so there is no AD, vmap or compile machinery to
    write, and the GPU phase becomes a device switch.
  - Build order:
    1. Binding generator emitting Java statics.
    2. `Mx` statics with long handles.
    3. Error-handler and closure-registry upcalls.
    4. Scopes plus the Cleaner queue.
    5. Port the ~245-function surface of `mlx.cljs`, using GenMLX's existing
       coverage matrix.
  - Benchmark per-op cost with JMH, on C2 and Graal, before any optimization.
- **(b) GPU phase.**
  - Same binding with default device GPU: Metal on macOS; CUDA on Linux, built
    from GenMLX's patched MLX fork with `-DMLX_BUILD_CUDA=ON` (Ollama precedent).
  - Linalg stays on the CPU stream.
  - Use `mlx_compile` for fixed-structure kernels such as leapfrog and MH steps.
- **Backend protocol: yes.**
  - Define it at the level of GenMLX's `mlx.cljs` membrane: primitives plus
    eval/grad/vjp/vmap/compile/random. Dispatch per function; the hot path goes
    straight to the `Mx` statics.
  - A pure-JVM reference backend can come later: eager double[]/float[] arrays
    (optionally on the Vector API), tape-based reverse mode, vmap as a loop,
    compile as identity, and Threefry matching MLX's bits. It serves as a test
    oracle and a no-native fallback, and may be **much faster for tiny scalar
    models**, where MLX's eval floor of tens of µs dominates and pure JVM code
    runs in nanoseconds.
  - Use Neanderthal or LAPACK via FFM for its Cholesky and triangular solves.
  - It must not block the MLX path.

## Could not verify
- Actual FFM downcall cost on M4; Graal vs C2 for FFM and the Vector API.
- NAPI per-call cost (the 100–300 ns figure is a rule of thumb).
- The ~1 µs MLX node-construction figure (estimate).
- Whether the mlx-cpu wheel's BLAS is reference netlib (inferred from size and
  symbols).
- Whether Cleaner-thread frees are safe.
- TornadoVM on Oracle GraalVM 25.
- Ollama's Linux MLX/CUDA production status.
- jax-metal being abandoned (forum report).
- Raster's macOS GPU support via Vulkan/MoltenVK.
- JDK 27 GA status.

Local evidence: `~/code/old.Gen.clj/src/gen/mlx/{ffi,hmc,transforms}.clj`,
`native/gen_mlx_shim.c`, `CLAUDE.md`; `~/code/genmlx/CLAUDE.md`,
`docs/genjax-parity.md`, `docs/gpu-cost-per-particle.md`,
`mlx-node/crates/mlx-sys/src/`.
