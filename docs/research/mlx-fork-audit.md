# GenMLX's MLX fork vs upstream (2026-10-19, bean genclj-zfgj)

**The fork.** GenMLX vendors MLX as the submodule
`mlx-node/crates/mlx-sys/mlx`, pinned at `2324cb204`
(`pin/mlx/2026-07-27-ny70-9`, from `robert-johansson/mlx`).
- **Base.** Upstream v0.32.0 plus 21 commits (`973e27f82`); gen.clj's
  mlx-c pins v0.32.2.
- **Fork-only commits.** 41.
- **Upstreamed.** None, by `git cherry` against upstream main
  (v0.32.4+9, `0390bd835`) on 2026-10-19.
- **Method.** A scratch clone of the submodule, with upstream fetched into
  it. GenMLX's own checkout was not touched.

## The patches, by what gen.clj needs

| Group | Commits | Needed by gen.clj? |
|---|---|---|
| **Autograd:** `Cholesky::vjp`, `Inverse::vjp` | 2 | **No.** Upstream still has neither. gen.clj differentiates on its own tape, which has VJPs for every linalg op. For native MLX autodiff (genclj-fuvr), the backend can give MLX these VJPs with `mlx_custom_vjp`, which the binding supports (`MlxRuntime/customVjp`), without patching MLX. |
| **CPU:** `Inverse` of a singular matrix gives NaN instead of aborting | 1 | **No.** gen.clj never calls MLX's `inv`. The backend keeps singular matrices away from MLX's solvers (below). |
| **Metal:** lazily register per-thread stream encoders | 1 | **No.** gen.clj's streams are made with `mlx_stream_new_thread_unsafe`, which any thread can use. |
| **CUDA:** sm_110 allocator and mempool, NVRTC cache and CCCL include, graph-cache growth and fallbacks, no throw from `~CaptureContext`, a `cp_async` drain in `qmm_sm80`, per-thread encoders, the LRU cap, stream-ordered deferred frees and the OOM ladder, a unified-memory sync | 13 | **Possibly**, for CUDA on Thor (sm_110) and the RTX PRO 6000 (sm_120). Decided in genclj-0kge, by trying upstream first. |
| **Quantization:** K-quants (Q4_K, Q5_K, Q6_K) on CPU, Metal and NAX, and an e8m0 rounding fix | 12 | Not for M2. For M3's MLX-native forwards (genclj-gvfa), if they run GGUF K-quants. llama.cpp, the first engine, has its own. |
| **NAX SDPA:** a head_dim 256 path, plus probes | 4 | Not for M2. M3, as above. |
| **Performance:** compile fusion, replay staging, a Threefry fast path, CUDA replay capture | 7 | **No.** The Threefry path is for MLX's own random layout, and gen.clj computes JAX's itself. Compile performance may matter for genclj-8b88, so measure first. |
| **Build:** `MLX_BUILD_JACCL` | 1 | **No.** The wheel is used as built. |

## Decision

**CPU and Metal build against upstream MLX, unpatched:** the official
0.32.2 wheel (`native/build-mlx.sh`). That follows the owner's preference
of 2026-10-19: patch MLX as little as possible, and when a patch is needed,
keep it clean so that following upstream stays easy.

**If CUDA needs fixes** (genclj-0kge), they come as a patch series applied
to a pinned upstream tag by `native/build-mlx.sh`, never as a long-lived
fork:
- one concern per patch, in `native/patches/NNNN-<topic>.patch`;
- a header naming the GenMLX bean it comes from and the upstream status;
- rebased at each MLX bump, and dropped once upstream has it.

**Upstreaming is the owner's call.** MLX's contributor rules forbid
agent-submitted PRs.

## Related findings in stock MLX 0.32.2

Found while auditing, handled in the backend without patches
(`docs/array-api.md` §13):
- **Cholesky.** A matrix that is not positive definite gets a partial
  factor, not an error or NaN. LAPACK leaves the failing pivot
  non-positive, so the backend writes NaN where a diagonal entry is not
  positive.
- **Solves.** A singular system fails inside MLX's CPU task. The failure is
  reported at a *later* evaluation, against other arrays. So the backend
  never gives MLX a degenerate matrix:
  - `solve`: the identity stands in for a matrix with a zero LU pivot, and
    its solution is NaN;
  - `solve-triangular`: a zero diagonal entry becomes NaN.
- **slogdet.** MLX's throws on a singular matrix. The backend computes it
  from the LU factor: `[0 -Inf]` there, as on the JVM backend.
- **MLX's CPU float64 transcendentals** are float32 approximations (exact
  routes in `gen.backend.mlx.exact`).
- **Full's VJP** (2026-10-22, genclj-hf5z) multiplies the cotangent by
  the fill value, so native gradients of `full` of a tracked value were
  scaled by it. The backend fills with an array value by `broadcast_to`.
  Draft issue: `docs/research/mlx-upstream/issue-1-full-vjp.md`.

The Metal toolchain is installed now (2026-10-19), so source builds of MLX
work on the M4 when a patch series needs one.
