# M2 verification record (2026-10-19 to 2026-10-20)

M2's done-means (bean genclj-rtif):
1. the MLX backend passes the shared backend conformance suite;
2. the full M1 test battery passes on MLX's CPU device, Metal and CUDA
   (per-arch tolerances, the float64 JVM backend as reference);
3. benchmarks against GenMLX on the same hardware.

CUDA here is the Jetson AGX Thor (aarch64, sm_110). The RTX PRO 6000
(x86_64, sm_120) comes later (owner, 2026-10-19; bean genclj-kw88).

Every configuration ran its full battery (slow and heavy tests included)
and a property soak: every test.check property run once as usual and once
with ten times its cases and fresh seeds (`test/gen/soak.clj`), since one
passing run shows only the cases it drew. One MLX JVM at a time on the Mac,
each run under an RSS guard.

## Results

Commits: the Mac's Metal battery ran on 57b0a27, and every later Mac step
on 7751b76, whose changes since 57b0a27 touch only the Linux and CUDA
paths and the docs. The Thor ran 7751b76 on the GPU and 0c71c44 with the
CUDA fixes' files on the CPU. Counts are clojure.test's (tests,
assertions) and the soak's (properties, each run as usual and then with
ten times its cases).

| Machine | Device | Run | Result | Time | Peak RSS |
|---|---|---|---|---|---|
| Mac | Metal | full battery | 298 tests, 1734 assertions, 0 failures | 2550 s | 1.4 GB |
| Mac | Metal | soak | 111 properties, 0 failed | 1250 s | 1.6 GB |
| Mac | Metal, native gradients | full battery | 298 tests, 1735 assertions, 0 failures | 2612 s | 1.3 GB |
| Mac | Metal, native gradients | soak | 111 properties, 0 failed | 1268 s | 1.3 GB |
| Mac | MLX CPU, float64 | light battery | 283 tests, 1681 assertions, 0 failures | 1007 s | 8.2 GB |
| Mac | MLX CPU, float64 | soak | 111 properties, 0 failed (87 in the first run, which reached its 90-minute limit; the other 24 in a second) | 5400 + ~1900 s | 2.2 GB |
| Mac | MLX CPU, float32 | light battery | 283 tests, 1681 assertions, 0 failures | 128 s | 4.4 GB |
| Mac | MLX CPU, float32 | soak | 111 properties, 0 failed | 880 s | 1.4 GB |
| Thor | CUDA (sm_110) | MLX and conformance tests | 35 tests, 70 assertions, 0 failures | 12 s | 5.5 GB |
| Thor | CUDA | full battery | 298 tests, 1734 assertions, 0 failures | 8909 s | 13.4 GB |
| Thor | CUDA | soak | 110 of 111 properties; the failure was the test's assumption below, and the conformance soak rerun with the fix passed all 20 | 5661 s | 13.1 GB |
| Thor | MLX CPU, float64 | full battery | namespaces 1–15 in full (slow and heavy included) before the 4-hour limit, 0 failures; 16–24 on the light tier: 51 tests, 581 assertions, 0 failures | 14400 s + 1550 s | 13.4 GB |
| Thor | MLX CPU, float64 | soak | 50 of 111 properties by 05:56, 0 failed (its 3-hour limit ends it at 06:16; the Mac's float64 soak covers all 111) | started 03:16 | |

The light battery skips the ^:slow and ^:heavy tests, which take minutes
each on MLX's CPU (one evaluation per value read); the full batteries on
Metal and CUDA include them. The day before, the full battery also passed
on the Mac's MLX CPU in float64 (all 24 namespaces, slow and heavy, in two
runs of about 100 minutes; bean genclj-rtif), on an earlier commit.

Benchmarks: on the Mac in `docs/benchmarks.md` (genclj-ls29); CUDA on the Thor in its own section there
(the GPU is 2.5–6× slower than Metal at these sizes, from per-operation
costs; 1.2× for compiled HMC at a million observations).

## After M4's merge (main at 4d11594)

M4 (Gen.jl parity) merged into main the same night (b3d74c1). On main:

| Device | Run | Result | Time | Peak RSS |
|---|---|---|---|---|
| JVM | full battery | 469 tests, 2437 assertions, 0 failures | 109 s | |
| Metal | full battery | 469 tests, 2404 assertions, 0 failures | 2948 s | 1.6 GB |
| MLX CPU, float32 | light battery | 444 tests, 2329 assertions, 0 failures | 199 s | 3.9 GB |
| MLX CPU, float64 | light battery | 444 tests, 2325 assertions, 0 failures | 1590 s | 7.4 GB |

Metal found what the JVM could not: three M4 tests that assumed host
numbers or float64, a statistical band set too tight (now measured), and
one bug, a transform branching with a plain `if` on a boolean that MLX
holds as an array (discrete reads are now host values).

## What the final runs found and fixed

The soaks found one fault; the Thor's bring-up found the rest, each fixed
before the final runs (commits 0a5201b to 7751b76):

- **Only Metal flushes float32 subnormals.** The CUDA soak drew
  `square` of 5.4e-20 (2.9e-39, a float32 subnormal): CUDA keeps it, as
  the CPU does, but the conformance suite flushed the reference on every
  GPU. `ts/flushes-subnormals?` is now Metal's alone (00bc330); one
  passing battery had not drawn such a value.
- **A CUDA abort at exit.** A soak run that had passed every property
  aborted as the JVM exited (`Destroy(handle_) failed: driver shutting
  down`): MLX destroys a CUDA handle after the driver has shut down, which
  turns a passing exit code into 134. Not every run hits it (the full
  battery exited cleanly). Bean filed under genclj-265q.
- From the bring-up: Linux wheels bundle BLAS and LAPACK in
  `<package>.libs` (the build script copies it); GCC 13 rejects MLX's CPU
  JIT header, so CPU compile falls back to running unfused (genclj-hjzh);
  the mlx-cuda-13 wheel has no sm_110 code, so MLX v0.32.2 is built from
  source, unpatched; NVRTC needs the CUDA headers beside the library;
  MLX's CUDA graph cache throws on cumulative misses, so graphs are off by
  default; CUDA evaluates on one thread; CUDA words a host read inside
  `compile` differently; a raw `value_and_grad` test built its backward
  pass on the GPU and hung.

## Machines

- Mac: Apple M4, 32 GB, macOS 26, Oracle GraalVM 25.0.4, MLX 0.32.2 from
  the mlx-metal wheel.
- Jetson AGX Thor: 14 cores, 122 GB, Ubuntu 24.04, CUDA 13.2 (driver 595),
  GraalVM 25.0.4; MLX 0.32.2 from the mlx-cpu wheel (CPU) and built from its
  source for sm_110 (GPU; `native/build-mlx.sh cuda13-source`), unpatched.
