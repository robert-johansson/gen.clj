# Verification: idiom-arrays.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

Every cited line was read in the snapshot worktree, with its callers. The
scripts ran on the JVM backend, nice'd, with `-J-Xmx1g`. MLX claims were
checked by reading gen.backend.mlx, MlxRuntime.java, and the mlx-c and MLX
sources under `native/build/src` in the main checkout. Nothing ran on MLX.

## Verdicts

| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| I1-01 | lgamma on a host scalar: ~700 ns, 1.7 KB a call, on the hot path of the gamma-family log densities | CONFIRMED | major → major |
| I1-02 | Both backends repeat host-data parsing and the shape checks | CONFIRMED (the drift evidence is weaker than stated) | major → major |
| I1-03 | gen.backend.mlx mixes five concerns; a layered split | CONFIRMED (the proposed layer order is wrong in two places) | moderate → moderate |
| I1-04 | The temporary-handle idiom, written out ten times, leaks where it slipped; device handles are never freed | CONFIRMED (and more sites than listed) | moderate → moderate |
| I1-05 | (correctness) gen.array.linalg gives the backend operands of two dtypes; the JVM result takes the matrix's | CONFIRMED (the direction needs two caveats) | moderate → moderate |
| I1-06 | collect! is a second, weaker collection beside collectNow | CONFIRMED | moderate → moderate |
| I1-07 | -native-ops promises a general mechanism; only :logaddexp is read | CONFIRMED | moderate → moderate |
| I1-08 | take's docstring says out-of-range indices throw; MLX clamps them | CONFIRMED | moderate → moderate |
| I1-09 | (correctness) diag of a vector and la/diagonal turn 0·Inf into NaN | CONFIRMED | minor → minor |
| I1-10 | gen.array stays one namespace; host math and trees move out; requiring-resolve per call | CONFIRMED | minor → minor |
| I1-11 | Small duplications | CONFIRMED | minor → minor |
| I1-12 | Small idiom points | CONFIRMED (one direction needs a change) | minor → minor |
| I1-13 | Docstrings, docs and one name | CONFIRMED, except the rename, which is DECIDED | minor → minor |
| I1-14 | Comments that narrate history | CONFIRMED in part (two of the seven sites are not history) | minor → minor |

Counts: 14 CONFIRMED, 0 PLAUSIBLE, 0 REFUTED, 0 DECIDED. No severity
changed.

Within the confirmed findings:
- I1-13's rename is a DECIDED sub-point;
- two of I1-14's sites are not history;
- the directions of I1-03 (layer order), I1-05 (inv and the tape) and I1-12
  (the uint64 code) need correcting.

Missed: three items, below. One is new and runnable: a 0-d right-hand side
gives a NullPointerException on the JVM backend.

## Findings

### I1-01: CONFIRMED (major)

**Read.** `src/gen/array.clj:468-525`. On a host Double, every op of the
composite goes through host-binary and host-unary. Both branches of the
reflection are computed. `lgamma-lanczos` destructures the coefficient
vector and sums a `map-indexed` lazy seq. The call lines in dist.clj are
right, and they show the finding is wider than "seven distributions": they
are the log densities of eleven families (beta, gamma, inv-gamma,
dirichlet, chi-squared, f-dist, nakagami, student-t, poisson, binom and
neg-binom), plus the poisson and binomial samplers (578, 606, 616).
digamma has the same shape, but nothing in `src/` calls it.

**Reran** `OUT/scripts/I1-01.clj`. A second JVM, another agent's, was
running at the same time, so treat these as upper bounds:

| Call | Reviewer | Rerun |
|---|---|---|
| `array/exp` | 8 ns, 8 B | 8.2 ns, 11 B |
| `array/lgamma` | 680–691 ns, 1,688 B | 711 ns, 1,688 B |
| primitive Lanczos (x ≥ 0.5 only) | 55–60 ns, 44 B | 21.7 ns, 45 B |
| poisson log-prob | 1.16–1.18 µs, 1,545 B | 1.17 µs, 1,546 B |
| normal log-prob | 78 ns, 216 B | 77.5 ns, 216 B |

**My own check**, `OUT/scripts/V-I1-01.clj` (also beside another agent's
JVM), used criterium's quick-bench (mean and sd) and the thread's
allocated bytes. It tested a primitive
`lgamma-host` that does the composite's double operations in the same
order, with the reflection and the poles included. That is the
direction's claim, not the reviewer's partial loop.

```
points checked: 200019  differing bitwise: 0  first: ()
long input 5 -> 3.1780538303479444 vs 3.1780538303479444
array/exp 3.7                       11.7 ns (sd    0.1)       14 bytes/call
array/lgamma 3.7                   701.7 ns (sd    3.7)     1688 bytes/call
array/lgamma 0.3 (reflected)       702.9 ns (sd    4.1)     1688 bytes/call
array/digamma 3.7                 1274.3 ns (sd   10.2)     2072 bytes/call
array/log-beta 2.5 1.5            2176.3 ns (sd    9.9)     5112 bytes/call
primitive lgamma-host 3.7           12.8 ns (sd    0.0)       24 bytes/call
primitive lgamma-host 0.3           21.0 ns (sd    0.1)       24 bytes/call
normal log-prob                     73.9 ns (sd    0.3)      192 bytes/call
gamma log-prob                     957.0 ns (sd    3.9)     1952 bytes/call
beta log-prob                     2534.2 ns (sd   11.8)     5400 bytes/call
poisson log-prob                  1224.6 ns (sd    4.6)     1544 bytes/call
```

The 200,019 points were ±0, the poles 0, -1 and -2, -2.5, -0.3, ±1e-300,
0.5 and the double just below it, 171.6, 1e10, 1e300, ±Inf, NaN, and
200,000 random points, log-spread over [-3,000, 7,000]. So the number holds:
about 700 ns and 1.7 KB a call. log-beta, as three of them, is 2.2 µs and
5.1 KB, so a scalar beta log density costs 34 times a normal one. The
direction holds too: a host path in the composite's operation order equals
the composite bit for bit on every point, at 13 to 21 ns, about 33 to 55
times faster.

**Direction.** Sound. It matches how erf and erfinv already sit in
host-unary, and tracked values (a Tracked is not a Number) and batched
values keep the composite. Two details:
- gen.array.special's ns docstring says lgamma and digamma are composites
  in gen.array, and would need updating.
- The guarding property belongs beside `unary-host-and-array-semantics-agree`
  (array_props_test.clj:157), whose op list has neither erf nor lgamma
  today.

For a Long argument (poisson's `(+ v 1)`), the composite runs part of the
work in long arithmetic. That is exact below 2^53, so converting to double
first gives the same value (checked at 5).

**Known?** No bean. genclj-4n51 (the CPU health pass) is the natural parent.

### I1-02: CONFIRMED (major), the drift evidence weaker than stated

**Read.**
- The host-data parsers:
  - `host-shape` is identical in both files (jvm.clj:761, mlx.clj:419).
  - `host-leaves` differs only in `host?` against the inline instance test
    (jvm.clj:767, mlx.clj:425).
  - `infer-dtype` differs only in reading the backend's defaults (jvm.clj:783,
    mlx.clj:441).
- Every row of the shape-check table is at the cited lines, with the same
  messages and data.
- gen.array does not validate before dispatch: `broadcast-to` (array.clj:838)
  and `concatenate` (:841) call the protocol directly.

**The drift.**
- *The solve-triangular rank guard* (jvm.clj:632 against :693) is real in
  the code, but no call through gen.array.linalg reaches it.
  `with-matrix-rhs` (linalg.clj:20-27) expands a 1-d right-hand side to
  [n 1]. A 0-d right-hand side reaches both JVM solves, and there
  `k (long (peek sb))`, at jvm.clj:631 and :692, comes before either guard.
  So the JVM's solve guard does not catch the rank-0 case either. See
  `OUT/scripts/V-I1-02.clj` below, and Missed.
- *take-along-axis* is not a drift. Both backends call `sh/broadcast` on
  each non-axis pair and throw the same error; only the code shape differs.

**The batch-tracer claim**, run in `OUT/scripts/V-I1-05.clj`:

```
  unbatched [3] -> [2 2]     Shapes [3] and [2 2] do not broadcast {:shapes [[3] [2 2]]}
  batched [5]x[3] -> [2 2]   Shapes [5 1 3] and [5 2 2] do not broadcast {:shapes [[5 1 3] [5 2 2]]}
  batched concatenate        Arrays must agree on every axis but the concatenation axis {:shapes [[5 3 2] [5 3 4]], :axis 1}
```

The claim holds, and holds for concatenate too. The model wrote
per-particle shapes [3] → [2 2], and [3 2] with [3 4] on axis 0, but the
error names the lifted shapes, and for concatenate a shifted axis (1).

**Direction.** Sound. It keeps rule 6, since the errors stay ex-info with
data, and the wrapping backends uphold the new guarantee by construction:
- the batch tracer lifts shapes that gen.array has already validated per
  particle;
- the tape passes the shapes of the primals.

The guarantee in protocols.clj then becomes a contract that the batch and
tape backends must keep when they call their inner backend, and it should
say so. On MLX, a check removed from the backend leaves MLX's own
exception as a second line, which is a loud MlxException wrapped by
`guarded`, not silence.

**Severity.** Major stands: the scale's "duplication of real logic", and the
batch-tracer errors improve as a result.

### I1-03: CONFIRMED (moderate); the proposed layer order is wrong in two places

**Read.** The five concerns and the line ranges are accurate, and so is the
seam through `wrap`, `with-operands`, `stream` and `evaluating`. The finding
stands. The direction says "each namespace requires only the ones above it",
and two dependencies break that order:
- **Handles need the runtime.** `made` (mlx.clj:276-283) calls
  `at-buffer-limit` (:227), and `to-host` (:537) calls `evaluating` (:264).
  Both are in layer 1, while `at-buffer-limit` and `evaluating` are layer 2.
  Nothing in the runtime needs the handles. The runtime goes first.
- **Everything needs the Config, and the Config is reached through the
  deftype.**
  - `stream`, `cpu-stream` and `defaults` (:145-147) read `(config b)`,
    which is `.-config ^MlxBackend` (:1519), and the deftype is in layer 5.
  - Layer 1's `from-host` calls `gpu-dtype!`, which calls `device`, also in
    layer 5.

  So Config and its accessors (stream, cpu-stream, defaults, device) must
  live in the lowest layer, with the backend's config reachable without the
  deftype: a `definterface` that MlxBackend implements, or the Config in the
  lowest namespace with MlxBackend built over it.

With those two changes the layering works. Transforms then need handles and
runtime, ops need transforms (for exact-route and custom-vjp), and the
backend needs ops, as the reviewer says.

**Known?** genclj-mkgo ("Split gen.backend.mlx into namespaces by concern …
after the idiom review has weighed in on the shape"), as cited.

### I1-04: CONFIRMED (moderate), with more sites than listed

**Read.**
- The ten sites in the table build `temps` and `t` by hand as described.
- The three leaks are as stated:
  - int-division makes `zero` and `one` (mlx.clj:668-669) before its try (:671);
  - the integer matmul makes `fx`, `fy` and `m` (:971-973) in the let
    before its try;
  - random-uniform makes `kw`, `k0`, `k1`, `c1` and `c0` (:1235-1243)
    before its try at :1244.

  In each case a throw from a later constructor (`made` throws after its
  retry at the buffer limit) leaves the earlier handles unfreed.

**Device handles.** mlx-c's `mlx_device_new_type` returns
`mlx_device_new_(mlx::core::Device(...))`, which is `new mlx::core::Device`
(`native/build/src/mlx-c/mlx/c/device.cpp:26-33` and
`mlx/c/private/device.h:16-17`). `mlx_stream_new_thread_unsafe` takes
`mlx_device_get_(dev)` by value (`stream.cpp:35-43`), so the device stays the
caller's to free. The generated `Mlx.mlx_device_new_type` returns the raw
handle (Mlx.java:1368), and the following never free it:
- gpu? (:1527), which make-backend calls up to twice (:1534-1535);
- make-backend's two device-handle calls (:1542-1543);
- aux-stream (:151);
- one-eval-thread? (:163).

with-default-stream (:1305) and MlxRuntime.maxArrays (MlxRuntime.java:484)
do free theirs. Each leak is a few bytes, once per backend made or per
`gpu?` call, so the point is consistency, not memory.

**Missed sites of the same slip:**
- unary's sign and arctan branches make `sg` and `nan`, or `at` and `zero`
  (:639-640, :647-648), before their try;
- custom-vjp makes `fun` and then `bwd` (:1134-1137) before its try;
- with-operands (:397-398) binds every operand's handle before its try, so
  when a later operand's `scalar-handle` fails (constant cache full, at the
  buffer limit), the earlier operand's new handle leaks.

These are all failure-path leaks of one or a few handles, on a path that is
already throwing loudly, so moderate is right. The direction (a
`with-temps` macro through which every handle goes) closes them by
construction, with-operands included if it registers its new handles the
same way.

### I1-05: CONFIRMED (moderate); the direction needs two caveats

**Read.** linalg.clj:12-16 converts each operand on its own, with no common
dtype, and protocols.clj:20-21 promises one shared dtype. The JVM's
solve-triangular and solve return `(dtype-of a)` (jvm.clj:653 and :718), and
`make` conforms the data to that dtype, so a float32 matrix rounds a
float64 solve to float32.

**Reran** `OUT/scripts/I1-05.clj`, plus my own values in
`OUT/scripts/V-I1-05.clj`:

```
matmul f32·f64          -> :float64 (gen.array promotes)
matmul f64·f32          -> :float64
solve f32 matrix, f64 rhs -> :float32
solve f64 matrix, f32 rhs -> :float64
solve-triangular f32, f64 -> :float32
solve-triangular f64, f32 -> :float64
  solve f64 m, f64 rhs               :float64 [-0.09999999999999998 0.39999999999999997]
  solve f32 m, f64 rhs               :float32 [-0.10000000149011612 0.4000000059604645]
  solve f64 m, f32 rhs               :float64 [-0.09999999701976776 0.3999999925494194]
  matmul f32 m, f64 v                :float64 [1.0 1.5]
  + f32, f64                         :float64 [0.20000000149011612 1.399999988079071]
  inv f32 m (eye is the default float) :float32 [[0.4000000059604645 -0.20000000298023224] ...]
```

The reviewer's table reproduces. The values show the asymmetry: with the
same exact float32 matrix ([[3 1] [1 2]]), a float64 right-hand side comes
back rounded to float32, while every elementwise op and matmul promote to
float64.

**MLX disagrees, by reading.** MLX's `validate_solve` takes
`promote_types(a.dtype(), b.dtype())`, and its `solve_triangular` is
`matmul(tri_inv(a), b)` (`native/build/src/mlx/mlx/linalg.cpp:681-718`),
so a mixed solve promotes to float64 there. The JVM backend returns float32
for the same call. So this is a cross-backend divergence as well as a
contract breach. It needs a CPU-device MLX backend with float64, which is
where mixed dtypes can arise on MLX. Not run on MLX.

**Two caveats to the direction:**
- **inv.** `la/inv` builds its identity with `(array/eye n)` on the
  default float dtype (linalg.clj:58). Today the JVM gives a float32
  matrix's inverse as float32, the matrix's dtype; see the output above.
  Promoting both operands would make it float64. The identity should take
  m's float dtype, `(eye n {:dtype …})`, so that promotion does not change
  inv's result dtype.
- **The tape.** The tape's linalg VJPs call `p/-linalg` directly, and also
  with an identity of the default dtype: `cholesky-vjp`'s `eye`
  (tape.clj:216) and slogdet's `(array/eye n)` against `m` (:251). So
  promoting in gen.array.linalg alone leaves the tape's backward pass
  passing mixed dtypes. The fix should cover linalg-vjp too, or put the
  promotion where both paths meet.

**Known?** New, as stated.

### I1-06: CONFIRMED (moderate)

**Read.** collect! (mlx.clj:1607-1614) is `System/gc`, a fixed
`Thread/sleep 20`, a `drain` and `(memory)`. collectNow
(MlxRuntime.java:414-437):
- counts the collection;
- resets FRESH;
- drains until the queue stays empty for 10 ms;
- resets both thresholds.

check-budget! (:199-215) uses collect!, at-buffer-limit (:238) uses
collectNow, and the user's collect! is the same weak one. Every
consequence the reviewer lists follows from the code:
- the budget's collection is missing from `:collections`;
- FRESH stays high;
- the loud `:mlx-budget-exceeded` decision follows a fixed guess.

The tests corroborate it: `settled-active` in mlx_test.clj:303-309 calls
collect! three times to settle, a sign that one call is not enough.

DESIGN §9 records no decision on collect!. The direction is sound: collect!
as collectNow followed by memory. collectNow also raises `byteThreshold` to
twice the active memory, which at-buffer-limit already accepts. Not run
(MLX).

### I1-07: CONFIRMED (moderate)

**Read.**
- The docstring (protocols.clj:64-67) and docs/array-api.md:159-160 describe
  a general mechanism.
- gen.array reads `-native-ops` only at array.clj:439, for `:logaddexp`.
- The implementations are #{} in jvm, batch and tape, and #{:logaddexp} in
  MLX.

A backend listing `:lgamma` gets nothing. Moderate fits the scale ("a
stale or wrong docstring") for a contract docstring. Of the two directions,
the narrower statement is the right one today. array-api.md is the approved
doc, so its line changes with the owner's sign-off. See also Missed (the
batch backend's #{}).

### I1-08: CONFIRMED (moderate)

**Read.**
- take's docstring (array.clj:897-900) says "out-of-range indices throw".
- MLX's take on an array of indices goes through safe-indices
  (mlx.clj:890-914), which clamps. A host index is checked (host-index,
  :880-888).
- The clamping is decided: DESIGN.md:703-704 ("Array indices are clamped,
  not checked") and array-api.md:799 and :818.

take-along-axis and scatter-add say nothing about bounds. A docstring that
gives the JVM backend's behavior as the API's is misleading, and the
direction's text matches the decision.

### I1-09: CONFIRMED (minor)

**Read.**
- `diag` of a vector is `(* eye x)` (array.clj:968).
- `la/diagonal` is `(sum (* m eye) {:axis -1})` (linalg.clj:72-76).
- `diag` of a matrix uses take-along-axis (array.clj:969-974).

**Reran** `OUT/scripts/I1-09.clj`, plus extra cases in
`OUT/scripts/V-I1-05.clj`:

```
  diag [1 Inf]                       [[1.0 ##NaN] [0.0 ##Inf]]  expected [[1 0] [0 Inf]]
  diag [NaN 1]                       [[##NaN 0.0] [##NaN 1.0]]  expected [[NaN 0] [0 1]]
  diag [-Inf 2]                      [[##-Inf 0.0] [##NaN 2.0]] expected [[-Inf 0] [0 2]]
  la/diagonal [[1 Inf] [0 2]]        [##NaN 2.0]                expected [1 2]
  la/diagonal [[1 0] [NaN 2]]        [1.0 ##NaN]                expected [1 2]
  array/diag [[1 Inf] [0 2]]         [1.0 2.0]                  expected [1 2]
  cholesky-logdet [[1 0] [Inf 2]]    ##NaN                      expected 1.3862943611198906
```

The reviewer's three rows reproduce. The extra rows show that an entry
below the diagonal poisons it the same way, and that it reaches
cholesky-logdet: a lower factor with an infinite entry below the diagonal
gives NaN for 2·(log 1 + log 2).

The expected values are NumPy's: `np.diag([1, inf])` is `[[1, 0], [0, inf]]`,
and `np.diagonal` reads entries, so `[[1, inf], [0, 2]]` gives `[1, 2]`.
The callers (mvnormal at dist.clj:197 and :225, cholesky-logdet) see finite
Cholesky factors in any valid use, so minor is right.

**Direction.** Sound:
- `where` keeps gradients exact (its discarded branch here is a finite 0);
- take-along-axis broadcasts the index over the batch axes, so one batched
  diagonal serves both readers.

### I1-10: CONFIRMED (minor)

**Read.**
- The cycle is real. gen.array requires gen.backend.jvm statically
  (array.clj:19), and `*backend*`'s initializer (:32-36) resolves MLX at
  load time when gen.backend=mlx, so a static require of gen.array from
  gen.backend.mlx would be a cyclic load.
- `tree-leaves` and `tree-map` resolve on every call (mlx.clj:1277-1278).
- exact-route resolves on every call (:1162, and :1167 inside the VJP), and
  `unary` sends every float64 exp, sin, cos, erf and erfinv there (:622-623).
- linalg's rule (:1175) resolves on every native-mode linalg call, and
  jvm.clj:896 and mlx.clj:1517 on every gradient.
- host-elementwise and host-reduce are public, and only the tape uses them
  (tape.clj:307 and :317).

The cost is in `OUT/scripts/V-I1-05.clj`:

```
  requiring-resolve 'gen.array/tree-leaves        118.9 ns
  satisfies? p/ICompiling jvm/backend (a miss)    572.8 ns
  instance? ICompiling jvm/backend                 17.3 ns
  compiled fn on the JVM backend                  880.2 ns
  the fn itself                                    81.6 ns
```

The run shared the machine with other agents' JVMs, so these are orders of
magnitude, not benchmarks. About 120 ns a resolve is small beside MLX's
per-op overhead (genclj-j55z measured hundreds of ns), but it is pure
waste. The last two rows bear on I1-12: on the JVM backend,
`array/compile` is documented as "f itself", yet the compiled function
costs about 0.8 µs more per call than f, and most of that is the
`satisfies?` miss.

**Direction.** Sound, with one point to keep: `tree-leaves` and `tree-map`
are public API used across the codebase (choicemap, batched, custom,
gradients, mcmc and learning call `array/tree-leaves` or
`array/tree-map`). So gen.array has to keep them as public names, for
example by re-exporting gen.array.tree's.

### I1-11: CONFIRMED (minor)

**Read.**
- **The weak host dtype.** All four sites hold the same expression:
  array.clj:559 and :577, tape.clj:46 and mlx.clj:316. Within gen.array,
  `full` (:559) could call `weak-dtype` (:577) itself.
- **The one-element read loop** is at :405 and :635.
- **The erfc table and constants** (exact.clj:38-49 against special.clj:9-21)
  are the same.
- **refill** (mlx.clj:1280-1284) restates array.clj:1039-1043 over a vector
  of trees.
- **full-batch and unwrap** do the same reshape-and-broadcast. They differ
  for a host scalar:
  - full-batch fills with `(array/dtype x)`, the host's strong dtype
    (float64 for a Double);
  - unwrap uses `(array/array x)`, the weak default.

  full-batch's host branch looks unreachable from its callers (concatenate,
  take and scatter-add need rank ≥ 1). One shared helper would also settle
  that difference.
- **MlxRuntime:**
  - adopt (290-295) and adoptCompiled (372-377) repeat the list insertion;
  - closureCall and customVjpCall (553-592) differ only in the call;
  - the two `release` overloads (388 and 596) do unrelated jobs.

### I1-12: CONFIRMED (minor); one direction needs a change

Each point holds as stated:
- **zeros and ones.** `full` already gives the backend's float for a host
  0.0 (array.clj:565 and :569, against :559).
- **xlogy and xlog1py** bind `zero?` (:449 and :453).
- **compile** tests `satisfies?` (:1082), while the file uses `instance?`
  elsewhere (:53, :75, :396, :1029). Its cost on the JVM is in the
  V-I1-05 output above.
- **from-host** calls `Class/forName "[D"` on each call (jvm.clj:794,
  mlx.clj:492).
- **set-budget!** reads the atom, then swaps it (mlx.clj:1642-1644).
- **The bare `(code …)` statements** are at :489, :500 and :1483.
- **`(:device (config b))`** is at :625 and :1293, beside `device`.
- **The bare `(int 4)`** is at :1254.
- **native-value-and-grad's else branch** has an `(if` and its else
  `(try` both at column 4 (:1331 and :1335).
- **with-operands** splices each operand form three times (:397, :401 and
  :404). Every call site passes a symbol (all 22 checked).
- **from-values** boxes the double[] (:494).
- **r-max, r-min, cummax and cummin.** Java's `Math.max(double, double)`
  already returns NaN when either argument is NaN, so the tests before it
  (jvm.clj:304-305, 394-395) are redundant.
- **scan-op** dispatches `case` per element (:384-397).

The one change: "`(int 4)` is a magic code for uint64 that is missing from
dtype-codes" should not be fixed by adding `:uint64` to `dtype-codes`. That
map is the set of dtypes the backend supports. `code-dtypes` is derived
from it, so `dtype-of-code` would then accept a uint64 result, and `code`
would accept `:uint64` as a user dtype. A named private constant (`uint64-code`)
says what it is without widening the dtypes.

### I1-13: CONFIRMED, except the rename, which is DECIDED

**Confirmed by reading:**
- the missing docstrings: array.clj:125, 133, 563, 567 and 589-590;
  batch.clj:78 and 302; tape.clj:37; dtype.clj:127, 159 and 165;
- the section header (array.clj:1101), whose functions' own docstring says
  "For validation";
- MlxRuntime's class doc (:23-25);
- array-api.md §6, which lists `var` (array.clj names it `variance`, :731)
  and `tile` (which does not exist) at :176 and :179;
- §3's "`(array/to-backend x b)`, comes with M2" (:103), never built.

**DECIDED.** The name `gen.array.memory` is part of DESIGN §9, "The shared
memory budget" (DESIGN.md:1345-1365: "2026-10-21, agent; confirmed by the
owner, 2026-10-21"), which names it explicitly. The reviewer's argument
belongs under "Decisions worth revisiting", and it is a good one. Every
require aliases the namespace `reservations` (mlx.clj:62, llama.clj:58 and
two tests), against CLAUDE.md rule 4 ("call the rest through the alias …
the namespace's last segment"), and it sits beside gen.backend.mlx's own
`memory` function, which means something else.

### I1-14: CONFIRMED in part (minor)

Comments that narrate history, site by site:
- **MlxRuntime.java:169-173** ("as large ones once were"), **443-445** ("an
  independent review, 2026-10-19") and **454-459** ("It did, and the cap
  was no limit on Metal … 176 errors"): confirmed. They recount code that no
  longer exists.
- **mlx.clj:555-558**, "Copying every array cost a device round trip per
  read": confirmed. It describes the replaced approach, although it doubles
  as the reason.
- **mlx.clj:903-905**: refuted. It says why the scalars go through `made`,
  a current constraint, with a bean id. There is no history in it.
- **mlx.clj:1107-1112** and **:17-20**: refuted as narration. Each states a
  current constraint (custom VJPs stay out of compiled graphs because of the
  exit crash; MLX's default limits are all of RAM, and its limit is no
  cap). The date is provenance. At most the dates themselves could go.

Dated comments are rare in `src/` and MlxRuntime.java: 11 lines, all in mlx.clj, MlxRuntime.java
and llama.clj. So the narrower version of the finding fits the codebase's
convention.

## Decisions worth revisiting (the reviewer's §3.4 note)

The list is accurate:
- protocols.clj:24-27 (lazy-backends?);
- batch.clj:292 (the batch backends' interning cache);
- memory.clj:16 (the reservations registry);
- array.clj:1070 (compile's trace cache);
- mlx.clj:186, 323 and 1405 (MLX's budget limits, constant cache and
  CPU-compile probe).

The three MLX items arguably fall under §3.4's third place, "native handles
inside the membrane". The four L0 ones are real additions. The proposed
restatement, that semantics never reads mutable state and the caches and
registries are listed, is the principle's point.

## Missed

- **A 0-d right-hand side gives a NullPointerException on the JVM
  backend** (rule 6; array-api.md §10: shape mismatches throw ex-info).
  Through `OUT/scripts/V-I1-02.clj`, `(la/solve (array/eye 2) 1.0)` and the
  same call to `la/solve-triangular` both throw
  `java.lang.NullPointerException` at `clojure.lang.RT.longCast` from
  `gen.backend.jvm$solve` (jvm.clj:692). `k (long (peek sb))` runs before
  any check of the right-hand side, at :631 in solve-triangular too.
  MLX's `check-rhs` (mlx.clj:1000-1005) rejects the same call with ex-info
  (by reading), so the backends disagree here as well. I1-02's direction
  (one validation in gen.array.linalg, before dispatch) would fix it. Rank 1
  never reaches the backends, since `with-matrix-rhs` expands it.
- **The batch backend reports no native ops** (batch.clj:99: #{}, not its
  inner backend's). `logaddexp` (array.clj:432-443) therefore takes MLX's
  native op when it runs unbatched and the composite under the batch
  tracer. The same model code takes two implementations, which can differ in
  rounding. Read, not run on MLX. No comment says the #{} is deliberate (the tape's is, tape.clj:15). If it is not,
  the fix is one line, `(-native-ops [_] (p/-native-ops inner))`, since
  the batch tracer's -elementwise passes any op keyword through.
- **The tape mixes dtypes in its linalg VJPs**: see I1-05. `cholesky-vjp`
  (tape.clj:216) and slogdet's rule (:251) build identities of the default
  float dtype and call `p/-linalg` directly.
