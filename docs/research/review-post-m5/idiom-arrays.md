# Post-M5 review (idiom): I1, arrays, backends and randomness

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

On the JVM the layer is in good shape. Its kernels are small primitive loops with no reflection or boxed-math warnings. Host scalars take a path of a few nanoseconds. The tape and the batch tracer are compact and purely functional, and Threefry is unrolled on primitive longs. The findings fall in three places:

- **A hot path.** `array/lgamma` on a host scalar costs about 690 ns and 1.7 KB per call, 80 times `exp`. So every scalar gamma, beta, poisson, binomial or negative-binomial log density pays about a microsecond (I1-01).
- **Duplication.** Both backends implement the same host-data parsing and shape validation (I1-02). gen.backend.mlx writes one resource idiom out by hand ten times, and it leaks on the failure paths where the copies slipped (I1-04).
- **Structure.** gen.backend.mlx now holds more than one idea. I1-03 proposes a layered split for genclj-mkgo. gen.array should stay one public namespace, with two internal parts moved out (I1-10).

Two correctness points came up along the way. gen.array.linalg hands the backends operands of two dtypes (I1-05). `diag` and `la/diagonal` turn 0·Inf into NaN (I1-09).

## Findings

### I1-01: lgamma (and log-beta) on a host scalar allocate 1.7 KB and cost about 690 ns, the hot path of seven distributions' log densities

- **Severity:** major
- **Where:**
  - `src/gen/array.clj:476-497` (`lgamma-lanczos`, `lgamma`), `:522-525` (`log-beta`), and `:499-520` (`digamma`, which has the same shape).
  - Callers in `src/gen/dist.clj`: lines 84, 93, 103, 292, 324, 347, 407, 432, 578, 606, 616, 628, 638 and 654.
- **What:** lgamma is a composite over gen.array ops. That is right for arrays and the tape, but there is no host path:
  - On a host Double, every op of the composite goes through host dispatch.
  - Both branches of the reflection formula are computed.
  - The Lanczos sum is built with `map-indexed` over a destructured vector, which makes a lazy seq and eight closures per call.

  The goal of array-api §1.3, "scalar models at near-primitive speed", holds for normal (79 ns). It does not hold for the families whose densities need log Γ.
- **Evidence:** `OUT/scripts/I1-01.clj` on the JVM backend, two runs:

  | Call | Time | Allocated |
  |---|---|---|
  | `array/exp` | 8 ns | 8 bytes |
  | `array/lgamma` | 680–691 ns | 1,688 bytes |
  | the same Lanczos sum as a primitive double loop | 55–60 ns | 44 bytes |
  | poisson log-prob | 1.16–1.18 µs | 1,545 bytes |
  | normal log-prob | 78 ns | 216 bytes |

  The primitive loop gives the same value, 1.428072326665391 at 3.7. log-beta, used in beta's density, is three lgamma calls.
- **Direction:**
  - Give lgamma a host-scalar path, as erf and erfinv already have (`gen.array.special`, through `host-unary`). That means a primitive lgamma (Lanczos, reflection and poles) and digamma in gen.array.special, taken by `gen.array/lgamma` when x is a host scalar.
  - Arrays and tracked values keep the composite, so gradients and agreement across backends do not change.
  - If the host path performs the same sequence of double operations, its host values equal the composite's bit for bit (rule 1: observationally equivalent).
  - A property test guards it: the host path equals the composite on 0-d inputs, including x < 0.5 and the poles.
  - This keeps the intent of array-api §12.5: one definition, differentiable and identical on every backend. Like host-unary's erf, the host path is an optimization of host math.
  - Even without a host path, the composite's `map-indexed` sum could be a `reduce` over a constant vector of `[i c]` pairs.
- **Known?** New. No bean covers lgamma's host cost; genclj-a7s7 holds the JVM backend's performance work.

### I1-02: Both backends implement the same host-data parsing and shape validation

- **Severity:** major
- **Where:**
  - Host data (`host-shape` and `host-leaves`, word for word, and `infer-dtype`): `src/gen/backend/jvm.clj:761-787` and `src/gen/backend/mlx.clj:419-445`.
  - The double[] test: jvm.clj:794 and mlx.clj:492.
  - The shape checks, in both files:

  | Check | jvm.clj | mlx.clj |
  |---|---|---|
  | `broadcast-to` | 417-421 | 836-840 |
  | `concatenate` | 424-431 | 844-851 |
  | `take-along-axis` | 483-490 | 927-933 |
  | `scatter-add` | 510-516 | 939-944 |
  | `matmul` | 529-540 | 955-962 |
  | square matrices | 581-586 | 977-981 |
  | right-hand sides | 632-634, 693-694 | 1000-1005 |
- **What:** None of this depends on the backend. It is the contract's validation, with the same error text and data in both files, and only the conformance suite's rule that "errors must agree" keeps the two copies in line. They have already drifted:
  - The JVM's solve-triangular checks the right-hand side without the rank guard that its solve has (jvm.clj:632 against :693).
  - The JVM checks take-along-axis's other axes while it computes the output shape; MLX checks them in a loop.

  Because the checks run inside the backend, under the batch tracer they see the lifted shapes. A per-particle broadcast error names the batch axes ([N 1 3] to [N 2 2]) instead of the model's shapes.
- **Evidence:** read. The table above pairs the copies.
- **Direction:**
  - gen.array and gen.array.linalg validate once, before dispatch, with the shape helpers in gen.array.shape (for example `sh/concatenate-shape`, `sh/matmul-shape`, `sh/check-square`).
  - Host data is parsed once into `[shape leaves dtype]`, in gen.array.dtype or a `gen.array.host`. Each backend's from-host then only fills its own storage.
  - The guarantees in protocols.clj gain "shapes are compatible; host data is rectangular".
  - The two backends lose about 100 lines between them, and the batch tracer's errors name the per-particle shapes.
- **Known?** New.

### I1-03: gen.backend.mlx mixes five concerns; a layered split

- **Severity:** moderate
- **Where:** `src/gen/backend/mlx.clj`, 1,651 lines.
- **What:** The namespace holds five things:
  - Handles and the host boundary: lines 79-137 and 270-583.
  - The runtime: the evaluation lock and thread, Metal's buffer limit, the budget, collection, and MLX's own limits. Lines 139-268 and 1548-1644.
  - The ops with NumPy's semantics: lines 585-1100 and 1184-1265.
  - The transformations: custom VJPs, native gradients, compile and its CPU probe. Lines 1102-1182 and 1267-1440.
  - The backend value: lines 1442-1546.

  These parts meet only through a few functions (`wrap`, `with-operands`, `stream`, `evaluating`), which marks a natural seam. This is the shape genclj-mkgo waits for, as a layering in dependency order:
  1. **`gen.backend.mlx.handles`:** MlxArray, wrap, fence, made, the scalar handles and the constant cache, operand, with-operands and retained, `with-temps` (I1-04), and from-host and to-host.
  2. **`gen.backend.mlx.runtime`** (the bean's ".memory"): the evaluation lock and thread, at-buffer-limit, evaluating, the budget, collect!, memory, the set-*-limit! functions and evaluated?.
  3. **`gen.backend.mlx.transforms`:** `*tracing-compile*`, `*differentiating*`, custom-vjp, exact-route, native-value-and-grad and compile-fn.
  4. **`gen.backend.mlx.ops`:**
     - elementwise, with the NumPy emulations (int-division, java-int-cast, int-power, sign, arctan, arg-reduce);
     - reductions, shape and indexing;
     - linalg* with its custom-VJP wrapper;
     - Threefry.

     The emulations are §13's table written as code. They could have a namespace of their own beside gen.backend.mlx.exact, which sets the precedent.
  5. **`gen.backend.mlx`:** the deftype, Config, make-backend, backend, device, and compiles? (the CPU probe needs the ops).

  Each namespace requires only the ones above it. The linalg wrapper sits in ops, so transforms never needs to require ops. The public functions (make-backend, backend, memory, budget, collect!, …) stay reachable from gen.backend.mlx, and the tests do not change.
- **Evidence:** read.
- **Direction:** as above, with I1-04's macro first, which takes out about 40 lines.
- **Known?** genclj-mkgo. This is the shape that bean is waiting for.

### I1-04: The temporary-handle idiom is written out ten times, and leaks where it slipped

- **Severity:** moderate
- **Where:**

  | Function | Lines in mlx.clj |
  |---|---|
  | to-host | 549/569 |
  | int-division | 666/681 |
  | java-int-cast | 692/705 |
  | int-power | 716/737 |
  | arg-reduce | 780/793 |
  | safe-indices | 897/914 |
  | linalg* | 1014/1100 |
  | random-uniform | 1230/1265 |
  | custom-vjp (tagged variant) | 1138/1150 |
  | native-value-and-grad (tagged variant) | 1330/1348 |
- **What:** Each site builds `temps (java.util.ArrayList.)` and a local `t` that registers a handle and returns it, then frees them all in a `finally`.
  - Because the idiom is written by hand, the try does not always cover every handle:
    - int-division makes `zero` and `one` before its try (668-669).
    - The integer matmul makes `fx`, `fy` and `m` outside a try (971-975).
    - random-uniform makes the key buffer, both key words and both counters (1235-1243) before its try at 1244.

    Each of these leaks on the failure path that `made`'s retry at Metal's buffer limit exists for.
  - Device handles from `mlx_device_new_type` are never freed in `gpu?` (1527), make-backend (1534-1543), aux-stream (151) and one-eval-thread? (163). In mlx-c each one is a heap-allocated Device (device.cpp:26-29, private/device.h:16-17). with-default-stream (1305) and MlxRuntime.maxArrays do free theirs.
- **Evidence:** read. The device allocation was checked in the mlx-c sources under `native/build/src/mlx-c`.
- **Direction:**
  - One macro, `(with-temps [t] body)`, binds a registering function and frees everything in its finally.
  - A variant takes a free function for each kind, for closures and vectors (or registers `[free h]` pairs).
  - Every handle a site makes then goes through `t` inside the macro, so the failure paths free by construction.
  - The callers of device-handle free what they make, or use a `with-device` helper.
- **Known?** New.

### I1-05 (correctness): gen.array.linalg gives the backends operands of two dtypes, and on the JVM the result takes the matrix's

- **Severity:** moderate
- **Where:**
  - `src/gen/array/linalg.clj:12-16` (`float-array-of`, applied to each operand), `:37-51` (solve-triangular, solve) and `:53-58` (inv).
  - The contract: `src/gen/array/protocols.clj:20-21`.
  - The JVM result dtype, `(dtype-of a)`: jvm.clj:653 and :718.
- **What:** The protocol promises "operands share one dtype (promotion already applied)", and gen.array's matmul does promote (array.clj:949-951). The linalg functions instead convert each operand to a float on its own and pass both on. So a float32 matrix with a float64 right-hand side reaches the backend mixed. The JVM backend returns the matrix's dtype, so the result's precision depends on which operand happened to be float32.
- **Evidence:** `OUT/scripts/I1-05.clj` on the JVM:

  | Call | Result dtype |
  |---|---|
  | matmul, f32 · f64 | :float64 |
  | matmul, f64 · f32 | :float64 |
  | solve or solve-triangular, f32 matrix, f64 rhs | :float32 |
  | solve or solve-triangular, f64 matrix, f32 rhs | :float64 |

  Not run on MLX (MLX JVMs are rationed). Either way the call is outside the contract.
- **Direction:** promote both operands to their common float dtype in gen.array.linalg (with `common-dtype`, as matmul does), so the backends and the tape's linalg-vjp see one dtype.
- **Known?** New.

### I1-06: collect! is a second, weaker collection beside MlxRuntime.collectNow, and the budget check uses it

- **Severity:** moderate
- **Where:**
  - `src/gen/backend/mlx.clj:1607-1614`, collect!: `System/gc`, a fixed 20 ms sleep, then drain. Its caller is check-budget! (199-215).
  - `MlxRuntime.java:414-437`, collectNow, which at-buffer-limit uses (mlx.clj:238).
- **What:** One backend has two procedures for one job:
  - collectNow waits until the reference queue stays empty for 10 ms. It also resets the fresh-bytes counter, raises the thresholds and counts the collection.
  - collect! guesses 20 ms and does none of the rest.

  So a budget check's collection is missing from `(:collections (memory))` and leaves FRESH high, so the next array made may start another collection at once. It also measures memory after a fixed guess rather than after the queue is quiet, and that measurement decides the loud `:mlx-budget-exceeded` error.
- **Evidence:** read.
- **Direction:** make collect! `(MlxRuntime/collectNow)` followed by `(memory)`, so the buffer limit, the budget and the user share one procedure.
- **Known?** New.

### I1-07: The backend contract offers native composites, but gen.array checks -native-ops only for logaddexp

- **Severity:** moderate
- **Where:**
  - `src/gen/array/protocols.clj:64-67`: "e.g. #{:logsumexp :lgamma}. gen.array uses the native version when present".
  - `array.clj:439`, the only place gen.array reads it.
  - `docs/array-api.md:159-160`.
- **What:** The docstring describes a general mechanism, in which any composite may be native, and names two composites that gen.array never looks up. What exists is narrower: a backend may handle `:logaddexp` as a primitive elementwise op keyword. That keyword is in `dt/elementwise-ops`, and MLX's -elementwise takes it. A backend author who reads the contract and lists `:lgamma` gets nothing.
- **Evidence:** grep for `-native-ops`: array.clj:439, batch.clj:99, tape.clj:280, jvm.clj:855, mlx.clj:1473.
- **Direction:** do one of two things:
  - State the narrower truth: "optional primitive ops: :logaddexp; gen.array uses its composite otherwise".
  - Make it general with one helper, `(native-or b :lgamma composite x)`, that lgamma, digamma, logsumexp and softplus all go through.

  The first is enough today.
- **Known?** New.

### I1-08: array/take's docstring says out-of-range indices throw, but on MLX array indices are clamped

- **Severity:** moderate
- **Where:**
  - `src/gen/array.clj:897-905`.
  - mlx.clj:890-925, where safe-indices clamps.
  - The decision: DESIGN §9 ("Array indices are clamped, not checked") and array-api.md §12.17 and §13.
- **What:** The public docstring gives the JVM backend's behavior as the API's. On MLX a host index is checked, but an array of indices is clamped (a decided behavior). So on a 3-element axis, `(array/take x (array/array [7]))` returns the last element there. The docstrings of take-along-axis and scatter-add say nothing about bounds. They should state the decided rule.
- **Direction:** in the docstrings of the three gathers: "Negative indices count from the end. Out-of-range indices throw on an eager backend and for host indices; a lazy backend clamps array indices (docs/array-api.md §13)".
- **Known?** New. The behavior is decided (§12.17); the docstring has not caught up.

### I1-09 (correctness): diag of a vector and la/diagonal multiply by the identity, so 0·Inf and 0·NaN become NaN; and there are two diagonals

- **Severity:** minor
- **Where:**
  - `src/gen/array.clj:964-975` (diag) and `src/gen/array/linalg.clj:72-76` (diagonal).
  - Callers of la/diagonal: dist.clj:197 and :225, and linalg.clj:81 (cholesky-logdet).
- **What:**
  - `(diag v)` is `(* eye v)`, and `(la/diagonal m)` is the sum over the last axis of `(* m eye)`. An infinite or NaN entry poisons the zeros beside it.
  - array/diag of a matrix already reads the diagonal exactly, with take-along-axis. So the library has two diagonal readers: one exact for 2-d, and one batched but wrong at non-finite entries, which also does O(n²) work for an O(n) read.
- **Evidence:** `OUT/scripts/I1-09.clj`:

  | Call | Result | Expected |
  |---|---|---|
  | `(diag [1 Inf])` | [[1 NaN] [0 Inf]] | [[1 0] [0 Inf]] |
  | `(diag [NaN 1])` | [[NaN 0] [NaN 1]] | [[NaN 0] [0 1]] |
  | `(la/diagonal [[1 Inf] [0 2]])` | [NaN 2] | [1 2] |

  array/diag of that same matrix gives [1 2]. In practice the covariances and Cholesky factors that la/diagonal sees are finite, hence minor.
- **Direction:** one batched diagonal by take-along-axis (gen.array's, which la/diagonal then uses), and `(where eye v 0)` for the matrix of a vector.
- **Known?** New.

### I1-10: gen.array stays one namespace; two internal parts move out, and the backend load cycle stops going through requiring-resolve

- **Severity:** minor
- **Where:**
  - Host math, public only so the tape can use it (host-elementwise, host-reduce): `src/gen/array.clj:154-235` and `673-681`.
  - Trees and compile's skeletons: `643-657` and `1037-1052`.
  - The cycle starts at array.clj:36: `*backend*`'s default resolves a backend at load time.
  - requiring-resolve at call sites: jvm.clj:896, and mlx.clj:1122, 1162, 1167, 1175, 1277-1278 and 1517.
- **What:**
  - gen.array's 1,133 lines are one idea: the public API that every model refers. Splitting that surface would cost every require, so it should not split.
  - Two internal parts do not belong on that surface:
    - **Host math**, an engine shared with the tape that is exposed as public functions.
    - **The tree functions**, which the MLX backend reaches through requiring-resolve on every call, because a static require of gen.array would close the load cycle.
  - The other requiring-resolve calls resolve once per op or per gradient. exact-route's runs on every float64 exp on MLX's CPU.
- **Direction:**
  - Add gen.array.host (host math, beside dtype and special) and gen.array.tree (tree-leaves, tree-map, skeleton, refill and leaves-in), both below the backends, so mlx.clj can require gen.array.tree statically.
  - Resolve the rest once, with a delay, as default-backend-var already does at mlx.clj:1122.
- **Known?** New.

### I1-11: Small duplications

- **Severity:** minor
- **Where / what:**
  - **The weak dtype of a host scalar**, `(dt/with-weak nil #{(dt/kind (dt/host-dtype x))} defaults)`, appears four times: array.clj:559 and :577 (weak-dtype), tape.clj:44-47 (primal-dtype) and mlx.clj:312-316 (host-dtype-on). One `dt/weak-host-dtype` would serve all four.
  - **The one-element read loop** appears twice: array.clj:405 (truthy?) and :635 (item).
  - **erfc's coefficients and two constants:** exact.clj:38-49 copies them from gen.array.special (special.clj:9-21). special can expose the table as a vector for both.
  - **refill:** mlx.clj:1280-1284 restates gen.array's refill (array.clj:1039-1043) over a vector of trees.
  - **Broadcasting an unbatched value to the batch:** done twice, by full-batch (batch.clj:280-290) on the inner backend and by unwrap (:335-344) through gen.array.
  - **MlxRuntime.java:**
    - adopt (290-295) and adoptCompiled (372-377) repeat the list insertion.
    - closureCall and customVjpCall (553-592) differ only in the call.
    - Two unrelated methods are both named release (388 and 596).
- **Known?** New.

### I1-12: Small idiom points

- **Severity:** minor
- **Where / what:**
  - **gen.array:**
    - zeros and ones pass `(or dtype (:float (defaults *backend*)))`, which full already gives for a host 0.0 (array.clj:565, 569).
    - xlogy and xlog1py bind a local named `zero?`, which shadows clojure.core's (array.clj:449, 453).
    - compile tests the backend with `satisfies?`, which is slow on a miss, the JVM backend's case. Every other protocol test in the file uses `instance?` on the interface (array.clj:1082; gfi.clj:82 says why).
  - **Both backends:** `(instance? (Class/forName "[D") data)` looks the class up on every from-host call (jvm.clj:794, mlx.clj:492). Clojure 1.12's `double/1` names it directly.
  - **gen.backend.mlx:**
    - set-budget! reads the atom, then swaps it; `swap-vals!` does both at once (mlx.clj:1642-1644).
    - `(code d)` stands alone as a statement, called only for its throw (mlx.clj:489, 500, 1483). jvm.clj names the same check check-dtype.
    - `(:device (config b))` appears beside the device function that returns the same thing (mlx.clj:625, 1293).
    - `(int 4)` is a magic code for uint64 (mlx.clj:1254) that is missing from dtype-codes.
    - native-value-and-grad's else branch is indented as if it were a sibling of its `if` (mlx.clj:1335-1352).
    - with-operands expands each operand form up to three times: in operand, in mlx-error's arguments and in fence (mlx.clj:397-404). Every call site passes a symbol today, but the macro should bind them once.
    - from-values turns a double[] into a vector of boxed Doubles and maps host-cast over it before copying (mlx.clj:494, 451-454). `MemorySegment/copy` from the array avoids that.
  - **JVM kernels:**
    - r-max, r-min, cummax and cummin test for NaN before calling Math/max and Math/min, which already propagate it (jvm.clj:304-305, 394-395).
    - scan-op dispatches `case op` on every element, against the one-loop-per-op rule the other kernels follow (jvm.clj:384-397).
- **Known?** New.

### I1-13: Docstrings, docs and one name

- **Severity:** minor
- **Where / what:**
  - **Missing docstrings.** Nearly every other public function has one, but these do not:
    - in array.clj: ndim (125), size (133), zeros (563), ones (567), zeros-like and ones-like (589-590);
    - in batch.clj: batched? (78) and batch-backend? (302);
    - in tape.clj: tracked? (37);
    - in dtype.clj: host-double (159), host-long (165) and reduce-result-dtype (127).
  - **A misleading section header.** "Comparison helpers (for tests and assertions)" heads all-true? and any-true?, which are for validation (array.clj:1101-1115).
  - **MlxRuntime's class doc** names loading, errors, scratch and closures. It leaves out its largest part, array lifetime and collection, and also compiled functions (MlxRuntime.java:23-25).
  - **docs/array-api.md:**
    - §6 lists `tile`, which does not exist, and `var`, which is named `variance`.
    - §3 promises `array/to-backend` "with M2"; it was not built.
  - **gen.array.memory's name.** The namespace holds reservations, and every require aliases it `reservations` (mlx.clj:62, llama.clj:58 and two tests), next to gen.backend.mlx's own `memory` function for MLX's counters. `gen.array.reservations` would say what it is.
- **Known?** New.

### I1-14: Comments that narrate the code's history

- **Severity:** minor
- **Where:**
  - MlxRuntime.java:169-173 ("as large ones once were"), 443-445 ("an independent review, 2026-10-19") and 454-459 ("It did, and the cap was no limit on Metal ... 176 errors").
  - mlx.clj:17-20, 555-558 ("Copying every array cost a device round trip per read"), 903-905, and 1107-1112 ("SIGILL at exit").
- **What:** The reasons are worth keeping: why the cap is read lazily, and why views are copied. The accounts of incidents, though, describe code that no longer exists, which comments should not do (idiom.md). They belong in DESIGN §9, the beans and the commit messages, which already hold them.
- **Direction:** keep the constraint and drop the story. For example: "Read at the first collection: the initializer cannot call mlx-c, since the first call initializes MlxRuntime from inside MlxRaw's holder class."
- **Known?** New.

## Decisions worth revisiting

- **§3.4, "Mutation is confined to three places."** This no longer describes L0. Mutation also lives in:
  - the lazy-backends? flag in gen.array.protocols (protocols.clj:24-27);
  - the batch backends' interning cache (batch.clj:292);
  - the registry in gen.array.memory (memory.clj:16);
  - compile's trace cache (array.clj:1070);
  - MLX's constant cache, budget limits and CPU-compile probe (mlx.clj:186, 323, 1405).

  Each is a cache or a registry that never changes what an operation means, which is the principle's real point. Restate §3.4 as that rule ("semantics never reads mutable state; the caches and registries are listed") instead of a count that has quietly become wrong.

## Checked and sound

- **Warnings.** Ran `OUT/scripts/I1-warnings.clj`:
  - No JVM-loadable file in the layer has a reflection warning.
  - gen.array.dtype sets no flags of its own and is clean under `:warn-on-boxed` too.
  - The kernel files (jvm, shape, special, threefry) set both flags and are warning-free.
  - With the flag forced on, boxed warnings appear only in shape arithmetic in the non-kernel files.
- **JVM kernels** follow CLAUDE.md's notes (I1-12's scan is the exception):
  - one small function per op loop (defunary, defbinary's four variants, defreduce);
  - fold-range in statement position for the linear algebra;
  - at most 4 primitive parameters;
  - 0-d results as host values (make and box).
- **Host math.** Two Doubles take a fast path (8 ns for a host exp, measured). Host math and the array kernels define each op separately, and the properties in array_props_test.clj:147 and :157 check that they agree.
- **Dtypes.** Promotion happens in one place (gen.array.dtype), with the op table compiled to a case. floor-mod, float->int and host-cast follow DESIGN §9's Java-cast and boolean rules (read).
- **The tape** (read):
  - its tracked values are immutable, and there is no global tape;
  - the topological sort is iterative, and cotangents accumulate in an identity map;
  - ops on host primals run on the host;
  - its linalg VJPs are shared with MLX's native mode.
- **The batch tracer** (read): equals, hash and count raise the batched-branch error, with the batch on either side; lift aligns the per-particle axes; gather is lazy.
- **Threefry** is unrolled by a macro on primitive longs, with no allocation per word. The MLX version follows the same rounds, injections and mantissa constructions (read, as the M2 review did).
- **Earlier review fixes are present:**
  - compile follows the cached skeleton's key order (leaves-in, array.clj:1045-1052);
  - the scratch buffer grows instead of allocating per call (MlxRuntime.java:166-182);
  - native linalg VJPs run with the backend as the default (custom-vjp's on-b);
  - the collection threshold backs off above the cap (nextArrayThreshold);
  - handles passed to MLX are fenced (with-operands, retained);
  - compile's errors are ex-info.
- **MlxRuntime** (read): upcalls catch everything; the error handler is installed in the static initializer; array lifetime uses phantom references in an intrusive list; Compiled erases its id from every thread's cache.
- **Known and open, not reported again:** genclj-l7xf (tracked host primals are strongly typed) stands as the M2 review described it (array.clj:251-255 with tape.clj's -dtype).

## Coverage

Every line was read, in the snapshot worktree:

| File | Lines read |
|---|---|
| `src/gen/array.clj` | 1-1133 |
| `src/gen/array/batch.clj` | 1-344 |
| `src/gen/array/dtype.clj` | 1-202 |
| `src/gen/array/linalg.clj` | 1-81 |
| `src/gen/array/memory.clj` | 1-50 |
| `src/gen/array/protocols.clj` | 1-139 |
| `src/gen/array/shape.clj` | 1-162 |
| `src/gen/array/special.clj` | 1-113 |
| `src/gen/array/tape.clj` | 1-488 |
| `src/gen/backend/jvm.clj` | 1-905 |
| `src/gen/backend/mlx.clj` | 1-1651 |
| `src/gen/backend/mlx/exact.clj` | 1-162 |
| `src/gen/random.clj` | 1-183 |
| `src/gen/random/threefry.clj` | 1-124 |
| `src-java/gen/mlx/MlxRuntime.java` | 1-653 |
| `src-java/gen/mlx/MlxException.java` | 1-18 |

Context read:

- CLAUDE.md.
- DESIGN.md §1-§8 and §9: all of §9 skimmed, and its L0 entries read in full.
- docs/array-api.md, all of it, and docs/porting-genmlx.md.
- The earlier reviews: review-2026-10-18, review-2026-10-19 and review-m2-2026-10-19.
- The device functions in the generated Mlx.java, and mlx-c's device.cpp, private/device.h and stream.cpp.
- Beans searched: genclj-mkgo, genclj-4n51, genclj-0oax, genclj-1ob9 and its children, genclj-a7s7, genclj-h6uu and genclj-j55z.

Scripts, all under `OUT/scripts/` and run on the JVM backend, one JVM at a time:

- `I1-01.clj`: lgamma's host cost.
- `I1-05.clj`: linalg dtypes.
- `I1-09.clj`: the diagonals.
- `I1-warnings.clj`: reflection and boxed-math warnings.
