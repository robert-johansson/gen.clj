# GenMLX → gen.clj: the porting surface

*Research notes, 2026-10-18. A read-only inventory of `~/code/genmlx`. `src/`
means `~/code/genmlx/src/genmlx/` and `rs/` means
`~/code/genmlx/mlx-node/crates/`. Size: 142 source files, ~61.7k LOC; tests are
501 files, ~118k LOC. The membrane `src/mlx.cljs` is 1759 LOC with 237 public
defs.*

## 1. The membrane (`mlx.cljs`, `mlx/random.cljs`, `mlx/constants.cljs`)

The native module is loaded once, via `(js/require "@genmlx/core")` at
mlx.cljs:31 and random.cljs:15. Most ops are bare references to NAPI exports
(`(def exp (.-exp c))`). Rust's `Either<&MxArray,f64>` coerces JS numbers to
arrays (rs/genmlx-core/src/genmlx.rs:17-24). **The JVM membrane must reproduce
that number→array coercion.**

| Category | Exports (mlx.cljs lines) |
|---|---|
| **dtype** (139-147) | float32; float64→**float32 alias**; int32; int64→**int32 alias**; bool-dt→int32; float16, bfloat16, uint32, uint8. `dtype` returns a numeric code, and `ensure-int-indices` relies on `(= dtypeOf 1)` (571). |
| **creation** (674-762) | scalar (int path via `scalarInt`); array (nested clj→`Float32Array`/`Int32Array`/`Uint32Array`, plus a `BigInt64Array` shape, 684-729); astype, zeros, ones, full, eye, arange, linspace, meshgrid (returns `#js`). All are wrapped in `with-alloc-retry` (catch a Metal OOM, run GC, retry once; 645-672). |
| **unary elementwise** (208-254) | exp expm1 log log2 log10 log1p sin cos tan arccos arcsin arctan sinh cosh tanh sigmoid erf erfinv lgamma digamma bessel-i0e/i1e floor ceil round negative square sqrt abs sign reciprocal flatten isnan isinf isfinite logical-and/or/not log-softmax cumprod roll |
| **binary** (258-267, 379-433) | logaddexp divide power maximum minimum floor-divide remainder matmul inner outer; variadic add/subtract/multiply reduce over the binary native ops. Comparisons equal not-equal greater(-equal) less(-equal) where, and the float-mask helpers eq? neq? gt? lt? and* or*. |
| **reductions/scan/sort** (437-522) | sum prod mean variance std amax amin all any logsumexp argmax argmin argsort sort-arr searchsorted topk cumsum logcumsumexp |
| **shape** (526-560) | reshape squeeze expand-dims transpose tile pad repeat-arr stack concatenate split-arr. `broadcast-to` is **re-implemented as zeros+add** to work around a native bug (534-541). |
| **indexing/gather/scatter** (564-634, 343-351) | take-idx (coerces indices to int32), idx, index, take-along-axis, put-along-axis, scatter-add, bincount (via scatter-add), slice (step via take), slice-nd, mat-get |
| **NN/fused** (766-810, 269-376) | softmax clip nan-to-num stop-gradient rms-norm rope scaled-dot-product-attention silu conv2d gather-qmm dequantize gated-delta-scan gated-delta-step fused-gdn-gating vlm-preprocess load-safetensors |
| **linalg** (1542-1649) | diag trace-mat einsum cholesky solve solve-triangular inv tri-inv cholesky-inv qr svd eigh eigvalsh norm spd-logdet spd-det logabsdet (QR-based). slogdet, det and logdet use a **host-side LU** written in pure CLJS (1604-1649). |
| **random** (random.cljs) | fresh-key (seeded from `js/Math.random`, 28-30), valid-key? (shape [2], non-f32), split, split-n, ensure-key, split-or-nils, split-n-or-nils; normal, uniform, bernoulli, categorical, randint, gumbel, laplace, truncated-normal, multivariate-normal (119-171). All keys are explicit; there is no global RNG. |
| **constants** | 13 cached scalars plus LOG-2PI (constants.cljs:15-29). |
| **queries** (819-828) | shape, shape-raw (a raw JS array), ndim, dtype, size, array? (`instance? MxArray`) |
| **eval/memory** (1042-1538) | item (fused eval and read to f64); ->clj (eval, then `.toFloat32/.toInt32/.toUint32`, then unflatten); eval! (`evalArrays`); materialize!, realize, realize-clj, synchronize!; async-eval! (`Promise.all evalAsync`, **unused in src**). tidy and the tidy-run/-scalar/-materialize(-light) family; force-gc!, sweep-dead-arrays!, jsc-cleanup! (Bun `bun:jsc` GC, 1147-1161), auto-cleanup!, gfi-cleanup!, with-resource-guard(-gc). Memory getters and setters, Metal buffer-count hysteresis (1340-1395), cost counters (1038-1053). |
| **device** (1713-1758) | cpu/gpu strings, default-device, set-default-device!, with-default-device*. There is **no stream API**. Linalg is pinned to the CPU stream inside the C++ shim (rs/mlx-sys/src/mlx_linalg.cpp:31-32). |

**How the transforms are implemented:**
- **grad / value-and-grad** (976-1019): native MLX autograd.
  - `computeGradients` and `valueAndGrad` (genmlx.rs:1093-1140) call
    `mlx_core::autograd` with a closure that synchronously re-enters the JS
    function via `napi_call_function`.
  - They always differentiate **all** arguments; CLJS picks the `argnums`
    afterwards.
  - A nil argument throws (947-969). A `grad-depth` atom backs `in-grad?`.
- **vmap** (937-945): native `mlx_vmap_apply` with a JS callback
  (rs/genmlx-core/src/transforms.rs:112). It returns only the first output.
  - It is used only in `inference/vi.cljs`, and it crashes on GFI score
    functions that use `volatile!` (vi.cljs:184, compiled_optimizer.cljs:620).
  - GenMLX's own "vmap" (`vmap.cljs`, `vectorized.cljs`, `dyn/vsimulate`) is
    **shape-based batching through the handler**, not this transform.
- **compile-fn** (838-849): **permanently the identity**. All 63 call sites are
  trivially portable.
  - Real compilation is opt-in through persistent handles: compile-create,
    compiled-call(1), compiled-call-captured (CUDA-graph replay),
    compiled-free!, with-compiled (867-935 → transforms.rs:396-576,
    `mlx_compile_apply`).
  - Multi-output builders must return a JS array, so `to-array` and `aget`
    appear in dynamic.cljs:1571-1616, mcmc.cljs, compiled_optimizer.cljs:106-166
    and compiled.cljs:914-921.
- **custom-vjp**: there is no native binding. `custom_gradient.cljs:14-41` uses
  the stop-gradient surrogate trick `sg(f)+s-sg(s)`, which is only correct for
  scalar outputs.
- **vjp / jvp / checkpoint**: not exposed; zero uses.
- **tidy**: not MLX's tidy. It is a depth counter plus a JSC GC plus `clearCache`
  (1167-1175).

## 2. Leakage outside the membrane

Only four files touch `@genmlx/core` directly:
- `mlx/random.cljs`;
- `llm/backend.cljs:40` (model classes and tokenizer);
- `llm/vision.cljs:38` (VLM classes);
- `world/train.cljs:69` (GrpoTrainingEngine).

`mx/core` is used raw for `.fromFloat32` and `.fromInt32` in
llm/grammar.cljs:610 and 745-746, and in llm/toolcall.cljs:356. Methods are
called directly on MxArray only 7 times: `.astype` in exact.cljs:218,
pomdp.cljs:173-174, vectorized.cljs:33 and belief.cljs:133, and `.toFloat32` in
llm/bytes.cljs:138 and 191.

Counts outside mlx.cljs (comments stripped):
- **`js/Math.*`, `js/Number.isFinite`, `js/isNaN`**: ~370 occurrences, a trivial
  swap to `Math/` or `Double/`. Heaviest files: gfi.cljs 49, dist.cljs 22,
  mcmc.cljs 21, pef.cljs 21, program.cljs 17, rb_mixture.cljs 13.
- **`.-` property access**:
  - 35× `.-trace rt` and 4× `.-splice`, on the `#js` runtime object
    (runtime.cljs:180 `rt #js {:trace :splice :param}`; gen.cljc:24-26; the
    gfi.cljs laws; pef.cljs).
  - 35× `.-message` on errors.
  - The rest: JSON config fields in the qwen forwards (`.-hidden_size` and so
    on), JS objects in pi_provider and pi_session, and `js/process.env`
    (mcmc.cljs:276 and 292; compiled_ops.cljs:1670; qwen35_forward.cljs:115 and
    610).
- **async (promesa)**: 17 files, all in llm/ and world/, plus agents/remote,
  program and sandbox_child (see §5).
- **typed arrays**: llm/grammar.cljs (`Float32Array`/`Int32Array` masks,
  499-746), toolcall.cljs:350-373, bytes.cljs:61, and `Uint32Array.from` for
  tokenizer decode in core, smc, backend and pi_*.
- **fs**:
  - llm forwards: llm/forward, qwen3_forward, qwen35_forward,
    qwen35_vision_forward.
  - Other llm files: backend, vision, pi_edit, pi_session.
  - Elsewhere: serialize.cljs:16, program.cljs:1030, sandbox_child,
    session_grpo, distill_sandbox, llm_proposer.
- **child_process**: sandbox.cljs:96, distill_sandbox.cljs:43, llm_proposer.cljs:416.
- **Bun-specific**: `bun:sqlite` and `node:crypto` in memory.cljs:71-76;
  Bun.serve, fetch and Response in world/net.cljs; `Bun.nanoseconds` in
  inference/cost.cljs:59; `bun:jsc` in mlx.cljs.
- **CLJS-specific syntax**:
  - `(:require-macros …)` in 16 files.
  - `catch :default` 79×.
  - `^:mutable` with `set!` only in mlx.cljs (22), pi_provider, pi_session and
    train.
  - `js/WeakMap` in qwen3_forward.cljs:314.
  - `cljs.core/PersistentQueue.EMPTY` in dep_graph.cljs:122.

**Pure-Clojure namespaces with zero JS interop** (only `mx/`, `rng/` and core
Clojure):
- Data algebra and GFI: choicemap, selection, trace, protocols, diff, dispatch,
  inspect, handler, edit.
- Combinators: combinators (3595 LOC, 225 `mx/`, 0 `js/`), vmap.
- Schema and static analysis: schema, schemas, rewrite, conjugacy, affine,
  linear_gaussian, method_selection, dep_graph.
- Gradients and support code: custom_gradient, gradients, tensor_trace,
  tabular, verify, dist/core, inference.cljs.
- Most of inference/*: smcp3, kernel, kalman, ekf, ekf_nd, conjugate,
  analytical, steppable, adev, differentiable*.
- agents/* (except remote) and control/* (except the world.proc dependency).
- llm/branched, llm/structured, llm/schema_grammar, llm/sampling.

Many more are pure apart from `js/Math`.

## 3. What `@genmlx/core` adds beyond stock MLX

`genmlx-core` (rs/genmlx-core/src, ~2.2k lines of Rust) is a cdylib. It
re-exports **all of stock mlx-node `mlx-core`** and adds the GenMLX free
functions. The live surface is 239 function exports per
docs/membrane-coverage.md (185 wrapped, 54 omitted). It sits on **mlx-sys, a
custom C++ shim, not mlx-c.**

**Generic numeric additions (needed by the PPL core):**
- Composites that MLX 0.32 lacks, built in C++ from primitives:
  - lgamma and digamma (rs/mlx-sys/src/mlx_genmlx_gamma.cpp:245, 255), using a
    "double-where" guard for finite gradients;
  - bessel_i0e and i1e (mlx_genmlx_bessel.cpp:125, 133);
  - searchsorted, with a broadcast path and a binary-search path
    (mlx_genmlx_searchsorted.cpp:90).

  In a JVM port these are easy to re-express as Clojure composites over mx ops.
- 14 stock ops that mlx-node had not shimmed (topk, logaddexp, logcumsumexp,
  einsum, …; mlx_genmlx_ext.cpp:14-24).
- Catchable native errors: a thread-local last-error slot instead of abort
  (mlx_genmlx_ext.cpp:70-79).
- Keyed PRNG free functions (genmlx.rs:778-960), all stock `mlx::random` with keys.
- Linalg forced onto the CPU stream.
- Autograd, vmap and compile with JS callbacks; persistent compile handles with
  CUDA-graph capture (transforms.rs).
- Memory and Metal introspection (memory_napi.rs).

**LLM- and model-specific additions:**
- In genmlx-core:
  - Custom Metal kernels via `fast::metal_kernel`: gated-delta step and scan,
    and fused GDN gating (mlx_gated_delta.cpp).
  - gather_qmm and dequantize (stock ops, newly exported).
  - `qwen35VlmPreprocess` (image decode and patchify; genmlx.rs:1472).
- Inherited from upstream mlx-core:
  - Native model classes: Qwen3Model, Qwen35Model, Qwen35MoeModel, Gemma4Model,
    HarrierModel, Lfm2Model, and OCR/ASR models (rs/mlx-core/src/models/*).
  - Their surface: `.load/.forward/.forwardWithCache/.initCaches/.resetCaches`;
    branch operations `.branchCache/.branchFrom/.forwardBranch/.disposeBranch`;
    `.vlmPrefillFlat`, `.chatSessionStart`.
  - Qwen3Tokenizer: HF `tokenizers` plus minijinja. Encode and decode are
    **async promises** (tokenizer.rs:215, 690, 786).
  - Engine and sessions, paged attention (mlx-paged-attn Metal), MoE dispatch,
    fp8 quantization, nn and optimizers.
  - GRPO and SFT training engines (`GrpoTrainingEngine`, used only by
    world/train.cljs).

## 4. LLM layer

**GenMLX-owned forward passes, in CLJS over mx ops.** These are the **default**
path for qwen3, qwen3_5 and qwen3_5_moe (forward.cljs:18-23, 41-52).
- `llm/qwen3_forward.cljs` (486 LOC).
- `llm/qwen35_forward.cljs` (921 LOC): hybrid GatedDeltaNet plus sparse MoE via
  `gather-qmm`.
- `llm/qwen35_vision_forward.cljs` (362 LOC): vision tower plus M-RoPE prefill,
  fed by the native `vlm-preprocess`.

What they need:
- Loading: load-safetensors (sharded, via index.json) and dequantize (affine
  2-, 4- and 8-bit).
- Fast ops: rms-norm, rope, sdpa, silu, gather-qmm, gated-delta-scan/step,
  fused-gdn-gating (opt-in).
- Indexing: slice-nd, put-along-axis, take-along-axis, take-idx.
- Sorting and selection: topk, argsort, softmax.
- Basics: matmul, concatenate, broadcast-to.

`gdn-recur-steps` (qwen35_forward.cljs:386) is a pure-mx reference for the GDN
kernels, so the custom kernels are optional for a first port. Config is read
with `fs` plus `JSON.parse`.

**Native classes are used only for:**
- families without an owned forward: qwen3_next MoE, gemma4, lfm2, harrier
  (backend.cljs:60-94);
- `generate-text` via `.chatSessionStart` (backend.cljs:977-1017);
- `llm/vision.cljs` (native VLM chat);
- the tokenizer (always native);
- GRPO training.

**The owned backend contract** (forward.cljs). Caches are **persistent vectors of
per-layer maps of immutable arrays**. Six core functions, plus batching (101-215):
- `load-model` and `init-cache`.
- `prefill ids → [last-logits cache]`, and `prefill-chunked`.
- `step cache offset id → [logits cache']`.
- `forward-cached` (multi-token continuation), `next-token-logits` (uncached)
  and `forward`.
- Batching: `step-batched` ([K] lanes), `prefill-batched`, `broadcast-cache`
  (B=1→K), `resample-cache-lanes` (one take-idx per array) and
  `materialize-cache!`.

`backend.cljs` wraps this in `CljsForwardModel [fwd cache-atom branches-atom]`
(120). A branch fork is just another reference (598-899).

**Minimal backend needs:**
- a cached forward that returns last-position logits as a tensor;
- immutable cache values, which make fork/branch free;
- [K]-lane batching, with broadcast and gather on the cache;
- a tokenizer: encode, decode, vocab-size, eos/pad ids, id↔token, and
  applyChatTemplate;
- `dist/categorical` over logits;
- `item` and `materialize!`.

**`llm/core.cljs` API:**
- **`make-llm-gf`** (60-127): a DynamicGF over `[prompt-ids max-tokens]` with
  sites `:t0..:tN` ~ categorical(logits).
  - It runs `init-cache!`, `forward-prefill` and `forward-step` inside
    try/finally, and uses `mx/item` for the EOS check.
  - Options: `:images`, `:prefill-chunk`, `:sweep-every`.
- **`with-slab-assess`** (129-196): assess in one teacher-forcing forward pass.
- **`make-llm-gf-uncached`** (198-220).
- **`make-llm-gf-batched`** (222-372): vectorization-safe, with a masked-EOS
  algebra (39-58), a `:hook` for grammar, and support for `dyn/vsimulate`.
- **Helpers:** vtrace-lane-tokens, decode-vtrace (promise),
  generate-texts-batched (promise), decode-trace (promise).

**`llm/branched.cljs`:**
- `make-llm-gf-branched` (218-256) attaches a `::custom-dispatch`.
  - `simulate` forks a branch at each token site into a ledger in the trace's
    metadata.
  - `regenerate` resumes in O(1) via `branch-from`, accumulates the
    retained-only weight, and can optionally apply grammar DFA masking.
- `with-llm-branches*` (93) is the disposal scope, using a dynamic
  `*branch-scope*`.
- `llm-mh-chain` (262) runs token MCMC and disposes of the losers.

**`llm/smc.cljs`:**
- An `ITokenDecoder` protocol (76-80) with three decoders: Native/branch (83),
  Replay (110) and Table (139).
- `token-smc opts model-map prompt-ids` (551) runs twisted/grammar SMC.
  - ESS-gated systematic resampling; `:lanes?` batched mode; rejuvenation at
    the end.
  - Returns `{:particles :log-ml-estimate :ess-trajectory}`.
- `with-token-smc*` (615) and `live-handles` (632).
- Everything is synchronous except `decode-particles!` (494).

## 5. Async

The **core GFI and all inference are synchronous**. There is no promesa in
protocols, dynamic, handler, combinators or inference/*, and `mx/async-eval!` has
zero call sites. Promises appear only at I/O boundaries:
- LLM backend: `load-model` (backend.cljs:246, because
  `Qwen3Tokenizer.fromPretrained` is async), `encode`/`decode` (302-312),
  `generate-text*` (`chatSessionStart`).
- LLM helpers: core.cljs:400-458 (decode and generate) and smc.cljs:494.
- Other LLM files: codegen, msa, structured, pi_*, vision.
- world/net (Bun.serve; promesa promises are not native Promises,
  net.cljs:48-52) and world/train (`trainStepAuto` awaits a native `Promise`,
  train.cljs:36-40).
- agents/remote, program, sandbox_child, session_reward, distill_sandbox.

The decode loops themselves are synchronous. A JVM tokenizer (for example DJL's
HF tokenizers) is synchronous, which removes most of these promises.

## 6. cljc, macros and SCI specifics
- **`gen`** (gen.cljc:16-29) and **`defdist`** (dist/macros.cljc:14-88) are defined
  **only inside `#?(:org.babashka/nbb …)` reader conditionals**, so they need a
  `:clj` branch.
- **`gen` expansion:** `(genmlx.dynamic/make-gen-fn (fn [ᐩrt ~@params]
  (let [trace (.-trace ᐩrt) …] ~@body)) '~(list* params body))`. The **quoted
  source** drives schema extraction and L1-L4 compilation (dynamic.cljs:1338+,
  schema.cljs). The analysis is symbolic and name-based.
- **`defdist`** emits a constructor that applies `ensure-array` to each
  parameter, plus `defmethod`s on `dist-sample*`, `dist-log-prob`,
  `dist-reparam` and `dist-support`, keyed by keyword. There are 43 defdists:
  26 in dist.cljs, the rest in kalman, ekf, ekf_nd, conjugate, hmm_forward and
  agents/helpers.
- **Compilation builds closures; it does not eval.** `compiled.cljs:503-534
  compile-expr` maps source forms to closures through an op table (305-340), but
  it **calls `(resolve form)` at runtime** (530).
  - That works under SCI. On the JVM it resolves against the current `*ns*`, so
    the macro should capture the defining namespace or resolve at expansion time.
  - `edit.cljs:124` also uses `resolve`.
- **Runtime eval (SCI):** `codegen/eval.cljs:67` (`sci/eval-string`),
  `program.cljs:256`, `pef.cljs:402, 429` and `llm/msa_score.cljs:73`;
  `nbb.core/load-string` in sandbox_child.cljs:68; `load-file` in session_reward.
  SCI and edamame are cljc and run on the JVM, or plain `eval` can replace them.

## 7. Dependencies
- **Vendored source dirs.** nbb.edn adds `malli/src`, `instaparse/src` and
  `test.check` to the classpath. All are cljc and run natively on the JVM. On
  the JVM, use the official releases; the forks exist only for nbb.
  - malli: schemas, dev, schema_grammar.
  - instaparse: llm/grammar, msa.
  - test.check: pef (in src) and the tests.
- **Bundled with nbb:** promesa (17 files), sci (4), edamame (2), cljs.reader (5).
  All are JVM-compatible: promesa runs on CompletableFuture, and cljs.reader
  becomes clojure.edn. `nbb.core`'s `load-string` and `load-file` need JVM
  equivalents.
- No goog.* or npm CLJS libraries are used.
- **Runtime JS dependencies:** `@genmlx/core`, `bun:jsc`, `bun:sqlite`,
  `node:crypto`, `fs`, `path`, `os`, `child_process`, Bun.serve/fetch.

## Porting surface summary

**(a) Port nearly verbatim.** Swap `js/Math`, `catch :default` and
`:require-macros`. Turn the `#js` runtime object into a deftype (`(.-trace rt)`
works on JVM deftypes). `aget` on compiled outputs becomes vector or array
access.
- Data algebra and GFI: protocols, choicemap, selection, trace, diff, dispatch,
  inspect, handler, runtime, dynamic, edit, gfi.
- Combinators and batching: combinators, vmap, vectorized.
- Static analysis: schema, schemas, rewrite, conjugacy, affine, linear_gaussian,
  method_selection, dep_graph.
- Compiled paths: compiled, compiled_gen, compiled_ops (fix `resolve` and
  process.env).
- Distributions and macros: dist, dist/core, dist/macros, gen.cljc (add a `:clj`
  branch), mlx/constants.
- Gradients and learning: custom_gradient, gradients, learning, nn, fit.
- Other support: encapsulated, tensor_trace, tabular, verify, sensorimotor.
- Libraries: all 30 of inference/*, agents/* (minus remote), control/*.
- LLM: llm/branched, smc, core, structured, schema_grammar, sampling.

**(b) Moderate JS→JVM rewrite:**
- `mlx.cljs` and `mlx/random.cljs`: a full re-binding to mlx-c via Panama,
  structurally 1:1. The GC-heuristics block (1124-1538) needs rethinking around
  Cleaner and Arena.
- LLM forward passes: llm/forward, qwen3_forward, qwen35_forward,
  qwen35_vision_forward (fs, JSON, WeakMap).
- llm/backend: the owned path only.
- llm/grammar, toolcall, bytes (typed arrays).
- serialize (fs) and memory (sqlite via JDBC, crypto).
- codegen/eval, program, pef, msa_score, msa and codegen (SCI → eval or SCI on
  the JVM).
- sandbox and sandbox_child (ProcessBuilder plus a JVM or bb child),
  inference/cost (`System/nanoTime`), world/proc.
- The pure parts of world/*: synth, search, curriculum, distill*, sft, harvest,
  probe, repl_corpus, t1_battery, train_reward.

**(c) Deeply tied to node or mlx-node:**
- Native model classes and `.chatSessionStart` (the backend's upstream path,
  generate-text, llm/vision).
- Qwen3Tokenizer (needs a JVM tokenizer plus a Jinja chat-template engine).
- world/train and session_grpo (the native GRPO engine).
- world/net (Bun.serve) and agents/remote.
- llm/pi_provider, pi_session, pi_edit, pi_assess (whose final form is a `#js`
  engine object for a TypeScript host).
- world/distill_sandbox and llm_proposer (child_process plus nbb).
- Bun JSC GC and the Metal buffer-count wall logic.
- Custom Metal GDN kernels: portable via `mlx_fast_metal_kernel` with the .metal
  sources, or replaced by the pure-mx reference.
- `vlm-preprocess` (image decode).

**(d) Minimal backend op set: `mx/` call counts across src/, excluding mlx.cljs.**
There are ~5.6k calls in total; comments are included, so the counts are rough.
- **Top of the list:** scalar 605, add 597, multiply 506, subtract 373,
  materialize! 189, item 179, shape 155, reshape 147, where 142, divide 135,
  array 130, log 127, index 108, sum 102, zeros 90, realize 90, stack 87,
  ->clj 85, astype 82.
- **Next:** take-idx 66, negative 66, array? 66, compile-fn 63 (identity),
  logsumexp 49, grad 48, exp 48, equal 46, ensure-array 41, square 37, eval! 37,
  transpose 35, clear-cache! 33, concatenate 30, value-and-grad 27,
  sweep-dead-arrays! 25, expand-dims 25.
- **Then:** matmul 24, broadcast-to 23, less 22, maximum 21, arange 21, dtype 19,
  vmap 18, greater 18, greater-equal 18, squeeze 17, lgamma 17, slice 16,
  stop-gradient 14, sqrt 14, argmax 14.
- **Long tail (1–13 uses each):**
  - math: abs, power, cos, sin, tan, arccos, tanh, sign, floor, ceil, expm1,
    log1p, erf, erfinv, bessel-i0e, reciprocal, remainder, floor-divide.
  - reductions, comparisons and logic: mean, amax, amin, all, any, cumsum,
    variance, std, logaddexp, log-softmax, softmax, less-equal, minimum,
    logical-and/or/not, isnan.
  - creation and shape: ones, eye, full, ndim, size, flatten, tile, split-arr,
    repeat-arr, clip.
  - indexing and sorting: take-along-axis, put-along-axis, slice-nd, argsort,
    searchsorted, topk.
  - linalg: diag, solve, inv, cholesky, spd-logdet, logabsdet, tri-inv, outer,
    inner.
  - compilation and devices: compile-create, compiled-call-captured,
    compiled-free!, with-default-device*.
  - NN and LLM: rms-norm, silu, sigmoid, sdpa, gather-qmm, load-safetensors,
    rope, gated-delta-scan/step, fused-gdn-gating, dequantize.
- **RNG:** ensure-key 153, split 150, uniform 93, fresh-key 89, normal 80,
  split-n 70, categorical 14, truncated-normal 2, randint 2, gumbel 1,
  bernoulli 1.
- **Dtypes:** float32, int32, uint32, bfloat16/float16.
- **Unused:** ~75 membrane exports have no call sites in src/ (tests may use
  them). These include digamma, bessel-i1e, conv2d, einsum, qr, svd, eigh,
  eigvalsh, norm, det, slogdet, logdet, cumprod, roll, pad, scatter-add, bincount,
  sort-arr, prod, argmin, linspace, meshgrid, async-eval! and most memory getters.

**Protocol-design notes:**
- Number→array auto-coercion must live in the backend.
- grad differentiates all arguments.
- Persistent compiled handles return multiple outputs.
- Linalg must run on the CPU stream.
- MLX has no f64, i64 or bool on the GPU (they are aliased).
- The numeric surface does not need vjp, jvp or custom_vjp.
