# GenMLX quality triage: what to port, and when

*Research notes, 2026-10-18. A read-only analysis of `~/code/genmlx` (commit
d8149c7). Sources: the authors' docs (CLAUDE.md, ARCHITECTURE.md, ROADMAP.md,
docs/api-freeze.md, test/TESTING.md); all 992 GenMLX beans; git churn and
fix-commit counts (1375 commits, 2026-02-18 → 2026-08-09); the `ns :require`
graph with transitive closures; and reading the code. Bean IDs like
`genmlx-xxxx` refer to GenMLX's private beans repo.*

**Headline.** The mathematical core is clean and small, about 3–4k LOC. It is
embedded in roughly 25k LOC of optimization paths that re-implement the same
semantics, plus roughly 18k LOC of research verticals. GenMLX's 2026-08-01 audit
(genmlx-pif1) checked about 40 derivations and found **zero formula errors**:
"The mathematics is not the risk surface; the duplication is."

Of 220 open GenMLX beans, only about 5 touch the core GFI. LLM, world, agents
and native infrastructure account for about 85%. The core is stable *now*, but
it took around 150 bug fixes to get there, and those fixes are the regression
oracles we inherit.

**Recommendations used in the tables:**
- **P1**: port now (core).
- **P2**: port soon after the core.
- **P3**: port later, as a cleaned-up redesign.
- **Defer**: maybe never (research).

## Triage tables

Quality grades run from A (clean) to C (messy).

### Core GFI

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| protocols.cljs | 84 | 11 single-op GFI protocols | A | stable | 0 (nxt1 deferred) | GFI | **P1**. Merge IUpdate, IUpdateWithDiffs and IUpdateWithArgs into Gen.jl's `update(tr,args,argdiffs,cm)`; drop IBatchedSplice. |
| choicemap, selection, diff | 287+92+41 | Value/Node choicemaps, selection algebra, argdiffs | A | stable | 0 open (fixed: ybw9, yey5, a6o5, dp60) | ChoiceMap, Selection, Diff | **P1**. Add retdiffs (GenMLX has none). |
| trace.cljs | 123 | Trace record, score-type tags, omega | B | stable | 0 | Trace | **P1**, minus score-type and omega. |
| handler.cljs | 620 | 7 pure scalar `(state,addr,dist)→[v state']` transitions, 6 batched copies, combinator fallback | A− scalar / C batched | stable | 0 open; 43 commits / 18 fixes | DynamicDSL internals | **P1** for the scalar part (~240 LOC); keep only the general (retained-only) regenerate. |
| runtime.cljs | 181 | `run-handler`: one `volatile!`, trace/splice/param closures | B | stable | 0 | DSL runtime | **P1** for the scalar half; split out batched splice and `mx/auto-cleanup!`. |
| dynamic.cljs | 1896 | DynamicGF, 4-level dispatcher stack, analytical/compiled/prefix run-fns, `make-gen-fn` pipeline, v* batched ops | **C** | evolving | 6q9o; 96 commits / 38 fixes; 80 bean refs; 22 regenerate bug beans | DynamicDSLFunction | **P1 as a ~400-LOC rewrite** (DynamicGF, `execute-sub` and handler dispatch only). |
| dispatch.cljs | 88 | IDispatcher, `with-handler` / `with-dispatch` | A | stable | 0 | (custom GFs) | **P2**. A good extension seam; the stack shrinks to custom + handler. |
| edit.cljs | 235 | Edit requests with backward requests | A | stable | 0 | none (GenJAX/GenSMCP3) | **P2** |
| gen.cljc, dist/macros.cljc | 29+88 | `gen` and `defdist` macros (nbb-only reader conditionals) | A | stable | 0 | @gen, Distribution | **P1** |
| dist/core.cljs | 341 | Single Distribution record as a GF; open multimethods; mixture and product | A− | stable | 0 | Distribution, Mixture/Product | **P1** |
| dist.cljs | 2152 | 36 distributions | B | stable-ish | ioc6; 84 commits / 41 fixes (7oen, 4x5w, rc81/g2iu/smrk/84mq, 2nec, bey6) | ~20 built-ins | **P1** for Gen.jl's set; **P2** for wishart, wrapped, truncated, iid, broadcasted. |

### Combinators and batched execution

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| combinators.cljs | 3595 | Map, Unfold, Switch, Scan, Mask, Mix, Recurse, Contramap, MapRetval; each op has fused, compiled and handler paths | **C** | stable now | nxt1; 58 commits / 29 fixes; 25 bug beans (v740, zek9, uizc, llpt, 5a87, 175y) | Map, Unfold, Switch, Recurse | **P1** for the handler paths of Map, Unfold, Switch and Recurse (about ¼ of the file); **P2** for Scan, Mask, Mix and dimap. |
| vectorized.cljs, v* in dynamic, batched transitions | 221 + ~370 + ~370 | Shape-based [N] batching, VectorizedTrace, IBatchedSplice | B/C | evolving | 6q9o, ioc6, 8egz; 29 bug beans | none (GenJAX vmap) | **P3**. Redesign as one transition set parameterized by a sampler. |
| vmap.cljs | 464 | GenJAX-style vmap | B− | evolving | `rng/fresh-key` at :226/:270 | none | **P3** |

### Compiled paths and static analysis

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| compiled, compiled_ops, compiled_gen, tensor_trace | 1502+2388+56+135 | L1/L2 noise-transform compiler over quoted source; 24 GFI-op builders (static, branch, prefix, replay); fused loops; affine-family tensor scores | **C** | evolving | rsgr, a1uh, 837m + epic z2gt, h5wg, xtfr, lrgl; 39 bug beans | static DSL compile | **Defer**, then P3 as an explicit static IR. |
| schema.cljs | 1048 | Walks quoted, unexpanded source by symbol name | B− | stable-ish | 0 open (q3x2, blkz, 7qdz, dv66) | static IR | **P3 redesign** |
| schemas.cljs | 283 | Malli shapes | B | stable | 0 | — | P2, optional |
| L3 stack: conjugacy, affine, linear_gaussian, rewrite, dep_graph, auto_analytical | 340+455+603+332+259+869 | Static conjugacy, Kalman and linear-Gaussian detection plus auto handlers | **C** | evolving | 3k36, q4v7, bg67, 71wh, akvp, tsvg, 9mos; 30 bug beans, including criticals ke9i, rmy7, 94qc, 5ytq, su6q, yy8u, jmk4, 540f | none | **Defer** (research) |
| method_selection, fit | 256+248 | Automatic method choice; one-call `fit` | B | experimental | 0 | none | **Defer** |
| pef.cljs | 1002 | Path-equivalence fuzzer | B+ | stable | 3k36 | none | Defer until a second execution path exists. |
| program.cljs | 1258 | Structure inference over SCI-compiled programs; requires llm.backend | B− | experimental | sjns, b4z9 history | none | **Defer** |

### Inference

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| inference/importance | 175 | Importance sampling and IS-resampling | B+ | stable | 0 | importance_sampling/resampling | **P1** |
| inference/kernel | 362 | Kernel algebra (chain, repeat, cycle, mix, reversals) | B+ | stable | 0 | @kern, mh kernels | **P2** |
| inference/mcmc | **3865** | MH: custom, involutive and Gibbs (~500 clean LOC); MALA, HMC, NUTS, elliptical slice, MAP; ~15 compiled, fused, vectorized and chunked variants | **C** | evolving | gth5, 837m, okeu; 102 commits / 37 fixes; 90 bean refs | mh, mala, hmc, elliptical_slice, map_optimize | **P1** for the MH family; **P2** for gradient methods via the GFI score; **Defer** all variants. |
| inference/smc | 846 | smc (fixed-args convention), smc-args, csmc, smc-unfold, vsmc | B−/C | evolving | im8n; 29 SMC bug beans (uxjm critical) | particle filter API | **P2**: the Gen.jl API with args-growing semantics only. |
| smcp3, pmcmc | 294+231 | SMCP3; PMMH and particle Gibbs | C / B | evolving | im8n | GenSMCP3.jl, GenParticleFilters | **P3** |
| vi, adev, amortized | 502+323+275 | ADVI over a flat log-density; programmable VI; ADEV; neural proposals | B− | evolving | znxi; 7sqe, 9rwf, vsmi history | black_box_vi!, vimco, train! | **P2** for VI, redesigned to be GFI-native; **P3** for ADEV and amortized. |
| learning, gradients, custom_gradient | 352+85+101 | Parameter store, optimizers, wake-sleep; `choice-gradients` (top-level scalars only); CustomGradientGF | B | stable | 0 | ParamStore, choice_gradients, accumulate_param_gradients! (missing in GenMLX), CustomGradientGF | **P2** (needs the JVM autodiff decision) |
| nn.cljs | 389 | Neural-network layers on MLX autograd | B | stable | 0 | GenFlux | **Defer** |
| exact, enumerate | 745+130 | Two enumeration implementations; exact also carries a memo/agents API | B | stable | 0 | enumerative_inference | **P2**: port one of them. |
| analytical middleware: analytical, conjugate, kalman, ekf, ekf_nd, hmm_forward | 43+408+279+237+391+268 | Ring-style handler middleware for closed forms | B | stable | rmy7 history | none | **P3**, opt-in via `with-handler` |
| rb_mixture | 215 | Rao-Blackwellized mixture scorer | B | experimental | 1 commit | none | **Defer** |
| translator | 416 | Trace translators, reversible-jump MCMC, coarse-to-fine | B+ | stable | 0 (jtou, rirn fixed) | trace translators | **P2** |
| encapsulated | 562 | Thesis §4.5 pseudo-marginal GFs | B | evolving | fq57 | (thesis only) | **P3** |
| diagnostics | 302 | ESS, R-hat | B+ | stable | 0 | — | **P2** |
| compiled_gradient, compiled_optimizer, compiled_smc, differentiable, differentiable_resample, fisher | 193+636+213+108+173+275 | L2/L4 fused gradients, Adam inside the graph, Gumbel resampling, Laplace | C/B | experimental | (z2gt) | none | **Defer** (fisher: P3) |
| cost, steppable | 94+125 | Cost meter, steppable SMC (control substrate) | B | experimental | 6qz4, im8n | none | **Defer** |
| inference/util | 638 | Weight and resampling utilities plus 8 score-function builders; requires compiled-ops | C | stable | 0 (fw2z, ph4g history) | — | **P1**, for the weight and resampling utilities only. |
| inference.cljs | 130 | Aggregator namespace | — | — | — | — | **Do not port** |

### Supporting code and the MLX membrane

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| gfi.cljs | 2523 | 87 executable GFI laws | A− content | stable | fq57 | — | **P1** as a *test* library, split per feature. |
| verify, serialize | 260+427 | Checks for the dynamic modeling language's restrictions; JSON round-trip of choices | B | stable | 0 (st0y, 94tu history) | GenSerialization.jl | **P2** |
| inspect, dev | 84+128 | Dispatch introspection; Malli dev-mode atoms | B | stable | 0 | — | **Defer** |
| memory, sandbox (+child), codegen/eval | 711, 278, 243 | bun:sqlite persistence; SIGKILL-able eval; SCI eval spine | B | experimental | — | — | **Defer** |
| sensorimotor, tabular | 374+310 | NARS-style concept memory; ARC-solver BFS tensors | B | experimental | 69wh | — | **Defer** |
| mlx.cljs, mlx/random, mlx/constants | 1759+171+29 | The MLX membrane, including ~400 LOC of Metal buffer-count heuristics | B | evolving | okeu, a9kf, z9li; 36 bug beans | — | **Do not port as is**: define a substrate protocol. Keep the split-key PRNG semantics (P1). |

### Research verticals

| Area / file(s) | LOC | What it does | Quality | Stability | Open bugs/features and history | Gen.jl equivalent | Recommendation |
|---|---|---|---|---|---|---|---|
| llm/* (22 files) | 9984 | LLM as a generative function, model-specific forward passes, grammars, pi agent | C | experimental | **14 open bugs** (lge5, 9k32, ey5v, k2cd, 5z51, 7fjh, 046s…); 73 open beans; 230 bean refs | GenGPT3.jl-ish | **Defer** |
| agents/* (14 files) | 2986 | agentmodels.org MDP, POMDP and inverse reinforcement learning | B | frozen by tests | ~12 draft remainders | none | **Defer** (separate library) |
| control/* (4 files) | 652 | Metareasoning | B | experimental | gjih, y39l, 6qz4 | none | **Defer** |
| world/* (18 files) | 4795 | GRPO, program synthesis, distillation | C | experimental | pnaw, lkt0, z92i… | none | **Defer** |

## 1. The beautiful core
- **Data algebra.** choicemap, selection, trace, diff, protocols and edit have
  essentially no `mx/` references (trace has 1, edit has 4, resolved lazily).
  They are persistent maps with thin protocols, and api-freeze §1–2 freezes them.
  Port almost verbatim.
- **Pure handler transitions with a single-volatile runtime** (handler.cljs:51-240,
  runtime.cljs). This is the algebraic-effects design:
  `(state, addr, dist) → [value state']`. Each GFI op is an init-state plus a
  transition, and enumeration and analytical inference are just other
  transitions. handler.cljs's dependency closure is only 8 namespaces / 3.5k LOC.
- **Distributions as generative functions** (dist/core.cljs:152-191): one record,
  open multimethods, and `defdist`.
- **`with-handler` / `with-dispatch`** (dispatch.cljs): Ring-middleware
  extensibility without new record types.
- **Inference that is already clean:** kernel.cljs (kernel algebra with declared
  reversals), translator.cljs (Eq 3.12 with an AD Jacobian), importance.cljs,
  and the MH family in mcmc.cljs:49-190 and :1488-1615.
- **The gfi.cljs laws.** An 87-law executable specification citing the thesis.
  Together with `gen_clj_compat_test` (356 assertions adapted from probcomp
  Gen.clj's suite, checked against scipy and Gen.jl), it is the best acceptance
  oracle to carry over.

## 2. The messiest and least stable areas

1. **dynamic.cljs (C).**
   - `regen-fast-eligible?` (:206-279) is a schema-based proof obligation. It
     produced the regenerate saga: 22 bug beans, including njzu (critical),
     yep2, dv66, 9yuw, 175y, wl1y and ovop.
   - The analytical dispatcher (:830-908) carries three classes of guard (jmk4,
     3k3x, reopen).
   - `joint-rescore-marginal` (:946-1004), plus a bail-and-retry with an
     analytical strip inside `run-dispatched*` (:1012-1037).
   - `make-gen-fn` (:1338-1436) runs schema extraction, compilation, conjugacy
     detection and the rewrite engine on **every** `gen` construction.
   - `vgenerate-compiled` (:1521-1627) throws on Metal and documents a
     "16-output NAPI ceiling".
2. **The L3 analytical stack (C).** CLAUDE.md calls it the "highest-risk
   surface". It has 30 bug beans, about 9 of them silent wrong marginals
   published as `:exact`:
   - ke9i, lwhw.
   - rmy7: a 5-nat error.
   - 94qc: matching by *name*.
   - 5ytq.
   - su6q: a diagonal-only probe.
   - yy8u: an affine probe that accepted a non-affine mean, producing wrong
     gradients on the default path.
   - jmk4, 3k3x.

   compiled_ops.cljs:1428-2106 is about 700 LOC of affinity proofs and numeric
   probes.
3. **The compiled path (C).** The noise transforms in compiled.cljs:615-770
   re-implement distribution log-probs ("Support guard ported verbatim from
   dist.cljs (genmlx-4x5w)"), and the copies drifted (b210, kftc, 4x5w). There
   are 24 op builders and 39 bug beans.
4. **mcmc.cljs (3865 LOC).** Every algorithm is multiplied across plain,
   compiled, fused, fused-compiled, vectorized, fused-vectorized and chunked
   variants.
   - Backend-measured magic constants: `persist-trace-ops-limits` (:229-253) and
     `fused-ops-limits {:metal… :cuda…}` (:880-937).
   - Routing through the `GENMLX_HMC_CHUNK_OPS` environment variable.
   - Six copies of the `degraded` volatile fallback (:341, 377, 416, 1067, 2267,
     3254).
   - 81 cleanup/materialize calls; 102 commits and 37 fixes.
5. **combinators.cljs (3595 LOC).** Every op has three paths (Unfold
   simulate/generate at :520-684, for example). There are 26 `ensure-joint-self`
   calls, the same record is reopened in up to six `extend-type` blocks, there
   are `fused-cache` atoms, and there are 25 bug beans.
6. **SMC.** The default fixed-args `smc` pre-traces the observations and
   reweights through `project` (smc.cljs:233-303; uxjm critical). That is not
   Gen.jl's semantics. There are 29 SMC bug beans. `smcp3-step` takes 10
   positional args with three weight conventions selected by which args are nil
   (smcp3.cljs:76-195).
7. **PRNG hygiene.** 34 bug beans (njaq, ivs0, g5ys, vv3t, wkbc, 9ssv, 0zr4, r9xv,
   n7du, plus gth5 still open). `vmap.cljs:226` and `:270` still call
   `(rng/fresh-key)` and ignore the threaded key; this is probably an unswept
   sibling of 9ssv (unverified).
8. **The LLM layer.** 14 open bugs and model-specific forward passes. Until
   2026-08-01, 40 of its 49 tests could not fail (n061).

## 3. Design smells not to carry into gen.clj

1. **Optimizations that change GFI semantics.** L3's `generate` returns
   eliminated latents at their *posterior mean* with a marginal weight (the
   docstring at protocols.cljs:13-30 admits it). That one choice forced:
   - score-type metadata;
   - `strip-analytical-path` in every sampler (mcmc ×9, kernel, smc, importance,
     translator);
   - `assert-joint!` and `joint-rescore-marginal`;
   - 26 `ensure-joint-self` calls;
   - a local copy of the strip in importance.cljs:11-17 that drifted (jr90).

   → Make marginalization an explicit, separately typed GF (for example a
   `marginalize` wrapper) with its own exact density, never a hidden dispatch
   path on the same GF.
2. **Duplicated semantics across paths:** compiled noise transforms vs dist.cljs;
   6 batched transitions copying the scalar ones (handler.cljs:265-436, differing
   only in `dist-sample-n`); triple-path combinators; exact vs enumerate;
   conjugate vs auto_analytical. → One implementation, parameterized.
3. **Static analysis of quoted, unexpanded source by symbol name.**
   schema.cljs:19-34 resolves `dist/gaussian` by its name string, and
   compiled.cljs:305-330 is a symbol→fn table. This caused false positives (94qc,
   7zuq) and missed closure variables (bg67). → If static structure is needed,
   use an explicit static DSL/IR, like Gen.jl's static modeling language.
4. **PRNG keys in gen-fn metadata,** with an auto-key sentinel and silent
   `fresh-key` fallbacks (dynamic.cljs:37-48, combinators.cljs:30-53). To dodge
   circular requires, keywords namespaced to dynamic (`:genmlx.dynamic/key`,
   `:genmlx.dynamic/param-store`) appear 45 times across 9 namespaces.
5. **Hidden global and dynamic state:**
   - the `dispatch-fn` atom (swapped by dev.cljs);
   - the `validate-fn` atom, dereferenced in the `run-handler` hot path;
   - the `fallback-notice-fn` atom;
   - `*force-general-regen*`, a test hook inside production dispatch;
   - 5 atoms and 4 mutable counters in mlx.cljs.
6. **Memory management inside the semantics layer:** `mx/auto-cleanup!` on every
   trace call (runtime.cljs), `gfi-cleanup!` on every DynamicGF op, and
   `sweep-dead-arrays!` / `eval!` inside the gamma sampler (dist.cljs:410-445).
7. **Protocol accretion.** Three update protocols, and no retdiff. → Target
   Gen.jl's single `update(trace, args, argdiffs, constraints) → (trace, w,
   retdiff, discard)`.
8. **Duck-typing on DynamicGF internals.** 22 `(:body-fn …)` checks, and about
   15 files read `(:schema gf)`.
9. **An aggregator namespace** (inference.cljs) that inflates impact sets. CLAUDE.md
   counts 113 test files in mcmc's reverse closure.
10. **Gradient MCMC over flat top-level parameter vectors.** It throws on nested
    latents (util.cljs:409-460). Gen.jl uses `choice_gradients` over selections
    instead.

## 4. Hidden coupling that would drag messy parts along with the core
- **dynamic.cljs pulls in the whole optimization stack.** It requires schema,
  compiled, compiled-ops, conjugacy, rewrite, linear-gaussian,
  inference.auto-analytical and vectorized, for a transitive closure of
  **27 namespaces / 16.6k LOC**. `make-gen-fn` calls all of them. **DynamicGF
  must be rewritten, not ported verbatim.**
- **gfi.cljs pulls in auto_analytical** via dynamic, plus combinators, mcmc, smc,
  translator, encapsulated, learning and verify: **38 namespaces / 30k LOC**.
  Split the laws by feature.
- **Combinators require compiled-ops** (→ compiled, dist) for their fast paths:
  17 namespaces / 13.9k LOC.
- **inference/util requires compiled-ops** (tensor scores), so *every* inference
  algorithm pulls in the compiled path.
- **importance requires smc** just for `log-ml-increment`, and smc requires
  combinators for `smc-unfold`. Importance's closure is therefore
  31 namespaces / 21.9k LOC.
- **handler.cljs knows about combinators** (`combinator-batched-fallback`,
  :484-620), and runtime inlines about 90 lines of batched-splice logic.
- **exact.cljs mixes generic enumeration with the agents/memo API** (`thinks`,
  `agent-marginal`, `with-cache`), and agents depend on it.
- **program.cljs is a top-level, core-looking file** that requires llm.backend.
- **The good news.**
  - No other core namespace requires llm, agents, world or control.
  - The closures of handler (3.5k LOC), dist (5.1k), edit (0.6k) and vmap (4.2k)
    are clean.
  - Cutting dynamic.cljs's L1/L3 requires and inference/util's compiled-ops
    require isolates the beautiful core.

**Gen.jl-parity gaps beyond GenMLX itself:** retdiffs, a static DSL with an IR,
`accumulate_param_gradients!`, argument- and selection-based `choice_gradients`,
and a Gen.jl-style particle-filter API.
