# GenMLX features gen.clj does not have (2026-10-24)

*Read-only survey. GenMLX at d8149c7 (the commit the 2026-10-18 triage
read), gen.clj main at e358e8c. Every "ported" entry was checked by
grepping gen.clj's `src/` and `test/` for the namespace or function, not
taken from docs. "Tracked" means an open bean names the GenMLX code.
"Dropped" cites the triage doc, DESIGN.md or the bean that decided it.
The earlier comparison (genclj-dglq, 2026-10-23) and round 1 of the extras
(genclj-0qds) are taken as the baseline; this survey looks for what both
missed.*

Classes: **a** ported · **b** tracked · **c** dropped · **d** an
untracked gap (G-numbers, detailed below).

## 1. Classification by GenMLX namespace (`src/genmlx/…`)

### Core data, GFI and modeling language

| GenMLX | LOC | Class | gen.clj, bean or reason |
|---|---|---|---|
| protocols | 84 | a | gen.gfi: one Gen.jl-shaped update. IBatchedSplice dropped (triage, core table). IHasArgumentGrads is not needed: `choice-gradients` takes `:argnums`, and gen.custom gives custom VJPs. |
| choicemap | 287 | a | gen.choicemap, with stack/unstack, pair/unpair and to/from-array. `loop-obs` and `from-flat-map` are not needed because vector paths do their job. |
| selection, diff | 92+41 | a | gen.selection, gen.diff. gen.clj goes further: IntDiff, SetDiff, DictDiff and retdiffs. |
| trace | 123 | a | gen.trace. Score-type tags dropped (triage §3.1, DESIGN §3.1). |
| handler, runtime | 620+181 | a | gen.dynamic, with one transition set. The batched copies are dropped (DESIGN §3.3). |
| dynamic | 1896 | a | gen.dynamic, plus gen.batched for the v* ops. The dispatcher stack, `with-key` and `auto-key` are dropped (triage §3.4, rule 2). `call` became directly callable GFs (genclj-ou82). |
| dispatch | 88 | a | gen.dynamic `with-transitions`, `with-dispatch` |
| gen.cljc, dist/macros.cljc | 29+88 | a | gen.dynamic/gen, gen.dist.core/defdist |
| edit | 235 | b | genclj-ob44 |
| gfi (87 laws) | 2523 | a, test only | test/gen/laws.clj (`deflaws`). Users cannot call the laws from src (**G7**). genclj-04h2 asks whether gen.laws belongs in 1.0. |
| verify | 260 | b | genclj-89yj. Repeated addresses are already a loud runtime error. |
| serialize | 427 | a | gen.serialize. EDN rather than JSON by decision (DESIGN §9, round 1). ω traces wait on genclj-t8uj. |
| inspect, dev, schemas | 84+128+283 | b | genclj-zg4g |
| custom_gradient | 101 | a | gen.custom (`deterministic`, `array/custom-vjp`) |
| gradients | 85 | a | gen.gradients, which goes further: selections, retgrad, param gradients, Jacobians. |
| learning | 352 | a | gen.learning (sgd, adam, train, lecture). GenMLX's `wake-sleep` (point values, "wake-sleep in name only" per its own docstring at learning.cljs:276) is covered by lecture plus vi's `:model-store`. |
| encapsulated | 562 | a | gen.custom/estimated. PMMH is MH on an estimated density. test/gen/encapsulated_test.clj covers GenMLX's mixture and marginalized-Gaussian examples. |
| nn | 389 | b | genclj-koxn |
| vectorized | 221 | a | gen.batched, gen.array.batch |
| **vmap** (vmap-gf, repeat-gf) | 464 | **d** | **G1** |
| combinators | 3595 | a | gen.combinators: map, unfold, switch, recurse, scan, mix, dimap, contramap, map-retval. The vectorized switch is batched by partition (gen.partition). Mask is dropped (DESIGN §9, round 1; genclj-2gma). The fused and compiled paths are genclj-syyi. |
| dist/core | 341 | a | gen.dist.core, plus gen.dist's mixtures and product. `map->dist` (a distribution built from closures) is covered by defdist and gen.dist.dsl. |
| dist | 2152 | a | gen.dist, every family except delta (dropped: DESIGN §9, round 1). The "reparameterized" flags have no equivalent (**G6**). |
| mlx, mlx/random, mlx/constants | 1759+171+29 | a, narrower by decision | gen.array (with .linalg, .special, .tape, .batch, .memory), gen.backend.mlx, gen.random. The op surface is kept to what the core uses (DESIGN §5.1). It has no sort, argsort, topk, searchsorted, qr, svd, eigh, einsum, tile, pad, roll, split, put-along-axis or conv2d. LLM ops belong to genclj-gvfa. |

### Static analysis, compilation and the L3 stack

| GenMLX | LOC | Class | gen.clj, bean or reason |
|---|---|---|---|
| schema | 1048 | a, redesigned | gen.dynamic.static, an explicit IR with `ir`, `dot` and `lint` (triage §3.3) |
| compiled, compiled_ops, compiled_gen, tensor_trace | 1502+2388+56+135 | b | genclj-syyi, plus genclj-f5fy for generated GFI code. Compiled scores are done (genclj-rj78). |
| conjugacy, affine, linear_gaussian, rewrite, auto_analytical | 340+455+603+332+869 | c | DESIGN §3.1's corollary. genclj-702q made conjugacy opt-in, with no detection (gen.inference.conjugate). Its extensions are genclj-mwha. |
| dep_graph (Markov blankets, Gibbs blocks) | 259 | c | Support for the L3 rewrite engine (702q's scope). There is no counterpart query on the static IR. Skip. |
| method_selection, fit | 256+248 | b | genclj-0sfu |
| pef | 1002 | a | test/gen/fuzz_test.clj, static_fuzz_test.clj (genclj-y0ij) |
| program | 1258 | b | genclj-0izr |

### Inference (`inference/…`)

| GenMLX | LOC | Class | gen.clj, bean or reason |
|---|---|---|---|
| inference.cljs (aggregator) | 130 | c | Triage says "do not port"; DESIGN §3.8 |
| importance | 175 | a | gen.inference.importance (`:batched?`) |
| kernel | 362 | a | gen.inference.kernel (chain, repeat, cycle, mix, checked, kern, run). `seed` is dropped (rule 2). The convenience kernels are absent (**G9**). |
| mcmc | 3865 | a, narrower | gen.inference.mh, .mcmc, .nuts. It has a diagonal metric only and no MALA adaptation (**G5**). Fused chains are genclj-l9ts. |
| smc | 846 | a | gen.inference.particle-filter |
| smcp3 | 294 | a | gen.inference.trace-kernel/smcp3, transform/updating-translator |
| pmcmc | 231 | a | gen.inference.particle-mcmc. Ancestor sampling is genclj-6wwn. |
| steppable | 125 | a | The particle filter's state is the steppable value (genclj-6mv2) |
| cost | 94 | b | genclj-6mv2 |
| translator | 416 | a | gen.inference.transform. Reversible jump is tested (transform_test.clj:268). `coarse-to-fine-smc` can be composed from pf/move-reweight and a translator to a new model, but is untested (**G10**). |
| diagnostics | 302 | a | gen.inference.diagnostics (does more) |
| enumerate, exact | 130+745 | a / **d** | gen.inference.enumerative. The queries are **G2**. The memo/agents API (`thinks`, `observes`, `pr`, `with-cache`, the collapsed `enumerate` GF) lives in memo.clj (~/code/memo.clj) and genclj-nt3u. |
| conjugate | 408 | a | gen.inference.conjugate (covers more families) |
| kalman, hmm_forward | 279+268 | a | gen.state-space. Missing observations are genclj-llbf. |
| analytical, ekf, ekf_nd | 43+237+391 | b, stale | genclj-mgvu (see §3) |
| fisher | 275 | a | gen.inference.laplace (genclj-07eb, genclj-5e0d) |
| rb_mixture | 215 | b | genclj-3gu0 |
| vi | 502 | a, narrower | gen.inference.vi: black-box-vi and vimco, score-function only. ADVI is **G6**. The programmable objectives are genclj-hc35. |
| adev | 323 | b | genclj-hc35 |
| amortized | 275 | b | genclj-koxn |
| compiled_gradient, compiled_optimizer, compiled_smc, differentiable, differentiable_resample | 193+636+213+108+173 | b | genclj-cdlp |
| util | 638 | a | gen.inference.weights. The flat-parameter score functions are dropped (triage §3.10). |

### LLM layer (`llm/…`)

| GenMLX | LOC | Class | gen.clj, bean or reason |
|---|---|---|---|
| backend | 1105 | a | gen.llm.llama, .lm, .slots, .cache, .branching. Model families come from GGUF. The MLX-native engine is genclj-gvfa. |
| forward, qwen3_forward, qwen35_forward | 215+486+921 | b | genclj-gvfa |
| qwen35_vision_forward, vision | 362+244 | b | genclj-oxz2 |
| core | 477 | a | gen.llm.token/completion. In a batched run a completion falls back to one particle at a time, by decision (DESIGN §9, token GF; llm-token-gf.md §7). Batched generation is token SMC's job. |
| branched | 287 | a | Token GF regenerate with `(token/from k)`, over the trie's score-once ledger |
| bytes | 457 | a / **d** | gen.llm.bytes, whose text probability is exact over all tokenizations (more than GenMLX's). The byte-level GF is **G4**. |
| grammar | 766 | a | gen.llm.grammar, .regex, .dfa |
| smc | 634 | a / **d** | gen.llm.smc. Rejuvenation is **G3**. |
| sampling | 144 | b | genclj-paid |
| schema_grammar, structured, toolcall | 271+219+545 | b | genclj-w120 |
| codegen, msa, msa_score | 486+498+231 | b | genclj-0izr |
| pi_session, pi_assess, pi_edit | 291+202+371 | b | genclj-p79i |
| pi_provider (+ `extensions/genmlx-tools.ts`) | 772+227 | b | genclj-5pxi. genclj-p79i says the provider is an application. |

### Research verticals and infrastructure

| GenMLX | LOC | Class | Bean |
|---|---|---|---|
| agents/* (14 files, including remote, pacman, belief, `factor-dist`) | 2986 | b | genclj-nt3u. The LLM-agent examples are genclj-xl5e. Theory of mind in memo's style is memo.clj's. |
| control/* (4) | 652 | b | genclj-rhzp |
| world/train, train_reward, session_grpo, session_reward, sft | ~1340 | b | genclj-q4xv |
| world/llm_proposer | 442 | b | genclj-0izr |
| world/* (the other 12) | ~3000 | b | genclj-sjex |
| sensorimotor, tabular | 374+310 | b | genclj-qtdh |
| memory, sandbox, sandbox_child | 711+199+79 | b | genclj-xu25 |
| codegen/eval | 243 | b | genclj-0izr |

### Outside `src/`

| GenMLX | Class | Note |
|---|---|---|
| docs/tutorial (10 chapters), docs/reference (8 pages), examples/genmlx (10), examples/distinctive (8) | **d** | **G8** |
| examples/memo, examples/agentmodels | b | memo.clj; genclj-nt3u |
| formal/ (λ_MLX calculus, semantics, 13 proof notes) | d, skip | No counterpart. gen.clj audits its laws against the thesis instead (docs/research/law-audit.md). Ask the owner only if a formal write-up is wanted. |

**The one triage item with neither a port nor a bean is vmap.cljs (P3).**
The rest of section 1's (d) items are features inside namespaces that
were otherwise ported.

## 2. Untracked gaps

### G1. A Map's elements in one batched run (GenMLX's vmap-gf, repeat-gf and fused Map)
- **GenMLX:**
  - vmap.cljs:219-300: the fast paths run the kernel's body once over [N] elements.
  - vmap.cljs:451-464: `vmap-gf` (`:in-axes`, `:axis-size`) and `repeat-gf`.
  - combinators.cljs:214-285: `map-simulate-fused`.
  - GenMLX's fast path covered only simulate and unconstrained generate. A constrained generate looped per element, and it drew with `rng/fresh-key`, the PRNG bug the triage flags (§2.7).
- **gen.clj today:** comb/map calls the kernel once per element (combinators.clj:1-66). Batching runs across particles, never across a trace's elements. The static benchmark's 1000-observation Map update costs about 2.9 ms (docs/benchmarks.md:615-623).
- **Size:** M (L when the Map sits inside a batched particle run, which needs two batch axes; gen.partition notes "a partition needs a one-axis batch").
- **Layer:** L4, gen.combinators, over L0's batch tracer and gen.batched's per-particle constraints (genclj-7oc1). Those cover the constrained case GenMLX's fast path lacked.
- **Recommendation:** a new bean, with a design question for the owner:
  - (a) An observationally equivalent plug-in inside comb/map (and a `repeat`). The elements run as the particles of one batched sub-run, the Map's trace and addresses are unchanged, and it falls back per element on the batched-branch error, as rule 1 requires.
  - (b) GenJAX's vmap: [N] leaves at one address, a new address scheme.
  - I recommend (a). Batched importance sampling's 14-26× (genclj-ffxh) is the same per-site amortization, so a similar gain is plausible but not measured.

### G2. Queries on an exact enumeration
- **GenMLX:**
  - exact.cljs:338-745: `normalize-joint`, `expectation` (:397), `entropy` (:412), `variance` (:431), `condition-on` (:454), `joint-marginal` (:500), `conditional-marginal` (:561), `mutual-info` (:730).
  - genjl-parity.md §2 lists "entropy/mutual-information queries" as a GenMLX extra.
- **gen.clj today:** gen.inference.enumerative has `marginal` only (enumerative.clj:230).
- **Size:** S.
- **Layer:** L5, gen.inference.enumerative, as host functions over results, batched or not.
- **Recommendation:** a new bean, low priority. Port the generic queries, not the memo API (`thinks`, `observes`, `pr`, `with-cache`), which genclj-3jp2 excluded and memo.clj owns. Ask the owner whether memo.clj or GenGDT already covers the need.

### G3. Rejuvenation in token SMC
- **GenMLX:** llm/smc.cljs:517-610 (`rejuvenate-particles`, `:rejuvenation {:steps :selection :gf}`, run at the filter's end).
- **gen.clj today:** docs/research/llm-token-smc.md:148 lists it as "Not built (later)", together with a batched potential protocol and SMCP3-style proposals. No bean exists for any of the three.
- **Size:** S-M.
- **Layer:** LLM, gen.llm.smc.
- **Recommendation:** a new bean, low priority: `rejuvenate` on a token-SMC state, applying a kernel (MH with `(token/from k)`) to each particle's trace with move-accept weights, as pf/rejuvenate does. File the other two items of that note beside it.

### G4. A byte-level generative function
- **GenMLX:** llm/bytes.cljs:302-400 (`make-byte-llm-gf`, `decode-byte-trace`, `constrain-bytes`). It followed one greedy tokenization.
- **gen.clj today:** docs/research/llm-grammar.md:331-341 ("For later (not built)") sketches an exact version over gen.llm.bytes' runs. No bean.
- **Size:** M.
- **Layer:** LLM, gen.llm.bytes plus a GF like gen.llm.token.
- **Recommendation:** a new draft bean under M6 or genclj-skwp. It gives character-level control, and grammar masks without a vocabulary sweep.

### G5. Dense HMC/NUTS metric and MALA step-size adaptation
- **GenMLX:**
  - mcmc.cljs:2556-2580: dense-metric momenta.
  - mcmc.cljs:2810-2835: hmc `:metric`, accepting a vector or a matrix.
  - mcmc.cljs:3517-3531: nuts `:metric`.
  - mcmc.cljs:1920-1975: mala `:adapt-step-size`, toward 0.574.
- **gen.clj today:** `:inv-metric` is diagonal (mcmc.clj:18-24), and nuts/adapt takes `:kernel :nuts` or `:hmc` only. genclj-tvc0's summary names "a dense metric, … adapting MALA" as follow-ups, but none was filed.
- **Size:** S each, or M with Stan's dense_e adaptation.
- **Layer:** L5, gen.inference.mcmc and .nuts.
- **Recommendation:** a new bean, low priority:
  - `:inv-metric` as a matrix, with momenta from its Cholesky factor.
  - Stan's dense adaptation as an option of `adapt`.
  - `:kernel :mala` in `adapt`.

### G6. Reparameterization-gradient VI (ADVI)
- **GenMLX:**
  - vi.cljs:17-145: `elbo-estimate`, `run-advi` and `vi`, a mean-field Gaussian with pathwise gradients.
  - vi.cljs:190: `vi-from-model`, an automatic guide.
  - dist/core.cljs:19-26: `dist-reparam` and the README's "Reparameterized" column.
- **gen.clj today:** black-box-vi and vimco use the score-function estimator only (vi.clj:9-17). genclj-hc35 ("ADEV and programmable VI objectives") does not name the pathwise estimator or an auto-guide. gen.clj's samplers are already pathwise where the math allows: normal draws `mu + std·ε` (dist.clj:138-139), which the tape differentiates.
- **Size:** M.
- **Layer:** L5, gen.inference.vi, plus L3, a per-family "reparameterizable" mark.
- **Recommendation:** widen genclj-hc35, or add a new bean:
  - `:estimator :reparam` in black-box-vi, loud when a q site's family has no pathwise sampler.
  - Optionally, an automatic mean-field guide over a trace's continuous selection.
  - This is the most-used VI estimator in practice, with much lower variance than REINFORCE.

### G7. Laws users can run on their own generative functions
- **GenMLX:** gfi.cljs:2455-2523 (`check-law`, `verify model args :law-names :tags :n-trials`), callable from src at the REPL.
- **gen.clj today:** the laws exist only in test/gen/laws.clj, which needs test.check and gen.test-support. memo.clj already uses `gen.laws` from its tests, and genclj-04h2 asks whether gen.laws joins the 1.0 surface.
- **Size:** S-M.
- **Layer:** test support, or an optional module.
- **Recommendation:** ask the owner. One option is an optional module like ext/commons-math, depending on test.check, with a `check` a user runs on a `defgf`, a `gen.custom` function or a new combinator. Rule 7 and rule 8 bear on the choice.

### G8. User documentation and runnable examples
- **GenMLX:** docs/tutorial (10 chapters, from the introduction through custom extensions), docs/reference (8 pages), examples/genmlx (10 programs, such as linear regression, HMM, adaptive HMC, VI, token SMC), examples/distinctive (8).
- **gen.clj today:** README.md (157 lines) is stale. It says "Local LLMs as generative functions (M3) come next" (:19) and "M3, in progress" (:147), although M3 and M5 are complete. It shows no token GF, token SMC or grammar. Beyond it there are docs/array-api.md and the porting guide. No bean covers user documentation; genclj-04h2 (1.0) does not list it.
- **Size:** S for the README refresh; M-L for a tutorial and examples.
- **Layer:** docs.
- **Recommendation:** a new bean under genclj-04h2:
  - Refresh the README now.
  - Then write an executable handbook (memo.clj's HANDBOOK.md with handbook_doc_test.clj is the pattern), porting GenMLX's tutorial chapters and examples as namespaces that run under `-X:test`.

### G9. Convenience kernels
- **GenMLX:** kernel.cljs:280-362 (`random-walk`, `prior`, `proposal`, `gibbs`).
- **Size:** S.
- **Layer:** L5, kernel.
- **Recommendation:** skip. Gen.jl has none of these. `(kernel/mh selection)` covers `prior`, and a two-line proposal GF covers a random walk. At most, add a cookbook entry in G8.

### G10. A coarse-to-fine SMC driver
- **GenMLX:** translator.cljs:338-400.
- **gen.clj today:** it can be built from pf/move-reweight with a general translator to a new model. No test does so.
- **Size:** S.
- **Recommendation:** skip as API. If wanted, add a test or example in G8.

## 3. Tracked, but the bean has gone stale

- **genclj-mgvu** ("Analytical middleware: Kalman, EKF, EKF-ND, HMM forward, conjugate handlers"). genclj-702q built Kalman and HMM as generative functions (gen.state-space) and conjugacy as proposals, neither as middleware. What is left is the EKF (ekf.cljs is 1-d, ekf_nd.cljs N-d), whose density would be the linearized model's, not the true marginal. Recommendation: retitle it to "EKF as a gen.state-space generative function (the linearized model's exact density)", or close it as skip.
- **genclj-hc35** does not cover G6's pathwise estimator or auto-guide.
- **genclj-tvc0** named its follow-ups (G5) but filed none.
- **Doc-only "later" items without beans:** llm-token-smc.md:148 (G3, a batched potential, SMCP3-style token proposals) and llm-grammar.md:331 (G4).

## 4. Ported, but narrower than GenMLX

- **VI:**
  - Score-function only (G6).
  - No PWake or QWake (genclj-hc35). IWELBO is the bound VIMCO already ascends.
- **MCMC:**
  - Diagonal metric only; MALA is not adapted (G5).
  - MH chains are not compiled (genclj-l9ts).
  - map-optimize is Gen.jl's line search, with no Adam option (GenMLX had `:optimizer :adam`). Laplace's Newton `choice-mode` covers the use; skip.
- **Enumeration:**
  - `marginal` only (G2).
  - A batched grid rather than broadcast tensor axes (by design, genclj-7oc1).
- **Conjugacy:**
  - One latent per wrapper, with no affine children, so no linear-Gaussian regression elimination (genclj-mwha).
- **State space:**
  - No missing observations (genclj-llbf).
  - No EKF (genclj-mgvu).
- **Particle MCMC and Laplace:**
  - No ancestor sampling (genclj-6wwn).
  - Complete-data Fisher information only (genclj-07eb).
  - A Gaussian Laplace proposal only (genclj-5e0d).
- **LLM:**
  - In a batched run, completions fall back per particle (by decision).
  - Token SMC has no rejuvenation (G3).
  - There is no byte-level GF (G4), though text probability is more exact than GenMLX's.
- **Combinators:**
  - No Mask (by decision).
  - A Map's elements run one call each (G1).
- **Array API:** minimal by decision (DESIGN §5.1-5.2: "EJML only if LU, QR or eig become necessary"). genclj-paid will need sort and topk, and genclj-gvfa needs rms-norm, rope, sdpa and quantized matmul. Add ops when a bean needs them.
- **Distributions:**
  - student-t takes one parameter; loc and scale come by pushforward.
  - No delta.
  - Both are decisions (porting guide; DESIGN §9).

## 5. Most worth doing, ranked

1. **G1, batched Map elements:** the largest performance gap against GenMLX for data-heavy models, and the machinery (per-particle constraints, the fallback) exists. The owner chooses the plug-in or GenJAX's vmap. I recommend the plug-in.
2. **G8, the README now (S), then a tutorial with examples:** a 1.0 prerequisite, since the README misstates the project's status.
3. **G6, pathwise ELBO gradients:** VI's workhorse estimator. Widen hc35 or file a bean.
4. **G5, dense metric and adaptive MALA:** small, Stan-compatible, and already named as follow-ups.
5. **G2, enumeration queries:** small. Check memo.clj and GenGDT first.
6. **G3, token-SMC rejuvenation:** small, and completes token SMC.
7. **G7, laws for users:** ask the owner, together with genclj-04h2's surface question.
8. **G4, the byte-level GF:** when M6 or structured output needs character-level control.
9. **Groom genclj-mgvu** (bean hygiene).

Skip: G9, G10, dep_graph, `map->dist`, `loop-obs`/`merge-obs`, formal/.
