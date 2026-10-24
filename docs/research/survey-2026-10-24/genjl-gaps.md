# What Gen.jl has that gen.clj does not (2026-10-24)

Read-only research at gen.clj e358e8c. Gen.jl paths are relative to
`~/.julia/packages/Gen/mP0Sq/` (0.4.8 = the tag); gen.clj paths to the repo.
Every gen.clj status below was checked by grepping `src/`, `test/` and `ext/`,
and beans with `beans list --json -S`. Nothing was run. Where a claim rests on
a package's public page and not its code, it says so.

**Method.** I went through every `export` in Gen.jl's `src/` (≈150 names)
and every `@docs` name in `docs/src/ref/`. I checked each against gen.clj's
public vars and against `docs/research/genjl-parity.md`,
`docs/research/genjl-test-suite.md` (§3's n/a reasons, §4.1 D1–D16, §4.2
G1–G15) and DESIGN.md §9. I also looked at Gen.jl's master after 0.4.8, the
probcomp GitHub org (58 repositories that match "Gen"), and gen.dev/ecosystem.

**Bottom line.** gen.clj covers essentially all of 0.4.8's exported core API,
because M4 and M5 were thorough. The real gaps are:
- in the **ecosystem**: AIS and SIR kernels, inference-as-a-distribution
  (GenSP), variable elimination, directional statistics, more optimizers,
  NN-as-GF and visualization;
- a few **core ergonomics and extension points**: the optimizer interface,
  `train!`'s semantics, tilde sugar and pretty printing;
- **one post-0.4.8 feature**: HMC with a dense metric;
- **documentation**: no tutorials, no migration note, a stale README;
- **scalar speed**, which beans already track.

---

## 1. Core Gen.jl gaps

### C1. The optimizer extension point (`init_update_state` / `apply_update!`)
- **Gen.jl:** `src/optimization.jl:2-16`, and `docs/src/ref/inference/parameter_optimization.md:38-43` ("Custom update configurations should implement the following methods"). Any user type can be an update configuration of a `ParamUpdate`. GenFluxOptimizers is built on this interface.
- **gen.clj:** absent. `gen.learning/init` and `step` dispatch with a closed `case` on `:kind` (`src/gen/learning.clj:73-104`), and `unknown-kind` throws for anything else (`:68`). A user cannot add an optimizer without editing gen.learning.
- **Deliberate or tracked?** Neither. No bean.
- **Size and layer:** S, L5 (`gen.learning`).
- **Recommendation: port.** Make `init`/`step` a multimethod on `:kind` (or a protocol on optimizer maps or records). It keeps the pure `[store' state']` contract, and E5 then becomes a few small definitions.

### C2. `train!`'s data regime, held-out evaluation and callback
- **Gen.jl:** `src/inference/train.jl:24-75`.
  - An epoch's dataset is generated once (`:40`), and each minibatch is a random subset of it (`:48`).
  - The history is the mean generate **weight** on `evaluation_size` fresh examples (`:63-70`).
  - `callback(epoch, minibatch, objective)` and `verbose` are also options.
- **gen.clj:** partial.
  - `learning/train` (`src/gen/learning.clj:105-130`) draws fresh data for every batch.
  - Its history is the last batch's mean **score**, that is, the joint log p of all of a trace's choices.
  - It has no held-out set and no callback.
  - So when the constraints do not cover every choice, the history measures something other than Gen.jl's objective. The gradients agree (both differentiate the trace's score).
- **Deliberate or tracked?** Not deliberate (M4 noted only "train! … tested"). No bean.
- **Size and layer:** S, L5.
- **Recommendation: port.** Add `:epoch-size` (minibatches drawn from the epoch's data), `:evaluation-size` (a held-out weight as the history) and `:callback`. The history's meaning is a small semantic choice for the owner: a weight like Gen.jl's, or keep the score.

### C3. HMC with a dense metric (Gen.jl master, after 0.4.8)
- **Gen.jl:** PR #559 (https://github.com/probcomp/Gen.jl/pull/559), merged 2025-10-22 and not yet released. `hmc(...; metric)` takes `nothing`, a vector, a `Diagonal`, or a dense matrix, whose momenta are drawn from and scored by `mvnormal`. The 0.4.8 `hmc` (`src/inference/hmc.jl:25`) has no metric. Master after 0.4.8 holds only this and doc renames (#562), so nothing else is new.
- **gen.clj:** diagonal only.
  - `:inv-metric` must be a positive scalar or one variance per value (`src/gen/inference/mcmc.clj:110-126`), for `mala` (`:287`), `hmc` (`:391`) and `nuts` (`:538`).
  - Stan's warmup adapts a diagonal only (`src/gen/inference/nuts.clj:216`).
- **Deliberate or tracked?** Not deliberate. Bean tvc0 chose a diagonal for Stan parity (`diag_e`). No bean for a dense metric.
- **Size and layer:** S for the kernels (one Cholesky per move, momenta from N(0, M), kinetic ½pᵀM⁻¹p). M with Stan's `dense_e` windowed adaptation. Layer L5.
- **Recommendation: port.** It helps strongly correlated posteriors. The owner decides whether the adaptation is in scope.

### C4. Tilde sugar: `x ~ dist(...)` binds `x` and addresses it `:x`
- **Gen.jl:** `src/dsl/dsl.jl:83-101` (`desugar_tildes`). `x ~ normal(0, 1)` is `x = @trace(normal(0, 1), :x)`, and `{addr} ~` and `{*} ~` are the explicit forms.
- **gen.clj:** absent. A choice is always `(let [x (trace :x (d/normal 0 1))] …)`. `(splice gf …)` covers `{*} ~`, and the explicit form is `trace`. The test port `test/gen/genjl/tilde_sugar_test.clj` writes the explicit form.
- **Deliberate or tracked?** Not discussed in DESIGN §9. No bean.
- **Size and layer:** S, L2 (the `gen` macro, `gen.dynamic.guard`).
- **Recommendation: ask the owner.** This is pure style. Two options:
  - a `trace` with a bare symbol in a `let` binding, `(let [x (trace (d/normal 0 1))])`, addressed by the binding's name;
  - leave it out, since explicit addresses are clearer in Clojure.

### C5. Trivial names and helpers (one line each)
- `metropolis_hastings` (an alias of `mh`), `uniform_continuous` (an alias of `uniform`): absent as names. Skip.
- `Base.isapprox` for choice maps (`src/choice_map.jl:323`): absent in `gen.choicemap`. No Gen.jl test uses it. Skip unless a user asks.
- `ParticleFilterState` copy semantics: n/a, since gen.clj's state is immutable (test-suite doc §3, 8 n/a).

### Checked, and not gaps (deliberate, or present under another name)
- **Default positional arguments.** n/a by decision (G3, DESIGN.md:741).
- **`(static, diffs)` and `nojuliacache`.** gen.clj always tracks diffs and keeps node values (D10, `genjl-parity.md:73`).
- **The static IR's internal shape.** Not ported by decision (G15, DESIGN.md:749).
- **`StaticIRBuilder`, `build_ir`, `add_*_node!` (`src/static_ir/dag.jl:82, 207-213`).** Programmatic IR construction is native in Clojure: forms are data, so build a `(gen ^:static …)` form and `eval` it. For a dynamic function, `dyn/make-gen-fn` takes a body fn (`src/gen/dynamic.clj:807`). Skip.
- **Julia type-level machinery.** Address schemas, `StaticChoiceMap`, `StaticSelection`, `Trie`, `InternalVectorChoiceMap`/`vectorize_internal` and `load_generated_functions` (test-suite doc §3's internal-API n/a).
- **Mutable parameter stores and gradient accumulators.** `get_param_grad`, `zero_param_grad!` and the scale factor: one explicit store per model, and gradients are values (D7, DESIGN.md:491).
- **Gradient flags.** `has_argument_grads`, `accepts_output_grad`, `has_output_grad` and `logpdf_grad` are moot under full AD (`genjl-parity.md:43-44`). A hand-written density gradient is `array/custom-vjp` inside a `defdist`'s log-prob.
- **Distributions as generative functions.** In 0.4.8 `Distribution` is not a GF (`src/modeling_library/modeling_library.jl:8`). gen.clj's distributions implement `IGenerativeFunction` (`src/gen/dist/core.clj:399`), so gen.clj goes further than Gen.jl here.
- **The rest of the exported API** (the GFI, choice maps, selections, diffs, combinators, `@dist`, mixtures, products, the 20 distributions, IS, MH, HMC, MALA, MAP, elliptical slice, enumeration, PF, `@kern`, `@transform`, translators, VI and wake-sleep) has a counterpart, per the parity checklist and the M5 port. Spot-checked against the public vars of every `gen.inference.*` namespace.

---

## 2. Ecosystem packages

### E1. GenPseudoMarginal ("GenSMC"): AIS and SIR kernels
- **Gen.jl:** https://github.com/probcomp/GenPseudoMarginal.jl (from its README, not read in code):
  - `ais(model, observations, args_seq, argdiffs, mh_kernel)`, which returns a log-ML estimate;
  - `ais_gf`/`AISGF`, AIS as a generative function whose intermediate choices are non-addressable randomness;
  - `selection_sir_gf`, SIR as a GF, for example inside an MH proposal;
  - `selection_sir_mh`, `selection_ais_mh` and `sir_pgibbs`.
- **gen.clj:** partial.
  - Present: conditional SMC and particle Gibbs (`src/gen/inference/particle_mcmc.clj`), and PMMH as MH on `gen.custom/estimated`.
  - Absent: AIS anywhere (no match for anneal, temper or ais in `src/`), and SIR as a GF.
  - `genjl-parity.md:146` says: "Not yet: ancestor sampling, the SIR and AIS kernels."
- **Deliberate or tracked?** Parity doc priority "later". Ancestor sampling is genclj-6wwn. No bean for AIS or SIR.
- **Size and layer:** M, L5.
- **Recommendation: port.** It composes from existing parts:
  - AIS is `pf/step` over a temperature argument with `pf/rejuvenate` and no resampling;
  - AIS-GF and SIR-GF are `gen.custom/estimated` around those estimators.
  - Do it with E2.

### E2. GenSP: inference algorithms as distributions with estimated densities
- **Gen.jl:** https://github.com/probcomp/GenSP.jl (Lew et al., PLDI 2023; the README and the `src/` file list):
  - `SPDistribution` (`random_weighted`, `estimate_logpdf`);
  - `Target`;
  - `Marginal(gen_fn, inf_alg, addr)`;
  - `ChoiceMapDistribution(gen_fn, selection, inf_alg)`, which marginalizes auxiliary choices;
  - importance, custom and composable SMC, MCMC and enumeration algorithms that are themselves SP distributions (`src/inference/*.jl`).
- **gen.clj:** partial. `gen.custom/estimated` (`src/gen/custom.clj:379`, with `:estimate` and `:simulate`) is the SPDistribution analogue. There is no ready-made marginalizing wrapper, so marginalizing a GF's auxiliary choices by IS or SMC must be written by hand each time.
- **Deliberate or tracked?** Not in the parity doc. No bean.
- **Size and layer:** M, L4/L5.
- **Recommendation: port.** A `marginal` wrapper: the value is a GF's selected choices, and the rest are marginalized by an IS or SMC estimator with proper weights, on top of `estimated`. This also gives E1's SIR-GF. The weighting conventions are the thesis §4.5's, which gen.clj's encapsulated-randomness laws already test.

### E3. GenVariableElimination
- **Gen.jl:** https://github.com/probcomp/GenVariableElimination.jl (README):
  - `compile_trace_to_factor_graph(trace, latents, observations)`, where a user declares DML structure, and `factor_graph_analysis` finds SML+Unfold structure automatically;
  - `variable_elimination(fg, order)`;
  - backward samplers, which are exact Gibbs proposals that always accept inside `mh` (FFBS for HMM chains).
- **gen.clj:** partial and different.
  - `with-exact-proposal` enumerates the joint of the chosen finite choices, with `:max-assignments` 1e5 (`src/gen/inference/enumerative.clj:440-463`), so its cost is exponential in their number.
  - `gen.state-space/hmm` handles chains.
  - There is no factorized elimination.
- **Deliberate or tracked?** Parity doc priority "later" (`genjl-parity.md:147`). No bean.
- **Size and layer:** L, L5.
- **Recommendation: ask the owner.** The static IR (`gen.dynamic.static/ir`) already holds the dependency graph that VE needs, so automatic factor graphs are feasible for static bodies. It pays off only for many discrete latents that are not chain-shaped.

### E4. GenDirectionalStats
- **Gen.jl:** https://github.com/probcomp/GenDirectionalStats.jl (README):
  - types Rot2, UnitVector3 and Rot3 (a unit quaternion);
  - `uniform_rot2`, `von_mises_rot2`, `uniform_3d_direction`, `vmf_3d_direction`, `uniform_rot3`, `vmf_rot3` and `uniform_vmf_rot3`;
  - SO(3) MH moves: `uniform_angle_fixed_axis_mh`, `small_angle_random_axis`, `small_angle_fixed_axis_mh` and `flip_around_fixed_axis_mh`.
- **gen.clj:** the circle only: `von-mises`, `wrapped-cauchy` and `wrapped-normal` (`src/gen/dist.clj:814-884`). Nothing on S² or SO(3) (no match for vmf, sphere, quaternion or rot3).
- **Deliberate or tracked?** Parity doc priority "later" (`:148`). No bean.
- **Size and layer:** S for vMF on S²; M for SO(3), which needs a value convention for quaternions and the moves as involutions. Layers L3 and L5.
- **Recommendation: later, or ask the owner.** Needed only for 3D pose and perception models.

### E5. GenFluxOptimizers (Flux's optimizers for Gen parameters)
- **Gen.jl:** https://github.com/probcomp/GenFluxOptimizers.jl. `FluxOptimConf(Flux.Optimise.X, args)` brings any Flux optimizer (RMSProp, ADAGrad, Momentum, Nesterov, AdamW, …) in as an update configuration.
- **gen.clj:** absent beyond `sgd`, `gradient-descent` and `adam` (`src/gen/learning.clj:52-66`).
- **Deliberate or tracked?** Parity doc priority "later" (`:150`). No bean.
- **Size and layer:** S once C1 is done, L5.
- **Recommendation: port.** Add momentum/Nesterov, RMSProp, AdaGrad, AdamW and gradient-norm clipping as pure `init`/`step` definitions. No dependency (rule 8).

### E6. GenTF / GenPyTorch / GenFlux (neural networks as generative functions)
- **Gen.jl:** GenTF, GenPyTorch.jl, and GenFlux.jl (`@genflux` wraps a Flux `Chain` as a GF, with `ParamUpdate` and `accumulate_param_gradients!`; README).
- **gen.clj:** partial. gen.array has `matmul`, `sigmoid` and `softmax` (`src/gen/array.clj:909, 368, 702`), so MLPs work today as parameters in a store, as in Gen.jl's data-driven-proposals tutorial. There are no layers and no convolutions.
- **Deliberate or tracked?** genclj-koxn (deferred-port, draft: "Amortized inference and neural-network layers").
- **Size and layer:** M–L, L4.
- **Recommendation: keep as tracked.** A small layer library over gen.array and `gen.custom/deterministic` is the JVM-native analogue. Skip foreign frameworks.

### E7. GenDistributions (Distributions.jl families)
- **Gen.jl:** https://github.com/probcomp/GenDistributions.jl (README). `DistributionsBacked` takes vector-valued families (its Dirichlet example), with Zygote gradients per parameter.
- **gen.clj:** present but weaker. `gen.dist.host/family` and `gen.dist.commons-math` are univariate only, and differentiating through them is an error (`src/gen/dist/host.clj:3, 22-25`). Many of Distributions.jl's families are native in gen.dist instead.
- **Deliberate or tracked?** Recorded in the parity doc (`:151`). No bean.
- **Size and layer:** M, L3.
- **Recommendation: skip unless needed.** Commons Math has few multivariate families, and native families cover the common ones.

### E8. GenParticleFilters: stratified `pf_update!` and `check=false`
- **Gen.jl:** https://probcomp.github.io/GenParticleFilters.jl/stable/update/.
  - `pf_update!(state, args, argdiffs, obs, strata, [fwd, fwd_args, bwd, bwd_args, transform]; layout=:interleaved)`, and the same with a translator.
  - `check=false` lets a custom proposal overwrite earlier observations.
- **gen.clj:** partial.
  - `:strata` and `:layout` exist only at `initialize` (`src/gen/inference/particle_filter.clj:83-105`).
  - `step` (`:149-185`) has no strata.
  - Without `:backward`, a non-empty discard is always an error (`:163, :176`). Core Gen.jl does the same (`src/inference/particle_filter.jl:228`).
- **Deliberate or tracked?** Not listed in the parity row (`genjl-parity.md:143`). No bean.
- **Size and layer:** S, L5.
- **Recommendation: port.** Add `:strata` and `:layout` to `pf/step`, reusing `strata-of`. A `:check? false` option is the owner's call: it weakens a loud error.

### E9. GenSerialization (traces with untraced randomness)
- **Gen.jl:** https://github.com/probcomp/GenSerialization.jl. It serializes "both traced and untraced randomness".
- **gen.clj:** partial. `gen.serialize` refuses a trace with an ω (DESIGN.md:2027).
- **Deliberate or tracked?** genclj-t8uj (deferred-port, draft).
- **Size and layer:** S–M, L2.
- **Recommendation: keep as tracked.**

### E10. GenViz (visualizing traces during inference)
- **Gen.jl:** https://github.com/probcomp/GenViz. A `VizServer`, trace renderers (HTML/JS), `putTrace!`, and live browser and notebook views.
- **gen.clj:** absent. No viewer, and no Portal, Clerk or Vega in `deps.edn` or the code.
- **Deliberate or tracked?** No bean.
- **Size and layer:** S–M, in a tooling alias or `ext/`, outside the core (rule 8).
- **Recommendation: ask the owner.** Portal or Clerk viewers for choice maps, traces and particle-filter states would be the idiomatic Clojure analogue, and they pair with W4.

### E11. GenPOMDPs and Gen2DAgentMotion
- **Gen.jl:** https://github.com/probcomp/GenPOMDPs.jl:
  - exports `GenPOMDP`, `ControlledTrajectoryModel`, `Controller` and `RolloutModel`;
  - `pf`, `bootstrap_pf`, `pf_observable` and `undiscounted_utility`;
  - accessors for state, observation and action sequences.

  Gen2DAgentMotion.jl adds 2-D path-planning and agent-motion models.
- **gen.clj:** absent.
- **Deliberate or tracked?** genclj-nt3u (deferred-port: agents, MDP/POMDP, inverse planning), with M7 (genclj-tw4p).
- **Size and layer:** M–L, a research vertical outside the core.
- **Recommendation: keep as tracked.**

### E12. Probcomp's Clojure GenSQL (adjacent, but Clojure)
- **Gen.jl:** https://github.com/probcomp/GenSQL.Gen.clj.adapter. It exposes Gen.clj models to GenSQL.query as GPMs: `simulate` and `logpdf` by importance sampling, with the queried variables declared in metadata (README).
- **gen.clj:** absent.
- **Deliberate or tracked?** No bean.
- **Size and layer:** M, an `ext/` module.
- **Recommendation: ask the owner.** gen.clj is the natural engine for it: batched IS and exact conditionals.

### E13. Skip list (checked)
- **GenRedner.** A Python renderer.
- **Genify.jl.** It converts plain Julia that calls `rand`. That is n/a: gen.clj's randomness is explicit keys (rule 2).
- **GenArrow.jl.** Arrow files for traces. EDN serialization exists, and rule 8 rules out the dependency.
- **GenGPT3.jl.** gen.llm covers this with local engines. Remote HTTP engines are "remote / approximate only" (`docs/research/llm-backends.md:34`).
- **GenExperimental.jl.** Superseded since 2020.
- **GenWorldModels.jl** (`georgematheos/`, on a Gen fork, "unreleased research software"). Open-universe, BLOG-like models with automatic MCMC kernels. Ask the owner only if open-universe models are in scope.
- **genlm-control (LLM side).** AWRS, its adaptive weighted rejection sampler, has no gen.clj counterpart: nothing matches in `src/gen/llm/`. JSON-schema potentials are genclj-w120. AWRS belongs to M3/M6 scope, so ask there.

---

## 3. Present, but notably weaker than Gen.jl

### W1. Scalar speed per traced choice
- **Measured** (`docs/benchmarks.md:186-247`, genclj-h6uu), Gen.jl's dynamic DSL against gen.clj:
  - building a trace: 6.4–7.4× faster (simulate 2.75 against 19.15 µs; 7.6–9.3× for each added choice);
  - choice gradients: 5.7× faster;
  - updates: 2.3–2.8× faster.
- gen.clj's batched paths win from about 1,000 particles.
- **Tracked:** genclj-0jeu (handler cost), genclj-4n51 (health), genclj-pxoi (cheaper tape AD), genclj-dqiw (mvnormal factored once).
- **Size and layer:** L, L0–L2.
- **Recommendation: keep as tracked.** This is the largest qualitative gap left.

### W2. The static language's generated code
- **Gen.jl:** it generates specialized Julia code per GFI method, with typed traces (`src/static_ir/generate.jl`, `update.jl`, `backprop.jl`).
- **gen.clj:** only the score is compiled (genclj-rj78). simulate, generate, update and regenerate interpret the graph.
- **Tracked:** genclj-f5fy (draft, waiting on idle re-measurement).
- **Unmeasured:** Gen.jl's *static* language was never benchmarked against gen.clj's. `bench/genjl/genjl_bench.jl` exists, so adding `@gen (static)` rows would size this gap. That is a Julia run, so it is the owner's go.
- **Size and layer:** M–L, L2.
- **Recommendation: measure, then decide f5fy.**

### W3. Documentation
Gen.jl ships 9 tutorials (`docs/src/tutorials/`), 5 how-tos (`docs/src/how_to/`), a reference site, and the gen-quickstart notebooks and GenExamples.jl. gen.clj has:
- **No tutorials or examples.** Nothing in `docs/`, and no `examples/`.
- **No Gen.jl → gen.clj migration note.** DESIGN.md:457 asks for one, for the 0-based indices; it would also cover loud lookups, `trace`/`splice` against `~`, keys, stores against `@param`, and parameterizations.
- **A stale README.** `README.md:12-20` still says "most of M4" and that M3 "comes next".
- **No generated API reference.**

No bean. Size S–M, docs only. **Recommendation: port.** Highest value for adoption and cheap. Start with the migration note and two or three of Gen.jl's tutorials as runnable namespaces (getting started, MCMC/MAP, SMC).

### W4. Printing choice maps and traces

> **Correction (2026-10-24, checked in the code):** choice maps do print, as plain maps (`src/gen/choicemap.clj:423-429`: `print-method` on `Node` and `Value`). Only traces lack a print method. The bean is genclj-qvnu.
- **Gen.jl:** it prints a choice map as a tree (`src/choice_map.jl:104-142`) and the static IR as text (`src/static_ir/print_ir.jl:1`).
- **gen.clj:** `Value`, `Node` and `Trace` are bare defrecords with no `print-method` (`src/gen/choicemap.clj:29, 36`; `src/gen/trace.clj:30`). A choice map therefore prints as nested `#gen.choicemap.Node{:m {… #gen.choicemap.Value{:v …}}}` records, and a trace prints its generative function, args and choices. This is inferred from the code, not run. Generative functions do have short print-methods (`src/gen/dynamic.clj:876`, `src/gen/combinators.clj:997-1007`).
- No bean. Size S, L1.
- **Recommendation: port.** Add a compact `print-method`: a tree, or an `#gen/choicemap {…}` that reads like `cm/to-map`, as `gen.serialize`'s tags do. A trace printer should show its maker, args, score and choices.

### W5. Progress hooks and results for learning and VI
- **Gen.jl:** `black_box_vi!` and `black_box_vimco!` take `callback` and `verbose`, and return `(elbo, var_traces, history, model_traces)` (`src/inference/variational.jl:127-128, 169`). `train!` takes a callback (C2).
- **gen.clj:** VI returns `{:store :history}` (`src/gen/inference/vi.clj:18, 208`): no final q traces and no callback. Users can loop on `vi/elbo-gradient` or `learning/step` themselves.
- No bean. Size S, L5.
- **Recommendation: port with C2.** Add `:callback`, and return q's last traces. Low priority.

### W6. Host-library distributions
E7: univariate only and not differentiable, where GenDistributions takes vector values and Zygote gradients. **Recommendation: skip**, as in E7.

---

## Most worth doing (ranked)

1. **Documentation (W3).** A Gen.jl → gen.clj migration note (DESIGN.md:457 already asks for one), an up-to-date README, and two or three of Gen.jl's tutorials as runnable namespaces. Size S–M, and nothing else here does more for a Gen.jl user.
2. **The optimizer extension point and more optimizers (C1 + E5).** S. Opens `gen.learning` to users and covers GenFluxOptimizers with no dependency.
3. **Printing choice maps and traces (W4).** S. Every REPL session would benefit.
4. **HMC, MALA and NUTS with a dense metric (C3).** S for the kernels, M with `dense_e` adaptation. Gen.jl master has it, and Stan's sampler has a dense-metric mode (`dense_e`).
5. **AIS and inference-as-estimator wrappers (E1 + E2).** M. Gives GenPseudoMarginal's `ais`/`ais_gf`/SIR-GF and GenSP's `Marginal`/`ChoiceMapDistribution`, built on `gen.custom/estimated` and the particle filter.
6. **`train!` parity (C2 + W5).** S. Epoch datasets, a held-out weight as the history, callbacks. The owner picks the history's meaning.
7. **Stratified `pf/step` (E8).** S. Closes the last GenParticleFilters method.
8. **Measure Gen.jl's static language, then decide f5fy (W2).** Needs the owner's go for a Julia run.
9. **Ask the owner:** tilde-style naming (C4), visualization viewers (E10), GenSQL GPM adapter (E12), variable elimination over static graphs (E3), SO(3)/S² distributions (E4).
