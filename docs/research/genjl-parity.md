# Gen.jl parity checklist and GenMLX mapping

*Research notes, 2026-10-18. The baseline is Gen.jl **0.4.8**, the latest
release: 0.4.8 added enumerative inference, 0.4.7 added `ProductDistribution`,
and 0.4.6 removed the need for `@load_generated_functions`
(https://github.com/probcomp/Gen.jl/releases).*

*Sources:*
- *Gen.jl reference pages under https://www.gen.dev/docs/stable/ref/:
  `core/{gfi,choice_maps,selections,change_hints}`,
  `modeling/{dml,sml,distributions,combinators,custom_gen_fns}`,
  `inference/{enumerative,importance,mcmc,pf,trace_translators,parameter_optimization,map,vi,wake_sleep}`.*
- *The Gen.jl `src/` tree.*
- *The ecosystem pages: https://www.gen.dev/ecosystem/ and
  https://probcomp.github.io/GenParticleFilters.jl/stable/.*
- *Thesis: `Marco Cusumano-Towner 2020 PhD thesis.pdf` in this repo. Printed page
  numbers equal PDF page numbers.*

*Prior art: `probcomp/Gen.clj` (https://github.com/probcomp/Gen.clj) is an
experimental Apache-2.0 Clojure port by Zane Shelby that covers a subset of Gen.
GenMLX adapted 356 of its assertions in `test/genmlx/gen_clj_compat_test.cljs`.
**The name "gen.clj" collides with it.***

*Status: **Y**es / **P**artial / **N**o. Paths are relative to
`~/code/genmlx/src/genmlx/`. Priority for gen.clj:*
- ***core**: needed for parity, M1 or M4;*
- ***soon**;*
- ***later**.*

## 1. Parity checklist

### 1a. The Generative Function Interface (Gen.jl core)
| Capability | GenMLX status + file | gen.clj |
|---|---|---|
| `simulate(gf,args)` → trace | Y: `protocols.cljs` IGenerativeFunction; `dynamic.cljs` DynamicGF | core |
| `generate(gf,args[,constraints])` → (trace, w) | Y: IGenerate. **Caveat:** in the L3 conjugate case it returns the posterior-mean latent with the marginal weight, tagged `:marginal` | core |
| `propose` → (choices, w, retval) | Y: IPropose | core |
| `assess` → (w, retval) | Y: IAssess | core |
| `update(trace,args,argdiffs,constraints)` → (trace, w, **retdiff**, discard) | P: split into `update` (constraints only), `update-with-args` and `update-with-diffs`; **no retdiff is returned** | core |
| `regenerate(trace,args,argdiffs,selection)` → (trace, w, retdiff) | P: `regenerate [gf trace sel]` has no args/argdiffs and no retdiff; `edit.cljs` CompositeEdit can emulate it | core |
| `project(trace, selection)` | Y: IProject | core |
| `get_args/get_retval/get_choices/get_score/get_gen_fn`, `trace[addr]`, `trace[]` | Y: `trace.cljs` | core |
| `has_argument_grads` | P: the protocol exists, but DynamicGF returns nil | core. *gen.clj (M1):* moot under full AD. `choice-gradients` differentiates the arguments named by `:argnums` (none by default, as Gen.jl differentiates only `(grad)` arguments), each a tree of floats; the others' gradients are nil |
| `accepts_output_grad` | N | core. *gen.clj (M1):* every generative function takes a retgrad (`choice-gradients`, `param-gradients`) |
| `choice_gradients(trace, sel, retgrad)` → (arg_grads, values, grads) | P: `gradients.cljs` `choice-gradients` takes an address vector; no retgrad, no arg grads | core |
| `accumulate_param_gradients!(trace, retgrad, scale)` | P: functional gradients via `mx/grad` (`learning.cljs make-param-loss-fn`); no stateful accumulator | core |
| `get_params(gf)` | P: a name-keyed store in `learning.cljs`; no per-GF registry | core |
| Non-addressable randomness r: `get_score = log p(r,t)/q(r;x,t)` | Y: `encapsulated.cljs`, `Trace.omega` | soon. *gen.clj (M4, 2026-10-20):* `(untraced dist)` and `(untraced-key)` in gen bodies, ω a key the trace keeps, kept unless the change hints say something changed (DESIGN.md §9); `gen.custom/estimated` for estimators; `assess` takes a key. Pseudo-marginal MH and PMMH are MH on such a model |

### 1b. Change hints and incremental computation (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `NoChange`, `UnknownChange` | Y: `diff.cljs` | core |
| `VectorDiff` | Y: `vector-diff` with a `:changed` set | core |
| `IntDiff`/`SetDiff`/`DictDiff`, and `Diffed{V,DV}` propagating through plain code | N | soon (needed for the static DSL). *gen.clj (M4, 2026-10-20):* `gen.diff/int-diff`, `set-diff`, `dict-diff`, `consistent?`, and `propagate`, an open multimethod per function var, applied in static bodies to one-call bindings; a splice passes its callee precise argdiffs (DESIGN.md §9) |
| Retdiffs, consumed by Unfold/Map/Recurse to skip work | N | core |

### 1c. Dynamic modeling language (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `@gen`, `@trace(dist, addr)` / `~` | Y: `gen.cljc`, `(trace :a dist)` | core |
| Pair addresses `:a => :b` | Y: vector paths `[:a :b]` | core |
| Nested call `@trace(gf(args), :ns)` | Y: `(splice :ns gf args)` | core |
| Shared-namespace call `@trace(gf(args))` / `{*} ~` | N: splice always nests | core |
| Untraced call `gf(args)` | Y: `dyn/call` | core. *gen.clj:* `(gf a b)`, IFn `invoke` on dynamic generative functions, the combinators and the deterministic ones: a simulated run's return value (`dyn/run-call` is the internal dispatch of traced calls, not this) |
| `@param` with `init_param!`, `get_param`, `set_param!`, `get_param_grad`, `set_param_grad!`, `zero_param_grad!` | P: `(param :name default)` plus a functional `::param-store`; no gradient accumulators | core |
| `(grad)` arg annotations, `@gen (grad)` | n/a (MLX AD covers all args). **gen.clj needs a JVM AD substrate** | core |
| Address discipline (no prefix collisions, fixed support per address) | Y: laws `:address-uniqueness`, `:no-mutation`, `:no-external-randomness`; `verify.cljs` | core |

### 1d. Static modeling language (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `@gen (static)`: a DAG IR where update skips unchanged nodes; cached Julia calls, `nojuliacache`, `draw_graph`, typed traces | P, different mechanism: implicit schema extraction (`schema.cljs`), compiled paths (L1), `dep_graph.cljs`. No explicit static DSL and no IR rendering | soon. *gen.clj (M4, 2026-10-20):* `(gen ^:static [args] (let [...] ret))` (DESIGN.md §9): the graph as data and DOT (`gen.dynamic.static/ir`, `dot`, Gen.jl's `draw_graph`); update and regenerate skip nodes whose inputs did not change; a law requires the same results as without the flag. Julia's node cache is the trace's node values, always kept (`nojuliacache` has no counterpart: `static_ir.jl`'s cached-node rerun counts agree, `docs/research/genjl-test-suite.md`); typed traces have none (the JVM's JIT compiles the node fns) |
| `@load_generated_functions` | n/a (unneeded since 0.4.6) | skip |

### 1e. Distributions (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| The 20 built-ins: `bernoulli, beta, beta_uniform, binom, categorical, cauchy, dirichlet, exponential, gamma, geometric, inv_gamma, laplace, mvnormal, neg_binom, normal, piecewise_uniform, poisson, uniform, uniform_discrete, broadcasted_normal` | Y: `dist.cljs`. Parameterizations differ: Gen `gamma(shape, scale)` vs GenMLX `gamma-dist(shape, rate)`; Gen `categorical(probs)` vs GenMLX logits; Gen's `bernoulli` returns Bool | core. *gen.clj (M4, 2026-10-20):* `mvnormal` rejects a covariance singular in floating point, at every precision (DESIGN §9: trace(H⁻¹) ≥ 1/ε, the measured bound), where Gen.jl rejects only a failed Cholesky; so in float32 a Gaussian-process kernel needs a jitter at float32's scale (1e-3 passes for 1,000 points, 1e-5 does not), and the error suggests one (`:suggested-jitter`) |
| Distribution API: `random, logpdf, logpdf_grad, has_output_grad, has_argument_grads, is_discrete` | P: `defdist` plus multimethods; gradients via AD; `support` | core |
| `HomogeneousMixture`, `HeterogeneousMixture` | P: `dist/core.cljs mixture` takes instantiated components and log-weights | core. *gen.clj (M4, 2026-10-20):* `homogeneous-mixture` takes Gen.jl's dims, so multivariate families mix (`(homogeneous-mixture mvnormal [1 2])`: means [K d], covariances [K d d]); both mixtures mix each component's density of the whole value, with one pick per element of it when sampling (b9eb9d2) |
| `ProductDistribution` | Y: `product` | core |
| `@dist` DSL (CONST→ARG→RND typing, exactly one random draw, pushforward) | N: `map->dist` is manual | soon. *gen.clj (M4, 2026-10-19):* `gen.dist.dsl/dist`, exactly one draw; discrete draws sum the mass mapped to a value (any function, repeated labels), continuous ones invert +, −, ×, ÷, negate, exp and log with their Jacobians; closed forms, normalization and laws checked |

### 1f. Custom generative functions (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `CustomGradientGF` (`apply`, `gradient`) | Y: `custom_gradient.cljs` (scalar output only) | soon. *gen.clj (M4, 2026-10-20):* `(gen.custom/deterministic {:apply f :gradient g})`, and `:gradient-with-state` for CustomDetermGF; array outputs, on `array/custom-vjp` (tape and MLX native) |
| `CustomDetermGF` / `CustomUpdateGF` (`apply_with_state`, `update_with_state` → retdiff, `num_args`) | N (closest is `nn.cljs nn->gen-fn`) | soon. *gen.clj (M4, 2026-10-19):* `gen.custom/deterministic` with `:apply-with-state` and `:update-with-state` (laws alone and spliced; incremental updates checked); custom gradients with `:gradient-with-state` (genclj-8cp8) |
| Hand-written GFI implementations | Y: implement the protocols | core |

### 1g. Combinators (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `Map`, VectorDiff-driven (revisits only changed or constrained elements) | Y: `combinators.cljs` MapCombinator. Keys are **0-based**; Gen.jl's are 1-based | core |
| `Unfold(kernel)` with args `(N, init, extras...)`; retdiff-driven skipping; changing N | Y: UnfoldCombinator, `unfold-extend`, structure-changing `update-with-args` | core |
| `Switch(gfs...)` / `Switch(dict, gfs...)`; a branch change gives a fresh subtrace plus a discard | Y (integer index) / P (no dict-keyed constructor) | core |
| `Recurse(production, aggregation, max_branch, U,V,W)` (PCFG trees) | P: `recurse` is a fixed-point "maker" combinator, not the production/reduction tree; `update-with-args` throws | core. *gen.clj (M4, 2026-10-20):* `(comb/recurse production aggregation max-branch)`, 0-based nodes, every GFI operation incremental (a node is revisited when its input, its value, its children or its target may have changed); the general laws, alone and spliced |
| `call_at(gf, K)` / `choice_at` | N | soon. *gen.clj (M4, 2026-10-19):* `comb/call-at` and `comb/choice-at`, the address as the last argument as in Gen.jl; dynamic generative functions over `splice` and `trace`, so their GFI is the handler's (laws in `combinators-test`) |
| Internal-proposal override (thesis §4.4, Alg. 16, p. 152; not found in the 0.4.8 modeling library) | P: via `dispatch/with-dispatch` | soon. *gen.clj (M4, 2026-10-19):* `gen.inference.proposal/with-internal-proposal`: q, called as `(q constraints & args)`, proposes some open choices and forward sampling the rest; project replays q against a trace |

### 1h. Choice maps, selections, traces (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| Dynamic/Static/Empty choicemap; `choicemap(...)`, `has_value`, `get_value`, `has_submap`, `get_submap`, `get_values_shallow`, `get_submaps_shallow` | Y: `choicemap.cljs` | core |
| `set_value!`, `set_submap!`, `merge`, `isempty`, `==` | Y: persistent `set-value`, `set-submap`, `merge-cm` | core |
| `to_array(cm, T)`, `from_array(proto, arr)` | N | core |
| `get_selected(cm, sel)` | N | core |
| `pair`/`unpair`, `nested_view` | N | later. *gen.clj (M4, 2026-10-19):* `cm/pair`, `cm/unpair` (loud when other keys are present); `nested_view` is `cm/to-map` |
| `select`, `selectall`, `complement`, Empty/All/Hierarchical/Dynamic/Static/Complement selections, `push!`, `set_subselection!` | Y: `selection.cljs` | core |

### 1i. Parameters, optimizers, learning (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `ParamUpdate(conf, gf=>params)`, `apply!`, `FixedStepGradientDescent`, `GradientDescent(init, beta)`, `ADAM`, `init_update_state`/`apply_update!` | P: `learning.cljs` sgd/adam/train; `nn/optimizer`; no decaying GD; no per-GF ParamUpdate | core. *gen.clj:* `gen.learning` sgd, gradient-descent, adam, `init`/`step` (M1); several generative functions' stores step together nested in a map, a subset when only it is given (M4, tested) |
| `train!(gf, data_gen, update, epochs, ...)` | P: `learning/train`, `amortized.cljs train-proposal` | core |
| `lecture!`, `lecture_batched!` (sleep phase); wake phase via `train!` | Y: `learning.cljs` wake-sleep | soon. *gen.clj (M4, 2026-10-19):* `learning/lecture`, `lecture-batched`: the proposal scored on the latent choices of the model's own traces (found by replaying it), with its parameter gradients; trains a proposal to the conjugate posterior |

### 1j. Inference library (Gen.jl core)
| Capability | GenMLX | gen.clj |
|---|---|---|
| `importance_sampling` / `importance_resampling` with the internal proposal | Y: `inference/importance.cljs` | core |
| …with a custom proposal GF | **N** in importance.cljs (only `amortized.cljs` and `smcp3-init`) | core |
| `mh(trace, selection)` | Y: `mcmc.cljs mh`, `kernel.cljs mh-kernel` | core |
| `mh(trace, proposal, args)` | Y: `mh-custom` | core |
| `mh(trace, proposal, args, involution)` / `involutive_mcmc` | Y/P: `involutive-mh` (no Jacobian); `translator.cljs` (AD Jacobian, `reversible-jump-mh`) | core. *gen.clj (M4, 2026-10-20):* `mh/involutive-mh` with `(transform/involution t)`, the Jacobian by AD; a reversible jump (split/merge) keeps the exact posterior over one step. A hand-written involution takes one more argument than Gen.jl's, a key, last: `(fn [trace fwd-choices fwd-retval proposal-args key])`, which its update uses, so encapsulated randomness stays exact (pseudo-marginal) |
| `check=`, `observations=` stationarity checks | N | soon. *gen.clj (M1):* `(k/checked kernel observations)` checks after every application, forward and reversed |
| `mala(trace, sel, tau)` | Y | core |
| `hmc(trace, sel; L, eps)` | Y (with dual-averaging adaptation) | core. *gen.clj (2026-10-23, genclj-tvc0):* hmc and mala take `:inv-metric`, a diagonal metric (the identity by default); beyond Gen.jl, `mcmc/nuts` is Stan's multinomial NUTS, and `gen.inference.nuts` Stan's warmup (dual averaging toward 0.8, the windowed diagonal metric) and sampler, for nuts or hmc |
| `elliptical_slice(trace, addr, mu, cov)` | Y | core |
| `@kern` (`for`/`if`/`let`, auto reversal), `@pkern`, `@rkern`, `reversal` | P: kernel.cljs combinators, `with-reversal`; no macro DSL, no check | core. *gen.clj (M4, 2026-10-19):* `(k/kern [args] stmt …)`, with doseq/for, if/when and let over the arguments: one chain, reversed automatically; `@pkern` is `k/kernel`, `@rkern` `(k/kernel f reverse-f)`, `reversal` `k/reversal`. *2026-10-20 (the owner, after the final law audit):* `(k/kern ^:trace [tr args] stmt …)` reads the trace in its loops, conditions, lets and kernel arguments, as `@kern` does, and checks the invariant of pp. 100–101 after the body of each loop, branch and let, as `@kern` with `check=true` does (gen.clj always checks): a body that changes what its construct read is a loud error, a later statement may change it (`a-kern-that-reads-the-trace`). The thesis's p. 102 kernel in its own order, which loops over `trace[:degree]` before the structure move, keeps the exact posterior forward and reversed (`the-thesis-s-degree-kernel-leaves-the-posterior-invariant`) |
| `map_optimize(trace, sel; ...)` (backtracking line search) | Y: `map-optimize` (Adam/SGD instead) | core |
| `enumerative_inference` + `choice_vol_grid` (including continuous grid volumes) | P: `enumerate.cljs`, `exact.cljs`; discrete only | core. *gen.clj (2026-10-19):* `gen.inference.enumerative`, discrete and continuous (N-d) grids; plus `with-exact-proposal`, the exact conditional as a model's internal proposal for its finite choices (Gibbs regenerate; exact likelihoods inside Unfold when the state does not depend on those choices, checked by `:check-retval?`) |
| `initialize_particle_filter` / `particle_filter_step!` (default or custom proposal), `maybe_resample!`, `log_ml_estimate`, `get_traces`, `get_log_weights`, `sample_unweighted_traces` | P: `smc.cljs` functional whole-sequence API; default proposal only (custom via `smcp3.cljs`); `steppable.cljs` | core |
| Trace transform DSL: `@transform`, `@read`/`@write` with `:discrete`/`:continuous`, `@copy`, `@tcall`, `pair_bijections!`, `is_involution!`, `inverse` | P: translator.cljs read/write/copy-choice, `jacobian-logdet`, `sparse-jacobian-logdet`; no macro DSL, no discrete/continuous tagging, no inverse pairing check | core. *gen.clj (M4, 2026-10-20):* `gen.inference.transform`: `deftransform`/`transform` with `read`, `write`, `copy` (tagged), `tcall`, `pair`, `involutive`, `inverse`; log |det J| by AD over the continuous reads and writes (copies of reads, and reads an update retains, as outputs; copies of unread choices left out) |
| `Deterministic`/`General`/`SimpleExtending`/`Symmetric` TraceTranslator | P: `trace-translator`/`apply-translator`, `coarse-to-fine-smc` | core. *gen.clj (M4, 2026-10-20):* `transform/translator`, `deterministic-translator`, `extending-translator`, `involution` (the symmetric one), with Gen.jl's `check` as `:check?` (and `:prev-observations`) |
| `black_box_vi!` (score-function estimator; variational model is a GF with `@param`) | P: `vi.cljs` (mean-field ADVI, programmable VI over raw log-density + flat params); `adev.cljs` | core. *gen.clj (M4, 2026-10-19):* `(vi/black-box-vi model args observations q q-args store optimizer opts)`, q a generative function over a parameter store; Gen.jl's estimator, or with `:baseline :leave-one-out` (unbiased, and exactly zero at the posterior); `vi/elbo-gradient` is one iteration |
| `black_box_vimco!` | Y: `vimco` | core. *gen.clj (M4, 2026-10-19):* `vi/black-box-vimco`, VIMCO's leave-one-out control variates with the geometric (default) or arithmetic baseline, as Gen.jl's `geometric` keyword; `vi/vimco-gradient` is one iteration |

### 1k. Ecosystem packages
| Package | What it adds | GenMLX | gen.clj |
|---|---|---|---|
| GenParticleFilters | `pf_initialize` (custom proposal, stratified), `pf_update!` (fwd/bwd proposals + transform, translator, stratified), `pf_resample!` (multinomial/residual/stratified, `priority_fn`), `pf_rejuvenate!`/`pf_move_accept!`/`pf_move_reweight!`, `pf_resize!` (multinomial/residual/optimal), `pf_replicate!`/`pf_dereplicate!`/`pf_coalesce!`/`pf_introduce!`, mean/var/proportionmap, ESS, log-ML, substate views; `UpdatingTraceTranslator`/`ExtendingTraceTranslator` | P: systematic/residual/stratified resampling, ESS threshold, MH move-accept rejuvenation, csmc, vsmc. Missing: priority weights, move-reweight, resize, stratified init, statistics helpers | soon. *gen.clj (M4, 2026-10-20):* `pf/initialize` with `:strata` (and `pf/choice-product`), `pf/resample` with `:priority`, `:n` and `:method :optimal`, `pf/resize`, `pf/rejuvenate` (move-accept), `pf/move-reweight` with the kernels of `pf/reweighting` (selection, proposal, backward proposal, involution), `pf/replicate`, `pf/dereplicate` (keepfirst, sample), `pf/coalesce`, `pf/introduce`, `pf/mean`, `pf/variance`, `pf/proportions`. Optimal resampling corrects GenParticleFilters' threshold search, which counts the threshold weight among the drawn and biases the estimate (for W = [0.1 0.2 0.7] to two particles the weights add to 1.2·n). `pf/step` takes `:backward` (pf_update! with a backward proposal), and a translator is a move-reweight kernel, so `(pf/move-reweight state translator)` is pf_update! with one. Substate views: `pf/view` (state[idxs]: its ess, statistics and estimate, the source's so far plus the view's log mean weight) and `(pf/update-view state idxs f & args)`, which runs any operation on those particles alone; a resampling there keeps the view's total weight, with the priority correction kept unbiased |
| GenSMCP3 | `SMCP3Update(fwd, bwd, ...)` with automatic sparse Jacobian correction | P: `smcp3.cljs` via edit ProposalEdit; no auto-Jacobian | soon. *gen.clj (M4, 2026-10-20):* `transform/updating-translator` with `:transform` (GenParticleFilters' UpdatingTraceTranslator, as GenSMCP3 builds it), run on a filter by `pf/move-reweight`; the Jacobian by AD, copies of unread choices left out; without a transform, Del Moral's SMC samplers. Tests: a reparameterized proposal weighs exactly as the optimal one, a backward kernel's exact weight, the estimate unbiased |
| GenTraceKernelDSL | `@kernel` mixing sampling and deterministic code; MH/SMC kernels returning (forward, reverse) choicemaps | N (closest: `edit.cljs`) | soon. *gen.clj (M4, 2026-10-20):* `gen.inference.trace-kernel`: a kernel is a `gen` function of the trace returning `[update backward]`; `tk/mh`, `tk/mh-kernel`, `tk/smcp3` (a translator for `pf/move-reweight`), log \|det J\| by AD over the trace's and the kernel's continuous choices, trimmed as GenTraceKernelDSL, reads the update keeps counted as unchanged outputs |
| GenPseudoMarginal / "GenSMC" | `selection_sir_gf`, `selection_sir_mh`, `sir_pgibbs`, `ais`, `ais_gf`, `selection_ais_mh` | P: `encapsulated.cljs` pseudo-marginal-mh, `pmcmc.cljs`; AIS only in laws | later |
| GenVariableElimination | Factor graphs from traces, variable elimination, exact backward samplers | P, different: `exact.cljs` tensor enumeration, `hmm_forward.cljs` | later |
| GenDirectionalStats | Distributions on orientations and rotations (SO(3)), involutive kernels | P: von-mises, wrapped-cauchy, wrapped-normal; no SO(3) | later. *gen.clj (2026-10-23, genclj-z2gh):* `von-mises`, `wrapped-cauchy` and `wrapped-normal` in `gen.dist`, values in [-π, π); no SO(3), no involutive kernels |
| GenTF / GenPyTorch | TF/PyTorch graphs as GFs | analog: `nn.cljs nn->gen-fn` | later |
| GenFluxOptimizers | Flux optimizers for Gen params | analog: `nn/optimizer` | later |
| GenDistributions | Distributions.jl wrapped via `DistributionsBacked` | n/a | soon (Commons Math / fastmath). *gen.clj (M4, 2026-10-20):* `gen.dist.host/family` (a family from a host object's quantile and log density; draws are quantiles of the key's uniforms) and `gen.dist.commons-math` (alias `:commons-math`): any Commons Math 3 distribution by `real`/`integer`, and hypergeometric and Zipf. Univariate; no gradients through a host library (loud), and batched runs fall back. Distributions.jl's continuous families beyond Gen.jl's are native in `gen.dist` with its parameters (chi-squared, f-dist, gumbel, levy, log-normal, logistic, nakagami, pareto, student-t, triangular, weibull; since genclj-z2gh, 2026-10-23, also truncated-normal, von-mises, wishart and inv-wishart, for `truncated(Normal)`, `VonMises`, `Wishart` and `InverseWishart`): batched, on every backend, differentiable |
| Gen2DAgentMotion, GenRedner | Agent-motion and renderer domain libraries | agents/* is an analog of the first | later / skip |

## 2. What GenMLX has that Gen.jl does not

- **Vectorized/batched GFI.**
  - `vsimulate`/`vgenerate`/`vupdate`/`vregenerate`/`vproject`, `VectorizedTrace`,
    `IBatchedSplice`, `vmap-gf`/`repeat-gf`, `vectorized-switch`.
  - Vectorized MH, MALA, HMC, MAP, IS, SMC, SMCP3 and ADEV.
- **Compilation ladder L0–L4.**
  - Schema extraction and compiled simulate/generate/update/regenerate.
  - Fused whole-chain MCMC and captured-graph replay; `vgenerate-compiled`.
  - Compiled SMC, VI and Adam.
  - `fit` / `method_selection`.
- **Automatic analytical elimination (L3).**
  - Conjugacy detection across 7 families, affine analysis, and joint
    linear-Gaussian elimination.
  - Rao-Blackwellization.
  - Kalman, EKF, N-d EKF and HMM-forward handlers.
- **A GenJAX-style edit interface:** typed EditRequests that return backward
  requests.
- **Extra combinators:** Scan, Mask, Mix, Contramap/MapRetval/Dimap.
  *gen.clj (round 1, 2026-10-23, genclj-2gma):* `comb/scan` (JAX's scan:
  `(kernel carry x & params)` returns `[carry' y]`, Scan returns `[carry
  ys]`, updates skip the steps whose carry, input and parameters are
  unchanged), `comb/mix` and `comb/mix-logits` (the index a categorical at
  `:index`, the component a Switch's call at `:component`), and
  `comb/dimap`, `comb/contramap`, `comb/map-retval` (the inner's traces).
  No Mask: in a body it is `(when active? (splice …))`, as a generative
  function `(switch {true gf false (gen [& _] v)})`, which runs by
  partition when active? differs across particles and v is a value of gf's
  return value's kind (nil and a number do not stack, and fall back).
- **Extra inference.**
  - NUTS, enumerative Gibbs, PMMH and particle Gibbs, conditional SMC.
    *gen.clj (round 1, 2026-10-23):* NUTS and Stan's warmup
    (`mcmc/nuts`, `gen.inference.nuts`); Gibbs by `with-exact-proposal`.
  - Differentiable resampling and IS.
  - ADEV with REINFORCE and baselines.
  - Programmable VI objectives (IWELBO, PWake, QWake).
  - Amortized neural IS.
  - Fisher/Laplace, steppable/budgeted inference with a cost meter.
  - Diagnostics (bulk/tail ESS, R-hat). *gen.clj (round 1, 2026-10-23):*
    `gen.inference.diagnostics`.
  - Exact enumeration with the full GFI and entropy/mutual-information queries.
- **16 extra distributions:** student-t, log-normal, delta, truncated-normal,
  wishart, inv-wishart, von-mises, wrapped-cauchy, wrapped-normal, gaussian-vec,
  iid, iid-gaussian, weighted, categorical-weights, … *gen.clj (round 1,
  2026-10-23):* all but delta (the owner's decision), the rest already
  covered by gen.dist's families.
- **Verification infrastructure.**
  - `gfi.cljs`: about 85 executable laws.
  - `pef.cljs`: path-equivalence fuzzing.
  - `verify.cljs`: a static validator.
  - Malli dev-mode validation and score-type tags.
- **Architecture:** pure handler transitions plus a data-driven dispatcher stack;
  functional PRNG key threading.
- **Infrastructure and domains.**
  - JSON serialization; sqlite persistence.
  - LLMs as GFs: grammar-constrained decoding, token-SMC, byte marginalization.
  - Agents (MDP/POMDP, inverse planning).
  - Structural inference over programs, metareasoning control, NN layers.

## 3. GFI invariants for gen.clj's property tests

Most laws are the thesis's (Cusumano-Towner 2020; printed page = PDF page);
laws 11, 12 and 18 are Gen.jl's, not the thesis's. The list was audited
against the thesis on 2026-10-18 (`docs/research/law-audit.md`); the
corrections are folded in below, and the coverage matrix at the end of this
section maps each law to its tests. Notation: t = (P, x, τ); σ = constraints;
q = internal proposal.

1. **Structured density / addresses** (Def 2.1.3, p. 49).
   - Distinct traces in the support disagree on some shared address. In
     practice: no address is a prefix of another, and leaf addresses are unique.
   - Choices are sufficient: the same (P, x, τ) always gives the same retval and
     score.
   - The dynamic language requires unique addresses (DML restriction 2, p. 63).
2. **Score = density.** `get_score(simulate(x)) = log p(τ;x) = assess(x,τ).weight`,
   and the retvals agree (pp. 68–69; Def 2.1.16, p. 56).
3. **Generate.**
   - Empty constraints: weight 0, and the trace is distributed as `simulate`.
   - Full constraints: the trace is deterministic and the weight equals the score
     (extended generate, p. 143; the generate of p. 68 has no weight).
   - The thesis ignores constraints the run does not visit (p. 69; σ|_B,
     p. 143). gen.clj throws instead, as Gen.jl does: a deliberate departure.
   - General case: log w = log p(τ;x) − log q(υ;x,σ) (§4.1.1, p. 143).
   - Forward-sampling q: the weight is the sum of the constrained addresses'
     log-densities as they are visited (p. 145).
   - Statistical check: E_q[w] = p̄(σ;x) against a closed-form marginal likelihood.
4. **Support condition.** q(υ;x,σ) > 0 ⇔ p(υ|σ;x) > 0 (Def 4.1.1, p. 142).
   Generate weights are never NaN, and −∞ only off-support.
5. **Update weight and discard** (Def 2.3.1, p. 69; p. 71; Gen docs).
   - log w = log p(τ';x') − log q(τ'_fresh; x', τ+σ) − log p(τ;x); fresh terms
     cancel. The q(fresh) term is Gen.jl's extension: in the thesis an
     unconstrained new address makes update undefined (Def 2.3.1; Alg 17,
     p. 168, raises an error).
   - The discard contains the overwritten old values **and** the old addresses
     that are no longer visited.
   - An identity update (x'=x, σ={}) has weight 0.
6. **Update is reversible** (Prop. 2.3.1, p. 70). update(t', x, ·, discard) gives
   back t, including for structure-changing updates and changed arguments. Its
   weight is −w only without fresh choices; in general it is −w − log q(fresh).
   The reverse discard is σ (σ ⊕ the fresh choices, with Gen.jl's extension).
7. **Compositionality** (Prop. 2.3.2, p. 73). For sequencing or splicing,
   log w₃ = log w₁ + log w₂ and discards merge. Combinator updates match the
   equivalent inlined loop in the dynamic modeling language.
8. **Change hints are sound** (Def 2.3.2, pp. 70–71; §5.3, pp. 176–177).
   - Results under `UnknownChange` equal those under precise hints.
   - Retdiffs over-approximate: f(x',τ') ∈ f(x,τ) ⊙ δY.
   - Map and Unfold skip only elements whose hint says unchanged.
9. **Regenerate** (Eq. 4.1, p. 144).
   - log w = log [p(τ';x') q(υ';x,σ') / (p(τ;x) q(υ;x',σ))].
   - Unselected addresses keep their values.
   - With the same args, selecting nothing gives weight 0 and an unchanged trace,
     and selecting everything gives weight 0 (this needs q(·;x,{}) = p(·;x), as
     with forward sampling or the optimal proposal).
   - Requires a supportive density (Def 2.1.7, p. 52); a fixed support per
     address is sufficient, not equivalent (pp. 148, 169). The forward-sampling
     weight is Alg 18 (p. 170).
10. **Selection MH is stationary** (Alg. 15, p. 148). Test against exact
    enumeration or a conjugate posterior. With the optimal internal proposal,
    regenerate is Gibbs (pp. 145–146). *Errata:* the acceptance probability 0.6
    in the example of pp. 148–149 contradicts Eq 4.1 and Alg 18 (the correct
    value is 1); in Alg 15, "r ≤ log w" means log r ≤ log w.
11. **Project.**
    - The all-selection gives the score; the empty selection gives 0.
    - For fixed-structure forward sampling, project(S) + project(Sᶜ) = score.
    - project(t, A) = log p(τ) − log q(τ|_Aᶜ; x, τ|_A), which equals the weight of
      generate constrained on t|_A when that generate reproduces t (in gen.clj,
      with t's key).
12. **Propose.** The weight equals `assess(choices).weight`, and the choices are
    distributed as `simulate`.
13. **Gradients** (Eqs. 2.10–2.12, p. 72).
    - Argument gradients: ∇ₓ log p + J·v.
    - Choice gradients: ∂log p/∂τ[a] + Σᵢ ∂fᵢ/∂τ[a]·vᵢ, where v is the retgrad.
    - Check against finite differences, including a nonzero retgrad.
    - `accumulate_param_gradients!` scales by its scale factor.
14. **Encapsulated randomness** (Def 4.5.1, p. 155; Eqs. 4.2–4.3, p. 156).
    - E_{q̊}[ξ] = p(τ;x).
    - An identity update does not have weight 0: it resamples ω' (p. 157), so
      log w = log ξ(x,τ,ω')/ξ(x,τ,ω). *gen.clj keeps ω when
      the change hints say nothing changed (owner, 2026-10-20; DESIGN.md
      §9), so its identity update weighs 0; both are exact.*
    - Pseudo-marginal MH targets the exact posterior.
15. **Custom-proposal MH requirements** (Alg. 5 and requirements, p. 92). The
    proposal never samples an observed address, and every move is reversible:
    q(σ';t') > 0. *Erratum:* Alg 5 and Eq 4.15 print exp(log w − s'.logpdf() +
    s.logpdf()), which swaps the forward and backward terms; the correct ratio
    is log w + log q(σ';t') − log q(σ;t), as in Algs 6, 9 and 11 (and
    `gen.inference.mh`).
16. **Trace translators** (Def 3.6.1 and Eq. 3.13, p. 110; Alg. 9, p. 111).
    - w = p₂q₂ / (p₁q₁) · |J_h|.
    - Round trip h(h⁻¹(z)) = z.
    - The sparse Jacobian equals the dense one (§3.6.2, pp. 111–112).
17. **Involutive MCMC** (Def 3.7.1, Eq. 3.15, p. 121; Eq. 3.17, p. 123).
    - h∘h = id.
    - Acceptance: min{1, p(υ')q(σ';υ') / (p(υ)q(σ;υ)) · |J|}.
    - Detailed balance holds on a split/merge example.
18. **Kernel DSL** (Gen.jl; compare the thesis's §3.4.4, pp. 98–100, Prop 3.4.1).
    - The reversal of a composite kernel is the reversed sequence of reversed
      kernels.
    - With `check`, observed choices are unchanged after each kernel.
19. **Internal-proposal override** (Alg. 16, p. 152). R keeps p and changes q.
    Generate under R stays unbiased, and regenerate/MH under R stays stationary.
20. **Estimators.** IS and particle-filter log-ML estimates are unbiased in exp
    space, and the PF incremental weight equals the update weight.

### Coverage matrix

`gen.laws` holds the general and forward-sampling laws as properties over
any models; each feature's suite instantiates them (`deflaws`). Status: ✓
tested, ◐ partly, — not yet (the feature does not exist yet).

| Law | Tests |
|---|---|
| 1 Structured density, unique addresses | ✓ `errors-are-loud` (repeated and nested addresses); `runs-are-pure-functions-of-the-key` |
| 2 Score = density | ✓ `score-is-the-assessed-density`, and trace consistency inside every generate/update/regenerate law |
| 3 Generate | ✓ `generate-with-every-choice-is-deterministic`, `generate-without-constraints-has-weight-zero`, `generate-weight-is-the-constrained-density`, `generated-traces-are-consistent`; for any internal proposal, `generate-weight-is-the-projection` (M4); E[w] = p̄(σ): `the-estimate-is-unbiased-in-exp-space`, `importance-sampling-with-stochastic-structure-is-unbiased`, the thesis's own example (Alg 2, p. 80: `the-thesis-s-infinite-squares-example`, M4, written with GFI operations and through `importance-sampling`); assess on the same key: `assess-is-generate-with-every-choice` (M4) |
| 4 Support condition | ✓ generate weights never NaN (`generated-traces-are-consistent`); for models whose every address has a fixed support, simulated scores and generate weights on their choices are finite (`generate-weights-are-finite-on-a-fixed-support`, `laws/fixed-support`, over the dynamic, combinator, internal-proposal, exact-proposal and static models; law audit gap E, M4) |
| 5 Update weight and discard | ✓ `updates-are-reversible-with-exact-discards` (exact Def 2.3.1 discard), `update-weight-is-the-score-change-less-fresh-choices` (forward sampling), `update-weight-is-the-projection-change` (general, any internal proposal; added 2026-10-20 after the law audit's planted +1 passed every general law), `identity-update-changes-nothing`; across generative functions, a trace updated under another parameter store (`laws/across`: `update-across-generative-functions`, as `revised-*` over ten kinds that read a store, `gen.law-instances-test`, and Scan, Mix and dimap since round 1; the final law audit's F5; and over LLM completions with other end tokens or another LM, `gen.llm.token-test`) |
| 6 Update is reversible | ✓ the same, with changed args and random constraint subsets; `update-without-fresh-choices-is-exactly-reversible`; `a-choice-that-becomes-a-call-is-discarded` |
| 7 Compositionality | ✓ `splicing-is-compositional`; `switch-branches-share-addresses` (Switch against the equivalent `if`); combinator laws over a model splicing Map/Unfold/Switch; `a-different-callee-at-an-address-is-a-new-call`; Map, Unfold and Recurse against equivalent inlined dynamic functions (`combinators-match-their-bodies-inlined`, gap G), and Mix against its documented body, a categorical and a Switch (`mix-is-a-categorical-and-a-switch`, round 1); elements independent (p. 176: `elements-are-independent`, M4: an element key that ignored the index passed every other law), also the elements, steps and nodes an update adds (`elements-an-update-adds-are-independent`, the final law audit's F9) |
| 8 Change hints | ✓ precise against unknown hints for Map, Unfold (`map-/unfold-precise-hints-match-unknown-change`), Switch and Recurse (`switch-and-recurse-precise-hints-match-unknown-change`, M4), and Scan (`scan-precise-hints-match-unknown-change`, round 1), whose updates skip the steps no change reaches (`scan-updates-skip-the-steps-nothing-changed`), with contramap's opaque argument map (`contramap-change-hints`); retdiff soundness inside the update/regenerate laws for every diff kind and nested element diffs (`gen.diff/consistent?`: vector-diffs since gap F, int-, set- and dict-diffs, M4); every propagation rule at every argument position against random edits (`every-rule-describes-the-change`, `int-diffs-add-up`, `map-diffs-describe-the-change`, `the-review-s-counterexamples`); static bodies pass precise diffs to callees (`change-hints-reach-a-callee`, `a-one-call-binding-has-a-precise-diff`, and `static-matches-dynamic` with `diff/compute`'s argdiffs) |
| 9 Regenerate | ✓ `regenerate-weight-is-the-retained-density-ratio`, `regenerate-keeps-unselected-choices`, `regenerating-nothing-changes-nothing`, `regenerating-everything-has-weight-zero` (random selections, changed args); for any internal proposal, `regenerate-weight-is-the-projection-ratio` (Eq. 4.1, M4); across generative functions (`regenerate-across-generative-functions`, F5; also over LLM completions) |
| 10 Selection MH stationary | ✓ `kernels-leave-the-posterior-invariant`, `structure-changing-moves-target-the-posterior` (thesis p_mh); with the optimal internal proposal, regenerate is Gibbs: `regenerating-the-enumerated-choices-is-gibbs` (weight 0, posterior frequencies); the p. 148 example's acceptances are 1 (`the-thesis-s-exact-acceptances`); over LLM completions, `mh-over-tokens-targets-the-posterior` (`gen.llm.token-test`, against enumeration: P(a token in a set) and the mean of log p(t1 \| t0), with a single-token move) |
| 11 Project | ✓ `project-of-everything-is-the-score`, `project-splits-the-score` |
| 12 Propose | ✓ `propose-weight-is-the-assessed-density` |
| 13 Gradients | ✓ every VJP against finite differences (`gen.array.tape-test`); `choice-gradients-match-finite-differences` (with a retgrad), `argument-gradients`, `argument-gradients-with-a-return-value-gradient` (Eq. 2.11, gap K), `parameter-gradients` (`gen.gradients-test`), also through every combinator's callees (Scan, Mix and dimap since round 1); custom gradients (`custom-vjps`, `gradients-pass-through`); refused with encapsulated randomness (`operations-that-need-exact-densities-are-loud`) |
| 14 Encapsulated randomness | ✓ (M4, `gen.encapsulated-test`) `laws/encapsulated`, the laws that hold for one ω, with `score-is-assessed-on-the-same-key` (Def 2.1.16 pointwise for untraced draws), over untraced draws (the thesis's mixture, Eqs. 4.8–4.10; black-box code between choices, Eqs. 4.11–4.14, with one draw and with two; Map, Unfold, growing ones, and splices) and estimated densities; every law over a model whose updates add untraced sites (`untraced-fresh-*`, among them `untraced-fresh-update-weight-is-the-score-change-less-fresh`); a body's draws are independent (`untraced-draws-in-a-body-are-independent`, E21); `assess-is-generate-with-every-choice` (general set, also under an override with ω); `omega-is-kept-unless-something-changes`, `untraced-draws-leave-the-sites-keys-alone`; Eq. 4.3 on the mixture, the estimator, the black-box model, untraced Map and Unfold, and a splice (`eq-4-3-the-estimate-is-unbiased`, `eq-4-3-on-black-box-code-and-combinators`); Eq. 4.4 through simulate; Eqs. 4.15–4.16 by selection MH and through update (`pseudo-marginal-mh-targets-the-exact-posterior`), and PMMH with a particle filter as the estimator (`pmmh-with-a-particle-filter-estimator-targets-the-exact-posterior`, E17); pseudo-marginal importance sampling (`pseudo-marginal-importance-sampling-is-unbiased`); §4.5.4 `importance-sampling-with-an-encapsulated-proposal`, `mh-with-an-encapsulated-proposal-keeps-the-posterior`, `smc-with-an-encapsulated-proposal-is-unbiased` (E19); `operations-that-need-exact-densities-are-loud`. Planted bugs documented in place |
| 15 Custom-proposal MH | ✓ `kernels-leave-the-posterior-invariant` (`:custom`) and `structure-changing-moves-target-the-posterior` (`:q4-mh`); `a-proposal-that-ignores-the-backward-move-is-caught`; `checked-kernels-catch-a-proposal-that-writes-an-observation`; the exact conditional as a custom proposal always accepts (p. 95, `the-thesis-s-exact-acceptances`) |
| 16 Trace translators | ✓ (M4) `a-pushforward-translates-with-weight-zero` (deterministic: p(z)·\|dz/dx\| = p(x)), `a-change-of-variables-translates-with-weight-zero` (general, both directions, with q-forward and q-backward), `extending-by-the-conditional-weighs-the-evidence`, the round-trip checks (inside `a-change-of-variables-translates-with-weight-zero` and the SMCP3 tests); Jacobians against closed forms (`jacobians-match-the-closed-forms`); translators across two models in SMC (`translating-particles-between-models-keeps-the-estimate-unbiased`, gap H); a transform calling another on the same spaces (`a-tcall-runs-the-other-body-on-the-same-spaces`, Gen.jl's @tcall) |
| 17 Involutive MCMC | ✓ discrete: stationarity and `the-involution-is-an-involution`; continuous, with transforms (M4): `involutive-mh-with-a-transform-keeps-the-posterior` (a multiplicative move, Jacobian e^u), `a-reversible-jump-keeps-the-posterior` (split/merge between dimensions, ±log 2), `a-read-the-update-keeps-is-an-output`, `the-involution-check`; `state-dependent-move-probabilities-keep-the-posterior`, `a-swap-involution-is-plain-mh`, `a-checked-involutive-kernel-catches-a-write-to-an-observation`; the symmetric translators of p. 121 (identity, 1/a under Gamma(1, 1), a conditional swap: `the-thesis-s-symmetric-translators`); with encapsulated randomness, the involution's key makes it pseudo-marginal (`involutive-mcmc-is-pseudo-marginal`) |
| 18 Kernel reversal | ✓ `reversals` (chain, cycle, repeat, mix) |
| 19 Internal-proposal override | ✓ `gen.inference.proposal/with-internal-proposal`: the general laws and the fixed-support law over proposals of some, all and a structure-deciding part of the open choices, with arguments; `the-exact-posterior-as-proposal-gives-the-evidence` (every key's weight is log p(y), also in a model that splices it); the general and fixed-support laws over models that splice an override (`spliced-internal-proposal-*`, the final law audit's 4.c); the worked examples of pp. 143–144, every outcome's weight (`the-thesis-s-generate-and-regenerate-examples`; p. 143's second table leaves b unvisited, a loud error in gen.clj); Eq. 4.1 with changed arguments (`regenerate-weight-with-new-arguments-is-the-projection-ratio`, gap C); unbiased generate (`generate-under-an-override-is-unbiased`, gap D); selection MH under an override keeps the posterior (`mh-under-an-override-keeps-the-posterior`, M4). regenerate keeps every unselected choice and is loud when a selection would drop one (`regenerating-a-structure-deciding-choice`; it was wrong before, law audit gap A); a move that adds one is weighed by the thesis's formula, so a chain on a structure-deciding choice fails only when it would drop one |
| 20 Estimators | ✓ IS and particle-filter log-ML unbiased in exp space (internal, custom, partial and optimal proposals; batched; a batched filter over a Scan HMM, against the forward algorithm, `scan-particle-filters-estimate-the-evidence`); the PF incremental weight; `importance-sampling-with-stochastic-structure-is-unbiased`; over models containing an LLM completion (`gen.llm.token-test`, against enumeration), and token SMC's estimate (`gen.llm.smc-test`); a completion under a grammar as a proposal for the LM conditioned on it, the conditional weights' mean is the probability of the grammar (`local-is-not-global`, `gen.llm.constrained-test`, against enumeration) |

Batched execution has no thesis law of its own; its law is vmap's
definition. `gen.array.batch-test` checks it per op family:
batched ops equal per-particle ops, stacked. `gen.batched-test` and
`gen.fuzz-test` check it per model: every batched particle is the scalar trace
of its choices, batched weights are the scalar ones, and batched updates obey
the update law, also over random programs.

The static language has no thesis law of its own either: `laws/static`
(`static-matches-dynamic`) requires every GFI operation on a static
function to give what the same body without `^:static` gives under the same
keys, also when update and regenerate skip nodes, and the general laws hold
over static models (`gen.static-test`). The compiled score's gradients,
which HMC, MALA and MAP use, equal the handler's and the closed form
(`choice-gradients-through-the-compiled-score-are-the-handler-s`, the final
law audit's F2).

## 4. Hazards found while mapping

- **Gen.jl's `elliptical_slice` appears to target the wrong distribution.**
  Its source (`src/inference/elliptical_slice.jl`, checked 2026-10-18)
  compares `update`'s weight with log u. For a model whose address has the
  N(μ, Σ) prior the method assumes, that weight includes the prior's own
  density ratio, which the ellipse already accounts for: the prior is counted
  twice. On the 2-d Gaussian test target of `gen.inference.mcmc-test`, the
  Gen.jl form gives a posterior mean of 0.381 and a variance of 0.220, against
  the exact 0.5625 and 0.3125. gen.clj subtracts the prior ratio and matches.
  Worth reporting upstream (owner's call).

- **GenMLX's L3 generate departs from the thesis semantics.** It returns a
  deterministic posterior-mean latent with the marginal weight. That is not a
  sample from any q in the sense of Def 4.1.1, so it breaks regenerate/MH, which
  is why GenMLX strips it inside inference.
  → gen.clj implements conjugacy as the **optimal internal proposal** (sample
  the latent from its exact conditional, weight log p̄(σ); pp. 145–146), which is
  thesis-legal and composes with regenerate.
- **GFI gaps that block parity:** retdiffs, regenerate with new args,
  shared-namespace splice, Gen.jl-style Recurse, `call_at`,
  `to_array`/`from_array`/`get_selected`, IS with a custom proposal, a stateful
  `particle_filter_step!` with a proposal, `@dist`, the `@transform` and `@kern`
  macros, and `check=`.
- **Conventions to decide early:**
  - Index base for Map/Unfold/Switch: Gen.jl is 1-based, GenMLX 0-based.
  - Distribution parameterizations: gamma by scale, categorical by probs.
  - Whether generate errors on unused constraints.
- **AD is a prerequisite on the JVM.** It must exist before choice gradients,
  HMC, MALA, MAP, VI and translator Jacobians.
