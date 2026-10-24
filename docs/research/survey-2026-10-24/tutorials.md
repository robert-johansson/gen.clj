# Tutorials for gen.clj: what exists, what to port, and how

*Research notes, 2026-10-24. Read-only survey: no file in either repo was
changed and no bean was created. Sources: Gen.jl 0.4.8's docs
(`~/.julia/packages/Gen/mP0Sq/docs/`), gen.dev, the gen-quickstart repo,
the ecosystem packages' pages, and GenMLX (`~/code/genmlx`: `docs/`,
`test/genmlx/tutorial/`, `examples/`, `.beans/`). gen.clj's features were
checked by grepping `src/`. Line references are to gen.clj's tree at
e358e8c.*

## 0. Summary

- **gen.clj has no tutorials.** It has no docs site or examples directory either.
  The README's one example (`README.md`, "A model and two kinds of
  inference") is the only user-facing code. `docs/` holds design and
  research notes. DESIGN §9 (D2) promises a 1-based → 0-based "migration
  note" for Gen.jl users, and it has not been written.
- **Gen.jl has 22 tutorial-like documents:**
  - 8 tutorials in its docs nav and 5 how-to guides. The 0.4.8 nav and today's `dev` docs are identical.
  - 1 more tutorial file that the docs don't build: `data_driven_proposals.md`. It is missing from `pages.jl` and `include`s helper files that exist only in gen-quickstart.
  - 5 gen.dev/gen-quickstart notebooks that never moved into the docs: Bottom-up Introduction, Reasoning about Regenerate, Reversible-Jump MCMC, TensorFlow MNIST and PyTorch proposals.
  - 3 small ecosystem examples: GenParticleFilters' home page, and GenSMCP3's two notebooks.
- **GenMLX has three books,** plus examples:
  - `docs/tutorial/`: 10 hand-written HTML chapters, superseded by v2.
  - `docs/tutorial-v2/`: an mdBook of 10 chapters plus an introduction, with companion nbb test scripts in `test/genmlx/tutorial/`.
  - `docs/agents/`: an 18-page mdBook that ports agentmodels.org onto `genmlx.agents`, with live figures.
  - `examples/`: 40 top-level demos and 8 subdirectories. Only `examples/genmlx/` (10 files) and `examples/memo/` (26 files) are tutorial-like.
- **`~/code/genmlx-book` (§2f), the owner's PDF book:**
  - 23 written chapters in five Parts (Foundations, Inference, Theory, The System, Applications); Applications and the appendices are stubs.
  - 10,466 lines of Markdown, 65 example files.
  - Built with Pandoc and eisvogel.
  - Parts I–II subsume tutorial-v2. Part III follows the thesis.
- **Every feature the core tutorials use is already in gen.clj.** That covers Gen.jl's 8 docs tutorials, the 5 how-tos, the two concept notebooks and v2's chapters. What is missing is outside the library:
  - plotting and animation;
  - the inverse-planning planner and geometry (data-driven proposals);
  - the airline dataset (RJ MCMC);
  - neural-network wrappers (GenTF and GenPyTorch, which gen.clj doesn't have);
  - `genmlx.agents` for the agents book (deferred-port genclj-nt3u).
- **Recommendation.**
  - **What to port:** all of Gen.jl's docs tutorials and how-tos, plus Bottom-up Introduction and Reasoning about Regenerate. Rewrite them in gen.clj idiom and our own prose.
  - **What to add:** gen.clj-original chapters on what Gen.jl doesn't teach, outlined by the genmlx-book rather than v2 (revised after §2f): the handler loop, combinators, kernels and diagnostics, gradient MCMC, SMC in depth, batched execution and the static language, the theory Part, exact structure, LLMs.
  - **The frame:** one book in the genmlx-book's five Parts, with the Gen.jl ports as its worked chapters (§5).
  - **What to leave for later:** data-driven proposals, RJ MCMC and the ecosystem examples. The agents book waits for gen.agents. Skip v1, tutorial-v2 (subsumed), the compilation-ladder chapter and the TensorFlow/PyTorch notebooks.
  - **Format:** each tutorial is a literate Clojure namespace in Clerk's file format that doesn't require Clerk. A companion clojure.test namespace loads it and checks every claim its prose makes. Plots are Vega-Lite specs as plain EDN.
  - **Rendering:** Clerk for the web, under an alias. A Markdown renderer feeds the genmlx-book's Pandoc pipeline if a PDF is wanted (§6b, §9).

## 1. Gen.jl's tutorials

Lengths are lines of Markdown source (`docs/src/…`). Effort: **S** is a day or less, **M** a few days, **L** a week or more, prose included.

### 1a. The docs' tutorials (nav of 0.4.8 = gen.dev/docs/dev)

**1. Getting Started** (`tutorials/getting_started.md`, 71 lines)
- **Topic:** linear regression, `generate` with constraints, then MH on the slope and intercept.
- **gen.clj features:**
  - `gen` (`src/gen/dynamic.clj:888`)
  - `gfi/generate` (`src/gen/gfi.clj:537`)
  - `mh/mh` (`src/gen/inference/mh.clj:81`)
  - `cm/choicemap`, `sel/select`
- **External needs:** none.
- **Effort:** S.
- **Worth porting:** yes, as page 1. It overlaps the README's example and GenMLX v2 ch01–02.

**2. Introduction to Modeling** (`modeling_in_gen.md`, 1,139 lines)
- **Topic:**
  - generative functions, `simulate`, traces and choice maps;
  - `importance_resampling` and prediction with `generate`;
  - a noise model;
  - `{*} ~` and addressed calls, `:a => :b` addresses;
  - a recursive changepoint model (a binary tree of segments).
- **gen.clj features:**
  - `gfi/simulate`, `gfi/generate`
  - `is/importance-resampling` (`src/gen/inference/importance.clj:81`)
  - shared-namespace `(splice gf args)` (`src/gen/dynamic.clj:603`) and `(splice :a gf args)`
  - vector paths
  - gamma, beta, uniform and bernoulli
  - recursive `gen` calls
- **External needs:**
  - Plots, heavily: trace grids, overlays and prediction scatters.
  - A model that returns a closure: fine in Clojure.
- **Effort:** M.
- **Worth porting:** yes, the flagship. probcomp/Gen.clj ported it as a Clerk notebook with Vega-Lite. That port is unseeded and untested, which is useful prior art for the format.

**3. Basics of MCMC and MAP** (`mcmc_map.md`, 1,291 lines; gen-quickstart's "Iterative inference", same content)
- **Topic:**
  - outlier regression, and why IS is weak on it;
  - block-resimulation MH;
  - Gaussian-drift proposals;
  - a RANSAC data-driven proposal;
  - MAP: `map_optimize` plus discrete MAP by `update`;
  - a bimodal exercise.
- **gen.clj features:**
  - `mh/mh` with a selection or a proposal
  - `gfi/update` (`gfi.clj:553`)
  - `mcmc/map-optimize` (`src/gen/inference/mcmc.clj:619`, with `:max-step-size` and `:min-step-size` as in Gen.jl)
  - least squares through `linalg/solve` (`src/gen/array/linalg.clj:73`) on the normal equations, or the 2-parameter closed form
- **External needs:**
  - Plots: scatter plus line plus a noise band.
  - `Plots.@animate` GIFs (6 animations).
  - `StatsBase.sample` without replacement.
- **Effort:** M.
- **Worth porting:** yes.
- **Porting note:** RANSAC's untraced RNG inside a proposal (§3, item 4).

**4. Debugging Models with Enumeration** (`enumerative.md`, 892 lines)
- **Topic:**
  - `choice_vol_grid` and `enumerative_inference` on a discrete regression and on a continuous one (a 24×16 grid × 2⁵ outlier flags);
  - outlier marginals and conditional slices;
  - misspecification found by enumeration where IS and MCMC mislead.
- **gen.clj features:**
  - `enum/choice-vol-grid` (`src/gen/inference/enumerative.clj:106`)
  - `enum/enumerative-inference` (`:174`, also `:batched?`)
  - `enum/marginal` (`:230`)
  - Results are flat vectors in row-major grid order, not Gen.jl's N-d arrays. Joint marginals take an `array/reshape` and `array/sum` over axes.
- **External needs:**
  - Plots: heatmaps with marginal histograms, bar charts, 2-D histograms.
  - `@sprintf`.
- **Effort:** M.
- **Worth porting:** yes. Exact answers make it the most checkable tutorial, and it shows off batched grids.

**5. Object Tracking with SMC** (`smc.md`, 764 lines)
- **Topic:**
  - bearing-only tracking with a particle filter, with and without resampling;
  - rejuvenation: resimulation MH on the initial state, and a drift proposal over a window;
  - Unfold with a static kernel;
  - a timing plot.
- **gen.clj features:**
  - `pf/initialize` (`src/gen/inference/particle_filter.clj:83`), `pf/step` (`:149`), `pf/maybe-resample` (`:421`), `pf/sample-unweighted-traces` (`:441`)
  - `pf/rejuvenate` (`:279`)
  - `comb/unfold` (`src/gen/combinators.clj:470`)
  - `(gen ^:static …)` (`src/gen/dynamic/static.clj`)
  - The struct `State` becomes a record or a map.
- **External needs:**
  - Plots: trajectories and bearing rays.
  - Wall-clock timing.
- **Effort:** M.
- **Worth porting:** yes. Add a gen.clj section on `{:batched? true}` filters.

**6. Variational Inference** (`vi.md`, 329 lines)
- **Topic:**
  - `black_box_vi!` with `@param` against an analytic Gaussian posterior;
  - amortized VI over a `Map` of observation sets;
  - a note on reparametrization.
- **gen.clj features:**
  - `vi/black-box-vi` (`src/gen/inference/vi.clj:211`)
  - `learn/gradient-descent` (`src/gen/learning.clj:55`)
  - `(param :x-mu)` with `dyn/with-params` (`dynamic.clj:839`)
  - `comb/map` (`combinators.clj:360`)
  - `sel/get-selected` (`src/gen/selection.clj:165`)
- **External needs:** none. Results print as text.
- **Effort:** S.
- **Worth porting:** yes. It has closed-form answers, so it is exactly testable. Add VIMCO (`vi.clj:223`) and the `:leave-one-out` baseline.

**7. Learning Generative Functions** (`learning_gen_fns.md`, 229 lines)
- **Topic:** prose with non-runnable snippets: maximum likelihood from complete data, Monte Carlo EM, online EM, wake-sleep, reweighted wake-sleep, VI and VAEs.
- **gen.clj features:**
  - `grad/param-gradients` (`src/gen/gradients.clj:132`)
  - `learn/train` (`learning.clj:105`), `learn/lecture` (`:132`), `learn/adam`
- **External needs:** none.
- **Effort:** S as prose; M made runnable.
- **Worth porting:** yes, made runnable. Gen.jl's version is the weakest page because nothing in it runs.

**8. Speeding Up Inference with the SML** (`scaling_with_sml.md`, 398 lines)
- **Topic:** the outlier model written four ways (a `for` loop, Map, a static outer function, static inner and outer), the timing scaling of each, and a check that all four agree.
- **gen.clj features:**
  - `comb/map`
  - `^:static`
  - `static/dot` (`static.clj:454`, Gen.jl's draw_graph)
  - `static/could-be-static` and `static/lint` (`:480`, `:492`)
- **External needs:**
  - Plots of time against n.
  - The static graph as a figure, from DOT.
- **Effort:** S–M.
- **Worth porting:** yes. Add batched execution as a fifth curve.

**9. Data-Driven Proposals** (`data_driven_proposals.md`, 1,200 lines; not in `pages.jl`, so not built)
- **Topic:**
  - inverse planning: an agent model whose path comes from an RRT planner;
  - a `piecewise_uniform` heuristic proposal;
  - IS with a custom proposal;
  - `train!` on a trainable parameter;
  - a hand-written two-layer NN proposal (plain matrices, no Flux);
  - parameters precomputed in JLD2.
- **gen.clj features:**
  - `d/piecewise-uniform` (`src/gen/dist.clj:256`)
  - `gfi/propose`
  - `:proposal` in `is/importance-resampling`
  - `learn/train`
  - gen.array matmuls and `array/tanh`
  - `gen.serialize/write` for saved parameters
- **External needs:**
  - gen-quickstart's `inverse-planning/` (`geometric_primitives.jl`, `scene.jl`, `planning.jl` at about 220 lines, `viz.jl`) must be ported to Clojure, about 300 LOC.
  - Scene plots.
  - Training (about 10 minutes in Julia) needs precomputed parameters, committed as EDN.
- **Effort:** L.
- **Worth porting:** yes, later. It is the data-driven and amortized showcase.
- **Porting note:** the planner's untraced RNG (§3, item 5).

### 1b. How-to guides

**1. Extending Gen** (12 lines)
- **Topic:** an index.
- **Effort:** S. Port as an index page.

**2. Adding New Distributions** (123 lines)
- **Topic:** `@dist` (labeled categorical, bit-flip, shifts, `exp` and `log`), and the Distribution API from scratch.
- **gen.clj features:** `dist` (`src/gen/dist/dsl.clj:210`), `defdist` (`src/gen/dist/core.clj:290`), `gen.dist.host`.
- **Effort:** S.
- **Worth porting:** yes.

**3. Adding New Generative Functions** (83 lines)
- **Topic:** implementing the GFI by hand.
- **gen.clj features:** `gfi/defgf` (`gfi.clj:629`) and the GFI protocols.
- **Effort:** S.
- **Worth porting:** yes. Add how to check one with `laws/deflaws`, which lives in `test/gen/laws.clj`, so users need the test path.

**4. Custom Gradients** (56 lines)
- **Topic:** CustomGradientGF, and custom parameter-update configurations.
- **gen.clj features:**
  - `custom/deterministic` with `:gradient` (`src/gen/custom.clj:241`)
  - optimizers are data (`learning.clj:50–71`)
  - `learn/step` is a closed `case`, so a custom optimizer is your own step function on stores, which are values
- **Effort:** S.
- **Worth porting:** yes. The second half becomes "an optimizer is a function of stores".

**5. Custom Incremental Computation** (54 lines)
- **Topic:** CustomUpdateGF summing a vector, with VectorDiff.
- **gen.clj features:** `custom/deterministic` with `:apply-with-state` and `:update-with-state`.
- **Effort:** S.
- **Worth porting:** yes.

### 1c. gen.dev and gen-quickstart notebooks missing from the package docs

gen.dev/tutorials lists 10 notebooks from gen-quickstart, all on older Gen.jl APIs. Five are older versions of the docs tutorials:
- intro-to-modeling
- iterative-inference = mcmc_map
- data-driven-proposals
- scaling-with-combinators-new = scaling_with_sml
- particle-filtering = smc

The other five:

**1. A Bottom-up Introduction to Gen** (about 4,000 words)
- **Topic:** a plain Julia program, then traces, then generative functions, then choice maps and scores; then constrained generation and hand-written IS; then `update`'s weight, discard and reversibility.
- **gen.clj features:** `uniform-discrete`, `bernoulli` and `categorical` (0-based), `gfi/update` and its discard.
- **External needs:** Plots histograms, StatsBase `mean`.
- **Effort:** S–M.
- **Worth porting:** yes, as the concepts page. It overlaps GenMLX v2 ch03 and ch05.

**2. Reasoning about Regenerate** (10–15 minute read)
- **Topic:** a 3-choice discrete model; regenerate's internal proposal, its weight tabulated and checked by hand, reversibility, irreducibility.
- **gen.clj features:** `gfi/regenerate` (`gfi.clj:571`), `sel/select`.
- **External needs:** none.
- **Effort:** S.
- **Worth porting:** yes. Every number in it is exact, and it is where to state gen.clj's Eq. 4.1 regenerate weight (DESIGN §9, U1).

**3. Reversible-Jump MCMC** (about 570-line script)
- **Topic:**
  - piecewise-constant regression;
  - a split-merge involution, hand-written with `update` and backward choice maps;
  - Bayesian synthesis of GP kernel programs (recursive grammar, `mvnormal`) on airline passenger data;
  - a subtree-regeneration proposal.
- **gen.clj features:**
  - `mh/involutive-mh` (`mh.clj:93`), using `transform/deftransform` (`src/gen/inference/transform.clj:198`) and `transform/involution` (`:505`) in place of the hand-built involution
  - `dist` (poisson_plus_one)
  - `d/dirichlet`, native (the notebook includes a `dirichlet.jl` helper)
  - `d/mvnormal`, which needs a jitter in float32 (DESIGN §9, F1)
  - `comb/recurse`, optionally
- **External needs:**
  - `airline.csv` (144 rows, vendored as EDN)
  - CSV and DataFrames, Plots with animations
- **Effort:** L.
- **Worth porting:** yes, later. It showcases involutive MCMC and program synthesis.

**4. Modeling with TensorFlow code** (tf-mnist)
- **Topic:** GenTF, supervised MNIST.
- **gen.clj features:** none (GenTF is "later" in the parity doc).
- **External needs:** TensorFlow, PyCall, MNIST.
- **Worth porting:** no.

**5. Proposals with PyTorch**
- **Topic:** GenPyTorch, extending data-driven proposals.
- **gen.clj features:** none. The hand-written NN in item 9 of §1a covers the idea with gen.array.
- **External needs:** PyTorch, PyCall.
- **Worth porting:** no.

### 1d. Ecosystem

- **GenParticleFilters:** one worked example on its docs' home page: `object_motion`, moving or still, with `pf_initialize`, residual resampling, `pf_rejuvenate!` and `pf_update!`. Every operation it uses is in `gen.inference.particle-filter`. **S. Worth porting:** yes, as a short page.
- **GenSMCP3:** `examples/notebooks/Very_Simple_Example.ipynb` (the README's toy) and `Tutorial.ipynb` (online state estimation). The tutorial is 27 MB with outputs; its contents were not readable here. gen.clj has `transform/updating-translator` (`transform.clj:462`), `tk/smcp3` (`src/gen/inference/trace_kernel.clj:155`) and `pf/move-reweight`. **S for the toy, M (unverified) for the tutorial.** Worth porting later.
- **The other packages on gen.dev/ecosystem** (GenTF, GenPyTorch, GenFluxOptimizers, GenPseudoMarginal, GenVariableElimination, Gen2DAgentMotion, GenDirectionalStats, GenRedner, GenTraceKernelDSL) link no tutorials.

## 2. GenMLX's tutorials

### 2a. `docs/tutorial/` (v1)

- **What it is:** 10 hand-written HTML chapters plus an index, 195 KB in all. They are reference-like. Chapters: introduction, generative models, conditioning, inference basics, combinators, advanced inference, vectorized inference, gradients and learning, neural models, custom extensions.
- **Status:** superseded. tutorial-v2's PLAN.md requires v2 to "cover everything the old tutorial covered".
- **Worth porting:** no. Port v2's ideas instead.

### 2b. `docs/tutorial-v2/` (mdBook)

- **Structure:** an introduction (63 lines) and 10 chapters (1,871 lines), PLAN.md (358 lines), 8 TikZ-to-PNG/SVG figures.
- **Tests:** a companion test per chapter in `test/genmlx/tutorial/chNN_test.cljs` (1,947 lines). They are nbb scripts with PASS/FAIL atoms and `println`, and the listings are typed twice, once in the chapter and once in its test.
- **Running example:** Bayesian linear regression throughout.

Chapter by chapter:

**ch01 Your first model** (238 lines; test 207)
- **Topic:** `gen`, `trace`, `simulate`, the score, MLX arrays, the distribution catalog.
- **gen.clj mapping:** everything is there. The MLX section becomes gen.array, where numbers are 0-d arrays.
- **Effort:** S.
- **Worth it:** fold into Gen.jl's Intro to Modeling.

**ch02 Conditioning** (190 lines; 243)
- **Topic:** `generate`, weights, IS by hand, the log-ML estimate.
- **gen.clj mapping:** everything is there.
- **Effort:** S.
- **Worth it:** fold into Bottom-up Introduction.

**ch03 The handler loop** (209 lines; 225)
- **Topic:** the transitions as pure functions, the one `volatile!`, the `gen` macro.
- **gen.clj mapping:** the same architecture (`gen.dynamic`'s transitions; DESIGN §3.4). Show the live source, for example with `clojure.repl/source-fn`, rather than quoting it, because GenMLX's quote drifted (dn28).
- **Effort:** S–M.
- **Worth it:** yes. It is gen.clj's rule 1.

**ch04 Choice maps, traces, selections** (258 lines; 274)
- **Topic:** Value and Node, `from-map` and `to-map`, stacking, selections as a Boolean algebra.
- **gen.clj mapping:** `gen.choicemap` and `gen.selection`. A plain map as constraints is an error (DESIGN §9).
- **Effort:** S.
- **Worth it:** yes, as a short reference-style page.

**ch05 Update and regenerate** (226 lines; 223)
- **Topic:** `update`'s weight and discard, `regenerate`, `assess`, `project`, MH by hand.
- **gen.clj mapping:** everything is there.
- **Effort:** S.
- **Worth it:** fold into Bottom-up Introduction and Reasoning about Regenerate.

**ch06 Composition** (221 lines; 219)
- **Topic:** splice, Map, Unfold, Switch, Scan, Mix, Recurse, dimap and contramap, vmap and repeat.
- **gen.clj mapping:** all are in `gen.combinators` except vmap-gf and repeat-gf, which batched execution replaces.
- **Effort:** M.
- **Worth it:** yes. Gen.jl has no combinators tutorial.

**ch07 The inference toolkit** (145 lines; 173)
- **Topic:** IS, the kernel algebra (chain, repeat, cycle, mix, reversal), HMC and NUTS, SMC, diagnostics, "when to use what".
- **gen.clj mapping:** `gen.inference.kernel` (chain, cycle, mix, repeat, kern, run), `mcmc`, `nuts`, `diagnostics`.
- **Effort:** M.
- **Worth it:** yes. Gen.jl's tutorials don't teach kernels or diagnostics.

**ch08 Vectorization and compilation** (144 lines; 156)
- **Topic:** vsimulate and vgenerate, the schema, the L1–L4 compilation ladder, automatic conjugacy, the fit API.
- **gen.clj mapping:** mostly GenMLX-only, because the ladder was not ported by design. Rewrite it as gen.clj's own chapter:
  - `gen.batched`: the batch tracer, fallbacks, spot checks;
  - the static language;
  - `with-conjugate-proposal`;
  - `gen.state-space`.
- **Effort:** M.
- **Worth it:** yes, rewritten. It is the most distinctive gen.clj material.

**ch09 Gradients, learning, neural models** (111 lines; 117)
- **Topic:** choice gradients, parameter stores, optimizers, `train`, wake-sleep, nn->gen-fn.
- **gen.clj mapping:**
  - `gen.gradients`, `gen.learning`
  - there is no NN layer, so write an MLP in gen.array
  - `custom/deterministic`
- **Effort:** S–M.
- **Worth it:** merge with Gen.jl's Learning tutorial.

**ch10 Extensions and verification** (129 lines; 110)
- **Topic:** `defdist`, `map->dist`, the GFI contracts, the static validator, the ladder.
- **gen.clj mapping:** `defdist`, `dist`, `laws/deflaws`. verify.cljs and the ladder were not ported.
- **Effort:** S.
- **Worth it:** merge with the how-tos.

### 2c. `docs/agents/`: "GenMLX Agents: Inferring Minds in the Maze"

- **What it is:**
  - An mdBook of 18 pages (2,522 lines) porting agentmodels.org chapters 1–8 to `genmlx.agents` (14 source files, 2,986 LOC), set in a Pac-Man maze, plus an LLM capstone.
  - The code is quoted verbatim from `examples/agentmodels/*.cljs` (14 files).
  - The figures are 24 live PNGs and GIFs made by a node pipeline (`docs/agents/render`: `@napi-rs/canvas`, `gifenc`).
  - `bin/agents-book-verify` runs the examples, regenerates the figures, builds the book and checks hygiene. It does not check that the quoted code matches its source.
- **gen.clj status:** blocked. Its substrate is the deferred-port bean genclj-nt3u (agents, which needs exact enumeration with caching), and it belongs with M7 (genclj-tw4p).
- **Effort:** L+.
- **Worth porting:** yes, eventually, with gen.agents. Not now.

### 2d. `examples/`, briefly

**Tutorial-like:**
- **`examples/genmlx/`** (10 files, 1,025 LOC): linear regression, HMM, compositional models, scan time series, adaptive HMC, VI, neural-probabilistic, analytical elimination (L3), fit API (L4), token SMC.
  - Seven port directly as a gallery, which is what genmlx-yo87 asks for.
  - Analytical elimination and fit become `with-conjugate-proposal` and `gen.state-space`.
  - Token SMC becomes `gen.llm.smc`, testable on the toy engines.
- **`examples/memo/`** (26 cljs files): ports of memo's recursive-reasoning demos by exact enumeration (Monty Hall, RSA, scalar implicature, politeness, Schelling, Sally-Anne, MDP, POMDP, EIG, empowerment, …). These make a gallery of small exact models. The nested ones (`thinks`, `with-cache`) wait for gen.agents.

**Showcases, not tutorials:**
- **`examples/distinctive/`** (8 demos): GenMLX's distinctive features. 03 and 04 (an LLM as a distribution; grammar conditioning) map to `gen.llm.token`, `gen.llm.grammar` and `constrained`, and 07 (shape vectorization) to `gen.batched`. The rest are about GenMLX's ladder and its lazy graph.
- **The agents examples:** `examples/agentmodels/` and `agentmodels-tui/` back the agents book.

**Neither:**
- **Research case studies** (the owner's cognitive-science replications: Gershman's habituation, rate estimation and latent-cause models, CRP operant, sensorimotor). These are paper companions; a few could become worked examples later.
- **LLM, VLM and program-synthesis demos:** `qwen36_*`, `fim_*`, `vlm_*`, `branched_smc`, `rejection_rate*`, `toolcall`, `codegen`, `cljs_expr`, `msa_matching`, the paper demos, `external_env`, `multi_model_agent`, `agents_llm_*`. They belong to M3, M6 and M7, and they need mlx-node and SCI.
- **Presentation tooling:** `genmlx-deck`, `genmlx-vlm-deck` and `genmlx-tui` (Ink terminal UIs). Skip.

### 2e. GenMLX's tutorial beans

- **genmlx-yo87** (draft): the owner "would love more worked, runnable examples". It asks for end-to-end case studies, mini-examples of each distribution, a worked combinator example, a vectorized example and a gallery appendix.
- **genmlx-dn28** (completed): an accuracy sweep of tutorial-v2 and reference-v2 against the source.
  - It found drift: the distribution count, a quoted transition, and 51 edits across 13 pages, mostly in the reference.
  - The book build could not be verified because `mdbook` was not installed.
- **genmlx-dn56** (completed): ch10's `defdist` example was wrong (it used Clojure arithmetic on MLX arrays). Its test passed vacuously, because it checked the type and not the value. The prose and the test both carried the bug.
- **genmlx-yf1h** (completed): ch02's statistical assertions failed about half the time on the Thor (unseeded, single-draw bounds). The fix was best-of-20 bounds and bands taken from about 2,200 measured draws.
- **Also relevant:**
  - genmlx-bw4t: "GenMLX Studio", genstudio-style interactive plots.
  - genmlx-ez2i, i3pp, v8sz, urk9 and o5ub: the agents book and its authoring contract. Its rule "code quoted verbatim from a runnable file" is the right one, but it was kept by hand.

**Lessons for gen.clj:**
1. Write code once. Duplicated code drifts (dn28, dn56).
2. Every claim needs a test that can fail, with fixed keys and variance-derived bands (yf1h; gen.clj's test conventions already require this).
3. Don't depend on a build tool the environment lacks (dn28's unverified mdbook build).
4. Figures should come from the code, never be drawn by hand (ez2i).

### 2f. `~/code/genmlx-book`: *The GenMLX Book: Probabilistic Programming with a Functional Core*

**What it is**
- A PDF book by the owner, written June 7–16, 2026. Its last PDF build is 852 KB (`build/genmlx-book.pdf`).
- It is not a git repository. Work is tracked in `.beans/` (42 beans):
  - complete: the 23 chapter beans and milestone M1 (Part I);
  - still todo: the Applications chapter, 5 appendices, the "Figures & diagrams" and "Example corpus" epics, and milestones M2–M6. Milestone M0 (Infrastructure) is in progress.
- **Size:** 10,466 lines of Markdown in `book/NNN-*.md`. The 23 written chapters run 357–612 lines each, 2–3 times tutorial-v2's. The preface, Applications and the 5 appendices are 11-line stubs.
- **Code:** 65 book-owned examples (7,407 lines) under `examples/NN-chapter/`, curated from GenMLX's examples. The book has 131 `clojure` listings and about 260 pasted output blocks. There are no figures (`figures/` is empty), and formulas are written as inline code, not typeset.
- **Five Parts:** I Foundations, II Inference, III Theory, IV The System, V Applications, plus appendices. Each chapter ends in a "Workshop" of exercises tied to the examples.
- **Relation to GenMLX's other books:**
  - Its notes (`notes/source-map.md`) say Parts I–II adapt tutorial-v2 and reference-v2. Its chapters subsume tutorial-v2's.
  - Part III follows Cusumano-Towner's thesis chapter by chapter (Ch. 2 → the GFI, Ch. 3 → inference as ADT operations and trace translators, Ch. 7 → design rationale).
  - It is separate from the agents book. Its Applications stub would tour the agents, memo and cognitive-science examples.

**Tooling**
- **PDF:** Pandoc 3 with a patched eisvogel LaTeX template and xelatex (`Makefile`: `make pdf`, `make verify`, `make check`).
- **Examples:** run with nbb against the sibling `../genmlx` checkout (`nbb.edn`; npm `file:` links to `mlx-node`).
- **Verification:** `scripts/verify-examples.sh` passes an example when nbb exits 0. LLM examples are skipped unless `RUN_LLM=1`.
- **Rule:** "write the example first, then quote it" (`CLAUDE.md`).
- **Weaknesses:**
  - Only 8 of the 65 examples check their results (rejection sampling, the two translator examples, the laws, ladder equivalence, grammar conditioning, defdist). The rest pass if they don't crash.
  - 26 examples set no seed.
  - Listings are quoted by hand. The one I compared, `geometric-coins` in ch. 31 against `examples/14-gfi/model_is_a_value.cljs`, matches.
  - Outputs are pasted in, so the prose's numbers are frozen from one run. For example, ch. 34 shows "retval slope/intercept: -2.9244 / 4.0449". Ch. 23's workshop quotes log-ML ranges.
  - The principle is right, but nothing enforces it beyond "it runs".

**The chapters**

Lines are Markdown lines; "ex" counts example files; effort and worth are for gen.clj.

**Part I: Foundations**

**001 Introduction** (468 lines, 2 ex)
- **Topic:** a 30-second demo; the pure model and the managed state; "a model is a value"; the FP/PP correspondence; GenMLX's "one idea, three depths" (automatic elimination, LLMs, synthesis); the GFI; the Gen ecosystem.
- **gen.clj:** rewrite around gen.clj's own principles (DESIGN §3): the handler is the semantics, laws are tests, one backend-neutral array API. Its depths are explicit, not automatic: conjugate proposals, `gen.llm`, and synthesis later.
- **Effort:** S–M. **Worth:** yes, rewritten.

**011 Your first model** (430, 3 ex)
- **Topic:** `gen` and `trace`, simulate, the fields of a trace, the score, the coin, MLX arrays, the regression.
- **gen.clj:** everything is there. MLX arrays become gen.array.
- **Effort:** S. **Worth:** merge with Getting Started and Intro to Modeling.

**012 Conditioning** (383, 3 ex)
- **Topic:** `generate`, the weight, `project`, IS by hand, the log-ML estimate, prior against posterior, rejection sampling.
- **gen.clj:** everything is there; rejection sampling is written by hand.
- **Effort:** S. **Worth:** merge with Bottom-up Introduction.

**013 The handler loop** (545, 3 ex)
- **Topic:** a random choice as an effect, the pure transitions, `generate` as simulate plus one `if`, the score as a fold, the `volatile!`, the `gen` macro, composition by recursion, value semantics on Metal.
- **gen.clj:** the same architecture (`gen.dynamic`; DESIGN §3.4). The Metal part becomes gen.array's lazy MLX backend.
- **Effort:** S–M. **Worth:** yes.

**014 Choice maps, traces, selections** (447, 3 ex)
- **gen.clj:** `gen.choicemap`, `gen.selection`, `gen.trace`.
- **Effort:** S. **Worth:** yes.

**015 Update and regenerate** (382, 2 ex)
- **Topic:** update and its discard, regenerate, assess, project, MH by hand.
- **gen.clj:** everything is there.
- **Effort:** S. **Worth:** merge with Reasoning about Regenerate.

**016 Composition** (357, 4 ex)
- **Topic:** splice, Map, Unfold and Scan, Switch, Mix, Recurse, Vmap and the transforms; addressing; closure under composition.
- **gen.clj:** all of `gen.combinators`; batched execution replaces Vmap.
- **Effort:** S–M. **Worth:** yes. It replaces tutorial-v2's ch06 as the outline.

**Part II: Inference**

**021 The inference toolkit** (400, 3 ex)
- **Topic:** IS and resampling, kernels as one type, composing kernels, running a chain, the edit interface, diagnostics, choosing an algorithm.
- **gen.clj:** `gen.inference.kernel` (chain, cycle, mix, repeat, kern, run) and `gen.inference.diagnostics`. Gibbs comes from `enum/with-exact-proposal` (`enumerative.clj:440`).
- **Missing:** the edit interface (deferred-port genclj-ob44). Gen.clj's analogue is `pf/move-reweight`.
- **Effort:** M. **Worth:** yes.

**022 MCMC** (455, 4 ex)
- **Topic:** a ridge where random walks stall, R-hat, MALA, HMC, NUTS, HMC over HMM states, Gibbs and elliptical slice, a pointer to involutive MCMC.
- **gen.clj:** `mcmc/mala`, `hmc` and `nuts`, `gen.inference.nuts` (Stan's warmup), `diagnostics/r-hat`, `mcmc/elliptical-slice`.
- **Effort:** M. **Worth:** yes. Gen.jl's MCMC tutorial has no gradient MCMC, and this chapter fills that gap.

**023 SMC** (463, 4 ex)
- **Topic:** a state-space model with a known (Kalman) answer, the price of resampling, ESS, low-variance resampling, path degeneracy and rejuvenation, SMCP3, PMMH.
- **gen.clj:**
  - `pf/*`
  - `state-space/kalman` (`state_space.clj:271`)
  - `transform/updating-translator`
  - `gen.inference.particle-mcmc`
  - PMMH as MH on `custom/estimated`
- **Effort:** M. **Worth:** yes. The Kalman ground truth makes it exactly testable, and it complements Gen.jl's tracking tutorial.

**024 VI and ADEV** (395, 3 ex)
- **Topic:** the ELBO; ADVI; when the guide family is the bias; programmable objectives (IWELBO); reparameterization against REINFORCE; ADEV.
- **gen.clj:** `vi/black-box-vi` and `vi/black-box-vimco`. A mean-field guide is written as a generative function; there is no one-call ADVI.
- **Missing:** programmable objectives and ADEV (deferred-port genclj-hc35).
- **Effort:** M for the ELBO and guide-bias half. **Worth:** yes, in part; the rest waits for hc35.

**025 Vectorization and the GPU** (485, 4 ex)
- **Topic:** shape polymorphism and its payoff, the lazy graph, branchless resampling and merge-by-mask, GPU memory management.
- **gen.clj:** a different mechanism: `gen.batched` with the batch tracer, fallbacks and spot checks; the lazy MLX backend; `gen.array.memory`'s bounds.
- **Effort:** M. **Worth:** yes, rewritten. With the static language it becomes the batching chapter.

**026 Gradients, learning, NNs** (506, 3 ex)
- **Topic:** the two gradient objects, choice gradients against parameter gradients, optimizers, the train loop, NNs as generative functions, custom gradients and amortized inference.
- **gen.clj:** `gen.gradients`, `gen.learning`, `custom/deterministic`.
- **Missing:** an NN layer (deferred-port genclj-koxn). Write the MLP in gen.array.
- **Effort:** M. **Worth:** merge with Gen.jl's Learning tutorial.

**Part III: Theory**

**031 The GFI** (612, 2 ex)
- **Topic:** thesis Ch. 2: address universes, choice dictionaries, structured, supportive and existentially sound distributions, marginal likelihood, the generative function as a tuple, each operation, the internal proposal (the five-tuple), namespaces, a model as data.
- **gen.clj:** every operation is there, and gen.clj's laws cite the same definitions (`docs/research/genjl-parity.md` §3).
- **Effort:** S–M. **Worth:** high.

**032 Inference as ADT operations** (512, 3 ex)
- **Topic:** thesis Ch. 3 as recipes: simple MC, IS, SNIS, training a proposal, MH over regenerate, HMC, the resample-move PF, AIS; a catalog table.
- **gen.clj:** everything but AIS, which isn't ported yet (parity §1k).
- **Effort:** S–M. **Worth:** high.

**033 Trace translators and involutive MCMC** (457, 2 ex)
- **Topic:** translators; the involutive acceptance ratio; reversible jump (split/merge, checked against an exact posterior); state-dependent mixtures; SMC.
- **gen.clj:** `transform/deftransform`, `involution`, `translator`, `updating-translator`; `mh/involutive-mh`; `pf/move-reweight`.
- **Effort:** M. **Worth:** high. Merge with Gen.jl's RJ notebook.

**034 Design rationale** (420, 1 ex)
- **Topic:** one source and six uses, algebraic effects, the correspondence, layered interfaces, explicit models, immutable traces, the one impurity, limitations.
- **gen.clj:** gen.clj's version is DESIGN §3 plus §9. Its "uses" differ: no ladder or automatic elimination, but the static graph, batched execution and the laws.
- **Effort:** S–M. **Worth:** yes, rewritten from DESIGN.

**035 Verification** (411, 2 ex)
- **Topic:** laws as values (GenMLX's 64-law registry with citations), running them, a taxonomy, static validation.
- **gen.clj:**
  - laws as test.check properties: `test/gen/laws.clj`, `deflaws`, the audit, parity §3's matrix;
  - no static validator;
  - the laws live on the test path, which a user-facing chapter must address (§9).
- **Effort:** M. **Worth:** high. It is gen.clj's strongest material.

**Part IV: The System**

**041 The compilation ladder** (451, 3 ex)
- **Topic:** L0–L4, bit-exactness, the fit API.
- **gen.clj:** not ported, by design. Its counterpart (the static language, `:compile?` HMC steps) goes in the batching chapter.
- **Worth:** skip.

**042 Analytical elimination** (455, 4 ex)
- **Topic:** the L3 analyzer, Rao-Blackwellization, five families, the Kalman forward filter.
- **gen.clj:** the explicit forms: `conj/with-conjugate-proposal` (`conjugate.clj:471`), `state-space/kalman` and `hmm`.
- **Effort:** M. **Worth:** yes, reframed as "exact structure, made explicit".

**043 LLMs as generative functions** (464, 2 ex, needs model weights)
- **Topic:** a token as a traced categorical, the coin and the LLM as one type, the GFI operations, grammar constraints, the generic mask, GenLM.
- **gen.clj:** `gen.llm.token`, `potential`, `grammar`, `constrained`, `smc`. Grammar constraints are a potential, not middleware.
- **External needs:** a GGUF for rendering; the tests run on the toy engines.
- **Effort:** M. **Worth:** yes (M3's showcase).

**044 Program synthesis** (404, 2 ex, needs weights)
- **Topic:** an LLM writes models; SCI evaluates them; the GFI scores them.
- **gen.clj:** SCI is "later" (DESIGN §8), and synthesis is M7.
- **Effort:** L. **Worth:** defer.

**045 Extensions** (474, 3 ex)
- **Topic:** `defdist` and its multimethods, enumerable support, `map->dist`, custom-gradient generative functions, serialization.
- **gen.clj:** `defdist`, `dist`, `custom/deterministic`, `gen.serialize`.
- **Effort:** S–M. **Worth:** merge with Gen.jl's how-tos.

**Part V and the appendices** (stubs)
- **Planned:** applications; a distribution reference, an API reference, installation, notation (the thesis's p. 219 table), a bibliography.
- **gen.clj:** generate the distribution and API references from docstrings. Port the notation table. Applications go in Phase 5 (§5).

**Verdict**
- **This is the best outline for gen.clj's own chapters, better than tutorial-v2.**
  - It is complete: Parts I–IV are written.
  - Its theory Part follows the thesis, which is gen.clj's backbone (the law audit, parity §3).
  - Its chapters are deeper and come with exercises.
  - It is the owner's own prose, so it can be adapted directly, unlike Gen.jl's, which needs re-authoring.
- **Its tooling confirms the principle** (code verified before it enters the text) **and shows the gaps to close:** quoting by hand, pasted outputs, "it ran" as the only check, unseeded runs, no figures.
- **Its PDF target is new information:** if the owner wants a gen.clj book as a PDF, the format must be able to emit Markdown for Pandoc (§6).

## 3. Porting notes: what each port must handle

These are the places where a straight transliteration of Gen.jl's code would be wrong or ugly in gen.clj.

1. **Indices are 0-based** (DESIGN §9, D2). `(:y, i)` for `i = 1:n` becomes `[:y i]` from 0, and `:data => i => :y` becomes `[:data i :y]`. `categorical` returns 0 to K−1. The "Coming from Gen.jl" page (§5) should come first, since every port depends on it.
2. **Keys are explicit** (rule 2). Gen.jl's `Random.seed!` is replaced by `(rng/key n)` passed to each operation. Tutorials should use explicit keys throughout, which is also what makes them testable. Mention `gfi/seed!` once.
3. **Stores replace mutation.**
   - `init_param!`, `ParamUpdate` and `apply!` become `dyn/with-params` and `learn/init`/`learn/step` on stores, which are values.
   - `get_param` becomes a lookup in the store.
   - The VI and learning ports change shape here. That's a feature to explain, not to hide.
4. **RANSAC inside a proposal** (mcmc_map). Gen.jl's proposal draws its subsets from the global RNG on each call, so the forward and backward runs see different guesses. gen.clj has no hidden randomness. The clean port draws the RANSAC guess in the inference loop, from that step's key, and passes it as a proposal argument. Each step is then an exact MH kernel given its guess, and a mixture of such kernels leaves the posterior invariant. Running RANSAC inside a `gen` body would also read values on the host, which the batched and MLX runs can't do (rule 3). This is a teaching point in gen.clj's favor.
5. **The RRT planner inside the model** (data-driven proposals). Its randomness becomes encapsulated randomness through `(untraced-key)`, following thesis §4.5. The score is then log p(t | ω), and importance sampling stays valid as a pseudo-marginal method. Explain this, because Gen.jl silently ignores the planner's randomness.
6. **Enumeration results are flat** and in row-major grid order. Gen.jl returns N-d arrays and its grid varies the first spec fastest. The port reshapes the weights or uses `enum/marginal`.
7. **Plots and animations.** No plotting exists (§7). Each `Plots.@animate` GIF becomes a Vega-Lite slider over iterations, or small multiples.
8. **Julia structs** (`State`, `Point`, `Scene`, `Interval`, `Node`) become maps or records. Julia multiple dispatch (`render_node!`) becomes a function on maps, or a protocol.
9. **Float32 on MLX.**
   - `mvnormal` GP kernels need a jitter at float32's scale (F1).
   - Ill-conditioned regressions near 4000 may fall back from batched execution, which is visible and loud.
   - Tutorials should avoid quoting exact numbers in prose.
10. **Timing sections** (SML scaling, SMC Unfold). Wall-clock curves can't be asserted. Show them in the notebook, but have the tests assert the deterministic statements behind them:
    - a static version gives the same results as its dynamic twin, which is a law in gen.clj;
    - all four versions give one posterior, within bands;
    - the static or Unfold version re-runs fewer nodes, which can be counted in the test.

## 4. Overlaps

| Topic | Gen.jl | GenMLX v2 | genmlx-book | Keep |
|---|---|---|---|---|
| First model, simulate, traces | Getting Started, Intro to Modeling | ch01 | 011 | Gen.jl's two, with 011's material |
| Conditioning, IS by hand, weights | Bottom-up, Intro to Modeling | ch02 | 012 | Bottom-up, merging 012 |
| Update and regenerate semantics | Bottom-up §5, Reasoning about Regenerate | ch05 | 015 | 015 with Reasoning about Regenerate |
| Handler internals | none | ch03 | 013 | 013 (gen.clj's own) |
| Choice maps and selections | none (reference pages) | ch04 | 014 | 014 |
| Combinators | Unfold in SMC; Map in SML and VI | ch06 | 016 | 016 (Gen.jl has none) |
| Kernels, diagnostics | none | ch07 | 021 | 021, without the edit interface |
| MCMC | MCMC & MAP | ch07 | 022 | both: Gen.jl's, then 022's gradient MCMC |
| SMC | SMC tracking | ch07 (short) | 023 | both: Gen.jl's, then 023's depth |
| VI | VI | ch07 (short) | 024 | Gen.jl's, with 024's ELBO half |
| Learning and NNs | Learning (prose), data-driven §4–5 | ch09 | 026 | Gen.jl's, made runnable, merged with 026 |
| Performance | SML scaling | ch08 | 025, 041 | Gen.jl's, plus 025 rewritten for gen.batched |
| GFI theory, inference as ADTs | none | none | 031, 032 | the book's |
| Involutive MCMC, RJ | RJ notebook | none | 033 | 033, then the RJ notebook as a case study |
| Laws and verification | none | ch10 (part) | 035 | 035, for gen.clj's laws |
| Exact structure | none | ch08 (L3) | 042 | 042, reframed |
| LLMs | none | none | 043 | 043 on gen.llm |
| Extending | how-tos | ch10 | 045 | the how-tos merged with 045 |
| Enumeration | Debugging with Enumeration | none | none | Gen.jl's |

## 5. What to port, in order

*Revised after §2f. The genmlx-book replaces tutorial-v2 as the outline for gen.clj's own chapters. Its five-Part shape is the recommended frame: the Gen.jl ports become the worked chapters of Parts I, II and V, rather than a separate set. Decision 5 in §9 is whether to take this frame.*

**Phase 0: infrastructure** (one bean)
- **The format spike** (§6): one chapter, Getting Started, end to end.
  - The literate namespace and its companion test.
  - The plot helper.
  - Rendered to HTML. If decision 6 in §9 wants a PDF, also rendered to Markdown and through Pandoc.
  - Runs on the JVM backend and on MLX.
- **"Coming from Gen.jl"** (S, an appendix): DESIGN D2's migration note. It covers:
  - 0-based indices;
  - vector paths for `=>`;
  - `(splice gf args)` for `{*}`;
  - keys;
  - Bool bernoulli and categorical probabilities;
  - stores;
  - D1–D6 and U1 from DESIGN §9;
  - loud lookups;
  - the names of every renamed function.

  Every later chapter links to it.

**Phase 1: Part I, Foundations**
1. Getting Started (S), then Introduction to Modeling (M). Gen.jl's two pages carry book 011's material.
2. A Bottom-up Introduction (S–M), merging book 012 (IS by hand, rejection sampling).
3. The handler loop (book 013; S–M).
4. Choice maps, traces, selections (book 014; S).
5. Update and regenerate: book 015 merged with Reasoning about Regenerate (S), stating the Eq. 4.1 weight.
6. Composition and combinators (book 016; S–M).

**Phase 2: Part II, Inference.** Gen.jl's tutorials are the worked chapters, and the book's chapters fill in what they lack.
1. The inference toolkit: kernels and diagnostics (book 021, without the edit interface; M).
2. Basics of MCMC and MAP (Gen.jl; M), then gradient MCMC: MALA, HMC, NUTS, R-hat, Gibbs, elliptical slice (book 022; M).
3. Debugging with Enumeration (Gen.jl; M).
4. Object Tracking with SMC (Gen.jl; M), then SMC in depth: Kalman ground truth, ESS, resampling, SMCP3, PMMH (book 023; M).
5. Variational Inference (Gen.jl; S), merging book 024's ELBO and guide-bias half. ADEV waits for genclj-hc35.
6. Learning (Gen.jl, made runnable), merged with book 026 (M). The MLP is written in gen.array; genclj-koxn is deferred.
7. Speeding up: the static language (Gen.jl's SML tutorial) plus batched execution (book 025, rewritten; M).

**Phase 3: Part III, Theory** (book 031–035)
- **Chapters:**
  - The GFI
  - Inference as ADT operations
  - Trace translators and involutive MCMC
  - Design rationale, rewritten from DESIGN §3 and §9
  - Verification: the laws
- **Effort:** S–M each, M for verification. The prose exists, and every feature exists except AIS and a static validator.
- **Why here:** this Part is gen.clj's most distinctive and best-backed material, because its laws were audited against the same thesis. It comes before the case studies because it costs less and says more. Decision 7 in §9 (where the laws live) must be settled before the verification chapter.

**Phase 4: Part IV, The System**
- Exact structure, made explicit: conjugate proposals and state-space models (book 042, reframed; M).
- LLMs as generative functions: `gen.llm.token`, `grammar`, `smc` (book 043; M). Tested on the toy engines; rendered with llama.cpp when a model is present, degrading as GenMLX's agents book does.
- Extensions: the five Gen.jl how-tos merged with book 045 (S each), plus "check your generative function against the laws".

**Phase 5: Part V, Applications and case studies**
- Data-Driven Proposals (L), with the inverse-planning port.
- Reversible-Jump MCMC (L), building on Part III's split/merge.
- GenParticleFilters' example (S) and GenSMCP3's tutorial (M).
- A gallery from `examples/genmlx/` (7 × S), which answers genmlx-yo87.
- The appendices: distribution and API references generated from docstrings, notation (the thesis's p. 219 table), "Coming from Gen.jl", installation.

**Deferred**
- Book 041 (the compilation ladder: not ported, by design).
- Book 044 (program synthesis: SCI, M7).
- The ADEV half of book 024 (genclj-hc35) and the edit-interface section of 021 (genclj-ob44).
- The agents book and the memo gallery's nested-reasoning demos, with genclj-nt3u and M7.
- TensorFlow MNIST and PyTorch proposals.
- GenMLX's v1 HTML and tutorial-v2: the genmlx-book subsumes v2.

**Tracking:** a `tutorials` (or `book`) epic with one task per chapter, created by the lead or the owner. Add a "tutorials" section to `docs/research/genjl-parity.md` listing each Gen.jl tutorial as ported or not, with its test.

## 6. Format

### 6a. The options

**1. GenMLX's way: Markdown chapters plus test files that re-type the listings**
- **For:** the prose is free-form Markdown, and the book builds with mdBook.
- **Against:**
  - Code is written twice and drifts; dn56 and dn28 are exactly that.
  - Tests are nbb scripts outside the runner.
  - mdBook is a Rust binary outside the Clojure toolchain.

**1b. The genmlx-book's way: Markdown chapters quoting verified example files, built to PDF with Pandoc** (§2f)
- **For:**
  - The examples come first, and every listing has a file that runs.
  - The output is a print-quality book.
- **Against:**
  - Quoting is by hand.
  - Outputs are pasted in, so numbers are frozen from one run, and 26 of the 65 examples are unseeded.
  - "Verified" means the process exited 0; only 8 of the 65 examples check anything.
  - Pandoc and xelatex are external tools.
  - It has no figures yet.

  Its principle is the one to keep. Its mechanics are what the recommended format automates.

**2. An mdBook whose code blocks are extracted and tested**
- **For:** code is written once, and Markdown is a familiar source.
- **Against:**
  - You need an extractor (fences to namespaces), with conventions for hidden setup code and assertions.
  - Markdown isn't REPL-loadable, which works against the `(t)` loop.
  - Figures need a separate image pipeline. GenMLX's agents book used node with canvas and gifenc.
  - Results shown in the book are frozen at authoring time.
  - It still needs mdbook.

**3. Clerk notebooks that call Clerk** (`clerk/vl`, `clerk/row`), as probcomp/Gen.clj did
- **For:** code is written once, it is REPL-native, results are live, and Vega-Lite is built in.
- **Against:**
  - Loading a tutorial in the test battery puts Clerk and its dependency tree on the test classpath.
  - Clerk's cache serializes values, and MLX arrays are native handles. This needs `no-cache` or a spike.

**4. Recommended: literate namespaces in Clerk's file format that never require Clerk**
- **For:**
  - Code is written once, in the file that runs.
  - Tests load the file directly, so a stale API call is an error at load time.
  - Clerk is needed only to render the file, under its own alias.
  - Falling back to a small in-house renderer stays cheap.
- **Against:**
  - Prose lives in `;;` comments.
  - Rendering details (viewer registration, caching) need a spike.

### 6b. The recommended layout

- **`tutorials/gen/tutorial/<slug>.clj`**, namespace `gen.tutorial.<slug>`.
  - A new directory: not `src`, so tutorials don't ship in the library, and not `test`, because they are documents.
  - Added to the `:test` alias's `:extra-paths`. Since `dev/user.clj`'s `(t)` scans the classpath directories, it reloads a tutorial and reruns its test when a namespace the tutorial uses changes.
- **The body:**
  - `;;` Markdown blocks and top-level forms, in Clerk's convention.
  - House style (rule 4: gen.array's operators referred).
  - Explicit keys, `(rng/key n)`, for every operation.
  - Results as `def`ed values, not prints.
  - No Clerk require; plots come from `gen.tutorial.plot` (§7).
- **The companion test: `test/gen/tutorial/<slug>_test.clj`.**
  - It requires the tutorial, so loading runs the tutorial top to bottom.
  - It checks every claim the prose makes:
    - exactly, against closed forms (VI's posterior, enumeration's grid, regenerate's tables, Kalman likelihoods);
    - or with 5-standard-error bands at fixed keys.
  - It documents one planted mutation that it catches (CLAUDE.md, Power), for example dropping the resampling step from the SMC tutorial.
  - It tags long tutorials `^:slow`, and scalar-heavy ones `^:heavy` for MLX.
  - It reads values with `ts/host`, `close?` and `tol`, so it runs on every backend.
- **Size:**
  - Pick tutorial sizes that run in seconds on the JVM backend, which is easy at Gen.jl's sizes (for example 5,000 particles × 50 steps, batched).
  - The prose can suggest larger runs.
  - No hidden "test size" knob, which would make the test check a different run than the reader sees.
- **Rendering: two renderers over the same files, both under a `:notebooks` alias.** Which one is primary depends on decision 6 (§9).
  - **Clerk** (`io.github.nextjournal/clerk`) plus a `dev/notebooks.clj`, for live writing and an HTML site.
    - It registers, with `clerk/add-viewers!`, a viewer that shows maps carrying a Vega-Lite `:$schema` as plots (to confirm in the spike).
    - It disables Clerk's cache for `gen.tutorial.*` (MLX arrays are native handles; to confirm).
    - It exposes `serve!` for writing and `build!` for a static site in `target/tutorials`.
  - **A Markdown renderer** of about 150 lines.
    - It reads the namespace with tools.reader. Comments become Markdown, forms become code blocks, and marked results become output blocks or figures.
    - Each output block is printed from the value the run computed, never pasted, which removes the genmlx-book's frozen numbers.
    - Its Markdown feeds an mdBook or a Pandoc build: the genmlx-book's `Makefile` and patched eisvogel template could be reused for a PDF.
    - If only a website is wanted, this renderer is the fallback in case Clerk's spike fails.
  - Rule 8 holds: neither renderer is a core or test dependency, and Pandoc stays an external, optional tool.
- **Chapters end in exercises** (the genmlx-book's "Workshop", Gen.jl's "Exercise" blocks). Their solutions live in the companion test, so the answers are checked too.
- **Math** is written as LaTeX (`$…$`), which Clerk (KaTeX), mdBook (MathJax) and Pandoc all render, rather than as the genmlx-book's inline code.
- **Publishing** (GitHub Pages or gen.clj's site) is outward-facing, so it is the owner's call. So is attribution: our own prose, crediting Gen.jl's tutorials as the curriculum, as GenMLX's agents book credits agentmodels.org. Gen.jl is Apache-2.0 (its LICENSE covers `docs/`). The gen-quickstart README states no license (a LICENSE file was not checked). `airline.csv` is the classic Box–Jenkins series; check its provenance before vendoring.

## 7. Plots without core dependencies

- **Plots are Vega-Lite specs as plain EDN data**, built by a small `gen.tutorial.plot` namespace in `tutorials/` (not `src`, so the core is untouched). It needs about 10 builders for every plot in Gen.jl's tutorials:
  - scatter, line and rule (the bearing rays);
  - an area band (the noise band);
  - histogram and 2-D histogram, heatmap with marginals (enumeration);
  - bar;
  - a `grid` of trace renderings (`concat` or `facet`; Gen.jl's `grid(render_trace, traces)`);
  - `frames`, a Vega-Lite `param` slider over an iteration field, replacing `Plots.@animate`.
- **Plots are testable as data.** A test can check a figure's data rows, for example that 200 traces are drawn, without rendering.
- **Rendering:**
  - Clerk renders them under `:notebooks`.
  - For a reader without Clerk, a `plot/html` function under the alias writes one HTML page with vega-embed. JSON comes from a small EDN printer, or from clojure.data.json in the alias.
  - Static PNG, SVG or PDF output needs vl-convert or vega-cli, which are external tools. It is needed only for a PDF edition (decision 6) or for GitHub READMEs. In a PDF build, the Markdown renderer would call it at build time, outside gen.clj's dependencies, as Pandoc is. The genmlx-book has no figures yet, so it offers no pipeline to reuse.
- **Why not hand-written SVG:** it works with no JavaScript, but it is more code to write and maintain (axes, scales, legends, layout) for less: no interaction, and no slider for animations.
- **genmlx-bw4t's Studio** (interactive, genstudio-like) stays a separate, later idea. These specs would feed it.

## 8. Keeping tutorials from rotting

1. **They run in the battery.** Each companion test loads its tutorial, so any API change that breaks a tutorial fails `-X:test:fast:par`. Slow ones fail `-X:test:par`. All of them fail the MLX batteries, since they use gen.array and the `ts/` helpers.
2. **Claims are tests.** Prose says nothing numeric that no test checks; values are shown live instead. Each test lists the prose claims it checks, in order, so that a reviewer can match the two.
3. **Symbols resolve.** A test scans each tutorial's comments for backticked `gen.*/name` symbols and resolves them. This automates GenMLX's dn28 sweep, and catches a rename that leaves the prose behind.
4. **Source is shown, not quoted.** Where a tutorial shows gen.clj's own code (the handler-loop chapter), it calls `clojure.repl/source-fn` at render time.
5. **Rendering is checked separately.** `clojure -X:notebooks` builds the site before a release or a publish, and the hand-off records that it ran. Unlike GenMLX's book build, which never ran (dn28), it is a Clojure alias, so it can always run.
6. **Laws, not samples.** Comparisons with Gen.jl's printed outputs are statistical, as for GenMLX (rule 7). Exact values come from closed forms. Any Gen.jl oracle values go in an EDN fixture, so the tests never need Julia.

## 9. Decisions for the owner

1. **The format** (§6b): literate namespaces in Clerk's format, checked by companion tests, and rendered under a `:notebooks` alias by Clerk, a Markdown renderer, or both (see 6). The alternatives are an mdBook with extraction, or the genmlx-book's hand-quoted examples. This is a DESIGN §9 entry once decided.
2. **The `tutorials/` path on the `:test` classpath**, which adds tutorials to the battery's time. The fast-tier budget is about 50 s in parallel. Tutorials that take more than a few seconds would be tagged `^:slow`.
3. **Whether and where to publish the rendered site**, and the attribution and licence wording.
4. **The order of the Parts** (§5): Theory (Part III) before the System and the case studies, as recommended, or the batching and static-language chapter moved into Part I as an early showcase.
5. **One book or a tutorial site** (added after §2f).
   - **Recommended:** one book in the genmlx-book's five Parts, with the Gen.jl ports as its worked chapters (§5).
   - **The alternative:** Gen.jl's shape, with tutorials and how-tos as separate pages, plus our own chapters.
   - Both use the same files.
6. **A PDF edition, or the web only.**
   - **PDF:** the Markdown renderer becomes primary, reusing the genmlx-book's Pandoc and eisvogel pipeline, and figures need an external Vega-Lite converter at build time.
   - **Web only:** Clerk can be primary.
7. **Where the laws live for users.** `gen.laws` and `deflaws` are in `test/gen/laws.clj`, on the test path only. The verification chapter (book 035) and the how-to "check your generative function against the laws" need a user-facing entry point. The options:
   - **Recommended:** an `ext/laws` extension with its own alias, following the `ext/commons-math` pattern, so that test.check stays out of the core (rule 8);
   - a separate test artifact;
   - telling readers to add `test/` to their classpath.

   This is architecture, so it is the owner's call before Phase 3.
8. **Reusing the genmlx-book's prose.** It is the owner's own, so it can be adapted directly, unlike Gen.jl's, which we re-author and credit. Confirm that the owner wants it reused rather than re-written.

## Sources

- Gen.jl 0.4.8 docs: `~/.julia/packages/Gen/mP0Sq/docs/` (`pages.jl`, `src/tutorials/`, `src/how_to/`); gen.dev docs nav: https://www.gen.dev/docs/dev/
- gen.dev tutorials: https://www.gen.dev/tutorials/ (rj, bottom-up-intro, regenerate pages)
- gen-quickstart: https://github.com/probcomp/gen-quickstart (`tutorials/`, `inverse-planning/planning.jl`, the "Reversible-Jump MCMC in Gen.jl" and "Iterative inference in Gen.jl" scripts)
- Ecosystem: https://www.gen.dev/ecosystem/, https://probcomp.github.io/GenParticleFilters.jl/stable/, https://github.com/probcomp/GenSMCP3.jl
- Prior art, Clerk: https://github.com/probcomp/gen.clj (`examples/intro_to_modeling.clj`, `deps.edn`)
- GenMLX: `docs/tutorial/`, `docs/tutorial-v2/` (with PLAN.md), `docs/agents/` (PLAN.md, render/), `test/genmlx/tutorial/`, `examples/`, `bin/agents-book-verify`, beans genmlx-yo87, dn28, dn56, yf1h, bw4t, ez2i, i3pp, v8sz, urk9, o5ub
- The GenMLX Book: `~/code/genmlx-book` (`README.md`, `CLAUDE.md`, `Makefile`, `scripts/verify-examples.sh`, `notes/`, `book/`, `examples/`, `.beans/`)
- gen.clj's deferred-port beans touched by the book: genclj-hc35 (ADEV and programmable VI), genclj-koxn (NN layers), genclj-ob44 (edit interface), genclj-nt3u (agents)
