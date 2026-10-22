# Post-M5 review (PPL soundness): P2, combinators, distributions and randomness

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The combinators are sound as far as I could push them. All 22 GFI laws pass
on Map, Unfold, Switch and Recurse with changed arguments: length changes,
index changes, compatible branches, a Switch of Maps. Precise argdiffs give
the same results as unknown-change, with constraints and selections. Every
density I derived by hand matches its code, and the samplers' algorithms
(Marsaglia–Tsang, PTRS, BTRS, Gumbel-max, the inversions) match their
references.

The problems are at the edges of the number range and in two compositions:

- **The most severe (P2-09, critical):** `fold-in` keeps only 32 bits of its
  data, and Recurse folds in node numbers, which pass 2^32 below depth 32 of
  a binary tree. Every node of a deep chain then draws the same values,
  silently.
- **Underflow (P2-01, high):** gamma, beta and dirichlet draws that underflow
  to 0 are not floored, as §9 says they are. A trace then has score +Inf,
  and the next move that keeps the value is NaN. About half the draws of a
  BUGS-style `gamma(0.001, 1000)` do this.
- **Mixtures (P2-02, P2-06, both high):** a mixture of a discrete and a
  continuous family adds a pmf to a pdf, which gives the wrong posterior.
  A homogeneous mixture of vector-valued normals mixes each coordinate
  separately, unlike Gen.jl's and unlike its docstring.
- **Pushforwards (P2-03, high):** a `dist` body that broadcasts its draw to
  a larger shape is scored as several independent draws.

## Findings

### P2-09: fold-in keeps 32 bits of its data, so Recurse's deep nodes share keys and repeat their draws
- **Severity:** critical
- **Where:** `src/gen/random/threefry.clj:85` (`(bit-and (long data) 0xFFFFFFFF)`), reached by `src/gen/combinators.clj:655` (`node-key` folds in the node number). The same masking sits under `src/gen/gfi.clj:451` (`next-key` folds in an AtomicLong counter).
- **What:** `fold-in(k, d)` hashes the counter `[0, d mod 2^32]`, so `d` and `d + 2^32` give one key. Recurse numbers node `j` of node `i` as `i·m + j + 1`, so numbers grow as `m^depth`. For an even `max-branch` `m`, the map `x ↦ m·x + c (mod 2^32)` drops bits, so every chain that keeps taking the same child index converges mod 2^32. With m = 2 that happens from depth 32 on, with m = 4 from depth 16, with m = 8 from depth 11. All those nodes then get one key, the same production and aggregation draws, while the score treats them as independent. In a stochastic chain, the decision to continue is then the same at every deep node. Either the tree stops at depth 32 or 33, or node numbers overflow a long and throw.
- **Evidence:** `OUT/scripts/P2-09.clj`, run:
  - `fold-in(k, 5) = fold-in(k, 5 + 2^32)` is true.
  - A 40-deep chain in a binary Recurse, `v ~ normal(0, 1)` at each node: the values at depths 32 to 40 are all 1.5664 (1 distinct value, where independent draws give 9).
  - A chain that continues with probability 0.95 should be deeper than 32 with probability 0.95^33 = 0.184. Of 400 simulations, none was deeper than 32, and 83 threw `ArithmeticException: long overflow`.
  - By the laws: simulate must sample p(τ), whose deep `v`s are i.i.d.; here they are equal. Every estimate over such trees is wrong, and nothing says so.
  - The `next-key` instance: the implicit key stream repeats after 2^32 draws (by reading; not run).
- **Direction:** Fold all 64 bits of the data into the counter, `[hi32(d), lo32(d)]`, which is split's counter. That is bit-identical to today's and to JAX's for d < 2^32, and distinct beyond. Alternatively, reject data ≥ 2^32 loudly. Separately, Recurse could fold each child index into its parent's key, so a node's key never depends on a global number.
- **Known?** new

### P2-01: gamma, beta and dirichlet draws that underflow to 0 give +Inf (and beta NaN) scores
- **Severity:** high
- **Where:** `src/gen/dist.clj:91` (gamma), `:76-81` (beta), `:284-287` (dirichlet); the boost in `src/gen/random.clj:174`.
- **What:** For a shape α < 1, `rng/gamma` returns `Γ(α+1)·U^(1/α)`, which underflows to exactly 0 for small α. §9 (DESIGN.md:603-604) says such draws are floored at the smallest normal number, and `floored` does this for chi-squared, f-dist, nakagami, student-t, weibull and levy. It does not for:
  - gamma, `(* theta (rng/gamma …))`;
  - beta, `x/(x+y)`: 0 or 1 when one gamma underflows, NaN (0/0) when both do;
  - dirichlet, `g/Σg`: an exact 0 component.

  The log density at those values is +Inf when α < 1, so simulate returns a +Inf score. Any update or regenerate that keeps the choice then computes `lp − lp_old = Inf − Inf`, which is NaN.
- **Evidence:** `OUT/scripts/P2-01.clj`, run (float64):
  - `gamma(0.001, 1000)`, BUGS' vague `dgamma(0.001, 0.001)`: 963 of 2,000 scores are +Inf. `gamma(0.01, 1)`: 8 of 20,000.
  - `beta(0.001, 0.001)`, 2,000 draws: 1,253 scores are +Inf, and 448 draws are NaN (scored −Inf).
  - `dirichlet(0.01·1₁₀₀₀)`, a sparse prior: 87 of 200 scores are +Inf.
  - Chi-squared(0.002), which is floored: 0 of 2,000.
  - Downstream, a model with `lam ~ gamma(0.001, 1000)`, `y ~ poisson(lam)` with y = 0, and `x`, `z`, starting from a generated trace with `lam = 0`:
    - selection MH on `:x` throws "The MH acceptance ratio is NaN" (with lam floored at `Double/MIN_NORMAL`, log-alpha is 0.038);
    - a custom-proposal move on `:lam` has log-alpha −Inf (silent).
  - In float32 (MLX), U^(1/α) underflows below 1.4e-45 (1.2e-38 where Metal flushes subnormals), so `gamma(0.001)` gives 0 for U < 0.90 (derived, not run).
- **Direction:** Floor gamma's draw as the other families do. For beta and dirichlet, draw the boosted gammas in log space (`log Γ(α+1) + log U / α`) and form `sigmoid(log x − log y)` and `softmax(log g)`. Clamp to the decision's floor, and for beta also to `1 − ε/2`, since values within 2^-53 of 1 are not representable.
- **Known?** new. Review 4 fixed this for the newer families only (`test/gen/dist_test.clj:473`).

### P2-06: a homogeneous mixture of vector-valued normal components mixes each coordinate separately
- **Severity:** high
- **Where:** `src/gen/dist.clj:740-742` (log-prob), `:724-738` (sampler), `:685-708` (docstring)
- **What:** `((homogeneous-mixture normal [1 1]) w mus stds)` with `mus` of shape [K d] is documented as mixing the components `(normal (mus k) (stds k))`. G6 in `docs/research/genjl-test-suite.md` says it "mixes the K components' densities of the whole value". normal's own docstring says its density with array parameters is the sum over elements (broadcasted_normal).
  - Expected density: Σ_k w_k Π_j N(v_j | μ_kj).
  - What log-prob computes: Π_j Σ_k w_k N(v_j | μ_kj), stacking each component's elementwise log density and mixing per element.
  - The sampler agrees with that density (one pick per element), which is why the laws pass.

  A mvnormal component (scalar log density) mixes whole vectors, as Gen.jl's HomogeneousMixture does. So a diagonal-covariance GMM written with `normal` is a different model from the same GMM written with `mvnormal`. The heterogeneous mixture mixes per element the same way.
- **Evidence:** `OUT/scripts/P2-06.clj`, run. Two clusters at (0, 0) and (10, 10), unit variance:
  - At v = (0, 10), the mixture's log density is −51.84, which is what `mvnormal` components give. The normal mixture scores −3.22, from [−1.61 −1.61] per coordinate.
  - Of 20,000 draws, 9,988 have one coordinate near each cluster; a mixture of the components gives none.
- **Direction:** Sum each component's log density over the component's own value axes (its `dc/value-shape`) before the logsumexp, and in the sampler pick one component per value. Only the axes beyond a component's value remain elementwise (batch) axes.
- **Known?** new. G6 records the intended behavior.

### P2-02: a heterogeneous mixture of a discrete and a continuous family adds a pmf to a pdf
- **Severity:** high
- **Where:** `src/gen/dist.clj:710-722`, `:740-742`; `:780` (`discrete?` false for such a mixture)
- **What:** `heterogeneous-mixture` accepts `[poisson normal]`; review 14 made its sampler keep both kinds of values. Its log-prob is `logsumexp(log w + [log pmf(v), log pdf(v)])`. Take the measure μ = counting + Lebesgue. At an integer v, which the Poisson component hits with positive probability, the mixture's density with respect to μ is w₁·pmf(v): the normal component has no atom there. The added w₂·pdf(v) is wrong on a set of positive probability, so posteriors conditioned on integer data are wrong. Gen.jl's types (`Vector{Distribution{T}}`, one T) rule this mixture out.
- **Evidence:** `OUT/scripts/P2-02.clj`, run. `w` picks weights [0.2 0.8] or [0.8 0.2] (prior ½ each) for a mixture of poisson(3) and normal(0, 1), and y = 0 is observed:
  - The exact posterior is P(w = 1 | y = 0) = 0.8, since only the Poisson part can give y = 0.
  - gen.clj's generate weights give 0.2666.
  - Rejection from simulate agrees with the exact value: 3,995 of 4,934 runs with y = 0 had w = 1, which is 0.810 ± 0.006.
- **Direction:** Reject loudly a mixture whose components disagree on `dc/discrete?`, as Gen.jl's types do. Alternatively, score with respect to counting + Lebesgue: at an atom of a discrete component, only the discrete components' mass.
- **Known?** new. Review 2026-10-18, item 14, fixed the sampler's casts but not the density.

### P2-03: a pushforward whose body broadcasts its draw to a larger shape is scored as several draws
- **Severity:** high
- **Where:** `src/gen/dist/dsl.clj:176-200`; no value-shape is declared (`:169`, defdist without `value-shape`, so nothing checks it)
- **What:** The namespace doc allows "gen.array code on the parameters and constants, elementwise". With a constant or parameter of a larger shape, one scalar draw becomes an array value. Both density paths then score that value element by element, as if each element were a draw, and the trace's score sums them:
  - the enumerable path compares elementwise (`maps-to` with `array/==`);
  - the inversion path inverts elementwise and adds the Jacobians per element.

  The value's true law puts its mass on one draw's image: a single atom per draw, or a curve with no density in R^n.
- **Evidence:** `OUT/scripts/P2-03.clj`, run.
  - `(dist coin-pair [p] (* (astype (bernoulli p) :float64) [1 2]))` with p = 0.3:
    - the score of [1 2] is −2.408, where the exact value is log 0.3 = −1.204 (the coin is counted twice);
    - the impossible [1 0] scores −1.561, where the exact value is −Inf.
  - `(dist shifted-pair [mu] (+ (normal mu 1) [0 1]))`: the off-line value [0.3 5.0], of probability 0, has a finite density.
- **Direction:** Give the pushforward the draw's value shape (`dc/value-shape` of the draw), so a body that changes the shape is a loud error at the site. Or check in log-prob that the body's value has the draw's shape.
- **Known?** new

### P2-04: a pushforward of a non-enumerable discrete draw gives mass to values off its lattice
- **Severity:** medium
- **Where:** `src/gen/dist/dsl.clj:185-194`
- **What:** The inverse is rounded to an integer and accepted when `|f(x) − v| ≤ 1e-5·(1 + |v|)`. The comment assumes that images of distinct integers lie much further apart. That fails for any scale below about 1e-5, and for reciprocals of large counts. Values that no draw produces then get the mass of a nearby one, so "−Inf outside the support" fails, and the "pmf" sums to far more than 1.
- **Evidence:** `OUT/scripts/P2-04.clj`, run. `(* 1e-6 (poisson 3))`:
  - log-prob is −1.496 at 2e-6 (correct), and also −1.496 at 1.5e-6 and at 2.4e-6, where the exact value is −Inf;
  - summed over a grid of [0, 1e-5] in steps of 1e-8, the mass is 97.4.

  `(/ 1 (+ 1 (poisson 500)))` gives mass to 1/501.5.
- **Direction:** Accept only within a few ulps of v, or within half the local lattice spacing `|f(x+1) − f(x)|/2`, instead of a fixed relative 1e-5.
- **Known?** new

### P2-05: invalid parameters accepted, giving NaN or −Inf densities silently
- **Severity:** medium
- **Where:** `src/gen/dist.clj:646` (neg-binom), `:65-66` (uniform), `:519-522` (uniform-discrete), `:625` (poisson), `:633-634` (binom)
- **What:** Rule 5 says to validate host parameters and to read concrete arrays. Several families check host values only, or miss a bound:
  - neg-binom accepts p = 0, which Distributions.jl rejects (as geometric here rejects p = 0);
  - uniform, uniform-discrete, poisson and binom do not check concrete array parameters.

  The results are draws with density −Inf, or NaN densities.
- **Evidence:** `OUT/scripts/P2-05.clj`, run:
  - `neg-binom(2, 0)` draws 0 with log-prob −Inf (`geometric(0)` throws);
  - `poisson([-1])` draws 1 with log-prob **NaN** (host `poisson(-1)` throws);
  - `uniform([1], [0])` draws 1.0 with log-prob −Inf;
  - `uniform-discrete([3], [1])` draws 2 with log-prob −Inf;
  - `binom([2.5], 0.5)` draws 0 with a finite log-prob −1.73;
  - `piecewise-uniform` with 2-d parameters, which `-value-shape` (`:791`) admits, throws an index error rather than a parameter error.
- **Direction:** Check concrete arrays as host values are: low < high, λ ≥ 0, integer bounds, p ∈ (0, 1] for neg-binom. Make piecewise-uniform's value shape 1-d, or loud, until it batches.
- **Known?** new

### P2-07: dirichlet's simplex check is float64's tolerance, which float32 sums can exceed for large K
- **Severity:** medium (suspected; MLX not run)
- **Where:** `src/gen/dist.clj:289-290` (`|Σv − 1| ≤ 1e-6`)
- **What:** The log density is −Inf when a value's sum differs from 1 by more than 1e-6, whatever the dtype. MLX's CPU reduction accumulates float32 in SIMD lanes (`native/build/src/mlx/mlx/backend/cpu/reduce.cpp:99-109`), sequentially within each lane. The sampler normalizes by one float32 sum, and the check takes another. For large K, a draw's own density can therefore be −Inf, and so can float32 probability vectors given as data.
- **Evidence:** `OUT/scripts/P2-07.clj`, run on the JVM as an emulation of float32 sums on real draws:
  - sequential accumulation: 15 of 200 dirichlet(1) draws fail at K = 1,000, and 126 of 200 at K = 10,000;
  - pairwise accumulation: none fail;
  - float64: none fail.

  MLX's lane-wise sums lie between the two (about K/8 sequential adds per lane), so failures are likely at K ≳ 10⁴ on MLX's CPU. Not run on MLX (rationed).
- **Direction:** Scale the tolerance with the dtype (for example K·ε), or check the sum in float64.
- **Known?** new

### P2-08: laplace's lower bound is a float64 constant, which is −1 in float32
- **Severity:** low
- **Where:** `src/gen/dist.clj:120`
- **What:** `(Math/nextAfter -1.0 0.0)` rounds to −1.0f, so in float32 the uniform can be exactly −1. Then `log1p(−|u|) = −Inf`, and the draw is +Inf with log density −Inf, about once in 2^23 draws. `rng/normal` uses a dtype-aware bound for exactly this reason (`src/gen/random.clj:94-96`).
- **Evidence:** `OUT/scripts/P2-08.clj`, run with `rng/uniform` at float32 on the JVM backend: 1 of 2^24 draws is exactly −1.0, and the transform there gives +Inf. With normal's float32 bound, none is.
- **Direction:** Pick the bound by dtype, as `rng/normal` does.
- **Known?** new

### P2-11: in float32, the binomial inversion can count past n
- **Severity:** low
- **Where:** `src/gen/dist.clj:592-596`
- **What:** The inversion adds 1 for each of 40 CDF values below u. Past k = n the terms are 0 and the CDF stops growing. When the float32 CDF at n is below the largest uniform, 1 − 2^-23, a draw above it counts every later step: k = 40 > n, or n − 40 < 0 after the flip for p > ½. Such a draw has density −Inf.
- **Evidence:** `OUT/scripts/P2-11.clj`, run as a host emulation of the sampler's float32 recurrence. In 220 of 1,670 (n, p) cases with n·p < 10, the CDF at n falls below 1 − 2^-23; for example (n = 7, p = 0.41) gives P(k > n) = 1.8e-7. It is negligible in float64.
- **Direction:** Clip the inversion's count at n.
- **Known?** new

### P2-10: no law instance runs Map, Unfold or Switch with changed arguments
- **Severity:** low (coverage; no bug found)
- **Where:** `test/gen/combinators_test.clj:73-84` (the `:map`, `:unfold`, `:switch` and `:keyed-switch` entries have no alternative arguments)
- **What:** The laws with new arguments (update reversibility and discards, the update weight's projection, regenerate with new arguments) never see a Map length change, an Unfold n or init change, or a Switch index change. The specific tests cover some of these paths, but the general laws do not.
- **Evidence:** `OUT/scripts/laws-alt-args.clj`, run: the 22 laws of `laws/all` at 150 cases over Map, Unfold, Switch (incompatible and compatible branches, Switch of Maps) and Recurse, with alternative arguments. All pass.
- **Direction:** Add those alternative arguments to the models list.
- **Known?** new

## Decisions worth revisiting

- **§9, "A draw beyond the float range is +Inf … floored at the dtype's smallest normal number"** (DESIGN.md:596-605). Flooring is right once P2-01 extends it to gamma, beta and dirichlet, but it is not a full answer for very small shapes. For `gamma(0.001, θ)`, about half the prior mass lies below `Double/MIN_NORMAL`. The floor turns that half into an atom whose recorded log density is about 700. A random-walk MH on `lam` started at the floor did not move in 2,000 steps (`OUT/scripts/P2-01.clj`). Suggestion: say in gen.dist's doc that such priors want a log-space parameterization.

## Checked and sound

- **Combinator weights, discards and retdiffs:**
  - Read: Map (generate, update, regenerate, assess, project, propose), Unfold (the skip path, the state chaining through retdiffs), Switch (same branch, compatible, incompatible), Recurse (both phases, dropped nodes), call-at and choice-at.
  - Derived: update weights are Δscore − log q(fresh); dropped elements are subtracted and discarded in update and cancel in regenerate (Eq. 4.1); new elements simulate with weight 0 in regenerate.
  - Ran: the 22 laws with changed arguments (above), and precise argdiffs (`diff/compute`) against unknown-change for update with constraints and regenerate with selections over Map, Unfold, Switch of Maps and Recurse. 0 of 1,120 comparisons (560 updates, 560 regenerates) differ beyond rounding (`OUT/scripts/precise-diffs.clj`).
  - U1 and U5 do not affect gen.clj: Switch's regenerate is Eq. 4.1, and generate's trace score is the branch's (`combinators.clj:443`).
- **Keys:**
  - Each element, step and node takes its own fold-in, Switch passes its key on, and every site consumes one key. These hold below 2^32 (P2-09 is the exception).
  - No sampler both splits a key and draws from it: beta, gamma, Poisson, binomial, neg-binom, beta-uniform, piecewise-uniform, mixtures and products all split first.
  - `fold-in(k, i) = split(k)[i]` holds by design. The handler's site keys are split and ω uses a large salt, so no collision (read).
- **Threefry:** the word construction matches JAX's partitionable scheme (read; known-answer tests exist); MLX's `random-uniform` hashes the same counters.
- **Uniform bits:** `uniform-doubles` gives [0, 1 − 2^-52]; the host families' `u + 2^-53` is strictly inside (0, 1).
- **Densities, derived by hand against the code:** normal, uniform, beta, gamma, inv-gamma, exponential, laplace, cauchy, beta-uniform, piecewise-uniform, mvnormal, dirichlet, chi-squared, f-dist (both branches), gumbel, levy, log-normal, logistic, nakagami, pareto, student-t, triangular (incl. the mode at a bound), weibull, bernoulli, categorical, categorical-logits, uniform-discrete, geometric, poisson, binom, neg-binom.
- **Supports:** guards are −Inf outside; xlogy and xlog1py give 0 at 0·log 0.
- **Samplers:**
  - Read against their references: Marsaglia–Tsang with the α < 1 boost; PTRS and BTRS step by step against Hörmann; geometric by inversion.
  - Read and checked algebraically: logistic as a difference of Gumbels; F and t from gammas; nakagami, pareto and weibull by transform; the triangular inverse CDF; laplace's sign convention.
  - The suite's moment tests cover the rest, except small shapes (see P2-01).
- **mvnormal's condition check:** H⁻¹ = MᵀM for M = L⁻¹ D^½, so `Σ m²` is trace(H⁻¹) (derived).
- **Distribution GFI methods (read):** generate, update, regenerate, assess, propose and project; their weights, discards and retdiffs (no-change exactly when the value is kept).
- **dist DSL inversions (derived):** each step's inverse and log-Jacobian (+, −, ×, ÷ on either side, negate, exp, log) and their scoping.
- **Host families (read):** quantile draws from the key and discrete non-integers at −Inf; Commons Math's hypergeometric argument order.
- **Backend ops (read):**
  - special.clj's erf, erfc and erfinv (Halley in the erfc form in the tails);
  - gen.array's lgamma (Lanczos plus reflection, poles), digamma, log-beta, logsumexp (−Inf, +Inf, NaN), logaddexp, softmax and log-softmax;
  - JVM kernels (rint rounding, NaN-propagating max, argmax);
  - MLX's float64 exact routes.
- **Not findings (JAX's float32 semantics, noted):** `bernoulli(p)` for p < 2^-23 is true with probability ≥ 2^-23 in float32; Gumbel and exponential draws are truncated at about 15.9 in float32.

## Coverage

- `src/gen/combinators.clj`: 1–792
- `src/gen/dist.clj`: 1–797
- `src/gen/dist/core.clj`: 1–364
- `src/gen/dist/dsl.clj`: 1–204
- `src/gen/dist/host.clj`: 1–116
- `ext/commons-math/src/gen/dist/commons_math.clj`: 1–66
- `src/gen/random.clj`: 1–183
- `src/gen/random/threefry.clj`: 1–124
- `src/gen/array/special.clj`: 1–113
- Backend ops the layer rests on:
  - `src/gen/backend/jvm.clj`: 1–905;
  - `src/gen/backend/mlx.clj`: 610–640 (unary routes; the op table at 598–613 by search), 742–746 (logaddexp, by search), 1185–1265 (random-uniform);
  - `src/gen/backend/mlx/exact.clj`: 1–162;
  - `src/gen/array.clj`: 415–540, 735–765 (softplus, logaddexp, xlogy, xlog1py, lgamma, digamma, log-beta, logsumexp, softmax, log-softmax).
- Context, read (not under review):
  - `src/gen/dynamic.clj`: 178–330 (site keys, compatible-callee?, run-call, splice-call) and 355–392 (ω);
  - `src/gen/gfi.clj`: 250–470 (revise, generate-again, run-op, the key source);
  - `src/gen/inference/mh.clj`;
  - `test/gen/laws.clj`, `combinators_test.clj`, and parts of `dist_test.clj` and `genjl/mixture_test.clj`.
