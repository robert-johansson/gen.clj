# Verification: ppl-models.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

All eleven of the reviewer's scripts were rerun in one JVM (`OUT/scripts/V-rerun.clj`, output
`V-rerun.out`), and every number in the report reproduces exactly. My own checks are in
`OUT/scripts/V-P2-09.clj`, `V-P2-01.clj`, `V-P2-02.clj`, `V-P2-06.clj`, `V-P2-03.clj`,
`V-P2-04.clj` and `V-P2-11.clj`, run together by `V-run.clj` (output `V-run.out`), plus
`V-extra.clj` (`V-extra.out`). All of them ran on the JVM backend, one JVM at a time. One
check ran JAX 0.9.1's CPU build from GenMLX's venv, to settle how `fold_in` treats data of
2^32 and above.

## Verdicts

| ID | Claim (short) | Verdict | Severity (reviewer → mine) |
|---|---|---|---|
| P2-09 | fold-in keeps 32 bits of its data, so Recurse's deep nodes share keys | CONFIRMED | critical → critical |
| P2-01 | gamma, beta and dirichlet zero draws are not floored: +Inf and NaN scores | CONFIRMED | high → high |
| P2-06 | a homogeneous mixture of vector normals mixes each coordinate separately | CONFIRMED (the direction needs refining) | high → high |
| P2-02 | a discrete + continuous heterogeneous mixture adds a pmf to a pdf | CONFIRMED | high → high |
| P2-03 | a `dist` body that broadcasts its draw is scored as several draws | CONFIRMED (the direction needs refining) | high → high |
| P2-04 | a non-enumerable discrete pushforward gives mass to values off its lattice | CONFIRMED, and wider than claimed (one direction is unsound) | medium → medium |
| P2-05 | invalid parameters accepted, giving NaN or −Inf silently | CONFIRMED | medium → medium |
| P2-07 | dirichlet's 1e-6 simplex check fails float32 sums at large K | PLAUSIBLE; its cause is open bean genclj-y0mp | medium → low |
| P2-08 | laplace's lower bound is −1 in float32 | CONFIRMED | low → low |
| P2-11 | in float32 the binomial inversion can count past n | CONFIRMED (also on the JVM backend) | low → low |
| P2-10 | no law instance runs Map, Unfold or Switch with changed arguments | REFUTED (a small coverage gap remains) | low → low (the gap) |

Counts: 8 CONFIRMED, 1 PLAUSIBLE, 1 REFUTED, 0 DECIDED. One severity changed: P2-07, medium → low.

## Findings

### P2-09: CONFIRMED (critical)

**Read.**
- `src/gen/random/threefry.clj:85` masks the data: `(bit-and (long data) 0xFFFFFFFF)`.
- `src/gen/combinators.clj:655` defines `node-key` as `(fold-in (fold-in key node) part)`.
- `child-of` (`:504-508`) numbers child j of node i as i·m + j + 1.
- `gfi/next-key` (`src/gen/gfi.clj:451`) folds in an AtomicLong counter.

**Reran.** `P2-09.clj` reproduces every number:
- `fold-in(k, 5) = fold-in(k, 5 + 2^32)`.
- In the 40-deep binary chain, depths 32 to 40 all have v = 1.5664, a single distinct value.
- Of the 400 stopping chains, 317 stopped at depth 32 or less and 83 threw `long overflow`. None went deeper than 32, where 0.184 of them should have.

**Independent checks** (`V-P2-09.clj`):
- **JAX's semantics.** gen.clj's `fold-in(key 1, 5)` is `[3243370355 1313272271]`, the same as JAX 0.9.1's. JAX gives the same key for `np.int64(2^32 + 5)`, and raises only for a Python int (`jnp.asarray(data, dtype='uint32')`, `jax/_src/prng.py:1169-1175`). So the masking is JAX's own semantics for typed data, and it is the decided scheme (DESIGN §7, `docs/array-api.md:205-212`). The defect is that Recurse and `next-key` feed it numbers of 2^32 and above.
- **The threshold is the 2-adic valuation of max-branch, not only m = 2.** With max-branch 8, a child-0 chain's node residues mod 2^32 are fixed from depth 11 on. The values at depths 11 to 14 are all 0.4729.
  - In general, two nodes collide when their last ⌈32/v₂(m)⌉ child indices agree: 32 steps for m = 2, 6 or 10; 16 for m = 4; 11 for m = 8; 8 for m = 16.
  - It is not only chains that repeat one index. With max-branch 4, two different subtrees (the root's children 0 and 1, each a chain of child 3) reach nodes 10021590356 and 14316557652 at depth 17. These are equal mod 2^32, and their v is the same, 0.71748. At depth 16 the residues differ and the values differ.
  - For odd m, x ↦ m·x + c is a bijection mod 2^32, so collisions happen only by chance (about 2^-32 per pair).
- **next-key's stream.** After `gfi/seed! 5`, the key at counter 0 equals the key at counter 2^32, so the implicit stream repeats with period 2^32. That needs about 4·10⁹ keyless calls, so it is unusual.
- **Long overflow.** The error is Clojure's checked multiplication in `child-of`, which overflows past depth 63 for m = 2. It is loud, and it is not the bug. Gen.jl's Int64 numbering wraps silently at the same depth.
- **The direction.** A 64-bit fold, `threefry(k, [hi32 d, lo32 d])`, equals today's output on 43 values below 2^32, including 2^32 − 1. It differs for 5 and 5 + 2^32. It also makes `fold-in(k, d) = split(k)[d]` hold for every d (`random_test.clj:70`), and the JAX fixtures (d = 7, 123456) are unaffected.

**Documented limits.** None: neither the namespace doc, DESIGN §9's Recurse entry nor Gen.jl mentions a depth limit. Depth 16 with max-branch 4, or 11 with max-branch 8, is ordinary for parse trees or ASTs. Since nothing reports the repeated draws, critical stands.

**Correction to the claim.** The reviewer wrote "stops at depth 32 or 33". Depths 32 and 33 are the first pair that share a key (nodes 2^32 − 1 and 2^33 − 1), so the stopping chain either stops at 32 or never stops (and overflows). That matches their own data.

**Known?** New; `beans list -S fold-in` shows only closed beans.

### P2-01: CONFIRMED (high)

**The decision.** DESIGN.md:603-604 (owner, 2026-10-20) says: "a draw that underflows to 0, where a density can be +Inf, is floored at the dtype's smallest normal number". The sentence names no families.

**What is floored.** `floored` (`dist.clj:309-315`) was added by review 4's R8 (ee69d5c). It is applied in chi-squared, f-dist, levy, nakagami, student-t and weibull (`:321, 334-335, 366, 404, 430, 468`), the families tested by `the-further-families-at-their-edges` (`dist_test.clj:472`). The built-ins gamma (`:91`), beta (`:81`) and dirichlet (`:287`) are not floored.

So the decision's wording covers these three families and the code does not: this is a gap, not a decision. Gen.jl behaves the same way (Distributions' `GammaIPSampler` underflows too, and logpdf at 0 is +Inf), but the owner's rule here is the floor.

**Reran.** Every count reproduces:
- gamma(0.001, 1000): 963 of 2,000 +Inf scores.
- beta(0.001, 0.001): 1,253 +Inf scores and 448 NaN draws.
- dirichlet(0.01·1₁₀₀₀): 87 of 200 +Inf scores.
- chi-squared(0.002): 0.
- Selection MH on `:x` throws "The MH acceptance ratio is NaN". With lam floored, log-α is 0.038.

**Independent checks** (`V-P2-01.clj`):
- **The zero rate, derived.** pow(U, 1/α) rounds to 0 when U^(1/α) < 2^-1075, so for gamma(0.005) P(zero) = 2^(−1075/200) = 0.0241. Expected count: 482 ± 108 (5 SE). Observed: 510 of 20,000.
- **The laws, stated directly.** In a model `g ~ gamma(0.005, 1); x ~ normal(s, 1)`, take a trace where g = 0:
  - `update` to s = 1 (g kept, so g contributes 0) has weight **NaN**, where the exact weight is −0.6094;
  - `regenerate` of `:x` has weight **NaN**, where it should be 0;
  - the score and assess are both +Inf.
- **For beta, most +Inf scores come from rounding to 1, not from underflow.** Of 20,000 beta(0.003, 0.003) draws:
  - 220 are NaN (derived p₀² · n = 229);
  - 1,867 are exactly 0 (derived p₀(1 − p₀) · n = 1,910);
  - but 8,802 are exactly 1, because x/(x+y) rounds to 1 whenever y < x·2^-53.

  So the reviewer's "0 or 1 when one gamma underflows" understates the 1 edge. Their direction already clamps at 1 − ε/2, and that clamp is the essential part for beta. §9's floor covers only the 0 edge, so the top clamp is a new numerics edge, the owner's call.

**Direction: sound.** Floor gamma, and draw beta and dirichlet in log space. After the softmax, the floor is still needed, since `exp` of a very negative log g underflows.

**Severity.** High stands: a valid program hits a loud error (the MH NaN), and simulate gives silent +Inf scores and NaN values.

**Known?** New.

**On the reviewer's "decision worth revisiting".** It reproduces: from lam = `MIN_NORMAL`, the score is 698.44, and the random-walk chain stays at 1 distinct value in 2,000 steps. The proposed documentation note is reasonable.

### P2-06: CONFIRMED (high); the direction needs refining

**Read.**
- `dist.clj:740-742` stacks each component's elementwise `dc/log-prob` and takes the logsumexp per element.
- The sampler (`:724-738`) picks "one per element of the log density".
- The docstring (`:685-693`) says it "mixes (normal (mus k) (stds k))", and normal's docstring says its array density is the sum over elements.
- G6 (`genjl-test-suite.md:586`) says it mixes "the K components' densities of the whole value".
- Gen.jl 0.4.8's `logpdf(::HomogeneousMixture)` (`src/modeling_library/mixture.jl`) is the logsumexp over k of the base distribution's scalar logpdf of the whole x. Its `random` draws one k per value.
- No §9 entry concerns mixtures.

**Reran.** At (0, 10): the mixture scores −3.224, the mvnormal mixture −51.838. Of 20,000 draws, 9,988 split across the clusters.

**Independent checks** (`V-P2-06.clj`):
- **A 3-d case** (weights 0.3/0.7, clusters at 0 and 4, v = (0, 4, 4)). gen.clj scores −4.673, which is the per-coordinate value. The whole-value mixture gives −11.113.
- **dims is ignored in the semantics.** `(homogeneous-mixture normal [1 1])` with means [K d] gives exactly the same log-prob array as `(homogeneous-mixture normal [0 0])` with the transposed means [d K]. So declaring a component's rank 1 changes nothing, which is the clearest sign that this is not the intended behavior.
- **The heterogeneous mixture** `[normal normal]` scores −4.673 too.

**The direction would break batched parameters as written.**
- The reviewer proposes summing each component's log density over its `dc/value-shape`. For an elementwise family, though, that shape includes the batch axes of the parameters.
- Today, weights [B K] with means [B K] and dims [0 0] are B independent scalar mixtures, and this works row by row (`V-extra.clj`: [−1.6118 −3.5689], against −1.6118 and −3.5689 computed by hand). Component 0's value-shape there is [2].
- So the direction would sum the batch into one mixture. The axes to sum are the component's own, as declared by dims: the trailing ndim(component log-prob) − ndim(batch) axes, where batch is the broadcast of the weights' leading axes and the parameters' axes before the component axis. Only those should be reduced, and the sampler should pick one component per batch element.

**Severity.** High stands: a diagonal GMM written as Gen.jl writes it is silently a different model. The laws pass because the sampler and the density agree on the wrong model.

**Known?** New.

### P2-02: CONFIRMED (high)

**Read.**
- `heterogeneous-mixture` (`dist.clj:710-722`) checks only the shape of values.
- `dc/-discrete?` for a mixture (`:780`) is "every component discrete", so `[poisson normal]` reports false.
- Review 14's regression test builds exactly this pair (`dist_test.clj:300`), so it is treated as legal.
- Gen.jl's constructor is `HeterogeneousMixture(distributions::Vector{D}) where {T, D <: Distribution{T}}`, and its docstring says "the base distributions must have the same output type". So `[poisson, normal]` (Int and Float64) fails Julia's dispatch.
- No decision in §9.

**The exact posterior** (derived independently of the reviewer). With weight w on poisson(3), P(Y = 0 | w) = w·e⁻³, and the normal part has no atom. So P(w = 0.8 | y = 0) = 0.8·e⁻³ / (0.8·e⁻³ + 0.2·e⁻³) = 0.8. The reviewer's 0.8 is right, gen.clj gives 0.2666, and the rejection estimate of 0.810 ± 0.006 agrees with the exact value.

**Independent check** (`V-P2-02.clj`): spike-and-slab, a point mass at 0 (`uniform-discrete 0 0`) mixed with normal(0, 1), weights 0.3/0.7.
- log-prob at 0 is −0.5460, which is log(0.3 + 0.7·φ(0)), against the exact log 0.3 = −1.2040 with respect to counting + Lebesgue.
- At 0.5 it is exact (−1.4006).
- Simulated P(Y = 0) is 0.30011.
- With prior ½ on a spike weight of 0.9 or 0.1 and one exact zero observed, gen.clj's P(w = 0.9 | y = 0) is 0.672, where the exact value is 0.9.

**Direction.** Rejecting mixed kinds loudly is sound but breaks the test at `dist_test.clj:300` as written. That test is about dtype promotion, and could use a discrete pair with float and int values instead (a DSL pushforward and a count). The counting + Lebesgue alternative would keep spike-and-slab, a common prior, usable.

**Severity.** High: silent, in an unusual but legal use.

**Known?** New.

### P2-03: CONFIRMED (high); the direction needs refining

**Read.**
- `dsl.clj:176-200` covers the enumerable path: `maps-to` is elementwise `array/==` and the logsumexp is per element.
- The inversion paths work per element too.
- The defdist declares no value-shape (`:169`).
- Gen.jl's `normal(mu, 1) + [0, 1]` would be a MethodError (no scalar + vector in Julia), so gen.clj alone accepts such a body.

**Reran.** [1 2] scores −2.408, where the exact score is log 0.3 = −1.204. The impossible [1 0] scores −1.561. The off-line shifted pair has a finite density.

**Independent check, with a more natural body** (`V-P2-03.clj`): picking a vector label, `(dist pick-point [probs] (array/take points (d/categorical probs) 0))` with points [[0 0] [1 1] [2 5]] and probs [0.2 0.3 0.5].
- [1 1] scores −2.408, twice log 0.3.
- The impossible values [1 5] and [0 5] score −1.897 and −2.303, where the exact score is −Inf: element 0 matches one label and element 1 another.
- The same body with host labels (`nth` on a vector of vectors) is right: log 0.3 and −Inf, since `=` compares whole values.

**The direction would forbid a legitimate use.**
- Giving the pushforward the draw's value shape would turn vector labels from arrays into a loud error, although host labels support them.
- For the enumerable path, compare whole values instead: reduce `maps-to` with `all` over the label's own axes, the trailing ndim(f(x)) axes for a scalar support value x. This keeps batches, where v is [n 2].
- For the inversion paths, a body that changes the shape has no density in Rⁿ, so the loud rejection is right there.

**Known?** New.

### P2-04: CONFIRMED (medium), and wider than claimed

**Read.** `dsl.clj:185-194`: the inverse is rounded to an integer, and the result is accepted when |f(x) − v| ≤ 1e-5·(1 + |v|).

**Reran.**
- micro-count gives −1.496 at 2e-6, 1.5e-6 and 2.4e-6.
- The grid mass is 97.4.
- reciprocal(500) gives mass at 1/501.5.

**Independent check** (`V-P2-04.clj`). The tolerance is relative, so a large shift fails as a small scale does. `(+ 1e6 (poisson 3))` gives −1.496 at 1000002.5 and at 1000002.3, where the tolerance is 10 against a lattice spacing of 1. `(+ 3 (poisson 3))` accepts 5.00001. So the reviewer's "any scale below about 1e-5" should read "whenever 1e-5·(1 + |v|) exceeds a small fraction of the lattice spacing", which includes large offsets such as timestamps.

**The direction's second option is unsound.** Accepting values "within half the local lattice spacing" snaps every real v to its nearest lattice point: 2.4e-6 is within 0.5e-6 of 2e-6. That is exactly today's failure, where the mass over a fine grid exceeds 1. The tolerance has to be a small fraction of the spacing (or a few ulps of v, if every float32 body round-trips that well).

**Severity.** Medium stands. The density is right on values the model itself produces, and is wrong only on off-lattice constraints.

**Known?** New.

### P2-05: CONFIRMED (medium)

**Read.**
- `dist.clj:646`: neg-binom checks p with `check-probability`, so p = 0 is accepted.
- `:65-66`: uniform checks order only for host values.
- `:519-522`: uniform-discrete's `check-integer` (`core.clj:147-152`) checks host values only.
- `:625`: poisson checks host values only.
- `:633-634`: binom checks n with `check-integer`, host only (p is checked for arrays too).
- `:791`: piecewise-uniform's value-shape broadcasts batch axes.

**Reran.** Every case reproduces, including NaN from `poisson([-1])` and a finite −1.733 from `binom([2.5], 0.5)`.

**Gen.jl.** Distributions' `NegativeBinomial` requires 0 < p ≤ 1, so Gen.jl is loud here.

**Severity.** Medium stands: rule 6 forbids a degradation into NaN, and these are invalid inputs, not a valid program.

**Direction.** Sound.

**Known?** New.

### P2-07: PLAUSIBLE; medium → low; its cause is open bean genclj-y0mp

**Read.**
- `dist.clj:289-290` checks the sum against a fixed 1e-6.
- MLX's `contiguous_reduce` (`native/build/src/mlx/mlx/backend/cpu/reduce.cpp`, about lines 99-110) accumulates in N SIMD lanes, sequentially within each lane, as the reviewer says.

**Reran.**
- Sequential float32 sums: 15 of 200 fail at K = 1,000, and 126 of 200 at K = 10,000.
- Pairwise sums: none fail.
- float64: none fail.

**Is the emulation faithful?** As a bracket, yes. The draws are float64 gammas rounded to float32, not MLX's own bits, but they have the same distribution; MLX's lane-wise sum lies between the sequential and pairwise accumulations; and both the normalization and the check use the emulated sum, as on MLX.

**Not established.** No run on MLX (rationed), so PLAUSIBLE.

**Why the severity drops.**
- The cause is filed as genclj-y0mp ("MLX CPU float32: array/sum accumulates sequentially"). The owner decided it on 2026-10-22, after the snapshot (DESIGN §9 on main, 72b5dd3): MLX's CPU device will sum float32 in blocks, so the error grows with log n. Metal and CUDA already reduce in trees.
- Once y0mp lands, a draw fails only at K beyond about 10⁵ or 10⁶, through worst-case accumulation. What remains is float32 data summed elsewhere.
- So low after y0mp. The dtype-scaled tolerance in the direction is still a cheap guard.

**Known?** genclj-y0mp (in progress).

### P2-08: CONFIRMED (low)

**Read.**
- `dist.clj:120` passes the float64 constant `(Math/nextAfter -1.0 0.0)` as the low bound.
- `rng/uniform` (`random.clj:79-82`) computes max(low, u·(high − low) + low), and in float32 low rounds to −1.0f, so u = 0 gives −1.
- `rng/normal` (`:94-96`) picks its bound by dtype.
- JAX's `_laplace` uses `minval=-1. + finfo(dtype).epsneg` (`jax/_src/random.py:1849-1853`), so in float32 gen.clj differs from JAX.

**Reran.** Of 2^24 float32 uniforms, one is −1.0, and the transform there gives +Inf. With normal's bound, none is −1.

**Is the emulation faithful?** Yes: it is the same `rng/uniform` code path at float32 on the JVM backend, which rounds per op as MLX does. At u = −1 the transform is +Inf in any precision.

**Not covered by the +Inf decision.** That decision concerns values beyond the float range. A true float32 Laplace draw is at most about 16 from loc, so this +Inf comes from the bound alone. Low stands (about 1.2e-7 per draw on MLX).

**Known?** New.

### P2-11: CONFIRMED (low), also on the JVM backend

**Read.** `dist.clj:588-596`: for j ≥ n the term is 0 and the CDF stops growing, and nothing clips the count at n. With flip, the result is n − 40.

**Reran.** 220 of 1,670 (n, p) cases have a float32 CDF at n below 1 − 2^-23.

**Is the emulation faithful?** In structure, yes. It is not bit-exact to MLX's `exp` and `log1p`, so the list of cases would differ. The "P(k > n)" column (u_max − CDF) also ignores the uniform's 2^-23 grid, so (7, 0.41)'s 1.8e-7 is about 2.4e-7 on the grid. Neither changes the claim.

**Independent check, through the real sampler** (`V-P2-11.clj`). A float32 p on the JVM backend accumulates the CDF in float32, while u is the backend's float64.
- Rounding per op, as the backend does, 649 of 1,646 cases have a float32 CDF at n below 1.
- The worst case is binom(37, 0.25), with CDF 1 − 1.07e-6.
- Of 8·10⁶ draws, 12 are 40 > n (expected 8.6). Their log-prob is −Inf.

So the effect is not confined to MLX, though a float32 parameter on the JVM is unusual. Low stands: at most about 1e-6 per draw, always with a −Inf score.

**Direction.** Sound.

**Known?** New.

### P2-10: REFUTED (a small coverage gap remains)

The combinators_test entries (`combinators_test.clj:73-81`) do lack alternative arguments, but Map, Unfold and Switch do get changed-argument laws elsewhere:

- **Map and Unfold length changes:**
  - `encapsulated_test.clj:129-130` has Map [0.1 0.2 0.3] → [0.1 0.5 0.3], [0.1 0.2] and [0.1 0.2 0.3 0.4], and Unfold [3 0.0] → [2 0.0], [3 0.5] and [5 0.0].
  - These run under `laws/encapsulated` (`:135`), which includes `update-weight-is-the-projection-change`, `regenerate-weight-with-new-arguments-is-the-projection-ratio`, `updates-restore-their-choices`, `update-weight-is-the-score-change-less-fresh` and `regenerate-weight-is-the-retained-density-ratio`. All of them take `next-args`.
- **Switch index changes under the full `laws/all`:** `static_test.clj:238`, `switch-of-static` [0 0.5] → [1 0.5] through a splice. Also the `spliced` model in combinators_test, whose index comes from the traced `:which` that the laws constrain and select.
- **All three across stores:** `law_instances_test.clj:72-75` (Map length, Unfold n, Switch index), under `laws/across`.

**What is left.** The exact-discard law (`updates-are-reversible-with-exact-discards`) and `update-weight-is-the-score-change-less-fresh-choices` never see a Map or Unfold length change with a kernel without ω. Adding alternative arguments to combinators_test's entries is still a cheap improvement. The reviewer's own run of all 22 laws with alternative arguments passed, so no bug hides there.

## Missed (found while checking)

- **Poisson's inversion has P2-11's structure** (`dist.clj:556-561`, low). With λ < 10 and a float32 λ, the float32 CDF at 39 can sit below 1; at λ = 9.1 it is 1 − 3.0e-7. The value 40 is then drawn at about 3e-7 instead of 4e-14 (exact). Its density is finite (−31.09), so this is a silent sampling bias, not an out-of-support value. `V-extra.clj` found one 40 in 8·10⁶ draws on the JVM with a float32 λ (expected 2.4).
- **`dc/check-probabilities`** (`src/gen/dist/core.clj:167-177`) uses the same fixed 1e-6 sum tolerance as P2-07. A large float32 probability vector summed sequentially on MLX's CPU is then rejected loudly ("must be probabilities"). This has the same cause, genclj-y0mp.
- **P2-01's beta part:** as above, most +Inf scores come from values that round to exactly 1, so beta needs the top clamp, a new numerics edge for the owner.
