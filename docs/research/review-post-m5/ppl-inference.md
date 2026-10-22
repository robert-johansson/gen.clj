# Post-M5 review (PPL soundness): P3, inference, gradients and learning

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The inference layer is mostly sound. I derived the weight of every
estimator and every move against its algorithm's correctness argument: IS,
the resamplers, the particle filter's operations (priority, optimal resize,
coalesce, introduce, views), MH in its three forms, kernel reversals, HMC,
MALA, MAP and elliptical slice, the internal and exact proposals,
translators and SMCP3, VI and VIMCO, the optimizers and the tape's VJPs. They
hold, and the tests check most of them. I found seven defects, each run on the
JVM backend. Four give silently wrong numbers:

- **P3-01:** VI's model gradient (`:model-grads`) is biased whenever the
  model draws a latent that q does not propose and whose distribution
  depends on θ. Gen.jl does the same, but gen.clj's docstring promises an
  exact, unbiased gradient.
- **P3-02:** a `kern ^:trace` kernel statement may read the trace in its
  arguments, and nothing checks the dependent-mixture invariant there. A
  chain then targets the wrong distribution.
- **P3-03:** `retgrad` is paired with the return value by leaf order, so a
  map return value gets the wrong cotangents.
- **P3-04:** a trace kernel's Jacobian does not see what flows through a
  call's return value.

The other three are a loud error on a valid HMC run (a diverging
trajectory), a false invariant violation when a `kern` let or loop holds a
kernel, and a non-bijection check that misses inputs a move throws away.
That last gap is silent for SMCP3 without `:check?`.

## Findings

### P3-01: VI's model gradient is biased when the model draws latents that q does not propose
- **Severity:** high (silent; reached when the model's parameters are
  trained and q does not propose every latent whose distribution depends
  on θ)
- **Where:** `src/gen/inference/vi.clj:57-60` (`model-gradients`), used at
  `:80` and `:108`. The claim is in the namespace doc at `:21-28`, and
  `draw` (`:41-50`) documents that the model's internal proposal draws the
  latents q leaves out.
- **What:** The log weight is log p_θ(x, z, r) − log f_θ(r) − log q(z). The
  latents r that q does not propose come from the model's internal proposal
  f_θ, which depends on θ. `:model-grads` is the mean of ∇θ log p_θ(x, z, r),
  the gradient of the model trace's score (`param-gradients` of the model
  trace). That leaves out −∇θ log f_θ(r), and it leaves out the
  score-function (or pathwise) term for r's θ-dependent sampling. So it is
  neither the gradient of the iteration's estimate nor unbiased for the
  objective's gradient. The docstring's argument, "q's draws do not depend on
  θ", overlooks r.
- **Evidence:** In `OUT/scripts/P3-01.clj`, z1 ~ N(0, 1) is proposed by q =
  N(m, 1) with m = 0. z2 ~ N(θ, 1) comes from the model, y ~ N(z1 + z2, 1),
  and y = 3. Since z2's prior cancels against its proposal, ELBO(θ) =
  E[−½(y − z1 − z2)²] + const, and ∂ELBO/∂θ = y − m − θ = 3 at θ = 0. Run:
  - Over 2,000 iterations, the ELBO's `:model-grads` averages −0.011 ± 0.007
    (it estimates E[z2 − θ] = 0). θ would never move.
  - With one key, the objective is a function of θ. Its central difference
    is 3.29, while `:model-grads` gives −0.47.
  - VIMCO, the same way: central difference 1.57, `:model-grads` 0.69.

  The existing test `the-model-gradient-is-the-objective's` uses a model
  that q covers fully, so it cannot see this. Gen.jl 0.4.8's
  `black_box_vi!` also accumulates ∇θ log p(model trace)
  (`variational.jl:147`), so the bias is shared, but gen.clj's docstring
  asserts more.
- **Direction:** Either make it loud: with `:model-store`, an error when the
  model's generate sampled a choice that is in neither the observations nor
  q's choices (the weight then has an internal-proposal term). Or compute the
  full estimator: ∇θ[log p_θ(x, z, r) − log f_θ(r)] plus (log w − b)·∇θ
  log f_θ(r) for the internally proposed r (for VIMCO, weighted by w̃).
  Then fix the docstring.
- **Known?** new. genclj-xjte (completed) built the feature with the
  claim.

### P3-02: a `kern ^:trace` kernel statement may read the trace in its arguments, with no invariant check
- **Severity:** high (silent; the docstring invites this use)
- **Where:** `src/gen/inference/kernel.clj:227`, `traced-form`'s default
  branch: `(~apply! (checked-kernel ~(at form) '~form))`. The docstring at
  `:242-247` says "tr is the trace as it is when each statement is
  evaluated … in a kernel statement's arguments and in the loops,
  conditions and lets". The DESIGN §9 entry is "Kernels that read the
  trace".
- **What:** A kernel statement whose arguments read `tr` is a dependent
  mixture K_{a(t)}. The thesis's Prop. 3.4.1 (p. 99) makes it stationary only
  if a(t) = a(t′) whenever K_{a(t)} moves t to t′. `kern ^:trace` checks
  that invariant for loops, branches and lets (`:175-184`), but not for
  kernel-statement arguments. The thesis (p. 101-102: "arbitrary
  deterministic pure Julia expressions may appear in … the arguments to
  stationary kernel application") and Gen.jl (`kernel_dsl.jl`'s
  `T_ ~ k_(T_, args__)` branch) leave the same hole. The reversal has the
  same problem.
- **Evidence:** `OUT/scripts/P3-02.clj`, two cases:
  - **A state-dependent selection.** Model: flag ~ bernoulli(0.5),
    x ~ bernoulli(0.5), no observations, so the target has P(flag) = 0.5.
    `(k/mh (sel/select (if (tr :flag) :x :flag)))` makes flag = true
    absorbing. After 20 steps, 2,000 chains give P(flag) = 1.0. The same
    choice written as a `let` is caught loudly ("changed what it reads").
  - **A step size read from the trace, the realistic trap.** Model:
    x ~ N(0, 1), so E[x²] = 1. `(k/mh walk [(+ 0.5 (abs (tr :x)))])`, with
    the scale passed as a proposal argument, gives E[x²] = 0.866 ± 0.0025
    over 8 chains of 50,000 steps. The backward is assessed with the old
    argument. The same proposal reading `(tr :x)` itself gives
    0.997 ± 0.004.
- **Direction:** Check a kernel statement whose form reads `tr` the way a
  let is checked: read the statement's arguments again after the
  application, as host control values, and compare. One way is to expand
  such a statement as a checked construct over its argument values. The
  other is to refuse `tr` in a kernel statement's arguments, pointing to
  `let`. Either way, the docstring should state the invariant.
- **Known?** new. genclj-jo7m's test "a kernel's arguments see the trace as
  the statements before it left it" asserts the read but no check.

### P3-03: `retgrad` is paired with the return value by leaf order, not by structure
- **Severity:** high (silent wrong gradient for a map return value; silent
  truncation when the trees differ)
- **Where:** `src/gen/gradients.clj:42`, `(map (fn [g v] …)
  (array/tree-leaves retgrad) (array/tree-leaves rv))`, the objective of
  `choice-gradients` and `param-gradients` (combinators included). The
  docstring at `:13-15` says "retgrad (a cotangent of the return value, of
  the same tree shape)".
- **What:** `tree-leaves` walks a map in iteration order. An array map
  iterates in insertion order, so a `retgrad` written with its keys in
  another order than the body built the return value pairs each cotangent
  with the wrong leaf. `map` also stops at the shorter sequence, so a
  `retgrad` with fewer leaves (a scalar for a vector, a missing entry)
  silently drops cotangents. The same class of bug was found and fixed in
  `array/compile`'s argument maps (2026-10-19, docs/array-api.md §8).
- **Evidence:** `OUT/scripts/P3-03.clj`. Model: x ~ N(0, 1), returning
  `{:b 2x :a x}`. With `retgrad {:a 1.0 :b 0.0}`, the objective is log N(x)
  + x, so ∂/∂x = 1 − x = 0.5 at x = 0.5. Results:

  | `retgrad` | ∂/∂x | Correct? |
  |---|---|---|
  | `{:a 1.0 :b 0.0}` | 1.5 | no: 1·(2x) was used |
  | `{:b 0.0 :a 1.0}` (the body's key order) | 0.5 | yes |

  For a return value `[x 3x]`, a `retgrad` of `[1.0]` or `1.0` gives 0.5
  with no error.
- **Direction:** Walk `retgrad` and the return value together by structure:
  map keys by key, vectors by position, `nil` for "no cotangent here". A key
  in one tree only, a length mismatch or a leaf against a node is a loud
  ex-info (rule 6).
- **Known?** new.

### P3-04: a trace kernel's Jacobian does not see through a call's return value
- **Severity:** high (silently wrong log |det J| in one shape, a false
  "not a bijection" in another; for kernels that read a call's value with
  `(tr addr)`, which the namespace doc offers)
- **Where:** `src/gen/inference/trace_kernel.clj:86-88`. The kernel is
  assessed on `(assoc trace :choices mc')`.
- **What:** The Jacobian substitutes tracked values for the trace's choices
  only. `(tr addr)` at a call reads that call's subtrace `:retval`
  (`src/gen/trace.clj:35-38`), and `(tr [])` and `:retval` read the trace's
  own. These stay the old, untracked values, so every derivative that flows
  through them is lost.
- **Evidence:** `OUT/scripts/P3-04.clj`. Model: x ~ N(0, 1) inside a call
  at `:sub` that returns it, then y ~ N(x, 1). Two cases:
  - **The scaling move** x′ = x·e^u, u′ = −u, whose log |det J| = u (here
    1.156). Written with `(tr [:sub :x])`, `log-abs-det` gives 1.156.
    Written with `(tr :sub)`, the same number, it throws "maps 1 dimensions
    to 2", and so does `tk/mh`.
  - **Silently wrong.** The call returns s = e^x, and the move is
    x′ = x·s·e^u, u′ = −u, so log |det J| = log(1 + x) + x + u = −0.292 at
    x = 0.3. Reading s from the choice gives −0.292. Reading it from the
    call gives −0.554, a log 1.3 too little, with no error.
- **Direction:** Hand the kernel a trace that is consistent with the
  substituted choices, the one `generate` gives on `mc'` with every choice
  constrained, under the same tape and `with-exact-density`. Calls' return
  values are then tracked too. This costs one generate per Jacobian row,
  next to the assess already run there. If that is too dear, refuse reads of
  call values under the Jacobian.
- **Known?** new.

### P3-05: a diverging HMC trajectory throws instead of being rejected
- **Severity:** high (loud: an error where the program is valid)
- **Where:** `src/gen/inference/mcmc.clj:153-165` (`leapfrogs`), through
  `log-density` (`:94-113`) into assess's NaN check.
- **What:** When the leapfrog is unstable (a step beyond about two
  posterior standard deviations), the state grows geometrically. Once
  values overflow, Inf − Inf gives NaN. The next gradient evaluation then
  assesses a NaN value, and the move throws "assess: a constrained value is
  NaN" instead of rejecting. A diverging proposal has acceptance
  probability 0 (Stan and PyMC reject it; Gen.jl's `log(rand()) < NaN` is
  false). One divergence ends a whole `k/run`. It is not a modeling error:
  the start trace is fine. DESIGN §9's rule that NaN constrained values are
  errors stays right for user values; HMC reaches it only because it does
  not stop at divergence.
- **Evidence:** `OUT/scripts/P3-05.clj`.
  - x ~ N(0, 1) with eps = 3: L = 40 and L = 120 are rejected (log α of
    −6e65 and −4e199), and L = 400 throws.
  - At a common setting, a posterior sd of 0.003, eps = 0.1:
    - L = 50: rejected, log α −1.3e305.
    - L = 100: rejected, log α −Inf.
    - L = 150, 200 and 300: all throw.
  - Float32 on MLX overflows at 3.4e38, so it throws at much smaller L
    (not run, MLX not allowed).
  - Leaving a support (gamma) is handled: rejected, 200 of 200 runs.
- **Direction:** In `leapfrogs`, stop at the first non-finite position,
  momentum or gradient and return a marker that `finish` turns into
  log α = −Inf (rejected, and reported, e.g. `:diverged? true`). The
  NaN-at-the-start error (an impossible trace) stays loud.
- **Known?** new.

### P3-06: `kern ^:trace` compares control values with `=`, so a kernel built again in a let or loop is a false violation
- **Severity:** high (loud: an error where the program is valid)
- **Where:** `src/gen/inference/kernel.clj:179` (`(= read again)`) and
  `control-value` (`:124-131`), which leaves records and fns as they are.
- **What:** A let's values and a loop's collections are read again after
  the body and compared with `=`. A `Kernel` (or any closure) built by that
  read is a new fn instance each time, so `=` is false even when the trace
  did not change. Binding a kernel in a let is a natural way to build it
  once and apply it several times.
- **Evidence:** `OUT/scripts/P3-06.clj`:
  - `(let [step (k/mh (sel/select :x))] step step)`, which reads no
    trace at all, throws "changed what it reads".
  - `(doseq [step (mapv k/mh [sel-x sel-scale])] step)` throws the same.
  - `(let [s (sel/select :x)] (k/mh s) (k/mh s))` works.
- **Direction:** Compare with `gfi/same-value?`, which compares fns by their
  closed-over values and records field by field (DESIGN §9, "The maker").
  Or compare only the host data that the construct read from `tr`. That is
  also the mechanism P3-02 needs.
- **Known?** new.

### P3-07: the Jacobians leave out inputs a move throws away, so some non-bijections pass silently
- **Severity:** medium (an invariant the docstring says is checked; silent
  for SMCP3 unless `:check? true`, which catches it)
- **Where:**
  - `src/gen/inference/trace_kernel.clj:101-104`. A trace choice that no
    output depends on is dropped as "left alone", even when the update
    rewrites or discards it.
  - `:74-75`, an early return of 0.0 whenever there are no continuous
    outputs.
  - `src/gen/inference/transform.clj:259-266` and `:278`: a transform's
    inputs are only its continuous reads.

  The namespace doc at `trace_kernel.clj:35-38` (and DESIGN §9, "Trace
  kernels") says moves that are not bijections "make the matrix non-square
  or singular, a loud error".
- **What:** A continuous trace choice that the update overwrites or removes
  belongs to the move's domain. If no output depends on its old value, the
  move is not injective. The code does not count it as an input, so the
  dimensions still match and log |det J| comes out as if the move were a
  bijection. The early return also skips the check when the kernel's
  continuous randomness feeds only discrete outputs. An MH move usually
  fails later on its reverse assess. For SMCP3 the backward kernel `l` is
  only assessed, so nothing fails.
- **Evidence:** `OUT/scripts/P3-07.clj`. Model: mu ~ N(0, 1),
  y ~ N(mu, 1), y = 0.8. Particles come from the exact posterior, and the
  SMCP3 step goes to the same model, so a valid move has E[exp w] = 1.
  - A k that sets mu := u, u ~ N(0, 1), with no backward choices, and an l
    that makes none: `log-abs-det` returns 0.0, no error. E[exp w] is
    13.1, 6.8 and 7.1 at 2,000, 20,000 and 200,000 draws: the weight lacks
    log N(mu_old), and its expectation diverges.
  - The swap (mu, u) → (u, mu) gives 1.008.
  - The same move as a transform under `updating-translator` (mu written
    from aux u, never read): E[exp w] = 6.4.
  - u ~ N(0, 1) with only flag′ = u > 0 as output: 0.0, no error.
  - `:check? true` catches the k/l case ("does not give back the forward
    kernel's choices").
- **Direction:**
  - In `tk/log-abs-det`, leave out a trace column only if the update
    neither rewrites nor discards it (retained and not in the update). A
    forgotten value then makes the matrix non-square.
  - Drop the early return when there are kernel or consumed continuous
    inputs.
  - For transforms: an overwritten or discarded continuous model-in choice
    that the body never read is a loud error. That is known where
    `retained?` is (updating, involution). For the general translator,
    every continuous choice of the old trace must be read or copied.
- **Known?** new. The law audit's planted "randomness thrown away" case
  (`trace_kernel_test.clj` `misuse-is-loud`) covers a constant output,
  not these.

## Decisions worth revisiting

- **DESIGN §9, "Kernels that read the trace".** The entry says the
  invariant is checked per construct, as Gen.jl checks it. Kernel-statement
  arguments are a construct in Prop. 3.4.1's sense (a dependent mixture
  over the argument's value), and they are the one left unchecked (P3-02).
  Gen.jl's and the thesis's omission is evidence, not proof. gen.clj
  already "always checks", so extending the check to them, or refusing `tr`
  there, completes the decision rather than reversing it.

## Checked and sound

Each item is marked with how I checked it: read, derived, or ran.

- **weights:**
  - `log-total`, `normalize`, `log-mean-exp` and `ess` (derived).
  - `search` gives a zero-probability sample an empty interval (read).
  - systematic, stratified and residual are unbiased (derived; residuals
    sum to n − Σ floor).
- **importance:** with a proposal, the weight is log p(t) − log q(choices) −
  log f(rest), via generate's weight (derived). Overlap is loud
  (`cm/merge` is loud too). Resampling: derived.
- **MH:**
  - The selection move's weight is regenerate's.
  - The custom proposal's ratio is update weight − fwd + bwd (derived; the
    same as Gen.jl's source).
  - Involutive MH passes the bwd choices under the proposal at (t′, args).
  - The key splits are independent, and `accept` handles −Inf and +Inf
    and is loud on NaN.
- **kernel:**
  - The reversal of a chain is its reversals in reverse order. A mix's
    reversal is the mix of reversals: p K = Σ w_i p K_i = Σ w_i p′ R_i
    (derived).
  - The `kern ^:trace` loop, branch and let checks match Prop. 3.4.1 and
    its loop corollary, the reversal reads each construct's control value
    from its own input, and the keys are folded per application (read and
    derived).
- **mcmc:**
  - HMC: the leapfrog, and log α = ΔU + ΔK with both kinetic terms
    (derived). A structure change at the tried values is loud, and leaving
    a support is rejected (ran).
  - MALA: the forward and reverse Gaussian densities (derived).
  - `map-optimize` matches Gen.jl's backtracking (read).
  - Elliptical slice matches Murray–Adams–MacKay. Its bracket shrinks
    toward θ = 0, where the likelihood ratio is 0 > log u, so it
    terminates (derived).
- **particle filter:**
  - step weights with a proposal, with and without a backward proposal
    (derived against `pf_update!`).
  - Stratified initialization: weight + log k, unbiased since each stratum
    is expected n/k times (derived).
  - `resample`: the estimate is invariant and unbiased under priorities
    (E = Σ π_a W_a/π_a = 1) and under resize (derived).
  - Optimal resampling: the threshold search, c, the weights summing to n,
    and unbiasedness (derived, the docstring's example recomputed).
  - The estimate is invariant under coalesce, replicate, dereplicate
    (`:sample` keeps the group mean) and `update-view`'s shift. Introduce
    is unbiased: E = (n_old Z + n Z)/(n_old + n) (derived).
  - `reweighting` by selection is regenerate's weight, a valid SMC-samplers
    increment (derived).
- **internal proposal:**
  - generate: log p − log q − log f.
  - project replays q and returns +Inf where q cannot reach the trace,
    correct as p/0.
  - update: fresh choices are reproposed by q∘f, with weight w − score.
  - regenerate: Eq. 4.1, the old trace weighed under the old args (derived).
- **exact proposal:**
  - The weight logsumexp(s) − logsumexp(r) is right even when the other
    choices depend on the enumerated ones. r marginalizes the forward
    draw's c₀, so q(o, c | σ) = Σ_c′ f(o, c′ | σ)·P(c | σ, o) (derived).
  - The weight is log p(σ) when everything else is observed, and Gibbs
    regenerate has weight 0 (derived).
- **transform:**
  - The output specs: writes, copies of reads, and retained reads (derived
    per translator).
  - The general translator subtracts fwd (Eq. 3.12).
  - Extending: update weight − fwd + ldj.
  - Updating, SMCP3: update weight + bwd − fwd + ldj (derived).
- **trace kernel:** the weights of mh and smcp3 (derived). The tests' moves
  are right.
- **VI:**
  - The score-function ELBO gradient, unbiased with the leave-one-out
    baseline (derived).
  - The VIMCO coefficients (L − L₋ⱼ) − w̃ⱼ, from ∇ log w_j = −∇ log q
    (derived).
  - q's gradient stays unbiased when the model proposes some latents (the
    joint φ-score is ∇ log q).
- **learning:** SGD, GradientDescent's schedule and bias-corrected Adam,
  all ascending (read against Gen.jl). `lecture` scores q on its replayed
  choices (read).
- **tape:**
  - VJP rules checked by derivation: power, remainder, abs, tan, erf,
    erfinv, arctan2, where, max/min ties, prod (exact at zeros), cumsum,
    cumprod, take, take-along-axis, scatter-add, matmul, Cholesky (Murray
    2016's Φ form), the triangular and general solves, slogdet.
  - The backward pass processes every consumer before its input: reversed
    post-order (derived).
  - The tape declares no native ops, so `logaddexp` and `logsumexp` reach
    it as composites (read).
  - Custom VJPs are checked for count and shape in `array/custom-vjp`
    (read).
- **gradients:**
  - `log-abs-det-jacobian` is one reverse pass per row (read).
  - `continuous-selection` is loud on discrete choices.
  - `choice-gradients` holds the other choices fixed (read).

## Coverage

Every line of each file in the layer:

- `src/gen/inference/enumerative.clj` 1-367
- `src/gen/inference/importance.clj` 1-89
- `src/gen/inference/kernel.clj` 1-358
- `src/gen/inference/mcmc.clj` 1-237
- `src/gen/inference/mh.clj` 1-129
- `src/gen/inference/particle_filter.clj` 1-491
- `src/gen/inference/proposal.clj` 1-182
- `src/gen/inference/trace_kernel.clj` 1-163
- `src/gen/inference/transform.clj` 1-511
- `src/gen/inference/vi.clj` 1-157
- `src/gen/inference/weights.clj` 1-197
- `src/gen/gradients.clj` 1-128
- `src/gen/learning.clj` 1-150
- `src/gen/array/tape.clj` 1-488

Context read outside the layer:

- **gen.clj:**
  - `src/gen/gfi.clj` 55-80 and 437-575
  - `src/gen/trace.clj` 1-80
  - `src/gen/array.clj`: `tree-leaves`, `tree-map`, `item`, `->clj`,
    `logaddexp`, `custom-vjp`
  - `src/gen/array/dtype.clj` 60-120
  - `src/gen/array/protocols.clj` 1-140
  - `src/gen/choicemap.clj` 282-303
  - `src/gen/random.clj` 21-47 and 127-140
- **Gen.jl 0.4.8:** `src/inference/kernel_dsl.jl` and
  `variational.jl:98-178`, and `docs/src/ref/inference/mcmc.md`.
- **The thesis,** pp. 98-102.
- **Tests:** `test/gen/inference/{kernel,trace_kernel,vi}_test.clj`, in
  part.
