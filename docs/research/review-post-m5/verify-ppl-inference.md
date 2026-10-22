# Verification: ppl-inference.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

Every reviewer script was rerun from the snapshot worktree on the JVM
backend (`nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/<id>.clj`, one
JVM at a time), and each finding got an independent check of my own,
`OUT/scripts/V-P3-0n.clj`: a different model, with the law stated directly
or the expected number derived by hand. Beans were searched (`model-grads`,
`model_update`, `retgrad`, `Jacobian`, `trace kernel`, `leapfrog`, `HMC`,
`diverg`, `NaN`, `same-value`), and none of the findings is an open bean.
None of them is in the earlier reviews either. P3-03 is the same class as
review-m2 finding 1 (`array/compile`'s map order), but it sits in another
function.

## Verdicts

| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| P3-01 | VI's `:model-grads` is biased for θ-dependent latents that q does not propose | CONFIRMED | high → high |
| P3-02 | `kern ^:trace` kernel-statement arguments that read `tr` are an unchecked dependent mixture | CONFIRMED | high → high |
| P3-03 | `retgrad` is paired with the return value by leaf order, not by structure | CONFIRMED | high → high |
| P3-04 | A trace kernel's Jacobian does not see through a call's return value | CONFIRMED | high → high |
| P3-05 | A diverging HMC trajectory throws instead of being rejected | CONFIRMED | high → high |
| P3-06 | `kern ^:trace` compares let and loop values with `=`, so kernels built in them are false violations | CONFIRMED | high → high |
| P3-07 | The Jacobians leave out inputs a move overwrites or discards, so some non-bijections pass | CONFIRMED | medium → medium |

Seven confirmed, none refuted or decided. No severity changed. Two of the
directions need amending before anyone implements them: P3-06's
`gfi/same-value?` and P3-07's column rule. P3-01's VIMCO direction needs a
precise coefficient. P3-05's fix is a NaN-semantics call for the owner.

## Findings

### P3-01: CONFIRMED

**Read.** `src/gen/inference/vi.clj:41-50` (`draw`: generate on observations
merged with q's choices, so latents q leaves out come from the model's
internal proposal), `:57-60` (`model-gradients`: `param-gradients` of the
model trace, which is ∇θ of its score), and `:80` and `:108`. The namespace
doc at `:21-28` reads: "q's draws do not depend on θ, so each is the exact
gradient of the iteration's objective estimate, and unbiased." DESIGN §9's
G4 entry ("Variational inference trains the model's parameters too",
DESIGN.md:1647-1660) and bean genclj-xjte's summary make the same argument
in the same words. None of the three states a restriction to models whose
latents q covers. The namespace doc's line 6 ("q … whose choices are the
model's latent ones") could be read as such a precondition. But `draw`'s
docstring explicitly handles latents q does not make, and the test
`models-latents-outside-q-come-from-the-model`
(`test/gen/inference/vi_test.clj:129`) makes that a supported, tested use.
So the case is legal, and nothing checks it.

**Derivation, the reviewer's model.** Latents: z1 is proposed by
q = N(m, 1), and z2 ~ N(θ, 1) comes from the model's prior (the internal
proposal). The data: y ~ N(z1 + z2, 1), y = 3. In log w, z2's prior cancels
against its proposal, which leaves log N(z1; 0, 1) − log q(z1) +
log N(3; z1 + z2, 1). So ELBO(θ) = E[−½(3 − z1 − z2)²] + const. With
z2 = θ + ε (pathwise), ∂/∂θ = E[3 − z1 − z2] = 3 − m − θ = 3 at m = θ = 0.
By the score function instead: E[f·(z2 − θ)] with f = −½(3 − z1 − z2)²,
which also gives 3. `:model-grads` estimates E[∇θ log N(z2; θ, 1)] =
E[z2 − θ] = 0. The reviewer's derivation is right.

**Rerun of `P3-01.clj`:**
```
elbo-gradient :model-grads mean, se over 2000 iterations: [-0.0111 0.0071]
vimco-gradient :model-grads mean, se over 2000 iterations: [0.8591 0.0122]
one key: central difference of the objective 3.2855  :model-grads -0.4744
VIMCO, one key: central difference of the bound 1.5653  :model-grads 0.6911
```

**Independent check, `V-P3-01.clj`: a discrete uncovered latent,** so only
the score-function term exists. The latents: z1 ~ N(0, 1), proposed by q at
the prior, so it cancels, and b ~ bernoulli(θ) from the model. The data:
y ~ N(b ? 1 : −1, 1), y = 1, at θ = 0.5. Then ELBO(θ) = c − 2(1 − θ), so
∂/∂θ = 2. For the K-sample bound (K = 10), I differentiated exactly the sum
over n ~ Bin(K, θ) of the b's that are true:
L(θ) = c + Σ Bin(n; K, θ) log((n + (K − n)e⁻²)/K).
```
elbo-gradient  :model-grads [0.0054 ± 0.0098]   exact ∂/∂θ 2.0
vimco-gradient :model-grads [1.4521 ± 0.0053]   exact ∂/∂θ 1.6225
host VIMCO: the direction's estimator [1.62196 ± 0.00052]  Σ w̃ s [1.44616 ± 0.00080]  exact 1.62246
```
The ELBO's is off by 200 standard errors, VIMCO's by 32.

**A nuance the report misses.** Gen.jl's `black_box_vimco!` computes its
weights with `assess` (`variational.jl:65`), so in Gen.jl uncovered latents
are a loud error for VIMCO. Gen.jl shares the bias only for BBVI
(`variational.jl:147`). gen.clj's `vimco-gradient` goes through `draw`
(generate), so it extends VIMCO to uncovered latents and is silently biased
there.

**Severity.** High is right: a silent error in a legal but unusual use
(training θ while q leaves out a latent whose distribution depends on θ).
With θ-independent uncovered latents, the model gradient stays unbiased.

**Direction.** It is sound for the ELBO, but its VIMCO part needs the exact
coefficient. "Weighted by w̃" is not enough. The K-sample bound's
θ-gradient is

Σ_j [(L − L₋ⱼ) − w̃_j] ∇θ log f_θ(r_j) + Σ_j w̃_j ∇θ log p_θ(x, z_j, r_j).

That is, q's VIMCO learning signal on the internal proposal's score, plus
the w̃-weighted joint score. The host check above confirms it (1.6220
against the exact 1.6225). Making the case loud whenever generate sampled a
choice would also refuse θ-independent uncovered latents, whose gradient is
fine. That is conservative but acceptable. The docstring and the §9 G4
entry both need the fix.

### P3-02: CONFIRMED

**Read.** `src/gen/inference/kernel.clj:227`: a kernel statement is
`(apply! (checked-kernel <form at @t> …))`, with no `construct` wrapper. The
docstring at `:242-247` says `tr` is read "in a kernel statement's
arguments", and the soundness comment at `:143-155` argues only about
loops, branches and lets.

**Prop. 3.4.1 and the thesis.** I read the thesis PDF, pp. 98-102. The
proposition requires θ_i(t) = θ_i(t′) whenever k_i moves t to t′. A kernel
statement whose argument a(t) reads the trace is the dependent mixture
K_{a(t)}, so the condition applies to it. The DSL's own list (p. 101-102)
says "Arbitrary deterministic pure Julia expressions may appear in (i) the
arguments to stationary kernel application", with no check, and its dynamic
checks are listed only for for, if-end and the two lets. Gen.jl's
`kernel_dsl.jl` matches: the `T_ ~ k_(T_, args__)` branch checks nothing.
So the hole is inherited, as the reviewer says.

**DESIGN §9.** The entry "Kernels that read the trace" (DESIGN.md:873-893)
scopes the check to "the body of each loop, branch and let", as Gen.jl
does. It does not address kernel-statement arguments as a mixture, and its
stated aim is "the thesis's invariant". I take the gap as an oversight, not
a decision. The reviewer rightly also lists it under "Decisions worth
revisiting".

**Rerun of `P3-02.clj`:**
```
kernel args read tr, P(flag) after 20 steps over 2000 chains: 1.0     (target 0.5)
the same through a let: loud: kern: the body of (let [a (if (tr :flag) :x :flag)] …) changed what it reads …
scale as a kernel argument read from tr: E[x²] 0.8660 ± 0.0025; read inside the proposal: 0.9967 ± 0.0042   (target 1)
```

**Independent check, `V-P3-02.clj`: the stationarity law stated directly.**
One kernel step applied to exact draws from the target (the prior, by
`gfi/simulate`) must leave the target's distribution unchanged.
- (a) The reviewer's flag kernel. Derived for the unchecked kernel: flag
  true stays true, and flag false turns true half the time, so P(flag) =
  0.5 + 0.25 = 0.75.
- (b) An MH walk with scale 1 + x² passed as a kernel argument, against the
  same scale read inside the proposal.
```
(a) P(flag) one step from the target: 0.74495 ± 0.0031   law 0.5, derived 0.75
(b) scale as a kernel argument:  E[x²] 1.0710 ± 0.0046   law 1.0
(b) scale read in the proposal:  E[x²] 1.0081 ± 0.0045   law 1.0
```

**Severity.** High is right. It is silent, and the docstring offers the
read. The silent cases are those where the kernel moves what its argument
read, which is less common than the benign `(sel/select [:z (tr :k)])`
whose kernel leaves k alone.

**Direction.** It is sound. Re-reading must compare what the statement read
from `tr` (or the argument values), not the rebuilt kernel. A rebuilt
kernel is a new closure, which is P3-06's false positive. Refusing `tr` in
kernel-statement arguments is simpler. It breaks only the genclj-jo7m test
that asserts the read.

### P3-03: CONFIRMED

**Read.** `src/gen/gradients.clj:42` pairs `(array/tree-leaves retgrad)`
with `(array/tree-leaves rv)` through `map`. `tree-leaves`
(`src/gen/array.clj:643-648`) walks a map by `vals`, in iteration order, and
`map` stops at the shorter sequence. Clojure maps are unordered, so
`{:a 1 :b 0}` is "the same tree shape" as the return value `{:b … :a …}`
under the docstring (`:13-15`). The mispairing is therefore a bug, not
misuse. The truncations break rule 6, since shape mismatches must be loud.

**Rerun of `P3-03.clj`:**
```
retval {:b 1.0, :a 0.5}
retgrad {:a 1.0 :b 0.0}: expected 1 - x = 0.5, got 1.5
retgrad {:b 0.0 :a 1.0} (the retval's order): got 0.5
retval [x 3x], retgrad [1.0]: got 0.5 (no error)
retval [x 3x], retgrad 1.0: got 0.5
```

**Independent check, `V-P3-03.clj`: through `param-gradients`,** the other
caller. The model returns `{:y w·x, :z 3w}` with x = 0.5, and the density
does not read the parameter w. So ∂/∂w of ⟨retgrad, rv⟩ is 3 for
`{:z 1, :y 0}` and 0.5 for `{:y 1, :z 0}`.
```
retgrad {:z 1.0, :y 0.0}  expected 3.0  got 0.5
retgrad {:y 0.0, :z 1.0}  expected 3.0  got 3.0
retgrad {:y 1.0, :z 0.0}  expected 0.5  got 0.5
retgrad {:z 0.0, :y 1.0}  expected 0.5  got 3.0
retgrad {:q 1.0 :y 1.0} (no :q in the retval): 3.5    (silently used)
retgrad {:z 1.0} (no :y): 0.5                         (paired with :y)
```
The mispairing hits array maps (eight keys or fewer), whose order is
insertion order, which is the common case. Hash maps with equal key sets
usually iterate alike.

**Severity.** High is right.

**Direction.** It is sound: walk the two trees together by key and
position, and make mismatches loud.

### P3-04: CONFIRMED

**Read.** `src/gen/inference/trace_kernel.clj:86-88` assesses the kernel on
`(assoc trace :choices mc')`. `Trace.invoke` (`src/gen/trace.clj:31-38`)
looks up the choice map first, then a call's `:retval` through `subtrace`.
That value, and the trace's own `:retval` and `:score`, stay the old,
untracked values. DESIGN §9's "Trace kernels" entry records the
substitution mechanism but does not consider call values. It is not a
decision about them.

**Rerun of `P3-04.clj`:**
```
reads the choice : expected log |det J| = u = 1.15645 , got 1.15645
reads the call's value : loud: … maps 1 dimensions to 2 …
MH with the call-reading kernel: loud: … maps 1 dimensions to 2 …
s = e^x from the choice : log |det J| = -0.29185  expected -0.29185
s from the call's value : log |det J| = -0.55422  expected -0.29185
```

**Independent check, `V-P3-04.clj`: the law stated directly.**
`log-abs-det` should equal log |det| of the move's actual map, which I
computed by central differences. Each perturbed input gets a consistent
trace (generate on the perturbed choice), so a call's value moves with its
choice. The model: x ~ N(0, 1) in a call returning s = x² + 1. The kernel:
u ~ N(0, 1), x′ = x + s·u and u′ = −u, with s read from the call. Derived:
log |det J| = log |1 + 2xu|.
```
derived log|1 + 2xu| = 0.672944   central differences of the move = 0.672944   tk/log-abs-det = 0.0
```
This is silently wrong, in a natural shape: a step scaled by a quantity the
model computes.

**Severity.** High is right.

**Direction.** It is sound: generate a consistent trace on the substituted
choices, under the tape, with `with-exact-density`. Alternatively, make
call-value reads loud under the Jacobian. The trace's `:score` and `(tr [])`
are covered by the same fix.

### P3-05: CONFIRMED

**Read.** `src/gen/inference/mcmc.clj:153-165`: the leapfrog never checks
finiteness, and each step's gradient goes through `log-density`
(`:94-113`), which is `gfi/assess`. The NaN check in assess on constrained
values (`src/gen/dist/core.clj:79-83`) then throws. `accept` (`:28-33`)
also throws on a NaN ratio, by design (bean genclj-er56: "A NaN ratio is a
loud error").

**DESIGN and the docs.** The only NaN decision is DESIGN.md:570-572: "NaN
parameters and NaN constrained values are errors too", next to the
Bernoulli entry. Its rationale is user data that would make every weight
NaN. No §9 entry, MCMC docstring or bean decides what a divergent
trajectory does (the searches covered `diverg`, `leapfrog`, `HMC` and
`NaN`). The behavior is inconsistent: at L = 100 the overflow stops at ±Inf
and is rejected, and at L = 150 it reaches NaN and throws. So it is not
deliberate. Gen.jl's `hmc.jl:66-67` (`log(rand()) < alpha`) rejects a NaN
alpha.

**Rerun of `P3-05.clj`.** The output matches the report:
- N(0, 1) at eps 3: L = 40 and L = 120 are rejected (log α −6.4e65 and
  −3.6e199), and L = 400 throws "assess: a constrained value is NaN".
- sd 0.003 at eps 0.1: L = 50 and L = 100 are rejected, and L = 150, 200
  and 300 throw.
- The gamma support case is rejected, never thrown.

**Independent check, `V-P3-05.clj`: Neal's funnel,** the textbook target
for occasional divergences, v ~ N(0, 3), x ~ N(0, e^{v/2}). HMC runs on
both choices from v = 0, x = 1:
```
L 10 eps 0.2 : {:steps 2000, :accepted 1780}
L 20 eps 0.3 : {:threw-at-step 33, :message assess: a constrained value is NaN, :accepted 30, :v -4.54, :x 0.0217}
L 10 eps 0.5 : {:threw-at-step 1309, :message assess: a constrained value is NaN, :accepted 881, :v -1.80, :x -0.0146}
```
A chain at ordinary settings, accepting 67% of its moves, dies at step 1,309
from a valid state. This is stronger than the report's evidence: one
divergence in a hierarchical model ends `k/run`. MALA runs on the same
funnel (`V-P3-05-mala.clj`; tau 0.05, 0.2 and 0.5, 3,000 steps each) never
threw, so the finding stays HMC's.

**Severity.** High is right (loud, on a valid program).

**Direction.** It is sound in intent, with three caveats.
1. Under `:compile?` (`array/compile`, one MLX graph) there is no
   data-dependent early exit, and a host read per leapfrog step forces an
   evaluation every step on a lazy backend. A masked freeze (`where` on
   finiteness, carrying a diverged flag) with one read in `finish` avoids
   both.
2. No NaN may reach `grad-v`, because assess throws on it before anything
   returns.
3. `finish` must skip `move-to` on a divergence, because update would throw
   on the NaN value too.

Per the project's practice, NaN and overflow semantics are the owner's
call, so the fix needs that decision before code.

### P3-06: CONFIRMED

**Read.** `src/gen/inference/kernel.clj:175-184` reads the construct's
values again and compares them with `=`. `control-value` (`:124-131`)
leaves records and fns as they are. A `Kernel` is a record of fns, so a
kernel rebuilt by the re-read is a new closure and never `=`. DESIGN §9's
"Values compare on the host" is about host reads, not about fns.

**Rerun of `P3-06.clj`:**
```
a selection in the let: true
a kernel in the let (it does not read the trace at all): loud: kern: the body of (let [step (k/mh (sel/select :x))] …) changed what it reads … read [#…Kernel{… fn__7001 0x3e0e3048 …}], then [#…Kernel{… fn__7001 0x7bc4ac3e …}]
a loop over kernels built in its collection: loud: kern: the body of (doseq [step (mapv k/mh [(sel/select :x) …
```
The printed values show identical kernels that differ only by instance.
Plain `kern` accepts the same let (`:121`). A let that binds a freshly
built generative function would fail the same way.

**Severity.** High, by the scale's definition (loud, on a valid program),
though at its low end: binding the selection instead is a trivial
workaround.

**Direction.** It needs amending. `gfi/same-value?` (`src/gen/gfi.clj:174`)
gives up after 256 values and answers "different". That is sound for its
purpose (re-running), but here "different" is a loud error. A loop over
`(range (tr :n))` with n = 1,000, or any large let value, would then be
falsely rejected. Use an unbudgeted comparison: `=` on host data, and
`same-value?`'s rule only for fns and generative functions. The
alternative, comparing only the host values the construct read from `tr`,
avoids the problem and serves P3-02 too.

### P3-07: CONFIRMED

**Read.**
- `src/gen/inference/trace_kernel.clj:101-104` keeps a model column only
  when some output row depends on it, whether or not the update rewrites or
  discards it.
- `:74-75` returns 0.0 whenever `outs` is empty.
- `src/gen/inference/transform.clj:278` takes a transform's inputs from its
  continuous reads only (`output-specs` at `:254-271`).

The trace-kernel namespace doc (`:35-38`) and DESIGN §9's "Trace kernels"
entry promise that "randomness thrown away" makes the matrix non-square, a
loud error. The early return contradicts that directly: u ~ N(0, 1) with
only flag′ = u > 0 returns 0.0. The rewritten-or-discarded case follows the
entry's literal rule ("It leaves out only the trace's choices no output
depends on"). That rule assumes such choices are left alone, which is false
when the update rewrites or discards them. It is a gap in the decision's
mechanism, not a decision that these moves are fine.

**Rerun of `P3-07.clj`:**
```
log |det J| of the forgetting kernel (no error): 0.0
E[exp w], the swap (a bijection), 20000 draws: 1.0082
E[exp w], the forgetting kernel, 2000 / 20000 / 200000 draws: 13.09 6.77 7.08
the forgetting kernel with :check? true: loud: smcp3: the backward kernel does not give back the forward kernel's choices
transform version, E[exp w] over 20000 draws: 6.437
u thrown away, only a discrete output: log |det J| = 0.0
```
I derived the forgetting kernel's weight: log N(y; u, 1) − log p(mu, y).
Under the posterior, E[1/p(mu, y)] = ∫ dmu / p(y) = ∞, so the reviewer's
"its expectation diverges" is right.

**Independent check, `V-P3-07.clj`: a discard instead of a rewrite,**
through the early return, with a finite expectation computed by quadrature
(no sampling).
- The model: flag ~ bernoulli(0.5), z ~ U(0, 1) when flag, and
  y ~ N(flag ? z : 0, 1), y = 1.
- The kernel: at flag true it sets flag false, which discards z, with no
  backward choices. At flag false it does nothing. It is not injective:
  every (true, z) goes to false.
- Derived: E[exp w] = P(false | y) + ∫₀¹ p(false, y)/p(y) dz =
  2 P(false | y). A valid move would give 1.
```
log |det J| of the discarding move: 0.0
P(false | y) = 0.41482   E[exp w] by quadrature = 0.82964   derived 2 P(false | y) = 0.82964   a valid move: 1
```

**Severity.** Medium is right. These are user errors in the move, which
`:check?` catches, so the gap is an unchecked invariant the docs say is
checked.

**Direction.** It needs one amendment. Counting every rewritten or
discarded trace choice as an input would also count observations that
SMCP3's `:new-observations` overwrites (`trace_kernel.clj:152`; they land
in the discard and no output depends on them), and so would falsely reject
every SMCP3 step that changes the data. The rule must exempt choices
written by `new-observations`. The transform part has the same issue: "for
the general translator, every continuous choice of the old trace must be
read or copied" would reject every translator whose old model has a
continuous observation, since the general translator does not know its
observations unless given `:prev-observations`. Dropping the early return
when there are kernel or consumed continuous inputs is fine as stated.

## Missed (optional)

- **P3-01:** Gen.jl's VIMCO cannot reach the case, since `assess` makes it
  loud. gen.clj's VIMCO is biased there silently, so "Gen.jl does the same"
  holds only for BBVI.
- **P3-05:** It is reachable at ordinary settings on a hierarchical target
  (the funnel run above), not only with a step 33 posterior sds wide.
