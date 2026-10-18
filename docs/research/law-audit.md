# Audit of the GFI-law tests against the thesis (2026-10-18)

An independent, read-only audit of the test suite and handler against
Cusumano-Towner (2020), pp. 46–124 and 141–170 (printed page = PDF page).
Status of each item is tracked in bean genclj-vq52; resolved items are marked.

## Errors

- **E1 (implementation).** `update` mishandled an address that is a choice in
  the old trace and a splice in the new one (or the reverse). `finish` tested
  "gone" by path in `:visited`, where both kinds count as visited, so the old
  record was neither subtracted nor discarded. Def 2.3.1 (p. 69): the discard
  is τ|_(A_τ'ᶜ ∪ A_σ); Alg 17 (p. 168). Counterexample: `(if (trace :s …)
  (trace :x …) (splice :x sub))`, update `:s` true → false gives weight 0
  instead of −log p(τ[:x]) and a discard without `:x`; the reverse update
  resamples `:x`. Fix: decide "gone" by record kind against the new state's
  own records. Related, lower severity: an old direct path `[:obs 0 :x]`
  followed by a new splice at `[:obs 0]` tracing `:x` resamples rather than
  retains (weights stay self-consistent).
- **E2 (laws stronger than the thesis).**
  - `identity-update-has-weight-zero` requires retdiff = no-change; Def 2.3.2
    allows ⊥ (unknown) at any time. It is a property of this implementation.
  - Several laws hold only for the forward-sampling internal proposal (one
    key per site; weight = project of constrained choices; the regenerate
    retained-ratio; project(none) = 0 and the split; the −fresh-lp terms).
    Under an overridden proposal (Alg 16, p. 152) or the optimal/conjugate
    proposal (pp. 145–146) they fail for correct code. Split gen.laws into
    general laws and forward-sampling laws.

## Weak properties (mutation-tested)

Mutations that passed the dynamic and inference suites: (A) regenerate
subtracts removed choices' log densities; (B) regenerate adds fresh choices'
log densities; (C) regenerate ignores new args for retained choices; (E)
update ignores new args for retained choices. combinators_test catches B and
E; A and C passed everything. Control: the printed Alg 5 sign (D) is caught.

- **W1.** The regenerate law always selects the canonically first address, so
  it never selects a structure-deciding address and never changes args. Use
  random subsets, include structure-deciding addresses, vary args.
- **W2.** `updates-are-reversible` always makes one constraint at the first
  common address; never constrains new or several addresses; never changes
  args; checks neither discard exactly. Draw random subsets of another run's
  choices (including addresses absent from the trace), vary args, assert both
  discards.
- **W3.** Models lack: an address that changes kind (E1), a spliced
  distribution (the parameter-rescoring bug passed every test), changed args.
- **W4.** Statistical MH/kernel tests use only fixed-structure models.
- **W5.** The resampler unbiasedness test uses n·pᵢ all integers, making
  systematic, stratified and residual deterministic. Use fractional n·pᵢ.

## Missing laws (most valuable first)

1. Trace consistency: for every t' from generate, update, regenerate:
   score(t') = assess(args(t'), choices(t')).weight and retvals equal
   (Def 2.1.16 p. 56). Catches stale scores (C, E, the distribution bug).
2. Update with changed args (Prop 2.3.1 for p₁, p₂): w = log p(τ';x') −
   log p(τ;x) − log q(fresh), and reversal restores t.
3. Update without fresh choices (constraints cover new addresses): w = Δscore
   exactly, reverse weight exactly −w, reverse discard = σ, discard =
   τ|_((A_τ∩A_σ) ∪ (A_τ∖A_τ')).
4. Regenerate (Eq 4.1 p. 144; Alg 18 p. 170): with changed args w =
   Σ_{a ∈ (A_τ∩A_τ')∖S} [log p(τ'[a];x') − log p(τ[a];x)]; unselected visited
   addresses keep values; selecting all with the same args gives 0.
5. Compositionality (Prop 2.3.2 p. 73): `hierarchical` equals its inlined
   version in update/regenerate weights and discards.
6. Stationarity on a stochastic-structure model: the thesis's p_mh with
   ρ = {c ↦ T}, exact posterior (1/6, 1/12, 3/4) (pp. 92–93), custom-proposal
   MH with q4_mh and selection MH with {a}. Check h∘h = id for involutions.
7. IS with a partial custom proposal (Alg 14 p. 147): unbiased in exp space,
   log w = log p(τ) − log q_ext − log q_int(rest).
8. Smaller: retdiff no-change ⇒ retval equal (Def 2.3.2); generate weights
   never NaN (Def 4.1.1); `checked` catches a proposal writing an observed
   address (Alg 5, p. 92).

## Catalog corrections (docs/research/genjl-parity.md §3)

- Law 5: the q(fresh) term is Gen.jl's extension; in the thesis an
  unconstrained new address makes update undefined (Alg 17, p. 168).
- Law 6: "weight −w" only without fresh choices; in general w_back = −w −
  log q(fresh). Add Prop 2.3.1's second half: the reverse discard is σ (σ ⊕
  fresh with Gen.jl's extension); it holds with changed args.
- Law 3: "weight equals score" for full constraints is from extended generate
  (p. 143). The thesis ignores extra constraints (p. 69, p. 143); gen.clj
  throws: a deliberate departure to record.
- Law 9: "selecting everything gives 0" needs q(·;x,{}) = p(·;x); fixed
  support per address is sufficient for a supportive density, not equivalent
  (pp. 148, 169); cite Alg 18 (p. 170).
- Law 10 errata: the 0.6 acceptance in the pp. 148–149 example contradicts
  Eq 4.1 and Alg 18 (correct: 1); in Alg 15 "r ≤ log w" means log r ≤ log w.
- Law 15 erratum: Alg 5 (p. 92) and Eq 4.15 print exp(log w − s'.logpdf() +
  s.logpdf()), swapping forward and backward terms; correct is log w +
  log q(σ';t') − log q(σ;t) (Algs 6, 9, 11). gen.inference.mh is correct.
- Law 11: project(t,A) equals the generate weight only when generate
  reproduces t; in general project(t,A) = log p(τ) − log q(τ|_Aᶜ; x, τ|_A).
- Laws 11, 12 and 18 are not in the thesis (18 could cite §3.4.4, pp. 98–100,
  Prop 3.4.1); the section title overstates.
- Law 14: an identity update does not have weight 0 with encapsulated
  randomness (p. 157): log w = log ξ(x,τ,ω')/ξ(x,τ,ω).
- Law 1: also cite DML restriction 2, unique addresses (p. 63).

Verified citations: Def 2.1.3 p. 49, Def 2.1.7 p. 52, Def 2.1.16 p. 56,
Def 2.3.1 p. 69, Prop 2.3.1 p. 70, Def 2.3.2 pp. 70–71, Eqs 2.10–2.12 p. 72,
Prop 2.3.2 p. 73, Alg 5 p. 92, Def 3.6.1 and Eq 3.13 p. 110, Alg 9 p. 111,
Def 3.7.1 and Eq 3.15 p. 121, Eq 3.17 p. 123, Def 4.1.1 p. 142, Eq 4.1
p. 144, Alg 15 p. 148, Alg 16 p. 152, Def 4.5.1 p. 155, Eqs 4.2–4.3 p. 156.

gfi.clj docstring nits: lines 22–23 say update/regenerate "assert" unchanged
args (nothing checks); line 83 writes q(τ;…) for q(υ; x, σ).

## Confirmed correct

Generate weights (p. 145, Alg 18); update weight and discard for fixed
structure, distinct appearing/disappearing addresses, new constrained
addresses, changed args with disappearing calls; regenerate (Eq 4.1, Alg 18);
project, propose, assess, retdiff soundness; selection, custom-proposal and
involutive MH; IS and SIR (Algs 13, 14); kernel reversals; the statistical
bands (E[w²] formula, χ²₅ threshold).
