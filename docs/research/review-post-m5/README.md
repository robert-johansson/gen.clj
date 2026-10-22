# The post-M5 review: the merged list

genclj-ta2i, 2026-10-22, on the snapshot main 34e9a44.

**Method.** Eight fresh-context reviewers each read every line of one layer:
four for idiom (I1 arrays and backends, I2 the modeling core, I3
distributions and inference, I4 the LLM layer) and four for PPL soundness
(P1 the GFI core, P2 combinators, distributions and randomness, P3
inference, P4 the LLM layer). Eight fresh verifiers then tried to refute
every finding, reran every script and wrote their own checks.

**Totals.** 86 findings:
- 84 confirmed. Of these, I3-02 holds only in its performance half (its correctness half is refuted), and I1-13's rename is a recorded decision.
- 1 plausible (P2-07, whose cause is genclj-y0mp).
- 1 refuted (P2-10).

The verifiers added 6 findings of their own (marked V).

The reports are `idiom-*.md` and `ppl-*.md`, the verifications
`verify-*.md`, and the scripts `scripts/`. The briefs are `brief.md`,
`idiom.md`, `ppl.md` and `verify.md`.

## 1. Decisions for the owner

| # | Finding | The question | Recommendation |
|---|---|---|---|
| 1 | P2-09 (critical) | `fold-in` keeps 32 bits of its data (JAX's scheme for typed data). Recurse folds in node numbers that pass 2^32 (depth 32 for max-branch 2, 16 for 4, 11 for 8), so deep nodes share keys and repeat their draws; `next-key`'s stream repeats after 2^32. | Fold all 64 bits, `[hi32 d, lo32 d]`: today's keys, and JAX's, below 2^32, distinct beyond, and `fold-in = split[d]` for every d. Alternatives: reject data ≥ 2^32 loudly; or Recurse folds each child index into its parent's key, which changes every Recurse program's draws. |
| 2 | P1-01, P1-03 (high) | "Unchanged" (the retdiff, `diff/compute`, `consistent?`) is `clojure.core/=`: 0.0 = −0.0 and a vector = a list give wrong weights, and an infinite lazy return value hangs the update. | One equality, `gfi/same-value?`'s (by class and order, −0.0 ≠ 0.0, bounded), named in D1. Beyond its budget, or for an unrealized lazy value, the answer is unknown-change. That stays sound but loses D1's precision there. |
| 3 | P3-05 (high) | A diverging HMC trajectory throws "constrained value is NaN" (Neal's funnel at L 10, eps 0.5 dies at step 1,309). | Reject the trajectory (acceptance 0), as Stan does, and count divergences in the result so they aren't silent. NaN in the starting trace stays an error. |
| 4 | P3-01 (high) | VI's model gradient is biased for a θ-dependent latent that q doesn't propose (exact 2, code 0.005; VIMCO 1.452 against 1.622). Nothing documents the restriction, and such latents are a tested use. | Make it unbiased: add the score-function term for latents the model samples, with VIMCO's own learning signal (a host check gives 1.6220 against 1.6225). The alternative is a documented restriction with a loud error. |
| 5 | P2-02 (high) | A heterogeneous mixture of a discrete and a continuous family adds a pmf to a pdf (posterior 0.267 where 0.80 is exact). Gen.jl's types forbid it. | Forbid it: a loud error when the mixture is made. |
| 6 | P2-01 (high) | Gamma, beta and dirichlet draws that underflow to 0 aren't floored, though DESIGN §9 says every family is: a fix. But most of beta's +Inf scores are draws that round to exactly 1 (8,802 of 20,000 at 0.001, 0.001). | Floor the zeros as §9 says, and clamp beta below 1 to the largest float under 1, the mirror of the floor. |
| 7 | P4-06 (low) | `token/dist` gives −Inf to an integral double like 3.0, where every discrete family in gen.dist accepts it. | Accept it, as gen.dist does. |
| 8 | Idiom (§4) | Which structural refactors to do. | §4's recommendation. |
| 9 | I1, I4 | DESIGN §3.4's "mutation is confined to three places" no longer holds: caches and registries were added that don't change meaning. | Restate the sentence; the code is fine. |

## 2. Correctness fixes (no decision needed)

Ranked by severity. Each becomes a bean with tests and a planted bug.

| Finding | Severity | What | Where it stands |
|---|---|---|---|
| I2-01 | high (silent) | A callee whose kind changes drops the addresses both share. A Switch from a Map to an equivalent dynamic function loses its observations on an update of `:i` alone, and regenerate with nothing selected redraws them. Gen.jl keeps them. Blocker: two combinators' plain ex-infos must become `gfi/unvisited` first. | to file |
| I2-02 | high (silent) | The static analysis recognizes a nested `gen` only by its spelling. Through an alias (`dyn/gen`), its trace is hoisted into the outer function: one draw where there should be two. | to file |
| P1-02 | high (silent) | An argdiff can't say the number of arguments changed. A rest parameter (G3's own idiom), an estimated density and `proposal.clj:121` (V) keep stale work, weight 0.0 where 2.0 is right, whatever argdiffs are passed. | to file |
| P1-04 | high (silent) | `realize!` forces only `LazySeq`. A `(cons x (map …))` or an `(iterate …)` return value loses its trace calls: simulate records 32 of 100 choices, and generate fails. | to file |
| P2-03 | high (silent) | A `dist` body that broadcasts its draw is scored as several draws (a coin counted twice). Fix: compare whole values in the enumerable path. | to file |
| P2-06 | high (silent) | A homogeneous mixture of vector normals mixes each coordinate separately (−3.22 where −51.84 is right). Fix: sum over the component's own axes per its dims. | to file |
| P3-02 | high (silent) | In `kern ^:trace`, a statement's arguments that read the trace are unchecked against Prop. 3.4.1. P(flag) is 0.745 after one step, where stationarity needs 0.5. Fix: check them as lets are checked. | to file |
| P3-03 | high (silent) | A retgrad is paired with the return value by leaf order: a map with keys in another order gets the wrong gradients, and a short one is truncated. Fix: pair by structure, loudly. | to file |
| P3-04 | high | A trace kernel's Jacobian doesn't see through a call's return value: a false "not a bijection", or log\|det J\| off by log 1.3. | to file |
| P3-06 | high (loud) | `kern ^:trace` compares let and loop values with `=`, so a kernel bound in a let is a false violation. Not same-value? as is: its 256-value budget would reject large collections. | to file |
| rnbd | high (silent) | From memo.clj: categorical's log-prob gradient is NaN when any probability is 0 (log before the pick). The same trap may sit in bernoulli, binomial and others. | filed, genclj-rnbd |
| V (I3) | medium | `(mcmc/hmc tr sel key)` ignores its key (rule 2): two calls with one key give different traces. | to file |
| P3-07 | medium | The Jacobians ignore inputs a move overwrites or discards: E[exp w] ≈ 7 in SMCP3 without `:check?`. Must not count `:new-observations`. | to file |
| P2-04 | medium | A pushforward of a non-enumerable discrete draw gives mass off its lattice (scales under 1e-5, large shifts such as `(+ 1e6 (poisson 3))`). | to file |
| P2-05 | medium | Invalid parameters accepted, giving NaN or −Inf silently (rule 5). | to file |
| I1-05 | medium | gen.array.linalg gives a backend operands of two dtypes, and the JVM and MLX disagree on the result's. Also `la/inv`'s identity and the tape's linalg rules. | to file |
| I1-04 | medium | Three hand-written temporary-handle cleanups leak MLX handles on failure, and device handles are never freed. | to file (with I1-03's macro) |
| I4-03 | medium | Token-of-vocabulary checks restated: bytes' and grammar's throw IllegalArgumentException on a big integer. | to file |
| I2-06 | medium | Map, Unfold, Switch, Recurse and deterministic functions throw AbstractMethodError past six arguments; an estimated function isn't callable. | to file |
| P1-05, P1-06/I2-08 | low | The overlap check misses a choice above an empty call (one order); `gfi/project` of a trace given another gen-fn projects to 0. The project fix must change `proposal-weight` first. | to file |
| P2-08, P2-11, V (P2) | low | Float32: laplace's bound is −1; the binomial and poisson inversions count past n. | to file |
| I1-09, V (I1) | low | `diag` turns 0·Inf into NaN (it reaches `cholesky-logdet`); on the JVM, `la/solve` with a scalar right-hand side throws a NullPointerException. | to file |
| I4-07 | low | The trie's `remove*` overflows the stack, and two cleanup paths lose an error. | to file |
| P2-10 (rest) | low | The one gap left: the exact-discard law on a length change. | to file |
| V (P2) | low | `dc/check-probabilities`' fixed 1e-6 tolerance (P2-07's cause), after genclj-y0mp. | with y0mp |
| V (I1) | low, unconfirmed | Read only, not run on MLX: the batch backend reports no native ops, so `logaddexp` is MLX's own op unbatched but the composite under the batch tracer. | to check |

Already in progress:
- genclj-y0mp: the blocked float32 sum (m3-llama).
- genclj-rd26, 1lpj, gy6p and kdre (m3-grammar), covering P4-01 to P4-05 and I4-01 and I4-02.

## 3. Performance (measured, JVM backend)

| Finding | Cost | Fix |
|---|---|---|
| I1-01 | `array/lgamma` on a host number: 702 ns and 1.7 KB a call. A beta log density costs 2.5 µs against a normal's 74 ns. 11 families pay. | A primitive host lgamma/digamma, as erf has: 13–21 ns, bit-identical on 200,019 points. |
| I3-02a | A `dist` over poisson throws and catches an ex-info on every log-prob: 7 µs against 1.3 µs. | Decide "no finite support" without an exception. |
| I3-03 | A finite-support pushforward scores the draw once per support value: 5.6 ms at 1,000 labels. | One call over the support where the parameters allow. |
| I3-05 | The value-shape check is 10–12 % of an MH step. | Check once per site kind. |
| I3-04 | The scalar particle filter evaluates each particle after building all: 51 evaluations a step at 50 particles (on MLX, inferred from DESIGN's floors: n evaluation floors). | Evaluate inside each particle's move. |
| I2-05 | The handler's 16-key state map. | genclj-0jeu (open). |

## 4. Structure and idiom: for the owner to choose

All confirmed. I recommend doing them in this order.

**Do:** rule breaks and real duplication.
- I2-04: `gen.dynamic.core` calls backend protocols directly (rule 3); "a backend array" is written five ways.
- I3-11: gen.dist calls gen.array.protocols directly.
- I3-01: with-internal-proposal and with-exact-proposal are one record written twice; share an update-under, as regenerate-under is shared.
- I1-02: both backends repeat host-data parsing and eight shape checks; validate once in gen.array. Under the batch tracer their errors name the lifted axes.
- I2-07: the batched spot checks are written twice, with four copies of the mismatch error.
- I4-04: the host thread pool is written twice (llama, smc).
- I4-05: constrained works on token's private run state.
- I3-06: defdist silently drops a misspelled clause (`valdate`).

**Then, as their own beans:** the splits.
- I2-03: gen.dynamic (1,296 lines) splits out the body rewriter (`gen.dynamic.guard`) and the static code generator (into `gen.dynamic.static`); generate the 25 IFn arities.
- I1-03 with I1-04: gen.backend.mlx (1,651 lines) in layers, with the verifier's corrected order, after one `with-temps` macro replaces the ten hand-written cleanups. This is the shape genclj-mkgo waits for.
- I3-07: value shapes and discreteness declared at their definitions.

**Batch:** the minor points, one cleanup bean per layer.
- I1-06 to I1-14
- I2-09 to I2-14
- I3-08 to I3-17
- I4-06 to I4-11

These are stale docstrings, small duplications, naming, `case` without a default, redundant hints, and comments that narrate history.

**Leave, or decide separately:**
- I1-13's rename of gen.array.memory (DESIGN §9 decided the name).
- I3-09: moving the poisson and binomial samplers into gen.random would add to docs/array-api.md §7's approved list.
- I4's splitting of gen.llm.bytes.
- I3-10: the global compiled-step cache is documented, and an LRU would not fix the cyclic case.

## 5. Refuted

- I3-02b: a `dist` over a host family can't be written through the public API; it fails loudly at macroexpansion.
- P2-10: changed-argument laws do run on Map, Unfold and Switch. Only the exact-discard gap on a length change remains, filed in §2.
