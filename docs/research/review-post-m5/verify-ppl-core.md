# Verification: ppl-core.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

I reran every script the reviewer wrote, on the JVM backend and one JVM at a time, from the snapshot worktree. I also wrote my own checks, each with a different model or route: `OUT/scripts/V-P1-01.clj` through `V-P1-05.clj`. Every cited line matches the snapshot. I derived every "expected" weight in the report again by hand, and all of them are right:

- −2π and −12.0 (P1-01);
- 0.5, −0.5, 4.5 and 1.5 (P1-02);
- −2.073 and −0.654 (P1-06).

I also searched beans for each finding and found no open bean covering any of them.

## Verdicts
| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| P1-01 | the dynamic retdiff, `diff/compute` and `consistent?` decide "unchanged" by `=` (0.0 = −0.0, a vector = a list), so the static walk and Unfold keep stale nodes | CONFIRMED | high → high |
| P1-02 | argdiffs cannot express a change in the number of arguments, so a static rest parameter or an estimated density keeps stale work | CONFIRMED | high → high |
| P1-03 | the retdiff's `=` on an infinite lazy return value with an unchanged prefix never returns | CONFIRMED | high → high |
| P1-04 | `realize!` forces only `LazySeq`, so a `Cons` onto a lazy tail or an `Iterate` keeps its trace calls for after the run | CONFIRMED | medium → **high** |
| P1-05 | the overlap check sees a call only through its choices, so a choice above an empty call passes | CONFIRMED | low → low |
| P1-06 | `gfi/project` of a trace given another gen-fn runs the new gen-fn's `-project` over the maker's records | CONFIRMED | low → low |

## Findings

### P1-01: CONFIRMED (high)

**What I read.**
- `retdiff` (`src/gen/dynamic.clj:655-668`): the `=` at :666.
- `diff/consistent?` (`src/gen/diff.clj:100`) and `diff/compute` (:127).
- `run-static` (:497-551): a node is kept when `unchanged? diffs in` holds and nothing reaches it.
- `unfold-revise` (`src/gen/combinators.clj:350-394`): a step whose state diff is no-change and that nothing targets is skipped.

The static walk itself compares values only by `identical?` (:478 in `computed`, :532 for a trace node). So the `=` reaches it only through a callee's retdiff, or through a user's `diff/compute`.

**Is it decided?** No.
- DESIGN §9's D1 entry (DESIGN.md:607) says "no-change when its return value is the previous one". It names no equality.
- The `same-value?` decision (DESIGN.md:468-490, the "maker" bullet; bean genclj-7xvz, review 6 R6-1) covers a different question: comparing a gen-fn's fields and a store. There the owner adopted exactly the principle this retdiff breaks: "as code holding them could tell … tells −0.0 from 0.0 … a vector against a list counts". Review 6's report (`scratchpad/review6/m4-review-6.md:34, 62`) found the same 0.0/−0.0 trap for `same-value?`.
- So P1-01 is the known trap, found again at an adjacent site that the fix did not reach. It is not a decided behavior.

**Reran `P1-01.clj`.** Its output matches the report:
- signed zero: static weight 0.0 against the derived −2π = −6.283; score −4.631 against assess −10.914;
- vector to list: weight 0.0 against −12.0;
- `(diff/consistent? 0.0 -0.0 no-change)` is true, and `(gfi/same-value? 0.0 -0.0)` is false.

**Derivation.**
- For the signed zero, the x terms cancel because N(0,1) is symmetric. What remains is lp(1; −π) − lp(1; π) = −½[(1+π)² − (1−π)²] = −2π.
- For the vector and the list, `(first (conj v 5.0))` is 1.0 for the vector and 5.0 for the list. So the weight is lp(0; 5) − lp(0; 1) = −12.5 + 0.5 = −12.0.

**Independent check `V-P1-01.clj`.**
1. **Unfold over a dynamic kernel, with no static language.** Step t's mean is atan2(state, −1) and its new state is x·0.0. Constraining step 0's x from 1 to −1 turns state 1 into −0.0. The kernel's retdiff is NoChange, so Unfold skips step 1.
   ```
   1. Unfold | states before [0.0 0.0] after [-0.0 0.0] | retdiff #gen.diff.NoChange{}
      update weight -6.283185307179586 | expected -12.566370614359172 | new score -12.707481467498702 | assess of its choices -18.99066677467829 | score - old score -12.566370614359172
   ```
   The expected weight is −4π: −2π from step 0 (mean π, x from 1 to −1) and −2π from step 1 (mean from π to −π). The Unfold's own retdiff is NoChange too, so a caller of the chain would also be told nothing changed.
2. **A sorted map against an array map with the same entries**, read by `(val (first m))`, in a static caller:
   ```
   2. sorted map -> array map | update weight 0.0 | expected (b's terms cancel) 1.5 | new score -3.612085713764618 | assess -2.112085713764618
      = : true | consistent? no-change: true | same-value?: false
   3. control, 2.0 -> 1.0 | update weight 1.5 | expected 1.5
   ```

**Nuance.**
- "The laws cannot see it" is slightly too strong. `sound-retdiff?` (`test/gen/laws.clj:101`) cannot see it, because it uses the same `=`.
- But `static-matches-dynamic` also compares the static and dynamic weights, and that comparison would catch the wrong weight on a model that produced a signed zero or switched between a list and a vector. No law model does either, so the gap is coverage rather than a blind spot of the law.

**Severity.** High is right: a silent wrong weight and a stale score in an unusual but legal use.

**Direction.** It is sound, but it is an owner call on D1's equality, and it costs precision.
- `same-value?` needs `counted?` for sequential values and stops after 256 values, answering "different".
- A kernel that returns a lazy seq, or more than 256 values, would therefore always get unknown-change. That is still sound, but Map and Unfold lose D1's skipping there.
- Comparing concrete JVM arrays bitwise keeps DESIGN's "Arrays are values" (DESIGN.md:673).
- Comparing lazy-backend arrays by identity also removes the host read that `MlxArray.equals` (`src/gen/backend/mlx.clj:106-111`) does inside a nested call's retdiff today. I read this but did not run it, since MLX is off limits here.

### P1-02: CONFIRMED (high)

**What I read.**
- `arg-diffs` (`src/gen/dynamic.clj:198-205`): it maps over the new args, with `(concat prev-args (repeat nil))`, so an added nil is `identical?` to nil.
- `rest-d` (:451): `(drop (count fixed) argdiffs)` is empty when the tail shrank to nothing, so `every?` holds vacuously.
- EstimatedGF (`src/gen/custom.clj:343, :357`).
- DeterministicGF (`src/gen/custom.clj:179`), which compares the counts.
- `check-argdiffs` (`src/gen/gfi.clj:511-516`): one argdiff per new argument.

**Is it reachable?** Yes, and through a documented use.
- DESIGN §9's G3 entry (DESIGN.md:635-638) says a rest parameter "covers calls, update and regenerate with any number of arguments".
- `genjl-test-suite.md:211` and `:583` say this was "(checked)". The check holds for the dynamic language only.
- The only static rest-parameter model in the laws, `rest-rebound` (`test/gen/static_test.clj:228`), keeps the number of arguments in its alternative args.

**Reran `P1-02.clj`.** All four cases match the report:
- spliced: 0.0 against 0.5;
- an added nil: 0.0 against −0.5;
- G3 at the top level: 0.0 against 4.5;
- estimated: weight 0.0, with the args still `[1.0 2.0]`.

**Independent check `V-P1-02.clj`.**
```
1. static | update weight 0.0 | regenerate weight 0.0 | expected both 2.0 | update's score -2.9189385332046727 | assess -0.9189385332046728
1. plain  | update weight 2.0 | regenerate weight 2.0 | expected both 2.0 | update's score -0.9189385332046728 | assess -0.9189385332046728
2. control [0.0] -> [0.0 5.0] | update weight -0.4999999999999999 | expected -0.5
3. estimated under a splice, 2 args -> 1 | update weight 0.0 | expected 1.5 | callee's args in the new trace [1.0 2.0] | score -2.9189385332046727 | assess -1.4189385332046727
```
- **Case 1 is stronger than the report.** It updates `[0.0 1.0 2.0]` to `[0.0]` with **unknown-change** for the one remaining argument, and still gets 0.0. No argdiff a caller can give avoids the bug, and regenerate fails the same way.
- **Case 2 delimits the finding.** A call that grows, given an honest unknown-change for the new argument, is right.
- **Case 3** is an estimated density spliced by a dynamic caller whose call shrinks. It needs the shared argument to be one object. Two literal `1.0`s in different branches are not `identical?`, so they get unknown-change; my first attempt got 1.5, the right weight, for exactly that reason.

**Severity.** High is right. The weights are silently wrong, but the use is unusual: `^:static` or an estimator, together with a call whose number of arguments changes.

**Two corrections to the direction.**
- **The run-op part alone does not fix the shrinking case.** Case 1 shows that unknown-change on every remaining argument still leaves `rest-d` at no-change. The necessary part is the count check in every "all no-change" test, as `custom.clj:179` does.
- **ω is not a soundness issue.** `keeps-omega?` on a count change only breaks DESIGN's "ω is drawn afresh" policy. ω's law does not depend on the args, so keeping ω is a valid move on the extended space.

The direction should also cover a missed instance at `proposal.clj:121` (see Missed).

### P1-03: CONFIRMED (high)

**What I read.**
- `retdiff`'s `=` (`dynamic.clj:666`): it runs for both update and regenerate (:690, :692).
- `holds-batch?` (:645): bounded by `(take realize-limit x)`.

**Is it reachable?** Yes.
- DESIGN §9, "Bodies run to completion" (DESIGN.md:662-671), and the `realize!` docstring ("the rest of a longer (or infinite) one stays lazy") make infinite return values legal.
- Two earlier reviews fixed hangs on exactly these: review 1 bug 10 and review 2 B7.

**Reran `P1-03.clj`.** It throws `OutOfMemoryError` within about 4 s. The full report's stack has `gen.dynamic$retdiff … dynamic.clj 666` under `DynamicGF _update`.

**Independent check `V-P1-03.clj`.**
```
1. control, update of :x0 -> {:weight 0.3715262984224659, :retdiff #gen.diff.UnknownChange{}} (0.0 s)
   heap before (MB) 87
2. (repeat x0), update of :y only -> still running after 10 s (10.0 s)
   heap after (MB) 91
3. (iterate f x0), regenerate of :y -> threw java.lang.OutOfMemoryError (2.6 s)
```
- With `(repeat x0)` the update **spins with a flat heap**. That is a silent hang, not a loud error: `Repeat` makes no new objects as it is walked.
- Regenerate fails the same way.
- When the first element changes, `=` stops at once (case 1).

**Severity.** High is right, at least: a valid program fails, and in the `repeat` case it never returns.

**Direction.** It is sound. `same-value?`'s `counted?` requirement makes any lazy seq "different" immediately, which is bounded.

### P1-04: CONFIRMED (medium → high)

**What I read.**
- `realize!` (`dynamic.clj:403-414`): `(instance? clojure.lang.LazySeq x)`. `list?` is false for a `Cons`, and nothing else is walked.

**Is it decided?** No.
- DESIGN §9 (DESIGN.md:662-665) promises that "lazy sequences in a body's return value are realized during the run". It describes the fixed failure as "a trace without its choices, silently".
- The docstring's "lazy sequences proper" is narrower than DESIGN, but it is not a recorded decision.

**Reran `P1-04.clj`.** Output as reported. `(cons 0.0 (map …))` and a bare `(iterate …)` give no choices and score 0.0. Generate with `[:x 0]` raises "not visited". Reading the elements raises the after-the-run error.

**Independent check `V-P1-04.clj`.**
```
(conj (map …) 0.0)       | class Cons | choices [[:x 0] [:x 1] [:x 2]] | score -4.076471988440374
[(cons 0.0 (map …))]     | class PersistentVector | choices [] | score 0.0
   generate with [:x 1] constrained: generate: the constraint at [:x 1] was not visited
(lazy-seq (cons …)), control | class LazySeq | choices [[:x 0] [:x 1] [:x 2]] | score -4.076471988440374
4. (conj (map … (range 100)) 0.0) | class Cons | choices after the run 32 of 100 | reading them all: trace or splice at [:x 32] after the generative function returned: a lazy sequence escaped the body (...)
```
- **A Cons inside a vector** is missed too.
- **`conj` onto a lazy map** calls `seq`, which realizes only the first chunk. With 100 elements, simulate's trace silently holds **32 of the 100 choices**, with the score of those 32.
- Prepending an initial state to a lazy map, `(conj (map …) x0)` or `(cons x0 (map …))`, is an ordinary Clojure idiom.

**Severity: raised to high.** By the PPL scale, high covers both "an error where the program is valid" and "a silent error confined to an unusual but legal use". This finding is both:
- generate and update raise "not visited" on a program DESIGN says is handled;
- simulate and propose return a silently partial trace.

Medium is for fragile-but-right paths, and this one is wrong.

**Direction.** It is sound: realize any `seq?` up to `realize-limit`. Two notes:
- Walking sets adds nothing. Putting a seq into a set hashes it, which realizes all of it during the run.
- Taking 10,000 elements of a `Repeat` or an `Iterate` on every run is the cost `LazySeq` already pays.

### P1-05: CONFIRMED (low)

**What I read.** `check-address!` (`dynamic.clj:48-61`).
- `visited` holds the call's path `[:a :b]`.
- A later `[:a]` passes: it is not in `visited`, it has no prefix in `visited`, and `has-submap?` sees no choices.
- The reverse order hits the prefix test.

**Reran `P1-checks.clj` §3.** It matches the report (`[[:a]]`, then "lies under the choice at [:a]").

**Independent check `V-P1-05.clj`.** It puts an empty Map at `[:a :b]`, then a choice at `:a`. Simulate passes. Both an update to a non-empty Map and a simulate with one raise "Address [:a] already holds nested choices". So the error depends on the arguments, and no weight is ever computed on the overlapping trace.

**Severity.** Low is right.

### P1-06: CONFIRMED (low)

**What I read.**
- `gfi/project` (`src/gen/gfi.clj:564-567`) goes by `:gen-fn`.
- `project*` (`dynamic.clj:768-772`) reads `::scores` and `::subtraces`, which a Map trace lacks. Hence the 0.0.
- `subtrace` and `revise` go by the maker.
- `proposal.clj:105` relies on projecting by `:gen-fn`.

**Reran `P1-06.clj`.** It matches the report: −1.419, which is g1's score, against g2's −2.073 and the revise weight −0.654. A Map trace given a dynamic gen-fn gives 0.0 against its score −2.088.

**Severity.** Low is right: the contract is undefined, and only internal code uses it.

**Direction: needs care.** If `project` went by the maker, the internal proposal would recurse forever:
- `InternalProposalGF`'s `-project` (`proposal.clj:159-160`) calls `proposal-weight`;
- `proposal-weight` projects `(assoc tr :gen-fn gf)`;
- that trace's maker is the wrapper itself, so `project` by maker would call the wrapper's `-project` again.

The report's exception, "unless the gen-fn is the maker's wrapper", names the reverse of this case: here the maker wraps the gen-fn. The docstring-only option is safe. The maker option needs `proposal.clj:105` to call `(gfi/-project gf …)` directly.

## Missed

**`regenerate-under` has the same count blind spot as P1-02** (`src/gen/inference/proposal.clj:121`). This is in P3's files.
- It keeps the trace when nothing is selected and every argdiff is no-change, without comparing the counts.
- It serves both `with-internal-proposal` and `enumerative/with-exact-proposal`.

`V-P1-02.clj` §4 uses a dynamic model `[x & more]` with an internal proposal, regenerated from `[0.0 1.0 2.0]` to `[0.0]` with the one honest no-change argdiff and nothing selected:
```
4. with-internal-proposal, [0.0 1.0 2.0] -> [0.0], nothing selected | regenerate weight 0.0 | its trace's args [0.0 1.0 2.0] | the model alone 2.0 | expected 2.0
```
The weight is 0.0, and the trace keeps the old args. The model alone gives the right 2.0.
