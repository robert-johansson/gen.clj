# Post-M5 review (PPL soundness): P1, the GFI core and the modeling languages

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The handler's transitions, the revise machinery, the static walk's skip rules,
the compiled score, the key discipline and the batch tracer all hold up
against the laws and the derivations. Six findings. All three high-severity
ones come from how the layer decides that "nothing changed".
- **P1-01.** The dynamic retdiff, `diff/compute` and `diff/consistent?` use
  `clojure.core/=`, which equates values that code can tell apart (0.0 and
  -0.0, a vector and a list). A no-change retdiff then lets the static walk
  (and Map/Unfold) keep stale work: a wrong update weight and a stale score,
  run on the JVM. The laws cannot see it, because their soundness check uses
  the same `=`.
- **P1-02.** Argdiffs cannot express a change in the number of arguments. A
  static function's rest parameter, the documented G3 idiom, and an estimated
  density then claim no-change when arguments were added or removed: wrong
  weights, run.
- **P1-03.** The same `=` on an infinite lazy return value, which DESIGN
  supports, makes update and regenerate run until they run out of memory.

The other three are medium and low: `realize!` misses lazy sequences that are
not `LazySeq`; the address check misses a choice above a call that made no
choices; `gfi/project` of a trace given another gen-fn mixes two generative
functions.

## Findings

### P1-01: a no-change retdiff from `clojure.core/=` lets the static walk keep a node whose input changed
- **Severity:** high (a silent wrong weight and a stale score, in an unusual but legal use)
- **Where:** `src/gen/dynamic.clj:655-668` (`retdiff`, the `=` at :666);
  `src/gen/diff.clj:127` (`compute`) and `:100` (`consistent?`, the laws'
  retdiff check). The retdiff is consumed by `run-static`
  (`src/gen/dynamic.clj:497-551`) and by the combinators (Map/Unfold skip
  elements whose diff is no-change).
- **What:** D1 makes a dynamic function's retdiff no-change "when its return
  value is the previous one". The docstring says this is "sound: equal values
  cannot differ", but the test is Clojure's `=`, under which values that code
  tells apart are equal:
  - `(= 0.0 -0.0)`;
  - `(= [1 2] '(1 2))` (where `conj` differs);
  - a sorted map and a hash map with the same entries (`first` differs);
  - `1` and `(int 1)`, `1` and `1N`.

  A caller told no-change keeps every node that depends only on that value,
  with its old weight. `gfi/same-value?` already encodes the right notion
  (`.equals` on atoms, the class, a collection's kind and order), but the
  retdiff does not use it. `diff/consistent?` uses `=` too, so
  `static-matches-dynamic`'s `sound-retdiff?` accepts these diffs. (On MLX
  the same `=` also reads both return values to the host inside the handler,
  which is rule 3's concern rather than soundness.)
- **Evidence:** `OUT/scripts/P1-01.clj`, run on the JVM backend.
  - *Signed zero.* The callee returns `(* x 0.0)`, and the static caller
    traces `:y ~ normal(atan2(r, -1), 1)` with y = 1. Updating `[:in :x]`
    from 1.0 to -1.0 turns r from 0.0 into -0.0, so the mean goes from π to
    -π. The inner retdiff is `NoChange`.
    - Expected weight: log N(1; -π, 1) − log N(1; π, 1) = −6.283 (the x terms
      cancel). The dynamic body gives −6.283.
    - The static body gives **0.0**, with score −4.631 against assess of the
      same choices −10.914. `(diff/consistent? 0.0 -0.0 no-change)` is true,
      while `(gfi/same-value? 0.0 -0.0)` is false.
  - *Vector and list.* The callee returns `[1.0 2.0]` or `'(1.0 2.0)` by a
    bernoulli, and the caller traces `:y ~ normal((first (conj v 5.0)), 1)`.
    Flipping the bernoulli should give −12.0. The static body gives **0.0**,
    with score −2.112 against assess −14.112.
- **Direction:** decide "unchanged" by a strict, bounded comparison in one
  place, and use it in `retdiff`, `diff/compute` and `diff/consistent?`.
  `same-value?`'s rules would do, extended to compare concrete JVM arrays by
  value (bitwise, as `JvmArray.equals` does) and lazy backends' arrays by
  identity, so no host read happens. Then the laws check the claim the code
  makes.
- **Known?** new. D1 (DESIGN §9, `genjl-test-suite.md` §4.1) records the
  precision, not the equality.

### P1-02: argdiffs ignore a change in the number of arguments, so a rest parameter or an estimator claims no-change
- **Severity:** high (silent wrong weights; one trigger is DESIGN's own G3 idiom)
- **Where:**
  - `src/gen/dynamic.clj:198-205`. `arg-diffs` maps over the new arguments
    only, so an added argument is compared with `nil` (an added `nil` is
    "identical") and a removed one leaves no argdiff at all.
  - `src/gen/dynamic.clj:451`. `rest-d` is no-change when every argdiff past
    the fixed parameters is no-change, which is vacuously true for a tail
    that shrank to nothing.
  - `src/gen/custom.clj:343` and `:357`. EstimatedGF keeps its trace, old
    arguments included, on all-no-change argdiffs.

  `src/gen/custom.clj:179` (DeterministicGF) checks the count; nothing else
  does.
- **What:** `gfi/update`'s `check-argdiffs` ties the argdiffs to the *new*
  arguments, one each, and no argdiff can say "an argument was removed". DESIGN
  §9 G3 makes a rest parameter, `[x & [y z]]`, the way to call, update and
  regenerate "with any number of arguments". Under `^:static`, or through a
  splice whose call site varies its arity, the rest names get no-change while
  their value changed. The walk then keeps every node that reads them. An
  estimated density's update returns the old trace, old `:args` and old
  estimate.
- **Evidence:** `OUT/scripts/P1-02.clj`, run on the JVM backend. The model is
  `(gen ^:static [x & more] (let [n (count more) y (trace :y (normal n 1))] y))`
  with y = 0.
  - *Spliced.* The call goes from 2 arguments to 1, and the outer argdiff is
    unknown-change. Expected weight: log N(0; 0, 1) − log N(0; 1, 1) = 0.5.
    The dynamic callee gives 0.5. The static callee gives **0.0**, with score
    −1.419 against assess −0.919.
  - *Added nil.* `(splice :c callee2)` becomes `(splice :c callee2 nil)`.
    Expected −0.5; the static callee gives **0.0**.
  - *Top level, G3 as written.* `(gen ^:static [x & [y]] …)` with mean `(if
    (nil? y) x y)` is updated from `[0.0 3.0]` to `[0.0]`, with the one
    (true) argdiff the new arguments take. Expected 4.5, got **0.0** (score
    −5.419 against assess −0.919).
  - *Estimated density.* Updating from `[1.0 2.0]` to `[1.0]` gives weight
    0.0, and the returned trace's `:args` are still `[1.0 2.0]` (score −2.919
    against assess at `[1.0]` −1.419).

  `static-matches-dynamic` never changes the number of arguments (`next-args`
  keeps the count), so the laws miss it.
- **Direction:** a change in the number of arguments is a change. The single
  place is `gfi/run-op`: when `(count args) ≠ (count (:args prev))`, it hands
  the callee unknown-change for the arguments past the shorter list, or for
  all of them. Each "all no-change" test (`arg-name-diffs`' `rest-d`,
  EstimatedGF, `keeps-omega?`) also requires equal counts, as `custom.clj:179`
  already does. `arg-diffs` gives unknown-change past the previous call's
  count.
- **Known?** new.

### P1-03: update and regenerate never return on an infinite lazy return value whose prefix is unchanged
- **Severity:** high (a valid program fails; DESIGN supports infinite return values)
- **Where:** `src/gen/dynamic.clj:666` (`(= (:retval prev-trace) (:retval state))`).
  `realize!` (:403) and `holds-batch?` (:645) are bounded by `realize-limit`;
  this comparison is not.
- **What:** DESIGN §9 ("Bodies run to completion") supports infinite return
  values: "Realization stops after 10,000 elements of a return value". When
  the old and new return values agree element by element (the usual case: an
  update that constrains only an observation), `=` walks both sequences
  forever. With `iterate`, which caches its elements, the heap fills; with
  `repeat` it would spin without allocating.
- **Evidence:** `OUT/scripts/P1-03.clj`, run on the JVM backend. The body
  returns `(iterate #(* 0.9 %) x0)` after `(trace :y (normal x0 1))`, and the
  update constrains `:y`. Result: `OutOfMemoryError ... at
  gen.dynamic/retdiff (dynamic.clj:666)` at `-Xmx1g`, within seconds.
- **Direction:** the comparison of P1-01's direction, bounded as
  `same-value?` is (its 256-value budget, past which it answers "different",
  which is sound), or identity plus a bounded walk of the realized prefix.
- **Known?** new. Review 1's bug 10 (the hang in `realize!`) was fixed in
  `realize!` only.

### P1-04: `realize!` forces only `clojure.lang.LazySeq`, so a lazy sequence of another class keeps its trace calls for after the run
- **Severity:** medium (simulate is silent; the rest is loud, on a program DESIGN says is handled)
- **Where:** `src/gen/dynamic.clj:403-414` (`(instance? clojure.lang.LazySeq x)`;
  `list?` is false for a `Cons`). This affects return values, and splice
  arguments too (`splice-at` and `static-call` realize through it).
- **What:** DESIGN §9 says lazy sequences in a body's return value are
  realized during the run, the fix for `(map … (trace …))` "giving a trace
  without its choices, silently". That holds only for `LazySeq` instances. A
  `Cons` onto a lazy tail, `(cons x (map …))`, and `iterate`'s `Iterate` are
  lazy sequences that `realize!` leaves alone.
- **Evidence:** `OUT/scripts/P1-04.clj`, run on the JVM backend, on
  `(cons 0.0 (map #(trace [:x %] …) [0 1 2]))` and `(iterate f [0 0.0])` with
  a traced f.
  - simulate's trace has **no choices and score 0.0** (`(map …)` and
    `(take 4 (iterate …))` give the three choices).
  - Constraining `[:x 0]` is "the constraint at [:x 0] was not visited".
  - Reading the return value later is the after-the-run error.
- **Direction:** realize any unrealized seq (`seq?`, or `IPending` not yet
  realized) up to `realize-limit`, not just `LazySeq`. Also walk sets, as
  `holds-batch?` walks `coll?`.
- **Known?** new.

### P1-05: the nested-address check sees a call only through its choices
- **Severity:** low (rule 6 holds in one order only; no weight is affected)
- **Where:** `src/gen/dynamic.clj:48-61`. The "above" test is
  `(cm/has-submap? (:choices state) p)`.
- **What:** a choice at `:a` after a call at `[:a :b]` that made no choices
  (a deterministic function, an empty Map) passes. The reverse order is loud
  ("Address [:a :b] lies under the choice at [:a]"). Once the call makes
  choices, the same model becomes loud, so the error depends on structure.
- **Evidence:** `OUT/scripts/P1-checks.clj` §3, run on the JVM backend. The
  first order gives choices `[[:a]]`; the second throws.
- **Direction:** test whether a visited path has p as a prefix (a set of
  visited prefixes, or a trie), not the choices.
- **Known?** new.

### P1-06: `gfi/project` of a trace given another gen-fn runs the new gen-fn's project over the maker's records
- **Severity:** low (an unclear contract, on an unusual use)
- **Where:** `src/gen/gfi.clj:564-567` (`(-project (:gen-fn trace) …)`);
  `project-by-maker` (:374) and `revise` (:380), which update and regenerate
  use.
- **What:** update and regenerate of a trace given another gen-fn revise it
  (it is projected by its maker). `project` instead calls the *new* gen-fn's
  `-project` on the maker's trace representation. For a dynamic trace this
  returns the maker's recorded densities, not the new gen-fn's. For a Map
  trace given a dynamic gen-fn it returns 0, which neither function's density
  gives. `proposal.clj:105` relies on projecting by `:gen-fn` (a wrapper
  projecting its model's trace), so the behavior is used, but it is not
  stated.
- **Evidence:** `OUT/scripts/P1-06.clj`, run on the JVM backend, with
  g1 = normal(0, 1) and g2 = normal(0, 3) at `:x` = 1.
  - `(project (assoc tr :gen-fn g2) all)` gives −1.419 (g1's score). g2's
    density is −2.073, and the revising update's weight is −0.654.
  - A Map trace given a dynamic gen-fn projects to **0.0** (its score is
    −2.088).
- **Direction:** say in `gfi/project`'s docstring which function projects a
  trace given another gen-fn. Make it the maker, unless the gen-fn is the
  maker's wrapper (`as-made-by`), and have the wrapper call that out
  explicitly. Or make project loud on a trace whose maker the gen-fn does not
  compute as.
- **Known?** new.

## Decisions worth revisiting

- **D1 (DESIGN §9, "Retdiffs are more precise"; `genjl-test-suite.md` D1).**
  The decision should name its equality. "No-change when the return value is
  the previous one" is sound for identity, or for a strict equality such as
  `same-value?`'s, but not for Clojure's `=` (P1-01). The laws' own check
  (`diff/consistent?`) should use that equality, so that a retdiff the code
  computes is tested by the same rule.
- **G3 (DESIGN §9, "No default positional arguments").** The rest-parameter
  idiom meets argdiffs that are one per *new* argument. Gen.jl never has this
  case, since its arities are fixed. Recommendation: record that a change in
  the number of arguments is a change of the rest parameter (P1-02's
  direction), so every kind that skips on no-change argdiffs also compares
  the counts.

## Checked and sound

- **Handler transitions** (`dynamic.clj:116-172`), derived against
  forward-sampling Eq. 4.1 / Alg. 18 and Def. 2.3.1:
  - generate weighs the constrained sites;
  - update weighs lp′ − lp_prev for retained and constrained sites, 0 for
    fresh ones, and subtracts gone sites in `finish`;
  - the discard is the overwritten and gone values, gone by kind (law audit
    E1, choice↔call);
  - regenerate weighs the retained ratio, with selected, new and dropped
    sites cancelling.

  Every site and call consumes one key in every mode, so assess draws what
  generate draws.
- **Replaced callees** (`compatible-callee?`, `replace-call`). The new-call
  semantics is a consistent internal proposal. Update subtracts the old score
  and discards its choices, and the reverse update reconstructs the old
  trace. Regenerate drops the old call, which cancels against the reverse
  move's prior draw.
- **Revise** (`gfi.clj:288-330`, `dynamic.clj:250-278, 816-819`). Update's
  weight is generate's weight less the old score, which is Def. 2.3.1 with
  q(fresh; x′, τ⊕σ), and the discard is every old choice not surviving
  unchanged. Regenerate's weight is generate's weight less the old
  projection on the kept choices, which is Eq. 4.1 for any q (also when
  nothing is kept). A store change at a call revises, projected under the old
  store. Ran (P1-checks §2): a revise under another store weighs −0.5994 for
  both update and regenerate, equal to the derived value, and keeps
  unselected choices.
- **`same-value?` / `same-gen-fn?`.** Sound by construction: values must be
  of one class; atoms compare by `.equals`; arrays, atoms and other mutable
  values only by identity; closures by their fields; the budget only ever
  answers "different". The static graph follows from the body fn's class.
- **Keys.** Sites take `split` (counters 0 and 1); ω is `fold-in` with a
  salt; the backward key uses its own salt. `split(k)[i] = fold-in(k, i)`
  (P1-checks §4), and nothing in this layer folds 0 or 1 into a key it also
  splits.
- **Encapsulated randomness.** `keeps-omega?` keeps ω only on no-change
  argdiffs, no constraint or selection, and no revise (but see P1-02 for the
  count). Regenerate with a fresh ω is MH on the extended space (τ, ω):
  deriving the ratio gives exactly the unselected sites' lp ratio, which the
  handler computes. Untraced draws and estimators fall back under batching
  and are refused under exact density. EstimatedGF's weights are the
  pseudo-marginal ratios, generate with all choices weighs log ξ, and
  regenerate of all choices is weight 0 by its joint sampler.
- **The static walk** (`dynamic.clj:497-551`). It skips only under these
  conditions: the same maker, the same store, no middleware, unchanged input
  diffs, and no constraint or selection reaching the node. A kept node
  consumes its key and re-records the old site or call, which is what an
  unchanged rerun gives. Hoisting (`static.clj:196-262`) keeps Clojure's
  evaluation order and refuses `and`, `or`, branches, `let` and `fn`.
  Computed addresses and the overlap check agree with the handler.
- **The compiled score** (`dynamic.clj:1126-1164`) splits keys as the
  handler's assess splits them and checks the choices as the handler does.
  Ran (P1-checks §1): it equals the handler's assess on 5 keys, with
  untraced draws in the callees, and equals the score on simulate's key.
- **Diff rules** (`diff.clj:144-228`). Each rule was derived for its arity:
  count, nth, get (including out-of-range and non-integer keys, and backend
  keys), conj, assoc, `+`, `-`, inc, dec, vec, identity. All are sound,
  given the `=` caveat of P1-01 for their element comparisons.
- **The batch tracer** (`array/batch.clj`). Elementwise lifting aligns the
  per-particle axes. Axes are normalized per particle before the batch shift
  (`sh/axes` on `-shape`). take with batched indices is a gather;
  take-along-axis and scatter-add have equal-rank operands; matmul handles
  per-particle vectors; linalg always gets a matrix right-hand side
  (`with-matrix-rhs`); draws gain the batch axes. Host reads, `=`, hashing,
  counting and `seq` raise the fallback.
- **`gen.batched`.** A fallback gives the scalar semantics, and a divergence
  after unchecked updates is loud. Spot checks regenerate a particle from its
  choices (deterministic under full constraints) and compare scores and
  array return values. Resampling by `gather` carries the scores,
  subtraces, node values and arguments, and keeps unbatched values by
  identity. `select` rebuilds from the merged choices and is marked
  unchecked. Unread batched bools that the trace does not hold raise the
  fallback.
- **Data algebra.** Choice maps: paths, `override`, `merge`, `unstack` at a
  root value. Selections: hierarchical, complement and `get-selected`.
  `-subselection` of a complement yields `Complement(all)` where `none` is
  meant, which only makes a walk rerun (sound). Traces: lookup and
  `subtrace`.
- **Guards** (`dynamic/core.clj`). Truthiness, lookups and representation
  tests answer per particle or raise. Vector invocation with a batched key
  throws an `IllegalArgumentException`, which `batched-or` turns into the
  fallback (read, not run).

## Coverage

Every line was read in the snapshot worktree:
- `src/gen/gfi.clj` 1–584
- `src/gen/dynamic.clj` 1–1296
- `src/gen/dynamic/core.clj` 1–250
- `src/gen/dynamic/static.clj` 1–385
- `src/gen/batched.clj` 1–532
- `src/gen/custom.clj` 1–387
- `src/gen/choicemap.clj` 1–405
- `src/gen/selection.clj` 1–154
- `src/gen/trace.clj` 1–93
- `src/gen/diff.clj` 1–228
- `src/gen/array/batch.clj` 1–344

Consulted where my layer depends on them:
- `src/gen/array.clj`: `truthy?`, `reduce*`, `all-true?`, `scatter-add`,
  `pick`;
- `src/gen/array/linalg.clj` 1–70;
- `src/gen/array/protocols.clj` 100–140;
- `src/gen/random.clj` and `random/threefry.clj` (split, fold-in);
- the `equals` methods in `backend/jvm.clj` and `backend/mlx.clj`;
- `test/gen/laws.clj` 440–495 (`static-matches-dynamic`).

Scripts, all run on the JVM backend:
`OUT/scripts/P1-01.clj`, `P1-02.clj`, `P1-03.clj`, `P1-04.clj`, `P1-06.clj`,
`P1-checks.clj`.
