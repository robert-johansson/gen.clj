# Verification: idiom-core.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

All scripts ran from the snapshot worktree on the pure-JVM backend, one JVM
at a time (`nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/<id>.clj`).
The reviewer's five scripts all reproduce their quoted output. My own checks
are `OUT/scripts/V-I2-01.clj`, `V-I2-01b.clj`, `V-I2-01c.clj`, `V-I2-02.clj`
and `V-I2-08.clj`.

## Verdicts

| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| I2-01 | A callee whose kind changes drops the addresses both callees share: the class rule runs before revise (correctness) | CONFIRMED; the direction has a second blocker | major → major |
| I2-02 | A nested `gen` written through an alias (`dyn/gen`) in a static body has its trace hoisted into the outer function (correctness) | CONFIRMED | major → major |
| I2-03 | `gen.dynamic` should split out the body rewriter and the static code generator | CONFIRMED (the facts and the seams hold; the split is a judgment) | major → major |
| I2-04 | `gen.dynamic.core:136` calls backend protocols directly (rule 3); "a backend array" is written five ways | CONFIRMED; one of the five sites is miscited | major → major (by the scale's letter) |
| I2-05 | The handler's per-site state is a 16-key hash map assoc'ed five times a site | CONFIRMED, already known (genclj-0jeu, docs/benchmarks.md) | major → major |
| I2-06 | Callability is written four ways; past six arguments a `defgf` or deterministic call throws AbstractMethodError; EstimatedGF is no fn | CONFIRMED | moderate → moderate |
| I2-07 | The batched spot checks are written twice, with four copies of the mismatch errors | CONFIRMED; the direction needs a dtype caveat | moderate → moderate |
| I2-08 | `gfi/project` dispatches on `:gen-fn`, so a trace given another kind projects to 0.0 silently (correctness) | CONFIRMED as behavior; **the direction as written breaks `gen.inference.proposal`** | minor → minor |
| I2-09 | `with-dispatch` takes any gf and is ignored on a non-dynamic one; `dyn/run` repeats `run-op`'s batching | CONFIRMED | minor → minor |
| I2-10 | The combinators read `gen.selection`'s internal records | CONFIRMED | minor → minor |
| I2-11 | `gen.choicemap`'s leaf functions walk from the root again for every address | CONFIRMED | minor → minor |
| I2-12 | Small helpers are duplicated across the layer | CONFIRMED | minor → minor |
| I2-13 | Stale docstrings, comments and dead code | CONFIRMED (all seven points) | minor → minor |
| I2-14 | Layout and small idioms | CONFIRMED (all points) | minor → minor |

**Counts:** 14 CONFIRMED, 0 PLAUSIBLE, 0 REFUTED, 0 DECIDED. I changed no
severity. Three findings are confirmed with corrections: I2-01 (its direction
misses a blocker), I2-04 (one site miscited) and I2-08 (its direction breaks a
caller).

## Findings

### I2-01: CONFIRMED (direction incomplete)

**What I read.** `dynamic.clj:217-242` (`compatible-callee?`, `replace-call`),
`:280-317` (`splice-call`), `combinators.clj:454-474` (`switch-revise`) and
the namespace doc at `:39-44`, and `gfi.clj:295-411` (`generate-again`,
`revise`, `run-op`). The class rule (`compatible-callee?`) decides first. When
it says no, `splice-call` calls `replace-call` and runs the new callee with no
`:prev`, and `switch-revise` calls `switch-generate`. Either way `gfi/revise`
never sees the old choices. `git log -S` dates the rule to a5b0219
(2026-10-19, review 3's B2 fix), before f1b19b9 (2026-10-20, review 5's single
revise), as the reviewer says.

**Reproduction.** `I2-01.clj` reproduces exactly. `V-I2-01.clj` states Def
2.3.1 directly, without `gen.laws`: every address in both traces keeps its
value, and the discard is exactly what the new trace lacks or changed. It
uses other kind pairs:
```
switch dyn -> unfold {:shared ([0 :x]), :kept? false, :discard {0 {:x 0.7266…}}, …}
switch unfold -> dyn {:shared ([0 :x]), :kept? false, :discard {0 {:x 2.1878…}}, …}
regenerate dyn -> map, sel/none: [0 :x] kept: false weight 0.0
```
So regenerate breaks too. With nothing selected, an unselected shared choice
is resampled with weight 0. This is review 3's B2 symptom, which that fix
removed only for same-kind branches. `V-I2-01b.clj` re-runs that test's
scenario (`combinators_test.clj`,
`switches-of-compatible-branches-keep-shared-addresses`) with one branch a Map
and the other a dynamic function making the same addresses. An update of `:i`
alone loses the observations:
```
two Maps (the test's case) | observations after the update: [1.5 1.7 2.2]
a Map, then a dynamic gf   | observations after the update: [2.168… 1.309… -0.950…]
```
In MH over `:i`, that silently resamples the data.

**Not a decision.** DESIGN §9 ("What an update or regenerate may keep of a
trace is decided in two places", l. 461-524) names the maker and the store,
"for every kind of generative function". It does not mention the class rule.
The only justification is the combinators namespace doc ("Branches of
different kinds share no addresses"), which is false. No D/U entry in
`genjl-test-suite.md` covers this, and Gen.jl 0.4.8 does the opposite. Its
Switch generates the new branch from the old choices whatever the branch's
kind: `switch/update.jl:91-94` (`update_recurse_merge`, then `generate`) and
`switch/regenerate.jl:20-21` (`get_selected(…, complement(selection))`, then
`generate`). No bean covers it (I searched for compatible, callee, kind, Switch
and revise).

**The direction.** I tested it by `with-redefs [dyn/compatible-callee?
(constantly true)]`.
- dyn↔Map, dyn→Unfold and the splice case all keep `[0 :x]`, with an empty
  discard and weight 0 (V-I2-01). distribution→dyn works.
- dyn→distribution fails, the blocker the reviewer named (`dist/core.clj:298-305`).
- There is a second blocker the reviewer missed (V-I2-01c). A distribution's
  single value given as old choices to a Map, an Unfold or a Recurse raises an
  ordinary ex-info: `combinators.clj:86-88` (`indices`: "a single value was
  given where indexed choices belong") and `:573-574` (Recurse's).
  `generate-dropping` re-raises it, so normal→Map and normal→Unfold throw where
  today they work.

So the fix must turn both checks into droppable `gfi/unvisited` errors, or
keep a replace only where the old and new choices really share no address (a
root value against nested choices: decide that by the choices, not by class,
since a Switch of distributions also has a root value). The rest of the
direction is sound: drop `replace-call`, add a kind-changing Switch law
instance, and correct the namespace doc.

Severity: major is right. It is a silent violation of Def 2.3.1 and
regenerate's retention on an ordinary model (a Switch of a Map and a dynamic
function), and the same duplicated decision the finding names.

### I2-02: CONFIRMED

**What I read.** `static.clj:79-84` (`outside-gens` skips a form only when its
head is literally `gen` or `gen.dynamic/gen`), `macro-name` (`:102-110`, which
does resolve, and excludes `gen.dynamic/gen`, so the aliased form is never
expanded), and `hoist-in` (`:211-246`). For `(dyn/gen [] (trace …))`,
`calls?` sees the inner `trace`. The head is no macro the analysis expands, no
special form and no `if`, so the form falls to the `:else` branch, and
`hoist-all` hoists the `(trace …)` argument into the outer graph. The dynamic
body is unaffected: there `trace` is the body fn's parameter, so a nested gen
shadows it lexically whatever its spelling.

**Reproduction.** `I2-02.clj` reproduces. `V-I2-02.clj` uses a model where the
change is not only a moved address: one inner gen spliced at `:y` and `:z`.
```
plain     static retval [-0.9128… -0.3295…] choices {:y {:a …} :z {:a …}} score -2.3088…
qualified static (gen.dynamic/gen) — identical to plain
aliased   static retval [-0.8542… -0.8542…] choices {:a -0.8542…} score -1.2837…
aliased   dynamic retval [-0.9128… -0.3295…] (the body without ^:static)
aliased   nodes [[:trace :a] [:compute nil] [:splice :y] [:splice :z]]
```
With the alias the static function draws one value and returns it twice, a
different joint distribution from the dynamic body's two independent draws.
`laws/static-matches-dynamic` (`laws.clj:454`) would fail on it, but no
instance spells the nested gen through an alias. Other aliased bodies (an
inner `let`) give a compile error ("a let inside an expression") rather than
a silent change, so the silent case is a nested gen whose body is a single
call or a call in argument position.

**The direction** (resolve the head with the analysis' `{:env :locals}`, as
`macro-name` does, and add an aliased law instance) is sound. Severity:
major is right.

### I2-03: CONFIRMED

**What I read.** `dynamic.clj:920-1115` and `:1117-1219`. The ranges and sizes
are right (196 and 99 lines). The rewriter uses only `gen.dynamic.core`
(`:guards` metadata, `host-value`, `caught`) and gen.array's `truthy?` and
`batched-branch`; it never touches the handler. `score-form` and `static-form`
use `path`, `guard-branches` and syntax-quoted runtime symbols (`static-site`,
`static-call`, `check-computed!`, `^:no-doc`). The proposed graph (guard ←
static ← dynamic) is acyclic.

Two notes for whoever does it. First, `gen.dynamic.static` today requires only
`clojure.string`. After the move it reaches gen.array through the guard
namespace, so the analysis is no longer loadable on its own (a small cost).
Second, its namespace doc ("builds its graph as data; gen.dynamic runs it")
would need rewording. "genclj-0oax asks the question" is loose: that bean asks
about `gen.backend.mlx` and `gen.array`, and `idiom.md` asks about
`gen.dynamic`. Severity: major per the scale (a namespace to split), though
mechanical.

### I2-04: CONFIRMED (one site miscited)

`dynamic/core.clj:136`, `(p/-full (p/-inner (p/-backend x)) [1] … dtype)`,
calls `IWrappingBackend`/`IBackend` methods on a backend. `docs/array-api.md`
§2 reserves `gen.array.protocols` for "backend implementers", and rule 3 says
core code never calls a backend. `gen.array.batch` has no public accessor for
the inner backend (`batch.clj:78-340`), so the direction (an accessor such as
`batch/particle-prototype`, then `array/full` under `with-backend`) is right.

On "five ways": `diff.clj:164`, `combinators.clj:325`, `:436` and
`batched.clj:269` are the backend-array predicate. **`dynamic.clj:931` is
not**: it is the batched test `(instance? gen.array.protocols.IBatchedValue
x)` (where `batch/batched?` is equivalent, since `Batched` is
`IBatchedValue`'s only implementation). The fifth backend-array spelling is
`dynamic/core.clj:62` (`scalar-array?`). The count holds; one citation is
wrong. `array/backend-array?` is private (`array.clj:51-52`), as stated.

Severity: major by the letter of the scale (a rule broken). The breach is one
line, so moderate would be defensible too. I kept major.

### I2-05: CONFIRMED (known)

`initial-state` (`dynamic.clj:380-387`) has 16 keys, so it is a
`PersistentHashMap`. A simulate site changes `:key` (`site-key`), `:choices`,
`:scores` and `:score` (`record`), and `:visited` (`trace-fn`): five updates.
`docs/benchmarks.md:260-330` reports 50.6% of the time and 66.2% of the
allocation for this, and recommends the same direction. It is the open bean
genclj-0jeu, as the reviewer says, so it adds nothing new. Severity: right.

### I2-06: CONFIRMED

`I2-06.clj` reproduces. Seven arguments on an Unfold and on a deterministic gf
throw `AbstractMethodError`, `apply` works, and `(est)` throws
ClassCastException. In `gfi.clj:573-584`, `defgf` emits 0 to 6 arities.
`DynamicGF` writes 0 to 20 plus a variadic arity by hand (`dynamic.clj:821-846`).
`DeterministicGF` writes 0 to 6 by hand, calling `run` (`custom.clj:206-214`).
`EstimatedGF` has no `IFn`.

Direction note: built with `defgf`, a deterministic call would go through
`gfi/simulate` and so draw a key from the implicit source on every call. That
is harmless but observable in the implicit key sequence. Beyond this layer,
`InternalProposalGF` (`proposal.clj:134`) and `ExactProposalGF`
(`enumerative.clj:305`) are not callable either (I3's layer). Severity:
moderate is right.

### I2-07: CONFIRMED (direction caveat)

`batched.clj:279-356`: two loops. `reference-spot-check!` gathers, evaluates
once, reads on the host and re-runs on the JVM. `spot-check-here!` takes each
particle with `array/take` and compares with the private `same-value?`
(`:223-232`). Each loop builds the two "disagrees with the scalar handler"
ex-infos itself (`:314`, `:321`, `:347`, `:353`), with slightly different
retval rules. `gfi/same-value?` is a public, unrelated notion that
`gen.dynamic` uses.

The caveat: one loop over host reads re-run on the batch's own backend (the
`mixed-backends?` case, a model closing over MLX arrays) rebuilds choices with
`(array/array v)`, which infers the dtype. An int32 or bool choice could come
back another dtype on MLX. The reads must carry each leaf's dtype, as
`from-array` does with `weak-dtype`. Severity: moderate is right.

### I2-08: CONFIRMED as behavior; the direction breaks a caller

`I2-08.clj` reproduces (0.0 against −0.9495). `V-I2-08.clj` shows the reverse
pair, a Map trace given a dynamic gen-fn: 0.0 against −5.8117.

The reviewer missed that `gfi/project`'s dispatch on `:gen-fn` is used
deliberately. `proposal.clj:105` (`proposal-weight`, which is
`InternalProposalGF`'s `-project`) projects `(assoc tr :gen-fn gf)`, where
`tr`'s maker is the wrapper (`trace/as-made-by`). It relies on `gfi/project`
going to the model `gf`. `project-by-maker`'s own docstring (`gfi.clj:374-378`:
"which a trace given another gen-fn is not projected by") records that
`gfi/project` goes by `:gen-fn`. Under `with-redefs [gfi/project
project-by-maker]` (the direction), V-I2-08 shows the effect:
```
internal proposal, project :y as built => -1.5400…
internal proposal, project :y by maker (the direction) => THROWS … the proposal makes the choice at [:z], which the constraints give …
internal proposal, regenerate :z by maker (the direction) => THROWS … (same)
```
The inner projection re-enters the wrapper's `-project` with a selection that
covers q's addresses. So the fix must first make `proposal-weight` call the
model's projection explicitly (`(gfi/-project gf tr …)`), and only then route
`gfi/project` by the maker. Alternatively, make a `:gen-fn` of another class
than the maker loud in `gfi/project`.

The claim "every other reader by its maker" is also overstated. The subtrace
projections (`dynamic.clj:771`, `combinators.clj:487`) go by the subtrace's
`:gen-fn` too, though there it always equals the maker. No §9 entry covers
`project`. Severity: minor is right, since the only `src/` caller that gives a
trace another gen-fn is the one that wants this dispatch.

### I2-09: CONFIRMED

`I2-09.clj` reproduces: the Map accepts `:dispatch` and the dispatcher is
called 0 times. `with-dispatch` (`dynamic.clj:906-915`) has no
`check-dynamic!`, unlike `with-params` and `with-transitions`. §9's "Local
checks" bullet lists only those two, so this is not a decision. `dyn/run`'s
`:batch-shape` branch (`:670-679`) is used only by `dynamic_test.clj:285` and
`:293`; every `src/` caller is a `DynamicGF` method, and nothing in `bench/`
uses it. Also update `with-dispatch`'s docstring ("opts as for run") when
`run` goes private. Severity: right.

### I2-10: CONFIRMED

`combinators.clj:94-99` is the only place outside `gen.selection` that names
`None`, `Hierarchical` or `(:m s)` (grep). Map, Unfold and Recurse's
regenerate (`:570`) all use it. A renamed field would give `(keys nil)`, no
indices: no element revisited, silently. Severity: right.

### I2-11: CONFIRMED

In `choicemap.clj`:
- `values` (`:220-223`) looks up every path from the root;
- `from-array` makes two `get-choice` walks per path (`:346-347`);
- `stack` (`:378`) and `unstack` (`:390-392`) walk per path, and `unstack`
  recomputes `(addresses cm)` (a sort at every node) once per particle.

`mcmc.clj:44, 51, 58, 102` call `to-array` and `from-array` every step. A
depth-first walk over `submaps` gives `addresses`' canonical order exactly
(D14 unchanged). Severity: right.

### I2-12: CONFIRMED

Every cited site checks out:
- `(if (vector? a) a [a])` at all five places;
- the store binding at `dynamic.clj:623-625`, `:682-684`, `:759` and `:798`;
- `custom.clj:146-156` repeating `with-store` and the missing-parameter check;
- `realize-limit` against the literal `10000` at `batched.clj:126`;
- the three assess paths (`dynamic.clj:299-303`, `combinators.clj:71-73`,
  `dynamic.clj:721`).

`run-call` with no `:prev` and op `:assess` reduces to `run-op` plus `under`,
so routing assess through it changes nothing. The other three also hold:
`without` at `gfi.clj:218-225`, and `check-steps`/`check-steps*` with the
`declare` at `combinators.clj:318-331`. Using the validating `cm/path` on the
handler's per-site path would add a check per site. The reviewer's "fast
unchecked variant" is the right choice there. Severity: right.

### I2-13: CONFIRMED

All seven points hold:
- `record-call`'s "(score sc)";
- `base {}` and its comment;
- `names-map` is dead (my grep over every private definition in the 11 files
  finds no other);
- `set-value` has no caller in `src/`, only a test of its path check;
- `diff`'s namespace doc omits `identity` (`diff.clj:228`);
- the namespace doc's transition signature lacks the key;
- DESIGN §3.4 says "the single `volatile!`", but `execute` holds two
  (`dynamic.clj:560-561`).

Severity: right.

### I2-14: CONFIRMED

All points hold:
- the misindented bodies (`combinators.clj:336-348`, `:397-404`, `static.clj:294`
  on);
- the long lines (193 characters at `dynamic.clj:566`, and the others cited);
- eight `_ (when …)` checks in `deterministic`;
- `(->NoChange)` at `diff.clj:47`, `:55` and `:65`, and `(set prev)` per
  element at `:110`;
- `(or (nil? kept) (cm/empty? kept))`, where `cm/empty?` is true of nil
  (`choicemap.clj:119`);
- `set-subselection`'s inline `put`;
- the positional `->DynamicGF … nil nil nil`;
- `check-args` returning a vector that `update` and `regenerate` then `vec`
  again;
- `check-argdiffs`'s ex-data without `:op`.

Severity: right.

### On "Decisions worth revisiting"

The observation is accurate. Namespaces exclude only the operators they use
(`gen.dynamic` `[+ -]`, `gen.combinators` `[map + -]`, `gen.custom` `[-]`),
and `docs/array-api.md` §2 lists operators "safe to refer", not ones that must
be. Rule 4's example reads as a template, so the wording suggestion is
harmless.

## Missed (found while checking)

- **I2-01:** the regenerate path and the observation loss (above). The
  direction's second blocker is in `combinators.clj:86-88` and `:573-574`.
- **I2-08:** `proposal.clj:105` depends on the `:gen-fn` dispatch the finding
  would remove.
- **I2-06 (outside I2's layer):** `InternalProposalGF` and `ExactProposalGF`
  are not callable either.
- **The reflection check:** `I2-reflection.clj` reloads all 11 namespaces with
  no reflection warnings (re-run). Its last `println`, after loading
  `gen.llm.token`, printed nothing although the JVM exited 0. That part of
  the script did not show its result, but the "no warnings" claim rests only
  on the reload, which did run.
