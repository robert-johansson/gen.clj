# Post-M5 review (idiom): I2, the modeling core

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The modeling core is in good shape. Its namespaces are small and data-oriented, the protocols are thin, loading it gives no reflection warnings, and only one private definition is dead. The docstrings are dense but nearly all of them are accurate. The two findings that matter most are correctness bugs I found in passing. First, whether an update keeps a callee's old choices is decided twice: a class-based rule from review 3 (`compatible-callee?`) runs before M4's choice-based `gfi/revise`, so changing a callee's kind (for example a Switch from a dynamic branch to a Map) resamples and discards addresses both callees share, against Def 2.3.1 as the law suite checks it (I2-01). Second, the static analysis recognizes a nested `gen` by its symbol's name, so an aliased `(dyn/gen …)` inside a static binding has its `trace` hoisted into the outer function, silently changing the model (I2-02). On structure, `gen.dynamic` should split along two clean seams: the body rewriter (about 190 lines) and the static code generator (about 95 lines) (I2-03). `gen.dynamic.core` breaks rule 3 in one line (I2-04). The rest is duplication and loose ends: making a generative function callable is written four ways (I2-06), the spot checks are written twice (I2-07), and there are smaller points.

## Findings

### I2-01: Whether a callee's old choices survive an update is decided twice, and the older rule drops addresses that Def 2.3.1 keeps (correctness)
- **Severity:** major (duplicated logic for one semantic, DESIGN §3.3; correctness in passing)
- **Where:**
  - `src/gen/dynamic.clj:217-242` (`compatible-callee?`, `replace-call`) and `:288-292` (in `splice-call`);
  - `src/gen/combinators.clj:454-474` (`switch-revise`, :462) and the namespace doc at `:39-44`;
  - the later mechanism is `src/gen/gfi.clj:390-411` (`run-op`'s `same-gen-fn?` → `revise`) with `:295-330` (`generate-again`).
- **What:** Two mechanisms decide what an update keeps when a different generative function sits where the old trace's callee was.
  - Review 3 added `compatible-callee?` (2026-10-19, a5b0219). It compares classes: two dynamic gfs, two distributions, or two combinators of one class with compatible fields. For any other pair it drops the old call wholesale: `replace-call` discards every old choice, and the callee is generated afresh.
  - Review 5 added `gfi/revise` (2026-10-20, f1b19b9). It decides by choices instead: the new callee runs against the old choices, and only those it leaves unvisited are dropped.

  The class rule runs first. So when the kind changes, an address both callees visit is resampled and discarded. The combinators doc justifies this with "Branches of different kinds share no addresses", which is false: a Map's choices `{0 {:x _}}` are exactly those of `(gen [xs] [(splice 0 kern (first xs))])`. The suite's own test of the principle (`combinators_test.clj:334`, "Def 2.3.1: a choice both branches make keeps its value across an index change") covers only same-kind branches.
- **Evidence:** `scripts/I2-01.clj` calls the law suite's own Def 2.3.1 checks (`gen.laws/applied?` and `exact-discard`, `laws.clj:151-167`). It updates a Switch's index from 0 to 1 with no constraints:
  ```
  switch dyn -> map       | [0 :x] kept: false | applied? (Def 2.3.1): false | discard is Def 2.3.1's: false | discard: {0 {:x -2.15…}}
  switch dyn -> other dyn | [0 :x] kept: true  | applied? (Def 2.3.1): true  | discard is Def 2.3.1's: true  | discard: {}
  switch map -> other map | [0 :x] kept: true  | applied? (Def 2.3.1): true  | …
  switch map -> dyn       | [0 :x] kept: false | applied? (Def 2.3.1): false | …
  splice :s dyn -> map    | [:s 0 :x] kept: false | applied?: false
  ```
  Expected by Def 2.3.1: [0 :x] is in both traces and unconstrained, so it keeps its value and is not in the discard. That is what happens whenever the class rule lets the call through to revise.
- **Direction:** Let `gfi/revise` decide for every pair.
  - Drop `compatible-callee?` and `replace-call`. `splice-call` and `switch-revise` then always run the new callee against the old subtrace, and `run-op` revises when that callee is not the maker.
  - The one blocker is a Distribution run against a node of old choices. `check-leaf-constraint` (`src/gen/dist/core.clj:298-305`) throws an ordinary ex-info, so `generate-dropping` would re-raise it. It should become a `gfi/unvisited` error naming the nested paths, which a revise drops; a constraint the run leaves stays the same loud error.
  - Add a kind-changing Switch to the law instances.
  - The PPL review owns the semantics.
- **Known?** New.

### I2-02: A `gen` nested in a static body is recognized by name only, so an aliased `(dyn/gen …)` has its trace hoisted into the outer function (correctness)
- **Severity:** major (a silent change of model; a one-line fix)
- **Where:** `src/gen/dynamic/static.clj:79-84` (`outside-gens`: `('#{gen gen.dynamic/gen} (first %))`), used by `pure!` (:86-91) and `calls?` (:184-187). Compare `macro-name` (:102-110), which resolves the head.
- **What:** The analysis skips the body of a nested `gen` (its `trace` is its own) only when the head is spelled `gen` or `gen.dynamic/gen`. Spelled any other way, the nested function is an ordinary call to `hoist-in`. Its `(trace …)` argument is hoisted as a node of the outer static function, and the inner gen is left closing over that value.
- **Evidence:** `scripts/I2-02.clj`, simulating three versions of one body with key 1:
  ```
  static, gen      choices: {:y {:a -2.15…}}  nodes: [[:compute nil] [:splice :y]]
  static, dyn/gen  choices: {:a 0.7158…}      nodes: [[:trace :a] [:compute nil] [:splice :y]]
  dynamic body     choices: {:y {:a -2.15…}}  nodes: []
  ```
  The static law requires the static and dynamic functions of one body to give identical traces. With the alias, `:a` moves from `[:y :a]` to the top level, and the call at `:y` makes no choices.
- **Direction:** Resolve the head, as `macro-name` does: a nested gen is a form whose head is not a local and resolves to `#'gen.dynamic/gen`. `outside-gens` then needs the analysis' `{:env :locals}`. Add a static-law instance that writes the nested gen through an alias.
- **Known?** New.

### I2-03: `gen.dynamic` (1,296 lines) holds three ideas; split out the body rewriter and the static code generator
- **Severity:** major (structural, but mechanical)
- **Where:** `src/gen/dynamic.clj`:
  - `:920-1115`: the body rewriter (`case-key`, `guarded-core`, `*outer-locals*`, `guard-branches` and its helpers, the lazy-call checks);
  - `:1121-1219`: the static code generator (`score-form`, `static-form`; `names-map` is dead, I2-13);
  - `:822-846`: 25 lines of hand-written IFn arities (I2-06).
- **What:** The handler (transitions, splices, `execute`, `run-static`, `DynamicGF`) is one idea. The rewriter is another: a code walker run at macroexpansion, with one runtime helper (`case-key`) and no dependency on the handler. The static code generator is a third. It turns `gen.dynamic.static`'s analysis into forms, so it belongs with that analysis rather than with the runtime that executes the graph. Today `gen.dynamic.static` holds the analysis and `gen.dynamic` the code generation, so the static language's compile side is split across two namespaces.
- **Evidence:** Read. The rewriter's only inputs are `gen.dynamic.core`'s `:guards` metadata, `array/truthy?` and `array/batched-branch`. The code generator's are `guard-branches`, `path`, and the symbols of the runtime functions it emits (`static-site`, `static-call`, `check-computed!`, all `^:no-doc` already).
- **Direction:**
  - Move the rewriter to `gen.dynamic.guard` (`guard-branches`, `*outer-locals*` and `case-key`). The public macros `gen`, `guarded-fn` and `defn-guarded` stay in `gen.dynamic` and call it.
  - Move `score-form` and `static-form` into `gen.dynamic.static` (which then requires `gen.dynamic.guard`), emitting fully qualified `gen.dynamic/static-site` and so on.
  - Generate the IFn arities (I2-06).
  - `gen.dynamic` is then about 980 lines and one idea: running bodies. The dependency graph stays acyclic: guard ← static ← dynamic.
  - `gen.gfi` (584 lines; the GFI, plus about 200 lines of revise machinery) and `gen.batched` (532) do not need to split.
- **Known?** New (genclj-0oax asks the question).

### I2-04: `gen.dynamic.core` calls a backend's protocol directly (rule 3), and "is this a backend array?" is written five ways
- **Severity:** major by the scale (a CLAUDE.md rule broken); the fix is small
- **Where:**
  - `src/gen/dynamic/core.clj:136`: `(p/-full (p/-inner (p/-backend x)) [1] … dtype)`, plus interface checks at `:49` and `:59-63`;
  - the same predicate in other forms: `src/gen/diff.clj:160-164` (`backend-key?`), `src/gen/combinators.clj:325` and `:436`, `src/gen/batched.clj:269`, `src/gen/dynamic.clj:931`.
- **What:** `particle` builds a particle's prototype by calling `IBackend/-full` on the batch backend's inner backend. That reaches past `gen.array` into `gen.array.protocols`, which `docs/array-api.md` §2 reserves for "backend implementers". Rule 3 says core code calls only `gen.array`. Because `gen.array`'s `backend-array?` is private (`array.clj:51-52`), callers in this layer spell "a backend array" as `(instance? gen.array.protocols.IArray x)`, `(and (array/array? x) (not (array/host? x)))` or `(or (array/host? x) (not (array/array? x)))`.
- **Evidence:** Read. `batch/value` is avoided here for a reason (it marks the value read, which the unread-comparison check counts). So `gen.array.batch` needs a public accessor for the inner backend, or for a particle's prototype.
- **Direction:**
  - Add `batch/particle-prototype` (or `batch/inner-backend`) in `gen.array.batch` and build the prototype with `array/full` under `with-backend`.
  - Make `array/backend-array?` public and use it at the five sites.
  - Outside this layer, `gen.dist.core:117-119` and `gen.dist:204` call `p/-defaults` and `p/-all-particles` too (I3's).
- **Known?** New.

### I2-05: The handler's per-choice bookkeeping is a 16-key hash map assoc'ed five times a site
- **Severity:** major (a hot path that allocates); already profiled and tracked
- **Where:** `src/gen/dynamic.clj:380-387` (`initial-state`), `:93-101` (`record`), `:179-184` (`site-key`), `:48-61` (`check-address!`), `:580-591` (`trace-fn`).
- **What:** Each choice is entered in three persistent collections keyed by its path: `:choices`, `:scores` and `:visited`. On top of that come `:key` and `:score`, and `check-address!` re-walks the prefixes. The profile in `docs/benchmarks.md` ("Where gen.clj's 1.7 µs a choice goes") puts this at 51% of the time and 66% of the allocation of a scalar simulate. A record for the run's state would also document its shape, which today is visible only in `initial-state`.
- **Evidence:** `docs/benchmarks.md`'s JFR profile; read.
- **Direction:** As that document recommends: one entry per used path (`path → [value lp]`) answering "visited", the scores and the choices, held in a record or in run-local transients. The laws check that nothing changes.
- **Known?** genclj-0jeu (open).

### I2-06: Making a generative function callable is written four ways, and two of them fail past six arguments with a non-ex-info error
- **Severity:** moderate
- **Where:**
  - `src/gen/gfi.clj:573-584` (`defgf`: arities 0 to 6 and `applyTo`; used by Map, Unfold, Switch, Recurse and `gen.llm.token`'s Completion);
  - `src/gen/dynamic.clj:821-846` (21 arities and a variadic one, by hand);
  - `src/gen/custom.clj:206-214` (0 to 6 by hand, calling `run` directly, not `simulate`);
  - `src/gen/custom.clj:335-369` (`EstimatedGF`: not callable).
- **What:** Gen.jl's `(gf args…)` is the return value of a simulated run, and `defgf`'s docstring promises it for every generative function. A record that implements `IFn` with only some arities throws `AbstractMethodError` for the others, so a seven-argument call on an Unfold fails, though `apply` with the same arguments works. An estimated generative function is not a function at all.
- **Evidence:** `scripts/I2-06.clj`:
  ```
  dynamic gf, 7 args           => 0.264…
  unfold (defgf), 7 args       => THROWS java.lang.AbstractMethodError: Method gen/combinators/UnfoldGF.invoke(…) is abstract
  unfold (defgf), apply 7 args => [-1.22…]
  deterministic, 7 args        => THROWS java.lang.AbstractMethodError …
  estimated, 0 args            => THROWS java.lang.ClassCastException: class gen.custom.EstimatedGF cannot be cast to class clojure.lang.IFn
  ```
- **Direction:** Have `defgf` generate arities 0 to 20 plus the variadic one, as `DynamicGF` does by hand, and use it for `DynamicGF`, `DeterministicGF` and `EstimatedGF` too. Alternatively, a small `callable` macro spliced into each record's body.
- **Known?** New.

### I2-07: The batched spot checks are written twice, with the mismatch errors built in four places
- **Severity:** moderate
- **Where:**
  - `src/gen/batched.clj:279-327` (`reference-spot-check!`), `:329-338` (`spot-check!`), `:340-356` (`spot-check-here!`);
  - the error constructors at `:314-316`, `:321-324`, `:347-349` and `:353-355`;
  - `same-value?` at `:223-232`.
- **What:** Two loops re-run particles through the scalar handler and compare them.
  - `reference-spot-check!` gathers the checked particles once, evaluates and reads them on the host, and re-runs on the JVM backend.
  - `spot-check-here!` reads each particle separately (`particle-value`) and compares with `same-value?`.
  - Each loop builds the same two ex-infos ("the scalar run of its choices fails", "the code branches on a batched value…") on its own, and each has its own retval rule.
  - The private `same-value?` (numeric closeness) shares its name with `gfi/same-value?` (the structural identity of revise), a different notion, two namespaces apart.

  This is the densest of the defensive layers genclj-0oax asks about. The layers themselves are each needed, since they give different guarantees (deterministic unread-comparison detection versus probabilistic re-runs), but this one can be written once.
- **Evidence:** Read.
- **Direction:** Write one loop over the gathered host reads, which works on any backend, taking the backend to re-run on: the JVM reference for a lazy batch, or the batch's own backend otherwise and when `mixed-backends?`. Add one `mismatch` constructor and one comparison of score and array retval (`close-host-values?`). Rename `same-value?` to something like `agrees?`. That saves about 40 lines.
- **Known?** New.

### I2-08: `gfi/project` reads a trace by its `:gen-fn`, every other reader by its maker (correctness)
- **Severity:** minor
- **Where:** `src/gen/gfi.clj:564-567`. Compare `project-by-maker` (`:374-378`), `trace/subtrace` (`src/gen/trace.clj:91`) and `run-op`'s revise (`gfi.clj:406-411`).
- **What:** `(assoc tr :gen-fn g)` is a documented idiom (`gen.trace` ns doc; `batched/traces` gives particles another gen-fn). Lookups and update both go by the maker, but `project` calls `-project` on `g`, which then reads a representation it did not make. For a `g` of another kind the result is a silent 0.0.
- **Evidence:** `scripts/I2-08.clj`, a dynamic trace given a Map gen-fn: `project … sel/all` gives `0.0`, while `project-by-maker` gives the score, `-0.9495…`, and `(t' :x)` still finds the choice.
- **Direction:** Have `gfi/project` call `project-by-maker`: a projection is a function of the trace, as in Gen.jl, where it dispatches on the trace's type.
- **Known?** New.

### I2-09: `with-dispatch` takes any generative function and is silently ignored on a non-dynamic one; `dyn/run` duplicates `run-op`'s batching
- **Severity:** minor
- **Where:** `src/gen/dynamic.clj:906-915`. Compare `with-params` (`:882-891`) and `with-transitions` (`:893-904`), which call `check-dynamic!`. Also `src/gen/dynamic.clj:670-679`.
- **What:**
  - Rule 6 says errors are loud. `(dyn/with-dispatch (comb/map k) f)` accepts the call, puts `:dispatch` in the Map record's extension map, and never calls `f`.
  - Separately, `dyn/run` is public and repeats `gfi/run-op`'s `:batch-shape` branch (`gfi.clj:400-401`). Every caller in `src/` is a `DynamicGF` method, which never passes `:batch-shape`. Only `dynamic_test.clj:285` and `:293` use the branch.
- **Evidence:** `scripts/I2-09.clj`: `with-dispatch on a Map: accepted? => true`, and the dispatcher is called 0 times.
- **Direction:** Call `check-dynamic!` in `with-dispatch`. Make `run` private without the batch branch, and have those two tests go through `gfi/run-op`.
- **Known?** New.

### I2-10: Combinators read selections' internal records
- **Severity:** minor
- **Where:** `src/gen/combinators.clj:94-99` (`selected-indices`: `(instance? gen.selection.None s)`, `(instance? gen.selection.Hierarchical s)`, `(:m s)`), used by Map, Unfold and Recurse.
- **What:** This is the only place outside `gen.selection` that knows its representation. A custom `ISelection` (the namespace doc invites one) gets `:all`, which is correct but slow, and a change to `Hierarchical` would break the combinators silently.
- **Direction:** Add a function in `gen.selection` that gives the top-level addresses a selection may select under (`:all` for `all`, complements and custom selections).
- **Known?** New.

### I2-11: `gen.choicemap`'s leaf functions re-walk from the root for every address
- **Severity:** minor
- **Where:**
  - `src/gen/choicemap.clj:220-223` (`values`);
  - `:339-359` (`from-array`, with two `get-choice` walks per path at `:346-347`);
  - `:369-380` (`stack`);
  - `:382-393` (`unstack`: every path walked once per particle).
- **What:** Each of these functions first lists `addresses` (sorting every node) and then looks every path up again from the root. `to-array` and `from-array` run once or twice every HMC or MALA step (`mcmc.clj:44`, `:51`, `:58`, `:102`).
- **Direction:** Add one depth-first `leaves` walk, `[[path value] …]` in canonical order. `values`, `to-array`, `from-array`, `stack` and `unstack` all follow from it.
- **Known?** New.

### I2-12: Small helpers duplicated across the layer
- **Severity:** minor
- **Where:**
  - `(if (vector? a) a [a])` beside `cm/path`: `src/gen/dynamic.clj:46`, `src/gen/dynamic/static.clj:279` and `:343`, `src/gen/custom.clj:384`, `src/gen/selection.clj:117`.
  - "Run under gf's store", in four copies: `src/gen/dynamic.clj:623-625`, `:682-684`, `:759`, `:798`. `src/gen/custom.clj:146-156` re-implements `with-store` (`dynamic.clj:191-196`) and `read-param`'s missing check (`:389-397`) through `dyn/*params*`.
  - The realization limit: `dynamic.clj:401` (`realize-limit`) and a literal `10000` in `batched.clj:126`.
  - A callee's assess, three ways: `dynamic.clj:299-303` (`run-op` plus `under`), `combinators.clj:71-73` (`gfi/-assess` plus `under`) and `dynamic.clj:721`. `run-call`, which its docstring calls "the one place a call's context is checked", would serve all three.
  - `gfi.clj:218-225` (`without`): a path removal that belongs in `gen.choicemap` beside the single-address `remove-submap`.
  - `combinators.clj:318-331`: `declare check-steps*`, then `check-steps`, then `check-steps*`, one function in three forms.
- **Direction:** Use `cm/path` (or a fast unchecked variant in `gen.choicemap`); add a private `with-store-of` macro, made `^:no-doc` public for `gen.custom`; share one limit; route assess through `run-call`; add `cm/remove-path`; merge `check-steps`.
- **Known?** New.

### I2-13: Stale docstrings, comments and dead code
- **Severity:** minor
- **Where:**
  - `src/gen/dynamic.clj:207-208`: `record-call`'s docstring says "(score sc)", a parameter it does not have.
  - `dynamic.clj:293-294`, `:298`: `base {}`, with a comment about inheriting the batch tracer; it is always `{}`.
  - `dynamic.clj:1121-1124`: `names-map` is never used. It is the layer's only dead private definition (checked by script).
  - `src/gen/choicemap.clj:236-238`: `set-value` is a "fast path for handlers", but no code in `src/` calls it.
  - `src/gen/diff.clj:19-20` (and DESIGN §9's list): the rules listed omit `identity` (`:228`).
  - `dynamic.clj:17-18`: the namespace doc gives transitions as `(state, address, distribution) → [value state']`; they take a key too (`:169-170`).
  - DESIGN §3.4 says "the single `volatile!` that `gen.dynamic/execute` threads", but `execute` holds two (`dynamic.clj:560-561`: `transition` and `vol`). Either move the transition into the state or correct the sentence.
- **Known?** New.

### I2-14: Layout and small idioms
- **Severity:** minor
- **Where:**
  - Bodies not indented under their `let` or arity: `src/gen/combinators.clj:336-348`, `:397-404`; `src/gen/dynamic/static.clj:294-326`.
  - Lines of 130 to 193 characters: `dynamic.clj:566`, `:1269`; `combinators.clj:285`, `:579`; `custom.clj:304`; `static.clj:354`.
  - `src/gen/custom.clj:226-254`: eight `_ (when …)` validations in a `let`, which a separate `check-spec!` would read better as.
  - `src/gen/diff.clj:47`, `:55`, `:65` build `(->NoChange)` where `no-change` exists. `:110` rebuilds `(set prev)` for every element in `consistent?` (quadratic, in the laws).
  - `src/gen/selection.clj:152`: `(or (nil? kept) (cm/empty? kept))` is `(cm/empty? kept)`. `:85-93` uses an inline named fn where `gen.choicemap` uses a private `put-at` for the same walk.
  - `src/gen/dynamic.clj:874`, `:1278`: positional `->DynamicGF … nil nil nil`.
  - `src/gen/gfi.clj:529-531`, `:545-547`: `check-args`, then `vec` of the raw args again; `check-argdiffs`'s ex-data (`:514-516`) lacks the `:op` its siblings carry.
- **Known?** New.

## Decisions worth revisiting

None of DESIGN §9's entries for this layer seemed wrong on reading. One point of wording: CLAUDE.md rule 4 shows `(:refer-clojure :exclude [+ - * / < > <= >= ==])`, but this layer (like the rest of `src/`) excludes only the operators each namespace uses on values, and keeps `clojure.core`'s for host index arithmetic, as `docs/array-api.md` §2 allows. That practice reads well. The rule could say so, so that a namespace like `gen.choicemap`, which does no array math, is not read as breaking it.

## Checked and sound

- **Warnings and dead code.** All 11 namespaces load with `*warn-on-reflection*` and no warnings (`scripts/I2-reflection.clj`). A script over every private `def` and `defn-` found only `names-map` unused.
- **The path rule and loud lookups (G9, F3, D4).** Read. `cm/path` validates; `walk` is shared by every lookup; the sentinel `not-found` arities never build an exception; and `lookup-error`'s data is `{:reason :path :at}`, as §9 specifies. `selected?`, `get-subselection` and `set-subselection` walk an address at a time, the empty path is the root, and nesting inside `all` or a complement is loud.
- **gen.trace.** Read. The maker survives `assoc`, `ICalls` keeps each combinator's representation private, and `(tr addr)` falls back from choice to call.
- **gen.gfi's revise machinery.** Read against §9 (the maker, the store, a revise, unvisited constraints). `same-value?`'s budget, `same-entries?` by iteration order and the comparator check match the entry. `outside-revise` wraps every public generate, update and regenerate.
- **gen.dynamic.core.** Read. It is data-driven (`defguard` and `:guards` metadata, `guarded-core` built from `ns-publics`), with an inline `get`. Apart from I2-04 it is tidy, and its guards are what §9's batched-execution entry lists.
- **The handler's transitions and `finish`.** Read. There is one key per site whether it samples or not, gone-by-kind accounting (law audit E1), and omega kept only under `keeps-omega?`'s conditions.
- **Map, Unfold and Recurse.** Read. Map and Unfold share the accumulator (`put-element`, `put-update`, `drop-elements`) instead of copying it, and Recurse's two phases are clear.
- **gen.dynamic.static.** Read. The hoisting rules, the and/or/branch/fn/let errors and the address-overlap check match §9's rewrites (G2, G11, G12), apart from I2-02.
- **gen.custom.** Read. Its spec validation is loud for every combination §9 lists (G5).
- **gen.batched.** Read against `docs/array-api.md` §8a–§8b. The fallback layers are distinct, `check` runs before every read, and `diverged` is loud after unchecked updates.
- **Ran on the JVM backend:** `scripts/I2-01.clj`, `I2-02.clj`, `I2-06.clj`, `I2-08.clj`, `I2-09.clj` (output quoted above).

## Coverage

Every line of every file was read.

| File | Lines read |
|---|---|
| `src/gen/choicemap.clj` | 1-405 (all) |
| `src/gen/selection.clj` | 1-154 (all) |
| `src/gen/trace.clj` | 1-93 (all) |
| `src/gen/diff.clj` | 1-228 (all) |
| `src/gen/gfi.clj` | 1-584 (all) |
| `src/gen/dynamic.clj` | 1-1296 (all) |
| `src/gen/dynamic/core.clj` | 1-250 (all) |
| `src/gen/dynamic/static.clj` | 1-385 (all) |
| `src/gen/batched.clj` | 1-532 (all) |
| `src/gen/custom.clj` | 1-387 (all) |
| `src/gen/combinators.clj` | 1-792 (all) |

Context, not reviewed for findings: `src/gen/dist/core.clj:285-360`, `src/gen/array.clj:20-130` and `:540-590`, `src/gen/array/batch.clj:60-340`, `test/gen/laws.clj:100-175`, and `test/gen/combinators_test.clj:330-367`.

Scripts are in `OUT/scripts/`: `I2-01.clj`, `I2-02.clj`, `I2-06.clj`, `I2-08.clj`, `I2-09.clj` and `I2-reflection.clj`.
