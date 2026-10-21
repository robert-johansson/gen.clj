# Gen.jl 0.4.8's test suite, mapped to gen.clj (M5, step 1)

*Research notes, 2026-10-20. Bean genclj-hzoa, the first step of M5
(genclj-xxuc): the coverage table, not the port. Gen.jl's source is the
0.4.8 tag (commit 0e8cfac, 2025-07-11), read only for the mapping; Gen.jl
was then run once on this Mac, with the owner's go-ahead, for its own suite
and the upstream findings (§4.3). gen.clj is m4-more at 12e788d. Each
mapping was checked in gen.clj's code; where a mapping was not clear from
the code, a short REPL run on the pure-JVM backend checked it ("checked"
below, listed in §7). F1, F2 and G6 were updated after their fixes on
m4-f12 (2beab42, b9eb9d2, 7e85fb3; F1 by the owner's decision, D15). The
categories were updated on 2026-10-21 (genclj-fevk) to the owner's
decisions and the ports merged at 17fd8bd (§3).*

## 0. Summary

- **Size.** `test/` holds 38 files: `runtests.jl`, three files that only
  include others (`dsl/dsl.jl`, `inference/inference.jl`,
  `modeling_library/modeling_library.jl`) and 34 test files. One of them,
  `inference/enumerative.jl`, is included by nothing, so Gen.jl's own suite
  never runs it (U2). The 34 files hold **1,699 assertions** (1,684 `@test`,
  15 `@test_throws`) in 195 leaf testsets, counted as written. Run, loops
  included, they are 1,863 assertions (1,832 without `enumerative.jl`):
  DESIGN.md §1's "about 1,900".
- **By category** (§3):

  | port | oracle | statistical | n/a | gap |
  |---:|---:|---:|---:|---:|
  | 1,372 (80.8 %) | 6 | 33 | 210 | 78 (4.6 %) |

  At first (2026-10-20) port was 1,262, n/a 110 and gap 288. The owner's
  decisions (DESIGN §9) moved D1's precise retdiffs (16) and D6's exact
  beta (4) to port, and G3's default arguments (60) and G15's IR shape (40)
  to n/a; G1, G5 and G8–G14 were built, and their 89 assertions port (with
  one of `selection.jl:49`'s, a port the first count took for a gap).
- **The gaps left each have a bean:** the static
  language's rewriting of computed addresses and nested traces (G2, 42,
  genclj-lgij), labels that are not numbers in `dist` (G7, 17,
  genclj-414l), loud lookups (D4, 16, genclj-ocwu) and VI that trains the
  model's parameters (G4, 3, genclj-xjte).
- **The oracle** (§5) wrote Gen.jl's values as fixtures (genclj-il3d):
  Switch's regenerate weights, which its 6 assertions are checked against
  (U1, genclj-c5rl), beta at its endpoints (D6), and the statistical
  testsets' statistics over 20 seeds, a cross-check of gen.clj's.
- **Gen.jl's own suite passes** on Julia 1.12.6 (run once, 2026-10-20):
  1,863 assertions pass, none fail or error, in 214 testsets, `enumerative.jl`
  included (§4.3).
- **Upstream findings, run** (§4.3):
  - U1 confirmed. A Switch called from `@gen` code weighs a choice resampled
    from its prior log p(z′) − log p(z) exactly, where Eq. 4.1 gives 0, and
    selection MH on that choice settles at p(z)² p(y | z): mean 0.4987 ±
    0.0018 against the exact posterior's 0.75 (142 standard errors), as the
    p² reading predicts (0.5).
  - U2 confirmed: `enumerative.jl` is never included; run, it passes (31 of 31).
  - U3 refuted: the singular covariance does not error. Cholesky succeeds
    with a pivot of 2.1e-8, and both sides of the assertion give 15.24.
    gen.clj rejects it, by the owner's decision (D15).
  - U4 confirmed: the batched `lecture!` test's assertion passes with
    q_batched never trained (−0.2250), and q_batched's own training does not
    converge (θ −5.70 for −2, its score −2.73 for −0.215).
- **gen.clj findings** (§4.4): five loud-error gaps, all fixed: a silently
  wrong `dc/log-prob` for a homogeneous mixture of mvnormals (F2, with G6),
  an `mvnormal` that accepted the singular 2·ones(2, 2), as Gen.jl's does
  (F1: an error now, D15), vector paths read as one address (F3, with G9),
  `full` of a tape value (F4, with G14) and quoted `trace` in a static body
  (F5, G12).

## 1. Method

- **Counting.** A script (kept out of the repo) parses each file's blocks:
  openers and `end` at bracket depth 0, comments and strings stripped. It
  attributes each `@test` and `@test_throws` to its innermost `@testset`. The
  table's counts were checked against it row by row. Counts are assertion
  sites: a `@test` in a loop counts once.
- **Categories, per assertion.** A testset's assertions can fall in several
  categories, so each row gives the split.
  - **port**: the expected value is in the test or follows from the model, so
    the assertion ports without Julia. A port may rely on a documented
    difference (D1–D3, D5–D16 in §4.1). A testset without assertions (a
    smoke test) ports as a run without error.
  - **oracle**: needs a value Gen.jl computes.
  - **statistical**: seeded sampling, compared by statistics with a
    variance-derived band (CLAUDE.md's test conventions).
  - **n/a**: Julia-only machinery, or a feature the owner decided gen.clj
    does without (G3, G15), with the reason in the row.
  - **gap**: gen.clj lacks the feature (G2, G4, G7), or the decided
    behavior is not built yet (D4). Each has a bean. The first mapping also
    counted D1 and D6 here, behaviors that differed without a documented
    decision, and the gaps since closed (G1, G5, G8–G14).
- **IDs.** D* are differences of behavior (§4.1), G* gaps (§4.2), U*
  upstream findings (§4.3), F* gen.clj findings (§4.4).
- **Index base.** Line numbers are Gen.jl's, as `test/<file>:<line>` of the
  testset's `@testset`. Addresses in the notes are gen.clj's, 0-based.

## 2. The table

One row per leaf testset, in `runtests.jl`'s order (`enumerative.jl`, which
nothing includes, sits after the other inference files). A testset nested in
a larger one is named `parent / child`; in `switch.jl` the comment headers
of the file (Hierarchy, (More complex) hierarchy, … to test discard) name the
groups.

#### `test/autodiff.jl` (2 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | autodiff / fill | 2 | gradient of a scalar filled into a vector and a matrix is the sum of the cotangents | `array/grad` over `array/full` of a tracked scalar | port 2 | G14, closed (genclj-3mx6): `full` takes a tape value |

#### `test/diff.jl` (32 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | diff arithmetic / + | 12 | NoChange + NoChange is NoChange, any UnknownChange gives UnknownChange; the value is the sum | `gen.diff/propagate` on `#'+`; `diff-test/every-rule-describes-the-change` | port 12 |  |
| 40 | diff vectors / length | 6 | length of a Diffed vector: NoChange, UnknownChange, IntDiff(-1) from VectorDiff(3, 4) | `propagate` on `#'count` | port 6 |  |
| 56 | diff vectors / getindex | 14 | indexing with a Diffed index or a Diffed vector: element diffs from VectorDiff.updated | `propagate` on `#'nth` (0-based) | port 14 | D2 |
| 94 | diff dictionaries | 0 | empty | `dict-diff` | — | empty upstream |
| 97 | diff sets | 0 | empty | `set-diff` | — | empty upstream |

#### `test/selection.jl` (35 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | dynamic selection | 18 | `in` for addresses and pair paths, `s[addr]`, set_subselection! | `sel/select`, `selected?`, `get-subselection`, `set-subselection`; `choicemap-test/selection-algebra` | port 17 · n/a 1 | G9, closed (genclj-cbec): paths and `set-subselection`; N: `isa DynamicSelection` |
| 36 | all selection | 4 | selectall contains every address and path | `sel/all` | port 4 | G9 (paths), closed |
| 49 | complement selection | 13 | complement of all, none, a nested selection, double complement, its subselection | `sel/complement`; `a-selection-and-its-complement-partition-the-choices` | port 13 | G9 (paths, 5 of them; the first count said 6), closed; semantics checked by hand (reduce over get-subselection) |

#### `test/assignment.jl` (132 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | static assignment to/from array | 12 | to_array order and from_array on a static choice map with vector leaves; shallow counts | `cm/to-array`, `cm/from-array`, `values-shallow`, `submaps-shallow` (one kind of choice map); `arrays-and-batches` | port 12 | vector leaves as arrays (a Clojure vector is a product value); D14 (to_array's order; the same here) |
| 24 | dynamic assignment to/from array | 12 | the same on a dynamic choice map | as above | port 12 | as above |
| 52 | dynamic assignment copy constructor | 4 | DynamicChoiceMap(other) copies values and submaps | persistent values: the map is its own copy | port 4 |  |
| 67 | internal vector assignment to/from array | 8 | vectorize_internal's index-keyed choice map to/from array | an int-keyed node (Map's choice maps), 0-based | port 8 | D2 |
| 86 | dynamic assignment merge | 10 | merge of overlapping-node choice maps | `cm/merge`; `functional-update` | port 10 |  |
| 114 | dynamic assignment variadic merge | 2 | merge of one and of three choice maps | `cm/merge` of 0 to n choice maps | port 2 | G10, closed (genclj-cbec) |
| 123 | static assignment merge | 10 | merge of static choice maps | `cm/merge` | port 10 |  |
| 143 | static assignment variadic merge | 2 | merge of one and of three | `cm/merge` of 0 to n | port 2 | G10, closed |
| 152 | static assignment errors | 12 | KeyError from get_submap/get_value at the wrong kind of address; empty submap when absent | `cm/get-submap`, `cm/get-choice` return a Value, EMPTY or nil | port 2 · n/a 5 · gap 5 | D4: decided (loud lookups), to build (genclj-ocwu); N: `static_get_submap(…, Val(:x))` |
| 221 | dynamic assignment errors | 7 | the same KeyErrors on a dynamic choice map | as above | port 2 · gap 5 | D4: decided, to build (genclj-ocwu) |
| 259 | dynamic assignment overwrite | 16 | overwriting values and submaps; has_value/has_submap after; illegal set under a value | `cm/set-choice`, `set-submap`, `has-submap?` | port 12 · gap 4 | D4, decided, to build (genclj-ocwu): `has-submap?` is true at a value, get-submap at a value; the illegal set ports through `set-choice`, and `set-submap` takes a path (G9) |
| 315 | dynamic assignment constructor | 2 | choicemap((:x, 1), (:y => :a, 2)) | `cm/choicemap` | port 2 |  |
| 322 | choice map equality | 6 | == and != of nested choice maps | records compare by value | port 6 |  |
| 337 | choice map nested view | 8 | nested_view: getindex, length, keys, == | `cm/to-map` (parity §1h) | port 8 |  |
| 358 | filtering choicemaps with selections | 6 | get_selected by an address and by a path | `sel/get-selected` | port 6 |  |
| 377 | invalid choice map constructor | 1 | a malformed pair throws | `cm/choicemap` with an odd number of arguments throws | port 1 |  |
| 383 | choicemap hashing | 14 | choice maps as Dict keys, order independence, no collisions among four | records hash by value | port 13 · n/a 1 | N: StaticChoiceMap conversion |

#### `test/gen_fn_interface.jl` (5 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 12 | update(...) shorthand assuming unchanged args ($lang) | 3 | update(trace, constraints) without args, dynamic and static (runs twice) | `gfi/update` 2- and 3-arity; dynamic and `gen ^:static` | port 3 |  |
| 25 | regenerate(...) shorthand assuming unchanged args ($lang) | 2 | regenerate(trace, selection) without args (runs twice) | `gfi/regenerate` 2- and 3-arity | port 2 |  |

#### `test/dsl/macros.jl` (14 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 38 | macros in dynamic DSL | 7 | user macros expanding to a distribution, an @trace, a `~` statement or Expr(:gentrace) | `gen` macroexpands its body (checked: a macro to a distribution and one to `(trace …)`) | port 3 · n/a 4 | N: macros that bind a variable in the enclosing scope (`y ~ …` statements) and Expr(:gentrace) |
| 122 | macros in static DSL | 7 | the same in static functions | `gen ^:static`: a macro inside a distribution or at a binding's head (macroexpanded) | port 3 · n/a 4 | G11, closed (genclj-9d9h) |

#### `test/dsl/dynamic_dsl.jl` (88 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 8 | Julia call syntax | 5 | calling a gen function, default arguments, (grad) one-liners, an addressless @trace | IFn on DynamicGF (`dynamic-test/untraced-calls`), addressless `splice` (`splices-without-an-address-share-the-namespace`) | port 3 · n/a 2 | n/a: default arguments (G3, decided: Clojure has none) |
| 38 | return type | 1 | get_return_type of a typed @gen | none | n/a 1 | Julia return-type annotations |
| 49 | address visitor | 4 | AddressVisitor/all_visited (internal) | the handler's visited set and `gfi/unvisited` (`dynamic-test/errors-are-loud`) | n/a 4 | internal API; the behavior is generate's loud unvisited check |
| 73 | simulate | 3 | retval, args and score of a bernoulli site | `gfi/simulate` | port 3 |  |
| 86 | update | 19 | a branch switch: discard, new choices, score, weight, retdiff; a loop's update; discard of a shrinking loop | `gfi/update`; `structure-changing-update`, laws `updates-are-reversible-with-exact-discards` | port 19 | D1 (decided): the loop body returns nil both times, retdiff no-change (Gen.jl UnknownChange); the port asserts gen.clj's |
| 199 | regenerate | 9 | 10 regenerations of the branch with a changed argument: score, values, weight, retdiff | `gfi/regenerate`; `regenerate-weight-is-the-retained-density-ratio` | port 9 | D1 (decided): when the branch stays, the return value is unchanged, retdiff no-change; the loop's mu = rand() becomes fixed host values |
| 293 | choice_gradients and accumulate_param_gradients! | 15 | argument, choice and parameter gradients (with a retval gradient) against finite differences | `grad/choice-gradients` (:argnums), `grad/param-gradients` (one store for foo and bar); `gradients-test/*` | port 15 | D7 |
| 381 | backprop params with splice | 1 | a parameter read in an addressless callee gets the retval gradient | `param-gradients` through an addressless `splice` (checked: 2.0) | port 1 |  |
| 400 | gradient descent with fixed step size | 2 | FixedStepGradientDescent one step; the accumulator is reset | `learning/sgd` + `learning/step` (checked: 0.001) | port 1 · n/a 1 | N: accumulator reset (D7) |
| 415 | gradient descent with shrinking step size | 2 | GradientDescent(0.001, 1000) one step | `learning/gradient-descent` (checked: 0.001) | port 1 · n/a 1 | N: accumulator reset (D7) |
| 430 | multi-component addresses | 14 | pair addresses :x => 1, nested calls, submaps | vector paths `[:x 0]`; `splices-nest-choices` | port 14 |  |
| 465 | project | 4 | project on none, one, a nested and two addresses | `gfi/project`; `project-splits-the-score` | port 4 |  |
| 487 | getindex(trace) | 8 | trace[addr] for choices, for a call (its return value), trace[] and a missing address | `(tr path)`, `:retval` | port 7 · gap 1 | G1 (a call's return value), closed (genclj-cbec); D4 (a missing address is nil), decided, to build (genclj-ocwu) |
| 535 | docstrings | 1 | @doc of a @gen function | a Clojure def's docstring | n/a 1 | Julia's @doc |

#### `test/dsl/static_dsl.jl` (175 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 118 | check IR | 77 | the IR of datum and model: argument, choice, call and Julia nodes, their inputs, types, grad flags, return node | `gen.dynamic.static/ir`; `static-test/the-graph-as-data-and-as-dot` | port 33 · n/a 44 | N: types and (grad) flags (19); G15, decided n/a (25): per-expression Julia nodes, node identity, dist objects |
| 266 | at_choice | 16 | `:x => i` in a static body becomes a choice_at call node | a compile error (computed address); `comb/choice-at` by hand works | gap 16 | G2: decided, to build (genclj-lgij) |
| 299 | at_call | 17 | `@trace(foo(mu), :x => i)` becomes a call_at node | a compile error; `comb/call-at` by hand | gap 17 | G2: decided, to build (genclj-lgij) |
| 333 | tracked diff annotation | 2 | (static, diffs) option | always on | n/a 2 | D10 |
| 349 | no julia cache annotation | 2 | (static, nojuliacache) option | node values always kept | n/a 2 | D10 |
| 367 | trainable parameters | 2 | @param makes a TrainableParameterNode that can be returned | `(param :theta)` as a compute binding (checked) | port 1 · n/a 1 | n/a: G15 (no parameter node kind) |
| 381 | returning a trace directly | 7 | `return @trace(...)` and `x = @trace(...); return x` have equivalent IRs | a call in return position is a last node (static/analyze) | port 4 · n/a 3 | n/a: G15 (constant argument nodes, 2), and typ |
| 410 | @trace within Julia expressions | 5 | a tuple of two @traces; @trace inside a distribution's argument; call and generate | a compile error in `gen ^:static` (checked); the rewritten bindings work (checked) | gap 5 | G2: decided, to build (genclj-lgij); also D5 (circle's std 0) |
| 435 | repeated assignments | 8 | x = x + 1 and y = y + 1: separate nodes; calls give 1 and 3 | static let rebinding (checked: 1 and 3) | port 3 · n/a 5 | n/a: G15 (argument nodes and node identity; the IR names both nodes y) |
| 466 | tuple assignments | 11 | destructuring, a ternary on an @trace, generate weights | destructuring works (`destructuring-binds-what-clojure-binds`); a trace inside the ternary is a compile error | n/a 7 · gap 4 | n/a: G15 (the IR's nodes and inputs, 7); G2 (the generates, 4): decided, to build (genclj-lgij); the rewritten body gives the weights (checked) |
| 501 | use of 'end' | 1 | x[3:end] in a static body | subvec | n/a 1 | Julia's `end` index |
| 513 | 'return nothing' | 1 | a static body returning nothing | `(gen ^:static [] (let [] nil))` (checked) | port 1 |  |
| 523 | getindex(trace) | 7 | trace[addr] for choices, calls and trace[] in a static function | `(tr path)`, `:retval` | port 7 | G1, closed (genclj-cbec) |
| 560 | docstrings | 1 | @doc of a static function | def docstring | n/a 1 | Julia's @doc |
| 573 | one-line definitions | 2 | @gen (static) f(x) = (...) | none needed | n/a 2 | Julia's short function syntax |
| 584 | module-defined static functions | 4 | two modules each defining foo | namespaces | n/a 4 | Julia modules and generated functions |
| 596 | static gen function choicemaps | 9 + 1 throws | the choices of a static function with a static callee; propose with a trace argument | static `splice`, `gfi/propose` (checked) | port 9 · gap 1 | D4 (@test_throws KeyError get_submap at a value): decided, to build (genclj-ocwu) |
| 633 | returning a SML function from macro | 1 | a macro that defines a static function | a Clojure macro emitting `(def … (gen ^:static …))` (checked) | port 1 |  |
| 640 | global variables | 1 | a static body reads a module global | vars in static bodies (`a-local-is-not-the-var-it-shadows`) | port 1 |  |

#### `test/optional_args.jl` (58 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | optional positional args (calling + GFI) | 42 | default arguments in calls, simulate/generate/update/regenerate, gradients, nested and gf-valued defaults | a rest parameter `[x & [y z]]` covers calls and the GFI (checked), but the trace's args stay as given | n/a 42 | n/a: G3, decided (Clojure has no default positional arguments; a rest parameter covers calls, update and regenerate) |
| 131 | optional positional args (combinators) | 16 | kernels with defaults under call_at, Map, Unfold and Recurse | rest parameters in kernels | n/a 16 | n/a: G3, decided, as above |

#### `test/static_ir/static_ir.jl` (138 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | static IR (outside the leaf testsets) | 3 | repr of a built IR contains '== Static IR ==' | `static/dot` renders DOT | n/a 3 | Julia's text/plain show; StaticIRBuilder is internal (the ports write the commented-out bodies) |
| 65 | Julia call | 1 | calling a constant static function | IFn | port 1 |  |
| 75 | simulate | 2 | score and retval of foo(a, b) with a callee and a parameter | `gen ^:static`, `with-params` | port 2 |  |
| 89 | generate | 4 | constrained choices, score, weight | `gfi/generate` | port 4 |  |
| 108 | project | 1 | project on two addresses | `gfi/project` | port 1 |  |
| 120 | update without tracked diffs | 45 | five argdiff/constraint cases: choices, retval, weight, retdiff, discard, a counter of node reruns, score | `static-test/update-and-regenerate-skip-what-did-not-change` (all five cases checked) | port 44 · n/a 1 | D1 (decided; case 3: b passed again is identical, so the callee gets no-change: counter 0, retdiff no-change); N: case 5's counter is 1 only under nojuliacache (D10) |
| 200 | regenerate without tracked diffs | 40 | the same five cases for regenerate | as above (checked) | port 39 · n/a 1 | D1 (decided; case 3), D10 (case 5's counter); cases 4 and 5 need another key than simulate's |
| 277 | backprop | 19 | choice and parameter gradients of a static function with a static callee | `choice-gradients`, `param-gradients`; `choice-gradients-through-the-compiled-score-are-the-handler-s` | port 17 · n/a 2 | N: get_param_grad before and after zero_param_grad! (D7) |
| 432 | update with tracked diffs | 8 | a callee whose retdiff is NoChange keeps the caller's NoChange; with and without track_diffs | diff propagation in static bodies (checked) | port 5 · n/a 3 | N: reruns counted under nojuliacache, and the untracked variant (D10) |
| 468 | regenerate with tracked diffs | 8 | the same for regenerate | as above | port 5 · n/a 3 | D10 |
| 521 | cached julia nodes | 7 | a Julia node runs once in simulate and only when its inputs change | node values kept in the trace (checked: 1 0 0 1 1 1) | port 7 |  |
| 552 | regression test for Gen issue 168 | 0 | mh with a StaticSelection on a static model does not crash | `mh/mh` with a selection (checked) | port (smoke test) | smoke test, ports |

#### `test/tilde_sugar.jl` (12 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 8 | tilde syntax smoke test (dynamic) | 2 | `{:coin => i} ~ bernoulli(p)` returns its value; the sum of two booleans; the score | `trace`; booleans count (checked: 2) | port 2 | D9 |
| 19 | tilde syntax smoke test (static) | 7 | tilde forms in a static body, call return values, `return {:ret} ~ …` | static `trace`/`splice` | port 7 | G1, closed (genclj-cbec) |
| 50 | tilde syntax desugars as expected (static) | 1 | desugar_tildes output | none | n/a 1 | Julia macro internals |
| 66 | tilde syntax preserved in quoted expressions | 2 | a quoted `x ~ normal(0, 1)` returned from a body | a quoted `(trace …)` is data in `gen` and `gen ^:static` | port 2 | G12, closed (genclj-9d9h) |

#### `test/inference/importance_sampling.jl` (16 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | importance sampling | 16 | importance_sampling and importance_resampling, internal and custom proposal: counts, normalization, lml not NaN, observations kept | `gen.inference.importance`; `importance-test/*` | port 16 |  |

#### `test/inference/train.jl` (8 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | SGD training | 6 | parameter gradients against finite differences of assess; train! learns the xor teacher | `learning/train`, `param-gradients`; `gradients-test/training-on-generated-data` | port 1 · statistical 5 | S: thresholds after 2000 epochs (slow) |
| 126 | lecture! | 2 | lecture! and lecture_batched! train q toward the posterior: score ≈ -0.21 | `learning/lecture`, `lecture-batched`; `lecture-trains-a-proposal-toward-the-posterior` | statistical 2 | -0.21 is the optimum, -½log 2π - log 0.3 - ½ = -0.2149; U4: the second assertion scores q, not q_batched |

#### `test/inference/variational.jl` (7 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | black box variational inference | 4 | BBVI recovers the prior's parameters (atol 0.001); VIMCO smoke runs | `vi/black-box-vi`, `black-box-vimco`; `vi-test/black-box-vi-finds-the-posterior` | statistical 4 | the score-function gradient is exactly 0 at q = p, so the tight band holds |
| 56 | minimal vae | 3 | BBVI and VIMCO training the model's parameter and q together | not in gen.clj (vi.clj: model_update is not here) | gap 3 | G4, to build (genclj-xjte) |

#### `test/inference/particle_filter.jl` (13 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 29 | hmm forward alg | 1 | the test's forward algorithm on a hand case | a test helper, or `enumerative-inference` | port 1 |  |
| 96 | particle filtering / custom proposal | 2 | a locally optimal proposal: log-ML within 0.02 of the forward algorithm (20,000 particles) | `pf/initialize`, `pf/step` (:proposal), `pf/maybe-resample`, `pf/log-ml-estimate`; `the-optimal-proposal-reduces-variance` | port 1 · statistical 1 | D2 (0-based categorical and steps) |
| 146 | particle filtering / default proposal | 2 | the same with the internal proposal | as above; `the-estimate-approaches-the-exact-marginal-likelihood` | port 1 · statistical 1 |  |
| 172 | particle filtering / PF state | 8 | copy, ==, hash and mutation independence of ParticleFilterState | persistent states | n/a 8 | Julia's mutable copy semantics |

#### `test/inference/mala.jl` (0 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | mala | 0 | smoke test | `mcmc/mala`; `mala-targets-the-posterior` | port (smoke test) | ports |

#### `test/inference/hmc.jl` (0 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | hmc | 0 | smoke test | `mcmc/hmc`; `hmc-targets-the-posterior` | port (smoke test) | ports |

#### `test/inference/map_optimize.jl` (0 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | map_optimize | 0 | smoke test | `mcmc/map-optimize`; `map-optimization-finds-the-mode` | port (smoke test) | ports |

#### `test/inference/elliptical_slice.jl` (0 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | elliptical slice | 0 | smoke test | `mcmc/elliptical-slice`; `elliptical-slice-targets-the-posterior` | port (smoke test) | ports (parity §4: Gen.jl's targets the wrong distribution) |

#### `test/inference/mh.jl` (3 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 13 | Involutive MCMC | 3 | an @transform involution and a Julia involution swap y and z; accepted, check=true | `mh/involutive-mh`, `deftransform`, `involution` with `:check?` (checked) | port 3 | D8 (a hand-written involution takes a key) |

#### `test/inference/trace_translators.jl` (5 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | DeterministicTraceTranslator | 2 | polar to cartesian and back with pair_bijections! and check=true | `transform/deterministic-translator`, `pair`, `array/atan2` | port 2 | G8, closed (genclj-3mx6) |
| 44 | SimpleExtendingTraceTranslator | 1 | weight = update weight - proposal score | `transform/extending-translator` (checked: equal) | port 1 |  |
| 73 | SymmetricTraceTranslator | 1 | a discrete/continuous involution with is_involution! and check=true | `transform/involution`, `involutive` | port 1 |  |
| 116 | GeneralTraceTranslator | 1 | continuous to discrete + auxiliary with pair_bijections! and check=true | `transform/translator` | port 1 |  |

#### `test/inference/kernel_dsl.jl` (2 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 28 | Kernel DSL | 2 | @pkern with metadata, @kern with a loop and its reversal | `kern/kernel`, `kern/kern`, `kern/reversal` (checked) | port 2 |  |

#### `test/inference/enumerative.jl` (31 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | enumerative inference | 31 | choice_vol_grid (discrete, continuous, 2-d), enumerative_inference weights, lml, MAP | `enum/choice-vol-grid`, `enumerative-inference`, `marginal`; `grids-follow-gen-jl` | port 27 · n/a 4 | U2: not included by runtests.jl, passes when run; N: size(grid) of the N-d grid (D12); exact == on grid values becomes close? |

#### `test/modeling_library/custom_determ.jl` (37 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | custom deterministic generative function with custom update and gradient | 23 | CustomDetermGF: simulate, generate, update (Unknown, VectorDiff), regenerate, gradients, its own trainable parameter | `gen.custom/deterministic` (:apply-with-state, :update-with-state, :gradient-with-state); `custom-test/updates-reuse-the-state` | port 23 | G5 (gen_fn.my_grad, scaled by 3), closed (genclj-2294): `:params` and `custom/with-params` |
| 126 | custom gradient GF | 2 | CustomGradientGF arg gradients, unscaled by accumulate | `deterministic` with :apply and :gradient (checked: [2 2]) | port 2 |  |
| 144 | custom update GF | 12 | CustomUpdateGF simulate and incremental updates | `deterministic` with :update-with-state (checked) | port 12 |  |

#### `test/modeling_library/distributions.jl` (138 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 4 | bernoulli | 2 | logpdf_grad of the probability | `array/grad` of `dc/log-prob` | port 2 |  |
| 19 | beta | 9 | range, out of support, finite densities at 0 and 1, gradients | `dist/beta` | port 9 | D6 (decided: exact, +Inf at 0 and 1; the port asserts it, the oracle records Gen.jl's finite values) |
| 43 | categorical | 7 | range, out of support, Distributions.jl's integer-probability bug, gradients | `dist/categorical` (0-based) | port 5 · n/a 2 | D2; N: Distributions.Categorical itself, and `nothing` for the value's gradient; finite_diff_vec over the probabilities steps 1e-7, since gen.clj checks that they sum to 1 within 1e-6 (rule 5) and Gen.jl's 1e-6 sits on that edge |
| 67 | gamma | 5 | range, out of support, gradients | `dist/gamma` | port 5 |  |
| 85 | inv_gamma | 5 | range, out of support, gradients | `dist/inv-gamma` | port 5 |  |
| 103 | normal | 5 | no overflow at 1e13 (exact -5e25 and gradients), gradients | `dist/normal` (checked: exact) | port 5 |  |
| 121 | zero-dimensional broadcasted normal | 6 | 0-d arrays: gradient shapes () and values | `dist/normal` (0-d values are host scalars on the JVM) | port 6 |  |
| 140 | array normal (trivially broadcasted: all args have same shape) | 6 | gradients of a 2×3 broadcasted normal | `dist/normal` with array parameters | port 6 |  |
| 166 | broadcasted normal | 6 + 8 throws | result shapes, DimensionMismatch cases, logpdf of broadcast vs expanded parameters | `dist/normal` (NumPy broadcasting); value-shape checks at a trace site (`gfi/assess`) | port 14 | D3: Julia's shapes reversed (checked); logpdf through assess, which sums and checks the value's shape |
| 217 | multivariate normal | 9 | gradients of x, mu and the symmetric cov | `dist/mvnormal` | port 9 | D3: the tape puts the cov gradient on the triangle the Cholesky reads; symmetrize |
| 237 | dirichlet | 20 | bounds, values against Distributions.jl, gradients through a softmax | `dist/dirichlet`; values from scipy (test/fixtures/generate.py) | port 20 | D5 (alpha < 0 throws, checked) |
| 292 | uniform | 5 | range, out of support, gradients | `dist/uniform` | port 5 |  |
| 310 | uniform_discrete | 3 | range, out of support, no gradients | `dist/uniform-discrete` | port 2 · n/a 1 | N: `nothing` gradients |
| 325 | piecewise_uniform | 8 | range, out of support, gradients of x, bounds and probs | `dist/piecewise-uniform` (checked) | port 8 | the probabilities' finite differences step 1e-7, as categorical's |
| 346 | beta uniform mixture | 6 | range, out of support, gradients | `dist/beta-uniform` | port 6 |  |
| 365 | geometric | 4 | range, out of support, gradient of p | `dist/geometric` | port 3 · n/a 1 | N: `nothing` |
| 381 | binom | 5 | range, out of support, gradient of p | `dist/binom` | port 3 · n/a 2 | N: `nothing` |
| 398 | neg_binom | 5 | range, out of support, gradients of r and p | `dist/neg-binom` (checked) | port 4 · n/a 1 | N: `nothing` |
| 415 | exponential | 4 | range, out of support, gradients | `dist/exponential` | port 4 |  |
| 431 | poisson | 4 | range, out of support, gradient of lambda | `dist/poisson` | port 3 · n/a 1 | N: `nothing` |
| 447 | laplace | 3 | gradients | `dist/laplace` | port 3 |  |
| 461 | cauchy | 3 | gradients | `dist/cauchy` | port 3 |  |

#### `test/modeling_library/choice_at.jl` (74 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 5 | choice_at combinator / assess | 2 | weight and value at key 3 | `comb/choice-at`; `combinators-test/call-at-and-choice-at` | port 2 |  |
| 13 | choice_at combinator / propose | 4 | weight and choices | as above | port 4 |  |
| 21 | choice_at combinator / generate | 9 | without and with a constraint | as above | port 9 |  |
| 52 | choice_at combinator / project | 2 | none and the key | as above | port 2 |  |
| 59 | choice_at combinator / update | 22 | changed args, same key, a constraint, another key | as above | port 22 |  |
| 104 | choice_at combinator / regenerate | 16 | not selected, selected, another key (weight 0, checked) | as above | port 16 |  |
| 142 | choice_at combinator / choice_gradients | 16 | argument and choice gradients; the key has none | `choice-gradients` (:argnums) | port 16 |  |
| 175 | choice_at combinator / accumulate_param_gradients! | 3 | argument gradients | `choice-gradients` (:argnums) | port 3 |  |

#### `test/modeling_library/call_at.jl` (74 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 9 | call_at combinator / assess | 2 | weight and value | `comb/call-at` | port 2 |  |
| 18 | call_at combinator / propose | 3 | weight and choices | as above | port 3 |  |
| 26 | call_at combinator / generate | 9 | without and with a constraint | as above | port 9 |  |
| 58 | call_at combinator / project | 2 | none and the key's choice | as above | port 2 |  |
| 65 | call_at combinator / force update | 25 | changed args, a constraint, another key | as above | port 25 |  |
| 115 | call_at combinator / free update | 16 | regenerate: not selected, selected, another key | as above | port 16 |  |
| 153 | call_at combinator / choice_gradients | 14 | argument and choice gradients with a retval gradient | `choice-gradients` (checked) | port 14 |  |
| 182 | call_at combinator / accumulate_param_gradients! | 3 | argument gradients | as above | port 3 |  |

#### `test/modeling_library/map.jl` (137 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 15 | map combinator / Julia call | 1 | calling a Map | IFn on MapGF | port 1 |  |
| 19 | map combinator / simulate | 3 | score; retval[i] is trace[i => :z] and trace[i] | `comb/map`; `map-basics` | port 3 | G1 (trace[i]), closed (genclj-cbec) |
| 30 | map combinator / generate | 2 | a constrained element | as above | port 2 |  |
| 41 | map combinator / propose | 1 | weight | as above | port 1 |  |
| 49 | map combinator / assess | 2 | weight and retval | as above | port 2 |  |
| 60 | map combinator / update | 74 | growing, shrinking, unchanged and custom-argdiff updates: choices, discard, score, weight, retval, retdiff | as above; `map-precise-hints-match-unknown-change` (cases checked) | port 74 | D1 (decided): an element whose value did not change is absent from the retdiff (checked); the port asserts gen.clj's |
| 232 | map combinator / regenerate | 45 | the same for regenerate | as above | port 45 | D1 (decided) |
| 345 | map combinator / choice_gradients | 6 | argument and choice gradients with a random retval gradient | `choice-gradients` | port 6 |  |
| 376 | map combinator / accumulate_param_gradients! | 3 | argument and parameter gradients | `choice-gradients`, `param-gradients` on the Map's trace (its kernel's store, genclj-umww) | port 3 | D7 |

#### `test/modeling_library/unfold.jl` (181 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 12 | unfold combinator / Julia call | 1 | calling an Unfold | IFn on UnfoldGF | port 1 |  |
| 16 | unfold combinator / simulate | 7 | choices, score, retval | `comb/unfold` with a static kernel; `unfold-basics` | port 7 |  |
| 41 | unfold combinator / generate | 10 | constraints at steps 1 and 3 | as above | port 10 |  |
| 72 | unfold combinator / propose | 7 | weight and retval | as above | port 7 |  |
| 92 | unfold combinator / assess | 5 | weight and retval | as above | port 5 |  |
| 114 | unfold combinator / update | 84 | growing, shrinking, unchanged, init- and params-changed updates with retdiffs | as above; `unfold-precise-hints-match-unknown-change` (every retdiff checked) | port 84 | the kernel is static, so Gen.jl's retdiffs are precise too |
| 327 | unfold combinator / regenerate | 59 | the same for regenerate | as above (checked) | port 59 |  |
| 481 | unfold combinator / test 1 - accumulate_param_gradients! | 5 | argument and parameter gradients of a dynamic kernel | `choice-gradients`, `param-gradients` on the Unfold's trace (genclj-umww) | port 5 | D7 |
| 523 | unfold combinator / test 2 - accumulate_param_gradients! | 1 | the gradient of 2^10 x through ten steps | `choice-gradients` (checked: 1024) | port 1 | `nothing` retgrads become zeros |
| 538 | unfold combinator / choice gradients | 2 | choice gradients propagate through the state, with and without a retval gradient (loops of 10) | `choice-gradients` | port 2 |  |

#### `test/modeling_library/recurse.jl` (122 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | recurse node numbering | 54 | get_child, get_child_num, get_parent for max_branch 1, 2 and 3 | `comb/get-child`, `get-child-num`, `get-parent` (0-based) | port 54 | G13, closed (genclj-cbec); D2 |
| 66 | simple pcfg | 68 | the 20 likeliest strings appear in 1,000 runs (twice); generate with every choice; updates of a value and of the structure | `comb/recurse`; `recurse-builds-and-folds-the-tree`, `recurse-updates-only-the-changed-path` (generate and the structure update checked) | port 48 · statistical 20 | D2, D13 |

#### `test/modeling_library/switch.jl` (51 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 10 | Trace / switch trace | 3 | a SwitchTrace built by hand | internal | n/a 3 | Gen.SwitchTrace's constructor |
| 35 | Bare combinator / simulate | 2 | scores of both branches | `comb/switch`; `switch-basics` | port 2 | D2 |
| 42 | Bare combinator / generate | 2 | a constrained branch | as above | port 2 |  |
| 51 | Bare combinator / assess | 1 | weight | as above | port 1 |  |
| 58 | Bare combinator / propose | 1 | weight | as above | port 1 |  |
| 63 | Bare combinator / update | 2 | old score = new score - weight, with a constraint at an address the switch lacks | `switch-branches-share-addresses` | port 2 | D11: the constraint is unvisited (loud, checked); the port drops it |
| 78 | Bare combinator / regenerate | 3 | old score = new score - weight under regenerate | `regenerate-weight-is-the-retained-density-ratio` (gen.clj weighs 0, checked) | oracle 3 | U1: checked on Gen.jl's recorded moves (genclj-c5rl) |
| 97 | Bare combinator / choice gradients | 2 | choice gradients in both branches (loop of 4) | `choice-gradients` | port 2 |  |
| 134 | Hierarchy / simulate | 1 | a Switch called from a gen function | as above | port 1 |  |
| 139 | Hierarchy / generate | 2 | weight of a constrained choice | as above | port 2 |  |
| 148 | Hierarchy / assess | 1 | weight | as above | port 1 |  |
| 155 | Hierarchy / propose | 1 | weight of the proposed choices | as above | port 1 |  |
| 160 | Hierarchy / update | 2 | old = new - weight, same and other branch | as above | port 2 |  |
| 171 | Hierarchy / regenerate | 3 | old = new - weight with select() | as above (branches share :z, so Eq. 4.1 agrees) | port 3 |  |
| 182 | Hierarchy / choice gradients | 2 | choice gradients in both branches (loop of 4) | `choice-gradients` | port 2 |  |
| 199 | Hierarchy / accumulate parameter gradients | 2 | each branch's std gradient | `param-gradients` of the Switch's trace, by branch, each holding its std (genclj-umww) | port 2 | D7, D11 (its constraint at :z is unvisited) |
| 236 | More complex hierarchy / simulate | 2 | branches with a nested call | as above | port 2 |  |
| 243 | More complex hierarchy / generate | 4 | weight of a constrained choice | as above | port 4 |  |
| 256 | More complex hierarchy / assess | 2 | weight | as above | port 2 |  |
| 266 | More complex hierarchy / propose | 2 | weight of the proposed choices | as above | port 2 |  |
| 273 | More complex hierarchy / update | 2 | old = new - weight across branches | as above | port 2 |  |
| 284 | More complex hierarchy / regenerate | 4 | old = new - weight under regenerate, a branch that drops a call | Eq. 4.1 differs (checked) | port 1 · oracle 3 | U1: checked on Gen.jl's recorded moves (genclj-c5rl) |
| 297 | More complex hierarchy / choice gradients | 1 | the first branch's choice gradient (loop of 4) | `choice-gradients` | port 1 |  |
| 331 | More complex hierarchy to test discard / update | 4 | the discard of a branch change, deep addresses | `switch-branches-share-addresses` | port 4 |  |

#### `test/modeling_library/dist_dsl.jl` (43 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 1 | dist DSL (untyped) | 29 | pushforwards (exp, + 1, broadcast parameters, mvnormal), labels from an Array, a Dict and an enum, 1 - uniform | `gen.dist.dsl/dist`; `dist-dsl-test/*` | port 17 · gap 12 | G7 (labels, checked), to build (genclj-414l); line 29's covariance 2·ones(2,2) is singular: Gen.jl gives 15.24 (U3), gen.clj rejects it (D15), so line 29's 4 assertions port as the error |
| 72 | dist DSL (typed) | 8 + 6 throws | typed @dist parameters: labels of a type, MethodErrors for wrong types | `dist` (untyped) | port 3 · n/a 6 · gap 5 | G7, to build (genclj-414l); N: Julia type dispatch (MethodError) |

#### `test/modeling_library/mixture.jl` (52 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | fixed mixture of different distributions | 11 | HeterogeneousMixture of uniform and beta: logpdf, gradients | `dist/heterogeneous-mixture` (checked) | port 9 · n/a 2 | N: has_output_grad, has_argument_grads |
| 44 | mixture of normals | 20 | HomogeneousMixture of normals: logpdf, gradients, through the tape | `dist/homogeneous-mixture`; `tape-test/log-density-gradients` | port 18 · n/a 2 | D3 (component axis last in both) |
| 104 | mixture of binomial | 9 | HomogeneousMixture of binomials | as above (checked) | port 6 · n/a 3 | N: grad flags and `nothing` |
| 138 | mixture of multivariate normals | 12 | HomogeneousMixture(mvnormal, [1, 2]): logpdf, gradients, a rotated covariance | `(dist/homogeneous-mixture mvnormal [1 2])`; `dist-test/mixtures-of-multivariate-families` (checked) | port 10 · n/a 2 | D3 (means [K d], covariances [K d d]); N: has_output_grad, has_argument_grads |

#### `test/modeling_library/product.jl` (14 assertions)

| Line | Testset | Asserts | What it checks | gen.clj counterpart | Category | Notes |
|---:|---|---:|---|---|---|---|
| 3 | product of discrete distributions | 4 | is_discrete, logpdf, gradients | `dist/product` (checked) | port 3 · n/a 1 | N: grad flags |
| 32 | product of continuous distributions | 6 | is_discrete, logpdf, gradients | as above | port 5 · n/a 1 | N: grad flags |
| 64 | product of dissimilarly-typed distributions | 4 | is_discrete, logpdf, gradients | as above | port 3 · n/a 1 | N: grad flags |

## 3. Totals

### Totals per file

| File | Testsets | Assertions | port | oracle | statistical | n/a | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `autodiff.jl` | 1 | 2 | 2 | 0 | 0 | 0 | 0 |
| `diff.jl` | 5 | 32 | 32 | 0 | 0 | 0 | 0 |
| `selection.jl` | 3 | 35 | 34 | 0 | 0 | 1 | 0 |
| `assignment.jl` | 17 | 132 | 112 | 0 | 0 | 6 | 14 |
| `gen_fn_interface.jl` | 2 | 5 | 5 | 0 | 0 | 0 | 0 |
| `dsl/macros.jl` | 2 | 14 | 6 | 0 | 0 | 8 | 0 |
| `dsl/dynamic_dsl.jl` | 14 | 88 | 77 | 0 | 0 | 10 | 1 |
| `dsl/static_dsl.jl` | 19 | 175 | 60 | 0 | 0 | 72 | 43 |
| `optional_args.jl` | 2 | 58 | 0 | 0 | 0 | 58 | 0 |
| `static_ir/static_ir.jl` | 11 | 138 | 125 | 0 | 0 | 13 | 0 |
| `tilde_sugar.jl` | 4 | 12 | 11 | 0 | 0 | 1 | 0 |
| `inference/importance_sampling.jl` | 1 | 16 | 16 | 0 | 0 | 0 | 0 |
| `inference/train.jl` | 2 | 8 | 1 | 0 | 7 | 0 | 0 |
| `inference/variational.jl` | 2 | 7 | 0 | 0 | 4 | 0 | 3 |
| `inference/particle_filter.jl` | 4 | 13 | 3 | 0 | 2 | 8 | 0 |
| `inference/mala.jl` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `inference/hmc.jl` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `inference/map_optimize.jl` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `inference/elliptical_slice.jl` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `inference/mh.jl` | 1 | 3 | 3 | 0 | 0 | 0 | 0 |
| `inference/trace_translators.jl` | 4 | 5 | 5 | 0 | 0 | 0 | 0 |
| `inference/kernel_dsl.jl` | 1 | 2 | 2 | 0 | 0 | 0 | 0 |
| `inference/enumerative.jl` | 1 | 31 | 27 | 0 | 0 | 4 | 0 |
| `modeling_library/custom_determ.jl` | 3 | 37 | 37 | 0 | 0 | 0 | 0 |
| `modeling_library/distributions.jl` | 22 | 138 | 130 | 0 | 0 | 8 | 0 |
| `modeling_library/choice_at.jl` | 8 | 74 | 74 | 0 | 0 | 0 | 0 |
| `modeling_library/call_at.jl` | 8 | 74 | 74 | 0 | 0 | 0 | 0 |
| `modeling_library/map.jl` | 9 | 137 | 137 | 0 | 0 | 0 | 0 |
| `modeling_library/unfold.jl` | 10 | 181 | 181 | 0 | 0 | 0 | 0 |
| `modeling_library/recurse.jl` | 2 | 122 | 102 | 0 | 20 | 0 | 0 |
| `modeling_library/switch.jl` | 24 | 51 | 42 | 6 | 0 | 3 | 0 |
| `modeling_library/dist_dsl.jl` | 2 | 43 | 20 | 0 | 0 | 6 | 17 |
| `modeling_library/mixture.jl` | 4 | 52 | 43 | 0 | 0 | 9 | 0 |
| `modeling_library/product.jl` | 3 | 14 | 11 | 0 | 0 | 3 | 0 |
| **all** | **195** | **1699** | **1372** | **6** | **33** | **210** | **78** |

### Totals per category

| Category | Assertions | Share | Testsets wholly in it |
|---|---:|---:|---:|
| port | 1372 | 80.8 % | 124 |
| oracle | 6 | 0.4 % | 1 |
| statistical | 33 | 1.9 % | 2 |
| n/a | 210 | 12.4 % | 14 |
| gap | 78 | 4.6 % | 4 |

Mixed testsets: 43. Testsets without assertions: 7 (smoke tests and empty testsets).

The first mapping (2026-10-20) counted port 1,262 (74.3 %), n/a 110 and
gap 288. The owner's decisions then moved D1 (16) and D6 (4) to port, and
G3 (60) and G15 (40, with 7 of `static_dsl.jl:466`'s assertions, which are
the IR's) to n/a; G1, G5 and G8–G14 were built (89 to port, and one of
`selection.jl:49`'s was a port the first count took for a gap). The 78 left
each wait on a bean: G2's static rewrites (42, genclj-lgij), G7's
non-numeric labels (17, genclj-414l), D4's loud lookups (16, genclj-ocwu)
and G4's model_update (3, genclj-xjte). Counted against the ports'
pending lists at batch 4's head (17fd8bd), with G5's port (genclj-2294).

### The n/a assertions, by reason

| Reason | Assertions |
|---|---:|
| Julia types and `(grad)` flags in the static IR, `get_return_type` | 21 |
| Gradient flags and `nothing` gradients of distributions (`has_output_grad`, `has_argument_grads`, a discrete value's gradient) | 20 |
| Julia syntax and macro machinery (statement macros that bind a variable, `Expr(:gentrace)`, `@doc`, one-line definitions, modules, `end` in an index, `desugar_tildes`) | 18 |
| Internal APIs (AddressVisitor, a hand-built SwitchTrace, the IR's text display, `static_get_submap`, the static choice-map kind, `isa DynamicSelection`) | 17 |
| The static options `diffs` and `nojuliacache` (D10) | 12 |
| Mutable copy semantics of ParticleFilterState | 8 |
| Julia type dispatch (`MethodError` for typed `@dist` parameters) | 6 |
| The N-d shape of an enumeration grid (D12) | 4 |
| Gradient accumulators (`get_param_grad` after a reset, D7) | 4 |
| No default positional arguments (G3, the owner's decision): `optional_args.jl` and `dynamic_dsl.jl:8` | 60 |
| The static IR's internal shape (G15, the owner's decision): argument and per-expression nodes, parameter nodes, node identity, the distribution and callee objects | 40 |
| **all** | **210** |

## 4. Differences and gaps

### 4.1 Differences of behavior

| ID | Difference | Documented? | Assertions | Proposal |
|---|---|---|---:|---|
| D1 | **Retdiffs are more precise.** A dynamic function's retdiff is no-change when its return value equals the previous one; Gen.jl's dynamic language always returns UnknownChange (`src/dynamic/update.jl:200`, `regenerate.jl:141`). Map's and Unfold's element retdiffs follow, so an unchanged element is absent from a VectorDiff, and a Map update with nothing changed returns no-change where Gen.jl returns a VectorDiff naming the element. A splice also turns a caller's argdiff into no-change for an argument identical to the previous call's, where Gen.jl trusts the argdiff, so a callee given UnknownChange for an unchanged value does not rerun. Sound (a diff only claims what holds), checked. | DESIGN §9 (the owner, 2026-10-20: kept) | 16 port | Decided: the ports assert gen.clj's diffs, and that they are consistent (`diff/consistent?`) |
| D2 | **0-based indices.** Map and Unfold addresses, Recurse nodes, categorical's values and Switch's branch index. | §9, with categorical's values and Switch's branch index (genclj-fevk) | (port) | — |
| D3 | **Row-major arrays, NumPy broadcasting.** Julia pads shapes on the right and is column-major; the ports reverse Julia's shapes (checked on the broadcasting cases). A covariance gradient from the tape sits on the triangle the Cholesky reads; the port symmetrizes it to compare with Gen.jl's. | `docs/array-api.md` (NumPy broadcasting) | (port) | A porting note |
| D4 | **Lenient choice-map and trace accessors.** `cm/get-submap` at a value returns the Value and through a value EMPTY; `cm/get-choice`, `cm/get-value` and `(tr addr)` at a missing or node address return nil; `cm/has-submap?` is true at a value. Gen.jl throws KeyError, or answers false. | DESIGN §9 (the owner, 2026-10-20: loud) | 16 gap until built | Decided: `cm/get-choice`, `cm/get-value` and `(tr addr)` throw at a missing or node address, as Gen.jl's KeyError; Clojure's `get` stays the nil-returning lookup. To build (genclj-ocwu), then the 16 port |
| D5 | **Invalid parameters throw** (dirichlet α ≤ 0, normal std 0) where Gen.jl returns −Inf or samples the mean. | CLAUDE.md rule 5 | (port) | The ports assert the error |
| D6 | **Beta at 0 and 1.** gen.clj gives the exact density, +Inf when α < 1 (β < 1); Gen.jl substitutes a finite value "as if x = eps/e" (`src/modeling_library/distributions/beta.jl`). | DESIGN §9 (the owner, 2026-10-20: exact) | 4 port | Decided: the port asserts +Inf; the oracle records Gen.jl's finite values (`beta-endpoints.edn`) |
| D7 | **One parameter store per model; gradients are values.** No per-function parameters, accumulators or scale factor (`get_param_grad`, `zero_param_grad!`, `set_param!`, `accumulate_param_gradients!(tr, g, scale)`); argument gradients come from `choice-gradients` with `:argnums`. | §9 | (port), 4 n/a | — |
| D8 | An involution takes a key, a fifth argument. | §9 | (port) | — |
| D9 | Booleans count in arithmetic (`tilde_sugar.jl`'s coin sum, checked). | §9 | (port) | — |
| D10 | The static language always tracks diffs and keeps node values (Gen.jl's default); `(static, diffs)` and `nojuliacache` have no counterpart, so rerun counts that assume `nojuliacache` are n/a. | §9 describes the behavior | 12 n/a | — |
| D11 | Unvisited constraints are loud. Gen.jl's dynamic `generate` ignores them (only its `update` checks, `src/dynamic/update.jl:197`), and `switch.jl` constrains addresses its models lack (`:x => :z` on a bare Switch, `:z` under `bam`). The ports drop those constraints. | §9 | (port) | — |
| D12 | An enumeration grid is a flat vector in row-major order, without an N-d shape. | §9 (the order) | 4 n/a | — |
| D13 | Recurse takes one argument (no `root_idx`). | §9 | (port) | — |
| D14 | `to-array`'s order: every leaf path sorted, where Gen.jl puts a node's values before its submaps, each group sorted. The suite's cases give the same order (checked on one that differs). | `gen.choicemap`'s namespace doc ("Leaf order is canonical") | (port) | — |
| D15 | **A covariance singular in floating point is an error.** `mvnormal` rejects a covariance whose Cholesky factorization fails, or whose condition leaves the density no correct digits: trace(H⁻¹) ≥ 1/ε, with H the correlation matrix of L·Lᵀ and ε that of the dtype the density is computed in (float32 on MLX, host data included). The bound is the measured one (`docs/research/covariance/f32_gp.py`): a float32 log density's error tracks (0.05–0.4)·ε·trace(H⁻¹) on squared-exponential kernels of 50 to 1,000 points, so the bound rejects where the error nears 0.1 nats; the worst case's 1/(n·ε), the first version, was n times too strict. Gen.jl, through LAPACK's potrf, rejects only a failed factorization, so it accepts `dist_dsl.jl:29`'s 2·ones(2, 2) (U3), whose trace(H⁻¹) is 2/ε. The bound holds at every precision, so in float32 a Gaussian-process kernel needs a jitter at float32's scale: the 1,000-point squared-exponential kernel (length scale 0.1 on [0, 1]) passes in float32 with 1e-3 on its diagonal (ε·trace(H⁻¹) = 0.12, an error of 0.009 nats) and fails with 1e-5 (11.7; 0.75 nats), and in float64 passes with 1e-10. When the factor exists, the error suggests the smallest jitter of the form 1, 2 or 5 × 10ᵏ that passes (`:suggested-jitter`): 2e-4 for the 1e-5 kernel. | §9 (the owner's decisions: F1, float32, the bound at 1/ε) | (port) | The port asserts the error |
| D16 | **A trace in a branch is not static.** In Gen.jl's static language a `@trace` in a ternary's branch (or an `&&`, `\|\|`) is hoisted into a node and runs whether or not the branch is taken, so the static function is a different model from the same body in the dynamic language. gen.clj's `gen ^:static` rejects it with a compile error naming the call (G2's hoisting covers only positions evaluated exactly once). No suite assertion depends on it. | §9 (static rewrites, the owner's decision of 2026-10-21) | — | — |

### 4.2 Gaps

| ID | Gap | What gen.clj has | Size | Assertions | Covered by a documented difference? |
|---|---|---|---|---:|---|
| G1 | `trace[addr]` at a call gives the call's return value (also Map's `trace[i]`); `get_call(tr, addr).subtrace` | Closed (genclj-cbec, DESIGN §9): `(tr addr)` is the call's return value where no choice is (`(tr [])` the trace's), and `(trace/subtrace tr addr)` its trace. Before, `(tr addr)` was nil at a call | — | — (its 7 port now) | — |
| G2 | The static language rewrites a computed address (`:x => i`) into a `choice_at`/`call_at` call node, and hoists `@trace` out of expressions (a tuple, a ternary, a distribution's argument) | Both are compile errors in `gen ^:static` (checked); written as separate bindings, or with `comb/choice-at`, they work (checked). Decided, to build (DESIGN §9, static rewrites; genclj-lgij) | medium | 42 (`static_dsl.jl:466`'s other 7 are the IR's, G15) | Decided |
| G3 | Optional positional arguments with defaults (`@gen function foo(x, y=2)`), in the trace's args too | One parameter vector per `gen`; a rest parameter `[x & [y z]]` covers calls, update and regenerate with any number of arguments (checked), but the trace's args stay as given | — | 60 n/a | n/a by the owner's decision (DESIGN §9: Clojure has none) |
| G4 | `black_box_vi!`/`black_box_vimco!` with `model_update`, training the model's parameters with q's (the minimal VAE) | Only q's store is trained (`gen.inference.vi`'s docstring says so); to build (genclj-xjte) | medium | 3 | No |
| G5 | A custom deterministic function's own trainable parameters (`accumulate_param_gradients_determ!`, the scale factor) | Closed (genclj-2294, DESIGN §9): `:params` names the parameters a deterministic function reads from the store in effect (its own, `custom/with-params`, over the caller's); `param-gradients` differentiates them through its code or by `:param-gradient-with-state` | — | — (its 1 ports now) | — |
| G6 | `HomogeneousMixture` of a multivariate family (`HomogeneousMixture(mvnormal, [1, 2])`, dims per argument) | Closed (b9eb9d2): `(homogeneous-mixture family dims)` takes Gen.jl's dims, slices each parameter on the axis before its own (means [K d], covariances [K d d]) and mixes the K components' densities of the whole value. Before, it took scalar families, and with mvnormal `dc/log-prob` returned a wrong number (F2) | — | — (its 10 port now) | — |
| G7 | `@dist` with non-numeric labels (`labels[categorical(p)]` over symbols, a Dict of labels, enums, user types) | Numeric labels through `array/take` work (checked); keywords fail ("equal: not an array: :a"). §9: "Non-numeric labels wait for parameters that are not arrays"; to build (genclj-414l) | medium | 17 | Noted in §9 as pending |
| G8 | `atan2` (and `asin`, `acos`, `atan`, `sinh`, `cosh`) in gen.array, differentiable and on MLX | Closed (genclj-3mx6, 241f0ef): `array/atan2`, `asin`, `acos`, `atan`, `sinh`, `cosh`, differentiable and on MLX | — | — (its 2 port now) | — |
| G9 | Paths in selections: `(:x => :u) in s`, `s[:y => :z]`, `set_subselection!` | Closed (genclj-cbec, DESIGN §9: a vector is a path, everywhere): `selected?` and `get-subselection` take an address or a path, and `sel/set-subselection` is `set_subselection!`, persistent. Before, a vector path silently answered false and none | — | — (its 17 port now; the first count said 18) | — |
| G10 | `merge` of one or of n choice maps | Closed (genclj-cbec): `cm/merge` takes 0 to n choice maps | — | — (its 4 port now) | — |
| G11 | A user macro at a static binding's head that expands to `(trace …)` | Closed (genclj-9d9h): `static/analyze` macroexpands a binding's head before classifying it, as the dynamic `gen` does | — | — (its 1 ports now) | — |
| G12 | A static body returning quoted data that mentions `trace` | Closed (genclj-9d9h): quoted data is data | — | — (its 1 ports now) | — |
| G13 | Recurse's numbering helpers `get_child`, `get_child_num`, `get_parent` | Closed (genclj-cbec): `comb/get-child`, `get-child-num` and `get-parent`, 0-based, their arguments checked | — | — (its 54 port now) | — |
| G14 | `fill` of a tracked scalar under autodiff | Closed (genclj-3mx6, DESIGN §9: filling with an array is loud): `array/full` takes a scalar fill, a tape value too; an array fill is an ex-info. Before, a tape value threw ClassCastException | — | — (its 2 port now) | — |
| G15 | The static IR's shape: argument nodes, one Julia node per argument expression, parameter nodes, node identity, the distribution or callee object | `static/ir`: one node per binding, inputs as names, the source form | — | 40 n/a (33, and `static_dsl.jl:466`'s 7) | n/a by the owner's decision (DESIGN §9: the static IR's internal shape is not ported) |

### 4.3 Upstream findings, run

Run on this Mac on 2026-10-20, with the owner's go-ahead: Homebrew's Julia
1.12.6 (Apple M4), the installed Gen 0.4.8 (`~/.julia/packages/Gen/mP0Sq`,
whose `src/` and `test/` equal the tag's), Distributions 0.25.123. Offline,
from a scratch project and depot, single-threaded, `nice -n 15`. The
scripts: `test/fixtures/genjl/run_suite.jl` (the recording test set, with
the project's `Project.toml` and `Manifest.toml` beside it), and the
reproductions `docs/research/genjl-upstream/u1.jl`, `u3.jl`, `u4.jl` and
`u5.jl`.
Regenerate the record with `env JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=1
julia --project=test/fixtures/genjl test/fixtures/genjl/run_suite.jl
<Gen's test dir> test/fixtures/genjl`; tests never need Julia.

**The suite.** `runtests.jl`, then `inference/enumerative.jl`, under a test
set that records each testset's pass, fail, error and broken counts and
every problem's line and message (`test/fixtures/genjl/upstream-run.edn`):
**1,863 pass, 0 fail, 0 error, 0 broken**, in 214 testsets, 169 s. Of
them 197 are leaves: §2's 195, of which `gen_fn_interface.jl`'s two run once
per modeling language.
Executed counts differ from §3's sites where loops repeat an assertion, or
where an if/else runs one branch (`kernel_dsl.jl`):

| File | Sites | Passed | File | Sites | Passed |
|---|---:|---:|---|---:|---:|
| `autodiff.jl` | 2 | 2 | `inference/trace_translators.jl` | 5 | 5 |
| `diff.jl` | 32 | 32 | `inference/kernel_dsl.jl` | 2 | 1 |
| `selection.jl` | 35 | 35 | `inference/enumerative.jl` | 31 | 31 |
| `assignment.jl` | 132 | 143 | `modeling_library/custom_determ.jl` | 37 | 37 |
| `gen_fn_interface.jl` | 5 | 10 | `modeling_library/distributions.jl` | 138 | 138 |
| `dsl/macros.jl` | 14 | 14 | `modeling_library/choice_at.jl` | 74 | 74 |
| `dsl/dynamic_dsl.jl` | 88 | 150 | `modeling_library/call_at.jl` | 74 | 74 |
| `dsl/static_dsl.jl` | 175 | 175 | `modeling_library/map.jl` | 137 | 139 |
| `optional_args.jl` | 58 | 58 | `modeling_library/unfold.jl` | 181 | 199 |
| `static_ir/static_ir.jl` | 138 | 138 | `modeling_library/recurse.jl` | 122 | 142 |
| `tilde_sugar.jl` | 12 | 12 | `modeling_library/switch.jl` | 51 | 72 |
| `inference/importance_sampling.jl` | 16 | 22 | `modeling_library/dist_dsl.jl` | 43 | 43 |
| `inference/train.jl` | 8 | 12 | `modeling_library/mixture.jl` | 52 | 55 |
| `inference/variational.jl` | 7 | 7 | `modeling_library/product.jl` | 14 | 20 |
| `inference/particle_filter.jl` | 13 | 17 | **all** | **1,699** | **1,863** |
| `inference/mh.jl` | 3 | 6 | | | |

The four smoke tests (mala, hmc, map_optimize, elliptical slice) and Gen
issue 168's regression test ran without error. `recurse.jl:223`, marked
`# FAIL`, passes.

- **U1. Switch's regenerate weight: confirmed.**
  `src/modeling_library/switch/regenerate.jl` has two `process!` methods. With
  a NoChange index argdiff it calls the branch's own regenerate. With any
  other argdiff, UnknownChange included, it generates the branch from the
  unselected old choices and weighs `w_generate − project(old, ¬S) +
  project(new, S) − project(old, S)`. Two terms depart from Eq. 4.1 (thesis
  p. 144) with forward sampling. A selected choice drawn from its prior should
  weigh 0, as the NoChange path and Gen.jl's dynamic regenerate give, but
  weighs log p(z′) − log p(z). An unselected old choice the new branch drops
  is subtracted, where it should cancel against the reverse move. The dynamic
  language calls every callee with UnknownChange argdiffs
  (`src/dynamic/regenerate.jl:78`), so this is the path of every Switch in a
  `@gen` function. `u1.jl` shows each part:
  - *The weight* (10 regenerations each of `select(:x => :z)`, z ~ N(0, 1)
    under `Switch(leaf, other)` at `:x`). Called from a `@gen` function, the
    weight equals log p(z′) − log p(z) to the last bit (it reaches 2.08);
    called from a static function (a constant index is NoChange) or with
    `leaf` spliced directly, it is 0. The bare Switch weighs 0 with a
    NoChange index argdiff and log p(z′) − log p(z) exactly with
    UnknownChange.
  - *A dropped choice.* From the branch with z ~ N(0, 1) and w ~ N(0, 1) to
    one with z ~ N(1, 2), `select()`: the weight is 1.260301, which is Eq.
    4.1's 0.233634 (w cancels) less log p(w).
  - *The consequence.* z ~ N(0, 1) under the Switch, y ~ N(z, 1), y = 1.5;
    selection MH on z, 200,000 steps after 1,000, standard errors from 200
    batch means. The exact posterior is N(0.75, 0.5); the p² reading
    predicts p(z)² p(y | z) ∝ N(0.5, 1/3).

    | Caller | Mean ± SE | Variance ± SE | Acceptance |
    |---|---|---|---:|
    | `@gen` function → Switch | 0.4987 ± 0.0018 | 0.3350 ± 0.0015 | 0.572 |
    | static function → Switch | 0.7472 ± 0.0028 | 0.5049 ± 0.0025 | 0.534 |
    | `@gen` function → `leaf`, no Switch | 0.7500 ± 0.0027 | 0.5030 ± 0.0025 | 0.534 |

    The first chain's mean is 142 standard errors from the exact 0.75 and
    0.7 from 0.5; its variance 107 from 0.5 and 1.1 from 1/3. The other two
    are within 2 standard errors of the exact posterior.

  `switch.jl:78` and `:284` assert this weight, as `old_score ≈ new_score −
  w`, so they pass upstream. gen.clj's regenerate is Eq. 4.1 (it weighs 0 in
  the bare case, checked), so their 6 assertions stay the oracle's: a fixture
  of Gen.jl's weights on recorded choices, against which the port shows both
  formulas. Worth reporting upstream (the owner's call).
- **U2: confirmed.** `test/inference/enumerative.jl` is not included by
  `test/inference/inference.jl`, so Gen.jl's CI never runs it. Run, its 31
  assertions pass.
- **U3: refuted.** `dist_dsl.jl:29` evaluates `mvnormal(zeros(2), ones(2), 2
  .* ones(2, 2))`, a singular covariance, and the testset passes (29 of 29).
  Distributions.jl's checked Cholesky succeeds: rounding leaves a positive
  pivot, the factor's diagonal is [1.414, 2.1e-8], and the log density is
  15.240802 on both sides of the `==` (15.490802 at [1, 1], on the line the
  matrix spans; −4.5e15 at [2, 0], off it). gen.clj gave the same 15.24
  (F1). The assertion holds but compares a density that has no meaning, so
  gen.clj now rejects the matrix, by the owner's decision (D15), and the
  port asserts the error on both sides.
- **U4: confirmed, by reading and by a run.** `train.jl:176`, after training
  `q_batched` with `lecture_batched!`, scores `q`, the first part's proposal.
  `u4.jl` runs the testset's code with its seed. Line 155's score is −0.2165
  (θ −2.0022, log σ −1.2093; the optimum is −½ log 2π − log 0.3 − ½ =
  −0.2150). Line 176 as written gives −0.2053, and −0.2250 with q_batched
  never trained, also within the test's ±0.05 of −0.21: the assertion does
  not depend on q_batched. q_batched itself does not reach the optimum:
  after its 100 iterations θ = −5.6965 and log σ = 1.2531, and its own score
  is −2.7342 (untrained: −2.9625). The likely cause is the step:
  `lecture_batched!` accumulates the gradient summed over the batch of
  1,000, so `FixedStepGradientDescent(0.001)` steps 1.0 along the mean
  gradient, ten times the first part's 1e-4 × 1,000 = 0.1.
- **U5: Switch's generate leaves its trace's score at 0** (found writing the
  oracle, 2026-10-21; by reading and by a run). `process!` in
  `src/modeling_library/switch/generate.jl` never adds the branch's score,
  as `simulate.jl`'s does, so `get_score` of a Switch trace from `generate`
  is 0, and a `@gen` caller's score lacks the Switch's choices. `update`
  with an UnknownChange index argdiff, the path of every Switch called from
  `@gen` code, subtracts that score (`update.jl:95`), so its weight is off
  by the old trace's log density. `u5.jl`: generate with z = 7 gives score
  0 where the log density, and `project(tr, AllSelection())`, are −2.9064;
  an update that changes nothing weighs −2.9064 after generate and 0 after
  simulate; from a caller, moving y from 6 to 6.5 weighs −2.531 where the
  exact weight is 0.375. Gen.jl's tests start their updates and
  regenerates from simulate, so they pass. generate's own weights are
  right; an update after it is not (by reading, a particle filter of a
  model with a Switch is one: `initialize_particle_filter` generates and
  each step updates). The oracle's Switch fixture starts from simulate.
  Worth reporting upstream with U1 (the owner's call).

### 4.4 gen.clj findings: errors that should be loud

| ID | Finding | Checked |
|---|---|---|
| F1 | `dist/mvnormal`'s positive-definiteness check looked for NaN in the Cholesky factor, so the singular `2·ones(2, 2)` passed and `log-prob` returned 15.24, as Gen.jl's 15.240802 (U3). Fixed in 2beab42 by the owner's decision (D15): a covariance singular in floating point is an error, at every precision, so in float32 a GP kernel needs a jitter at float32's scale; the error suggests one (7e85fb3) | yes |
| F2 | `dc/log-prob` of `(homogeneous-mixture mvnormal)` returned −4.135 where the density is −4.754: nothing rejected a multivariate family outside a trace site. Fixed in b9eb9d2 with G6, as were the mixtures' samplers, whose component pick met a vector draw's own axis | yes |
| F3 | `sel/selected?`, `sel/get-subselection` and `cm/set-submap` took a vector path as one (absent) address, silently; `cm/set-submap` then held a vector key. Fixed in genclj-cbec (ad6b675) by the owner's decision (a vector is a path, everywhere; with G9) | yes |
| F4 | `array/full` of a tape value threw ClassCastException. Fixed in genclj-3mx6 (241f0ef; with G14) | yes |
| F5 | A quoted `(trace …)` in a static body was a compile error. Fixed in genclj-9d9h (fd20b6b; G12) | yes |

## 5. The oracle

Nearly every expected value is in the tests or follows from their models, so
Gen.jl is needed for few assertions. Its uses are four; the last ran on
2026-10-20 (§4.3), with U1–U4:

1. **Values for assertions (2 testsets, 6 assertions).** `switch.jl:78` and
   `:284` (U1). Fixture: for the bare Switch (`bang0`/`fuzz0`) and the nested
   one (`bang2`/`fuzz2` under `bam2`), a simulated initial trace (a
   generated one has score 0, U5), then 10 regenerations per case (the
   test's args, argdiffs and selections, plus the plain `mh(tr, select(:x =>
   :z))` move): old and new choices, scores, and Gen.jl's weight. The
   gen.clj test assesses both choice maps, shows that Gen.jl's weight is
   `score(t′) − score(t)` on them and that Eq. 4.1 gives gen.clj's, and
   records the difference in §9. (U1's p² is already shown by an MH chain,
   §4.3.) Done: `switch_test.clj`'s `regenerate-weights` (genclj-c5rl).
2. **Values for a decision (1 testset, 4 assertions).** `distributions.jl:19`:
   `logpdf(beta, x, a, b)` for x ∈ {0, eps, 1 − eps, 1} and (a, b) ∈ {(0.5,
   0.5), (0.5, 1.5), (1.5, 0.5), (1, 1), (2, 2)}, if the owner wants Gen.jl's
   convention (D6), or to state the difference.
3. **Cross-checks for the statistical testsets (6).** `train.jl:1` (θ₁…θ₅
   after training), `train.jl:126` (the trained score; its exact optimum is
   −0.2149), `variational.jl:1` (the four parameters), `particle_filter.jl:96`
   and `:146` (log-ML; the exact value is the forward algorithm's) and
   `recurse.jl:66` (the 20 strings' frequencies in 1,000 runs). Each over 20
   seeds: mean and standard deviation, compared with gen.clj's statistics,
   never sample by sample.
4. **The upstream suite once** (M5's fourth checklist item): done, every
   assertion passes, `enumerative.jl` included (§4.3).

**The fixtures** (genclj-il3d, 2026-10-21). `test/fixtures/genjl/generate.jl`
writes items 1–3 as EDN in gen.clj's terms (keyword addresses, Switch's
branch index 0-based), beside `upstream-run.edn` (item 4), with the EDN
writer it shares with `run_suite.jl` (`edn.jl`). It ran in 81 s on this
Mac's Julia 1.12.6 and Gen 0.4.8, offline, single-threaded, from a scratch
depot (`JULIA_DEPOT_PATH=<scratch>:~/.julia`, so nothing is written to
`~/.julia`):

    env JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=1 \
      nice -n 15 julia --project=test/fixtures/genjl test/fixtures/genjl/generate.jl

- `switch-regenerate.edn`: six cases, 10 regenerations each, of a simulated
  trace: switch.jl:85, :89 and :94 (the bare Switch), :288 and :292 (one
  case), :294, and the move `mh(tr, select(:x => :z))` makes (NoChange
  argdiffs from the caller), each with the old and new arguments, choices
  and scores, the argdiffs and selection, and Gen.jl's weight.
- `beta-endpoints.edn`: Gen.jl's beta log density at x ∈ {0, eps, 1 − eps,
  1} for the five (α, β) above.
- `statistics.edn`: each statistical testset's quantities over seeds 1 to
  20 (seed 1 is Gen.jl's own run), values, mean and sd.

Tests read them through `gen.genjl.oracle` (`switch-regenerations`, with
choice maps, gen.diff argdiffs and a selection; `beta-endpoints`;
`(statistic testset & path)`), and `gen.genjl.oracle-test` checks the
translation: gen.clj's assess of every recorded choice map gives Gen.jl's
score, and beta's densities agree except at D6's four points.

Gen.jl's statistics, against gen.clj's measurements for the statistical
ports (genclj-zgkq; gen.clj's keys differ, so samples are never compared):

| Testset | Quantity | Gen.jl, 20 seeds (mean, sd) | gen.clj (mean, sd, keys) |
|---|---|---|---|
| `train.jl:1` | θ₁ | 0.0041, 0.0745 | 0.0105, 0.0813, 12 (derived sd 0.073) |
| | θ₂, θ₃, θ₄, θ₅ | −5.505, 5.502, 5.507, −5.504; sd 0.005–0.006 | −5.503, 5.505, 5.505, −5.501; sd 0.004–0.008, 12 |
| `train.jl:126` | θ | −2.0033, 0.0106 | −2.0040, 0.0096, 6 |
| | log σ | −1.2036, 0.0065 | −1.2014, 0.0057, 6 |
| | score (optimum −0.2149) | −0.2139, 0.0082 | −0.2129 in one run (SE 0.007) |
| | line 176's score (of q, U4) | −0.2138, 0.0069 | n/a: the port scores what lecture-batched trains |
| `variational.jl:1` | the four parameters | within 2.4e-9 of the targets (μ_intercept; the others 4.5e-11) | within 2e-9, 8 |
| `particle_filter.jl:96` | log-ML (exact −4.8765) | −4.8782, 0.0073 | −4.8752, 0.0051, 24 |
| `particle_filter.jl:146` | log-ML | −4.8737, 0.0106 | −4.8754, 0.0095, 24 |
| `recurse.jl:66` | the 20 strings' counts in 1,000 runs | every mean within 2.5 standard errors of 1,000·p; no count below 2 | the port's 10,000 runs, within 5 binomial standard errors |

Gen.jl's two PCFG loops (`generate` with no constraints, then `simulate`)
each start from `Random.seed!(1)` and draw the same strings.

## 6. Proposed port order and beans

For the owner's approval. Ports first, one bean per group of files, each
writing `test/gen/genjl/<file>_test.clj` with every `deftest` citing its
Gen.jl file and line. An assertion that waits on a gap or a decision is
listed in its namespace as pending, citing the bean. Each port bean plants
one bug per file (a sign, an index, an off-by-one in a diff), to check the
ported tests can fail, as the suites do. Sizes: small is under a day of
agent time, medium a day or two.

| # | Bean | Contents | Assertions | Size | Depends on |
|---|---|---|---:|---|---|
| 1 | Port: the data layer | `diff.jl`, `selection.jl`, `assignment.jl`, `autodiff.jl` (through `broadcast-to`) | 156 port | small | — |
| 2 | Port: the GFI and the dynamic language | `gen_fn_interface.jl`, `dsl/dynamic_dsl.jl`, `dsl/macros.jl` (dynamic), `tilde_sugar.jl` | 89 port | small | — |
| 3 | Port: the static language | `dsl/static_dsl.jl`, `static_ir/static_ir.jl` (its StaticIRBuilder functions written as the bodies in its comments), `dsl/macros.jl` (static) | 181 port | medium | — |
| 4 | Port: call_at, choice_at, Map | `call_at.jl`, `choice_at.jl`, `map.jl` | 274 port | medium | — |
| 5 | Port: Unfold, Switch, Recurse | `unfold.jl`, `switch.jl`, `recurse.jl` | 271 port | medium | — |
| 6 | Port: distributions | `distributions.jl` (dirichlet's reference values added to `generate.py`), `mixture.jl`, `product.jl`, `dist_dsl.jl` | 200 port | medium | — |
| 7 | Port: custom functions and inference | `custom_determ.jl`, `importance_sampling.jl`, `mh.jl`, `trace_translators.jl`, `kernel_dsl.jl`, `enumerative.jl`, `particle_filter.jl`, `train.jl`'s gradient check, the four smoke tests | 91 port | small | — |
| 8 | Port: the statistical testsets | `train.jl` (SGD, lecture), `variational.jl` (BBVI), `particle_filter.jl` (two log-MLs), `recurse.jl` (the pcfg's strings); `^:slow`, bands from the estimators' variance | 33 statistical | medium | 5, 7 |
| 9 | Decisions (owner) | D1 (document precise retdiffs), D4 (loud accessors or a documented idiom), D6 (beta's endpoints), G3 (default arguments or a difference), G15 (IR shape n/a or extended), D2 (categorical and Switch in §9) | (up to 129 gap) | — | — |
| 10 | Small API gaps | G1 (a call's return value by address, a public subtrace), G9 (paths in selections, `set-subselection`, loud vector addresses: F3), G10 (variadic merge), G13 (Recurse's numbering) | 83 gap | small | — |
| 11 | Array functions | G8 (atan2 and inverse trigonometric and hyperbolic functions: VJPs, MLX, conformance), G14 (`full` of a tape value: F4) | 4 gap | small | — |
| 12 | Static analyzer | G11 (macroexpand a binding's head), G12 (quoted data, F5) | 2 gap | small | — |
| 13 | Static rewrites | G2 (computed addresses into `call-at`/`choice-at` nodes, nested traces hoisted into bindings), the static law over the rewritten bodies | 49 gap | medium | 12 |
| 14 | VI with the model's parameters | G4 (`model_update`), the minimal VAE | 3 gap | medium | 8 |
| 15 | Mixtures of multivariate families | Done (b9eb9d2): G6, F2; its 10 assertions are in 6 | — | — | — |
| 16 | `dist` with non-numeric labels | G7 (parameters that are not arrays) | 17 gap | medium | — |
| 17 | Custom functions' parameters | G5 | 1 gap | small | — |
| 18 | mvnormal's check | Done (2beab42): F1, D15 | — | — | — |
| 19 | The oracle | `test/fixtures/genjl/generate.jl` and its three fixtures (switch-regenerate, beta-endpoints, statistics), run locally as the upstream run was; the upstream record is done (§4.3) | — | small–medium | — |
| 20 | Switch's regenerate upstream | U1, confirmed: the 6 assertions against the fixture, a §9 note; report upstream (owner's call) | 6 oracle | small | 19 |

Beans 1–7 can run in parallel; with 9–17 done, every assertion is a port,
an oracle comparison, a statistical test or an n/a with its reason.

## 7. REPL checks

Short runs on the pure-JVM backend (gen.clj 12e788d), each against what the
Gen.jl test expects.

| Check | Gen.jl expects | gen.clj gives |
|---|---|---|
| grad of `sum(full([3], x) .* g)` | sum(g) | ClassCastException (F4); `broadcast-to`: 6.0 |
| `has-submap?` at a value; `get-submap` at and through a value; `get-choice` missing | false; KeyError ×2; KeyError | true; `#gen/value 1`; `{}`; nil (D4) |
| `(cm/merge c)` | c | ArityException (G10) |
| `to-array` of `{:z 9.0 :a {:b 1.0}}` | leaves first: [9.0 1.0] | [1.0 9.0] (sorted addresses; the suite's cases agree) |
| loop body's update retdiff | UnknownChange | no-change (D1) |
| Map 2 → 3, the first element unchanged; a custom argdiff | both retained elements in the VectorDiff; a VectorDiff naming the first | only the changed element; no-change (D1) |
| `static_ir.jl` foo/bar, the five update cases | counters 0 0 1 1 1, retdiffs N U U U U | counters 0 0 0 (1 with a fresh 2.0) 1 0, retdiffs N U N U U (D1, D10) |
| tracked-diffs foo/bar, UnknownChange | counter 1, NoChange | counter 0, no-change (D1, D10) |
| cached Julia nodes, five runs | 1 0 0 1 1 | 1 0 0 1 1 |
| `(tr :x)` at a call | 1 | nil (G1); the subtrace's retval is 1 |
| user macro to `(trace …)`, dynamic; static | works | works; "Unable to resolve symbol: trace" (G11) |
| quoted `(trace …)` returned, dynamic; static | works | works; compile error (G12) |
| trace in a tuple; `[:x i]` address, static | works | compile errors (G2); `choice-at` by hand works |
| static `(let [] nil)`, `(param :theta)`, destructuring with an `if` binding, rebinding | — | nil; 2.5; Gen.jl's weight; 1 and 3 |
| normal std 0; dirichlet α = −1 | the mean; −Inf | errors (D5) |
| beta(0.5, 0.5) at 0, beta(1.5, 0.5) at 1 | finite | +Inf (D6) |
| normal at 1e13: log-prob and gradients | −5e25; (−1e13, 1e13, 1e26) | the same, exactly |
| shapes [1 3] and [2 1 1] broadcast | (Julia) (2, 3, 1) | [2 1 3], Julia's reversed; shape errors through `assess` |
| keyword labels in `dist`; numeric labels; 1 − uniform | work | error (G7); work; 0.0 |
| homogeneous mixture of mvnormals | −4.754 | −4.135 from `dc/log-prob` (F2), a shape error at a trace site; heterogeneous: −4.754. Since b9eb9d2, with dims [1 2]: −4.754 |
| `mixture.jl:138` at b9eb9d2: is_discrete; the weights' gradient; the means' and covariances' gradient shapes; the rotated covariance's gradient at θ = π/2 | false; exp(logpdfₖ − logpdf); (2, 2), (2, 2, 2); the central difference | false; [2.29050 0.13967], the same; [2 2], [2 2 2]; 0.120832588 against 0.120832588 |
| Switch regenerate, bare, same branch, select z | old = new − w | w = 0 (U1) |
| Switch, nested, branch 2 → 1 with select() | old = new − w | old −4.73, new − w −2.28: the dropped call cancels (U1) |
| bare Switch update with `:x => :z` constrained | ignored | unvisited error (D11) |
| selection paths by reducing `get-subselection` | Gen.jl's `in` | the same; `selected?` on a path: false (G9, F3) |
| `(+ true true)` from two coins | 2 | 2 (D9) |
| call_at gradients with a retgrad | arg 2.068, y 0.4 | the same |
| pcfg: generate with every choice; the structure update | weights in the test | the same weights, retval, discard and retdiff |
| propose a static function of a trace | — | works |
| atan2, asin, acos, atan, sinh, cosh in gen.array | — | none (G8) |
| importance sampling (internal, custom proposal) | n, normalized, lml not NaN | the same |
| involutive MH with a transform and `:check?` | accepted, x false, z 0.5 | the same |
| extending translator: weight = update − proposal | `==` | `==` |
| custom deterministic: update with a VectorDiff; CustomPlus gradients | 12; (2, 2) | 12; [2.0 2.0] |
| parameter gradient through an addressless splice; one SGD and one GD step | 2.0; 0.001 | 2.0; 0.001 |
| Unfold: d x₁₀ / d x₀ for x → 2x | 1024 | 1024.0 |
| Unfold with a static kernel: seven update and four regenerate retdiffs | as the test | as the test |
| a rest parameter for optional arguments | — | calls and updates with 1 or 3 arguments work |
| `gen` with two arities | — | compile error |
| `mh` with a selection on a static model (Gen issue 168) | runs | runs |
| mvnormal with 2·ones(2, 2) | (Gen.jl, run: 15.240802, U3) | 15.24 (F1); since 2beab42, an error (D15) |
| the 50-point squared-exponential kernel with 1e-6 on its diagonal, in float32, at 7e85fb3 | (float64 only: accepted) | an error suggesting a jitter of 2e-4; with it, accepted; with 8e-5, an error (D15). At the bound 1/ε: still an error, suggesting 5e-6 |
