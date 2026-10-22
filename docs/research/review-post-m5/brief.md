# The post-M5 review: brief

gen.clj (a Clojure/JVM port of GenMLX, aiming at Gen.jl's capabilities) closed
milestone M5 on 2026-10-22: Gen.jl 0.4.8's test suite is ported and passes on
every backend. The owner now wants every line of `src/` reviewed by fresh
eyes, in two independent reviews:

- **Idiom:** is this idiomatic, beautiful Clojure, with steady JVM
  performance? The owner's bar: "a truly great working reboot of GenMLX, no
  ugly code, super-stable JVM performance, 100% beautiful idiomatic Clojure".
- **PPL soundness:** is it solid as a probabilistic programming system
  (weights, scores, diffs, randomness, inference), by the GFI laws and the
  math, not just by matching Gen.jl?

You are one reviewer, with one layer of one review. The other reviewers
handle the other layers, and a separate adversarial pass verifies every
finding before anyone acts on it. So precision matters more than volume: 8
real findings beat 40 plausible ones, and a refuted finding is noise.

## Where things are

- **The code to review:** the snapshot worktree
  `/private/tmp/claude-501/-Users-robert-code-gen-clj/826ae76f-f183-43cd-bbe4-ed91ce6aaf6f/scratchpad/wt-review`
  (detached at main 34e9a44). Read from there, so line numbers are stable.
  Cite paths relative to it (`src/gen/dynamic.clj:412`).
- **Your output directory:**
  `/private/tmp/claude-501/-Users-robert-code-gen-clj/826ae76f-f183-43cd-bbe4-ed91ce6aaf6f/scratchpad/review-post-m5`
  (call it OUT). Your report goes to `OUT/<your report name>.md`. Any
  scripts you write go in `OUT/scripts/<finding id>.clj`.
- **Read first:**
  1. `CLAUDE.md`, the project's rules and house style;
  2. `DESIGN.md`, especially §3 (principles), §4 (architecture) and §9
     (the decisions, about 1,200 lines: skim it, then return to the entries
     that touch your layer);
  3. the docs CLAUDE.md and DESIGN point to for your layer (`docs/array-api.md`,
     `docs/porting-genmlx.md`, `docs/research/*.md`);
  4. the earlier reviews `docs/research/review-2026-10-18.md`,
     `review-2026-10-19.md` and `review-m2-2026-10-19.md`, so you don't
     repeat what they found and the code has since fixed.
- **Tests** (`test/`) are context: they tell you what is checked. They are
  out of scope for findings, except that a path with no law or test is
  worth saying.
- **Known open work** lives in beans (an issue tracker). You may search it,
  read-only, from the main checkout (the snapshot has no beans):
  `cd /Users/robert/code/gen.clj && beans list -S "<term>"` and `beans show
  <id>`. If a finding is already an open bean, give the bean's id rather
  than presenting it as new.

## Rules

- **Read-only.** Don't modify, create or delete anything in the repository,
  the worktrees or the beans. No git operations beyond reading (`log`,
  `show`, `blame`, `grep`). Write only under OUT.
- **Running code: the pure-JVM backend only.** This is a 32 GB Mac where
  MLX and llama.cpp JVMs are rationed. So never pass `-Dgen.backend=mlx` or
  any `-Dgen.llama.*` property, and never run the whole test suite. You may
  run, from the snapshot worktree, one JVM at a time:
  - a script: `nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/<id>.clj`
  - one test namespace: `nice -n 10 clojure -J-Xmx1g -X:test :nses '[gen.foo-test]'`

  A smoke script that works is `OUT/smoke.clj`. Note the API's shapes: `trace`
  is a form inside `gen` bodies; constraints are choice maps
  (`cm/from-map`); `gfi/generate` returns `{:trace :weight}`; a trace is a
  record with `:choices`, `:retval` and `:score`.
- **Settled decisions.** DESIGN §9 records the owner's decisions. Don't
  report a decided behavior as a defect. If you think a decision is wrong,
  put it under "Decisions worth revisiting", with the argument and the §9
  entry it concerns.
- **Every line.** Read every line of every file in your layer. The report's
  coverage section lists each file with the line ranges you read.
- **No fixes.** Describe the direction a fix would take; don't write a patch.

## The report

Write it to `OUT/<your report name>.md` in this shape:

```
# Post-M5 review (<idiom | PPL soundness>): <layer>

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary
Three to six sentences: the layer's state and the findings that matter most.

## Findings
Ranked, most severe first. Each:

### <ID>: <one-line claim>
- **Severity:** (see your review's scale)
- **Where:** path:line, and others if it recurs
- **What:** the claim, in a short paragraph.
- **Evidence:** a short excerpt, or the reasoning. For a correctness claim,
  a concrete scenario: inputs → expected (with the law or the derivation
  that says so) → actual, as observed by running
  `OUT/scripts/<ID>.clj` on the JVM backend (or "not run", and why).
- **Direction:** what a fix would look like.
- **Known?** a bean id, an earlier review's finding, or "new".

## Decisions worth revisiting
(optional) Each with its §9 entry and the argument.

## Checked and sound
The main paths you verified and found right, so the coverage is visible:
what you checked, and how (read, derived, ran).

## Coverage
Every file in your layer, with the line ranges read.
```

Finding IDs are your layer's code and a number: `I2-01`, `P3-07`, …

When the report is written, reply with its path, the number of findings at
each severity, and the two or three that matter most, in a few lines.
