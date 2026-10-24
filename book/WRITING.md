# Writing a chapter

How the book's chapters are written, so that they read as one book. The
mechanics (regions, includes, checks, figures) are in `README.md`. Chapter
011, `011-getting-started.md`, is the model to follow.

## The book

One book in five Parts (DESIGN.md §9): I Foundations (01x), II Inference
(02x), III Theory (03x), IV The System (04x), V Applications (05x), and
appendices (06x). Chapters are numbered in order within their Part. Gen.jl's
tutorials are worked chapters inside the Parts, and the GenMLX book's
chapters are ported into them.

Part I, Foundations:

| File | Chapter | Sources |
|------|---------|---------|
| 011-getting-started.md | Getting Started: Linear Regression | Gen.jl's Getting Started; GenMLX book 011 |
| 012-modeling.md | Introduction to Modeling | Gen.jl's Introduction to Modeling |
| 013-conditioning.md | Conditioning | Gen.jl's A Bottom-up Introduction; GenMLX book 012 |
| 014-handler-loop.md | The Handler Loop | GenMLX book 013 |
| 015-choice-maps-traces.md | Choice Maps, Traces and Selections | GenMLX book 014 |
| 016-update-regenerate.md | Update and Regenerate | GenMLX book 015; Gen.jl's Reasoning about Regenerate |
| 017-composition.md | Composition | GenMLX book 016 |

## Prose

- **Plain and exact.** Short sentences, active voice, American spelling.
  Say what a thing does and why it matters; no hype ("powerful",
  "seamless", "simply", "just").
- **Define a term where it first appears**, in italics: a *trace*, a
  *choice map*. Afterwards, use that term and no synonym.
- **Every claim about behavior is shown and checked.** If the text says
  the score is a sum, a region computes the sum and checks it. A number in
  the prose comes from a checked result in the code.
- **Say where gen.clj differs from Gen.jl or GenMLX** when a reader of
  either would expect otherwise: explicit keys, 0-based indices, loud
  errors, batched execution, the laws. Collect the details for the appendix
  *Coming from Gen.jl*, not in every chapter.
- **Refer to other chapters by title** ("as *Conditioning* showed"), not
  by number.
- **Open with what the chapter does and why**, and name its sources in one
  sentence ("This chapter ports Gen.jl's *Introduction to Modeling* …").
  Close the main text where the next chapter picks up.

## Porting the sources

- **The GenMLX book:** adapt its prose where gen.clj's semantics carry
  over, and rewrite where they differ. Never port one of its claims
  without checking it in gen.clj: run it.
- **Gen.jl's tutorials:** port the examples and the narrative, in
  idiomatic Clojure. Julia's 1-based indices become 0-based, `~` becomes
  `trace`, `{*}` becomes `splice`, and the global random number generator
  becomes explicit keys. Plots become figures.
- Where a source's example no longer makes its point in gen.clj (a
  different API, a different behavior), find one that does, and say why
  if a reader of the source would wonder.

## Code

- **One namespace per chapter**, `book.<name>` in `src/book/<name>.clj`,
  written and run at the REPL. Its first region is the ns form, with the
  house style: `(:refer-clojure :exclude [+ - * /])` and gen.array's
  operators referred. Aliases as chapter 011's: `array`, `cm`, `d`, `dc`,
  `gfi`, `rng`, `sel`, `mh`, `diag`, `svg`.
- **Idiomatic, small and documented:** functions with docstrings, `let`
  over repetition, no clever code. A model's arithmetic is gen.array's,
  and a model never reads a value on the host.
- **Explicit keys everywhere.** Each use gets its own `(rng/key n)` or a
  split; no implicit keys, so every result is reproducible.
- **Lines of at most 80 characters** (the test enforces it).
- **Regions:** kebab-case names, one idea each, mostly in the order the
  text includes them. Helpers the text does not show can be regions it
  never includes.

## Checks

- `;=> value` for a deterministic result. A float checks to the digits it
  shows, so show the digits that matter.
- `(array/allclose? a b) ;=> true` where two values must agree up to
  rounding; `;=> 0.0` would accept anything within 0.05.
- `;=> value ; float64` for a sampled value: a key draws other values in
  float32 (MLX). Results of chains and importance samplers, which can
  differ across precisions, are either marked so or stated as checks.
- **A statistical claim is a check with a band from its standard error**
  (five standard errors, or an exact value from theory), written in the
  code and shown as `;=> true`. Choose thresholds that hold on every
  backend, not ones that happen to pass.
- `;; error: message` as a region's first line shows a loud error, where
  the chapter wants the reader to see one. Write each of its forms behind
  `#_` in the source, so that requiring the namespace does not throw; the
  test runs them, and the Markdown shows them, without it.
- **A check must be able to fail.** Before trusting a statistical check,
  plant a bug (a wrong prior, a skipped move) and see it fail; note what
  you planted in the chapter's bean.

## Figures

- A region that begins `;; figure: figures/0NN-name.svg` and returns
  `svg/lines` or `svg/series` (`test/gen/book/svg.clj`). The Markdown's
  image text is its caption, so leave the SVG untitled.
- One idea per figure, close to the text that discusses it.

## The workshop

Each chapter ends with a *Workshop* of three to five exercises, then
*Solutions*, each solution a region with its checks. Exercises ask the
reader to predict, then run: change a prior, break a model, compare two
algorithms.

## Running

```
clojure -M:test -m gen.book sync book/0NN-name.md    # write its includes and figures
clojure -M:test -m gen.book check book/0NN-name.md   # run its checks: a summary, exit 1 on failure
clojure -M:test -m gen.book pdf                      # the whole book as a PDF
```

At the REPL: `(require 'book.<name> :reload)`.
