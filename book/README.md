# The gen.clj book

Probabilistic programming with gen.clj: GenMLX's book ported, with Gen.jl's
tutorials as its worked chapters (bean genclj-ceds). It is written as
Markdown, and its code is verified for real (DESIGN.md §9):

- Every ```clojure block of a chapter runs, in order, in the book's test
  (`gen.book-test`), on every backend.
- `;=> value` after a form shows its result, and the test checks it. A
  float checks to the digits it shows, so one text holds on float64 and
  float32 backends.
- `;=> value ; float64` is checked only where the default float is
  float64. Sampled values carry it, since a key draws other values in
  float32.
- A statistical claim is a check in the code, with a band from the
  estimator's standard error, shown as `;=> true`.
- `;; error: message` as a block's first line shows a loud error. A line
  `prints` and a ```text block after a block show what it printed.
- Figures are SVG made by the chapter's own code, from a block that begins
  `;; figure: figures/…`. The test fails when a committed figure is stale.

The mechanics are in `test/gen/book.clj`, the figure helper in
`test/gen/book/svg.clj`.

## Running

```
clojure -X:test :nses '[gen.book-test]'                            # every chapter
clojure -J-Dgen.backend=mlx -X:test :nses '[gen.book-test]'        # on MLX
clojure -M:test -m gen.book figures                                # rewrite the figures
```

At the REPL (`clojure -M:test:dev`), `(gen.book/run-chapter
"book/011-getting-started.md")` runs one chapter, after which its
namespace holds its definitions. A chapter's code needs the `:test`
classpath, for the figure helper.

## Chapters

Numbered as the GenMLX book's: 00x front matter, 01x Part I
(Foundations), 02x Part II (Inference), 03x Part III (Theory), 04x Part IV
(The System), 05x Part V (Applications), 06x appendices. A chapter that
ports one of the GenMLX book's takes its number.

- `011-getting-started.md`: Getting Started: Linear Regression (Gen.jl's
  Getting Started with the GenMLX book's chapter 011).
