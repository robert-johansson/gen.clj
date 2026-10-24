# The gen.clj book

Probabilistic programming with gen.clj: GenMLX's book ported, with Gen.jl's
tutorials as its worked chapters (bean genclj-ceds). The text is Markdown,
and its code is verified for real (DESIGN.md §9).

## How a chapter is made

- **The code is a namespace.** Chapter `011-getting-started.md` keeps its
  code in `src/book/getting_started.clj`, written and run at the REPL
  like any Clojure code. A line `;; book: name` starts a region, which
  runs to the next such line.
- **The Markdown includes regions.** A line `<!-- code: name -->` is
  followed by a ```clojure block holding that region's text. You write
  only the marker; `clojure -M:test -m gen.book sync` writes the blocks
  (and the figures). Every ```clojure block in a chapter is an include,
  so no code in the book goes untested. Regions the text does not include
  (setup, helpers) still run.
- **The test** (`gen.book-test`, on every backend) fails when an include
  differs from its region. It then runs the namespace region by region and
  checks the results the code states:
  - `;=> value` after a form shows its result. A float checks to the digits
    it shows, so one text holds on float64 and float32 backends. Few
    digits make a weak check: `;=> 0.0` accepts anything within 0.05. Where
    two values must agree up to rounding (a score and its sum by hand),
    state it so: `(array/allclose? a b)` and `;=> true`.
  - `;=> value ; float64` is checked only where the default float is
    float64. Sampled values carry it, since a key draws other values in
    float32.
  - A statistical claim is a check in the code, with a band from the
    estimator's standard error, shown as `;=> true`.
  - `;; error: message` as a region's first line shows a loud error. Its
    forms are written behind `#_` in the source, so the namespace still
    loads; the test runs them, and the Markdown shows them, without it.
  - `;; figure: figures/…` as a region's first line makes its last form an
    SVG figure (from `test/gen/book/svg.clj`). The test fails when the
    committed file is stale. The Markdown's image text is its caption.
  - A region's lines are at most 80 characters, so they fit a printed
    page.

The mechanics are in `test/gen/book.clj`.

## Running

```
clojure -X:test :nses '[gen.book-test]'                       # every chapter
clojure -J-Dgen.backend=mlx -X:test :nses '[gen.book-test]'   # on MLX
clojure -M:test -m gen.book sync                              # write the includes and figures
clojure -M:test -m gen.book pdf                               # the PDF: target/book/gen-clj-book.pdf
```

The PDF is Pandoc's: every chapter in order, with `metadata.yaml` (title,
contents, page layout). It needs pandoc, a TeX distribution with xelatex,
and librsvg's `rsvg-convert` for the SVG figures (on macOS: `brew install
pandoc librsvg` and MacTeX). It refuses to build from stale includes.

At the REPL (`clojure -M:test:dev`), `(require 'book.getting-started
:reload)` loads a chapter's namespace, and `(gen.book/run-chapter
"book/011-getting-started.md")` runs its checks. `book/src` is on the
`:test` classpath, beside the figure helper.

## Chapters

One book in five Parts: 01x Part I (Foundations), 02x Part II
(Inference), 03x Part III (Theory), 04x Part IV (The System), 05x Part V
(Applications), 06x appendices, numbered in order within each Part.
`WRITING.md` has the plan of Part I and how a chapter is written.

- `011-getting-started.md`: Getting Started: Linear Regression (Gen.jl's
  Getting Started with the GenMLX book's chapter 011).
