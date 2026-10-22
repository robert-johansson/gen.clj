# The idiom review

Read `brief.md` first; this adds what is particular to the idiom review.

## The question

Is this code idiomatic, beautiful Clojure that a seasoned Clojurist would
be proud of, with steady JVM performance? Most of it was written by one
agent in long sessions, and no review since M1 has looked at idiom.

## What to look for

- **CLAUDE.md's house style and rules:** gen.array's operators referred in
  place of clojure.core's; aliases by the namespace's last segment; core code
  calling only gen.array, never a backend; no `eval!` in models, handlers or
  distributions; no hidden mutable state; loud `ex-info` errors with data.
- **The JVM performance notes:** reflection and boxed-math warnings, kernel
  size, loops used as values (they box), primitive hints, lazy sequences or
  allocation on a hot path (the handler, a backend's op, a distribution's
  logpdf).
- **Naming:** names that say what a thing is; consistent vocabulary across
  namespaces; Gen.jl's names where the API follows Gen.jl.
- **Shape:** function size; a namespace that has grown past one idea and
  where it would split; data-oriented design (plain maps and functions)
  versus needless protocols, records or macros; macros only where a
  function can't do it.
- **Duplication and dead code:** the same logic in two places; private
  functions nobody calls; options nobody passes.
- **Defensive layers:** checks, guards and fallbacks that could be fewer or
  simpler without losing a loud error (in gen.batched and gen.dynamic.core
  especially: guarded clojure.core, spot checks re-run on the JVM,
  mixed-backend detection, the verified-updates schedule).
- **Docstrings and comments:** match the codebase's density; say what and
  why, not how; no stale comments that describe code that has changed.
- **Java (where your layer has it):** idiomatic, safe resource handling.

If you notice a correctness problem in passing, report it too, marked
**(correctness)**. The PPL review owns those, but don't drop one.

## Severity

Ranked by how much the change would improve the code:

- **major:** a structural problem (a namespace to split, a design that fights
  Clojure, duplication of real logic, a hot path that allocates or boxes),
  or a rule of CLAUDE.md broken.
- **moderate:** a function or a few that read badly and would be clearly
  better rewritten; a misleading name; a stale or wrong docstring.
- **minor:** small idiom points (a `(if x x y)` for `(or x y)`, a threading
  macro that would read better). Group minor points of one kind into one
  finding with several locations.

## Layers and report names

- **I1, arrays, backends and randomness** → `idiom-arrays.md`.
  `src/gen/array.clj`, `src/gen/array/*.clj`, `src/gen/backend/jvm.clj`,
  `src/gen/backend/mlx.clj`, `src/gen/backend/mlx/exact.clj`,
  `src/gen/random.clj`, `src/gen/random/threefry.clj`, and the hand-written
  Java `src-java/gen/mlx/MlxRuntime.java` and `MlxException.java`
  (`Mlx.java` and `MlxRaw.java` are generated: out of scope). About 5,700
  lines of Clojure. Ask in particular how `gen.backend.mlx` (1,651 lines)
  and `gen.array` (1,133) should split, if at all.
- **I2, the modeling core** → `idiom-core.md`. `src/gen/choicemap.clj`,
  `selection.clj`, `trace.clj`, `diff.clj`, `gfi.clj`, `dynamic.clj`,
  `dynamic/core.clj`, `dynamic/static.clj`, `batched.clj`, `custom.clj`,
  `combinators.clj`. About 5,100 lines. `gen.dynamic` is 1,296 lines: does
  it split?
- **I3, distributions, inference and learning** → `idiom-inference.md`.
  `src/gen/dist.clj`, `src/gen/dist/*.clj`,
  `ext/commons-math/src/gen/dist/commons_math.clj`,
  `src/gen/inference/*.clj`, `src/gen/gradients.clj`, `src/gen/learning.clj`.
  About 4,700 lines.
- **I4, the LLM layer** → `idiom-llm.md`. `src/gen/llm/**/*.clj` and
  `src-java/gen/llama/LlamaRuntime.java`. About 5,200 lines of Clojure.
