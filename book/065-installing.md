# Installing and Running

## What gen.clj needs

A JDK and the Clojure CLI. gen.clj is developed on Oracle GraalVM 25; its
core depends on Clojure alone, and its default backend is pure Clojure, so
every chapter of this book runs with nothing else installed. The MLX
backend and the llama.cpp engine are optional, and each needs native
libraries built once (below).

gen.clj is used as a git dependency. In a project's `deps.edn`:

```edn
{:deps {gen.clj/gen.clj {:git/url "https://github.com/robert-johansson/gen.clj"
                         :git/sha "…"}}}
```

As a git dependency, `clojure -X:deps prep` compiles the Java binding the
native backends load; the pure-Clojure backend does not need it.

## The REPL and the tests

In gen.clj's own checkout:

```text
clojure -M:test:dev            # a REPL with the test loop: (t) after each change
clojure -X:test:fast:par       # the tests without the slow statistical ones, in 5 JVMs
clojure -X:test:par            # every test, in 5 JVMs
clojure -X:test                # every test, in one JVM
```

At the REPL, `(t)` reloads what changed and runs the tests it can break,
usually in seconds; `(t 'ns)` runs one namespace's tests.

## The book

The chapters are Markdown in `book/`, their code namespaces in
`book/src/book/`, on the `:test` classpath:

```text
clojure -X:test :patterns '["gen.book.*"]'  # every chapter, its results checked
clojure -M:test -m gen.book sync           # write the chapters' code and figures in
clojure -M:test -m gen.book check book/011-getting-started.md   # one chapter
clojure -M:test -m gen.book pdf            # this book, as target/book/gen-clj-book.pdf
```

A chapter loads at the REPL like any namespace:
`(require 'book.getting-started :reload)`. The PDF needs Pandoc, a TeX
distribution with xelatex, and librsvg's `rsvg-convert` for the figures
(on macOS, `brew install pandoc librsvg` and MacTeX).

## The MLX backend

MLX runs gen.clj on Apple's GPUs (Metal), on NVIDIA GPUs (CUDA, on
Linux), and on its own CPU backend. Build it once:

```text
native/build-mlx.sh            # libmlxc against MLX's official wheel, into native/build/
clojure -T:build java          # the Java binding, into target/classes
```

Then any program, test or chapter runs on it with system properties:

```text
clojure -J-Dgen.backend=mlx -X:test:light                          # Metal
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.device=cpu -X:test:light   # MLX's CPU backend
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.dtypes=64 …                # float64 (CPU only)
```

MLX computes in float32 by default, and its arrays are lazy: an inference
loop evaluates them once per iteration. A model that runs one particle at
a time pays MLX's cost per operation many times, so on MLX batch the
particles where the model allows (*Speeding Up*). `-X:test:light` skips the
statistical tests that take minutes there, and the book's chapters run on
MLX only when `-Dgen.book.mlx=true` asks.

## Local language models

The language-model chapters run on small engines written in Clojure,
which the tests use. Real models run through llama.cpp, built once:

```text
native/build-llama.sh          # llama.cpp's libraries (Metal on macOS; cpu or cuda on Linux)
clojure -T:build java
```

A model is a GGUF file, given to `gen.llm.llama/engine`; the JVM finds the
libraries with `-Dgen.llama.lib=<dir>`.
