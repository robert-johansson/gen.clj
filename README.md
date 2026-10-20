# gen.clj

Probabilistic programming in Clojure on the JVM: a port of GenMLX
(ClojureScript on MLX), shaped by [Gen.jl](https://www.gen.dev) and Marco
Cusumano-Towner's 2020 thesis.

Models are generative functions written as Clojure code. Inference uses the
Generative Function Interface (GFI): `simulate`, `generate`, `update`,
`regenerate`, `assess`, `propose` and `project`, each a pure function of its
arguments and an explicit random key.

**Status: M1, M2 and most of M4 are done.** The core runs on the JVM,
CPU-only, with nothing but a JDK: a pure-Clojure array backend with
reverse-mode autodiff. An MLX backend (M2) runs the same code on Apple's GPU
(Metal), on MLX's CPU backend and on CUDA (verified on a Jetson AGX Thor):
it passes the backend conformance suite and the test battery on each. Gen.jl
parity (M4) covers the core and GenParticleFilters, GenSMCP3 and
GenTraceKernelDSL, and encapsulated randomness for pseudo-marginal
inference, except the static modeling language. Local LLMs as generative functions (M3) come next. See
`DESIGN.md`.

## A model and two kinds of inference

```clojure
(ns example
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :refer [+ *]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def line
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 0.5)))
      slope)))

(def xs [-2.0 -1.0 0.0 1.0 2.0])
(def ys (cm/choicemap [:y 0] -3.1 [:y 1] -1.0 [:y 2] 0.9 [:y 3] 3.2 [:y 4] 4.8))

;; importance sampling, the 10,000 particles run as one batched execution
(:log-ml-estimate (is/importance-sampling line [xs] ys 10000 {:batched? true :key (rng/key 1)}))

;; MCMC: MH on the intercept, HMC on the slope
(reduce (fn [tr k]
          (let [[k1 k2] (rng/split k)]
            (-> tr (mh/mh (sel/select :intercept) k1) :trace
                (mcmc/hmc (sel/select :slope) {:L 5 :eps 0.05} k2) :trace)))
        (:trace (gfi/generate line [xs] ys (rng/key 2)))
        (rng/split (rng/key 3) 500))
;; → a trace; (tr :slope) is its choice at :slope, (tr [:y 0]) at a path
```

## What is there

- **Arrays** (`gen.array`). One backend-neutral API. Clojure numbers are 0-d
  arrays, and the operators are meant to be referred.
  - Broadcasting, reductions, indexing and linear algebra.
  - Special functions that match scipy to about 1e-15.
  - Reverse-mode autodiff (`array/grad`, `array/value-and-grad`).
  - Splittable Threefry keys, bit-identical to JAX.
- **The modeling language** (`gen.dynamic`). `gen` with `trace`, addressed
  and shared-namespace `splice`, trainable `param`s, and conditionals on
  values. Plus handler middleware and a dispatch seam.
- **Distributions** (`gen.dist`). Gen.jl's built-ins with Gen.jl's names and
  parameterizations, plus mixtures and products.
- **Combinators** (`gen.combinators`). `map`, `unfold` and `switch`, with
  incremental updates driven by change hints.
- **Batching** (`gen.batched`). N particles in one execution, by a batch
  tracer with vmap semantics. It falls back to one particle at a time when a
  model branches on particle values. Inside `gen` bodies that is guaranteed:
  the macro guards `if`, `case`, `catch`, and clojure.core's truthiness,
  lookup and type tests. `gen.dynamic/defn-guarded` gives helper functions
  the same guards (Java interop on a choice value, and multimethod or
  protocol dispatch on its class, still see the batch). In a plain helper, a raw `if` on a comparison used for nothing
  else is caught (the comparison is never read). Other branches in plain
  helpers are found by spot checks against the scalar handler, with high
  probability rather than certainty (`gen.batched` docstring;
  `docs/array-api.md` §8a). Batched importance sampling is 5–90× faster on
  the CPU (100–10,000 particles). For one key, batched and scalar runs draw
  different samples, with the same distribution.
- **Inference** (`gen.inference.*`):
  - importance sampling and resampling;
  - MH (selection, custom proposals, involutive with AD Jacobians);
  - kernel combinators with reversals;
  - particle filtering;
  - HMC, MALA (both optionally compiled into one MLX graph per step), MAP
    and elliptical slice;
  - exact enumeration: Gen.jl's enumerative inference over grids, and
    models whose finite latents are drawn from their exact conditional
    (exact likelihoods with those latents summed out, also step by step in
    an Unfold whose state does not depend on them);
  - choice and parameter gradients (`gen.gradients`) and optimizers
    (`gen.learning`).

## Tests

```
clojure -X:test          # everything, ~90 s
clojure -X:test:fast     # without the slow statistical tests, ~25 s
```

The GFI's laws are test.check properties, audited against the thesis
(`docs/research/law-audit.md`). They run over hand-written models and over
random generated programs. Inference is checked against exact answers:
- enumerated posteriors;
- conjugate models;
- the Kalman filter;
- estimators unbiased in exp space.

Every VJP is checked against finite differences. `docs/research/genjl-parity.md`
§3 maps each law to its tests.

## Requirements

A JDK (developed on Oracle GraalVM 25) and the Clojure CLI. The core
depends on Clojure alone.

## The MLX backend (optional)

`native/build-mlx.sh` fetches the official MLX 0.32.2 wheel (no Metal
compiler needed), builds mlx-c v0.7.0 against it, and puts the libraries in
`native/build/`; `clojure -T:build java` compiles the Java binding. For CUDA
on a GPU the wheels do not cover (the Thor's sm_110),
`native/build-mlx.sh cuda13-source` builds MLX from source; point
`-Dgen.mlx.lib` at the result. Then any program or test runs on MLX with
system properties:

```
clojure -J-Dgen.backend=mlx -X:test:light                         # Metal (the GPU)
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.device=cpu -X:test:light  # MLX's CPU backend
clojure -J-Dgen.backend=mlx -J-Dgen.mlx.device=cpu -J-Dgen.mlx.dtypes=64 -X:test:light
```

MLX is float32 by default (float64 only on its CPU backend). Arrays are lazy:
inference loops evaluate once per iteration. `docs/array-api.md` §13
describes the backend and the few ways it differs from the JVM backend.
