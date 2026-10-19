# M1 acceptance tests (2026-10-19)

After three code reviews (`review-2026-10-18.md`, `review-2026-10-19.md`),
M1 went through four black-box acceptance passes. Each was an independent
agent acting as a Gen.jl user, working from the README and docstrings only.
Each checked answers against hand-derived exact values: conjugate models,
the Kalman filter and enumerated posteriors. Each pass's findings were fixed
before the next began; the scripts are in the session scratchpad
(`accept/`, `accept2/` … `accept4/`).

| Pass | Commit fixed in | Bean | Silent wrong answers | Other bugs |
|---|---|---|---|---|
| 1 | 325b0cb | genclj-gabj | 0 | 11 (validation, error messages, batched PF with proposals) |
| 2 | 1059cca | genclj-hup8 | 4 (batched branches in clojure.core and helpers, unchecked updates, plain-map constraints) | 8 |
| 3 | 58bd8d7 | genclj-5azs | 2 (comparison also read elsewhere, fallback after unchecked updates) | 3 macro regressions, 6 minor |
| 4 | f449582 | genclj-xwk0 | 3 (try/catch, representation tests, final unchecked update) | 4 |

## What held throughout

The scalar semantics, which is the handler and the ground truth, never
gave a wrong answer:
- **The GFI.** About 60 hand-derived weight and discard checks.
- **Combinators and distributions.** Map, Unfold and Switch updates were
  exact, and all 26 distributions matched.
- **Inference.**
  - Importance sampling with exact-posterior proposals matched to 1e-15.
  - Particle filters agreed with the Kalman filter.
  - MH, HMC, MALA, MAP, elliptical slice and involutive MH reached the
    exact posteriors.
- **Learning.** Gradients and optimizers converged to the MLE.

Every silent wrong answer was in batched execution.

## Why batching is the risk surface

Clojure's `if` tests an object, with no hook such as Python's `__bool__`
(which JAX's tracer uses). A batched value is one object standing for every
particle's value, so any test of it that the tracer does not see takes one
branch for all particles. The passes found such tests in:
- `if`, `case` and `catch` in bodies;
- clojure.core's own `if`s (`not`, `filter`, `get`, …);
- representation tests;
- helper functions;
- updates that were never checked.

The defences, in order of strength:
1. **The `gen` macro guards bodies.** It wraps `if` tests in `truthy?`,
   guards `case` and `catch`, and substitutes `gen.dynamic.core` for
   clojure.core's truthiness, lookup and type tests. `defn-guarded` does
   the same for helpers. These are deterministic.
2. **The tracer raises the fallback** on host reads, comparisons and hashes
   of batched values, and on comparisons nothing read. Batched-branch
   errors are logged, so catching one does not hide it. Deterministic.
3. **Spot checks** re-run particles through the scalar handler after
   simulate, generate and a doubling schedule of updates, and again when
   results are read. These are probabilistic.

## Documented limits

Each limit is stated in `docs/array-api.md` §8a–§8b and the `gen.batched`
docstring.
- **Plain helper functions,** outside `gen` and not `defn-guarded`, that
  branch on a choice in a way the tracer cannot see. Examples: a `case` on
  a categorical, a vector lookup keyed by one, a raw `if` on a bernoulli's
  value, or a raw `if` on a comparison that is also read elsewhere. Spot
  checks find these with high probability, but a rarely taken branch can
  escape 8 checks. `*spot-checks* :all` verifies every particle.
- **Java interop and dispatch on the class of a choice value**
  (multimethods, protocols extended to `Object`). These see the batch
  object.
- **Known deliberate differences** from plain Clojure inside bodies:
  non-canonical `Boolean` objects, function identity, and macros that read
  `&env`.
