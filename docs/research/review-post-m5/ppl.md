# The PPL soundness review

Read `brief.md` first; this adds what is particular to the PPL soundness
review.

## The question

Is gen.clj solid as a probabilistic programming system? Not "does it do what
Gen.jl does" (M5 checked that), but: is every weight, score, diff and
estimate right by the generative function interface's laws and the math?
Gen.jl itself has bugs (M5 found five, U1 to U5 in
`docs/research/genjl-test-suite.md`), so its behavior is evidence, never
proof.

## Grounding

Tie every finding to one of:

- **a GFI law:** the laws are in `test/gen/laws.clj` (and
  `docs/research/genjl-parity.md` §3, `docs/research/law-audit.md`). For
  example: generate with every choice constrained weighs the score; update's
  weight is the score's change less the fresh choices' and plus the
  discarded ones' proposal density; regenerate's is the retained density
  ratio; updates are reversible with their discards.
- **a derivation:** the density or the estimator, written out.
- **a counterexample program:** a model and the call that gives a wrong
  number, run on the JVM backend (`OUT/scripts/<id>.clj`), with the right
  number derived.

## What to look for

- **Weights and scores in every GFI operation** (simulate, generate, update,
  regenerate, assess, propose, project), in the dynamic handler, the static
  language's runner and score function, and in every combinator's paths
  (Map, Unfold, Switch, Recurse, call-at, choice-at).
- **Retdiffs and argdiffs:** a diff claims only what holds (a "no change"
  that is wrong silently skips work and leaves a stale score); the static
  language's skip conditions.
- **Randomness:** every site consumes one key, keys split purely, no key is
  used twice, no key lives in metadata (rule 2).
- **Batched execution** (gen.array.batch, gen.batched): the same semantics
  as scalar execution (rule 1).
- **Distributions:** densities, supports (log density -Inf outside it),
  parameter validation, sampling against the density, gradients.
- **Inference:** importance sampling, MH (selection, custom proposal,
  involutive), the particle filter (resampling, the log marginal likelihood
  estimate), MCMC (HMC, MALA, MAP, elliptical slice), enumeration, internal
  proposals, trace translators and SMCP3, the trace-kernel DSL, BBVI and
  VIMCO, parameter gradients and learning: each against its algorithm's
  correctness argument (detailed balance, unbiasedness, the right
  importance weight).
- **The documented differences** from Gen.jl (D1 to D16, U1 to U5, in
  `docs/research/genjl-test-suite.md`) that touch your layer: each should be
  justified by the laws or the math, not by convenience.

## Severity

- **critical:** a silently wrong weight, score, density or estimate in a
  reachable case: inference would be wrong and nothing would say so.
- **high:** wrong in a reachable case but loud (an error where the program
  is valid), or a silent error confined to an unusual but legal use.
- **medium:** fragile: a law that holds only by coincidence, an invariant
  nothing checks, a path with no law or test, a diff that is right but
  only because of a caller's habit.
- **low:** an unclear contract, a misleading docstring about semantics.

## Layers and report names

- **P1, the GFI core and the modeling languages** → `ppl-core.md`.
  `src/gen/gfi.clj`, `dynamic.clj`, `dynamic/core.clj`, `dynamic/static.clj`,
  `batched.clj`, `custom.clj`, `choicemap.clj`, `selection.clj`, `trace.clj`,
  `diff.clj`, and the batching tracer `src/gen/array/batch.clj`. About 4,700
  lines.
- **P2, combinators, distributions and randomness** → `ppl-models.md`.
  `src/gen/combinators.clj`, `src/gen/dist.clj`, `src/gen/dist/*.clj`,
  `ext/commons-math/src/gen/dist/commons_math.clj`, `src/gen/random.clj`,
  `src/gen/random/threefry.clj`, `src/gen/array/special.clj`. Where a
  density's or a sampler's correctness rests on a backend op (logsumexp,
  log1p, lgamma, a sampling primitive), check that op in
  `src/gen/backend/jvm.clj` and `src/gen/backend/mlx.clj` too. About 2,800
  lines, plus those ops.
- **P3, inference, gradients and learning** → `ppl-inference.md`.
  `src/gen/inference/*.clj`, `src/gen/gradients.clj`, `src/gen/learning.clj`,
  and reverse-mode autodiff `src/gen/array/tape.clj`. About 3,600 lines.
- **P4, the LLM layer** → `ppl-llm.md`. `src/gen/llm/**/*.clj`. The
  generative functions and inference (token, potential, smc, constrained,
  grammar, dfa, regex, bytes) are the heart; the runtime (cache, cells,
  trie, lm, branching, slots, llama, llama/recurrent) matters where a wrong
  cache or ledger would give wrong log probabilities. Compare token SMC and
  potentials with genlm-control's semantics (`docs/research/llm-token-smc.md`,
  `llm-token-gf.md`, `llm-grammar.md`). About 5,200 lines. Note: LLM tests
  run on the pure-JVM toy engines (`test/gen/llm/toy_engine.clj`,
  `cell_engine.clj`); use those for any script, never llama.cpp.
