# Token-level SMC and potentials

*2026-10-21, bean genclj-loyp (M3), built by an agent on branch m3-smc
over the token generative function (`docs/research/llm-token-gf.md`) and
the LM protocol (`docs/research/llm-lm-protocol.md`). The choices that are
the owner's are marked **Recommendation**; the owner being away, the lead
records them PROVISIONAL in DESIGN.md §9.*

## What exists

| Piece | Where |
|---|---|
| Potentials: weights of token sequences (prefix, complete, the next token's), after genlm-control's `Potential`; built from functions; products | `gen.llm.potential` (a83022f, shared with the grammar work) |
| The potential contract as a check any potential runs (`check-consistent`) | `test/gen/llm/potential_contract.clj` |
| Token SMC over a completion, twisted by a potential | `gen.llm.smc` |
| Tests on an enumerable toy vocabulary | `test/gen/llm/potential_test.clj`, `test/gen/llm/smc_test.clj` |
| The same over llama.cpp, a lipogram as the potential (skipped without `-J-Dgen.llama.test-model`) | `test/gen/llm/llama_smc_test.clj` |
| Costs, on the toy engine or a GGUF model | `bench/gen/smc_bench.clj` |

## Potentials

```clojure
(potential/prefix p context)      ; log weight as a prefix (-Inf: nothing it begins has weight)
(potential/complete p context)    ; log weight as a whole sequence
(potential/logw-next p context)   ; float[n-vocab + 1]: token t at t, the end last
(potential/potential n-vocab {:prefix f :complete g :logw-next h})   ; h optional (dense default)
(potential/product p q …)         ; the sum of log weights
```

A potential is consistent when `logw-next` holds prefix(ctx ++ [t]) −
prefix(ctx) at t and complete(ctx) − prefix(ctx) at the end, as
genlm-control's. Contexts are the generated tokens, never the prompt, and
never hold the end: an LM's end-of-generation token that ends a
completion stands for it. Weights are floats, as an LM's rows (Clojure's
`float` refuses infinities: `unchecked-float`).

## Token SMC

```clojure
(require '[gen.llm.smc :as smc] '[gen.llm.token :as token])

(def answer (token/completion m))
(def s (smc/run answer [prompt 64] 32 {:potential grammar :proposal :masked :key k}))
(smc/log-ml-estimate s)      ; log Ẑ: the probability that a completion satisfies the grammar
(smc/particles s)            ; token vectors
(smc/traces s)               ; the completion's traces, scores = the particles' log p
```

| Function | Meaning |
|---|---|
| `(initialize completion [prompt max-tokens] n opts)` | n particles, no tokens yet; opts `:potential`, `:proposal`, `:key` |
| `(step state)`, `(step state {:key k})` | every live particle one token further, one LM read |
| `(maybe-resample state opts)`, `(resample state opts)` | ESS-gated (`:ess-threshold`, n/2 by default), `:method` (`:systematic` by default) |
| `(run completion args n opts)` | initialize, then step and maybe-resample until every particle has finished |
| `(log-ml-estimate state)`, `(ess state)`, `(done? state)` | as gen.inference.particle-filter's |
| `(particles state)`, `(traces state)` | token vectors; the completion's traces (finished particles; nil for a dead one) |

**The target.** π(x) ∝ p(x)·exp(complete(x)): p the completion's density
(its end tokens and max-tokens, as `gen.llm.token` has them), complete the
potential's weight of x as a whole sequence (the tokens before an end
token). The prefix weights twist the intermediate targets: step t adds the
drawn token's next-token weight (the end entry for an end token), and a
completion stopped at max-tokens adds complete − prefix, so a particle's
log weight telescopes to log p(x)/q(x) + complete(x), from the initial
prefix([]). exp(log-ml-estimate) is unbiased for Z = Σ p(x)·exp(complete(x));
for a constraint (weights 0 or −∞), the probability that a completion
satisfies it.

**Proposals.** `:lm` draws from the LM's row (the bootstrap); `:masked`
from the row times the twist's next-token weights, renormalized (for a
constraint, the row restricted to its allowed tokens: the locally optimal
proposal of a one-step twist, GenMLX's grammar-masked), by
`gen.llm.constrained/weighted`, the one implementation, which a
constrained completion's sites use too; a function
`(fn [context row] → q)` gives any other, q a normalized float row. A
token's weight is log p − log q + its twist, with p the LM's score from
its ledger (as the completion's), so a particle's log p is its trace's
score bit for bit.

**Reads and branches.** Each step reads the live particles' contexts in one
`lm/batch-next-logprobs`, one advance when the LM holds them (it read them
the step before), the first step the prompt. Particles are token vectors;
the LM's trie owns every branch (GenMLX's live-handle ledger and disposal
scope are not needed); resampling copies vectors. A particle whose weight
falls to 0 stops (dead).

**Keys.** Step t draws with (fold-in (fold-in key t) 0), particle i's token
with (fold-in … i), by `token/dist`; resampling after step t uses (fold-in
(fold-in key t) 1).

## Decisions

1. **Recommendation: the names and shape**: `gen.llm.smc` with
   gen.inference.particle-filter's functions and state keys, the
   resampling by gen.inference.weights. Alternatives: GenMLX's single
   `token-smc` returning a result map; genlm-control's `SMC` object.
2. **Recommendation: the model is a completion and its arguments**, not an
   LM and options: the target is then exactly the token generative
   function's distribution (end tokens, max-tokens) times the potential,
   and the particles are its traces. Alternatives: an LM, as GenMLX's and
   genlm-control's.
3. **Recommendation: the target's ends.** A completion that stops at
   max-tokens is complete (as the token GF has it), weighed by complete;
   the potential weighs the generated tokens only, never the prompt; an end
   token takes the potential's end entry.
4. **Recommendation: the weight's p is the LM's ledger's**, not the row the
   token was drawn from: the particles then agree with the completion's
   traces exactly. On llama.cpp the bootstrap's log p − log q is then the
   noise between the ledger and the row (0 unless the LM read the token
   before in another shape). Alternative: p = q's row for the bootstrap
   (weights exactly 0 without a twist on every engine, traces within the
   noise).
5. **Recommendation: resampling** is systematic by default (GenMLX's; the
   particle filter's default is multinomial) and ESS-gated at n/2 (the
   particle filter's default).
6. **Recommendation: potentials** follow genlm-control's (prefix, complete,
   logw-next over the vocabulary and the end), with weights as float
   arrays, and the end an index past the vocabulary rather than a token.
7. Not built (later): rejuvenation (MCMC moves within or after SMC, e.g.
   regenerating `token/from k` on the completion's traces, as GenMLX does
   after its loop); a batched potential protocol (a grammar's masks for all
   particles at once); SMCP3-style proposals.

## Tests

`gen.llm.smc-test`, on the toy engine with a vocabulary of five tokens
(token 4 ends a completion), max-tokens 4, so every completion (625 at
most) is enumerated and Z computed exactly:

- **Unbiasedness in exp space**: for the three proposals × a constraint and
  a soft twist × resampling at every step, adaptively or never, the mean of
  40 runs' Ẑ (50 particles) is within 5 standard errors of Z; so is the
  unnormalized estimate of a test function's integral.
- **Exact identities**: with no potential the bootstrap's weights are 0 and
  Ẑ = 1; max-tokens 0 weighs the empty sequence; a masked constraint's
  particles all satisfy it, from the first step; particles that all fall
  to weight 0 estimate Z = 0 (an error there would bias Ẑ upwards; only
  resampling them is loud).
- **Draws**: the first tokens of 400 particles follow the row within 5
  standard errors (independence).
- **Reads, traces, keys**: each step is one LM read; particles convert to
  the completion's traces with their log p as scores, consistent with
  assess; runs are pure functions of the key; no GC backstop; misuse is
  loud, each error with a `:reason` (`:invalid-model`, `:invalid-args`,
  `:no-tokens`, `:invalid-option`, `:invalid-potential`, `:invalid-row`,
  `:not-done`, `:impossible`), which the test checks.
- **The host pool**: runs on pools of 1, 2 and 8 threads are the calling
  thread's bit for bit, for every proposal; a recording LM, potential and
  proposal function see every call on the calling thread; a task's error
  is raised as itself, and a task sees the caller's bindings (a batch
  tracer's batched-branch error, from the pool); the default pool is
  shared.
- The masked proposal's rows, renormalized in double and stored as floats,
  pass `token/dist`'s normalization check (within 1e-4 of 1), which a
  proposal function's rows must pass too.
- **`:masked` by `constrained/weighted`** (2026-10-21, review R6, replacing
  smc's own log-sum-exp in double): runs with the soft twist, and with a
  potential nothing satisfies, are the same bit for bit; under a
  constraint that disallows few tokens (`no-repeats`, one or two of five)
  weighted takes the normalizer as log1p(−the disallowed mass), so a
  proposal row can differ in its last float bit. Over 6 runs of 30
  particles the particles were the same, the log weights within 3.7e-9
  and log Ẑ within 5.3e-10. No test moved: the tests of the estimate are
  statistical, and the exact ones (masked constraints satisfied, Ẑ of 0)
  do not depend on the normalizer's last bit.

`gen.llm.potential-test`: dense next weights are consistent, products sum,
misuse is loud, and the contract check catches planted inconsistencies.

Mutations checked (2026-10-21, each applied, the suite run, reverted),
twenty-three, all caught (the last seven of the host pool, m3-par):

| Mutation | Caught by |
|---|---|
| a token's weight without log p − log q | the-estimate-is-unbiased… (12 failures) |
| without the twist | the-estimate-is-unbiased…, a-test-function…, misuse |
| end tokens weighed as tokens, not the end | the-estimate-is-unbiased… |
| no complete − prefix at max-tokens | the-estimate-is-unbiased… (first missed: both potentials had complete = prefix at the capped length; the soft twist now differs) |
| resampling that forgets the weights' mean | the-estimate-is-unbiased… |
| resampling that ignores the weights | the-estimate-is-unbiased… |
| an estimate without the current weights | the-estimate-is-unbiased…, no-tokens… |
| no initial prefix([]) | the-estimate-is-unbiased…, a-test-function… (first invisible: prefix([]) was 0; the soft twist now carries 0.3) |
| one key for every particle | particles-draw-independently, the estimates |
| one key for every step | the estimates |
| the masked proposal without the twist (a bootstrap, still unbiased) | a-masked-constraint-is-always-satisfied |
| the masked proposal weighing end tokens as tokens | a-masked-constraint-is-always-satisfied (its first step; first missed: the constraint let both end alike, and the dead were resampled away) |
| an all-dead state estimated as Z = 1 | particles-all-of-weight-0-estimate-z-as-0 |
| run resampling after every particle finished | particles-all-of-weight-0… |
| potential: dense weights without the base; a product of its first | potential-test |
| the host pool's results in another order | the-host-pool-changes-nothing, the estimates (38) |
| a draw's key by thread, not by particle | the-host-pool-changes-nothing, the reads and draws tests (33) |
| the potential's calls on the pool | no-lm-or-potential-call-runs-on-the-pool |
| a task's error wrapped (ExecutionException) | a-pool-task-s-error-is-raised-as-itself, a-pool-task-sees… |
| a new pool for every run | the-default-pool-is-shared |
| a task without the calling thread's bindings | a-pool-task-sees-the-calling-thread-s-bindings (a batch tracer's batched-branch error, lost on the pool) |
| an unparsable gen.llm.host-threads falling back to the default | the-pool-s-size-is-a-positive-integer-or-loud |

## Measurements

`clojure -M:bench -m gen.smc-bench`: a step of n particles over 32 steps
(resampling at ESS < n/2), against the batched LM read alone, ms a step:

| vocabulary | particles | the LM's batch alone | SMC, no potential | a mask, bootstrap | a mask, masked proposal |
|---|---|---|---|---|---|
| 97 | 8 | 0.60 | 0.65 | 0.32 | 0.41 |
| 97 | 32 | 1.34 | 1.18 | 0.82 | 1.11 |
| 97 | 64 | 1.37 | 1.85 | 1.20 | 1.38 |
| 151,936 | 8 | 10.6 | 12.3 | 12.4 | 19.7 |
| 151,936 | 32 | 41.2 | 48.3 | 48.4 | 78.9 |
| 151,936 | 64 | 59.9 | 95.4 | 93.8 | 150.5 |

(The masked rows with a mask are cheaper than without at 97 tokens: dead
particles stop.) At Qwen3's vocabulary the SMC's own work a particle-step
is 0.2 to 0.55 ms, nearly all of it the host draw (about 335 µs on average
over keys: a sum of exp over the row in token order to the uniform, half
the vocabulary on average); the masked proposal adds about 0.9 ms (the
twist row, and the normalizer over the whole row; measured before R6
replaced smc's own with `constrained/weighted`, which allocates only the
row it returns). Against llama.cpp's
batched decode (13.5 ms for 15 rows, M2 Max) that is a third of a row's
share for the bootstrap, more for the masked proposal.

## The host pool (built 2026-10-21, m3-par)

A particle's proposal row (the LM's row, or the masked row) and its draw
are pure functions of its row, its twist and its key. So a step runs them
on a fixed pool of daemon threads, and collects them in particle order,
each task under the calling thread's bindings. The LM's calls (the step's
batch read, each token's score from the ledger), the potential's (the
next-token weights, prefix and complete) and a proposal function's stay
on the calling thread: none of them need be thread-safe, and the LM reads
in the order the program gives. A task's error is raised as itself once
every task has ended. With one live particle, or no pool, the work runs on
the calling thread.

- **Determinism.** A run is the calling thread's alone, bit for bit, at
  every pool size: particles, weights and the estimate, tested at 1, 2 and
  8 threads for each proposal (the LM's rows, masked, a function).
- **Interface.** `:host-pool` at initialize: an ExecutorService
  (`smc/host-pool n` makes a fixed pool of daemon threads; close it, or
  let the JVM's exit end it), `false` for the calling thread alone, or by
  default the shared pool. Never run SMC inside a task of the pool it is
  given: a step waits for its tasks, queued behind the one that waits, so
  the pool can deadlock.

**PROVISIONAL (owner to confirm), 2026-10-21:**
- **Ownership:** `gen.llm.smc` owns one shared pool, made on first use, of
  daemon threads named `gen.llm.smc host-N`; never the common pool. A
  caller may pass its own. Alternatives: the LM owning it (it would serve
  the token generative function's draws too, which are one particle at a
  time and gain nothing); no default (every caller passes a pool); a pool
  for each run (thread churn).
- **Size:** half the available processors up to 4, as the llama.cpp
  engine's row pool (4 on the M4, its performance cores; the engine's
  thread, the GC and the row pool need the rest), or the system property
  `gen.llm.host-threads`, which is loud (`:invalid-option`) unless a
  positive integer. Alternative: the processors less one.
- **Off switch:** `:host-pool false` for a run; `-Dgen.llm.host-threads=1`
  for the process.

**Measured** (`clojure -M:bench -m gen.smc-bench host`, 2026-10-21 07:50,
this Mac with no other JVM running, after R6): a step of n particles over
16 steps at Qwen3's vocabulary (151,936 tokens) on the toy engine, never
resampling; ms a step:

| particles | proposal | the LM's batch alone | calling thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|---|---|
| 8 | bootstrap | 5.71 | 11.30 | 10.08 | 9.46 | 9.32 |
| 8 | masked, a mask | 5.71 | 12.24 | 10.66 | 9.82 | 9.70 |
| 32 | bootstrap | 41.84 | 44.25 | 39.43 | 37.41 | 36.30 |
| 32 | masked, a mask | 41.84 | 49.44 | 42.41 | 39.37 | 38.38 |
| 64 | bootstrap | 59.32 | 88.40 | 78.43 | 73.36 | 72.50 |
| 64 | masked, a mask | 59.32 | 99.55 | 85.11 | 79.18 | 76.73 |

At 64 particles the pool's 4 threads (the default here) take 17% off the
bootstrap's step and 20% off the masked proposal's; 8 threads a little
more. The host work (the step less the batch) falls from 29 to 14 ms
(bootstrap) and from 40 to 20 (masked). The rest of a step is the LM's
batch (the toy engine's rows, on one thread) and the sequential scores.
(The batch alone reads n contexts at every step, SMC one at its first,
where every particle is at the prompt; so at 32 particles SMC's step can
come under the batch's.) Two earlier runs beside a Metal battery, before
R6, were noisier and slower (the masked proposal at 64 particles took
171 to 177 ms on the calling thread, smc's own normalizer then
allocating a double row besides the proposal's).

**Recommendation (not built): rows with their cumulative probabilities.**
A draw sums exp over the row in token order until the uniform: about
335 µs at 151,936 tokens, half the row on average, for every particle,
even when particles share a row after resampling. An engine computes
those exponentials already when it normalizes a row (llama.cpp's,
`LlamaRuntime/logSumExp`, takes exp(l − m) of every logit).
If a row came with its cumulative probabilities in double (1.2 MB at
Qwen3's vocabulary, besides the row's 600 KB of floats), a draw would be
a binary search, under a microsecond, and its normalization check the
array's last entry. The cost is in the Branching engine (m3-llama's): a
second array per row, written in the same pass, held while the row is.
Better made lazily, per row and once (a branch is a value, so a delay on
it), so that rows no one draws from cost nothing. It does not help the
masked proposal, whose row is the LM's times the twist, made per particle.

## On llama.cpp

`gen.llm.llama-smc-test` runs token SMC over a real model with a
lipogram as the potential (no token whose bytes hold an e: 0 or -Inf, a
grammar's mask in miniature), 32 particles of 12 tokens, by the masked
proposal and by the bootstrap. It checks that every live particle's text
obeys it, that the estimate is finite and at most 0 (a probability), that
the particles' log p are their traces' scores and assess's bit for bit
(score once), that a step after the first is one read, and that the GC
backstop never runs.

`clojure -J-Dgen.llama.lib=<lib> -M:bench -m gen.smc-bench <gguf>` gives a
step of 8, 32 and 64 particles over 16 steps against the LM's batch alone,
with no potential and with a mask (bootstrap and masked), resampling at
ESS < n/2. With Qwen3-0.6B on the M4 (Metal), ms a step, first on the
calling thread (m3-main, 2026-10-21 02:40, beside other agents' JVMs),
then with the host pool and R6 (m3-par, 07:55, no other JVM):

| particles | the LM's batch alone | SMC, no potential | a mask, bootstrap | a mask, masked proposal |
|---|---|---|---|---|
| 8 | 16.2 / 16.1 | 13.0 / 12.4 | 13.0 / 12.5 | 19.0 / 12.7 |
| 32 | 70.4 / 69.4 | 29.4 / 27.4 | 29.0 / 27.2 | 54.8 / 28.9 |
| 64 | 105.1 / 103.5 | 50.5 / 47.2 | 51.4 / 48.2 | 106.4 / 51.3 |

The masked proposal at 64 particles went from 106 to 51 ms a step, now 3
ms over the bootstrap rather than 55: its rows on the pool, and
`constrained/weighted` allocating only the row it returns. SMC comes under
the batch alone because it resamples, after which particles share
contexts and a step reads fewer. Peak RSS 3.5 GB, no GC backstop. With
Qwen3.5 and Gemma-3-270M, and with a grammar potential: in later runs.
