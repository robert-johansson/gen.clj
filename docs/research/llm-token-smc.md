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
proposal of a one-step twist, GenMLX's grammar-masked); a function
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
- The masked proposal's rows, renormalized in double and stored as floats,
  pass `token/dist`'s normalization check (within 1e-4 of 1), which a
  proposal function's rows must pass too.

`gen.llm.potential-test`: dense next weights are consistent, products sum,
misuse is loud, and the contract check catches planted inconsistencies.

Mutations checked (2026-10-21, each applied, the suite run, reverted),
twenty-two, all caught (the last six of the host pool, m3-par):

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
twist row, and a log-sum-exp over the whole row). Against llama.cpp's
batched decode (13.5 ms for 15 rows, M2 Max) that is a third of a row's
share for the bootstrap, more for the masked proposal.

**Recommendation (not built): parallel host work.** A particle's proposal
row and draw are pure functions of its row, its twist and its key, so they
can run on a small fixed pool (as the engine's row normalization, the
Branching review's N1), deterministic whatever the threads; the LM's
scores and the potential's calls stay sequential (a potential need not be
thread-safe). And rows that came with their probabilities (the engine
computes the exponentials when it normalizes) would make a draw a float
sum, tens of microseconds.

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
with no potential and with a mask (bootstrap and masked). With a grammar
potential (m3-grammar) once it joins. The runs on the M4 are recorded
below when made.
