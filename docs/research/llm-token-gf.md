# The token generative function

*2026-10-21, bean genclj-0cvv (M3's core). Built by an agent on branch
m3-tgf, over the LM protocol (`docs/research/llm-lm-protocol.md`) and the
foundations the owner decided on 2026-10-20 (DESIGN.md §9: score once, the
trie owns the branches, host draws). The choices that are the owner's
(addresses, arguments, end and length semantics, the return value, names)
are marked **Recommendation**; the owner confirmed them all on 2026-10-21
(DESIGN.md §9).*

## What exists

| Piece | Where |
|---|---|
| The completion generative function, the token distribution, forced tokens, suffix selections | `gen.llm.token` |
| Tests on the toy engines: the GFI laws, each operation, inference, misuse | `test/gen/llm/token_test.clj` |
| The same over llama.cpp (skipped without `-J-Dgen.llama.test-model`) | `test/gen/llm/llama_token_test.clj` |
| Costs per token and per operation | `bench/gen/token_bench.clj` |

## The API

```clojure
(require '[gen.llm.token :as token] '[gen.llm.lm :as lm] '[gen.gfi :as gfi])

(def answer (token/completion m))                    ; m an LM (gen.llm.lm)
(def prompt (lm/encode m "The capital of Sweden is"))

(gfi/simulate answer [prompt 20])                    ; up to 20 tokens, or to an end token
(gfi/generate answer [prompt 20] (token/forced (lm/encode m " Stockholm" {:add-special? false})))
(gfi/regenerate tr (token/from 3))                   ; resample every token from position 3 on

(def story                                           ; a prompt that depends on a choice
  (gen [n]
    (let [topic (trace :topic (d/bernoulli 0.4))
          tokens (splice :story answer (if topic sweden norway) n)]
      (trace :liked (d/bernoulli (if (some good tokens) 0.9 0.3)))
      tokens)))
```

| Function | Meaning |
|---|---|
| `(completion lm)`, `(completion lm {:end-tokens tokens})` | the generative function of `[prompt max-tokens]`; `:end-tokens`, the set of end tokens (default the LM's `:eog`; `#{}` for exactly max-tokens) |
| `(dist row)` | the distribution of one token given its log-probabilities (a float array, as `lm/next-logprobs` gives): a host draw, its log-probability the row's entry; usable at any `trace` site |
| `(forced tokens)`, `(forced at tokens)` | constraints forcing tokens at the positions at, at + 1, … (0 by default) |
| `(from k)` | the selection of every token from position k on |

A completion's trace holds token i at address i, drawn from the LM given
the prompt and the tokens before it, until an end token (drawn, scored and
kept) or max-tokens of them; its return value is the vector of tokens. The
prompt and max-tokens together must fit the LM's `:limit` (`lm/info`),
checked before any read, so whether they are valid never depends on the
draws (review-lm, D-M2: before, a completion that ended early passed and
one that did not failed in the LM, deep into a run).

## The operations

- **simulate** draws each token from the LM's row at its context, with its
  site's key.
- **generate** walks the positions in order: a run of constrained tokens is
  scored by the LM in one call (`lm/score`: a forced prefix or forced text
  is one teacher-forced read), the others are drawn. The weight is the
  constrained tokens' scores. A forced end token ends the completion; a
  constraint after the end, or past max-tokens, is loud ("not visited").
- **assess** scores the tokens in one LM call; the tokens must be at the
  positions 0 to n-1 without a gap, with no end token but the last, and end
  at an end token or max-tokens, otherwise loud.
- **update** keeps the tokens before the first change (a constraint that
  differs from the token there, or a new prompt) with their scores, and
  rescores the rest through the LM, in one call for each run of known
  tokens. A token made an end token ends the
  completion there and discards the rest; an end token made a common one, or
  a larger max-tokens, draws fresh tokens, whose scores stay out of the
  weight (forward sampling); a smaller max-tokens truncates. The discard is
  the overwritten tokens and those no longer visited. The weight is the new
  score less the old, less the fresh tokens'.
- **regenerate** keeps the tokens before the first change with their
  scores (a selected token redrawn as it was is no change), resamples the
  selected, rescores the others after it; `(from k)` resamples the suffix,
  the tokens whose distributions depend on token k. The weight is the retained
  tokens' score ratio (Eq. 4.1), 0 for a suffix with unchanged arguments.
- **project** sums the selected tokens' scores; **propose** is a simulate.
- **Change hints:** the retdiff of update and regenerate is a vector-diff
  of the token vector (both lengths and the positions whose tokens
  changed), or no-change. A Map of completions reads it through its
  elements' retdiffs, as does any caller of update; inside a `gen` body,
  gen.dynamic marks every argument that is not identical unknown-change, so
  a Map over the tokens there revisits them all until gen.dynamic
  propagates diffs (DESIGN §9, the static language). An update or regenerate that changes nothing (the
  same arguments, no constraint, nothing selected) returns the trace as it
  is, after the same checks.
- **Choice gradients** over tokens are a loud error (`gen.gradients`: the
  choices are not continuous).
- **On M4's GFI** (2026-10-21, m3-main): assess takes a key, which a
  completion does not use (it has no encapsulated randomness). A constraint
  a run does not visit (past max-tokens, after an end token, at an address
  that is no position) is gen.gfi's unvisited error, with every such
  address at once, so a caller sees the paths under its own address
  (`gfi/under`) and a revise can drop old tokens. A token assess lacks is
  reported as gen.dynamic's are (`:missing-choice`, `:path`), which HMC
  reads. A completion reads no parameter store and keeps `IRevisable`'s
  defaults: two compute the same when they share the LM (the same object),
  the end tokens and the vocabulary; a trace another one made (other end
  tokens, another LM) is revised from its choices (`gfi/revise`): each
  token rescored through this one's LM, those past a new end token or a
  smaller max-tokens dropped, fresh tokens drawn past an end that no longer
  ends it. No store or gen-fn swap needs more.

**Score once** (the owner's foundation): every token's score is the LM's
ledger's, `lm/score`, computed once and kept by its trie, never the entry
of the row the token was drawn from. A trace also keeps its tokens' scores,
which update and regenerate reuse wherever a token's context is unchanged,
so a revision that changes no token weighs exactly 0 even after the LM
evicted the rows. (The first build kept them only before the first
constraint or selection, so a constraint equal to its token rescored the
rest; corrected 2026-10-21 after review-lm.) Within one LM, while it holds
the rows it read, assess and a trace's score agree bit for bit and the laws
hold exactly; after an eviction assess reads the rows again, within the
engine's noise. On llama.cpp the row a token was drawn from and the score the LM
keeps for it differ by the engine's noise when the LM read that token
before in another shape (a teacher-forced read of a forced or assessed
sequence): sampling is then from the row, scoring from the ledger, within
the band (`llm-branching.md`).

**Draws** (the owner's foundation): token i is drawn with `(fold-in key i)`
of the operation's key, by `token/dist`: one uniform from gen.random on the
JVM backend (float64), its inverse CDF over the row in token order. So the
same key draws the same tokens from the same rows, and in update and
regenerate a fresh token at position i draws from the same schedule. (On
llama.cpp a row read again after an eviction differs within the engine's
noise, so a run's scores, and rarely its tokens, depend on the LM's history
as well.) Under a batch tracer (`gen.batched`) the operations, and
`token/dist`'s draws, raise the batched-branch error with its reason, so a
batched run falls back to one particle at a time on every backend. (Without
the check a batched key fails in the draw with a ClassCastException, and
gen.batched falls back all the same: the check is there for the reason.)

## Decisions

1. **Recommendation: the addresses** are the positions 0, 1, … at the
   completion's root, as Map's elements and Unfold's steps are, so a
   spliced completion's tokens are `[:story 0]`, `[:story 1]`, … and a
   trace is called `(tr 3)`. Alternatives: GenMLX's `:t0`, `:t1`, …
   (keywords built from strings, which selections and loops then parse),
   or `[:tokens i]` (room for other choices beside the tokens, which a
   completion does not have).
2. **Recommendation: the arguments** are `[prompt max-tokens]`: a token
   vector (text is encoded by the caller, `lm/encode`, so the generative
   function depends on tokens only) and a non-negative integer, both
   arguments so that update can change them (a larger max-tokens extends a
   completion that stopped at the maximum, a new prompt rescores every
   token). Alternatives: the prompt as text; max-tokens as a construction
   option.
3. **Recommendation: end and length.** An end token is a choice like any
   other: drawn, scored, kept in the trace and the return value, and it ends
   the completion; the end tokens default to the LM's `:eog` (every
   end-of-generation token, which the GGUF names), `:end-tokens` replaces
   them, and `#{}` gives completions of exactly max-tokens. The option was
   `:eos` as first built; renamed 2026-10-21 (confirmed by the owner
   that day), because it is a set while the vocabulary's `:eos` is one token
   (review-lm). Alternatives: `:eog`, the vocabulary's name for its set. A completion that reaches
   max-tokens without one simply stops. Alternatives: the end token kept
   out of the return value, or GenMLX's single EOS id.
4. **Recommendation: the return value** is the vector of drawn tokens, the
   end token included: exactly the choices in order, decodable with
   `lm/decode`. Alternatives: the prompt and the tokens (GenMLX's), or
   `{:tokens :ended?}`.
5. **Recommendation: the names**: the namespace `gen.llm.token` (alias
   `token`), `completion`, `dist`, `forced`, `from`. Alternatives:
   `gen.llm.gf/lm-gf`, `token/sequence`, `token/next`.
6. **Recommendation: no sampling transforms in a completion.** Temperature,
   top-k and top-p (GenMLX's `sampling.cljs`) define another distribution;
   a tempered completion would be its own option, renormalizing each row on
   the host (O(vocabulary) a token, as the draw), scored by its own rows
   rather than the LM's ledger. Left for a later bean.
7. **Recommendation: batching** is gen.batched's per-particle fallback
   (below). Native batching belongs to the inference algorithms that know
   their particles move together (genclj-loyp).
8. Smaller choices (recommended):
   - the trace keeps per-token scores (`:gen.llm.token/scores`), as
     GenMLX's ledger, for project and the reuse in update and regenerate;
   - `from` lives in `gen.llm.token`; a suffix selection over integer
     addresses would serve Unfold too, and could move to `gen.selection`;
   - `token/dist` checks its row's type and size when made, and a draw its
     normalization, as far as the walk goes, from the total it sums anyway:
     a row of no mass, of probabilities in place of their logs, with NaN,
     or short of 1 where the walk passes it is loud (`:invalid-row`;
     review-lm). A full pass over the row (O(vocabulary) a token) is not
     made.

## Batching

A batched run (`gen.batched`, or importance sampling and a particle filter
with `:batched? true`) of a model containing a completion falls back to one
particle at a time, with the reason "token draws are on the host, one
particle at a time". That is correct and costs nothing beyond the scalar
runs, but each particle reads its own tokens: N particles of a token SMC
step are N one-token reads, not one advance.

A native batched completion would take: in a batched simulate, the N
particles' contexts advanced together by `lm/batch-next-logprobs` (one
advance), N uniforms from `(rng/uniform (fold-in key i) [N])`, an inverse
CDF per row, and the scores from the ledger (one `lm/score` per particle,
or a batch score in the LM). The obstacle is the trace, not the reads:
particles end at different lengths, and a batched trace holds one value
array per address for every particle, so ragged completions need a mask or
padding (GenMLX's masked-EOS algebra pads with a token of log-probability
0), which changes what a completion's choices are. Clean only for
completions of exactly max-tokens (`:end-tokens #{}`). **Recommendation:** keep the
fallback, and batch where the inference algorithm moves its particles in
step: token SMC (genclj-loyp) calls `lm/batch-next-logprobs` over its
particles' contexts each step, one advance for all of them, and resamples
contexts.

## Tests

`gen.llm.token-test`, on the toy engine (exact distributions) and the cell
engine (`gen.llm.cell-engine`):

- **The GFI laws** (`laws/deflaws`, the general, forward-sampling and
  fixed-support sets, 100 trials each) over a completion that ends on a
  third of the tokens, one with no end tokens, one ending on EOS, a gen
  model whose prompt depends on a choice (`:topic`) and whose observation
  depends on the tokens, a Map of completions over two prompts, and a
  conversation (an Unfold whose turn's prompt is the context so far,
  extended by the turn's completion), with changed arguments (other
  lengths, other prompts).
- **Across generative functions** (`laws/across`): a trace made by one
  completion, updated and regenerated by another, with other end tokens
  (both ways) or over another LM; and a test of the drops (past a new end
  token, past a smaller max-tokens) and the fresh tokens a revise draws.
- **Score once:** the same laws (40 trials, a fixed seed) on the noisy cell
  engine, whose reads differ with their shape and cells as llama.cpp's do,
  with room for every read: exact. Over 97 tokens, random programs seldom
  draw a token the LM read before in another shape, so the laws run over 5
  tokens as well, completions of exactly 8, where draws keep following
  paths read teacher-forced before. Targeted tests cover the same ground
  directly: at 20 prompts, a token drawn after the LM scored it
  teacher-forced (an end token at some) keeps the LM's first score, so
  assess agrees, while the row it was drawn from differs; and after
  evictions, an update constraining a token to its value, one extending the
  completion, and a regenerate redrawing a token as it was all weigh exactly
  0, while the LM's re-read scores differ. In the LM's suite,
  `a-score-is-the-first-reads` holds the cache to the first read.
- **Rows that are not normalized** are loud at the draw: no mass, the
  probabilities in place of their logs, NaN (every key), a mass of 0.2 (the
  keys whose uniform falls past it, within 5 standard errors of 4 in 5).
- **Draws:** `token/dist`'s frequencies over 5,000 keys within 5 standard
  errors; a completion's first token over 4,000 keys against the exact row
  (each token expected 10 times or more on its own, the rarer pooled);
  token i is exactly `token/dist`'s draw with `(fold-in key i)`.
- **Each operation against the exact model:** end tokens and max-tokens,
  forced prefixes and text, forced ends, assess (and its loud refusals),
  update of a token mid-sequence (the suffix rescored, the discard), of the
  prompt, of max-tokens, of an end token both ways, regenerate from a
  boundary (weight 0, the suffix resampled) and of one token, project.
- **Selection MH is stationary** (law 10): 300 chains of 30 moves (the
  whole completion, its second token, its first token alone) end
  distributed as the exact posterior given liked, by enumeration over 97²
  pairs: P(a token in {3, 4, 5}) within 5 standard errors, and the mean of
  log p(t1 | t0) within 5 standard errors of the chains' mean. The
  first-token move keeps t1 with its context changed, so the completion's
  regenerate weight enters the acceptance ratio; with suffix moves alone it
  is always 0, and the test could not see it (review-lm, D-M3).
- **The laws check a revised trace's arguments** (gen.laws, for every
  generative function): the update and regenerate laws require the
  revised trace to hold the new arguments, so a revision that drops them
  fails (review-lm, D-M1; main's 13 law suites, 818 tests, pass with it).
  And a regenerate with nothing selected and a new max-tokens extends or
  truncates the completion.
- **Inference:** importance sampling's estimate (2,000 samples, within 5
  standard errors of the mean importance weight) and 20 particle filters
  of 100 particles over two steps estimate the exact marginal likelihood
  (by enumeration over 97² token pairs) within 5 standard errors; MH over
  token traces (suffix and topic moves) keeps the observation, moves,
  stays consistent; batched runs fall back with independent particles; the
  LM's engine never ran its GC backstop.
- **Misuse** is loud, each error with a `:reason`: arity, prompts,
  max-tokens, constraints (past the maximum, at keywords, not tokens,
  nested), bad positions and end sets, a completion over something not an
  LM, a completion past the LM's limit (before any read, for a completion
  that would end early too; one of exactly the limit is drawn).

Mutations of `gen.llm.token` checked (2026-10-21, each applied, the suite
run, reverted; again after review-lm's fixes, the set below), twenty-nine,
all caught:

| Mutation | Caught by |
|---|---|
| a drawn token scored from its row, not the LM's ledger | drawn-tokens-are-scored-as-the-lm-keeps-them (40 failures), and the five-token score-once laws (regenerate-keeps-unselected-choices at the suite's seed; one to seven laws at each of six seeds). Before the five-token case, the 97-token laws caught it in 2 runs of 4 (review-lm), never at a fixed seed |
| one key for every site | token-i-is-drawn-with-its-site-key, importance-sampling…, batched-runs…, drawn-tokens… (62) |
| the inverse CDF one token off | the-distribution-draws…, a-completion-draws…, a-particle-filter… (8) |
| token/dist reading the next entry | the-distribution-reads-its-row (1) |
| end tokens ignored | most laws and tests (178 failures, 21 errors) |
| a run of known tokens passing an end token | update laws, on both engines; update-rescores-after-a-change (6) |
| update weighing fresh tokens | the update weight laws, a-particle-filter…, unchanged-tokens-keep-their-scores-after-eviction (7) |
| update not discarding truncated tokens | updates-are-reversible…, update-weight…, update-rescores… (6) |
| regenerate weighing selected tokens | the regenerate weight laws, regenerate-from-a-boundary… (10) |
| regenerate keeping selected tokens | regenerate-from-a-boundary…, mh-moves…, mh-over-tokens-targets-the-posterior (3) |
| generate weighing unconstrained tokens | the generate weight laws, generate-forces-tokens, a-particle-filter… (12) |
| project ignoring the selection | the projection laws (19) |
| the retdiff missing the changed positions | the laws' retdiff soundness, update-rescores… (5) |
| the retdiff's lengths swapped | the same (5) |
| update's no-change shortcut taken with constraints | the update laws, on both engines; update-rescores… (12) |
| regenerate's no-change shortcut taken with a selection | mh-moves…, mh-over-tokens-targets-the-posterior, regenerate-from-a-boundary… (3) |
| no batched-branch check | batched-runs-fall-back… (1) |
| update's shortcut keyed on the prompt alone (review-lm's D1) | update-rescores…, the update laws on both engines, a-particle-filter… (6) |
| regenerate's shortcut keyed on the prompt alone (D2, which the suite missed before) | regenerating-nothing-with-new-arguments-revises, the regenerate laws with the arguments check (7) |
| regenerate's weight negated (D3) | the regenerate weight laws, regenerate-from-a-boundary…, and law 10's log p(t1 \| t0) (8) |
| regenerate's weight 0 (D4, which law 10 missed before) | the same (8) |
| the limit check one token short | a-completion-past-the-lms-limit-is-loud-before-any-read (2) |
| no limit check | the same (2) |
| no reuse of the trace's scores (review-lm's T2 and T3, which the suite missed before) | unchanged-tokens-keep-their-scores-after-eviction (5) |
| reuse across a new prompt | update-rescores-after-a-change, the update and regenerate laws, mh-moves… (6) |
| reuse of the first changed token's score | update-rescores…, regenerate-from-a-boundary…, the update and regenerate laws on both engines, mh-moves… (9) |
| reuse after the run has left the trace | regenerate-from-a-boundary…, the regenerate laws on both engines (3) |
| no normalization check where the walk stops | a-row-that-is-not-normalized-is-loud (1) |
| none at the row's end | the same (3) |

And of `gen.llm.cache` (review-lm's L1): a held entry's row scoring the
token after it before the path's first read is caught by
`gen.llm.lm-test/a-score-is-the-first-reads`, drawn-tokens… and the
score-once laws (35 failures); the LM's suite alone missed it before, its
exact engines unable to tell the two reads apart.

## Measurements (toy engine, this Mac, pure JVM)

`clojure -M:bench -m gen.token-bench` (criterium means):

| | µs |
|---|---|
| simulate, per token, 50 tokens, each a one-token read | 37.5 |
| the LM calls alone (next-logprobs and an argmax), per token | 33.8 |
| assess of a held 50-token completion (its scores from the trie) | 20.3 |
| update of token 25 of 50 (the 24 after it rescored in one read) | 65.4 |
| regenerate from token 25 (25 tokens drawn) | 875 |

So a completion adds about 4 µs a token to the LM calls under it, at a
97-token vocabulary. At Qwen3's 151,936 tokens its own work a token is the
row's copy (18 µs) and the host draw, the score's lookup under a
microsecond. The draw costs about 335 µs on average over keys (353 µs on a
peaked row): it sums exp over the row in token order until the uniform, half
the vocabulary on average. (The bench's single key gave 80 µs, the figure
first reported; corrected 2026-10-21.) So about 350 µs a token, against a
llama.cpp decode of 5.8 ms (M2 Max, one branch): 6%. Rows that came with
their probabilities as well (the engine computes the exponentials when it
normalizes) would make a draw a float sum, tens of microseconds: a
recommendation for the engine layer, not built.

## For the llama.cpp runs

`gen.llm.llama-token-test`: every law (10 trials each) on a real model
within one LM, forced text (" Stockholm" after "The capital of Sweden is":
a likely continuation, assess agreeing bit for bit), a completion decoded
to text, no backstop. With Qwen3-0.6B, Qwen3.5-0.8B and Gemma-3-270M, once
the M3 move to this Mac is done.

## For the next beans

- **genclj-loyp (token SMC).** Particles hold token contexts; a step is one
  `lm/batch-next-logprobs` over them (one advance), a draw per particle from
  `token/dist` with its key, its score from the LM's ledger; resampling
  chooses contexts. A completion's trace can be built afterwards with
  `generate` (`token/forced`), which reads nothing new.
- **genclj-q38h (grammar).** Masks apply to rows on the host before
  `token/dist`; a masked completion is another distribution, scored by its
  masked, renormalized rows (as a tempered one would be).
