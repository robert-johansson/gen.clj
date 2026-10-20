# The token generative function

*2026-10-21, bean genclj-0cvv (M3's core). Built by an agent on branch
m3-tgf, over the LM protocol (`docs/research/llm-lm-protocol.md`) and the
foundations the owner decided on 2026-10-20 (DESIGN.md §9: score once, the
trie owns the branches, host draws). The choices that are the owner's
(addresses, arguments, end and length semantics, the return value, names)
are marked **Recommendation**; the owner being away, the lead records them
PROVISIONAL in DESIGN.md §9.*

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
| `(completion lm)`, `(completion lm {:eos tokens})` | the generative function of `[prompt max-tokens]`; `:eos`, the set of end tokens (default the LM's `:eog`; `#{}` for exactly max-tokens) |
| `(dist row)` | the distribution of one token given its log-probabilities (a float array, as `lm/next-logprobs` gives): a host draw, its log-probability the row's entry; usable at any `trace` site |
| `(forced tokens)`, `(forced at tokens)` | constraints forcing tokens at the positions at, at + 1, … (0 by default) |
| `(from k)` | the selection of every token from position k on |

A completion's trace holds token i at address i, drawn from the LM given
the prompt and the tokens before it, until an end token (drawn, scored and
kept) or max-tokens of them; its return value is the vector of tokens.

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
- **update** keeps the tokens before the first change (a constraint, or a
  new prompt) with their scores, and rescores the rest through the LM, in
  one call for each run of known tokens. A token made an end token ends the
  completion there and discards the rest; an end token made a common one, or
  a larger max-tokens, draws fresh tokens, whose scores stay out of the
  weight (forward sampling); a smaller max-tokens truncates. The discard is
  the overwritten tokens and those no longer visited. The weight is the new
  score less the old, less the fresh tokens'.
- **regenerate** keeps the tokens before the first selected one, resamples
  the selected, rescores the others; `(from k)` resamples the suffix, the
  tokens whose distributions depend on token k. The weight is the retained
  tokens' score ratio (Eq. 4.1), 0 for a suffix with unchanged arguments.
- **project** sums the selected tokens' scores; **propose** is a simulate.
- **Change hints:** the retdiff of update and regenerate is a vector-diff
  of the token vector (both lengths and the positions whose tokens
  changed), or no-change, so code downstream (a Map over the tokens, say)
  revisits only those. An update or regenerate that changes nothing (the
  same arguments, no constraint, nothing selected) returns the trace as it
  is, after the same checks.
- **Choice gradients** over tokens are a loud error (`gen.gradients`: the
  choices are not continuous).

**Score once** (the owner's foundation): every token's score is the LM's
ledger's, `lm/score`, computed once and kept by its trie, never the entry
of the row the token was drawn from. A trace also keeps its tokens' scores,
which update and regenerate reuse wherever a token's context is unchanged,
so an identity update weighs exactly 0 even after the LM evicted the rows.
Within one LM, assess and a trace's score agree bit for bit; the laws hold
exactly. On llama.cpp the row a token was drawn from and the score the LM
keeps for it differ by the engine's noise when the LM read that token
before in another shape (a teacher-forced read of a forced or assessed
sequence): sampling is then from the row, scoring from the ledger, within
the band (`llm-branching.md`).

**Draws** (the owner's foundation): token i is drawn with `(fold-in key i)`
of the operation's key, by `token/dist`: one uniform from gen.random on the
JVM backend (float64), its inverse CDF over the row in token order. So a
completion is a pure function of its key, and in update and regenerate a
fresh token at position i draws from the same schedule. Under a batch
tracer (`gen.batched`) the operations, and `token/dist`'s draws, raise the
batched-branch error, so a batched run falls back to one particle at a time
on every backend (forcing the JVM backend's draw under a tracer over MLX
would otherwise give every particle the same token).

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
   end-of-generation token, which the GGUF names), `:eos` replaces them, and
   `#{}` gives completions of exactly max-tokens. A completion that reaches
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
   - `token/dist` checks its row's type and size, not its normalization
     (an O(vocabulary) pass a token): rows come from an LM.

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
completions of exactly max-tokens (`:eos #{}`). **Recommendation:** keep the
fallback, and batch where the inference algorithm moves its particles in
step: token SMC (genclj-loyp) calls `lm/batch-next-logprobs` over its
particles' contexts each step, one advance for all of them, and resamples
contexts.

## Tests

`gen.llm.token-test`, on the toy engine (exact distributions) and the cell
engine (`gen.llm.cell-engine`):

- **The GFI laws** (`laws/deflaws`, general and forward-sampling, 21 laws,
  100 trials each) over a completion that ends on a third of the tokens,
  one with no end tokens, one ending on EOS, and a gen model whose prompt
  depends on a choice (`:topic`) and whose observation depends on the
  tokens, with changed arguments (other lengths, another prompt).
- **Score once:** the same laws (40 trials) on the noisy cell engine, whose
  reads differ with their shape and cells as llama.cpp's do, with room for
  every read: exact. And a targeted test: a token drawn after the LM scored
  it teacher-forced keeps the LM's first score, so assess agrees, while the
  row it was drawn from differs.
- **Draws:** `token/dist`'s frequencies over 5,000 keys within 5 standard
  errors; a completion's first token over 4,000 keys against the exact row;
  token i is exactly `token/dist`'s draw with `(fold-in key i)`.
- **Each operation against the exact model:** end tokens and max-tokens,
  forced prefixes and text, forced ends, assess (and its loud refusals),
  update of a token mid-sequence (the suffix rescored, the discard), of the
  prompt, of max-tokens, of an end token both ways, regenerate from a
  boundary (weight 0, the suffix resampled) and of one token, project.
- **Selection MH is stationary** (law 10): 300 chains of 30 moves (the
  whole completion, then its second token) end distributed as the exact
  posterior of P(a token in {3, 4, 5} | liked), by enumeration, within 5
  standard errors.
- **Inference:** importance sampling (2,000 samples) and 20 particle filters
  of 100 particles over two steps estimate the exact marginal likelihood
  (by enumeration over 97² token pairs) within 5 standard errors; MH over
  token traces (suffix and topic moves) keeps the observation, moves,
  stays consistent; batched runs fall back with independent particles; the
  LM's engine never ran its GC backstop.
- **Misuse** is loud: arity, prompts, max-tokens, constraints (past the
  maximum, at keywords, not tokens, nested), bad positions and end sets.

Mutations of `gen.llm.token` checked (2026-10-21, each applied, the suite
run, reverted), eighteen, all caught:

| Mutation | Caught by |
|---|---|
| a drawn token scored from its row, not the LM's ledger | a-drawn-token-is-scored-as-the-lm-keeps-it (the laws cannot see it: random programs never draw a token the LM read before in another shape) |
| one key for every site | token-i-is-drawn-with-its-site-key, importance-sampling…, batched-runs… (22 failures) |
| the inverse CDF one token off | the-distribution-draws…, a-completion-draws…, a-particle-filter… (7 failures) |
| token/dist reading the next entry | the-distribution-reads-its-row (1) |
| end tokens ignored | most laws and tests (178 failures) |
| a run of known tokens passing an end token | update laws, on both engines (6) |
| update keeping the first changed token's old value | identity-update…, the update laws, a-particle-filter… (20) |
| update weighing fresh tokens | the update weight laws, a-particle-filter… (6) |
| update not discarding truncated tokens | updates-are-reversible…, update-weight… (6) |
| regenerate weighing selected tokens | the regenerate weight laws (10) |
| regenerate keeping selected tokens | regenerate-from-a-boundary…, mh-moves…, mh-over-tokens-targets-the-posterior (3) |
| generate weighing unconstrained tokens | the generate weight laws, a-particle-filter… (9) |
| project ignoring the selection | the projection laws (19) |
| the retdiff missing the changed positions | the laws' retdiff soundness (5) |
| the retdiff's lengths swapped | the laws' retdiff soundness (5) |
| update's no-change shortcut taken with constraints | the update laws, on both engines (12) |
| regenerate's no-change shortcut taken with a selection | mh-moves…, mh-over-tokens-targets-the-posterior, regenerate-from-a-boundary… (3) |
| no batched-branch check | batched-runs-fall-back… (1) |

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
row's copy (18 µs) and the host draw (80 µs), the score's lookup under a
microsecond: about 100 µs, against a llama.cpp decode of 5.8 ms (M2 Max,
one branch): under 2%.

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
