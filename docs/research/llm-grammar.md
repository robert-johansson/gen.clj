# Grammar-constrained decoding

*2026-10-21, bean genclj-q38h (M3). Built by an agent on branch m3-grammar,
over the token generative function (`docs/research/llm-token-gf.md`) and
the potentials of `gen.llm.potential` (m3-smc, a83022f). The owner is away
overnight: the choices that are theirs (the dependency, the semantics of a
constrained completion, names) are marked **Recommendation**; the owner
confirmed them all on 2026-10-21 (DESIGN.md §9).*

## What exists

| Piece | Where |
|---|---|
| Regular languages as terms; a regex syntax; DFAs by Brzozowski derivatives | `gen.llm.regex` |
| DFAs over bytes: trimmed, minimal (Hopcroft), canonical | `gen.llm.dfa` |
| The token byte trie; the sweep of every token from a DFA state; a text's probability over its tokenizations | `gen.llm.bytes` |
| A language read through a vocabulary: exact masks, a potential | `gen.llm.grammar` |
| A completion under a potential (the middleware) | `gen.llm.token` (`:potential`) |
| Tests against java.util.regex, brute force, exact enumeration | `test/gen/llm/` (`regex_test`, `grammar_test`, `bytes_test`, `constrained_test`; `byte_tokenizer`) |
| Costs at Qwen3's, Qwen3.5's and Gemma 3's vocabularies | `bench/gen/grammar_bench.clj` |

No new dependency. Nothing outside `gen.llm.*` requires these namespaces.

## The API

```clojure
(require '[gen.llm.grammar :as grammar] '[gen.llm.regex :as regex] '[gen.llm.token :as token]
         '[gen.llm.constrained :as constrained] '[gen.llm.potential :as potential] '[gen.gfi :as gfi])

(def phone (grammar/grammar m "[0-9]{3}-[0-9]{4}"))   ; m an LM: masks over its vocabulary
(def answer (token/completion m {:potential phone}))  ; completions held to the grammar
(def tr (gfi/simulate answer [prompt 12]))           ; a phone number, token by token
(constrained/conditional-weight tr)                  ; its importance weight for the LM conditioned on it

(def no-close                                         ; terms compose: text without a closing tag
  (regex/intersect (regex/star regex/any-char)
                   (regex/complement (regex/cat (regex/star regex/any-byte) (regex/text "</p>")
                                                (regex/star regex/any-byte)))))
(grammar/grammar m (regex/cat (regex/text "<p>") no-close (regex/text "</p>")))
```

| Function | Meaning |
|---|---|
| `(regex/parse s)`, the term constructors (above) | a regular language over bytes, as a value |
| `(regex/dfa r)`, `(regex/dfa r {:max-states n})`, `(regex/matches? r text)` | its minimal DFA; membership of a string |
| `(dfa/dfa {:start :accept :transitions})`, `step`, `walk`, `accepting?`, `matches?`, `->data` | DFAs over bytes from data, and their operations |
| `(grammar/grammar vocab language)`, `(grammar/grammar vocab language {:end-tokens ts})` | a language (regex, term or DFA) over a vocabulary (an LM, or a byte vocabulary): masks for every state the tokens reach; a potential. The end tokens (default an LM's `:eog`, none for a byte vocabulary) are not text, and must be the completion's |
| `(grammar/state g tokens)`, `advance`, `live?`, `accepting?`, `allowed?`, `allowed`, `automaton`, `end-tokens`, `info` | a grammar's states and masks |
| `(potential/prefix g ctx)`, `complete`, `logw-next` | a grammar as a potential: 0 or -Inf; the mask as n-vocab + 1 weights |
| `(token/completion m {:potential p})` | the completion under a potential (any potential: a grammar, a product, a weighted one), whose declared end tokens (`potential/end-tokens`) must be the completion's |
| `(constrained/weighted end-tokens row ws)` | a site's distribution: the row times the weights, renormalized, `[q log-z]` (for token SMC's masked proposal too) |
| `(constrained/conditional-weight tr)` | log p(x) + w(x) - log q(x) |
| `(bytes/byte-lm m opts)`, `prefix-logprob`, `text-logprob`, `next-byte-logprobs` | a text's probability over its tokenizations |
| `(bytes/byte-trie byte-vocab)`, `sweep` | the token byte trie and its walk from a DFA state |


## Regular languages over bytes (`gen.llm.regex`, `gen.llm.dfa`)

A language is a **term**, a value: `nothing`, `epsilon`, `(text s)`,
`(chars spec …)`, `(chars-except spec …)`, `any-char`, `(bytes-of bs)`,
`any-byte`, `(cat r …)`, `(alt r …)`, `(star r)`, `(plus r)`, `(opt r)`,
`(repeat r lo hi)`, `(intersect r …)`, `(complement r)`. `(parse regex)` reads
the common regex syntax into one; `(dfa r)` compiles a term or a regex.

- **Bytes, with characters as UTF-8.** The DFA reads bytes, because tokens
  are byte strings and many cross character boundaries (a byte-level BPE
  token can hold half of an `é`). A character class is lowered to the UTF-8
  encodings of its code points (the splitting of Rust's regex-syntax
  `Utf8Sequences`; surrogates excluded), so `[^"]` admits every character
  but `"`, of any length, where GenMLX's grammar admitted printable ASCII
  only (its genmlx-6dax: non-ASCII text was off-grammar).
- **Derivatives.** Each DFA state is a derivative of the term, the terms
  kept in a normal form (sets for alternatives and conjuncts, flat
  concatenations), which makes them finitely many; more rules (a repeat of
  a nullable body has lower bound 0, `(x|ε){0,d}` is `x{0,d}`,
  `(x{0,b}){0,d}` is `x{0,bd}`, ε goes beside a nullable alternative) make
  equal languages' terms meet sooner. A DFA can still be exponentially
  larger than its regex (nested counted repeats of overlapping
  alternatives, `((a|b)*a(a|b)){2,4}`, make it track sets of positions),
  so construction stops loudly at `:max-states` (default 20,000); the
  random tests found one such term, now a test. Derivatives give
  intersection and complement for free: "text without `</tool_call>`",
  which GenMLX's tool calls needed a hybrid masker for, is
  `(intersect (star any-char) (complement (cat (star any-byte) (text "</tool_call>") (star any-byte))))`.
  Counted repeats stay counters (`a{0,999}` has 1,000 states, no truncation
  at 999 as GenMLX's). Each state's derivatives are taken once for each
  class of bytes the term distinguishes.
- **Minimal and canonical.** `gen.llm.dfa` trims (every state reached and
  live, a dead byte leads to -1), merges equivalent states by Hopcroft's
  algorithm, and numbers states breadth-first. So two DFAs of one language
  are `=`, a walk that is not -1 has read a prefix of the language, and a
  DFA's size is its language's.
- **The syntax agrees with java.util.regex** (UNIX_LINES) on every regex it
  accepts, and refuses loudly what it does not support rather than give it
  another dialect's meaning: backreferences, lookaround, word boundaries,
  anchors inside, inline flags, possessive quantifiers, `\p`, nested classes
  and `&&`, unescaped `]` `{` `}` outside a class or `[` inside, and a lone
  surrogate (no character; the constructors refuse one too). `^` first and
  `$` last are allowed (matching is whole-string); GenMLX read them as
  literal characters (genmlx-bgyu). A lazy `?` is accepted: it changes no
  language. A backslash before an ASCII letter or digit is reserved, as
  java.util.regex and Python have it; before any other character it is
  that character. Code points are compared as numbers throughout: a
  character past U+FFFF is never read by its low 16 bits (U+1002A once
  parsed as `*`, found by the round-3 review; the tests' alphabet holds
  U+1002A and U+2007C, also escaped).
- **Nested repeats** `(x{0,b}){0,d}` become `x{0,bd}` exactly while bd
  fits a long; past it the nesting stays (65536⁴ once wrapped to the
  empty string's language).

## Masks over a vocabulary (`gen.llm.bytes`, `gen.llm.grammar`)

`(grammar vocab language)` reads a language through a vocabulary (an LM, or
a vector of byte arrays by token). A token sequence's text is its tokens'
bytes, concatenated (the LM's decode, special tokens rendered); the
grammar holds the text.

- **The trie.** The vocabulary's tokens sorted by their bytes are the
  trie's preorder, so the trie is built in one pass and lives in flat
  arrays: node i's subtree is the nodes i to end[i] - 1, and the tokens
  under it are one contiguous run.
- **The sweep.** A state's mask is one walk of the trie with a stack of DFA
  states by depth: each shared prefix is advanced once, a dead subtree is
  skipped in one jump, and nothing is allocated per token. The mask is a
  bitset over the vocabulary (19 KB at 151,936 tokens).
- **Exact by reachability.** Every state the vocabulary reaches from the
  start is explored when the grammar is made, and a token is allowed only
  when its bytes lead to a state from which some sequence of tokens (not
  merely of bytes) reaches acceptance. With a vocabulary that has every
  byte as a token (byte-level BPE) the two coincide; without (the toy
  ASCII tokenizer, SentencePiece vocabularies missing bytes), a byte-live
  state may be one no token leaves, and a constrained completion that
  entered it could not go on. So `prefix` is exact (0 or -Inf by
  reachability), and a constrained completion never strands. An empty
  language is a loud `:empty-language`, and one no sequence of tokens spells
  a loud `:inexpressible`.
- **Special tokens** are read by their bytes (llama.cpp renders them), as
  the text is. **End tokens are not text**: a grammar knows its end tokens
  (an LM's `:eog` by default, or `:end-tokens`), leaves them out of its
  trie and liveness (advancing by one is dead), and the end is the
  potential's end entry, allowed where the text is complete. A completion
  under a grammar must have the grammar's end tokens (loud
  `:mismatched-end-tokens` otherwise), and a language that needs an end
  token's bytes is `:inexpressible`. (The first build read every token as
  text: a grammar `[ab]{3}\n` with a newline end token passed `prefix` and
  stranded every completion, found by the round-3 review.)
- **A potential declares its end tokens** (2026-10-22, the post-M5
  review's P4-02 and I4-02, bean genclj-1lpj; DESIGN §9). The agreement
  was a type test for `Grammar` on the completion's own `:potential`, so a
  grammar inside `potential/product` passed it (and the completion failed
  mid-run with `:zero-weight`), and token SMC never checked (a mismatched
  pair estimated another Z: 0.37 where the matched pair gives 1). Now a
  potential declares the end tokens it reads contexts with through an
  optional protocol, `(potential/end-tokens p)`: a set, or nil for no
  opinion. A grammar declares its own; `potential` takes `:end-tokens`; a
  product carries its factors' set and is `:mismatched-end-tokens` when
  made if two disagree. One check, `potential/check-end-tokens`, compares
  a potential's set with a completion's, from `token/completion` and from
  token SMC's `initialize` for its `:potential` (only that check there: a
  potential that rules every sequence out stays an estimate of Z = 0).

## Measurements

`clojure -M:bench -m gen.grammar-bench` on the Mac mini M4 (pure JVM,
Graal JIT), alone on a quiet machine (2026-10-21, m3-round3 764469e),
with the vocabularies of the three GGUF files in `/Users/robert/gen-clj-models`
(read from their headers; llama.cpp not loaded): Qwen3's byte-level BPE
(151,936 tokens; its trie 333,802 nodes, built in 185 ms), Qwen3.5's
(248,320; 592,346 nodes, 224 ms) and Gemma 3's SentencePiece (262,144;
600,436 nodes, 161 ms).

| language | DFA states | DFA ms (Qwen3's run / the others') | states reached by tokens | every mask, ms (Qwen3 / Qwen3.5 / Gemma 3) | per state, ms (Qwen3) |
|---|---|---|---|---|---|
| a phone number, `[0-9]{3}-[0-9]{4}` | 9 | 2.4 / 0.4–4.7 | 9 | 1 / 0 / 0 | 0.11 |
| yes or no, ` ?(?:yes\|no)` | 6 | 2.2 / 0.4–2.1 | 6 | 0 / 0 / 0 | 0.07 |
| a JSON string | 15 | 7.8 / 2.0–3.0 | 15 | 10 / 3 / 3 | 0.70 |
| a JSON number | 9 | 2.3 / 0.6–1.2 | 9 | 0 / 0 / 0 | 0.02 |
| a JSON object (name, age, up to 5 tags) | 132 | 75 / 34–42 | 132 | 41 / 17 / 17 | 0.31 |
| tool calls, prose and blocks (GenMLX's qwen3_xml shape) | 117 | 39 / 27–30 | 117 | 5 / 8 / 8 | 0.05 |
| any text, `.*` | 8 | 1.6 / 1.0–1.1 | 8 | 2 / 3 / 3 | 0.20 |

(Qwen3's run is the first, the JIT still warming: its DFA and mask times
are the highest.)

- **A state's mask** costs its sweep: from a permissive state (`.*`'s
  start, nearly every token allowed) 1.15 ms at Qwen3's vocabulary and 2.1
  to 2.2 ms at the 250,000-token ones; from a restrictive one (a digit, 10
  or 20 tokens allowed) 0.8 to 1.1 µs, since the sweep skips every dead
  subtree in one jump. Nothing is allocated per token; a mask is a 19 KB
  bitset at Qwen3's vocabulary.
- **For comparison**, GenMLX's naive sweep took about eight minutes for a
  773-state tool-call grammar at 151,000 tokens, and its fix (bucketing
  tokens by first character, genmlx-4tcd) about a hundredth of that.
- **Per token, decoding:** advancing a grammar's state by a token, 5 ns; the
  state of a 12-token context from the start, 60 ns.
- **A site under a grammar**, µs a token (the toy engine at each
  vocabulary, its rows of the GGUF's size):

  | | Qwen3 (151,936) | Qwen3.5 (248,320) | Gemma 3 (262,144) |
  |---|---|---|---|
  | `constrained/weighted`, permissive (`[^"]*`: all but about 1,500 allowed) | 165 | 298 | 275 |
  | the grammar's dense `logw-next` row there | 103 | 161 | 175 |
  | `constrained/weighted`, restrictive (`[0-9]*`: 10 or 20 allowed) | 255 | 459 | 445 |
  | the grammar's dense `logw-next` row there | 17 | 28 | 29 |

  A permissive constraint's normalizer is 1 less the disallowed mass (an
  exp for each disallowed token: before that, about 880 µs at Qwen3's
  vocabulary on a loaded machine, exp-bound); a restrictive one's is the
  allowed tokens' sum, over passes that each read the whole row (four,
  against the permissive three), so it costs more. Soft weights (a general
  potential) cost an exp a token. So a constrained token costs 270 µs
  (Qwen3) to 490 µs (the 250,000-token vocabularies) beyond its draw,
  against a llama.cpp decode of 5.8 ms (M2 Max, one branch, Qwen3): 5 to
  8 %. (The bench's "what the grammar adds" row compares
  simulate's per-token time with and without the grammar; the plain
  completion makes one more LM call a token, its score, so that row is
  not the grammar's cost and is not quoted.)
- **Where the rest would come from:** the passes are O(vocabulary) even
  where a grammar allows ten tokens, because `logw-next` is a dense row; a
  sparse form of it (Decisions, 10) would make a restrictive site cost its
  allowed tokens.

## Grammars as potentials

A grammar implements `gen.llm.potential`'s protocol, after genlm-control's
`Potential` (`BoolFSA` coerced to the LM's tokens): `prefix` is log 1 when
the state after the context is live and -Inf otherwise, `complete` log 1
when it accepts, and `logw-next` the mask, a new float row of n-vocab + 1
entries (0 for an allowed token, the end last). The potential contract's
check (`gen.llm.potential-contract`) holds exactly; so does the
autoregressive factorization (complete(x) - prefix([]) is the sum of the
next-token weights along x and the end's).

## The middleware: a completion under a potential

`(token/completion m {:potential p})` is the completion whose token i has
the distribution

> q_i(t) = p(t | prompt, x_<i) · exp(w_i(t)) / Z_i,  Z_i = Σ_t p(t | …) exp(w_i(t)),

with w_i the potential's next-token weights after x_<i (an end token takes
the end's weight, the potential's weight of x_<i as a whole sequence).
Under a grammar, w_i is 0 for the tokens the grammar allows and -Inf for
the others: the LM's row masked and renormalized, GenMLX's
`wrap-grammar`. Every operation is the completion's own (gen.llm.token's
one set of transitions, rule 1); the potential changes each site's
distribution, as GenMLX's middleware changed each categorical site's. So
the bean's "handler middleware" is an option of the completion. The token
GF is not a dynamic gen function with a handler; its transitions are
written once, and a potential reweighs them all, simulate to regenerate.
`gen.llm.constrained` holds the site's semantics: the weighting, the
scores of a run of known tokens, the draw. `gen.llm.token` calls it at
its two sites (a run of known tokens, a fresh token), its only change.

**Which distribution it is.** q(x) = Π_i q_i(x_i), the **locally
renormalized** distribution. It is not the LM conditioned on the
potential, π(x) ∝ p(x) exp(w(x)), unless Π_i Z_i is the same for every x:
renormalizing token by token favors sequences whose early tokens leave
little of the LM's mass disallowed (the test `local-is-not-global` shows a
gap of more than 0.02 in a completion's probability on the toy engine).
**Recommendation: the constrained completion's GFI weights are q's**: its
score is log q(x), generate and assess weigh by q, update and regenerate
by q's ratios, and every GFI law holds for q (tested). For π, the
completion is a proposal, and `constrained/conditional-weight` is the
importance weight

> log p(x) + w(x) - log q(x) = prefix([]) + Σ_i log Z_i (+ complete(x) - prefix(x) when x stopped at max-tokens),

with p the unconstrained completion's density (the LM's ledger, as its
score, so the weight is exact for that target whatever the engine's
noise) and w(x) the potential's weight of x's tokens before its end token.
Its mean over q is Z = Σ p(x) exp(w(x)); for a grammar, the probability
that the LM's completion satisfies it (tested exactly and in the mean).
Importance sampling with these weights, or token SMC's `:masked` proposal
(gen.llm.smc, whose steps add the same log Z_i), targets π; SMC runs over
the plain completion, the potential given as its `:potential` (it refuses
a constrained completion, `:constrained-completion`). A completion
that stops at max-tokens is a prefix of the language, not necessarily one
of its strings; π weighs it by `complete`, so incomplete ones get weight 0.

**Scores.** A site's score is log q_i(x_i), computed from the LM's row at
its context (the normalizer needs every token's probability, so a run of
k known tokens takes k rows, read 32 contexts at a time, each row reduced
to its score before the next read, where the unconstrained completion
reads one teacher-forced sequence; at Qwen3's vocabulary the rows of 500
tokens at once would be 304 MB). Within
one LM the rows are held, so the laws hold exactly, also on the noisy
cell engine; after an evicted row is read again on llama.cpp, within the
engine's noise, as the token GF's scores.

**Weight 0.** A constraint (or a token update or regenerate keeps) that the
potential gives weight 0 scores -Inf: the trace has density 0 and an MH
move to it is rejected. After it no sequence has weight, so the
potential's weights are undefined and the sites are the LM's rows (drawn
from and scored by them): every site keeps a distribution, and update's
weight (the score change less the fresh tokens' scores) stays -Inf rather
than NaN. A context of positive weight where the potential allows no
token (a grammar complete there, with no end token to end the completion,
`:end-tokens #{}`) is a loud `:zero-weight`.

## A text's probability over its tokenizations (`gen.llm.bytes`)

GenMLX's `bytes.cljs` generated byte by byte through a trie, committing a
token greedily (longest match) or eagerly (shortest): one tokenization, so
its byte probabilities were the probability of that tokenization, not of
the text. Here a text's probability marginalizes over every tokenization:

```clojure
(def b (bytes/byte-lm m))                    ; opts :end-tokens, :beam
(bytes/prefix-logprob b prompt "Stockholm")  ; log P(the completion's text begins with it)
(bytes/text-logprob b prompt "Stockholm")    ; log P(the text is it, then an end token)
(bytes/next-byte-logprobs b prompt "Stock")  ; 256 bytes and the end, a distribution
```

A tokenization of a text that begins with b is a run of tokens spelling
b's first j bytes (j a token boundary inside b) and a token that begins
with the rest: P(b) = Σ over runs of p(run) · Σ_{t begins with b[j:]}
p(t | run), the inner sum one contiguous run of the trie's tokens. The runs
are walked boundary by boundary, each boundary's contexts read in one
`lm/batch-next-logprobs`. Exact, it enumerates every tokenization of every
prefix of b, a number that can grow exponentially with b's length (each
needs its own read: the LM is not Markov); with `:beam k` it keeps the k
most probable runs at each boundary, a lower bound (genlm-bytes' beam).
Tokens with no bytes are never in a text (they would give a text
infinitely many tokenizations); end tokens end it.

**For later (not built):** a byte-level generative function (a byte a
choice, each drawn from `next-byte-logprobs`), whose score of a text is
log P(text) exactly, by the chain rule of these prefix probabilities. It
fits cleanly over this: its state is the runs at the boundaries inside the
text so far, carried from byte to byte (each byte prunes the runs whose
next token disagrees, and adds the runs ending at it), so a byte costs one
batch read at most. Under a grammar, each byte would be masked by one DFA
transition, with no sweep of the vocabulary; the price is a read per run
at each token boundary rather than one per token.

## Schema → grammar (design, for genclj-w120)

GenMLX's `schema_grammar.cljs` compiled a malli schema to a regex *string*
over the value's canonical EDN, and its `structured.cljs` generated it byte
by byte; `toolcall.cljs` built a regex over the declared tools. Here:

- **Schemas compile to terms, not strings.** `(schema/term schema {:format
  :json})` returns a `gen.llm.regex` term, so literals need no escaping
  (GenMLX's `re-quote`, and the bugs its escaping invited), and `[:and …]`
  is `regex/intersect` of its children's terms, not "the first child, the
  rest validated after".
- **Formats:** JSON (what chat models are trained to emit for tools and
  structured output) and EDN (GenMLX's). One canonical serialization each:
  fixed key order, one space where the format needs a separator, no
  optional whitespace. Canonical keeps value and string a bijection, so a
  value's text, and with it its probability, is unique; a grammar that
  admits whitespace variants defines a distribution over strings, which
  only a byte-level marginal over the variants would turn into one over
  values.
- **The leaves:** `:int` (no leading zeros, `-0` excluded), `:double`,
  `:string` (any character but the quote and backslash, or JSON's escapes;
  `:min`/`:max` lengths as `repeat`, counted in characters since a
  character is one term), `:boolean`, `:nil`, `:keyword`, `[:enum …]` and
  `[:= v]` (alternations of their canonical texts), `[:re r]` (the regex,
  between quotes, intersected with the string's characters), `:uuid`.
- **The composites:** `[:map …]` closed, in the schema's key order, an
  optional key an `opt` of its entry (linear, not 2^k alternatives),
  `[:vector X]` and `[:sequential X]` with `:min`/`:max`, `[:tuple …]`,
  `[:maybe X]`, `[:or …]` (`alt`), `[:and …]` (`intersect`).
- **Recursive schemas are context-free** (a registry reference to itself:
  trees, nested JSON). A regular term can only unroll them to a fixed
  depth, a loud `:max-depth` option; the exact form needs a context-free
  potential (below).
- **Tool calls** (`toolcall.cljs`'s qwen3_xml dialect) are a term over the
  declared tools: prose (any text without `<`), then blocks whose names
  are the declared ones and whose values are their schemas' terms;
  "text without the closing tag" is an intersection with a complement
  (above), where GenMLX needed its hybrid masker.
- **The generative function** of structured output is a completion under
  the schema's grammar whose return value is the parsed value
  (`gen.llm.token`'s tokens, read by the format's reader), so its choices
  stay tokens and every law the completion satisfies holds. Its density
  over values is over the tokenizations of the canonical text; assess of
  a value (not of tokens) needs the byte-level marginal under the grammar.
- **Context-free and reader potentials** (`codegen.cljs`'s reader as
  grammar: complete, incomplete or invalid). Any prefix oracle is a
  potential, `(potential/potential n-vocab {:prefix … :complete …})`, but
  its dense next-token weights cost a reader call per token. The cheaper
  form is the sweep with the oracle in place of the DFA: a call per trie
  node visited, each node's verdict shared by every token under it, and a
  dead node's subtree skipped. For context-free grammars proper, an Earley
  or pushdown recognizer whose state advances byte by byte fits the sweep
  directly (its state at each trie depth, as the DFA's is), which is how
  llguidance and XGrammar make CFG masks cheap. Neither is built; both are
  potentials, so nothing above them changes.


## The dependency: our own engine, not instaparse

**Recommendation: our own regex → DFA engine** (built), and no instaparse.
DESIGN §8 left instaparse to decide in M3 for the grammar module. The
evidence:

- **What masks need.** A mask asks, for every token, whether the text
  with the token appended is still a prefix of some string of the
  language. A DFA answers by one transition per byte, and the trie shares
  those transitions between tokens: a whole mask is one sweep. instaparse
  is a parser for context-free grammars (GLL): it answers whether a whole
  string parses, and a failure reports where it stopped (a failure at the
  end of the input marks a viable prefix), but it has no parser state to
  carry from one token to the next, so a token's viability is a reparse of
  the text with the token appended. A mask would be a parse per token
  (151,936 at Qwen3's vocabulary) or per trie node visited, each from the
  start of the text, against one table read per trie node.
- **What GenMLX used it for.** Only to parse the regex syntax
  (`grammar.cljs`'s 25-line instaparse grammar); the engine itself
  (Thompson, subset construction, masks) was its own code. That parse is
  a 150-line recursive descent here, with loud errors for every
  unsupported construct.
- **Correctness.** The engine is tested against java.util.regex on random
  regexes over one- to four-byte characters, its UTF-8 lowering
  exhaustively over every one- and two-byte string, its minimality
  pairwise, and its masks against brute force (Tests). GenMLX's grammar
  bugs were in its own code too (ASCII-only alphabet, anchors as literals,
  class escapes, `a{0}`, the end token walked through the DFA, an
  eight-minute mask), none of them instaparse's to fix.
- **Performance.** A JSON object grammar's 132 states: the DFA in 34–75 ms,
  every mask at Qwen3's vocabulary in 41 ms; a tool-call grammar of 117
  states, its masks in 5 ms (GenMLX's naive sweep took about eight minutes
  for a 773-state tool-call grammar, genmlx-4tcd, and about a hundredth of
  that after its fix). Measurements below.
- **Size and maintenance.** About 1,300 lines of our own code (regex, dfa,
  bytes, grammar) against a dependency we would use a sliver of; rule 8
  prefers none.
- **Context-free grammars later** need an incremental recognizer whose
  state advances byte by byte (Earley, a pushdown automaton), which
  instaparse does not expose either; it would be a potential of its own
  (Schema → grammar, above).

So DESIGN §8's row becomes: instaparse, **no** (confirmed by the owner,
2026-10-21).


## Decisions

1. **Recommendation: our own regex → DFA engine, no instaparse** (The
   dependency, above). DESIGN §8's instaparse row becomes "no".
2. **Recommendation: the middleware is the completion's `:potential`
   option**, `(token/completion m {:potential p})`, with the site's
   semantics in `gen.llm.constrained`: one set of transitions, every
   potential (a grammar, a product, a weighted one) through the same two
   sites. `constrained/weighted` is the one definition of "the row times
   the twist, renormalized", for token SMC's `:masked` proposal too (to
   replace `gen.llm.smc`'s private `masked`). Alternatives: a
   `(token/constrain completion p)` wrapper (the same, a second way), or a
   distribution transformer at a `trace` site (for gen bodies drawing
   tokens one at a time with `token/dist`; possible later, as
   `(constrained/dist row ws end-tokens)`).
3. **Recommendation: a constrained completion's GFI weights are the local
   distribution q's**, and `constrained/conditional-weight` gives the
   importance weight for π, the LM conditioned on the potential (above).
   The alternative, a completion whose density is π, has no tractable
   score (Z is a sum over every sequence).
4. **Recommendation: what a grammar holds** is the text, the tokens' bytes
   concatenated (the LM's decode: special tokens by their rendered bytes,
   as GenMLX's tool calls relied on), read byte by byte; end tokens are not
   text (the end is allowed where the text is complete); a token with no
   bytes is allowed wherever the text may go on (it changes no text), as
   genlm's coerced potentials have it.
5. **Recommendation: exact masks, made eagerly.** Every state the tokens
   reach is explored when the grammar is made (5 to 41 ms for the grammars
   measured at Qwen3's vocabulary), and a token is allowed only into a
   state tokens can complete. The alternative, masks made lazily per state
   from byte-level liveness, is genlm's and GenMLX's, and lets a
   completion strand where its vocabulary cannot go on.
6. **Recommendation: the regex syntax** is java.util.regex's on what it
   accepts (`.` excludes only `\n`; `\d \w \s` are ASCII), loud on the
   rest, and matching is whole-string; `complement` is over byte strings.
7. **Recommendation: weight 0** (above): a token of weight 0 scores -Inf,
   the sites after it are the LM's rows; no token where the context has
   weight is a loud `:zero-weight`.
8. **Recommendation: byte-level marginals** are exact by default, `:beam`
   for a lower bound; tokens with no bytes are never in a text; end tokens
   end it.
9. **Recommendation: the names**: `gen.llm.regex` (`parse`, `dfa`,
   `matches?`, and the terms `nothing`, `epsilon`, `text`, `chars`,
   `chars-except`, `any-char`, `bytes-of`, `any-byte`, `cat`, `alt`, `star`,
   `plus`, `opt`, `repeat`, `intersect`, `complement`), `gen.llm.dfa`,
   `gen.llm.bytes` (`byte-trie`, `sweep`, `byte-lm`, `prefix-logprob`,
   `text-logprob`, `next-byte-logprobs`), `gen.llm.grammar` (`grammar`,
   `state`, `advance`, `live?`, `accepting?`, `allowed?`, `allowed`,
   `automaton`, `info`), `gen.llm.constrained` (`weighted`,
   `conditional-weight`), and the option `:potential`.
10. **For the potential protocol (to its author):** `logw-next` is a dense
    row, so a constrained token costs O(vocabulary) passes even where a
    grammar allows ten tokens (Measurements). A sparse form (the allowed
    tokens and their weights, or a bitset for a 0/-Inf potential) would
    let `weighted` touch only those; worth adding when a profile asks.


## Tests

All on the pure JVM, on the toy engines (exact distributions) and toy
tokenizers; `test/gen/llm/byte_tokenizer.clj` is a tokenizer of arbitrary
byte pieces, for vocabularies whose tokens span several characters or
split one.

- `gen.llm.regex-test`. **The syntax against java.util.regex**
  (UNIX_LINES): 300 random regexes (characters of one to four UTF-8 bytes,
  classes with ranges and escapes, negated ones, `.`, groups, alternation,
  every quantifier), each matched against 20 random strings, 10 members
  drawn from its own tree (an oracle independent of both engines) and 10
  strings its DFA accepts by a random walk. **The constructors** build the
  same DFAs as the syntax (200 trees). **UTF-8:** `(chars [lo hi])` over
  random ranges accepts exactly the encodings of its characters, checked
  over every string of one and two bytes, 300 random ones of three and
  four, and the code points where UTF-8's lengths change; surrogates,
  overlong encodings and code points past U+10FFFF are refused.
  **intersect and complement** over random byte terms. **Minimality**: every
  two states of 100 random DFAs are distinguished by some string;
  **canonical**: equal languages give equal DFAs; transition maps keep
  their language. **Misuse**: 31 malformed or unsupported regexes, bad
  terms, bad maps, `:max-states`.
- `gen.llm.grammar-test`. **The trie** against the vocabulary (each node's
  tokens, its subtree's, its children). **Masks against brute force**: 300
  random byte languages with random vocabularies (empty tokens included)
  and 300 random languages of characters with vocabularies of UTF-8 pieces
  that split them; brute force walks every token from every state and
  finds the live states by a fixed point (no trie, no sweep). About a
  sixth of the text cases reach states that bytes can complete and tokens
  cannot. **The potential contract** (`gen.llm.potential-contract`)
  exactly, and the autoregressive factorization, along random constrained
  walks. Masks over vocabularies of more than 64 tokens (every word of the
  bitset). **Misuse**: an empty language, an inexpressible one, bad
  vocabularies.
- `gen.llm.bytes-test`. **Exact against brute force**: prefix and text
  probabilities of every text of up to four bytes over a vocabulary with
  overlapping tokens and é whole and in halves, after two prompts, against
  a recursion over the vocabulary that compares bytes as strings. **The
  completion's own frequencies** (gen.llm.token, 4,000 keys) of texts
  beginning with five strings, within 5 standard errors. `next-byte-logprobs`
  is the prefixes' ratios and sums to 1 (within the rows' float rounding).
  A beam's lower bounds rise to the exact value, and a beam of one keeps
  the more probable run (the mass it misses is the other's).
- `gen.llm.constrained-test`. **The GFI laws** (general and forward-sampling,
  100 trials each) over a completion under a grammar of single-character
  tokens, one of fixed length, one of multi-byte tokens splitting a
  character, one of free text, a gen model splicing one with an
  observation, and a Map of them; on the noisy cell engine (40 trials):
  exact. **The laws across generative functions** over traces revised from
  a constrained completion to a plain one, back, and to another grammar.
  Not the fixed-support law: under a grammar a token's support depends on
  the tokens before it, and the law fails at its first trial, as it
  should. **The site's distribution** against its definition (300 random
  rows and weights: constraints disallowing few or many tokens, and soft
  weights). **In the language**: 300 completions
  each, every text a prefix of the language's, every ended one a string of
  it. **The density exactly**: every completion of a small grammar on the
  multi-byte vocabulary enumerated (more than 40), its density computed
  independently (the toy model's rows, masks by brute force), against
  assess within float rounding; the densities sum to 1; simulate's
  frequencies over 4,000 keys within 5 standard errors. **Local is not
  global**: the enumerated π and q differ by more than 0.02 somewhere; the
  conditional weights equal log p + w - log q for every completion, and
  their mean over 4,000 traces is Z within 5 standard errors. generate's
  weights, weight 0 and what follows it, MH within the language, batched
  fallback. **Misuse**: not a potential, another vocabulary's, a potential
  of no weight, a grammar complete with no end token to end it.
- `gen.llm.token-test` (62 tests) passes unchanged: the unconstrained
  completion is the same code path.

### Mutations

Each applied in a worktree of its own, the namespace's suite run, and
reverted (2026-10-21, `scratchpad/grammar-mut/run.py`): 42 planted bugs.
38 were caught by failing tests; three were missed at first and are
caught by tests added for them (marked *); one first showed as a stall
(marked †): the UTF-8 property decoded 65,536 strings in every trial and
shrank vectors of 300 of them, so its failing case shrank for minutes.
It now decodes them once and generates three integers (its random strings
drawn from a seed inside the trial), and the mutated suite fails in 4 s.

| Mutation | Caught by |
|---|---|
| UTF-8: surrogates kept | every-character-and-nothing-else |
| UTF-8: the lead byte of two-byte characters wrong | †chars-accepts-exactly-the-utf8-of-its-characters, the-syntax-agrees…, every-character-and-nothing-else, text-not-containing-a-string, within 4 s |
| derivative of a concatenation ignores a nullable head | the-syntax-agrees-with-java-util-regex, the-constructors-agree…, equal-languages…, text-not-containing-a-string (6 failures) |
| `.` matches a newline | the-syntax-agrees-with-java-util-regex, the-constructors-agree… |
| `\s` without `\x0B` | the same two |
| a negated class not negated | the same two |
| a counted repeat never counts down | the-syntax-agrees…, dfas-are-minimal, equal-languages… (5) |
| alternatives of single bytes merged into the first | chars-accepts-exactly…, the-syntax-agrees…, equal-languages… (5) |
| `(x{0,b}){0,d}` read as `x{0,b+d}` | nullable-repeats-normalize |
| byte classes drop the bytes outside a set | 20 failures and errors across the regex suite |
| a lazy `?` read as a quantifier | lazy-quantifiers-are-the-same-language |
| Hopcroft: a split of a waiting block adds the smaller half only | the-syntax-agrees…, the-constructors-agree… |
| no minimization | dfas-are-minimal, equal-languages…, a-transition-map-keeps-its-language… (10) |
| the empty language keeps a dead start | every-state-is-live-and-reached |
| a block accepts as the first state | 9 failures across the regex suite |
| trie: one byte too few in common | the-trie-holds-the-vocabulary |
| sweep: a dead subtree entered | byte- and text-masks-match-brute-force, the-weights-are-consistent |
| sweep: tokens with no bytes never allowed | byte-masks-match-brute-force |
| grammar: tokens into states tokens cannot complete allowed | a-token-into-a-state-tokens-cannot-complete-is-not-allowed, both masks properties, the-weights-are-consistent |
| grammar: only the start explored | the masks properties, a-dead-context-has-no-next-weights… (5) |
| logw-next: the end allowed in live states | the-weights-are-consistent |
| allowed?: the wrong bit (t & 31) | *masks-over-a-vocabulary-of-many-words (every random vocabulary had fewer than 32 tokens) |
| log-mass: the tokens at a node, not under it | marginals-equal-brute-force, the-marginal-is-the-completions, the-next-byte… (44) |
| marginals: a text ends where it is a prefix | marginals-equal-brute-force, the-next-byte… (41) |
| marginals: end tokens counted as text | end-tokens-hold-no-text, the-next-byte… (39) |
| marginals: the beam keeps the least probable | *a-beam-keeps-the-most-probable-runs |
| weighted: an end token weighs as its own token, not the end | completions-never-leave-the-language and the laws (37) |
| weighted: not renormalized | the laws, the exact densities (73) |
| the normalizer's complement forgets the disallowed mass | a-permissive-grammar-scores-exactly, the laws (42) |
| soft weights read as a constraint's | weighted-is-the-renormalized-product |
| the disallowed mass takes end tokens by their own entries | weighted-is-the-renormalized-product, the-density-is-the-local-one, local-is-not-global |
| draw-next: drawn from the LM's row, scored by the site's | completions-never-leave-the-language, the laws (25) |
| run-scores: the weights one token late | the laws (34) |
| draw-next: -Inf after a token of weight 0 | the update weight laws (NaN) |
| run-scores: -Inf after a token of weight 0 | *generate-weighs-the-forced-tokens-by-their-sites (no GFI weight sees it: the trace's scores do) |
| dead?: only the last token | the laws (11 errors: weights asked of a dead context) |
| conditional-weight without the whole-sequence weight | local-is-not-global |
| conditional-weight with the end token in the text | local-is-not-global |
| check-potential: no vocabulary check | misuse-is-loud |
| token.clj: assess ignores the potential | the laws (14) |
| token.clj: draws ignore the potential | completions-never-leave-the-language, the laws (25) |
| token.clj: a run of known tokens ignores the potential | the laws (28) |

2026-10-22 (genclj-1lpj), each caught: a product accepting factors that
disagree (`potentials-declare-their-end-tokens`); a product declaring no
set, `check-end-tokens` never refusing, a grammar with no opinion (those,
`an-end-token-is-the-end-not-text`, `misuse-is-loud` and smc's
`the-potential-s-end-tokens-are-the-completion-s`); `potential` dropping
`:end-tokens`; `token/completion` or token SMC calling no end-token check.


