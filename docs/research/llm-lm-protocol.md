# The LM protocol

*2026-10-20, bean genclj-mse0 (M3's second). Built by an agent on branch
m3-lm, over the Branching protocol (`docs/research/llm-branching.md`). The
choices that are the owner's to make (names, the value shape, the eviction
policy, an autobatcher, and smaller ones), marked **Approved**, were all
approved by the owner the same night, with the token generative function's
foundations (score once, the trie owns the branches, host draws; DESIGN.md
§9). Summarized in DESIGN.md §6 and §9.*

## What exists

| Piece | Where |
|---|---|
| The LM protocol and its API; the LM over a Branching engine | `gen.llm.lm` |
| The cache under it: content addressing, reads, batching, room, eviction | `gen.llm.cache` |
| A persistent radix trie of token sequences (scores on paths, entries at nodes) | `gen.llm.trie` |
| The KV-cell ledger (llama.cpp's unified-cache accounting) | `gen.llm.cells` |
| The GGUF's tokenizer as the LM's (additive) | `gen.llm.llama/tokenizer` |
| A count of GC backstop collections in `(br/info engine)` (`:collections`) | `gen.llm.slots` (four lines) |
| The contract as engine-agnostic checks; a llama.cpp-faithful toy engine; a toy tokenizer; tests | `test/gen/llm/` (`lm_contract`, `cell_engine`, `toy_tokenizer`, `lm_test`, `trie_test`, `llama_lm_test`) |
| Costs, hit rates, allocation, the value shape | `bench/gen/lm_bench.clj` |

Nothing in L0–L5 requires `gen.llm.*`; `gen.llm.lm` requires only
`gen.llm.branching` and its own namespaces, never an engine.

## The API

```clojure
(require '[gen.llm.lm :as lm] '[gen.llm.llama :as llama])

(with-open [e (llama/engine {:model "Qwen3-0.6B-Q8_0.gguf" :max-branches 64 :n-ctx 8192})
            m (lm/lm e (llama/tokenizer e))]
  (let [prompt (lm/encode m "The capital of Sweden is")
        lp (lm/next-logprobs m prompt)                         ; float[] over the vocabulary
        rows (lm/batch-next-logprobs m [(conj prompt 1) (conj prompt 2)])  ; one advance
        s (lm/score m prompt (lm/encode m " Stockholm" {:add-special? false}))]
    (lm/info m)))                                              ; sizes and counts
```

| Function | Meaning |
|---|---|
| `(lm engine tokenizer)`, `(lm engine tokenizer opts)` | an LM over the engine's branches (`java.io.Closeable`; closing disposes of them, the engine stays the caller's); opts `:max-branches`, `:max-cells` |
| `(vocab m)` | `{:n-vocab :bos :eos :eog :add-bos?}` (`:eog` a set) |
| `(encode m text)`, `(encode m text opts)`, `(decode m tokens)` | the model's tokenizer (llama.cpp's opts: `:add-special?`, `:parse-special?`) |
| `(token-bytes m t)`, `(byte-vocab m)` | a token's raw bytes; every token's (genlm's `byte_vocab`), new arrays |
| `(next-logprobs m context)` | log p(t \| context) for every token t: a new float array |
| `(batch-next-logprobs m contexts)` | the same for each context, read together (below) |
| `(score m context continuation)` | [log p(c_i \| context, c_<i)], doubles; `[]` for an empty continuation |
| `(info m)` | `:live :cells :tokens :max-branches :max-cells :limit :closed?` and since creation `:requests :hits :reads :read-tokens :forks :evictions`; `:limit` is the most tokens a context, or a context and its scored continuation, may hold |

`LM` is a protocol of four methods (`-tokenizer`, `-next-logprobs` over a
vector of contexts, `-score`, `-info`): the LM over an engine is one
implementation, and the token GF's tests can implement it with an exact
table (GenMLX's `TableDecoder`, enumerable posteriors, no engine).

A **tokenizer** is a map: `:n-vocab :bos :eos :eog :add-bos?` and the
functions `:token-bytes`, `:encode`, `:decode`. `llama/tokenizer` reads
every token's bytes once, in one engine call (the calls inside it run on
the engine's thread with no handoff), so the byte vocabulary costs no round
trip per token. `lm` checks that the tokenizer's vocabulary is the engine's.

## How it works

**Callers hold tokens; the cache holds branches.** A context is a token
vector. The cache (`gen.llm.cache`) keeps a trie (`gen.llm.trie`) of what
the engine has read, in which each token on a path keeps its **score**, the
log-probability it was read with, and a node marks each context the cache
holds a live branch for: its **entry** (the branch, its cells' segment, its
last use), kept in a map from an id to the entry. Every live branch is an
entry's. A context is answered in one of four ways, by the deepest entry at
a prefix of it:

| The deepest entry | Read |
|---|---|
| at the context (a **hit**) | nothing: the entry's log-probabilities, copied |
| at the context less its last token | one token, gathered with the call's other one-token reads into **one `advance`** |
| at a shorter prefix | a fork of the entry, `extend`ed teacher-forced by the rest |
| none | the first token (`open`), then the rest teacher-forced (`extend`), so that every token after the first is scored |

The new branch becomes the entry at its context. A call answers its
contexts in order (a context asked again is a hit, and two one-token
reads of one context in a round are one): hits and longer reads at once,
one-token reads gathered; then one advance of those that fit (all of them,
unless more than `:max-branches` or the cells say otherwise, in which case
the longest fitting prefix, halving; the rest wait for the next round,
which also takes any whose entry a read evicted). A token SMC step (each
particle one token past its held context) is one advance; resampling needs
no operation, since a context asked twice is read once and one never asked
again ages out.

`score` reads nothing when the sequence lies along what was read (every
token on a path, and the token after a held context from its entry's
log-probabilities); otherwise it reads the whole sequence like
`next-logprobs` and keeps it. So a token SMC's prefix potential of a
particle (`score` of its context given the prompt), assess after simulate
in the same LM, and genlm-control's `complete` (a score, then the EOS
log-probabilities at the end) read nothing more.

**Bounds.** The cache holds at most `:max-branches` live branches (default:
every branch the engine has free) and `:max-cells` KV cells (default: the
engine's `:n-ctx`, all of llama.cpp's unified cache). Cells are counted as
llama.cpp counts them (`gen.llm.cells`, checked against llama.cpp's
`find_slot`, which takes any free cells, not a contiguous run): a read of n
tokens takes n new cells, a fork shares its source's, and a cell is freed
when no branch holds it. The ledger keeps a segment per read; a branch's
state is a segment, held by the branches on it and the segments read after
it. The trie keeps a path only while it leads to an entry, so it holds no
more tokens than the cells. A context longer than `min(:n-ctx,
:max-cells)` is a loud `:context-full`.

**Room and eviction.** Before a read, the cache evicts until the read's
branches and cells fit: the entry used longest ago, or, when cells are
short, the one used longest ago among those whose disposal frees cells (an
ancestor of a held branch frees none). It chooses the victims on its books,
then disposes of them in one `dispose!`. The read's sources are pinned. A
read forks its source so the source stays held; when branches are short
even after eviction, it consumes the source instead (`extend` or `advance`
of the entry's own branch). Uses (hits, sources, new entries)
advance a tick; the least recent tick is evicted.

**Failures.** Every check runs before the engine is touched (each token
against the vocabulary, each context's length), so misuse changes
nothing. If an engine call fails, the forks made for it are disposed and the
entries whose branches the engine no longer holds (a lost slot, a closed
engine) are forgotten; the books then match the engine's again.

**Threads.** An LM is synchronous (DESIGN §8); its calls are serialized by
a lock, in arrival order, and run the engine's operations, which are
serialized on the engine's thread.

## Determinism

The cache's choices (what it reads, in which batch, from which source,
what it evicts and consumes) are a function of the sequence of calls: ticks
are unique, the maps it walks are keyed by tokens and entry ids, and maps
keyed by branches (which hash by identity) are only ever looked up, never
walked. No branch is ever dropped, so the engine's phantom queue and
GC backstop never run, and nothing depends on when the GC runs. llama.cpp
computes a read from the same state bit-identically, so **the same calls on
a new engine give the same results, bit for bit**, whatever threads make
them, as long as the calls arrive in the same order.

What is not guaranteed: concurrent callers racing on one LM arrive in an
order the scheduler picks, and on llama.cpp a context's log-probabilities
depend on how it was read (the batch shape, where its cells land; up to
5e-2 nats, `llm-branching.md`). A program that needs bit-identical reruns
makes its LM calls from one thread, or in a fixed order, and batches
explicitly.

A context read twice (after an eviction) may differ within that noise, and
a context's log-probabilities and the stored scores of the tokens after it
may come from different reads. The scores on a path are the first read's
(**score once**, as GenMLX's ledger): the cache never overwrites them.

## Decisions

1. **Approved (owner, 2026-10-20): the names.** `gen.llm.lm` (alias `lm`), `lm/lm`,
   `vocab`, `encode`, `decode`, `token-bytes`, `byte-vocab`,
   `next-logprobs`, `batch-next-logprobs`, `score`, `info`: DESIGN §6's
   list, genlm's `next_token_logprobs` shortened as §6 has it.
   Alternatives: `gen.llm.model`, `next-token-logprobs`.
2. **Approved (owner, 2026-10-20): the value shape: host float arrays, a new one per
   call.** The engine's form; every consumer below the token site reads it
   on the host (byte marginals over the token-byte trie, grammar masks, SMC
   weights, top-k), and the LM layer stays below gen.array as the engine
   layer does (DESIGN §9). The engine normalizes every row on the host
   already (1.25 ms a row, now on a pool of its own: the Branching review's
   N1), so an array on MLX's device would add a transfer on top. Measured at
   Qwen3's 151,936 tokens (tables below): the copy costs 17 µs, a conversion
   to a JVM-backend array 105 µs, against a llama.cpp decode of 5.8 ms (M2
   Max). The copy is a humongous object below a 4 GB heap (N11), reclaimed
   eagerly: 5,000 hits at `-Xmx2g` cost 14 ms of GC. The token GF converts
   at its site. Alternatives: (a) a shared read-only view, no copy (saves 17 µs
   and the row's 600 KB, most of the caller's allocation per token, with
   values still safe from mutation: a `FloatBuffer.asReadOnlyBuffer` or a
   small immutable type, which every host consumer then reads through); (b) gen.array arrays on the current backend (the
   conversion moves into the LM; every host consumer then pays for it).
   **For genclj-0cvv:** sampling the site through gen.dist's
   `categorical-logits` on the JVM backend costs 4.6 ms (Gumbel-max: a
   threefry draw and two logs per token of the vocabulary) and its
   log-probability 1.0 ms (a log-softmax), together about one decode; a
   host draw from the float row with one gen.random uniform (inverse CDF)
   costs 78 µs, and the row is already normalized, so its log-probability
   is a read. Worth a token distribution that does this (or a cheaper
   `categorical-logits` sampler for long rows), and `from-host` taking
   float arrays.
3. **Approved (owner, 2026-10-20): the eviction policy.** LRU for branches; for cells,
   LRU among the branches whose disposal frees cells; sources pinned and
   forked, consumed only when branches are short. The bounds default to
   the engine's (every free branch, all of `:n-ctx`), so an LM should be its
   engine's only user. Alternatives: plain LRU for both (evicts ancestors
   that free no cells, losing the prompt), advance in place by default
   (fewer forks, no backtracking to a parent), a pinned-prompt API.
4. **Approved (owner, 2026-10-20): no autobatcher.** One that coalesces concurrent
   callers by timing (genlm-backend's `_join_batch` with its timeout) makes
   batch composition, and on llama.cpp the results, depend on thread
   timing: runs would not repeat bit for bit, and the laws would need
   bands where they now hold exactly. It is not built. If concurrent
   batching is wanted (independent chains on threads), a deterministic
   form exists: a **lockstep batcher** for a fixed set of workers (a
   `java.util.concurrent.Phaser`): each round waits for every worker's
   request and batches them in worker order, never arrival order, so
   composition is a function of the program. It costs the slowest worker's
   latency each round and needs workers to register and deregister.
   Recommended as a later bean only if a workload needs it; SMC and batched
   MH call `batch-next-logprobs` directly.
5. **Approved (owner, 2026-10-20): the tokenizer is a map** (data and three functions)
   rather than a protocol: four implementations (llama.cpp, MLX's HF
   tokenizers, the toy, a token GF's table) with three functions each.
6. **Smaller choices** (recommended; the owner may overrule):
   - A fresh context is read as its first token, then the rest
     teacher-forced: two decodes instead of one prefill (on llama.cpp, a
     one-token decode, ~6 ms, once per context the LM holds no prefix of),
     so that every token after the first has a score and the trie needs no
     unknown scores.
   - `score` keeps what it reads (the whole sequence becomes an entry), and
     scores along what was read are a hit, first read wins.
   - Score hits do not count as a use for the LRU order (they read no
     branch); next-logprobs hits and reads do.
   - `(lm engine tokenizer)` borrows the engine; closing the LM disposes of
     its branches, the engine is closed by its owner.

## Tests

`gen.llm.lm-test` (on the toy engines), `gen.llm.trie-test`, and
`gen.llm.llama-lm-test` (skipped without `-J-Dgen.llama.test-model`).

- **Over random programs** (test.check): moves over the contexts asked
  before, as inference asks (children alone or together, prefixes,
  continuations), with random bounds (1 to 6 branches, 14 to 60 cells), on
  the toy engine and the cell engine:
  - *every answer is the model's*, exactly: whatever the cache held, so a
    cached context equals a fresh read and eviction changes costs only; the
    books (bounds, the engine's live count, the cells counted physically by
    the cell engine, the trie's size, no refused read, no backstop) after
    every call;
  - *a batch answers as its contexts one by one*;
  - *the same calls give the same results* on the noisy cell engine (whose
    results depend on the read's shape and cells, as llama.cpp's), and the
    same contexts held after every call;
  - *a failed read leaves the books straight* (planted decode failures).
- **The contract** (`gen.llm.lm-contract`, engine-agnostic: a hit is the
  read, cached = fresh, batch = single, score = steps and from the trie bit
  for bit, the books under an SMC loop, determinism on new engines) holds
  with band 0 on the toy and cell engines; the toy engine's five planted
  Slots bugs each fail their checks.
- **The cache's choices**, one test each: a fresh read, one advance for a
  batch, extending the deepest prefix, scoring from the trie (and the token
  after a held context), keeping a scored sequence, LRU (and a hit as a
  use), evicting what frees cells, consuming when branches are short, a
  batch larger than the branches, repeated contexts; SMC and MH loops under
  tight bounds with no backstop; the backstop count counting; misuse;
  closing; a closed engine; the vocabulary.
- **The trie and the ledger** against naive models (a map of prefixes; sets
  of physical cells), and the trie's compressed structure.

Mutations checked (2026-10-20, each applied, the suites run, reverted):

| Mutation | Caught by |
|---|---|
| eviction of the most recently used | eviction-is-least-recently-used |
| eviction ignoring which branches free cells | short-of-cells-the-cache-evicts-what-frees-them |
| no use recorded for hits and sources | eviction-is-least-recently-used (a hit is a use) |
| an advance's rows assigned in rotation | every-answer-is-the-models and five more |
| a step's score stored one position off | a-continuation-on-a-path-is-scored-from-the-trie, the-same-calls…, and more |
| a step's cell not counted | every-answer-is-the-models (the engine's cells), a-failed-read…, the backstop tests (:kv-full) |
| a read's source not pinned | every property (a fork of a disposed branch) |
| fits? ignoring cells | inference-loops-never-reach-the-backstop (:lm-full) |
| a consumed parent kept as an entry | every-answer-is-the-models, short-of-branches… |
| the consumed parents chosen in a set's (identity-hash) order | the-same-calls-give-the-same-results, every run of four |
| entries the engine freed not forgotten | a-closed-engine-is-loud-and-forgotten |
| a failed read's forks not disposed | a-failed-read-leaves-the-books-straight |
| a batch's answers misassigned | every-answer…, a-batch-answers…, and five more |
| the victims chosen but not disposed of | every property (the engine's branches run out) |
| a round's one-token reads not resolved again after a read in it | every property (a fork of an evicted branch) |
| a round's one-token reads of one context not deduplicated | every property (a second entry at one context, loud) |
| trie: match descending a partly matched edge; insert overwriting scores; leaf scores shifted; no join; dead paths kept | trie-matches-its-model, the-trie-stays-compressed, a-path-keeps-its-first-scores |
| ledger: share not counting; freed counting a shared segment | ledger-matches-the-cells |

The identity-order mutation first slipped through: random contexts almost
never made a batch's parents held, so the choice never arose. The programs
were then generated from moves over the contexts asked before, which is how
inference calls an LM, and the determinism property also compares the held
contexts after each call.

## Measurements (toy engines, this Mac, pure JVM)

`clojure -M:bench -m gen.lm-bench` (criterium means; M4, GraalVM 25,
nice'd, sharing the machine; before the merge of the Branching review,
whose costs on the toy engines are the same). The toy engine's 97-token vocabulary keeps
its own work small, so the first table is the LM's cost over the Branching
operations it makes (each of those is a hop to the engine's thread).

Per call, µs:

| context | hit | one-token read (an eviction at steady state) | fork + advance + dispose alone | 32 one-token reads, one advance | fork 32 + advance + dispose alone | 8 tokens scored from the trie |
|---|---|---|---|---|---|---|
| 10 tokens | 0.7 | 34.0 | 20.9 | 240 | 116 | 0.6 |
| 100 | 1.4 | 35.7 | 21.6 | 287 | 122 | 1.3 |
| 1000 | 7.9 | 50.4 | 26.8 | 737 | 234 | 7.7 |

So the LM adds 13 to 24 µs to a one-token read (its books, and evicting a
branch, one more engine call), and 4 to 16 µs to each read of a batch,
growing with the context: content addressing reads the context, once to
check its tokens and once to find it in the trie (about 9 ns a token
together). Against llama.cpp's 5.8 ms for one branch's advance and 13.5 ms
for 15 (M2 Max, with the review's normalization pool, `llm-branching.md`),
that is under 1% for one branch and a few percent for a batch of long
contexts. Two rounds of
measurement changed the cache: victims are disposed of in one call, the
books are keyed by entry ids instead of contexts (hashing a 1000-token
vector costs 10 µs), and a context's tokens are checked by a loop instead
of a trie walk; the 32-read batch at 1000 tokens went from 1,900 µs to 737.

Allocation on the caller's thread: 5.0 KB for a hit (of which 404 bytes
are the row), 26 KB for a one-token read with an eviction, of which the
Branching operations' own checks are 4.6 KB. The rest is spread over the
persistent updates of the books (no site above 13% in a JFR allocation
profile), the cost of keeping the cache's state as values; at Qwen3's
vocabulary a row is 600 KB, which the engine's thread allocates anyway.

Workloads on the cell engine (llama.cpp's cell accounting):

| workload | bounds | requests | hit rate | reads | tokens read / request | forks | evictions | µs / request | backstop |
|---|---|---|---|---|---|---|---|---|---|
| token SMC, 32 particles × 100 steps, resampled each step | 64 branches, 4096 cells | 3200 | 0.38 | 101 | 0.62 | 1982 | 1919 | 23.0 | 0 |
| the same, short of cells | 64 branches, 1024 cells | 3200 | 0.38 | 101 | 0.62 | 1982 | 1919 | 18.6 | 0 |
| MH over a 50-token trace, 200 regenerations, each scored | 16 branches, 1024 cells | 5342 | 0.04 | 5303 | 1.78 | 4945 | 5108 | 63.6 | 0 |
| the same, 4 branches | 4 branches, 1024 cells | 5342 | 0.04 | 5333 | 1.91 | 4941 | 5133 | 51.2 | 0 |

The SMC is one read a step (101: two for the prompt, its first token and
the rest, then an advance for each of the 99 later steps), its hits the
particles resampling duplicated. MH's next-token requests are new
contexts by construction (a regeneration samples new tokens); its hits are
the scores, all from the trie. A regeneration reads from the deepest held
prefix of its site: 1.8 tokens a request with 16 branches, 1.9 with 4.

next-logprobs' value shape at Qwen3's 151,936 tokens, JVM backend, µs:

| operation | µs |
|---|---|
| the toy engine computing a row (its "decode") | 1,161 |
| next-logprobs' copy (`aclone` of the float row) | 17 |
| float row → double[] → gen.array (`array/array` clones it again) | 105 |
| `categorical-logits` draw (Gumbel-max over the row) | 4,642 |
| `categorical-logits` log-probability of a token (log-softmax, pick) | 981 |
| a host draw: one gen.random uniform, inverse CDF over the float row | 78 |

**Memory** (`clojure -M:bench -m gen.lm-bench memory`, at the default heap,
8 GB here, and at `-Xmx2g`). A held entry costs its branch's row and about 7
KiB of books: the cache never copies a row to hold it, and it holds rows
only at its entries (at most `:max-branches`); a path without an entry
holds tokens and scores only, about 20 bytes a token (about 36 with token ids
above 127, whose boxed longs are objects of their own).

| | default heap (4 MB regions) | `-Xmx2g` (1 MB regions) |
|---|---|---|
| held per entry, Qwen3 (a 594 KiB row) | 600 KiB | 1,025 KiB: the row is humongous, a region of its own |
| 5,000 hits, each copying a row | 28.9 µs a hit; 14 collections, 6 ms of GC | 28.1 µs; 17 collections, 14 ms |

So the bound on held rows is `:max-branches` rows (64 × 594 KiB = 37 MiB for
Qwen3; 255 × 1 MiB for Gemma 3), and below the heap where a row stops
being humongous (4 GB for Qwen3, 8 GB for Gemma 3's 1,048,592-byte rows)
each held row takes a whole region, up to 2 MiB for Gemma 3's. Copies
cost nothing measurable either way. **Approved (owner, 2026-10-20):** run LLM work
with a heap of 8 GB or more, or `-XX:G1HeapRegionSize=4m`, where no row is
humongous.

## For the llama.cpp runs

Not run yet: llama.cpp's libraries and a GGUF model are not on this Mac
(the Branching runs were on the MacBook), and this Mac is shared until M4
closes. To do, with
Qwen3-0.6B, Qwen3.5-0.8B (hybrid) and Gemma-3-270M (sliding window):

- `clojure -J-Dgen.llama.test-model=<gguf> -X:test :nses '[gen.llm.llama-lm-test]'`:
  the contract within 0.1 nats, bit-identical reruns on new engines, the
  GGUF tokenizer's byte vocabulary (and its cost).
- `clojure -M:bench -m gen.lm-bench <gguf>`: per-call costs against the
  Branching operations (hit, one-token read, 8 and 32 in one advance, a
  16-token score read and from the trie), and a token SMC of 16 particles ×
  64 steps (ms a step, reads, evictions, hit rate, backstop count).

## For the next beans

- **genclj-0cvv (token GF).** Traces hold tokens; resume, regenerate and
  assess are LM calls. assess after simulate in one LM reads nothing (the
  scores along the path are the steps', bit for bit), so law 2 holds
  exactly while the path is held, and within the band after an eviction:
  the GF should keep its own scores (score once) to make it exact always.
  The site's draw (decision 2) dominates on the JVM backend.
- **genclj-loyp (token SMC).** A step is `batch-next-logprobs` over the
  particles' contexts; resampling is choosing contexts; prefix potentials
  are `score`s from the trie. The LM's bounds replace GenMLX's live-handle
  ledger (R1, R2), and nothing needs a disposal scope.
- **genclj-q38h (grammar).** `byte-vocab` gives the token-byte trie's
  input; masks apply to the float rows on the host.
