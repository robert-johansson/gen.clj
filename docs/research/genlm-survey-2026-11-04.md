# The GenLM projects against gen.clj's LLM stack

*Written by a side agent at the owner's request (2026-11-04; bean
genclj-7rjv), reviewed by the lead the same day: the claims about gen.clj
were spot-checked against the code (gap 1, `gen.llm.smc/twist-row` and
`gen.llm.potential`'s dense row from prefix and complete, n-vocab + 2
calls; gap 4, `gen.llm.bytes/byte-lm`; the completion that ends at
max-tokens), the rest taken as the agent wrote them. Its recommendations
(§4) are proposals: beans follow the owner's choice.*

*Survey of 2026-11-04, bean genclj-7rjv. Sources: the nine public
repositories of github.com/genlm, cloned at their `main` heads of
2026-11-04 and read (nothing run, installed or built), their release notes,
docs and tests, the papers they cite (arXiv abstracts; two papers' result
tables from their arXiv HTML), and genlm.org. gen.clj side: DESIGN.md §6
and §9, `docs/research/llm-*.md`, book chapters 043 and 044, the docstrings
of `src/gen/llm/`, `ext/codegen`, and the beans named below.*

Effort scale used below: **S** a few days, **M** one to two weeks, **L**
several weeks. Value is for gen.clj's goals (DESIGN §1 and §6, M6 and M7).

---

## 1. Summary

GenLM is the Python ecosystem of the "GenLM Consortium" (genlm.org: McGill,
Mila, MIT, ETH Zürich, Yale, Johns Hopkins) for language-model
probabilistic programming. It has nine public repositories, none of them
forks: six libraries (Apache-2.0) and three paper artifacts. The
centre is **genlm-control** (v0.4.1, 2026-10-08; 192 stars), which
generates text under constraints by SMC over token sequences. It sits on
**genlm-backend** (v0.3.1, 2026-10-14: vLLM, SGLang, transformers and
MLX-LM behind one async, autobatched `next_token_logprobs`),
**genlm-bytes** (v0.2.0, 2026-04-07: token LMs read as byte LMs by beam
summing over tokenizations, with adaptive token healing) and
**genlm-grammar** (v0.2.0, 2025-04-10: weighted CFGs, Earley and CKY,
weighted automata and transducers over semirings; maintenance only since).
**genlm-eval** (v0.3.0, 2026-09-26, not on PyPI) holds the benchmark
domains. **llamppl** (v0.2.3, 2026-04-09) is the older model-level PPL
(SMC steering); genlm-control dropped its dependency on it in 0.4.0.

The work of the last six months is about speed and breadth:
genlm-control 0.4.0 replaced llamppl's loop with its own SMC with batched
draws and potentials autobatched by default, added proposals from other
potentials (a small LM proposing for a large one), forced end-of-sequence at
`max_tokens` with an importance-weight correction, and tempering and
normalization of potentials; 0.4.1 added SMC ensembles of LMs, also of
models with different tokenizers through byte-level LMs (Chan et al.,
arXiv 2603.05432, 2026). genlm-backend 0.3.0 serves SMC from vLLM v1's own
scheduler and selects LoRA adapters per request; 0.2.1 added SGLang.
genlm-eval 0.3.0 added LiveCodeBench and a multilingual variant, beside
0.2.0's DS-1000, all with potentials that run the generated code. Open pull
requests add conditional SMC to genlm-control (#144) and llamppl (#43).

gen.clj already has the foundations GenLM is built on, in several places
with stronger guarantees: the LM protocol (after genlm-backend's AsyncLM)
with a bounded, deterministic branch cache; potentials (after
genlm-control's) with products; token SMC with four resampling schemes and
an unbiased Ẑ tested against enumeration; regular-language constraints with
exact, never-stranding masks; byte-level marginals over tokenizations,
exact or by beam. Beyond GenLM, completions are generative functions with
every GFI operation (update and regenerate on text, GFI laws), and every
run is a function of its key with batch composition independent of thread
timing. GenLM has none of these; its draws use the global torch RNG.

The gaps that matter most:

1. **Critics evaluated at the drawn token only.** genlm-control's SMC
   applies an expensive potential (a classifier, a code runner, a
   planner) once per particle and step. gen.llm.smc twists by the
   potential's full next-token row, which for a potential given by
   `:prefix`/`:complete` costs n-vocab + 2 calls per particle and step
   (`src/gen/llm/smc.clj`, `twist-row`). This blocks the architecture of
   Loula et al. (ICLR 2025): cheap potentials in the proposal, expensive
   ones as critics. Lacks; no bean.
2. **Adaptive weighted rejection sampling (AWRS)**, the sampler GenLM
   recommends for every Boolean constraint (Lipkin et al., COLM 2025,
   outstanding paper). It checks a constraint on a few tokens a step
   instead of the whole vocabulary and returns an unbiased weight. Lacks;
   no bean (noted in `docs/research/survey-2026-10-24/genjl-gaps.md`, E13).
3. **Context-free grammars and JSON Schema as potentials** (BoolCFG from
   Lark syntax, Earley; JsonSchema as an incremental parser times a Draft 7
   validator). Lacks; JSON over regular terms is the draft genclj-w120;
   CFGs are sketched in `docs/research/llm-grammar.md` with no bean.
4. **Byte- and character-level generation**: ByteLLM as a potential,
   character-level SMC, adaptive token healing, ensembles across
   tokenizers. Partial: `gen.llm.bytes/byte-lm` gives exact or beam
   marginals, recomputed per call; the generative function is genclj-ealq.
5. **The potential algebra beyond products**: an LM as a potential (prompt
   intersection), coercion of byte-level potentials to tokens, tempering
   (`p ** β`), local normalization, f-ensembles. Partial (grammars are
   coerced internally); temperature is genclj-paid (draft).
6. **Proposals and steps of other shapes**: a second LM as proposal,
   set-based samplers for soft byte-level potentials, multi-token units
   with boundary predicates, a stop predicate (`terminate_when`). Partial
   at best (`:proposal` takes a function, per particle).
7. **Stateful and batched potentials.** Partial: the grammar precomputes
   its masks per DFA state, but `gen.llm.potential`'s protocol takes the
   whole context on every call and one context at a time. Batched
   potentials are in genclj-9dvz.
8. **Evaluation domains.** genlm-eval has eight domains with published
   numbers (pattern matching, JSONSchemaBench, Spider, molecular
   synthesis, Planetarium, DS-1000, LiveCodeBench and its multilingual
   variant). gen.clj has the codegen bake-off's harness (`ext/codegen`)
   and no shared benchmark to compare against GenLM's numbers.
9. **Canonical BPE tokenization** as a potential (Vieira et al., ICML
   2025). Lacks; no bean.
10. **Engines and adapters**: vLLM, SGLang, transformers, MLX-LM; LoRA per
    request. gen.clj has llama.cpp only (Metal, CUDA, CPU); MLX-native is
    genclj-gvfa (draft), adapters are only in genclj-q4xv's training plan.
    Python engines are out of gen.clj's scope; llama.cpp's own LoRA
    support is in reach.

---

## 2. The repositories

All nine are public, Python, and not forks. The six libraries and
prefix-acl-26 are Apache-2.0; awrs-colm-2025 and control-iclr-2025
declare no license. Star counts and dates are from the GitHub API on
2026-11-04.

### 2.1 genlm-control

- **Purpose.** Controlled generation from LMs under programmable
  constraints: potentials (weights of token sequences), token samplers
  (properly weighted proposals of the next token) and SMC.
- **State.** v0.4.1 (2026-10-08), on PyPI; v0.4.0 (2026-09-26); v0.3.0
  (2026-04-07); v0.2.13 (2026-02-02); v0.2.12 (2025-11-24). 392 commits,
  192 stars, last push 2026-10-08. One open PR: #144, conditional SMC
  (2026-05-20). Python ≥ 3.11; depends on genlm-backend ≥ 0.3.0,
  genlm-bytes ≥ 0.2.0, genlm-grammar ≥ 0.2.0, torch, jsonschema,
  json-stream. About 9,200 lines in `genlm/control/`.
- **Potentials** (`genlm/control/potential/`):
  - The contract (`base.py`, `docs/potentials.md`, "Formalization"):
    `prefix`, `complete`, `logw_next` over the vocabulary and EOS,
    consistent by the autoregressive factorization; batch variants of
    each; `is_terminal_only`.
  - Built in: `PromptedLLM` (an LM conditioned on a prompt, with
    temperature, EOS byte strings, `spawn`, a LoRA adapter name);
    `ByteLLM` (an LM over bytes through genlm-bytes' beam); `WFSA` and
    `BoolFSA` from regexes; `WCFG` and `BoolCFG` (Lark syntax, Earley,
    through genlm-grammar); `JsonSchema` (`json.py`, 991 lines: a
    streaming parser-combinator potential times a Draft 7 validator run at
    "logical units"; enum, const, pattern, anyOf, required,
    additionalProperties); `CanonicalTokenization` (`canonical.py`: BPE
    merges read from the tokenizer, a bigram canonicality filter);
    `HarmonyPotential` (applies a potential to chosen channels of the
    Harmony chat format only); `Ensemble` (two potentials combined by a
    weighted power mean, p any real, sum, prod, harmonic, min or max).
  - Operators (`operators.py`, `product.py`, `coerce.py`, `tempered.py`,
    `normalized.py`): `p * q` (a product over the intersection of their
    vocabularies, warning under 10% overlap), `p.coerce(q, f)` (a potential
    read through a map of sequences, with a trie fast path for FSAs when
    `f` distributes over concatenation), `p ** β`, `p.normalize()`,
    `to_autobatched()`, `to_multiprocess(n)`.
  - `StatefulPotential` and `AsyncStreamingPotential` (`stateful.py`,
    `streaming.py`): a per-context state cache with LRU eviction, used by
    the JSON potential.
  - `PotentialTests` (`testing.py`): `assert_logw_next_consistency`,
    `assert_autoreg_fact`, `assert_batch_consistency`, inherited by every
    potential.
- **Samplers** (`genlm/control/sampler/`, `docs/samplers.md`):
  - `DirectTokenSampler`: draws from the normalized `logw_next` of a
    potential (usually an LM times a constraint); with `proposal=`, from
    another potential with an importance correction (0.4.0, #140).
  - `AWRS` (`token.py`): adaptive weighted rejection sampling for an LM
    times a Boolean condition. It walks tokens in a Gumbel-perturbed order
    without replacement, checks the condition only on tokens it reaches,
    and estimates the local normalizer unbiasedly: `recursive_awrs`, or
    `geometric_awrs` (reconstructs a with-replacement run by geometric
    "phantom" draws; chosen when the top token's probability exceeds a
    threshold or `max_accepts` > 2). Options `max_accepts`, `max_rejects`,
    `proper_weights`, `proposal`. Tested for unbiasedness
    (`tests/sampler/test_awrs.py`).
  - `SetTokenSampler` over `EagerSetSampler` or `TopKSetSampler`: a
    weighted set of tokens drawn through the byte trie, for non-Boolean
    byte-level potentials (Loula et al. 2025, appendix C).
  - `MultiTokenUnitSampler` (`unit.py`): a unit of several tokens until a
    boundary predicate holds (`TokenSetBoundary`, `FixedLengthBoundary`,
    `CFGBoundary`), properly weighted.
  - Draw methods: `gumbel_max` (default), `multinomial`, `inverse_cdf`, on
    the global torch RNG (`torch.manual_seed` for reproducibility).
- **SMC** (`sampler/sequence.py`, `sampler/smc.py`): particles extended by
  a token sampler; a `critic` potential twists each step by its score of
  the context (untwisted next step), or scores once at termination if it is
  terminal-only; ESS threshold; resampling `multinomial` (default),
  `stratified`, `systematic`, `residual` (vendored from llamppl);
  `terminate_when` (a stop predicate, part of the target); EOS forced at
  `max_tokens` with a weight correction, so particles target the
  length-conditioned distribution (#145). Results as `Sequences`:
  `posterior`, `decoded_posterior` (by UTF-8 text), `log_ml`, `ess`. A JSON
  record of a run and an HTML visualizer (`viz.py`, `html/smc.html`).
  Concurrent SMC runs over one LM share batches (`docs/performance.md`).
- **The last months.** 0.2.13: character-level SMC through genlm-bytes,
  generate-until. 0.3.0: `Token` objects (several tokens may share a byte
  string), the Harmony format. 0.4.0: SMC native to the library (llamppl
  removed), batched draws, autobatching on by default, arbitrary
  proposals, forced EOS, Boolean semiring for `BoolFSA`, tempering and
  normalization. 0.4.1: SMC ensembles. PR #149's branch history (commit
  490d1d9's message) describes an engine-native "burst" SMC path over vLLM
  and a stateful-potential interface (`state0`, `advance`,
  `logw_next_from_state`); neither is in the merged tree (no
  `controller.py`, no `logw_next_from_state` on `main`).
- **Links.** https://github.com/genlm/genlm-control,
  https://genlm.org/genlm-control/, https://pypi.org/project/genlm-control/

### 2.2 genlm-backend

- **Purpose.** The LM layer: an async `AsyncLM` with
  `next_token_logprobs`, `batch_next_token_logprobs`, `sample`,
  `batch_sample`; concurrent requests batched automatically; a
  tokenizer's byte vocabulary.
- **State.** v0.3.1 (2026-10-14), on PyPI; v0.3.0 (2026-09-26), v0.2.1
  (2026-04-27), v0.2.0 (2026-04-07), v0.1.8 (2026-01-30), v0.1.7
  (2025-11-13), v0.1.6 (2025-10-17). 224 commits, 17 stars. Open PRs: #66
  (log-probabilities in fp32 for HF and MLX), #65 (LoRA for SGLang), #81
  (an embedding-inversion example).
- **Features.**
  - Engines (`genlm/backend/llm/`): `AsyncVirtualLM` (vLLM v1; since
    0.3.0 each live context is a request in vLLM's scheduler, extended by
    fed tokens, full rows captured through the sampler slot), `AsyncTransformer`
    (transformers, with a token trie caching KV and logits), `AsyncMlxLM`
    (MLX-LM, 0.1.6; batching and a KV slot pool that extends the previous
    batch, 0.1.7), `AsyncSGLTransformer` (SGLang, 0.2.1), `MockAsyncLM`.
  - LoRA adapters registered by name and chosen per request (vLLM, HF,
    MLX; 0.1.8 and 0.3.0).
  - `Token` objects (id and bytes; 0.2.0).
  - `TokenCharacterTrie` and a GPU `ParallelTokenCharacterTrie`
    (`trie/`): the probability mass under each trie node, by a sparse
    reachability matrix on the GPU.
  - Benchmark scripts (`benchmark/`: backends, trie mass sums, vLLM
    prefix caching); no published numbers.
- **Links.** https://github.com/genlm/genlm-backend,
  https://genlm.org/genlm-backend/

### 2.3 genlm-bytes

- **Purpose.** Token-level LMs as byte-level LMs.
- **State.** v0.2.0 (2026-04-07, duplicate tokens), v0.1.2 (2025-12-02,
  adaptive token healing), v0.1.1 (2025-11-04, EOS), v0.1.0 (2025-06-06).
  75 commits, 25 stars; last code change 2026-05-14 (numerical stability
  of the trie mass, #15). One open issue: prefill speed (#10).
- **Features.** `ByteBeamState` (`byte_lm/beam.py`): a beam of K
  tokenizations consistent with the bytes so far, carried byte to byte
  (`prefill`, `logp_next` over 256 bytes and EOS, `prune`, `<<` to extend
  by a byte); `BeamParams` (K, `prune_threshold`, EOS byte strings,
  healing options); `TokenHealer` (`byte_lm/heal.py`): when no
  tokenization in the beam can take the next byte, back off and
  re-tokenize the partial token. Grounded in Vieira et al., "From Language
  Models over Tokens to Language Models over Characters" (ICML 2025,
  arXiv 2412.03719; README alone does not cite it, genlm.org does).
- **Links.** https://github.com/genlm/genlm-bytes,
  https://genlm.org/genlm-bytes/

### 2.4 genlm-grammar

- **Purpose.** Weighted CFGs, weighted FSAs and FSTs over semirings.
- **State.** v0.2.0 (2025-04-10). 74 commits, 10 stars. Since then only
  maintenance: Log-semiring input validation (2026-05-21) and PyPI
  publishing (2026-09-26). Open issues #16 (a bug in nonterminal
  equality), #18, #21.
- **Features** (README and `genlm/grammar/`): CFGs over semirings
  (Boolean, Float, Real, Log, MaxPlus, MaxTimes, Expectation, Entropy);
  transformations (local normalization, nullary and unary-cycle removal,
  binarization, length truncation, renumbering); Earley (plain and
  rescaled) and incremental CKY with prefix weights; `BoolCFGLM`,
  `EarleyLM`, `CKYLM` with `p_next`; WFSAs (epsilon removal,
  determinization, Brzozowski minimization, composition, reversal, star);
  WFSTs and CFG–FST composition; Lark grammars (`lark_interface.py`).
- **Links.** https://github.com/genlm/genlm-grammar,
  https://genlm.org/genlm-grammar/

### 2.5 genlm-eval

- **Purpose.** Evaluation of constrained generation: `Dataset`,
  `Evaluator`, `ModelAdaptor`, `run_evaluation` (with cached outputs and
  replicates).
- **State.** v0.3.0 (2026-09-26), v0.2.0 (2025-11-19), v0.1.0
  (2025-05-07); not on PyPI (installed from git). 108 commits, 3 stars,
  last push 2026-09-27.
- **Domains** (`genlm/eval/domains/`): pattern matching (400+ patterns,
  features beyond regular expressions through Python's `regex`),
  JSONSchemaBench, Spider text-to-SQL (with a table–column verifier
  potential), molecular synthesis (SMILES, a `PartialSMILES` potential),
  goal inference on Planetarium (a VAL-checked plan potential), DS-1000
  (a runtime no-error potential in a fork server), LiveCodeBench and a
  multilingual variant (0.3.0: a runtime no-error potential that runs the
  code's leading statements on public inputs, and a soft public-test
  potential with a repair prompt).
- **Links.** https://github.com/genlm/genlm-eval,
  https://genlm.org/genlm-eval/

### 2.6 llamppl

- **Purpose.** Model-level LM probabilistic programming: a `Model` class
  with `step`, `sample(dist, proposal)`, `observe`, `condition`, `twist`,
  `intervene`, submodels; `LMContext` with next-token and mask
  distributions; inference by `smc_standard` and `smc_steer` (SMC
  steering: each particle extends to K children, then N are kept by
  without-replacement resampling; Lew et al., arXiv 2306.03081).
- **State.** v0.2.3 (2026-04-09: stratified, residual and systematic
  resampling), v0.2.2 (2025-11-18: MLX), v0.2.0 (2025-04-10: renamed from
  hfppl). 146 commits, 183 stars. Since then: stratified resampling by
  default (#44, 2026-05-27); an impossible mask now kills the particle
  instead of aborting the run (2026-06-02). Open PRs: #43 conditional SMC,
  #52 MLX tests against genlm-backend 0.3.0.
- **Links.** https://github.com/genlm/llamppl, https://genlm.org/llamppl/

### 2.7 awrs-colm-2025 (paper artifact)

Experiment scripts for "Fast Controlled Generation from Language Models
with Adaptive Weighted Rejection Sampling" (Lipkin et al., COLM 2025,
arXiv 2504.05410). 8 commits, last push 2025-08-08; the algorithm itself
lives in genlm-control. Reported (Table 1, Llama 3.1 8B; AWRS-SMC with 5
particles, the other SMC methods with 10):

| Domain | Base LM | ARS-LCD | Sample-verify | Twisted SMC | AWRS-SMC | AWRS-SMC s/example |
|---|---|---|---|---|---|---|
| Spider (text-to-SQL) | 0.523 | 0.572 | 0.609 | 0.608 | 0.600 | 3.02 |
| JSONSchemaBench | 0.688 | 0.773 | 0.858 | 0.871 | 0.898 | 12.71 |
| Planetarium (goals) | 0.032 | 0.18 | 0.205 | 0.479 | 0.528 | 2.62 |
| Molecular synthesis | 0.271 | 0.522 | 0.594 | 0.591 | 0.615 | 3.41 |
| Pattern matching | 0.562 | 0.980 | 0.786 | 0.813 | 0.990 | 0.43 |

On pattern matching, masking every token (TM-LCD) takes 6.91 s per example
against ARS-LCD's 0.16 s at the same accuracy (0.978). The paper reports
about 2 to 3 constraint checks per step there, against 100,000+ for a full
mask.

### 2.8 control-iclr-2025 (paper artifact)

Experiments of "Syntactic and Semantic Control of Large Language Models via
Sequential Monte Carlo" (Loula et al., ICLR 2025 oral; arXiv 2504.13139).
10 commits, last push 2026-02-03. The architecture: cheap potentials
(grammars, parsers) shape the proposal; expensive ones (running code,
simulating a plan, checking SQL aliases and columns, validating SMILES)
reweight partial sequences, at up to about 30 ms per token on CPU.
Reported (Table 2, 10 particles; Llama 3.1 8B, 70B for data science):

| Method | Goal inference | Molecules | Data science | Text-to-SQL |
|---|---|---|---|---|
| Base LM | 0.063 | 0.132 | 0.213 | 0.531 |
| Locally constrained decoding | 0.086 | 0.189 | – | 0.559 |
| Sample-rerank | 0.289 | 0.392 | – | 0.581 |
| Grammar-only SMC | 0.401 | 0.205 | – | 0.596 |
| Full IS | 0.257 | 0.404 | 0.346 | 0.618 |
| Full SMC | 0.419 | 0.577 | 0.407 | 0.620 |

### 2.9 prefix-acl-26 (paper artifact)

A frozen snapshot of genlm-grammar for "Prefix Parsing is Just Parsing"
(Pasti et al., ACL 2026; arXiv 2604.21191), created 2026-04-16, last push
2026-04-23, 84 commits. It adds the prefix grammar transformation
(`cfg.py`, `prefix_grammar`: a grammar of the prefixes of a grammar's
strings, so any parser computes prefix weights), next-token weights by
algorithmic differentiation (`next_token_weights` in the Earley and CKY
parsers), an Earley parser in Rust (`rust/src/lib.rs`, 1,078 lines, PyO3),
and benchmarks against EarleyX. The README says the maintained code is
genlm-grammar. genlm-grammar's `main` has `prefix_grammar` and
`next_token_weights` in Python, but not the Rust parser.

### 2.10 Outside the organization

- "Self-Steering Language Models" (DisCIPL; Grand et al., COLM 2025,
  arXiv 2504.07081) is on genlm.org's publication list: a planner LM
  writes inference programs that small follower LMs run. Its code is not in
  the organization; not checked.
- PR #144 names a "genlm-latent" trainer that uses conditional SMC. No
  such public repository exists in the organization.

---

## 3. The gap table

has: gen.clj covers it. partial: some of it, or by composition. lacks: no
counterpart. Paths are relative to the gen.clj repository.

### 3.1 Potentials

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Potential contract: prefix, complete, next-token weights over vocabulary + EOS | control `potential/base.py` | **has**: `gen.llm.potential` (same contract, DESIGN §9) | — | — |
| Products of potentials | control `product.py` | **has**: `potential/product` (one vocabulary; GenLM intersects two) | — | — |
| Contract checks for users | control `potential/testing.py` | **partial**: `test/gen/llm/potential_contract.clj` (test code, not shipped; not in `ext/laws`) | medium: rule 7 wants laws for every new kind | S |
| Boolean FSA from regex | control `wfsa.py` | **has**: `gen.llm.regex`, `gen.llm.dfa`, `gen.llm.grammar` (intersection, complement, minimal DFAs, exact token-level liveness) | — | — |
| Weighted FSA (soft regular potentials) | control `wfsa.py`, grammar `wfsa/` | **lacks** | low | M |
| CFG potentials (BoolCFG, WCFG; Earley; Lark syntax) | control `wcfg.py`, genlm-grammar | **lacks**: sketched in `docs/research/llm-grammar.md` ("Context-free and reader potentials"); no bean | high: recursive JSON, code, SQL | L |
| JSON Schema potential | control `json.py` | **lacks**: genclj-w120 (draft; malli → regular terms, recursive schemas need a CFG) | high: tool calls, structured output for M6 | M (regular part) |
| Canonical BPE tokenization | control `canonical.py` | **lacks**; no bean | medium: better likelihoods (Vieira et al. 2025); needs merges from the GGUF | M |
| Channel-scoped potential (Harmony) | control `harmony.py` | **partial**: a regular "think block, then answer in L" is writable with `regex/complement` and `intersect`; no combinator scoping any potential to a channel; `gen.llm.chat/split-reasoning` parses after the fact | medium: Qwen thinking models | S |
| LM as a potential; prompt intersection (`llm1 * llm2`) | control `llm.py` | **lacks**: the LM is the completion's base measure; a second prompt's weights must be written by hand with `potential/potential` and `lm/next-logprobs` | medium | S |
| Coercion of any byte-level potential to tokens | control `coerce.py` | **partial**: grammars are read through the vocabulary internally (`gen.llm.bytes/sweep`); no general combinator | medium | S–M |
| Tempering (`p ** β`), local normalization | control `tempered.py`, `normalized.py` | **lacks**: temperature exists only as a sampler (`gen.llm.sample`); genclj-paid (draft) designs it as a potential | medium | S |
| Ensembles (f-ensembles, across tokenizers) | control `built_in/ensemble.py`; Chan et al. 2026 | **lacks**; no bean | low–medium | S (same tokenizer), L (bytes) |
| Stateful / streaming potentials (state per context, LRU) | control `stateful.py`, `streaming.py` | **partial**: grammar masks precomputed per DFA state (`gen.llm.grammar`); the protocol passes the whole context every call | medium: cost of oracle potentials | M |
| Batched potentials (`batch_logw_next` …) | control `base.py`, `autobatch.py` | **partial**: one context per call; batched protocol is genclj-9dvz; autobatching rejected by design (DESIGN §9) | medium | M |
| Multiprocess potentials | control `multi_proc.py` | **partial**: token SMC's host pool computes proposal rows and draws (`gen.llm.host`); potential calls stay on the calling thread | low | S |

### 3.2 Samplers and SMC

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Direct sampling from LM × potential | control `DirectTokenSampler` | **has**: `:masked` proposal and constrained completions (`gen.llm.constrained/weighted`) | — | — |
| Critic applied at the drawn token only (one call per particle and step); terminal-only critics | control `sampler/smc.py` (`twist`, `is_terminal_only`) | **lacks**: `gen.llm.smc`'s `twist-row` reads `potential/logw-next` for every live particle; a potential without `:logw-next` costs n-vocab + 2 calls (`src/gen/llm/potential.clj`, `dense-logw-next`) | **high** | S |
| AWRS (Boolean constraints by adaptive rejection, unbiased weights) | control `token.py` `AWRS` | **lacks**; no bean | **high** for oracle constraints (parsers, code checks); low for DFA grammars, whose masks are precomputed | M |
| Proposal from another potential (small LM for a large one) | control `DirectTokenSampler(proposal=)` | **partial**: `:proposal` function `(fn [context row] → q)` may read a second LM, one particle at a time | medium | S |
| Set-based samplers for soft byte-level potentials | control `sampler/set.py` | **lacks** | low–medium | M |
| Multi-token units with boundary predicates | control `sampler/unit.py` | **lacks** | medium: word-, line-, statement-level critics | M |
| Stop predicate (`terminate_when`) | control `sequence.py` | **partial**: `:end-tokens` only (`gen.llm.token`, `gen.llm.sample`) | medium | S |
| EOS forced at max_tokens, weight-corrected | control #145 | **different by design**: a completion stopped at max-tokens is complete (DESIGN §9, token SMC) | — | — |
| Resampling: multinomial, stratified, systematic, residual; ESS-gated | control `resampling.py`, llamppl 0.2.3 | **has**: `gen.inference.weights`; token SMC systematic by default | — | — |
| SMC steering (without-replacement resampling, beam factor) | llamppl `smc_steer.py` | **lacks** | low | S |
| Conditional SMC on token sequences | control PR #144, llamppl PR #43 (open) | **partial**: `gen.inference.particle-mcmc` for gen models; not for `gen.llm.smc` | low | M |
| Rejuvenation after resampling | not in GenLM | **lacks** in both: genclj-9dvz | (gen.clj ahead once built) | M |
| Posterior summaries (posterior, decoded posterior, log ML, ESS) | control `Sequences` | **partial**: `smc/log-ml-estimate`, `ess`, `particles`, `traces`; weighted estimates are genclj-oa7v | medium | S |
| Several SMC runs sharing LM batches | control `docs/performance.md` | **lacks** (no autobatcher, by design; a lockstep batcher is allowed by DESIGN §9) | low–medium | M |
| Run record and HTML visualizer | control `smc_record.py`, `viz.py` | **lacks** | low | S |
| Draw methods, reproducibility | control `set_draw_method`; torch RNG | **has**, stronger: host inverse-CDF draws from `gen.random` keys, results independent of pool size (DESIGN §9) | — | — |
| Model-level LM programs (sample, observe, condition, twist) with LM calls batched across particles | llamppl `modeling.py` | **partial**: gen models splice completions and `token/dist` sites, and IS and particle filters run them (book 043, "A language model inside a model"), but token operations under a batch tracer fall back to one particle at a time (DESIGN §9) | medium–high for M7 (genclj-tw4p) | L |

### 3.3 Bytes and characters

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Text probability over tokenizations, exact or beam | genlm-bytes | **has**: `gen.llm.bytes/byte-lm` (`prefix-logprob`, `text-logprob`, `next-byte-logprobs`; exact, or `:beam k` as a lower bound) | — | — |
| Incremental byte state (beam carried byte to byte, pruning threshold) | genlm-bytes `ByteBeamState` | **partial**: each call walks the text from its start (the LM's cache makes re-reads cheap) | medium | M |
| Adaptive token healing | genlm-bytes `heal.py` | **lacks** | medium: prompts that end mid-token | M |
| Byte LM as a potential; character-level SMC | control `ByteLLM` (0.2.13) | **lacks**: genclj-ealq (a byte-level generative function) | high: character-level control, cross-tokenizer work | M–L |
| Several tokens with one byte string | backend 0.2.0, bytes 0.2.0 | **has**: tokens are integers; the byte trie keeps every token at its node (`gen.llm.bytes/tokens-at`) | — | — |

### 3.4 Backends

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Engines | backend: vLLM, SGLang, transformers, MLX-LM | **partial**: llama.cpp (Metal, CUDA, CPU; `gen.llm.llama`); MLX-native is genclj-gvfa (draft) | Python engines: none (out of scope) | — |
| Prefix and KV caching | backend `cache.py`, vLLM scheduler | **has**: `gen.llm.cache`, `gen.llm.trie`, `gen.llm.cells` (bounded by branches and cells, forks by `llama_memory_seq_cp`, deterministic) | — | — |
| Batching | backend: automatic (async) | **has**, explicit: `lm/batch-next-logprobs`, one advance per SMC step | — | — |
| LoRA adapters chosen per request | backend 0.1.8, 0.3.0 | **lacks**: genclj-q4xv plans training, and notes llama.cpp runs adapters | medium: specialised critics or proposals from one base model | S–M |
| Byte vocabulary | backend `byte_vocab` | **has**: `lm/token-bytes`, `lm/byte-vocab` | — | — |
| Sampling with temperature | backend `sample` | **has**: `gen.llm.sample` (temperature, top-k, top-p; lockstep) | — | — |
| Mock LM | backend `MockAsyncLM` | **has**: `test/gen/llm/toy_engine.clj`, `cell_engine.clj` | — | — |
| Trie mass sums on the GPU | backend `trie/parallel.py` | **partial**: `gen.llm.bytes/log-mass` on the host | low | — |

### 3.5 Grammars (genlm-grammar, prefix-acl-26)

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Weighted CFGs over semirings; transformations | genlm-grammar `cfg.py`, `semiring.py` | **lacks** | medium (needed only for weighted grammars) | L |
| Earley / CKY with prefix weights and next-token weights | genlm-grammar `parse/`; prefix-acl-26 (Rust, prefix grammar) | **lacks** | high, as the CFG potential's engine | L |
| WFSA operations, WFSTs | genlm-grammar `wfsa/`, `fst.py` | **partial**: unweighted terms with intersection and complement, minimal DFAs (`gen.llm.regex`, `gen.llm.dfa`); no weights, no transducers | low | M |

### 3.6 Evaluation

| Feature | Where in GenLM | gen.clj | Value | Effort |
|---|---|---|---|---|
| Harness (dataset, evaluator, model adaptor, cached runs, replicates) | genlm-eval `core/` | **partial**: `ext/codegen` (`gen.codegen.driver`, `.grade`, `.report`, `.tasks`), for writing gen.clj models only | medium | S–M |
| Domains with published numbers | genlm-eval `domains/` | **lacks** | medium: the only way to compare with §2.7–2.8 | M |
| Execution potentials (runtime no-error, public tests with repair feedback) | genlm-eval LiveCodeBench, DS-1000 | **partial**: `gen.codegen.session` and `.process` read, repair, vet and evaluate code with timeouts; not used as potentials in token SMC (genclj-hu8z, level 1–2) | high for the codegen line (genclj-hu8z, genclj-ihc4) | M |

---

## 4. Recommendations, ranked

Ranked by value for effort. Each names its layer (CLAUDE.md's table) and the
bean that covers it, if one does.

1. **A critic in token SMC, applied at the drawn token** (S, high).
   Layer: LLM generative functions (`gen.llm.smc`). Add `:critic` to
   `smc/initialize`: after a token is drawn, the particle's weight gains
   prefix(ctx ++ t) − prefix(ctx) of the critic (its complete at the end),
   one call per particle and step; a critic with zero prefix weights is
   applied at the end only. `:potential` keeps its role in the proposal
   (the mask). The target is unchanged in form: π ∝ p · exp(complete of
   the product). Tests: Ẑ against enumeration as now; a planted bug
   (twisting by the critic's prefix of the context before the token).
   This is the Loula et al. architecture and the hook for genclj-hu8z's
   code checks as weights. No bean: propose one.
2. **AWRS as a token-SMC proposal** (M, high for oracle constraints).
   Layer: `gen.llm.smc` (a `:proposal :awrs` with a Boolean potential,
   or an accept function). Its weights are random, so it belongs to SMC,
   not to a constrained completion's exact GFI density. The Gumbel noise
   comes from the step's key. DFA grammars keep their precomputed masks;
   AWRS is for constraints with no mask: Clojure's reader and clj-kondo on
   partial code (genclj-hu8z), parsers, SMILES, SQL checks. Tests: the
   weights' unbiasedness against exact Z on the toy vocabulary, as
   genlm-control's `test_awrs_is_unbiased`. Port both variants
   (`recursive_awrs`, `geometric_awrs`) from the paper, not the Python. No
   bean: propose one.
3. **Potential combinators** (S each, medium). Layer: `gen.llm.potential`.
   Tempering and normalization (with genclj-paid's design: temperature
   as `p^(1/T)` renormalized), an LM as a potential (a prompted LM's rows
   as `logw-next`, for prompt intersection), a coercion combinator for
   byte-level potentials through the byte trie, and a channel scope for
   Qwen's think blocks. Ensembles of one tokenizer follow from these.
   genclj-paid covers temperature; the rest need a bean.
4. **The potential contract as laws** (S, medium). Layer: `ext/laws`. Move
   `test/gen/llm/potential_contract.clj` into a law set for potentials
   (consistency, autoregressive factorization, a zero prefix forcing zero
   extensions), so every potential gen.clj adds is checked as rule 7
   wants. No bean.
5. **Stateful and batched potentials** (M, medium). Layer:
   `gen.llm.potential`. An optional protocol (a state per context,
   advanced by a token, as the grammar's DFA state already is) and a
   batched call for all particles. genclj-9dvz holds the batched part; add
   the state there.
6. **A byte LM behind the LM protocol** (M–L, high). Layers: LLM runtime
   and generative functions. genclj-ealq. Suggestion for its design:
   implement `gen.llm.lm/LM` over a vocabulary of 256 bytes and an end
   token, with the beam (or exact runs) carried from byte to byte as
   genlm-bytes does. `token/completion`, `gen.llm.smc` and grammars over a
   byte vocabulary (which `gen.llm.grammar` already accepts) then give
   byte-level generation, character-level SMC and per-byte DFA masks with
   no new GFI code; token healing and cross-tokenizer products follow.
7. **Context-free potentials** (L, high). Layer: LLM generative functions
   (a new `gen.llm.cfg`, a potential). An Earley recognizer advanced byte
   by byte, its state carried down the token trie as the DFA's is
   (`docs/research/llm-grammar.md`); prefix-acl-26's prefix grammar and
   next-token weights are the reference. Needed for recursive schemas
   (genclj-w120) and for Clojure's delimiters past a fixed depth
   (genclj-hu8z). Pair with AWRS where the sweep is too slow. No bean.
8. **JSON Schema and malli structured output** (M for the regular part).
   genclj-w120, as designed there; recursive schemas wait for item 7.
9. **Execution potentials for codegen** (M, high for that line). Layer:
   `ext/codegen`. Wrap `gen.codegen.session`'s checks as a critic (item 1)
   and an AWRS condition (item 2): partial code that reads and resolves
   its names. genclj-hu8z's first unchecked item.
10. **A second LM as a batched proposal** (S, medium). Layer:
    `gen.llm.smc`. `:proposal-lm` read with `lm/batch-next-logprobs` once
    a step, for models that share a vocabulary. No bean.
11. **Stop predicates and multi-token units** (S–M, medium). Layer:
    `gen.llm.token` (a stop predicate as part of the completion's
    distribution, with GFI laws) and `gen.llm.smc` (steps of a unit). No
    bean.
12. **Canonical tokenization** (M, medium). Layer: `gen.llm.potential`,
    with the BPE merges read from the GGUF's metadata by `gen.llm.llama`.
    No bean.
13. **Evaluation against GenLM's numbers** (M, medium). A module
    `ext/llm-eval` with two domains whose checkers need no Python:
    pattern matching (java.util.regex where it agrees with Python's
    `regex`) and JSONSchemaBench (a JVM JSON Schema validator as an
    `ext/` dependency, rule 8). Same model family as the papers where
    llama.cpp runs it. No bean; fits beside genclj-whi8.
14. **LoRA adapters through llama.cpp** (S–M, medium). Layer: LLM runtime
    (`gen.llm.llama`, an adapter per LM). genclj-q4xv's inference half.
15. **Not recommended now**: Python engines (vLLM, SGLang, transformers);
    a run visualizer; SMC steering; conditional SMC on tokens (GenLM's is
    unmerged; gen.clj has particle MCMC for gen models); autobatching
    (rejected in DESIGN §9).

Where gen.clj leads, keep it: completions as generative functions with
update and regenerate, exact masks that never strand, results independent of
thread timing, and Ẑ tested for unbiasedness against enumeration with
bands from the estimator's variance. GenLM's tests also compare log ML
with exact sums on small weighted sets (property-based, hypothesis), but
with fixed tolerances (its critic test: `rtol=0.05, atol=0.5`,
`tests/sampler/test_seq_sampler.py`).

---

## 5. What I could not check

- The arXiv papers were read as abstracts; only the AWRS (2504.05410v2)
  and ICLR (2504.13139v2) result tables were read from their HTML (the
  first 100,000 characters of each). AWRS's "recursive" and "geometric"
  variants are described from the code, not the paper.
- No GenLM code was run, so no speed comparison with gen.clj exists. gen.clj's
  measured costs (DESIGN §9: a 64-particle masked step 51 ms on
  llama.cpp with Qwen3-0.6B; a constrained token 270–490 µs beyond its
  draw, M4) and
  GenLM's (Llama 3.1 8B on A100/H100) share no setting.
- genlm.org's rendered docs sites were not read; their sources in each
  repository's `docs/` were.
- DisCIPL's code (outside the organization) and "genlm-latent" (named in
  PR #144, not public) were not read. Private repositories of the
  organization, if any, are invisible.
- The engine-native "burst" SMC is described only in PR #149's commit
  messages; it is not in the merged code.

---

## 6. Sources

Repositories (cloned 2026-11-04, `main`):
- https://github.com/genlm/genlm-control
- https://github.com/genlm/genlm-backend
- https://github.com/genlm/genlm-bytes
- https://github.com/genlm/genlm-grammar
- https://github.com/genlm/genlm-eval
- https://github.com/genlm/llamppl
- https://github.com/genlm/awrs-colm-2025
- https://github.com/genlm/control-iclr-2025
- https://github.com/genlm/prefix-acl-26
- Release notes: `https://github.com/genlm/<repo>/releases` for the six
  libraries; pull requests genlm-control #136, #140, #144, #145, #149,
  llamppl #43, #44, genlm-backend #65, #66, #74, #81.
- PyPI: https://pypi.org/project/genlm-control/,
  https://pypi.org/project/genlm-backend/,
  https://pypi.org/project/genlm-bytes/,
  https://pypi.org/project/genlm-grammar/,
  https://pypi.org/project/llamppl/ (genlm-eval is not on PyPI).

Sites:
- https://genlm.org/ and https://genlm.org/publications/

Papers:
- Lipkin et al., "Fast Controlled Generation from Language Models with
  Adaptive Weighted Rejection Sampling", COLM 2025.
  https://arxiv.org/abs/2504.05410 (HTML v2 for Table 1)
- Loula et al., "Syntactic and Semantic Control of Large Language Models
  via Sequential Monte Carlo", ICLR 2025. https://arxiv.org/abs/2504.13139
  (HTML v2 for Table 2); https://openreview.net/pdf?id=xoXn62FzD0
- Vieira et al., "From Language Models over Tokens to Language Models over
  Characters", ICML 2025. https://arxiv.org/abs/2412.03719
- Vieira et al., "Language Models over Canonical Byte-Pair Encodings", ICML
  2025. https://arxiv.org/abs/2506.07956
- Chan et al., "Ensembling Language Models with Sequential Monte Carlo",
  2026. https://arxiv.org/abs/2603.05432
- Pasti et al., "Prefix Parsing is Just Parsing", ACL 2026.
  https://arxiv.org/abs/2604.21191
- Lew et al., "Sequential Monte Carlo Steering of Large Language Models
  using Probabilistic Programs", 2023. https://arxiv.org/abs/2306.03081
- Grand et al., "Self-Steering Language Models", COLM 2025.
  https://arxiv.org/abs/2504.07081
- Datasets cited by genlm-eval: Planetarium
  (https://arxiv.org/abs/2407.03321), LiveCodeBench
  (https://arxiv.org/abs/2403.07974), DS-1000
  (https://arxiv.org/abs/2211.11501), Spider
  (https://aclanthology.org/D18-1425/).

gen.clj (read, not changed): `CLAUDE.md`, `DESIGN.md` §6 and §9,
`docs/research/llm-backends.md`, `llm-branching.md`, `llm-grammar.md`,
`llm-lm-protocol.md`, `llm-runtime-layers.md`, `llm-token-gf.md`,
`llm-token-smc.md`, `docs/research/survey-2026-10-24/genjl-gaps.md`,
`book/043-llms.md`, `book/044-constrained-generation.md`,
`src/gen/llm/*.clj`, `ext/codegen/src/gen/codegen/*.clj`,
`test/gen/llm/potential_contract.clj`; beans genclj-7rjv, genclj-9dvz,
genclj-ealq, genclj-0izr, genclj-w120, genclj-paid, genclj-oa7v,
genclj-hu8z, genclj-edqy, genclj-q4xv, genclj-xl5e, genclj-whi8,
genclj-ihc4, genclj-ifxh, genclj-gvfa.
