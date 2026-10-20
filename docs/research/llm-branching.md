# The Branching protocol and the llama.cpp engine

*2026-10-20, bean genclj-kwxd (M3's first). Built by an agent; the choices
that were the owner's to make (names, lifetime policy, additions to
DESIGN.md §6's list), marked **Approved**, were approved by the owner the
same day, as were the smaller ones: the scratch slot for `score`, the
unified cache, full-size sliding-window layers, and host floats in the
engine layer. Summarized in DESIGN.md §6 and §9.*

## What exists

| Piece | Where |
|---|---|
| llama.cpp's libraries, v0.7.0 (commit 620d222, checked), Metal on macOS; Linux `cpu` and `cuda` written, untested | `native/build-llama.sh` → `native/build/llama/<platform>/{lib,include}` |
| The binding: jextract 25 over `llama.h`, filtered to the 43 functions called, 3 structs, 4 typedefs (a function jextract leaves out fails the recipe); generated sources committed | `native/jextract-llama.sh` → `src-java/gen/llama/ffi` |
| Library loading (`gen.llama.lib`), version check, log routing (`gen.llama.log`, an unknown level an error), the batch (`Batch`, its views made once), the loops over a row of logits | `src-java/gen/llama/LlamaRuntime.java` |
| The Branching protocol, the Branch value, the checked operations | `gen.llm.branching` |
| An engine over numbered sequence slots: threading, lifetimes, capacity | `gen.llm.slots` |
| The llama.cpp engine | `gen.llm.llama` |
| The contract as engine-agnostic checks; a pure-JVM toy engine; tests | `test/gen/llm/` |
| Float-noise and cost measurement, for short prompts and for branches past a sliding window (the contract's own statistic) | `bench/gen/llama_noise.clj`, `bench/gen/llama_long_noise.clj` |

The core stays Clojure-only (CLAUDE.md rule 8): the binding is Java source
compiled by `clojure -T:build java`, like MLX's, and loads libllama only when
an engine is made. Nothing in L0–L5 requires `gen.llm.*`.

## The API

```clojure
(require '[gen.llm.branching :as br] '[gen.llm.llama :as llama])

(with-open [e (llama/engine {:model "Qwen3-0.6B-Q8_0.gguf" :max-branches 16 :n-ctx 4096})]
  (let [b (br/open e (llama/tokenize e "The capital of Sweden is"))   ; prefill
        [x y] (br/fork b 2)                                             ; seq_cp, b stays live
        [x' y'] (br/advance [x y] [t1 t2])                              ; one decode; x, y consumed
        lp (br/logprobs x')                                             ; float[] of the vocabulary
        s (br/score b continuation)                                     ; [log p(c_i | b, c_<i)], b untouched
        {:keys [branch scores]} (br/extend y' continuation)             ; teacher-forced; y' consumed
        kids (br/resample [x' branch] [1 1 0])]                         ; ancestors' states; parents consumed
    (br/dispose! (conj kids b))))
```

| Function | Meaning |
|---|---|
| `(open engine tokens)` | a branch that has read `tokens` (≥ 1; a BOS for an unconditional start) |
| `(fork b)`, `(fork b n)` | new branches with b's state; b stays live and unchanged |
| `(advance branches tokens)` | each branch one token further, in **one decode**; inputs consumed |
| `(extend b tokens)` | b advanced by many tokens, teacher-forced: `{:branch :scores}`; b consumed |
| `(score b tokens)` | the continuation's per-token log-probabilities; b live and unchanged |
| `(resample branches ancestors)` | child j gets `(branches (ancestors j))`'s state; all inputs consumed, the dead freed; a state is copied only for its second and later children |
| `(dispose! b-or-bs)` | free now; each must be live |
| `(live? b)`, `(info engine)` | liveness; sizes and counts (`:n-vocab :max-branches :n-ctx :live :free :closed?`, and llama.cpp's `:eog :bos :eos :model …`) |
| `(check-tokens engine tokens what)` | tokens as longs of the engine's vocabulary, or `:invalid-token` (a big integer too), for engine-specific functions |
| `slots/slot-of` | diagnostics: the slot holding a live branch's state (a Branch carries no slot: the engine's books do) |
| `(tokens b)`, `(position b)`, `(logprobs b)`, `(logprob b t)` | the branch's value |
| `llama/tokenize`, `llama/token-bytes`, `llama/detokenize`, `llama/kv-positions`, `llama/system-info` | the engine's own: the GGUF's tokenizer, raw byte pieces, diagnostics |

Engines never sample: they give log-probabilities only, so a token is drawn
by gen.random with its site's key (CLAUDE.md rule 2), and no engine adds
llama.cpp's sampler (owner, 2026-10-20).

**Approved (owner, 2026-10-20): the names.** DESIGN.md §6 listed open · fork · advance-batch ·
dispose · teacher-forced score · resample. Here `advance` is the batch
operation (one branch is a batch of one), `dispose!` carries a bang, and
`extend` is **an addition**: a teacher-forced multi-token advance returning
its scores, which `score` is (on a scratch copy) and which the token GF's
assess and a chat continuation need. `extend` shadows `clojure.core/extend`
inside `gen.llm.branching` (`:refer-clojure :exclude`); callers write
`br/extend`. Alternatives if the owner prefers: `advance-batch`, `feed`,
`teacher-force`. The engine constructor is `gen.llm.llama/engine`; the system
properties are `gen.llama.lib`, `gen.llama.log` and (tests)
`gen.llama.test-model`, following `gen.mlx.*`.

## Lifetimes

- **A branch is a value plus a state.** Its tokens and next-token
  log-probabilities never change and stay readable forever. Its state (a
  sequence of the KV cache) is what `advance`, `extend` and `resample`
  **consume**: they return successors that take over the sequence, and the
  inputs are no longer live. Every operation needs live branches; a consumed
  or disposed branch, or one of a closed engine, is an ex-info
  (`:reason :not-live` / `:closed`), never a silent use of another state.
  *Why:* llama.cpp's state is mutable; with mutable handles (`advance!`), a
  second reference to an advanced branch would silently read the new state.
  Consumption makes that misuse loud at the cost of a fork where an old state
  must be kept. **Approved (owner, 2026-10-20).**
- **dispose! is loud on a non-live branch** (GenMLX's `dispose-branch!` is
  idempotent). A consumed branch owns nothing, so disposing it must not free
  its successor's sequence, and is most likely a bug. **Approved (owner, 2026-10-20).**
- **Dropped branches are freed (a GC backstop).** A live branch that becomes
  unreachable is freed by the engine: a phantom reference per live branch,
  drained on the engine thread at the start of every operation (so `:live`
  counts dropped branches until then). When slots run short, or a read
  finds the KV cache full (a dropped branch's cells may be what is
  missing), the engine runs the GC, frees what it found, and tries once
  more before refusing (`:no-free-branches`, `:kv-full`). A consumed
  branch's reference is cleared, so collecting it frees nothing. *Why:* the token GF (genclj-0cvv) will keep
  branches in traces, which inference drops freely (a rejected MH proposal);
  explicit disposal there would need GenMLX's scopes. This is MLX's array
  lifetime (`MlxRuntime/adopt`) applied to sequences, and like it lives in
  the native membrane. The alternative is explicit disposal only, with
  exhaustion an error. **Approved (owner, 2026-10-20).**
- **Engines are closed explicitly** (`java.io.Closeable`, `with-open`);
  closing frees the model and context, and branches keep their values. An
  engine that becomes unreachable with all its branches is closed by a
  `Cleaner`: on JDK 25 its executor would otherwise shut itself down when
  collected and leave the model and context allocated (found by the
  review). Relying on that for gigabytes of weights is a last resort;
  close engines.

## Threading

One platform thread per engine (`gen.llm.slots`): every native call runs
there. The operations are synchronous: they submit a task and wait, so
callers on any thread (virtual threads included) are serialized; a call made
on the engine's own thread runs at once. A caller interrupted while it
waits still gets the result, its interrupt flag set again: the operation
runs to its end regardless and has consumed its branches, so throwing would
lose their successors. No futures in the API (DESIGN.md §8: the core is
synchronous). An autobatcher (genclj-mse0) coalesces concurrent
requests above this layer, into `advance` batches.

## The structure: one engine machinery, small per-engine operations

`gen.llm.slots` implements the protocol once over six operations an engine
provides (`Slots`: info, run, step, copy, clear, close), called only on its
thread, with arguments already checked. The llama.cpp engine is those six
(about 80 lines) and its construction; the tests' toy engine is another.
The MLX-native engine (later) can implement the same `Slots` (a copy is a
shared reference to an immutable cache) or the `Branching` protocol
directly (resampling as one lane gather).

## llama.cpp's limits

- **Process aborts.** `GGML_ASSERT` aborts the JVM: `seq_cp`/`seq_rm`/
  `seq_keep` with an out-of-range sequence id, a batch over `n_batch`, a
  cross-stream copy of a partial range. Nothing reaches llama.cpp unchecked:
  sequence ids are the engine's own slots, runs are split into `n_batch`
  decodes, a step holds at most `n_seq_max ≤ n_batch` tokens, and tokens,
  positions, capacity and context room are checked before any native call.
- **Sequences:** `LLAMA_MAX_SEQ` = 256, so `:max-branches` ≤ 255 (one
  scratch slot for `score`).
- **`kv_unified` is always on**: a fork is a metadata copy (38 µs with the
  thread handoff), forks share their prefix's cells, and `:n-ctx` counts
  cells shared by every branch and bounds each branch's length. A batch that
  finds no room is `:kv-full`, with llama.cpp's state restored.
- **Recurrent and hybrid models** (Mamba, Qwen3.5's GatedDeltaNet) cannot
  remove a sequence's end past `n_rs_seq` checkpoints. This engine never
  needs to: `score` reads into a scratch copy (a copy shares the recurrent
  state until written), and branches only grow. After a failed read the
  engine removes what it may have added (`seq_rm` from the read's start),
  whatever the failure: when nothing was added this removes nothing, and
  for recurrent memory too v0.7.0 counts that a success (its `seq_rm`
  refuses only a range starting inside the state, `0 < p0 <= cell.pos`,
  and a removal from pos leaves the state at pos - 1 alone; tested on
  Qwen3.5: a read whose first decode finds the cache full leaves its
  branch live and unchanged). Where something was added, a recurrent
  model's removal fails, and the slot is freed with its branch, a loud
  error naming it (`:lost-slots`), never read again: a full cache retries
  after a collection only when no slot was lost. `n_rs_seq` stays 0. Tested with
  Qwen3.5-0.8B (`qwen35`, hybrid): the whole contract passes, so forks,
  resampled children and scratch scores of the shared recurrent state are
  right, and advancing a fork leaves its parent's state alone. A hybrid
  slot's `seq_pos_min` is its recurrent state's, the last position. A
  removal is trusted only if the slot then ends where it should
  (`seq_pos_max` = pos - 1). The `:lost-slots` path is tested: a hybrid
  fork whose extension finds the cache full after two of its decodes cannot
  be rolled back (v0.7.0's recurrent `seq_rm` refuses), so it is freed,
  loudly; on an attention model the same failure leaves the fork as it
  was. So is a real fatal decode (below).
- **Sliding-window models** (Gemma): `swa_full` by default, so a branch
  keeps every token a fork may need. Tested with Gemma-3-270M (window 512;
  llama.cpp logs "using full-size SWA cache"): the contract passes, and
  so it does past the window (genclj-vdxm): forks taken at 600 and 1,000
  tokens, a resample there, and scores of a continuation from 472 across
  512 give fresh reads' distributions, within a band measured for long
  branches (0.6 nats, Numerics). A planted copy keeping only the window's
  positions, as if a sliding-window model needed no more, fails it by 4
  to 39 nats: Gemma 3's full-attention layers need the whole history.
  - *Without `swa_full`* (measured, not a test: it is llama.cpp's
    behaviour), the window cache has pad(n_swa · n_seq_max + n_ubatch,
    256) cells, so a branch longer than that keeps only its latest
    positions there (positions 840 to 2,599 of 2,600, in 1,792 cells), and
    llama.cpp reuses a cell only when its one sequence has moved past it.
    Below that size nothing changes: forks, resamples and scores at 600 and
    1,000 tokens give the same values bit for bit. Above it, a fork pins the
    window cache: the copy shares every cell, none is reusable, and the
    next decode of the fork finds no room (`:kv-full`; on Metal at 4,000
    tokens in 3,072 cells, on the CPU at 2,600 in 1,792). Forks thus break
    loudly, not silently. A read that fails after overwriting cells still
    inside the window could not be undone; the one probed (a read of 48
    aborted after its first decode, at 2,600 tokens) found free cells and
    was undone exactly, as with `swa_full`.
- **Refused:** encoder-decoder, diffusion and memoryless (embedding) models.
- **Decode failures** are ex-infos (`:kv-full`, `:aborted`,
  `:invalid-batch`, `:decode-failed`) carrying llama.cpp's last logged
  error, and a row of NaN or infinite logits is one too. A fatal one is
  tested through llama.cpp's abort callback (genclj-vdxm), set by the
  test alone through `llama_set_abort_callback` (found in libllama; the
  binding leaves it out): status 2, after which llama.cpp keeps the
  ubatches it finished and removes the aborted one. Only the CPU backend
  checks the callback (once before each graph node; ggml-metal enqueues
  its first two command buffers regardless, and checks only when
  capturing), so the test's engine is CPU-only with 16-token decodes
  (below the 32 at which llama.cpp hands a CPU model's work to the GPU),
  and the abort lands at a counted check: the first, or the first of a
  read's second decode, counted on a decode of the same size (537 checks
  for Qwen3-0.6B, 785 for Qwen3.5, the same every time). An attention
  model's branch is restored exactly (its next step equals, bit for bit, a
  fork's taken before); a hybrid one's is lost, named and freed, even when aborted at
  once, since llama.cpp moves the recurrent state before it computes.
- **Version pinning:** the binding passes `llama_model_params` and
  `llama_context_params` by value, whose layouts change between releases, so
  `LlamaRuntime` refuses a libllama whose `llama_version()` is not "0.7.0".
  `native/build-llama.sh` builds with `LLAMA_BUILD_IS_DEV=OFF` (a dev build
  reports "0.7.0-dev"). jextract's layouts are its host's; these structs hold
  only fixed-width fields, so the macOS generation serves every LP64
  platform (not Windows).

## Numerics (measured)

Log-probabilities are normalized in double and stored as floats (their sum
is 1 within 1e-6, a derived band); scores are doubles.
`bench/gen/llama_noise.clj` compared every pair of computations the contract
compares, over 20 prompts and batches of 1 to 13 (M2 Max, Metal; |Δ log p|
over the reference's 64 most likely tokens). Qwen3-0.6B Q8_0, on which the
band was set:

| Comparison | n | median | p99 | max |
|---|---|---|---|---|
| fork + advance vs fresh prefill | 560 | 3.8e-3 | 2.8e-2 | 5.2e-2 |
| batched advance vs alone | 560 | 2.4e-3 | 1.2e-2 | 2.4e-2 |
| extend's branch vs fresh prefill | 20 | 8.0e-3 | 1.8e-2 | 1.8e-2 |
| score / extend vs step by step | 480 | 2.7e-4 | 5.6e-3 | 6.5e-3 |
| the same tokens re-read, the first read still live | 20 | 4.6e-3 | 2.5e-2 | 2.5e-2 |
| **the same read from the same state** | 20 | 0 | 0 | 0 |

The same computation is deterministic, bit for bit; the differences come
from batch shapes and the unified cache's layout: ggml-metal's
matrix-matrix kernels (prefills, larger batches) round activations to half
precision, the single-token decode's matrix-vector kernel does not. The
worst prompt was a chat-formatted one (5.2e-2); ordinary text stayed under
1.8e-2. The tests' band is **0.1 nats**, twice the measured maximum; three
mutations of the engine failed it by 3.8 to 18 nats. The two other models
stay inside it, with the same determinism:

| Model | fork vs fresh, max | batched vs alone, max | overall max | same state |
|---|---|---|---|---|
| Qwen3.5-0.8B Q8_0 (hybrid) | 1.6e-2 | 1.0e-3 | 1.6e-2 | 0 |
| Gemma-3-270M Q8_0 (SWA) | 2.3e-2 | 4.7e-3 | 3.0e-2 | 0 |

Branches past a sliding window (`bench/gen/llama_long_noise.clj`: the
contract's own comparisons, for four long texts at 600, 800, 1,000 and
1,400 tokens and a continuation across 512; the engine of the long-branch
test) are noisier for Gemma:

| Model | fork vs fresh, max | child vs fresh, max | overall max (n 2,200) |
|---|---|---|---|
| Gemma-3-270M Q8_0 | 0.28 | 5.0e-2 | 0.28 |
| Qwen3-0.6B Q8_0 | 1.6e-2 | 3.0e-2 | 3.0e-2 |
| Qwen3.5-0.8B Q8_0 | 8.3e-3 | 8.9e-3 | 8.9e-3 |

Gemma's 0.28 is one comparison (the next largest is 5.0e-2): a fork at
1,000 tokens of a code text, after a newline whose successor has log p of
-6e-6, in the deep tail of the reference's 64 tokens (log p of -10 to
-22), where a fresh read's prefill and a one-token step round
differently; the fork and the branch advanced alone agree there to 1e-3.
The long-branch test's band is twice it, 0.6 nats; the short-prompt band
of 0.1 does not hold for Gemma's long branches.

The engine's paths agree exactly with same-shape paths: a resampled
child equals the same steps without resample bit for bit (from 600
tokens on; at 100, 1e-4 to 2e-3 from the cache's layout), and `score`
equals `extend` to 1e-8 (the float rounding of the last score). The
differences above are kernel noise between shapes.

**On the Mac mini M4** (2026-10-21, Metal, 10-core GPU; M3's development
machine since genclj-m54h), both benches again, with the same models
(copied from the MacBook, their sha256 checked). Every same-state read is
bit-identical, and both bands hold:

| Model | short prompts, max (M4 / M2 Max) | long branches, max (M4 / M2 Max) |
|---|---|---|
| Qwen3-0.6B Q8_0 | 4.3e-2 / 5.2e-2 | 6.9e-2 / 3.0e-2 |
| Qwen3.5-0.8B Q8_0 | 1.5e-2 / 1.6e-2 | 1.1e-2 / 8.9e-3 |
| Gemma-3-270M Q8_0 | 2.7e-2 / 3.0e-2 | 0.15 / 0.28 |

The short-prompt band (0.1) is 2.3 times the M4's largest difference, the
chat-formatted prompt's again; the long-branch band (0.6) 3.9 times, at
Gemma's same deep-tail case (the code text at 1,000 tokens, token 107).
Only Qwen3's long branches moved up, to 6.9e-2 at one case (the prose
text at 1,400 tokens), still inside both bands. The medians are those of
the M2 Max (2e-3 to 1e-2). The bench's costs ran beside another agent's
JVM tests, so they are not recorded as the M4's.

**CPU builds are far noisier across shapes.** On the M2 Max's CPU backend
(16-token decodes, Q8_0's repacked matrix kernels), a fresh read and the
same tokens stepped differ by up to 0.46 nats (Gemma at 600 tokens; Qwen3
0.17), while same shapes agree exactly. CPU and CUDA builds need bands
of their own: measure them with the benches before using these.

## Costs (M2 Max, Metal, Qwen3-0.6B Q8_0, means)

Normalizing a row of logits on the host (a log-sum-exp in double over the
vocabulary, then the log-softmax) takes 1.25 ms: for 15 rows, most of what
an advance cost. Since the quality review (N1) each engine normalizes a
decode's rows on a pool of its own (four platform threads on this 8+4-core
machine, half the cores up to four), its thread waiting; the rows'
pointers are still taken on the engine's thread (a native call), and stay
valid until the next decode, which the pool's end precedes. Measured old
and new in one session, two rounds each:

| Operation | before (ms) | with the pool (ms) |
|---|---|---|
| advance, 1 branch | 7.4–9.5 | 5.8–5.9 |
| advance, 4 / 8 / 15 branches (one decode) | 10.4 / 16.1 / 27.9–28.5 | 7.2–7.7 / 8.1–8.5 / 13.4–13.5 |
| score, 16 tokens | 25.5–25.9 | 13.5–13.6 |
| host normalization of one row (one thread) | 1.25–1.28 | 1.25–1.28 |

An open (an 8-token prefill) takes 7 to 10 ms and a fork with its dispose
0.07 ms, either way (separate sessions; a session's state moves the
one-branch numbers by a few ms, hence the alternation above).

**The heap.** A branch's log-probabilities are one float array: 607,760
bytes for Qwen3's vocabulary of 151,936, 1,048,592 for Gemma 3's 262,144.
G1 makes an object over half a region humongous: with a 2 GB heap (1 MB
regions) every row is (the bench, under `-Xlog:gc+humongous`, logged 903
humongous rows, all reclaimed eagerly as reclaim candidates, at no
measurable cost: 13.8 ms for an advance of 15); with the default heap of a
64 GB machine (16 GB, 8 MB regions) none is, and with 4 GB (2 MB regions)
Gemma's are. Forks and resampled children share their parent's array.

## For the next beans

- **genclj-mse0 (LM protocol).** A trie node can hold a live Branch: its
  `tokens` are the node's path, LRU eviction is `dispose!`, `fork` extends a
  cached prefix, and the autobatcher's batch is one `advance`. Next-token
  log-probabilities are `logprobs` (a float[] copy). The byte vocabulary is
  `llama/token-bytes` over every token; EOS/EOG are in `(info e)`. Those
  vocabulary functions are the engine's for now, for the LM protocol to wrap.
- **genclj-0cvv (token GF).** (1) *Scores depend on the batch shape* by up to
  5e-2 nats a token (median 2e-3 to 4e-3): a sequence's log-probability
  computed teacher-forced (assess) and step by step (simulate) differ by the
  sum, ~0.1 to 1 nat over a few hundred tokens. Laws comparing them need a
  band, or the GF should score each token once and keep it (GenMLX's
  "score-once" ledger, for the same reason). (2) *Slots are few:* a
  fork-at-each-site ledger (GenMLX's branched GF) costs a sequence per token,
  so 255 at most for all live traces; resume regenerate from forks at a few
  sites, or re-prefill from the trace's tokens (deterministic for a given
  state). (3) The GC backstop lets traces own branches.
- **genclj-zztc (shared memory budget).** `(:bytes (:model (info e)))` is the
  weights' size. llama.h v0.7.0 reports no KV-cache size: compute it from
  the model's dimensions (cells × layers × KV heads × head size × 2 × 2
  bytes for f16, for plain attention), or read llama.cpp's info-level log
  line `llama_kv_cache: size = … MiB (… cells, … layers, …)` (and its
  per-buffer "KV buffer size" lines) through the log callback.
- **Not done here:** Linux CUDA and CPU builds (written, not run), MLX and
  llama.cpp in one test JVM (the spike showed it), the 8B and 30B models,
  and a fatal decode on Metal (only the CPU backend honours an abort, and
  no other deterministic way exists without a `GGML_ASSERT`).

## The review

An independent review of the branch (an agent, 2026-10-20: fuzzing the toy
engine's bookkeeping over random operation sequences, with and without
injected failures, and reading llama.cpp's source) found no path to a
`GGML_ASSERT` or an FFM fault, and these, all fixed with a test that fails
without the fix:
1. A removal after a failed decode was trusted when `seq_rm` returned true;
   now the slot must also end at pos - 1 (in v0.7.0 the recurrent `seq_rm`
   already refuses the removal the review feared; the check guards other
   versions).
2. The GC backstop covered slots but not KV cells: a dropped branch's cells
   could make a read fail `:kv-full` with nothing freed. Now a full cache
   collects and retries, and every operation drains first.
3. An unreachable engine leaked its model (JDK 25's self-closing executor).
   Now a `Cleaner` closes it.
4. resample leaked slots if a copy threw. Copies do not fail by the
   `Slots` contract, but resample now frees what it was placing.
5. No test read more than `n_batch` tokens. Now a 16-token batch reads a
   25-token prompt and a 40-token continuation against step-by-step scores,
   and a read fails midway (above).
6. `(fork b 0)` of a dead branch returned `[]`; now it is loud.
7. An interrupted caller lost an operation's result; now it waits it out.
8. Docs said drains ran before every operation; now they do.
9. Smaller: `slots-info` ran off the engine thread; a second sizing failure
   in `token-bytes`/`detokenize` was an `IndexOutOfBoundsException`;
   `:max-branches` is checked against `llama_max_parallel_sequences`; two
   contract checks compared an array with itself and are gone.

## The quality review

A second review (an agent, 2026-10-20, against the owner's bar: beautiful,
idiomatic Clojure and steady JVM performance) found nine things that had
to change and twelve that would be nicer, all applied:
- **Bugs.** A read that lost its slot was retried after a collection,
  leaving a live branch whose state was not its value (M1: now only a
  read whose slots were restored is retried); a construction interrupted
  while the model loaded leaked it (M2: it waits as every call does); two
  error paths had no test (M3: the toy's `:lossy` mode, and resample past
  capacity in the misuse cases); token checks coerced to long, so a big
  integer was an `IllegalArgumentException` (M6); and error data without
  or with a borrowed `:reason` (M9).
- **Structure.** A Branch carries no slot (M4: the engine's books hold it,
  `slots/slot-of` shows it); the books' transitions are pure functions
  with one effectful `forget!` (M5); the construction is named steps
  (M7); a read is one loop that always restores (M8, above); resample's
  placement is a pure, tested function (N5); one close path through the
  `Cleanable` (N4); `CompletableFuture/join` for the uninterruptible wait
  (N3); the batch's views made once (N2).
- **Performance.** Rows normalized on the engine's pool (N1, above); the
  contract's `top-tokens` one packed sort, 7 ms rather than 50 (N8).
- **Loudness.** Unknown engine options and `gen.llama.log` levels are
  errors (N10); llama.cpp's functions refuse another engine
  (`:not-a-llama-engine`, N12); the jextract recipe checks every function
  was emitted (N9).

## Models tested

All on the owner's MacBook Pro (M2 Max, 64 GB, macOS 27, Metal, Oracle
GraalVM 25.0.4), under `~/gen-clj-spikes/mguard.sh`, one model-holding JVM
at a time; `gen.llm.branching-test` and `gen.llm.llama-test` pass with each
(23 tests, 122 assertions, after the review's fixes). Each sha256 below is Hugging Face's, checked on
download.

| Model | Repository / file | sha256 | Kind |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | `Qwen/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q8_0.gguf` (the spike's download) | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | attention (`qwen3`) |
| Qwen3.5-0.8B Q8_0 | `ggml-org/Qwen3.5-0.8B-GGUF` / `Qwen3.5-0.8B-Q8_0.gguf` (Qwen publishes no GGUF of it) | `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f` | hybrid GatedDeltaNet (`qwen35`) |
| Gemma-3-270M Q8_0 | `ggml-org/gemma-3-270m-GGUF` / `gemma-3-270m-Q8_0.gguf` | `e00cf79514204dfe2f4d6943f277ffea7fd8a4c8e955b4b7a869cfc157694881` | sliding window 512 (`gemma3`) |

Both new models load with llama.cpp v0.7.0. Qwen3.5's GGUF carries an MTP
head that llama.cpp does not load into a default context (it logs the
unused `blk.24.*` tensors as warnings).
