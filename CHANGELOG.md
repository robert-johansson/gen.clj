# Changelog

What changed in gen.clj that code built on it (memo.clj, GenGDT) must know
about. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
keep the promise below. gen.clj is a git dependency: a release is a tag,
taken as `{:git/url … :git/tag "v1.0.0" :git/sha …}` (ext/laws at the same
tag, with `:deps/root "ext/laws"`); between releases a dependent may pin
a commit on main. Each entry names the commits on main that carry its
change and the bean that tracks it.

## What 1.x promises

From DESIGN.md §9, "The 1.0's shape" (owner, 2026-10-25):

- **Kept through 1.x:** the names, arities, result-map keys and documented
  semantics (the GFI and its laws) of the stable namespaces. Minor
  releases add to them.
- **Not promised across releases:** the exact draws for a key. Their
  distributions are promised.
- **Removals** wait for 2.0, after a deprecation cycle: `:deprecated`
  metadata and a docstring note, nothing printed.
- **Not promised:** the text of error messages (an error stays loud and
  keeps its meaning; its wording may improve), and anything in the
  experimental and internal tiers.
- **Three tiers,** marked on each namespace (`{:api :stable | :experimental
  | :internal}`). Experimental namespaces (gen.llm.* among them) ship
  outside the promise. Internal ones are not for dependents. A var of a
  stable namespace may carry its own `{:api :experimental}` (gen.array's
  `scan`): it too is outside the promise.
- **gen.laws:** a minor release may add a law to a set or change
  `laws/close?`'s tolerances, each with a line here and its reason;
  renaming or removing a law waits for 2.0 (owner, 2026-10-25).

In the entries, **Breaking** marks a change that can make working code
fail (throw, or read another shape or result), and **Draws** one that gives
other values for the same key, from the same distribution.

## [Unreleased]

### Changes

- **Added (experimental): `gen.array/load-safetensors`, and Hugging Face
  checkpoints in `qwen3/load` and `qwen35/load`** (genclj-9pn4,
  PROVISIONAL).
  - `(array/load-safetensors path)` reads a .safetensors file, an index
    (model.safetensors.index.json, every shard it maps) or a checkpoint's
    directory: `{:metadata :tensors :types :files}`, as `load-gguf`'s.
    F32, I32, I64, BOOL and U32 load as their dtypes, and F64 on the JVM
    backend (MLX 0.32.4's loader refuses a file holding F64: loud there);
    F16 and BF16 as float16 and bfloat16 on MLX, float32 on the JVM backend
    (the same values). `:select`, a predicate of names, loads only those
    tensors (the rest are never read).
    Other dtypes (I8, U8, the float8 types, …), an int64 past 2^53 on the
    JVM backend, and malformed files (a header that is no JSON object of
    entries, tensors that leave a gap or overlap, a shard missing or
    holding other tensors than its index says) are loud. MLX reads by
    `mlx_load_safetensors`, the JVM backend by a host reader
    (gen.array.safetensors), with a JSON reader of its own
    (gen.array.json, internal): no dependency. `ILoading` gains
    `-load-safetensors`.
  - `qwen3/load` and `qwen35/load` take a checkpoint (its directory, or
    its .safetensors file or index) where they take a GGUF file, read as
    the GGUF llama.cpp's converter writes from it, so a checkpoint and
    its GGUF give one model: config.json (Qwen3; Qwen3.5 and Qwen3.6's
    text_config, dense or MoE) as the GGUF's metadata, the tensors renamed,
    and the converter's transforms done: Qwen3-Next's +1 on the RMSNorm
    weights (but the gated norm's) in float32, ssm_a = -exp(A_log) in
    float32, the value heads from grouped to tiled order, the conv kernel
    squeezed, one matrix an expert stacked. A tied output drops a stored
    lm_head copy (Qwen3-0.6B's), as transformers does. The vision tower
    and the MTP block are never read.
  - MLX's quantized checkpoints (mlx-lm's, mlx-community's): a matrix's
    uint32 words, scales and biases load as `load-gguf`'s quantized
    matrices, at config.json's quantization (its bits and group size, and
    a matrix's own entry). mlx-lm's saved layout (language_model.model.,
    the conv kernel [C K 1], the +1 already in its norms) is told by its
    conv kernels. Another quantization (GPTQ, AWQ, mxfp4, …) is loud, as
    are the experts' fused gate_up_proj and the configurations llama.cpp
    does not compute (sliding windows, RoPE scaling, …).
  - Checked: each tiny random checkpoint of test/fixtures/safetensors
    (generate.py: test/fixtures/gguf's models, the converter undone) gives
    its GGUF's config, weights and logits exactly on the JVM backend:
    Qwen3, Qwen3.5 with 2 key heads of 2 value heads each, Qwen3.5-MoE,
    and their MLX 8-bit checkpoints, one sharded in three. Planted bugs
    (the +1 missed, A_log's sign, the heads left grouped, mlx-lm's layout
    read as Hugging Face's, the experts reversed) move the logits 4.7 to
    9.4 in a logit. On MLX the same within float32's band (MLX's exp
    rounds ssm_a an ulp apart).
  - Real checkpoints against their GGUFs in this forward, on Metal in
    float32 (gen.llm.models.checkpoint-real-test, 4 prompts of 32
    tokens): Qwen/Qwen3.5-0.8B against ggml-org's BF16 GGUF, every weight
    equal, max |Δ log p| 2.7e-5 over the GGUF's top 64, argmax 128 of 128
    (the +1 missed moves it 25.6; A_log's sign gives NaN); Qwen/Qwen3-0.6B
    against the F16 GGUF, 4.4e-5, argmax 128 of 128 (weights apart by
    float16's subnormals only, 3e-8).
- **Changed (experimental): `gen.array/load-gguf` loads BF16** (genclj-9pn4,
  PROVISIONAL), as bfloat16 on MLX (by the host reader: MLX's loader is
  not trusted with it) and float32 on the JVM backend, where it was
  `:unsupported`; test/fixtures/gguf/unsupported-bf16.gguf is now
  bf16.gguf, with its values (bf16.edn).

- **Added (experimental): K-quants in `gen.array/load-gguf`** (genclj-kqxr,
  PROVISIONAL): Q4_K, Q5_K and Q6_K, and Q5_0 and Q5_1 (Q4_K_M's
  fallbacks for rows not of 256 values), each value the file's exactly
  (llama.cpp's `dequantize_row_*`, bit for bit), so Q4_K_M and its kin
  load whole, each tensor at its own bits.
  - Q4_K and Q5_K load as MLX's affine quantization at 4 and 5 bits, a
    sub-block of 32 a group: the scale d·sc and the bias -(dmin·m) in
    float32, as llama.cpp computes them (both products exact). Q5_0 and
    Q5_1 at 5 bits, groups of 32.
  - Q6_K loads as float32 arrays (d·sc·(q - 32), exact): its sub-blocks of
    16 with their own scales fit no affine group of MLX's (32, 64, 128).
    It is 4.9 times its file's bytes in memory; a model's Q4_K and Q5_K
    take 1.33 and 1.27 times theirs (float32 scales and biases).
  - On MLX, a file holding one of these types loads by gen.array.gguf's
    host reader (every tensor of it, made MLX's arrays), not by
    mlx_load_gguf, which would turn them into float16; other files load
    as before. The host reader decodes a large tensor in ranges on the
    common fork-join pool: on the M4, 5.4 GB/s of Q4_K, 7.2 of Q8_0 and
    2.6 of Q6_K (bench/gen/gguf_bench.clj, Qwen3.6-35B-A3B-Q4_K_M's
    tensors); a whole Qwen3-0.6B Q4_K_M loads on MLX at 1.2-1.4 GB/s of
    its file (mlx_load_gguf reads its Q8_0 at 3.4).
  - Checked against llama.cpp on Metal: Qwen3-0.6B quantized as Q4_K_M by
    the pinned llama.cpp's quantizer, gen.llm.models.qwen3's forward
    within 0.231 nats of llama.cpp's (max |Δ log p| over its top 64, 200
    comparisons; argmax equal in all), as large as llama.cpp's own spread
    on that file (0.226; 0.052 on Q8_0). Qwen3.5-0.8B as Q4_K_M: 0.414,
    llama.cpp's own 0.460.
  - Still `:unsupported`: Q2_K, Q3_K, the IQ types and the rest (BF16 loads
    since genclj-9pn4, above).
    `gen.array.protocols/ILoading`'s contract names the new types.
  - Test files: test/fixtures/gguf/kquants.gguf (random blocks) and
    qwen3-small-f16.gguf with its Q4_K_M by llama.cpp's quantizer
    (generate.py's `--llama-lib`); unsupported-q4_k.gguf is now
    unsupported-q3_k.gguf.

- **Added (experimental): paged KV caches and paged attention**
  (genclj-1daf; route A of the spike, PROVISIONAL: mlx_fast_metal_kernel
  and MLX's donated scatter, no C++ of gen.clj's).
  - `gen.array/paged-write`, `copy-blocks` and `paged-attention`, in
    vLLM's layouts: key pools [NB Hk D/x BS x], value pools [NB Hk D BS],
    block tables [B MB] and lengths [B]. Each is a composite of gen.array's
    ops (put-along-axis, take, sdpa), its own reference; on MLX's Metal
    paged-attention runs kernels-community's paged_attention kernel (v1;
    ported per THIRD-PARTY-NOTICES, Apache-2.0, its text in
    LICENSE-APACHE-2.0), and the writes are MLX's scatter on the pools'
    block and offset axes, in place under `array/compile`'s `:donate`.
    `p/IFused` gains `:paged-write` and `:paged-attention`.
  - `gen.llm.paged`: a block allocator and block tables as values, forks
    sharing blocks and copying a shared partial block on write.
  - `gen.llm.models.qwen3`: `paged-pools`, `paged-step`,
    `compile-paged-step`, `paged-cache`, `paged-entries`.
  - `gen.llm.array-engine`: `:paged? true` runs one pool of blocks for all
    branches (options `:block-size` 16, `:pool-blocks` 128, growing by
    doubling, `:min-rows` 1), so token SMC's steps stack no rows; `stats`
    reports its counts and `blocks` its books. A failed paged step loses
    every branch (`:lost-slots`), its pools having been donated.

- **Added (experimental): continuous batching** (genclj-b3ow, PROVISIONAL:
  concurrent callers' results then depend on how their calls were
  gathered, within the engine's band).
  - `gen.llm.slots`: an optional protocol `Batching` (`slots-forward`),
    for Slots whose `slots-info` has `:batching {:wait w}`. Their engine
    admits advances, extends and opens as requests and reads them in
    forwards over every read in flight, waiting before a forward for the
    callers the last one answered (at most w of its time); an operation
    on a branch a request holds waits for it. `slots/schedule` gives the
    scheduler's counts. Every other engine runs its operations one at a
    time, in order, as before: the engine's thread is now a task loop of
    gen.llm.slots's own in place of a single-thread executor.
  - `gen.llm.array-engine`: `:continuous? true` (with `:paged? true`;
    `:wait` 0.25) reads every one-token read in flight and a chunk of
    the oldest longer read (an open's or extend's prefill, up to
    `:n-batch` tokens a forward) in one compiled forward. `stats` reports
    its forwards, decode and chunk rows.
  - `gen.llm.models.qwen3`: `paged-batch-step` and
    `compile-paged-batch-step`, a forward of decode rows (paged
    attention) and one sequence's prefill chunk (its blocks gathered)
    over paged pools, the logits of the rows read only, and the scores
    of their targets.
  - Measured on Metal (Qwen3-0.6B Q8_0): 8 concurrent completion clients
    302 tokens/s in float16 against 90 in the paged mode and 103 through
    llama.cpp (175 against 82 and 92 in float32); a lone caller as
    before.

- **Changed (experimental): the paged mode copies a block only for a
  second writer; Qwen3.5 runs paged** (genclj-6qb6, genclj-1ovd).
  - `gen.llm.paged`: an allocator records each block's written offsets
    (`:written`, `paged/written`). `prepare` appends to a shared block
    filled in part in place when no holder has written past the
    sequence's end, and copies it only otherwise: an LM's kept entry and
    the fork that goes on from it share the block, and only a second
    writer (a resampled duplicate, or the entry itself written later)
    copies. The values read are the same; the copies, and `stats`'
    `:copied-blocks`, are fewer: token SMC at 16 particles copied 44
    blocks where it copied 1012 (Qwen3-0.6B on Metal: 680-703 tokens/s
    in float32 against 604, 742-758 in float16 against 713).
  - `gen.llm.models.qwen35`: `paged-pools`, `paged-states`,
    `paged-cache`, `paged-entries`, `paged-step` and
    `compile-paged-step`: the attention layers' keys and values in paged
    pools, the GatedDeltaNet layers' states a row each, the step's
    layers the two merged by layer.
  - `gen.llm.array-engine`: `:paged? true` runs Qwen3.5 and Qwen3.6
    (`:continuous?` stays Qwen3's, loud otherwise). A branch's recurrent
    states are a row of a group, as the static mode's: a lane steps in
    place, any other step stacks its rows' states (`stats`'
    `:stacked-steps`).
  - `p/IFused` gains `:put-blocks` (gen.array.paged's block write): on
    MLX one scatter of the blocks as slices.
  - The donated writes' cost of 8 ms a GB (genclj-1daf) was the GC that
    genclj-c7uf removed: on Metal a write into pools of 0.14 to 3 GB now
    takes 0.3-0.7 ms whatever their size.

- **Added (experimental): LoRA adapters, `gen.llm.lora`** (genclj-yc3b,
  genclj-p4mk): low-rank adapters trained beside a model's frozen, dense
  or quantized, weights, and served by llama.cpp.
  - `(lora/init m opts key)` is an adapter `{:config {:architecture :rank
    :alpha :dropout} :params {layer {target {:a :b}}}}`, its `:params`
    gen.learning's store: A [rank in] uniform on ±1/√in from a key per
    site, B [out rank] zeros. Options `:rank` (8), `:alpha` (16),
    `:dropout` (0), `:targets` and `:layers` (all). `lora/targets` maps
    each architecture to its projections and llama.cpp's names for them:
    Qwen3's `:q :k :v :o :gate :up :down` (default `[:q :v]`); Qwen3.5's
    and Qwen3.6's below. A target adapts the layers that have it; one of
    another family, or that none of the layers has, is loud.
  - `(lora/adapt m adapter opts)` is m with the adapter beside its
    weights: x·wᵀ + s·(x·Aᵀ)·Bᵀ, s = `:scale`·alpha/rank (llama.cpp's
    convention), every gen.llm.models.qwen3 function and
    `gen.llm.array-engine` taking it as a model, differentiable with
    respect to the adapter (quantized-matmul's gradient is with respect
    to x). `:key` draws the dropout of a training step.
  - `(lora/merge m adapter)` folds it into dense weights.
  - `(lora/save-gguf adapter path)` writes llama.cpp's LoRA GGUF
    (`blk.N.attn_q.weight.lora_a`/`_b`, F32, `general.type` adapter,
    `adapter.lora.alpha`), which `gen.llm.llama/adapter` loads;
    `(lora/load-gguf path)` reads one. Checked against llama.cpp on
    Qwen3-0.6B (fenrir's CUDA): gguf-py's adapter and ours within the
    platform's band, planted convention errors (A and B swapped or
    transposed, the scale alpha, the names) refused or off by 3 to 17
    nats.
  - `gen.array.gguf/write-gguf` (internal): a GGUF writer beside the
    reader, writing gguf-py's bytes.
- **Added (experimental): training for Qwen3.5 and Qwen3.6**
  (genclj-8h0l): LoRA through the hybrid layers, their training loss,
  and their adapters for llama.cpp.
  - `lora/targets` for qwen35 and qwen35moe: the gated attention's `:q`
    (attn_q: each head's query and output gate) `:k :v :o`;
    GatedDeltaNet's input projections `:qkv` (attn_qkv), `:z`
    (attn_gate), `:beta` (ssm_beta), `:alpha` (ssm_alpha) and its output
    `:out` (ssm_out); the feed-forward's `:gate :up :down` (qwen35), or
    the shared expert's `:gate-shexp :up-shexp :down-shexp` and the routed
    experts' `:gate-exps :up-exps :down-exps` (qwen35moe: an adapter per
    expert, A [experts rank in] and B [experts out rank], its term taken
    by the tokens routed to the expert, llama.cpp's `build_lora_mm_id`;
    `lora/delta-gathered`). `lora/init`'s default for both: `[:q :v
    :qkv]`, the experts only when named. `save-gguf` and `load-gguf` use
    these names, the experts' stacks as 3-D tensors.
  - `lora/adapt` refuses an adapter of another architecture than the
    model's (`:invalid-args`), as llama.cpp does; `init`, `adapt`,
    `save-gguf` and `load-gguf` refuse an architecture LoRA does not know
    (`:unsupported`); `merge` reaches the MoE's nested weights (a stack
    per expert).
  - `qwen35/log-probs` and `qwen35/nll`, with `:mask`, `:checkpoint?`
    and `:chunk`, as Qwen3's: their gradients go through gated-delta's
    composite (MLX's Metal kernel has no VJP, and is not used under
    gradients).
  - gen.llm.models.common's `linear` (and so every qwen35 projection)
    takes an adapted weight. The MoE's routing indices carry no gradient
    (`stop-gradient`), which MLX's native gradients need.
  - Checked: the tape's gradients against central differences on the
    tiny models (2.5e-7 relative; 5.8e-7 over 70 tokens, the chunked
    rule), MLX's (tape and native, Metal) against the JVM's within
    1.8e-5 of the largest; on Qwen3.5-0.8B-Q8_0 (Metal), an adapter on
    every target agrees with llama.cpp's reading of its file within
    0.0039 nats (moving log p by 1.9), the tiny MoE's routed and shared
    experts within 0.025; planted convention errors (A and B swapped, the
    layers off by one, names swapped, alpha/rank, the scale alpha) are
    refused or off by 1.7 to 17.6 nats; five AdamW steps on 37 tokens
    take the NLL from 162.4 to 1.0, and llama.cpp serves the trained
    adapter within 0.0053.
  - The rule's gradient through the composite costs a fifth to a quarter
    of a training step on Metal at 256 to 1024 tokens
    (bench/gen/gated_delta_bench.clj).
- **Added (experimental): supervised fine-tuning, `gen.llm.sft`**
  (genclj-stpo): a LoRA adapter trained on chat examples.
  - `(sft/example tokenizer ex opts)` tokenizes `{:messages [...]}` or
    `{:prompt :completion}` with Qwen's chat template, giving `{:tokens
    :mask}`. The tokens trained on are the assistant turns' content and
    their end token `<|im_end|>` (`:end-token?`, default true; mlx-node
    trains no end token). Tokens are matched to the text's bytes, so a
    token crossing a trained span's edge is loud (`:straddling-token`),
    as is an example with nothing to train on (`:nothing-trained`).
  - `(sft/batch examples opts)` right-pads (`:pad`) and truncates
    (`:max-length`, keeping the start): `{:tokens :mask :count}`.
  - `(sft/nll m batch)` and `(sft/evaluate m examples)` give the summed
    and mean NLL per trained token. The family's nll is checkpointed and
    chunked (`:checkpoint?` true, `:chunk` 16384).
  - `(sft/trainer adapter optimizer key)` is a value. `(sft/step m
    trainer batches)` adds the batches' summed losses and gradients, then
    divides once by their trained tokens, and steps any gen.learning
    optimizer. It returns `[trainer' {:step :loss :tokens :grad-norm}]`.
    A loss or gradient that is not finite is loud (`:non-finite`); the
    step is never silently skipped.
  - `(sft/train m trainer examples opts)` runs epochs in an order drawn
    from the trainer's key, with `:accumulate`, `:held-out` evaluations
    and `:save-dir` checkpoints. Step n draws its dropout from
    `(fold-in k n)`.
  - `(sft/save-checkpoint trainer dir)` writes `dir/step-<n>/`, holding
    `adapter.gguf` (llama.cpp's) and `trainer.edn` (gen.serialize's,
    exact). The directory is renamed into place at once.
    `load-checkpoint` and `latest-checkpoint` read it back. Resuming
    gives the uninterrupted run bit for bit (the JVM backend, MLX's
    CUDA and CPU).
  - MLX's memory limit (genclj-az5s) is left to the user, a process-wide
    setting. Under the default (half of RAM), MLX keeps queued kernels'
    buffers and most of the checkpoints' saving is lost. The docstring
    and `bench/gen/sft_bench.clj` give the limit to set: what MLX holds
    before a step plus about 1.25 times a step's own peak. Measured on
    fenrir's CUDA (Qwen3-0.6B, rank 8 on the last 8 blocks,
    checkpointed), the peak of a step of 4×256 tokens was 6.1 GB at the
    default limit and 1.8 GB at that limit, in the same time.
  - Checked on Qwen3-0.6B (`gen.llm.sft-llama-test`, fenrir's CUDA). 24
    steps teach it a JSON reply format it does not use: held-out NLL
    3.61 → 0.093 nats a token, greedy replies in the format 0/16 → 15/16
    through gen.llm.array-engine. llama.cpp serves the adapter within
    its CUDA band (0.53 nats against 0.65), its replies 14/16 in the
    format. On Metal (genclj-h7ru) the same run gives the same held-out
    NLL, and llama.cpp serves the adapter within 0.017 nats (band 0.075),
    15/16 in the format. Metal's default memory limit loses little of the
    checkpoints' saving (2.2 GB against 2.1 GB at the measured limit).
- **Added (experimental): GRPO and its variants, `gen.llm.grpo`**
  (genclj-waym): a LoRA adapter trained by group relative policy
  optimization with verifiable rewards, over any family of gen.llm.models.
  - `(grpo/rollouts lm prompts opts)` draws K completions of each prompt
    (`{:prompt …}` and the reward's keys) in lockstep, every token from its
    key, and records each token's old log-probability where it is drawn:
    the policy's at the sampling temperature (`grpo/policy-sampling`, the
    model at temperature 1, by default).
  - Rewards are pure functions `(fn [prompt completion-text])` run on the
    host; `exact-match` (`:extract` a regex) and `format-match` are
    built in. A reward that is not a finite number is loud
    (`:non-finite-reward`).
  - `(grpo/advantages rewards)`: (r - mean)/(std + 1e-4) by group, the
    population's std; a flat group's are exactly 0, and rounds count them.
  - `(grpo/batches rollouts advantages opts)`: each row the prompt, the
    completion, then padding at the end (never inside a row), the mask
    exactly the completion's tokens (its end token among them).
  - `(grpo/loss m batch opts)`: the clipped surrogate with `:eps-low` and
    `:eps-high`, plus β times the k3 KL to the base model (the adapter at
    scale 0, no copy), aggregated by `:loss-type` as each paper defines
    it: `:grpo` (DeepSeekMath Eq. 3, β 0.04), `:dapo` (DAPO Eq. 8/12,
    token-level, ε_high 0.28), `:dr-grpo` (Dr. GRPO: the sum over N times
    the generation budget, no std), `:bnpo` (TRL's name: the token mean of
    each microbatch; the BNPO paper defines an advantage, not this).
  - `(grpo/trainer adapter optimizer key)`, `(grpo/round …)` (rollouts,
    rewards, advantages, then `:iterations` optimizer steps on them) and
    `(grpo/train …)`, logging each round's reward mean and std, flat
    groups, entropy, and each step's loss, max |ρ - 1|, clip fraction and
    KL. Checkpoints are sft's; a resumed run is the uninterrupted one bit
    for bit (the JVM backend, MLX's CUDA and CPU).
    `(grpo/evaluate lm prompts reward opts)`.
  - Checked on Qwen3-0.6B (`gen.llm.grpo-llama-test`, fenrir's CUDA), a
    LoRA on the last 8 blocks, two-digit products with an exact checker:
    at the first step every ratio is 1 within 1.0e-4; 40 rounds of 8
    prompts × 8 completions (5.5 min) take the held-out pass rate from
    0.615 to 0.771 (128 prompts × 8 samples, paired difference 0.155 ±
    0.022), greedy 0.703 → 0.789; llama.cpp serves the adapter with
    every argmax the same, its greedy pass rate 0.781.
- **Added (experimental): `gen.llm.sft/step`'s `:loss`** (genclj-waym):
  another summed loss in the NLL's place, whose reported log-probabilities
  come back in the step's `:reported`; `sft/order` is public.
- **Added (experimental): sampling from the whole distribution**
  (gen.llm.sample, genclj-waym): `pick` takes `:top-k nil` (with `:top-p`
  1) and draws from every token at the temperature, in one pass;
  `tempered`; `complete`'s `:token-logprobs?` gives each drawn token's
  log-probability at the sampling temperature, its end token's last.
- **Added (experimental): `qwen3/log-probs`'s `:temperature`**
  (genclj-waym): the log-probabilities of the logits divided by it.
  `qwen35/log-probs` takes it too (genclj-qfu2), and so
  `gen.llm.models/log-probs` on `"qwen35"` and `"qwen35moe"`: GRPO trains
  Qwen3.5 and Qwen3.6 at a sampling temperature other than 1, which was
  loud (`:unknown-options`).
- **Added (experimental): `gen.llm.models`** (genclj-stpo, genclj-8h0l):
  `log-probs` and `nll` by a model's architecture (its config's `:arch`,
  else its `:architecture`). These are multimethods each family
  registers in its own namespace; gen.llm.models.qwen3 registers
  `"qwen3"`, gen.llm.models.qwen35 `"qwen35"` and `"qwen35moe"`.
  `lora/init` reads the architecture through it.
- **Added (experimental): `gen.llm.chat/segments`** (genclj-stpo): the
  rendering by message and part (`:content`, `:reasoning`, `:end`,
  `:template`). `render` is now their concatenation, with the same text.
- **Added (experimental vars of gen.learning): `warmup-cosine` and
  `clip-value`** (genclj-stpo).
  - `(learn/warmup-cosine optimizer {:warmup :steps :final})` scales
    each step's move θ' - θ by Hugging Face's
    get_cosine_schedule_with_warmup factor: t/warmup, then a half cosine
    down to `:final` at `:steps`. Every optimizer gets it. With a warmup
    the first step's factor is 0.
  - `(learn/clip-value optimizer {:max-value v})` clips gradients
    elementwise. A wrapper acts before the optimizer it wraps.
- **Changed: `qwen3/config` (and a model's `:config`) holds
  `:architecture "qwen3"`** (gen.llm.models.qwen3, experimental;
  genclj-yc3b), the GGUF's general.architecture, which an adapter's file
  names.
- **Added (experimental): Qwen3.5's and Qwen3.6's forward passes over
  gen.array, `gen.llm.models.qwen35`** (genclj-xnx3): a GGUF of
  architecture qwen35 (dense) or qwen35moe (mixture of experts), with
  qwen3's functions (`load`, `model`, `config`, `forward`, `prefill`,
  `step`, `static-cache`, `compile-step` with `:donate?`, `select-rows`,
  `concat-rows`, `grow`, `offset`, `offsets`, `rewind`). llama.cpp's
  conventions (its qwen35, qwen35moe and delta-net graphs): GatedDeltaNet
  layers (a causal depthwise conv with its conv state, l2-normed q and k
  repeated to the value heads tiled, the gated delta rule with its
  recurrent state, a gated RMSNorm), gated attention every fourth layer
  (attn_q's query and output gate, RoPE on rope.dimension_count dims of
  a head), SwiGLU or the experts (softmax router, the top k renormalized,
  gather-qmm for quantized stacks, and the shared expert gated by a
  sigmoid); the MTP block's tensors recognized and not read. A cache is a
  value: attention layers' `{:k :v}`, GatedDeltaNet layers' `{:conv
  :state}`, replaced by each read. A recurrent state cannot be moved
  back: `rewind` takes a row to its own offset or to 0, anything else is
  loud (`:unsupported`). On Metal, Qwen3.5-0.8B-Q8_0's next-token
  log-probabilities are llama.cpp's on the same file within 1.6e-2 nats
  (20 prompts, whole, chunked and stepped; median 1.3e-3), the argmax
  and top 5 always the same (gen.llm.models.qwen35-llama-test). Compiled
  decode runs at 74.5 tokens/s at batch 1 and 437 at batch 16 (llama.cpp
  through gen.llm.llama: 79.3 and 418; bench/gen/qwen3_bench.clj, which
  now runs either family, as bench/gen/qwen3_llama_noise.clj does).
- **Added (experimental): `array/gated-delta`** (genclj-xnx3): the gated
  delta rule, GatedDeltaNet's recurrence, `[y state']` of q, k, v, the
  log decays g, beta and a state; `:method` `:loop` (token by token),
  `:chunked` (in chunks of `:chunk-size`, 64: the WY form llama.cpp
  computes, with a triangular inverse by doubling blocks) or `:auto` (a
  backend's fused kernel, else chunks from 64 tokens). A composite of
  gen.array's ops (the tape differentiates it, the batch tracer batches
  it), behind `p/IFused`'s new op `:gated-delta`: on MLX's Metal, mlx-lm's
  gated-delta kernel (ported with its notice: THIRD-PARTY-NOTICES,
  gen.backend.mlx.gated-delta), declined under MLX's native gradients,
  on the CPU device and for a key size not a multiple of 32.
- **Performance: a compiled step that donates its arguments no longer
  collects** (MLX backend, genclj-c7uf). The backend runs a collection
  (System.gc) once the arrays made since the last one hold
  `gen.mlx.gc-new-bytes` (a sixteenth of the memory), and counted a
  step's outputs as new even when they took over the donated inputs'
  buffers: a decode step consuming its static KV cache counted the whole
  cache every step and collected every few steps, 1.25 ms a step at batch
  1 on Metal (Qwen3-0.6B) and 12.5 ms at 16 rows. A donated array's bytes
  now come off that count (`MlxRuntime/donate`, which replaces
  `repoint`'s use, and `settleFresh`; new, rebuild with `clojure -T:build
  java`). Values are unchanged.
- **Performance (experimental): `gen.llm.array-engine` normalizes its
  rows about 2.5x faster** (genclj-9y6y), by
  `gen.llm.host/log-softmax-rows` (new): a row's log-normalizer is taken
  by blocks of 4096 logits, each block's largest and its sum of
  exponentials, combined in block order, the exponentials by
  `gen.llm.host/exp` (new: Tang's table-driven method, within 0.51 ulp on
  [-708, 0], about a third of `Math/exp`'s cost where that is fdlibm's).
  Its units of blocks are spread over the engine's host pool and its own
  thread, so a row's log-normalizer is the same, bit for bit, for any
  number of threads. ns an entry, one thread and 16 rows of Qwen3's
  151,936 on the pool of four: M4 4.8 to 2.0 and 1.30 to 0.65; fenrir's
  x86 under the Graal JIT 13 to 5.0 and 3.5 to 1.6 (floats are converted
  four at a time there: Graal serializes lone conversions). The
  log-normalizer moves in its last bits (within 1 ulp of an exact sum,
  where the sequential sum was within 9), so a log-probability, and a
  token drawn from the row, can change only where that moves a float's
  rounding: none of 9.7 million did. Not finite logits are loud as
  before (`:decode-failed`).
- **Performance (experimental): MLX's `array/->floats` reads a float16
  or bfloat16 array as its bits, widened on the host** (genclj-9y6y), not
  cast to float32 on the device first: the same floats, bit for bit. On
  CUDA the cast's buffer, freed at once, came back from MLX's buffer
  cache for the next read's cast in unified memory, so that the device
  wrote, and the host read, pages the other side held: Qwen3-0.6B's 16
  rows of float16 logits took 26 ms to read on fenrir, against 4.6 now.
- **Performance (experimental): `gen.llm.token`'s draw walks a row
  about 2x faster** (genclj-9y6y): its inverse CDF sums each token's
  probability by `gen.llm.host/exp`. The total moves in its last bits, so
  a draw for a key can change only when its uniform falls within an ulp
  or so of a token's cumulative probability.
- **Performance (experimental): `gen.llm.array-engine` normalizes a
  decode of fewer rows than its host pool's threads in parts across them**
  (genclj-c7uf): a row of Qwen3's 151,936 logits in 0.33 ms instead of
  0.73 (M4, four threads). Its log-normalizer and log-probabilities are
  the same, bit for bit: the terms are summed in the same order.
- **Changed (experimental): `gen.llm.array-engine` runs Qwen3.5 and
  Qwen3.6** (genclj-xnx3): `:model` a GGUF of architecture qwen3, qwen35
  or qwen35moe (read from its header), or a model either family's `load`
  made (its config's `:arch`); another architecture is loud
  (`:unsupported`). Its info's `:model` gains `:arch`.

- **Fixed (experimental): gen.llm.array-engine after a failed call**
  (genclj-5t5l).
  - A score whose read fails in a call that consumes the paged pools
    (the paged and continuous modes) now loses every branch the pools
    held, loudly: its error names them (`:lost-slots`) and they are no
    longer live. It used to drop `:lost-slots`, leaving them live over
    emptied slots, so their next operation failed as an engine bug
    (`:internal`).
  - A paged step whose rows fail to normalize keeps the pools it wrote,
    its slots as they were. It used to leave the pools it had consumed
    in the engine's state, which MLX refuses at the next step, losing
    every branch.
  - A run (every mode) normalizes its row before it writes its books, so
    one that fails leaves its slot as it was. It used to leave the slot
    ahead of its branch, so that the branch's next operation failed as an
    engine bug.
- **Fixed: `gen.llm.trie/insert` loops down a path** (gen.llm.trie,
  experimental; genclj-gn5i). insert recursed once for each node on the
  path it walked, so a trie with an entry at each of a few thousand
  prefixes of one context overflowed the stack, a StackOverflowError (at
  ~2,770 nodes on a 1 MB thread stack, fenrir's default). It now walks
  down in a loop and rebuilds the path back up in another, as
  remove-entry, entries and size already did (genclj-6h3b); its results
  and its `:entry-exists` error are unchanged. A node stands only at an
  entry or where paths part, so a long context alone never recursed
  deeply, and gen.llm.cache's trie, at most :max-branches entries, passes
  fewer than twice that many nodes on a path.
- **Fixed: a batched particle outside its support moves as its scalar
  trace does** (gen.batched; genclj-n1s3). A batched run cannot stop where
  a particle's choice leaves its support, so the particle's batched trace
  holds the choices the run made after that one, which its scalar trace
  (`b/traces`) stops short of. `update`, `regenerate` and `move` kept
  those choices' values and weighed them (an update discarded them too),
  and `project` counted them, where the particle's scalar operation makes
  them afresh: a particle moved back inside its support (its choice that
  left it regenerated) left its scalar trace, its weight log p(c |
  new)/p(c | old) for such a choice c where the scalar law gives 0. Such
  a particle (score -Inf, on a concrete backend) now runs its scalar
  operation alone, the result set in its row, and the collection stays
  batched. **Draws:** those particles draw from their own keys (the key
  split n ways, as the fallback's). A lazy backend's (MLX's) scalar runs
  go on as batched ones do, so nothing changes there. Since 2026-10-25 (a
  choice outside its support stops the run, 9277423); found by
  gen.fuzz-test's chains of operations (seed 1794085868851).
- **Added (experimental): `array/compile`'s `:donate`** (genclj-tjvd):
  `(array/compile f {:donate #{0}})` names arguments the compiled function
  consumes, after JAX's donate_argnums. On MLX their arrays are released
  once a call has returned its lazy result, so MLX reuses their buffers
  for it (a scatter into a large array writes in place instead of
  copying), and any later use of a donated array is an ex-info,
  `:reason :donated`. Where compile is the function itself (the JVM
  backend, the tape, the batch tracer) it does nothing and a reuse is not
  caught; the values are the same either way. A malformed `:donate`, or
  one naming a position past the call's arguments, is `:invalid-args`.
  The MLX backend reads every handle through a checked accessor
  (`gen.backend.mlx.array/handle`); a donated handle is repointed at a
  shared husk (`MlxRuntime/repoint`, new: rebuild with `clojure -T:build
  java`) and still freed once, by its phantom reference.
- **Added (experimental): gen.array's language-model engine,
  `gen.llm.array-engine`** (genclj-8rnc): a Branching engine over
  gen.array, gen.llm.slots's Slots over Qwen3's static KV caches
  (gen.llm.models.qwen3), on MLX (Metal, CPU) or the JVM backend.
  `(array-engine/engine {:model gguf-or-model})` takes `:dtype`,
  `:max-branches` (32; memory is the bound, no sequence cap), `:n-ctx`,
  `:n-batch` and `:block` (capacities grow by blocks). A branch is a row
  of a group (a static cache): a fork shares it; a read (open, extend,
  score) runs eagerly in forwards of `:n-batch` tokens, each target's
  score taken on the device; a step is one compiled decode that consumes
  its cache: a group stepped by exactly its own rows (lockstep
  particles) in place, any other step's rows stacked first. Rows are
  read at the step's end and normalized on the host in double. Its
  arrays count in MLX's own memory, inside the shared budget; a read or
  step past it is `:mlx-budget-exceeded` (marked `:exhausted?`).
  `stats` and `groups` are diagnostics. The LM, token and SMC layers
  run over it unchanged, with any tokenizer of the model's vocabulary.
- **Added (experimental): `(llama/vocabulary path)`**, llama.cpp's
  vocabulary of a GGUF loaded alone (vocab_only: no weights), a
  `java.io.Closeable` that `tokenize`, `token-bytes`, `detokenize` and
  `tokenizer` take as they take an engine: the tokenizer of another
  engine over the same file. A closed one is loud (`:closed`).
- **Added (experimental): `array/->floats`**: an array's values,
  row-major, as a new float array, read at once (an optional protocol,
  `IHostFloats`; the JVM backend and MLX implement it, any other backend
  reads through `->clj`). **Added (experimental): `(qwen3/grow cache
  capacity)`**, a static cache with zero slots added.
- **Added (experimental): `array/put-along-axis`** (genclj-8rnc): x with
  values written over its entries at indices along an axis, the overwrite
  `scatter-add` (which accumulates) is not; indices and values shaped as
  `scatter-add`'s. A new optional protocol, `IOverwriting`
  (`-put-along-axis`), so `IBackend` gains nothing: the JVM backend, MLX
  (`mlx_put_along_axis`), the tape (x's cotangent zeroed where written, the
  values' taken there), the batch tracer and staging (`:put-along-axis`;
  the device lowerings refuse it) implement it. Repeated indices along a
  line of the axis are loud on the JVM backend (`:repeated-indices`) and
  unspecified on MLX.
- **Changed: Qwen3's static caches write their keys and values with
  `put-along-axis`**, not `scatter-add` (genclj-8rnc), so a slot written
  again holds the new value instead of the sum; the slots past a row's
  offset need no longer be zeros. **Added (experimental):
  `(qwen3/rewind cache offsets)`**, a static cache with its rows moved
  back (each to at most its own offset): the positions past them are
  dropped and written over by the next steps (a rollback, a slot reused).
- **Added (experimental): static-capacity KV caches and compiled steps
  for Qwen3** (gen.llm.models.qwen3; genclj-tjvd). `(static-cache m rows
  capacity)`, or prefill's `:capacity`, is a cache of fixed shapes:
  preallocated keys and values per layer, written at each row's offset by
  `scatter-add`, a mask for each row's valid length, and the positions an
  int array (`:positions`) beside the host's `:offsets`, so a step's
  shapes stay put and rows may sit at different offsets (`offsets`,
  `select-rows`, `concat-rows`). Its logits are the concatenating
  cache's (exactly on the JVM backend). A step past the capacity is
  `:capacity-exceeded`. `(compile-step m opts)` compiles the step over
  static caches with `array/compile`, one trace per shape (a decode step,
  a prefill chunk); `:donate? true` consumes the cache it steps, written
  in place on MLX. `offset` of a static cache with rows at different
  offsets is loud (`offsets` reads them); step does not take `:capacity`.
  bench/gen/qwen3_bench.clj measures prefill and decode, eager
  and compiled, against llama.cpp on the same GGUF, on Metal or MLX's
  CUDA build.
- **Added: LoRA adapters on llama.cpp, and models engines share**
  (gen.llm.llama, experimental; genclj-vdu4). `(llama/model path)` loads
  a GGUF once for several engines, each over a context of its own
  (`llama/engine`'s `:model` takes it, or a path as before);
  `(llama/adapter m path)` loads a LoRA adapter's GGUF for model m, and
  `:adapters {adapter scale}` sets adapters with their scales on an
  engine, under which all its branches read: specialised critics or
  proposals from one base model. `model-info` and `adapter-info` describe
  them (an adapter's GGUF metadata, alpha, aLoRA invocation tokens);
  `(br/info e)` gains `:adapters`. Models and adapters are Closeable,
  freed by the Cleaner when unreachable; one an open engine uses does not
  close (`:in-use`), and an engine refuses an adapter of another model, a
  closed one (`:closed`) and an aLoRA (`:unsupported-adapter`). The
  binding gains llama.h's adapter functions (native/jextract-llama.sh);
  rebuild with `clojure -T:build java`. In gen.array.memory a model
  reserves its weights, an adapter its tensors and an engine its
  context's buffers, so an engine that loads its model holds two
  reservations where it held one, of the same total; errors of a model's
  loading say `llama/model` (their `:reason`s are unchanged). Test
  adapters for Qwen3-0.6B are in test/fixtures/lora, with their
  generator.
- **Breaking: the inference, learning and gradient functions refuse an
  option key they do not read** (gen.inference.mcmc, .nuts, .kernel,
  .particle-filter, .importance, .vi, .mh, .enumerative, .transform,
  .diagnostics, gen.learning, gen.gradients; genclj-vux3, genclj-yrb6;
  DESIGN §9, 1.2.1's owner calls, item 6). They ignored it and ran on
  their defaults, silently: `(mcmc/hmc t s {:steps 50 :epsilon 0.3})` ran
  at :L 10 and :eps 0.1, `kernel/run`'s `{:burnin 50}` burned nothing,
  `nuts/sample`'s `{:adapt-delta 0.95}` kept 0.8, `(learn/adam
  {:learning-rate 0.1})` stepped at 0.001, `diagnostics/summary`'s
  `{:quantiles …}` gave the default quantiles. Each now throws an
  `ex-info`, `:reason :unknown-options`, its data naming the function
  (`:fn`), the keys refused (`:unknown`) and the keys it takes
  (`:accepted`), and an options map that is not a map
  (`:invalid-options`). The functions: mcmc's hmc, mala, nuts,
  map-optimize and their -particles forms; kernel's kernel, hmc, mala,
  nuts and map-optimize (when the kernel is made) and run; nuts' adapt
  and sample; mh's move; importance's importance-sampling,
  importance-resampling, estimate and weighted-sample; the particle
  filter's initialize, step, resample, resize, maybe-resample, replicate,
  dereplicate, coalesce, introduce and reweighting; vi's black-box-vi,
  black-box-vimco, elbo-gradient and vimco-gradient; enumerative's
  choice-vol-grid (its trailing map), enumerative-inference and
  with-exact-proposal; the translators' maps (translator,
  deterministic-translator, extending-translator, updating-translator)
  and involution's options; diagnostics' summary; gen.learning's
  optimizers (sgd, gradient-descent, momentum, rmsprop, adagrad, adam,
  adamw, clip-norm) and train; gen.gradients' choice-gradients.
  kernel/kernel and tf/run refused unknown keys already, and now say so
  in the same form (tf/run's data was `{:options …}`); a deterministic
  translator's `:q-forward` and `:q-backward` keep their own error. This
  breaks correct code that shares one options map between functions that
  read different keys: `nuts/adapt` refuses sample's `:samples` and
  `:keep`, `kernel/run` refuses nuts' `:warmup`, `pf/step` refuses
  initialize's `:batched?`, `pf/resample` maybe-resample's
  `:ess-threshold`, `updating-translator` translator's `:p-new`. Give each
  function its own keys, `(select-keys opts (:accepted (ex-data e)))`. A
  qualified key is refused as any other. The API snapshot records `:as
  opts` added to the arglists of weighted-sample, resample, replicate,
  dereplicate, coalesce, vimco-gradient, summary, enumerative-inference,
  with-exact-proposal, translator, extending-translator,
  updating-translator, involution, train and the eight optimizers (their
  arities are unchanged).
- **Added (experimental): gen.array's neural-network operations**
  `rms-norm`, `rope` and `sdpa` (gen.array, vars marked `{:api
  :experimental}`; genclj-ezdd; docs/array-api.md §6). Composites of
  gen.array's ops, so differentiated, batched and staged as every
  composite; MLX computes them by its fast kernels (mlx_fast_rms_norm,
  mlx_fast_rope and mlx_fast_rope_dynamic,
  mlx_fast_scaled_dot_product_attention) on its own float32 and half
  arrays, through a new optional protocol,
  `gen.array.protocols/IFused` (experimental). `rope` takes NEOX's or
  the traditional pairs, partial dims, a base or frequencies, a scale
  and an offset per row; `sdpa` grouped-query attention with a causal,
  bool or additive mask. A query that sees no key gets zeros.
- **Added (experimental): quantized weights**, `quantize`, `dequantize`,
  `quantized-matmul` and `gather-qmm` (gen.array, `{:api
  :experimental}`; genclj-vjwr): MLX's affine layout (2 to 8 bits, groups
  of 32, 64 or 128), primitives of a new optional protocol,
  `gen.array.protocols/IQuantized` (experimental). MLX calls
  mlx_quantize, mlx_dequantize, mlx_quantized_matmul and mlx_gather_qmm;
  the JVM backend quantizes as MLX does (the same words, scales and
  biases) and dequantizes on the host. Differentiable with respect to x
  only: a gradient with respect to the weights is loud on the tape. A
  backend without the protocol, the staging backend among them, is loud
  (`:unsupported`). MLX's mxfp4, nvfp4 and mxfp8 modes are loud too:
  their scales are uint8, a dtype gen.array lacks.
- **Added: uint32 arrays on the JVM backend** (genclj-vjwr): the
  quantized words' dtype, exact in its doubles, wrapping as int32
  does. `(array/array data {:dtype :uint32})` and `(astype x :uint32)`
  threw there.
- **Added (experimental): GGUF files, `gen.array/load-gguf`** (gen.array,
  `{:api :experimental}`; genclj-cr5a): a GGUF file's metadata (host
  values) and its tensors by llama.cpp's names, on `*backend*`. F32, F16
  and I32 tensors load as arrays (F16 as float32 on the JVM backend,
  which has no float16); Q8_0, Q4_0 and Q4_1 as MLX's affine
  quantization, `{:wq :scales :biases :bits :group-size}`, what
  `dequantize` and `quantized-matmul` take, with float32 scales and
  biases, so that the values are the file's exactly. A model split by
  llama.cpp's gguf-split loads from its first file. Every other tensor
  type (the K-quants, BF16) is a loud `:unsupported`, a file that is no
  GGUF or is cut short `:invalid-gguf`. MLX loads by mlx_load_gguf, the
  JVM backend by a reader in plain Clojure (gen.array.gguf): the same
  words, scales and biases, bit for bit. A new optional protocol,
  `gen.array.protocols/ILoading` (experimental). On MLX a file with
  float64 metadata is refused: MLX 0.32.4's loader misreads it and the
  process dies (SIGBUS). Test files are in test/fixtures/gguf, with
  their generator.
- **Added (experimental): Qwen3's forward pass over gen.array,
  `gen.llm.models.qwen3`** (genclj-gvfa): a Qwen3 GGUF's model
  (`load`, `model`, `config`), its logits for a whole context
  (`forward`), and a cached prefill and steps (`prefill`, `step`) over
  caches that are values (`{:offset :layers}`, grown by concatenation,
  so a fork is free). llama.cpp's conventions (its qwen3 graph): RMSNorm,
  per-head q/k norms, NEOX's RoPE, grouped KV heads, causal attention,
  SwiGLU, a tied or untied output; quantized matrices multiplied by
  `quantized-matmul`, the embedding by `take` and `dequantize`. A file
  with tensors the forward does not read, RoPE scaling, a token outside
  the vocabulary and an option key it does not read are loud. The
  same backend-neutral code runs on MLX (its fused kernels) and on the
  JVM backend (tiny models, float64), and the tape differentiates it. On
  Metal, Qwen3-0.6B-Q8_0's next-token log-probabilities are llama.cpp's
  on the same file within 3.6e-2 nats (20 prompts, whole, chunked and
  stepped), the argmax always the same (gen.llm.models.qwen3-llama-test;
  bench/gen/qwen3_llama_noise.clj).
- **Added: THIRD-PARTY-NOTICES** (genclj-ty8m): the origins and licenses
  of files derived from other projects, and the header a ported file
  carries (CLAUDE.md, rule 10).
- **Added (experimental): `array/checkpoint`, gradients that recompute**
  (gen.array; genclj-pnu6). `(array/checkpoint f)` is f, with the same
  values and gradients, whose gradient does not keep f's intermediates
  from the forward pass: under value-and-grad f runs again in the
  backward pass when its cotangent arrives, so a stack of checkpointed
  layers holds one layer's activations at a time. Its arguments are
  arrays or trees; their float arrays are the inputs (host numbers and
  int arrays pass through), and f must not close over a value being
  differentiated (loud on the tape, `:reason :closes-over`). On the tape
  one node whose VJP runs f on a tape of its own (exactly the unchecked
  gradients on the JVM backend); under MLX's native gradients one
  `mlx_custom_vjp` recomputing f on its inputs made to depend on its
  outputs and cotangents (`mlx_depends`), mlx-node's memory-ordered
  per-layer checkpoint; f itself elsewhere (no gradient, the batch
  tracer, inside `array/compile` under native gradients). A backend
  implements the new optional protocol `gen.array.protocols/ICheckpoint`
  (experimental). Measured on fenrir's CUDA (bench/gen/checkpoint_bench.clj,
  docs/array-api.md §8): a gradient of Qwen3-0.6B's training loss over
  4 × 256 tokens peaked at 1.5 GB checkpointed against 5.6 GB, 87 → 120
  ms, when MLX's memory limit is near what the step needs (at a loose
  limit MLX keeps queued kernels' buffers either way).
- **Added (experimental): `array/selective-log-softmax`** (gen.array;
  genclj-pnu6): log softmax(logits) at each index, the logit there less
  the logsumexp over the last axis, its logsumexp taken `:chunk` entries
  at a time, never the [.. V] log-softmax. The logits are an array, or a
  function of a range of vocabulary entries (with `:vocab`), so an LLM's
  output projection runs a chunk at a time and its [B T V] logits are
  never made; with `:args` each chunk is checkpointed. Its gradient is
  the full computation's within round-off. Indices out of range are
  loud on the JVM backend and clamped on MLX; an unknown option is loud
  (`:unknown-options`).
- **Added (experimental): Qwen3's training loss** (gen.llm.models.qwen3;
  genclj-jksq): `log-probs`, each next token's log-probability [B T-1],
  and `nll`, the negative log-likelihood of supervised fine-tuning
  (`:mask` weights per position), both differentiable with respect to
  the dense weights (a quantized matrix stays frozen), with
  `:checkpoint?` (every block and output chunk checkpointed) and
  `:chunk` (the output projection by vocabulary chunks). Their tape
  gradients match central differences on the tiny model (4.4e-8
  relative), the checkpointed ones are the plain ones exactly on the JVM
  backend, and MLX's (native or tape, CUDA and CPU) are the JVM
  backend's within 2.6e-7 of the largest gradient.

## [1.2.1] - 2026-11-06

The health check's fixes (DESIGN §9; genclj-9mq3). A review of 1.2.0 by
six read-only reviewers found silent wrong results and raw exceptions
where rule 6 wants a loud `ex-info`, places where the tests broke their
own conventions, 1.2.0's LLM work with no consumer and its llama.cpp
claim unmeasured, the device path out of a dependent's reach, and docs
drift. All 43 of its beans are done. The owner asked for the release to
be worked autonomously, so its calls are PROVISIONAL, for the owner to
confirm (docs/decisions.md, "1.2.1's owner calls"), but for its scope
and its numeric edge rules, which the owner approved before the tag.

Read first, the **Breaking** changes. Each makes code throw that was
already silently wrong (it scored NaN, recorded nothing, or read another
value than it named), which a patch release may do (DESIGN §9): `gen`
refuses a parameter or an enclosing local named `trace`, `splice`,
`param` or `untraced` (Gen.jl's usual `(gen [trace] …)` proposal recorded
nothing); a distribution refuses an infinite parameter where it scored
NaN; `defdist` and `dist` refuse names that shadow their own; hmc's
`:eps` and `:L` and mala's tau are checked; `max`, `min`, `argmax` and
`argmin` over an empty axis, `arange`'s zero step and bool dtype, a
negative index in a selection, `sample-states` of an impossible trace
and the potential combinators `coerce` and `after` are loud; importance
sampling with `:device` follows `:batched?`. Refusing unknown option
keys, which correct code meets (a shared options map), waits for 1.3.0
(genclj-vux3). Two results change for correct code: float `floor-div` is
NumPy's (`(floor-div 1.0 0.1)` is 9.0, as `mod` agrees), and `randint`
and `uniform-discrete` draw other integers for a key in float32 (exact
now; float64's are unchanged).

The calls taken autonomously (docs/decisions.md, 1.2.1): non-finite
parameters refused where a family's code gave NaN, the limits kept where
it computes them; int64 overflow an `ex-info` on the JVM backend, MLX's
wrap documented; empty extrema loud on every backend; `=` on arrays and
`counted?` documented as they are; `solve`'s rule kept and `inv` fixed;
an infinite value outside every support; exact integers for `randint`;
the CUDA C lowering parked; one bounded cache of compiled code; CFG
masks built as lexemes, a lazy DFA of Earley sets and masks in parts;
the decision log split from DESIGN.md.

No native library changes: nothing needs rebuilding. The MLX source
build's script (`native/build-mlx.sh cuda13-source`) works again with a
CUDA toolkit from pip.

Checked before the tag: every test on the JVM backend, the book included
(2,215 tests, 29,709 assertions), with ext/codegen's (25), ext/commons-math's
(27) and ext/device's (47); on the Mac mini M4 the full battery on Metal
(2,215 tests, 24,486 assertions), on MLX's CPU in float32 (2,215, 24,503) and
its light tier in float64 (2,097, 21,811); on CUDA, an RTX PRO 6000 with MLX
0.32.4 built from source (`cuda13-source`, sm_120), all 2,215 tests in five
JVMs and ten exits with no abort; llama.cpp v0.9.0's tests on its three test
models on Metal and on CUDA (37 each); memo.clj (231 tests) and its ext/synth
(17) against the candidate. The batteries on Metal, MLX's CPU in float32 and
CUDA each met one error, in tempered SMC's new test of impossible starts (a
move the model rules out read a NaN likelihood on MLX, and the new loud rule
threw): fixed before the tag, and that namespace rerun on each, green.
GenGDT: 14 of its 15 tests passed; the other, its own pin check, compares
its gen.clj pin (v1.0.0) with memo.clj's (v1.2.0), not gen.clj's code.

### Known issues

- **MLX clamps an out-of-range array index** in a gather by default, where
  the JVM backend throws (docs/array-api.md §13): `-Dgen.mlx.check-gathers=true`
  checks each gather's indices, at an evaluation per gather.
- **The book on MLX:** some chapters of Parts II-V do host math on results
  and fail on the MLX backend (genclj-ac70). The book's gate is the JVM
  backend, where every chapter passes.
- **Context-free masks** are exact for the text's bytes, not its tokens: a
  vocabulary without a byte the language needs can reach a prefix that no
  token sequence completes (byte-level BPE vocabularies have every byte).
- **The CUDA C lowering is parked** (gen.device.cuda; genclj-z1gj): no
  consumer yet, so `:device` lowers to `:jvm` and `:metal`, and its tests
  that need a CUDA device skip by name (`-Dgen.cuda.parked=false` runs
  them).
- **llama.cpp v0.9.0 brings no gain to many branches** (genclj-ajse):
  measured against v0.7.0, it ties up to 8 branches and is 5-16% slower at
  4 to 16 after a long prompt; gen.llm.cache re-reads a prompt that no
  live branch holds (a bean of its own).

### Changes

- **Breaking (experimental): tempered SMC's per-step keys renamed**
  (gen.inference.tempering, outside the 1.x promise; genclj-vjbo; DESIGN
  §9's result keys). `:ess`, a vector of each step's effective sample
  size where the result keys make `:ess` one host number, is
  `:ess-by-step`, and `:acceptance` is `:acceptance-rates`, each step's
  `:acceptance-rate`. Code that read `:ess` or `:acceptance` reads them
  under the new names. `smc`'s docstring names every key of its result,
  `:scales` among them.
- **Breaking: `coerce` checks that its potential is over bytes when it is
  made** (gen.llm.potential, experimental; genclj-kdau). Only `logw-next`
  checked the 256 bytes, so `prefix` and `complete` of a coerced potential
  over another vocabulary read the text's bytes as its tokens, silently
  wrong. `coerce` now reads p's next weights at the empty context and
  refuses any other vocabulary (`:invalid-potential`); a potential of no
  weight there (none anywhere) is not checked. Only code that was already
  wrong throws.
- **Breaking: `after` checks its closes against the potential's
  vocabulary** (gen.llm.potential, experimental; genclj-kdau). A close at
  or past p's vocabulary size was skipped, so a sequence could never close
  and the open row weighed nothing (Z = 0), silently. `after` now refuses
  it (`:invalid-token`), reading p's next weights at the empty context when
  it is made (it read them at its first open row before). Only code that
  was already wrong throws.
- **Breaking: an infinite parameter is refused where its family gave NaN**
  (gen.dist, gen.dist.core; DESIGN §9, 1.2.1's owner calls, item 1). Only
  NaN was refused, and `check-positive` takes +Inf, so `(normal ##Inf 1)`,
  `(normal 0 ##Inf)` and `(exponential ##Inf)` constructed and their draws
  scored NaN; 30 other families scored NaN at an infinite parameter, or
  drew NaN or a meaningless point (`(log-normal ##-Inf 0.5)`'s zeros).
  Each throws at construction now (`ex-info` naming `:distribution` and
  `:param`, `:reason :non-finite`), for a host value and for an array's
  elements when `array/concrete?` (Clojure data on every backend), as the
  other parameter checks do; an HMC or MALA step that meets one counts a
  divergence, as for any parameter check. Code that constructed such a
  distribution only ever scored NaN. A family whose code computes the limit
  keeps it: the truncated normal's infinite bounds and wrapped-normal's
  infinite sigma (the uniform on the circle). A -Inf logit of
  `categorical-logits` stays a zero probability; +Inf, or a row of -Infs,
  is refused. Added: `gen.dist.core/check-finite`, for a family of one's
  own.
- **Breaking: `defdist` refuses a parameter that shadows a clause's
  argument, and `dist` a draw that names a local of its body**
  (gen.dist.core, gen.dist.dsl; genclj-8326). Each is an error when the
  definition expands, and only a definition that was already wrong meets
  it. A `defdist` parameter named as one of `sample`'s or `log-prob`'s
  arguments (`key`, `shape`, `v`, whatever the clause calls them) was
  bound around the clause's body, so the body read the parameter in place
  of the key, the batch shape or the value, silently; a parameter named
  twice shadowed the first likewise (`:shadowed`, `:repeated-params` in
  the error's data). `dist` makes its draw outside the body, so a draw
  naming a name the body binds (a `let`'s) failed to compile when the name
  was unbound elsewhere and read the other value when a parameter or a var
  had it, silently; now it is an error naming the draw and the names
  (`:bound`).
- **Breaking: `arange` refuses a bool dtype, a zero step and non-finite
  arguments** (gen.array, every backend; genclj-06ba, genclj-kdau):
  `ex-info` with `:reason :invalid-dtype` or `:invalid-args`. Only code
  that was already wrong meets it: a zero step threw a raw
  IllegalArgumentException on the JVM backend and MLX's own error; MLX has
  no bool arange, and the JVM backend's held fractions past its second
  element (NumPy's fails there). Compare an int arange for bools.
- **Breaking: `max`, `min`, `argmax` and `argmin` over an empty axis are
  loud** (gen.array, every backend; DESIGN §9): `ex-info` with `:reason
  :empty-reduction`, as NumPy and JAX raise. The JVM backend gave -Inf and
  +Inf for a float max and min (and a raw IllegalArgumentException for an
  int dtype), MLX its own error. An empty result (an empty axis that is
  not reduced) is no error, now on MLX too for argmax and argmin;
  `logsumexp` over an empty axis is still -Inf and `softmax` of one an
  empty array. Code that read ∓Inf from an empty max on the JVM backend
  must test for the empty axis.
- **Breaking: hmc's `:eps` and `:L` and mala's tau are checked**
  (gen.inference.mcmc; genclj-mhgn; DESIGN §9, 1.2.1's owner calls). As
  nuts checks its own: `hmc`, `hmc-particles` and kernel/hmc throw
  `ex-info` for an `:eps` that is not a positive finite number
  (`:reason :invalid-eps`) or an `:L` that is not a positive integer
  (`:invalid-L`), and `mala`, `mala-particles` and kernel/mala for such a
  tau (`:invalid-tau`). Only code that was silently wrong throws: `:eps 0`
  or `:L 0` "accepted" every move as a no-op (an acceptance rate of 1),
  `:L 2.5` ran 2 steps, and a negative tau made every move a divergence.
- **Breaking: `gen` refuses a parameter, or a local around its body that
  the body reads, named `trace`, `splice`, `param`, `untraced` or
  `untraced-key`** (gen.dynamic, the static form too; genclj-k3xn). The
  body binds those names itself, so such code compiled and was silently
  wrong: `(gen [trace] (trace :x …))`, Gen.jl's usual name for a proposal's
  argument, recorded no choice and scored 0 (an MH step with it did
  nothing), and `(let [trace t] (gen [] (:score trace)))` read the body's
  own `trace`. Now each is a compile error (`:reason :reserved-name`,
  `:binding :parameter` or `:local`) naming the rename. A gen nested in a
  body reads its own names, as before, and a body may still bind one of
  them in a `let` of its own.
- **Breaking: a negative index in a selection is a loud error in every
  combinator that numbers its calls** (gen.combinators: Map, Unfold, Scan
  and Recurse, in regenerate and project; genclj-kdau). Map threw a raw
  IndexOutOfBoundsException, and the kinds disagreed: some threw it,
  others skipped the index silently, so the selection selected less than
  it named (code that relied on that was silently wrong). Now each throws
  `ex-info` with `:reason :negative-index` and the `:indices`. An index
  past the elements still selects nothing, as an absent address does.
- **Breaking: `sample-states` of a trace whose observations are
  impossible is a loud error** (gen.state-space; genclj-vjbo). On the host
  such a run stops at the impossible step (gfi/impossible?), and
  sample-states returned fewer than n states for it, where the states have
  no posterior (code that used them was silently wrong). Now it throws
  `ex-info` with `:reason :impossible`. The namespace doc says what an
  impossible observation does: on the host the run stops there, and a run
  whose score is an array (on MLX, or batched where the particles differ)
  goes on from the prediction; it said the belief stayed at its
  prediction.
- **Breaking: importance sampling with `:device` follows `:batched?`**
  (gen.inference.importance, its experimental `:device`; genclj-cxrx;
  DESIGN §9). `:batched?` was ignored under `:device`, so code asking for
  a batched run got n traces, and `:batched? :required` was dropped
  silently: only such code, already wrong, changes. Batched, the result
  holds `:particles`, a gen.batched collection over the program's outputs
  (its draws still per particle, the unbatched run's), in place of
  `:traces` (`importance/traces` reads either); a model the device cannot
  run falls back to the batched run, which `:required` refuses where it
  cannot batch. A trace's host values (a count, a flag) are now its
  handler's, not arrays made from the program's outputs.
- **Fixed: `native/build-mlx.sh cuda13-source` with a toolkit from pip**
  (genclj-ajse). MLX 0.32.4 links cuSOLVER (with cuSPARSE and nvJitLink
  under it) when it is built, and CMake stopped at "CUDA::cusolver not
  found" on a CUDA toolkit made of NVIDIA's pip packages (fenrir's). The
  script fetches the missing libraries' wheels before configuring, as it
  already did for the wheel variant's run time, and points CMake at them.
  1.2.0's source build was never rebuilt after its backported patch was
  dropped; it builds and runs again (sm_120).
- **Fixed: a value outside a distribution's support sends no NaN
  gradient** (gen.dist, gen.dist.dsl; genclj-zkbw). gamma, inv-gamma,
  beta, chi-squared, nakagami and dirichlet guarded their log density with
  a where whose other branch was NaN outside the support, so a parameter's
  gradient was 0·NaN = NaN there, and one such value in a batch made a
  shared parameter's whole gradient NaN (beta-uniform's and a
  pushforward's through them too). Each now scores a stand-in value
  there, as geometric and weibull did. A value at ±Inf has log density
  -Inf in every family, where gamma, chi-squared, nakagami, weibull,
  poisson and neg-binom gave NaN at +Inf and gumbel and logistic at -Inf,
  and its gradient is never NaN; a pushforward's value outside exp's range
  (at or below 0) sends no gradient either. Log densities elsewhere are
  unchanged (logistic's is computed in |z|, within rounding).
- **Fixed: `la/inv` of a batch of matrices** (gen.array.linalg;
  genclj-l0n4). inv solved against the `[n n]` identity, which the
  right-hand side rule reads as a batch of vectors beside `[b n n]`
  matrices: a `[3 3 3]` batch gave a `[3 3]` result (each matrix's inverse
  times one basis vector) and a `[2 3 3]` batch threw. The identity is now
  broadcast to the matrices' shape, as the tape's rules do; one matrix's
  inverse is unchanged. `la/solve`'s rule stays as documented (DESIGN §9).
- **Fixed: an int `arange` holds NumPy's values on every backend**
  (gen.backend.jvm, gen.array.stage; genclj-06ba). The JVM backend stored
  start + i·step unrounded under an int dtype: `(arange 0.5 3 1 {:dtype
  :int64})` read `[0 1 2]` but added to itself gave `[1 3 5]` and equalled
  0 nowhere. Its values, and the stager's, are now MLX's and NumPy's: the
  first value is start cast to the dtype, the step dtype(start + step) -
  dtype(start), so a fractional step may repeat a value (`(arange 0 3 0.5
  {:dtype :int64})` is six zeros). Integral arguments, the common case,
  give what they gave.
- **Fixed: int64 overflow is an `ex-info` on the JVM backend**
  (gen.backend.jvm; DESIGN §9): `:reason :overflow` where an int64 result
  leaves the range, at the op. The JVM backend holds int64 values in
  doubles, and one past 2^63 threw a raw IllegalArgumentException when it
  was read. MLX wraps, as host numbers do on every backend (documented,
  docs/array-api.md §13).
- **Fixed: loud errors where gen.array, gen.array.stage and gen.random
  threw raw exceptions or gave silent values** (genclj-kdau; rule 6), each
  an `ex-info` with a `:reason`: a negative dimension is `:invalid-shape`
  (`(zeros [-2 0])` made an array of that shape, `(zeros [-2 3])` claimed
  more than Long/MAX_VALUE cells); `expand-dims` at a repeated axis is
  `:repeated-axis`, as NumPy (a set dropped it, one axis too few);
  `grad`'s `:argnums` past the arguments `:invalid-option` (an
  IndexOutOfBoundsException); `rng/key-data` of a staged key `:staged-key`
  (a ClassCastException); `(rng/key 2.5)` `:invalid-data`, as `fold-in`
  (it was `(rng/key 2)`). `array/scan` (experimental) with n 0 over xs runs
  no step and returns empty ys of the step's shapes (it threw "Index 0 out
  of bounds"); its host loop checks that every leaf of xs has a leading
  axis of n (`:invalid-args`) and that a step keeps the carry's structure,
  shapes and dtypes (`:invalid-carry`), as the stager checked shapes and
  dtypes (it now checks the structure too). A staged 0-d value fills
  `array/full` (a ClassCastException). Breaking only for code that was
  already wrong: a fractional or a float seed (2.0 included, as JAX's
  PRNGKey), a repeated axis.
- **Fixed: `rng/randint`, and so `uniform-discrete`, stay inside their
  range** (gen.random; genclj-psgi). randint computed floor(low + u·(high
  - low)) in the backend's default float, which rounds up to high: in
  float32 for about 0.3% of the draws at low 10^6, high 10^6 + 10 (MLX's
  default), and in float64 for the largest uniforms far from 0. So
  uniform-discrete sampled outside its support on MLX, a trace of score
  -Inf. Now the draw is exact integer arithmetic on the same uniforms'
  bits, low + floor(k·r / 2^b): in float64 k is one uniform's 52 bits,
  floor(u·r) exactly; in float32 three uniforms' 23 bits each (b = 69).
  The same key draws the same integers on every backend of the same
  default float. An empty range is loud (`:empty-range`) where its bounds
  can be read (a lazy backend's array range clamps to low), a fractional
  host bound `:invalid-args`, and host bounds outside int32 on a backend
  whose int is int32 (they wrapped) `:invalid-args`.
- **Fixed: tempered SMC from starts the observations rule out**
  (gen.inference.tempering, experimental; genclj-dm71). A start that
  gives the observations density 0 (a likelihood whose support depends on
  a latent: truncation, uniform noise, a hard constraint) had the log
  weight -Inf - -Inf = NaN, and `smc` threw "A log weight is NaN"; such a
  start now weighs 0 from the first step on, and the estimate is
  unchanged by it, whatever the internal proposal. A move's NaN
  acceptance ratio, which the moves rejected silently, is a loud error
  (`:reason :nan-log-alpha`), but for a move to a trace of density 0,
  rejected as before: one the observations rule out, or one the model
  rules out whatever its likelihood reads (on MLX a likelihood whose
  parameter left its domain reads NaN).
- **Fixed: a particle filter's unbatched `:log-weights` are host doubles**
  (gen.inference.particle-filter, gen.inference.particle-mcmc;
  genclj-sq31). DESIGN §9's result keys make them host numbers, as the
  batched filter's and importance sampling's are; `initialize`, `step`,
  `move-reweight` and `introduce` one particle at a time, and conditional
  SMC's reference, kept generate's and update's weights, 0-d arrays: MLX
  scalars on MLX (host doubles already on the JVM backend).
- **Fixed: Map, Unfold, Switch and Recurse refuse what is not a
  generative function when they are made** (gen.combinators;
  genclj-8326). They took anything and failed at its first call; Scan, Mix
  and Dimap checked already. Each throws `ex-info` with `:reason
  :not-a-generative-function` and the `:value`.
- **Fixed: choice maps' address order is total** (gen.choicemap;
  genclj-7y8r). 1 and 1.0 tied, and so did two addresses of one class
  without an order of their own (lists, maps, sets), so the leaf order
  (`addresses`, `leaves`, `to-array`) followed the order the map was built
  in; a nil address beside another threw a NullPointerException. A tie
  now breaks by class name, then by printed form. The order of such
  addresses may change, and so may that of numbers of classes other than
  Long and Double (BigInt, Ratio, BigDecimal) beside other kinds of
  address: every number now sorts among the numbers, by value.
- **Fixed: CUDA C's negation of a literal** (gen.device.cuda, ext/device,
  internal; genclj-z1gj). It wrote `(--1.5f)`, a decrement NVRTC refuses;
  now `(- -1.5f)`, as the JVM and Metal lowerings write it. The CUDA
  lowering is parked (DESIGN §9, 1.2.1's owner calls): no consumer,
  gen.device's devices stay `:jvm` and `:metal`, and its tests that need
  a CUDA device skip, by name.
- **Fixed: run-chains' batched fallback counts as kernel/run does**
  (gen.inference.device, experimental; genclj-qucg). The device refuses a
  combinator's kernel, so such chains always ran as one batched run,
  where `:acceptance-rates` was the share of steps in which any move was
  accepted and `:divergences` always 0 (the batched moves' `:diverged`
  was dropped). Now they are kernel/run's per chain: accepted moves over
  every move, and the proposals that diverged. And only the device's own
  refusals fall back: a chain whose device run found an error (an index
  out of range, a start whose density is not finite) runs on the
  handler, and its error is thrown, as kernel/run per chain throws it,
  where the batched run, on other draws, could return.
- **Fixed: one chain on `{:device :metal}` runs on a CPU core**
  (gen.inference.kernel/run, gen.inference.device; genclj-cxrx; DESIGN §9,
  option 1). It ran on the handler (Metal's kernel runs many chains and
  refused one), slowly; now the JVM lowering runs it, as with `{:device
  :jvm}`, and its result's `:device` is `:jvm`. Many chains on the GPU
  stay run-chains'.
- **Fixed: Earley items past 2047 symbols** (gen.llm.cfg; genclj-mkos).
  An item packed its dot in 11 bits, so an alternative longer than 2047
  symbols (a long string, an instaparse counted repeat) overflowed into
  its rule: a literal of 2048 bytes was refused, its prefix plus another
  rule's byte taken, or an index went out of bounds. An item is now its
  origin and its position, a number for each rule and dot, so no
  alternative's length is bounded.
- **Fixed: 1.2.0's claim for llama.cpp v0.9.0's branches** (genclj-ajse).
  1.2.0's entry said v0.9.0's unified KV cache serves multi-sequence work
  (token SMC, lockstep sampling) better; it was not measured. Measured
  since on the M4 with Qwen3-0.6B (`bench/gen/llama_branches_bench.clj`,
  docs/research/llm-branching.md), it brings no gain: after a short
  prompt the two builds tie up to 8 branches and v0.9.0 is about 7%
  faster at 16; after an 843-token prompt it is 5-16% slower at 4 to 16
  branches. Nothing in gen.clj changes.
- **Changed: AWRS exponentiates its row once** (gen.llm.awrs,
  experimental; genclj-0d00). A sample computed `exp` over the whole row
  two or three times a draw; it now makes one pass a sample and sums the
  masses after: at 151,936 tokens (Qwen's vocabulary) 2.5 ms becomes 1.0
  ms with every token accepted, 7.0 ms becomes 1.8 with one token in
  seven (the M4). The draws for a key are unchanged, bit for bit. The
  docstring's budget is corrected to what the code does: at most R checks
  for `:recursive` and R + `:max-accepts` - 1 for `:geometric`, where it
  said R + 1 and R + `:max-accepts`.
- **Changed: float `floor-div` is NumPy's** (gen.array, every backend;
  DESIGN §9, 1.2.1's owner calls). It was floor(x / y), which disagreed
  with `mod`: `(floor-div 1.0 0.1)` was 10.0 beside `(mod 1.0 0.1)`'s
  0.0999…, so x = y·q + r failed. Now NumPy's `npy_floor_divide`: (x -
  fmod(x, y)) / y, one less where fmod's sign is not y's, snapped to an
  integer, so `(floor-div 1.0 0.1)` is 9.0. Other values that change: an
  infinite dividend gives NaN (was ±Inf), `(floor-div -1.0 ##Inf)` -1.0
  (was -0.0). A zero divisor still gives x / y; int `floor-div` is
  unchanged.
- **Changed: `kernel/checked-kernel` is `^:no-doc`** (gen.inference.kernel;
  genclj-vjbo). Public only because `kern`'s expansion calls it, as
  `control-value`, it leaves the documented surface (the API snapshot and
  the book's API page); the var stays, so code that calls it is
  unaffected.
- **Changed: one bounded cache of compiled code** (gen.inference.cache,
  internal: gen.inference.mcmc's `:compile?` steps and
  gen.inference.device's programs; genclj-2j7v). The 64 most recently used
  entries in all, where the device kept 64 programs and mcmc 16 steps,
  each cache emptied whole when full: an entry pins its model and data
  arrays (Metal buffers on MLX). And a Metal kernel's handle
  (gen.device.metal) is freed once its compiled program is unreachable;
  it never was.
- **Changed: context-free masks inside a string** (gen.llm.cfg,
  gen.llm.bytes; genclj-jq2d; DESIGN §9, PROVISIONAL). A JSON grammar's
  mask at Qwen3's vocabulary took 238-363 ms with the context inside a
  string (nearly every token allowed: each trie node visited built an
  Earley set), where most of a JSON text's bytes are; now 0.2 ms, about
  what one at a value's start costs. A regex element is a lexeme its DFA
  scans, the Earley sets are a lazy DFA (successors kept, equal sets
  shared), and the potential makes, for each state of each lexeme, the
  tokens that stay inside it, so a mask sweeps only the tokens that end a
  lexeme. Weights and masks are those of before (exact for the text's
  bytes); making a potential costs about its token trie's time. Added to
  gen.llm.bytes: `sweep-into!`, `sweep-roots!`, `sweep-split` and
  `bare-mask`.
- **Changed (internal): gen.combinators, gen.backend.mlx and gen.array
  in parts** (genclj-i1jl; behaviour unchanged, checked against the old
  code by differential runs). gen.combinators sits over six internal
  namespaces (`gen.combinators.calls`, `.elements`, `.chain`, `.switch`,
  `.recurse`, `.wrap`), its records, constructors and class names kept;
  gen.backend.mlx over six (`.array`, `.memory`, `.operands`, `.linalg`,
  `.random`, `.transforms`), so the MLX array's class is
  `gen.backend.mlx.array.MlxArray`; gen.array's scan, compile, vmap and
  particle functions are in `gen.array.transform` behind the same vars.
  The decision log moved from DESIGN.md §9 to `docs/decisions.md`
  (genclj-vp8u); §9 is now a summary of the conventions in force, linking
  each entry, so a citation "DESIGN §9, <title>" still finds it. And
  `.clj-kondo/` lints gen.clj's macros (genclj-08xk): `clojure -M:codegen -m
  clj-kondo.main --lint src ext/…` is clean.
- **Deprecated: `gen.batched/written` and `gen.batched/refused`**
  (gen.batched; genclj-08xk). Public only by accident, gen.batched's own
  helpers (a failure's message, the error `:batched? :required` throws),
  called nowhere else: `:deprecated "1.2.1"` and a docstring note, to be
  removed in 2.0. They behave as before. A collection's reason stays
  `(fallback ps)` and `fallback-reason`, and `project*` stays, the way to
  `project`'s.
- **Added: the chat template's options in agent/run** (gen.llm.agent,
  experimental; genclj-fshl). `agent/run`'s opts take `:template`
  (`:qwen3.8` for Qwen3.8's), `:reasoning-effort` and
  `:preserve-thinking?`, passed to gen.llm.chat for every turn: 1.2.0's
  Qwen3.8 template could not be reached through the agent, which rendered
  Qwen3.5's always. Options the template refuses are refused as
  `chat/render` refuses them (`:invalid-option`), before the session opens.
- **Added: ext/device and ext/commons-math as dependencies** (their
  `deps.edn`; genclj-ri2d). A dependent takes the device lowerings with
  `:deps/root "ext/device"`, so kernel/run's and importance sampling's
  `:device` and `gen.inference.device/run-chains` reach them (1.2.0's
  dependent could not put ext/device on its classpath, and always got
  `:device-fallback`, "not on the classpath"), and gen.dist.commons-math
  with `:deps/root "ext/commons-math"` (with Commons Math 3.6.1); each
  brings gen.clj from the same checkout. The README lists every module and
  how a project takes it. ext/device's tests no longer read bench/: the
  parity bench's models are `gen.parity-models`, in test/.
- **Added: a combinator's batched form reports each move**
  (gen.inference.kernel/batched-form of chain, repeat, cycle and kern;
  genclj-qucg). Beside `:accepted` (did any move take each particle), its
  result holds `:accepted-moves` (each move's n decisions, in order) and,
  when any proposal diverged, `:divergences` (n counts).
- **Added: native suites' skips are counted** (genclj-ruh8). The tests of
  MLX's binding and of llama.cpp pass with no assertions where their
  library or test model is missing; `clojure -X:test` (now
  `gen.test-runner`, cognitect's runner then this) and `-X:test:par` print
  at their end how many of them ran and skipped, and why.
  `-J-Dgen.require-native=mlx,llama` fails a run in which a suite it names
  skipped a test or ran none, for a release gate's runs.
- **Draws: `rng/randint` and `uniform-discrete` in float32** (MLX's default;
  genclj-psgi) draw other integers for the same key, from three float32
  uniforms where one was drawn. In float64 (the JVM backend's default)
  the integers are unchanged, except a draw the old rounding moved (one
  that reached high among them).

## [1.2.0] - 2026-11-06

A language-model release (DESIGN §9, "1.2.0's theme"; genclj-ucfg),
which memo.clj 1.2.0 pins: GenLM parity's core (the potential contract as
laws, potential combinators, a critic in token SMC, adaptive weighted
rejection sampling, instaparse grammars as constraints, context-free
grammars by Earley), Qwen3.8's chat template, llama.cpp v0.9.0 and MLX
0.32.4. The LLM layers are experimental; the stable API changes by one
Breaking change and additions.

Read first, the **Breaking** change: `get`, `nth` and `count` on an array
throw (`:not-a-collection`), where `get` answered nil. Code that looked up
an array's elements with `get` reads them with `gen.array/->clj` or `item`
first.

The owner approved the milestone's decisions (DESIGN §9, 2026-11-05):
loud lookups on arrays; MLX's gathers keep the documented clamp, with an
opt-in check; `temper` keeps zero weights for every beta, and `after`
gives a sequence that never closes its think block no weight; AWRS's
recursive estimator by default; instaparse in the module `ext/instaparse`
(on the `:test` alias), the core's dependencies unchanged; context-free
masks exact for the text's bytes.

Rebuild the native libraries after updating: `native/build-llama.sh`,
`native/build-mlx.sh`, then `clojure -T:build java` (each library's version
is checked when it loads).

Checked before the tag: every test on the JVM backend, the book
included (2,145 tests, 25,866 assertions); on the Mac mini M4 the
full battery on Metal (2,145 tests, 20,685 assertions), on MLX's CPU in
float32 (2,145, 20,701) and its light tier in float64 (2,027, 18,007);
on CUDA, an RTX PRO 6000 with MLX 0.32.4's CUDA 13 wheel, the full battery
in parallel JVMs (all 2,145 tests; the slow MCMC suite in eight JVMs), and
ten exits with none of 1.1.0's abort; llama.cpp v0.9.0's tests on its three
test models on Metal and on CUDA, its float noise inside the tests' band,
and Qwen3.8-27B loading and scoring on CUDA; memo.clj (231 tests), its
ext/synth (17) and GenGDT (15) against the candidate. None failed.

### Known issues

- **MLX clamps an out-of-range array index** in a gather by default, where
  the JVM backend throws (docs/array-api.md §13): `-Dgen.mlx.check-gathers=true`
  checks each gather's indices, at an evaluation per gather.
- **The book on MLX:** some chapters of Parts II-V do host math on results
  and fail on the MLX backend (genclj-ac70). The book's gate is the JVM
  backend, where every chapter passes.
- **Context-free masks** are exact for the text's bytes, not its tokens: a
  vocabulary without a byte the language needs can reach a prefix that no
  token sequence completes (byte-level BPE vocabularies have every byte).

### Changes

- **Breaking: `get`, `nth` and `count` on an array are loud** (gen.array's
  arrays on every backend, the tape's values, staged and batched values;
  genclj-fzez; DESIGN §9). `(get a 1)` answered nil, a
  silent wrong value (rule 6), and `nth` and `vec` threw messages that
  said nothing. Now each throws `ex-info` with `:reason :not-a-collection`,
  naming `gen.array/->clj` (host values), `item` (one element) and `take`
  (an index on the backend); a batched value raises the batched-branch
  error. Code that used `get` on an array and relied on nil must read the
  array first. A 0-d array on the JVM backend is a host number, as before.
- **Changed: llama.cpp v0.9.0** (from v0.7.0; genclj-adsl). Rebuild with
  `native/build-llama.sh` and `clojure -T:build java`: `gen.llama.LlamaRuntime`
  refuses any other libllama. Its unified KV cache serves multi-sequence
  work (token SMC, lockstep sampling) better; llama.h changed only by an
  ftype and a `#define`, so the binding stands, and gen.llm.llama's
  recurrent replay holds (read against v0.9.0's
  llama-memory-recurrent.cpp). Log-probabilities may differ from v0.7.0's
  within llama.cpp's float noise.
- **Changed: MLX 0.32.4 with mlx-c 0.8.0** (from 0.32.2 and 0.7.0;
  genclj-4lzg). Rebuild with `native/build-mlx.sh` and `clojure -T:build
  java`. MLX's CUDA wheel no longer aborts the JVM at exit (1.1.0's known
  issue: the fix is upstream from 0.32.3), so the source build drops its
  backported patch. The binding is regenerated from mlx-c 0.8.0's headers
  (new functions; distributed groups' signatures, which gen.clj does not
  call).
- **Added: an opt-in check of MLX's gathers** (gen.backend.mlx;
  genclj-08un; DESIGN §9). With
  `-Dgen.mlx.check-gathers=true`, `gen.array/take` and `take-along-axis`
  read their array indices and throw on one out of range, as the JVM
  backend does, at an evaluation per gather. Off by default: the
  documented clamp is unchanged.
- **Added: the potential contract as laws** (`gen.laws.potential`, in
  ext/laws, experimental as gen.llm.potential is; genclj-uxf6). Four laws
  over a potential's weights, run as gen.laws runs the GFI's
  (`potential-laws/verify`, `potential-laws/deflaws`): the next-token
  weights are the prefixes' differences (and complete less prefix at the
  end), the weights factor autoregressively, a context of zero prefix
  weight has zero-weight extensions and no complete weight, and the
  weights are pure functions of the context (a stateful potential's state
  must not leak). Cases are random walks over the tokens of finite weight;
  `:tokens` samples the tokens compared, for a large vocabulary. It
  replaces the test-only `gen.llm.potential-contract` check, which was
  never part of a release's API. `gen.laws/failure-of` takes a second
  argument, a function that describes a case of another shape.
- **Added: potential combinators** (gen.llm.potential, experimental;
  genclj-f3dv; DESIGN §9). `temper` (weights times beta, zeros kept),
  `normalize` (next weights as a distribution at every context),
  `prompted` (an LM's log-probabilities after a prompt: prompt
  intersection under another completion, a temperature when tempered),
  `coerce` (a potential over bytes read through a vocabulary, as a
  grammar over a byte vocabulary through an LM's tokens) and `after` (a
  potential over the tokens after a close token, for a think block), with
  `potential?`. `product` checks its factors are potentials when it is
  made. In gen.llm.bytes, `weigh` (every token's weight from its bytes')
  and `vocab-bytes`.
- **Added: a critic in token SMC** (gen.llm.smc's `:critic`;
  genclj-di22). A second potential applied at the drawn token only: a
  particle's weight gains the critic's prefix weight difference (its
  complete weight at an end or at max-tokens), one call a particle and
  step, and the target gains the critic's weight. The proposal never
  reads it: cheap potentials propose, expensive ones (a code check, a
  classifier) criticize.
- **Added: AWRS, adaptive weighted rejection sampling** (`gen.llm.awrs`,
  and token SMC's `:proposal :awrs`; genclj-oigp; DESIGN §9). For a
  potential with no cheap row (a parser, a reader on partial text): a
  token drawn from the LM's row restricted to the tokens of finite twist,
  checking only the tokens it draws, with an unbiased estimate of their
  mass as the weight (Lipkin et al., COLM 2025). The recursive (default)
  and geometric estimators, each with an optional budget of rejections
  (`:awrs {:method :max-rejects :max-accepts}`).
- **Added: instaparse grammars as constraints** (`gen.llm.instaparse`, a
  new module `ext/instaparse`, experimental; genclj-kb88; DESIGN §9). An
  instaparse grammar (EBNF text or a parser) read as a regular language,
  `(ip/term g)`, and as a grammar over a vocabulary, `(ip/grammar m g)`,
  with gen.llm.grammar's exact masks; recursion unrolled to `:max-depth`.
  A dependent takes it with `:deps/root "ext/instaparse"`; it brings
  instaparse 1.5.0. In gen.clj's repository the `:test` alias has it.
- **Added: context-free grammars as potentials** (`gen.llm.cfg`,
  experimental; genclj-h5rb; DESIGN §9). Rules over bytes (rules,
  strings, regexes), recognized by Earley's algorithm byte by byte:
  `cfg/recognizes?`, `cfg/prefix?`, and `cfg/potential` (prefix and
  complete weights, masks by a sweep of the token trie) for recursive
  JSON, brackets, code's nesting. `gen.llm.instaparse/cfg` reads an
  instaparse grammar into one, recursion and all. In gen.llm.bytes,
  `sweep-with` (a mask from any recognizer's state carried down the trie).
- **Added: Qwen3.8's chat template** (gen.llm.chat, experimental;
  genclj-6m7t). `(chat/render messages {:template :qwen3.8})`: with
  thinking on, the reasoning-effort sentence opens the system turn
  (`:reasoning-effort` :xhigh by default, :medium, :low), and every
  assistant turn keeps its think block (`:preserve-thinking?`, true by
  default), as the template Qwen3.8's model files carry; checked against
  its own renderings. Qwen3.5's and 3.6's rendering (the default) is
  unchanged.
- **Fixed: a boxed-math warning on loading gen.llm.llama** (genclj-tpvn).
  A new test loads every source namespace in a JVM of its own and fails
  on any reflection or boxed-math warning.

## [1.1.0] - 2026-11-05

A consolidation release (DESIGN §9, "gen.clj 1.1.0"; genclj-qsa2): Gen.jl
0.4.9 parity, the bugs the book found in stable namespaces, 1.0.0's CUDA
known issues, and what main gained since v1.0.0. Every change to the
stable API is an addition (the API snapshot: `kernel/run`'s `:device`,
`mcmc/dense-inv-metric`, `laws/*score-scale*`).

Read first, two **Breaking** changes:

- `=` on a 0-d MLX array and a host number or boolean throws, where it
  answered false: compare draws with `array/==`, or inside a `gen` body.
- A callee spliced without an address runs under its caller's
  middleware: middleware of your own on such a model now changes those
  sites too.

The **Draws** lines change the values for a key, not their distribution:
full float32 matrix products on CUDA, HMC's and MALA's divergence rule,
llama.cpp's threads on a Linux hybrid CPU.

Checked before the tag (genclj-w2pg), on code identical to the tag's
(only tests and docs changed after the runs began): on the JVM backend,
every test and the whole book (2,052 tests, 25,167 assertions); on Metal,
the full battery (2,053 tests, 20,005 assertions); on MLX's CPU, the full
battery in float32 (2,053 tests, 20,021 assertions) and the light tier in
float64 (1,935 tests, 17,327 assertions, on an M2 Max); on CUDA, the full
battery on a Jetson AGX Thor (MLX built from source), every namespace
once across five JVMs side by side; on an RTX PRO 6000 (MLX's CUDA wheel),
the namespaces that failed under TF32 at 1.0 with conformance and the MLX
backend's tests (222 tests), and with TF32 forced on, 1.0's 29 failures
again; memo.clj (230 tests on the JVM backend, 231 on Metal), its
ext/synth (11) and GenGDT (15) on the release candidate. The runs
found three faults, all in tests (a dense-metric test making float64
arrays on a GPU; tempering-test's importance-sampling claim resting on one
key's luck, and its exact value computed in float32), each fixed and
rerun on every backend; none in gen.clj.

### Known issues

- **CUDA: the JVM may abort at exit** with "Destroy(handle_) failed:
  driver shutting down" when MLX comes from its CUDA wheel, after the
  program's work is done (5 runs of 10 on an RTX PRO 6000), so the exit
  status is not the result. The source
  build (`native/build-mlx.sh cuda13-source`) carries MLX v0.32.3's fix; a
  later MLX pin will bring it to the wheel (genclj-4lzg).
- **MLX clamps an out-of-range array index** in a gather, where the JVM
  backend throws: `gen.array/take`'s documented contract for a lazy
  backend (docs/array-api.md §13), which the conformance suite now checks.
  Whether MLX should be loud is the owner's call after 1.1.0
  (genclj-d05w, genclj-08un).
- **The book on MLX:** some chapters of Parts II-V do host math on results
  and fail on the MLX backend (genclj-ac70). The book's gate is the JVM
  backend, where every chapter passes.

### Changes

On main since v1.0.0.

- **Breaking: a callee spliced without an address runs under its caller's
  middleware** (`gen.dynamic/with-transitions`; d6ca867; DESIGN §9;
  genclj-u23m). Its choices are the caller's, so the caller's
  middleware now runs at its sites too, unless the callee has middleware
  of its own for the operation, which runs there as before; such a
  callee's body ran under the standard transitions. So gen.inference's
  middleware reaches those choices: `enum/with-exact-proposal` enumerates
  them, and `conj/with-conjugate-proposal` finds a latent or child there,
  where both threw "… is made by a generative function the model calls at
  an address; wrap that one"; `with-internal-proposal`'s project and
  regenerate replay a proposal's own address-less splices, where they
  threw; and the reparameterization estimator (`gen.inference.vi`) refuses
  a choice there from a family without a pathwise sampler, which it used
  silently. Middleware of your own that changes the model changes it at
  those sites too. The errors now say what applies: with-exact-proposal's
  names the address of the call that made the choice, or the spliced
  callee's own middleware, and an address-less splice of a generative
  function that is not a `gen` (a combinator, a distribution, a wrapper)
  says to give the splice an address. Breaking for middleware of your own
  on a model with such a splice: it reads another result there.
- **Fixed: half-float sums on CUDA in float32** (gen.backend.mlx; b3feee7;
  genclj-tblx). A float16 or bfloat16 sum on CUDA accumulated in its own
  precision (1.0.0's known issue: a long sum lost precision), as MLX's CPU
  did before it summed them in float32 (DESIGN §9): CUDA now does too.
  Metal, which accumulates them in float32, is unchanged.
- **Draws: MLX's float32 matrix products are full float32 by default**
  (gen.backend.mlx's MlxRuntime; 3ff86ae; DESIGN §9; genclj-p3pr). MLX
  computes them on CUDA in TF32 unless `MLX_ENABLE_TF32=0` is set (a
  10-bit mantissa, about 1e-3 relative, which failed the matrix families'
  gradients, the batched matrix VJPs and the Cholesky and solve checks:
  1.0.0's known issue). gen.clj now sets it to 0 when MLX loads, unless
  the environment sets it, and `-Dgen.mlx.tf32=true` (or `false`) decides
  over both. On CUDA a matrix model's values for a key change, to full
  float32's; on the M4 nothing changes; a Mac whose Metal runs float32
  products through its neural accelerators keeps full float32 too.
- **with-internal-proposal's regenerate drops the choices its new run
  leaves** (`gen.inference.proposal`, the thesis's Eq. 4.1; 746095e;
  genclj-bnlx).
  A selected choice that decides whether an unselected one exists (the
  thesis's abc, regenerating a alone of {a T b F c T}) threw "regenerate
  leaves an unselected choice unvisited", where Eq. 4.1 drops b
  (B ⊆ A_σ); the laws regenerate-weight-is-the-projection-ratio and
  regenerate-keeps-unselected-choices failed by exception on such an
  override. The move now drops it, with Eq. 4.1's weight: q proposes
  given every unselected choice (σ) and the reverse move given
  σ' = τ'|_{S^c} (DESIGN §9). A move that drops nothing
  keeps its draws, and its weight unless the reverse move's q reads a
  choice the move added unselected (σ' holds it now).
  `with-exact-proposal` and `with-conjugate-proposal` still throw on a
  move that drops a choice. The book's *The Generative Function
  Interface* no longer lists it as a departure from the thesis.
- **Draws: HMC and MALA count an energy error past 1000 as a divergence**
  (`gen.inference.mcmc`, as NUTS does: Stan's max_deltaH; 4f22cae;
  DESIGN §9, genclj-gqd3). hmc, mala, hmc-particles, mala-particles and the device
  chains counted only non-finite energies, so a finite energy error of
  1e28 (the book's chapter 023, at eps 0.11) was rejected but reported
  with `:diverged? false` and not counted in `:divergences`. A proposal
  whose −log α (HMC's H′ − H; MALA's, which is one leapfrog step's)
  passes 1000 now diverges: `:diverged? true`, `:log-alpha` −Inf,
  counted by kernel/run, nuts/sample's `:kernel :hmc` and run-chains.
  Its acceptance probability was below e^-1000, so an accept decision
  changes only where the uniform drawn was exactly 0, which accepted it.
- **Arrays print as gen.serialize writes them, so printed text reads
  back** (the backends' print methods, gen.choicemap; ee1c060;
  genclj-5k7q). A
  backend array printed as `#gen/array[:float64 [[1.0 2.0]]]`, under the
  tag that `serialize/read` takes as a map, so `(serialize/read (pr-str
  c))` threw on a choice map with an array leaf. It prints as
  `#gen/array {:dtype :float64, :shape [1 2], :data [1.0 2.0]}`, write's
  text (its elements row-major, which `*print-length*` shortens), and a
  choice map's leaf whose value is a map prints tagged `#gen/value`, as
  written, where it printed as a node. A printed array, choice map or key
  reads back. gen.serialize's text is unchanged.
- **Parity with Gen.jl 0.4.9** (genclj-u420). Its fixes (#571: the
  general trace translator's forward score, elliptical slice's threshold)
  were gen.clj's from the port, and its new tests are ported; its hmc
  metric (#559) is the dense metric below. The oracle's fixtures
  (`test/fixtures/genjl`) come from Gen 0.4.9 now, with the same values
  as 0.4.8's, and Gen.jl 0.4.9's own suite passes here.
- **A dense metric for hmc, mala and nuts** (`gen.inference.mcmc`, Gen.jl
  0.4.9's #559; DESIGN §9, owner, 2026-11-04; genclj-q9ge).
  `:inv-metric` takes a dense M⁻¹ (Stan's dense inv_metric):
  `(mcmc/dense-inv-metric addresses matrix)` names the order of its rows,
  and a bare `[n n]` matrix is taken when one choice of n elements is
  selected. Momenta come from N(0, M) as L⁻ᵀ·z (M⁻¹ = L·Lᵀ) and
  positions move by M⁻¹·p; MALA proposes from N(v + tau·M⁻¹∇,
  2 tau·M⁻¹). A dense metric that is not finite, not symmetric (within
  1e-8 of its largest entry in float64, Stan's, and 1e-5 in float32, a
  computed metric's rounding; its symmetric part is the one used), not
  positive definite or not the selected choices' is refused
  (`:invalid-inv-metric`), and a bare matrix for several choices
  names the dense form in its error. `mcmc/inv-metric` returns a dense
  metric labeled in the selection's order. gen.inference.nuts' warmup
  adapts a diagonal metric only: it refuses a dense `:inv-metric` unless
  `:adapt-metric?` is false (`:dense-metric-adaptation`), and then keeps
  it. The diagonal path is unchanged, bit for bit. Gen.jl's `metric` is
  M and gen.clj's option its inverse; Gen.jl 0.4.9's hmc also leaves M⁻¹
  out of its position step, so with a metric its moves differ from
  gen.clj's, though both target the same distribution.

- **Experimental: an LLM's sampling, chat and agent loop in gen.llm**
  (outside the 1.x promise; DESIGN §9, owner, 2026-11-04; genclj-uw7w).
  `gen.llm.sample`: temperature, top-k and top-p, and n completions of
  contexts drawn in lockstep over the LM protocol, a loud error when they
  would not fit the LM's cache. `gen.llm.chat`: Qwen's chat template
  (thinking on or off, tools in its format) and the tool-call parser.
  `gen.llm.agent`: a tool-using agent's episodes (turn and context
  budgets, K candidates and a chooser, a reply cut inside its think block
  a loud error), its tools the caller's functions. The core evaluates no
  generated code.
- **A module, ext/codegen** (`gen.codegen`; a git dependency with
  `:deps/root "ext/codegen"`, as ext/laws): what evaluates LLM-written
  code, a namespace session with read, repair, vet and explain hooks, and
  `:each?` to show every form's value (`gen.codegen.session`) and tools in a JVM of their own
  (`gen.codegen.process`), both experimental; and the codegen bake-off's
  harness (internal), moved from test/gen/codegen. Its dependencies
  (clj-kondo, parinferish) left the :test alias for :codegen, which puts
  the module on the classpath: `clojure -X:test:codegen` runs its tests
  too.
- **Draws: llama.cpp's default threads on a Linux hybrid CPU are its
  physical performance cores** (`gen.llm.llama`, genclj-xe8c). Without
  `:threads`, llama.cpp used every core, and on a Core Ultra 9 285K (8
  P-cores, 16 E-cores) the 3.6-35B decoded 4.4 times slower than on the
  8 P-cores alone (8.9 against 38.9 tokens/s, measured by memo.clj's
  session). The engine now passes the P-cores' count, read from
  /sys/devices/cpu_core/cpus; elsewhere nothing changes. Another thread
  count can change llama.cpp's float noise, so the draws for a key on
  such a machine may differ.
- **gen.laws: close?'s absolute tolerance scales with the scores** (owner,
  2026-11-01; DESIGN §9, genclj-6txl). It was 1e-12 in float64 (float32's
  1e-3), which assumed scores up to ~100: correct models whose traces
  score in the thousands failed update and regenerate laws by ~1e-11 (the
  codegen grader found it on four of the book's models). It is now that
  times the largest |score| of the traces a check's weights come from
  (`laws/*score-scale*`, which each law binds; 1 outside a law, so a
  reader's own `(close? x y)` is unchanged), the relative tolerances as
  before. A planted error of 1e-6 still fails for scores below ~1e5.
- **gen.laws: a weight that is 0 compares as weights do** (DESIGN §9,
  2026-11-04; genclj-s5fj). Five checks asked for exactly 0:
  generate without constraints, an update without constraints, a
  regenerate of nothing and of everything, and project on nothing. A
  hand-written generate that computes its weight as the score less the
  log density of what it drew is 0 only up to rounding, so it had to
  special-case "no constraints" (chapter 045's urn did). They now use
  `laws/close?` at the scale of the trace's scores, as every other weight
  check; a planted error of 1e-6 still fails in float64 (0.1 in float32).
- **Experimental: tempered SMC** (`gen.inference.tempering/smc`, outside
  the 1.x promise; DESIGN §9, genclj-87x1): the log marginal likelihood
  of observations under any generative function, and posterior
  particles, by adaptive annealing from the prior to the posterior with
  Metropolis-Hastings moves on each tempered target (regenerations from
  the prior, random walks for continuous latents). It works where
  importance sampling from the prior fails (a posterior far narrower than
  the prior) and for models whose structure varies. Checked against
  exact values: a conjugate normal model, a branch on a draw, a Kalman
  random walk, an enumerated HMM. The estimate is consistent; with
  `:betas` and `:scales`, a previous run's, nothing adapts and its
  exponential is unbiased.
- **Breaking: = on a 0-d MLX array and a host number or boolean throws**
  (gen.backend.mlx). It answered false, so plain Clojure outside a gen
  body, (= k 0) on a draw, took the other branch on every run where the
  JVM backend compares values. Compare draws with gen.array
  (array/== k 0), or inside a gen body. Other comparisons are unchanged.
  The test generator's :select sites compared that way: on MLX they took
  the second branch always (DESIGN §9, owner, 2026-10-31; genclj-8rdm).
- **gen.clj is under the MIT license** (`LICENSE`). The ports of Gen.jl's
  test suite in `test/gen/genjl/` keep Gen.jl's Apache License 2.0, a copy
  beside them (owner, 2026-10-26; genclj-fhdz).
- **A shared matrix times per-particle vectors is one matrix product in
  batched runs** (`gen.array.batch`): the particles' vectors as the rows of
  one product with the matrix, which reads it once, where the broadcast
  product read it once per particle (on MLX, one GEMV per particle).
  Batched importance sampling on the parity suite's ndreg cells (1000
  particles, D × M = 1000 × 10000) takes 6.8 → 4.3 s on the JVM backend
  and 102 → 23 ms on MLX's GPU (M4, without spot checks). The JVM
  backend's values are bit-identical; MLX's can differ at float32
  rounding (genclj-c8gl).
- **Experimental: staging, the core of the device programs**
  (`gen.array.stage`, outside the 1.x promise; DESIGN §9, genclj-yuau).
  `stage` runs a function once on a backend that records gen.array's
  operations as a graph (gradients through the tape, particles through the
  batch tracer, draws through Threefry on staged keys) and returns the
  program; `evaluate` replays a program on an eager backend, equal on the
  JVM backend in float64 to the function run directly, bit for bit.
  `*float*` stages float32 or float64 programs. The lowerings (generated
  JVM code, Metal through MLX, CUDA C) are the extension `ext/device`.
- **Experimental: `gen.array/scan`**, a loop of n steps, `(f carry x)` →
  `[carry' y]`: a host loop on an eager backend, one loop node when staged.
  A var may now carry its own tier: a stable namespace's var marked
  `{:api :experimental}` is outside the surface and the promise.
- **gen.random takes staged keys** (`split`, `fold-in`; `key?` is true of
  them): a staged key is a value too (rule 2). Host keys are unchanged.
- **Experimental: `:device` on `kernel/run` and importance sampling**
  (`gen.inference.device`; the lowerings' entry point `gen.device` in
  ext/device; outside the 1.x promise; DESIGN §9, genclj-fpnf).
  `{:device :jvm}` stages the run once from gen.clj's own code (kernel/run's
  chain as one scan of the kernel's step, importance sampling's particle as
  one constrained generate), compiles it with the device lowerings and runs
  it. The result is the handler's (bit for bit on the JVM backend in
  float64) with `:device` added, and its traces are whole: an update from
  one is the handler's. `:metal` runs importance sampling's particles on
  the GPU. A run that does not stage or lower (a branch on a draw the chain
  moves, a generative function made anew each run, an op a lowering lacks,
  ext/device absent) runs on the handler, its result saying why in
  `:device-fallback {:device :reason}`. A compiled program is reused by a
  run with a kernel of the same kind and parameters, the same model and
  argument arrays and the same length. On the parity suite's linear
  regression (M4, JVM backend), kernel/run's MALA for 1000 steps takes
  113.6 → 5.5 ms (the static model's 226 → 12.9 ms) and importance
  sampling of 1000 particles 9.1 → 3.8 ms. kernel/mala and kernel/hmc
  carry their step for it (`:device-f`, `:device-key`), and kernel/run's
  arglist names `:device`.
- **Experimental: kernel/run's `:device` path costs little beyond its
  program** (`gen.inference.device`, `gen.array.stage`; DESIGN §9, owner,
  2026-11-04; genclj-rtcd). Its result is built on the host from the
  program's outputs: each kept step's trace from that step's rows alone,
  0-d leaves made from their values, where it made a backend array of
  every output and sliced it; a leaf the chain never changes (the
  observations) is the start's in every trace and no longer an output of
  every step. gen.array.stage records where(c, x, x) as x, no node. One
  chain of 1000 steps on the M4, `{:device :jvm}`: HMC L=32 4.4 → 2.4 ms
  on the JVM backend; on MLX 7.2 → 4.7 ms, and 139 → 4.9 ms with a
  `:keep` per step (each kept value was a lazy slice, a GPU evaluation
  to read). The results are as before: the handler's bit for bit on the
  JVM backend in float64, to float32 rounding with the same accepts on
  MLX (a test of its own now).
- **Experimental: `gen.inference.device/run-chains`, many chains in one
  call** (DESIGN §9, owner, 2026-11-04; genclj-ke07). n chains of a
  kernel, chain i from its own trace on `(rng/split key n)[i]`, each
  `kernel/run`'s: `{:device :jvm}` runs them on every core, `:metal` as
  one Metal kernel (a GPU thread a chain); without `:device`, `kernel/run`
  per chain. `:keep-choices [address …]` keeps choices straight from the
  program's outputs, no trace made (a function `:keep` makes a trace a
  kept step: millions for many chains). Returns `{:traces` or `:draws`,
  `:acceptance-rates :divergences}`, one entry per chain. A model the
  device cannot run falls back to one batched run of the kernel's batched
  form (`:device-fallback {:ran :batched}`; its draws the batched run's,
  the same distribution), chain by chain for a kernel without one. The
  device lowerings gained per-run inputs (each chain's start), one shared
  input for what every chain holds (the observations), and one input for
  an array a trace holds twice (a model whose return value is its
  observation, which every step wrote out: past a Java array's length at
  4096 chains of a 2000-observation model); outputs too large for a Java
  array are a loud error before anything is allocated. `:metal`'s outputs
  are copied into Java arrays when first read, and a 0-d input goes to
  Metal as [1] (MLX hands a kernel a 0-d array as a scalar): the static
  parity model's importance sampling runs on `:metal` now (genclj-649c).
  On the M4 with `:keep-choices`, 1000 steps: 4096 MALA chains 0.38 s on
  `:jvm` and 0.18 s on `:metal` (32768: 1.5 s), 4096 HMC chains of L=32
  0.82 s and 0.22 s (32768: 1.9 s), against GenJAX's 0.89 s for the 4096
  MALA chains on the M4's CPU, its device there. A `:keep` function, or no
  `:keep`, makes a trace a kept step and chain: on MLX past 50,000 traces
  that is a loud error naming `:keep-choices` (Metal holds about 499,000
  arrays at once).
- **Experimental: importance sampling's `:device` result built on the
  host** (`gen.inference.device`, ext/device's `gen.device`; genclj-m5o6).
  `gen.device`'s run-many gives its outputs on the host (`:metal`'s read
  once from the GPU), and each particle's trace is made from its rows; an
  observation is the value as given, as the handler's trace holds it (on
  MLX it was a float32 array of the program's), and no longer an output of
  every particle. On MLX a trace's leaves were lazy slices of the
  outputs, each read a GPU evaluation: 3000 particles with every trace's
  slope read took 423 → 13 ms, 30000 took 4.6 → 0.39 s, and `:metal`
  completes now. On the JVM backend 30000 particles take 91 → 34 ms.
  Results otherwise as before: the handler's bit for bit on the JVM
  backend, to float32 rounding on MLX (a test of its own now).

## [1.0.0] - 2026-10-26

The first release, tag `v1.0.0` (the release gate, genclj-v97q).
Checked before the tag: on the JVM backend, the battery and the whole
book; on Metal and on MLX's CPU in float32, the full battery (1981 tests,
0 failures each); on MLX's CPU in float64, the light tier; on CUDA (an RTX
PRO 6000), the full battery, its failing and unreached namespaces again
with `MLX_ENABLE_TF32=0`, all passing but one half-float sum (Known
issues); and memo.clj (JVM and Metal) and GenGDT on the release
candidate.

### Known issues

- **CUDA: set `MLX_ENABLE_TF32=0`.** MLX computes float32 matrix products
  on CUDA in TF32 by default, a 10-bit mantissa (about 1e-3 relative): a
  log density of 100 can be off by 0.1 nats. On the release run it failed
  the matrix families' gradients, batched matrix VJPs, a vmapped matmul,
  and Cholesky and solve checks; with `MLX_ENABLE_TF32=0` in the
  environment they pass. gen.clj will turn TF32 off itself, with an
  opt-in (DESIGN §9, owner, 2026-10-25; genclj-p3pr).
- **CUDA: half-float sums** accumulate in their own precision, as MLX's
  CPU did before genclj-y0mp: a long float16 or bfloat16 sum may lose
  precision (backend conformance's long-half-float-sums fails there). Sum
  in float32 (genclj-tblx). Metal and MLX's CPU are unaffected.
- **CUDA: the JVM aborts at exit** with "Destroy(handle_) failed: driver
  shutting down" when MLX comes from its CUDA wheel, after the program's
  work is done, so the exit status is not the result (genclj-codn).
  native/build-mlx.sh's source build (`cuda13-source`) applies MLX
  v0.32.3's fix for it.
- **The book on MLX:** some chapters of Part II do host math on results
  and fail on the MLX backend (genclj-ac70). The book's gate is the JVM
  backend, where every chapter passes.

Changes on main since the post-M5 review's fixes (bd6b8b4, 2026-10-22),
newest first in each section; the milestones before them are summarized
at the end.

### Changed

- **Breaking. The handler's state and middleware are public API** (DESIGN
  §9, "The handler's state and middleware"; genclj-q6sc).
  - `(dyn/initial-state op)`, `(dyn/initial-state op opts)`: the state an
    operation starts from, its sixteen keys documented in the docstring,
    for calling a transition directly. An unknown op or input is an error.
  - Middleware's own keys reach the result: a transition that writes a
    qualified key of its own namespace (`::seen`) finds it on the trace
    the run makes (simulate, generate, update, regenerate) or in the
    result of assess and propose. Each run starts without them; a
    batched run's are on its batched trace, batched where computed from
    batched values. A trace made under such middleware holds those keys,
    so it is no longer `=` to one made without it.
  - Breaking: middleware is checked. A transition that changes `:mode`,
    `:prev`, `:selection`, `:argdiffs`, `:key`, `:visited`, `:subtraces`
    or the ω keys, records no choice at its path, or returns other than
    `[value state']` throws, naming the path and key; so does a run whose
    middleware left an unqualified key the state does not have, or a key
    of gen.trace's. `:constraints` stays writable. Writing the totals
    (`:choices :scores :score :weight :discard`) is unsupported.
- **Breaking. gen.laws moved to the module ext/laws** (DESIGN §9, "The
  laws for users"; 4a910bf, genclj-u6je).
  - Its file is `ext/laws/src/gen/laws.clj`, its namespace still
    `gen.laws`, and it needs nothing of gen.clj's `test/`. A dependent
    takes it as a git dependency on gen.clj's repository,
    `{:git/url … :git/sha … :deps/root "ext/laws"}` at the commit of the
    gen.clj it uses: its deps.edn brings test.check, and gen.clj from the
    same checkout unless the dependent names `gen.clj/gen.clj` itself.
    gen.clj's `test/` on the classpath no longer gives it. In gen.clj's
    repository `:test` and `:laws` (`clojure -M:laws`) include it.
  - **Breaking.** `identity-update-reports-no-change`, a convention of
    gen.clj's that a correct generative function may fail, left
    `laws/forward-sampling` (7 laws now, `laws/all` 22) for
    `laws/conventions`, so `deflaws` over the default set or over
    `laws/encapsulated` no longer runs it. gen.clj's own suites run
    `(merge laws/all laws/conventions)`.
  - A `deflaws` spec prints nothing: it printed each run's result map
    through defspec's reporter. It draws a fresh seed at each run, as
    before and now documented in its docstring; `(spec n :seed s)` fixes
    it. Its law set may be any expression of sets, evaluated when the
    `deflaws` compiles: `(merge laws/all laws/fixed-support)`.
  - `propose-weight-is-the-assessed-density` and `dispatch-matches-handler`
    give assess the case's key, where they drew one from the implicit
    source, so a law's run is a function of its seed (a generative function
    with untraced randomness failed them differently at each run).
  - Loud: models that are not model entries (`laws/cases`), options that
    are not a number or a map of known keys (a law set in their place),
    and a law set that is not a map of laws.
- **Breaking. Draws. A batched particle filter stays batched through
  rejuvenation** (DESIGN §9, "Kernels' batched form"; af25c97,
  genclj-xgz4).
  - `pf/rejuvenate` of a batched state moves every particle at once when
    its kernel has a batched form (`kernel/batched-form`: `kernel/mh` of a
    selection or of a proposal, `hmc`, `mala`, `map-optimize`, and
    `chain`, `repeat`, `cycle` and `kern` of them), and the state keeps
    `:particles`. It returned a state of `:traces`: read the particles
    with `pf/traces` or `pf/choice`. The draws for a key change, from the
    same distribution. With another kernel (`nuts`, `elliptical-slice`,
    `involutive-mh`, `mix`, `checked`, `kern ^:trace`, a function) it
    moves one particle at a time as before, and the state records
    `:fallback`, saying so; a `:batched? :required` filter throws instead.
  - `pf/mean`, `pf/variance` and `pf/proportions` of a batched state run
    their function once, on the batched trace (`gen.batched/map-particles`),
    where they rebuilt every scalar trace on each call. Forty batched
    filters of chapter 053's model (100 particles, 10 steps, MH
    rejuvenation, ten means each) took 3.3 s, 2.9 s of it rebuilding, and
    take 0.23 s now, against 0.5 s for its scalar filters.
    A function that cannot run batched (it reads a value on the host)
    runs on each trace, and a `:batched? :required` filter throws. Their
    sums may differ from before in the last bits.
  - `kernel/->Kernel` takes four fields, `[f reverse-f particles
    reverse-particles]`, the last two the batched forms (nil for none):
    `kernel/kernel` and `kernel/symmetric` build kernels as before.
  - `gfi/assess-backward`'s check reads every particle of a batched move
    as a validation (`array/any-true?`), where its host read made the
    move fall back to one particle at a time. A particle that fails it
    fails the batched run, and its scalar run raises the same error.
  - An unchecked gen.batched collection (after `select`, `set-choices` or
    an unchecked update) is spot-checked once, however often it is read:
    every read re-ran the spot check's particles, 0.75 ms of each
    `pf/mean` of chapter 053's filters. The errors are the same.
- **The transform checks' messages say what ran** (2e3f9d6, genclj-xsml). An
  involution's `:check?` throws "The involution applied twice does not
  give back the trace" (or "the forward choices"), where it said "The
  inverse transform does not give back …", which a translator's check
  still says. `tf/inverse` of a transform made by `tf/transform` without
  one throws "The anonymous transform (tf/transform) has no inverse …",
  where the name printed empty.
- **Breaking. `:inv-metric` is labeled by address** (DESIGN §9, "Labeled
  metrics"; f04ef86, genclj-qs62).
  - `mcmc/hmc`, `mala` and `nuts`, their `-particles` kernels, the
    kernels of gen.inference.kernel and `nuts/adapt` and `sample` take
    a variance for each selected choice by its address: a choice map or
    a map, `{:slope 0.012 :intercept 0.46}`, each a number or an array of
    the choice's shape. One number for every value is unchanged.
  - A vector or array of variances for a selection of more than one
    choice is an error (`:unlabeled-inv-metric`). It followed the choice
    map's flat order (addresses sorted), not the selection's, so
    `{:inv-metric [0.012 0.46]}` with `(sel/select :slope :intercept)`
    gave 0.012 to `:intercept`. Write the map in its place. For a
    selection of one choice a vector is taken as before.
  - `nuts/adapt` and `nuts/sample` return `:inv-metric` as a choice map
    of variances by address, where it was a flat array in that order;
    with no metric given or adapted, the identity as ones. Read
    `(cm/get-choice (:inv-metric r) :slope)`; passing it back as
    `:inv-metric` works as before.
  - A metric that lacks a selected address, or holds one not selected,
    is an error (`:invalid-inv-metric`).
- **Breaking. `learn/train` is Gen.jl's `train!`: epoch datasets, a
  held-out history, a callback and resumable state** (DESIGN §9, "The
  1.0's shape", training's history; 56d81a3, genclj-59cf).
  - Each epoch generates `:epoch-size` examples once, from the data
    function, and each of its `:batches-per-epoch` minibatches draws
    `:batch-size` of them without replacement. Every example of every
    minibatch was drawn fresh. The defaults are Gen.jl's, 1 each, and a
    `:batch-size` above `:epoch-size` is an error, so a call that passed
    `:batch-size` alone passes `:epoch-size` too (at least as large).
  - The history holds each epoch's mean generate weight on
    `:evaluation-size` fresh examples (default `:epoch-size`), held out
    from training and scored under the epoch's final store. It held the
    mean score of the epoch's last minibatch, which measures another
    quantity when the constraints leave choices open.
  - It returns `{:store :state :history}`: `:state` is the optimizer's,
    which the option `:state` takes back to resume training. `:callback`
    is called after every step with `{:epoch :minibatch :objective
    :store}`.
  - **Draws.** Training's draws for a key differ.
- **Breaking. Optimizers' options are checked when they are made**
  (ccd6b7e, genclj-y2ab). A missing or non-numeric option, a negative rate or
  weight decay, a momentum, decay or beta outside [0, 1), and an eps or
  max-norm of 0 or less throw ex-info naming the option (`{:option
  :value}`). A missing rate failed at the first step with a
  NullPointerException, and a negative rate stepped downhill. An unknown
  `:kind` names every kind that has a `-step`, those added outside
  gen.learning too, in `:kinds` (sorted by name).
- **Breaking. Every kernel returns `{:trace :accepted? …}`** (DESIGN §9,
  "The 1.0's shape", kernels' shape; a509ad5, genclj-4hu0).
  - `mcmc/map-optimize` and `mcmc/elliptical-slice` return `{:trace
    :accepted?}` where they returned the trace: read `:trace`.
    map-optimize's `:accepted?` is whether its line search found a step,
    elliptical slice's always true. `mcmc/map-optimize-particles` returns
    `{:particles :accepted}` (and `:fallback`) where it returned the
    particles: read `:particles`.
  - `mcmc/nuts` adds `:accepted?`, whether the state drawn is other than
    the start, and so does each of `nuts/sample`'s `:stats`.
    `particle-mcmc/particle-gibbs` adds `:accepted? true`. A combinator
    adds `:accepted?`, whether any of its moves was accepted, beside
    `:accepted`.
  - A kernel's result without `:trace`, such as a bare trace, is a loud
    error in `kernel/run`, the combinators and `pf/rejuvenate`, where it
    carried nil on as the trace.
- **Breaking. `(kernel/kern [] …)` and `(kernel/kern ^:trace [tr] …)` are
  the kernel** (a509ad5, genclj-4hu0), where they were functions of no arguments
  that returned it: drop the extra call, `((kern [] …))`.
- **Breaking. One meaning per result key** (DESIGN §9, "The 1.0's
  shape", the result keys, a glossary of every key; 1117ee3, genclj-4jpw).
  - `weights/normalize` returns `{:log-total :log-normalized-weights}`.
    Its `:log-weights` held the normalized logs, where importance
    sampling's holds the raw ones and calls the normalized ones
    `:log-normalized-weights`. Read `:log-normalized-weights` instead;
    `:log-weights` is now always the raw logs.
  - `mh/move` returns `{:trace :log-alpha}`, the key `mh/mh` gives the
    same ratio, as a host double. It was `:weight`, an array.
    `pf/reweighting`'s kernels still return `{:trace :weight}`.
  - `mh/mh-particles`, `mcmc/hmc-particles` and `mcmc/mala-particles`
    return the scalar kernels' keys in the plural, one value per particle,
    as gen.batched's `:weights` are generate's `:weight`: `:accepted` for
    `:accepted?`, `:log-alphas` for `:log-alpha` and `:diverged` for
    `:diverged?`. A vector under `:accepted?` was always truthy.
  - Unchanged: gen.batched's `{:particles :weights :discards}`, which
    mirror gfi's `{:trace :weight :discard}`.
- **Breaking. An invalid proposal is a loud error again** (the owner's
  second ruling on decision 5; genclj-ocxs). When a move's new trace is
  possible but the backward proposal gives the move's reverse choices
  density 0, the proposal cannot reverse its own move (the thesis's q3_mh,
  p. 92). MH with a proposal, involutive MH, the translators, trace kernels
  and SMCP3, and a particle filter's backward proposal, rejuvenation and
  move-reweight throw ex-info with `:proposal` and `:reverse-choices`
  (`gfi/assess-backward`), where decision 5 had made such a move a quiet
  rejection. So does a reverse choice of density 0 with nothing left
  unvisited, which was quiet before decision 5 too. A move to an impossible
  trace is still rejected quietly, and a move from one still throws MH's
  NaN error.
- **Breaking. A choice outside its own support stops the run** (decision 5;
  9277423, genclj-q7cq, genclj-2870). When a site's log density or a call's
  score is −Inf, generate, update, regenerate and assess return weight −Inf
  without running the rest of the body: the trace has score −Inf, the
  choices up to and including the impossible one, and retval nil; an
  update's discard restores the old trace. MH rejects such a move and
  map-optimize's line search shrinks its step, where both used to throw in a
  downstream parameter check. Code that reads a proposed trace before
  accepting it can meet missing choices. Unfold, Scan and Recurse stop at a
  stopped part. Batched, MLX and gradient runs cannot stop: they force the
  particle's score and weight to −Inf. A move from an impossible trace to
  another weighs NaN, which MH reports loudly. with-exact-proposal scores
  its own runs' observations as 0, and an impossible observation inside a
  call is a loud error there.
- **Draws. Residual resampling keeps its whole copies through rounding**
  (decision 8; 3fde123, genclj-tx1h): n·p_i rounds up within a relative 1e-9
  below the next integer, so equal weights give each index once, where they
  used to fall to the multinomial residual.
- **A batched fallback is recorded** (decision 9; 8344a1e, genclj-ynpc).
  Every batched entry point's result carries `:fallback`, a written
  message, when the run went one particle at a time; `:batched? :required`
  throws with that reason instead. map-particles and project keep returning
  arrays; map-particles* and project* return `{:result :fallback}`.
  `fallback-reason` still works. Nothing prints.
- **A splice argument equal to the last is unchanged** (decision 10;
  33fec66, genclj-nvo0): an identical value, or an equal host number,
  boolean, keyword or string (−0.0 apart from 0.0), gives no-change, so a
  callee is not rerun for an equal fresh number; collections and arrays
  still compare by identity.

- **Documented, behaviour unchanged.** Promotion follows JAX's rule where an
  integer meets a float (int64 + float32 is float32), as the code already
  did; docs/array-api.md §5 says so now (decision 6; ed51adf, genclj-7ndy).
  MLX arrays compare with `=` by dtype, shape and elements, reading both on
  the host, as JVM arrays do (b925210, genclj-aa3s).
- **Breaking. Reading a value being differentiated on the host is a loud
  error.** Under the tape, `array/item` and `array/->clj` of a tracked float
  throw ex-info (`:shape`, `:dtype`). Before, they returned the primal, and
  the path through it was missing from the gradient without a word. Host
  arithmetic on a tracked value (clojure.core's operators, `Math/…`,
  `double`) throws the same kind of ex-info, where it was a raw
  ClassCastException. Booleans and integers read freely (comparisons,
  `array/floor`), so `if` on a comparison works, but a read only to branch
  or index, `(int (array/item x))`, is refused too. The ways out:
  gen.array's operators, a custom gradient (gen.custom's `:gradient`,
  `array/custom-vjp`), `(array/stop-gradient x)` to drop the gradient on
  purpose, or `(array/floor x)` to index. MLX's opt-in native gradients
  still detach such a read silently. (c510d2a, genclj-de4g)
- **Draws. Unbatched `importance-resampling` keeps a different trace for
  the same key;** the distribution is unchanged. One sample at a time, each
  replaces the kept one with probability its share of the weight so far
  (Gen.jl's), so it holds one trace whatever n. When every weight is −Inf it
  throws (`:reason :impossible`). The batched path is unchanged. (6b0b2c7,
  genclj-1t38)
- **Draws. A plain call in a body draws from the body's ω.** `(gf & args)`
  inside a `gen` body ran gf with a key from the implicit source, so a run
  with a fixed key did not repeat. Now gf runs with the next key from the
  body's encapsulated randomness ω, which the body takes only when gf draws
  (makes choices, or keeps an ω of its own). Such a call is an untraced
  draw: the trace keeps ω, the score is an estimate, operations that need
  exact densities (gradients, HMC, MALA, MAP, VI, elliptical slice) refuse
  it loudly, and a batched run falls back to one particle at a time. A call
  that draws nothing changes nothing. Outside a body a plain call still
  takes the implicit key. (4646203, genclj-zxis)
- **A boolean choice reads as a host boolean outside a body.** On MLX a 0-d
  boolean is an array object, which Clojure's `if` takes as true, so
  `(if (tr :b) …)` in host code always took the true branch. Outside a
  body, a trace called with an address now returns a 0-d boolean as a host
  boolean (evaluating it on a lazy backend). Inside a body, and for batched
  values, values stay arrays. Other host code that branches on a boolean
  array uses `array/truthy?`. (4646203, genclj-x6ox)
- **Map's, Unfold's and Scan's scores are a finite sum beside two counts**
  of infinite element scores. After any chain of updates the score is the
  IEEE sum of the element scores, as assess computes it, +Inf and NaN
  included. Measured: a batched update revisiting one step of a 500-step
  Unfold is about 2.4× faster, a batched extension by one step costs 12%
  more, and on MLX a scalar generate costs 12–17% more. (ec8e777, kept by
  the owner in 2682118; genclj-ul72)
- **Spot checks on a lazy backend decide on the batch's own backend.** The
  JVM re-run (float64) agreeing is enough. A mismatch there is re-run on the
  batch's backend, which decides. So a float32 batch of an ill-conditioned
  model on MLX stays batched, where it fell back to one particle at a time,
  or, after unchecked updates, threw the divergence error on a correct
  model. (4704394, genclj-i8ui)
- **Breaking. gen.laws' laws return test.check results** (gen.laws is
  under `test/`, outside the core). A failing law's `checks` name the check
  and the values it compared, and a case the law says nothing about is
  `vacuous`. `deflaws` fails a law whose cases are more than 60% vacuous
  (`laws/max-vacuous`), so a law set that passed vacuously can fail now.
  Keys come from `laws/seeds` (2^31 seeds, where `gen/nat` drew 0–99), and
  the fuzz's random models take an argument with alternatives. (2cd752a,
  genclj-4tjd)
- **Draws. A Switch whose index differs across a batch's particles runs
  batched, by partition** (`gen.partition`): each branch runs over its own
  particles and the results are scattered back, where the whole batch fell
  back to one particle at a time. A particle draws the same in any
  partition. Loud: branch return values that cannot be one batched value
  (shape, dtype or structure differing, an int against a float), a
  partitioned trace revised by another generative function, a selection
  under choices held by part, and constraints by part whose particles span
  groups. `gen.batched/select` checks its mask's length.
  (dd63174, 249855b; genclj-7tqw)
- **HMC's and MALA's acceptance ratio comes from the trajectory's log
  densities** (assess at both ends) rather than an update's weight, which
  agrees up to rounding. The trace is updated only when a move is accepted.
  (f2a61fa, genclj-7dti)
- **MLX's CPU sums float32 in blocks of 32,** not 256, for sums of more
  than 256 terms. An iterated computation compounded the larger blocks'
  error: memo.clj's empowerment demo missed float64 by 1.89e-5, and by
  5.1e-6 now. A long sum takes 1.3–1.5× as long. Shorter sums, Metal and
  CUDA are unchanged. (73f22a0, 3022944; genclj-v3uw)

### Added

- **The laws for your own generative functions** (gen.laws in ext/laws,
  stable; DESIGN §9, "The laws for users"; genclj-u6je).
  - `(laws/verify models)`, `(laws/verify models opts)` and
    `(laws/verify models opts laws)`, deflaws' arguments but the prefix:
    every law of the set (`laws/all` by default) over cases of the model
    entries, returning a sorted map from each law that failed to what it
    says: `:check` and `:values` (or what the check's explain gives), or
    `:error`, `:data` and `:exception`, or `{:vacuous :of :max-vacuous}`;
    with `:case` `{:model :key-seed :picks :alt?}` (test.check's smallest,
    the model by its label) and the run's `:seed`. `{}` when every law
    holds. opts is the number of cases (100) or a map of it and `:seed`,
    `:max-size`, `:max-vacuous`. `laws/failure-of` reads a result of
    `laws/quick-check` the same way.
  - **Breaking.** `laws/check` is `laws/quick-check` (owner, 2026-10-25:
    one letter from `checks`, which does something else). It takes
    `(quick-check property)` and `(quick-check n property)`, and its
    options as keywords or a map (it required the map).
  - `laws/estimated`: `laws/encapsulated` without
    `:score-is-assessed-on-the-same-key`, for an estimated density with a
    joint sampler (gen.custom/estimated with `:simulate`).
  - `laws/conventions`: `identity-update-reports-no-change` (Changed,
    below), gen.clj's habits that a user's generative function need not
    keep.
  - A law set may gain a law in a minor release, and `laws/close?`'s
    tolerances may change, each with a line here and its reason, since a
    dependent's suite may then fail; renaming or removing a law is
    breaking (owner, 2026-10-25).
  - `laws/checks`, `laws/vacuous?`, `laws/vacuous-on-unvisitable` and
    `laws/unvisitable?` are public, for a law of your own. Vacuity is
    documented: a case is vacuous when its operation's error says a
    constraint was not visited (or is a nested choice map or a single value
    where the other belongs), so a hand-written generative function throws
    `gfi/unvisited`'s error.
  - Every law has a docstring, and gen.laws joins the stable snapshot and
    the API appendix.

- **Batched MH with a proposal, and a batched filter's accessor**
  (af25c97, genclj-xgz4).
  - `(mh/mh-particles ps proposal proposal-args key)`, as `mh/mh`'s
    arities (a selection or a proposal, the key optional): the proposal,
    the update and the backward proposal's assess run batched, as the
    model does, the scalar move's own code under the batch tracer.
    Chapter 025's filter of 1000 particles over 50 bearings, with a
    random-walk rejuvenation at each step, takes 0.6 s (0.2 s without
    it), where it took 6.6 s one particle at a time.
  - `gen.batched/move`: `(move ps f key & xs)`, every particle's trace
    moved by `(f trace key x …)` → `{:trace trace' …}` at once, as
    `map-particles` runs a function: `{:particles :result}`, and
    `:fallback` when f ran on each trace.
  - `kernel/batched-form`: a kernel's batched form, `(fn [ps key])` →
    `{:particles …}`, or nil. `(kernel/kernel f reverse-f {:particles p
    :reverse-particles rp})` and `(kernel/symmetric f p)` declare one.
  - `pf/choice`: `(choice state addr)`, every particle's choice at addr,
    an `[n …]` array, read from a batched state's collection without
    rebuilding its traces (as `gen.batched/choice`).
- **Traces print compactly** (4a13162, genclj-qvnu). A trace prints as
  `#gen/trace[#gen/fn[xs] {:args … :score … :choices {…}}]`: its maker,
  arguments, score and choices as a nested map, where it printed its
  whole record, every call's trace included (a 500-step Unfold's trace
  printed 500 subtraces). A batched trace prints `{:args …
  :batch-shape [n]}`. The form is for reading: gen.serialize refuses it
  loudly, and `write-trace` and `read-trace` remain the round trip.
- **`sel/union` and `sel/intersection`, and selections print as their
  trees** (DESIGN §9, "The 1.0's shape", before the tag; 13ad2b3,
  genclj-ufco).
  - `(sel/union s t …)` selects what any of the selections selects,
    `(sel/intersection s t …)` what each does; `(sel/union)` is `sel/none`
    and `(sel/intersection)` is `sel/all`, and a non-selection is a loud
    error. all and none absorb or drop out, and hierarchical selections
    join or meet address by address, so a union of selects is the select
    of their paths (`=` to it). Others combine as `Union` and
    `Intersection` records (`:parts`), which `root-selected?`,
    `top-addresses` and every generative function read. De Morgan's laws
    hold with `sel/complement`.
  - A selection prints as `#gen/selection {:x all, :y {0 all}}`, with
    `all`, `none`, `(complement …)`, `(union …)` and `(intersection …)`,
    where it printed as its record. The form is for reading, not for the
    reader.
  - A complement's subselection is simplified: under an address that its
    inner selection selects whole it is `sel/none`, where it was a
    complement of `sel/all` (the same set).
  - `gen.laws`'s `project-splits-the-score` also checks the projection on
    the union of two disjoint selections, and on the intersection of their
    complements, on every generative function it runs on.
- **Choice maps are callable, as traces are** (DESIGN §9, "The 1.0's
  shape", the eight more; 2797a1d, genclj-a2t8). `(c :slope)`,
  `(c [:y 0])` and `(c :b nil)` read the value there as `cm/get-choice`
  does, loud in the same places, with a not-found arity. At an address
  that holds nested choices the call is an error that points to
  `cm/get-submap`, not the submap, as Gen.jl's `cm[addr]` throws there.
  Like a trace's call, it reads a 0-d boolean as a host boolean outside a
  body (genclj-x6ox), so `((:choices tr) a)` is `(tr a)` for every choice
  on every backend; `cm/get-choice` still returns the value as stored. A
  trace's call at nested choices now ends its message with where to read
  them, and `ifn?` is true of a choice map.
- **The trace-transform API's missing pieces** (DESIGN §9, "The
  transform API"; 2e3f9d6, genclj-xsml).
  - `tf/run` takes options: `:retained`, a selection of the model
    choices the move keeps, whose continuous reads (neither written nor
    copied) count as unchanged outputs of the Jacobian, as an
    involution's update keeps them, so `tf/run` inspects a move that
    reads x to move y and keeps x, where it threw "maps 3 dimensions to
    2"; and `:aux-retval`, aux-in's return value for `tf/retval`.
  - The results of `tf/involution`'s involutions and of every translator
    hold `:log-abs-det` beside `:weight`, which still includes it. A move
    outside the support has weight -Inf and no `:log-abs-det`.
  - `mh/move` takes `:fwd-choices u`: the move's `:log-alpha` for given
    auxiliary choices, assessed under the proposal, in place of drawing
    them; a u the proposal gives density 0 is an error.
  - An involution from `tf/involution` is also called `(h trace u key)`
    or `(h trace u)`, without the two placeholders of `(h trace u nil []
    key)`.
  - `(tf/retval space)` in a transform's body: the return value of what
    made an input space's choices (a trace's for model-in; for aux-in
    the proposal's: an involution's fwd-retval, a translator's
    q-forward's, in a check the backward proposal's), read on the host
    and constant to the Jacobian, for a move's structure such as a path.
    An involution's check of a transform that reads it needs the
    proposal, `{:check? true :proposal q}`.
- **`mcmc/inv-metric`: a metric labeled** (f04ef86, genclj-qs62). `(mcmc/inv-metric
  trace selection m)` is the inverse metric the kernels read from the
  `:inv-metric` option m (any form they take; nil the identity), as a
  choice map of each selected choice's variances, the form `nuts/adapt`
  returns.
- **VI's fits return q's last draws and their optimizer state, and take
  a callback,** as Gen.jl's (56d81a3, genclj-59cf). `black-box-vi` returns
  `:traces`, q's draws in the last iteration (`:particles` with
  `:batched?`), and `black-box-vimco` one trace for each estimate,
  resampled from its K draws by their normalized weights. Both return
  `:state` (and `:model-state`), which the options of those names take
  back to resume a fit, and call `:callback` after each iteration with
  `{:iter :objective :traces :store}` (and `:model-store`).
  `elbo-gradient` and `vimco-gradient` take `:traces? true` for the draws
  (VIMCO's with their `:log-normalized-weights`). The draws for a key are
  unchanged. A negative `:iters` is an error.
- **An open optimizer interface, and the common optimizers** (DESIGN §9,
  "The 1.0's shape", the optimizer interface; ccd6b7e, genclj-y2ab).
  - An optimizer is a map with a `:kind`, which the multimethods
    `learn/-init` (its initial state for a store) and `learn/-step` (a
    pure `[store' state']`) dispatch on, so any namespace adds one with two
    methods, and `learn/step`, `learn/train` and VI's fits step it.
    `learn/update-given`, the walk of a gradient tree that the built-in
    steps use, is public for a user's `-step`.
  - `learn/momentum` (`:nesterov? true` for Nesterov's), `learn/rmsprop`,
    `learn/adagrad` and `learn/adamw`, as PyTorch's update rules and
    defaults, ascending; `learn/clip-norm` wraps another optimizer and
    scales each step's gradients to a global norm of at most `:max-norm`
    (optax's clip_by_global_norm). The book's chapter 045 adds momentum
    through the interface, where it stepped its own loop.
- **`array/vmap`: a function batched over a leading axis,** as JAX's
  vmap (DESIGN §9, "The 1.0's shape"; fbf0d21, genclj-t3ur).
  `((array/vmap f {:in-axes [0 nil]}) xs shared)` runs f once for the
  rows of `xs` under the batch tracer and returns its value (an array, a
  host number, or maps and vectors of them) with the leading batch axis,
  a shared leaf broadcast. An in-axis is 0 (the default) or nil; the batched arrays
  must agree on the batch size. Gradients inside are per particle. A
  gradient through a vmap, a nested vmap, a host read of a batched value
  and a batched comparison nothing read (a raw if) are loud, and nothing
  falls back to one row at a time. Draws are per particle, even from a
  shared key. It replaces gen.array.batch's batched, wrap and unwrap,
  which stay internal: `(batch/unwrap (batch/batched [n] (f (batch/wrap
  xs [n]))) [n])` is `((array/vmap f) xs)`. The book's chapter 041 uses
  it.
- **Kernels for mcmc's moves, and chains that keep host values**
  (a509ad5, genclj-4hu0).
  - `kernel/hmc`, `kernel/mala`, `kernel/nuts` and
    `kernel/elliptical-slice` make `gen.inference.mcmc`'s moves kernels,
    as `kernel/mh` does MH's: `(k/hmc selection opts)`, `(k/mala selection
    tau opts)`, `(k/nuts selection opts)` (one transition, at given `:eps`
    and `:inv-metric`, such as `nuts/adapt`'s), `(k/elliptical-slice addr
    mu cov)`. Each is its own reversal. `kernel/map-optimize` is a step of
    `mcmc/map-optimize` as a kernel, with no reversal. Options take no
    `:key` (loud): the kernel takes one at each application.
  - `kernel/run` and `nuts/sample` take `:keep`, a function of a trace or
    a map of them: the result holds `:draws`, `(keep t)` for each kept
    trace with its arrays read on the host (a map of vectors for a map),
    in place of `:traces`, and the chain holds its current trace alone.
- **The API's tiers, the stable surface's snapshot and the API appendix**
  (DESIGN §9, "The 1.0's shape"; e6f549f, genclj-dv8x).
  - Every namespace of `src/` and `ext/`, and gen.laws, carries `{:api
    :stable}`, `{:api :experimental}` or `{:api :internal}` in its ns form.
    Stable: gen.array, gen.array.linalg, gen.array.protocols, gen.random,
    gen.choicemap, gen.selection, gen.trace, gen.diff, gen.gfi,
    gen.dynamic, gen.serialize, gen.batched, gen.dist, gen.dist.core,
    gen.dist.dsl, gen.combinators, gen.custom, gen.state-space,
    gen.inference's importance, mh, kernel, mcmc, nuts, particle-filter,
    enumerative, proposal, transform, vi, diagnostics and weights,
    gen.gradients and gen.learning. gen.gfi and gen.array.protocols add an
    `:implementers` note: through 1.x their protocols keep their methods
    and gain none. Experimental: every gen.llm namespace, gen.inference's
    trace-kernel, conjugate, laplace and particle-mcmc, gen.partition,
    gen.dist.host, gen.dist.commons-math and gen.laws. Internal: gen.array's
    tape, batch, dtype, shape, special, host, tree and memory,
    gen.random.threefry, gen.dynamic's guard, static and core, and the
    backends. Of these, DESIGN's list did not name gen.array.linalg,
    gen.dist.core (defdist's), gen.inference.proposal and nuts (stable),
    gen.dist.commons-math (experimental), gen.array.special, host, tree
    and memory, gen.random.threefry and gen.backend.mlx.exact (internal).
  - `test/fixtures/api-snapshot.edn` holds the stable surface: every
    public var but `^:no-doc` ones, with its arglists or its kind (33
    namespaces and 743 vars with gen.laws; it
    grows with each deliberate addition). gen.api-test fails when
    the code differs. It
    names what was added, removed or changed, and calls out apart a var
    removed without a deprecation first. `clojure -M:test -m gen.api
    write` writes the snapshot after a deliberate change; `check` prints
    the differences.
  - Deprecation is `:deprecated` metadata (the version) and a docstring
    note naming the replacement, nothing printed. The snapshot records it.
  - The book's appendix C, *The Public API* (`book/063-api.md`), is written
    from the tiers and docstrings: the stable namespaces by layer with
    their vars, the experimental ones by name. The book's test fails when
    it is stale.
- **Public replacements for the internals memo.clj uses,** in gen.array
  (DESIGN §9, "The 1.0's shape"; b509bce, genclj-9efy). memo.clj can move
  off the internals, which stay as they are, marked internal.
  `tracked?` replaces gen.array.tape/tracked?. `(broadcast-shapes & shapes)`,
  NumPy's broadcast_shapes, replaces gen.array.shape/broadcast; shapes that
  do not broadcast are a loud error naming them all. `dtype-kind` gives a
  dtype keyword's :bool, :int or :float (of an array, `(dtype-kind (dtype
  x))`) and replaces gen.array.dtype/kind; anything else is a loud error. A
  batched value's particles replace gen.array.batch's batched?, value and
  inner-backend: `batched?` (true of a value being differentiated whose
  primal is batched), `batch-shape`, `particle-backend` (the backend the
  batch wraps), `(particle x i)` and `(particles x idx)`. These gather
  unbatched arrays without reading on the host, and refuse a value being
  differentiated with a batched primal, whose particles would carry no
  gradient. gen.array.protocols adds `ITrackedValue`, which the tape's
  values implement.
- **This changelog,** kept from 2026-10-25 (DESIGN §9, "The 1.0's shape";
  genclj-6g66).
- **VI's pathwise estimator, ADVI's.** `black-box-vi` and `elbo-gradient`
  take `:estimator :reparam` beside the score-function estimator (the
  default, Gen.jl's). On the conjugate test model its variance is under a
  quarter of the score-function estimator's. A family declares a pathwise
  sampler with `^:pathwise` on `defdist` (`gen.dist.core/pathwise?`):
  normal, uniform, exponential, laplace, cauchy, gumbel, log-normal and
  logistic. Loud: a choice of q's from another family, a latent q does not
  propose, and `:batched?`. (a0f3358, genclj-uiqm)
- **Importance sampling's raw weights, ESS and estimate.**
  `importance-sampling`'s result adds `:log-weights` (raw) and `:ess`, and
  its weights are host doubles on every backend. `is/estimate` gives the
  self-normalized estimate of E[(f trace)] with `:log-ml-estimate` and
  `:ess`, keeping no traces, so n can be large on any backend. (6b0b2c7,
  genclj-1t38)
- **`gen.trace/omega-at`,** the path to a trace's encapsulated randomness
  (private in gen.serialize before). (4646203, genclj-zxis)
- **Conjugacy and state-space models, chosen by the user.**
  `gen.inference.conjugate/with-conjugate-proposal` makes one latent's
  exact posterior, given its observed children, a model's internal proposal
  (normal–normal, mvnormal–mvnormal, beta–bernoulli, beta–binomial,
  gamma–poisson, dirichlet–categorical). `gen.state-space/kalman` and `hmm`
  are generative functions with exact marginal likelihoods, and
  `sample-states` draws their states by forward filtering, backward
  sampling. (aaa7a75, genclj-702q)
- **Particle MCMC.** `gen.inference.particle-mcmc/conditional-smc`, a
  particle filter conditioned on a trace's reference trajectory, and
  `particle-gibbs`, its move as a kernel. PMMH is MH on a
  `gen.custom/estimated` density whose estimate runs a particle filter.
  (d2e5c1a, genclj-gcus)
- **Laplace approximations and Fisher information.** `gen.inference.laplace`:
  `hessian` and `param-hessian` (central differences of the gradient),
  `choice-mode` and `param-mode` (Newton's method), `laplace` and its
  `proposal`, `observed-fisher`, `natural-gradient` and `bic`. (17ffbae,
  genclj-xg1j)
- **Scan, Mix and dimap.** `comb/scan` (JAX's scan, built like Unfold),
  `comb/mix` and `mix-logits` (a categorical at `:index` and a Switch at
  `:component`), and `comb/dimap`, `contramap` and `map-retval`. (19a7f0f,
  genclj-2gma)
- **Gradients under the batch tracer** are vmap's: each particle gets the
  value and gradient of its own scalar run. On them: `gen.batched`'s
  `map-particles` and `set-choices`, `choice-gradients` of a particle
  collection, `mcmc/hmc-particles`, `mala-particles` and
  `map-optimize-particles` (n chains, each accepted on its own), and
  black-box VI's `:batched?`. (7399000, e24c1d8; genclj-7dti)
- **Per-particle constraints and batched enumeration.** `gen.batched`'s
  `generate` and `update` take `{:per-particle cs}`, and `gen.batched/project`
  gives the particles' projections. `enumerative-inference` takes
  `:batched?` and `:chunk-size` (a batched result holds `:batches` in place
  of `:traces`), and `with-exact-proposal` takes `:batched?`. (24323fd,
  genclj-7oc1)
- **NUTS and Stan's warmup.** `mcmc/nuts` is Stan's base_nuts transition,
  and `gen.inference.nuts` (`adapt`, `sample`) is Stan's windowed warmup
  with a diagonal metric, for NUTS or for HMC with L fixed. `hmc`, `mala`
  and `nuts` take `:inv-metric`, and `hmc` and `nuts` take Stan's
  `:jitter`. The defaults (the identity metric, no jitter) leave hmc's and
  mala's moves bit-identical. (18c7b09, 3466f2a; genclj-tvc0)
- **GenMLX's further distributions:** `truncated-normal`, `von-mises`,
  `wrapped-cauchy`, `wrapped-normal`, `wishart` and `inv-wishart`, exact at
  their extremes. Circular values lie in [−π, π). (808ac58, genclj-z2gh)
- **Serialization as EDN.** `gen.serialize` (`write`, `read`,
  `write-trace`, `read-trace`, `read-trace-data`) writes choice maps, arrays
  and traces with gen.clj's tags only, never eval. A trace is rebuilt by
  generate and checked exact. It records the backend and float dtype it was
  written under, and `read-trace` refuses another (`:other-numerics`).
  Traces with encapsulated randomness are refused for now.
  `array/backend-name` names a backend. (9566fdd, genclj-unjc)
- **MCMC diagnostics.** `gen.inference.diagnostics`: bulk and tail ESS, the
  ESS of the mean, rank-normalized and split R-hat, the MCSE of the mean and sd,
  and `summary`, as Vehtari et al. (2021) and ArviZ define them. (82e8241,
  genclj-hfca)

### Fixed

- **Map's, Unfold's and Scan's running totals no longer drift** (genclj-ul72's
  totals; found by the walk laws). The finite part of a total is a
  compensated sum (Knuth's two-sum), so a long chain of updates that take large
  scores in and out of a small total keeps the score at assess's sum to
  round-off; a plain running total drifted 2.9e-11 from -2.82 after 1000
  updates. Scores differ from before in the last bits.
- **write-trace accepts generate's weight up to rounding** (genclj-s5fj).
  It required the weight to equal the trace's score bit for bit, and so
  refused a hand-written generative function, or a running total, that sums
  the same terms in another order, blaming missing choices. It now allows
  relative 1e-9 (float64) or 1e-4 (float32), and the message says "beyond
  rounding".

- **Map's, Unfold's and Scan's scores never turn NaN** (−Inf + Inf) after a
  choice leaves its support and comes back. A general law now holds every
  kind's updated score to the assessed density through such chains. The
  remedy is now genclj-ul72's, above. (a3e7276, genclj-awd0)
- **gen.diff's `+`, `-`, `inc` and `dec` rules** give an int-diff only when
  every operand is an integer. Through a float operand the result rounds,
  and the int-diff was an unsound retdiff; it is unknown-change now.
  (fc428f7, genclj-y0ij)

### Decided, not yet in the code

Owner decisions in DESIGN §9 that dependents will see. Each moves to the
sections above in the commit that builds it. None is waiting now:

- **From the book's friction** (2026-10-25; 797a269). Decisions 1–4 and 6
  concern the book and the docs; 5, 8, 9 and 10 are built (Changed, above);
  7 changes nothing in the code (dynamic bodies keep passing unknown-change
  to splices; gen.dynamic's docstring names the static language as the way
  to incremental splices; 33fec66).
- **The 1.0's shape** (2026-10-25; 872f04f; genclj-04h2). Built, each with
  its entry above: the tier marks, the snapshot and the API appendix; the
  public replacements for internals (gen.array's `tracked?`,
  `broadcast-shapes`, `dtype-kind` and a batched value's `particle` and
  `particles`); one result-key vocabulary; kernels returning `{:trace
  :accepted?}`; labeled metrics; matching batched return shapes;
  selections' union and intersection; callable choice maps; trace keyword
  lookups (kept and documented); public handler state; an optimizer
  interface and training callbacks; laws for users (ext/laws); the
  particle filter's accessors; and the transform API.

## History before 1.0.0, by milestone

gen.clj was built in milestones before this file began. These summaries
are for orientation. DESIGN.md (§1 and §9) and the milestone beans have the
detail.

### The book (2026-10-24 to 2026-10-25; genclj-ceds)

The GenMLX book ported, with Gen.jl's tutorials as its worked chapters, in
`book/`: five Parts and appendices. A chapter's code lives in its own
namespace (`book/src/book/…`), and the Markdown includes its regions, so
every listing is code that runs. A result shown after `;=>` is checked, and
a statistical claim passes at 5 standard errors (genclj-fblt).
`gen.book sync` writes the includes and the SVG figures, `gen.book pdf`
builds a PDF with Pandoc, and `gen.book-test` with `gen.book.part2-test` to
`part5-test` runs the chapters on the JVM backend (on MLX with
`-Dgen.book.mlx=true`). All of it is merged and approved by the owner:
Parts I (011–017), II (021–029), III (031–035), IV (041–045) and V
(051–054), and the appendices Coming from Gen.jl, The Distributions
(generated from gen.dist), The Public API (generated from the tiers),
Notation, Installing and Running, and the Bibliography. Writing it found genclj-zxis, x6ox, 1t38 and
de4g, and the ten decisions (1.0.0, above).

### The post-M5 review (2026-10-22; genclj-ta2i)

Eight fresh-context reviewers read every line of `src/`, four for idiom and
four for PPL soundness, and eight verifiers confirmed 84 of the 86 findings
(docs/research/review-post-m5/). Its fix batches, 36 beans, are on main at
bd6b8b4. What a dependent sees:

- **Draws.** `fold-in` hashes all 64 bits of its data: keys are unchanged
  below 2^32 and distinct beyond, so Recurse's deep nodes no longer share
  keys. Negative or fractional data is a loud error.
- **Draws.** Every beta and dirichlet draw changes for a given key (drawn
  from their gammas' logs). A gamma, beta or dirichlet draw that underflows
  to 0 is floored, and a beta draw that rounds to 1 is clamped below 1.
- "Unchanged" (retdiffs, `diff/compute`, `diff/consistent?`) means
  `gfi/same-value?`, so −0.0 differs from 0.0 and a vector from a list. A
  change in the number of arguments is a change.
- A diverging HMC or MALA trajectory is a rejected proposal, counted
  (`:diverged?`, `:divergences`), where it threw. So is a parameter check
  failing at a tried position. A start whose log density or gradient is not
  finite is a loud error.
- **Breaking.** Loud now: a mixture of a discrete and a continuous family,
  a boolean given as a numeric parameter, ±Inf given as a count, token SMC
  given a constrained completion, a product of potentials whose end tokens
  disagree, and `constrained/conditional-weight` of anything but a
  completion's trace.
- VI's model gradient adds the score-function term for latents the model
  samples itself, without which it was biased.
- Numerics: lgamma and digamma at their edges are SciPy's, a log mass of 0
  has gradient 0 (not NaN), probabilities must sum to 1 within a tolerance
  set by their dtype and number, and MLX's CPU sums float32 in blocks and
  half floats in float32.

### M5 — Gen.jl's own test suite passes (2026-10-20 to 2026-10-22; genclj-xxuc)

Gen.jl 0.4.8's test suite, ported file by file to `test/gen/genjl/` (37
namespaces, each test citing its source line), passes on the JVM backend
and on MLX. Of 1,699 assertion sites, 1,418 port and 239 are not
applicable, each with its reason (docs/research/genjl-test-suite.md).
Gen.jl itself writes the oracle's EDN fixtures (`test/fixtures/genjl/`), so
the tests never need Julia. The gaps the port found were built:

- **Breaking. Lookups are loud.** `(tr a)`, `cm/get-choice`, `cm/get-value`
  and `trace/subtrace` throw ex-info (`{:reason :path :at}`) where nothing
  is, as Gen.jl throws KeyError. Pass a not-found value, `(tr :b nil)`, or
  ask `cm/has-value?` first.
- **Breaking.** A vector is a path everywhere, selections included
  (`sel/selected?`, `sel/get-subselection`, `sel/set-subselection`,
  `cm/set-submap`), and `array/full` with an array fill is a loud error.
- `trace/subtrace` and `(tr addr)` give a call's trace and return value;
  `cm/merge` takes 0 to n choice maps; Recurse has `get-child`,
  `get-child-num` and `get-parent`.
- The static language takes computed addresses, nested calls (hoisted) and
  macros in its bindings.
- `param-gradients` reaches through combinators' traces (`gfi/stores`,
  `gfi/with-stores`), a custom deterministic function names its `:params`,
  and VI trains the model's parameters too (`:model-store`,
  `:model-optimizer`).
- `learn/step` evaluates what it returns, so a hand-written training loop
  no longer grows a graph on a lazy backend.

The documented differences from Gen.jl (more precise retdiffs, Eq. 4.1's
regenerate weight under a Switch, no default positional arguments, and
others) are in DESIGN §9 and the mapping.

### M4 — Gen.jl parity (2026-10-19 to 2026-10-21; genclj-wcga)

Everything gen.clj targets in Gen.jl 0.4.8's core and its core-adjacent
ecosystem (GenParticleFilters, GenSMCP3, GenTraceKernelDSL), each law the
thesis states for them a test (docs/research/genjl-parity.md §1, §3):

- the static modeling language, `(gen ^:static …)`, with compiled scores
  and a lint, and change hints (`int-diff`, `set-diff`, `dict-diff`,
  `diff/propagate`);
- call-at, choice-at and Recurse; encapsulated randomness (`untraced`,
  `gen.custom/estimated`), with assess taking a key; deterministic
  functions with custom gradients (`array/custom-vjp`);
- `gen.dist.dsl/dist` (Gen.jl's `@dist`), host-library families
  (`gen.dist.host`, with Commons Math in `ext/commons-math`) and
  Distributions.jl's further continuous families;
- exact enumeration (`gen.inference.enumerative`, built early for GenGDT),
  the internal-proposal override, `kern` with `^:trace`, trace transforms
  and translators, SMCP3, trace kernels, GenParticleFilters' operations,
  black-box VI and VIMCO, and wake-sleep.

The owner's decisions then: booleans count in arithmetic as in Julia, casts
to integers are Java's, a draw beyond the float range is +Inf, and a
covariance singular in floating point is an error at every precision.

### M3 — Local LLMs as generative functions (2026-10-20 to 2026-10-21; genclj-vjfd)

A local LLM is a generative function. The stack: the Branching protocol
(`gen.llm.branching`) with llama.cpp v0.7.0 as its engine (`gen.llm.llama`);
the LM protocol (`gen.llm.lm`) with its cache, prefix trie and KV-cell
ledger; the token generative function (`gen.llm.token/completion`), every
GFI operation over token sites; potentials (`gen.llm.potential`) and token
SMC with fork-on-resample (`gen.llm.smc`); grammar-constrained decoding on
gen.clj's own regex and DFA engine (`gen.llm.grammar`,
`gen.llm.constrained`); and a memory budget that MLX and native engines
share (`gen.array.memory`). Engines never sample: tokens are drawn by
gen.random with each site's key. Verified on Metal (five models) and CUDA
(an RTX PRO 6000, seven models), with bands by device.

### M2 — The MLX backend (2026-10-19 to 2026-10-20; genclj-rtif)

The same code on MLX, through mlx-c v0.7.0 (MLX v0.32.2) and Java's FFM
API, with a binding generated from the headers. It is chosen by
`-Dgen.backend=mlx`, `-Dgen.mlx.device=cpu|gpu`, `-Dgen.mlx.dtypes=32|64`
or `array/with-backend`. It passes the backend conformance suite against
the float64 JVM backend, and the full battery on Metal, on MLX's CPU and on
CUDA (a Jetson AGX Thor, MLX built from source). A key draws the same
uniforms on both backends. Where MLX's semantics differ from NumPy's the
backend follows NumPy's, except that Metal flushes float32 subnormals and
array indices are clamped. Memory is bounded by default, and native
gradients are opt-in (`-Dgen.mlx.grad=native`).

### M1 — The core on the JVM (2026-10-18 to 2026-10-19; genclj-r2vy)

gen.clj's core on a pure-JVM backend that needs nothing but a JDK: the
backend-neutral array API (`gen.array`) with tape autodiff and JAX's
Threefry PRNG; the data layer (choice maps, selections, traces, diffs); the
Gen.jl-shaped GFI with the `gen` macro and DynamicGF, every operation taking
an explicit key; Gen.jl's distributions, mixtures and products; Map, Unfold
and Switch with incremental updates; batched execution of the same code
under a batch tracer, with spot checks and a fallback; and the inference
library (importance sampling, MH in its forms, kernel combinators, the
particle filter, HMC, MALA, MAP, elliptical slice, gradients, and learning
with an explicit parameter store). It set the conventions kept since:
0-based indices, Gen.jl's names and parameterizations, booleans as
bernoulli's values, constraints as choice maps (a plain map is an error),
and bodies that run to completion. The GFI laws are test.check properties.
