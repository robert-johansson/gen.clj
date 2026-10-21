# The MLX-native engine: a design (genclj-gvfa)

*2026-10-21, branch m3-mlx, by an agent. This note covers design and
pure-JVM investigation only; the checks that need MLX or the GPU are
listed in §7 and wait for a slot. The choices that are the owner's to make
are marked **Recommendation**. The engine is DESIGN §6's engine 2: Qwen3's
and Qwen3.5's forward passes ported from GenMLX over `gen.array`, for
gradients, numeric parity with GenMLX and logits that stay on the device.*

## Summary

- **Weights.** No download is needed. MLX's own GGUF loader (`mlx_load_gguf`,
  already in our binding) reads the three models in `~/gen-clj-models`
  unchanged: each holds only F32 and Q8_0 tensors. MLX turns Q8_0 into its
  affine 8-bit quantization (groups of 32), exact once its float16 scales
  are cast to float32 (measured, §7). What a GGUF holds follows
  llama.cpp's conventions, not Hugging Face's, so the forward follows
  llama.cpp's graphs for the tensors' meaning (§1).
- **The forward over gen.array.** Every layer of both models can be written
  with gen.array's current operations, as composites. A prototype runs
  tiny Qwen3- and Qwen3.5-shaped models on the pure JVM:
  - a cached prefill plus steps gives the full forward's logits exactly;
  - a batch of contexts gives each context's own logits exactly;
  - the tape's gradients match central differences to 6e-9.

  What is missing is speed and quantization: native RMSNorm, RoPE and
  attention, quantized matmul, and a fused GatedDeltaNet scan (§2).
- **The engine.** **Recommendation:** an implementation of gen.llm.slots's
  `Slots`, whose slots are entries in a map of immutable cache values, so
  that a fork is a shared value and the slots machinery, the Branching
  contract and the LM, token and SMC layers are reused as they are (§3).
- **Tests.**
  - Self-consistency on tiny models.
  - Logits against llama.cpp on the same GGUF, within a measured band: the
    only check that catches a wrong convention (§5).
  - The Branching contract.
  - Gradients against finite differences.
  - Exact parity with GenMLX's golden values needs GenMLX's own checkpoints,
    a download for the owner to decide.

## 1. Weights without a download

**MLX reads GGUF.** mlx-c v0.7.0 has `mlx_load_gguf` and the `mlx_io_gguf_*`
accessors (keys, arrays, metadata strings and arrays), and our binding
(`native/bindgen.clj`, every mlx-c header) has them all. In MLX 0.32.2
(`mlx/io/gguf.cpp`, `gguf_quants.cpp`):
- F32, F16, I8, I16 and I32 tensors are copied as they are.
- Q4_0, Q4_1 and Q8_0 become MLX's affine quantization: `name.weight`
  (uint32, packed), `name.scales` and `name.biases` (float16, one of each
  per group of 32).
  - For Q8_0 the int8 values are shifted to uint8 (`x ^ 0x80`) and the bias
    is −128·scale, so scale·u + bias = scale·q in exact arithmetic. MLX
    computes in the scales' dtype, though: with the float16 scales as
    loaded, the dequantized weights are d·q rounded to float16 (§7).
    Cast to float32 first, they are the file's d·q exactly.
  - These feed `mlx_quantized_matmul` (bits 8, group size 32, `transpose`
    true) and `mlx_dequantize` directly.
- Any other type (the K-quants among them) is dequantized to float16 by
  gguflib, at load. BF16 support there is unverified.
- MLX copies every tensor into its own buffers: the file's size stays
  resident, with no mmap.

**Our files** (read on the pure JVM by `llm-mlx-engine/gguf_header.clj`):

| Model | Architecture | Tensors | Notes |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | `qwen3` | 113 F32, 197 Q8_0 | 28 blocks; 16 query and 8 KV heads of 128; tied output (no `output.weight`) |
| Qwen3.5-0.8B Q8_0 | `qwen35` | 140 F32, 195 Q8_0 | 24 blocks plus an MTP block (`blk.24`, unused); GatedDeltaNet with 16 heads of 128; every fourth layer full attention, 8 query and 2 KV heads of 256, RoPE on 64 dims; tied output |
| Gemma-3-270M Q8_0 | `gemma3` | 109 F32, 127 Q8_0 | not a target of this engine |

Every Q8_0 tensor's innermost dimension is a multiple of 32, so all of them
load. GGML lists dimensions innermost first and MLX reverses them, so a
matmul weight is [out, in] as in Hugging Face's layout.

**A GGUF is llama.cpp's model, not Hugging Face's.** llama.cpp's converter
(`conversion/qwen.py` at 620d222) changes tensors on the way:
- RMSNorm weights carry Qwen3-Next's +1 already (except the gated norm's);
- `ssm_a` is −exp(A_log);
- `dt_bias` becomes `ssm_dt.bias`;
- the conv kernel is squeezed to [channels, 4];
- Qwen3.5's in_proj_qkv, z, b and a become `attn_qkv`, `attn_gate`,
  `ssm_beta` and `ssm_alpha`;
- V heads are reordered from grouped to tiled, so a GQA repeat must tile
  rather than interleave. This is the identity at 0.8B (16 key and 16
  value heads), not at 9B (16 and 32).

So the tensors' meaning comes from llama.cpp's graphs (`src/models/qwen3.cpp`,
`qwen35.cpp`, `delta-net-base.cpp`), and the structure from GenMLX's
forwards, which read Hugging Face's checkpoints. A convention got wrong is
self-consistent. In the prototype, KV heads tiled (query head h reading
KV head h mod KV) where llama.cpp groups them (h / (H/KV)) pass every
self-consistency check, so only logits against
llama.cpp catch such errors (§5).

**The tokenizer.** **Recommendation:** llama.cpp's vocabulary from the same
GGUF, loaded with `vocab_only` (a field our binding's `llama_model_params`
already has):
- it needs no new dependency;
- the tokens are the llama engine's, which the comparisons in §5 need;
- `llama/tokenizer` serves the LM protocol unchanged.

DESIGN §6's `ai.djl.huggingface:tokenizers` would then be needed only for
safetensors checkpoints.

**GenMLX's checkpoints.** GenMLX reads Hugging Face's safetensors (bf16, or
MLX-quantized), and its golden values (`forward_golden_test`, the parity
tests) were taken from the bf16 weights. Parity with them needs those
checkpoints downloaded, which is the owner's decision. The engine's loader
can take both formats; safetensors is `mlx_load_safetensors`, also bound.

## 2. The forward pass over gen.array

**What GenMLX does** (`src/genmlx/llm/`). The owner graded it C as code: a
source of ideas rather than code to port.
- `forward.cljs`: one six-function interface per family (load-model,
  forward, next-token-logits, prefill, step, init-cache), with per-layer
  caches as immutable values, so forks are free.
  - It adds a batch axis of K lanes in lockstep: `broadcast-cache` tiles a
    B=1 cache, and resampling is one `take` along the lanes.
- `qwen3_forward.cljs`:
  - embeddings by `take`;
  - per layer: RMSNorm, the q/k/v projections, per-head RMSNorm of q and k,
    RoPE at an offset, the KV cache grown by `concatenate`, fused attention,
    and a SwiGLU MLP;
  - a tied output.
- `qwen35_forward.cljs`: GatedDeltaNet layers, and every fourth layer full
  attention with partial RoPE and a sigmoid output gate.
  - The GatedDeltaNet layer: a causal depthwise conv over the q/k/v
    projection, built as K shifted slices; normalized q and k; the gates
    beta = sigmoid(b) and g = exp(A·softplus(a + dt_bias)); and the gated
    delta rule in float32 (decay the state, read it with k, write the
    correction).
  - Speed came from fused kernels in GenMLX's patched mlx-node
    (`gated-delta-scan`, `gated-delta-step`, `fused-gdn-gating`). A
    host-loop version is kept as the reference.
  - An opt-in static-capacity KV cache (`put-along-axis` into blocks of
    256) gave no speed on CUDA.

**What gen.array has, and lacks.** The prototype
(`llm-mlx-engine/tiny_forward.clj`) writes every layer of both families with
today's operations:

| Operation | As a composite today | MLX's native | Gradient |
|---|---|---|---|
| RMSNorm | `x·w/√(mean(x²)+ε)` | `mlx_fast_rms_norm` | the tape; MLX's VJP |
| RoPE, NEOX-style, partial, positions per row | cos/sin of positions × inverse frequencies, halves rotated | `mlx_fast_rope` (one offset), `mlx_fast_rope_dynamic` (an offset for each batch row) | the tape; MLX's (not for the offset) |
| attention with GQA and a mask | matmul, `where`, `softmax`, matmul | `mlx_fast_scaled_dot_product_attention` | the tape; MLX's |
| SiLU, l2-norm, softplus | composites (softplus exists) | — | the tape |
| causal depthwise conv1d | K shifted slices | `mlx_conv1d` (grouped) | the tape |
| GatedDeltaNet, one token | the delta rule's broadcasts and sums | — | the tape |
| GatedDeltaNet, a prefill | a token loop. The chunked form llama.cpp uses (chunks of 64, a triangular solve per chunk) needs only `la/solve-triangular`, `cumsum` and matmuls | a kernel through `mlx_fast_metal_kernel` / `mlx_fast_cuda_kernel` (bound): mlx-lm's own route, with no patch to MLX | the tape for the composite |
| quantized matmul, dequantize | **missing** | `mlx_quantized_matmul`, `mlx_dequantize`, `mlx_gather_qmm` | MLX: with respect to x (weights frozen) |
| float16 and bfloat16 activations | the dtypes exist (`:float16 :bfloat16`) | yes | — |

array-api.md §6 already lists "sdpa, rope, rms-norm, quantized matmul" as
M3's. **Recommendation:**
- Add `rms-norm`, `rope`, `sdpa`, `quantized-matmul` and `dequantize` to
  gen.array, names for the owner:
  - the first three as composites, which MLX replaces with its fast ops
    through `-native-ops` (§6 of array-api.md), so the tape differentiates
    them on every backend and the JVM gives the reference;
  - `quantized-matmul` and `dequantize` as primitives (the JVM:
    dequantize, then matmul).
  - Only L0 knows a backend (CLAUDE.md), so they cannot live in gen.llm.
- Run the GatedDeltaNet as a composite first, and later a native kernel per
  backend behind it, as `-native-ops` do now.
- Use the JVM backend for tiny test models only. Its `double[]` storage
  would hold a 0.6B model's weights in about 4.8 GB.

**The prototype's results** (pure JVM, float64, random weights; Qwen3-like:
4 attention layers; Qwen3.5-like: 3 GatedDeltaNet layers and one gated
attention layer with RoPE on half of each head; hidden 32, vocabulary 97):

| Check | Qwen3-like | Qwen3.5-like |
|---|---|---|
| prefill of 4, a chunk of 3, two steps, against the full forward of 9 | \|Δ logit\| 0 | 0 |
| a batch of two contexts against one alone | 0 | 0 |
| the tape's gradients (one weight of layer 0, the embedding) against central differences | relative 6.3e-9 | 3.4e-9 |
| one cache value continued twice | identical; another token's differ by 2.1 | identical; 3.5 |

The checks can fail:
- the chunk's positions shifted by one fail the first check by 1.1 and
  0.052 logits;
- the conv state dropped from the cache fails it on the Qwen3.5-like model
  by 4.1, and so does the conv state keeping its oldest inputs instead of
  its newest.

KV heads tiled instead of grouped pass all four checks, as do the conv
kernel applied backwards and q's query and gate swapped. A wrong convention
is self-consistent, which is why §5 compares with llama.cpp.

## 3. The engine under the Branching protocol

**Slots, not Branching directly.** **Recommendation:** the engine implements
gen.llm.slots's `Slots`, as llama.cpp's does, with each slot an entry in a
map from slot to cache value:
- `slots-copy` shares the value (a fork is free);
- `slots-clear` drops the entry;
- `slots-run` is a prefill, or a teacher-forced extension with its scores;
- `slots-step` is one batched decode.

It inherits the engine thread, lifetimes (phantom references), capacity
errors, the scratch slot for `score`, the Branching contract and everything
above it (the LM's cache, the token GF, token SMC) with no change. The slot
count is the engine's option, not llama.cpp's 255, and memory is the only
bound. MLX's arrays count against the shared budget (genclj-zztc) by
construction.

**Caches.** A layer's entry is an immutable value: {:k :v} [B KV T D] for
attention, {:conv :state} for GatedDeltaNet (a fixed size: 3×6,144 and
16×128×128 floats at 0.8B).
- **(a) Concatenate each step (GenMLX's default).** Simple, and immutable by
  construction. A step copies the branch's context: Qwen3-0.6B holds 112
  KiB a token in float16 (llama.cpp's 112 MiB at 1,024 cells), so a
  1,024-token branch copies 112 MiB a step, about what attention reads
  anyway.
- **(b) Static capacity with `slice_update`.**
  - MLX's Metal and CUDA `copy_gpu` reuse an input buffer that nothing else
    references (`is_donatable`), so an append writes in place unless a fork
    still shares the buffer. That is copy-on-write for free.
  - It needs the engine to release a consumed state's handles at once (an
    advance consumes its inputs), before the next evaluation. Today the
    backend frees handles only through phantom references, so this needs
    an explicit release in `gen.backend.mlx` (a new name).
  - Capacity grows by blocks, which also keeps shapes stable for
    `array/compile`.
- **(c) Paged blocks and shared-prefix attention.** Full blocks are shared
  and immutable, and only a tail block is copied; attention runs per block
  and merges with log-sum-exp. It is the most work, and only worth it if
  (b)'s measurements ask for it.

**Recommendation:** (a) first, for correctness. Then (b) with an explicit
release, measured on Metal; (c) only if long contexts need it.

**Batching across branches.** A step's branches can sit at different
positions:
- the engine stacks their caches (padded to a common length) with a mask
  per row and RoPE positions per row (`mlx_fast_rope_dynamic` takes one
  offset for each batch row). That is a copy of each stepped branch's cache
  per step;
- *lane groups* avoid it for token SMC, whose particles move in lockstep. A
  group is a batched cache whose rows are branches; a step of exactly its
  rows reuses it, and a resample is one `take` along the rows, as in
  GenMLX's lanes.

**Recommendation:** lane groups inside the Slots implementation, recognized
from the step's slots, with stacking for every other step. With (b), a
group's step writes in place only while no older entry shares its buffers.
The LM's cache keeps parents as entries, so its steps copy once per step,
as (a) does.

**Determinism and logits.**
- MLX's results depend on batch shapes as llama.cpp's do. The LM's
  score-once ledger already makes that harmless, and plans (lane groups,
  stacking) are a function of the engine's own operations.
- `slots-step` returns host rows, normalized on the device and copied (600
  KB at Qwen3's vocabulary), so the protocols are unchanged.
- **Recommendation:** logits that stay on the device come from the forward
  functions themselves, an engine-level API over gen.array arrays for
  training and for potentials on the device. Host draws stay as DESIGN §9
  decided.

## 4. Gradients

The forward is gen.array code, so `array/grad` differentiates it:
- the tape by default, on both backends: the prototype's gradients match
  central differences to 6e-9;
- or MLX's own (`-Dgen.mlx.grad=native`), whose fast ops have VJPs with
  respect to their inputs.

The quantized weights stay frozen: MLX differentiates `quantized_matmul`
with respect to x only. Fine-tuning adds dense adapters (LoRA) beside them,
for genclj-q4xv. **Recommendation:** train in native mode for speed; the
tape on tiny models is the reference that tests it.

## 5. The test plan

1. **Tiny models on the JVM (float64), as tests:** the prototype's four
   checks for both families, each shown to fail under planted bugs (a
   shifted position, the conv state dropped, a cache overwritten).
2. **MLX against the JVM on the same tiny models:** logits within a
   float32 band measured from their spread; the native ops against their
   composites.
3. **Real models against llama.cpp on the same GGUF.** For Qwen3-0.6B and
   Qwen3.5-0.8B, the next-token log-probabilities over the top 64 tokens
   should agree within a band measured as `llama_noise` measures llama.cpp's
   own, with the same argmax and top 5. This is the check for conventions:
   planted errors (KV heads tiled instead of grouped, the norm's +1 added twice, the
   `ssm_a` sign) should fail it by nats. Q8_0 matmuls differ between the
   two engines (on the CPU llama.cpp quantizes the activations as well), so
   the band is measured per platform.
4. **The Branching contract on the MLX engine,** with its own band, and the
   LM, token GF and token SMC suites over it.
5. **Gradients:** finite differences on tiny JVM models; MLX native against
   the tape; a LoRA step on a tiny model lowers its loss.
6. **GenMLX:** statistics only (CLAUDE.md), since the weights differ, until
   GenMLX's checkpoints are here. Then its golden pins: argmax and top 5
   exact on the bf16 weights, logprobs within its bands.

## 6. Phases, as proposed beans (not created)

genclj-gvfa would become a feature under genclj-54u1, with one task per
phase. Each phase is blocked by the one before it unless its entry says
otherwise. The owner's decisions in the last section come first: P1's
names, and whether to download GenMLX's checkpoints (P2's parity item).

**P1. gen.array's LLM operations** (task, llm, array; needs the pure JVM,
then MLX)
- `rms-norm`, `rope` (NEOX and traditional, partial dims, positions per
  row) and `sdpa` (GQA, a boolean or additive mask), as composites, with
  MLX's fast ops as `-native-ops`.
- `quantized-matmul` and `dequantize` as primitives: the JVM dequantizes,
  MLX calls `mlx_quantized_matmul` and `mlx_dequantize`. Their VJP is with
  respect to x only.
- Conformance tests (MLX against the JVM), the tape's VJP checks, and
  mutations planted.
- array-api.md §6's table updated.
- Done when: the conformance suite passes on the JVM, MLX's CPU and Metal.

**P2. GGUF weights and Qwen3's forward** (task, llm, mlx; needs an MLX slot)
- `gen.llm.mlx.gguf`: `mlx_load_gguf`, the metadata read into a config,
  the tensors by llama.cpp's names, and Q8_0 as quantized triples.
- `gen.llm.mlx.qwen3`: the forward from the prototype, over P1's ops, with
  embeddings by `take` and dequantize and a tied output by
  `quantized-matmul`.
- The tokenizer: llama.cpp's vocabulary, loaded `vocab_only`.
- Tests:
  - tiny-model self-consistency on the JVM (the prototype's four checks);
  - Qwen3-0.6B's log-probabilities against the llama engine within a band
    measured as `llama_noise` measures, the band and its spread recorded;
  - planted convention errors fail by nats.
- Optional, if the owner downloads GenMLX's checkpoint: GenMLX's golden
  pins.

**P3. The engine: Slots over cache values** (task, llm, mlx; needs MLX)
- `gen.llm.mlx` (or `gen.llm.mlx.engine`): a `Slots` implementation.
  - Copy shares a value and clear drops it.
  - A run is a prefill or a teacher-forced extension with its scores.
  - A step is one decode: caches concatenated, the step's caches stacked
    with masks and RoPE positions per row, and lane groups for steps of
    exactly a group's rows.
- The Branching contract with its own band; gen.llm.lm, token and SMC
  suites over it.
- The shared memory budget's tests with the engine open.

**P4. Qwen3.5: GatedDeltaNet** (task, llm, mlx; blocked by P2, not P3)
- `gen.llm.mlx.qwen35`, with llama.cpp's conventions (§1):
  - the GatedDeltaNet as a composite: the token loop, then the chunked
    form of chunks of 64;
  - partial RoPE and the gated attention.
- A Metal kernel for the step and the chunked scan through
  `mlx_fast_metal_kernel`, behind the composite as a native op, tested
  against it.
- Qwen3.5-0.8B against the llama engine within its band.

**P5. Gradients and LoRA** (task, llm, mlx; blocked by P3; unblocks
genclj-q4xv)
- Dense adapters beside the frozen quantized weights.
- `array/value-and-grad` through the forward, in the tape and in MLX's
  native mode: finite differences on tiny JVM models, and native against
  the tape on MLX.
- One LoRA step on a tiny model lowers its loss.

**P6. Speed** (task, llm, mlx, performance; blocked by P3)
- Static-capacity caches with `slice_update`, after an explicit release of
  consumed handles in gen.backend.mlx (a new name). The donation measured.
- `array/compile` of a step at a stable shape.
- CUDA kernels on the Thor (genclj-mxi9's machine).
- A bench against the llama engine on the LM's rows (`gen.lm-bench`).

## 7. Checked on MLX (2026-10-21)

`llm-mlx-engine/mlx_checks.clj` made these checks in MLX JVMs on the Mac
mini M4 (Metal), one at a time, with llama.cpp for the last two. Each run exited normally
under the memory guard (peak 2.6 GB). To repeat them:
`clojure -Sdeps '{:paths ["src" "test" "target/classes"]}' -J-Dgen.mlx.lib=… -J-Dgen.llama.lib=… -M docs/research/llm-mlx-engine/mlx_checks.clj ~/gen-clj-models`,
with check numbers after it to run some of them, or `4-heads`, `4-rope`,
`5-conv-shift`, `5-conv-kernel` and `5-q-gate` for the planted errors.

1. **Loading.** `mlx_load_gguf` reads both models, every array as the
   header says:
   - Qwen3-0.6B: 310 tensors, 704 arrays;
   - Qwen3.5-0.8B: 335 tensors, 725 arrays.

   F32 tensors come out as float32 with their dimensions reversed. Each
   Q8_0 tensor comes out as a uint32 weight [out, in/4] with float16 scales
   and biases [out, in/32].
2. **Q8_0's values** (`blk.0.attn_q`, rows 0 to 2, against the file's blocks
   read on the host):

   | Scales | `dequantize` | `quantized_matmul` against `dequantize` then `matmul` (max \|y\| 7.28) |
   |---|---|---|
   | float16, as loaded | d·q rounded to float16, exactly (not d·q) | max \|Δ\| 1.7e-3 |
   | cast to float32 | d·q exactly | max \|Δ\| 6.4e-6 |

   **Recommendation:** cast the scales and biases to float32 at load, so
   that MLX, the JVM's reference and the file agree exactly. It costs 40
   bytes per 32 weights against 36, about 11% more weight memory. Float16
   scales remain an option for speed (P6).
3. **Donation.** A `slice_update` of an unshared 4 MiB array wrote in place
   (the same buffer address), its handle freed before evaluation. With the
   input still held, it copied. Copy-on-write therefore works through our
   binding, given an explicit release.
4. **Qwen3-0.6B against the llama engine**, the same GGUF, two prompts of 8
   tokens. The prototype's composites ran on MLX, with today's gen.array
   operations and the weights dequantized to float32. Argmax and the top 5
   (as sets) equal llama.cpp's, with the max |Δ log p| over llama.cpp's top
   64:

   | Scales | "The capital of Sweden is a city called" | "def fibonacci(n):\n    if n <" |
   |---|---|---|
   | float16 | 0.0039 | 0.0080 |
   | float32 | 0.0060 | 0.0052 |

   That is inside llama.cpp's own spread between batch shapes (up to 5e-2,
   median 2e-3 to 4e-3, `llm-branching.md`). Planted convention errors fail
   it:
   - KV heads tiled where Qwen3 groups them: 24.3 and 21.7 nats, argmax
     wrong;
   - RoPE rotating the other way: 3.5 and 8.4 nats.

   Swapping RoPE's two halves in both q and k is no error at all: the dot
   products, and so the logits, do not change.

5. **Qwen3.5-0.8B against the llama engine** (the same prompts, the
   scales cast to float32). The prototype's 24 layers ran from the GGUF's
   tensors by llama.cpp's names: GatedDeltaNet, and at every fourth layer
   gated attention with RoPE on 64 of each head's 256 dims. Each prompt was
   computed whole, and as a cached prefill of all but its last token plus
   one step, which carries the KV cache, the conv state and the recurrent
   state.

   | Prompt | whole | cached |
   |---|---|---|
   | "The capital of Sweden is a city called" (8 tokens) | 0.00078 | 0.00078 |
   | "def fibonacci(n):\n    if n <" (9 tokens) | 0.0019 | 0.0019 |

   Argmax and the top 5 equal llama.cpp's in all four cases. Planted
   errors specific to Qwen3.5 fail it:

   | Planted error | whole | cached |
   |---|---|---|
   | the conv state keeping its oldest inputs, not its newest | unchanged (0.00078, 0.0019) | 1.7 and 6.9 nats, top 5 wrong |
   | the conv kernel applied newest input first | 17.0 and 8.9 nats, argmax wrong | the same |
   | q's query and output gate swapped | 25.2 and 14.8 nats, argmax wrong | the same |

   The first shows only in a cached step, which is why the check makes
   one. The last two are self-consistent: they pass every check on the
   tiny models (§2), so only this comparison catches them.

So P2's central risk, whether the GGUF's weights and llama.cpp's
conventions give llama.cpp's distribution through gen.array, is retired for
both families, Qwen3 and Qwen3.5. That covers the hybrid layers and their
caches, at 0.8B, where key and value heads are equal (§1's V-head order is
the identity there).

## Recommendations for the owner

1. Weights from the GGUF files already here, through MLX's loader, with the
   Q8_0 scales and biases cast to float32 at load (§7); GenMLX's
   safetensors only if exact GenMLX parity is wanted (a download).
2. The tokenizer: llama.cpp's vocabulary, loaded `vocab_only`.
3. gen.array gains `rms-norm`, `rope`, `sdpa` (composites with MLX natives),
   `quantized-matmul` and `dequantize` (primitives). Names to confirm.
4. The engine is a `Slots` implementation.
5. Caches concatenate first; static capacity with an explicit release in
   gen.backend.mlx (a new name) after measurement; paged blocks only if
   needed.
6. Lane groups for lockstep particles; logits on the device through the
   forward functions, the protocols unchanged.
7. Gradients: MLX native mode for training, the tape as reference;
   quantized weights frozen, LoRA beside them.
