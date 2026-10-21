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
  affine 8-bit quantization (groups of 32, exact). What a GGUF holds follows
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
    is −128·scale, so scale·u + bias = scale·q exactly.
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
self-consistent. In the prototype, KV heads interleaved where they should
be tiled pass every self-consistency check, so only logits against
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
4 attention layers, Qwen3.5-like: 3 GatedDeltaNet layers and one attention
layer; hidden 32, vocabulary 97):

| Check | Qwen3-like | Qwen3.5-like |
|---|---|---|
| prefill of 4, a chunk of 3, two steps, against the full forward of 9 | \|Δ logit\| 0 | 0 |
| a batch of two contexts against one alone | 0 | 0 |
| the tape's gradients (one weight of layer 0, the embedding) against central differences | relative 6.3e-9 | 1.5e-9 |
| one cache value continued twice | identical; another token's differ by 2.1 | identical; 3.6 |

The checks can fail: the chunk's positions shifted by one fail the first
check by 1.1 and 0.26 logits, and the conv state dropped from the cache fails
it on the Qwen3.5-like model by 4.2. KV heads interleaved instead of tiled
pass all four, which is why §5 compares with llama.cpp.

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
   planted errors (KV heads interleaved, the norm's +1 added twice, the
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

## 6. Phases (bean suggestions)

| Phase | Work | Needs |
|---|---|---|
| P1 | gen.array's LLM ops: composites, MLX natives, VJPs, conformance tests | pure JVM, then MLX |
| P2 | GGUF loading and Qwen3's forward; Qwen3-0.6B against llama.cpp | an MLX slot |
| P3 | The Slots engine: caches (a), stacking, lane groups; the contract and the LM layers on it | MLX |
| P4 | Qwen3.5: GatedDeltaNet as a composite (token loop, then chunked), then a Metal kernel; Qwen3.5-0.8B against llama.cpp | MLX, Metal |
| P5 | Gradients and LoRA (genclj-q4xv) | MLX |
| P6 | Speed: caches (b), `array/compile` of a step, CUDA kernels on the Thor | Metal, CUDA |

## 7. To check when MLX is free

Each is a few minutes in one MLX JVM:
- `mlx_load_gguf` on both files: names, shapes and dtypes;
- `quantized_matmul` against `dequantize` then `matmul` on a real tensor;
- donation: a `slice_update` of an unshared array leaves its buffer in
  place, through our binding, once its handle is released;
- the first P2 measurement: the logits for one prompt against the llama
  engine's.

## Recommendations for the owner

1. Weights from the GGUF files already here, through MLX's loader; GenMLX's
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
