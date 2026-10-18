# Local-LLM backends for gen.clj on the JVM

*Research notes, 2026-10-18. Produced by a research agent with web access; the
`llama.h` symbols `llama_memory_seq_cp`, `llama_get_logits_ith`, `kv_unified`
and `n_seq_max` were spot-checked against master. Unverified items are flagged
at the end. Summarized in DESIGN.md §6.*

## Bottom line
- **Protocol:** a small backend protocol shaped like genlm's `next_token_logprobs`
  (content-addressed), over a branch/handle layer that mirrors GenMLX's existing
  `branch-cache!` / `forward-branch-batched` / `resample-lanes!`.
- **Implementation #1: llama.cpp through our own filtered jextract FFM bindings.**
  It runs on Metal and CUDA today, covers every model family we use, and has
  exactly the primitives we need.
- **Implementation #2: MLX-native over mlx-c**, porting GenMLX's own
  Qwen3/3.5/3.6 forward passes. This gives GenMLX parity, MLX checkpoints and
  autodiff for training.
- **TornadoVM/jitllm** is impressive but not usable for SMC today.

## 1. Comparison

| Option | Metal | CUDA | CPU | Full logits | KV fork / multi-seq | Batching | Qwen3/3.5/3.6/VL, Gemma | Quant | Clojure fit | Maintenance | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **llama.cpp C API + own jextract FFM** | ✅ | ✅ (official ubuntu-cuda-12.8/13.4 x64+arm64 archives) | ✅ | ✅ `llama_get_logits_ith` at any or all positions | ✅ `llama_memory_seq_cp/rm/keep/add`, up to 256 seqs (`LLAMA_MAX_SEQ`); fork is metadata-only with `kv_unified` | ✅ one `llama_decode` over many seq ids | ✅ `qwen3`, `qwen3moe`, `qwen3next`, `qwen35`, `qwen35moe`, `qwen3vl(moe)`, `gemma3/3n/4`, plus libmtmd vision | GGUF (K-quants, IQ…) | Good: static Java calls, `MemorySegment` float views | Very active: b11719 today; semver v0.7.0 on 2026-10-15 | **Build first** |
| gravitee-io/llamaj.cpp | ✅ prebuilt | custom build | ✅ | ✅ (raw `llama_h`; `getLogprobs` reads the full row) | ✅ `LlamaMemory.seqCp/seqRm/copyPrefix` | ✅ `LlamaBatch`, `BatchIterator` | as llama.cpp v0.6.0 | GGUF | Good, but Java 25-only | 2.9.1 on 2026-10-07; 16★; no longer on Maven Central | Reference; vendor its build recipe |
| kherud/java-llama.cpp (JNI) | build yourself | build yourself | prebuilt | ❌ (server-style API) | ❌ | ❌ | stops at mid-2025 llama.cpp (README badge: b4916); no Qwen3.5 or Gemma 4 | GGUF | High-level only | v4.2.0 on 2025-06-20; stale | Avoid |
| **MLX-native over mlx-c (own)** | ✅ | ✅ (MLX CUDA backend; build mlx-c from source) | ✅ | ✅ on device | ✅ immutable arrays, so fork = shared reference (GenMLX already does this) | yours (GenMLX has batched lanes) | what we port: GenMLX owns `qwen3`, `qwen3_5`, `qwen3_5_moe` (Qwen3.6-A3B) and Qwen3.5 vision | MLX 4/8-bit safetensors (same numerics as GenMLX) | Good; we own memory and threading | mlx-c v0.7.0 on 2026-09-29 (pins MLX v0.32.2); MLX v0.32.4 | **Build second** (parity, training) |
| Alipsa/jmlx (Java 25 + mlx-c) | ✅ | ❌ | ❌ | ✅ | KV cache + batch scheduler | ✅ | Llama, Qwen2, Qwen3, Mistral, Gemma1, Phi-3, Mixtral; *not* qwen3_moe or gemma3 | MLX | Java | created 2026-08; 0★ | Prior art |
| beehive-lab/jitllm (formerly GPULlama3.java; TornadoVM) | ✅ (TornadoVM Metal backend) | ✅ (PTX/NVRTC + cuBLAS/cuDNN) | ✅ Vector API | internal only; `DecisionSession.score` returns logits for candidate tokens | ❌ no public fork; paged KV is internal | batched prefill; batched decode "landing" | Llama3, Mistral, Qwen2.5/3/3.5/3.6, Phi-3, Granite, Gemma 4, GLM-4.7-Flash | GGUF F16/Q8_0/Q4_0/Q4_K | Needs the TornadoVM SDK + argfile | v1.0.2 on 2026-09-27; 290★ | Watch; not for SMC |
| mukel qwen35.java / gemma4.java / llama3.java | ❌ | ❌ | ✅ Vector API | ✅ (it's your code) | ❌ single sequence | ❌ | Qwen3.5 0.8B–35B-A3B; all Gemma 4; Llama 3.x | GGUF F16/BF16/Q4_0..Q8_0 | Single file, easy to read or vendor | pushed 2026-04/05 | Reference / CPU fallback |
| tjake/Jlama | ❌ (WebGPU experimental) | ❌ | ✅ | internal | ❌ | limited | no Qwen3 | Q4/Q8 | Java 21 preview | v0.8.4 on 2025-01-01 | Avoid |
| ONNX Runtime GenAI (Java) | ❌ no Metal path found | ✅ | ✅ | `getOutput("logits")` | ❌ only `rewindTo` | static batch | needs ONNX export | int4 | JNI; docs say "Package publication is pending" | v0.17.0 on 2026-09-28 | Avoid |
| DJL | – | – | – | llama engine dead (`ai.djl.llama` 0.28.0, 2024-05) | – | – | – | – | **tokenizers 0.38.0 is good** | DJL v0.38.0 | Tokenizers only |
| vLLM / SGLang over HTTP | ❌ | ✅ | limited | top-k only (vLLM `max_logprobs` default 20, −1 = all; SGLang `token_ids_logprob`) | ❌ (automatic prefix cache only) | server-side | broad | many | HTTP JSON | vLLM v0.32.0 today; SGLang v0.5.21 | Remote / approximate only |
| Ollama / LM Studio / llama-server over HTTP | ✅ | ✅ | ✅ | top-k only (Ollama specifics unverified) | ❌ | server-side | broad | GGUF / MLX | HTTP | active | Approximate only |
| TensorRT-LLM | ❌ | ✅ | ❌ | ✅ (generation logits) | Python/C++ only | ✅ | broad | FP8/INT4 | Poor | active | Avoid |
| Candle / mistral.rs via FFM | ✅ | ✅ | ✅ | in Rust | needs a Rust cdylib C-ABI shim | – | Candle: no qwen3.5 or qwen3next; mistral.rs has Gemma 4 | GGUF / ISQ | Extra native project | active | Not worth it |

## 2. Layered design

**Layer 0: engine bindings.**
- llama.cpp: FFM over `llama.h` (later `mtmd.h`), generated with jextract
  (25-jextract+2-4, JDK 25-based) using `--include-function` / `--include-struct`
  filters. We need about 40 symbols.
- MLX: jextract over mlx-c, which we need anyway if the gen.clj tensor core is
  on MLX.
- **Thread confinement per engine.** A `llama_context` is not thread-safe, and
  MLX streams are thread-bound (jmlx docs; MLX issues #3727 and #2133). In
  Clojure, use a single engine-loop thread fed by a queue that returns promises.

**Layer 1: a `Branching` protocol, one implementation per engine.** This is
GenMLX's current API: `branch-cache!`, `branch-from`,
`forward-branch(-tokens|-batched)`, `forward-branch-scores`, `resample-lanes!`,
`dispose-branch!`. The llama.cpp mapping:
- **open**: allocate a seq id, then `llama_decode` with `logits` set on the last token.
- **fork**: `llama_memory_seq_cp(mem, src, dst, -1, -1)`. With `kv_unified=true`
  this is a cell-metadata update, so prefix cells are shared copy-on-write. For
  recurrent (GatedDeltaNet) layers, the destination shares the state cell until
  it is next written.
- **advance batch**: one `llama_batch` with one token per branch (its own seq id
  and position, `logits=1`), a single `llama_decode`, then
  `llama_get_logits_ith(i)` for each row.
- **dispose**: `llama_memory_seq_rm(mem, s, -1, -1)`.
- **teacher-forced score**: set `logits=1` on every continuation token and read
  every row.
- **resample**: `seq_cp` the parents onto the children, then `seq_rm` the dead.

The MLX mapping is exactly what GenMLX does today: a fork shares the persistent
cache value, and resampling is a lane gather.

**Layer 2: a generic, content-addressed `LM` protocol** (genlm-shaped), written
once over any `Branching` implementation:
```clojure
(defprotocol LM
  (vocab [lm])                 ; {:n .. :eos #{..} :byte-vocab [^bytes ..] :special #{..}}
  (encode [lm s opts]) (decode [lm ids])
  (next-logprobs [lm ctx])     ; int vector -> float[] (full-vocab log-softmax)
  (batch-next-logprobs [lm ctxs])
  (score [lm ctx cont]))       ; per-token teacher-forced logprobs
```
- **Machinery:** backed by a token-prefix trie (live branch handles plus LRU
  eviction) and an autobatcher (gather concurrent requests for a few ms, then
  make one batched call). This copies genlm-backend.
- **genlm-backend's interface:** `AsyncLM.next_token_logprobs(token_ids)`,
  `batch_next_token_logprobs`, sync variants, `byte_vocab` / `str_vocab` via
  `decode_vocab`, LoRA add/remove/set, and `clear_cache`.
- **genlm-backend's per-backend caching:**
  - HF: a `TokenTrie` with `past_key_values` on each node.
  - MLX: a `_SlotPool`. It finds the live row with the longest matching prefix
    and forks by repeating an index through `cache.filter`; otherwise it does a
    left-padded prefill.
  - vLLM: in-process with a custom `_CaptureSampler` and
    `enable_prefix_caching=True` (not over HTTP).
- **Grammar plug-in point:** genlm-control's `Potential` (`prefix`, `complete`,
  `logw_next`, `batch_logw_next`, `sparse_logw_next`) is the right shape for
  plugging our grammar engine into SMC.
- llamppl (formerly hfppl, now `genlm/llamppl` v0.2.3) depends on genlm-backend
  with the vllm and mlx extras. Whether early LLaMPPL used llama.cpp is
  unconfirmed.

**Layer 3: GenMLX's generative-function layer.** Token GF, `assess`, SMC, and
grammar masks over `byte-vocab`.

**Why llama.cpp first:**
- It works on the M4 with Metal and on Linux CUDA with official prebuilt binaries.
- It covers what GenMLX currently borrows from mlx-node's native classes
  (Gemma 4, Qwen3-Next 80B, VL).
- It exposes exactly the primitives above.
- It validates the protocol against an independent engine (cross-backend logprob
  conformance tests within a tolerance).

**Why MLX-native second:**
- It uses the same kernels and checkpoints as GenMLX, so tests and golden values
  carry over.
- Autodiff (`mlx_value_and_grad`, `mlx_vjp` and closures, which work via FFM
  upcalls) enables LoRA, SFT and GRPO.
- Logits stay resident on the device.
- If the gen.clj MLX core is already working when LLM work starts, flip the
  order for Qwen-family work.

### MLX-native vs llama.cpp-via-FFM

| | MLX-native (port of GenMLX) | llama.cpp via FFM |
|---|---|---|
| **(a) Fidelity / control** | Best. Same numerics as GenMLX. KV and recurrent state are values you hold: no sequence cap, cheap arbitrary rewind and checkpoint, masking can stay on device, and gradients are available. | High, but the state is opaque. Logits are copied to a host `float*`. Fork is excellent (metadata-only). Limits: 256 seqs, and n_ctx cells shared across all seqs. For hybrid models, partial `seq_rm` rollback works only within `n_rs_seq` snapshots (EXPERIMENTAL, single-use; `llama-memory-recurrent.cpp`), so keep checkpoint seq slots. Gemma SWA layers need `swa_full=true` for arbitrary prefix reuse. GGUF quantization differs from MLX 4-bit. Training is basic `llama_opt_*` only. |
| **(b) Model coverage** | Only what you port. mlx-lm (v0.32.2) has Python references for `qwen3_5(_moe)`, `qwen3_next`, `qwen3_vl(_moe)`, `gemma3/3n/4`. | About 150 architectures, with new ones arriving within days (`qwen4exp`, the GLM5 family, …). |
| **(c) Maintenance burden** | You own every forward pass, quantization layout, custom kernel, tokenizer and chat template. mlx-c itself is small and fairly stable. | The API churns: kv_self was renamed to `llama_memory_*` in 2025, `llama_set_adapters_lora` is now plural, and v0.6.0 added `llama_batch_ext` / `llama_process`. New context-param fields come tagged `[EXPERIMENTAL]`. Mitigations: semver releases since mid-2026 (v0.3.0 on 2026-08-25, v0.7.0 on 2026-10-15) with a versioned `libllama.0.6.0.dylib`; filtered jextract regeneration with a CI compile check; always start from `llama_context_default_params()`. |
| **(d) Metal and CUDA parity** | Metal is first-class. CUDA is newer: `mlx[cuda12/13]` needs SM ≥ 7.5 and glibc ≥ 2.35, and the C build is from source (`MLX_BUILD_CUDA=ON`). Custom Metal kernels need CUDA counterparts; mlx-lm's gated-delta kernel is Metal-only and falls back to ops (`gated_delta.py`). CUDA performance parity is unverified. | Mature on both. ggml's `GATED_DELTA_NET` op has Metal, CUDA and CPU implementations, and the backend is a dynamically loaded `.so`/`.dylib`, so the bindings don't change across platforms. |

## 3. The two links the owner asked about

**https://github.com/ggml-org/llama.cpp**
- **Project:** MIT license, ~131.6k★, pushed today (b11719). Semver releases:
  v0.6.0 (2026-10-05, = b11429) and v0.7.0 (2026-10-15).
- **What `llama.h` provides (verified against current master):**
  - Decoding: `llama_decode` / `llama_batch`, with a seq-id list and logits flag
    per token; `llama_get_logits_ith` (negative indices count from the end).
  - KV memory: `llama_get_memory` and
    `llama_memory_{clear,seq_rm,seq_cp,seq_keep,seq_add,seq_div,seq_pos_min/max,can_shift}`;
    context params `n_seq_max`, `kv_unified`, `n_rs_seq`.
  - Per-sequence state: `llama_state_seq_get/set_data` serializes one sequence
    (useful for checkpoints and for moving a sequence across contexts).
  - Vocabulary: `llama_vocab_n_tokens`; `llama_token_to_piece` returns raw
    bytes, so the byte vocab comes for free; `llama_vocab_get_text/attr`;
    `llama_tokenize/detokenize`.
  - Grammar: `llama_sampler_init_grammar` (GBNF). llguidance lives only in
    `common/`.
  - Adapters and training: LoRA and aLoRA, `llama_opt_*`, experimental backend
    sampling.
  - Chat templates: `llama_chat_apply_template` does **not** parse Jinja; it
    supports a fixed list of templates.
- **Vision:** libmtmd (`tools/mtmd/mtmd.h`) has projectors for Qwen2/2.5/3-VL and
  Gemma 3/3n/4.
- **Verdict:** this is the right engine for coverage. Bind it ourselves.

**https://github.com/gravitee-io/llamaj.cpp**
- **Project:** Apache-2.0, 238 commits, pushed 2026-10-09, release 2.9.1 on
  2026-10-07. The README says "no longer published to Maven Central"; the last
  artifact there is 2.8.1 (2026-09-17).
- **Bindings:** jextract over `llama.h`, `mtmd.h`, `mtmd-helper.h` and
  `ggml-rpc.h`, as a per-platform `llama_h` class. It pins llama.cpp v0.6.0,
  about 2 weeks behind.
- **API level:** low enough for our needs:
  - `LlamaBatch.add(token, pos, seqIds, logits)`.
  - `LlamaMemory.seqRm/seqCp/seqKeep/seqAdd/seqDiv/copyPrefix`. `copyPrefix`
    refuses to run without `kv_unified`, because otherwise "a partial range
    aborts the process".
  - `getLogprobs` reads the full `llama_get_logits_ith` row.
  - LoRA, mtmd, speculative decoding (MTP / Eagle3 drafts), `BatchIterator`.
- **Caveats:**
  - It is chat/serving oriented; grammar isn't in its docs index.
  - Bus factor: 16★.
  - The pom compiles with `--enable-preview` at release 25, which (inferred)
    pins the classes to JDK 25. That's fine on GraalVM 25 but needs a rebuild
    for 26+.
  - Its Linux build appears to download the CPU-only `ubuntu-x64` archive. CUDA
    needs a custom build, or possibly dropping in upstream's ubuntu-cuda ggml
    libs (unverified).
- **Verdict:** use it as a reference implementation and borrow its jextract and
  post-processing recipe. Depend at most on its raw `llama_h`, not its iterators.

## GraalVM 25 / TornadoVM specifics
- **FFM** is final since JDK 22, so every FFM option works on Oracle GraalVM 25.
  From Clojure, call jextract's static methods with type hints, or use coffi
  (`org.suskalo/coffi` 1.0.615, JDK 22+). Keep vocab-sized loops (log-softmax
  over 150k–250k entries) in Java or primitive-hinted code.
- **Vector API** is still incubating in JDK 25 (JEP 508, tenth incubator):
  `--add-modules jdk.incubator.vector`.
  - mukel's engines run with `--enable-preview --add-modules jdk.incubator.vector`
    and claim GraalVM Native Image and Graal tuning.
  - jitllm needs `@$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector`.
  - Jlama needs Java 21 preview features, so it likely won't run on 25
    (unverified).
  - How well the Graal JIT intrinsifies the Vector API on 25 is unverified.
- **TornadoVM 7.3.1** (2026-10-15):
  - Supports JDK 21–27 (a `jdk22plus` SDK line).
  - Backends: CUDA PTX via NVRTC (with cuBLAS/cuDNN library tasks), OpenCL, and
    a new native Metal (MSL) backend for M1–M4. It even has `tornado-mlx`
    library tasks.
  - The FAQ says it plugs into "OpenJDK or GraalVM". Coexistence with Oracle
    GraalVM 25's own Graal JIT was not verified.
  - Kernels must be TornadoVM-compilable Java bytecode. Clojure-generated
    bytecode is unlikely to qualify (assessment), so kernels would be written in
    Java.
- **jitllm** (renamed from GPULlama3.java):
  - **Requirements:** TornadoVM ≥ 7.3; Java 21 or 22+; the `jitllm4j` launcher
    needs Java 25.
  - **Platforms:** the Apple Silicon GPU via Metal, and CUDA. MLX-assisted
    prefill requires an M4-class GPU and `brew install mlx` 0.32.3.
  - **Models:** includes Qwen3 and Qwen3.5/3.6 (`qwen35`, `qwen35moe`); GGUF
    Q8_0, Q4_0, Q4_K.
  - **For SMC:** logits and KV are internal (a `Logits` interface and a paged
    `KvLease`); the public API is session-based. The nearest thing to scoring is
    `DecisionSession.score(prefix, branches, candidates)`, which runs branches
    sequentially and overwrites the KV, so there is no fork.
  - **Batched decode:** PRs #129 (batched decode) and #134 (on-device sampling)
    appear merged in Jul/Aug 2026, but the README still says "landing next"; the
    release status is unclear.
  - **Verdict for SMC:** this would mean forking the engine. Watch; don't build
    on it now.

## Tokenizers
- **llama.cpp path:** use the GGUF's embedded tokenizer; `llama_token_to_piece`
  gives the byte vocab.
- **MLX path:** `ai.djl.huggingface:tokenizers` 0.38.0 (2026-09-09) wraps the
  Rust HF tokenizers over JNI and bundles natives for linux-x86_64,
  linux-aarch64 and osx-aarch64. Build the byte vocab by inverting the GPT-2
  byte↔unicode map (port genlm's `decode_vocab` or GenMLX's `bytes.cljs`).
- **Ruled out:** jtokkit is tiktoken-only (it can't load Qwen or Gemma), and
  jmlx's pure-Java HF tokenizer is immature.
- **Chat templates:** still need a Java Jinja engine, e.g. Jinjava (Qwen3.5
  template compatibility unverified), or the GGUF template string via
  `llama_model_chat_template`.

## Unverified / inferred
- jitllm's batched-decode release status.
- TornadoVM coexisting with Oracle GraalVM 25's Graal JIT.
- Jlama on JDK 25.
- llamaj.cpp needing `--enable-preview` at runtime, and its Linux build being CPU-only.
- Dropping CUDA ggml libs into llamaj.cpp.
- MLX CUDA performance.
- **Loading llama.cpp's Metal backend and MLX in the same process.**
- Qwen3.6 using the `qwen35` architecture (per the jitllm README).
- A Qwen3.5-vision projector existing in mtmd.
- ORT GenAI Metal support and Maven artifacts.
- Ollama's exact logprobs limits.

## Sources
- **llama.cpp:**
  - https://github.com/ggml-org/llama.cpp
  - https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h
  - https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0
  - https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0
  - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp
  - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-recurrent.cpp
  - https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
- **JVM bindings and engines:**
  - https://github.com/gravitee-io/llamaj.cpp (`LlamaMemory.java`, `LlamaContext.java`, `pom.xml`, `docs/custom-builds`)
  - https://github.com/kherud/java-llama.cpp
  - https://github.com/tjake/Jlama
  - https://github.com/mukel/qwen35.java · https://github.com/mukel/gemma4.java · https://github.com/mukel/llama3.java
  - https://github.com/Alipsa/jmlx
  - https://github.com/beehive-lab/jitllm · https://github.com/beehive-lab/TornadoVM/releases/tag/v7.3.1
- **MLX:**
  - https://github.com/ml-explore/mlx-c
  - https://ml-explore.github.io/mlx/build/html/install.html
  - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
  - MLX threading issues: https://github.com/ml-explore/mlx/issues/3727 · https://github.com/ml-explore/mlx/issues/2133
- **genlm:**
  - https://github.com/genlm/genlm-backend (`llm/base.py`, `llm/mlx.py`, `llm/hf.py`, `llm/vllm.py`)
  - https://github.com/genlm/genlm-control
  - https://github.com/genlm/llamppl
- **Other runtimes and tokenizers:**
  - https://onnxruntime.ai/docs/genai/api/java.html
  - https://repo1.maven.org/maven2/ai/djl/huggingface/tokenizers/
  - https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py
  - https://github.com/EricLBuehler/mistral.rs · https://github.com/huggingface/candle
- **JVM tooling:**
  - https://github.com/IGJoshua/coffi
  - https://jdk.java.net/jextract/
  - https://openjdk.org/projects/jdk/25/
