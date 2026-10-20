# A "JVM mlx-node"? gen.clj's layers, and what of mlx-node they need

*2026-10-20. The owner's question: should gen.clj grow a JVM counterpart of
mlx-node, a library it loads, or should everything stay inside gen.clj? The
owner added that GenMLX's mlx-node is heavily patched for PPL-like features.
A recommendation, to decide at M6's spike (genclj-kaa8). Bean: the module
split, genclj-wqka, as a draft.*

## 1. Upstream mlx-node

`mlx-node/mlx-node` (MIT) is a local-AI platform, not a binding: a Rust
compute layer (`crates/mlx-sys`, `mlx-core`, `mlx-metal`, `mlx-paged-attn`)
under TypeScript packages. It provides:
- **model inference:** its own forward passes for Qwen3, Qwen3.5 dense and
  MoE, Gemma4, LFM2.5, VLMs and Qwen3-ASR; chat sessions that reuse the KV
  cache; tool calling; speculative decoding; paged attention;
- **training:** GRPO with four loss variants (`packages/trl`);
- **applications:** a pi-based coding agent (`packages/agent`), a server, a
  TUI, a desktop app.

Upstream moves fast: 137 commits in the fork's history are from one author.

## 2. GenMLX's fork (`~/code/genmlx/mlx-node`)

`robert-johansson/mlx-node`, with `upstream` the original. It has two lines
of development, neither of which contains the other:
- **`main`:** 45 commits beyond upstream.
- **The checked-out head:** 118 commits beyond upstream; 142 files,
  +17,220/−883 lines.

Its MLX is a submodule pointing at a patched MLX
(`robert-johansson/mlx`, 6a7dca09). The patches are what made GenMLX's LLM
work a PPL:

| Feature | Commits |
|---|---|
| `genmlx-core`, a superset NAPI addon: 138 module-level exports for GenMLX's layer 0 (ops, reductions, einsum), `vmap` and `compile`, `valueAndGrad` and `computeGradients`, a keyed PRNG, transforms, memory control | e9b2b1b5, 9f01b283, 13705fea, 5f3c8ee5, f86a3440 (`main`) |
| Per-token `forward` and `forwardWithCache` exports: Qwen3, Gemma4, Qwen3.5 dense and MoE, and on CUDA | 8d2e2b46, 46e25cec (`main`); c678fa43 |
| Branchable KV caches: Qwen3.5's isolated deep-copy fork, the MoE `CacheHandle` with `branch_from` | 0359a82b, ac8a11e9 |
| An O(1) fork of a pi session on the genmlx provider | 863c9123 |
| A flat VLM prefill, for branchable image conditioning | 01cc22c1 |
| GRPO's KL penalty to the base model through autograd; batched gradient evaluation | af2d2c56, 1d96be8b |

Keeping that current means rebasing those commits, and the patched MLX
under them, onto a fast-moving upstream.

## 3. What gen.clj has, and where the rest goes

| The fork's PPL feature | gen.clj |
|---|---|
| The layer-0 surface (ops, `vmap`, `compile`, gradients, keyed PRNG, memory) | **Replaced in M1/M2.** Our own binding, jextract over mlx-c's stable C API (`src-java/gen/mlx`), under `gen.array`'s autodiff tape, batching tracer, `array/compile` and `gen.random`. Only small backported patches to MLX (`native/patches/`). |
| Per-token forward passes | llama.cpp's public API returns full logits per token (`gen.llm.llama`). The MLX-native engine (genclj-gvfa) writes its forward passes in Clojure over `gen.array`, exposed by construction. |
| Branchable KV caches | llama.cpp's `seq_cp` under the Branching protocol, with no patch (`gen.llm.branching`). In the MLX-native engine a cache is made of immutable MLX arrays, so a fork is a shared reference: free, with no sequence cap. |
| The O(1) session fork | The LM protocol's trie (`gen.llm.lm`); kmet's sessions in M6. |
| A branchable VLM prefill | genclj-oxz2 (vision). |
| GRPO with a KL penalty through autograd | genclj-q4xv (training), on the MLX-native engine. |
| The agent provider | M6, as a kmet extension (`docs/research/kmet-integration.md`). |

**Not ported:**
- the model zoo: llama.cpp covers about 150 architectures, quantized, on
  Metal, CUDA and CPU (DESIGN §6: it keeps gen.clj a PPL rather than an
  inference engine);
- the agent: kmet takes its place;
- the server, the TUI and the desktop app: applications;
- ASR, unless a later bean asks for it.

So the fork serves as a **requirements document**. The LLM layer must provide
per-token logits, cheap forks, device-resident values and gradients through
forward passes. Nothing of mlx-node is ported as code. And gen.clj should not
fork a runtime again: no patched third-party binding, no patched llama.cpp,
and its own forward passes where it needs MLX.

## 4. The layers

Three layers, as JAX, Flax and GenJAX are three libraries, plus applications
above them:

| Layer | Namespaces (today) | Depends on |
|---|---|---|
| **1. Arrays** | `gen.array` (`.dtype`, `.shape`, `.special`, `.linalg`, `.tape`, `.batch`, `.protocols`), `gen.backend.jvm`, `gen.backend.mlx` (+ `.exact`, `src-java/gen/mlx`), `gen.random` (+ `.threefry`) | Clojure; MLX optional, loaded only when chosen |
| **2. The LLM runtime** | `gen.llm.branching`, `.slots`, `.llama` (+ `src-java/gen/llama`), `gen.llm.lm`, `.cache`, `.trie`, `.cells`; later the MLX-native engine, model loading, tokenizers, chat templates, training | layer 1 (for the MLX engine); llama.cpp optional |
| **3. The PPL** | `gen.choicemap`, `.selection`, `.trace`, `.diff`, `gen.gfi`, `gen.dynamic`, `gen.dist*`, `gen.combinators`, `gen.custom`, `gen.inference.*`, `gen.gradients`, `gen.learning` | layer 1 |
| **3′. LLM generative functions** | the token generative function (genclj-0cvv), grammar middleware (genclj-q38h), token SMC (genclj-loyp), structured output (genclj-w120) | layers 2 and 3 |
| **Applications** | the kmet integration (M6, M7), servers, examples | whatever they use |

- Layer 1 is already a clean boundary: only L0 knows a backend (CLAUDE.md).
  It is the most reusable piece, NumPy and JAX for Clojure with an optional
  GPU.
- Layer 2 is the "JVM mlx-node" part. It is usable without the PPL: kmet's
  local-model provider (M6) needs it, but not the GFI. That second consumer is
  the real argument for drawing the line.
- The LLM layer's exception to rule 3 (host floats, host token draws; DESIGN
  §9) lives in layers 2 and 3′.

## 5. When, and how

- **Not yet.** Extract once there is a second consumer and the APIs have
  settled. The Branching and LM protocols were approved only on 2026-10-20,
  and the token generative function does not exist yet.
- **The cheap first step is modules in one repo,** each with its own
  `deps.edn` and a layering check that fails the build when a module requires
  one above it, as kmet's `modules/` with their `:rank` and layering guard
  (`kmet.test-module-deps`). The boundaries become real without separate
  repos, releases or version skew. Splitting into repos later is mechanical.
- **Decide at M6's spike** (genclj-kaa8), which needs layer 2 inside kmet.

## 6. What it implies

- **API stability between the layers.** A layer's public functions become a
  contract, with the laws and the conformance suite as its tests.
- **One memory budget across layers 1 and 2** (genclj-zztc): MLX's arrays and
  llama.cpp's caches share the GPU.
- **Licenses.** mlx-node is MIT, kmet EPL-2.0, and gen.clj has no license
  yet. Each artifact needs one before it is shared.
- **The docs follow:** CLAUDE.md's layers table, DESIGN §2 and §6.
