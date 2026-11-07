"""GGUF files for the tests of gen.array/load-gguf, gen.llm.models.qwen3 and
gen.llm.models.qwen35 (test/gen/array/gguf_test.clj,
test/gen/llm/models/qwen3_test.clj, qwen35_test.clj).

Run with any Python that has numpy (e.g. GenMLX's venv), from the
repository's root, with llama.cpp's gguf-py (native/build-llama.sh's
checkout):

    ~/code/genmlx/.venv/bin/python -I test/fixtures/gguf/generate.py \
        [native/build/src/llama.cpp/gguf-py]

The files:

- qwen3-tiny-f32.gguf: a tiny Qwen3 with random weights, as llama.cpp's
  converter writes a Qwen3 (general.architecture qwen3, llama.cpp's
  tensor names and metadata keys), every tensor F32: vocabulary 128,
  hidden 64, 3 blocks, 4 query heads and 2 KV heads of 16, feed-forward
  128, RoPE base 10000, eps 1e-6, a tied output (no output.weight).
  Matrices N(0, 1/n_in) (the embedding N(0, 0.2^2)), norm weights
  1 + N(0, 0.1^2).
- qwen3-tiny-q8_0.gguf: the same weights with the matrices (the
  embedding among them) Q8_0, by gguf-py's quantizer; the norms F32, as
  llama.cpp's Q8_0 files have them.
- qwen3-tiny-q8_0-0000N-of-00003.gguf: qwen3-tiny-q8_0.gguf split in
  three, as llama.cpp's gguf-split writes a split model (split.no,
  split.count and split.tensors.count; the metadata in the first).
- qwen35-tiny-f32.gguf and qwen35-tiny-q8_0.gguf: a tiny Qwen3.5
  (general.architecture qwen35, as llama.cpp's converter writes one:
  its tensor names and metadata keys, RMSNorm weights with Qwen3-Next's
  +1 in them, ssm_a as -exp(A_log)): vocabulary 128, hidden 32, 4 blocks,
  blocks 0-2 GatedDeltaNet (2 key heads and 4 value heads of 32, conv
  kernel 4) and block 3 gated attention (4 query and 2 KV heads of 16,
  RoPE on 8 of them, sections [2 1 1 0], base 10000), feed-forward 64,
  eps 1e-6, a tied output, and one MTP block (blk.4, nextn_predict_layers
  1), which llama.cpp loads and its forward does not run. Matrices
  N(0, 1/n_in), the embedding N(0, 0.2^2), norms 1 + N(0, 0.1^2), the
  conv kernel N(0, 0.3^2), ssm_a -exp(N(0, 0.5^2)), ssm_dt.bias
  N(0, 0.5^2). Q8_0 quantizes the matrices, as qwen3-tiny-q8_0.gguf.
- qwen35moe-tiny-f32.gguf and qwen35moe-tiny-q8_0.gguf: the same layers
  (general.architecture qwen35moe, no MTP block) with a mixture of
  experts for the feed-forward: 4 experts of 32, 2 used a token, and a
  shared expert of 32 gated by ffn_gate_inp_shexp. The router
  (ffn_gate_inp) and the shared expert's gate stay F32, as llama.cpp's
  quantizer keeps them.
- types.gguf: one tensor of each type the loader reads (F32, F16, Q8_0,
  Q4_0, Q4_1 and I32; a quantized one of 3 axes, and a Q8_0 block whose
  scale is past 511.75, among them) and a metadata value of every GGUF
  value type but float64.
- types.edn: types.gguf's tensors' values by gguf-py's reader and
  dequantizer, the loader's reference.
- float64-metadata.gguf: float64 metadata values (which MLX's loader
  misreads: gen.backend.mlx.io) before a string, and one F32 tensor.
- unsupported-q4_k.gguf and unsupported-bf16.gguf: a tensor of a type the
  loader refuses (Q4_K's bytes are random: only its type is read).

With --with-vocabulary <a Qwen3.5 GGUF> <dir>, it writes instead, into
dir, tiny random Qwen3.5 and Qwen3.5-MoE models with that GGUF's
tokenizer (llama.cpp loads a model only with one), for
gen.llm.models.qwen35-llama-test's check of the conventions a model of
unequal key and value heads needs (Qwen3.5-0.8B has 16 of each): F32 and
Q8_0, hidden 64, 3 GatedDeltaNet blocks of 2 key and 4 value heads of
64 and a gated attention block of 4 query and 2 KV heads of 64 (RoPE on
16, sections [3 3 2 0]); the MoE's 4 experts of 64, 2 used, and a shared
one. About 76 MB (F32) and 28 MB (Q8_0) each, its embedding of the whole
vocabulary: not committed. Their check runs with
-J-Dgen.qwen35.vocab-models=<dir>:

    ~/code/genmlx/.venv/bin/python -I test/fixtures/gguf/generate.py \
        --with-vocabulary ~/gen-clj-models/Qwen3.5-0.8B-Q8_0.gguf target/qwen35-vocab-models

Fixed seed: the same files each run (numpy's PCG64). gguf-py's metadata
module imports yaml for model cards, which this script never reads; it
is stubbed when yaml is not installed.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ARGS = sys.argv[1:]
VOCABULARY = None
if "--with-vocabulary" in ARGS:
    i = ARGS.index("--with-vocabulary")
    VOCABULARY = ARGS[i + 1:i + 3]
    ARGS = ARGS[:i] + ARGS[i + 3:]
GGUF_PY = ARGS[0] if ARGS else os.path.join(ROOT, "native/build/src/llama.cpp/gguf-py")

try:
    import yaml  # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

Q = gguf.GGMLQuantizationType

VOCAB, HIDDEN, BLOCKS, HEADS, KV_HEADS, HEAD_DIM, FFN = 128, 64, 3, 4, 2, 16, 128
CONTEXT, ROPE_BASE, EPS = 256, 10000.0, 1e-6


def tiny_weights(rng):
    """[(name, array, quantizable?)] of the tiny Qwen3, in llama.cpp's
    names; numpy's shapes are ggml's reversed, so a matrix is [n_out, n_in]."""
    def matrix(n_out, n_in, sd=None):
        return rng.normal(0.0, sd if sd is not None else 1.0 / np.sqrt(n_in), size=(n_out, n_in))

    def norm(n):
        return 1.0 + rng.normal(0.0, 0.1, size=(n,))

    out = [("token_embd.weight", matrix(VOCAB, HIDDEN, 0.2), True),
           ("output_norm.weight", norm(HIDDEN), False)]
    for i in range(BLOCKS):
        p = "blk.%d." % i
        out += [(p + "attn_norm.weight", norm(HIDDEN), False),
                (p + "attn_q.weight", matrix(HEADS * HEAD_DIM, HIDDEN), True),
                (p + "attn_k.weight", matrix(KV_HEADS * HEAD_DIM, HIDDEN), True),
                (p + "attn_v.weight", matrix(KV_HEADS * HEAD_DIM, HIDDEN), True),
                (p + "attn_q_norm.weight", norm(HEAD_DIM), False),
                (p + "attn_k_norm.weight", norm(HEAD_DIM), False),
                (p + "attn_output.weight", matrix(HIDDEN, HEADS * HEAD_DIM), True),
                (p + "ffn_norm.weight", norm(HIDDEN), False),
                (p + "ffn_gate.weight", matrix(FFN, HIDDEN), True),
                (p + "ffn_up.weight", matrix(FFN, HIDDEN), True),
                (p + "ffn_down.weight", matrix(HIDDEN, FFN), True)]
    return out


def qwen3_metadata(w, file_type):
    w.add_type(gguf.GGUFType.MODEL)
    w.add_name("gen.clj test model: a tiny random Qwen3")
    w.add_file_type(file_type)
    w.add_context_length(CONTEXT)
    w.add_embedding_length(HIDDEN)
    w.add_block_count(BLOCKS)
    w.add_feed_forward_length(FFN)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_layer_norm_rms_eps(EPS)


# A tiny Qwen3.5 (qwen35, and qwen35moe with experts)
Q35_VOCAB, Q35_HIDDEN, Q35_BLOCKS, Q35_FFN = 128, 32, 4, 64
Q35_HEADS, Q35_KV_HEADS, Q35_HEAD_DIM, Q35_ROPE_DIMS = 4, 2, 16, 8
Q35_K_HEADS, Q35_V_HEADS, Q35_STATE, Q35_CONV = 2, 4, 32, 4
Q35_EXPERTS, Q35_USED, Q35_EXPERT_FFN, Q35_SHARED_FFN = 4, 2, 32, 32
Q35_SIZES = {"VOCAB": Q35_VOCAB, "HIDDEN": Q35_HIDDEN, "BLOCKS": Q35_BLOCKS, "FFN": Q35_FFN, "HEADS": Q35_HEADS,
             "KV_HEADS": Q35_KV_HEADS, "HEAD_DIM": Q35_HEAD_DIM, "ROPE_DIMS": Q35_ROPE_DIMS, "SECTIONS": [2, 1, 1, 0],
             "K_HEADS": Q35_K_HEADS, "V_HEADS": Q35_V_HEADS, "STATE": Q35_STATE, "CONV": Q35_CONV, "EXPERTS": Q35_EXPERTS,
             "USED": Q35_USED, "EXPERT_FFN": Q35_EXPERT_FFN, "SHARED_FFN": Q35_SHARED_FFN, "ROPE_BASE": ROPE_BASE,
             "CONTEXT": CONTEXT}


def qwen35_weights(rng, moe, sizes=None):
    """[(name, array, quantizable?)] of the tiny Qwen3.5 (moe: Qwen3.5-MoE,
    without the MTP block), in llama.cpp's names, shapes [n_out, n_in];
    sizes (a dict) overrides the Q35_ constants, by their names' rest
    (and writes no MTP block)."""
    z = dict(Q35_SIZES, **(sizes or {}))
    Q35_VOCAB, Q35_HIDDEN, Q35_BLOCKS, Q35_FFN = z["VOCAB"], z["HIDDEN"], z["BLOCKS"], z["FFN"]
    Q35_HEADS, Q35_KV_HEADS, Q35_HEAD_DIM = z["HEADS"], z["KV_HEADS"], z["HEAD_DIM"]
    Q35_K_HEADS, Q35_V_HEADS, Q35_STATE, Q35_CONV = z["K_HEADS"], z["V_HEADS"], z["STATE"], z["CONV"]
    Q35_EXPERTS, Q35_EXPERT_FFN, Q35_SHARED_FFN = z["EXPERTS"], z["EXPERT_FFN"], z["SHARED_FFN"]
    def matrix(*shape, sd=None):
        return rng.normal(0.0, sd if sd is not None else 1.0 / np.sqrt(shape[-1]), size=shape)

    def norm(n):
        return 1.0 + rng.normal(0.0, 0.1, size=(n,))

    key_dim, value_dim = Q35_K_HEADS * Q35_STATE, Q35_V_HEADS * Q35_STATE
    conv_dim = 2 * key_dim + value_dim
    out = [("token_embd.weight", matrix(Q35_VOCAB, Q35_HIDDEN, sd=0.2), True),
           ("output_norm.weight", norm(Q35_HIDDEN), False)]

    def attention(p):
        return [(p + "attn_q.weight", matrix(2 * Q35_HEADS * Q35_HEAD_DIM, Q35_HIDDEN), True),
                (p + "attn_k.weight", matrix(Q35_KV_HEADS * Q35_HEAD_DIM, Q35_HIDDEN), True),
                (p + "attn_v.weight", matrix(Q35_KV_HEADS * Q35_HEAD_DIM, Q35_HIDDEN), True),
                (p + "attn_q_norm.weight", norm(Q35_HEAD_DIM), False),
                (p + "attn_k_norm.weight", norm(Q35_HEAD_DIM), False),
                (p + "attn_output.weight", matrix(Q35_HIDDEN, Q35_HEADS * Q35_HEAD_DIM), True)]

    def ffn(p):
        if not moe:
            return [(p + "ffn_gate.weight", matrix(Q35_FFN, Q35_HIDDEN), True),
                    (p + "ffn_up.weight", matrix(Q35_FFN, Q35_HIDDEN), True),
                    (p + "ffn_down.weight", matrix(Q35_HIDDEN, Q35_FFN), True)]
        return [(p + "ffn_gate_inp.weight", matrix(Q35_EXPERTS, Q35_HIDDEN), False),
                (p + "ffn_gate_exps.weight", matrix(Q35_EXPERTS, Q35_EXPERT_FFN, Q35_HIDDEN), True),
                (p + "ffn_up_exps.weight", matrix(Q35_EXPERTS, Q35_EXPERT_FFN, Q35_HIDDEN), True),
                (p + "ffn_down_exps.weight", matrix(Q35_EXPERTS, Q35_HIDDEN, Q35_EXPERT_FFN), True),
                (p + "ffn_gate_inp_shexp.weight", matrix(Q35_HIDDEN, sd=1.0 / np.sqrt(Q35_HIDDEN)), False),
                (p + "ffn_gate_shexp.weight", matrix(Q35_SHARED_FFN, Q35_HIDDEN), True),
                (p + "ffn_up_shexp.weight", matrix(Q35_SHARED_FFN, Q35_HIDDEN), True),
                (p + "ffn_down_shexp.weight", matrix(Q35_HIDDEN, Q35_SHARED_FFN), True)]

    for i in range(Q35_BLOCKS):
        p = "blk.%d." % i
        out += [(p + "attn_norm.weight", norm(Q35_HIDDEN), False),
                (p + "post_attention_norm.weight", norm(Q35_HIDDEN), False)]
        if (i + 1) % 4 == 0:
            out += attention(p)
        else:
            out += [(p + "attn_qkv.weight", matrix(conv_dim, Q35_HIDDEN), True),
                    (p + "attn_gate.weight", matrix(value_dim, Q35_HIDDEN), True),
                    (p + "ssm_conv1d.weight", matrix(conv_dim, Q35_CONV, sd=0.3), False),
                    (p + "ssm_dt.bias", rng.normal(0.0, 0.5, size=(Q35_V_HEADS,)), False),
                    (p + "ssm_a", -np.exp(rng.normal(0.0, 0.5, size=(Q35_V_HEADS,))), False),
                    (p + "ssm_beta.weight", matrix(Q35_V_HEADS, Q35_HIDDEN), True),
                    (p + "ssm_alpha.weight", matrix(Q35_V_HEADS, Q35_HIDDEN), True),
                    (p + "ssm_norm.weight", norm(Q35_STATE), False),
                    (p + "ssm_out.weight", matrix(Q35_HIDDEN, value_dim), True)]
        out += ffn(p)
    if not moe and not sizes:
        # the MTP block: a gated attention block and its NextN tensors
        p = "blk.%d." % Q35_BLOCKS
        out += [(p + "attn_norm.weight", norm(Q35_HIDDEN), False),
                (p + "post_attention_norm.weight", norm(Q35_HIDDEN), False)]
        out += attention(p) + ffn(p)
        out += [(p + "nextn.eh_proj.weight", matrix(Q35_HIDDEN, 2 * Q35_HIDDEN), True),
                (p + "nextn.enorm.weight", norm(Q35_HIDDEN), False),
                (p + "nextn.hnorm.weight", norm(Q35_HIDDEN), False),
                (p + "nextn.shared_head_norm.weight", norm(Q35_HIDDEN), False)]
    return out


def qwen35_metadata(w, file_type, moe, sizes=None, mtp=True):
    z = dict(Q35_SIZES, **(sizes or {}))
    w.add_type(gguf.GGUFType.MODEL)
    w.add_name("gen.clj test model: a tiny random Qwen3.5" + ("-MoE" if moe else ""))
    w.add_file_type(file_type)
    w.add_context_length(z["CONTEXT"])
    w.add_embedding_length(z["HIDDEN"])
    w.add_block_count(z["BLOCKS"] + (1 if mtp else 0))
    if mtp:
        w.add_nextn_predict_layers(1)
    w.add_feed_forward_length(z["FFN"])
    w.add_head_count(z["HEADS"])
    w.add_head_count_kv(z["KV_HEADS"])
    w.add_key_length(z["HEAD_DIM"])
    w.add_value_length(z["HEAD_DIM"])
    w.add_rope_freq_base(z["ROPE_BASE"])
    w.add_rope_dimension_count(z["ROPE_DIMS"])
    w.add_rope_dimension_sections(z["SECTIONS"])
    w.add_layer_norm_rms_eps(EPS)
    w.add_full_attention_interval(4)
    w.add_ssm_conv_kernel(z["CONV"])
    w.add_ssm_state_size(z["STATE"])
    w.add_ssm_group_count(z["K_HEADS"])
    w.add_ssm_time_step_rank(z["V_HEADS"])
    w.add_ssm_inner_size(z["V_HEADS"] * z["STATE"])
    if moe:
        w.add_expert_count(z["EXPERTS"])
        w.add_expert_used_count(z["USED"])
        w.add_expert_feed_forward_length(z["EXPERT_FFN"])
        w.add_expert_shared_feed_forward_length(z["SHARED_FFN"])


def write_qwen35(name, weights, quantized, moe, sizes=None, out_dir=HERE, extra_kv=()):
    path = os.path.join(out_dir, name)
    w = gguf.GGUFWriter(path, "qwen35moe" if moe else "qwen35")
    qwen35_metadata(w, gguf.LlamaFileType.MOSTLY_Q8_0 if quantized else gguf.LlamaFileType.ALL_F32, moe, sizes,
                    mtp=not moe and not sizes)
    for key, value, vtype, sub in extra_kv:
        if sub is None:
            w.add_key_value(key, value, vtype)
        else:
            w.add_key_value(key, value, vtype, sub)
    for tname, arr, quantizable in weights:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if quantized and quantizable:
            w.add_tensor(tname, gguf.quants.quantize(arr, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(tname, arr)
    finish(w, [path])


def finish(w, paths):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    for p in paths:
        print("wrote %s (%d bytes)" % (os.path.relpath(p, ROOT), os.path.getsize(p)))


def write_model(name, weights, quantized, split=0):
    path = os.path.join(HERE, name)
    w = gguf.GGUFWriter(path, "qwen3", split_max_tensors=split)
    qwen3_metadata(w, gguf.LlamaFileType.MOSTLY_Q8_0 if quantized else gguf.LlamaFileType.ALL_F32)
    for tname, arr, quantizable in weights:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if quantized and quantizable:
            w.add_tensor(tname, gguf.quants.quantize(arr, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(tname, arr)
    if split:
        stem = path[:-len(".gguf")]
        n = (len(weights) + split - 1) // split
        finish(w, ["%s-%05d-of-%05d.gguf" % (stem, i + 1, n) for i in range(n)])
    else:
        finish(w, [path])


def edn(x):
    """x (nested lists of numbers) as EDN, each float32 value as the
    shortest decimal that reads back as it in a double."""
    if isinstance(x, list):
        return "[" + " ".join(edn(v) for v in x) + "]"
    return repr(float(x)) if isinstance(x, float) else str(int(x))


def write_types(rng):
    path = os.path.join(HERE, "types.gguf")
    w = gguf.GGUFWriter(path, "test")
    V = gguf.GGUFValueType
    for key, value, vtype in [("test.u8", 200, V.UINT8), ("test.i8", -100, V.INT8),
                              ("test.u16", 60000, V.UINT16), ("test.i16", -30000, V.INT16),
                              ("test.u32", 4000000000, V.UINT32), ("test.i32", -2000000000, V.INT32),
                              ("test.f32", 0.1, V.FLOAT32), ("test.bool", True, V.BOOL),
                              ("test.string", "Stockholm är Sveriges huvudstad", V.STRING),
                              ("test.u64", 2 ** 63 + 5, V.UINT64), ("test.i64", -(2 ** 62), V.INT64)]:
        w.add_key_value(key, value, vtype)
    w.add_key_value("test.ints", [1, -2, 3], V.ARRAY, V.INT32)
    w.add_key_value("test.strings", ["a", "", "ö"], V.ARRAY, V.STRING)
    f = lambda *shape: rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    w.add_tensor("t.f32.weight", f(2, 64))
    w.add_tensor("t.vector", f(5))
    w.add_tensor("t.f16.weight", f(2, 64).astype(np.float16))
    w.add_tensor("t.i32", np.array([7, -3, 2 ** 31 - 1, -2 ** 31, 0], dtype=np.int32))
    # Q8_0 with a block of zeros, and one whose scale d is past 511.75,
    # where -128·d overflows float16 (MLX's loader computes the bias so);
    # and a 3-axis tensor
    q8 = f(3, 64)
    q8[1, :32] = 0.0
    q8[2, 32:] *= 1000.0
    q8[2, 33] = 80000.0
    w.add_tensor("t.q8_0.weight", gguf.quants.quantize(q8, Q.Q8_0), raw_dtype=Q.Q8_0)
    w.add_tensor("t.q8_0_3d.weight", gguf.quants.quantize(f(2, 3, 64), Q.Q8_0), raw_dtype=Q.Q8_0)
    w.add_tensor("t.q4_0.weight", gguf.quants.quantize(f(3, 64), Q.Q4_0), raw_dtype=Q.Q4_0)
    w.add_tensor("t.q4_1.weight", gguf.quants.quantize(f(3, 64) + 0.5, Q.Q4_1), raw_dtype=Q.Q4_1)
    finish(w, [path])
    # every tensor's values by gguf-py's own reader and dequantizer (float32)
    r = gguf.GGUFReader(path)
    values = {}
    for t in r.tensors:
        shape = [int(d) for d in reversed(t.shape)]
        v = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type in (Q.Q8_0, Q.Q4_0, Q.Q4_1) else t.data
        values[t.name] = np.asarray(v).reshape(shape).tolist()
    out = os.path.join(HERE, "types.edn")
    with open(out, "w") as fo:
        fo.write(";; types.gguf's tensors, by gguf-py's reader and dequantizer (generate.py)\n{")
        fo.write("\n ".join('"%s" %s' % (k, edn(v)) for k, v in sorted(values.items())))
        fo.write("}\n")
    print("wrote %s (%d bytes)" % (os.path.relpath(out, ROOT), os.path.getsize(out)))


def write_float64():
    # float64 metadata, which MLX's loader misreads (gen.backend.mlx.io)
    path = os.path.join(HERE, "float64-metadata.gguf")
    w = gguf.GGUFWriter(path, "test")
    w.add_key_value("test.f64", 0.1, gguf.GGUFValueType.FLOAT64)
    w.add_key_value("test.f64s", [0.5, -2.25], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT64)
    w.add_key_value("test.after", "read after the float64 values", gguf.GGUFValueType.STRING)
    w.add_tensor("t.weight", np.arange(64, dtype=np.float32).reshape(2, 32))
    finish(w, [path])


def write_unsupported(rng):
    for name, data, qtype in [("unsupported-q4_k.gguf", rng.integers(0, 256, size=(2, 144), dtype=np.uint8), Q.Q4_K),
                              ("unsupported-bf16.gguf",
                               gguf.quants.quantize(rng.normal(size=(2, 32)).astype(np.float32), Q.BF16), Q.BF16)]:
        path = os.path.join(HERE, name)
        w = gguf.GGUFWriter(path, "test")
        w.add_tensor("ok.weight", np.ones((2, 32), dtype=np.float32))
        w.add_tensor("refused.weight", data, raw_dtype=qtype)
        finish(w, [path])


def write_with_vocabulary(source, out_dir):
    """Tiny random Qwen3.5 and Qwen3.5-MoE models with the tokenizer of the
    GGUF source (its tokenizer.* keys but the chat template), into out_dir:
    for checks against llama.cpp (the module's docstring)."""
    V = gguf.GGUFValueType
    r = gguf.GGUFReader(source)
    kv = []
    for f in r.fields.values():
        if f.name.startswith("tokenizer.") and f.name != "tokenizer.chat_template":
            vtype = V(f.types[0])
            kv.append((f.name, f.contents(), vtype, V(f.types[-1]) if vtype == V.ARRAY else None))
    sizes = {"VOCAB": len(r.fields["tokenizer.ggml.tokens"].data), "HIDDEN": 64, "FFN": 128, "HEAD_DIM": 64,
             "ROPE_DIMS": 16, "SECTIONS": [3, 3, 2, 0], "STATE": 64, "EXPERT_FFN": 64, "SHARED_FFN": 64,
             "ROPE_BASE": 1.0e7, "CONTEXT": 4096}
    os.makedirs(out_dir, exist_ok=True)
    for moe, stem in [(False, "qwen35"), (True, "qwen35moe")]:
        weights = qwen35_weights(np.random.default_rng(1108 + moe), moe, sizes)
        for quantized, kind in [(False, "f32"), (True, "q8_0")]:
            write_qwen35("%s-vocab-%s.gguf" % (stem, kind), weights, quantized, moe, sizes, out_dir, kv)


def main():
    if VOCABULARY:
        write_with_vocabulary(*VOCABULARY)
        return
    rng = np.random.default_rng(20261107)
    weights = tiny_weights(rng)
    write_model("qwen3-tiny-f32.gguf", weights, False)
    write_model("qwen3-tiny-q8_0.gguf", weights, True)
    write_model("qwen3-tiny-q8_0.gguf", weights, True, split=14)
    write_types(np.random.default_rng(7))
    write_float64()
    write_unsupported(np.random.default_rng(8))
    for moe, stem in [(False, "qwen35"), (True, "qwen35moe")]:
        weights = qwen35_weights(np.random.default_rng(20261108 + moe), moe)
        write_qwen35(stem + "-tiny-f32.gguf", weights, False, moe)
        write_qwen35(stem + "-tiny-q8_0.gguf", weights, True, moe)


if __name__ == "__main__":
    main()
