"""Safetensors files and Hugging Face checkpoints for the tests of
gen.array/load-safetensors and of qwen3/load and qwen35/load on a
checkpoint (test/gen/array/safetensors_test.clj,
test/gen/llm/models/checkpoint_test.clj).

Run with any Python that has numpy (e.g. GenMLX's venv), from the
repository's root, with llama.cpp's gguf-py (native/build-llama.sh's
checkout), which reads the GGUF fixtures the checkpoints are made from:

    ~/code/genmlx/.venv/bin/python -I test/fixtures/safetensors/generate.py \
        [native/build/src/llama.cpp/gguf-py]

The safetensors files are written here, with numpy and the format's own
layout (an 8-byte header length, the JSON header padded with spaces to 8
bytes, the data); no safetensors or torch package is needed.

The checkpoints are the GGUF fixtures of test/fixtures/gguf (generate.py
there) undone: each is the checkpoint llama.cpp's converter
(convert_hf_to_gguf.py) would write that GGUF from, so that a checkpoint
and its GGUF give the same model, exactly. The converter's transforms are
inverted, each exactly in float32:
- the names (model.layers.i.self_attn.q_proj.weight from
  blk.i.attn_q.weight, …) and config.json from the metadata;
- Qwen3.5's RMSNorm weights less 1 (the converter adds Qwen3-Next's +1;
  w - 1 is exact for w in [1/2, 2], checked), but the gated norm's;
- A_log from ssm_a = -exp(A_log): the float32 nearest log(-ssm_a) whose
  -exp, computed in float64 and rounded to float32 (the JVM backend's
  exp), is ssm_a again, searched a few ulps around it (checked);
- the value heads from tiled back to grouped order (the converter's
  _reorder_v_heads, inverted) in attn_qkv's value rows, attn_gate,
  ssm_beta, ssm_alpha, ssm_a, ssm_dt.bias, ssm_conv1d's value channels
  and ssm_out's columns (2 key and 4 value heads here, so the order
  matters);
- the conv kernel [C, 4] as PyTorch's Conv1d holds it, [C, 1, 4];
- the experts' stacks as one matrix an expert, the shared expert's gate
  [1, hidden].
MLX's quantized checkpoints (mlx-lm's layout, as mlx-community's) are made
from the Q8_0 GGUFs: a Q8_0 block is MLX's affine quantization at 8 bits,
group 32, exactly (words q + 128, scale d, bias -128·d in float16, as the
block's d is), so the triples are those load-gguf gives.

The files:

- types.safetensors and types.edn: a tensor of each dtype the loader
  reads but F64 (F32, F16, BF16 with a subnormal and -0, I32, I64 at
  ±2^53, U32 past 2^31, BOOL), a 0-d and an empty one, an I8 one (refused
  when selected), and __metadata__; types.edn their values (BF16's from
  its bits), the reference.
- f64.safetensors: an F64 tensor (beside an F32 one), which the JVM
  backend reads and MLX's loader refuses, the whole file.
- i64-past-2-53.safetensors: an I64 tensor holding 2^53 + 1, which the
  JVM backend's doubles do not hold exactly.
- qwen3-tiny/: Qwen3 (Qwen3ForCausalLM), from qwen3-tiny-f32.gguf, F32,
  with a tied output and an lm_head copy of the embedding stored, as
  Qwen3-0.6B's checkpoint has one.
- qwen3-tiny-8bit/: MLX's 8-bit checkpoint (group 32) of
  qwen3-tiny-q8_0.gguf, sharded in three by model.safetensors.index.json,
  some matrices' words, scales and biases in different shards.
- qwen3-tiny-4bit/: MLX's 4-bit checkpoint at group 64 of the same shapes,
  random codes, scales and biases (float16), its embedding quantized, one
  matrix (model.layers.1.mlp.down_proj) at 8 bits, group 32 by its own
  quantization entry, the norms BF16 (qwen3-tiny-f32.gguf's, rounded);
  qwen3-tiny-4bit.edn two rows of two matrices dequantized, MLX's way
  (scale·q + bias, the product and the sum each rounded to float32).
- qwen35-tiny/: Qwen3.5 (Qwen3_5ForConditionalGeneration, its text model
  under model.language_model, config.json's text_config) from
  qwen35-tiny-f32.gguf, F32 (A_log too), its MTP block (mtp.*) and a
  vision tensor (model.visual.*, I8: loading it would be loud), which the
  text model never reads.
- qwen35-tiny-mlx-8bit/: Qwen3.5 as mlx-lm saves it, from
  qwen35-tiny-q8_0.gguf: its names under language_model.model, its
  RMSNorm weights already holding the +1 (mlx-lm's sanitize adds it
  before saving), the conv kernel [C, 4, 1] (MLX's layout), no MTP
  block, the value heads grouped and A_log as in Hugging Face's, the
  matrices 8-bit.
- qwen35moe-tiny/: Qwen3.5-MoE (qwen3_5_moe) from qwen35moe-tiny-f32.gguf,
  one matrix an expert (model.language_model.layers.i.mlp.experts.e.*),
  the router mlp.gate, the shared expert and its gate.
- qwen35moe-tiny-mlx-8bit/: the same as mlx-lm saves it, from
  qwen35moe-tiny-q8_0.gguf: the experts stacked (mlp.switch_mlp.*), 8-bit.

Fixed seed: the same files each run (numpy's PCG64).
"""
import json
import os
import struct
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GGUF_DIR = os.path.join(ROOT, "test/fixtures/gguf")
GGUF_PY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "native/build/src/llama.cpp/gguf-py")

try:
    import yaml  # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

Q = gguf.GGMLQuantizationType

# ---------------------------------------------------------------------------
# Writing safetensors
# ---------------------------------------------------------------------------

DTYPES = {np.dtype(np.float32): "F32", np.dtype(np.float64): "F64", np.dtype(np.float16): "F16",
          np.dtype(np.int32): "I32", np.dtype(np.int64): "I64", np.dtype(np.uint32): "U32",
          np.dtype(np.bool_): "BOOL", np.dtype(np.int8): "I8"}


class BF16:
    """A bfloat16 tensor's bits (numpy has no bfloat16): bits a uint16 array."""

    def __init__(self, bits):
        self.bits = np.ascontiguousarray(bits, dtype=np.uint16)
        self.shape = self.bits.shape


def bf16_of(x):
    """float32 values rounded to bfloat16 (to nearest, ties to even)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return BF16(rounded.astype(np.uint16))


def bf16_values(t):
    return (t.bits.astype(np.uint32) << 16).view(np.float32)


def entry(arr):
    if isinstance(arr, BF16):
        return "BF16", list(arr.shape), arr.bits.astype("<u2").tobytes()
    arr = np.asarray(arr)
    dt = DTYPES[arr.dtype]
    return dt, list(arr.shape), np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()


def write_safetensors(path, tensors, metadata=None):
    """tensors: [(name, array)] in the data's order."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    blobs = []
    at = 0
    for name, arr in tensors:
        dt, shape, data = entry(arr)
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    h = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    h += b" " * (-len(h) % 8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    print("wrote %s (%d bytes)" % (os.path.relpath(path, ROOT), os.path.getsize(path)))


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    print("wrote %s" % os.path.relpath(path, ROOT))


def write_checkpoint(name, config, tensors, shards=None):
    """A checkpoint's directory: config.json and model.safetensors, or with
    shards (a list of sets of names, one a file) model-0000i-of-0000N
    files and their index."""
    d = os.path.join(HERE, name)
    write_json(os.path.join(d, "config.json"), config)
    meta = {"format": "pt"} if not name.endswith("bit") else {"format": "mlx"}
    if not shards:
        write_safetensors(os.path.join(d, "model.safetensors"), tensors, meta)
        return
    n = len(shards)
    weight_map = {}
    for i, names in enumerate(shards):
        file = "model-%05d-of-%05d.safetensors" % (i + 1, n)
        part = [(k, v) for k, v in tensors if k in names]
        write_safetensors(os.path.join(d, file), part, meta)
        weight_map.update({k: file for k, _ in part})
    assert len(weight_map) == len(tensors)
    write_json(os.path.join(d, "model.safetensors.index.json"),
               {"metadata": {"total_size": sum(len(entry(v)[2]) for _, v in tensors)}, "weight_map": weight_map})


def edn(x):
    if isinstance(x, list):
        return "[" + " ".join(edn(v) for v in x) + "]"
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if v != v:
            return "##NaN"
        return repr(v)
    return str(int(x))


# ---------------------------------------------------------------------------
# Reading the GGUF fixtures
# ---------------------------------------------------------------------------

def gguf_tensors(name):
    """{name: array (F32, [d q] for Q8_0)} of a GGUF fixture, gen.array's
    shapes (outermost first): a Q8_0 tensor [.. n] as its scales d
    (float16 [.. n/32]) and codes q (int8 [.. n])."""
    r = gguf.GGUFReader(os.path.join(GGUF_DIR, name))
    out = {}
    for t in r.tensors:
        shape = [int(s) for s in reversed(t.shape)]
        if t.tensor_type == Q.F32:
            out[t.name] = np.asarray(t.data, dtype=np.float32).reshape(shape)
        elif t.tensor_type == Q.Q8_0:
            blocks = np.ascontiguousarray(t.data).reshape(shape[:-1] + [shape[-1] // 32, 34])
            d = blocks[..., :2].copy().view(np.float16).reshape(shape[:-1] + [shape[-1] // 32])
            q = blocks[..., 2:].copy().view(np.int8).reshape(shape)
            out[t.name] = Q8(d, q)
        else:
            raise ValueError("%s: %s" % (t.name, t.tensor_type))
    return out


class Q8:
    """A Q8_0 matrix: its scales d [.. n/32] (float16) and codes q [.. n]."""

    def __init__(self, d, q):
        self.d, self.q = d, q

    def rows(self, perm):
        return Q8(self.d[perm], self.q[perm])

    def columns(self, perm32):
        """Columns reordered a block of 32 at once: perm32 the blocks'
        order."""
        q = self.q.reshape(self.q.shape[:-1] + (-1, 32))[..., perm32, :].reshape(self.q.shape)
        return Q8(self.d[..., perm32], q)

    def mlx(self, base):
        """MLX's 8-bit triple at group 32: base.weight, .scales, .biases."""
        words = np.ascontiguousarray((self.q.astype(np.int16) + 128).astype(np.uint8)).view("<u4")
        biases = (np.float16(-128.0) * self.d.astype(np.float32)).astype(np.float16)
        assert np.array_equal(biases.astype(np.float32), -128.0 * self.d.astype(np.float32)), base
        return [(base + ".weight", words.astype(np.uint32)), (base + ".scales", self.d), (base + ".biases", biases)]


def matrix(base, w):
    """[(name, array)] of a weight: dense, or a Q8_0 one as MLX's triple."""
    return w.mlx(base) if isinstance(w, Q8) else [(base + ".weight", w)]


# ---------------------------------------------------------------------------
# The converter's transforms, inverted
# ---------------------------------------------------------------------------

def unshift(w):
    """A norm weight w (the GGUF's, Qwen3-Next's +1 in it) as the
    checkpoint holds it: w - 1, exact."""
    h = (w.astype(np.float32) - np.float32(1.0)).astype(np.float32)
    assert np.array_equal((h + np.float32(1.0)).astype(np.float32), w), "w - 1 is not exact"
    return h


def a_log(ssm_a):
    """A_log (float32) for ssm_a = -exp(A_log): the one nearest log(-ssm_a)
    whose -exp, in float64 rounded to float32, is ssm_a."""
    out = np.empty_like(ssm_a, dtype=np.float32)
    for i, v in enumerate(ssm_a.reshape(-1)):
        a0 = np.float32(np.log(-np.float64(v)))
        for k in range(0, 64):
            found = None
            for c in ([a0] if k == 0 else [step(a0, k), step(a0, -k)]):
                if np.float32(-np.exp(np.float64(c))) == v:
                    found = c
                    break
            if found is not None:
                out.reshape(-1)[i] = found
                break
        else:
            raise ValueError("no A_log for ssm_a %r" % v)
    return out


def step(x, k):
    for _ in range(abs(k)):
        x = np.nextafter(x, np.float32(np.inf if k > 0 else -np.inf), dtype=np.float32)
    return x


def grouped(n_k, r):
    """The permutation from tiled to grouped order of n_k·r heads: grouped
    head g·r + v is tiled head v·n_k + g."""
    return np.array([v * n_k + g for g in range(n_k) for v in range(r)])


def expand(perm, d):
    """A permutation of heads as one of their d values each."""
    return (perm[:, None] * d + np.arange(d)[None, :]).reshape(-1)


# ---------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------

def write_types(rng):
    f16 = np.array([[1.5, -0.0], [6.0e-8, 65504.0]], dtype=np.float16)
    bf16 = BF16(np.array([0x3FC0, 0x8000, 0x0001, 0x7F7F, 0xC2F7], dtype=np.uint16))
    tensors = [("t.f32", rng.normal(size=(2, 3)).astype(np.float32)),
               ("t.f16", f16),
               ("t.bf16", bf16),
               ("t.i32", np.array([7, -3, 2 ** 31 - 1, -2 ** 31], dtype=np.int32)),
               ("t.i64", np.array([2 ** 53, -2 ** 53, 12345678901], dtype=np.int64)),
               ("t.u32", np.array([0, 2 ** 31 + 5, 2 ** 32 - 1], dtype=np.uint32)),
               ("t.bool", np.array([True, False, False, True])),
               ("t.scalar", np.array(2.25, dtype=np.float32)),
               ("t.empty", np.zeros((0, 3), dtype=np.float32)),
               ("t.i8", np.array([1, -2], dtype=np.int8))]
    write_safetensors(os.path.join(HERE, "types.safetensors"), tensors,
                      {"format": "pt", "note": "Stockholm är Sveriges huvudstad"})
    values = {}
    for name, arr in tensors:
        if name == "t.i8":
            continue
        v = bf16_values(arr) if isinstance(arr, BF16) else np.asarray(arr)
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        values[name] = v.tolist()
    out = os.path.join(HERE, "types.edn")
    with open(out, "w") as fo:
        fo.write(";; types.safetensors's tensors, their values (generate.py)\n{")
        fo.write("\n ".join('"%s" %s' % (k, edn(v)) for k, v in sorted(values.items())))
        fo.write("}\n")
    print("wrote %s" % os.path.relpath(out, ROOT))
    write_safetensors(os.path.join(HERE, "i64-past-2-53.safetensors"),
                      [("t.i64", np.array([2 ** 53 + 1], dtype=np.int64))])
    write_safetensors(os.path.join(HERE, "f64.safetensors"),
                      [("t.f64", np.array([0.1, -2.5e300, 5e-324])), ("t.f32", np.array([1.5], dtype=np.float32))])


QWEN3_NAMES = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
               "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
               "attn_k_norm": "self_attn.k_norm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
               "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def qwen3_config(sizes, quantization=None):
    c = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False,
         "attention_dropout": 0.0, "head_dim": sizes["HEAD_DIM"], "hidden_act": "silu",
         "hidden_size": sizes["HIDDEN"], "intermediate_size": sizes["FFN"], "max_position_embeddings": 256,
         "max_window_layers": sizes["BLOCKS"], "num_attention_heads": sizes["HEADS"],
         "num_hidden_layers": sizes["BLOCKS"], "num_key_value_heads": sizes["KV_HEADS"], "rms_norm_eps": 1e-06,
         "rope_scaling": None, "rope_theta": 10000, "sliding_window": None, "tie_word_embeddings": True,
         "torch_dtype": "float32", "use_cache": True, "use_sliding_window": False, "vocab_size": sizes["VOCAB"]}
    if quantization:
        c["quantization"] = quantization
        c["quantization_config"] = quantization
    return c


QWEN3_SIZES = {"VOCAB": 128, "HIDDEN": 64, "BLOCKS": 3, "HEADS": 4, "KV_HEADS": 2, "HEAD_DIM": 16, "FFN": 128}


def qwen3_hf(g):
    """[(name, weight)] of a Qwen3 checkpoint from a GGUF's tensors."""
    out = [("model.embed_tokens", g["token_embd.weight"]), ("model.norm", g["output_norm.weight"])]
    for i in range(QWEN3_SIZES["BLOCKS"]):
        for k, v in QWEN3_NAMES.items():
            out.append(("model.layers.%d.%s" % (i, v), g["blk.%d.%s.weight" % (i, k)]))
    return out


def write_qwen3():
    g = gguf_tensors("qwen3-tiny-f32.gguf")
    ws = qwen3_hf(g)
    tensors = [(b + ".weight", w) for b, w in ws] + [("lm_head.weight", g["token_embd.weight"].copy())]
    write_checkpoint("qwen3-tiny", qwen3_config(QWEN3_SIZES), sorted(tensors))
    # MLX's 8-bit checkpoint of the Q8_0 GGUF, in three shards: a few
    # matrices' scales and biases in another shard than their words
    q = gguf_tensors("qwen3-tiny-q8_0.gguf")
    tensors = sorted(t for b, w in qwen3_hf(q) for t in matrix(b, w))
    names = [k for k, _ in tensors]
    shards = [set(names[0:20]), set(names[20:45]), set(names[45:])]
    split = [b for b in {k.rsplit(".", 1)[0] for k in names}
             if len({i for i, s in enumerate(shards) for k in s if k.rsplit(".", 1)[0] == b}) > 1]
    assert split, "no matrix split over shards"
    print("  matrices over two shards: %s" % ", ".join(sorted(split)))
    write_checkpoint("qwen3-tiny-8bit", qwen3_config(QWEN3_SIZES, {"group_size": 32, "bits": 8}), tensors, shards)


def mlx_random(rng, base, n_out, n_in, bits, gs):
    """A random quantized matrix of MLX's: words, scales (float16 in
    [0.004, 0.02]) and biases (float16, about -2^(bits-1)·scale), and its
    values dequantized as MLX computes them."""
    codes = rng.integers(0, 2 ** bits, size=(n_out, n_in)).astype(np.uint64)
    per = 32 // bits
    words = np.zeros((n_out, n_in // per), dtype=np.uint64)
    for j in range(per):
        words |= codes[:, j::per] << np.uint64(bits * j)
    scales = rng.uniform(0.004, 0.02, size=(n_out, n_in // gs)).astype(np.float16)
    biases = (-(2 ** (bits - 1)) * scales.astype(np.float32) * rng.uniform(0.9, 1.1, size=scales.shape)).astype(np.float16)
    s = np.repeat(scales.astype(np.float32), gs, axis=1)
    b = np.repeat(biases.astype(np.float32), gs, axis=1)
    values = ((s * codes.astype(np.float32)).astype(np.float32) + b).astype(np.float32)
    return [(base + ".weight", words.astype(np.uint32)), (base + ".scales", scales), (base + ".biases", biases)], values


def write_qwen3_4bit(rng):
    g = gguf_tensors("qwen3-tiny-f32.gguf")
    override = "model.layers.1.mlp.down_proj"
    tensors, reference = [], {}
    for base, w in qwen3_hf(g):
        if w.ndim == 1:
            tensors.append((base + ".weight", bf16_of(w)))
            continue
        bits, gs = (8, 32) if base == override else (4, 64)
        t, values = mlx_random(rng, base, w.shape[0], w.shape[1], bits, gs)
        tensors += t
        if base in ("model.layers.0.self_attn.q_proj", override):
            reference[base] = values[:2].tolist()
    q = {"group_size": 64, "bits": 4, override: {"group_size": 32, "bits": 8}}
    write_checkpoint("qwen3-tiny-4bit", qwen3_config(QWEN3_SIZES, q), sorted(tensors))
    out = os.path.join(HERE, "qwen3-tiny-4bit.edn")
    with open(out, "w") as fo:
        fo.write(";; qwen3-tiny-4bit's matrices' first two rows, dequantized as MLX computes them (generate.py)\n{")
        fo.write("\n ".join('"%s" %s' % (k, edn(v)) for k, v in sorted(reference.items())))
        fo.write("}\n")
    print("wrote %s" % os.path.relpath(out, ROOT))


# Qwen3.5: the tiny model of test/fixtures/gguf/generate.py
Q35 = {"VOCAB": 128, "HIDDEN": 32, "BLOCKS": 4, "FFN": 64, "HEADS": 4, "KV_HEADS": 2, "HEAD_DIM": 16, "ROPE_DIMS": 8,
       "K_HEADS": 2, "V_HEADS": 4, "STATE": 32, "CONV": 4, "EXPERTS": 4, "USED": 2, "EXPERT_FFN": 32, "SHARED_FFN": 32}

ATTENTION = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj"}
DENSE_FFN = {"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
SHARED = {"ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj",
          "ffn_down_shexp": "mlp.shared_expert.down_proj"}


def qwen35_config(moe, mtp, quantization=None):
    z = Q35
    text = {"model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text", "attention_bias": False,
            "attn_output_gate": True, "dtype": "float32", "full_attention_interval": 4, "head_dim": z["HEAD_DIM"],
            "hidden_act": "silu", "hidden_size": z["HIDDEN"],
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(z["BLOCKS"])],
            "linear_conv_kernel_dim": z["CONV"], "linear_key_head_dim": z["STATE"],
            "linear_num_key_heads": z["K_HEADS"], "linear_num_value_heads": z["V_HEADS"],
            "linear_value_head_dim": z["STATE"], "max_position_embeddings": 256, "mtp_num_hidden_layers": mtp,
            "num_attention_heads": z["HEADS"], "num_hidden_layers": z["BLOCKS"], "num_key_value_heads": z["KV_HEADS"],
            "rms_norm_eps": 1e-06, "tie_word_embeddings": True, "vocab_size": z["VOCAB"],
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                                "rope_theta": 10000, "partial_rotary_factor": z["ROPE_DIMS"] / z["HEAD_DIM"]}}
    if moe:
        text.update({"num_experts": z["EXPERTS"], "num_experts_per_tok": z["USED"],
                     "moe_intermediate_size": z["EXPERT_FFN"], "shared_expert_intermediate_size": z["SHARED_FFN"],
                     "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": []})
    else:
        text["intermediate_size"] = z["FFN"]
    c = {"architectures": ["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"],
         "model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": text, "tie_word_embeddings": True,
         "vision_config": {"depth": 1, "hidden_size": 8, "model_type": "qwen3_5"}}
    if quantization:
        c["quantization"] = quantization
        c["quantization_config"] = quantization
    return c


def qwen35_hf(g, moe, prefix, mlx):
    """[(base or name, weight)] of a Qwen3.5 checkpoint from a GGUF's tensors:
    Hugging Face's (mlx False) or mlx-lm's saved layout (mlx True)."""
    z = Q35
    n_k, r, dv = z["K_HEADS"], z["V_HEADS"] // z["K_HEADS"], z["STATE"]
    key_width = n_k * z["STATE"]
    heads = grouped(n_k, r)                     # value heads, tiled to grouped
    channels = expand(heads, dv)                # their values
    norm = (lambda w: w) if mlx else unshift
    out = [(prefix + "embed_tokens", g["token_embd.weight"]), (prefix + "norm", norm(g["output_norm.weight"]))]
    for i in range(z["BLOCKS"]):
        p, b = prefix + "layers.%d." % i, "blk.%d." % i
        out += [(p + "input_layernorm", norm(g[b + "attn_norm.weight"])),
                (p + "post_attention_layernorm", norm(g[b + "post_attention_norm.weight"]))]
        if (i + 1) % 4 == 0:
            out += [(p + v, g[b + k + ".weight"]) for k, v in ATTENTION.items()]
            out += [(p + "self_attn.q_norm", norm(g[b + "attn_q_norm.weight"])),
                    (p + "self_attn.k_norm", norm(g[b + "attn_k_norm.weight"]))]
        else:
            la = p + "linear_attn."
            qkv = g[b + "attn_qkv.weight"]
            rows = np.concatenate([np.arange(2 * key_width), 2 * key_width + channels])
            conv = g[b + "ssm_conv1d.weight"][rows]
            out += [(la + "in_proj_qkv", qkv.rows(rows) if isinstance(qkv, Q8) else qkv[rows]),
                    (la + "in_proj_z", rows_of(g[b + "attn_gate.weight"], channels)),
                    (la + "in_proj_b", rows_of(g[b + "ssm_beta.weight"], heads)),
                    (la + "in_proj_a", rows_of(g[b + "ssm_alpha.weight"], heads)),
                    (la + "A_log", a_log(g[b + "ssm_a"][heads])),
                    (la + "dt_bias", g[b + "ssm_dt.bias"][heads]),
                    (la + "conv1d", conv.reshape(conv.shape[0], conv.shape[1], 1) if mlx
                     else conv.reshape(conv.shape[0], 1, conv.shape[1])),
                    (la + "norm", g[b + "ssm_norm.weight"]),
                    (la + "out_proj", columns_of(g[b + "ssm_out.weight"], heads, dv))]
        if moe:
            out += [(p + "mlp.gate", g[b + "ffn_gate_inp.weight"])]
            for k, name in [("ffn_gate_exps", "gate_proj"), ("ffn_up_exps", "up_proj"), ("ffn_down_exps", "down_proj")]:
                stack = g[b + k + ".weight"]
                if mlx:
                    out.append((p + "mlp.switch_mlp." + name, stack))
                else:
                    out += [(p + "mlp.experts.%d.%s" % (e, name), stack[e]) for e in range(z["EXPERTS"])]
            out += [(p + v, g[b + k + ".weight"]) for k, v in SHARED.items()]
            out.append((p + "mlp.shared_expert_gate", g[b + "ffn_gate_inp_shexp.weight"].reshape(1, -1)))
        else:
            out += [(p + v, g[b + k + ".weight"]) for k, v in DENSE_FFN.items()]
    return out


def rows_of(w, perm):
    return w.rows(perm) if isinstance(w, Q8) else w[perm]


def columns_of(w, heads, dv):
    if isinstance(w, Q8):
        assert dv % 32 == 0
        return w.columns(expand(heads, dv // 32))
    return w[:, expand(heads, dv)]


def named(ws):
    """[(name, array)]: a 1-d or [C 1 K] / [C K 1] tensor, A_log and
    dt_bias under their own names, every other one base.weight (or its
    triple)."""
    out = []
    for base, w in ws:
        if base.endswith(("A_log", "dt_bias")):
            out.append((base, w))
        else:
            out += matrix(base, w)
    return out


def mtp_block(g):
    """The MTP block of qwen35-tiny-f32.gguf (blk.4) under Hugging Face's
    names (mtp.*): read by neither forward."""
    b, p = "blk.4.", "mtp.layers.0."
    out = [(p + "input_layernorm", unshift(g[b + "attn_norm.weight"])),
           (p + "post_attention_layernorm", unshift(g[b + "post_attention_norm.weight"]))]
    out += [(p + v, g[b + k + ".weight"]) for k, v in ATTENTION.items()]
    out += [(p + "self_attn.q_norm", unshift(g[b + "attn_q_norm.weight"])),
            (p + "self_attn.k_norm", unshift(g[b + "attn_k_norm.weight"]))]
    out += [(p + v, g[b + k + ".weight"]) for k, v in DENSE_FFN.items()]
    out += [("mtp.fc", g[b + "nextn.eh_proj.weight"]),
            ("mtp.pre_fc_norm_embedding", unshift(g[b + "nextn.enorm.weight"])),
            ("mtp.pre_fc_norm_hidden", unshift(g[b + "nextn.hnorm.weight"])),
            ("mtp.norm", unshift(g[b + "nextn.shared_head_norm.weight"]))]
    return out


def write_qwen35():
    g = gguf_tensors("qwen35-tiny-f32.gguf")
    hf = named(qwen35_hf(g, False, "model.language_model.", False)) + named(mtp_block(g))
    hf.append(("model.visual.patch_embed.proj.weight", np.arange(8, dtype=np.int8)))
    write_checkpoint("qwen35-tiny", qwen35_config(False, 1), sorted(hf))
    q = gguf_tensors("qwen35-tiny-q8_0.gguf")
    write_checkpoint("qwen35-tiny-mlx-8bit", qwen35_config(False, 1, {"group_size": 32, "bits": 8}),
                     sorted(named(qwen35_hf(q, False, "language_model.model.", True))))
    g = gguf_tensors("qwen35moe-tiny-f32.gguf")
    write_checkpoint("qwen35moe-tiny", qwen35_config(True, 0), sorted(named(qwen35_hf(g, True, "model.language_model.", False))))
    q = gguf_tensors("qwen35moe-tiny-q8_0.gguf")
    write_checkpoint("qwen35moe-tiny-mlx-8bit", qwen35_config(True, 0, {"group_size": 32, "bits": 8}),
                     sorted(named(qwen35_hf(q, True, "language_model.model.", True))))


def main():
    write_types(np.random.default_rng(11))
    write_qwen3()
    write_qwen3_4bit(np.random.default_rng(12))
    write_qwen35()


if __name__ == "__main__":
    main()
