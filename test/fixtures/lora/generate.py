"""LoRA adapters for Qwen3-0.6B, as GGUF files, for the tests of
gen.llm.llama's adapters (test/gen/llm/llama_lora_test.clj).

Run with any Python that has numpy (e.g. GenMLX's venv), from the
repository's root, with llama.cpp's gguf-py (native/build-llama.sh's
checkout) and the base model's GGUF, whose header gives the shapes:

    ~/code/genmlx/.venv/bin/python -I test/fixtures/lora/generate.py \
        [~/gen-clj-models/Qwen3-0.6B-Q8_0.gguf [native/build/src/llama.cpp/gguf-py]]

The files, as llama.cpp's convert_lora_to_gguf.py writes them
(general.type adapter, general.architecture qwen3, adapter.type lora,
adapter.lora.alpha; tensors blk.N.attn_q.weight.lora_a and .lora_b, F32):

- qwen3-0.6b-lora.gguf: rank 2 on attn_q and attn_v of layers 0, 7, 14
  and 21, alpha 4 (so llama.cpp scales B A by 4 / 2 = 2 at scale 1). A's
  rows are random directions of the layer's input (entries N(0, 1/n_in)),
  B's entries N(0, 0.25^2): at scale 1, B A's two singular values are
  about 2 * 0.25 * sqrt(n_out), 16 for attn_v and 23 for attn_q, against
  the base weights' largest of 2 to 10, along directions that carry about
  1/32 of a hidden state's norm. A large change, by design, so that a
  test sees it far outside llama.cpp's float noise.
- qwen3-0.6b-lora-zero.gguf: the same A, B = 0. An adapter that changes
  nothing: llama.cpp computes B (A x) and adds it, exactly zero.
- qwen3-0.6b-alora.gguf: an activated LoRA (aLoRA): rank 1 on layer 0's
  attn_q, B = 0, with adapter.alora.invocation_tokens [151644, 77091, 198]
  (Qwen3's "<|im_start|>assistant\\n").

Fixed seed: the same files each run (numpy's PCG64 and its normal
sampler). gguf-py's metadata module imports yaml for model cards, which
this script never reads; it is stubbed when yaml is not installed.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/gen-clj-models/Qwen3-0.6B-Q8_0.gguf")
GGUF_PY = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "native/build/src/llama.cpp/gguf-py")

try:
    import yaml  # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

RANK = 2
ALPHA = 4.0
LAYERS = [0, 7, 14, 21]
KINDS = ["attn_q", "attn_v"]
B_SD = 0.25
INVOCATION = [151644, 77091, 198]


def base_shapes():
    """{tensor name: (n_out, n_in)} of the base model, and its architecture."""
    r = gguf.GGUFReader(MODEL)
    arch = r.fields["general.architecture"].contents()
    # ggml's ne[0] is a weight's input size, ne[1] its output size
    return {t.name: (int(t.shape[1]), int(t.shape[0])) for t in r.tensors if len(t.shape) == 2}, arch


def write(name, arch, tensors, extra=None):
    path = os.path.join(HERE, name)
    w = gguf.GGUFWriter(path, arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_name("gen.clj test adapter: " + name)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, ALPHA)
    if extra:
        extra(w)
    for tname, arr in tensors:
        w.add_tensor(tname, np.ascontiguousarray(arr, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s (%d bytes)" % (os.path.relpath(path, ROOT), os.path.getsize(path)))


def main():
    shapes, arch = base_shapes()
    if arch != "qwen3":
        raise SystemExit("the base model's architecture is %r, not qwen3" % arch)
    rng = np.random.default_rng(20261107)
    lora, zero = [], []
    for layer in LAYERS:
        for kind in KINDS:
            base = "blk.%d.%s.weight" % (layer, kind)
            n_out, n_in = shapes[base]
            # numpy's shapes are ggml's reversed: lora_a has ne [n_in, rank],
            # lora_b [rank, n_out], as llama.cpp checks against the base
            a = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(RANK, n_in))
            b = rng.normal(0.0, B_SD, size=(n_out, RANK))
            lora += [(base + ".lora_a", a), (base + ".lora_b", b)]
            zero += [(base + ".lora_a", a), (base + ".lora_b", np.zeros_like(b))]
    write("qwen3-0.6b-lora.gguf", arch, lora)
    write("qwen3-0.6b-lora-zero.gguf", arch, zero)
    base = "blk.0.attn_q.weight"
    n_out, n_in = shapes[base]
    write("qwen3-0.6b-alora.gguf", arch,
          [(base + ".lora_a", rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(1, n_in))),
           (base + ".lora_b", np.zeros((n_out, 1)))],
          extra=lambda w: w.add_key_value(gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS, INVOCATION,
                                          gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32))


if __name__ == "__main__":
    main()
