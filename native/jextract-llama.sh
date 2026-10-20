#!/usr/bin/env bash
# Generates the Java binding of llama.cpp (src-java/gen/llama/ffi) from the
# llama.h of native/build-llama.sh's build, with jextract, filtered to the
# symbols gen.llm.llama uses. The generated sources are committed, as the MLX
# binding's are; regenerate them when the pinned llama.cpp release changes
# (and LlamaRuntime.VERSION with it), then compile: clojure -T:build java.
#
# The binding names no library (no --library): its symbols are looked up in
# the libraries its class loader has loaded, and gen.llama.LlamaRuntime loads
# libllama (gen.llama.lib) before any call.
#
# The struct layouts jextract computes are those of the platform it runs on.
# The structs here hold only fixed-width integers, floats, bools, enums,
# size_t and pointers, so one generation serves every LP64 platform (macOS
# and Linux, arm64 and x86_64), not Windows.
#
# Usage: native/jextract-llama.sh [include-dir]
#   include-dir defaults to native/build/llama/<platform>/include. jextract
#   25 is $JEXTRACT, or jextract on the PATH (https://jdk.java.net/jextract/).
#   Generated 2026-10-20 with jextract 25 (JDK 25+37, libclang 13) on macOS
#   arm64, from llama.cpp v0.7.0.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=macos-arm64 ;;
  Linux-*) platform=linux-$(uname -m)-cpu ;;
  *) echo "unsupported platform" >&2; exit 1 ;;
esac
include="${1:-$here/build/llama/$platform/include}"
jextract="${JEXTRACT:-jextract}"
out="$root/src-java"

# The functions gen.llm.llama calls. Adding one here and regenerating is how
# the binding grows.
functions="
  llama_version llama_backend_init llama_log_set llama_print_system_info
  llama_supports_gpu_offload llama_max_parallel_sequences
  llama_model_default_params llama_context_default_params
  llama_model_load_from_file llama_model_free llama_init_from_model llama_free
  llama_n_ctx_seq llama_n_batch llama_n_seq_max
  llama_get_memory llama_model_get_vocab llama_model_desc llama_model_size
  llama_model_n_params llama_model_n_ctx_train llama_model_n_swa
  llama_model_has_encoder llama_model_has_decoder llama_model_is_recurrent
  llama_model_is_hybrid llama_model_is_diffusion
  llama_vocab_n_tokens llama_vocab_bos llama_vocab_eos llama_vocab_is_eog
  llama_vocab_get_add_bos
  llama_tokenize llama_token_to_piece llama_detokenize
  llama_batch_init llama_batch_free llama_decode llama_get_logits_ith
  llama_memory_seq_rm llama_memory_seq_cp
  llama_memory_seq_pos_min llama_memory_seq_pos_max"
structs="llama_model_params llama_context_params llama_batch"
typedefs="llama_token llama_pos llama_seq_id llama_memory_t"

args=()
for f in $functions; do args+=(--include-function "$f"); done
for s in $structs; do args+=(--include-struct "$s"); done
for t in $typedefs; do args+=(--include-typedef "$t"); done

rm -rf "$out/gen/llama/ffi"
# (without the macOS SDK's nullability warnings, which are about its own
# headers)
"$jextract" --target-package gen.llama.ffi --header-class-name Llama \
  -I "$include" --output "$out" "${args[@]}" "$include/llama.h" 2>&1 |
  { grep -v -e 'nullability type specifier' -e '^WARNING: Skipping' || true; }
# a function jextract left out would surface only when gen.llm.llama loads:
# make it loud here
for f in $functions; do
  grep -q "public static .* $f(" "$out/gen/llama/ffi/Llama.java" ||
    { echo "jextract did not emit $f" >&2; exit 1; }
done
echo "generated $(find "$out/gen/llama/ffi" -name '*.java' | wc -l | tr -d ' ') files in src-java/gen/llama/ffi" \
  "($(echo $functions | wc -w | tr -d ' ') functions)"
