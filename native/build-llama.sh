#!/usr/bin/env bash
# Builds llama.cpp's shared libraries, pinned to a release, for the llama.cpp
# engine (gen.llm.llama). The result is native/build/llama/<platform>:
#   lib/      libllama, and ggml's libraries it links (libggml, libggml-base,
#             the CPU backend, and Metal and BLAS on macOS or CUDA on Linux)
#   include/  llama.h and ggml's headers, from which native/jextract-llama.sh
#             generates the Java binding (src-java/gen/llama/ffi)
#   bin/      with --bench: llama-bench, llama.cpp's own benchmark, the
#             reference for gen.qwen3-bench's -Dgen.bench.llama-bench
#
# The Java binding is generated from this version's llama.h, and passes
# structs by value whose layouts change between releases, so the two must
# match exactly: gen.llama.LlamaRuntime refuses a library whose
# llama_version() is not LLAMA_VERSION below.
#
# Usage: native/build-llama.sh [cpu|cuda] [--bench]
#   macOS arm64: Metal (and the CPU, with Accelerate); cpu or cuda is
#     ignored. Tested on an M2 Max, macOS 27 (2026-10-20).
#   Linux x86_64 / aarch64: cpu (default), or cuda (GGML_CUDA; needs the CUDA
#     toolkit, and builds for CMAKE_CUDA_ARCHITECTURES, default this
#     machine's GPU). Both tested on x86_64 with CUDA 13.3 and an RTX PRO
#     6000 (sm_120), 2026-10-21. A toolkit installed from pip lacks CCCL's
#     headers (cub, thrust): give them as CUDAFLAGS="-isystem <cccl>/include"
#     (MLX's CUDA build fetches CCCL into its cmake directory, _deps/cccl-src).
#     The JVM then needs the toolkit's libraries on LD_LIBRARY_PATH (pip's
#     nvidia/cu13/lib) when they are not on the system's loader path.
#   The JVM finds the libraries with -Dgen.llama.lib=native/build/llama/<platform>/lib
#   (default: this platform's, under the working directory).
#   LLAMA_CPP_REPO clones from another place than GitHub (a local clone, say);
#   the commit is checked either way.
#   --bench also builds llama-bench (bin/llama-bench) from the same source
#     with the same backend, in a build directory of its own with llama.cpp's
#     common library and tools on (LLAMA_BUILD_COMMON, LLAMA_BUILD_TOOLS),
#     linked statically, so that it is one file and the libraries above are
#     built as without it. Tested on the M4, macOS 27 (2026-11-08).
#
# Requires: git, cmake (≥ 3.14; Ninja if installed, else make), a C/C++17
# compiler.
set -euo pipefail

LLAMA_VERSION=0.9.0
LLAMA_TAG=v$LLAMA_VERSION
LLAMA_COMMIT=cd959bae99ff4bed22bef2604f45241c09bbb8c8

here="$(cd "$(dirname "$0")" && pwd)"
if command -v ninja >/dev/null 2>&1; then generator="-G Ninja"; else generator=""; fi
os="$(uname -s)"
arch="$(uname -m)"
variant=cpu
bench=0
for a in "$@"; do
  case "$a" in
    cpu|cuda) variant="$a" ;;
    --bench) bench=1 ;;
    *) echo "unknown argument: $a (cpu, cuda or --bench)" >&2; exit 1 ;;
  esac
done

case "$os-$arch" in
  Darwin-arm64)
    platform=macos-arm64
    backend="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
    rpath='@loader_path'
    ;;
  Linux-x86_64|Linux-aarch64)
    case "$variant" in
      cpu) backend="-DGGML_METAL=OFF" ;;
      cuda) backend="-DGGML_CUDA=ON${CMAKE_CUDA_ARCHITECTURES:+ -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES}" ;;
      *) echo "unknown variant: $variant (cpu or cuda)" >&2; exit 1 ;;
    esac
    platform=linux-$arch-$variant
    rpath='$ORIGIN'
    ;;
  *)
    echo "unsupported platform: $os-$arch" >&2
    exit 1
    ;;
esac

build="$here/build"
out="$build/llama/$platform"
src="$build/src/llama.cpp"
mkdir -p "$build/src"

if [ ! -d "$src/.git" ]; then
  git clone -q --depth 1 --branch "$LLAMA_TAG" "${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}" "$src"
fi
commit="$(git -C "$src" rev-parse HEAD)"
if [ "$commit" != "$LLAMA_COMMIT" ]; then
  echo "llama.cpp at $commit, expected $LLAMA_COMMIT ($LLAMA_TAG)" >&2
  exit 1
fi

# only the library: no common utilities, tools, examples, server or tests;
# a release build, whose llama_version() is "0.9.0" (a dev build's is
# "0.9.0-dev", as on every commit after the tag)
cmake -S "$src" -B "$build/cmake-llama-$platform" $generator \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_IS_DEV=OFF \
  -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_OPENSSL=OFF \
  $backend \
  -DCMAKE_INSTALL_RPATH="$rpath" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_PREFIX="$out"
cmake --build "$build/cmake-llama-$platform" --config Release
cmake --install "$build/cmake-llama-$platform" --config Release

echo "built $out/lib:"
ls "$out/lib"

if [ "$bench" = 1 ]; then
  # llama-bench alone of the tools; its libraries static, the backend the
  # library's
  cmake -S "$src" -B "$build/cmake-llama-bench-$platform" $generator \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_IS_DEV=OFF \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_TOOLS_INSTALL=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_OPENSSL=OFF \
    $backend
  cmake --build "$build/cmake-llama-bench-$platform" --config Release --target llama-bench
  mkdir -p "$out/bin"
  cp "$build/cmake-llama-bench-$platform/bin/llama-bench" "$out/bin/llama-bench"
  echo "built $out/bin/llama-bench"
fi
