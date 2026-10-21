#!/usr/bin/env bash
# Builds llama.cpp's shared libraries, pinned to a release, for the llama.cpp
# engine (gen.llm.llama). The result is native/build/llama/<platform>:
#   lib/      libllama, and ggml's libraries it links (libggml, libggml-base,
#             the CPU backend, and Metal and BLAS on macOS or CUDA on Linux)
#   include/  llama.h and ggml's headers, from which native/jextract-llama.sh
#             generates the Java binding (src-java/gen/llama/ffi)
#
# The Java binding is generated from this version's llama.h, and passes
# structs by value whose layouts change between releases, so the two must
# match exactly: gen.llama.LlamaRuntime refuses a library whose
# llama_version() is not LLAMA_VERSION below.
#
# Usage: native/build-llama.sh [cpu|cuda]
#   macOS arm64: Metal (and the CPU, with Accelerate); the argument is
#     ignored. Tested on an M2 Max, macOS 27 (2026-10-20).
#   Linux x86_64 / aarch64: cpu (default), or cuda (GGML_CUDA; needs the CUDA
#     toolkit, and builds for CMAKE_CUDA_ARCHITECTURES, default this
#     machine's GPU). Neither is tested yet.
#   The JVM finds the libraries with -Dgen.llama.lib=native/build/llama/<platform>/lib
#   (default: this platform's, under the working directory).
#   LLAMA_CPP_REPO clones from another place than GitHub (a local clone, say);
#   the commit is checked either way.
#
# Requires: git, cmake (≥ 3.14; Ninja if installed, else make), a C/C++17
# compiler.
set -euo pipefail

LLAMA_VERSION=0.7.0
LLAMA_TAG=v$LLAMA_VERSION
LLAMA_COMMIT=620d2225cc436ab0b001e3200e590e3b286ef9ab

here="$(cd "$(dirname "$0")" && pwd)"
if command -v ninja >/dev/null 2>&1; then generator="-G Ninja"; else generator=""; fi
os="$(uname -s)"
arch="$(uname -m)"
variant="${1:-cpu}"

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
# a release build, whose llama_version() is "0.7.0" (a dev build's is
# "0.7.0-dev", as on every commit after the tag)
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
