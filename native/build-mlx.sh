#!/usr/bin/env bash
# Builds libmlxc (MLX's C API) against a pinned, prebuilt MLX, for the MLX
# backend (gen.backend.mlx). The result is native/build/<os>-<arch>/lib:
#   libmlxc   mlx-c, built here
#   libmlx    MLX, from the official wheel of the matching version
#   (macOS) libjaccl and mlx.metallib, which libmlx loads from its own directory
#
# mlx-c depends on MLX's C++ ABI, so the two must match exactly: mlx-c v0.7.0
# pins MLX v0.32.2, and this script takes MLX from that version's wheel on
# PyPI (checked against its sha256). Building MLX itself is not needed, so no
# Metal compiler is either.
#
# Usage: native/build-mlx.sh [cpu|cuda13|cuda13-source]
#   macOS arm64: the mlx-metal wheel (CPU and Metal); the argument is ignored.
#   Linux x86_64 / aarch64: cpu (default) uses the mlx-cpu wheel; cuda13 the
#     mlx-cuda-13 wheel, with the system's CUDA libraries where it has them
#     and the missing ones fetched from PyPI (below). Tested on aarch64 (the
#     Jetson AGX Thor, Ubuntu 24.04, CUDA 13.2, 2026-10-19); x86_64 not yet.
#     cuda13-source builds MLX itself from its pinned source, for this
#     machine's GPU (nvidia-smi's compute capability, or MLX_CUDA_ARCHITECTURES):
#     the wheel's kernels cover sm_75 to sm_121 but not every GPU, and the
#     Thor's sm_110 has neither code nor compatible PTX in it. About six
#     minutes on the Thor; needs the CUDA toolkit, cuDNN, BLAS and LAPACK.
#     The JVM finds it with -Dgen.mlx.lib=native/build/linux-<arch>-<variant>/lib.
#
# Requires: curl, unzip, cmake (≥ 3.16), a C++20 compiler, shasum or sha256sum.
set -euo pipefail

MLX_VERSION=0.32.2
MLX_C_TAG=v0.7.0
MLX_C_COMMIT=a341b4925024b88b2c593468f16e12f5e17315da
MLX_COMMIT=1f8e74e3f12f31365464a6867c6579f0e9b29d85

here="$(cd "$(dirname "$0")" && pwd)"
os="$(uname -s)"
arch="$(uname -m)"
variant="${1:-cpu}"

# wheel file name and sha256, by platform (from https://pypi.org/pypi/<name>/0.32.2/json)
case "$os-$arch" in
  Darwin-arm64)
    platform=macos-arm64
    major="$(sw_vers -productVersion | cut -d. -f1)"
    if [ "$major" -ge 26 ]; then
      wheel=mlx_metal-$MLX_VERSION-py3-none-macosx_26_0_arm64.whl
      sha=e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab
    elif [ "$major" -ge 15 ]; then
      wheel=mlx_metal-$MLX_VERSION-py3-none-macosx_15_0_arm64.whl
      sha=55a369250d220b2cf10213a87a2ac1b1a420608c5b35b1df4e7147ac8e32f121
    else
      wheel=mlx_metal-$MLX_VERSION-py3-none-macosx_14_0_arm64.whl
      sha=3825fff379dbc107dd3413e564a06caeaa24819910ec49c0439e454c06a1b9b8
    fi
    package=mlx-metal
    ;;
  Linux-x86_64|Linux-aarch64)
    platform=linux-$arch
    case "$variant" in
      cpu) package=mlx-cpu ;;
      cuda13) package=mlx-cuda-13 ;;
      cuda13-source) package="" ;;
      *) echo "unknown variant: $variant (cpu, cuda13 or cuda13-source)" >&2; exit 1 ;;
    esac
    platform=$platform-$variant
    wheel="" # looked up below: Linux wheel names carry a manylinux tag
    sha=""
    ;;
  *)
    echo "unsupported platform: $os-$arch" >&2
    exit 1
    ;;
esac

build="$here/build"
out="$build/$platform"
mkdir -p "$build/dl" "$build/src"

sha256() {
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi
}

# --- MLX: built from source, or the wheel ---------------------------------------
if [ "$variant" = cuda13-source ]; then
  msrc="$build/src/mlx"
  if [ ! -d "$msrc/.git" ]; then
    git clone -q --depth 1 --branch "v$MLX_VERSION" https://github.com/ml-explore/mlx "$msrc"
  fi
  [ "$(git -C "$msrc" rev-parse HEAD)" = "$MLX_COMMIT" ] ||
    { echo "mlx at $(git -C "$msrc" rev-parse HEAD), expected $MLX_COMMIT (v$MLX_VERSION)" >&2; exit 1; }
  archs="${MLX_CUDA_ARCHITECTURES:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')}"
  [ -n "$archs" ] || { echo "no GPU found: set MLX_CUDA_ARCHITECTURES" >&2; exit 1; }
  echo "building MLX v$MLX_VERSION for CUDA architecture $archs"
  cmake -S "$msrc" -B "$build/cmake-mlx-$platform" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DMLX_BUILD_CUDA=ON -DMLX_CUDA_ARCHITECTURES="$archs" -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_INSTALL_PREFIX="$build/mlx-$platform"
  cmake --build "$build/cmake-mlx-$platform"
  cmake --install "$build/cmake-mlx-$platform"
  mlx_root="$build/mlx-$platform"
else
if [ -z "$wheel" ]; then
  # Linux: find the wheel for this architecture in PyPI's JSON, with its sha256
  read -r wheel sha url < <(curl -sSf "https://pypi.org/pypi/$package/$MLX_VERSION/json" |
    python3 -I -c "
import json, sys
arch = sys.argv[1]
for f in json.load(sys.stdin)['urls']:
    if f['filename'].endswith(arch + '.whl'):
        print(f['filename'], f['digests']['sha256'], f['url']); break
" "$arch")
  [ -n "$wheel" ] || { echo "no $package $MLX_VERSION wheel for $arch" >&2; exit 1; }
else
  url="$(curl -sSf "https://pypi.org/pypi/$package/$MLX_VERSION/json" |
    python3 -I -c "
import json, sys
for f in json.load(sys.stdin)['urls']:
    if f['filename'] == sys.argv[1]: print(f['url'])
" "$wheel")"
fi
if [ ! -f "$build/dl/$wheel" ]; then
  echo "downloading $wheel"
  curl -sSfL -o "$build/dl/$wheel.part" "$url"
  mv "$build/dl/$wheel.part" "$build/dl/$wheel"
fi
actual="$(sha256 "$build/dl/$wheel")"
if [ "$actual" != "$sha" ]; then
  echo "sha256 mismatch for $wheel: $actual, expected $sha" >&2
  exit 1
fi
rm -rf "$build/wheel-$platform"
mkdir -p "$build/wheel-$platform"
unzip -q "$build/dl/$wheel" -d "$build/wheel-$platform"
mlx_root="$build/wheel-$platform/mlx"
fi

# --- mlx-c --------------------------------------------------------------------
src="$build/src/mlx-c"
if [ ! -d "$src/.git" ]; then
  git clone -q --depth 1 --branch "$MLX_C_TAG" https://github.com/ml-explore/mlx-c "$src"
fi
commit="$(git -C "$src" rev-parse HEAD)"
if [ "$commit" != "$MLX_C_COMMIT" ]; then
  echo "mlx-c at $commit, expected $MLX_C_COMMIT ($MLX_C_TAG)" >&2
  exit 1
fi

if [ "$os" = Darwin ]; then rpath='@loader_path'; else rpath='$ORIGIN'; fi
cmake -S "$src" -B "$build/cmake-$platform" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_DIR="$mlx_root/share/cmake/MLX" \
  -DCMAKE_INSTALL_RPATH="$rpath" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_PREFIX="$out"
cmake --build "$build/cmake-$platform"
cmake --install "$build/cmake-$platform"

# libmlx and what it loads from its own directory, next to libmlxc
for f in "$mlx_root"/lib/*; do
  [ -f "$f" ] && cp -f "$f" "$out/lib/"
done
# CUDA: MLX compiles some kernels at run time with NVRTC, from headers it looks
# for at $ORIGIN/../include (its own and CCCL's, in include/cccl): copy them
# next to libmlxc's own
if [ "$variant" = cuda13 ] || [ "$variant" = cuda13-source ]; then
  [ -d "$mlx_root/include" ] && cp -R "$mlx_root/include/." "$out/include/"
fi
# Linux wheels bundle their own dependencies (BLAS, LAPACK, CUDA support) in
# <package>.libs, which libmlx's RPATH finds at $ORIGIN/../../<package>.libs:
# keep that layout relative to $out/lib, rather than editing the binary
for d in "$build/wheel-$platform"/*.libs; do
  [ -d "$d" ] || continue
  rm -rf "$build/$(basename "$d")"
  cp -R "$d" "$build/"
done

# CUDA: libmlx links NVIDIA's libraries, which its RPATH finds at
# $ORIGIN/../../nvidia/<name>/lib (pip's layout). The system's are used
# where it has them (a CUDA or JetPack install matches its driver); only the
# missing ones are fetched, from their PyPI wheels at pinned versions (those
# current with MLX 0.32.2's requirements, 2026-10-19).
fetch_nvidia() {
  local pkg=$1 ver=$2 w s u
  read -r w s u < <(curl -sSf "https://pypi.org/pypi/$pkg/$ver/json" |
    python3 -I -c "
import json, sys
fs = [f for f in json.load(sys.stdin)['urls'] if f['filename'].endswith('.whl') and '$arch' in f['filename']]
print(fs[0]['filename'], fs[0]['digests']['sha256'], fs[0]['url']) if fs else None
")
  [ -n "${w:-}" ] || { echo "no $pkg $ver wheel for $arch" >&2; exit 1; }
  if [ ! -f "$build/dl/$w" ]; then
    echo "downloading $w"
    curl -sSfL -o "$build/dl/$w.part" "$u" && mv "$build/dl/$w.part" "$build/dl/$w"
  fi
  [ "$(sha256 "$build/dl/$w")" = "$s" ] || { echo "sha256 mismatch for $w" >&2; exit 1; }
  unzip -q -o "$build/dl/$w" 'nvidia/*' -d "$build"
}
if [ "$variant" = cuda13 ] || [ "$variant" = cuda13-source ]; then
  for lib in $(ldd "$out/lib/libmlx.so" | awk '/not found/ {print $1}'); do
    case "$lib" in
      libnccl.so.*) fetch_nvidia nvidia-nccl-cu13 2.32.3 ;;
      libcudnn*) fetch_nvidia nvidia-cudnn-cu13 9.27.0.42 ;;
      libcublas*) fetch_nvidia nvidia-cublas 13.8.1.7 ;;
      libcufft*) fetch_nvidia nvidia-cufft 12.4.0.43 ;;
      libnvrtc*) fetch_nvidia nvidia-cuda-nvrtc 13.4.92 ;;
      libcudart*) fetch_nvidia nvidia-cuda-runtime 13.4.92 ;;
      libcuda.so.*) echo "libcuda.so.1 not found: the NVIDIA driver is missing" >&2; exit 1 ;;
      *) echo "libmlx needs $lib, which this script does not know how to fetch" >&2; exit 1 ;;
    esac
  done
  missing="$(ldd "$out/lib/libmlx.so" | awk '/not found/ {print $1}')"
  [ -z "$missing" ] || { echo "still missing: $missing" >&2; exit 1; }
fi
echo "built $out/lib:"
ls "$out/lib"
