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
# Usage: native/build-mlx.sh [cpu|cuda13]
#   macOS arm64: the mlx-metal wheel (CPU and Metal); the argument is ignored.
#   Linux x86_64 / aarch64: cpu (default) uses the mlx-cpu wheel; cuda13 the
#     mlx-cuda-13 wheel (needs its CUDA runtime libraries; see
#     docs/research/numerics-backends.md). Untested so far.
#
# Requires: curl, unzip, cmake (≥ 3.16), a C++20 compiler, shasum or sha256sum.
set -euo pipefail

MLX_VERSION=0.32.2
MLX_C_TAG=v0.7.0
MLX_C_COMMIT=a341b4925024b88b2c593468f16e12f5e17315da

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
      *) echo "unknown variant: $variant (cpu or cuda13)" >&2; exit 1 ;;
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

# --- the MLX wheel ------------------------------------------------------------
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
echo "built $out/lib:"
ls "$out/lib"
