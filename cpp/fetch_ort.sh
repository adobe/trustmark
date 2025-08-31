#!/usr/bin/env bash

set -euo pipefail

# Fetch and stage ONNX Runtime locally under cpp/third_party/ort
# Usage:
#   bash cpp/fetch_ort.sh            # installs default version
#   bash cpp/fetch_ort.sh 1.19.2     # installs specified version

VER="${1:-1.19.2}"

# Resolve script directory to allow invocation from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

PKG=""
SRC_DIR=""

case "$OS_NAME" in
  Darwin)
    # Try universal2 first (newer naming), fallback to universal
    CANDIDATES=(
      "onnxruntime-osx-universal2-${VER}.tgz|onnxruntime-osx-universal2-${VER}"
      "onnxruntime-osx-universal-${VER}.tgz|onnxruntime-osx-universal-${VER}"
    )
    ;;
  Linux)
    case "$ARCH_NAME" in
      x86_64|amd64)
        CANDIDATES=("onnxruntime-linux-x64-${VER}.tgz|onnxruntime-linux-x64-${VER}")
        ;;
      aarch64|arm64)
        CANDIDATES=("onnxruntime-linux-aarch64-${VER}.tgz|onnxruntime-linux-aarch64-${VER}")
        ;;
      *)
        echo "Unsupported Linux arch: $ARCH_NAME" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS_NAME" >&2
    exit 1
    ;;
esac

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

download_and_extract() {
  local pkg_name="$1"
  local src_dir_name="$2"
  local url="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${pkg_name}"
  echo "Attempting: $url"
  if curl -L --fail -o "$TMP_DIR/$pkg_name" "$url"; then
    tar -xzf "$TMP_DIR/$pkg_name" -C "$TMP_DIR"
    PKG="$pkg_name"
    SRC_DIR="$src_dir_name"
    return 0
  fi
  return 1
}

success=0
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r candidate_pkg candidate_src <<< "$entry"
  if download_and_extract "$candidate_pkg" "$candidate_src"; then
    success=1
    break
  fi
done

if [[ "$success" -ne 1 ]]; then
  echo "Failed to download ONNX Runtime ${VER} for ${OS_NAME}-${ARCH_NAME}" >&2
  exit 1
fi

mkdir -p third_party/ort/include third_party/ort/lib
rsync -a "$TMP_DIR/$SRC_DIR/include/" third_party/ort/include/
rsync -a "$TMP_DIR/$SRC_DIR/lib/" third_party/ort/lib/

if ! ls third_party/ort/lib/libonnxruntime.* >/dev/null 2>&1; then
  echo "libonnxruntime.* not found after extraction" >&2
  exit 1
fi

echo "ONNX Runtime ${VER} staged to cpp/third_party/ort"

