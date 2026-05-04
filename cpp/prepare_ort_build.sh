#!/usr/bin/env bash

# Copies TrustMark-specific source files into onnxruntime-wasi before building.
#
# These files live in the trustmark repo (not onnxruntime) and must be staged
# into the correct onnxruntime/wasm/ location before running build_wasi.sh.
#
# Usage: ./prepare_ort_build.sh [/path/to/onnxruntime-wasi]
#   Default onnxruntime-wasi path: ./onnxruntime-wasi (symlink or directory)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_DIR="${1:-$SCRIPT_DIR/onnxruntime-wasi}"
ORT_WASM="$ORT_DIR/onnxruntime/wasm"

if [[ ! -d "$ORT_WASM" ]]; then
    echo "Error: $ORT_WASM does not exist." >&2
    echo "Make sure onnxruntime-wasi is cloned or symlinked." >&2
    exit 1
fi

echo "Copying TrustMark sources into $ORT_WASM ..."

# Image utilities (general WASM image I/O, stb-based)
cp "$SCRIPT_DIR/wasm/image_utils.cpp"       "$ORT_WASM/image_utils.cpp"
cp "$SCRIPT_DIR/wasm/image_utils.h"         "$ORT_WASM/image_utils.h"
cp "$SCRIPT_DIR/wasm/stb_image.h"           "$ORT_WASM/stb_image.h"
cp "$SCRIPT_DIR/wasm/stb_image_resize2.h"   "$ORT_WASM/stb_image_resize2.h"
cp "$SCRIPT_DIR/wasm/stb_image_write.h"     "$ORT_WASM/stb_image_write.h"

# Entry point: trustmark_wasm_image.cpp -> simple.cpp
# Fix include path: "../wasm/image_utils.h" -> "image_utils.h" (same dir after copy)
sed 's|#include "\.\./wasm/image_utils\.h"|#include "image_utils.h"|' \
    "$SCRIPT_DIR/examples/trustmark_wasm_image.cpp" > "$ORT_WASM/simple.cpp"

echo "Done. Files staged in $ORT_WASM:"
echo "  image_utils.cpp, image_utils.h"
echo "  stb_image.h, stb_image_resize2.h, stb_image_write.h"
echo "  simple.cpp (from examples/trustmark_wasm_image.cpp)"
echo ""
echo "Now run build_wasi.sh from $ORT_DIR"
