#!/usr/bin/env bash

set -euo pipefail

# Fetch TrustMark ONNX models from CDN
# Usage: ./fetch_models.sh [model_type]
#   model_type: P, Q, B, C, or "all" (default)

MODEL_TYPE="${1:-all}"
MODEL_ROOT="https://cc-assets.netlify.app/watermarking/trustmark-models"

# Resolve script directory to allow invocation from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create models directory if it doesn't exist
mkdir -p models

fetch_model() {
    local filename="$1"
    local url="${MODEL_ROOT}/${filename}"
    local output="models/${filename}"

    if [[ -f "$output" ]]; then
        echo "? ${filename} already exists, skipping..."
        return 0
    fi

    echo "Downloading ${filename}..."
    if curl -L --fail -o "$output" "$url"; then
        echo "? ${filename} downloaded successfully"
        return 0
    else
        echo "? Failed to download ${filename}" >&2
        return 1
    fi
}

case "$MODEL_TYPE" in
    P|p)
        echo "Fetching P variant models..."
        fetch_model "encoder_P.onnx"
        fetch_model "decoder_P.onnx"
        ;;
    Q|q)
        echo "Fetching Q variant models..."
        fetch_model "encoder_Q.onnx"
        fetch_model "decoder_Q.onnx"
        ;;
    B|b)
        echo "Fetching B variant models..."
        fetch_model "encoder_B.onnx"
        fetch_model "decoder_B.onnx"
        ;;
    C|c)
        echo "Fetching C variant models..."
        fetch_model "encoder_C.onnx"
        fetch_model "decoder_C.onnx"
        ;;
    all|ALL)
        echo "Fetching all model variants..."
        fetch_model "encoder_P.onnx"
        fetch_model "decoder_P.onnx"
        fetch_model "encoder_Q.onnx"
        fetch_model "decoder_Q.onnx"
        fetch_model "encoder_B.onnx"
        fetch_model "decoder_B.onnx"
        fetch_model "encoder_C.onnx"
        fetch_model "decoder_C.onnx"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE" >&2
        echo "Usage: $0 [P|Q|B|C|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Models fetched successfully to: $(pwd)/models/"
echo "You can now build and run the example."
