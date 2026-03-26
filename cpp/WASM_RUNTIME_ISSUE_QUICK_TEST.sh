#!/bin/bash
# Quick test script to reproduce the WASM ONNX Runtime issue

set -e

echo "=== WASM ONNX Runtime Issue Reproduction ==="
echo ""

# Check prerequisites
if [ ! -f "models/encoder_P.ort" ]; then
    echo "❌ models/encoder_P.ort not found"
    echo "Run: cd models && fetch_models.sh && convert models to .ort"
    exit 1
fi

if [ ! -f "../images/ufo_240.jpg" ]; then
    echo "❌ ../images/ufo_240.jpg not found"
    exit 1
fi

if [ ! -f "onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm" ]; then
    echo "❌ WASM binary not found. Building..."
    cd onnxruntime-wasi
    export WASI_SDK_PATH=/opt/wasi-sdk
    ./build_wasi_simple.sh
    cd ..
fi

echo "Running WASM test..."
echo ""

wasmtime --dir=.::.  --dir=models::/models --dir=../images::/images \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  /models/encoder_P.ort /images/ufo_240.jpg 2>&1 | tee wasm_output.log

echo ""
echo "=== Results ==="
echo ""

# Extract key values
INPUT=$(grep "DEBUG: Input CHW" wasm_output.log || echo "Not found")
OUTPUT=$(grep "DEBUG: CHW channel" wasm_output.log | head -3 || echo "Not found")

echo "Input (first pixel):"
echo "$INPUT"
echo ""
echo "Output (first pixel):"
echo "$OUTPUT"
echo ""

# Check if output is near zero (indicating the bug)
if grep -q "CHW channel 0 (first pixel): -0.0" wasm_output.log; then
    echo "❌ BUG CONFIRMED: Output values near zero"
    echo ""
    echo "Expected output: ~-0.77, ~-0.78, ~-0.95"
    echo "Actual output: near 0.0"
    echo ""
    echo "See WASM_RUNTIME_ISSUE.md for full details"
else
    echo "✅ Output values look reasonable"
    echo "The issue may have been fixed!"
fi

echo ""
echo "Output image: output_watermarked.png"
echo "Compare with native: test_ort_native_output.png"

