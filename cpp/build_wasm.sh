#!/bin/bash
# Build TrustMark C++ for WASM32-WASIP2
# Usage: ./build_wasm.sh [debug|release]

set -e

BUILD_TYPE="${1:-Release}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_wasm"

# WASI SDK path
export WASI_SDK_PATH="${WASI_SDK:-/opt/wasi-sdk}"

if [ ! -d "$WASI_SDK_PATH" ]; then
    echo "Error: WASI_SDK not found at $WASI_SDK_PATH"
    echo "Please set WASI_SDK environment variable or install WASI SDK to /opt/wasi-sdk"
    exit 1
fi

echo "=========================================="
echo "Building TrustMark for WASM32-WASIP2"
echo "=========================================="
echo "WASI SDK: $WASI_SDK_PATH"
echo "Build Type: $BUILD_TYPE"
echo "Build Directory: $BUILD_DIR"
echo "=========================================="

# First, build ONNX Runtime for WASI if not already built
ORT_WASI_DIR="${SCRIPT_DIR}/onnxruntime-wasi/build_wasi"
if [ ! -d "$ORT_WASI_DIR" ]; then
    echo "Building ONNX Runtime for WASI..."
    cd "${SCRIPT_DIR}/onnxruntime-wasi"
    WASI_SDK_PATH="$WASI_SDK_PATH" ./build_wasi.sh "$BUILD_TYPE"
    echo "ONNX Runtime build complete"
    cd "$SCRIPT_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake configuration for WASM
cmake .. \
    -DCMAKE_SYSTEM_NAME=WASI \
    -DCMAKE_SYSTEM_VERSION=1 \
    -DCMAKE_SYSTEM_PROCESSOR=wasm32 \
    -DCMAKE_C_COMPILER="$WASI_SDK_PATH/bin/clang" \
    -DCMAKE_CXX_COMPILER="$WASI_SDK_PATH/bin/clang++" \
    -DCMAKE_AR="$WASI_SDK_PATH/bin/llvm-ar" \
    -DCMAKE_RANLIB="$WASI_SDK_PATH/bin/llvm-ranlib" \
    -DCMAKE_C_COMPILER_TARGET=wasm32-wasip2 \
    -DCMAKE_CXX_COMPILER_TARGET=wasm32-wasip2 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_SYSROOT="$WASI_SDK_PATH/share/wasi-sysroot" \
    -DCMAKE_FIND_ROOT_PATH="$WASI_SDK_PATH/share/wasi-sysroot" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DBUILD_FOR_WASM=ON \
    -DCMAKE_EXECUTABLE_SUFFIX=".wasm" \
    -DCMAKE_CXX_FLAGS="-fno-exceptions -fno-rtti" \
    -DONNXRUNTIME_DIR="$ORT_WASI_DIR"

# Build
echo ""
echo "=========================================="
echo "Building TrustMark..."
echo "=========================================="
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Output:"
find . -name "*.wasm" -type f

echo ""
echo "To run with wasmtime:"
echo "  wasmtime run --dir=. ./trustmark_example.wasm <args>"
