# Building TrustMark for WASM32-WASIP2

This guide explains how to build TrustMark for WebAssembly using WASI Preview 2.

**IMPORTANT**: WASI is for **standalone/server-side** execution, NOT browser-based.
It runs with wasmtime/wasmer runtimes, no JavaScript involved.

## Prerequisites

1. **WASI SDK 28 or newer** - Download and install to `/opt/wasi-sdk`
   ```bash
   # macOS ARM64
   wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-28/wasi-sdk-28.0-arm64-macos.tar.gz
   sudo tar xzf wasi-sdk-28.0-arm64-macos.tar.gz -C /opt/
   sudo ln -s /opt/wasi-sdk-28.0-arm64-macos /opt/wasi-sdk

   # macOS x86_64
   wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-28/wasi-sdk-28.0-x86_64-macos.tar.gz
   sudo tar xzf wasi-sdk-28.0-x86_64-macos.tar.gz -C /opt/
   sudo ln -s /opt/wasi-sdk-28.0-x86_64-macos /opt/wasi-sdk

   # Linux x86_64
   wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-28/wasi-sdk-28.0-x86_64-linux.tar.gz
   sudo tar xzf wasi-sdk-28.0-x86_64-linux.tar.gz -C /opt/
   sudo ln -s /opt/wasi-sdk-28.0-x86_64-linux /opt/wasi-sdk
   ```

   **Note**: WASI SDK 28 or newer is required for wasm32-wasip2 support. Check [releases](https://github.com/WebAssembly/wasi-sdk/releases) for the latest version.

2. **Modified ONNX Runtime** - Already included as a git submodule
   ```bash
   # Initialize submodules when cloning the repository
   git clone --recurse-submodules -b cpp_wasi https://github.com/cdmurph32/trustmark.git

   # Or if already cloned, initialize the submodule
   cd /path/to/trustmark
   git checkout cpp_wasi
   git submodule update --init --recursive
   ```

3. **Wasmtime** (for running WASM modules)
   ```bash
   brew install wasmtime
   ```

4. **Python 3.11** (for model conversion - onnxruntime not yet on Python 3.14)
   ```bash
   brew install python@3.11
   python3.11 -m pip install --break-system-packages onnxruntime onnx
   ```

## Build Process

### Step 1: Build ONNX Runtime for WASI

The ONNX Runtime WASI fork (included as a submodule at `cpp/onnxruntime-wasi`) includes a minimal build that produces `ort-wasi-simd.wasm`. This WASM module contains both ONNX Runtime and your custom application code.

```bash
cd /path/to/trustmark/cpp/onnxruntime-wasi
export WASI_SDK_PATH=/opt/wasi-sdk
./build_wasi_simple.sh
```

This creates:
- `build_wasi/ort-wasi-simd.wasm` (21MB) - The WASM binary
- `build_wasi/lib/libonnxruntime*.a` - Static libraries

### Step 2: Create TrustMark WASM Example

The ONNX Runtime build compiles `onnxruntime/wasm/simple.cpp` into the WASM module. To add TrustMark functionality, replace this file with your TrustMark example:

```bash
cd /path/to/trustmark/cpp/onnxruntime-wasi

# Create TrustMark example (or copy from cpp/examples/trustmark_wasm.cpp)
cat > onnxruntime/wasm/simple.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.ort>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    std::cout << "Loading model: " << model_path << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrustMarkWASM");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path, session_options);

    std::cout << "? Model loaded successfully!" << std::endl;

    // Add your TrustMark encoding/decoding logic here

    return 0;
}
EOF

# Rebuild
export WASI_SDK_PATH=/opt/wasi-sdk
./build_wasi_simple.sh
```

**Note**: Do NOT use `try-catch` blocks in WASM code - the build uses `-fno-exceptions`.

### Step 3: Convert Models to ORT Format

The minimal ONNX Runtime build only supports `.ort` (optimized ONNX Runtime) format, not `.onnx` format.

```bash
cd /path/to/trustmark/cpp/models

# Convert all ONNX models to ORT format
cd /path/to/trustmark/cpp/onnxruntime-wasi/tools/python
PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages python3.11 convert_onnx_models_to_ort.py \
  /path/to/trustmark/cpp/models \
  --output_dir /path/to/trustmark/cpp/models
```

This creates:
- `encoder_P.ort` (33MB) - Fixed optimization
- `encoder_P.with_runtime_opt.ort` (17MB) - Runtime optimization (smaller)
- `decoder_P.ort` (91MB) - Fixed optimization
- `decoder_P.with_runtime_opt.ort` (45MB) - Runtime optimization (smaller)
- Similar files for Q, B, C variants

Use the `with_runtime_opt.ort` versions for smaller file sizes.

## Running the WASM Module

The WASM binary is located at:
```
/path/to/trustmark/cpp/onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm
```

### Basic Test

```bash
cd /path/to/trustmark/cpp

# Test encoder
wasmtime --dir=models::/models \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  /models/encoder_P.ort

# Test decoder
wasmtime --dir=models::/models \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  /models/decoder_P.ort
```

### Example Output

```
TrustMark WASM Example
======================

Loading model: /models/encoder_P.ort
? ONNX Runtime initialized
? Session options configured
? Model loaded successfully!

Model Information:
  Number of inputs: 2
  Input 0: onnx::Concat_0 [1, 3, 256, 256]
  Input 1: onnx::Gemm_1 [1, 100]
  Number of outputs: 1
  Output 0: image

? Detected TrustMark Encoder model
  Input 0 (image): expecting shape [1, 3, 256, 256]
  Input 1 (secret): expecting shape [1, 100]

Running inference with dummy data...
? Inference completed successfully!
  Output shape: [1, 3, 256, 256]

? TrustMark WASM example completed successfully!
```

## Current Limitations

### WASM Build Status: **CPU Only, Standalone Runtime**

WASI (WebAssembly System Interface) runs **outside the browser** as a standalone application.

The current WASM build supports:
- ? ONNX Runtime CPU execution
- ? Model loading and inference via WASI filesystem
- ? Basic watermark encoding/decoding logic
- ? WASI file I/O for models
- ? OpenCV (not available in WASM) - needs custom image handling
- ? GPU acceleration (CPU only for now)
- ? Standard `.onnx` models (must use `.ort` format)

### What Works

1. **Standalone Execution** - Runs with wasmtime/wasmer as a command-line tool
2. **ONNX Model Inference** - Full CPU inference via ONNX Runtime
3. **ORT Model Format** - Optimized model format for minimal builds
4. **BCH Error Correction** - Pure C++ implementation works in WASM
5. **Core Watermarking Logic** - Encoding/decoding algorithms
6. **WASI File Access** - Read models through WASI APIs

### What Doesn't Work Yet

1. **Standard ONNX Models** - Only `.ort` format supported in minimal build
   - Convert models using the Python script (see Step 3)
2. **Image I/O** - OpenCV not available in WASM
   - Need to implement custom image reading (e.g., PNG/JPEG parsers)
   - Or accept raw pixel data as input
3. **GPU Acceleration** - No GPU support in WASI yet
   - CPU-only execution for now
4. **Exception Handling** - C++ exceptions disabled (`-fno-exceptions`)
   - Use error codes and return values instead

## Architecture

```
????????????????????????????????????????????
?     Host System (Linux/macOS/Windows)   ?
?                                          ?
?  ?????????????????????????????????????? ?
?  ?   Wasmtime / Wasmer Runtime        ? ?
?  ?                                    ? ?
?  ?  ???????????????????????????????? ? ?
?  ?  ?  ort-wasi-simd.wasm          ? ? ?
?  ?  ?  (21MB WASM binary)          ? ? ?
?  ?  ?                              ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ?  ?  TrustMark Logic       ?  ? ? ?
?  ?  ?  ?  (simple.cpp)          ?  ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ?                              ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ?  ?  ONNX Runtime (CPU)    ?  ? ? ?
?  ?  ?  ?  - Minimal build       ?  ? ? ?
?  ?  ?  ?  - ORT format only     ?  ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ?                              ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ?  ?  BCH Error Correction  ?  ? ? ?
?  ?  ?  ??????????????????????????  ? ? ?
?  ?  ???????????????????????????????? ? ?
?  ?????????????????????????????????????? ?
?                  ?                       ?
?  ?????????????????????????????????????? ?
?  ?   WASI APIs                        ? ?
?  ?   - File I/O (read .ort models)    ? ?
?  ?   - Command-line arguments         ? ?
?  ?   - Environment variables          ? ?
?  ?????????????????????????????????????? ?
????????????????????????????????????????????
```

## Use Cases

WASI enables TrustMark to run in:
- **Command-line tools** - Process images from the terminal
- **Server-side applications** - Watermark images on servers
- **Edge computing** - Deploy to edge devices
- **Containerized environments** - Run in lightweight WASM containers
- **Cross-platform** - Single WASM binary runs everywhere

## Troubleshooting

### Build Errors

**Error: WASI_SDK_PATH environment variable is not set**
```bash
export WASI_SDK_PATH=/opt/wasi-sdk
```

**Error: `cannot use 'try' with exceptions disabled`**
- Remove all `try-catch` blocks from your code
- The WASI build uses `-fno-exceptions -fno-rtti`
- Use error codes and return values instead

**Error: Abseil signal support issues**
- Make sure you're using `build_wasi_simple.sh` which disables Abseil
- This script sets `-Donnxruntime_DISABLE_ABSEIL=ON`

### Runtime Errors

**Error: `ONNX format model is not supported in this build`**
- You're trying to load a `.onnx` file instead of `.ort`
- Convert your models using the Python script (see Step 3)
- Use the converted `.ort` files

**Error: Cannot open model file**
```bash
# Make sure to grant filesystem access with --dir
wasmtime --dir=models::/models ./ort-wasi-simd.wasm /models/encoder_P.ort
```

**Error: File not found**
- Check that the path inside WASM uses the mapped directory name
- Example: `--dir=models::/models` maps host `models/` to `/models` in WASM

## Files and Locations

```
onnxruntime/ (WASI fork)
??? build_wasi_simple.sh              # Build script for WASM
??? onnxruntime/wasm/simple.cpp       # Your TrustMark code goes here
??? build_wasi/
    ??? ort-wasi-simd.wasm           # Output WASM binary (21MB)

trustmark/cpp/
??? models/
?   ??? encoder_P.onnx               # Original models
?   ??? encoder_P.ort                # Converted for WASM (33MB)
?   ??? encoder_P.with_runtime_opt.ort  # Smaller (17MB)
?   ??? decoder_P.ort                # Converted for WASM (91MB)
?   ??? decoder_P.with_runtime_opt.ort  # Smaller (45MB)
??? examples/
    ??? trustmark_wasm.cpp           # Reference TrustMark WASM code
```

## Complete Example Workflow

```bash
# 1. Clone repository with submodules (cpp_wasi branch)
git clone --recurse-submodules -b cpp_wasi https://github.com/cdmurph32/trustmark.git
cd trustmark

# 2. Install prerequisites
brew install wasmtime python@3.11
python3.11 -m pip install --break-system-packages onnxruntime onnx

# 3. Build ONNX Runtime WASM
cd cpp/onnxruntime-wasi
export WASI_SDK_PATH=/opt/wasi-sdk
./build_wasi_simple.sh

# 4. Convert models to ORT format
cd tools/python
PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages python3.11 convert_onnx_models_to_ort.py \
  ../../models \
  --output_dir ../../models

# 5. Test the WASM module
cd ../../..  # Back to cpp directory
wasmtime --dir=models::/models \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  /models/encoder_P.ort
```

## WASI vs Browser WebAssembly

| Feature | WASI (This Build) | Browser WASM |
|---------|-------------------|--------------|
| Runtime | Wasmtime, Wasmer | Web browser |
| APIs | POSIX-like (file I/O, etc.) | JavaScript APIs |
| Use Case | Server, CLI, edge | Web applications |
| Language | C/C++ standalone | C++ + JS bindings |
| File Access | Direct filesystem | Virtual filesystem |
| Model Format | `.ort` (optimized) | `.onnx` or `.ort` |

**For browser-based watermarking**, you would use:
- Emscripten (not WASI)
- JavaScript bindings
- Canvas API for images
- WebGPU for GPU acceleration
- Full ONNX Runtime build (not minimal)

**This WASI build is for** standalone, server-side, and command-line usage.

## References

- [WASI SDK](https://github.com/WebAssembly/wasi-sdk)
- [WASI Specification](https://github.com/WebAssembly/WASI)
- [ONNX Runtime WASI Fork](https://github.com/MendyBerger/onnxruntime)
- [Wasmtime](https://wasmtime.dev/)
- [Wasmer](https://wasmer.io/)
- [ONNX Runtime ORT Format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
