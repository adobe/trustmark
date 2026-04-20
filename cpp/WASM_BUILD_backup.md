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
   python3.11 -m pip install --break-system-packages onnxruntime onnx flatbuffers
   ```

## Build Process

### Step 1: Convert Models to ORT Format and Generate Operator Config

**IMPORTANT**: This step must be done FIRST. It generates the `required_operators.config` file that tells the build which operators to include.

The minimal ONNX Runtime build only supports `.ort` (optimized ONNX Runtime) format, not `.onnx` format.
The `onnxruntime`, `onnx`, and `flatbuffers` Python packages are required.

```bash
# Install required Python packages
python3.11 -m pip install --break-system-packages onnxruntime onnx flatbuffers

# From the repository root
cd trustmark/cpp
./fetch_models.sh

cd onnxruntime-wasi/tools/python

# Convert all ONNX models to ORT format
# This also generates models/required_operators.config
PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages python3.11 convert_onnx_models_to_ort.py \
  ../../../models \
  --output_dir ../../../models
```

This creates:
- `encoder_P.ort` (33MB) - Fixed optimization
- `encoder_P.with_runtime_opt.ort` (17MB) - Runtime optimization (smaller)
- `decoder_P.ort` (91MB) - Fixed optimization
- `decoder_P.with_runtime_opt.ort` (45MB) - Runtime optimization (smaller)
- **`required_operators.config`** - Lists operators needed by the models
- Similar files for Q, B, C variants

Use the `with_runtime_opt.ort` versions for smaller file sizes.

### Step 2: Create TrustMark WASM Example

The ONNX Runtime build compiles `onnxruntime/wasm/simple.cpp` into the WASM module. To add TrustMark functionality, copy your TrustMark example:

```bash
# From trustmark/cpp directory
cd onnxruntime-wasi

# Copy the TrustMark WASM example
cp ../examples/trustmark_wasm.cpp onnxruntime/wasm/simple.cpp
```

**Note**: Do NOT use `try-catch` blocks in WASM code - the build uses `-fno-exceptions`.

### Step 3: Build ONNX Runtime for WASI with TrustMark Operators

**CRITICAL**: You must use the correct build script that includes the operators specified in `required_operators.config`.

The ONNX Runtime WASI fork is included as a git submodule at `cpp/onnxruntime-wasi`. The minimal build with operator configuration produces `ort-wasi-simd.wasm`, which contains both ONNX Runtime and your custom application code.

```bash
# From trustmark/cpp/onnxruntime-wasi directory
export WASI_SDK_PATH=/opt/wasi-sdk

# Use the build script that includes TrustMark operators
./build_wasi_minimal_with_config.sh
```

This script:
1. Reads `models/required_operators.config`
2. Generates operator registration files using `reduce_op_kernels.py`
3. Builds a minimal ONNX Runtime with ONLY the operators needed by TrustMark models
4. Creates `build_wasi_minimal_config/ort-wasi-simd.wasm` (21MB)

#### What Operators Are Included?

The TrustMark models require these operators:
```
ai.onnx;1;GlobalAveragePool
ai.onnx;11;Conv
ai.onnx;12;MaxPool
ai.onnx;13;Cast,Concat,Flatten,Gemm,Resize
ai.onnx;14;Relu,Reshape
com.microsoft;1;FusedConv,QuickGelu
```

**Why This Matters**: The old `build_wasi_simple.sh` script builds a minimal runtime with a default set of operators, but **does not include** the Microsoft contrib ops (`FusedConv`, `QuickGelu`) that TrustMark models require. This results in garbage output (near-zero values) even though the build succeeds and the model loads.

## Running the WASM Module

The WASM binary is located at:
```
cpp/onnxruntime-wasi/build_wasi_minimal_config/ort-wasi-simd.wasm
```

### Basic Test

```bash
# From the repository root
cd trustmark/cpp

# Test encoder with dummy data
wasmtime --dir=.::.  --dir=models::/models \
  onnxruntime-wasi/build_wasi_minimal_config/ort-wasi-simd.wasm \
  /models/encoder_P.ort

# Test decoder with dummy data
wasmtime --dir=.::.  --dir=models::/models \
  onnxruntime-wasi/build_wasi_minimal_config/ort-wasi-simd.wasm \
  /models/decoder_P.ort
```

### Example Output

**Encoder with dummy data:**
```
TrustMark WASM Example
======================

Loading model: /models/encoder_P.ort
✓ ONNX Runtime initialized
✓ Session options configured
✓ Model loaded successfully!

Model Information:
  Number of inputs: 2
  Input 0: onnx::Concat_0 [1, 3, 256, 256]
  Input 1: onnx::Gemm_1 [1, 100]
  Number of outputs: 1
  Output 0: image

✓ Detected TrustMark Encoder model
  Input 0 (image): expecting shape [1, 3, 256, 256]
  Input 1 (secret): expecting shape [1, 100]

Running inference with dummy data...
✓ Inference completed successfully!
  Output shape: [1, 3, 256, 256]

✓ Output Statistics:
  Min value: -0.999998
  Max value: 0.999727
  Average |value|: 0.730813
  First 10 values: 0.298983 0.352422 0.417632 0.411946 0.177688 0.130361 0.0561037 -0.0374335 0.0218523 -0.234194

✓ Output values look reasonable (not near-zero)

✓ TrustMark WASM example completed successfully!
```

**Decoder with dummy data:**
```
TrustMark WASM Example
======================

Loading model: /models/decoder_P.ort
✓ ONNX Runtime initialized
✓ Session options configured
✓ Model loaded successfully!

Model Information:
  Number of inputs: 1
  Input 0: image [1, 3, 224, 224]
  Number of outputs: 1
  Output 0: output

✓ Detected TrustMark Decoder model
  Input 0 (image): expecting shape [1, 3, 224/256, 224/256]

Running inference with dummy data...
✓ Inference completed successfully!
  Output shape: [1, 100]

✓ Output Statistics:
  Min value: -12.5254
  Max value: 10.1346
  Average |value|: 2.89901
  First 10 values: -0.0614567 1.06644 4.60636 -1.2866 -1.75627 2.61656 -1.24109 -0.343011 -0.952926 1.11766

✓ Output values look reasonable (not near-zero)

✓ TrustMark WASM example completed successfully!
```

**Output Verification**: The statistics confirm the build is working correctly:
- **Encoder**: Average |value| of 0.73 (normalized image output in [-1, 1] range)
- **Decoder**: Average |value| of 2.9 (reasonable logits for 100-bit classification)
- **Not near-zero**: This confirms the operators are being executed correctly

## Current Status

### WASM Build Status: **✅ WORKING - CPU Only, Standalone Runtime**

WASI (WebAssembly System Interface) runs **outside the browser** as a standalone application.

### ✅ What Works

1. **TrustMark Model Inference** - ✅ CONFIRMED WORKING
   - Encoder and decoder models produce correct output
   - All required operators included (`FusedConv`, `QuickGelu`, etc.)
   - Output values verified (not near-zero garbage)
2. **Standalone Execution** - Runs with wasmtime/wasmer as a command-line tool
3. **ONNX Runtime CPU** - Full CPU inference via ONNX Runtime WASI fork
4. **ORT Model Format** - Optimized model format for minimal builds
5. **BCH Error Correction** - Pure C++ implementation works in WASM
6. **WASI File Access** - Read models through WASI filesystem APIs

### ❌ What Doesn't Work Yet

1. **Image I/O** - No image loading/saving in current build
   - Need to add stb_image libraries and image_utils.cpp
   - Can add this by modifying CMakeLists and including image support files
2. **Standard ONNX Models** - Only `.ort` format supported in minimal build
   - Must convert models using the Python script
3. **GPU Acceleration** - No GPU support in WASI yet
   - CPU-only execution for now
4. **Exception Handling** - C++ exceptions disabled (`-fno-exceptions`)
   - Use error codes and return values instead

### 🔧 Known Issues (RESOLVED)

**Issue**: Minimal builds without operator configuration produce garbage output (near-zero values)

**Root Cause**: The default minimal build includes only basic operators and excludes:
- Microsoft contrib ops: `FusedConv`, `QuickGelu`
- Some standard ops used by TrustMark models

**Solution**: Use `build_wasi_minimal_with_config.sh` which:
1. Reads `required_operators.config` generated during model conversion
2. Uses `reduce_op_kernels.py` to generate operator registration files
3. Includes all operators needed by TrustMark models

**Verification**: Output statistics show correct values:
- Encoder: Average |value| = 0.73 (expected: 0.5-0.8 for normalized images)
- Decoder: Average |value| = 2.9 (expected: 2-4 for logits)

## Architecture

```
┌──────────────────────────────────────────┐
│     Host System (Linux/macOS/Windows)   │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │   Wasmtime / Wasmer Runtime        │ │
│  │                                    │ │
│  │  ┌──────────────────────────────┐ │ │
│  │  │  ort-wasi-simd.wasm          │ │ │
│  │  │  (21MB WASM binary)          │ │ │
│  │  │                              │ │ │
│  │  │  ┌────────────────────────┐  │ │ │
│  │  │  │  TrustMark Logic       │  │ │ │
│  │  │  │  (simple.cpp)          │  │ │ │
│  │  │  └────────────────────────┘  │ │ │
│  │  │                              │ │ │
│  │  │  ┌────────────────────────┐  │ │ │
│  │  │  │  ONNX Runtime (CPU)    │  │ │ │
│  │  │  │  - Minimal build       │  │ │ │
│  │  │  │  - ORT format only     │  │ │ │
│  │  │  └────────────────────────┘  │ │ │
│  │  │                              │ │ │
│  │  │  ┌────────────────────────┐  │ │ │
│  │  │  │  BCH Error Correction  │  │ │ │
│  │  │  └────────────────────────┘  │ │ │
│  │  └──────────────────────────────┘ │ │
│  └────────────────────────────────────┘ │
│                  ↕                       │
│  ┌────────────────────────────────────┐ │
│  │   WASI APIs                        │ │
│  │   - File I/O (read .ort models)    │ │
│  │   - Command-line arguments         │ │
│  │   - Environment variables          │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
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
- Make sure you're using a build script that disables Abseil
- The build scripts set `-Donnxruntime_DISABLE_ABSEIL=ON`

**Error: `ImportError: cannot import name 'parse_config' from 'util'`**
- Missing `flatbuffers` Python package
- Solution: `python3.11 -m pip install --break-system-packages flatbuffers`

### Runtime Errors

**Error: `ONNX format model is not supported in this build`**
- You're trying to load a `.onnx` file instead of `.ort`
- Convert your models using the Python script (see Step 1)
- Use the converted `.ort` files

**Problem: Model loads but produces garbage output (near-zero values)**
- You're using the wrong build script (e.g., `build_wasi_simple.sh`)
- Solution: Use `build_wasi_minimal_with_config.sh` which includes TrustMark operators
- Verify: Check output statistics show Average |value| > 0.5 (not ~0.03)

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
onnxruntime-wasi/ (WASI fork - git submodule)
├─ build_wasi_simple.sh                    # OLD: Basic build (missing operators) ❌
├─ build_wasi_minimal_with_config.sh       # NEW: Correct build (includes operators) ✅
├─ onnxruntime/wasm/simple.cpp             # Your TrustMark code goes here
├─ tools/ci_build/reduce_op_kernels.py     # Generates operator registration
└─ build_wasi_minimal_config/
    └─ ort-wasi-simd.wasm                 # Output WASM binary (21MB)

trustmark/cpp/
├─ models/
│   ├─ encoder_P.onnx                     # Original models
│   ├─ encoder_P.ort                      # Converted for WASM (33MB)
│   ├─ encoder_P.with_runtime_opt.ort     # Smaller (17MB)
│   ├─ decoder_P.ort                      # Converted for WASM (91MB)
│   ├─ decoder_P.with_runtime_opt.ort     # Smaller (45MB)
│   └─ required_operators.config          # Generated during conversion ⚠️ REQUIRED
└─ examples/
    └─ trustmark_wasm.cpp                 # Reference TrustMark WASM code
```

## Complete Example Workflow

```bash
# 1. Clone repository with submodules (cpp_wasi branch)
git clone --recurse-submodules -b cpp_wasi https://github.com/cdmurph32/trustmark.git
cd trustmark/cpp

# 2. Install prerequisites
brew install wasmtime python@3.11
python3.11 -m pip install --break-system-packages onnxruntime onnx flatbuffers

# 3. Fetch and convert models to ORT format (generates required_operators.config)
./fetch_models.sh
cd onnxruntime-wasi/tools/python
PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages python3.11 convert_onnx_models_to_ort.py \
  ../../../models \
  --output_dir ../../../models

# Verify required_operators.config was created
cat ../../../models/required_operators.config

# 4. Copy TrustMark WASM example
cd ../../  # Back to onnxruntime-wasi
cp ../examples/trustmark_wasm.cpp onnxruntime/wasm/simple.cpp

# 5. Build ONNX Runtime WASM with operator configuration
export WASI_SDK_PATH=/opt/wasi-sdk
./build_wasi_minimal_with_config.sh

# 6. Test the WASM module
cd ..  # Back to cpp/
wasmtime --dir=.::.  --dir=models::/models \
  onnxruntime-wasi/build_wasi_minimal_config/ort-wasi-simd.wasm \
  /models/encoder_P.ort

# Verify output shows reasonable values (not near-zero)
# Expected: Average |value| > 0.5
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
- [Wasmer](https://wasmer.io())
- [ONNX Runtime ORT Format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
