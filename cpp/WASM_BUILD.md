# Building TrustMark for WASM32-WASIP2

This guide explains how to build TrustMark for WebAssembly using WASI Preview 2.

**IMPORTANT**: WASI is for **standalone/server-side** execution, NOT browser-based.
It runs with wasmtime/graphtime runtimes, no JavaScript involved.

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

   **Note**: WASI SDK 28 or newer is required for wasm32-wasip2 support.

2. **Modified ONNX Runtime with WebGPU** - Already included as `onnxruntime-wasi/`
   ```bash
   # Clone the repository
   git clone -b cpp_wasi https://github.com/cdmurph32/trustmark.git
   cd trustmark/cpp

   # onnxruntime-wasi/ points to https://github.com/cdmurph32/onnxruntime (cpp_wasi branch)
   # Set it up as a symlink or clone:
   git clone -b cpp_wasi https://github.com/cdmurph32/onnxruntime onnxruntime-wasi
   ```

3. **Graphtime** (for running WASM modules with WebGPU support)
   ```bash
   git clone https://github.com/cdmurph32/graphtime
   cd graphtime
   cargo build --release
   ```

   Or use wasmtime for CPU-only execution:
   ```bash
   brew install wasmtime
   ```

4. **Python 3.11** (for model conversion)
   ```bash
   brew install python@3.11
   python3.11 -m pip install --break-system-packages onnxruntime onnx flatbuffers
   ```

## Build Process

### Step 1: Download Models

```bash
# From the trustmark/cpp directory
./fetch_models.sh
```

This downloads `.onnx` files (e.g. `encoder_P.onnx`, `decoder_P.onnx`).

### Step 2: Convert Models to ORT Format

The WASM build requires `.ort` format models. **For WebGPU, you need the `with_runtime_opt.ort`
variants** — see the [WebGPU f16 precision fix](#4-webgpu-f16-precision-fix) below for why.

```bash
# From the trustmark/cpp directory
cd onnxruntime-wasi/tools/python

# Convert all models — produces both .ort and .with_runtime_opt.ort variants
python3.11 convert_onnx_models_to_ort.py \
  ../../../models \
  --output_dir ../../../models

# Or generate only the with_runtime_opt.ort files (what WebGPU needs)
python3.11 convert_onnx_models_to_ort.py \
  --optimization_style Runtime \
  ../../../models \
  --output_dir ../../../models
```

Output files per model:
| File | Description |
|------|-------------|
| `encoder_P.ort` | Basic ORT format, fixed graph optimizations only |
| `encoder_P.with_runtime_opt.ort` | ORT format with runtime graph optimizations pre-applied |

**Use `with_runtime_opt.ort` for WebGPU execution.** The plain `.ort` files are sufficient for
CPU/wasmtime.

### Step 3: Stage TrustMark Sources into onnxruntime-wasi

TrustMark-specific source files live in this repo (`cpp/wasm/`, `cpp/examples/`) and must
be copied into onnxruntime-wasi before building. `prepare_ort_build.sh` handles this:

```bash
# From the trustmark/cpp directory
./prepare_ort_build.sh
```

This copies:
| Source (trustmark/cpp/) | Destination (onnxruntime-wasi/onnxruntime/wasm/) |
|-------------------------|--------------------------------------------------|
| `wasm/image_utils.cpp` | `image_utils.cpp` |
| `wasm/image_utils.h` | `image_utils.h` |
| `wasm/stb_image.h` | `stb_image.h` |
| `wasm/stb_image_resize2.h` | `stb_image_resize2.h` |
| `wasm/stb_image_write.h` | `stb_image_write.h` |
| `examples/trustmark_wasm_image.cpp` | `simple.cpp` (include path adjusted) |

Re-run this script any time you change the entry point or image utilities.

### Step 4: Build ONNX Runtime for WASI with WebGPU

> **IMPORTANT**: Run Step 3 (`prepare_ort_build.sh`) before this step. If `simple.cpp` is
> absent when CMake first runs, the `onnxruntime_webassembly` target will not be generated
> and `make` will fail with "No rule to make target". Fix: stage the files, then re-run
> `build_wasi.sh` — CMake will reconfigure automatically via `CONFIGURE_DEPENDS`.

```bash
# From the onnxruntime-wasi directory
cd onnxruntime-wasi
export WASI_SDK_PATH=/opt/wasi-sdk

# Build with WebGPU and SIMD support
./build_wasi.sh Release \
  -Donnxruntime_USE_WEBGPU=ON \
  -Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=ON \
  -Donnxruntime_EXTENDED_MINIMAL_BUILD=ON \
  -Donnxruntime_WGSL_TEMPLATE=static
```

Output: `build_wasi/ort-wasi-simd.wasm` (~12MB plain WASM module)

The build includes:
- WebGPU execution provider (wasi:webgpu component model)
- MLAS SIMD optimizations
- All TrustMark operators (`ai.onnx;17;Add,Cast,Concat,...` + `com.microsoft;1;FusedConv,QuickGelu`)
- Image utilities (stb_image, stb_image_resize2, stb_image_write)

**REQUIRED**: After building, convert the WASM module to a WASI Preview 2 component:

```bash
# Still in onnxruntime-wasi directory
ADAPTER=$(find ~/.cargo/registry -name "wasi_snapshot_preview1.command.wasm" | \
  grep "wasi-preview1-component-adapter-provider" | sort -rV | head -1)

wasm-tools component new build_wasi/ort-wasi-simd.wasm \
  --adapt "wasi_snapshot_preview1=$ADAPTER" \
  -o build_wasi/ort-wasi-simd.wasm
```

Verify it's a component (magic bytes should be `00 61 73 6d 0d 00 01 00`):
```bash
xxd build_wasi/ort-wasi-simd.wasm | head -1
```

### Step 5: Verify Entry Point

`onnxruntime/wasm/simple.cpp` is compiled into the WASM binary. It is generated from
`examples/trustmark_wasm_image.cpp` by `prepare_ort_build.sh` (Step 3). Do not edit
`simple.cpp` directly — edit `examples/trustmark_wasm_image.cpp` and re-run the script.

**Note**: Do NOT use `try-catch` — the build uses `-fno-exceptions`.

## Running the WASM Module

The WASM binary is at: `onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm`

### With Graphtime (WebGPU)

Graphtime is a WASI runtime with wasi:webgpu support. It runs the WASM component
and provides GPU acceleration via the host's WebGPU implementation.

#### Directory Mapping Syntax

Graphtime uses `GUEST_PATH::HOST_PATH` to map directories:

```
--dir "/guest/path::host/path"
```

- Guest path: the path the WASM program sees
- Host path: the actual path on your filesystem (absolute or relative to cwd)

#### Example Usage

```bash
# From trustmark/cpp directory
cd /path/to/trustmark/cpp

GRAPHTIME=/path/to/graphtime/target/release/graphtime
WASM=onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm

# Run encoder (embeds watermark into image)
# NOTE: must use with_runtime_opt.ort for both encoder and decoder
USE_WEBGPU=1 $GRAPHTIME \
  --dir "/models::$(pwd)/models" \
  --dir "/images::$(pwd)/../images" \
  --dir ".::./" \
  $WASM -- /models/encoder_P.with_runtime_opt.ort /images/ufo_240.jpg

# Run decoder (recovers watermark bits from image)
# NOTE: must use with_runtime_opt.ort for correct WebGPU decoder output
USE_WEBGPU=1 $GRAPHTIME \
  --dir "/models::$(pwd)/models" \
  --dir "/images::$(pwd)/output" \
  --dir ".::./" \
  $WASM -- /models/decoder_P.with_runtime_opt.ort /images/output_watermarked.png
```

`USE_WEBGPU=1` enables GPU execution via WebGPU. Omit for CPU fallback.

Output file `output_watermarked.png` is written to the directory mapped as `.` in the guest.

### With Wasmtime (CPU Only)

```bash
cd /path/to/trustmark/cpp

wasmtime \
  --dir "/models::$(pwd)/models" \
  --dir "/images::$(pwd)/../images" \
  --dir ".::./" \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  -- /models/encoder_P.ort /images/ufo_240.jpg
```

### Example Output

**Encoder (WebGPU):**
```
TrustMark WASM Example with Image Support
==========================================

Loading model: /models/encoder_P.ort
Input image: /images/ufo_240.jpg
✓ ONNX Runtime initialized
✓ WebGPU execution provider enabled (NCHW layout)
✓ Session options configured
✓ Model loaded successfully!

Model Information:
  Number of inputs: 2
  Input 0: onnx::Concat_0 [1, 3, 256, 256]
  Input 1: onnx::Gemm_1 [1, 100]
  Number of outputs: 1
  Output 0: image

Loading image...
✓ Image loaded: 240x240 with 3 channels

✓ Detected TrustMark Encoder model
✓ Image resized to 256x256
✓ Image normalized to [-1, 1] (RGB, CHW)

Running encoder inference...
✓ Inference completed successfully!
✓ Saved watermarked image: output_watermarked.png

✓ TrustMark WASM completed!
```

**Decoder (WebGPU):**
```
✓ Detected TrustMark Decoder model
✓ Image resized to 224x224, normalized (RGB, CHW)

Running decoder inference...
✓ Inference completed!
  Output: [1, 100]

Decoded bits: 1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001

✓ TrustMark WASM completed!
```

## Current Status

### ✅ FULLY WORKING (WebGPU + CPU/SIMD)

| Feature | Status |
|---------|--------|
| TrustMark encoder inference | ✅ WebGPU + CPU |
| TrustMark decoder inference | ✅ WebGPU + CPU |
| WebGPU execution provider | ✅ Working (f16 ops on GPU, 0 bit-flip errors) |
| f16 shader precision | ✅ Fixed (use with_runtime_opt.ort models, 0 bit flips) |
| MLAS SIMD optimizations | ✅ Working |
| Image I/O (stb) | ✅ Working |
| WASI file access | ✅ Working |
| WASM Component Model | ✅ Proper `(component ...)` output |

### Fixes Applied

**1. Missing Operators**
- Added: `Add`, `Mul`, `Sigmoid`, `Tanh`, `Pad`, `Slice`, `Transpose`, `Constant`, `ConstantOfShape`, `Shape`
- Config: `models/required_operators_complete.config`

**2. MLAS SIMD for WASI**
- `cmake/onnxruntime_mlas.cmake`: added `WASI` alongside `Emscripten` check

**3. Image Utilities**
- Added `image_utils.cpp` to CMake build

**4. WebGPU f16 precision fix** <a name="4-webgpu-f16-precision-fix"></a>
- Root cause: plain `.ort` models trigger ORT runtime graph optimization which inserts
  `InsertedPrecisionFreeCast` f16 Cast nodes; these break WebGPU EP data flow for **both**
  encoder and decoder
- Fix: use `with_runtime_opt.ort` models — graph optimization is pre-applied offline, Cast nodes
  are baked into the model and execute correctly on WebGPU
- `encoder_P.with_runtime_opt.ort` + `decoder_P.with_runtime_opt.ort` → 0 bit-flip errors
- Using `encoder_P.ort` (plain) on WebGPU produces ~39 bit-flip errors (corrupt watermark)
- Generated via `convert_onnx_models_to_ort.py --optimization_style Runtime` (see Step 2)
- `onnxruntime/core/platform/posix/env.cc`: added `#ifdef __wasi__` in `GetSelfPid()` to avoid
  `env::getpid` WASM import (blocks `wasm-tools component new`)
- `onnxruntime/core/common/logging/logging.cc`: added `#ifdef __wasi__` before `__wasm__` in
  `GetProcessId()` for same reason

**5. WGPU adapter info crash**
- `wasi-webgpu-headers/webgpu.c`: `wgpuDeviceGetAdapterInfo` uses `WGPU_SAFE_STRING_VIEW` macro
  to avoid storing `cabi_realloc` sentinel pointers (which `free()` then crashes on)

**6. Preprocessing correctness**
- Removed incorrect BGR channel swap; stb_image loads RGB, model expects RGB
- Encoder output is a residual; correct blend: `final = clamp(input + clamp(stego - input, ±0.2) * 1.25, -1, 1)`

## Architecture

```
Host System
│
├── graphtime (WASI runtime + WebGPU host)
│   ├── wasmtime (WASM execution engine)
│   └── wgpu (WebGPU implementation via wasi:webgpu WIT)
│
└── ort-wasi-simd.wasm  (WASM component)
    ├── TrustMark logic (simple.cpp ← examples/trustmark_wasm_image.cpp)
    ├── ONNX Runtime (WebGPU EP + CPU fallback)
    ├── MLAS SIMD kernels
    └── Image utilities (stb ← wasm/image_utils.cpp)
```

## Key Repositories

| Repo | Purpose |
|------|---------|
| `cdmurph32/trustmark` (cpp_wasi branch) | This repo |
| `cdmurph32/onnxruntime` (cpp_wasi branch) | ORT with wasi:webgpu EP |
| `cdmurph32/wasi-gfx-runtime` | wasi-webgpu wasmtime host impl |
| `cdmurph32/wasi-webgpu-headers` | C bridge (webgpu.h → WIT) |
| `cdmurph32/graphtime` | WASI runtime with wasi:webgpu |

## Repository Separation

TrustMark-specific files belong in **this repo** (trustmark), not in onnxruntime-wasi.
`prepare_ort_build.sh` copies them before building — onnxruntime-wasi should not commit them.

### Files to remove from onnxruntime-wasi

These were checked in during development and must be cleaned up:

**`onnxruntime/wasm/` — delete, replaced by `prepare_ort_build.sh` copy:**
```
onnxruntime/wasm/simple.cpp
onnxruntime/wasm/image_utils.cpp
onnxruntime/wasm/image_utils.h
onnxruntime/wasm/stb_image.h
onnxruntime/wasm/stb_image_resize2.h
onnxruntime/wasm/stb_image_write.h
```

### Source of truth

| File | Canonical location |
|------|--------------------|
| `simple.cpp` (entry point) | `trustmark/cpp/examples/trustmark_wasm_image.cpp` |
| `image_utils.cpp/h` | `trustmark/cpp/wasm/` |
| `stb_image*.h` | `trustmark/cpp/wasm/` |
| ONNX Runtime WASI patches | `onnxruntime-wasi` (in git, correct) |
| WASI build scripts (`build_wasi.sh`) | `onnxruntime-wasi` (in git, correct) |

## Troubleshooting

**`WASI_SDK_PATH` not set**
```bash
export WASI_SDK_PATH=/opt/wasi-sdk
```

**`cannot use 'try' with exceptions disabled`**
Remove all `try-catch` blocks. Use `-fno-exceptions` compatible error handling.

**`ONNX format model is not supported`**
Load `.ort` files, not `.onnx`. Convert with:
```bash
cd onnxruntime-wasi/tools/python
python3.11 convert_onnx_models_to_ort.py ../../../models --output_dir ../../../models
```
For WebGPU, use the `with_runtime_opt.ort` variants (see Step 2).

**`No such file or directory` at runtime**
Check `--dir` mapping. Guest path (left of `::`) must match what the WASM program passes to `fopen`.

**Model loads but output is garbage (near-zero values)**
Wrong build — missing TrustMark operators. Rebuild with the complete operator config.

## WASI vs Browser WebAssembly

| Feature | WASI (This Build) | Browser WASM |
|---------|-------------------|--------------|
| Runtime | graphtime, wasmtime | Web browser |
| GPU | wasi:webgpu (host wgpu) | WebGPU (browser) |
| File access | WASI filesystem | Virtual FS |
| Model format | `.ort` | `.onnx` or `.ort` |
| Use case | Server, CLI, edge | Web applications |

## References

- [WASI SDK](https://github.com/WebAssembly/wasi-sdk)
- [ONNX Runtime](https://github.com/cdmurph32/onnxruntime)
- [wasi-gfx-runtime](https://github.com/cdmurph32/wasi-gfx-runtime)
- [graphtime](https://github.com/cdmurph32/graphtime)
- [wasi-webgpu-headers](https://github.com/cdmurph32/wasi-webgpu-headers)
- [ORT Format Models](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
