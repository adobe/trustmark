# Building TrustMark for wasmCloud

TrustMark ships as a WASIP2 HTTP component that runs under wasmCloud:

| Component | Input | Output | Routes |
|-----------|-------|--------|--------|
| `trustmark-http.wasm` | JPEG or PNG image | Watermarked PNG | `POST /encode[?bits=<100-bit-string>]` |

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| WASI SDK | 24+ | Set `WASI_SDK_PATH=/opt/wasi-sdk` |
| CMake | 3.15+ | |
| Python | 3.11+ | `pip3 install --break-system-packages flatbuffers` |
| wasm-tools | any | `cargo install wasm-tools` |
| wasi-preview1-component-adapter-provider | any | `cargo install wasi-preview1-component-adapter-provider` |
| wash (wasmCloud) | wasmCloud/wasmCloud main | `cargo build --bin wash` in the repo |

All commands below run from the `cpp/` directory unless noted.

## First-time setup

```bash
# From repo root — initialise submodule
git submodule update --init cpp/onnxruntime-wasi

# Stage TrustMark sources into the onnxruntime checkout
bash prepare_ort_build.sh

# Build ORT (once; ~20 min)
export WASI_SDK_PATH=/opt/wasi-sdk
cd onnxruntime-wasi && ./build_wasi.sh Release \
  -Donnxruntime_USE_WEBGPU=ON \
  -Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=ON \
  -Donnxruntime_EXTENDED_MINIMAL_BUILD=ON \
  -Donnxruntime_WGSL_TEMPLATE=static && cd -
```

## Convert models

Models must be in `.with_runtime_opt.ort` format and placed in `models/`:

```bash
cd onnxruntime-wasi
python3 tools/python/convert_onnx_models_to_ort.py \
  ../models/encoder_P.onnx \
  ../models/decoder_P.onnx
cp *.with_runtime_opt.ort ../models/
cd -
```

## Build component

```bash
export WASI_SDK_PATH=/opt/wasi-sdk

# → build_http/trustmark-http.wasm
bash build_wasm_http.sh
```

## Run with wasmCloud

Each demo in `demo/` is a self-contained `wash dev` session.

```bash
# Build wash from wasmCloud/wasmCloud main
cd /path/to/wasmCloud && cargo build --bin wash
WASH=/path/to/wasmCloud/target/debug/wash

# CPU (port 8000) or GPU (port 8001)
cd demo/trustmark-cpu && $WASH dev
cd demo/trustmark-gpu && $WASH dev
```

**CPU vs GPU**: same `.wasm` binary, both modes. GPU is enabled by `USE_WEBGPU=1` in
`host_interfaces.config` of the GPU demo config; without it ORT uses CPU.

WebGPU requires a wasmCloud build that implements `wasi:webgpu` backed by the host
GPU (Metal/Vulkan/DX12). This support is in the `wasmCloud/wasmCloud` main branch.

Both demos mount `models/` at `/models` inside the component.

## Test

```bash
IMAGE=/path/to/trustmark/images/ufo_240.jpg

# Encode (CPU, port 8000)
curl -X POST http://localhost:8000/encode \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked.png

# Encode (GPU, port 8001)
curl -X POST http://localhost:8001/encode \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked_gpu.png

# Custom 100-bit watermark
AID=1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010
curl -X POST "http://localhost:8001/encode?bits=$AID" \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked_custom.png
```

## Models

Models are not in this repository. Place the following in `models/`:

- `encoder_P.with_runtime_opt.ort`
