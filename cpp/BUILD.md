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

## First-time setup

```bash
# Initialise submodule
git submodule update --init cpp/onnxruntime-wasi

# Stage TrustMark sources into the onnxruntime checkout
bash cpp/prepare_ort_build.sh

# Build ORT (once; ~20 min)
export WASI_SDK_PATH=/opt/wasi-sdk
cd cpp/onnxruntime-wasi && ./build_wasi.sh Release \
  -Donnxruntime_USE_WEBGPU=ON \
  -Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=ON \
  -Donnxruntime_EXTENDED_MINIMAL_BUILD=ON \
  -Donnxruntime_WGSL_TEMPLATE=static && cd -
```

## Convert models

Models must be in `.with_runtime_opt.ort` format and placed in `cpp/models/`:

```bash
cd cpp/onnxruntime-wasi
python3 tools/python/convert_onnx_models_to_ort.py \
  ../../models/encoder_P.onnx \
  ../../models/decoder_P.onnx
cp *.with_runtime_opt.ort ../../models/
```

## Build component

```bash
export WASI_SDK_PATH=/opt/wasi-sdk

# Image watermarking component → cpp/build_http/trustmark-http.wasm
bash cpp/build_wasm_http.sh
```

## Run with wasmCloud

Each demo in `demo/` is a self-contained `wash dev` session.

```bash
# Build wash from wasmCloud/wasmCloud main
cd /path/to/wasmCloud && cargo build --bin wash
WASH=/path/to/wasmCloud/target/debug/wash

# Image watermarking — CPU (port 8000) or GPU (port 8001)
cd demo/trustmark-cpu && $WASH dev
cd demo/trustmark-gpu && $WASH dev
```

**CPU vs GPU**: same `.wasm` binary, both modes. GPU enabled by `USE_WEBGPU=1` in
`environment:` of the GPU demo config. Without it ORT falls back to CPU automatically.

WebGPU requires a wasmCloud build that implements `wasi:webgpu` backed by the host
GPU (Metal/Vulkan/DX12). This support is in the `wasmCloud/wasmCloud` main branch.

Both demos mount `cpp/models/` at `/models` inside the component.

## Test

```bash
IMAGE=/path/to/trustmark/images/ufo_240.jpg

# Encode watermark (CPU, port 8000)
curl -X POST http://localhost:8000/encode \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked.png

# Encode watermark (GPU, port 8001)
curl -X POST http://localhost:8001/encode \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked_gpu.png

# Encode with custom 100-bit watermark
AID=1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010
curl -X POST "http://localhost:8001/encode?bits=$AID" \
  -H 'Content-Type: image/jpeg' --data-binary @"$IMAGE" -o watermarked_custom.png
```

## Models

Models are not in this repository. Place the following `.ort` files in `cpp/models/`:

- `encoder_P.with_runtime_opt.ort`
