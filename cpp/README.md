# TrustMark C++

This C++17 runner uses OpenCV for image loading, crop/resize, and writing, and
[ONNX Runtime](https://onnxruntime.ai/) for the TrustMark B, C, P, and Q encoder
and decoder models. Like the Rust and Python interfaces, it supports binary
payload encoding and decoding with BCH error correction. Text encoding and
watermark removal are not implemented.

## Build

Install an ONNX Runtime C/C++ distribution with headers and a shared library,
and OpenCV with CMake packages. From the repository root:

```sh
cmake -S cpp -B cpp/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build cpp/build
```

If CMake cannot find OpenCV, also pass `-DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4`.
Fetch ONNX models from the `rust/` directory with `cargo xtask fetch-models`
([instructions](../rust/README.md#download-models)).

## Encode and decode images

```sh
cpp/build/trustmark-runner encode-image rust/models Q images/ghost.png encoded.png \
  1011011110011000111111000000011111011111011100000110110110111 BCH_5
cpp/build/trustmark-runner decode-image rust/models Q encoded.png
```

The payload is a binary string of at most 40 (`BCH_SUPER`), 61 (`BCH_5`, the
default), 68 (`BCH_4`), or 75 (`BCH_3`) bits. Short payloads are padded with
zeros. The variant can be B, C, P, or Q. Optionally supply a strength in `[0,1]`
after the BCH version; the default is 0.95. The output is an image file; use
PNG for lossless roundtrips. The original alpha channel is preserved for RGBA
inputs. Decode prints `Found watermark (BCH_5): <payload bits>` when BCH
validation succeeds, or `Corrupt or missing watermark` otherwise.

OpenCV accepts 8-bit RGB/RGBA images here. The runner center-crops P images and
images with an aspect ratio above 2:1, resizes to the ONNX model's input size
(224 for P decoding, 256 otherwise), and normalizes RGB channels to `[-1,1]`.
For encoding, it scales and clamps the model residual before compositing it
onto the original image. OpenCV and the Rust/Python image resize routines can
produce slightly different pixel values. The Python PyTorch decoder currently
uses 245 for B/C/Q; the ONNX models used here and by Rust require 256.

## Raw model inference

The tensor interface remains available for callers with their own image
pipeline:

```sh
cpp/build/trustmark-runner inspect rust/models/decoder_Q.onnx
cpp/build/trustmark-runner decode rust/models/decoder_Q.onnx input.f32
cpp/build/trustmark-runner encode rust/models/encoder_Q.onnx input.f32 "$BITS100" output.f32
```

`input.f32` is a native-endian float32 RGB tensor in NCHW order, normalized to
`[-1,1]`. Raw `decode` prints 100 unvalidated bits. Raw `encode` takes a full
100-bit packet including BCH parity and schema bits, and writes the model's
float32 output tensor. Use `encode-image`/`decode-image` for BCH and image I/O.

To run the BCH checks and image inference smoke test:

```sh
ctest --test-dir cpp/build --output-on-failure
python3 cpp/tests/smoke.py cpp/build/trustmark-runner rust/models
```

To verify interoperability in both directions with the Rust CLI for B, C, P,
and Q on square PNG and JPEG inputs, plus P and Q on an image wider than 2:1
(writing lossless PNG outputs):

```sh
cargo build --manifest-path rust/Cargo.toml -p trustmark-cli
python3 cpp/tests/interoperability.py cpp/build/trustmark-runner \
  rust/target/debug/trustmark rust/models
```

The Rust implementation does not reliably decode its own B/C watermarks on
very wide images, so those cases are not used as cross-implementation fixtures.
