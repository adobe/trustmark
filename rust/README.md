# TrustMark — Rust implementation

<div style={{display: 'none'}}>

An implementation in Rust of TrustMark watermarking, as described in [**TrustMark - Universal Watermarking for Arbitrary Resolution Images**](https://arxiv.org/abs/2311.18297) (`arXiv:2311.18297`) by [Tu Bui](https://www.surrey.ac.uk/people/tu-bui)[^1], [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)[^2], and [John Collomosse](https://www.collomosse.com)[^1] [^2].

[^1]: [DECaDE](https://decade.ac.uk/) Centre for the Decentralized Digital Economy, University of Surrey, UK.

[^2]: [Adobe Research](https://research.adobe.com/), San Jose, CA.

</div>

This crate implements a subset of the functionality of the TrustMark Python implementation, including encoding and decoding of watermarks for all variants in binary mode. The Rust implementation provides the same levels of error correction as the Python implementation.

Text mode watermarks are not implemented. Watermark removal is available through
the explicitly loaded `WatermarkRemover` and the CLI `remove` command.

Open an issue if there's something in the Python version that want added to this crate!

## Platform Support

Building this crate requires Rust 1.88 or newer. TrustMark 0.4.0 uses [ort](https://ort.pyke.io) 2.0.0-rc.10 as its ONNX runtime, so it can only be used on platforms supported by that crate. See [the documentation](https://ort.pyke.io/setup/platforms) for supported platforms.

In particular, the ONNX runtime requires either:

* A recent version of Windows 10/11 & Visual Studio 2022 (≥ 17.11)
* glibc ≥ 2.35 & libstdc++ >= 12 (Ubuntu ≥ 22.04, Debian ≥ 12 ‘Bookworm’)
* macOS ≥ 10.15

## Quick start

### Download models

In order to encode or decode watermarks, you'll need to fetch the model files. The models are distributed as ONNX files.

From the workspace root (the `rust/` directory), run:

```
cargo xtask fetch-models
```

This command downloads models to the `models/` directory. You can move them from there as needed.

For watermark removal with P or Q, download the published remover ONNX models:

```sh
cargo xtask fetch-removers
```

This downloads `remover_P.onnx` and `remover_Q.onnx` from the same CDN as the
encoder/decoder models. Python is not required to download or use them. Both
download commands refuse to overwrite existing files.

### Export removal models (optional)

The remover loads a separate `remover_{variant}.onnx` model. To regenerate these
files from the checksum-verified Python checkpoints, use the Python-owned
[exporter](../python/export-remover.py). In a Python environment with this
repository's Python dependencies, also install `onnx` and `onnxruntime`, then
run from `rust/`:

```sh
/path/to/venv/bin/python ../python/export-remover.py --variant Q
```

Omit `--variant` to export B, C, P and Q. This creates `models/remover_Q.onnx`
(or the selected variants), checks the graph, and verifies ONNX/PyTorch inference
parity before publishing each file. Existing ONNX files are never overwritten.
Checkpoints are reused from `python/trustmark/models/`. The Python environment is
needed only for export and reference tests, not for Rust inference. P and Q can
use the published downloads above; export B/C locally if those variants are needed.

### Run the CLI

As a first step, you can run the `trustmark-cli` which is defined in this repository.

From the workspace root, run:

```sh
cargo run --release -p trustmark-cli -- -m ./models encode -i ../images/ghost.png -o ../images/encoded.png
cargo run --release -p trustmark-cli -- -m ./models decode -i ../images/encoded.png
cargo run --release -p trustmark-cli -- -m ./models remove -i ../images/encoded.png -o ../images/recovered.png --variant Q
```

The argument to the `-m` option is the path to the models downloaded; if you moved them, pass the relative file path as the option value.

### Use the library

Add `trustmark` to your project's `cargo` manifest with:

```
cargo add trustmark
```

This installs the version available on crates.io; TrustMark 0.4.0 contains the ORT rc.10 change.

A basic example of using `trustmark` is:

```rust
use trustmark::{Trustmark, Version, Variant};

let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
let input = image::open("../images/ghost.png").unwrap();
let output = tm.encode("0010101".to_owned(), input, 0.95);
```

`encode` and `decode` take `&self`, so a loaded model can be shared between callers. Inference on the same encoder or decoder session is serialized; encoding and decoding use separate sessions and can run concurrently.

To remove a watermark, load only the remover for the variant used to embed it:

```rust
use trustmark::{WatermarkRemover, Variant};

let remover = WatermarkRemover::new("./models", Variant::Q)?;
let input = image::open("watermarked.png")?;
let recovered = remover.remove(input)?;
recovered.to_rgba8().save("recovered.png")?;
```

Removal preserves image dimensions and alpha, processes RGB at 8-bit precision
like the Python implementation, supports all four model variants,
and uses Python's default strength (1.0, or 1.25 for P). Images without a detectable
watermark can also be processed, but may still change. Removal is lossy: it does
not guarantee the original pixels or erase arbitrary third-party watermarks.
The CLI writes PNG or JPEG (`--quality 90` by default for JPEG); the library
accepts decoded `DynamicImage` values. Missing or incompatible remover models
return errors. A shared remover serializes inference independently of encoders
and decoders. Existing callers incur no removal-model startup or memory cost.
The library returns a floating-point RGB/RGBA image; convert it to 8-bit pixels
before saving PNG/JPEG. The CLI performs this conversion and replaces the output
only after encoding succeeds. `--quality` is accepted only for JPEG output.

## Running the benchmarks

### Rust benchmarks

To run the Rust benchmarks, run the following from the workspace root:

```
cargo bench --bench load --bench encode
```

After exporting the remover models, use `cargo bench --bench remove` to measure
removal separately. Run real-model removal and re-watermarking tests with:

```sh
cargo test --test removal -- --ignored
```

These tests are opt-in because they require the exported remover models. The
normal test suite covers residual interpolation, cropping, alpha preservation,
output clamping, invalid model layouts and the CLI without those exports.
The dedicated removal CI job downloads P/Q and exports B/C, runs the real-model tests,
and compares native and Python output on RGB/RGBA images across tiny, odd-sized,
wide and tall dimensions, including both sides of the square-crop threshold.

Run the same parity suite locally after building the CLI:

```sh
cargo build -p trustmark-cli
/path/to/venv/bin/python ../python/test-rust-removal.py
```

The Python-owned [parity test](../python/test-rust-removal.py) includes the
reference loader and output checker. It permits small raster-decoder/resizer
differences: at most four 8-bit RGB levels per channel and half a level on average.
Use `--variant Q` to limit the suite to one variant, or `--cache` to reuse
checkpoints in another directory. Model export parity is checked separately on
float32 tensors with tighter tolerances.

### Python benchmarks

To run the Python benchmarks, run the following from the workspace root:

```
benches/load.sh && benches/encode.sh
```

## Version 0.5.0 compatibility

Watermark removal adds `Error::RemoverModel` and `Error::InvalidRemoverModel` to
the public `Error` enum. This is a breaking change for exhaustive matches, so
the crate uses a 0.x minor version bump rather than a patch release. Update
exhaustive matches to handle the new variants (or use a wildcard arm). Existing
encode/decode behavior and model loading remain unchanged. These removal APIs
require version 0.5.0; they are not available in the previously published 0.4.0.

## Releasing

Changes to this crate are validated by the [Rust CI](../.github/workflows/rust-ci.yml)
workflow, and releases to crates.io are performed by the
[Rust release](../.github/workflows/rust-release.yml) workflow when a
`rust-v*` tag is pushed.

See [RELEASING.md](RELEASING.md) for the full process, including the manual
fallback and the setup required on crates.io. Notable changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
