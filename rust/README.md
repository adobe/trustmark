# TrustMark — Rust implementation

<div style={{display: 'none'}}>

An implementation in Rust of TrustMark watermarking, as described in [**TrustMark - Universal Watermarking for Arbitrary Resolution Images**](https://arxiv.org/abs/2311.18297) (`arXiv:2311.18297`) by [Tu Bui](https://www.surrey.ac.uk/people/tu-bui)[^1], [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)[^2], and [John Collomosse](https://www.collomosse.com)[^1] [^2].

[^1]: [DECaDE](https://decade.ac.uk/) Centre for the Decentralized Digital Economy, University of Surrey, UK.

[^2]: [Adobe Research](https://research.adobe.com/), San Jose, CA.

</div>

This crate implements a subset of the functionality of the TrustMark Python implementation, including encoding and decoding of watermarks for all variants in binary mode. The Rust implementation provides the same levels of error correction as the Python implementation.

Text mode watermarks and watermark removal are not implemented.

Open an issue if there's something in the Python version that want added to this crate!

## Platform Support

Building this crate requires Rust 1.88 or newer. TrustMark 0.3.0 uses [ort](https://ort.pyke.io) 2.0.0-rc.12 as its ONNX runtime, so it can only be used on platforms supported by that crate. See [the documentation](https://ort.pyke.io/setup/platforms) for supported platforms.

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

### Run the CLI

As a first step, you can run the `trustmark-cli` which is defined in this repository.

From the workspace root, run:

```sh
cargo run --release -p trustmark-cli -- -m ./models encode -i ../images/ghost.png -o ../images/encoded.png
cargo run --release -p trustmark-cli -- -m ./models decode -i ../images/encoded.png
```

The argument to the `-m` option is the path to the models downloaded; if you moved them, pass the relative file path as the option value.

### Use the library

Add `trustmark` to your project's `cargo` manifest with:

```
cargo add trustmark
```

This installs the version available on crates.io; TrustMark 0.3.0 contains the ORT rc.12 upgrade.

A basic example of using `trustmark` is:

```rust
use trustmark::{Trustmark, Version, Variant};

let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
let input = image::open("../images/ghost.png").unwrap();
let output = tm.encode("0010101".to_owned(), input, 0.95);
```

`encode` and `decode` take `&self`, so a loaded model can be shared between callers. Inference on the same encoder or decoder session is serialized; encoding and decoding use separate sessions and can run concurrently.

## Running the benchmarks

### Rust benchmarks

To run the Rust benchmarks, run the following from the workspace root:

```
cargo bench
```

### Python benchmarks

To run the Python benchmarks, run the following from the workspace root:

```
benches/load.sh && benches/encode.sh
```

## Releasing

Changes to this crate are validated by the [Rust CI](../.github/workflows/rust-ci.yml)
workflow, and releases to crates.io are performed by the
[Rust release](../.github/workflows/rust-release.yml) workflow when a
`rust-v*` tag is pushed.

See [RELEASING.md](RELEASING.md) for the full process, including the manual
fallback and the setup required on crates.io. Notable changes are recorded in
[CHANGELOG.md](CHANGELOG.md).
