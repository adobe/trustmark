# Changelog

All notable changes to the `trustmark` Rust crate are documented here. This
crate is versioned independently of the Python package in this repository.

Entries for releases before 0.3.0 were reconstructed from the commit history,
since those versions were published manually and without tags.

## 0.3.0

- Upgrade `ort` to 2.0.0-rc.12. This is the fix for 0.2.2 no longer being
  buildable: the prebuilt ONNX Runtime binaries that `ort-sys` 2.0.0-rc.8
  downloads at build time are no longer hosted.
- Upgrade `ndarray` to 0.17.
- `encode` and `decode` now take `&self`, so a loaded model can be shared
  between callers. Encoding and decoding use separate sessions and can run
  concurrently; inference on a single session is serialized.
- Raise the minimum supported Rust version to 1.88.
- Add release automation and this changelog; see [`RELEASING.md`](RELEASING.md).

## 0.2.2

- Update the model download URL used by `cargo xtask fetch-models`.
- Document platform support.

## 0.2.1

- Fix an integer truncation bug.
- Fix the documentation build.

## 0.2.0

- Switch image resizing from `image` to `fast_image_resize` for better
  performance.
- Support a user-specified JPEG output quality.
- Save JPEG output as RGB instead of RGBA.
- Fix resizing when the target is larger than the source image.

## 0.1.0

- Initial release: encoding and decoding of TrustMark watermarks in binary mode
  for all variants, with the same error correction levels as the Python
  implementation.
