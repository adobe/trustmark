# Configuring TrustMark (Hailo-8L port)

## Overview

All watermarking algorithms trade off between three properties:

- **Capacity (bits)**
- **Robustness (to various transformations)**
- **Visibility (of watermark)**

This document explains how to configure `trustmark_hailo.TrustMark` to tune these
properties. It mirrors `../python/CONFIG.md`, but only covers the parameters 
available in this port - see [Differences from the PyTorch port](#differences-from-the-pytorch-port).
The default configuration (variant Q, 100% strength, BCH_5 error correction) is sufficient for most use cases.

## Model variant

Just like the PyTorch port, this port has four model variants (**B**, **C**, **P**, and
**Q**) selected via `model_type` when instantiating `TrustMark`. All encode/decode calls
on the object use this variant.

In general, we recommend using **P** or **Q**:
- **P** is useful for creative applications where very high visual quality is required.
- **Q** is a good all-rounder and is the default.

> **Note:** Images encoded with one model variant cannot be decoded with another.
> Images encoded with the CPU-only PyTorch port (`../python`) also cannot be decoded
> here (or vice versa) - the NPU/onnxruntime models are separately trained/quantized
> weights, not just a different runtime for the same weights.

| Variant | Bbox detector | Description |
|---------|----------------|-------------|
| **Q**   | yes | Default (**Q**uality). Good trade-off between robustness and imperceptibility. |
| **B**   | no  | (**B**eta). Very similar to Q, included mainly for reproducing the paper. |
| **C**   | no  | (**C**ompact). Smaller decoder model. Slightly lower visual quality. |
| **P**   | yes | (**P**erceptual). Highest visual quality; forces centre-square crop (see below). Encoder is quantized at higher precision than the others to preserve its subtle residual. |

Due to INT8 quantization for the NPU, PSNR on this port is roughly 5 dB lower than the
equivalent PyTorch model on `encoder_backend='npu'`. If your use case tolerates slower
encoding, `encoder_backend='cpu'` (onnxruntime, no quantization) gets you back to
PyTorch-comparable PSNR - see the backend selection section below.

## Backend selection (NPU vs CPU)

Unique to this port: encoder and decoder each independently choose which runtime backend
executes their neural network, via the `encoder_backend`/`decoder_backend` constructor
arguments:

```python
tm = TrustMark(model_type='Q', encoder_backend='npu', decoder_backend='npu')
```

- `'npu'` (default for both) - runs on Hailo-8L hardware via HailoRT, using the `.hef`
  model files. Fast, but INT8-quantized (lower PSNR on encoding, see above).
- `'cpu'` - runs via onnxruntime, using the `.onnx` model files. No Hailo hardware
  required (useful for development on a regular machine), full floating-point
  precision, but slower.
- `None`/`False` - don't load that side at all (e.g. `decoder_backend=None` if you only
  ever call `encode()` on this instance). Calling the unloaded method later raises
  `RuntimeError`.

## Watermark strength

Set the optional `WM_STRENGTH` parameter when encoding (at runtime). Its default value
is `1.0`, and changing it provides a trade-off between **robustness** and **visibility**,
same as the PyTorch port:

- Raising its value (for example, to 1.5) improves robustness but increases the
  likelihood of ripple artifacts.
- Lowering its value (for example, to 0.8) reduces any likelihood of artifacts but
  compromises on robustness.

```python
encoded_image = tm.encode(cover_image, secret, MODE='binary', WM_STRENGTH=1.5)
```

## Error correction level

TrustMark encodes a payload (the watermark data embedded within the image) of 100 bits.
The data schema implemented in `trustmark_hailo/datalayer.py` - byte-for-byte the same
schema as `../python/trustmark/datalayer.py` - enables you to choose an error correction
level over the raw 100 bits of payload to maintain reliability under transformations or
noise.

### Encoding modes

Set the error correction level using one of the four encoding modes:

| Encoding | Protected payload | Number of bit flips allowed |
|----------|-------------------|-----------------------------|
| `Encoding.BCH_5` | 61 bits (+ 35 ECC bits) | 5 |
| `Encoding.BCH_4` | 68 bits (+ 28 ECC bits) | 4 |
| `Encoding.BCH_3` | 75 bits (+ 21 ECC bits) | 3 |
| `Encoding.BCH_SUPER` | 40 bits (+ 56 ECC bits) | 8 |

Specify the mode when you instantiate `TrustMark`:

```python
tm = TrustMark(verbose=True, model_type='Q', encoding_type=TrustMark.Encoding.BCH_5)
```

Where the constant is `BCH_5`, `BCH_4`, `BCH_3`, or `BCH_SUPER`.

The decoder automatically detects the data schema in a watermark, so you can choose the
level of robustness that best suits your use case.

> **Note:** `use_ECC=False` (raw, uncorrected payload bits) is accepted as a constructor
> argument for interface parity with the PyTorch port, but raises `NotImplementedError`
> in this port - only ECC-protected payloads are supported here.

## Bounding-box detection (`loadBBoxDetector`)

Pass `loadBBoxDetector=True` when constructing `TrustMark` (variants **Q**/**P** only -
the bbox detector's trained weights aren't published for **C**/**B**, same restriction as
upstream) to enable `tm.localize()` and `tm.decode(..., DETECTFIRST=True)`:

```python
tm = TrustMark(model_type='Q', loadBBoxDetector=True)
box = tm.localize(stego_image)                       # normalized [x1,y1,x2,y2] or None
secret, present, schema = tm.decode(stego_image, MODE='binary', DETECTFIRST=True)
```

The bbox trunk always runs on the NPU (there's no `bbox_backend='cpu'` option in this
port).

## Rotation-invariant decoding

Same as upstream: pass `ROTATION=True` to `decode()` to also try 90/180/270 degree
rotations of the image before giving up:

```python
secret, present, schema = tm.decode(stego_image, MODE='binary', ROTATION=True)
```

## Differences from the PyTorch port

A few knobs documented in `../python/CONFIG.md` are **not** configurable in this port -
they're fixed at their PyTorch-port defaults internally:

- **Center cropping (`ASPECT_RATIO_LIM`)** - still applied automatically for extreme
  aspect ratios (limit 2.0), and still forced to always-center-crop for variant **P**,
  but there's no constructor/runtime argument to override the threshold here.
- **`CONCENTRATE_WM_REGION` (zero padding)** - not exposed; always effectively 100% (no
  concentration), which is the recommended setting on the PyTorch port anyway.
- **`WM_MERGE` (upscaling interpolation mode)** - not exposed; bilinear is always used
  when scaling the 256x256 residual back to the original resolution.
- **Watermark remover** - not implemented in this port at all (no `remove_watermark()`,
  no `loadRemover` argument). Only `encode()`/`decode()`/`localize()` are available.

## License

This package including its models is distributed under the terms of the [MIT license](https://github.com/adobe/trustmark/blob/main/LICENSE), same as the rest of this repository.
