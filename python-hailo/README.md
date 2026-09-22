# TrustMark (Hailo-8L port)

This is a port of TrustMark's encoder/decoder to run on a Hailo-8L NPU (Raspberry Pi 5
AI Kit). Same underlying watermarking method as `../python` - see that directory's README for
the paper and citation - but the encoder/decoder neural networks run on Hailo-8L
hardware (or, for CPU-only testing, via onnxruntime) instead of PyTorch. To achieve this the
models have been quantized to INT8.

## Overview

`trustmark_hailo.TrustMark` mirrors `trustmark.TrustMark`'s interface (construct with a
`model_type`, call `.encode()`/`.decode()`), so code written against the real thing
needs minimal changes to run against this port. The main differences:

- No watermark remover - only `encode()`/`decode()`/`localize()` (bounding-box detection
  is supported for variants `Q` and `P` only).
- No PyTorch dependency - `numpy` and `pillow` are the only runtime dependencies.
- Due to lower precision approximations of the model trained for the NPU, the PSNR is 
  considerably lower e.g. -5 on the encoders.  Please consider running the CPU vs. the NPU encoder
  unless your use case requires fast encoding.  Note due to the subtle residual of the P encoder
  it has been quantized at higher precision when training the NPU model.

Model files are not packaged in this repository due to their size, but are downloaded
upon first use, same as `../python` does - see [Model loading](#model-loading) below.

## Installation

The repo is available via PyPi, using `pip install trustmark-hailo`.

Or after cloning this repository, install from the `python-hailo` directory:

```
cd trustmark/python-hailo
pip install .
```

## Quickstart

To get started quickly, run the `python-hailo/test.py` script, which watermarks
`images/ripley.jpg` (the same example image `../python/test-decode.py` uses) with
variant Q, decodes it back, and reports whether the exact secret was recovered.

### Run the example

```sh
cd trustmark/python-hailo
python test.py
```

You'll see output like this (note the PSNR on encoding via the NPU is lower than the main PyTorch implementation):

```
Initializing TrustMark (Hailo port) model_type='Q' encoder=[npu] decoder=[npu] models_dir=.../trustmark_hailo/models
Fetching model file (once only): .../trustmark_hailo/models/encoder_Q.hef
Fetching model file (once only): .../trustmark_hailo/models/secret2image_Q.npz
Fetching model file (once only): .../trustmark_hailo/models/decoder_Q.hef
Wrote ripley_Q_hailo.png
Extracted secret: 1000000100001110000010010001011110010001011000100000100110110 (schema 1)
Exact match: True
PSNR = 37.355618 
```

The "Fetching model file" lines only appear the first time - after that, the same
files are reused from `trustmark_hailo/models/` (verified by checksum on every run, same
as `../python`).

### Live camera demo

`test-camera.py` runs the bbox detector + decoder live against the Raspberry Pi
camera - draws a rectangle around every detection and logs to the console. Needs the Pi
camera stack on top of the base install: `picamera2` and `opencv` (both ship as
system packages on Raspberry Pi OS - `sudo apt install python3-picamera2
python3-opencv`, then make them visible to your env the same way as
`hailo_platform` above).

```sh
python test-camera.py
```

### Example script

```python
from trustmark_hailo import TrustMark
from PIL import Image

# init - encoder_backend/decoder_backend both default to 'npu' for every variant
tm = TrustMark(model_type='Q', verbose=True)

# encoding example
cover = Image.open('images/ripley.jpg').convert('RGB')
secret = '0' * tm.schemaCapacity()   # a real secret would come from your own payload
stego = tm.encode(cover, secret, MODE='binary')
stego.save('ripley_Q.png')

# decoding example
stego = Image.open('ripley_Q.png').convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary')

if wm_present:
    print(f'Extracted secret: {wm_secret}')
else:
    print('No watermark decoded')

tm.close()
```

## Model loading

Similar behavior to `../python`, with one difference: on first use, the larger model
files are fetched over HTTP and cached (with MD5 verification)

Each `model_type` (`C`, `Q`, `B`, or `P`) needs a subset of these files, named
`{name}_{TYPE}.{ext}`:

| File pattern | Purpose | Needed for | How it gets there |
|---|---|---|---|
| `encoder_{TYPE}.hef` | encoder weights (NPU backend) | `encoder_backend='npu'` | downloaded on first use |
| `secret2image_{TYPE}.npz` | encoder weights (NPU backend, small CPU-side piece) | `encoder_backend='npu'` | ships with the package |
| `encoder_{TYPE}.onnx` | encoder weights (CPU backend) | `encoder_backend='cpu'` | downloaded on first use |
| `decoder_{TYPE}.hef` | decoder weights (NPU backend) | `decoder_backend='npu'` | downloaded on first use |
| `decoder_{TYPE}.onnx` | decoder weights (CPU backend) | `decoder_backend='cpu'` | downloaded on first use |
| `bbox_trunk_{TYPE}.hef` | bbox detector backbone+RPN weights (NPU) | `loadBBoxDetector=True` (`Q`/`P` only) | downloaded on first use |
| `box_head_{TYPE}.npz` | bbox detector weights (small CPU-side piece) | `loadBBoxDetector=True` (`Q`/`P` only) | downloaded on first use |

## Model variants

| `model_type` | Character | Trade-off |
|---|---|---|
| `'Q'` | Balanced (default) | Good quality and robustness |
| `'P'` | High visual quality | Best PSNR, forces centre-square crop. |
| `'B'` | Base | Original paper model |
| `'C'` | Compact decoder | Smaller decoder model |

## Data schema

Same 100-bit payload and BCH error-correction schema as `../python` - see that
directory's README for details. Always call `tm.schemaCapacity()` to get the exact
capacity for the chosen encoding; don't hardcode it.

## License

This package including its models is distributed under the terms of the [MIT license](https://github.com/adobe/trustmark/blob/main/LICENSE), same as the rest of this repository.
