# TrustMark (Hailo-8L port)

This is a port of TrustMark's encoder/decoder to run on a Hailo-8L NPU (Raspberry Pi 5
AI Kit), for real-time watermark detection from a live camera feed. It implements the
same underlying watermarking method as `../python` - see that directory's README for
the paper and citation - but the encoder/decoder neural networks run on Hailo-8L
hardware (or, for CPU-only testing, via onnxruntime) instead of PyTorch.

## Overview

`trustmark_hailo.TrustMark` mirrors `trustmark.TrustMark`'s interface (construct with a
`model_type`, call `.encode()`/`.decode()`), so code written against the real thing
needs minimal changes to run against this port. The main differences:

- No watermark remover - only `encode()`/`decode()`/`localize()` (bounding-box detection
  is supported for variants `Q` and `P`, same as upstream).
- No PyTorch dependency at all - `numpy` and `pillow` are the only runtime dependencies,
  including for bounding-box detection (see [Limitations](#limitations)).
- Due to lower precision approximations of the model trained for the NPU, the PSNR is 
  considerably lower e.g. -5 on the encoders.  Please consider running the CPU vs. the NPU encoder
  unless your use case requires fast encoding.

Model files are not packaged in this repository due to their size, but are downloaded
upon first use, same as `../python` does - see [Model loading](#model-loading) below.

## Installation

### Prerequisite

You must have Python 3.9 or higher. On the Raspberry Pi 5, the NPU backend also needs
HailoRT's Python bindings (`sudo apt install hailo-all`), which install into the system
Python - not into a virtualenv/conda env - so make that visible to your env if you're
using one (e.g. by symlinking `hailo_platform` into your env's `site-packages`).

### Installing

After cloning the repository, install from the `python-hailo` directory:

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

You'll see output like this (note the PSNR drop for the NPU encoder vs the main repo):

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

Press `q` in the preview window to quit. Expect roughly 0.5-0.7 fps for
detection - see [Limitations](#limitations) for why.

### Example script

```python
from trustmark_hailo import TrustMark
from PIL import Image

# init - encoder_backend/decoder_backend default to 'npu' (except variant P's encoder,
# which defaults to 'cpu' - see Model variants below)
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
files are fetched over HTTP and cached (with MD5 verification) in
`trustmark_hailo/models/`, inside the installed package - see
`MODEL_REMOTE_HOST`/`MODEL_CHECKSUMS`/`check_and_download()` in
`trustmark_hailo/trustmark_hailo.py`. `MODEL_REMOTE_HOST` currently points at a local
test server, not real hosting yet - update it once the compiled models are actually
published. The small `secret2image_{TYPE}.npz` files (~300KB each, vs multi-MB for
everything else) are the exception - they ship directly in `trustmark_hailo/models/`
as package data, not downloaded.

Each `model_type` (`C`, `Q`, `B`, or `P`) needs a subset of these files, named
`{name}_{TYPE}.{ext}`:

| File pattern | Purpose | Needed for | How it gets there |
|---|---|---|---|
| `encoder_{TYPE}.hef` | encoder weights (NPU backend) | `encoder_backend='npu'` | downloaded on first use |
| `secret2image_{TYPE}.npz` | encoder weights (NPU backend, small CPU-side piece) | `encoder_backend='npu'` | ships with the package |
| `encoder_{TYPE}.onnx` | encoder weights (CPU backend) | `encoder_backend='cpu'` | downloaded on first use |
| `decoder_{TYPE}.hef` | decoder weights (NPU backend) | `decoder_backend='npu'` | downloaded on first use |
| `decoder_{TYPE}.onnx` | decoder weights (CPU backend) | `decoder_backend='cpu'` | downloaded on first use |
| `bbox_trunk_{TYPE}.hef` | bbox detector backbone+RPN weights (NPU) | `loadBBoxDetector=True` (`Q` only) | downloaded on first use |
| `box_head_{TYPE}.npz` | bbox detector weights (small CPU-side piece) | `loadBBoxDetector=True` (`Q` only) | downloaded on first use |

## Model variants

| `model_type` | Character | Trade-off |
|---|---|---|
| `'Q'` | Balanced (default) | Good quality and robustness |
| `'B'` | Base | Original paper model |
| `'C'` | Compact decoder | Smaller decoder model |

## Data schema

Same 100-bit payload and BCH error-correction schema as `../python` - see that
directory's README for details. Always call `tm.schemaCapacity()` to get the exact
capacity for the chosen encoding; don't hardcode it.

## decode options

```python
tm.decode(
    img,
    MODE='binary',        # 'binary' (bit string) or 'text' (7-bit ASCII string)
    DETECTFIRST=False,    # True = run bbox detector first (requires loadBBoxDetector=True); Q/P only
    ROTATION=False,        # True = try 0/90/180/270 rotations - combines with DETECTFIRST, same as upstream
)
# returns: (secret_string, wm_present: bool, wm_schema: int)
# wm_present=False -> no watermark detected; secret_string will be ''
```

## Bounding-box detection

Supported for variant `Q` only. Requires `loadBBoxDetector=True at construction time:

```python
tm = TrustMark(model_type='Q', loadBBoxDetector=True)

# locate a watermarked region before decoding (e.g. a cropped/composited image)
wm_secret, wm_present, wm_schema = tm.decode(img, MODE='binary', DETECTFIRST=True)

# or call the detector directly
box = tm.localize(img)               # top detection, normalized [x1,y1,x2,y2] in [0,1], or None
boxes = tm.localize(img, return_all=True)  # all detections
```

No confidence threshold is applied - matches upstream, which returns the top-scoring
detection unconditionally whenever the detector produces any box at all.

## encode options

```python
tm.encode(
    cover_image,          # PIL image, any resolution
    string_secret,        # bit string (MODE='binary') or 7-bit ASCII string (MODE='text')
    MODE='binary',
    WM_STRENGTH=1.0,       # increase for stronger watermark, at cost of visual quality
)
# returns: PIL image (RGB), same resolution as input
```

## License

This package including its models is distributed under the terms of the [MIT license](https://github.com/adobe/trustmark/blob/main/LICENSE), same as the rest of this repository.
