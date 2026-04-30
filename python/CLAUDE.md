# TrustMark — LLM Agent Guide

## Install

The installable Python package lives in this `python/` subdirectory, not the repo root.

```bash
# End-user (preferred)
pip install trustmark

# Development / editable
pip install -e python/   # run from repo root
# or
pip install -e .         # run from inside python/
```

## Minimal working example

```python
from trustmark import TrustMark
from PIL import Image

tm = TrustMark(model_type='Q', loadRemover=False)  # skip remover download for encode/decode only

capacity = tm.schemaCapacity()          # always query — do not hardcode a bit length
secret = ...                            # binary string of exactly `capacity` bits

cover = Image.open('input.jpg').convert('RGB')
watermarked = tm.encode(cover, secret, MODE='binary')
watermarked.save('output.png')

wm_secret, wm_present, wm_schema = tm.decode(watermarked, MODE='binary')
```

**`MODE` must match between `encode` and `decode`.** Mismatching them produces silently
wrong results — no error is raised.

Two modes are available:

- `MODE='binary'` — payload is a bit string (`'0'` and `'1'` characters) of exactly
  `schemaCapacity()` bits.
- `MODE='text'` — payload is a 7-bit ASCII string. Each character occupies 7 bits, so
  the usable capacity is `schemaCapacity() // 7` characters. Non-ASCII (Unicode, bytes)
  is not supported in this mode.

The default is `MODE='text'`. When working with raw bits or arbitrary binary data always
pass `MODE='binary'` explicitly on both calls.

Text mode example:

```python
tm = TrustMark(model_type='Q', loadRemover=False)
max_chars = tm.schemaCapacity() // 7      # e.g. 61 // 7 = 8 characters for BCH_5
secret_text = 'hello'                     # must be 7-bit ASCII, <= max_chars

watermarked = tm.encode(cover, secret_text, MODE='text')
wm_secret, wm_present, wm_schema = tm.decode(watermarked, MODE='text')
```

## Model loading — what gets downloaded and when

On first use, model weights are fetched from Adobe's CDN and cached locally inside the
package directory. Three separate models exist; not all are needed for every use case:

| Model | Constructor arg | Needed for | Approx size |
|---|---|---|---|
| Encoder + Decoder | always loaded | `encode`, `decode` | moderate |
| Remover | `loadRemover=True` (default) | `remove_watermark` | large |
| BBox detector | `loadBBoxDetector=False` (default) | `decode(..., DETECTFIRST=True)` | large |

**Skip the remover download** if you only need to embed or verify watermarks:

```python
tm = TrustMark(model_type='Q', loadRemover=False)
```

**Load the bbox detector** only when you need to locate a watermarked region before
decoding (e.g. a cropped or composited image):

```python
tm = TrustMark(model_type='Q', loadRemover=False, loadBBoxDetector=True)
wm_secret, wm_present, wm_schema = tm.decode(img, MODE='binary', DETECTFIRST=True)
```

## Model variants

| `model_type` | Character | Trade-off |
|---|---|---|
| `'Q'` | Balanced (default) | Good quality and robustness |
| `'P'` | High visual quality | Best PSNR, forces centre-square crop |
| `'B'` | Base | Original paper model |
| `'C'` | Compact decoder | Smaller decoder model |

## ECC encoding types and bit capacity

The `encoding_type` parameter controls error-correction strength and usable payload size.
Pass via `TrustMark.Encoding.*`:

| Encoding | Constant | Payload bits | ECC strength |
|---|---|---|---|
| BCH_5 | `Encoding.BCH_5` | 61 bits | strongest (default) |
| BCH_4 | `Encoding.BCH_4` | 68 bits | moderate |
| BCH_3 | `Encoding.BCH_3` | 75 bits | light |
| BCH_SUPER | `Encoding.BCH_SUPER` | 40 bits | maximum robustness |

Always call `tm.schemaCapacity()` after construction to get the exact capacity for the
chosen encoding — do not hardcode the number.

## decode options

```python
tm.decode(
    img,
    MODE='binary',        # 'binary' (bit string) or 'text' (7-bit ASCII string)
    DETECTFIRST=False,    # True = run bbox detector first (requires loadBBoxDetector=True)
    ROTATION=False,       # True = try 0/90/180/270° rotations
)
# returns: (secret_string, wm_present: bool, wm_schema: int)
# wm_present=False → no watermark detected; secret_string will be ''
```

## encode options

```python
tm.encode(
    cover_image,          # PIL image, any resolution
    string_secret,        # bit string (MODE='binary') or 7-bit ASCII string (MODE='text')
    MODE='binary',
    WM_STRENGTH=1.0,      # increase for stronger watermark, at cost of visual quality
    WM_MERGE='bilinear',  # interpolation mode when upscaling residual back to original res
)
# returns: PIL image (RGB), same resolution as input
```

## Watermark removal

Requires `loadRemover=True` (the default) at construction time. Calling
`remove_watermark` without the remover model loaded raises `ModelNotLoadedError`.

```python
tm = TrustMark(model_type='Q')   # loadRemover=True is the default
recovered = tm.remove_watermark(watermarked_image)
```
