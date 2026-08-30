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

## 16-bit PNG support

`PIL.Image.open` silently flattens a 16-bit-per-channel PNG to 8-bit on load — no
error, no signal anything was lost. Since no TrustMark model is trained on more than
8-bit input anyway, `tm.encode()` alone can never deliver output that's genuinely
16-bit: the source has already lost its extra precision by the time it reaches PIL.

`encode_high_bit_depth` solves this by reading the true 16-bit pixels directly (never
through PIL), running the real encoder on a throwaway 8-bit copy, and adding back only
the watermark's own perturbation onto the untouched full-precision pixels. It operates
on raw PNG bytes in and out — unlike every other method on this class, it does **not**
take or return a PIL image, since PIL can't represent the data being preserved.

Requires the optional extra:

```bash
pip install trustmark[highbitdepth]
```

```python
with open('input_16bit.png', 'rb') as f:
    raw_png_bytes = f.read()

watermarked_bytes = tm.encode_high_bit_depth(raw_png_bytes, secret, MODE='binary')

with open('output_16bit.png', 'wb') as f:
    f.write(watermarked_bytes)
```

- Supports plain RGB and RGBA 16-bit-per-channel PNGs only. Anything else (8-bit,
  grayscale, palette, non-PNG) raises `ValueError` — use `tm.encode()` instead for
  ordinary 8-bit sources.
- Raises `ImportError` with an install hint if `raw_png_bytes` is genuinely high-bit-depth
  but the required codec (`OpenImageIO` for RGB, `pypng` for RGBA) isn't installed.
- **There is no `decode_high_bit_depth`.** Decode watermarked 16-bit output with the
  ordinary `tm.decode()`, on the image as loaded (and silently flattened to 8-bit) by
  PIL — precision loss on read doesn't materially affect watermark *detection*, only
  the delivered pixel precision on encode, so no separate high-bit-depth decode path
  is needed:

```python
stego = Image.open('output_16bit.png').convert('RGB')  # flattened to 8-bit by PIL, that's fine
secret_out, wm_present, wm_schema = tm.decode(stego, MODE='binary')
```

## Watermark removal

Requires `loadRemover=True` (the default) at construction time. Calling
`remove_watermark` without the remover model loaded raises `ModelNotLoadedError`.

```python
tm = TrustMark(model_type='Q')   # loadRemover=True is the default
recovered = tm.remove_watermark(watermarked_image)
```
