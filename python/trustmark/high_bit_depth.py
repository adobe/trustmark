# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Bypasses PIL for images with real per-channel precision above 8 bits.

`PIL.Image.open` silently flattens a 16-bit RGB PNG to 8-bit `"RGB"` mode on load -- no
error, no signal anything was lost (Pillow can't even represent a `(H, W, 3)` uint16
array -- `Image.fromarray` raises `TypeError` on one). TrustMark itself is only ever
trained on 8-bit input regardless, so this module exists purely to read/write pixels at
their native bit depth around the model -- see `TrustMark.encode_high_bit_depth` in
`trustmark.py` for how the watermark itself gets applied without losing that precision.

`read_high_bit_depth_rgb`/`read_high_bit_depth_rgba` return `None` for anything that
isn't a genuine 16-bit-per-channel PNG of the matching channel count -- callers should
treat that as "not applicable here", not an error.

Two separate codecs are used, and neither is a hard dependency of the base package --
both are lazily imported only once a PNG header confirms they're actually needed, so
`import trustmark` never requires either. Install both with `pip install
trustmark[highbitdepth]`.

- RGB uses OpenImageIO, which can read/write native 16-bit and carry forward full
  source color metadata (ICC profile, gamma, etc.) via `copy_metadata`. Its Python
  binding has no in-memory I/O path, so reads/writes here go through a temp file.
- RGBA uses `pypng` (pure Python) instead of OpenImageIO: OpenImageIO's PNG writer
  unconditionally premultiplies RGB by alpha on write for any 4-channel PNG, with no
  attribute that disables it -- PNG itself only ever stores straight (unassociated)
  alpha, so that would be non-conformant output. `pypng` does exact, literal
  sample-value round trips instead, natively over in-memory bytes, at the cost of only
  carrying the PNG `gAMA` chunk forward as color metadata (no ICC-profile pass-through).
"""

import io
import os
import struct
import tempfile
from typing import NamedTuple, Optional

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# PNG IHDR color types (see the PNG spec) relevant to this module.
_COLOR_TYPE_RGB = 2
_COLOR_TYPE_RGBA = 6


class HighBitDepthImage(NamedTuple):
    pixels: np.ndarray  # normalized (0..1) float32 RGB, shape (H, W, 3)
    source: object  # opaque oiio.ImageBuf -- never read pixels from this directly, metadata only


class HighBitDepthRGBAImage(NamedTuple):
    pixels: np.ndarray  # normalized (0..1) float32 RGBA, shape (H, W, 4)
    gamma: Optional[float]  # from the source PNG's gAMA chunk, if any


def _png_header(raw_bytes: bytes) -> Optional[dict]:
    """Parses just the PNG signature + IHDR chunk (zero dependencies) to cheaply check
    bit depth and color type before deciding whether a codec import is even needed.
    Returns None if `raw_bytes` doesn't start with the PNG magic number.
    """
    if raw_bytes[:8] != PNG_SIGNATURE:
        return None

    # IHDR is always the first chunk: 4-byte length, 4-byte type "IHDR", then 13 bytes
    # of data (width, height, bit depth, color type, compression, filter, interlace).
    if len(raw_bytes) < 8 + 8 + 13 or raw_bytes[12:16] != b"IHDR":
        return None

    width, height, bit_depth, color_type = struct.unpack(">IIBB", raw_bytes[16:26])
    return {"width": width, "height": height, "bit_depth": bit_depth, "color_type": color_type}


def read_high_bit_depth_rgb(raw_bytes: bytes) -> Optional[HighBitDepthImage]:
    """Returns the normalized (0..1) float32 RGB pixels plus source metadata if `raw_bytes`
    decodes to a genuine 16-bit-per-channel, alpha-free, 3-channel PNG -- None otherwise.
    """
    header = _png_header(raw_bytes)
    if header is None or header["bit_depth"] != 16 or header["color_type"] != _COLOR_TYPE_RGB:
        return None

    try:
        import OpenImageIO as oiio
    except ImportError as e:
        raise ImportError(
            "Reading/writing 16-bit RGB PNGs requires OpenImageIO. Install with: pip install 'trustmark[highbitdepth]'"
        ) from e

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(raw_bytes)

        buf = oiio.ImageBuf(tmp_path)
        if buf.has_error:
            return None

        spec = buf.spec()
        if spec.format.basetype != oiio.BASETYPE.UINT16 or spec.nchannels != 3:
            return None

        # Forces full pixel+metadata materialization into memory now, while the backing
        # temp file still exists -- `source` (used later by write_16bit_rgb_png's
        # copy_metadata()) needs to outlive this file, and ImageBuf's read-on-demand
        # behavior isn't a documented guarantee to lean on once the temp file is gone.
        if not buf.read(force=True):
            return None

        pixels = buf.get_pixels(oiio.FLOAT)
        # OpenImageIO normalizes integer formats to 0..1 float on read.
        return HighBitDepthImage(pixels=np.asarray(pixels, dtype=np.float32), source=buf)
    finally:
        os.unlink(tmp_path)


def write_16bit_rgb_png(pixels: np.ndarray, source: Optional[object] = None) -> bytes:
    """Encodes normalized (0..1) float32 RGB pixels, shape (H, W, 3), as a 16-bit-per-channel
    PNG. Pass the `source` from `read_high_bit_depth_rgb` to carry its color metadata forward.
    """
    import OpenImageIO as oiio

    height, width, _channels = pixels.shape
    spec = oiio.ImageSpec(width, height, 3, oiio.UINT16)
    buf = oiio.ImageBuf(spec)
    if source is not None:
        buf.copy_metadata(source)

    pixels_u16 = (np.clip(pixels, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    buf.set_pixels(oiio.ROI(0, width, 0, height, 0, 1, 0, 3), pixels_u16)

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        if not buf.write(tmp_path):
            raise RuntimeError(f"Failed to encode 16-bit PNG: {buf.geterror()}")
        with open(tmp_path, "rb") as tmp:
            return tmp.read()
    finally:
        os.unlink(tmp_path)


def read_high_bit_depth_rgba(raw_bytes: bytes) -> Optional[HighBitDepthRGBAImage]:
    """Returns the normalized (0..1) float32 RGBA pixels plus gamma if `raw_bytes` decodes to a
    genuine 16-bit-per-channel, 4-channel (RGBA) PNG -- None otherwise.
    """
    header = _png_header(raw_bytes)
    if header is None or header["bit_depth"] != 16 or header["color_type"] != _COLOR_TYPE_RGBA:
        return None

    try:
        import png as pypng
    except ImportError as e:
        raise ImportError(
            "Reading/writing 16-bit RGBA PNGs requires pypng. Install with: pip install 'trustmark[highbitdepth]'"
        ) from e

    reader = pypng.Reader(bytes=raw_bytes)
    width, height, rows, info = reader.asDirect()
    raw = np.array(list(rows), dtype=np.uint16)

    pixels = raw.reshape(height, width, 4).astype(np.float32) / 65535.0
    return HighBitDepthRGBAImage(pixels=pixels, gamma=info.get("gamma"))


def write_16bit_rgba_png(pixels: np.ndarray, gamma: Optional[float] = None) -> bytes:
    """Encodes normalized (0..1) float32 RGBA pixels, shape (H, W, 4), as a 16-bit-per-channel
    PNG. Pass the `gamma` from `read_high_bit_depth_rgba` to carry it forward.
    """
    import png as pypng

    height, width, _channels = pixels.shape
    raw = (np.clip(pixels, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)

    writer_kwargs = {"width": width, "height": height, "bitdepth": 16, "alpha": True, "greyscale": False}
    if gamma is not None:
        writer_kwargs["gamma"] = gamma
    writer = pypng.Writer(**writer_kwargs)

    buffer = io.BytesIO()
    writer.write(buffer, raw.reshape(height, width * 4))
    return buffer.getvalue()
