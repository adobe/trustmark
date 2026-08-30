# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from trustmark.high_bit_depth import (
    read_high_bit_depth_rgb,
    write_16bit_rgb_png,
    read_high_bit_depth_rgba,
    write_16bit_rgba_png,
    _png_header,
)
from PIL import Image
from pathlib import Path
import io
import math,random
import numpy as np


#EXAMPLE_FILE = '../images/ufo_240.jpg'     # JPEG example
#EXAMPLE_FILE = '../images/ghost.png'        # PNG RGBA example
EXAMPLE_FILE = '../images/ripley.jpg'     # JPEG example

# Available modes: Q=balance, P=high visual quality, C=compact decoder, B=base from paper
MODE='P'
DETECTFIRST=False  # False = decode full image watermark, True = detect watermarked region before decoding
tm=TrustMark(verbose=True, model_type=MODE, encoding_type=TrustMark.Encoding.BCH_5,  loadBBoxDetector=DETECTFIRST)

# encoding example
cover = Image.open(EXAMPLE_FILE)
rgb=cover.convert('RGB')
has_alpha=cover.mode== 'RGBA'
if (has_alpha):
  alpha=cover.split()[-1]

random.seed(1234)
capacity=tm.schemaCapacity()
bitstring=''.join([random.choice(['0', '1']) for _ in range(capacity)])
encoded=tm.encode(rgb, bitstring, MODE='binary')

if (has_alpha):
  encoded.putalpha(alpha)
outfile=Path(EXAMPLE_FILE).stem+'_'+MODE+'.png'
encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))


# decoding example
stego = Image.open(outfile).convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary', DETECTFIRST=DETECTFIRST, ROTATION=False)
if wm_present:
  print(f'Extracted secret: {wm_secret} (schema {wm_schema})')
else:
  print('No watermark detected')

# psnr (quality, higher is better)
mse = np.mean(np.square(np.subtract(np.asarray(stego).astype(np.int16), np.asarray(rgb).astype(np.int16))))
if mse > 0:
  PIXEL_MAX = 255.0
  psnr= 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)
  print('PSNR = %f' % psnr)

# removal
stego = Image.open(outfile)
rgb=stego.convert('RGB')
has_alpha=stego.mode== 'RGBA'
if (has_alpha):
  alpha=stego.split()[-1]
im_recover = tm.remove_watermark(rgb)
rm_wm_secret, rm_wm_present, rm_wm_schema = tm.decode(im_recover, MODE='binary', DETECTFIRST=DETECTFIRST)
if rm_wm_present and rm_wm_schema==wm_schema:
  print(f'Extracted secret: {rm_wm_secret} (schema {rm_wm_schema})')
else:
   print('No secret after removal')
if (has_alpha):
  im_recover.putalpha(alpha)
im_recover.save('recovered.png', exif=stego.info.get('exif'), icc_profile=stego.info.get('icc_profile'), dpi=stego.info.get('dpi'))

# 16-bit PNG support (requires: pip install trustmark[highbitdepth])
# Skipped gracefully if OpenImageIO/pypng aren't installed.
try:
  import OpenImageIO  # noqa: F401
  import png  # noqa: F401
  HAVE_HIGHBITDEPTH = True
except ImportError:
  HAVE_HIGHBITDEPTH = False
  print('Skipping 16-bit PNG tests: pip install trustmark[highbitdepth] not installed')

if HAVE_HIGHBITDEPTH:
  rng = np.random.default_rng(1234)

  # --- pure I/O round trip: RGB, no model involved ---
  rgb_pixels = rng.integers(0, 65536, (48, 64, 3)).astype(np.float32) / 65535.0
  rgb_bytes = write_16bit_rgb_png(rgb_pixels)
  assert _png_header(rgb_bytes)['bit_depth'] == 16
  read_back = read_high_bit_depth_rgb(rgb_bytes)
  max_err = np.max(np.abs(read_back.pixels - rgb_pixels))
  print(f'16-bit RGB I/O round trip max error: {max_err:.6f} (expect near uint16 quantization step)')

  # --- pure I/O round trip: RGBA, no model involved ---
  rgba_pixels = rng.integers(0, 65536, (48, 64, 4)).astype(np.float32) / 65535.0
  rgba_bytes = write_16bit_rgba_png(rgba_pixels, gamma=0.45455)
  assert _png_header(rgba_bytes)['bit_depth'] == 16
  read_back_rgba = read_high_bit_depth_rgba(rgba_bytes)
  max_err_rgba = np.max(np.abs(read_back_rgba.pixels - rgba_pixels))
  print(f'16-bit RGBA I/O round trip max error: {max_err_rgba:.6f}, gamma preserved: {read_back_rgba.gamma}')

  # --- end-to-end: watermark a genuine 16-bit RGB PNG, confirm precision + watermark survive ---
  # Uses real photo content (upscaled 8-bit -> 16-bit via *257, the standard bit-depth
  # expansion) rather than random noise: TrustMark's detector relies on structural image
  # content, and pure per-pixel noise is unreliable to decode regardless of image size
  # (confirmed separately -- not specific to the high-bit-depth path).
  hb_source_8bit = np.asarray(Image.open('../images/ripley.jpg').convert('RGB').resize((256, 256)), dtype=np.uint16)
  hb_rgb_pixels = (hb_source_8bit * 257).astype(np.float32) / 65535.0
  hb_in_bytes = write_16bit_rgb_png(hb_rgb_pixels)
  hb_secret = ''.join([random.choice(['0', '1']) for _ in range(capacity)])
  hb_out_bytes = tm.encode_high_bit_depth(hb_in_bytes, hb_secret, MODE='binary')
  assert _png_header(hb_out_bytes)['bit_depth'] == 16, 'encode_high_bit_depth must deliver a genuinely 16-bit PNG'

  # No decode_high_bit_depth exists -- ordinary decode() on the PIL-flattened 8-bit
  # image is expected to still detect the watermark.
  hb_stego_8bit = Image.open(io.BytesIO(hb_out_bytes)).convert('RGB')
  hb_secret_out, hb_present, hb_schema = tm.decode(hb_stego_8bit, MODE='binary')
  if hb_present and hb_secret_out == hb_secret:
    print('16-bit encode_high_bit_depth -> 8-bit decode() round trip: secret recovered correctly')
  else:
    print(f'16-bit round trip FAILED: present={hb_present} secret_out={hb_secret_out!r} expected={hb_secret!r}')

  # --- negative check: ordinary 8-bit PNG bytes must be rejected, not silently degraded ---
  buf = io.BytesIO()
  Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(buf, format='PNG')
  try:
    tm.encode_high_bit_depth(buf.getvalue(), hb_secret, MODE='binary')
    print('encode_high_bit_depth FAILED to reject an 8-bit PNG')
  except ValueError:
    print('encode_high_bit_depth correctly rejects 8-bit PNG input')

