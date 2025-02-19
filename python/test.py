# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from PIL import Image
from pathlib import Path
import math,random
import numpy as np


EXAMPLE_FILE = '../images/ufo_240.jpg'     # JPEG example
#EXAMPLE_FILE = '../images/ripley.jpg'     # JPEG example
#EXAMPLE_FILE = '../images/bfly_rgba.png'   # Transparent PNG example

# Available modes: C=compact, Q=quality, B=base
MODE='Q'
tm=TrustMark(verbose=True, model_type=MODE, encoding_type=TrustMark.Encoding.BCH_5)

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
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary')
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
wm_secret, wm_present, wm_schema = tm.decode(im_recover)
if wm_present:
  print(f'Extracted secret: {wm_secret} (schema {wm_schema})')
else:
   print('No secret after removal')
if (has_alpha):
  im_recover.putalpha(alpha)
im_recover.save('recovered.png', exif=stego.info.get('exif'), icc_profile=stego.info.get('icc_profile'), dpi=stego.info.get('dpi'))

