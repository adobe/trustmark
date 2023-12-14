# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from PIL import Image
from pathlib import Path

EXAMPLE_FILE = 'ufo_240.jpg'     # JPEG example
#EXAMPLE_FILE = 'bfly_rgba.png'   # Transparent PNG example

MODE='C'
tm=TrustMark(verbose=True, model_type=MODE)

# encoding example
cover = Image.open(EXAMPLE_FILE)
rgb=cover.convert('RGB')
has_alpha=cover.mode== 'RGBA'
if (has_alpha):
  alpha=cover.split()[-1]
encoded=tm.encode(rgb, 'mysecret')
if (has_alpha):
  encoded.putalpha(alpha)
outfile=Path(EXAMPLE_FILE).stem+'_'+MODE+'.png'
encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))

# decoding example
cover = Image.open(outfile).convert('RGB')
wm_secret, wm_present = tm.decode(cover)
if wm_present:
  print(f'Extracted secret: {wm_secret}')
else:
  print('No watermark detected')

# removal
stego = Image.open(outfile).convert('RGB')
im_recover = tm.remove_watermark(stego)
im_recover.save('recovered.png', exif=stego.info.get('exif'), icc_profile=stego.info.get('icc_profile'), dpi=stego.info.get('dpi'))
wm_secret, wm_present = tm.decode(im_recover)
if wm_present:
   print(f'Extracted secret after removal: {wm_secret}')
else:
   print('No secret after removal')
