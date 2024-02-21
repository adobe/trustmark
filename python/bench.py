# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from PIL import Image
from pathlib import Path
import time

EXAMPLE_FILE = 'ufo_240.jpg'     # JPEG example
#EXAMPLE_FILE = 'bfly_rgba.png'   # Transparent PNG example

N=10 # repetitions for bench

MODE='Q'
tm=TrustMark(verbose=True, model_type=MODE)

# encoding example
cover = Image.open(EXAMPLE_FILE)
rgb=cover.convert('RGB')
has_alpha=cover.mode== 'RGBA'
for i in range (0,N):
  if i==1:
    tic=time.time()
  if (has_alpha):
    alpha=cover.split()[-1]
  encoded=tm.encode(rgb, 'mysecret')
  if (has_alpha):
    encoded.putalpha(alpha)
toc=time.time()
print('Timing::Encode single image %f millisec' % (((toc-tic)/(N-1))*1000))
outfile=Path(EXAMPLE_FILE).stem+'_'+MODE+'.png'
encoded.save(outfile, exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))

# decoding example
cover = Image.open(outfile).convert('RGB')
for i in range (0,N):
  if i==1:
    tic=time.time()
  wm_secret, wm_present, wm_schema = tm.decode(cover)
toc=time.time()
print('Timing::Decode single image %f millisec' % (((toc-tic)/(N-1))*1000))
if wm_present:
  print(f'Extracted secret: {wm_secret}')
else:
  print('No watermark detected')

# removal
stego = Image.open(outfile).convert('RGB')
tic=time.time()
for i in range (0,N):
  if i==1:
    tic=time.time()
  im_recover = tm.remove_watermark(stego)
toc=time.time()
print('Timing::Remove wm single image %f millisec' % (((toc-tic)/(N-1))*1000))
im_recover.save('recovered.png', exif=stego.info.get('exif'), icc_profile=stego.info.get('icc_profile'), dpi=stego.info.get('dpi'))
wm_secret, wm_present, wm_schema = tm.decode(im_recover)
if wm_present:
   print(f'Extracted secret after removal: {wm_secret}')
else:
   print('No secret after removal')
