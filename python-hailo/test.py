import math
import random

import numpy as np
from PIL import Image
from trustmark_hailo import TrustMark

EXAMPLE_FILE = "../images/ripley.jpg"  # JPEG example

MODEL_TYPE = "Q"
tm = TrustMark(model_type=MODEL_TYPE, verbose=True)

# encoding example
cover = Image.open(EXAMPLE_FILE)
rgb = cover.convert("RGB")

random.seed(1234)
capacity = tm.schemaCapacity()
bitstring = "".join([random.choice(["0", "1"]) for _ in range(capacity)])
encoded = tm.encode(rgb, bitstring, MODE="binary")

outfile = f"ripley_{MODEL_TYPE}_hailo.png"
encoded.save(outfile)
print(f"Wrote {outfile}")

# decoding example
stego = Image.open(outfile).convert("RGB")
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE="binary")
if wm_present:
    print(f"Extracted secret: {wm_secret} (schema {wm_schema})")
    print(f"Exact match: {wm_secret == bitstring}")
else:
    print("No valid watermark decoded")

# psnr (quality, higher is better)
mse = np.mean(np.square(np.subtract(np.asarray(stego).astype(np.int16), np.asarray(rgb).astype(np.int16))))
if mse > 0:
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)
    print("PSNR = %f" % psnr)

tm.close()
