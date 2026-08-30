# TrustMark

This repository contains the official, open source implementation of TrustMark watermarking for the Content Authenticity Initiative (CAI) as described in:

[**TrustMark - Universal Watermarking for Arbitrary Resolution Images**](https://arxiv.org/abs/2311.18297) (`arXiv:2311.18297`) by [Tu Bui](https://www.surrey.ac.uk/people/tu-bui)[^1], [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)[^2], and [John Collomosse](https://www.collomosse.com)[^1] [^2].

and later published at [ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/1714) as [**TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images**](https://collomosse.com/pubs/Bui-ICCV-2025.pdf).

[^1]: [DECaDE](https://decade.ac.uk/) Centre for the Decentralized Digital Economy, University of Surrey, UK.

[^2]: [Adobe Research](https://research.adobe.com/), San Jose, CA.

## Overview

This repository contains the following directories:

- `/python`: Python implementation of TrustMark for encoding, decoding and removing image watermarks (using PyTorch). For information on configuring TrustMark in Python, see [Configuring TrustMark](python/CONFIG.md). 
- `/js`: Javascript implementation of TrustMark decoding of image watermarks (using ONNX).  For more information, see [TrustMark - JavaScript implementation](js/README.md).
- `/rust`: Rust implementation of TrustMark. for more information, see [TrustMark — Rust implementation](rust/README.md).
- `/c2pa`: Python example of how to indicate the presence of a TrustMark watermark in a C2PA manifest. For more information, see [Using TrustMark with C2PA](c2pa/README.md).

Model files (**ckpt** PyTorch file for Python and **onnx** ONNX file for JavaScript) are not packaged in this repository due to their size, but are downloaded upon first use.  See the code for [URLs and md5 hashes](https://github.com/adobe/trustmark/blob/010a8f3bf885eec6f2aaa418380afc41d68bb3e5/python/trustmark/trustmark.py#L29) for a direct download link.  Note that the location path was updated from Netlify to S3 in April 2026.

More information:

- For answers to common questions, see the [FAQ](FAQ.md).
- For information on configuring TrustMark in Python, see [Configuring TrustMark](python/CONFIG.md).

## Installation

### Prerequisite

You must have Python 3.8.5 or higher to use the TrustMark Python implementation.

### Installing from PyPI

The easiest way to install TrustMark is from the [Python Package Index (PyPI)](https://pypi.org/project/trustmark/) by entering this command:

```
pip install trustmark
```

Alternatively, after you've cloned the repository, you can install from the `python` directory:

```
cd trustmark/python
pip install .
```

## Quickstart

To get started quickly, run the `python/test.py` script that provides examples of watermarking several 
image files from the `images` directory. 

### Run the example

Run the example as follows:

```sh
cd trustmark/python
python test.py
```

You'll see output like this:

```
Initializing TrustMark watermarking with ECC using [cpu]
Extracted secret: 1000000100001110000010010001011110010001011000100000100110110 (schema 1)
PSNR = 50.357909
No secret after removal
```

### Example script

The `python/test.py` script provides examples of watermarking a JPEG photo, a JPEG GenAI image, and an RGBA PNG image. The example uses TrustMark variant Q to encode the word `mysecret` in ASCII7 encoding into the image `ufo_240.jpg` which is then decoded, and then removed from the image.

```python
from trustmark import TrustMark
from PIL import Image

# init
tm=TrustMark(verbose=True, model_type='Q') # or try P

# encoding example
cover = Image.open('images/ufo_240.jpg').convert('RGB')
tm.encode(cover, 'mysecret').save('ufo_240_Q.png')

# decoding example
cover = Image.open('images/ufo_240_Q.png').convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(cover)

if wm_present:
   print(f'Extracted secret: {wm_secret}')
else:
   print('No watermark detected')

# removal example
stego = Image.open('images/ufo_240_Q.png').convert('RGB')
im_recover = tm.remove_watermark(stego)
im_recover.save('images/recovered.png')
```

### 16-bit PNG support

Watermarking a genuine 16-bit-per-channel PNG through `tm.encode()` loses that extra
precision, since PIL flattens 16-bit PNGs to 8-bit on load. Use
`tm.encode_high_bit_depth()` instead to keep full precision in the delivered image —
see the "16-bit PNG support" section of [`CLAUDE.md`](CLAUDE.md) for details and the
required `pip install trustmark[highbitdepth]` extra.

## GPU setup

TrustMark runs well on CPU hardware.  

To leverage GPU compute for the PyTorch implementation on Ubuntu Linux, first install Conda, then use the following commands to install:

```sh
conda create --name trustmark python=3.10
conda activate trustmark
conda install pytorch cudatoolkit=12.8 -c pytorch -c conda-forge
pip install torch==2.1.2 torchvision==0.16.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
```

For the JavaScript implementation, a Chromium browser automatically uses WebGPU, if available.

## Data schema

TrustMark encodes a payload (the watermark data embedded within the image) of 100 bits.
You can configure an error correction level over the raw 100 bits of payload to maintain reliability under transformations or noise. 

In payload encoding, the version bits comprise two reserved (unused) bits, and two bits encoding an integer value 0-3 that specifies the data schema as follows: 
- 0: BCH_SUPER
- 1: BCH_5
- 2: BCH_4
- 3: BCH_3

For more details and information on configuring the encoding mode in Python, see [Configuring TrustMark](python/CONFIG.md). 

## Citation

If you find this work useful, please give us a star ⭐ and cite the repository and/or TrustMark paper, preferrably as follows:
```
@inproceedings{Trustmark-ICCV-2025,
  title = {TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images},
  author={Bui, Tu and Agarwal, Shruti and Collomosse, John},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year = {2025},
  month = oct
}
```
or cite the earlier ArXiv version:
```
@article{Trustmark-ArXiv-2023,
title={Trustmark: Universal Watermarking for Arbitrary Resolution Images},
author={Bui, Tu and Agarwal, Shruti and Collomosse, John},
journal = {ArXiv e-prints},
archivePrefix = "arXiv",
eprint = {2311.18297},
year = 2023,
month = nov
}
```

## License 

This package is is distributed under the terms of the [MIT license](https://github.com/adobe/trustmark/blob/main/LICENSE).
