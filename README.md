# TrustMark - Universal Watermarking for Arbitrary Resolution Images

An Open Source, MIT licensed implementation of TrustMark watemarking for the Content Authenticity Initiative (CAI) as described in: 

**TrustMark - Universal Watermarking for Arbitrary Resolution Images**

https://arxiv.org/abs/2311.18297 

[Tu Bui](https://www.surrey.ac.uk/people/tu-bui) <sup>1</sup>, [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)  <sup>2</sup> , [John Collomosse](https://www.collomosse.com)  <sup>1,2</sup> 

<sup>1</sup> DECaDE Centre for the Decentralized Digital Economy, University of Surrey, UK. \
<sup>2</sup> Adobe Research, San Jose CA.


This repo contains a Python (3.8.5 or higher) implementation of TrustMark for  encoding, decoding and removing image watermarks.  


### Quick start 

Within the python folder run `pip install .` \

The `python/test.py` script provides examples of watermarking images (a JPEG and a transparent PNG image are provided as examples).  To test the installation the following code snippet in Python shows typical usage:

```python
from trustmark import TrustMark
from PIL import Image

# init
tm=TrustMark(verbose=True, model_type='Q')

# encoding example
cover = Image.open('ufo_240.jpg').convert('RGB')
tm.encode(cover, 'mysecret').save('ufo_240_Q.png')

# decoding example
cover = Image.open('ufo_240_Q.png').convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(cover)
if wm_present:
  print(f'Extracted secret: {wm_secret}')
else:
  print('No watermark detected')

# removal example
stego = Image.open('ufo_240_Q.png').convert('RGB')
im_recover = tm.remove_watermark(stego)
im_recover.save('recovered.png')
```


### TrustMark Models

Models are now fetched on first use, due to the number of variants and size of models they are not packaged as binary any more.

We recommend use of the Q (quality) model variant.  Other variants are packaged for historial / academic paper reproduction purposes but exhibit a lower PSNR.

### CUDA troubleshoot

The following clean install should work for getting up and running on GPU using the python implementation in this repo.

```
conda create --name trustmark python=3.10
conda activate trustmark
conda install pytorch cudatoolkit=12.8 -c pytorch -c conda-forge
pip install torch==2.1.2 torchvision==0.16.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
```

### TrustMark Data Schema

Packaged TrustMark models/code are trained to encode a payload of 100 bits.  

To promote interoperability we recommend following the data schema implemented in `python/datalayer`.  This affords for a user selectable level of error correction over the raw 100 bits of payload.

#### Supported modes

* `Encoding.BCH_5` - Protected payload of 61 bits (+ 35 ECC bits) - allows for 5 bit flips.
* `Encoding.BCH_4` - Protected payload of 68 bits (+ 28 ECC bits) - allows for 4 bit flips.
* `Encoding.BCH_3` - Protected payload of 75 bits (+ 21 ECC bits) - allows for 3 bit flips.
* `Encoding.BCH_SUPER` - Protected payload of 40 bits (+ 56 ECC bits) - allows for 8 bit flips.

For example instantiate the encoder as:
```
tm=TrustMark(verbose=True, model_type='Q', encoding_type=TrustMark.Encoding.BCH_5)
```

The decoder will automatically detect the data schema in a given TrustMark, allowing for user selectable level of robustness.

#### Payload encoding

The raw 100 bits break down into D+E+V=100 bits, where D is the protect payload (e.g. 61) and E are the error correction parity bits (e.g. 35) and V are the version bits (always 4). The version bits comprise 2 reserved (unused) bits, and 2 bits encoding an integer in range 0-3 which indicate the trustmark data schema in use (see `python/datalayer.py` for the numeric codes).

### Usage with C2PA

TrustMark may be used to directly encode a 'soft binding' identifier, which may be used to look up provenace metadata (manifest). This identifier should be encoded via one of the Encoding types BCH_n described above.

TrustMark may alternatively be used to indicate the presence of another watermarking technology that carries an identifier.  In this mode the encoding should be Encoding.BCH_SUPER and the payload contain an integer identifer that describes the co-present watermarking technology.  This value should be taken from the C2PA Soft Binding Algorithm List.

An example of direct encoding for C2PA is included in `c2pa/c2pa_watermark_example.py` including the C2PA manifest that should be used to describe the watermark insertion.

### Quickstart with CUDA

TrustMark will detect if CUDA is available and use GPU if so, else default to CPU encode/decode.

The following clean install should work for getting up and running on GPU using the python implementation in this repo.

```
conda create --name trustmark python=3.10
conda activate trustmark
conda install pytorch cudatoolkit=12.8 -c pytorch -c conda-forge
pip install torch==2.1.2 torchvision==0.16.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
```

## Citation

If you find this work useful we request you please cite the repo and/or TrustMark paper as follows.

```
@article{trustmark,
  title={Trustmark: Universal Watermarking for Arbitrary Resolution Images},
  author={Bui, Tu and Agarwal, Shruti and Collomosse, John},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {2311.18297},
  year = 2023,
  month = nov
}
```


