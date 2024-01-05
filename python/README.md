# TrustMark - Universal Watermarking for Arbitrary Resolution Images

An Open Source, MIT licensed implementation of TrustMark watemarking for the Content Authenticity Initiative (CAI) as described in: 

**TrustMark - Universal Watermarking for Arbitrary Resolution Images**

https://arxiv.org/abs/2311.18297 

[Tu Bui](https://www.surrey.ac.uk/people/tu-bui) <sup>1</sup>, [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)  <sup>2</sup> , [John Collomosse](https://www.collomosse.com)  <sup>1,2</sup> 

<sup>1</sup> DECaDE Centre for the Decentralized Digital Economy, University of Surrey, UK. \
<sup>2</sup> Adobe Research, San Jose CA.


This repo contains a Python (3.8.5 or higher) (`python/`)and a browser deployable Javascript (`js/`) implementation of 
TrustMark for  encoding, decoding and removing image watermarks.  In addition, a C2PA (`c2pa/`) example is provided to 
demonstrate signing of a watermarked image using the CAI supplied python library for C2PA / Content Credentials.



### Quick start for Python

Within the python folder run `pip install .` \
The TrustMark package is also available via PyPi - run `pip install trustmark` to install it indepedent of this repo.

The `python/test.py` script provides examples of watermarking images (a JPEG and a transparent PNG image are provided as examples).  To test the installation the following code snippet in Python shows typical usage:

```python
from trustmark import TrustMark
from PIL import Image

# init
tm=TrustMark(verbose=True, model_type='C')

# encoding example
cover = Image.open('ufo_240.jpg').convert('RGB')
tm.encode(cover, 'mysecret').save('ufo_240_C.png')

# decoding example
cover = Image.open('ufo_240_C.png').convert('RGB')
wm_secret, wm_present = tm.decode(cover)
if wm_present:
  print(f'Extracted secret: {wm_secret}')
else:
  print('No watermark detected')

# removal example
stego = Image.open('ufo_240_C.png').convert('RGB')
im_recover = tm.remove_watermark(stego)
im_recover.save('recovered.png')
```

### Quick start for Javascript

Locally serve the example in the `js` folder by running within that folder `python -m http.server 8080` and visiting `http://localhost:8080'

Examples are included showing decoding (`wm_decoder.html`) and encoding (`wm_encoder.html`) browse to the relevant file to run the demo


### Quick start for C2PA Signing 

C2PA is an open technical specification for media provenance and a common use case is to use watermarks to match images stripped of C2PA metadata to a database keyed by watermarked identifiers.

A python example of embedding a watermark and signing the resulting image with a C2PA manifest containing the watermark ID is provided in the `c2pa/` folder.


### TrustMark Models

The repo contains several model variants of TrustMark (`model/`) for flexibility and reproducability of the associated technical paper. The (B,Q) variants described in the technical paper, as well as the (C) variant use for the purposes of embedding provenance signals in images for the Content Authenticity Initiative (CAI).  Use the C variant to encode/decode identifiers for CAI images.

Some of these are lazy loaded from CAI servers due to bandwidth limitations


## Citation

If you find this work useful we request you please cite the repo and/or TrustMark paper as follows.

```
@article{trustmark,
  title={Trustmark: Universal Watermarking for Arbitrary Resolution Images},
  author={Bui, Tu and Agarwal, Shruti and Collomosse, John},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {2311.18297},
  year = 2013,
  month = nov
}
```


