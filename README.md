# TrustMark - Universal Watermarking for Arbitrary Resolution Images

The official, open source (MIT licensed) implementation of TrustMark watermarking  for the Content Authenticity Initiative (CAI) as described in:


**TrustMark - Universal Watermarking for Arbitrary Resolution Images**
https://arxiv.org/abs/2311.18297
[Tu Bui](https://www.surrey.ac.uk/people/tu-bui) <sup>1</sup>, [Shruti Agarwal](https://research.adobe.com/person/shruti-agarwal/)  <sup>2</sup> , [John Collomosse](https://www.collomosse.com)  <sup>1,2</sup>

<sup>1</sup> [DECaDE](https://decade.ac.uk/) Centre for the Decentralized Digital Economy, University of Surrey, UK.
<sup>2</sup> [Adobe Research](https://research.adobe.com/), San Jose, CA.

**This repo contains:**
`/python` a Python (3.8.5 or higher) implementation of TrustMark for encoding, decoding and removing image watermarks (using PyTorch).  
`/js` a Javascript implementation of TrustMark decoding of image watermarks (using ONNX)
`/c2pa` a Python example showing how to indicate the presence of a TrustMark watermark in C2PA metadata (manifest)

Models (**ckpt** and **onnx**) are no longer packaged in this repo due to size, but are downloaded upon first use.  Please check the code for [URLs and md5 hashes](https://github.com/adobe/trustmark/blob/4ef0dde4abd84d1c6873e7c5024482f849db2c73/python/trustmark/trustmark.py#L30) if direct download is preferred.
 
  
### Quickest start

Install from [PyPi](https://pypi.org/project/trustmark/) directly `pip install trustmark ` and try the code snippet below

### Quick start

Or within the `python/` folder run `pip install .` 

The `python/test.py` script provides examples of watermarking images (a JPEG photo, a JPEG GenAI image, and a transparent PNG image are provided as examples).  To test the installation the following code snippet in Python shows typical usage:

```python
from trustmark import TrustMark
from PIL import Image

# init
tm=TrustMark(verbose=True, model_type='Q') # or try P

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
In this example, TrustMark variant Q is being used to encode the word `mysecret` in ASCII7 encoding into the image `ufo_240.jpg` which is then decoded, and then removed from the image.

### Frequently Asked Questions (FAQs)

An extensive [TrustMark FAQ](https://github.com/adobe/trustmark/blob/main/FAQ.md) is now available answering typical questions on TrustMark configuration and integration.

### TrustMark Configuration

Detailed information on TrustMark configuration has now moved to the [FAQ](https://github.com/adobe/trustmark/blob/main/FAQ.md) and in the [CONFIG](https://github.com/adobe/trustmark/blob/main/CONFIG.md) guide.

### GPU Setup

The following clean install should work for getting up and running on Ubuntu using GPU for the PyTorch implementation in this repo.  Otherwise the repo will work fine  on CPU.  For the Javascript implementation, if you are using a Chromium browser it will take advantage of WebGPU automatically if available.
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

The raw 100 bits break down into D+E+V=100 bits, where D is the protect payload (e.g. 61) and E are the error correction parity bits (e.g. 35) and V are the version bits (always 4). The version bits comprise 2 reserved (unused) bits, and 2 bits encoding an $
  
### C2PA Usage

####  Durable Content Credentials

Please refer to the C2PA questions within the [FAQ](https://github.com/adobe/trustmark/blob/main/FAQ.md) for full details.

Open standards such as Content Credentials ([C2PA](https://c2pa.org/)), developed by the [Coalition for Content Provenance and Authenticity](https://c2pa.org/), describe ways to encode information about an image’s history or ‘provenance’ such as how and when it was made. This information is usually carried within the image’s metadata.

However, C2PA metadata can be accidentally removed when the image is shared through platforms that do not yet support the standard. If a copy of that metadata is retained in a database, the TrustMark identifier carried inside the watermark can be used as a key to look up that information from the database. This is referred to as a [Durable Content Credential](https://contentauthenticity.org/blog/durable-content-credentials) and the technical term for the identified is a 'soft binding'.

When used as a soft binding, TrustMark should be used to encode a random identifier via one of the Encoding types `BCH_n` described in the data schema described in the previous section of this document.  Example `c2pa/c2pa_watermark_example.py` provides an example use of TrustMark, and also how the identifer should be reflected within the C2PA metadata (manifest) via a 'soft binding assertion'.

#### Signpost watermark
TrustMark [coexists well with most other image watermarks](https://arxiv.org/abs/2501.17356) and so can be used as a 'signpost' to indicate the co-presence of another watermarking technology.  This can be helpful, as TrustMark is an open technology it can be used to 'signpost' which decoder to obtain and run on an image to decode a soft binding identifier for C2PA.

In this mode the encoding should be `Encoding.BCH_SUPER` and the payload contain an integer identifer that describes the co-present watermark.  The integer should be taken from the registry of C2PA approved watermarks listed in this normative [repo](https://github.com/c2pa-org/softbinding-algorithms-list)  which is also available in machine readable format as JSON.

## License

TrustMark is released under an MIT Open Source License.

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
