# TrustMark JavaScript example

The [`js`](https://github.com/adobe/trustmark/tree/main/js) directory contains an example JavaScript implementation of decoding TrustMark watermarks embedded in images. It provides a minimal example of a client-side JavaScript application, would could be applied, for example, to a browser extension.

## Overview

The example consists of key modules that handle image preprocessing, watermark detection, and data decoding. It supports the `Q` variant of the TrustMark watermarking schema.

NOTE: The TrustMark JavaScript implementation only decodes watermarked images, in contrast to the Python implementation, which can both encode and decode.

## Components

This example consists of a simple HTML file, `index.html` that loads two JavaScript files:

- [`tm_watermark.js`](https://github.com/adobe/trustmark/blob/main/js/tm_watermark.js), the core module that handles watermark detection and decoding and defines key functions for processing images and extracting watermark data. The `modelConfigs` array specifies parameters such as the TrustMark model variant. As provided, the code checks for both Q and P variants.
- [`tm_datalayer.js`](https://github.com/adobe/trustmark/blob/main/js/tm_datalayer.js) handles data decoding and schema-specific processing.  It also implements error correction and interpretation of binary watermark data.

If GPU compute is available (if you're using Google Chrome, check `chrome://gpu`), then the code will automatically use WebGPU to process the ONNX models.  If you use WebGPU it will only run in a secure context, which means on localhost or an HTTPS link.  You can start a local HTTPS server by running the `server.py` script and a suitable OpenSSL certificate in `server.pem`.

## Key parameters

The desired TrustMark watermark variants for decoding are listed in the `modelConfigs` array at the top of `tm_watermark.js` for example, B, C, Q and P variants:

```js
const modelConfigs = [
  { variantcode: 'Q', fname: 'decoder_Q', sessionVar: 'session_wmarkQ', resolution: 256, squarecrop: false }, 
  { variantcode: 'P', fname: 'decoder_P', sessionVar: 'session_wmarkP', resolution: 224, squarecrop: true },
];
```

## Run the example

To run the example:

1. Start a local web server; for example, using Python: 
    ```
    python -m http.server 8000
    ```
1. Open `index.html` in a browser to run the example.
1. Drag and drop images (for example from the provided [`images`](https://github.com/adobe/trustmark/tree/main/images) directory) onto the indicated area in the web page, which will display information if the image contains a TrustMark watermark (see example below).

To use the code in your own project, simply include the two JavaScript files as usual.

## BCH library

[`deps/bch_ecc.js`](deps/bch_ecc.js) contains the readable BCH source. The example loads
[`deps/bch_ecc.min.js`](deps/bch_ecc.min.js), generated from that source without obfuscation.
After changing the source, regenerate the minified file from the `js` directory:

```sh
npx --yes esbuild@0.27.4 deps/bch_ecc.js --minify --target=es2020 --legal-comments=inline --outfile=deps/bch_ecc.min.js
```

Run the CPU-only ECC regressions against both files with `node --test test-ecc.cjs`.

## Example output

For an image containing a TrustMark watermark:

```
[4:12:48 PM] Decoding watermark...
[4:12:48 PM] Watermark Found (BCH_5):
1101101111100111100111100101110001111100101100101111010000000
C2PA Assertion:
{
  "c2pa.soft-binding": {
    "alg": "com.adobe.trustmark.Q",
    "blocks": [
      {
        "scope": {},
        "value": "1*1101101111100111100111100101110001111100101100101111010000000"
      }
    ]
  }
}
```

