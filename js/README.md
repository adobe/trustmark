# TrustMark JavaScript Demonstrator

This repository contains a JavaScript-based demonstrator for decoding TrustMark watermarks embedded in images. The aim is to provide a minimal example for developers to build client-side JS applications for example browser extensions.

## Overview
The demonstrator consists of key modules that handle image preprocessing, watermark detection, and data decoding. It supports the `Q` variant of the TrustMark watermarking schema.

## Components
- **tm_watermark.js**
  - Core module for handling watermark detection and decoding.
  - Defines key functions for processing images and extracting watermark data.
  - `TRUSTMARK_VARIANT`: Specifies the TurstMark model variant.

- **tm_datalayer.js**
  - Handles data decoding and schema-specific processing.
  - Implements error correction and interpretation of binary watermark data.

## Key Parameters

The desired TrustMark watermark variants for decoding may be listed in the `modelConfigs` array at the top of `tm_watermark.js` for example, B, C, Q and P varaints.

## Usage

Open `index.html` in a browser to run the demonstrator or include the modules in your JavaScript project.

The JS will use WebGPU to process the ONNX models, if GPU is available (check `chrome://gpu`).  If you use WebGPU it will only run in a secure context, which means off localhost or an https link.  You can start a local https server via the `server.py` and a suitable openssl cert in `server.pem`.

## Example Output
For an image containing a TrustMark watermark:
```
Watermark Found (BCH_SUPER):
10100110100000110110111011011010011000111110010010000010111011111101
C2PA Info:
{
  "alg": "com.adobe.trustmark.Q",
  "blocks": [
    {
      "scope": {},
      "value": "2*10100110100000110110111011011010011000111110010010000010111011111101"
    }
  ]
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
