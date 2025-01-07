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

## How It Works
1. **Image Preprocessing**
   - Images are loaded into tensors and resized to the target size using the `runResizeModelSquare` function.
   - Extreme aspect ratios are handled with center cropping to ensure square input dimensions.

2. **Watermark Detection**
   - The resized image tensor is passed to the watermark detection model.
   - Detected watermark data is returned as an array of confidence scores.

3. **Data Decoding**
   - The watermark data is interpreted and decoded based on the schema version (e.g., BCH_SUPER).
   - Decoded data includes binary content, schema, and optionally, soft binding information.

## Usage

Open `index.html` in a browser to run the demonstrator or include the modules in your JavaScript project.

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
