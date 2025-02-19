# Configuring TrustMark

## Overview

The default configuration for **TrustMark** (variant Q, 100% strength, BCH_5 error correction)is sufficient for most use cases. All watermarking algorithms trade off between three properties:

- **Capacity (bits)**  
- **Robustness (to various transformations)**  
- **Visibility (of watermark)**

This document outlines the main parameters for configuring TrustMark to tune these properties.

---

## Model Variant

**TrustMark** ships with four model variants (**B**, **C**, **P**, and **Q**) that may be selected when instantiating TrustMark.  All encode/decode calls on the object will use this variant.

In general, we recommend using **P** or **Q**:
- **P** is useful for creative applications where very high visual quality is required.  
- **Q** is a good all-rounder and is the default.

> **Note:** Images encoded with one model variant cannot be decoded with another.

| Variant | Typical PSNR | Model Size Enc/Dec (MB) | Description                                                                                         |
|---------|--------------|-------------------------|-----------------------------------------------------------------------------------------------------|
| **Q**   | 43-45        | 17/45                  | Default (**Q**uality). Good trade-off between robustness and imperceptibility. Uses ResNet-50 decoder. |
| **B**   | 43-45        | 17/45                  | (**B**eta). Very similar to Q, included mainly for reproducing the paper. Uses ResNet-50 decoder.   |
| **C**   | 38-39        | 17/21                  | (**C**ompact). Uses a ResNet-18 decoder (smaller model size). Slightly lower visual quality.        |
| **P**   | 48-50        | 16/45                  | (**P**erceptual). Very high visual quality and good robustness.  ResNet-50 decoder trained with much higher weight on perceptual loss (see paper).   |

---

## Watermark Strength (Runtime Parameter)

An optional parameter `WM_STRENGTH` (default `1.0`) can be provided when encoding. Strength enables a trade-off between **robustness** and **visibility**.

- **Raising `WM_STRENGTH`** (e.g., to `1.5`) improves robustness (e.g., survives printing) but increases the likelihood of ripple artifacts.  
- **Lowering `WM_STRENGTH`** (e.g., to `0.8`) reduces any potential artifacts further but  compromises on robustness (still survives lower noise, screenshotting, or social media).

```python
# Example usage:
encoded_image = tm.encode(input_image, payload="example", WM_STRENGTH=1.5)
```

## Error Correction Level

**TrustMark** affords a user-selectable level of error correction over the raw 100 bits of payload. This helps maintain reliability under transformations or noise.

### Supported Modes

- `Encoding.BCH_5`  
  Protected payload of 61 bits (+ 35 ECC bits) – allows for 5 bit flips
- `Encoding.BCH_4`  
  Protected payload of 68 bits (+ 28 ECC bits) – allows for 4 bit flips
- `Encoding.BCH_3`  
  Protected payload of 75 bits (+ 21 ECC bits) – allows for 3 bit flips
- `Encoding.BCH_SUPER`  
  Protected payload of 40 bits (+ 56 ECC bits) – allows for 8 bit flips

For example, instantiate the encoder with:

```python
tm = TrustMark(verbose=True, model_type='Q', encoding_type=TrustMark.Encoding.BCH_5)
```

Having selected an appropriate model and strength, you are implicitly selecting the level of robustness and visibility of the watermark. If you have chosen to reduce robustness for lower visibility, you can regain some robustness by increasing error correction (at the cost of payload capacity).  Note that even 40 bits gives a key space of around one trillion.

---

## Centre Cropping

**TrustMark** generates residuals at \`256 x 256\` and then scales/blends them into the original image. Several derivative papers have adopted this universal resolution-scaling trick.

- If the original image is extremely long/thin (aspect ratio beyond \`2:1\`), the residual watermark will degrade when scaled.  
- **TrustMark** addresses this by automatically centre-cropping the image to a square if the aspect ratio exceeds \`2.0\`.  
  - Example: For a \`1000 x 200\` image, only a \`200 x 200\` region in the center carries the watermark.
- The aspect ratio limit can be overridden via the \`ASPECT_RATIO_LIM\` parameter.
  - Setting it to \`1.0\` always forces centre-crop behavior (useful for content platforms that square-crop images).  This is the default when using model variant **P**.

---

## Watermark Concentration (Don’t Do This)

Visual quality is often measured via **PSNR** (Peak Signal-to-Noise Ratio), but PSNR does not perfectly correlate with human perception of watermark visibility.

Some derivative works have tried to improve PSNR by "zero padding" or concentrating the watermark into only the central region, effectively altering fewer pixels and artificially raising the PSNR score. However, this approach **can increase human-visible artifacts** in the concentrated area.

**Parameter:** \`CONCENTRATE_WM_REGION\`  
- Default is \`100%\` (no zero padding).  
- If you set, for example, \`80%\` zero padding, you might inflate PSNR by ~5 dB.  
- At \`50%\` zero padding, PSNR might inflate by ~10 dB.  
- In some extreme cases, PSNR can reach 55–60 dB but at the cost of noticeable artifacts in that smaller region.

> **In summary**: While the functionality exists for comparison purposes, it’s not recommended for production. High concentration setting yields high PSNR but paradoxically more visible artifacts in the watermarked area.







