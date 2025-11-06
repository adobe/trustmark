# TrustMark WASM Image Utilities

Minimal image processing for WebAssembly using stb libraries.

## Overview

This directory contains lightweight image utilities that replace OpenCV for WASM builds. Uses the excellent [stb](https://github.com/nothings/stb) single-header libraries.

## Files

- `stb_image.h` (276KB) - Image loading (PNG, JPEG, BMP, TGA, etc.)
- `stb_image_resize2.h` (446KB) - High-quality image resizing
- `stb_image_write.h` (70KB) - Image saving (PNG, JPEG, BMP, TGA)
- `image_utils.h` - Simple C++ wrapper API
- `image_utils.cpp` - Implementation

## Size Comparison

```
OpenCV (full):           ~100 MB
OpenCV (minimal):         ~30 MB
stb libraries:          ~0.8 MB (800KB of headers)
Compiled in WASM:         ~1 MB additional

Size savings: 99% smaller than OpenCV!
```

## API

### Loading Images

```cpp
#include "image_utils.h"

// Load any supported format
ImageUtils::Image img = ImageUtils::loadImage("photo.jpg");

if (!img.empty()) {
    std::cout << "Loaded: " << img.width << "x" << img.height 
              << " with " << img.channels << " channels" << std::endl;
}
```

### Resizing

```cpp
// Bilinear interpolation
ImageUtils::Image resized = ImageUtils::resizeImage(img, 256, 256);
```

### Saving Images

```cpp
// Format determined by extension
ImageUtils::saveImage("output.png", img);  // PNG
ImageUtils::saveImage("output.jpg", img);  // JPEG (quality 90)
ImageUtils::saveImage("output.bmp", img);  // BMP
```

### Type Conversions

```cpp
// uint8 [0,255] to float [0,1]
std::vector<float> floats = ImageUtils::uint8ToFloat(img);

// Normalize to [-1, 1]
std::vector<float> normalized = ImageUtils::normalizeImage(img);

// float to uint8
ImageUtils::Image result = ImageUtils::floatToUint8(floats, width, height, 3);
```

### Color Space

```cpp
// BGR ? RGB (swap R and B channels)
ImageUtils::Image rgb = ImageUtils::bgrToRgb(bgr_img);
ImageUtils::Image bgr = ImageUtils::rgbToBgr(rgb_img);
```

## Supported Formats

### Loading (via stb_image.h)
- JPEG (baseline & progressive)
- PNG (1/2/4/8/16-bit-per-channel)
- TGA
- BMP
- PSD
- GIF
- HDR
- PIC
- PNM

### Saving (via stb_image_write.h)
- PNG
- JPEG (quality 90)
- BMP
- TGA

## Integration with ONNX Runtime WASM

The image utilities are compiled into the WASM module by adding to `cmake/onnxruntime_webassembly.cmake`:

```cmake
file(GLOB_RECURSE onnxruntime_webassembly_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/wasm/simple.cpp"
    "${ONNXRUNTIME_ROOT}/wasm/image_utils.cpp"  # Add this line
)
```

## Example Usage in WASM

See `examples/trustmark_wasm_image.cpp` for complete example:

```cpp
// Load image
ImageUtils::Image img = ImageUtils::loadImage(image_path);

// Resize for encoder
ImageUtils::Image resized = ImageUtils::resizeImage(img, 256, 256);

// Normalize to [-1, 1]
std::vector<float> normalized = ImageUtils::normalizeImage(resized);

// Convert HWC to CHW format for ONNX
// ... tensor preparation ...

// Run inference
auto output = session.Run(...);

// Save result
ImageUtils::saveImage("output.png", result_img);
```

## Why Not OpenCV?

| Feature | OpenCV | stb libraries |
|---------|--------|---------------|
| Size | 100+ MB | < 1 MB |
| Dependencies | Many | None |
| WASI Support | Complex | Easy |
| Build Time | Hours | Seconds |
| What We Need | 0.01% of it | 100% of it |

We only need 5 functions from OpenCV (imread, imwrite, resize, cvtColor, convertTo). stb libraries provide exactly what we need with 99% size savings.

## Credits

stb libraries by Sean Barrett: https://github.com/nothings/stb

Public domain, no attribution required (but appreciated!).
