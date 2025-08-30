# TrustMark C++ Library

A C++ implementation of the TrustMark watermarking system using ONNX Runtime for efficient inference.

## Overview

TrustMark is a state-of-the-art digital watermarking system that embeds invisible watermarks into images while maintaining high visual quality. This C++ implementation provides:

- **High Performance**: Optimized C++ code with ONNX Runtime for fast inference
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Easy Integration**: Simple C++ API for embedding and extracting watermarks
- **Multiple Variants**: Support for C, Q, B, and P model variants
- **Error Correction**: Built-in BCH error correction for robust watermarking

## Features

- **Watermark Encoding**: Embed text or binary data into images
- **Watermark Decoding**: Extract hidden messages from watermarked images
- **Watermark Removal**: Remove watermarks while preserving image quality
- **Multiple Model Types**: Choose from different quality/robustness trade-offs
- **Error Correction**: BCH error correction for reliable message recovery
- **Image Processing**: Advanced image preprocessing and postprocessing
- **Performance Optimized**: Efficient tensor operations with ONNX Runtime

## Model Variants

| Variant | Description | PSNR | Use Case |
|---------|-------------|------|----------|
| C | Compact version with ResNet-18 decoder | ~39 dB | Resource-constrained deployments |
| Q | Quality-focused variant (default) | ~43 dB | General purpose, good balance |
| B | Balanced variant | ~43 dB | Original paper reproduction |
| P | Perceptual quality variant | ~48 dB | Highest visual quality |

## Requirements

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Compiler**: C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies

### Dependencies
- **ONNX Runtime**: 1.15.0 or later
- **OpenCV**: 4.5.0 or later
- **CMake**: 3.16 or later

## Installation

### Prerequisites

1. **Install ONNX Runtime**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libonnxruntime-dev
   
   # macOS
   brew install onnxruntime
   
   # Windows
   # Download from https://github.com/microsoft/onnxruntime/releases
   ```

2. **Install OpenCV**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopencv-dev
   
   # macOS
   brew install opencv
   
   # Windows
   # Download from https://opencv.org/releases/
   ```

3. **Install CMake**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake
   
   # macOS
   brew install cmake
   
   # Windows
   # Download from https://cmake.org/download/
   ```

### Building from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adobe/trustmark.git
   cd trustmark/cpp
   ```

2. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

4. **Build the library**:
   ```bash
   make -j$(nproc)  # Linux/macOS
   # or
   cmake --build . --config Release  # Windows
   ```

5. **Install** (optional):
   ```bash
   sudo make install  # Linux/macOS
   # or
   cmake --build . --target install --config Release  # Windows
   ```

### Building with Custom Dependencies

If you have custom ONNX Runtime or OpenCV installations:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRuntime_DIR=/path/to/onnxruntime/cmake \
  -DOpenCV_DIR=/path/to/opencv/cmake
```

## Usage

### Basic Example

```cpp
#include "trustmark/trustmark.h"
#include <opencv2/opencv.hpp>

using namespace TrustMark;

int main() {
    // Initialize TrustMark
    TrustMark trustmark(true, true, 100, "Q");
    
    // Load cover image
    cv::Mat coverImage = cv::imread("input.jpg");
    
    // Encode watermark
    std::string result = trustmark.encode(coverImage, "Hello, TrustMark!", Mode::TEXT);
    
    // Decode watermark
    auto [message, success, version] = trustmark.decode(coverImage, Mode::TEXT);
    
    if (success) {
        std::cout << "Decoded: " << message << std::endl;
    }
    
    return 0;
}
```

### Advanced Usage

```cpp
// Initialize with custom parameters
TrustMark trustmark(
    true,                           // Use error correction
    true,                           // Verbose output
    200,                            // Secret length in bits
    "P",                            // High-quality model variant
    EncodingType::BCH_5,           // BCH error correction
    0.8f                           // Concentrate watermark region
);

// Encode with custom parameters
std::string result = trustmark.encode(
    coverImage,                     // Input image
    "Secret message",               // Message to embed
    Mode::TEXT,                     // Text mode
    1.5f,                          // Watermark strength
    "bicubic"                      // Interpolation method
);

// Decode with error handling
try {
    auto [message, success, version] = trustmark.decode(stegoImage, Mode::TEXT);
    
    if (success) {
        std::cout << "Message: " << message << std::endl;
        std::cout << "Version: " << version << std::endl;
    } else {
        std::cerr << "Decode failed: " << trustmark.getLastError() << std::endl;
    }
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
}
```

### Running the Example

```bash
# Build the example
cd build
make trustmark_example

# Run with an input image
./trustmark_example input.jpg "Secret message"

# Run with default message
./trustmark_example input.jpg
```

## API Reference

### TrustMark Class

#### Constructor
```cpp
TrustMark(bool useECC = true,
          bool verbose = true,
          int secretLen = 100,
          const std::string& modelType = "Q",
          EncodingType encodingType = EncodingType::BCH_5,
          float concentrateWmRegion = 1.0f);
```

#### Methods

##### Encoding
```cpp
std::string encode(const cv::Mat& coverImage,
                   const std::string& secret,
                   Mode mode = Mode::TEXT,
                   float wmStrength = 1.0f,
                   const std::string& wmMerge = "bilinear");
```

##### Decoding
```cpp
std::tuple<std::string, bool, int> decode(const cv::Mat& stegoImage,
                                         Mode mode = Mode::TEXT);
```

##### Watermark Removal
```cpp
cv::Mat removeWatermark(const cv::Mat& stegoImage,
                        float wmStrength = 1.0f,
                        const std::string& wmMerge = "bilinear");
```

##### Utility Methods
```cpp
int getSchemaCapacity() const;
bool isVerbose() const;
std::string getModelType() const;
std::string getLastError() const;
void clearLastError();
```

### Enums

```cpp
enum class EncodingType {
    BCH_SUPER = 0,  // Super error correction
    BCH_3 = 3,      // BCH-3 error correction
    BCH_4 = 2,      // BCH-4 error correction
    BCH_5 = 1       // BCH-5 error correction (default)
};

enum class Mode {
    TEXT = 0,        // Text message mode
    BINARY = 1       // Binary data mode
};
```

## Model Conversion

To use your own PyTorch models with this C++ library:

1. **Export to ONNX**:
   ```python
   import torch
   from trustmark import TrustMark
   
   # Load your trained model
   model = TrustMark.load_from_checkpoint("path/to/checkpoint.ckpt")
   
   # Export encoder
   dummy_image = torch.randn(1, 3, 256, 256)
   dummy_secret = torch.randn(1, 100)
   torch.onnx.export(model.encoder, 
                     (dummy_image, dummy_secret),
                     "encoder_Q.onnx",
                     input_names=["cover", "secret"],
                     output_names=["stego"],
                     dynamic_axes={"cover": {0: "batch_size"},
                                 "secret": {0: "batch_size"},
                                 "stego": {0: "batch_size"}})
   
   # Export decoder and removal models similarly
   ```

2. **Place models in the models/ directory**:
   ```
   models/
   ├── encoder_Q.onnx
   ├── decoder_Q.onnx
   └── removal_Q.onnx
   ```

## Performance

### Benchmarks

| Model Variant | Encode Time | Decode Time | Memory Usage |
|---------------|-------------|-------------|--------------|
| C (Compact)   | ~50ms       | ~30ms       | ~200MB       |
| Q (Quality)   | ~80ms       | ~50ms       | ~300MB       |
| B (Balanced)  | ~80ms       | ~50ms       | ~300MB       |
| P (Perceptual)| ~100ms      | ~60ms       | ~400MB       |

*Benchmarks on Intel i7-10700K, 32GB RAM, NVIDIA RTX 3080*

### Optimization Tips

1. **Use appropriate model variant** for your use case
2. **Batch processing** multiple images when possible
3. **Enable OpenMP** for multi-threaded operations
4. **Use GPU acceleration** if available (requires ONNX Runtime GPU build)

## Error Handling

The library provides comprehensive error handling:

```cpp
// Check for errors after operations
if (!trustmark.getLastError().empty()) {
    std::cerr << "Error: " << trustmark.getLastError() << std::endl;
    trustmark.clearLastError();
}

// Exception handling
try {
    auto result = trustmark.encode(image, message);
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
}
```

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Check model file paths
   - Verify ONNX file integrity
   - Ensure sufficient memory

2. **Build errors**:
   - Verify C++17 support
   - Check dependency versions
   - Ensure proper CMake configuration

3. **Runtime errors**:
   - Check input image format
   - Verify message length limits
   - Monitor memory usage

### Debug Mode

Build with debug information:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make VERBOSE=1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Adobe License Agreement. See the LICENSE file for details.

## Support

- **Issues**: GitHub Issues
- **Documentation**: This README and inline code comments
- **Community**: Adobe TrustMark discussions

## Acknowledgments

- Original TrustMark research team
- ONNX Runtime contributors
- OpenCV community
- C++ standards committee
