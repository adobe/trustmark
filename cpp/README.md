# TrustMark C++ Library

A C++ implementation of the TrustMark watermarking system using ONNX Runtime for efficient inference.

## Requirements

### Dependencies
- **ONNX Runtime**: exactly 1.19.2 (staged locally under `cpp/third_party/ort`)
- **OpenCV**
- **CMake**

## Installation

### Prerequisites

1. **Install ONNX Runtime locally (no system install)**:
   ```bash
   # From repo root or within cpp/
   bash cpp/fetch_ort.sh 1.19.2
   ```

   This stages headers and libraries under `cpp/third_party/ort` for isolated linking.

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

#### Option 1: Using the Build Script (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adobe/trustmark.git
   cd trustmark/cpp
   ```

2. **Run the build script**:
   ```bash
   bash build.sh
   ```

   This script will automatically:
   - Create the build directory
   - Configure CMake with Release build type
   - Build the library (`libtrustmark_cpp.a`) and example executable (`trustmark_example`)
   - Use optimal parallel compilation

#### Option 2: Manual CMake Build

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

4. **Build the library and example**:
   ```bash
   make -j$(nproc)  # Linux/macOS
   # or
   cmake --build . --config Release  # Cross-platform
   ```

5. **Place models in the models/ directory**:
    Use the rust xtask `cargo xtask fetch-models` and copy the files from `../rust/models`

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
# The example is built automatically with the library
# Navigate to the build directory
cd build

# Run with an input image and custom message
./trustmark_example input.jpg "Secret message"

# Run with default message
./trustmark_example input.jpg

# Example with a test image (if available)
./trustmark_example ../images/ripley.jpg "Hello TrustMark!"
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
