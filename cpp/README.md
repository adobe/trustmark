# TrustMark C++ Implementation

GPU-accelerated watermarking library using ONNX Runtime.

**Features:**
- ? Native C++ implementation with GPU acceleration (CoreML/CUDA/DirectML)
- ? WebAssembly (WASM) support for wasm32-wasip2 (see [WASM_BUILD.md](WASM_BUILD.md))
- ? Cross-platform (macOS, Linux, Windows)
- ? Command-line and library API

## Repository Structure

### Files to Include in Git

```
cpp/
??? .gitignore              # Ignore build artifacts and dependencies
??? README.md               # This file
??? CMakeLists.txt          # Build configuration
??? build.sh                # Build script
??? fetch_ort.sh            # Script to download ONNX Runtime
??? cmake/                  # CMake configuration files
?   ??? TrustMarkCppConfig.cmake.in
??? trustmark/              # Source code (INCLUDE)
?   ??? execution_provider.h
?   ??? onnx_session.h
?   ??? onnx_session.cpp
?   ??? trustmark.h
?   ??? trustmark.cpp
?   ??? image_processor.h
?   ??? image_processor.cpp
?   ??? bch_ecc.h
?   ??? bch_ecc.cpp
??? examples/               # Example code (INCLUDE)
?   ??? example.cpp
??? models/                 # ONNX models (INCLUDE if distributing)
?   ??? .gitkeep
?   ??? encoder_P.onnx
?   ??? encoder_Q.onnx
?   ??? decoder_P.onnx
?   ??? decoder_Q.onnx
??? output/                 # Output directory (EXCLUDE contents)
    ??? .gitkeep            # Keep directory structure
```

### Files to Exclude (in .gitignore)

**Build Artifacts:**
- `build/` - CMake build directory
- `*.o`, `*.a`, `*.so`, `*.dylib` - Compiled binaries
- `trustmark_example` - Compiled executable
- `CMakeCache.txt`, `CMakeFiles/` - CMake generated files
- `compile_commands.json` - Clang tooling database

**Dependencies:**
- `onnxruntime/` - Full ONNX Runtime source (fetch via script)
- `third_party/ort/` - Pre-built ONNX Runtime libraries (fetch via script)

**Generated Files:**
- `output/*.jpg`, `output/*.png` - Watermarked images (runtime output)
- `.cache/` - IDE cache
- `.DS_Store`, `Thumbs.db` - OS files

## Quick Start

### 1. Install Dependencies

```bash
# macOS
brew install cmake opencv

# Linux
sudo apt install cmake libopencv-dev

# Windows
# Use vcpkg or download OpenCV manually
```

### 2. Fetch Dependencies

```bash
cd cpp

# Fetch ONNX Runtime library
./fetch_ort.sh

# Fetch ONNX models (required)
./fetch_models.sh
# Or fetch specific variant only: ./fetch_models.sh P
```

### 3. Build

```bash
mkdir -p build
cd build
cmake ..
make -j8
```

### 4. Run

**With CPU:**
```bash
./trustmark_example /path/to/image.jpg
```

**With GPU (macOS with CoreML):**
```bash
TRUSTMARK_USE_GPU=1 ./trustmark_example /path/to/image.jpg
```

**With GPU (Linux with CUDA):**
```bash
TRUSTMARK_USE_GPU=1 ./trustmark_example /path/to/image.jpg
```

## GPU Support

### Execution Providers

- **CPU** - Default, works everywhere
- **CoreML** - macOS/iOS (Neural Engine + GPU)
- **CUDA** - Linux/Windows with NVIDIA GPUs
- **DirectML** - Windows with any GPU

### Environment Variable

Set `TRUSTMARK_USE_GPU=1` to enable GPU acceleration. The library will automatically:
1. Select the appropriate provider for your platform
2. Fall back to CPU if GPU is unavailable
3. Log which provider is being used

## API Usage

```cpp
#include "trustmark/trustmark.h"

// Initialize with GPU support
TrustMark::TrustMark trustmark(
    false,                              // useECC
    true,                               // verbose
    100,                                // secretLen
    "P",                                // modelType
    TrustMark::EncodingType::BCH_5,     // encodingType
    1.0f,                               // concentrateWmRegion
    TrustMark::ExecutionProvider::CoreML, // GPU acceleration
    0                                   // device ID
);

// Encode watermark
cv::Mat watermarked = trustmark.encode(
    coverImage,
    "0110111100000110...", // 100-bit secret
    TrustMark::Mode::BINARY,
    0.95f,
    "bilinear"
);

// Decode watermark
auto [bits, ok, version] = trustmark.decode(
    watermarkedImage,
    TrustMark::Mode::BINARY
);
```

## Directory Structure After Build

```
cpp/
??? build/                      # Build directory (git ignored)
?   ??? trustmark_example       # Compiled executable
?   ??? libtrustmark_cpp.a      # Static library
??? output/                     # Output directory (contents ignored)
?   ??? watermarked_*.jpg       # Generated watermarked images
?   ??? debug_stego_*.png       # Debug outputs
??? third_party/                # Downloaded dependencies (git ignored)
?   ??? ort/                    # ONNX Runtime
?       ??? include/
?       ??? lib/
??? onnxruntime/                # Full source (git ignored, optional)
```

## Development Workflow

### Clean Build
```bash
cd build
make clean
cmake ..
make -j8
```

### Rebuild After Code Changes
```bash
cd build
make -j8
```

### Add to Git
```bash
# Add source code
git add trustmark/
git add examples/
git add CMakeLists.txt
git add README.md

# Models (if distributing)
git add models/*.onnx

# DON'T add build artifacts
# (already in .gitignore)
```

## CI/CD Considerations

For CI/CD pipelines:
1. Run `fetch_ort.sh` to download ONNX Runtime
2. Cache `third_party/ort/` between builds
3. Don't commit `third_party/` to git
4. Models should be versioned separately or via Git LFS

## License

See LICENSE file in repository root.
