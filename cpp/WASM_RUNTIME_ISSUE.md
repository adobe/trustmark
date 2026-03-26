# WASM ONNX Runtime Issue: Incorrect Model Output

## Summary

The ONNX Runtime WASI build (minimal) produces incorrect output when running TrustMark encoder models, despite correct input preprocessing. Native ONNX Runtime works correctly with the same `.ort` model file.

## Status

**BLOCKER**: WASM image watermarking does not work due to ONNX Runtime WASI producing garbage output.

- ? Image preprocessing: **CORRECT** (verified identical to OpenCV)
- ? Model file (.ort): **CORRECT** (works with native ONNX Runtime)
- ? Input tensors: **CORRECT** (verified identical values)
- ? Model inference: **BROKEN** in WASM (produces near-zero output)

## Test Case

### Environment

- **ONNX Runtime WASI**: Commit `df82cb919729d675ff4ae637fd2ee2f50d294df9` from https://github.com/MendyBerger/onnxruntime
- **WASI SDK**: Version 28
- **Model File**: `encoder_P.ort` (converted from `encoder_P.onnx`)
- **Input Image**: `images/ufo_240.jpg` (240x240 JPEG, dark blue sky, UFO flying saucer)

### Model Details

```
Model: encoder_P.ort
Type: TrustMark encoder (image watermarking)
Inputs:
  - Input 0: "onnx::Concat_0" [1, 3, 256, 256] float32 (BGR image, normalized to [-1, 1])
  - Input 1: "onnx::Gemm_1" [1, 100] float32 (secret watermark bits)
Output:
  - Output 0: "image" [1, 3, 256, 256] float32 (watermarked image in [-1, 1])
```

### Expected Behavior (Native ONNX Runtime)

**Input Image First Pixel (after preprocessing):**
```
Format: BGR, normalized to [-1, 1]
Channel 0 (Blue):  -0.6
Channel 1 (Green): -0.717647
Channel 2 (Red):   -0.921569
```

**Model Output First Pixel:**
```
Format: CHW (channel, height, width)
Channel 0: -0.772725
Channel 1: -0.7849
Channel 2: -0.951786
```

**Output Statistics:**
```
Range: [-0.988117, 0.978502]
Average |value|: 0.500844
Result: Correct watermarked UFO image
```

### Actual Behavior (WASM ONNX Runtime)

**Input Image First Pixel (after preprocessing):**
```
Format: BGR, normalized to [-1, 1]
Channel 0 (Blue):  -0.6          ? CORRECT
Channel 1 (Green): -0.717647     ? CORRECT
Channel 2 (Red):   -0.921569     ? CORRECT
```

**Model Output First Pixel:**
```
Format: CHW (channel, height, width)
Channel 0: -0.000845723    ? WRONG (expected: -0.77)
Channel 1:  0.0242998      ? WRONG (expected: -0.78)
Channel 2: -0.0576184      ? WRONG (expected: -0.95)
```

**Output Statistics:**
```
Range: [-1, 0.999898]
All values near zero (~0.0)
Average |value|: ~0.03 (expected: ~0.5)
Result: Gray/greenish garbage image
```

## Reproduction Steps

### 1. Native ONNX Runtime Test (Works Correctly)

```bash
cd trustmark/cpp

# Compile native test
g++ -std=c++17 test_ort_model_direct.cpp \
  -I/opt/homebrew/include/opencv4 \
  -Ithird_party/ort/include \
  -L/opt/homebrew/lib -Lthird_party/ort/lib \
  -lonnxruntime \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
  -o test_ort_direct

# Run test
DYLD_LIBRARY_PATH=third_party/ort/lib:$DYLD_LIBRARY_PATH ./test_ort_direct

# Output: test_ort_native_output.png (correct UFO image)
```

**Expected Output:**
```
? Model loaded successfully
? Inference completed
Output CHW (first pixel):
  Channel 0: -0.772725
  Channel 1: -0.7849
  Channel 2: -0.951786
Average |value|: 0.500844
? Output has real values - .ort model works!
```

### 2. WASM ONNX Runtime Test (Produces Garbage)

```bash
cd trustmark/cpp

# Build WASM (already includes TrustMark code)
cd onnxruntime-wasi
export WASI_SDK_PATH=/opt/wasi-sdk
./build_wasi_simple.sh

# Run WASM
cd ..
wasmtime --dir=.::.  --dir=models::/models --dir=../images::/images \
  onnxruntime-wasi/build_wasi/ort-wasi-simd.wasm \
  /models/encoder_P.ort /images/ufo_240.jpg

# Output: output_watermarked.png (gray/green garbage)
```

**Actual Output:**
```
? Model loaded successfully
? Inference completed
DEBUG: Input CHW (first pixel): -0.6, -0.717647, -0.921569  ? INPUT CORRECT
DEBUG: Output CHW (first pixel): -0.0008, 0.024, -0.058    ? OUTPUT WRONG
```

## Verification: Preprocessing is Identical

Verified that stb_image + our preprocessing produces **identical** results to OpenCV:

```cpp
// OpenCV
OpenCV first pixel (BGR): (51,36,10)
After normalize [-1,1]: (-0.6,-0.717647,-0.921569)

// stb_image + ImageUtils
stb_image first pixel (RGB): (10,36,51)
After RGB->BGR: (51,36,10)
After normalize [-1,1]: (-0.6,-0.717647,-0.921569)

// Difference
Difference (B,G,R): (5.96046e-08, 0, 0)
? Preprocessing is IDENTICAL!
```

## Analysis

### What We Know

1. **? The .ort model is valid**
   - Works correctly with native ONNX Runtime 1.19.2
   - Conversion from .onnx to .ort is successful

2. **? Input preprocessing is correct**
   - stb_image loads images identically to OpenCV
   - BGR conversion works correctly
   - Normalization produces identical values
   - Input tensor contains correct values

3. **? WASM ONNX Runtime produces wrong output**
   - Model loads successfully (no errors)
   - Inference completes (no errors)
   - Output tensor shape is correct [1, 3, 256, 256]
   - Output values are wrong (near zero instead of ~[-1, 1] range)

### Possible Root Causes

1. **Missing Operators in Minimal Build**
   - The ONNX Runtime WASI minimal build may not include all operators
   - TrustMark models use complex operations that may be stubbed out
   - Hypothesis: Operators run but produce identity/zero outputs

2. **Floating Point Precision Issues**
   - WASM may handle float32 differently than native
   - However, input values are correct, suggesting this is not the issue

3. **Memory/Buffer Issues**
   - Tensor memory may not be properly initialized or passed
   - However, shape is correct and no crashes occur

4. **Build Configuration**
   - Missing compiler flags or optimizations
   - The minimal build uses `-Donnxruntime_DISABLE_ABSEIL=ON` and other flags

5. **WASI Runtime Limitations**
   - wasmtime may not support all operations needed
   - Some CPU instructions may not be available

## Impact

- **WASM builds cannot perform actual watermarking** with real images
- WASM can load models and demonstrate structure/API but not inference
- **Native C++ implementation works perfectly** and should be used for production

## Workarounds

### For Development
- Use native C++ (`./encode_decode`) for actual watermarking
- Use WASM only for model loading/structure demonstration

### For Deployment
- Deploy native C++ binaries, not WASM
- If WASM is required, investigate:
  - Full ONNX Runtime build (not minimal)
  - Different WASM runtime (wasmer, wavm)
  - Alternative model format (TensorFlow Lite, etc.)

## Files for Reproduction

### Test Files
- `test_ort_model_direct.cpp` - Native ONNX Runtime test (works)
- `compare_preprocessing.cpp` - Verifies stb_image vs OpenCV (identical)
- `onnxruntime-wasi/onnxruntime/wasm/simple.cpp` - WASM test (fails)

### Model Files
- `models/encoder_P.onnx` - Original ONNX model (98MB)
- `models/encoder_P.ort` - Converted ORT model (33MB)
- Conversion command:
  ```bash
  python3.11 /path/to/onnxruntime/tools/python/convert_onnx_models_to_ort.py \
    encoder_P.onnx --output_dir .
  ```

### Image Files
- Input: `../images/ufo_240.jpg` (18KB, 240x240, dark blue sky with UFO)
- Expected output: Watermarked UFO image with subtle modifications
- Actual WASM output: Gray/green distorted image

## Next Steps

### For Users
1. **Use native C++ implementation** - fully functional
2. Avoid WASM for actual watermarking until this is resolved

### For Debugging
1. Test with simpler models to isolate the issue
2. Try full ONNX Runtime WASM build (not minimal)
3. Check ONNX Runtime WASI issue tracker
4. Report to https://github.com/MendyBerger/onnxruntime/issues

### For Reporting
This document can be attached to bug reports with:
- Specific model file: `encoder_P.ort`
- Specific input: `ufo_240.jpg`
- Expected output values: Listed above
- Actual output values: Listed above
- Reproduction steps: Listed above

## Technical Details

### Build Configuration

**WASM Build:**
```cmake
WASI_SDK_PATH=/opt/wasi-sdk
-Donnxruntime_DISABLE_ABSEIL=ON
-Donnxruntime_BUILD_SHARED_LIB=OFF
-Donnxruntime_MINIMAL_BUILD=ON
-DCMAKE_BUILD_TYPE=MinSizeRel
```

**Compiler:**
```
wasm32-wasip2-clang++ from WASI SDK 28
Target: wasm32-wasi-preview2
```

### Runtime

```
wasmtime 27.0.0
Invoked with: --dir=.::.  --dir=models::/models --dir=../images::/images
```

### Model Conversion

```bash
python3.11 onnxruntime/tools/python/convert_onnx_models_to_ort.py \
  --output_dir models/ \
  encoder_P.onnx
```

## References

- ONNX Runtime: https://github.com/microsoft/onnxruntime
- ONNX Runtime WASI Fork: https://github.com/MendyBerger/onnxruntime
- WASI SDK: https://github.com/WebAssembly/wasi-sdk
- TrustMark Project: (internal)

---

**Document Version**: 1.0  
**Date**: November 6, 2024  
**Status**: OPEN - Awaiting ONNX Runtime WASI fix or investigation
