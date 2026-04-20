# WASM Runtime Crash Investigation

## Status: BLOCKER

## Problem
All WASM binaries (including previously built ones) crash on startup with:
```
wasm trap: wasm `unreachable` instruction executed
```

The crash happens in `__main_void` before `main()` is even called, indicating a global constructor or C++ runtime initialization failure.

## Environment
- **Wasmtime Version**: 38.0.3 (d9dc16b28 2025-10-24)
- **WASI SDK**: `/opt/wasi-sdk` (wasm32-wasip2-clang)
- **WASI Preview**: 2 (wasip2)
- **OS**: darwin 25.0.0

## Timeline
- **Nov 6 10:46**: WASM binary built and stored at `onnxruntime-wasi/ort-wasi-simd.wasm`
- **Nov 6 14:56**: Multiple rebuild attempts, all producing binaries that crash
- **Current**: Even "Hello World" minimal C++ programs crash

## What Was Tested
1. ? **Simple "Hello World" WASM** - Crashes
2. ? **Basic ONNX inference test** - Crashes
3. ? **Full TrustMark image processing** - Crashes
4. ? **Clean rebuild from scratch** - Still crashes
5. ? **Older WASM binary from Nov 6 10:46** - Also crashes now

## Key Findings

### 1. Memory Limit Discovery
- Initial WASM memory was set to **16MB**
- TrustMark `encoder_P.ort` model is **33MB**
- This is likely causing memory exhaustion issues (if we could get past the init crash)

### 2. Global Constructor Crash
The crash backtrace shows:
```
0: 0x2cfac2 - ort-wasi-simd.wasm!undefined_weak:main
1: 0xb72c88 - ort-wasi-simd.wasm!__main_void  <-- Crash happens here
2: 0x2cfafa - ort-wasi-simd.wasm!_start
```

`__main_void` is responsible for calling C++ global constructors. This suggests:
- A global object constructor is failing
- WASI Preview 2 ABI incompatibility
- Wasmtime version incompatibility

### 3. WASI Preview 2 Migration
Recent commits in `onnxruntime-wasi`:
- `bbfb429bf`: "make it export wasi:cli/run" (Oct 31, 2025)
- `7c2f11346`: "Webgpu enabled!"
- Working commit: `df82cb919`: "Correct output with a simple model"

The toolchain uses WASI Preview 2 (`wasm32-wasip2-clang`) but wasmtime shows `wasi:cli/run@0.2.3`.

## Hypotheses

### Most Likely: Wasmtime Update Broke Compatibility
- Wasmtime 38.0.3 was released 2025-10-24
- Previously built binaries that may have worked now crash
- WASI Preview 2 is still evolving and breaking changes are common

### Also Possible: WASI SDK Incompatibility
- WASI SDK and wasmtime versions may be mismatched
- The adapter layer (`wit-component:adapter:wasi_snapshot_preview1`) may be incompatible

### Less Likely: ONNX Runtime Issue
- Even minimal C++ programs crash
- No ONNX Runtime code is involved in global constructors for the simple test

## Next Steps to Investigate

1. **Downgrade wasmtime** to a known working version (if one exists)
2. **Check wasmtime compatibility matrix** with WASI SDK
3. **Try WASI Preview 1** instead of Preview 2
4. **Add runtime debug logging** to wasmtime with `WASMTIME_BACKTRACE_DETAILS=1`
5. **Check if there's a component model adapter issue**
6. **Try different WASI SDK version**

## Impact
- **BLOCKER**: Cannot run any WASM code at all
- **Scope**: Affects all WASM builds, not just TrustMark
- **Workaround**: None found yet

## Original Issue (Now Secondary)
The original investigation was into why TrustMark model inference produces garbage output in WASM. We discovered:
- **Root cause identified**: 16MB initial memory vs 33MB model size
- **Fix attempted**: Increase to 64MB or 256MB
- **Result**: Cannot test due to initialization crash

## Files Modified
- `cmake/onnxruntime_webassembly.cmake`: Memory limit changes (reverted to 16MB)
- `onnxruntime/wasm/simple.cpp`: Various test versions
- `onnxruntime/wasm/image_utils.{h,cpp}`: Image processing utilities
- `onnxruntime/wasm/stb_*.h`: STB library headers

## Related Documentation
- `WASM_BUILD.md`: WASM build instructions
- `WASM_RUNTIME_ISSUE.md`: Original inference output issue (now superseded by crash)
- `WASM_RUNTIME_ISSUE_QUICK_TEST.sh`: Test script (non-functional due to crash)
