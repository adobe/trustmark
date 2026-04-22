#!/usr/bin/env bash
# Build TrustMark HTTP component (wasi:http/incoming-handler + wasi:webgpu)
#
# Produces: build_http/trustmark-http.wasm  (WASI P2 component)
#
# Reuses existing onnxruntime-wasi/build_wasi libs — no full ORT cmake rebuild.
# Only required if build_wasi does not already exist:
#   cd onnxruntime-wasi && ./build_wasi.sh Release -Donnxruntime_USE_WEBGPU=ON
#
# Prerequisites:
#   - WASI_SDK_PATH set to wasi-sdk installation
#   - onnxruntime-wasi symlink/dir with a completed build_wasi/
#   - wasm-tools in PATH (cargo install wasm-tools)
#   - wasi-preview1-component-adapter-provider reactor adapter in ~/.cargo/registry

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_DIR="$SCRIPT_DIR/onnxruntime-wasi"
ORT_BUILD="$ORT_DIR/build_wasi"
BUILD_HTTP="$SCRIPT_DIR/build_http"

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ -z "${WASI_SDK_PATH:-}" ]]; then
    echo "Error: WASI_SDK_PATH not set" >&2; exit 1
fi
CXX="$WASI_SDK_PATH/bin/clang++"
CC="$WASI_SDK_PATH/bin/clang"
SYSROOT="$WASI_SDK_PATH/share/wasi-sysroot"

if [[ ! -f "$CXX" ]]; then
    echo "Error: $CXX not found" >&2; exit 1
fi
if [[ ! -d "$ORT_BUILD" ]]; then
    echo "Error: $ORT_BUILD does not exist." >&2
    echo "Run: cd $ORT_DIR && ./build_wasi.sh Release -Donnxruntime_USE_WEBGPU=ON" >&2
    exit 1
fi

WEBGPU_HEADERS="$ORT_BUILD/_deps/wasi_webgpu_headers-src"
ORT_INCLUDE="$ORT_DIR/include/onnxruntime/core/session"
WASM_SRC="$SCRIPT_DIR/wasm"
ABSEIL_STUBS="$ORT_DIR/cmake/wasi_abseil_stubs.cc"

echo "=========================================="
echo "TrustMark HTTP Component Build"
echo "=========================================="
echo "  ORT build  : $ORT_BUILD"
echo "  Output     : $BUILD_HTTP/trustmark-http.wasm"
echo "=========================================="

# ── Step 1: Stage TrustMark sources into onnxruntime-wasi ────────────────────
echo ""
echo "[1/4] Staging TrustMark sources..."
"$SCRIPT_DIR/prepare_ort_build.sh" "$ORT_DIR"

# ── Step 2: Compile TrustMark HTTP sources ────────────────────────────────────
echo ""
echo "[2/4] Compiling HTTP component sources..."
mkdir -p "$BUILD_HTTP"

# Flags matching the existing ORT build (from build_wasi/CMakeFiles/.../link.txt)
BASE_FLAGS=(
    "--target=wasm32-wasi"
    "--sysroot=$SYSROOT"
    "-fno-rtti"
    "-fno-unwind-tables"
    "-fno-asynchronous-unwind-tables"
    "-ffunction-sections"
    "-fdata-sections"
    "-msimd128"
    "-O3"
    "-D_WASI_EMULATED_SIGNAL"
    "-D_WASI_EMULATED_MMAN"
    "-D__wasm__"
    "-D__wasi__"
    "-DNDEBUG"
    "-Wno-deprecated"
)

INCLUDES=(
    "-I$ORT_INCLUDE"
    "-I$WASM_SRC"
    "-I$WASM_SRC/wasi_http"
    "-I$WEBGPU_HEADERS"
)

echo "  Compiling http_handler.cpp..."
"$CXX" "${BASE_FLAGS[@]}" "${INCLUDES[@]}" -std=c++17 -fno-exceptions -DORT_NO_EXCEPTIONS \
    -c "$WASM_SRC/http_handler.cpp" -o "$BUILD_HTTP/http_handler.o"

# image_utils.cpp: no exceptions
echo "  Compiling image_utils.cpp..."
"$CXX" "${BASE_FLAGS[@]}" "${INCLUDES[@]}" -std=c++17 -fno-exceptions \
    -c "$WASM_SRC/image_utils.cpp" -o "$BUILD_HTTP/image_utils.o"

# trustmark_http.c: generated wasi:http bindings
echo "  Compiling trustmark_http.c..."
"$CC" "${BASE_FLAGS[@]}" "${INCLUDES[@]}" -std=c11 -fno-exceptions \
    -c "$WASM_SRC/wasi_http/trustmark_http.c" -o "$BUILD_HTTP/trustmark_http.o"

# wasi_abseil_stubs.cc
echo "  Compiling wasi_abseil_stubs.cc..."
"$CXX" "${BASE_FLAGS[@]}" -std=c++17 -fno-exceptions \
    -c "$ABSEIL_STUBS" -o "$BUILD_HTTP/wasi_abseil_stubs.o"

# WebGPU C adapter
echo "  Compiling webgpu.c / imports.c..."
"$CC" "${BASE_FLAGS[@]}" "-I$WEBGPU_HEADERS" -std=c11 -fno-exceptions \
    -c "$WEBGPU_HEADERS/webgpu.c" -o "$BUILD_HTTP/webgpu.o"
"$CC" "${BASE_FLAGS[@]}" "-I$WEBGPU_HEADERS" -std=c11 -fno-exceptions \
    -c "$WEBGPU_HEADERS/imports.c" -o "$BUILD_HTTP/imports.o"

echo "  ✓ Compilation complete"

# ── Step 3: Link ──────────────────────────────────────────────────────────────
# Uses the same lib list as the existing build_wasi/link.txt, plus:
#   - trustmark_http_component_type.o  (declares wasi:http export)
#   - trustmark_http.o                 (wasi:http C bindings)
#   - http_handler.o                   (entry point — exports handle())
# Omits: simple.cpp.obj (no main()); uses -mexec-model=reactor instead.
echo ""
echo "[3/4] Linking..."

CORE_WASM="$BUILD_HTTP/trustmark-http-core.wasm"

"$CXX" \
    "--target=wasm32-wasi" \
    "--sysroot=$SYSROOT" \
    "-fno-rtti" \
    "-msimd128" \
    "-O3" \
    "-D_WASI_EMULATED_SIGNAL" \
    "-D_WASI_EMULATED_MMAN" \
    "-Wno-deprecated" \
    \
    "-mexec-model=reactor" \
    "-Wl,--allow-undefined" \
    "-Wl,--stack-first" \
    "-Wl,-z,stack-size=1048576" \
    "-Wl,--initial-memory=16777216" \
    "-Wl,--max-memory=4294967296" \
    "-Wl,--gc-sections" \
    \
    "$BUILD_HTTP/http_handler.o" \
    "$BUILD_HTTP/image_utils.o" \
    "$BUILD_HTTP/trustmark_http.o" \
    "$BUILD_HTTP/wasi_abseil_stubs.o" \
    "$BUILD_HTTP/webgpu.o" \
    "$BUILD_HTTP/imports.o" \
    \
    "$WEBGPU_HEADERS/imports_component_type.o" \
    "$WASM_SRC/wasi_http/trustmark_http_component_type.o" \
    \
    "-lwasi-emulated-signal" \
    "-lwasi-emulated-mman" \
    \
    "$ORT_BUILD/_deps/protobuf-build/libprotobuf-lite.a" \
    "$ORT_BUILD/_deps/onnx-build/libonnx.a" \
    "$ORT_BUILD/_deps/onnx-build/libonnx_proto.a" \
    "$ORT_BUILD/libonnxruntime_common.a" \
    "$ORT_BUILD/libonnxruntime_lora.a" \
    "$ORT_BUILD/libonnxruntime_flatbuffers.a" \
    "$ORT_BUILD/libonnxruntime_framework.a" \
    "$ORT_BUILD/libonnxruntime_graph.a" \
    "$ORT_BUILD/libonnxruntime_mlas.a" \
    "$ORT_BUILD/libonnxruntime_optimizer.a" \
    "$ORT_BUILD/libonnxruntime_providers.a" \
    "$ORT_BUILD/libonnxruntime_providers_webgpu.a" \
    "$ORT_BUILD/libonnxruntime_session.a" \
    "$ORT_BUILD/libonnxruntime_util.a" \
    "$ORT_BUILD/_deps/re2-build/libre2.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_reflection.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_private_handle_accessor.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_commandlineflag.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_commandlineflag_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_marshalling.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_config.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/flags/libabsl_flags_program_name.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_cord.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_info.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_cord_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_functions.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_cordz_handle.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/crc/libabsl_crc_cord_state.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/crc/libabsl_crc32c.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/crc/libabsl_crc_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/crc/libabsl_crc_cpu_detect.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/container/libabsl_hashtablez_sampler.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/profiling/libabsl_exponential_biased.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_conditions.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_check_op.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_message.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_format.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_str_format_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_log_sink_set.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_globals.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_sink.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_globals.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/hash/libabsl_hash.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/hash/libabsl_city.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_vlog_config_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_fnmatch.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_examine_stack.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_strerror.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_structured_proto.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_proto.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/log/libabsl_log_internal_nullguard.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_leak_check.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/synchronization/libabsl_synchronization.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/synchronization/libabsl_kernel_timeout_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/time/libabsl_time.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/time/libabsl_time_zone.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/time/libabsl_civil_time.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_symbolize.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_strings.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_strings_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/numeric/libabsl_int128.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/strings/libabsl_string_view.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_demangle_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_demangle_rust.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_decode_rust_punycode.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_utf8_for_code_point.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_stacktrace.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/debugging/libabsl_debugging_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/synchronization/libabsl_graphcycles_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_malloc_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_tracing_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_base.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_spinlock_wait.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a" \
    "$ORT_BUILD/_deps/abseil_cpp-build/absl/base/libabsl_log_severity.a" \
    \
    -o "$CORE_WASM"

echo "✓ Core WASM module: $CORE_WASM ($(du -sh "$CORE_WASM" | cut -f1))"

# ── Step 4: Wrap as WASI P2 component ────────────────────────────────────────
echo ""
echo "[4/4] Wrapping as WASI Preview 2 component..."

ADAPTER=$(find ~/.cargo/registry -name "wasi_snapshot_preview1.reactor.wasm" \
    -path "*/wasi-preview1-component-adapter-provider*" 2>/dev/null \
    | sort -rV | head -1)

if [[ -z "$ADAPTER" ]]; then
    echo "Warning: wasi_snapshot_preview1.reactor.wasm not found" >&2
    echo "Install: cargo add wasi-preview1-component-adapter-provider" >&2
    echo "Skipping component wrap — $CORE_WASM is a raw WASM module" >&2
    echo ""
    echo "Output: $CORE_WASM"
else
    COMPONENT_WASM="$BUILD_HTTP/trustmark-http.wasm"
    wasm-tools component new "$CORE_WASM" \
        --adapt "wasi_snapshot_preview1=$ADAPTER" \
        -o "$COMPONENT_WASM"

    MAGIC=$(dd if="$COMPONENT_WASM" bs=1 skip=4 count=2 2>/dev/null | xxd -p)
    if [[ "$MAGIC" == "0d00" ]]; then
        echo "✓ WASI P2 component: $COMPONENT_WASM ($(du -sh "$COMPONENT_WASM" | cut -f1)) — magic ok"
    else
        echo "✓ Component: $COMPONENT_WASM — warning: unexpected magic bytes ($MAGIC)"
    fi
fi

echo ""
echo "Test with curl once wasmCloud is running:"
echo "  curl -X POST http://localhost:8080/encode \\"
echo "    -H 'Content-Type: image/png' \\"
echo "    --data-binary @input.png \\"
echo "    -o watermarked.png"
