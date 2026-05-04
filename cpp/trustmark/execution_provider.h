#pragma once

namespace TrustMark {

// Execution provider enum for ONNX Runtime
enum class ExecutionProvider {
    CPU,
    CUDA,
    CoreML,
    DirectML
};

} // namespace TrustMark
