#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <onnxruntime_cxx_api.h>

// Simple TrustMark WASM example
// Demonstrates loading and running TrustMark encoder/decoder models

void print_model_info(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    std::cout << "\nModel Information:" << std::endl;
    std::cout << "  Number of inputs: " << num_inputs << std::endl;

    for (size_t i = 0; i < num_inputs; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "  Input " << i << ": " << input_name.get() << " [";
        for (size_t j = 0; j < shape.size(); j++) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "  Number of outputs: " << num_outputs << std::endl;
    for (size_t i = 0; i < num_outputs; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "  Output " << i << ": " << output_name.get() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "TrustMark WASM Example" << std::endl;
    std::cout << "======================" << std::endl;

    // Check arguments
    if (argc < 2) {
        std::cout << "\nUsage: " << argv[0] << " <model.ort>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  wasmtime --dir=models::/models " << argv[0] << " /models/encoder_P.ort" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    std::cout << "\nLoading model: " << model_path << std::endl;

    // Initialize ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrustMarkWASM");
    std::cout << "? ONNX Runtime initialized" << std::endl;

    // Configure session options (CPU only for WASI)
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    std::cout << "? Session options configured" << std::endl;

    // Load the model
    Ort::Session session(env, model_path, session_options);
    std::cout << "? Model loaded successfully!" << std::endl;

    // Print model information
    print_model_info(session);

    // Run a simple test inference if it's a TrustMark encoder
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::string input_name_str(input_name.get());

    // Check if this looks like a TrustMark encoder (has 2 inputs: image and secret)
    if (session.GetInputCount() == 2) {
        std::cout << "\n? Detected TrustMark Encoder model" << std::endl;
        std::cout << "  Input 0 (image): expecting shape [1, 3, 256, 256]" << std::endl;
        std::cout << "  Input 1 (secret): expecting shape [1, 100]" << std::endl;

        // Create dummy input tensors for testing
        std::vector<float> image_data(1 * 3 * 256 * 256, 0.5f);  // Dummy image
        std::vector<float> secret_data(100, 0.0f);  // Dummy secret
        std::vector<int64_t> image_shape = {1, 3, 256, 256};
        std::vector<int64_t> secret_shape = {1, 100};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
            memory_info, image_data.data(), image_data.size(),
            image_shape.data(), image_shape.size()
        );

        Ort::Value secret_tensor = Ort::Value::CreateTensor<float>(
            memory_info, secret_data.data(), secret_data.size(),
            secret_shape.data(), secret_shape.size()
        );

        // Get input/output names
        auto input_name_0 = session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = session.GetInputNameAllocated(1, allocator);
        auto output_name_0 = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name_0.get(), input_name_1.get()};
        const char* output_names[] = {output_name_0.get()};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(image_tensor));
        input_tensors.push_back(std::move(secret_tensor));

        std::cout << "\nRunning inference with dummy data..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            2,
            output_names,
            1
        );

        std::cout << "? Inference completed successfully!" << std::endl;

        // Check output shape
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "  Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

    } else if (session.GetInputCount() == 1) {
        std::cout << "\n? Detected TrustMark Decoder model" << std::endl;
        std::cout << "  Input 0 (image): expecting shape [1, 3, 224/256, 224/256]" << std::endl;

        // Create dummy input tensor
        auto type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= (dim > 0 ? dim : 256);  // Use 256 for dynamic dims
        }

        std::vector<float> image_data(total_size, 0.5f);
        std::vector<int64_t> input_shape = {1, 3, 224, 224};  // Try decoder shape

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
            memory_info, image_data.data(), image_data.size(),
            input_shape.data(), input_shape.size()
        );

        auto input_name_0 = session.GetInputNameAllocated(0, allocator);
        auto output_name_0 = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name_0.get()};
        const char* output_names[] = {output_name_0.get()};

        std::cout << "\nRunning inference with dummy data..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &image_tensor,
            1,
            output_names,
            1
        );

        std::cout << "? Inference completed successfully!" << std::endl;

        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "  Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\n? TrustMark WASM example completed successfully!" << std::endl;
    return 0;
}
