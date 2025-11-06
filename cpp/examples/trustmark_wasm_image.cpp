#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <onnxruntime_cxx_api.h>

// Include our minimal image utilities
#include "../wasm/image_utils.h"

// Simple TrustMark WASM example with real image support
// Uses stb libraries for image I/O (no OpenCV needed)

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
    std::cout << "TrustMark WASM Example with Image Support" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Check arguments
    if (argc < 3) {
        std::cout << "\nUsage: " << argv[0] << " <encoder.ort> <input_image.jpg>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  wasmtime --dir=.::.  --dir=models::/models \\" << std::endl;
        std::cout << "    trustmark.wasm /models/encoder_P.ort input.jpg" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    
    std::cout << "\nLoading model: " << model_path << std::endl;
    std::cout << "Input image: " << image_path << std::endl;
    
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
    
    // Load the input image
    std::cout << "\nLoading image..." << std::endl;
    ImageUtils::Image img = ImageUtils::loadImage(image_path);
    
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    
    std::cout << "? Image loaded: " << img.width << "x" << img.height 
              << " with " << img.channels << " channels" << std::endl;
    
    // Resize to 256x256 (encoder expects this)
    std::cout << "Resizing to 256x256..." << std::endl;
    ImageUtils::Image resized = ImageUtils::resizeImage(img, 256, 256);
    std::cout << "? Image resized" << std::endl;
    
    // Ensure RGB format
    if (resized.channels == 4) {
        std::cout << "Converting RGBA to RGB..." << std::endl;
        // Simple RGBA to RGB conversion
        ImageUtils::Image rgb(256, 256, 3);
        for (int i = 0; i < 256 * 256; i++) {
            rgb.data[i * 3 + 0] = resized.data[i * 4 + 0];
            rgb.data[i * 3 + 1] = resized.data[i * 4 + 1];
            rgb.data[i * 3 + 2] = resized.data[i * 4 + 2];
        }
        resized = rgb;
    }
    
    // Convert RGB to BGR (TrustMark models expect BGR like OpenCV)
    std::cout << "Converting RGB to BGR..." << std::endl;
    ImageUtils::Image bgr = ImageUtils::rgbToBgr(resized);
    std::cout << "? Converted to BGR format" << std::endl;
    
    // Normalize image to [-1, 1] (what the model expects)
    std::cout << "Normalizing image..." << std::endl;
    std::vector<float> normalized = ImageUtils::normalizeImage(bgr);
    std::cout << "? Image normalized to [-1, 1]" << std::endl;
    
    // Prepare ONNX input tensor [1, 3, 256, 256]
    std::vector<int64_t> image_shape = {1, 3, 256, 256};
    std::vector<float> image_data(1 * 3 * 256 * 256);
    
    // Convert from HWC to CHW format
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            int hwc_idx = (h * 256 + w) * 3;
            int chw_idx_r = 0 * 256 * 256 + h * 256 + w;
            int chw_idx_g = 1 * 256 * 256 + h * 256 + w;
            int chw_idx_b = 2 * 256 * 256 + h * 256 + w;
            
            image_data[chw_idx_r] = normalized[hwc_idx + 0];
            image_data[chw_idx_g] = normalized[hwc_idx + 1];
            image_data[chw_idx_b] = normalized[hwc_idx + 2];
        }
    }
    
    std::cout << "? Image converted to CHW format" << std::endl;
    
    // Check if this is an encoder (2 inputs) or decoder (1 input)
    Ort::AllocatorWithDefaultOptions allocator;
    
    if (session.GetInputCount() == 2) {
        std::cout << "\n? Detected TrustMark Encoder model" << std::endl;
        
        // Create dummy secret for testing
        std::vector<float> secret_data(100, 0.0f);
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
        
        auto input_name_0 = session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = session.GetInputNameAllocated(1, allocator);
        auto output_name_0 = session.GetOutputNameAllocated(0, allocator);
        
        const char* input_names[] = {input_name_0.get(), input_name_1.get()};
        const char* output_names[] = {output_name_0.get()};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(image_tensor));
        input_tensors.push_back(std::move(secret_tensor));
        
        std::cout << "\nRunning encoder inference..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            2,
            output_names,
            1
        );
        
        std::cout << "? Inference completed successfully!" << std::endl;
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "  Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Convert output back to image and save
        std::cout << "\nConverting output to image..." << std::endl;
        std::vector<float> output_hwc(256 * 256 * 3);
        
        // CHW to HWC
        for (int h = 0; h < 256; h++) {
            for (int w = 0; w < 256; w++) {
                int chw_idx_r = 0 * 256 * 256 + h * 256 + w;
                int chw_idx_g = 1 * 256 * 256 + h * 256 + w;
                int chw_idx_b = 2 * 256 * 256 + h * 256 + w;
                int hwc_idx = (h * 256 + w) * 3;
                
                output_hwc[hwc_idx + 0] = output_data[chw_idx_r];
                output_hwc[hwc_idx + 1] = output_data[chw_idx_g];
                output_hwc[hwc_idx + 2] = output_data[chw_idx_b];
            }
        }
        
        ImageUtils::Image output_img = ImageUtils::denormalizeImage(output_hwc, 256, 256, 3);
        
        // Convert back from BGR to RGB for saving
        ImageUtils::Image output_rgb = ImageUtils::bgrToRgb(output_img);
        
        if (ImageUtils::saveImage("output_watermarked.png", output_rgb)) {
            std::cout << "? Saved watermarked image: output_watermarked.png" << std::endl;
        }
        
        
    } else if (session.GetInputCount() == 1) {
        std::cout << "\n? Detected TrustMark Decoder model" << std::endl;
        
        // Resize to 224x224
        ImageUtils::Image resized_dec = ImageUtils::resizeImage(resized, 224, 224);
        std::vector<float> normalized_dec = ImageUtils::normalizeImage(resized_dec);
        
        // Prepare input [1, 3, 224, 224] CHW
        std::vector<int64_t> dec_shape = {1, 3, 224, 224};
        std::vector<float> dec_data(1 * 3 * 224 * 224);
        
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                int hwc = (h * 224 + w) * 3;
                int r = 0 * 224 * 224 + h * 224 + w;
                int g = 1 * 224 * 224 + h * 224 + w;
                int b = 2 * 224 * 224 + h * 224 + w;
                dec_data[r] = normalized_dec[hwc + 0];
                dec_data[g] = normalized_dec[hwc + 1];
                dec_data[b] = normalized_dec[hwc + 2];
            }
        }
        
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, dec_data.data(), dec_data.size(), dec_shape.data(), dec_shape.size()
        );
        
        auto in_name = session.GetInputNameAllocated(0, allocator);
        auto out_name = session.GetOutputNameAllocated(0, allocator);
        const char* ins[] = {in_name.get()};
        const char* outs[] = {out_name.get()};
        
        std::cout << "\nRunning decoder inference..." << std::endl;
        auto out_tensors = session.Run(Ort::RunOptions{nullptr}, ins, &in_tensor, 1, outs, 1);
        std::cout << "? Inference completed!" << std::endl;
        
        float* out_data = out_tensors[0].GetTensorMutableData<float>();
        auto out_shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t out_size = out_shape[1];
        
        std::cout << "  Output: [" << out_shape[0] << ", " << out_size << "]" << std::endl;
        std::cout << "\nDecoded bits: ";
        for (size_t i = 0; i < std::min(out_size, (size_t)100); i++) {
            std::cout << (out_data[i] > 0 ? "1" : "0");
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n? TrustMark WASM completed!" << std::endl;
    return 0;
}
