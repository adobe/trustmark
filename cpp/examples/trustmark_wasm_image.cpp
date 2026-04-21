#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>

// Include our minimal image utilities
#include "../wasm/image_utils.h"

// Simple TrustMark WASM example with real image support
// Uses stb libraries for image I/O (no OpenCV needed)

// Watermark strength for P variant (matches Python: WM_STRENGTH=1.0 * 1.25 for P)
static const float WM_STRENGTH = 1.25f;
// Residual clamp range (matches native C++ implementation)
static const float RESIDUAL_CLAMP = 0.2f;

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

// Convert HWC uint8 RGB image to CHW float32 in [-1, 1]
// stb_image loads as RGB, model expects RGB — no channel swap needed.
static std::vector<float> imageToTensor(const ImageUtils::Image& img) {
    int H = img.height, W = img.width, C = img.channels;
    std::vector<float> tensor(C * H * W);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                float v = img.data[(h * W + w) * C + c] / 255.0f; // [0,1]
                tensor[c * H * W + h * W + w] = v * 2.0f - 1.0f; // [-1,1]
            }
        }
    }
    return tensor;
}

// Convert CHW float32 [-1,1] back to HWC uint8 RGB
static ImageUtils::Image tensorToImage(const float* data, int H, int W, int C) {
    ImageUtils::Image img(W, H, C);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                float v = (data[c * H * W + h * W + w] + 1.0f) * 0.5f * 255.0f;
                img.data[(h * W + w) * C + c] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, v)));
            }
        }
    }
    return img;
}

int main(int argc, char* argv[]) {
    std::cout << "TrustMark WASM Example with Image Support" << std::endl;
    std::cout << "==========================================" << std::endl;

    if (argc < 3) {
        std::cout << "\nUsage: " << argv[0] << " <encoder_or_decoder.ort> <input_image.jpg>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    std::cout << "\nLoading model: " << model_path << std::endl;
    std::cout << "Input image: " << image_path << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrustMarkWASM");
    std::cout << "✓ ONNX Runtime initialized" << std::endl;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (getenv("USE_WEBGPU")) {
        std::unordered_map<std::string, std::string> webgpu_options;
        webgpu_options["preferredLayout"] = "NCHW";
        session_options.AppendExecutionProvider("WebGPU", webgpu_options);
        std::cout << "✓ WebGPU execution provider enabled (NCHW layout)" << std::endl;
    }

    std::cout << "✓ Session options configured" << std::endl;

    Ort::Session session(env, model_path, session_options);
    std::cout << "✓ Model loaded successfully!" << std::endl;

    print_model_info(session);

    // Load input image (stb_image loads as RGB)
    std::cout << "\nLoading image..." << std::endl;
    ImageUtils::Image img = ImageUtils::loadImage(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "✓ Image loaded: " << img.width << "x" << img.height
              << " with " << img.channels << " channels" << std::endl;

    // Convert RGBA → RGB if needed
    if (img.channels == 4) {
        ImageUtils::Image rgb(img.width, img.height, 3);
        for (int i = 0; i < img.width * img.height; i++) {
            rgb.data[i * 3 + 0] = img.data[i * 4 + 0];
            rgb.data[i * 3 + 1] = img.data[i * 4 + 1];
            rgb.data[i * 3 + 2] = img.data[i * 4 + 2];
        }
        img = rgb;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // ── ENCODER (2 inputs: image + secret) ──────────────────────────────────
    if (session.GetInputCount() == 2) {
        std::cout << "\n✓ Detected TrustMark Encoder model" << std::endl;

        // Resize to 256×256 for encoder
        ImageUtils::Image cover = ImageUtils::resizeImage(img, 256, 256);
        std::cout << "✓ Image resized to 256x256" << std::endl;

        // Convert to CHW float [-1,1] in RGB order (model trained on RGB)
        std::vector<float> image_data = imageToTensor(cover);
        std::cout << "✓ Image normalized to [-1, 1] (RGB, CHW)" << std::endl;

        // All-zeros secret (100 bits)
        std::vector<float> secret_data(100, 0.0f);
        std::vector<int64_t> image_shape = {1, 3, 256, 256};
        std::vector<int64_t> secret_shape = {1, 100};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
            memory_info, image_data.data(), image_data.size(),
            image_shape.data(), image_shape.size());
        Ort::Value secret_tensor = Ort::Value::CreateTensor<float>(
            memory_info, secret_data.data(), secret_data.size(),
            secret_shape.data(), secret_shape.size());

        auto in0 = session.GetInputNameAllocated(0, allocator);
        auto in1 = session.GetInputNameAllocated(1, allocator);
        auto out0 = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[]  = {in0.get(), in1.get()};
        const char* output_names[] = {out0.get()};

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(image_tensor));
        inputs.push_back(std::move(secret_tensor));

        std::cout << "\nRunning encoder inference..." << std::endl;
        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   input_names, inputs.data(), 2,
                                   output_names, 1);
        std::cout << "✓ Inference completed successfully!" << std::endl;

        // Encoder output: raw stego tensor in [-1,1], CHW, RGB
        const float* stego = outputs[0].GetTensorMutableData<float>();

        // Correct blend:
        //   residual = clamp(stego, -1, 1) - input_normalized
        //   residual *= WM_STRENGTH; clamp to [-RESIDUAL_CLAMP, +RESIDUAL_CLAMP]
        //   final = clamp(original_normalized + residual, -1, 1)
        //   final_uint8 = (final + 1) * 0.5 * 255
        const int N = 3 * 256 * 256;
        std::vector<float> final_chw(N);
        for (int i = 0; i < N; i++) {
            float s = std::max(-1.0f, std::min(1.0f, stego[i]));
            float residual = (s - image_data[i]) * WM_STRENGTH;
            residual = std::max(-RESIDUAL_CLAMP, std::min(RESIDUAL_CLAMP, residual));
            float blended = std::max(-1.0f, std::min(1.0f, image_data[i] + residual));
            final_chw[i] = blended;
        }

        // Back to HWC uint8
        ImageUtils::Image output_img = tensorToImage(final_chw.data(), 256, 256, 3);

        if (ImageUtils::saveImage("output_watermarked.png", output_img)) {
            std::cout << "✓ Saved watermarked image: output_watermarked.png" << std::endl;
        } else {
            std::cerr << "✗ Failed to save output_watermarked.png" << std::endl;
        }

    // ── DECODER (1 input: watermarked image) ────────────────────────────────
    } else if (session.GetInputCount() == 1) {
        std::cout << "\n✓ Detected TrustMark Decoder model" << std::endl;

        // Resize to 224×224 for decoder
        ImageUtils::Image dec_img = ImageUtils::resizeImage(img, 224, 224);
        std::vector<float> dec_data = imageToTensor(dec_img);
        std::cout << "✓ Image resized to 224x224, normalized (RGB, CHW)" << std::endl;

        std::vector<int64_t> dec_shape = {1, 3, 224, 224};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, dec_data.data(), dec_data.size(),
            dec_shape.data(), dec_shape.size());

        auto in_name  = session.GetInputNameAllocated(0, allocator);
        auto out_name = session.GetOutputNameAllocated(0, allocator);
        const char* ins[]  = {in_name.get()};
        const char* outs[] = {out_name.get()};

        std::cout << "\nRunning decoder inference..." << std::endl;
        auto out_tensors = session.Run(Ort::RunOptions{nullptr}, ins, &in_tensor, 1, outs, 1);
        std::cout << "✓ Inference completed!" << std::endl;

        float* out_data = out_tensors[0].GetTensorMutableData<float>();
        auto out_shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t out_size = static_cast<size_t>(out_shape[1]);

        std::cout << "  Output: [" << out_shape[0] << ", " << out_size << "]" << std::endl;
        std::cout << "\nDecoded bits: ";
        for (size_t i = 0; i < std::min(out_size, (size_t)100); i++) {
            std::cout << (out_data[i] > 0 ? "1" : "0");
        }
        std::cout << std::endl;
    }

    std::cout << "\n✓ TrustMark WASM completed!" << std::endl;
    return 0;
}
