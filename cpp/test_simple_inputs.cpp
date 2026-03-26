#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> image_data(1 * 3 * 256 * 256);
    // Set specific values that match WASM
    image_data[0] = -0.6f;
    image_data[256*256] = -0.717647f;
    image_data[2*256*256] = -0.921569f;
    // Rest are 0.5f
    for (size_t i = 0; i < image_data.size(); i++) {
        if (i != 0 && i != 256*256 && i != 2*256*256) {
            image_data[i] = 0.5f;
        }
    }
    
    std::vector<float> secret_data(100, 0.0f);
    
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "Debug");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    
    Ort::Session session(env, "models/encoder_P.onnx", opts);
    
    std::vector<int64_t> image_shape = {1, 3, 256, 256};
    std::vector<int64_t> secret_shape = {1, 100};
    
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
        mem, image_data.data(), image_data.size(), image_shape.data(), image_shape.size());
    Ort::Value secret_tensor = Ort::Value::CreateTensor<float>(
        mem, secret_data.data(), secret_data.size(), secret_shape.data(), secret_shape.size());
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto in0 = session.GetInputNameAllocated(0, allocator);
    auto in1 = session.GetInputNameAllocated(1, allocator);
    auto out0 = session.GetOutputNameAllocated(0, allocator);
    
    const char* input_names[] = {in0.get(), in1.get()};
    const char* output_names[] = {out0.get()};
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(img_tensor));
    inputs.push_back(std::move(secret_tensor));
    
    std::cout << "Running inference..." << std::endl;
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, 1);
    
    const float* output = outputs[0].GetTensorData<float>();
    
    std::cout << "\nNative .onnx output (first 10): ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
