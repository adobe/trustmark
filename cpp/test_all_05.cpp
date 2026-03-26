#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    const char* model_path = "models/encoder_P.ort";
    
    // ALL 0.5 to match WASM
    std::vector<float> image_data(1 * 3 * 256 * 256, 0.5f);
    std::vector<float> secret_data(100, 0.0f);
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path, opts);
    
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
    
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, 1);
    
    const float* output = outputs[0].GetTensorData<float>();
    
    std::cout << "Native Output (ALL 0.5 inputs, first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
