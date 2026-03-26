#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.ort>" << std::endl;
        return 1;
    }
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, argv[1], opts);
    
    std::cout << "Model: " << argv[1] << std::endl;
    
    // Test Sigmoid with known inputs
    std::vector<float> input(10);
    for (int i = 0; i < 10; i++) input[i] = -2.0f + i * 0.5f;  // -2, -1.5, -1, ..., 2
    
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 10};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape.data(), shape.size());
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {in_name.get()};
    const char* output_names[] = {out_name.get()};
    
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    const float* output = outputs[0].GetTensorData<float>();
    
    std::cout << "  Output: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
