#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "../wasm/image_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.ort> <image.jpg>" << std::endl;
        return 1;
    }
    
    std::cout << "Testing simple Conv in WASM" << std::endl;
    
    // Load and preprocess
    auto img = ImageUtils::loadImage(argv[2]);
    auto resized = ImageUtils::resizeImage(img, 256, 256);
    auto bgr = ImageUtils::rgbToBgr(resized);
    auto normalized = ImageUtils::normalizeImage(bgr);
    
    // To CHW
    std::vector<float> input_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            int hwc = (h * 256 + w) * 3;
            input_data[0 * 256 * 256 + h * 256 + w] = normalized[hwc + 0];
            input_data[1 * 256 * 256 + h * 256 + w] = normalized[hwc + 1];
            input_data[2 * 256 * 256 + h * 256 + w] = normalized[hwc + 2];
        }
    }
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
    Ort::SessionOptions opts;
    Ort::Session session(env, argv[1], opts);
    
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, 256, 256};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), shape.data(), shape.size());
    
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char* ins[] = {in_name.get()};
    const char* outs[] = {out_name.get()};
    
    std::cout << "Input[0]:  " << input_data[0] << std::endl;
    std::cout << "Input[256*256]:  " << input_data[256*256] << std::endl;
    
    auto outputs = session.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);
    
    float* out_data = outputs[0].GetTensorMutableData<float>();
    std::cout << "Output[0]: " << out_data[0] << std::endl;
    std::cout << "Output[256*256]: " << out_data[256*256] << std::endl;
    
    // Check if output changed from input
    float diff = std::abs(out_data[0] - input_data[0]);
    std::cout << "Difference: " << diff << std::endl;
    
    return 0;
}
