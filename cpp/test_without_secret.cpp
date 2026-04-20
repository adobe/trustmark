// Test TrustMark encoder but pass ZEROS for secret
// This should bypass the Gemm layer effectively
// If output is still wrong, Gemm is not the problem

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "../wasm/image_utils.h"

int main(int argc, char* argv[]) {
    std::cout << "Testing encoder with ZERO secret (bypass Gemm effectively)" << std::endl;

    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    auto resized = ImageUtils::resizeImage(img, 256, 256);
    auto bgr = ImageUtils::rgbToBgr(resized);
    auto normalized = ImageUtils::normalizeImage(bgr);

    // To CHW
    std::vector<float> image_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            int hwc = (h * 256 + w) * 3;
            image_data[0 * 256 * 256 + h * 256 + w] = normalized[hwc + 0];
            image_data[1 * 256 * 256 + h * 256 + w] = normalized[hwc + 1];
            image_data[2 * 256 * 256 + h * 256 + w] = normalized[hwc + 2];
        }
    }

    // ALL ZEROS secret
    std::vector<float> secret_data(100, 0.0f);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/encoder_P.ort", opts);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> image_shape = {1, 3, 256, 256};
    std::vector<int64_t> secret_shape = {1, 100};

    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(mem, image_data.data(), image_data.size(), image_shape.data(), image_shape.size());
    Ort::Value secret_tensor = Ort::Value::CreateTensor<float>(mem, secret_data.data(), secret_data.size(), secret_shape.data(), secret_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in0 = session.GetInputNameAllocated(0, alloc);
    auto in1 = session.GetInputNameAllocated(1, alloc);
    auto out0 = session.GetOutputNameAllocated(0, alloc);

    const char* ins[] = {in0.get(), in1.get()};
    const char* outs[] = {out0.get()};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(image_tensor));
    inputs.push_back(std::move(secret_tensor));

    std::cout << "Input image[0]: " << image_data[0] << std::endl;
    std::cout << "Input secret[0]: " << secret_data[0] << std::endl;

    auto outputs = session.Run(Ort::RunOptions{nullptr}, ins, inputs.data(), 2, outs, 1);

    float* out_data = outputs[0].GetTensorMutableData<float>();
    std::cout << "Output[0]: " << out_data[0] << std::endl;
    std::cout << "Output[65536]: " << out_data[65536] << std::endl;
    std::cout << "Output[131072]: " << out_data[131072] << std::endl;

    // Check range
    float min_v = out_data[0], max_v = out_data[0];
    for (int i = 0; i < 256*256*3; i++) {
        if (out_data[i] < min_v) min_v = out_data[i];
        if (out_data[i] > max_v) max_v = out_data[i];
    }
    std::cout << "Output range: [" << min_v << ", " << max_v << "]" << std::endl;

    return 0;
}
