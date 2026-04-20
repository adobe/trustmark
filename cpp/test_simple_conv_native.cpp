#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    // Test simple conv model
    cv::Mat img = cv::imread("../images/ufo_240.jpg");
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0/255.0);
    floatImg = floatImg * 2.0 - 1.0;

    // To CHW
    std::vector<float> input_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            cv::Vec3f pixel = floatImg.at<cv::Vec3f>(h, w);
            input_data[0 * 256 * 256 + h * 256 + w] = pixel[0];
            input_data[1 * 256 * 256 + h * 256 + w] = pixel[1];
            input_data[2 * 256 * 256 + h * 256 + w] = pixel[2];
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Test");
    Ort::SessionOptions opts;

    std::cout << "Testing: test_simple_conv.ort" << std::endl;
    Ort::Session session(env, "test_simple_conv.ort", opts);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, 256, 256};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session.GetInputNameAllocated(0, alloc);
    auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char* ins[] = {in_name.get()};
    const char* outs[] = {out_name.get()};

    std::cout << "Running inference..." << std::endl;
    auto outputs = session.Run(Ort::RunOptions{nullptr}, ins, &tensor, 1, outs, 1);

    float* out_data = outputs[0].GetTensorMutableData<float>();
    std::cout << "✓ Completed" << std::endl;
    std::cout << "  Input[0]:  " << input_data[0] << std::endl;
    std::cout << "  Output[0]: " << out_data[0] << std::endl;

    // Check if output is reasonable (should be similar to input for a simple averaging conv)
    float diff = std::abs(out_data[0] - input_data[0]);
    if (diff < 0.5) {
        std::cout << "✅ Simple Conv works in native!" << std::endl;
    } else {
        std::cout << "❌ Unexpected output" << std::endl;
    }

    return 0;
}
