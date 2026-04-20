#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    const char* model_path = "models/encoder_P.ort";
    const char* image_path = "../images/ufo_240.jpg";

    // Load actual UFO image
    cv::Mat img = cv::imread(image_path);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0/255.0);
    floatImg = floatImg * 2.0 - 1.0;

    // Convert to CHW
    std::vector<float> image_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            cv::Vec3f pixel = floatImg.at<cv::Vec3f>(h, w);
            image_data[0 * 256 * 256 + h * 256 + w] = pixel[0];  // B
            image_data[1 * 256 * 256 + h * 256 + w] = pixel[1];  // G
            image_data[2 * 256 * 256 + h * 256 + w] = pixel[2];  // R
        }
    }

    std::vector<float> secret_data(100, 0.0f);

    std::cout << "Native Input (first pixel): " << image_data[0] << ", "
              << image_data[256*256] << ", " << image_data[2*256*256] << std::endl;

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

    std::cout << "Native Output (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
