#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing .ort model with native ONNX Runtime (direct)" << std::endl;
    
    // Use the static library from the build
    const char* model_path = "models/encoder_P.ort";
    const char* image_path = "../images/ufo_240.jpg";
    
    std::cout << "Model: " << model_path << std::endl;
    
    // Load and prepare image
    cv::Mat img = cv::imread(image_path);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));
    
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0/255.0);
    floatImg = floatImg * 2.0 - 1.0;
    
    std::cout << "Image prepared: 256x256 BGR, normalized to [-1,1]" << std::endl;
    std::cout << "First pixel: (" 
              << floatImg.at<cv::Vec3f>(0, 0)[0] << "," 
              << floatImg.at<cv::Vec3f>(0, 0)[1] << "," 
              << floatImg.at<cv::Vec3f>(0, 0)[2] << ")" << std::endl;
    
    // Convert to CHW (BGR order from OpenCV)
    std::vector<float> input_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            cv::Vec3f pixel = floatImg.at<cv::Vec3f>(h, w);
            input_data[0 * 256 * 256 + h * 256 + w] = pixel[0];  // B
            input_data[1 * 256 * 256 + h * 256 + w] = pixel[1];  // G
            input_data[2 * 256 * 256 + h * 256 + w] = pixel[2];  // R
        }
    }
    
    std::cout << "Input CHW (first pixel): " 
              << input_data[0] << ", " 
              << input_data[256*256] << ", " 
              << input_data[2*256*256] << std::endl;
    
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestORT");
    Ort::SessionOptions session_options;
    
    std::cout << "Loading .ort model..." << std::endl;
    Ort::Session session(env, model_path, session_options);
    std::cout << "✓ Model loaded successfully" << std::endl;
    
    // Create tensors
    std::vector<float> secret_data(100, 0.0f);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> image_shape = {1, 3, 256, 256};
    std::vector<int64_t> secret_shape = {1, 100};
    
    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        image_shape.data(), image_shape.size()
    );
    
    Ort::Value secret_tensor = Ort::Value::CreateTensor<float>(
        memory_info, secret_data.data(), secret_data.size(),
        secret_shape.data(), secret_shape.size()
    );
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_0 = session.GetInputNameAllocated(0, allocator);
    auto input_name_1 = session.GetInputNameAllocated(1, allocator);
    auto output_name_0 = session.GetOutputNameAllocated(0, allocator);
    
    const char* input_names[] = {input_name_0.get(), input_name_1.get()};
    const char* output_names[] = {output_name_0.get()};
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(image_tensor));
    input_tensors.push_back(std::move(secret_tensor));
    
    std::cout << "Running inference on .ort model..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        2,
        output_names,
        1
    );
    
    std::cout << "✓ Inference completed" << std::endl;
    
    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
    std::cout << "\nOutput CHW (first pixel):" << std::endl;
    std::cout << "  Channel 0: " << output_data[0] << std::endl;
    std::cout << "  Channel 1: " << output_data[256*256] << std::endl;
    std::cout << "  Channel 2: " << output_data[2*256*256] << std::endl;
    
    // Statistics
    float sum = 0, min_val = output_data[0], max_val = output_data[0];
    for (int i = 0; i < 256*256*3; i++) {
        sum += std::abs(output_data[i]);
        if (output_data[i] < min_val) min_val = output_data[i];
        if (output_data[i] > max_val) max_val = output_data[i];
    }
    
    std::cout << "\nOutput range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "Average |value|: " << (sum / (256*256*3)) << std::endl;
    
    if (sum / (256*256*3) < 0.1) {
        std::cout << "\n❌ Output near zero - .ort model NOT working!" << std::endl;
        return 1;
    } else {
        std::cout << "\n✅ Output has real values - .ort model works!" << std::endl;
    }
    
    // Save output
    cv::Mat output_hwc(256, 256, CV_32FC3);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            cv::Vec3f& pixel = output_hwc.at<cv::Vec3f>(h, w);
            pixel[0] = output_data[0 * 256 * 256 + h * 256 + w];
            pixel[1] = output_data[1 * 256 * 256 + h * 256 + w];
            pixel[2] = output_data[2 * 256 * 256 + h * 256 + w];
        }
    }
    
    output_hwc = (output_hwc + 1.0) * 0.5 * 255.0;
    cv::Mat output_uint8;
    output_hwc.convertTo(output_uint8, CV_8U);
    cv::imwrite("test_ort_native_output.png", output_uint8);
    std::cout << "Saved: test_ort_native_output.png" << std::endl;
    
    return 0;
}
