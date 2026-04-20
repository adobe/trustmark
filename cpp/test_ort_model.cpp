#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.ort> <image.jpg>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    std::cout << "Testing .ort model with native ONNX Runtime" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;

    // Load image with OpenCV
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << img.cols << "x" << img.rows << " BGR" << std::endl;

    // Resize to 256x256
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256));

    // Convert to float and normalize to [-1, 1]
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0/255.0);
    floatImg = floatImg * 2.0 - 1.0;

    std::cout << "First pixel after normalize: ("
              << floatImg.at<cv::Vec3f>(0, 0)[0] << ","
              << floatImg.at<cv::Vec3f>(0, 0)[1] << ","
              << floatImg.at<cv::Vec3f>(0, 0)[2] << ")" << std::endl;

    // Convert to CHW format
    std::vector<float> input_data(1 * 3 * 256 * 256);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            cv::Vec3f pixel = floatImg.at<cv::Vec3f>(h, w);
            int chw_idx_b = 0 * 256 * 256 + h * 256 + w;
            int chw_idx_g = 1 * 256 * 256 + h * 256 + w;
            int chw_idx_r = 2 * 256 * 256 + h * 256 + w;
            input_data[chw_idx_b] = pixel[0];  // B
            input_data[chw_idx_g] = pixel[1];  // G
            input_data[chw_idx_r] = pixel[2];  // R
        }
    }

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestORT");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    std::cout << "Loading model..." << std::endl;
    Ort::Session session(env, model_path, session_options);
    std::cout << "✓ Model loaded" << std::endl;

    // Check inputs
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    std::cout << "Model has " << num_inputs << " inputs" << std::endl;

    if (num_inputs == 2) {
        // Encoder - create dummy secret
        std::vector<float> secret_data(100, 0.0f);

        // Create tensors
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

        auto input_name_0 = session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = session.GetInputNameAllocated(1, allocator);
        auto output_name_0 = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name_0.get(), input_name_1.get()};
        const char* output_names[] = {output_name_0.get()};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(image_tensor));
        input_tensors.push_back(std::move(secret_tensor));

        std::cout << "Running inference..." << std::endl;
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

        // Check first pixel CHW values
        std::cout << "\nOutput CHW values (first pixel):" << std::endl;
        std::cout << "  Channel 0: " << output_data[0] << std::endl;
        std::cout << "  Channel 1: " << output_data[256*256] << std::endl;
        std::cout << "  Channel 2: " << output_data[2*256*256] << std::endl;

        // Check if output is all near zero (bad) or has real values (good)
        float sum = 0;
        float min_val = output_data[0], max_val = output_data[0];
        for (int i = 0; i < 256*256*3; i++) {
            sum += std::abs(output_data[i]);
            if (output_data[i] < min_val) min_val = output_data[i];
            if (output_data[i] > max_val) max_val = output_data[i];
        }
        float avg_abs = sum / (256*256*3);

        std::cout << "\nOutput statistics:" << std::endl;
        std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "  Average absolute value: " << avg_abs << std::endl;

        if (avg_abs < 0.1) {
            std::cout << "\n❌ Output values near zero - model not working properly!" << std::endl;
        } else {
            std::cout << "\n✅ Output has real values - model working!" << std::endl;
        }

        // Save output
        cv::Mat output_hwc(256, 256, CV_32FC3);
        for (int h = 0; h < 256; h++) {
            for (int w = 0; w < 256; w++) {
                int chw_idx_b = 0 * 256 * 256 + h * 256 + w;
                int chw_idx_g = 1 * 256 * 256 + h * 256 + w;
                int chw_idx_r = 2 * 256 * 256 + h * 256 + w;

                cv::Vec3f& pixel = output_hwc.at<cv::Vec3f>(h, w);
                pixel[0] = output_data[chw_idx_b];
                pixel[1] = output_data[chw_idx_g];
                pixel[2] = output_data[chw_idx_r];
            }
        }

        // Denormalize: [-1, 1] -> [0, 255]
        output_hwc = (output_hwc + 1.0) * 0.5 * 255.0;
        cv::Mat output_uint8;
        output_hwc.convertTo(output_uint8, CV_8U);

        cv::imwrite("test_ort_native_output.png", output_uint8);
        std::cout << "Saved: test_ort_native_output.png" << std::endl;
    }

    return 0;
}
