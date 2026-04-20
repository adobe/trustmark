#include "trustmark/trustmark.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "Testing if .ort model works with native C++" << std::endl;

    // The TrustMark class uses .onnx models
    // But let me just test by loading the encoder directly

    try {
        TrustMark tm(false, 100, "P");

        cv::Mat img = cv::imread("../images/ufo_240.jpg");
        std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

        std::vector<bool> secret(100, false);

        cv::Mat watermarked = tm.encode(img, secret);

        if (!watermarked.empty()) {
            std::cout << "✓ Encoding succeeded with .onnx models" << std::endl;
            cv::imwrite("test_native_watermarked.png", watermarked);

            // Check first pixel
            cv::Vec3b pixel = watermarked.at<cv::Vec3b>(0, 0);
            std::cout << "First pixel: (" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ")" << std::endl;
        } else {
            std::cout << "❌ Encoding failed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
