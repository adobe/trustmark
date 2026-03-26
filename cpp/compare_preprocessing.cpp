#include <opencv2/opencv.hpp>
#include "wasm/image_utils.h"
#include <iostream>

int main() {
    const char* image_path = "../images/ufo_240.jpg";
    
    std::cout << "=== OPENCV PREPROCESSING ===" << std::endl;
    cv::Mat cv_img = cv::imread(image_path);
    cv::Mat cv_resized;
    cv::resize(cv_img, cv_resized, cv::Size(256, 256));
    
    std::cout << "OpenCV first pixel (BGR): (" 
              << (int)cv_resized.at<cv::Vec3b>(0, 0)[0] << "," 
              << (int)cv_resized.at<cv::Vec3b>(0, 0)[1] << "," 
              << (int)cv_resized.at<cv::Vec3b>(0, 0)[2] << ")" << std::endl;
    
    // Normalize like OpenCV
    cv::Mat cv_float;
    cv_resized.convertTo(cv_float, CV_32F, 1.0/255.0);
    cv_float = cv_float * 2.0 - 1.0;
    
    std::cout << "After normalize [-1,1]: (" 
              << cv_float.at<cv::Vec3f>(0, 0)[0] << "," 
              << cv_float.at<cv::Vec3f>(0, 0)[1] << "," 
              << cv_float.at<cv::Vec3f>(0, 0)[2] << ")" << std::endl;
    
    std::cout << "\n=== STB_IMAGE PREPROCESSING ===" << std::endl;
    auto stb_img = ImageUtils::loadImage(image_path);
    auto stb_resized = ImageUtils::resizeImage(stb_img, 256, 256);
    
    std::cout << "stb_image first pixel (RGB): (" 
              << (int)stb_resized.data[0] << "," 
              << (int)stb_resized.data[1] << "," 
              << (int)stb_resized.data[2] << ")" << std::endl;
    
    // Convert to BGR like we do in WASM
    auto stb_bgr = ImageUtils::rgbToBgr(stb_resized);
    std::cout << "After RGB->BGR: (" 
              << (int)stb_bgr.data[0] << "," 
              << (int)stb_bgr.data[1] << "," 
              << (int)stb_bgr.data[2] << ")" << std::endl;
    
    // Normalize
    auto stb_norm = ImageUtils::normalizeImage(stb_bgr);
    std::cout << "After normalize [-1,1]: (" 
              << stb_norm[0] << "," 
              << stb_norm[1] << "," 
              << stb_norm[2] << ")" << std::endl;
    
    std::cout << "\n=== COMPARISON ===" << std::endl;
    float diff_b = std::abs(cv_float.at<cv::Vec3f>(0, 0)[0] - stb_norm[0]);
    float diff_g = std::abs(cv_float.at<cv::Vec3f>(0, 0)[1] - stb_norm[1]);
    float diff_r = std::abs(cv_float.at<cv::Vec3f>(0, 0)[2] - stb_norm[2]);
    
    std::cout << "Difference (B,G,R): (" << diff_b << "," << diff_g << "," << diff_r << ")" << std::endl;
    
    if (diff_b < 0.01 && diff_g < 0.01 && diff_r < 0.01) {
        std::cout << "✅ Preprocessing is IDENTICAL!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Preprocessing is DIFFERENT!" << std::endl;
        return 1;
    }
}
