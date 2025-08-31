#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "trustmark/trustmark.h"
#include "trustmark/image_processor.h"




int main(int argc, char* argv[]) {
    try {
        std::cout << "TrustMark C++ Example" << std::endl;
        std::cout << "=====================" << std::endl;

        // Check command line arguments
        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <input_image_path> [secret_message]" << std::endl;
            std::cout << "Example: " << argv[0] << " input.jpg \"Hello, TrustMark!\"" << std::endl;
            return 1;
        }

        std::string inputImagePath = argv[1];
        // Use the fully-encoded 100-bit string (data+ECC+version) to validate pipeline
        std::string secretMessage = (argc > 2) ? argv[2] : "0110111100000110010111010000100000011110000000100100111000010110100011110111101110011010010011010001";

        std::cout << "Input image: " << inputImagePath << std::endl;
        std::cout << "Secret bitstring: " << secretMessage << std::endl;
        std::cout << "Bitstring length: " << secretMessage.length() << " bits" << std::endl;

        // Load input image
        cv::Mat coverImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);
        if (coverImage.empty()) {
            std::cerr << "Error: Could not load image: " << inputImagePath << std::endl;
            return 1;
        }

        std::cout << "Image loaded successfully. Size: "
                  << coverImage.cols << "x" << coverImage.rows << std::endl;

        // Initialize TrustMark with P variant (disable ECC since we pass full 100-bit schema)
        TrustMark::TrustMark trustmark(false, true, 100, "P", TrustMark::EncodingType::BCH_5, 1.0f);

        if (!trustmark.getLastError().empty()) {
            std::cerr << "Error initializing TrustMark: " << trustmark.getLastError() << std::endl;
            return 1;
        }

        std::cout << "TrustMark initialized successfully" << std::endl;
        std::cout << "Schema capacity: " << trustmark.getSchemaCapacity() << " bits" << std::endl;

        // Encode watermark
        std::cout << "\nEncoding watermark..." << std::endl;
        cv::Mat watermarkedImage = trustmark.encode(coverImage, secretMessage, TrustMark::Mode::BINARY, 0.95f, "bilinear");

        if (watermarkedImage.empty()) {
            std::cerr << "Error encoding watermark: " << trustmark.getLastError() << std::endl;
            return 1;
        }

        std::cout << "Watermark encoded successfully!" << std::endl;

        // Convert to BGR for OpenCV encoding APIs
        cv::Mat watermarkedBGR;
        cv::cvtColor(watermarkedImage, watermarkedBGR, cv::COLOR_RGB2BGR);

        // Save JPEG (high quality)
        std::string outputPath = "../output/watermarked_" + std::to_string(time(nullptr)) + ".jpg";
        std::vector<int> params; 
        params.push_back(cv::IMWRITE_JPEG_QUALITY); params.push_back(90);
        params.push_back(cv::IMWRITE_JPEG_OPTIMIZE); params.push_back(1);
        if (cv::imwrite(outputPath, watermarkedBGR, params)) {
            std::cout << "Watermarked image saved as: " << outputPath << std::endl;
        } else {
            std::cerr << "Warning: Could not save watermarked image" << std::endl;
        }
        // Also save PNG (lossless) for decoding test
        std::string outputPng = "../output/watermarked_" + std::to_string(time(nullptr)) + ".png";
        if (cv::imwrite(outputPng, watermarkedBGR)) {
            std::cout << "Watermarked PNG saved as: " << outputPng << std::endl;
        }

        // Run C++ decoder on both outputs to validate end-to-end
        std::cout << "\nDecoding via C++..." << std::endl;
        TrustMark::TrustMark tmDec(false, true, 100, "P", TrustMark::EncodingType::BCH_5, 1.0f);
        cv::Mat jpg = cv::imread(outputPath, cv::IMREAD_COLOR);
        cv::Mat png = cv::imread(outputPng, cv::IMREAD_COLOR);
        auto [bitsJpg, okJpg, vJpg] = tmDec.decode(jpg, TrustMark::Mode::BINARY);
        auto [bitsPng, okPng, vPng] = tmDec.decode(png, TrustMark::Mode::BINARY);
        std::cout << "Decoded JPG (ok=" << okJpg << "): " << bitsJpg << std::endl;
        std::cout << "Decoded PNG (ok=" << okPng << "): " << bitsPng << std::endl;

        std::cout << "\nExample completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
