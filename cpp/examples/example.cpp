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
        std::string secretMessage = (argc > 2) ? argv[2] : "1011011110011000111111000000011111011111011100000110110110111";
        
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
        
        // Initialize TrustMark with P variant (newest, always center crops)
        TrustMark::TrustMark trustmark(true, true, 100, "P", TrustMark::EncodingType::BCH_5, 1.0f);
        
        if (!trustmark.getLastError().empty()) {
            std::cerr << "Error initializing TrustMark: " << trustmark.getLastError() << std::endl;
            return 1;
        }
        
        std::cout << "TrustMark initialized successfully" << std::endl;
        std::cout << "Schema capacity: " << trustmark.getSchemaCapacity() << " bits" << std::endl;
        
        // Encode watermark
        std::cout << "\nEncoding watermark..." << std::endl;
        cv::Mat watermarkedImage = trustmark.encode(coverImage, secretMessage, TrustMark::Mode::BINARY, 1.0f, "bilinear");
        
        if (watermarkedImage.empty()) {
            std::cerr << "Error encoding watermark: " << trustmark.getLastError() << std::endl;
            return 1;
        }
        
        std::cout << "Watermark encoded successfully!" << std::endl;
        
        // Save the actual watermarked image from the encoder
        std::string outputPath = "../output/watermarked_" + std::to_string(time(nullptr)) + ".jpg";
        if (cv::imwrite(outputPath, watermarkedImage)) {
            std::cout << "Watermarked image saved as: " << outputPath << std::endl;
        } else {
            std::cerr << "Warning: Could not save watermarked image" << std::endl;
        }
        
        // Decode watermark
        std::cout << "\nDecoding watermark..." << std::endl;
        auto decodeResult = trustmark.decode(watermarkedImage, TrustMark::Mode::BINARY);
        
        std::string decodedMessage = std::get<0>(decodeResult);
        bool decodeSuccess = std::get<1>(decodeResult);
        int version = std::get<2>(decodeResult);
        
        if (decodeSuccess) {
            std::cout << "Watermark decoded successfully!" << std::endl;
            std::cout << "Decoded message: " << decodedMessage << std::endl;
            std::cout << "Version: " << version << std::endl;
            
            // Check if decoded message matches original
            if (decodedMessage == secretMessage) {
                std::cout << "✓ Bitstring matches perfectly!" << std::endl;
            } else {
                std::cout << "✗ Bitstring mismatch. Original: \"" << secretMessage 
                          << "\", Decoded: \"" << decodedMessage << "\"" << std::endl;
            }
        } else {
            std::cerr << "Error decoding watermark: " << trustmark.getLastError() << std::endl;
        }
        
        // Demonstrate watermark removal
        std::cout << "\nDemonstrating watermark removal..." << std::endl;
        cv::Mat cleanedImage = trustmark.removeWatermark(watermarkedImage, 1.0f, "bilinear");
        
        if (!cleanedImage.empty()) {
            std::string cleanedPath = "../output/cleaned_" + std::to_string(time(nullptr)) + ".jpg";
            if (cv::imwrite(cleanedPath, cleanedImage)) {
                std::cout << "Cleaned image saved as: " << cleanedPath << std::endl;
            }
            
            // Calculate PSNR between original and cleaned image
            double psnr = TrustMark::image_utils::getImagePSNR(coverImage, cleanedImage);
            if (psnr > 0) {
                std::cout << "PSNR between original and cleaned: " << psnr << " dB" << std::endl;
            }
        } else {
            std::cerr << "Warning: Could not remove watermark" << std::endl;
        }
        
        std::cout << "\nExample completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
