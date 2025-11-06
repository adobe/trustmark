#include <iostream>
#include <string>
#include <bitset>
#include <opencv2/opencv.hpp>
#include "trustmark/trustmark.h"

// Convert string to binary bit string
std::string stringToBits(const std::string& str, size_t targetBits = 48) {
    std::string bits;
    // Convert each character to 8 bits
    for (char c : str) {
        std::bitset<8> charBits(c);
        bits += charBits.to_string();
    }

    // Pad with zeros if needed (up to 6 chars = 48 bits for 52-bit data capacity)
    while (bits.length() < targetBits) {
        bits += "0";
    }

    // Truncate if too long
    if (bits.length() > targetBits) {
        bits = bits.substr(0, targetBits);
        std::cout << "Warning: Message truncated to " << targetBits << " bits (" << targetBits/8 << " chars)" << std::endl;
    }

    return bits;
}

// Convert binary bit string back to string
std::string bitsToString(const std::string& bits) {
    std::string result;
    // Process 8 bits at a time
    for (size_t i = 0; i + 8 <= bits.length(); i += 8) {
        std::string byte = bits.substr(i, 8);
        std::bitset<8> charBits(byte);
        char c = static_cast<char>(charBits.to_ulong());
        // Stop at null terminator or non-printable characters
        if (c == '\0') break;
        if (c >= 32 && c <= 126) {  // Printable ASCII
            result += c;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "TrustMark Encode/Decode Example" << std::endl;
        std::cout << "================================" << std::endl;

        // Check command line arguments
        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <input_image_path> [message]" << std::endl;
            std::cout << "Example: " << argv[0] << " ../images/ufo_240.jpg \"Hello!\"" << std::endl;
            return 1;
        }

        std::string inputImagePath = argv[1];
        std::string message = (argc > 2) ? argv[2] : "Adobe!";

        std::cout << "\nInput image: " << inputImagePath << std::endl;
        std::cout << "Message to encode: \"" << message << "\"" << std::endl;

        // Load input image
        cv::Mat coverImage = cv::imread(inputImagePath, cv::IMREAD_COLOR);
        if (coverImage.empty()) {
            std::cerr << "Error: Could not load image: " << inputImagePath << std::endl;
            return 1;
        }

        std::cout << "Image loaded: " << coverImage.cols << "x" << coverImage.rows << std::endl;

        // Determine execution provider
        TrustMark::ExecutionProvider provider = TrustMark::ExecutionProvider::CPU;
        const char* useGpuEnv = std::getenv("TRUSTMARK_USE_GPU");
        if (useGpuEnv && std::string(useGpuEnv) == "1") {
            #ifdef __APPLE__
                provider = TrustMark::ExecutionProvider::CoreML;
                std::cout << "Using CoreML (Apple Neural Engine + GPU)" << std::endl;
            #elif defined(_WIN32)
                provider = TrustMark::ExecutionProvider::DirectML;
                std::cout << "Using DirectML" << std::endl;
            #else
                provider = TrustMark::ExecutionProvider::CUDA;
                std::cout << "Using CUDA" << std::endl;
            #endif
        } else {
            std::cout << "Using CPU (set TRUSTMARK_USE_GPU=1 for GPU acceleration)" << std::endl;
        }

        // Convert message to bits (48 bits = 6 characters max)
        std::string messageBits = stringToBits(message, 48);
        std::cout << "\nMessage as bits (" << messageBits.length() << " bits): " << messageBits.substr(0, 48) << "..." << std::endl;

        // Initialize TrustMark - disable ECC since we'll pass the full 100-bit string
        // The models expect 100 bits total (data + ECC pre-encoded)
        std::cout << "\nInitializing TrustMark (P variant, 100-bit schema)..." << std::endl;
        TrustMark::TrustMark trustmark(
            false,  // enable_ecc: false = we provide full 100 bits (data already encoded with ECC)
            true,   // BCH_SUPER: use BCH super mode
            100,    // schema_size: full 100 bits expected by model
            "P",    // variant
            TrustMark::EncodingType::BCH_5,  // BCH_5 encoding type
            1.0f,   // normalization_constant
            provider,
            0       // deviceId
        );

        // Pad messageBits to 100 bits (simple approach: just pad with zeros)
        // In production, you'd use proper BCH encoding
        while (messageBits.length() < 100) {
            messageBits += "0";
        }

        if (!trustmark.getLastError().empty()) {
            std::cerr << "Error initializing TrustMark: " << trustmark.getLastError() << std::endl;
            return 1;
        }

        std::cout << "TrustMark initialized" << std::endl;
        std::cout << "Schema capacity: " << trustmark.getSchemaCapacity() << " bits" << std::endl;

        // ENCODE
        std::cout << "\n=== ENCODING ===" << std::endl;
        std::cout << "Encoding " << messageBits.length() << " bits..." << std::endl;
        cv::Mat watermarkedImage = trustmark.encode(
            coverImage,
            messageBits,  // Full 100-bit string
            TrustMark::Mode::BINARY,
            0.95f,        // quality
            "bilinear"    // resize mode
        );

        if (watermarkedImage.empty()) {
            std::cerr << "Error encoding: " << trustmark.getLastError() << std::endl;
            return 1;
        }

        std::cout << "? Watermark encoded successfully!" << std::endl;

        // Save watermarked image
        cv::Mat watermarkedBGR;
        cv::cvtColor(watermarkedImage, watermarkedBGR, cv::COLOR_RGB2BGR);

        std::string outputJpg = "../output/encoded_" + std::to_string(time(nullptr)) + ".jpg";
        std::string outputPng = "../output/encoded_" + std::to_string(time(nullptr)) + ".png";

        std::vector<int> jpgParams = {cv::IMWRITE_JPEG_QUALITY, 95, cv::IMWRITE_JPEG_OPTIMIZE, 1};
        cv::imwrite(outputJpg, watermarkedBGR, jpgParams);
        cv::imwrite(outputPng, watermarkedBGR);

        std::cout << "? Saved: " << outputJpg << std::endl;
        std::cout << "? Saved: " << outputPng << std::endl;

        // DECODE from both JPG and PNG
        std::cout << "\n=== DECODING ===" << std::endl;

        TrustMark::TrustMark decoder(
            false,  // enable_ecc: false = returns full 100 bits
            true,
            100,
            "P",
            TrustMark::EncodingType::BCH_5,
            1.0f,
            provider,
            0
        );

        // Decode from JPG
        std::cout << "\nDecoding from JPG..." << std::endl;
        cv::Mat jpgImage = cv::imread(outputJpg, cv::IMREAD_COLOR);
        auto [decodedBitsJpg, detectedJpg, versionJpg] = decoder.decode(jpgImage, TrustMark::Mode::BINARY);

        std::cout << "  Detected: " << (detectedJpg ? "YES" : "NO") << std::endl;
        std::cout << "  Decoded bits (" << decodedBitsJpg.length() << " bits): " << decodedBitsJpg.substr(0, 48) << std::endl;

        if (detectedJpg && decodedBitsJpg.length() >= 48) {
            std::string recoveredMessageJpg = bitsToString(decodedBitsJpg.substr(0, 48));
            std::cout << "  ? RECOVERED MESSAGE: \"" << recoveredMessageJpg << "\"" << std::endl;

            if (recoveredMessageJpg == message) {
                std::cout << "  ??? PERFECT MATCH! ???" << std::endl;
            } else {
                std::cout << "  ? Message differs from original" << std::endl;
            }
        }

        // Decode from PNG
        std::cout << "\nDecoding from PNG..." << std::endl;
        cv::Mat pngImage = cv::imread(outputPng, cv::IMREAD_COLOR);
        auto [decodedBitsPng, detectedPng, versionPng] = decoder.decode(pngImage, TrustMark::Mode::BINARY);

        std::cout << "  Detected: " << (detectedPng ? "YES" : "NO") << std::endl;
        std::cout << "  Decoded bits (" << decodedBitsPng.length() << " bits): " << decodedBitsPng.substr(0, 48) << std::endl;

        if (detectedPng && decodedBitsPng.length() >= 48) {
            std::string recoveredMessagePng = bitsToString(decodedBitsPng.substr(0, 48));
            std::cout << "  ? RECOVERED MESSAGE: \"" << recoveredMessagePng << "\"" << std::endl;

            if (recoveredMessagePng == message) {
                std::cout << "  ??? PERFECT MATCH! ???" << std::endl;
            } else {
                std::cout << "  ? Message differs from original" << std::endl;
            }
        }

        std::cout << "\n=== SUCCESS ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
