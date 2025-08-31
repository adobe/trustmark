#include "trustmark.h"
#include "onnx_session.h"
#include "image_processor.h"
#include "bch_ecc.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace TrustMark {



// Constructor
TrustMark::TrustMark(bool useECC, bool verbose, int secretLen,
                     const std::string& modelType, EncodingType encodingType,
                     float concentrateWmRegion)
    : useECC_(useECC)
    , verbose_(verbose)
    , secretLen_(secretLen)
    , modelType_(modelType)
    , encodingType_(encodingType)
    , concentrateWmRegion_(concentrateWmRegion)
    , aspectRatioLim_(2.0f)
    , modelResolutionEnc_(256)
    , modelResolutionDec_(256)

    , imageProcessor_(std::make_unique<ImageProcessor>())
{
    // Validate model type
    if (modelType != "C" && modelType != "Q" && modelType != "B" && modelType != "P") {
        setLastError("Invalid model type. Must be one of: C, Q, B, P");
        return;
    }

    // Set model-specific parameters
    if (modelType == "P") {
        modelResolutionEnc_ = 256;
        modelResolutionDec_ = 256;
        aspectRatioLim_ = 0.0f; // Force center square crop
        strengthMultiplier_ = 1.25f; // P variant specific strength multiplier
    } else {
        modelResolutionEnc_ = 256;
        modelResolutionDec_ = 256;
        aspectRatioLim_ = 2.0f;
        strengthMultiplier_ = 1.0f; // Default strength multiplier
    }

    if (verbose_) {
        std::cout << "Initializing TrustMark watermarking "
                  << (useECC ? "with" : "without") << " ECC using ["
                  << modelType << "]" << std::endl;
    }

    // Initialize models
    if (!initializeModels()) {
        setLastError("Failed to initialize models: " + getLastError());
        return;
    }
}

// Destructor
TrustMark::~TrustMark() = default;

// Move constructor
TrustMark::TrustMark(TrustMark&& other) noexcept
    : useECC_(other.useECC_)
    , verbose_(other.verbose_)
    , secretLen_(other.secretLen_)
    , modelType_(std::move(other.modelType_))
    , encodingType_(other.encodingType_)
    , concentrateWmRegion_(other.concentrateWmRegion_)
    , aspectRatioLim_(other.aspectRatioLim_)
    , modelResolutionEnc_(other.modelResolutionEnc_)
    , modelResolutionDec_(other.modelResolutionDec_)

    , encoderSession_(std::move(other.encoderSession_))
    , decoderSession_(std::move(other.decoderSession_))
    
    , imageProcessor_(std::move(other.imageProcessor_))
    , lastError_(std::move(other.lastError_))
{
}

// Move assignment
TrustMark& TrustMark::operator=(TrustMark&& other) noexcept {
    if (this != &other) {
        useECC_ = other.useECC_;
        verbose_ = other.verbose_;
        secretLen_ = other.secretLen_;
        modelType_ = std::move(other.modelType_);
        encodingType_ = other.encodingType_;
        concentrateWmRegion_ = other.concentrateWmRegion_;
        aspectRatioLim_ = other.aspectRatioLim_;
        modelResolutionEnc_ = other.modelResolutionEnc_;
        modelResolutionDec_ = other.modelResolutionDec_;

        encoderSession_ = std::move(other.encoderSession_);
        decoderSession_ = std::move(other.decoderSession_);

        imageProcessor_ = std::move(other.imageProcessor_);
        lastError_ = std::move(other.lastError_);
    }
    return *this;
}

// Initialize models
bool TrustMark::initializeModels() {
    // Get model paths (this would need to be adapted for your model structure)
    std::string basePath = "../models/"; // Adjust path as needed
    std::string encoderPath = basePath + "encoder_" + modelType_ + ".onnx";
    std::string decoderPath = basePath + "decoder_" + modelType_ + ".onnx";
    // Check if model files exist
    std::ifstream encoderFile(encoderPath);
    std::ifstream decoderFile(decoderPath);

    bool encoderExists = encoderFile.good();
    bool decoderExists = decoderFile.good();

    encoderFile.close();
    decoderFile.close();

    if (!encoderExists || !decoderExists) {
        if (verbose_) {
            std::cout << "Required ONNX model files not found. Running in demo mode." << std::endl;
            std::cout << "Expected models:" << std::endl;
            std::cout << "  - " << encoderPath << " " << (encoderExists ? "✓" : "✗") << std::endl;
            std::cout << "  - " << decoderPath << " " << (decoderExists ? "✓" : "✗") << std::endl;

        }
        // Return true to allow demo mode to continue
        return true;
    }



    // Initialize encoder session
    encoderSession_ = std::make_unique<ONNXRuntimeSession>(encoderPath, "encoder");
    if (!encoderSession_->isInitialized()) {
        setLastError("Failed to initialize encoder: " + encoderSession_->getLastError());
        return false;
    }

    // Initialize decoder session
    decoderSession_ = std::make_unique<ONNXRuntimeSession>(decoderPath, "decoder");
    if (!decoderSession_->isInitialized()) {
        setLastError("Failed to initialize decoder: " + decoderSession_->getLastError());
        return false;
    }



    if (verbose_) {
        std::cout << "All models initialized successfully" << std::endl;
    }

    return true;
}

// Encode watermark
cv::Mat TrustMark::encode(const cv::Mat& coverImage, const std::string& secret,
                         Mode mode, float wmStrength, const std::string& wmMerge) {
    try {
        // Check if models are loaded first
        if (!encoderSession_ || !encoderSession_->isInitialized()) {
            setLastError("Encoder model not available or not initialized");
            return cv::Mat();
        }

        // Preprocess cover image
        cv::Mat croppedImage = getImageForProcessing(coverImage);
        cv::Mat normalizedImage = imageProcessor_->preprocessForEncoder(croppedImage, modelResolutionEnc_);

        // Note: preprocessForEncoder already converts to [-1, 1] range

        // Prepare secret data
        std::vector<float> secretData;
        if (!useECC_) {
            if (mode == Mode::BINARY) {
                // Convert binary string to float array
                for (char c : secret) {
                    secretData.push_back(static_cast<float>(c - '0'));
                }
            } else {
                // Convert text to ASCII bits
                std::vector<bool> bits = utils::encodeText(secret, secretLen_);
                for (bool bit : bits) {
                    secretData.push_back(static_cast<float>(bit));
                }
            }
        } else {
            // Use BCH encoding (simplified - would need full BCH implementation)
            std::vector<bool> bits = utils::encodeText(secret, secretLen_);
            for (bool bit : bits) {
                secretData.push_back(static_cast<float>(bit));
            }
        }

        // Pad or truncate secret to match expected length
        if (secretData.size() < secretLen_) {
            secretData.resize(secretLen_, 0.0f);
        } else if (secretData.size() > secretLen_) {
            secretData.resize(secretLen_);
        }

        // Debug: Print the secret being encoded
        if (verbose_) {
            std::cout << "Encoding secret: " << secret << std::endl;
            std::cout << "Secret length: " << secret.length() << " characters" << std::endl;
            std::cout << "Mode: " << (mode == Mode::BINARY ? "BINARY" : "TEXT") << std::endl;
            std::cout << "Secret data (first 20 values): ";
            for (size_t i = 0; i < std::min(secretData.size(), size_t(20)); ++i) {
                std::cout << secretData[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Total secret data size: " << secretData.size() << std::endl;

            // Also show image info
            std::cout << "Input image - size: " << coverImage.cols << "x" << coverImage.rows
                      << ", channels: " << coverImage.channels() << std::endl;
            std::cout << "Cropped image - size: " << croppedImage.cols << "x" << croppedImage.rows
                      << ", channels: " << croppedImage.channels() << std::endl;
            std::cout << "Normalized image - size: " << normalizedImage.cols << "x" << normalizedImage.rows
                      << ", channels: " << normalizedImage.channels() << std::endl;

            // Show normalized image data range
            double minVal, maxVal;
            cv::minMaxLoc(normalizedImage, &minVal, &maxVal);
            std::cout << "Normalized image data range: [" << minVal << ", " << maxVal << "]" << std::endl;

            // Show first few pixel values from normalized image
            std::cout << "Normalized image first 5 pixels: ";
            for (int i = 0; i < std::min(5, normalizedImage.cols * normalizedImage.rows); ++i) {
                int row = i / normalizedImage.cols;
                int col = i % normalizedImage.cols;
                cv::Vec3f pixel = normalizedImage.at<cv::Vec3f>(row, col);
                std::cout << "(" << pixel[0] << "," << pixel[1] << "," << pixel[2] << ") ";
            }
            std::cout << std::endl;
        }

        // Create input tensors
        std::vector<Ort::Value> inputs;

        // Image input: (1, 3, H, W)
        std::vector<int64_t> imageShape = {1, 3, modelResolutionEnc_, modelResolutionEnc_};
        Ort::Value imageTensor = onnx_utils::createInputTensor(normalizedImage, "onnx::Concat_0", imageShape);
        inputs.push_back(std::move(imageTensor));

        // Secret input: (1, secretLen)
        std::vector<int64_t> secretShape = {1, secretLen_};
        Ort::Value secretTensor = onnx_utils::createInputTensor(secretData, "onnx::Gemm_1", secretShape);
        inputs.push_back(std::move(secretTensor));

        // Debug: Print tensor info
        if (verbose_) {
            std::cout << "Created tensors:" << std::endl;
            std::cout << "  Image tensor - shape: ";
            for (auto dim : imageShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            std::cout << "  Secret tensor - shape: ";
            for (auto dim : secretShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            // Try to extract some tensor data for verification
            try {
                auto imageData = inputs[0].GetTensorTypeAndShapeInfo();
                auto secretData = inputs[1].GetTensorTypeAndShapeInfo();
                std::cout << "  Image tensor type: " << imageData.GetElementType() << std::endl;
                std::cout << "  Secret tensor type: " << secretData.GetElementType() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  Could not inspect tensor types: " << e.what() << std::endl;
            }
        }

        // Run encoder
        std::vector<Ort::Value> outputs = encoderSession_->run(inputs);
        if (outputs.empty()) {
            setLastError("Encoder failed to produce output: " + encoderSession_->getLastError());
            return cv::Mat(); // Return empty mat on error
        }

        // Check if we got valid output and handle NaN
        bool hasNaN = false;
        std::vector<int64_t> outputShape;
        try {
            outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

            // Check if output contains NaN
            float* outputData = const_cast<Ort::Value&>(outputs[0]).GetTensorMutableData<float>();
            size_t maxCheck = std::min(size_t(100), static_cast<size_t>(outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3]));
            for (size_t i = 0; i < maxCheck; ++i) {
                if (std::isnan(outputData[i])) {
                    hasNaN = true;
                    break;
                }
            }

            if (verbose_) {
                std::cout << "Encoder output shape: ";
                for (auto dim : outputShape) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
                std::cout << "Output contains NaN: " << (hasNaN ? "YES" : "NO") << std::endl;
            }
        } catch (const std::exception& e) {
            if (verbose_) {
                std::cout << "Could not inspect output: " << e.what() << std::endl;
            }
            hasNaN = true; // Assume failure if we can't inspect
        }

        // If we got NaN output, fail gracefully
        if (hasNaN) {
            if (verbose_) {
                std::cout << "NaN detected! Encoder failed to produce valid output." << std::endl;
            }
            setLastError("Encoder produced NaN output - model inference failed");
            return cv::Mat(); // Return empty mat on error
        }

        // Extract stego and residual
        cv::Mat stego = onnx_utils::extractOutputTensor(outputs[0],
                                                       {1, 3, modelResolutionEnc_, modelResolutionEnc_});
        cv::Mat residual = stego - normalizedImage;

        // Debug: Print encoder output info
        if (verbose_) {
            std::cout << "Encoder output - stego shape: " << stego.cols << "x" << stego.rows
                      << ", channels: " << stego.channels() << std::endl;
            std::cout << "Stego data (first 10 values): ";
            for (int i = 0; i < std::min(10, stego.cols * stego.rows); ++i) {
                int row = i / stego.cols;
                int col = i % stego.cols;
                cv::Vec3f pixel = stego.at<cv::Vec3f>(row, col);
                std::cout << "(" << pixel[0] << "," << pixel[1] << "," << pixel[2] << ") ";
            }
            std::cout << std::endl;

            std::cout << "Residual mean: " << cv::mean(residual)[0] << std::endl;
            cv::Scalar residualStdDev;
            cv::meanStdDev(residual, cv::noArray(), residualStdDev);
            std::cout << "Residual std dev: " << residualStdDev[0] << std::endl;

            // Save raw stego (256x256, RGB) as debug to verify bits via Rust decoder
            try {
                cv::Mat stego01 = (stego + 1.0f) * 0.5f; // [-1,1] -> [0,1]
                cv::Mat stegoU8; stego01.convertTo(stegoU8, CV_8UC3, 255.0);
                std::string debugPath = "../output/debug_stego_" + std::to_string(time(nullptr)) + ".png";
                cv::imwrite(debugPath, stegoU8);
                std::cout << "Saved debug stego: " << debugPath << std::endl;
            } catch (...) {}
        }

        // Apply watermark strength multiplier (variant-specific) and clamp to [-0.2, 0.2]
        residual = residual * (wmStrength * strengthMultiplier_);
        cv::max(residual, cv::Scalar(-0.2f, -0.2f, -0.2f), residual);
        cv::min(residual, cv::Scalar(0.2f, 0.2f, 0.2f), residual);

        // Mitigate boundary artifact: replace a small border with mean
        const int border = 2;
        cv::Scalar chMean = cv::mean(residual);
        // top and bottom rows
        if (residual.rows >= border) {
            residual.rowRange(0, border).setTo(chMean);
            residual.rowRange(residual.rows - border, residual.rows).setTo(chMean);
        }
        // left and right cols
        if (residual.cols >= border) {
            residual.colRange(0, border).setTo(chMean);
            residual.colRange(residual.cols - border, residual.cols).setTo(chMean);
        }

        // Build mean-padded residual canvas to match original aspect
        int origW = coverImage.cols;
        int origH = coverImage.rows;
        chMean = cv::mean(residual);
        cv::Mat meanPadded;
        if (origW > origH) {
            int other = static_cast<int>(std::round((static_cast<double>(origW) / origH) * 256.0));
            meanPadded = cv::Mat(256, other, CV_32FC3, chMean);
            int leftover = (other - 256) / 2;
            cv::Rect roi(leftover, 0, 256, 256);
            residual.copyTo(meanPadded(roi));
        } else {
            int other = static_cast<int>(std::round((static_cast<double>(origH) / origW) * 256.0));
            meanPadded = cv::Mat(other, 256, CV_32FC3, chMean);
            int leftover = (other - 256) / 2;
            cv::Rect roi(0, leftover, 256, 256);
            residual.copyTo(meanPadded(roi));
        }

        // Resize mean-padded residual to original size
        cv::Mat residualResized;
        cv::resize(meanPadded, residualResized, coverImage.size(), 0, 0, cv::INTER_LINEAR);

        // Convert original BGR image to RGB float [0,1]
        cv::Mat origRGB;
        if (coverImage.channels() == 3) {
            cv::cvtColor(coverImage, origRGB, cv::COLOR_BGR2RGB);
        } else if (coverImage.channels() == 4) {
            cv::cvtColor(coverImage, origRGB, cv::COLOR_BGRA2RGB);
        } else if (coverImage.channels() == 1) {
            cv::cvtColor(coverImage, origRGB, cv::COLOR_GRAY2RGB);
        } else {
            origRGB = coverImage.clone();
        }
        cv::Mat origFloat01;
        origRGB.convertTo(origFloat01, CV_32FC3, 1.0 / 255.0);

        // Original to [-1,1], add residual, clamp high-end at 1.0
        cv::Mat origNeg1 = origFloat01 * 2.0f - 1.0f;
        cv::Mat sumNeg1;
        cv::add(origNeg1, residualResized, sumNeg1, cv::noArray(), CV_32FC3);
        cv::min(sumNeg1, cv::Scalar(1.0f, 1.0f, 1.0f), sumNeg1);

        // Back to [0,1] then to 8-bit RGB
        cv::Mat outFloat01 = (sumNeg1 + 1.0f) * 0.5f;
        cv::Mat outU8;
        outFloat01.convertTo(outU8, CV_8UC3, 255.0);
        return outU8;

    } catch (const std::exception& e) {
        setLastError("Encode failed: " + std::string(e.what()));
        return cv::Mat(); // Return empty mat on error
    }
}

// Decode watermark
std::tuple<std::string, bool, int> TrustMark::decode(const cv::Mat& stegoImage, Mode mode) {
    try {
        // Check if models are loaded first
        if (!decoderSession_ || !decoderSession_->isInitialized()) {
            setLastError("Decoder model not available or not initialized");
            return {"", false, -1};
        }

        // Preprocess stego image
        cv::Mat croppedImage = getImageForProcessing(stegoImage);
        cv::Mat normalizedImage = imageProcessor_->preprocessForDecoder(croppedImage, modelResolutionDec_);

        // Note: preprocessForDecoder already converts to [-1, 1] range

        // Create input tensor
        std::vector<Ort::Value> inputs;
        std::vector<int64_t> imageShape = {1, 3, modelResolutionDec_, modelResolutionDec_};
        Ort::Value imageTensor = onnx_utils::createInputTensor(normalizedImage, "image", imageShape);
        inputs.push_back(std::move(imageTensor));

        // Run decoder
        if (verbose_) {
            std::cout << "About to run decoder with " << inputs.size() << " inputs" << std::endl;
            std::cout << "Input tensor shape: " << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[0] << " "
                      << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[1] << " "
                      << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[2] << " "
                      << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[3] << std::endl;
        }

        std::vector<Ort::Value> outputs;
        try {
            if (verbose_) {
                std::cout << "About to run decoder with " << inputs.size() << " inputs" << std::endl;
                std::cout << "Input tensor shape: " << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[0] << " "
                          << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[1] << " "
                          << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[2] << " "
                          << inputs[0].GetTensorTypeAndShapeInfo().GetShape()[3] << std::endl;
            }
            
            outputs = decoderSession_->run(inputs);
            
            if (verbose_) {
                std::cout << "Decoder run completed, got " << outputs.size() << " outputs" << std::endl;
            }
            
            if (outputs.empty()) {
                setLastError("Decoder failed to produce output");
                return {"", false, -1};
            }
            
        } catch (const std::exception& e) {
            if (verbose_) {
                std::cout << "Decoder run failed with exception: " << e.what() << std::endl;
            }
            setLastError("Decoder run failed: " + std::string(e.what()));
            return {"", false, -1};
        }

        if (outputs.empty()) {
            setLastError("Decoder failed to produce output");
            return {"", false, -1};
        }

        // Extract secret bits
        std::vector<float> secretOutput = onnx_utils::extractOutputTensor(outputs[0]);

        // Debug: Print raw decoder output
        if (verbose_) {
            std::cout << "Raw decoder output (first 20 values): ";
            for (size_t i = 0; i < std::min(secretOutput.size(), size_t(20)); ++i) {
                std::cout << secretOutput[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Total decoder output size: " << secretOutput.size() << std::endl;

            // Also show the input image info
            std::cout << "Input image for decoding - size: " << stegoImage.cols << "x" << stegoImage.rows
                      << ", channels: " << stegoImage.channels() << std::endl;
            std::cout << "Input image data (first 10 pixels): ";
            for (int i = 0; i < std::min(10, stegoImage.cols * stegoImage.rows); ++i) {
                cv::Vec3b pixel = stegoImage.at<cv::Vec3b>(i / stegoImage.cols, i % stegoImage.cols);
                std::cout << "(" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ") ";
            }
            std::cout << std::endl;
        }

        // Convert to binary array
        std::vector<bool> secretBinaryArray;
        for (float val : secretOutput) {
            secretBinaryArray.push_back(val > 0.0f);
        }

        // Decode secret
        if (useECC_) {
            // Use BCH decoding (simplified)
            std::string secret = utils::decodeText(secretBinaryArray);
            return {secret, true, 0};
        } else {
            // Direct binary decoding
            std::string secret = utils::decodeBinary(secretBinaryArray);
            return {secret, true, -1};
        }

    } catch (const std::exception& e) {
        setLastError("Decode failed: " + std::string(e.what()));
        return {"", false, -1};
    }
}

// Remove watermark




// Get schema capacity
int TrustMark::getSchemaCapacity() const {
    if (useECC_) {
        // Simplified - would need full BCH implementation
        return secretLen_ - 10; // Approximate overhead
    } else {
        return secretLen_;
    }
}

// Get image for processing
cv::Mat TrustMark::getImageForProcessing(const cv::Mat& inputImage) {
    return imageProcessor_->cropCenterRegion(inputImage, concentrateWmRegion_, aspectRatioLim_);
}

// Put image after processing
cv::Mat TrustMark::putImageAfterProcessing(const cv::Mat& wmImage,
                                          const cv::Mat& coverImage, bool feather) {
    return imageProcessor_->featherBlend(wmImage, coverImage,
                                       static_cast<int>(std::min(wmImage.cols, wmImage.rows) * FEATHERING_RESIDUAL));
}

// Feather paste (legacy method)
void TrustMark::featherPaste(cv::Mat& outImage, const cv::Mat& coverImage,
                            const cv::Mat& wmImage, int top, int bottom, int left, int right,
                            int featherSize) {
    // Implementation would go here - simplified for now
    outImage = imageProcessor_->featherBlend(wmImage, coverImage, featherSize);
}

// Set last error
void TrustMark::setLastError(const std::string& error) const {
    lastError_ = error;
    if (verbose_) {
        std::cerr << "TrustMark Error: " << error << std::endl;
    }
}

// Utility functions implementation
namespace utils {

    // Image conversion utilities
    cv::Mat pilToCv(const std::string& imagePath) {
        return cv::imread(imagePath, cv::IMREAD_COLOR);
    }

    cv::Mat resizeImage(const cv::Mat& image, int width, int height) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));
        return resized;
    }

    cv::Mat normalizeImage(const cv::Mat& image, float minVal, float maxVal) {
        cv::Mat normalized;
        image.convertTo(normalized, CV_32F, (maxVal - minVal) / 255.0, minVal);
        return normalized;
    }

    cv::Mat denormalizeImage(const cv::Mat& image, float minVal, float maxVal) {
        cv::Mat denormalized;
        image.convertTo(denormalized, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        return denormalized;
    }

    // BCH error correction utilities (simplified interface)
    std::vector<bool> encodeText(const std::string& text, int secretLen) {
        std::vector<bool> bits;
        for (char c : text) {
            for (int i = 7; i >= 0; --i) {
                bits.push_back((c >> i) & 1);
            }
        }
        // Pad or truncate to secretLen
        if (bits.size() < secretLen) {
            bits.resize(secretLen, false);
        } else if (bits.size() > secretLen) {
            bits.resize(secretLen);
        }
        return bits;
    }

    std::string decodeText(const std::vector<bool>& bits) {
        std::string text;
        for (size_t i = 0; i < bits.size(); i += 8) {
            if (i + 7 >= bits.size()) break;
            char c = 0;
            for (int j = 0; j < 8; ++j) {
                if (bits[i + j]) {
                    c |= (1 << (7 - j));
                }
            }
            text += c;
        }
        return text;
    }

    std::vector<bool> encodeBinary(const std::string& binaryString) {
        std::vector<bool> bits;
        for (char c : binaryString) {
            bits.push_back(c == '1');
        }
        return bits;
    }

    std::string decodeBinary(const std::vector<bool>& bits) {
        std::string binaryString;
        for (bool bit : bits) {
            binaryString += (bit ? '1' : '0');
        }
        return binaryString;
    }

} // namespace utils

} // namespace TrustMark
