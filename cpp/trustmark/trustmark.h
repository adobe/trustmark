#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace TrustMark {

// Forward declarations
class ONNXRuntimeSession;
class ImageProcessor;

// Encoding types enum
enum class EncodingType {
    BCH_SUPER = 0,
    BCH_3 = 3,
    BCH_4 = 2,
    BCH_5 = 1
};

// Mode types
enum class Mode {
    TEXT = 0,
    BINARY = 1
};

// Main TrustMark class
class TrustMark {
public:
    // Constructor
    TrustMark(bool useECC = true, 
              bool verbose = true, 
              int secretLen = 100, 
              const std::string& modelType = "Q",
              EncodingType encodingType = EncodingType::BCH_5,
              float concentrateWmRegion = 1.0f);
    
    // Destructor
    ~TrustMark();
    
    // Disable copy constructor and assignment
    TrustMark(const TrustMark&) = delete;
    TrustMark& operator=(const TrustMark&) = delete;
    
    // Move constructor and assignment
    TrustMark(TrustMark&&) noexcept;
    TrustMark& operator=(TrustMark&&) noexcept;
    
    // Main methods
    cv::Mat encode(const cv::Mat& coverImage, 
                  const std::string& secret, 
                  Mode mode = Mode::TEXT,
                  float wmStrength = 1.0f,
                  const std::string& wmMerge = "bilinear");
    
    std::tuple<std::string, bool, int> decode(const cv::Mat& stegoImage, 
                                             Mode mode = Mode::TEXT);
    
    cv::Mat removeWatermark(const cv::Mat& stegoImage, 
                           float wmStrength = 1.0f,
                           const std::string& wmMerge = "bilinear");
    
    // Utility methods
    int getSchemaCapacity() const;
    bool isVerbose() const { return verbose_; }
    std::string getModelType() const { return modelType_; }
    
    // Error handling
    std::string getLastError() const { return lastError_; }
    void clearLastError() { lastError_.clear(); }

private:
    // Private helper methods
    bool initializeModels();
    cv::Mat getImageForProcessing(const cv::Mat& inputImage);
    cv::Mat putImageAfterProcessing(const cv::Mat& wmImage, 
                                   const cv::Mat& coverImage, 
                                   bool feather = true);
    void featherPaste(cv::Mat& outImage,
                     const cv::Mat& coverImage,
                     const cv::Mat& wmImage,
                     int top, int bottom, int left, int right,
                     int featherSize = 9);
    
    // Error handling
    void setLastError(const std::string& error) const;
    
    // Member variables
    bool useECC_;
    bool verbose_;
    int secretLen_;
    std::string modelType_;
    EncodingType encodingType_;
    float concentrateWmRegion_;
    float aspectRatioLim_;
    
    // Model resolution settings
    int modelResolutionEnc_;
    int modelResolutionDec_;
    int modelResolutionRemove_;
    float strengthMultiplier_; // P variant uses 1.25, others use 1.0
    
    // ONNX Runtime sessions
    std::unique_ptr<ONNXRuntimeSession> encoderSession_;
    std::unique_ptr<ONNXRuntimeSession> decoderSession_;
    std::unique_ptr<ONNXRuntimeSession> removalSession_;
    
    // Image processor
    std::unique_ptr<ImageProcessor> imageProcessor_;
    
    // Error state
    mutable std::string lastError_;
    
    // Constants
    static constexpr float FEATHERING_RESIDUAL = 0.01f;
    static constexpr bool FALLBACK_ALL_SCHEMAS = true;
};

// Utility functions
namespace utils {
    // Image conversion utilities
    cv::Mat pilToCv(const std::string& imagePath);
    cv::Mat resizeImage(const cv::Mat& image, int width, int height);
    cv::Mat normalizeImage(const cv::Mat& image, float minVal = -1.0f, float maxVal = 1.0f);
    cv::Mat denormalizeImage(const cv::Mat& image, float minVal = 0.0f, float maxVal = 255.0f);
    
    // BCH error correction utilities (simplified interface)
    std::vector<bool> encodeText(const std::string& text, int secretLen);
    std::string decodeText(const std::vector<bool>& bits);
    std::vector<bool> encodeBinary(const std::string& binaryString);
    std::string decodeBinary(const std::vector<bool>& bits);
}

} // namespace TrustMark
