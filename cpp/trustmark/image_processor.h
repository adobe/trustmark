#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace TrustMark {

class ImageProcessor {
public:
    // Constructor
    ImageProcessor();

    // Destructor
    ~ImageProcessor() = default;

    // Image preprocessing for encoder
    cv::Mat preprocessForEncoder(const cv::Mat& image, int targetSize = 256);

    // Image preprocessing for decoder
    cv::Mat preprocessForDecoder(const cv::Mat& image, int targetSize = 224);



    // Image postprocessing
    cv::Mat postprocessFromEncoder(const cv::Mat& residual,
                                  const cv::Mat& originalImage,
                                  float wmStrength = 1.0f,
                                  const std::string& wmMerge = "bilinear");

    // Image resizing with interpolation
    cv::Mat resizeImage(const cv::Mat& image,
                        int width,
                        int height,
                        const std::string& interpolation = "bilinear");

    // Image normalization
    cv::Mat normalizeImage(const cv::Mat& image,
                          float minVal = -1.0f,
                          float maxVal = 1.0f);

    // Image denormalization
    cv::Mat denormalizeImage(const cv::Mat& image,
                            float minVal = 0.0f,
                            float maxVal = 255.0f);

    // Color space conversion
    cv::Mat bgrToRgb(const cv::Mat& image);
    cv::Mat rgbToBgr(const cv::Mat& image);

    // Image cropping and region selection
    cv::Mat cropCenterRegion(const cv::Mat& image,
                            float scale = 1.0f,
                            float aspectRatioLimit = 2.0f);

    // Feathering and blending
    cv::Mat featherBlend(const cv::Mat& foreground,
                         const cv::Mat& background,
                         int featherSize = 9);

    // Utility methods
    std::string getLastError() const { return lastError_; }
    void clearLastError() { lastError_.clear(); }

    // Constants
    static constexpr float DEFAULT_ASPECT_RATIO_LIMIT = 2.0f;
    static constexpr float DEFAULT_FEATHERING_RESIDUAL = 0.01f;

private:
    // Private helper methods
    void setLastError(const std::string& error) const;
    int getInterpolationMethod(const std::string& method) const;
    cv::Mat applyFeathering(const cv::Mat& image, int featherSize);

    // Member variables
    mutable std::string lastError_;

    // Interpolation method mapping
    std::map<std::string, int> interpolationMethods_;
};

// Utility functions for image processing
namespace image_utils {
    // Load image from file
    cv::Mat loadImage(const std::string& path);

    // Save image to file
    bool saveImage(const std::string& path, const cv::Mat& image);

    // Convert PIL-style image to OpenCV Mat
    cv::Mat pilToCv(const std::string& imagePath);

    // Convert OpenCV Mat to PIL-style image
    std::string cvToPil(const cv::Mat& image);

    // Image validation
    bool isValidImage(const cv::Mat& image);
    bool isColorImage(const cv::Mat& image);
    bool isGrayscaleImage(const cv::Mat& image);

    // Image statistics
    double getImagePSNR(const cv::Mat& original, const cv::Mat& processed);
    double getImageMSE(const cv::Mat& original, const cv::Mat& processed);

    // Image enhancement
    cv::Mat enhanceContrast(const cv::Mat& image, double alpha = 1.5, double beta = 0.0);
    cv::Mat reduceNoise(const cv::Mat& image, int kernelSize = 5);
}

} // namespace TrustMark
