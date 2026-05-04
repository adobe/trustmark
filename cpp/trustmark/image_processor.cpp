#include "image_processor.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace TrustMark {

// Constructor
ImageProcessor::ImageProcessor() {
    // Initialize interpolation method mapping
    interpolationMethods_["nearest"] = cv::INTER_NEAREST;
    interpolationMethods_["bilinear"] = cv::INTER_LINEAR;
    interpolationMethods_["bicubic"] = cv::INTER_CUBIC;
    interpolationMethods_["lanczos"] = cv::INTER_LANCZOS4;
}

// Preprocess for encoder
cv::Mat ImageProcessor::preprocessForEncoder(const cv::Mat& image, int targetSize) {
    try {
        // Ensure image is in RGB format
        cv::Mat processedImage = image;
        if (image.channels() == 1) {
            cv::cvtColor(image, processedImage, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 3) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGR2RGB);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGRA2RGB);
        }

        // Debug: Print input image info
        std::cout << "DEBUG: Input image - size: " << processedImage.cols << "x" << processedImage.rows
                  << ", channels: " << processedImage.channels() << ", depth: " << processedImage.depth() << std::endl;

        // Print first few pixel values of input image
        std::cout << "DEBUG: Input image first 5 pixels: ";
        for (int i = 0; i < std::min(5, processedImage.cols * processedImage.rows); ++i) {
            int row = i / processedImage.cols;
            int col = i % processedImage.cols;
            if (processedImage.channels() == 3) {
                cv::Vec3b pixel = processedImage.at<cv::Vec3b>(row, col);
                std::cout << "(" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ") ";
            } else {
                uchar pixel = processedImage.at<uchar>(row, col);
                std::cout << (int)pixel << " ";
            }
        }
        std::cout << std::endl;

        // For P variant: Always center crop to square (like Rust's center_crop_size_and_offset)
        cv::Mat croppedImage = processedImage;
        int width = processedImage.cols;
        int height = processedImage.rows;

        // P variant always forces center square crop regardless of aspect ratio
        int cropSize = std::min(width, height);
        int xOffset = (width - cropSize) / 2;
        int yOffset = (height - cropSize) / 2;

        // Extract center square region
        cv::Rect cropRect(xOffset, yOffset, cropSize, cropSize);
        croppedImage = processedImage(cropRect);

        // Debug: Print cropped image info
        std::cout << "DEBUG: Cropped image - size: " << croppedImage.cols << "x" << croppedImage.rows
                  << ", channels: " << croppedImage.channels() << std::endl;

        // Print first few pixel values of cropped image
        std::cout << "DEBUG: Cropped image first 5 pixels: ";
        for (int i = 0; i < std::min(5, croppedImage.cols * croppedImage.rows); ++i) {
            int row = i / croppedImage.cols;
            int col = i % croppedImage.cols;
            if (croppedImage.channels() == 3) {
                cv::Vec3b pixel = croppedImage.at<cv::Vec3b>(row, col);
                std::cout << "(" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ") ";
            } else {
                uchar pixel = croppedImage.at<uchar>(row, col);
                std::cout << (int)pixel << " ";
            }
        }
        std::cout << std::endl;

        // Resize to target size (256x256 for P variant)
        cv::Mat resizedImage = resizeImage(croppedImage, targetSize, targetSize, "bilinear");

        // Debug: Print resized image info
        std::cout << "DEBUG: Resized image - size: " << resizedImage.cols << "x" << resizedImage.rows
                  << ", channels: " << resizedImage.channels() << std::endl;

        // Print first few pixel values of resized image
        std::cout << "DEBUG: Resized image first 5 pixels: ";
        for (int i = 0; i < std::min(5, resizedImage.cols * resizedImage.rows); ++i) {
            int row = i / resizedImage.cols;
            int col = i % resizedImage.cols;
            if (resizedImage.channels() == 3) {
                cv::Vec3b pixel = resizedImage.at<cv::Vec3b>(row, col);
                std::cout << "(" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ") ";
            } else {
                uchar pixel = resizedImage.at<uchar>(row, col);
                std::cout << (int)pixel << " ";
            }
        }
        std::cout << std::endl;

        // Convert to float32 and normalize to [0, 1] range (like Rust's into_rgb32f())
        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32F, 1.0/255.0);

        // Debug: Print float image info after convertTo
        std::cout << "DEBUG: After convertTo(1.0/255.0) - first 5 pixels: ";
        for (int i = 0; i < std::min(5, floatImage.cols * floatImage.rows); ++i) {
            int row = i / floatImage.cols;
            int col = i % floatImage.cols;
            if (floatImage.channels() == 3) {
                cv::Vec3f pixel = floatImage.at<cv::Vec3f>(row, col);
                std::cout << "(" << pixel[0] << "," << pixel[1] << "," << pixel[2] << ") ";
            } else {
                float pixel = floatImage.at<float>(row, col);
                std::cout << pixel << " ";
            }
        }
        std::cout << std::endl;

        // Convert from [0,1] to [-1,1] range (like Rust's convert_from_0_1_to_neg1_1! macro)
        cv::Mat normalizedImage = floatImage * 2.0f - 1.0f;

        // Debug: Print final normalized image info
        std::cout << "DEBUG: After *2.0-1.0 - first 5 pixels: ";
        for (int i = 0; i < std::min(5, normalizedImage.cols * normalizedImage.rows); ++i) {
            int row = i / normalizedImage.cols;
            int col = i % normalizedImage.cols;
            if (normalizedImage.channels() == 3) {
                cv::Vec3f pixel = normalizedImage.at<cv::Vec3f>(row, col);
                std::cout << "(" << pixel[0] << "," << pixel[1] << "," << pixel[2] << ") ";
            } else {
                float pixel = normalizedImage.at<float>(row, col);
                std::cout << pixel << " ";
            }
        }
        std::cout << std::endl;

        return normalizedImage;
    } catch (const std::exception& e) {
        setLastError("Preprocessing for encoder failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Preprocess for decoder
cv::Mat ImageProcessor::preprocessForDecoder(const cv::Mat& image, int targetSize) {
    try {
        // Convert to RGB to match model training
        cv::Mat processedImage = image;
        if (image.channels() == 1) {
            cv::cvtColor(image, processedImage, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 3) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGR2RGB);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGRA2RGB);
        }

        // For P variant, decode size is 224. We rely on caller passing targetSize accordingly
        cv::Mat resizedImage = resizeImage(processedImage, targetSize, targetSize, "bilinear");

        // Normalize to [0,1] then map to [-1,1]
        cv::Mat floatImage; resizedImage.convertTo(floatImage, CV_32F, 1.0/255.0);
        cv::Mat normalizedImage = floatImage * 2.0f - 1.0f;

        return normalizedImage;

    } catch (const std::exception& e) {
        setLastError("Preprocessing for decoder failed: " + std::string(e.what()));
        return cv::Mat();
    }
}



// Postprocess from encoder
cv::Mat ImageProcessor::postprocessFromEncoder(const cv::Mat& residual,
                                              const cv::Mat& originalImage,
                                              float wmStrength, const std::string& wmMerge) {
    try {
        // Resize residual to match original image size
        cv::Mat resizedResidual = resizeImage(residual,
                                            originalImage.cols,
                                            originalImage.rows,
                                            wmMerge);

        // Denormalize residual
        cv::Mat denormalizedResidual = denormalizeImage(resizedResidual, -1.0f, 1.0f);

        // Denormalize original image
        cv::Mat denormalizedOriginal = denormalizeImage(originalImage, 0.0f, 255.0f);

        // Apply watermark strength and merge
        cv::Mat watermarkedImage = denormalizedOriginal + wmStrength * denormalizedResidual;
        cv::threshold(watermarkedImage, watermarkedImage, 0.0, 255.0, cv::THRESH_TOZERO);
        cv::threshold(watermarkedImage, watermarkedImage, 255.0, 255.0, cv::THRESH_TRUNC);

        return watermarkedImage;

    } catch (const std::exception& e) {
        setLastError("Postprocessing from encoder failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Resize image
cv::Mat ImageProcessor::resizeImage(const cv::Mat& image, int width, int height,
                                   const std::string& interpolation) {
    try {
        int method = getInterpolationMethod(interpolation);
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(width, height), 0, 0, method);
        return resizedImage;

    } catch (const std::exception& e) {
        setLastError("Image resizing failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Normalize image
cv::Mat ImageProcessor::normalizeImage(const cv::Mat& image, float minVal, float maxVal) {
    try {
        cv::Mat normalizedImage;

        if (image.depth() == CV_8U) {
            // Convert from [0, 255] to [minVal, maxVal]
            image.convertTo(normalizedImage, CV_32F, (maxVal - minVal) / 255.0, minVal);
        } else if (image.depth() == CV_32F) {
            // Already float, just scale if needed
            if (minVal != 0.0f || maxVal != 1.0f) {
                image.convertTo(normalizedImage, CV_32F, maxVal - minVal, minVal);
            } else {
                normalizedImage = image.clone();
            }
        } else {
            // Convert to float first
            image.convertTo(normalizedImage, CV_32F);
        }

        return normalizedImage;

    } catch (const std::exception& e) {
        setLastError("Image normalization failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Denormalize image
cv::Mat ImageProcessor::denormalizeImage(const cv::Mat& image, float minVal, float maxVal) {
    try {
        cv::Mat denormalizedImage;

        if (image.depth() == CV_32F) {
            // Convert from [minVal, maxVal] to [0, 255]
            image.convertTo(denormalizedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        } else {
            // Convert to float first, then denormalize
            cv::Mat floatImage;
            image.convertTo(floatImage, CV_32F);
            floatImage.convertTo(denormalizedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        }

        return denormalizedImage;

    } catch (const std::exception& e) {
        setLastError("Image denormalization failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// BGR to RGB conversion
cv::Mat ImageProcessor::bgrToRgb(const cv::Mat& image) {
    try {
        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
        return rgbImage;

    } catch (const std::exception& e) {
        setLastError("BGR to RGB conversion failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// RGB to BGR conversion
cv::Mat ImageProcessor::rgbToBgr(const cv::Mat& image) {
    try {
        cv::Mat bgrImage;
        cv::cvtColor(image, bgrImage, cv::COLOR_RGB2BGR);
        return bgrImage;

    } catch (const std::exception& e) {
        setLastError("RGB to BGR conversion failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Crop center region
cv::Mat ImageProcessor::cropCenterRegion(const cv::Mat& image, float scale, float aspectRatioLimit) {
    try {
        int width = image.cols;
        int height = image.rows;

        // Calculate aspect ratio
        float aspectRatio = static_cast<float>(std::max(width, height)) / std::min(width, height);

        cv::Mat croppedImage;

        if (aspectRatio > aspectRatioLimit) {
            // Center-square approach
            int squareSize = std::min(width, height);
            int scaledSize = static_cast<int>(squareSize * scale);

            int left = (width - scaledSize) / 2;
            int top = (height - scaledSize) / 2;
            int right = left + scaledSize;
            int bottom = top + scaledSize;

            croppedImage = image(cv::Rect(left, top, scaledSize, scaledSize));

        } else {
            // Normal aspect ratio, scale entire region
            int scaledWidth = static_cast<int>(width * scale);
            int scaledHeight = static_cast<int>(height * scale);

            int left = (width - scaledWidth) / 2;
            int top = (height - scaledHeight) / 2;
            int right = left + scaledWidth;
            int bottom = top + scaledHeight;

            croppedImage = image(cv::Rect(left, top, scaledWidth, scaledHeight));
        }

        return croppedImage;

    } catch (const std::exception& e) {
        setLastError("Center region cropping failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Feather blend
cv::Mat ImageProcessor::featherBlend(const cv::Mat& foreground, const cv::Mat& background, int featherSize) {
    try {
        // Ensure both images have the same size
        if (foreground.size() != background.size()) {
            cv::Mat resizedForeground;
            cv::resize(foreground, resizedForeground, background.size());
            return featherBlend(resizedForeground, background, featherSize);
        }

        // Create output image
        cv::Mat outputImage = background.clone();

        // Apply feathering
        cv::Mat featheredForeground = applyFeathering(foreground, featherSize);

        // Blend images
        cv::addWeighted(background, 1.0 - 0.5, featheredForeground, 0.5, 0.0, outputImage);

        return outputImage;

    } catch (const std::exception& e) {
        setLastError("Feather blending failed: " + std::string(e.what()));
        return cv::Mat();
    }
}

// Set last error
void ImageProcessor::setLastError(const std::string& error) const {
    lastError_ = error;
}

// Get interpolation method
int ImageProcessor::getInterpolationMethod(const std::string& method) const {
    auto it = interpolationMethods_.find(method);
    if (it != interpolationMethods_.end()) {
        return it->second;
    }
    // Default to bilinear
    return cv::INTER_LINEAR;
}

// Apply feathering
cv::Mat ImageProcessor::applyFeathering(const cv::Mat& image, int featherSize) {
    try {
        cv::Mat featheredImage = image.clone();

        // Create feathering kernel
        cv::Mat kernel = cv::getGaussianKernel(featherSize * 2 + 1, featherSize / 3.0, CV_32F);
        cv::Mat kernel2D = kernel * kernel.t();

        // Apply feathering
        cv::filter2D(featheredImage, featheredImage, -1, kernel2D);

        return featheredImage;

    } catch (const std::exception& e) {
        setLastError("Feathering application failed: " + std::string(e.what()));
        return image.clone();
    }
}

// Image utility functions
namespace image_utils {

// Load image from file
cv::Mat loadImage(const std::string& path) {
    try {
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + path);
        }
        return image;

    } catch (const std::exception& e) {
        throw std::runtime_error("Image loading failed: " + std::string(e.what()));
    }
}

// Save image to file
bool saveImage(const std::string& path, const cv::Mat& image) {
    try {
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
        compressionParams.push_back(95);

        return cv::imwrite(path, image, compressionParams);

    } catch (const std::exception& e) {
        return false;
    }
}

// Convert PIL-style image to OpenCV Mat
cv::Mat pilToCv(const std::string& imagePath) {
    return loadImage(imagePath);
}

// Convert OpenCV Mat to PIL-style image
std::string cvToPil(const cv::Mat& image) {
    // This is a placeholder - in practice you'd save to a temporary file
    // and return the path, or implement actual PIL conversion
    return "converted_image.jpg";
}

// Image validation
bool isValidImage(const cv::Mat& image) {
    return !image.empty() && image.data != nullptr;
}

bool isColorImage(const cv::Mat& image) {
    return isValidImage(image) && image.channels() == 3;
}

bool isGrayscaleImage(const cv::Mat& image) {
    return isValidImage(image) && image.channels() == 1;
}

// Image statistics
double getImagePSNR(const cv::Mat& original, const cv::Mat& processed) {
    try {
        cv::Mat diff;
        cv::absdiff(original, processed, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);

        double mse = cv::mean(diff)[0];
        if (mse <= 1e-10) {
            return 100.0;
        }

        return 20.0 * std::log10(255.0 / std::sqrt(mse));

    } catch (const std::exception& e) {
        return -1.0;
    }
}

double getImageMSE(const cv::Mat& original, const cv::Mat& processed) {
    try {
        cv::Mat diff;
        cv::absdiff(original, processed, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);

        return cv::mean(diff)[0];

    } catch (const std::exception& e) {
        return -1.0;
    }
}

// Image enhancement
cv::Mat enhanceContrast(const cv::Mat& image, double alpha, double beta) {
    try {
        cv::Mat enhancedImage;
        image.convertTo(enhancedImage, -1, alpha, beta);
        return enhancedImage;

    } catch (const std::exception& e) {
        return image.clone();
    }
}

cv::Mat reduceNoise(const cv::Mat& image, int kernelSize) {
    try {
        cv::Mat denoisedImage;
        cv::GaussianBlur(image, denoisedImage, cv::Size(kernelSize, kernelSize), 0);
        return denoisedImage;

    } catch (const std::exception& e) {
        return image.clone();
    }
}

} // namespace image_utils

} // namespace TrustMark
