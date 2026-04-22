#pragma once

#include <vector>
#include <string>
#include <cstdint>

// Minimal image utilities for WASM using stb libraries
// Replaces OpenCV functionality with lightweight alternatives

namespace ImageUtils {

// Simple image structure
struct Image {
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;

    Image() : width(0), height(0), channels(0) {}
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }

    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
};

// Load image from file (supports PNG, JPEG, BMP, etc.)
Image loadImage(const char* filename);
Image loadImage(const std::string& filename);

// Load image from memory (e.g. HTTP request body)
Image loadImageFromMemory(const uint8_t* data, size_t size);

// Save image to file
bool saveImage(const char* filename, const Image& img);
bool saveImage(const std::string& filename, const Image& img);

// Encode image as PNG into a memory buffer (for HTTP response body)
bool savePNGToMemory(const Image& img, std::vector<uint8_t>& out);

// Resize image (bilinear interpolation)
Image resizeImage(const Image& img, int newWidth, int newHeight);

// Color space conversions
Image bgrToRgb(const Image& img);
Image rgbToBgr(const Image& img);

// Type conversions
std::vector<float> uint8ToFloat(const Image& img, float scale = 1.0f/255.0f);
Image floatToUint8(const std::vector<float>& data, int width, int height, int channels, float scale = 255.0f);

// Convert to normalized float [-1, 1]
std::vector<float> normalizeImage(const Image& img);

// Convert from normalized float [-1, 1] to uint8
Image denormalizeImage(const std::vector<float>& data, int width, int height, int channels);

} // namespace ImageUtils
