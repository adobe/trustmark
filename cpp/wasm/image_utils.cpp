#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"
#include "image_utils.h"

#include <iostream>
#include <algorithm>
#include <cmath>

namespace ImageUtils {

// Load image from file
Image loadImage(const char* filename) {
    Image img;

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        return img;
    }

    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data.assign(data, data + (width * height * channels));

    stbi_image_free(data);

    return img;
}

Image loadImage(const std::string& filename) {
    return loadImage(filename.c_str());
}

// Save image to file
bool saveImage(const char* filename, const Image& img) {
    if (img.empty()) {
        std::cerr << "Cannot save empty image" << std::endl;
        return false;
    }

    std::string fname(filename);

    // Determine format from extension
    if (fname.find(".png") != std::string::npos) {
        return stbi_write_png(filename, img.width, img.height, img.channels,
                            img.data.data(), img.width * img.channels) != 0;
    } else if (fname.find(".jpg") != std::string::npos ||
               fname.find(".jpeg") != std::string::npos) {
        return stbi_write_jpg(filename, img.width, img.height, img.channels,
                            img.data.data(), 90) != 0;
    } else if (fname.find(".bmp") != std::string::npos) {
        return stbi_write_bmp(filename, img.width, img.height, img.channels,
                            img.data.data()) != 0;
    }

    std::cerr << "Unsupported image format: " << filename << std::endl;
    return false;
}

bool saveImage(const std::string& filename, const Image& img) {
    return saveImage(filename.c_str(), img);
}

// Resize image
Image resizeImage(const Image& img, int newWidth, int newHeight) {
    if (img.empty()) {
        return Image();
    }

    Image resized(newWidth, newHeight, img.channels);

    stbir_resize_uint8_linear(
        img.data.data(), img.width, img.height, 0,
        resized.data.data(), newWidth, newHeight, 0,
        (stbir_pixel_layout)img.channels
    );

    return resized;
}

// BGR to RGB conversion (swap R and B channels)
Image bgrToRgb(const Image& img) {
    if (img.channels != 3) {
        std::cerr << "BGR to RGB conversion requires 3-channel image" << std::endl;
        return img;
    }

    Image rgb = img;

    for (size_t i = 0; i < rgb.data.size(); i += 3) {
        std::swap(rgb.data[i], rgb.data[i + 2]);
    }

    return rgb;
}

Image rgbToBgr(const Image& img) {
    // Same operation as bgrToRgb (swap R and B)
    return bgrToRgb(img);
}

// Convert uint8 [0,255] to float with scaling
std::vector<float> uint8ToFloat(const Image& img, float scale) {
    std::vector<float> result(img.data.size());

    for (size_t i = 0; i < img.data.size(); ++i) {
        result[i] = static_cast<float>(img.data[i]) * scale;
    }

    return result;
}

// Convert float to uint8 [0,255]
Image floatToUint8(const std::vector<float>& data, int width, int height, int channels, float scale) {
    Image img(width, height, channels);

    for (size_t i = 0; i < data.size(); ++i) {
        float val = data[i] * scale;
        val = std::max(0.0f, std::min(255.0f, val));
        img.data[i] = static_cast<uint8_t>(val);
    }

    return img;
}

// Normalize to [-1, 1]
std::vector<float> normalizeImage(const Image& img) {
    std::vector<float> result(img.data.size());

    for (size_t i = 0; i < img.data.size(); ++i) {
        // [0, 255] -> [0, 1] -> [-1, 1]
        result[i] = (static_cast<float>(img.data[i]) / 255.0f) * 2.0f - 1.0f;
    }

    return result;
}

// Denormalize from [-1, 1] to uint8
Image denormalizeImage(const std::vector<float>& data, int width, int height, int channels) {
    Image img(width, height, channels);

    for (size_t i = 0; i < data.size(); ++i) {
        // [-1, 1] -> [0, 1] -> [0, 255]
        float val = (data[i] + 1.0f) * 0.5f * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        img.data[i] = static_cast<uint8_t>(val);
    }

    return img;
}

} // namespace ImageUtils
