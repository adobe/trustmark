#include "wasm/image_utils.h"
#include <iostream>

int main() {
    // Create a test image with known RGB values
    ImageUtils::Image rgb(2, 2, 3);
    // Pixel 0: Red (255, 0, 0)
    rgb.data[0] = 255; rgb.data[1] = 0; rgb.data[2] = 0;
    // Pixel 1: Green (0, 255, 0)
    rgb.data[3] = 0; rgb.data[4] = 255; rgb.data[5] = 0;
    // Pixel 2: Blue (0, 0, 255)
    rgb.data[6] = 0; rgb.data[7] = 0; rgb.data[8] = 255;
    // Pixel 3: White (255, 255, 255)
    rgb.data[9] = 255; rgb.data[10] = 255; rgb.data[11] = 255;

    std::cout << "Original RGB:" << std::endl;
    std::cout << "  Pixel 0: (" << (int)rgb.data[0] << "," << (int)rgb.data[1] << "," << (int)rgb.data[2] << ")" << std::endl;

    // Convert to BGR
    auto bgr = ImageUtils::rgbToBgr(rgb);
    std::cout << "\nAfter rgbToBgr:" << std::endl;
    std::cout << "  Pixel 0: (" << (int)bgr.data[0] << "," << (int)bgr.data[1] << "," << (int)bgr.data[2] << ")" << std::endl;

    // Convert back to RGB
    auto rgb2 = ImageUtils::bgrToRgb(bgr);
    std::cout << "\nAfter bgrToRgb:" << std::endl;
    std::cout << "  Pixel 0: (" << (int)rgb2.data[0] << "," << (int)rgb2.data[1] << "," << (int)rgb2.data[2] << ")" << std::endl;

    if (rgb2.data[0] == 255 && rgb2.data[1] == 0 && rgb2.data[2] == 0) {
        std::cout << "\n✅ RGB <-> BGR conversion works correctly!" << std::endl;
    } else {
        std::cout << "\n❌ RGB <-> BGR conversion is broken!" << std::endl;
    }

    return 0;
}
