#include "wasm/image_utils.h"
#include <iostream>
#include <vector>

int main() {
    // Load and process image like WASM does
    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    auto resized = ImageUtils::resizeImage(img, 256, 256);
    auto normalized = ImageUtils::normalizeImage(resized);

    // Show first few normalized values (this is what goes INTO the model)
    std::cout << "Input to model (first 12 values, should be RRRGGBBBB pattern in CHW):" << std::endl;
    for (int i = 0; i < 12; i++) {
        std::cout << normalized[i] << " ";
    }
    std::cout << std::endl;

    // Now let's simulate what happens if we DON'T convert (assuming output is already HWC)
    auto denorm_no_convert = ImageUtils::denormalizeImage(normalized, 256, 256, 3);
    ImageUtils::saveImage("test_no_convert.png", denorm_no_convert);
    std::cout << "Saved test_no_convert.png (assuming data is already HWC)" << std::endl;

    // And test WITH CHW to HWC conversion
    std::vector<float> converted(256 * 256 * 3);
    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            int chw_idx_r = 0 * 256 * 256 + h * 256 + w;
            int chw_idx_g = 1 * 256 * 256 + h * 256 + w;
            int chw_idx_b = 2 * 256 * 256 + h * 256 + w;
            int hwc_idx = (h * 256 + w) * 3;

            converted[hwc_idx + 0] = normalized[chw_idx_r];
            converted[hwc_idx + 1] = normalized[chw_idx_g];
            converted[hwc_idx + 2] = normalized[chw_idx_b];
        }
    }
    auto denorm_converted = ImageUtils::denormalizeImage(converted, 256, 256, 3);
    ImageUtils::saveImage("test_with_convert.png", denorm_converted);
    std::cout << "Saved test_with_convert.png (with CHW to HWC conversion)" << std::endl;

    return 0;
}
