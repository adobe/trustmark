#include "wasm/image_utils.h"
#include <iostream>

int main() {
    // Load image
    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    std::cout << "Loaded: " << img.width << "x" << img.height << " ch=" << img.channels << std::endl;
    
    // Resize
    auto resized = ImageUtils::resizeImage(img, 256, 256);
    std::cout << "Resized: " << resized.width << "x" << resized.height << std::endl;
    
    // Save directly (should look correct)
    ImageUtils::saveImage("test_direct.png", resized);
    std::cout << "Saved test_direct.png" << std::endl;
    
    // Now test normalize + denormalize
    auto norm = ImageUtils::normalizeImage(resized);
    auto denorm = ImageUtils::denormalizeImage(norm, 256, 256, 3);
    ImageUtils::saveImage("test_roundtrip.png", denorm);
    std::cout << "Saved test_roundtrip.png" << std::endl;
    
    return 0;
}
