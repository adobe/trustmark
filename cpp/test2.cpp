#include "wasm/image_utils.h"
#include <iostream>

int main() {
    std::cout << "Testing..." << std::endl;
    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    std::cout << "Loaded: " << img.width << "x" << img.height << std::endl;
    bool ok = ImageUtils::saveImage("output/test_save.png", img);
    std::cout << "Save result: " << (ok ? "SUCCESS" : "FAILED") << std::endl;
    return 0;
}
