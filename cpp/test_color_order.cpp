#include "wasm/image_utils.h"
#include <iostream>

int main() {
    // Load a simple test - we know ufo_240.jpg has dark blue sky at top-left
    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    
    std::cout << "First pixel (should be dark, mostly blue):" << std::endl;
    std::cout << "  Channel 0: " << (int)img.data[0] << std::endl;
    std::cout << "  Channel 1: " << (int)img.data[1] << std::endl;
    std::cout << "  Channel 2: " << (int)img.data[2] << std::endl;
    
    // The sky should have B > R and B > G
    // If stb loads as RGB: we expect something like (10, 36, 51) - low R, medium G, higher B
    // If stb loads as BGR: we expect (51, 36, 10) - high "R" (actually B), medium G, low "B" (actually R)
    
    std::cout << "\nExpected for dark blue sky: R=low, G=medium, B=high" << std::endl;
    
    if (img.data[2] > img.data[0] && img.data[2] > img.data[1]) {
        std::cout << "✓ Looks like RGB format (channel 2 is highest = Blue)" << std::endl;
    } else if (img.data[0] > img.data[1] && img.data[0] > img.data[2]) {
        std::cout << "✗ Looks like BGR format (channel 0 is highest = Blue in BGR)" << std::endl;
    } else {
        std::cout << "? Unclear color order" << std::endl;
    }
    
    return 0;
}
