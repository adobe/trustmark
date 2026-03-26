#include "wasm/image_utils.h"
#include <iostream>
#include <vector>

int main() {
    // Simulate what happens: load image, convert to BGR, normalize
    auto img = ImageUtils::loadImage("../images/ufo_240.jpg");
    auto resized = ImageUtils::resizeImage(img, 256, 256);
    
    std::cout << "Original pixel 0 (RGB from stb): (" 
              << (int)resized.data[0] << "," 
              << (int)resized.data[1] << "," 
              << (int)resized.data[2] << ")" << std::endl;
    
    auto bgr = ImageUtils::rgbToBgr(resized);
    std::cout << "After RGB->BGR: (" 
              << (int)bgr.data[0] << "," 
              << (int)bgr.data[1] << "," 
              << (int)bgr.data[2] << ")" << std::endl;
    
    // Model would process this and output in BGR
    // Let's simulate: model outputs same image back (identity)
    auto model_output_bgr = bgr;
    
    std::cout << "Model output (still BGR): (" 
              << (int)model_output_bgr.data[0] << "," 
              << (int)model_output_bgr.data[1] << "," 
              << (int)model_output_bgr.data[2] << ")" << std::endl;
    
    // Convert back to RGB for saving
    auto final_rgb = ImageUtils::bgrToRgb(model_output_bgr);
    std::cout << "After BGR->RGB: (" 
              << (int)final_rgb.data[0] << "," 
              << (int)final_rgb.data[1] << "," 
              << (int)final_rgb.data[2] << ")" << std::endl;
    
    // Should match original
    if (final_rgb.data[0] == resized.data[0] && 
        final_rgb.data[1] == resized.data[1] && 
        final_rgb.data[2] == resized.data[2]) {
        std::cout << "\n✅ Round-trip conversion works!" << std::endl;
    } else {
        std::cout << "\n❌ Round-trip conversion failed!" << std::endl;
    }
    
    return 0;
}
