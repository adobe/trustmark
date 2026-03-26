#include "wasm/image_utils.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image.png>" << std::endl;
        return 1;
    }
    
    auto img = ImageUtils::loadImage(argv[1]);
    std::cout << "Image: " << img.width << "x" << img.height << " channels=" << img.channels << std::endl;
    
    // Check first few pixels
    std::cout << "\nFirst 10 pixels (R,G,B):" << std::endl;
    for (int i = 0; i < 10 && i < img.width; i++) {
        int idx = i * 3;
        std::cout << "  Pixel " << i << ": (" 
                  << (int)img.data[idx] << "," 
                  << (int)img.data[idx+1] << "," 
                  << (int)img.data[idx+2] << ")" << std::endl;
    }
    
    // Check value distribution
    long r_sum = 0, g_sum = 0, b_sum = 0;
    int r_min = 255, g_min = 255, b_min = 255;
    int r_max = 0, g_max = 0, b_max = 0;
    
    for (size_t i = 0; i < img.data.size(); i += 3) {
        int r = img.data[i];
        int g = img.data[i+1];
        int b = img.data[i+2];
        
        r_sum += r; g_sum += g; b_sum += b;
        
        if (r < r_min) r_min = r;
        if (g < g_min) g_min = g;
        if (b < b_min) b_min = b;
        
        if (r > r_max) r_max = r;
        if (g > g_max) g_max = g;
        if (b > b_max) b_max = b;
    }
    
    int num_pixels = img.width * img.height;
    std::cout << "\nChannel statistics:" << std::endl;
    std::cout << "  R: sum=" << r_sum << " avg=" << (r_sum / num_pixels) << " range=[" << r_min << "," << r_max << "]" << std::endl;
    std::cout << "  G: sum=" << g_sum << " avg=" << (g_sum / num_pixels) << " range=[" << g_min << "," << g_max << "]" << std::endl;
    std::cout << "  B: sum=" << b_sum << " avg=" << (b_sum / num_pixels) << " range=[" << b_min << "," << b_max << "]" << std::endl;
    
    return 0;
}
