#include "wasm/image_utils.h"
#include <iostream>

int main() {
    auto img = ImageUtils::loadImage("output_watermarked.png");

    if (img.empty()) {
        std::cout << "Failed to load" << std::endl;
        return 1;
    }

    // Calculate average per channel
    long long r_sum = 0, g_sum = 0, b_sum = 0;
    for (size_t i = 0; i < img.data.size(); i += 3) {
        r_sum += img.data[i];
        g_sum += img.data[i+1];
        b_sum += img.data[i+2];
    }

    int num_pixels = img.width * img.height;
    std::cout << "Channel averages:" << std::endl;
    std::cout << "  R: " << (r_sum / num_pixels) << std::endl;
    std::cout << "  G: " << (g_sum / num_pixels) << std::endl;
    std::cout << "  B: " << (b_sum / num_pixels) << std::endl;

    // If G is way higher than R and B, it's still wrong
    int r_avg = r_sum / num_pixels;
    int g_avg = g_sum / num_pixels;
    int b_avg = b_sum / num_pixels;

    if (g_avg > r_avg * 1.5 && g_avg > b_avg * 1.5) {
        std::cout << "\n❌ STILL TOO GREEN - BGR conversion not working!" << std::endl;
    } else if (std::abs(r_avg - g_avg) < 30 && std::abs(g_avg - b_avg) < 30) {
        std::cout << "\n✅ Channels balanced - colors look correct!" << std::endl;
    } else {
        std::cout << "\n? Colors somewhat balanced but check visually" << std::endl;
    }

    return 0;
}
