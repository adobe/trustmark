// Test MLAS sigmoid directly
#include <iostream>
#include <vector>
#include <cmath>

// From MLAS
extern "C" void MlasComputeLogistic(const float* Input, float* Output, size_t N);

int main() {
    std::vector<float> input(10);
    std::vector<float> output(10);

    // Same test inputs
    for (int i = 0; i < 10; i++) input[i] = -2.0f + i * 0.5f;

    MlasComputeLogistic(input.data(), output.data(), 10);

    std::cout << "MLAS Sigmoid output: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Compare with std::exp sigmoid
    std::cout << "Expected (1/(1+exp(-x))): ";
    for (int i = 0; i < 10; i++) {
        float expected = 1.0f / (1.0f + std::exp(-input[i]));
        std::cout << expected << " ";
    }
    std::cout << std::endl;

    return 0;
}
