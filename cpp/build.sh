#!/bin/bash

# TrustMark C++ Build Script

set -e

echo "Building TrustMark C++ Library..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"
echo "You can now run the example with: ./trustmark_example <image_path> [message]"
