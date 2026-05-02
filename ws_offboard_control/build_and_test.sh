#!/bin/bash
# Build and run the control math test

echo "=========================================="
echo "Building Control Math Test"
echo "=========================================="

# Compile with Eigen
g++ -o test_control_math test_control_math.cpp \
    -I/usr/include/eigen3 \
    -std=c++17 \
    -O2 \
    -Wall

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "=========================================="
    echo "Running Tests"
    echo "=========================================="
    ./test_control_math
else
    echo "✗ Build failed!"
    exit 1
fi
