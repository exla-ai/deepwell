#!/bin/bash
#
# Production build script for Deepwell on B200
# Run this on your Blackwell machine after git pull
#

set -e  # Exit on error

echo "========================================="
echo "Building Deepwell Production for B200"
echo "========================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -f src/deepwell/*.so

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "Found CUDA version: $CUDA_VERSION"

# Check for CUTLASS
if [ -d "/usr/local/cutlass/include" ]; then
    echo "Found CUTLASS at /usr/local/cutlass"
    export CUTLASS_PATH=/usr/local/cutlass
elif [ -d "third_party/cutlass" ]; then
    echo "Found CUTLASS at third_party/cutlass"
    export CUTLASS_PATH=$(pwd)/third_party/cutlass
else
    echo "WARNING: CUTLASS not found. Some features may be limited."
    echo "To install CUTLASS:"
    echo "  git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass"
    echo "  cd third_party/cutlass && git checkout v3.5.0"
fi

# Build with production settings
echo "Building C++ extension..."
python setup.py build_ext --inplace

# Test that it built correctly
echo ""
echo "Testing import..."
python -c "import deepwell.cutlass_kernels; print('✓ C++ extension loaded successfully')"

# Run correctness test
echo ""
echo "Running correctness test..."
python test_production.py

echo ""
echo "========================================="
echo "Build complete! Next steps:"
echo "1. Run benchmarks: python benchmarks/benchmark.py"
echo "2. Check correctness: python test_production.py"
echo "========================================="
