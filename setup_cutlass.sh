#!/bin/bash
#
# Setup script to clone and build CUTLASS with Blackwell support
#

set -e

echo "Setting up CUTLASS for Blackwell tcgen05.mma support..."

# Check if CUTLASS already exists
if [ ! -d "third_party/cutlass" ]; then
    echo "Cloning CUTLASS..."
    mkdir -p third_party
    cd third_party
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    git checkout v3.8.0  # Latest with Blackwell support
    cd ../..
else
    echo "CUTLASS already exists in third_party/cutlass"
fi

# Build CUTLASS Blackwell examples
echo "Building CUTLASS Blackwell kernels..."
cd third_party/cutlass

# Create build directory
mkdir -p build
cd build

# Configure for Blackwell (SM100)
cmake .. \
    -DCUTLASS_NVCC_ARCHS="100" \
    -DCUTLASS_ENABLE_TCGEN05=ON \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCUTLASS_EXAMPLE_BLACKWELL_GEMM=ON \
    -DCUTLASS_EXAMPLE_BLACKWELL_GROUPED_GEMM=ON \
    -DCUTLASS_EXAMPLE_BLACKWELL_BLOCKWISE=ON

# Build specific Blackwell examples
echo "Building tcgen05.mma kernels..."
make 70_blackwell_gemm -j8
make 73_blackwell_gemm_preferred -j8
make 75_blackwell_grouped_gemm -j8
make 81_blackwell_gemm_blockwise -j8

echo "✓ CUTLASS Blackwell kernels built successfully!"
echo ""
echo "Available kernels with tcgen05.mma:"
echo "  - 70_blackwell_gemm: Basic Blackwell GEMM"
echo "  - 73_blackwell_gemm_preferred: Optimized GEMM"
echo "  - 75_blackwell_grouped_gemm: MoE grouped GEMM"
echo "  - 81_blackwell_gemm_blockwise: MXFP8/FP4 with microscaling"
echo ""
echo "These kernels use real tcgen05.mma instructions!"
