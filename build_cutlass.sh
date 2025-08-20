#!/bin/bash
# Build script for CUTLASS C++ extensions

echo "========================================="
echo "Building Deepwell CUTLASS Extensions"
echo "========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "CUDA Version: $CUDA_VERSION"

# Check for CUTLASS (optional - will use bundled if not found)
if [ -z "$CUTLASS_PATH" ]; then
    echo "CUTLASS_PATH not set. Checking default locations..."
    
    if [ -d "/usr/local/cutlass" ]; then
        export CUTLASS_PATH="/usr/local/cutlass"
    elif [ -d "$HOME/cutlass" ]; then
        export CUTLASS_PATH="$HOME/cutlass"
    else
        echo "WARNING: CUTLASS not found. Will attempt to download..."
        
        # Clone CUTLASS if not present
        if [ ! -d "third_party/cutlass" ]; then
            echo "Cloning CUTLASS from GitHub..."
            mkdir -p third_party
            cd third_party
            git clone https://github.com/NVIDIA/cutlass.git
            cd ..
        fi
        export CUTLASS_PATH="$(pwd)/third_party/cutlass"
    fi
fi

echo "CUTLASS Path: $CUTLASS_PATH"

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected GPU Architecture: SM$GPU_ARCH"

# Check for Blackwell
if [ "$GPU_ARCH" -ge "100" ]; then
    echo "✓ Blackwell GPU detected (SM$GPU_ARCH)"
    echo "  Enabling MXFP8 and NVFP4 support"
elif [ "$GPU_ARCH" -ge "90" ]; then
    echo "✓ Hopper GPU detected (SM$GPU_ARCH)"
    echo "  Enabling FP8 support"
else
    echo "⚠ Non-Blackwell GPU detected (SM$GPU_ARCH)"
    echo "  Some optimizations may not be available"
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete

# Build the extension
echo ""
echo "Building C++ extension..."
python setup.py build_ext --inplace

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    
    # Test import
    echo ""
    echo "Testing import..."
    python -c "from deepwell import cutlass_kernels; print('✓ CUTLASS extension loaded successfully')" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "Build Complete - Ready for B200 Testing!"
        echo "========================================="
    else
        echo "⚠ Warning: Extension built but import failed"
        echo "This might be due to missing dependencies"
    fi
else
    echo ""
    echo "✗ Build failed"
    echo "Please check the error messages above"
    exit 1
fi
