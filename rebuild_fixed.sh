#!/bin/bash
echo "========================================"
echo "Rebuilding with fixed kernels..."
echo "========================================"

# Clean
echo "Cleaning previous builds..."
rm -rf build/
rm -f src/deepwell/*.so

# Build
echo "Building..."
python setup.py build_ext --inplace 2>&1 | tee build.log

# Check if build succeeded
if [ -f "src/deepwell/cutlass_kernels.cpython-311-x86_64-linux-gnu.so" ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Testing kernels..."
    python test_kernel_dispatch.py
else
    echo ""
    echo "❌ Build failed. Check build.log for errors."
    tail -20 build.log
fi
