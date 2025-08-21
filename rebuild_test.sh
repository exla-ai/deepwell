#!/bin/bash
echo "Rebuilding CUTLASS extension with fixes..."
cd /root/deepwell

# Clean previous builds
rm -rf build/
rm -f src/deepwell/*.so

# Build
python setup.py build_ext --inplace

# Test
echo ""
echo "Testing fixed kernels..."
python test_kernel_dispatch.py
