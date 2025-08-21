#!/bin/bash
#
# Build script specifically for B200
# This ensures the C++ extension is properly built
#

set -e

echo "========================================="
echo "Building Deepwell C++ Extension on B200"
echo "========================================="

# 1. Clean everything
echo "Step 1: Cleaning old builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -name "*.so" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 2. Check CUDA
echo ""
echo "Step 2: Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found!"
    exit 1
fi
nvcc --version | head -1

# 3. Build the extension
echo ""
echo "Step 3: Building C++ extension..."
python setup.py build_ext --inplace

# 4. Check if .so was created
echo ""
echo "Step 4: Looking for compiled extension..."
SO_FILE=$(find . -name "cutlass_kernels*.so" 2>/dev/null | head -1)

if [ -z "$SO_FILE" ]; then
    echo "❌ ERROR: No .so file created!"
    echo "Looking for any .so files:"
    find . -name "*.so" 2>/dev/null
    exit 1
else
    echo "✓ Found extension: $SO_FILE"
    ls -la "$SO_FILE"
fi

# 5. Try to import
echo ""
echo "Step 5: Testing import..."
python -c "
try:
    import deepwell.cutlass_kernels
    print('✅ Import successful!')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    # Try direct import
    import sys
    import os
    so_file = '$SO_FILE'
    if os.path.exists(so_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location('cutlass_kernels', so_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print('✓ Direct import worked')
"

# 6. Run verification
echo ""
echo "Step 6: Running verification..."
python verify_fix.py

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
