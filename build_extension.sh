#!/bin/bash
#
# Correctly build the C++ extension for src/deepwell structure
#

set -e

echo "========================================="
echo "Building Deepwell C++ Extension"
echo "========================================="

# 1. Clean everything
echo "1. Cleaning..."
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# 2. Build extension
echo ""
echo "2. Building C++ extension..."
python setup.py build_ext --inplace

# 3. Find the .so file
echo ""
echo "3. Looking for .so file..."
SO_FILE=$(find . -name "cutlass_kernels*.so" 2>/dev/null | head -1)

if [ -z "$SO_FILE" ]; then
    echo "❌ No .so file found!"
    echo "All .so files:"
    find . -name "*.so" 2>/dev/null
else
    echo "✓ Found: $SO_FILE"
    
    # 4. Copy to correct location
    echo ""
    echo "4. Installing to src/deepwell/..."
    cp "$SO_FILE" src/deepwell/
    echo "✓ Copied to src/deepwell/"
    
    # Verify
    ls -la src/deepwell/*.so
fi

# 5. Test import
echo ""
echo "5. Testing import..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    import deepwell.cutlass_kernels
    print('✅ Import successful!')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
