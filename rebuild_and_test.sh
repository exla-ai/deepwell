#!/bin/bash
# Force clean rebuild and test

set -e

echo "==================================="
echo "FORCE CLEAN REBUILD"
echo "==================================="

# Clean EVERYTHING
echo "1. Cleaning old builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -f src/deepwell/*.so
rm -f deepwell/*.so
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Clear Python cache
echo "2. Clearing Python cache..."
python -c "import sys; sys.path_importer_cache.clear()"

# Rebuild
echo "3. Building extension..."
python setup.py build_ext --inplace --force

# Verify the .so file exists
echo "4. Checking for compiled extension..."
if [ -f "deepwell/cutlass_kernels*.so" ]; then
    echo "✓ Extension found: deepwell/cutlass_kernels*.so"
else
    echo "❌ Extension NOT found!"
    echo "Looking for .so files:"
    find . -name "*.so" 2>/dev/null
fi

# Test import
echo "5. Testing import..."
python -c "import deepwell.cutlass_kernels; print('✓ Import successful')"

# Run verification
echo "6. Running verification..."
python verify_fix.py

echo "==================================="
echo "DONE"
echo "==================================="
