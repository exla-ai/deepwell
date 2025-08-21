#!/bin/bash
#
# CORRECT build script for src/deepwell structure
#

set -e

echo "========================================="
echo "Building Deepwell on B200"
echo "========================================="

# 1. Clean
echo "Cleaning..."
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# 2. Build
echo ""
echo "Building C++ extension..."
python setup.py build_ext --inplace

# 3. The .so should be in src/deepwell/
echo ""
echo "Checking for extension..."
if [ -f "src/deepwell/cutlass_kernels.cpython"*".so" ]; then
    echo "✅ Extension built successfully:"
    ls -la src/deepwell/*.so
else
    echo "❌ Extension not found in src/deepwell/"
    echo "Looking for .so files:"
    find . -name "*.so" 2>/dev/null
    
    # Try to copy from build directory
    SO_FILE=$(find build -name "cutlass_kernels*.so" 2>/dev/null | head -1)
    if [ ! -z "$SO_FILE" ]; then
        echo "Found in build directory, copying..."
        cp "$SO_FILE" src/deepwell/
        echo "✓ Copied to src/deepwell/"
    fi
fi

# 4. Test
echo ""
echo "Testing import..."
cd /root/deepwell  # Use absolute path
export PYTHONPATH=/root/deepwell/src:$PYTHONPATH

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name()}')

try:
    import deepwell.cutlass_kernels as ck
    print('✅ C++ extension imported!')
    
    # Test kernel
    kernel = ck.BlackwellGemmKernel()
    kernel.initialize(64, 64, 64, 'bf16', False, 32)
    
    a = torch.ones(64, 64, dtype=torch.bfloat16, device='cuda')
    b = torch.ones(64, 64, dtype=torch.bfloat16, device='cuda')
    c = kernel.gemm(a, b)
    
    print(f'GEMM result: min={c.min():.2f}, max={c.max():.2f}, mean={c.mean():.2f}')
    
    if torch.all(c == 0):
        print('❌ STILL OUTPUTTING ZEROS!')
    elif torch.allclose(c, torch.full_like(c, 64.0), atol=1.0):
        print('✅ KERNEL WORKS! (expected ~64.0)')
    else:
        print(f'⚠ Unexpected result (expected 64.0)')
        
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
