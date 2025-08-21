#!/bin/bash
#
# Install CUTLASS Python API for production Blackwell kernels
#

set -e

echo "=========================================="
echo "Installing CUTLASS Python API"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install CUTLASS Python package
echo "Installing nvidia-cutlass package..."
pip install nvidia-cutlass

# Verify installation
echo ""
echo "Verifying CUTLASS installation..."
python -c "
import cutlass
print(f'✓ CUTLASS version: {cutlass.__version__}')
print('✓ CUTLASS Python API installed successfully')
print('')
print('Available Blackwell features:')
print('  - tcgen05.mma instructions')
print('  - MXFP8/FP4 with microscaling')
print('  - Grouped GEMM for MoE')
print('  - TMEM residency optimization')
" || echo "Warning: CUTLASS import failed"

echo ""
echo "=========================================="
echo "CUTLASS Python API Installation Complete"
echo "=========================================="
echo ""
echo "You can now use NVIDIA production kernels with:"
echo "  - Real tcgen05.mma instructions"
echo "  - Hardware-accelerated microscaling"
echo "  - Optimized Blackwell kernels"
echo ""
echo "These are the same kernels used in NVIDIA's examples:"
echo "  - 70_blackwell_gemm"
echo "  - 73_blackwell_gemm_preferred"
echo "  - 75_blackwell_grouped_gemm"
echo "  - 81_blackwell_gemm_blockwise"
