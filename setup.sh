#!/bin/bash
#
# Setup script for Deepwell framework
#

set -e

echo "=========================================="
echo "Deepwell Setup"
echo "=========================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e .

# Build C++ extensions
echo "Building C++ extensions..."
python setup.py build_ext --inplace

# Optional: Install CUTLASS Python API for additional features
echo ""
echo "Optional: Install CUTLASS Python API for tcgen05.mma support"
echo "Run: pip install nvidia-cutlass"

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
echo ""
echo "Test installation with:"
echo "  python test.py"
echo ""
echo "Run benchmarks with:"
echo "  python benchmarks/benchmark.py"
