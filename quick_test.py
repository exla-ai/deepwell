#!/usr/bin/env python3
"""
Quick test after building to verify the kernel works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch

print("Quick Kernel Test")
print("=" * 40)

# Test PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Test our extension
try:
    import deepwell.cutlass_kernels as ck
    print("\n✅ C++ extension loaded!")
except ImportError as e:
    print(f"\n❌ C++ extension not found: {e}")
    print("\nTry building with:")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)

# Test kernel
print("\nTesting GEMM kernel...")
kernel = ck.BlackwellGemmKernel()
kernel.initialize(64, 64, 64, "bf16", False, 32)

# Simple test: 1s * 1s should give 64s
a = torch.ones(64, 64, dtype=torch.bfloat16, device='cuda')
b = torch.ones(64, 64, dtype=torch.bfloat16, device='cuda')
c = kernel.gemm(a, b)

print(f"Input A: all ones, shape {a.shape}")
print(f"Input B: all ones, shape {b.shape}")
print(f"Output C: shape {c.shape}")
print(f"  Min: {c.min():.2f}")
print(f"  Max: {c.max():.2f}")
print(f"  Mean: {c.mean():.2f}")

# Check result
if torch.all(c == 0):
    print("\n❌ FAILED: Output is all zeros!")
    print("The cuBLAS parameters are still wrong.")
elif torch.allclose(c, torch.full_like(c, 64.0), atol=1.0):
    print("\n✅ SUCCESS: Kernel produces correct output!")
    print("Expected 64.0, got ~64.0")
else:
    print(f"\n⚠ WARNING: Unexpected output")
    print(f"Expected all 64.0, got various values")
    print(f"First row: {c[0,:5].tolist()}")

# Also test with random values
print("\n" + "=" * 40)
print("Testing with random matrices...")
torch.manual_seed(42)
a_rand = torch.randn(128, 256, dtype=torch.bfloat16, device='cuda')
b_rand = torch.randn(256, 512, dtype=torch.bfloat16, device='cuda')

kernel2 = ck.BlackwellGemmKernel()
kernel2.initialize(128, 512, 256, "bf16", False, 32)
c_rand = kernel2.gemm(a_rand, b_rand)

# Compare with PyTorch
c_ref = torch.matmul(a_rand, b_rand)
diff = torch.abs(c_rand - c_ref).max().item()

print(f"Random GEMM (128x256 @ 256x512):")
print(f"  Output range: [{c_rand.min():.2f}, {c_rand.max():.2f}]")
print(f"  Max diff vs PyTorch: {diff:.6f}")

if diff < 0.01:
    print("✅ Random GEMM is correct!")
else:
    print(f"❌ Random GEMM has high error: {diff}")

print("\n" + "=" * 40)
print("Test complete!")
