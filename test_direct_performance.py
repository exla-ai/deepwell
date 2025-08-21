#!/usr/bin/env python3
"""
Direct performance test to isolate kernel performance from framework overhead.
"""

import torch
import time
import sys
import os

sys.path.insert(0, 'src')

def test_direct_performance():
    """Test raw kernel performance without framework overhead."""
    
    print("=" * 60)
    print("Direct Kernel Performance Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("No CUDA device available")
        return
    
    device = torch.device("cuda")
    
    # Test dimensions
    m, n, k = 1024, 1024, 1024
    iterations = 100
    
    # Create test tensors
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    
    # 1. Baseline PyTorch
    print("\n1. PyTorch Baseline (BF16):")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / iterations * 1000
    print(f"   Time per GEMM: {pytorch_time:.2f} ms")
    
    # Calculate TFLOPS
    flops = 2 * m * n * k  # 2 ops per multiply-add
    tflops = flops / (pytorch_time / 1000) / 1e12
    print(f"   Performance: {tflops:.2f} TFLOPS")
    
    # 2. torch.compile baseline
    print("\n2. torch.compile Baseline:")
    compiled_matmul = torch.compile(torch.matmul)
    
    # Warmup
    for _ in range(10):
        c = compiled_matmul(a, b)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        c = compiled_matmul(a, b)
    torch.cuda.synchronize()
    compiled_time = (time.time() - start) / iterations * 1000
    print(f"   Time per GEMM: {compiled_time:.2f} ms")
    
    tflops_compiled = flops / (compiled_time / 1000) / 1e12
    print(f"   Performance: {tflops_compiled:.2f} TFLOPS")
    print(f"   Speedup vs PyTorch: {pytorch_time/compiled_time:.2f}x")
    
    # 3. Test our CUTLASS kernel directly (if available)
    try:
        from deepwell import cutlass_kernels
        
        print("\n3. CUTLASS C++ Extension (cuBLAS backend):")
        
        # Create kernel
        kernel = cutlass_kernels.BlackwellGemmKernel()
        kernel.initialize(m, n, k, "bf16", False, 32)
        
        # Warmup
        for _ in range(10):
            c = kernel.gemm(a, b)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        cutlass_time = (time.time() - start) / iterations * 1000
        print(f"   Time per GEMM: {cutlass_time:.2f} ms")
        
        tflops_cutlass = flops / (cutlass_time / 1000) / 1e12
        print(f"   Performance: {tflops_cutlass:.2f} TFLOPS")
        print(f"   Speedup vs PyTorch: {pytorch_time/cutlass_time:.2f}x")
        print(f"   Speedup vs compiled: {compiled_time/cutlass_time:.2f}x")
        
    except ImportError:
        print("\n3. CUTLASS extension not available")
    except Exception as e:
        print(f"\n3. CUTLASS test failed: {e}")
    
    # 4. Test small sizes (like in benchmark)
    print("\n4. Small Matrix Test (256x256x256):")
    m, n, k = 256, 256, 256
    
    a_small = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_small = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    
    # PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        c = torch.matmul(a_small, b_small)
    torch.cuda.synchronize()
    small_time = (time.time() - start) / 1000 * 1000
    print(f"   PyTorch: {small_time:.3f} ms")
    
    # Compiled
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        c = compiled_matmul(a_small, b_small)
    torch.cuda.synchronize()
    small_compiled = (time.time() - start) / 1000 * 1000
    print(f"   Compiled: {small_compiled:.3f} ms")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Large matrices (1024x1024):")
    print(f"  PyTorch:      {pytorch_time:.2f} ms ({tflops:.2f} TFLOPS)")
    print(f"  torch.compile: {compiled_time:.2f} ms ({tflops_compiled:.2f} TFLOPS)")
    
    print(f"\nSmall matrices (256x256):")
    print(f"  PyTorch:      {small_time:.3f} ms")
    print(f"  torch.compile: {small_compiled:.3f} ms")
    
    # Expected B200 performance
    print("\n" + "=" * 60)
    print("Expected B200 Performance:")
    print("=" * 60)
    print("BF16 peak: ~2,500 TFLOPS")
    print("MXFP8 peak: ~5,000 TFLOPS (2x BF16)")
    print("FP4 peak: ~10,000 TFLOPS (4x BF16)")
    
    if tflops_compiled < 100:
        print("\n⚠️  Performance is much lower than expected!")
        print("   This suggests the GPU might not be fully utilized")
        print("   or there's significant overhead in the execution path.")


if __name__ == "__main__":
    test_direct_performance()
