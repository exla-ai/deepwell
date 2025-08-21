#!/usr/bin/env python3
"""
Debug script to figure out why the kernel outputs zeros.
"""

import torch
import sys

def debug_kernel():
    """Debug the kernel step by step."""
    
    print("=" * 60)
    print("DEBUGGING KERNEL")
    print("=" * 60)
    
    # Tiny test case
    M, N, K = 4, 4, 4
    
    print(f"\nTest size: {M}x{N}x{K}")
    
    # Create simple test matrices
    A = torch.ones(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.ones(K, N, dtype=torch.bfloat16, device='cuda')
    
    print(f"A: all ones, shape {A.shape}")
    print(f"B: all ones, shape {B.shape}")
    
    # Expected: all elements should be K (4.0)
    expected = torch.full((M, N), K, dtype=torch.bfloat16, device='cuda')
    print(f"Expected: all {K}")
    
    # Test our kernel
    try:
        from deepwell import cutlass_kernels
        
        print("\n1. Creating kernel...")
        kernel = cutlass_kernels.BlackwellGemmKernel()
        
        print("2. Initializing kernel...")
        kernel.initialize(M, N, K, "bf16", False, 32)
        
        print("3. Running GEMM...")
        result = kernel.gemm(A, B)
        
        print(f"\nResult:")
        print(f"  Shape: {result.shape}")
        print(f"  Dtype: {result.dtype}")
        print(f"  Device: {result.device}")
        print(f"  Min: {result.min().item()}")
        print(f"  Max: {result.max().item()}")
        print(f"  Mean: {result.mean().item()}")
        print(f"  All zeros? {torch.all(result == 0).item()}")
        
        if torch.all(result == 0):
            print("\n❌ OUTPUT IS ALL ZEROS!")
            print("This means:")
            print("  1. The kernel isn't executing, OR")
            print("  2. The cuBLAS parameters are wrong, OR")
            print("  3. The output buffer isn't being written")
        else:
            print(f"\n✓ Got non-zero output!")
            print(f"  First row: {result[0].tolist()}")
            
            # Check correctness
            diff = torch.abs(result - expected).max().item()
            if diff < 0.01:
                print(f"✅ Result is correct! (max diff: {diff})")
            else:
                print(f"⚠ Result is wrong (max diff: {diff})")
                print(f"  Expected: {expected[0].tolist()}")
                print(f"  Got: {result[0].tolist()}")
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_kernel()
