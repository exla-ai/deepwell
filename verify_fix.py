#!/usr/bin/env python3
"""
Quick verification that the GEMM fix works.
Run this after rebuilding to check correctness.
"""

import torch
import sys

def test_gemm():
    """Test if GEMM produces correct output (not zeros)."""
    print("Testing GEMM correctness...")
    
    # Small test case
    M, N, K = 128, 128, 128
    
    # Create test matrices
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    
    # PyTorch reference
    ref = torch.matmul(a, b)
    
    # Deepwell
    try:
        from deepwell import cutlass_kernels
        
        kernel = cutlass_kernels.BlackwellGemmKernel()
        kernel.initialize(M, N, K, "bf16", False, 32)
        
        out = kernel.gemm(a, b)
        
        # Check if output is not all zeros
        if torch.all(out == 0):
            print("❌ FAILED: Output is all zeros!")
            print(f"  Expected range: [{ref.min():.3f}, {ref.max():.3f}]")
            print(f"  Got: all zeros")
            return False
        
        # Check correctness
        diff = torch.abs(ref - out).max().item()
        rel_error = (torch.abs(ref - out) / (torch.abs(ref) + 1e-6)).max().item()
        
        print(f"  Max difference: {diff:.6f}")
        print(f"  Relative error: {rel_error:.6f}")
        print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
        
        if diff < 0.01 and rel_error < 0.01:
            print("✅ PASSED: GEMM is correct!")
            return True
        else:
            print(f"❌ FAILED: Difference too large")
            print(f"  First 5 ref:  {ref.flatten()[:5].tolist()}")
            print(f"  First 5 out:  {out.flatten()[:5].tolist()}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_gemm():
        sys.exit(0)
    else:
        sys.exit(1)
