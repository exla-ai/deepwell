#!/usr/bin/env python3
"""
Direct test of cuBLAS to verify the parameters are correct.
This bypasses our wrapper to test cuBLAS directly.
"""

import torch
import numpy as np

def test_cublas_directly():
    """Test cuBLAS directly via PyTorch's low-level API."""
    print("Testing cuBLAS directly through PyTorch...")
    
    M, N, K = 64, 64, 64
    
    # Create test matrices
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    
    # PyTorch matmul (this uses cuBLAS internally)
    C_ref = torch.matmul(A, B)
    
    print(f"PyTorch result range: [{C_ref.min():.3f}, {C_ref.max():.3f}]")
    print(f"PyTorch result mean: {C_ref.mean():.3f}")
    
    # Now test with BF16
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    C_bf16 = torch.matmul(A_bf16, B_bf16)
    
    print(f"BF16 result range: [{C_bf16.min():.3f}, {C_bf16.max():.3f}]")
    print(f"BF16 result mean: {C_bf16.mean():.3f}")
    
    # Compare
    diff = torch.abs(C_ref - C_bf16.float()).max().item()
    print(f"Max difference FP32 vs BF16: {diff:.6f}")
    
    # This proves that PyTorch's cuBLAS works
    # If our kernel outputs zeros, our parameters are wrong
    
    return C_ref, C_bf16

if __name__ == "__main__":
    test_cublas_directly()
