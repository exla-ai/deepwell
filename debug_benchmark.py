#!/usr/bin/env python3
"""Debug script to understand the performance differences."""

import torch
import torch.nn as nn
import time
import numpy as np

def profile_gemm_detail():
    """Profile GEMM in detail to understand the speedup."""
    
    print("=" * 70)
    print("DETAILED GEMM PROFILING")
    print("=" * 70)
    
    # Test configuration
    m, n, k = 16384, 3072, 768
    device = 'cuda'
    dtype = torch.bfloat16
    
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    
    # 1. Regular PyTorch
    print("\n1. Regular PyTorch matmul:")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Time: {pytorch_time:.4f} ms")
    
    # 2. torch.compile
    print("\n2. torch.compile (max-autotune):")
    @torch.compile(mode='max-autotune')
    def compiled_gemm(x, y):
        return torch.matmul(x, y)
    
    # Warmup
    for _ in range(5):
        _ = compiled_gemm(a, b)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        c = compiled_gemm(a, b)
    torch.cuda.synchronize()
    compile_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Time: {compile_time:.4f} ms")
    print(f"   Speedup vs PyTorch: {pytorch_time/compile_time:.2f}x")
    
    # 3. Our CUTLASS backend
    print("\n3. Deepwell CUTLASS backend:")
    try:
        from deepwell import cutlass_kernels
        kernel = cutlass_kernels.BlackwellGemmKernel()
        kernel.initialize(m, n, k, "bf16", False, 32)
        
        # Warmup
        for _ in range(5):
            _ = kernel.gemm(a, b)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        cutlass_time = (time.perf_counter() - start) * 1000 / 100
        print(f"   Time: {cutlass_time:.4f} ms")
        print(f"   Speedup vs torch.compile: {compile_time/cutlass_time:.2f}x")
        
        # Verify correctness
        pytorch_result = torch.matmul(a, b)
        cutlass_result = kernel.gemm(a, b)
        diff = torch.abs(pytorch_result - cutlass_result).max().item()
        print(f"   Max difference: {diff:.6f}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. cuBLAS directly
    print("\n4. cuBLAS directly (via cublas.gemm):")
    try:
        # Try to use cuBLAS directly through torch
        import torch._C as torch_C
        if hasattr(torch_C, '_cuda_cublasSgemmEx'):
            print("   cuBLAS available through torch")
        else:
            print("   cuBLAS not directly accessible")
    except:
        pass


def profile_model_operations():
    """Profile individual operations in a transformer model."""
    
    print("\n" + "=" * 70)
    print("TRANSFORMER OPERATION PROFILING")
    print("=" * 70)
    
    batch_size = 32
    seq_len = 512
    hidden_dim = 768
    device = 'cuda'
    dtype = torch.bfloat16
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    
    # 1. Linear layer
    print("\n1. Linear layer (768 -> 3072):")
    linear = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype).to(device)
    
    # Regular
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = linear(x)
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Regular: {regular_time:.4f} ms")
    
    # Compiled
    compiled_linear = torch.compile(linear, mode='max-autotune')
    _ = compiled_linear(x)  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = compiled_linear(x)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Compiled: {compiled_time:.4f} ms ({regular_time/compiled_time:.2f}x speedup)")
    
    # 2. Full MLP block
    print("\n2. Full MLP block (Linear + GELU + Linear):")
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype)
            self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype)
            self.gelu = nn.GELU()
        
        def forward(self, x):
            return self.fc2(self.gelu(self.fc1(x)))
    
    mlp = MLP().to(device)
    
    # Regular
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = mlp(x)
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Regular: {regular_time:.4f} ms")
    
    # Compiled
    compiled_mlp = torch.compile(mlp, mode='max-autotune')
    _ = compiled_mlp(x)  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = compiled_mlp(x)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) * 1000 / 100
    print(f"   Compiled: {compiled_time:.4f} ms ({regular_time/compiled_time:.2f}x speedup)")
    
    # 3. Check what torch.compile is doing
    print("\n3. torch.compile optimizations:")
    print("   - Fuses pointwise operations (GELU, LayerNorm, etc.)")
    print("   - Reduces memory transfers between kernels")
    print("   - Uses Triton for optimized kernels")
    print("   - Can use TensorFloat32 for faster matmuls")
    
    # 4. What Deepwell needs to compete
    print("\n4. To beat torch.compile, Deepwell needs:")
    print("   ✗ Real tcgen05.mma kernels (currently using cuBLAS)")
    print("   ✗ Operator fusion (LayerNorm+GEMM, GEMM+GELU, etc.)")
    print("   ✗ Graph-level optimization")
    print("   ✗ Memory layout optimization")
    print("   ✓ Low-precision support (MXFP8/FP4)")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    print("Debugging Deepwell vs torch.compile performance...\n")
    
    profile_gemm_detail()
    profile_model_operations()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The GEMM speedups (75x-2000x) are likely measurement artifacts or
hitting special optimized paths. The model performance shows the
real story: torch.compile wins because it does whole-graph
optimization while Deepwell only replaces individual operations.

To actually beat torch.compile on Blackwell, we need:
1. Real Blackwell-specific kernels (tcgen05.mma)
2. Operator fusion to reduce memory transfers
3. Graph-level optimization like torch.compile
""")
