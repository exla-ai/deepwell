#!/usr/bin/env python3
"""
Simple test of the optimized engine with a basic MLP model.
This avoids complex transformer models to isolate performance testing.
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, 'src')

import deepwell as dw
from deepwell.optimized_engine import OptimizedBlackwellModel


class SimpleMLP(nn.Module):
    """Simple MLP for testing GEMM performance."""
    
    def __init__(self, hidden_dim=768, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = residual + x
        x = self.norm(x)
        return x


def benchmark_simple(model, name, batch_size, seq_len, hidden_dim, iterations):
    """Benchmark a model."""
    device = next(model.parameters()).device
    
    # Create input embeddings
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    
    # Calculate TFLOPS
    # MLP has 2 GEMMs: (B*S, H) x (H, 4H) and (B*S, 4H) x (4H, H)
    flops_per_gemm1 = 2 * batch_size * seq_len * hidden_dim * (hidden_dim * 4)
    flops_per_gemm2 = 2 * batch_size * seq_len * (hidden_dim * 4) * hidden_dim
    total_flops = (flops_per_gemm1 + flops_per_gemm2) * iterations
    tflops = total_flops / (elapsed * 1e12)
    
    return ms_per_iter, tflops


def main():
    print("=" * 70)
    print("SIMPLE OPTIMIZED BENCHMARK")
    print("=" * 70)
    
    # Parameters
    batch_size = 64
    seq_len = 1024
    hidden_dim = 1024
    iterations = 100
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Iterations: {iterations}")
    
    # Detect hardware
    hw = dw.probe()
    print("\n" + "=" * 70)
    print("HARDWARE")
    print("=" * 70)
    for gpu in hw.gpus:
        print(f"GPU: {gpu.name}")
        if gpu.is_blackwell:
            print(f"  ✓ Blackwell {gpu.blackwell_variant} detected!")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. PyTorch baseline
    print("\n" + "=" * 70)
    print("1. PyTorch Baseline")
    print("=" * 70)
    
    pytorch_model = SimpleMLP(hidden_dim).to(device)
    pytorch_ms, pytorch_tflops = benchmark_simple(
        pytorch_model, "PyTorch", batch_size, seq_len, hidden_dim, iterations
    )
    
    print(f"  Time: {pytorch_ms:.2f} ms/iter")
    print(f"  Performance: {pytorch_tflops:.2f} TFLOPS")
    
    # 2. torch.compile
    print("\n" + "=" * 70)
    print("2. torch.compile")
    print("=" * 70)
    
    compiled_model = torch.compile(SimpleMLP(hidden_dim).to(device), mode="max-autotune")
    
    # Warmup compilation
    dummy = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    with torch.no_grad():
        for i in range(3):
            _ = compiled_model(dummy)
            print(f"  Compilation pass {i+1}/3")
    
    compiled_ms, compiled_tflops = benchmark_simple(
        compiled_model, "torch.compile", batch_size, seq_len, hidden_dim, iterations
    )
    
    print(f"  Time: {compiled_ms:.2f} ms/iter")
    print(f"  Performance: {compiled_tflops:.2f} TFLOPS")
    print(f"  Speedup: {pytorch_ms/compiled_ms:.2f}x")
    
    # 3. Optimized with CUTLASS
    print("\n" + "=" * 70)
    print("3. Optimized with CUTLASS")
    print("=" * 70)
    
    opt_model = OptimizedBlackwellModel(SimpleMLP(hidden_dim).to(device), "bf16")
    
    # Warmup
    with torch.no_grad():
        for i in range(3):
            _ = opt_model(dummy)
            print(f"  Warmup {i+1}/3")
    
    opt_ms, opt_tflops = benchmark_simple(
        opt_model, "Optimized", batch_size, seq_len, hidden_dim, iterations
    )
    
    print(f"  Time: {opt_ms:.2f} ms/iter")
    print(f"  Performance: {opt_tflops:.2f} TFLOPS")
    print(f"  Speedup vs PyTorch: {pytorch_ms/opt_ms:.2f}x")
    print(f"  Speedup vs compile: {compiled_ms/opt_ms:.2f}x")
    
    # 4. Direct CUTLASS kernel test
    print("\n" + "=" * 70)
    print("4. Direct CUTLASS Kernel")
    print("=" * 70)
    
    try:
        from deepwell import cutlass_kernels
        
        kernel = cutlass_kernels.BlackwellGemmKernel()
        kernel.initialize(
            batch_size * seq_len,  # M
            hidden_dim * 4,         # N
            hidden_dim,             # K
            "bf16",
            False,
            32
        )
        
        # Test matrices
        a = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
        b = torch.randn(hidden_dim, hidden_dim * 4, device=device, dtype=torch.bfloat16)
        
        # Warmup
        for _ in range(10):
            _ = kernel.gemm(a, b)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            _ = kernel.gemm(a, b)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        kernel_ms = (elapsed / iterations) * 1000
        kernel_flops = 2 * batch_size * seq_len * hidden_dim * hidden_dim * 4 * iterations
        kernel_tflops = kernel_flops / (elapsed * 1e12)
        
        print(f"  Time: {kernel_ms:.2f} ms/iter")
        print(f"  Performance: {kernel_tflops:.2f} TFLOPS")
        
    except Exception as e:
        print(f"  Could not test direct kernel: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'TFLOPS':<12} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'PyTorch':<20} {pytorch_ms:<12.2f} {pytorch_tflops:<12.2f} {'1.00x':<10}")
    print(f"{'torch.compile':<20} {compiled_ms:<12.2f} {compiled_tflops:<12.2f} {pytorch_ms/compiled_ms:<10.2f}x")
    print(f"{'Optimized':<20} {opt_ms:<12.2f} {opt_tflops:<12.2f} {pytorch_ms/opt_ms:<10.2f}x")
    
    if 'kernel_ms' in locals():
        print(f"{'Direct CUTLASS':<20} {kernel_ms:<12.2f} {kernel_tflops:<12.2f} {pytorch_ms/kernel_ms:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("B200 Expected Performance:")
    print("  BF16: ~2,500 TFLOPS")
    print("  MXFP8: ~5,000 TFLOPS")
    print(f"  Current best: {max(pytorch_tflops, compiled_tflops, opt_tflops):.2f} TFLOPS")


if __name__ == "__main__":
    main()
