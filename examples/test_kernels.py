#!/usr/bin/env python3
"""
Example: Testing CUTLASS kernel performance with Deepwell.

This example shows:
1. Direct kernel testing
2. Performance comparison
3. Mixed precision operations
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import deepwell as dw


def test_cutlass_gemm():
    """Test CUTLASS GEMM kernels directly."""
    print("\n" + "=" * 70)
    print("CUTLASS GEMM KERNEL TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different matrix sizes
    test_configs = [
        {"name": "Small", "m": 1024, "n": 1024, "k": 1024},
        {"name": "Medium", "m": 4096, "n": 4096, "k": 1024},
        {"name": "Large", "m": 8192, "n": 8192, "k": 2048},
    ]
    
    for config in test_configs:
        print(f"\n{config['name']} GEMM ({config['m']}×{config['n']}×{config['k']}):")
        print("-" * 40)
        
        m, n, k = config['m'], config['n'], config['k']
        
        # Create test matrices
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        
        # PyTorch baseline
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            c_pytorch = torch.matmul(a, b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start
        
        # CUTLASS kernel
        try:
            from deepwell.kernels.cutlass_bindings import CutlassKernel

            kernel = CutlassKernel()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                c_cutlass = kernel.gemm(a, b)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cutlass_time = time.perf_counter() - start
            
            # Calculate metrics
            speedup = pytorch_time / cutlass_time
            flops = 2 * m * n * k * 100
            cutlass_tflops = flops / (cutlass_time * 1e12)
            
            print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
            print(f"  CUTLASS time: {cutlass_time*1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Performance: {cutlass_tflops:.2f} TFLOPS")
            
            # Verify correctness
            if torch.allclose(c_pytorch, c_cutlass, rtol=1e-2, atol=1e-2):
                print("  ✓ Results match")
            else:
                print("  ⚠ Results differ (may be due to precision)")
                
        except ImportError:
            print("  CUTLASS not available")


def test_model_optimization():
    """Test model-level optimization."""
    print("\n" + "=" * 70)
    print("MODEL OPTIMIZATION TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a simple MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=3072):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
        
        def forward(self, x):
            return self.layers(x)
    
    print("\nCreating MLP model...")
    model = MLP().to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Optimize with Deepwell
    print("\nOptimizing with Deepwell...")
    optimized = dw.optimize_for_blackwell(model)
    
    # Test input
    batch_size = 64
    seq_len = 512
    input_dim = 768
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            _ = optimized(x)

    # Benchmark
    iterations = 100

    # Baseline
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_time = time.perf_counter() - start

    # Optimized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = optimized(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    optimized_time = time.perf_counter() - start
    
    # Results
    speedup = baseline_time / optimized_time
    baseline_throughput = batch_size * seq_len * iterations / baseline_time
    optimized_throughput = batch_size * seq_len * iterations / optimized_time
    
    print(f"\nResults:")
    print(f"  Baseline: {baseline_time*1000:.2f} ms")
    print(f"  Optimized: {optimized_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Baseline throughput: {baseline_throughput:.0f} tokens/sec")
    print(f"  Optimized throughput: {optimized_throughput:.0f} tokens/sec")


def test_mixed_precision():
    """Test mixed precision operations."""
    print("\n" + "=" * 70)
    print("MIXED PRECISION TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test quantization/dequantization
    try:
        from deepwell import cutlass_kernels
        
        print("\nTesting MXFP8 quantization...")
        
        # Create test tensor
        x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        
        # Quantize to MXFP8
        manager = cutlass_kernels.MicroscaleManager()
        x_quant, scales = manager.quantize_mxfp8(x)
        
        print(f"  Original shape: {x.shape}")
        print(f"  Quantized shape: {x_quant.shape}")
        print(f"  Scales shape: {scales.shape}")
        
        # Dequantize
        x_dequant = manager.dequantize_mxfp8(x_quant, scales)
        
        # Check error
        error = torch.abs(x - x_dequant).mean()
        max_error = torch.abs(x - x_dequant).max()
        
        print(f"  Mean error: {error:.6f}")
        print(f"  Max error: {max_error:.6f}")
        
        if error < 0.01:
            print("  ✓ Quantization working correctly")
        else:
            print("  ⚠ High quantization error")
            
    except Exception as e:
        print(f"  Mixed precision test failed: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("DEEPWELL KERNEL TESTING".center(70))
    print("=" * 70)
    
    # Hardware detection
    hw = dw.probe()
    print("\nHardware:")
    for gpu in hw.gpus:
        print(f"  GPU: {gpu.name}")
        if gpu.is_blackwell:
            print(f"    ✓ Blackwell {gpu.blackwell_variant}")
            print(f"    MXFP8: {gpu.supports_mxfp8}")
            print(f"    FP4: {gpu.supports_fp4}")
    
    # Run tests
    test_cutlass_gemm()
    test_model_optimization()
    test_mixed_precision()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
