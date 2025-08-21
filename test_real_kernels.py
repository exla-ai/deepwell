#!/usr/bin/env python3
"""
Test script for real kernel dispatch on NVIDIA B200.
This actually executes CUTLASS kernels when available.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import time
import deepwell as dw


class SimpleGEMM(nn.Module):
    """Simple model for testing GEMM kernels."""
    
    def __init__(self, input_dim=4096, hidden_dim=16384, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


def test_cutlass_gemm():
    """Test CUTLASS GEMM kernel directly."""
    print("="*60)
    print("Testing CUTLASS GEMM Kernel on B200")
    print("="*60)
    
    try:
        from deepwell import cutlass_kernels
        print("✓ CUTLASS module loaded")
        
        # Create kernel
        kernel = cutlass_kernels.BlackwellGemmKernel()
        print("✓ Blackwell kernel created")
        
        # Test dimensions
        m, n, k = 4096, 4096, 4096
        
        # Initialize kernel
        kernel.initialize(m, n, k, "bf16", use_microscaling=False)
        print(f"✓ Kernel initialized for {m}x{n}x{k}")
        
        # Create test matrices
        device = torch.device('cuda')
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        print("Benchmarking...")
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate performance
        flops = 2 * m * n * k * iterations
        tflops = flops / elapsed / 1e12
        
        print(f"\nResults:")
        print(f"  Time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Performance: {tflops:.1f} TFLOPS")
        print(f"  Expected on B200: ~1600-2000 TFLOPS")
        
        return True
        
    except Exception as e:
        print(f"✗ CUTLASS test failed: {e}")
        return False


def test_full_pipeline():
    """Test full Deepwell pipeline with real execution."""
    print("\n" + "="*60)
    print("Testing Full Pipeline with Real Kernel Dispatch")
    print("="*60)
    
    # Create model
    model = SimpleGEMM()
    batch_size = 64
    seq_len = 1024
    input_dim = 4096
    
    print(f"Model: SimpleGEMM")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Dim: {input_dim}")
    
    # Probe hardware
    hw = dw.probe()
    print(f"\nHardware: {hw.gpus[0].name if hw.gpus else 'No GPU'}")
    
    # Capture and compile
    print("\nCompiling with Deepwell...")
    engine = dw.optimize_for_blackwell(
        model,
        precision="mxfp8",
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    # Create executable model
    exec_model = dw.create_executable_model(engine, model)
    
    # Check if CUTLASS is active
    print(f"CUTLASS active: {exec_model.use_cutlass}")
    
    # Create input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Benchmark original
    print("\nBenchmarking original model...")
    model = model.to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
        original_time = time.time() - start
    
    original_throughput = batch_size * seq_len * iterations / original_time
    print(f"  Original: {original_throughput/1000:.0f}K tokens/sec")
    
    # Benchmark optimized
    print("\nBenchmarking optimized model...")
    exec_model = exec_model.to(device)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = exec_model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = exec_model(x)
        torch.cuda.synchronize()
        optimized_time = time.time() - start
    
    optimized_throughput = batch_size * seq_len * iterations / optimized_time
    print(f"  Optimized: {optimized_throughput/1000:.0f}K tokens/sec")
    
    # Calculate speedup
    speedup = optimized_throughput / original_throughput
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if exec_model.use_cutlass:
        print("✓ Using real CUTLASS kernels")
    else:
        print("⚠ Using PyTorch fallback (CUTLASS not active)")
        print("  Note: Real speedup requires CUTLASS + quantization")


def main():
    print("🚀 B200 Real Kernel Test")
    print("="*60)
    
    # Test direct CUTLASS
    cutlass_works = test_cutlass_gemm()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n" + "="*60)
    if cutlass_works:
        print("✅ CUTLASS kernels are working!")
        print("Note: Full MXFP8/FP4 speedup requires quantization implementation")
    else:
        print("⚠ CUTLASS not fully working - using PyTorch kernels")
        print("The framework is ready but needs quantization for real speedup")
    print("="*60)


if __name__ == "__main__":
    main()
