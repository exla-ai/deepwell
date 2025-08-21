#!/usr/bin/env python3
"""
Real benchmark that properly measures Blackwell performance.
This ensures warmup is excluded and kernels are properly launched.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os

sys.path.insert(0, 'src')

import deepwell as dw


class RealBenchmark:
    """Professional benchmark harness that does things correctly."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.results = {}
    
    def warmup(self, model, input_data, iterations=10):
        """Proper warmup that's NOT measured."""
        print("    Warming up...", end='', flush=True)
        with torch.no_grad():
            for i in range(iterations):
                _ = model(input_data)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                if i % 2 == 0:
                    print('.', end='', flush=True)
        print(" done!")
    
    def measure(self, model, input_data, iterations=100, name="Model"):
        """Measure performance AFTER warmup."""
        # CRITICAL: Warmup first (not measured!)
        self.warmup(model, input_data)
        
        # Now measure real performance
        print(f"    Measuring {name}...", end='', flush=True)
        
        # Use CUDA events for accurate timing
        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_data)
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_s = elapsed_ms / 1000.0
        else:
            # CPU timing
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_data)
            elapsed_s = time.perf_counter() - start
        
        ms_per_iter = (elapsed_s * 1000) / iterations
        print(f" {ms_per_iter:.2f} ms/iter")
        
        return {
            'total_time_s': elapsed_s,
            'ms_per_iter': ms_per_iter,
            'iterations': iterations,
            'throughput_per_sec': iterations / elapsed_s
        }
    
    def compare(self, baseline_ms, optimized_ms):
        """Calculate speedup correctly."""
        speedup = baseline_ms / optimized_ms
        return speedup


class OptimizedGEMM(nn.Module):
    """Direct GEMM using our CUTLASS kernels."""
    
    def __init__(self, m, n, k, use_cutlass=True):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.use_cutlass = use_cutlass
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(n, k))
        
        if use_cutlass:
            try:
                from deepwell import cutlass_kernels
                self.kernel = cutlass_kernels.BlackwellGemmKernel()
                self.kernel.initialize(m, n, k, "bf16", False, 32)
                print(f"      ✓ CUTLASS kernel initialized for {m}x{n}x{k}")
            except Exception as e:
                print(f"      ⚠ CUTLASS init failed: {e}")
                self.use_cutlass = False
    
    def forward(self, x):
        if self.use_cutlass and hasattr(self, 'kernel'):
            # Use CUTLASS kernel directly
            x_2d = x.view(-1, self.k)
            output = self.kernel.gemm(
                x_2d.to(torch.bfloat16),
                self.weight.t().contiguous().to(torch.bfloat16)
            )
            return output.view(x.shape[0], x.shape[1], -1)
        else:
            # Fallback to PyTorch
            return torch.matmul(x, self.weight.t())


def create_transformer_block(hidden_dim=768, num_heads=12):
    """Create a realistic transformer block."""
    
    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
        
        def forward(self, x):
            # Self-attention
            residual = x
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = residual + x
            
            # MLP
            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + x
            
            return x
    
    return TransformerBlock()


def main():
    print("=" * 80)
    print("REAL BLACKWELL BENCHMARK")
    print("=" * 80)
    
    # Detect hardware
    hw = dw.probe()
    has_blackwell = False
    for gpu in hw.gpus:
        print(f"GPU: {gpu.name}")
        if gpu.is_blackwell:
            print(f"  ✓ Blackwell {gpu.blackwell_variant} detected!")
            print(f"  MXFP8: {gpu.supports_mxfp8}")
            print(f"  FP4: {gpu.supports_fp4}")
            has_blackwell = True
    
    if not has_blackwell:
        print("\n⚠ WARNING: No Blackwell GPU detected!")
    
    # Create benchmark harness
    bench = RealBenchmark()
    
    # Test configurations
    configs = [
        {"name": "Small", "batch": 32, "seq": 512, "hidden": 768},
        {"name": "Medium", "batch": 64, "seq": 1024, "hidden": 1024},
        {"name": "Large", "batch": 128, "seq": 2048, "hidden": 1280},
    ]
    
    for config in configs:
        print(f"\n" + "=" * 80)
        print(f"TEST: {config['name']} ({config['batch']}x{config['seq']}x{config['hidden']})")
        print("=" * 80)
        
        batch_size = config['batch']
        seq_len = config['seq']
        hidden_dim = config['hidden']
        
        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=bench.device)
        
        # 1. Test raw GEMM performance
        print("\n1. RAW GEMM PERFORMANCE")
        print("-" * 40)
        
        # Large GEMM (like in MLP)
        m = batch_size * seq_len
        n = hidden_dim * 4
        k = hidden_dim
        
        # PyTorch GEMM
        pytorch_gemm = nn.Linear(k, n).to(bench.device)
        pytorch_result = bench.measure(
            pytorch_gemm, 
            x.view(-1, k),
            iterations=100,
            name="PyTorch GEMM"
        )
        
        # CUTLASS GEMM
        cutlass_gemm = OptimizedGEMM(m, n, k, use_cutlass=True).to(bench.device)
        cutlass_result = bench.measure(
            cutlass_gemm,
            x,
            iterations=100,
            name="CUTLASS GEMM"
        )
        
        gemm_speedup = bench.compare(pytorch_result['ms_per_iter'], 
                                     cutlass_result['ms_per_iter'])
        
        print(f"    Speedup: {gemm_speedup:.2f}x")
        
        # Calculate TFLOPS
        flops_per_gemm = 2 * m * n * k
        pytorch_tflops = (flops_per_gemm * 100) / (pytorch_result['total_time_s'] * 1e12)
        cutlass_tflops = (flops_per_gemm * 100) / (cutlass_result['total_time_s'] * 1e12)
        
        print(f"    PyTorch: {pytorch_tflops:.2f} TFLOPS")
        print(f"    CUTLASS: {cutlass_tflops:.2f} TFLOPS")
        
        # 2. Test full transformer block
        print("\n2. TRANSFORMER BLOCK")
        print("-" * 40)
        
        # Create models
        pytorch_model = create_transformer_block(hidden_dim).to(bench.device)
        
        # PyTorch baseline
        pytorch_result = bench.measure(
            pytorch_model,
            x,
            iterations=50,
            name="PyTorch"
        )
        
        # torch.compile
        compiled_model = torch.compile(pytorch_model, mode='max-autotune')
        # Trigger compilation
        with torch.no_grad():
            _ = compiled_model(x)
        
        compiled_result = bench.measure(
            compiled_model,
            x,
            iterations=50,
            name="torch.compile"
        )
        
        compile_speedup = bench.compare(pytorch_result['ms_per_iter'],
                                       compiled_result['ms_per_iter'])
        
        print(f"    torch.compile speedup: {compile_speedup:.2f}x")
        
        # 3. Memory bandwidth test
        print("\n3. MEMORY BANDWIDTH")
        print("-" * 40)
        
        # Create large tensors for bandwidth test
        size_gb = 1  # 1 GB of data
        elements = (size_gb * 1024 * 1024 * 1024) // 4  # float32 = 4 bytes
        a = torch.randn(elements // 1024, 1024, device=bench.device)
        b = torch.randn(elements // 1024, 1024, device=bench.device)
        
        # Measure copy bandwidth
        if bench.device.type == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(10):
                c = a + b  # Simple memory-bound operation
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            
            # Each iteration reads 2GB and writes 1GB = 3GB
            bandwidth_gb_s = (3 * size_gb * 10) / elapsed
            print(f"    Measured: {bandwidth_gb_s:.2f} GB/s")
            print(f"    B200 theoretical: ~8,000 GB/s")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\nKey Findings:")
    print("1. Raw GEMM performance shows kernel efficiency")
    print("2. Transformer block tests real-world usage")
    print("3. Memory bandwidth confirms hardware capabilities")
    
    print("\nExpected B200 Performance:")
    print("  - BF16: ~2,500 TFLOPS peak")
    print("  - MXFP8: ~5,000 TFLOPS peak (2x BF16)")
    print("  - FP4: ~10,000 TFLOPS peak (4x BF16)")
    print("  - Memory: ~8,000 GB/s")
    
    if has_blackwell:
        print("\n✓ You have Blackwell hardware!")
        print("  To achieve peak performance:")
        print("  1. Use large enough matrices (>= 1024x1024)")
        print("  2. Enable MXFP8/FP4 quantization")
        print("  3. Minimize kernel launch overhead")
        print("  4. Use tensor core aligned dimensions")


if __name__ == "__main__":
    main()
