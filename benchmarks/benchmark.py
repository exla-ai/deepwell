#!/usr/bin/env python3
"""
Deepwell Blackwell Benchmark Suite.
Measures performance on NVIDIA Blackwell GPUs.
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import deepwell as dw


class BenchmarkHarness:
    """Benchmark harness with proper warmup and measurement."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.results = {}
    
    def warmup(self, model, input_data, iterations=10):
        """Warmup phase (not measured)."""
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
        """Measure performance after warmup."""
        # Warmup first (not measured)
        self.warmup(model, input_data)
        
        # Measure real performance
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


def benchmark_gemm(device='cuda'):
    """Benchmark raw GEMM performance."""
    print("\n" + "=" * 70)
    print("GEMM PERFORMANCE")
    print("=" * 70)
    
    bench = BenchmarkHarness(device)
    
    # Test different sizes
    configs = [
        {"name": "Small", "m": 16384, "n": 3072, "k": 768},
        {"name": "Medium", "m": 65536, "n": 4096, "k": 1024},
        {"name": "Large", "m": 262144, "n": 5120, "k": 1280},
    ]
    
    for config in configs:
        print(f"\n{config['name']} ({config['m']}×{config['n']}×{config['k']}):")
        print("-" * 40)
        
        m, n, k = config['m'], config['n'], config['k']
        
        # PyTorch baseline
        a = torch.randn(m, k, device=device, dtype=torch.float32)
        b = torch.randn(k, n, device=device, dtype=torch.float32)
        
        pytorch_gemm = lambda x: torch.matmul(x, b)
        pytorch_result = bench.measure(pytorch_gemm, a, iterations=100, name="PyTorch")
        
        # Calculate TFLOPS
        flops = 2 * m * n * k
        pytorch_tflops = (flops * 100) / (pytorch_result['total_time_s'] * 1e12)
        print(f"    PyTorch: {pytorch_tflops:.2f} TFLOPS")
        
        # CUTLASS if available
        try:
            from deepwell import cutlass_kernels
            
            kernel = cutlass_kernels.BlackwellGemmKernel()
            kernel.initialize(m, n, k, "bf16", False, 32)
            
            a_bf16 = a.to(torch.bfloat16)
            b_bf16 = b.to(torch.bfloat16)
            
            cutlass_gemm = lambda x: kernel.gemm(x, b_bf16)
            cutlass_result = bench.measure(cutlass_gemm, a_bf16, iterations=100, name="CUTLASS")
            
            cutlass_tflops = (flops * 100) / (cutlass_result['total_time_s'] * 1e12)
            speedup = pytorch_result['ms_per_iter'] / cutlass_result['ms_per_iter']
            
            print(f"    CUTLASS: {cutlass_tflops:.2f} TFLOPS")
            print(f"    Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"    CUTLASS not available: {e}")


def benchmark_model(model_size='small', device='cuda'):
    """Benchmark a transformer model."""
    print("\n" + "=" * 70)
    print(f"MODEL BENCHMARK ({model_size.upper()})")
    print("=" * 70)
    
    # Create model based on size
    configs = {
        'small': {'hidden': 768, 'layers': 6, 'heads': 12},
        'medium': {'hidden': 1024, 'layers': 12, 'heads': 16},
        'large': {'hidden': 1280, 'layers': 24, 'heads': 20},
    }
    
    config = configs.get(model_size, configs['small'])
    
    # Simple transformer model
    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50257, config['hidden'])
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config['hidden'],
                    nhead=config['heads'],
                    dim_feedforward=config['hidden'] * 4,
                    batch_first=True
                )
                for _ in range(config['layers'])
            ])
            self.output = nn.Linear(config['hidden'], 50257)
        
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    bench = BenchmarkHarness(device)
    model = TransformerModel().to(device)
    
    # Test input
    batch_size = 32
    seq_len = 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Baseline
    baseline_result = bench.measure(model, input_ids, iterations=20, name="Baseline")
    
    # Optimized with Deepwell
    try:
        optimized_model = dw.optimize_for_blackwell(model)
        opt_result = bench.measure(optimized_model, input_ids, iterations=20, name="Deepwell")
        
        speedup = baseline_result['ms_per_iter'] / opt_result['ms_per_iter']
        print(f"\n    Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"    Could not optimize: {e}")
    
    return baseline_result


def main():
    """Main benchmark entry point."""
    print("=" * 70)
    print("DEEPWELL BLACKWELL BENCHMARK".center(70))
    print("=" * 70)
    
    # Detect hardware
    hw = dw.probe()
    print("\n📊 Hardware Detection:")
    for gpu in hw.gpus:
        print(f"  GPU: {gpu.name}")
        if gpu.is_blackwell:
            print(f"    ✅ Blackwell {gpu.blackwell_variant} detected!")
            print(f"    MXFP8: {gpu.supports_mxfp8}")
            print(f"    FP4: {gpu.supports_fp4}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run benchmarks
    benchmark_gemm(device)
    benchmark_model('small', device)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
