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
    """Benchmark raw GEMM performance with proper timing."""
    print("\n" + "=" * 70)
    print("GEMM PERFORMANCE (vs torch.compile)")
    print("=" * 70)
    
    bench = BenchmarkHarness(device)
    
    # Test different sizes with appropriate iteration counts
    configs = [
        {"name": "Small", "m": 16384, "n": 3072, "k": 768, "iters": 1000},
        {"name": "Medium", "m": 65536, "n": 4096, "k": 1024, "iters": 100},
        {"name": "Large", "m": 262144, "n": 5120, "k": 1280, "iters": 10},
    ]
    
    for config in configs:
        print(f"\n{config['name']} ({config['m']}×{config['n']}×{config['k']}):")
        print("-" * 40)
        
        m, n, k = config['m'], config['n'], config['k']
        iterations = config['iters']
        
        # Use BF16 for all tests (fair comparison)
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        
        # 1. PyTorch baseline
        pytorch_gemm = lambda x: torch.matmul(x, b)
        pytorch_result = bench.measure(pytorch_gemm, a, iterations=iterations, name="PyTorch")
        
        # 2. torch.compile baseline
        compiled_gemm = torch.compile(pytorch_gemm, mode='max-autotune')
        # Warmup compilation
        _ = compiled_gemm(a)
        compiled_result = bench.measure(compiled_gemm, a, iterations=iterations, name="torch.compile")
        
        # 3. CUTLASS if available
        try:
            from deepwell import cutlass_kernels
            
            kernel = cutlass_kernels.BlackwellGemmKernel()
            kernel.initialize(m, n, k, "bf16", False, 32)
            
            cutlass_gemm = lambda x: kernel.gemm(x, b)
            cutlass_result = bench.measure(cutlass_gemm, a, iterations=iterations, name="CUTLASS")
            
            # Calculate metrics
            flops = 2 * m * n * k
            pytorch_tflops = (flops * iterations) / (pytorch_result['total_time_s'] * 1e12)
            compiled_tflops = (flops * iterations) / (compiled_result['total_time_s'] * 1e12)
            cutlass_tflops = (flops * iterations) / (cutlass_result['total_time_s'] * 1e12)
            
            # Ensure we have meaningful timing (not 0.00)
            if cutlass_result['ms_per_iter'] < 0.01:
                print(f"    ⚠ Timing too fast, increasing iterations...")
                # Re-run with more iterations
                new_iters = iterations * 10
                cutlass_result = bench.measure(cutlass_gemm, a, iterations=new_iters, name="CUTLASS")
                cutlass_tflops = (flops * new_iters) / (cutlass_result['total_time_s'] * 1e12)
            
            print(f"\n    Results:")
            print(f"    PyTorch:      {pytorch_result['ms_per_iter']:.4f} ms/iter, {pytorch_tflops:.2f} TFLOPS")
            print(f"    torch.compile: {compiled_result['ms_per_iter']:.4f} ms/iter, {compiled_tflops:.2f} TFLOPS")
            print(f"    CUTLASS:      {cutlass_result['ms_per_iter']:.4f} ms/iter, {cutlass_tflops:.2f} TFLOPS")
            
            compile_speedup = pytorch_result['ms_per_iter'] / compiled_result['ms_per_iter']
            cutlass_speedup = compiled_result['ms_per_iter'] / cutlass_result['ms_per_iter']
            
            print(f"\n    Speedups:")
            print(f"    torch.compile vs PyTorch: {compile_speedup:.2f}x")
            print(f"    CUTLASS vs torch.compile: {cutlass_speedup:.2f}x")
            
        except Exception as e:
            print(f"    CUTLASS error: {e}")


def benchmark_model(model_size='small', device='cuda'):
    """Benchmark a transformer model against torch.compile."""
    print("\n" + "=" * 70)
    print(f"TRANSFORMER MODEL BENCHMARK ({model_size.upper()})")
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
    
    # Test input
    batch_size = 32
    seq_len = 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    print(f"\nModel config:")
    print(f"  Layers: {config['layers']}")
    print(f"  Hidden dim: {config['hidden']}")
    print(f"  Heads: {config['heads']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # 1. PyTorch baseline
    model = TransformerModel().to(device)
    baseline_result = bench.measure(model, input_ids, iterations=10, name="PyTorch")
    
    # 2. torch.compile baseline
    compiled_model = torch.compile(model, mode='max-autotune')
    # Trigger compilation
    _ = compiled_model(input_ids)
    compiled_result = bench.measure(compiled_model, input_ids, iterations=10, name="torch.compile")
    
    # 3. Deepwell optimized
    try:
        # Create fresh model for Deepwell
        model_for_deepwell = TransformerModel().to(device)
        optimized_model = dw.optimize_for_blackwell(model_for_deepwell)
        opt_result = bench.measure(optimized_model, input_ids, iterations=10, name="Deepwell")
        
        # Calculate speedups
        compile_vs_pytorch = baseline_result['ms_per_iter'] / compiled_result['ms_per_iter']
        deepwell_vs_pytorch = baseline_result['ms_per_iter'] / opt_result['ms_per_iter']
        deepwell_vs_compile = compiled_result['ms_per_iter'] / opt_result['ms_per_iter']
        
        # Calculate throughput
        tokens_per_iter = batch_size * seq_len
        pytorch_throughput = tokens_per_iter / (baseline_result['ms_per_iter'] / 1000)
        compile_throughput = tokens_per_iter / (compiled_result['ms_per_iter'] / 1000)
        deepwell_throughput = tokens_per_iter / (opt_result['ms_per_iter'] / 1000)
        
        print(f"\nResults:")
        print(f"  PyTorch:       {baseline_result['ms_per_iter']:.2f} ms/iter ({pytorch_throughput:.0f} tokens/sec)")
        print(f"  torch.compile: {compiled_result['ms_per_iter']:.2f} ms/iter ({compile_throughput:.0f} tokens/sec)")
        print(f"  Deepwell:      {opt_result['ms_per_iter']:.2f} ms/iter ({deepwell_throughput:.0f} tokens/sec)")
        
        print(f"\nSpeedups:")
        print(f"  torch.compile vs PyTorch: {compile_vs_pytorch:.2f}x")
        print(f"  Deepwell vs PyTorch:      {deepwell_vs_pytorch:.2f}x")
        print(f"  Deepwell vs torch.compile: {deepwell_vs_compile:.2f}x")
        
        if deepwell_vs_compile > 1.0:
            print(f"\n✅ Deepwell is {deepwell_vs_compile:.2f}x faster than torch.compile!")
        elif deepwell_vs_compile > 0.9:
            print(f"\n✓ Deepwell matches torch.compile performance")
        else:
            print(f"\n⚠ torch.compile is currently faster")
            
    except Exception as e:
        print(f"\n❌ Deepwell optimization failed: {e}")
    
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
