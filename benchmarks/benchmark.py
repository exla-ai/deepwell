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
    """Benchmark raw GEMM performance against torch.compile."""
    print("\n" + "=" * 70)
    print("GEMM PERFORMANCE")
    print("=" * 70)
    
    # Enable TensorFloat32 for better performance
    torch.set_float32_matmul_precision('high')
    
    bench = BenchmarkHarness(device)
    
    # Test different sizes with appropriate iteration counts
    configs = [
        {"name": "Small", "m": 16384, "n": 3072, "k": 768, "iters": 100},
        {"name": "Medium", "m": 65536, "n": 4096, "k": 1024, "iters": 20},
        {"name": "Large", "m": 262144, "n": 5120, "k": 1280, "iters": 5},
    ]
    
    for config in configs:
        print(f"\n{config['name']} ({config['m']}×{config['n']}×{config['k']}):")
        print("-" * 40)
        
        m, n, k = config['m'], config['n'], config['k']
        iterations = config['iters']
        
        # Use BF16 for all tests (fair comparison)
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        
        # 1. torch.compile baseline (THE baseline)
        def gemm_fn(x, y):
            return torch.matmul(x, y)
        
        compiled_gemm = torch.compile(gemm_fn, mode='max-autotune', fullgraph=True)
        # Proper warmup - run it multiple times to ensure compilation is complete
        for _ in range(5):
            _ = compiled_gemm(a, b)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        compiled_result = bench.measure(lambda _: compiled_gemm(a, b), None, iterations=iterations, name="torch.compile (baseline)")
        
        # 2. Deepwell/CUTLASS
        try:
            from deepwell.kernels.cutlass_bindings import CutlassKernel

            kernel = CutlassKernel()

            def cutlass_gemm(_=None):
                result = kernel.gemm(a, b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure kernel completes on GPU
                return result

            # Warmup
            for _ in range(5):
                _ = cutlass_gemm()

            cutlass_result = bench.measure(cutlass_gemm, None, iterations=iterations, name="Deepwell")
            
            # If timing is still too fast, use CUDA events for more precision
            if torch.cuda.is_available() and cutlass_result['ms_per_iter'] < 0.1:
                print(f"    Using CUDA events for precise timing...")
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Measure with CUDA events
                torch.cuda.synchronize()
                start_event.record()
                for _ in range(iterations * 10):  # More iterations for precision
                    _ = kernel.gemm(a, b)
                end_event.record()
                torch.cuda.synchronize()

                cuda_time_ms = start_event.elapsed_time(end_event)
                cutlass_ms_per_iter = cuda_time_ms / (iterations * 10)
                cutlass_result['ms_per_iter'] = cutlass_ms_per_iter
                cutlass_result['total_time_s'] = cuda_time_ms / 1000.0
            
            # Calculate metrics
            flops = 2 * m * n * k
            compiled_tflops = (flops * iterations) / (compiled_result['total_time_s'] * 1e12)
            cutlass_tflops = (flops * iterations) / (cutlass_result['total_time_s'] * 1e12)
            
            speedup = compiled_result['ms_per_iter'] / cutlass_result['ms_per_iter']
            
            print(f"\n    Results:")
            print(f"    torch.compile: {compiled_result['ms_per_iter']:.4f} ms/iter, {compiled_tflops:.2f} TFLOPS")
            print(f"    Deepwell:      {cutlass_result['ms_per_iter']:.4f} ms/iter, {cutlass_tflops:.2f} TFLOPS")
            print(f"    Speedup:       {speedup:.2f}x")
            
            if speedup > 1.0:
                print(f"    ✅ Deepwell is {speedup:.2f}x faster!")
            else:
                print(f"    ⚠ torch.compile is {1/speedup:.2f}x faster")
            
        except Exception as e:
            print(f"    Deepwell error: {e}")


def benchmark_model(model_size='small', device='cuda'):
    """Benchmark a transformer model - torch.compile vs Deepwell."""
    print("\n" + "=" * 70)
    print(f"TRANSFORMER MODEL ({model_size.upper()})")
    print("=" * 70)
    
    # Enable TensorFloat32 for better performance
    torch.set_float32_matmul_precision('high')
    
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
                    batch_first=True,
                    dtype=torch.bfloat16  # Use BF16 for performance
                )
                for _ in range(config['layers'])
            ])
            self.output = nn.Linear(config['hidden'], 50257, dtype=torch.bfloat16)
            
            # Convert embedding to BF16
            self.embed = self.embed.to(torch.bfloat16)
        
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
    
    print(f"\nConfiguration:")
    print(f"  Layers: {config['layers']}")
    print(f"  Hidden dim: {config['hidden']}")
    print(f"  Heads: {config['heads']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Precision: bfloat16")
    
    # 1. torch.compile baseline (THE baseline)
    model = TransformerModel().to(device)
    
    # Compile with max optimization
    with torch.no_grad():
        compiled_model = torch.compile(model, mode='max-autotune', fullgraph=True)
        # Proper warmup - multiple runs to ensure full compilation
        for _ in range(3):
            _ = compiled_model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    compiled_result = bench.measure(compiled_model, input_ids, iterations=10, name="torch.compile (baseline)")
    
    # 2. Deepwell optimized
    try:
        # Create fresh model for Deepwell
        model_for_deepwell = TransformerModel().to(device)
        optimized_model = dw.optimize_for_blackwell(
            model_for_deepwell,
            precision="bf16",  # Match the model precision
            batch_size=batch_size,
            seq_len=seq_len
        )

        # Warmup Deepwell
        with torch.no_grad():
            for _ in range(3):
                _ = optimized_model(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        opt_result = bench.measure(optimized_model, input_ids, iterations=10, name="Deepwell")
        
        # Calculate metrics
        speedup = compiled_result['ms_per_iter'] / opt_result['ms_per_iter']
        
        # Calculate throughput
        tokens_per_iter = batch_size * seq_len
        compile_throughput = tokens_per_iter / (compiled_result['ms_per_iter'] / 1000)
        deepwell_throughput = tokens_per_iter / (opt_result['ms_per_iter'] / 1000)
        
        print(f"\nResults:")
        print(f"  torch.compile: {compiled_result['ms_per_iter']:.2f} ms/iter ({compile_throughput:.0f} tokens/sec)")
        print(f"  Deepwell:      {opt_result['ms_per_iter']:.2f} ms/iter ({deepwell_throughput:.0f} tokens/sec)")
        print(f"  Speedup:       {speedup:.2f}x")
        
        # Token throughput improvement
        throughput_gain = deepwell_throughput - compile_throughput
        print(f"\n  Throughput gain: +{throughput_gain:.0f} tokens/sec")
        
        if speedup > 1.0:
            print(f"\n✅ Deepwell is {speedup:.2f}x faster than torch.compile!")
        elif speedup > 0.9:
            print(f"\n✓ Deepwell matches torch.compile performance")
        else:
            print(f"\n⚠ torch.compile is {1/speedup:.2f}x faster")
            
    except Exception as e:
        print(f"\n❌ Deepwell optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return compiled_result


def main():
    """Main benchmark entry point."""
    print("=" * 70)
    print("DEEPWELL vs TORCH.COMPILE BENCHMARK".center(70))
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
