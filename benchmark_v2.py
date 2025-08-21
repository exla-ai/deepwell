#!/usr/bin/env python3
"""
Optimized benchmark that properly uses NVIDIA production kernels.
This should show the real performance of Blackwell.
"""

import argparse
import time
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, 'src')

# Import both engines for comparison
import deepwell as dw
from deepwell.optimized_engine import optimize_for_blackwell_v2


def create_model(model_size: str = "small"):
    """Create a test model."""
    
    configs = {
        "small": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 50257
        },
        "medium": {
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "vocab_size": 50257
        },
        "large": {
            "hidden_dim": 1280,
            "num_layers": 36,
            "num_heads": 20,
            "vocab_size": 50257
        }
    }
    
    config = configs[model_size]
    
    # Simple transformer model
    from benchmarks.blackwell_speedup import BenchmarkModel
    return BenchmarkModel(config)


def benchmark_model(model, name: str, batch_size: int, seq_len: int, iterations: int):
    """Benchmark a model."""
    
    device = next(model.parameters()).device
    
    # Create input
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    tokens_per_sec = (batch_size * seq_len * iterations) / elapsed
    
    return ms_per_iter, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description="Optimized Blackwell Benchmark")
    parser.add_argument("--model", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--precision", default="mxfp8", choices=["bf16", "mxfp8", "fp4"])
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTIMIZED BLACKWELL BENCHMARK V2")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Iterations: {args.iterations}")
    print(f"Precision: {args.precision}")
    
    # Detect hardware
    hw = dw.probe()
    print("\n" + "=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)
    for gpu in hw.gpus:
        print(f"GPU: {gpu.name}")
        if gpu.is_blackwell:
            print(f"  ✓ Blackwell {gpu.blackwell_variant} detected!")
            print(f"  MXFP8: {gpu.supports_mxfp8}")
            print(f"  FP4: {gpu.supports_fp4}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = create_model(args.model).to(device)
    
    params = sum(p.numel() for p in base_model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # 1. Baseline: PyTorch
    print("\n" + "=" * 70)
    print("1. BASELINE: Standard PyTorch")
    print("=" * 70)
    
    pytorch_ms, pytorch_tps = benchmark_model(
        base_model, "PyTorch", 
        args.batch_size, args.seq_len, args.iterations
    )
    
    print(f"  Time per iteration: {pytorch_ms:.2f} ms")
    print(f"  Throughput: {pytorch_tps:,.0f} tokens/sec")
    
    # 2. torch.compile baseline
    print("\n" + "=" * 70)
    print("2. BASELINE: torch.compile")
    print("=" * 70)
    
    compiled_model = torch.compile(base_model, mode="max-autotune")
    
    # Warmup to trigger compilation
    print("  Compiling model...")
    input_ids = torch.randint(0, 50257, (args.batch_size, args.seq_len), device=device)
    with torch.no_grad():
        for i in range(3):
            _ = compiled_model(input_ids)
            print(f"    Compilation pass {i+1}/3")
    
    compiled_ms, compiled_tps = benchmark_model(
        compiled_model, "torch.compile",
        args.batch_size, args.seq_len, args.iterations
    )
    
    print(f"  Time per iteration: {compiled_ms:.2f} ms")
    print(f"  Throughput: {compiled_tps:,.0f} tokens/sec")
    print(f"  Speedup vs PyTorch: {pytorch_ms/compiled_ms:.2f}x")
    
    # 3. Optimized Deepwell (V2)
    print("\n" + "=" * 70)
    print("3. DEEPWELL V2: Optimized for Blackwell")
    print("=" * 70)
    
    # Create a fresh model for optimization
    base_model_v2 = create_model(args.model).to(device)
    
    # Optimize with V2 engine
    optimized_model = optimize_for_blackwell_v2(
        base_model_v2, 
        precision=args.precision,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    
    deepwell_ms, deepwell_tps = benchmark_model(
        optimized_model, "Deepwell V2",
        args.batch_size, args.seq_len, args.iterations
    )
    
    print(f"\n  Time per iteration: {deepwell_ms:.2f} ms")
    print(f"  Throughput: {deepwell_tps:,.0f} tokens/sec")
    print(f"  Speedup vs PyTorch: {pytorch_ms/deepwell_ms:.2f}x")
    print(f"  Speedup vs torch.compile: {compiled_ms/deepwell_ms:.2f}x")
    
    # Calculate theoretical TFLOPS
    flops_per_iter = 2 * params * args.batch_size * args.seq_len
    deepwell_tflops = (flops_per_iter * args.iterations) / (deepwell_ms * args.iterations / 1000) / 1e12
    
    print(f"\n  Estimated performance: {deepwell_tflops:.2f} TFLOPS")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Tokens/sec':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'PyTorch':<20} {pytorch_ms:<12.2f} {pytorch_tps:<15,.0f} {'1.00x':<10}")
    print(f"{'torch.compile':<20} {compiled_ms:<12.2f} {compiled_tps:<15,.0f} {pytorch_ms/compiled_ms:<10.2f}x")
    print(f"{'Deepwell V2':<20} {deepwell_ms:<12.2f} {deepwell_tps:<15,.0f} {pytorch_ms/deepwell_ms:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if deepwell_ms < compiled_ms:
        print("✅ SUCCESS: Deepwell V2 is faster than torch.compile!")
        print(f"   Achieved {compiled_ms/deepwell_ms:.2f}x speedup over torch.compile")
        print("   Using CUTLASS kernels for large GEMMs")
        print("   Minimal overhead from framework")
    else:
        print("⚠ Deepwell V2 is not faster than torch.compile")
        print("  This may be due to:")
        print("  - Small model size (not enough large GEMMs)")
        print("  - torch.compile already optimizing well")
        print("  - Try larger model or batch size")
    
    print("\nExpected on B200:")
    print("  BF16: ~2,500 TFLOPS peak")
    print("  MXFP8: ~5,000 TFLOPS peak")
    print(f"  Current: {deepwell_tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
