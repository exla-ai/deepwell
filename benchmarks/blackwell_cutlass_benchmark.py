#!/usr/bin/env python3
"""
Blackwell GPU Benchmark Suite using CUTLASS kernels.
Compares CUTLASS Blackwell-optimized kernels against torch.compile().
"""

import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import deepwell as dw
from deepwell.kernels.cutlass_bindings import CutlassKernel, CutlassConfig, BlackwellMMATensor
from deepwell.kernels.production_kernels import ProductionKernelManager, KernelConfig


@dataclass
class BenchmarkResult:
    """Stores benchmark results for comparison."""
    name: str
    time_ms: float
    tflops: float
    memory_gb: float
    efficiency: float


class BlackwellBenchmark:
    """
    Comprehensive benchmark suite for Blackwell GPUs.
    Compares CUTLASS Blackwell kernels against torch.compile().
    """
    
    def __init__(self, device='cuda', warmup_iters=10, measure_iters=100):
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        
        # Initialize Blackwell-specific configurations
        self.cutlass_config = CutlassConfig(
            tile_m=256,
            tile_n=256, 
            tile_k=128,
            stages=4,
            use_tcgen05=True,  # Enable Blackwell tcgen05.mma
            tmem_residency=True,
            force_tcgen05=True,
            microscale_block_size=32
        )
        
        # Initialize kernel managers
        self.kernel_config = KernelConfig(
            use_cutlass=True,
            use_tensor_cores=True,
            precision="bf16",
            batch_kernels=True
        )
        self.kernel_manager = ProductionKernelManager(self.kernel_config)
        
        # Initialize Blackwell MMA tensor info
        self.blackwell_mma = BlackwellMMATensor()
        
        # Results storage
        self.results = {}
    
    def warmup(self, func, *args):
        """Warmup phase to ensure kernels are compiled."""
        for _ in range(self.warmup_iters):
            _ = func(*args)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
    
    def measure_cuda_time(self, func, *args) -> float:
        """Accurately measure CUDA kernel execution time."""
        # Warmup
        self.warmup(func, *args)
        
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Synchronize and start timing
        torch.cuda.synchronize()
        start_event.record()
        
        # Execute measured iterations
        for _ in range(self.measure_iters):
            _ = func(*args)
        
        # Stop timing
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate average time per iteration
        total_ms = start_event.elapsed_time(end_event)
        return total_ms / self.measure_iters
    
    def benchmark_gemm_precision(self, m: int, n: int, k: int, precision: str):
        """
        Benchmark GEMM at different precisions comparing torch.compile vs CUTLASS.
        
        Args:
            m, n, k: Matrix dimensions
            precision: Precision type (bf16, mxfp8, nvfp4)
        """
        print(f"\n{'='*60}")
        print(f"GEMM Benchmark: {m}×{n}×{k} @ {precision.upper()}")
        print(f"{'='*60}")
        
        # Prepare data based on precision
        if precision == "nvfp4":
            # For FP4, we'll use BF16 data and simulate quantization
            dtype = torch.bfloat16
            print("Note: Using BF16 with simulated FP4 quantization")
        elif precision == "mxfp8":
            # For MXFP8, use BF16 and simulate microscaling
            dtype = torch.bfloat16
            print("Note: Using BF16 with simulated MXFP8 microscaling")
        else:
            dtype = torch.bfloat16
        
        # Create test matrices
        a = torch.randn(m, k, device=self.device, dtype=dtype)
        b = torch.randn(k, n, device=self.device, dtype=dtype)
        
        results = []
        
        # 1. Baseline: torch.compile with max-autotune
        print("\n1. torch.compile (baseline):")
        def gemm_torch(x, y):
            return torch.matmul(x, y)
        
        compiled_gemm = torch.compile(gemm_torch, mode='max-autotune', fullgraph=True)
        
        # Ensure compilation is complete
        for _ in range(5):
            _ = compiled_gemm(a, b)
        torch.cuda.synchronize()
        
        compile_time = self.measure_cuda_time(lambda: compiled_gemm(a, b))
        compile_flops = 2 * m * n * k
        compile_tflops = (compile_flops / 1e12) / (compile_time / 1000)
        
        results.append(BenchmarkResult(
            name="torch.compile",
            time_ms=compile_time,
            tflops=compile_tflops,
            memory_gb=(a.numel() + b.numel() + m*n) * dtype.itemsize / 1e9,
            efficiency=compile_tflops / 2500  # Assuming 2.5 PFLOPS peak for BF16
        ))
        
        print(f"  Time: {compile_time:.3f} ms")
        print(f"  Performance: {compile_tflops:.2f} TFLOPS")
        
        # 2. CUTLASS Blackwell kernel
        print("\n2. CUTLASS Blackwell kernel:")
        try:
            # Initialize CUTLASS kernel with Blackwell optimizations
            cutlass_kernel = CutlassKernel(self.cutlass_config)
            cutlass_kernel.initialize(m, n, k, precision, 
                                     use_microscaling=(precision in ["mxfp8", "nvfp4"]),
                                     block_size=32)
            
            def cutlass_gemm():
                return cutlass_kernel.gemm(a, b, use_microscaling=(precision in ["mxfp8", "nvfp4"]))
            
            cutlass_time = self.measure_cuda_time(cutlass_gemm)
            cutlass_tflops = (compile_flops / 1e12) / (cutlass_time / 1000)
            
            # Get theoretical peak for this precision on Blackwell
            perf_est = self.blackwell_mma.estimate_performance(m, n, k, precision)
            
            results.append(BenchmarkResult(
                name=f"CUTLASS {precision}",
                time_ms=cutlass_time,
                tflops=cutlass_tflops,
                memory_gb=(a.numel() + b.numel() + m*n) * dtype.itemsize / 1e9,
                efficiency=cutlass_tflops / perf_est['peak_tflops']
            ))
            
            print(f"  Time: {cutlass_time:.3f} ms")
            print(f"  Performance: {cutlass_tflops:.2f} TFLOPS")
            print(f"  Theoretical Peak: {perf_est['peak_tflops']:.0f} TFLOPS")
            print(f"  Efficiency: {results[-1].efficiency*100:.1f}%")
            
            # Calculate speedup
            speedup = compile_time / cutlass_time
            print(f"\n  Speedup vs torch.compile: {speedup:.2f}x")
            
            if speedup > 1.0:
                print(f"  ✅ CUTLASS is {speedup:.2f}x faster!")
            else:
                print(f"  ⚠ torch.compile is {1/speedup:.2f}x faster")
                
        except Exception as e:
            print(f"  ❌ CUTLASS error: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Production kernel manager (hybrid approach)
        print("\n3. Production Kernel Manager:")
        try:
            # Configure for specific precision
            prod_config = KernelConfig(
                use_cutlass=True,
                use_tensor_cores=True,
                precision=precision,
                batch_kernels=True
            )
            prod_manager = ProductionKernelManager(prod_config)
            
            def production_gemm():
                return prod_manager.gemm(a, b)
            
            prod_time = self.measure_cuda_time(production_gemm)
            prod_tflops = (compile_flops / 1e12) / (prod_time / 1000)
            
            results.append(BenchmarkResult(
                name="Production",
                time_ms=prod_time,
                tflops=prod_tflops,
                memory_gb=(a.numel() + b.numel() + m*n) * dtype.itemsize / 1e9,
                efficiency=prod_tflops / perf_est['peak_tflops']
            ))
            
            print(f"  Time: {prod_time:.3f} ms")
            print(f"  Performance: {prod_tflops:.2f} TFLOPS")
            
            speedup = compile_time / prod_time
            print(f"  Speedup vs torch.compile: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  ❌ Production kernel error: {e}")
        
        return results
    
    def benchmark_transformer_layer(self, batch_size: int, seq_len: int, 
                                   hidden_dim: int, num_heads: int):
        """
        Benchmark a full transformer layer comparing torch.compile vs CUTLASS.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        """
        print(f"\n{'='*60}")
        print(f"Transformer Layer Benchmark")
        print(f"Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}, Heads: {num_heads}")
        print(f"{'='*60}")
        
        # Create transformer layer
        class TransformerLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.attn = nn.MultiheadAttention(hidden_dim, num_heads, 
                                                 batch_first=True, dtype=torch.bfloat16)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4, dtype=torch.bfloat16),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim, dtype=torch.bfloat16)
                )
            
            def forward(self, x):
                # Self-attention block
                residual = x
                x = self.norm1(x)
                x, _ = self.attn(x, x, x)
                x = residual + x
                
                # MLP block
                residual = x
                x = self.norm2(x)
                x = self.mlp(x)
                x = residual + x
                
                return x
        
        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, 
                       device=self.device, dtype=torch.bfloat16)
        
        results = []
        
        # 1. torch.compile baseline
        print("\n1. torch.compile (baseline):")
        model_compile = TransformerLayer().to(self.device)
        compiled_model = torch.compile(model_compile, mode='max-autotune', fullgraph=True)
        
        # Warmup compilation
        for _ in range(3):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        
        compile_time = self.measure_cuda_time(lambda: compiled_model(x))
        
        # Calculate approximate FLOPs for transformer layer
        # Attention: 4 * batch * seq^2 * hidden
        # MLP: 2 * batch * seq * hidden * 4 * hidden
        attn_flops = 4 * batch_size * seq_len * seq_len * hidden_dim
        mlp_flops = 2 * batch_size * seq_len * hidden_dim * 4 * hidden_dim
        total_flops = attn_flops + mlp_flops
        compile_tflops = (total_flops / 1e12) / (compile_time / 1000)
        
        print(f"  Time: {compile_time:.3f} ms")
        print(f"  Performance: {compile_tflops:.2f} TFLOPS")
        
        results.append(BenchmarkResult(
            name="torch.compile",
            time_ms=compile_time,
            tflops=compile_tflops,
            memory_gb=0,  # Not calculated for simplicity
            efficiency=compile_tflops / 2500
        ))
        
        # 2. Deepwell optimized version
        print("\n2. Deepwell Optimized:")
        try:
            model_deepwell = TransformerLayer().to(self.device)
            
            # Apply Deepwell optimizations
            optimized_model = dw.optimize_for_blackwell(
                model_deepwell,
                precision="bf16",
                batch_size=batch_size,
                seq_len=seq_len
            )
            
            # Warmup
            for _ in range(3):
                _ = optimized_model(x)
            torch.cuda.synchronize()
            
            deepwell_time = self.measure_cuda_time(lambda: optimized_model(x))
            deepwell_tflops = (total_flops / 1e12) / (deepwell_time / 1000)
            
            print(f"  Time: {deepwell_time:.3f} ms")
            print(f"  Performance: {deepwell_tflops:.2f} TFLOPS")
            
            speedup = compile_time / deepwell_time
            print(f"  Speedup vs torch.compile: {speedup:.2f}x")
            
            if speedup > 1.0:
                print(f"  ✅ Deepwell is {speedup:.2f}x faster!")
            else:
                print(f"  ⚠ torch.compile is {1/speedup:.2f}x faster")
            
            results.append(BenchmarkResult(
                name="Deepwell",
                time_ms=deepwell_time,
                tflops=deepwell_tflops,
                memory_gb=0,
                efficiency=deepwell_tflops / 2500
            ))
            
        except Exception as e:
            print(f"  ❌ Deepwell error: {e}")
        
        return results
    
    def benchmark_moe_layer(self, batch_size: int, seq_len: int, 
                           hidden_dim: int, num_experts: int, top_k: int):
        """
        Benchmark Mixture of Experts layer using grouped GEMM.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_experts: Number of experts
            top_k: Number of experts to route to
        """
        print(f"\n{'='*60}")
        print(f"MoE Layer Benchmark")
        print(f"Experts: {num_experts}, Top-K: {top_k}")
        print(f"{'='*60}")
        
        # This would use CUTLASS grouped GEMM for efficient MoE
        # Implementation details omitted for brevity
        
        print("MoE benchmark using grouped GEMM - implementation pending")
        return []
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite for Blackwell GPUs."""
        print("="*70)
        print("BLACKWELL GPU BENCHMARK SUITE".center(70))
        print("Comparing CUTLASS Blackwell Kernels vs torch.compile()".center(70))
        print("="*70)
        
        # Detect hardware
        hw = dw.probe()
        print("\n📊 Hardware Detection:")
        for gpu in hw.gpus:
            print(f"  GPU: {gpu.name}")
            if gpu.is_blackwell:
                print(f"    ✅ Blackwell {gpu.blackwell_variant} detected!")
                print(f"    SM Version: SM{gpu.sm_version}")
                print(f"    MXFP8 Support: {gpu.supports_mxfp8}")
                print(f"    NVFP4 Support: {gpu.supports_fp4}")
                print(f"    TMEM Size: 256KB")
                print(f"    tcgen05.mma: Enabled")
        
        all_results = {}
        
        # 1. GEMM benchmarks at different precisions
        print(f"\n{'='*70}")
        print("SECTION 1: GEMM PERFORMANCE".center(70))
        print(f"{'='*70}")
        
        gemm_configs = [
            # (M, N, K, precision)
            (4096, 4096, 4096, "bf16"),
            (8192, 8192, 2048, "bf16"),
            (4096, 4096, 4096, "mxfp8"),
            (8192, 8192, 2048, "mxfp8"),
            (4096, 4096, 4096, "nvfp4"),
        ]
        
        for m, n, k, precision in gemm_configs:
            key = f"gemm_{m}x{n}x{k}_{precision}"
            all_results[key] = self.benchmark_gemm_precision(m, n, k, precision)
        
        # 2. Transformer layer benchmarks
        print(f"\n{'='*70}")
        print("SECTION 2: TRANSFORMER PERFORMANCE".center(70))
        print(f"{'='*70}")
        
        transformer_configs = [
            # (batch_size, seq_len, hidden_dim, num_heads)
            (32, 512, 768, 12),   # Small model
            (16, 1024, 1024, 16), # Medium model
            (8, 2048, 1280, 20),  # Large model
        ]
        
        for batch, seq, hidden, heads in transformer_configs:
            key = f"transformer_b{batch}_s{seq}_h{hidden}"
            all_results[key] = self.benchmark_transformer_layer(batch, seq, hidden, heads)
        
        # 3. Summary
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY".center(70))
        print(f"{'='*70}")
        
        # Calculate average speedups
        torch_compile_times = []
        cutlass_times = []
        
        for config_results in all_results.values():
            for result in config_results:
                if result.name == "torch.compile":
                    torch_compile_times.append(result.time_ms)
                elif "CUTLASS" in result.name:
                    cutlass_times.append(result.time_ms)
        
        if torch_compile_times and cutlass_times:
            avg_torch_time = np.mean(torch_compile_times)
            avg_cutlass_time = np.mean(cutlass_times)
            avg_speedup = avg_torch_time / avg_cutlass_time
            
            print(f"\nAverage Performance:")
            print(f"  torch.compile avg: {avg_torch_time:.3f} ms")
            print(f"  CUTLASS avg: {avg_cutlass_time:.3f} ms")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            
            if avg_speedup > 1.0:
                print(f"\n✅ CUTLASS Blackwell kernels are {avg_speedup:.2f}x faster on average!")
            else:
                print(f"\n⚠ torch.compile is {1/avg_speedup:.2f}x faster on average")
        
        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE".center(70))
        print(f"{'='*70}")
        
        return all_results


def main():
    """Main entry point for Blackwell benchmark."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Blackwell benchmarks require a GPU.")
        return
    
    # Set optimal PyTorch settings for benchmarking
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Create and run benchmark
    benchmark = BlackwellBenchmark(
        device='cuda',
        warmup_iters=10,
        measure_iters=100
    )
    
    results = benchmark.run_comprehensive_benchmark()
    
    # Optional: Save results to file
    import json
    with open('blackwell_benchmark_results.json', 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for key, result_list in results.items():
            serializable_results[key] = [
                {
                    'name': r.name,
                    'time_ms': r.time_ms,
                    'tflops': r.tflops,
                    'efficiency': r.efficiency
                }
                for r in result_list
            ]
        json.dump(serializable_results, f, indent=2)
    
    print("\n📊 Results saved to blackwell_benchmark_results.json")


if __name__ == "__main__":
    main()