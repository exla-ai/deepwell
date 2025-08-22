#!/usr/bin/env python3
"""
Comprehensive benchmark comparing PyTorch (with torch.compile) vs Deepwell with CUTLASS FMHA
Tests both isolated attention and full transformer models
"""

import time
import torch
import torch.nn as nn
import os
import sys

# Add src to path
sys.path.insert(0, '/root/deepwell/src')

from deepwell.kernels.blackwell_production import BlackwellFlashAttention, BlackwellConfig, DWSelfAttention


def time_function(fn, warmup=10, iterations=100):
    """Time a function with warmup and return average time per iteration"""
    # Warmup
    for _ in range(warmup):
        _ = fn()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


class SimpleTransformerTorch(nn.Module):
    """Simple transformer using PyTorch MultiheadAttention"""
    def __init__(self, hidden_dim, heads, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dtype=torch.bfloat16)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ])
    
    def forward(self, x):
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device, dtype=x.dtype)
        
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x, _ = attn(x, x, x, attn_mask=mask)
            x = norm(x + residual)
        return x


class SimpleTransformerDeepwell(nn.Module):
    """Simple transformer using Deepwell DWSelfAttention"""
    def __init__(self, hidden_dim, heads, layers=4):
        super().__init__()
        config = BlackwellConfig()
        self.layers = nn.ModuleList([
            DWSelfAttention(hidden_dim, heads, config)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ])
    
    def forward(self, x):
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        return x


def benchmark_attention_only():
    """Benchmark just the attention operation"""
    print("\n" + "="*60)
    print("ATTENTION-ONLY BENCHMARK (CUTLASS FMHA vs PyTorch SDPA)")
    print("="*60)
    
    shapes = [
        # (batch, heads, seq_len, head_dim)
        (1, 8, 128, 64),
        (2, 8, 256, 64),
        (4, 16, 256, 128),
        (8, 16, 512, 128),
        (1, 32, 1024, 128),
    ]
    
    results = []
    
    for B, H, S, D in shapes:
        # Skip if constraints not met
        if S % 64 != 0 or D not in [64, 128]:
            continue
            
        print(f"\nShape: B={B}, H={H}, S={S}, D={D}")
        
        # Prepare data
        q = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16).contiguous()
        k = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16).contiguous()
        v = torch.randn(B, H, S, D, device='cuda', dtype=torch.bfloat16).contiguous()
        
        # Deepwell CUTLASS
        fa = BlackwellFlashAttention(BlackwellConfig())
        def run_deepwell():
            return fa.forward(q, k, v, causal=True)
        
        # PyTorch SDPA
        def run_pytorch():
            # Convert to [B*H, S, D] for SDPA
            q_flat = q.view(B*H, S, D)
            k_flat = k.view(B*H, S, D)
            v_flat = v.view(B*H, S, D)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_flat, k_flat, v_flat, is_causal=True
            )
            return out.view(B, H, S, D)
        
        # Compile PyTorch version
        run_pytorch_compiled = torch.compile(run_pytorch, mode='reduce-overhead')
        
        # Benchmark
        try:
            time_dw = time_function(run_deepwell, warmup=20, iterations=100)
            time_pt = time_function(run_pytorch, warmup=20, iterations=100)
            time_pt_compiled = time_function(run_pytorch_compiled, warmup=20, iterations=100)
            
            speedup_eager = time_pt / time_dw
            speedup_compiled = time_pt_compiled / time_dw
            
            print(f"  Deepwell CUTLASS:  {1000*time_dw:.3f} ms")
            print(f"  PyTorch (eager):   {1000*time_pt:.3f} ms ({speedup_eager:.2f}x speedup)")
            print(f"  PyTorch (compile): {1000*time_pt_compiled:.3f} ms ({speedup_compiled:.2f}x speedup)")
            
            results.append({
                'shape': (B, H, S, D),
                'deepwell_ms': time_dw * 1000,
                'pytorch_eager_ms': time_pt * 1000,
                'pytorch_compile_ms': time_pt_compiled * 1000,
                'speedup_eager': speedup_eager,
                'speedup_compiled': speedup_compiled
            })
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def benchmark_transformer_model():
    """Benchmark full transformer models"""
    print("\n" + "="*60)
    print("FULL TRANSFORMER MODEL BENCHMARK")
    print("="*60)
    
    configs = [
        # (hidden_dim, heads, layers, batch_size, seq_len)
        (256, 4, 4, 4, 128),   # Small model
        (512, 8, 4, 4, 256),   # Medium model
        (1024, 16, 4, 2, 512), # Large model
    ]
    
    results = []
    
    for hidden_dim, heads, layers, batch_size, seq_len in configs:
        # Check constraints
        head_dim = hidden_dim // heads
        if seq_len % 64 != 0 or head_dim not in [64, 128]:
            continue
            
        print(f"\nConfig: hidden={hidden_dim}, heads={heads}, layers={layers}, batch={batch_size}, seq={seq_len}")
        
        # Create models
        model_torch = SimpleTransformerTorch(hidden_dim, heads, layers).cuda()
        model_deepwell = SimpleTransformerDeepwell(hidden_dim, heads, layers).cuda()
        
        # Compile torch model
        model_torch_compiled = torch.compile(model_torch, mode='reduce-overhead')
        
        # Input data
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.bfloat16)
        
        # Benchmark functions
        def run_torch():
            return model_torch(x)
        
        def run_torch_compiled():
            return model_torch_compiled(x)
        
        def run_deepwell():
            return model_deepwell(x)
        
        # Time them
        try:
            time_dw = time_function(run_deepwell, warmup=10, iterations=50)
            time_torch = time_function(run_torch, warmup=10, iterations=50)
            time_torch_comp = time_function(run_torch_compiled, warmup=10, iterations=50)
            
            speedup_eager = time_torch / time_dw
            speedup_compiled = time_torch_comp / time_dw
            
            print(f"  Deepwell:          {1000*time_dw:.3f} ms")
            print(f"  PyTorch (eager):   {1000*time_torch:.3f} ms ({speedup_eager:.2f}x speedup)")
            print(f"  PyTorch (compile): {1000*time_torch_comp:.3f} ms ({speedup_compiled:.2f}x speedup)")
            
            results.append({
                'config': (hidden_dim, heads, layers, batch_size, seq_len),
                'deepwell_ms': time_dw * 1000,
                'pytorch_eager_ms': time_torch * 1000,
                'pytorch_compile_ms': time_torch_comp * 1000,
                'speedup_eager': speedup_eager,
                'speedup_compiled': speedup_compiled
            })
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def print_summary(attention_results, transformer_results):
    """Print summary table of results"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if attention_results:
        print("\n### Attention-Only Performance ###")
        print("Shape (B,H,S,D) | Deepwell(ms) | PyTorch Eager | PyTorch Compile | Speedup(eager) | Speedup(compile)")
        print("-" * 90)
        for r in attention_results:
            B, H, S, D = r['shape']
            print(f"({B},{H},{S:4},{D:3}) | {r['deepwell_ms']:8.3f} | {r['pytorch_eager_ms']:8.3f} | {r['pytorch_compile_ms']:8.3f} | {r['speedup_eager']:7.2f}x | {r['speedup_compiled']:7.2f}x")
        
        # Average speedups
        avg_speedup_eager = sum(r['speedup_eager'] for r in attention_results) / len(attention_results)
        avg_speedup_compiled = sum(r['speedup_compiled'] for r in attention_results) / len(attention_results)
        print(f"\nAverage Speedup: {avg_speedup_eager:.2f}x vs eager, {avg_speedup_compiled:.2f}x vs compiled")
    
    if transformer_results:
        print("\n### Full Transformer Performance ###")
        print("Config (H,h,L,B,S) | Deepwell(ms) | PyTorch Eager | PyTorch Compile | Speedup(eager) | Speedup(compile)")
        print("-" * 90)
        for r in transformer_results:
            hidden, heads, layers, batch, seq = r['config']
            print(f"({hidden:4},{heads:2},{layers},{batch},{seq:3}) | {r['deepwell_ms']:8.3f} | {r['pytorch_eager_ms']:8.3f} | {r['pytorch_compile_ms']:8.3f} | {r['speedup_eager']:7.2f}x | {r['speedup_compiled']:7.2f}x")
        
        # Average speedups
        if transformer_results:
            avg_speedup_eager = sum(r['speedup_eager'] for r in transformer_results) / len(transformer_results)
            avg_speedup_compiled = sum(r['speedup_compiled'] for r in transformer_results) / len(transformer_results)
            print(f"\nAverage Speedup: {avg_speedup_eager:.2f}x vs eager, {avg_speedup_compiled:.2f}x vs compiled")


def main():
    # Set environment for CUTLASS bridge
    os.environ['DW_ENABLE_FMHA_BRIDGE'] = '1'
    os.environ['DW_FMHA_BRIDGE_PATH'] = '/root/deepwell/csrc/fmha_bridge_min/build/libdw_fmha_bridge_min.so'
    os.environ['DW_FMHA_DEBUG'] = '0'  # Turn off debug for cleaner output
    
    # PyTorch settings for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    print("="*60)
    print("DEEPWELL vs PYTORCH COMPREHENSIVE BENCHMARK")
    print("Using CUTLASS FMHA with sm_100a optimizations")
    print("="*60)
    
    # Run benchmarks
    attention_results = benchmark_attention_only()
    transformer_results = benchmark_transformer_model()
    
    # Print summary
    print_summary(attention_results, transformer_results)


if __name__ == '__main__':
    main()
