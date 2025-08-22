#!/usr/bin/env python3
import time
import torch
import os

from deepwell.kernels.blackwell_production import BlackwellFlashAttention, BlackwellConfig


def bench_fmha_vs_torch(B=1, H=4, S=64, D=64, warmup=20, steps=200, causal=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'CUDA required'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Data
    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16).contiguous()

    # Deepwell CUTLASS FMHA (bridge)
    fa = BlackwellFlashAttention(BlackwellConfig())
    def run_dw():
        return fa.forward(q, k, v, causal=causal)

    # Torch SDPA baseline expects [S, B*H, D]
    def run_torch():
        qs = q.view(B*H, S, D).transpose(0, 1).contiguous().to(torch.float32)
        ks = k.view(B*H, S, D).transpose(0, 1).contiguous().to(torch.float32)
        vs = v.view(B*H, S, D).transpose(0, 1).contiguous().to(torch.float32)
        out = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=causal)
        return out.transpose(0, 1).contiguous().view(B, H, S, D)

    # Optional torch.compile on SDPA
    try:
        run_torch_compiled = torch.compile(run_torch, mode='reduce-overhead')
    except Exception:
        run_torch_compiled = run_torch

    def time_it(fn):
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Run
    os.environ.setdefault('DW_FMHA_DEBUG', '1')
    t_dw = time_it(run_dw)
    t_sdpa = time_it(run_torch)
    t_sdpa_comp = time_it(run_torch_compiled)

    itps_dw = steps / t_dw
    itps_sdpa = steps / t_sdpa
    itps_sdpa_comp = steps / t_sdpa_comp

    print('\n=== FMHA Micro-benchmark ===')
    print(f'Shape: B={B} H={H} S={S} D={D} causal={causal}')
    print(f'Deepwell CUTLASS bridge: {itps_dw:.2f} it/s (time {t_dw:.3f}s)')
    print(f'Torch SDPA (eager):      {itps_sdpa:.2f} it/s (time {t_sdpa:.3f}s)')
    print(f'Torch SDPA (compile):    {itps_sdpa_comp:.2f} it/s (time {t_sdpa_comp:.3f}s)')
    print(f'Speedup vs eager:        {itps_dw/itps_sdpa:.2f}x')
    print(f'Speedup vs torch.compile:{itps_dw/itps_sdpa_comp:.2f}x')


if __name__ == '__main__':
    # Test multiple shapes
    shapes = [
        (1, 4, 64, 64),       # Small (working shape)
        (1, 4, 128, 64),      # Medium seq len
        (1, 4, 256, 64),      # Larger seq len  
        (2, 8, 128, 128),     # Different head dim
        (4, 16, 256, 128),    # Larger batch/heads
    ]
    
    for B, H, S, D in shapes:
        # S must be multiple of 64, D must be 64 or 128 for bridge
        if S % 64 != 0 or D not in [64, 128]:
            print(f"\nSkipping B={B} H={H} S={S} D={D} (constraints not met)")
            continue
        try:
            bench_fmha_vs_torch(B=B, H=H, S=S, D=D, warmup=10, steps=100)
        except Exception as e:
            print(f"\nFailed for B={B} H={H} S={S} D={D}: {e}")


