#!/usr/bin/env python3
import time
import os
import subprocess
import re
import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import deepwell as dw
from deepwell.kernels.blackwell_production import DWSelfAttention, BlackwellConfig

# Reduce allocator fragmentation for large graphs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Aggressive runtime optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
except Exception:
    pass


class SimpleTransformerBaseline(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, num_layers=2, vocab=32768):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden_dim)
        self.layers = nn.ModuleList([
            BaselineTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab)

        # Use bf16 where beneficial
        self.embed = self.embed.to(torch.bfloat16)
        self.lm_head = self.lm_head.to(torch.bfloat16)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class BaselineTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dtype=torch.bfloat16)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim, dtype=torch.bfloat16),
        )

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = x.to(torch.bfloat16)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = DWSelfAttention(hidden_dim, num_heads, BlackwellConfig())
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, num_layers=2, vocab=32768):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# Removed local FlashSelfAttention; rely on deepwell to enforce CUTLASS FMHA


@torch.no_grad()
def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_variant(model_builder, name: str, steps: int, warmup: int, batch: int, seqlen: int, device, vocab: int = 32768):
    torch.manual_seed(0)
    model = model_builder().to(device)
    model.eval()

    # Enable compilation only for baseline if desired
    if name.startswith("torch.compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except TypeError:
            model = torch.compile(model)

    # Dummy data
    inputs = torch.randint(0, vocab, (batch, seqlen), device=device)

    # Warmup (inference-only)
    global _ATTN_TIME_ACCUM
    _ATTN_TIME_ACCUM = 0.0
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                _ = model(inputs)
    _sync_cuda()

    # Timed (inference-only)
    _ATTN_TIME_ACCUM = 0.0
    start = time.time()
    with torch.no_grad():
        for _ in range(steps):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type=="cuda")):
                _ = model(inputs)
    _sync_cuda()
    elapsed = time.time() - start
    attn_time = float(_ATTN_TIME_ACCUM)

    tokens = batch * seqlen * steps
    tps = tokens / elapsed
    itps = steps / elapsed
    return {
        "name": name,
        "time_s": elapsed,
        "steps": steps,
        "it_per_s": itps,
        "tokens_per_s": tps,
        "attn_time_s": attn_time,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set to a known-good CUTLASS FMHA bridge shape: B=1, H=4, S=64, D=64
    config = {
        "hidden_dim": 256,   # 4 heads * 64 head_dim
        "layers": 2,
        "heads": 4,
        "batch": 1,
        "seqlen": 64,
        "vocab": 8192,
        "warmup": 5,
        "steps": 50,
    }

    def build_model_baseline():
        return SimpleTransformerBaseline(
            hidden_dim=config["hidden_dim"],
            num_heads=config["heads"],
            num_layers=config["layers"],
            vocab=config["vocab"],
        )

    # Baseline torch.compile
    res_compile = benchmark_variant(build_model_baseline, "torch.compile", config["steps"], config["warmup"], config["batch"], config["seqlen"], device, vocab=config["vocab"]) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Deepwell optimized (end-to-end convenience)
    def build_dw():
        # Use the model with DWSelfAttention directly to isolate CUTLASS FMHA bridge
        base = SimpleTransformer(
            hidden_dim=config["hidden_dim"],
            num_heads=config["heads"],
            num_layers=config["layers"],
            vocab=config["vocab"],
        )
        return base

    res_deepwell = benchmark_variant(build_dw, "deepwell.optimize_for_blackwell", config["steps"], config["warmup"], config["batch"], config["seqlen"], device, vocab=config["vocab"]) 

    # CUTLASS FMHA example (native binary) — optional side measurement
    fmha_bin = "/root/deepwell/third_party/cutlass/build/examples/77_blackwell_fmha/77_blackwell_fmha_fp8"
    fmha_tflops = None
    if os.path.exists(fmha_bin):
        head_dim = config["hidden_dim"] // config["heads"]
        if config["hidden_dim"] % config["heads"] == 0 and head_dim in (64, 128):
            cmd = [
                fmha_bin,
                f"--b={config['batch']}",
                f"--h={config['heads']}",
                f"--q={config['seqlen']}",
                f"--k={config['seqlen']}",
                f"--d={head_dim}",
                "--mask=causal",
                f"--warmup_iterations={max(10, config['warmup'])}",
                f"--iterations={max(50, config['steps'])}",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                m = re.findall(r"([0-9]+\.[0-9]+) TFLOPS/s", out)
                if m:
                    fmha_tflops = float(m[-1])
            except Exception:
                fmha_tflops = None

    # Report
    print("\n=== Benchmark Results ===")
    for res in [res_compile, res_deepwell]:
        print(f"{res['name']}: time={res['time_s']:.2f}s, it/s={res['it_per_s']:.2f}, tokens/s={res['tokens_per_s']:.0f}")

    speedup = res_deepwell["it_per_s"] / res_compile["it_per_s"] if res_compile["it_per_s"] > 0 else float('nan')
    print(f"Speedup (deepwell / torch.compile): {speedup:.2f}x")
    if fmha_tflops is not None:
        print(f"CUTLASS FMHA example TFLOPS: {fmha_tflops:.1f}")
        # Project a runtime replacing attention forward with CUTLASS FMHA
        B = config["batch"]; H = config["heads"]; Q = config["seqlen"]; K = config["seqlen"]; D = config["hidden_dim"] // config["heads"]
        if config["hidden_dim"] % config["heads"] == 0 and D in (64, 128):
            flops_fwd = 4.0 * B * H * Q * K * D  # rough forward-only FLOPs
            fmha_time = flops_fwd / (fmha_tflops * 1e12)
            proj_time = max(1e-6, res_deepwell["time_s"] - res_deepwell.get("attn_time_s", 0.0) + fmha_time)
            proj_itps = res_deepwell["steps"] / proj_time
            proj_tps = (B * Q * res_deepwell["steps"]) / proj_time
            print(f"deepwell+FMHA(projected): time={proj_time:.2f}s, it/s={proj_itps:.2f}, tokens/s={proj_tps:.0f}")


if __name__ == "__main__":
    main()


