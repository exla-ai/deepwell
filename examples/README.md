# Deepwell Examples

This directory contains examples demonstrating how to use the Deepwell library.

## Available Examples

### 1. simple_usage.py
Basic examples showing:
- How to use `BlackwellFlashAttention` directly
- Using `DWSelfAttention` as a drop-in replacement for `nn.MultiheadAttention`
- Building a transformer layer with Deepwell attention

Run it:
```bash
python examples/simple_usage.py
```

## Quick Start

```python
import torch
from deepwell.kernels.blackwell_production import (
    DWSelfAttention,
    BlackwellConfig
)

# Create attention module
config = BlackwellConfig()
attention = DWSelfAttention(
    hidden_dim=512,
    num_heads=8,
    config=config
)

# Use it like nn.MultiheadAttention
x = torch.randn(4, 256, 512, device='cuda', dtype=torch.bfloat16)
output = attention(x)
```

## Environment Setup

Before running examples, ensure the CUTLASS bridge is enabled:

```bash
export DW_ENABLE_FMHA_BRIDGE=1
export DW_FMHA_BRIDGE_PATH=/path/to/libdw_fmha_bridge_min.so
```

## Requirements

- NVIDIA Blackwell GPU (SM100a)
- CUDA 12.8+
- PyTorch 2.0+
- Built CUTLASS FMHA bridge (see main README)