# Deepwell Examples

This directory contains examples demonstrating how to use Deepwell for optimizing deep learning models on NVIDIA Blackwell GPUs.

## Examples

### 1. `train_moe.py` - Mixture of Experts Training
Complete example showing how to:
- Build an MoE transformer model
- Optimize it with Deepwell for Blackwell GPUs
- Train with mixed precision (MXFP8)
- Leverage grouped GEMM for expert routing

```bash
python examples/train_moe.py
```

**Key Features:**
- 8 experts per layer
- Top-2 routing
- Automatic optimization for Blackwell
- Performance benchmarking

### 2. `test_kernels.py` - Direct Kernel Testing
Test CUTLASS kernels and optimization directly:
- GEMM kernel performance
- Model optimization speedups
- Mixed precision operations

```bash
python examples/test_kernels.py
```

**Tests Include:**
- Direct CUTLASS vs PyTorch GEMM
- MLP model optimization
- MXFP8 quantization accuracy

## Running Examples

### Prerequisites
1. Blackwell GPU (B100/B200)
2. Deepwell installed:
```bash
pip install git+https://github.com/exla-ai/deepwell.git
```

### Expected Results

On B200 GPU:
- **MoE Training**: 2-4x speedup with Deepwell optimization
- **GEMM Kernels**: 1,000-10,000 TFLOPS depending on size
- **Model Optimization**: 2-5x speedup on transformer models

## Creating Your Own Examples

To optimize your model with Deepwell:

```python
import deepwell as dw

# Your PyTorch model
model = create_your_model()

# Optimize for Blackwell
optimized = dw.optimize_for_blackwell(
    model,
    precision="mxfp8",  # or "fp4" for max speed
    batch_size=32,
    seq_len=512
)

# Use as normal
output = optimized(input)
```

## Performance Tips

1. **Use large enough matrices** - Blackwell excels at large GEMMs (>1024×1024)
2. **Enable MXFP8/FP4** - Get 2-4x speedup with minimal accuracy loss
3. **Batch operations** - Use grouped GEMM for parallel expert execution
4. **Profile your code** - Use `torch.profiler` to identify bottlenecks

## Troubleshooting

If you see warnings about CUTLASS not being available:
1. Ensure CUDA kernels compiled during install
2. Check GPU is detected: `python -c "import deepwell as dw; dw.probe()"`
3. Rebuild extensions: `python setup.py build_ext --inplace`

## More Information

- [Deepwell Documentation](../README.md)
- [Blackwell Architecture Guide](https://developer.nvidia.com/blackwell)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
