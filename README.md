# Deepwell: Production Framework for NVIDIA Blackwell GPUs

A high-performance deep learning optimization framework that leverages NVIDIA Blackwell's advanced tensor core capabilities to achieve massive speedups through intelligent kernel dispatch and precision optimization.

## Overview

Deepwell is designed to exploit NVIDIA Blackwell (B100/B200) GPU features including:
- **tcgen05.mma instructions** - Native Blackwell tensor core operations
- **MXFP8/FP4 precision** - Microscaling with hardware acceleration
- **TMEM residency** - On-chip accumulator optimization
- **Smart kernel dispatch** - Automatic selection of optimal kernels

The framework provides a clean pipeline for optimizing PyTorch models to run at peak hardware efficiency on Blackwell GPUs.

## Architecture

The Deepwell framework follows a clear optimization pipeline:

```
1. Hardware Detection (probe)
   ↓
2. Model Capture (capture) 
   ↓
3. IR Generation (ir)
   ↓
4. Kernel Selection (kernels/)
   ↓
5. Compilation (compile)
   ↓
6. Execution (engine)
```

### Core Components

#### 1. Hardware Detection (`probe.py`)
Detects GPU capabilities and Blackwell-specific features:
- Identifies Blackwell variants (B100/B200)
- Checks for MXFP8/FP4 support
- Determines available tensor core features

#### 2. Model Capture (`capture.py`)
Captures PyTorch models into an intermediate representation:
- FX graph tracing for dynamic models
- Static graph construction fallback
- Operation dependency analysis

#### 3. IR Generation (`ir.py`)
Creates an optimized intermediate representation:
- Operation fusion opportunities
- Precision assignment
- Memory layout optimization

#### 4. Kernel System (`kernels/`)
Manages high-performance kernel implementations:
- **`cutlass_bindings.py`** - CUTLASS kernel integration
- **`production_kernels.py`** - Production-ready kernel manager
- **`registry.py`** - Dynamic kernel selection
- **`tcgen05_ops.py`** - Blackwell-specific operations

#### 5. Compilation Engine (`compile.py`)
Compiles IR to executable format:
- Kernel binding and optimization
- Memory planning
- Graph optimization passes

#### 6. Execution Engine (`engine.py`)
Executes optimized models:
- Efficient kernel dispatch
- Automatic mixed precision
- Memory management

## Installation

### Prerequisites
- NVIDIA GPU (Blackwell B100/B200 for full features)
- CUDA 12.0+
- Python 3.8+
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepwell.git
cd deepwell
```

2. Run setup:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install Python dependencies
- Build C++ CUDA extensions
- Compile CUTLASS kernels

3. (Optional) Install CUTLASS Python API for additional features:
```bash
pip install nvidia-cutlass
```

### Verify Installation

Run the test suite:
```bash
python test.py
```

## Usage

### Quick Start

```python
import deepwell as dw
import torch.nn as nn

# Create your model
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768)
)

# Optimize for Blackwell
optimized_model = dw.optimize_for_blackwell(
    model,
    precision="mxfp8",  # Use MXFP8 precision
    batch_size=32,
    seq_len=512
)

# Use like normal PyTorch model
output = optimized_model(input_tensor)
```

### Advanced Usage

#### Manual Pipeline Control

```python
import deepwell as dw

# 1. Detect hardware
hw = dw.probe()
print(f"GPU: {hw.gpus[0].name}")
print(f"Blackwell: {hw.gpus[0].is_blackwell}")

# 2. Capture model to IR
ir = dw.capture(model)

# 3. Compile with specific options
engine = dw.compile(
    ir,
    hw,
    precision="mxfp8",
    use_cutlass=True
)

# 4. Execute
output = engine(input_tensor)
```

#### Precision Policies

```python
from deepwell.precision import PrecisionPolicy

# Create custom precision policy
policy = PrecisionPolicy()
policy.set_default("mxfp8")
policy.set_layer_precision("attention", "bf16")
policy.set_layer_precision("mlp", "fp4")

# Apply during compilation
engine = dw.compile(ir, hw, precision_policy=policy)
```

#### Kernel Registry

```python
from deepwell.kernels import KernelRegistry

# Register custom kernel
registry = KernelRegistry()
registry.register(
    "custom_gemm",
    my_custom_kernel,
    precision="mxfp8",
    min_size=1024
)

# Use in compilation
engine = dw.compile(ir, hw, kernel_registry=registry)
```

## Performance

### Benchmarks

Run benchmarks on your hardware:
```bash
python benchmarks/benchmark.py
```

### Expected Performance on B200

<!-- Benchmark results will be filled in after running on actual hardware -->

| Configuration | Baseline | Deepwell | Speedup |
|--------------|----------|----------|---------|
| Small GEMM   | TBD      | TBD      | TBD     |
| Medium GEMM  | TBD      | TBD      | TBD     |
| Large GEMM   | TBD      | TBD      | TBD     |
| Transformer  | TBD      | TBD      | TBD     |

### Theoretical Peaks (B200)

- **BF16**: 2,500 TFLOPS
- **MXFP8**: 5,000 TFLOPS  
- **FP4**: 10,000 TFLOPS
- **Memory**: 8,000 GB/s

## Project Structure

```
deepwell/
├── src/deepwell/          # Core framework
│   ├── __init__.py        # Main API
│   ├── probe.py           # Hardware detection
│   ├── capture.py         # Model capture
│   ├── ir.py              # IR generation
│   ├── compile.py         # Compilation engine
│   ├── engine.py          # Execution engine
│   ├── kernels/           # Kernel implementations
│   │   ├── cutlass_bindings.py
│   │   ├── production_kernels.py
│   │   ├── registry.py
│   │   └── tcgen05_ops.py
│   └── precision/         # Precision management
│       └── policy.py
├── csrc/                  # C++ CUDA extensions
│   ├── blackwell_gemm_kernel.cu
│   ├── mxfp8_quantization.cu
│   └── cutlass_kernels.cpp
├── benchmarks/            # Benchmarking suite
│   ├── benchmark.py
│   └── blackwell_speedup.py
├── tests/                 # Test suite
│   └── test_basic_api.py
├── test.py                # Main test runner
└── setup.py               # Build configuration
```

## Technical Details

### Blackwell Tensor Core Features

Deepwell leverages Blackwell's advanced features:

1. **tcgen05.mma Instructions**
   - Native block-scaled matrix multiplication
   - Hardware-accelerated microscaling
   - Efficient mixed-precision computation

2. **MXFP8 Format**
   - 8-bit floating point with shared scale
   - 2x throughput vs BF16
   - Minimal accuracy loss

3. **FP4 Precision**
   - 4-bit floating point
   - 4x throughput vs BF16
   - Ideal for inference and some training

4. **TMEM Residency**
   - On-chip accumulator storage
   - Reduced memory traffic
   - Higher effective bandwidth

### Kernel Dispatch Strategy

Deepwell uses intelligent kernel selection:

```python
if matrix_size >= large_threshold:
    use_cutlass_kernel()  # Optimized for large matrices
elif matrix_size >= medium_threshold:
    use_cublas_kernel()   # Good general performance
else:
    use_pytorch_kernel()  # Low overhead for small ops
```

### Memory Management

- Automatic tensor layout optimization
- Efficient memory pooling
- Minimal allocation overhead

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run specific test
python test.py
```

## Citation

If you use Deepwell in your research, please cite:

```bibtex
@software{deepwell2024,
  title = {Deepwell: Production Framework for NVIDIA Blackwell GPUs},
  year = {2024},
  url = {https://github.com/yourusername/deepwell}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NVIDIA CUTLASS team for kernel libraries
- PyTorch team for framework integration
- Blackwell architecture team for hardware capabilities

## Support

For issues and questions:
- GitHub Issues: [Report bugs](https://github.com/yourusername/deepwell/issues)
- Documentation: [Full docs](https://deepwell.readthedocs.io)
- Discord: [Community chat](https://discord.gg/deepwell)