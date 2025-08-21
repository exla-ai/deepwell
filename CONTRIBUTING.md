# Contributing to Deepwell

Thank you for your interest in contributing to Deepwell! This document provides guidelines and instructions for contributing to the project.

## Project Goals

Deepwell aims to provide production-ready optimization for NVIDIA Blackwell GPUs (B100/B200) using tcgen05 Tensor Core instructions. We focus on:
- Performance without sacrificing accuracy
- Clean, maintainable code
- Comprehensive documentation
- Robust testing

## Getting Started

### Prerequisites
- NVIDIA GPU (Blackwell preferred, but Hopper/Ada for development)
- CUDA 12.4+
- Python 3.10+
- PyTorch 2.2+

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/deepwell.git
cd deepwell

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Development Process

### 1. Find or Create an Issue
- Check existing issues before starting work
- For new features, create an issue for discussion
- For bugs, provide reproduction steps

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Your Changes
- Follow the code style guide (below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Submit a Pull Request
- Push your branch to your fork
- Create a PR against `main` branch
- Fill out the PR template
- Link related issues

## Code Style

### Python
- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

```python
# Good
def quantize_tensor(
    tensor: torch.Tensor,
    precision: Precision = Precision.MXFP8,
    block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to specified precision.
    
    Args:
        tensor: Input tensor to quantize
        precision: Target precision
        block_size: Block size for microscaling
        
    Returns:
        Tuple of (quantized_tensor, scale_factors)
    """
    ...
```

### C++/CUDA
- Follow NVIDIA CUDA coding guidelines
- Use meaningful kernel names
- Document shared memory usage

```cpp
// Good
template <typename T>
__global__ void blackwell_mxfp8_gemm_kernel(
    T* __restrict__ d,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M, int N, int K
) {
    // Kernel implementation
}
```

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_kernels.py

# Run with coverage
pytest --cov=deepwell tests/

# Run benchmarks
python benchmarks/blackwell_speedup.py
```

### Writing Tests
- Test files should start with `test_`
- Use descriptive test names
- Include edge cases
- Mock hardware-specific features when needed

```python
def test_mxfp8_quantization_accuracy():
    """Test that MXFP8 quantization maintains acceptable accuracy."""
    tensor = torch.randn(128, 256, dtype=torch.bfloat16)
    quantized, scales = quantize_mxfp8(tensor)
    
    dequantized = dequantize_mxfp8(quantized, scales)
    error = torch.abs(tensor - dequantized).max()
    
    assert error < 0.1, f"Quantization error {error} exceeds threshold"
```

## Documentation

### Docstrings
Use Google-style docstrings:

```python
def optimize_model(
    model: nn.Module,
    config: OptimizationConfig
) -> OptimizedModel:
    """
    Optimize PyTorch model for Blackwell execution.
    
    Args:
        model: PyTorch model to optimize
        config: Optimization configuration
        
    Returns:
        Optimized model ready for Blackwell
        
    Raises:
        ValueError: If model architecture is unsupported
        
    Example:
        >>> model = MyTransformer()
        >>> config = OptimizationConfig(precision="mxfp8")
        >>> opt_model = optimize_model(model, config)
    """
```

### README Updates
- Update README.md for user-facing changes
- Update README_V1.md for technical details
- Include examples for new features

## Architecture Guidelines

### Adding New Kernels
1. Define kernel in `csrc/` directory
2. Add Python bindings in `csrc/python_bindings.cpp`
3. Create wrapper in `src/deepwell/kernels/`
4. Register in kernel registry
5. Add tests

### Adding New Precisions
1. Add to `Precision` enum in `precision/policy.py`
2. Implement quantization/dequantization
3. Add kernel support
4. Update documentation

### tcgen05 Operations
When adding tcgen05 operations:
1. Follow CUTLASS Python API conventions
2. Document PTX instruction mapping
3. Provide fallback for non-Blackwell GPUs
4. Include performance benchmarks

## Debugging

### Common Issues

#### CUDA Memory Errors
```bash
# Run with memory checking
CUDA_LAUNCH_BLOCKING=1 python your_script.py

# Enable device-side assertions
export TORCH_USE_CUDA_DSA=1
```

#### Build Issues
```bash
# Clean build
rm -rf build/ dist/ *.egg-info
python setup.py clean --all
python setup.py build_ext --inplace
```

## Performance Guidelines

### Benchmarking
- Always benchmark on actual Blackwell hardware when possible
- Compare against PyTorch baseline
- Report tokens/sec and TFLOPS
- Include warmup iterations

```python
def benchmark_kernel(kernel, inputs, warmup=10, iterations=100):
    """Benchmark kernel performance."""
    # Warmup
    for _ in range(warmup):
        kernel(*inputs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        kernel(*inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations
```

## Security

### Reporting Security Issues
- Do NOT open public issues for security vulnerabilities
- Email security@deepwell.ai with details
- Include steps to reproduce

### Code Security
- Validate all inputs
- Avoid unsafe memory operations
- No hardcoded credentials
- Sanitize file paths

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## Contact

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Email: contributors@deepwell.ai

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Deepwell! Together we're accelerating AI on NVIDIA Blackwell.
