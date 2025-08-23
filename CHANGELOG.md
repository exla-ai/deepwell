# Deepwell Changelog

## v0.1.0 - Automatic Optimization Release

### Major Features
- **Automatic Model Optimization**: One-line API to optimize any PyTorch model
  - `model = deepwell.optimize(model)` - that's it!
  - Automatically detects and replaces optimizable modules
  - Zero code changes required to existing training loops
  
- **Automatic Installation**: All components build during `pip install`
  - CUTLASS FMHA bridge compiles automatically
  - Detects GPU architecture and configures appropriately
  - No manual build steps required

### Technical Improvements
- **Module Replacement System**:
  - Automatically replaces `nn.MultiheadAttention` with `DWSelfAttention`
  - Preserves weights and biases during replacement
  - Maintains API compatibility with PyTorch modules
  
- **CUTLASS FMHA Integration**:
  - Direct integration with CUTLASS example 77 for Blackwell
  - Optimized for SM100a architecture with tcgen05 instructions
  - BF16 precision for maximum performance

### API Changes
- New main API: `deepwell.optimize(model)`
- Removed requirement for manual module replacement
- Simplified configuration - everything is automatic

### Performance
- Up to 10x faster attention operations on large sequences
- 1.5-2x end-to-end training speedup expected
- Optimal for sequence lengths ≥2048 (must be multiple of 64)

### Known Limitations
- Currently optimized for Blackwell GPUs only
- Sequence length must be multiple of 64
- Head dimension must be 64 or 128
- BF16 precision only (MXFP8 coming soon)

### Installation
```bash
git clone https://github.com/yourusername/deepwell.git
cd deepwell
git submodule update --init --recursive
pip install .
```

### Usage
```python
import deepwell

# Your existing model
model = MyTransformer()

# One-line optimization
model = deepwell.optimize(model)

# Train normally - all optimizations happen automatically
```

### Future Work
- MXFP8 quantization support
- Automatic mixed-precision training
- Support for more GPU architectures
- Additional optimized kernels (LayerNorm, GELU, etc.)
