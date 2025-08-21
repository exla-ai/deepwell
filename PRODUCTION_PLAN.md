# Deepwell Production Plan for GPT-6 Scale Training

## Current Status
- ❌ **Correctness Issue**: Max difference 160.0 (completely broken)
- ❌ **Performance**: 1.53x slower than torch.compile on models
- ⚠️ **Using cuBLAS fallback** instead of real Blackwell kernels

## Critical Fixes Needed

### 1. Fix Correctness (URGENT)
The cuBLAS kernel has wrong parameters causing 160.0 max difference.

**Root Cause**: Incorrect transpose/leading dimension handling in cuBLAS call.

**Fix**: Proper row-major to column-major conversion:
```cpp
// CORRECT cuBLAS call for PyTorch row-major tensors
cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both
    M, N, K,                    // Output dimensions
    &alpha,
    B, CUDA_R_16BF, ldb,        // B transposed
    A, CUDA_R_16BF, lda,        // A transposed  
    &beta,
    C, CUDA_R_16BF, ldc,        // C output
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
);
```

### 2. Integrate Real Blackwell Kernels

Based on CUTLASS examples, we need:

#### A. Example 73: Blackwell GEMM with Preferred Cluster
```cpp
// From examples/73_blackwell_gemm_preferred_cluster
using GemmKernel = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::RowMajor,                 // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OpClass
    cutlass::arch::Sm100,                       // ArchTag (Blackwell)
    cutlass::gemm::GemmShape<256, 128, 64>,    // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,      // WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>,       // InstructionShape (tcgen05.mma)
    // Cluster configuration
    cutlass::gemm::ClusterShape<2, 1, 1>       // Preferred cluster
>;
```

#### B. Example 74: Stream-K Scheduler
```cpp
// From examples/74_blackwell_gemm_streamk
using StreamKScheduler = cutlass::gemm::kernel::StreamKScheduler<
    cutlass::arch::Sm100,
    256,  // ThreadblockShapeM
    128,  // ThreadblockShapeN
    64    // ThreadblockShapeK
>;
```

#### C. Example 75: Grouped GEMM for MoE
```cpp
// From examples/75_blackwell_grouped_gemm
using GroupedGemmKernel = cutlass::gemm::device::GroupedGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>  // tcgen05.mma
>;
```

#### D. Example 77: Flash Attention
```cpp
// From examples/77_blackwell_fmha
using FlashAttentionKernel = cutlass::gemm::device::FlashAttention<
    cutlass::half_t,                         // Element
    cutlass::arch::Sm100,                    // Architecture
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape
    true                                      // Causal mask
>;
```

### 3. Implement Operator Fusion

To beat torch.compile, we need kernel fusion:

#### A. Fused Linear + GELU
```cpp
// Single kernel that does: out = GELU(A @ B + bias)
template<typename Epilogue>
struct FusedLinearGELU {
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementCompute,
        cutlass::epilogue::thread::ScaleType::Nothing
    >;
};
```

#### B. Fused LayerNorm + Linear
```cpp
// Single kernel: out = Linear(LayerNorm(x))
template<typename MainloopKernel>
struct FusedLayerNormLinear {
    // Custom prologue that applies LayerNorm during loading
    using Prologue = cutlass::transform::collective::LayerNorm<
        ElementA,
        cutlass::layout::RowMajor
    >;
};
```

### 4. Graph-Level Optimization

Create a graph optimizer that:
1. **Captures model graph** (like torch.compile)
2. **Identifies fusion opportunities**
3. **Selects optimal kernels** for each operation
4. **Minimizes memory transfers**

```python
class BlackwellGraphOptimizer:
    def optimize(self, graph: fx.GraphModule) -> fx.GraphModule:
        # Pattern match for fusion opportunities
        patterns = [
            (Linear + GELU) -> FusedLinearGELU,
            (LayerNorm + Linear) -> FusedLayerNormLinear,
            (Q @ K^T) -> FlashAttention,
        ]
        
        # Replace patterns with fused kernels
        for pattern in patterns:
            graph = replace_pattern(graph, pattern)
        
        # Select optimal kernel for each op
        for node in graph.nodes:
            if node.op == 'call_module':
                select_blackwell_kernel(node)
        
        return graph
```

## Implementation Priority

1. **IMMEDIATE**: Fix correctness issue in cuBLAS kernel
2. **TODAY**: Integrate example 73 (preferred cluster GEMM)
3. **TOMORROW**: Add example 77 (Flash Attention)
4. **THIS WEEK**: Implement fusion (Linear+GELU, LayerNorm+Linear)
5. **NEXT WEEK**: Full graph optimization

## Performance Targets

On B200 GPU:
- **GEMM**: >1000 TFLOPS (real, not measurement artifact)
- **Model**: >2x faster than torch.compile
- **MoE**: >3x faster with grouped GEMM
- **Attention**: >4x faster with Flash Attention

## Testing Plan

```bash
# On your Mac:
git add -A && git commit -m "Production Blackwell kernels"
git push

# On B200 machine:
git pull
python setup.py build_ext --inplace
python test_production.py  # Verify correctness first
python benchmark.py        # Then benchmark
```

## Success Criteria

✅ Correctness: Max difference < 0.001
✅ GEMM: Beats torch.compile by >2x
✅ Model: Beats torch.compile by >1.5x
✅ Scales to GPT-6 size models
