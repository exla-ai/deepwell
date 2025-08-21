import torch

# This module provides a pure Python fallback implementation for environments
# where the compiled CUTLASS extension is unavailable.  Importers can check for
# this flag to determine that no high performance kernels are present.
CUTLASS_PYTHON_FALLBACK = True


class BlackwellGemmKernel:
    """Python fallback when compiled CUTLASS extension is unavailable."""

    def __init__(self):
        self.initialized = False
        self.tmem_residency = True

    def initialize(self, m, n, k, precision="bf16", use_microscaling=False, block_size=32):
        self.m = m
        self.n = n
        self.k = k
        self.precision = precision
        self.use_microscaling = use_microscaling
        self.block_size = block_size
        self.initialized = True

    def enable_tmem_residency(self, enable=True):
        self.tmem_residency = enable

    def gemm(self, a, b, c=None, alpha=1.0, beta=0.0):
        if c is not None:
            return alpha * torch.matmul(a, b) + beta * c
        else:
            return alpha * torch.matmul(a, b)


class BlackwellGroupedGemmKernel:
    """Simplified grouped GEMM kernel for environments without the C++ extension."""

    def __init__(self):
        self.initialized = False
        self.expert_parallel = False

    def initialize(self, problem_sizes, precision="bf16", use_tcgen05=False):
        self.problem_sizes = problem_sizes
        self.precision = precision
        self.use_tcgen05 = use_tcgen05
        self.initialized = True

    def grouped_gemm(self, inputs, weights):
        outputs = []
        for x, w in zip(inputs, weights):
            outputs.append(torch.matmul(x, w))
        return outputs

    def set_expert_parallel_strategy(self, enable):
        self.expert_parallel = enable
