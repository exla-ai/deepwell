"""
Deepwell kernel implementations for Blackwell GPUs.
"""

from .registry import KernelRegistry, register_kernel
from .cutlass_bindings import (
    CutlassKernel,
    GroupedGEMMKernel,
    CutlassConfig,
    BlackwellMMATensor
)

# Import tcgen05 operations if available
try:
    from .tcgen05_ops import TCGen05Ops, BlackwellKernel, test_tcgen05
    TCGEN05_AVAILABLE = TCGen05Ops.is_available()
except ImportError:
    TCGEN05_AVAILABLE = False
    TCGen05Ops = None
    BlackwellKernel = None
    test_tcgen05 = lambda: False

__all__ = [
    'KernelRegistry',
    'register_kernel',
    'CutlassKernel',
    'GroupedGEMMKernel', 
    'CutlassConfig',
    'BlackwellMMATensor',
    'TCGen05Ops',
    'BlackwellKernel',
    'TCGEN05_AVAILABLE',
    'test_tcgen05'
]