"""Kernel registry for managing and selecting optimal kernels."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import warnings


class KernelBackend(Enum):
    """Available kernel backends."""
    CUTLASS = "cutlass"  # NVIDIA CUTLASS kernels
    TE = "transformer_engine"  # NVIDIA Transformer Engine
    CUBLAS = "cublas"  # cuBLAS/cuBLASLt
    TRITON = "triton"  # Triton kernels
    TORCH = "torch"  # PyTorch native
    CUSTOM = "custom"  # Custom CUDA kernels


@dataclass
class KernelSpec:
    """Specification for a kernel implementation."""
    name: str
    backend: KernelBackend
    op_type: str  # 'gemm', 'grouped_gemm', 'attention', etc.
    sm_version: int  # Minimum SM version required
    precision_support: List[str]  # Supported precisions
    has_microscaling: bool = False
    is_fused: bool = False  # Fused epilogue operations
    max_batch_size: Optional[int] = None
    tile_sizes: Optional[List[Tuple[int, int, int]]] = None  # M, N, K tiles
    
    def supports_precision(self, precision: str) -> bool:
        """Check if kernel supports given precision."""
        return precision.lower() in [p.lower() for p in self.precision_support]
    
    def supports_sm(self, sm_version: int) -> bool:
        """Check if kernel supports given SM version."""
        return sm_version >= self.sm_version


class KernelRegistry:
    """
    Registry for managing kernel implementations.
    Selects optimal kernels based on hardware and precision requirements.
    """
    
    def __init__(self):
        """Initialize kernel registry."""
        self.kernels: Dict[str, List[KernelSpec]] = {}
        self._register_default_kernels()
        
    def _register_default_kernels(self):
        """Register default kernel implementations."""
        
        # Blackwell CUTLASS kernels with MXFP8/FP4
        self.register(KernelSpec(
            name="cutlass_sm100_mxfp8_gemm",
            backend=KernelBackend.CUTLASS,
            op_type="gemm",
            sm_version=100,  # Blackwell SM100
            precision_support=["mxfp8", "e4m3", "e5m2"],
            has_microscaling=True,
            is_fused=True,
            tile_sizes=[(256, 128, 64), (128, 256, 64), (128, 128, 64)]
        ))
        
        self.register(KernelSpec(
            name="cutlass_sm100_nvfp4_gemm",
            backend=KernelBackend.CUTLASS,
            op_type="gemm",
            sm_version=100,
            precision_support=["nvfp4", "mxfp4"],
            has_microscaling=True,
            is_fused=True,
            tile_sizes=[(256, 128, 128), (128, 256, 128)]
        ))
        
        # Grouped GEMM for MoE
        self.register(KernelSpec(
            name="cutlass_sm100_grouped_gemm_mxfp8",
            backend=KernelBackend.CUTLASS,
            op_type="grouped_gemm",
            sm_version=100,
            precision_support=["mxfp8"],
            has_microscaling=True,
            is_fused=True,
            max_batch_size=128  # Max experts in single kernel
        ))
        
        # Hopper kernels (fallback for non-Blackwell)
        self.register(KernelSpec(
            name="cutlass_sm90_fp8_gemm",
            backend=KernelBackend.CUTLASS,
            op_type="gemm",
            sm_version=90,  # Hopper
            precision_support=["fp8", "e4m3", "e5m2"],
            has_microscaling=False,
            is_fused=True
        ))
        
        # Transformer Engine kernels
        self.register(KernelSpec(
            name="te_mxfp8_linear",
            backend=KernelBackend.TE,
            op_type="linear",
            sm_version=90,  # Works on Hopper and Blackwell
            precision_support=["mxfp8", "fp8", "fp16", "bf16"],
            has_microscaling=True,
            is_fused=True
        ))
        
        self.register(KernelSpec(
            name="te_attention",
            backend=KernelBackend.TE,
            op_type="attention",
            sm_version=80,  # Ampere+
            precision_support=["fp16", "bf16", "fp8"],
            has_microscaling=False,
            is_fused=True
        ))
        
        # cuBLAS fallbacks
        self.register(KernelSpec(
            name="cublas_gemm",
            backend=KernelBackend.CUBLAS,
            op_type="gemm",
            sm_version=70,  # Volta+
            precision_support=["fp32", "fp16", "bf16"],
            has_microscaling=False,
            is_fused=False
        ))
        
        self.register(KernelSpec(
            name="cublas_grouped_gemm",
            backend=KernelBackend.CUBLAS,
            op_type="grouped_gemm",
            sm_version=70,
            precision_support=["fp32", "fp16", "bf16"],
            has_microscaling=False,
            is_fused=False
        ))
        
        # PyTorch native fallback
        self.register(KernelSpec(
            name="torch_linear",
            backend=KernelBackend.TORCH,
            op_type="linear",
            sm_version=0,  # Any GPU
            precision_support=["fp32", "fp16", "bf16"],
            has_microscaling=False,
            is_fused=False
        ))
    
    def register(self, kernel: KernelSpec):
        """
        Register a kernel implementation.
        
        Args:
            kernel: Kernel specification to register
        """
        if kernel.op_type not in self.kernels:
            self.kernels[kernel.op_type] = []
        self.kernels[kernel.op_type].append(kernel)
    
    def find_kernel(self,
                   op_type: str,
                   precision: str,
                   sm_version: int,
                   prefer_fused: bool = True,
                   require_microscaling: bool = False) -> Optional[KernelSpec]:
        """
        Find the best kernel for given requirements.
        
        Args:
            op_type: Operation type (gemm, attention, etc.)
            precision: Required precision
            sm_version: SM version of target GPU
            prefer_fused: Prefer fused kernels
            require_microscaling: Require microscaling support
            
        Returns:
            Best matching kernel or None
        """
        if op_type not in self.kernels:
            return None
        
        candidates = []
        
        for kernel in self.kernels[op_type]:
            # Check SM version
            if not kernel.supports_sm(sm_version):
                continue
            
            # Check precision support
            if not kernel.supports_precision(precision):
                continue
            
            # Check microscaling requirement
            if require_microscaling and not kernel.has_microscaling:
                continue
            
            candidates.append(kernel)
        
        if not candidates:
            return None
        
        # Sort by preference
        # Priority: 1) Blackwell kernels, 2) Fused, 3) CUTLASS, 4) TE
        def score_kernel(k: KernelSpec) -> int:
            score = 0
            if k.sm_version >= 100:  # Blackwell
                score += 1000
            if k.is_fused and prefer_fused:
                score += 100
            if k.backend == KernelBackend.CUTLASS:
                score += 50
            elif k.backend == KernelBackend.TE:
                score += 40
            if k.has_microscaling:
                score += 20
            return score
        
        candidates.sort(key=score_kernel, reverse=True)
        return candidates[0]
    
    def find_fallback_chain(self,
                           op_type: str,
                           precision: str,
                           sm_version: int) -> List[KernelSpec]:
        """
        Find a chain of fallback kernels from best to worst.
        
        Args:
            op_type: Operation type
            precision: Preferred precision
            sm_version: SM version
            
        Returns:
            List of kernels from best to fallback
        """
        chain = []
        
        # Try exact match first
        kernel = self.find_kernel(op_type, precision, sm_version)
        if kernel:
            chain.append(kernel)
        
        # Try without microscaling
        if precision in ["mxfp8", "nvfp4", "mxfp4"]:
            kernel = self.find_kernel(op_type, "fp8", sm_version, require_microscaling=False)
            if kernel and kernel not in chain:
                chain.append(kernel)
        
        # Try higher precision fallbacks
        fallback_precisions = {
            "nvfp4": ["mxfp8", "fp8", "fp16", "bf16", "fp32"],
            "mxfp4": ["mxfp8", "fp8", "fp16", "bf16", "fp32"],
            "mxfp8": ["fp8", "fp16", "bf16", "fp32"],
            "fp8": ["fp16", "bf16", "fp32"],
            "fp16": ["bf16", "fp32"],
            "bf16": ["fp32"],
        }
        
        if precision in fallback_precisions:
            for fallback_prec in fallback_precisions[precision]:
                kernel = self.find_kernel(op_type, fallback_prec, sm_version)
                if kernel and kernel not in chain:
                    chain.append(kernel)
        
        # Final fallback to PyTorch
        if op_type in ["gemm", "linear"]:
            kernel = self.find_kernel("linear", "fp32", 0)  # PyTorch fallback
            if kernel and kernel not in chain:
                chain.append(kernel)
        
        return chain
    
    def get_kernel_info(self, kernel_name: str) -> Optional[KernelSpec]:
        """Get kernel info by name."""
        for kernels in self.kernels.values():
            for kernel in kernels:
                if kernel.name == kernel_name:
                    return kernel
        return None
    
    def list_kernels(self, op_type: Optional[str] = None) -> List[KernelSpec]:
        """List all registered kernels, optionally filtered by op type."""
        if op_type:
            return self.kernels.get(op_type, [])
        
        all_kernels = []
        for kernels in self.kernels.values():
            all_kernels.extend(kernels)
        return all_kernels
    
    def validate_for_hardware(self, sm_version: int) -> Dict[str, List[str]]:
        """
        Validate which kernels are available for given hardware.
        
        Args:
            sm_version: SM version of target GPU
            
        Returns:
            Dictionary mapping op types to available kernel names
        """
        available = {}
        
        for op_type, kernels in self.kernels.items():
            available[op_type] = []
            for kernel in kernels:
                if kernel.supports_sm(sm_version):
                    available[op_type].append(kernel.name)
        
        return available


# Global registry instance
_global_registry = KernelRegistry()


def get_registry() -> KernelRegistry:
    """Get the global kernel registry."""
    return _global_registry


def register_kernel(kernel: KernelSpec):
    """Register a kernel in the global registry."""
    _global_registry.register(kernel)
