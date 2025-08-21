"""Compilation engine for binding kernels and creating executable engines."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings

from .ir import IR, Op, TensorTy
from .precision.policy import PrecisionPolicy, LayerPrecision, Precision
from .kernels.registry import KernelRegistry, KernelSpec, get_registry
from .kernels.cutlass_bindings import CutlassKernel, GroupedGEMMKernel, CUTLASS_AVAILABLE


@dataclass
class CompiledOp:
    """A compiled operation with bound kernel."""
    op: Op
    kernel: KernelSpec
    precision: LayerPrecision
    backend_op: Optional[Any] = None  # Actual kernel implementation
    stream_id: int = 0  # CUDA stream for execution
    
    
@dataclass
class ExecutionEngine:
    """
    Executable engine with compiled operations.
    Ready for training/inference execution.
    """
    ir: IR
    compiled_ops: List[CompiledOp]
    precision_policy: PrecisionPolicy
    kernel_registry: KernelRegistry
    memory_pool: Optional[Any] = None
    streams: List[Any] = field(default_factory=list)
    
    # Performance metrics
    estimated_flops: float = 0
    estimated_memory_gb: float = 0
    estimated_runtime_ms: float = 0
    
    def __call__(self, *args, **kwargs):
        """Call-through to a simple PyTorch fallback until IR executor is ready."""
        # Minimal executable behavior: forward the first tensor through a no-op to validate callability
        if args and isinstance(args[0], __import__('torch').Tensor):
            return args[0]
        return args

    def validate(self) -> List[str]:
        """Validate the compiled engine."""
        issues = []
        
        # Check all ops have kernels
        for cop in self.compiled_ops:
            if cop.kernel is None:
                issues.append(f"Op {cop.op.id} has no kernel assigned")
        
        # Check memory requirements
        if self.estimated_memory_gb > 192:  # B200 has 192GB HBM
            issues.append(f"Memory requirement {self.estimated_memory_gb:.1f}GB exceeds B200 capacity")
        
        return issues
    
    def get_kernel_summary(self) -> Dict[str, int]:
        """Get summary of kernel usage."""
        summary = {}
        for cop in self.compiled_ops:
            if cop.kernel:
                backend = cop.kernel.backend.value
                summary[backend] = summary.get(backend, 0) + 1
        return summary


class Compiler:
    """
    Compiles IR to executable engine with kernel bindings.
    """
    
    def __init__(self,
                sm_version: int = 100,  # Default to Blackwell SM100
                use_cutlass: bool = True,
                use_te: bool = False,  # Transformer Engine
                enable_fusion: bool = True):
        """
        Initialize compiler.
        
        Args:
            sm_version: Target SM version (100 for Blackwell)
            use_cutlass: Enable CUTLASS kernels
            use_te: Enable Transformer Engine
            enable_fusion: Enable kernel fusion
        """
        self.sm_version = sm_version
        self.use_cutlass = use_cutlass
        self.use_te = use_te
        self.enable_fusion = enable_fusion
        self.kernel_registry = get_registry()
        
        # Kernel implementations (would be actual kernel objects)
        self.cutlass_kernel = CutlassKernel() if use_cutlass else None
        self.grouped_gemm_kernel = None  # Created on demand for MoE
        
    def compile(self,
               ir: IR,
               precision_policy: PrecisionPolicy,
               plan: Optional[Dict[str, Any]] = None) -> ExecutionEngine:
        """
        Compile IR to executable engine.
        
        Args:
            ir: Intermediate representation
            precision_policy: Precision assignments for layers
            plan: Optional execution plan with parallelism strategy
            
        Returns:
            Compiled execution engine
        """
        compiled_ops = []
        
        # Process each operation
        for op in ir.ops:
            # Get precision for this layer
            layer_name = op.attrs.get('source', op.id)
            if layer_name not in precision_policy.layer_precisions:
                # Assign precision if not already done
                layer_prec = precision_policy.assign_layer_precision(
                    layer_name=layer_name,
                    layer_type=op.kind,
                    param_count=None  # Would calculate from IR
                )
            else:
                layer_prec = precision_policy.layer_precisions[layer_name]
            
            # Find best kernel for this operation
            kernel = self._select_kernel(op, layer_prec)
            
            if kernel is None:
                warnings.warn(f"No kernel found for op {op.id}, using fallback")
                # Use PyTorch fallback
                kernel = self.kernel_registry.find_kernel(
                    "linear", "fp32", 0
                )
            
            # Create backend operation (actual kernel instance)
            backend_op = self._create_backend_op(op, kernel, layer_prec)
            
            # Determine execution stream (for overlap)
            stream_id = self._assign_stream(op, plan)
            
            compiled_op = CompiledOp(
                op=op,
                kernel=kernel,
                precision=layer_prec,
                backend_op=backend_op,
                stream_id=stream_id
            )
            compiled_ops.append(compiled_op)
        
        # Create execution engine
        engine = ExecutionEngine(
            ir=ir,
            compiled_ops=compiled_ops,
            precision_policy=precision_policy,
            kernel_registry=self.kernel_registry
        )
        
        # Estimate performance metrics
        self._estimate_performance(engine)
        
        return engine
    
    def _select_kernel(self, op: Op, precision: LayerPrecision) -> Optional[KernelSpec]:
        """Select best kernel for operation."""
        # Map op kind to kernel op type
        op_type_map = {
            'linear': 'gemm',
            'attention': 'attention',
            'mlp': 'gemm',
            'norm': 'norm',
            'embed': 'embed',
        }
        
        kernel_op_type = op_type_map.get(op.kind, 'gemm')
        
        # Check if this is a grouped operation (for MoE)
        if 'num_experts' in op.attrs:
            kernel_op_type = 'grouped_gemm'
        
        # Get precision string
        prec_str = precision.compute_dtype.value
        
        # Find kernel
        kernel = self.kernel_registry.find_kernel(
            op_type=kernel_op_type,
            precision=prec_str,
            sm_version=self.sm_version,
            prefer_fused=self.enable_fusion,
            require_microscaling=precision.microscaling is not None
        )
        
        return kernel
    
    def _create_backend_op(self, 
                          op: Op,
                          kernel: KernelSpec,
                          precision: LayerPrecision) -> Optional[Any]:
        """Create actual backend operation."""
        if kernel is None:
            return None
        
        # Based on kernel backend, create appropriate operation
        if kernel.backend.value == "cutlass":
            if CUTLASS_AVAILABLE:
                # Use actual CUTLASS kernels
                if kernel.op_type == "grouped_gemm":
                    # Create grouped GEMM kernel for MoE
                    if self.grouped_gemm_kernel is None:
                        num_experts = op.attrs.get('num_experts', 8)
                        expert_dim = op.attrs.get('expert_dim', 4096)
                        hidden_dim = op.attrs.get('hidden_dim', 16384)
                        self.grouped_gemm_kernel = GroupedGEMMKernel(
                            num_experts=num_experts,
                            expert_dim=expert_dim,
                            hidden_dim=hidden_dim
                        )
                    return self.grouped_gemm_kernel
                else:
                    # Regular GEMM kernel
                    if self.cutlass_kernel is None:
                        self.cutlass_kernel = CutlassKernel()
                    return self.cutlass_kernel
            else:
                # Fallback if CUTLASS not built
                import warnings
                warnings.warn(
                    "CUTLASS kernels selected but extension not available. "
                    "Build with: python setup.py build_ext --inplace"
                )
                return None
        elif kernel.backend.value == "transformer_engine":
            # Would create TE operation
            return None
        elif kernel.backend.value == "cublas":
            # Would create cuBLAS operation
            return None
        else:
            # PyTorch or other fallback
            return None
    
    def _assign_stream(self, op: Op, plan: Optional[Dict[str, Any]]) -> int:
        """
        Assign CUDA stream for operation execution.
        Enables overlap of computation and communication.
        """
        if plan and 'stream_assignment' in plan:
            # Use plan's stream assignment
            return plan['stream_assignment'].get(op.id, 0)
        
        # Simple heuristic: alternate streams for independent ops
        if op.kind == 'attention':
            return 1  # Attention on stream 1
        elif op.kind == 'mlp':
            return 2  # MLP on stream 2
        else:
            return 0  # Default stream
    
    def _estimate_performance(self, engine: ExecutionEngine):
        """Estimate performance metrics for compiled engine."""
        total_flops = 0
        total_memory = 0
        total_time = 0
        
        for cop in engine.compiled_ops:
            op = cop.op
            
            # Estimate FLOPs (simplified)
            if op.kind in ['linear', 'mlp']:
                # Assume shapes from attributes
                m = op.attrs.get('batch_size', 1) * op.attrs.get('seq_len', 1024)
                n = op.attrs.get('out_features', 4096)
                k = op.attrs.get('in_features', 4096)
                flops = 2 * m * n * k
                total_flops += flops
                
                # Estimate memory (weights + activations)
                precision_bits = cop.precision.compute_dtype.bits
                weight_mem = n * k * precision_bits / 8 / 1e9
                act_mem = m * (n + k) * precision_bits / 8 / 1e9
                total_memory += weight_mem + act_mem
                
                # Estimate time (very simplified)
                if cop.kernel and cop.kernel.sm_version >= 100:
                    # Blackwell performance
                    tflops = 5000 if cop.precision.compute_dtype == Precision.MXFP8 else 2500
                else:
                    tflops = 1000  # Non-Blackwell
                
                time_ms = (flops / 1e12) / tflops * 1000
                total_time += time_ms
        
        engine.estimated_flops = total_flops
        engine.estimated_memory_gb = total_memory
        engine.estimated_runtime_ms = total_time


def compile(ir: IR,
           plan: Optional[Dict[str, Any]] = None,
           precision: str = "mxfp8",
           fallback: str = "safe",
           sm_version: int = 100) -> ExecutionEngine:
    """
    Main compilation interface.
    
    Args:
        ir: Model IR
        plan: Execution plan
        precision: Target precision (mxfp8, nvfp4, etc.)
        fallback: Fallback strategy (safe, aggressive, none)
        sm_version: Target SM version
        
    Returns:
        Compiled execution engine
    """
    # Create precision policy
    from .precision.policy import PrecisionConfig, Precision
    
    prec_map = {
        "mxfp8": Precision.MXFP8,
        "nvfp4": Precision.NVFP4,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
    }
    
    config = PrecisionConfig(
        default_compute=prec_map.get(precision, Precision.MXFP8),
        enable_auto_fallback=(fallback == "safe")
    )
    
    precision_policy = PrecisionPolicy(config)
    
    # Create compiler
    compiler = Compiler(
        sm_version=sm_version,
        use_cutlass=(sm_version >= 100),  # Use CUTLASS on Blackwell
        use_te=False,  # Would enable based on availability
        enable_fusion=True
    )
    
    # Compile
    engine = compiler.compile(ir, precision_policy, plan)
    
    # Validate
    issues = engine.validate()
    if issues:
        for issue in issues:
            warnings.warn(f"Compilation issue: {issue}")
    
    return engine
