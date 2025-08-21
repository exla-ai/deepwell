"""
Deepwell - Automatic optimization for PyTorch models on NVIDIA Blackwell GPUs.

Train efficiently on Blackwell via low-precision kernels, planning, and MoE support.
"""

__version__ = "0.0.1"

# Core imports
from .probe import probe, HardwareConfig, GPUInfo
from .capture import capture, ModelTracer
from .ir import IR, Op, TensorTy
from .compile import compile, ExecutionEngine, Compiler
from .engine import create_executable_model, benchmark_engine

# Precision management
from .precision.policy import (
    PrecisionPolicy,
    PrecisionConfig,
    LayerPrecision,
    Precision,
    MicroscalingConfig
)

# Kernel management  
from .kernels.registry import KernelRegistry, KernelSpec, get_registry
from .kernels.cutlass_bindings import CutlassKernel, GroupedGEMMKernel

# Import C++ extension if available
try:
    from . import cutlass_kernels
except ImportError:
    try:
        # Try alternate import path
        import deepwell.cutlass_kernels as cutlass_kernels
    except ImportError:
        import warnings
        warnings.warn("cutlass_kernels C++ extension not found. Build with: python setup.py build_ext --inplace")
        cutlass_kernels = None

# Planning (to be implemented)
def autoplan(ir: IR, 
            hw: HardwareConfig,
            seq_len: int = 2048,
            global_batch: int = 512,
            arch: str = "blackwell-sm100",
            moe: dict = None) -> dict:
    """
    Auto-plan parallelism and kernel configurations.
    
    Args:
        ir: Model intermediate representation
        hw: Hardware configuration from probe()
        seq_len: Sequence length
        global_batch: Global batch size
        arch: Target architecture
        moe: MoE configuration if applicable
        
    Returns:
        Execution plan dictionary
    """
    # Placeholder implementation
    plan = {
        'parallelism': {
            'tensor_parallel': 1,
            'pipeline_parallel': 1,
            'data_parallel': hw.total_gpus,
            'expert_parallel': 1 if not moe else moe.get('experts', 1),
        },
        'kernel_config': {
            'use_cutlass': arch.startswith('blackwell'),
            'use_grouped_gemm': moe is not None,
            'use_microscaling': 'mxfp8' in arch or 'fp4' in arch,
        },
        'memory': {
            'activation_checkpointing': seq_len > 4096,
            'optimizer_sharding': global_batch > 1024,
        }
    }
    return plan


# Dry run simulator (to be implemented)
def dryrun(engine: ExecutionEngine) -> dict:
    """
    Simulate execution to check memory and performance.
    
    Args:
        engine: Compiled execution engine
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'memory_gb': engine.estimated_memory_gb,
        'runtime_ms': engine.estimated_runtime_ms,
        'flops': engine.estimated_flops,
        'kernel_summary': engine.get_kernel_summary(),
        'validation': engine.validate(),
        'status': 'passed' if not engine.validate() else 'has_issues'
    }
    return results


# Trainer class (placeholder)
class Trainer:
    """Training orchestrator for Deepwell engines."""
    
    def __init__(self, engine: ExecutionEngine, optimizer: str = "adamw"):
        """Initialize trainer with compiled engine."""
        self.engine = engine
        self.optimizer = optimizer
        self.step = 0
        
    @classmethod
    def from_engine(cls,
                   engine: ExecutionEngine,
                   optimizer: str = "adamw_dw",
                   dataloader = None,
                   elastic: bool = False):
        """
        Create trainer from compiled engine.
        
        Args:
            engine: Compiled execution engine
            optimizer: Optimizer type
            dataloader: Training data loader
            elastic: Enable elastic training
            
        Returns:
            Configured trainer
        """
        trainer = cls(engine, optimizer)
        trainer.dataloader = dataloader
        trainer.elastic = elastic
        return trainer
    
    def fit(self, num_epochs: int = 1):
        """Run training loop."""
        import warnings
        warnings.warn("Trainer.fit() is a placeholder - actual training not implemented")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            # Training loop would go here
            self.step += 100  # Mock steps
        
        print("Training complete (mock)")


# Export function (placeholder)
def export(engine: ExecutionEngine, path: str) -> dict:
    """
    Export compiled engine for deployment.
    
    Args:
        engine: Compiled execution engine
        path: Output path for engine artifact
        
    Returns:
        Export metadata
    """
    import json
    import os
    
    # Create export directory
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Export metadata
    metadata = {
        'version': __version__,
        'sm_version': 100,  # Blackwell
        'precision': engine.precision_policy.config.default_compute.value,
        'kernel_summary': engine.get_kernel_summary(),
        'estimated_memory_gb': engine.estimated_memory_gb,
        'estimated_flops': engine.estimated_flops,
    }
    
    # Save metadata
    with open(path + '.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Engine exported to {path}")
    return metadata


# Convenience function for end-to-end compilation
def optimize_for_blackwell(model,
                          precision: str = "mxfp8",
                          seq_len: int = 2048,
                          batch_size: int = 32):
    """
    One-shot optimization for Blackwell GPUs.
    
    Args:
        model: PyTorch model
        precision: Target precision (mxfp8, nvfp4, etc.)
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        Optimized callable model
    """
    # Try simple optimization first (direct kernel replacement)
    try:
        from .kernels.production_kernels import optimize_model_inplace, KernelConfig
        import copy
        
        # Create optimized model with production kernels
        optimized_model = copy.deepcopy(model)
        
        config = KernelConfig(
            use_cutlass=True,
            precision=precision if precision in ["bf16", "mxfp8", "fp4"] else "bf16",
            min_size_for_cutlass=512
        )
        
        optimized_model = optimize_model_inplace(optimized_model, config)
        return optimized_model
        
    except Exception as e:
        # Fall back to full pipeline if simple optimization fails
        pass
    
    # Full pipeline (for when all components are ready)
    try:
        # Probe hardware
        hw = probe()
        
        # Check for Blackwell
        has_blackwell = any(gpu.is_blackwell for gpu in hw.gpus)
        if not has_blackwell:
            import warnings
            warnings.warn("No Blackwell GPU detected - optimizations may not apply")
        
        # Capture model
        ir = capture(model)
        
        # Auto-plan
        plan = autoplan(
            ir, hw,
            seq_len=seq_len,
            global_batch=batch_size * hw.total_gpus,
            arch="blackwell-sm100" if has_blackwell else "cuda"
        )
        
        # Compile
        engine = compile(
            ir,
            plan=plan,
            precision=precision,
            fallback="safe",
            sm_version=100 if has_blackwell else 90
        )
        
        # Validate
        results = dryrun(engine)
        if results['status'] == 'has_issues':
            import warnings
            for issue in results['validation']:
                warnings.warn(f"Validation issue: {issue}")
        
        # Create executable model
        from .engine import create_executable_model
        executable = create_executable_model(engine, model)
        
        return executable
        
    except Exception as e:
        # If all else fails, return original model
        import warnings
        warnings.warn(f"Optimization failed: {e}. Returning original model.")
        return model


__all__ = [
    # Core functions
    'probe',
    'capture', 
    'autoplan',
    'compile',
    'dryrun',
    'export',
    'create_executable_model',
    'benchmark_engine',
    
    # Classes
    'Trainer',
    'ExecutionEngine',
    'HardwareConfig',
    'GPUInfo',
    'IR',
    'PrecisionPolicy',
    'PrecisionConfig',
    'KernelRegistry',
    
    # C++ extension
    'cutlass_kernels',
    
    # Convenience
    'optimize_for_blackwell',
    
    # Version
    '__version__',
]
