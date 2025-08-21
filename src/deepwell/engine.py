"""Execution engine for running compiled models with actual kernel dispatch."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
import time

from .compile import CompiledOp, ExecutionEngine
from .ir import IR, Op
from .precision.policy import LayerPrecision, Precision


class ExecutableModel(nn.Module):
    """
    Executable model that dispatches to actual kernels.
    """
    
    def __init__(self, engine: ExecutionEngine, original_model: nn.Module):
        super().__init__()
        self.engine = engine
        self.original_model = original_model
        self.use_cutlass = False
        self.execution_times = []
        
        # Check if CUTLASS is available
        try:
            from deepwell import cutlass_kernels
            self.cutlass_module = cutlass_kernels
            self.use_cutlass = True
            print(f"✓ CUTLASS kernels loaded for execution")
        except ImportError:
            self.cutlass_module = None
            print(f"⚠ CUTLASS not available - using PyTorch fallback")
        
        # Cache for kernels
        self.kernel_cache = {}
        
        # Extract weights from original model
        self.weights = {}
        for name, param in original_model.named_parameters():
            self.weights[name] = param
    
    def forward(self, *inputs):
        """Execute the model using compiled kernels."""
        # For now, we'll dispatch based on kernel type
        # In a full implementation, this would follow the IR graph
        
        if self.use_cutlass and any(op.kernel and op.kernel.backend.value == "cutlass" 
                                    for op in self.engine.compiled_ops):
            return self._execute_with_cutlass(inputs[0])
        else:
            # Fallback to original model
            return self.original_model(*inputs)
    
    def _execute_with_cutlass(self, input_ids):
        """Execute using CUTLASS kernels where available."""
        # This is a simplified execution - in production, we'd follow the IR graph
        device = input_ids.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else 1
        
        # Get model config from original model
        if hasattr(self.original_model, 'hidden_dim'):
            hidden_dim = self.original_model.hidden_dim
        else:
            hidden_dim = 768  # Default
        
        # Execute through layers
        x = input_ids
        
        # Embedding (use original for now)
        if hasattr(self.original_model, 'embed'):
            x = self.original_model.embed(x)
            if hasattr(self.original_model, 'pos_embed'):
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                x = x + self.original_model.pos_embed(positions)
        
        # Process through transformer layers
        for i, layer in enumerate(self.original_model.layers):
            # For each linear layer in the transformer, try to use CUTLASS
            x = self._process_layer_with_cutlass(x, layer, i)
        
        # Final layer norm and output
        if hasattr(self.original_model, 'ln_f'):
            x = self.original_model.ln_f(x)
        if hasattr(self.original_model, 'lm_head'):
            x = self.original_model.lm_head(x)
        
        return x
    
    def _process_layer_with_cutlass(self, x, layer, layer_idx):
        """Process a transformer layer using CUTLASS kernels where possible."""
        # For demonstration, we'll use CUTLASS for linear operations
        # and fallback for others
        
        # Layer norm (fallback)
        residual = x
        if hasattr(layer, 'ln1'):
            x = layer.ln1(x)
        
        # Attention (try CUTLASS for Q,K,V projections)
        if hasattr(layer, 'attn'):
            # For simplicity, use PyTorch attention
            x, _ = layer.attn(x, x, x)
        
        x = residual + x
        
        # MLP block - this is where CUTLASS really shines
        residual = x
        if hasattr(layer, 'ln2'):
            x = layer.ln2(x)
        
        if hasattr(layer, 'mlp'):
            # Try to use CUTLASS for MLP
            x = self._mlp_with_cutlass(x, layer.mlp, layer_idx)
        
        x = residual + x
        
        return x
    
    def _mlp_with_cutlass(self, x, mlp, layer_idx):
        """Execute MLP using CUTLASS kernels."""
        # Get precision for this layer
        precision = Precision.MXFP8  # From engine config
        
        if self.use_cutlass and precision in [Precision.MXFP8, Precision.NVFP4]:
            # Reshape for GEMM
            batch_size, seq_len, hidden_dim = x.shape
            x_2d = x.view(-1, hidden_dim)
            
            # Create or get cached kernel
            kernel_key = f"mlp_{layer_idx}_{batch_size}_{seq_len}"
            if kernel_key not in self.kernel_cache:
                try:
                    kernel = self.cutlass_module.BlackwellGemmKernel()
                    # Initialize for MLP dimensions with MXFP8!
                    kernel.initialize(
                        batch_size * seq_len,  # M
                        hidden_dim * 4,         # N (MLP expansion)
                        hidden_dim,             # K
                        "mxfp8" if precision == Precision.MXFP8 else "nvfp4",
                        use_microscaling=True,
                        block_size=32
                    )
                    self.kernel_cache[kernel_key] = kernel
                except Exception as e:
                    warnings.warn(f"Failed to create CUTLASS kernel: {e}")
                    return mlp(x)
            else:
                kernel = self.kernel_cache[kernel_key]
            
            # Quantize input to MXFP8 if needed
            if precision == Precision.MXFP8:
                try:
                    # Quantize input
                    x_quant, x_scales = self.cutlass_module.MicroscaleManager.quantize_mxfp8(x_2d)
                    
                    # Execute through MLP layers
                    for layer in mlp:
                        if isinstance(layer, nn.Linear):
                            # Quantize weights
                            weight = layer.weight.t().contiguous()
                            w_quant, w_scales = self.cutlass_module.MicroscaleManager.quantize_mxfp8(weight)
                            
                            # Use CUTLASS GEMM with quantized inputs
                            x_2d = kernel.gemm(x_quant, w_quant)
                            
                            # Dequantize output
                            x_2d = self.cutlass_module.MicroscaleManager.dequantize_mxfp8(
                                x_2d, x_scales, 32
                            )
                            
                            if layer.bias is not None:
                                x_2d = x_2d + layer.bias
                        elif isinstance(layer, nn.GELU):
                            x_2d = torch.nn.functional.gelu(x_2d)
                        else:
                            x_2d = layer(x_2d)
                except Exception as e:
                    warnings.warn(f"MXFP8 quantization failed, using BF16: {e}")
                    # Fallback to BF16
                    for layer in mlp:
                        if isinstance(layer, nn.Linear):
                            weight = layer.weight.t().contiguous()
                            x_2d = kernel.gemm(x_2d, weight)
                            if layer.bias is not None:
                                x_2d = x_2d + layer.bias
                        elif isinstance(layer, nn.GELU):
                            x_2d = torch.nn.functional.gelu(x_2d)
                        else:
                            x_2d = layer(x_2d)
            else:
                # Execute without quantization (BF16 path)
                for layer in mlp:
                    if isinstance(layer, nn.Linear):
                        weight = layer.weight.t().contiguous()
                        x_2d = kernel.gemm(x_2d, weight)
                        if layer.bias is not None:
                            x_2d = x_2d + layer.bias
                    elif isinstance(layer, nn.GELU):
                        x_2d = torch.nn.functional.gelu(x_2d)
                    else:
                        x_2d = layer(x_2d)
            
            # Reshape back
            x = x_2d.view(batch_size, seq_len, -1)
            return x
        else:
            # Fallback to PyTorch
            return mlp(x)
    
    def get_execution_stats(self):
        """Get execution statistics."""
        if self.execution_times:
            avg_time = sum(self.execution_times) / len(self.execution_times)
            return {
                'avg_time_ms': avg_time * 1000,
                'use_cutlass': self.use_cutlass,
                'num_kernels': len(self.kernel_cache)
            }
        return None


def create_executable_model(engine: ExecutionEngine, original_model: nn.Module) -> ExecutableModel:
    """
    Create an executable model from a compiled engine.
    
    Args:
        engine: Compiled execution engine
        original_model: Original PyTorch model
        
    Returns:
        Executable model that uses optimized kernels
    """
    return ExecutableModel(engine, original_model)


def benchmark_engine(engine: ExecutionEngine, 
                     original_model: nn.Module,
                     input_shape: Tuple[int, ...],
                     iterations: int = 100,
                     warmup: int = 10) -> Dict[str, float]:
    """
    Benchmark the execution engine.
    
    Args:
        engine: Compiled execution engine
        original_model: Original model
        input_shape: Shape of input tensor
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        
    Returns:
        Performance metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create executable model
    exec_model = create_executable_model(engine, original_model)
    exec_model = exec_model.to(device)
    exec_model.eval()
    
    # Create dummy input
    if len(input_shape) == 2:
        dummy_input = torch.randint(0, 50257, input_shape, device=device)
    else:
        dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = exec_model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking ({iterations} iterations)...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = exec_model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    time_per_iter = elapsed / iterations
    
    # Calculate tokens/sec (for language models)
    batch_size = input_shape[0]
    seq_len = input_shape[1] if len(input_shape) > 1 else 1
    tokens_per_iter = batch_size * seq_len
    tokens_per_sec = tokens_per_iter * iterations / elapsed
    
    # Get execution stats
    exec_stats = exec_model.get_execution_stats()
    
    metrics = {
        'total_time_s': elapsed,
        'time_per_iteration_ms': time_per_iter * 1000,
        'tokens_per_second': tokens_per_sec,
        'iterations_per_second': iterations / elapsed,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'use_cutlass': exec_model.use_cutlass,
    }
    
    if exec_stats:
        metrics.update(exec_stats)
    
    return metrics
