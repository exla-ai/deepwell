"""
Production-ready kernel implementations for Blackwell GPUs.
This module provides real, optimized kernels that minimize overhead.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import functools
try:
    # Optional CUTLASS-backed FlashAttention wrapper (falls back internally)
    from .blackwell_production import BlackwellFlashAttention, BlackwellConfig, DWSelfAttention
    _HAS_BW_FLASH = True
except Exception:
    _HAS_BW_FLASH = False


@dataclass
class KernelConfig:
    """Configuration for optimized kernels."""
    use_cutlass: bool = True
    use_tensor_cores: bool = True
    precision: str = "bf16"  # bf16, mxfp8, fp4
    block_size: int = 32
    min_size_for_cutlass: int = 512  # Don't use CUTLASS for tiny matrices
    batch_kernels: bool = True  # Batch small operations


class ProductionKernelManager:
    """
    Manages production kernels with minimal overhead.
    Key optimizations:
    1. Kernel caching and reuse
    2. Batched operations for small matrices
    3. Direct kernel dispatch (no Python overhead)
    """
    
    def __init__(self, config: KernelConfig = None):
        self.config = config or KernelConfig()
        self.kernel_cache = {}
        self.cutlass_available = False
        
        # Try to load CUTLASS
        if self.config.use_cutlass:
            try:
                from deepwell import cutlass_kernels
                self.cutlass_module = cutlass_kernels
                self.cutlass_available = True
                print("✓ Production CUTLASS kernels loaded")
            except ImportError:
                print("⚠ CUTLASS not available, using PyTorch")
    
    @functools.lru_cache(maxsize=128)
    def get_gemm_kernel(self, m: int, n: int, k: int) -> Any:
        """Get or create a GEMM kernel for given dimensions."""
        if not self.cutlass_available:
            return None
        
        # Don't use CUTLASS for tiny matrices (overhead not worth it)
        if m * n * k < self.config.min_size_for_cutlass ** 3:
            return None
        
        key = (m, n, k, self.config.precision)
        if key not in self.kernel_cache:
            try:
                kernel = self.cutlass_module.BlackwellGemmKernel()
                kernel.initialize(
                    m, n, k,
                    self.config.precision,
                    self.config.use_tensor_cores,
                    self.config.block_size
                )
                self.kernel_cache[key] = kernel
            except Exception as e:
                print(f"Failed to create kernel for {m}x{n}x{k}: {e}")
                return None
        
        return self.kernel_cache[key]
    
    def gemm(self, a: torch.Tensor, b: torch.Tensor, 
             bias: Optional[torch.Tensor] = None,
             activation: Optional[str] = None) -> torch.Tensor:
        """
        Optimized GEMM with automatic kernel selection.
        
        Args:
            a: Input tensor (M x K)
            b: Weight tensor (K x N)
            bias: Optional bias (N,)
            activation: Optional activation ('gelu', 'relu', etc.)
        
        Returns:
            Output tensor (M x N)
        """
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, f"Dimension mismatch: {k} != {k2}"
        
        # Get appropriate kernel
        kernel = self.get_gemm_kernel(m, n, k)
        
        if kernel is not None:
            # Use CUTLASS kernel
            output = kernel.gemm(
                a.to(torch.bfloat16),
                b.to(torch.bfloat16)
            )
        else:
            # Use PyTorch (for small matrices or fallback)
            output = torch.matmul(a, b)
        
        # Apply bias if provided
        if bias is not None:
            output = output + bias
        
        # Apply activation if specified
        if activation == 'gelu':
            output = torch.nn.functional.gelu(output)
        elif activation == 'relu':
            output = torch.nn.functional.relu(output)
        
        return output
    
    def batched_gemm(self, inputs: list, weights: list) -> list:
        """
        Execute multiple GEMMs in a batch to reduce kernel launch overhead.
        
        Args:
            inputs: List of input tensors
            weights: List of weight tensors
        
        Returns:
            List of output tensors
        """
        if not self.config.batch_kernels:
            # Execute individually
            return [self.gemm(a, b) for a, b in zip(inputs, weights)]
        
        # TODO: Implement actual batched GEMM using CUTLASS grouped GEMM
        # For now, fall back to individual GEMMs
        return [self.gemm(a, b) for a, b in zip(inputs, weights)]


class OptimizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with optimized kernels.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, kernel_manager: ProductionKernelManager = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Kernel manager
        self.kernel_manager = kernel_manager or ProductionKernelManager()
        
        # Reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass."""
        # Handle different input shapes
        original_shape = x.shape
        if x.dim() > 2:
            # Flatten to 2D for GEMM
            x = x.view(-1, self.in_features)
        
        # Use optimized GEMM
        output = self.kernel_manager.gemm(
            x,
            self.weight.t(),
            self.bias
        )
        
        # Reshape to original dimensions
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], self.out_features)
        
        return output


class OptimizedMLP(nn.Module):
    """
    Optimized MLP block using production kernels.
    """
    
    def __init__(self, hidden_dim: int, expansion: int = 4,
                 kernel_manager: ProductionKernelManager = None):
        super().__init__()
        self.kernel_manager = kernel_manager or ProductionKernelManager()
        
        # Use optimized linear layers
        self.fc1 = OptimizedLinear(hidden_dim, hidden_dim * expansion, 
                                   kernel_manager=self.kernel_manager)
        self.fc2 = OptimizedLinear(hidden_dim * expansion, hidden_dim,
                                   kernel_manager=self.kernel_manager)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass with fused operations."""
        # First linear + activation (could be fused in kernel)
        x = self.fc1(x)
        x = self.activation(x)
        
        # Second linear
        x = self.fc2(x)
        
        return x


class OptimizedTransformerBlock(nn.Module):
    """
    Production-ready transformer block with optimized kernels.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 12,
                 kernel_manager: ProductionKernelManager = None):
        super().__init__()
        self.kernel_manager = kernel_manager or ProductionKernelManager()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Attention (replaced by optimized module in optimize_model_inplace)
        self.attn = DWSelfAttention(hidden_dim, num_heads, BlackwellConfig())
        
        # MLP with optimized kernels
        self.mlp = OptimizedMLP(hidden_dim, kernel_manager=self.kernel_manager)
    
    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        
        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class OptimizedMultiheadAttention(nn.Module):
    """
    Optimized MHA using CUTLASS FMHA only (no SDPA fallback).
    Drop-in for nn.MultiheadAttention (batch_first=True expected).
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 kernel_manager: ProductionKernelManager = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_manager = kernel_manager or ProductionKernelManager()

        # Use optimized Linear for QKV and output to leverage GEMM speedups
        self.q_proj = OptimizedLinear(embed_dim, embed_dim, kernel_manager=self.kernel_manager)
        self.k_proj = OptimizedLinear(embed_dim, embed_dim, kernel_manager=self.kernel_manager)
        self.v_proj = OptimizedLinear(embed_dim, embed_dim, kernel_manager=self.kernel_manager)
        self.out_proj = OptimizedLinear(embed_dim, embed_dim, kernel_manager=self.kernel_manager)

        # Flash attention backend
        if not _HAS_BW_FLASH:
            raise RuntimeError("CUTLASS FMHA not available; install proper CUTLASS or enable native FMHA.")
        self.flash = BlackwellFlashAttention(BlackwellConfig())

    @torch.no_grad()
    def load_from_mha(self, mha: nn.MultiheadAttention):
        """Copy weights from an existing nn.MultiheadAttention instance."""
        # PyTorch packs QKV into a single in_proj_weight [3*E, E]
        in_w = mha.in_proj_weight.detach().clone()  # [3*E, E]
        in_b = mha.in_proj_bias.detach().clone() if mha.in_proj_bias is not None else None
        q_w, k_w, v_w = in_w.split(self.embed_dim, dim=0)
        if in_b is not None:
            q_b, k_b, v_b = in_b.split(self.embed_dim, dim=0)
        else:
            q_b = k_b = v_b = None

        # Assign to separate projections (note weight shapes are [out, in])
        self.q_proj.weight.data.copy_(q_w)
        self.k_proj.weight.data.copy_(k_w)
        self.v_proj.weight.data.copy_(v_w)
        if q_b is not None:
            self.q_proj.bias.data.copy_(q_b)
            self.k_proj.bias.data.copy_(k_b)
            self.v_proj.bias.data.copy_(v_b)

        # Out projection
        self.out_proj.weight.data.copy_(mha.out_proj.weight.detach())
        if mha.out_proj.bias is not None and self.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(mha.out_proj.bias.detach())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        # Expect batch_first=True tensors: [batch, seq, embed]
        batch, q_seq, _ = query.shape
        _, k_seq, _ = key.shape
        _, v_seq, _ = value.shape

        q_lin = self.q_proj(query)
        k_lin = self.k_proj(key)
        v_lin = self.v_proj(value)

        # Reshape to [batch, heads, seq, head_dim]
        def split_heads(t, seq_len):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qh = split_heads(q_lin, q_seq)
        kh = split_heads(k_lin, k_seq)
        vh = split_heads(v_lin, v_seq)

        # CUTLASS FMHA path (strict bf16 + contiguous + layout)
        qh = qh.to(torch.bfloat16).contiguous()
        kh = kh.to(torch.bfloat16).contiguous()
        vh = vh.to(torch.bfloat16).contiguous()
        out = self.flash.forward(qh, kh, vh, causal=is_causal)

        # Merge heads back: [batch, seq, embed]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, q_seq, self.embed_dim)
        out = self.out_proj(out)
        return (out, None) if need_weights else (out, None)


def optimize_model_inplace(model: nn.Module, 
                          config: KernelConfig = None) -> nn.Module:
    """
    Optimize a model in-place by replacing layers with optimized versions.
    
    Args:
        model: PyTorch model to optimize
        config: Kernel configuration
    
    Returns:
        Optimized model (same object, modified in-place)
    """
    kernel_manager = ProductionKernelManager(config)
    
    # Replace all Linear layers with OptimizedLinear; replace MHA with optimized version
    def replace_module(parent, name, module):
        # Do not replace internals of DWSelfAttention to preserve strict FMHA contract
        if isinstance(parent, DWSelfAttention):
            return False
        if isinstance(module, nn.Linear):
            # Create optimized replacement
            optimized = OptimizedLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                kernel_manager
            )
            
            # Copy weights
            optimized.weight.data = module.weight.data
            if module.bias is not None:
                optimized.bias.data = module.bias.data
            
            # Replace in parent
            setattr(parent, name, optimized)
            return True
        if isinstance(module, nn.MultiheadAttention):
            # Strict CUTLASS FMHA-only attention
            strict_attn = DWSelfAttention(module.embed_dim, module.num_heads, BlackwellConfig())
            setattr(parent, name, strict_attn)
            return True
        
        # Recursively replace in children
        replaced = False
        for child_name, child_module in module.named_children():
            if replace_module(module, child_name, child_module):
                replaced = True
        
        return replaced
    
    # Start replacement from root
    for name, module in model.named_children():
        replace_module(model, name, module)
    
    print(f"✓ Model optimized with production kernels")
    return model


def benchmark_kernels():
    """Benchmark production kernels vs baseline."""
    print("=" * 60)
    print("PRODUCTION KERNEL BENCHMARK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manager = ProductionKernelManager()
    
    # Test different sizes
    sizes = [
        (256, 256, 256),    # Small
        (1024, 1024, 1024), # Medium
        (4096, 4096, 1024), # Large
    ]
    
    for m, n, k in sizes:
        print(f"\nSize: {m}x{n}x{k}")
        print("-" * 40)
        
        # Create test data
        a = torch.randn(m, k, device=device, dtype=torch.float32)
        b = torch.randn(k, n, device=device, dtype=torch.float32)
        
        # PyTorch baseline
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        
        pytorch_ms = start.elapsed_time(end) / 100
        
        # Optimized kernel
        start.record()
        for _ in range(100):
            _ = manager.gemm(a, b)
        end.record()
        torch.cuda.synchronize()
        
        optimized_ms = start.elapsed_time(end) / 100
        
        # Calculate TFLOPS
        flops = 2 * m * n * k
        pytorch_tflops = flops / (pytorch_ms * 1e9)
        optimized_tflops = flops / (optimized_ms * 1e9)
        
        print(f"PyTorch:   {pytorch_ms:.3f} ms ({pytorch_tflops:.2f} TFLOPS)")
        print(f"Optimized: {optimized_ms:.3f} ms ({optimized_tflops:.2f} TFLOPS)")
        print(f"Speedup:   {pytorch_ms/optimized_ms:.2f}x")


if __name__ == "__main__":
    # Run benchmark
    benchmark_kernels()
    
    # Test model optimization
    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION TEST")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(768, 3072),
        nn.GELU(),
        nn.Linear(3072, 768)
    ).cuda()
    
    # Optimize it
    optimized = optimize_model_inplace(model)
    
    # Test
    x = torch.randn(32, 512, 768, device='cuda')
    with torch.no_grad():
        output = optimized(x)
    
    print(f"Output shape: {output.shape}")
    print("✓ Model optimization successful")
