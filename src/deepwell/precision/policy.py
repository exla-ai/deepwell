"""Precision policy for MXFP8/FP4 with Blackwell microscaling support."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple
from enum import Enum
import warnings


class Precision(Enum):
    """Supported precision formats."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MXFP8 = "mxfp8"  # Blackwell microscaled FP8 (E4M3 with block scaling)
    NVFP4 = "nvfp4"  # NVIDIA's FP4 format for Blackwell
    MXFP4 = "mxfp4"  # OCP standard microscaled FP4
    MXFP6 = "mxfp6"  # OCP standard microscaled FP6
    
    @property
    def is_microscaled(self) -> bool:
        """Check if this precision uses microscaling."""
        return self in [Precision.MXFP8, Precision.NVFP4, Precision.MXFP4, Precision.MXFP6]
    
    @property
    def bits(self) -> int:
        """Number of bits for this precision."""
        mapping = {
            Precision.FP32: 32,
            Precision.FP16: 16,
            Precision.BF16: 16,
            Precision.MXFP8: 8,
            Precision.NVFP4: 4,
            Precision.MXFP4: 4,
            Precision.MXFP6: 6,
        }
        return mapping[self]
    
    @property
    def requires_blackwell(self) -> bool:
        """Check if this precision requires Blackwell architecture."""
        return self in [Precision.MXFP8, Precision.NVFP4, Precision.MXFP4]


@dataclass
class MicroscalingConfig:
    """Configuration for microscaling (block-wise scaling)."""
    block_size: int = 32  # Number of elements per scaling block
    scale_dtype: str = "e8m0"  # E8M0 format for scale factors (8-bit power-of-two)
    transpose_aware: bool = True  # Handle transpose quantization requirements
    amax_history_len: int = 16  # History length for amax tracking
    
    def validate_for_blackwell(self) -> bool:
        """Validate config for Blackwell architecture."""
        # Blackwell uses 32-element blocks for MXFP8
        if self.block_size not in [16, 32, 64]:
            warnings.warn(f"Block size {self.block_size} may not be optimal for Blackwell")
        return True


@dataclass
class LayerPrecision:
    """Precision configuration for a specific layer."""
    layer_name: str
    compute_dtype: Precision  # Precision for computation
    accumulate_dtype: Precision  # Precision for accumulation
    weight_dtype: Precision  # Precision for weights
    activation_dtype: Precision  # Precision for activations
    gradient_dtype: Optional[Precision] = None  # Precision for gradients
    microscaling: Optional[MicroscalingConfig] = None
    fallback_dtype: Precision = Precision.BF16  # Fallback for stability
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Set gradient dtype if not specified
        if self.gradient_dtype is None:
            self.gradient_dtype = self.compute_dtype
        
        # Add microscaling config for microscaled types
        if any(p.is_microscaled for p in [self.compute_dtype, self.weight_dtype, self.activation_dtype]):
            if self.microscaling is None:
                self.microscaling = MicroscalingConfig()
    
    @property
    def needs_transpose_handling(self) -> bool:
        """Check if layer needs special transpose handling for MXFP8."""
        # MXFP8 requires requantization when transposing
        return (self.compute_dtype == Precision.MXFP8 and 
                self.microscaling and 
                self.microscaling.transpose_aware)


@dataclass
class PrecisionConfig:
    """Global precision configuration."""
    default_compute: Precision = Precision.MXFP8
    default_accumulate: Precision = Precision.FP32
    default_weight: Precision = Precision.MXFP8
    default_activation: Precision = Precision.MXFP8
    default_gradient: Precision = Precision.MXFP8
    
    # Sensitive layers that should use higher precision
    sensitive_layers: List[str] = field(default_factory=lambda: [
        "embedding",
        "final_linear",
        "logits",
        "layer_norm",
        "rmsnorm",
        "first_linear",
    ])
    
    # Fallback configuration
    enable_auto_fallback: bool = True
    fallback_threshold: float = 1e6  # Loss spike threshold for fallback
    max_fallback_attempts: int = 3
    
    # Microscaling defaults
    default_microscaling: MicroscalingConfig = field(default_factory=MicroscalingConfig)
    
    # Blackwell-specific optimizations
    use_cutlass_kernels: bool = True
    use_grouped_gemm: bool = True
    fuse_transpose_quantization: bool = True


class PrecisionPolicy:
    """
    Manages precision assignment and fallback policies for model layers.
    Optimized for Blackwell's MXFP8/FP4 capabilities.
    """
    
    def __init__(self, config: Optional[PrecisionConfig] = None):
        """Initialize precision policy with configuration."""
        self.config = config or PrecisionConfig()
        self.layer_precisions: Dict[str, LayerPrecision] = {}
        self.fallback_history: Dict[str, List[Tuple[int, float]]] = {}
        self.scale_buffers: Dict[str, any] = {}  # Will store actual scale tensors
        
    def assign_layer_precision(self, 
                              layer_name: str,
                              layer_type: str,
                              param_count: Optional[int] = None) -> LayerPrecision:
        """
        Assign precision to a layer based on type and configuration.
        
        Args:
            layer_name: Name/path of the layer
            layer_type: Type of layer (linear, attention, norm, etc.)
            param_count: Number of parameters in the layer
            
        Returns:
            LayerPrecision configuration for the layer
        """
        # Check if layer is in sensitive list
        is_sensitive = any(
            sensitive in layer_name.lower() 
            for sensitive in self.config.sensitive_layers
        )
        
        # Determine precision based on layer type and sensitivity
        if is_sensitive:
            # Use higher precision for sensitive layers
            compute_dtype = Precision.BF16
            weight_dtype = Precision.BF16
            activation_dtype = Precision.BF16
            accumulate_dtype = Precision.FP32
        elif layer_type == "norm":
            # Layer norm should use higher precision
            compute_dtype = Precision.BF16
            weight_dtype = Precision.BF16
            activation_dtype = Precision.BF16
            accumulate_dtype = Precision.FP32
        elif layer_type == "attention":
            # Attention can use MXFP8 but with FP32 accumulation
            compute_dtype = self.config.default_compute
            weight_dtype = self.config.default_weight
            activation_dtype = self.config.default_activation
            accumulate_dtype = Precision.FP32  # Always FP32 for attention
        elif layer_type == "linear" or layer_type == "mlp":
            # Regular linear layers can use full low precision
            compute_dtype = self.config.default_compute
            weight_dtype = self.config.default_weight
            activation_dtype = self.config.default_activation
            accumulate_dtype = self.config.default_accumulate
        else:
            # Default configuration
            compute_dtype = self.config.default_compute
            weight_dtype = self.config.default_weight
            activation_dtype = self.config.default_activation
            accumulate_dtype = self.config.default_accumulate
        
        # Create layer precision config
        layer_precision = LayerPrecision(
            layer_name=layer_name,
            compute_dtype=compute_dtype,
            accumulate_dtype=accumulate_dtype,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            gradient_dtype=self.config.default_gradient,
            microscaling=self.config.default_microscaling if compute_dtype.is_microscaled else None,
            fallback_dtype=Precision.BF16
        )
        
        # Store in registry
        self.layer_precisions[layer_name] = layer_precision
        
        return layer_precision
    
    def should_fallback(self, layer_name: str, loss_value: float, step: int) -> bool:
        """
        Determine if a layer should fallback to higher precision.
        
        Args:
            layer_name: Name of the layer
            loss_value: Current loss value
            step: Current training step
            
        Returns:
            True if layer should fallback to higher precision
        """
        if not self.config.enable_auto_fallback:
            return False
        
        # Track loss history
        if layer_name not in self.fallback_history:
            self.fallback_history[layer_name] = []
        
        history = self.fallback_history[layer_name]
        history.append((step, loss_value))
        
        # Keep only recent history
        history = history[-10:]
        self.fallback_history[layer_name] = history
        
        # Check for loss spike
        if len(history) >= 2:
            prev_loss = history[-2][1]
            if loss_value > prev_loss * self.config.fallback_threshold:
                return True
        
        # Check for NaN/Inf
        import math
        if math.isnan(loss_value) or math.isinf(loss_value):
            return True
        
        return False
    
    def apply_fallback(self, layer_name: str) -> LayerPrecision:
        """
        Apply fallback to higher precision for a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Updated LayerPrecision configuration
        """
        if layer_name not in self.layer_precisions:
            raise ValueError(f"Layer {layer_name} not found in precision policy")
        
        layer_precision = self.layer_precisions[layer_name]
        
        # Upgrade to fallback precision
        layer_precision.compute_dtype = layer_precision.fallback_dtype
        layer_precision.weight_dtype = layer_precision.fallback_dtype
        layer_precision.activation_dtype = layer_precision.fallback_dtype
        layer_precision.microscaling = None
        
        warnings.warn(f"Falling back layer {layer_name} to {layer_precision.fallback_dtype.value}")
        
        return layer_precision
    
    def get_memory_footprint(self, model_params: int) -> Dict[str, float]:
        """
        Estimate memory footprint with current precision policy.
        
        Args:
            model_params: Total number of model parameters
            
        Returns:
            Dictionary with memory estimates in GB
        """
        total_weight_bits = 0
        total_activation_bits = 0
        total_gradient_bits = 0
        
        # Estimate based on layer precisions
        for layer_name, layer_prec in self.layer_precisions.items():
            # Simplified estimation (would need actual param counts per layer)
            layer_weight = model_params / len(self.layer_precisions)
            
            total_weight_bits += layer_weight * layer_prec.weight_dtype.bits
            total_activation_bits += layer_weight * layer_prec.activation_dtype.bits
            total_gradient_bits += layer_weight * layer_prec.gradient_dtype.bits
        
        # Convert to GB
        bytes_per_bit = 1/8
        gb_per_byte = 1/(1024**3)
        
        return {
            'weights_gb': total_weight_bits * bytes_per_bit * gb_per_byte,
            'activations_gb': total_activation_bits * bytes_per_bit * gb_per_byte,
            'gradients_gb': total_gradient_bits * bytes_per_bit * gb_per_byte,
            'total_gb': (total_weight_bits + total_activation_bits + total_gradient_bits) * bytes_per_bit * gb_per_byte
        }
    
    def validate_for_hardware(self, sm_version: int) -> List[str]:
        """
        Validate precision policy for target hardware.
        
        Args:
            sm_version: SM version (e.g., 100 for Blackwell SM100)
            
        Returns:
            List of warnings/errors
        """
        issues = []
        
        for layer_name, layer_prec in self.layer_precisions.items():
            # Check if precision is supported on hardware
            if layer_prec.compute_dtype.requires_blackwell and sm_version < 100:
                issues.append(f"Layer {layer_name} uses {layer_prec.compute_dtype.value} which requires Blackwell (SM100+)")
            
            # Check microscaling configuration
            if layer_prec.microscaling and sm_version < 100:
                issues.append(f"Layer {layer_name} uses microscaling which requires Blackwell")
            
            # Check for FP4 on non-Blackwell
            if layer_prec.compute_dtype in [Precision.NVFP4, Precision.MXFP4] and sm_version < 100:
                issues.append(f"Layer {layer_name} uses FP4 which requires Blackwell")
        
        return issues
    
    def export_config(self) -> Dict:
        """Export precision configuration as dictionary."""
        return {
            'config': {
                'default_compute': self.config.default_compute.value,
                'default_accumulate': self.config.default_accumulate.value,
                'default_weight': self.config.default_weight.value,
                'default_activation': self.config.default_activation.value,
                'enable_auto_fallback': self.config.enable_auto_fallback,
                'use_cutlass_kernels': self.config.use_cutlass_kernels,
                'use_grouped_gemm': self.config.use_grouped_gemm,
            },
            'layer_precisions': {
                name: {
                    'compute': prec.compute_dtype.value,
                    'accumulate': prec.accumulate_dtype.value,
                    'weight': prec.weight_dtype.value,
                    'activation': prec.activation_dtype.value,
                    'has_microscaling': prec.microscaling is not None,
                }
                for name, prec in self.layer_precisions.items()
            }
        }
