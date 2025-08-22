"""Basic API tests for Deepwell."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import deepwell as dw
import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, hidden_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embed = nn.Embedding(50257, hidden_dim)  # GPT-2 vocab size
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 50257)
        
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x):
        # Attention block
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # MLP block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def test_probe():
    """Test hardware probing."""
    print("\n" + "="*60)
    print("Testing Hardware Probe")
    print("="*60)
    
    hw = dw.probe()
    print(f"Total GPUs detected: {hw.total_gpus}")
    print(f"CUDA Version: {hw.cuda_version}")
    print(f"System Memory: {hw.system_memory_gb:.1f} GB")
    
    for gpu in hw.gpus:
        print(f"\nGPU {gpu.device_id}: {gpu.name}")
        print(f"  Compute Capability: {gpu.compute_capability}")
        print(f"  SM Version: {gpu.sm_version}")
        print(f"  Memory: {gpu.memory_gb:.1f} GB")
        print(f"  Is Blackwell: {gpu.is_blackwell}")
        if gpu.is_blackwell:
            print(f"  Blackwell Variant: {gpu.blackwell_variant}")
            print(f"  Supports MXFP8: {gpu.supports_mxfp8}")
            print(f"  Supports FP4: {gpu.supports_fp4}")
    
    return hw


def test_capture():
    """Test model capture."""
    print("\n" + "="*60)
    print("Testing Model Capture")
    print("="*60)
    
    # Create a simple model
    model = SimpleTransformer(hidden_dim=768, num_heads=12, num_layers=2)
    
    # Capture the model
    ir = dw.capture(model)
    
    print(f"Captured IR:")
    print(f"  Parameters: {len(ir.params)}")
    print(f"  Tensors: {len(ir.tensors)}")
    print(f"  Operations: {len(ir.ops)}")
    
    # Show operation types
    op_types = {}
    for op in ir.ops:
        op_types[op.kind] = op_types.get(op.kind, 0) + 1
    
    print(f"\nOperation types:")
    for kind, count in op_types.items():
        print(f"  {kind}: {count}")
    
    return ir


def test_precision_policy():
    """Test precision policy."""
    print("\n" + "="*60)
    print("Testing Precision Policy")
    print("="*60)
    
    # Create precision config for Blackwell
    config = dw.PrecisionConfig(
        default_compute=dw.Precision.MXFP8,
        default_weight=dw.Precision.MXFP8,
        default_activation=dw.Precision.MXFP8,
        enable_auto_fallback=True
    )
    
    policy = dw.PrecisionPolicy(config)
    
    # Assign precision to different layer types
    layers = [
        ("transformer.embed", "embed"),
        ("transformer.layer1.attn", "attention"),
        ("transformer.layer1.mlp", "mlp"),
        ("transformer.ln_f", "norm"),
        ("transformer.lm_head", "linear"),
    ]
    
    for layer_name, layer_type in layers:
        prec = policy.assign_layer_precision(layer_name, layer_type)
        print(f"{layer_name} ({layer_type}):")
        print(f"  Compute: {prec.compute_dtype.value}")
        print(f"  Weight: {prec.weight_dtype.value}")
        print(f"  Has microscaling: {prec.microscaling is not None}")
    
    # Estimate memory
    memory = policy.get_memory_footprint(350_000_000)  # 350M params
    print(f"\nMemory estimates for 350M model:")
    print(f"  Weights: {memory['weights_gb']:.2f} GB")
    print(f"  Activations: {memory['activations_gb']:.2f} GB")
    print(f"  Total: {memory['total_gb']:.2f} GB")
    
    return policy


def test_kernel_registry():
    """Test kernel registry."""
    print("\n" + "="*60)
    print("Testing Kernel Registry")
    print("="*60)
    
    registry = dw.get_registry()
    
    # Find kernels for different scenarios
    scenarios = [
        ("Blackwell MXFP8 GEMM", "gemm", "mxfp8", 100),
        ("Blackwell FP4 GEMM", "gemm", "nvfp4", 100),
        ("Blackwell Grouped GEMM", "grouped_gemm", "mxfp8", 100),
        ("Hopper FP8 GEMM", "gemm", "fp8", 90),
        ("Ampere FP16 GEMM", "gemm", "fp16", 80),
    ]
    
    for desc, op_type, precision, sm in scenarios:
        kernel = registry.find_kernel(op_type, precision, sm)
        if kernel:
            print(f"{desc}:")
            print(f"  Kernel: {kernel.name}")
            print(f"  Backend: {kernel.backend.value}")
            print(f"  Has microscaling: {kernel.has_microscaling}")
        else:
            print(f"{desc}: No kernel found")
    
    # Show available kernels for Blackwell
    print(f"\nKernels available for Blackwell (SM100):")
    available = registry.validate_for_hardware(100)
    for op_type, kernels in available.items():
        if kernels:
            print(f"  {op_type}: {len(kernels)} kernels")
    
    return registry


def test_compilation():
    """Test model compilation."""
    print("\n" + "="*60)
    print("Testing Model Compilation")
    print("="*60)
    
    # Create and capture model
    model = SimpleTransformer(hidden_dim=768, num_heads=12, num_layers=2)
    ir = dw.capture(model)
    
    # Compile for Blackwell with MXFP8
    engine = dw.compile(
        ir,
        precision="mxfp8",
        fallback="safe",
        sm_version=100  # Blackwell
    )
    
    print(f"Compilation successful!")
    print(f"  Compiled ops: {len(engine.compiled_ops)}")
    print(f"  Estimated memory: {engine.estimated_memory_gb:.2f} GB")
    print(f"  Estimated FLOPs: {engine.estimated_flops:.2e}")
    
    # Show kernel usage
    kernel_summary = engine.get_kernel_summary()
    print(f"\nKernel backends used:")
    for backend, count in kernel_summary.items():
        print(f"  {backend}: {count} ops")
    
    # Validate
    issues = engine.validate()
    if issues:
        print(f"\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nValidation: PASSED")
    
    return engine


def test_end_to_end():
    """Test end-to-end optimization."""
    print("\n" + "="*60)
    print("Testing End-to-End Optimization")
    print("="*60)
    
    # Create model
    model = SimpleTransformer(hidden_dim=768, num_heads=12, num_layers=12)
    
    # One-shot optimization
    optimized = dw.optimize_for_blackwell(
        model,
        precision="mxfp8",
        seq_len=2048,
        batch_size=32,
    )

    print("Model optimized for Blackwell!")

    # Ensure the optimized model is trainable by running a single training step
    optimizer = torch.optim.SGD(optimized.parameters(), lr=0.01)
    dummy_input = torch.randint(0, 50257, (2, 16))
    before = optimized.embed.weight.clone()
    optimizer.zero_grad()
    output = optimized(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Verify that parameters were updated
    assert not torch.allclose(before, optimized.embed.weight)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Deepwell API Tests")
    print("="*60)
    
    # Test each component
    hw = test_probe()
    ir = test_capture()
    policy = test_precision_policy()
    registry = test_kernel_registry()
    engine = test_compilation()
    
    # Test end-to-end
    engine = test_end_to_end()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
