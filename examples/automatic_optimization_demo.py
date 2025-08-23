"""
Automatic optimization example - optimize any PyTorch model with one line!
Similar to Cursor's MXFP8 optimization approach.
"""

import torch
import torch.nn as nn
import deepwell
import time


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstration"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1000)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


def benchmark_model(model, input_tensor, name="Model", num_iters=50):
    """Benchmark a model's forward pass"""
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed:.3f}s for {num_iters} iterations ({elapsed/num_iters*1000:.2f}ms per iter)")
    return elapsed


def main():
    print("="*60)
    print("Deepwell Automatic Optimization Demo")
    print("="*60)
    
    # Create a model
    print("\n1. Creating a standard PyTorch transformer model...")
    model = SimpleTransformer(d_model=512, nhead=8, num_layers=6)
    model = model.cuda().to(torch.bfloat16)
    
    # Create test input - use larger sequence for better CUTLASS performance
    batch_size = 8
    seq_len = 2048  # Must be multiple of 64 for CUTLASS FMHA
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)
    
    # Benchmark original model
    print("\n2. Benchmarking original PyTorch model...")
    orig_time = benchmark_model(model, x, "Original PyTorch")
    
    # AUTOMATIC OPTIMIZATION - ONE LINE!
    print("\n3. Optimizing with Deepwell (one line!)...")
    print("   >>> model = deepwell.optimize(model)")
    optimized_model = deepwell.optimize(model, verbose=True)
    
    # Benchmark optimized model
    print("\n4. Benchmarking optimized model...")
    opt_time = benchmark_model(optimized_model, x, "Deepwell Optimized")
    
    # Show speedup
    speedup = orig_time / opt_time
    print(f"\n{'='*60}")
    print(f"RESULTS: {speedup:.2f}x speedup with Deepwell!")
    print(f"{'='*60}")
    
    # Demonstrate that the model works exactly the same
    print("\n5. Verifying correctness...")
    model.eval()
    optimized_model.eval()
    
    with torch.no_grad():
        orig_out = model(x)
        opt_out = optimized_model(x)
        
        # Check outputs are close (some numerical differences expected)
        max_diff = (orig_out - opt_out).abs().max().item()
        print(f"   Max difference in outputs: {max_diff:.6f}")
        
        if max_diff < 0.01:
            print("   ✓ Outputs match! Optimization is correct.")
        else:
            print("   ⚠ Warning: Outputs differ. This might be due to numerical precision.")
    
    print("\n" + "="*60)
    print("Demo complete! Deepwell automatically optimized your model.")
    print("No code changes needed - just one line: deepwell.optimize(model)")
    print("="*60)


if __name__ == "__main__":
    main()
