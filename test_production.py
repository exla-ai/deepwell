#!/usr/bin/env python3
"""
Test script to validate Deepwell production kernels on B200.
Run this FIRST before benchmarking to ensure correctness.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import pytest

def test_gemm_correctness():
    """Test GEMM kernel correctness."""
    print("=" * 70)
    print("TESTING GEMM CORRECTNESS")
    print("=" * 70)
    
    # Test sizes
    test_cases = [
        (512, 512, 512),
        (1024, 768, 768),
        (4096, 4096, 1024),
    ]
    
    for m, n, k in test_cases:
        print(f"\nTest {m}x{n}x{k}:")
        
        # Create test matrices
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
        
        # PyTorch reference
        ref_output = torch.matmul(a, b)
        
        # Deepwell kernel
        try:
            from deepwell import cutlass_kernels
            kernel = cutlass_kernels.BlackwellGemmKernel()
            kernel.initialize(m, n, k, "bf16", False, 32)
            
            deepwell_output = kernel.gemm(a, b)
            
            # Check correctness
            diff = torch.abs(ref_output - deepwell_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_error = (diff / (torch.abs(ref_output) + 1e-6)).max().item()
            
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Relative error: {rel_error:.6f}")
            
            # Pass/Fail
            if max_diff < 0.01 and rel_error < 0.01:
                print(f"  ✅ PASSED")
            else:
                print(f"  ❌ FAILED - difference too large!")
                
                # Debug info
                print(f"\n  Debug info:")
                print(f"  PyTorch output range: [{ref_output.min():.3f}, {ref_output.max():.3f}]")
                print(f"  Deepwell output range: [{deepwell_output.min():.3f}, {deepwell_output.max():.3f}]")
                print(f"  PyTorch output mean: {ref_output.mean():.3f}")
                print(f"  Deepwell output mean: {deepwell_output.mean():.3f}")
                
                # Show a few values
                print(f"\n  First 5 values:")
                print(f"  PyTorch:  {ref_output.flatten()[:5].tolist()}")
                print(f"  Deepwell: {deepwell_output.flatten()[:5].tolist()}")
                
                return False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            return False
    
    return True


def test_fused_operations():
    """Test fused kernel correctness."""
    print("\n" + "=" * 70)
    print("TESTING FUSED OPERATIONS")
    print("=" * 70)
    # Skip if CUTLASS Python API is not available
    try:
        import deepwell.kernels.blackwell_production as bp
        if not getattr(bp, "CUTLASS_AVAILABLE", False):
            pytest.skip("CUTLASS Python API not available")
    except Exception:
        pytest.skip("CUTLASS Python API not available")
    
    batch_size = 32
    seq_len = 512
    hidden_dim = 768
    
    # Test Linear + GELU fusion
    print("\n1. Testing Linear + GELU fusion:")
    
    x = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.bfloat16, device='cuda')
    linear = nn.Linear(hidden_dim, hidden_dim * 4, dtype=torch.bfloat16).cuda()
    
    # Reference
    ref_output = torch.nn.functional.gelu(linear(x))
    
    from deepwell.kernels.blackwell_production import FusedLinearGELU, BlackwellConfig
    
    config = BlackwellConfig()
    fused = FusedLinearGELU(hidden_dim, hidden_dim * 4, config).cuda()
    
    # Copy weights
    fused.weight.data = linear.weight.data
    fused.bias.data = linear.bias.data
    
    # Execute
    fused_output = fused(x)
    
    # Check
    diff = torch.abs(ref_output - fused_output).max().item()
    print(f"  Max difference: {diff:.6f}")
    
    if diff < 0.01:
        print(f"  ✅ PASSED")
    else:
        print(f"  ❌ FAILED")
        assert False


def test_flash_attention():
    """Test Flash Attention correctness."""
    print("\n" + "=" * 70)
    print("TESTING FLASH ATTENTION")
    print("=" * 70)
    
    batch_size = 4
    num_heads = 12
    seq_len = 512
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda')
    
    # Reference
    ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    try:
        from deepwell.kernels.blackwell_production import BlackwellFlashAttention, BlackwellConfig
        
        config = BlackwellConfig()
        flash_attn = BlackwellFlashAttention(config)
        
        # Execute
        flash_output = flash_attn.forward(q, k, v, causal=True)
        
        # Check
        diff = torch.abs(ref_output - flash_output).max().item()
        print(f"  Max difference: {diff:.6f}")
        
        if diff < 0.01:
            print(f"  ✅ PASSED")
        else:
            print(f"  ❌ FAILED")
            
    except ImportError:
        print("  ⚠ Flash Attention not available yet")


def test_grouped_gemm():
    """Test Grouped GEMM for MoE."""
    print("\n" + "=" * 70)
    print("TESTING GROUPED GEMM (MoE)")
    print("=" * 70)
    # Skip if CUTLASS Python API is not available
    try:
        import deepwell.kernels.blackwell_production as bp
        if not getattr(bp, "CUTLASS_AVAILABLE", False):
            pytest.skip("CUTLASS Python API not available")
    except Exception:
        pytest.skip("CUTLASS Python API not available")
    
    num_experts = 8
    batch_size = 256
    hidden_dim = 768
    expert_dim = 3072
    
    print(f"\nTesting {num_experts} experts:")
    
    # Create expert weights and inputs
    inputs = []
    weights = []
    
    for i in range(num_experts):
        # Each expert gets different number of tokens
        num_tokens = batch_size // num_experts + (i * 10)
        inputs.append(torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device='cuda'))
        weights.append(torch.randn(expert_dim, hidden_dim, dtype=torch.bfloat16, device='cuda'))
    
    # Reference: Sequential GEMM
    ref_outputs = []
    for inp, weight in zip(inputs, weights):
        ref_outputs.append(torch.matmul(inp, weight.t()))
    
    from deepwell.kernels.blackwell_production import BlackwellGroupedGEMM, BlackwellConfig
    
    config = BlackwellConfig()
    grouped_gemm = BlackwellGroupedGEMM(config)
    
    # Execute grouped
    grouped_outputs = grouped_gemm.grouped_gemm(inputs, weights)
    
    # Check each expert
    all_pass = True
    for i, (ref, out) in enumerate(zip(ref_outputs, grouped_outputs)):
        diff = torch.abs(ref - out).max().item()
        print(f"  Expert {i}: max diff = {diff:.6f}")
        if diff > 0.01:
            all_pass = False
    
    if all_pass:
        print(f"  ✅ ALL PASSED")
    else:
        print(f"  ❌ SOME FAILED")
        assert False


def main():
    """Run all tests."""
    print("Deepwell Production Kernel Tests")
    print("Running on:", torch.cuda.get_device_name())
    print()
    
    # Check if we're on Blackwell
    device_name = torch.cuda.get_device_name()
    if "B200" not in device_name and "B100" not in device_name:
        print("⚠️  WARNING: Not running on Blackwell GPU!")
        print("   These kernels are optimized for B200/B100")
    
    # Run tests
    passed = True
    
    # 1. GEMM correctness (CRITICAL)
    if not test_gemm_correctness():
        passed = False
        print("\n❌ GEMM CORRECTNESS FAILED - FIX THIS FIRST!")
    
    # 2. Fused operations
    test_fused_operations()
    
    # 3. Flash Attention
    test_flash_attention()
    
    # 4. Grouped GEMM
    test_grouped_gemm()
    
    # Summary
    print("\n" + "=" * 70)
    if passed:
        print("✅ CORE TESTS PASSED - Ready for benchmarking")
    else:
        print("❌ TESTS FAILED - Fix correctness before benchmarking!")
    print("=" * 70)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
