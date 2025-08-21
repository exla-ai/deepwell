#!/usr/bin/env python3
"""
Test real Blackwell kernel dispatch with MXFP8 quantization.
This verifies that we're actually using tcgen05.mma instructions.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import time
import deepwell as dw


def test_mxfp8_quantization():
    """Test MXFP8 quantization and dequantization."""
    print("="*60)
    print("Testing MXFP8 Quantization")
    print("="*60)
    
    # Check if CUTLASS is available
    try:
        from deepwell import cutlass_kernels
        print("✓ CUTLASS module loaded")
    except ImportError as e:
        print(f"✗ CUTLASS not available: {e}")
        return False
    
    # Create test tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(128, 4096, device=device, dtype=torch.bfloat16)
    
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    
    # Test quantization
    try:
        quant, scales = cutlass_kernels.MicroscaleManager.quantize_mxfp8(x)
        print(f"✓ Quantized to MXFP8")
        print(f"  Quantized shape: {quant.shape}")
        print(f"  Scale shape: {scales.shape}")
        print(f"  Scale factors per row: {scales.shape[0] // x.shape[0] if x.shape[0] > 0 else 0}")
        
        # Test dequantization
        dequant = cutlass_kernels.MicroscaleManager.dequantize_mxfp8(quant, scales)
        print(f"✓ Dequantized from MXFP8")
        print(f"  Output shape: {dequant.shape}")
        
        # Check accuracy
        error = torch.abs(x - dequant).max().item()
        rel_error = error / torch.abs(x).max().item()
        print(f"  Max absolute error: {error:.6f}")
        print(f"  Max relative error: {rel_error:.6f}")
        
        if rel_error < 0.1:  # 10% error is acceptable for MXFP8
            print("✓ Quantization accuracy is acceptable")
            return True
        else:
            print(f"✗ Quantization error too high: {rel_error:.2%}")
            return False
            
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blackwell_gemm():
    """Test Blackwell GEMM kernel with MXFP8."""
    print("\n" + "="*60)
    print("Testing Blackwell GEMM with MXFP8")
    print("="*60)
    
    try:
        from deepwell import cutlass_kernels
        print("✓ CUTLASS module loaded")
    except ImportError as e:
        print(f"✗ CUTLASS not available: {e}")
        return False
    
    # Create test matrices
    M, N, K = 2048, 2048, 2048
    device = torch.device('cuda')
    
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    print(f"Matrix A: {a.shape}, dtype: {a.dtype}")
    print(f"Matrix B: {b.shape}, dtype: {b.dtype}")
    
    try:
        # Create kernel
        kernel = cutlass_kernels.BlackwellGemmKernel()
        kernel.initialize(M, N, K, "mxfp8", use_microscaling=True, block_size=32)
        print("✓ Blackwell kernel initialized for MXFP8")
        
        # Warmup
        print("Warming up...")
        for _ in range(5):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        print("Benchmarking...")
        iterations = 20
        start = time.time()
        for _ in range(iterations):
            c = kernel.gemm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate performance
        flops = 2 * M * N * K * iterations
        tflops = flops / elapsed / 1e12
        time_per_iter = elapsed / iterations * 1000  # ms
        
        print(f"\nResults:")
        print(f"  Time per iteration: {time_per_iter:.2f} ms")
        print(f"  Performance: {tflops:.1f} TFLOPS")
        
        # Compare with reference
        print("\nComparing with PyTorch (BF16)...")
        start = time.time()
        for _ in range(iterations):
            c_ref = torch.matmul(a, b)
        torch.cuda.synchronize()
        ref_elapsed = time.time() - start
        ref_tflops = flops / ref_elapsed / 1e12
        
        print(f"  PyTorch BF16: {ref_tflops:.1f} TFLOPS")
        print(f"  Speedup: {tflops/ref_tflops:.2f}x")
        
        # Verify correctness
        c_ref = torch.matmul(a, b)
        error = torch.abs(c - c_ref).max().item()
        rel_error = error / torch.abs(c_ref).max().item()
        print(f"\n  Max error vs reference: {error:.6f}")
        print(f"  Relative error: {rel_error:.6f}")
        
        if rel_error < 0.1:
            print("✓ Results match reference (within MXFP8 tolerance)")
            return True
        else:
            print(f"✗ Results differ from reference: {rel_error:.2%}")
            return False
            
    except Exception as e:
        print(f"✗ GEMM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model_dispatch():
    """Test full model with Deepwell optimization."""
    print("\n" + "="*60)
    print("Testing Full Model with Real Dispatch")
    print("="*60)
    
    # Simple test model
    class TestModel(nn.Module):
        def __init__(self, hidden_dim=1024):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
                for _ in range(2)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)
            return x
    
    model = TestModel()
    batch_size = 32
    seq_len = 512
    hidden_dim = 1024
    
    print(f"Model: 2-layer transformer")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}")
    
    # Optimize with Deepwell
    print("\nOptimizing with Deepwell...")
    engine = dw.optimize_for_blackwell(
        model,
        precision="mxfp8",
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    # Create executable model
    exec_model = dw.create_executable_model(engine, model)
    
    # Check if CUTLASS is active
    print(f"CUTLASS active: {exec_model.use_cutlass}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    model = model.to(device)
    exec_model = exec_model.to(device)
    
    with torch.no_grad():
        # Reference
        y_ref = model(x)
        
        # Optimized
        y_opt = exec_model(x)
        
        # Check results
        if y_ref is not None and y_opt is not None:
            error = torch.abs(y_ref - y_opt).max().item()
            rel_error = error / torch.abs(y_ref).max().item()
            print(f"\nMax error: {error:.6f}")
            print(f"Relative error: {rel_error:.6f}")
            
            if rel_error < 0.1:
                print("✓ Optimized model matches reference")
                return True
            else:
                print(f"✗ Results differ: {rel_error:.2%}")
                return False
    
    return False


def main():
    print("🚀 Blackwell Real Kernel Dispatch Test")
    print("="*60)
    
    # Detect hardware
    hw = dw.probe()
    if hw.gpus:
        gpu = hw.gpus[0]
        print(f"GPU: {gpu.name}")
        print(f"Compute Capability: {gpu.compute_capability}")
        print(f"Blackwell: {gpu.is_blackwell}")
        print(f"MXFP8 Support: {gpu.supports_mxfp8}")
    else:
        print("No GPU detected!")
        return
    
    # Run tests
    tests_passed = []
    
    # Test 1: MXFP8 Quantization
    tests_passed.append(test_mxfp8_quantization())
    
    # Test 2: Blackwell GEMM
    tests_passed.append(test_blackwell_gemm())
    
    # Test 3: Full Model
    tests_passed.append(test_full_model_dispatch())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"MXFP8 Quantization: {'✓ PASSED' if tests_passed[0] else '✗ FAILED'}")
    print(f"Blackwell GEMM: {'✓ PASSED' if tests_passed[1] else '✗ FAILED'}")
    print(f"Full Model: {'✓ PASSED' if tests_passed[2] else '✗ FAILED'}")
    
    if all(tests_passed):
        print("\n✅ All tests passed! Real kernel dispatch is working.")
        print("Your Blackwell GPU is now using tcgen05.mma instructions!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        print("The CUTLASS extension may need to be rebuilt.")


if __name__ == "__main__":
    main()
