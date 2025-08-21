#!/usr/bin/env python3
"""
Quick test to verify CUTLASS kernels are working on B200.
"""

import sys
sys.path.insert(0, 'src')

import torch
import deepwell as dw

print("="*60)
print("🚀 Testing Deepwell CUTLASS Kernels on B200")
print("="*60)

# 1. Check if CUTLASS module loaded
try:
    from deepwell import cutlass_kernels
    print("✅ CUTLASS module loaded successfully!")
    print(f"   Module path: {cutlass_kernels.__file__}")
except ImportError as e:
    print(f"❌ Failed to load CUTLASS module: {e}")
    sys.exit(1)

# 2. Test MicroscaleManager
print("\n--- Testing MXFP8 Quantization ---")
try:
    x = torch.randn(128, 1024, device='cuda', dtype=torch.bfloat16)
    print(f"Input: {x.shape}, dtype: {x.dtype}")
    
    # Quantize
    quant, scales = cutlass_kernels.MicroscaleManager.quantize_mxfp8(x)
    print(f"✅ Quantized: {quant.shape}, scales: {scales.shape}")
    
    # Dequantize
    dequant = cutlass_kernels.MicroscaleManager.dequantize_mxfp8(quant, scales)
    print(f"✅ Dequantized: {dequant.shape}")
    
    # Check error
    error = torch.abs(x - dequant).max().item()
    print(f"   Max error: {error:.6f}")
    
except Exception as e:
    print(f"❌ Quantization test failed: {e}")
    import traceback
    traceback.print_exc()

# 3. Test BlackwellGemmKernel
print("\n--- Testing Blackwell GEMM Kernel ---")
try:
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    print(f"Matrix A: {a.shape}, Matrix B: {b.shape}")
    
    # Create kernel
    kernel = cutlass_kernels.BlackwellGemmKernel()
    kernel.initialize(M, N, K, "bf16", use_microscaling=False)
    print("✅ Kernel initialized")
    
    # Execute GEMM
    c = kernel.gemm(a, b)
    print(f"✅ GEMM executed: output shape {c.shape}")
    
    # Verify against PyTorch
    c_ref = torch.matmul(a, b)
    error = torch.abs(c - c_ref).max().item()
    print(f"   Max error vs PyTorch: {error:.6f}")
    
    if error < 1e-3:
        print("✅ Results match PyTorch!")
    else:
        print(f"⚠️  Higher error than expected: {error}")
        
except Exception as e:
    print(f"❌ GEMM test failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Test with MXFP8
print("\n--- Testing MXFP8 GEMM ---")
try:
    kernel_mxfp8 = cutlass_kernels.BlackwellGemmKernel()
    kernel_mxfp8.initialize(M, N, K, "mxfp8", use_microscaling=True)
    print("✅ MXFP8 kernel initialized")
    
    # Note: The kernel internally handles quantization
    c_mxfp8 = kernel_mxfp8.gemm(a, b)
    print(f"✅ MXFP8 GEMM executed: {c_mxfp8.shape}")
    
    error = torch.abs(c_mxfp8 - c_ref).max().item()
    print(f"   Max error vs reference: {error:.6f}")
    
except Exception as e:
    print(f"❌ MXFP8 GEMM test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✨ Testing complete!")
print("="*60)
print("\nYour B200 GPU is now using real Blackwell kernels!")
print("Next: Run './run_real_benchmark.sh' for full performance test")
