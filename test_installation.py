#!/usr/bin/env python3
"""Test Deepwell installation."""

import sys

def test_import():
    """Test basic import."""
    try:
        import deepwell
        print(f"✅ Deepwell {deepwell.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import deepwell: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            major, minor = torch.cuda.get_device_capability()
            print(f"   Compute Capability: {major}.{minor}")
            if major >= 10:
                print(f"   ✅ Blackwell GPU detected (SM{major}{minor})")
            return True
        else:
            print("⚠️  CUDA not available")
            return False
    except Exception as e:
        print(f"⚠️  Error checking CUDA: {e}")
        return False

def test_optimize():
    """Test model optimization."""
    try:
        import deepwell
        import torch
        import torch.nn as nn
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Optimize the model
        optimized = deepwell.optimize(model, verbose=False)
        print("✅ Model optimization works")
        
        # Test forward pass
        if torch.cuda.is_available():
            x = torch.randn(4, 128).cuda()
        else:
            x = torch.randn(4, 128)
            
        output = optimized(x)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernels():
    """Test kernel availability."""
    try:
        import deepwell
        from deepwell.kernels.cutlass_bindings import CUTLASS_AVAILABLE
        
        if CUTLASS_AVAILABLE:
            print("✅ CUTLASS kernels available")
        else:
            print("⚠️  CUTLASS kernels not available (Python-only mode)")
            
        # Check for FMHA bridge
        try:
            from deepwell.kernels.blackwell_flash_attention import BlackwellFlashAttention
            print("✅ BlackwellFlashAttention available")
        except ImportError:
            print("⚠️  BlackwellFlashAttention not available")
            
        return True
    except Exception as e:
        print(f"⚠️  Error checking kernels: {e}")
        return False

if __name__ == "__main__":
    print("Testing Deepwell Installation")
    print("="*40)
    
    results = [
        test_import(),
        test_cuda(),
        test_optimize(),
        test_kernels(),
    ]
    
    print("\n" + "="*40)
    if all(results):
        print("✅ All tests passed!")
        sys.exit(0)
    elif any(results):
        print("⚠️  Some tests failed (library partially functional)")
        sys.exit(0)
    else:
        print("❌ All tests failed")
        sys.exit(1)
