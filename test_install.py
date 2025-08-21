#!/usr/bin/env python3
"""
Quick test to verify Deepwell installation.
Run after pip/uv install to confirm everything works.
"""

import sys

def test_import():
    """Test basic import."""
    try:
        import deepwell as dw
        print("✓ Deepwell imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import deepwell: {e}")
        return False

def test_hardware():
    """Test hardware detection."""
    try:
        import deepwell as dw
        hw = dw.probe()
        
        if hw.gpus:
            gpu = hw.gpus[0]
            print(f"✓ Detected GPU: {gpu.name}")
            if gpu.is_blackwell:
                print(f"  ✓ Blackwell {gpu.blackwell_variant} confirmed!")
                print(f"  - MXFP8 support: {gpu.supports_mxfp8}")
                print(f"  - FP4 support: {gpu.supports_fp4}")
        else:
            print("⚠ No GPU detected (CPU-only mode)")
        return True
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
        return False

def test_kernels():
    """Test CUDA kernel availability."""
    try:
        from deepwell import cutlass_kernels
        print("✓ CUDA kernels compiled and available")
        return True
    except ImportError:
        print("⚠ CUDA kernels not available (will use PyTorch fallback)")
        return True  # Not a failure, just a warning

def test_optimization():
    """Test basic optimization."""
    try:
        import torch
        import torch.nn as nn
        import deepwell as dw
        
        # Simple test model
        model = nn.Linear(10, 10)
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Try to optimize
        optimized = dw.optimize_for_blackwell(model)
        print("✓ Model optimization working")
        return True
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DEEPWELL INSTALLATION TEST".center(60))
    print("=" * 60)
    print()
    
    tests = [
        ("Import", test_import),
        ("Hardware Detection", test_hardware),
        ("CUDA Kernels", test_kernels),
        ("Optimization", test_optimization),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        success = test_func()
        results.append((name, success))
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY".center(60))
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Deepwell is ready to use.")
        print("\nNext steps:")
        print("  - Run full test suite: python test.py")
        print("  - Run benchmarks: python benchmarks/benchmark.py")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
