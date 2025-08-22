#!/bin/bash

echo "=========================================="
echo "   Blackwell GPU Benchmark Suite"
echo "=========================================="
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="10.0"  # Blackwell architecture
export CUTLASS_ENABLE_TENSOR_CORE=1
export CUTLASS_ENABLE_SM100=1

# Check Python environment
echo "🐍 Python Environment:"
python3 --version
echo ""

# Build the extension if needed
if [ ! -d "build" ]; then
    echo "📦 Building CUTLASS extension..."
    python3 setup.py build_ext --inplace
    if [ $? -ne 0 ]; then
        echo "⚠️  Build failed, continuing with PyTorch fallback..."
    fi
    echo ""
fi

# Run the main benchmark
echo "🚀 Running Blackwell CUTLASS Benchmark..."
echo "=========================================="
python3 benchmarks/blackwell_cutlass_benchmark.py

# Check if benchmark was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    
    # Display results if available
    if [ -f "blackwell_benchmark_results.json" ]; then
        echo ""
        echo "📊 Results saved to: blackwell_benchmark_results.json"
        echo ""
        echo "Quick Summary:"
        python3 -c "
import json
with open('blackwell_benchmark_results.json', 'r') as f:
    results = json.load(f)
    
    # Extract and display key metrics
    torch_times = []
    cutlass_times = []
    
    for config, runs in results.items():
        for run in runs:
            if run['name'] == 'torch.compile':
                torch_times.append(run['time_ms'])
            elif 'CUTLASS' in run['name']:
                cutlass_times.append(run['time_ms'])
    
    if torch_times and cutlass_times:
        avg_torch = sum(torch_times) / len(torch_times)
        avg_cutlass = sum(cutlass_times) / len(cutlass_times)
        speedup = avg_torch / avg_cutlass
        
        print(f'  Average torch.compile time: {avg_torch:.3f} ms')
        print(f'  Average CUTLASS time: {avg_cutlass:.3f} ms')
        print(f'  Average speedup: {speedup:.2f}x')
        
        if speedup > 1.0:
            print(f'  ✅ CUTLASS is {speedup:.2f}x faster on average!')
        else:
            print(f'  ⚠️  torch.compile is {1/speedup:.2f}x faster on average')
"
    fi
else
    echo ""
    echo "❌ Benchmark failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "=========================================="
echo "   Benchmark Complete"
echo "=========================================="