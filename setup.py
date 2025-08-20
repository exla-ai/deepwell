"""
Setup script for building CUTLASS C++ extensions.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import torch

# Get paths
ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"
CUTLASS_DIR = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# Check for CUDA
if not torch.cuda.is_available():
    print("Warning: CUDA not available. Building CPU-only version.")
    USE_CUDA = False
else:
    USE_CUDA = True
    
# Get CUDA architecture flags
if USE_CUDA:
    # Detect GPU architecture
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().replace(".", "")
            # Convert to CUDA arch flag
            cuda_arch = f"-gencode=arch=compute_{compute_cap},code=sm_{compute_cap}"
            
            # Add PTX for forward compatibility
            if int(compute_cap) >= 100:  # Blackwell
                cuda_arch += f" -gencode=arch=compute_100a,code=sm_100a"
                print(f"Detected Blackwell GPU (SM{compute_cap})")
            elif int(compute_cap) >= 90:  # Hopper
                cuda_arch += f" -gencode=arch=compute_90a,code=sm_90a"
                print(f"Detected Hopper GPU (SM{compute_cap})")
        else:
            # Default to common architectures
            cuda_arch = "-gencode=arch=compute_80,code=sm_80"
            print("Using default CUDA architecture (SM80)")
    except:
        cuda_arch = "-gencode=arch=compute_80,code=sm_80"
else:
    cuda_arch = ""

# Extension sources
sources = [
    str(CSRC_DIR / "cutlass_kernels.cpp"),
    str(CSRC_DIR / "python_bindings.cpp"),
]

# Include directories
include_dirs = [
    str(CSRC_DIR),
    str(CUTLASS_DIR / "include"),
    str(CUTLASS_DIR / "tools/util/include"),
]

# Add PyTorch include directories
include_dirs.extend(cpp_extension.include_paths())

# Libraries to link
libraries = []
library_dirs = []

if USE_CUDA:
    libraries.extend(["cudart", "cublas", "cublasLt"])
    library_dirs.append(cpp_extension.library_paths()[0])

# Compiler flags
extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-DUSE_CUTLASS",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        cuda_arch,
        "-DUSE_CUTLASS",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    ],
}

# Define extension
if USE_CUDA:
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="deepwell.cutlass_kernels",
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=["-Wl,-rpath,$ORIGIN"],
        )
    ]
else:
    ext_modules = [
        cpp_extension.CppExtension(
            name="deepwell.cutlass_kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args["cxx"],
        )
    ]

# Setup configuration
setup(
    name="deepwell-cutlass",
    version="0.0.1",
    author="Deepwell Team",
    description="CUTLASS kernels for Blackwell optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
        ],
    },
    zip_safe=False,
)

# Print build info
print("\n" + "="*60)
print("Building Deepwell CUTLASS Extensions")
print("="*60)
print(f"CUDA Available: {USE_CUDA}")
if USE_CUDA:
    print(f"CUDA Architecture: {cuda_arch}")
print(f"CUTLASS Path: {CUTLASS_DIR}")
print(f"Sources: {sources}")
print("="*60 + "\n")
