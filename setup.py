"""
Deepwell: Automatic PyTorch optimization for NVIDIA Blackwell GPUs
Builds all necessary components during installation
"""

import os
import sys
import subprocess
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop

# Check if we should skip building C++ extensions
SKIP_BUILD_EXTENSIONS = os.environ.get('DEEPWELL_NO_BUILD_EXTENSIONS', '0') == '1'

# Conditional torch import
try:
    if not SKIP_BUILD_EXTENSIONS:
        import torch
        from torch.utils import cpp_extension
        TORCH_AVAILABLE = True
    else:
        print("Skipping C++ extension build (DEEPWELL_NO_BUILD_EXTENSIONS=1)")
        torch = None
        cpp_extension = None
        TORCH_AVAILABLE = False
except ImportError:
    print("Warning: PyTorch not found during build, C++ extensions will be skipped")
    print("The library will still be installed but with limited functionality.")
    torch = None
    cpp_extension = None
    TORCH_AVAILABLE = False


def download_cutlass():
    """Download CUTLASS if not present"""
    root_dir = Path(__file__).parent
    cutlass_dir = root_dir / "third_party" / "cutlass"
    
    if cutlass_dir.exists():
        print(f"CUTLASS already exists at {cutlass_dir}")
        return cutlass_dir
    
    print("Downloading CUTLASS from GitHub...")
    cutlass_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Download CUTLASS v3.5.1 (latest stable with Blackwell support)
    cutlass_url = "https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.5.1.zip"
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "cutlass.zip"
            print(f"Downloading {cutlass_url}...")
            urllib.request.urlretrieve(cutlass_url, zip_path)
            
            print("Extracting CUTLASS...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Move to final location
            extracted_dir = Path(tmpdir) / "cutlass-3.5.1"
            shutil.move(str(extracted_dir), str(cutlass_dir))
            
        print(f"CUTLASS downloaded to {cutlass_dir}")
        return cutlass_dir
        
    except Exception as e:
        print(f"Warning: Failed to download CUTLASS: {e}")
        print("CUTLASS kernels will not be available.")
        print("You can manually install CUTLASS or use: pip install nvidia-cutlass")
        return None


class BuildFMHABridge:
    """Build the CUTLASS FMHA bridge library"""
    
    @staticmethod
    def build():
        root_dir = Path(__file__).parent
        bridge_dir = root_dir / "csrc" / "fmha_bridge_min"
        build_dir = bridge_dir / "build"
        
        print("\n" + "="*60)
        print("Building CUTLASS FMHA Bridge for Blackwell")
        print("="*60)
        
        # Create build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Run CMake
        print("Configuring with CMake...")
        cmake_cmd = [
            "cmake",
            "-S", str(bridge_dir),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=100a"
        ]
        
        result = subprocess.run(cmake_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"CMake configuration failed: {result.stderr}")
            print("Warning: FMHA bridge build failed, continuing without it")
            return None
            
        # Build
        print("Building FMHA bridge...")
        build_cmd = ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count())]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            print("Warning: FMHA bridge build failed, continuing without it")
            return None
            
        # Find the built library
        bridge_lib = build_dir / "libdw_fmha_bridge_min.so"
        if not bridge_lib.exists():
            print("Warning: FMHA bridge library not found after build")
            return None
            
        print(f"Successfully built FMHA bridge: {bridge_lib}")
        
        # Copy to package directory for distribution
        target_dir = root_dir / "src" / "deepwell" / "lib"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_lib = target_dir / "libdw_fmha_bridge_min.so"
        shutil.copy2(bridge_lib, target_lib)
        print(f"Installed bridge to: {target_lib}")
        
        return str(target_lib)


if TORCH_AVAILABLE and cpp_extension:
    class CustomBuildExt(cpp_extension.BuildExtension):
        """Custom build extension to also build FMHA bridge"""
        
        def run(self):
            # Download CUTLASS if needed
            cutlass_dir = download_cutlass()
            
            # Build FMHA bridge first
            bridge_path = BuildFMHABridge.build()
            
            # Store bridge path for runtime
            if bridge_path:
                config_file = Path(self.build_lib) / "deepwell" / "_bridge_config.py"
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text(f'BRIDGE_PATH = "{bridge_path}"\n')
            
            # Continue with normal extension build
            super().run()
else:
    from setuptools.command.build_ext import build_ext
    class CustomBuildExt(build_ext):
        """Dummy build extension when torch not available"""
        def run(self):
            print("Skipping C++ extension build (PyTorch not available)")
            super().run()


class CustomInstall(install):
    """Custom install to ensure everything is built"""
    
    def run(self):
        self.run_command('build_ext')
        super().run()


class CustomDevelop(develop):
    """Custom develop to ensure everything is built"""
    
    def run(self):
        self.run_command('build_ext')
        super().run()


# Detect Blackwell GPU
def detect_gpu():
    """Detect GPU architecture"""
    # If building without extensions, return default
    if not TORCH_AVAILABLE:
        return 80
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().replace(".", "")
            return int(compute_cap)
    except:
        pass
    return 80  # Default to Ampere


# Get paths
ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"

# Ensure CUTLASS is available (download if needed for source installs)
LOCAL_CUTLASS = ROOT_DIR / "third_party" / "cutlass"
if not LOCAL_CUTLASS.exists():
    # Try to download CUTLASS for source installations
    # This won't run for pip install git+https:// but that's OK
    # because nvidia-cutlass will be installed as a dependency
    LOCAL_CUTLASS = download_cutlass() or LOCAL_CUTLASS

# Check for CUDA and detect GPU architecture
if TORCH_AVAILABLE and not SKIP_BUILD_EXTENSIONS:
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. C++ extensions will not be built.")
        print("For full functionality, please install CUDA 12.8+")
        cuda_arch_flags = []
    else:
        # Detect GPU and set architecture
        compute_cap = detect_gpu()
        if compute_cap >= 100:
            # Blackwell - use sm_100a for tcgen05 support
            cuda_arch_flags = ["-arch=sm_100a"]
            print(f"Detected Blackwell GPU (SM{compute_cap})")
        else:
            print(f"Warning: Non-Blackwell GPU detected (SM{compute_cap})")
            print("Deepwell is optimized for Blackwell GPUs (RTX 50 series, H200, GB200)")
            cuda_arch_flags = [f"-gencode=arch=compute_{compute_cap},code=sm_{compute_cap}"]
else:
    cuda_arch_flags = []

# Build extensions only if torch is available and not skipping
if TORCH_AVAILABLE and not SKIP_BUILD_EXTENSIONS:
    # Extension sources - use relative paths
    cpp_sources = [
        "csrc/cutlass_kernels.cpp",
        "csrc/python_bindings.cpp",
    ]

    cuda_sources = []
    cuda_files = ["mxfp8_quantization.cu"]
    for cuda_file in cuda_files:
        cuda_path = CSRC_DIR / cuda_file
        if cuda_path.exists():
            cuda_sources.append(f"csrc/{cuda_file}")

    # Combine sources
    sources = cpp_sources + cuda_sources

    # Include directories
    include_dirs = [
        str(CSRC_DIR),
    ]

    # Add CUTLASS include paths if available
    if LOCAL_CUTLASS.exists():
        include_dirs.extend([
            str(LOCAL_CUTLASS / "include"),
            str(LOCAL_CUTLASS / "examples/77_blackwell_fmha"),
            str(LOCAL_CUTLASS / "examples/77_blackwell_fmha/collective"),
            str(LOCAL_CUTLASS / "examples/77_blackwell_fmha/device"),
            str(LOCAL_CUTLASS / "examples/77_blackwell_fmha/kernel"),
        ])
    else:
        print("Warning: CUTLASS include directories not found. Some features may be limited.")

    include_dirs.extend(cpp_extension.include_paths())
    library_dirs = [cpp_extension.library_paths()[0]]

    # Libraries
    libraries = ["cudart", "cublas", "cublasLt"]

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
            "-rdc=true",
            *cuda_arch_flags,
            "-DUSE_CUTLASS",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_ENABLE_SM100_TCGEN05=1",
            "-DCUTE_ARCH_TMA_SM100_ENABLED=1",
            "-DCUTE_ARCH_TCGEN05_TMEM_ENABLED=1",
            "-DCUTLASS_ENABLE_SYNCLOG=0",
        ],
    }
else:
    # Empty definitions when not building extensions
    sources = []
    include_dirs = []
    libraries = []
    library_dirs = []
    extra_compile_args = {}

# Define extensions
if TORCH_AVAILABLE and not SKIP_BUILD_EXTENSIONS:
    if cuda_sources:
        # If we have CUDA sources, use CUDAExtension
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
        # Otherwise use CppExtension
        ext_modules = [
            cpp_extension.CppExtension(
                name="deepwell.cutlass_kernels",
                sources=cpp_sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args["cxx"],
                extra_link_args=["-Wl,-rpath,$ORIGIN"],
            )
        ]
else:
    # No extensions if torch not available or skipping build
    ext_modules = []
    print("Not building C++ extensions")

# Read README for long description
with open("README.md", "r") as f:
    long_description = f.read()

# Setup configuration
setup(
    name="deepwell",
    version="0.1.0",
    author="Deepwell Team",
    description="Automatic PyTorch optimization for NVIDIA Blackwell GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "deepwell": ["lib/*.so"],  # Include built libraries
    },
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CustomBuildExt,
        "install": CustomInstall,
        "develop": CustomDevelop,
    } if ext_modules else {},
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

print("\n" + "="*60)
print("Deepwell Installation Complete!")
print("="*60)
print("Automatic optimization is now available:")
print("  import deepwell")
print("  model = deepwell.optimize(your_model)")
print("="*60)