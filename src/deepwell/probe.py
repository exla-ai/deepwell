"""Hardware probing module for detecting GPU capabilities and Blackwell features."""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    sm_version: int  # e.g., 100 for SM100 (Blackwell), 90 for SM90 (Hopper)
    memory_gb: float
    memory_bandwidth_gb_s: float
    is_blackwell: bool
    blackwell_variant: Optional[str] = None  # 'sm100' or 'sm120'
    supports_mxfp8: bool = False
    supports_fp4: bool = False
    tensor_core_version: Optional[int] = None


@dataclass
class NVLinkInfo:
    """NVLink topology information."""
    version: int  # NVLink generation (e.g., 5 for Blackwell)
    bandwidth_gb_s: float
    topology_matrix: List[List[bool]]  # Adjacency matrix for NVLink connections
    nvswitch_present: bool = False


@dataclass
class HardwareConfig:
    """Complete hardware configuration."""
    gpus: List[GPUInfo]
    nvlink: Optional[NVLinkInfo]
    total_gpus: int
    cuda_version: Tuple[int, int]
    driver_version: str
    system_memory_gb: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'gpus': [
                {
                    'device_id': g.device_id,
                    'name': g.name,
                    'compute_capability': list(g.compute_capability),
                    'sm_version': g.sm_version,
                    'memory_gb': g.memory_gb,
                    'memory_bandwidth_gb_s': g.memory_bandwidth_gb_s,
                    'is_blackwell': g.is_blackwell,
                    'blackwell_variant': g.blackwell_variant,
                    'supports_mxfp8': g.supports_mxfp8,
                    'supports_fp4': g.supports_fp4,
                    'tensor_core_version': g.tensor_core_version
                }
                for g in self.gpus
            ],
            'nvlink': {
                'version': self.nvlink.version,
                'bandwidth_gb_s': self.nvlink.bandwidth_gb_s,
                'topology_matrix': self.nvlink.topology_matrix,
                'nvswitch_present': self.nvlink.nvswitch_present
            } if self.nvlink else None,
            'total_gpus': self.total_gpus,
            'cuda_version': list(self.cuda_version),
            'driver_version': self.driver_version,
            'system_memory_gb': self.system_memory_gb
        }


def _get_cuda_info() -> Tuple[Tuple[int, int], str]:
    """Get CUDA toolkit and driver version."""
    try:
        # Try to get CUDA version from nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # Parse version from output like "release 12.8, V12.8.89"
            import re
            match = re.search(r'release (\d+)\.(\d+)', output)
            if match:
                cuda_version = (int(match.group(1)), int(match.group(2)))
            else:
                cuda_version = (0, 0)
        else:
            cuda_version = (0, 0)
    except:
        cuda_version = (0, 0)
    
    try:
        # Get driver version using nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
        else:
            driver_version = "unknown"
    except:
        driver_version = "unknown"
    
    return cuda_version, driver_version


def _detect_gpu_info() -> List[GPUInfo]:
    """Detect GPU information using nvidia-smi and other tools."""
    gpus = []
    
    try:
        # Query GPU information using nvidia-smi
        query = 'gpu_name,compute_cap,memory.total,clocks.mem'
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return gpus
            
        lines = result.stdout.strip().split('\n')
        
        for device_id, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                name = parts[0]
                compute_cap = parts[1]
                memory_mb = float(parts[2]) if parts[2] != '[N/A]' else 0
                mem_clock_mhz = float(parts[3]) if parts[3] != '[N/A]' else 0
                
                # Parse compute capability
                cc_parts = compute_cap.split('.')
                cc = (int(cc_parts[0]), int(cc_parts[1])) if len(cc_parts) == 2 else (0, 0)
                
                # Determine SM version and capabilities
                sm_version = cc[0] * 10 + cc[1]
                
                # Check for Blackwell (SM100/SM120)
                is_blackwell = sm_version >= 100
                blackwell_variant = None
                supports_mxfp8 = False
                supports_fp4 = False
                tensor_core_version = None
                
                if sm_version == 100:
                    blackwell_variant = 'sm100'
                    supports_mxfp8 = True
                    supports_fp4 = True
                    tensor_core_version = 5  # 5th gen Tensor Cores
                elif sm_version == 120:
                    blackwell_variant = 'sm120'  # RTX 50 series
                    supports_mxfp8 = True
                    supports_fp4 = True
                    tensor_core_version = 5
                elif sm_version == 90:
                    # Hopper
                    supports_mxfp8 = True
                    tensor_core_version = 4
                elif sm_version >= 80:
                    # Ampere
                    tensor_core_version = 3
                elif sm_version >= 70:
                    # Volta/Turing
                    tensor_core_version = 1 if sm_version == 70 else 2
                
                # Estimate memory bandwidth (simplified)
                # For Blackwell B200: ~8TB/s, Hopper H100: ~3.35TB/s
                if is_blackwell:
                    memory_bandwidth_gb_s = 8000  # 8 TB/s for B200
                elif sm_version == 90:
                    memory_bandwidth_gb_s = 3350  # 3.35 TB/s for H100
                else:
                    # Rough estimate based on memory clock
                    memory_bandwidth_gb_s = mem_clock_mhz * 2 * 0.384  # Simplified
                
                gpu = GPUInfo(
                    device_id=device_id,
                    name=name,
                    compute_capability=cc,
                    sm_version=sm_version,
                    memory_gb=memory_mb / 1024,
                    memory_bandwidth_gb_s=memory_bandwidth_gb_s,
                    is_blackwell=is_blackwell,
                    blackwell_variant=blackwell_variant,
                    supports_mxfp8=supports_mxfp8,
                    supports_fp4=supports_fp4,
                    tensor_core_version=tensor_core_version
                )
                gpus.append(gpu)
                
    except Exception as e:
        # If nvidia-smi fails, return empty list
        pass
    
    return gpus


def _detect_nvlink_topology(num_gpus: int) -> Optional[NVLinkInfo]:
    """Detect NVLink topology between GPUs."""
    if num_gpus <= 1:
        return None
    
    try:
        # Use nvidia-smi to query NVLink status
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Parse NVLink information (simplified)
            # For Blackwell, expect 5th gen NVLink
            nvlink_version = 5 if 'NVLink 5' in output else 4
            bandwidth_gb_s = 1800 if nvlink_version == 5 else 900  # Simplified
            
            # Build adjacency matrix (simplified - assumes full connectivity)
            topology_matrix = [[i != j for j in range(num_gpus)] for i in range(num_gpus)]
            
            # Check for NVSwitch
            nvswitch_present = 'NVSwitch' in output or num_gpus >= 8
            
            return NVLinkInfo(
                version=nvlink_version,
                bandwidth_gb_s=bandwidth_gb_s,
                topology_matrix=topology_matrix,
                nvswitch_present=nvswitch_present
            )
    except:
        pass
    
    return None


def _get_system_memory() -> float:
    """Get system RAM in GB."""
    try:
        # Try to get memory info on Linux
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    parts = line.split()
                    mem_kb = int(parts[1])
                    return mem_kb / (1024 * 1024)
    except:
        pass
    
    # Fallback for macOS or if /proc/meminfo doesn't exist
    try:
        import platform
        if platform.system() == 'Darwin':
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split(':')[1].strip())
                return mem_bytes / (1024 * 1024 * 1024)
    except:
        pass
    
    return 0.0


def probe() -> HardwareConfig:
    """
    Probe hardware to detect GPU capabilities and Blackwell features.
    
    Returns:
        HardwareConfig object containing all detected hardware information.
    """
    # Get CUDA and driver info
    cuda_version, driver_version = _get_cuda_info()
    
    # Detect GPUs
    gpus = _detect_gpu_info()
    
    # Detect NVLink topology
    nvlink = _detect_nvlink_topology(len(gpus)) if gpus else None
    
    # Get system memory
    system_memory_gb = _get_system_memory()
    
    config = HardwareConfig(
        gpus=gpus,
        nvlink=nvlink,
        total_gpus=len(gpus),
        cuda_version=cuda_version,
        driver_version=driver_version,
        system_memory_gb=system_memory_gb
    )
    
    return config


def print_hardware_info(config: HardwareConfig) -> None:
    """Pretty print hardware configuration."""
    print("=" * 60)
    print("Hardware Configuration")
    print("=" * 60)
    print(f"CUDA Version: {config.cuda_version[0]}.{config.cuda_version[1]}")
    print(f"Driver Version: {config.driver_version}")
    print(f"System Memory: {config.system_memory_gb:.1f} GB")
    print(f"Total GPUs: {config.total_gpus}")
    
    if config.gpus:
        print("\nGPU Devices:")
        for gpu in config.gpus:
            print(f"  [{gpu.device_id}] {gpu.name}")
            print(f"      Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]} (SM{gpu.sm_version})")
            print(f"      Memory: {gpu.memory_gb:.1f} GB")
            print(f"      Memory Bandwidth: {gpu.memory_bandwidth_gb_s:.1f} GB/s")
            if gpu.is_blackwell:
                print(f"      Blackwell Architecture: {gpu.blackwell_variant}")
                print(f"      Supports MXFP8: {gpu.supports_mxfp8}")
                print(f"      Supports FP4: {gpu.supports_fp4}")
            if gpu.tensor_core_version:
                print(f"      Tensor Core Generation: {gpu.tensor_core_version}")
    
    if config.nvlink:
        print(f"\nNVLink Configuration:")
        print(f"  Version: NVLink {config.nvlink.version}")
        print(f"  Bandwidth: {config.nvlink.bandwidth_gb_s} GB/s")
        print(f"  NVSwitch Present: {config.nvlink.nvswitch_present}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the probe function
    config = probe()
    print_hardware_info(config)
    
    # Save to JSON for inspection
    with open('hardware_config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print("\nConfiguration saved to hardware_config.json")
