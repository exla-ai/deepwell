from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class DeviceInfo(TypedDict, total=False):
    id: int
    name: str
    sm: str
    hbm_gb: float


class LinkInfo(TypedDict, total=False):
    src: int
    dst: int
    kind: str
    hops: int


class ProbeResult(TypedDict, total=False):
    devices: List[DeviceInfo]
    links: List[LinkInfo]
    hbw_gbps: Dict[str, float]


def probe() -> ProbeResult:
    devices: List[DeviceInfo] = []
    links: List[LinkInfo] = []
    hbw_gbps: Dict[str, float] = {}

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            hbm_gb = float(mem_info.total) / (1024**3)
            try:
                sm = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                sm_str = f"sm{sm[0]}{sm[1]}"
            except Exception:
                sm_str = "unknown"
            devices.append(DeviceInfo(id=i, name=name, sm=sm_str, hbm_gb=hbm_gb))
        # Links and bandwidth estimates are left empty for now; filled by NCCL topo later.
        pynvml.nvmlShutdown()
    except Exception:
        # Safe fallback when NVML is unavailable.
        pass

    return ProbeResult(devices=devices, links=links, hbw_gbps=hbw_gbps)


