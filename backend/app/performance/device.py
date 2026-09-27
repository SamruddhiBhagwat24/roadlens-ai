"""Hardware device detection and acceleration management for RoadLens AI.

Supports AMD ROCm / HIP, NVIDIA CUDA, Apple MPS, and CPU fallback.
Strict Zero-Fabrication Policy:
- Never claims AMD acceleration unless running on AMD ROCm/HIP.
- Never fabricates GPU utilization, memory, or FPS metrics.
- Seamlessly falls back to CPU when AMD hardware is absent.
"""
import platform
from typing import Dict, Any, Optional

class DeviceManager:
    """Manages compute device detection, execution target, and hardware telemetry."""

    def __init__(self, preference: str = "auto"):
        self.preference = preference.lower().strip()
        self._device_info = self._detect_device()

    def _detect_device(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "device_type": "CPU",
            "device_name": f"{platform.processor() or platform.machine()} CPU",
            "torch_device": "cpu",
            "rocm_enabled": False,
            "rocm_version": None,
            "cuda_available": False,
            "mps_available": False,
            "gpu_count": 0,
            "gpu_memory_total_mb": None,
        }

        # If user explicitly requested CPU, bypass accelerator detection
        if self.preference == "cpu":
            info["status_note"] = "Explicitly configured for CPU execution"
            return info

        try:
            import torch

            # 1. Check for AMD ROCm / HIP
            hip_version = getattr(torch.version, "hip", None)
            cuda_available = torch.cuda.is_available()

            if cuda_available and hip_version is not None:
                # Confirmed AMD ROCm environment
                dev_id = 0
                dev_name = torch.cuda.get_device_name(dev_id)
                info.update({
                    "device_type": "AMD_ROCM",
                    "device_name": dev_name,
                    "torch_device": "cuda:0",
                    "rocm_enabled": True,
                    "rocm_version": str(hip_version),
                    "cuda_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "status_note": f"Accelerated via AMD ROCm HIP {hip_version} on {dev_name}",
                })
                try:
                    total_mem = torch.cuda.get_device_properties(dev_id).total_memory
                    info["gpu_memory_total_mb"] = round(total_mem / (1024 * 1024), 2)
                except Exception:
                    pass
                return info

            # 2. Check for standard NVIDIA CUDA
            if cuda_available:
                dev_id = 0
                dev_name = torch.cuda.get_device_name(dev_id)
                # Check if device name reveals an AMD GPU running under CUDA API
                if any(x in dev_name.upper() for x in ["AMD", "RADEON", "INSTINCT"]):
                    info.update({
                        "device_type": "AMD_ROCM",
                        "device_name": dev_name,
                        "torch_device": "cuda:0",
                        "rocm_enabled": True,
                        "cuda_available": True,
                        "gpu_count": torch.cuda.device_count(),
                        "status_note": f"Accelerated on AMD GPU: {dev_name}",
                    })
                else:
                    info.update({
                        "device_type": "CUDA",
                        "device_name": dev_name,
                        "torch_device": "cuda:0",
                        "rocm_enabled": False,
                        "cuda_available": True,
                        "gpu_count": torch.cuda.device_count(),
                        "status_note": f"Accelerated via CUDA on {dev_name}",
                    })
                return info

            # 3. Check for Apple Silicon MPS (local dev acceleration)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.update({
                    "device_type": "APPLE_MPS",
                    "device_name": f"Apple Silicon ({platform.processor() or 'MPS'})",
                    "torch_device": "mps",
                    "rocm_enabled": False,
                    "mps_available": True,
                    "status_note": "Local development on Apple Silicon MPS (CPU fallback available)",
                })
                return info

        except ImportError:
            info["status_note"] = "PyTorch not loaded; running on CPU using OpenCV/NumPy"
            return info
        except Exception as e:
            info["status_note"] = f"Accelerator detection fallback to CPU ({str(e)})"
            return info

        info["status_note"] = "No GPU accelerator detected; executing on CPU"
        return info

    @property
    def info(self) -> Dict[str, Any]:
        return self._device_info

    @property
    def device_type(self) -> str:
        return self._device_info["device_type"]

    @property
    def device_name(self) -> str:
        return self._device_info["device_name"]

    @property
    def is_rocm(self) -> bool:
        return self._device_info["rocm_enabled"]

    @property
    def torch_device(self) -> str:
        return self._device_info.get("torch_device", "cpu")

    def get_gpu_memory_used_mb(self) -> Optional[float]:
        """Returns measured GPU allocated memory, or None if unmeasured/CPU."""
        if not self._device_info.get("cuda_available", False):
            return None
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                return round(allocated / (1024 * 1024), 2)
        except Exception:
            return None
        return None


# Global device manager singleton
device_manager = DeviceManager()
