"""Performance benchmarking and hardware acceleration package for RoadLens AI."""
from backend.app.performance.device import device_manager, DeviceManager
from backend.app.performance.benchmark import PipelineProfiler

__all__ = [
    "device_manager",
    "DeviceManager",
    "PipelineProfiler",
]
