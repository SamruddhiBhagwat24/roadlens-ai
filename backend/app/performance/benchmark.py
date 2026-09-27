"""Performance measurement and stage latency profiler for RoadLens AI.

Strictly records measured elapsed times using high-precision monotonic clocks.
Never fabricates latency, FPS, or utilization numbers.
"""
import time
from typing import Dict, Any, Optional
from backend.app.performance.device import device_manager


class PipelineProfiler:
    """Records real execution timings across pipeline stages."""

    def __init__(self):
        self.stage_times: Dict[str, float] = {}
        self._start_total: float = 0.0
        self._current_stage: Optional[str] = None
        self._stage_start: float = 0.0

    def start_pipeline(self) -> None:
        self.stage_times.clear()
        self._start_total = time.perf_counter()

    def start_stage(self, stage_name: str) -> None:
        self._current_stage = stage_name
        self._stage_start = time.perf_counter()

    def end_stage(self, stage_name: Optional[str] = None) -> float:
        now = time.perf_counter()
        name = stage_name or self._current_stage
        if not name:
            return 0.0
        elapsed_ms = round((now - self._stage_start) * 1000.0, 2)
        self.stage_times[name] = elapsed_ms
        self._current_stage = None
        return elapsed_ms

    def finish_pipeline(self) -> Dict[str, Any]:
        total_ms = round((time.perf_counter() - self._start_total) * 1000.0, 2)
        self.stage_times["total"] = total_ms

        dev_info = device_manager.info
        gpu_mem = device_manager.get_gpu_memory_used_mb()

        return {
            "device_type": dev_info["device_type"],
            "device_name": dev_info["device_name"],
            "rocm_enabled": dev_info["rocm_enabled"],
            "latency_ms": self.stage_times.copy(),
            "gpu_utilization": None,  # Strictly None unless hardware profiler is attached
            "gpu_memory_mb": gpu_mem,
            "status_note": dev_info.get("status_note", "Execution successful"),
        }
