"""Unit tests for AMD ROCm detection architecture and CPU fallback."""
import unittest
from backend.app.performance.device import DeviceManager, device_manager


class TestDeviceArchitecture(unittest.TestCase):
    """Verifies acceleration detection and zero-fabrication guarantees."""

    def test_cpu_fallback_mode(self):
        # Force CPU mode
        cpu_manager = DeviceManager(preference="cpu")
        self.assertEqual(cpu_manager.device_type, "CPU")
        self.assertFalse(cpu_manager.is_rocm)
        self.assertIsNone(cpu_manager.get_gpu_memory_used_mb())

    def test_global_device_manager_initialization(self):
        self.assertIsNotNone(device_manager.device_type)
        self.assertIsNotNone(device_manager.device_name)
        # Verify ROCm is boolean
        self.assertIsInstance(device_manager.is_rocm, bool)

    def test_zero_fabrication_contract(self):
        info = device_manager.info
        if not info.get("cuda_available", False):
            # If not running on GPU, memory must be None (never a fabricated number)
            self.assertIsNone(device_manager.get_gpu_memory_used_mb())


if __name__ == "__main__":
    unittest.main()
