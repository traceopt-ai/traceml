import time
import sys
import numpy as np
from unittest.mock import patch

from traceml.samplers.system_sampler import SystemSampler
from traceml.manager.tracker_manager import TrackerManager
from traceml.renderers.system_process_renderer import SystemProcessRenderer
from traceml.renderers.display.cli_display_manager import CLIDisplayManager


class _MockUtilization:
    def __init__(self, gpu=42):
        self.gpu = gpu


class _MockMemInfo:
    def __init__(self, used=512, total=4096):
        self.used = int(used * 1024 * 1024)  # bytes
        self.total = int(total * 1024 * 1024)  # bytes


def test_system_sampler_with_heavy_task():
    """
    Runs a short CPU+RAM heavy workload and ensures SystemSampler:
      - Produces snapshots
      - get_summary() returns consistent fields
      - Works regardless of GPU presence
    """
    system_sampler = SystemSampler()
    system_process_stdout_logger = SystemProcessRenderer()
    tracker_components = [([system_sampler], [system_process_stdout_logger])]

    tracker = TrackerManager(components=tracker_components, interval_sec=0.5)
    try:
        tracker.start()
        test_duration = 5
        end_time = time.time() + test_duration
        iterations = 0
        while time.time() < end_time:
            # Matrix multiply (burn CPU and allocate RAM)
            a = np.random.randn(200_000, 50)
            b = np.random.randn(50, 100)
            c = a @ b
            _ = float(c.sum())
            iterations += 1
            time.sleep(0.01)

        print(
            f"\n[TraceML Test] Heavy task finished ({iterations} iterations).",
            file=sys.stderr,
        )

        # Snapshot checks
        snap = system_sampler.latest
        assert snap is not None, "SystemSampler did not produce a snapshot"
        assert snap.cpu_percent >= 0.0
        assert snap.ram_used > 0.0
        assert snap.ram_total >= snap.ram_used
        assert isinstance(snap.gpu_available, bool)
        assert isinstance(snap.gpu_count, int)

        summary = system_sampler.get_summary()
        assert isinstance(summary, dict)
        for key in [
            "total_samples",
            "cpu_average_percent",
            "cpu_peak_percent",
            "ram_average_used",
            "ram_peak_used",
        ]:
            assert key in summary, f"Missing summary key: {key}"

        if snap.gpu_available and snap.gpu_count > 0:
            for key in [
                "gpu_average_util_percent",
                "gpu_peak_util_percent",
                "gpu_memory_peak_used",
                "gpu_memory_average_used",
                "gpu_memory_total",
            ]:
                assert key in summary, f"Missing GPU summary key (GPU present): {key}"

    finally:
        tracker.stop()
        tracker.log_summaries()
        CLIDisplayManager.stop_display()


def test_system_sampler_handles_nvml_errors_gracefully():
    try:
        from pynvml import NVMLError

        nvml_error = NVMLError(999)
    except Exception:
        nvml_error = Exception("NVML init fail")

    with patch("traceml.samplers.system_sampler.nvmlInit", side_effect=nvml_error):
        sampler = SystemSampler()
        assert sampler.gpu_available is False
        assert sampler.gpu_count == 0

        envelope = sampler.sample()
        assert isinstance(envelope, dict)
        assert "ok" in envelope
        summary = sampler.get_summary()
        assert isinstance(summary, dict)


def test_system_sampler_gpu_present_or_mocked():
    """
    Ensures we validate GPU metric paths regardless of actual hardware:
      - If a real GPU is present, use real NVML calls.
      - If not, mock NVML to simulate one GPU (util=42%, mem=512/4096MB).
    """
    real_sampler = SystemSampler()
    has_real_gpu = bool(real_sampler.gpu_available and real_sampler.gpu_count > 0)

    if has_real_gpu:
        sampler = real_sampler
        for _ in range(3):
            _ = sampler.sample()
            time.sleep(0.05)

        snap = sampler.latest
        assert snap is not None
        assert snap.gpu_available is True
        assert snap.gpu_count >= 1
        assert snap.gpu_util_avg is not None
        assert snap.gpu_mem_sum_used is not None
        summary = sampler.get_summary()
        for key in [
            "gpu_average_util_percent",
            "gpu_peak_util_percent",
            "gpu_memory_peak_used",
            "gpu_memory_average_used",
            "gpu_memory_total",
        ]:
            assert key in summary, f"Missing GPU summary key (real GPU): {key}"

    else:
        # No real GPU → mock the NVML stack to simulate one GPU
        with (
            patch("traceml.samplers.system_sampler.nvmlInit", return_value=None),
            patch("traceml.samplers.system_sampler.nvmlDeviceGetCount", return_value=1),
            patch(
                "traceml.samplers.system_sampler.nvmlDeviceGetHandleByIndex",
                return_value="handle0",
            ),
            patch(
                "traceml.samplers.system_sampler.nvmlDeviceGetUtilizationRates",
                return_value=_MockUtilization(gpu=42),
            ),
            patch(
                "traceml.samplers.system_sampler.nvmlDeviceGetMemoryInfo",
                return_value=_MockMemInfo(used=512, total=4096),
            ),
        ):
            sampler = SystemSampler()
            assert sampler.gpu_available is True
            assert sampler.gpu_count == 1

            # Take a few samples to populate histories
            for _ in range(3):
                _ = sampler.sample()
                time.sleep(0.02)

            snap = sampler.latest
            assert snap is not None
            assert snap.gpu_available is True
            assert snap.gpu_count == 1
            assert snap.gpu_util_avg is not None
            assert 0 <= snap.gpu_util_avg <= 100
            assert snap.gpu_mem_total is not None
            assert snap.gpu_mem_total >= 4096 - 1  # allow rounding
            assert snap.gpu_mem_sum_used is not None
            assert snap.gpu_mem_sum_used >= 512 - 1  # allow rounding

            # Summary keys for GPU-present path
            summary = sampler.get_summary()
            for key in [
                "gpu_average_util_percent",
                "gpu_peak_util_percent",
                "gpu_memory_peak_used",
                "gpu_memory_average_used",
                "gpu_memory_total",
            ]:
                assert key in summary, f"Missing GPU summary key (mocked GPU): {key}"


if __name__ == "__main__":
    test_system_sampler_with_heavy_task()
    test_system_sampler_handles_nvml_errors_gracefully()
    test_system_sampler_gpu_present_or_mocked()
