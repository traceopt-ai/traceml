import time
import pytest
from unittest.mock import patch, MagicMock
from traceml.samplers.process_sampler import ProcessSampler


class _MockNVMLProcess:
    def __init__(self, pid, used_mb):
        self.pid = pid
        self.usedGpuMemory = int(used_mb * 1024 * 1024)  # bytes


def test_process_sampler_real_or_mocked_gpu():
    """
    Use real GPU if available; otherwise mock NVML to simulate GPU memory usage.
    Always validates CPU/RAM sampling, snapshot creation, and summary fields.
    """
    # Probe environment first without mocks
    real_sampler = ProcessSampler()
    has_real_gpu = bool(real_sampler.gpu_available and real_sampler.gpu_count > 0)
    if has_real_gpu:
        # Use real NVML path
        for _ in range(3):
            env = real_sampler.sample()
            assert isinstance(env, dict) and "ok" in env
            time.sleep(0.05)

        snap = real_sampler.latest
        assert snap is not None
        assert snap.process_cpu_percent >= 0.0
        assert snap.process_ram > 0.0
        # If the process isn't using GPU memory, process_gpu_memory may be None
        assert (snap.process_gpu_memory is None) or (snap.process_gpu_memory >= 0.0)

        summary = real_sampler.get_summary()
        for key in [
            "total_process_samples",
            "cpu_average_percent",
            "cpu_peak_percent",
            "ram_average",
            "ram_peak",
            "gpu_average_memory",
            "gpu_peak_memory",
        ]:
            assert key in summary

    else:
        # No real GPU → mock NVML
        fake_pid = 43210
        fake_used_mb = 256.0

        with (
            patch("traceml.samplers.process_sampler.os.getpid", return_value=fake_pid),
            patch("traceml.samplers.process_sampler.psutil.Process") as mock_proc,
            patch("traceml.samplers.process_sampler.nvmlInit", return_value=None),
            patch(
                "traceml.samplers.process_sampler.nvmlDeviceGetCount", return_value=1
            ),
            patch(
                "traceml.samplers.process_sampler.nvmlDeviceGetHandleByIndex",
                return_value="handle0",
            ),
            patch(
                "traceml.samplers.process_sampler.nvmlDeviceGetComputeRunningProcesses",
                return_value=[
                    _MockNVMLProcess(fake_pid, fake_used_mb),
                    _MockNVMLProcess(
                        99999, 1024.0
                    ),  # another process; should be ignored
                ],
            ),
        ):
            proc_instance = MagicMock()
            proc_instance.cpu_percent.return_value = 12.5
            proc_instance.memory_info.return_value.rss = 80 * 1024 * 1024
            mock_proc.return_value = proc_instance

            sampler = ProcessSampler()

            for _ in range(3):
                env = sampler.sample()
                assert isinstance(env, dict) and "ok" in env and env["ok"] is True
                time.sleep(0.02)

            snap = sampler.latest
            assert snap is not None
            assert snap.process_cpu_percent == 12.5
            assert snap.process_ram == 80.0
            assert snap.process_gpu_memory == pytest.approx(
                fake_used_mb, rel=0, abs=0.1
            )

            summary = sampler.get_summary()
            assert summary["total_process_samples"] >= 1
            assert summary["cpu_average_percent"] >= 0.0
            assert summary["ram_average"] >= 0.0
            assert summary["gpu_average_memory"] == pytest.approx(
                fake_used_mb, rel=0, abs=0.1
            )
            assert summary["gpu_peak_memory"] == pytest.approx(
                fake_used_mb, rel=0, abs=0.1
            )


def test_process_sampler_nvml_init_failure_graceful_fallback():
    """
    Simulate NVML init failure and verify sampler falls back to gpu_available=False
    and still samples CPU/RAM.
    """
    try:
        from pynvml import NVMLError

        nvml_error = NVMLError(999)
    except Exception:
        nvml_error = Exception("NVML init fail")

    with patch("traceml.samplers.process_sampler.nvmlInit", side_effect=nvml_error):
        sampler = ProcessSampler()
        assert sampler.gpu_available is False
        assert sampler.gpu_count == 0

        env = sampler.sample()
        assert isinstance(env, dict) and "ok" in env
        snap = sampler.latest
        assert snap is not None
        assert snap.process_cpu_percent >= 0.0
        assert snap.process_ram > 0.0
        assert snap.process_gpu_memory is None

        summary = sampler.get_summary()
        for key in [
            "total_process_samples",
            "cpu_average_percent",
            "cpu_peak_percent",
            "ram_average",
            "ram_peak",
            "gpu_average_memory",
            "gpu_peak_memory",
        ]:
            assert key in summary


if __name__ == "__main__":
    test_process_sampler_real_or_mocked_gpu()
    test_process_sampler_nvml_init_failure_graceful_fallback()
