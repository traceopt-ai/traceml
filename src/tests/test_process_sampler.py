import time
import sys
import numpy as np
from unittest.mock import patch

from traceml.samplers.process_sampler import ProcessSampler
from traceml.manager.tracker_manager import TrackerManager
from traceml.loggers.stdout.system_process_logger import SystemProcessStdoutLogger
from traceml.loggers.stdout.display_manager import StdoutDisplayManager


def test_process_sampler_with_heavy_task():
    """
    Runs a short CPU+RAM heavy workload
    """
    sampler = ProcessSampler()
    stdout_logger = SystemProcessStdoutLogger()
    tracker_components = [(sampler, [stdout_logger])]

    tracker = TrackerManager(components=tracker_components, interval_sec=0.5)
    try:
        tracker.start()
        test_duration = 5
        end_time = time.time() + test_duration
        iterations = 0

        # Burn some CPU/RAM in the current process (the one ProcessSampler should watch)
        while time.time() < end_time:
            a = np.random.randn(200_000, 50)
            b = np.random.randn(50, 100)
            c = a @ b
            _ = float(c.sum())
            iterations += 1
            time.sleep(0.01)

        print(
            f"\n[TraceML Test] (ProcessSampler) Heavy task finished ({iterations} iterations).",
            file=sys.stderr,
        )

        # Snapshot checks
        snap = getattr(sampler, "latest", None)
        assert snap is not None, "ProcessSampler did not produce a snapshot"

        assert hasattr(snap, "process_cpu_percent"), \
            "Expected process CPU metric on snapshot"
        assert hasattr(snap, "process_ram"), \
            "Expected process RAM metric on snapshot"
        assert hasattr(snap, "process_gpu_memory"), \
            "Expected process GPU memory metric on snapshot"

        # Summary should at least be a dict; keys may vary by implementation
        summary = sampler.get_summary()
        assert isinstance(summary, dict)

        # If common keys exist, sanity check their types/values
        for k in [
            "total_process_samples",
            "process_average_cpu",
            "process_peak_cpu",
            "process_average_ram",
            "process_peak_ram",
        ]:
            if k in summary:
                v = summary[k]
                assert isinstance(v, (int, float)), f"Summary key {k} should be numeric"

    finally:
        tracker.stop()
        tracker.log_summaries()
        StdoutDisplayManager.stop_display()


def test_process_sampler_handles_missing_nvml_gracefully():
    """
    If ProcessSampler internally tries to use NVML, simulate failures to
    ensure it degrades gracefully (e.g., process_gpu_memory stays None).
    """
    # Simulate NVML import or init failure paths
    with (
            patch("traceml.samplers.process_sampler.nvmlInit",
                  return_value=None
            ),
            patch("traceml.samplers.process_sampler.nvmlDeviceGetCount",
                 return_value=1
            ),
    ):
        sampler = ProcessSampler()
        # Take a couple samples
        for _ in range(2):
            _ = sampler.sample()
            time.sleep(0.05)

        snap = getattr(sampler, "latest", None)
        assert snap is not None, \
        "ProcessSampler did not produce a snapshot under NVML failure simulation"

        if hasattr(snap, "process_gpu_memory"):
            val = snap.process_gpu_memory
            assert (val is None) or (isinstance(val, (int, float)) and val >= 0.0)

        summary = sampler.get_summary()
        assert isinstance(summary, dict)


def test_process_sampler_multiple_samples_summary_trends():
    sampler = ProcessSampler()

    for _ in range(5):
        # light workload
        a = np.random.randn(50_000, 50)
        b = np.random.randn(50, 50)
        _ = (a @ b).sum()
        _ = sampler.sample()
        time.sleep(0.05)

    summary = sampler.get_summary()
    assert isinstance(summary, dict)

    numeric_keys = [
        "total_process_samples",
        "process_average_cpu",
        "process_peak_cpu",
        "process_average_ram",
        "process_peak_ram",
    ]
    for k in numeric_keys:
        if k in summary:
            assert isinstance(summary[k], (int, float))


if __name__ == "__main_l_":
    test_process_sampler_with_heavy_task()
    test_process_sampler_handles_missing_nvml_gracefully()
    test_process_sampler_multiple_samples_summary_trends()
