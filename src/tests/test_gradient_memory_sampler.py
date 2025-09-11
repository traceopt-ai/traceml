import time
import sys
from unittest.mock import patch

import torch
import torch.nn as nn

from traceml.samplers.gradient_memory_sampler import (
    GradientMemorySampler,
    GradientSnapshot,
)
from traceml.loggers.stdout.activation_gradient_memory_logger import (
    ActivationGradientMemoryStdoutLogger,
)

from traceml.manager.tracker_manager import TrackerManager
from traceml.loggers.stdout.display_manager import StdoutDisplayManager
from traceml.decorator import trace_model_instance


def _tiny_model():
    return nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
    ).to("cpu")


def test_gradient_sampler_with_tracker_and_backward_activity():
    """
    For a tiny model, runs several forward + backward passes to produce gradients,
    and checks that GradientMemorySampler produces a snapshot with device stats and
    overall metrics.
    """
    model = _tiny_model()
    trace_model_instance(model, include_module=True)

    sampler = GradientMemorySampler()
    loggers = ActivationGradientMemoryStdoutLogger()

    tracker = TrackerManager(components=[(sampler, [loggers])], interval_sec=0.25)

    with patch("shutil.get_terminal_size", return_value=(120, 40)):
        try:
            tracker.start()

            end_time = time.time() + 2.0
            iterations = 0
            while time.time() < end_time:
                x = torch.randn(128, 32)
                y = model(x).sum()
                y.backward()  # 🔑 triggers gradient hooks
                model.zero_grad()
                iterations += 1
                time.sleep(0.01)

            print(
                f"\n[TraceML Test] (GradientSampler) forward+backward iterations: {iterations}",
                file=sys.stderr,
            )
            time.sleep(0.35)

            snap = getattr(sampler, "_latest_snapshot", None)
            assert isinstance(
                snap, GradientSnapshot
            ), "GradientSampler produced no snapshot"

            # Expected shape (keep flexible): {'devices': {...}, 'overall_avg_mb': float, ...}
            devices = snap.devices or {}
            assert isinstance(
                devices, dict
            ), "Expected 'devices' to be a dict in gradient snapshot"

            for dev, stats in devices.items():
                assert isinstance(stats, dict), f"Device stats for {dev} must be a dict"
                for k in (
                    "count",
                    "sum_memory",
                    "avg_memory",
                    "max_memory",
                    "min_nonzero_memory",
                ):
                    if k in stats:
                        v = stats[k]
                        if k == "count":
                            assert isinstance(v, int) and v >= 0
                        else:
                            assert isinstance(v, (int, float)) and v >= 0.0

            # ---- Summary checks ----
            summary = sampler.get_summary()
            assert isinstance(summary, dict)
            assert summary["ever_seen"] > 0
            assert isinstance(summary["per_device_cumulative"], dict)

        finally:
            tracker.stop()
            tracker.log_summaries()
            StdoutDisplayManager.stop_display()


def test_gradient_sampler_no_activity_yet_returns_empty_or_zero_safely():
    """
    Ensure calling sample/get_summary before any backward events does not crash
    and returns empty or zero-like structures safely.
    """
    sampler = GradientMemorySampler()
    # No model registered, no backward runs
    sampler.sample() if hasattr(sampler, "sample") else None

    summary = sampler.get_summary()
    assert isinstance(summary, dict)

    for key in ("ever_seen", "raw_events_kept"):
        if key in summary:
            v = summary[key]
            assert isinstance(v, (int, float))
            assert v >= 0


if __name__ == "__main__":
    test_gradient_sampler_with_tracker_and_backward_activity()
    test_gradient_sampler_no_activity_yet_returns_empty_or_zero_safely()
