import time
import sys
import pytest
from unittest.mock import patch

import torch
import torch.nn as nn

from traceml.samplers.activation_memory_sampler import  ActivationMemorySampler, ActivationSnapshot
from traceml.loggers.stdout.activation_gradient_memory_logger import ActivationGradientMemoryStdoutLogger

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


def test_activation_sampler_with_tracker_and_registered_model_forward_activity():
    """
    For a tiny model, runs several forward passes to produce activations,
    and checks that ActivationSampler produces a snapshot with device stats and
    overall metrics
    """
    model = _tiny_model()
    trace_model_instance(model)

    sampler = ActivationMemorySampler()
    loggers = ActivationGradientMemoryStdoutLogger()

    tracker = TrackerManager(components=[(sampler, [loggers])], interval_sec=0.25)

    with patch("shutil.get_terminal_size", return_value=(120, 40)):
        try:
            tracker.start()

            end_time = time.time() + 2.0
            iterations = 0
            while time.time() < end_time:
                x = torch.randn(128, 32)
                with torch.no_grad():
                    _ = model(x)
                iterations += 1
                time.sleep(0.01)

            print(f"\n[TraceML Test] (ActivationSampler) forward-only iterations: {iterations}", file=sys.stderr)
            time.sleep(0.35)

            snap = getattr(sampler, "_latest_snapshot", None)
            assert isinstance(snap, ActivationSnapshot), "ActivationSampler produced no snapshot"

            # Expected shape (keep flexible): {'devices': {...}, 'overall_avg_mb': float, ...}
            devices = snap.devices or {}
            assert isinstance(devices, dict), "Expected 'devices' to be a dict in activation snapshot"

            for dev, stats in devices.items():
                assert isinstance(stats, dict), f"Device stats for {dev} must be a dict"
                for k in ("count", "sum_memory", "avg_memory", "max_memory", "min_nonzero_memory"):
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


# def test_activation_sampler_no_activity_yet_returns_empty_or_zero_safely():
#     """
#     Ensure calling sample/get_summary before any activation events does not crash
#     and returns empty or zero-like structures safely.
#     """
#     sampler = ActivationMemorySampler()
#     # No model registered, no forwards run
#     env = sampler.sample() if hasattr(sampler, "sample") else None
#
#     # sampler.latest/_latest_snapshot may be None; summary still should be a dict
#     summary = sampler.get_summary()
#     assert isinstance(summary, dict)
#
#     # If keys exist, they should be zeros or empty structures
#     for key in ("total_activation_events", "overall_avg_mb", "overall_peak_mb"):
#         if key in summary:
#             v = summary[key]
#             assert isinstance(v, (int, float))
#             assert v >= 0


if __name__ == "__main__":
    test_activation_sampler_with_tracker_and_registered_model_forward_activity()
    # test_activation_sampler_no_activity_yet_returns_empty_or_zero_safely()
