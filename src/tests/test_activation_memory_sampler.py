import time
import sys
from unittest.mock import patch

import torch
import torch.nn as nn

from traceml.samplers.activation_memory_sampler import (
    ActivationMemorySampler,
    ActivationSnapshot,
)
from traceml.renderers.activation_gradient_memory_renderer import (
    ActivationGradientRenderer,
)

from traceml.manager.tracker_manager import TrackerManager
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.decorators import trace_model_instance


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
    and checks that ActivationSampler produces a snapshot with layer stats and
    global peak metrics.
    """
    model = _tiny_model()
    trace_model_instance(model)

    sampler = ActivationMemorySampler()
    loggers = ActivationGradientRenderer()

    tracker = TrackerManager(components=[([sampler], [loggers])], interval_sec=0.25)

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

            print(
                f"\n[TraceML Test] (ActivationSampler) forward-only iterations: {iterations}",
                file=sys.stderr,
            )
            time.sleep(0.35)

            snap = getattr(sampler, "_latest_snapshot", None)
            assert isinstance(
                snap, ActivationSnapshot
            ), "ActivationSampler produced no snapshot"

            # ---- Layer stats ----
            layers = snap.layers or {}
            assert isinstance(layers, dict), "Expected 'layers' to be a dict"
            for lname, stats in layers.items():
                assert "current_peak" in stats
                assert "global_peak" in stats
                assert isinstance(stats["current_peak"], float)
                assert isinstance(stats["global_peak"], float)

        finally:
            tracker.stop()
            tracker.log_summaries()
            CLIDisplayManager.stop_display()


if __name__ == "__main__":
    test_activation_sampler_with_tracker_and_registered_model_forward_activity()
