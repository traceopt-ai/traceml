import time
import sys
from unittest.mock import patch

import torch
import torch.nn as nn

from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.gradient_memory_sampler import (
    GradientMemorySampler,
    GradientSnapshot,
)
from traceml.renderers.layer_combined_memory_renderer import (
    LayerCombinedRenderer,
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


def test_gradient_sampler_with_tracker_and_backward_activity():
    """
    For a tiny model, runs several forward + backward passes to produce gradients,
    and checks that GradientMemorySampler produces a snapshot with layer + param stats.
    """
    model = _tiny_model()
    trace_model_instance(model)

    sampler1 = LayerMemorySampler()
    sampler2 = GradientMemorySampler()
    loggers1 = LayerCombinedRenderer()
    loggers2 = ActivationGradientRenderer()

    tracker = TrackerManager(
        components=[([sampler1, sampler2], [loggers1, loggers2])], interval_sec=0.25
    )

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

            snap = getattr(sampler2, "_latest_snapshot", None)
            assert isinstance(
                snap, GradientSnapshot
            ), "GradientSampler produced no snapshot"

            # ---- Layer + param stats ----
            layers = snap.layers or {}
            assert isinstance(layers, dict), "Expected 'layers' to be a dict"
            for lname, stats in layers.items():
                assert "current_peak" in stats
                assert "global_peak" in stats
                assert isinstance(stats["current_peak"], float)
                assert isinstance(stats["global_peak"], float)

                params = stats.get("params", {}) or {}
                for pname, pstats in params.items():
                    assert "current_peak" in pstats
                    assert "global_peak" in pstats
                    assert isinstance(pstats["current_peak"], float)
                    assert isinstance(pstats["global_peak"], float)

        finally:
            tracker.stop()
            tracker.log_summaries()
            CLIDisplayManager.stop_display()


if __name__ == "__main__":
    test_gradient_sampler_with_tracker_and_backward_activity()
