from queue import Queue
from unittest.mock import patch
import time
import sys
import numpy as np
import torch.nn as nn
import pytest

from traceml.samplers.layer_memory_sampler import (
    LayerMemorySampler,
    ModelMemorySnapshot,
)
from traceml.renderers.layer_combined_memory_renderer import (
    LayerCombinedRenderer,
)
from traceml.manager.tracker_manager import TrackerManager
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.decorators import trace_model_instance


def _make_linear(in_f=8, out_f=4):
    return nn.Linear(in_f, out_f)


def _sum_param_mem_mb(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.named_parameters():
        t = p[1]
        total_bytes += t.element_size() * t.nelement()
    return round(total_bytes, 4)


def _queue_with_models(*models):
    q = Queue()
    for m in models:
        q.put(m)
    return q


def test_layer_memory_sampler_from_queue_basic():
    model = nn.Sequential(_make_linear(16, 8), nn.ReLU(), _make_linear(8, 2))
    expected_total = _sum_param_mem_mb(model)

    sampler = LayerMemorySampler()

    with patch(
        "traceml.samplers.layer_memory_sampler.get_model_queue",
        return_value=_queue_with_models(model),
    ):
        env = sampler.sample()

    snap = sampler._latest_snapshot
    assert isinstance(snap, ModelMemorySnapshot)
    assert snap.error is None

    assert isinstance(env, dict)
    assert env.get("ok") is True
    assert "data" in env and isinstance(env["data"], dict)

    layer_mem = snap.layer_memory
    assert isinstance(layer_mem, dict) and len(layer_mem) > 0

    layers_sum = round(sum(layer_mem.values()), 4)
    assert pytest.approx(snap.total_memory, rel=0, abs=1e-3) == layers_sum
    assert pytest.approx(snap.total_memory, rel=0, abs=1e-3) == expected_total


def test_layer_memory_sampler_deduplicates_by_signature():
    # Two models with the same parameter shapes → identical signature
    m1 = nn.Sequential(_make_linear(32, 16), nn.ReLU(), _make_linear(16, 4))
    m2 = nn.Sequential(_make_linear(32, 16), nn.ReLU(), _make_linear(16, 4))

    sampler = LayerMemorySampler()

    with patch(
        "traceml.samplers.layer_memory_sampler.get_model_queue",
        return_value=_queue_with_models(m1, m2),
    ):
        env1 = sampler.sample()

    assert env1.get("ok") is True
    assert sampler.total_samples == 1
    assert len(sampler.memory_history) == 1
    assert len(sampler.seen_signatures) == 1


def test_layer_memory_sampler_multiple_models_summary():
    a = nn.Sequential(_make_linear(10, 5), nn.ReLU(), _make_linear(5, 2))
    b = nn.Sequential(_make_linear(64, 32), nn.ReLU(), _make_linear(32, 8))

    sampler = LayerMemorySampler()

    with patch(
        "traceml.samplers.layer_memory_sampler.get_model_queue",
        return_value=_queue_with_models(a, b),
    ):
        sampler.sample()
        sampler.sample()

    assert sampler.total_samples == 2
    assert len(sampler.seen_signatures) == 2
    assert len(sampler.memory_history) == 2


def test_layer_memory_sampler_empty_queue_returns_no_model_found():
    sampler = LayerMemorySampler()

    with patch(
        "traceml.samplers.layer_memory_sampler.get_model_queue",
        return_value=_queue_with_models(),
    ):
        env = sampler.sample()

    assert isinstance(env, dict)
    assert env.get("ok") is False
    assert env.get("message") == "no model found"
    assert sampler._latest_snapshot is None


def test_layer_memory_sampler_with_tracker_and_registered_model():
    # Define a small model
    model = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    ).to("cpu")

    trace_model_instance(model)

    sampler = LayerMemorySampler()
    stdout_logger = LayerCombinedRenderer()
    tracker = TrackerManager(
        components=[([sampler], [stdout_logger])], interval_sec=0.25
    )

    try:
        tracker.start()

        end_time = time.time() + 2.0
        iterations = 0
        while time.time() < end_time:
            a = np.random.randn(2_000, 16)
            b = np.random.randn(16, 4)
            _ = float((a @ b).sum())
            iterations += 1
            time.sleep(0.01)

        print(
            f"\n[TraceML Test] (LayerMemorySampler) loop iterations: {iterations}",
            file=sys.stderr,
        )

        # Give tracker a moment to flush one more tick
        time.sleep(0.35)

        # ---- Snapshot checks ----
        snap = getattr(sampler, "_latest_snapshot", None)
        assert isinstance(
            snap, ModelMemorySnapshot
        ), "No model memory snapshot produced"
        assert snap.error is None, f"Snapshot error: {snap.error}"
        assert (
            isinstance(snap.layer_memory, dict) and len(snap.layer_memory) > 0
        ), "Layer memory dict should be non-empty"

        # Total matches sum of layers
        layers_sum = round(sum(float(v) for v in snap.layer_memory.values()), 4)
        assert pytest.approx(snap.total_memory, rel=0, abs=1e-3) == layers_sum

    finally:
        tracker.stop()
        tracker.log_summaries()
        CLIDisplayManager.stop_display()


if __name__ == "__main__":
    test_layer_memory_sampler_from_queue_basic()
    test_layer_memory_sampler_deduplicates_by_signature()
    test_layer_memory_sampler_multiple_models_summary()
    test_layer_memory_sampler_empty_queue_returns_no_model_found()
    test_layer_memory_sampler_with_tracker_and_registered_model()
