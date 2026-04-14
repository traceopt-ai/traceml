"""
Shared helpers for layer-level time samplers.
"""

from __future__ import annotations

from typing import Dict, Optional

from traceml.samplers.schema.layer_forward_backward_time import (
    LayerForwardBackwardTimePayload,
)


def all_layer_events_resolved(layers) -> bool:
    """
    Return True when all layer timing events in one step have resolved.
    """
    for layer_evt in layers:
        if not layer_evt.try_resolve():
            return False
    return True


def aggregate_layer_time_payload(layers) -> LayerForwardBackwardTimePayload:
    """
    Aggregate per-call layer timing events into a per-layer payload.

    Aggregation semantics
    ---------------------
    - CPU time: summed across calls
    - GPU time: summed across calls when available
    - Call count: total number of invocations
    """
    agg: Dict[str, Dict[str, Optional[float]]] = {}

    for evt in layers:
        rec = agg.setdefault(
            evt.layer_name,
            {
                "cpu_ms": 0.0,
                "gpu_ms": None,
                "n_calls": 0,
            },
        )

        rec["cpu_ms"] += float(evt.cpu_duration_ms)

        if evt.gpu_duration_ms is not None:
            rec["gpu_ms"] = (rec["gpu_ms"] or 0.0) + float(evt.gpu_duration_ms)

        rec["n_calls"] += 1

    layer_names = sorted(agg.keys())

    return LayerForwardBackwardTimePayload(
        layer_names=layer_names,
        cpu_time_ms=[agg[k]["cpu_ms"] for k in layer_names],
        gpu_time_ms=[agg[k]["gpu_ms"] for k in layer_names],
        n_calls=[int(agg[k]["n_calls"]) for k in layer_names],
    )
