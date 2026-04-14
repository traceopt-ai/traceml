"""
Shared helpers for layer-level memory samplers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemoryPayload,
)


def aggregate_layer_memory_payload_max(
    layers: List[Tuple[str, float]],
) -> LayerForwardBackwardMemoryPayload:
    """
    Aggregate raw per-call layer memory observations using MAX.

    Memory is a capacity metric, so repeated observations within a step do not
    add together. We conservatively track the maximum observed value per layer.
    """
    agg: Dict[str, float] = {}

    for layer_name, mem in layers:
        mem = float(mem)
        prev = agg.get(layer_name)
        agg[layer_name] = mem if prev is None else max(prev, mem)

    layer_names = sorted(agg.keys())
    layer_bytes = [agg[name] for name in layer_names]

    return LayerForwardBackwardMemoryPayload(
        layer_names=layer_names,
        layer_memory_bytes=layer_bytes,
    )
