from dataclasses import dataclass
from typing import List, Optional


# Per-layer combined memory row
@dataclass(frozen=True)
class LayerCombinedMemoryRow:
    """
    Renderer-facing per-layer combined memory view.

    Semantics
    ---------
    - Values are already aggregated across ranks (worst-case view)
    - Param memory is static
    - Forward/backward values reflect the chosen safe step
    """

    layer: str
    param_memory: float
    forward_current: float
    forward_peak: float
    backward_current: float
    backward_peak: float
    total_current_memory: float
    total_peak_memory: float
    pct: float


# Aggregated "other layers" bucket
@dataclass(frozen=True)
class LayerCombinedOther:
    """
    Aggregated bucket for layers outside top-N.
    """

    param_memory: float
    forward_current: float
    forward_peak: float
    backward_current: float
    backward_peak: float
    total_current_memory: float
    pct: float


# Final renderer payload
@dataclass(frozen=True)
class LayerCombinedMemoryResult:
    """
    Final renderer-ready payload for combined layer memory.

    This is the *only* object the UI layer should consume.
    """

    model_index: Optional[int]

    top_items: List[LayerCombinedMemoryRow]
    other: LayerCombinedOther
    all_items: List[LayerCombinedMemoryRow]

    total_current_sum: float
    total_peak_sum: float
    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int

    status_message: str
