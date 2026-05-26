"""
Renderer-facing schema for combined layer execution time.

This schema represents a *capacity / tail-latency oriented* view of
per-layer execution time aggregated across DDP ranks.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LayerCombinedTimerRow:
    """
    Per-layer combined timing row (renderer-facing).

    Semantics
    ---------
    - Values are aggregated across ranks (worst-rank view)
    - Forward/backward values reflect the chosen safe step
    - `avg` fields are EMA-smoothed values
    """

    layer: str

    forward_current: float
    forward_avg: float
    forward_peak: float

    backward_current: float
    backward_avg: float
    backward_peak: float

    total_current: float
    total_avg: float
    total_peak: float

    pct: float
    worst_rank: Optional[int]
    on_gpu: bool


@dataclass(frozen=True)
class LayerCombinedTimerOther:
    """
    Aggregated bucket for layers outside top-N.
    """

    total_forward_current: float
    total_forward_avg: float
    total_backward_current: float
    total_backward_avg: float
    pct: float


@dataclass(frozen=True)
class LayerCombinedTimerResult:
    """
    Final renderer-ready payload for combined layer timing.

    This is the ONLY object UI layers should consume.
    """

    top_items: List[LayerCombinedTimerRow]
    all_items: List[LayerCombinedTimerRow]
    other: LayerCombinedTimerOther

    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int

    status_message: str
