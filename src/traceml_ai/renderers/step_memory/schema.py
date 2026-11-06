"""Renderer-specific result for combined Step Memory measurements."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from traceml_ai.measurements.step_memory import StepMemoryCombinedMetric
from traceml_ai.renderers.shared.freshness import RankReporting


@dataclass(frozen=True)
class StepMemoryCombinedResult:
    """
    Final renderer-facing payload for step memory combined.

    This is the ONLY object UI layers should consume.

    `gpu_total_bytes` is best-effort device capacity read from process
    telemetry. It is None when the run reported no capacity, and diagnosis
    then simply cannot evaluate capacity-relative rules.

    `rank_reporting` is every rank's Process reporting status, set by the
    terminal computer only. The metrics align on the slowest rank's latest
    step, so a rank that stopped freezes them; this names it. None when
    the status was not read or could not be read; empty when it was read
    and no rank has reported.
    """

    metrics: List[StepMemoryCombinedMetric]
    status_message: str
    gpu_total_bytes: Optional[float] = None
    rank_reporting: Optional[Tuple[RankReporting, ...]] = None
