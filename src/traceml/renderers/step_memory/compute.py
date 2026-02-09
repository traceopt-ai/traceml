"""
Step Memory Combined Computation (DDP-aware, rank-agnostic)

Pure compute layer for TraceML step-level peak memory metrics.

- Reads StepMemorySample from RemoteDBStore
- Aligns steps across ranks via intersection (up to min completed step)
- Aggregates per-step across ranks (median + worst)
- Computes window summary (median peak, worst peak, imbalance)
- Outputs renderer-facing StepMemoryCombinedResult
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.step_memory import StepMemorySample

from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedResult,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)


class StepMemoryCombinedComputer:
    """
    Rank-agnostic computation of step-level memory combined metrics.

    Notes
    -----
    - This computation is rank-agnostic: it aggregates across all ranks found
      in the RemoteDBStore.
    - Step alignment is intersection-based to avoid index drift.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
        table_name: str = "step_memory",
        sampler_name: str = "StepMemorySampler",
    ) -> None:
        self.store = remote_store
        self.window_size = int(window_size)
        self.table_name = table_name
        self.sampler_name = sampler_name
        self.logger = get_error_logger("StepMemoryCombinedComputer")

        # Two output metrics derived from StepMemorySample
        self._metrics = [
            ("peak_allocated",),
            ("peak_reserved",),
        ]

    def compute(self) -> StepMemoryCombinedResult:
        """
        Compute combined step memory metrics across ranks.
        """
        ranks = list(self.store.ranks())
        world_size = len(ranks)

        if world_size == 0:
            return StepMemoryCombinedResult(
                metrics=[], status_message="No ranks available"
            )

        out: List[StepMemoryCombinedMetric] = []

        # Compute both metrics independently (allocated, reserved)
        for (metric_key,) in self._metrics:
            m = self._compute_one(
                metric_key=metric_key, ranks=ranks, world_size=world_size
            )
            if m is not None:
                out.append(m)

        status = "OK" if out else "No complete memory metrics available"
        return StepMemoryCombinedResult(metrics=out, status_message=status)

    # ---------------------------------------------------------------------
    # Core computation
    # ---------------------------------------------------------------------

    def _compute_one(
        self,
        *,
        metric_key: str,
        ranks: List[int],
        world_size: int,
    ) -> Optional[StepMemoryCombinedMetric]:
        """
        Compute one combined memory metric across ranks.

        metric_key:
          - "peak_allocated"
          - "peak_reserved"
        """
        per_rank_steps: Dict[int, Dict[int, float]] = {}
        per_rank_completed: List[int] = []
        per_rank_device: Dict[int, Optional[str]] = {}

        for rank in ranks:
            db = self.store.get_db(rank, self.sampler_name)
            if not db:
                continue

            tbl = db.get_table(self.table_name)
            if not tbl:
                continue

            step_map: Dict[int, float] = {}
            device: Optional[str] = None

            # Walk backwards to get most recent samples
            for row in reversed(tbl):
                try:
                    s = StepMemorySample.from_wire(row)
                except Exception:
                    continue

                # Pick the metric value
                v = self._extract_metric_value(s, metric_key)
                if v is None:
                    continue

                if s.step is None:
                    continue

                step_map[int(s.step)] = float(v)
                device = s.device or device

                if len(step_map) >= self.window_size:
                    break

            if step_map:
                per_rank_steps[rank] = step_map
                per_rank_device[rank] = device
                per_rank_completed.append(max(step_map.keys()))

        if not per_rank_steps:
            return None

        ranks_present = len(per_rank_steps)
        incomplete = ranks_present < world_size

        completed_step = min(per_rank_completed)

        # Intersection step alignment across ranks (only steps <= completed_step)
        common_steps = self._intersect_steps(per_rank_steps, completed_step)
        if not common_steps:
            return None

        steps = sorted(common_steps)[-self.window_size :]

        # Per-step aggregation across ranks
        median: List[float] = []
        worst: List[float] = []

        for s in steps:
            vals = [
                per_rank_steps[r][s]
                for r in per_rank_steps
                if s in per_rank_steps[r]
            ]
            if not vals:
                # Defensive: should not happen due to intersection, but keep safe
                return None

            vals.sort()
            median.append(float(np.median(vals)))
            worst.append(float(vals[-1]))

        # Window summary: compute per-rank peak over the window, then median/worst across ranks
        per_rank_peak: Dict[int, float] = {}
        for r, step_map in per_rank_steps.items():
            peaks = [step_map[s] for s in steps if s in step_map]
            if not peaks:
                continue
            per_rank_peak[r] = float(max(peaks))

        if not per_rank_peak:
            return None

        peaks = list(per_rank_peak.values())
        median_peak = float(np.median(peaks))
        worst_peak = float(max(peaks))
        worst_rank = max(per_rank_peak, key=lambda rr: per_rank_peak[rr])

        skew_ratio = worst_peak / median_peak if median_peak > 0 else 0.0
        skew_pct = (
            (worst_peak - median_peak) / median_peak
            if median_peak > 0
            else 0.0
        )

        device = self._majority_device(per_rank_device)

        return StepMemoryCombinedMetric(
            metric=metric_key,
            device=device,
            series=StepMemoryCombinedSeries(
                steps=steps,
                median=median,
                worst=worst,
            ),
            summary=StepMemoryCombinedSummary(
                window_size=self.window_size,
                steps_used=len(steps),
                median_peak=median_peak,
                worst_peak=worst_peak,
                worst_rank=worst_rank,
                skew_ratio=skew_ratio,
                skew_pct=skew_pct,
            ),
            coverage=StepMemoryCombinedCoverage(
                expected_steps=self.window_size,
                steps_used=len(steps),
                completed_step=completed_step,
                world_size=world_size,
                ranks_present=ranks_present,
                incomplete=incomplete,
            ),
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_metric_value(
        s: StepMemorySample, metric_key: str
    ) -> Optional[float]:
        if metric_key == "peak_allocated":
            return s.peak_allocated
        if metric_key == "peak_reserved":
            return s.peak_reserved
        return None

    @staticmethod
    def _intersect_steps(
        per_rank_steps: Dict[int, Dict[int, float]],
        completed_step: int,
    ) -> List[int]:
        common: Optional[set[int]] = None
        for step_map in per_rank_steps.values():
            steps = {st for st in step_map.keys() if st <= completed_step}
            common = steps if common is None else common & steps
        return sorted(common) if common else []

    @staticmethod
    def _majority_device(
        per_rank_device: Dict[int, Optional[str]],
    ) -> Optional[str]:
        devices = [d for d in per_rank_device.values() if d]
        if not devices:
            return None
        return max(set(devices), key=devices.count)
