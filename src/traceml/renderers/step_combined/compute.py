"""
Model Step Summary Metric Computation (DDP-aware, rank-agnostic)

This module implements the *pure computation layer* for TraceML's
step-level model summary metrics.

Key properties
--------------
- Reads ONLY from RemoteDBStore
- Rank-agnostic (no rank 0 special casing)
- No rendering, formatting, or UI dependencies
- Safe for partial / delayed rank data
- Deterministic and testable

All aggregations are performed per training step across ranks:
  - WORST  = max across ranks (tail / gating rank)
  - MEDIAN = median across ranks (typical rank)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger



class StepCombinedComputer:
    """
    Rank-agnostic computation of model step summary metrics
    from RemoteDBStore.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        max_points: int = 300,
    ):
        self.store = remote_store
        self.max_points = int(max_points)
        self.logger = get_error_logger("StepCombinedComputer")

    def _completed_step(self, table: str, sampler: str) -> Optional[int]:
        """
        completed_step = min_r latest_step(rank=r, table)
        """
        steps: List[int] = []

        for rank in self.store.ranks():
            db = self.store.get_db(rank, sampler)
            if not db:
                return None
            tbl = db.get_table(table)
            if not tbl:
                return None
            try:
                steps.append(int(tbl[-1]["step"]))
            except Exception:
                return None

        return min(steps) if steps else None

    def _collect_series(
        self,
        *,
        table: str,
        sampler: str,
        value_key: str,
    ) -> Dict[int, List[Tuple[int, float]]]:
        out: Dict[int, List[Tuple[int, float]]] = {}

        for rank in self.store.ranks():
            db = self.store.get_db(rank, sampler)
            if not db:
                continue
            tbl = db.create_or_get_table(table)
            pairs = tail_rows(
                tbl,
                step_key="step",
                value_key=value_key,
                limit=self.max_points,
            )
            if pairs:
                out[int(rank)] = pairs

        return out

    def compute(self) -> Dict[str, Any]:
        """
        Compute all model summary metrics.

        Returns payload compatible with all existing renderers.
        """
        payload: Dict[str, Any] = {}

        specs = [
            (
                "_traceml_internal:dataloader_next",
                "TimeSampler",
                "duration_ms",
                "dataloading_time",
            ),
            (
                "_traceml_internal:step_time",
                "TimeSampler",
                "duration_ms",
                "step_time",
            ),
            (
                "step_memory",
                "StepMemorySampler",
                "peak_allocated_mb",
                "step_gpu_memory",
            ),
        ]

        for table, sampler, value_key, out_key in specs:
            completed = self._completed_step(table, sampler)
            if completed is None:
                continue

            per_rank = self._collect_series(
                table=table,
                sampler=sampler,
                value_key=value_key,
            )

            agg = aggregate_worst_and_median(
                per_rank=per_rank,
                completed_step=completed,
                max_points=self.max_points,
            )

            if agg:
                payload[out_key] = agg

        return payload


def tail_rows(
    table: Any,
    step_key: str,
    value_key: str,
    limit: int,
) -> List[Tuple[int, float]]:
    """
    Read up to `limit` (step, value) pairs from the tail of a table.

    Assumes table is an append-only, iterable structure (deque/list).

    Returns rows in ascending step order.
    """
    if not table:
        return []

    out: List[Tuple[int, float]] = []
    for row in reversed(table):
        try:
            out.append((int(row[step_key]), float(row[value_key])))
        except Exception:
            continue
        if len(out) >= limit:
            break

    out.reverse()
    return out


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute rolling statistics over a 1D series.

    Stats are computed over the last <=100 samples.
    """
    if arr.size == 0:
        return dict(last=0.0, p50=0.0, p95=0.0, avg100=0.0, trend="")

    last = float(arr[-1])
    win = arr[-min(100, arr.size):]

    p50 = float(np.percentile(win, 50))
    p95 = float(np.percentile(win, 95))
    avg100 = float(win.mean())

    trend = ""
    if arr.size >= 200:
        prev = float(arr[-200:-100].mean())
        if prev > 1e-9:
            pct = (avg100 - prev) / prev * 100.0
            trend = f"{pct:+.1f}%" if abs(pct) >= 1.0 else "≈0%"

    return dict(last=last, p50=p50, p95=p95, avg100=avg100, trend=trend)


def aggregate_worst_and_median(
    per_rank: Dict[int, List[Tuple[int, float]]],
    completed_step: int,
    max_points: int,
) -> Dict[str, Any]:
    """
    Aggregate per-rank (step, value) series into per-step WORST and MEDIAN.

    Parameters
    ----------
    per_rank:
        { rank_id -> [(step, value), ...] }
    completed_step:
        Upper bound on steps considered. Ensures all ranks emitted data.
    max_points:
        Number of most recent steps to retain.

    Returns
    -------
    Dict with keys:
      - steps
      - worst  : { y, stats }
      - median : { y, stats }
      - rank_skew_abs
      - rank_skew_pct
      - slowest_rank
    """
    aligned: Dict[int, Dict[int, float]] = defaultdict(dict)

    for rank, pairs in per_rank.items():
        for step, val in pairs:
            if step <= completed_step:
                aligned[step][rank] = val

    if not aligned:
        return {}

    steps = sorted(aligned.keys())[-max_points:]

    worst_vals: List[float] = []
    median_vals: List[float] = []

    slowest_rank: Optional[int] = None
    skew_abs = 0.0
    skew_pct = 0.0

    for s in steps:
        vals = aligned[s]
        if not vals:
            continue

        values = list(vals.values())
        values.sort()

        worst = values[-1]
        median = float(np.median(values))

        worst_vals.append(worst)
        median_vals.append(median)

        if s == completed_step:
            slowest_rank = max(vals, key=lambda r: vals[r])
            skew_abs = worst - values[0]
            skew_pct = (skew_abs / worst) if worst > 0 else 0.0

    worst_arr = np.asarray(worst_vals, dtype=float)
    median_arr = np.asarray(median_vals, dtype=float)

    return dict(
        steps=steps,
        worst=dict(y=worst_arr, stats=compute_stats(worst_arr)),
        median=dict(y=median_arr, stats=compute_stats(median_arr)),
        rank_skew_abs=skew_abs,
        rank_skew_pct=skew_pct,
        slowest_rank=slowest_rank,
    )

