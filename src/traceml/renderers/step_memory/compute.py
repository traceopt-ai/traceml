"""
Step Memory Combined Computation (DDP-aware, rank-agnostic)

Compute layer for **step-level peak GPU memory** across DDP ranks.

What it computes
----------------
For each metric:
  - "peak_allocated"  (bytes)
  - "peak_reserved"   (bytes)

it produces a **step-aligned** time series and a compact window summary.

Algorithm (per metric)
----------------------
1) Read recent step memory samples from each rank’s `step_memory` table
   (wire rows with keys: step, peak_alloc, peak_resv, device).
2) Build a per-rank map:
       { step_index -> value_bytes }
   using up to `window_size` most recent unique steps.
3) Define:
       completed_step = min(max_step_per_rank)
   to ensure we only use steps that are complete on all ranks.
4) Align steps across ranks by taking the **largest suffix** of step indices
   (up to `window_size`) that are present on every rank and ≤ completed_step.
5) For each aligned step:
   - median across ranks
   - worst (max) across ranks
6) Compute window summary:
   - per-rank peak over the aligned window
   - median_peak (across ranks)
   - worst_peak (across ranks)
   - worst_rank (rank producing worst_peak)
   - skew_ratio / skew_pct

Output
------
Returns `StepMemoryCombinedResult` containing:
- one `StepMemoryCombinedMetric` per metric_key
- each metric has:
    series.steps, series.median, series.worst
    summary (window stats)
    coverage (ranks present, steps used, completed_step, etc.)

Units
-----
All values are in **bytes**. The renderer/UI may convert to GB for display.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger

from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedResult,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)


class StepMemoryCombinedComputer:
    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
        table_name: str = "step_memory",
        sampler_name: str = "StepMemorySampler",
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self.store = remote_store
        self.window_size = int(window_size)
        self.table_name = table_name
        self.sampler_name = sampler_name
        self.logger = get_error_logger("StepMemoryCombinedComputer")

        self._metrics: Tuple[str, str] = ("peak_allocated", "peak_reserved")

        # last-good cache
        self._last_ok: Optional[StepMemoryCombinedResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = float(stale_ttl_s) if stale_ttl_s is not None else None

    # ----------------------------
    # Public API (cached)
    # ----------------------------
    def compute(self) -> StepMemoryCombinedResult:
        try:
            res = self._compute_impl()
        except Exception as e:
            self.logger.exception("StepMemoryCombined compute failed")
            return self._return_stale_or_empty(f"STALE (exception: {type(e).__name__})")

        if not res.metrics:
            return self._return_stale_or_empty("STALE (no metrics this tick)")

        self._last_ok = res
        self._last_ok_ts = time.time()
        return res

    def _return_stale_or_empty(self, msg: str) -> StepMemoryCombinedResult:
        now = time.time()
        if self._last_ok is not None:
            if self._stale_ttl_s is None or (now - self._last_ok_ts) <= self._stale_ttl_s:
                return StepMemoryCombinedResult(metrics=self._last_ok.metrics, status_message=msg)
        return StepMemoryCombinedResult(metrics=[], status_message="No complete memory metrics available")

    # ----------------------------
    # Core compute
    # ----------------------------
    def _compute_impl(self) -> StepMemoryCombinedResult:
        ranks = list(self.store.ranks())
        world_size = len(ranks)
        if world_size == 0:
            return StepMemoryCombinedResult(metrics=[], status_message="No ranks available")

        out: List[StepMemoryCombinedMetric] = []
        for metric_key in self._metrics:
            m = self._compute_one(metric_key=metric_key, ranks=ranks, world_size=world_size)
            if m is not None:
                out.append(m)

        status = "OK" if out else "No complete memory metrics available"
        return StepMemoryCombinedResult(metrics=out, status_message=status)

    def _compute_one(
        self,
        *,
        metric_key: str,
        ranks: List[int],
        world_size: int,
    ) -> Optional[StepMemoryCombinedMetric]:
        ws = self.window_size
        store = self.store
        sampler = self.sampler_name
        table_name = self.table_name

        per_rank_steps: Dict[int, Dict[int, float]] = {}
        per_rank_completed: List[int] = []
        per_rank_device: Dict[int, Optional[str]] = {}

        # Build per-rank step maps (latest ws unique steps)
        for rank in ranks:
            db = store.get_db(rank, sampler)
            if not db:
                continue
            tbl = db.get_table(table_name)
            if not tbl:
                continue

            step_map: Dict[int, float] = {}
            device: Optional[str] = None

            # IMPORTANT: deque indexing is slow; use reversed(tbl) only.
            for row in reversed(tbl):
                step, alloc_b, resv_b, dev = _extract_wire_row(row)
                if step is None:
                    continue
                if device is None and dev:
                    device = dev

                if metric_key == "peak_allocated":
                    if alloc_b is None:
                        continue
                    step_map[step] = float(alloc_b)
                else:  # "peak_reserved"
                    if resv_b is None:
                        continue
                    step_map[step] = float(resv_b)

                if len(step_map) >= ws:
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

        # last ws steps common across ranks (<= completed_step)
        steps = _common_suffix_steps_fast(per_rank_steps, completed_step, ws)
        if not steps:
            return None

        ranks_list = list(per_rank_steps.keys())
        R = len(ranks_list)
        K = len(steps)

        # Build rank x step matrix once -> vectorized stats
        V = np.empty((R, K), dtype=np.float64)
        for i, r in enumerate(ranks_list):
            sm = per_rank_steps[r]
            for j, st in enumerate(steps):
                V[i, j] = sm[st]  # membership guaranteed by suffix selection

        median_arr = np.median(V, axis=0)
        worst_arr = np.max(V, axis=0)

        # Summary: per-rank peak over window
        peaks = np.max(V, axis=1)
        median_peak = float(np.median(peaks))
        worst_peak = float(np.max(peaks))
        worst_rank = int(ranks_list[int(np.argmax(peaks))])

        skew_ratio = worst_peak / median_peak if median_peak > 0 else 0.0
        skew_pct = ((worst_peak - median_peak) / median_peak) if median_peak > 0 else 0.0

        device = _majority_device(per_rank_device)

        return StepMemoryCombinedMetric(
            metric=metric_key,
            device=device,
            series=StepMemoryCombinedSeries(
                steps=steps,
                median=median_arr.astype(float).tolist(),
                worst=worst_arr.astype(float).tolist(),
            ),
            summary=StepMemoryCombinedSummary(
                window_size=ws,
                steps_used=len(steps),
                median_peak=median_peak,
                worst_peak=worst_peak,
                worst_rank=worst_rank,
                skew_ratio=float(skew_ratio),
                skew_pct=float(skew_pct),
            ),
            coverage=StepMemoryCombinedCoverage(
                expected_steps=ws,
                steps_used=len(steps),
                completed_step=completed_step,
                world_size=world_size,
                ranks_present=ranks_present,
                incomplete=incomplete,
            ),
        )


# ----------------------------
# Wire extraction (NO from_wire)
# ----------------------------
def _extract_wire_row(row: Any) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    """
    Extract from StepMemorySample.to_wire() dict:
      step, peak_alloc, peak_resv, device

    Returns (step, alloc_bytes, resv_bytes, device).
    Never raises.
    """
    try:
        if isinstance(row, dict):
            st = row.get("step", None)
            if st is None:
                return None, None, None, None
            step = int(st)
            alloc = row.get("peak_alloc", None)
            resv = row.get("peak_resv", None)
            dev = row.get("device", None)
            alloc_f = float(alloc) if alloc is not None else None
            resv_f = float(resv) if resv is not None else None
            dev_s = str(dev) if dev else None
            return step, alloc_f, resv_f, dev_s

        # If somehow an object slipped in (rare), try attribute names from dataclass
        st = getattr(row, "step", None)
        if st is None:
            return None, None, None, None
        step = int(st)
        alloc = getattr(row, "peak_allocated", None)
        resv = getattr(row, "peak_reserved", None)
        dev = getattr(row, "device", None)
        alloc_f = float(alloc) if alloc is not None else None
        resv_f = float(resv) if resv is not None else None
        dev_s = str(dev) if dev else None
        return step, alloc_f, resv_f, dev_s
    except Exception:
        return None, None, None, None


def _common_suffix_steps_fast(
    per_rank_steps: Dict[int, Dict[int, float]],
    completed_step: int,
    window_size: int,
) -> List[int]:
    """
    Equivalent to:
      intersect steps across ranks (<= completed_step),
      then take sorted(common)[-window_size:].

    Faster: walk backward from completed_step and collect the top-K common steps.

    Stability guard:
      - cap backward scan to avoid pathological sparse-step hangs
        (still returns best-effort common suffix if found).
    """
    maps = list(per_rank_steps.values())
    if not maps or window_size <= 0 or completed_step < 0:
        return []

    ref = maps[0]
    out_rev: List[int] = []

    st = completed_step
    scan_cap = max(window_size * 20, window_size + 1)  # conservative cap
    scanned = 0

    while st >= 0 and len(out_rev) < window_size:
        scanned += 1
        if scanned > scan_cap:
            break

        if st in ref:
            ok = True
            for m in maps[1:]:
                if st not in m:
                    ok = False
                    break
            if ok:
                out_rev.append(st)

        st -= 1

    if not out_rev:
        return []
    out_rev.reverse()
    return out_rev


def _majority_device(per_rank_device: Dict[int, Optional[str]]) -> Optional[str]:
    devices = [d for d in per_rank_device.values() if d]
    if not devices:
        return None
    return max(set(devices), key=devices.count)