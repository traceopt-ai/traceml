# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process numbers mean.

This layer turns rows into the facts the card states. It owns the
metric decisions and nothing else: the SQL is in ``repository.py`` and the
drawing is in ``process_section.py``.

Semantics, unchanged from 0.3.7:
- seq-aligned across ranks, one entry per globally committed seq
- CPU and RAM are the max across ranks for that step
- the GPU entry comes from the rank with least headroom
- imbalance is the used-bytes spread across reporting ranks
- the card describes the last ``DASHBOARD_WINDOW`` committed steps

The window bound and the percentiles moved here from the section, where
they used to run on every tick inside the view. They are statements about
the metric, so they belong on this side of the boundary.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple

from .dashboard_models import (
    ChartSeries,
    ChartTrace,
    GpuSnapshot,
    MetricRollup,
    ProcessDashboardPayload,
    ProcessHistoryEntry,
)
from .repository import ProcessRepository

# How many committed steps the card describes. Kept at the value the
# section applied to its own history slice on 0.3.7.
DASHBOARD_WINDOW = 100


def percentile(values: Sequence[Optional[float]], p: float) -> float:
    """Linear-interpolated percentile over the values that exist.

    Missing samples are dropped rather than read as zero: a rank that did
    not report is not a rank reporting nothing.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return 0.0
    k = (len(clean) - 1) * p / 100.0
    low = int(k)
    high = min(low + 1, len(clean) - 1)
    if low == high:
        return float(clean[low])
    return float(clean[low] * (high - k) + clean[high] * (k - low))


class ProcessDashboardComputer:
    """Advance the Process history and describe it.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    dashboard_max_rows:
        Maximum retained rolling rows for UI history.
    stale_ttl_s:
        How long a cached payload may stand in for a failed read. This is
        a fallback for a broken READ, not a statement about whether the
        telemetry itself is current.
    """

    def __init__(
        self,
        db_path: str,
        dashboard_max_rows: int = 200,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db = ProcessRepository(db_path=db_path)
        self._dashboard_rollup: Deque[ProcessHistoryEntry] = deque(
            maxlen=max(1, int(dashboard_max_rows))
        )
        self._last_completed_seq: int = -1

        self._last_ok: Optional[ProcessDashboardPayload] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> ProcessDashboardPayload:
        """Advance the history and return one payload for the card."""
        try:
            with self._db.connect() as conn:
                self._advance(conn)
                out = self._build_payload()
        except Exception:
            return self._return_stale()

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    # --- advancing -------------------------------------------------------
    def _advance(self, conn: Any) -> None:
        committed_upto = self._db.fetch_committed_seq(conn)
        if (
            committed_upto is None
            or committed_upto <= self._last_completed_seq
        ):
            return

        start_seq = self._last_completed_seq + 1
        end_seq = int(committed_upto)

        for row in self._db.fetch_seq_range_aggregates(
            conn, start_seq=start_seq, end_seq=end_seq
        ):
            self._dashboard_rollup.append(_entry_from_row(row))

        self._last_completed_seq = end_seq

    # --- describing ------------------------------------------------------
    def _build_payload(self) -> ProcessDashboardPayload:
        history = tuple(self._dashboard_rollup)
        if not history:
            return ProcessDashboardPayload()

        window = history[-DASHBOARD_WINDOW:]
        latest = window[-1]

        cpu_values = [e.cpu_percent_max for e in window]
        ram_values = [e.ram_used_bytes_max for e in window]
        gpu_values = [e.gpu.used_bytes for e in window if e.gpu is not None]

        return ProcessDashboardPayload(
            history=window,
            window_len=len(window),
            cpu=MetricRollup(
                now=cpu_values[-1],
                p50=percentile(cpu_values, 50),
                p95=percentile(cpu_values, 95),
            ),
            ram=MetricRollup(
                now=ram_values[-1],
                p95=percentile(ram_values, 95),
                total=latest.ram_total_bytes,
            ),
            gpu=(
                MetricRollup(
                    now=gpu_values[-1] if gpu_values else 0.0,
                    p95=percentile(gpu_values, 95) if gpu_values else 0.0,
                    total=(
                        latest.gpu.total_bytes
                        if latest.gpu is not None
                        else None
                    ),
                )
                if latest.gpu is not None
                else None
            ),
            gpu_used_imbalance_bytes=(
                latest.gpu.used_imbalance_bytes
                if latest.gpu is not None
                else None
            ),
            chart=_build_chart(window),
        )

    # --- degraded reads --------------------------------------------------
    def _return_stale(self) -> ProcessDashboardPayload:
        """Reuse the last good payload while a read keeps failing.

        This is about the READ, not the run: it says the database could
        not be queried just now, never that the ranks are still healthy.
        """
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return self._last_ok
        return ProcessDashboardPayload()


def _entry_from_row(row: Any) -> ProcessHistoryEntry:
    """One aggregated row, typed.

    The GPU block is built only when the query found a candidate rank, so
    a CPU-only step carries ``None`` rather than a set of absent keys.
    """
    gpu: Optional[GpuSnapshot] = None
    if row["gpu_used"] is not None:
        gpu = GpuSnapshot(
            used_bytes=float(row["gpu_used"] or 0.0),
            total_bytes=float(row["gpu_total"] or 0.0),
            headroom_bytes=float(row["gpu_headroom"] or 0.0),
            rank=(
                int(row["gpu_rank"]) if row["gpu_rank"] is not None else None
            ),
            used_imbalance_bytes=(
                float(row["gpu_used_imbalance"])
                if row["gpu_used_imbalance"] is not None
                else 0.0
            ),
        )

    return ProcessHistoryEntry(
        seq=int(row["seq"]),
        ts=(
            float(row["sample_ts_s"])
            if row["sample_ts_s"] is not None
            else None
        ),
        cpu_percent_max=float(row["cpu_max"] or 0.0),
        ram_used_bytes_max=float(row["ram_used_max"] or 0.0),
        ram_total_bytes=float(row["ram_total"] or 0.0),
        gpu=gpu,
    )


def _build_chart(window: Sequence[ProcessHistoryEntry]) -> ChartSeries:
    """Both traces as a share of their own denominator.

    A share of capacity is what the metric means, so it is computed here.
    The axis, the colours and the tick labels are the card's business.
    """
    timestamps: Tuple[Optional[float], ...] = tuple(e.ts for e in window)

    ram_total = max(float(window[-1].ram_total_bytes or 0.0), 1.0)
    ram = ChartTrace(
        label="RAM",
        timestamps=timestamps,
        values=tuple(
            (e.ram_used_bytes_max / ram_total) * 100.0 for e in window
        ),
    )

    gpu_trace: Optional[ChartTrace] = None
    latest_gpu = window[-1].gpu
    if latest_gpu is not None and latest_gpu.total_bytes:
        gpu_total = max(float(latest_gpu.total_bytes), 1.0)
        stamps: List[Optional[float]] = []
        values: List[float] = []
        for entry in window:
            if entry.gpu is None:
                continue
            stamps.append(entry.ts)
            values.append((entry.gpu.used_bytes / gpu_total) * 100.0)
        gpu_trace = ChartTrace(
            label="GPU mem",
            timestamps=tuple(stamps),
            values=tuple(values),
        )

    return ChartSeries(ram_percent=ram, gpu_percent=gpu_trace)


__all__ = [
    "DASHBOARD_WINDOW",
    "ProcessDashboardComputer",
    "percentile",
]
