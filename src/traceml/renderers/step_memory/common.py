"""
Shared models and SQLite helpers for step-memory telemetry compute.

This module centralizes:
- SQLite read helpers
- step/rank alignment
- metric aggregation into StepMemoryCombinedResult

"""

import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedResult,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)

_METRIC_TO_COLUMN = {
    "peak_allocated": "peak_alloc_bytes",
    "peak_reserved": "peak_reserved_bytes",
}


class StepMemoryMetricsDB:
    """SQLite helper used by step-memory compute paths."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        """Open a short-lived SQLite connection configured for named rows."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_latest_step_per_rank(
        self, conn: sqlite3.Connection
    ) -> Dict[int, int]:
        """
        Return latest observed step per rank.

        Returns
        -------
        dict[int, int]
            Mapping rank -> max(step), excluding NULL rank/step rows.
        """
        rows = conn.execute(
            """
            SELECT rank, MAX(step) AS max_step
            FROM step_memory_samples
            WHERE rank IS NOT NULL
              AND step IS NOT NULL
            GROUP BY rank
            ORDER BY rank ASC;
            """
        ).fetchall()

        out: Dict[int, int] = {}
        for row in rows:
            rank = row["rank"]
            max_step = row["max_step"]
            if rank is None or max_step is None:
                continue
            out[int(rank)] = int(max_step)
        return out

    def detect_gpu_available(self, conn: sqlite3.Connection) -> Optional[bool]:
        """
        Best-effort check for whether this run reported GPU availability.

        Notes
        -----
        Step-memory telemetry is GPU-specific. When there are no
        `step_memory_samples`, distinguishing "waiting for first memory sample"
        from "this run has no GPU" makes the live UX much clearer.
        """
        queries = (
            "SELECT MAX(gpu_available) FROM process_samples;",
            "SELECT MAX(gpu_available) FROM system_samples;",
        )
        saw_signal = False
        for query in queries:
            try:
                row = conn.execute(query).fetchone()
            except Exception:
                continue
            if not row:
                continue
            value = row[0]
            if value is None:
                continue
            saw_signal = True
            try:
                if bool(int(value)):
                    return True
            except Exception:
                if bool(value):
                    return True
        if saw_signal:
            return False
        return None

    def fetch_rank_step_maps(
        self,
        conn: sqlite3.Connection,
        *,
        metric_key: str,
        start_step: int,
        end_step: int,
        max_unique_steps_per_rank: int,
    ) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Optional[str]]]:
        """
        Build per-rank step->value maps for one metric in [start_step, end_step].

        Dedup rule
        ---------
        If multiple rows exist for (rank, step), keep the latest by highest `id`.

        Returns
        -------
        tuple[dict[int, dict[int, float]], dict[int, Optional[str]]]
            - rank -> {step -> value_bytes}
            - rank -> device string (best-effort latest known)
        """
        metric_col = _METRIC_TO_COLUMN.get(metric_key)
        if metric_col is None:
            return {}, {}

        query = f"""
            SELECT rank, step, {metric_col} AS value, device, id
            FROM step_memory_samples
            WHERE rank IS NOT NULL
              AND step IS NOT NULL
              AND {metric_col} IS NOT NULL
              AND step BETWEEN ? AND ?
            ORDER BY rank ASC, step DESC, id DESC;
        """

        rows = conn.execute(query, (int(start_step), int(end_step))).fetchall()

        rank_to_steps: Dict[int, Dict[int, float]] = {}
        rank_to_device: Dict[int, Optional[str]] = {}
        seen_rank_step: set[Tuple[int, int]] = set()
        unique_count: Dict[int, int] = {}

        max_unique = max(1, int(max_unique_steps_per_rank))

        for row in rows:
            try:
                rank = int(row["rank"])
                step = int(row["step"])
                value = float(row["value"])
            except Exception:
                continue

            if unique_count.get(rank, 0) >= max_unique:
                continue

            key = (rank, step)
            if key in seen_rank_step:
                continue

            seen_rank_step.add(key)
            rank_to_steps.setdefault(rank, {})[step] = value
            unique_count[rank] = unique_count.get(rank, 0) + 1

            if rank not in rank_to_device:
                dev = row["device"]
                rank_to_device[rank] = str(dev) if dev else None

        return rank_to_steps, rank_to_device

    def count_rows_for_completed_window(
        self,
        conn: sqlite3.Connection,
        *,
        start_step: int,
        end_step: int,
    ) -> int:
        """
        Count step-memory rows across the bounded completed-step window.

        This includes rows whose memory values are NULL, which is important for
        distinguishing:
        - "still waiting for first completed step"
        - "completed steps exist, but GPU step-memory telemetry is not
          applicable for this run"
        """
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM step_memory_samples
            WHERE rank IS NOT NULL
              AND step IS NOT NULL
              AND step BETWEEN ? AND ?;
            """,
            (int(start_step), int(end_step)),
        ).fetchone()
        return int(row[0] or 0) if row else 0


def build_step_memory_combined_result(
    conn: sqlite3.Connection,
    *,
    db: StepMemoryMetricsDB,
    window_size: int = 100,
    metric_keys: Sequence[str] = ("peak_allocated", "peak_reserved"),
) -> StepMemoryCombinedResult:
    """
    Compute combined step-memory metrics for CLI/dashboard.

    Semantics
    ---------
    - Align across ranks on completed steps only.
    - Use largest common suffix up to `window_size`.
    - Keep all values in bytes.
    """
    ws = max(1, int(window_size))
    gpu_available = db.detect_gpu_available(conn)
    latest_per_rank = db.fetch_latest_step_per_rank(conn)

    if not latest_per_rank:
        return StepMemoryCombinedResult(
            metrics=[],
            status_message="Waiting for first fully completed step across all ranks…",
        )

    world_size = len(latest_per_rank)
    completed_step = min(latest_per_rank.values())

    # Scan slightly wider than window to tolerate sparse/missing steps.
    scan_span = max(ws * 20, ws + 1)
    start_step = max(0, int(completed_step) - scan_span + 1)
    rows_in_window = db.count_rows_for_completed_window(
        conn,
        start_step=start_step,
        end_step=int(completed_step),
    )

    out: List[StepMemoryCombinedMetric] = []

    for metric_key in metric_keys:
        rank_maps, rank_devices = db.fetch_rank_step_maps(
            conn,
            metric_key=metric_key,
            start_step=start_step,
            end_step=int(completed_step),
            max_unique_steps_per_rank=scan_span,
        )
        if not rank_maps:
            continue

        steps = _common_suffix_steps_fast(
            per_rank_steps=rank_maps,
            completed_step=int(completed_step),
            window_size=ws,
        )
        if not steps:
            continue

        ranks_list = sorted(rank_maps.keys())
        ranks_present = len(ranks_list)
        incomplete = ranks_present < world_size

        k = len(steps)
        r = len(ranks_list)

        values = np.empty((r, k), dtype=np.float64)
        missing = False

        for i, rank in enumerate(ranks_list):
            step_map = rank_maps.get(rank, {})
            for j, step in enumerate(steps):
                v = step_map.get(step)
                if v is None:
                    missing = True
                    break
                values[i, j] = float(v)
            if missing:
                break

        if missing:
            continue

        median_arr = np.median(values, axis=0)
        worst_arr = np.max(values, axis=0)

        peaks = np.max(values, axis=1)
        median_peak = float(np.median(peaks))
        worst_peak = float(np.max(peaks))
        worst_idx = int(np.argmax(peaks))
        worst_rank = int(ranks_list[worst_idx])

        skew_ratio = (worst_peak / median_peak) if median_peak > 0.0 else 0.0
        skew_pct = (
            (worst_peak - median_peak) / median_peak
            if median_peak > 0.0
            else 0.0
        )

        device = _majority_device(rank_devices)

        out.append(
            StepMemoryCombinedMetric(
                metric=str(metric_key),
                device=device,
                series=StepMemoryCombinedSeries(
                    steps=[int(s) for s in steps],
                    median=median_arr.astype(float).tolist(),
                    worst=worst_arr.astype(float).tolist(),
                ),
                summary=StepMemoryCombinedSummary(
                    window_size=ws,
                    steps_used=len(steps),
                    median_peak=float(median_peak),
                    worst_peak=float(worst_peak),
                    worst_rank=worst_rank,
                    skew_ratio=float(skew_ratio),
                    skew_pct=float(skew_pct),
                ),
                coverage=StepMemoryCombinedCoverage(
                    expected_steps=ws,
                    steps_used=len(steps),
                    completed_step=int(completed_step),
                    world_size=int(world_size),
                    ranks_present=int(ranks_present),
                    incomplete=bool(incomplete),
                ),
            )
        )

    return StepMemoryCombinedResult(
        metrics=out,
        status_message=(
            "OK"
            if out
            else (
                "No GPU detected. Step memory uses torch-based GPU memory telemetry."
                if gpu_available is False and rows_in_window > 0
                else "No complete memory metrics available"
            )
        ),
    )


def _common_suffix_steps_fast(
    per_rank_steps: Dict[int, Dict[int, float]],
    completed_step: int,
    window_size: int,
) -> List[int]:
    """
    Return last `window_size` common steps across all ranks, up to completed_step.
    """
    maps = list(per_rank_steps.values())
    if not maps or window_size <= 0 or completed_step < 0:
        return []

    reference = maps[0]
    out_rev: List[int] = []

    step = int(completed_step)
    scan_cap = max(window_size * 20, window_size + 1)
    scanned = 0

    while step >= 0 and len(out_rev) < window_size:
        scanned += 1
        if scanned > scan_cap:
            break

        if step in reference:
            ok = True
            for m in maps[1:]:
                if step not in m:
                    ok = False
                    break
            if ok:
                out_rev.append(step)

        step -= 1

    if not out_rev:
        return []
    out_rev.reverse()
    return out_rev


def _majority_device(
    per_rank_device: Dict[int, Optional[str]]
) -> Optional[str]:
    """Return majority device string from per-rank device map."""
    devices = [d for d in per_rank_device.values() if d]
    if not devices:
        return None
    return max(set(devices), key=devices.count)
