# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-memory data shaping for the final-report section."""

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.diagnostics.step_memory import StepMemoryDiagnosis
from traceml.renderers.step_memory.common import StepMemoryMetricsDB
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.reporting.schema import BaseGlobal, GlobalWindow

MAX_SUMMARY_WINDOW_ROWS = 10_000

STEP_MEMORY_METRIC_NAMES = [
    "peak_allocated_bytes",
    "peak_reserved_bytes",
]


@dataclass(frozen=True)
class StepMemoryGlobalRankIdentity:
    """Runtime identity observed for one step-memory global rank."""

    global_rank: int
    local_rank: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    local_world_size: Optional[int]
    world_size: Optional[int]


def metric_sort_key(metric: StepMemoryCombinedMetric) -> int:
    """Stable metric ordering: allocated first, reserved second."""
    if metric.metric == "peak_allocated":
        return 0
    if metric.metric == "peak_reserved":
        return 1
    return 99


def metric_label(metric_name: str) -> str:
    """Human-friendly label for one step-memory metric key."""
    if metric_name == "peak_allocated":
        return "peak allocated"
    if metric_name == "peak_reserved":
        return "peak reserved"
    return metric_name.replace("_", " ")


def _metric_output_name(metric_name: str) -> str:
    """Return the public metric name used in summary JSON."""
    return f"{metric_name}_bytes"


def no_gpu_diagnosis_json() -> Dict[str, Any]:
    """
    Stable summary diagnosis block for CPU-only / no-GPU runs.
    """
    return {
        "kind": "NO_GPU",
        "status": "NO GPU",
        "severity": "info",
        "metric": None,
        "steps_used": 0,
        "worst_global_rank": None,
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Treat step memory as not applicable for this run.",
        "note": None,
        "confidence": 1.0,
    }


def no_gpu_diagnosis_presented() -> Dict[str, Any]:
    """
    Stable end-of-run presentation block for CPU-only / no-GPU runs.
    """
    return {
        "status": "NO GPU",
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Step memory is not applicable for this run.",
        "note": None,
    }


def load_latest_step_observed(conn: sqlite3.Connection) -> Optional[int]:
    """Return the latest observed memory step across all ranks."""
    row = conn.execute("SELECT MAX(step) FROM step_memory_samples;").fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def load_gpu_total_bytes(conn: sqlite3.Connection) -> Optional[float]:
    """
    Best-effort device total memory from process telemetry.

    This is optional but useful for HIGH_PRESSURE diagnosis.
    """
    try:
        row = conn.execute(
            "SELECT MAX(gpu_mem_total_bytes) FROM process_samples;"
        ).fetchone()
    except Exception:
        return None

    if not row or row[0] is None:
        return None
    try:
        value = float(row[0])
    except Exception:
        return None
    return value if value > 0.0 else None


def primary_metric(
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Optional[StepMemoryCombinedMetric]:
    """
    Pick the primary metric to surface in the printed summary.

    Preference order:
    1. diagnosis.metric, if available
    2. peak_reserved
    3. peak_allocated
    4. first metric
    """
    if not metrics:
        return None

    by_name = {m.metric: m for m in metrics}

    if diagnosis is not None and diagnosis.metric in by_name:
        return by_name[diagnosis.metric]
    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    return metrics[0]


def _load_global_rank_identities(
    conn: sqlite3.Connection,
    global_ranks: list[int],
) -> Dict[int, StepMemoryGlobalRankIdentity]:
    """Load latest runtime identity metadata for step-memory global ranks."""
    identities: Dict[int, StepMemoryGlobalRankIdentity] = {}
    for global_rank in global_ranks:
        row = conn.execute(
            """
            SELECT local_rank, node_rank, hostname, local_world_size,
                   world_size
            FROM step_memory_samples
            WHERE global_rank = ?
            ORDER BY sample_ts_s DESC, id DESC
            LIMIT 1;
            """,
            (int(global_rank),),
        ).fetchone()
        if row is None:
            continue
        identities[int(global_rank)] = StepMemoryGlobalRankIdentity(
            global_rank=int(global_rank),
            local_rank=int(row[0]) if row[0] is not None else None,
            node_rank=int(row[1]) if row[1] is not None else None,
            hostname=str(row[2]) if row[2] is not None else None,
            local_world_size=int(row[3]) if row[3] is not None else None,
            world_size=int(row[4]) if row[4] is not None else None,
        )
    return identities


def _identity_to_json(
    *,
    global_rank: int,
    identity: Optional[StepMemoryGlobalRankIdentity],
) -> Dict[str, Any]:
    """Serialize the runtime identity block for one memory rank."""
    return {
        "global_rank": int(global_rank),
        "local_rank": identity.local_rank if identity else None,
        "node_rank": identity.node_rank if identity else None,
        "hostname": identity.hostname if identity else None,
        "local_world_size": identity.local_world_size if identity else None,
        "world_size": identity.world_size if identity else None,
    }


def load_per_global_rank_summary(
    conn: sqlite3.Connection,
    *,
    db: StepMemoryMetricsDB,
    metrics: list[StepMemoryCombinedMetric],
    window_size: int,
) -> Dict[str, Any]:
    """
    Build per-global-rank summaries for the analyzed aligned tail window.

    Combined worst/median summaries are excellent for end-of-run diagnosis, but
    one often need to answer:
    - which global rank is gating or drifting?
    - is one global rank materially less stable than others?
    - do allocated and reserved tell the same story?

    This function reuses each combined metric's aligned step ids, so
    `global.window` and per-rank metric averages describe the same completed
    steps by construction.
    """
    latest_per_global_rank = db.fetch_latest_step_per_global_rank(conn)
    if not latest_per_global_rank:
        return {}

    scan_span = max(int(window_size) * 20, int(window_size) + 1)
    identities = _load_global_rank_identities(
        conn,
        sorted(latest_per_global_rank.keys()),
    )

    per_global_rank: Dict[str, Any] = {}

    for metric in metrics:
        common_steps = [int(step) for step in metric.series.steps]
        if not common_steps:
            continue

        rank_maps, _ = db.fetch_global_rank_step_maps(
            conn,
            metric_key=metric.metric,
            start_step=min(common_steps),
            end_step=max(common_steps),
            max_unique_steps_per_rank=scan_span,
        )

        if not rank_maps:
            continue

        for rank in sorted(rank_maps.keys()):
            step_map = rank_maps.get(rank, {})
            values = [
                float(step_map[s]) for s in common_steps if s in step_map
            ]
            if len(values) != len(common_steps) or not values:
                continue

            rank_key = str(rank)
            identity = identities.get(int(rank))

            entry = per_global_rank.setdefault(
                rank_key,
                {
                    "identity": _identity_to_json(
                        global_rank=int(rank),
                        identity=identity,
                    ),
                    "diagnosis": None,
                    "issues": [],
                    "metrics": {},
                },
            )

            entry["metrics"][_metric_output_name(metric.metric)] = (
                sum(values) / len(values) if values else None
            )

    return dict(sorted(per_global_rank.items(), key=lambda item: int(item[0])))


def empty_global_rollup() -> Dict[str, Any]:
    """
    Return a stable empty global rollup for missing-data cases.
    """
    average = {metric: None for metric in STEP_MEMORY_METRIC_NAMES}
    rank_points = _empty_rank_points()
    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=0,
        ).to_json(),
        average=average,
        median=rank_points,
        worst=rank_points,
    ).to_json()


def _row_metric_values(
    per_global_rank: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Return finite metric values keyed by metric name and row id."""
    values: Dict[str, Dict[str, float]] = {
        metric_name: {} for metric_name in STEP_MEMORY_METRIC_NAMES
    }
    for rank_key, entry in per_global_rank.items():
        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        for metric_name in STEP_MEMORY_METRIC_NAMES:
            value = metrics.get(metric_name)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except Exception:
                continue
            if math.isfinite(metric_value):
                values[metric_name][str(rank_key)] = metric_value
    return values


def _average_memory_by_metric(per_global_rank: Dict[str, Any]) -> Dict:
    """Average public memory metrics across per-rank memory rows."""
    values = _row_metric_values(per_global_rank)
    return {
        metric_name: (
            sum(metric_values.values()) / len(metric_values)
            if metric_values
            else None
        )
        for metric_name, metric_values in sorted(values.items())
    }


def _rank_sort_value(rank_key: str) -> int:
    """Sort numeric rank keys predictably while tolerating unknown strings."""
    try:
        return int(rank_key)
    except Exception:
        return 0


def _closest_rank_to_median(values: Dict[str, float]) -> Optional[str]:
    """Return the row id whose value is closest to the metric median."""
    if not values:
        return None

    ordered_values = sorted(values.values())
    mid = len(ordered_values) // 2
    if len(ordered_values) % 2:
        median_value = ordered_values[mid]
    else:
        median_value = (ordered_values[mid - 1] + ordered_values[mid]) / 2.0

    return min(
        values,
        key=lambda rank_key: (
            abs(values[rank_key] - median_value),
            values[rank_key],
            _rank_sort_value(rank_key),
        ),
    )


def _rank_points_from_rows(
    per_global_rank: Dict[str, Any],
    *,
    kind: str,
) -> Dict[str, Dict[str, Any]]:
    """Build `{value, idx}` points from grouped step-memory rows."""
    values_by_metric = _row_metric_values(per_global_rank)
    points: Dict[str, Dict[str, Any]] = {}
    for metric_name, values in sorted(values_by_metric.items()):
        if not values:
            points[metric_name] = {"value": None, "idx": None}
            continue
        if kind == "median":
            rank_key = _closest_rank_to_median(values)
        elif kind == "worst":
            rank_key = max(
                values,
                key=lambda item: (values[item], -_rank_sort_value(item)),
            )
        else:
            raise ValueError(f"Unsupported Step Memory point kind: {kind}")

        points[metric_name] = {
            "value": values.get(rank_key) if rank_key is not None else None,
            "idx": rank_key,
        }
    return points


def _empty_rank_points() -> Dict[str, Dict[str, Any]]:
    """Return null rank points for every public step-memory metric."""
    return {
        metric_name: {"value": None, "idx": None}
        for metric_name in STEP_MEMORY_METRIC_NAMES
    }


def build_global_rollup(
    *,
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
    per_global_rank: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the global memory rollup for the analyzed window."""
    if not metrics:
        return empty_global_rollup()

    primary = metrics[0]
    median = _rank_points_from_rows(per_global_rank, kind="median")
    worst = _rank_points_from_rows(per_global_rank, kind="worst")

    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=primary.summary.steps_used,
            end_step=primary.coverage.completed_step,
            completed_step=primary.coverage.completed_step,
            window_size=primary.summary.window_size,
        ).to_json(),
        average=_average_memory_by_metric(per_global_rank),
        median=median or _empty_rank_points(),
        worst=worst or _empty_rank_points(),
    ).to_json()


def topology_mode(
    *,
    global_ranks_used: int,
    per_global_rank: Dict[str, Any],
) -> str:
    """Return run topology from observed Step Memory runtime identity."""
    if global_ranks_used <= 0:
        return "no_data"

    identities = [
        entry.get("identity", {})
        for entry in per_global_rank.values()
        if isinstance(entry, dict)
    ]
    node_ranks = {
        identity.get("node_rank")
        for identity in identities
        if identity.get("node_rank") is not None
    }
    if len(node_ranks) > 1:
        return "multi_node"

    for identity in identities:
        world_size = identity.get("world_size")
        local_world_size = identity.get("local_world_size")
        if (
            world_size is not None
            and local_world_size is not None
            and int(world_size) > int(local_world_size)
        ):
            return "multi_node"

    return "single_node"
