# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time domain objects and pure helpers for the final-report section."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.reporting.schema import BaseGlobal, GlobalWindow
from traceml_ai.reporting.summaries.summary_formatting import safe_float
from traceml_ai.utils.step_time_diagnosis_clock import (
    diagnosis_clock_fields_from_events,
)

MAX_SUMMARY_WINDOW_ROWS = DEFAULT_SUMMARY_WINDOW_ROWS
STEP_TIME_METRIC_NAMES = [
    "total_step_ms",
    "dataloader_ms",
    "h2d_ms",
    "compute_ms",
    "residual_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
]


def finite_float(x: Any) -> float:
    """Convert to float; coerce non-finite values to 0.0."""
    v = safe_float(x)
    return v if np.isfinite(v) else 0.0


def _event_total_ms(by_dev: Any) -> float:
    """Sum ``duration_ms`` across all devices for one event."""
    if not isinstance(by_dev, dict):
        return 0.0

    total = 0.0
    for stats in by_dev.values():
        if not isinstance(stats, dict):
            continue
        total += finite_float(stats.get("duration_ms"))
    return total


def _event_bucket(name: str) -> Optional[str]:
    """Map a raw event name to a step-time bucket."""
    n = str(name).lower()

    if "step_time" in n:
        return "step_time"
    if "dataloader_next" in n:
        return "dataloader"
    if "h2d_time" in n or "host_to_device" in n:
        return "h2d"
    if "forward_time" in n:
        return "forward"
    if "backward_time" in n:
        return "backward"
    if "optimizer_step" in n:
        return "optimizer"

    if "data" in n or "dataloader" in n or "input" in n or "batch" in n:
        return "dataloader"
    if "forward" in n or n == "fwd":
        return "forward"
    if "backward" in n or "bwd" in n:
        return "backward"
    if "optim" in n or "optimizer" in n or n in {"step", "update"}:
        return "optimizer"

    return None


def closest_rank_to_median(rank_to_value: Dict[int, float]) -> Optional[int]:
    """
    Return the rank whose value is closest to the median of all values.

    Tie-breaker order:
    1) closest absolute distance to median
    2) smaller metric value
    3) smaller rank id
    """
    if not rank_to_value:
        return None

    vals = np.asarray(
        [finite_float(v) for v in rank_to_value.values()],
        dtype=np.float64,
    )
    if vals.size == 0:
        return None

    median_val = float(np.median(vals))

    return min(
        rank_to_value.keys(),
        key=lambda r: (
            abs(finite_float(rank_to_value[r]) - median_val),
            finite_float(rank_to_value[r]),
            r,
        ),
    )


@dataclass
class RankStepSummary:
    """Per-rank averaged step-time summary."""

    steps_analyzed: int
    avg_dataloader_ms: float
    avg_h2d_ms: float
    avg_forward_ms: float
    avg_backward_ms: float
    avg_optimizer_ms: float
    avg_step_cpu_ms: float
    avg_traced_step_ms: float
    avg_gpu_compute_ms: float
    avg_total_step_ms: float


@dataclass
class RankStepAnalysis:
    """Per-rank summary plus per-step metrics."""

    summary: RankStepSummary
    per_step_metrics: Dict[int, Dict[str, float]]


@dataclass(frozen=True)
class GlobalRankIdentity:
    """Runtime identity observed for one global rank."""

    global_rank: int
    local_rank: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    local_world_size: Optional[int]
    world_size: Optional[int]


def _row_metrics(events: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Convert one step's event map into canonical timing buckets.

    Returns
    -------
    dict with keys:
      - dataloader
      - forward
      - backward
      - optimizer
      - step_time

    or None if nothing usable was found.
    """
    metrics = {
        "dataloader": 0.0,
        "h2d": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "step_time": 0.0,
    }

    for evt_name, by_dev in events.items():
        bucket = _event_bucket(str(evt_name))
        if bucket is None:
            continue
        metrics[bucket] += _event_total_ms(by_dev)

    metrics.update(diagnosis_clock_fields_from_events(events))

    if (
        metrics["dataloader"] <= 0.0
        and metrics["forward"] <= 0.0
        and metrics["backward"] <= 0.0
        and metrics["optimizer"] <= 0.0
        and metrics["step_time"] <= 0.0
    ):
        return None

    return metrics


def build_rank_summary(
    step_rows: list[Dict[str, Any]],
) -> Optional[RankStepAnalysis]:
    """
    Build a per-rank summary and per-step canonical metrics over provided rows.

    For each step:
        known_step_ms = h2d_ms + forward_ms + backward_ms + optimizer_ms
        residual_ms = traced_step_ms - known_step_ms
        total_step_ms = dataloader_ms + traced_step_ms
    """
    if not step_rows:
        return None

    sum_dl = 0.0
    sum_h2d = 0.0
    sum_fwd = 0.0
    sum_bwd = 0.0
    sum_opt = 0.0
    sum_step_cpu = 0.0
    sum_traced_step = 0.0
    sum_total = 0.0
    n = 0

    per_step_metrics: Dict[int, Dict[str, float]] = {}

    for row in step_rows:
        step_id = row.get("step")
        metrics = _row_metrics(row["events"])
        if metrics is None or step_id is None:
            continue

        dl = finite_float(metrics["dataloader"])
        h2d = finite_float(metrics["h2d"])
        fwd = finite_float(metrics["forward"])
        bwd = finite_float(metrics["backward"])
        opt = finite_float(metrics["optimizer"])
        step_cpu = finite_float(metrics["step_time"])

        compute_ms = fwd + bwd + opt
        known_step_ms = h2d + compute_ms
        # raw step = the direct trace_step wall timer.
        # traced step = max(raw step, known_step_ms) to avoid impossible negative residual
        # when CPU wall timing and CUDA event timing differ slightly.
        traced_step = max(step_cpu, known_step_ms)
        residual_proxy = max(0.0, traced_step - known_step_ms)
        total_step = dl + traced_step

        per_step_metric = {
            "dataloader_fetch": dl,
            "h2d": h2d,
            "forward": fwd,
            "backward": bwd,
            "optimizer_step": opt,
            "step_time": traced_step,
            "residual_proxy": residual_proxy,
        }
        for key, value in metrics.items():
            if key.endswith("_cpu_ms") or key.endswith("_gpu_ms"):
                per_step_metric[key] = finite_float(value)
        per_step_metrics[int(step_id)] = per_step_metric

        sum_dl += dl
        sum_h2d += h2d
        sum_fwd += fwd
        sum_bwd += bwd
        sum_opt += opt
        sum_step_cpu += step_cpu
        sum_traced_step += traced_step
        sum_total += total_step
        n += 1

    if n == 0:
        return None

    summary = RankStepSummary(
        steps_analyzed=n,
        avg_dataloader_ms=sum_dl / n,
        avg_h2d_ms=sum_h2d / n,
        avg_forward_ms=sum_fwd / n,
        avg_backward_ms=sum_bwd / n,
        avg_optimizer_ms=sum_opt / n,
        avg_step_cpu_ms=sum_step_cpu / n,
        avg_traced_step_ms=sum_traced_step / n,
        avg_gpu_compute_ms=(sum_fwd + sum_bwd + sum_opt) / n,
        avg_total_step_ms=sum_total / n,
    )
    return RankStepAnalysis(summary=summary, per_step_metrics=per_step_metrics)


def compute_residual_avg_ms(s: RankStepSummary) -> float:
    """
    Return average residual proxy for one rank summary.

    known_step_ms = h2d_ms + forward_ms + backward_ms + optimizer_ms
    residual_ms = traced_step_ms - known_step_ms
    total_step_ms = dataloader_ms + traced_step_ms
    """
    return max(
        0.0,
        finite_float(s.avg_traced_step_ms)
        - (
            finite_float(s.avg_h2d_ms)
            + finite_float(s.avg_forward_ms)
            + finite_float(s.avg_backward_ms)
            + finite_float(s.avg_optimizer_ms)
        ),
    )


def _rank_metric_values(
    per_global_rank_summary: Dict[int, RankStepSummary],
) -> Dict[str, Dict[int, float]]:
    """Return global-rank values for each Step Time metric."""
    return {
        "total_step_ms": {
            int(rank): finite_float(summary.avg_total_step_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "dataloader_ms": {
            int(rank): finite_float(summary.avg_dataloader_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "h2d_ms": {
            int(rank): finite_float(summary.avg_h2d_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "compute_ms": {
            int(rank): finite_float(summary.avg_gpu_compute_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "residual_ms": {
            int(rank): compute_residual_avg_ms(summary)
            for rank, summary in per_global_rank_summary.items()
        },
        "forward_ms": {
            int(rank): finite_float(summary.avg_forward_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "backward_ms": {
            int(rank): finite_float(summary.avg_backward_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "optimizer_ms": {
            int(rank): finite_float(summary.avg_optimizer_ms)
            for rank, summary in per_global_rank_summary.items()
        },
    }


def summary_metric_values(summary: RankStepSummary) -> Dict[str, float]:
    """Return public row metrics for one global-rank step-time summary."""
    return {
        "total_step_ms": finite_float(summary.avg_total_step_ms),
        "dataloader_ms": finite_float(summary.avg_dataloader_ms),
        "h2d_ms": finite_float(summary.avg_h2d_ms),
        "compute_ms": finite_float(summary.avg_gpu_compute_ms),
        "residual_ms": compute_residual_avg_ms(summary),
        "forward_ms": finite_float(summary.avg_forward_ms),
        "backward_ms": finite_float(summary.avg_backward_ms),
        "optimizer_ms": finite_float(summary.avg_optimizer_ms),
    }


def _average_points(values_by_metric: Dict[str, Dict[int, float]]) -> Dict:
    """Average each Step Time metric across observed global ranks."""
    out: Dict[str, Optional[float]] = {}
    for metric, values in values_by_metric.items():
        vals = list(values.values())
        out[metric] = sum(vals) / len(vals) if vals else None
    return out


def _empty_average(metric_names: Iterable[str]) -> Dict[str, None]:
    """Return a null-valued average block for missing-data sections."""
    return {str(metric): None for metric in metric_names}


def _empty_rank_points(
    metric_names: Iterable[str],
) -> Dict[str, Dict[str, None]]:
    """Return stable null-valued median/worst rank points."""
    return {
        str(metric): {"value": None, "idx": None} for metric in metric_names
    }


def _rank_points(
    values_by_metric: Dict[str, Dict[int, float]],
    *,
    kind: str,
) -> Dict[str, Dict[str, Any]]:
    """Return `{value, global_rank}` points for median or worst values."""
    out: Dict[str, Dict[str, Any]] = {}
    for metric, values in values_by_metric.items():
        if not values:
            out[metric] = {"value": None, "idx": None}
            continue
        if kind == "median":
            rank = closest_rank_to_median(values)
        elif kind == "worst":
            rank = max(values, key=lambda item: (values[item], -int(item)))
        else:
            raise ValueError(f"Unsupported Step Time point kind: {kind}")
        out[metric] = {
            "value": values.get(rank) if rank is not None else None,
            "idx": str(rank) if rank is not None else None,
        }
    return out


def build_global_rollup(
    *,
    per_global_rank_summary: Dict[int, RankStepSummary],
    median_global_rank: Optional[int],
    worst_global_rank: Optional[int],
    analysis_window: Any,
) -> Dict[str, Any]:
    """Build the top-level step-time rollup for the run window."""
    window_json = dict(analysis_window.to_json())
    steps_analyzed = window_json.get("aligned_steps_analyzed", 0)
    end_step = window_json.get("end_step")
    window = GlobalWindow(
        kind="step_window",
        alignment=str(window_json.get("alignment") or "common_steps"),
        steps_analyzed=int(steps_analyzed or 0),
        start_step=window_json.get("start_step"),
        end_step=end_step,
        completed_step=end_step,
        window_size=window_json.get("window_size"),
    ).to_json()
    if not per_global_rank_summary:
        return BaseGlobal(
            index_by="global_rank",
            window=window,
            average=_empty_average(STEP_TIME_METRIC_NAMES),
            median=_empty_rank_points(STEP_TIME_METRIC_NAMES),
            worst=_empty_rank_points(STEP_TIME_METRIC_NAMES),
        ).to_json()

    values_by_metric = _rank_metric_values(per_global_rank_summary)
    return BaseGlobal(
        index_by="global_rank",
        window=window,
        average=_average_points(values_by_metric),
        median=_rank_points(values_by_metric, kind="median"),
        worst=_rank_points(values_by_metric, kind="worst"),
    ).to_json()


def build_overview(
    *,
    per_global_rank_summary: Dict[int, RankStepSummary],
) -> Dict[str, Any]:
    """Build high-level overview fields from global-rank timing summaries."""
    if not per_global_rank_summary:
        return {
            "rank_comparison": "no_data",
            "median_global_rank": None,
            "worst_global_rank": None,
            "median_avg_step_ms": None,
            "worst_avg_step_ms": None,
            "step_time_skew_percent": None,
        }

    avg_total_by_rank = {
        rank: s.avg_total_step_ms
        for rank, s in per_global_rank_summary.items()
    }
    worst_global_rank = max(avg_total_by_rank, key=avg_total_by_rank.get)
    median_global_rank = closest_rank_to_median(avg_total_by_rank)

    worst_avg_step_ms = avg_total_by_rank.get(worst_global_rank)
    median_avg_step_ms = (
        avg_total_by_rank.get(median_global_rank)
        if median_global_rank is not None
        else None
    )

    step_time_skew_percent = None
    if (
        worst_avg_step_ms is not None
        and median_avg_step_ms is not None
        and median_avg_step_ms > 0.0
        and worst_global_rank != median_global_rank
    ):
        step_time_skew_percent = (
            100.0
            * (worst_avg_step_ms - median_avg_step_ms)
            / median_avg_step_ms
        )

    return {
        "rank_comparison": (
            "single_rank"
            if len(per_global_rank_summary) <= 1
            else "distributed"
        ),
        "median_global_rank": median_global_rank,
        "worst_global_rank": worst_global_rank,
        "median_avg_step_ms": median_avg_step_ms,
        "worst_avg_step_ms": worst_avg_step_ms,
        "step_time_skew_percent": step_time_skew_percent,
    }


__all__ = [
    "MAX_SUMMARY_WINDOW_ROWS",
    "STEP_TIME_METRIC_NAMES",
    "GlobalRankIdentity",
    "RankStepAnalysis",
    "RankStepSummary",
    "build_global_rollup",
    "build_overview",
    "build_rank_summary",
    "closest_rank_to_median",
    "compute_residual_avg_ms",
    "finite_float",
    "summary_metric_values",
]
