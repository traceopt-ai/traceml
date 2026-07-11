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
from traceml_ai.utils.step_time_window import (
    StepTimeWindow,
    public_step_time_metric_values,
)

MAX_SUMMARY_WINDOW_ROWS = DEFAULT_SUMMARY_WINDOW_ROWS
STEP_TIME_METRIC_NAMES = [
    "total_step_ms",
    "dataloader_ms",
    "input_wait_ms",
    "step_time_ms",
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
    """Per-rank Step Time summary with compatibility and diagnosis fields."""

    steps_analyzed: int
    avg_dataloader_ms: float
    avg_input_wait_ms: float
    avg_step_time_ms: float
    avg_h2d_ms: float
    avg_forward_ms: float
    avg_backward_ms: float
    avg_optimizer_ms: float
    avg_traced_step_ms: float
    avg_compute_ms: float
    avg_compute_ms: float
    avg_residual_ms: float
    avg_total_step_ms: float


@dataclass(frozen=True)
class GlobalRankIdentity:
    """Runtime identity observed for one global rank."""

    global_rank: int
    local_rank: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    local_world_size: Optional[int]
    world_size: Optional[int]


def rank_summary_from_timing(
    timing: Dict[str, float],
    *,
    steps_analyzed: int,
) -> RankStepSummary:
    """Build the public summary view from one canonical rank timing row."""
    public = public_step_time_metric_values(timing)
    return RankStepSummary(
        steps_analyzed=max(0, int(steps_analyzed)),
        avg_dataloader_ms=finite_float(public["dataloader_ms"]),
        avg_input_wait_ms=finite_float(public["input_wait_ms"]),
        avg_step_time_ms=finite_float(public["step_time_ms"]),
        avg_h2d_ms=finite_float(public["h2d_ms"]),
        avg_forward_ms=finite_float(public["forward_ms"]),
        avg_backward_ms=finite_float(public["backward_ms"]),
        avg_optimizer_ms=finite_float(public["optimizer_ms"]),
        avg_traced_step_ms=finite_float(timing.get("step_time")),
        avg_compute_ms=finite_float(public["compute_ms"]),
        avg_residual_ms=finite_float(public["residual_ms"]),
        avg_total_step_ms=finite_float(public["total_step_ms"]),
    )


def rank_summaries_from_window(
    window: StepTimeWindow,
) -> Dict[int, RankStepSummary]:
    """Return final-summary rank rows from one canonical Step Time window."""
    return {
        int(rank): rank_summary_from_timing(
            dict(timing),
            steps_analyzed=int(window.coverage.steps_used),
        )
        for rank, timing in window.per_rank_timing.items()
    }


def compute_residual_avg_ms(s: RankStepSummary) -> float:
    """
    Return canonical average residual for one rank summary.

    residual_ms is averaged from per-step clamped residuals:
    max(0, step_time_ms - h2d_ms - compute_ms). This intentionally differs
    from clamping the already-averaged phase totals.
    """
    return finite_float(s.avg_residual_ms)


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
        "input_wait_ms": {
            int(rank): finite_float(summary.avg_input_wait_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "step_time_ms": {
            int(rank): finite_float(summary.avg_step_time_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "h2d_ms": {
            int(rank): finite_float(summary.avg_h2d_ms)
            for rank, summary in per_global_rank_summary.items()
        },
        "compute_ms": {
            int(rank): finite_float(summary.avg_compute_ms)
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
        "input_wait_ms": finite_float(summary.avg_input_wait_ms),
        "step_time_ms": finite_float(summary.avg_step_time_ms),
        "h2d_ms": finite_float(summary.avg_h2d_ms),
        "compute_ms": finite_float(summary.avg_compute_ms),
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
    analysis_window: StepTimeWindow,
) -> Dict[str, Any]:
    """Build the top-level step-time rollup for the run window."""
    steps = [int(step) for step in analysis_window.steps]
    start_step = steps[0] if steps else None
    end_step = steps[-1] if steps else None
    window = GlobalWindow(
        kind="step_window",
        alignment="common_steps",
        steps_analyzed=int(analysis_window.coverage.steps_used),
        start_step=start_step,
        end_step=end_step,
        completed_step=end_step,
        window_size=int(analysis_window.coverage.expected_steps),
    ).to_json()
    window["diagnosis_clock"] = analysis_window.clock
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
    "RankStepSummary",
    "build_global_rollup",
    "build_overview",
    "closest_rank_to_median",
    "compute_residual_avg_ms",
    "finite_float",
    "rank_summaries_from_window",
    "rank_summary_from_timing",
    "summary_metric_values",
]
