# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Pure final-summary projection for canonical Step Time analyses.

The projector owns the versioned public timing contract, identities, and card
text. It consumes analyzer-owned values directly; loading, alignment,
statistics, and diagnosis remain in the central Step Time pipeline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Mapping, NamedTuple, Optional

from traceml_ai.reporting.schema import (
    BaseGlobal,
    BaseGroups,
    BaseSectionPayload,
    GlobalWindow,
    GroupRow,
    StepMetadata,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.reporting.summaries.summary_formatting import format_ms
from traceml_ai.reporting.topology import topology_mode_from_identities
from traceml_ai.step_time.model import (
    StepTimeMetric,
    StepTimeRankIdentity,
    StepTimeValues,
)

if TYPE_CHECKING:
    from traceml_ai.step_time.pipeline import StepTimeAnalysis

STEP_TIME_METRIC_NAMES = [
    "input_wait_ms",
    "step_time_ms",
    "traced_step_time_ms",
    "step_time_cpu_ms",
    "step_time_gpu_ms",
    "traced_step_time_cpu_ms",
    "traced_step_time_gpu_ms",
    "dataloader_fetch_cpu_ms",
    "h2d_ms",
    "compute_ms",
    "residual_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
]

_PUBLIC_METRICS = {
    "input_wait_ms": ("input_wait_ms", "input_wait"),
    "step_time_ms": ("step_time_ms", "step_time"),
    "traced_step_time_ms": (
        "traced_step_time_ms",
        "traced_step_time",
    ),
    "step_time_cpu_ms": ("step_time_cpu_ms", "step_time_cpu"),
    "step_time_gpu_ms": ("step_time_gpu_ms", "step_time_gpu"),
    "traced_step_time_cpu_ms": (
        "traced_step_time_cpu_ms",
        "traced_step_time_cpu",
    ),
    "traced_step_time_gpu_ms": (
        "traced_step_time_gpu_ms",
        "traced_step_time_gpu",
    ),
    "dataloader_fetch_cpu_ms": (
        "dataloader_fetch_cpu_ms",
        "dataloader_fetch",
    ),
    "h2d_ms": ("h2d_ms", "h2d"),
    "compute_ms": ("compute_ms", "compute"),
    "residual_ms": ("residual_ms", "residual_proxy"),
    "forward_ms": ("forward_ms", "forward"),
    "backward_ms": ("backward_ms", "backward"),
    "optimizer_ms": ("optimizer_step_ms", "optimizer_step"),
}
"""Stable schema-1.7 fields projected from the canonical timing contract."""


class _MetricProjection(NamedTuple):
    """One public metric aggregated across ranks that measured it."""

    average: Optional[float] = None
    mathematical_median: Optional[float] = None
    representative_value: Optional[float] = None
    representative_rank: Optional[int] = None
    worst_value: Optional[float] = None
    worst_rank: Optional[int] = None


def _optional_float(value: Any) -> Optional[float]:
    """Return one finite float while preserving unavailable values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _public_metric_values(
    values: StepTimeValues,
) -> Dict[str, Optional[float]]:
    """Project one typed rank average to stable public metric names."""
    return {
        metric: _optional_float(getattr(values, field_name))
        for metric, (field_name, _statistic) in _PUBLIC_METRICS.items()
    }


def _project_metric(
    rank_values: Mapping[int, float],
    statistic: Optional[StepTimeMetric],
) -> _MetricProjection:
    """Combine one public mean with analyzer-owned rank statistics."""
    if not rank_values:
        return _MetricProjection()
    if statistic is None:
        raise ValueError("Step Time analysis is missing public metric facts")
    return _MetricProjection(
        average=float(sum(rank_values.values()) / len(rank_values)),
        mathematical_median=float(statistic.median_total),
        representative_value=statistic.representative_total,
        representative_rank=statistic.representative_rank,
        worst_value=float(statistic.worst_total),
        worst_rank=statistic.worst_rank,
    )


def _project_metrics(
    rows: Mapping[int, Mapping[str, Optional[float]]],
    statistics: Mapping[str, StepTimeMetric],
) -> Dict[str, _MetricProjection]:
    """Project every public metric from one canonical statistic set."""
    return {
        metric: _project_metric(
            {
                int(rank): float(value)
                for rank, row in rows.items()
                if (value := row[metric]) is not None
            },
            statistics.get(statistic),
        )
        for metric, (_field, statistic) in _PUBLIC_METRICS.items()
    }


def _global_rollup(
    analysis: StepTimeAnalysis,
    metrics: Mapping[str, _MetricProjection],
) -> Dict[str, Any]:
    """Project stable run-level metric and aligned-window blocks."""
    window = analysis.window
    start_step = window.steps[0] if window.steps else None
    end_step = window.steps[-1] if window.steps else None
    window_json = GlobalWindow(
        kind="step_window",
        alignment="common_steps",
        steps_analyzed=int(window.coverage.steps_used),
        start_step=start_step,
        end_step=end_step,
        completed_step=end_step,
    ).to_json()
    window_json["diagnosis_clock"] = window.clock

    return BaseGlobal(
        index_by="global_rank",
        window=window_json,
        average={name: point.average for name, point in metrics.items()},
        median={
            name: {
                "value": point.representative_value,
                "idx": (
                    str(point.representative_rank)
                    if point.representative_rank is not None
                    else None
                ),
            }
            for name, point in metrics.items()
        },
        worst={
            name: {
                "value": point.worst_value,
                "idx": (
                    str(point.worst_rank)
                    if point.worst_rank is not None
                    else None
                ),
            }
            for name, point in metrics.items()
        },
    ).to_json()


def _global_rank_entry(
    global_rank: int,
    metrics: Mapping[str, Optional[float]],
    identity: Optional[StepTimeRankIdentity],
) -> Dict[str, Any]:
    """Serialize one aligned global-rank row."""
    return GroupRow(
        identity={
            "global_rank": int(global_rank),
            "local_rank": identity.local_rank if identity else None,
            "node_rank": identity.node_rank if identity else None,
            "hostname": identity.hostname if identity else None,
            "local_world_size": (
                identity.local_world_size if identity else None
            ),
            "world_size": identity.world_size if identity else None,
        },
        metrics=metrics,
    ).to_json()


def _global_rank_label(global_rank: Optional[int]) -> str:
    """Format an optional global-rank id for card text."""
    return f"r{int(global_rank)}" if global_rank is not None else "n/a"


def _format_metric_pair(point: _MetricProjection) -> str:
    """Format one mathematical-median/worst value pair."""
    if point.mathematical_median is None or point.worst_value is None:
        return "n/a"
    return f"{point.mathematical_median:.1f}/{point.worst_value:.1f}ms"


def _format_rank_pair(point: _MetricProjection) -> str:
    """Format one representative/worst global-rank pair."""
    return (
        f"{_global_rank_label(point.representative_rank)}/"
        f"{_global_rank_label(point.worst_rank)}"
    )


def _format_card_stats(
    metrics: Mapping[str, _MetricProjection],
    *,
    multi_rank: bool,
) -> str:
    """Render the compact Step Time statistics lines."""
    if multi_rank:
        return (
            "- Stats: median/worst | "
            f"Step Time {_format_metric_pair(metrics['step_time_ms'])} | "
            f"input {_format_metric_pair(metrics['input_wait_ms'])} | "
            f"H2D {_format_metric_pair(metrics['h2d_ms'])} | "
            f"compute {_format_metric_pair(metrics['compute_ms'])}\n"
            "- Residual: median/worst "
            f"{_format_metric_pair(metrics['residual_ms'])}"
        )

    return (
        "- Stats: "
        f"Step Time {format_ms(metrics['step_time_ms'].worst_value)} | "
        f"input {format_ms(metrics['input_wait_ms'].worst_value)} | "
        f"H2D {format_ms(metrics['h2d_ms'].worst_value)} | "
        f"compute {format_ms(metrics['compute_ms'].worst_value)}\n"
        f"- Residual: {format_ms(metrics['residual_ms'].worst_value)}"
    )


def _format_card_ranks(metrics: Mapping[str, _MetricProjection]) -> str:
    """Render representative/worst global-rank lines."""
    return (
        "- Ranks: median/worst | "
        f"Step Time {_format_rank_pair(metrics['step_time_ms'])} | "
        f"input {_format_rank_pair(metrics['input_wait_ms'])} | "
        f"H2D {_format_rank_pair(metrics['h2d_ms'])} | "
        f"compute {_format_rank_pair(metrics['compute_ms'])}\n"
        "- Residual ranks: median/worst "
        f"{_format_rank_pair(metrics['residual_ms'])}"
    )


def _build_card(
    analysis: StepTimeAnalysis,
    metrics: Mapping[str, _MetricProjection],
    *,
    training_steps: int,
    latest_step: Optional[int],
    ranks_seen: int,
    ranks_used: tuple[int, ...],
) -> str:
    """Build the historical compact card from projected canonical facts."""
    diagnosis = analysis.diagnosis
    primary = diagnosis.primary
    primary_issue = diagnosis.issues[0]
    reason = str(
        primary_issue.summary
        or primary.reason
        or "No clear timing bottleneck."
    ).strip()
    lines = [
        (
            f"TraceML Step Timing Summary | steps {training_steps} | "
            f"global ranks {ranks_seen}"
        ),
        "Step Time",
    ]

    if not ranks_used:
        latest_text: Any = latest_step if latest_step is not None else "n/a"
        lines.extend(
            [
                f"- Diagnosis: {primary.status}",
                f"- Scope: latest step {latest_text}",
                "- Stats: n/a",
                f"- Why: {reason}",
            ]
        )
        return "\n".join(lines)

    lines.append(f"- Diagnosis: {primary.status}")
    if len(ranks_used) == 1:
        lines.append(
            f"- Scope: last {analysis.window.coverage.steps_used} "
            f"aligned steps on global rank r{ranks_used[0]}"
        )
        lines.append(_format_card_stats(metrics, multi_rank=False))
    else:
        lines.append(
            "- Scope: compared over "
            f"last {analysis.window.coverage.steps_used} aligned steps "
            f"across {analysis.window.coverage.ranks_present} global ranks"
        )
        lines.append(_format_card_stats(metrics, multi_rank=True))
        lines.append(_format_card_ranks(metrics))
    lines.append(f"- Why: {reason}")
    return "\n".join(lines)


def project_step_time_summary(analysis: StepTimeAnalysis) -> Dict[str, Any]:
    """Project one canonical analysis to the stable final-summary schema."""
    snapshot = analysis.snapshot
    window = analysis.window
    identities = snapshot.identities
    latest_step = snapshot.cursor.latest_step
    training_steps = latest_step + 1 if latest_step is not None else 0

    rows = {
        facts.global_rank: _public_metric_values(facts.average)
        for facts in window.rank_facts
    }
    ranks_used = tuple(sorted(rows))
    seen = tuple(sorted(set(identities) | set(ranks_used)))
    metrics = _project_metrics(
        rows,
        {metric.metric: metric for metric in window.metrics},
    )
    diagnosis_json, issues_json = diagnostic_result_to_json(analysis.diagnosis)
    card = _build_card(
        analysis,
        metrics,
        training_steps=training_steps,
        latest_step=latest_step,
        ranks_seen=len(seen),
        ranks_used=ranks_used,
    )

    metadata = StepMetadata(
        mode=topology_mode_from_identities(
            (identities.get(rank) for rank in ranks_used),
            has_data=bool(ranks_used),
        ),
        global_ranks_seen=len(seen),
        global_ranks_used=len(ranks_used),
        training_total_steps=training_steps,
        training_latest_step=latest_step,
        samples=len(snapshot.rows),
        section_metric_names=STEP_TIME_METRIC_NAMES,
    )
    rank_json = {
        str(rank): _global_rank_entry(
            rank,
            rows[rank],
            identities.get(rank),
        )
        for rank in ranks_used
    }
    return BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_json,
        issues=issues_json,
        global_summary=_global_rollup(analysis, metrics),
        groups=BaseGroups(by="global_rank", rows=rank_json).to_json(),
        units={"time": "ms"},
        card=card,
    ).to_json()


__all__ = [
    "STEP_TIME_METRIC_NAMES",
    "project_step_time_summary",
]
