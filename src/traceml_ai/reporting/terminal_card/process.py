# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run-card Process Metrics pane.

Process values are observation-window rank averages.  Each median/worst cell
uses the stored point for that metric; memory byte/percentage pairs are read
from the single rank row selected by their point anchor.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from traceml_ai.reporting.terminal_card.common import (
    as_float,
    as_int,
    as_mapping,
    average,
    diagnosis,
    format_capacity,
    format_memory_value,
    format_percent,
    format_scope,
    group_row,
    group_rows,
    identity,
    metadata,
    point,
    point_value,
    status_spans,
)
from traceml_ai.reporting.terminal_card.layout import (
    RUN_RIGHT_PANE_WIDTH,
    STYLE_DIM,
    CardDoc,
    PaneStages,
    append_table_header,
    append_table_row,
    new_pane_stages,
)
from traceml_ai.reporting.terminal_card.run_evidence import format_run_evidence

_PROCESS_LABEL_WIDTH = 21
_PROCESS_MEDIAN_WIDTH = 18


def _rank_count(process_summary: Mapping[str, Any]) -> int:
    """Return the Process ranks represented by stored telemetry."""
    section_metadata = metadata(process_summary)
    count = as_int(section_metadata.get("global_ranks_used"))
    if count is not None:
        return max(0, count)

    ranks = set()
    anonymous_rows = 0
    for raw_row in group_rows(process_summary).values():
        row_identity = as_mapping(as_mapping(raw_row).get("identity"))
        rank = as_int(row_identity.get("global_rank"))
        if rank is None:
            anonymous_rows += 1
        else:
            ranks.add(rank)
    return len(ranks) + anonymous_rows


def _coverage_text(
    process_summary: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> Optional[str]:
    """Return observed/expected Process-rank coverage for the status line."""
    observed = _rank_count(process_summary)
    expected = as_int(meta.get("world_size"))
    if expected is not None and expected > 0:
        if observed == expected == 1:
            return None
        unit = "rank" if expected == 1 else "ranks"
        return f"{observed}/{expected} {unit}"
    if observed > 0:
        return f"{observed} {'rank' if observed == 1 else 'ranks'} observed"
    return None


def _point_scope(
    process_summary: Mapping[str, Any], idx: Any
) -> Optional[str]:
    """Resolve one stored Process point to its grouped rank/node identity."""
    row_identity = identity(process_summary, idx)
    rank = as_int(row_identity.get("global_rank"))
    if rank is None:
        return None
    node = as_int(row_identity.get("node_rank"))
    return format_scope(rank=rank, node=node)


def _point_metrics(
    process_summary: Mapping[str, Any],
    block: str,
    *,
    anchors: Sequence[str],
) -> Tuple[Mapping[str, Any], Optional[str]]:
    """Resolve one Process point to a coherent grouped-rank row."""
    for metric in anchors:
        stored_point = point(process_summary, block, metric)
        value = as_float(stored_point.get("value"))
        if value is None:
            continue
        idx = stored_point.get("idx")
        row = group_row(process_summary, idx)
        metrics = as_mapping(row.get("metrics"))
        if not metrics:
            metrics = {metric: value}
        return metrics, _point_scope(process_summary, idx)
    return {}, None


def _format_scalar(
    value: Optional[float], *, value_kind: str
) -> Optional[str]:
    """Format a stored Process scalar without changing its meaning."""
    if value_kind == "capacity":
        return format_capacity(value)
    percent = format_percent(value)
    return f"{percent}%" if percent is not None else None


def _append_scalar_metric(
    doc: CardDoc,
    process_summary: Mapping[str, Any],
    *,
    label: str,
    metric: str,
    value_kind: str,
    multi_rank: bool,
) -> None:
    """Append one Process scalar from stored average/point fields."""
    median_point = point(process_summary, "median", metric)
    worst_point = point(process_summary, "worst", metric)
    append_table_row(
        doc,
        label=label,
        average=_format_scalar(
            average(process_summary, metric), value_kind=value_kind
        ),
        median=_format_scalar(
            as_float(median_point.get("value")), value_kind=value_kind
        ),
        worst=_format_scalar(
            as_float(worst_point.get("value")), value_kind=value_kind
        ),
        worst_scope=_point_scope(process_summary, worst_point.get("idx")),
        multi=multi_rank,
        label_width=_PROCESS_LABEL_WIDTH,
        median_width=_PROCESS_MEDIAN_WIDTH,
    )


def _append_memory_metric(
    doc: CardDoc,
    process_summary: Mapping[str, Any],
    *,
    label: str,
    bytes_metric: str,
    percent_metric: str,
    multi_rank: bool,
) -> None:
    """Append a coherent Process memory pair from stored rank averages."""
    average_metrics = as_mapping(
        as_mapping(process_summary.get("global")).get("average")
    )
    median_metrics, _ = _point_metrics(
        process_summary, "median", anchors=(percent_metric, bytes_metric)
    )
    worst_metrics, worst_scope = _point_metrics(
        process_summary, "worst", anchors=(percent_metric, bytes_metric)
    )
    append_table_row(
        doc,
        label=label,
        average=format_memory_value(
            average_metrics,
            bytes_metric=bytes_metric,
            percent_metric=percent_metric,
        ),
        median=format_memory_value(
            median_metrics,
            bytes_metric=bytes_metric,
            percent_metric=percent_metric,
        ),
        worst=format_memory_value(
            worst_metrics,
            bytes_metric=bytes_metric,
            percent_metric=percent_metric,
        ),
        worst_scope=worst_scope,
        multi=multi_rank,
        label_width=_PROCESS_LABEL_WIDTH,
        median_width=_PROCESS_MEDIAN_WIDTH,
    )


def _has_table_data(
    process_summary: Mapping[str, Any], *, multi_rank: bool
) -> bool:
    """Return whether the selected Process table has a measured value."""
    metrics = (
        "cpu_capacity_percent",
        "ram_bytes",
        "ram_percent",
        "gpu_mem_used_bytes",
        "gpu_mem_reserved_bytes",
        "gpu_mem_reserved_percent",
    )
    if not multi_rank:
        return any(
            average(process_summary, metric) is not None for metric in metrics
        )
    return any(
        point_value(process_summary, block, metric) is not None
        for block in ("median", "worst")
        for metric in metrics
    )


def build_run_process_pane(
    process_summary: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> PaneStages:
    """Build staged Process Metrics content for the lower-right Run pane."""
    stages = new_pane_stages(RUN_RIGHT_PANE_WIDTH)
    section_diagnosis = diagnosis(process_summary)
    multi_rank = _rank_count(process_summary) > 1
    stages.heading.wrapped_spans(
        *status_spans(
            "PROCESS METRICS",
            section_diagnosis,
            _coverage_text(process_summary, meta=meta),
            details_style=STYLE_DIM,
        )
    )
    stages.evidence.wrapped(
        format_run_evidence("process", process_summary) or "None",
        label="Evidence: ",
    )
    stages.spacer.blank()
    if str(section_diagnosis.get("kind") or "NO_DATA") == "NO_DATA":
        return stages
    if not _has_table_data(process_summary, multi_rank=multi_rank):
        return stages

    append_table_header(
        stages.table,
        multi=multi_rank,
        median_label="median rank avg",
        worst_label="worst rank avg",
        label_width=_PROCESS_LABEL_WIDTH,
        median_width=_PROCESS_MEDIAN_WIDTH,
    )
    _append_scalar_metric(
        stages.table,
        process_summary,
        label="CPU capacity",
        metric="cpu_capacity_percent",
        value_kind="percent",
        multi_rank=multi_rank,
    )
    _append_memory_metric(
        stages.table,
        process_summary,
        label="RSS used",
        bytes_metric="ram_bytes",
        percent_metric="ram_percent",
        multi_rank=multi_rank,
    )
    _append_scalar_metric(
        stages.table,
        process_summary,
        label="CUDA used",
        metric="gpu_mem_used_bytes",
        value_kind="capacity",
        multi_rank=multi_rank,
    )
    _append_memory_metric(
        stages.table,
        process_summary,
        label="CUDA reserved",
        bytes_metric="gpu_mem_reserved_bytes",
        percent_metric="gpu_mem_reserved_percent",
        multi_rank=multi_rank,
    )
    return stages


__all__ = ["build_run_process_pane"]
