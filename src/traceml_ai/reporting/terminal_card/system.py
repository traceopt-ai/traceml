# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared Run/Watch System Metrics pane.

System values are observation-window averages.  In the multi-node table each
metric independently uses its stored median/worst node point; related memory
bytes and percentages are read from the row selected by the same point.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from traceml_ai.reporting.terminal_card.common import (
    as_float,
    as_int,
    as_mapping,
    average,
    diagnosis,
    format_memory_value,
    format_percent,
    format_scope,
    group_row,
    group_rows,
    metadata,
    node_of,
    point,
    point_value,
    status_spans,
)
from traceml_ai.reporting.terminal_card.evidence import format_section_evidence
from traceml_ai.reporting.terminal_card.layout import (
    RUN_LEFT_PANE_WIDTH,
    CardDoc,
    PaneStages,
    append_table_header,
    append_table_row,
    new_pane_stages,
)

_SYSTEM_LABEL_WIDTH = 23
_SYSTEM_MEDIAN_WIDTH = 18


def observed_gpu_count(system_summary: Mapping[str, Any]) -> Optional[int]:
    """Return only the physical GPU count reported by System telemetry."""
    section_metadata = metadata(system_summary)
    if "gpus_observed" not in section_metadata:
        return None
    return as_int(section_metadata.get("gpus_observed"))


def _node_count(system_summary: Mapping[str, Any]) -> int:
    """Return observed System nodes without consulting run topology."""
    observed = as_int(metadata(system_summary).get("nodes_observed"))
    if observed is not None:
        return max(0, observed)

    identities = set()
    for row_id, raw_row in group_rows(system_summary).items():
        row_identity = as_mapping(as_mapping(raw_row).get("identity"))
        node = as_int(row_identity.get("node_rank"))
        hostname = str(row_identity.get("hostname") or "").strip()
        if node is not None:
            identities.add(("node", node))
        elif hostname:
            identities.add(("host", hostname))
        else:
            identities.add(("row", str(row_id)))
    return len(identities)


def coverage_text(system_summary: Mapping[str, Any]) -> Optional[str]:
    """Return stored node coverage when it adds information to the pane."""
    section_metadata = metadata(system_summary)
    coverage = str(section_metadata.get("nodes_coverage") or "").strip()
    observed = as_int(section_metadata.get("nodes_observed"))
    expected = as_int(section_metadata.get("nodes_expected"))
    if not coverage and observed is not None and expected is not None:
        coverage = f"{observed}/{expected}"

    node_count = _node_count(system_summary)
    if not coverage:
        return f"{node_count} nodes" if node_count > 1 else None
    if coverage == "1/1" and node_count <= 1:
        return None
    return f"{coverage} {'node' if expected == 1 else 'nodes'}"


def _point_metrics(
    system_summary: Mapping[str, Any],
    block: str,
    *,
    anchors: Sequence[str],
) -> Tuple[Mapping[str, Any], Optional[int]]:
    """Resolve one stored System point to its coherent node row."""
    for metric in anchors:
        stored_point = point(system_summary, block, metric)
        value = as_float(stored_point.get("value"))
        if value is None:
            continue
        idx = stored_point.get("idx")
        row = group_row(system_summary, idx)
        metrics = as_mapping(row.get("metrics"))
        if not metrics:
            metrics = {metric: value}
        return metrics, node_of(system_summary, idx)
    return {}, None


def _scalar_point(
    system_summary: Mapping[str, Any],
    block: str,
    metric: str,
    *,
    suffix: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Format one stored node-average point and resolve its node identity."""
    stored_point = point(system_summary, block, metric)
    value = as_float(stored_point.get("value"))
    if value is None:
        return None, None
    formatted = format_percent(value)
    return (
        f"{formatted}{suffix}" if formatted is not None else None,
        node_of(system_summary, stored_point.get("idx")),
    )


def _append_scalar_metric(
    doc: CardDoc,
    system_summary: Mapping[str, Any],
    *,
    label: str,
    metric: str,
    suffix: str,
    multi_node: bool,
) -> None:
    """Append one scalar System metric from stored average/point fields."""
    average_text = format_percent(average(system_summary, metric))
    median, _ = _scalar_point(system_summary, "median", metric, suffix=suffix)
    worst, worst_node = _scalar_point(
        system_summary, "worst", metric, suffix=suffix
    )
    append_table_row(
        doc,
        label=label,
        average=(
            f"{average_text}{suffix}" if average_text is not None else None
        ),
        median=median,
        worst=worst,
        worst_scope=format_scope(node=worst_node),
        multi=multi_node,
        label_width=_SYSTEM_LABEL_WIDTH,
        median_width=_SYSTEM_MEDIAN_WIDTH,
    )


def _append_memory_metric(
    doc: CardDoc,
    system_summary: Mapping[str, Any],
    *,
    label: str,
    bytes_metric: str,
    percent_metric: str,
    multi_node: bool,
) -> None:
    """Append a coherent System memory pair from stored node-average fields."""
    average_metrics = as_mapping(
        as_mapping(system_summary.get("global")).get("average")
    )
    median_metrics, _ = _point_metrics(
        system_summary, "median", anchors=(percent_metric, bytes_metric)
    )
    worst_metrics, worst_node = _point_metrics(
        system_summary, "worst", anchors=(percent_metric, bytes_metric)
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
        worst_scope=format_scope(node=worst_node),
        multi=multi_node,
        label_width=_SYSTEM_LABEL_WIDTH,
        median_width=_SYSTEM_MEDIAN_WIDTH,
    )


def _has_table_data(
    system_summary: Mapping[str, Any], *, multi_node: bool, show_gpu: bool
) -> bool:
    """Return whether the selected System table layout has a measured value."""
    metrics = ["cpu_percent", "ram_bytes", "ram_percent"]
    if show_gpu:
        metrics.extend(
            [
                "gpu_util_percent",
                "gpu_mem_bytes",
                "gpu_mem_percent",
                "gpu_temp_c",
                "gpu_power_w",
            ]
        )
    if not multi_node:
        return any(
            average(system_summary, metric) is not None for metric in metrics
        )
    return any(
        point_value(system_summary, block, metric) is not None
        for block in ("median", "worst")
        for metric in metrics
    )


def build_system_pane(system_summary: Mapping[str, Any]) -> PaneStages:
    """Build staged System Metrics content shared by Run and Watch."""
    stages = new_pane_stages(RUN_LEFT_PANE_WIDTH)
    section_diagnosis = diagnosis(system_summary)
    multi_node = _node_count(system_summary) > 1
    stages.heading.wrapped_spans(
        *status_spans(
            "SYSTEM METRICS", section_diagnosis, coverage_text(system_summary)
        )
    )

    show_gpu = observed_gpu_count(system_summary) != 0
    if _has_table_data(
        system_summary, multi_node=multi_node, show_gpu=show_gpu
    ):
        append_table_header(
            stages.table,
            multi=multi_node,
            median_label="median node avg",
            worst_label="worst node avg",
            label_width=_SYSTEM_LABEL_WIDTH,
            median_width=_SYSTEM_MEDIAN_WIDTH,
        )
        _append_scalar_metric(
            stages.table,
            system_summary,
            label="CPU",
            metric="cpu_percent",
            suffix="%",
            multi_node=multi_node,
        )
        _append_memory_metric(
            stages.table,
            system_summary,
            label="RAM used",
            bytes_metric="ram_bytes",
            percent_metric="ram_percent",
            multi_node=multi_node,
        )
        if show_gpu:
            _append_scalar_metric(
                stages.table,
                system_summary,
                label="GPU util",
                metric="gpu_util_percent",
                suffix="%",
                multi_node=multi_node,
            )
            _append_memory_metric(
                stages.table,
                system_summary,
                label="GPU memory/device",
                bytes_metric="gpu_mem_bytes",
                percent_metric="gpu_mem_percent",
                multi_node=multi_node,
            )
            _append_scalar_metric(
                stages.table,
                system_summary,
                label="GPU temperature",
                metric="gpu_temp_c",
                suffix="C",
                multi_node=multi_node,
            )
            _append_scalar_metric(
                stages.table,
                system_summary,
                label="GPU power",
                metric="gpu_power_w",
                suffix="W",
                multi_node=multi_node,
            )

    evidence = format_section_evidence("system", system_summary)
    if evidence:
        stages.evidence.wrapped(evidence, label="Evidence: ")
    else:
        stages.evidence.blank()
    stages.spacer.blank()
    return stages


__all__ = ["build_system_pane", "coverage_text", "observed_gpu_count"]
