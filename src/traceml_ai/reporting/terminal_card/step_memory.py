# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run-card Step Memory pane.

The stored values are averages of per-step peaks. In distributed output the
reserved-memory median/worst points select coherent grouped rank rows, and
both Allocated and Reserved are read from each selected row.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from traceml_ai.reporting.terminal_card.common import (
    as_float,
    as_int,
    as_mapping,
    average,
    diagnosis,
    format_capacity,
    format_scope,
    group_row,
    point,
    rank_count,
    status_spans,
)
from traceml_ai.reporting.terminal_card.evidence import format_section_evidence
from traceml_ai.reporting.terminal_card.layout import (
    STYLE_DIM,
    CardDoc,
    append_table_row,
)

_LABEL_WIDTH = 28
_MEDIAN_WIDTH = 20


def _coverage_text(
    step_memory_summary: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> Optional[str]:
    """Return observed/expected Step Memory rank coverage."""
    observed = rank_count(step_memory_summary)
    expected = as_int(meta.get("world_size"))
    if expected is not None and expected > 0:
        if observed == expected == 1:
            return None
        return f"{observed}/{expected} {'rank' if expected == 1 else 'ranks'}"
    if observed > 0:
        return f"{observed} {'rank' if observed == 1 else 'ranks'} observed"
    return None


def _selected_row(
    step_memory_summary: Mapping[str, Any], block: str
) -> Mapping[str, Any]:
    """Return the coherent rank row selected for a comparison column."""
    for metric in ("peak_reserved_bytes", "peak_allocated_bytes"):
        selected = point(step_memory_summary, block, metric)
        if as_float(selected.get("value")) is None:
            continue
        row = group_row(step_memory_summary, selected.get("idx"))
        if row:
            return row
    return {}


def _row_scope(row: Mapping[str, Any]) -> Optional[str]:
    """Return the rank/node identity of one selected grouped rank row."""
    row_identity = as_mapping(row.get("identity"))
    rank = as_int(row_identity.get("global_rank"))
    if rank is None:
        return None
    node = as_int(row_identity.get("node_rank"))
    return format_scope(rank=rank, node=node)


def _append_table_header(doc: CardDoc, *, multi_rank: bool) -> None:
    """Append the Step Memory table header for average per-step peaks."""
    if multi_rank:
        doc.text(
            f"{'avg per-step peak':<{_LABEL_WIDTH}}"
            f"{'median rank avg':<{_MEDIAN_WIDTH}}"
            "worst rank avg",
            STYLE_DIM,
        )
    else:
        doc.text(f"{'avg per-step peak':<{_LABEL_WIDTH}}avg", STYLE_DIM)


def build_run_step_memory_pane(
    step_memory_summary: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    width: int = 78,
) -> CardDoc:
    """Build the Run Step Memory pane using only stored rank rollups."""
    doc = CardDoc(width=width)
    section_diagnosis = diagnosis(step_memory_summary)
    multi_rank = rank_count(step_memory_summary) > 1
    median_row = _selected_row(step_memory_summary, "median")
    worst_row = _selected_row(step_memory_summary, "worst")
    median_metrics = as_mapping(median_row.get("metrics"))
    worst_metrics = as_mapping(worst_row.get("metrics"))
    doc.wrapped_spans(
        *status_spans(
            "STEP MEMORY",
            section_diagnosis,
            _coverage_text(step_memory_summary, meta=meta),
            details_style=STYLE_DIM,
        )
    )
    evidence = format_section_evidence("step_memory", step_memory_summary)
    if evidence:
        doc.wrapped(evidence, label="Evidence: ")
    else:
        doc.blank()
    if str(section_diagnosis.get("kind") or "NO_DATA") in {
        "NO_DATA",
        "NO_GPU",
    }:
        return doc
    metrics = (
        ("Allocated", "peak_allocated_bytes"),
        ("Reserved", "peak_reserved_bytes"),
    )
    if multi_rank:
        has_table_data = any(
            as_float(row.get(metric)) is not None
            for row in (median_metrics, worst_metrics)
            for _, metric in metrics
        )
    else:
        has_table_data = any(
            average(step_memory_summary, metric) is not None
            for _, metric in metrics
        )
    if not has_table_data:
        return doc

    doc.blank()
    _append_table_header(doc, multi_rank=multi_rank)
    for label, metric in metrics:
        append_table_row(
            doc,
            label=label,
            average=format_capacity(average(step_memory_summary, metric)),
            median=format_capacity(as_float(median_metrics.get(metric))),
            worst=format_capacity(as_float(worst_metrics.get(metric))),
            worst_scope=_row_scope(worst_row),
            multi=multi_rank,
        )
    return doc


__all__ = ["build_run_step_memory_pane"]
