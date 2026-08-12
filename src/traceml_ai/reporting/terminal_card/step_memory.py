# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run-card Step Memory pane.

The stored values are averages of per-step peaks.  In distributed output each
Allocated/Reserved metric independently uses its stored median/worst rank
point; the pane deliberately does not create a synthetic coherent pair.
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
    group_rows,
    identity,
    metadata,
    point,
    point_value,
    status_spans,
)
from traceml_ai.reporting.terminal_card.layout import (
    STYLE_DIM,
    CardDoc,
    append_table_row,
)
from traceml_ai.reporting.terminal_card.run_evidence import format_run_evidence

_LABEL_WIDTH = 28
_MEDIAN_WIDTH = 20


def _rank_count(step_memory_summary: Mapping[str, Any]) -> int:
    """Return Step Memory ranks from stored metadata or grouped rows."""
    used = as_int(metadata(step_memory_summary).get("global_ranks_used"))
    if used is not None:
        return max(0, used)

    ranks = set()
    anonymous_rows = 0
    for raw_row in group_rows(step_memory_summary).values():
        rank = as_int(
            as_mapping(as_mapping(raw_row).get("identity")).get("global_rank")
        )
        if rank is None:
            anonymous_rows += 1
        else:
            ranks.add(rank)
    return len(ranks) + anonymous_rows


def _coverage_text(
    step_memory_summary: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> Optional[str]:
    """Return observed/expected Step Memory rank coverage."""
    observed = _rank_count(step_memory_summary)
    expected = as_int(meta.get("world_size"))
    if expected is not None and expected > 0:
        if observed == expected == 1:
            return None
        return f"{observed}/{expected} {'rank' if expected == 1 else 'ranks'}"
    if observed > 0:
        return f"{observed} {'rank' if observed == 1 else 'ranks'} observed"
    return None


def _point_scope(
    step_memory_summary: Mapping[str, Any], idx: Any
) -> Optional[str]:
    """Resolve one stored Step Memory point to its rank/node identity."""
    row_identity = identity(step_memory_summary, idx)
    rank = as_int(row_identity.get("global_rank"))
    if rank is None:
        return None
    node = as_int(row_identity.get("node_rank"))
    return format_scope(rank=rank, node=node)


def _has_table_data(
    step_memory_summary: Mapping[str, Any], *, multi_rank: bool
) -> bool:
    """Return whether the selected Step Memory table has measured values."""
    metrics = ("peak_allocated_bytes", "peak_reserved_bytes")
    if not multi_rank:
        return any(
            average(step_memory_summary, metric) is not None
            for metric in metrics
        )
    return any(
        point_value(step_memory_summary, block, metric) is not None
        for block in ("median", "worst")
        for metric in metrics
    )


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


def _append_metric(
    doc: CardDoc,
    step_memory_summary: Mapping[str, Any],
    *,
    label: str,
    metric: str,
    multi_rank: bool,
) -> None:
    """Append one per-metric average-of-per-step-peaks row from JSON."""
    median_point = point(step_memory_summary, "median", metric)
    worst_point = point(step_memory_summary, "worst", metric)
    append_table_row(
        doc,
        label=label,
        average=format_capacity(average(step_memory_summary, metric)),
        median=format_capacity(as_float(median_point.get("value"))),
        worst=format_capacity(as_float(worst_point.get("value"))),
        worst_scope=_point_scope(step_memory_summary, worst_point.get("idx")),
        multi=multi_rank,
    )


def build_run_step_memory_pane(
    step_memory_summary: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    width: int = 78,
) -> CardDoc:
    """Build the Run Step Memory pane using only stored rank rollups."""
    doc = CardDoc(width=width)
    section_diagnosis = diagnosis(step_memory_summary)
    multi_rank = _rank_count(step_memory_summary) > 1
    doc.wrapped_spans(
        *status_spans(
            "STEP MEMORY",
            section_diagnosis,
            _coverage_text(step_memory_summary, meta=meta),
            details_style=STYLE_DIM,
        )
    )
    doc.wrapped(
        format_run_evidence("step_memory", step_memory_summary) or "None",
        label="Evidence: ",
    )
    if str(section_diagnosis.get("kind") or "NO_DATA") in {
        "NO_DATA",
        "NO_GPU",
    }:
        return doc
    if not _has_table_data(step_memory_summary, multi_rank=multi_rank):
        return doc

    doc.blank()
    _append_table_header(doc, multi_rank=multi_rank)
    _append_metric(
        doc,
        step_memory_summary,
        label="Allocated",
        metric="peak_allocated_bytes",
        multi_rank=multi_rank,
    )
    _append_metric(
        doc,
        step_memory_summary,
        label="Reserved",
        metric="peak_reserved_bytes",
        multi_rank=multi_rank,
    )
    return doc


__all__ = ["build_run_step_memory_pane"]
