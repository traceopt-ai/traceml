# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Unchanged Watch-card rendering.

Watch intentionally remains a host/process health surface: it does not render
Run step timing or use Run-only diagnostic presentation helpers.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from traceml_ai.reporting.terminal_card.common import (
    SEVERITY_RANK,
    as_float,
    as_int,
    average,
    diagnosis,
    format_duration,
    format_gb,
    format_percent,
    join_segments,
    node_of,
    plural,
    point,
    point_value,
    severity,
    severity_label,
    severity_style,
    status_text,
)
from traceml_ai.reporting.terminal_card.layout import (
    STYLE_BOLD,
    STYLE_DIM,
    STYLE_NEXT,
    STYLE_OK,
    STYLE_PLAIN,
    CardDoc,
    Span,
)

LOW_GPU_UTIL_KINDS = frozenset(
    {"LOW_GPU_UTILIZATION", "MODERATE_GPU_UTILIZATION"}
)
_WATCH_LEFT_LABEL_WIDTH = 11
_WATCH_RIGHT_COLUMN = 28
_WATCH_RIGHT_LABEL_WIDTH = 9
_WATCH_MULTI_LABEL_WIDTH = 12
_WATCH_MULTI_WORST_COLUMN = 26
_WATCH_BOUNDARY = (
    "Watch monitors host and process health only; it does not measure "
    "step time. To diagnose training speed: traceml run <your-script>.py"
)
_WATCH_RUN_ACTION = (
    "traceml run <your-script>.py -- measures step time and finds the cause."
)


def watch_header_meta(
    *, meta: Mapping[str, Any], duration_s: Optional[float], multi: bool
) -> str:
    """Compose the Watch card's identity and coverage line."""
    gpus = as_int(meta.get("gpus_observed"))
    nodes = as_int(meta.get("nodes_observed"))

    segments: List[Optional[str]] = []
    if multi and nodes is not None:
        segments.append(plural(nodes, "node"))
    else:
        segments.append("1 machine")
    if gpus == 0:
        segments.append("no GPU detected")
    elif gpus is not None:
        segments.append(plural(gpus, "GPU"))
    duration = format_duration(duration_s)
    if duration is not None:
        segments.append(f"observed for {duration}")
    return join_segments(segments)


def _health_lines(
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Return the host-health lines and the finding action, when present."""
    findings = []
    for section in (system_summary, process_summary):
        section_diagnosis = diagnosis(section)
        level = severity(section_diagnosis.get("severity"))
        if level in SEVERITY_RANK:
            findings.append((SEVERITY_RANK[level], section_diagnosis, level))
    findings.sort(key=lambda item: item[0])

    if not findings:
        nodes = as_int(meta.get("nodes_observed"))
        scope = f" on all {nodes} nodes" if multi and nodes else ""
        return [(f"Host health: NORMAL{scope}", STYLE_OK)], None

    _rank, section_diagnosis, level = findings[0]
    status = status_text(section_diagnosis.get("status"))
    label = severity_label(level)
    lines = [(f"Host health: {status}  ({label})", severity_style(level))]
    summary = str(section_diagnosis.get("summary") or "").strip()
    if summary:
        lines.append((summary, STYLE_PLAIN))
    action = str(section_diagnosis.get("action") or "").strip()
    return lines, (action or None)


def _total_gb(
    used_bytes: Optional[float], percent: Optional[float]
) -> Optional[float]:
    """Derive a capacity total from a used-bytes and percent pair."""
    if used_bytes is None or percent is None or percent <= 0.0:
        return None
    return float(used_bytes) / (float(percent) / 100.0)


def _capacity_cell(
    label: str, used_bytes: Optional[float], percent: Optional[float]
) -> Optional[str]:
    """Return a `RAM  6.2 / 32.0 GB  (19%)` style Watch right-column cell."""
    used = format_gb(used_bytes)
    if used is None:
        return None
    total = format_gb(_total_gb(used_bytes, percent))
    head = f"{label:<{_WATCH_RIGHT_LABEL_WIDTH}}{used:>4}"
    if total is None:
        return f"{head} GB used"
    percent_text = format_percent(percent)
    return f"{head} / {total:>4} GB  ({percent_text}%)"


def _single_metric_rows(
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    show_gpu: bool,
) -> List[str]:
    """Return the two-column Watch metric block for one machine."""
    left: List[str] = []
    cpu = format_percent(average(system_summary, "cpu_percent"))
    if cpu is not None:
        left.append(f"{'CPU Util':<{_WATCH_LEFT_LABEL_WIDTH}}{cpu}% avg")
    if show_gpu:
        util = format_percent(average(system_summary, "gpu_util_percent"))
        if util is not None:
            left.append(f"{'GPU Util':<{_WATCH_LEFT_LABEL_WIDTH}}{util}% avg")
        temp = point_value(system_summary, "worst", "gpu_temp_c")
        if temp is None:
            temp = average(system_summary, "gpu_temp_c")
        temp_text = format_percent(temp)
        if temp_text is not None:
            left.append(
                f"{'GPU Temp':<{_WATCH_LEFT_LABEL_WIDTH}}{temp_text}C max"
            )

    right: List[str] = []
    ram = _capacity_cell(
        "RAM",
        average(system_summary, "ram_bytes"),
        average(system_summary, "ram_percent"),
    )
    if ram is not None:
        right.append(ram)
    if show_gpu:
        gpu_mem = _capacity_cell(
            "GPU Mem",
            average(system_summary, "gpu_mem_bytes"),
            average(system_summary, "gpu_mem_percent"),
        )
        if gpu_mem is not None:
            right.append(gpu_mem)
    rss = format_gb(point_value(process_summary, "worst", "ram_bytes"))
    if rss is None:
        rss = format_gb(average(process_summary, "ram_bytes"))
    if rss is not None:
        right.append(f"{'Proc RSS':<{_WATCH_RIGHT_LABEL_WIDTH}}{rss:>4} GB")

    rows: List[str] = []
    for index in range(max(len(left), len(right))):
        head = left[index] if index < len(left) else ""
        tail = right[index] if index < len(right) else ""
        if tail:
            rows.append(f"{head:<{_WATCH_RIGHT_COLUMN}}{tail}")
        else:
            rows.append(head)
    return rows


_MULTI_METRICS = (
    ("CPU Util", "cpu_percent", "%", "avg"),
    ("RAM", "ram_percent", "%", "used"),
    ("GPU Util", "gpu_util_percent", "%", "avg"),
    ("GPU Mem", "gpu_mem_percent", "%", "used"),
    ("GPU Temp", "gpu_temp_c", "C", "max"),
)


def _multi_metric_rows(
    system_summary: Mapping[str, Any], *, show_gpu: bool
) -> List[str]:
    """Return the median / worst-node Watch metric table."""
    rows: List[str] = []
    for label, metric, unit, qualifier in _MULTI_METRICS:
        if not show_gpu and metric.startswith("gpu_"):
            continue
        median = format_percent(point_value(system_summary, "median", metric))
        if median is None:
            continue
        cell = f"{label:<{_WATCH_MULTI_LABEL_WIDTH}}{median}{unit} {qualifier}"
        worst_point = point(system_summary, "worst", metric)
        worst = format_percent(as_float(worst_point.get("value")))
        node = node_of(system_summary, worst_point.get("idx"))
        if worst is not None and node is not None:
            cell = (
                f"{cell:<{_WATCH_MULTI_WORST_COLUMN}}{worst}{unit}  (n{node})"
            )
        rows.append(cell)
    if rows:
        header = (
            f"{'':<{_WATCH_MULTI_LABEL_WIDTH}}median"
            f"{'':<{_WATCH_MULTI_WORST_COLUMN - _WATCH_MULTI_LABEL_WIDTH - 6}}"
            "worst node"
        )
        rows.insert(0, header)
    return rows


def append_watch_body(
    doc: CardDoc,
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> None:
    """Append the Watch card body between the header and the footer."""
    show_gpu = as_int(meta.get("gpus_observed")) != 0

    doc.blank()
    health_lines, health_action = _health_lines(
        system_summary=system_summary,
        process_summary=process_summary,
        meta=meta,
        multi=multi,
    )
    for index, (text, style) in enumerate(health_lines):
        if index == 0:
            doc.add(Span("Host health: ", STYLE_BOLD), Span(text[13:], style))
        else:
            doc.wrapped(text, style=style)
    doc.blank()

    if multi:
        rows = _multi_metric_rows(system_summary, show_gpu=show_gpu)
    else:
        rows = _single_metric_rows(
            system_summary=system_summary,
            process_summary=process_summary,
            show_gpu=show_gpu,
        )
    for text in rows:
        doc.text(text)
    if rows:
        doc.blank()

    system_kind = str(diagnosis(system_summary).get("kind") or "")
    printed_next = False
    if show_gpu and system_kind in LOW_GPU_UTIL_KINDS:
        util = format_percent(average(system_summary, "gpu_util_percent"))
        doc.wrapped(
            f"GPU utilization stayed low ({util}% avg). Watch cannot tell "
            "whether that is input, transfer, sync, or idle time.",
            label="Observation: ",
        )
        doc.wrapped(_WATCH_RUN_ACTION, label="Next: ", style=STYLE_NEXT)
        printed_next = True
    elif health_action:
        doc.wrapped(health_action, label="Next: ", style=STYLE_NEXT)
        doc.blank()

    if not printed_next:
        doc.wrapped(_WATCH_BOUNDARY, style=STYLE_DIM)


__all__ = ["append_watch_body", "watch_header_meta"]
