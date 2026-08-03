# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Server-side inline bars for the report (no JS, no chart library)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .sections_helpers import sorted_rows
from .textutils import esc, fmt_value

# step_time selected-clock breakdown, in stacking order.
_PHASES = (
    ("input_wait_ms", "input wait", "var(--dl)"),
    ("h2d_ms", "h2d", "var(--h2d)"),
    ("forward_ms", "forward", "var(--fwd)"),
    ("backward_ms", "backward", "var(--bwd)"),
    ("optimizer_ms", "optimizer", "var(--opt)"),
    ("residual_ms", "residual", "var(--residual)"),
)


def _schema_number(value: Any) -> Optional[float]:
    """Return a usable schema number from a final-summary payload value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase_bar_denominator(
    avg: Dict[str, Any],
    *,
    schema_version: Any,
) -> Optional[float]:
    """Return canonical Step Time, adapting only pre-1.8 report payloads."""
    schema = _schema_number(schema_version)
    if schema is not None and schema >= 1.8:
        value = avg.get("step_time_ms")
        return float(value) if isinstance(value, (int, float)) else None

    # Compatibility reader for historical final summaries only: before 1.8,
    # step_time_ms was the traced envelope and the outer duration was absent.
    traced = avg.get("step_time_ms")
    input_wait = avg.get("input_wait_ms")
    if input_wait is None and "input_wait_ms" not in avg:
        input_wait = avg.get("dataloader_ms")
    if isinstance(traced, (int, float)) and isinstance(
        input_wait, (int, float)
    ):
        return float(traced) + float(input_wait)
    return None


def phase_bar(
    step_time_section: Dict[str, Any],
    *,
    schema_version: Any = 1.8,
) -> str:
    """Render phase widths against canonical Step Time without rescaling."""
    avg = (step_time_section.get("global") or {}).get("average") or {}
    if not isinstance(avg, dict):
        return ""
    step_time = _phase_bar_denominator(avg, schema_version=schema_version)
    if step_time is None or step_time <= 0.0:
        return ""

    present: List[Tuple[str, str, float]] = []
    for metric, label, color in _PHASES:
        value = avg.get(metric)
        if (
            value is None
            and metric == "input_wait_ms"
            and "input_wait_ms" not in avg
        ):
            # Pre-1.6 reports never carried this key at all. A newer report
            # always carries the key, so a
            # present-but-null value here means genuinely unmeasured,
            # not "borrow dataloader_ms" -- fall through to the isinstance
            # check below and drop this phase from the chart.
            value = avg.get("dataloader_ms")
        if isinstance(value, (int, float)) and value > 0:
            present.append((label, color, float(value)))
    if not present:
        return ""

    rects: List[str] = []
    legend: List[str] = []
    x = 0.0
    for label, color, value in present:
        width = 100.0 * value / step_time
        rects.append(
            f'<rect x="{x:.2f}%" y="4" width="{width:.2f}%" height="18" '
            f'fill="{color}"/>'
        )
        legend.append(
            f'<span><i class="sw" style="background:{color}"></i>'
            f"{esc(label)} {value:,.1f}&thinsp;ms</span>"
        )
        x += width
    return (
        '<svg width="100%" height="26" role="img" '
        'aria-label="average step phase breakdown">'
        + "".join(rects)
        + "</svg>"
        + f'<div class="legend">{"".join(legend)}</div>'
    )


def _process_capacity(process_section: Dict[str, Any]) -> Dict[str, float]:
    """Per-rank GPU total = reserved + headroom, keyed by rank label."""
    rows = (process_section.get("groups") or {}).get("rows") or {}
    if not isinstance(rows, dict):
        return {}
    capacity: Dict[str, float] = {}
    for label, row in rows.items():
        metrics = row.get("metrics") or {}
        reserved = metrics.get("gpu_mem_reserved_bytes")
        headroom = metrics.get("gpu_mem_headroom_bytes")
        if isinstance(reserved, (int, float)) and isinstance(
            headroom, (int, float)
        ):
            total = float(reserved) + float(headroom)
            if total > 0:
                capacity[str(label)] = total
    return capacity


def memory_bars(
    step_memory_section: Dict[str, Any],
    process_section: Dict[str, Any],
) -> str:
    """
    Per-rank peak-reserved bars.

    Denominator preference: per-rank GPU capacity derived from the process
    section (reserved + headroom), so the bar reads as true capacity
    pressure. Fallback when process data is absent: scale to the worst rank,
    with a caption that says so (a balanced low-utilization run must not look
    saturated).
    """
    rows = (step_memory_section.get("groups") or {}).get("rows") or {}
    if not isinstance(rows, dict) or not rows:
        return ""

    peaks: List[Tuple[str, str, float]] = []
    for label, row in sorted_rows(rows):
        identity = row.get("identity") or {}
        value = (row.get("metrics") or {}).get("peak_reserved_bytes")
        if isinstance(value, (int, float)):
            peaks.append(
                (str(label), str(identity.get("hostname", "")), float(value))
            )
    if not peaks:
        return ""

    capacity = _process_capacity(process_section)
    use_capacity = all(label in capacity for label, _, _ in peaks)
    if use_capacity:
        caption = "bars = share of per-rank GPU capacity (process section)"
    else:
        worst = max(value for _, _, value in peaks) or 1.0
        caption = "bars relative to worst rank &mdash; not % of GPU capacity"

    bars: List[str] = []
    for label, host, value in peaks:
        denom = capacity[label] if use_capacity else worst
        width = max(2.0, min(100.0, 100.0 * value / denom))
        bars.append(
            f'<div class="membar"><span>{esc(label)} &middot; '
            f"{esc(host)}</span>"
            f'<div class="track"><div class="fill" '
            f'style="width:{width:.0f}%"></div></div>'
            f'<span class="num">'
            f"{fmt_value('peak_reserved_bytes', value)}</span></div>"
        )
    return (
        f'<div class="membars">{"".join(bars)}</div>'
        f'<div class="legend">{caption}</div>'
    )


__all__ = ["memory_bars", "phase_bar"]
