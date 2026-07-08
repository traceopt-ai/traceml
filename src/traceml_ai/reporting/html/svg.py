# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Server-side inline bars for the report (no JS, no chart library)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


def phase_bar(step_time_section: Dict[str, Any]) -> str:
    """Stacked horizontal bar of the average step's phase breakdown."""
    avg = (step_time_section.get("global") or {}).get("average") or {}
    present: List[Tuple[str, str, float]] = []
    total = 0.0
    for metric, label, color in _PHASES:
        value = avg.get(metric)
        if isinstance(value, (int, float)) and value > 0:
            present.append((label, color, float(value)))
            total += float(value)
    if total <= 0:
        return ""

    rects: List[str] = []
    legend: List[str] = []
    x = 0.0
    for label, color, value in present:
        width = 100.0 * value / total
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
