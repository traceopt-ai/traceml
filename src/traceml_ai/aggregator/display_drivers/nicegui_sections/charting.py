# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shaping a chart: axes, ranges, per-rank series, sparklines.

``theme`` owns the look (palette, fonts, the base ECharts option dicts).
This module owns the arithmetic of fitting a chart to its data: what the
y range should be for a given kind of signal, how a time axis is pinned and
labelled, and how one series per rank is built.

The split matters because the two change for different reasons. A palette
change is a brand decision; an axis-range change is a decision about what a
metric's information IS. Keeping them in one file meant every axis fix
touched the module that defines the brand.

No function here reads a payload or decides severity. They take numbers and
return option fragments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# One colour per rank, shared by the charts and the rows' chips so a line
# and a row are recognisably the same rank. Red is not among them: on this
# page red reads as a verdict, and a rank identifier is not a verdict.
RANK_COLORS: Tuple[str, ...] = (
    "#f97316",
    "#3b82f6",
    "#0d9488",
    "#a855f7",
    "#0ea5e9",
    "#eab308",
    "#ec4899",
    "#10b981",
)


def rank_color(rank: int) -> str:
    """The colour that identifies one rank, stable across every surface."""
    return RANK_COLORS[int(rank) % len(RANK_COLORS)]


def capacity_axis_max(values: Sequence[Any]) -> float:
    """Zero-anchored ceiling whose half is a whole percent (0 / 5 / 10).

    Right for a share of a capacity: the distance from zero is the reading,
    so zero must be on the axis.
    """
    numbers = [float(v) for v in values if v is not None]
    peak = (max(numbers) * 1.2) if numbers else 0.0
    for top in (4.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0):
        if peak <= top:
            return top
    return 100.0


def drift_axis_bounds(
    values: Sequence[Any],
) -> Tuple[float, float, float]:
    """A y range fitted to the data, for a series whose signal is DRIFT.

    RSS is a level a few GB high that moves by tens of MB across a run.
    Zero-anchoring it, which is right for a share of capacity, puts that
    whole movement inside one pixel: on the three-hour capture the ranks
    sit at 1.48 to 1.50 GB on an axis that would run to 5. The leak this
    chart exists to show would be invisible.
    """
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return (0.0, 1.0, 0.5)
    low, high = min(numbers), max(numbers)
    if high <= low:
        high = low + max(abs(low) * 0.01, 0.01)
    pad = (high - low) * 0.25
    low, high = max(0.0, low - pad), high + pad
    return (low, high, (high - low) / 2.0)


def value_axis_formatter(span: float, unit: str) -> str:
    """Tick formatter whose precision comes from the axis RANGE.

    Magnitude alone is not enough: an axis fitted to a 20 MB drift around
    1.4 GB would label every tick "1.4 GB" and say nothing.
    """
    if span >= 20:
        decimals = 0
    elif span >= 2:
        decimals = 1
    elif span >= 0.2:
        decimals = 2
    else:
        decimals = 3
    return f"v=>v.toFixed({decimals})+'{unit}'"


def apply_span_axis(
    options: Dict[str, Any],
    span: float,
    newest_epoch: Optional[float] = None,
) -> None:
    """Pin a chart to its span and label it in wall-clock time.

    The x values are seconds before the newest sample, which keeps the
    series arithmetic simple, but a reader debugging a slowdown needs the
    clock: it is what their logs are keyed on. The formatters convert on
    the fly from the newest sample's epoch, and the hover label carries
    both readings ("19:10 · 45 min ago") so the axis and the tooltip never
    speak two different vocabularies.
    """
    span = max(float(span), 1.0)
    axis = options["xAxis"]
    axis["min"] = -span
    axis["max"] = 0
    axis["interval"] = span / 3.0
    if newest_epoch is None:
        axis["axisLabel"]["show"] = False
        return
    axis["axisLabel"]["show"] = True
    clock = (
        "const d=new Date((%f+%%s)*1000);const q=n=>('0'+n).slice(-2);"
        "const c=q(d.getHours())+':'+q(d.getMinutes())%s;"
        % (float(newest_epoch), "+':'+q(d.getSeconds())" if span < 600 else "")
    )
    axis["axisLabel"][":formatter"] = "v=>{%s return c;}" % (clock % "v")
    pointer = options.get("tooltip", {}).get("axisPointer", {}).get("label")
    if pointer is not None:
        pointer[":formatter"] = (
            "p=>{%s const s=Math.round(-p.value);"
            "return c+(s<1?' · now':(s<120?' · '+s+' s ago':"
            "' · '+Math.floor(s/60)+' min ago'));}" % (clock % "p.value")
        )


def sparkline_svg(
    values: Sequence[Optional[float]],
    color: str,
    *,
    width: int = 64,
    height: int = 14,
) -> str:
    """Inline SVG polyline; gaps are dropped, an empty trace is no SVG."""
    points = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if not points:
        return ""
    low = min(value for _i, value in points)
    high = max(value for _i, value in points)
    span = (high - low) or 1.0
    step = width / max(len(values) - 1, 1)
    inner = height - 4
    coords = " ".join(
        f"{i * step:.1f},{1 + inner - (value - low) / span * inner:.1f}"
        for i, value in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'style="width:{width}px;height:{height}px;vertical-align:middle">'
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="1.4"/></svg>'
    )


def shared_span(*traces: Sequence[Any]) -> Optional[Tuple[float, float]]:
    """One anchor and span covering every trace given.

    Both Process charts are pinned to it so a vertical read across the pair
    lands on the same moment. Each trace is expected to expose a
    ``timestamps`` sequence.
    """
    starts: List[float] = []
    ends: List[float] = []
    for group in traces:
        for trace in group or ():
            stamps = [
                float(value)
                for value in getattr(trace, "timestamps", ()) or ()
                if value is not None
            ]
            if stamps:
                starts.append(stamps[0])
                ends.append(stamps[-1])
    if not ends:
        return None
    newest = max(ends)
    return (newest, max(newest - min(starts), 1.0))


__all__ = [
    "RANK_COLORS",
    "apply_span_axis",
    "capacity_axis_max",
    "drift_axis_bounds",
    "rank_color",
    "shared_span",
    "sparkline_svg",
    "value_axis_formatter",
]
