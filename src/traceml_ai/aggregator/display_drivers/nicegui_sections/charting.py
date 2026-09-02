# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shaping a chart: axes, ranges, per-rank series, sparklines.

``theme`` owns the look: palette, fonts, CSS. The ECharts option dicts
moved here in part 6 of #403, because building one is chart construction
rather than styling.
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

# The palette stays in theme; a chart builder borrows from it
# rather than keeping a second copy of the brand colours.
from .theme import BORDER, INK

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


_GRID = "rgba(17,24,39,0.05)"
_AXIS = "rgba(17,24,39,0.14)"
_TXT = "#9aa3af"


def _area(c1: str, c2: str) -> Dict[str, Any]:
    return {
        "type": "linear",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 1,
        "colorStops": [
            {"offset": 0, "color": c1},
            {"offset": 0.85, "color": c2},
            {"offset": 1, "color": "rgba(255,255,255,0)"},
        ],
    }


def _span_axis() -> Dict[str, Any]:
    """Seconds-before-newest x axis; the section sets min/interval/labels."""
    return {
        "type": "value",
        "min": -1,
        "max": 0,
        "interval": 1,
        # No tick labels: the window is named once in the chart's label row.
        "axisLabel": {"show": False},
        "axisLine": {"lineStyle": {"color": _AXIS, "opacity": 0.5}},
        "axisTick": {"show": False},
        "splitLine": {"show": False},
    }


def _value_axis(unit: str, *, zero: bool) -> Dict[str, Any]:
    ax: Dict[str, Any] = {
        "type": "value",
        "splitNumber": 2,
        "axisLabel": {
            "color": _TXT,
            "fontFamily": "Geist Mono",
            "fontSize": 10,
            ":formatter": f"v=>v+'{unit}'",
        },
        "axisLine": {"show": False},
        "axisTick": {"show": False},
        "splitLine": {"lineStyle": {"color": _GRID}},
    }
    if zero:
        ax["min"] = 0
    return ax


_SPAN_POINTER_LABEL = (
    "p=>{const s=Math.round(-p.value);return s<1?'now':"
    "(s<120?s+' s ago':Math.floor(s/60)+' min '+(s%60)+' s ago');}"
)


def _tooltip(unit: str) -> Dict[str, Any]:
    return {
        "trigger": "axis",
        "backgroundColor": "rgba(255,253,250,0.97)",
        "borderColor": BORDER,
        "textStyle": {
            "color": INK,
            "fontFamily": "Geist Mono",
            "fontSize": 11,
        },
        "axisPointer": {
            "type": "line",
            "lineStyle": {"color": _AXIS, "type": "dashed"},
            "label": {
                "backgroundColor": INK,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
                ":formatter": _SPAN_POINTER_LABEL,
            },
        },
        ":valueFormatter": f"v=>(v==null?'-':Math.round(v)+'{unit}')",
    }


def line_series(
    name: str, col: str, data: List[Any], *, width: float = 1.6
) -> Dict[str, Any]:
    """One plain line (no area, no end label) for a multi-trace chart."""
    return {
        "name": name,
        "type": "line",
        "smooth": True,
        "showSymbol": False,
        "lineStyle": {"width": width, "color": col},
        "itemStyle": {"color": col},
        "data": data,
    }


def mark_lines(entries: List[Any]) -> Dict[str, Any]:
    """Several horizontal reference lines on one series.

    Each entry is (y, label, colour, position); ECharts takes them as
    markLine data with per-item style, so one series can carry a limit and
    a floor. Give two lines different positions: a label anchored at the
    same end as its neighbour collides with it, and a long one anchored at
    the right end is cut off by the card edge.
    """
    return {
        "silent": True,
        "symbol": "none",
        "animation": False,
        "data": [
            {
                "yAxis": y,
                "lineStyle": {"color": col, "type": "dashed", "width": 1},
                "label": {
                    "show": True,
                    "position": pos,
                    "formatter": label,
                    "color": col,
                    "fontFamily": "Geist Mono",
                    "fontSize": 10,
                },
            }
            for y, label, col, pos in entries
        ],
    }


def span_line_options(col: str, unit: str) -> Dict[str, Any]:
    """Small zero-anchored single series over a window-span x axis."""
    return {
        "backgroundColor": "transparent",
        "animationDuration": 300,
        "color": [col],
        "grid": {
            "left": 4,
            # room for the last clock label, which is centred on the axis
            # maximum and would otherwise be cut in half by the card edge
            "right": 26,
            "top": 8,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": _tooltip(unit),
        "xAxis": _span_axis(),
        "yAxis": _value_axis(unit, zero=True),
        "series": [
            {
                **line_series("", col, []),
                "areaStyle": {
                    "color": _area(
                        "rgba(37,99,235,0.14)", "rgba(37,99,235,0.03)"
                    )
                },
            }
        ],
    }


def multi_line_options(unit: str) -> Dict[str, Any]:
    """Several plain traces over a window-span x axis (one per GPU)."""
    return {
        "backgroundColor": "transparent",
        "animationDuration": 300,
        "grid": {
            "left": 4,
            # room for the last clock label, which is centred on the axis
            # maximum and would otherwise be cut in half by the card edge
            "right": 26,
            "top": 10,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": _tooltip(unit),
        "xAxis": _span_axis(),
        "yAxis": _value_axis(unit, zero=False),
        "series": [],
    }


__all__ = [
    "multi_line_options",
    "span_line_options",
    "mark_lines",
    "line_series",
    "RANK_COLORS",
    "apply_span_axis",
    "capacity_axis_max",
    "drift_axis_bounds",
    "rank_color",
    "shared_span",
    "sparkline_svg",
    "value_axis_formatter",
]
