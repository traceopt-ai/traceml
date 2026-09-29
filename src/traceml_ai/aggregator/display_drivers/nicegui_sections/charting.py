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

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The palette stays in theme; a chart builder borrows from it
# rather than keeping a second copy of the brand colours.
from .theme import BORDER, INK


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
    *,
    min_span: float = 0.0,
) -> Tuple[float, float, float]:
    """A y range fitted to the data, for a series whose signal is DRIFT.

    RSS is a level a few GB high that moves by tens of MB across a run.
    Zero-anchoring it, which is right for a share of capacity, puts that
    whole movement inside one pixel: on the three-hour capture the ranks
    sit at 1.48 to 1.50 GB on an axis that would run to 5. The leak this
    chart exists to show would be invisible.

    Returns ``(low, high, tick)``: two equal steps whose ends sit on one
    decimal grid, so the three labels step evenly and no two print alike.
    A range fitted to a steady process otherwise spanned a few kilobytes,
    and its three ticks all read "0.586 GB". ``min_span`` is the smallest
    range drawn, in the caller's unit, so a flat line sits mid-plot on
    ticks that still say something.
    """
    numbers = [float(value) for value in values if value is not None]
    numbers = [value for value in numbers if math.isfinite(value)]
    if not numbers:
        return (0.0, 1.0, 0.5)
    low, high = min(numbers), max(numbers)
    need = max(min_span, max(abs(low) * 0.01, 0.01) if high <= low else 0.0)
    if high - low < need:
        middle = (low + high) / 2.0
        low, high = middle - need / 2.0, middle + need / 2.0
    # A tenth of the range each side keeps the line off the plot's edges.
    # Snapping to the grid below widens it further, so no more is needed.
    pad = (high - low) * 0.1
    low, high = max(0.0, low - pad), high + pad

    # The step is a whole number of the half-range's leading digit, the
    # smallest that covers the data once the floor is snapped down to the
    # step's own precision. Labels printed at that precision are then
    # exact: they step evenly and no two of them are alike.
    half = (high - low) / 2.0
    grid = 10.0 ** math.floor(math.log10(half))
    count = math.ceil(round(half / grid, 9))
    while True:
        tick = round(count * grid, 12)
        digits = _decimals(tick)
        unit = 10.0**-digits
        base = round(math.floor(round(low / unit, 9)) * unit, digits)
        if base + 2 * tick >= high:
            break
        count += 1
    return (base, round(base + 2 * tick, digits), tick)


def _decimals(step: float) -> int:
    """The decimals that print ``step`` exactly (0.05 -> 2, 1.1 -> 1)."""
    if not step > 0 or not math.isfinite(step):
        return 0
    for digits in range(10):
        if abs(round(step, digits) - step) <= 1e-9 * max(1.0, step):
            return digits
    return 10


def value_axis_formatter(step: float, unit: str) -> str:
    """Tick formatter whose precision comes from the tick STEP.

    Magnitude alone is not enough: an axis fitted to a 20 MB drift around
    1.4 GB would label every tick "1.4 GB" and say nothing. The range is
    not enough either, because rounding it can still print two ticks
    alike. At the step's own precision neighbouring ticks always differ.
    """
    return f"v=>v.toFixed({_decimals(step)})+'{unit}'"


def value_axis_label(value: float, step: float, unit: str) -> str:
    """The label :func:`value_axis_formatter` writes for ``value``."""
    return f"{float(value):.{_decimals(step)}f}{unit}"


def unit_axis_formatter(unit: str) -> str:
    """Tick formatter that prints the value as it is, then the unit."""
    return f"v=>v+'{unit}'"


def unit_axis_label(value: float, unit: str) -> str:
    """The label :func:`unit_axis_formatter` writes for a round ``value``."""
    return f"{float(value):g}{unit}"


def pad_tick_labels(formatter: str, width: int) -> str:
    """Wrap a tick formatter so every label is ``width`` characters wide.

    Two charts stacked in one card fit their y labels separately, so a
    "30%" axis and a "1.111 GB" axis start their plots at different x and
    the same moment sits at two positions. Tick labels are monospace and
    right-aligned against the axis, so leading spaces move no visible
    text; they make both charts reserve the same room for their labels.
    """
    return f"v=>({formatter})(v).padStart({int(width)})"


# Seconds before the newest sample, written the way
# ``formatting.format_elapsed`` writes a duration ("58s", "2m 26s",
# "3h 12m") behind a minus sign, and "Now" at the newest sample. The card
# header prints its window with that same helper, so the leftmost tick and
# the header name the same span in the same words.
_RELATIVE = (
    "(v=>{const t=Math.round(-v);if(t<1)return 'Now';"
    "const q=n=>('0'+n).slice(-2);const h=Math.floor(t/3600),"
    "m=Math.floor((t%3600)/60),s=t%60;"
    "return '\u2212'+(h?h+'h '+q(m)+'m':(m?m+'m '+q(s)+'s':s+'s'));})"
)


# Round steps of the clock, in seconds, for the ticks between the ends.
_TICK_STEPS = (
    1,
    2,
    5,
    10,
    15,
    30,
    60,
    120,
    300,
    600,
    900,
    1800,
    3600,
    7200,
    10800,
    21600,
    43200,
    86400,
)


def relative_ticks(span: float) -> List[float]:
    """Where the relative time axis is labelled, oldest first.

    The first is the span itself, so the leftmost label names the period
    the card header names. The last is the newest sample, "Now". Between
    them sit at most two round steps of the clock ("-30s", "-1h 00m"):
    ticks at thirds of the span read "-57m 17s" on an hour axis, and on
    the first seconds of a run two thirds rounded to the same second. A
    round tick nearer the leftmost than half a step is dropped rather
    than crowding it.
    """
    span = max(float(span), 1.0)
    step = next((s for s in _TICK_STEPS if span / s <= 3.0), None)
    if step is None:
        step = math.ceil(span / 3.0 / 86400.0) * 86400
    inner = [
        -float(k * step)
        for k in range(int(span // step), 0, -1)
        if span - k * step > step / 2.0
    ]
    return [-span, *inner, 0.0]


def apply_span_axis(
    options: Dict[str, Any],
    span: float,
    newest_epoch: Optional[float] = None,
) -> None:
    """Pin a chart to its span and label it relative to the newest sample.

    The x values are seconds before the newest sample, and the ticks say
    so ("-58s, -30s, Now"): the card names its window once, in its header,
    and two stacked charts labelled in wall-clock time read as two
    different periods. A reader debugging a slowdown still needs the
    clock, because it is what their logs are keyed on, so the hover label
    carries both readings ("19:10:05 · -45s").

    The labels sit where :func:`relative_ticks` puts them. The interval is
    the whole span, so an ECharts too old to honour ``customValues``
    labels the two ends and nothing that could collide.
    """
    span = max(float(span), 1.0)
    axis = options["xAxis"]
    axis["min"] = -span
    axis["max"] = 0
    axis["interval"] = span
    label = axis["axisLabel"]
    label.update(show=True, color=_TXT, fontFamily="Geist Mono", fontSize=10)
    label["customValues"] = relative_ticks(span)
    label[":formatter"] = _RELATIVE
    if newest_epoch is None:
        return
    clock = (
        "const d=new Date((%f+p.value)*1000);const q=n=>('0'+n).slice(-2);"
        "const c=q(d.getHours())+':'+q(d.getMinutes())%s;"
        % (float(newest_epoch), "+':'+q(d.getSeconds())" if span < 600 else "")
    )
    pointer = options.get("tooltip", {}).get("axisPointer", {}).get("label")
    if pointer is not None:
        pointer[":formatter"] = (
            "p=>{" + clock + "return c+' · '+" + _RELATIVE + "(p.value);}"
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

    The span is what was observed: zero until a second moment arrives.
    :func:`apply_span_axis` gives the axis its own width floor, so a
    header printing this span never reports the drawing floor as time.
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
    return (newest, newest - min(starts))


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
        # Hidden until the card pins the span: each card's span helper
        # turns the labels on with the formatter it wants.
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
            ":formatter": unit_axis_formatter(unit),
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
    "apply_span_axis",
    "capacity_axis_max",
    "drift_axis_bounds",
    "relative_ticks",
    "shared_span",
    "pad_tick_labels",
    "sparkline_svg",
    "unit_axis_formatter",
    "unit_axis_label",
    "value_axis_formatter",
    "value_axis_label",
]
