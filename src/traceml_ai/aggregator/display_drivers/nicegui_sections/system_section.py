# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
System block: the host and its GPUs, as four tiles, two charts and rows.

Tiles (one plain number each): GPU utilisation = across-GPU average, a
short-window median; GPU memory = the max-used GPU against its capacity;
GPU temperature = the hottest GPU; host RAM = used against total. Levels
always carry their denominator, rates carry the estimator in words.

Charts: GPU power per GPU against the board limit (power is the one system
metric that moves on every run), and host CPU as a small zero-anchored
series (its shape drifts over long runs). Both label their gridlines and
their window span, and neither narrates its mechanism.

Rows: one line per GPU behind a disclosure that opens on its own when the
across-GPU utilisation spread crosses ``SPREAD_EXPAND_PTS``. The average
describes the node; the rows are what make it honest when one GPU of four
is busy, so the two are one design, not two.

No verdict words anywhere: judgement comes from the diagnosis engine or it
does not appear.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from . import theme

# Across-GPU utilisation spread (max - min, percentage points, window p95)
# above which the per-GPU rows open on their own. Measured separation on
# the reference runs is 0 (all busy) vs 100 (one busy of four).
SPREAD_EXPAND_PTS = 20.0

_GPU_COLORS = (
    "#f97316",
    "#3b82f6",
    "#10b981",
    "#a855f7",
    "#0d9488",
    "#0ea5e9",
    "#eab308",
    "#ec4899",
)
_LIMIT_RED = "#dc2626"
_MONO = "font-family:var(--mono);"


# --- pure display rules ----------------------------------------------------
def gpu_color(index: int) -> str:
    return _GPU_COLORS[int(index) % len(_GPU_COLORS)]


def format_span(seconds: Optional[float]) -> str:
    """The window a chart covers, as one phrase in its label: 'last 3 min'."""
    if not seconds or seconds <= 0:
        return ""
    if seconds < 90:
        return f"last {seconds:.0f} s"
    return f"last {seconds / 60.0:.0f} min"


# Per-GPU utilisation below this reads as idle in the rows' header.
IDLE_UTIL_PCT = 20.0


def format_gb_pair(used_bytes: Any, total_bytes: Any) -> Tuple[str, str]:
    """A level against its capacity: ('6.3', '/ 16.1 GB')."""
    used = theme.gb(used_bytes) if used_bytes is not None else None
    if used is None:
        return (NA, "")
    num = f"{used:.1f}"
    total = theme.gb(total_bytes) if total_bytes is not None else None
    if total is None or total <= 0:
        return (num, "GB")
    total_s = f"{total:.0f}" if total >= 100 else f"{total:.1f}"
    return (num, f"/ {total_s} GB")


def disclosure_text(
    gpus: List[Dict[str, Any]], *, over: bool, is_open: bool
) -> str:
    """Header of the per-GPU rows, in GPU words: what the user needs to know.

    ``over`` is the trigger's verdict (the across-GPU spread crossed the
    bar); the words say what that means on this node, never how it was
    computed. The tail follows the rows' real state, which a click may have
    changed: the rows open on their own only on a rising edge.
    """
    utils = []
    for g in gpus:
        u = g.get("util_p50")
        if u is None:
            u = g.get("util_now")
        if u is not None:
            utils.append(float(u))
    n = len(utils)
    if n == 0:
        return ""
    tail = " · click to close" if is_open else " · click to open"
    if n == 1:
        return "1 GPU" + tail
    if over:
        idle = sum(1 for u in utils if u < IDLE_UTIL_PCT)
        if idle:
            head = f"{n - idle} of {n} GPUs busy, {idle} idle"
        else:
            head = "uneven load across GPUs"
    else:
        head = f"all {n} GPUs alike"
    return head + tail


def node_scope_text(ctx: Dict[str, Any]) -> str:
    """'node 0 of 2' when the payload dropped other machines, else ''.

    Reads only the System payload's own ``system_node`` (the hosts seen in
    this window); the strip's node count lives on the CONTEXT payload and
    is not this block's to read.
    """
    node = ctx.get("system_node") if isinstance(ctx, dict) else None
    if not isinstance(node, dict):
        return ""
    in_window = int(node.get("nodes_in_window") or 1)
    if in_window <= 1:
        return ""
    rank = node.get("node_rank")
    label = f"node {rank}" if rank is not None else node.get("hostname", "")
    return f"{label} of {in_window}"


def should_auto_open(*, prev_over: bool, over: bool) -> bool:
    """Open on the rising edge only: never fight a user who closed it."""
    return bool(over and not prev_over)


def power_axis_bounds(
    values: List[Optional[float]], limit: Optional[float]
) -> Tuple[float, float, float]:
    """(min, max, tick) in round watts that keep data and limit in frame.

    At most four intervals of a round size (10 / 20 / 25 / 50 W), the
    bounds multiples of it, so labels read 40 / 60 / 80 / 100 / 120 and
    never 20 / 50 / 110.
    """
    vals = [float(v) for v in values if v is not None]
    if limit is not None:
        vals.append(float(limit))
    if not vals:
        return (0.0, 100.0, 50.0)
    low = max(0.0, min(vals) - 5.0)
    high = max(vals) + 5.0
    for tick in (10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0):
        lo = math.floor(low / tick) * tick
        hi = math.ceil(high / tick) * tick
        if hi <= lo:
            hi = lo + tick
        if (hi - lo) / tick <= 4:
            return (lo, hi, tick)
    return (lo, hi, tick)


def cpu_axis_max(values: List[Any]) -> float:
    """Zero-anchored ceiling whose half is a whole percent (0 / 5 / 10)."""
    vals = [float(v) for v in values if v is not None]
    peak = (max(vals) * 1.2) if vals else 0.0
    for top in (4.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0):
        if peak <= top:
            return top
    return 100.0


def sparkline_svg(
    values: List[Optional[float]],
    color: str,
    *,
    width: int = 64,
    height: int = 14,
) -> str:
    """Inline SVG polyline; gaps are dropped, an empty trace is no SVG."""
    points = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if not points:
        return ""
    lo = min(v for _, v in points)
    hi = max(v for _, v in points)
    span = (hi - lo) or 1.0
    step = width / max(len(values) - 1, 1)
    inner = height - 4
    coords = " ".join(
        f"{i * step:.1f},{1 + inner - (v - lo) / span * inner:.1f}"
        for i, v in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'style="width:{width}px;height:{height}px;vertical-align:middle">'
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        'stroke-width="1.4"/></svg>'
    )


def odd_ones_out(gpus: List[Dict[str, Any]]) -> set:
    """GPU indices on the smaller side of the utilisation split.

    Split at the midpoint between the highest and lowest per-GPU window
    median; the smaller group is the anomaly (ties go to the busy side,
    which keeps the 1-busy-of-N reference shape tinting the busy GPU).
    """
    utils = []
    for g in gpus:
        u = g.get("util_p50")
        if u is None:
            u = g.get("util_now")
        if u is not None:
            utils.append((int(g.get("gpu_idx", 0)), float(u)))
    if len(utils) < 2:
        return set()
    lo = min(u for _, u in utils)
    hi = max(u for _, u in utils)
    if hi <= lo:
        return set()
    mid = (hi + lo) / 2.0
    high = {i for i, u in utils if u > mid}
    low = {i for i, u in utils if u <= mid}
    return high if len(high) <= len(low) else low


# Absent value marker. "n/a" rather than a dash: the Process card on the
# same page already reads "N/A", and one page should not spell absence two
# ways.
NA = "n/a"


def _num(value: Any, fmt: str = "{:.0f}") -> str:
    return fmt.format(float(value)) if value is not None else NA


def rows_html(
    gpus: List[Dict[str, Any]],
    power_series: List[Dict[str, Any]],
    *,
    spread: Optional[float],
) -> str:
    """Per-GPU table: util (window median), power trend, mem, temp, W/limit.

    When the spread is over the bar, the rows on the minority side of it
    are tinted so the eye lands on the odd ones out: the one busy GPU of
    four, or the one idle GPU of four. Absent values read as a dash,
    never as a zero.
    """
    series_by_idx = {
        int(p.get("gpu_idx", -1)): p.get("values") or []
        for p in power_series or []
    }
    over = spread is not None and spread > SPREAD_EXPAND_PTS
    marked = odd_ones_out(gpus) if over else set()
    head = (
        "<tr><th>gpu</th><th>util</th><th>power trend</th>"
        "<th>mem GB</th><th>temp °C</th><th>W / limit</th></tr>"
    )
    rows = []
    for g in gpus:
        idx = int(g.get("gpu_idx", 0))
        color = gpu_color(idx)
        used, total = g.get("mem_used"), g.get("mem_total")
        if used is not None and total:
            mem = f"{theme.gb(used):.2f} / {theme.gb(total):.1f}"
        elif used is not None:
            mem = f"{theme.gb(used):.2f}"
        else:
            mem = NA
        power, limit = g.get("power"), g.get("power_limit")
        if power is not None and limit:
            watts = f"{power:.0f} / {limit:.0f}"
        elif power is not None:
            watts = f"{power:.0f} W"
        else:
            watts = NA
        util = g.get("util_p50")
        if util is None:
            util = g.get("util_now")
        cls = ' class="tml-mark"' if idx in marked else ""
        rows.append(
            f"<tr{cls}>"
            f'<td><span style="color:{color}">■</span> gpu{idx}</td>'
            f'<td class="tml-util">{_num(util)}</td>'
            f"<td>{sparkline_svg(series_by_idx.get(idx, []), color)}</td>"
            f"<td>{mem}</td><td>{_num(g.get('temp'))}</td><td>{watts}</td>"
            "</tr>"
        )
    return f'<table class="tml-gpus">{head}{"".join(rows)}</table>'


# --- relative time axis ----------------------------------------------------
def _relative_seconds(x_time: List[str]) -> List[Optional[float]]:
    """Seconds before the newest sample (<= 0), aligned with ``x_time``."""
    stamps: List[Optional[float]] = []
    for s in x_time:
        try:
            stamps.append(datetime.fromisoformat(s).timestamp())
        except Exception:
            stamps.append(None)
    present = [t for t in stamps if t is not None]
    if not present:
        return []
    last = max(present)
    return [t - last if t is not None else None for t in stamps]


def _apply_span_axis(axis: Dict[str, Any], span: float) -> None:
    """Pin the axis to the window; the span itself is said in the label
    row ('last 3 min'), not as axis furniture."""
    span = max(float(span), 1.0)
    axis["min"] = -span
    axis["max"] = 0
    axis["interval"] = span
    axis["axisLabel"]["show"] = False


# --- build / update --------------------------------------------------------
def build_system_section() -> Dict[str, Any]:
    panel: Dict[str, Any] = {
        "tiles": {},
        "subs": {},
        "tile_els": {},
        "gpu_visible": True,
        "_over": False,
        "_sig": None,
    }
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:10px; gap:12px;")
        ):
            ui.label("System").classes("ctitle")
            ui.element("div").style("flex:1;")
            panel["note"] = ui.label("waiting for data").classes("cmeta")

        with (
            ui.row()
            .classes("w-full")
            .style("gap:9px; flex-wrap:wrap; margin-bottom:10px;")
        ):
            for key, label, acc in (
                ("util", "gpu util", theme.C_GPU),
                ("mem", "gpu mem", theme.C_GPU),
                ("temp", "gpu temp", theme.C_GPU),
                ("ram", "host ram", theme.C_CPU),
            ):
                tile = (
                    ui.element("div")
                    .classes("kpi")
                    .style(
                        f"--acc:{acc}; flex:1 1 0; min-width:112px; "
                        "max-width:320px;"
                    )
                )
                with tile:
                    ui.label(label).classes("klab")
                    panel["tiles"][key] = ui.html(NA, sanitize=False).classes(
                        "kval"
                    )
                    panel["subs"][key] = ui.label("").classes("ksub")
                panel["tile_els"][key] = tile

        with (
            ui.row()
            .classes("w-full items-baseline")
            .style("gap:8px; margin:2px 0 2px;")
        ):
            panel["cpu_label"] = ui.label("host cpu %").classes("estlabel")
            ui.element("div").style("flex:1;")
            panel["cpu_value"] = ui.label("").style(
                f"{_MONO} font-size:13px; font-weight:600; color:var(--ink);"
            )
        panel["cpu_chart"] = ui.echart(
            theme.span_line_options(theme.C_CPU, "%")
        ).style("height:92px; width:100%;")

        panel["power_head"] = (
            ui.row()
            .classes("w-full items-baseline")
            .style("gap:8px; margin:8px 0 2px;")
        )
        with panel["power_head"]:
            panel["power_label"] = ui.label("gpu power · per GPU").classes(
                "estlabel"
            )
        panel["power_chart"] = ui.echart(theme.multi_line_options(" W")).style(
            "height:150px; width:100%;"
        )
        # Same slot, same height, when there is no GPU: the card keeps its
        # shape on every host instead of shrinking around a missing chart.
        panel["power_placeholder"] = ui.label("no GPU reported").style(
            f"{_MONO} font-size:11px; color:var(--muted); height:150px; "
            "width:100%; display:none; align-items:center; "
            "justify-content:center;"
        )

        exp = (
            ui.expansion()
            .classes("w-full tml-exp")
            .props("dense expand-icon-class=text-grey-6")
            .style("margin-top:6px;")
        )
        with exp.add_slot("header"):
            with (
                ui.row()
                .classes("w-full items-center")
                .style("gap:10px; min-width:0;")
            ):
                ui.label("per-GPU rows").style(
                    f"{_MONO} font-size:11px; font-weight:600; "
                    "color:var(--ink);"
                )
                panel["rows_hint"] = ui.label("").classes("cmeta")
        with exp:
            panel["rows_html"] = ui.html("", sanitize=False).classes("w-full")
        panel["rows"] = exp
        panel["rows_placeholder"] = ui.label("per-GPU rows · no GPU").style(
            f"{_MONO} font-size:11px; color:var(--muted); margin-top:6px; "
            "padding:4px 2px; display:none;"
        )
    return panel


def _set_gpu_visible(panel: Dict[str, Any], visible: bool) -> None:
    if panel.get("gpu_visible") == visible:
        return
    panel["gpu_visible"] = visible
    # Every slot keeps its place and height on every host (one shape to
    # learn); the GPU slots show their one-line state instead of vanishing.
    disp = "block" if visible else "none"
    panel["power_chart"].style(f"display:{disp};")
    panel["power_placeholder"].style(
        f"display:{'none' if visible else 'flex'};"
    )
    panel["rows"].style(f"display:{disp};")
    panel["rows_placeholder"].style(
        f"display:{'none' if visible else 'block'};"
    )


def update_system_section(panel: Dict[str, Any], data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    roll = data.get("rollups", {}) or {}
    series = data.get("series", {}) or {}
    gpu_on = bool(data.get("gpu_available") or roll.get("gpu_available"))
    _set_gpu_visible(panel, gpu_on)

    x_time = series.get("x_time", []) or []
    secs = _relative_seconds(x_time)
    present = [s for s in secs if s is not None]
    span = -min(present) if present else 0.0

    # The UI timer runs faster than the data arrives; re-sending a full
    # chart options dict for an unchanged window is pure websocket
    # traffic, so the charts only update when the window moved.
    gp = roll.get("gpu_power", {}) or {}
    limit = gp.get("limit")
    pseries = series.get("gpu_power", []) or []
    sig = (
        x_time[-1] if x_time else None,
        data.get("window_len"),
        len(pseries),
        limit,
        gpu_on,
    )
    changed = sig != panel.get("_sig")
    panel["_sig"] = sig

    # Host CPU: a small zero-anchored series, the window median as its
    # one number.
    cpu = series.get("cpu", []) or []
    chart = panel["cpu_chart"]
    if changed:
        chart.options["series"][0]["data"] = [
            [s, v] for s, v in zip(secs, cpu) if s is not None
        ]
        _apply_span_axis(chart.options["xAxis"], span)
        ymax = cpu_axis_max(cpu)
        chart.options["yAxis"]["max"] = ymax
        chart.options["yAxis"]["interval"] = ymax / 2.0
        chart.update()
    cpu_p50 = (roll.get("cpu", {}) or {}).get("p50")
    panel["cpu_value"].text = f"{cpu_p50:.0f}%" if cpu_p50 is not None else ""

    # Host RAM: a level against its capacity.
    ram = roll.get("ram", {}) or {}
    num, rest = format_gb_pair(ram.get("now"), ram.get("total"))
    panel["tiles"]["ram"].content = theme.kval(num, f" {rest}" if rest else "")
    panel["subs"]["ram"].text = "used / total"

    notes = []
    node_note = node_scope_text(roll.get("ctx", {}) or {})
    if node_note:
        notes.append(node_note)
    if not data.get("window_len"):
        notes.append("waiting for data")
    panel["note"].text = " · ".join(notes)
    span_words = format_span(span)
    panel["cpu_label"].text = (
        f"host cpu % · {span_words}" if span_words else "host cpu %"
    )
    if not gpu_on:
        seen = bool(data.get("window_len"))
        for key in ("util", "mem", "temp"):
            panel["tiles"][key].content = NA
            panel["subs"][key].text = "no GPU" if seen else ""
        panel["power_label"].text = "gpu power"
        panel["power_placeholder"].text = (
            "no GPU reported" if seen else "waiting for data"
        )
        panel["rows_placeholder"].text = (
            "per-GPU rows · no GPU" if seen else "per-GPU rows"
        )
        return

    gpus = roll.get("gpus", []) or []
    n_gpus = len(gpus) or int((roll.get("ctx", {}) or {}).get("gpu_count", 0))
    # Every GPU unreported on the newest tick (the sampler's all-zero
    # fallback): the level tiles say so instead of printing zeros.
    unreported = bool(gpus) and all(
        g.get("mem_total") is None and g.get("power") is None for g in gpus
    )

    util_p50 = (roll.get("gpu_util", {}) or {}).get("p50")
    panel["tiles"]["util"].content = theme.kval(_num(util_p50), "%")
    panel["subs"]["util"].text = (
        f"avg of {n_gpus} GPUs" if n_gpus != 1 else "1 GPU"
    )

    gm = roll.get("gpu_mem", {}) or {}
    if unreported:
        panel["tiles"]["mem"].content = NA
        panel["subs"]["mem"].text = "GPU sample unreported"
        panel["tiles"]["temp"].content = NA
        panel["subs"]["temp"].text = "GPU sample unreported"
    else:
        num, rest = format_gb_pair(gm.get("now"), gm.get("total"))
        panel["tiles"]["mem"].content = theme.kval(
            num, f" {rest}" if rest else ""
        )
        panel["subs"]["mem"].text = (
            "max GPU" if n_gpus != 1 else "used / total"
        )
        temp_now = (roll.get("temp", {}) or {}).get("now")
        panel["tiles"]["temp"].content = theme.kval(_num(temp_now), " °C")
        panel["subs"]["temp"].text = "max GPU" if n_gpus != 1 else ""

    # GPU power per GPU against the board limit.
    flat = [v for p in pseries for v in (p.get("values") or [])]
    pchart = panel["power_chart"]
    if not any(v is not None for v in flat):
        pchart.style("display:none;")
        panel["power_label"].text = "gpu power · not reported"
    elif changed:
        lines = []
        for p in pseries:
            idx = int(p.get("gpu_idx", 0))
            lines.append(
                theme.line_series(
                    f"gpu{idx}",
                    gpu_color(idx),
                    [
                        [s, v]
                        for s, v in zip(secs, p.get("values") or [])
                        if s is not None
                    ],
                )
            )
        if lines:
            # An explicit empty markLine clears a previous limit line:
            # ECharts merges options, so omitting the key would keep it.
            lines[0]["markLine"] = (
                theme.limit_mark_line(
                    float(limit), f"{float(limit):.0f} W limit", _LIMIT_RED
                )
                if limit is not None
                else {"data": []}
            )
        pchart.options["series"] = lines
        lo, hi, tick = power_axis_bounds(flat, limit)
        pchart.options["yAxis"]["min"] = lo
        pchart.options["yAxis"]["max"] = hi
        pchart.options["yAxis"]["interval"] = tick
        _apply_span_axis(pchart.options["xAxis"], span)
        pchart.style("display:block;")
        pchart.update()
        head = (
            f"gpu power · per GPU vs {float(limit):.0f} W limit"
            if limit is not None
            else "gpu power · per GPU"
        )
        panel["power_label"].text = (
            f"{head} · {span_words}" if span_words else head
        )

    # Per-GPU rows, opened by the spread on its rising edge; the hint
    # follows the rows' real state, which a click may have changed.
    spread = (roll.get("gpu_delta", {}) or {}).get("p95")
    over = spread is not None and float(spread) > SPREAD_EXPAND_PTS
    if should_auto_open(prev_over=panel["_over"], over=over):
        panel["rows"].value = True
    panel["_over"] = over
    panel["rows_hint"].text = disclosure_text(
        gpus, over=over, is_open=bool(panel["rows"].value)
    )
    panel["rows_html"].content = rows_html(gpus, pseries, spread=spread)
