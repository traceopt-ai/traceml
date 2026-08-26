# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Process block: the trainer process on every rank, as four tiles, two charts
and per-rank rows.

Scope, stated on the block because it is the block's most useful fact: one
process per rank, its own PID only. DataLoader workers are separate
processes, so their CPU lands in the System block and never here. Reading
the two blocks together is what the page is for: host CPU high while every
rank's process CPU is low means the work is outside the trainer.

Tiles are levels with their denominator; charts carry the two quantities
whose information is their shape over time (CPU capacity, RSS); memory is a
step function and stays a number. No verdict words: the diagnosis engine
owns those.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from . import theme
from .theme import (
    apply_span_axis,
    cpu_axis_max,
    format_gb_pair,
    format_span,
    format_window,
    num as _num,
    sparkline_svg,
)

# One colour per rank, shared by the charts and the rows' chips so a line
# and a row are recognisably the same rank. Red is not among them: it reads
# as a verdict.
_RANK_COLORS = (
    "#f97316",
    "#3b82f6",
    "#0d9488",
    "#a855f7",
    "#0ea5e9",
    "#eab308",
    "#ec4899",
    "#10b981",
)

NA = "n/a"
_MONO = "font-family:var(--mono);"

# The block's scope, on screen rather than in a tooltip: without it the
# CPU numbers read as the whole box's, and the pairing that makes them
# diagnostic is invisible.
SCOPE_NOTE = "one process per rank · DataLoader workers not included"


def rank_color(rank: int) -> str:
    return _RANK_COLORS[int(rank) % len(_RANK_COLORS)]


def should_auto_open(*, prev_over: bool, over: bool) -> bool:
    """Open on the rising edge only.

    Twin of the System block's rule: a reader who closes the rows must not
    be fought every tick while the condition persists.
    """
    return bool(over and not prev_over)


def rows_hint(roll: Dict[str, Any], *, is_open: bool) -> str:
    """Header of the per-rank rows: coverage, imbalance, and nothing else.

    It states what was observed. Whether that is bad is the engine's call,
    so no word here classifies it.
    """
    total = int(roll.get("ranks_total") or 0)
    if not total:
        return ""
    stale = int(roll.get("ranks_stale") or 0)
    parts = [f"{total} rank{'s' if total != 1 else ''}"]
    if stale:
        parts.append(f"{stale} stale, excluded")
    imbalance = roll.get("reserved_imbalance_pct")
    if imbalance is not None:
        parts.append(f"reserved imbalance {float(imbalance):.0f}%")
    parts.append("click to close" if is_open else "click to open")
    return " · ".join(parts)


def format_age(seconds: Any) -> str:
    """A per-rank age in the same words the strip uses."""
    if seconds is None:
        return NA
    value = float(seconds)
    if value < 90:
        return f"{value:.0f} s"
    return f"{value / 60.0:.0f} min"


def rows_html(
    ranks: List[Dict[str, Any]], series: List[Dict[str, Any]]
) -> str:
    """Per-rank table: identity, capacity, memory, and how fresh it is.

    A stale rank is dimmed and kept. Dropping it would hide the one fact
    worth having when a job stalls, which is WHICH rank stopped.
    """
    trend = {
        int(entry.get("global_rank", -1)): (
            entry.get("avg") or entry.get("v") or []
        )
        for entry in series or []
    }
    head = (
        "<tr><th>rank</th><th>gpu</th><th>node</th><th>cpu cap</th>"
        "<th>rss</th><th>cpu trend</th><th>cuda allocated</th>"
        "<th>cuda reserved</th><th>age</th></tr>"
    )
    body = ""
    for rank in ranks:
        idx = int(rank.get("global_rank", 0))
        colour = rank_color(idx)
        used, rest = format_gb_pair(
            rank.get("ram_used"), rank.get("ram_total")
        )
        reserved, reserved_rest = format_gb_pair(
            rank.get("gpu_reserved"), rank.get("gpu_total")
        )
        alloc, alloc_rest = format_gb_pair(rank.get("gpu_alloc"), None)
        cap = rank.get("cpu_capacity")
        stale = bool(rank.get("stale"))
        row_open = '<tr class="tml-stale">' if stale else "<tr>"
        gpu_index = rank.get("gpu_index")
        node_rank = rank.get("node_rank")
        gpu_cell = f"G{int(gpu_index)}" if gpu_index is not None else NA
        node_cell = f"N{int(node_rank)}" if node_rank is not None else NA
        cap_cell = f"{_num(cap, '{:.1f}')} %" if cap is not None else NA
        body += (
            row_open + f'<td><span style="color:{colour}">■</span> R{idx}</td>'
            f"<td>{gpu_cell}</td>"
            f"<td>{node_cell}</td>"
            f"<td>{cap_cell}</td>"
            f"<td>{used} {rest}</td>"
            f"<td>{sparkline_svg(trend.get(idx, []), colour)}</td>"
            f"<td>{alloc} {alloc_rest}</td>"
            f"<td>{reserved} {reserved_rest}</td>"
            f"<td>{format_age(rank.get('age_s'))}</td>"
            "</tr>"
        )
    return f'<table class="tml-gpus">{head}{body}</table>'


def drift_axis_bounds(values: List[Any]) -> Tuple[float, float, float]:
    """A y range that fits the data, for a series whose signal is DRIFT.

    RSS is a level a few GB high that moves by tens of MB across a run.
    Zero-anchoring it (right for a share of capacity, like CPU) puts that
    whole movement inside one pixel: measured on the 3-hour capture, the
    ranks sit at 1.48-1.50 GB on an axis that would run to 5. The leak
    this chart exists to show would be invisible.
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


def _series_axis(
    *groups: List[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """One anchor and span covering every series on the block.

    Both charts are pinned to it so a vertical read across the pair, and
    across to the System block's charts above, lands on the same moment.
    """
    starts, ends = [], []
    for group in groups:
        for entry in group or []:
            stamps = entry.get("t") or []
            if stamps:
                starts.append(stamps[0])
                ends.append(stamps[-1])
    if not ends:
        return None
    newest = max(ends)
    return (newest, max(newest - min(starts), 1.0))


def build_process_section() -> Dict[str, Any]:
    """Build the block once; ``update_process_section`` fills it per tick."""
    panel: Dict[str, Any] = {"tiles": {}, "subs": {}}
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; display:flex; "
        "flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:10px; gap:12px;")
        ):
            ui.label("Process").classes("ctitle")
            ui.element("div").style("flex:1;")
            panel["note"] = ui.label(SCOPE_NOTE).classes("cmeta")

        with ui.element("div").classes("tilerow").style("margin-bottom:10px;"):
            for key, label, accent in (
                ("cpu", "cpu capacity", theme.C_CPU),
                ("rss", "rss", theme.C_CPU),
                ("reserved", "cuda reserved", theme.C_GPU),
                ("alloc", "cuda allocated", theme.C_GPU),
            ):
                with (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{accent}; min-width:0;")
                ):
                    ui.label(label).classes("klab")
                    panel["tiles"][key] = ui.html(NA, sanitize=False).classes(
                        "kval"
                    )
                    panel["subs"][key] = ui.label("").classes("ksub")

        with (
            ui.row()
            .classes("w-full items-baseline")
            .style("gap:8px; margin:2px 0 2px;")
        ):
            panel["cpu_label"] = ui.label("process cpu").classes("estlabel")
            ui.element("div").style("flex:1;")
            panel["cpu_value"] = ui.label("").style(
                f"{_MONO} font-size:14px; font-weight:600;"
            )
        panel["cpu_chart"] = ui.echart(theme.multi_line_options("%")).style(
            "height:92px; width:100%;"
        )

        with (
            ui.row()
            .classes("w-full items-baseline")
            .style("gap:8px; margin:8px 0 2px;")
        ):
            panel["rss_label"] = ui.label("rss").classes("estlabel")
            ui.element("div").style("flex:1;")
            panel["rss_value"] = ui.label("").style(
                f"{_MONO} font-size:14px; font-weight:600;"
            )
        panel["rss_chart"] = ui.echart(theme.multi_line_options(" GB")).style(
            "height:92px; width:100%;"
        )

        exp = (
            ui.expansion()
            .classes("w-full tml-exp")
            .props("dense dense-toggle expand-icon-toggle")
            .style("margin-top:6px;")
        )
        with exp.add_slot("header"):
            with (
                ui.row()
                .classes("w-full items-center")
                .style("gap:10px; min-width:0;")
            ):
                ui.label("per-rank rows").style(
                    f"{_MONO} font-size:12px; font-weight:700;"
                )
                panel["rows_hint"] = ui.label("").classes("cmeta")
        with exp:
            panel["rows_html"] = ui.html("", sanitize=False).classes("w-full")
        panel["rows"] = exp
        panel["_over"] = False
        panel["_sig"] = None
    panel["card"] = card
    return panel


def _chart_series(
    entries: List[Dict[str, Any]], anchor: float, key: str, scale: float
) -> List[Dict[str, Any]]:
    """One line per rank, x in seconds before the shared anchor."""
    lines = []
    for entry in entries or []:
        idx = int(entry.get("global_rank", 0))
        stamps = entry.get("t") or []
        values = entry.get(key) or []
        lines.append(
            theme.line_series(
                f"R{idx}",
                rank_color(idx),
                [
                    [stamp - anchor, value / scale]
                    for stamp, value in zip(stamps, values)
                ],
            )
        )
    return lines


def _update_chart(
    panel: Dict[str, Any],
    *,
    chart_key: str,
    label_key: str,
    head: str,
    run: List[Dict[str, Any]],
    window: List[Dict[str, Any]],
    aligned: Optional[Tuple[float, float]],
    scale: float,
    ymax_of: Any,
    tooltip: str,
    unit: str = "",
) -> None:
    """Draw whichever view the payload carried, and name it."""
    chart = panel[chart_key]
    entries = run or window
    if not entries or aligned is None:
        chart.options["series"] = []
        chart.update()
        panel[label_key].text = head
        return
    anchor, span = aligned
    values_key = "avg" if run else "v"
    chart.options["series"] = _chart_series(entries, anchor, values_key, scale)
    apply_span_axis(chart.options, span, anchor)
    flat = [
        value / scale
        for entry in entries
        for value in (entry.get("max") or entry.get(values_key) or [])
    ]
    bounds = ymax_of(flat)
    if isinstance(bounds, tuple):
        low, high, tick = bounds
        chart.options["yAxis"]["min"] = low
        chart.options["yAxis"]["max"] = high
        chart.options["yAxis"]["interval"] = tick
        chart.options["yAxis"]["axisLabel"][":formatter"] = (
            theme.value_axis_formatter(high - low, unit)
        )
    else:
        chart.options["yAxis"]["max"] = bounds
    chart.update()
    window_s = float((run[0].get("window_s") if run else 0.0) or 0.0)
    words = format_window(window_s)
    panel[label_key].text = f"{head} · {format_span(span)}" + (
        f" · rolling {words}" if run and words else ""
    )
    panel[label_key].tooltip(tooltip)


def update_process_section(
    panel: Dict[str, Any], data: Dict[str, Any]
) -> None:
    """Fill the block from one PROCESS payload."""
    if not isinstance(data, dict):
        return
    roll = data.get("rollups", {}) or {}
    series = data.get("series", {}) or {}
    ranks = roll.get("ranks", []) or []

    # The charts are re-sent only when the newest point moved: the UI timer
    # ticks faster than telemetry arrives, and a full options dict per tick
    # is pure websocket traffic.
    signature = (
        data.get("window_len"),
        len(ranks),
        roll.get("ranks_stale"),
        tuple(
            (entry.get("t") or [None])[-1]
            for entry in series.get("cpu_capacity_run")
            or series.get("cpu_capacity")
            or []
        ),
    )
    changed = signature != panel.get("_sig")
    panel["_sig"] = signature

    cpu = roll.get("cpu_capacity", {}) or {}
    rss = roll.get("rss", {}) or {}
    cuda = roll.get("cuda", {}) or {}
    has_gpu = bool(data.get("gpu_available") or roll.get("gpu_available"))

    p50 = cpu.get("p50")
    panel["tiles"]["cpu"].content = theme.kval(
        _num(p50, "{:.1f}") if p50 is not None else NA,
        "%" if p50 is not None else "",
    )
    panel["subs"]["cpu"].text = "median rank · of host capacity"

    used, rest = format_gb_pair(rss.get("used"), rss.get("total"))
    panel["tiles"]["rss"].content = theme.kval(
        used, f" {rest}" if rest else ""
    )
    worst_rank = rss.get("rank")
    panel["subs"]["rss"].text = (
        f"worst rank · R{int(worst_rank)}"
        if worst_rank is not None
        else "used / total"
    )

    if has_gpu:
        reserved, reserved_rest = format_gb_pair(
            cuda.get("reserved"), cuda.get("reserved_total")
        )
        panel["tiles"]["reserved"].content = theme.kval(
            reserved, f" {reserved_rest}" if reserved_rest else ""
        )
        tight = cuda.get("reserved_rank")
        panel["subs"]["reserved"].text = (
            f"least-headroom rank · R{int(tight)}"
            if tight is not None
            else "reserved / total"
        )
        alloc, alloc_rest = format_gb_pair(cuda.get("alloc_p50"), None)
        panel["tiles"]["alloc"].content = theme.kval(
            alloc, f" {alloc_rest}" if alloc_rest else ""
        )
        panel["subs"]["alloc"].text = "median rank · live tensors"
    else:
        seen = bool(data.get("window_len"))
        for key in ("reserved", "alloc"):
            panel["tiles"][key].content = NA
            panel["subs"][key].text = "no GPU" if seen else ""

    aligned = _series_axis(
        series.get("cpu_capacity_run") or series.get("cpu_capacity"),
        series.get("rss_run") or series.get("rss"),
    )
    if changed:
        _update_chart(
            panel,
            chart_key="cpu_chart",
            label_key="cpu_label",
            head="process cpu · capacity per rank",
            run=series.get("cpu_capacity_run") or [],
            window=series.get("cpu_capacity") or [],
            aligned=aligned,
            scale=1.0,
            ymax_of=cpu_axis_max,
            tooltip=(
                "Each rank's trainer process, as a share of the host's "
                "total CPU capacity: 100% means every logical core busy. "
                "DataLoader workers are separate processes and are not "
                "included, so host CPU high with these low means the work "
                "is outside the trainer. On a hyperthreaded host, logical "
                "capacity is reached before the cores saturate."
            ),
        )
        _update_chart(
            panel,
            chart_key="rss_chart",
            label_key="rss_label",
            head="rss · per rank",
            run=series.get("rss_run") or [],
            window=series.get("rss") or [],
            aligned=aligned,
            scale=float(1024**3),
            ymax_of=drift_axis_bounds,
            unit=" GB",
            tooltip=(
                "Host memory held by each rank's trainer process. The "
                "shape is the point: steady growth across the run is the "
                "signature of a leak, which ends with the OS killing one "
                "rank. It covers the history retained for this run, not "
                "necessarily the whole run."
            ),
        )
    panel["cpu_value"].text = f"{float(p50):.1f}%" if p50 is not None else ""
    worst_used, worst_rest = format_gb_pair(rss.get("used"), None)
    panel["rss_value"].text = (
        f"{worst_used} {worst_rest}".strip() if rss.get("used") else ""
    )

    over = bool(roll.get("rows_over"))
    if should_auto_open(prev_over=bool(panel.get("_over")), over=over):
        panel["rows"].value = True
    panel["_over"] = over
    panel["rows_hint"].text = rows_hint(
        roll, is_open=bool(panel["rows"].value)
    )
    panel["rows_html"].content = rows_html(
        ranks, series.get("cpu_capacity_run") or series.get("cpu_capacity")
    )
