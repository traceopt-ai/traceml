# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process block: the trainer process on every rank.

Four tiles, two per-rank charts and a per-rank table. Scope is stated on
the block because it is the block's most useful fact: one process per rank,
its own PID only. DataLoader workers are separate processes, so their CPU
lands in the System block and never here. Reading the two together is what
the page is for: host CPU high while every rank's process CPU is low means
the work is outside the trainer.

Presentation only. Every number arrives on a ``ProcessDashboardPayload``
already decided: which rank is worst, what the spread is, which history a
chart is made of, and whether the rows have earned opening. This module
chooses layout, colour, units and wording, and nothing else. In particular
it never compares a value to a threshold, because that is a severity
judgement and the diagnosis engine owns those.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from nicegui import ui

from traceml_ai.renderers.process.dashboard_models import (
    MetricRollup,
    ProcessDashboardPayload,
    RankChart,
    RankSnapshot,
)

from . import charting, theme
from .formatting import (
    NA,
    format_age,
    format_gb_pair,
    format_percent,
    format_span,
    format_window,
    num,
)

_MONO = "font-family:var(--mono);"

# The block's scope, on screen rather than in a tooltip: without it the CPU
# numbers read as the whole box's, and the pairing that makes them
# diagnostic is invisible.
SCOPE_NOTE = "one process per rank · DataLoader workers not included"

# The span the tiles summarise, stated on the card. It is a duration
# rather than a sample count so it means the same thing at every
# sampling rate.
SECTION_TITLE = "Process resources · recent 60s"

_CPU_TOOLTIP = (
    "Each rank's trainer process, as a share of the host's total CPU "
    "capacity: 100% means every logical core busy. DataLoader workers are "
    "separate processes and are not included, so host CPU high with these "
    "low means the work is outside the trainer. On a hyperthreaded host, "
    "logical capacity is reached before the cores saturate."
)
_RSS_TOOLTIP = (
    "Host memory held by each rank's trainer process. The shape is the "
    "point: steady growth across the run is the signature of a leak, which "
    "ends with the OS killing one rank. It covers the history retained for "
    "this run, not necessarily the whole run."
)


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
            ui.label(SECTION_TITLE).classes("ctitle")
            ui.element("div").style("flex:1;")
            panel["note"] = ui.label(SCOPE_NOTE).classes("cmeta")

        with ui.element("div").classes("tilerow").style("margin-bottom:10px;"):
            for key, label, accent in (
                ("cpu", "cpu", theme.C_CPU),
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

        for chart_key, label_key, value_key, head, unit, margin in (
            ("cpu_chart", "cpu_label", "cpu_value", "process cpu", "%", "2px"),
            ("rss_chart", "rss_label", "rss_value", "rss", " GB", "8px"),
        ):
            with (
                ui.row()
                .classes("w-full items-baseline")
                .style(f"gap:8px; margin:{margin} 0 2px;")
            ):
                panel[label_key] = ui.label(head).classes("estlabel")
                ui.element("div").style("flex:1;")
                panel[value_key] = ui.label("").style(
                    f"{_MONO} font-size:14px; font-weight:600;"
                )
            panel[chart_key] = ui.echart(
                charting.multi_line_options(unit)
            ).style("height:92px; width:100%;")

        expansion = (
            ui.expansion()
            .classes("w-full tml-exp")
            .props("dense dense-toggle expand-icon-toggle")
            .style("margin-top:6px;")
        )
        with expansion.add_slot("header"):
            with (
                ui.row()
                .classes("w-full items-center")
                .style("gap:10px; min-width:0;")
            ):
                ui.label("per-rank rows").style(
                    f"{_MONO} font-size:12px; font-weight:700;"
                )
                panel["rows_hint"] = ui.label("").classes("cmeta")
        with expansion:
            panel["rows_html"] = ui.html("", sanitize=False).classes("w-full")
        panel["rows"] = expansion
        panel["_was_open"] = False
        panel["_signature"] = None
    panel["card"] = card
    return panel


def should_auto_open(*, prev_over: bool, over: bool) -> bool:
    """Open on the rising edge only.

    A reader who closes the rows must not be fought every tick while the
    condition that opened them persists.
    """
    return bool(over and not prev_over)


def rows_hint(payload: ProcessDashboardPayload, *, is_open: bool) -> str:
    """Header of the per-rank rows: coverage, spread, and nothing else.

    It states what was observed. Whether that is bad is the engine's call,
    so no word here classifies it.
    """
    coverage = payload.coverage
    if not coverage.total:
        return ""
    parts = [f"{coverage.total} rank{'s' if coverage.total != 1 else ''}"]
    if coverage.stale and coverage.excluding_stale:
        parts.append(f"{coverage.stale} stale, excluded")
    elif coverage.stale:
        # Nothing is reporting, so nothing was excluded: the numbers above
        # are the last ones these ranks sent.
        parts.append("none reporting")
    if coverage.unknown:
        parts.append(f"{coverage.unknown} without a clock")
    if payload.reserved_imbalance_percent is not None:
        shown = format_percent(payload.reserved_imbalance_percent)
        parts.append(f"reserved imbalance {shown}%")
    parts.append("click to close" if is_open else "click to open")
    return " · ".join(parts)


def rows_html(
    ranks: Sequence[RankSnapshot], chart: Optional[RankChart]
) -> str:
    """Per-rank table: identity, capacity, memory, and how fresh it is.

    A rank that stopped is dimmed and kept. Dropping it would hide the one
    fact worth having when a job stalls, which is WHICH rank stopped.

    Each row's CUDA columns come from that rank's least-headroom sample in
    the recent window. Keeping allocated and reserved on the paired sample
    makes the selected headline row directly verifiable in this table.
    """
    trend = {
        trace.global_rank: trace.values
        for trace in (chart.traces if chart else ())
    }
    head = (
        "<tr><th>rank</th><th>gpu</th><th>node</th><th>cpu cap</th>"
        "<th>rss</th><th>cpu trend</th><th>cuda allocated</th>"
        "<th>cuda reserved</th><th>age</th></tr>"
    )
    body = ""
    for rank in ranks:
        index = int(rank.global_rank)
        colour = charting.rank_color(index)
        # The median over the window, matching the tile above. Showing the
        # newest sample here made the card contradict itself: the tile
        # named R1 at its median while R1's own row showed its post
        # teardown value, so the number a reader went to the rows to
        # verify disagreed with the one that sent them there.
        used, used_rest = format_gb_pair(
            (
                rank.ram_used_p50_bytes
                if rank.ram_used_p50_bytes is not None
                else rank.ram_used_bytes
            ),
            rank.ram_total_bytes,
        )
        cuda = rank.cuda_least_headroom_sample
        reserved, reserved_rest = format_gb_pair(
            cuda.reserved_bytes if cuda is not None else None,
            cuda.total_bytes if cuda is not None else None,
        )
        alloc, alloc_rest = format_gb_pair(
            cuda.allocated_bytes if cuda is not None else None,
            cuda.total_bytes if cuda is not None else None,
        )
        capacity = rank.cpu_capacity_percent
        row = '<tr class="tml-stale">' if rank.freshness == "stale" else "<tr>"
        gpu_cell = (
            f"G{int(rank.gpu_index)}" if rank.gpu_index is not None else NA
        )
        node_cell = (
            f"N{int(rank.node_rank)}" if rank.node_rank is not None else NA
        )
        capacity_cell = (
            f"{num(capacity, '{:.1f}')} %" if capacity is not None else NA
        )
        body += (
            row + f'<td><span style="color:{colour}">■</span> R{index}</td>'
            f"<td>{gpu_cell}</td>"
            f"<td>{node_cell}</td>"
            f"<td>{capacity_cell}</td>"
            f"<td>{used} {used_rest}</td>"
            f"<td>{charting.sparkline_svg(trend.get(index, ()), colour)}</td>"
            f"<td>{alloc} {alloc_rest}</td>"
            f"<td>{reserved} {reserved_rest}</td>"
            f"<td>{format_age(rank.age_s)}</td>"
            "</tr>"
        )
    return f'<table class="tml-gpus">{head}{body}</table>'


def _rank_series(
    chart: RankChart, anchor: float, scale: float
) -> List[Dict[str, Any]]:
    """One line per rank, x in seconds before the shared anchor."""
    # A line through one point draws nothing, and the first ticks of every
    # run are exactly that. Show the markers until there is a second
    # sample to join them.
    sparse = any(len(trace.timestamps) < 2 for trace in chart.traces)
    series = []
    if chart.total is not None and chart.total.timestamps:
        # Drawn first so the rank lines sit on top of it, and in the ink
        # colour rather than a rank colour: it is not a rank, and giving it
        # one would make it look like the fifth GPU.
        total = charting.line_series(
            "Total",
            theme.INK,
            [
                [stamp - anchor, value / scale]
                for stamp, value in zip(
                    chart.total.timestamps, chart.total.values
                )
            ],
            width=2.2,
        )
        series.append(total)
    for trace in chart.traces:
        line = charting.line_series(
            f"R{int(trace.global_rank)}",
            charting.rank_color(int(trace.global_rank)),
            [
                [stamp - anchor, value / scale]
                for stamp, value in zip(trace.timestamps, trace.values)
            ],
        )
        if sparse:
            line["showSymbol"] = True
            line["symbolSize"] = 4
        series.append(line)
    return series


def _draw_chart(
    panel: Dict[str, Any],
    *,
    chart_key: str,
    label_key: str,
    head: str,
    chart: Optional[RankChart],
    aligned: Optional[Tuple[float, float]],
    scale: float,
    bounds_of: Any,
    tooltip: str,
    unit: str = "",
) -> None:
    """Draw whichever history the payload carried, and name it."""
    element = panel[chart_key]
    if chart is None or not chart.traces or aligned is None:
        element.options["series"] = []
        element.update()
        panel[label_key].text = head
        return

    anchor, span = aligned
    element.options["series"] = _rank_series(chart, anchor, scale)
    charting.apply_span_axis(element.options, span, anchor)

    # The values, never the peaks. `peaks` are the rolling maxima and
    # nothing draws them; fitting the axis to those put its floor above
    # the drawn line, which clipped the early samples and understated the
    # very drift this chart exists to show.
    # Every line that is DRAWN, the total included. It is the sum across
    # ranks, so it sits above all of them; fitting the axis to the rank
    # traces alone would put the one line the reader most wants above the
    # ceiling and clip it.
    drawn = list(chart.traces)
    if chart.total is not None and chart.total.timestamps:
        drawn.append(chart.total)
    flat = [value / scale for trace in drawn for value in trace.values]
    bounds = bounds_of(flat)
    if isinstance(bounds, tuple):
        low, high, tick = bounds
        element.options["yAxis"]["min"] = low
        element.options["yAxis"]["max"] = high
        element.options["yAxis"]["interval"] = tick
        element.options["yAxis"]["axisLabel"][":formatter"] = (
            charting.value_axis_formatter(high - low, unit)
        )
    else:
        element.options["yAxis"]["max"] = bounds
    element.update()

    words = format_window(chart.window_s) if chart.is_retained else ""
    panel[label_key].text = f"{head} · {format_span(span)}" + (
        f" · rolling {words}" if words else ""
    )
    panel[label_key].tooltip(tooltip)


def _chart_signature(payload: ProcessDashboardPayload) -> Tuple[Any, ...]:
    """What must change before the charts are worth re-sending.

    The UI timer ticks faster than telemetry arrives, and a full options
    dict per tick is pure websocket traffic.
    """
    chart = payload.cpu_capacity_chart
    return (
        payload.window_len,
        payload.coverage.total,
        payload.coverage.stale,
        tuple(
            trace.timestamps[-1] if trace.timestamps else None
            for trace in (chart.traces if chart else ())
        ),
    )


def _tile_gb(
    panel: Dict[str, Any], key: str, rollup: Optional[MetricRollup], sub: str
) -> None:
    """A byte level against the capacity it is measured out of.

    Every memory tile carries its denominator. A level without one cannot
    be read: 14 GB is unremarkable on an 80 GB card and nearly fatal on a
    16 GB one.
    """
    value, rest = format_gb_pair(
        rollup.now if rollup else None,
        rollup.total if rollup else None,
    )
    panel["tiles"][key].content = theme.kval(value, f" {rest}" if rest else "")
    panel["subs"][key].text = sub


def update_process_section(panel: Dict[str, Any], data: Any) -> None:
    """Fill the block from one Process payload."""
    if not isinstance(data, ProcessDashboardPayload):
        return

    changed = _chart_signature(data) != panel.get("_signature")
    panel["_signature"] = _chart_signature(data)

    capacity = data.cpu_capacity
    worst = capacity.now if capacity else None
    panel["tiles"]["cpu"].content = theme.kval(
        num(worst, "{:.1f}") if worst is not None else NA,
        "% of host" if worst is not None else "",
    )
    panel["subs"]["cpu"].text = (
        f"highest median · R{int(capacity.worst_rank)}"
        if capacity is not None and capacity.worst_rank is not None
        else "of host"
    )

    rss = data.rss_worst
    _tile_gb(
        panel,
        "rss",
        rss,
        (
            f"highest median · R{int(rss.worst_rank)}"
            if rss is not None and rss.worst_rank is not None
            else "used / total"
        ),
    )

    if data.gpu_available:
        reserved = data.gpu_reserved
        _tile_gb(
            panel,
            "reserved",
            reserved,
            (
                f"least headroom · R{int(reserved.worst_rank)}"
                if reserved is not None and reserved.worst_rank is not None
                else "reserved / total"
            ),
        )
        allocated = data.gpu_allocated
        _tile_gb(
            panel,
            "alloc",
            allocated,
            (
                f"least-headroom rank · R{int(allocated.worst_rank)}"
                if allocated is not None and allocated.worst_rank is not None
                else "live tensors"
            ),
        )
    else:
        seen = bool(data.window_len or data.ranks)
        for key in ("reserved", "alloc"):
            panel["tiles"][key].content = NA
            panel["subs"][key].text = "no GPU" if seen else ""

    aligned = charting.shared_span(
        data.cpu_capacity_chart.traces if data.cpu_capacity_chart else (),
        data.rss_chart.traces if data.rss_chart else (),
    )
    if changed:
        _draw_chart(
            panel,
            chart_key="cpu_chart",
            label_key="cpu_label",
            head="process cpu · capacity per rank",
            chart=data.cpu_capacity_chart,
            aligned=aligned,
            scale=1.0,
            bounds_of=charting.capacity_axis_max,
            tooltip=_CPU_TOOLTIP,
        )
        _draw_chart(
            panel,
            chart_key="rss_chart",
            label_key="rss_label",
            head="rss · per rank",
            chart=data.rss_chart,
            aligned=aligned,
            scale=float(1024**3),
            bounds_of=charting.drift_axis_bounds,
            unit=" GB",
            tooltip=_RSS_TOOLTIP,
        )

    panel["cpu_value"].text = (
        f"{float(worst):.1f}%" if worst is not None else ""
    )
    rss_value, rss_rest = format_gb_pair(rss.now if rss else None, None)
    panel["rss_value"].text = (
        f"{rss_value} {rss_rest}".strip() if rss is not None else ""
    )

    if should_auto_open(
        prev_over=bool(panel.get("_was_open")), over=data.rows_open
    ):
        panel["rows"].value = True
    panel["_was_open"] = data.rows_open
    panel["rows_hint"].text = rows_hint(
        data, is_open=bool(panel["rows"].value)
    )
    panel["rows_html"].content = rows_html(data.ranks, data.cpu_capacity_chart)
