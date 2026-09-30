# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process block: the trainer process on every rank.

A header, summary tiles, two per-rank graphs and a per-rank table. Scope
is stated in the header because it is the block's most useful fact: one
process per rank, its own PID only. DataLoader workers are separate
processes, so their CPU lands in the System block and never here. Reading
the two together is what the page is for: host CPU high while every
rank's process CPU is low means the work is outside the trainer.

One rank reads as usage ("CPU usage"), never as a comparison: there is
nothing to compare it with. Several ranks read as a selection ("Busiest
rank CPU"), and each graph highlights the rank its tile selected.

Presentation only. Every number arrives on a ``ProcessDashboardPayload``
already decided: which rank is worst, what the spread is, which history a
chart is made of, and whether the rows have earned opening. This module
chooses layout, colour, units and wording, and nothing else. In particular
it never compares a value to a threshold, because that is a severity
judgement and the diagnosis engine owns those.
"""

from __future__ import annotations

import html
import math
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
    format_elapsed,
    format_gb_pair,
    format_percent,
    num,
)

_MONO = "font-family:var(--mono);"

# The smallest range the memory graph draws, in GB. A steady trainer moves
# by kilobytes, and an axis fitted to that printed "0.586 GB" on all three
# ticks. A hundredth of a GB is below any movement worth reading here.
_MEMORY_MIN_SPAN_GB = 0.01

WAITING = "waiting for data"
ROWS_DESCRIPTION = "CPU, process memory, node and sample age"

# The period a tile's median covers. While the graphs draw the recent
# window it is the period they show. Once the run outgrows that window the
# graphs switch to whole-run history, but the tiles still summarise the
# recent window (``RECENT_WINDOW_S``, 60 s, in
# ``renderers/process/dashboard_compute.py``), so the tooltip names that
# window instead of pointing at a graph that now shows something longer.
_SHOWN_PERIOD = "the shown period"
_RECENT_MINUTE = "the most recent minute"

_CPU_TOOLTIP_ONE = (
    "CPU used by the trainer process, as a share of the host's total CPU "
    "capacity: 100% means every logical core busy. DataLoader workers are "
    "separate processes and are not included, so host CPU high with this "
    "low means the work is outside the trainer. On a hyperthreaded host, "
    "logical capacity is reached before the cores saturate."
)
_CPU_TOOLTIP_MANY = (
    "Each rank's trainer process, as a share of the host's total CPU "
    "capacity: 100% means every logical core busy. DataLoader workers are "
    "separate processes and are not included, so host CPU high with these "
    "low means the work is outside the trainer. On a hyperthreaded host, "
    "logical capacity is reached before the cores saturate."
)
_MEMORY_TOOLTIP_ONE = (
    "Physical RAM held by the trainer process (its resident set). The "
    "shape is the point: steady growth across the run is the signature of "
    "a leak, which ends with the OS killing the process. It covers the "
    "history retained for this run, not necessarily the whole run."
)
_MEMORY_TOOLTIP_MANY = (
    "Physical RAM held by each rank's trainer process (its resident set). "
    "The shape is the point: steady growth across the run is the signature "
    "of a leak, which ends with the OS killing one rank. It covers the "
    "history retained for this run, not necessarily the whole run."
)

# Hover values at the precision the tiles print, not rounded to a whole
# unit: a CPU hover of "23%" beside a "22.9%" tile reads as a second
# measurement.
_CPU_HOVER = "v=>v.toFixed(1)+'%'"


def build_process_section() -> Dict[str, Any]:
    """Build the block once; ``update_process_section`` fills it per tick."""
    panel: Dict[str, Any] = {
        "tiles": {},
        "subs": {},
        "tile_els": {},
        "tile_labels": {},
        "tile_tips": {},
        "_gpu_tiles": True,
        # What ``_show`` last set; every one of these starts hidden.
        "_rows_shown": False,
        "_cpu_sep_shown": False,
        "_rss_sep_shown": False,
    }
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; display:flex; "
        "flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:2px; gap:12px;")
        ):
            panel["title"] = ui.label("Trainer process").classes("ctitle")
            ui.element("div").style("flex:1;")
            panel["duration"] = ui.label("").classes("cmeta")
        panel["context"] = (
            ui.label(WAITING).classes("cmeta").style("margin-bottom:10px;")
        )

        # Two tiles to a row on every host. A quarter of a half-width card
        # wraps "Highest process memory" and "of host CPU · Rank 7", and a
        # CPU-only host then shows exactly the top row of a GPU host's card.
        panel["tilerow"] = (
            ui.element("div")
            .classes("tilerow tml-tiles-2")
            .style("margin-bottom:10px;")
        )
        with panel["tilerow"]:
            for key, label, accent in (
                ("cpu", "CPU usage", theme.C_CPU),
                ("rss", "Process memory", theme.C_MEM),
                ("reserved", "cuda reserved", theme.C_GPU),
                ("alloc", "cuda allocated", theme.C_GPU),
            ):
                tile = (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{accent}; min-width:0;")
                )
                with tile:
                    panel["tile_labels"][key] = ui.label(label).classes("klab")
                    panel["tiles"][key] = ui.html(NA, sanitize=False).classes(
                        "kval"
                    )
                    panel["subs"][key] = ui.label("").classes("ksub")
                    if key in ("cpu", "rss"):
                        panel["tile_tips"][key] = ui.tooltip(
                            _median_tooltip(many=False, retained=False)
                        )
                panel["tile_els"][key] = tile

        # Each graph in its own bordered block, so the pair never reads as
        # one chart with two halves.
        for key, head, unit, margin in (
            ("cpu", "CPU usage", "%", "0"),
            ("rss", "Process memory", " GB", "8px"),
        ):
            block = ui.element("div").style(
                f"margin-top:{margin}; padding:8px 10px 4px; "
                f"border:1px solid {theme.BORDER}; border-radius:12px;"
            )
            with block:
                with (
                    ui.row()
                    .classes("w-full items-baseline")
                    .style("gap:6px; margin:0 0 2px;")
                ):
                    # The title leads: ink and weight, at the size of the
                    # line beside it rather than smaller than it.
                    panel[f"{key}_label"] = (
                        ui.label(head)
                        .classes("estlabel")
                        .style(
                            "font-size:10px; color:var(--ink); "
                            "font-weight:600;"
                        )
                    )
                    with panel[f"{key}_label"]:
                        panel[f"{key}_tip"] = ui.tooltip("")
                    # "BY RANK" then "Rank 2" needs a mark between them.
                    panel[f"{key}_sep"] = (
                        ui.label("·")
                        .classes("cmeta")
                        .style("font-size:10px; display:none;")
                    )
                    panel[f"{key}_sub"] = (
                        ui.label("").classes("cmeta").style("font-size:10px;")
                    )
                panel[f"{key}_chart"] = ui.echart(
                    charting.multi_line_options(unit)
                ).style("height:92px; width:100%;")

        # Hidden until more than one rank reports: one rank's details are
        # the tiles above.
        expansion = (
            ui.expansion()
            .classes("w-full tml-exp")
            .props("dense dense-toggle expand-icon-toggle")
            .style("margin-top:6px; display:none;")
        )
        # The header is one line: the title and what the rows hold. What
        # the rows found (stale ranks, the reserved spread) is a caption
        # above the table, so a long run of facts cannot wrap the header
        # onto two lines and strand the chevron between them.
        with expansion.add_slot("header"):
            with (
                ui.row()
                .classes("w-full items-center no-wrap")
                .style("gap:10px; min-width:0;")
            ):
                panel["rows_title"] = ui.label("Rank details").style(
                    f"{_MONO} font-size:12px; font-weight:700; "
                    "white-space:nowrap;"
                )
                panel["rows_hint"] = (
                    ui.label(ROWS_DESCRIPTION)
                    .classes("cmeta")
                    .style(
                        "flex:1; min-width:0; white-space:nowrap; "
                        "overflow:hidden; text-overflow:ellipsis;"
                    )
                )
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


def header_context(payload: ProcessDashboardPayload) -> str:
    """Whose process the block describes, under the title.

    A stale rank is not counted as reporting: the tiles exclude it, and a
    header that counted it would describe ranks the numbers do not cover.
    One rank has no rank details to carry that, so its line says it.
    """
    ranks = payload.ranks
    if not ranks:
        return WAITING
    if len(ranks) == 1:
        line = f"Rank {int(ranks[0].global_rank)} · Trainer process only"
        return line + (" · not reporting" if payload.coverage.stale else "")
    total = len(ranks)
    reporting = max(total - payload.coverage.stale, 0)
    lead = (
        f"{total} ranks reporting"
        if reporting == total
        else f"{reporting} of {total} ranks reporting"
    )
    return f"{lead} · one trainer process per rank"


def rolling_words(chart: Optional[RankChart]) -> str:
    """``rolling 30s averages`` for a whole-run graph, else nothing.

    A point on whole-run history is not a sample, and a reader comparing
    one with a tile needs to know that. The window is written the way the
    header writes its span, so one line never mixes "5m 17s" with "30 s".
    """
    if chart is None or not chart.is_retained:
        return ""
    window = chart.window_s
    if window is None or not math.isfinite(window) or window <= 0:
        return "rolling averages"
    return f"rolling {format_elapsed(window)} averages"


def header_duration(span: Optional[float], rolled: Optional[RankChart]) -> str:
    """The period the graphs cover, once, and what their points are.

    Nothing until a span has been observed: one sample spans no time,
    and the axis's one-second drawing floor is not an observation.
    """
    if span is None or span <= 0:
        return ""
    words = rolling_words(rolled)
    last = f"Last {format_elapsed(span)}"
    return f"{last} · {words}" if words else last


def rows_facts(payload: ProcessDashboardPayload) -> str:
    """What the rank details found: coverage and spread, if any.

    It states what was observed. Whether that is bad is the engine's call,
    so no word here classifies it.
    """
    coverage = payload.coverage
    parts = []
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
    return " · ".join(parts)


def _cell(text: str, colour: Optional[str] = None) -> str:
    if colour is None:
        return f"<td>{text}</td>"
    return f'<td style="color:{colour};font-weight:600">{text}</td>'


def rows_html(
    ranks: Sequence[RankSnapshot],
    chart: Optional[RankChart],
    *,
    gpu: bool = True,
    cpu_rank: Optional[int] = None,
    mem_rank: Optional[int] = None,
    facts: str = "",
) -> str:
    """Per-rank table: identity, CPU, memory, and how fresh it is.

    A rank that stopped is dimmed and kept. Dropping it would hide the one
    fact worth having when a job stalls, which is WHICH rank stopped.

    Each row's CUDA columns come from that rank's least-headroom sample in
    the recent window. Keeping allocated and reserved on the paired sample
    makes the selected headline row directly verifiable in this table. A
    CPU-only host has no CUDA columns rather than three of "n/a".

    Colour follows the graphs: ranks are not told apart by colour, and the
    rank each graph highlights is marked in that graph's colour.

    ``facts`` (see :func:`rows_facts`) is a caption above the table, in
    the same markup so no layout gap opens between the two.
    """
    trend = {
        trace.global_rank: trace.values
        for trace in (chart.traces if chart else ())
    }
    columns = ["rank"] + (["gpu"] if gpu else [])
    columns += ["node", "cpu usage", "process memory", "cpu trend"]
    # Reserved before allocated, the order of the tiles above.
    columns += (["cuda reserved", "cuda allocated"] if gpu else []) + ["age"]
    head = "<tr>" + "".join(f"<th>{name}</th>" for name in columns) + "</tr>"
    body = ""
    for rank in ranks:
        index = int(rank.global_rank)
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
        capacity = rank.cpu_capacity_percent
        cpu_colour = theme.C_CPU if index == cpu_rank else None
        cells = [_cell(f"R{index}")]
        if gpu:
            cells.append(
                _cell(
                    f"G{int(rank.gpu_index)}"
                    if rank.gpu_index is not None
                    else NA
                )
            )
        cells.append(
            _cell(
                f"N{int(rank.node_rank)}" if rank.node_rank is not None else NA
            )
        )
        cells.append(
            _cell(
                (
                    f"{num(capacity, '{:.1f}')}%"
                    if capacity is not None
                    else NA
                ),
                cpu_colour,
            )
        )
        cells.append(
            _cell(
                f"{used} {used_rest}",
                theme.C_MEM if index == mem_rank else None,
            )
        )
        cells.append(
            _cell(
                charting.sparkline_svg(
                    trend.get(index, ()), cpu_colour or theme.MUTED
                )
            )
        )
        if gpu:
            cuda = rank.cuda_least_headroom_sample
            reserved, reserved_rest = format_gb_pair(
                cuda.reserved_bytes if cuda is not None else None,
                cuda.total_bytes if cuda is not None else None,
            )
            alloc, alloc_rest = format_gb_pair(
                cuda.allocated_bytes if cuda is not None else None,
                cuda.total_bytes if cuda is not None else None,
            )
            cells.append(_cell(f"{reserved} {reserved_rest}"))
            cells.append(_cell(f"{alloc} {alloc_rest}"))
        cells.append(_cell(format_age(rank.age_s)))
        row = '<tr class="tml-stale">' if rank.freshness == "stale" else "<tr>"
        body += row + "".join(cells) + "</tr>"
    table = f'<table class="tml-gpus">{head}{body}</table>'
    if not facts:
        return table
    caption = (
        '<div class="cmeta" style="padding:0 8px 4px">'
        f"{html.escape(facts)}</div>"
    )
    return caption + table


def _rank_series(
    chart: RankChart,
    anchor: float,
    scale: float,
    *,
    accent: str,
    highlight: Optional[int],
) -> List[Dict[str, Any]]:
    """One line per rank, x in seconds before the shared anchor.

    The highlighted rank is drawn in the graph's colour and on top; every
    other rank is a thin muted line. With ``highlight`` ``None`` (one rank,
    or no rank selected) every line is drawn in the graph's colour, so the
    CPU and memory graphs stay distinguishable.

    The highlighted rank is the LAST series, because ECharts paints later
    series over earlier ones. ``z`` and the opacity are set on every
    series, never only on the ones that differ: when the series count is
    unchanged NiceGUI merges new options into the old by index, so a key
    set on one tick outlives the rank it was set for. A ``z`` left behind
    by an earlier highlight painted muted ranks over the accent line.
    """
    # A line through one point draws nothing, and the first ticks of every
    # run are exactly that. Show the markers until there is a second
    # sample to join them.
    sparse = any(len(trace.timestamps) < 2 for trace in chart.traces)
    series = []
    for trace in sorted(
        chart.traces, key=lambda t: int(t.global_rank) == highlight
    ):
        rank = int(trace.global_rank)
        lead = highlight is None or rank == highlight
        line = charting.line_series(
            f"Rank {rank}",
            accent if lead else theme.MUTED,
            [
                [stamp - anchor, value / scale]
                for stamp, value in zip(trace.timestamps, trace.values)
            ],
            width=2.0 if lead else 1.0,
        )
        line["z"] = 3 if lead else 2
        line["lineStyle"]["opacity"] = 1.0 if lead else 0.55
        if sparse:
            line["showSymbol"] = True
            line["symbolSize"] = 4
        series.append(line)
    return series


def _draw_chart(
    element: Any,
    *,
    chart: Optional[RankChart],
    aligned: Optional[Tuple[float, float]],
    scale: float,
    bounds_of: Any,
    accent: str,
    highlight: Optional[int],
    unit: str,
    hover: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Draw whichever history the payload carried.

    Returns the y tick formatter and the widest label it writes, so the
    caller can give both graphs the same label room; ``None`` when there
    was nothing to draw. The caller sends the chart.
    """
    if chart is None or not chart.traces or aligned is None:
        element.options["series"] = []
        return None

    anchor, span = aligned
    element.options["series"] = _rank_series(
        chart, anchor, scale, accent=accent, highlight=highlight
    )
    charting.apply_span_axis(element.options, span, anchor)

    # The values, never the peaks. `peaks` are the rolling maxima and
    # nothing draws them; fitting the axis to those put its floor above
    # the drawn line, which clipped the early samples and understated the
    # very drift this chart exists to show. The same holds for the total
    # across ranks: it is not drawn, so it does not size the axis.
    flat = [value / scale for trace in chart.traces for value in trace.values]
    bounds = bounds_of(flat)
    axis = element.options["yAxis"]
    if isinstance(bounds, tuple):
        low, high, tick = bounds
        formatter = charting.value_axis_formatter(tick, unit)
        widest = charting.value_axis_label(high, tick, unit)
        axis["min"] = low
        axis["max"] = high
        axis["interval"] = tick
    else:
        formatter = charting.unit_axis_formatter(unit)
        widest = charting.unit_axis_label(bounds, unit)
        # Zero to the ceiling in two equal steps (0 / 15 / 30). With only
        # the ceiling set, ECharts picked its own step and drew 0 / 20 / 30.
        axis["min"] = 0
        axis["max"] = bounds
        axis["interval"] = bounds / 2.0
    tooltip = element.options["tooltip"]
    tooltip[":valueFormatter"] = f"v=>(v==null?'-':({hover or formatter})(v))"
    return formatter, widest


def _memory_bounds(values: Sequence[Any]) -> Tuple[float, float, float]:
    return charting.drift_axis_bounds(values, min_span=_MEMORY_MIN_SPAN_GB)


def _chart_signature(
    payload: ProcessDashboardPayload,
) -> Tuple[Any, ...]:
    """What must change before the charts are worth re-sending.

    The UI timer ticks faster than telemetry arrives, and a full options
    dict per tick is pure websocket traffic. The highlighted ranks are
    part of it because they decide each line's colour.
    """
    chart = payload.cpu_capacity_chart
    return (
        payload.window_len,
        payload.coverage.total,
        payload.coverage.stale,
        _worst(payload.cpu_capacity),
        _worst(payload.rss_worst),
        tuple(
            trace.timestamps[-1] if trace.timestamps else None
            for trace in (chart.traces if chart else ())
        ),
    )


def _worst(rollup: Optional[MetricRollup]) -> Optional[int]:
    if rollup is None or rollup.worst_rank is None:
        return None
    return int(rollup.worst_rank)


def _drawn(payload: ProcessDashboardPayload) -> List[RankChart]:
    return [
        chart
        for chart in (payload.cpu_capacity_chart, payload.rss_chart)
        if chart is not None and chart.traces
    ]


def _rolled(payload: ProcessDashboardPayload) -> Optional[RankChart]:
    """The rolling window every drawn graph shares, if they share one.

    Each metric picks its history from its own run statistics, so for a
    tick one graph can draw whole-run averages while the other still draws
    samples, or the two can average over different windows. The header
    then names no window, and each whole-run graph names its own.
    """
    drawn = _drawn(payload)
    if not drawn or not all(chart.is_retained for chart in drawn):
        return None
    if len({chart.window_s for chart in drawn}) != 1:
        return None
    return drawn[0]


def _median_tooltip(*, many: bool, retained: bool) -> str:
    period = _RECENT_MINUTE if retained else _SHOWN_PERIOD
    if many:
        return (
            f"The displayed rank has the highest median value during "
            f"{period}."
        )
    return f"Median over {period}."


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


def _cuda_sub(
    rollup: Optional[MetricRollup], *, many: bool, plain: str
) -> str:
    """Which rank a CUDA tile describes, when there is a choice of ranks."""
    rank = _worst(rollup)
    if many and rank is not None:
        return f"least headroom · Rank {rank}"
    return plain


def _set_gpu_tiles(panel: Dict[str, Any], shown: bool) -> None:
    """Show or drop the CUDA tiles, which fill the tile row's second line.

    Dropped rather than marked absent: a CPU-only host has no CUDA memory
    to report, and two tiles saying so tell the reader nothing.
    """
    if panel.get("_gpu_tiles") == shown:
        return
    panel["_gpu_tiles"] = shown
    for key in ("reserved", "alloc"):
        panel["tile_els"][key].style(
            f"display:{'block' if shown else 'none'};"
        )


def _show(panel: Dict[str, Any], key: str, shown: bool) -> None:
    """Show or hide one element, touching it only when that changes."""
    flag = f"_{key}_shown"
    if panel.get(flag) == shown:
        return
    panel[flag] = shown
    panel[key].style(f"display:{'block' if shown else 'none'};")


def _update_tiles(
    panel: Dict[str, Any],
    data: ProcessDashboardPayload,
    *,
    many: bool,
    retained: bool,
) -> None:
    capacity = data.cpu_capacity
    worst = capacity.now if capacity else None
    cpu_rank = _worst(capacity)
    panel["tile_labels"]["cpu"].text = (
        "Busiest rank CPU" if many else "CPU usage"
    )
    panel["tiles"]["cpu"].content = theme.kval(
        num(worst, "{:.1f}") if worst is not None else NA,
        "%" if worst is not None else "",
    )
    cpu_sub = "of host CPU" if worst is not None else ""
    if cpu_sub and many and cpu_rank is not None:
        cpu_sub += f" · Rank {cpu_rank}"
    panel["subs"]["cpu"].text = cpu_sub

    rss = data.rss_worst
    mem_rank = _worst(rss)
    panel["tile_labels"]["rss"].text = (
        "Highest process memory" if many else "Process memory"
    )
    used, rest = format_gb_pair(
        rss.now if rss else None, rss.total if rss else None
    )
    panel["tiles"]["rss"].content = theme.kval(
        used, " GB" if rss is not None else ""
    )
    if many:
        panel["subs"]["rss"].text = (
            f"Rank {mem_rank}" if mem_rank is not None else ""
        )
    else:
        # The denominator as ``format_gb_pair`` writes it ("/ 24.0 GB"),
        # so the host's RAM reads at the precision of every other GB here.
        panel["subs"]["rss"].text = (
            f"of {rest[2:]} host RAM" if rest.startswith("/ ") else ""
        )

    tip = _median_tooltip(many=many, retained=retained)
    for key in ("cpu", "rss"):
        panel["tile_tips"][key].text = tip

    # Before any rank reports, nothing is known about the GPU either way:
    # the empty payload answers ``gpu_available`` with False only because
    # it has no ranks to ask. Keeping the four-tile row until data arrives
    # also keeps a GPU host's card from jumping on its first sample.
    seen = data.has_data
    _set_gpu_tiles(panel, data.gpu_available or not seen)
    if data.gpu_available:
        reserved = data.gpu_reserved
        _tile_gb(
            panel,
            "reserved",
            reserved,
            _cuda_sub(reserved, many=many, plain="held by the process"),
        )
        allocated = data.gpu_allocated
        _tile_gb(
            panel,
            "alloc",
            allocated,
            _cuda_sub(allocated, many=many, plain="live tensors"),
        )
    elif not seen:
        for key in ("reserved", "alloc"):
            panel["tiles"][key].content = NA
            panel["subs"][key].text = ""


def update_process_section(panel: Dict[str, Any], data: Any) -> None:
    """Fill the block from one Process payload."""
    if not isinstance(data, ProcessDashboardPayload):
        return

    signature = _chart_signature(data)
    changed = signature != panel.get("_signature")
    panel["_signature"] = signature

    count = len(data.ranks)
    many = count > 1
    only = int(data.ranks[0].global_rank) if count == 1 else None
    cpu_rank = _worst(data.cpu_capacity)
    mem_rank = _worst(data.rss_worst)
    rolled = _rolled(data)

    # Whole seconds, rounded up: the header and the leftmost tick then name
    # the same span, and no sample falls off the left edge.
    aligned = charting.shared_span(
        data.cpu_capacity_chart.traces if data.cpu_capacity_chart else (),
        data.rss_chart.traces if data.rss_chart else (),
    )
    if aligned is not None:
        aligned = (aligned[0], float(math.ceil(aligned[1])))

    panel["title"].text = "Trainer processes" if many else "Trainer process"
    panel["context"].text = header_context(data)
    panel["duration"].text = header_duration(
        aligned[1] if aligned is not None else None, rolled
    )

    _update_tiles(
        panel,
        data,
        many=many,
        retained=any(chart.is_retained for chart in _drawn(data)),
    )

    if changed:
        drawn = (
            (
                panel["cpu_chart"],
                _draw_chart(
                    panel["cpu_chart"],
                    chart=data.cpu_capacity_chart,
                    aligned=aligned,
                    scale=1.0,
                    bounds_of=charting.capacity_axis_max,
                    accent=theme.C_CPU,
                    highlight=cpu_rank if many else None,
                    unit="%",
                    hover=_CPU_HOVER,
                ),
            ),
            (
                panel["rss_chart"],
                _draw_chart(
                    panel["rss_chart"],
                    chart=data.rss_chart,
                    aligned=aligned,
                    scale=float(1024**3),
                    bounds_of=_memory_bounds,
                    accent=theme.C_MEM,
                    highlight=mem_rank if many else None,
                    unit=" GB",
                ),
            ),
        )
        # One label width for both graphs, so their plots start at the
        # same x and a vertical read across the pair is one moment.
        width = max(
            (len(axis[1]) for _el, axis in drawn if axis is not None),
            default=0,
        )
        for element, axis in drawn:
            if axis is not None:
                element.options["yAxis"]["axisLabel"][":formatter"] = (
                    charting.pad_tick_labels(axis[0], width)
                )
            element.update()

    if many:
        panel["cpu_label"].text = "CPU usage by rank"
        panel["rss_label"].text = "Process memory by rank"
        cpu_sub = (
            f"Rank {cpu_rank} highlighted" if cpu_rank is not None else ""
        )
        rss_sub = (
            f"Rank {mem_rank} highlighted" if mem_rank is not None else ""
        )
        panel["cpu_tip"].text = _CPU_TOOLTIP_MANY
        panel["rss_tip"].text = _MEMORY_TOOLTIP_MANY
    else:
        panel["cpu_label"].text = "CPU usage"
        panel["rss_label"].text = "Process memory"
        cpu_sub = (
            f"Rank {only} · share of host CPU" if only is not None else ""
        )
        rss_sub = (
            f"Rank {only} · physical RAM used" if only is not None else ""
        )
        panel["cpu_tip"].text = _CPU_TOOLTIP_ONE
        panel["rss_tip"].text = _MEMORY_TOOLTIP_ONE
    for key, sub, chart in (
        ("cpu", cpu_sub, data.cpu_capacity_chart),
        ("rss", rss_sub, data.rss_chart),
    ):
        # The header names a rolling window only when both graphs share
        # it; otherwise a whole-run graph names its own here.
        note = ""
        if rolled is None and chart is not None and chart.traces:
            note = rolling_words(chart)
        text = " · ".join(part for part in (sub, note) if part)
        panel[f"{key}_sub"].text = text
        _show(panel, f"{key}_sep", bool(text))

    _show(panel, "rows", many)
    panel["rows_title"].text = (
        f"Rank details ({count})" if many else "Rank details"
    )
    if should_auto_open(
        prev_over=bool(panel.get("_was_open")), over=data.rows_open
    ):
        panel["rows"].value = True
    panel["_was_open"] = data.rows_open
    panel["rows_html"].content = rows_html(
        data.ranks,
        data.cpu_capacity_chart,
        gpu=data.gpu_available,
        cpu_rank=cpu_rank if many else only,
        mem_rank=mem_rank if many else only,
        facts=rows_facts(data),
    )
