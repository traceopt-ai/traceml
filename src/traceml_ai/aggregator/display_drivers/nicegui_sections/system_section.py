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
# The floor reference is deliberately not red: red belongs to the limit.
_FLOOR_GREY = "#94a3b8"
_MONO = "font-family:var(--mono);"


# --- pure display rules ----------------------------------------------------
def gpu_color(index: int) -> str:
    return _GPU_COLORS[int(index) % len(_GPU_COLORS)]


def format_window(window_s: float) -> str:
    """The rolling window in words: '30 s', '2 min'.

    The payload picks round windows (see ``choose_window_s``) so this reads
    as a duration a person recognises.
    """
    if not window_s or window_s <= 0:
        return ""
    if window_s < 60:
        return f"{window_s:.0f} s"
    return f"{window_s / 60.0:.0f} min"


def format_span(seconds: Optional[float]) -> str:
    """The window a chart covers, as one phrase in its label: 'last 3 min'."""
    if not seconds or seconds <= 0:
        return ""
    if seconds < 90:
        return f"last {seconds:.0f} s"
    return f"last {seconds / 60.0:.0f} min"


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


def disclosure_text(gpus: List[Dict[str, Any]], *, is_open: bool) -> str:
    """Header of the per-GPU rows: the utilisation range, and nothing else.

    It states what the GPUs read, low to high, and leaves the reading to
    the person or to the diagnosis engine. An earlier version said "1 of 4
    GPUs busy, 3 idle" off a 20 % bar invented here, while the engine calls
    anything under 30 % low (``GPUUtilizationBands``): one page, two
    thresholds, and a word this layer had no standing to say. The trigger
    that opens the rows is unchanged and stays off the screen. The tail
    follows the rows' real state, which a click may have changed.
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
    lo, hi = min(utils), max(utils)
    return f"{n} GPUs · util {lo:.0f} to {hi:.0f}%" + tail


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


# The host CPU series is psutil.cpu_percent(): the MEAN utilisation across
# all logical cores, so 100% is every core saturated and one busy core of
# ten reads 10%. Measured 2026-08-21 on a 10-core box with one core
# spinning: cpu_percent() == mean(percpu) == 36.6%, sum(percpu) == 366.5%.
# The wording mirrors the GPU tile's "avg of 4 GPUs", and it has to be
# explicit because the Process block on the same page reports PER-RANK cpu
# from psutil.Process.cpu_percent(), which sums across cores and can pass
# 100%. (The sampler reads psutil.cpu_count() but never puts it on the
# wire; carrying it would let this read "avg of 10 cores".)
CPU_LABEL = "host cpu util · avg across cores"

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
def shared_run_axis(
    run_t: List[float], prun: List[Dict[str, Any]]
) -> Optional[Tuple[float, float]]:
    """One (anchor, span) covering both whole-run series.

    Returns the newest clock across the two and the reach back to the
    oldest, so both charts can be pinned to the same interval.
    """
    starts = [run_t[0]] + [e["t"][0] for e in prun if e.get("t")]
    ends = [run_t[-1]] + [e["t"][-1] for e in prun if e.get("t")]
    if not starts or not ends:
        return None
    newest = max(ends)
    return (newest, max(newest - min(starts), 1.0))


def _newest_epoch(x_time: List[str]) -> Optional[float]:
    """Epoch seconds of the newest parseable stamp, for the clock axis."""
    for value in reversed(x_time):
        try:
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            continue
    return None


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


def _apply_span_axis(
    options: Dict[str, Any], span: float, newest_epoch: Optional[float] = None
) -> None:
    """Pin a chart to its span and label it in wall-clock time.

    The x values are seconds before the newest sample, which keeps the
    series arithmetic simple, but a reader debugging a slowdown needs the
    clock: it is what their logs and every other dashboard are keyed on,
    and it is what the Process block beside this one already shows. The
    formatters convert on the fly from the newest sample's epoch, and the
    hover label carries both readings ("19:10 · 45 min ago") so the axis
    and the tooltip never speak two different vocabularies.
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

        with ui.element("div").classes("tilerow").style("margin-bottom:10px;"):
            for key, label, acc in (
                ("util", "gpu util", theme.C_GPU),
                ("mem", "gpu mem", theme.C_GPU),
                ("temp", "gpu temp", theme.C_GPU),
                ("ram", "host ram", theme.C_CPU),
            ):
                tile = (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{acc}; min-width:0;")
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
            panel["cpu_label"] = ui.label(CPU_LABEL).classes("estlabel")
            ui.element("div").style("flex:1;")
            panel["cpu_value"] = ui.label("").style(
                f"{_MONO} font-size:13px; font-weight:600; color:var(--ink);"
            )
        panel["cpu_chart"] = ui.echart(
            theme.span_line_options(theme.C_CPU, "%")
        ).style("height:92px; width:100%;")
        # Faint upper trace: each slice's peak, so decimating the whole run
        # cannot hide a spike.

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


def _update_cpu_chart(
    panel: Dict[str, Any],
    roll: Dict[str, Any],
    series: Dict[str, Any],
    *,
    changed: bool,
    secs: List[Optional[float]],
    span: float,
    newest_epoch: Optional[float],
    whole_run: bool,
    aligned: Optional[Tuple[float, float]],
) -> None:
    """Update the CPU trace, axis, value and label for the selected view."""
    cpu = series.get("cpu", []) or []
    run = series.get("cpu_run") or {}
    run_t = run.get("t") or []
    run_avg = run.get("avg") or []
    run_max = run.get("max") or []
    run_span = float(run.get("span_s") or 0.0)
    chart = panel["cpu_chart"]

    if changed and whole_run:
        newest, axis_span = aligned or (run_t[-1], run_span)
        chart.options["series"][0]["data"] = [
            [t - newest, value] for t, value in zip(run_t, run_avg)
        ]
        _apply_span_axis(chart.options, axis_span, newest)
        # Peaks are not drawn, but they keep a smoothed spike inside the axis.
        ymax = cpu_axis_max(run_max or run_avg)
    elif changed:
        chart.options["series"][0]["data"] = [
            [second, value]
            for second, value in zip(secs, cpu)
            if second is not None
        ]
        _apply_span_axis(chart.options, span, newest_epoch)
        ymax = cpu_axis_max(cpu)
    if changed:
        chart.options["yAxis"]["max"] = ymax
        chart.options["yAxis"]["interval"] = ymax / 2.0
        chart.update()

    cpu_p50 = (roll.get("cpu", {}) or {}).get("p50")
    panel["cpu_value"].text = f"{cpu_p50:.0f}%" if cpu_p50 is not None else ""
    span_words = format_span(span)
    if whole_run:
        window = format_window(float(run.get("window_s") or 0.0))
        panel["cpu_label"].text = f"{CPU_LABEL} · whole run" + (
            f" · rolling {window}" if window else ""
        )
        panel["cpu_label"].tooltip(
            f"Whole run, {format_span(run_span)[5:]}. Each point shows "
            f"average host CPU use over the previous {window}. 100% means "
            "all logical CPU cores are fully used."
        )
    else:
        panel["cpu_label"].text = (
            f"{CPU_LABEL} · {span_words}" if span_words else CPU_LABEL
        )


def _update_system_tiles(
    panel: Dict[str, Any],
    roll: Dict[str, Any],
    *,
    gpu_on: bool,
    has_data: bool,
) -> None:
    """Update the System header and four resource tiles."""
    ram = roll.get("ram", {}) or {}
    num, rest = format_gb_pair(ram.get("now"), ram.get("total"))
    panel["tiles"]["ram"].content = theme.kval(num, f" {rest}" if rest else "")
    panel["subs"]["ram"].text = "used / total"

    notes = []
    node_note = node_scope_text(roll.get("ctx", {}) or {})
    if node_note:
        notes.append(node_note)
    if not has_data:
        notes.append("waiting for data")
    panel["note"].text = " · ".join(notes)

    if not gpu_on:
        for key in ("util", "mem", "temp"):
            panel["tiles"][key].content = NA
            panel["subs"][key].text = "no GPU" if has_data else ""
        return

    gpus = roll.get("gpus", []) or []
    ctx = roll.get("ctx", {}) or {}
    n_gpus = len(gpus) or int(ctx.get("gpu_count", 0))
    # The sampler's all-zero fallback is unreported, not a zero-valued GPU.
    unreported = bool(gpus) and all(
        gpu.get("mem_total") is None and gpu.get("power") is None
        for gpu in gpus
    )

    util_p50 = (roll.get("gpu_util", {}) or {}).get("p50")
    panel["tiles"]["util"].content = theme.kval(_num(util_p50), "%")
    one_gpu = n_gpus == 1
    panel["subs"]["util"].text = (
        "1 GPU" if one_gpu else f"avg of {n_gpus} GPUs"
    )

    gpu_mem = roll.get("gpu_mem", {}) or {}
    if unreported:
        panel["tiles"]["mem"].content = NA
        panel["subs"]["mem"].text = "GPU sample unreported"
        panel["tiles"]["temp"].content = NA
        panel["subs"]["temp"].text = "GPU sample unreported"
        return

    num, rest = format_gb_pair(gpu_mem.get("now"), gpu_mem.get("total"))
    panel["tiles"]["mem"].content = theme.kval(num, f" {rest}" if rest else "")
    panel["subs"]["mem"].text = "used / total" if one_gpu else "max GPU"
    temp_now = (roll.get("temp", {}) or {}).get("now")
    panel["tiles"]["temp"].content = theme.kval(_num(temp_now), " °C")
    # A qualifier keeps all four tile heights equal on single-GPU hosts.
    panel["subs"]["temp"].text = "1 GPU" if one_gpu else "max GPU"


def _update_power_chart(
    panel: Dict[str, Any],
    roll: Dict[str, Any],
    series: Dict[str, Any],
    *,
    gpu_on: bool,
    has_data: bool,
    changed: bool,
    secs: List[Optional[float]],
    span: float,
    newest_epoch: Optional[float],
    whole_run: bool,
    aligned: Optional[Tuple[float, float]],
) -> None:
    """Update the per-GPU power chart or its no-data presentation."""
    if not gpu_on:
        panel["power_label"].text = "gpu power"
        panel["power_placeholder"].text = (
            "no GPU reported" if has_data else "waiting for data"
        )
        return

    gpu_power = roll.get("gpu_power", {}) or {}
    limit = gpu_power.get("limit")
    pseries = series.get("gpu_power", []) or []
    flat = [value for item in pseries for value in (item.get("values") or [])]
    chart = panel["power_chart"]
    if not any(value is not None for value in flat):
        chart.style("display:none;")
        panel["power_label"].text = "gpu power · not reported"
        return
    if not changed:
        return

    run = series.get("gpu_power_run") or []
    run_span = float(run[0].get("span_s") or 0.0) if run else 0.0
    if whole_run:
        # Whole-run mean shows sustained draw; its faint floor exposes idle
        # intervals without adding a third peak trace that usually saturates.
        newest = (
            aligned or (max(item["t"][-1] for item in run if item.get("t")),)
        )[0]
        lines = []
        for item in run:
            index = int(item.get("gpu_idx", 0))
            xs = [timestamp - newest for timestamp in item["t"]]
            floor_line = theme.line_series(
                f"gpu{index} floor",
                gpu_color(index),
                [[x, value] for x, value in zip(xs, item.get("min") or [])],
                width=0.9,
            )
            floor_line["lineStyle"]["opacity"] = 0.35
            floor_line["tooltip"] = {"show": False}
            lines.append(floor_line)
            lines.append(
                theme.line_series(
                    f"gpu{index}",
                    gpu_color(index),
                    [[x, value] for x, value in zip(xs, item["avg"])],
                )
            )
    else:
        lines = [
            theme.line_series(
                f"gpu{int(item.get('gpu_idx', 0))}",
                gpu_color(int(item.get("gpu_idx", 0))),
                [
                    [second, value]
                    for second, value in zip(secs, item.get("values") or [])
                    if second is not None
                ],
            )
            for item in pseries
        ]

    if lines:
        refs = []
        if limit is not None:
            refs.append(
                (
                    float(limit),
                    f"{float(limit):.0f} W limit",
                    _LIMIT_RED,
                    "insideEndTop",
                )
            )
        floor_w = gpu_power.get("floor")
        if floor_w is not None and (
            limit is None or float(floor_w) < float(limit) * 0.9
        ):
            # Opposite label corners prevent the two reference lines from
            # colliding or being clipped at the card edge.
            refs.append(
                (
                    float(floor_w),
                    f"{float(floor_w):.0f} W lowest seen",
                    _FLOOR_GREY,
                    "insideStartBottom",
                )
            )
        # ECharts merges options, so an explicit empty markLine clears old
        # reference lines when a later payload no longer reports them.
        lines[0]["markLine"] = theme.mark_lines(refs) if refs else {"data": []}

    chart.options["series"] = lines
    bounds_source = (
        [
            value
            for item in run
            for value in item["avg"] + (item.get("min") or [])
        ]
        if whole_run
        else flat
    )
    low, high, tick = power_axis_bounds(bounds_source, limit)
    chart.options["yAxis"]["min"] = low
    chart.options["yAxis"]["max"] = high
    chart.options["yAxis"]["interval"] = tick
    if whole_run:
        anchor, axis_span = aligned or (
            max(item["t"][-1] for item in run if item.get("t")),
            run_span,
        )
        _apply_span_axis(chart.options, axis_span, anchor)
    else:
        _apply_span_axis(chart.options, span, newest_epoch)
    chart.style("display:block;")
    chart.update()

    head = (
        f"gpu power · per GPU vs {float(limit):.0f} W limit"
        if limit is not None
        else "gpu power · per GPU"
    )
    if whole_run:
        window = format_window(float(run[0].get("window_s") or 0.0))
        panel["power_label"].text = f"{head} · whole run" + (
            f" · average and lowest every {window}" if window else ""
        )
        panel["power_label"].tooltip(
            f"Whole run, {format_span(run_span)[5:]}. Solid lines show each "
            f"GPU's average power every {window}; faint lines show the "
            "lowest reading during the same interval."
        )
    else:
        span_words = format_span(span)
        panel["power_label"].text = (
            f"{head} · {span_words}" if span_words else head
        )
        panel["power_label"].tooltip(
            "Recent power readings for each GPU, compared with its power "
            "limit."
        )


def _update_gpu_rows(
    panel: Dict[str, Any],
    roll: Dict[str, Any],
    series: Dict[str, Any],
    *,
    gpu_on: bool,
    has_data: bool,
) -> None:
    """Update per-GPU disclosure state, summary text and row contents."""
    if not gpu_on:
        panel["rows_placeholder"].text = (
            "per-GPU rows · no GPU" if has_data else "per-GPU rows"
        )
        return

    gpus = roll.get("gpus", []) or []
    pseries = series.get("gpu_power", []) or []
    spread = (roll.get("gpu_delta", {}) or {}).get("p95")
    over = spread is not None and float(spread) > SPREAD_EXPAND_PTS
    if should_auto_open(prev_over=panel["_over"], over=over):
        panel["rows"].value = True
    panel["_over"] = over
    panel["rows_hint"].text = disclosure_text(
        gpus, is_open=bool(panel["rows"].value)
    )
    panel["rows_html"].content = rows_html(gpus, pseries, spread=spread)


def update_system_section(panel: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Coordinate System presentation updates from one computed payload."""
    if not isinstance(data, dict):
        return
    roll = data.get("rollups", {}) or {}
    series = data.get("series", {}) or {}
    gpu_on = bool(data.get("gpu_available") or roll.get("gpu_available"))
    has_data = bool(data.get("window_len"))
    _set_gpu_visible(panel, gpu_on)

    x_time = series.get("x_time", []) or []
    secs = _relative_seconds(x_time)
    newest_epoch = _newest_epoch(x_time)
    present = [second for second in secs if second is not None]
    span = -min(present) if present else 0.0

    # The UI timer is faster than sampling; avoid resending unchanged charts.
    gpu_power = roll.get("gpu_power", {}) or {}
    pseries = series.get("gpu_power", []) or []
    signature = (
        x_time[-1] if x_time else None,
        data.get("window_len"),
        len(pseries),
        gpu_power.get("limit"),
        gpu_on,
    )
    changed = signature != panel.get("_sig")
    panel["_sig"] = signature

    cpu_run = series.get("cpu_run") or {}
    cpu_run_t = cpu_run.get("t") or []
    power_run = series.get("gpu_power_run") or []
    cpu_whole = len(cpu_run_t) > 2
    power_whole = bool(power_run)
    # Whole-run charts share one clock so vertical comparisons align.
    aligned = (
        shared_run_axis(cpu_run_t, power_run)
        if cpu_whole and power_whole
        else None
    )

    _update_cpu_chart(
        panel,
        roll,
        series,
        changed=changed,
        secs=secs,
        span=span,
        newest_epoch=newest_epoch,
        whole_run=cpu_whole,
        aligned=aligned,
    )
    _update_system_tiles(
        panel,
        roll,
        gpu_on=gpu_on,
        has_data=has_data,
    )
    _update_power_chart(
        panel,
        roll,
        series,
        gpu_on=gpu_on,
        has_data=has_data,
        changed=changed,
        secs=secs,
        span=span,
        newest_epoch=newest_epoch,
        whole_run=power_whole,
        aligned=aligned,
    )
    _update_gpu_rows(
        panel,
        roll,
        series,
        gpu_on=gpu_on,
        has_data=has_data,
    )
