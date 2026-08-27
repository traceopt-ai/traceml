"""Process metrics card: RAM and GPU-memory over time, plus four tiles.

Presentation only. Every number here arrives on a
``ProcessDashboardPayload`` already decided: the percentiles, the window
bound and the share-of-capacity arithmetic moved to
``renderers/process/dashboard_compute.py``, which is where what a metric
MEANS is settled. What is left in this module is layout, units, colours
and chart options.

Whether the run has a GPU is read from ``payload.gpu_available``, not
inferred from which keys happen to be present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.renderers.process.dashboard_models import (
    ChartTrace,
    ProcessDashboardPayload,
)

from . import theme

NA_GPU = "N/A"
DASH = "—"


def build_process_section() -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:8px; gap:12px;")
        ):
            ui.label("Process").classes("ctitle")
            for nm, col in [("RAM", theme.C_CPU), ("GPU mem", theme.C_GPU)]:
                with ui.element("div").classes("legchip"):
                    ui.element("div").classes("legdot").style(
                        f"background:{col};"
                    )
                    ui.label(nm)
            ui.element("div").style("flex:1;")
            win = ui.label("waiting for data").classes("cmeta")
        chart = ui.echart(theme.dual_line_options("RAM", "GPU mem")).style(
            "height:200px; width:100%; flex:1; min-height:160px;"
        )
        with (
            ui.row()
            .classes("w-full")
            .style("gap:8px; margin-top:12px; flex-wrap:nowrap;")
        ):
            for key, lab, acc, qual in [
                ("cpu", "CPU", theme.C_CPU, "max · rank"),
                ("ram", "RAM", theme.C_CPU, "max · rank"),
                ("gmem", "GPU MEM", theme.C_GPU, "worst rank"),
                ("gimb", "GPU IMBAL", theme.C_GPU, "spread"),
            ]:
                with (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{acc}; flex:1 1 0; min-width:0;")
                ):
                    ui.html(
                        f"{lab} <span class='kq'>{qual}</span>",
                        sanitize=False,
                    ).classes("klab")
                    kpis[key] = ui.html(DASH, sanitize=False).classes("kval")
    return {"chart": chart, "win": win, "kpis": kpis}


def to_ms(seconds: Optional[float]) -> Optional[int]:
    """A sample time as ECharts wants it, or ``None`` when unusable."""
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return int(value * 1000.0)


def gb_value(value: Optional[float]) -> str:
    """Bytes as gigabytes, or a dash when the number does not exist."""
    number = theme.gb(value) if value is not None else None
    return f"{number:.2f}" if number is not None else DASH


def trace_points(trace: Optional[ChartTrace]) -> List[List[Any]]:
    """One trace as [time, value] pairs, dropping unplottable samples."""
    if trace is None:
        return []
    return [
        [ms, value]
        for ms, value in zip(
            (to_ms(t) for t in trace.timestamps), trace.values
        )
        if ms is not None
    ]


def window_text(payload: ProcessDashboardPayload) -> str:
    return f"last {payload.window_len} samples"


def update_process_section(panel: Dict[str, Any], data: Any) -> None:
    if not isinstance(data, ProcessDashboardPayload) or not data.has_data:
        return

    chart = panel["chart"]
    ram_points = trace_points(data.chart.ram_percent if data.chart else None)
    gpu_points = trace_points(data.chart.gpu_percent if data.chart else None)
    chart.options["series"][0]["data"] = ram_points
    chart.options["series"][1]["data"] = gpu_points

    ymax = theme.nice_ymax(
        [v for _t, v in ram_points] + [v for _t, v in gpu_points]
    )
    chart.options["yAxis"][0]["max"] = ymax
    chart.options["yAxis"][1]["max"] = ymax
    chart.update()

    panel["win"].text = window_text(data)
    kpis = panel["kpis"]
    kpis["cpu"].content = theme.kval(
        f"{data.cpu.now:.0f}" if data.cpu else DASH, "%"
    )
    kpis["ram"].content = theme.kval(
        gb_value(data.ram.now if data.ram else None), "GB"
    )

    if not data.gpu_available:
        kpis["gmem"].content = NA_GPU
        kpis["gimb"].content = DASH
        return

    kpis["gmem"].content = theme.kval(
        gb_value(data.gpu.now if data.gpu else None), "GB"
    )
    imbalance = gb_value(data.gpu_used_imbalance_bytes)
    kpis["gimb"].content = theme.kval(
        imbalance, "GB" if imbalance != DASH else ""
    )
