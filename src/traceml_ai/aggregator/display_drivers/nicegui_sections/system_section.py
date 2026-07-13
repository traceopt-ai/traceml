"""System metrics — CPU/GPU utilization time-series + a GPU-util gauge (PR2).

Two subscribers on the same System payload (the driver's multi-subscriber
registry): the dual-line chart card and the gauge card. Payload unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from nicegui import ui

from . import theme


def _to_ms(s: str) -> Optional[int]:
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        return None


def build_system_section() -> Dict[str, Any]:
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
            ui.label("System").classes("ctitle")
            for nm, col in [("CPU", theme.C_CPU), ("GPU", theme.C_GPU)]:
                with ui.element("div").classes("legchip"):
                    ui.element("div").classes("legdot").style(
                        f"background:{col};"
                    )
                    ui.label(nm)
            ui.element("div").style("flex:1;")
            win = ui.label("waiting for data").classes("cmeta")
        chart = ui.echart(theme.dual_line_options("CPU", "GPU")).style(
            "height:210px; width:100%; flex:1; min-height:170px;"
        )
        with (
            ui.row()
            .classes("w-full")
            .style("gap:9px; margin-top:12px; flex-wrap:wrap;")
        ):
            for key, lab, acc, qual in [
                ("cpu", "CPU", theme.C_CPU, "now"),
                ("ram", "RAM", theme.C_CPU, "now"),
                ("gmem", "GPU MEM", theme.C_GPU, "worst gpu"),
                ("gtemp", "GPU TEMP", theme.C_GPU, "hottest"),
            ]:
                with (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{acc}; min-width:104px;")
                ):
                    ui.html(
                        f"{lab} <span class='kq'>{qual}</span>",
                        sanitize=False,
                    ).classes("klab")
                    kpis[key] = ui.html("—", sanitize=False).classes("kval")
    return {"chart": chart, "win": win, "kpis": kpis}


def update_system_section(panel: Dict[str, Any], data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    series = data.get("series", {}) or {}
    roll = data.get("rollups", {}) or {}
    xs = series.get("x_time", []) or []
    cpu = series.get("cpu", []) or []
    gpu = series.get("gpu_avg", []) or []
    xms = [_to_ms(s) for s in xs]
    chart = panel["chart"]
    chart.options["series"][0]["data"] = [
        [t, v] for t, v in zip(xms, cpu) if t is not None
    ]
    chart.options["series"][1]["data"] = (
        [[t, v] for t, v in zip(xms, gpu) if t is not None] if gpu else []
    )
    ymax = theme.nice_ymax(list(cpu) + list(gpu))
    chart.options["yAxis"][0]["max"] = ymax
    chart.options["yAxis"][1]["max"] = ymax
    chart.update()

    wl = data.get("window_len", 0)
    panel["win"].text = f"last {wl} samples" if wl else "waiting for data"
    k = panel["kpis"]
    c = roll.get("cpu", {}) or {}
    r = roll.get("ram", {}) or {}
    k["cpu"].content = theme.kval(f"{c.get('now', 0):.0f}", "%")
    ram_now = theme.gb(r.get("now"))
    k["ram"].content = theme.kval(
        f"{ram_now:.2f}" if ram_now is not None else "—", "GB"
    )
    if roll.get("gpu_available", False):
        gm = theme.gb((roll.get("gpu_mem", {}) or {}).get("now"))
        gt = (roll.get("temp", {}) or {}).get("now", 0)
        k["gmem"].content = theme.kval(
            f"{gm:.2f}" if gm is not None else "—", "GB"
        )
        k["gtemp"].content = theme.kval(f"{gt:.0f}", "°C")
    else:
        k["gmem"].content = "N/A"
        k["gtemp"].content = "N/A"


# --- GPU utilization gauge (second subscriber on the System payload) ------
def build_gpu_gauge_section() -> Dict[str, Any]:
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow:hidden;"
    )
    with card:
        ui.html(
            "GPU utilization <span class='kq'>avg · now</span>",
            sanitize=False,
        ).classes("ctitle")
        chart = ui.echart(theme.gauge_options()).style(
            "height:180px; width:100%; flex:1; min-height:150px;"
        )
        sub = ui.label("").style(
            "font-family:var(--mono); font-size:11px; color:var(--muted); "
            "text-align:center; margin-top:-4px;"
        )
    return {"chart": chart, "sub": sub}


def update_gpu_gauge_section(
    panel: Dict[str, Any], data: Dict[str, Any]
) -> None:
    if not isinstance(data, dict):
        return
    roll = data.get("rollups", {}) or {}
    chart = panel["chart"]
    if roll.get("gpu_available", False):
        gu = (roll.get("gpu_util", {}) or {}).get("now", 0)
        gm = theme.gb((roll.get("gpu_mem", {}) or {}).get("now"))
        gt = (roll.get("temp", {}) or {}).get("now", 0)
        chart.options["series"][0]["data"][0]["value"] = float(gu)
        chart.update()
        panel["sub"].text = (
            f"{gm:.1f} GB · {gt:.0f}°C" if gm is not None else f"{gt:.0f}°C"
        )
    else:
        chart.options["series"][0]["data"][0]["value"] = 0
        chart.update()
        panel["sub"].text = "no GPU"
