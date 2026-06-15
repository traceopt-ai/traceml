"""Process metrics — RAM% / GPU-mem% time-series + KPIs (PR2 revamp).

Reuses the original client-side rollup math; renders with glass + ECharts.
Renderer payload (history rows + series + gpu_used_imbalance) is unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from nicegui import ui

from . import theme


def _to_ms(s: str) -> Optional[int]:
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        return None


def _percentile(vals: List[float], p: float) -> float:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(int(k) + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] * (c - k) + vals[c] * (k - f)


def _compute_rollups(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    last = window[-1]
    cpu_hist = [float(r.get("cpu_max", 0.0) or 0.0) for r in window]
    ram_hist = [float(r.get("ram_used_max", 0.0) or 0.0) for r in window]
    gpu_hist = [
        float(r.get("gpu_used", 0.0) or 0.0)
        for r in window
        if r.get("gpu_used") is not None
    ]
    return {
        "gpu_available": last.get("gpu_used") is not None,
        "cpu": {
            "now": cpu_hist[-1],
            "p50": _percentile(cpu_hist, 50),
            "p95": _percentile(cpu_hist, 95),
        },
        "ram": {
            "now": ram_hist[-1],
            "p95": _percentile(ram_hist, 95),
            "total": float(last.get("ram_total", 0.0) or 0.0),
        },
        "gpu": {
            "now": gpu_hist[-1] if gpu_hist else 0.0,
            "p95": _percentile(gpu_hist, 95) if gpu_hist else 0.0,
        },
    }


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
            .style("gap:9px; margin-top:12px; flex-wrap:wrap;")
        ):
            for key, lab, acc in [
                ("cpu", "CPU", theme.C_CPU),
                ("ram", "RAM", theme.C_CPU),
                ("gmem", "GPU MEM", theme.C_GPU),
                ("gimb", "GPU IMBAL", theme.C_GPU),
            ]:
                with (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{acc}; min-width:102px;")
                ):
                    ui.label(lab).classes("klab")
                    kpis[key] = ui.html("—").classes("kval")
    return {"chart": chart, "win": win, "kpis": kpis}


def update_process_section(
    panel: Dict[str, Any], data: Dict[str, Any]
) -> None:
    if not isinstance(data, dict):
        return
    history = data.get("history", []) or []
    if not history:
        return
    window = history[-100:]
    series = data.get("series", {}) or {}
    x_time = series.get("x_time", []) or []

    ram_total = max(float(window[-1].get("ram_total", 1.0) or 1.0), 1.0)
    ram_pct = [
        (float(r.get("ram_used_max", 0.0) or 0.0) / ram_total) * 100.0
        for r in window
    ]
    if x_time and len(x_time) >= len(window):
        xms = [_to_ms(s) for s in x_time[-len(window) :]]
    else:
        xms = [None] * len(window)

    chart = panel["chart"]
    chart.options["series"][0]["data"] = [
        [t, v] for t, v in zip(xms, ram_pct) if t is not None
    ]

    gpu_window = [
        (i, r) for i, r in enumerate(window) if r.get("gpu_used") is not None
    ]
    if gpu_window and window[-1].get("gpu_total") is not None:
        gtot = max(float(window[-1].get("gpu_total", 1.0) or 1.0), 1.0)
        gdata = [
            [xms[i], (float(r.get("gpu_used", 0.0) or 0.0) / gtot) * 100.0]
            for i, r in gpu_window
            if i < len(xms) and xms[i] is not None
        ]
        chart.options["series"][1]["data"] = gdata
    else:
        chart.options["series"][1]["data"] = []
    chart.update()

    roll = _compute_rollups(window)
    panel["win"].text = f"last {len(window)} samples"
    k = panel["kpis"]
    k["cpu"].content = theme.kval(f"{roll['cpu']['now']:.0f}", "%")
    rn = theme.gb(roll["ram"]["now"])
    k["ram"].content = theme.kval(f"{rn:.2f}" if rn is not None else "—", "GB")
    if roll["gpu_available"]:
        gm = theme.gb(roll["gpu"]["now"])
        k["gmem"].content = theme.kval(
            f"{gm:.2f}" if gm is not None else "—", "GB"
        )
        imb = data.get("gpu_used_imbalance")
        gi = theme.gb(imb) if imb is not None else None
        k["gimb"].content = theme.kval(
            f"{gi:.2f}" if gi is not None else "—",
            "GB" if gi is not None else "",
        )
    else:
        k["gmem"].content = "N/A"
        k["gimb"].content = "—"
