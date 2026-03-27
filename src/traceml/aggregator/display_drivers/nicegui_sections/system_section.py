"""
System Metrics dashboard section (NiceGUI).

UI-only module:
- Builds the System Metrics card
- Updates it from the dashboard payload produced by the system dashboard
  compute layer

Expected payload schema
-----------------------
{
  "window_len": int,
  "gpu_available": bool,
  "rollups": {
    "gpu_available": bool,
    "cpu": {"now","p50","p95"},
    "ram": {"now","p95","total","headroom"},
    "gpu_util": {"now","p50","p95"},
    "gpu_delta": {"now","p95"},
    "gpu_mem": {"now","p95","headroom"},
    "temp": {"now","p95","status"},
    # optional:
    "status": str,
  },
  "series": {
    "x_time": [..],    # ISO-8601 timestamps
    "cpu": [..],
    "gpu_avg": [..],   # may be empty if GPU is unavailable
  }
}

Notes
-----
- This module is intentionally presentation-only. It does not compute metrics.
- The chart uses real time values on the x-axis.
- Update failures are handled silently so the dashboard keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from nicegui import ui

from traceml.utils.formatting import fmt_mem_new

METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL = "text-[12.5px] text-gray-700 leading-tight"
SUB = "text-[11px] text-gray-500 leading-tight"

_EMPTY_TEXT = "–"
_NOT_AVAILABLE_TEXT = "N/A"
_DEFAULT_STATUS = "Data missing"


def build_system_section() -> dict:
    """
    Build the static System Metrics dashboard card and return UI handles.

    Returns
    -------
    dict
        Handle map consumed by `update_system_section`.
    """
    card = ui.card().classes("m-2 p-2 w-full")
    card.style(
        """
        background: #ffffff;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        height: 360px;
        overflow-y: auto;
        overflow-x: hidden;
        """
    )

    with card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("System Metrics").classes(METRIC_TITLE).style(
                "color:#d47a00;"
            )
            with ui.icon("info").classes("text-gray-400 cursor-pointer"):
                with (
                    ui.menu()
                    .props("anchor='bottom left' self='top left' auto-close")
                    .classes("w-96 p-2")
                ):
                    ui.markdown(
                        """
                        **System Metrics (node-local)**

                        - **CPU**: host CPU usage over the rolling window.
                        - **RAM**: host memory usage over the rolling window.
                        - **GPU Util**: average across GPUs visible on this node.
                        - **GPU Skew**: max GPU util minus min GPU util on this node.
                        - **GPU Mem**: worst local GPU memory usage (max across GPUs).
                        - **Temp**: max GPU temperature on this node.
                        """
                    )
            window_text = ui.html("window: –", sanitize=False).classes(
                "text-xs text-gray-500 mr-1"
            )

        graph = _build_graph()

        with ui.grid(columns=3).classes("w-full gap-1 mt-1"):
            _, cpu_v, cpu_s = _tile("CPU (now/p50/p95)")
            _, gpu_v, gpu_s = _tile("GPU Util (now/p50/p95)")
            _, imb_v, imb_s = _tile("GPU Skew (now/p95)")
            _, ram_v, ram_s = _tile("RAM (now/p95/total)")
            _, gmem_v, gmem_s = _tile("GPU Mem (now/p95)")
            _, temp_v, temp_s = _tile("GPU Temp (now/p95)")

        status_text = ui.html("", sanitize=False).classes(
            "text-[11px] text-gray-500 mt-1 ml-1"
        )

    return {
        "window_text": window_text,
        "graph": graph,
        "cpu_v": cpu_v,
        "cpu_s": cpu_s,
        "gpu_v": gpu_v,
        "gpu_s": gpu_s,
        "imb_v": imb_v,
        "imb_s": imb_s,
        "ram_v": ram_v,
        "ram_s": ram_s,
        "gmem_v": gmem_v,
        "gmem_s": gmem_s,
        "temp_v": temp_v,
        "temp_s": temp_s,
        "status_text": status_text,
        "_has_usable_data": False,
    }


def update_system_section(panel: dict, payload: Optional[dict]) -> None:
    """
    Update the System Metrics section from one dashboard payload.

    Behavior
    --------
    - Good payload: update values and chart.
    - Partial or bad payload:
        * keep previous visible values if any good data has already been shown
        * otherwise reset to a clean empty state
    """
    try:
        payload = payload or {}
        window_len = int(payload.get("window_len", 0) or 0)
        roll = payload.get("rollups") or {}
        series = payload.get("series") or {}

        if not _is_usable_payload(window_len, roll, series):
            _handle_unusable_payload(panel, roll)
            return

        panel["window_text"].content = f"window: last {window_len} samples"
        panel["status_text"].content = _safe_status_text(roll)

        _update_tiles(panel, roll)
        _update_graph(panel, series)

        panel["_has_usable_data"] = True

    except Exception:
        _handle_render_error(panel)


def _build_graph():
    """
    Create the Plotly graph once with persistent traces and fixed layout.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="#4caf50"),
            yaxis="y",
            name="CPU",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="#ff9800"),
            yaxis="y2",
            name="GPU",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(
            type="date",
            showgrid=False,
            title=dict(text="Time"),
            tickformat="%H:%M:%S",
            hoverformat="%H:%M:%S",
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 100],
            title=dict(text="CPU (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
    )

    return ui.plotly(fig).classes("w-full")


def _tile(title: str):
    """Create a compact metric tile."""
    box = ui.column().classes("w-full px-1 py-1").style("min-height: 46px;")
    with box:
        ui.html(title, sanitize=False).classes(LABEL).style("color:#ff9800;")
        value = ui.html(_EMPTY_TEXT, sanitize=False).classes(VAL)
        subtitle = ui.html("", sanitize=False).classes(SUB)
    return box, value, subtitle


def _is_usable_payload(
    window_len: int,
    roll: Dict[str, Any],
    series: Dict[str, Any],
) -> bool:
    """
    Return True if the payload has enough structure to update the UI safely.
    """
    if window_len <= 0 or not isinstance(roll, dict) or not roll:
        return False

    x_time = series.get("x_time")
    cpu_hist = series.get("cpu")
    if not isinstance(x_time, list) or not isinstance(cpu_hist, list):
        return False
    if not x_time or not cpu_hist:
        return False

    return True


def _handle_unusable_payload(panel: dict, roll: Dict[str, Any]) -> None:
    """
    Handle missing or malformed payloads without disrupting the dashboard.
    """
    try:
        status = _safe_status_text(roll)
        panel["status_text"].content = status or _DEFAULT_STATUS

        if panel.get("_has_usable_data", False):
            return

        _reset_section(panel)
    except Exception:
        pass


def _handle_render_error(panel: dict) -> None:
    """
    Handle unexpected UI update errors.

    Preserve current visible values if possible.
    """
    try:
        if panel.get("_has_usable_data", False):
            return
        _reset_section(panel)
    except Exception:
        pass


def _safe_status_text(roll: Dict[str, Any]) -> str:
    """
    Extract an optional human-readable status string from rollups.
    """
    try:
        raw = roll.get("status")
        return str(raw) if raw else ""
    except Exception:
        return ""


def _reset_section(panel: dict) -> None:
    """
    Reset the section to a clean empty state.
    """
    panel["window_text"].content = "window: –"
    panel["status_text"].content = _DEFAULT_STATUS

    for key in ("cpu_v", "gpu_v", "imb_v", "ram_v", "gmem_v", "temp_v"):
        panel[key].content = _EMPTY_TEXT

    for key in ("cpu_s", "gpu_s", "imb_s", "ram_s", "gmem_s", "temp_s"):
        panel[key].content = ""

    _clear_graph(panel)
    panel["_has_usable_data"] = False


def _clear_graph(panel: dict) -> None:
    """
    Clear both chart traces without rebuilding the figure widget.
    """
    try:
        plot = panel["graph"]
        fig = plot.figure

        if len(fig.data) >= 2:
            fig.data[0].x, fig.data[0].y = [], []
            fig.data[1].x, fig.data[1].y = [], []

        _refresh_plot(plot, fig)
    except Exception:
        pass


def _refresh_plot(plot, fig) -> None:
    """
    Refresh the Plotly widget using the lightest available NiceGUI method.
    """
    try:
        if hasattr(plot, "update"):
            plot.update()
        else:
            plot.update_figure(fig)
    except Exception:
        pass


def _update_tiles(panel: dict, roll: Dict[str, Any]) -> None:
    """
    Update metric tiles from dashboard rollups.

    Missing sub-sections do not clear already visible good values.
    """
    try:
        cpu = roll.get("cpu")
        ram = roll.get("ram")

        if isinstance(cpu, dict):
            panel["cpu_v"].content = (
                f"{_safe_num(cpu, 'now'):.0f}% / "
                f"{_safe_num(cpu, 'p50'):.0f}% / "
                f"{_safe_num(cpu, 'p95'):.0f}%"
            )
            panel["cpu_s"].content = ""

        if isinstance(ram, dict):
            total = _safe_num(ram, "total")
            if total > 0:
                panel["ram_v"].content = (
                    f"{fmt_mem_new(_safe_num(ram, 'now'))}/"
                    f"{fmt_mem_new(_safe_num(ram, 'p95'))}/"
                    f"({fmt_mem_new(total)})"
                )
                panel["ram_s"].content = (
                    f"Headroom: {fmt_mem_new(_safe_num(ram, 'headroom'))}"
                )
            else:
                panel["ram_v"].content = _EMPTY_TEXT
                panel["ram_s"].content = ""

        if not bool(roll.get("gpu_available", False)):
            for key in ("gpu_v", "imb_v", "gmem_v", "temp_v"):
                panel[key].content = _NOT_AVAILABLE_TEXT
            for key in ("gpu_s", "imb_s", "gmem_s", "temp_s"):
                panel[key].content = ""
            return

        gpu = roll.get("gpu_util")
        imb = roll.get("gpu_delta")
        gmem = roll.get("gpu_mem")
        temp = roll.get("temp")

        if isinstance(gpu, dict):
            panel["gpu_v"].content = (
                f"{_safe_num(gpu, 'now'):.0f}% / "
                f"{_safe_num(gpu, 'p50'):.0f}% / "
                f"{_safe_num(gpu, 'p95'):.0f}%"
            )
            panel["gpu_s"].content = ""

        if isinstance(imb, dict):
            panel["imb_v"].content = (
                f"{_safe_num(imb, 'now'):.0f}% / "
                f"{_safe_num(imb, 'p95'):.0f}%"
            )
            panel["imb_s"].content = ""

        if isinstance(gmem, dict):
            panel["gmem_v"].content = (
                f"{fmt_mem_new(_safe_num(gmem, 'now'))}/"
                f"{fmt_mem_new(_safe_num(gmem, 'p95'))}"
            )
            panel["gmem_s"].content = (
                f"Headroom: {fmt_mem_new(_safe_num(gmem, 'headroom'))}"
            )

        if isinstance(temp, dict):
            panel["temp_v"].content = (
                f"{_safe_num(temp, 'now'):.0f}°C / "
                f"{_safe_num(temp, 'p95'):.0f}°C"
            )
            status = temp.get("status")
            panel["temp_s"].content = f"Status: {status}" if status else ""

    except Exception:
        pass


def _update_graph(panel: dict, series: Dict[str, Any]) -> None:
    """
    Update chart trace data on the persistent Plotly figure.

    The x-axis uses real timestamps (`x_time`) from the compute layer.
    """
    try:
        x_time, cpu_hist, gpu_hist = _extract_graph_series(series)
        if not x_time or not cpu_hist:
            return

        plot = panel["graph"]
        fig = plot.figure

        if len(fig.data) < 2:
            return

        cpu_n = min(len(x_time), len(cpu_hist))
        fig.data[0].x = x_time[:cpu_n]
        fig.data[0].y = cpu_hist[:cpu_n]

        if gpu_hist:
            gpu_n = min(len(x_time), len(gpu_hist))
            fig.data[1].x = x_time[:gpu_n]
            fig.data[1].y = gpu_hist[:gpu_n]
        else:
            fig.data[1].x, fig.data[1].y = [], []

        _refresh_plot(plot, fig)
    except Exception:
        pass


def _extract_graph_series(
    series: Dict[str, Any],
) -> Tuple[List[str], List[float], List[float]]:
    """
    Extract graph series safely from the payload.

    Returns
    -------
    tuple[list[str], list[float], list[float]]
        Time labels, CPU history, and GPU history.
    """
    x_time = _coerce_time_list(series.get("x_time"))
    cpu_hist = _coerce_float_list(series.get("cpu"))
    gpu_hist = _coerce_float_list(series.get("gpu_avg"))
    return x_time, cpu_hist, gpu_hist


def _coerce_time_list(value: Any) -> List[str]:
    """
    Coerce a payload value to a list of non-empty timestamp strings.
    """
    if not isinstance(value, list):
        return []

    out: List[str] = []
    for item in value:
        try:
            s = str(item).strip()
            if s:
                out.append(s)
        except Exception:
            continue
    return out


def _coerce_float_list(value: Any) -> List[float]:
    """
    Coerce a payload value to a list of floats, dropping bad items.
    """
    if not isinstance(value, list):
        return []

    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _safe_num(
    mapping: Dict[str, Any], key: str, default: float = 0.0
) -> float:
    """
    Read one numeric field from a mapping safely.
    """
    try:
        value = mapping.get(key, default)
        return float(value if value is not None else default)
    except Exception:
        return default
