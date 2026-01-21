"""
NiceGUI: Model Summary (Combined)

Renders three per-step charts + a compact stats table under each chart.

Inputs:
  telemetry = ModelCombinedRenderer.get_dashboard_renderable(), shape:
    {
      "dataLoader_fetch": {
        "x": np.ndarray,
        "y": np.ndarray,           # ms
        "stats": {
          "last": float, "p50": float, "p95": float, "avg100": float, "trend": str,
          "rank_skew_abs": float, "rank_skew_pct": float, "slowest_rank": Optional[int],
        }
      },
      "step_time": {...},          # ms
      "step_gpu_memory": {...},    # MB (renderer currently returns MB)
    }

Semantics:
  - Series are already aggregated in the renderer ("worst-rank wins").
  - rank_skew_abs / rank_skew_pct reflect imbalance at the latest completed step.
  - slowest_rank is the rank that was worst at the latest completed step.
"""

from typing import Any, Dict, Optional

from nicegui import ui
import plotly.graph_objects as go

from traceml.renderers.utils import fmt_time_run, fmt_mem_new


METRIC_TEXT = "text-sm leading-normal text-gray-700"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"

LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL = "text-[12.5px] text-gray-700 leading-tight"
SUB = "text-[11px] text-gray-500 leading-tight"


def _build_metric_card(title: str) -> Dict[str, Any]:
    """One metric card: title + graph + stats html block."""
    card = ui.card().classes("p-2 w-full")
    card.style(
        """
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 10px rgba(0,0,0,0.10);
        height: 300px;
        """
    )

    with card:
        ui.label(title).classes(METRIC_TITLE).style("color:#d47a00;")
        graph = _build_graph()
        stats_html = ui.html("", sanitize=False).classes(METRIC_TEXT).style("color:#333")

    return {"card": card, "graph": graph, "stats_html": stats_html}


def _build_graph():
    fig = go.Figure()
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(showgrid=False, title="Training Step", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return ui.plotly(fig).classes("w-full")


def build_model_combined_section():
    """
    Build the 3-up grid: DataLoader / Step Time / GPU Step Memory.
    Returns a dict of card handles so update_model_summary_section can mutate them.
    """
    with ui.grid(columns=3).classes("m-2 w-full gap-x-3"):
        dataloader_card = _build_metric_card("Dataloader Fetch")
        step_time_card = _build_metric_card("Training Step Time")
        step_mem_card = _build_metric_card("GPU Step Memory")

    return {
        "dataloader": dataloader_card,
        "step_time": step_time_card,
        "step_memory": step_mem_card,
    }


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:.1f}%"


def _trend_badge(trend: str) -> str:
    """
    Trend is emitted by renderer as "+x.x%", "-x.x%", "≈0%" or "".
    Keep it subtle and readable.
    """
    t = (trend or "").strip()
    if not t:
        return "<span style='color:#888;'>—</span>"

    color = "#666"
    prefix = ""
    if t.startswith("+"):
        color = "#d32f2f"  # regression
        prefix = "↑ "
    elif t.startswith("-"):
        color = "#2e7d32"  # improvement
        prefix = "↓ "
    elif "≈" in t:
        color = "#666"

    return f"<span style='color:{color}; font-weight:700;'>{prefix}{t}</span>"


def _stats_table_html(
    stats: Dict[str, Any],
    value_fmt,
    skew_fmt,  # how to format skew_abs (time or mem)
) -> str:
    """
    Pretty compact stats block (like your process section philosophy):
      Last | p50 | p95 | Avg(100) | Trend
      Imbalance (abs / pct) | Worst rank
    """
    last = value_fmt(_safe_float(stats.get("last", 0.0)))
    p50 = value_fmt(_safe_float(stats.get("p50", 0.0)))
    p95 = value_fmt(_safe_float(stats.get("p95", 0.0)))
    avg = value_fmt(_safe_float(stats.get("avg100", 0.0)))
    trend = _trend_badge(str(stats.get("trend", "") or ""))

    skew_abs = _safe_float(stats.get("rank_skew_abs", 0.0))
    skew_pct = _safe_float(stats.get("rank_skew_pct", 0.0))
    slowest = stats.get("slowest_rank", None)
    slowest = str(slowest) if slowest is not None else "—"

    imbalance = f"{skew_fmt(skew_abs)} ({_fmt_pct(skew_pct)})"

    return f"""
    <div style="margin-top:6px;">
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr style="border-bottom:1px solid #e0e0e0;">
            <th style="text-align:left; padding:4px 6px;">Last</th>
            <th style="text-align:right; padding:4px 6px;">p50</th>
            <th style="text-align:right; padding:4px 6px;">p95</th>
            <th style="text-align:right; padding:4px 6px;">Avg(100)</th>
            <th style="text-align:center; padding:4px 6px;">Trend</th>
            <th style="text-align:center; padding:4px 6px;">Imbalance</th>
             <th style="text-align:center; padding:4px 6px;">Worst Rank</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="text-align:left; padding:6px;">{last}</td>
            <td style="text-align:right; padding:6px;">{p50}</td>
            <td style="text-align:right; padding:6px;">{p95}</td>
            <td style="text-align:right; padding:6px;">{avg}</td>
            <td style="text-align:center; padding:6px;">{trend}</td>
            <td style="text-align:center; padding:6px;">{imbalance}</td>
            <td style="text-align:center; padding:6px;">{slowest}</td>
          </tr>
        </tbody>
      </table>
    </div>
    """


def _update_graph(graph, x, y, y_label: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2)))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        xaxis=dict(
            title=dict(text="Training Step", font=dict(color="#4caf50")),
            showgrid=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#4caf50")),
            tickfont=dict(size=9),
        ),
    )
    graph.update_figure(fig)


def update_model_combined_section(panel: Dict[str, Any], telemetry: Dict[str, Any]):
    """
    Update the model summary cards from ModelCombinedRenderer telemetry.
    Safe for numpy arrays (no `or []` on arrays).
    """
    if telemetry is None:
        return

    mapping = {
        "dataloader": ("dataLoader_fetch", "Time (ms)", fmt_time_run, fmt_time_run),
        "step_time": ("step_time", "Time (ms)", fmt_time_run, fmt_time_run),
        "step_memory": ("step_gpu_memory", "Memory (MB)", fmt_mem_new, fmt_mem_new),
    }

    for key, (metric_name, y_label, value_fmt, skew_fmt) in mapping.items():
        entry = panel.get(key)
        tlm = telemetry.get(metric_name)

        if entry is None or tlm is None:
            continue

        x_raw = tlm.get("x", None)
        y_raw = tlm.get("y", None)
        stats = tlm.get("stats", {}) or {}

        # Convert x/y safely (works for numpy arrays + lists)
        x = [] if x_raw is None else list(x_raw)
        y = [] if y_raw is None else list(y_raw)

        _update_graph(entry["graph"], x, y, y_label)
        entry["stats_html"].content = _stats_table_html(stats, value_fmt, skew_fmt)