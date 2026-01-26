"""
Model Summary (Worst vs Median)

Section that renders:
  - 3 metric cards in a 3-column grid
  - Each card shows:
      (1) a Plotly line chart with TWO curves: Worst + Median (same units)
      (2) a compact 2-row stats table under the chart:
            Worst:  Last, p50, p95, Avg(100), Trend, Imbalance, Worst Rank
            Median: Last, p50, p95, Avg(100), Trend (optional), —, —

Expected input telemetry (from ModelCombinedRenderer.get_dashboard_renderable()):
  telemetry: Dict[str, Any] shaped like:
    {
      "dataLoader_fetch": {
        "steps": List[int],
        "worst":  { "y": np.ndarray, "stats": {last,p50,p95,avg100,trend} },
        "median": { "y": np.ndarray, "stats": {last,p50,p95,avg100,trend} },
        "rank_skew_abs": float,
        "rank_skew_pct": float,      # 0..1
        "slowest_rank": Optional[int]
      },
      "step_time": {...},
      "step_gpu_memory": {...}
    }

Notes
-----
- We plot the per-step aggregated series (already step-synchronized via renderer).
- We show imbalance + worst rank only for the "Worst" row (because it refers to gating rank).
- Card size and graph height remain unchanged (card 300px; graph 160px).
- Safe with numpy arrays: never uses `or []` on a ndarray.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Callable, Tuple

from nicegui import ui
import plotly.graph_objects as go

from traceml.renderers.utils import fmt_time_run, fmt_mem_new


METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
METRIC_TEXT = "text-sm leading-normal text-gray-700"


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert to float safely; return default on failure."""
    try:
        return float(v)
    except Exception:
        return default


def _fmt_pct(v01: float) -> str:
    """Format a 0..1 fraction as percent string."""
    return f"{v01 * 100.0:.1f}%"


def _trend_badge(trend: str) -> str:
    """
    Trend emitted by renderer as "+x.x%", "-x.x%", "≈0%" or "".
    Returns small HTML badge.
    """
    t = (trend or "").strip()
    if not t:
        return "<span style='color:#888;'>—</span>"

    # Subtle but readable
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


def _to_list(v: Any) -> List[Any]:
    """
    Convert numpy arrays / iterables to list.
    - None => []
    - ndarray => list(ndarray)
    - list/tuple => list(...)
    """
    if v is None:
        return []
    try:
        return list(v)
    except Exception:
        return []


@dataclass(frozen=True)
class SeriesStats:
    last: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    avg100: float = 0.0
    trend: str = ""


@dataclass(frozen=True)
class AggregatedMetric:
    steps: List[int]
    worst_y: List[float]
    median_y: List[float]
    worst_stats: SeriesStats
    median_stats: SeriesStats
    rank_skew_abs: float
    rank_skew_pct: float  # 0..1
    slowest_rank: Optional[int]


def _parse_series_stats(d: Dict[str, Any]) -> SeriesStats:
    """Parse stats dict coming from renderer."""
    return SeriesStats(
        last=_safe_float(d.get("last", 0.0)),
        p50=_safe_float(d.get("p50", 0.0)),
        p95=_safe_float(d.get("p95", 0.0)),
        avg100=_safe_float(d.get("avg100", 0.0)),
        trend=str(d.get("trend", "") or ""),
    )


def _parse_metric(tlm: Dict[str, Any]) -> Optional[AggregatedMetric]:
    """
    Convert renderer metric dict into a normalized object for UI rendering.

    Returns None if tlm is missing required fields.
    """
    if not isinstance(tlm, dict):
        return None

    steps = _to_list(tlm.get("steps"))
    worst = tlm.get("worst", {}) or {}
    median = tlm.get("median", {}) or {}

    worst_y = _to_list(worst.get("y"))
    median_y = _to_list(median.get("y"))

    # Must have steps and at least one series to plot
    if not steps or (not worst_y and not median_y):
        return None

    # Keep arrays aligned defensively (clip to min length)
    n = min(len(steps), len(worst_y) if worst_y else len(steps), len(median_y) if median_y else len(steps))
    steps = steps[:n]
    if worst_y:
        worst_y = worst_y[:n]
    else:
        worst_y = [0.0] * n
    if median_y:
        median_y = median_y[:n]
    else:
        median_y = [0.0] * n

    return AggregatedMetric(
        steps=[int(s) for s in steps],
        worst_y=[_safe_float(v) for v in worst_y],
        median_y=[_safe_float(v) for v in median_y],
        worst_stats=_parse_series_stats(worst.get("stats", {}) or {}),
        median_stats=_parse_series_stats(median.get("stats", {}) or {}),
        rank_skew_abs=_safe_float(tlm.get("rank_skew_abs", 0.0)),
        rank_skew_pct=_safe_float(tlm.get("rank_skew_pct", 0.0)),
        slowest_rank=tlm.get("slowest_rank", None),
    )


def _make_empty_figure() -> go.Figure:
    """Create a baseline empty figure with your current sizing."""
    fig = go.Figure()
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=9),
        ),
        xaxis=dict(showgrid=False, title="Training Step", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def _update_metric_graph(plotly_ui, x: List[int], y_worst: List[float], y_median: List[float], y_label: str) -> None:
    """Update Plotly graph with worst + median curves (same units)."""
    fig = go.Figure()

    # Worst curve (thicker)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_worst,
            mode="lines",
            name="Worst",
            line=dict(width=2),
        )
    )

    # Median curve (dashed, slightly thinner)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_median,
            mode="lines",
            name="Median",
            line=dict(width=1, dash="dash"),
        )
    )

    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=9),
        ),
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

    plotly_ui.update_figure(fig)


def _stats_table_html_dual(
    worst: SeriesStats,
    median: SeriesStats,
    value_fmt: Callable[[float], str],
    skew_fmt: Callable[[float], str],
    skew_abs: float,
    skew_pct: float,
    slowest_rank: Optional[int],
    show_median_trend: bool = False,  # default off to reduce noise
) -> str:
    """
    Render a compact 2-row stats table.
    - Worst row includes imbalance + worst rank.
    - Median row shows '—' for those fields.
    """
    def fmt_row(label: str, s: SeriesStats, show_skew: bool) -> str:
        last = value_fmt(s.last)
        p50 = value_fmt(s.p50)
        p95 = value_fmt(s.p95)
        avg = value_fmt(s.avg100)
        trend_html = _trend_badge(s.trend) if (label == "Worst" or show_median_trend) else "<span style='color:#888;'>—</span>"

        if show_skew:
            imbalance = f"{skew_fmt(skew_abs)} ({_fmt_pct(skew_pct)})"
            rank = f"r{slowest_rank}" if slowest_rank is not None else "—"
        else:
            imbalance = "—"
            rank = "—"

        return f"""
        <tr>
          <td style="text-align:left; padding:5px 6px;"><b>{label}</b></td>
          <td style="text-align:right; padding:5px 6px;">{last}</td>
          <td style="text-align:right; padding:5px 6px;">{p50}</td>
          <td style="text-align:right; padding:5px 6px;">{p95}</td>
          <td style="text-align:right; padding:5px 6px;">{avg}</td>
          <td style="text-align:center; padding:5px 6px;">{trend_html}</td>
          <td style="text-align:center; padding:5px 6px;">{imbalance}</td>
          <td style="text-align:center; padding:5px 6px;">{rank}</td>
        </tr>
        """

    return f"""
    <div style="margin-top:6px;">
      <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
        <thead>
          <tr style="border-bottom:1px solid #e0e0e0;">
            <th style="text-align:left; padding:4px 6px;">Series</th>
            <th style="text-align:right; padding:4px 6px;">Last</th>
            <th style="text-align:right; padding:4px 6px;">p50</th>
            <th style="text-align:right; padding:4px 6px;">p95</th>
            <th style="text-align:right; padding:4px 6px;">Avg(100)</th>
            <th style="text-align:center; padding:4px 6px;">Trend</th>
            <th style="text-align:center; padding:4px 6px;">Imbalance</th>
            <th style="text-align:center; padding:4px 6px;">Worst Rank</th>
          </tr>
        </thead>
        <tbody>
          {fmt_row("Worst", worst, show_skew=True)}
          {fmt_row("Median", median, show_skew=False)}
        </tbody>
      </table>
    </div>
    """


def _build_metric_card(title: str) -> Dict[str, Any]:
    """
    Create a single metric card:
      - title label
      - plotly graph
      - stats HTML block
    Returns handles needed for updates.
    """
    card = ui.card().classes("p-2 w-full")
    card.style(
        """
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 10px rgba(0,0,0,0.10);
        height: 360px;
        """
    )

    with card:
        ui.label(title).classes(METRIC_TITLE).style("color:#d47a00;")
        plotly_ui = ui.plotly(_make_empty_figure()).classes("w-full")
        stats_html = ui.html("", sanitize=False).classes(METRIC_TEXT).style("color:#333")

    return {"card": card, "graph": plotly_ui, "stats_html": stats_html}


def build_model_combined_section() -> Dict[str, Any]:
    """
    Build a 3-up grid section:
      - Dataloader Fetch
      - Training Step Time
      - GPU Step Memory

    Returns a dict of "handles" to be mutated by update_model_combined_section(...).
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



def update_model_combined_section(panel: Dict[str, Any], telemetry: Optional[Dict[str, Any]]) -> None:
    """
    Update the model summary cards from ModelCombinedRenderer telemetry.

    This is intentionally defensive:
      - handles missing metrics gracefully
      - converts numpy arrays to lists safely
      - preserves card + graph sizes (no layout changes)
    """
    if not telemetry or not isinstance(telemetry, dict):
        return

    # Map UI card keys -> renderer metric keys + y labels + formatters
    mapping: Dict[str, Tuple[str, str, Callable[[float], str], Callable[[float], str]]] = {
        "dataloader": ("dataLoader_fetch", "Time (ms)", fmt_time_run, fmt_time_run),
        "step_time": ("step_time", "Time (ms)", fmt_time_run, fmt_time_run),
        "step_memory": ("step_gpu_memory", "Memory (MB)", fmt_mem_new, fmt_mem_new),
    }

    for card_key, (metric_name, y_label, value_fmt, skew_fmt) in mapping.items():
        card_handles = panel.get(card_key)
        if not card_handles:
            continue

        tlm_raw = telemetry.get(metric_name)
        metric = _parse_metric(tlm_raw) if tlm_raw else None
        if metric is None:
            continue

        # Update chart with two curves (worst + median)
        _update_metric_graph(
            card_handles["graph"],
            x=metric.steps,
            y_worst=metric.worst_y,
            y_median=metric.median_y,
            y_label=y_label,
        )

        # Update two-row stats table
        card_handles["stats_html"].content = _stats_table_html_dual(
            worst=metric.worst_stats,
            median=metric.median_stats,
            value_fmt=value_fmt,
            skew_fmt=skew_fmt,
            skew_abs=metric.rank_skew_abs,
            skew_pct=metric.rank_skew_pct,
            slowest_rank=metric.slowest_rank,
            show_median_trend=False,  # keep Median trend off by default (less noise)
        )
