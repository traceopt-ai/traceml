"""
Compact Step Memory analysis section for the overview dashboard.

This section stays focused:
- one merged line chart for worst vs median
- one compact KPI strip

It keeps the last good rendering during transient payload gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from nicegui import ui

from traceml.diagnostics.trends import compute_trend_pct

from .ui_shell import CARD_STYLE, compact_metric_html, safe_mem, safe_pct

BYTES_PER_GB = 1e9


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_list(v: Any) -> List[Any]:
    if v is None:
        return []
    try:
        return list(v)
    except Exception:
        return []


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    pos = (q / 100.0) * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _avg_last_k(xs: List[float], k: int = 100) -> float:
    if not xs:
        return 0.0
    tail = xs[-k:] if len(xs) > k else xs
    return float(sum(float(x) for x in tail) / max(1, len(tail)))


def _bytes_to_gb_list(values: List[float]) -> List[float]:
    return [float(v) / BYTES_PER_GB for v in values] if values else []


@dataclass(frozen=True)
class SeriesStats:
    last_bytes: float
    p50_bytes: float
    p95_bytes: float
    avg100_bytes: float
    trend_pct: float


@dataclass(frozen=True)
class StepMemoryMetricView:
    steps: List[int]
    worst_y_gb: List[float]
    median_y_gb: List[float]
    worst_stats: SeriesStats
    median_stats: SeriesStats
    skew_pct: float
    worst_rank: Optional[int]


def build_step_memory_section(
    *, title: str = "Step Memory Analysis"
) -> Dict[str, Any]:
    """Build the Step Memory analysis card."""
    card = ui.card().classes("w-full h-full p-3")
    card.style(
        CARD_STYLE + "height: 100%; overflow-y: auto; overflow-x: hidden;"
    )

    fig = _make_figure()

    with card:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label(title).classes("text-sm font-bold").style(
                "color:#d47a00;"
            )
            window_text = ui.html("window: -", sanitize=False).classes(
                "text-[11px] text-gray-500"
            )

        graph = ui.plotly(fig).classes("w-full")
        kpis = ui.html("", sanitize=False).classes("mt-2")
        empty_hint = ui.html(
            "<div style='text-align:center; padding:10px; color:#888; font-style:italic;'>"
            "Waiting for memory samples...</div>",
            sanitize=False,
        )

    return {
        "graph": graph,
        "window_text": window_text,
        "kpis": kpis,
        "empty_hint": empty_hint,
        "_fig": fig,
        "_last_ok_view": None,
        "_last_kpis": None,
        "_last_signature": None,
        "_last_window_text": None,
    }


def update_step_memory_section(
    panel: Dict[str, Any],
    payload: Any,
    *,
    prefer_metric: str = "peak_allocated",
) -> None:
    """Update the Step Memory card with stale-value fallback."""
    try:
        metric = _select_metric(payload, preferred=prefer_metric)
        view = _normalize_metric(metric)
        if view is None:
            view = panel.get("_last_ok_view")

        if view is None:
            return

        panel["_last_ok_view"] = view
        panel["empty_hint"].content = ""

        signature = (
            tuple(round(v, 5) for v in view.worst_y_gb[-12:]),
            tuple(round(v, 5) for v in view.median_y_gb[-12:]),
            round(view.skew_pct, 5),
            int(view.worst_rank or -1),
            len(view.steps),
        )

        if panel.get("_last_signature") != signature:
            _update_graph(
                panel["graph"],
                panel["_fig"],
                view.steps,
                view.worst_y_gb,
                view.median_y_gb,
            )
            panel["_last_signature"] = signature

        window_text = f"window: {len(view.steps)} aligned steps"
        if panel.get("_last_window_text") != window_text:
            panel["window_text"].content = window_text
            panel["_last_window_text"] = window_text

        kpis_html = _render_kpis(view)
        if panel.get("_last_kpis") != kpis_html:
            panel["kpis"].content = kpis_html
            panel["_last_kpis"] = kpis_html

    except Exception:
        pass


def _make_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Worst",
            line=dict(width=2, color="#4f67ff"),
            hovertemplate="step=%{x}<br>worst=%{y:.2f} GB<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Median",
            line=dict(width=1, dash="dash", color="#ff7e67"),
            hovertemplate="step=%{x}<br>median=%{y:.2f} GB<extra></extra>",
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=8, r=8, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, title="Training Step"),
        yaxis=dict(title="Memory (GB)"),
    )
    return fig


def _update_graph(
    plotly_ui,
    fig: go.Figure,
    x: List[int],
    y_worst: List[float],
    y_median: List[float],
) -> None:
    fig.data[0].x = x
    fig.data[0].y = y_worst
    fig.data[1].x = x
    fig.data[1].y = y_median
    plotly_ui.update_figure(fig)


def _select_metric(payload: Any, preferred: str) -> Optional[Any]:
    metrics = getattr(payload, "metrics", None)
    if metrics is None and isinstance(payload, dict):
        metrics = payload.get("metrics")
    if not metrics:
        return None

    for metric in metrics:
        key = (
            getattr(metric, "metric", None)
            if not isinstance(metric, dict)
            else metric.get("metric")
        )
        if key == preferred:
            return metric

    return metrics[0] if metrics else None


def _normalize_metric(metric: Any) -> Optional[StepMemoryMetricView]:
    if metric is None:
        return None

    series = (
        getattr(metric, "series", None)
        if not isinstance(metric, dict)
        else metric.get("series", {})
    )
    summary = (
        getattr(metric, "summary", None)
        if not isinstance(metric, dict)
        else metric.get("summary", {})
    )

    steps = _to_list(
        getattr(series, "steps", None)
        if not isinstance(series, dict)
        else series.get("steps")
    )
    worst_b = _to_list(
        getattr(series, "worst", None)
        if not isinstance(series, dict)
        else series.get("worst")
    )
    median_b = _to_list(
        getattr(series, "median", None)
        if not isinstance(series, dict)
        else series.get("median")
    )

    if not steps or not worst_b or not median_b:
        return None

    n = min(len(steps), len(worst_b), len(median_b))
    if n <= 0:
        return None

    steps = [int(_safe_float(v, 0.0)) for v in steps[:n]]
    worst_b = [_safe_float(v) for v in worst_b[:n]]
    median_b = [_safe_float(v) for v in median_b[:n]]

    worst_y_gb = _bytes_to_gb_list(worst_b)
    median_y_gb = _bytes_to_gb_list(median_b)

    worst_stats = _compute_stats(worst_b)
    median_stats = _compute_stats(median_b)

    skew_pct = (
        _safe_float(getattr(summary, "skew_pct", None))
        if not isinstance(summary, dict)
        else _safe_float(summary.get("skew_pct", 0.0))
    )
    worst_rank = (
        getattr(summary, "worst_rank", None)
        if not isinstance(summary, dict)
        else summary.get("worst_rank")
    )

    return StepMemoryMetricView(
        steps=steps,
        worst_y_gb=worst_y_gb,
        median_y_gb=median_y_gb,
        worst_stats=worst_stats,
        median_stats=median_stats,
        skew_pct=float(skew_pct),
        worst_rank=(
            worst_rank
            if worst_rank is None
            else int(_safe_float(worst_rank, 0.0))
        ),
    )


def _compute_stats(values_bytes: List[float]) -> SeriesStats:
    if not values_bytes:
        return SeriesStats(
            last_bytes=0.0,
            p50_bytes=0.0,
            p95_bytes=0.0,
            avg100_bytes=0.0,
            trend_pct=0.0,
        )

    trend_pct = compute_trend_pct(values_bytes)
    if trend_pct is None:
        trend_pct = 0.0

    return SeriesStats(
        last_bytes=float(values_bytes[-1]),
        p50_bytes=float(_percentile(values_bytes, 50.0)),
        p95_bytes=float(_percentile(values_bytes, 95.0)),
        avg100_bytes=float(_avg_last_k(values_bytes, 100)),
        trend_pct=float(trend_pct),
    )


def _render_kpis(view: StepMemoryMetricView) -> str:
    items = [
        compact_metric_html(
            "Worst Last", safe_mem(view.worst_stats.last_bytes)
        ),
        compact_metric_html(
            "Median Last", safe_mem(view.median_stats.last_bytes)
        ),
        compact_metric_html("Worst p95", safe_mem(view.worst_stats.p95_bytes)),
        compact_metric_html(
            "Avg(100)", safe_mem(view.worst_stats.avg100_bytes)
        ),
        compact_metric_html("Trend", safe_pct(view.worst_stats.trend_pct)),
        compact_metric_html("Imbalance", safe_pct(view.skew_pct)),
        compact_metric_html(
            "Worst Rank",
            f"r{view.worst_rank}" if view.worst_rank is not None else "-",
        ),
    ]

    return (
        "<div style='display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); "
        "gap:8px; padding-top:6px; border-top:1px solid #ececec;'>"
        + "".join(items)
        + "</div>"
    )
