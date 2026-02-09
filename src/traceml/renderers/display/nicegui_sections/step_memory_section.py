"""
NiceGUI Step Memory Summary (Worst vs Median)

This section replaces the old "user_time" / step timing table with a
**step-level peak memory** view derived from the StepMemoryCombinedComputer
renderer payload.

UI
--
Single card (same height as your other 360px cards):
  - Plotly line chart with TWO curves: Worst + Median (bytes)
  - Compact 2-row stats table:
        Worst:  Last, p50, p95, Avg(100), Trend, Imbalance, Worst Rank
        Median: Last, p50, p95, Avg(100), Trend (optional), —, —

Expected input (`payload`)
--------------------------
payload is StepMemoryCombinedResult (renderer-facing dataclasses) OR a dict
with the same shape.

We assume the compute result contains per-metric:
  - metric.metric in {"peak_allocated", "peak_reserved"} (bytes)
  - series.steps: List[int]
  - series.worst: List[float]
  - series.median: List[float]
  - summary.skew_pct, summary.worst_rank
  - coverage (optional for display)

Notes
-----
- We plot already step-aligned series (aligned across ranks in compute).
- We never "or []" a numpy array; we convert via list() defensively.
- Units: bytes (formatted with fmt_mem_new).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui
import plotly.graph_objects as go

from traceml.utils.formatting import fmt_mem_new


# -----------------------------
# Styling (match your dashboard)
# -----------------------------
CARD_STYLE = """
background: #ffffff;
backdrop-filter: blur(12px);
border-radius: 14px;
border: 1px solid rgba(255,255,255,0.25);
box-shadow: 0 4px 12px rgba(0,0,0,0.12);
"""

TITLE_STYLE = "color:#d47a00;"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
METRIC_TEXT = "text-sm leading-normal text-gray-700"


# -----------------------------
# Helpers
# -----------------------------
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


def _fmt_pct(v01: float) -> str:
    return f"{v01 * 100.0:.1f}%"


def _trend_badge(trend: str) -> str:
    """
    Trend string emitted by UI (we compute here) as "+x.x%", "-x.x%", "≈0%".
    Returns small HTML badge.
    """
    t = (trend or "").strip()
    if not t:
        return "<span style='color:#888;'>—</span>"

    color = "#666"
    prefix = ""
    if t.startswith("+"):
        color = "#d32f2f"  # regression (more memory)
        prefix = "↑ "
    elif t.startswith("-"):
        color = "#2e7d32"  # improvement
        prefix = "↓ "
    elif "≈" in t:
        color = "#666"

    return f"<span style='color:{color}; font-weight:700;'>{prefix}{t}</span>"


def _percent_trend(old: float, new: float, eps: float = 1e-9) -> str:
    if old <= eps:
        return ""
    pct = (new - old) / old
    if abs(pct) < 0.002:  # ~0.2% noise threshold
        return "≈0%"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct * 100.0:.1f}%"


def _percentile(xs: List[float], q: float) -> float:
    """
    Simple numpy-free percentile (xs already small: <= 100 steps).
    q in [0, 100].
    """
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


@dataclass(frozen=True)
class SeriesStats:
    last: float
    p50: float
    p95: float
    avg100: float
    trend: str


@dataclass(frozen=True)
class StepMemoryMetricView:
    """
    Normalized metric for UI (one metric, e.g. peak_allocated).

    All values are bytes.
    """

    metric: str
    steps: List[int]
    worst_y: List[float]
    median_y: List[float]
    worst_stats: SeriesStats
    median_stats: SeriesStats
    skew_pct: float
    worst_rank: Optional[int]


def _make_empty_figure(title: str) -> go.Figure:
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
        xaxis=dict(
            showgrid=False, title="Training Step", tickfont=dict(size=9)
        ),
        yaxis=dict(title="Memory", tickfont=dict(size=9)),
    )
    return fig


def _update_graph(
    plotly_ui,
    x: List[int],
    y_worst: List[float],
    y_median: List[float],
    title: str,
) -> None:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_worst,
            mode="lines",
            name="Worst",
            line=dict(width=2),
            hovertemplate="step=%{x}<br>worst=%{y:.0f} B<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_median,
            mode="lines",
            name="Median",
            line=dict(width=1, dash="dash"),
            hovertemplate="step=%{x}<br>median=%{y:.0f} B<extra></extra>",
        )
    )

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
        xaxis=dict(
            showgrid=False, title="Training Step", tickfont=dict(size=9)
        ),
        yaxis=dict(title="Memory", tickfont=dict(size=9)),
    )

    plotly_ui.update_figure(fig)


def _stats_table_html_dual(
    *,
    worst: SeriesStats,
    median: SeriesStats,
    skew_pct: float,
    worst_rank: Optional[int],
    show_median_trend: bool = False,
) -> str:
    def row(label: str, s: SeriesStats, show_skew: bool) -> str:
        trend_html = (
            _trend_badge(s.trend)
            if (label == "Worst" or show_median_trend)
            else "<span style='color:#888;'>—</span>"
        )
        imbalance = _fmt_pct(skew_pct) if show_skew else "—"
        rank = (
            f"r{worst_rank}" if (show_skew and worst_rank is not None) else "—"
        )

        return f"""
        <tr>
          <td style="text-align:left; padding:5px 6px;"><b>{label}</b></td>
          <td style="text-align:right; padding:5px 6px;">{fmt_mem_new(s.last)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_mem_new(s.p50)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_mem_new(s.p95)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_mem_new(s.avg100)}</td>
          <td style="text-align:center; padding:5px 6px;">{trend_html}</td>
          <td style="text-align:center; padding:5px 6px;">{imbalance}</td>
          <td style="text-align:center; padding:5px 6px;">{rank}</td>
        </tr>
        """

    return f"""
    <div style="margin-top:6px;">
      <table style="
          width:100%;
          border-collapse:collapse;
          font-size:12.5px;
          table-layout: fixed;          /* critical: prevents wide cells expanding table */
        ">
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
          {row("Worst", worst, show_skew=True)}
          {row("Median", median, show_skew=False)}
        </tbody>
      </table>
    </div>
    """


def _compute_series_stats(y: List[float]) -> SeriesStats:
    if not y:
        return SeriesStats(last=0.0, p50=0.0, p95=0.0, avg100=0.0, trend="")

    last = float(y[-1])
    p50 = float(_percentile(y, 50.0))
    p95 = float(_percentile(y, 95.0))
    avg100 = float(_avg_last_k(y, 100))

    # Trend: compare first half avg vs second half avg (cheap + stable)
    mid = max(1, len(y) // 2)
    a = float(sum(y[:mid]) / max(1, len(y[:mid])))
    b = float(sum(y[mid:]) / max(1, len(y[mid:])))
    trend = _percent_trend(a, b)

    return SeriesStats(last=last, p50=p50, p95=p95, avg100=avg100, trend=trend)


def _select_metric(payload: Any, preferred: str) -> Optional[Any]:
    """
    Pick one metric out of StepMemoryCombinedResult.metrics.
    """
    metrics = getattr(payload, "metrics", None)
    if metrics is None and isinstance(payload, dict):
        metrics = payload.get("metrics")

    if not metrics:
        return None

    # metrics may be dataclasses or dicts
    for m in metrics:
        key = (
            getattr(m, "metric", None)
            if not isinstance(m, dict)
            else m.get("metric")
        )
        if key == preferred:
            return m
    return None


def _normalize_metric(m: Any) -> Optional[StepMemoryMetricView]:
    """
    Convert a StepMemoryCombinedMetric (dataclass) OR dict into a UI object.
    """
    if m is None:
        return None

    metric = (
        getattr(m, "metric", None)
        if not isinstance(m, dict)
        else m.get("metric")
    )
    series = (
        getattr(m, "series", None)
        if not isinstance(m, dict)
        else m.get("series", {})
    )
    summary = (
        getattr(m, "summary", None)
        if not isinstance(m, dict)
        else m.get("summary", {})
    )

    steps = _to_list(
        getattr(series, "steps", None)
        if not isinstance(series, dict)
        else series.get("steps")
    )
    worst_y = _to_list(
        getattr(series, "worst", None)
        if not isinstance(series, dict)
        else series.get("worst")
    )
    median_y = _to_list(
        getattr(series, "median", None)
        if not isinstance(series, dict)
        else series.get("median")
    )

    if not steps or (not worst_y and not median_y):
        return None

    n = min(
        len(steps),
        len(worst_y) if worst_y else len(steps),
        len(median_y) if median_y else len(steps),
    )
    steps = [int(s) for s in steps[:n]]
    worst_y = [_safe_float(v) for v in (worst_y[:n] if worst_y else [0.0] * n)]
    median_y = [
        _safe_float(v) for v in (median_y[:n] if median_y else [0.0] * n)
    ]

    skew_pct = (
        _safe_float(getattr(summary, "skew_pct", None))
        if not isinstance(summary, dict)
        else _safe_float(summary.get("skew_pct", 0.0))
    )
    worst_rank = (
        getattr(summary, "worst_rank", None)
        if not isinstance(summary, dict)
        else summary.get("worst_rank", None)
    )

    return StepMemoryMetricView(
        metric=str(metric or ""),
        steps=steps,
        worst_y=worst_y,
        median_y=median_y,
        worst_stats=_compute_series_stats(worst_y),
        median_stats=_compute_series_stats(median_y),
        skew_pct=float(skew_pct),
        worst_rank=worst_rank if worst_rank is None else int(worst_rank),
    )


# -----------------------------
# Public build/update
# -----------------------------
def build_step_memory_section(
    *, title: str = "Step Memory (Peak Allocated)"
) -> Dict[str, Any]:
    """
    Build the Step Memory card.

    Returns handles required for incremental updates:
      - graph: ui.plotly
      - stats_html: ui.html
      - empty: ui.html (hint)
    """
    card = ui.card().classes("m-2 p-4 flex-1")
    card.style(
        f"""
        height: 360px;
        display: flex;
        flex-direction: column;

        /* critical: allow shrinking inside ui.row()/flex */
        min-width: 0;

        /* prevent any child (plotly/table) from expanding width */
        max-width: 100%;
        overflow-x: hidden;

        {CARD_STYLE}
        """
    )

    with card:
        ui.label(title).classes(METRIC_TITLE).style(TITLE_STYLE)

        plotly_ui = ui.plotly(_make_empty_figure(title)).classes("w-full")
        plotly_ui.style("max-width: 100%; overflow-x: hidden;")

        stats_html = (
            ui.html("", sanitize=False)
            .classes(METRIC_TEXT)
            .style("color:#333")
        )

        empty_hint = ui.html(
            "<div style='text-align:center; padding:12px; color:#888; font-style:italic;'>"
            "Waiting for memory samples…</div>",
            sanitize=False,
        )

    return {
        "card": card,
        "graph": plotly_ui,
        "stats_html": stats_html,
        "empty": empty_hint,
    }


def update_step_memory_section(
    panel: Dict[str, Any],
    payload: Any,
    *,
    prefer_metric: str = "peak_allocated",
) -> None:
    """
    Update Step Memory section from StepMemoryCombinedResult.

    Parameters
    ----------
    panel:
        Dict returned by build_step_memory_section().
    payload:
        StepMemoryCombinedResult or dict with same shape.
    prefer_metric:
        Which metric to display in this card: "peak_allocated" or "peak_reserved".
    """
    m = _select_metric(payload, preferred=prefer_metric)
    view = _normalize_metric(m)
    if view is None:
        panel["empty"].content = (
            "<div style='text-align:center; padding:12px; color:#888; font-style:italic;'>"
            "No step memory data detected.</div>"
        )
        return

    # Hide empty hint once real data arrives
    panel["empty"].content = ""

    title = (
        "Step Memory (Peak Allocated)"
        if view.metric == "peak_allocated"
        else "Step Memory (Peak Reserved)"
    )
    _update_graph(
        panel["graph"], view.steps, view.worst_y, view.median_y, title=title
    )

    panel["stats_html"].content = _stats_table_html_dual(
        worst=view.worst_stats,
        median=view.median_stats,
        skew_pct=view.skew_pct,
        worst_rank=view.worst_rank,
        show_median_trend=False,
    )
