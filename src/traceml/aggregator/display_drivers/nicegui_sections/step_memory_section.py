"""
NiceGUI Step Memory Summary (Worst vs Median)

UI shows **GB**, but compute payload is **BYTES** (sampler reports bytes).
So we convert BYTES -> GB inside the UI normalization.

Stability:
- Caches last-good view in panel dict so transient bad ticks don't blank the card.
- Wraps update logic in try/except to avoid "silent break until refresh".

Expected input (`payload`)
--------------------------
payload is StepMemoryCombinedResult (dataclasses) OR a dict with the same shape.
Per metric:
  - metric.metric in {"peak_allocated", "peak_reserved"}
  - series.steps: List[int]
  - series.worst: List[float]   # BYTES
  - series.median: List[float]  # BYTES
  - summary.skew_pct, summary.worst_rank
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nicegui import ui
import plotly.graph_objects as go

from traceml.loggers.error_log import get_error_logger

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



BYTES_PER_GB = 1e9  # decimal GB (matches "GB" label)


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


def fmt_GB(v_gb: float) -> str:
    """
    Format GB as a compact string. Input is GB.
    """
    v = _safe_float(v_gb, 0.0)
    if v <= 0.0:
        return "0.0 GB"
    if v < 1.0:
        # Keep your original behavior (MiB label). If you prefer MB, change label/math.
        return f"{v * 1024.0:.0f} MiB"
    if v < 10.0:
        return f"{v:.2f} GB"
    return f"{v:.1f} GB"


def _trend_badge(trend: str) -> str:
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


def _bytes_to_gb_list(y_bytes: List[float]) -> List[float]:
    # Convert bytes -> GB for display/plot/stats
    if not y_bytes:
        return []
    inv = 1.0 / BYTES_PER_GB
    return [float(v) * inv for v in y_bytes]


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

    All values are GB (display units).
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
        xaxis=dict(showgrid=False, title="Training Step", tickfont=dict(size=9)),
        yaxis=dict(title="Memory (GB)", tickfont=dict(size=9)),
    )
    return fig


def _update_graph(plotly_ui, fig: go.Figure, x: List[int], y_worst: List[float], y_median: List[float]) -> None:
    """
    Update existing figure in-place (fast).
    Assumes fig already has 2 traces: Worst (0) and Median (1).
    """
    # update traces (no new figure)
    fig.data[0].x = x
    fig.data[0].y = y_worst
    fig.data[1].x = x
    fig.data[1].y = y_median

    # push update (same fig object)
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
        rank = f"r{worst_rank}" if (show_skew and worst_rank is not None) else "—"

        return f"""
        <tr>
          <td style="text-align:left; padding:5px 6px;"><b>{label}</b></td>
          <td style="text-align:right; padding:5px 6px;">{fmt_GB(s.last)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_GB(s.p50)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_GB(s.p95)}</td>
          <td style="text-align:right; padding:5px 6px;">{fmt_GB(s.avg100)}</td>
          <td style="text-align:center; padding:5px 6px;">{trend_html}</td>
          <td style="text-align:center; padding:5px 6px;">{imbalance}</td>
          <td style="text-align:center; padding:5px 6px;">{rank}</td>
        </tr>
        """

    return f"""
    <div style="margin-top:6px;">
      <table style="width:100%; border-collapse:collapse; font-size:12.5px; table-layout: fixed;">
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


def _compute_series_stats(y_gb: List[float]) -> SeriesStats:
    if not y_gb:
        return SeriesStats(last=0.0, p50=0.0, p95=0.0, avg100=0.0, trend="")

    last = float(y_gb[-1])
    p50 = float(_percentile(y_gb, 50.0))
    p95 = float(_percentile(y_gb, 95.0))
    avg100 = float(_avg_last_k(y_gb, 100))

    mid = max(1, len(y_gb) // 2)
    a = float(sum(y_gb[:mid]) / max(1, mid))
    b = float(sum(y_gb[mid:]) / max(1, len(y_gb) - mid))
    trend = _percent_trend(a, b)

    return SeriesStats(last=last, p50=p50, p95=p95, avg100=avg100, trend=trend)


def _select_metric(payload: Any, preferred: str) -> Optional[Any]:
    metrics = getattr(payload, "metrics", None)
    if metrics is None and isinstance(payload, dict):
        metrics = payload.get("metrics")
    if not metrics:
        return None

    for m in metrics:
        key = getattr(m, "metric", None) if not isinstance(m, dict) else m.get("metric")
        if key == preferred:
            return m
    return None


def _normalize_metric(m: Any) -> Optional[StepMemoryMetricView]:
    if m is None:
        return None

    metric = getattr(m, "metric", None) if not isinstance(m, dict) else m.get("metric")
    series = getattr(m, "series", None) if not isinstance(m, dict) else m.get("series", {})
    summary = getattr(m, "summary", None) if not isinstance(m, dict) else m.get("summary", {})

    steps = _to_list(getattr(series, "steps", None) if not isinstance(series, dict) else series.get("steps"))
    worst_b = _to_list(getattr(series, "worst", None) if not isinstance(series, dict) else series.get("worst"))
    median_b = _to_list(getattr(series, "median", None) if not isinstance(series, dict) else series.get("median"))

    if not steps or (not worst_b and not median_b):
        return None

    # Compute safe aligned length
    n_steps = len(steps)
    n_w = len(worst_b) if worst_b else n_steps
    n_m = len(median_b) if median_b else n_steps
    n = min(n_steps, n_w, n_m)
    if n <= 0:
        return None

    # Safe step conversion (never throws)
    steps = [int(_safe_float(s, 0.0)) for s in steps[:n]]

    # bytes -> float bytes (never throws)
    worst_b = [_safe_float(v) for v in (worst_b[:n] if worst_b else [0.0] * n)]
    median_b = [_safe_float(v) for v in (median_b[:n] if median_b else [0.0] * n)]

    # CONVERT HERE: bytes -> GB for display
    worst_y = _bytes_to_gb_list(worst_b)
    median_y = _bytes_to_gb_list(median_b)

    skew_pct = (
        _safe_float(getattr(summary, "skew_pct", None))
        if not isinstance(summary, dict)
        else _safe_float(summary.get("skew_pct", 0.0))
    )
    worst_rank = getattr(summary, "worst_rank", None) if not isinstance(summary, dict) else summary.get("worst_rank")

    return StepMemoryMetricView(
        metric=str(metric or ""),
        steps=steps,
        worst_y=worst_y,
        median_y=median_y,
        worst_stats=_compute_series_stats(worst_y),
        median_stats=_compute_series_stats(median_y),
        skew_pct=float(skew_pct),
        worst_rank=worst_rank if worst_rank is None else int(_safe_float(worst_rank, 0.0)),
    )


# -----------------------------
# Public build/update
# -----------------------------
def build_step_memory_section(*, title: str = "Step Memory (Peak Allocated)") -> Dict[str, Any]:
    card = ui.card().classes("m-2 p-4 flex-1")
    card.style(
        f"""
        height: 360px;
        display: flex;
        flex-direction: column;
        min-width: 0;
        max-width: 100%;
        overflow-x: hidden;
        {CARD_STYLE}
        """
    )

    # Build ONCE
    fig = _make_empty_figure(title)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Worst",
            line=dict(width=2),
            hovertemplate="step=%{x}<br>worst=%{y:.2f} GB<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Median",
            line=dict(width=1, dash="dash"),
            hovertemplate="step=%{x}<br>median=%{y:.2f} GB<extra></extra>",
        )
    )

    with card:
        ui.label(title).classes(METRIC_TITLE).style(TITLE_STYLE)

        plotly_ui = ui.plotly(fig).classes("w-full")
        plotly_ui.style("max-width: 100%; overflow-x: hidden;")

        stats_html = ui.html("", sanitize=False).classes(METRIC_TEXT).style("color:#333")

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
        "_last_ok_view": None,
        "_fig": fig,
    }

def update_step_memory_section(
    panel: Dict[str, Any],
    payload: Any,
    *,
    prefer_metric: str = "peak_allocated",
) -> None:
    """
    Robust update:
    - never blanks the card on transient failures
    - always falls back to last-good view if anything goes wrong
    """

    def _show_waiting(msg: str) -> None:
        try:
            panel["empty"].content = (
                "<div style='text-align:center; padding:12px; color:#888; font-style:italic;'>"
                f"{msg}</div>"
            )
        except Exception:
            pass

    def _render_view(view: StepMemoryMetricView) -> None:
        # only now hide the empty hint
        try:
            panel["empty"].content = ""
        except Exception:
            pass

        title = "Step Memory (Peak Allocated)" if view.metric == "peak_allocated" else "Step Memory (Peak Reserved)"

        try:
            _update_graph(panel["graph"], panel["_fig"], view.steps, view.worst_y, view.median_y)
        except Exception:
            pass

        try:
            panel["stats_html"].content = _stats_table_html_dual(
                worst=view.worst_stats,
                median=view.median_stats,
                skew_pct=view.skew_pct,
                worst_rank=view.worst_rank,
                show_median_trend=False,
            )
        except Exception:
            pass

    try:
        # Try to compute a fresh view
        m = _select_metric(payload, preferred=prefer_metric)
        view = _normalize_metric(m)

        if view is not None:
            panel["_last_ok_view"] = view
            _render_view(view)
            return

        # Fresh view unavailable -> fall back
        last = panel.get("_last_ok_view")
        if last is not None:
            _render_view(last)
        else:
            _show_waiting("Waiting for memory samples…")
        return

    except Exception:
        # On ANY exception: render last-good if available; otherwise show waiting message
        last = panel.get("_last_ok_view")
        if last is not None:
            try:
                _render_view(last)
            except Exception:
                pass
        else:
            _show_waiting("Waiting for memory samples…")
        return