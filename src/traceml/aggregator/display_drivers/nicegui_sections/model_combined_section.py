"""
Model Step Breakdown (Median vs Worst)

Fast NiceGUI section for TraceML "Model Combined" metrics.

UI goals
--------
- Match System card visual tone (white, subtle border/shadow, rounded)
- Avoid repeated titles (card titles only; Plotly title hidden)
- Update Plotly traces in-place (no re-creating figures each tick)
- Skip updates if values didn't change

Data contract
-------------
Expects canonical metric keys:
  dataloader_fetch, forward, backward, optimizer_step, wait_proxy, step_time
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from nicegui import ui

from traceml.renderers.step_combined.schema import (
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
)

# UI labels (fixed order)
_LABELS: List[str] = ["Dataloader", "Forward", "Backward", "Optimizer", "WAIT*", "Step Time"]

# Keep your existing palette (fine)
_COLORS: Dict[str, str] = {
    "Dataloader": "#d32f2f",
    "Forward": "#1976d2",
    "Backward": "#512da8",
    "Optimizer": "#2e7d32",
    "WAIT*": "#f9a825",
    "Step Time": "#455a64",
}
_LABEL_COLORS: List[str] = [_COLORS.get(l, "#999999") for l in _LABELS]

_REQUIRED_KEYS = {
    "dataloader_fetch",
    "forward",
    "backward",
    "optimizer_step",
    "wait_proxy",
    "step_time",
}

_ORDER: List[Tuple[str, str]] = [
    ("Dataloader", "dataloader_fetch"),
    ("Forward", "forward"),
    ("Backward", "backward"),
    ("Optimizer", "optimizer_step"),
    ("WAIT*", "wait_proxy"),
    ("Step Time", "step_time"),
]


def _card_style() -> str:
    """
    Visual tone aligned with System card:
    - white background
    - subtle border + shadow
    - rounded corners
    """
    return """
    background: #ffffff;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    """


def _init_bar_figure() -> go.Figure:
    """
    Create the Plotly figure once.

    Titles are hidden to avoid repeating card headings.
    Later updates patch only:
      - fig.data[0].y
    """
    fig = go.Figure(
        go.Bar(
            x=_LABELS,
            y=[0.0] * len(_LABELS),
            marker=dict(color=_LABEL_COLORS),
            hovertemplate="%{x}<br>%{y:.2f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        # Hide plot title (card title is the title)
        title=None,
        height=220,
        margin=dict(l=10, r=10, t=8, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        yaxis=dict(title="Time (ms)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def build_model_combined_section() -> Dict[str, Any]:
    """
    Build the Model Combined section: Median chart, Worst chart, and Summary.

    Returns a dict of UI handles used by update_model_combined_section().
    """
    container = ui.row().classes("w-full gap-3").style("flex-wrap: nowrap;")

    with container:
        # -------- Card 1: Median --------
        median_card = ui.card().classes("p-3 flex-1").style(_card_style())
        with median_card:
            ui.label("Median Step Breakdown").classes("text-sm font-bold mb-1").style(
                "color:#2e7d32;"
            )
            median_plot = ui.plotly(_init_bar_figure()).classes("w-full")

        # -------- Card 2: Worst --------
        worst_card = ui.card().classes("p-3 flex-1").style(_card_style())
        with worst_card:
            # This title will be updated with the actual worst rank
            worst_title = ui.label("Worst Rank Breakdown").classes("text-sm font-bold mb-1").style(
                "color:#c62828;"
            )
            worst_plot = ui.plotly(_init_bar_figure()).classes("w-full")

        # -------- Card 3: Summary --------
        stats_card = ui.card().classes("p-3 flex-1").style(_card_style())
        with stats_card:
            ui.label("Summary & Interpretation").classes("text-sm font-bold mb-2").style(
                "color:#455a64;"
            )
            stats = ui.markdown("").classes("text-xs text-gray-700 leading-relaxed")

    # Cache to avoid re-sending identical updates
    cache = {
        "median_vals": None,      # Optional[Tuple[float, ...]]
        "worst_vals": None,       # Optional[Tuple[float, ...]]
        "worst_title": None,      # Optional[str]
        "stats_md": None,         # Optional[str]
    }

    return {
        "container": container,
        "median_plot": median_plot,
        "worst_plot": worst_plot,
        "worst_title": worst_title,
        "stats": stats,
        "_cache": cache,
    }


def update_model_combined_section(
    panel: Dict[str, Any],
    payload: Optional[StepCombinedTimeResult],
) -> None:
    """
    Update section (fast path).

    - Build fixed-size tuples of values
    - Patch y-values in-place
    - Update card title for worst rank
    - Skip if unchanged
    """
    if not payload or not payload.metrics:
        return

    metrics = _index_metrics(payload.metrics)
    if not _REQUIRED_KEYS.issubset(metrics):
        return

    cache: Dict[str, Any] = panel.get("_cache", {})

    # Build arrays in fixed order (cheap)
    median_vals = tuple(round(metrics[k].summary.median_total, 3) for _, k in _ORDER)
    worst_vals = tuple(round(metrics[k].summary.worst_total, 3) for _, k in _ORDER)

    step = metrics["step_time"]
    worst_rank = step.summary.worst_rank
    worst_title = f"Worst Rank Breakdown (r{worst_rank})" if worst_rank is not None else "Worst Rank Breakdown"

    # --- Median plot update ---
    if cache.get("median_vals") != median_vals:
        fig = panel["median_plot"].figure
        fig.data[0].y = list(median_vals)
        panel["median_plot"].update()
        cache["median_vals"] = median_vals

    # --- Worst plot update ---
    if cache.get("worst_vals") != worst_vals:
        fig = panel["worst_plot"].figure
        fig.data[0].y = list(worst_vals)
        panel["worst_plot"].update()
        cache["worst_vals"] = worst_vals

    # --- Worst title update (avoid repeating plot title) ---
    if cache.get("worst_title") != worst_title:
        panel["worst_title"].text = worst_title
        cache["worst_title"] = worst_title

    # --- Stats markdown update ---
    stats_md = _render_stats_block(metrics, step.summary.steps_used)
    if cache.get("stats_md") != stats_md:
        panel["stats"].set_content(stats_md)
        cache["stats_md"] = stats_md

    panel["_cache"] = cache


def _index_metrics(metrics: List[StepCombinedTimeMetric]) -> Dict[str, StepCombinedTimeMetric]:
    """Index metrics by metric key."""
    return {m.metric: m for m in metrics}


def _render_stats_block(
    metrics: Dict[str, StepCombinedTimeMetric],
    steps: int,
) -> str:
    """
    Render a consistent, action-oriented summary.

    - Step skew is the primary straggler signal.
    - WAIT* share and dataloader share are primary pipeline health signals.
    - If component skew is high but step skew is low: "hidden imbalance".
    """
    wait = metrics["wait_proxy"]
    step = metrics["step_time"]
    dl = metrics["dataloader_fetch"]
    fwd = metrics["forward"]
    bwd = metrics["backward"]
    opt = metrics["optimizer_step"]

    step_med = step.summary.median_total
    if step_med <= 0:
        return f"**Window:** {steps} steps  \nNo data yet."

    def share(x: float) -> float:
        return x / step_med

    exec_med = fwd.summary.median_total + bwd.summary.median_total + opt.summary.median_total
    exec_share = share(exec_med)
    dl_share = share(dl.summary.median_total)
    wait_share = share(wait.summary.median_total)

    step_skew = step.summary.skew_pct
    exec_skew = max(fwd.summary.skew_pct, bwd.summary.skew_pct, opt.summary.skew_pct)
    dl_skew = dl.summary.skew_pct
    wait_skew = wait.summary.skew_pct

    if step_skew >= 0.10:
        status = "⚠️ Straggler"
        north_star = "Step time is imbalanced across ranks."
        next_step = "Inspect worst rank and compare rank heatmap for step_time and dataloader."
    elif wait_share >= 0.20:
        status = "⚠️ Stalls / Sync (proxy)"
        north_star = f"WAIT* is high ({wait_share*100:.0f}% of step)."
        next_step = "Check synchronization/communication points, CPU stalls, and H2D copies."
    elif dl_share >= 0.25:
        status = "⚠️ Input-bound"
        north_star = f"Dataloader is large ({dl_share*100:.0f}% of step)."
        next_step = "Increase input throughput: workers, prefetch, pinned memory, storage."
    else:
        status = "✅ Balanced"
        north_star = "No strong bottleneck signals in this window."
        next_step = "Monitor trends; if throughput is the goal, optimize model/compile/mixed precision."

    hidden_notes = []
    if step_skew < 0.05 and exec_skew >= 0.10:
        hidden_notes.append(
            f"Compute components are imbalanced (max exec skew +{exec_skew*100:.1f}%), "
            "but step time is balanced—likely overlapped/compensated."
        )
    if step_skew < 0.05 and dl_skew >= 0.10:
        hidden_notes.append(
            f"Dataloader is imbalanced (+{dl_skew*100:.1f}%), but step time is balanced—"
            "could be hidden by buffering/prefetch."
        )

    worst_rank = step.summary.worst_rank
    worst_rank_str = f"r{worst_rank}" if worst_rank is not None else "—"

    hidden_md = ""
    if hidden_notes:
        hidden_md = "\n".join(f"- {n}" for n in hidden_notes[:2])
        hidden_md = f"\n\n**Secondary signals**\n{hidden_md}"

    return (
        f"**Status:** {status}  \n"
        f"**Window:** {steps} steps · **Worst Rank:** {worst_rank_str} · **Step Skew:** +{step_skew*100:.1f}%  \n\n"
        f"**North star:** {north_star}  \n"
        f"**Next step:** {next_step}  \n\n"
        f"**Breakdown (median):** Exec {exec_share*100:.0f}% · Dataloader {dl_share*100:.0f}% · WAIT* {wait_share*100:.0f}%  \n"
        f"**Component skew:** Forward +{fwd.summary.skew_pct*100:.1f}% · Backward +{bwd.summary.skew_pct*100:.1f}% · "
        f"Opt +{opt.summary.skew_pct*100:.1f}% · Dataloader +{dl_skew*100:.1f}% · WAIT* +{wait_skew*100:.1f}%"
        f"{hidden_md}\n\n"
        f"*WAIT* = step − (forward + backward + optimizer). Mixed CPU/GPU clocks → proxy.*"
    )