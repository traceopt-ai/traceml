"""
Unified Step Time analysis section for the overview dashboard.

This card is intentionally compact and space-efficient:
- one grouped stacked bar chart comparing Median vs Worst
- one dense KPI strip

It does not render diagnosis prose; interpretation belongs in the diagnostics rail.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from nicegui import ui

from traceml.renderers.step_time.schema import (
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
)

from .ui_shell import CARD_STYLE, compact_metric_html, safe_ms, safe_pct

_STACK_KEYS = [
    ("Dataloader", "dataloader_fetch", "#d32f2f"),
    ("Forward", "forward", "#1976d2"),
    ("Backward", "backward", "#512da8"),
    ("Optimizer", "optimizer_step", "#2e7d32"),
    ("WAIT*", "wait_proxy", "#f9a825"),
]

_REQUIRED_KEYS = {
    "dataloader_fetch",
    "forward",
    "backward",
    "optimizer_step",
    "wait_proxy",
    "step_time",
}


def build_model_combined_section() -> Dict[str, Any]:
    """Build the Step Time analysis card."""
    card = ui.card().classes("w-full h-full p-3")
    card.style(
        CARD_STYLE + "height: 100%; overflow-y: auto; overflow-x: hidden;"
    )

    fig = _init_figure()

    with card:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label("Step Time Analysis").classes("text-sm font-bold").style(
                "color:#455a64;"
            )
            window_text = ui.html("window: -", sanitize=False).classes(
                "text-[11px] text-gray-500"
            )

        plot = ui.plotly(fig).classes("w-full")
        kpis = ui.html("", sanitize=False).classes("mt-2")

    return {
        "window_text": window_text,
        "plot": plot,
        "kpis": kpis,
        "_fig": fig,
        "_last_kpis": None,
        "_last_window_text": None,
        "_last_signature": None,
    }


def update_model_combined_section(
    panel: Dict[str, Any],
    payload: Optional[StepCombinedTimeResult],
) -> None:
    """Update the Step Time analysis card in place with cached signatures."""
    if not payload or not payload.metrics:
        return

    metrics = _index_metrics(payload.metrics)
    if not _REQUIRED_KEYS.issubset(metrics):
        return

    step = metrics["step_time"]
    wait = metrics["wait_proxy"]

    signature = (
        tuple(
            round(float(metrics[key].summary.median_total or 0.0), 4)
            for _, key, _ in _STACK_KEYS
        )
        + tuple(
            round(float(metrics[key].summary.worst_total or 0.0), 4)
            for _, key, _ in _STACK_KEYS
        )
        + (
            round(float(step.summary.median_total or 0.0), 4),
            round(float(step.summary.worst_total or 0.0), 4),
            int(step.summary.steps_used or 0),
            int(step.summary.worst_rank or -1),
        )
    )

    if panel.get("_last_signature") != signature:
        _update_plot(panel, metrics)
        panel["_last_signature"] = signature

    window_text = f"window: {int(step.summary.steps_used or 0)} aligned steps"
    if panel.get("_last_window_text") != window_text:
        panel["window_text"].content = window_text
        panel["_last_window_text"] = window_text

    kpis_html = _render_kpis(metrics, step, wait)
    if panel.get("_last_kpis") != kpis_html:
        panel["kpis"].content = kpis_html
        panel["_last_kpis"] = kpis_html


def _init_figure() -> go.Figure:
    fig = go.Figure()
    for label, _metric_key, color in _STACK_KEYS:
        fig.add_trace(
            go.Bar(
                x=["Median", "Worst"],
                y=[0.0, 0.0],
                name=label,
                marker=dict(color=color),
                hovertemplate="%{x}<br>"
                + label
                + ": %{y:.2f} ms<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        height=235,
        margin=dict(l=8, r=8, t=8, b=28),
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
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Time (ms)"),
    )
    return fig


def _update_plot(
    panel: Dict[str, Any], metrics: Dict[str, StepCombinedTimeMetric]
) -> None:
    fig = panel["_fig"]
    for idx, (_label, key, _color) in enumerate(_STACK_KEYS):
        metric = metrics[key]
        fig.data[idx].y = [
            float(metric.summary.median_total or 0.0),
            float(metric.summary.worst_total or 0.0),
        ]
    panel["plot"].update_figure(fig)


def _index_metrics(
    metrics: List[StepCombinedTimeMetric],
) -> Dict[str, StepCombinedTimeMetric]:
    return {metric.metric: metric for metric in metrics}


def _render_kpis(
    metrics: Dict[str, StepCombinedTimeMetric],
    step: StepCombinedTimeMetric,
    wait: StepCombinedTimeMetric,
) -> str:
    median_total = float(step.summary.median_total or 0.0)
    worst_total = float(step.summary.worst_total or 0.0)
    wait_share = (
        float(wait.summary.median_total or 0.0) / median_total
        if median_total > 0.0
        else 0.0
    )

    items = [
        compact_metric_html("Median Total", safe_ms(median_total)),
        compact_metric_html("Worst Total", safe_ms(worst_total)),
        compact_metric_html("Gap", safe_pct(step.summary.skew_pct)),
        compact_metric_html(
            "Worst Rank",
            (
                f"r{int(step.summary.worst_rank)}"
                if step.summary.worst_rank is not None
                else "-"
            ),
        ),
        compact_metric_html("WAIT Share", safe_pct(wait_share)),
        compact_metric_html(
            "Dominant Split",
            _dominant_metric(metrics, mode="worst_total"),
        ),
    ]

    return (
        "<div style='display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); "
        "gap:8px; padding-top:6px; border-top:1px solid #ececec;'>"
        + "".join(items)
        + "</div>"
    )


def _dominant_metric(
    metrics: Dict[str, StepCombinedTimeMetric], mode: str
) -> str:
    best_label = "-"
    best_value = -1.0

    for label, key, _ in _STACK_KEYS:
        value = float(getattr(metrics[key].summary, mode, 0.0) or 0.0)
        if value > best_value:
            best_value = value
            best_label = label

    return f"{best_label} ({best_value:.1f} ms)" if best_value >= 0.0 else "-"
