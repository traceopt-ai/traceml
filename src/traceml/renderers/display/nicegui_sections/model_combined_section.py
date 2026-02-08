"""
Model StepBreakdown (Median vs Worst)

UI section that renders two simple bar charts showing window-aggregated
step time components derived from StepCombinedTimeResult.

Charts
------
1. Median (typical rank) breakdown
2. Worst-rank (tail) breakdown

Bars represent independent metrics:
- Dataloader
- GPU Compute (forward + backward + optimizer)
- WAIT* (derived proxy)
- Step Time

No stacking is used to avoid mixed-clock ambiguity.
"""

from typing import Any, Dict, Optional, List

from nicegui import ui
import plotly.graph_objects as go

from traceml.renderers.step_combined.schema import (
    StepCombinedTimeResult, StepCombinedTimeMetric
)


COLORS = {
    "Dataloader": "#d32f2f",
    "Forward": "#1976d2",
    "Backward": "#512da8",
    "Optimizer": "#2e7d32",
    "WAIT*": "#f9a825",
    "Step Time": "#455a64",
}


def _card_style() -> str:
    return """
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    """

def build_model_combined_section() -> Dict[str, Any]:
    """
    Build the Model Step Time Breakdown section.

    Layout
    ------
    Single row with three cards:
      1) Median breakdown bar chart
      2) Worst-rank breakdown bar chart
      3) Stats / interpretation panel
    """
    container = ui.row().classes("w-full gap-4").style(
        "flex-wrap: nowrap;"
    )

    with container:
        # -------- Card 1: Median --------
        median_card = ui.card().classes("p-3 flex-1")
        median_card.style(_card_style())

        with median_card:
            ui.label("Median Step Breakdown").classes(
                "text-sm font-bold mb-1"
            ).style("color:#2e7d32;")

            median_plot = ui.plotly(
                _empty_bar_figure("Median (Typical Rank)")
            ).classes("w-full")

        # -------- Card 2: Worst --------
        worst_card = ui.card().classes("p-3 flex-1")
        worst_card.style(_card_style())

        with worst_card:
            ui.label("Worst Rank Breakdown").classes(
                "text-sm font-bold mb-1"
            ).style("color:#c62828;")

            worst_plot = ui.plotly(
                _empty_bar_figure("Worst Rank")
            ).classes("w-full")

        # -------- Card 3: Stats --------
        stats_card = ui.card().classes("p-3 flex-1")
        stats_card.style(_card_style())

        with stats_card:
            ui.label("Summary & Interpretation").classes(
                "text-sm font-bold mb-2"
            ).style("color:#455a64;")

            stats = ui.markdown("").classes(
                "text-xs text-gray-700 leading-relaxed"
            )

    return {
        "container": container,
        "median_plot": median_plot,
        "worst_plot": worst_plot,
        "stats": stats,
    }


def update_model_combined_section(
    panel: Dict[str, Any],
    payload: Optional[StepCombinedTimeResult],
) -> None:
    """
    Update the Model Step Time Breakdown section.

    Parameters
    ----------
    panel : Dict[str, Any]
        UI handles returned by build_model_combined_section.
    payload : Optional[StepCombinedTimeResult]
        Renderer-facing aggregated step metrics.
    """
    if not payload or not payload.metrics:
        return

    metrics = _index_metrics(payload.metrics)

    required = {
        "dataloader_fetch",
        "forward",
        "backward",
        "optimizer_step",
        "wait_proxy",
        "step_time_ms",
    }
    if not required.issubset(metrics):
        return

    order = [
        ("Dataloader", metrics["dataloader_fetch"]),
        ("Forward", metrics["forward"]),
        ("Backward", metrics["backward"]),
        ("Optimizer", metrics["optimizer_step"]),
        ("WAIT*", metrics["wait_proxy"]),
        ("Step Time", metrics["step_time_ms"]),
    ]

    median_fig = _build_bar_chart(
        title="Median (Typical Rank)",
        values=[m.summary.median_total for _, m in order],
        labels=[k for k, _ in order],
    )

    worst_fig = _build_bar_chart(
        title=f"Worst Rank r{metrics['step_time_ms'].summary.worst_rank}",
        values=[m.summary.worst_total for _, m in order],
        labels=[k for k, _ in order],
    )

    panel["median_plot"].update_figure(median_fig)
    panel["worst_plot"].update_figure(worst_fig)

    panel["stats"].set_content(
        _render_stats_block(metrics, payload.metrics[0].summary.steps_used)
    )



def _index_metrics(
    metrics: List[StepCombinedTimeMetric],
) -> Dict[str, StepCombinedTimeMetric]:
    """
    Index metrics by metric key.
    """
    return {m.metric: m for m in metrics}



def _build_bar_chart(
    *,
    title: str,
    values: List[float],
    labels: List[str],
) -> go.Figure:
    """
    Build a simple vertical bar chart.
    """
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=[COLORS.get(l, "#999999") for l in labels]
            ),
            hovertemplate="%{x}<br>%{y:.2f} ms<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        yaxis=dict(title="Time (ms)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def _render_stats_block(
    metrics: Dict[str, StepCombinedTimeMetric],
    steps: int,
) -> str:
    """
    Render side statistics block.
    """
    wait = metrics["wait_proxy"]
    step = metrics["step_time_ms"]
    fwd = metrics["forward"]
    bwd = metrics["backward"]
    opt = metrics["optimizer_step"]

    exec_skew = max(
        fwd.summary.skew_pct,
        bwd.summary.skew_pct,
        opt.summary.skew_pct,
    )

    wait_share = (
        wait.summary.median_total / step.summary.median_total
        if step.summary.median_total > 0
        else 0.0
    )

    return f"""
        **WAIT Share:** {wait_share * 100:.1f}%  
        **Worst Rank:** r{step.summary.worst_rank}  
        **Step Skew:** +{step.summary.skew_pct * 100:.1f}%  
        **Execution Skew:** +{exec_skew * 100:.1f}%  
        **Window Size:** {steps} steps  
        
        *WAIT = step time − (forward + backward + optimizer)*  
        *Includes H2D copies, synchronization, and CPU-side overhead*
        """



def _empty_bar_figure(title: str) -> go.Figure:
    """Create an empty placeholder bar figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    return fig
