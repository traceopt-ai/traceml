"""
Model Step Time Breakdown (Dataloader vs Compute)

UI section that renders a single combined time-series chart showing:

  - Dataloader fetch time (red)
  - Training step compute time (green)

Key properties
--------------
- UI-only derived visualization
- Reads pre-aggregated telemetry from renderer
- Aligns series by *step intersection*, not by index
- Respects renderer truncation ("last X points")
- Never extrapolates or pads missing steps
- Safe under partial / delayed sampler emission

Expected telemetry shape
------------------------
telemetry: Dict[str, Any] containing:
  - "dataloading_time"
  - "step_time"

Each metric follows the StepCombinedComputer contract.
"""

from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui
import plotly.graph_objects as go




def build_model_combined_section() -> Dict[str, Any]:
    """
    Build the Model Step Time Breakdown section.

    Returns handles required for incremental updates.
    """
    card = ui.card().classes("p-3 w-full")
    card.style(
        """
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 10px rgba(0,0,0,0.10);
        """
    )

    with card:
        ui.label("Step Time Breakdown").classes(
            "text-lg font-bold mb-1"
        ).style("color:#d47a00;")

        plot = ui.plotly(_empty_figure()).classes("w-full")

        hint = ui.label("").classes("text-xs text-gray-500 mt-1")

    return {
        "card": card,
        "plot": plot,
        "hint": hint,
    }



def update_model_combined_section(
    panel: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]],
) -> None:
    """
    Update the combined step time graph.

    Visualization policy
    --------------------
    - Uses sum series for both metrics
    - Plots only the intersection of step ranges
    - Never pads or extrapolates
    - Safe under partial data arrival
    """
    if not telemetry or not isinstance(telemetry, dict):
        return

    dl_steps, dl_vals = _extract_series(
        telemetry.get("dataloading_time")
    )
    st_steps, st_vals = _extract_series(
        telemetry.get("step_time")
    )

    intersection = _intersect_step_ranges(dl_steps, st_steps)
    if not intersection:
        return

    start, end = intersection

    dl_x, dl_y = _clip_series(dl_steps, dl_vals, start, end)
    st_x, st_y = _clip_series(st_steps, st_vals, start, end)

    if not dl_x or not st_x:
        return

    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=st_x,
            y=st_y,
            name="Step Compute",
            marker_color="#2e7d32",
        )
    )

    fig.add_trace(
        go.Bar(
            x=dl_x,
            y=dl_y,
            name="Dataloader Fetch",
            marker_color="#d32f2f",
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=6, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10),
        ),
        xaxis=dict(
            title="Training Step",
            showgrid=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Time (ms)",
            tickfont=dict(size=9),
        ),
        barmode="stack"
    )

    panel["plot"].update_figure(fig)




def _to_list(v: Any) -> List[Any]:
    """Safely convert iterables / numpy arrays to list."""
    if v is None:
        return []
    try:
        return list(v)
    except Exception:
        return []


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert to float defensively."""
    try:
        return float(v)
    except Exception:
        return default


def _extract_series(metric: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    """
    Extract (steps, sum_values) from renderer metric payload.

    Returns empty lists if data is missing or malformed.
    """
    if not isinstance(metric, dict):
        return [], []

    steps = _to_list(metric.get("steps"))
    summation = metric.get("sum", {}) or {}
    values = _to_list(summation.get("y"))

    if not steps or not values:
        return [], []

    n = min(len(steps), len(values))
    return (
        [int(s) for s in steps[:n]],
        [_safe_float(v) for v in values[:n]],
    )


def _intersect_step_ranges(
    a_steps: List[int],
    b_steps: List[int],
) -> Optional[Tuple[int, int]]:
    """
    Compute the valid intersection [start, end] between two step ranges.

    Returns None if no overlap exists.
    """
    if not a_steps or not b_steps:
        return None

    start = max(a_steps[0], b_steps[0])
    end = min(a_steps[-1], b_steps[-1])

    if start > end:
        return None

    return start, end


def _clip_series(
    steps: List[int],
    values: List[float],
    start: int,
    end: int,
) -> Tuple[List[int], List[float]]:
    """Clip a (steps, values) series to [start, end] inclusive."""
    out_steps, out_vals = [], []
    for s, v in zip(steps, values):
        if start <= s <= end:
            out_steps.append(s)
            out_vals.append(v)
    return out_steps, out_vals


def _empty_figure() -> go.Figure:
    """Baseline empty Plotly figure."""
    fig = go.Figure()
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=6, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10),
        ),
        xaxis=dict(
            title="Training Step",
            showgrid=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="Time (ms)",
            tickfont=dict(size=9),
        ),
    )
    return fig
