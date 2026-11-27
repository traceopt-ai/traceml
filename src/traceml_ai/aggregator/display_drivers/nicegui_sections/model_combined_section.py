"""Step Time phase breakdown for the dashboard.

The tree reads the diagnostic reference rank's averages from the analysis
window. Each phase share uses that rank's Step Time as its denominator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from traceml_ai.step_time.model import StepTimeValues, StepTimeWindow
from traceml_ai.step_time.pipeline import LiveStepTimeResult

# (tree prefix, label, canonical metric key). Compute is the sum of its three
# children, so it appears once with them nested below it.
_TREE: Tuple[Tuple[str, str, str], ...] = (
    ("├─ ", "Input wait", "input_wait"),
    ("├─ ", "Compute", "compute"),
    ("│  ├─ ", "Forward", "forward"),
    ("│  ├─ ", "Backward", "backward"),
    ("│  └─ ", "Optimizer", "optimizer_step"),
    ("├─ ", "H2D", "h2d"),
    ("└─ ", "Residual", "residual_proxy"),
)
# Signals checked for partial coverage, named as the tree names them.
_COVERAGE: Tuple[Tuple[str, str], ...] = (
    ("input_wait", "Input wait"),
    ("forward", "Forward"),
    ("backward", "Backward"),
    ("optimizer_step", "Optimizer"),
    ("residual_proxy", "Residual"),
    ("step_time", "Step Time"),
)

_ROW = (
    "<tr><td class='tml-tree-name'>{name}</td>"
    "<td class='tml-num'>{value}</td><td class='tml-num'>{extra}</td></tr>"
)


def build_model_combined_section() -> Dict[str, Any]:
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:22px 24px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:14px; gap:12px;")
        ):
            ui.label("Step Time").classes("ctitle")
            ui.element("div").style("flex:1;")
            win = ui.label("waiting for steps").classes("cmeta")
        tree = ui.html("", sanitize=False).classes("w-full")

    return {"win": win, "tree": tree, "_last_sig": None}


def _show(panel: Dict[str, Any], win: str, tree: str) -> None:
    sig = (win, tree)
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig
    panel["win"].text = win
    panel["tree"].content = tree


def _ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _tree_html(title: str, values: StepTimeValues) -> str:
    """One rank's phases with shares of that rank's own Step Time.

    Unavailable values stay ``n/a`` and are never rescaled into the others.
    Without a positive Step Time there is no denominator, so no shares.
    """
    step_ms = values.step_time_ms

    def share(value: Optional[float]) -> str:
        if step_ms is None or step_ms <= 0.0:
            return ""
        if value is None:
            return "—"
        return f"{value / step_ms * 100.0:.0f}%"

    rows: List[str] = [
        _ROW.format(name="Step Time", value=_ms(step_ms), extra=share(step_ms))
    ]
    for prefix, label, key in _TREE:
        value = values.value(key)
        if key == "h2d" and value is None:
            # H2D is occurrence-driven: no copies in the window is a
            # measured zero, as everywhere else Step Time is read.
            value = 0.0
        rows.append(
            _ROW.format(
                name=f"{prefix}{label}", value=_ms(value), extra=share(value)
            )
        )
    return (
        f"<div class='tml-tree-title'>{title}</div>"
        f"<table class='tml-tree'>{''.join(rows)}</table>"
    )


def _note(text: str) -> str:
    return f"<div class='tml-tree-note'>{text}</div>"


def _partial_names(
    window: StepTimeWindow, values: StepTimeValues
) -> List[str]:
    """Signals missing on the reference rank or on only some ranks."""
    universe_size = len(window.rank_universe)
    return [
        label
        for key, label in _COVERAGE
        if (key != "step_time" and values.value(key) is None)
        or 0 < len(window.ranks_for(key)) < universe_size
    ]


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[LiveStepTimeResult]
) -> None:
    if payload is None or payload.freshness == "cold":
        return
    analysis = payload.analysis
    window = analysis.window
    if payload.freshness == "expired" or not window.metrics:
        # A stale complete view left on screen is read as current.
        _show(panel, "window expired", "")
        return
    clock = f"{window.clock.upper()} clock"
    if not any(m.metric == "step_time" for m in window.metrics):
        _show(panel, f"{clock} · Step Time unavailable", "")
        return

    win = f"{int(window.coverage.steps_used)} steps · {clock}"
    reference_rank = analysis.diagnosis.reference_rank
    facts = window.rank(reference_rank) if reference_rank is not None else None
    if facts is None:
        _show(
            panel,
            win,
            _note("Phase breakdown unavailable for the reference rank."),
        )
        return

    values = facts.average
    title = "Phase breakdown"
    if len(window.rank_universe) > 1:
        title += f" · Reference rank {reference_rank}"
    tree = _tree_html(title, values)
    if values.step_time_ms is None or values.step_time_ms <= 0.0:
        tree += _note("No phase shares without a positive Step Time.")
    partial = _partial_names(window, values)
    if partial:
        win += f" · partial: {', '.join(partial)}"
    _show(panel, win, tree)
