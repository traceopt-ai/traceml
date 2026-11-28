"""Step Time analysis dashboard hero.

A compact timing summary (median and worst rank Step Time, or one Step Time
on a single-rank run) followed by a numeric phase tree for one rank.

Everything is read from the canonical window in ``LiveStepTimeResult``; the
card computes no statistics of its own. The tree reads every value, Compute
and Residual included, from the averages of the window's composition rank,
and every share uses that rank's Step Time as its denominator. The step-time
verdict lives in the Diagnostics rail only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from traceml_ai.step_time.model import (
    StepTimeMetric,
    StepTimeValues,
    StepTimeWindow,
)
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
        summary = ui.html("", sanitize=False).classes("w-full")
        tree = ui.html("", sanitize=False).classes("w-full")

    return {"win": win, "summary": summary, "tree": tree, "_last_sig": None}


def _show(panel: Dict[str, Any], win: str, summary: str, tree: str) -> None:
    sig = (win, summary, tree)
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig
    panel["win"].text = win
    panel["summary"].content = summary
    panel["tree"].content = tree


def _ms(value: Optional[float], fmt: str = "{:.1f}") -> str:
    return "n/a" if value is None else f"{fmt.format(value)} ms"


def _summary_html(window: StepTimeWindow, step: StepTimeMetric) -> str:
    """Median and worst rank Step Time, or one value on a single rank."""
    if len(window.rank_universe) <= 1:
        rows = _ROW.format(
            name="Step Time",
            value=_ms(step.median_total, "{:.0f}"),
            extra="",
        )
        return f"<table class='tml-tree'>{rows}</table>"
    worst_scope = (
        f"Rank {int(step.worst_rank)}" if step.worst_rank is not None else "—"
    )
    rows = "".join(
        (
            "<tr><th></th><th class='tml-num'>Step time</th>"
            "<th class='tml-num'>Scope</th></tr>",
            _ROW.format(
                name="Median",
                value=_ms(step.median_total, "{:.0f}"),
                extra="Across ranks",
            ),
            _ROW.format(
                name="Worst",
                value=_ms(step.worst_total, "{:.0f}"),
                extra=worst_scope,
            ),
        )
    )
    return f"<table class='tml-tree'>{rows}</table>"


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
    """Signals missing on the composition rank or on only some ranks."""
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
    window = payload.analysis.window
    if payload.freshness == "expired" or not window.metrics:
        # A stale complete view left on screen is read as current.
        _show(panel, "window expired", "", "")
        return
    clock = f"{window.clock.upper()} clock"
    step = next((m for m in window.metrics if m.metric == "step_time"), None)
    if step is None:
        _show(panel, f"{clock} · Step Time unavailable", "", "")
        return

    win = f"{int(window.coverage.steps_used)} steps · {clock}"
    summary = _summary_html(window, step)
    representative = window.composition_representative_rank
    facts = window.rank(representative) if representative is not None else None
    if facts is None:
        # The summary stays valid without a rank to break down.
        _show(
            panel,
            win,
            summary,
            _note("Phase breakdown unavailable: no coherent phase cohort."),
        )
        return

    values = facts.average
    title = "Phase breakdown"
    if len(window.rank_universe) > 1:
        title += f" · Rank {representative}"
    tree = _tree_html(title, values)
    if values.step_time_ms is None or values.step_time_ms <= 0.0:
        tree += _note("No phase shares without a positive Step Time.")
    partial = _partial_names(window, values)
    if partial:
        win += f" · partial: {', '.join(partial)}"
    _show(panel, win, summary, tree)
