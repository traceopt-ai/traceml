"""Step Time analysis dashboard hero.

Signature element: a phase RIBBON (selected-clock average phase proportions)
plus a VERDICT, then a compact step-KPI strip. The ribbon recomposes as the
bottleneck shifts.

The ribbon and KPI strip read the canonical window from
``LiveStepTimeResult``. The VERDICT is NOT computed here:
it is taken verbatim from the diagnosis engine's step-time ``status`` via
``update_step_verdict`` (fed the model-diagnostics payload), so it is identical
to the Diagnostics rail, the CLI, and final_summary, and tracks any future
change to the diagnosis vocabulary automatically. The card never derives its
own classification — interpretation belongs to the engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.step_time.model import StepTimeMetric, StepTimeWindow
from traceml_ai.step_time.pipeline import LiveStepTimeResult

from . import theme

# Step Time is not a ribbon phase, so it has no entry in theme.PHASES. It
# still needs a clear name when partial rank coverage makes it unavailable.
_STEP_TIME_LABEL = "STEP TIME"


def build_model_combined_section() -> Dict[str, Any]:
    seg_divs: List[Any] = []
    seg_labs: List[Any] = []
    kpis: Dict[str, Any] = {}

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

        with ui.element("div").classes("ribbon"):
            for lab, _key, col in theme.PHASES:
                seg = (
                    ui.element("div")
                    .classes("pseg")
                    .style(f"background:{col}; width:0%;")
                )
                with seg:
                    seg_labs.append(ui.label("").classes("seglab"))
                seg_divs.append(seg)

        with (
            ui.row()
            .classes("w-full")
            .style("gap:14px; margin-top:9px; flex-wrap:wrap;")
        ):
            for lab, _key, col in theme.PHASES:
                with ui.element("div").classes("legchip"):
                    ui.element("div").classes("legdot").style(
                        f"background:{col};"
                    )
                    ui.label(theme.PHASE_LEGEND_LABELS.get(_key, lab))

        with (
            ui.row()
            .classes("items-center")
            .style("gap:12px; margin-top:16px;")
        ):
            verdict = ui.label("analyzing step composition").classes("verdict")

        with (
            ui.row()
            .classes("w-full")
            .style("gap:11px; margin-top:16px; flex-wrap:wrap;")
        ):
            for key, lab, acc in [
                ("median", "MEDIAN STEP TIME", theme.C_GPU),
                ("worst", "WORST STEP TIME", "#512da8"),
                ("gap", "GAP", "#f9a825"),
                ("residual", "RESIDUAL SHARE", theme.C_CPU),
                ("rank", "WORST RANK", "#2e7d32"),
            ]:
                with ui.element("div").classes("kpi").style(f"--acc:{acc};"):
                    ui.label(lab).classes("klab")
                    kpis[key] = ui.html("—", sanitize=False).classes("kval")

    return {
        "seg_divs": seg_divs,
        "seg_labs": seg_labs,
        "win": win,
        "verdict": verdict,
        "kpis": kpis,
        "_last_sig": None,
    }


def _index(
    metrics: List[StepTimeMetric],
) -> Dict[str, StepTimeMetric]:
    return {m.metric: m for m in metrics}


_EXPIRED_SIG = "__expired__"
_NO_ENVELOPE_SIG = "__no_envelope__"
_NO_COHORT_SIG = "__no_cohort__"


def _clear_view(panel: Dict[str, Any], sig: str, label: str) -> None:
    """Blank the ribbon and KPIs, then state why.

    Any path that cannot draw a trustworthy ribbon must clear it rather
    than return early: a stale complete view left on screen is read as
    current, which is the failure this issue exists to remove.
    """
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig
    for seg, sl in zip(panel["seg_divs"], panel["seg_labs"]):
        seg.style("width:0%; background:transparent;")
        sl.text = ""
    for kpi in panel["kpis"].values():
        kpi.content = theme.kval("—")
    panel["win"].text = label


def _clear_ribbon(panel: Dict[str, Any], sig: str, label: str) -> None:
    """Clear an untrustworthy composition while retaining valid KPIs."""
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig
    for seg, sl in zip(panel["seg_divs"], panel["seg_labs"]):
        seg.style("width:0%; background:transparent;")
        sl.text = ""
    panel["win"].text = label


def _update_kpis(
    panel: Dict[str, Any],
    window: StepTimeWindow,
    step_metric: StepTimeMetric,
) -> None:
    """Update independently valid Step Time KPI values."""
    kpis = panel["kpis"]
    kpis["median"].content = theme.kval(
        f"{step_metric.median_total:.0f}", "ms"
    )
    kpis["worst"].content = theme.kval(f"{step_metric.worst_total:.0f}", "ms")
    kpis["gap"].content = (
        theme.kval("n/a")
        if step_metric.skew_pct is None
        else theme.kval(f"{step_metric.skew_pct * 100.0:.0f}", "%")
    )
    residual_share = window.residual_share
    kpis["residual"].content = (
        theme.kval("n/a")
        if residual_share is None
        else theme.kval(f"{residual_share * 100.0:.0f}", "%")
    )
    kpis["rank"].content = theme.kval(
        (
            f"r{int(step_metric.worst_rank)}"
            if step_metric.worst_rank is not None
            else "—"
        )
    )


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[LiveStepTimeResult]
) -> None:
    if payload is None or payload.freshness == "cold":
        return
    window = payload.analysis.window
    if payload.freshness == "expired" or not window.metrics:
        _clear_view(panel, _EXPIRED_SIG, "window expired")
        return
    m = _index(window.metrics)
    if "step_time" not in m:
        _clear_view(
            panel,
            _NO_ENVELOPE_SIG,
            f"{window.clock.upper()} selected · Step Time unavailable",
        )
        return

    step_time_metric = m["step_time"]
    steps_used = window.coverage.steps_used
    _update_kpis(panel, window, step_time_metric)
    representative = window.composition_representative_rank
    if representative is None:
        _clear_ribbon(
            panel,
            _NO_COHORT_SIG,
            f"{int(steps_used)} aligned steps · "
            f"{window.clock.upper()} selected · no coherent phase cohort",
        )
        return

    rank_facts = window.rank(representative)
    if rank_facts is None:
        _clear_ribbon(panel, _NO_COHORT_SIG, "rank facts unavailable")
        return
    rank_values = rank_facts.average
    vals: Dict[str, Optional[float]] = {}
    for _label, key, _color in theme.PHASES:
        if key == "h2d":
            vals[key] = float(rank_values.value(key) or 0.0)
        else:
            value = rank_values.value(key)
            vals[key] = float(value) if value is not None else None

    unmeasured = [
        key for key, value in vals.items() if value is None and key != "h2d"
    ]
    universe_size = len(window.rank_universe)
    partial_metrics = [
        key
        for key in [phase[1] for phase in theme.PHASES] + ["step_time"]
        if key != "h2d" and 0 < len(window.ranks_for(key)) < universe_size
    ]

    step_time_ms = rank_values.step_time_ms
    if step_time_ms is None or step_time_ms <= 0.0:
        _clear_ribbon(
            panel,
            _NO_COHORT_SIG,
            f"{window.clock.upper()} selected · Step Time unavailable for "
            "phase shares",
        )
        return

    sig = (
        tuple(
            round(vals[k], 3) if vals[k] is not None else None
            for _, k, _ in theme.PHASES
        )
        + tuple(sorted(partial_metrics))
        + (
            representative,
            round(float(step_time_metric.median_total), 3),
            round(float(step_time_metric.worst_total), 3),
            int(steps_used),
            int(
                step_time_metric.worst_rank
                if step_time_metric.worst_rank is not None
                else -1
            ),
        )
    )
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig

    for (lab, key, col), seg, sl in zip(
        theme.PHASES, panel["seg_divs"], panel["seg_labs"]
    ):
        value = vals[key]
        if key in unmeasured:
            # The existing window metadata names missing signals. Leave the
            # corresponding width blank rather than inventing a visual share
            # or rescaling the measured phases.
            seg.style("width:0%; background:transparent;")
            sl.text = ""
            continue
        pct = float(value / step_time_ms * 100.0) if value is not None else 0.0
        seg.style(f"width:{pct:.3f}%; background:{col};")
        sl.text = lab if pct >= 7.0 else ""

    # The verdict is intentionally NOT set here. It is owned by the diagnosis
    # engine and set via update_step_verdict (fed the model-diagnostics
    # payload), so the card never asserts a classification of its own.

    steps_text = (
        f"{int(steps_used)} aligned steps · {window.clock.upper()} selected"
    )
    steps_text += f" · representative r{representative}"
    if unmeasured or partial_metrics:
        incomplete = set(unmeasured) | set(partial_metrics)
        names = [lab for lab, key, _c in theme.PHASES if key in incomplete]
        # step_time has no ribbon segment, so it is not in PHASES, but it can
        # still be the incomplete signal. Name it rather than printing a bare
        # "partial:".
        if "step_time" in incomplete:
            names.append(_STEP_TIME_LABEL)
        steps_text += f" · partial: {','.join(names)}"
    panel["win"].text = steps_text


def update_step_verdict(panel: Dict[str, Any], diag_payload: Any) -> None:
    """Set the hero verdict from the diagnosis engine's step-time status.

    Single source of truth: the verdict text is the engine's canonical
    ``status`` string for the step-time domain — the exact value shown by the
    Diagnostics rail, the CLI, and final_summary. The card derives no
    classification of its own, so it tracks any change to the diagnosis
    vocabulary automatically. Fed the model-diagnostics payload (the same
    payload the Diagnostics rail consumes); an unavailable item renders
    ``NO DATA`` instead of preserving a stale verdict.
    """
    items = (
        diag_payload.get("items") if isinstance(diag_payload, dict) else None
    )
    if not isinstance(items, list):
        panel["verdict"].text = "NO DATA"
        return
    for it in items:
        if isinstance(it, dict) and it.get("source") == "step_time":
            status = str(it.get("status") or "").strip()
            if status:
                panel["verdict"].text = status
            return
    panel["verdict"].text = "NO DATA"
