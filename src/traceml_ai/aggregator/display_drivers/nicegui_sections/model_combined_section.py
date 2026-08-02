"""Step Time analysis dashboard hero.

Signature element: a phase RIBBON (selected-clock average phase proportions)
plus a VERDICT, then a compact step-KPI strip. The ribbon recomposes as the
bottleneck shifts.

The ribbon and KPI strip are driven by the canonical StepTimeWindow carried by
StepTimeResult (``update_model_combined_section``). The VERDICT is NOT
computed here:
it is taken verbatim from the diagnosis engine's step-time ``status`` via
``update_step_verdict`` (fed the model-diagnostics payload), so it is identical
to the Diagnostics rail, the CLI, and final_summary, and tracks any future
change to the diagnosis vocabulary automatically. The card never derives its
own classification — interpretation belongs to the engine.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.step_time.model import (
    StepTimeMetric,
    StepTimeResult,
    StepTimeWindow,
)
from traceml_ai.utils.step_time_window import median_iteration_component_share

from . import theme

# Fixed ribbon width for a phase that was never measured this window --
# large enough to read as a deliberate marker, not a rounding artifact of
# a real proportional segment.
_UNMEASURED_SLIVER_PCT = 6.0

# step_time is the envelope, not a ribbon phase, so it has no entry in
# theme.PHASES. It still needs a name when it is the incomplete signal.
# Matches the CLI table's abbreviation.
_STEP_TIME_LABEL = "STEP"


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
            ui.label("Step time").classes("ctitle")
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
                ("median", "MEDIAN STEP", theme.C_GPU),
                ("worst", "WORST STEP", "#512da8"),
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


def _representative_rank(window: StepTimeWindow) -> Optional[int]:
    """Choose a real median-iteration rank with a coherent phase row."""
    present_phases = [
        key
        for _label, key, _color in theme.PHASES
        if key != "h2d" and window.ranks_for(key)
    ]
    eligible = window.eligible_ranks(("step_time", *present_phases))
    if not eligible:
        return None

    anchors = {
        rank: float(
            window.per_rank_timing[rank].get(
                "total_step",
                window.per_rank_timing[rank]["step_time"],
            )
        )
        for rank in eligible
    }
    middle = float(statistics.median(anchors.values()))
    return min(eligible, key=lambda rank: (abs(anchors[rank] - middle), rank))


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
    st = step_metric.summary
    kpis = panel["kpis"]
    kpis["median"].content = theme.kval(f"{st.median_total:.0f}", "ms")
    kpis["worst"].content = theme.kval(f"{st.worst_total:.0f}", "ms")
    kpis["gap"].content = (
        theme.kval("n/a")
        if st.skew_pct is None
        else theme.kval(f"{st.skew_pct * 100.0:.0f}", "%")
    )
    residual_share = median_iteration_component_share(
        window.per_rank_timing,
        "residual_proxy",
    )
    kpis["residual"].content = (
        theme.kval("n/a")
        if residual_share is None
        else theme.kval(f"{residual_share * 100.0:.0f}", "%")
    )
    kpis["rank"].content = theme.kval(
        f"r{int(st.worst_rank)}" if st.worst_rank is not None else "—"
    )


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[StepTimeResult]
) -> None:
    window = payload.window if payload is not None else None
    if window is None or not window.metrics:
        if payload is not None and getattr(payload, "had_ok", False):
            _clear_view(panel, _EXPIRED_SIG, "window expired")
        return
    m = _index(window.metrics)
    if "step_time" not in m:
        _clear_view(panel, _NO_ENVELOPE_SIG, "step envelope unavailable")
        return

    step_metric = m["step_time"]
    st = step_metric.summary
    _update_kpis(panel, window, step_metric)
    representative = _representative_rank(window)
    if representative is None:
        _clear_ribbon(
            panel,
            _NO_COHORT_SIG,
            f"{int(st.steps_used)} aligned steps · no coherent phase cohort",
        )
        return

    rank_values = window.per_rank_timing[representative]
    vals: Dict[str, Optional[float]] = {}
    for _label, key, _color in theme.PHASES:
        if key == "h2d":
            vals[key] = float(rank_values.get(key, 0.0))
        else:
            value = rank_values.get(key)
            vals[key] = float(value) if value is not None else None

    measured = {key: value for key, value in vals.items() if value is not None}
    unmeasured = [
        key for key, value in vals.items() if value is None and key != "h2d"
    ]
    universe_size = len(window.rank_universe)
    partial_metrics = [
        key
        for key in [phase[1] for phase in theme.PHASES] + ["step_time"]
        if key != "h2d" and 0 < len(window.ranks_for(key)) < universe_size
    ]

    input_wait_value = vals.get("input_wait")
    envelope = float(rank_values["step_time"]) + (
        input_wait_value if input_wait_value is not None else 0.0
    )
    tot = max(sum(measured.values()), envelope) or 1.0

    # A measured-zero segment and a genuinely unmeasured one must never
    # render identically -- an absent input_wait would otherwise let the
    # OTHER phases sum to exactly step_time (residual is a remainder), so
    # the ribbon fills to 100% and a dark dataloader stream reads as a
    # confirmed-fast one. Non-h2d unmeasured phases get a fixed hatched
    # sliver instead of a proportional width; measured phases give up
    # that width so the row still totals 100%.
    reserved_pct = _UNMEASURED_SLIVER_PCT * len(unmeasured)
    measured_scale = max(0.0, (100.0 - reserved_pct) / 100.0)

    sig = (
        tuple(
            round(vals[k], 3) if vals[k] is not None else None
            for _, k, _ in theme.PHASES
        )
        + tuple(sorted(partial_metrics))
        + (
            representative,
            round(float(st.median_total), 3),
            round(float(st.worst_total), 3),
            int(st.steps_used),
            int(st.worst_rank if st.worst_rank is not None else -1),
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
            # Unmeasured phase: a hatched sliver in the PHASE'S OWN color
            # marks "unknown, not zero" (a dark stream must never read as
            # confirmed-fast). The color still identifies the phase (blue =
            # forward, gold = residual, matching the legend); the diagonal
            # hatch overlay is what says "unmeasured". No on-sliver text --
            # it would overlap the hatch illegibly at this width, and the
            # window meta already names the missing phases.
            seg.style(
                f"width:{_UNMEASURED_SLIVER_PCT:.1f}%; "
                "background:repeating-linear-gradient(45deg, "
                f"{col} 0 2px, transparent 2px 5px);"
            )
            sl.text = ""
            continue
        pct = (
            (value / tot * 100.0 * measured_scale)
            if value is not None
            else 0.0
        )
        # Always restate the background. Style updates MERGE, so a phase
        # that was hatched while unmeasured would keep the hatch forever
        # once it recovers if this path only set the width.
        seg.style(f"width:{pct:.3f}%; background:{col};")
        sl.text = lab if pct >= 7.0 else ""

    # The verdict is intentionally NOT set here. It is owned by the diagnosis
    # engine and set via update_step_verdict (fed the model-diagnostics
    # payload), so the card never asserts a classification of its own.

    steps_text = f"{int(st.steps_used)} aligned steps"
    steps_text += f" · representative r{representative}"
    if unmeasured or partial_metrics:
        incomplete = set(unmeasured) | set(partial_metrics)
        names = [lab for lab, key, _c in theme.PHASES if key in incomplete]
        # step_time has no ribbon segment, so it is not in PHASES, but it
        # can still be the thing that is incomplete. Name it with the same
        # abbreviation the CLI table uses rather than dropping it and
        # printing a bare "partial:".
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
    payload the Diagnostics rail consumes); missing/empty ticks leave the
    previous verdict untouched rather than blanking it.
    """
    items = (
        diag_payload.get("items") if isinstance(diag_payload, dict) else None
    )
    if not isinstance(items, list):
        return
    for it in items:
        if isinstance(it, dict) and it.get("source") == "step_time":
            status = str(it.get("status") or "").strip()
            if status:
                panel["verdict"].text = status
            return
