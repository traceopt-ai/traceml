"""Step Time analysis dashboard hero.

Signature element: a phase RIBBON (selected-clock average phase proportions)
plus a VERDICT, then a compact step-KPI strip. The ribbon recomposes as the
bottleneck shifts.

The ribbon and KPI strip are driven by StepCombinedTimeResult diagnosis
metrics (``update_model_combined_section``). The VERDICT is NOT computed here:
it is taken verbatim from the diagnosis engine's step-time ``status`` via
``update_step_verdict`` (fed the model-diagnostics payload), so it is identical
to the Diagnostics rail, the CLI, and final_summary, and tracks any future
change to the diagnosis vocabulary automatically. The card never derives its
own classification — interpretation belongs to the engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
)

from . import theme

# Fixed ribbon width for a non-h2d phase that was never measured this
# window -- large enough to read as a deliberate marker, not a rounding
# artifact of a real proportional segment.
_UNMEASURED_SLIVER_PCT = 6.0


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
    metrics: List[StepCombinedTimeMetric],
) -> Dict[str, StepCombinedTimeMetric]:
    return {m.metric: m for m in metrics}


_EXPIRED_SIG = "__expired__"


def _mark_expired(panel: Dict[str, Any]) -> None:
    """Had data before, none now: the run stopped reporting and the
    computer's stale window expired. Clear the card instead of freezing
    on the last complete view (the CLI sibling does the same)."""
    if panel.get("_last_sig") == _EXPIRED_SIG:
        return
    panel["_last_sig"] = _EXPIRED_SIG
    for seg, sl in zip(panel["seg_divs"], panel["seg_labs"]):
        seg.style("width:0%;")
        sl.text = ""
    for kpi in panel["kpis"].values():
        kpi.content = theme.kval("—")
    panel["win"].text = "window expired"


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[StepCombinedTimeResult]
) -> None:
    if not payload or not getattr(payload, "diagnosis_metrics", None):
        if payload is not None and getattr(payload, "had_ok", False):
            _mark_expired(panel)
        return
    m = _index(payload.diagnosis_metrics)
    if "step_time" not in m:
        return

    # A metric absent from the payload was never measured this window: it
    # renders as an empty segment instead of freezing the whole card on
    # the last complete view. Measured zeros stay zero-width but count as
    # measured. H2D events are occurrence-driven (no transfers, no
    # events), so an absent H2D counts as measured-zero, not as partial
    # coverage.
    vals: Dict[str, Optional[float]] = {
        k: (float(m[k].summary.median_total or 0.0) if k in m else None)
        for _, k, _ in theme.PHASES
    }
    measured = {k: v for k, v in vals.items() if v is not None}
    st = m["step_time"].summary
    missing = [key for key, value in vals.items() if value is None]
    partial = any(key != "h2d" for key in missing)
    # Denominator: at least the median iteration envelope (input wait +
    # step envelope), so unmeasured time shows as empty ribbon space
    # instead of stretching the measured phases to fill 100%.
    input_wait_value = vals.get("input_wait")
    envelope = float(st.median_total or 0.0) + (
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
    unmeasured_non_h2d = [key for key in missing if key != "h2d"]
    reserved_pct = _UNMEASURED_SLIVER_PCT * len(unmeasured_non_h2d)
    measured_scale = max(0.0, (100.0 - reserved_pct) / 100.0)

    sig = tuple(
        round(vals[k], 3) if vals[k] is not None else None
        for _, k, _ in theme.PHASES
    ) + (
        round(float(st.median_total or 0), 3),
        round(float(st.worst_total or 0), 3),
        int(st.steps_used or 0),
        int(st.worst_rank if st.worst_rank is not None else -1),
    )
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig

    for (lab, key, col), seg, sl in zip(
        theme.PHASES, panel["seg_divs"], panel["seg_labs"]
    ):
        value = vals[key]
        if key in unmeasured_non_h2d:
            # Unmeasured non-h2d phase: a hatched sliver in the PHASE'S OWN
            # color marks "unknown, not zero" (a dark stream must never read
            # as confirmed-fast). The color still identifies the phase (blue
            # = forward, gold = residual, matching the legend); the diagonal
            # hatch overlay is what says "unmeasured". No on-sliver text --
            # it would overlap the hatch illegibly at this width, and the
            # window meta already names the missing phases ("partial:
            # FWD,RESIDUAL").
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
        seg.style(f"width:{pct:.3f}%")
        sl.text = lab if pct >= 7.0 else ""

    # The verdict is intentionally NOT set here. It is owned by the diagnosis
    # engine and set via update_step_verdict (fed the model-diagnostics
    # payload), so the card never asserts a classification of its own.

    k = panel["kpis"]
    # Step Time metrics are already selected-clock per-step averages.
    k["median"].content = theme.kval(
        f"{float(st.median_total or 0):.0f}", "ms"
    )
    k["worst"].content = theme.kval(f"{float(st.worst_total or 0):.0f}", "ms")
    k["gap"].content = theme.kval(f"{float(st.skew_pct or 0):.0f}", "%")
    residual_value = vals.get("residual_proxy")
    if residual_value is not None and tot > 0:
        k["residual"].content = theme.kval(
            f"{residual_value / tot * 100.0:.0f}", "%"
        )
    else:
        k["residual"].content = theme.kval("n/a")
    k["rank"].content = theme.kval(
        f"r{int(st.worst_rank)}" if st.worst_rank is not None else "—"
    )
    steps_text = f"{int(st.steps_used or 0)} aligned steps"
    if partial:
        missing_labels = ",".join(
            lab for lab, key, _c in theme.PHASES if key in unmeasured_non_h2d
        )
        steps_text += f" · partial: {missing_labels}"
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
