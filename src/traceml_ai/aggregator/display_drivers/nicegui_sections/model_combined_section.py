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

# Fixed ribbon width for a phase that was never measured this window --
# large enough to read as a deliberate marker, not a rounding artifact of
# a real proportional segment.
_UNMEASURED_SLIVER_PCT = 6.0

# Derived remainder, never carried per-rank: its availability comes from
# the aggregate window (the engine emits it only when derivable), so the
# per-rank coverage check below does not apply to it.
_DERIVED_METRICS = frozenset(("residual_proxy",))

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
    metrics: List[StepCombinedTimeMetric],
) -> Dict[str, StepCombinedTimeMetric]:
    return {m.metric: m for m in metrics}


def _partially_covered(
    per_rank_timing: Dict[int, Dict[str, float]],
    keys: List[str],
) -> List[str]:
    """Return signals measured on some observed ranks but not all.

    Mirrors the diagnosis engine's rule (a signal counts as measured only
    when every observed rank measured it). Only h2d is exempt, matching
    ``_missing_signal_report``: occurrence-driven means an absent event is
    no observed transfer, and that is true per rank too. Derived
    remainders are never carried per-rank, and with fewer than two ranks
    there is nothing to compare, so nothing is reported.

    ``step_time`` is checked alongside the ribbon phases even though it is
    not one: it is the envelope every share is computed against, so a rank
    missing it makes the window incomplete in exactly the way the engine
    reports.
    """
    if len(per_rank_timing) < 2:
        return []
    thin: List[str] = []
    for key in keys:
        if key == "h2d" or key in _DERIVED_METRICS:
            continue
        seen = sum(1 for vals in per_rank_timing.values() if key in vals)
        if 0 < seen < len(per_rank_timing):
            thin.append(key)
    return thin


_EXPIRED_SIG = "__expired__"
_NO_ENVELOPE_SIG = "__no_envelope__"


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


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[StepCombinedTimeResult]
) -> None:
    if not payload or not getattr(payload, "diagnosis_metrics", None):
        if payload is not None and getattr(payload, "had_ok", False):
            # Had data before, none now: the run stopped reporting and the
            # computer's stale window expired (the CLI sibling does the
            # same).
            _clear_view(panel, _EXPIRED_SIG, "window expired")
        return
    m = _index(payload.diagnosis_metrics)
    if "step_time" not in m:
        # No step envelope means no denominator, so every phase share
        # would be invented. Clear rather than leave the previous ribbon
        # and KPIs standing as if they described this window.
        _clear_view(panel, _NO_ENVELOPE_SIG, "step envelope unavailable")
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
    # Aggregate presence is not coverage: a metric measured on rank 0 and
    # absent on rank 1 still appears in the aggregate window, while the
    # canonical diagnosis reports INCOMPLETE DATA for it. Re-derive
    # coverage per rank so this card cannot claim complete data the
    # engine is calling incomplete.
    thin = _partially_covered(
        getattr(payload, "per_rank_timing", None) or {},
        [key for _, key, _ in theme.PHASES] + ["step_time"],
    )
    # Only h2d is exempt. Occurrence-driven governs INTERMITTENT presence
    # (seen at least once, gaps zero-filled); a phase never seen at all is
    # unavailable even when it is occurrence-driven, which is what
    # `_rank_metric_availability` does by only considering metrics it
    # actually observed.
    unmeasured = [key for key in missing if key != "h2d"]
    partial = bool(unmeasured or thin)
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
    reserved_pct = _UNMEASURED_SLIVER_PCT * len(unmeasured)
    measured_scale = max(0.0, (100.0 - reserved_pct) / 100.0)

    sig = (
        tuple(
            round(vals[k], 3) if vals[k] is not None else None
            for _, k, _ in theme.PHASES
        )
        + tuple(sorted(thin))
        + (
            round(float(st.median_total or 0), 3),
            round(float(st.worst_total or 0), 3),
            int(st.steps_used or 0),
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

    k = panel["kpis"]
    # Step Time metrics are already selected-clock per-step averages.
    k["median"].content = theme.kval(
        f"{float(st.median_total or 0):.0f}", "ms"
    )
    k["worst"].content = theme.kval(f"{float(st.worst_total or 0):.0f}", "ms")
    k["gap"].content = theme.kval(f"{float(st.skew_pct or 0):.0f}", "%")
    # A share needs BOTH a numerator and a trustworthy denominator.
    # residual_proxy derives from step_time and the compute phases, so it
    # survives an unmeasured input_wait -- but the envelope does not: it
    # substitutes zero for the missing wait and is then short by an
    # unknown amount. Reporting a share against it would state a confident
    # percentage of a total we do not know.
    denominator_is_whole = vals.get("input_wait") is not None
    residual_value = vals.get("residual_proxy")
    if residual_value is not None and tot > 0 and denominator_is_whole:
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
        incomplete = set(unmeasured) | set(thin)
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
