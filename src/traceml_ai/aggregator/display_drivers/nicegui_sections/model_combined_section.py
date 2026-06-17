"""
Step Time analysis — the dashboard hero (PR2 revamp).

Signature element: a phase RIBBON (current-step phase proportions) plus a
plain-language VERDICT sentence ("Compute-bound. optimizer is 62% of step."),
then a compact step-KPI strip. The ribbon recomposes (CSS width transition) as
the bottleneck shifts. Renderer payload (StepCombinedTimeResult) is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
)

from . import theme

_REQUIRED = {k for _, k, _ in theme.PHASES} | {"step_time"}


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
                    ui.label(lab)

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
                ("wait", "WAIT SHARE", theme.C_CPU),
                ("rank", "WORST RANK", "#2e7d32"),
            ]:
                with ui.element("div").classes("kpi").style(f"--acc:{acc};"):
                    ui.label(lab).classes("klab")
                    kpis[key] = ui.html("—").classes("kval")

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


def update_model_combined_section(
    panel: Dict[str, Any], payload: Optional[StepCombinedTimeResult]
) -> None:
    if not payload or not getattr(payload, "metrics", None):
        return
    m = _index(payload.metrics)
    if not _REQUIRED.issubset(m):
        return

    vals = {
        k: float(m[k].summary.median_total or 0.0) for _, k, _ in theme.PHASES
    }
    tot = sum(vals.values()) or 1.0
    st = m["step_time"].summary

    sig = tuple(round(vals[k], 3) for _, k, _ in theme.PHASES) + (
        round(float(st.median_total or 0), 3),
        round(float(st.worst_total or 0), 3),
        int(st.steps_used or 0),
        int(st.worst_rank if st.worst_rank is not None else -1),
    )
    if panel.get("_last_sig") == sig:
        return
    panel["_last_sig"] = sig

    for (lab, key, _c), seg, sl in zip(
        theme.PHASES, panel["seg_divs"], panel["seg_labs"]
    ):
        pct = vals[key] / tot * 100.0
        seg.style(f"width:{pct:.3f}%")
        sl.text = lab if pct >= 7.0 else ""

    dom_key = max(theme.PHASES, key=lambda p: vals[p[1]])[1]
    share = vals[dom_key] / tot if tot > 0 else 0.0
    # Descriptive only: severity/diagnosis lives in the Diagnostics rail
    # (single source of truth); the step-time card never asserts health.
    name = theme.verdict_for(dom_key)[1]
    label = name[:1].upper() + name[1:]
    panel["verdict"].text = (
        f"{label} is the largest phase ({share * 100:.0f}% of step)."
    )

    k = panel["kpis"]
    k["median"].content = theme.kval(
        f"{float(st.median_total or 0):.0f}", "ms"
    )
    k["worst"].content = theme.kval(f"{float(st.worst_total or 0):.0f}", "ms")
    k["gap"].content = theme.kval(f"{float(st.skew_pct or 0):.0f}", "%")
    wsh = vals["wait_proxy"] / tot * 100.0 if tot > 0 else 0.0
    k["wait"].content = theme.kval(f"{wsh:.0f}", "%")
    k["rank"].content = theme.kval(
        f"r{int(st.worst_rank)}" if st.worst_rank is not None else "—"
    )
    panel["win"].text = f"{int(st.steps_used or 0)} aligned steps"
