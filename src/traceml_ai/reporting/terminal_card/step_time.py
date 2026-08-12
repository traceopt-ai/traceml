# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run-card Step Timing content.

The distributed timing pane selects the row indexed by the stored median
``step_time_ms`` point once.  Every displayed phase is then read from that
same row; this module never combines independent per-metric medians.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from traceml_ai.reporting.terminal_card.common import (
    as_float,
    as_int,
    as_mapping,
    as_sequence,
    clock_label,
)
from traceml_ai.reporting.terminal_card.common import (
    diagnosis as section_diagnosis,
)
from traceml_ai.reporting.terminal_card.common import (
    format_ms,
    format_scope,
    format_share,
    group_row,
    group_rows,
    identity_for_rank,
    point,
    severity_style,
)
from traceml_ai.reporting.terminal_card.layout import (
    STYLE_BOLD,
    STYLE_DIM,
    CardDoc,
    Span,
)

PHASE_SHARE_KINDS = frozenset(
    {"INPUT_BOUND", "H2D_BOUND", "RESIDUAL_HEAVY", "COMPUTE_BOUND"}
)
STRAGGLER_KINDS = frozenset(
    {
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
        "STRAGGLER",
    }
)

# Timing tree rows: (label, metric, depth). Depth drives the tree glyph.
_TREE_ROWS: Tuple[Tuple[str, str, int], ...] = (
    ("Step Time", "step_time_ms", 0),
    ("Input Wait", "input_wait_ms", 1),
    ("Compute", "compute_ms", 1),
    ("H2D", "h2d_ms", 1),
    ("Residual", "residual_ms", 1),
)
_CULPRIT_METRIC_BY_KIND = {
    "INPUT_BOUND": "input_wait_ms",
    "H2D_BOUND": "h2d_ms",
    "RESIDUAL_HEAVY": "residual_ms",
    "COMPUTE_BOUND": "compute_ms",
}
_TREE_LABEL_WIDTH = 19
_MARKER = "◀"


def _rank_scope(
    section: Mapping[str, Any],
    rank: Optional[int],
) -> Optional[str]:
    """Return a compact stored rank/node scope without inferring a node."""
    if rank is None:
        return None
    row_identity = identity_for_rank(section, rank)
    node = as_int(row_identity.get("node_rank"))
    return format_scope(rank=rank, node=node)


def _group_row_for_rank(
    section: Mapping[str, Any], rank: Optional[int]
) -> Mapping[str, Any]:
    """Return the grouped row whose stored identity matches ``rank``.

    Group keys usually equal global ranks, but identity is authoritative for
    compatible payloads whose rows use another key. The renderer never
    substitutes a median or worst point when the diagnosed rank is absent.
    """
    if rank is None:
        return {}
    direct = group_row(section, rank)
    direct_identity = as_mapping(direct.get("identity"))
    if as_int(direct_identity.get("global_rank")) == rank:
        return direct
    for raw_row in group_rows(section).values():
        row = as_mapping(raw_row)
        row_identity = as_mapping(row.get("identity"))
        if as_int(row_identity.get("global_rank")) == rank:
            return row
    return {}


def _why_share(share: Optional[float]) -> Optional[str]:
    """Format one stored ratio as the same compact percentage as the tree."""
    if share is None:
        return None
    return format_share(share, 1.0).strip() or None


def _why_phase_bound(kind: str, diagnosis: Mapping[str, Any]) -> Optional[str]:
    """Format one bound verdict from its stored share and phase."""
    share = _why_share(as_float(diagnosis.get("share_pct")))
    if share is None:
        return None
    if kind == "INPUT_BOUND":
        return f"Input Wait took {share} of Step Time."
    if kind == "H2D_BOUND":
        return f"H2D transfers took {share} of Step Time."
    if kind == "RESIDUAL_HEAVY":
        return f"Residual time took {share} of Step Time."
    if kind != "COMPUTE_BOUND":
        return None
    phase = str(diagnosis.get("phase") or "").strip().lower()
    phase_label = {
        "forward": "Forward",
        "backward": "Backward",
        "optimizer": "Optimizer",
    }.get(phase)
    if phase_label is None:
        return None
    return (
        f"Compute took {share} of Step Time; {phase_label} was the largest "
        "compute phase."
    )


def _why_specific_straggler(
    kind: str,
    diagnosis: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
) -> Optional[str]:
    """Format attributed straggler evidence from the diagnosed rank rows."""
    evidence = as_mapping(diagnosis.get("evidence"))
    culprit_rank = as_int(evidence.get("culprit_rank"))
    victim_rank = as_int(evidence.get("victim_rank"))
    culprit_scope = _rank_scope(step_time_summary, culprit_rank)
    victim_scope = _rank_scope(step_time_summary, victim_rank)
    phase = str(diagnosis.get("phase") or "").strip().lower()
    if kind == "INPUT_STRAGGLER":
        metric, wording = "input_wait_ms", "waited {value} ms for input"
    elif kind == "H2D_STRAGGLER":
        metric, wording = "h2d_ms", "spent {value} ms on H2D transfers"
    elif kind == "COMPUTE_STRAGGLER":
        phase_label = {
            "forward": "Forward",
            "backward": "Backward",
            "optimizer": "Optimizer",
        }.get(phase)
        if phase_label is None:
            return None
        metric = f"{phase}_ms"
        wording = f"spent {{value}} ms in {phase_label}"
    else:
        return None

    culprit_row = _group_row_for_rank(step_time_summary, culprit_rank)
    victim_row = _group_row_for_rank(step_time_summary, victim_rank)
    culprit_value = as_float(
        as_mapping(culprit_row.get("metrics")).get(metric)
    )
    victim_value = as_float(as_mapping(victim_row.get("metrics")).get(metric))
    if (
        culprit_scope is None
        or victim_scope is None
        or culprit_value is None
        or victim_value is None
    ):
        return None
    culprit_text = wording.format(value=f"{culprit_value:.1f}")
    victim_text = wording.format(value=f"{victim_value:.1f}")
    return f"{culprit_scope} {culprit_text}; {victim_scope} {victim_text}."


def _why_generic_straggler(
    diagnosis: Mapping[str, Any], step_time_summary: Mapping[str, Any]
) -> Optional[str]:
    """Format the stored synchronization gap for a generic straggler."""
    evidence = as_mapping(diagnosis.get("evidence"))
    culprit_rank = as_int(evidence.get("culprit_rank"))
    victim_rank = as_int(evidence.get("victim_rank"))
    visible_cost = as_float(evidence.get("visible_cost_ms"))
    metric_label = {
        "backward": "Backward",
        "forward_backward": "Forward + Backward",
    }.get(str(evidence.get("visible_metric") or "").strip().lower())
    culprit_scope = _rank_scope(step_time_summary, culprit_rank)
    victim_scope = _rank_scope(step_time_summary, victim_rank)
    if (
        culprit_scope is None
        or victim_scope is None
        or visible_cost is None
        or metric_label is None
    ):
        return None
    return (
        f"{culprit_scope} was the straggler; {victim_scope} spent "
        f"{visible_cost:.1f} ms longer in {metric_label} waiting at "
        "synchronization. No measured component clearly explained the gap."
    )


def format_run_why(
    primary: Mapping[str, Any], step_time_summary: Mapping[str, Any]
) -> str:
    """Return a concise Run explanation using only stored summary fields.

    Each specialized sentence is emitted only when all fields required by its
    stored diagnosis are available. Older or partial payloads retain their
    original top-level summary verbatim instead of inviting inference here.
    """
    fallback = str(primary.get("summary") or "")
    kind = str(primary.get("kind") or "")
    diagnosis = section_diagnosis(step_time_summary)

    if kind in PHASE_SHARE_KINDS:
        return _why_phase_bound(kind, diagnosis) or fallback
    if kind in STRAGGLER_KINDS - {"STRAGGLER"}:
        return (
            _why_specific_straggler(kind, diagnosis, step_time_summary)
            or fallback
        )
    if kind == "STRAGGLER":
        return _why_generic_straggler(diagnosis, step_time_summary) or fallback
    if kind == "LOW_GPU_UTILIZATION_UNEXPLAINED":
        util = as_float(
            as_mapping(primary.get("evidence")).get("gpu_util_avg_percent")
        )
        if util is None:
            return fallback
        util_text = f"{util:.1f}".rstrip("0").rstrip(".")
        return (
            f"GPU utilization averaged {util_text}%; the measured Step Time "
            "phases did not identify why."
        )
    if kind == "NO_CLEAR_PERFORMANCE_BOTTLENECK":
        return "The measured Step Time phases did not identify a material bottleneck."
    if kind == "INSUFFICIENT_STEP_TIME_DATA":
        evidence = as_mapping(primary.get("evidence"))
        status = (
            str(evidence.get("step_time_status") or "")
            .replace("_", " ")
            .strip()
            .upper()
        )
        if status == "INCOMPLETE DATA":
            missing = [
                str(name).strip()
                for name in as_sequence(
                    as_mapping(diagnosis.get("evidence")).get(
                        "missing_signals"
                    )
                )
                if str(name).strip()
            ]
            if missing:
                return f"Missing timing signals: {', '.join(missing)}."
            return fallback
        steps = as_int(evidence.get("steps_analyzed"))
        if steps is not None:
            noun = "step was" if steps == 1 else "steps were"
            return f"Only {steps} completed {noun} available, so the diagnosis is not stable."
    return fallback


def run_why_and_next(
    primary: Mapping[str, Any], step_time_summary: Mapping[str, Any]
) -> Tuple[str, str]:
    """Return the Run-only explanation and the stored action unchanged."""
    return format_run_why(primary, step_time_summary), str(
        primary.get("action") or ""
    )


def _timing_source(
    step_time_summary: Mapping[str, Any], *, multi: bool
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Optional[str]]:
    """Return one coherent timing source, its identity, and an error.

    A distributed card selects the rank indexed by median Step Time once and
    reads every displayed phase from that row. It never combines independent
    per-metric medians into a synthetic median-rank row.
    """
    if not multi:
        return (
            as_mapping(
                as_mapping(step_time_summary.get("global")).get("average")
            ),
            {},
            None,
        )

    median_point = point(step_time_summary, "median", "step_time_ms")
    idx = median_point.get("idx")
    if as_float(median_point.get("value")) is None or idx is None:
        return {}, {}, "STEP TIMING unavailable: no median rank was selected."
    row = group_row(step_time_summary, idx)
    metrics = as_mapping(row.get("metrics"))
    row_identity = as_mapping(row.get("identity"))
    if not row or not metrics:
        return (
            {},
            row_identity,
            "STEP TIMING unavailable: selected rank row missing.",
        )
    return metrics, row_identity, None


def _tree_glyph(depth: int, last: bool) -> str:
    """Return the tree glyph prefix for one row."""
    if depth == 0:
        return ""
    branch = "└─ " if last else "├─ "
    return branch if depth == 1 else f"   {branch}"


def _source_float(source: Mapping[str, Any], metric: str) -> Optional[float]:
    """Read one nullable numeric value from a selected presentation source."""
    return as_float(source.get(metric))


def _median_rank_caption(
    step_time_summary: Mapping[str, Any], identity: Mapping[str, Any]
) -> str:
    """Return the caption for the row selected by median Step Time."""
    rank = as_int(identity.get("global_rank"))
    node = as_int(identity.get("node_rank"))
    clock = clock_label(step_time_summary)
    if rank is None:
        selection = "Median Rank Unavailable"
    else:
        if node is None:
            node = as_int(
                identity_for_rank(step_time_summary, rank).get("node_rank")
            )
        scope = format_scope(rank=rank, node=node)
        selection = f"Median {scope}" if scope else "Median Rank Unavailable"
    return f"STEP TIMING ({selection}), {clock} Clock"


def _compute_children(source: Mapping[str, Any]) -> List[Tuple[str, float]]:
    """Return measured Compute children from the selected coherent row."""
    children: List[Tuple[str, float]] = []
    for label, metric in (
        ("Forward", "forward_ms"),
        ("Backward", "backward_ms"),
        ("Optimizer", "optimizer_ms"),
    ):
        value = _source_float(source, metric)
        if value is not None:
            children.append((label, value))
    return children


def _tree_row_text(
    label: str, *, ms_text: str, share: str, suffix: Optional[str]
) -> str:
    """Assemble one fixed-column timing tree row."""
    text = f"{label:<{_TREE_LABEL_WIDTH}}{ms_text} ms  {share:>4}"
    if suffix:
        text = f"{text}  {suffix}"
    return text


def _append_compute_children(
    doc: CardDoc,
    *,
    source: Mapping[str, Any],
    step_time_ms: float,
    later_sibling: bool,
) -> None:
    """Append coherent Forward/Backward/Optimizer rows below Compute."""
    children = _compute_children(source)
    for index, (label, value) in enumerate(children):
        last = index == len(children) - 1
        ancestor = "│  " if later_sibling else "   "
        branch = "└─ " if last else "├─ "
        doc.text(
            _tree_row_text(
                f"{ancestor}{branch}{label}",
                ms_text=format_ms(value),
                share=format_share(value, step_time_ms),
                suffix=None,
            )
        )


def _culprit_suffix() -> str:
    """Return the bound-phase culprit marker text."""
    return f"{_MARKER}  cause"


def build_run_step_time_pane(
    primary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    *,
    multi: bool,
    width: int = 78,
) -> CardDoc:
    """Build the Run Step Timing pane from the selected stored timing row."""
    doc = CardDoc(width=width)
    source, row_identity, error = _timing_source(
        step_time_summary, multi=multi
    )
    if error:
        doc.wrapped(error, style=STYLE_DIM)
        return doc

    values = {
        metric: _source_float(source, metric)
        for _label, metric, _depth in _TREE_ROWS
    }
    total = values.get("step_time_ms")
    if total is None or total <= 0.0:
        return doc

    kind = str(primary.get("kind") or "")
    present = [
        (label, metric, depth)
        for label, metric, depth in _TREE_ROWS
        if values.get(metric) is not None
    ]
    last_by_depth: Dict[int, str] = {}
    for _label, metric, depth in present:
        last_by_depth[depth] = metric

    culprit_metric = _CULPRIT_METRIC_BY_KIND.get(kind)
    if kind in STRAGGLER_KINDS:
        # A diagnosed straggler is not a phase on the median Step Time row.
        # Its culprit/victim evidence is already stated in Why.
        culprit_metric = None

    culprit_style = severity_style(primary.get("severity"))
    caption = (
        _median_rank_caption(step_time_summary, row_identity)
        if multi
        else f"STEP TIMING (Window Average), {clock_label(step_time_summary)} Clock"
    )
    doc.text(caption, STYLE_BOLD)

    for label, metric, depth in present:
        value = values[metric]
        glyph = _tree_glyph(depth, last_by_depth.get(depth) == metric)
        suffix = (
            _culprit_suffix()
            if metric == culprit_metric and metric != "step_time_ms"
            else None
        )
        text = _tree_row_text(
            f"{glyph}{label}",
            ms_text=format_ms(float(value or 0.0)),
            share=format_share(value, total),
            suffix=suffix,
        )
        if suffix:
            doc.add(Span(text[: -len(suffix)]), Span(suffix, culprit_style))
        else:
            doc.text(text)

        if metric == "compute_ms":
            _append_compute_children(
                doc,
                source=source,
                step_time_ms=total,
                later_sibling=any(
                    values.get(name) is not None
                    for name in ("h2d_ms", "residual_ms")
                ),
            )

    fetch = _source_float(source, "dataloader_fetch_cpu_ms")
    if fetch is not None:
        doc.text(f"DataLoader fetch: {fetch:.1f} ms (CPU, supplemental)")
    return doc


__all__ = [
    "PHASE_SHARE_KINDS",
    "STRAGGLER_KINDS",
    "build_run_step_time_pane",
    "format_run_why",
    "run_why_and_next",
]
