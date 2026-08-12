# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Public terminal-card façade and Run 2×2 frame composer.

This module owns profile routing, the header/footer, promoted verdict, and the
fixed Run frame.  Payload-aware section rendering lives in the sibling modules
so contributors can change one pane without tracing through unrelated Run or
Watch presentation code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from traceml_ai.reporting.terminal_card.common import (
    DOT,
    SEVERITY_RANK,
    analysis_window,
    as_int,
    as_mapping,
    as_sequence,
)
from traceml_ai.reporting.terminal_card.common import (
    format_duration as fmt_duration,
)
from traceml_ai.reporting.terminal_card.common import (
    is_multi_node,
    is_multi_process,
    metadata,
    pack_segments,
    plural,
    run_is_multi_process,
    severity,
    severity_label,
    severity_style,
    status_text,
    steps_analyzed,
)
from traceml_ai.reporting.terminal_card.layout import (
    CARD_WIDTH,
    RUN_CARD_WIDTH,
    RUN_INNER_WIDTH,
    RUN_LEFT_PANE_WIDTH,
    RUN_RIGHT_PANE_WIDTH,
    STYLE_BOLD,
    STYLE_BORDER,
    STYLE_CRIT,
    STYLE_DIM,
    STYLE_NEXT,
    STYLE_OK,
    STYLE_PLAIN,
    STYLE_WARN,
    CardDoc,
    CardLine,
    Span,
    append_parallel_panes,
    append_staged_panes,
    card_to_ansi,
    card_to_plain,
)
from traceml_ai.reporting.terminal_card.process import build_run_process_pane
from traceml_ai.reporting.terminal_card.step_memory import (
    build_run_step_memory_pane,
)
from traceml_ai.reporting.terminal_card.step_time import (
    build_run_step_time_pane,
    run_why_and_next,
)
from traceml_ai.reporting.terminal_card.system import (
    build_run_system_pane,
    observed_gpu_count,
)
from traceml_ai.reporting.terminal_card.watch import (
    append_watch_body,
    watch_header_meta,
)

RUN_PROFILE = "run"
WATCH_PROFILE = "watch"

FINAL_SUMMARY_JSON_NAME = "final_summary.json"
FINAL_SUMMARY_HTML_NAME = "final_summary.html"

RUN_TITLE = "TraceML Run Summary"
WATCH_TITLE = "TraceML Watch Summary"

# These style names were directly importable from the former monolithic module.
# Keep that established import surface on the public façade without changing
# ``__all__`` (and therefore existing star-import behavior).
_DIRECT_STYLE_EXPORTS = (
    STYLE_BORDER,
    STYLE_CRIT,
    STYLE_OK,
    STYLE_PLAIN,
    STYLE_WARN,
)


def _rank_coverage_text(
    *,
    meta: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    multi: bool,
) -> Optional[str]:
    """Return rank topology/coverage without inventing missing counts."""
    used = as_int(metadata(step_time_summary).get("global_ranks_used"))
    world_size = as_int(meta.get("world_size"))
    if multi:
        if used is not None and world_size is not None:
            return f"{used}/{world_size} ranks"
        count = world_size if world_size is not None else used
        return plural(count, "rank") if count is not None else None
    if world_size is not None or used is not None:
        return "1 rank"
    return None


def _node_coverage_text(
    *,
    system_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> Optional[str]:
    """Return System node coverage for a multi-process Run."""
    if not multi:
        return None
    section_metadata = metadata(system_summary)
    coverage = str(section_metadata.get("nodes_coverage") or "").strip()
    expected = as_int(section_metadata.get("nodes_expected"))
    observed = as_int(section_metadata.get("nodes_observed"))
    if not coverage and observed is not None and expected is not None:
        coverage = f"{observed}/{expected}"
    if coverage:
        singular = expected == 1 or coverage.endswith("/1")
        return f"{coverage} {'node' if singular else 'nodes'}"
    nodes = observed
    if nodes is None:
        nodes = as_int(meta.get("nodes_observed"))
    return plural(nodes, "node") if nodes is not None else None


def _step_coverage_text(step_time_summary: Mapping[str, Any]) -> Optional[str]:
    """Return stored Step Time count with its actual alignment label."""
    steps = steps_analyzed(step_time_summary)
    if steps is None:
        return None
    alignment = str(analysis_window(step_time_summary).get("alignment") or "")
    if alignment == "common_steps":
        return f"{steps} common steps"
    return f"{steps} steps analyzed"


def _run_header_meta(
    *,
    meta: Mapping[str, Any],
    system_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    duration_s: Optional[float],
    multi: bool,
) -> str:
    """Compose the Run card's identity and coverage line."""
    gpus = observed_gpu_count(system_summary)
    segments: List[Optional[str]] = [
        str(meta.get("run_name")) if meta.get("run_name") else None,
        _rank_coverage_text(
            meta=meta,
            step_time_summary=step_time_summary,
            multi=multi,
        ),
    ]
    if gpus == 0:
        segments.append("CPU only (no GPU detected)")
    elif gpus is not None:
        segments.append(f"{plural(gpus, 'GPU')} observed")
    segments.append(
        _node_coverage_text(
            system_summary=system_summary,
            meta=meta,
            multi=multi,
        )
    )
    steps = _step_coverage_text(step_time_summary)
    duration = fmt_duration(duration_s)
    if steps and duration:
        segments.append(f"{steps} {DOT} {duration}")
    else:
        segments.extend((steps, duration))
    return pack_segments(segments, width=RUN_INNER_WIDTH)


def _append_header(doc: CardDoc, *, title: str, meta_line: str) -> None:
    """Append the shared terminal-card header block."""
    doc.rule()
    doc.text(title, STYLE_BOLD)
    if meta_line:
        # Run names can be long session ids, so coverage wraps rather than
        # losing identity or steps-analyzed to a clip.
        for logical_line in meta_line.splitlines():
            doc.wrapped(logical_line, style=STYLE_DIM)
    doc.rule()


def _append_verdict(doc: CardDoc, primary: Mapping[str, Any]) -> None:
    """Append the promoted stored verdict."""
    status = status_text(primary.get("status"))
    level = severity(primary.get("severity"))
    if level not in {"info", ""}:
        status = f"{status}  ({severity_label(level)})"
    doc.add(Span("Verdict: ", STYLE_BOLD), Span(status, severity_style(level)))


def _append_run_timing_memory_blocks(
    doc: CardDoc,
    *,
    primary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    step_memory_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
    show_timing: bool,
) -> None:
    """Compose the upper fixed Step Timing / Step Memory Run panes."""
    timing = CardDoc(width=RUN_LEFT_PANE_WIDTH + 4)
    if show_timing:
        timing = build_run_step_time_pane(
            primary,
            step_time_summary,
            multi=multi,
            width=RUN_LEFT_PANE_WIDTH + 4,
        )
    memory = build_run_step_memory_pane(
        step_memory_summary,
        meta=meta,
        width=RUN_RIGHT_PANE_WIDTH + 4,
    )
    append_parallel_panes(doc, timing, memory)


def _also_findings(
    *, sections: Sequence[Tuple[str, Mapping[str, Any]]]
) -> List[Tuple[str, str]]:
    """Return up to two secondary warning/critical resource findings."""
    found: List[Tuple[int, str, str]] = []
    for _name, section in sections:
        # Schema 1.7 guarantees diagnosis == issues[0]. That finding is
        # already displayed with the section metrics; later issues stay here.
        for raw_issue in as_sequence(section.get("issues"))[1:]:
            issue = as_mapping(raw_issue)
            level = severity(issue.get("severity"))
            if level not in SEVERITY_RANK:
                continue
            summary = str(issue.get("summary") or "").strip()
            if summary:
                found.append((SEVERITY_RANK[level], summary, level))
    found.sort(key=lambda item: item[0])
    return [(summary, level) for _rank, summary, level in found[:2]]


def _footer_line(
    artifact_hint: Optional[str], html_hint: Optional[str]
) -> str:
    """Return the terminal-card artifact footer line."""
    path = artifact_hint or FINAL_SUMMARY_JSON_NAME
    if html_hint:
        return f"Full evidence: {path} {DOT} {html_hint}"
    return f"Full evidence: {path}  (--html-report)"


def _append_run_body(
    doc: CardDoc,
    *,
    primary: Mapping[str, Any],
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    step_memory_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> None:
    """Append the Run body inside the public header/footer frame."""
    why, action = run_why_and_next(primary, step_time_summary)
    degraded = str(primary.get("kind") or "") == "INSUFFICIENT_STEP_TIME_DATA"

    doc.blank()
    _append_verdict(doc, primary)
    if why:
        doc.wrapped(why, label="Why: ")
    if action:
        doc.wrapped(action, label="Next: ", style=STYLE_NEXT)
    if multi:
        doc.text(
            "Scope: N = node · R = global rank · G = GPU index", STYLE_DIM
        )
    doc.blank()

    _append_run_timing_memory_blocks(
        doc,
        primary=primary,
        step_time_summary=step_time_summary,
        step_memory_summary=step_memory_summary,
        meta=meta,
        multi=multi,
        show_timing=not degraded,
    )
    doc.blank()

    also = _also_findings(
        sections=(
            ("system", system_summary),
            ("process", process_summary),
            ("step_memory", step_memory_summary),
        )
    )
    append_staged_panes(
        doc,
        build_run_system_pane(system_summary),
        build_run_process_pane(process_summary, meta=meta),
    )
    doc.blank()

    if also:
        doc.blank()
        doc.wrapped("Other findings:", style=STYLE_DIM)
        for summary, level in also:
            text = f"! {summary}  ({severity_label(level)})"
            doc.wrapped_with_severity(
                text,
                severity_label=severity_label(level),
                severity_style=severity_style(level),
            )


def build_summary_card(
    *,
    profile: str = RUN_PROFILE,
    primary_diagnosis: Mapping[str, Any],
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    step_memory_summary: Mapping[str, Any],
    duration_s: Optional[float] = None,
    meta: Optional[Mapping[str, Any]] = None,
    artifact_hint: Optional[str] = None,
    html_hint: Optional[str] = None,
) -> CardDoc:
    """Build the terminal card from already-built final-summary payloads."""
    primary = as_mapping(primary_diagnosis)
    system = as_mapping(system_summary)
    process = as_mapping(process_summary)
    step_time = as_mapping(step_time_summary)
    step_memory = as_mapping(step_memory_summary)
    run_meta = as_mapping(meta)

    watch = str(profile or RUN_PROFILE).strip().lower() == WATCH_PROFILE
    doc = CardDoc(width=(CARD_WIDTH if watch else RUN_CARD_WIDTH))
    if watch:
        multi = is_multi_node(run_meta)
        _append_header(
            doc,
            title=WATCH_TITLE,
            meta_line=watch_header_meta(
                meta=run_meta,
                duration_s=duration_s,
                multi=multi,
            ),
        )
        append_watch_body(
            doc,
            system_summary=system,
            process_summary=process,
            meta=run_meta,
            multi=multi,
        )
    else:
        multi = run_is_multi_process(
            run_meta, (step_time, process, step_memory)
        )
        _append_header(
            doc,
            title=RUN_TITLE,
            meta_line=_run_header_meta(
                meta=run_meta,
                system_summary=system,
                step_time_summary=step_time,
                duration_s=duration_s,
                multi=multi,
            ),
        )
        _append_run_body(
            doc,
            primary=primary,
            system_summary=system,
            process_summary=process,
            step_time_summary=step_time,
            step_memory_summary=step_memory,
            meta=run_meta,
            multi=multi,
        )

    doc.blank()
    doc.wrapped(_footer_line(artifact_hint, html_hint), style=STYLE_DIM)
    doc.rule()
    return doc


def build_fallback_card(
    *,
    profile: str = RUN_PROFILE,
    primary_diagnosis: Optional[Mapping[str, Any]] = None,
    artifact_hint: Optional[str] = None,
    html_hint: Optional[str] = None,
) -> CardDoc:
    """Build the truthful minimal card used if rich rendering fails."""
    primary = as_mapping(primary_diagnosis)
    watch = str(profile or RUN_PROFILE).strip().lower() == WATCH_PROFILE
    doc = CardDoc(width=(CARD_WIDTH if watch else RUN_CARD_WIDTH))
    _append_header(
        doc,
        title=(WATCH_TITLE if watch else RUN_TITLE),
        meta_line="",
    )
    doc.blank()
    _append_verdict(doc, primary)
    doc.blank()
    doc.wrapped(_footer_line(artifact_hint, html_hint), style=STYLE_DIM)
    doc.rule()
    return doc


def stdout_supports_color() -> bool:
    """Return whether stdout can carry severity-scoped ANSI color."""
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    return os.environ.get("TERM", "") != "dumb"


def artifact_hint(session_root: Optional[str], name: str) -> str:
    """Return a short session-relative artifact path for the card footer."""
    if not session_root:
        return name
    root = Path(str(session_root))
    parent = root.parent.name
    if parent:
        return f"{parent}/{root.name}/{name}"
    return f"{root.name}/{name}"


def card_hints(
    session_root: Optional[str], *, write_html: bool = False
) -> Dict[str, Optional[str]]:
    """Return the JSON and optional HTML artifact hints for the footer."""
    return {
        "artifact_hint": artifact_hint(session_root, FINAL_SUMMARY_JSON_NAME),
        "html_hint": (
            artifact_hint(session_root, FINAL_SUMMARY_HTML_NAME)
            if write_html
            else None
        ),
    }


def card_profile_from_text(text: str) -> str:
    """Infer the profile of a stored card from its title line."""
    for line in str(text).splitlines()[:4]:
        if WATCH_TITLE in line:
            return WATCH_PROFILE
    return RUN_PROFILE


def build_card_from_payload(
    payload: Mapping[str, Any],
    *,
    profile: str = RUN_PROFILE,
    session_root: Optional[str] = None,
    write_html: bool = False,
) -> CardDoc:
    """Build the card for an already-built final-summary payload."""
    return build_summary_card(
        profile=profile,
        primary_diagnosis=payload.get("primary_diagnosis") or {},
        system_summary=payload.get("system") or {},
        process_summary=payload.get("process") or {},
        step_time_summary=payload.get("step_time") or {},
        step_memory_summary=payload.get("step_memory") or {},
        duration_s=payload.get("duration_s"),
        meta=payload.get("meta") or {},
        **card_hints(session_root, write_html=write_html),
    )


def colorize_stored_card(
    text: str,
    payload: Mapping[str, Any],
    *,
    session_root: Optional[str] = None,
    write_html: bool = False,
) -> str:
    """Return ANSI card text only when a re-render exactly matches storage."""
    if not text or not stdout_supports_color():
        return text

    body = text[:-1] if text.endswith("\n") else text
    suffix = text[len(body) :]
    try:
        doc = build_card_from_payload(
            payload,
            profile=card_profile_from_text(body),
            session_root=session_root,
            write_html=write_html,
        )
        if card_to_plain(doc) != body:
            return text
        return card_to_ansi(doc) + suffix
    except Exception:
        return text


__all__ = [
    "CARD_WIDTH",
    "RUN_CARD_WIDTH",
    "FINAL_SUMMARY_HTML_NAME",
    "FINAL_SUMMARY_JSON_NAME",
    "CardDoc",
    "CardLine",
    "RUN_PROFILE",
    "RUN_TITLE",
    "Span",
    "WATCH_PROFILE",
    "WATCH_TITLE",
    "artifact_hint",
    "build_card_from_payload",
    "build_fallback_card",
    "build_summary_card",
    "card_hints",
    "card_profile_from_text",
    "card_to_ansi",
    "card_to_plain",
    "colorize_stored_card",
    "fmt_duration",
    "is_multi_process",
    "stdout_supports_color",
]
