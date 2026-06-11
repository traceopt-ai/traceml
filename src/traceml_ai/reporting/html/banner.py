# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Verdict banner: select and render the worst existing diagnosis.

The banner only *selects* among the per-section diagnoses already in the
payload; it computes no new severity. Severity vocabulary is the real
payload contract crit | warn | info (diagnostics/common.py). Within the
info tier the presentation is kind-aware, because healthy and absent-data
states are both info severity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .textutils import esc

# Fixed cross-section tie-break order when severities are equal.
_SECTION_ORDER = ("step_time", "step_memory", "system", "process")
_SEVERITY_RANK = {"crit": 3, "warn": 2, "info": 1}
_HEALTHY_KINDS = {"BALANCED", "NORMAL"}
_NEUTRAL_KINDS = {"NO_DATA", "WARMUP", "NO_GPU"}


def select_verdict(
    payload: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Return ``(section_name, diagnosis)`` for the worst existing diagnosis.

    Worst = highest severity rank; ties broken by ``_SECTION_ORDER``.
    Unknown severities rank below ``info``. Returns ``None`` when no section
    carries a diagnosis.
    """
    best: Optional[Tuple[str, Dict[str, Any]]] = None
    best_rank = -1
    for name in _SECTION_ORDER:
        section = payload.get(name) or {}
        diag = section.get("diagnosis")
        if not isinstance(diag, dict):
            continue
        rank = _SEVERITY_RANK.get(str(diag.get("severity") or "").lower(), 0)
        if rank > best_rank:
            best, best_rank = (name, diag), rank
    return best


def severity_tier(diag: Dict[str, Any]) -> str:
    """Map a diagnosis to a CSS/severity tier: crit|warn|good|info|neutral."""
    severity = str(diag.get("severity") or "").lower()
    if severity == "crit":
        return "crit"
    if severity == "warn":
        return "warn"
    if severity == "info":
        kind = str(diag.get("kind") or "").upper()
        if kind in _HEALTHY_KINDS:
            return "good"
        if kind in _NEUTRAL_KINDS:
            return "neutral"
        return "info"
    return "neutral"


def render_banner(payload: Dict[str, Any]) -> str:
    """Render the top-of-page verdict banner."""
    picked = select_verdict(payload)
    if picked is None:
        return (
            '<div class="banner neutral"><span class="kind">NO DATA</span>'
            "<h2>No diagnosis available for this run.</h2></div>"
        )
    section, diag = picked
    tier = severity_tier(diag)
    status = diag.get("status") or diag.get("kind") or "—"
    summary = diag.get("summary") or status
    parts = [
        f'<div class="banner {tier}">',
        f'<span class="kind">{esc(status)}</span>',
        f"<h2>{esc(summary)}</h2>",
    ]
    action = diag.get("action")
    if action:
        parts.append(
            f'<p class="action"><b>Action:</b> {esc(action)} '
            f'<span class="why">&middot; from {esc(section)} diagnosis</span>'
            "</p>"
        )
    parts.append("</div>")
    return "".join(parts)


__all__ = ["render_banner", "select_verdict", "severity_tier"]
