# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Per-section blocks: diagnosis cards, global table, per-rank table."""

from __future__ import annotations

from typing import Any, Dict, List

from .banner import severity_tier
from .sections_helpers import sorted_rows
from .svg import memory_bars, phase_bar
from .textutils import esc, fmt_value, metric_label

_SECTION_ORDER = ("step_time", "step_memory", "system", "process")
_TITLES = {
    "step_time": "Step time",
    "step_memory": "Step memory",
    "system": "System",
    "process": "Process",
}
_BY_LABEL = {"global_rank": "rank", "node_rank": "node"}
_BADGE = {"crit": "CRIT", "warn": "WARN"}
_OPEN_ROW_LIMIT = 8


def _badge_label(diag: Dict[str, Any]) -> str:
    return _BADGE.get(str(diag.get("severity") or "").lower(), "INFO")


def _diag_card(diag: Dict[str, Any]) -> str:
    tier = severity_tier(diag)
    status = diag.get("status") or diag.get("kind") or "—"
    summary = diag.get("summary") or status
    parts = [
        f'<div class="diag {tier}"><span class="badge {tier}">'
        f"{_badge_label(diag)}</span><b>{esc(status)}</b> &mdash; "
        f"{esc(summary)}"
    ]
    action = diag.get("action")
    if action:
        parts.append(f'<div class="why">Action: {esc(action)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _diag_cards(section: Dict[str, Any]) -> str:
    issues = section.get("issues")
    if not issues:
        diag = section.get("diagnosis")
        issues = [diag] if isinstance(diag, dict) else []
    return "".join(_diag_card(i) for i in issues if isinstance(i, dict))


def _metric_cell(entry: Any, name: str) -> str:
    """A median/worst cell is a {value, idx} object; average is a scalar."""
    if isinstance(entry, dict):
        idx = entry.get("idx")
        suffix = (
            f' <span class="idx">r{esc(idx)}</span>' if idx is not None else ""
        )
        return f'{fmt_value(name, entry.get("value"))}{suffix}'
    return fmt_value(name, entry)


def _global_table(section: Dict[str, Any]) -> str:
    metric_names = (section.get("metadata") or {}).get(
        "section_metric_names"
    ) or []
    glob = section.get("global") or {}
    avg = glob.get("average") or {}
    median = glob.get("median") or {}
    worst = glob.get("worst") or {}
    rows: List[str] = []
    for name in metric_names:
        rows.append(
            f'<tr><td class="metric">{esc(metric_label(name))}</td>'
            f'<td class="num">{_metric_cell(avg.get(name), name)}</td>'
            f'<td class="num">{_metric_cell(median.get(name), name)}</td>'
            f'<td class="num">{_metric_cell(worst.get(name), name)}</td></tr>'
        )
    if not rows:
        return ""
    return (
        "<details open><summary>Global metrics</summary>"
        '<div class="tablewrap"><table>'
        "<tr><th>metric</th><th>average</th><th>median</th><th>worst</th></tr>"
        + "".join(rows)
        + "</table></div></details>"
    )


def _rank_table(section: Dict[str, Any]) -> str:
    metric_names = (section.get("metadata") or {}).get(
        "section_metric_names"
    ) or []
    groups = section.get("groups") or {}
    rows = groups.get("rows") or {}
    if not isinstance(rows, dict) or not rows:
        return ""
    by = _BY_LABEL.get(str(groups.get("by") or ""), "rank")
    header = "".join(f"<th>{esc(metric_label(m))}</th>" for m in metric_names)
    body: List[str] = []
    for label, row in sorted_rows(rows):
        identity = row.get("identity") or {}
        metrics = row.get("metrics") or {}
        cells = "".join(
            f'<td class="num">{fmt_value(m, metrics.get(m))}</td>'
            for m in metric_names
        )
        body.append(
            f"<tr><td>{esc(label)} &middot; "
            f"{esc(identity.get('hostname'))}</td>{cells}</tr>"
        )
    open_attr = " open" if len(body) <= _OPEN_ROW_LIMIT else ""
    return (
        f"<details{open_attr}><summary>Per-{by} window stats "
        f"({len(body)})</summary>"
        f'<div class="tablewrap"><table><tr><th>{by}</th>{header}</tr>'
        + "".join(body)
        + "</table></div></details>"
    )


def _section_bars(
    name: str, section: Dict[str, Any], payload: Dict[str, Any]
) -> str:
    """Section-specific inline chart, between the diagnosis and the tables."""
    if name == "step_time":
        return phase_bar(section)
    if name == "step_memory":
        return memory_bars(section, payload.get("process") or {})
    return ""


def _section_card(
    name: str, section: Dict[str, Any], payload: Dict[str, Any]
) -> str:
    rows = (section.get("groups") or {}).get("rows") or {}
    n_rows = len(rows) if isinstance(rows, dict) else 0
    by = _BY_LABEL.get(
        str((section.get("groups") or {}).get("by") or ""), "rank"
    )
    sub = f"{n_rows} {by}s" if n_rows else "no data"
    return (
        f'<section class="card"><h3>{esc(_TITLES.get(name, name))}</h3>'
        f'<div class="sub">{esc(sub)}</div>'
        f"{_diag_cards(section)}"
        f"{_section_bars(name, section, payload)}"
        f"{_global_table(section)}"
        f"{_rank_table(section)}</section>"
    )


def render_sections(payload: Dict[str, Any]) -> str:
    """Render the four section cards in their fixed display order."""
    return "".join(
        _section_card(name, payload.get(name) or {}, payload)
        for name in _SECTION_ORDER
    )


__all__ = ["render_sections"]
