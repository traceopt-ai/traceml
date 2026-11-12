# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Compact diagnosis card for a tracker run (W&B, MLflow).

One self-contained compact HTML document, sized for an iframe inside a
tracker's run page. The verdict comes first: the stored
``primary_diagnosis`` status, summary and action, then coverage, the
phase bar and three Step Time averages. The card consumes the decided
verdict only; it never selects or recomputes a diagnosis. It carries no
raw JSON, no host details, no per-rank tables, no scripts and no links.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from traceml_ai import __version__

from ..terminal_card.card import summary_header_coverage
from ..terminal_card.common import (
    analysis_window,
    as_int,
    as_mapping,
    clock_label,
    section_block,
)
from .banner import severity_tier
from .style import PALETTE
from .svg import phase_bar
from .textutils import esc, fmt_ms_compact

# Card layout only; colour tokens come from the full report's PALETTE so
# the chip colours match the report banner.
_CARD_STYLE = """
  * { box-sizing:border-box; }
  body { margin:0; background:var(--card); color:var(--ink);
    font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    Helvetica,Arial,sans-serif; }
  .card { max-width:720px; padding:14px 16px; }
  .chip { display:inline-block; font-size:11.5px; font-weight:700;
    letter-spacing:.5px; color:#fff; border-radius:4px; padding:2px 8px;
    background:var(--neutral); }
  .chip.crit{background:var(--crit);} .chip.warn{background:var(--warn);}
  .chip.good{background:var(--good);} .chip.info{background:var(--info);}
  .chip.neutral{background:var(--neutral);}
  @media (prefers-color-scheme: dark) { .chip { color:var(--bg); } }
  h1 { font-size:17px; font-weight:650; line-height:1.3; margin:8px 0 4px; }
  p { margin:2px 0; }
  .next b { color:var(--muted); margin-right:4px; }
  .cov { color:var(--muted); font-size:12px; margin:8px 0 10px; }
  svg { display:block; }
  .legend { display:flex; flex-wrap:wrap; gap:4px 14px; margin-top:4px;
    font-size:12px; color:var(--muted); }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px;
    margin-right:5px; vertical-align:-1px; }
  .sw.na { background:transparent; border:1px dashed var(--muted); }
  .legend .na { font-style:italic; }
  table { border-collapse:collapse; margin-top:10px; font-size:12.5px; }
  td { padding:2px 18px 2px 0; }
  td:first-child { color:var(--muted); }
  td:last-child { padding-right:0; text-align:right;
    font-variant-numeric:tabular-nums; }
  footer { margin-top:10px; font-size:11px; color:var(--muted); }
"""

# Step Time averages shown in the table: (average metric key, label).
_ROWS = (
    ("step_time_ms", "Step time"),
    ("input_wait_ms", "Input wait"),
    ("compute_ms", "Compute"),
)


def _primary(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the stored primary diagnosis when it has displayable content."""
    primary = payload.get("primary_diagnosis")
    if isinstance(primary, Mapping) and any(
        primary.get(key) for key in ("kind", "status", "summary")
    ):
        return primary
    return None


def _verdict(primary: Optional[Mapping[str, Any]]) -> str:
    if primary is None:
        return '<span class="chip neutral">NO PRIMARY DIAGNOSIS</span>'
    tier = severity_tier(dict(primary))
    status = primary.get("status") or primary.get("kind")
    parts = [f'<span class="chip {tier}">{esc(status)}</span>']
    summary = primary.get("summary")
    if summary:
        parts.append(f"<h1>{esc(summary)}</h1>")
    action = primary.get("action")
    if action:
        parts.append(f'<p class="next"><b>Next</b> {esc(action)}</p>')
    return "".join(parts)


def _coverage(payload: Mapping[str, Any], step_time: Mapping[str, Any]) -> str:
    coverage = summary_header_coverage(payload)
    window = analysis_window(step_time)
    parts: List[str] = []
    run_name = as_mapping(payload.get("meta")).get("run_name")
    if run_name:
        parts.append(esc(run_name))
    if coverage.ranks:
        parts.append(esc(coverage.ranks))
    if coverage.steps:
        parts.append(esc(coverage.steps))
    start = as_int(window.get("start_step"))
    end = as_int(window.get("end_step"))
    if start is not None and end is not None:
        parts.append(f"steps {start}–{end}")
    if window.get("diagnosis_clock"):
        parts.append(f"{esc(clock_label(step_time))} clock")
    if not parts:
        return ""
    return f'<p class="cov">{" &middot; ".join(parts)}</p>'


def _phases(payload: Mapping[str, Any], step_time: Mapping[str, Any]) -> str:
    # An empty bar renders nothing: the table's Step time row is the single
    # place an unmeasured Step Time is stated.
    return phase_bar(
        dict(step_time),
        schema_version=payload.get("schema_version"),
    )


def _table(step_time: Mapping[str, Any]) -> str:
    average = section_block(step_time, "average")
    rows = "".join(
        f"<tr><td>{esc(label)}</td>"
        f"<td>{fmt_ms_compact(average.get(key))}</td></tr>"
        for key, label in _ROWS
    )
    return f"<table>{rows}</table>"


def _footer(payload: Mapping[str, Any], source_label: Optional[str]) -> str:
    # The payload records no producer version, so the footer names the
    # installed version that rendered the card, not the one that wrote it.
    schema = payload.get("schema_version", "—")
    text = (
        f"Rendered by TraceML {esc(__version__)} &middot; "
        f"schema {esc(schema)} &middot; "
        "averages over the analyzed window; Not measured means no signal, "
        "not zero"
    )
    if source_label:
        text += f" &middot; {esc(source_label)}"
    return f"<footer>{text}</footer>"


def render_training_report(
    payload: Mapping[str, Any],
    *,
    source_label: Optional[str] = None,
) -> str:
    """
    Render a final-summary payload to the compact tracker diagnosis card.

    Pure function of (payload, source_label, installed traceml_ai version):
    no I/O, no clock, no randomness. Every payload-derived string is escaped.
    """
    step_time = as_mapping(payload.get("step_time"))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1">\n'
        "<title>TraceML diagnosis</title>\n"
        f"<style>{PALETTE}{_CARD_STYLE}</style></head><body>\n"
        '<div class="card">\n'
        f"{_verdict(_primary(payload))}\n"
        f"{_coverage(payload, step_time)}\n"
        f"{_phases(payload, step_time)}\n"
        f"{_table(step_time)}\n"
        f"{_footer(payload, source_label)}\n"
        "</div></body></html>"
    )


__all__ = ["render_training_report"]
