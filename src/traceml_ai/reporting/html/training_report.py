# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Compact diagnosis card for a tracker run (W&B, MLflow).

One self-contained compact HTML document, sized for an iframe inside a
tracker's run page. Top to bottom: a coverage line (run name, ranks,
steps analyzed), the stored ``primary_diagnosis`` status chip, the Run
Why as the headline, the stored action, the step-timing tree with the
cause marked, and a footer. The Why and the tree come from the terminal
card's shared readers, so both surfaces state the same verdict and rows.
The card consumes the decided verdict only; it never selects or
recomputes a diagnosis. It carries no raw JSON, no host details, no
per-rank tables, no scripts and no links.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from traceml_ai import __version__

from ..terminal_card.card import summary_header_coverage
from ..terminal_card.common import (
    as_mapping,
    plural,
    resolve_multi_process,
    steps_analyzed,
)
from ..terminal_card.step_time import (
    StepTimingTree,
    run_why_and_next,
    step_timing_tree,
)
from .banner import severity_tier
from .style import PALETTE
from .textutils import esc, fmt_ms_compact

# Card layout only; colour tokens come from the full report's PALETTE so
# the chip and cause colours match the report banner.
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
  .cov { color:var(--muted); font-size:12px; margin:0 0 8px; }
  .muted { color:var(--muted); }
  .cap { color:var(--muted); font-size:12px; margin-top:12px; }
  table { border-collapse:collapse; margin-top:4px; font-size:12.5px;
    line-height:1.25; }
  td { padding:0; }
  td.lbl { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
    monospace; white-space:pre; padding-right:18px; }
  td.ms { text-align:right; font-variant-numeric:tabular-nums; }
  td.cause { padding-left:12px; font-weight:600; }
  .cause.crit{color:var(--crit);} .cause.warn{color:var(--warn);}
  .cause.good{color:var(--good);} .cause.info{color:var(--info);}
  .cause.neutral{color:var(--neutral);}
  footer { margin-top:10px; font-size:11px; color:var(--muted); }
"""

# Display text for the raw stored diagnosis clock.
_CLOCK_TEXT = {"CPU": "CPU wall clock", "GPU": "CUDA events"}


def _primary(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the stored primary diagnosis when it has displayable content."""
    primary = payload.get("primary_diagnosis")
    if isinstance(primary, Mapping) and any(
        primary.get(key) for key in ("kind", "status", "summary")
    ):
        return primary
    return None


def _coverage(payload: Mapping[str, Any], step_time: Mapping[str, Any]) -> str:
    parts: List[str] = []
    run_name = as_mapping(payload.get("meta")).get("run_name")
    if run_name:
        parts.append(esc(run_name))
    ranks = summary_header_coverage(payload).ranks
    if ranks:
        parts.append(esc(ranks))
    steps = steps_analyzed(step_time)
    if steps is not None:
        parts.append(f"{plural(steps, 'step')} analyzed")
    if not parts:
        return ""
    return f'<p class="cov">{" &middot; ".join(parts)}</p>'


def _verdict(
    primary: Optional[Mapping[str, Any]], step_time: Mapping[str, Any]
) -> str:
    if primary is None:
        return '<span class="chip neutral">NO PRIMARY DIAGNOSIS</span>'
    tier = severity_tier(dict(primary))
    status = primary.get("status") or primary.get("kind")
    parts = [f'<span class="chip {tier}">{esc(status)}</span>']
    why, action = run_why_and_next(primary, step_time)
    headline = why or primary.get("summary")
    if headline:
        parts.append(f"<h1>{esc(headline)}</h1>")
    if action:
        parts.append(f'<p class="next"><b>Next</b> {esc(action)}</p>')
    return "".join(parts)


def _caption(tree: StepTimingTree, *, multi: bool) -> str:
    parts = ["Step timing"]
    if multi:
        parts.append(esc(tree.selection))
    parts.append(esc(_CLOCK_TEXT.get(tree.clock, tree.clock)))
    return f'<p class="cap">{" &middot; ".join(parts)}</p>'


def _timing(
    payload: Mapping[str, Any],
    primary: Optional[Mapping[str, Any]],
    step_time: Mapping[str, Any],
) -> str:
    multi = resolve_multi_process(
        as_mapping(payload.get("meta")),
        (
            step_time,
            as_mapping(payload.get("process")),
            as_mapping(payload.get("step_memory")),
        ),
    )
    tree = step_timing_tree(primary or {}, step_time, multi=multi)
    if not tree.rows:
        text = tree.error or "Step timing not available."
        return f'<p class="cap muted">{esc(text)}</p>'
    tier = severity_tier(dict(primary)) if primary is not None else "neutral"
    rows = []
    for row in tree.rows:
        cause = (
            f'<td class="cause {tier}">◀ cause</td>' if row.is_cause else ""
        )
        rows.append(
            f'<tr><td class="lbl">{esc(row.glyph + row.label)}</td>'
            f'<td class="ms">{fmt_ms_compact(row.value_ms)}</td>'
            f"{cause}</tr>"
        )
    return f"{_caption(tree, multi=multi)}<table>{''.join(rows)}</table>"


def _footer(source_label: Optional[str]) -> str:
    # The payload records no producer version, so the footer names the
    # installed version that rendered the card, not the one that wrote it.
    text = f"Rendered by TraceML {esc(__version__)}"
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
    primary = _primary(payload)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, '
        'initial-scale=1">\n'
        "<title>TraceML diagnosis</title>\n"
        f"<style>{PALETTE}{_CARD_STYLE}</style></head><body>\n"
        '<div class="card">\n'
        f"{_coverage(payload, step_time)}\n"
        f"{_verdict(primary, step_time)}\n"
        f"{_timing(payload, primary, step_time)}\n"
        f"{_footer(source_label)}\n"
        "</div></body></html>"
    )


__all__ = ["render_training_report"]
