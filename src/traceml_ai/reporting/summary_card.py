# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Terminal end-of-run summary card.

This module owns the presentation of the final summary ``text`` field. It is
presentation-only: every value it prints already exists in the final summary
payload and no metric, threshold, or diagnosis is recomputed here.

Two profiles are rendered from one skeleton:

``run``
    verdict, a hierarchical timing tree, the next action, and bounded
    corroboration (supporting utilization, secondary findings, peak memory).

``watch``
    host and process health only. ``watch`` never collects step timing, so the
    card never mentions Step Time, Step Memory, or steps analyzed.

A card is built as a ``CardDoc`` (lines of styled spans) so the plain and the
ANSI renderings share identical padding math. The plain rendering is the one
stored in JSON and in the ``.txt`` artifact; ANSI is used only when printing
to an interactive terminal.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from traceml_ai.reporting.summaries.summary_formatting import bytes_to_gb
from traceml_ai.reporting.summaries.summary_layout import (
    border,
    row,
    wrap_lines,
)

CARD_WIDTH = 78
INNER_WIDTH = CARD_WIDTH - 4

RUN_PROFILE = "run"
WATCH_PROFILE = "watch"

FINAL_SUMMARY_JSON_NAME = "final_summary.json"
FINAL_SUMMARY_HTML_NAME = "final_summary.html"

RUN_TITLE = "TraceML Run Summary"
WATCH_TITLE = "TraceML Watch Summary"

# Terminal presentation contract
# -----------------------------
# - Chrome (borders, run metadata, section captions, and artifact hints) is
#   dim neutral text.
# - Titles and action text are bold in the terminal's default foreground.
# - Measurements, explanations, and evidence rows use the terminal's default
#   foreground; they are never colour-coded as a whole.
# - The primary Verdict and Host health values are colour-coded by severity:
#   normal is green, warning is amber, and critical is red. Secondary severity
#   labels and culprit markers use the same colours, but their surrounding
#   evidence stays neutral.
#
# Keep this contract when extending the card. The stored JSON/text artifacts
# are deliberately plain; ANSI styling is an interactive-terminal enhancement
# only and must not carry information that is absent from plain text.
STYLE_PLAIN = "plain"
STYLE_BORDER = "border"
STYLE_DIM = "dim"
STYLE_BOLD = "bold"
STYLE_CRIT = "sev_crit"
STYLE_WARN = "sev_warn"
STYLE_OK = "sev_ok"
STYLE_NEXT = "next"

_ANSI_CODES = {
    STYLE_BORDER: "\x1b[2m",
    STYLE_DIM: "\x1b[2m",
    STYLE_BOLD: "\x1b[1m",
    STYLE_CRIT: "\x1b[1;31m",
    STYLE_WARN: "\x1b[1;33m",
    STYLE_OK: "\x1b[32m",
    STYLE_NEXT: "\x1b[1m",
}
_ANSI_RESET = "\x1b[0m"

_SEVERITY_LABELS = {
    "crit": "CRITICAL",
    "critical": "CRITICAL",
    "warn": "WARNING",
    "warning": "WARNING",
    "info": "INFO",
}
_SEVERITY_STYLES = {
    "crit": STYLE_CRIT,
    "critical": STYLE_CRIT,
    "warn": STYLE_WARN,
    "warning": STYLE_WARN,
}
_SEVERITY_RANK = {"crit": 0, "critical": 0, "warn": 1, "warning": 1}

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
LOW_GPU_UTIL_KINDS = frozenset(
    {"LOW_GPU_UTILIZATION", "MODERATE_GPU_UTILIZATION"}
)

_DISPLAY_STATUS = {
    "INPUT_BOUND": "INPUT-BOUND",
    "H2D_BOUND": "H2D-BOUND",
    "COMPUTE_BOUND": "COMPUTE-BOUND",
    "RESIDUAL_HEAVY": "RESIDUAL-HEAVY",
    "INPUT_STRAGGLER": "INPUT STRAGGLER",
    "COMPUTE_STRAGGLER": "COMPUTE STRAGGLER",
    "H2D_STRAGGLER": "H2D STRAGGLER",
    "STRAGGLER": "RANK STRAGGLER",
    "LOW_GPU_UTILIZATION_UNEXPLAINED": (
        "LOW GPU UTILIZATION -- cause unclear"
    ),
    "NO_CLEAR_PERFORMANCE_BOTTLENECK": "NO CLEAR BOTTLENECK",
}

# Timing tree rows: (label, metric, depth). Depth drives the tree glyph.
_TREE_ROWS: Tuple[Tuple[str, str, int], ...] = (
    ("Step Time", "step_time_ms", 0),
    ("Input Wait", "input_wait_ms", 1),
    ("Traced Step Time", "traced_step_time_ms", 1),
    ("Compute", "compute_ms", 2),
    ("H2D", "h2d_ms", 2),
    ("Residual", "residual_ms", 2),
)
_CULPRIT_METRIC_BY_KIND = {
    "INPUT_BOUND": "input_wait_ms",
    "H2D_BOUND": "h2d_ms",
    "RESIDUAL_HEAVY": "residual_ms",
    "COMPUTE_BOUND": "compute_ms",
}

_TREE_LABEL_WIDTH = 19
_TREE_WORST_COLUMN = 50
_BAR_COLUMN = 36
_BAR_WIDTH = 28
_BAR_FULL = "█"
_BAR_EMPTY = "░"
_MARKER = "◀"
_DOT = "·"

_WATCH_LEFT_LABEL_WIDTH = 11
_WATCH_RIGHT_COLUMN = 28
_WATCH_RIGHT_LABEL_WIDTH = 9
_WATCH_MULTI_LABEL_WIDTH = 12
_WATCH_MULTI_WORST_COLUMN = 26

_SKEW_MATERIAL_RATIO = 0.20
_SKEW_MATERIAL_DELTA_MS = 1.0
_MEMORY_SKEW_EVEN_RATIO = 0.10
_SUPPORTING_GPU_UTIL_MAX = 50.0

_WATCH_BOUNDARY = (
    "Watch monitors host and process health only; it does not measure "
    "step time. To diagnose training speed: traceml run <your-script>.py"
)
_WATCH_RUN_ACTION = (
    "traceml run <your-script>.py -- measures step time and finds the cause."
)


@dataclass(frozen=True)
class Span:
    """One styled run of visible text inside a card line."""

    text: str
    style: str = STYLE_PLAIN


@dataclass(frozen=True)
class CardLine:
    """One card line: either a horizontal rule or a row of spans."""

    spans: Tuple[Span, ...] = ()
    rule: bool = False


@dataclass
class CardDoc:
    """A rendered-independent card: an ordered list of card lines."""

    lines: List[CardLine] = field(default_factory=list)

    def rule(self) -> None:
        """Append a horizontal border rule."""
        self.lines.append(CardLine(rule=True))

    def blank(self) -> None:
        """Append an empty content row."""
        self.lines.append(CardLine())

    def add(self, *spans: Span) -> None:
        """Append one content row built from styled spans."""
        self.lines.append(CardLine(spans=tuple(spans)))

    def text(self, value: str, style: str = STYLE_PLAIN) -> None:
        """Append one single-style content row."""
        self.add(Span(value, style))

    def wrapped(
        self,
        value: str,
        *,
        label: str = "",
        label_style: str = STYLE_BOLD,
        style: str = STYLE_PLAIN,
    ) -> None:
        """Append a logical line wrapped inside the card, with a label."""
        full = f"{label}{value}" if label else value
        for index, line in enumerate(wrap_lines(full, INNER_WIDTH)):
            if index == 0 and label and line.startswith(label):
                self.add(
                    Span(label, label_style),
                    Span(line[len(label) :], style),
                )
            else:
                self.text(line, style)

    def wrapped_with_severity(
        self,
        value: str,
        *,
        severity_label: str,
        severity_style: str,
    ) -> None:
        """Append wrapped text with only its final severity word coloured."""
        for line in wrap_lines(value, INNER_WIDTH):
            before, marker, after = line.rpartition(severity_label)
            if marker:
                self.add(
                    Span(before),
                    Span(marker, severity_style),
                    Span(after),
                )
            else:
                self.text(line)


def _visible_spans(line: CardLine) -> List[Span]:
    """Return the spans of one row clipped to the card's inner width."""
    out: List[Span] = []
    used = 0
    for span in line.spans:
        if used >= INNER_WIDTH:
            break
        room = INNER_WIDTH - used
        text = span.text[:room]
        used += len(text)
        if text:
            out.append(Span(text, span.style))
    return out


def card_to_plain(doc: CardDoc) -> str:
    """Render a card as plain text with no escape sequences."""
    lines: List[str] = []
    for line in doc.lines:
        if line.rule:
            lines.append(border(width=CARD_WIDTH))
            continue
        text = "".join(span.text for span in _visible_spans(line))
        lines.append(row(text, width=CARD_WIDTH))
    return "\n".join(lines)


def _paint(text: str, style: str) -> str:
    """Wrap one span in its ANSI code, if the style has one."""
    code = _ANSI_CODES.get(style)
    if not code or not text:
        return text
    return f"{code}{text}{_ANSI_RESET}"


def card_to_ansi(doc: CardDoc) -> str:
    """Render a card with severity-scoped ANSI, same geometry as plain."""
    lines: List[str] = []
    edge = _paint("|", STYLE_BORDER)
    for line in doc.lines:
        if line.rule:
            lines.append(_paint(border(width=CARD_WIDTH), STYLE_BORDER))
            continue
        spans = _visible_spans(line)
        visible = sum(len(span.text) for span in spans)
        body = "".join(_paint(span.text, span.style) for span in spans)
        padding = " " * max(0, INNER_WIDTH - visible)
        lines.append(f"{edge}  {body}{padding}{edge}")
    return "\n".join(lines)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping, or an empty mapping for malformed blocks."""
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    """Return a sequence, or an empty tuple for malformed blocks."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _float(value: Any) -> Optional[float]:
    """Return a float when a payload value is numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> Optional[int]:
    """Return an int when a payload value is numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _block(section: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one ``global`` rollup block from a section payload."""
    return _mapping(_mapping(section.get("global")).get(name))


def _window(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's analysis window block."""
    return _block(section, "window")


def _average(section: Mapping[str, Any], metric: str) -> Optional[float]:
    """Return one average metric value."""
    return _float(_block(section, "average").get(metric))


def _point(
    section: Mapping[str, Any],
    block: str,
    metric: str,
) -> Mapping[str, Any]:
    """Return one median/worst point for a metric."""
    return _mapping(_block(section, block).get(metric))


def _point_value(
    section: Mapping[str, Any],
    block: str,
    metric: str,
) -> Optional[float]:
    """Return one median/worst point value for a metric."""
    return _float(_point(section, block, metric).get("value"))


def _identity(section: Mapping[str, Any], idx: Any) -> Mapping[str, Any]:
    """Return the grouped-row identity behind a median/worst index."""
    if idx is None:
        return {}
    rows = _mapping(_mapping(section.get("groups")).get("rows"))
    return _mapping(_mapping(rows.get(str(idx))).get("identity"))


def _rank_of(section: Mapping[str, Any], idx: Any) -> Optional[int]:
    """Return the global rank behind a median/worst index."""
    rank = _int(_identity(section, idx).get("global_rank"))
    return rank if rank is not None else _int(idx)


def _node_of(section: Mapping[str, Any], idx: Any) -> Optional[int]:
    """Return the node rank behind a median/worst index."""
    node = _int(_identity(section, idx).get("node_rank"))
    return node if node is not None else _int(idx)


def _diagnosis(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section diagnosis block."""
    return _mapping(section.get("diagnosis"))


def _severity(value: Any) -> str:
    """Return a normalized severity key."""
    return str(value or "info").strip().lower()


def _severity_label(value: Any) -> str:
    """Return the uppercase display label for a severity."""
    key = _severity(value)
    return _SEVERITY_LABELS.get(key, key.upper() or "INFO")


def _severity_style(value: Any) -> str:
    """Return the span style for a severity."""
    return _SEVERITY_STYLES.get(_severity(value), STYLE_OK)


def _status_text(value: Any) -> str:
    """Return an uppercase status label for an unknown diagnosis kind."""
    text = str(value or "NO DATA").replace("_", " ").strip()
    return " ".join(text.upper().split()) or "NO DATA"


def fmt_duration(duration_s: Optional[float]) -> Optional[str]:
    """Format a run duration as `52.4s`, `5m 12s`, or `1h 4m`."""
    if duration_s is None:
        return None
    seconds = float(duration_s)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    if total < 3600:
        minutes, rest = divmod(total, 60)
        return f"{minutes}m {rest}s"
    hours, rest = divmod(total, 3600)
    return f"{hours}h {rest // 60}m"


def _fmt_ms(value: float) -> str:
    """Format one millisecond value for the timing tree column."""
    return f"{value:>6.1f}"


def _fmt_share(value: Optional[float], total: Optional[float]) -> str:
    """Format a phase share as a 4-character right-aligned percentage."""
    if value is None or total is None or total <= 0.0:
        return ""
    percent = 100.0 * value / total
    text = f"{percent:.0f}%"
    if text == "0%" and value > 0.0:
        text = "<1%"
    return f"{text:>4}"


def _fmt_percent(value: Optional[float]) -> Optional[str]:
    """Format a percentage with no decimals."""
    return None if value is None else f"{value:.0f}"


def _fmt_gb(value: Optional[float]) -> Optional[str]:
    """Format a byte count as decimal gigabytes with one decimal."""
    gb = bytes_to_gb(value)
    return None if gb is None else f"{gb:.1f}"


def _fmt_capacity(value: Optional[float]) -> Optional[str]:
    """
    Format a byte count with a unit that keeps one significant decimal.

    Small training footprints round to `0.0 GB`, which reads as "nothing was
    measured". Anything under 0.1 GB is reported in MB instead, on the same
    decimal basis as ``bytes_to_gb``.
    """
    gb = bytes_to_gb(value)
    if gb is None:
        return None
    if gb < 0.1:
        return f"{float(value or 0.0) / 1e6:.1f} MB"
    return f"{gb:.1f} GB"


def _join_segments(segments: Sequence[Optional[str]]) -> str:
    """Join header/observation segments, dropping missing ones."""
    return f" {_DOT} ".join([text for text in segments if text])


def _plural(count: int, singular: str) -> str:
    """Return `1 GPU` / `2 GPUs` style text."""
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def is_multi_process(step_time_summary: Mapping[str, Any]) -> bool:
    """Return whether Step Time observed more than one global rank."""
    metadata = _mapping(step_time_summary.get("metadata"))
    used = _int(metadata.get("global_ranks_used"))
    return bool(used is not None and used > 1)


def _is_multi_node(meta: Mapping[str, Any]) -> bool:
    """Return whether more than one node was observed."""
    nodes = _int(meta.get("nodes_observed"))
    return bool(nodes is not None and nodes > 1)


def _clock(step_time_summary: Mapping[str, Any]) -> str:
    """Return the selected diagnosis clock label (`GPU` / `CPU`)."""
    clock = str(_window(step_time_summary).get("diagnosis_clock") or "")
    return "CPU" if clock.strip().lower() == "cpu" else "GPU"


def _steps_analyzed(step_time_summary: Mapping[str, Any]) -> Optional[int]:
    """Return the number of analyzed steps behind the Step Time window."""
    return _int(_window(step_time_summary).get("steps_analyzed"))


def _incomplete_timing(primary: Mapping[str, Any]) -> bool:
    """Return whether the insufficient-data primary is a missing-signal one."""
    evidence = _mapping(primary.get("evidence"))
    status = str(evidence.get("step_time_status") or "")
    return _status_text(status) == "INCOMPLETE DATA"


def _display_status(primary: Mapping[str, Any]) -> str:
    """Return the card's verdict status label for a primary diagnosis."""
    kind = str(primary.get("kind") or "")
    if kind == "INSUFFICIENT_STEP_TIME_DATA":
        if _incomplete_timing(primary):
            return "STEP TIMING INCOMPLETE"
        return "NOT ENOUGH STEP DATA"
    display = _DISPLAY_STATUS.get(kind)
    return display if display else _status_text(primary.get("status"))


def _score_percent(primary: Mapping[str, Any]) -> Optional[str]:
    """Return the diagnosis score as an integer percentage string."""
    score = _float(_mapping(primary.get("evidence")).get("score"))
    return None if score is None else f"{score * 100:.0f}"


def _rank_clause(rank: Optional[int], node: Optional[int]) -> str:
    """Return `rank 0 (node n0)` style text for a straggler rank."""
    if rank is None:
        return "the slow rank"
    if node is None:
        return f"rank {rank}"
    return f"rank {rank} (node n{node})"


def _comparison(primary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the rank comparison behind a straggler primary diagnosis."""
    evidence = _mapping(primary.get("evidence"))
    comparisons = _sequence(evidence.get("comparisons"))
    if comparisons:
        return _mapping(comparisons[0])
    return evidence


def _straggler_why(
    primary: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    verb: str,
    phrase: str,
) -> str:
    """Compose the straggler `Why` sentence from rank comparison evidence."""
    comparison = _comparison(primary)
    worst = _mapping(comparison.get("worst"))
    median = _mapping(comparison.get("median"))
    rank = _int(worst.get("rank"))
    node = _int(meta.get("node_rank_of_worst"))
    worst_ms = _float(worst.get("value_ms"))
    median_ms = _float(median.get("value_ms"))
    world_size = _int(meta.get("world_size"))

    who = _rank_clause(rank, node)
    parts = [f"{who} {verb} "]
    parts.append("? ms" if worst_ms is None else f"{worst_ms:.1f} ms")
    parts.append(f" per step {phrase}; the median rank {verb} ")
    parts.append("? ms" if median_ms is None else f"{median_ms:.1f} ms")
    parts.append(".")
    if world_size is not None and rank is not None:
        parts.append(
            f" All {world_size} ranks then advance at rank {rank}'s pace."
        )
    return "".join(parts)


def _phase_word(primary: Mapping[str, Any]) -> str:
    """Return the phase word used in straggler wording."""
    phase = str(_comparison(primary).get("phase") or "").strip()
    return phase or "compute"


def _why_next(
    *,
    primary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> Tuple[str, str]:
    """Compose the plain-language `Why` and `Next` text for a diagnosis."""
    kind = str(primary.get("kind") or "")
    score = _score_percent(primary)
    evidence = _mapping(primary.get("evidence"))

    if kind == "INPUT_BOUND" and score is not None:
        if _clock(step_time_summary) == "CPU":
            why = f"input wait is {score}% of the typical step (CPU clock)."
        else:
            why = (
                f"input wait is {score}% of the typical step; the GPU sits "
                "idle while batches arrive."
            )
        if _int(meta.get("gpus_observed")) == 0:
            # pin_memory only helps host-to-device transfer, so it is not
            # advice worth giving on a machine with no GPU.
            return why, (
                "raise DataLoader num_workers or prefetch, or check storage "
                "read throughput."
            )
        return why, (
            "raise DataLoader num_workers, enable pin_memory / prefetch, or "
            "check storage read throughput."
        )

    if kind == "H2D_BOUND" and score is not None:
        return (
            f"host-to-device copies take {score}% of the typical step.",
            "use pin_memory + non_blocking copies, batch transfers, or move "
            "preprocessing to the GPU.",
        )

    if kind == "COMPUTE_BOUND" and score is not None:
        return (
            f"compute (forward/backward/optimizer) takes {score}% of the "
            "typical step.",
            "training is compute-dominated; for more speed profile kernels "
            "(torch.profiler / Nsight).",
        )

    if kind == "RESIDUAL_HEAVY" and score is not None:
        return (
            f"{score}% of the typical step is time outside the traced "
            "phases.",
            "look for untraced work between steps -- validation, "
            "checkpointing, logging.",
        )

    if kind in STRAGGLER_KINDS:
        return _straggler_why_next(primary, meta=meta, kind=kind)

    if kind == "LOW_GPU_UTILIZATION_UNEXPLAINED":
        util = _fmt_percent(_float(evidence.get("gpu_util_avg_percent")))
        if util is not None:
            # Single-machine cards never use rank/node/skew vocabulary, so
            # the cross-rank clause only appears on multi-rank runs.
            unexplained = (
                "input, transfer, or timing skew."
                if multi
                else "input or transfer."
            )
            why = (
                "step timing is balanced, but GPU util averaged "
                f"{util}%. The idle time is not explained by {unexplained}"
            )
            return why, (
                "look for untraced work between steps -- validation, "
                "checkpointing, logging -- or inefficient kernels "
                "(torch.profiler)."
            )

    if kind == "NO_CLEAR_PERFORMANCE_BOTTLENECK":
        skew = _step_time_skew_percent(step_time_summary) if multi else None
        if multi and skew is not None:
            why = (
                "step timing is balanced and ranks are even (worst step "
                f"time +{skew} vs median)."
            )
        else:
            why = (
                "step timing is balanced -- no input, transfer, or residual "
                "issue."
            )
        return why, (
            "training is compute-dominated; for more speed profile kernels "
            "(torch.profiler / Nsight)."
        )

    if kind == "INSUFFICIENT_STEP_TIME_DATA":
        if _incomplete_timing(primary):
            names = _missing_signals(step_time_summary)
            if names:
                why = (
                    "some timing signals were never measured (missing: "
                    f"{names}), so phases don't add up to a reliable step "
                    "time."
                )
            else:
                why = (
                    "some timing signals were never measured, so phases "
                    "don't add up to a reliable step time."
                )
            return why, (
                "check the integration wiring for the missing phases; "
                "per-signal coverage is listed in the JSON step_time section."
            )
        steps = _steps_analyzed(step_time_summary)
        if steps:
            why = (
                f"only {steps} completed steps were captured; a stable "
                "diagnosis needs a larger sample."
            )
        else:
            why = "no completed steps were captured."
        return why, (
            "run more steps, or check that traceml.trace_step(...) wraps "
            "the training loop."
        )

    return (
        str(primary.get("summary") or ""),
        str(primary.get("action") or ""),
    )


def _straggler_why_next(
    primary: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    kind: str,
) -> Tuple[str, str]:
    """Compose `Why` and `Next` for the rank-straggler diagnoses."""
    comparison = _comparison(primary)
    rank = _int(_mapping(comparison.get("worst")).get("rank"))
    node = _int(meta.get("node_rank_of_worst"))
    who = _rank_clause(rank, node)

    if kind == "INPUT_STRAGGLER":
        why = _straggler_why(
            primary,
            meta=meta,
            verb="waits",
            phrase="for input",
        )
        action = (
            "inspect dataloader, collate_fn, preprocessing, and storage on "
            f"{who}."
        )
        return why, action

    if kind == "H2D_STRAGGLER":
        why = _straggler_why(
            primary,
            meta=meta,
            verb="spends",
            phrase="in host-to-device copies",
        )
        target = "the slow rank" if rank is None else f"rank {rank}"
        return why, f"inspect PCIe topology / transfer path on {target}."

    if kind == "COMPUTE_STRAGGLER":
        why = _straggler_why(
            primary,
            meta=meta,
            verb="spends",
            phrase=f"in {_phase_word(primary)}",
        )
        target = "the slow rank" if rank is None else f"rank {rank}"
        return why, (
            f"inspect {target}: thermal/clock throttling, imbalanced work, "
            "or a slow device."
        )

    why = _straggler_why(
        primary,
        meta=meta,
        verb="spends",
        phrase=f"in {_phase_word(primary)}",
    )
    target = "the slow rank" if rank is None else f"rank {rank}"
    return why, (
        f"inspect {target} end-to-end; multiple phases are slow on it."
    )


def _missing_signals(step_time_summary: Mapping[str, Any]) -> str:
    """Return the comma-joined missing Step Time signal names."""
    evidence = _mapping(_diagnosis(step_time_summary).get("evidence"))
    names = [str(name) for name in _sequence(evidence.get("missing_signals"))]
    return ", ".join(names)


def _step_time_skew_percent(
    step_time_summary: Mapping[str, Any],
) -> Optional[str]:
    """Return the worst-vs-median Step Time skew as a percentage string."""
    median = _point_value(step_time_summary, "median", "step_time_ms")
    worst = _point_value(step_time_summary, "worst", "step_time_ms")
    if median is None or worst is None or median <= 0.0:
        return None
    return f"{100.0 * (worst - median) / median:.1f}%"


def _worst_node_rank(
    step_time_summary: Mapping[str, Any],
    primary: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Optional[int]:
    """Return the node rank of the straggler rank, on multi-node runs."""
    if not _is_multi_node(meta):
        return None
    metric = str(_comparison(primary).get("metric") or "step_time_ms")
    idx = _point(step_time_summary, "worst", metric).get("idx")
    return _node_of(step_time_summary, idx)


def _tree_values(
    step_time_summary: Mapping[str, Any],
    *,
    multi: bool,
) -> Dict[str, Optional[float]]:
    """Return the timing values used by the tree for this card variant."""
    values: Dict[str, Optional[float]] = {}
    for _, metric, _depth in _TREE_ROWS:
        if multi:
            values[metric] = _point_value(step_time_summary, "median", metric)
        else:
            values[metric] = _average(step_time_summary, metric)
    return values


def _tree_glyph(depth: int, last: bool) -> str:
    """Return the tree glyph prefix for one row."""
    if depth == 0:
        return ""
    branch = "└─ " if last else "├─ "
    return branch if depth == 1 else f"   {branch}"


def _bar(share: float) -> str:
    """Return a 28-character share bar."""
    filled = int(round(max(0.0, min(1.0, share)) * _BAR_WIDTH))
    return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)


def _worst_cell(
    step_time_summary: Mapping[str, Any],
    metric: str,
) -> Optional[str]:
    """Return the `worst rank` cell text for one tree row, when material."""
    point = _point(step_time_summary, "worst", metric)
    value = _float(point.get("value"))
    if value is None:
        return None
    rank = _rank_of(step_time_summary, point.get("idx"))
    if rank is None:
        return None
    return f"{_fmt_ms(value)} ms (r{rank})"


def _is_material_skew(
    step_time_summary: Mapping[str, Any],
    metric: str,
) -> bool:
    """Return whether worst-vs-median skew is worth a `worst rank` cell."""
    median = _point_value(step_time_summary, "median", metric)
    worst = _point_value(step_time_summary, "worst", metric)
    if median is None or worst is None or median <= 0.0:
        return False
    delta = abs(worst - median)
    if delta < _SKEW_MATERIAL_DELTA_MS:
        return False
    return delta / median >= _SKEW_MATERIAL_RATIO


def _culprit_suffix(primary: Mapping[str, Any], *, kind: str) -> str:
    """Return the culprit-row marker text."""
    if kind in STRAGGLER_KINDS:
        ratio = _float(_comparison(primary).get("ratio"))
        if ratio is not None and ratio >= 2.0:
            return f"{_MARKER}  {ratio:.0f}x"
    return f"{_MARKER}  cause"


def _tree_row_text(
    label: str,
    *,
    ms_text: str,
    share: str,
    bar: Optional[str],
    worst: Optional[str],
    suffix: Optional[str],
    suffix_column: int = 0,
) -> str:
    """Assemble one fixed-column timing tree row."""
    text = f"{label:<{_TREE_LABEL_WIDTH}}{ms_text} ms  {share:>4}"
    if bar is not None:
        text = f"{text}  {bar}"
    if worst is not None:
        text = f"{text:<{_TREE_WORST_COLUMN}}{worst}"
    if suffix:
        # Culprit markers line up in one column so a marker on a leaf row
        # does not land inside the share-bar column of the rows above it.
        text = f"{text:<{suffix_column}}  {suffix}"
    return text


def _append_timing_tree(
    doc: CardDoc,
    *,
    primary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    multi: bool,
) -> bool:
    """Append the timing tree; return whether anything was drawn."""
    values = _tree_values(step_time_summary, multi=multi)
    total = values.get("step_time_ms")
    if total is None or total <= 0.0:
        return False

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
    if culprit_metric is None and kind in STRAGGLER_KINDS:
        culprit_metric = str(
            _comparison(primary).get("metric") or "step_time_ms"
        )

    straggler_culprit = culprit_metric if kind in STRAGGLER_KINDS else None
    worst_cells: Dict[str, str] = {}
    if multi:
        for _label, metric, _depth in present:
            if metric == "step_time_ms":
                continue
            if metric == straggler_culprit or _is_material_skew(
                step_time_summary, metric
            ):
                cell = _worst_cell(step_time_summary, metric)
                if cell:
                    worst_cells[metric] = cell
        if worst_cells:
            cell = _worst_cell(step_time_summary, "step_time_ms")
            if cell:
                worst_cells["step_time_ms"] = cell

    draw_bars = bool(kind in PHASE_SHARE_KINDS and not multi)
    suffix_column = _BAR_COLUMN + _BAR_WIDTH if draw_bars else 0
    severity_style = _severity_style(primary.get("severity"))

    caption = (
        "Where a step goes (median rank, "
        if multi
        else "Where a step goes (average, "
    ) + f"{_clock(step_time_summary)} clock)"
    if worst_cells:
        header_column = _TREE_WORST_COLUMN + 1
        caption = f"{caption:<{header_column}}worst rank"
    doc.text(caption, STYLE_DIM)

    for label, metric, depth in present:
        value = values[metric]
        glyph = _tree_glyph(depth, last_by_depth.get(depth) == metric)
        suffix = (
            _culprit_suffix(primary, kind=kind)
            if metric == culprit_metric and metric != "step_time_ms"
            else None
        )
        text = _tree_row_text(
            f"{glyph}{label}",
            ms_text=_fmt_ms(float(value or 0.0)),
            share=_fmt_share(value, total),
            bar=(
                _bar(float(value or 0.0) / total)
                if draw_bars and depth <= 1
                else None
            ),
            worst=worst_cells.get(metric),
            suffix=suffix,
            suffix_column=suffix_column,
        )
        if suffix:
            doc.add(
                Span(text[: -len(suffix)]),
                Span(suffix, severity_style),
            )
        else:
            doc.text(text)
    return True


def _run_gpu_count(meta: Mapping[str, Any]) -> Optional[int]:
    """
    Return the GPU count a run card should report.

    ``meta.gpus_observed`` counts the GPUs on the host, which overstates a
    run that only used some of them (a one-process run on a four-GPU box).
    The run card describes the run, so it is capped by the world size.
    """
    gpus = _int(meta.get("gpus_observed"))
    world_size = _int(meta.get("world_size"))
    if gpus is not None and gpus > 0 and world_size is not None:
        if world_size > 0:
            return min(gpus, world_size)
    return gpus


def _run_header_meta(
    *,
    meta: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    duration_s: Optional[float],
    multi: bool,
) -> str:
    """Compose the run card's identity and coverage line."""
    gpus = _run_gpu_count(meta)
    nodes = _int(meta.get("nodes_observed"))
    steps = _steps_analyzed(step_time_summary)

    segments: List[Optional[str]] = [
        str(meta.get("run_name")) if meta.get("run_name") else None
    ]
    if gpus == 0:
        segments.append("CPU only (no GPU detected)")
    elif gpus is not None:
        segments.append(_plural(gpus, "GPU"))
    if multi and nodes is not None:
        segments.append(_plural(nodes, "node"))
    if steps is not None:
        segments.append(f"{steps} steps analyzed")
    segments.append(fmt_duration(duration_s))
    return _join_segments(segments)


def _watch_header_meta(
    *,
    meta: Mapping[str, Any],
    duration_s: Optional[float],
    multi: bool,
) -> str:
    """Compose the watch card's identity and coverage line."""
    gpus = _int(meta.get("gpus_observed"))
    nodes = _int(meta.get("nodes_observed"))

    segments: List[Optional[str]] = []
    if multi and nodes is not None:
        segments.append(_plural(nodes, "node"))
    else:
        segments.append("1 machine")
    if gpus == 0:
        segments.append("no GPU detected")
    elif gpus is not None:
        segments.append(_plural(gpus, "GPU"))
    duration = fmt_duration(duration_s)
    if duration is not None:
        segments.append(f"observed for {duration}")
    return _join_segments(segments)


def _append_header(doc: CardDoc, *, title: str, meta_line: str) -> None:
    """Append the card header block."""
    doc.rule()
    doc.text(title, STYLE_BOLD)
    if meta_line:
        # Run names can be long session ids, so the coverage line wraps
        # rather than losing identity or steps-analyzed to a clip.
        doc.wrapped(meta_line, style=STYLE_DIM)
    doc.rule()


def _append_verdict(doc: CardDoc, primary: Mapping[str, Any]) -> None:
    """Append the verdict line."""
    status = _display_status(primary)
    severity = _severity(primary.get("severity"))
    status_style = _severity_style(severity)
    if severity not in {"info", ""}:
        status = f"{status}  ({_severity_label(severity)})"
    doc.add(
        Span("Verdict: ", STYLE_BOLD),
        Span(status, status_style),
    )


def _supporting_line(
    *,
    primary: Mapping[str, Any],
    multi: bool,
) -> Optional[str]:
    """Return the supporting GPU-utilization line, when it applies."""
    kind = str(primary.get("kind") or "")
    if kind not in PHASE_SHARE_KINDS and kind not in STRAGGLER_KINDS:
        return None
    evidence = _mapping(primary.get("evidence"))
    util = _float(evidence.get("gpu_util_avg_percent"))
    if util is None or util >= _SUPPORTING_GPU_UTIL_MAX:
        return None
    value = _fmt_percent(util)
    if multi and kind in STRAGGLER_KINDS:
        return (
            f"Supporting: GPU util median {value}% -- ranks idle at the step "
            "barrier."
        )
    if kind == "INPUT_BOUND":
        return (
            f"Supporting: GPU util {value}% avg -- consistent with input "
            "starvation."
        )
    return f"Supporting: GPU util {value}% avg."


def _peak_memory_line(
    step_memory_summary: Mapping[str, Any],
    *,
    multi: bool,
) -> Optional[str]:
    """Return the one-line peak step-memory summary, when measured."""
    reserved_point = _point(
        step_memory_summary, "worst", "peak_reserved_bytes"
    )
    reserved = _float(reserved_point.get("value"))
    allocated = _point_value(
        step_memory_summary,
        "worst",
        "peak_allocated_bytes",
    )
    if reserved is None and allocated is None:
        return None

    if not multi:
        alloc_text = _fmt_capacity(allocated)
        reserved_text = _fmt_capacity(reserved)
        parts = []
        if alloc_text is not None:
            parts.append(f"{alloc_text} allocated")
        if reserved_text is not None:
            parts.append(f"{reserved_text} reserved")
        body = f" {_DOT} ".join(parts)
        return f"Peak step memory: {body}"

    if reserved is None:
        return None
    reserved_text = _fmt_capacity(reserved)
    median = _point_value(
        step_memory_summary,
        "median",
        "peak_reserved_bytes",
    )
    even = (
        median is not None
        and median > 0.0
        and abs(reserved - median) / median < _MEMORY_SKEW_EVEN_RATIO
    )
    if even:
        scope = "even across ranks"
    else:
        rank = _rank_of(step_memory_summary, reserved_point.get("idx"))
        scope = "worst rank" if rank is None else f"worst rank r{rank}"
    return f"Peak step memory: {reserved_text} reserved {_DOT} {scope}"


def _all_normal_line(
    *,
    primary: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    system_summary: Mapping[str, Any],
    multi: bool,
) -> Optional[str]:
    """Return the collapsed all-healthy line, when every section is info."""
    if str(primary.get("kind") or "") != "NO_CLEAR_PERFORMANCE_BOTTLENECK":
        return None
    for section in sections:
        if _severity(_diagnosis(section).get("severity")) != "info":
            return None
    util = _float(
        _mapping(primary.get("evidence")).get("gpu_util_avg_percent")
    )
    if util is None:
        util = _average(system_summary, "gpu_util_percent")
    scope = " on every rank" if multi else ""
    line = f"System, process, and memory: all normal{scope}."
    value = _fmt_percent(util)
    if value is not None:
        line = f"{line} GPU util {value}% avg."
    return line


def _also_findings(
    *,
    primary: Mapping[str, Any],
    sections: Sequence[Tuple[str, Mapping[str, Any]]],
) -> List[Tuple[str, str]]:
    """
    Return up to two secondary warn/crit findings as (summary, severity).

    Only resource sections are eligible. A secondary Step Time finding may
    well be a real cause of slow steps, so listing it under "not the cause of
    slow steps" would be untrue; those stay in the JSON.
    """
    primary_kind = str(primary.get("kind") or "")
    primary_section = str(primary.get("section") or "")
    found: List[Tuple[int, str, str]] = []
    for name, section in sections:
        for raw_issue in _sequence(section.get("issues")):
            issue = _mapping(raw_issue)
            severity = _severity(issue.get("severity"))
            if severity not in _SEVERITY_RANK:
                continue
            kind = str(issue.get("kind") or "")
            if kind == primary_kind and name == primary_section:
                continue
            summary = str(issue.get("summary") or "").strip()
            if not summary:
                continue
            found.append((_SEVERITY_RANK[severity], summary, severity))
    found.sort(key=lambda item: item[0])
    return [(summary, severity) for _rank, summary, severity in found[:2]]


def _observed_anyway_line(
    system_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Optional[str]:
    """Return the degraded-state observation line, when anything was seen."""
    segments: List[Optional[str]] = []
    if _int(meta.get("gpus_observed")) == 0:
        segments.append("no GPU detected")
    else:
        gpu = _fmt_percent(_average(system_summary, "gpu_util_percent"))
        if gpu is not None:
            segments.append(f"GPU util {gpu}% avg")
    cpu = _fmt_percent(_average(system_summary, "cpu_percent"))
    if cpu is not None:
        segments.append(f"CPU util {cpu}% avg")
    body = _join_segments(segments)
    return f"Observed anyway: {body}" if body else None


def _footer_line(
    artifact_hint: Optional[str],
    html_hint: Optional[str],
) -> str:
    """Return the artifact footer line."""
    path = artifact_hint or "final_summary.json"
    if html_hint:
        return f"Full evidence: {path} {_DOT} {html_hint}"
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
    """Append the run card body between the header and the footer."""
    header_meta = dict(meta)
    header_meta["node_rank_of_worst"] = _worst_node_rank(
        step_time_summary,
        primary,
        meta,
    )
    why, action = _why_next(
        primary=primary,
        step_time_summary=step_time_summary,
        meta=header_meta,
        multi=multi,
    )

    degraded = str(primary.get("kind") or "") == "INSUFFICIENT_STEP_TIME_DATA"

    doc.blank()
    _append_verdict(doc, primary)
    if why:
        doc.wrapped(why, label="Why: ")
    doc.blank()

    if not degraded and _append_timing_tree(
        doc,
        primary=primary,
        step_time_summary=step_time_summary,
        multi=multi,
    ):
        doc.blank()

    cpu_only = _int(meta.get("gpus_observed")) == 0
    all_normal = _all_normal_line(
        primary=primary,
        sections=(
            system_summary,
            process_summary,
            step_time_summary,
            step_memory_summary,
        ),
        system_summary=system_summary,
        multi=multi,
    )
    peak = _peak_memory_line(step_memory_summary, multi=multi)

    before_next: List[str] = []
    after_next: List[Tuple[str, str]] = []
    also = _also_findings(
        primary=primary,
        sections=(
            ("system", system_summary),
            ("process", process_summary),
            ("step_memory", step_memory_summary),
        ),
    )

    if degraded:
        observed = _observed_anyway_line(system_summary, meta)
        if observed:
            before_next.append(observed)
    elif all_normal is not None:
        before_next.append(all_normal)
        if peak is not None:
            before_next.append(peak)
    elif cpu_only:
        before_next.append("H2D and step memory not measured (no GPU).")
        fetch = _average(step_time_summary, "dataloader_fetch_cpu_ms")
        if fetch is not None:
            before_next.append(
                f"DataLoader fetch: {fetch:.1f} ms (supplemental)."
            )
    else:
        supporting = _supporting_line(primary=primary, multi=multi)
        if supporting:
            after_next.append((supporting, STYLE_DIM))
        if also:
            after_next.append(
                ("Also, not the cause of slow steps:", STYLE_DIM)
            )
            for summary, severity in also:
                after_next.append(
                    (
                        f"! {summary}  ({_severity_label(severity)})",
                        severity,
                    )
                )
        if peak is not None:
            # Keep the measured memory evidence beside the timing evidence,
            # before the recommended action.  This also keeps the ordering
            # consistent with the all-normal path above.
            before_next.append(peak)

    for line in before_next:
        doc.text(line, STYLE_DIM)
    if before_next:
        doc.blank()

    if action:
        doc.wrapped(action, label="Next: ", style=STYLE_NEXT)

    if after_next:
        doc.blank()
        for text, style in after_next:
            if style in _SEVERITY_RANK:
                doc.wrapped_with_severity(
                    text,
                    severity_label=_severity_label(style),
                    severity_style=_severity_style(style),
                )
            else:
                doc.wrapped(text, style=style)


def _watch_health_lines(
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Return the host-health lines and the finding action, when present."""
    findings = []
    for section in (system_summary, process_summary):
        diagnosis = _diagnosis(section)
        severity = _severity(diagnosis.get("severity"))
        if severity in _SEVERITY_RANK:
            findings.append((_SEVERITY_RANK[severity], diagnosis, severity))
    findings.sort(key=lambda item: item[0])

    if not findings:
        nodes = _int(meta.get("nodes_observed"))
        scope = f" on all {nodes} nodes" if multi and nodes else ""
        return [(f"Host health: NORMAL{scope}", STYLE_OK)], None

    _rank, diagnosis, severity = findings[0]
    status = _status_text(diagnosis.get("status"))
    label = _severity_label(severity)
    lines = [(f"Host health: {status}  ({label})", _severity_style(severity))]
    summary = str(diagnosis.get("summary") or "").strip()
    if summary:
        lines.append((summary, STYLE_PLAIN))
    action = str(diagnosis.get("action") or "").strip()
    return lines, (action or None)


def _watch_total_gb(
    used_bytes: Optional[float],
    percent: Optional[float],
) -> Optional[float]:
    """Derive a capacity total from a used-bytes and percent pair."""
    if used_bytes is None or percent is None or percent <= 0.0:
        return None
    return float(used_bytes) / (float(percent) / 100.0)


def _watch_capacity_cell(
    label: str,
    used_bytes: Optional[float],
    percent: Optional[float],
) -> Optional[str]:
    """Return a `RAM  6.2 / 32.0 GB  (19%)` style right-column cell."""
    used = _fmt_gb(used_bytes)
    if used is None:
        return None
    total = _fmt_gb(_watch_total_gb(used_bytes, percent))
    head = f"{label:<{_WATCH_RIGHT_LABEL_WIDTH}}{used:>4}"
    if total is None:
        return f"{head} GB used"
    return f"{head} / {total:>4} GB  ({_fmt_percent(percent)}%)"


def _watch_single_metric_rows(
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    show_gpu: bool,
) -> List[str]:
    """Return the two-column watch metric block for one machine."""
    left: List[str] = []
    cpu = _fmt_percent(_average(system_summary, "cpu_percent"))
    if cpu is not None:
        left.append(f"{'CPU Util':<{_WATCH_LEFT_LABEL_WIDTH}}{cpu}% avg")
    if show_gpu:
        util = _fmt_percent(_average(system_summary, "gpu_util_percent"))
        if util is not None:
            left.append(f"{'GPU Util':<{_WATCH_LEFT_LABEL_WIDTH}}{util}% avg")
        temp = _point_value(system_summary, "worst", "gpu_temp_c")
        if temp is None:
            temp = _average(system_summary, "gpu_temp_c")
        temp_text = _fmt_percent(temp)
        if temp_text is not None:
            left.append(
                f"{'GPU Temp':<{_WATCH_LEFT_LABEL_WIDTH}}{temp_text}C max"
            )

    right: List[str] = []
    ram = _watch_capacity_cell(
        "RAM",
        _average(system_summary, "ram_bytes"),
        _average(system_summary, "ram_percent"),
    )
    if ram is not None:
        right.append(ram)
    if show_gpu:
        gpu_mem = _watch_capacity_cell(
            "GPU Mem",
            _average(system_summary, "gpu_mem_bytes"),
            _average(system_summary, "gpu_mem_percent"),
        )
        if gpu_mem is not None:
            right.append(gpu_mem)
    rss = _fmt_gb(_point_value(process_summary, "worst", "ram_bytes"))
    if rss is None:
        rss = _fmt_gb(_average(process_summary, "ram_bytes"))
    if rss is not None:
        right.append(f"{'Proc RSS':<{_WATCH_RIGHT_LABEL_WIDTH}}{rss:>4} GB")

    rows: List[str] = []
    for index in range(max(len(left), len(right))):
        head = left[index] if index < len(left) else ""
        tail = right[index] if index < len(right) else ""
        if tail:
            rows.append(f"{head:<{_WATCH_RIGHT_COLUMN}}{tail}")
        else:
            rows.append(head)
    return rows


_WATCH_MULTI_METRICS = (
    ("CPU Util", "cpu_percent", "%", "avg"),
    ("RAM", "ram_percent", "%", "used"),
    ("GPU Util", "gpu_util_percent", "%", "avg"),
    ("GPU Mem", "gpu_mem_percent", "%", "used"),
    ("GPU Temp", "gpu_temp_c", "C", "max"),
)


def _watch_multi_metric_rows(
    system_summary: Mapping[str, Any],
    *,
    show_gpu: bool,
) -> List[str]:
    """Return the median / worst-node watch metric table."""
    rows: List[str] = []
    for label, metric, unit, qualifier in _WATCH_MULTI_METRICS:
        if not show_gpu and metric.startswith("gpu_"):
            continue
        median = _fmt_percent(_point_value(system_summary, "median", metric))
        if median is None:
            continue
        cell = f"{label:<{_WATCH_MULTI_LABEL_WIDTH}}{median}{unit} {qualifier}"
        worst_point = _point(system_summary, "worst", metric)
        worst = _fmt_percent(_float(worst_point.get("value")))
        node = _node_of(system_summary, worst_point.get("idx"))
        if worst is not None and node is not None:
            cell = (
                f"{cell:<{_WATCH_MULTI_WORST_COLUMN}}{worst}{unit}  (n{node})"
            )
        rows.append(cell)
    if rows:
        header = (
            f"{'':<{_WATCH_MULTI_LABEL_WIDTH}}median"
            f"{'':<{_WATCH_MULTI_WORST_COLUMN - _WATCH_MULTI_LABEL_WIDTH - 6}}"
            "worst node"
        )
        rows.insert(0, header)
    return rows


def _append_watch_body(
    doc: CardDoc,
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    multi: bool,
) -> None:
    """Append the watch card body between the header and the footer."""
    show_gpu = _int(meta.get("gpus_observed")) != 0

    doc.blank()
    health_lines, health_action = _watch_health_lines(
        system_summary=system_summary,
        process_summary=process_summary,
        meta=meta,
        multi=multi,
    )
    for index, (text, style) in enumerate(health_lines):
        if index == 0:
            doc.add(Span("Host health: ", STYLE_BOLD), Span(text[13:], style))
        else:
            doc.wrapped(text, style=style)
    doc.blank()

    if multi:
        rows = _watch_multi_metric_rows(system_summary, show_gpu=show_gpu)
    else:
        rows = _watch_single_metric_rows(
            system_summary=system_summary,
            process_summary=process_summary,
            show_gpu=show_gpu,
        )
    for text in rows:
        doc.text(text)
    if rows:
        doc.blank()

    system_kind = str(_diagnosis(system_summary).get("kind") or "")
    printed_next = False
    if show_gpu and system_kind in LOW_GPU_UTIL_KINDS:
        util = _fmt_percent(_average(system_summary, "gpu_util_percent"))
        doc.wrapped(
            f"GPU utilization stayed low ({util}% avg). Watch cannot tell "
            "whether that is input, transfer, sync, or idle time.",
            label="Observation: ",
        )
        doc.wrapped(_WATCH_RUN_ACTION, label="Next: ", style=STYLE_NEXT)
        printed_next = True
    elif health_action:
        doc.wrapped(health_action, label="Next: ", style=STYLE_NEXT)
        doc.blank()

    if not printed_next:
        doc.wrapped(_WATCH_BOUNDARY, style=STYLE_DIM)


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
    """
    Build the end-of-run terminal card for one already-built summary payload.

    The card is presentation-only: values come from the section payloads and
    the promoted ``primary_diagnosis`` exactly as they were computed.
    """
    primary = _mapping(primary_diagnosis)
    system = _mapping(system_summary)
    process = _mapping(process_summary)
    step_time = _mapping(step_time_summary)
    step_memory = _mapping(step_memory_summary)
    run_meta = _mapping(meta)

    doc = CardDoc()
    watch = str(profile or RUN_PROFILE).strip().lower() == WATCH_PROFILE
    if watch:
        multi = _is_multi_node(run_meta)
        _append_header(
            doc,
            title=WATCH_TITLE,
            meta_line=_watch_header_meta(
                meta=run_meta,
                duration_s=duration_s,
                multi=multi,
            ),
        )
        _append_watch_body(
            doc,
            system_summary=system,
            process_summary=process,
            meta=run_meta,
            multi=multi,
        )
    else:
        multi = is_multi_process(step_time)
        _append_header(
            doc,
            title=RUN_TITLE,
            meta_line=_run_header_meta(
                meta=run_meta,
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
    """
    Build the minimal verdict-and-footer card used when rendering fails.

    End-of-run reporting is best-effort: a card that cannot be composed must
    still print something truthful instead of breaking shutdown.
    """
    primary = _mapping(primary_diagnosis)
    watch = str(profile or RUN_PROFILE).strip().lower() == WATCH_PROFILE
    doc = CardDoc()
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
    """
    Return whether stdout can carry severity-scoped ANSI color.

    Shared by every print path so the CLI and the SDK make the same call.
    """
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
    session_root: Optional[str],
    *,
    write_html: bool = False,
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
    """
    Infer the profile a stored card was rendered with.

    The title is written by this module, so the header line is a reliable
    marker: only watch cards carry the watch title.
    """
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
    """
    Return an ANSI rendering of a stored card, or the stored text unchanged.

    The colored output is only used when the card rebuilt from ``payload``
    renders back to exactly the stored text, so colorizing can never change
    what a reader sees beyond the escape codes. Anything unexpected -- a
    non-terminal stdout, a payload that no longer reproduces the stored card,
    or any exception -- falls back to the stored text.
    """
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
