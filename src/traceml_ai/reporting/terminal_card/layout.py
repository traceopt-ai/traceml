# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Terminal-card primitives and fixed-pane layout.

This module is deliberately payload-agnostic.  It owns the styled document
model, plain/ANSI rendering, and the shared Run-card pane compositor.  Section
modules provide ``CardDoc`` content; this module makes that content fit the
same fixed geometry in every Run topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from traceml_ai.reporting.summaries.summary_layout import (
    border,
    row,
    wrap_lines,
)

CARD_WIDTH = 78
RUN_CARD_WIDTH = CARD_WIDTH * 2
INNER_WIDTH = CARD_WIDTH - 4
RUN_INNER_WIDTH = RUN_CARD_WIDTH - 4

RUN_LEFT_PANE_WIDTH = 61
RUN_RIGHT_PANE_WIDTH = RUN_INNER_WIDTH - RUN_LEFT_PANE_WIDTH - 4
RUN_PANE_SEPARATOR = "||  "

# Terminal presentation contract
# -----------------------------
# Chrome is dim, labels/actions are bold in the terminal default foreground,
# measurements/evidence are neutral, and only statuses/severity markers carry
# colour.  Plain text is the stored artifact, so ANSI never adds information.
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
    """A rendering-independent ordered collection of card lines."""

    width: int = CARD_WIDTH
    lines: List[CardLine] = field(default_factory=list)

    @property
    def inner_width(self) -> int:
        """Return the visible content width between the card margins."""
        return max(0, self.width - 4)

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
        for index, line in enumerate(wrap_lines(full, self.inner_width)):
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
        for line in wrap_lines(value, self.inner_width):
            before, marker, after = line.rpartition(severity_label)
            if marker:
                self.add(
                    Span(before),
                    Span(marker, severity_style),
                    Span(after),
                )
            else:
                self.text(line)

    def wrapped_spans(self, *spans: Span) -> None:
        """Append wrapped styled spans without changing visible text."""
        full = "".join(span.text for span in spans)
        if not full:
            self.blank()
            return

        ranges: List[Tuple[int, int, Span]] = []
        offset = 0
        for span in spans:
            end = offset + len(span.text)
            ranges.append((offset, end, span))
            offset = end

        cursor = 0
        for wrapped in wrap_lines(full, self.inner_width):
            start = full.find(wrapped, cursor)
            if start < 0:
                self.text(wrapped)
                cursor += len(wrapped)
                continue
            end = start + len(wrapped)
            line_spans: List[Span] = []
            for span_start, span_end, span in ranges:
                left = max(start, span_start)
                right = min(end, span_end)
                if left < right:
                    line_spans.append(Span(full[left:right], span.style))
            self.add(*line_spans)
            cursor = end


@dataclass
class PaneStages:
    """Heading, evidence, spacer, and table documents for one resource pane."""

    heading: CardDoc
    evidence: CardDoc
    spacer: CardDoc
    table: CardDoc


def new_pane_stages(width: int) -> PaneStages:
    """Create the four independent line groups used by a lower Run pane."""
    pane_width = width + 4
    return PaneStages(
        heading=CardDoc(width=pane_width),
        evidence=CardDoc(width=pane_width),
        spacer=CardDoc(width=pane_width),
        table=CardDoc(width=pane_width),
    )


def _visible_spans(line: CardLine, inner_width: int) -> List[Span]:
    """Return the spans of one row clipped to the card's inner width."""
    out: List[Span] = []
    used = 0
    for span in line.spans:
        if used >= inner_width:
            break
        room = inner_width - used
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
            lines.append(border(width=doc.width))
            continue
        text = "".join(
            span.text for span in _visible_spans(line, doc.inner_width)
        )
        lines.append(row(text, width=doc.width))
    return "\n".join(lines)


def _paint(text: str, style: str) -> str:
    """Wrap one span in its ANSI code, if the style has one."""
    code = _ANSI_CODES.get(style)
    if not code or not text:
        return text
    return f"{code}{text}{_ANSI_RESET}"


def card_to_ansi(doc: CardDoc) -> str:
    """Render a card with severity-scoped ANSI and plain-text geometry."""
    lines: List[str] = []
    edge = _paint("|", STYLE_BORDER)
    for line in doc.lines:
        if line.rule:
            lines.append(_paint(border(width=doc.width), STYLE_BORDER))
            continue
        spans = _visible_spans(line, doc.inner_width)
        visible = sum(len(span.text) for span in spans)
        body = "".join(_paint(span.text, span.style) for span in spans)
        padding = " " * max(0, doc.inner_width - visible)
        lines.append(f"{edge}  {body}{padding}{edge}")
    return "\n".join(lines)


def append_table_header(
    doc: CardDoc,
    *,
    multi: bool,
    median_label: str,
    worst_label: str,
    label_width: int = 28,
    median_width: int = 20,
) -> None:
    """Append the shared Run resource-table header."""
    if multi:
        doc.text(
            f"{'':<{label_width}}"
            f"{median_label:<{median_width}}"
            f"{worst_label}",
            STYLE_DIM,
        )
    else:
        doc.text(f"{'':<{label_width}}avg", STYLE_DIM)


def append_table_row(
    doc: CardDoc,
    *,
    label: str,
    average: str | None = None,
    median: str | None = None,
    worst: str | None = None,
    worst_scope: str | None = None,
    multi: bool,
    label_width: int = 28,
    median_width: int = 20,
) -> None:
    """Append one Run resource row while preserving measured zero.

    A long worst-rank/node scope is continued below its values instead of
    leaking across, or silently clipping at, the fixed pane boundary.
    """
    if multi:
        worst_cell = (
            f"{worst}, {worst_scope}"
            if worst and worst_scope
            else (worst or "")
        )
        if median is None and not worst_cell:
            return
        body = (
            f"{label:<{label_width}}"
            f"{(median or ''):<{median_width}}"
            f"{worst_cell}"
        )
        if len(body) <= doc.inner_width:
            doc.text(body)
            return
        values = (
            f"{label:<{label_width}}"
            f"{(median or ''):<{median_width}}"
            f"{worst or ''}"
        )
        if worst_scope and len(values) <= doc.inner_width:
            doc.text(values)
            doc.text(f"{'':<{label_width + median_width}}({worst_scope})")
        else:
            doc.wrapped(body)
        return
    if average is not None:
        doc.text(f"{label:<{label_width}}{average}")


def _split_pane_line(line: CardLine, width: int) -> List[CardLine]:
    """Wrap one pane line without clipping text or crossing the divider."""
    if line.rule:
        return [line]
    wrapped = CardDoc(width=width + 4)
    wrapped.wrapped_spans(*line.spans)
    output: List[CardLine] = []
    for wrapped_line in wrapped.lines:
        pending = list(wrapped_line.spans)
        if not pending:
            output.append(CardLine())
            continue
        while pending:
            visible = sum(len(span.text) for span in pending)
            if visible <= width:
                output.append(CardLine(spans=tuple(pending)))
                break
            chunk = _visible_spans(CardLine(spans=tuple(pending)), width)
            output.append(CardLine(spans=tuple(chunk)))
            consumed = sum(len(span.text) for span in chunk)
            remainder: List[Span] = []
            remaining = consumed
            for span in pending:
                if remaining >= len(span.text):
                    remaining -= len(span.text)
                    continue
                remainder.append(Span(span.text[remaining:], span.style))
                remaining = 0
            pending = remainder
    return output or [CardLine()]


def _padded_pane_spans(line: CardLine, width: int) -> List[Span]:
    """Return one already-wrapped pane row padded to its fixed width."""
    spans = list(line.spans)
    visible = sum(len(span.text) for span in spans)
    if visible < width:
        spans.append(Span(" " * (width - visible)))
    return spans


def append_parallel_panes(doc: CardDoc, left: CardDoc, right: CardDoc) -> None:
    """Append two Run panes with a fixed dim divider on every physical row."""
    left_lines = [
        output
        for line in left.lines
        for output in _split_pane_line(line, RUN_LEFT_PANE_WIDTH)
    ]
    right_lines = [
        output
        for line in right.lines
        for output in _split_pane_line(line, RUN_RIGHT_PANE_WIDTH)
    ]
    for index in range(max(len(left_lines), len(right_lines))):
        left_line = (
            left_lines[index] if index < len(left_lines) else CardLine()
        )
        right_line = (
            right_lines[index] if index < len(right_lines) else CardLine()
        )
        doc.add(
            *_padded_pane_spans(left_line, RUN_LEFT_PANE_WIDTH),
            Span(RUN_PANE_SEPARATOR, STYLE_DIM),
            *_padded_pane_spans(right_line, RUN_RIGHT_PANE_WIDTH),
        )


def append_staged_panes(
    doc: CardDoc, left: PaneStages, right: PaneStages
) -> None:
    """Compose aligned heading, evidence, spacer, and table pane groups."""
    append_parallel_panes(doc, left.heading, right.heading)
    append_parallel_panes(doc, left.evidence, right.evidence)
    append_parallel_panes(doc, left.spacer, right.spacer)
    append_parallel_panes(doc, left.table, right.table)


__all__ = [
    "CARD_WIDTH",
    "INNER_WIDTH",
    "RUN_CARD_WIDTH",
    "RUN_INNER_WIDTH",
    "RUN_LEFT_PANE_WIDTH",
    "RUN_RIGHT_PANE_WIDTH",
    "STYLE_BOLD",
    "STYLE_BORDER",
    "STYLE_CRIT",
    "STYLE_DIM",
    "STYLE_NEXT",
    "STYLE_OK",
    "STYLE_PLAIN",
    "STYLE_WARN",
    "CardDoc",
    "CardLine",
    "PaneStages",
    "Span",
    "append_parallel_panes",
    "append_staged_panes",
    "append_table_header",
    "append_table_row",
    "card_to_ansi",
    "card_to_plain",
    "new_pane_stages",
]
