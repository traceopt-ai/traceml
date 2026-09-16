# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload readers and presentation-only helpers for terminal-card sections.

All functions in this module read already-computed final-summary payloads.
They intentionally do not open databases, calculate diagnostics, or introduce
new aggregation.  Section modules use them to select and format stored values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from traceml_ai.reporting.summaries.summary_formatting import bytes_to_gb
from traceml_ai.reporting.terminal_card.layout import (
    STYLE_BOLD,
    STYLE_CRIT,
    STYLE_OK,
    STYLE_WARN,
    Span,
)

DOT = "·"

SEVERITY_LABELS = {
    "crit": "CRITICAL",
    "critical": "CRITICAL",
    "warn": "WARNING",
    "warning": "WARNING",
    "info": "INFO",
}
SEVERITY_STYLES = {
    "crit": STYLE_CRIT,
    "critical": STYLE_CRIT,
    "warn": STYLE_WARN,
    "warning": STYLE_WARN,
}
SEVERITY_RANK = {"crit": 0, "critical": 0, "warn": 1, "warning": 1}


@dataclass(frozen=True)
class RankCoverage:
    """Ranks represented by a section and the optional expected world size."""

    observed: int
    expected: Optional[int]

    @property
    def distributed(self) -> bool:
        """Return whether observed or expected coverage spans multiple ranks."""
        return self.observed > 1 or bool(self.expected and self.expected > 1)

    def header_text(self) -> Optional[str]:
        """Format truthful header coverage without inventing observations."""
        if self.expected is not None:
            if self.observed == self.expected == 1:
                return "1 rank"
            unit = "rank" if self.expected == 1 else "ranks"
            return f"{self.observed}/{self.expected} {unit}"
        if self.observed <= 0:
            return None
        unit = "rank" if self.observed == 1 else "ranks"
        return f"{self.observed} {unit}"

    def detail_text(self) -> Optional[str]:
        """Format section coverage, omitting an uninformative complete 1/1."""
        if self.expected is not None:
            if self.observed == self.expected == 1:
                return None
            unit = "rank" if self.expected == 1 else "ranks"
            return f"{self.observed}/{self.expected} {unit}"
        if self.observed <= 0:
            return None
        unit = "rank" if self.observed == 1 else "ranks"
        return f"{self.observed} {unit} observed"


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping, or an empty mapping for malformed payload blocks."""
    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> Sequence[Any]:
    """Return a non-string sequence, or an empty tuple for malformed data."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def as_float(value: Any) -> Optional[float]:
    """Return a float only for numeric, non-boolean payload values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    """Return an integer only for numeric, non-boolean payload values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def section_block(section: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one ``global`` rollup block from a section payload."""
    return as_mapping(as_mapping(section.get("global")).get(name))


def analysis_window(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's stored analysis-window block."""
    return section_block(section, "window")


def metadata(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's metadata block."""
    return as_mapping(section.get("metadata"))


def average(section: Mapping[str, Any], metric: str) -> Optional[float]:
    """Return one stored average metric value."""
    return as_float(section_block(section, "average").get(metric))


def point(
    section: Mapping[str, Any], block: str, metric: str
) -> Mapping[str, Any]:
    """Return one stored median/worst metric point."""
    return as_mapping(section_block(section, block).get(metric))


def point_value(
    section: Mapping[str, Any], block: str, metric: str
) -> Optional[float]:
    """Return the numeric value from one stored median/worst point."""
    return as_float(point(section, block, metric).get("value"))


def group_rows(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the grouped summary rows for a section."""
    return as_mapping(as_mapping(section.get("groups")).get("rows"))


def rank_count(section: Mapping[str, Any]) -> int:
    """Return ranks represented by metadata or grouped section rows."""
    used = as_int(metadata(section).get("global_ranks_used"))
    if used is not None:
        return max(0, used)

    ranks = set()
    anonymous_rows = 0
    for raw_row in group_rows(section).values():
        row_identity = as_mapping(as_mapping(raw_row).get("identity"))
        rank = as_int(row_identity.get("global_rank"))
        if rank is None:
            anonymous_rows += 1
        else:
            ranks.add(rank)
    return len(ranks) + anonymous_rows


def rank_coverage(
    section: Mapping[str, Any], *, meta: Mapping[str, Any]
) -> RankCoverage:
    """Return represented section ranks and expected run topology."""
    expected = as_int(meta.get("world_size"))
    if expected is not None and expected <= 0:
        expected = None
    return RankCoverage(observed=rank_count(section), expected=expected)


def group_row(section: Mapping[str, Any], idx: Any) -> Mapping[str, Any]:
    """Return one grouped row addressed by a stored global point index."""
    if idx is None:
        return {}
    return as_mapping(group_rows(section).get(str(idx)))


def identity(section: Mapping[str, Any], idx: Any) -> Mapping[str, Any]:
    """Return the grouped-row identity behind a stored point index."""
    return as_mapping(group_row(section, idx).get("identity"))


def identity_for_rank(
    section: Mapping[str, Any], rank: Optional[int]
) -> Mapping[str, Any]:
    """Resolve a stored global rank to a grouped-row identity."""
    if rank is None:
        return {}
    direct = identity(section, rank)
    if as_int(direct.get("global_rank")) == rank:
        return direct
    for raw_row in group_rows(section).values():
        candidate = as_mapping(as_mapping(raw_row).get("identity"))
        if as_int(candidate.get("global_rank")) == rank:
            return candidate
    return direct


def format_scope(
    *,
    rank: Optional[int] = None,
    node: Optional[int] = None,
    gpu: Optional[int] = None,
) -> Optional[str]:
    """Format an already-resolved terminal-card identity compactly.

    ``R``, ``N``, and ``G`` denote global rank, node rank, and GPU index.
    Callers must resolve those values from the stored final-summary payload
    before calling this helper; it deliberately performs no identity lookup or
    topology inference. Missing identity parts are omitted rather than guessed.
    """
    parts = []
    if rank is not None:
        parts.append(f"R{rank}")
    if node is not None:
        parts.append(f"N{node}")
    if gpu is not None:
        parts.append(f"G{gpu}")
    return "/".join(parts) or None


def node_of(section: Mapping[str, Any], idx: Any) -> Optional[int]:
    """Return the stored node rank behind a System point index."""
    node = as_int(identity(section, idx).get("node_rank"))
    return node if node is not None else as_int(idx)


def diagnosis(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's stored diagnosis block."""
    return as_mapping(section.get("diagnosis"))


def severity(value: Any) -> str:
    """Return a normalized severity key."""
    return str(value or "info").strip().lower()


def severity_label(value: Any) -> str:
    """Return the uppercase display label for a severity."""
    key = severity(value)
    return SEVERITY_LABELS.get(key, key.upper() or "INFO")


def severity_style(value: Any) -> str:
    """Return the terminal style for a severity."""
    return SEVERITY_STYLES.get(severity(value), STYLE_OK)


def status_text(value: Any) -> str:
    """Return an uppercase status label without interpreting the diagnosis."""
    text = str(value or "NO DATA").replace("_", " ").strip()
    return " ".join(text.upper().split()) or "NO DATA"


def status_spans(
    label: str,
    section_diagnosis: Mapping[str, Any],
    details: Optional[str] = None,
    *,
    details_style: str = "plain",
) -> Tuple[Span, ...]:
    """Return a bold label, severity-styled status, and optional details."""
    level = severity(section_diagnosis.get("severity"))
    status = status_text(section_diagnosis.get("status"))
    if level not in {"info", ""}:
        status = f"{status}  ({severity_label(level)})"
    spans = [
        Span(f"{label}: ", STYLE_BOLD),
        Span(status, severity_style(level)),
    ]
    if details:
        spans.append(Span(f" {DOT} {details}", details_style))
    return tuple(spans)


def format_duration(duration_s: Optional[float]) -> Optional[str]:
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


def format_ms(value: float) -> str:
    """Format one millisecond value for the fixed timing column."""
    return f"{value:>6.1f}"


def format_share(value: Optional[float], total: Optional[float]) -> str:
    """Format a phase share as a four-character right-aligned percentage."""
    if value is None or total is None or total <= 0.0:
        return ""
    percent = 100.0 * value / total
    text = f"{percent:.0f}%"
    if text == "0%" and value > 0.0:
        text = "<1%"
    return f"{text:>4}"


def format_percent(value: Optional[float]) -> Optional[str]:
    """Format a percentage without decimal places."""
    return None if value is None else f"{value:.0f}"


def format_gb(value: Optional[float]) -> Optional[str]:
    """Format a byte count as decimal gigabytes with one decimal."""
    gb = bytes_to_gb(value)
    return None if gb is None else f"{gb:.1f}"


def format_capacity(value: Optional[float]) -> Optional[str]:
    """Format a byte count in GB, or MB when it would round to zero GB."""
    gb = bytes_to_gb(value)
    if gb is None:
        return None
    if gb < 0.1:
        return f"{float(value or 0.0) / 1e6:.1f} MB"
    return f"{gb:.1f} GB"


def format_memory_value(
    metrics: Mapping[str, Any], *, bytes_metric: str, percent_metric: str
) -> Optional[str]:
    """Format stored memory bytes/percent without deriving a capacity total."""
    memory = format_capacity(as_float(metrics.get(bytes_metric)))
    percent = format_percent(as_float(metrics.get(percent_metric)))
    if memory is None and percent is None:
        return None
    if memory is None:
        return f"{percent}%"
    return f"{memory} ({percent}%)" if percent is not None else memory


def join_segments(segments: Sequence[Optional[str]]) -> str:
    """Join header/observation segments, dropping missing values."""
    return f" {DOT} ".join(text for text in segments if text)


def pack_segments(segments: Sequence[Optional[str]], *, width: int) -> str:
    """Pack complete metadata segments onto fixed-width logical lines."""
    lines = []
    current = ""
    for segment in (text for text in segments if text):
        candidate = f"{current} {DOT} {segment}" if current else segment
        if current and len(candidate) > width:
            lines.append(current)
            current = segment
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def plural(count: int, singular: str) -> str:
    """Return `1 GPU` / `2 GPUs` style text."""
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def is_multi_process(step_time_summary: Mapping[str, Any]) -> bool:
    """Return whether Step Time stored more than one global rank."""
    used = as_int(metadata(step_time_summary).get("global_ranks_used"))
    return bool(used is not None and used > 1)


def resolve_multi_process(
    meta: Mapping[str, Any], sections: Sequence[Mapping[str, Any]]
) -> bool:
    """Resolve topology from world size, then stored section rank evidence."""
    world_size = as_int(meta.get("world_size"))
    if world_size is not None:
        return world_size > 1
    for section in sections:
        section_metadata = metadata(section)
        for name in ("global_ranks_seen", "global_ranks_used"):
            count = as_int(section_metadata.get(name))
            if count is not None and count > 1:
                return True
        ranks = {
            as_int(
                as_mapping(as_mapping(row).get("identity")).get("global_rank")
            )
            for row in group_rows(section).values()
        }
        ranks.discard(None)
        if len(ranks) > 1:
            return True
    return False


def clock_label(step_time_summary: Mapping[str, Any]) -> str:
    """Return the stored diagnosis clock label (`GPU` or `CPU`)."""
    clock = str(
        analysis_window(step_time_summary).get("diagnosis_clock") or ""
    )
    return "CPU" if clock.strip().lower() == "cpu" else "GPU"


def steps_analyzed(step_time_summary: Mapping[str, Any]) -> Optional[int]:
    """Return the number of analyzed steps in the Step Time window."""
    return as_int(analysis_window(step_time_summary).get("steps_analyzed"))


__all__ = [
    "DOT",
    "RankCoverage",
    "SEVERITY_RANK",
    "analysis_window",
    "as_float",
    "as_int",
    "as_mapping",
    "as_sequence",
    "average",
    "clock_label",
    "diagnosis",
    "format_capacity",
    "format_duration",
    "format_gb",
    "format_memory_value",
    "format_ms",
    "format_percent",
    "format_share",
    "format_scope",
    "group_row",
    "group_rows",
    "identity",
    "identity_for_rank",
    "is_multi_process",
    "join_segments",
    "metadata",
    "node_of",
    "pack_segments",
    "plural",
    "point",
    "point_value",
    "rank_count",
    "rank_coverage",
    "resolve_multi_process",
    "section_block",
    "severity",
    "severity_label",
    "severity_style",
    "status_spans",
    "status_text",
    "steps_analyzed",
]
