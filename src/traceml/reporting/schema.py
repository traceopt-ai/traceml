# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared JSON schema helpers for final-report sections.

The reporting sections collect very different signals, but the summary JSON
should still have a predictable outer shape. These small dataclasses centralize
the common field names without hiding section-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

JsonDict = dict[str, Any]


def _copy_mapping(value: Optional[Mapping[str, Any]]) -> Optional[JsonDict]:
    """Return a plain dict copy while preserving explicit None."""
    return dict(value) if value is not None else None


def _validate_metric_sets(
    *,
    metadata: Mapping[str, Any],
    global_summary: Mapping[str, Any],
) -> None:
    """Ensure public metric blocks match `section_metric_names` exactly."""
    metric_names = metadata.get("section_metric_names")
    if metric_names is None:
        return

    expected = {str(metric) for metric in metric_names}
    for block_name in ("average", "median", "worst"):
        block = global_summary.get(block_name)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            raise ValueError(
                f"global.{block_name} must be a mapping when "
                "section_metric_names is set"
            )
        actual = {str(metric) for metric in block.keys()}
        if actual != expected:
            raise ValueError(
                f"global.{block_name} keys must match "
                "metadata.section_metric_names"
            )


@dataclass(frozen=True)
class BaseMetadata:
    """Flat, table-friendly metadata shared by every report section."""

    mode: str
    duration_s: Optional[float] = None
    samples: Optional[int] = None
    nodes_expected: Optional[int] = None
    nodes_observed: Optional[int] = None
    nodes_coverage: Optional[str] = None
    nodes_partial: Optional[bool] = None
    gpus_observed: Optional[int] = None
    global_ranks_seen: Optional[int] = None
    global_ranks_used: Optional[int] = None
    training_total_steps: Optional[int] = None
    training_latest_step: Optional[int] = None
    section_metric_names: Optional[Sequence[str]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        """Serialize section metadata."""
        payload: JsonDict = {
            "mode": self.mode,
            "duration_s": self.duration_s,
            "samples": self.samples,
            "nodes_expected": self.nodes_expected,
            "nodes_observed": self.nodes_observed,
            "nodes_coverage": self.nodes_coverage,
            "nodes_partial": self.nodes_partial,
            "gpus_observed": self.gpus_observed,
            "global_ranks_seen": self.global_ranks_seen,
            "global_ranks_used": self.global_ranks_used,
            "training_total_steps": self.training_total_steps,
            "training_latest_step": self.training_latest_step,
            "section_metric_names": (
                list(self.section_metric_names)
                if self.section_metric_names is not None
                else None
            ),
        }
        payload.update(dict(self.extra))
        return payload


@dataclass(frozen=True)
class RankMetadata(BaseMetadata):
    """Metadata for sections organized by distributed global rank."""

    global_ranks_seen: int = 0
    global_ranks_used: int = 0

    def to_json(self) -> JsonDict:
        payload = super().to_json()
        payload.update({"global_ranks_seen": int(self.global_ranks_seen)})
        payload.update({"global_ranks_used": int(self.global_ranks_used)})
        return payload


@dataclass(frozen=True)
class StepMetadata(RankMetadata):
    """Metadata for rank-scoped sections tied to training progress.

    Window-specific fields such as aligned step counts belong in
    `global.window`; keeping them out of metadata avoids duplicate meanings
    across the final JSON.
    """

    training_total_steps: int = 0
    training_latest_step: Optional[int] = None

    def to_json(self) -> JsonDict:
        payload = super().to_json()
        payload.update(
            {
                "training_total_steps": int(self.training_total_steps),
                "training_latest_step": self.training_latest_step,
            }
        )
        return payload


@dataclass(frozen=True)
class BaseGlobal:
    """Common shape for run-level global section summaries."""

    index_by: str
    window: Optional[Mapping[str, Any]] = None
    average: Optional[Mapping[str, Any]] = None
    median: Optional[Mapping[str, Any]] = None
    worst: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        """Serialize the stable global summary shape."""
        payload: JsonDict = {
            "index_by": self.index_by,
            "window": _copy_mapping(self.window),
            "average": _copy_mapping(self.average),
            "median": _copy_mapping(self.median),
            "worst": _copy_mapping(self.worst),
        }
        payload.update(dict(self.extra))
        return payload


@dataclass(frozen=True)
class GlobalWindow:
    """Flat window metadata for global section calculations."""

    kind: str
    alignment: str
    samples: Optional[int] = None
    steps_analyzed: Optional[int] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    completed_step: Optional[int] = None
    window_size: Optional[int] = None

    def to_json(self) -> JsonDict:
        """Serialize the global calculation window with stable keys."""
        return {
            "kind": self.kind,
            "alignment": self.alignment,
            "samples": self.samples,
            "steps_analyzed": self.steps_analyzed,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "completed_step": self.completed_step,
            "window_size": self.window_size,
        }


@dataclass(frozen=True)
class BaseGroups:
    """Common entity-level detail container for final-report sections."""

    by: str
    rows: Mapping[str, Mapping[str, Any]]

    def to_json(self) -> JsonDict:
        """Serialize the grouped rows using stable row keys."""
        return {
            "by": self.by,
            "rows": {
                str(key): dict(value) for key, value in self.rows.items()
            },
        }


@dataclass(frozen=True)
class GroupRow:
    """Stable shape for one row inside `groups.rows`."""

    identity: Mapping[str, Any]
    metrics: Mapping[str, Any]
    issues: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    diagnosis: Optional[Mapping[str, Any]] = None

    def to_json(self) -> JsonDict:
        """Serialize one grouped row in a predictable order."""
        return {
            "identity": dict(self.identity),
            "diagnosis": _copy_mapping(self.diagnosis),
            "issues": [dict(issue) for issue in self.issues],
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class BaseSectionPayload:
    """Stable outer JSON contract for one final-report section."""

    metadata: Mapping[str, Any]
    diagnosis: Optional[Mapping[str, Any]]
    issues: Sequence[Mapping[str, Any]]
    global_summary: Mapping[str, Any]
    groups: Mapping[str, Any]
    units: Mapping[str, str]
    card: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        """Serialize the final section payload in a consistent order."""
        metadata = dict(self.metadata)
        global_summary = dict(self.global_summary)
        _validate_metric_sets(
            metadata=metadata,
            global_summary=global_summary,
        )
        payload: JsonDict = {
            "metadata": metadata,
            "diagnosis": _copy_mapping(self.diagnosis),
            "issues": [dict(issue) for issue in self.issues],
            "global": global_summary,
            "groups": dict(self.groups),
        }
        payload.update(dict(self.extra))
        payload["units"] = dict(self.units)
        payload["card"] = self.card
        return payload


def empty_section_payload(
    *,
    section_name: str,
    index_by: str = "global_rank",
    reason: str = "Section summary unavailable.",
) -> JsonDict:
    """Build a schema-valid payload for a section that failed to summarize."""
    title = section_name.replace("_", " ").title()
    card = f"TraceML {title} Summary\n- Status: unavailable"
    diagnosis = {
        "severity": "info",
        "status": "NO DATA",
        "reason": reason,
        "kind": "NO_DATA",
    }

    return BaseSectionPayload(
        metadata=BaseMetadata(
            mode="no_data",
            samples=0,
            section_metric_names=[],
        ).to_json(),
        diagnosis=diagnosis,
        issues=[],
        global_summary=BaseGlobal(
            index_by=index_by,
            window=GlobalWindow(
                kind="sample_window",
                alignment="none",
                samples=0,
            ).to_json(),
            average={},
            median={},
            worst={},
        ).to_json(),
        groups=BaseGroups(by=index_by, rows={}).to_json(),
        units={},
        card=card,
    ).to_json()


__all__ = [
    "BaseGlobal",
    "BaseGroups",
    "BaseMetadata",
    "BaseSectionPayload",
    "GlobalWindow",
    "GroupRow",
    "JsonDict",
    "RankMetadata",
    "StepMetadata",
    "empty_section_payload",
]
