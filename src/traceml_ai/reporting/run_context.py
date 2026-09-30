# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Portable run context projected from the launcher manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from traceml_ai.regression.contract import (
    ContractValidationError,
    parse_guard_contract,
)

_RUN_STATUSES = frozenset(
    {"starting", "running", "completed", "failed", "interrupted"}
)
_LAUNCH_PROFILES = frozenset({"run", "watch"})
_LAUNCHER_COMPLETION_STATUSES = frozenset({"completed", "incomplete"})
_LAUNCHER_REASON_CODES = frozenset(
    {
        "node_outcome_missing",
        "node_outcome_invalid",
        "node_outcome_conflict",
        "node_training_failed",
        "node_outcome_collection_failed",
    }
)


@dataclass(frozen=True, slots=True)
class RunManifestProjection:
    """Allowlisted manifest facts used by the final report."""

    run_name: Optional[str]
    duration_s: Optional[float]
    run_context: dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _known_string(value: Any, allowed: frozenset[str]) -> Optional[str]:
    return value if isinstance(value, str) and value in allowed else None


def _run_name(manifest: Mapping[str, Any], root: Path) -> Optional[str]:
    run = _mapping(manifest.get("run"))
    candidates = (run.get("run_name"), manifest.get("session_id"), root.name)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _duration_s(manifest: Mapping[str, Any]) -> Optional[float]:
    lifecycle = _mapping(manifest.get("lifecycle"))
    started_at = lifecycle.get("training_started_at")
    ended_at = lifecycle.get("training_ended_at")
    if not isinstance(started_at, str) or not isinstance(ended_at, str):
        return None
    try:
        duration = (
            datetime.fromisoformat(ended_at)
            - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (TypeError, ValueError):
        return None
    return duration if duration >= 0.0 else None


def _declaration(manifest: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    guard = _mapping(manifest.get("guard"))
    if "contract" not in guard:
        return None
    try:
        return parse_guard_contract(guard["contract"]).to_dict()
    except ContractValidationError:
        return None


def _launcher_completion(
    manifest: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    training = _mapping(_mapping(manifest.get("guard")).get("training"))
    status = _known_string(
        training.get("status"), _LAUNCHER_COMPLETION_STATUSES
    )
    nodes_observed = training.get("nodes_observed")
    reasons = training.get("reasons")
    if (
        status is None
        or isinstance(nodes_observed, bool)
        or not isinstance(nodes_observed, int)
        or nodes_observed < 0
        or not isinstance(reasons, list)
        or any(
            not isinstance(reason, str) or reason not in _LAUNCHER_REASON_CODES
            for reason in reasons
        )
    ):
        return None
    return {
        "status": status,
        "nodes_observed": nodes_observed,
        "reason_codes": list(reasons),
    }


def _run_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    # Build from an allowlist instead of copying manifest blocks: manifests
    # contain paths and machine identity that must not enter portable reports.
    context: dict[str, Any] = {}

    status = _known_string(manifest.get("status"), _RUN_STATUSES)
    profile = _known_string(
        _mapping(manifest.get("launch")).get("profile"), _LAUNCH_PROFILES
    )
    run = {}
    if status is not None:
        run["status"] = status
    if profile is not None:
        run["profile"] = profile
    if run:
        context["run"] = run

    declaration = _declaration(manifest)
    if declaration is not None:
        context["declaration"] = declaration

    launch = _mapping(manifest.get("launch"))
    expected_nodes = _positive_int(launch.get("nnodes"))
    processes_per_node = _positive_int(launch.get("nproc_per_node"))
    execution: dict[str, Any] = {}
    if expected_nodes is not None:
        execution["expected_nodes"] = expected_nodes
    if processes_per_node is not None:
        execution["processes_per_node"] = processes_per_node
    if expected_nodes is not None and processes_per_node is not None:
        execution["expected_world_size"] = expected_nodes * processes_per_node

    launcher_completion = _launcher_completion(manifest)
    if launcher_completion is not None:
        execution["launcher_completion"] = launcher_completion
    if execution:
        context["execution"] = execution
    return context


def load_run_manifest_projection(
    session_root: Optional[str | Path],
) -> RunManifestProjection:
    """Read one root manifest and return only final-report-safe facts.

    Missing or malformed manifests are tolerated because final reporting runs
    during cleanup and must not replace the training process result.
    """
    if session_root is None:
        return RunManifestProjection(None, None, {})

    root = Path(session_root).resolve()
    manifest: Mapping[str, Any] = {}
    try:
        with open(root / "manifest.json", "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            manifest = loaded
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass

    return RunManifestProjection(
        run_name=_run_name(manifest, root),
        duration_s=_duration_s(manifest),
        run_context=_run_context(manifest),
    )


__all__ = ["RunManifestProjection", "load_run_manifest_projection"]
