# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Node-scoped training outcomes for guarded launcher runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traceml_ai.regression.contract import MeasurementContract
from traceml_ai.utils.atomic_io import write_json_atomic

OUTCOME_FILENAME = "guard_outcome.json"
OUTCOME_SCHEMA_VERSION = 1


class OutcomeValidationError(ValueError):
    """Raised when a node outcome is invalid or belongs to another run."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def contract_digest(contract: MeasurementContract) -> str:
    """Return a stable digest of one normalized measurement contract."""
    encoded = json.dumps(
        contract.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _manifest_created_at(manifest_path: Path, session_id: str) -> str:
    """Read the current root manifest identity from the shared run directory."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest: Any = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("root manifest must contain a JSON object")
    if manifest.get("session_id") != session_id:
        raise ValueError("root manifest belongs to a different session")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("root manifest is missing created_at")
    return created_at


def validate_guard_outcome_binding(
    outcome: Any,
    *,
    session_id: str,
    manifest_created_at: str,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    contract: MeasurementContract,
) -> None:
    """Require a node outcome to belong to the expected guarded run.

    Session and manifest identity distinguish a new run from files left in a
    reused run directory. Topology and contract checks then ensure that every
    accepted node participated in the same launch and measurement declaration.
    Parsing the training result belongs to node-0 collection, which consumes
    this binding check before accepting an outcome.
    """
    if not isinstance(outcome, dict):
        raise OutcomeValidationError(
            "node_outcome_invalid",
            "node outcome must contain a JSON object",
        )
    schema_version = outcome.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != OUTCOME_SCHEMA_VERSION
    ):
        raise OutcomeValidationError(
            "node_outcome_invalid",
            "node outcome schema version is unsupported",
        )

    if not isinstance(outcome.get("session_id"), str) or not isinstance(
        outcome.get("manifest_created_at"), str
    ):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome run identity is invalid"
        )
    if (
        outcome["session_id"] != session_id
        or outcome["manifest_created_at"] != manifest_created_at
    ):
        raise OutcomeValidationError(
            "node_outcome_stale", "node outcome belongs to another run"
        )

    topology = tuple(
        outcome.get(field)
        for field in ("node_rank", "nnodes", "nproc_per_node")
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in topology
    ):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome launch topology is invalid"
        )
    if topology != (node_rank, nnodes, nproc_per_node):
        raise OutcomeValidationError(
            "node_outcome_conflict",
            "node outcome conflicts with the captured launch topology",
        )

    digest = outcome.get("contract_digest")
    if not isinstance(digest, str):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome contract digest is invalid"
        )
    if digest != contract_digest(contract):
        raise OutcomeValidationError(
            "node_outcome_conflict",
            "node outcome conflicts with the captured measurement contract",
        )


def write_guard_outcome(
    *,
    path: Path,
    manifest_path: Path,
    session_id: str,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    contract: MeasurementContract,
    exit_code: int,
) -> Path:
    """Atomically write the completed local launcher's guarded outcome."""
    destination = Path(path).resolve()
    payload = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "session_id": session_id,
        "manifest_created_at": _manifest_created_at(
            Path(manifest_path).resolve(), session_id
        ),
        "node_rank": int(node_rank),
        "nnodes": int(nnodes),
        "nproc_per_node": int(nproc_per_node),
        "contract_digest": contract_digest(contract),
        "training": {
            "status": "completed" if int(exit_code) == 0 else "failed",
            "exit_code": int(exit_code),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(destination, payload, sort_keys=True)
    return destination


__all__ = [
    "OUTCOME_FILENAME",
    "OUTCOME_SCHEMA_VERSION",
    "OutcomeValidationError",
    "contract_digest",
    "validate_guard_outcome_binding",
    "write_guard_outcome",
]
