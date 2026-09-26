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
    "contract_digest",
    "write_guard_outcome",
]
