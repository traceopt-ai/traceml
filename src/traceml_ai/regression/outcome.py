# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Node-scoped training outcomes for guarded launcher runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traceml_ai.regression.contract import MeasurementContract
from traceml_ai.utils.atomic_io import write_json_atomic

OUTCOME_FILENAME = "guard_outcome.json"
OUTCOME_SCHEMA_VERSION = 1
MAX_OUTCOME_BYTES = 16 * 1024
OUTCOME_POLL_INTERVAL_S = 0.05


class OutcomeValidationError(ValueError):
    """Raised when a node outcome is invalid or belongs to another run."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class NodeTrainingOutcome:
    """Validated training result reported by one launcher node."""

    node_rank: int
    exit_code: int

    def to_dict(self) -> dict[str, int]:
        return {"node_rank": self.node_rank, "exit_code": self.exit_code}


@dataclass(frozen=True, slots=True)
class GuardTrainingResult:
    """Consolidated launcher outcome stored in the root run manifest."""

    status: str
    nodes_expected: int
    nodes_observed: int
    reasons: tuple[str, ...]
    nodes: tuple[NodeTrainingOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "nodes_expected": self.nodes_expected,
            "nodes_observed": self.nodes_observed,
            "reasons": list(self.reasons),
            "nodes": [node.to_dict() for node in self.nodes],
        }


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


def _manifest_run_name(manifest_path: Path, session_id: str) -> str:
    """Read the public run identity from the current root manifest."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest: Any = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("root manifest must contain a JSON object")
    if manifest.get("session_id") != session_id:
        raise ValueError("root manifest belongs to a different session")
    run = manifest.get("run")
    run_name = run.get("run_name") if isinstance(run, dict) else None
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("root manifest is missing run.run_name")
    return run_name


def validate_guard_outcome_binding(
    outcome: Any,
    *,
    run_name: str,
    node_rank: int,
    nnodes: int,
    nproc_per_node: int,
    contract: MeasurementContract,
) -> NodeTrainingOutcome:
    """Validate and return one node's outcome for the expected guarded run.

    The public run name identifies the run directory. Topology and contract
    checks then ensure that every accepted node participated in the same launch
    and measurement declaration. Parsing the training result belongs to node-0
    collection, which consumes this check before accepting an outcome.
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

    if not isinstance(outcome.get("run_name"), str):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome run identity is invalid"
        )
    if outcome["run_name"] != run_name:
        raise OutcomeValidationError(
            "node_outcome_conflict", "node outcome belongs to another run"
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

    if (
        not isinstance(outcome.get("completed_at"), str)
        or not outcome["completed_at"]
    ):
        raise OutcomeValidationError(
            "node_outcome_invalid",
            "node outcome completed_at is invalid",
        )

    training = outcome.get("training")
    if not isinstance(training, dict):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome training result is invalid"
        )
    exit_code = training.get("exit_code")
    status = training.get("status")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code < 0
        or status not in {"completed", "failed"}
        or (exit_code == 0) != (status == "completed")
    ):
        raise OutcomeValidationError(
            "node_outcome_invalid", "node outcome training result is invalid"
        )
    return NodeTrainingOutcome(node_rank=node_rank, exit_code=exit_code)


def _read_outcome(path: Path) -> Any:
    """Read one bounded JSON outcome file."""
    with open(path, "rb") as handle:
        encoded = handle.read(MAX_OUTCOME_BYTES + 1)
    if len(encoded) > MAX_OUTCOME_BYTES:
        raise ValueError("node outcome exceeds the size limit")
    return json.loads(encoded.decode("utf-8"))


def collect_guard_outcomes(
    *,
    session_root: Path,
    run_name: str,
    nnodes: int,
    nproc_per_node: int,
    contract: MeasurementContract,
    timeout_s: float,
) -> GuardTrainingResult:
    """Collect the expected launcher outcomes within one bounded wait."""
    expected = max(1, int(nnodes))
    pending = set(range(expected))
    observed: dict[int, NodeTrainingOutcome] = {}
    reasons: list[str] = []
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    root = Path(session_root).resolve()

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    while pending:
        for node_rank in sorted(pending):
            path = root / "nodes" / f"node_{node_rank}" / OUTCOME_FILENAME
            try:
                payload = _read_outcome(path)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                add_reason("node_outcome_invalid")
                pending.remove(node_rank)
                continue

            try:
                node = validate_guard_outcome_binding(
                    payload,
                    run_name=run_name,
                    node_rank=node_rank,
                    nnodes=expected,
                    nproc_per_node=nproc_per_node,
                    contract=contract,
                )
            except OutcomeValidationError as exc:
                add_reason(exc.reason)
            else:
                observed[node_rank] = node
                if node.exit_code != 0:
                    add_reason("node_training_failed")
            pending.remove(node_rank)

        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(
            min(OUTCOME_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        )

    if pending:
        add_reason("node_outcome_missing")

    nodes = tuple(observed[rank] for rank in sorted(observed))
    return GuardTrainingResult(
        status=(
            "completed"
            if not reasons and len(nodes) == expected
            else "incomplete"
        ),
        nodes_expected=expected,
        nodes_observed=len(nodes),
        reasons=tuple(reasons),
        nodes=nodes,
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
    run_name = _manifest_run_name(Path(manifest_path).resolve(), session_id)
    payload = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "run_name": run_name,
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
    "GuardTrainingResult",
    "MAX_OUTCOME_BYTES",
    "NodeTrainingOutcome",
    "OUTCOME_FILENAME",
    "OUTCOME_SCHEMA_VERSION",
    "OutcomeValidationError",
    "collect_guard_outcomes",
    "contract_digest",
    "validate_guard_outcome_binding",
    "write_guard_outcome",
]
