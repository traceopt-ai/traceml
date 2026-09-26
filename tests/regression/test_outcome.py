# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for node-scoped guarded training outcomes."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from traceml_ai.regression.contract import parse_guard_contract
from traceml_ai.regression.outcome import (
    OUTCOME_SCHEMA_VERSION,
    contract_digest,
    write_guard_outcome,
)


def _contract():
    return parse_guard_contract(
        {
            "schema_version": 1,
            "workload": {
                "name": "image-training",
                "parameters": {"precision": "bf16", "batch_size": 32},
            },
            "measurement": {"start_step": 10, "completed_steps": 50},
        }
    )


@pytest.mark.parametrize(
    ("exit_code", "status"), [(0, "completed"), (17, "failed")]
)
def test_write_guard_outcome_records_only_bounded_node_facts(
    tmp_path, exit_code, status
) -> None:
    manifest_path = tmp_path / "run" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": "guarded-run",
                "created_at": "2026-09-26T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    outcome_path = tmp_path / "run" / "nodes" / "node_1" / "guard_outcome.json"
    contract = _contract()

    written = write_guard_outcome(
        path=outcome_path,
        manifest_path=manifest_path,
        session_id="guarded-run",
        node_rank=1,
        nnodes=2,
        nproc_per_node=4,
        contract=contract,
        exit_code=exit_code,
    )

    assert written == outcome_path.resolve()
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "session_id": "guarded-run",
        "manifest_created_at": "2026-09-26T10:00:00+00:00",
        "node_rank": 1,
        "nnodes": 2,
        "nproc_per_node": 4,
        "contract_digest": contract_digest(contract),
        "training": {"status": status, "exit_code": exit_code},
        "completed_at": payload["completed_at"],
    }
    assert datetime.fromisoformat(payload["completed_at"]).tzinfo is not None


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"session_id": "another-run", "created_at": "created"},
        {"session_id": "guarded-run"},
    ],
)
def test_write_guard_outcome_requires_current_root_manifest(
    tmp_path, manifest
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        write_guard_outcome(
            path=tmp_path / "guard_outcome.json",
            manifest_path=manifest_path,
            session_id="guarded-run",
            node_rank=0,
            nnodes=1,
            nproc_per_node=1,
            contract=_contract(),
            exit_code=0,
        )

    assert not (tmp_path / "guard_outcome.json").exists()
