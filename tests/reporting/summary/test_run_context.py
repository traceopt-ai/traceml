# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from traceml_ai.reporting.run_context import load_run_manifest_projection


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "session_id": "internal-session",
        "status": "completed",
        "run": {"run_name": "candidate-run", "session_id": "internal-session"},
        "lifecycle": {
            "training_started_at": "2026-09-29T10:00:00+00:00",
            "training_ended_at": "2026-09-29T10:02:03.500000+00:00",
        },
        "launch": {
            "profile": "run",
            "nnodes": 2,
            "nproc_per_node": 4,
            "script_path": "/secret/train.py",
            "aggregator_host": "10.0.0.8",
            "aggregator_port": 29765,
        },
        "host": {"hostname": "private-host"},
        "paths": {"session_root": "/secret/logs/candidate-run"},
        "guard": {
            "contract": {
                "schema_version": 1,
                "workload": {
                    "name": "resnet50-training",
                    "parameters": {
                        "precision": "bf16",
                        "batch_size": 32,
                        "compile": True,
                    },
                },
                "measurement": {"start_step": 10, "completed_steps": 50},
            },
            "training": {
                "status": "completed",
                "nodes_expected": 2,
                "nodes_observed": 2,
                "reasons": [],
                "nodes": [
                    {"node_rank": 0, "exit_code": 0},
                    {"node_rank": 1, "exit_code": 0},
                ],
            },
        },
    }


def _load(tmp_path, manifest: object):
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return load_run_manifest_projection(tmp_path)


def test_loads_one_complete_portable_projection(tmp_path) -> None:
    projection = _load(tmp_path, _manifest())

    assert projection.run_name == "candidate-run"
    assert projection.duration_s == pytest.approx(123.5)
    assert projection.run_context == {
        "run": {"status": "completed", "profile": "run"},
        "declaration": {
            "schema_version": 1,
            "workload": {
                "name": "resnet50-training",
                "parameters": {
                    "batch_size": 32,
                    "compile": True,
                    "precision": "bf16",
                },
            },
            "measurement": {"start_step": 10, "completed_steps": 50},
        },
        "execution": {
            "expected_nodes": 2,
            "processes_per_node": 4,
            "expected_world_size": 8,
            "launcher_completion": {
                "status": "completed",
                "nodes_observed": 2,
                "reason_codes": [],
            },
        },
    }


def test_ordinary_run_omits_guard_sections(tmp_path) -> None:
    manifest = _manifest()
    manifest.pop("guard")

    projection = _load(tmp_path, manifest)

    assert projection.run_context == {
        "run": {"status": "completed", "profile": "run"},
        "execution": {
            "expected_nodes": 2,
            "processes_per_node": 4,
            "expected_world_size": 8,
        },
    }


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_preserves_known_non_success_run_status(tmp_path, status) -> None:
    manifest = _manifest()
    manifest["status"] = status

    projection = _load(tmp_path, manifest)

    assert projection.run_context["run"]["status"] == status


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "future-status"), ("profile", "future-profile")],
)
def test_omits_unknown_run_values(tmp_path, field, value) -> None:
    manifest = _manifest()
    if field == "status":
        manifest[field] = value
    else:
        manifest["launch"][field] = value

    run = _load(tmp_path, manifest).run_context["run"]

    assert field not in run


def test_omits_malformed_contract_but_keeps_valid_completion(tmp_path) -> None:
    manifest = _manifest()
    manifest["guard"]["contract"] = {"schema_version": 1}

    context = _load(tmp_path, manifest).run_context

    assert "declaration" not in context
    assert "launcher_completion" in context["execution"]


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest["guard"].update(
            training={"status": "completed"}
        ),
        lambda manifest: manifest["guard"]["training"].update(
            nodes_observed=-1
        ),
        lambda manifest: manifest["guard"]["training"].update(reasons=[[]]),
        lambda manifest: manifest["guard"]["training"].update(
            status="future-status"
        ),
        lambda manifest: manifest["guard"]["training"].update(
            reasons=["future-reason"]
        ),
    ],
)
def test_omits_malformed_optional_guard_block(tmp_path, change) -> None:
    manifest = _manifest()
    change(manifest)

    context = _load(tmp_path, manifest).run_context

    assert "declaration" in context
    assert "launcher_completion" not in context["execution"]


def test_projects_truthful_incomplete_launcher_result(tmp_path) -> None:
    manifest = _manifest()
    manifest["guard"]["training"].update(
        status="incomplete",
        nodes_observed=1,
        reasons=["node_outcome_missing"],
    )

    completion = _load(tmp_path, manifest).run_context["execution"][
        "launcher_completion"
    ]

    assert completion == {
        "status": "incomplete",
        "nodes_observed": 1,
        "reason_codes": ["node_outcome_missing"],
    }


def test_does_not_export_non_allowlisted_manifest_values(tmp_path) -> None:
    manifest = _manifest()
    manifest["environment"] = {"TOKEN": "top-secret"}
    manifest["guard"]["contract_digest"] = "sha256:private"
    manifest["guard"]["training"]["completed_at"] = "private-timestamp"

    encoded = json.dumps(_load(tmp_path, manifest).run_context, sort_keys=True)

    for forbidden in (
        "internal-session",
        "/secret",
        "10.0.0.8",
        "29765",
        "private-host",
        "top-secret",
        "sha256:private",
        "private-timestamp",
        "node_rank",
        "exit_code",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    "started,ended",
    [
        (None, "2026-09-29T10:01:00+00:00"),
        ("invalid", "2026-09-29T10:01:00+00:00"),
        ("2026-09-29T10:02:00+00:00", "2026-09-29T10:01:00+00:00"),
        ("2026-09-29T10:00:00", "2026-09-29T10:01:00+00:00"),
    ],
)
def test_invalid_lifecycle_has_no_duration(tmp_path, started, ended) -> None:
    manifest = _manifest()
    manifest["lifecycle"] = {
        "training_started_at": started,
        "training_ended_at": ended,
    }

    assert _load(tmp_path, manifest).duration_s is None


@pytest.mark.parametrize("contents", [None, b"{bad json", b"\xff\xfe{", b"[]"])
def test_malformed_or_non_object_manifest_falls_back_to_directory_name(
    tmp_path, contents
) -> None:
    run_root = tmp_path / "fallback-run"
    run_root.mkdir()
    if contents is not None:
        (run_root / "manifest.json").write_bytes(contents)

    projection = load_run_manifest_projection(run_root)

    assert projection.run_name == "fallback-run"
    assert projection.duration_s is None
    assert projection.run_context == {}


def test_manifest_value_error_falls_back_to_directory_name(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "fallback-run"
    run_root.mkdir()
    (run_root / "manifest.json").write_text("{}", encoding="utf-8")

    def reject_value(_handle):
        raise ValueError("value cannot be decoded")

    monkeypatch.setattr(json, "load", reject_value)

    projection = load_run_manifest_projection(run_root)

    assert projection.run_name == "fallback-run"
    assert projection.duration_s is None
    assert projection.run_context == {}


def test_missing_session_root_returns_empty_projection() -> None:
    projection = load_run_manifest_projection(None)

    assert projection.run_name is None
    assert projection.duration_s is None
    assert projection.run_context == {}
