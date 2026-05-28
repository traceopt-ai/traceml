# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path

import pytest

from traceml_ai.launcher.cli import build_parser
from traceml_ai.launcher.commands import (
    resolve_existing_script_path,
    validate_launch_args,
)
from traceml_ai.launcher.manifest import (
    collect_existing_artifacts,
    load_json_or_warn,
    update_run_manifest,
    write_run_manifest,
)
from traceml_ai.launcher.launch_config import (
    DistributedLaunchConfig,
    RunIdentity,
)
from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS


def test_build_parser_preserves_launch_commands() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "train.py",
            "--mode",
            "summary",
            "--nproc-per-node",
            "2",
            "--args",
            "--epochs",
            "1",
        ]
    )

    assert args.command == "run"
    assert args.mode == "summary"
    assert args.nproc_per_node == 2
    assert args.nnodes == 1
    assert args.node_rank == 0
    assert args.master_addr == "127.0.0.1"
    assert args.run_name == ""
    assert args.session_id == ""
    assert args.summary_window_rows == DEFAULT_SUMMARY_WINDOW_ROWS
    assert args.args == ["--epochs", "1"]

    default_args = parser.parse_args(["watch", "train.py"])
    assert default_args.mode == "summary"


def test_build_parser_accepts_multinode_launch_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "train.py",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "4",
            "--node-rank",
            "1",
            "--master-addr",
            "10.0.0.10",
            "--master-port",
            "29511",
            "--aggregator-host",
            "10.0.0.10",
            "--aggregator-bind-host",
            "0.0.0.0",
            "--aggregator-port",
            "29888",
            "--summary-window-rows",
            "2048",
            "--run-name",
            "multi_node_run",
        ]
    )

    assert args.nnodes == 2
    assert args.nproc_per_node == 4
    assert args.node_rank == 1
    assert args.master_addr == "10.0.0.10"
    assert args.master_port == 29511
    assert args.aggregator_host == "10.0.0.10"
    assert args.aggregator_bind_host == "0.0.0.0"
    assert args.aggregator_port == 29888
    assert args.summary_window_rows == 2048
    assert args.run_name == "multi_node_run"


def test_summary_mode_requires_history() -> None:
    args = argparse.Namespace(
        mode="summary",
        no_history=True,
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="",
        session_id="test-session",
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
    )

    with pytest.raises(SystemExit):
        validate_launch_args(args)


def test_summary_window_rows_must_be_positive() -> None:
    args = argparse.Namespace(
        mode="cli",
        no_history=False,
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="",
        session_id="",
        summary_window_rows=0,
    )

    with pytest.raises(SystemExit):
        validate_launch_args(args)


def test_dashboard_mode_requires_dashboard_extra(monkeypatch) -> None:
    args = argparse.Namespace(
        mode="dashboard",
        no_history=False,
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="",
        session_id="",
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
    )

    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None if package == "nicegui" else object(),
    )

    with pytest.raises(SystemExit, match=r"traceml-ai\[dashboard\]"):
        validate_launch_args(args)


def test_multinode_launch_requires_run_name_or_session_id() -> None:
    args = argparse.Namespace(
        mode="summary",
        no_history=False,
        nnodes=2,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="",
        session_id="",
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
    )

    with pytest.raises(SystemExit, match="--run-name is required"):
        validate_launch_args(args)


def test_multinode_launch_accepts_run_name() -> None:
    args = argparse.Namespace(
        mode="summary",
        no_history=False,
        nnodes=2,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="multi_node_run",
        session_id="",
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
    )

    validate_launch_args(args)


def test_launch_args_reject_conflicting_run_name_and_session_id() -> None:
    args = argparse.Namespace(
        mode="summary",
        no_history=False,
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="run_a",
        session_id="run_b",
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
    )

    with pytest.raises(SystemExit, match="must match"):
        validate_launch_args(args)


def test_resolve_existing_script_path_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_existing_script_path(str(tmp_path / "missing.py"))


def test_run_manifest_write_and_update_are_atomic(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    session_root = tmp_path / "logs" / "session"

    manifest_path = write_run_manifest(
        session_root=session_root,
        session_id="session",
        script_path=str(script),
        profile="run",
        ui_mode="cli",
        logs_dir=str(tmp_path / "logs"),
        aggregator_host="127.0.0.1",
        aggregator_bind_host="127.0.0.1",
        aggregator_port=29765,
        nnodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        nproc_per_node=1,
        history_enabled=True,
        summary_window_rows=DEFAULT_SUMMARY_WINDOW_ROWS,
        status="starting",
        launch_cwd=str(tmp_path),
    )
    update_run_manifest(
        manifest_path,
        status="completed",
        artifacts={"summary_card_json": "summary.json"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["session_id"] == "session"
    assert payload["run"]["run_name"] == "session"
    assert payload["run"]["session_id"] == "session"
    assert payload["launch"]["profile"] == "run"
    assert payload["launch"]["aggregator_host"] == "127.0.0.1"
    assert payload["launch"]["aggregator_port"] == 29765
    assert payload["launch"]["nnodes"] == 1
    assert (
        payload["launch"]["summary_window_rows"] == DEFAULT_SUMMARY_WINDOW_ROWS
    )
    assert payload["paths"]["run_root"] == str(session_root.resolve())
    assert payload["artifacts"]["summary_card_json"] == "summary.json"


def test_load_json_or_warn_preserves_corrupt_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{bad json", encoding="utf-8")

    assert load_json_or_warn(manifest_path) == {}
    assert manifest_path.with_suffix(".json.corrupt").exists()


def test_collect_existing_artifacts_only_returns_existing_files(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry"
    summary_path = Path(str(db_path) + ".summary_card.txt")
    db_path.write_text("", encoding="utf-8")
    summary_path.write_text("summary", encoding="utf-8")

    artifacts = collect_existing_artifacts(db_path)

    assert artifacts == {
        "db": str(db_path),
        "summary_card_txt": str(summary_path),
    }


def test_distributed_launch_config_builds_torchrun_command() -> None:
    args = argparse.Namespace(
        nnodes=2,
        nproc_per_node=3,
        node_rank=1,
        master_addr="10.0.0.10",
        master_port=29511,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
        run_name="",
        session_id="test-session",
    )

    cfg = DistributedLaunchConfig.from_args(args)
    cmd = cfg.torchrun.to_command()

    assert cmd == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc_per_node=3",
        "--node_rank=1",
        "--master_addr=10.0.0.10",
        "--master_port=29511",
    ]
    assert cfg.aggregator.connect_host == "10.0.0.10"
    assert cfg.aggregator.bind_host == "0.0.0.0"
    assert not cfg.aggregator.is_owner(node_rank=1)


def test_single_node_launch_config_keeps_local_defaults() -> None:
    args = argparse.Namespace(
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=29765,
    )

    cfg = DistributedLaunchConfig.from_args(args)

    assert cfg.aggregator.connect_host == "127.0.0.1"
    assert cfg.aggregator.bind_host == "127.0.0.1"
    assert cfg.aggregator.is_owner(node_rank=0)


def test_run_identity_prefers_run_name() -> None:
    args = argparse.Namespace(run_name="trial_017", session_id="")

    identity = RunIdentity.from_args(args, generated_session_id="generated")

    assert identity.run_name == "trial_017"
    assert identity.session_id == "trial_017"
    assert identity.source == "run_name"
    assert identity.to_manifest() == {
        "run_name": "trial_017",
        "session_id": "trial_017",
        "identity_source": "run_name",
    }


def test_run_identity_keeps_session_id_alias() -> None:
    args = argparse.Namespace(run_name="", session_id="legacy_run")

    identity = RunIdentity.from_args(args, generated_session_id="generated")

    assert identity.run_name == "legacy_run"
    assert identity.session_id == "legacy_run"
    assert identity.source == "session_id"


def test_run_identity_allows_matching_run_name_and_session_id() -> None:
    args = argparse.Namespace(run_name="same_run", session_id="same_run")

    identity = RunIdentity.from_args(args)

    assert identity.run_name == "same_run"
    assert identity.session_id == "same_run"


def test_run_identity_rejects_conflicting_names() -> None:
    args = argparse.Namespace(run_name="run_a", session_id="run_b")

    with pytest.raises(ValueError, match="must match"):
        RunIdentity.from_args(args)


def test_run_identity_requires_explicit_name_when_requested() -> None:
    args = argparse.Namespace(run_name="", session_id="")

    with pytest.raises(ValueError, match="--run-name is required"):
        RunIdentity.from_args(
            args,
            generated_session_id="generated",
            require_explicit=True,
        )


def test_run_identity_rejects_path_segments() -> None:
    args = argparse.Namespace(run_name="sweep/run", session_id="")

    with pytest.raises(ValueError, match="single path segment"):
        RunIdentity.from_args(args)
