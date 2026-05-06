import argparse
import json
from pathlib import Path

import pytest

from traceml.launcher.cli import build_parser
from traceml.launcher.commands import (
    resolve_existing_script_path,
    validate_launch_args,
)
from traceml.launcher.manifest import (
    collect_existing_artifacts,
    load_json_or_warn,
    update_run_manifest,
    write_run_manifest,
)
from traceml.launcher.process import build_torchrun_base_cmd


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
    assert args.args == ["--epochs", "1"]


def test_summary_mode_requires_history() -> None:
    args = argparse.Namespace(mode="summary", no_history=True)

    with pytest.raises(SystemExit):
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
        tcp_host="127.0.0.1",
        tcp_port=29765,
        nproc_per_node=1,
        history_enabled=True,
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
    assert payload["launch"]["profile"] == "run"
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


def test_build_torchrun_base_cmd_uses_current_interpreter() -> None:
    cmd = build_torchrun_base_cmd(3)

    assert cmd[-2:] == ["torch.distributed.run", "--nproc_per_node=3"]
