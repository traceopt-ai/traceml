# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import traceml_ai.launcher.commands as launcher_commands
from traceml_ai.launcher.cli import build_parser
from traceml_ai.launcher.commands import (
    _dashboard_access_box,
    _derive_final_telemetry,
    _launch_defaults_for_topology,
    _require_dashboard_dependencies,
    _resolve_serve_settings,
    launch_process,
    resolve_existing_script_path,
    run_view,
    validate_launch_args,
)
from traceml_ai.launcher.launch_config import (
    DistributedLaunchConfig,
    RunIdentity,
)
from traceml_ai.launcher.manifest import (
    collect_existing_artifacts,
    load_json_or_warn,
    read_current_finalization_reason,
    update_run_manifest,
    write_run_manifest,
)
from traceml_ai.launcher.process import TrainingOutcome
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    resolve_on_missing_aggregator,
)
from traceml_ai.telemetry.retention import DEFAULT_HISTORY_RETENTION_S


def test_serve_is_a_public_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "serve",
            "--mode",
            "cli",
            "--logs-dir",
            "mylogs",
            "--run-name",
            "demo",
            "--aggregator-host",
            "10.0.0.9",
            "--aggregator-bind-host",
            "0.0.0.0",
            "--aggregator-port",
            "40000",
        ]
    )

    assert args.command == "serve"
    assert args.mode == "cli"
    assert args.aggregator_host == "10.0.0.9"
    assert args.aggregator_bind_host == "0.0.0.0"
    assert args.aggregator_port == 40000


def test_run_and_watch_accept_missing_aggregator_policy() -> None:
    parser = build_parser()

    run_args = parser.parse_args(
        ["run", "train.py", "--on-missing-aggregator", "warn"]
    )
    watch_args = parser.parse_args(["watch", "train.py"])

    assert run_args.on_missing_aggregator == "warn"
    assert watch_args.on_missing_aggregator is None


@pytest.mark.parametrize("default", ["raise", "warn"])
def test_shared_missing_aggregator_resolver_uses_caller_default(
    monkeypatch, default
) -> None:
    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    assert resolve_on_missing_aggregator(None, default=default) == default

    monkeypatch.setenv("TRACEML_ON_MISSING_AGGREGATOR", "warn")
    assert resolve_on_missing_aggregator(None, default=default) == "warn"
    assert resolve_on_missing_aggregator("raise", default=default) == "raise"


def test_shared_missing_aggregator_resolver_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="must be 'raise' or 'warn'"):
        resolve_on_missing_aggregator("continue", default="raise")


def test_serve_maps_flags_into_aggregator_settings(monkeypatch) -> None:
    # Isolate from any TRACEML_* env so the CLI flags drive the result.
    for var in (
        "TRACEML_UI_MODE",
        "TRACEML_MODE",
        "TRACEML_LOGS_DIR",
        "TRACEML_INTERVAL",
        "TRACEML_ENABLE_LOGGING",
    ):
        monkeypatch.delenv(var, raising=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--mode",
            "cli",
            "--logs-dir",
            "mylogs",
            "--run-name",
            "demo",
            "--aggregator-host",
            "10.0.0.9",
            "--aggregator-bind-host",
            "0.0.0.0",
            "--aggregator-port",
            "40000",
        ]
    )

    settings = _resolve_serve_settings(args)

    assert settings.mode == "cli"
    assert settings.logs_dir == "mylogs"
    assert settings.session_id == "demo"
    assert settings.aggregator.connect_host == "10.0.0.9"
    assert settings.aggregator.bind_host == "0.0.0.0"
    assert settings.aggregator.port == 40000


def test_serve_defaults_to_summary(monkeypatch, tmp_path) -> None:
    for var in ("TRACEML_UI_MODE", "TRACEML_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["serve"])

    assert _resolve_serve_settings(args).mode == "summary"


def test_serve_threads_expected_world_size(monkeypatch) -> None:
    monkeypatch.delenv("TRACEML_EXPECTED_WORLD_SIZE", raising=False)
    parser = build_parser()

    # Explicit --nnodes x --nproc-per-node sets the rank count so the
    # aggregator waits for ALL ranks before finalizing.
    args = parser.parse_args(
        ["serve", "--nnodes", "2", "--nproc-per-node", "4"]
    )
    assert _resolve_serve_settings(args).expected_world_size == 8

    # Falls back to TRACEML_EXPECTED_WORLD_SIZE (matching `traceml run`).
    monkeypatch.setenv("TRACEML_EXPECTED_WORLD_SIZE", "3")
    args = parser.parse_args(["serve"])
    assert _resolve_serve_settings(args).expected_world_size == 3

    # Default is 1 when neither flags nor env are set.
    monkeypatch.delenv("TRACEML_EXPECTED_WORLD_SIZE", raising=False)
    args = parser.parse_args(["serve"])
    assert _resolve_serve_settings(args).expected_world_size == 1


def test_serve_dashboard_missing_deps_reports_hint_not_nameerror(
    monkeypatch,
) -> None:
    # Regression: run_serve referenced an undefined constant on the
    # dashboard-deps-missing path, so it raised NameError instead of the
    # SystemExit install hint. Pin that it names the missing deps and exits.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "nicegui" else real_find_spec(name, *a, **k)
        ),
    )
    from traceml_ai.launcher.commands import run_serve

    args = build_parser().parse_args(["serve", "--mode", "dashboard"])
    with pytest.raises(SystemExit) as excinfo:
        run_serve(args)

    message = str(excinfo.value)
    assert "nicegui" in message
    assert "Missing:" in message


def test_dashboard_dep_check_passes_without_plotly(monkeypatch) -> None:
    # Acceptance for the plotly removal: plotly is not imported anywhere,
    # so its absence must not block dashboard mode (previously the guard
    # tuples refused to start the dashboard over a package nothing used).
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "plotly" else real_find_spec(name, *a, **k)
        ),
    )
    from traceml_ai.launcher.commands import _require_dashboard_dependencies

    # Must not raise: nicegui present, plotly absent.
    _require_dashboard_dependencies("dashboard")


def test_serve_configures_logging_without_preset_env(
    monkeypatch, tmp_path
) -> None:
    import logging

    import traceml_ai.aggregator.aggregator_main as agg_main
    from traceml_ai.runtime.settings import (
        AggregatorTransportSettings,
        TraceMLSettings,
    )

    saved_env = {
        key: os.environ.get(key)
        for key in ("TRACEML_LOGS_DIR", "TRACEML_SESSION_ID")
    }
    os.environ.pop("TRACEML_LOGS_DIR", None)
    os.environ.pop("TRACEML_SESSION_ID", None)

    traceml_logger = logging.getLogger("traceml")
    saved_handlers = traceml_logger.handlers[:]
    traceml_logger.handlers.clear()

    # Do not clobber the process signal handlers, and stop before the blocking
    # wait by making aggregator startup raise a controlled error.
    monkeypatch.setattr(agg_main, "_install_signal_handlers", lambda ev: None)

    def _boom(*args, **kwargs):
        raise RuntimeError("stop before blocking")

    monkeypatch.setattr(agg_main, "start_aggregator", _boom)

    settings = TraceMLSettings(
        mode="summary",
        logs_dir=str(tmp_path),
        session_id="serve-test",
        aggregator=AggregatorTransportSettings(
            connect_host="127.0.0.1", bind_host="127.0.0.1", port=0
        ),
    )

    try:
        rc = agg_main.run_aggregator(settings)

        # Clean fatal exit (return 1), not a TypeError in logging setup.
        assert rc == 1
        assert os.environ["TRACEML_LOGS_DIR"] == str(tmp_path)
        assert os.environ["TRACEML_SESSION_ID"] == "serve-test"
    finally:
        for handler in traceml_logger.handlers[:]:
            traceml_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in saved_handlers:
            traceml_logger.addHandler(handler)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    assert args.history_retention is None
    assert args.finalize_timeout_sec is None
    assert args.trace_max_steps is None
    assert not args.capture_stderr
    assert args.args == ["--epochs", "1"]

    # The launcher defers UI/telemetry defaults to the traceml.yaml config
    # resolver, so the argparse default is None ("flag not supplied"). The
    # effective mode default is selected from the launch topology.
    default_args = parser.parse_args(["watch", "train.py"])
    assert default_args.mode is None


def test_build_parser_accepts_disable_traceml_aliases() -> None:
    parser = build_parser()

    dashed = parser.parse_args(["run", "train.py", "--disable-traceml"])
    underscored = parser.parse_args(["run", "train.py", "--disable_traceml"])

    assert dashed.disable_traceml is True
    assert underscored.disable_traceml is True


def test_launch_defaults_use_summary_for_single_node_topologies() -> None:
    defaults = {"mode": "dashboard", "interval": 2.0}

    assert (
        _launch_defaults_for_topology(defaults, nnodes=1)["mode"] == "summary"
    )


def test_launch_defaults_use_summary_for_multinode_topologies() -> None:
    defaults = {"mode": "cli", "interval": 2.0}

    result = _launch_defaults_for_topology(defaults, nnodes=2)

    assert result["mode"] == "summary"
    assert result["interval"] == 2.0


def test_build_parser_accepts_view_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["view", "summary.json"])

    assert args.command == "view"
    assert args.summary == "summary.json"


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
            "--history-retention",
            "2h",
            "--finalize-timeout-sec",
            "120",
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
    assert args.history_retention == 7200.0
    assert args.finalize_timeout_sec == 120.0
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
        history_retention=None,
    )

    with pytest.raises(SystemExit):
        validate_launch_args(args)


@pytest.mark.parametrize("command", ["run", "watch"])
def test_implicit_summary_mode_requires_history(
    command, monkeypatch, tmp_path
) -> None:
    for var in ("TRACEML_UI_MODE", "TRACEML_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    args = build_parser().parse_args([command, str(script), "--no-history"])

    with pytest.raises(SystemExit, match="mode=summary requires history"):
        launch_process(str(script), args)


def test_explicit_live_mode_allows_no_history() -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", "--no-history"]
    )

    validate_launch_args(args)


def test_disabled_launch_validation_skips_traceml_only_checks(
    monkeypatch,
) -> None:
    args = argparse.Namespace(
        mode="dashboard",
        no_history=True,
        html_report=True,
        nnodes=2,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=0,
        run_name="",
        session_id="",
        history_retention=None,
        finalize_timeout_sec=-1.0,
        trace_max_steps=0,
        disable_traceml=True,
    )
    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None,
    )

    validate_launch_args(args)


def test_disabled_launch_validation_honors_env_kill_switch(
    monkeypatch,
) -> None:
    args = argparse.Namespace(
        mode="summary",
        no_history=True,
        html_report=True,
        nnodes=2,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        aggregator_host=None,
        aggregator_bind_host=None,
        aggregator_port=0,
        run_name="",
        session_id="",
        history_retention=None,
        finalize_timeout_sec=-1.0,
        trace_max_steps=0,
        disable_traceml=None,
    )
    monkeypatch.setenv("TRACEML_DISABLED", "1")

    validate_launch_args(args)


def test_disabled_launch_runs_script_directly_and_skips_traceml_setup(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('native')\n", encoding="utf-8")
    (tmp_path / "traceml.yaml").write_text("mode: [\n", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            str(script),
            "--disable-traceml",
            "--mode",
            "summary",
            "--no-history",
            "--html-report",
            "--capture-stderr",
            "--logs-dir",
            str(tmp_path / "logs"),
            "--aggregator-port",
            "0",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "3",
            "--node-rank",
            "1",
            "--master-addr",
            "10.0.0.10",
            "--master-port",
            "29511",
            "--args",
            "--epochs",
            "1",
        ]
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEML_AGGREGATOR_PORT", "9999")
    observed = {}

    class _Proc:
        pid = 12345
        returncode = 17

        def wait(self):
            return self.returncode

    def _start_training_process(train_cmd, env, cwd, *, capture_stderr=False):
        observed["train_cmd"] = train_cmd
        observed["env"] = env
        observed["cwd"] = cwd
        observed["capture_stderr"] = capture_stderr
        return _Proc()

    def _record_shutdown_handler(get_procs, manifest_path=None):
        observed["manifest_path"] = manifest_path

    def _forbidden(*args, **kwargs):
        raise AssertionError("TraceML setup must not run when disabled")

    monkeypatch.setattr(
        launcher_commands,
        "start_training_process",
        _start_training_process,
    )
    monkeypatch.setattr(
        launcher_commands,
        "install_shutdown_handlers",
        _record_shutdown_handler,
    )
    monkeypatch.setattr(
        launcher_commands, "start_aggregator_process", _forbidden
    )
    monkeypatch.setattr(launcher_commands, "write_code_manifest", _forbidden)
    monkeypatch.setattr(launcher_commands, "write_run_manifest", _forbidden)
    monkeypatch.setattr(launcher_commands, "update_run_manifest", _forbidden)

    with pytest.raises(SystemExit) as exc:
        launcher_commands.launch_process(str(script), args)

    assert exc.value.code == 17
    assert observed["train_cmd"] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc_per_node=3",
        "--node_rank=1",
        "--master_addr=10.0.0.10",
        "--master_port=29511",
        str(script),
        "--epochs",
        "1",
    ]
    assert observed["env"]["TRACEML_DISABLED"] == "1"
    assert [key for key in observed["env"] if key.startswith("TRACEML_")] == [
        "TRACEML_DISABLED"
    ]
    assert observed["cwd"] == str(tmp_path.resolve())
    assert observed["capture_stderr"] is False
    assert observed["manifest_path"] is None
    assert not (tmp_path / "logs").exists()
    assert capsys.readouterr().err.splitlines()[-1] == (
        "[TraceML] Training failed — torchrun exited with code 17."
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_training_outcome_only_classifies_direct_negative_signals() -> None:
    signaled = TrainingOutcome(-signal.SIGSEGV)

    assert signaled.signal_name == "SIGSEGV"
    assert signaled.cli_exit_code == 128 + signal.SIGSEGV
    assert TrainingOutcome(1).signal_name is None
    assert TrainingOutcome(1).cli_exit_code == 1


@pytest.mark.parametrize(
    ("train_rc", "aggregator_rc", "finalization_fails"),
    [
        (0, 9, False),
        (0, 0, False),
        (1, 0, False),
        (0, 0, True),
    ],
)
@pytest.mark.parametrize("mode", ["summary", "cli", "dashboard"])
def test_started_training_result_is_authoritative(
    monkeypatch,
    tmp_path,
    capsys,
    train_rc,
    aggregator_rc,
    finalization_fails,
    mode,
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--mode",
            mode,
            "--logs-dir",
            str(tmp_path / "logs"),
            "--run-name",
            "outcome-test",
        ]
    )

    aggregator = Mock(pid=10, returncode=aggregator_rc)
    training = Mock()
    training.poll.return_value = train_rc

    monkeypatch.chdir(tmp_path)
    replacements = {
        "install_shutdown_handlers": Mock(),
        "start_aggregator_process": Mock(return_value=aggregator),
        "wait_for_tcp_listen": Mock(return_value=True),
        "start_training_process": Mock(return_value=training),
        "terminate_process_group": Mock(),
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": Mock(),
        "_require_dashboard_dependencies": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(launcher_commands, name, replacement)

    if finalization_fails:
        monkeypatch.setattr(
            launcher_commands,
            "collect_existing_artifacts",
            Mock(side_effect=OSError("manifest unavailable")),
        )

    with pytest.raises(SystemExit) as exc:
        launcher_commands.launch_process(str(script), args)

    assert exc.value.code == TrainingOutcome(train_rc).cli_exit_code
    stderr_lines = capsys.readouterr().err.splitlines()
    final_line = stderr_lines[-1]
    expected_status = "completed successfully" if train_rc == 0 else "failed"
    assert f"Training {expected_status}" in final_line
    assert stderr_lines[-2].startswith("[TraceML] Telemetry ")
    if train_rc != 0:
        assert "torchrun exited with code 1" in final_line
    if (
        train_rc == 0
        and aggregator_rc == 0
        and not finalization_fails
        and mode == "summary"
    ):
        final_update = replacements["update_run_manifest"].call_args_list[-1]
        assert final_update.kwargs["status"] == "completed"
        assert final_update.kwargs["telemetry_status"] == "failed"
        assert final_update.kwargs["telemetry_reason"] == "summary_missing"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("spawn", "aggregator_spawn_failed"),
        ("readiness", "aggregator_not_ready"),
    ],
)
def test_strict_aggregator_failure_does_not_start_training(
    monkeypatch, tmp_path, failure, reason
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["run", str(script), "--logs-dir", str(tmp_path / "logs")]
    )
    start_training = Mock()
    update_manifest = Mock()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    aggregator = Mock(pid=10, returncode=7)
    aggregator.poll.return_value = 7
    start_aggregator = Mock(return_value=aggregator)
    if failure == "spawn":
        start_aggregator.side_effect = FileNotFoundError("aggregator missing")
    monkeypatch.setattr(
        launcher_commands, "start_aggregator_process", start_aggregator
    )
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=False)
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", start_training
    )
    monkeypatch.setattr(
        launcher_commands, "write_code_manifest", Mock(return_value=None)
    )
    monkeypatch.setattr(
        launcher_commands,
        "write_run_manifest",
        Mock(return_value=tmp_path / "manifest.json"),
    )
    monkeypatch.setattr(
        launcher_commands, "update_run_manifest", update_manifest
    )

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    start_training.assert_not_called()
    assert update_manifest.call_args.kwargs["status"] == "failed"
    assert update_manifest.call_args.kwargs["telemetry_status"] == (
        "unavailable"
    )
    assert update_manifest.call_args.kwargs["telemetry_reason"] == reason


@pytest.mark.parametrize("owner", [True, False])
def test_warn_policy_starts_supervised_training_without_telemetry(
    monkeypatch, tmp_path, owner
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    cli = [
        "run",
        str(script),
        "--mode",
        "cli",
        "--logs-dir",
        str(tmp_path / "logs"),
        "--run-name",
        "warn-run",
        "--on-missing-aggregator",
        "warn",
    ]
    if not owner:
        cli.extend(["--nnodes", "2", "--node-rank", "1"])
    args = build_parser().parse_args(cli)

    events = []
    aggregator = Mock(pid=10, returncode=7)
    aggregator.poll.return_value = 7
    training = Mock()
    training.poll.return_value = 0

    def _stop_aggregator(*_args, **_kwargs):
        events.append("stop")

    def _start_training(*, train_cmd, env, cwd, capture_stderr):
        events.append("train")
        assert env["TRACEML_DISABLED"] == "1"
        return training

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    start_aggregator = Mock(return_value=aggregator)
    monkeypatch.setattr(
        launcher_commands, "start_aggregator_process", start_aggregator
    )
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=False)
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", _start_training
    )
    monkeypatch.setattr(
        launcher_commands, "terminate_process_group", _stop_aggregator
    )
    monkeypatch.setattr(
        launcher_commands, "write_code_manifest", Mock(return_value=None)
    )
    monkeypatch.setattr(
        launcher_commands,
        "write_run_manifest",
        Mock(return_value=tmp_path / "manifest.json"),
    )
    monkeypatch.setattr(launcher_commands, "update_run_manifest", Mock())

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 0
    if owner:
        assert events == ["stop", "train"]
        start_aggregator.assert_called_once()
    else:
        assert events == ["train"]
        start_aggregator.assert_not_called()


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            {
                "telemetry_available": False,
                "startup_reason": "aggregator_not_ready",
            },
            ("unavailable", "aggregator_not_ready", None),
        ),
        (
            {"finalization_reason": "finalization_failed"},
            ("failed", "finalization_failed", None),
        ),
        (
            {"summary_required": True, "summary_exists": False},
            ("failed", "summary_missing", None),
        ),
        (
            {"aggregator_exited_early": True, "aggregator_exit_code": 4},
            ("degraded", "aggregator_exited_early", 4),
        ),
        (
            {"finalization_reason": "finalization_warning"},
            ("degraded", "finalization_warning", None),
        ),
        (
            {"aggregator_exit_code": 0},
            ("complete", None, 0),
        ),
    ],
)
def test_final_telemetry_precedence(inputs, expected) -> None:
    defaults = {
        "telemetry_available": True,
        "startup_reason": None,
        "aggregator_exited_early": False,
        "aggregator_exit_code": None,
        "finalization_reason": None,
        "summary_required": False,
        "summary_exists": True,
    }
    assert _derive_final_telemetry(**{**defaults, **inputs}) == expected


@pytest.mark.parametrize("value", ["0", "-1m", "bad", "nan", "inf"])
def test_history_retention_rejects_invalid_duration(value: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "train.py", "--history-retention", value])


def test_trace_max_steps_must_be_positive() -> None:
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
        history_retention=None,
        trace_max_steps=0,
    )

    with pytest.raises(SystemExit):
        validate_launch_args(args)


def test_dashboard_mode_requires_dashboard_dependencies(monkeypatch) -> None:
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
        history_retention=None,
    )

    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None if package == "nicegui" else object(),
    )

    with pytest.raises(SystemExit, match="pip install -U traceml-ai"):
        validate_launch_args(args)


def test_summary_mode_does_not_require_dashboard_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None,
    )

    _require_dashboard_dependencies("summary")


def test_implicit_mode_defers_dashboard_dependency_check_until_config_resolution(
    monkeypatch,
) -> None:
    args = argparse.Namespace(
        mode=None,
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
        history_retention=None,
    )

    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None if package == "nicegui" else object(),
    )

    validate_launch_args(args)


def test_dashboard_access_box_highlights_url_and_ssh_tunnel() -> None:
    box = _dashboard_access_box(9000)

    assert "TraceML dashboard" in box
    assert "http://127.0.0.1:9000" in box
    assert "ssh -L 9000:127.0.0.1:9000 user@remote-host" in box
    assert box.splitlines()[0].startswith("+")
    assert box.splitlines()[-1].startswith("+")


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
        history_retention=None,
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
        history_retention=None,
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
        history_retention=None,
    )

    with pytest.raises(SystemExit, match="must match"):
        validate_launch_args(args)


def test_resolve_existing_script_path_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_existing_script_path(str(tmp_path / "missing.py"))


def test_run_manifest_write_and_update_merge_correctly(tmp_path) -> None:
    # Atomicity of the underlying write (crash-mid-write leaves the target
    # untouched, no stray temp file) is pinned directly in
    # tests/runtime/test_atomic_io.py; this test only covers the content
    # merge across write_run_manifest -> update_run_manifest.
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
        history_retention_s=DEFAULT_HISTORY_RETENTION_S,
        finalize_timeout_sec=DEFAULT_FINALIZE_TIMEOUT_SEC,
        status="starting",
        telemetry_status="starting",
        launch_cwd=str(tmp_path),
    )
    update_run_manifest(
        manifest_path,
        status="completed",
        artifacts={"summary_card_json": "summary.json"},
        telemetry_status="degraded",
        telemetry_reason="finalization_warning",
        aggregator_exit_code=3,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["telemetry_status"] == "degraded"
    assert payload["telemetry_reason"] == "finalization_warning"
    assert payload["aggregator_exit_code"] == 3
    assert payload["session_id"] == "session"
    assert payload["run"]["run_name"] == "session"
    assert payload["run"]["session_id"] == "session"
    assert payload["launch"]["profile"] == "run"
    assert payload["launch"]["aggregator_host"] == "127.0.0.1"
    assert payload["launch"]["aggregator_port"] == 29765
    assert payload["launch"]["nnodes"] == 1
    assert payload["launch"]["history_retention_s"] == 1800.0
    assert (
        payload["launch"]["finalize_timeout_sec"]
        == DEFAULT_FINALIZE_TIMEOUT_SEC
    )
    assert payload["paths"]["run_root"] == str(session_root.resolve())
    assert payload["artifacts"]["summary_card_json"] == "summary.json"

    update_run_manifest(manifest_path, telemetry_status="complete")
    completed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed["telemetry_status"] == "complete"
    assert "telemetry_reason" not in completed
    assert "aggregator_exit_code" not in completed


def test_finalization_evidence_ignores_stale_and_malformed_files(
    tmp_path,
) -> None:
    aggregator_dir = tmp_path / "aggregator"
    aggregator_dir.mkdir()
    warning_path = aggregator_dir / "finalization_warning.json"
    error_path = aggregator_dir / "finalization_error.json"
    started_at = "2026-08-31T10:00:00+00:00"

    warning_path.write_text(
        json.dumps({"completed_at": "2026-08-31T09:59:59+00:00"}),
        encoding="utf-8",
    )
    error_path.write_text("{malformed", encoding="utf-8")
    assert (
        read_current_finalization_reason(
            aggregator_dir, aggregator_started_at=started_at
        )
        is None
    )

    warning_path.write_text(
        json.dumps({"completed_at": "2026-08-31T10:00:01+00:00"}),
        encoding="utf-8",
    )
    assert (
        read_current_finalization_reason(
            aggregator_dir, aggregator_started_at=started_at
        )
        == "finalization_warning"
    )

    error_path.write_text(
        json.dumps({"completed_at": "2026-08-31T10:00:02+00:00"}),
        encoding="utf-8",
    )
    assert (
        read_current_finalization_reason(
            aggregator_dir, aggregator_started_at=started_at
        )
        == "finalization_failed"
    )


def test_load_json_or_warn_preserves_corrupt_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{bad json", encoding="utf-8")

    assert load_json_or_warn(manifest_path) == {}
    assert manifest_path.with_suffix(".json.corrupt").exists()


def test_collect_existing_artifacts_only_returns_existing_files(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry"
    summary_path = Path(str(db_path) + "_summary_card.txt")
    db_path.write_text("", encoding="utf-8")
    summary_path.write_text("summary", encoding="utf-8")

    artifacts = collect_existing_artifacts(db_path)

    assert artifacts == {
        "db": str(db_path),
        "summary_card_txt": str(summary_path),
    }


def test_collect_existing_artifacts_includes_stderr_tail(tmp_path) -> None:
    db_path = tmp_path / "aggregator" / "telemetry"
    stderr_path = tmp_path / "crash_stderr.log"
    stderr_path.write_bytes(b"native crash details\n")

    artifacts = collect_existing_artifacts(
        db_path,
        session_root=tmp_path,
    )

    assert artifacts["crash_stderr_log"] == str(stderr_path)


def test_run_view_reports_user_facing_errors(tmp_path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        run_view(argparse.Namespace(summary=str(summary_path)))

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert "[TraceML] ERROR:" in captured.err
    assert "does not contain printable text" in captured.err


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
