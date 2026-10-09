# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import argparse
import errno
import io
import json
import os
import signal
import socket
import subprocess
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
    is_current_summary_artifact,
    load_json_or_warn,
    node_artifact_dir,
    read_current_finalization_reason,
    update_run_manifest,
    write_run_manifest,
)
from traceml_ai.launcher.process import (
    AggregatorPortInUseError,
    ProcessOutputResult,
    TrainingOutcome,
    ensure_aggregator_port_free,
)
from traceml_ai.launcher.run_directory import reserve_run_root
from traceml_ai.regression.outcome import (
    GuardTrainingResult,
    NodeTrainingOutcome,
)
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    resolve_on_missing_aggregator,
)
from traceml_ai.telemetry.retention import DEFAULT_HISTORY_RETENTION_S
from traceml_ai.transport.tcp_transport import (
    TCPConfig,
    TCPServer,
    bind_exclusive_listener,
)


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


def test_compare_passes_optional_step_time_regression_threshold(
    monkeypatch,
) -> None:
    # Keep this as one CLI smoke test; policy edge cases belong in policy tests.
    parser = build_parser()
    default_args = parser.parse_args(["compare", "before.json", "after.json"])
    policy_args = parser.parse_args(
        [
            "compare",
            "before.json",
            "after.json",
            "--max-step-time-regression-pct",
            "5",
        ]
    )

    assert default_args.max_step_time_regression_pct is None
    assert policy_args.max_step_time_regression_pct == "5"
    compare_summaries = Mock()
    monkeypatch.setattr(
        "traceml_ai.reporting.compare.compare_summaries",
        compare_summaries,
    )
    launcher_commands.run_compare(default_args)
    launcher_commands.run_compare(policy_args)

    thresholds = [
        item.kwargs["max_step_time_regression_pct"]
        for item in compare_summaries.call_args_list
    ]
    assert thresholds == [None, "5"]


@pytest.mark.parametrize(
    ("payload", "expected_exit"),
    [
        ({}, None),
        ({"ci_policy": {"result": "FASTER_IN_THIS_PAIR"}}, None),
        ({"ci_policy": {"result": "WITHIN_THRESHOLD_IN_THIS_PAIR"}}, None),
        ({"ci_policy": {"result": "SLOWER_IN_THIS_PAIR"}}, 4),
        ({"ci_policy": {"result": "INCONCLUSIVE"}}, 3),
    ],
)
def test_compare_applies_ci_exit_code_after_compare_returns(
    monkeypatch, payload, expected_exit
) -> None:
    args = build_parser().parse_args(
        [
            "compare",
            "before.json",
            "after.json",
            "--max-step-time-regression-pct",
            "5",
        ]
    )
    events = []

    def compare_summaries(*_args, **_kwargs):
        events.append("compare_complete")
        return payload

    monkeypatch.setattr(
        "traceml_ai.reporting.compare.compare_summaries",
        compare_summaries,
    )

    if expected_exit is None:
        launcher_commands.run_compare(args)
    else:
        with pytest.raises(SystemExit) as exc_info:
            launcher_commands.run_compare(args)
        assert exc_info.value.code == expected_exit

    assert events == ["compare_complete"]


def test_compare_usage_error_is_distinct_from_regression_exit() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "compare",
                "before.json",
                "after.json",
                "--max-step-time-regression-pct",
            ]
        )

    assert exc_info.value.code == 2


def test_compare_reports_invalid_threshold_as_user_error(
    monkeypatch, capsys
) -> None:
    args = build_parser().parse_args(
        [
            "compare",
            "before.json",
            "after.json",
            "--max-step-time-regression-pct",
            "bad",
        ]
    )
    log_exception = Mock()
    monkeypatch.setattr(
        "traceml_ai.reporting.compare.compare_summaries",
        Mock(side_effect=RuntimeError("threshold must be finite")),
    )
    monkeypatch.setattr(
        launcher_commands, "_log_launcher_exception", log_exception
    )

    with pytest.raises(SystemExit) as exc_info:
        launcher_commands.run_compare(args)

    assert exc_info.value.code == 1
    assert (
        "[TraceML] ERROR: threshold must be finite" in capsys.readouterr().err
    )
    log_exception.assert_called_once()
    assert log_exception.call_args.args[0] == (
        "compare failed with a user-facing error"
    )


def test_training_output_help_describes_default_and_opt_out(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--help"])

    help_text = "".join(capsys.readouterr().out.split())
    assert exc_info.value.code == 0
    assert "--save-training-output" in help_text
    assert "--no-save-training-output" in help_text
    assert "Default:enabled" in help_text
    assert "--capture-stderr" not in help_text


@pytest.mark.parametrize(
    ("mode", "mirrors_live"),
    [("summary", True), ("dashboard", True), ("cli", False)],
)
def test_training_output_display_policy_saves_both_streams(
    monkeypatch, tmp_path, mode, mirrors_live
) -> None:
    stdout_terminal = io.BytesIO()
    stderr_terminal = io.BytesIO()
    terminals = iter((stdout_terminal, stderr_terminal))
    monkeypatch.setattr(
        launcher_commands, "_binary_stream", lambda _stream: next(terminals)
    )
    proc = Mock(
        stdout=io.BytesIO(b"training stdout\n"),
        stderr=io.BytesIO(b"training stderr\n"),
    )
    stdout_path = tmp_path / "training.stdout.log"
    stderr_path = tmp_path / "training.stderr.log"

    result = launcher_commands._start_training_output(
        proc,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        mode=mode,
    ).finish()

    assert result.warning is None
    assert stdout_path.read_bytes() == b"training stdout\n"
    assert stderr_path.read_bytes() == b"training stderr\n"
    expected_stdout = b"training stdout\n" if mirrors_live else b""
    expected_stderr = b"training stderr\n" if mirrors_live else b""
    assert stdout_terminal.getvalue() == expected_stdout
    assert stderr_terminal.getvalue() == expected_stderr


@pytest.mark.parametrize(
    ("mode", "mirrors_live"),
    [("summary", True), ("dashboard", True), ("cli", False)],
)
def test_aggregator_output_policy_saves_only_stderr(
    monkeypatch, tmp_path, mode, mirrors_live
) -> None:
    terminal = io.BytesIO()
    monkeypatch.setattr(
        launcher_commands, "_binary_stream", lambda _stream: terminal
    )
    proc = Mock(stderr=io.BytesIO(b"aggregator failure\n"))
    stderr_path = tmp_path / "aggregator" / "process.stderr.log"

    result = launcher_commands._start_aggregator_output(
        proc,
        stderr_path=stderr_path,
        mode=mode,
    ).finish()

    assert result.warning is None
    assert result.stdout_path is None
    assert stderr_path.read_bytes() == b"aggregator failure\n"
    expected = b"aggregator failure\n" if mirrors_live else b""
    assert terminal.getvalue() == expected


def test_aggregator_output_sink_failure_falls_back(
    monkeypatch, tmp_path
) -> None:
    terminal = io.BytesIO()
    monkeypatch.setattr(
        launcher_commands, "_binary_stream", lambda _stream: terminal
    )
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked", encoding="utf-8")
    proc = Mock(stderr=io.BytesIO(b"aggregator failure\n"))

    result = launcher_commands._start_aggregator_output(
        proc,
        stderr_path=blocked_parent / "process.stderr.log",
        mode="cli",
    ).finish()

    assert result.stderr_path is None
    assert terminal.getvalue() == b"aggregator failure\n"
    assert result.warning is not None
    assert "stderr output file could not be opened" in result.warning


def test_cli_failure_output_is_bounded_and_prints_confirmed_paths(
    tmp_path, capsys
) -> None:
    stderr_tail = "".join(f"stderr line {line}\n" for line in range(100))
    stdout_path = tmp_path / "training.stdout.log"
    stderr_path = tmp_path / "training.stderr.log"

    launcher_commands._print_training_output(
        ProcessOutputResult(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stderr_tail=stderr_tail.encode(),
            warning=None,
        ),
        mode="cli",
        training_failed=True,
    )

    stderr = capsys.readouterr().err
    assert "stderr line 59\n" not in stderr
    assert "stderr line 60\n" in stderr
    assert "stderr line 99\n" in stderr
    assert f"[TraceML] Stderr: {stderr_path}" in stderr
    assert f"[TraceML] Stdout: {stdout_path}" in stderr


@pytest.mark.parametrize("default", ["raise", "warn"])
def test_shared_missing_aggregator_resolver_uses_caller_default(
    monkeypatch, default
) -> None:
    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    assert resolve_on_missing_aggregator(None, default=default) == default

    monkeypatch.setenv("TRACEML_ON_MISSING_AGGREGATOR", "warn")
    assert resolve_on_missing_aggregator(None, default=default) == "warn"
    assert resolve_on_missing_aggregator("raise", default=default) == "raise"


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("default", ["raise", "warn"])
def test_shared_missing_aggregator_resolver_treats_blank_as_unset(
    monkeypatch, blank, default
) -> None:
    """An exported-but-empty variable means "not chosen", not "invalid"."""
    monkeypatch.setenv("TRACEML_ON_MISSING_AGGREGATOR", blank)
    assert resolve_on_missing_aggregator(None, default=default) == default

    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    assert resolve_on_missing_aggregator(blank, default=default) == default


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
    monkeypatch, tmp_path
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
    monkeypatch.chdir(tmp_path)
    start_aggregator = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(
        "traceml_ai.aggregator.aggregator_main.run_aggregator",
        start_aggregator,
    )
    from traceml_ai.launcher.commands import run_serve

    args = build_parser().parse_args(["serve", "--mode", "dashboard"])
    with pytest.raises(SystemExit) as excinfo:
        run_serve(args)

    message = str(excinfo.value)
    assert "nicegui" in message
    assert "Missing:" in message
    assert 'pip install "traceml-ai[dashboard]"' in message
    start_aggregator.assert_not_called()
    assert not (tmp_path / "logs").exists()


def test_serve_checks_resolved_dashboard_mode_before_start(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRACEML_UI_MODE", raising=False)
    monkeypatch.delenv("TRACEML_MODE", raising=False)
    (tmp_path / "traceml.yaml").write_text(
        "mode: dashboard\nlogs_dir: ./configured-logs\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "traceml_ai.launcher.commands.importlib.util.find_spec",
        lambda package: None if package == "nicegui" else object(),
    )
    start_aggregator = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(
        "traceml_ai.aggregator.aggregator_main.run_aggregator",
        start_aggregator,
    )
    from traceml_ai.launcher.commands import run_serve

    with pytest.raises(SystemExit) as excinfo:
        run_serve(build_parser().parse_args(["serve"]))

    assert 'pip install "traceml-ai[dashboard]"' in str(excinfo.value)
    start_aggregator.assert_not_called()
    assert not (tmp_path / "configured-logs").exists()


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
    monkeypatch, tmp_path, capsys
) -> None:
    import logging

    import traceml_ai.aggregator.aggregator_main as agg_main
    import traceml_ai.runtime.lifecycle as lifecycle
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

    traceml_logger = logging.getLogger("traceml_ai")
    saved_handlers = traceml_logger.handlers[:]
    saved_level = traceml_logger.level
    saved_propagate = traceml_logger.propagate
    traceml_logger.handlers.clear()

    # Do not clobber the process signal handlers, and stop before the blocking
    # wait by making aggregator startup raise a controlled error.
    monkeypatch.setattr(agg_main, "_install_signal_handlers", lambda ev: None)

    class _FailingAggregator:
        def start(self):
            raise RuntimeError("stop before blocking")

        def stop(self, timeout_sec):
            return None

    monkeypatch.setattr(
        lifecycle,
        "_build_aggregator",
        lambda **kwargs: _FailingAggregator(),
    )

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
        structured_log = (
            tmp_path / "serve-test" / "aggregator" / "traceml_errors.log"
        )
        raw_log = tmp_path / "serve-test" / "aggregator" / "process.stderr.log"
        content = structured_log.read_text(encoding="utf-8")
        assert content.count("Aggregator startup failed") == 1
        assert "Aggregator exiting due to error" not in content
        assert "RuntimeError: stop before blocking" in content
        assert not (tmp_path / "serve-test" / "aggregator_error.log").exists()
        stderr = capsys.readouterr().err
        assert str(structured_log) in stderr
        assert str(raw_log) in stderr
    finally:
        for handler in traceml_logger.handlers[:]:
            traceml_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in saved_handlers:
            traceml_logger.addHandler(handler)
        traceml_logger.setLevel(saved_level)
        traceml_logger.propagate = saved_propagate
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
    assert args.save_training_output
    assert args.args == ["--epochs", "1"]

    no_save_args = parser.parse_args(
        ["run", "train.py", "--no-save-training-output"]
    )
    assert not no_save_args.save_training_output

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


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_validate_rejects_non_finite_or_non_positive_finalize_timeout(
    value: str,
) -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", "--finalize-timeout-sec", value]
    )

    with pytest.raises(SystemExit, match="--finalize-timeout-sec"):
        validate_launch_args(args)


@pytest.mark.parametrize("flag", ["--master-port", "--aggregator-port"])
def test_validate_rejects_port_above_65535(flag: str) -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", flag, "65536"]
    )

    with pytest.raises(SystemExit, match=f"{flag} must be <= 65535"):
        validate_launch_args(args)


@pytest.mark.parametrize("flag", ["--master-port", "--aggregator-port"])
def test_validate_accepts_highest_port(flag: str) -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", flag, "65535"]
    )

    validate_launch_args(args)


@pytest.mark.parametrize("value", ["0", "-1", "65536", "70000"])
def test_validate_rejects_out_of_range_dashboard_port(value: str) -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", "--dashboard-port", value]
    )

    with pytest.raises(
        SystemExit, match="--dashboard-port must be an integer between"
    ):
        validate_launch_args(args)


def test_validate_accepts_highest_dashboard_port() -> None:
    args = build_parser().parse_args(
        ["run", "train.py", "--mode=cli", "--dashboard-port", "65535"]
    )

    validate_launch_args(args)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("65536", "--aggregator-port must be <= 65535"),
        ("0", "--aggregator-port must be >= 1"),
    ],
)
def test_serve_rejects_out_of_range_aggregator_port(
    monkeypatch, tmp_path, value: str, message: str
) -> None:
    monkeypatch.chdir(tmp_path)
    run_aggregator = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(
        "traceml_ai.aggregator.aggregator_main.run_aggregator",
        run_aggregator,
    )
    from traceml_ai.launcher.commands import run_serve

    args = build_parser().parse_args(["serve", "--aggregator-port", value])

    with pytest.raises(SystemExit, match=f"\\[TraceML\\] ERROR: {message}"):
        run_serve(args)

    run_aggregator.assert_not_called()


def test_run_rejects_out_of_range_dashboard_port_env_before_launch(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEML_DASHBOARD_PORT", "70000")
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    forbidden = _forbid_guard_launch_side_effects(monkeypatch)
    args = build_parser().parse_args(["run", str(script)])

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "TRACEML_DASHBOARD_PORT='70000'" in stderr
    assert "between 1 and 65535" in stderr
    for replacement in forbidden.values():
        replacement.assert_not_called()


def _write_guard_config(tmp_path: Path, guard: str) -> None:
    (tmp_path / "traceml.yaml").write_text(
        "mode: summary\nhistory_enabled: true\nguard:\n" + guard,
        encoding="utf-8",
    )


def _forbid_guard_launch_side_effects(monkeypatch) -> dict[str, Mock]:
    forbidden = {
        "write_run_manifest": Mock(side_effect=AssertionError("manifest")),
        "start_aggregator_process": Mock(
            side_effect=AssertionError("aggregator")
        ),
        "start_training_process": Mock(side_effect=AssertionError("training")),
    }
    for name, replacement in forbidden.items():
        monkeypatch.setattr(launcher_commands, name, replacement)
    return forbidden


def test_invalid_guard_stops_before_manifest_or_processes(
    monkeypatch, tmp_path, capsys
) -> None:
    project_dir = tmp_path / "project"
    launch_dir = project_dir / "runs" / "current"
    launch_dir.mkdir(parents=True)
    monkeypatch.chdir(launch_dir)
    script = launch_dir / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    _write_guard_config(
        project_dir,
        "  schema_version: 1\n"
        "  workload:\n"
        "    name: smoke\n"
        "  measurement:\n"
        "    start_step: 0\n"
        "    completed_steps: 5\n",
    )
    forbidden = _forbid_guard_launch_side_effects(monkeypatch)
    args = build_parser().parse_args(["run", str(script)])

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert (
        f"invalid measurement contract in {project_dir / 'traceml.yaml'}"
        in stderr
    )
    assert "guard.measurement.start_step" in stderr
    for replacement in forbidden.values():
        replacement.assert_not_called()


def test_run_rejects_duplicate_guard_before_manifest_or_processes(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    (tmp_path / "traceml.yaml").write_text(
        "mode: summary\n"
        "guard:\n"
        "  schema_version: 1\n"
        "  schema_version: 1\n",
        encoding="utf-8",
    )
    forbidden = _forbid_guard_launch_side_effects(monkeypatch)
    args = build_parser().parse_args(["run", str(script)])

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    assert "duplicate key in guard" in capsys.readouterr().err
    for replacement in forbidden.values():
        replacement.assert_not_called()


@pytest.mark.parametrize(
    ("config_prefix", "launch_args", "match"),
    [
        ("mode: summary\n", ["--mode", "cli"], "require mode=summary"),
        (
            "mode: summary\n",
            ["--mode", "summary", "--no-history"],
            "mode=summary requires history",
        ),
        (
            "mode: summary\n",
            ["--mode", "summary", "--trace-max-steps", "12"],
            "ends after --trace-max-steps",
        ),
    ],
)
def test_guard_rejects_unsupported_launch_before_processes(
    monkeypatch, tmp_path, config_prefix, launch_args, match
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    (tmp_path / "traceml.yaml").write_text(
        config_prefix
        + "guard:\n"
        + "  schema_version: 1\n"
        + "  workload:\n"
        + "    name: smoke\n"
        + "  measurement:\n"
        + "    start_step: 10\n"
        + "    completed_steps: 5\n",
        encoding="utf-8",
    )
    forbidden = _forbid_guard_launch_side_effects(monkeypatch)
    args = build_parser().parse_args(["run", str(script), *launch_args])

    with pytest.raises(SystemExit, match=match):
        launch_process(str(script), args)

    for replacement in forbidden.values():
        replacement.assert_not_called()


def test_guarded_root_refuses_an_existing_run_manifest_before_side_effects(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    _write_guard_config(
        tmp_path,
        "  schema_version: 1\n"
        "  workload:\n"
        "    name: smoke\n"
        "  measurement:\n"
        "    start_step: 1\n"
        "    completed_steps: 5\n",
    )
    manifest_path = tmp_path / "logs" / "guarded-run" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    original_manifest = '{"existing": true}\n'
    manifest_path.write_text(original_manifest, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--run-name",
            "guarded-run",
            "--logs-dir",
            str(tmp_path / "logs"),
        ]
    )
    forbidden = _forbid_guard_launch_side_effects(monkeypatch)
    setup_logger = Mock(side_effect=AssertionError("logger"))
    write_code_manifest = Mock(side_effect=AssertionError("code manifest"))
    monkeypatch.setattr(launcher_commands, "setup_error_logger", setup_logger)
    monkeypatch.setattr(
        launcher_commands, "write_code_manifest", write_code_manifest
    )

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    assert "guarded run name already exists" in capsys.readouterr().err
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    setup_logger.assert_not_called()
    write_code_manifest.assert_not_called()
    for replacement in forbidden.values():
        replacement.assert_not_called()


def test_guard_contract_is_captured_once_for_multi_node_run(
    monkeypatch, tmp_path, capsys
) -> None:
    project_dir = tmp_path / "project"
    launch_dir = project_dir / "runs" / "current"
    launch_dir.mkdir(parents=True)
    monkeypatch.chdir(launch_dir)
    script = launch_dir / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    config_path = project_dir / "traceml.yaml"
    _write_guard_config(
        project_dir,
        "  schema_version: 1\n"
        "  workload:\n"
        "    name: smoke\n"
        "    parameters:\n"
        "      precision: fp32\n"
        "  measurement:\n"
        "    start_step: 1\n"
        "    completed_steps: 5\n",
    )
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--nnodes",
            "2",
            "--nproc-per-node",
            "1",
            "--run-name",
            "guard-run",
            "--logs-dir",
            str(project_dir / "logs"),
        ]
    )

    write_manifest = Mock(return_value=project_dir / "manifest.json")
    start_training = Mock(side_effect=AssertionError("training"))

    def change_source_config(**_kwargs):
        _write_guard_config(
            project_dir,
            "  schema_version: 1\n"
            "  workload:\n"
            "    name: changed-after-capture\n"
            "  measurement:\n"
            "    start_step: 20\n"
            "    completed_steps: 2\n",
        )
        return None

    monkeypatch.setattr(launcher_commands, "setup_error_logger", Mock())
    monkeypatch.setattr(
        launcher_commands,
        "write_code_manifest",
        Mock(side_effect=change_source_config),
    )
    monkeypatch.setattr(
        launcher_commands, "write_run_manifest", write_manifest
    )
    monkeypatch.setattr(launcher_commands, "update_run_manifest", Mock())
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    monkeypatch.setattr(
        launcher_commands,
        "ensure_aggregator_port_free",
        Mock(
            side_effect=AggregatorPortInUseError(
                "port already in use", host="127.0.0.1"
            )
        ),
    )
    monkeypatch.setattr(
        launcher_commands,
        "start_aggregator_process",
        Mock(side_effect=AssertionError("aggregator")),
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", start_training
    )

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    assert write_manifest.call_args.kwargs["extra"]["guard"] == {
        "contract": {
            "schema_version": 1,
            "workload": {
                "name": "smoke",
                "parameters": {"precision": "fp32"},
            },
            "measurement": {"start_step": 1, "completed_steps": 5},
        }
    }
    assert write_manifest.call_args.kwargs["nnodes"] == 2
    assert write_manifest.call_args.kwargs["nproc_per_node"] == 1
    confirmation = (
        f"[TraceML] Measurement contract captured from {config_path}: "
        "workload=smoke, steps=1-5"
    )
    assert capsys.readouterr().err.count(confirmation) == 1
    start_training.assert_not_called()


@pytest.mark.parametrize("training_returncode", [0, 17, -signal.SIGTERM])
@pytest.mark.parametrize("recording_fails", [False, True])
def test_guarded_node_records_training_outcome_with_cli_exit_code(
    monkeypatch, tmp_path, capsys, recording_fails, training_returncode
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    _write_guard_config(
        tmp_path,
        "  schema_version: 1\n"
        "  workload:\n"
        "    name: multi-node-smoke\n"
        "  measurement:\n"
        "    start_step: 1\n"
        "    completed_steps: 5\n",
    )
    session_root = tmp_path / "logs" / "guarded-run"
    session_root.mkdir(parents=True)
    (session_root / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "guarded-run",
                "status": "running",
                "run": {"run_name": "guarded-run"},
                "created_at": "2026-09-26T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--nproc-per-node",
            "1",
            "--master-addr",
            "node-0.internal",
            "--run-name",
            "guarded-run",
            "--logs-dir",
            str(tmp_path / "logs"),
            "--no-save-training-output",
        ]
    )
    training = Mock(returncode=training_returncode)
    training.poll.return_value = training_returncode

    monkeypatch.setattr(launcher_commands, "setup_error_logger", Mock())
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=True)
    )
    monkeypatch.setattr(
        launcher_commands,
        "start_training_process",
        Mock(return_value=training),
    )
    monkeypatch.setattr(
        launcher_commands.TorchrunLaunchConfig,
        "to_command",
        Mock(return_value=[sys.executable]),
    )
    if recording_fails:
        monkeypatch.setattr(
            launcher_commands,
            "write_guard_outcome",
            Mock(side_effect=OSError("shared run directory unavailable")),
        )

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    expected_exit_code = TrainingOutcome(training_returncode).cli_exit_code
    assert exc.value.code == expected_exit_code
    outcome_path = session_root / "nodes" / "node_1" / "guard_outcome.json"
    if recording_fails:
        assert not outcome_path.exists()
        assert "failed to record the guarded training outcome" in (
            capsys.readouterr().err
        )
    else:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        assert outcome["node_rank"] == 1
        assert outcome["training"] == {
            "status": ("completed" if expected_exit_code == 0 else "failed"),
            "exit_code": expected_exit_code,
        }


@pytest.mark.parametrize(
    (
        "training_returncode",
        "recording_fails",
        "collection_fails",
        "expects_wait",
    ),
    [
        (0, False, False, True),
        (17, False, False, False),
        (0, True, False, False),
        (0, False, True, True),
    ],
)
def test_guarded_root_consolidates_node_outcomes_before_shutdown(
    monkeypatch,
    tmp_path,
    capsys,
    training_returncode,
    recording_fails,
    collection_fails,
    expects_wait,
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    _write_guard_config(
        tmp_path,
        "  schema_version: 1\n"
        "  workload:\n"
        "    name: multi-node-smoke\n"
        "  measurement:\n"
        "    start_step: 1\n"
        "    completed_steps: 5\n",
    )
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--nproc-per-node",
            "1",
            "--run-name",
            "guarded-run",
            "--logs-dir",
            str(tmp_path / "logs"),
            "--no-save-training-output",
        ]
    )
    aggregator = Mock(pid=10, returncode=0)
    training = Mock(returncode=training_returncode)
    training.poll.return_value = training_returncode
    aggregator_output = Mock()
    aggregator_output.finish.return_value = ProcessOutputResult(
        stdout_path=None,
        stderr_path=None,
        stderr_tail=b"",
        warning=None,
    )
    if recording_fails:
        collected_nodes = ()
        collected_reasons = ("node_outcome_missing",)
    elif training_returncode != 0:
        collected_nodes = (
            NodeTrainingOutcome(node_rank=0, exit_code=training_returncode),
        )
        collected_reasons = (
            "node_training_failed",
            "node_outcome_missing",
        )
    else:
        collected_nodes = (
            NodeTrainingOutcome(node_rank=0, exit_code=0),
            NodeTrainingOutcome(node_rank=1, exit_code=0),
        )
        collected_reasons = ()
    collected = GuardTrainingResult(
        status="completed" if not collected_reasons else "incomplete",
        nodes_expected=2,
        nodes_observed=len(collected_nodes),
        reasons=collected_reasons,
        nodes=collected_nodes,
    )
    events: list[str] = []
    collect = Mock(return_value=collected)
    if collection_fails:
        collect.side_effect = OSError("shared run directory unavailable")

    def start_training(**_kwargs):
        events.append("training_spawned")
        return training

    def stop_aggregator(*_args, **_kwargs) -> None:
        events.append("stop")

    def collect_outcomes(**_kwargs):
        events.append("collect")
        return collect(**_kwargs)

    def record_manifest_update(*_args, **kwargs) -> None:
        lifecycle = kwargs.get("extra", {}).get("lifecycle", {})
        if "training_started_at" in lifecycle:
            events.append("training_start")
        elif "training_ended_at" in lifecycle:
            assert kwargs["status"] == (
                "completed" if training_returncode == 0 else "failed"
            )
            events.append("training_end")
        guard = kwargs.get("extra", {}).get("guard", {})
        if "training" in guard:
            events.append("guard_training")

    update_manifest = Mock(side_effect=record_manifest_update)
    replacements = {
        "setup_error_logger": Mock(),
        "install_shutdown_handlers": Mock(),
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": update_manifest,
        "ensure_aggregator_port_free": Mock(),
        "start_aggregator_process": Mock(return_value=aggregator),
        "_start_aggregator_output": Mock(return_value=aggregator_output),
        "wait_for_tcp_listen": Mock(return_value=True),
        "start_training_process": Mock(side_effect=start_training),
        "terminate_process_group": Mock(side_effect=stop_aggregator),
        "write_guard_outcome": Mock(
            side_effect=(
                OSError("shared run directory unavailable")
                if recording_fails
                else None
            )
        ),
        "collect_guard_outcomes": Mock(side_effect=collect_outcomes),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(launcher_commands, name, replacement)

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == training_returncode
    assert events == [
        "training_spawned",
        "training_start",
        "training_end",
        "collect",
        "guard_training",
        "stop",
    ]
    collect_kwargs = collect.call_args.kwargs
    assert collect_kwargs["run_name"] == "guarded-run"
    assert collect_kwargs["nnodes"] == 2
    assert collect_kwargs["nproc_per_node"] == 1
    assert (collect_kwargs["timeout_s"] > 0) is expects_wait

    guard_update = next(
        call.kwargs
        for call in update_manifest.call_args_list
        if "training" in call.kwargs.get("extra", {}).get("guard", {})
    )
    training_result = guard_update["extra"]["guard"]["training"]
    stderr = capsys.readouterr().err
    assert ("Waiting up to" in stderr) is expects_wait
    if recording_fails:
        assert "failed to record the guarded training outcome" in stderr
    if collection_fails:
        assert training_result == {
            "status": "incomplete",
            "nodes_expected": 2,
            "nodes_observed": 0,
            "reasons": ["node_outcome_collection_failed"],
            "nodes": [],
        }
        assert "failed to collect guarded training outcomes" in stderr
    else:
        assert training_result == collected.to_dict()


def test_watch_ignores_guard_semantics(monkeypatch, tmp_path) -> None:
    for name in ("TRACEML_UI_MODE", "TRACEML_MODE", "TRACEML_LOGS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    (tmp_path / "traceml.yaml").write_text(
        "mode: cli\n"
        "logs_dir: ./my_logs\n"
        "guard:\n"
        "  workload:\n"
        "    name: first\n"
        "    name: second\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["watch", str(script)])

    write_manifest = Mock(return_value=tmp_path / "manifest.json")
    monkeypatch.setattr(launcher_commands, "setup_error_logger", Mock())
    monkeypatch.setattr(
        launcher_commands, "write_code_manifest", Mock(return_value=None)
    )
    monkeypatch.setattr(
        launcher_commands, "write_run_manifest", write_manifest
    )
    monkeypatch.setattr(launcher_commands, "update_run_manifest", Mock())
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    monkeypatch.setattr(
        launcher_commands,
        "ensure_aggregator_port_free",
        Mock(
            side_effect=AggregatorPortInUseError(
                "port already in use", host="127.0.0.1"
            )
        ),
    )
    monkeypatch.setattr(
        launcher_commands,
        "start_training_process",
        Mock(side_effect=AssertionError("training")),
    )

    with pytest.raises(SystemExit):
        launch_process(str(script), args)

    assert write_manifest.call_args.kwargs["ui_mode"] == "cli"
    assert write_manifest.call_args.kwargs["logs_dir"] == "./my_logs"
    assert "guard" not in write_manifest.call_args.kwargs["extra"]


def test_serve_ignores_guard_semantics(monkeypatch, tmp_path) -> None:
    for name in ("TRACEML_UI_MODE", "TRACEML_MODE", "TRACEML_LOGS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "traceml.yaml").write_text(
        "mode: cli\n"
        "logs_dir: ./my_logs\n"
        "guard:\n"
        "  workload:\n"
        "    name: first\n"
        "    name: second\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["serve"])

    settings = _resolve_serve_settings(args)

    assert settings.mode == "cli"
    assert settings.logs_dir == "./my_logs"


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
            "--no-save-training-output",
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

    def _start_training_process(train_cmd, env, cwd, *, capture_output=False):
        observed["train_cmd"] = train_cmd
        observed["env"] = env
        observed["cwd"] = cwd
        observed["capture_output"] = capture_output
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
    assert observed["capture_output"] is False
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
    ("train_rc", "aggregator_rc", "finalization_fails", "save_output"),
    [
        (0, 9, False, True),
        (0, 0, False, False),
        (1, 0, False, True),
        (0, 0, True, True),
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
    save_output,
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
    args.save_training_output = save_output

    aggregator = Mock(pid=10, returncode=aggregator_rc)
    training = Mock()
    if aggregator_rc == 9:
        training.poll.side_effect = [None, train_rc]
    else:
        training.poll.return_value = train_rc
    aggregator_stderr_path = (
        tmp_path
        / "logs"
        / "outcome-test"
        / "aggregator"
        / "process.stderr.log"
    )

    def start_aggregator(**_kwargs):
        # Written after node 0 reserved the run directory.
        aggregator_stderr_path.parent.mkdir(parents=True)
        aggregator_stderr_path.write_bytes(b"aggregator details\n")
        return aggregator

    aggregator_output_result = ProcessOutputResult(
        stdout_path=None,
        stderr_path=aggregator_stderr_path,
        stderr_tail=b"aggregator details\n",
        warning=(
            "aggregator output capture degraded" if not save_output else None
        ),
    )
    aggregator_output = Mock()
    training_output = Mock()
    training_output.finish.return_value = ProcessOutputResult(
        stdout_path=None,
        stderr_path=None,
        stderr_tail=b"captured failure\n",
        warning="output persistence unavailable",
    )
    events = []
    print_training_output = launcher_commands._print_training_output

    def stop_aggregator(*_args, **_kwargs):
        events.append("stop")

    def finish_aggregator_output():
        events.append("aggregator-output")
        return aggregator_output_result

    aggregator_output.finish.side_effect = finish_aggregator_output

    def print_output(*args, **kwargs):
        events.append("output")
        print_training_output(*args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONUNBUFFERED", "user-choice")

    def record_manifest_update(*_args, **kwargs):
        if kwargs.get("status") in {"completed", "failed"}:
            events.append("training-state")

    replacements = {
        "install_shutdown_handlers": Mock(),
        "ensure_aggregator_port_free": Mock(),
        "start_aggregator_process": Mock(side_effect=start_aggregator),
        "wait_for_tcp_listen": Mock(return_value=True),
        "start_training_process": Mock(return_value=training),
        "_start_aggregator_output": Mock(return_value=aggregator_output),
        "_start_training_output": Mock(return_value=training_output),
        "terminate_process_group": Mock(side_effect=stop_aggregator),
        "_print_training_output": Mock(side_effect=print_output),
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": Mock(side_effect=record_manifest_update),
        "_require_dashboard_dependencies": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(launcher_commands, name, replacement)
    monkeypatch.setattr(launcher_commands.time, "sleep", Mock())

    if finalization_fails:
        monkeypatch.setattr(
            launcher_commands,
            "collect_existing_artifacts",
            Mock(side_effect=OSError("manifest unavailable")),
        )

    with pytest.raises(SystemExit) as exc:
        launcher_commands.launch_process(str(script), args)

    assert exc.value.code == TrainingOutcome(train_rc).cli_exit_code
    assert (
        replacements["start_training_process"].call_args.kwargs[
            "capture_output"
        ]
        is save_output
    )
    assert (
        replacements["start_training_process"].call_args.kwargs["env"][
            "PYTHONUNBUFFERED"
        ]
        == "user-choice"
    )
    output_manifest = replacements["write_run_manifest"].call_args.kwargs[
        "extra"
    ]["training_output"]
    assert (
        "guard"
        not in replacements["write_run_manifest"].call_args.kwargs["extra"]
    )
    assert not list((tmp_path / "logs").rglob("guard_outcome.json"))
    assert output_manifest["enabled"] is save_output
    assert output_manifest["scope"] == "node"
    replacements["_start_aggregator_output"].assert_called_once_with(
        aggregator,
        stderr_path=aggregator_stderr_path,
        mode=mode,
    )
    aggregator_output.finish.assert_called_once_with()
    if aggregator_rc == 9:
        assert events.index("aggregator-output") < events.index("output")
    else:
        assert events.index("training-state") < events.index("stop")
        assert events.index("stop") < events.index("aggregator-output")
    if save_output:
        assert output_manifest["stdout_pattern"] == (
            "nodes/node_<node_rank>/training.stdout.log"
        )
        assert output_manifest["stderr_pattern"] == (
            "nodes/node_<node_rank>/training.stderr.log"
        )
        training_output.finish.assert_called_once_with()
    else:
        assert "stdout_pattern" not in output_manifest
        assert "stderr_pattern" not in output_manifest
        replacements["_start_training_output"].assert_not_called()
        training_output.finish.assert_not_called()
        assert not list((tmp_path / "logs").rglob("training.*.log"))
    stderr_lines = capsys.readouterr().err.splitlines()
    assert not any(
        "Measurement contract captured" in line for line in stderr_lines
    )
    if not save_output:
        assert (
            stderr_lines.count(
                "[TraceML] WARNING: aggregator output capture degraded"
            )
            == 1
        )
    final_line = stderr_lines[-1]
    expected_status = "completed successfully" if train_rc == 0 else "failed"
    assert f"Training {expected_status}" in final_line
    assert stderr_lines[-2].startswith("[TraceML] Telemetry ")
    if train_rc != 0:
        assert "torchrun exited with code 1" in final_line
    if mode == "cli" and aggregator_rc != 9:
        assert events.index("stop") < events.index("output")
        if train_rc != 0 and save_output:
            assert "captured failure" in "\n".join(stderr_lines)
    if aggregator_rc == 9:
        assert "aggregator exited early" in "\n".join(stderr_lines)
        assert any(
            call.kwargs.get("artifacts")
            == {"aggregator_stderr_log": str(aggregator_stderr_path)}
            for call in replacements["update_run_manifest"].call_args_list
        )
    training_state_update = next(
        call.kwargs
        for call in replacements["update_run_manifest"].call_args_list
        if call.kwargs.get("status") in {"completed", "failed"}
    )
    assert training_state_update["status"] == (
        "completed" if train_rc == 0 else "failed"
    )
    assert "training_ended_at" in training_state_update["extra"]["lifecycle"]
    if (
        train_rc == 0
        and aggregator_rc == 0
        and not finalization_fails
        and mode == "summary"
    ):
        final_update = replacements["update_run_manifest"].call_args_list[-1]
        assert "status" not in final_update.kwargs
        assert final_update.kwargs["telemetry_status"] == "failed"
        assert final_update.kwargs["telemetry_reason"] == "summary_missing"


@pytest.mark.parametrize(
    ("failure", "reason", "owner"),
    [
        ("spawn", "aggregator_spawn_failed", True),
        ("port_in_use", "aggregator_port_in_use", True),
        ("readiness", "aggregator_not_ready", True),
        ("readiness", "aggregator_not_ready", False),
    ],
)
def test_strict_aggregator_failure_does_not_start_training(
    monkeypatch, tmp_path, capsys, failure, reason, owner
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    cli = [
        "run",
        str(script),
        "--logs-dir",
        str(tmp_path / "logs"),
        "--aggregator-host",
        "telemetry.internal",
        "--aggregator-port",
        "43170",
    ]
    if not owner:
        cli.extend(
            ["--nnodes", "2", "--node-rank", "1", "--run-name", "strict-run"]
        )
    args = build_parser().parse_args(cli)
    start_training = Mock()
    update_manifest = Mock()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    aggregator = Mock(pid=10, returncode=7)
    aggregator.poll.return_value = 7
    aggregator_stderr_path = tmp_path / "aggregator" / "process.stderr.log"
    aggregator_output = Mock()
    aggregator_output.finish.return_value = ProcessOutputResult(
        stdout_path=None,
        stderr_path=aggregator_stderr_path,
        stderr_tail=b"failed before readiness\n",
        warning=None,
    )
    start_aggregator_output = Mock(return_value=aggregator_output)
    start_aggregator = Mock(return_value=aggregator)
    if failure == "spawn":
        start_aggregator.side_effect = FileNotFoundError("aggregator missing")
    port_check = Mock()
    if failure == "port_in_use":
        port_check.side_effect = AggregatorPortInUseError(
            "aggregator port 127.0.0.1:43170 is already in use.",
            host="127.0.0.1",
        )
    monkeypatch.setattr(
        launcher_commands, "ensure_aggregator_port_free", port_check
    )
    monkeypatch.setattr(
        launcher_commands, "start_aggregator_process", start_aggregator
    )
    monkeypatch.setattr(
        launcher_commands,
        "_start_aggregator_output",
        start_aggregator_output,
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
    stderr = capsys.readouterr().err
    if failure == "port_in_use":
        # The stale listener answered, so "not reachable" would be false.
        assert "not reachable" not in stderr
        assert (
            "aggregator port 127.0.0.1:43170 is already in use; training "
            "was not started" in stderr
        )
        assert stderr.count("aggregator port 127.0.0.1:43170") == 1
    else:
        assert (
            "aggregator was not reachable at telemetry.internal:43170"
            in stderr
        )
    assert "--on-missing-aggregator=warn" in stderr
    assert "TRACEML_ON_MISSING_AGGREGATOR=warn" in stderr
    if failure == "readiness" and owner:
        assert "(exit=7)" in stderr
        assert str(aggregator_stderr_path) in stderr
        assert stderr.index("Aggregator stderr") < stderr.index("ERROR")
        aggregator_output.finish.assert_called_once_with()
        assert update_manifest.call_args.kwargs["artifacts"] == {
            "aggregator_stderr_log": str(aggregator_stderr_path)
        }
    else:
        assert "(exit=" not in stderr
        start_aggregator_output.assert_not_called()
    start_training.assert_not_called()
    if failure == "port_in_use":
        assert "127.0.0.1:43170 is already in use" in stderr
        start_aggregator.assert_not_called()
    if owner:
        assert update_manifest.call_args.kwargs["status"] == "failed"
        assert (
            update_manifest.call_args.kwargs["telemetry_status"]
            == "unavailable"
        )
        assert update_manifest.call_args.kwargs["telemetry_reason"] == reason
    else:
        start_aggregator.assert_not_called()
        update_manifest.assert_not_called()
        # A non-root launcher never creates the run directory itself.
        assert not (tmp_path / "logs" / "strict-run").exists()


def test_stderr_from_process_exiting_before_readiness_is_exact(
    tmp_path,
) -> None:
    expected = b"pre-readiness failure: \xff\n"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; os.write(2, "
            "b'pre-readiness failure: \\xff\\n'); raise SystemExit(7)",
        ],
        stderr=subprocess.PIPE,
    )
    stderr_path = tmp_path / "aggregator" / "process.stderr.log"
    output = launcher_commands._start_aggregator_output(
        proc, stderr_path=stderr_path, mode="cli"
    )

    assert proc.wait(timeout=5) == 7
    assert not launcher_commands.wait_for_tcp_listen(
        host="127.0.0.1", port=0, proc=proc, timeout_sec=0.1
    )
    result = output.finish()

    assert stderr_path.read_bytes() == expected
    assert result.stderr_path == stderr_path.resolve()


def test_wildcard_port_conflict_names_the_conflicting_address(
    monkeypatch, tmp_path, capsys
) -> None:
    # The aggregator binds 0.0.0.0 but the stale listener sits on loopback;
    # the error must name the address that actually conflicted.
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--logs-dir",
            str(tmp_path / "logs"),
            "--aggregator-bind-host",
            "0.0.0.0",
            "--aggregator-port",
            "43170",
        ]
    )

    def _bind(host, port, backlog):
        if host == "127.0.0.1":
            raise OSError(errno.EADDRINUSE, "Address already in use")
        return Mock()

    monkeypatch.setattr(
        "traceml_ai.launcher.process.bind_exclusive_listener", _bind
    )
    start_aggregator = Mock()
    start_training = Mock()
    monkeypatch.chdir(tmp_path)
    for name, replacement in {
        "install_shutdown_handlers": Mock(),
        "start_aggregator_process": start_aggregator,
        "start_training_process": start_training,
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": Mock(),
    }.items():
        monkeypatch.setattr(launcher_commands, name, replacement)

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "aggregator port 127.0.0.1:43170 is already in use" in stderr
    assert "training was not started" in stderr
    assert stderr.count("aggregator port 127.0.0.1:43170") == 1
    assert "0.0.0.0:43170" not in stderr
    start_aggregator.assert_not_called()
    start_training.assert_not_called()


def test_aggregator_port_probe_rejects_a_live_listener() -> None:
    # A stale aggregator from a killed run still accepts connections, so the
    # readiness probe alone would mistake it for the new one.
    stale = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    stale.start()
    try:
        with pytest.raises(AggregatorPortInUseError) as excinfo:
            ensure_aggregator_port_free("127.0.0.1", stale.port)
        message = str(excinfo.value)
        assert f"127.0.0.1:{stale.port} is already in use" in message
        assert f"lsof -i :{stale.port}" in message
        assert "--aggregator-port" in message
    finally:
        stale.stop()


def test_wildcard_port_probe_rejects_a_loopback_listener() -> None:
    # BSD SO_REUSEADDR lets 0.0.0.0 bind beside a 127.0.0.1 listener, and
    # ranks connecting to 127.0.0.1 would then reach the stale listener.
    stale = bind_exclusive_listener("127.0.0.1", 0, backlog=1)
    try:
        port = stale.getsockname()[1]
        with pytest.raises(AggregatorPortInUseError):
            ensure_aggregator_port_free("0.0.0.0", port)
    finally:
        stale.close()


@pytest.mark.parametrize(
    ("host", "connect_host", "expected_hosts"),
    [
        ("0.0.0.0", "10.0.0.8", ["0.0.0.0", "127.0.0.1", "10.0.0.8"]),
        ("", "127.0.0.1", ["", "127.0.0.1"]),
        ("127.0.0.1", "10.0.0.8", ["127.0.0.1"]),
    ],
)
def test_port_probe_adds_worker_addresses_for_wildcard_hosts(
    monkeypatch, host, connect_host, expected_hosts
) -> None:
    # Linux already refuses the wildcard bind, so recording the probes proves
    # the worker-facing addresses are checked on every platform.
    calls = []
    probes = []

    def _record(probe_host, port, backlog):
        calls.append((probe_host, port))
        probes.append(Mock())
        return probes[-1]

    monkeypatch.setattr(
        "traceml_ai.launcher.process.bind_exclusive_listener", _record
    )

    ensure_aggregator_port_free(host, 43170, connect_host=connect_host)

    assert calls == [(probe_host, 43170) for probe_host in expected_hosts]
    for probe in probes:
        probe.close.assert_called_once_with()


def test_windows_access_denied_is_reported_as_port_in_use(monkeypatch) -> None:
    def _deny(_host, _port, backlog):
        del backlog
        raise OSError(10013, "Permission denied")

    monkeypatch.setattr("traceml_ai.launcher.process._IS_WINDOWS", True)
    monkeypatch.setattr(
        "traceml_ai.launcher.process.bind_exclusive_listener", _deny
    )

    with pytest.raises(AggregatorPortInUseError) as excinfo:
        ensure_aggregator_port_free("127.0.0.1", 43170)

    message = str(excinfo.value)
    assert "in use or reserved by Windows" in message
    assert "unreserved port" in message


def test_unusable_worker_probe_does_not_skip_later_addresses(
    monkeypatch,
) -> None:
    calls = []

    def _bind(host, _port, backlog):
        del backlog
        calls.append(host)
        if host == "127.0.0.1":
            raise OSError(errno.EADDRNOTAVAIL, "Address not available")
        if host == "10.0.0.8":
            raise OSError(errno.EADDRINUSE, "Address already in use")
        return Mock()

    monkeypatch.setattr(
        "traceml_ai.launcher.process.bind_exclusive_listener", _bind
    )

    with pytest.raises(AggregatorPortInUseError) as excinfo:
        ensure_aggregator_port_free("0.0.0.0", 43170, connect_host="10.0.0.8")

    assert excinfo.value.host == "10.0.0.8"
    assert calls == ["0.0.0.0", "127.0.0.1", "10.0.0.8"]


def test_aggregator_port_probe_releases_a_free_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    ensure_aggregator_port_free("127.0.0.1", port)
    ensure_aggregator_port_free("127.0.0.1", 0)

    # The probe must not keep the port it checked.
    server = TCPServer(TCPConfig(host="127.0.0.1", port=port))
    server.start()
    server.stop()


def _reserve_as_node_zero(run_root: Path, *, status: str = "running") -> None:
    """Stand in for node 0: reserve the run directory and write its manifest."""
    reserve_run_root(run_root)
    (run_root / "manifest.json").write_text(
        json.dumps({"status": status, "run": {"run_name": run_root.name}}),
        encoding="utf-8",
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else b"<dir>"
        )
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize(
    ("owner", "ready"),
    [(True, False), (False, False), (False, True)],
)
def test_launcher_scopes_telemetry_health_to_aggregator_owner(
    monkeypatch, tmp_path, capsys, owner, ready
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
    node_rank = 0 if owner else 1
    node_dir = tmp_path / "logs" / "warn-run" / "nodes" / f"node_{node_rank}"
    output_result = ProcessOutputResult(
        stdout_path=node_dir / "training.stdout.log",
        stderr_path=node_dir / "training.stderr.log",
        stderr_tail=b"",
        warning=None,
    )
    training_output = Mock()
    training_output.finish.return_value = output_result
    aggregator_stderr_path = (
        tmp_path / "logs" / "warn-run" / "aggregator" / "process.stderr.log"
    )
    aggregator_output = Mock()

    def _finish_aggregator_output():
        events.append("finish-output")
        return ProcessOutputResult(
            stdout_path=None,
            stderr_path=aggregator_stderr_path,
            stderr_tail=b"startup failure\n",
            warning=None,
        )

    aggregator_output.finish.side_effect = _finish_aggregator_output
    start_aggregator_output = Mock(return_value=aggregator_output)

    def _stop_aggregator(*_args, **_kwargs):
        events.append("stop")

    def _start_training(*, train_cmd, env, cwd, capture_output):
        events.append("train")
        assert env["TRACEML_DISABLED"] == ("0" if ready else "1")
        assert capture_output is True
        return training

    monkeypatch.chdir(tmp_path)
    if not owner:
        _reserve_as_node_zero(tmp_path / "logs" / "warn-run")
    setup_error_logger = Mock()
    monkeypatch.setattr(
        launcher_commands, "setup_error_logger", setup_error_logger
    )
    install_shutdown_handlers = Mock()
    monkeypatch.setattr(
        launcher_commands,
        "install_shutdown_handlers",
        install_shutdown_handlers,
    )
    start_aggregator = Mock(return_value=aggregator)
    monkeypatch.setattr(
        launcher_commands, "ensure_aggregator_port_free", Mock()
    )
    monkeypatch.setattr(
        launcher_commands, "start_aggregator_process", start_aggregator
    )
    monkeypatch.setattr(
        launcher_commands,
        "_start_aggregator_output",
        start_aggregator_output,
    )
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=ready)
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", _start_training
    )
    monkeypatch.setattr(
        launcher_commands,
        "_start_training_output",
        Mock(return_value=training_output),
    )
    monkeypatch.setattr(
        launcher_commands, "terminate_process_group", _stop_aggregator
    )
    write_code_manifest = Mock(return_value=None)
    monkeypatch.setattr(
        launcher_commands, "write_code_manifest", write_code_manifest
    )
    write_manifest = Mock(return_value=tmp_path / "manifest.json")
    update_manifest = Mock()
    monkeypatch.setattr(
        launcher_commands, "write_run_manifest", write_manifest
    )
    monkeypatch.setattr(
        launcher_commands, "update_run_manifest", update_manifest
    )

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), args)

    assert exc.value.code == 0
    stderr = capsys.readouterr().err
    if owner and not ready:
        assert "[TraceML] ERROR:" not in stderr
        assert "training will continue with TraceML disabled" in stderr
        assert "traceml.summary()" in stderr
    setup_error_logger.assert_called_once_with(
        role="launcher",
        session_root=(tmp_path / "logs" / "warn-run").resolve(),
        node_rank=node_rank,
    )
    launcher_commands._start_training_output.assert_called_once_with(
        training,
        stdout_path=(
            tmp_path.resolve()
            / "logs"
            / "warn-run"
            / "nodes"
            / f"node_{node_rank}"
            / "training.stdout.log"
        ),
        stderr_path=(
            tmp_path.resolve()
            / "logs"
            / "warn-run"
            / "nodes"
            / f"node_{node_rank}"
            / "training.stderr.log"
        ),
        mode="cli",
    )
    training_output.finish.assert_called_once_with()
    if owner:
        assert events == ["stop", "finish-output", "train"]
        start_aggregator.assert_called_once()
        start_aggregator_output.assert_called_once_with(
            aggregator,
            stderr_path=aggregator_stderr_path,
            mode="cli",
        )
        aggregator_output.finish.assert_called_once_with()
        write_code_manifest.assert_called_once()
        write_manifest.assert_called_once()
        assert install_shutdown_handlers.call_args.kwargs["manifest_path"] == (
            tmp_path / "manifest.json"
        )
        initial_telemetry = write_manifest.call_args.kwargs["telemetry_status"]
        reported_statuses = [
            call.kwargs.get("telemetry_status")
            for call in update_manifest.call_args_list
        ]
        assert initial_telemetry == "starting"
        assert "unavailable" in reported_statuses
        assert any(
            call.kwargs.get("artifacts")
            == {"aggregator_stderr_log": str(aggregator_stderr_path)}
            for call in update_manifest.call_args_list
        )

        # Signal teardown invokes this callback after marking the run
        # interrupted. It must then merge the paths confirmed by the drainers.
        updates_before_cleanup = update_manifest.call_count
        install_shutdown_handlers.call_args.kwargs["cleanup"]()
        assert update_manifest.call_count == updates_before_cleanup + 1
        assert update_manifest.call_args.kwargs["artifacts"] == {
            "aggregator_stderr_log": str(aggregator_stderr_path),
            "training_stdout_log": str(output_result.stdout_path),
            "training_stderr_log": str(output_result.stderr_path),
        }
    else:
        assert events == ["train"]
        start_aggregator.assert_not_called()
        start_aggregator_output.assert_not_called()
        write_code_manifest.assert_not_called()
        write_manifest.assert_not_called()
        update_manifest.assert_not_called()
        assert (
            install_shutdown_handlers.call_args.kwargs["manifest_path"] is None
        )


def _node_args(script: Path, logs_dir: Path, run_name: str, node_rank: int):
    return build_parser().parse_args(
        [
            "run",
            str(script),
            "--logs-dir",
            str(logs_dir),
            "--run-name",
            run_name,
            "--nnodes",
            "2",
            "--node-rank",
            str(node_rank),
        ]
    )


@pytest.mark.parametrize("flag", ["--run-name", "--session-id"])
def test_existing_run_directory_is_left_untouched(
    monkeypatch, tmp_path, flag
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    run_root = tmp_path / "logs" / "taken"
    (run_root / "aggregator").mkdir(parents=True)
    (run_root / "aggregator" / "telemetry").write_bytes(b"old telemetry")
    (run_root / "manifest.json").write_text(
        '{"status": "completed"}', encoding="utf-8"
    )
    before = _tree(run_root)
    forbidden = {
        name: Mock(side_effect=AssertionError(name))
        for name in (
            "setup_error_logger",
            "write_code_manifest",
            "write_run_manifest",
            "ensure_aggregator_port_free",
            "start_aggregator_process",
            "start_training_process",
        )
    }
    for name, replacement in forbidden.items():
        monkeypatch.setattr(launcher_commands, name, replacement)
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--logs-dir",
            str(tmp_path / "logs"),
            flag,
            "taken",
        ]
    )

    with pytest.raises(SystemExit, match="run directory already exists"):
        launch_process(str(script), args)

    assert _tree(run_root) == before
    for replacement in forbidden.values():
        replacement.assert_not_called()


def test_non_root_launcher_needs_node_zero_reservation(
    monkeypatch, tmp_path
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    start_training = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=True)
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", start_training
    )

    with pytest.raises(SystemExit, match="node 0 has not reserved"):
        launch_process(
            str(script), _node_args(script, tmp_path / "logs", "free", 1)
        )

    assert not (tmp_path / "logs").exists()
    start_training.assert_not_called()


@pytest.mark.parametrize(
    ("status", "node_one_exists"),
    [("completed", False), ("running", True)],
)
def test_non_root_launcher_refuses_an_earlier_run(
    monkeypatch, tmp_path, status, node_one_exists
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    run_root = tmp_path / "logs" / "old-run"
    _reserve_as_node_zero(run_root, status=status)
    if node_one_exists:
        (run_root / "nodes" / "node_1").mkdir()
    before = _tree(run_root)
    start_training = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_commands, "install_shutdown_handlers", Mock())
    monkeypatch.setattr(
        launcher_commands, "wait_for_tcp_listen", Mock(return_value=True)
    )
    monkeypatch.setattr(
        launcher_commands, "start_training_process", start_training
    )

    with pytest.raises(SystemExit, match="belongs to an earlier launch"):
        launch_process(
            str(script), _node_args(script, tmp_path / "logs", "old-run", 1)
        )

    assert _tree(run_root) == before
    start_training.assert_not_called()


def test_multinode_nodes_share_one_run_directory(
    monkeypatch, tmp_path
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    node_one_exit = []

    def start_training(*, train_cmd, env, cwd, capture_output):
        if env["TRACEML_NODE_RANK"] == "0":
            # Node 1 joins while node 0 is running.
            with pytest.raises(SystemExit) as exc:
                launch_process(
                    str(script), _node_args(script, logs_dir, "shared", 1)
                )
            node_one_exit.append(exc.value.code)
        return Mock(**{"poll.return_value": 0})

    def output_drainer(*_args, **_kwargs):
        return Mock(
            **{
                "finish.return_value": ProcessOutputResult(
                    stdout_path=None,
                    stderr_path=None,
                    stderr_tail=b"",
                    warning=None,
                )
            }
        )

    replacements = {
        "install_shutdown_handlers": Mock(),
        "setup_error_logger": Mock(),
        "write_code_manifest": Mock(return_value=None),
        "ensure_aggregator_port_free": Mock(),
        "start_aggregator_process": Mock(
            return_value=Mock(
                pid=10, returncode=0, **{"poll.return_value": None}
            )
        ),
        "wait_for_tcp_listen": Mock(return_value=True),
        "start_training_process": start_training,
        "_start_aggregator_output": output_drainer,
        "_start_training_output": output_drainer,
        "terminate_process_group": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(launcher_commands, name, replacement)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        launch_process(str(script), _node_args(script, logs_dir, "shared", 0))

    assert exc.value.code == 0
    assert node_one_exit == [0]
    run_root = logs_dir / "shared"
    assert (run_root / "nodes" / "node_0").is_dir()
    assert (run_root / "nodes" / "node_1").is_dir()
    manifest = json.loads((run_root / "manifest.json").read_text("utf-8"))
    assert manifest["run"]["run_name"] == "shared"
    assert manifest["launch"]["node_rank"] == 0


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
            {
                "telemetry_available": False,
                "startup_reason": "aggregator_port_in_use",
            },
            ("unavailable", "aggregator_port_in_use", None),
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
            {
                "aggregator_exited_early": True,
                "aggregator_exit_code": 4,
                "summary_required": True,
                "summary_exists": False,
            },
            ("failed", "aggregator_exited_early", 4),
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

    with pytest.raises(SystemExit) as excinfo:
        validate_launch_args(args)

    assert 'pip install "traceml-ai[dashboard]"' in str(excinfo.value)


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


@pytest.mark.parametrize(
    ("initial", "expected"),
    [
        (None, None),
        ("starting", "degraded"),
        ("running", "degraded"),
        ("unavailable", "unavailable"),
        ("degraded", "degraded"),
        ("failed", "failed"),
        ("complete", "complete"),
    ],
)
def test_interruption_terminalizes_only_transient_telemetry(
    tmp_path, initial, expected
) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {"status": "running"}
    if initial is not None:
        payload["telemetry_status"] = initial
    if initial in {"starting", "running"}:
        payload["telemetry_reason"] = "stale"
        payload["aggregator_exit_code"] = 7
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    update_run_manifest(manifest_path, status="interrupted")

    interrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert interrupted["status"] == "interrupted"
    if expected is None:
        assert "telemetry_status" not in interrupted
    else:
        assert interrupted["telemetry_status"] == expected
    if initial in {"starting", "running"}:
        assert "telemetry_reason" not in interrupted
        assert "aggregator_exit_code" not in interrupted


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


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (None, False),
        ("{malformed", False),
        ({"generated_at": "2026-08-31T09:59:59+00:00"}, False),
        ({"generated_at": "2026-08-31T10:00:01+00:00"}, True),
    ],
)
def test_summary_evidence_requires_current_generation(
    tmp_path, contents, expected
) -> None:
    summary_path = tmp_path / "final_summary.json"
    if isinstance(contents, dict):
        summary_path.write_text(json.dumps(contents), encoding="utf-8")
    elif contents is not None:
        summary_path.write_text(contents, encoding="utf-8")

    assert (
        is_current_summary_artifact(
            summary_path,
            aggregator_started_at="2026-08-31T10:00:00+00:00",
        )
        is expected
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


def test_node_artifact_directories_are_node_scoped(tmp_path) -> None:
    node_0 = node_artifact_dir(tmp_path, 0)
    node_1 = node_artifact_dir(tmp_path, 1)

    assert node_0 == tmp_path.resolve() / "nodes" / "node_0"
    assert node_1 == tmp_path.resolve() / "nodes" / "node_1"
    assert node_0 != node_1

    node_0.mkdir(parents=True)
    node_1.mkdir(parents=True)
    node_0_stderr = node_0 / "training.stderr.log"
    node_1_stderr = node_1 / "training.stderr.log"
    node_0_stderr.write_text("node 0", encoding="utf-8")
    node_1_stderr.write_text("node 1", encoding="utf-8")

    assert node_0_stderr.read_text(encoding="utf-8") == "node 0"
    assert node_1_stderr.read_text(encoding="utf-8") == "node 1"


def test_collect_existing_artifacts_includes_confirmed_training_output(
    tmp_path,
) -> None:
    db_path = tmp_path / "aggregator" / "telemetry"
    node_dir = node_artifact_dir(tmp_path, 0)
    stdout_path = node_dir / "training.stdout.log"
    stderr_path = node_dir / "training.stderr.log"
    aggregator_stderr_path = tmp_path / "aggregator" / "process.stderr.log"
    stderr_path.parent.mkdir(parents=True)
    aggregator_stderr_path.parent.mkdir(parents=True)
    stdout_path.write_bytes(b"training output\n")
    stderr_path.write_bytes(b"native crash details\n")
    aggregator_stderr_path.write_bytes(b"aggregator details\n")

    artifacts = collect_existing_artifacts(
        db_path,
        session_root=tmp_path,
        training_stdout_path=stdout_path,
        training_stderr_path=stderr_path,
        aggregator_stderr_path=aggregator_stderr_path,
    )

    assert artifacts["training_stdout_log"] == str(stdout_path)
    assert artifacts["training_stderr_log"] == str(stderr_path)
    assert artifacts["aggregator_stderr_log"] == str(aggregator_stderr_path)


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


@pytest.mark.parametrize(
    ("extra", "enforced"),
    [
        ([], False),
        (["--run-name", "demo"], True),
        (["--session-id", "x"], True),
    ],
)
def test_serve_enforces_session_id_only_when_explicit(
    monkeypatch, tmp_path, extra, enforced
) -> None:
    """A plain `python train.py` generates its own id and must be admitted."""
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["serve", *extra])

    settings = _resolve_serve_settings(args)

    assert settings.enforce_session_id is enforced
    assert settings.run_nonce == ""
    assert settings.admit_generated_session_id is True


@pytest.mark.parametrize("nnodes", [1, 2])
def test_launcher_shares_run_nonce_only_on_single_node(
    monkeypatch, tmp_path, nnodes
) -> None:
    """Other nodes' launchers cannot know the owner node's nonce."""
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--logs-dir",
            str(tmp_path / "logs"),
            "--run-name",
            "nonce-run",
            "--on-missing-aggregator",
            "warn",
            "--nnodes",
            str(nnodes),
        ]
    )
    monkeypatch.chdir(tmp_path)
    # A stale value inherited from the parent shell must not leak through.
    monkeypatch.setenv("TRACEML_RUN_NONCE", "inherited")
    envs = {}

    def _start_aggregator(*, env, cwd):
        envs["aggregator"] = dict(env)
        return Mock(pid=10, returncode=7, **{"poll.return_value": 7})

    def _start_training(*, train_cmd, env, cwd, capture_output):
        envs["training"] = dict(env)
        return Mock(**{"poll.return_value": 0})

    output = ProcessOutputResult(
        stdout_path=None, stderr_path=None, stderr_tail=b"", warning=None
    )
    drainer = Mock(**{"finish.return_value": output})
    for name, value in {
        "setup_error_logger": Mock(),
        "install_shutdown_handlers": Mock(),
        "ensure_aggregator_port_free": Mock(),
        "start_aggregator_process": _start_aggregator,
        "_start_aggregator_output": Mock(return_value=drainer),
        "wait_for_tcp_listen": Mock(return_value=False),
        "start_training_process": _start_training,
        "_start_training_output": Mock(return_value=drainer),
        "terminate_process_group": Mock(),
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": Mock(),
    }.items():
        monkeypatch.setattr(launcher_commands, name, value)

    with pytest.raises(SystemExit):
        launch_process(str(script), args)

    nonce = envs["aggregator"]["TRACEML_RUN_NONCE"]
    assert envs["training"]["TRACEML_RUN_NONCE"] == nonce
    if nnodes == 1:
        assert len(nonce) >= 16 and nonce != "inherited"
    else:
        assert nonce == ""


_FOREIGN_WARNING = (
    "[TraceML] WARNING: ignored 3 payload(s) from another TraceML run: "
    "host=h pid=1 session=run-a (3). A training process from an earlier run "
    "may still be running; stop it so it no longer sends telemetry to this "
    "run."
)


def _aggregator_output(tmp_path, *, in_file: bool, in_tail: bool):
    path = None
    if in_file:
        path = tmp_path / "process.stderr.log"
        path.write_text(
            f"noise\n{_FOREIGN_WARNING}\n[TraceML] Aggregator stopped.\n",
            encoding="utf-8",
        )
    tail = f"{_FOREIGN_WARNING}\n".encode() if in_tail else b"late noise\n"
    return ProcessOutputResult(
        stdout_path=None, stderr_path=path, stderr_tail=tail, warning=None
    )


@pytest.mark.parametrize(
    ("in_file", "in_tail"),
    [
        # The bounded tail can lose the line; the persisted file cannot.
        (True, False),
        (False, True),
    ],
)
def test_cli_mode_repeats_the_foreign_run_warning(
    tmp_path, capsys, in_file, in_tail
) -> None:
    """cli mode does not mirror aggregator stderr while rendering."""
    result = _aggregator_output(tmp_path, in_file=in_file, in_tail=in_tail)

    launcher_commands._print_foreign_run_warnings(result, mode="cli")

    assert capsys.readouterr().err.splitlines() == [_FOREIGN_WARNING]


@pytest.mark.parametrize("mode", ["summary", "dashboard"])
def test_mirrored_modes_do_not_repeat_the_foreign_run_warning(
    tmp_path, capsys, mode
) -> None:
    result = _aggregator_output(tmp_path, in_file=True, in_tail=True)

    launcher_commands._print_foreign_run_warnings(result, mode=mode)

    assert capsys.readouterr().err == ""


def test_launcher_prints_the_foreign_run_warning_in_cli_mode(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run",
            str(script),
            "--mode",
            "cli",
            "--logs-dir",
            str(tmp_path / "logs"),
            "--on-missing-aggregator",
            "warn",
        ]
    )
    monkeypatch.chdir(tmp_path)
    output = _aggregator_output(tmp_path, in_file=True, in_tail=False)
    drainer = Mock(**{"finish.return_value": output})
    for name, value in {
        "setup_error_logger": Mock(),
        "install_shutdown_handlers": Mock(),
        "ensure_aggregator_port_free": Mock(),
        "start_aggregator_process": Mock(
            return_value=Mock(pid=10, returncode=7, **{"poll.return_value": 7})
        ),
        "_start_aggregator_output": Mock(return_value=drainer),
        "wait_for_tcp_listen": Mock(return_value=False),
        "start_training_process": Mock(
            return_value=Mock(**{"poll.return_value": 0})
        ),
        "_start_training_output": Mock(return_value=drainer),
        "terminate_process_group": Mock(),
        "write_code_manifest": Mock(return_value=None),
        "write_run_manifest": Mock(return_value=tmp_path / "manifest.json"),
        "update_run_manifest": Mock(),
    }.items():
        monkeypatch.setattr(launcher_commands, name, value)

    with pytest.raises(SystemExit):
        launch_process(str(script), args)

    err = capsys.readouterr().err.splitlines()
    assert err.count(_FOREIGN_WARNING) == 1
