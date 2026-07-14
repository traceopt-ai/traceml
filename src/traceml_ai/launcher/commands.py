# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Command handlers for the TraceML launcher CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from traceml_ai.launcher.launch_config import (
    DistributedLaunchConfig,
    RunIdentity,
)
from traceml_ai.launcher.manifest import (
    collect_existing_artifacts,
    update_run_manifest,
    write_code_manifest,
    write_run_manifest,
)
from traceml_ai.launcher.process import (
    DEFAULT_SHUTDOWN_TIMEOUT_SEC,
    DEFAULT_TCP_READY_TIMEOUT_SEC,
    StderrTailCapture,
    install_shutdown_handlers,
    start_aggregator_process,
    start_stderr_tail_capture,
    start_training_process,
    terminate_process_group,
    wait_for_tcp_listen,
)
from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.runtime.launch_context import LaunchContext
from traceml_ai.runtime.session import get_session_id
from traceml_ai.runtime.settings import DEFAULT_FINALIZE_TIMEOUT_SEC
from traceml_ai.utils.msgpack_codec import Decoder as MsgpackDecoder

DASHBOARD_DEPENDENCY_INSTALL_HINT = (
    "Dashboard mode requires nicegui and plotly. They are included in the "
    "default TraceML install; if they are missing, run "
    "`pip install -U traceml-ai` or `pip install nicegui plotly`."
)

SINGLE_NODE_DEFAULT_MODE = "dashboard"
MULTI_NODE_DEFAULT_MODE = "summary"


def _launch_defaults_for_topology(
    defaults: Mapping[str, Any],
    *,
    nnodes: int,
) -> dict[str, Any]:
    """Return built-in launch defaults adjusted for the requested topology."""
    resolved = dict(defaults)
    resolved["mode"] = (
        MULTI_NODE_DEFAULT_MODE
        if int(nnodes) > 1
        else SINGLE_NODE_DEFAULT_MODE
    )
    return resolved


def _require_dashboard_dependencies(mode: str) -> None:
    """Validate dashboard dependencies when dashboard mode is active."""
    if mode != "dashboard":
        return

    missing = [
        package
        for package in ("nicegui", "plotly")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise SystemExit(
            "[TraceML] ERROR: "
            f"{DASHBOARD_DEPENDENCY_INSTALL_HINT} "
            f"Missing: {', '.join(missing)}."
        )


def _dashboard_url(dashboard_port: int) -> str:
    """Return the local browser URL for the dashboard."""
    return f"http://127.0.0.1:{int(dashboard_port)}"


def _dashboard_ssh_tunnel_command(dashboard_port: int) -> str:
    """Return an SSH tunnel template for viewing a remote dashboard locally."""
    port = int(dashboard_port)
    return f"ssh -L {port}:127.0.0.1:{port} user@remote-host"


def _boxed_message(title: str, lines: list[str]) -> str:
    """Format a high-visibility ASCII box for important launch messages."""
    content = [title, *lines]
    width = max(len(line) for line in content)
    border = "+" + "-" * (width + 2) + "+"
    rows = [border]
    rows.extend(f"| {line.ljust(width)} |" for line in content)
    rows.append(border)
    return "\n".join(rows)


def _dashboard_access_box(dashboard_port: int) -> str:
    """Return the dashboard access instructions shown after launch."""
    url = _dashboard_url(dashboard_port)
    ssh_cmd = _dashboard_ssh_tunnel_command(dashboard_port)
    return _boxed_message(
        "TraceML dashboard",
        [
            f"Open locally: {url}",
            f"Remote SSH tunnel: {ssh_cmd}",
            "Then open the local URL above in your browser.",
        ],
    )


def _log_launcher_exception(message: str, exc: Exception) -> None:
    """Log launcher failures when the shared error logger is available."""
    try:
        from traceml_ai.loggers.error_log import get_error_logger

        get_error_logger("TraceMLLauncher").exception("[TraceML] %s", message)
    except Exception:
        pass


def _stderr_capture_enabled(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> bool:
    """Resolve the opt-in stderr capture flag and environment variable."""
    return bool(getattr(args, "capture_stderr", False)) or (
        environ.get("TRACEML_CAPTURE_STDERR") == "1"
    )


def _finish_stderr_capture(
    capture: Optional[StderrTailCapture], session_root: Path
) -> None:
    """Persist an optional stderr tail without affecting the training result."""
    if capture is None:
        return
    capture.finish(session_root / "crash_stderr.log")


def resolve_existing_script_path(script_path: str) -> str:
    """Resolve and validate the target training script path."""
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not path.is_file():
        raise IsADirectoryError(f"Script path is not a file: {script_path}")
    return str(path.resolve())


def validate_launch_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints for TraceML launch commands."""
    try:
        launch_cfg = DistributedLaunchConfig.from_args(args)
        RunIdentity.from_args(
            args,
            generated_session_id="validation_placeholder",
            require_explicit=launch_cfg.torchrun.nnodes > 1,
        )
    except ValueError as exc:
        raise SystemExit(f"[TraceML] ERROR: {exc}") from exc

    explicit_mode = getattr(args, "mode", None)
    if explicit_mode is not None:
        _require_dashboard_dependencies(str(explicit_mode))

    if explicit_mode == "summary" and getattr(args, "no_history", False):
        raise SystemExit(
            "[TraceML] ERROR: --mode=summary requires history. "
            "Remove --no-history to enable final summary generation."
        )
    if getattr(args, "html_report", False) and getattr(
        args, "no_history", False
    ):
        raise SystemExit(
            "[TraceML] ERROR: --html-report requires history. "
            "Remove --no-history to enable HTML report generation."
        )
    if int(getattr(args, "summary_window_rows", 1)) <= 0:
        raise SystemExit(
            "[TraceML] ERROR: --summary-window-rows must be greater than 0."
        )
    finalize_timeout_sec = getattr(args, "finalize_timeout_sec", None)
    if finalize_timeout_sec is not None and float(finalize_timeout_sec) <= 0.0:
        raise SystemExit(
            "[TraceML] ERROR: --finalize-timeout-sec must be greater than 0."
        )
    trace_max_steps = getattr(args, "trace_max_steps", None)
    if trace_max_steps is not None and int(trace_max_steps) <= 0:
        raise SystemExit(
            "[TraceML] ERROR: --trace-max-steps must be greater than 0."
        )


def launch_process(script_path: str, args: argparse.Namespace) -> None:
    """Launch the TraceML aggregator and target training process."""
    from traceml_ai.config.yaml_loader import (
        BUILT_IN_DEFAULTS,
        find_config_file,
        load_yaml_config,
        resolve_config,
    )

    launch_context = LaunchContext.capture()

    config_path = find_config_file(Path(launch_context.launch_cwd))
    try:
        yaml_cfg = (
            load_yaml_config(config_path) if config_path is not None else {}
        )
    except (ValueError, OSError) as exc:
        print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Normalize the deprecated TRACEML_MODE env var to TRACEML_UI_MODE so that
    # resolve_config sees it. Matches the fallback logic in executor and aggregator.
    launcher_env: Mapping[str, str] = os.environ
    if "TRACEML_UI_MODE" not in os.environ and "TRACEML_MODE" in os.environ:
        launcher_env = {
            **os.environ,
            "TRACEML_UI_MODE": os.environ["TRACEML_MODE"],
        }

    launch_cfg = DistributedLaunchConfig.from_args(args)
    torchrun_cfg = launch_cfg.torchrun
    aggregator_cfg = launch_cfg.aggregator
    launch_defaults = _launch_defaults_for_topology(
        BUILT_IN_DEFAULTS,
        nnodes=torchrun_cfg.nnodes,
    )

    # None = flag not supplied; resolver falls through to env/yaml/default.
    # --no-history inverts history_enabled: True flag → False override, absent → None.
    # Distributed/identity settings (nproc, nnodes, master/aggregator address,
    # run name) are owned by the typed launch configs below, not traceml.yaml.
    cli_overrides = {
        "mode": args.mode,
        "interval": args.interval,
        "enable_logging": args.enable_logging,
        "logs_dir": args.logs_dir,
        "history_enabled": (False if args.no_history else None),
        "finalize_timeout_sec": args.finalize_timeout_sec,
        "dashboard_port": args.dashboard_port,
        "dashboard_auto_open": (
            False if args.no_dashboard_auto_open else None
        ),
    }

    cfg = resolve_config(
        cli_overrides=cli_overrides,
        parent_env=launcher_env,
        yaml_config=yaml_cfg,
        defaults=launch_defaults,
    )
    cfg["finalize_timeout_sec"] = float(
        cfg.get("finalize_timeout_sec") or DEFAULT_FINALIZE_TIMEOUT_SEC
    )

    # Cross-field validation after all sources are merged.
    if cfg["mode"] == "summary" and not cfg["history_enabled"]:
        raise SystemExit(
            "[TraceML] ERROR: mode=summary requires history to be enabled. "
            "Remove --no-history (or set history_enabled: true in traceml.yaml) "
            "to enable final summary generation."
        )

    supported_modes = {"cli", "dashboard", "summary"}
    if cfg["mode"] not in supported_modes:
        raise SystemExit(
            f"[TraceML] ERROR: invalid mode '{cfg['mode']}'. "
            f"Valid modes: {sorted(supported_modes)}"
        )
    _require_dashboard_dependencies(str(cfg["mode"]))

    owns_aggregator = aggregator_cfg.is_owner(node_rank=torchrun_cfg.node_rank)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    env["TRACEML_DISABLED"] = (
        "1" if getattr(args, "disable_traceml", False) else "0"
    )
    capture_stderr = _stderr_capture_enabled(args, launcher_env)
    env["TRACEML_CAPTURE_STDERR"] = "1" if capture_stderr else "0"
    env["TRACEML_PROFILE"] = getattr(args, "profile", "watch")
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_UI_MODE"] = cfg["mode"]
    env["TRACEML_INTERVAL"] = str(cfg["interval"])
    env["TRACEML_ENABLE_LOGGING"] = "1" if cfg["enable_logging"] else "0"
    env["TRACEML_LOGS_DIR"] = cfg["logs_dir"]
    run_identity = RunIdentity.from_args(
        args,
        generated_session_id=get_session_id(),
        require_explicit=torchrun_cfg.nnodes > 1,
    )
    env["TRACEML_SESSION_ID"] = run_identity.session_id
    env["TRACEML_AGGREGATOR_HOST"] = aggregator_cfg.connect_host
    env["TRACEML_AGGREGATOR_BIND_HOST"] = aggregator_cfg.bind_host
    env["TRACEML_AGGREGATOR_PORT"] = str(aggregator_cfg.port)
    env["TRACEML_DASHBOARD_PORT"] = str(cfg["dashboard_port"])
    env["TRACEML_DASHBOARD_AUTO_OPEN"] = (
        "1" if cfg["dashboard_auto_open"] else "0"
    )
    env["TRACEML_SUMMARY_WINDOW_ROWS"] = str(
        int(getattr(args, "summary_window_rows", DEFAULT_SUMMARY_WINDOW_ROWS))
    )
    env["TRACEML_FINALIZE_TIMEOUT_SEC"] = str(
        float(cfg["finalize_timeout_sec"])
    )
    trace_max_steps = getattr(args, "trace_max_steps", None)
    env["TRACEML_TRACE_MAX_STEPS"] = (
        "" if trace_max_steps is None else str(int(trace_max_steps))
    )
    env["TRACEML_NNODES"] = str(torchrun_cfg.nnodes)
    env["TRACEML_NPROC_PER_NODE"] = str(torchrun_cfg.nproc_per_node)
    env["TRACEML_EXPECTED_WORLD_SIZE"] = str(
        int(torchrun_cfg.nnodes) * int(torchrun_cfg.nproc_per_node)
    )
    env["TRACEML_NODE_RANK"] = str(torchrun_cfg.node_rank)
    env["TRACEML_MASTER_ADDR"] = torchrun_cfg.master_addr
    env["TRACEML_MASTER_PORT"] = str(torchrun_cfg.master_port)
    env["TRACEML_HISTORY_ENABLED"] = "1" if cfg["history_enabled"] else "0"
    env["TRACEML_HTML_REPORT"] = (
        "1" if getattr(args, "html_report", False) else "0"
    )
    env["NODE_RANK"] = str(torchrun_cfg.node_rank)

    env.update(launch_context.to_env())
    execution_cwd = launch_context.launch_cwd

    session_id = env["TRACEML_SESSION_ID"]
    session_root = Path(cfg["logs_dir"]).resolve() / session_id
    aggregator_dir = session_root / "aggregator"
    db_path = aggregator_dir / "telemetry"

    code_manifest_path = write_code_manifest(
        session_root=session_root,
        script_path=script_path,
    )

    manifest_path = write_run_manifest(
        session_root=session_root,
        session_id=session_id,
        run=run_identity.to_manifest(),
        script_path=script_path,
        profile=env["TRACEML_PROFILE"],
        ui_mode=cfg["mode"],
        logs_dir=cfg["logs_dir"],
        aggregator_host=aggregator_cfg.connect_host,
        aggregator_bind_host=aggregator_cfg.bind_host,
        aggregator_port=aggregator_cfg.port,
        nnodes=torchrun_cfg.nnodes,
        node_rank=torchrun_cfg.node_rank,
        master_addr=torchrun_cfg.master_addr,
        master_port=torchrun_cfg.master_port,
        nproc_per_node=torchrun_cfg.nproc_per_node,
        history_enabled=cfg["history_enabled"],
        summary_window_rows=int(env["TRACEML_SUMMARY_WINDOW_ROWS"]),
        finalize_timeout_sec=float(env["TRACEML_FINALIZE_TIMEOUT_SEC"]),
        status="starting",
        launch_cwd=execution_cwd,
        aggregator_dir=aggregator_dir,
        db_path=db_path,
        extra=(
            {"artifacts": {"code_manifest": str(code_manifest_path)}}
            if code_manifest_path is not None
            else None
        ),
    )

    traceml_root = Path(__file__).resolve().parents[1]
    runner_path = str(traceml_root / "runtime" / "executor.py")
    script_args = args.args or []

    if env["TRACEML_DISABLED"] == "1":
        print(
            "[TraceML] TraceML is disabled via --disable-traceml. Running natively."
        )
        if getattr(args, "html_report", False):
            print(
                "[TraceML] --html-report ignored: TraceML is disabled.",
                file=sys.stderr,
            )
        train_cmd = [
            *torchrun_cfg.to_command(),
            str(script_path),
            *script_args,
        ]
        train_proc = start_training_process(
            train_cmd=train_cmd,
            env=env,
            cwd=execution_cwd,
            capture_stderr=capture_stderr,
        )
        stderr_capture = (
            start_stderr_tail_capture(train_proc) if capture_stderr else None
        )
        install_shutdown_handlers(
            lambda: (train_proc, None), manifest_path=manifest_path
        )
        train_proc.wait()
        _finish_stderr_capture(stderr_capture, session_root)
        final_status = "completed" if train_proc.returncode == 0 else "failed"
        update_run_manifest(
            manifest_path,
            status=final_status,
            artifacts=collect_existing_artifacts(
                db_path, session_root=session_root
            ),
        )
        raise SystemExit(train_proc.returncode)

    train_cmd = [
        *torchrun_cfg.to_command(),
        runner_path,
        "--",
        *script_args,
    ]

    agg_proc: Optional[subprocess.Popen] = None
    train_proc: Optional[subprocess.Popen] = None

    install_shutdown_handlers(
        lambda: (train_proc, agg_proc), manifest_path=manifest_path
    )

    if owns_aggregator:
        print(
            "[TraceML] Starting aggregator on "
            f"{aggregator_cfg.bind_host}:{aggregator_cfg.port} "
            f"(connect={aggregator_cfg.connect_host}, "
            f"ui={cfg['mode']}, profile={env['TRACEML_PROFILE']})"
        )
        try:
            agg_proc = start_aggregator_process(env=env, cwd=execution_cwd)
        except FileNotFoundError as exc:
            _log_launcher_exception("aggregator entrypoint was not found", exc)
            print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
            update_run_manifest(manifest_path, status="failed")
            raise SystemExit(1)

        print(f"[TraceML] Aggregator PID: {agg_proc.pid}")

        ready = wait_for_tcp_listen(
            host=aggregator_cfg.connect_host,
            port=aggregator_cfg.port,
            proc=agg_proc,
            timeout_sec=DEFAULT_TCP_READY_TIMEOUT_SEC,
        )
    else:
        print(
            "[TraceML] Waiting for aggregator on "
            f"{aggregator_cfg.connect_host}:{aggregator_cfg.port} "
            f"(node_rank={torchrun_cfg.node_rank})"
        )
        ready = wait_for_tcp_listen(
            host=aggregator_cfg.connect_host,
            port=aggregator_cfg.port,
            timeout_sec=DEFAULT_TCP_READY_TIMEOUT_SEC,
        )

    if not ready:
        rc = agg_proc.poll() if agg_proc is not None else None
        print(
            "[TraceML] ERROR: aggregator was not reachable at "
            f"{aggregator_cfg.connect_host}:{aggregator_cfg.port} "
            f"(exit={rc}). See output above for details.",
            file=sys.stderr,
        )
        if agg_proc is not None:
            terminate_process_group(agg_proc, timeout_sec=3.0)
        update_run_manifest(manifest_path, status="failed")
        raise SystemExit(1)

    print("[TraceML] Aggregator ready.")
    update_run_manifest(manifest_path, status="running")

    train_proc = start_training_process(
        train_cmd=train_cmd,
        env=env,
        cwd=execution_cwd,
        capture_stderr=capture_stderr,
    )
    stderr_capture = (
        start_stderr_tail_capture(train_proc) if capture_stderr else None
    )
    if owns_aggregator and cfg["mode"] == "dashboard":
        print(_dashboard_access_box(int(cfg["dashboard_port"])))

    while True:
        train_rc = train_proc.poll()
        if train_rc is not None:
            _finish_stderr_capture(stderr_capture, session_root)
            if agg_proc is not None:
                print(
                    "[TraceML] Training finished; stopping aggregator...",
                    file=sys.stderr,
                )
                terminate_process_group(
                    agg_proc,
                    timeout_sec=(
                        float(env["TRACEML_FINALIZE_TIMEOUT_SEC"])
                        + DEFAULT_SHUTDOWN_TIMEOUT_SEC
                    ),
                )

            final_status = "completed" if train_rc == 0 else "failed"
            update_run_manifest(
                manifest_path,
                status=final_status,
                artifacts=collect_existing_artifacts(
                    db_path, session_root=session_root
                ),
            )
            if (
                train_rc == 0
                and owns_aggregator
                and cfg["mode"] == "summary"
                and cfg["history_enabled"]
                and not (session_root / "final_summary.json").is_file()
            ):
                print(
                    "[TraceML] ERROR: training finished successfully, but "
                    "TraceML did not produce final_summary.json.",
                    file=sys.stderr,
                )
                update_run_manifest(manifest_path, status="failed")
                raise SystemExit(1)
            if agg_proc is not None and agg_proc.returncode not in (0, None):
                if train_rc == 0:
                    raise SystemExit(int(agg_proc.returncode))
            raise SystemExit(train_rc)

        if agg_proc is not None and agg_proc.poll() is not None:
            agg_rc = agg_proc.returncode
            print(
                f"[TraceML] WARNING: aggregator exited early (code={agg_rc}). "
                "Training will continue without TraceML telemetry.",
                file=sys.stderr,
            )
            update_run_manifest(
                manifest_path,
                extra={
                    "telemetry_status": "degraded",
                    "aggregator_exited_early": True,
                    "aggregator_exit_code": agg_rc,
                },
            )
            agg_proc = None

        time.sleep(1.0)


def run_with_tracing(args: argparse.Namespace, profile: str) -> None:
    """Run a script with TraceML enabled for a selected telemetry profile."""
    args.profile = profile
    try:
        script_path = resolve_existing_script_path(args.script)
    except (FileNotFoundError, IsADirectoryError) as exc:
        print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    launch_process(script_path=script_path, args=args)


def run_inspect(args: argparse.Namespace) -> None:
    """Decode and print binary msgpack logs for debugging."""
    path = Path(args.file)
    if not path.exists() or not path.is_file():
        print(f"[TraceML] ERROR: file not found: {args.file}", file=sys.stderr)
        raise SystemExit(1)

    decoder = MsgpackDecoder()
    with open(path, "rb") as f:
        try:
            while True:
                header = f.read(4)
                if not header:
                    break
                if len(header) < 4:
                    print(
                        "[TraceML] WARNING: truncated frame header",
                        file=sys.stderr,
                    )
                    break
                length = struct.unpack("!I", header)[0]
                payload = f.read(length)
                if len(payload) < length:
                    print(
                        "[TraceML] WARNING: truncated frame payload",
                        file=sys.stderr,
                    )
                    break
                record = decoder.decode(payload)
                print(json.dumps(record, indent=2))
        except Exception as exc:
            _log_launcher_exception(f"inspect failed for {path}", exc)
            print(
                f"[TraceML] ERROR: decoding failed for {path.name}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1)


def run_compare(args: argparse.Namespace) -> None:
    """Compare two TraceML final summary JSON files."""
    try:
        from traceml_ai.reporting.compare import compare_summaries

        compare_summaries(
            args.left,
            args.right,
            output=args.output,
            print_to_stdout=True,
        )
    except RuntimeError as exc:
        _log_launcher_exception("compare failed with a user-facing error", exc)
        print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        _log_launcher_exception("compare failed unexpectedly", exc)
        print(f"[TraceML] ERROR: compare failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


def run_view(args: argparse.Namespace) -> None:
    """Print the stored text from a TraceML summary JSON file.

    With ``--html`` (``args.html`` is not None), render an HTML report from
    the JSON instead of printing text; an empty string means the default
    ``<summary>.html`` output path.
    """
    html_out = getattr(args, "html", None)
    try:
        if html_out is not None:
            from traceml_ai.reporting.html import render_html_report_from_file

            written = render_html_report_from_file(
                args.summary, html_out or None
            )
            print(f"[TraceML] Wrote HTML report: {written}")
            return

        from traceml_ai.reporting.view import view_summary

        view_summary(args.summary, print_to_stdout=True)
    except RuntimeError as exc:
        print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        _log_launcher_exception("view failed unexpectedly", exc)
        print(f"[TraceML] ERROR: view failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
