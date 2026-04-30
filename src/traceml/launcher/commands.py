"""Command handlers for the TraceML launcher CLI."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from traceml.launcher.manifest import (
    collect_existing_artifacts,
    update_run_manifest,
    write_code_manifest,
    write_run_manifest,
)
from traceml.launcher.process import (
    DEFAULT_SHUTDOWN_TIMEOUT_SEC,
    DEFAULT_TCP_READY_TIMEOUT_SEC,
    build_torchrun_base_cmd,
    install_shutdown_handlers,
    start_aggregator_process,
    start_training_process,
    terminate_process_group,
    wait_for_tcp_listen,
)
from traceml.runtime.launch_context import LaunchContext
from traceml.runtime.session import get_session_id
from traceml.utils.msgpack_codec import Decoder as MsgpackDecoder


def _log_launcher_exception(message: str, exc: Exception) -> None:
    """Log launcher failures when the shared error logger is available."""
    try:
        from traceml.loggers.error_log import get_error_logger

        get_error_logger("TraceMLLauncher").exception("[TraceML] %s", message)
    except Exception:
        pass


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
    if getattr(args, "mode", None) == "summary" and getattr(
        args, "no_history", False
    ):
        raise SystemExit(
            "[TraceML] ERROR: --mode=summary requires history. "
            "Remove --no-history to enable final summary generation."
        )


def launch_process(script_path: str, args: argparse.Namespace) -> None:
    """Launch the TraceML aggregator and target training process."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    env["TRACEML_DISABLED"] = (
        "1" if getattr(args, "disable_traceml", False) else "0"
    )
    env["TRACEML_PROFILE"] = getattr(args, "profile", "watch")
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_UI_MODE"] = args.mode
    env["TRACEML_INTERVAL"] = str(args.interval)
    env["TRACEML_ENABLE_LOGGING"] = "1" if args.enable_logging else "0"
    env["TRACEML_LOGS_DIR"] = args.logs_dir
    env["TRACEML_NUM_DISPLAY_LAYERS"] = str(args.num_display_layers)
    env["TRACEML_SESSION_ID"] = (
        args.session_id if args.session_id else get_session_id()
    )
    env["TRACEML_TCP_HOST"] = args.tcp_host
    env["TRACEML_TCP_PORT"] = str(args.tcp_port)
    env["TRACEML_REMOTE_MAX_ROWS"] = str(args.remote_max_rows)
    env["TRACEML_NPROC_PER_NODE"] = str(args.nproc_per_node)
    env["TRACEML_HISTORY_ENABLED"] = "0" if args.no_history else "1"

    launch_context = LaunchContext.capture()
    env.update(launch_context.to_env())
    execution_cwd = launch_context.launch_cwd

    session_id = env["TRACEML_SESSION_ID"]
    session_root = Path(args.logs_dir).resolve() / session_id
    aggregator_dir = session_root / "aggregator"
    db_path = aggregator_dir / "telemetry"

    code_manifest_path = write_code_manifest(
        session_root=session_root,
        script_path=script_path,
    )

    manifest_path = write_run_manifest(
        session_root=session_root,
        session_id=session_id,
        script_path=script_path,
        profile=env["TRACEML_PROFILE"],
        ui_mode=args.mode,
        logs_dir=args.logs_dir,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        nproc_per_node=args.nproc_per_node,
        history_enabled=not args.no_history,
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
        train_cmd = [
            *build_torchrun_base_cmd(args.nproc_per_node),
            str(script_path),
            *script_args,
        ]
        train_proc = start_training_process(
            train_cmd=train_cmd,
            env=env,
            cwd=execution_cwd,
        )
        install_shutdown_handlers(
            lambda: (train_proc, None), manifest_path=manifest_path
        )
        train_proc.wait()
        final_status = "completed" if train_proc.returncode == 0 else "failed"
        update_run_manifest(manifest_path, status=final_status)
        raise SystemExit(train_proc.returncode)

    supported_modes = {"cli", "dashboard", "summary"}
    if args.mode not in supported_modes:
        raise ValueError(
            f"Invalid display mode '{args.mode}'. "
            f"Supported modes: {sorted(supported_modes)}"
        )

    train_cmd = [
        *build_torchrun_base_cmd(args.nproc_per_node),
        runner_path,
        "--",
        *script_args,
    ]

    agg_proc: Optional[subprocess.Popen] = None
    train_proc: Optional[subprocess.Popen] = None

    install_shutdown_handlers(
        lambda: (train_proc, agg_proc), manifest_path=manifest_path
    )

    print(
        f"[TraceML] Starting aggregator on {args.tcp_host}:{args.tcp_port} "
        f"(ui={args.mode}, profile={env['TRACEML_PROFILE']})"
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
        host=args.tcp_host,
        port=int(args.tcp_port),
        proc=agg_proc,
        timeout_sec=DEFAULT_TCP_READY_TIMEOUT_SEC,
    )
    if not ready:
        rc = agg_proc.poll()
        print(
            f"[TraceML] ERROR: aggregator failed to start or bind on "
            f"{args.tcp_host}:{args.tcp_port} (exit={rc}). See output above for details.",
            file=sys.stderr,
        )
        terminate_process_group(agg_proc, timeout_sec=3.0)
        update_run_manifest(manifest_path, status="failed")
        raise SystemExit(1)

    print("[TraceML] Aggregator ready.")
    update_run_manifest(manifest_path, status="running")

    train_proc = start_training_process(
        train_cmd=train_cmd,
        env=env,
        cwd=execution_cwd,
    )

    while True:
        train_rc = train_proc.poll()
        if train_rc is not None:
            if agg_proc is not None:
                print(
                    "[TraceML] Training finished; stopping aggregator...",
                    file=sys.stderr,
                )
                terminate_process_group(
                    agg_proc, timeout_sec=DEFAULT_SHUTDOWN_TIMEOUT_SEC
                )

            final_status = "completed" if train_rc == 0 else "failed"
            update_run_manifest(
                manifest_path,
                status=final_status,
                artifacts=collect_existing_artifacts(
                    db_path, session_root=session_root
                ),
            )
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
        from traceml.reporting.compare import compare_summaries

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
