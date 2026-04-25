import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import msgspec

from traceml.runtime.launch_context import LaunchContext
from traceml.runtime.session import get_session_id
from traceml.utils.ast_analysis import analyze_script, build_code_manifest

DEFAULT_TCP_READY_TIMEOUT_SEC = 15.0
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 5.0
INTERRUPTED_EXIT_CODE = 130


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written files on interruption.

    The file is first written to a temporary file in the same directory and then
    replaced atomically.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def _load_json_or_warn(path: Path) -> Dict[str, Any]:
    """Load JSON from disk.

    Returns an empty dict if the file does not exist. If the file exists but is
    unreadable or malformed, preserves the original file as a `.corrupt` copy
    and returns an empty dict.
    """
    path = Path(path).resolve()

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        corrupt_path = path.with_suffix(path.suffix + ".corrupt")
        try:
            corrupt_path.write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except Exception:
            pass
        print(
            f"[TraceML] WARNING: manifest is malformed and will be rebuilt: {path} ({exc})",
            file=sys.stderr,
        )
        return {}
    except OSError as exc:
        print(
            f"[TraceML] WARNING: manifest could not be read and will be rebuilt: {path} ({exc})",
            file=sys.stderr,
        )
        return {}


def _resolve_existing_script_path(script_path: str) -> str:
    """Resolve and validate the target training script path.

    Raises:
        FileNotFoundError: if the path does not exist.
        IsADirectoryError: if the path exists but is not a file.
    """
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not path.is_file():
        raise IsADirectoryError(f"Script path is not a file: {script_path}")
    return str(path.resolve())


def _build_torchrun_base_cmd(nproc_per_node: int) -> list[str]:
    """Build a torchrun command using the current Python interpreter."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={int(nproc_per_node)}",
    ]


def _collect_existing_artifacts(
    db_path: Path,
    session_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Return only artifacts that exist on disk."""
    candidates = {
        "db": db_path,
        "summary_card_json": Path(str(db_path) + ".summary_card.json"),
        "summary_card_txt": Path(str(db_path) + ".summary_card.txt"),
    }
    if session_root is not None:
        candidates["code_manifest"] = (
            Path(session_root).resolve() / "code_manifest.json"
        )

    return {
        name: str(path) for name, path in candidates.items() if path.exists()
    }


def _validate_launch_args(args: argparse.Namespace) -> None:
    """
    Validate cross-argument constraints for TraceML launch commands.

    Summary mode depends on SQLite-backed history because the final end-of-run
    summary is generated from the persisted session database. We reject invalid
    combinations early in the CLI so runtime and aggregator code can stay
    simple and mode-agnostic.
    """
    if getattr(args, "mode", None) == "summary" and getattr(
        args, "no_history", False
    ):
        raise SystemExit(
            "[TraceML] ERROR: --mode=summary requires history. "
            "Remove --no-history to enable final summary generation."
        )


def write_code_manifest(
    session_root: Path,
    script_path: str,
) -> Optional[Path]:
    """Write a separate static-analysis manifest under the session directory.

    This helper must never break the CLI flow. If AST analysis fails, it writes
    a minimal fallback manifest when possible and otherwise returns ``None``.
    """
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "code_manifest.json"

    try:
        findings = analyze_script(str(Path(script_path).resolve()))
        manifest = build_code_manifest(findings)
        manifest["analysis_status"] = (
            "ok" if not findings.parse_errors else "partial"
        )
        _write_json_atomic(manifest_path, manifest)
        return manifest_path
    except Exception as exc:
        fallback: Dict[str, Any] = {
            "schema_version": 1,
            "script_path": str(Path(script_path).resolve()),
            "generated_at": _utc_now_iso(),
            "analysis_status": "failed",
            "parse_errors": [f"Static analysis failed: {exc}"],
        }
        try:
            _write_json_atomic(manifest_path, fallback)
            return manifest_path
        except Exception:
            return None


def write_run_manifest(
    session_root: Path,
    session_id: str,
    script_path: str,
    profile: str,
    ui_mode: str,
    logs_dir: str,
    tcp_host: str,
    tcp_port: int,
    nproc_per_node: int,
    history_enabled: bool,
    status: str,
    launch_cwd: str,
    aggregator_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write or overwrite the run manifest under ``session_root``.

    Parameters
    ----------
    status:
        Expected values include ``starting``, ``running``, ``completed``,
        ``failed``, and ``interrupted``.
    extra:
        Optional non-reserved fields to merge into the top-level manifest.
    """
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "session_id": str(session_id),
        "status": str(status),
        "created_at": _utc_now_iso(),
        "host": {"hostname": socket.gethostname()},
        "launch": {
            "script_path": str(Path(script_path).resolve()),
            "profile": str(profile),
            "ui_mode": str(ui_mode),
            "logs_dir": str(Path(logs_dir).resolve()),
            "tcp_host": str(tcp_host),
            "tcp_port": int(tcp_port),
            "nproc_per_node": int(nproc_per_node),
            "history_enabled": bool(history_enabled),
            "launch_cwd": str(Path(launch_cwd).resolve()),
        },
        "paths": {
            "session_root": str(session_root),
            "aggregator_dir": (
                str(aggregator_dir.resolve()) if aggregator_dir else None
            ),
            "db_path": str(db_path.resolve()) if db_path else None,
        },
        "artifacts": {},
    }

    if extra:
        for key, value in extra.items():
            if key == "artifacts" and isinstance(value, dict):
                manifest.setdefault("artifacts", {}).update(value)
            else:
                manifest[key] = value

    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def update_run_manifest(
    manifest_path: Path,
    *,
    status: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Update an existing manifest in place using an atomic rewrite."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json_or_warn(manifest_path)

    if status is not None:
        manifest["status"] = str(status)

    manifest["updated_at"] = _utc_now_iso()

    if artifacts:
        manifest.setdefault("artifacts", {}).update(artifacts)

    if extra:
        manifest.update(extra)

    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def terminate_process_group(
    proc: Optional[subprocess.Popen],
    timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
) -> None:
    """Best-effort termination for a subprocess started with ``start_new_session=True``.

    Termination order:
    1. SIGTERM the whole process group
    2. Wait up to ``timeout_sec``
    3. SIGKILL if still alive
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=timeout_sec)
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_for_tcp_listen(
    host: str,
    port: int,
    proc: subprocess.Popen,
    timeout_sec: float = DEFAULT_TCP_READY_TIMEOUT_SEC,
    poll_interval_sec: float = 0.05,
) -> bool:
    """Wait until ``(host, port)`` starts accepting TCP connections.

    Returns ``False`` early if ``proc`` exits while waiting.
    """
    deadline = time.time() + float(timeout_sec)
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, int(port)), timeout=0.25):
                return True
        except Exception as exc:
            last_err = exc
            time.sleep(float(poll_interval_sec))

    if last_err is not None:
        print(
            f"[TraceML] Aggregator did not become ready on {host}:{port} "
            f"(last error: {last_err})",
            file=sys.stderr,
        )
    return False


def install_shutdown_handlers(
    get_procs: Callable[[], Iterable[Optional[subprocess.Popen]]],
    manifest_path: Optional[Path] = None,
) -> None:
    """Install SIGINT/SIGTERM handlers that terminate child process groups.

    Child processes are started with ``start_new_session=True`` and therefore do
    not receive Ctrl+C automatically from the parent shell.
    """
    already_handled = {"value": False}

    def _handler(signum: int, _frame: Any) -> None:
        if already_handled["value"]:
            raise SystemExit(INTERRUPTED_EXIT_CODE)
        already_handled["value"] = True

        print(
            f"\n[TraceML] Signal {signum} received — terminating processes…",
            file=sys.stderr,
        )

        if manifest_path is not None:
            try:
                update_run_manifest(manifest_path, status="interrupted")
            except Exception:
                pass

        for proc in get_procs():
            terminate_process_group(
                proc, timeout_sec=DEFAULT_SHUTDOWN_TIMEOUT_SEC
            )

        raise SystemExit(INTERRUPTED_EXIT_CODE)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def start_aggregator_process(
    env: Dict[str, str], cwd: str
) -> subprocess.Popen:
    """Start the TraceML aggregator as a separate process.

    The subprocess cwd is set explicitly so all child processes inherit a
    deterministic working directory rather than depending on ambient shell
    state.
    """
    aggregator_path = (
        Path(__file__).parent / "aggregator" / "aggregator_main.py"
    )
    if not aggregator_path.exists():
        raise FileNotFoundError(
            f"Aggregator entrypoint not found: {aggregator_path}"
        )

    cmd = [sys.executable, str(aggregator_path)]
    print("[TraceML] Launching TraceML aggregator:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )


def start_training_process(
    train_cmd: list[str], env: Dict[str, str], cwd: str
) -> subprocess.Popen:
    """Start the training process in a new process group.

    The subprocess cwd is set explicitly so worker processes see the same
    working directory the user launched TraceML from.
    """
    print("[TraceML] Launching TraceML executor:", " ".join(train_cmd))
    return subprocess.Popen(
        train_cmd,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )


def launch_process(script_path: str, args: argparse.Namespace) -> None:
    """Launch the TraceML aggregator and the target training process.

    Flow
    ----
    1. Prepare TraceML environment variables
    2. Start aggregator process
    3. Wait for aggregator TCP readiness
    4. Start training process
    5. Keep training as the primary process; aggregator may fail open
    6. On shutdown, terminate child process groups and update the manifest
    """

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

    runner_path = str(Path(__file__).parent / "runtime" / "executor.py")
    script_args = args.args or []

    if env["TRACEML_DISABLED"] == "1":
        print(
            "[TraceML] TraceML is disabled via --disable-traceml. Running natively."
        )
        train_cmd = [
            *_build_torchrun_base_cmd(args.nproc_per_node),
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
        *_build_torchrun_base_cmd(args.nproc_per_node),
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
                    "[TraceML] Training finished — stopping aggregator…",
                    file=sys.stderr,
                )
                terminate_process_group(
                    agg_proc, timeout_sec=DEFAULT_SHUTDOWN_TIMEOUT_SEC
                )

            final_status = "completed" if train_rc == 0 else "failed"
            update_run_manifest(
                manifest_path,
                status=final_status,
                artifacts=_collect_existing_artifacts(
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
    """Entry point for ``traceml watch``, ``traceml run``, and ``traceml deep``."""
    args.profile = profile
    try:
        script_path = _resolve_existing_script_path(args.script)
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

    decoder = msgspec.msgpack.Decoder()
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
            print(
                f"[TraceML] ERROR: decoding failed for {path.name}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1)


def run_compare(args: argparse.Namespace) -> None:
    """
    Compare two TraceML final summary JSON files.
    """
    try:
        from traceml.reporting.compare import compare_summaries

        compare_summaries(
            args.left,
            args.right,
            output=args.output,
            print_to_stdout=True,
        )
    except RuntimeError as exc:
        print(f"[TraceML] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"[TraceML] ERROR: compare failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _add_launch_args(parser: argparse.ArgumentParser) -> None:
    """Add shared launch arguments for TraceML run commands."""
    parser.add_argument(
        "script", help="Path to the target Python training script."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "dashboard", "summary"],
        help=(
            "TraceML display mode to launch. "
            "Use 'summary' for final-summary-only runs. Default: cli."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable TraceML logging output.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="./logs",
        help="Directory for TraceML session logs.",
    )
    parser.add_argument(
        "--num-display-layers",
        type=int,
        default=5,
        help="Maximum number of model layers to display in the live UI.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional explicit session id.",
    )
    parser.add_argument(
        "--tcp-host",
        type=str,
        default="127.0.0.1",
        help="Aggregator bind host.",
    )
    parser.add_argument(
        "--tcp-port", type=int, default=29765, help="Aggregator bind port."
    )
    parser.add_argument(
        "--remote-max-rows",
        type=int,
        default=200,
        help="Maximum number of rows returned by remote telemetry queries.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="torchrun nproc_per_node value.",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to the target training script. "
            "Usage: traceml <watch|run|deep> <script> --args -- <script args>"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history saving (live view only; summaries and comparisons unavailable).",
    )
    parser.add_argument(
        "--disable-traceml",
        action="store_true",
        help="Disable TraceML telemetry and run the script natively via torchrun.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level TraceML CLI parser."""
    parser = argparse.ArgumentParser(
        "traceml",
        description=(
            "Run TraceML around a training script.\n\n"
            "Examples:\n"
            "  traceml watch train.py\n"
            "  traceml run train.py --args -- --epochs 10 --lr 1e-3\n"
            "  traceml deep train.py --args -- --config config.yaml"
            "  traceml compare run_a.json run_b.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    watch_parser = sub.add_parser(
        "watch",
        help="Run a script in lightweight watch mode (system and process telemetry only).",
    )
    _add_launch_args(watch_parser)

    run_parser = sub.add_parser(
        "run",
        help="Run a script with TraceML bottleneck instrumentation.",
    )
    _add_launch_args(run_parser)

    deep_parser = sub.add_parser(
        "deep",
        help="Run a script with TraceML deep layerwise instrumentation.",
    )
    _add_launch_args(deep_parser)

    compare_parser = sub.add_parser(
        "compare",
        help="Compare two TraceML final summary JSON files.",
    )
    compare_parser.add_argument(
        "left",
        help="Path to the left-hand TraceML final summary JSON file.",
    )
    compare_parser.add_argument(
        "right",
        help="Path to the right-hand TraceML final summary JSON file.",
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output base path. "
            "Writes both <base>.json and <base>.txt. "
            "Default: compare/<left>_vs_<right> in the current directory."
        ),
    )

    inspect_parser = sub.add_parser(
        "inspect", help="Inspect binary .msgpack logs."
    )
    inspect_parser.add_argument("file", help="Path to a .msgpack file.")

    return parser


def main() -> None:
    """CLI entrypoint for the TraceML launcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"watch", "run", "deep"}:
        _validate_launch_args(args)

    if args.command == "watch":
        run_with_tracing(args, profile="watch")
    elif args.command == "run":
        run_with_tracing(args, profile="run")
    elif args.command == "deep":
        run_with_tracing(args, profile="deep")
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "inspect":
        run_inspect(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
