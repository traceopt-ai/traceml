import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import msgspec

from traceml.runtime.session import get_session_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(
    session_root: Path,
    session_id: str,
    script_path: str,
    mode: str,
    logs_dir: str,
    tcp_host: str,
    tcp_port: int,
    nproc_per_node: int,
    history_enabled: bool,
    status: str,
    aggregator_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write or overwrite a small run manifest.json under session_root.

    Parameters
    ----------
    session_root:
        Run/session root directory.
    status:
        Example values: "starting", "running", "completed", "failed", "interrupted"
    extra:
        Optional extra fields to merge at top level.
    """
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "manifest.json"

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "session_id": str(session_id),
        "status": str(status),
        "created_at": _utc_now_iso(),
        "host": {
            "hostname": socket.gethostname(),
        },
        "launch": {
            "script_path": str(Path(script_path).resolve()),
            "mode": str(mode),
            "logs_dir": str(Path(logs_dir).resolve()),
            "tcp_host": str(tcp_host),
            "tcp_port": int(tcp_port),
            "nproc_per_node": int(nproc_per_node),
            "history_enabled": bool(history_enabled),
        },
        "paths": {
            "session_root": str(session_root),
            "aggregator_dir": (
                str(aggregator_dir.resolve())
                if aggregator_dir is not None
                else None
            ),
            "db_path": str(db_path.resolve()) if db_path is not None else None,
        },
        "artifacts": {},
    }

    if extra:
        manifest.update(extra)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def update_run_manifest(
    manifest_path: Path,
    *,
    status: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Update an existing manifest.json in place.
    """
    manifest_path = Path(manifest_path).resolve()

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}

    if status is not None:
        manifest["status"] = str(status)

    manifest["updated_at"] = _utc_now_iso()

    if artifacts:
        manifest.setdefault("artifacts", {}).update(artifacts)

    if extra:
        manifest.update(extra)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def validate_script_path(script_path: str) -> str:
    """
    Validate that the target training script exists.
    We resolve the absolute path so downstream subprocesses
    always receive a stable, unambiguous path.
    """
    p = Path(script_path)
    if not p.exists():
        print(f"Error: Script '{script_path}' not found.", file=sys.stderr)
        sys.exit(1)
    return str(p.resolve())


def terminate_process_group(
    p: subprocess.Popen, timeout_sec: float = 5.0
) -> None:
    """
    Best-effort termination for a subprocess started with start_new_session=True.

    - Sends SIGTERM to the whole process group
    - Escalates to SIGKILL if it doesn't exit within timeout_sec
    """
    if p is None:
        return
    if p.poll() is not None:
        return

    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass

    try:
        p.wait(timeout=timeout_sec)
        return
    except Exception:
        pass

    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def wait_for_tcp_listen(
    host: str,
    port: int,
    proc: subprocess.Popen,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.05,
) -> bool:
    """
    Wait until (host, port) is accepting TCP connections.

    Also fails fast if `proc` exits while waiting.
    """
    deadline = time.time() + float(timeout_sec)
    last_err = None

    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, int(port)), timeout=0.25):
                return True
        except Exception as e:
            last_err = e
            time.sleep(float(poll_interval_sec))

    if last_err is not None:
        print(
            f"[TraceML] Aggregator did not become ready on {host}:{port} "
            f"(last error: {last_err})",
            file=sys.stderr,
        )
    return False


def install_shutdown_handlers(get_procs, manifest_path=None):
    """
    Install SIGINT/SIGTERM handlers that terminate child process groups.

    Children are started with start_new_session=True, so they will NOT receive
    Ctrl+C automatically; the parent must terminate them.
    """

    def _handler(_signum, _frame):
        print(
            "\n[TraceML] Signal received — terminating processes…",
            file=sys.stderr,
        )

        if manifest_path is not None:
            try:
                update_run_manifest(manifest_path, status="interrupted")
            except Exception:
                pass

        for p in get_procs():
            if p is not None:
                terminate_process_group(p, timeout_sec=5.0)

        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def start_aggregator_process(env: dict) -> subprocess.Popen:
    """
    Start the TraceML aggregator as a separate process.

    IMPORTANT:
    - Do NOT pipe stdout/stderr. The aggregator uses Rich to render to a TTY.
    - We rely on the aggregator to print its own errors/tracebacks to screen.
    """
    aggregator_path = (
        Path(__file__).parent / "aggregator" / "aggregator_main.py"
    )
    if not aggregator_path.exists():
        print(
            f"[TraceML] Aggregator entrypoint not found: {aggregator_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [sys.executable, str(aggregator_path)]
    print("[TraceML] Launching TraceML aggregator:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def start_training_process(
    train_cmd: list[str], env: dict
) -> subprocess.Popen:
    """
    Start torchrun in a new process group.

    stdout/stderr are inherited so user errors/tracebacks remain visible.
    """
    print("[TraceML] Launching TraceML executor:", " ".join(train_cmd))
    return subprocess.Popen(train_cmd, env=env, start_new_session=True)


def launch_tracer_process(script_path, args):
    """
    Parent launcher.

    Flow:
    1) Set TraceML env vars
    2) Start aggregator process (TTY / Rich UI)
    3) Wait for aggregator to listen on TCP
    4) Start torchrun
    5) Wait for training
       - If aggregator dies mid-run: warn; training continues (fail-open)
       - When training exits: terminate aggregator and exit with training code
    6) On Ctrl+C / SIGTERM: terminate both process groups
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # makes prints flush promptly in children

    env["TRACEML_DISABLED"] = (
        "1" if getattr(args, "disable_traceml", False) else "0"
    )
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_MODE"] = args.mode
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

    session_id = env["TRACEML_SESSION_ID"]
    session_root = Path(args.logs_dir).resolve() / session_id
    aggregator_dir = session_root / "aggregator"
    db_path = aggregator_dir / "telemetry"

    manifest_path = write_run_manifest(
        session_root=session_root,
        session_id=session_id,
        script_path=script_path,
        mode=args.mode,
        logs_dir=args.logs_dir,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        nproc_per_node=args.nproc_per_node,
        history_enabled=not args.no_history,
        status="starting",
        aggregator_dir=aggregator_dir,
        db_path=db_path,
    )

    runner_path = str(Path(__file__).parent / "runtime/executor.py")
    script_args = args.args or []
    if env["TRACEML_DISABLED"] == "1":
        print(
            "[TraceML] TraceML is disabled via --disable-traceml. Running natively."
        )
        train_cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            str(script_path),
            *script_args,
        ]
        train_proc = start_training_process(train_cmd=train_cmd, env=env)
        install_shutdown_handlers(lambda: (train_proc, None))
        train_proc.wait()
        sys.exit(train_proc.returncode)

    if args.mode in ["cli", "dashboard"]:
        train_cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            runner_path,
            "--",
            *script_args,
        ]
    else:
        raise ValueError(f"Invalid mode '{args.mode}'")

    agg_proc = None
    train_proc = None

    # Ensure Ctrl+C kills both, even during readiness wait.
    install_shutdown_handlers(
        lambda: (train_proc, agg_proc), manifest_path=manifest_path
    )

    # 1) Start aggregator and wait for it to be ready
    print(
        f"[TraceML] Starting aggregator on {args.tcp_host}:{args.tcp_port} (mode={args.mode})"
    )
    agg_proc = start_aggregator_process(env=env)
    print(f"[TraceML] Aggregator PID: {agg_proc.pid}")

    ok = wait_for_tcp_listen(
        host=args.tcp_host,
        port=int(args.tcp_port),
        proc=agg_proc,
        timeout_sec=15.0,  # cold-start (torch/pynvml import) can exceed 5 s
    )
    if not ok:
        # Aggregator failed before training began: fail fast.
        rc = agg_proc.poll()
        print(
            f"[TraceML] Aggregator failed to start (exit={rc}). "
            "See aggregator output above for details.",
            file=sys.stderr,
        )
        terminate_process_group(agg_proc, timeout_sec=3.0)
        update_run_manifest(manifest_path, status="failed")
        sys.exit(1)

    print("[TraceML] Aggregator ready.")
    update_run_manifest(manifest_path, status="running")

    # 2) Start training
    train_proc = start_training_process(train_cmd=train_cmd, env=env)

    # 3) Wait loop (training is primary)
    while True:
        train_rc = train_proc.poll()
        if train_rc is not None:
            # Training finished. Terminate aggregator first, then exit with training rc.
            if agg_proc is not None:
                print(
                    "[TraceML] Training finished — stopping aggregator…",
                    file=sys.stderr,
                )
                terminate_process_group(agg_proc, timeout_sec=5.0)
            final_status = "completed" if train_rc == 0 else "failed"
            update_run_manifest(
                manifest_path,
                status=final_status,
                artifacts={
                    "db": str(db_path),
                    "summary_card_json": str(db_path) + ".summary_card.json",
                    "summary_card_txt": str(db_path) + ".summary_card.txt",
                },
            )
            sys.exit(train_rc)

        # If aggregator died mid-run: warn, but do not affect training.
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
                    "aggregator_exited_early": True,
                    "aggregator_exit_code": agg_rc,
                },
            )
            agg_proc = None

        time.sleep(1)


def run_with_tracing(args):
    """
    Entry point for `traceml run ...`
    """
    script_path = validate_script_path(args.script)
    launch_tracer_process(script_path=script_path, args=args)


def run_inspect(args):
    """Decodes and prints binary .msgpack logs for debugging."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    decoder = msgspec.msgpack.Decoder()
    with open(path, "rb") as f:
        try:
            while True:
                header = f.read(4)
                if not header:
                    break
                if len(header) < 4:
                    print("Warning: truncated frame header", file=sys.stderr)
                    break
                length = struct.unpack("!I", header)[0]
                payload = f.read(length)
                if len(payload) < length:
                    print("Warning: truncated frame payload", file=sys.stderr)
                    break
                record = decoder.decode(payload)
                print(json.dumps(record, indent=2))
        except Exception as e:
            print(f"Error decoding {path.name}: {e}", file=sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser("traceml")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="Run a script with TraceML enabled"
    )
    run_parser.add_argument("script")
    run_parser.add_argument("--mode", type=str, default="cli")
    run_parser.add_argument("--interval", type=float, default=2.0)
    run_parser.add_argument("--enable-logging", action="store_true")
    run_parser.add_argument("--logs-dir", type=str, default="./logs")
    run_parser.add_argument("--num-display-layers", type=int, default=5)
    run_parser.add_argument("--session-id", type=str, default="")
    run_parser.add_argument("--tcp-host", type=str, default="127.0.0.1")
    run_parser.add_argument("--tcp-port", type=int, default=29765)
    run_parser.add_argument("--remote-max-rows", type=int, default=200)
    run_parser.add_argument("--nproc-per-node", type=int, default=1)
    run_parser.add_argument("--args", nargs=argparse.REMAINDER)
    run_parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history saving (live view only; summaries/comparisons unavailable).",
    )
    run_parser.add_argument(
        "--disable-traceml",
        action="store_true",
        help="Disable TraceML telemetry and run the script natively.",
    )

    inspect_parser = sub.add_parser(
        "inspect", help="Inspect binary .msgpack logs"
    )
    inspect_parser.add_argument("file", help="Path to a .msgpack file")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_with_tracing(args)
    elif args.command == "inspect":
        run_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
