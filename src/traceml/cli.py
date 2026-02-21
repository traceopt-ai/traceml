import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import msgspec

from traceml.runtime.session import get_session_id


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


def terminate_process_group(p: subprocess.Popen, timeout_sec: float = 5.0) -> None:
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


def install_shutdown_handlers(get_procs):
    """
    Install SIGINT/SIGTERM handlers that terminate child process groups.

    Children are started with start_new_session=True, so they will NOT receive
    Ctrl+C automatically; the parent must terminate them.
    """
    def _handler(_signum, _frame):
        print("\n[TraceML] Signal received — terminating processes…", file=sys.stderr)
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
    aggregator_path = Path(__file__).parent / "aggregator" / "aggregator_main.py"
    if not aggregator_path.exists():
        print(
            f"[TraceML] Aggregator entrypoint not found: {aggregator_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [sys.executable, str(aggregator_path)]
    print("[TraceML] Launching TraceML aggregator:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def start_training_process(train_cmd: list[str], env: dict) -> subprocess.Popen:
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

    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_MODE"] = args.mode
    env["TRACEML_INTERVAL"] = str(args.interval)
    env["TRACEML_ENABLE_LOGGING"] = "1" if args.enable_logging else "0"
    env["TRACEML_LOGS_DIR"] = args.logs_dir
    env["TRACEML_NUM_DISPLAY_LAYERS"] = str(args.num_display_layers)
    env["TRACEML_SESSION_ID"] = args.session_id if args.session_id else get_session_id()
    env["TRACEML_TCP_HOST"] = args.tcp_host
    env["TRACEML_TCP_PORT"] = str(args.tcp_port)
    env["TRACEML_REMOTE_MAX_ROWS"] = str(args.remote_max_rows)
    env["TRACEML_NPROC_PER_NODE"] = str(args.nproc_per_node)

    runner_path = str(Path(__file__).parent / "runtime/executor.py")
    script_args = args.args or []

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
    install_shutdown_handlers(lambda: (train_proc, agg_proc))

    # 1) Start aggregator and wait for it to be ready
    print(f"[TraceML] Starting aggregator on {args.tcp_host}:{args.tcp_port} (mode={args.mode})")
    agg_proc = start_aggregator_process(env=env)
    print(f"[TraceML] Aggregator PID: {agg_proc.pid}")

    ok = wait_for_tcp_listen(
        host=args.tcp_host,
        port=int(args.tcp_port),
        proc=agg_proc,
        timeout_sec=5.0,
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
        sys.exit(1)

    print("[TraceML] Aggregator ready.")

    # 2) Start training
    train_proc = start_training_process(train_cmd=train_cmd, env=env)

    # 3) Wait loop (training is primary)
    while True:
        train_rc = train_proc.poll()
        if train_rc is not None:
            # Training finished. Terminate aggregator first, then exit with training rc.
            if agg_proc is not None:
                print("[TraceML] Training finished — stopping aggregator…", file=sys.stderr)
                terminate_process_group(agg_proc, timeout_sec=5.0)
            sys.exit(train_rc)

        # If aggregator died mid-run: warn, but do not affect training.
        if agg_proc is not None and agg_proc.poll() is not None:
            agg_rc = agg_proc.returncode
            print(
                f"[TraceML] WARNING: aggregator exited early (code={agg_rc}). "
                "Training will continue without TraceML telemetry.",
                file=sys.stderr,
            )
            agg_proc = None

        time.sleep(0.2)


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

    run_parser = sub.add_parser("run", help="Run a script with TraceML enabled")
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

    inspect_parser = sub.add_parser("inspect", help="Inspect binary .msgpack logs")
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