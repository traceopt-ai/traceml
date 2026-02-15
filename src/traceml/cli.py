import argparse
import json
import os
import signal
import subprocess
import sys
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


def prepare_log_directory(log_dir: str = None) -> str:
    """
    Prepare the directory where TraceML will store logs and artifacts.

    If no directory is provided, we create a timestamped folder under:
        .traceml_runs/YYYY-MM-DD_HH-MM-SS
    """
    if not log_dir:
        log_dir = os.path.join(os.getcwd(), ".logs/")
    os.makedirs(log_dir, exist_ok=True)
    return str(Path(log_dir).resolve())


def launch_tracer_process(script_path, args):
    """
    Parent launcher.

    This function:
    1. Sets TraceML configuration via environment variables
    2. Launches a *child Python process* via torchrun
    3. Hands off execution to executor.py, which then runs the user script

    We intentionally isolate tracing in a subprocess so:
    - user code remains untouched
    - crashes do not corrupt the launcher
    - tracing can be fully disabled by not using this entrypoint
    """
    env = os.environ.copy()
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_MODE"] = args.mode
    env["TRACEML_INTERVAL"] = str(args.interval)
    env["TRACEML_ENABLE_LOGGING"] = "1" if args.enable_logging else "0"
    env["TRACEML_LOGS_DIR"] = args.logs_dir
    env["TRACEML_NUM_DISPLAY_LAYERS"] = str(args.num_display_layers)
    env["TRACEML_SESSION_ID"] = (
        args.session_id if args.session_id else get_session_id()
    )
    env["TRACEML_DDP_TELEMETRY"] = "0" if args.disable_ddp_telemetry else "1"
    env["TRACEML_TCP_HOST"] = args.tcp_host
    env["TRACEML_TCP_PORT"] = str(args.tcp_port)
    env["TRACEML_REMOTE_MAX_ROWS"] = str(args.remote_max_rows)
    env["TRACEML_NPROC_PER_NODE"] = str(args.nproc_per_node)
    script_args = args.args or []

    runner_path = str(Path(__file__).parent / "runtime/executor.py")

    if args.mode in ["cli", "dashboard"]:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            runner_path,
            "--",
            *script_args,
        ]
    else:
        raise ValueError(f"Invalid mode '{args.mode}'")

    print("Launching TraceML tracer:", " ".join(cmd))

    # Start torchrun in a NEW process group (critical)
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        return_code = p.wait()
        sys.exit(return_code)
    except KeyboardInterrupt:
        print(
            "\n[TraceML] Interrupt received — terminating torchrun process group…",
            file=sys.stderr,
        )
        # Send SIGTERM to the whole process group
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
        # If it doesn't die quickly, SIGKILL
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        sys.exit(0)


def run_with_tracing(args):
    """
    Entry point for `traceml run ...`
    """
    script_path = validate_script_path(args.script)

    launch_tracer_process(script_path=script_path, args=args)


def run_inspect(args):
    """Decodes and prints binary logs for debugging."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        # read() the whole file or stream via msgspec.msgpack.decode_all
        try:
            data = f.read()
            # Decodes multiple concatenated MessagePack messages into a list
            records = msgspec.msgpack.decode_all(data)
            for r in records:
                print(json.dumps(r, indent=2))
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
    run_parser.add_argument(
        "--disable-ddp-telemetry",
        type=bool,
        default=False,
        help="Disable cross-rank TraceML telemetry (TCP-based)",
    )
    run_parser.add_argument(
        "--tcp-host",
        type=str,
        default="127.0.0.1",
        help="TCP host for TraceML rank-0 telemetry server",
    )
    run_parser.add_argument(
        "--tcp-port",
        type=int,
        default=29765,
        help="TCP port for TraceML rank-0 telemetry server",
    )
    run_parser.add_argument(
        "--remote-max-rows",
        type=int,
        default=200,
        help="Max rows kept per remote rank in TraceML",
    )
    run_parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help=("Number of processes to launch via torchrun."),
    )

    run_parser.add_argument("--args", nargs=argparse.REMAINDER)
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
