import argparse
import os
import sys
import subprocess
from pathlib import Path


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


def launch_tracer_process(
    script_path,
    mode,
    interval,
    enable_logging,
    logs_dir,
    script_args,
    num_display_layers,
    nproc_per_node,
    session_name
):
    """
    Parent launcher.

    This function:
    1. Sets TraceML configuration via environment variables
    2. Launches a *child Python process* via torchrun
    3. Hands off execution to tracer.py, which then runs the user script

    We intentionally isolate tracing in a subprocess so:
    - user code remains untouched
    - crashes do not corrupt the launcher
    - tracing can be fully disabled by not using this entrypoint
    """
    env = os.environ.copy()
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_MODE"] = mode
    env["TRACEML_INTERVAL"] = str(interval)
    env["TRACEML_ENABLE_LOGGING"] = "1" if enable_logging else "0"
    env["TRACEML_LOGS_DIR"] = logs_dir
    env["TRACEML_NUM_DISPLAY_LAYERS"] = str(num_display_layers)
    env["TRACEML_SESSION_NAME"] = session_name if session_name else ""

    tracer_path = str(Path(__file__).parent / "tracer.py")

    if nproc_per_node > 1:
        print(
            "\n[TraceML WARNING]\n"
            f"Requested --nproc-per-node={nproc_per_node}, but distributed "
            "training (DDP / multi-GPU) tracing is NOT supported yet.\n"
            "Tracing will run, but results may be incomplete or incorrect.\n"
            "For now, please use --nproc-per-node=1.\n",
            file=sys.stderr,
        )

    if mode in ["cli", "dashboard"]:
        cmd = ["torchrun", "--nproc_per_node=1", tracer_path, "--", *script_args]
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    print("Launching TraceML tracer:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=env))


def run_with_tracing(args):
    """
    Entry point for `traceml run ...`
    """
    script_path = validate_script_path(args.script)
    logs_dir = prepare_log_directory(args.logs_dir)

    launch_tracer_process(
        script_path=script_path,
        mode=args.mode,
        interval=args.interval,
        enable_logging=args.enable_logging,
        logs_dir=logs_dir,
        script_args=args.args or [],
        num_display_layers=args.num_display_layers,
        nproc_per_node=args.nproc_per_node,
        session_name=args.session_name,
    )


def build_parser():
    parser = argparse.ArgumentParser("traceml")

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a script with TraceML enabled")
    run_parser.add_argument("script")
    run_parser.add_argument("--mode", type=str, default="cli")
    run_parser.add_argument("--interval", type=float, default=2.0)
    run_parser.add_argument("--enable-logging", action="store_true")
    run_parser.add_argument("--logs-dir", type=str, default="./logs")
    run_parser.add_argument("--num-display-layers", type=int, default=10)
    run_parser.add_argument("--session-name", type=str, default="")

    run_parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help=(
            "Number of processes to launch via torchrun. "
            "Default is 1. Distributed tracing is not supported yet."
        ),
    )
    run_parser.add_argument("--args", nargs=argparse.REMAINDER)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_with_tracing(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
