import argparse
import os
import sys
import time
import subprocess
from pathlib import Path


def validate_script_path(script_path: str) -> str:
    p = Path(script_path)
    if not p.exists():
        print(f"Error: Script '{script_path}' not found.", file=sys.stderr)
        sys.exit(1)
    return str(p.resolve())


def prepare_log_directory(log_dir: str = "./logs") -> str:
    if not log_dir:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(os.getcwd(), f".traceml_runs/{timestamp}")
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
):
    """
    Parent launcher: sets env vars, calls tracer.py in child process.
    """
    env = os.environ.copy()
    env["TRACEML_SCRIPT_PATH"] = script_path
    env["TRACEML_MODE"] = mode
    env["TRACEML_INTERVAL"] = str(interval)
    env["TRACEML_ENABLE_LOGGING"] = "1" if enable_logging else "0"
    env["TRACEML_LOGS_DIR"] = logs_dir
    env["TRACEML_NUM_DISPLAY_LAYERS"] = str(num_display_layers)

    tracer_path = str(Path(__file__).parent / "tracer.py")

    if mode in ["cli", "dashboard"]:
        cmd = [sys.executable, tracer_path, "--", *script_args]
    else:
        raise ValueError(f"Invalid mode '{mode}'")

    print("Launching TraceML tracer:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=env))


def run_with_tracing(args):
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
    )


def build_parser():
    parser = argparse.ArgumentParser("traceml")

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("script")
    run_parser.add_argument("--mode", type=str, default="cli")
    run_parser.add_argument("--interval", type=float, default=2.0)
    run_parser.add_argument("--enable-logging", action="store_true")
    run_parser.add_argument("--logs-dir", type=str, default="./logs")
    run_parser.add_argument("--num-display-layers", type=int, default=10)
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
