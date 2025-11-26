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


def prepare_log_directory(log_dir:str="./logs") -> str:
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
    num_display_layers
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
        cmd = [
            sys.executable,
            tracer_path,
            "--",
            *script_args
        ]
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
    parser = argparse.ArgumentParser("TraceML CLI")

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("script")
    run_parser.add_argument("--mode", type=str, default="cli")
    run_parser.add_argument("--interval", type=float, default=1.0)
    run_parser.add_argument("--enable-logging", action="store_true")
    run_parser.add_argument("--logs-dir", type=str, default=None)
    run_parser.add_argument("--num-display-layers", type=int, default=20)
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




# import argparse
# import runpy
# import sys
# import os
# import time
# import traceback
# import subprocess
# from typing import Union
# from traceml.manager.tracker_manager import TrackerManager
#
#
# def validate_script_path(script_path: str) -> str:
#     """Ensure that the script path exists and is valid."""
#     if not script_path or not os.path.exists(script_path):
#         print(
#             f"Error: Script '{script_path}' not found or not specified.",
#             file=sys.stderr,
#         )
#         sys.exit(1)
#     return os.path.abspath(script_path)
#
#
# def prepare_log_directory(log_dir: Union[str, None]) -> str:
#     """Create and return a valid logging directory path."""
#     if not log_dir:
#         timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
#         log_dir = os.path.join(os.getcwd(), f".traceml_runs/{timestamp}")
#     log_dir = os.path.abspath(log_dir)
#     os.makedirs(log_dir, exist_ok=True)
#     return log_dir
#
#
# def prepare_environment(script_path: str) -> None:
#     """Ensure that the script and current directories are importable."""
#     script_dir = os.path.dirname(os.path.abspath(script_path))
#     cwd = os.path.abspath(os.getcwd())
#
#     for path in [script_dir, cwd]:
#         if path not in sys.path:
#             sys.path.insert(0, path)
#
#     os.environ["PYTHONPATH"] = os.pathsep.join(
#         [p for p in [script_dir, cwd] if p] + [os.environ.get("PYTHONPATH", "")]
#     )
#
#
# def execute_user_script(script_path: str, script_args: Union[list[str], None]) -> int:
#     """
#     Execute the user script via runpy and handle normal exits.
#     Returns an exit code (0 = success, nonzero = failure).
#     """
#     sys.argv = [script_path] + (script_args or [])
#     print(f"\n--- Running: {sys.argv[0]} {' '.join(sys.argv[1:])} ---\n")
#
#     try:
#         runpy.run_path(script_path, run_name="__main__")
#         print("\n--- User script finished successfully ---")
#         return 0
#     except SystemExit as e:
#         print(f"\n--- User script exited with code {e.code} ---", file=sys.stderr)
#         return e.code or 0
#
#
# def handle_script_exception(e: Exception) -> None:
#     """Print and log exception details to stderr."""
#     print("\n--- User script crashed! ---", file=sys.stderr)
#     print(f"Exception Type: {type(e).__name__}", file=sys.stderr)
#     print(f"Exception Message: {e}", file=sys.stderr)
#     print("\n--- TraceML Crash Report (Partial) ---", file=sys.stderr)
#     traceback.print_exc(file=sys.stderr)
#
#
# def run_with_tracing(
#     script_path: str,
#     mode:str = "cli",
#     interval: float = 1.0,
#     enable_logging = False,
#     logs_dir: str = None,
#     script_args: list = None,
#     num_display_layers: int = 20,
# ):
#     """
#     Entry point for CLI: starts tracking and runs the target script.
#
#     Args:
#         script_path (str): Script to execute with tracing.
#         interval (int): Sampling interval.
#         log_dir (str): Root directory for log output.
#         notebook (bool): Whether to run in notebook mode.
#         script_args (list): List of arguments to pass to the target script.
#         num_display_layers (int): Number of recent layers to show in the CLI live display.
#     """
#     # Checking script path existence
#     script_path = validate_script_path(script_path)
#
#     # Checking if log dir exists or provided by the user
#     logs_dir = prepare_log_directory(logs_dir)
#
#     prepare_environment(script_path)
#
#     # Print info for the user
#     print(f"Running script '{os.path.basename(script_path)}' with TraceML tracing...")
#     print(f"Sampling interval: {interval} seconds")
#     print(f"Log directory: {logs_dir}")
#
#     tracker = TrackerManager(
#         interval_sec=interval, mode=mode, num_display_layers=num_display_layers,
#         enable_logging=enable_logging, logs_dir=logs_dir
#     )
#
#     # --- Arguments for the target script ---
#     original_argv = list(sys.argv)
#     exit_code = 0
#     exception_caught = None
#
#     tracker.start()
#
#     try:
#         exit_code = execute_user_script(script_path, script_args)
#     except SystemExit as e:
#         exit_code = e.code
#         print(f"\n--- User script exited with code {exit_code} ---", file=sys.stderr)
#
#     except Exception as e:
#         # Catch any other unhandled exceptions (e.g., RuntimeError, MemoryError)
#         exception_caught = e
#         exit_code = 1  # Failure
#
#     finally:
#         tracker.stop()
#         tracker.log_summaries()
#
#         # Restore original sys.argv
#         sys.argv = original_argv
#
#         # Re-raise the exception so the main CLI process exits with an error status
#         if exception_caught:
#             raise exception_caught from None
#         elif exit_code != 0:
#             sys.exit(exit_code)
#
#
# def build_parser() -> argparse.ArgumentParser:
#     """Build and return the argument parser."""
#     parser = argparse.ArgumentParser(
#         "TraceML: Automatic PyTorch memory, CPU, and GPU profiling for training runs."
#     )
#
#     parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.4")
#
#     subparsers = parser.add_subparsers(
#         dest="command",
#         title="Available Commands",
#         required=True,
#         help="Use 'traceml <command> --help' for details on each command.",
#     )
#
#     # --- 'run' subcommand ---
#     run_parser = subparsers.add_parser(
#         "run",
#         help="Run a Python script with TraceML profiling.",
#         description="Executes a given Python script while automatically tracking its PyTorch memory, CPU, and GPU usage.",
#     )
#     run_parser.add_argument("script", help="Path to the Python script to run.")
#     run_parser.add_argument(
#         "--interval",
#         type=float,
#         default=1.0,
#         help="Sampling interval in seconds (default: 1.0).",
#     )
#     run_parser.add_argument(
#         "--enable-logging",
#         type=bool,
#         default=False,
#         help="Logging samplers to json.",
#     )
#     run_parser.add_argument(
#         "--logs-dir",
#         type=str,
#         default=os.path.join(os.getcwd(), "./logs"),
#         help="Root log directory.",
#     )
#     run_parser.add_argument(
#         "--num-display-layers",
#         type=int,
#         default=20,
#         help="Number of layers to show in CLI dashboard.",
#     )
#     run_parser.add_argument(
#         "--args",
#         nargs=argparse.REMAINDER,
#         help="Extra args to pass to the target script.",
#     )
#     run_parser.add_argument(
#         "--mode",
#         type=str,
#         default="cli",
#         help="Extra args to pass to the target script.",
#     )
#     return parser
#
#
# def main():
#     parser = build_parser()
#     args = parser.parse_args()
#
#     if args.command == "run":
#         run_with_tracing(
#             script_path=args.script,
#             mode=args.mode,
#             interval=args.interval,
#             enable_logging=args.enable_logging,
#             logs_dir=args.logs_dir,
#             script_args=args.args or [],
#             num_display_layers=args.num_display_layers,
#         )
#     else:
#         parser.print_help()
#         sys.exit(1)
