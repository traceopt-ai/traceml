import argparse
import runpy
import sys
import os
import time
import traceback

from traceml.manager.tracker_manager import TrackerManager


def run_with_tracing(
    script_path: str,
    interval: float = 1.0,
    log_dir: str = None,
    notebook: bool = False,
    script_args: list = None,
):
    """
    Entry point for CLI: starts tracking and runs the target script.

    Args:
        script_path (str): Script to execute with tracing.
        interval (int): Sampling interval.
        log_dir (str): Root directory for log output.
        notebook (bool): Whether to run in notebook mode.
        script_args (list): List of arguments to pass to the target script.
    """
    # Checking script path existence
    if not script_path or not os.path.exists(script_path):
        print(
            f"Error: Script '{script_path}' not found or not specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Checking if log dir exists or provided by the user
    if log_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(os.getcwd(), f".traceml_runs/{timestamp}")
    else:
        log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Print info for the user
    print(f"Running script '{os.path.basename(script_path)}' with TraceML tracing...")
    print(f"Sampling interval: {interval} seconds")
    print(f"Log directory: {log_dir}")

    tracker = TrackerManager(interval_sec=interval, mode="cli")

    # --- Arguments for the target script ---
    original_argv = list(sys.argv)
    # Set sys.argv for the target script
    # The first element is always the script name itself
    sys.argv = [script_path] + (script_args if script_args is not None else [])

    exit_code = 0
    exception_caught = None

    tracker.start()

    try:
        # Run the user's script as sub  process
        print(f"\n--- Running: {sys.argv[0]} {' '.join(sys.argv[1:])} ---\n")

        # Emulate Python interpreter import behavior
        script_dir = os.path.dirname(os.path.abspath(script_path))
        cwd = os.path.abspath(os.getcwd())

        # Ensure both current working directory and script directory are importable
        for path in [script_dir, cwd]:
            if path not in sys.path:
                sys.path.insert(0, path)

        # Also normalize environment variable for subprocesses (e.g. multiprocessing)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [p for p in [script_dir, cwd] if p] + [os.environ.get("PYTHONPATH", "")]
        )
        # --------------------------------------------------------

        # Execute user script in the same process, like `python script.py`
        runpy.run_path(script_path, run_name="__main__")
        print("\n--- User script finished successfully ---")

    except SystemExit as e:
        exit_code = e.code
        print(f"\n--- User script exited with code {exit_code} ---", file=sys.stderr)

    except Exception as e:
        # Catch any other unhandled exceptions (e.g., RuntimeError, MemoryError)
        exception_caught = e
        exit_code = 1  # Failure
        print("\n--- User script crashed! ---", file=sys.stderr)
        print(f"Exception Type: {type(e).__name__}", file=sys.stderr)
        print(f"Exception Message: {e}", file=sys.stderr)
        print("\n--- TraceML Crash Report (Partial) ---", file=sys.stderr)

        # Log detailed traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        tracker.stop()
        tracker.log_summaries()

        # Restore original sys.argv
        sys.argv = original_argv

        # Re-raise the exception so the main CLI process exits with an error status
        if exception_caught:
            raise exception_caught from None
        elif exit_code != 0:
            sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        "TraceML: Automatic PyTorch memory, CPU, and GPU profiling for training runs."
    )

    # Add a version argument
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")
    # Subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Available Commands",
        required=True,
        help="Use 'traceml <command> --help' for more information on a specific command.",
    )

    # --- 'run' subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run a Python script with TraceML's profiling enabled.",
        description="Executes a given Python script while automatically tracking its PyTorch memory, CPU, and GPU usage.",
    )
    run_parser.add_argument(
        "script", help="Path to the Python script to run with TraceML profiling."
    )
    run_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval for resource metrics in seconds (default: 1.0).",
    )
    run_parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.getcwd(), ".traceml_runs"),
        help="Root directory to save TraceML's profiling logs and reports (default: '.traceml_runs/').",
    )
    run_parser.add_argument(
        "--notebook",
        action="store_true",
        help="Render TraceML output for Jupyter notebook (inline HTML) instead of terminal live display.",
    )

    # This captures all remaining arguments after --log-dir etc.
    run_parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,  # Capture all remaining arguments
        help="Additional arguments to pass directly to the script being run.",
    )
    args = parser.parse_args()

    if args.command == "run":
        # Pass the training script's arguments separately
        script_args = args.args if args.args else []
        run_with_tracing(
            args.script,
            interval=args.interval,
            log_dir=args.log_dir,
            notebook=args.notebook,
            script_args=script_args,
        )
    else:
        parser.print_help()
        sys.exit(1)
