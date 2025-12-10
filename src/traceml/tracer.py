import os
import sys
import runpy
import traceback

from traceml.manager.tracker_manager import TrackerManager


def main():
    # Environment from CLI
    script_path = os.environ["TRACEML_SCRIPT_PATH"]
    mode = os.environ.get("TRACEML_MODE", "cli")
    interval = float(os.environ.get("TRACEML_INTERVAL", "1.0"))
    enable_logging = os.environ.get("TRACEML_ENABLE_LOGGING", "") == "1"
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "./logs")
    num_display_layers = int(os.environ.get("TRACEML_NUM_DISPLAY_LAYERS", "20"))

    # Start tracker in *child process*
    tracker = TrackerManager(
        interval_sec=interval,
        mode=mode,
        num_display_layers=num_display_layers,
        enable_logging=enable_logging,
        logs_dir=logs_dir,
    )
    print("starting tracker")
    tracker.start()

    # Extract script args after "--"
    try:
        sep = sys.argv.index("--")
        script_args = sys.argv[sep + 1 :]
    except ValueError:
        script_args = []

    sys.argv = [script_path, *script_args]

    exit_code = 0
    error = None

    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        exit_code = e.code
    except Exception as e:
        error = e
        exit_code = 1
    finally:
        tracker.stop()
        tracker.log_summaries()

    if error:
        print("\n--- script crashed here ---", file=sys.stderr)
        traceback.print_exception(type(error), error, error.__traceback__)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
