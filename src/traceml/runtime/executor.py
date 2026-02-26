"""
TraceML executor

This module is the execution wrapper used by TraceML to run user scripts
in a controlled environment.

Responsibilities:
- Read TraceML configuration from environment variables
- Start and stop the TrackerManager lifecycle
- Execute the user script in-process via runpy
- Capture crashes and enrich error reporting

This module intentionally runs in the same Python process as the user
script to ensure hooks, stack traces, and execution context are accurate.
"""

import os
import runpy
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

from traceml.runtime.runtime import TraceMLRuntime
from traceml.runtime.settings import TraceMLSettings, TraceMLTCPSettings
from traceml.utils.shared_utils import EXECUTION_LAYER


def write_session_error_log(cfg, header: str, error: Optional[BaseException] = None) -> None:
    """
    Append a crash/interrupt report to:
        <logs_dir>/<session_id>/torchrun_error.log

    Best-effort: never raises.
    """
    try:
        logs_dir = Path(cfg.get("logs_dir", "./logs"))
        session_id = cfg.get("session_id", "") or "no_session"
        out_dir = logs_dir / session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / "torchrun_error.log"
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{datetime.now().isoformat()}  {header}\n")
            if getattr(EXECUTION_LAYER, "current", None) is not None:
                f.write(f"[TraceML] Last execution point: {EXECUTION_LAYER.current}\n")
            if error is not None:
                traceback.print_exception(type(error), error, error.__traceback__, file=f)
            f.flush()
        print(f"[TraceML] Wrote error report to: {path}", file=sys.stderr, flush=True)
    except Exception:
        # Never break the user's run because logging failed
        pass


class NoOpRuntime:
    def start(self):
        pass

    def stop(self):
        pass

    def log_summaries(self, path=None):
        pass


def read_traceml_env():
    """
    Read TraceML configuration injected by the CLI launcher.

    All configuration is passed via environment variables so that:
    - user scripts remain untouched
    - tracer can run in a clean child process
    """
    return {
        "script_path": os.environ["TRACEML_SCRIPT_PATH"],
        "mode": os.environ.get("TRACEML_MODE", "cli"),
        "interval": float(os.environ.get("TRACEML_INTERVAL", "1.0")),
        "enable_logging": os.environ.get("TRACEML_ENABLE_LOGGING", "") == "1",
        "logs_dir": os.environ.get("TRACEML_LOGS_DIR", "./logs"),
        "num_display_layers": int(
            os.environ.get("TRACEML_NUM_DISPLAY_LAYERS", "20")
        ),
        "tcp_host": os.environ.get("TRACEML_TCP_HOST", "127.0.0.1"),
        "tcp_port": int(os.environ.get("TRACEML_TCP_PORT", "29765")),
        "remote_max_rows": int(
            os.environ.get("TRACEML_REMOTE_MAX_ROWS", "200")
        ),
        "session_id": os.environ.get("TRACEML_SESSION_ID", ""),
    }


def extract_script_args():
    """
    Extract arguments intended for the user script.

    Convention:
        traceml run train.py -- --epochs 10 --lr 1e-3

    Everything after '--' is forwarded to the target script.
    """
    try:
        sep = sys.argv.index("--")
        return sys.argv[sep + 1 :]
    except ValueError:
        return []


def start_runtime(cfg):
    """
    Initialize and start the TraceML tracker.

    This must happen *before* user code executes so we can:
    - attach hooks
    - start background samplers
    - capture early allocations
    """
    try:
        settings = TraceMLSettings(
            mode=cfg["mode"],
            sampler_interval_sec=cfg["interval"],
            enable_logging=cfg["enable_logging"],
            logs_dir=cfg["logs_dir"],
            session_id=cfg["session_id"],
            tcp=TraceMLTCPSettings(
                host=cfg["tcp_host"],
                port=cfg["tcp_port"],
            ),
        )
        runtime = TraceMLRuntime(settings=settings)

        print(f"[TraceML] Starting Runtime with Client port at {cfg['tcp_host']}:{cfg['tcp_port']}")
        runtime.start()
        return runtime
    except Exception as e:
        print(
            f"[TraceML] Failed to start TraceMLRuntime: {e}", file=sys.stderr, flush=True
        )
        traceback.print_exception(type(e), e, e.__traceback__)
        return NoOpRuntime()


def stop_runtime(runtime):
    """
    Best-effort shutdown.
    Never raise.
    """
    try:
        runtime.stop()
        ## Summaries are disabled
        # TODO: clear summaries and show only rank 0 for now
        # runtime.log_summaries(path=None)
    except Exception as e:
        print(
            "[TraceML] Error during shutdown (ignored)",
            file=sys.stderr,
        )
        traceback.print_exception(type(e), e, e.__traceback__)


def run_user_script(script_path, script_args):
    """
    Execute the user script in-process using runpy.

    We intentionally do NOT spawn another subprocess here so that:
    - hooks attach to the real Python objects
    - stacktraces remain meaningful
    """
    sys.argv = [script_path, *script_args]
    runpy.run_path(script_path, run_name="__main__")


def report_crash(error):
    """
    Print a TraceML-enhanced crash report.

    If available, we also show the last known execution layer
    (forward / backward / optimizer / etc.).
    """
    print("\n--- script crashed here ---", file=sys.stderr)

    if getattr(EXECUTION_LAYER, "current", None) is not None:
        print(
            f"[TraceML] Last execution point: {EXECUTION_LAYER.current}",
            file=sys.stderr,
        )

    traceback.print_exception(type(error), error, error.__traceback__)


def _coerce_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1

def main():
    """
    TraceML child process entrypoint.

    Execution flow:
        1. Read configuration from environment
        2. Start tracker
        3. Execute user script
        4. Stop tracker and flush data
        5. Report crash context if needed
    """
    cfg = read_traceml_env()
    script_args = extract_script_args()

    runtime = start_runtime(cfg)

    exit_code = 0
    error = None

    try:
        run_user_script(cfg["script_path"], script_args)

    except KeyboardInterrupt as e:
        print("\n[TraceML] KeyboardInterrupt received (Ctrl+C).", file=sys.stderr, flush=True)
        write_session_error_log(cfg, header="KeyboardInterrupt (Ctrl+C)", error=e)
        exit_code = 130
        error = None

    except SystemExit as e:
        exit_code = _coerce_exit_code(e.code)
        if exit_code != 0:
            write_session_error_log(cfg, header=f"SystemExit (code={exit_code})", error=e)
        error = None

    except Exception as e:
        error = e
        exit_code = 1
        write_session_error_log(cfg, header="Unhandled exception in user script", error=e)

    finally:
        stop_runtime(runtime)

    if error:
        report_crash(error)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
