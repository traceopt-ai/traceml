"""
TraceML aggregator process entrypoint.

This module runs the TraceMLAggregator as a standalone process. It reads the
same TRACEML_* environment variables as the executor, starts the TCP server +
UI loop, and shuts down gracefully on SIGTERM/SIGINT.

Expected usage (via CLI):
- Start this process first (aggregator), wait until it is listening
- Then start torchrun workers that run traceml/runtime/executor.py
"""

import os
import signal
import sys
import threading
import traceback

from traceml.loggers.error_log import get_error_logger, setup_error_logger

from traceml.runtime.settings import TraceMLSettings, TraceMLTCPSettings
from traceml.aggregator.trace_aggregator import TraceMLAggregator


def read_traceml_env() -> dict:
    """
    Read aggregator configuration from environment variables.

    The aggregator uses the same TRACEML_* variables as the executor/CLI so the
    launcher can configure both components consistently.
    """
    return {
        "mode": os.environ.get("TRACEML_MODE", "cli"),
        "interval": float(os.environ.get("TRACEML_INTERVAL", "1.0")),
        "enable_logging": os.environ.get("TRACEML_ENABLE_LOGGING", "") == "1",
        "logs_dir": os.environ.get("TRACEML_LOGS_DIR", "./logs"),
        "num_display_layers": int(os.environ.get("TRACEML_NUM_DISPLAY_LAYERS", "20")),
        "tcp_host": os.environ.get("TRACEML_TCP_HOST", "127.0.0.1"),
        "tcp_port": int(os.environ.get("TRACEML_TCP_PORT", "29765")),
        "remote_max_rows": int(os.environ.get("TRACEML_REMOTE_MAX_ROWS", "200")),
        "session_id": os.environ.get("TRACEML_SESSION_ID", ""),
    }


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """Install SIGINT/SIGTERM handlers to stop the aggregator gracefully."""

    def _handler(signum, _frame):
        # Keep handler minimal and async-signal-safe.
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    setup_error_logger(is_aggregator=True)
    logger = get_error_logger("TraceMLAggregatorMain")

    cfg = read_traceml_env()
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    agg = None
    try:
        settings = TraceMLSettings(
            mode=cfg["mode"],
            render_interval_sec=cfg["interval"],
            num_display_layers=cfg["num_display_layers"],
            enable_logging=cfg["enable_logging"],
            logs_dir=cfg["logs_dir"],
            remote_max_rows=cfg["remote_max_rows"],
            session_id=cfg["session_id"],
            tcp=TraceMLTCPSettings(host=cfg["tcp_host"], port=cfg["tcp_port"]),
        )

        agg = TraceMLAggregator(logger=logger, stop_event=stop_event, settings=settings)
        logger.info("[TraceML] Starting Aggregator")
        agg.start()

        stop_event.wait()

    except BaseException as e:
        err = e
    else:
        err = None
    finally:
        if agg is not None:
            try:
                logger.info("[TraceML] Stopping Aggregator")
                agg.stop(timeout_sec=5.0)
            except Exception:
                pass

        if err is not None:
            print("\n[TraceML] Aggregator exiting due to error:", file=sys.stderr, flush=True)
            traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)

    print("\n[TraceML] Aggregator stopped.", file=sys.stderr, flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()