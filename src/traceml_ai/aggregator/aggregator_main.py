# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
TraceML aggregator process entrypoint.

This module runs the TraceMLAggregator as a standalone process. It reads the
same TRACEML_* environment variables as the executor, starts the TCP server
and UI loop, and shuts down gracefully on SIGTERM/SIGINT.

Expected usage (via CLI)
------------------------
1. Start this process first (aggregator) and wait until it is listening.
2. Then start torchrun workers that run ``traceml/runtime/executor.py``.

Error handling
--------------
- Fatal aggregator errors are logged through the configured logger.
- Fatal aggregator errors are also written to ``aggregator_error.log`` under
  the current session directory.
- A brief error is printed to stderr as a last-resort fallback in case the
  terminal UI has already been torn down or failed.
"""

import os
import signal
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from traceml_ai.loggers.error_log import get_error_logger, setup_error_logger
from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.runtime.lifecycle import start_aggregator
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    AggregatorTransportSettings,
    TraceMLSettings,
)

AGGREGATOR_ERROR_LOG_NAME = "aggregator_error.log"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def write_aggregator_error_log(
    session_root: Path,
    header: str,
    error: BaseException,
) -> None:
    """
    Append a fatal aggregator error report to ``aggregator_error.log``.

    This function is best-effort and must never raise. Aggregator failures
    should still surface through stderr and exit codes even if file logging
    fails.
    """
    try:
        session_root.mkdir(parents=True, exist_ok=True)
        path = session_root / AGGREGATOR_ERROR_LOG_NAME

        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{_utc_now_iso()}  {header}\n")
            traceback.print_exception(
                type(error),
                error,
                error.__traceback__,
                file=f,
            )
            f.flush()
    except Exception:
        # Best-effort only: never mask the original failure.
        pass


def read_traceml_env() -> dict[str, Any]:
    """
    Read aggregator configuration from environment variables.

    The aggregator uses the same TRACEML_* variables as the executor/CLI so the
    launcher can configure both components consistently.

    Backward compatibility
    ----------------------
    - ``TRACEML_UI_MODE`` is preferred
    - ``TRACEML_MODE`` is still accepted

    Supported display modes
    -----------------------
    - ``cli``: live terminal UI
    - ``dashboard``: NiceGUI browser UI
    - ``summary``: no live UI, final summary only
    """

    ui_mode = os.environ.get(
        "TRACEML_UI_MODE",
        os.environ.get("TRACEML_MODE", "cli"),
    )

    return {
        "mode": ui_mode,
        "profile": os.environ.get("TRACEML_PROFILE", "run"),
        "interval": float(os.environ.get("TRACEML_INTERVAL", "1.0")),
        "enable_logging": os.environ.get("TRACEML_ENABLE_LOGGING", "0") == "1",
        "logs_dir": os.environ.get("TRACEML_LOGS_DIR", "./logs"),
        "aggregator_host": os.environ.get(
            "TRACEML_AGGREGATOR_HOST",
            "127.0.0.1",
        ),
        "aggregator_bind_host": os.environ.get(
            "TRACEML_AGGREGATOR_BIND_HOST",
            "127.0.0.1",
        ),
        "aggregator_port": int(
            os.environ.get("TRACEML_AGGREGATOR_PORT", "29765")
        ),
        "dashboard_port": int(
            os.environ.get("TRACEML_DASHBOARD_PORT", "8765")
        ),
        "dashboard_auto_open": os.environ.get(
            "TRACEML_DASHBOARD_AUTO_OPEN", "1"
        )
        == "1",
        "session_id": os.environ.get("TRACEML_SESSION_ID", ""),
        "history_enabled": os.environ.get("TRACEML_HISTORY_ENABLED", "1")
        == "1",
        "finalize_timeout_sec": float(
            os.environ.get(
                "TRACEML_FINALIZE_TIMEOUT_SEC",
                str(DEFAULT_FINALIZE_TIMEOUT_SEC),
            )
        ),
        "expected_world_size": int(
            os.environ.get("TRACEML_EXPECTED_WORLD_SIZE", "1")
        ),
        "summary_window_rows": int(
            os.environ.get(
                "TRACEML_SUMMARY_WINDOW_ROWS",
                str(DEFAULT_SUMMARY_WINDOW_ROWS),
            )
        ),
        "html_report": os.environ.get("TRACEML_HTML_REPORT", "0") == "1",
    }


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """
    Install SIGINT/SIGTERM handlers that request aggregator shutdown.

    The handler is intentionally minimal and only signals the main loop to stop.
    """

    def _handler(signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    """
    Aggregator process entrypoint.

    Execution flow
    --------------
    1. Initialize logging
    2. Read configuration from environment variables
    3. Create the session directory and settings
    4. Start the aggregator
    5. Wait for a shutdown request
    6. Stop the aggregator and report any fatal errors
    """
    setup_error_logger(is_aggregator=True)
    logger = get_error_logger("TraceMLAggregatorMain")

    cfg = read_traceml_env()

    session_id = str(cfg["session_id"] or "default")
    session_root = Path(str(cfg["logs_dir"])).resolve() / session_id
    session_dir = session_root / "aggregator"
    session_dir.mkdir(parents=True, exist_ok=True)
    db_path = session_dir / "telemetry"

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    handle = None
    err: Optional[BaseException] = None

    try:
        settings = TraceMLSettings(
            mode=str(cfg["mode"]),
            profile=str(cfg["profile"]),
            render_interval_sec=float(cfg["interval"]),
            enable_logging=bool(cfg["enable_logging"]),
            logs_dir=str(cfg["logs_dir"]),
            dashboard_port=int(cfg["dashboard_port"]),
            dashboard_auto_open=bool(cfg["dashboard_auto_open"]),
            session_id=session_id,
            history_enabled=bool(cfg["history_enabled"]),
            summary_window_rows=int(cfg["summary_window_rows"]),
            html_report=bool(cfg["html_report"]),
            finalize_timeout_sec=float(cfg["finalize_timeout_sec"]),
            expected_world_size=int(cfg["expected_world_size"]),
            aggregator=AggregatorTransportSettings(
                connect_host=str(cfg["aggregator_host"]),
                bind_host=str(cfg["aggregator_bind_host"]),
                port=int(cfg["aggregator_port"]),
            ),
            db_path=str(db_path),
        )

        logger.info("[TraceML] Starting aggregator")
        handle = start_aggregator(
            settings,
            logger=logger,
            stop_event=stop_event,
        )
        stop_event.wait()

    except BaseException as exc:
        err = exc

    finally:
        if handle is not None:
            try:
                logger.info("[TraceML] Stopping aggregator")
                handle.stop(timeout_sec=float(cfg["finalize_timeout_sec"]))
            except Exception as stop_exc:
                if err is None:
                    err = stop_exc

        print(
            f"[TraceML] Logs saved under: {session_root}",
            file=sys.stderr,
            flush=True,
        )

        if err is not None:
            try:
                logger.exception("[TraceML] Aggregator exiting due to error")
            except Exception:
                pass

            write_aggregator_error_log(
                session_root=session_root,
                header="Fatal aggregator error",
                error=err,
            )

            print(
                "\n[TraceML] Aggregator exiting due to error. "
                f"See {session_root / AGGREGATOR_ERROR_LOG_NAME}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(
                type(err),
                err,
                err.__traceback__,
                file=sys.stderr,
            )
            sys.stderr.flush()
            raise SystemExit(1)

    print("\n[TraceML] Aggregator stopped.", file=sys.stderr, flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
