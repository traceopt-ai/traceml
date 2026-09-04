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
- Fatal aggregator errors are logged once through the configured logger.
- A brief error is printed to stderr as a last-resort fallback in case the
  terminal UI has already been torn down or failed.
"""

import os
import signal
import sys
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from traceml_ai.loggers.error_log import get_error_logger, setup_error_logger
from traceml_ai.runtime.lifecycle import start_aggregator
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_UI_MODE,
    AggregatorTransportSettings,
    TraceMLSettings,
)
from traceml_ai.telemetry.retention import (
    DEFAULT_HISTORY_RETENTION_S,
    parse_history_retention,
)


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
        os.environ.get("TRACEML_MODE", DEFAULT_UI_MODE),
    )

    return {
        "mode": ui_mode,
        "profile": os.environ.get("TRACEML_PROFILE", "run"),
        "interval": float(
            os.environ.get("TRACEML_INTERVAL", str(DEFAULT_INTERVAL_SEC))
        ),
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
        "history_retention_s": parse_history_retention(
            os.environ.get(
                "TRACEML_HISTORY_RETENTION",
                str(DEFAULT_HISTORY_RETENTION_S),
            )
        ),
        "finalize_timeout_sec": float(
            os.environ.get(
                "TRACEML_FINALIZE_TIMEOUT_SEC",
                str(DEFAULT_FINALIZE_TIMEOUT_SEC),
            )
        ),
        "expected_world_size": int(
            os.environ.get("TRACEML_EXPECTED_WORLD_SIZE", "1")
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


def run_aggregator(
    settings: TraceMLSettings,
    *,
    logger: Optional[Any] = None,
) -> int:
    """
    Own one aggregator lifecycle until a shutdown signal, then exit cleanly.

    This is the single aggregator-owning entry point shared by the standalone
    aggregator process (started by ``traceml run``) and the ``traceml serve``
    command. It:

    - starts only the aggregator (never a user training script)
    - prints the reachable endpoint
    - blocks until SIGINT/SIGTERM
    - shuts down cleanly, preserving final-summary behavior
    - logs fatal errors to its structured process-owned log

    Returns a process exit code (0 on clean shutdown, 1 on fatal error).
    """
    session_id = str(settings.session_id or "default")
    session_root = Path(str(settings.logs_dir)).resolve() / session_id
    session_dir = session_root / "aggregator"
    session_dir.mkdir(parents=True, exist_ok=True)
    db_path = session_dir / "telemetry"
    settings = replace(settings, session_id=session_id, db_path=str(db_path))

    # The shared error logger resolves its directory from TRACEML_LOGS_DIR and
    # TRACEML_SESSION_ID. The `traceml run` launcher sets these before spawning
    # the aggregator subprocess; the `traceml serve` path runs in-process and
    # does not, so mirror them from the resolved settings before logging setup.
    os.environ["TRACEML_LOGS_DIR"] = str(settings.logs_dir)
    os.environ["TRACEML_SESSION_ID"] = session_id

    if logger is None:
        setup_error_logger(role="aggregator")
        logger = get_error_logger("TraceMLAggregatorMain")

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    handle = None
    err: Optional[BaseException] = None

    try:
        logger.info("[TraceML] Starting aggregator")
        handle = start_aggregator(
            settings,
            logger=logger,
            stop_event=stop_event,
        )

        endpoint = handle.endpoint
        print(
            "[TraceML] Aggregator ready on "
            f"{settings.aggregator.bind_host}:{endpoint.port} "
            f"(workers connect to {endpoint.host}:{endpoint.port}, "
            f"session={endpoint.session_id}, ui={settings.mode}). "
            "Press Ctrl+C to stop.",
            file=sys.stderr,
            flush=True,
        )

        stop_event.wait()

    except BaseException as exc:
        err = exc

    finally:
        if handle is not None:
            try:
                logger.info("[TraceML] Stopping aggregator")
                handle.stop(timeout_sec=float(settings.finalize_timeout_sec))
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
                logger.error(
                    "[TraceML] Aggregator exiting due to error",
                    exc_info=(type(err), err, err.__traceback__),
                )
            except Exception:
                pass

            print(
                "\n[TraceML] Aggregator exiting due to error. "
                f"Structured log: {session_dir / 'traceml_errors.log'}. "
                "Launcher-owned runs also preserve raw stderr at "
                f"{session_dir / 'process.stderr.log'}.",
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
            return 1

    print("\n[TraceML] Aggregator stopped.", file=sys.stderr, flush=True)
    return 0


def main() -> None:
    """
    Standalone aggregator process entrypoint.

    Reads configuration from the ``TRACEML_*`` environment variables set by the
    ``traceml run`` launcher, builds settings, and delegates the lifecycle to
    :func:`run_aggregator`.
    """
    setup_error_logger(role="aggregator")
    logger = get_error_logger("TraceMLAggregatorMain")

    cfg = read_traceml_env()

    settings = TraceMLSettings(
        mode=str(cfg["mode"]),
        profile=str(cfg["profile"]),
        render_interval_sec=float(cfg["interval"]),
        enable_logging=bool(cfg["enable_logging"]),
        logs_dir=str(cfg["logs_dir"]),
        dashboard_port=int(cfg["dashboard_port"]),
        dashboard_auto_open=bool(cfg["dashboard_auto_open"]),
        session_id=str(cfg["session_id"] or "default"),
        history_enabled=bool(cfg["history_enabled"]),
        history_retention_s=float(cfg["history_retention_s"]),
        html_report=bool(cfg["html_report"]),
        finalize_timeout_sec=float(cfg["finalize_timeout_sec"]),
        expected_world_size=int(cfg["expected_world_size"]),
        aggregator=AggregatorTransportSettings(
            connect_host=str(cfg["aggregator_host"]),
            bind_host=str(cfg["aggregator_bind_host"]),
            port=int(cfg["aggregator_port"]),
        ),
    )

    raise SystemExit(run_aggregator(settings, logger=logger))


if __name__ == "__main__":
    main()
