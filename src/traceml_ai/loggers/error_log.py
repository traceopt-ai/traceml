# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from traceml_ai.runtime.identity import resolve_runtime_identity
from traceml_ai.runtime.session import rank_dir_name


def setup_error_logger(is_aggregator=False) -> logging.Logger:
    """
    Configure and initialize the global TraceML error logger.

    This function sets up a process-wide logger named ``"traceml"`` with:
      - WARNING and above logged to stderr (for visibility during runs)
      - ERROR and above logged to a rotating file on disk

    The logger is initialized only once per process. Subsequent calls
    return the already-configured logger.

    Logging behavior
    ----------------
    - stderr:
        * Level: WARNING+
        * Intended for immediate user feedback
    - file:
        * Level: ERROR+
        * Rotating log file with size-based rollover

    Distributed assumptions
    -----------------------
    - Uses global rank to separate process-owned log files.
    - This avoids write contention between ranks across all nodes.

    Returns
    -------
    logging.Logger
        The configured TraceML root logger.
    """
    logger = logging.getLogger("traceml")

    # If handlers are already attached, assume the logger
    # has been initialized and return it as-is.
    if logger.handlers:
        return logger

    # Root level is ERROR: more verbose output is controlled
    # at the handler level.
    logger.setLevel(logging.ERROR)

    # ----------------------------
    # File handler (ERROR+)
    # ----------------------------
    if is_aggregator is False:
        rank_dir = rank_dir_name(resolve_runtime_identity().global_rank)
    else:
        rank_dir = "aggregator"

    logs_dir = os.environ.get("TRACEML_LOGS_DIR")
    session_id = os.environ.get("TRACEML_SESSION_ID")

    # Directory layout:
    #   <logs_dir>/<session_id>/rank_<global_rank>/traceml_errors.log
    #   <logs_dir>/<session_id>/aggregator/traceml_errors.log
    errors_dir = Path(logs_dir) / session_id / rank_dir

    errors_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        errors_dir / "traceml_errors.log",
        maxBytes=50_000_000,  # ~5 MB per file
        backupCount=3,  # keep last 3 rotated files
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # Prevent propagation to the root logger to avoid
    # duplicate log lines in user applications.
    logger.propagate = False

    return logger


def get_error_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the TraceML namespace.

    This is a lightweight convenience wrapper that ensures all
    TraceML loggers share a common root configuration created by
    ``setup_error_logger()``.

    Parameters
    ----------
    name : str
        Sub-logger name (e.g., "RemoteDBStore", "Sampler").

    Returns
    -------
    logging.Logger
        Logger instance named ``f"traceml_ai.{name}"``.
    """
    return logging.getLogger(f"traceml_ai.{name}")
