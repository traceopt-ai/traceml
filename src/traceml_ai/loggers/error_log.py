# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal, Optional

from traceml_ai.runtime.identity import resolve_runtime_identity
from traceml_ai.runtime.session import rank_dir_name

ErrorLogRole = Literal["rank", "aggregator", "launcher"]
_HANDLER_MARKER = "_traceml_error_file_handler"
_SETUP_LOCK = threading.Lock()


class _SilentRotatingFileHandler(RotatingFileHandler):
    """Keep diagnostic write failures from leaking into workload stderr."""

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        pass


def _error_log_path(
    role: ErrorLogRole,
    *,
    session_root: Optional[Path] = None,
    node_rank: Optional[int] = None,
) -> Path:
    """Resolve the structured error path owned by one TraceML process."""
    if session_root is None:
        logs_dir = os.environ.get("TRACEML_LOGS_DIR", "./logs")
        session_id = os.environ.get("TRACEML_SESSION_ID") or "default"
        session_root = Path(logs_dir) / session_id

    if role == "rank":
        owner_dir = rank_dir_name(resolve_runtime_identity().global_rank)
        filename = "traceml_errors.log"
    elif role == "aggregator":
        owner_dir = "aggregator"
        filename = "traceml_errors.log"
    elif role == "launcher":
        if node_rank is None:
            raise ValueError("node_rank is required for launcher error logs")
        owner_dir = f"nodes/node_{int(node_rank)}"
        filename = "launcher_errors.log"
    else:
        raise ValueError(f"unknown TraceML error logger role: {role!r}")

    return Path(session_root).resolve() / owner_dir / filename


def setup_error_logger(
    role: ErrorLogRole,
    *,
    session_root: Optional[Path] = None,
    node_rank: Optional[int] = None,
) -> logging.Logger:
    """Configure the process-owned TraceML ERROR-level rotating file.

    No stderr handler is installed. Initialization is idempotent for the same
    role and session, and filesystem failures leave logging disabled rather
    than affecting training or telemetry control flow.
    """
    logger = logging.getLogger("traceml_ai")
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    try:
        path = _error_log_path(
            role,
            session_root=session_root,
            node_rank=node_rank,
        )
    except Exception:
        path = None

    with _SETUP_LOCK:
        for handler in list(logger.handlers):
            if not getattr(handler, _HANDLER_MARKER, False):
                continue
            base_filename = getattr(handler, "baseFilename", None)
            if (
                path is not None
                and base_filename is not None
                and Path(base_filename) == path
            ):
                return logger
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        if path is None:
            handler = logging.NullHandler()
        else:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = _SilentRotatingFileHandler(
                    path,
                    maxBytes=50_000_000,
                    backupCount=3,
                    encoding="utf-8",
                )
            except Exception:
                handler = logging.NullHandler()

        if isinstance(handler, logging.NullHandler):
            setattr(handler, _HANDLER_MARKER, True)
            logger.addHandler(handler)
            return logger

        handler.setLevel(logging.ERROR)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
        return logger


def get_error_logger(name: str) -> logging.Logger:
    """Return a component child of the process-owned TraceML logger."""
    return logging.getLogger(f"traceml_ai.{name}")
