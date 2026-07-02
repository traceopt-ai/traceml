"""Start/stop TraceML components in the current Python process.

CLI launchers, Ray actors, and future framework integrations should use this
module instead of hand-building TraceMLAggregator or TraceMLRuntime directly.
The caller still owns process orchestration; this module owns component
lifecycle inside one process.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from traceml_ai.loggers.error_log import get_error_logger, setup_error_logger
from traceml_ai.runtime.runtime import TraceMLRuntime
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    AggregatorEndpoint,
    TraceMLSettings,
)

if TYPE_CHECKING:
    from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator


class NoOpRuntime:
    """Fallback runtime used when TraceML is disabled or startup fails."""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


_RUNTIME_STATE_LOCK = threading.Lock()
_ACTIVE_RUNTIME_HANDLE: Optional["RuntimeHandle"] = None


def get_active_runtime_handle() -> Optional["RuntimeHandle"]:
    """Return the RuntimeHandle started in this process, or None.

    Lets ``traceml.init()`` detect a runtime already started by the CLI
    executor (the ``traceml run`` path) and avoid starting a second one.
    """
    return _ACTIVE_RUNTIME_HANDLE


def _register_active_runtime_handle(handle: "RuntimeHandle") -> None:
    """Record the runtime handle started in this process."""
    global _ACTIVE_RUNTIME_HANDLE
    with _RUNTIME_STATE_LOCK:
        _ACTIVE_RUNTIME_HANDLE = handle


def wait_for_aggregator(
    host: str,
    port: int,
    *,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.25,
) -> bool:
    """Return True once ``(host, port)`` accepts a TCP connection.
    Retries for a bounded period. Used by ``traceml.init()`` to fail fast and
    clearly when the aggregator is not running, instead of silently producing
    no telemetry.
    """
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(float(poll_interval_sec))


@dataclass
class AggregatorHandle:
    """Owns one in-process TraceML aggregator lifecycle."""

    settings: TraceMLSettings
    session_root: Path
    db_path: Path
    stop_event: threading.Event
    aggregator: "TraceMLAggregator"

    _stopped: bool = False

    @property
    def endpoint(self) -> AggregatorEndpoint:
        return AggregatorEndpoint(
            host=str(self.settings.aggregator.connect_host),
            port=int(self.aggregator.endpoint.port),
            session_id=str(self.settings.session_id),
        )

    def stop(self, timeout_sec: float = DEFAULT_FINALIZE_TIMEOUT_SEC) -> None:
        """Stop the aggregator once, flushing history and final summary."""
        if self._stopped:
            return
        self._stopped = True
        self.stop_event.set()
        self.aggregator.stop(timeout_sec=float(timeout_sec))


@dataclass
class RuntimeHandle:
    """Owns one per-worker TraceML runtime lifecycle."""

    runtime: TraceMLRuntime | NoOpRuntime
    _stopped: bool = False

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.runtime.stop()


def _apply_settings_env(
    settings: TraceMLSettings, *, disabled: bool = False
) -> None:
    """
    Mirror in-process lifecycle settings into TRACEML_* environment variables.

    Some lower-level utilities still read process environment, because the CLI
    launcher historically configured TraceML that way. Framework integrations
    that start TraceML in-process use this bridge so those paths keep working
    without depending on the CLI launcher.
    """
    os.environ["TRACEML_DISABLED"] = "1" if disabled else "0"
    os.environ["TRACEML_PROFILE"] = str(settings.profile)
    os.environ["TRACEML_UI_MODE"] = str(settings.mode)
    os.environ["TRACEML_INTERVAL"] = str(settings.sampler_interval_sec)
    os.environ["TRACEML_LOGS_DIR"] = str(settings.logs_dir)
    os.environ["TRACEML_SESSION_ID"] = str(settings.session_id or "default")
    os.environ["TRACEML_AGGREGATOR_HOST"] = str(
        settings.aggregator.connect_host
    )
    os.environ["TRACEML_AGGREGATOR_BIND_HOST"] = str(
        settings.aggregator.bind_host
    )
    os.environ["TRACEML_AGGREGATOR_PORT"] = str(settings.aggregator.port)
    os.environ["TRACEML_HISTORY_ENABLED"] = (
        "1" if settings.history_enabled else "0"
    )
    os.environ["TRACEML_SUMMARY_WINDOW_ROWS"] = str(
        settings.summary_window_rows
    )
    os.environ["TRACEML_FINALIZE_TIMEOUT_SEC"] = str(
        settings.finalize_timeout_sec
    )
    os.environ["TRACEML_EXPECTED_WORLD_SIZE"] = str(
        settings.expected_world_size
    )


def _build_aggregator(
    *,
    logger: Any,
    stop_event: threading.Event,
    settings: TraceMLSettings,
) -> "TraceMLAggregator":
    """Build the aggregator lazily so worker runtime imports stay lightweight."""
    from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator

    return TraceMLAggregator(
        logger=logger,
        stop_event=stop_event,
        settings=settings,
    )


def start_aggregator(
    settings: TraceMLSettings,
    *,
    logger: Optional[Any] = None,
    stop_event: Optional[threading.Event] = None,
) -> AggregatorHandle:
    """Start a TraceML aggregator in the current process."""
    session_id = str(settings.session_id or "default")
    normalized = replace(settings, session_id=session_id)
    _apply_settings_env(normalized)

    if logger is None:
        setup_error_logger(is_aggregator=True)
        logger = get_error_logger("TraceMLAggregatorLifecycle")

    session_root = Path(str(normalized.logs_dir)).resolve() / session_id
    aggregator_dir = session_root / "aggregator"
    aggregator_dir.mkdir(parents=True, exist_ok=True)

    db_path = (
        Path(str(normalized.db_path))
        if normalized.db_path
        else aggregator_dir / "telemetry"
    )
    normalized = replace(normalized, db_path=str(db_path))

    event = stop_event or threading.Event()
    aggregator = _build_aggregator(
        logger=logger,
        stop_event=event,
        settings=normalized,
    )
    try:
        aggregator.start()
    except BaseException:
        event.set()
        try:
            aggregator.stop(timeout_sec=1.0)
        except Exception:
            pass
        raise

    os.environ["TRACEML_AGGREGATOR_PORT"] = str(aggregator.endpoint.port)

    return AggregatorHandle(
        settings=normalized,
        session_root=session_root,
        db_path=db_path,
        stop_event=event,
        aggregator=aggregator,
    )


def start_runtime(
    settings: TraceMLSettings,
    *,
    disabled: bool = False,
    fail_open: bool = True,
    on_error: Optional[Callable[[BaseException], None]] = None,
) -> RuntimeHandle:
    """Start a per-worker TraceML runtime in the current process."""
    normalized = replace(
        settings,
        session_id=str(settings.session_id or "default"),
    )
    _apply_settings_env(normalized, disabled=disabled)

    if disabled:
        handle = RuntimeHandle(NoOpRuntime())
        _register_active_runtime_handle(handle)
        return handle

    try:
        runtime = TraceMLRuntime(settings=normalized)
        runtime.start()
        handle = RuntimeHandle(runtime)
        _register_active_runtime_handle(handle)
        return handle
    except BaseException as exc:
        if on_error is not None:
            on_error(exc)
        if not fail_open:
            raise
        return RuntimeHandle(NoOpRuntime())


__all__ = [
    "AggregatorHandle",
    "NoOpRuntime",
    "RuntimeHandle",
    "get_active_runtime_handle",
    "start_aggregator",
    "start_runtime",
    "wait_for_aggregator",
]
