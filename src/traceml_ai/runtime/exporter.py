# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Background TCP exporter that drains queued payload batches off a dedicated
thread, isolating the sampler thread from a slow or unreachable aggregator."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, List, Optional

from traceml_ai.loggers.error_log import get_error_logger

DEFAULT_EXPORT_QUEUE_SIZE = 2048
DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC = 2.0
_EXPORT_POLL_INTERVAL_SEC = 0.5


class TelemetryExporter:
    """
    Async, thread-backed stand-in for the TCPClient send surface.

    Exposes send/send_batch/close so it can replace the TCP client the
    publisher holds, but enqueues onto a bounded queue instead of sending
    inline. One background thread drains the queue via TCPClient.send_batch.
    """

    def __init__(
        self,
        *,
        tcp_client: Any,
        max_queue_size: int = DEFAULT_EXPORT_QUEUE_SIZE,
        drain_timeout_sec: float = DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC,
        poll_interval_sec: float = _EXPORT_POLL_INTERVAL_SEC,
        logger: Optional[Any] = None,
    ) -> None:
        self._tcp_client = tcp_client
        self._max_queue_size = max(1, int(max_queue_size))
        self._drain_timeout_sec = max(0.0, float(drain_timeout_sec))
        self._poll_interval_sec = max(0.01, float(poll_interval_sec))
        self._logger = logger or get_error_logger("TelemetryExporter")

        self._queue: Deque[List[Any]] = deque()
        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._dropped = 0
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="TraceMLExporter",
            daemon=True,
        )

    @property
    def dropped_count(self) -> int:
        """Number of payload batches dropped on overflow or shutdown."""
        with self._cond:
            return self._dropped

    @property
    def queue_size(self) -> int:
        """Current number of queued payload batches."""
        with self._cond:
            return len(self._queue)

    def start(self) -> None:
        """Start the exporter thread."""
        self._thread.start()

    def send_batch(self, batch: List[Any]) -> None:
        """Enqueue a collected payload batch. Never blocks on the network."""
        if not batch:
            return
        self._enqueue(list(batch))

    def send(self, payload: Any) -> None:
        """Enqueue a single payload as a one-item batch (used for control)."""
        if not payload:
            return
        self._enqueue([payload])

    def close(self) -> None:
        """TCPClient.close parity. Stops with the configured drain budget."""
        self.stop()

    def stop(self, timeout_sec: Optional[float] = None) -> None:
        """Signal stop, let the thread do a bounded final drain, then close."""
        with self._cond:
            if self._stopped:
                return
            self._stopped = True
            if timeout_sec is not None:
                self._drain_timeout_sec = max(0.0, float(timeout_sec))
            self._stop_event.set()
            self._cond.notify_all()

        if self._thread.is_alive():
            self._thread.join(
                timeout=self._drain_timeout_sec + self._poll_interval_sec + 1.0
            )

        if self._dropped:
            self._logger.error(
                f"[TraceML] exporter dropped {self._dropped} payload batches"
            )

        try:
            self._tcp_client.close()
        except Exception as exc:
            self._log_exception("TCPClient.close failed", exc)

    def _enqueue(self, batch: List[Any]) -> None:
        with self._cond:
            if self._stopped:
                self._dropped += 1
                return
            if len(self._queue) >= self._max_queue_size:
                self._queue.popleft()  # drop oldest
                self._dropped += 1
            self._queue.append(batch)
            self._cond.notify()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._take_next()
            if batch is not None:
                self._send(batch)
        self._final_drain()

    def _take_next(self) -> Optional[List[Any]]:
        with self._cond:
            if not self._queue and not self._stop_event.is_set():
                self._cond.wait(timeout=self._poll_interval_sec)
            if self._queue:
                return self._queue.popleft()
            return None

    def _pop_nowait(self) -> Optional[List[Any]]:
        with self._cond:
            if self._queue:
                return self._queue.popleft()
            return None

    def _final_drain(self) -> None:
        deadline = time.monotonic() + self._drain_timeout_sec
        while time.monotonic() < deadline:
            batch = self._pop_nowait()
            if batch is None:
                return
            self._send(batch)

        with self._cond:
            remaining = len(self._queue)
            if remaining:
                self._queue.clear()
                self._dropped += remaining

        if remaining:
            self._logger.error(
                "[TraceML] exporter shutdown dropped %d queued payload batches "
                "after drain timeout",
                remaining,
            )

    def _send(self, batch: List[Any]) -> None:
        # Network send runs outside the queue lock so enqueue never blocks.
        try:
            self._tcp_client.send_batch(batch)
        except Exception as exc:
            self._log_exception("TCPClient.send_batch failed", exc)

    def _log_exception(self, label: str, exc: Exception) -> None:
        log = getattr(self._logger, "exception", None)
        if callable(log):
            log("[TraceML] %s: %s", label, exc)
            return
        fallback = getattr(self._logger, "error", None)
        if callable(fallback):
            fallback(f"[TraceML] {label}: {exc}")


__all__ = [
    "DEFAULT_EXPORT_QUEUE_SIZE",
    "DEFAULT_EXPORT_DRAIN_TIMEOUT_SEC",
    "TelemetryExporter",
]
