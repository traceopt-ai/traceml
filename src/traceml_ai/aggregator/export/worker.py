# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Background worker that moves admitted payloads to an exporter.

``submit`` puts a payload on a bounded queue and never blocks. One daemon
thread wakes every ``interval_sec``, turns everything queued into export
records and hands them to the exporter in one ``write`` and ``flush``.
A failing exporter is closed and abandoned; the counts keep going.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from traceml_ai.aggregator.export.base import Exporter
from traceml_ai.aggregator.export.records import records_from_payload
from traceml_ai.loggers.error_log import get_error_logger


@dataclass(frozen=True)
class ExportStats:
    """
    Counters of one export worker.

    Attributes
    ----------
    written_records:
        Records a ``write()`` accepted (and the ``flush()`` after it).
    dropped_batches:
        Payload batches refused by a full queue, submitted after ``stop()``
        began, or cleared at the stop deadline.
    failed_records:
        Records handed to a ``write()`` or ``flush()`` that raised, plus
        records built while the exporter is failed.
    skipped_items:
        Payload items ``records_from_payload`` could not turn into records.
    exporter_failed:
        True once ``open``, ``write`` or ``flush`` has raised.
    """

    written_records: int
    dropped_batches: int
    failed_records: int
    skipped_items: int
    exporter_failed: bool


class ExportWorker:
    """Bounded queue drained to one exporter on a fixed tick."""

    def __init__(
        self,
        exporter: Exporter,
        *,
        interval_sec: float = 1.0,
        max_queue: int = 4096,
        logger: Optional[Any] = None,
    ) -> None:
        self._exporter = exporter
        self._name = type(exporter).__name__
        self._interval_sec = max(0.0, float(interval_sec))
        self._logger = logger or get_error_logger("TraceML-ExportWorker")

        self._q: "queue.Queue[Any]" = queue.Queue(
            maxsize=max(1, int(max_queue))
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-ExportWorker",
            daemon=True,
        )
        self._started = False
        self._stopping = False
        self._stats: Optional[ExportStats] = None

        # Ownership of the exporter's close, guarded by ``_lock``.
        self._closed = False
        self._thread_done = False
        self._close_on_exit = False

        self._written = 0
        self._dropped = 0
        self._failed_records = 0
        self._skipped = 0
        self._failed = False

    def start(self) -> None:
        """Open the exporter and start the worker thread (idempotent)."""
        if self._started or self._stopping:
            return
        self._started = True
        try:
            self._exporter.open()
        except Exception as exc:
            self._failed = True
            self._log_error(
                f"[TraceML] {self._name}.open failed, export disabled: "
                f"{exc!r}"
            )
        try:
            self._thread.start()
        except Exception as exc:
            self._log_error(
                f"[TraceML] export worker thread did not start: {exc!r}"
            )

    def submit(self, payload: Any) -> None:
        """
        Enqueue one admitted payload (non-blocking, never raises).

        A full queue, or a call after ``stop()`` began, counts the payload
        as one dropped batch.
        """
        try:
            with self._lock:
                if self._stopping:
                    self._dropped += 1
                    return
                try:
                    self._q.put_nowait(payload)
                except queue.Full:
                    self._dropped += 1
        except Exception:
            pass

    def stats(self) -> ExportStats:
        """Return a snapshot of the counters."""
        with self._lock:
            return self._snapshot()

    def stop(self, timeout_sec: float = 2.0) -> ExportStats:
        """
        Stop the thread, drain what is queued within ``timeout_sec``, close.

        The thread is joined, then queued payloads are exported one at a
        time until the queue is empty or the deadline passes; the rest are
        counted as dropped. The bound is ``timeout_sec`` plus one ``write``
        already in progress. If the thread is still inside a ``write`` at
        the deadline, the drain is skipped and the thread closes the
        exporter when that call returns, so no two exporter calls overlap.
        Safe to call twice; never raises.
        """
        if self._stats is not None:
            return self._stats
        try:
            return self._stop(float(timeout_sec))
        except Exception as exc:
            self._log_error(f"[TraceML] export worker stop failed: {exc!r}")
            self._stats = self.stats()
            return self._stats

    def _stop(self, timeout_sec: float) -> ExportStats:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self._lock:
            if self._stopping:
                return self._snapshot()
            self._stopping = True
        self._stop_event.set()

        thread_busy = False
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            with self._lock:
                if not self._thread_done:
                    self._close_on_exit = True
                    thread_busy = True

        if self._started and not thread_busy:
            while time.monotonic() < deadline:
                try:
                    payload = self._q.get_nowait()
                except queue.Empty:
                    break
                self._export([payload])

        remaining = self._clear_queue()
        if remaining:
            self._log_error(
                f"[TraceML] export worker dropped {remaining} queued "
                "payload batches after the stop deadline"
            )

        if self._started and not thread_busy:
            self._close_exporter()

        self._stats = self.stats()
        return self._stats

    def _snapshot(self) -> ExportStats:
        """Counters as they are now; the caller holds ``_lock``."""
        return ExportStats(
            written_records=self._written,
            dropped_batches=self._dropped,
            failed_records=self._failed_records,
            skipped_items=self._skipped,
            exporter_failed=self._failed,
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._interval_sec):
                self._export(self._drain())
        except Exception as exc:
            self._log_error(f"[TraceML] export worker loop failed: {exc!r}")
        finally:
            with self._lock:
                self._thread_done = True
                close = self._close_on_exit
            if close:
                self._close_exporter()

    def _drain(self) -> List[Any]:
        """Pop everything queued now, without waiting."""
        payloads: List[Any] = []
        while True:
            try:
                payloads.append(self._q.get_nowait())
            except queue.Empty:
                return payloads

    def _clear_queue(self) -> int:
        """Count every payload still queued as dropped."""
        cleared = len(self._drain())
        if cleared:
            with self._lock:
                self._dropped += cleared
        return cleared

    def _export(self, payloads: List[Any]) -> None:
        """Build records from ``payloads`` and hand them to the exporter."""
        if not payloads:
            return
        records: List[Any] = []
        skipped = 0
        for payload in payloads:
            try:
                batch = records_from_payload(payload)
            except Exception:
                skipped += 1
                continue
            records.extend(batch.records)
            skipped += batch.skipped

        with self._lock:
            self._skipped += skipped
            if not records:
                return
            if self._failed:
                self._failed_records += len(records)
                return

        # No lock is held across exporter calls.
        try:
            self._exporter.write(records)
            self._exporter.flush()
        except Exception as exc:
            with self._lock:
                self._failed_records += len(records)
                self._failed = True
            self._log_error(
                f"[TraceML] {self._name} write failed, export stopped: "
                f"{exc!r}"
            )
            self._close_exporter()
            return

        with self._lock:
            self._written += len(records)

    def _close_exporter(self) -> None:
        """Close the exporter once, passing the current drop count."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            dropped = self._dropped
        try:
            self._exporter.close(dropped_batches=dropped)
        except Exception as exc:
            self._log_error(f"[TraceML] {self._name}.close failed: {exc!r}")

    def _log_error(self, msg: str) -> None:
        """Log an internal worker error without raising."""
        try:
            self._logger.error(msg)
        except Exception:
            pass


__all__ = ["ExportStats", "ExportWorker"]
