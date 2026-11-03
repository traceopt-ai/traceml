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
The complete exporter lifecycle runs on that thread, so no two calls overlap.
A failing exporter is closed and abandoned; the counts keep going.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from traceml_ai.aggregator.export.base import Exporter
from traceml_ai.aggregator.export.records import records_from_payload
from traceml_ai.loggers.error_log import get_error_logger

_MIN_INTERVAL_SEC = 0.01


@dataclass(frozen=True)
class ExportStats:
    """
    Counters of one export worker.

    Attributes
    ----------
    written_records:
        Records a ``write()`` accepted. A ``flush()`` that raises after it
        marks the exporter failed but leaves these counted as written.
    dropped_batches:
        Payload batches refused by a full queue, submitted after ``stop()``
        began, still queued at the stop deadline, or queued on a worker
        whose thread never ran.
    failed_records:
        Records handed to a ``write()`` that raised, plus records built
        while the exporter is failed.
    skipped_items:
        Payload items ``records_from_payload`` could not turn into records.
        A payload whose conversion raised counts its length when it is a
        list, else one.
    exporter_failed:
        True once ``open``, ``write``, ``flush`` or ``close`` has raised.
    """

    written_records: int
    dropped_batches: int
    failed_records: int
    skipped_items: int
    exporter_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for diagnostics."""
        return {
            "written_records": int(self.written_records),
            "dropped_batches": int(self.dropped_batches),
            "failed_records": int(self.failed_records),
            "skipped_items": int(self.skipped_items),
            "exporter_failed": bool(self.exporter_failed),
        }


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
        interval = float(interval_sec)
        if not math.isfinite(interval):
            interval = 1.0
        self._interval_sec = max(_MIN_INTERVAL_SEC, interval)
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
        # Lifecycle flags, guarded by ``_lock``.
        self._started = False
        self._thread_started = False
        self._stopping = False
        self._deadline: Optional[float] = None
        self._opened = False
        self._closed = False

        self._written = 0
        self._dropped = 0
        self._failed_records = 0
        self._skipped = 0
        self._failed = False

    def start(self) -> None:
        """Start the worker thread (idempotent)."""
        with self._lock:
            if self._started or self._stopping:
                return
            self._started = True
        # Held across ``Thread.start`` (which returns once the thread runs)
        # so ``stop`` sees either no thread or one it can join.
        start_error: Optional[Exception] = None
        with self._lock:
            try:
                self._thread.start()
                self._thread_started = True
            except Exception as exc:
                start_error = exc
        if start_error is not None:
            self._log_error(
                "[TraceML] export worker thread did not start: "
                f"{start_error!r}"
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
        Stop the thread and return the counters; never raises.

        The thread exports what is queued, one payload at a time, until the
        queue is empty or ``timeout_sec`` has passed, counts the rest as
        dropped and closes the exporter. ``stop`` waits for it by joining
        the thread for ``timeout_sec``, and that join is the whole bound.
        If the thread is still inside an exporter call at the deadline, the
        queue is cleared into ``dropped_batches`` and the thread closes the
        exporter when that call returns; the snapshot returned then is not
        final, and ``stats()`` after the call returns is. A second call
        returns a live ``stats()`` snapshot.
        """
        try:
            return self._stop(timeout_sec)
        except Exception as exc:
            self._log_error(f"[TraceML] export worker stop failed: {exc!r}")
            return self.stats()

    def _stop(self, timeout_sec: float) -> ExportStats:
        budget = max(0.0, float(timeout_sec))
        with self._lock:
            if self._stopping:
                return self._snapshot()
            self._stopping = True
            self._deadline = time.monotonic() + budget
            thread_started = self._thread_started
        self._stop_event.set()

        if not thread_started:
            # No thread will ever run, so the exporter was never opened.
            self._clear_queue()
            return self.stats()

        self._thread.join(timeout=max(0.0, self._deadline - time.monotonic()))
        if self._thread.is_alive():
            cleared = self._clear_queue()
            self._log_error(
                f"[TraceML] {self._name} still busy at the export stop "
                f"deadline; dropped {cleared} queued payload batches, the "
                "exporter closes when the call returns"
            )
        return self.stats()

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
        # Keep the complete lifecycle on this thread. Custom exporters may
        # create thread-affine clients or other resources in ``open``.
        self._open_exporter()
        try:
            while not self._stop_event.wait(self._interval_sec):
                self._export(self._drain())
            while self._before_deadline():
                try:
                    payload = self._q.get_nowait()
                except queue.Empty:
                    break
                self._export([payload])
            cleared = self._clear_queue()
            if cleared:
                self._log_error(
                    f"[TraceML] export worker dropped {cleared} queued "
                    "payload batches after the stop deadline"
                )
        except Exception as exc:
            self._log_error(f"[TraceML] export worker loop failed: {exc!r}")
        finally:
            self._close_exporter()

    def _open_exporter(self) -> None:
        """Open the exporter, disabling it if setup fails."""
        try:
            self._exporter.open()
        except Exception as exc:
            with self._lock:
                self._failed = True
            self._log_error(
                f"[TraceML] {self._name}.open failed, export disabled: "
                f"{exc!r}"
            )
            return
        with self._lock:
            self._opened = True

    def _before_deadline(self) -> bool:
        """True while the stop deadline has not passed."""
        with self._lock:
            deadline = self._deadline
        return deadline is not None and time.monotonic() < deadline

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
                records.extend(batch.records)
                skipped += batch.skipped
            except Exception:
                skipped += len(payload) if isinstance(payload, list) else 1

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
        except Exception as exc:
            self._fail("write", exc, failed_records=len(records))
            return
        with self._lock:
            self._written += len(records)

        try:
            self._exporter.flush()
        except Exception as exc:
            # ``write`` accepted the records, so they stay counted written.
            self._fail("flush", exc, failed_records=0)

    def _fail(self, call: str, exc: Exception, *, failed_records: int) -> None:
        """Mark the exporter failed, log once and close it."""
        with self._lock:
            self._failed_records += failed_records
            self._failed = True
        self._log_error(
            f"[TraceML] {self._name}.{call} failed, export stopped: {exc!r}"
        )
        self._close_exporter()

    def _close_exporter(self) -> None:
        """
        Close the exporter once, passing the current drop count.

        Skipped when ``open`` raised or never ran.
        """
        with self._lock:
            if self._closed or not self._opened:
                return
            self._closed = True
            dropped = self._dropped
        try:
            self._exporter.close(dropped_batches=dropped)
        except Exception as exc:
            with self._lock:
                self._failed = True
            self._log_error(f"[TraceML] {self._name}.close failed: {exc!r}")

    def _log_error(self, msg: str) -> None:
        """Log an internal worker error without raising."""
        try:
            self._logger.error(msg)
        except Exception:
            pass
