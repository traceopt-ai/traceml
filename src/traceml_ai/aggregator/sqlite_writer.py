# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous SQLite telemetry writer."""

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from traceml_ai.aggregator.sqlite_writers import process as process_sql_writer
from traceml_ai.aggregator.sqlite_writers import (
    stdout_stderr as stdout_stderr_sql_writer,
)
from traceml_ai.aggregator.sqlite_writers import (
    step_memory as step_memory_sql_writer,
)
from traceml_ai.aggregator.sqlite_writers import (
    step_time as step_time_sql_writer,
)
from traceml_ai.aggregator.sqlite_writers import system as system_sql_writer
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.reporting.config import (
    DEFAULT_SUMMARY_WINDOW_ROWS,
    summary_retention_rows_for_window,
)
from traceml_ai.telemetry.envelope import (
    TelemetryEnvelope,
    normalize_telemetry_envelope,
)

_PROJECTION_WRITERS = [
    system_sql_writer,
    process_sql_writer,
    step_time_sql_writer,
    step_memory_sql_writer,
    stdout_stderr_sql_writer,
]
_RETENTION_TABLES = frozenset(
    str(table)
    for writer in _PROJECTION_WRITERS
    for table in getattr(writer, "RETENTION_TABLES", ())
)
_RETENTION_PARTITION_SQL = {
    "system_samples": "COALESCE(node_rank, global_rank, 0)",
    "process_samples": "COALESCE(global_rank, rank, 0)",
    "step_time_samples": "COALESCE(global_rank, rank, 0)",
    "step_memory_samples": "COALESCE(global_rank, rank, 0)",
}


@dataclass(frozen=True)
class _FlushBarrier:
    """Queue item used to establish a flush barrier."""

    done: threading.Event


@dataclass(frozen=True)
class SQLiteWriterConfig:
    """Configuration for SQLiteWriterSimple."""

    path: str
    enabled: bool = True
    max_queue: int = 50_000
    flush_interval_sec: float = 0.5
    max_flush_items: int = 20_000
    summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS
    synchronous: str = "NORMAL"


class SQLiteWriterSimple:
    """Asynchronous SQLite telemetry writer."""

    def __init__(
        self, cfg: SQLiteWriterConfig, logger: Optional[Any] = None
    ) -> None:
        self._cfg = cfg
        self._logger = logger or get_error_logger("TraceML-SQLiteWriterSimple")

        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=int(cfg.max_queue))
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-SQLiteWriter",
            daemon=True,
        )
        self._started = False

        # Stats (best-effort; telemetry doesn't need perfect atomicity)
        self._enqueued = 0
        self._dropped = 0
        self._written = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Start the writer thread (idempotent)."""
        if not self._cfg.enabled or self._started:
            return
        self._started = True
        self._thread.start()

    def ingest(self, msg: Dict[str, Any]) -> None:
        """
        Enqueue one telemetry message (non-blocking).

        If the internal queue is full, the message is dropped.
        """
        if not self._cfg.enabled or msg is None:
            return
        try:
            self._q.put_nowait(msg)
            self._enqueued += 1
            self._wake.set()
        except queue.Full:
            self._dropped += 1

    def flush_now(self, timeout_sec: float = 5.0) -> bool:
        """
        Block until all messages enqueued before this call have been flushed.

        Parameters
        ----------
        timeout_sec:
            Maximum time to wait for the flush barrier to be processed.

        Returns
        -------
        bool
            True if the flush barrier was processed in time, otherwise False.

        Notes
        -----
        This method is intended for low-frequency control-plane operations such
        as on-demand final summary generation. It should not be called on every
        training step.
        """
        if not self._cfg.enabled or not self._started:
            return True

        done = threading.Event()
        barrier = _FlushBarrier(done=done)

        try:
            self._q.put(barrier, timeout=float(timeout_sec))
            self._wake.set()
        except queue.Full:
            return False

        return done.wait(timeout=float(timeout_sec))

    def stop(self, timeout_sec: float = 2.0) -> None:
        """
        Request stop and flush best-effort.

        The writer thread is daemonized; shutdown should never hang the process.
        """
        if not self._cfg.enabled:
            return

        self._stop.set()
        self._wake.set()

        if self._started:
            self._thread.join(timeout=float(timeout_sec))
            if self._thread.is_alive():
                self._log_error(
                    "[TraceML] SQLiteWriter thread did not terminate cleanly"
                )

    def stats(self) -> Dict[str, Any]:
        """Return basic counters for debugging/observability."""
        return {
            "enabled": self._cfg.enabled,
            "path": self._cfg.path,
            "enqueued": self._enqueued,
            "dropped": self._dropped,
            "written": self._written,
            "queue_size": self._q.qsize(),
            "last_error": self._last_error,
        }

    def _log_error(self, msg: str) -> None:
        """Log an internal writer error without raising."""
        self._last_error = msg
        if self._logger is not None:
            try:
                self._logger.error(msg)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        """Open and configure the SQLite connection used by the writer thread."""
        Path(self._cfg.path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            self._cfg.path,
            isolation_level=None,  # autocommit; we manage BEGIN/COMMIT
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(f"PRAGMA synchronous={self._cfg.synchronous};")
        conn.execute("PRAGMA cache_size=-2000;")  # ~2MB cache
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _drain_nowait(self, max_items: int) -> list[Dict[str, Any]]:
        """Drain up to ``max_items`` messages from the in-memory queue."""
        items: list[Dict[str, Any]] = []
        for _ in range(int(max_items)):
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def _iter_envelopes(self, msg: Any) -> Iterator[TelemetryEnvelope]:
        """
        Yield normalized telemetry envelopes from a single message or batch.

        Accepted forms:
        - dict
        - list[dict]
        """
        if msg is None:
            return

        if isinstance(msg, list):
            for item in msg:
                envelope = normalize_telemetry_envelope(item)
                if envelope is not None:
                    yield envelope
        elif isinstance(msg, dict):
            envelope = normalize_telemetry_envelope(msg)
            if envelope is not None:
                yield envelope

    def _collect_projection_rows(
        self,
        items: list[Dict[str, Any]],
    ) -> dict[Any, dict[str, list[tuple]]]:
        """Convert queued telemetry payloads into structured projection rows."""
        projection_rows: dict[Any, dict[str, list[tuple]]] = {
            writer: {} for writer in _PROJECTION_WRITERS
        }

        for item in items:
            for envelope in self._iter_envelopes(item):
                try:
                    recv_ts_ns = time.time_ns()
                    sampler = envelope.meta.sampler

                    for writer in _PROJECTION_WRITERS:
                        if not writer.accepts_sampler(sampler):
                            continue

                        rows_by_table = writer.build_rows(
                            envelope=envelope,
                            recv_ts_ns=recv_ts_ns,
                        )
                        for table_name, rows in rows_by_table.items():
                            if not rows:
                                continue
                            projection_rows[writer].setdefault(
                                table_name, []
                            ).extend(rows)

                except Exception:
                    # Best-effort persistence: skip malformed payloads and continue.
                    continue

        return projection_rows

    @staticmethod
    def _projection_row_count(
        projection_rows: dict[Any, dict[str, list[tuple]]],
    ) -> int:
        """Return the total number of structured projection rows prepared."""
        return sum(
            len(rows)
            for rows_by_table in projection_rows.values()
            for rows in rows_by_table.values()
        )

    def _write_projection_rows(
        self,
        conn: sqlite3.Connection,
        projection_rows: dict[Any, dict[str, list[tuple]]],
    ) -> None:
        """Write prepared projection rows in one SQLite transaction."""
        row_count = self._projection_row_count(projection_rows)
        if row_count <= 0:
            return

        conn.execute("BEGIN;")

        for writer in _PROJECTION_WRITERS:
            writer.insert_rows(conn, projection_rows[writer])

        self._prune_retained_rows(conn, projection_rows)

        conn.execute("COMMIT;")
        self._written += row_count

    def _prune_retained_rows(
        self,
        conn: sqlite3.Connection,
        projection_rows: dict[Any, dict[str, list[tuple]]],
    ) -> None:
        """
        Keep bounded history for the high-frequency summary tables.

        The final report reads a fixed summary window. We retain a larger
        buffer in SQLite so long jobs stay bounded while still having enough
        recent rows for aligned multi-rank summaries.
        """
        retention_rows = summary_retention_rows_for_window(
            self._cfg.summary_window_rows
        )

        for writer, rows_by_table in projection_rows.items():
            if not any(rows_by_table.values()):
                continue

            for table in getattr(writer, "RETENTION_TABLES", ()):
                self._prune_table_by_identity(
                    conn,
                    table=str(table),
                    retention_rows=retention_rows,
                )

            if writer is system_sql_writer:
                self._prune_system_gpu_samples_to_retained_system_samples(conn)

    @staticmethod
    def _prune_table_by_identity(
        conn: sqlite3.Connection,
        *,
        table: str,
        retention_rows: int,
    ) -> None:
        """Delete rows outside the retained window for each table identity."""
        if table not in _RETENTION_TABLES:
            raise ValueError(f"Unknown retained SQLite table: {table}")
        partition_sql = _RETENTION_PARTITION_SQL.get(table)
        if partition_sql is None:
            raise ValueError(f"Missing retention identity for table: {table}")

        conn.execute(
            f"""
            DELETE FROM {table}
            WHERE id IN (
                SELECT id FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY {partition_sql}
                            ORDER BY id DESC
                        ) AS row_num
                    FROM {table}
                )
                WHERE row_num > ?
            );
            """,
            (int(retention_rows),),
        )

    @staticmethod
    def _prune_system_gpu_samples_to_retained_system_samples(
        conn: sqlite3.Connection,
    ) -> None:
        """Keep GPU rows only for retained system snapshots."""
        conn.execute(
            """
            DELETE FROM system_gpu_samples AS gpu
            WHERE NOT EXISTS (
                SELECT 1
                FROM system_samples AS sample
                WHERE sample.global_rank IS gpu.global_rank
                  AND sample.node_rank IS gpu.node_rank
                  AND sample.seq IS gpu.seq
            );
            """
        )

    def _flush_once(self, conn: sqlite3.Connection) -> None:
        """
        Drain up to ``max_flush_items`` queued items and write them to SQLite.

        Flush barriers are processed in-order and guarantee that all telemetry
        queued before the barrier has been committed before the barrier is
        acknowledged.
        """
        items = self._drain_nowait(self._cfg.max_flush_items)
        if not items:
            return

        pending_payloads: list[Any] = []

        def _flush_payload_batch(batch: list[Any]) -> None:
            if not batch:
                return

            projection_rows = self._collect_projection_rows(batch)
            if self._projection_row_count(projection_rows) <= 0:
                return

            try:
                self._write_projection_rows(conn, projection_rows)
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                self._log_error(f"[TraceML] SQLiteWriter flush failed: {exc}")

        for item in items:
            if isinstance(item, _FlushBarrier):
                _flush_payload_batch(pending_payloads)
                pending_payloads = []
                item.done.set()
                continue

            if isinstance(item, (dict, list)):
                pending_payloads.append(item)

        _flush_payload_batch(pending_payloads)

    def _run(self) -> None:
        """
        Writer thread loop.

        Flow
        ----
        - Open and configure SQLite
        - Initialize projection schemas
        - Sleep for ``flush_interval_sec``
        - Flush pending messages
        - On stop: perform one final best-effort flush
        """
        try:
            conn = self._connect()
            for writer in _PROJECTION_WRITERS:
                writer.init_schema(conn)
        except Exception as exc:
            self._log_error(f"[TraceML] SQLiteWriter init failed: {exc}")
            return

        interval = float(self._cfg.flush_interval_sec)

        try:
            while not self._stop.is_set():
                self._wake.wait(timeout=interval)
                self._wake.clear()
                self._flush_once(conn)

                # Best-effort final flush on stop.
            while not self._q.empty():
                self._flush_once(conn)
        finally:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.close()
            except Exception:
                pass
