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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from traceml_ai.aggregator.sqlite_writers import process as process_sql_writer
from traceml_ai.aggregator.sqlite_writers import (
    runtime_environment as runtime_environment_sql_writer,
)
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
from traceml_ai.telemetry.envelope import (
    TelemetryEnvelope,
    normalize_telemetry_envelope,
)
from traceml_ai.telemetry.retention import (
    DEFAULT_HISTORY_RETENTION_S,
    retention_cutoff_recv_ts_ns,
)

_PROJECTION_WRITERS = [
    runtime_environment_sql_writer,
    system_sql_writer,
    process_sql_writer,
    step_time_sql_writer,
    step_memory_sql_writer,
    stdout_stderr_sql_writer,
]

_HISTORY_TABLES = (
    "system_samples",
    "system_gpu_samples",
    "process_samples",
    "step_time_samples",
    "step_memory_samples",
    "stdout_stderr_samples",
)
_STEP_HISTORY_TABLES = ("step_time_samples", "step_memory_samples")
_TIME_HISTORY_TABLES = tuple(
    table for table in _HISTORY_TABLES if table not in _STEP_HISTORY_TABLES
)
_HISTORY_ROW_POSITIONS = {
    "system_samples": (7, None),
    "system_gpu_samples": (7, None),
    "process_samples": (8, None),
    "step_time_samples": (8, 10),
    "step_memory_samples": (8, 11),
    "stdout_stderr_samples": (2, None),
}
_RETENTION_PRUNE_INTERVAL_S = 60.0


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
    history_retention_s: float = DEFAULT_HISTORY_RETENTION_S
    synchronous: str = "NORMAL"


@dataclass(frozen=True)
class SQLiteFinalizeResult:
    """Outcome of closing the SQLite history writer at end of run."""

    ok: bool
    elapsed_sec: float
    enqueued: int
    written: int
    dropped: int
    queue_size: int
    checkpoint_ok: bool
    error: Optional[str] = None
    prune_ok: bool = True
    prune_error: Optional[str] = None
    checkpoint_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for diagnostics."""
        return {
            "ok": bool(self.ok),
            "elapsed_sec": float(self.elapsed_sec),
            "enqueued": int(self.enqueued),
            "written": int(self.written),
            "dropped": int(self.dropped),
            "queue_size": int(self.queue_size),
            "checkpoint_ok": bool(self.checkpoint_ok),
            "error": self.error,
            "prune_ok": bool(self.prune_ok),
            "prune_error": self.prune_error,
            "checkpoint_error": self.checkpoint_error,
        }


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
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-SQLiteWriter",
            daemon=True,
        )
        self._started = False
        self._accepting = True

        # Stats (best-effort; telemetry doesn't need perfect atomicity)
        self._enqueued = 0
        self._dropped = 0
        self._written = 0
        self._last_prune_monotonic: Optional[float] = None
        self._deleted_through_step: Optional[int] = None
        self._deleted_through_ts_s: Optional[float] = None
        self._last_error: Optional[str] = None
        self._finalize_result: Optional[SQLiteFinalizeResult] = None

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
        if (
            not self._cfg.enabled
            or not self._accepting
            or self._stop.is_set()
            or msg is None
        ):
            return
        try:
            self._q.put_nowait(msg)
            self._enqueued += 1
            self._wake.set()
        except queue.Full:
            self._dropped += 1

    def force_flush(self, timeout_sec: float = 5.0) -> bool:
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
        if self._closed.is_set():
            return True
        if self._stop.is_set():
            return False

        done = threading.Event()
        barrier = _FlushBarrier(done=done)

        try:
            self._q.put(barrier, timeout=float(timeout_sec))
            self._wake.set()
        except queue.Full:
            return False

        return done.wait(timeout=float(timeout_sec))

    def finalize(self, timeout_sec: float = 300.0) -> SQLiteFinalizeResult:
        """
        Stop accepting writes, drain queued telemetry, checkpoint WAL, and close.

        ``force_flush`` is for on-demand readers while the aggregator keeps
        running. ``finalize`` is the end-of-run close path: after this call the
        writer is closed and summary generation can safely open a new SQLite
        reader without racing the writer connection.
        """
        start = time.monotonic()
        if not self._cfg.enabled:
            return SQLiteFinalizeResult(
                ok=True,
                elapsed_sec=0.0,
                enqueued=self._enqueued,
                written=self._written,
                dropped=self._dropped,
                queue_size=0,
                checkpoint_ok=True,
                error=None,
            )
        if not self._started:
            return SQLiteFinalizeResult(
                ok=True,
                elapsed_sec=0.0,
                enqueued=self._enqueued,
                written=self._written,
                dropped=self._dropped,
                queue_size=self._q.qsize(),
                checkpoint_ok=True,
                error=None,
            )

        self._accepting = False
        self._stop.set()
        self._wake.set()

        if not self._closed.wait(timeout=float(timeout_sec)):
            error = (
                "Timed out while finalizing SQLite history writer "
                f"after {float(timeout_sec):.1f}s."
            )
            self._log_error(f"[TraceML] {error}")
            return SQLiteFinalizeResult(
                ok=False,
                elapsed_sec=time.monotonic() - start,
                enqueued=self._enqueued,
                written=self._written,
                dropped=self._dropped,
                queue_size=self._q.qsize(),
                checkpoint_ok=False,
                error=error,
            )

        result = self._finalize_result
        if result is None:
            result = SQLiteFinalizeResult(
                ok=False,
                elapsed_sec=0.0,
                enqueued=self._enqueued,
                written=self._written,
                dropped=self._dropped,
                queue_size=self._q.qsize(),
                checkpoint_ok=False,
                error="SQLite writer closed without a finalization result.",
            )
        return replace(result, elapsed_sec=time.monotonic() - start)

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

    def _drop_expired_projection_rows(
        self,
        projection_rows: dict[Any, dict[str, list[tuple]]],
    ) -> None:
        """Drop known-old history rows before they reach SQLite."""
        for rows_by_table in projection_rows.values():
            for table, rows in rows_by_table.items():
                positions = _HISTORY_ROW_POSITIONS.get(table)
                if positions is None:
                    continue
                sample_position, step_position = positions

                def keep(row: tuple) -> bool:
                    if step_position is not None:
                        step = row[step_position]
                        if (
                            step is not None
                            and self._deleted_through_step is not None
                        ):
                            return int(step) > self._deleted_through_step
                    sample_ts_s = row[sample_position]
                    return (
                        sample_ts_s is None
                        or self._deleted_through_ts_s is None
                        or float(sample_ts_s) > self._deleted_through_ts_s
                    )

                rows_by_table[table] = [row for row in rows if keep(row)]

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
        *,
        prune: bool = True,
    ) -> None:
        """Write prepared projection rows in one SQLite transaction."""
        self._drop_expired_projection_rows(projection_rows)
        row_count = self._projection_row_count(projection_rows)
        if row_count <= 0:
            return

        conn.execute("BEGIN;")

        for writer in _PROJECTION_WRITERS:
            writer.insert_rows(conn, projection_rows.get(writer, {}))

        pruned_through = None
        if prune and self._retention_prune_due():
            pruned_through = self._prune_all_retained_rows(conn)

        conn.execute("COMMIT;")
        if pruned_through is not None:
            self._advance_retention_floor(pruned_through)
        self._written += row_count

    def _advance_retention_floor(
        self,
        frontier: tuple[Optional[int], Optional[float]],
    ) -> None:
        """Remember committed deletion boundaries for future inserts."""
        step, sample_ts_s = frontier
        if step is not None:
            self._deleted_through_step = max(
                step, self._deleted_through_step or step
            )
        if sample_ts_s is not None:
            self._deleted_through_ts_s = max(
                sample_ts_s, self._deleted_through_ts_s or sample_ts_s
            )

    def _prune_all_retained_rows(
        self,
        conn: sqlite3.Connection,
    ) -> Optional[tuple[Optional[int], Optional[float]]]:
        """Prune history through one Step Time/Step Memory aligned frontier."""
        owns_transaction = not conn.in_transaction
        pruned_through = None
        if owns_transaction:
            conn.execute("BEGIN;")
        try:
            watermark = self._history_watermark_recv_ts_ns(conn)
            if watermark is not None:
                cutoff = retention_cutoff_recv_ts_ns(
                    watermark,
                    self._cfg.history_retention_s,
                )
                frontier = self._retention_frontier(
                    conn,
                    cutoff_recv_ts_ns=cutoff,
                )
                if frontier is not None:
                    pruned_through = frontier
                    step, boundary_ts_s = frontier
                    if step is not None:
                        for table in _STEP_HISTORY_TABLES:
                            self._prune_history_table(
                                conn,
                                table=table,
                                where_sql="step <= ?",
                                where_params=(step,),
                            )
                    for table in _TIME_HISTORY_TABLES:
                        if boundary_ts_s is None:
                            where_sql = "recv_ts_ns < ?"
                            where_params: tuple[Any, ...] = (cutoff,)
                        else:
                            # Periodic history follows the shared step timestamp.
                            # Malformed rows still age out by receive time.
                            where_sql = (
                                "sample_ts_s <= ? OR "
                                "(sample_ts_s IS NULL AND recv_ts_ns < ?)"
                            )
                            where_params = (boundary_ts_s, cutoff)
                        self._prune_history_table(
                            conn,
                            table=table,
                            where_sql=where_sql,
                            where_params=where_params,
                        )
            if owns_transaction:
                conn.execute("COMMIT;")
                if pruned_through is not None:
                    self._advance_retention_floor(pruned_through)
        except Exception:
            if owns_transaction and conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            self._last_prune_monotonic = time.monotonic()
        return pruned_through

    def _retention_prune_due(self) -> bool:
        """Return whether the low-frequency retention pass is due."""
        if self._last_prune_monotonic is None:
            return True
        return (
            time.monotonic() - self._last_prune_monotonic
            >= _RETENTION_PRUNE_INTERVAL_S
        )

    @staticmethod
    def _history_watermark_recv_ts_ns(
        conn: sqlite3.Connection,
    ) -> Optional[int]:
        selects = " UNION ALL ".join(
            f"SELECT MAX(recv_ts_ns) AS value FROM {table}"
            for table in _HISTORY_TABLES
        )
        row = conn.execute(f"SELECT MAX(value) FROM ({selects});").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _expected_step_ranks(conn: sqlite3.Connection) -> Optional[int]:
        """Return the stable world size carried by step telemetry."""
        row = conn.execute(
            """
            SELECT MAX(value)
            FROM (
                SELECT MAX(world_size) AS value FROM step_time_samples
                UNION ALL
                SELECT MAX(world_size) AS value FROM step_memory_samples
            );
            """
        ).fetchone()
        if row and row[0] is not None and int(row[0]) > 0:
            return int(row[0])

        row = conn.execute(
            """
            SELECT COUNT(DISTINCT rank_id)
            FROM (
                SELECT COALESCE(global_rank, rank) AS rank_id
                FROM step_time_samples
                UNION ALL
                SELECT COALESCE(global_rank, rank) AS rank_id
                FROM step_memory_samples
            )
            WHERE rank_id IS NOT NULL;
            """
        ).fetchone()
        observed = int(row[0] or 0) if row else 0
        return observed or None

    @staticmethod
    def _highest_aligned_step(
        conn: sqlite3.Connection,
        *,
        table: str,
        expected_ranks: int,
    ) -> Optional[int]:
        """Return the highest step reported by every expected rank."""
        if table not in _STEP_HISTORY_TABLES:
            raise ValueError(f"Unknown step history table: {table}")
        complete_row = (
            "AND events_json IS NOT NULL AND events_json != ''"
            if table == "step_time_samples"
            else ""
        )
        row = conn.execute(
            f"""
            SELECT MAX(step)
            FROM (
                SELECT step
                FROM {table}
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                  AND step IS NOT NULL
                  {complete_row}
                GROUP BY step
                HAVING COUNT(DISTINCT COALESCE(global_rank, rank)) = ?
            );
            """,
            (int(expected_ranks),),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _highest_aged_step(
        conn: sqlite3.Connection,
        *,
        cutoff_recv_ts_ns: int,
    ) -> Optional[int]:
        """Return the highest step whose Step Time and Memory rows are old."""
        row = conn.execute(
            """
            SELECT MAX(step)
            FROM (
                SELECT step
                FROM (
                    SELECT step, recv_ts_ns FROM step_time_samples
                    UNION ALL
                    SELECT step, recv_ts_ns FROM step_memory_samples
                )
                WHERE step IS NOT NULL
                GROUP BY step
                HAVING MAX(recv_ts_ns) < ?
            );
            """,
            (int(cutoff_recv_ts_ns),),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @classmethod
    def _retention_frontier(
        cls,
        conn: sqlite3.Connection,
        *,
        cutoff_recv_ts_ns: int,
    ) -> Optional[tuple[Optional[int], Optional[float]]]:
        """Return the shared step/timestamp frontier, or a time-only fallback."""
        expected_ranks = cls._expected_step_ranks(conn)
        if expected_ranks is None:
            return None, None

        highest_step_time = cls._highest_aligned_step(
            conn,
            table="step_time_samples",
            expected_ranks=expected_ranks,
        )
        highest_step_memory = cls._highest_aligned_step(
            conn,
            table="step_memory_samples",
            expected_ranks=expected_ranks,
        )
        highest_aged_step = cls._highest_aged_step(
            conn,
            cutoff_recv_ts_ns=cutoff_recv_ts_ns,
        )
        candidates = (
            highest_step_time,
            highest_step_memory,
            highest_aged_step,
        )
        if any(step is None for step in candidates):
            return None
        frontier_step = min(
            int(step) for step in candidates if step is not None
        )
        row = conn.execute(
            """
            SELECT MAX(sample_ts_s)
            FROM step_time_samples
            WHERE step = ?;
            """,
            (frontier_step,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        return frontier_step, float(row[0])

    @staticmethod
    def _prune_history_table(
        conn: sqlite3.Connection,
        *,
        table: str,
        where_sql: str,
        where_params: tuple[Any, ...],
    ) -> None:
        """Prune one history table through the shared frontier."""
        if table not in _HISTORY_TABLES:
            raise ValueError(f"Unknown telemetry history table: {table}")
        conn.execute(
            f"DELETE FROM {table} WHERE {where_sql};",
            where_params,
        )

    @staticmethod
    def _init_retention_indexes(conn: sqlite3.Connection) -> None:
        """Create indexes used by periodic retention queries."""
        for table in _HISTORY_TABLES:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_recv_ts "
                f"ON {table}(recv_ts_ns);"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_sample_ts_global "
                f"ON {table}(sample_ts_s, id);"
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
        run_start = time.monotonic()
        conn: Optional[sqlite3.Connection] = None
        fatal_error: Optional[str] = None
        prune_error: Optional[str] = None
        checkpoint_error: Optional[str] = None
        checkpoint_ok = False
        try:
            conn = self._connect()
            for writer in _PROJECTION_WRITERS:
                writer.init_schema(conn)
            self._init_retention_indexes(conn)
        except Exception as exc:
            fatal_error = f"SQLiteWriter init failed: {exc}"
            self._log_error(f"[TraceML] {fatal_error}")
            self._finalize_result = self._build_finalize_result(
                elapsed_sec=time.monotonic() - run_start,
                checkpoint_ok=False,
                error=fatal_error,
            )
            self._closed.set()
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
            try:
                self._prune_all_retained_rows(conn)
            except Exception as exc:
                prune_error = f"Final SQLite retention prune failed: {exc}"
                self._log_error(f"[TraceML] {prune_error}")
        finally:
            try:
                if conn is not None:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    checkpoint_ok = True
                    conn.close()
            except Exception as exc:
                checkpoint_ok = False
                checkpoint_error = f"SQLite checkpoint/close failed: {exc}"
                fatal_error = checkpoint_error
                self._log_error(f"[TraceML] {checkpoint_error}")
            finally:
                self._finalize_result = self._build_finalize_result(
                    elapsed_sec=time.monotonic() - run_start,
                    checkpoint_ok=checkpoint_ok,
                    error=fatal_error,
                    prune_error=prune_error,
                    checkpoint_error=checkpoint_error,
                )
                self._closed.set()

    def _build_finalize_result(
        self,
        *,
        elapsed_sec: float,
        checkpoint_ok: bool,
        error: Optional[str],
        prune_error: Optional[str] = None,
        checkpoint_error: Optional[str] = None,
    ) -> SQLiteFinalizeResult:
        """Build a finalization result from current writer counters."""
        fatal_error = error or checkpoint_error
        return SQLiteFinalizeResult(
            ok=bool(fatal_error is None and checkpoint_ok),
            elapsed_sec=float(elapsed_sec),
            enqueued=int(self._enqueued),
            written=int(self._written),
            dropped=int(self._dropped),
            queue_size=int(self._q.qsize()),
            checkpoint_ok=bool(checkpoint_ok),
            error=fatal_error,
            prune_ok=prune_error is None,
            prune_error=prune_error,
            checkpoint_error=checkpoint_error,
        )
