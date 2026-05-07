"""Asynchronous SQLite telemetry writer."""

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from traceml.aggregator.sqlite_writers import process as process_sql_writer
from traceml.aggregator.sqlite_writers import (
    stdout_stderr as stdout_stderr_sql_writer,
)
from traceml.aggregator.sqlite_writers import (
    step_memory as step_memory_sql_writer,
)
from traceml.aggregator.sqlite_writers import step_time as step_time_sql_writer
from traceml.aggregator.sqlite_writers import system as system_sql_writer
from traceml.loggers.error_log import get_error_logger
from traceml.utils.msgpack_codec import Encoder as MsgpackEncoder

_PROJECTION_WRITERS = [
    system_sql_writer,
    process_sql_writer,
    step_time_sql_writer,
    step_memory_sql_writer,
    stdout_stderr_sql_writer,
]


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

        self._encoder = MsgpackEncoder()

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

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Create the base raw message schema and indexes if needed."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns INTEGER NOT NULL,
                rank       INTEGER,
                sampler    TEXT,
                payload_mp BLOB NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_sampler_rank_id
            ON raw_messages(sampler, rank, id);
            """
        )

    def _drain_nowait(self, max_items: int) -> list[Dict[str, Any]]:
        """Drain up to ``max_items`` messages from the in-memory queue."""
        items: list[Dict[str, Any]] = []
        for _ in range(int(max_items)):
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def _extract_rank_sampler(
        self, msg: Dict[str, Any]
    ) -> tuple[Optional[int], Optional[str]]:
        """Extract best-effort ``rank`` and ``sampler`` metadata from a message."""
        rank_val = msg.get("rank", None)
        sampler_val = msg.get("sampler", None)

        rank: Optional[int]
        try:
            rank = int(rank_val) if rank_val is not None else None
        except Exception:
            rank = None

        sampler: Optional[str]
        try:
            sampler = str(sampler_val) if sampler_val is not None else None
        except Exception:
            sampler = None

        return rank, sampler

    def _iter_payload_dicts(self, msg: Any) -> Iterator[Dict[str, Any]]:
        """
        Yield payload dicts from a single message or batch.

        Accepted forms:
        - dict
        - list[dict]
        """
        if msg is None:
            return

        if isinstance(msg, list):
            for item in msg:
                if isinstance(item, dict):
                    yield item
        elif isinstance(msg, dict):
            yield msg

    def _collect_flush_rows(
        self,
        items: list[Dict[str, Any]],
    ) -> tuple[
        list[tuple[int, Optional[int], Optional[str], bytes]],
        dict[Any, dict[str, list[tuple]]],
    ]:
        """
        Convert queued payloads into raw SQLite rows and projection rows.

        All decodable payloads are preserved in ``raw_messages``. A selected
        subset of samplers is also expanded into structured projection tables.
        """
        raw_rows: list[tuple[int, Optional[int], Optional[str], bytes]] = []
        projection_rows: dict[Any, dict[str, list[tuple]]] = {
            writer: {} for writer in _PROJECTION_WRITERS
        }

        for item in items:
            for payload_dict in self._iter_payload_dicts(item):
                try:
                    recv_ts_ns = time.time_ns()
                    rank, sampler = self._extract_rank_sampler(payload_dict)
                    payload = self._encoder.encode(payload_dict)
                    raw_rows.append((recv_ts_ns, rank, sampler, payload))

                    for writer in _PROJECTION_WRITERS:
                        if not writer.accepts_sampler(sampler):
                            continue

                        rows_by_table = writer.build_rows(
                            payload_dict=payload_dict,
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

        return raw_rows, projection_rows

    def _write_flush_rows(
        self,
        conn: sqlite3.Connection,
        raw_rows: list[tuple[int, Optional[int], Optional[str], bytes]],
        projection_rows: dict[Any, dict[str, list[tuple]]],
    ) -> None:
        """
        Write prepared raw rows and projection rows in one SQLite transaction.
        """
        if not raw_rows:
            return

        conn.execute("BEGIN;")

        conn.executemany(
            "INSERT INTO raw_messages(recv_ts_ns, rank, sampler, payload_mp) "
            "VALUES (?, ?, ?, ?);",
            raw_rows,
        )

        for writer in _PROJECTION_WRITERS:
            writer.insert_rows(conn, projection_rows[writer])

        conn.execute("COMMIT;")
        self._written += len(raw_rows)

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

            raw_rows, projection_rows = self._collect_flush_rows(batch)
            if not raw_rows:
                return

            try:
                self._write_flush_rows(conn, raw_rows, projection_rows)
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
        - Initialize raw + projection schemas
        - Sleep for ``flush_interval_sec``
        - Flush pending messages
        - On stop: perform one final best-effort flush
        """
        try:
            conn = self._connect()
            self._init_schema(conn)
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
