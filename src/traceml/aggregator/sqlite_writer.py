"""
SQLite telemetry writer for TraceML (simple + sampler metadata).

This module provides an asynchronous, low-overhead SQLite writer intended for
telemetry persistence in the TraceML aggregator process.

Design goals
-----------
- Non-blocking ingestion: `ingest()` never blocks the caller.
- Simple timing model: writer wakes every `flush_interval_sec` and flushes.
- Single-writer SQLite: one background thread owns the sqlite connection.
- Bounded memory: internal queue is bounded; overflow is dropped (telemetry-first).
- Minimal schema + filterability:
  Store raw MessagePack payloads, but also persist rank + sampler columns so you
  can query by sampler without decoding everything.

Storage
-------
One SQLite DB file per session (recommended path: session_id/<session>.db).

Table: raw_messages
- id         INTEGER PRIMARY KEY AUTOINCREMENT
- recv_ts_ns INTEGER NOT NULL
- rank       INTEGER NULL
- sampler    TEXT NULL
- payload_mp BLOB NOT NULL

Index:
- (sampler, rank, id) to support fast streaming reads per sampler/rank.

Usage (Aggregator fan-out)
--------------------------
1) Create and start writer in the aggregator process:
    writer = SQLiteWriterSimple(SQLiteWriterConfig(path=".../session.db"), logger=logger)
    writer.start()

2) In your aggregator drain loop (single reader of TCPServer queue):
    for msg in tcp_server.poll():
        store.ingest(msg)       # existing bounded in-memory store
        writer.ingest(msg)      # enqueue for disk persistence (non-blocking)

3) Stop on shutdown:
    writer.stop()

Notes
-----
- Best-effort: if disk falls behind, messages may be dropped (queue overflow).
- This writer does not depend on UI/RemoteDBStore/TCP.
- Depends only on stdlib sqlite3 + msgspec (already in your deps).
"""

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import msgspec

from traceml.loggers.error_log import get_error_logger


@dataclass(frozen=True)
class SQLiteWriterConfig:
    """
    Configuration for SQLiteWriterSimple.

    Parameters
    ----------
    path:
        Output SQLite DB path (one DB per session is recommended).
    enabled:
        If False, this writer becomes a no-op.
    max_queue:
        Maximum number of pending messages buffered in RAM before dropping.
    flush_interval_sec:
        Writer wakes up every X seconds and flushes pending messages.
    max_flush_items:
        Upper bound on how many messages are written per wake-up.
    synchronous:
        SQLite synchronous level. "NORMAL" is a good default for telemetry.
        Use "FULL" for strongest durability at higher overhead.
    """

    path: str
    enabled: bool = True
    max_queue: int = 50_000
    flush_interval_sec: float = 0.5
    max_flush_items: int = 20_000
    synchronous: str = "NORMAL"


class SQLiteWriterSimple:
    """
    Asynchronous SQLite telemetry writer (sleep-based).

    API
    ---
    - start(): start writer thread
    - ingest(msg): enqueue a dict message (non-blocking)
    - stop(): request stop and flush best-effort

    Threading
    ---------
    - Any thread may call ingest().
    - The writer thread is the only thread that touches sqlite.
    """

    def __init__(self, cfg: SQLiteWriterConfig, logger: Optional[Any] = None):
        self._cfg = cfg
        self._logger = get_error_logger("TraceML-SQLiteWriterSimple")

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=int(cfg.max_queue)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-SQLiteWriter",
            daemon=True,
        )
        self._started = False

        self._encoder = msgspec.msgpack.Encoder()

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
        except queue.Full:
            self._dropped += 1

    def stop(self, timeout_sec: float = 2.0) -> None:
        """
        Request stop and flush best-effort.

        The writer thread is daemonized; shutdown should never hang the process.
        """
        if not self._cfg.enabled:
            return
        self._stop.set()
        if self._started:
            self._thread.join(timeout=float(timeout_sec))

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

    # -------------------- internal --------------------

    def _log_error(self, msg: str) -> None:
        self._last_error = msg
        if self._logger is not None:
            try:
                self._logger.error(msg)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
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

    def _iter_payload_dicts(self, m: Any):
        if m is None:
            return
        if isinstance(m, list):
            for it in m:
                if isinstance(it, dict):
                    yield it
        elif isinstance(m, dict):
            yield m

    def _flush_once(self, conn: sqlite3.Connection) -> None:
        """
        Drain up to max_flush_items from the queue and write them to SQLite.
        """
        items = self._drain_nowait(self._cfg.max_flush_items)
        if not items:
            return

        rows: list[tuple[int, Optional[int], Optional[str], bytes]] = []
        for m in items:
            for payload_dict in self._iter_payload_dicts(m):
                try:
                    recv_ts_ns = time.time_ns()
                    rank, sampler = self._extract_rank_sampler(payload_dict)
                    payload = self._encoder.encode(payload_dict)
                    rows.append((recv_ts_ns, rank, sampler, payload))
                except Exception:
                    continue

        if not rows:
            return

        try:
            conn.execute("BEGIN;")
            conn.executemany(
                "INSERT INTO raw_messages(recv_ts_ns, rank, sampler, payload_mp) "
                "VALUES (?, ?, ?, ?);",
                rows,
            )
            conn.execute("COMMIT;")
            self._written += len(rows)
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            self._log_error(f"[TraceML] SQLiteWriter flush failed: {e}")

    def _run(self) -> None:
        """
        Writer thread loop:
        - connect + init schema
        - sleep for flush_interval
        - flush pending messages
        - on stop: flush a final time
        """
        try:
            conn = self._connect()
            self._init_schema(conn)
        except Exception as e:
            self._log_error(f"[TraceML] SQLiteWriter init failed: {e}")
            return

        interval = float(self._cfg.flush_interval_sec)

        try:
            while not self._stop.is_set():
                time.sleep(interval)
                self._flush_once(conn)

            # Best-effort final flush on stop
            self._flush_once(conn)
        finally:
            try:
                conn.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE);"
                )  # merges + truncates WAL
                conn.close()
            except Exception:
                pass
