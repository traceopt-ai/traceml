"""
SQLite telemetry sink for TraceML.

This module provides a small, production-friendly component that persists
telemetry messages to a local SQLite database without slowing training or UI.

Design goals
-----------
- Non-blocking ingestion: `ingest()` never blocks the caller.
- Single-writer model: one background thread owns the sqlite connection.
- Batching: amortize disk sync cost by committing in batches.
- Bounded memory: internal queue is bounded; overflow is dropped (telemetry-first).

Storage model (v0)
------------------
We store raw messages as MessagePack BLOBs so schema changes do not require
DB migrations. We  compute summaries later by reading the DB and decoding
the payloads.

Notes
-----
- This is intentionally a "sink": it does not know about UI, RemoteDBStore, or TCP.
- It is safe to call `ingest()` from any thread.
- SQLiteWriter does not raise on ingest; it drops on overflow by default.
"""

import queue
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import msgspec


@dataclass(frozen=True)
class SQLiteWriterConfig:
    """
    Configuration for SQLiteWriter.

    Parameters
    ----------
    path:
        File path to the SQLite database (e.g., "traceml_run.db").
    session_id:
        Logical run identifier. If you don't have one, pass a timestamp string
        from the aggregator on startup.
    enabled:
        If False, writer becomes a no-op.
    max_queue:
        Maximum number of pending messages buffered in RAM before dropping.
    flush_interval_sec:
        Max time between commits. Lower => more durable, slightly more overhead.
    batch_size:
        Target number of rows to insert per transaction.
    synchronous:
        SQLite synchronous setting. "NORMAL" is a good balance for telemetry.
        Use "FULL" for strongest durability at higher overhead.
    """

    path: str
    session_id: str
    enabled: bool = True
    max_queue: int = 50_000
    flush_interval_sec: float = 0.2
    batch_size: int = 5_000
    synchronous: str = "NORMAL"


class SQLiteWriter:
    """
    Asynchronous SQLite writer for telemetry messages.

    Public API
    ----------
    - start(): start background writer thread
    - ingest(msg): non-blocking enqueue of a dict message
    - stop(): request stop and flush

    Threading model
    ---------------
    - Any thread may call ingest().
    - A single background thread owns the sqlite connection and performs writes.
    """

    def __init__(
        self,
        cfg: SQLiteWriterConfig,
        logger: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._logger = logger

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

        # Stats (thread-safe enough for telemetry usage)
        self._enqueued = 0
        self._dropped = 0
        self._written = 0
        self._last_error: Optional[str] = None

        # Local encoder (fast, no schema required)
        self._encoder = msgspec.msgpack.Encoder()

    def start(self) -> None:
        """Start the background writer thread (idempotent)."""
        if not self._cfg.enabled:
            return
        if self._started:
            return
        self._started = True
        self._thread.start()

    def ingest(self, msg: Dict[str, Any]) -> None:
        """
        Enqueue a message for persistence (non-blocking).

        Behavior
        --------
        - Never blocks the caller.
        - If queue is full, message is dropped and a counter is incremented.
        - Does not raise.
        """
        if not self._cfg.enabled:
            return
        if msg is None:
            return

        try:
            self._q.put_nowait(msg)
            self._enqueued += 1
        except queue.Full:
            self._dropped += 1

    def stop(self, timeout_sec: float = 0.5) -> None:
        """
        Stop the writer thread and flush pending messages.

        Parameters
        ----------
        timeout_sec:
            Join timeout. Writer is daemon thread; process will still exit
            even if it can't stop cleanly, but we try.

        Notes
        -----
        Should not block shutdown.
        """
        if not self._cfg.enabled:
            return
        self._stop.set()
        if self._started:
            self._thread.join(timeout=float(timeout_sec))

    def stats(self) -> Dict[str, Any]:
        """
        Return simple counters for observability and debugging.
        """
        return {
            "enabled": self._cfg.enabled,
            "path": self._cfg.path,
            "session_id": self._cfg.session_id,
            "enqueued": self._enqueued,
            "dropped": self._dropped,
            "written": self._written,
            "queue_size": self._q.qsize(),
            "last_error": self._last_error,
        }

    def _log_warning(self, msg: str) -> None:
        if self._logger is not None:
            try:
                self._logger.warning(msg)
            except Exception:
                pass

    def _log_error(self, msg: str) -> None:
        self._last_error = msg
        if self._logger is not None:
            try:
                self._logger.error(msg)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because we only use this connection in the
        # writer thread, but sqlite3 validates thread usage; set False anyway
        # to avoid accidental issues if code evolves.
        conn = sqlite3.connect(
            self._cfg.path,
            isolation_level=None,  # autocommit mode; we control transactions
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(f"PRAGMA synchronous={self._cfg.synchronous};")
        # Keep RAM usage small; negative is KB.
        conn.execute("PRAGMA cache_size=-2000;")  # ~2MB
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_messages (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL,
                recv_ts_ns          INTEGER NOT NULL,
                payload_msgpack     BLOB NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_messages_session_id_id "
            "ON raw_messages(session_id, id);"
        )
