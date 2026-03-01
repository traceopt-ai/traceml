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
