import struct
from pathlib import Path
from typing import Dict

import msgspec

from traceml.runtime.config import config
from traceml.runtime.session import get_session_id
from traceml.transport.distributed import get_ddp_info


def _rank_suffix() -> str:
    """
    Return a rank suffix for filesystem isolation.

    NOTE:
    This currently uses LOCAL_RANK via get_ddp_info(), which is sufficient
    for single-node multi-GPU DDP.

    TODO:
    Switch to global RANK when enabling multi-node support to avoid
    cross-node file collisions.
    """
    _, local_rank, _ = get_ddp_info()
    return f"rank_{local_rank}"


class DatabaseWriter:
    """
    Incrementally writes Database tables to disk in length-prefixed MessagePack format.

    This writer is designed to work with bounded deques where old entries
    may be evicted at any time. Uses monotonic append counters for O(1)
    new-row detection.

    Design principles
    -----------------
    - Append-only MessagePack files (stream-friendly, crash-safe)
    - Delta writes only (no full table rewrites)
    - Safe under deque eviction
    - DDP-safe: each rank writes to its own directory
    - Low overhead: flush frequency is throttled locally

    Directory layout
    ----------------
        <logs_dir>/<session_id>/data/<rank_suffix>/<sampler_name>/<table>.msgpack
    """

    def __init__(self, db, sampler_name: str, flush_every: int = 100):
        """
        Parameters
        ----------
        db : Database
            In-memory database instance to flush.
        sampler_name : str
            Name of the sampler owning this writer.
        flush_every : int, optional
            Flush once every N calls to `flush()` (default: 100).
        """
        self.db = db
        self.flush_every = max(1, int(flush_every))
        self._flush_counter = 0

        session_id = get_session_id()
        rank_dir = _rank_suffix()

        # Pathlib-based log root
        self.logs_dir = (
            Path(config.logs_dir)
            / session_id
            / "data"
            / rank_dir
            / sampler_name
        )

        # Tracks the last written append-count per table.
        # Used for O(1) new-row detection via db.get_append_count().
        self._last_written_seq: Dict[str, int] = {}
        self.encoder = msgspec.msgpack.Encoder()

    def flush(self):
        """
        Flush new records from all tables to disk.

        Notes
        -----
        - This method may be called very frequently (e.g., every training step).
        - Actual disk I/O is throttled via `flush_every`.
        - Only records that were not written in previous flushes are appended.
        """
        if not config.enable_logging:
            return

        self._flush_counter += 1
        if self._flush_counter % self.flush_every != 0:
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        for table_name, rows in self.db.all_tables().items():
            if not rows:
                continue

            total = self.db.get_append_count(table_name)
            last_seq = self._last_written_seq.get(table_name, 0)
            new_count = total - last_seq

            # O(1) fast path: nothing new since last write
            if new_count <= 0:
                continue

            n = len(rows)
            if new_count >= n:
                # First flush or eviction — write entire deque
                new_rows = list(rows)
            else:
                # Slice only the tail (new rows)
                new_rows = [rows[i] for i in range(n - new_count, n)]

            path = self.logs_dir / f"{table_name}.msgpack"
            with open(path, "ab") as f:
                for r in new_rows:
                    payload = self.encoder.encode(r)
                    f.write(struct.pack("!I", len(payload)))
                    f.write(payload)

            self._last_written_seq[table_name] = total
