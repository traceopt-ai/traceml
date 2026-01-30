import json
from pathlib import Path
from typing import Any, Dict

from traceml.config import config
from traceml.distributed import get_ddp_info
from traceml.session import get_session_id


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
    """ "
    Incrementally writes Database tables to disk in JSONL format.

    This writer is designed to work with bounded deques where old entries
    may be evicted at any time. As a result, we do NOT rely on positional
    indices for incremental writes.

    Design principles
    -----------------
    - Append-only JSONL files (stream-friendly, crash-safe)
    - Delta writes only (no full table rewrites)
    - Safe under deque eviction
    - DDP-safe: each rank writes to its own directory
    - Low overhead: flush frequency is throttled locally

    Directory layout
    ----------------
        <logs_dir>/<session_id>/data/<rank_suffix>/<sampler_name>/<table>.jsonl
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

        # Tracks the *last written record object* per table.
        # Used instead of positional offsets because deques may evict data.
        self._last_written_record: Dict[str, Any] = {}

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

            path = self.logs_dir / f"{table_name}.jsonl"
            last_written = self._last_written_record.get(table_name)

            # Collect unseen rows by scanning from the tail backwards.
            new_rows = []
            for r in reversed(rows):
                if last_written is not None and r == last_written:
                    break
                new_rows.append(r)

            if not new_rows:
                continue

            # Restore original order before writing
            new_rows.reverse()

            with open(path, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")

            # Track the newest record we just wrote
            self._last_written_record[table_name] = new_rows[-1]
