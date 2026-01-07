import json
import os
from traceml.session import get_session_id
from traceml.config import config


def _rank_suffix() -> str:
    """Return a stable rank suffix for DDP workers.

    We intentionally *do not* depend on torch.distributed here to avoid any chance
    of interacting with the user's process group. We only read env vars that are
    set by torchrun / torch.distributed.run.

    Returns:
        str: e.g. "rank_0" if LOCAL_RANK is set, otherwise "rank_-1".
    """
    try:
        # torchrun sets LOCAL_RANK for each worker.
        lr = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        lr = -1
    return f"rank_{lr}"



class DatabaseWriter:
    """Writes incremental updates from a Database instance to JSONL files.

    Design goals:
      - Append-only JSONL per table (cheap, streamable).
      - Only write new rows since the last flush (tracked via per-table offsets).
      - DDP-safe: when running under torchrun, each worker writes to a rank-scoped
        folder so multiple ranks never append to the same file.

    Directory layout:

        <logs_dir>/<session_id>/data/<rank_suffix>/<sampler_name>/<table>.jsonl

    """

    def __init__(self, db, sampler_name):
        self.db = db
        session_id = get_session_id()

        # IMPORTANT: rank-scoped folder to avoid file collisions in DDP.
        # This is a minimal change that keeps existing layout compatible.
        rank_dir = _rank_suffix()

        self.logs_dir = os.path.join(
            config.logs_dir, session_id, "data", rank_dir, sampler_name)

        self._last_written = {}  # table_name → index

    def flush(self):
        """Write only new rows from each table to its own file."""
        if not config.enable_logging:
            return

        for table_name, rows in self.db.all_tables().items():
            os.makedirs(self.logs_dir, exist_ok=True)
            path = os.path.join(self.logs_dir, f"{table_name}.jsonl")

            last = self._last_written.get(table_name, 0)
            new_rows = rows[last:]
            if not new_rows:
                continue

            with open(path, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")

            self._last_written[table_name] = len(rows)
