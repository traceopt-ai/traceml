import json
import os
from traceml.session import get_session_id
from traceml.config import config


class DatabaseWriter:
    """
    Writes incremental updates from a Database instance to a file.
    Keeps track of per-table write offsets.
    """

    def __init__(self, db, sampler_name):
        self.db = db
        session_id = get_session_id()
        self.logs_dir = os.path.join(config.logs_dir, session_id, "data", sampler_name)
        self._last_written = {}  # table_name → index

    def flush(self):
        """Write only new rows from each table to its own file."""
        if config.enable_logging:
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
