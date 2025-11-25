import json
import os
from traceml.session import get_session_id

class DatabaseWriter:
    """
    Writes incremental updates from a Database instance to a file.
    Keeps track of per-table write offsets.
    """

    def __init__(self, db, sampler_name, log_dir: str="./logs"):
        self.db = db
        session_id = get_session_id()
        self.log_dir = os.path.join(log_dir, session_id, "data", sampler_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self._last_written = {}  # table_name → index

    def flush(self):
        """Write only new rows from each table to its own file."""
        for table_name, rows in self.db.all_tables().items():

            path = os.path.join(self.log_dir, f"{table_name}.jsonl")
            last = self._last_written.get(table_name, 0)
            new_rows = rows[last:]
            if not new_rows:
                continue
            with open(path, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")

            self._last_written[table_name] = len(rows)
