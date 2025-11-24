import json
import os


class DatabaseWriter:
    """
    Writes incremental updates from a Database instance to a file.
    Keeps track of per-table write offsets.
    """

    def __init__(self, db, log_dir: str):
        self.db = db
        self.log_dir = log_dir
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
