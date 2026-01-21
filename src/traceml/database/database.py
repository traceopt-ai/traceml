from typing import Any, Dict, List, Optional, Deque
from collections import deque
from traceml.database.database_writer import DatabaseWriter


class Database:
    """
    In-memory, bounded database for sampler telemetry.
    Bounded memory per table (keeps only the last N records)
    Fast append (O(1)) and fast drop-oldest (O(1)) via deque(maxlen=...)
    """
    DEFAULT_MAX_ROWS = 3000

    def __init__(self, sampler_name, max_rows: Optional[int] = None):
        self.max_rows: int = int(max_rows) if max_rows is not None else self.DEFAULT_MAX_ROWS
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}")

        self._tables: Dict[str, Deque[Any]] = {}
        self.writer = DatabaseWriter(self, sampler_name=sampler_name)
        self.sender = None
        self.sampler_name = sampler_name


    def create_table(self, name: str) -> Deque[Any]:
        """
        Create a new empty table if not exists.
        Raise ValueError if table already exists.
        """
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists.")
        self._tables[name] = deque(maxlen=self.max_rows)
        return self._tables[name]


    def create_or_get_table(self, name: str) -> Deque[Any]:
        """
        Create table if missing, otherwise return existing table.
        """
        if name not in self._tables:
            self._tables[name] = deque(maxlen=self.max_rows)
        return self._tables[name]


    def add_record(self, table: str, record: Any) -> None:
        """
        Add a single record to a table.
        If the table does not exist, it is created automatically.
        When the table reaches max_rows, oldest items are dropped automatically.
        """
        rows = self._tables.get(table)
        if rows is None:
            rows = self.create_or_get_table(table)
        rows.append(record)


    def get_last_record(self, table: str) -> Optional[Any]:
        rows = self._tables.get(table)
        if not rows:
            return None
        return next(reversed(rows))


    def all_tables(self) -> Dict[str, Deque[Any]]:
        """
        Return a snapshot of all tables as lists.

        Important:
          - This copies data (deque -> list) to provide list semantics to callers.
          - Prefer using get_table() for iteration without copying if needed.
        """
        return self._tables


    def get_table(self, name: str) -> Optional[Deque[Any]]:
        """
        Return the underlying deque for a table (no copy).
        Useful for fast iteration or tail access (e.g., rows[-1]).
        """
        return self._tables.get(name)


    def clear(self):
        """Clear all tables."""
        self._tables.clear()
