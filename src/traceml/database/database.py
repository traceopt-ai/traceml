from typing import Any, Dict, List
from traceml.database.database_writer import DatabaseWriter


class Database:
    """
    Each "table" is a dict. Table names must be unique.
    """

    def __init__(self, sampler_name):
        self._tables: Dict[str, List[Any]] = {}
        self.writer = DatabaseWriter(self, sampler_name=sampler_name)

    def create_table(self, name: str) -> List[Any]:
        """
        Create a new empty table if not exists.
        Raise ValueError if table already exists.
        """
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists.")
        self._tables[name] = []
        return self._tables[name]

    def create_or_get_table(self, name: str) -> List[Any]:
        """
        Create table if missing, otherwise return existing table.
        """
        if name not in self._tables:
            self._tables[name] = []
        return self._tables[name]

    def add_record(self, table: str, record: Any):
        """
        Add a single record to a table.
        Automatically creates table if it doesn't exist.
        """
        if table not in self._tables:
            raise ValueError(f"Table '{table}' does not exist.")
        self._tables[table].append(record)

    def get_record_at_index(self, table: str, index: int) -> Any:
        """
        Return the record at a given index from a table.
        Returns None if table does not exist or index is out of range.
        """
        if table not in self._tables:
            return None

        rows = self._tables[table]

        # Allow negative indexing like Python lists
        if -len(rows) <= index < len(rows):
            return rows[index]

        return None

    def all_tables(self) -> Dict[str, List[Any]]:
        """Return a dict of all tables."""
        return self._tables

    def clear(self):
        """Clear all tables."""
        self._tables.clear()
