from typing import Any, Dict, List


class Database:
    """
    Each "table" is a dict. Table names must be unique.
    """

    def __init__(self):
        self._tables: Dict[str, List[Any]] = {}

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

    def all_tables(self) -> Dict[str, List[Any]]:
        """Return a dict of all tables."""
        return self._tables

    def clear(self):
        """Clear all tables."""
        self._tables.clear()
