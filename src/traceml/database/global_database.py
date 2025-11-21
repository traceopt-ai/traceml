from typing import Any, Dict, List
import threading


class GlobalDatabase:
    """
    Each "table" is a dict. Table names must be unique.
    """

    def __init__(self):
        self._tables: Dict[str, List[Any]] = {}
        self._lock = threading.Lock()

    def create_table(self, name: str) -> List[Any]:
        """
        Create a new empty table if not exists.
        Raise ValueError if table already exists.
        """
        with self._lock:
            if name in self._tables:
                raise ValueError(f"Table '{name}' already exists.")
            self._tables[name] = []
            return self._tables[name]

    def create_or_get_table(self, name: str) -> List[Any]:
        """
        Create table if missing, otherwise return existing table.
        """
        with self._lock:
            if name not in self._tables:
                self._tables[name] = []
            return self._tables[name]

    def all_tables(self) -> Dict[str, List[Any]]:
        """Return a dict of all tables."""
        return self._tables

    def clear(self):
        """Clear all tables."""
        with self._lock:
            self._tables.clear()
