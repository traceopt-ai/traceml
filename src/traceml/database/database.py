from collections import deque
from typing import Any, Deque, Dict, Optional

from .database_writer import DatabaseWriter


class Database:
    """
    Lightweight in-memory database for sampler-side telemetry.

    This class acts as a bounded, append-only storage layer for time-series
    or step-wise telemetry emitted by samplers (e.g., memory usage, timing,
    queue depths).

    Design goals
    ------------
    - Bounded memory usage per table
    - O(1) append and O(1) eviction of old records
    - Minimal overhead suitable for long-running training jobs

    Implementation notes
    --------------------
    - Each "table" is backed by a `collections.deque` with a fixed `maxlen`.
      Once the limit is reached, the oldest records are automatically dropped.
    - Tables are created lazily on first use.
    - This database is intentionally simple and in-memory only; persistence
      and export are handled by `DatabaseWriter`.
    """

    DEFAULT_MAX_ROWS = 3000

    def __init__(self, sampler_name, max_rows: Optional[int] = None):
        """
        Initialize the in-memory database.

        Parameters
        ----------
        sampler_name : str
            Name of the sampler owning this database. Used for labeling
            and downstream writers/exporters.
        max_rows : Optional[int]
            Maximum number of rows per table. If None, DEFAULT_MAX_ROWS is used.

        Raises
        ------
        ValueError
            If `max_rows` is provided and is <= 0.
        """
        self.sampler_name = sampler_name

        # Resolve max row limit
        self.max_rows: int = (
            int(max_rows) if max_rows is not None else self.DEFAULT_MAX_ROWS
        )
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}")

        # Internal storage:
        #   table_name -> deque(records)
        self._tables: Dict[str, Deque[Any]] = {}

        # Writer handles persistence / export of database contents
        self.writer = DatabaseWriter(self, sampler_name=sampler_name)

    def create_table(self, name: str) -> Deque[Any]:
        """
        Create a new empty table.

        Parameters
        ----------
        name : str
            Name of the table to create.

        Returns
        -------
        Deque[Any]
            The newly created deque backing the table.

        Raises
        ------
        ValueError
            If a table with the same name already exists.
        """
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists.")

        # Use deque with fixed maxlen to enforce bounded memory
        self._tables[name] = deque(maxlen=self.max_rows)
        return self._tables[name]

    def create_or_get_table(self, name: str) -> Deque[Any]:
        """
        Create a table if it does not exist, otherwise return the existing one.

        This is the preferred method for callers that want to append data
        without worrying about initialization order.

        Parameters
        ----------
        name : str
            Name of the table.

        Returns
        -------
        Deque[Any]
            The deque backing the table.
        """
        if name not in self._tables:
            self._tables[name] = deque(maxlen=self.max_rows)
        return self._tables[name]

    def add_record(self, table: str, record: Any) -> None:
        """
        Append a single record to a table.

        If the table does not exist, it is created automatically.
        When the table exceeds `max_rows`, the oldest records are
        evicted automatically by the deque.

        Parameters
        ----------
        table : str
            Name of the target table.
        record : Any
            Arbitrary record object to store (e.g., dict, dataclass, tuple).
        """
        rows = self._tables.get(table)
        if rows is None:
            rows = self.create_or_get_table(table)

        # O(1) append; eviction handled automatically by deque(maxlen)
        rows.append(record)

    def get_last_record(self, table: str) -> Optional[Any]:
        """
        Return the most recently added record from a table.

        Parameters
        ----------
        table : str
            Name of the table.

        Returns
        -------
        Optional[Any]
            The most recent record, or None if the table does not exist
            or contains no records.
        """
        rows = self._tables.get(table)
        if not rows:
            return None

        # Access last element without converting to list
        return next(reversed(rows))

    def all_tables(self) -> Dict[str, Deque[Any]]:
        """
        Return all tables in the database.

        Notes
        -----
        - This returns the internal table mapping directly (no copy).
        - Callers should treat the returned deques as read-only unless they
          fully understand the implications.
        - For most iteration use-cases, prefer `get_table()`.

        Returns
        -------
        Dict[str, Deque[Any]]
            Mapping from table name to backing deque.
        """
        return self._tables

    def get_table(self, name: str) -> Optional[Deque[Any]]:
        """
        Retrieve the underlying deque for a specific table.

        This method does not copy data and is suitable for:
        - Fast iteration
        - Accessing recent values (e.g., rows[-1])

        Parameters
        ----------
        name : str
            Table name.

        Returns
        -------
        Optional[Deque[Any]]
            The deque backing the table, or None if the table does not exist.
        """
        return self._tables.get(name)

    def clear(self):
        """
        Remove all tables and records from the database.

        This resets the database to an empty state but does not
        affect the configured `max_rows` or attached writer.
        """
        self._tables.clear()
